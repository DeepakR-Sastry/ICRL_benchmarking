"""PPO: maximises reward only.

Follows Malik et al.'s fork of stable-baselines3 0.9.0 (github.com/
shehryar-malik/icrl, commit f28214e): stable_baselines3/ppo/ppo.py and the
SB3 files it relies on, cited where each piece is used."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Independent, Normal


class ActorCritic(nn.Module):
    """Policy and value networks, no shared layers.

    SB3 common/policies.py ActorCriticPolicy (default net_arch
    [dict(pi=[64, 64], vf=[64, 64])], Tanh, orthogonal init) and
    common/distributions.py (action heads). Observations must be a Box;
    they are flattened and cast to float (common/preprocessing.py)."""

    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))

        self.policy_body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.value_body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.value_head = nn.Linear(64, 1)

        self.discrete = isinstance(action_space, spaces.Discrete)
        if self.discrete:
            self.action_head = nn.Linear(64, int(action_space.n))  # logits
        else:
            action_dim = int(np.prod(action_space.shape))
            self.action_head = nn.Linear(64, action_dim)  # Gaussian mean
            self.log_std = nn.Parameter(torch.zeros(action_dim))  # not state-dependent

        # Orthogonal weights, zero biases; small gain on the action head so the
        # first policy is close to uniform (SB3 policies.py ActorCriticPolicy._build).
        for module, gain in [
            (self.policy_body, np.sqrt(2)),
            (self.value_body, np.sqrt(2)),
            (self.action_head, 0.01),
            (self.value_head, 1.0),
        ]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=gain)
                    nn.init.zeros_(layer.bias)

    def distribution(self, obs):
        """The action distribution for a batch of observations."""
        out = self.action_head(self.policy_body(obs.float()))
        if self.discrete:
            return Categorical(logits=out)
        # Independent(..., 1) treats the action dimensions as one joint action:
        # log_prob and entropy are summed over them, as in SB3's DiagGaussian.
        return Independent(Normal(out, self.log_std.exp()), 1)

    def value(self, obs):
        """Predicted future reward, one number per observation."""
        return self.value_head(self.value_body(obs.float())).squeeze(-1)


def compute_gae(rewards, values, episode_starts, last_value, last_done, gamma, gae_lambda):
    """GAE advantages and returns for one rollout (SB3 common/buffers.py
    RolloutBuffer.compute_returns_and_advantage).

    rewards, values, episode_starts: (n_steps, n_envs). episode_starts[t] = 1
    if the observation at step t is the first of an episode (SB3 0.9.0 calls
    this `dones`). last_value, last_done: (n_envs,), for the state after the
    last step. Returns (advantages, returns), both (n_steps, n_envs)."""
    n_steps = len(rewards)
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0.0
    for step in reversed(range(n_steps)):
        if step == n_steps - 1:
            next_non_terminal = 1.0 - last_done
            next_value = last_value
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1]
            next_value = values[step + 1]
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[step] = last_gae_lam
    returns = advantages + values
    return advantages, returns


def collect_rollout(net, envs, obs, episode_starts, n_steps, gamma, gae_lambda):
    """Run the policy for n_steps in every env copy and compute GAE (SB3
    common/on_policy_algorithm.py collect_rollouts).

    envs: gymnasium vector env with autoreset_mode=SAME_STEP. obs and
    episode_starts are where the previous rollout stopped (envs are not reset
    between rollouts). Returns (rollout, obs, episode_starts): rollout is a
    dict of arrays, each (n_steps, n_envs, ...), plus "episode_returns": the
    total reward of each episode that finished during the rollout (needs each
    env copy wrapped in gymnasium's RecordEpisodeStatistics, SB3's Monitor;
    empty otherwise). The other two are where the next rollout continues."""
    device = next(net.parameters()).device
    action_space = envs.single_action_space
    rollout = {key: [] for key in
               ["observations", "actions", "rewards", "episode_starts", "values", "log_probs"]}
    episode_returns = []

    for _ in range(n_steps):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=device)
            dist = net.distribution(obs_tensor)
            actions = dist.sample()
            values = net.value(obs_tensor)
            log_probs = dist.log_prob(actions)
        actions = actions.cpu().numpy()

        # The env gets clipped actions; the rollout keeps the sampled ones.
        env_actions = actions
        if isinstance(action_space, spaces.Box):
            env_actions = np.clip(actions, action_space.low, action_space.high)
        new_obs, rewards, terminated, truncated, infos = envs.step(env_actions)
        # With SAME_STEP resets, a finished episode's info is under "final_info".
        final = infos.get("final_info", {})
        if "episode" in final:
            episode_returns.extend(final["episode"]["r"][final["_episode"]].tolist())

        rollout["observations"].append(obs)
        rollout["actions"].append(actions)
        rollout["rewards"].append(rewards)
        rollout["episode_starts"].append(episode_starts)
        rollout["values"].append(values.cpu().numpy())
        rollout["log_probs"].append(log_probs.cpu().numpy())

        obs = new_obs
        # A time-limit cutoff counts as a true end, as in SB3 0.9.0.
        episode_starts = np.logical_or(terminated, truncated).astype(np.float32)

    rollout = {key: np.stack(arrays) for key, arrays in rollout.items()}
    for key in ["rewards", "episode_starts", "values", "log_probs"]:
        rollout[key] = rollout[key].astype(np.float32)

    # Malik's SB3 0.9.0 bootstraps the last step with the leftover `values` =
    # V(last stored obs), not V(obs). See "Known differences" in docs/agents.md.
    rollout["advantages"], rollout["returns"] = compute_gae(
        rollout["rewards"], rollout["values"], rollout["episode_starts"],
        last_value=values.cpu().numpy(), last_done=episode_starts,
        gamma=gamma, gae_lambda=gae_lambda,
    )
    rollout["episode_returns"] = np.array(episode_returns, dtype=np.float32)
    return rollout, obs, episode_starts


def ppo_update(net, optimizer, rollout, n_epochs, batch_size, clip_range,
               ent_coef, vf_coef, max_grad_norm, target_kl):
    """Improve the policy and value networks on one rollout (SB3 0.9.0
    ppo/ppo.py PPO.train). Returns mean losses and stats for logging.

    target_kl=None disables the KL early stop. Not ported (unused in Malik's
    runs): value-function clipping, learning-rate and clip-range schedules."""
    device = next(net.parameters()).device
    n = rollout["rewards"].size  # n_steps * n_envs
    # Flatten (n_steps, n_envs, ...) to (n, ...): minibatches mix steps and envs.
    data = {key: torch.as_tensor(rollout[key].reshape(n, *rollout[key].shape[2:]), device=device)
            for key in ["observations", "actions", "log_probs", "advantages", "returns"]}

    stats = {key: [] for key in
             ["policy_loss", "value_loss", "entropy_loss", "approx_kl", "clip_fraction"]}
    epochs_run = 0
    for _ in range(n_epochs):
        epoch_kls = []
        indices = torch.as_tensor(np.random.permutation(n), device=device)
        for start in range(0, n, batch_size):
            batch = {key: value[indices[start:start + batch_size]] for key, value in data.items()}

            dist = net.distribution(batch["observations"])
            log_prob = dist.log_prob(batch["actions"])
            values = net.value(batch["observations"])

            # Per-minibatch normalisation.
            advantages = batch["advantages"]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Clipped surrogate loss: the ratio can't push the policy more
            # than clip_range away from the policy that collected the rollout.
            ratio = torch.exp(log_prob - batch["log_probs"])
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            value_loss = F.mse_loss(batch["returns"], values)
            entropy_loss = -dist.entropy().mean()
            loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            # Measured with log_prob from before this minibatch's step, as in SB3.
            approx_kl = torch.mean(batch["log_probs"] - log_prob).item()
            epoch_kls.append(approx_kl)
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy_loss"].append(entropy_loss.item())
            stats["approx_kl"].append(approx_kl)
            stats["clip_fraction"].append(torch.mean((torch.abs(ratio - 1) > clip_range).float()).item())

        epochs_run += 1
        # Checked once per epoch, on the epoch's mean.
        if target_kl is not None and np.mean(epoch_kls) > 1.5 * target_kl:
            break

    stats = {key: float(np.mean(values)) for key, values in stats.items()}
    stats["epochs"] = epochs_run
    return stats


@dataclass
class PPOSettings:
    """Defaults: Malik's CLI defaults (icrl/gail.py, icrl/icrl.py argparse).
    n_envs is not a setting: it is the number of copies in the vector env."""
    learning_rate: float = 3e-4
    n_steps: int = 2048  # per env copy, per rollout
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    seed: int = 0
    device: str = "cpu"


class PPO:
    """PPO with a policy and a value network (SB3 0.9.0 ppo/ppo.py PPO).
    Maximises reward; ignores info["cost"]."""

    def __init__(self, observation_space, action_space, settings=None):
        self.settings = settings or PPOSettings()
        self.action_space = action_space
        # Seeded before the network is built, so its initial weights depend
        # only on the seed (SB3 common/utils.py set_random_seed).
        np.random.seed(self.settings.seed)
        torch.manual_seed(self.settings.seed)
        self.net = ActorCritic(observation_space, action_space).to(self.settings.device)
        # One Adam over both networks; eps=1e-5 (SB3 ActorCriticPolicy).
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.settings.learning_rate, eps=1e-5)
        self.envs_seeded = False

    def train(self, envs, total_steps):
        """Alternate collecting a rollout and updating, until at least
        total_steps env steps (all copies together) have been taken (SB3
        OnPolicyAlgorithm.learn). envs: gymnasium vector env,
        autoreset_mode=SAME_STEP, each copy wrapped in RecordEpisodeStatistics.

        Resets envs on every call, as Malik's ICRL loop does (learn's default
        reset_num_timesteps=True); the seed is used on the first call only.
        Returns one dict per rollout for logging."""
        s = self.settings
        obs, _ = envs.reset(seed=None if self.envs_seeded else s.seed)
        self.envs_seeded = True
        episode_starts = np.zeros(envs.num_envs, dtype=np.float32)
        recent_returns = deque(maxlen=100)  # SB3's ep_info_buffer
        log = []
        steps = 0
        while steps < total_steps:
            rollout, obs, episode_starts = collect_rollout(
                self.net, envs, obs, episode_starts, s.n_steps, s.gamma, s.gae_lambda)
            steps += s.n_steps * envs.num_envs
            stats = ppo_update(
                self.net, self.optimizer, rollout, s.n_epochs, s.batch_size, s.clip_range,
                s.ent_coef, s.vf_coef, s.max_grad_norm, s.target_kl)
            recent_returns.extend(rollout["episode_returns"].tolist())
            ep_rew_mean = float(np.mean(recent_returns)) if recent_returns else float("nan")
            log.append({"steps": steps, "ep_rew_mean": ep_rew_mean, **stats})
        return log

    def policy(self, obs):
        """One observation in, one sampled action out (Malik samples
        trajectories with deterministic=False); continuous actions are clipped
        to the action bounds (SB3 BasePolicy.predict)."""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.settings.device).unsqueeze(0)
            action = self.net.distribution(obs_tensor).sample()[0].cpu().numpy()
        if isinstance(self.action_space, spaces.Box):
            return np.clip(action, self.action_space.low, self.action_space.high)
        return int(action)
