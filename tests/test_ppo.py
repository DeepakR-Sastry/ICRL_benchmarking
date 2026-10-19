import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from torch.distributions import Normal

from icrl_bench.agents.ppo import PPO, ActorCritic, PPOSettings, collect_rollout, compute_gae, ppo_update
from icrl_bench.core.cost_wrapper import CostWrapper
from icrl_bench.envs.lap_grid_world import LapGridWorld, true_cost


def test_discrete_actions_start_close_to_uniform():
    net = ActorCritic(spaces.Box(-1.0, 1.0, shape=(1,)), spaces.Discrete(2))
    obs = torch.tensor([[-1.0], [0.0], [0.5]])  # 3 observations
    probs = net.distribution(obs).probs
    assert probs.shape == (3, 2)
    assert torch.allclose(probs, torch.full((3, 2), 0.5), atol=0.02)
    assert net.value(obs).shape == (3,)


def test_continuous_log_prob_sums_over_action_dimensions():
    net = ActorCritic(spaces.Box(-1.0, 1.0, shape=(4,)), spaces.Box(-1.0, 1.0, shape=(2,)))
    obs = torch.zeros(3, 4)
    actions = torch.tensor([[0.1, -0.2], [0.0, 0.0], [1.0, 0.5]])
    dist = net.distribution(obs)
    # log std starts at 0 (std 1); with a near-zero mean, each dimension is ~N(0, 1)
    per_dimension = Normal(dist.mean, torch.ones(2)).log_prob(actions)
    assert dist.log_prob(actions).shape == (3,)
    assert torch.allclose(dist.log_prob(actions), per_dimension.sum(-1))
    assert torch.allclose(dist.mean, torch.zeros(3, 2), atol=0.05)


def test_orthogonal_init_gains_and_zero_biases():
    net = ActorCritic(spaces.Box(-1.0, 1.0, shape=(4,)), spaces.Discrete(2))
    hidden = net.policy_body[3].weight  # 64x64: orthogonal * sqrt(2)
    assert torch.allclose(hidden @ hidden.T, 2 * torch.eye(64), atol=1e-4)
    head = net.action_head.weight  # 2x64: rows orthonormal * 0.01
    assert torch.allclose(head @ head.T, 0.01**2 * torch.eye(2), atol=1e-8)
    for layer in [net.policy_body[1], net.value_body[1], net.value_head, net.action_head]:
        assert torch.count_nonzero(layer.bias) == 0
    assert np.isclose(net.value_head.weight.norm().item(), 1.0, atol=1e-5)


def test_gae_on_a_3_step_rollout_with_an_episode_boundary():
    # One env. Step 1 ended an episode, so step 2 starts a new one.
    rewards = np.array([[1.0], [0.0], [2.0]], dtype=np.float32)
    values = np.array([[0.5], [1.0], [0.0]], dtype=np.float32)
    episode_starts = np.array([[1.0], [0.0], [1.0]], dtype=np.float32)
    advantages, returns = compute_gae(
        rewards, values, episode_starts,
        last_value=np.array([3.0]), last_done=np.array([0.0]),
        gamma=0.9, gae_lambda=0.8,
    )
    np.testing.assert_allclose(advantages[:, 0], [0.68, -1.0, 4.7], rtol=1e-6)
    np.testing.assert_allclose(returns[:, 0], [1.18, 0.0, 4.7], rtol=1e-6)


def test_gae_ignores_last_value_when_the_rollout_ends_an_episode():
    rewards = np.array([[1.0, 1.0]], dtype=np.float32)  # 1 step, 2 envs
    values = np.zeros((1, 2), dtype=np.float32)
    advantages, _ = compute_gae(
        rewards, values, np.zeros((1, 2), dtype=np.float32),
        last_value=np.array([5.0, 5.0]), last_done=np.array([1.0, 0.0]),
        gamma=0.9, gae_lambda=0.8,
    )
    np.testing.assert_allclose(advantages[0], [1.0, 1.0 + 0.9 * 5.0], rtol=1e-6)


def lgw_envs(n_envs):
    envs = SyncVectorEnv([lambda: CostWrapper(LapGridWorld(), true_cost) for _ in range(n_envs)],
                         autoreset_mode=AutoresetMode.SAME_STEP)
    obs, _ = envs.reset(seed=0)
    return envs, obs


def test_rollout_shapes_and_episode_boundary():
    # LapGridWorld episodes are 200 steps, so step 200 starts a new episode.
    envs, obs = lgw_envs(2)
    net = ActorCritic(envs.single_observation_space, envs.single_action_space)
    rollout, _, _ = collect_rollout(net, envs, obs, np.zeros(2, np.float32), 250, 0.99, 0.95)
    assert rollout["observations"].shape == (250, 2, 1)
    assert rollout["actions"].shape == (250, 2)
    for key in ["rewards", "episode_starts", "values", "log_probs", "advantages", "returns"]:
        assert rollout[key].shape == (250, 2)
    starts = rollout["episode_starts"]
    assert starts[200].tolist() == [1.0, 1.0]
    assert starts.sum() == 2  # nowhere else (step 0 is 0, as in SB3)


def test_next_rollout_continues_where_the_last_stopped():
    envs, obs = lgw_envs(2)
    net = ActorCritic(envs.single_observation_space, envs.single_action_space)
    first, obs, starts = collect_rollout(net, envs, obs, np.zeros(2, np.float32), 10, 0.99, 0.95)
    second, _, _ = collect_rollout(net, envs, obs, starts, 10, 0.99, 0.95)
    np.testing.assert_array_equal(second["observations"][0], obs)
    assert not np.array_equal(second["observations"][0], first["observations"][0])


def test_last_step_bootstraps_with_its_own_value_as_in_malik():
    envs, obs = lgw_envs(2)
    net = ActorCritic(envs.single_observation_space, envs.single_action_space)
    r, _, last_done = collect_rollout(net, envs, obs, np.zeros(2, np.float32), 10, 0.99, 0.95)
    v = r["values"][-1]  # V(last stored obs), not V(next obs)
    expected = r["rewards"][-1] + 0.99 * v * (1 - last_done) - v
    np.testing.assert_allclose(r["advantages"][-1], expected, rtol=1e-6)


def test_continuous_rollout_stores_unclipped_actions():
    envs = SyncVectorEnv([lambda: gym.make("Pendulum-v1") for _ in range(2)],
                         autoreset_mode=AutoresetMode.SAME_STEP)  # actions in [-2, 2]
    obs, _ = envs.reset(seed=0)
    net = ActorCritic(envs.single_observation_space, envs.single_action_space)
    with torch.no_grad():
        net.log_std.fill_(3.0)  # std ≈ 20, so most samples fall outside [-2, 2]
    rollout, _, _ = collect_rollout(net, envs, obs, np.zeros(2, np.float32), 20, 0.99, 0.95)
    assert rollout["actions"].shape == (20, 2, 1)
    assert np.abs(rollout["actions"]).max() > 2.0


def bandit_rollout(net, n_steps=64):
    """One env, one observation; action 0 was good (A = +1), action 1 bad (A = -1)."""
    obs = np.zeros((n_steps, 1, 1), dtype=np.float32)
    actions = np.array([[0], [1]] * (n_steps // 2))
    with torch.no_grad():
        log_probs = net.distribution(torch.as_tensor(obs[:, 0])).log_prob(torch.as_tensor(actions[:, 0]))
    return {
        "observations": obs,
        "actions": actions,
        "rewards": np.zeros((n_steps, 1), np.float32),
        "log_probs": log_probs.numpy()[:, None],
        "advantages": np.where(actions == 0, 1.0, -1.0).astype(np.float32),
        "returns": np.zeros((n_steps, 1), np.float32),
    }


def prob_of_action_0(net):
    with torch.no_grad():
        return net.distribution(torch.zeros(1, 1)).probs[0, 0].item()


def update(net, rollout, optimizer=None, n_epochs=10, batch_size=16, target_kl=None):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)
    return ppo_update(net, optimizer, rollout, n_epochs=n_epochs, batch_size=batch_size, clip_range=0.2,
                      ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, target_kl=target_kl)


def test_update_makes_the_good_action_more_likely():
    net = ActorCritic(spaces.Box(-1.0, 1.0, shape=(1,)), spaces.Discrete(2))
    before = prob_of_action_0(net)
    stats = update(net, bandit_rollout(net))
    assert prob_of_action_0(net) > before
    assert stats["epochs"] == 10


def test_clipping_caps_how_far_one_rollout_moves_the_policy():
    # Without clipping, 200 epochs would push action 0 to ~1. With clip 0.2 the
    # gradient is zero once its ratio passes 1.2: 0.5 * 1.2 = 0.6, plus one
    # step of overshoot. Plain SGD: Adam's momentum would keep moving past it.
    net = ActorCritic(spaces.Box(-1.0, 1.0, shape=(1,)), spaces.Discrete(2))
    rollout = bandit_rollout(net)
    update(net, rollout, optimizer=torch.optim.SGD(net.parameters(), lr=0.2),
           n_epochs=200, batch_size=64)
    assert 0.6 < prob_of_action_0(net) < 0.62


def test_kl_early_stop_lags_one_minibatch():
    # Full batch (32 of each action; with unbalanced minibatches Malik's KL
    # estimate mean(old - new) can come out negative). KL is measured with
    # log-probs from before each minibatch's step, so with one minibatch per
    # epoch, epoch 1 always measures 0 and the stop comes after epoch 2.
    net = ActorCritic(spaces.Box(-1.0, 1.0, shape=(1,)), spaces.Discrete(2))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, eps=1e-5)
    stats = update(net, bandit_rollout(net), optimizer=optimizer, batch_size=64, target_kl=1e-6)
    assert stats["epochs"] == 2


def recorded_lgw_envs(n_envs):
    return SyncVectorEnv(
        [lambda: CostWrapper(RecordEpisodeStatistics(LapGridWorld()), true_cost) for _ in range(n_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP)


def test_train_rounds_up_to_whole_rollouts_and_logs_episode_returns():
    # 2 copies x 256 steps = 512 per rollout; 1000 steps needs 2 rollouts.
    # LapGridWorld episodes are 200 steps, so each copy finishes 1 per rollout
    # (the 2nd rollout's episode ends at step 400 of 512).
    agent = PPO(spaces.Box(-1.0, 1.0, shape=(1,)), spaces.Discrete(2), PPOSettings(n_steps=256))
    log = agent.train(recorded_lgw_envs(2), total_steps=1000)
    assert [row["steps"] for row in log] == [512, 1024]
    assert all(np.isfinite(row["ep_rew_mean"]) for row in log)
    assert {"policy_loss", "value_loss", "approx_kl", "clip_fraction", "epochs"} <= log[0].keys()


def test_same_seed_same_training():
    logs = []
    for _ in range(2):
        # 256 steps per copy: each finishes one 200-step episode (no NaN mean).
        agent = PPO(spaces.Box(-1.0, 1.0, shape=(1,)), spaces.Discrete(2), PPOSettings(n_steps=256))
        logs.append(agent.train(recorded_lgw_envs(2), total_steps=1024))
    assert logs[0] == logs[1]


def test_policy_returns_one_env_action():
    discrete = PPO(spaces.Box(-1.0, 1.0, shape=(1,)), spaces.Discrete(2))
    assert discrete.policy(np.array([0.5], dtype=np.float32)) in (0, 1)
    continuous = PPO(spaces.Box(-1.0, 1.0, shape=(3,)), spaces.Box(-1.0, 1.0, shape=(2,)))
    with torch.no_grad():
        continuous.net.log_std.fill_(3.0)  # std ≈ 20: samples almost always out of bounds
    action = continuous.policy(np.zeros(3, dtype=np.float32))
    assert action.shape == (2,)
    assert np.all(np.abs(action) <= 1.0)
