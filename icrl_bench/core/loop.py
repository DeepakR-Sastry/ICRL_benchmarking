"""The ICRL loop: forward step (agent trains on the learned cost), sample
nominal trajectories, backward step (method updates the constraint).

Follows Malik et al.'s icrl/icrl.py (github.com/shehryar-malik/icrl, commit
f28214e), cited where each piece is used. Order and design: docs/design.md.

GACL has its own loop, run_gacl (Malik's icrl/gail.py): one training run in
which the method is updated after every rollout (docs/decisions.md,
2026-11-01)."""

from dataclasses import dataclass

import numpy as np

from icrl_bench.core.rollout import rollout
from icrl_bench.core.types import Step, Trajectory
from icrl_bench.eval.stage2 import feasible_reward_stats, reward_stats, true_cost_rate


@dataclass
class ICRLLoopSettings:
    """Defaults: Malik's CLI defaults (icrl/icrl.py argparse). Per-env values
    live in configs."""
    n_rounds: int = 100  # -ni
    steps_per_round: int = 1_000_000  # -ft, agent steps per forward step
    n_eval_episodes: int = 10  # hard-coded in icrl.py L248
    # Liu's use_buffer (train_icrl.py L74-77, L385-391; IRLDataQueue): keep
    # every nominal trajectory sampled so far and train the method on
    # len(expert) of them drawn uniformly with replacement. Off = only this
    # round's, as Malik (docs/decisions.md, 2026-11-01).
    nominal_buffer: bool = False
    seed: int = 0


def learned_cost_fn(featurize, method):
    """A cost_fn(obs, action) for CostWrapper: the step is featurized as a
    one-step trajectory and scored by method.cost (Malik passes
    constraint_net.cost_function to his cost wrapper, icrl.py L116, L239).

    It keeps a reference to `method`, so it always uses the current
    constraint. Featurizers never read the reward, so it is set to 0."""
    def cost_fn(obs, action):
        features = featurize(Trajectory([Step(obs, action, reward=0.0)]))
        return float(method.cost(features)[0])
    return cost_fn


def run_icrl(agent, envs, method, featurize, expert, make_nominal_env, make_eval_env,
             true_cost, settings=None, callback=None):
    """Malik's ICRL loop (icrl.py L199-304). Each round: the agent trains on
    the learned cost, nominal trajectories are sampled from it, the method
    updates the constraint, and the agent is graded with the true cost.

    agent: has train(envs, steps) and policy(obs). envs: the agent's vector
    env, a CostWrapper on every copy. expert: list[Trajectory].
    make_nominal_env / make_eval_env: build an unconstrained / constrained
    env; each is built once, as Malik's sampling and eval envs (L64, L68).
    true_cost: used for grading only. Returns one metrics dict per round;
    callback, if given, is also called with each dict as soon as its round
    is done (as PPOLagrangian.train's), so long runs can save as they go."""
    s = settings or ICRLLoopSettings()
    # Round 0's forward step trains on the untrained constraint (icrl.py L116).
    envs.call("set_cost_fn", learned_cost_fn(featurize, method))
    nominal_env, eval_env = make_nominal_env(), make_eval_env()
    store = []  # every nominal trajectory so far (nominal_buffer only)
    rng = np.random.default_rng(s.seed)  # its own stream: changes nothing else
    timesteps = 0
    log = []
    for round_ in range(s.n_rounds):
        # Both envs are seeded once, in round 0; later rounds continue their
        # random streams, so each round starts from new poses (Malik's envs
        # are never reseeded either).
        seed = s.seed if round_ == 0 else None
        # Forward step (L206-213).
        forward = agent.train(envs, s.steps_per_round)
        timesteps += forward[-1]["steps"]

        # As many nominal trajectories as expert ones, sampled actions, no
        # cost wrapper (L216-218).
        nominal = rollout(nominal_env, agent.policy, len(expert), seed)

        # With the buffer, the method's pile is drawn from every round so
        # far, this one included; repeats allowed (Liu's IRLDataQueue.get).
        # Liu caps the store at 500 trajectories; his 30 rounds x 10 never
        # reach it, so there is no cap here.
        pile = nominal
        if s.nominal_buffer:
            store += nominal
            pile = [store[i] for i in rng.integers(len(store), size=len(expert))]

        # Backward step, then hand the new constraint to every env copy
        # (L235-239). Both piles are featurized every round, so they always
        # use the same featurizer state (docs/decisions.md, 2026-10-19).
        backward = method.update([featurize(t) for t in expert], [featurize(t) for t in pile])
        envs.call("set_cost_fn", learned_cost_fn(featurize, method))

        # Grading with the true cost (L243-249).
        evaluation = rollout(eval_env, agent.policy, s.n_eval_episodes, seed)
        reward_mean, reward_std = reward_stats(evaluation)
        # Liu's evaluation: reward before the first violation, and how many
        # episodes broke the rule at all (docs/decisions.md, 2026-11-01).
        feasible_mean, feasible_std, violating = feasible_reward_stats(evaluation, true_cost)
        log.append({
            "round": round_,
            "timesteps": timesteps,
            "true/reward": reward_mean,
            "true/reward_std": reward_std,
            "true/feasible_reward": feasible_mean,
            "true/feasible_reward_std": feasible_std,
            "true/violating_episodes": violating,
            "true/cost": true_cost_rate(nominal, true_cost),
            **{f"forward/{k}": v for k, v in forward[-1].items()},
            **{f"backward/{k}": v for k, v in backward.items()},
        })
        if callback is not None:
            callback(log[-1])
    return log


@dataclass
class GACLLoopSettings:
    """Defaults: Malik's CLI defaults (icrl/gail.py argparse, SB3
    EvalCallback). Per-env values live in configs."""
    total_steps: int = 1_000_000  # -t, agent steps in the one training run
    eval_every: int = 2048  # -ee, in vector-env steps (all copies step together)
    n_eval_episodes: int = 5  # EvalCallback default
    seed: int = 0


def run_gacl(agent, envs, method, featurize, expert, make_eval_env, true_cost,
             settings=None, callback=None):
    """Malik's GACL (gail.py L189-191 + GailCallback): one agent.train run;
    after each rollout the method is updated on that rollout and its log D
    is added to the rollout's rewards (agent.train's reward_hook).

    agent: has train(envs, steps, callback, reward_hook) and policy(obs).
    envs: the agent's vector env; its cost is set to zero and never read. expert:
    list[Trajectory]. make_eval_env builds the constrained env, once.
    true_cost: used for grading only. Returns one metrics dict per rollout;
    callback, if given, is also called with each dict as soon as it is
    complete."""
    s = settings or GACLLoopSettings()
    # Malik's GACL envs have no cost wrapper; ours carry a zero cost, which
    # PPO never reads (each loop sets its copies' cost first, as run_icrl).
    envs.call("set_cost_fn", lambda obs, action: 0.0)
    n_envs = envs.num_envs
    eval_env = make_eval_env()
    pending = {}  # the hook's metrics, waiting for the PPO update's row
    log = []
    n_evaluations = 0

    def hook(observations, actions):
        # GailCallback._on_rollout_end. Step-major order, as his flatten:
        # (step 0, env 0), (step 0, env 1), ...; quirk A keeps the first rows.
        n_steps = actions.shape[0]
        steps = [Step(observations[t, e], actions[t, e], reward=0.0)
                 for t in range(n_steps) for e in range(n_envs)]
        nominal = Trajectory(steps)
        features = featurize(nominal)
        # Both piles featurized each time, so they always use the same
        # featurizer state (as run_icrl, docs/decisions.md 2026-10-19).
        backward = method.update([featurize(t) for t in expert], [features])
        pending.clear()
        pending.update({f"backward/{k}": v for k, v in backward.items()})
        # His eval/mean_cost: true cost per step of this training rollout.
        pending["true/cost"] = true_cost_rate([nominal], true_cost)
        # Rewards from the updated D (his reward_function after train).
        return method.reward(features).reshape(n_steps, n_envs)

    def on_row(row):
        # His EvalCallback evaluates when its call count (one per vector
        # step) hits a multiple of eval_every, i.e. during a rollout, with
        # the policy of the last update. So after this update, evaluate if
        # the next rollout would contain such a step and training goes on.
        nonlocal n_evaluations
        done = row["steps"] // n_envs  # vector steps so far
        next_end = done + done // (len(log) + 1)  # + one rollout's vector steps
        reward_mean = reward_std = feasible_mean = feasible_std = violating = float("nan")
        if row["steps"] < s.total_steps and next_end // s.eval_every > done // s.eval_every:
            # The eval env is seeded at its first use only, as in run_icrl.
            seed = s.seed if n_evaluations == 0 else None
            evaluation = rollout(eval_env, agent.policy, s.n_eval_episodes, seed)
            reward_mean, reward_std = reward_stats(evaluation)
            feasible_mean, feasible_std, violating = feasible_reward_stats(evaluation, true_cost)
            n_evaluations += 1
        log.append({
            "rollout": len(log),
            "timesteps": row["steps"],
            "true/reward": reward_mean,
            "true/reward_std": reward_std,
            "true/feasible_reward": feasible_mean,
            "true/feasible_reward_std": feasible_std,
            "true/violating_episodes": violating,
            **pending,
            **{f"forward/{k}": v for k, v in row.items()},
        })
        if callback is not None:
            callback(log[-1])

    agent.train(envs, s.total_steps, callback=on_row, reward_hook=hook)
    return log
