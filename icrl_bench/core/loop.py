"""The ICRL loop: forward step (agent trains on the learned cost), sample
nominal trajectories, backward step (method updates the constraint).

Follows Malik et al.'s icrl/icrl.py (github.com/shehryar-malik/icrl, commit
f28214e), cited where each piece is used. Order and design: docs/design.md."""

from dataclasses import dataclass

from icrl_bench.core.rollout import rollout
from icrl_bench.core.types import Step, Trajectory
from icrl_bench.eval.stage2 import reward_stats, true_cost_rate


@dataclass
class ICRLLoopSettings:
    """Defaults: Malik's CLI defaults (icrl/icrl.py argparse). Per-env values
    live in configs."""
    n_rounds: int = 100  # -ni
    steps_per_round: int = 1_000_000  # -ft, agent steps per forward step
    n_eval_episodes: int = 10  # hard-coded in icrl.py L248
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
             true_cost, settings=None):
    """Malik's ICRL loop (icrl.py L199-304). Each round: the agent trains on
    the learned cost, nominal trajectories are sampled from it, the method
    updates the constraint, and the agent is graded with the true cost.

    agent: has train(envs, steps) and policy(obs). envs: the agent's vector
    env, a CostWrapper on every copy. expert: list[Trajectory].
    make_nominal_env / make_eval_env: build a fresh unconstrained / constrained
    env. true_cost: used for grading only. Returns one metrics dict per round."""
    s = settings or ICRLLoopSettings()
    # Round 0's forward step trains on the untrained constraint (icrl.py L116).
    envs.call("set_cost_fn", learned_cost_fn(featurize, method))
    timesteps = 0
    log = []
    for round_ in range(s.n_rounds):
        # Forward step (L206-213).
        forward = agent.train(envs, s.steps_per_round)
        timesteps += forward[-1]["steps"]

        # As many nominal trajectories as expert ones, sampled actions, no
        # cost wrapper (L216-218).
        nominal = rollout(make_nominal_env(), agent.policy, len(expert), s.seed)

        # Backward step, then hand the new constraint to every env copy
        # (L235-239). Both piles are featurized every round, so they always
        # use the same featurizer state (docs/decisions.md, 2026-10-19).
        backward = method.update([featurize(t) for t in expert], [featurize(t) for t in nominal])
        envs.call("set_cost_fn", learned_cost_fn(featurize, method))

        # Grading with the true cost (L243-249).
        evaluation = rollout(make_eval_env(), agent.policy, s.n_eval_episodes, s.seed)
        reward_mean, reward_std = reward_stats(evaluation)
        log.append({
            "round": round_,
            "timesteps": timesteps,
            "true/reward": reward_mean,
            "true/reward_std": reward_std,
            "true/cost": true_cost_rate(nominal, true_cost),
            **{f"forward/{k}": v for k, v in forward[-1].items()},
            **{f"backward/{k}": v for k, v in backward.items()},
        })
    return log
