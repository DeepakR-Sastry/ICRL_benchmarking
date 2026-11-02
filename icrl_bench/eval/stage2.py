"""Stage-2 metrics: how safe and how good the agent is, graded with the true
cost. As Malik et al. log them each ICRL round (icrl/icrl.py L241-252,
github.com/shehryar-malik/icrl, commit f28214e), plus Liu et al.'s feasible
reward and per-episode violation flag (github.com/Guiliang/ICRL-benchmarks-public,
commit 0d79d2c; docs/decisions.md, 2026-11-01)."""

import numpy as np


def true_cost_rate(trajectories, true_cost):
    """Average true cost per step over all steps of all trajectories: the
    paper's "average number of constraint violations per timestep"
    (Malik's true/cost, computed on the sampled nominal trajectories)."""
    costs = [true_cost(step.state, step.action) for t in trajectories for step in t.steps]
    return float(np.mean(costs))


def reward_stats(trajectories):
    """Mean and standard deviation (np.std, as SB3's evaluate_policy) of the
    episode rewards. Malik's true/reward and true/reward_std: 10 episodes
    in the constrained env, sampled actions."""
    returns = [sum(step.reward for step in t.steps) for t in trajectories]
    return float(np.mean(returns)), float(np.std(returns))


def liu_first_violation(trajectory, true_cost):
    """Index of the first step Liu et al. count as a violation, or None.
    His MujocoExternalSignalWrapper (utils/env_utils.py, commit 0d79d2c)
    applies the rule to the observation a step *led to*: here the next
    step's state. Our true_cost(obs, action) is otherwise used on the
    observation the action was chosen on, so his first violation can be one
    step earlier. The last step's resulting observation is not stored in a
    Trajectory; that step is judged on its own state (only matters if an
    episode's first violation is its last step)."""
    steps = trajectory.steps
    for t, step in enumerate(steps):
        landed_on = steps[t + 1].state if t + 1 < len(steps) else step.state
        if true_cost(landed_on, step.action) > 0:
            return t
    return None


def feasible_reward_stats(trajectories, true_cost):
    """Liu's per-episode evaluation (common/cns_evaluation.py
    evaluate_icrl_policy): feasible reward = the reward of the steps before
    the first violation (that step excluded), and whether the episode broke
    the rule at least once. Returns (mean and np.std of feasible reward,
    fraction of episodes with a violation)."""
    feasible, violated = [], []
    for trajectory in trajectories:
        first = liu_first_violation(trajectory, true_cost)
        end = len(trajectory.steps) if first is None else first
        feasible.append(sum(step.reward for step in trajectory.steps[:end]))
        violated.append(first is not None)
    return float(np.mean(feasible)), float(np.std(feasible)), float(np.mean(violated))
