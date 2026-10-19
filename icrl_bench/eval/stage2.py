"""Stage-2 metrics: how safe and how good the agent is, graded with the true
cost. As Malik et al. log them each ICRL round (icrl/icrl.py L241-252,
github.com/shehryar-malik/icrl, commit f28214e)."""

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
