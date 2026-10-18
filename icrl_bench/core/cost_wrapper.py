"""Puts a cost into info["cost"] on every step, so any agent can read it.

Follows Malik et al.'s VecCostWrapper (github.com/shehryar-malik/icrl, commit
f28214e, stable_baselines3/common/vec_env/vec_cost_wrapper.py): the cost is
cost_fn(previous observation, action), and the wrapper goes directly on the
raw env, before any normalization (icrl/utils.py, make_train_env)."""

import gymnasium as gym


class CostWrapper(gym.Wrapper):
    """cost_fn(obs, action) -> float is an env's true cost, a learned
    constraint, or zero. `obs` is the observation the action was chosen on."""

    def __init__(self, env, cost_fn):
        super().__init__(env)
        self.cost_fn = cost_fn

    def set_cost_fn(self, cost_fn):
        """Swap the cost, e.g. for the newly learned one between ICRL rounds."""
        self.cost_fn = cost_fn

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.previous_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["cost"] = self.cost_fn(self.previous_obs, action)
        self.previous_obs = obs
        return obs, reward, terminated, truncated, info
