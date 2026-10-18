"""LapGridWorld, ported from Malik et al. (github.com/shehryar-malik/icrl,
commit f28214e): custom_envs/custom_envs/envs/lap_grid_world.py for the env,
icrl/true_constraint_net.py for the true cost. Port notes: docs/envs.md."""

import gymnasium as gym
import numpy as np

N_CELLS = 40                  # a square lap, 11 cells per side
COIN_CELLS = (5, 15, 25, 35)  # the middle cell of each side
COIN_REWARD = 3.0
FORWARD, BACKWARD = 0, 1


class LapGridWorld(gym.Env):
    """Walk around a lap of 40 cells, starting at cell 0. Landing on a coin
    cell pays +3; coins never run out. Moving forward (clockwise) laps the
    track; moving backward is the behaviour the constraint should forbid.

    constrained=False is Malik's LGW-v0 (backward allowed).
    constrained=True is Malik's CLGW-v0 (backward: stay put, reward -1,
    episode ends). It is the "real" world, used only for evaluation.
    """

    def __init__(self, constrained=False, max_episode_steps=200):
        self.constrained = constrained
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cell = 0
        self.time = 0
        return self._obs(), {}

    def step(self, action):
        assert action in (FORWARD, BACKWARD), action
        if action == BACKWARD and self.constrained:
            reward, terminated = -1.0, True
        else:
            move = 1 if action == FORWARD else -1
            self.cell = (self.cell + move) % N_CELLS
            reward = COIN_REWARD if self.cell in COIN_CELLS else 0.0
            terminated = False
        self.time += 1
        truncated = self.time == self.max_episode_steps
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        # Malik's scaling: divides by 40, not 39, so cells 0..39 -> -1.0..0.95.
        return np.array([2 * self.cell / N_CELLS - 1], dtype=np.float32)


def true_cost(obs, action):
    """The hidden rule: a backward step costs 1, a forward step 0."""
    return float(action == BACKWARD)
