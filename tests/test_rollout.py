import gymnasium as gym
import numpy as np

from icrl_bench.core.rollout import rollout
from icrl_bench.envs.lap_grid_world import BACKWARD, FORWARD, LapGridWorld


def cell_of(obs):
    return round((obs[0] + 1) * 20)


def bounce_on_first_coin(obs):
    return BACKWARD if cell_of(obs) == 5 else FORWARD


class RecordingPolicy:
    """Bounces on the first coin and remembers every obs it is shown."""

    def __init__(self):
        self.seen = []

    def __call__(self, obs):
        self.seen.append(obs)
        return bounce_on_first_coin(obs)


class ResetSpy(gym.Wrapper):
    """Remembers the seed passed to every reset."""

    def __init__(self, env):
        super().__init__(env)
        self.seeds = []

    def reset(self, *, seed=None, options=None):
        self.seeds.append(seed)
        return self.env.reset(seed=seed, options=options)


def test_step_holds_the_obs_the_action_was_chosen_on():
    policy = RecordingPolicy()
    [trajectory] = rollout(LapGridWorld(), policy, n_episodes=1, seed=0)
    assert len(policy.seen) == len(trajectory.steps)
    for step, seen in zip(trajectory.steps, policy.seen):
        np.testing.assert_array_equal(step.state, seen)
        assert step.action == bounce_on_first_coin(seen)
    # Spelled out: forward from cells 0-4, then back from the coin at cell 5.
    assert [cell_of(s.state) for s in trajectory.steps[:7]] == [0, 1, 2, 3, 4, 5, 4]
    assert [s.action for s in trajectory.steps[:7]] == [FORWARD] * 5 + [BACKWARD, FORWARD]


def test_one_trajectory_per_episode_ending_where_the_env_ends():
    nominal = rollout(LapGridWorld(), bounce_on_first_coin, n_episodes=3, seed=0)
    constrained = rollout(LapGridWorld(constrained=True), bounce_on_first_coin, n_episodes=3, seed=0)
    assert [len(t.steps) for t in nominal] == [200, 200, 200]
    assert [len(t.steps) for t in constrained] == [6, 6, 6]
    assert [sum(s.reward for s in t.steps) for t in nominal] == [294, 294, 294]
    assert [sum(s.reward for s in t.steps) for t in constrained] == [2, 2, 2]


def test_env_is_seeded_only_on_the_first_reset():
    env = ResetSpy(LapGridWorld())
    rollout(env, bounce_on_first_coin, n_episodes=3, seed=7)
    assert env.seeds == [7, None, None]


def test_rewards_are_plain_floats():
    [trajectory] = rollout(LapGridWorld(), bounce_on_first_coin, n_episodes=1, seed=0)
    assert all(type(s.reward) is float for s in trajectory.steps)
