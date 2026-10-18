import numpy as np

from icrl_bench.core.cost_wrapper import CostWrapper
from icrl_bench.envs.lap_grid_world import BACKWARD, FORWARD, LapGridWorld, true_cost


class RecordingCost:
    """A cost function that remembers every (obs, action) it is called with."""

    def __init__(self):
        self.calls = []

    def __call__(self, obs, action):
        self.calls.append((obs, action))
        return 0.0


def test_cost_sees_the_observation_the_action_was_chosen_on():
    cost = RecordingCost()
    env = CostWrapper(LapGridWorld(), cost)
    obs0, _ = env.reset()
    obs1, *_ = env.step(FORWARD)
    obs2, *_ = env.step(BACKWARD)
    env.step(FORWARD)
    assert [action for _, action in cost.calls] == [FORWARD, BACKWARD, FORWARD]
    for (seen, _), expected in zip(cost.calls, [obs0, obs1, obs2]):
        np.testing.assert_array_equal(seen, expected)


def test_reset_starts_from_the_new_episode_observation():
    cost = RecordingCost()
    env = CostWrapper(LapGridWorld(), cost)
    env.reset()
    for _ in range(3):
        env.step(FORWARD)  # now at cell 3
    obs0, _ = env.reset()  # back to cell 0
    env.step(FORWARD)
    np.testing.assert_array_equal(cost.calls[-1][0], obs0)


def test_only_info_cost_is_added():
    wrapped = CostWrapper(LapGridWorld(constrained=True), true_cost)
    bare = LapGridWorld(constrained=True)
    np.testing.assert_array_equal(wrapped.reset(seed=0)[0], bare.reset(seed=0)[0])
    for action in [FORWARD] * 5 + [BACKWARD]:
        w_obs, w_rew, w_term, w_trunc, w_info = wrapped.step(action)
        b_obs, b_rew, b_term, b_trunc, b_info = bare.step(action)
        np.testing.assert_array_equal(w_obs, b_obs)
        assert (w_rew, w_term, w_trunc) == (b_rew, b_term, b_trunc)
        assert w_info == {**b_info, "cost": true_cost(None, action)}


def test_set_cost_fn_swaps_the_cost():
    env = CostWrapper(LapGridWorld(), true_cost)
    env.reset()
    assert env.step(BACKWARD)[4]["cost"] == 1.0
    env.set_cost_fn(lambda obs, action: 0.0)
    assert env.step(BACKWARD)[4]["cost"] == 0.0
