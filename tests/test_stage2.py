import numpy as np

from icrl_bench.core.types import Step, Trajectory
from icrl_bench.eval.stage2 import feasible_reward_stats, liu_first_violation, true_cost_rate


def negative_is_unsafe(obs, action):
    """A toy rule: the step is a violation if the observation is negative."""
    return float(obs < 0)


def trajectory(states, rewards):
    return Trajectory([Step(s, 0, r) for s, r in zip(states, rewards)])


def test_liu_judges_a_step_on_the_observation_it_led_to():
    # Step 1 (state 2) leads to state -1: Liu's first violation is step 1.
    # Under our convention (the state the action was chosen on) it is step 2.
    t = trajectory([1, 2, -1, -2, 3], [3, 4, 5, 6, 7])
    assert liu_first_violation(t, negative_is_unsafe) == 1
    assert [negative_is_unsafe(s.state, s.action) for s in t.steps].index(1.0) == 2


def test_feasible_reward_stops_before_the_first_violation():
    unsafe = trajectory([1, 2, -1, -2, 3], [3, 4, 5, 6, 7])   # feasible 3 (step 1 excluded)
    safe = trajectory([1, 2, 3], [1, 1, 1])                   # feasible = all = 3
    mean, std, violated = feasible_reward_stats([unsafe, safe], negative_is_unsafe)
    assert (mean, std, violated) == (3.0, 0.0, 0.5)


def test_violation_at_the_first_step_gives_zero_feasible_reward():
    t = trajectory([1, -1, 2], [5, 5, 5])                     # step 0 leads to -1
    assert feasible_reward_stats([t], negative_is_unsafe) == (0.0, 0.0, 1.0)


def test_the_last_step_is_judged_on_its_own_state():
    # No observation after the last step is stored: a negative last state
    # counts as the last step's own violation.
    t = trajectory([1, 2, -1], [1, 1, 1])
    assert liu_first_violation(t, negative_is_unsafe) == 1   # step 1 leads to -1
    t = trajectory([1, 2, 3], [1, 1, 1])
    assert liu_first_violation(t, negative_is_unsafe) is None


def test_true_cost_rate_is_per_step_over_all_trajectories():
    t1 = trajectory([1, -1, -2, 3], [0] * 4)                  # 2 of 4 steps
    t2 = trajectory([1, 1], [0] * 2)                          # 0 of 2
    assert np.isclose(true_cost_rate([t1, t2], negative_is_unsafe), 2 / 6)
