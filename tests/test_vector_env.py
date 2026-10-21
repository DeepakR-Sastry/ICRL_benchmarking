import numpy as np

from icrl_bench.core.vector_env import make_vector_env
from icrl_bench.envs.lap_grid_world import BACKWARD, FORWARD, LapGridWorld, true_cost


def test_every_copy_reports_its_own_cost():
    envs = make_vector_env(LapGridWorld, 3, true_cost)
    envs.reset(seed=0)
    infos = envs.step(np.array([BACKWARD, FORWARD, BACKWARD]))[4]
    np.testing.assert_array_equal(infos["cost"], [1.0, 0.0, 1.0])


def test_set_cost_fn_reaches_every_copy():
    envs = make_vector_env(LapGridWorld, 3, None)  # as the ICRL loop builds it
    envs.call("set_cost_fn", true_cost)
    envs.reset(seed=0)
    infos = envs.step(np.array([BACKWARD] * 3))[4]
    np.testing.assert_array_equal(infos["cost"], [1.0, 1.0, 1.0])


def test_copies_are_separate_envs():
    envs = make_vector_env(LapGridWorld, 2, true_cost)
    assert envs.envs[0] is not envs.envs[1]


def test_episode_statistics_are_recorded_and_the_env_resets_in_the_same_step():
    envs = make_vector_env(lambda: LapGridWorld(max_episode_steps=2), 2, true_cost)
    first_obs, _ = envs.reset(seed=0)
    envs.step(np.array([FORWARD, FORWARD]))
    obs, rewards, terminated, truncated, infos = envs.step(np.array([FORWARD, FORWARD]))
    assert truncated.all()
    # The finished episode's info is under "final_info", where the agents read it.
    np.testing.assert_array_equal(infos["final_info"]["episode"]["l"], [2, 2])
    np.testing.assert_array_equal(obs, first_obs)  # SAME_STEP: already reset
