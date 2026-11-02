import pickle

import numpy as np

from icrl_bench.data.liu import load_liu_rollouts


def write_episode(path, name, first_obs, length):
    """A file in Liu's format; row t's observation is first_obs + t."""
    obs = np.arange(length, dtype=np.float64)[:, None] + first_obs + np.zeros((1, 18))
    data = {"original_observations": obs, "observations": obs / 10,
            "actions": np.arange(length)[:, None] + np.zeros((1, 6)),
            "reward": np.ones(length), "reward_sum": float(length)}
    with open(path / name, "wb") as f:
        pickle.dump(data, f)


def test_reads_the_first_files_in_sorted_name_order_without_re_pairing(tmp_path):
    # Written out of order; his loader sorts the names and takes the first n.
    write_episode(tmp_path, "scene-2_len-4_Game Finished.pkl", 200.0, 4)
    write_episode(tmp_path, "scene-0_len-3_Game Finished.pkl", 0.0, 3)
    write_episode(tmp_path, "scene-1_len-5_Game Finished.pkl", 100.0, 5)

    trajectories = load_liu_rollouts(tmp_path, 2)

    assert [len(t.steps) for t in trajectories] == [3, 5]     # scene-0, scene-1
    first = trajectories[0].steps
    # Row t = (raw obs t, action t): no shift, raw (not the normalised) obs.
    assert [s.state[0] for s in first] == [0.0, 1.0, 2.0]
    assert [s.action[0] for s in first] == [0.0, 1.0, 2.0]
    assert trajectories[1].steps[0].state[0] == 100.0
    assert all(s.reward is None for t in trajectories for s in t.steps)
