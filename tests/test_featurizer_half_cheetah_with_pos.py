import numpy as np

from icrl_bench.core.types import Step, Trajectory
from icrl_bench.featurizers.half_cheetah_with_pos import featurize


def test_rows_are_clipped_obs_then_clipped_action():
    obs = np.zeros(18)
    obs[0], obs[15] = 103.0, -25.0         # x far ahead; a fast joint velocity
    action = np.array([0.5, 1.5, -2.0, 0.0, -0.3, 1.0])
    small_obs = np.full(18, 0.1)
    trajectory = Trajectory([Step(obs, action, 0.0), Step(small_obs, np.zeros(6), 0.0)])

    features = featurize(trajectory)

    assert features.shape == (2, 24) and features.dtype == np.float32
    assert features[0, 0] == 20.0 and features[0, 15] == -20.0     # obs clipped to +-20
    np.testing.assert_array_equal(features[0, 18:],
                                  np.float32([0.5, 1.0, -1.0, 0.0, -0.3, 1.0]))  # action clipped
    np.testing.assert_allclose(features[1, :18], 0.1, rtol=1e-6)    # in range: unchanged
    np.testing.assert_array_equal(features[1, 18:], 0.0)
