import numpy as np

from icrl_bench.core.types import Step, Trajectory
from icrl_bench.data.piles import load_pile, save_pile


def assert_same_pile(a, b):
    """Compare field by field: `==` on Steps breaks for multi-number arrays."""
    assert len(a) == len(b)
    for traj_a, traj_b in zip(a, b):
        assert len(traj_a.steps) == len(traj_b.steps)
        for step_a, step_b in zip(traj_a.steps, traj_b.steps):
            for field in ("state", "action"):
                value_a, value_b = getattr(step_a, field), getattr(step_b, field)
                assert type(value_a) is type(value_b)
                np.testing.assert_array_equal(value_a, value_b)
            assert step_a.reward == step_b.reward


def test_round_trip_keeps_every_kind_of_step(tmp_path):
    pile = [
        Trajectory([  # one-number states, discrete actions (like LapGridWorld)
            Step(np.array([-1.0], dtype=np.float32), 0, 0.0),
            Step(np.array([-0.95], dtype=np.float32), 1, 3.0),
        ]),
        Trajectory([  # several numbers per state (like HalfCheetah)
            Step(np.array([0.1, -2.9, 0.5]), np.array([0.2, -1.0]), -0.5),
        ]),
        Trajectory([  # text (like a future web-agent env)
            Step("Delete all files?", "click: confirm", 1.0),
        ]),
    ]
    metadata = {"env": "hand-made", "policy": "none", "n_episodes": 3, "seed": 0}

    save_pile(tmp_path / "pile.pkl", pile, metadata)
    loaded, loaded_metadata = load_pile(tmp_path / "pile.pkl")

    assert_same_pile(loaded, pile)
    assert loaded[0].steps[0].state.dtype == np.float32
    assert loaded_metadata == metadata


def test_save_creates_missing_folders(tmp_path):
    path = tmp_path / "expert_data" / "lgw" / "expert.pkl"
    save_pile(path, [Trajectory([Step(0, 0, 0.0)])], {})
    loaded, _ = load_pile(path)
    assert loaded[0].steps[0] == Step(0, 0, 0.0)
