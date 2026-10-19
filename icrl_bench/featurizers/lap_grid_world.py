"""Featurizer for LapGridWorld: the observation and the one-hot action, as
Malik's ConstraintNet.prepare_data builds them (icrl/constraint_net.py,
commit f28214e) with his LapGridWorld settings (no normalisation). His
network then keeps only some of these columns, by a bug; that is a method
setting (MaximumEntropySettings.input_columns), not the featurizer's job."""

import numpy as np

from icrl_bench.envs.lap_grid_world import BACKWARD, FORWARD


def featurize(trajectory):
    """One row per step: [cell scaled to -1..1, is_forward, is_backward].

    The cell is the env's observation unchanged. Malik also clips
    observations to +-20 (clip_obs), which never changes values in -1..1, so
    it is left out. The action is one-hot encoded, as in his reshape_actions.
    """
    rows = []
    for step in trajectory.steps:
        assert step.action in (FORWARD, BACKWARD), step.action
        cell = float(step.state[0])
        rows.append([cell, float(step.action == FORWARD), float(step.action == BACKWARD)])
    return np.array(rows, dtype=np.float32)
