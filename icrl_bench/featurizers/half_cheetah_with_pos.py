"""Featurizer for HalfCheetahWithPos: the clipped observation and the clipped
action, as Malik's ConstraintNet.prepare_data builds them
(icrl/constraint_net.py, commit f28214e) with his HCWithPos settings (no
normalisation, --clip_obs 20). His network then keeps only some of these
columns, by a bug; that is a method setting (input_columns), not the
featurizer's job."""

import numpy as np

CLIP_OBS = 20.0  # Malik's --clip_obs default (icrl/icrl.py)
ACTION_LOW, ACTION_HIGH = -1.0, 1.0  # the env's action bounds


def featurize(trajectory):
    """One row per step: 18 observation numbers (x-position first) clipped to
    +-20, then 6 torques clipped to [-1, 1]. 24 columns."""
    obs = np.array([step.state for step in trajectory.steps], dtype=np.float64)
    actions = np.array([step.action for step in trajectory.steps], dtype=np.float64)
    assert obs.shape[1:] == (18,) and actions.shape[1:] == (6,), (obs.shape, actions.shape)
    obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)
    actions = np.clip(actions, ACTION_LOW, ACTION_HIGH)
    return np.concatenate([obs, actions], axis=1).astype(np.float32)
