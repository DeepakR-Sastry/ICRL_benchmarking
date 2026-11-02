"""Load the expert trajectories Liu et al. distribute for their benchmark
(github.com/Guiliang/ICRL-benchmarks-public, commit 0d79d2c; downloaded
separately as expert_data.zip, README "Download the Expert Data") as our
Trajectories, exactly as his code reads them.

Each file in an expert folder (e.g. expert_data/BlockedHalf-cheetah/) is
one episode, a pickled dict with `original_observations` (T, obs_dim) raw
and `actions` (T, act_dim), among other keys. His loader pairs row t with
row t; so does this one.

**Bug (5), reproduced on purpose (docs/methods.md; decisions.md
2026-11-01):** his BlockedHalf-cheetah files are Malik's rollouts, recorded
*after* each step — row t = (observation action t produced, action t), the
last row the next episode's reset — while his nominal sampler (and ours)
pairs each action with the observation it was chosen on. Reading them
unshifted, as he does, reproduces his runs; the corrected pile is
data/malik.py's re-pairing of the same files. Step.reward is None: no
method reads expert rewards."""

import os
import pickle
from pathlib import Path

from icrl_bench.core.types import Step, Trajectory


def load_liu_rollouts(expert_path, n_rollouts):
    """Return the first n_rollouts files of `expert_path`, in sorted file-name
    order, as his utils/data_utils.py load_expert_data reads them
    (add_next_step=False, raw `original_observations`). `expert_path` is his
    config's running.expert_path, `n_rollouts` its expert_rollouts."""
    names = sorted(os.listdir(expert_path))
    assert n_rollouts <= len(names), f"{expert_path} holds {len(names)} files, asked for {n_rollouts}"
    trajectories = []
    for name in names[:n_rollouts]:
        with open(Path(expert_path) / name, "rb") as f:
            data = pickle.load(f)
        obs, actions = data["original_observations"], data["actions"]
        assert len(obs) == len(actions), f"{name}: {len(obs)} observations, {len(actions)} actions"
        trajectories.append(Trajectory([Step(obs[t], actions[t], None) for t in range(len(obs))]))
    return trajectories
