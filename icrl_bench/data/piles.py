"""Save and load piles (lists of Trajectories) together with metadata on how
they were made. Pickle stores Steps of any type (ints, arrays, text) as-is.
Loading a pickle can run code: only load piles we made or the papers' repos
ship."""

import pickle
from pathlib import Path


def save_pile(path, trajectories, metadata):
    """Write `trajectories` and a `metadata` dict (env, policy, seed, ...) to one file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"metadata": metadata, "trajectories": trajectories}, f)


def load_pile(path):
    """Return (trajectories, metadata) from a file written by save_pile."""
    with open(path, "rb") as f:
        pile = pickle.load(f)
    return pile["trajectories"], pile["metadata"]
