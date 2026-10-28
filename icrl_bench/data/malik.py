"""Load the expert rollouts Malik et al. ship with their code (commit
f28214e: icrl/expert_data/<env>/files/EXPERT/rollouts/<i>.pkl) as our
Trajectories.

Each file is one episode, a dict: `observations` (T, obs_dim), `actions`
(T, act_dim), `rewards` (1,) = the episode's total, `lengths` (1,) = T.
His recorder (`utils.sample_from_agent`) stores the obs *after*
`env.step(action)`, so his row t = (obs that action t produced, action t).
We re-pair to our convention, Step(obs the action was chosen on, action):
Step(obs row t-1, action t) for t = 1 ... T-1 (decisions.md, 2026-10-25).
The first action is dropped (the reset obs it was chosen on is not stored)
and the last obs is never used (it is the next episode's reset). Per-step
rewards are not stored, so Step.reward is None (decisions.md, 2026-10-28)."""

import pickle
from pathlib import Path

from icrl_bench.core.types import Step, Trajectory


def load_malik_rollouts(expert_path, n_rollouts):
    """Return files 0 ... n_rollouts-1 as Trajectories, in file order, as his
    `icrl.py` `load_expert_data` reads them: `expert_path` is his `-ep`
    (e.g. .../expert_data/HCWithPos-New), `n_rollouts` his `-er`."""
    trajectories = []
    for i in range(n_rollouts):
        with open(Path(expert_path) / "files/EXPERT/rollouts" / f"{i}.pkl", "rb") as f:
            data = pickle.load(f)
        obs, actions = data["observations"], data["actions"]
        assert len(obs) == len(actions) == data["lengths"][0], f"{i}.pkl is not one whole episode"
        steps = [Step(obs[t - 1], actions[t], None) for t in range(1, len(obs))]
        trajectories.append(Trajectory(steps))
    return trajectories
