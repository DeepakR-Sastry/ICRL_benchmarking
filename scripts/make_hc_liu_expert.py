"""Converts Liu's HCWithPos expert trajectories into one of our pile files,
so configs point at it like any other pile. 10 trajectories, as his VICRL
config (running.expert_rollouts: 10), read exactly as his code reads them —
including his pairing mismatch, bug (5) in docs/methods.md. Needs his
expert_data.zip unpacked (README "Download the Expert Data").

Run from anywhere: python scripts/make_hc_liu_expert.py
    [--source DIR]  his expert_path folder; default
                    ~/refs/downloads/liu_expert_data/expert_data/BlockedHalf-cheetah"""

import argparse
import os
from datetime import date
from pathlib import Path

from icrl_bench.data.liu import load_liu_rollouts
from icrl_bench.data.piles import save_pile
from icrl_bench.envs.half_cheetah_with_pos import true_cost

N_ROLLOUTS = 10
SOURCE = Path.home() / "refs/downloads/liu_expert_data/expert_data/BlockedHalf-cheetah"
COMMIT = "0d79d2c"
PATH = Path(__file__).resolve().parents[1] / "expert_data" / "hc_liu.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    source = parser.parse_args().source.expanduser()
    trajectories = load_liu_rollouts(source, N_ROLLOUTS)
    costs = [sum(true_cost(step.state, step.action) for step in t.steps) for t in trajectories]

    metadata = {
        "env": "HCWithPos-v0 (Malik's rollouts, MuJoCo 2)",
        "policy": "Liu's distributed expert data = Malik's shipped rollouts, renamed",
        "source": f"{source}: {sorted(os.listdir(source))[:N_ROLLOUTS]}",
        "malik_files": "0, 2, 18, 19, 20, 21, 22, 23, 24, 25 (same content)",
        "commit": COMMIT,
        "n_episodes": N_ROLLOUTS,
        "pairing": "AS LIU READS IT: Step(row t, action t), t = 0..499 — rows are "
                   "'after the action' (Malik's), so this reproduces bug (5); reward None",
        "created": date.today().isoformat(),
        "episode_true_costs": costs,
    }
    save_pile(PATH, trajectories, metadata)

    print(f"Saved {len(trajectories)} trajectories to {PATH}")
    print(f"steps per episode: {sorted({len(t.steps) for t in trajectories})}")
    print(f"true cost per episode: min {min(costs)}, max {max(costs)}")


if __name__ == "__main__":
    main()
