"""Maximum Entropy ICRL on LapGridWorld, compared with Malik et al.'s
Figure 3a (ICRL, 5 seeds): reward in the constrained world rises to ~58 by
~120k timesteps; violations per step fall from ~0.53 to ~0.01 by ~100k.

Settings: Malik's README command, in configs/lgw/. Three schedules, because
the README and the figure disagree (docs/methods.md):
  A: as the README, 10 rounds x 5e4 agent steps (maximum_entropy_A.yaml);
  B: as the figure's x-axis implies, 12 rounds x 1e4 (maximum_entropy.yaml);
  B_long: B for twice as long, 24 rounds x 1e4 (does the result last?).

Run from the repo root: python scripts/validate_icrl_lgw.py
  [--variants A B B_long] [--seeds 0 1 2 3 4]   (default: A and B, seeds 0-4)
Writes runs/validate_icrl_lgw/<variant>_seed<k>/ (config.yaml, metrics.csv);
prints mean ± standard error over seeds per round."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from icrl_bench.config import load_config
from icrl_bench.run import run

OUT = Path("runs") / "validate_icrl_lgw"
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = {"A": "configs/lgw/maximum_entropy_A.yaml",
            "B": "configs/lgw/maximum_entropy.yaml",
            "B_long": "configs/lgw/maximum_entropy_B_long.yaml"}


def job(variant_seed):
    variant, seed = variant_seed
    # One thread per process: the runs go in parallel, and a fixed thread
    # count keeps float results identical from run to run.
    torch.set_num_threads(1)
    log = run(load_config(VARIANTS[variant], seed=seed), OUT / f"{variant}_seed{seed}")
    return variant, seed, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["A", "B"], choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    jobs = [(v, s) for v in args.variants for s in args.seeds]
    with ProcessPoolExecutor(max_workers=min(len(jobs), 12)) as pool:
        results = list(pool.map(job, jobs))
    for variant in args.variants:
        logs = [log for v, _, log in results if v == variant]
        print(f"\nVariant {variant} ({VARIANTS[variant]}), mean ± standard error over {len(logs)} seeds")
        for r in range(len(logs[0])):
            rows = [log[r] for log in logs]
            line = f"  round {r:2d}, {rows[0]['timesteps']:>7} steps:"
            for key in ["true/reward", "true/cost", "forward/penalty", "backward/early_stop_itr"]:
                values = np.array([row[key] for row in rows], dtype=float)
                line += f"  {key.split('/')[1]} {values.mean():6.2f} ± {values.std(ddof=1) / np.sqrt(len(values)):.2f}"
            print(line)


if __name__ == "__main__":
    main()
