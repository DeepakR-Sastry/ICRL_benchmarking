"""Runs one ICRL experiment from a config file (docs/design.md, "One entry
point") and writes runs/<name>/: config.yaml and metrics.csv.

Run from the repo root (paths inside configs are relative to it):
  python scripts/run_icrl.py --config configs/lgw/maximum_entropy.yaml
      [--seed 3]           replaces the config's seed
      [--name my_run]      default: <config file name>_seed<seed>"""

import argparse
from pathlib import Path

from icrl_bench.config import load_config
from icrl_bench.run import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--name")
    args = parser.parse_args()

    config = load_config(args.config, seed=args.seed)
    name = args.name or f"{Path(args.config).stem}_seed{config.seed}"
    log = run(config, Path("runs") / name)
    last = log[-1]
    print(f"runs/{name}: {len(log)} rounds, {last['timesteps']} steps, "
          f"reward {last['true/reward']:.2f}, true cost per step {last['true/cost']:.3f}")


if __name__ == "__main__":
    main()
