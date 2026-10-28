"""Runs one ICRL experiment from a config file (docs/design.md, "One entry
point") and writes runs/<name>/: config.yaml and metrics.csv.

Run from the repo root (paths inside configs are relative to it):
  python scripts/run_icrl.py --config configs/lgw/maximum_entropy.yaml
      [--seed 3]           replaces the config's seed
      [--name my_run]      default: <config file name>_seed<seed>
      [--wandb-project P]  also log every round to wandb, under Malik's
      [--wandb-group G]    icrl.py metric names (default group "ours")"""

import argparse
from dataclasses import asdict
from pathlib import Path

from icrl_bench.config import load_config
from icrl_bench.run import run

# Our metrics.csv column -> the name Malik's icrl.py logs it under (L279-303).
# His forward names are his SB3 logger's with "train/" -> "forward/", as in
# validate_ppo_lagrangian_hc.py's WANDB_NAMES. Columns not listed already
# have his name (true/*, backward/*, timesteps) or are ours only.
MALIK_NAMES = {
    "round": "iteration",
    "forward/ep_rew_mean": "rollout/ep_rew_mean",
    "forward/penalty": "forward/nu",
    "forward/policy_loss": "forward/policy_gradient_loss",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--name")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-group", default="ours")
    args = parser.parse_args()

    config = load_config(args.config, seed=args.seed)
    name = args.name or f"{Path(args.config).stem}_seed{config.seed}"

    callback, wandb_run = None, None
    if args.wandb_project is not None:
        import wandb  # only needed when logging to wandb
        wandb_run = wandb.init(project=args.wandb_project, group=args.wandb_group, name=name,
                               config=asdict(config))

        def callback(row):
            # wandb's step = the round, as Malik's wandb.log(metrics) once per round.
            wandb_run.log({MALIK_NAMES.get(k, k): v for k, v in row.items()}, step=row["round"])

    log = run(config, Path("runs") / name, callback)
    if wandb_run is not None:
        wandb_run.finish()
    last = log[-1]
    print(f"runs/{name}: {len(log)} rounds, {last['timesteps']} steps, "
          f"reward {last['true/reward']:.2f}, true cost per step {last['true/cost']:.3f}")


if __name__ == "__main__":
    main()
