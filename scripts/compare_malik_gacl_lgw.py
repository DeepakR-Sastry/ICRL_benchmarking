"""Runs Malik's own gail.py (his repo, through his venv) on LapGridWorld with
his README command, and compares it per rollout with our GACL runs
(configs/lgw/gacl.yaml, runs/gacl_lgw/seed*/metrics.csv).

His settings, README:
  run_me.py gail -er 20 -ep icrl/expert_data/LGW -tei LGW-v0 -eei CLGW-v0
  -tk 0.01 -dl 20 -dlr 0.01 -t 120000 --n_steps 2000 -dno -dnr -lc
5 seeds, his default 5 envs.

His gail.py crashes on LapGridWorld at f28214e: get_plot_func's
`elif 'DD2B' or 'C2B' in env_id` is always true, so a 2-D scatter plot is
drawn from the 1-column observation (plot_utils.py L14, L113). We start his
gail.main() from a wrapper that replaces get_plot_func with a no-op; his
files are untouched, and plotting only reads the buffer (no random numbers,
nothing training sees) (docs/decisions.md, 2026-11-01).

Expert data: the same files compare_malik_icrl_lgw.py makes with his own
sample_from_agent (runs/compare_malik_icrl_lgw/expert; run that script once
first).

Alignment: his EvalCallback evaluates during rollout k+1 with the policy of
update k, logged in his row k+1; ours evaluates right after update k,
logged in our row k. Evaluations are compared by update; cost and
discriminator metrics by rollout.

Needs Malik's venv (docs/decisions.md, 2026-10-18). Our runs must exist.
Run from anywhere: python scripts/compare_malik_gacl_lgw.py [--seeds 0 1 2 3 4]
Writes runs/compare_malik_gacl_lgw/malik_seed<k>.csv; runs whose CSV already
exists are read back instead of rerun (delete a CSV to redo it)."""

import argparse
import csv
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SEEDS = [0, 1, 2, 3, 4]
MAX_PARALLEL = 4  # as compare_malik_icrl_lgw.py
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "compare_malik_gacl_lgw"
OURS = ROOT / "runs" / "gacl_lgw"
EXPERT = ROOT / "runs" / "compare_malik_icrl_lgw" / "expert"
MALIK_REPO = Path.home() / "refs" / "malik_icrl"
MALIK_PYTHON = Path.home() / "refs" / "malik_venv" / "bin" / "python"
ENV = dict(os.environ,
           PATH=f"{MALIK_PYTHON.parent}:{os.environ['PATH']}",
           LD_LIBRARY_PATH=f"{Path.home()}/.mujoco/mujoco200/bin:{os.environ.get('LD_LIBRARY_PATH', '')}",
           PYTHONPATH=str(MALIK_REPO), MPLBACKEND="Agg", WANDB_MODE="dryrun", WANDB_SILENT="true",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
README_FLAGS = ["-er", "20", "-ep", str(EXPERT), "-tei", "LGW-v0", "-eei", "CLGW-v0", "-tk", "0.01",
                "-dl", "20", "-dlr", "0.01", "-t", "120000", "--n_steps", "2000", "-dno", "-dnr", "-lc"]

# Runs his gail.main() with plotting replaced by a no-op (see above).
WRAPPER = """
import sys
import icrl.gail as gail
gail.get_plot_func = lambda env_id: (lambda *args: None)
sys.argv = ["run_me.py", "gail"] + sys.argv[1:]
gail.main()
"""


def parse(stdout):
    """One dict per rollout from the tables his SB3 logger prints. Group rows
    ("| eval/ |  |") prefix the indented rows under them."""
    rollouts, row, group = [], {}, ""
    for line in stdout.splitlines():
        if line.startswith("---"):
            if "time/total_timesteps" in row:
                rollouts.append(row)
            row, group = {}, ""
            continue
        match = re.match(r"\|(\s*)(\S+)\s*\|\s*(\S*)\s*\|", line)
        if not match:
            continue
        indent, key, value = match.groups()
        if not value:
            group = key
        else:
            row[(group if len(indent) > 1 else "") + key] = float(value)
    return rollouts


def run_malik(seed):
    path = OUT / f"malik_seed{seed}.csv"
    if not path.exists():
        command = [str(MALIK_PYTHON), "-c", WRAPPER, "-p", "compare", *README_FLAGS, "-s", str(seed)]
        result = subprocess.run(command, cwd=MALIK_REPO, env=ENV, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Malik's run failed (seed {seed}):\n{result.stderr[-3000:]}")
        rollouts = parse(result.stdout)
        # Rows differ in keys (eval and train rows appear from rollout 2 on).
        keys = sorted({k for row in rollouts for k in row})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, restval="nan")
            writer.writeheader()
            writer.writerows(rollouts)
    with open(path) as f:
        return seed, [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def read_ours(seed):
    with open(OURS / f"seed{seed}" / "metrics.csv") as f:
        return [{k: float(v) for k, v in row.items() if v != ""} for row in csv.DictReader(f)]


def mean_se(values):
    values = np.array(values)
    return f"{values.mean():6.3f} ± {values.std(ddof=1) / np.sqrt(len(values)):5.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    seeds = args.seeds
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        malik = dict(pool.map(run_malik, seeds))
    ours = {seed: read_ours(seed) for seed in seeds}
    n = len(ours[seeds[0]])

    print(f"Mean ± standard error over {len(seeds)} seeds, per rollout (rollout k ends at (k+1) x 10,000 steps)")
    print("Violations per step on the training rollout; discriminator on nominal / expert (that update's batch)")
    print(f"{'k':>3}  {'malik cost':>15}  {'ours cost':>15}  {'malik D nom/exp':>16}  {'ours D nom/exp':>16}")
    for k in range(n):
        m = [malik[s][k] for s in seeds]
        o = [ours[s][k] for s in seeds]
        d_m = f"{np.mean([r['discriminator/mean_nominal_preds'] for r in m]):.2f} / " \
              f"{np.mean([r['discriminator/mean_expert_preds'] for r in m]):.2f}"
        d_o = f"{np.mean([r['backward/mean_nominal_preds'] for r in o]):.2f} / " \
              f"{np.mean([r['backward/mean_expert_preds'] for r in o]):.2f}"
        print(f"{k:>3}  {mean_se([r['eval/mean_cost'] for r in m]):>15}  "
              f"{mean_se([r['true/cost'] for r in o]):>15}  {d_m:>16}  {d_o:>16}")

    print("\nEvaluation reward (5 sampled episodes, constrained env) of the policy after rollout k's update")
    print(f"{'k':>3}  {'malik reward':>15}  {'ours reward':>15}")
    for k in range(n - 1):  # his row k+1 = ours row k; neither evaluates after the last update
        print(f"{k:>3}  {mean_se([malik[s][k + 1]['eval/mean_reward'] for s in seeds]):>15}  "
              f"{mean_se([ours[s][k]['true/reward'] for s in seeds]):>15}")

    print("\nLast rollout per seed (cost)")
    for name, runs, key in [("malik", malik, "eval/mean_cost"), ("ours", ours, "true/cost")]:
        print(f"  {name:>5}: " + ", ".join(f"{runs[s][-1][key]:.3f}" for s in seeds))


if __name__ == "__main__":
    main()
