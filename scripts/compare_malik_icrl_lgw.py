"""Runs Malik's own icrl.py (his repo, through his venv) on LapGridWorld with
the settings of our validation's schedule B, and compares it per round with
our runs (scripts/validate_icrl_lgw.py, runs/validate_icrl_lgw/B_seed*/metrics.csv).

His settings: the README command with Figure 3a's schedule:
  -er 20 -tk 0.01 -cl 20 -clr 0.003 -bi 20 -dno -dnr -dnc -ft 1e4 -ni 12,
5 seeds, his default 5 envs (-nt 5).

Expert data: he ships the LapGridWorld expert model but not its rollouts.
We make 20 always-forward rollouts with his own sample_from_agent (his
pairing: action + next obs) on CLGW-v0, as his run_policy.py saves them,
plus his best_model.zip (icrl.py loads it, for KL metrics only).

Needs Malik's venv (docs/decisions.md, 2026-10-18). Our runs must exist.
Run from anywhere: python scripts/compare_malik_icrl_lgw.py [--rounds 12] [--seeds 0 1 2 3 4]
(--rounds 24 compares with our B_long runs.)
Writes runs/compare_malik_icrl_lgw/malik_r<rounds>_seed<k>.csv; runs whose CSV already
exists are read back instead of rerun (delete a CSV to redo it)."""

import argparse
import csv
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SEEDS = [0, 1, 2, 3, 4]
MAX_PARALLEL = 4  # his runs are heavy; see compare_malik_ppo_lagrangian_lgw.py
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "compare_malik_icrl_lgw"
OURS = ROOT / "runs" / "validate_icrl_lgw"
EXPERT = OUT / "expert"
MALIK_REPO = Path.home() / "refs" / "malik_icrl"
MALIK_PYTHON = Path.home() / "refs" / "malik_venv" / "bin" / "python"
ENV = dict(os.environ,
           PATH=f"{MALIK_PYTHON.parent}:{os.environ['PATH']}",  # his venv's ffmpeg (final video)
           LD_LIBRARY_PATH=f"{Path.home()}/.mujoco/mujoco200/bin:{os.environ.get('LD_LIBRARY_PATH', '')}",
           PYTHONPATH=str(MALIK_REPO), MPLBACKEND="Agg", WANDB_MODE="dryrun", WANDB_SILENT="true",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
KEYS = ["timesteps", "true/reward", "true/cost"]

MAKE_EXPERT = """
import os, sys
import numpy as np
import icrl.utils as utils

class Forward:
    def predict(self, obs, state=None, deterministic=False):
        return np.zeros(len(obs), dtype=int), None

env = utils.make_eval_env("CLGW-v0", use_cost_wrapper=False, normalize_obs=False)
rollouts = os.path.join(sys.argv[1], "files", "EXPERT", "rollouts")
os.makedirs(rollouts)
for i in range(20):
    obs, _, acs, rewards, lengths = utils.sample_from_agent(Forward(), env, 1)
    utils.save_dict_as_pkl(dict(observations=obs, actions=acs, rewards=rewards, lengths=lengths,
                                save_scheme="not_airl"), rollouts, str(i))
"""


def make_expert():
    if EXPERT.exists():
        return
    subprocess.run([str(MALIK_PYTHON), "-c", MAKE_EXPERT, str(EXPERT)], cwd=MALIK_REPO, env=ENV,
                   check=True, capture_output=True)
    shutil.copy(MALIK_REPO / "icrl" / "expert_data" / "LGW" / "files" / "best_model.zip",
                EXPERT / "files" / "best_model.zip")


def parse(stdout):
    """One dict per round from the tables his icrl_logger prints. Group rows
    ("| true/ |  |") prefix the indented rows under them."""
    rounds, row, group = [], {}, ""
    for line in stdout.splitlines():
        if line.startswith("---"):
            if "iteration" in row:
                rounds.append(row)
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
    return rounds


def run_malik(job):
    seed, rounds = job
    path = OUT / f"malik_r{rounds}_seed{seed}.csv"
    if not path.exists():
        command = [str(MALIK_PYTHON), "run_me.py", "icrl", "-p", "compare", "-er", "20",
                   "-ep", str(EXPERT), "-tei", "LGW-v0", "-eei", "CLGW-v0", "-tk", "0.01",
                   "-cl", "20", "-clr", "0.003", "-ft", "1e4", "-ni", str(rounds), "-bi", "20",
                   "-dno", "-dnr", "-dnc", "-s", str(seed)]
        result = subprocess.run(command, cwd=MALIK_REPO, env=ENV, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Malik's run failed (seed {seed}):\n{result.stderr[-3000:]}")
        rounds = parse(result.stdout)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rounds[0]))
            writer.writeheader()
            writer.writerows(rounds)
    with open(path) as f:
        return seed, [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def read_ours(seed, rounds):
    variant = {12: "B", 24: "B_long"}[rounds]
    with open(OURS / f"{variant}_seed{seed}" / "metrics.csv") as f:
        return [{k: float(row[k]) for k in KEYS} for row in csv.DictReader(f)]


def mean_se(values):
    values = np.array(values)
    return f"{values.mean():6.2f} ± {values.std(ddof=1) / np.sqrt(len(values)):5.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=12, choices=[12, 24])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    seeds = args.seeds
    OUT.mkdir(parents=True, exist_ok=True)
    make_expert()
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        malik = dict(pool.map(run_malik, [(s, args.rounds) for s in seeds]))
    ours = {seed: read_ours(seed, args.rounds) for seed in seeds}

    print(f"Mean ± standard error over {len(seeds)} seeds, per round (round k ends at (k+1) x 10,240 steps)")
    print(f"{'round':>5}  {'malik reward':>15}  {'ours reward':>15}  {'malik cost':>15}  {'ours cost':>15}")
    for r in range(args.rounds):
        line = f"{r:>5}"
        for key in ["true/reward", "true/cost"]:
            for runs in [malik, ours]:
                line += f"  {mean_se([runs[s][r][key] for s in seeds]):>15}"
        print(line)
    print("\nFinal round per seed (reward, cost)")
    for name, runs in [("malik", malik), ("ours", ours)]:
        print(f"  {name:>5}: " + ", ".join(f"({runs[s][-1]['true/reward']:.1f}, {runs[s][-1]['true/cost']:.3f})"
                                          for s in seeds))


if __name__ == "__main__":
    main()
