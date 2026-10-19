"""Compares our PPO-Lagrangian with Malik's own (his repo, run through his
venv) on LapGridWorld with the true cost: same settings, several seeds.
Closes the gap noted in docs/results.md (Phase 2).

Settings: defaults + target_kl 0.01, no normalisation, budget 0, 1 env.
1 env, not his 5: at f28214e his cpg.py logging callback crashes with more
than one env (true_constraint_net.lap_grid_world gets (n_envs, 1) actions).

Needs Malik's venv (docs/decisions.md, 2026-10-18).
Run from anywhere: python scripts/compare_malik_ppo_lagrangian_lgw.py
Writes runs/compare_malik_ppo_lagrangian_lgw/<impl>_seed<k>.csv; runs whose
CSV already exists are read back instead of rerun (delete a CSV to redo it)."""

import csv
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from icrl_bench.agents.ppo_lagrangian import PPOLagrangian, PPOLagrangianSettings
from icrl_bench.core.cost_wrapper import CostWrapper
from icrl_bench.envs.lap_grid_world import LapGridWorld, true_cost

TOTAL_STEPS = 50_000
SEEDS = [0, 1, 2, 3, 4]
COST_DONE = 0.01  # "cost has reached ~0": rollout average below this
MAX_PARALLEL = 3  # Malik's runs are heavy (torch, wandb, MuJoCo); 10 at once ran out of 15 GB
OUT = Path(__file__).resolve().parents[1] / "runs" / "compare_malik_ppo_lagrangian_lgw"
MALIK_REPO = Path.home() / "refs" / "malik_icrl"
MALIK_PYTHON = Path.home() / "refs" / "malik_venv" / "bin" / "python"


def run_ours(seed):
    """Rows (steps, cost, λ used, ep_rew_mean), one per rollout."""
    envs = SyncVectorEnv([lambda: CostWrapper(RecordEpisodeStatistics(LapGridWorld()), true_cost)],
                         autoreset_mode=AutoresetMode.SAME_STEP)
    agent = PPOLagrangian(envs.single_observation_space, envs.single_action_space,
                          PPOLagrangianSettings(target_kl=0.01, seed=seed))
    return [(row["steps"], row["average_cost"], row["penalty"], row["ep_rew_mean"])
            for row in agent.train(envs, TOTAL_STEPS)]


def run_malik(seed):
    """Runs his cpg.py and parses the tables his logger prints. The block
    for rollout k has that rollout's true_cost and ep_rew_mean, and nu = λ
    after update k-1 (absent in block 1, where λ is still 1) — the same
    alignment as our rows."""
    env = dict(os.environ,
               PATH=f"{MALIK_PYTHON.parent}:{os.environ['PATH']}",  # his venv's ffmpeg (final video)
               LD_LIBRARY_PATH=f"{Path.home()}/.mujoco/mujoco200/bin:{os.environ.get('LD_LIBRARY_PATH', '')}",
               MPLBACKEND="Agg", WANDB_MODE="dryrun", WANDB_SILENT="true",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")  # one core per run, as ours
    command = [str(MALIK_PYTHON), "run_me.py", "cpg", "-p", "compare", "-tei", "LGW-v0",
               "-eei", "CLGW-v0", "-tk", "0.01", "-t", str(TOTAL_STEPS), "-s", str(seed),
               "-nt", "1", "-dno", "-dnr", "-dnc"]
    result = subprocess.run(command, cwd=MALIK_REPO, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Malik's run failed (seed {seed}):\n{result.stderr[-3000:]}")

    blocks, block = [], {}
    for line in result.stdout.splitlines():
        if line.startswith("---"):
            if "total_timesteps" in block:
                blocks.append(block)
            block = {}
            continue
        match = re.match(r"\|\s*(\S+)\s*\|\s*(\S+)\s*\|", line)
        if match:
            block[match.group(1)] = float(match.group(2))
    return [(int(b["total_timesteps"]), b["true_cost"], b.get("nu", 1.0), b["ep_rew_mean"])
            for b in blocks]


def run(job):
    impl, seed = job
    path = OUT / f"{impl}_seed{seed}.csv"
    if path.exists():
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # header
            return job, [(int(row[0]), *map(float, row[1:])) for row in reader]
    torch.set_num_threads(1)
    rows = run_ours(seed) if impl == "ours" else run_malik(seed)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["steps", "cost", "penalty", "ep_rew_mean"])
        writer.writerows(rows)
    return job, rows


def rollouts_until_done(rows):
    for k, (_, cost, _, _) in enumerate(rows, start=1):
        if cost < COST_DONE:
            return k
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(impl, seed) for impl in ["malik", "ours"] for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        results = dict(pool.map(run, jobs))

    print(f"Per seed: rollouts until cost < {COST_DONE}, final λ, final ep_rew_mean")
    for impl in ["malik", "ours"]:
        for seed in SEEDS:
            rows = results[(impl, seed)]
            print(f"  {impl:>5} seed {seed}: {rollouts_until_done(rows)!s:>4} rollouts, "
                  f"λ {rows[-1][2]:.2f}, reward {rows[-1][3]:.1f}")

    print("\nMean over seeds, first 12 rollouts (2048 steps each)")
    print(f"{'rollout':>7}  {'malik cost':>10}  {'ours cost':>10}  {'malik λ':>8}  {'ours λ':>8}")
    for k in range(12):
        line = f"{k + 1:>7}"
        for column, width in [(1, 10), (2, 8)]:
            for impl in ["malik", "ours"]:
                line += f"  {np.mean([results[(impl, s)][k][column] for s in SEEDS]):>{width}.4f}"
        print(line)


if __name__ == "__main__":
    main()
