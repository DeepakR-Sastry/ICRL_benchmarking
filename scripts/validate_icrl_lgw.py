"""Maximum Entropy ICRL on LapGridWorld, compared with Malik et al.'s
Figure 3a (ICRL, 5 seeds): reward in the constrained world rises to ~58 by
~120k timesteps; violations per step fall from ~0.53 to ~0.01 by ~100k.

Settings: Malik's README command (-er 20 -tk 0.01 -cl 20 -clr 0.003 -bi 20
-dno -dnr -dnc; everything else his CLI defaults). Two schedules, because
the README and the figure disagree (docs/methods.md):
  A: as the README, 10 rounds x 5e4 agent steps (-ni 10 -ft 0.5e5);
  B: as the figure's x-axis implies, 12 rounds x 1e4 (-ni 12 -ft 1e4);
  B_long: B for twice as long, 24 rounds x 1e4 (does the result last?).

Run from anywhere: python scripts/validate_icrl_lgw.py
  [--variants A B B_long] [--seeds 0 1 2 3 4]   (default: A and B, seeds 0-4)
Writes runs/validate_icrl_lgw/<variant>_seed<k>.csv (one row per round) and
config.json; prints mean ± standard error over seeds per round."""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from icrl_bench.agents.ppo_lagrangian import PPOLagrangian, PPOLagrangianSettings
from icrl_bench.core.cost_wrapper import CostWrapper
from icrl_bench.core.loop import ICRLLoopSettings, run_icrl
from icrl_bench.data.piles import load_pile
from icrl_bench.envs.lap_grid_world import LapGridWorld, true_cost
from icrl_bench.featurizers.lap_grid_world import featurize
from icrl_bench.methods.maximum_entropy import MaximumEntropy, MaximumEntropySettings

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "validate_icrl_lgw"
EXPERT = ROOT / "expert_data" / "lgw_forward.pkl"
SEEDS = [0, 1, 2, 3, 4]
N_ENVS = 5
VARIANTS = {"A": dict(n_rounds=10, steps_per_round=50_000),
            "B": dict(n_rounds=12, steps_per_round=10_000),
            "B_long": dict(n_rounds=24, steps_per_round=10_000)}


def settings(variant, seed):
    agent = PPOLagrangianSettings(target_kl=0.01, seed=seed)
    # input_columns (0, 0, 1): the network sees [cell, cell, is_forward], as
    # Malik's does by a bug (docs/methods.md, "Bugs in the official code" (2)).
    method = MaximumEntropySettings(hidden_sizes=(20,), learning_rate=0.003, backward_iters=20,
                                    input_columns=(0, 0, 1), seed=seed)
    loop = ICRLLoopSettings(**VARIANTS[variant], seed=seed)
    return agent, method, loop


def run(job):
    variant, seed = job
    torch.set_num_threads(1)
    agent_s, method_s, loop_s = settings(variant, seed)
    expert, _ = load_pile(EXPERT)
    envs = SyncVectorEnv(
        [lambda: CostWrapper(RecordEpisodeStatistics(LapGridWorld()), None) for _ in range(N_ENVS)],
        autoreset_mode=AutoresetMode.SAME_STEP)  # run_icrl sets the cost before training
    agent = PPOLagrangian(envs.single_observation_space, envs.single_action_space, agent_s)
    method = MaximumEntropy(3, method_s)
    log = run_icrl(agent, envs, method, featurize, expert, LapGridWorld,
                   lambda: LapGridWorld(constrained=True), true_cost, loop_s)
    with open(OUT / f"{variant}_seed{seed}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0]))
        writer.writeheader()
        writer.writerows(log)
    return variant, seed, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["A", "B"], choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = {v: {k: asdict(s) for k, s in zip(["agent", "method", "loop"], settings(v, "<seed>"))}
              for v in VARIANTS}
    (OUT / "config.json").write_text(json.dumps({"n_envs": N_ENVS, "expert": str(EXPERT),
                                                 "variants": config}, indent=2))
    jobs = [(v, s) for v in args.variants for s in args.seeds]
    with ProcessPoolExecutor(max_workers=min(len(jobs), 12)) as pool:
        results = list(pool.map(run, jobs))
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
