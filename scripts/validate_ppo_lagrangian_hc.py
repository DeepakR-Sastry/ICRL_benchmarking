"""Our PPO-Lagrangian on HalfCheetah-with-position with the TRUE cost (x <= -3
costs 1), budget 0. Expected: it runs forward, like Malik's expert — about
4.4 reward per step and no violations (his 50 shipped episodes: 2194 ± 316
per 500 steps, x never below -0.19; docs/envs.md).

Settings as Malik's HalfCheetah expert (icrl/expert_data/HCWithPos-New/files/
config.json; docs/agents.md): defaults + target_kl 0.01, 5 envs, 5M steps,
observations and rewards normalised, costs not.

Run from the repo root: python scripts/validate_ppo_lagrangian_hc.py
    [--seeds 0 1 2]         which seeds to run, one process each (default 0 1 2).
    [--wandb-project NAME]  also log every row to wandb (group "ours"), under
                            Malik's SB3 metric names where the meaning is
                            the same, so his synced cpg.py runs share charts.
Writes runs/validate_ppo_lagrangian_hc/seed<k>/: train.csv (one row per
rollout) and eval.csv (one row per checkpoint), each row written as soon as
it exists. Two evaluations, both with the sampling policy:
- Every rollout (10,240 steps), as Malik's EvalCallback (cpg.py; eval_freq
  2048 calls of a 5-copy vector env): FINE_EVAL_EPISODES in the constrained
  env, mean and std reward added to that rollout's train.csv row. One env,
  seeded on its first reset only, as his single eval env. His runs just
  before each rollout's update, ours just after: our step N = his N + 10,240.
- Every EVAL_EVERY steps, into eval.csv:
N_EVAL_EPISODES in the constrained env (Malik's HCWithPosTest-v0: ends at
x <= -3 with reward 0) and N_EVAL_EPISODES in the nominal env, where
violations are counted with true_cost (Liu's measure)."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

from icrl_bench.agents.ppo_lagrangian import PPOLagrangian, PPOLagrangianSettings
from icrl_bench.core.rollout import rollout
from icrl_bench.core.vector_env import make_vector_env
from icrl_bench.envs.half_cheetah_with_pos import HalfCheetahWithPos, true_cost

N_ENVS = 5
TOTAL_STEPS = 5_000_000
EVAL_EVERY = 250_000
N_EVAL_EPISODES = 10
FINE_EVAL_EPISODES = 5  # SB3 EvalCallback's default n_eval_episodes
OUT = Path("runs/validate_ppo_lagrangian_hc")
# Our train.csv column -> the name Malik's cpg.py logs it under (SB3 0.9.0
# logger, read from a synced run). Columns not listed keep "ours/<column>".
WANDB_NAMES = {
    "ep_rew_mean": "rollout/ep_rew_mean",
    "average_cost": "train/average_cost",
    "penalty": "train/nu",
    "policy_loss": "train/policy_gradient_loss",
    "reward_value_loss": "train/reward_value_loss",
    "cost_value_loss": "train/cost_value_loss",
    "entropy_loss": "train/entropy_loss",
    "approx_kl": "train/approx_kl",
    "clip_fraction": "train/clip_fraction",
    "eval_constrained_reward": "eval/mean_reward",
}


class RowWriter:
    """Appends dict rows to a CSV and flushes each one, so a crash loses
    nothing already written. The columns are the first row's keys."""

    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()


def evaluate(agent, steps, seed):
    """Reward in the constrained world, and reward per step, violations and
    lowest x in the nominal world."""
    constrained = rollout(HalfCheetahWithPos(constrained=True), agent.policy, N_EVAL_EPISODES, seed)
    nominal = rollout(HalfCheetahWithPos(), agent.policy, N_EVAL_EPISODES, seed)
    constrained_rewards = [sum(step.reward for step in t.steps) for t in constrained]
    nominal_steps = [step for t in nominal for step in t.steps]
    return {
        "steps": steps,
        "constrained_reward": float(np.mean(constrained_rewards)),
        "constrained_reward_std": float(np.std(constrained_rewards)),
        "constrained_length": float(np.mean([len(t.steps) for t in constrained])),
        "nominal_reward_per_step": float(np.mean([step.reward for step in nominal_steps])),
        "nominal_violation_rate": float(np.mean([true_cost(step.state, step.action) for step in nominal_steps])),
        "nominal_min_x": float(min(step.state[0] for step in nominal_steps)),
        "nominal_final_x": float(np.mean([t.steps[-1].state[0] for t in nominal])),
    }


def run(seed, wandb_project=None):
    torch.set_num_threads(1)
    out = OUT / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    train_csv, eval_csv = RowWriter(out / "train.csv"), RowWriter(out / "eval.csv")

    envs = make_vector_env(HalfCheetahWithPos, N_ENVS, true_cost)
    settings = PPOLagrangianSettings(target_kl=0.01, normalize_obs=True, normalize_reward=True, seed=seed)
    agent = PPOLagrangian(envs.single_observation_space, envs.single_action_space, settings)
    wandb_run = None
    if wandb_project is not None:
        import wandb  # only needed when logging to wandb
        wandb_run = wandb.init(project=wandb_project, group="ours", name=f"ours_s{seed}",
                               config={**asdict(settings), "n_envs": N_ENVS, "total_steps": TOTAL_STEPS})
    next_eval = EVAL_EVERY
    last_eval = None
    fine_eval_env = HalfCheetahWithPos(constrained=True)
    fine_eval_seed = 2000 + seed  # used on the first evaluation only

    def callback(row):
        nonlocal next_eval, last_eval, fine_eval_seed
        episodes = rollout(fine_eval_env, agent.policy, FINE_EVAL_EPISODES, fine_eval_seed)
        fine_eval_seed = None  # later evaluations continue the env's random state
        rewards = [sum(step.reward for step in t.steps) for t in episodes]
        row = {**row, "eval_constrained_reward": float(np.mean(rewards)),
               "eval_constrained_reward_std": float(np.std(rewards))}
        train_csv.write(row)
        if wandb_run is not None:
            # wandb's step = env steps, as in Malik's runs.
            wandb_run.log({WANDB_NAMES.get(k, f"ours/{k}"): v for k, v in row.items() if k != "steps"},
                          step=row["steps"])
        if row["steps"] >= next_eval:
            last_eval = evaluate(agent, row["steps"], seed=1000 + seed)
            eval_csv.write(last_eval)
            next_eval += EVAL_EVERY
            if wandb_run is not None:
                wandb_run.log({f"ours_eval/{k}": v for k, v in last_eval.items() if k != "steps"},
                              step=row["steps"])

    log = agent.train(envs, TOTAL_STEPS, callback=callback)
    # The final agent is evaluated once, at the step count it really reached
    # (training stops after the first rollout past TOTAL_STEPS).
    if last_eval is None or last_eval["steps"] != log[-1]["steps"]:
        last_eval = evaluate(agent, log[-1]["steps"], seed=1000 + seed)
        eval_csv.write(last_eval)
    if wandb_run is not None:
        wandb_run.finish()
    return seed, last_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--wandb-project")
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=len(args.seeds)) as pool:
        for seed, final in pool.map(partial(run, wandb_project=args.wandb_project), args.seeds):
            print(f"seed {seed}: constrained reward {final['constrained_reward']:.0f} "
                  f"(length {final['constrained_length']:.0f}), nominal reward/step "
                  f"{final['nominal_reward_per_step']:.2f}, violation rate "
                  f"{final['nominal_violation_rate']:.4f}, min x {final['nominal_min_x']:.2f}, "
                  f"final x {final['nominal_final_x']:.1f}")


if __name__ == "__main__":
    main()
