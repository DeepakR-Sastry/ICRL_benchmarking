"""Validates our PPO (icrl_bench/agents/ppo.py) against stable-baselines3 on
CartPole-v1 (discrete actions) and Pendulum-v1 (continuous), same settings
(Malik's defaults), 3 seeds each. Matches = our curve inside SB3's spread.

Latest SB3 differs from Malik's 0.9.0 in a few places (docs/agents.md, "Known
differences"). Two are switched off here: time-limit bootstrapping (every env
reports a cutoff as a true end — ours already treats them the same) and the
KL early stop (target_kl unset). Left: the last-value bug (small).

Run from anywhere: python scripts/validate_ppo.py   (needs .[validate])
Options: --env Pendulum-v1 --seeds 10 --steps 2000000 (one env, more seeds or
steps). Writes runs/validate_ppo/<env>_<impl>_seed<k>.csv and prints a summary."""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from icrl_bench.agents.ppo import PPO, PPOSettings

ENVS = {"CartPole-v1": 200_000, "Pendulum-v1": 1_000_000}  # env -> total steps
SEEDS = [0, 1, 2]
N_ENVS = 5  # Malik's default
OUT = Path(__file__).resolve().parents[1] / "runs" / "validate_ppo"


class CutoffAsEnd(gym.Wrapper):
    """Reports a time-limit cutoff as a true end (terminated), so latest SB3
    doesn't bootstrap with V(final obs) — matching Malik's SB3 0.9.0."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, False, info


def make_env(env_id):
    return CutoffAsEnd(RecordEpisodeStatistics(gym.make(env_id)))


def run_ours(env_id, total_steps, seed):
    envs = SyncVectorEnv([lambda: make_env(env_id) for _ in range(N_ENVS)],
                         autoreset_mode=AutoresetMode.SAME_STEP)
    agent = PPO(envs.single_observation_space, envs.single_action_space, PPOSettings(seed=seed))
    return [(row["steps"], row["ep_rew_mean"]) for row in agent.train(envs, total_steps)]


def run_sb3(env_id, total_steps, seed):
    from stable_baselines3 import PPO as SB3PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env

    class LogRollouts(BaseCallback):
        """One row per rollout: steps so far and mean of the last 100 episodes."""

        def __init__(self):
            super().__init__()
            self.rows = []

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            returns = [info["r"] for info in self.model.ep_info_buffer]
            self.rows.append((self.num_timesteps, float(np.mean(returns)) if returns else float("nan")))

    # make_vec_env adds SB3's Monitor (records episode returns) inside our wrapper.
    envs = make_vec_env(env_id, n_envs=N_ENVS, seed=seed, wrapper_class=CutoffAsEnd)
    s = PPOSettings(seed=seed)
    model = SB3PPO(
        "MlpPolicy", envs, learning_rate=s.learning_rate, n_steps=s.n_steps,
        batch_size=s.batch_size, n_epochs=s.n_epochs, gamma=s.gamma,
        gae_lambda=s.gae_lambda, clip_range=s.clip_range, ent_coef=s.ent_coef,
        vf_coef=s.vf_coef, max_grad_norm=s.max_grad_norm, target_kl=s.target_kl,
        seed=seed, device=s.device,
    )
    callback = LogRollouts()
    model.learn(total_steps, callback=callback)
    return callback.rows


def run(job):
    env_id, impl, seed, total_steps = job
    torch.set_num_threads(1)  # one core per run; runs go in parallel
    rows = (run_ours if impl == "ours" else run_sb3)(env_id, total_steps, seed)
    with open(OUT / f"{env_id}_{impl}_seed{seed}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["steps", "ep_rew_mean"])
        writer.writerows(rows)
    return (env_id, impl, seed), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=list(ENVS), help="one env only (default: all)")
    parser.add_argument("--seeds", type=int, default=len(SEEDS), help="seeds 0..n-1")
    parser.add_argument("--steps", type=int, help="total steps (default: per env, ENVS)")
    args = parser.parse_args()
    envs = {args.env: ENVS[args.env]} if args.env else dict(ENVS)
    if args.steps:
        envs = {env_id: args.steps for env_id in envs}
    seeds = list(range(args.seeds))

    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(env_id, impl, seed, steps) for env_id, steps in envs.items()
            for impl in ["ours", "sb3"] for seed in seeds]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count())) as pool:
        results = dict(pool.map(run, jobs))

    for env_id in envs:
        print(f"\n{env_id}: ep_rew_mean, mean ± std over seeds {seeds}")
        print(f"{'steps':>9}  {'ours':>16}  {'sb3':>16}")
        n_rows = min(len(results[(env_id, impl, seed)]) for impl in ["ours", "sb3"] for seed in seeds)
        for i in np.linspace(0, n_rows - 1, 10).astype(int):
            line = f"{results[(env_id, 'ours', seeds[0])][i][0]:>9}"
            for impl in ["ours", "sb3"]:
                values = [results[(env_id, impl, seed)][i][1] for seed in seeds]
                line += f"  {np.mean(values):>8.1f} ± {np.std(values):>5.1f}"
            print(line)


if __name__ == "__main__":
    main()
