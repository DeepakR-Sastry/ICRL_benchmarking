"""Our PPO on nominal LapGridWorld (no constraint) with Malik's LapGridWorld
settings (defaults + target_kl 0.01, no normalisation). Expected: it finds
coin-bouncing, reward well above the 60 of running laps. Old repo, plain SB3
2.7.0, seed 0, deterministic eval: 258 at 200k steps, 282 at 1M; best
possible 294 (docs/results.md).

Run from anywhere: python scripts/validate_ppo_lgw.py
Evaluation: 20 episodes per checkpoint with the sampling policy (as Malik
samples trajectories); backward steps counted with true_cost (grading only)."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from icrl_bench.agents.ppo import PPO, PPOSettings
from icrl_bench.core.rollout import rollout
from icrl_bench.envs.lap_grid_world import LapGridWorld, true_cost

CHECKPOINTS = [200_000, 1_000_000]  # total training steps at each evaluation
SEEDS = [0, 1, 2]
N_ENVS = 5
N_EVAL_EPISODES = 20


def run(seed):
    torch.set_num_threads(1)
    envs = SyncVectorEnv([lambda: RecordEpisodeStatistics(LapGridWorld()) for _ in range(N_ENVS)],
                         autoreset_mode=AutoresetMode.SAME_STEP)
    agent = PPO(envs.single_observation_space, envs.single_action_space,
                PPOSettings(target_kl=0.01, seed=seed))
    results, trained = [], 0
    for checkpoint in CHECKPOINTS:
        agent.train(envs, checkpoint - trained)
        trained = checkpoint
        trajectories = rollout(LapGridWorld(), agent.policy, N_EVAL_EPISODES, seed=1000 + seed)
        rewards = [sum(step.reward for step in t.steps) for t in trajectories]
        backward = [sum(true_cost(step.state, step.action) for step in t.steps) for t in trajectories]
        results.append((checkpoint, np.mean(rewards), np.min(rewards), np.max(rewards), np.mean(backward)))
    return seed, results


def main():
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        for seed, results in pool.map(run, SEEDS):
            for checkpoint, mean, low, high, backward in results:
                print(f"seed {seed}, {checkpoint:>9} steps: reward {mean:6.1f} "
                      f"(min {low:.0f}, max {high:.0f}), backward steps/episode {backward:5.1f}")


if __name__ == "__main__":
    main()
