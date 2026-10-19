"""Our PPO-Lagrangian on LapGridWorld with the TRUE cost (never move
backward), budget 0. Expected: it runs laps — reward ≈ 60, ~0 backward steps
— like Malik's expert, instead of the coin-bouncing plain PPO finds (255–268,
docs/results.md).

Settings as Malik's ICRL forward step on LapGridWorld (README: -tk 0.01
-dno -dnr -dnc): defaults + target_kl 0.01, no normalisation, 5 envs. (His
expert was trained by an older script with normalisation on — see
docs/agents.md.)

Run from anywhere: python scripts/validate_ppo_lagrangian_lgw.py
Evaluation: 20 episodes per checkpoint with the sampling policy on the
nominal env; backward steps counted with true_cost."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from icrl_bench.agents.ppo_lagrangian import PPOLagrangian, PPOLagrangianSettings
from icrl_bench.core.cost_wrapper import CostWrapper
from icrl_bench.core.rollout import rollout
from icrl_bench.envs.lap_grid_world import LapGridWorld, true_cost

CHECKPOINTS = [200_000, 1_000_000]  # total training steps at each evaluation
SEEDS = [0, 1, 2]
N_ENVS = 5
N_EVAL_EPISODES = 20


def run(seed):
    torch.set_num_threads(1)
    envs = SyncVectorEnv(
        [lambda: CostWrapper(RecordEpisodeStatistics(LapGridWorld()), true_cost) for _ in range(N_ENVS)],
        autoreset_mode=AutoresetMode.SAME_STEP)
    agent = PPOLagrangian(envs.single_observation_space, envs.single_action_space,
                          PPOLagrangianSettings(target_kl=0.01, seed=seed))
    results, trained = [], 0
    for checkpoint in CHECKPOINTS:
        log = agent.train(envs, checkpoint - trained)
        trained = checkpoint
        trajectories = rollout(LapGridWorld(), agent.policy, N_EVAL_EPISODES, seed=1000 + seed)
        rewards = [sum(step.reward for step in t.steps) for t in trajectories]
        backward = [sum(true_cost(step.state, step.action) for step in t.steps) for t in trajectories]
        results.append((checkpoint, np.mean(rewards), np.min(rewards), np.max(rewards),
                        np.mean(backward), log[-1]["average_cost"], agent.lagrange.value))
    return seed, results


def main():
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        for seed, results in pool.map(run, SEEDS):
            for checkpoint, mean, low, high, backward, train_cost, penalty in results:
                print(f"seed {seed}, {checkpoint:>9} steps: reward {mean:6.1f} "
                      f"(min {low:.0f}, max {high:.0f}), backward steps/episode {backward:5.1f} | "
                      f"last rollout avg cost {train_cost:.4f}, λ {penalty:.2f}")


if __name__ == "__main__":
    main()
