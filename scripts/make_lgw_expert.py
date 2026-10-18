"""Records the LapGridWorld expert pile: the scripted always-forward policy
(see docs/decisions.md), 20 episodes as in Malik's README LapGridWorld run.

Run from anywhere: python scripts/make_lgw_expert.py"""

from datetime import date
from pathlib import Path

from icrl_bench.core.rollout import rollout
from icrl_bench.data.piles import save_pile
from icrl_bench.envs.lap_grid_world import FORWARD, LapGridWorld, true_cost

N_EPISODES = 20
SEED = 0
PATH = Path(__file__).resolve().parents[1] / "expert_data" / "lgw_forward.pkl"


def always_forward(obs):
    return FORWARD


def main():
    trajectories = rollout(LapGridWorld(), always_forward, N_EPISODES, SEED)
    rewards = [sum(step.reward for step in t.steps) for t in trajectories]
    costs = [sum(true_cost(step.state, step.action) for step in t.steps) for t in trajectories]

    metadata = {
        "env": "LapGridWorld(constrained=False)",
        "policy": "scripted: always forward",
        "n_episodes": N_EPISODES,
        "seed": SEED,
        "created": date.today().isoformat(),
        "episode_rewards": rewards,
        "episode_true_costs": costs,
    }
    save_pile(PATH, trajectories, metadata)

    print(f"Saved {len(trajectories)} trajectories to {PATH}")
    print(f"reward per episode: min {min(rewards)}, max {max(rewards)}")
    print(f"true cost per episode: min {min(costs)}, max {max(costs)}")


if __name__ == "__main__":
    main()
