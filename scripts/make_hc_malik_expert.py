"""Converts Malik's shipped HCWithPos expert rollouts into one of our pile
files, so configs point at it like any other pile. 10 rollouts, as his
README HalfCheetah run (-er 10). Needs his repo at ~/refs/malik_icrl.

Run from anywhere: python scripts/make_hc_malik_expert.py"""

from datetime import date
from pathlib import Path

from icrl_bench.data.malik import load_malik_rollouts
from icrl_bench.data.piles import save_pile
from icrl_bench.envs.half_cheetah_with_pos import true_cost

N_ROLLOUTS = 10
SOURCE = Path.home() / "refs/malik_icrl/icrl/expert_data/HCWithPos-New"  # his -ep
COMMIT = "f28214e"
PATH = Path(__file__).resolve().parents[1] / "expert_data" / "hc_malik.pkl"


def main():
    trajectories = load_malik_rollouts(SOURCE, N_ROLLOUTS)
    costs = [sum(true_cost(step.state, step.action) for step in t.steps) for t in trajectories]

    metadata = {
        "env": "HCWithPos-v0 (Malik's, MuJoCo 2)",
        "policy": "Malik's shipped expert rollouts",
        "source": f"{SOURCE}/files/EXPERT/rollouts/0..{N_ROLLOUTS - 1}.pkl",
        "commit": COMMIT,
        "n_episodes": N_ROLLOUTS,
        "pairing": "Step(his obs row t-1, his action t), t = 1..499; reward None",
        "created": date.today().isoformat(),
        "episode_true_costs": costs,
    }
    save_pile(PATH, trajectories, metadata)

    print(f"Saved {len(trajectories)} trajectories to {PATH}")
    print(f"steps per episode: {sorted({len(t.steps) for t in trajectories})}")
    print(f"true cost per episode: min {min(costs)}, max {max(costs)}")


if __name__ == "__main__":
    main()
