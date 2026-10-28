"""Plots scripts/validate_ppo_lagrangian_hc.py's results next to Malik's
HalfCheetah expert, laid out like his Figure 3b (arXiv 2011.09999, page 6):
left, reward per episode in the constrained env; right, violations per step
in the nominal env. Ours: mean over seeds, shaded ± standard error (as in
the paper). Works on a run still in progress (uses the checkpoints every
seed has reached).

Run from the repo root: python scripts/plot_ppo_lagrangian_hc.py
Writes runs/validate_ppo_lagrangian_hc/plot.png."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

RUN_DIR = Path("runs/validate_ppo_lagrangian_hc")
# Malik's Figure 3b, expert line, read off the plot (docs/results.md).
EXPERT_REWARD = 3800
EXPERT_VIOLATIONS = 0.0

OURS, EXPERT = "#eb6834", "#2a78d6"  # orange; blue as in the paper's expert line
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_seeds():
    """{seed name: {column: array}} from each seed's eval.csv."""
    seeds = {}
    for path in sorted(RUN_DIR.glob("seed*/eval.csv")):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            seeds[path.parent.name] = {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}
    return seeds


def mean_and_stderr(seeds, column):
    """Steps every seed has reached, the mean over seeds there, and the
    standard error (std / sqrt(number of seeds))."""
    n = min(len(s["steps"]) for s in seeds.values())
    values = np.stack([s[column][:n] for s in seeds.values()])
    steps = next(iter(seeds.values()))["steps"][:n]
    return steps, values.mean(axis=0), values.std(axis=0) / np.sqrt(len(seeds))


def panel(ax, seeds, column, expert_value, title, ylabel):
    steps, mean, stderr = mean_and_stderr(seeds, column)
    ax.fill_between(steps, mean - stderr, mean + stderr, color=OURS, alpha=0.2, linewidth=0)
    ax.plot(steps, mean, color=OURS, linewidth=2, label=f"ours (PPO-Lag + true cost, {len(seeds)} seeds)")
    ax.axhline(expert_value, color=EXPERT, linewidth=2, linestyle="--", label="Malik's expert (Fig. 3b)")
    # Direct labels at the right end, in text ink; the line beside each carries the colour.
    label = f"{mean[-1]:,.0f}" if abs(mean[-1]) >= 10 else f"{mean[-1]:.3f}"
    ax.annotate(f"ours {label}", (steps[-1], mean[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_xlabel("environment steps", color=MUTED)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))
    ax.set_xlim(left=0)  # from 0, as in the paper
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def main():
    seeds = load_seeds()
    if not seeds:
        raise SystemExit(f"no eval.csv rows under {RUN_DIR} yet")
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    panel(left, seeds, "constrained_reward", EXPERT_REWARD,
          "Reward (higher is better)", "reward per episode, constrained env")
    panel(right, seeds, "nominal_violation_rate", EXPERT_VIOLATIONS,
          "Constraint violations (lower is better)", "violations per step, nominal env")
    left.set_ylim(bottom=0)
    right.set_ylim(bottom=-0.02)
    # One legend above both panels, so it never covers a curve.
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False,
               labelcolor=INK, fontsize=9)
    out = RUN_DIR / "plot.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
