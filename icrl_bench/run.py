"""From a checked Config to a finished run: build every box, then run the
ICRL loop (docs/design.md, "One entry point")."""

import csv
from functools import partial
from pathlib import Path

from icrl_bench.config import save_config
from icrl_bench.core.loop import run_icrl
from icrl_bench.core.vector_env import make_vector_env
from icrl_bench.data.piles import load_pile
from icrl_bench.registry import AGENTS, ENVS, FEATURIZERS, METHODS, TRUE_COSTS


def build(config):
    """Config -> the keyword arguments of run_icrl except settings.

    Builds in the order scripts/validate_icrl_lgw.py does (expert, envs,
    agent, method): each constructor reseeds torch, so the order changes
    the numbers. The expert path is relative to the working directory."""
    def make_nominal_env():
        return ENVS[config.nominal_env.name](**config.nominal_env.args)

    def make_eval_env():
        return ENVS[config.eval_env.name](**config.eval_env.args)

    # A rule's config arguments are fixed ahead of time, as Malik's
    # partial(wall_behind, -3).
    true_cost_fn, _ = TRUE_COSTS[config.true_cost.name]
    true_cost = partial(true_cost_fn, **config.true_cost.args)
    featurize = FEATURIZERS[config.featurizer.name]

    expert, _ = load_pile(config.expert)
    # No cost yet: run_icrl sets the learned cost before the first step.
    envs = make_vector_env(make_nominal_env, config.n_envs, None)
    agent_cls, _ = AGENTS[config.agent.name]
    agent = agent_cls(envs.single_observation_space, envs.single_action_space, config.agent.args)
    method_cls, _ = METHODS[config.method.name]
    n_features = featurize(expert[0]).shape[1]
    method = method_cls(n_features, config.method.args)

    return dict(agent=agent, envs=envs, method=method, featurize=featurize, expert=expert,
                make_nominal_env=make_nominal_env, make_eval_env=make_eval_env,
                true_cost=true_cost)


def run(config, run_dir):
    """Build every box, run the ICRL loop, and write the run folder:
    config.yaml (every setting, defaults filled in) and metrics.csv (one
    row per round). Refuses a folder that already holds a run, so results
    are never overwritten. Returns the per-round log."""
    run_dir = Path(run_dir)
    if (run_dir / "config.yaml").exists():
        raise FileExistsError(f"{run_dir} already holds a run")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Written first, so even a crashed run records what it was.
    save_config(config, run_dir / "config.yaml")

    log = run_icrl(**build(config), settings=config.loop)

    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0]))
        writer.writeheader()
        writer.writerows(log)
    return log
