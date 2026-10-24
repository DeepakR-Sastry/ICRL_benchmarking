"""Name -> object, so a config can name each box (docs/design.md,
"Conventions"). A new box is one new file plus one entry here."""

from icrl_bench.agents.ppo import PPO, PPOSettings
from icrl_bench.agents.ppo_lagrangian import PPOLagrangian, PPOLagrangianSettings
from icrl_bench.envs import lap_grid_world
from icrl_bench.featurizers import lap_grid_world as lap_grid_world_features
from icrl_bench.methods.binary_classifier import BinaryClassifier, BinaryClassifierSettings
from icrl_bench.methods.maximum_entropy import MaximumEntropy, MaximumEntropySettings

# name -> gymnasium env class; config arguments go to its constructor.
ENVS = {
    "lap_grid_world": lap_grid_world.LapGridWorld,
}

# name -> (true_cost(obs, action) -> float, names of the envs it applies to).
# A rule assumes what the observation holds, so it lists the envs that
# meet its assumptions (docs/decisions.md, 2026-10-21).
TRUE_COSTS = {
    "lap_grid_world": (lap_grid_world.true_cost, {"lap_grid_world"}),
}

# name -> featurize(trajectory) -> array (steps, features).
FEATURIZERS = {
    "lap_grid_world": lap_grid_world_features.featurize,
}

# name -> (class, settings dataclass).
AGENTS = {
    "ppo": (PPO, PPOSettings),
    "ppo_lagrangian": (PPOLagrangian, PPOLagrangianSettings),
}

METHODS = {
    "maximum_entropy": (MaximumEntropy, MaximumEntropySettings),
    "binary_classifier": (BinaryClassifier, BinaryClassifierSettings),
}


def check_true_cost_fits(true_cost_name, env_name):
    """Raise if the rule is not listed for this env: on the wrong env a rule
    can return meaningless numbers without any error."""
    _, env_names = TRUE_COSTS[true_cost_name]
    if env_name not in env_names:
        raise ValueError(f"true cost {true_cost_name!r} does not apply to env {env_name!r} "
                         f"(it applies to: {sorted(env_names)})")
