"""Reads an experiment's YAML config into a checked Config (docs/design.md,
"Conventions"). Only reads and checks; building the boxes is the run
script's job."""

from dataclasses import asdict, dataclass, fields, is_dataclass

import yaml

from icrl_bench.core.loop import ICRLLoopSettings
from icrl_bench.registry import AGENTS, ENVS, FEATURIZERS, METHODS, TRUE_COSTS, check_true_cost_fits

KEYS = {"nominal_env", "eval_env", "true_cost", "featurizer", "expert", "n_envs",
        "agent", "method", "loop", "seed"}


@dataclass(frozen=True)
class Named:
    """A box chosen by registry name. args: keyword arguments (envs, true
    costs, featurizers) or a settings dataclass (agents, methods)."""
    name: str
    args: object


@dataclass(frozen=True)
class Config:
    nominal_env: Named
    eval_env: Named
    true_cost: Named
    featurizer: Named
    expert: str  # path of the expert pile file
    n_envs: int
    agent: Named
    method: Named
    loop: ICRLLoopSettings
    seed: int


def load_config(path, seed=None):
    """seed, if given, replaces the file's seed before checking, so one
    file serves every seed of an experiment."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if seed is not None:
        raw["seed"] = seed
    return parse_config(raw)


def parse_config(raw):
    """raw: the dict read from YAML. Raises ValueError on any mistake."""
    if set(raw) != KEYS:
        raise ValueError(f"config keys: missing {sorted(KEYS - set(raw))}, unknown {sorted(set(raw) - KEYS)}")
    seed = raw["seed"]
    config = Config(
        nominal_env=_named(raw["nominal_env"], ENVS, "nominal_env"),
        eval_env=_named(raw["eval_env"], ENVS, "eval_env"),
        true_cost=_named(raw["true_cost"], TRUE_COSTS, "true_cost"),
        featurizer=_named(raw["featurizer"], FEATURIZERS, "featurizer"),
        expert=raw["expert"],
        n_envs=raw["n_envs"],
        agent=_named(raw["agent"], AGENTS, "agent", seed),
        method=_named(raw["method"], METHODS, "method", seed),
        loop=_settings(ICRLLoopSettings, raw["loop"], seed, "loop"),
        seed=seed,
    )
    if config.featurizer.args:
        raise ValueError("featurizer: plain-function featurizers take no arguments")
    for env in (config.nominal_env, config.eval_env):
        check_true_cost_fits(config.true_cost.name, env.name)
    return config


def save_config(config, path):
    """Write the config with every setting filled in (defaults too), in the
    format load_config reads, so the file alone reproduces the run."""
    def block(settings):
        values = asdict(settings) if is_dataclass(settings) else dict(settings)
        values.pop("seed", None)  # written once, at the top
        return {k: list(v) if isinstance(v, tuple) else v for k, v in values.items()}

    raw = {f.name: getattr(config, f.name) for f in fields(config)}  # Config's order
    for key in ("nominal_env", "eval_env", "true_cost", "featurizer", "agent", "method"):
        raw[key] = {"name": raw[key].name, **block(raw[key].args)}
    raw["loop"] = block(raw["loop"])
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)


def _named(block, registry, where, seed=None):
    """{name: ..., rest} -> Named. With a seed, the registry holds (class,
    settings dataclass) and the rest becomes that dataclass."""
    block = dict(block)
    name = block.pop("name", None)
    if name not in registry:
        raise ValueError(f"{where}: name {name!r} is not one of {sorted(registry)}")
    if seed is None:
        _check_no_numbers_or_booleans_as_text(block, where)
        return Named(name, block)
    _, settings_cls = registry[name]
    return Named(name, _settings(settings_cls, block, seed, where))


def _settings(settings_cls, block, seed, where):
    """Fill a settings dataclass from a config block; the config's one seed
    is added here."""
    _check_no_numbers_or_booleans_as_text(block, where)
    known = {f.name for f in fields(settings_cls)}
    for key in block:
        if key not in known:
            raise ValueError(f"{where}: unknown setting {key!r}")
        if key == "seed":
            raise ValueError(f"{where}: set the seed once, at the top of the config")
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in block.items()}
    return settings_cls(**values, seed=seed)


# YAML reads these unquoted as booleans; quoted, they are text, and any
# non-empty text counts as True in Python ("false" would mean on).
BOOLEAN_WORDS = {"true", "false", "yes", "no", "on", "off"}


def _check_no_numbers_or_booleans_as_text(block, where):
    """PyYAML reads 1e4 and 1.0e4 as text, not numbers (it needs 1.0e+4),
    and "false" in quotes as text, not a boolean; refuse any text that is
    really a number or a boolean."""
    for key, value in block.items():
        if isinstance(value, str):
            if value.lower() in BOOLEAN_WORDS:
                raise ValueError(f"{where}: {key} = {value!r} is text; write booleans unquoted: true or false")
            try:
                float(value)
            except ValueError:
                continue
            raise ValueError(f"{where}: {key} = {value!r} is text; write numbers like 1.0e+4 or 10000")
