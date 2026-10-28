import pytest
import yaml

from icrl_bench.agents.ppo_lagrangian import PPOLagrangianSettings
from icrl_bench.config import load_config, parse_config, save_config
from icrl_bench.core.loop import ICRLLoopSettings
from icrl_bench.methods.maximum_entropy import MaximumEntropySettings

TEXT = """
nominal_env: {name: lap_grid_world}
eval_env: {name: lap_grid_world, constrained: true}
true_cost: {name: lap_grid_world}
featurizer: {name: lap_grid_world}
expert: expert_data/lgw_forward.pkl
n_envs: 5
agent: {name: ppo_lagrangian, target_kl: 0.01}
method: {name: maximum_entropy, hidden_sizes: [20], learning_rate: 0.003,
         backward_iters: 20, input_columns: [0, 0, 1]}
loop: {n_rounds: 12, steps_per_round: 10000}
seed: 3
"""


def raw(**changes):
    """The config above as a dict, with some top-level entries replaced."""
    return {**yaml.safe_load(TEXT), **changes}


def test_reads_the_settings_validate_icrl_lgw_uses(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(TEXT)
    config = load_config(path)
    assert config.eval_env.name == "lap_grid_world"
    assert config.eval_env.args == {"constrained": True}
    assert config.agent.args == PPOLagrangianSettings(target_kl=0.01, seed=3)
    # Lists become tuples, as the dataclass fields are typed.
    assert config.method.args == MaximumEntropySettings(
        hidden_sizes=(20,), learning_rate=0.003, backward_iters=20, input_columns=(0, 0, 1), seed=3)
    assert config.loop == ICRLLoopSettings(n_rounds=12, steps_per_round=10_000, seed=3)


@pytest.mark.parametrize("changes, message", [
    (dict(extra=1), "unknown \\['extra'\\]"),
    (dict(method={"name": "maximum_entropi"}), "is not one of"),
    (dict(method={"name": "maximum_entropy", "hidden_size": [20]}), "unknown setting 'hidden_size'"),
    (dict(agent={"name": "ppo_lagrangian", "seed": 1}), "set the seed once"),
    (dict(loop={"steps_per_round": "1e4"}), "is text"),
    (dict(eval_env={"name": "lap_grid_world", "max_episode_steps": "1.0e4"}), "is text"),
    (dict(method={"name": "maximum_entropy", "importance_sampling": "false"}), "is text"),
    (dict(nominal_env={"name": "lap_grid_world", "constrained": "False"}), "is text"),
    (dict(featurizer={"name": "lap_grid_world", "scale": 2}), "take no arguments"),
])
def test_mistakes_are_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        parse_config(raw(**changes))


def test_a_true_cost_that_does_not_fit_an_env_is_refused(monkeypatch):
    from icrl_bench import registry
    monkeypatch.setitem(registry.TRUE_COSTS, "wall_behind", (None, {"some_mujoco_env"}))
    with pytest.raises(ValueError, match="does not apply"):
        parse_config(raw(true_cost={"name": "wall_behind"}))


def test_a_saved_config_reads_back_identical(tmp_path):
    config = parse_config(raw())
    save_config(config, tmp_path / "saved.yaml")
    assert load_config(tmp_path / "saved.yaml") == config
    # Defaults are written out, not just what the file set.
    saved = yaml.safe_load((tmp_path / "saved.yaml").read_text())
    assert saved["method"]["reg_coeff"] == MaximumEntropySettings().reg_coeff
    assert "seed" not in saved["agent"]


def test_a_seed_override_reaches_every_settings_block(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(TEXT)
    config = load_config(path, seed=7)
    assert config.seed == config.agent.args.seed == config.method.args.seed == config.loop.seed == 7
