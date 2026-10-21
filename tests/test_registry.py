import pytest

from icrl_bench.registry import AGENTS, ENVS, METHODS, TRUE_COSTS, check_true_cost_fits


def test_every_rule_lists_only_registered_envs():
    for name, (_, env_names) in TRUE_COSTS.items():
        assert env_names <= set(ENVS), name


def test_a_listed_pairing_passes():
    check_true_cost_fits("lap_grid_world", "lap_grid_world")


def test_an_unlisted_pairing_is_refused():
    with pytest.raises(ValueError, match="does not apply"):
        check_true_cost_fits("lap_grid_world", "other_env")


def test_every_agent_and_method_builds_from_its_default_settings():
    env = ENVS["lap_grid_world"]()
    for cls, settings_cls in AGENTS.values():
        cls(env.observation_space, env.action_space, settings_cls())
    for cls, settings_cls in METHODS.values():
        cls(3, settings_cls())
