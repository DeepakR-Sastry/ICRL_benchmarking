import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from icrl_bench.agents.ppo_lagrangian import PPOLagrangian, PPOLagrangianSettings
from icrl_bench.config import load_config, parse_config
from icrl_bench.core.loop import ICRLLoopSettings, run_icrl
from icrl_bench.core.vector_env import make_vector_env
from icrl_bench.data.piles import load_pile
from icrl_bench.envs.lap_grid_world import BACKWARD, FORWARD, LapGridWorld, true_cost
from icrl_bench.featurizers.lap_grid_world import featurize
from icrl_bench.methods.maximum_entropy import MaximumEntropy, MaximumEntropySettings
from icrl_bench.run import build, run

EXPERT = Path(__file__).resolve().parent.parent / "expert_data" / "lgw_forward.pkl"

# validate_icrl_lgw.py's settings, with a tiny agent and loop so it runs in seconds.
CONFIG = f"""
nominal_env: {{name: lap_grid_world}}
eval_env: {{name: lap_grid_world, constrained: true}}
true_cost: {{name: lap_grid_world}}
featurizer: {{name: lap_grid_world}}
expert: {EXPERT}
n_envs: 2
agent: {{name: ppo_lagrangian, target_kl: 0.01, n_steps: 32, batch_size: 16, n_epochs: 2}}
method: {{name: maximum_entropy, hidden_sizes: [20], learning_rate: 0.003,
         backward_iters: 20, input_columns: [0, 0, 1]}}
loop: {{n_rounds: 2, steps_per_round: 64, n_eval_episodes: 2}}
seed: 3
"""


def by_hand():
    """What validate_icrl_lgw.py builds, with CONFIG's settings."""
    expert, _ = load_pile(EXPERT)
    envs = make_vector_env(LapGridWorld, 2, None)
    agent = PPOLagrangian(envs.single_observation_space, envs.single_action_space,
                          PPOLagrangianSettings(target_kl=0.01, n_steps=32, batch_size=16, n_epochs=2, seed=3))
    method = MaximumEntropy(3, MaximumEntropySettings(hidden_sizes=(20,), learning_rate=0.003, backward_iters=20,
                                                      input_columns=(0, 0, 1), seed=3))
    return dict(agent=agent, envs=envs, method=method, featurize=featurize, expert=expert,
                make_nominal_env=LapGridWorld, make_eval_env=lambda: LapGridWorld(constrained=True),
                true_cost=true_cost)


def test_build_makes_the_envs_and_rule_the_config_names():
    built = build(parse_config(yaml.safe_load(CONFIG)))
    assert built["make_nominal_env"]().constrained is False
    assert built["make_eval_env"]().constrained is True
    assert built["envs"].num_envs == 2
    assert built["true_cost"](None, BACKWARD) == 1.0
    assert built["true_cost"](None, FORWARD) == 0.0


def test_a_run_built_from_the_config_matches_one_built_by_hand():
    config = parse_config(yaml.safe_load(CONFIG))
    from_config = run_icrl(**build(config), settings=config.loop)
    hand = run_icrl(**by_hand(), settings=ICRLLoopSettings(n_rounds=2, steps_per_round=64, n_eval_episodes=2,
                                                           seed=3))
    # assert_equal counts nan == nan (ep_rew_mean is nan: no episode ends in 64 steps).
    np.testing.assert_equal(from_config, hand)


def test_run_writes_the_run_folder_and_never_overwrites_it(tmp_path):
    config = parse_config(yaml.safe_load(CONFIG))
    log = run(config, tmp_path / "my_run")
    assert load_config(tmp_path / "my_run" / "config.yaml") == config
    with open(tmp_path / "my_run" / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["round"]) for row in rows] == [0, 1]
    assert float(rows[1]["true/cost"]) == log[1]["true/cost"]
    with pytest.raises(FileExistsError):
        run(config, tmp_path / "my_run")


def test_run_keeps_finished_rounds_when_a_later_round_crashes(tmp_path, monkeypatch):
    # The method's second update fails, as a crash in round 1 would.
    update = MaximumEntropy.update
    calls = []

    def update_then_crash(self, expert, nominal):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("crash in round 1")
        return update(self, expert, nominal)

    monkeypatch.setattr(MaximumEntropy, "update", update_then_crash)
    config = parse_config(yaml.safe_load(CONFIG))
    with pytest.raises(RuntimeError):
        run(config, tmp_path / "my_run")
    with open(tmp_path / "my_run" / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["round"]) for row in rows] == [0]
