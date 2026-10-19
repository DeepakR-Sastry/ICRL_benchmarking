import numpy as np
import pytest

from icrl_bench.core.loop import ICRLLoopSettings, learned_cost_fn, run_icrl
from icrl_bench.core.rollout import rollout
from icrl_bench.core.types import Step, Trajectory
from icrl_bench.envs.lap_grid_world import BACKWARD, FORWARD, LapGridWorld, true_cost
from icrl_bench.featurizers.lap_grid_world import featurize
from icrl_bench.methods.maximum_entropy import MaximumEntropy, MaximumEntropySettings


def always_forward(obs):
    return FORWARD


def test_learned_cost_fn_scores_the_step_it_is_given():
    method = MaximumEntropy(3, MaximumEntropySettings(hidden_sizes=(20,)))
    cost_fn = learned_cost_fn(featurize, method)
    obs = np.array([-0.9], dtype=np.float32)
    for action in (FORWARD, BACKWARD):
        direct = method.cost(featurize(Trajectory([Step(obs, action, 0.0)])))[0]
        assert cost_fn(obs, action) == pytest.approx(direct)


def test_learned_cost_fn_follows_the_method_after_an_update():
    method = MaximumEntropy(3, MaximumEntropySettings(hidden_sizes=(20,), learning_rate=0.01, backward_iters=20))
    cost_fn = learned_cost_fn(featurize, method)  # built before the update
    obs = np.array([-0.9], dtype=np.float32)
    before = cost_fn(obs, BACKWARD)
    forward_row = np.array([[-0.9, 1.0, 0.0]], dtype=np.float32)
    backward_row = np.array([[-0.9, 0.0, 1.0]], dtype=np.float32)
    method.update([forward_row], [backward_row])
    assert cost_fn(obs, BACKWARD) != before
    assert cost_fn(obs, BACKWARD) == pytest.approx(method.cost(backward_row)[0])


class Recorder:
    """Stand-ins for the agent, the method and the vector env: they record
    what the loop asks of them, in order, and do no learning."""

    def __init__(self):
        self.events = []
        self.nominal_sizes = []

    # agent
    def train(self, envs, total_steps):
        self.events.append("train")
        return [{"steps": total_steps, "penalty": 1.0}]

    def policy(self, obs):
        return FORWARD

    # method
    def update(self, expert, nominal):
        self.events.append("update")
        self.nominal_sizes.append(len(nominal))
        return {"cn_loss": 0.5}

    def cost(self, features):
        return np.zeros(len(features))

    # vector env
    def call(self, name, cost_fn):
        assert name == "set_cost_fn"
        self.events.append("set cost")


def test_run_icrl_follows_maliks_order_and_logs_each_round():
    expert = rollout(LapGridWorld(), always_forward, 3, seed=0)
    stub = Recorder()
    log = run_icrl(stub, stub, stub, featurize, expert, LapGridWorld,
                   lambda: LapGridWorld(constrained=True), true_cost,
                   ICRLLoopSettings(n_rounds=2, steps_per_round=100, n_eval_episodes=2))

    assert stub.events == ["set cost", "train", "update", "set cost", "train", "update", "set cost"]
    assert stub.nominal_sizes == [3, 3]  # as many nominal trajectories as expert ones
    assert [row["round"] for row in log] == [0, 1]
    assert [row["timesteps"] for row in log] == [100, 200]
    # An always-forward agent: laps in the constrained world, no violations.
    assert log[0]["true/reward"] == 60.0 and log[0]["true/reward_std"] == 0.0
    assert log[0]["true/cost"] == 0.0
    assert log[0]["forward/penalty"] == 1.0 and log[0]["backward/cn_loss"] == 0.5
