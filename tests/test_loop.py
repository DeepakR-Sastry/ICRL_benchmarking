import numpy as np
import pytest

from icrl_bench.core.loop import GACLLoopSettings, ICRLLoopSettings, learned_cost_fn, run_gacl, run_icrl
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


class GACLRecorder:
    """Stand-ins for the agent, the method and the vector env in run_gacl.
    The agent's rollouts have obs[t, e] = 10t + e (so the order is visible)
    and actions forward in env 0, backward in env 1; it hands each to the
    reward hook, then reports the update's row to the callback."""

    def __init__(self, n_steps, n_envs):
        self.n_steps, self.num_envs = n_steps, n_envs
        self.nominal, self.extra_rewards = [], []

    # agent
    def train(self, envs, total_steps, callback, reward_hook):
        t, e = np.meshgrid(np.arange(self.n_steps), np.arange(self.num_envs), indexing="ij")
        obs = (10.0 * t + e)[..., None]
        acs = np.where(e == 0, FORWARD, BACKWARD)
        steps = 0
        while steps < total_steps:
            self.extra_rewards.append(reward_hook(obs, acs))
            steps += self.n_steps * self.num_envs
            callback({"steps": steps, "ep_rew_mean": 0.0})

    def policy(self, obs):
        return FORWARD

    # method
    def update(self, expert, nominal):
        self.nominal.append(nominal)
        return {"disc_loss": 0.7}

    def reward(self, features):
        return features[:, 0]  # the cell: shows which step each reward goes to

    # vector env
    def call(self, name, cost_fn):
        assert name == "set_cost_fn" and cost_fn(None, BACKWARD) == 0.0


def test_run_gacl_flattens_step_major_and_returns_rewards_in_rollout_shape():
    expert = rollout(LapGridWorld(), always_forward, 3, seed=0)
    stub = GACLRecorder(n_steps=3, n_envs=2)
    run_gacl(stub, stub, stub, featurize, expert, lambda: LapGridWorld(constrained=True),
             true_cost, GACLLoopSettings(total_steps=6, n_eval_episodes=1))
    [nominal] = stub.nominal[0]  # one array: the whole rollout
    # (step 0, env 0), (step 0, env 1), (step 1, env 0), ... as Malik's flatten.
    assert nominal[:, 0].tolist() == [0, 1, 10, 11, 20, 21]
    assert nominal[:, 1].tolist() == [1, 0, 1, 0, 1, 0]  # is_forward: env 0 only
    # Each step's reward goes back to its own (step, env) slot.
    assert stub.extra_rewards[0].tolist() == [[0, 1], [10, 11], [20, 21]]


def test_run_gacl_evaluates_when_malik_would_and_logs_each_rollout():
    # 100 vector steps per rollout, eval every 128: after update k the next
    # rollout covers (100k, 100k + 100]. It holds a multiple of 128 for
    # k = 1, 2, 3, 5 but not 4; after the last update (k = 6) training ends.
    expert = rollout(LapGridWorld(), always_forward, 3, seed=0)
    stub = GACLRecorder(n_steps=100, n_envs=2)
    rows = []
    log = run_gacl(stub, stub, stub, featurize, expert, lambda: LapGridWorld(constrained=True),
                   true_cost, GACLLoopSettings(total_steps=1200, eval_every=128, n_eval_episodes=2),
                   callback=rows.append)
    assert rows == log
    assert [row["rollout"] for row in log] == [0, 1, 2, 3, 4, 5]
    assert [row["timesteps"] for row in log] == [200, 400, 600, 800, 1000, 1200]
    evaluated = [bool(np.isfinite(row["true/reward"])) for row in log]
    assert evaluated == [True, True, True, False, True, False]
    # Always forward in the constrained world: 60 every episode.
    assert log[0]["true/reward"] == 60.0 and log[0]["true/reward_std"] == 0.0
    # True cost of the training rollout: env 1 always steps backward.
    assert log[0]["true/cost"] == 0.5
    assert log[0]["backward/disc_loss"] == 0.7 and log[0]["forward/ep_rew_mean"] == 0.0
