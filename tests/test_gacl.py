import numpy as np
import torch

from icrl_bench.methods.gacl import GACL, GACLSettings

# LapGridWorld features [cell, is_forward, is_backward].
EXPERT = [np.array([[0.0, 1, 0], [1.0, 1, 0], [2.0, 1, 0]])]
NOMINAL = [np.array([[0.0, 0, 1], [-1.0, 0, 1]]),
           np.array([[3.0, 1, 0], [4.0, 0, 1], [5.0, 0, 1]])]


def weights(method):
    return [p.detach().clone() for p in method.net.parameters()]


def test_update_uses_only_the_first_rows_of_the_bigger_pile():
    # 3 expert rows, 5 nominal rows: only the first 3 nominal rows count (quirk A).
    a, b = GACL(3, GACLSettings(hidden_sizes=(4,))), GACL(3, GACLSettings(hidden_sizes=(4,)))
    a.update(EXPERT, NOMINAL)
    b.update(EXPERT, [np.concatenate(NOMINAL)[:3]])
    assert all(torch.equal(x, y) for x, y in zip(weights(a), weights(b)))


def test_update_is_one_adam_step_with_a_constant_learning_rate():
    method = GACL(3, GACLSettings(hidden_sizes=(4,), learning_rate=0.01))
    before = weights(method)
    metrics = method.update(EXPERT, NOMINAL)
    assert not all(torch.equal(x, y) for x, y in zip(before, weights(method)))
    assert method.optimizer.state[method.net[0].weight]["step"] == 1
    method.update(EXPERT, NOMINAL)
    assert method.optimizer.param_groups[0]["lr"] == 0.01
    assert np.isclose(metrics["disc_loss"], metrics["expert_loss"] + metrics["nominal_loss"])


def test_repeated_updates_separate_expert_from_nominal():
    method = GACL(3, GACLSettings(hidden_sizes=(8,), learning_rate=0.01))
    for _ in range(300):
        metrics = method.update(EXPERT, NOMINAL)
    assert metrics["mean_expert_preds"] > 0.9 > 0.1 > metrics["mean_nominal_preds"]


def test_reward_is_log_d_plus_eps_and_cost_is_one_minus_d():
    method = GACL(3, GACLSettings(hidden_sizes=(4,)))
    features = np.concatenate(NOMINAL)
    d = 1 - method.cost(features)
    reward = method.reward(features)
    assert reward.shape == method.cost(features).shape == (5,)
    assert reward.dtype == np.float32
    assert np.allclose(reward, np.log(d + 1e-5))
    assert np.all(reward <= 0)


def test_input_columns_pick_the_network_input():
    # Malik's LapGridWorld input [cell, cell, is_forward] (bug (2)).
    method = GACL(3, GACLSettings(hidden_sizes=(4,), input_columns=(0, 0, 1)))
    assert method.net[0].in_features == 3
    a = np.array([[2.0, 1, 0]])
    b = np.array([[2.0, 1, 1]])  # differs only in the dropped column
    assert np.array_equal(method.reward(a), method.reward(b))
