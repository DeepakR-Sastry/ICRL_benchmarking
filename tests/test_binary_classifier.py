import numpy as np
import pytest
import torch

from icrl_bench.methods.binary_classifier import BinaryClassifier, BinaryClassifierSettings

# LapGridWorld features [cell, is_forward, is_backward], all at cell 0.
FWD = [0.0, 1.0, 0.0]
BWD = [0.0, 0.0, 1.0]


def test_a_step_in_both_piles_is_partly_blamed():
    # The loss is mean BCE over each pile, added, so each pile counts equally
    # whatever its size. A step that is a share e of the expert pile and n of
    # the nominal pile gets ζ = e / (e + n). Forward: e = 3/3, n = 1/4, so
    # ζ = 0.8, cost 0.2: a step the expert takes is blamed only because the
    # agent took it too. Backward: e = 0, so cost 1.
    expert = [np.array([FWD, FWD, FWD], dtype=np.float32)]
    nominal = [np.array([FWD, BWD, BWD, BWD], dtype=np.float32)]
    method = BinaryClassifier(3, BinaryClassifierSettings(hidden_sizes=(20,), learning_rate=0.01,
                                                          backward_iters=2000))
    method.update(expert, nominal)
    cost_fwd, cost_bwd = method.cost(np.array([FWD, BWD], dtype=np.float32))
    assert cost_fwd == pytest.approx(0.2, abs=0.01)
    assert cost_bwd == pytest.approx(1.0, abs=0.01)


def test_metrics_are_the_bce_terms():
    # learning_rate 0: the network does not change, so the metrics can be
    # computed by hand from its predictions.
    expert = [np.array([FWD, FWD], dtype=np.float32)]
    nominal = [np.array([FWD, BWD], dtype=np.float32), np.array([BWD], dtype=np.float32)]
    method = BinaryClassifier(3, BinaryClassifierSettings(hidden_sizes=(20,), learning_rate=0.0, backward_iters=1))
    zeta_fwd, zeta_bwd = 1 - method.cost(np.array([FWD, BWD], dtype=np.float32))
    metrics = method.update(expert, nominal)
    expert_loss = -np.log(zeta_fwd)
    nominal_loss = -(np.log(1 - zeta_fwd) + 2 * np.log(1 - zeta_bwd)) / 3
    assert metrics["expert_loss"] == pytest.approx(expert_loss, rel=1e-5)
    assert metrics["nominal_loss"] == pytest.approx(nominal_loss, rel=1e-5)
    assert metrics["cn_loss"] == pytest.approx(expert_loss + nominal_loss, rel=1e-5)
    assert metrics["unweighted_nominal_loss"] == pytest.approx(
        (np.log(zeta_fwd + 1e-5) + 2 * np.log(zeta_bwd + 1e-5)) / 3, rel=1e-5)
    assert metrics["nominal_preds_mean"] == pytest.approx((zeta_fwd + 2 * zeta_bwd) / 3, rel=1e-5)


def test_every_update_takes_exactly_backward_iters_steps_with_an_annealed_learning_rate():
    expert = [np.array([FWD], dtype=np.float32)]
    nominal = [np.array([BWD], dtype=np.float32)]
    method = BinaryClassifier(3, BinaryClassifierSettings(learning_rate=0.1, anneal_lr_by_factor=0.5,
                                                          backward_iters=7))
    for k in range(3):
        method.update(expert, nominal)
        assert method.optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 0.5 ** k)
    first_weight = next(method.net.parameters())
    assert method.optimizer.state[first_weight]["step"] == 3 * 7  # Adam counts its steps


def test_input_columns_choose_what_the_network_sees():
    method = BinaryClassifier(3, BinaryClassifierSettings(input_columns=(0, 0, 1)))
    torch.testing.assert_close(method._inputs(np.array([[0.5, 1.0, 0.0]])), torch.tensor([[0.5, 0.5, 1.0]]))
    assert method.net[0].in_features == 3
