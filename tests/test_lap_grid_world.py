import numpy as np
import pytest

from icrl_bench.envs.lap_grid_world import BACKWARD, FORWARD, LapGridWorld, true_cost


def cell_of(obs):
    """Undo the env's scaling: -1.0..0.95 back to cells 0..39."""
    return round((obs[0] + 1) * 20)


def always_forward(obs):
    return FORWARD


def bounce_on_first_coin(obs):
    """Walk to the coin at cell 5, then step back and forth onto it."""
    return BACKWARD if cell_of(obs) == 5 else FORWARD


def play(policy, constrained):
    """Run one episode; return (total reward, total true cost, steps, terminated)."""
    env = LapGridWorld(constrained=constrained)
    obs, _ = env.reset(seed=0)
    total_reward = total_cost = steps = 0
    done = False
    while not done:
        action = policy(obs)
        total_cost += true_cost(obs, action)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    return total_reward, total_cost, steps, terminated


# Hand-play checks from docs/envs.md.
@pytest.mark.parametrize(
    "policy, constrained, expected",
    [
        (always_forward, False, (60, 0, 200, False)),
        (always_forward, True, (60, 0, 200, False)),
        (bounce_on_first_coin, False, (294, 98, 200, False)),
        (bounce_on_first_coin, True, (2, 1, 6, True)),
    ],
)
def test_hand_play(policy, constrained, expected):
    assert play(policy, constrained) == expected


def test_wraps_around_both_ways():
    env = LapGridWorld()
    env.reset()
    env.step(BACKWARD)
    assert env.cell == 39
    env.step(FORWARD)
    assert env.cell == 0


def test_observation_scaling_matches_malik():
    env = LapGridWorld()
    obs, _ = env.reset()
    assert obs.dtype == np.float32 and obs.shape == (1,)
    assert obs[0] == -1.0
    obs, *_ = env.step(BACKWARD)  # to cell 39
    assert obs[0] == pytest.approx(0.95)


def test_constrained_backward_stays_put():
    env = LapGridWorld(constrained=True)
    env.reset()
    env.step(FORWARD)
    obs, reward, terminated, truncated, _ = env.step(BACKWARD)
    assert env.cell == 1 and reward == -1 and terminated and not truncated


def test_true_cost():
    assert true_cost(np.array([0.0]), BACKWARD) == 1.0
    assert true_cost(np.array([0.0]), FORWARD) == 0.0
