import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from icrl_bench.agents.ppo_lagrangian import (
    ActorTwoCritics, LagrangeMultiplier, PPOLagrangian, PPOLagrangianSettings, collect_rollout,
    compute_gae, ppo_update, softplus_inverse,
)
from icrl_bench.core.cost_wrapper import CostWrapper
from icrl_bench.envs.lap_grid_world import LapGridWorld, true_cost


def lgw_net():
    return ActorTwoCritics(spaces.Box(-1.0, 1.0, shape=(1,)), spaces.Discrete(2))


def test_two_critics_one_number_per_observation():
    net = lgw_net()
    obs = torch.tensor([[-1.0], [0.0], [0.5]])
    assert net.distribution(obs).probs.shape == (3, 2)
    assert net.reward_value(obs).shape == (3,)
    assert net.cost_value(obs).shape == (3,)


def test_critics_share_no_weights():
    net = lgw_net()
    reward_params = {id(p) for p in [*net.reward_value_body.parameters(), *net.reward_value_head.parameters()]}
    cost_params = {id(p) for p in [*net.cost_value_body.parameters(), *net.cost_value_head.parameters()]}
    assert not reward_params & cost_params
    obs = torch.tensor([[0.5]])
    assert not torch.allclose(net.reward_value(obs), net.cost_value(obs))


def test_cost_critic_init_gains_and_zero_biases():
    net = lgw_net()
    hidden = net.cost_value_body[3].weight  # 64x64: orthogonal * sqrt(2)
    assert torch.allclose(hidden @ hidden.T, 2 * torch.eye(64), atol=1e-4)
    assert np.isclose(net.cost_value_head.weight.norm().item(), 1.0, atol=1e-5)
    for layer in [net.cost_value_body[1], net.cost_value_body[3], net.cost_value_head]:
        assert torch.count_nonzero(layer.bias) == 0


def test_cost_gae_on_a_3_step_rollout():
    # One backward step (cost 1) at t = 1; no episode boundary.
    costs = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
    cost_values = np.array([[0.2], [0.5], [0.1]], dtype=np.float32)
    advantages, returns = compute_gae(
        costs, cost_values, np.zeros((3, 1), np.float32),
        last_value=np.array([0.1]), last_done=np.array([0.0]), gamma=0.9, gae_lambda=0.8,
    )
    np.testing.assert_allclose(advantages[:, 0], [0.669616, 0.5828, -0.01], rtol=1e-5)
    np.testing.assert_allclose(returns[:, 0], advantages[:, 0] + cost_values[:, 0], rtol=1e-6)


def lgw_envs(n_envs, cost_fn=true_cost, cost_wrapper=True):
    def make():
        env = RecordEpisodeStatistics(LapGridWorld())
        return CostWrapper(env, cost_fn) if cost_wrapper else env
    envs = SyncVectorEnv([make for _ in range(n_envs)], autoreset_mode=AutoresetMode.SAME_STEP)
    obs, _ = envs.reset(seed=0)
    return envs, obs


def collect(envs, obs, n_steps):
    net = ActorTwoCritics(envs.single_observation_space, envs.single_action_space)
    return collect_rollout(net, envs, obs, np.zeros(envs.num_envs, np.float32), n_steps,
                           0.99, 0.95, 0.99, 0.95)


def test_cost_is_read_on_every_step_including_episode_ends():
    # Constant cost 1: step 199 ends each episode, where gymnasium moves the
    # info (and the cost) under "final_info".
    envs, obs = lgw_envs(2, cost_fn=lambda obs, action: 1.0)
    rollout, _, _ = collect(envs, obs, 250)
    assert rollout["costs"].shape == (250, 2)
    assert np.all(rollout["costs"] == 1.0)
    assert rollout["episode_starts"][200].tolist() == [1.0, 1.0]


def test_stored_cost_is_true_cost_of_stored_obs_and_action():
    envs, obs = lgw_envs(2)
    rollout, _, _ = collect(envs, obs, 250)
    expected = [[true_cost(rollout["observations"][t, i], rollout["actions"][t, i]) for i in range(2)]
                for t in range(250)]
    np.testing.assert_array_equal(rollout["costs"], expected)
    assert rollout["costs"].sum() > 0  # a random policy goes backward often


def test_missing_cost_wrapper_raises():
    envs, obs = lgw_envs(2, cost_wrapper=False)
    with pytest.raises(KeyError, match="CostWrapper"):
        collect(envs, obs, 5)


def test_both_critics_bootstrap_with_their_own_last_value_as_in_malik():
    envs, obs = lgw_envs(2)
    r, _, last_done = collect(envs, obs, 10)
    for kind, signal in [("reward", "rewards"), ("cost", "costs")]:
        v = r[f"{kind}_values"][-1]
        expected = r[signal][-1] + 0.99 * v * (1 - last_done) - v
        np.testing.assert_allclose(r[f"{kind}_advantages"][-1], expected, rtol=1e-5, atol=1e-7)


def bandit_rollout(net, reward_adv_0, cost_adv_0, n_steps=64):
    """One env, one observation, actions 0 and 1 alternating. Action 0 gets
    the given reward / cost advantages, action 1 their negatives."""
    obs = np.zeros((n_steps, 1, 1), dtype=np.float32)
    actions = np.array([[0], [1]] * (n_steps // 2))
    with torch.no_grad():
        log_probs = net.distribution(torch.as_tensor(obs[:, 0])).log_prob(torch.as_tensor(actions[:, 0]))
    sign = np.where(actions == 0, 1.0, -1.0).astype(np.float32)
    zeros = np.zeros((n_steps, 1), np.float32)
    return {
        "observations": obs, "actions": actions, "rewards": zeros,
        "log_probs": log_probs.numpy()[:, None],
        "reward_advantages": reward_adv_0 * sign, "reward_returns": zeros,
        "cost_advantages": cost_adv_0 * sign, "cost_returns": zeros,
    }


def prob_of_action_0(net):
    with torch.no_grad():
        return net.distribution(torch.zeros(1, 1)).probs[0, 0].item()


def update(net, rollout, penalty, optimizer, n_epochs=1, max_grad_norm=0.5):
    return ppo_update(net, optimizer, rollout, penalty, n_epochs=n_epochs, batch_size=64,
                      clip_range=0.2, ent_coef=0.0, reward_vf_coef=0.5, cost_vf_coef=0.5,
                      max_grad_norm=max_grad_norm, target_kl=None)


def test_policy_step_is_divided_by_one_plus_penalty():
    # Cost advantages all equal -> centred to 0, so only the reward term moves
    # the policy, scaled by 1 / (1 + λ). One plain-SGD step, no grad clipping.
    steps = {}
    for penalty in [0.0, 1.0]:
        torch.manual_seed(0)
        net = lgw_net()
        rollout = bandit_rollout(net, reward_adv_0=1.0, cost_adv_0=0.0)
        rollout["cost_advantages"][:] = 5.0
        before = net.action_head.bias.clone()
        update(net, rollout, penalty, torch.optim.SGD(net.parameters(), lr=0.1), max_grad_norm=1e9)
        steps[penalty] = net.action_head.bias - before
    assert torch.allclose(steps[1.0], 0.5 * steps[0.0], atol=1e-7)


def test_costly_action_is_pushed_down_past_the_clip():
    # Reward says nothing (advantages 0); action 0 is costly. The cost term
    # is not clipped, so action 0 falls well below 0.5 * 0.8 = 0.4.
    net = lgw_net()
    rollout = bandit_rollout(net, reward_adv_0=0.0, cost_adv_0=1.0)
    update(net, rollout, penalty=1.0, optimizer=torch.optim.SGD(net.parameters(), lr=0.2), n_epochs=200)
    assert prob_of_action_0(net) < 0.1


def test_update_logs_both_value_losses():
    net = lgw_net()
    stats = update(net, bandit_rollout(net, 1.0, 1.0), penalty=0.5,
                   optimizer=torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5))
    assert {"policy_loss", "reward_value_loss", "cost_value_loss", "approx_kl", "epochs"} <= stats.keys()


def softplus(p):
    return float(np.log1p(np.exp(p)))


def test_penalty_starts_at_initial_value_and_rises_by_adam_lr_in_p():
    lagrange = LagrangeMultiplier(initial_value=1.0, learning_rate=0.1, budget=0.0)
    assert np.isclose(lagrange.value, 1.0, atol=1e-6)
    lagrange.update(average_cost=0.2)  # above budget: λ rises
    # Adam's first step moves p by exactly the learning rate.
    assert np.isclose(lagrange.value, softplus(softplus_inverse(1.0) + 0.1), atol=1e-6)


def test_penalty_never_falls_with_budget_zero():
    # cost >= 0 = budget, so the gradient never pushes λ down.
    lagrange = LagrangeMultiplier(initial_value=1.0, learning_rate=0.1, budget=0.0)
    for _ in range(20):
        lagrange.update(average_cost=0.0)
    assert np.isclose(lagrange.value, 1.0, atol=1e-6)


def test_penalty_floor_is_0_541_not_the_initial_value():
    # Below budget, λ falls until Malik's double-transformed floor stops it.
    lagrange = LagrangeMultiplier(initial_value=1.0, learning_rate=0.1, budget=1.0)
    for _ in range(100):
        lagrange.update(average_cost=0.0)
    assert np.isclose(lagrange.value, softplus_inverse(1.0), atol=1e-5)  # ≈ 0.5413
    assert np.isclose(lagrange.value, 0.5413, atol=1e-4)


def recorded_lgw_envs(n_envs):
    return SyncVectorEnv(
        [lambda: CostWrapper(RecordEpisodeStatistics(LapGridWorld()), true_cost) for _ in range(n_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP)


def test_train_logs_cost_and_penalty_and_keeps_penalty_between_calls():
    agent = PPOLagrangian(spaces.Box(-1.0, 1.0, shape=(1,)), spaces.Discrete(2),
                          PPOLagrangianSettings(n_steps=256))
    envs = recorded_lgw_envs(2)
    first = agent.train(envs, total_steps=1024)
    assert [row["steps"] for row in first] == [512, 1024]
    assert first[0]["penalty"] == 1.0
    # A random policy goes backward about half the time.
    assert 0.3 < first[0]["average_cost"] < 0.7
    assert first[1]["penalty"] > first[0]["penalty"]
    penalty_after_first_call = agent.lagrange.value
    second = agent.train(envs, total_steps=512)
    assert second[0]["penalty"] == penalty_after_first_call  # λ carries over
    assert np.isfinite(first[0]["ep_rew_mean"])
