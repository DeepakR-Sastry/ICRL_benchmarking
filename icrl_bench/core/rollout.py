"""Runs a policy in an env and records what happened as Trajectories."""

from icrl_bench.core.types import Step, Trajectory


def rollout(env, policy, n_episodes, seed):
    """Play n_episodes with `policy` (obs -> action); return one Trajectory each.

    Each Step pairs an action with the observation it was chosen on, as Liu
    et al.'s sampler does (Malik's pairs it with the next one; see
    docs/decisions.md). The env is seeded once, on the first reset; later
    resets continue from that random state."""
    trajectories = []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed if episode == 0 else None)
        steps = []
        done = False
        while not done:
            action = policy(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            steps.append(Step(obs, action, float(reward)))
            obs = next_obs
            done = terminated or truncated
        trajectories.append(Trajectory(steps))
    return trajectories
