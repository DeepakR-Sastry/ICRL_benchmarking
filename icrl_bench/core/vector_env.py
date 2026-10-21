"""Builds the agent's vector env: n_envs copies of an env, each recording
episode statistics and putting a cost into info["cost"] (docs/design.md,
"Agents")."""

from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from icrl_bench.core.cost_wrapper import CostWrapper


def make_vector_env(make_env, n_envs, cost_fn):
    """make_env: a function with no arguments that returns a fresh env.
    cost_fn(obs, action) -> float, or None if it is set (with
    envs.call("set_cost_fn", ...)) before the first step, as the ICRL loop does.

    CostWrapper goes outermost, so cost_fn sees the raw observation.
    RecordEpisodeStatistics records raw episode returns, like the Monitor
    Malik puts on every env. SAME_STEP resets an env in the step its episode
    ends, as SB3 does."""
    return SyncVectorEnv(
        [lambda: CostWrapper(RecordEpisodeStatistics(make_env()), cost_fn) for _ in range(n_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP)
