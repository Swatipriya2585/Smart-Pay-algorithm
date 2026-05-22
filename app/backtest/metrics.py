"""
Backtest metrics aggregation (Step 12.3).

**Honesty constraint:** Metrics summarize a synthetic backtest where outcomes are
model-generated (GARCH + Gaussian noise). They validate learning mechanics and
relative performance vs baselines — not real-world edge.

**None rewards:** Choices with ``reward is None`` (e.g. ``DATA_MISSING``) are
excluded from sums and averages (never treated as zero). They are counted in
``n_missing`` only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.backtest.harness import BacktestRecord, PolicyChoice


@dataclass(frozen=True)
class PolicyMetrics:
    """Per-policy summary over a backtest run."""

    policy_name: str
    policy_category: str
    n_choices: int
    n_missing: int
    total_reward: float
    mean_reward: float
    mean_reward_first_half: float
    mean_reward_second_half: float


@dataclass(frozen=True)
class BacktestMetrics:
    """Aggregated metrics for one :class:`~app.backtest.harness.BacktestRecord`."""

    per_policy: dict[str, PolicyMetrics]
    n_episodes: int
    n_skipped: int
    cumulative_reward: dict[str, list[float]]
    total_regret: dict[str, float]
    bandit_winrate_vs: dict[str, float]
    bandit_policy_name: str
    oracle_policy_name: str


def _acted_episode_ids(record: BacktestRecord) -> list[int]:
    return sorted({c.episode_id for c in record.choices})


def _choices_by_episode(record: BacktestRecord) -> dict[int, dict[str, PolicyChoice]]:
    by_episode: dict[int, dict[str, PolicyChoice]] = defaultdict(dict)
    for choice in record.choices:
        by_episode[choice.episode_id][choice.policy_name] = choice
    return dict(by_episode)


def _mean_of_rewards(rewards: list[float]) -> float:
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def compute_metrics(
    record: BacktestRecord,
    bandit_policy_name: str = "linucb",
    oracle_policy_name: str = "oracle",
) -> BacktestMetrics:
    """Aggregate a :class:`~app.backtest.harness.BacktestRecord` into metrics."""
    if bandit_policy_name not in record.policy_names:
        raise ValueError(
            f"bandit_policy_name {bandit_policy_name!r} not in record.policy_names"
        )
    if oracle_policy_name not in record.policy_names:
        raise ValueError(
            f"oracle_policy_name {oracle_policy_name!r} not in record.policy_names"
        )

    episode_ids = _acted_episode_ids(record)
    by_episode = _choices_by_episode(record)
    mid = len(episode_ids) // 2
    first_half_ids = set(episode_ids[:mid])
    second_half_ids = set(episode_ids[mid:])

    per_policy: dict[str, PolicyMetrics] = {}
    cumulative_reward: dict[str, list[float]] = {}
    total_regret: dict[str, float] = {}

    for policy_name in record.policy_names:
        category = record.policy_categories[policy_name]
        n_choices = 0
        n_missing = 0
        valid_rewards: list[float] = []
        first_half_valid: list[float] = []
        second_half_valid: list[float] = []
        running = 0.0
        cum_series: list[float] = []

        for ep_id in episode_ids:
            choice = by_episode[ep_id][policy_name]
            n_choices += 1
            reward = choice.reward
            if reward is None:
                n_missing += 1
                cum_series.append(running)
                continue
            valid_rewards.append(reward)
            running += reward
            cum_series.append(running)
            if ep_id in first_half_ids:
                first_half_valid.append(reward)
            if ep_id in second_half_ids:
                second_half_valid.append(reward)

        n_valid = n_choices - n_missing
        total = sum(valid_rewards)
        per_policy[policy_name] = PolicyMetrics(
            policy_name=policy_name,
            policy_category=category,
            n_choices=n_choices,
            n_missing=n_missing,
            total_reward=total,
            mean_reward=total / n_valid if n_valid > 0 else 0.0,
            mean_reward_first_half=_mean_of_rewards(first_half_valid),
            mean_reward_second_half=_mean_of_rewards(second_half_valid),
        )
        cumulative_reward[policy_name] = cum_series

        regret_sum = 0.0
        for ep_id in episode_ids:
            ep_choices = by_episode.get(ep_id, {})
            oracle_c = ep_choices.get(oracle_policy_name)
            policy_c = ep_choices.get(policy_name)
            if (
                oracle_c is None
                or policy_c is None
                or oracle_c.reward is None
                or policy_c.reward is None
            ):
                continue
            regret_sum += oracle_c.reward - policy_c.reward
        total_regret[policy_name] = regret_sum

    bandit_winrate_vs: dict[str, float] = {}
    for comp_name in record.policy_names:
        if comp_name == bandit_policy_name:
            continue
        wins = 0
        comparable = 0
        for ep_id in episode_ids:
            ep_choices = by_episode.get(ep_id, {})
            bandit_c = ep_choices.get(bandit_policy_name)
            comp_c = ep_choices.get(comp_name)
            if (
                bandit_c is None
                or comp_c is None
                or bandit_c.reward is None
                or comp_c.reward is None
            ):
                continue
            comparable += 1
            if bandit_c.reward >= comp_c.reward:
                wins += 1
        bandit_winrate_vs[comp_name] = wins / comparable if comparable > 0 else 0.0

    return BacktestMetrics(
        per_policy=per_policy,
        n_episodes=record.n_episodes,
        n_skipped=record.n_skipped_episodes,
        cumulative_reward=cumulative_reward,
        total_regret=total_regret,
        bandit_winrate_vs=bandit_winrate_vs,
        bandit_policy_name=bandit_policy_name,
        oracle_policy_name=oracle_policy_name,
    )


def learning_lift(metrics: BacktestMetrics) -> float:
    """Bandit ``mean_reward_second_half`` minus ``mean_reward_first_half``."""
    bandit = metrics.per_policy[metrics.bandit_policy_name]
    return bandit.mean_reward_second_half - bandit.mean_reward_first_half
