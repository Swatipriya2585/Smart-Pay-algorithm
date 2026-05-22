"""Tests for backtest metrics (Step 12.3). Hand-built records only — no harness."""

from __future__ import annotations

import pytest

from app.backtest.harness import BacktestRecord, PolicyChoice
from app.backtest.metrics import BacktestMetrics, compute_metrics, learning_lift


def _record(
    choices: list[PolicyChoice],
    *,
    n_episodes: int = 10,
    n_skipped: int = 0,
    seed: int = 0,
) -> BacktestRecord:
    names: list[str] = []
    categories: dict[str, str] = {}
    for c in choices:
        if c.policy_name not in categories:
            names.append(c.policy_name)
            categories[c.policy_name] = c.policy_category
    return BacktestRecord(
        choices=choices,
        n_episodes=n_episodes,
        n_skipped_episodes=n_skipped,
        policy_names=tuple(names),
        policy_categories=categories,
        seed=seed,
    )


def test_mean_reward_excludes_none() -> None:
    choices = [
        PolicyChoice(0, "p", "naive_baseline", "SOL", 0.1),
        PolicyChoice(1, "p", "naive_baseline", "SOL", None),
        PolicyChoice(2, "p", "naive_baseline", "SOL", 0.3),
    ]
    m = compute_metrics(
        _record(choices, n_episodes=3),
        bandit_policy_name="p",
        oracle_policy_name="p",
    )
    pm = m.per_policy["p"]
    assert pm.n_choices == 3
    assert pm.n_missing == 1
    assert pm.total_reward == pytest.approx(0.4)
    assert pm.mean_reward == pytest.approx(0.2)


def test_cumulative_reward_running_sum() -> None:
    choices = [
        PolicyChoice(0, "a", "ml_ablation", "X", 0.1),
        PolicyChoice(1, "a", "ml_ablation", "X", None),
        PolicyChoice(2, "a", "ml_ablation", "X", 0.2),
        PolicyChoice(0, "oracle", "oracle", "X", 1.0),
        PolicyChoice(1, "oracle", "oracle", "X", 1.0),
        PolicyChoice(2, "oracle", "oracle", "X", 1.0),
    ]
    m = compute_metrics(
        _record(choices, n_episodes=3),
        bandit_policy_name="a",
        oracle_policy_name="oracle",
    )
    assert m.cumulative_reward["a"] == pytest.approx([0.1, 0.1, 0.3])


def test_regret_vs_oracle() -> None:
    choices = []
    for ep_id, reward in enumerate([1.0, 1.0, 1.0]):
        choices.append(
            PolicyChoice(ep_id, "oracle", "oracle", "X", 1.0)
        )
    for ep_id, reward in enumerate([0.4, 0.6, 1.0]):
        choices.append(
            PolicyChoice(ep_id, "sub", "ml_ablation", "X", reward)
        )
    m = compute_metrics(
        _record(choices, n_episodes=3),
        bandit_policy_name="sub",
        oracle_policy_name="oracle",
    )
    assert m.total_regret["sub"] == pytest.approx(1.0)
    assert abs(m.total_regret["oracle"]) < 1e-9


def test_bandit_winrate_vs_comparator() -> None:
    choices = []
    # Bandit wins in episodes 0, 1, 2 (3 of 4).
    bandit_rewards = [0.5, 0.6, 0.7, 0.2]
    comp_rewards = [0.4, 0.5, 0.6, 0.8]
    for i, (br, cr) in enumerate(zip(bandit_rewards, comp_rewards, strict=True)):
        choices.append(PolicyChoice(i, "linucb", "bandit", "X", br))
        choices.append(PolicyChoice(i, "comp", "naive_baseline", "X", cr))
        choices.append(PolicyChoice(i, "oracle", "oracle", "X", 1.0))
    m = compute_metrics(_record(choices, n_episodes=4))
    assert m.bandit_winrate_vs["comp"] == pytest.approx(0.75)


def test_first_second_half_split_and_learning_lift() -> None:
    choices = []
    for ep_id in range(6):
        reward = 0.1 * ep_id
        choices.append(PolicyChoice(ep_id, "linucb", "bandit", "X", reward))
        choices.append(PolicyChoice(ep_id, "oracle", "oracle", "X", 1.0))
    m = compute_metrics(_record(choices, n_episodes=6))
    bandit = m.per_policy["linucb"]
    assert bandit.mean_reward_second_half > bandit.mean_reward_first_half
    assert learning_lift(m) > 0


def test_missing_bandit_name_raises() -> None:
    choices = [PolicyChoice(0, "oracle", "oracle", "X", 1.0)]
    with pytest.raises(ValueError, match="bandit_policy_name"):
        compute_metrics(_record(choices))


def test_missing_oracle_name_raises() -> None:
    choices = [PolicyChoice(0, "linucb", "bandit", "X", 0.1)]
    with pytest.raises(ValueError, match="oracle_policy_name"):
        compute_metrics(_record(choices))


def test_oracle_regret_is_zero() -> None:
    choices = []
    for ep_id in range(4):
        choices.append(PolicyChoice(ep_id, "oracle", "oracle", "X", 0.5 + ep_id * 0.01))
        choices.append(PolicyChoice(ep_id, "linucb", "bandit", "X", 0.1))
    m = compute_metrics(_record(choices, n_episodes=4))
    assert abs(m.total_regret["oracle"]) < 1e-9


def test_metrics_cover_all_policies() -> None:
    names = ("linucb", "oracle", "random")
    choices = []
    for ep_id in range(2):
        for name, cat in (
            ("linucb", "bandit"),
            ("oracle", "oracle"),
            ("random", "naive_baseline"),
        ):
            choices.append(PolicyChoice(ep_id, name, cat, "X", 0.1))
    m = compute_metrics(_record(choices))
    assert set(m.per_policy.keys()) == set(names)
