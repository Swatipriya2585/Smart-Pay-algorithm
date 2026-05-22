"""Tests for the backtest harness (Step 12.2)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from app.backtest.episode import BacktestEpisode, EpisodeConfig, generate_episodes
from app.backtest.harness import (
    BacktestRecord,
    HarnessConfig,
    PolicyChoice,
    build_survivors_for_episode,
    run_backtest,
)
from app.backtest.policies import (
    CheapestRawSpreadPolicy,
    HighestCvarPolicy,
    HighestReturnPolicy,
    LargestBalancePolicy,
    LinUCBPolicy,
    LowestCostPolicy,
    OraclePolicy,
    RandomPolicy,
    StablecoinFirstPolicy,
)
from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import LinUCBConfig
from app.cost.scorer import SolanaCostScorer
from app.forecasting.garch import GARCHForecaster
from app.market_data.calibration import Calibration
from app.market_data.mock import MockConfig, MockMarketData
from app.pareto.contracts import ParetoConfig
from app.regime.threshold import ThresholdRegimeDetector
from app.risk.monte_carlo import MonteCarloCVaR
from app.routing.risk_adaptive import RuleBasedRiskAdaptiveRouter


def _stage_deps(deps: dict[str, Any]) -> dict[str, Any]:
    """Keys accepted by :func:`~app.backtest.harness.build_survivors_for_episode`."""
    return {
        k: deps[k]
        for k in (
            "market_data_source",
            "calibration",
            "forecaster",
            "risk_estimator",
            "cost_scorer",
            "regime_detector",
            "router",
            "pareto_config",
        )
    }


def build_all_deps_and_policies(
    seed: int,
) -> tuple[dict[str, Any], list, dict]:
    """Stages, pareto config, shared bandit arms, and all nine policies."""
    cal = Calibration()
    bcal = BanditCalibration()
    mock = MockMarketData(calibration=cal, config=MockConfig(seed=seed))
    deps: dict[str, Any] = {
        "market_data_source": mock,
        "calibration": cal,
        "bandit_calibration": bcal,
        "forecaster": GARCHForecaster(calibration=cal),
        "risk_estimator": MonteCarloCVaR(),
        "cost_scorer": SolanaCostScorer(),
        "regime_detector": ThresholdRegimeDetector(calibration=cal),
        "router": RuleBasedRiskAdaptiveRouter(),
        "pareto_config": ParetoConfig(),
    }
    bandit_arms: dict = {}
    policies = [
        RandomPolicy(seed=seed),
        CheapestRawSpreadPolicy(),
        LargestBalancePolicy(),
        StablecoinFirstPolicy(calibration=cal),
        LowestCostPolicy(),
        HighestReturnPolicy(),
        HighestCvarPolicy(),
        LinUCBPolicy(bandit_arms, LinUCBConfig(), cal, bcal),
        OraclePolicy(),
    ]
    return deps, policies, bandit_arms


def _run(
    episodes: list[BacktestEpisode],
    *,
    seed: int = 42,
    harness_config: HarnessConfig | None = None,
    policies: list | None = None,
    bandit_arms: dict | None = None,
) -> tuple[BacktestRecord, dict, list]:
    deps, default_policies, default_arms = build_all_deps_and_policies(seed)
    pol = policies if policies is not None else default_policies
    arms = bandit_arms if bandit_arms is not None else default_arms
    cfg = harness_config if harness_config is not None else HarnessConfig()
    record = run_backtest(
        episodes,
        pol,
        market_data_source=deps["market_data_source"],
        calibration=deps["calibration"],
        bandit_calibration=deps["bandit_calibration"],
        forecaster=deps["forecaster"],
        risk_estimator=deps["risk_estimator"],
        cost_scorer=deps["cost_scorer"],
        regime_detector=deps["regime_detector"],
        router=deps["router"],
        pareto_config=deps["pareto_config"],
        config=cfg,
        seed=seed,
    )
    return record, arms, pol


@pytest.fixture
def episodes() -> list[BacktestEpisode]:
    return generate_episodes(
        EpisodeConfig(n_episodes=30, seed=99, symbols=("SOL", "USDC", "BONK", "JUP"))
    )


@pytest.fixture
def deps_and_policies() -> tuple[dict[str, Any], list]:
    d, p, _ = build_all_deps_and_policies(42)
    return d, p


def test_returns_candidate_scores(
    episodes: list[BacktestEpisode],
    deps_and_policies: tuple[dict[str, Any], list],
) -> None:
    deps, _ = deps_and_policies
    ep = episodes[0]
    survivors = build_survivors_for_episode(ep, **_stage_deps(deps))
    assert survivors
    eligible = set(ep.eligible_symbols())
    assert {s.symbol for s in survivors}.issubset(eligible)


def test_survivors_bounded_by_pareto(
    episodes: list[BacktestEpisode],
    deps_and_policies: tuple[dict[str, Any], list],
) -> None:
    deps, _ = deps_and_policies
    survivors = build_survivors_for_episode(episodes[0], **_stage_deps(deps))
    assert len(survivors) <= deps["pareto_config"].max_survivors


def test_runs_all_policies_over_all_episodes(episodes: list[BacktestEpisode]) -> None:
    record, _, policies = _run(episodes, seed=7)
    n_policies = len(policies)
    n_played = record.n_episodes - record.n_skipped_episodes
    assert record.n_episodes == 30
    assert len(record.choices) == n_played * n_policies


def test_every_choice_symbol_in_survivors(episodes: list[BacktestEpisode]) -> None:
    deps, policies, _ = build_all_deps_and_policies(11)
    for ep in episodes[:5]:
        survivors = build_survivors_for_episode(ep, **_stage_deps(deps))
        if not survivors:
            continue
        allowed = {s.symbol for s in survivors}
        for policy in policies:
            chosen = policy.choose(ep, survivors)
            assert chosen in allowed


def test_oracle_reward_is_max_among_policies_per_episode(
    episodes: list[BacktestEpisode],
) -> None:
    record, _, _ = _run(episodes, seed=13)
    by_episode: dict[int, list[PolicyChoice]] = {}
    for choice in record.choices:
        by_episode.setdefault(choice.episode_id, []).append(choice)

    for ep_id, ep_choices in by_episode.items():
        oracle = next(c for c in ep_choices if c.policy_name == "oracle")
        if oracle.reward is None:
            continue
        for other in ep_choices:
            if other.policy_name == "oracle":
                continue
            if other.reward is None:
                continue
            assert oracle.reward >= other.reward - 1e-12, (
                f"episode {ep_id}: {other.policy_name} beat oracle "
                f"({other.reward} > {oracle.reward})"
            )


def test_skipped_episodes_counted(episodes: list[BacktestEpisode]) -> None:
    target = episodes[5]

    def _no_survivors(ep: BacktestEpisode, **kwargs: Any) -> list:
        if ep.episode_id == target.episode_id:
            return []
        return build_survivors_for_episode(ep, **kwargs)

    deps, policies, arms = build_all_deps_and_policies(3)
    with patch(
        "app.backtest.harness.build_survivors_for_episode",
        side_effect=_no_survivors,
    ):
        record = run_backtest(
            episodes,
            policies,
            market_data_source=deps["market_data_source"],
            calibration=deps["calibration"],
            bandit_calibration=deps["bandit_calibration"],
            forecaster=deps["forecaster"],
            risk_estimator=deps["risk_estimator"],
            cost_scorer=deps["cost_scorer"],
            regime_detector=deps["regime_detector"],
            router=deps["router"],
            pareto_config=deps["pareto_config"],
            config=HarnessConfig(),
            seed=3,
        )
    assert record.n_skipped_episodes >= 1
    assert not any(c.episode_id == target.episode_id for c in record.choices)


def test_normal_run_zero_skipped(episodes: list[BacktestEpisode]) -> None:
    record, _, _ = _run(episodes, seed=5)
    assert record.n_skipped_episodes == 0


def test_bandit_learns_arms_evolve(episodes: list[BacktestEpisode]) -> None:
    _, arms, _ = _run(
        episodes,
        seed=20,
        harness_config=HarnessConfig(bandit_learns=True),
    )
    assert arms
    assert any(arm.n_updates > 0 for arm in arms.values())


def test_bandit_no_learning_when_disabled(episodes: list[BacktestEpisode]) -> None:
    _, arms, _ = _run(
        episodes,
        harness_config=HarnessConfig(bandit_learns=False),
    )
    assert not arms or all(arm.n_updates == 0 for arm in arms.values())


def test_bandit_only_learns_own_choices(episodes: list[BacktestEpisode]) -> None:
    deps, policies, arms = build_all_deps_and_policies(31)
    cfg = HarnessConfig(bandit_learns=True)
    record = run_backtest(
        episodes,
        policies,
        market_data_source=deps["market_data_source"],
        calibration=deps["calibration"],
        bandit_calibration=deps["bandit_calibration"],
        forecaster=deps["forecaster"],
        risk_estimator=deps["risk_estimator"],
        cost_scorer=deps["cost_scorer"],
        regime_detector=deps["regime_detector"],
        router=deps["router"],
        pareto_config=deps["pareto_config"],
        config=cfg,
        seed=31,
    )
    expected_updates = sum(
        1
        for c in record.choices
        if c.policy_name == "linucb" and c.reward is not None
    )
    actual_updates = sum(arm.n_updates for arm in arms.values())
    assert actual_updates == expected_updates


def test_backtest_deterministic(episodes: list[BacktestEpisode]) -> None:
    eps = episodes[:20]
    r1, _, _ = _run(eps, seed=100)
    r2, _, _ = _run(eps, seed=100)
    assert r1 == r2


def test_record_has_all_policy_categories(episodes: list[BacktestEpisode]) -> None:
    record, _, policies = _run(episodes[:10], seed=8)
    for policy in policies:
        assert record.policy_categories[policy.name] == policy.category


def test_record_policy_names_stable_order(episodes: list[BacktestEpisode]) -> None:
    _, policies, _ = build_all_deps_and_policies(1)
    record, _, _ = _run(episodes[:5], seed=1, policies=policies)
    assert record.policy_names == tuple(p.name for p in policies)
