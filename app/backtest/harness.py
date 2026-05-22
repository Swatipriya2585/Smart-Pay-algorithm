"""
Backtest harness: run policies over episodes and record per-choice rewards (Step 12.2).

**Honesty constraint:** Episodes use synthetic outcomes (GARCH + noise). This
validates learning mechanics and relative policy performance, not real-world edge
(Step 13). Metrics and charts are produced in Step 12.3 from :class:`BacktestRecord`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.backtest.episode import BacktestEpisode
from app.backtest.policies import BacktestPolicy, LinUCBPolicy
from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import LinUCBArmState, LinUCBConfig
from app.bandit.linucb import update_arm
from app.bandit.persistence import get_or_create_arm
from app.bandit.vectorize import build_feature_vector
from app.cost.base import MultiHorizonCostEstimate
from app.cost.scorer import SolanaCostScorer
from app.feedback.contracts import RewardConfig
from app.feedback.reward import compute_reward
from app.forecasting.base import DEFAULT_HORIZONS, MultiHorizonForecast
from app.forecasting.garch import GARCHForecaster
from app.market_data.base import MarketDataSource
from app.market_data.calibration import Calibration
from app.orchestrator import (
    apply_live_snapshot,
    select_regime_reference_symbol,
    to_network_conditions,
)
from app.pareto.contracts import CandidateScore, ParetoConfig
from app.pareto.pipeline import run_pareto_stage
from app.regime.threshold import ThresholdRegimeDetector
from app.risk.base import MultiHorizonRiskEstimate
from app.risk.monte_carlo import MonteCarloCVaR
from app.routing.risk_adaptive import RuleBasedRiskAdaptiveRouter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyChoice:
    """One policy's decision and resulting reward for one episode."""

    episode_id: int
    policy_name: str
    policy_category: str
    chosen_symbol: str
    reward: Optional[float]


@dataclass(frozen=True)
class BacktestRecord:
    """Raw backtest output; Step 12.3 aggregates this into metrics."""

    choices: list[PolicyChoice]
    n_episodes: int
    n_skipped_episodes: int
    policy_names: tuple[str, ...]
    policy_categories: dict[str, str]
    seed: int


@dataclass(frozen=True)
class HarnessConfig:
    """Harness tuning knobs."""

    reward_config: RewardConfig = field(default_factory=RewardConfig)
    linucb_config: LinUCBConfig = field(default_factory=LinUCBConfig)
    bandit_learns: bool = True
    forecast_horizons: tuple[float, ...] = DEFAULT_HORIZONS


def _eligible_symbols_ordered(episode: BacktestEpisode) -> list[str]:
    """Eligible symbols in ``context.tokens`` order."""
    return [t.symbol for t in episode.context.tokens if t.symbol in episode.outcomes_by_symbol]


def _bandit_arms_from_policies(policies: list[BacktestPolicy]) -> dict[str, LinUCBArmState]:
    for policy in policies:
        if isinstance(policy, LinUCBPolicy):
            return policy._arms
    return {}


def _episode_now_iso(episode_id: int) -> str:
    """Deterministic timestamp for bandit arm updates (no wall-clock dependency)."""
    return f"2026-01-01T00:00:{episode_id:06d}Z"


def build_survivors_for_episode(
    episode: BacktestEpisode,
    *,
    market_data_source: MarketDataSource,
    calibration: Calibration,
    forecaster: GARCHForecaster,
    risk_estimator: MonteCarloCVaR,
    cost_scorer: SolanaCostScorer,
    regime_detector: ThresholdRegimeDetector,
    router: RuleBasedRiskAdaptiveRouter,
    pareto_config: ParetoConfig,
    forecast_horizons: tuple[float, ...] = DEFAULT_HORIZONS,
) -> list[CandidateScore]:
    """Run scoring stages through Pareto; no bandit or outbox.

    Mirrors :func:`~app.orchestrator.run_orchestration` up to
    :func:`~app.pareto.pipeline.run_pareto_stage`. Returns ``[]`` when there
    are no eligible symbols or Pareto produces no survivors.
    """
    eligible = _eligible_symbols_ordered(episode)
    if not eligible:
        return []

    snapshot_by_symbol = {t.symbol: t for t in episode.context.tokens}
    network = to_network_conditions(episode.context.network)

    md_list = market_data_source.fetch(eligible)
    md_by_symbol = {d.symbol: d for d in md_list}

    for symbol in eligible:
        md_by_symbol[symbol] = apply_live_snapshot(
            md_by_symbol[symbol],
            snapshot_by_symbol[symbol],
        )

    position_value_usd = float(episode.context.intent.amount_usd)

    forecasts: dict[str, MultiHorizonForecast] = {}
    risks: dict[str, MultiHorizonRiskEstimate] = {}
    costs: dict[str, MultiHorizonCostEstimate] = {}
    liquidity_by_symbol: dict[str, float] = {}

    for symbol in eligible:
        data = md_by_symbol[symbol]
        forecast = forecaster.forecast(data, forecast_horizons)
        risk = risk_estimator.estimate(data, forecast, position_value_usd)
        cost = cost_scorer.estimate(data, forecast, network, position_value_usd)
        forecasts[symbol] = forecast
        risks[symbol] = risk
        costs[symbol] = cost
        liquidity_by_symbol[symbol] = data.liquidity_depth_usd

    ref_symbol = select_regime_reference_symbol(eligible, calibration)
    regime = regime_detector.classify(md_by_symbol[ref_symbol])

    is_stablecoin = {
        symbol: calibration.get(symbol).is_stablecoin for symbol in eligible
    }

    routing_decision = router.decide(regime, risks, is_stablecoin, network)

    return run_pareto_stage(
        forecasts=forecasts,
        risks=risks,
        costs=costs,
        liquidity_usd_by_symbol=liquidity_by_symbol,
        trade_size_dollar=position_value_usd,
        routing_decision=routing_decision,
        config=pareto_config,
    )


def run_backtest(
    episodes: list[BacktestEpisode],
    policies: list[BacktestPolicy],
    *,
    market_data_source: MarketDataSource,
    calibration: Calibration,
    bandit_calibration: BanditCalibration,
    forecaster: GARCHForecaster,
    risk_estimator: MonteCarloCVaR,
    cost_scorer: SolanaCostScorer,
    regime_detector: ThresholdRegimeDetector,
    router: RuleBasedRiskAdaptiveRouter,
    pareto_config: ParetoConfig,
    config: HarnessConfig,
    seed: int = 0,
) -> BacktestRecord:
    """Run every policy on every episode; only the bandit learns in-memory.

    Survivors are built once per episode (Pareto stage). All policies choose
    among the same survivor set. Bandit arm updates use
    :func:`~app.bandit.vectorize.build_feature_vector` after each episode when
    ``config.bandit_learns`` is True (on-policy, from the bandit's own reward).
    """
    if not policies:
        raise ValueError("policies must be non-empty")

    policy_names = tuple(p.name for p in policies)
    policy_categories = {p.name: p.category for p in policies}
    bandit_arms = _bandit_arms_from_policies(policies)

    logger.info(
        "backtest start: %d episodes, %d policies, seed=%d, bandit_learns=%s",
        len(episodes),
        len(policies),
        seed,
        config.bandit_learns,
    )

    choices: list[PolicyChoice] = []
    n_skipped = 0

    for episode in episodes:
        survivors = build_survivors_for_episode(
            episode,
            market_data_source=market_data_source,
            calibration=calibration,
            forecaster=forecaster,
            risk_estimator=risk_estimator,
            cost_scorer=cost_scorer,
            regime_detector=regime_detector,
            router=router,
            pareto_config=pareto_config,
            forecast_horizons=config.forecast_horizons,
        )

        if not survivors:
            n_skipped += 1
            logger.info(
                "episode %d skipped: no Pareto survivors (eligible=%s)",
                episode.episode_id,
                _eligible_symbols_ordered(episode),
            )
            continue

        survivor_symbols = {s.symbol for s in survivors}
        amount_usd = float(episode.context.intent.amount_usd)
        episode_choices: list[PolicyChoice] = []

        for policy in policies:
            chosen = policy.choose(episode, survivors)
            if chosen not in survivor_symbols:
                raise ValueError(
                    f"policy {policy.name!r} chose {chosen!r} not in survivors "
                    f"{sorted(survivor_symbols)} for episode {episode.episode_id}"
                )
            outcome = episode.outcome_if_chosen(chosen)
            reward = compute_reward(outcome, amount_usd, config.reward_config)
            episode_choices.append(
                PolicyChoice(
                    episode_id=episode.episode_id,
                    policy_name=policy.name,
                    policy_category=policy.category,
                    chosen_symbol=chosen,
                    reward=reward,
                )
            )

        choices.extend(episode_choices)

        if config.bandit_learns:
            bandit_choice = next(
                (c for c in episode_choices if c.policy_name == "linucb"),
                None,
            )
            if bandit_choice is not None and bandit_choice.reward is not None:
                chosen = bandit_choice.chosen_symbol
                x = build_feature_vector(
                    episode.context,
                    chosen,
                    calibration,
                    bandit_calibration,
                )
                arm = get_or_create_arm(bandit_arms, chosen, config.linucb_config)
                bandit_arms[chosen] = update_arm(
                    arm,
                    x,
                    float(bandit_choice.reward),
                    _episode_now_iso(episode.episode_id),
                )

    logger.info(
        "backtest complete: %d choices, %d skipped episodes (of %d)",
        len(choices),
        n_skipped,
        len(episodes),
    )

    return BacktestRecord(
        choices=choices,
        n_episodes=len(episodes),
        n_skipped_episodes=n_skipped,
        policy_names=policy_names,
        policy_categories=policy_categories,
        seed=seed,
    )
