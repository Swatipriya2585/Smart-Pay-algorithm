"""
RAMHD orchestrator — wires stages 1–10 into one context-in → decision-out call.

This is Step 11: the first integration point where each stage's real output
feeds the next stage's real input. Stages are injected (not instantiated here)
so Prompt 11b can wire production singletons and tests can use real components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import LinUCBConfig
from app.cost.base import MultiHorizonCostEstimate
from app.cost.scorer import SolanaCostScorer
from app.feedback.outbox import OutboxStore
from app.feedback.pipeline import RecordedDecision, record_decision
from app.forecasting.base import DEFAULT_HORIZONS, MultiHorizonForecast
from app.forecasting.garch import GARCHForecaster
from app.market_data.base import (
    MarketDataSource,
    NetworkConditions,
    TokenMarketData,
)
from app.market_data.calibration import Calibration
from app.pareto.contracts import CandidateScore, ParetoConfig
from app.pareto.pipeline import run_pareto_stage
from app.regime.base import RegimeEstimate
from app.regime.threshold import ThresholdRegimeDetector
from app.risk.base import MultiHorizonRiskEstimate
from app.risk.monte_carlo import MonteCarloCVaR
from app.routing.base import MultiTokenRoutingDecision
from app.routing.risk_adaptive import RuleBasedRiskAdaptiveRouter
from app.schemas import NetworkState, RamhdContext, TokenMarketSnapshot

logger = logging.getLogger(__name__)


def to_network_conditions(network: NetworkState) -> NetworkConditions:
    """Map the request's NetworkState (Pydantic) to NetworkConditions (dataclass)."""
    return NetworkConditions(
        priority_fee_lamports=float(network.priority_fee_lamports),
        congestion_score=float(network.congestion_score),
        slot_time_ms=float(network.slot_time_ms),
    )


def apply_live_snapshot(
    market_data: TokenMarketData,
    snapshot: TokenMarketSnapshot,
) -> TokenMarketData:
    """Override liquidity/spread from the live request; keep the synthetic price path."""
    return replace(
        market_data,
        liquidity_depth_usd=float(snapshot.liquidity_depth_usd),
        spread_bps=float(snapshot.spread_bps),
    )


@dataclass(frozen=True)
class OrchestratorConfig:
    pareto_config: ParetoConfig = field(default_factory=ParetoConfig)
    linucb_config: LinUCBConfig = field(default_factory=LinUCBConfig)
    forecast_horizons: tuple[float, ...] = DEFAULT_HORIZONS


@dataclass(frozen=True)
class OrchestratorResult:
    tx_id: str
    chosen_symbol: str
    recorded_decision: RecordedDecision
    survivors: list[CandidateScore]
    regime: RegimeEstimate
    excluded_symbols: tuple[str, ...]
    eligible_symbols: tuple[str, ...]
    skipped_symbols: tuple[str, ...]


def select_regime_reference_symbol(
    eligible_symbols: list[str],
    calibration: Calibration,
) -> str:
    """Pick the token whose market data drives regime classification."""
    if not eligible_symbols:
        raise ValueError("eligible_symbols must be non-empty")

    for symbol in eligible_symbols:
        if not calibration.get(symbol).is_stablecoin:
            return symbol
    return eligible_symbols[0]


def run_orchestration(
    tx_id: str,
    context: RamhdContext,
    *,
    market_data_source: MarketDataSource,
    calibration: Calibration,
    bandit_calibration: BanditCalibration,
    forecaster: GARCHForecaster,
    risk_estimator: MonteCarloCVaR,
    cost_scorer: SolanaCostScorer,
    regime_detector: ThresholdRegimeDetector,
    router: RuleBasedRiskAdaptiveRouter,
    outbox: OutboxStore,
    config: OrchestratorConfig,
    state_path: Optional[Path] = None,
    now_utc_iso: Optional[str] = None,
) -> OrchestratorResult:
    """Run the full RAMHD pipeline for one payment request."""
    if not tx_id:
        raise ValueError("tx_id must be non-empty")
    if not context.tokens:
        raise ValueError("context.tokens must be non-empty")

    eligible: list[str] = []
    skipped: list[str] = []
    for token in context.tokens:
        if calibration.has(token.symbol):
            eligible.append(token.symbol)
        else:
            skipped.append(token.symbol)

    if skipped:
        logger.warning(
            "tx_id=%s skipping %d symbol(s) not in calibration universe: %s",
            tx_id,
            len(skipped),
            skipped,
        )

    if not eligible:
        raise ValueError(
            "no eligible symbols after calibration filter; "
            f"requested={[t.symbol for t in context.tokens]}"
        )

    logger.info(
        "tx_id=%s orchestration start: %d eligible, %d skipped",
        tx_id,
        len(eligible),
        len(skipped),
    )

    snapshot_by_symbol = {t.symbol: t for t in context.tokens}
    network = to_network_conditions(context.network)

    md_list = market_data_source.fetch(eligible)
    md_by_symbol = {d.symbol: d for d in md_list}

    for symbol in eligible:
        md_by_symbol[symbol] = apply_live_snapshot(
            md_by_symbol[symbol],
            snapshot_by_symbol[symbol],
        )

    position_value_usd = float(context.intent.amount_usd)

    forecasts: dict[str, MultiHorizonForecast] = {}
    risks: dict[str, MultiHorizonRiskEstimate] = {}
    costs: dict[str, MultiHorizonCostEstimate] = {}
    liquidity_by_symbol: dict[str, float] = {}

    for symbol in eligible:
        data = md_by_symbol[symbol]
        forecast = forecaster.forecast(data, config.forecast_horizons)
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

    survivors = run_pareto_stage(
        forecasts=forecasts,
        risks=risks,
        costs=costs,
        liquidity_usd_by_symbol=liquidity_by_symbol,
        trade_size_dollar=position_value_usd,
        routing_decision=routing_decision,
        config=config.pareto_config,
    )

    if not survivors:
        raise ValueError(
            f"tx_id={tx_id}: no Pareto survivors after routing "
            f"(eligible={eligible}, excluded={routing_decision.excluded_symbols()})"
        )

    recorded = record_decision(
        tx_id=tx_id,
        context=context,
        survivors=survivors,
        config=config.linucb_config,
        calibration=calibration,
        bandit_calibration=bandit_calibration,
        outbox=outbox,
        state_path=state_path,
        now_utc_iso=now_utc_iso,
    )

    chosen_symbol = recorded.decision.chosen_symbol

    logger.info(
        "tx_id=%s orchestration complete: chosen=%s, %d survivors, regime=%s",
        tx_id,
        chosen_symbol,
        len(survivors),
        regime.regime,
    )

    return OrchestratorResult(
        tx_id=tx_id,
        chosen_symbol=chosen_symbol,
        recorded_decision=recorded,
        survivors=survivors,
        regime=regime,
        excluded_symbols=routing_decision.excluded_symbols(),
        eligible_symbols=tuple(eligible),
        skipped_symbols=tuple(skipped),
    )
