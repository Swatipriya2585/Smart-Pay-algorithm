"""
Assemble :class:`~app.pareto.contracts.CandidateScore` rows from upstream RAMHD stages.

Router output defines the candidate set; Pareto math runs only on
``included_symbols()``. ``score_bias_bps`` is folded into ``effective_cost_bps``.
"""

from __future__ import annotations

import logging

from app.cost.base import MultiHorizonCostEstimate
from app.forecasting.base import MultiHorizonForecast
from app.pareto.contracts import CandidateScore
from app.risk.base import MultiHorizonRiskEstimate
from app.routing.base import MultiTokenRoutingDecision

logger = logging.getLogger(__name__)

PARETO_HORIZON_SECONDS = 120.0
PARETO_CVAR_CONFIDENCE = 0.95

__all__ = [
    "PARETO_CVAR_CONFIDENCE",
    "PARETO_HORIZON_SECONDS",
    "assemble_candidate_scores",
]


def assemble_candidate_scores(
    forecasts: dict[str, MultiHorizonForecast],
    risks: dict[str, MultiHorizonRiskEstimate],
    costs: dict[str, MultiHorizonCostEstimate],
    liquidity_usd_by_symbol: dict[str, float],
    trade_size_dollar: float,
    routing_decision: MultiTokenRoutingDecision,
) -> list[CandidateScore]:
    """
    Build CandidateScore objects for symbols included by the router.

    Iteration order follows ``routing_decision.included_symbols()``. Excluded
    symbols are not assembled; each excluded symbol is logged once at INFO.

    For each included symbol, missing entries in ``forecasts`` / ``risks`` /
    ``costs`` / ``liquidity_usd_by_symbol``, or missing 120s horizons in the
    multi-horizon objects, produce a WARNING and the symbol is skipped.

    Raises:
        ValueError: if ``trade_size_dollar`` is not positive, or if the 120s
            tail-risk estimate exists but ``confidence_level != 0.95``.
    """
    if trade_size_dollar <= 0:
        raise ValueError("trade_size_dollar must be positive")

    for sym in routing_decision.excluded_symbols():
        adj = routing_decision.for_symbol(sym)
        logger.info(
            "assembler: skipping excluded symbol %s (%s)",
            sym,
            adj.exclusion_reason,
        )

    out: list[CandidateScore] = []

    for s in routing_decision.included_symbols():
        if s not in forecasts:
            logger.warning(
                "assembler: skipping %s — not present in forecasts dict",
                s,
            )
            continue
        if s not in risks:
            logger.warning("assembler: skipping %s — not present in risks dict", s)
            continue
        if s not in costs:
            logger.warning("assembler: skipping %s — not present in costs dict", s)
            continue
        if s not in liquidity_usd_by_symbol:
            logger.warning(
                "assembler: skipping %s — not present in liquidity_usd_by_symbol dict",
                s,
            )
            continue

        try:
            horizon_f = forecasts[s].at(PARETO_HORIZON_SECONDS)
        except KeyError:
            logger.warning(
                "assembler: skipping %s — forecasts missing horizon %.1fs",
                s,
                PARETO_HORIZON_SECONDS,
            )
            continue

        try:
            tail = risks[s].at(PARETO_HORIZON_SECONDS)
        except KeyError:
            logger.warning(
                "assembler: skipping %s — risks missing horizon %.1fs",
                s,
                PARETO_HORIZON_SECONDS,
            )
            continue

        if tail.confidence_level != PARETO_CVAR_CONFIDENCE:
            raise ValueError(s)

        try:
            cost_br = costs[s].at(PARETO_HORIZON_SECONDS)
        except KeyError:
            logger.warning(
                "assembler: skipping %s — costs missing horizon %.1fs",
                s,
                PARETO_HORIZON_SECONDS,
            )
            continue

        adjustment = routing_decision.for_symbol(s)
        gross_cost_bps = (
            -cost_br.total_cost_dollar / trade_size_dollar
        ) * 10_000.0
        effective_cost_bps = gross_cost_bps - adjustment.score_bias_bps
        liquidity_usd = liquidity_usd_by_symbol[s]

        out.append(
            CandidateScore(
                symbol=s,
                expected_return_120s=horizon_f.predicted_return,
                cvar_95_120s=tail.cvar,
                effective_cost_bps=effective_cost_bps,
                liquidity_usd=liquidity_usd,
            )
        )

    return out
