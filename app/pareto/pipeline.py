"""
Pareto stage entry point: assemble upstream objects, then apply the survivor filter.
"""

from __future__ import annotations

from app.cost.base import MultiHorizonCostEstimate
from app.forecasting.base import MultiHorizonForecast
from app.pareto.assembler import assemble_candidate_scores
from app.pareto.contracts import CandidateScore, ParetoConfig
from app.pareto.filter import apply_pareto_filter
from app.risk.base import MultiHorizonRiskEstimate
from app.routing.base import MultiTokenRoutingDecision

__all__ = ["run_pareto_stage"]


def run_pareto_stage(
    forecasts: dict[str, MultiHorizonForecast],
    risks: dict[str, MultiHorizonRiskEstimate],
    costs: dict[str, MultiHorizonCostEstimate],
    liquidity_usd_by_symbol: dict[str, float],
    trade_size_dollar: float,
    routing_decision: MultiTokenRoutingDecision,
    config: ParetoConfig,
) -> list[CandidateScore]:
    """
    Single entry point for Step 9 (LinUCB bandit).

    Runs :func:`~app.pareto.assembler.assemble_candidate_scores` then
    :func:`~app.pareto.filter.apply_pareto_filter`.

    Returns between 0 and ``config.max_survivors`` candidates (filter rules apply).
    """
    assembled = assemble_candidate_scores(
        forecasts=forecasts,
        risks=risks,
        costs=costs,
        liquidity_usd_by_symbol=liquidity_usd_by_symbol,
        trade_size_dollar=trade_size_dollar,
        routing_decision=routing_decision,
    )
    return apply_pareto_filter(assembled, config)
