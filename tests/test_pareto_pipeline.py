"""
Tests for :func:`~app.pareto.pipeline.run_pareto_stage`.
"""

from __future__ import annotations

import random

import pytest

from app.cost.base import CostBreakdown, MultiHorizonCostEstimate
from app.forecasting.base import HorizonForecast, MultiHorizonForecast
from app.pareto.contracts import ParetoConfig
from app.pareto.pipeline import run_pareto_stage
from app.regime.base import RegimeEstimate
from app.risk.base import MultiHorizonRiskEstimate, TailRiskEstimate
from app.routing.base import MultiTokenRoutingDecision, RoutingAdjustment


def _valid_regime() -> RegimeEstimate:
    return RegimeEstimate(
        symbol="SOL",
        regime="calm",
        confidence=0.9,
        realized_volatility=0.001,
        baseline_volatility=0.0015,
        ratio=0.667,
    )


def _adj(
    symbol: str,
    *,
    excluded: bool = False,
    exclusion_reason: str | None = None,
    score_bias_bps: float = 0.0,
    bias_reasons: tuple[str, ...] = (),
) -> RoutingAdjustment:
    return RoutingAdjustment(
        symbol=symbol,
        excluded=excluded,
        exclusion_reason=exclusion_reason,
        score_bias_bps=score_bias_bps,
        bias_reasons=bias_reasons,
    )


def _routing(adjustments: tuple[RoutingAdjustment, ...]) -> MultiTokenRoutingDecision:
    return MultiTokenRoutingDecision(adjustments=adjustments, regime=_valid_regime())


def _hz120(ret: float, vol: float = 0.02) -> HorizonForecast:
    return HorizonForecast(
        horizon_seconds=120.0,
        predicted_return=ret,
        predicted_volatility=vol,
        confidence_lower_95=ret - 2 * vol,
        confidence_upper_95=ret + 2 * vol,
    )


def _forecast(symbol: str, ret: float) -> MultiHorizonForecast:
    return MultiHorizonForecast(symbol=symbol, horizons={120.0: _hz120(ret)})


def _tail(cvar: float, var: float) -> TailRiskEstimate:
    return TailRiskEstimate(
        horizon_seconds=120.0,
        confidence_level=0.95,
        var=var,
        cvar=cvar,
        var_dollar=var * 1000.0,
        cvar_dollar=cvar * 1000.0,
        n_samples=5000,
    )


def _risk(symbol: str, cvar: float, var: float) -> MultiHorizonRiskEstimate:
    return MultiHorizonRiskEstimate(
        symbol=symbol,
        position_value_usd=1000.0,
        estimates={120.0: _tail(cvar, var)},
    )


def _cost_br(total: float) -> CostBreakdown:
    return CostBreakdown(
        horizon_seconds=120.0,
        slippage_dollar=total,
        gas_dollar=0.0,
        settlement_risk_dollar=0.0,
        total_cost_dollar=total,
        settlement_seconds=1.0,
    )


def _cost(symbol: str, total: float) -> MultiHorizonCostEstimate:
    b = _cost_br(total)
    return MultiHorizonCostEstimate(
        symbol=symbol,
        position_value_usd=1000.0,
        breakdowns={120.0: b},
    )


def _eight_symbols() -> tuple[str, ...]:
    return ("SOL", "USDC", "PYTH", "AERO", "JUP", "BRETT", "WIF", "BONK")


def _build_eight_token_bundle(
    rng: random.Random,
) -> tuple[
    dict[str, MultiHorizonForecast],
    dict[str, MultiHorizonRiskEstimate],
    dict[str, MultiHorizonCostEstimate],
    dict[str, float],
    MultiTokenRoutingDecision,
]:
    symbols = _eight_symbols()
    forecasts: dict[str, MultiHorizonForecast] = {}
    risks: dict[str, MultiHorizonRiskEstimate] = {}
    costs: dict[str, MultiHorizonCostEstimate] = {}
    liq: dict[str, float] = {}
    for sym in symbols:
        ret = rng.uniform(-0.008, 0.025)
        cvar = -rng.uniform(0.005, 0.12)
        # VaR milder than CVaR: var > cvar numerically while cvar <= var holds.
        var = max(cvar * 0.5, cvar + rng.uniform(0.001, 0.03))
        forecasts[sym] = _forecast(sym, ret)
        risks[sym] = _risk(sym, cvar, var)
        cost_usd = -rng.uniform(2.0, 120.0)
        costs[sym] = _cost(sym, cost_usd)
        liq[sym] = rng.uniform(500_000.0, 80_000_000.0)
    adjs = tuple(_adj(s) for s in symbols)
    routing = _routing(adjs)
    return forecasts, risks, costs, liq, routing


def test_pipeline_runs_end_to_end_on_8_calibration_tokens() -> None:
    rng = random.Random(42)
    forecasts, risks, costs, liq, routing = _build_eight_token_bundle(rng)
    cfg = ParetoConfig()
    result = run_pareto_stage(
        forecasts,
        risks,
        costs,
        liq,
        trade_size_dollar=1000.0,
        routing_decision=routing,
        config=cfg,
    )
    assert 2 <= len(result) <= 5


def test_pipeline_router_exclusions_propagate() -> None:
    rng = random.Random(43)
    forecasts, risks, costs, liq, _ = _build_eight_token_bundle(rng)
    symbols = _eight_symbols()
    adjs_list: list[RoutingAdjustment] = []
    for s in symbols:
        if s in ("BONK", "WIF", "BRETT"):
            adjs_list.append(
                _adj(s, excluded=True, exclusion_reason=f"excluded_{s}")
            )
        else:
            adjs_list.append(_adj(s))
    routing = _routing(tuple(adjs_list))
    cfg = ParetoConfig()
    result = run_pareto_stage(
        forecasts,
        risks,
        costs,
        liq,
        1000.0,
        routing,
        cfg,
    )
    out_syms = {c.symbol for c in result}
    assert "BONK" not in out_syms
    assert "WIF" not in out_syms
    assert "BRETT" not in out_syms


def test_pipeline_dominated_token_dropped() -> None:
    """SOL strictly dominates BONK on all four Pareto dimensions."""
    routing = _routing((_adj("SOL"), _adj("BONK")))
    forecasts = {
        "SOL": _forecast("SOL", 0.05),
        "BONK": _forecast("BONK", -0.02),
    }
    risks = {
        "SOL": _risk("SOL", -0.01, -0.005),
        "BONK": _risk("BONK", -0.9, -0.5),
    }
    costs = {
        "SOL": _cost("SOL", -5.0),
        "BONK": _cost("BONK", -800.0),
    }
    liq = {"SOL": 50_000_000.0, "BONK": 50_000.0}
    cfg = ParetoConfig(min_survivors=1, max_survivors=5)
    result = run_pareto_stage(
        forecasts, risks, costs, liq, 1000.0, routing, cfg
    )
    syms = [c.symbol for c in result]
    assert "SOL" in syms
    assert "BONK" not in syms


def test_pipeline_stablecoin_survives_in_stress_scenario() -> None:
    """USDC is best on all fronts vs speculative names."""
    routing = _routing(
        (
            _adj("USDC"),
            _adj("SOL"),
            _adj("BONK"),
        )
    )
    forecasts = {
        "USDC": _forecast("USDC", 0.0001),
        "SOL": _forecast("SOL", 0.02),
        "BONK": _forecast("BONK", 0.015),
    }
    risks = {
        "USDC": _risk("USDC", -0.0005, -0.0001),
        "SOL": _risk("SOL", -0.08, -0.05),
        "BONK": _risk("BONK", -0.35, -0.2),
    }
    costs = {
        "USDC": _cost("USDC", -2.0),
        "SOL": _cost("SOL", -80.0),
        "BONK": _cost("BONK", -120.0),
    }
    liq = {
        "USDC": 500_000_000.0,
        "SOL": 8_000_000.0,
        "BONK": 2_000_000.0,
    }
    cfg = ParetoConfig()
    result = run_pareto_stage(
        forecasts, risks, costs, liq, 1000.0, routing, cfg
    )
    assert any(c.symbol == "USDC" for c in result)


def test_pipeline_determinism() -> None:
    rng = random.Random(99)
    forecasts, risks, costs, liq, routing = _build_eight_token_bundle(rng)
    cfg = ParetoConfig(tiebreaker="expected_return_120s")
    runs: list[list[object]] = []
    for _ in range(5):
        runs.append(
            run_pareto_stage(
                forecasts, risks, costs, liq, 1000.0, routing, cfg
            )
        )
    first = runs[0]
    assert all(r == first for r in runs[1:])


def test_pipeline_empty_when_router_excludes_all() -> None:
    """Router marks the sole adjustment as excluded → no candidates."""
    routing = _routing(
        (_adj("SOL", excluded=True, exclusion_reason="manual_kill"),)
    )
    forecasts = {"SOL": _forecast("SOL", 0.01)}
    risks = {"SOL": _risk("SOL", -0.02, -0.01)}
    costs = {"SOL": _cost("SOL", -10.0)}
    liq = {"SOL": 1e6}
    cfg = ParetoConfig()
    result = run_pareto_stage(
        forecasts, risks, costs, liq, 1000.0, routing, cfg
    )
    assert result == []
