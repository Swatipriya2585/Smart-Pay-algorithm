"""
Tests for :func:`~app.pareto.assembler.assemble_candidate_scores`.
"""

from __future__ import annotations

import logging

import pytest

from app.cost.base import CostBreakdown, MultiHorizonCostEstimate
from app.forecasting.base import HorizonForecast, MultiHorizonForecast
from app.pareto.assembler import assemble_candidate_scores
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


def _valid_adjustment(
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


def _routing(
    adjustments: tuple[RoutingAdjustment, ...],
) -> MultiTokenRoutingDecision:
    return MultiTokenRoutingDecision(
        adjustments=adjustments,
        regime=_valid_regime(),
    )


def _horizon_120(
    predicted_return: float,
    vol: float = 0.01,
) -> HorizonForecast:
    return HorizonForecast(
        horizon_seconds=120.0,
        predicted_return=predicted_return,
        predicted_volatility=vol,
        confidence_lower_95=predicted_return - 2 * vol,
        confidence_upper_95=predicted_return + 2 * vol,
    )


def _forecast(symbol: str, ret_120: float, vol: float = 0.01) -> MultiHorizonForecast:
    return MultiHorizonForecast(
        symbol=symbol,
        horizons={120.0: _horizon_120(ret_120, vol=vol)},
    )


def _tail_120(
    cvar: float,
    *,
    confidence_level: float = 0.95,
    var: float | None = None,
    position_value_usd: float = 1000.0,
) -> TailRiskEstimate:
    # Invariant: cvar <= var (tail average at least as severe as VaR threshold).
    if var is None:
        v = 0.5 * cvar if cvar < 0 else cvar + 0.01
    else:
        v = var
    return TailRiskEstimate(
        horizon_seconds=120.0,
        confidence_level=confidence_level,
        var=v,
        cvar=cvar,
        var_dollar=v * position_value_usd,
        cvar_dollar=cvar * position_value_usd,
        n_samples=10_000,
    )


def _risk(
    symbol: str,
    cvar: float,
    *,
    confidence_level: float = 0.95,
    var: float | None = None,
) -> MultiHorizonRiskEstimate:
    tail = _tail_120(cvar, confidence_level=confidence_level, var=var)
    return MultiHorizonRiskEstimate(
        symbol=symbol,
        position_value_usd=1000.0,
        estimates={120.0: tail},
    )


def _cost_breakdown_120(total_cost_dollar: float) -> CostBreakdown:
    """Single-horizon breakdown: all loss in slippage for easy totals."""
    return CostBreakdown(
        horizon_seconds=120.0,
        slippage_dollar=total_cost_dollar,
        gas_dollar=0.0,
        settlement_risk_dollar=0.0,
        total_cost_dollar=total_cost_dollar,
        settlement_seconds=1.0,
    )


def _cost(symbol: str, total_cost_dollar: float) -> MultiHorizonCostEstimate:
    b = _cost_breakdown_120(total_cost_dollar)
    return MultiHorizonCostEstimate(
        symbol=symbol,
        position_value_usd=1000.0,
        breakdowns={120.0: b},
    )


def test_excluded_symbol_skipped(caplog: pytest.LogCaptureFixture) -> None:
    """BONK excluded by router; still present upstream → not assembled; INFO has reason."""
    adjs = (
        _valid_adjustment(symbol="BONK", excluded=True, exclusion_reason="shock_regime"),
        _valid_adjustment(symbol="SOL"),
    )
    routing = _routing(adjs)
    forecasts = {
        "BONK": _forecast("BONK", 0.01),
        "SOL": _forecast("SOL", 0.02),
    }
    risks = {
        "BONK": _risk("BONK", -0.1),
        "SOL": _risk("SOL", -0.05),
    }
    costs = {
        "BONK": _cost("BONK", -10.0),
        "SOL": _cost("SOL", -5.0),
    }
    liq = {"BONK": 1e6, "SOL": 2e6}
    with caplog.at_level(logging.INFO, logger="app.pareto.assembler"):
        out = assemble_candidate_scores(
            forecasts, risks, costs, liq, 1000.0, routing
        )
    assert [c.symbol for c in out] == ["SOL"]
    assert "shock_regime" in caplog.text


def test_included_symbol_assembled_correctly() -> None:
    """
    Hand math:
      gross_cost_bps = (-(-50.0) / 1000.0) * 10_000 = 500.0
      effective_cost_bps = 500.0 - 0.0 = 500.0
    """
    routing = _routing((_valid_adjustment(symbol="SOL"),))
    forecasts = {"SOL": _forecast("SOL", 0.005)}
    risks = {"SOL": _risk("SOL", -0.04)}
    costs = {"SOL": _cost("SOL", -50.0)}
    liq = {"SOL": 5_000_000.0}
    out = assemble_candidate_scores(
        forecasts, risks, costs, liq, 1000.0, routing
    )
    assert len(out) == 1
    c = out[0]
    assert c.symbol == "SOL"
    assert c.expected_return_120s == 0.005
    assert c.cvar_95_120s == -0.04
    assert c.effective_cost_bps == pytest.approx(500.0)
    assert c.liquidity_usd == 5_000_000.0


def test_positive_bias_lowers_effective_cost() -> None:
    """effective_cost_bps = 500 - 30 = 470 when router adds +30 bps preference."""
    routing = _routing(
        (
            _valid_adjustment(
                symbol="SOL",
                score_bias_bps=30.0,
                bias_reasons=("router_prefers",),
            ),
        )
    )
    forecasts = {"SOL": _forecast("SOL", 0.005)}
    risks = {"SOL": _risk("SOL", -0.04)}
    costs = {"SOL": _cost("SOL", -50.0)}
    liq = {"SOL": 5_000_000.0}
    out = assemble_candidate_scores(
        forecasts, risks, costs, liq, 1000.0, routing
    )
    assert out[0].effective_cost_bps == pytest.approx(470.0)


def test_negative_bias_raises_effective_cost() -> None:
    """effective_cost_bps = 500 - (-40) = 540."""
    routing = _routing(
        (
            _valid_adjustment(
                symbol="SOL",
                score_bias_bps=-40.0,
                bias_reasons=("penalty",),
            ),
        )
    )
    forecasts = {"SOL": _forecast("SOL", 0.005)}
    risks = {"SOL": _risk("SOL", -0.04)}
    costs = {"SOL": _cost("SOL", -50.0)}
    liq = {"SOL": 5_000_000.0}
    out = assemble_candidate_scores(
        forecasts, risks, costs, liq, 1000.0, routing
    )
    assert out[0].effective_cost_bps == pytest.approx(540.0)


def test_cvar_confidence_mismatch_raises() -> None:
    """120s tail risk at 99% confidence → ValueError(symbol)."""
    routing = _routing((_valid_adjustment(symbol="SOL"),))
    forecasts = {"SOL": _forecast("SOL", 0.0)}
    bad_tail = _tail_120(-0.03, confidence_level=0.99, var=-0.02)
    risks = {
        "SOL": MultiHorizonRiskEstimate(
            symbol="SOL",
            position_value_usd=1000.0,
            estimates={120.0: bad_tail},
        )
    }
    costs = {"SOL": _cost("SOL", -1.0)}
    liq = {"SOL": 1e6}
    with pytest.raises(ValueError, match="SOL"):
        assemble_candidate_scores(
            forecasts, risks, costs, liq, 1000.0, routing
        )


def test_missing_forecast_skips_symbol(caplog: pytest.LogCaptureFixture) -> None:
    routing = _routing((_valid_adjustment(symbol="SOL"),))
    forecasts: dict[str, MultiHorizonForecast] = {}
    risks = {"SOL": _risk("SOL", -0.02)}
    costs = {"SOL": _cost("SOL", -10.0)}
    liq = {"SOL": 1e6}
    with caplog.at_level(logging.WARNING, logger="app.pareto.assembler"):
        out = assemble_candidate_scores(
            forecasts, risks, costs, liq, 1000.0, routing
        )
    assert out == []
    assert "forecasts" in caplog.text and "SOL" in caplog.text


def test_missing_cost_skips_symbol(caplog: pytest.LogCaptureFixture) -> None:
    routing = _routing((_valid_adjustment(symbol="SOL"),))
    forecasts = {"SOL": _forecast("SOL", 0.01)}
    risks = {"SOL": _risk("SOL", -0.02)}
    costs: dict[str, MultiHorizonCostEstimate] = {}
    liq = {"SOL": 1e6}
    with caplog.at_level(logging.WARNING, logger="app.pareto.assembler"):
        out = assemble_candidate_scores(
            forecasts, risks, costs, liq, 1000.0, routing
        )
    assert out == []
    assert "costs" in caplog.text and "SOL" in caplog.text


def test_missing_liquidity_skips_symbol(caplog: pytest.LogCaptureFixture) -> None:
    routing = _routing((_valid_adjustment(symbol="SOL"),))
    forecasts = {"SOL": _forecast("SOL", 0.01)}
    risks = {"SOL": _risk("SOL", -0.02)}
    costs = {"SOL": _cost("SOL", -10.0)}
    liq: dict[str, float] = {}
    with caplog.at_level(logging.WARNING, logger="app.pareto.assembler"):
        out = assemble_candidate_scores(
            forecasts, risks, costs, liq, 1000.0, routing
        )
    assert out == []
    assert "liquidity_usd_by_symbol" in caplog.text and "SOL" in caplog.text


def test_trade_size_zero_raises() -> None:
    routing = _routing((_valid_adjustment(symbol="SOL"),))
    with pytest.raises(ValueError, match="positive"):
        assemble_candidate_scores({}, {}, {}, {}, 0.0, routing)


def test_iteration_order_matches_router_included() -> None:
    symbols = ("BONK", "SOL", "USDC", "JUP", "WIF")
    adjs = tuple(_valid_adjustment(symbol=s) for s in symbols)
    routing = _routing(adjs)
    rng_vals = {"BONK": 0.1, "SOL": 0.2, "USDC": 0.3, "JUP": 0.4, "WIF": 0.5}
    forecasts = {s: _forecast(s, rng_vals[s]) for s in symbols}
    risks = {s: _risk(s, -0.01 * (1 + rng_vals[s])) for s in symbols}
    costs = {s: _cost(s, -float(10 + 5 * rng_vals[s])) for s in symbols}
    liq = {s: 1e6 * (1 + rng_vals[s]) for s in symbols}
    out = assemble_candidate_scores(
        forecasts, risks, costs, liq, 1000.0, routing
    )
    assert [c.symbol for c in out] == list(symbols)


def test_sign_convention_cost_negation() -> None:
    """
    Cost is reported as <=0; assembler negates so effective_cost_bps is
    positive-larger-is-worse, matching DIMENSION_DIRECTIONS[effective_cost_bps]=-1.
    gross_cost_bps = (-(-100)/1000)*10000 = 1000
    """
    routing = _routing((_valid_adjustment(symbol="SOL"),))
    forecasts = {"SOL": _forecast("SOL", 0.0)}
    risks = {"SOL": _risk("SOL", -0.01)}
    costs = {"SOL": _cost("SOL", -100.0)}
    liq = {"SOL": 1e6}
    out = assemble_candidate_scores(
        forecasts, risks, costs, liq, 1000.0, routing
    )
    assert out[0].effective_cost_bps == pytest.approx(1000.0)
