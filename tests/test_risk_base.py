"""Verify the tail-risk data contracts behave correctly on hand-built inputs."""

import pytest

from app.risk.base import MultiHorizonRiskEstimate, TailRiskEstimate


# -----------------------------------------------------------------------------
# TailRiskEstimate validation
# -----------------------------------------------------------------------------


def test_tail_risk_estimate_constructs_with_valid_inputs() -> None:
    e = TailRiskEstimate(
        horizon_seconds=120.0,
        confidence_level=0.95,
        var=-0.02,
        cvar=-0.03,
        var_dollar=-20.0,
        cvar_dollar=-30.0,
        n_samples=10_000,
    )
    assert e.horizon_seconds == 120.0
    assert e.confidence_level == 0.95
    assert e.cvar < e.var


def test_tail_risk_estimate_rejects_zero_horizon() -> None:
    with pytest.raises(ValueError, match="horizon_seconds"):
        TailRiskEstimate(
            horizon_seconds=0.0,
            confidence_level=0.95,
            var=-0.01,
            cvar=-0.02,
            var_dollar=-10.0,
            cvar_dollar=-20.0,
            n_samples=10_000,
        )


def test_tail_risk_estimate_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        TailRiskEstimate(
            horizon_seconds=120.0,
            confidence_level=1.5,
            var=-0.01,
            cvar=-0.02,
            var_dollar=-10.0,
            cvar_dollar=-20.0,
            n_samples=10_000,
        )
    with pytest.raises(ValueError, match="confidence_level"):
        TailRiskEstimate(
            horizon_seconds=120.0,
            confidence_level=0.0,
            var=-0.01,
            cvar=-0.02,
            var_dollar=-10.0,
            cvar_dollar=-20.0,
            n_samples=10_000,
        )


def test_tail_risk_estimate_enforces_cvar_at_least_as_severe_as_var() -> None:
    """CVaR is the average of the tail BEYOND VaR; it cannot be milder."""
    with pytest.raises(ValueError, match="cvar"):
        TailRiskEstimate(
            horizon_seconds=120.0,
            confidence_level=0.95,
            var=-0.03,
            cvar=-0.02,  # less severe than VaR — invalid
            var_dollar=-30.0,
            cvar_dollar=-20.0,
            n_samples=10_000,
        )


def test_tail_risk_estimate_allows_cvar_equal_to_var() -> None:
    """Edge case: CVaR == VaR is mathematically possible (degenerate distribution)."""
    e = TailRiskEstimate(
        horizon_seconds=120.0,
        confidence_level=0.95,
        var=-0.02,
        cvar=-0.02,
        var_dollar=-20.0,
        cvar_dollar=-20.0,
        n_samples=10_000,
    )
    assert e.var == e.cvar


def test_tail_risk_estimate_rejects_negative_n_samples() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        TailRiskEstimate(
            horizon_seconds=120.0,
            confidence_level=0.95,
            var=-0.01,
            cvar=-0.02,
            var_dollar=-10.0,
            cvar_dollar=-20.0,
            n_samples=-1,
        )


def test_tail_risk_estimate_rejects_sign_disagreement() -> None:
    """If returns are negative, dollar amounts must also be non-positive."""
    with pytest.raises(ValueError, match="var_dollar"):
        TailRiskEstimate(
            horizon_seconds=120.0,
            confidence_level=0.95,
            var=-0.02,
            cvar=-0.03,
            var_dollar=20.0,  # positive while var is negative — invalid
            cvar_dollar=-30.0,
            n_samples=10_000,
        )


def test_tail_risk_estimate_allows_zero_returns() -> None:
    """A perfectly safe token (e.g. perfect stablecoin) has all zeros."""
    e = TailRiskEstimate(
        horizon_seconds=120.0,
        confidence_level=0.95,
        var=0.0,
        cvar=0.0,
        var_dollar=0.0,
        cvar_dollar=0.0,
        n_samples=10_000,
    )
    assert e.var_dollar == 0.0


def test_tail_risk_estimate_allows_n_samples_zero() -> None:
    """Analytical estimators that don't sample should report n_samples = 0."""
    e = TailRiskEstimate(
        horizon_seconds=120.0,
        confidence_level=0.95,
        var=-0.01,
        cvar=-0.015,
        var_dollar=-10.0,
        cvar_dollar=-15.0,
        n_samples=0,
    )
    assert e.n_samples == 0


# -----------------------------------------------------------------------------
# MultiHorizonRiskEstimate construction and lookup
# -----------------------------------------------------------------------------


def _make_estimate(
    h: float,
    var: float = -0.01,
    cvar: float = -0.02,
    pos: float = 1000.0,
) -> TailRiskEstimate:
    return TailRiskEstimate(
        horizon_seconds=h,
        confidence_level=0.95,
        var=var,
        cvar=cvar,
        var_dollar=var * pos,
        cvar_dollar=cvar * pos,
        n_samples=10_000,
    )


def test_multihorizon_constructs_with_valid_inputs() -> None:
    estimates = {h: _make_estimate(h) for h in (5.0, 30.0, 120.0)}
    mhre = MultiHorizonRiskEstimate(
        symbol="SOL",
        position_value_usd=1000.0,
        estimates=estimates,
    )
    assert mhre.symbol == "SOL"
    assert len(mhre.estimates) == 3


def test_multihorizon_at_returns_correct_horizon() -> None:
    estimates = {h: _make_estimate(h, var=-h / 1000, cvar=-h / 500) for h in (5.0, 30.0, 120.0)}
    mhre = MultiHorizonRiskEstimate(
        symbol="SOL",
        position_value_usd=1000.0,
        estimates=estimates,
    )
    assert abs(mhre.at(30.0).var - (-0.030)) < 1e-12


def test_multihorizon_at_unknown_horizon_raises() -> None:
    estimates = {30.0: _make_estimate(30.0)}
    mhre = MultiHorizonRiskEstimate(symbol="SOL", position_value_usd=1000.0, estimates=estimates)
    with pytest.raises(KeyError, match="not in this estimate"):
        mhre.at(60.0)


def test_multihorizon_rejects_empty_estimates() -> None:
    with pytest.raises(ValueError, match="at least one horizon"):
        MultiHorizonRiskEstimate(symbol="SOL", position_value_usd=1000.0, estimates={})


def test_multihorizon_rejects_empty_symbol() -> None:
    estimates = {30.0: _make_estimate(30.0)}
    with pytest.raises(ValueError, match="non-empty string"):
        MultiHorizonRiskEstimate(symbol="", position_value_usd=1000.0, estimates=estimates)


def test_multihorizon_rejects_negative_position_value() -> None:
    estimates = {30.0: _make_estimate(30.0)}
    with pytest.raises(ValueError, match="position_value_usd"):
        MultiHorizonRiskEstimate(
            symbol="SOL", position_value_usd=-100.0, estimates=estimates
        )


def test_multihorizon_rejects_horizon_key_mismatch() -> None:
    estimates = {30.0: _make_estimate(60.0)}  # key 30, estimate says 60
    with pytest.raises(ValueError, match="does not match"):
        MultiHorizonRiskEstimate(
            symbol="SOL", position_value_usd=1000.0, estimates=estimates
        )


def test_horizon_seconds_list_is_sorted() -> None:
    estimates = {120.0: _make_estimate(120.0), 5.0: _make_estimate(5.0), 30.0: _make_estimate(30.0)}
    mhre = MultiHorizonRiskEstimate(
        symbol="SOL", position_value_usd=1000.0, estimates=estimates
    )
    assert mhre.horizon_seconds_list() == [5.0, 30.0, 120.0]


def test_worst_cvar_dollar_returns_most_negative() -> None:
    """The risk-adaptive router uses this method — it must return the worst CVaR.

    Each horizon needs its own var that is at least as mild as its cvar
    (cvar <= var), so the constructor's invariant is satisfied. The point
    of the test is the worst-CVaR aggregation, not the individual values.
    """
    estimates = {
        5.0:   _make_estimate(5.0,   var=-0.003, cvar=-0.005, pos=1000.0),   # cvar_dollar = -5
        30.0:  _make_estimate(30.0,  var=-0.010, cvar=-0.015, pos=1000.0),   # cvar_dollar = -15
        120.0: _make_estimate(120.0, var=-0.030, cvar=-0.040, pos=1000.0),   # cvar_dollar = -40
    }
    mhre = MultiHorizonRiskEstimate(
        symbol="SOL", position_value_usd=1000.0, estimates=estimates
    )
    assert mhre.worst_cvar_dollar() == -40.0


def test_worst_cvar_dollar_returns_zero_for_safe_token() -> None:
    """If all CVaRs are zero (e.g. stablecoin), worst is zero, not None."""
    estimates = {
        h: _make_estimate(h, var=0.0, cvar=0.0, pos=1000.0)
        for h in (5.0, 30.0, 120.0)
    }
    mhre = MultiHorizonRiskEstimate(
        symbol="USDC", position_value_usd=1000.0, estimates=estimates
    )
    assert mhre.worst_cvar_dollar() == 0.0
