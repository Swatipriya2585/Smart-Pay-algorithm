"""Verify the routing data contracts behave correctly on hand-built inputs."""

import pytest

from app.regime.base import RegimeEstimate
from app.routing.base import MultiTokenRoutingDecision, RoutingAdjustment


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


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
    symbol: str = "SOL",
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


# -----------------------------------------------------------------------------
# RoutingAdjustment validation
# -----------------------------------------------------------------------------


def test_adjustment_constructs_with_no_exclusion_no_bias() -> None:
    a = _valid_adjustment()
    assert a.symbol == "SOL"
    assert not a.excluded


def test_adjustment_constructs_with_exclusion_and_reason() -> None:
    a = _valid_adjustment(
        excluded=True,
        exclusion_reason="CVaR exceeds threshold",
    )
    assert a.excluded
    assert a.exclusion_reason == "CVaR exceeds threshold"


def test_adjustment_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        _valid_adjustment(symbol="")


def test_adjustment_rejects_excluded_without_reason() -> None:
    with pytest.raises(ValueError, match="exclusion_reason"):
        _valid_adjustment(excluded=True, exclusion_reason=None)


def test_adjustment_rejects_excluded_with_empty_reason() -> None:
    with pytest.raises(ValueError, match="exclusion_reason"):
        _valid_adjustment(excluded=True, exclusion_reason="")


def test_adjustment_rejects_not_excluded_with_reason() -> None:
    """If not excluded, exclusion_reason must be None."""
    with pytest.raises(ValueError, match="exclusion_reason"):
        _valid_adjustment(excluded=False, exclusion_reason="some reason")


def test_adjustment_with_bias_requires_reasons() -> None:
    with pytest.raises(ValueError, match="bias_reasons"):
        _valid_adjustment(score_bias_bps=100.0, bias_reasons=())


def test_adjustment_with_reasons_requires_bias() -> None:
    with pytest.raises(ValueError, match="score_bias_bps"):
        _valid_adjustment(score_bias_bps=0.0, bias_reasons=("stress regime",))


def test_adjustment_with_negative_bias_and_reasons_is_valid() -> None:
    """Negative bias (penalty) is valid — represents a discouragement."""
    a = _valid_adjustment(
        score_bias_bps=-100.0,
        bias_reasons=("shallow liquidity penalty",),
    )
    assert a.score_bias_bps == -100.0


def test_adjustment_can_have_both_exclusion_and_bias() -> None:
    """A token can be excluded AND carry a bias (the bias is then ignored

    by the selector, but recorded for audit)."""
    a = _valid_adjustment(
        excluded=True,
        exclusion_reason="CVaR breach",
        score_bias_bps=50.0,
        bias_reasons=("stress stablecoin preference",),
    )
    assert a.excluded
    assert a.score_bias_bps == 50.0


def test_adjustment_immutability() -> None:
    a = _valid_adjustment()
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        a.score_bias_bps = 999.0  # type: ignore[misc]


# -----------------------------------------------------------------------------
# MultiTokenRoutingDecision validation
# -----------------------------------------------------------------------------


def test_decision_constructs_with_valid_adjustments() -> None:
    adjs = (
        _valid_adjustment(symbol="SOL"),
        _valid_adjustment(symbol="USDC"),
    )
    d = MultiTokenRoutingDecision(adjustments=adjs, regime=_valid_regime())
    assert len(d.adjustments) == 2


def test_decision_rejects_empty_adjustments() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MultiTokenRoutingDecision(adjustments=(), regime=_valid_regime())


def test_decision_rejects_duplicate_symbols() -> None:
    adjs = (
        _valid_adjustment(symbol="SOL"),
        _valid_adjustment(symbol="SOL"),
    )
    with pytest.raises(ValueError, match="Duplicate symbol"):
        MultiTokenRoutingDecision(adjustments=adjs, regime=_valid_regime())


def test_decision_for_symbol_returns_correct_adjustment() -> None:
    adjs = (
        _valid_adjustment(symbol="SOL", score_bias_bps=100.0, bias_reasons=("a",)),
        _valid_adjustment(symbol="USDC", score_bias_bps=200.0, bias_reasons=("b",)),
    )
    d = MultiTokenRoutingDecision(adjustments=adjs, regime=_valid_regime())
    assert d.for_symbol("USDC").score_bias_bps == 200.0


def test_decision_for_symbol_unknown_raises() -> None:
    adjs = (_valid_adjustment(symbol="SOL"),)
    d = MultiTokenRoutingDecision(adjustments=adjs, regime=_valid_regime())
    with pytest.raises(KeyError, match="not in routing decision"):
        d.for_symbol("DOGE")


def test_decision_included_symbols_filters_excluded() -> None:
    adjs = (
        _valid_adjustment(symbol="SOL"),
        _valid_adjustment(symbol="BONK", excluded=True, exclusion_reason="CVaR"),
        _valid_adjustment(symbol="USDC"),
    )
    d = MultiTokenRoutingDecision(adjustments=adjs, regime=_valid_regime())
    assert d.included_symbols() == ("SOL", "USDC")


def test_decision_excluded_symbols_returns_excluded_only() -> None:
    adjs = (
        _valid_adjustment(symbol="SOL"),
        _valid_adjustment(symbol="BONK", excluded=True, exclusion_reason="CVaR"),
        _valid_adjustment(symbol="USDC"),
    )
    d = MultiTokenRoutingDecision(adjustments=adjs, regime=_valid_regime())
    assert d.excluded_symbols() == ("BONK",)
