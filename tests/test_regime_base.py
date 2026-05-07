"""Verify the regime data contracts behave correctly on hand-built inputs."""

import pytest

from app.regime.base import RegimeEstimate


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _valid_estimate(
    symbol: str = "SOL",
    regime: str = "calm",
    confidence: float = 0.8,
    realized: float = 0.001,
    baseline: float = 0.0015,
    ratio: float = 0.667,
) -> RegimeEstimate:
    return RegimeEstimate(
        symbol=symbol,
        regime=regime,  # type: ignore[arg-type]
        confidence=confidence,
        realized_volatility=realized,
        baseline_volatility=baseline,
        ratio=ratio,
    )


# -----------------------------------------------------------------------------
# RegimeEstimate validation
# -----------------------------------------------------------------------------


def test_regime_estimate_constructs_with_valid_inputs() -> None:
    e = _valid_estimate()
    assert e.symbol == "SOL"
    assert e.regime == "calm"
    assert e.confidence == 0.8


def test_regime_estimate_accepts_all_three_regimes() -> None:
    for r in ("calm", "stress", "shock"):
        e = _valid_estimate(regime=r)
        assert e.regime == r


def test_regime_estimate_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        _valid_estimate(symbol="")


def test_regime_estimate_rejects_invalid_regime() -> None:
    with pytest.raises(ValueError, match="regime must be"):
        _valid_estimate(regime="boom")


def test_regime_estimate_rejects_confidence_above_one() -> None:
    with pytest.raises(ValueError, match="confidence"):
        _valid_estimate(confidence=1.5)


def test_regime_estimate_rejects_negative_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        _valid_estimate(confidence=-0.1)


def test_regime_estimate_allows_confidence_at_bounds() -> None:
    """Edge case: 0.0 and 1.0 are both valid."""
    e_zero = _valid_estimate(confidence=0.0)
    e_one = _valid_estimate(confidence=1.0)
    assert e_zero.confidence == 0.0
    assert e_one.confidence == 1.0


def test_regime_estimate_rejects_negative_realized_volatility() -> None:
    with pytest.raises(ValueError, match="realized_volatility"):
        _valid_estimate(realized=-0.001)


def test_regime_estimate_allows_zero_realized_volatility() -> None:
    """A perfectly stable path (zero vol) is unusual but mathematically valid."""
    e = _valid_estimate(realized=0.0)
    assert e.realized_volatility == 0.0


def test_regime_estimate_rejects_zero_baseline_volatility() -> None:
    """Baseline must be positive — division by zero in ratio computation."""
    with pytest.raises(ValueError, match="baseline_volatility"):
        _valid_estimate(baseline=0.0)


def test_regime_estimate_rejects_negative_baseline_volatility() -> None:
    with pytest.raises(ValueError, match="baseline_volatility"):
        _valid_estimate(baseline=-0.001)


def test_regime_estimate_rejects_negative_ratio() -> None:
    with pytest.raises(ValueError, match="ratio"):
        _valid_estimate(ratio=-0.5)


def test_regime_estimate_allows_zero_ratio() -> None:
    """Edge case: realized vol = 0 means ratio = 0, which is valid."""
    e = _valid_estimate(realized=0.0, ratio=0.0)
    assert e.ratio == 0.0


def test_regime_estimate_immutability() -> None:
    """RegimeEstimate is a frozen dataclass — fields cannot be reassigned."""
    e = _valid_estimate()
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        e.regime = "shock"  # type: ignore[misc]


def test_regime_estimate_high_ratio_can_pair_with_calm() -> None:
    """The dataclass doesn't enforce regime↔ratio consistency — that's the
    detector's job, not the data contract's. The contract just stores values."""
    # Wildly mismatched but each value is individually valid — should construct.
    e = _valid_estimate(regime="calm", ratio=5.0)
    assert e.regime == "calm"
    assert e.ratio == 5.0
