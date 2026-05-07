"""Verify the forecaster data contracts behave correctly on hand-built inputs.

These tests do NOT use any actual forecasting model — they verify the
shape, validation, and lookup behavior of the dataclasses themselves.
"""

import pytest

from app.forecasting.base import (
    DEFAULT_HORIZONS,
    HorizonForecast,
    MultiHorizonForecast,
)


# -----------------------------------------------------------------------------
# HorizonForecast validation
# -----------------------------------------------------------------------------


def test_horizon_forecast_constructs_with_valid_inputs() -> None:
    f = HorizonForecast(
        horizon_seconds=30.0,
        predicted_return=-0.001,
        predicted_volatility=0.01,
        confidence_lower_95=-0.025,
        confidence_upper_95=0.023,
    )
    assert f.horizon_seconds == 30.0
    assert f.predicted_return == -0.001


def test_horizon_forecast_rejects_zero_horizon() -> None:
    with pytest.raises(ValueError, match="horizon_seconds"):
        HorizonForecast(
            horizon_seconds=0.0,
            predicted_return=0.0,
            predicted_volatility=0.01,
            confidence_lower_95=-0.02,
            confidence_upper_95=0.02,
        )


def test_horizon_forecast_rejects_negative_horizon() -> None:
    with pytest.raises(ValueError, match="horizon_seconds"):
        HorizonForecast(
            horizon_seconds=-5.0,
            predicted_return=0.0,
            predicted_volatility=0.01,
            confidence_lower_95=-0.02,
            confidence_upper_95=0.02,
        )


def test_horizon_forecast_rejects_negative_volatility() -> None:
    with pytest.raises(ValueError, match="predicted_volatility"):
        HorizonForecast(
            horizon_seconds=30.0,
            predicted_return=0.0,
            predicted_volatility=-0.01,
            confidence_lower_95=-0.02,
            confidence_upper_95=0.02,
        )


def test_horizon_forecast_rejects_inverted_confidence_band() -> None:
    """If lower > upper, that's nonsensical — reject."""
    with pytest.raises(ValueError, match="confidence"):
        HorizonForecast(
            horizon_seconds=30.0,
            predicted_return=0.0,
            predicted_volatility=0.01,
            confidence_lower_95=0.05,
            confidence_upper_95=-0.05,
        )


def test_horizon_forecast_allows_zero_volatility() -> None:
    """A perfect-prediction forecast (zero vol) is unusual but mathematically valid."""
    f = HorizonForecast(
        horizon_seconds=30.0,
        predicted_return=0.0,
        predicted_volatility=0.0,
        confidence_lower_95=0.0,
        confidence_upper_95=0.0,
    )
    assert f.predicted_volatility == 0.0


# -----------------------------------------------------------------------------
# MultiHorizonForecast construction and lookup
# -----------------------------------------------------------------------------


def _make_horizon(h: float, ret: float = 0.0, vol: float = 0.01) -> HorizonForecast:
    return HorizonForecast(
        horizon_seconds=h,
        predicted_return=ret,
        predicted_volatility=vol,
        confidence_lower_95=ret - 2 * vol,
        confidence_upper_95=ret + 2 * vol,
    )


def test_multihorizon_constructs_with_default_horizons() -> None:
    horizons = {h: _make_horizon(h) for h in DEFAULT_HORIZONS}
    mhf = MultiHorizonForecast(symbol="SOL", horizons=horizons)
    assert mhf.symbol == "SOL"
    assert len(mhf.horizons) == 3


def test_multihorizon_at_returns_correct_horizon() -> None:
    horizons = {h: _make_horizon(h, ret=h / 1000.0) for h in DEFAULT_HORIZONS}
    mhf = MultiHorizonForecast(symbol="SOL", horizons=horizons)
    f30 = mhf.at(30.0)
    assert f30.horizon_seconds == 30.0
    assert abs(f30.predicted_return - 0.030) < 1e-12


def test_multihorizon_at_unknown_horizon_raises() -> None:
    horizons = {h: _make_horizon(h) for h in DEFAULT_HORIZONS}
    mhf = MultiHorizonForecast(symbol="SOL", horizons=horizons)
    with pytest.raises(KeyError, match="not in this forecast"):
        mhf.at(60.0)


def test_multihorizon_horizon_seconds_list_is_sorted() -> None:
    """Out-of-order horizons in the dict should still come out sorted."""
    horizons = {120.0: _make_horizon(120.0), 5.0: _make_horizon(5.0), 30.0: _make_horizon(30.0)}
    mhf = MultiHorizonForecast(symbol="SOL", horizons=horizons)
    assert mhf.horizon_seconds_list() == [5.0, 30.0, 120.0]


def test_multihorizon_rejects_empty_horizons() -> None:
    with pytest.raises(ValueError, match="at least one forecast"):
        MultiHorizonForecast(symbol="SOL", horizons={})


def test_multihorizon_rejects_empty_symbol() -> None:
    horizons = {30.0: _make_horizon(30.0)}
    with pytest.raises(ValueError, match="non-empty string"):
        MultiHorizonForecast(symbol="", horizons=horizons)


def test_multihorizon_rejects_key_mismatch() -> None:
    """If the dict key disagrees with the forecast's own horizon_seconds, that's a bug."""
    horizons = {30.0: _make_horizon(60.0)}  # key says 30, forecast says 60
    with pytest.raises(ValueError, match="does not match"):
        MultiHorizonForecast(symbol="SOL", horizons=horizons)


def test_default_horizons_are_the_ramhd_spec() -> None:
    """Sanity: confirm we're shipping the horizons the original spec calls for."""
    assert DEFAULT_HORIZONS == (5.0, 30.0, 120.0)
