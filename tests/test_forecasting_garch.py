"""GARCH forecaster tests.

Two tolerance regimes:

- Functional claims (shapes, monotonicity, bounds): tight, ~1e-9.
- Statistical claims (forecast vs. realized vol): loose, 15-30% relative.

GARCH is a fitted noisy estimator; tight tolerances would flake.
"""

import math

import numpy as np
import pytest

from app.forecasting.base import DEFAULT_HORIZONS, HorizonForecast, MultiHorizonForecast
from app.forecasting.garch import GARCHConfig, GARCHForecaster
from app.market_data.calibration import BlendingConfig, Calibration
from app.market_data.mock import MockConfig, MockMarketData


# -----------------------------------------------------------------------------
# Functional / structural tests
# -----------------------------------------------------------------------------


def _fetch(symbol: str, n_obs: int = 1440, seed: int = 7) -> "TokenMarketData":  # type: ignore[name-defined]
    mock = MockMarketData(config=MockConfig(n_observations=n_obs, seed=seed))
    return mock.fetch([symbol])[0]


def test_forecast_returns_multihorizon_with_default_horizons() -> None:
    f = GARCHForecaster(calibration=Calibration())
    out = f.forecast(_fetch("SOL"))
    assert isinstance(out, MultiHorizonForecast)
    assert out.symbol == "SOL"
    assert sorted(out.horizons.keys()) == [5.0, 30.0, 120.0]


def test_forecast_per_horizon_returns_horizon_forecast() -> None:
    f = GARCHForecaster(calibration=Calibration())
    out = f.forecast(_fetch("SOL"))
    for h in DEFAULT_HORIZONS:
        hf = out.at(h)
        assert isinstance(hf, HorizonForecast)
        assert hf.horizon_seconds == h
        assert math.isfinite(hf.predicted_return)
        assert math.isfinite(hf.predicted_volatility)
        assert hf.predicted_volatility >= 0


def test_forecast_volatility_monotone_in_horizon() -> None:
    """Variance is additive over time, so vol must increase with horizon length.

    This must hold for every token in our universe — if it ever doesn't,
    something is fundamentally wrong with the aggregation math.
    """
    f = GARCHForecaster(calibration=Calibration())
    for symbol in ("SOL", "JUP", "BONK", "BRETT", "WIF", "AERO", "PYTH"):
        out = f.forecast(_fetch(symbol))
        v5 = out.at(5.0).predicted_volatility
        v30 = out.at(30.0).predicted_volatility
        v120 = out.at(120.0).predicted_volatility
        assert v5 < v30 < v120, (
            f"{symbol}: vol not monotone (5s={v5:.6e}, 30s={v30:.6e}, 120s={v120:.6e})"
        )


def test_forecast_volatility_strictly_monotone_for_close_horizons() -> None:
    """5s vs 6s should produce strictly different forecasts.

    With the original integer-step rounding bug, both would round to 0 steps
    (clamped to 1) and produce identical output. This test verifies the
    fractional-step interpolation actually works.
    """
    f = GARCHForecaster(calibration=Calibration())
    out = f.forecast(_fetch("SOL"), horizons=(5.0, 6.0, 10.0, 30.0))
    v5 = out.at(5.0).predicted_volatility
    v6 = out.at(6.0).predicted_volatility
    v10 = out.at(10.0).predicted_volatility
    v30 = out.at(30.0).predicted_volatility
    assert v5 < v6 < v10 < v30


def test_forecast_confidence_band_brackets_predicted_return() -> None:
    f = GARCHForecaster(calibration=Calibration())
    out = f.forecast(_fetch("SOL"))
    for h in DEFAULT_HORIZONS:
        hf = out.at(h)
        assert hf.confidence_lower_95 <= hf.predicted_return
        assert hf.predicted_return <= hf.confidence_upper_95


def test_forecast_handles_custom_horizons() -> None:
    f = GARCHForecaster(calibration=Calibration())
    out = f.forecast(_fetch("SOL"), horizons=(10.0, 60.0))
    assert sorted(out.horizons.keys()) == [10.0, 60.0]


def test_forecast_rejects_empty_horizons() -> None:
    f = GARCHForecaster(calibration=Calibration())
    with pytest.raises(ValueError, match="at least one"):
        f.forecast(_fetch("SOL"), horizons=())


def test_forecast_rejects_too_short_path() -> None:
    f = GARCHForecaster(calibration=Calibration())
    short = _fetch("SOL", n_obs=20)
    with pytest.raises(ValueError, match=">= 50 observations"):
        f.forecast(short)


# -----------------------------------------------------------------------------
# Stablecoin handling
# -----------------------------------------------------------------------------


def test_stablecoin_forecast_has_near_zero_volatility() -> None:
    """USDC is calibrated as a stablecoin; the forecast should reflect that."""
    f = GARCHForecaster(calibration=Calibration())
    out = f.forecast(_fetch("USDC"))
    # Stablecoin per-step vol is tiny (~5 bps over 24h calibration).
    # The aggregated 120s vol should still be far below any non-stable.
    sol_out = f.forecast(_fetch("SOL"))
    usdc_120 = out.at(120.0).predicted_volatility
    sol_120 = sol_out.at(120.0).predicted_volatility
    assert usdc_120 < sol_120 / 50  # at least 50x smaller


def test_stablecoin_detected_by_calibration_flag() -> None:
    f = GARCHForecaster(calibration=Calibration())
    cal = Calibration()
    assert cal.get("USDC").is_stablecoin is True
    out = f.forecast(_fetch("USDC"))
    # Confidence band should be very tight around zero for a stablecoin.
    h120 = out.at(120.0)
    band_width = h120.confidence_upper_95 - h120.confidence_lower_95
    assert band_width < 0.01  # less than 1% over a 2-min window


def test_stablecoin_detected_without_calibration_via_threshold() -> None:
    """Without a Calibration handle, the heuristic threshold should kick in."""
    f = GARCHForecaster(calibration=None)
    # USDC path has tiny per-step std, well below the threshold.
    out = f.forecast(_fetch("USDC"))
    assert out.at(120.0).predicted_volatility < 1e-3


# -----------------------------------------------------------------------------
# Statistical sanity (loose tolerances)
# -----------------------------------------------------------------------------


def test_garch_forecast_in_same_order_of_magnitude_as_realized() -> None:
    """SOL's GARCH 120s forecast should be in the right ballpark of realized vol.

    "Right ballpark" = within a factor of 3. Tighter would flake; looser
    would miss real bugs.
    """
    n_obs = 1440
    interval = 60.0
    f = GARCHForecaster(calibration=Calibration())
    sol_data = _fetch("SOL", n_obs=n_obs)

    out = f.forecast(sol_data)
    forecast_120 = out.at(120.0).predicted_volatility

    # Realized 120s std-dev: from per-minute returns.
    log_returns = np.diff(np.log(sol_data.path.prices_usd))
    per_step_std = float(np.std(log_returns, ddof=1))
    n_steps_120 = int(round(120.0 / interval))
    realized_120 = per_step_std * math.sqrt(n_steps_120)

    ratio = forecast_120 / realized_120
    assert 0.33 < ratio < 3.0, (
        f"forecast {forecast_120:.4e} vs realized {realized_120:.4e}, ratio {ratio:.2f}"
    )


def test_higher_vol_token_has_higher_forecast_than_lower_vol_token() -> None:
    """BONK (memecoin, high vol) should forecast higher vol than SOL at every horizon."""
    f = GARCHForecaster(calibration=Calibration())
    sol_out = f.forecast(_fetch("SOL"))
    bonk_out = f.forecast(_fetch("BONK"))
    for h in DEFAULT_HORIZONS:
        assert bonk_out.at(h).predicted_volatility > sol_out.at(h).predicted_volatility


# -----------------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------------


def test_garch_config_validates_confidence_level() -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        GARCHConfig(confidence_level=1.5)
    with pytest.raises(ValueError, match="confidence_level"):
        GARCHConfig(confidence_level=0.0)


def test_garch_config_validates_threshold_non_negative() -> None:
    with pytest.raises(ValueError, match="stablecoin_per_step_threshold"):
        GARCHConfig(stablecoin_per_step_threshold=-1.0)


# -----------------------------------------------------------------------------
# End-to-end smoke
# -----------------------------------------------------------------------------


def test_full_universe_forecast_smoke() -> None:
    """All 8 calibrated tokens should forecast cleanly without errors."""
    f = GARCHForecaster(calibration=Calibration())
    universe = ["SOL", "USDC", "PYTH", "AERO", "JUP", "BRETT", "WIF", "BONK"]
    for sym in universe:
        out = f.forecast(_fetch(sym))
        assert out.symbol == sym
        for h in DEFAULT_HORIZONS:
            hf = out.at(h)
            assert math.isfinite(hf.predicted_volatility)
            assert hf.predicted_volatility >= 0
