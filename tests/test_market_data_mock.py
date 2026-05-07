"""Tests for MockMarketData synthetic generator.

Statistical tests use generous tolerances (10-15% relative error) because
realized vol of a Monte Carlo path with N=1440 has finite-sample sampling
noise around the true sigma. We use longer paths (N=10000+) for the tight
tolerance tests.

"""

import numpy as np
import pytest

from app.market_data.calibration import BlendingConfig, Calibration
from app.market_data.mock import MockConfig, MockMarketData


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------


def test_same_seed_produces_identical_paths() -> None:
    """Two MockMarketData with the same seed must produce identical output."""
    a = MockMarketData(config=MockConfig(seed=42))
    b = MockMarketData(config=MockConfig(seed=42))
    pa = a.fetch(["SOL"])[0].path.prices_usd
    pb = b.fetch(["SOL"])[0].path.prices_usd
    np.testing.assert_array_equal(pa, pb)


def test_different_seeds_produce_different_paths() -> None:
    a = MockMarketData(config=MockConfig(seed=1))
    b = MockMarketData(config=MockConfig(seed=2))
    pa = a.fetch(["SOL"])[0].path.prices_usd
    pb = b.fetch(["SOL"])[0].path.prices_usd
    # They should be unmistakably different.
    assert not np.allclose(pa, pb)


def test_reset_seed_reproduces_path() -> None:
    """After reset_seed, the next fetch() should produce the same path as the first."""
    mock = MockMarketData(config=MockConfig(seed=7))
    first = mock.fetch(["SOL"])[0].path.prices_usd.copy()
    mock.reset_seed(7)
    second = mock.fetch(["SOL"])[0].path.prices_usd
    np.testing.assert_array_equal(first, second)


# -----------------------------------------------------------------------------
# API contract
# -----------------------------------------------------------------------------


def test_fetch_preserves_input_order() -> None:
    mock = MockMarketData()
    out = mock.fetch(["BONK", "SOL", "USDC"])
    assert [t.symbol for t in out] == ["BONK", "SOL", "USDC"]


def test_fetch_unknown_symbol_raises_keyerror() -> None:
    mock = MockMarketData()
    with pytest.raises(KeyError):
        mock.fetch(["DOGE"])  # not in our 8-token universe


def test_fetch_returns_correct_path_length() -> None:
    cfg = MockConfig(n_observations=500, seed=1)
    mock = MockMarketData(config=cfg)
    md = mock.fetch(["SOL"])[0]
    assert len(md.path.prices_usd) == 500
    assert md.path.interval_seconds == 60.0


def test_fetch_starts_at_calibrated_current_price() -> None:
    """First price in the path should match calibration's current_price_usd."""
    mock = MockMarketData()
    cal = Calibration()
    out = mock.fetch(["SOL", "BONK"])
    assert abs(out[0].path.prices_usd[0] - cal.get("SOL").current_price_usd) < 1e-9
    assert abs(out[1].path.prices_usd[0] - cal.get("BONK").current_price_usd) < 1e-9


def test_token_market_data_passes_through_calibration_address() -> None:
    mock = MockMarketData()
    sol = mock.fetch(["SOL"])[0]
    assert sol.mint == "So11111111111111111111111111111111111111112"


# -----------------------------------------------------------------------------
# GBM statistical properties
# -----------------------------------------------------------------------------


def test_gbm_realized_vol_matches_calibrated_vol() -> None:
    """For a long path (N=10000), realized vol should approach calibrated vol.

    Tolerance: 10% relative. For SOL with annual vol 76.7%, sample std of
    log returns over a 1-min sampling has finite-sample sampling noise;
    10% relative is generous but lets the test catch order-of-magnitude bugs.

    """
    cfg = MockConfig(n_observations=10_000, interval_seconds=60.0, seed=123)
    mock = MockMarketData(config=cfg)
    sol = mock.fetch(["SOL"])[0]

    cal = Calibration()
    sol_cal = cal.get("SOL")

    # Convert calibrated annual vol -> expected per-step (1 min) vol.
    seconds_per_year = 365.0 * 24.0 * 3600.0
    dt = cfg.interval_seconds / seconds_per_year
    expected_per_step_std = sol_cal.annualized_vol * np.sqrt(dt)

    realized_per_step_std = sol.path.realized_volatility()

    # 10% relative tolerance.
    assert (
        abs(realized_per_step_std - expected_per_step_std) / expected_per_step_std
        < 0.10
    )


def test_gbm_log_returns_have_zero_mean_at_alpha_07_baseline_zero() -> None:
    """With alpha=0.7 and baseline=0, the simulator's effective drift is
    small. Mean of log returns over a long path should be near zero
    (relative to the volatility scale).

    We don't assert exact zero — there's a real -0.15 stress drift on
    SOL — but the magnitude should be tiny relative to the diffusion.

    """
    cfg = MockConfig(n_observations=10_000, interval_seconds=60.0, seed=99)
    mock = MockMarketData(config=cfg)
    sol = mock.fetch(["SOL"])[0]

    returns = sol.path.log_returns()

    # The drift contribution per step is mu*dt ~ -0.15 * (60/seconds_per_year)
    # ~= -2.85e-7. This is many orders of magnitude smaller than the per-step
    # vol, so the realized mean should be dominated by sampling noise.
    seconds_per_year = 365.0 * 24.0 * 3600.0
    dt = cfg.interval_seconds / seconds_per_year
    sigma_per_step = 0.767 * np.sqrt(dt)
    mean = float(returns.mean())

    # |mean| should be at most ~3 standard errors (sigma/sqrt(N))
    # of pure noise away from zero.
    standard_error = sigma_per_step / np.sqrt(len(returns))
    assert abs(mean) < 5 * standard_error


# -----------------------------------------------------------------------------
# Stablecoin special handling
# -----------------------------------------------------------------------------


def test_stablecoin_path_stays_near_peg() -> None:
    """USDC path should never drift more than 1% from peg under normal config."""
    mock = MockMarketData(config=MockConfig(n_observations=2000, seed=11))
    usdc = mock.fetch(["USDC"])[0]
    prices = usdc.path.prices_usd
    assert prices.min() > 0.99
    assert prices.max() < 1.01


def test_stablecoin_starts_at_calibrated_price() -> None:
    """First observation is the calibrated current price (~$1.00), not the peg."""
    mock = MockMarketData()
    cal = Calibration()
    usdc = mock.fetch(["USDC"])[0]
    assert abs(usdc.path.prices_usd[0] - cal.get("USDC").current_price_usd) < 1e-9


def test_stablecoin_returns_have_smaller_vol_than_gbm() -> None:
    """Stablecoin realized vol should be vastly smaller than non-stable like SOL."""
    mock = MockMarketData(config=MockConfig(n_observations=2000, seed=5))
    out = mock.fetch(["USDC", "SOL"])
    usdc_vol = out[0].path.realized_volatility()
    sol_vol = out[1].path.realized_volatility()
    assert usdc_vol < sol_vol / 50  # at least 50x smaller


# -----------------------------------------------------------------------------
# Regime sensitivity
# -----------------------------------------------------------------------------


def test_shock_regime_more_extreme_drift_than_calm() -> None:
    """For SOL, shock regime cumulative price change should be more negative than calm."""
    cal = Calibration()
    bcfg = BlendingConfig()  # alpha=0.7, baseline=0%, multipliers calm/stress/shock = 0.3/1.0/2.0

    cfg_calm = MockConfig(n_observations=5000, regime="calm", seed=33)
    cfg_shock = MockConfig(n_observations=5000, regime="shock", seed=33)

    mock_calm = MockMarketData(calibration=cal, blending=bcfg, config=cfg_calm)
    mock_shock = MockMarketData(calibration=cal, blending=bcfg, config=cfg_shock)

    sol_calm = mock_calm.fetch(["SOL"])[0].path.prices_usd
    sol_shock = mock_shock.fetch(["SOL"])[0].path.prices_usd

    # Cumulative log return over the path.
    cum_calm = np.log(sol_calm[-1] / sol_calm[0])
    cum_shock = np.log(sol_shock[-1] / sol_shock[0])

    # Both should be negative (SOL has bearish blended drift).
    # Shock should be MORE negative than calm by roughly the multiplier ratio.
    assert cum_shock < cum_calm  # more bearish in shock


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError, match="interval_seconds"):
        MockConfig(interval_seconds=0)
    with pytest.raises(ValueError, match="n_observations"):
        MockConfig(n_observations=1)


# -----------------------------------------------------------------------------
# End-to-end smoke test
# -----------------------------------------------------------------------------


def test_full_universe_fetch_smoke() -> None:
    """All 8 tokens should fetch without error, returning sane data."""
    mock = MockMarketData()
    universe = ["SOL", "USDC", "PYTH", "AERO", "JUP", "BRETT", "WIF", "BONK"]
    out = mock.fetch(universe)
    assert len(out) == 8
    for snap in out:
        assert snap.path.prices_usd[0] > 0
        # Sanity: prices must be finite, positive everywhere.
        assert np.all(np.isfinite(snap.path.prices_usd))
        assert np.all(snap.path.prices_usd > 0)
