"""Monte Carlo CVaR tests.

Two tolerance regimes:
- Functional invariants (shapes, sign, monotonicity bounds): tight, ~1e-9.
- Statistical claims (sample-CVaR vs analytical CVaR): loose, 5-10% relative.
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from app.forecasting.base import HorizonForecast, MultiHorizonForecast
from app.forecasting.garch import GARCHForecaster
from app.market_data.base import PricePath, TokenMarketData
from app.market_data.calibration import Calibration
from app.market_data.mock import MockConfig, MockMarketData
from app.risk.monte_carlo import MonteCarloConfig, MonteCarloCVaR


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _fetch(symbol: str, n_obs: int = 1440, seed: int = 7) -> TokenMarketData:
    mock = MockMarketData(config=MockConfig(n_observations=n_obs, seed=seed))
    return mock.fetch([symbol])[0]


def _forecast(symbol: str, n_obs: int = 1440, seed: int = 7) -> MultiHorizonForecast:
    f = GARCHForecaster(calibration=Calibration())
    return f.forecast(_fetch(symbol, n_obs=n_obs, seed=seed))


def _synthetic_forecast(
    symbol: str,
    mean: float,
    vol: float,
    horizons: tuple[float, ...] = (5.0, 30.0, 120.0),
) -> MultiHorizonForecast:
    """Build a forecast with hand-chosen mean and vol for analytical comparison."""
    forecasts = {
        h: HorizonForecast(
            horizon_seconds=h,
            predicted_return=mean,
            predicted_volatility=vol,
            confidence_lower_95=mean - 2 * vol,
            confidence_upper_95=mean + 2 * vol,
        )
        for h in horizons
    }
    return MultiHorizonForecast(symbol=symbol, horizons=forecasts)


def _synthetic_token(symbol: str = "TEST") -> TokenMarketData:
    """Minimal valid TokenMarketData for tests that don't care about the path."""
    return TokenMarketData(
        symbol=symbol,
        mint="testmint",
        path=PricePath(
            symbol=symbol,
            prices_usd=np.array([100.0] * 100),
            interval_seconds=60.0,
        ),
        liquidity_depth_usd=1_000_000.0,
        spread_bps=5.0,
    )


# -----------------------------------------------------------------------------
# Functional invariants
# -----------------------------------------------------------------------------


def test_estimate_returns_multihorizon_with_all_forecast_horizons() -> None:
    risk = MonteCarloCVaR()
    f = GARCHForecaster(calibration=Calibration())
    data = _fetch("SOL")
    out = risk.estimate(data, f.forecast(data), position_value_usd=1000.0)
    assert sorted(out.estimates.keys()) == [5.0, 30.0, 120.0]
    assert out.symbol == "SOL"
    assert out.position_value_usd == 1000.0


def test_each_horizon_estimate_is_finite() -> None:
    risk = MonteCarloCVaR()
    out = risk.estimate(_fetch("SOL"), _forecast("SOL"), position_value_usd=1000.0)
    for h, est in out.estimates.items():
        assert math.isfinite(est.var)
        assert math.isfinite(est.cvar)
        assert math.isfinite(est.var_dollar)
        assert math.isfinite(est.cvar_dollar)


def test_cvar_is_at_least_as_severe_as_var() -> None:
    """Hard invariant — schema enforces it but we test the producer respects it."""
    risk = MonteCarloCVaR()
    for symbol in ("SOL", "BONK", "BRETT", "WIF", "AERO", "JUP", "PYTH"):
        out = risk.estimate(
            _fetch(symbol), _forecast(symbol), position_value_usd=1000.0
        )
        for est in out.estimates.values():
            assert est.cvar <= est.var + 1e-9


def test_dollar_amounts_match_return_times_position() -> None:
    risk = MonteCarloCVaR()
    pos = 5000.0
    out = risk.estimate(_fetch("SOL"), _forecast("SOL"), position_value_usd=pos)
    for est in out.estimates.values():
        assert abs(est.var_dollar - est.var * pos) < 1e-9
        assert abs(est.cvar_dollar - est.cvar * pos) < 1e-9


def test_n_samples_recorded_in_output() -> None:
    risk = MonteCarloCVaR(MonteCarloConfig(n_samples=5_000))
    out = risk.estimate(_fetch("SOL"), _forecast("SOL"), position_value_usd=1000.0)
    for est in out.estimates.values():
        assert est.n_samples == 5_000


def test_negative_position_rejected() -> None:
    risk = MonteCarloCVaR()
    with pytest.raises(ValueError, match="position_value_usd"):
        risk.estimate(_fetch("SOL"), _forecast("SOL"), position_value_usd=-100.0)


def test_zero_position_produces_zero_dollar_amounts() -> None:
    """Edge case: a hypothetical zero-size payment yields zero dollar risk."""
    risk = MonteCarloCVaR()
    out = risk.estimate(_fetch("SOL"), _forecast("SOL"), position_value_usd=0.0)
    for est in out.estimates.values():
        assert est.var_dollar == 0.0
        assert est.cvar_dollar == 0.0


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------


def test_same_seed_produces_identical_estimates() -> None:
    a = MonteCarloCVaR(MonteCarloConfig(seed=42))
    b = MonteCarloCVaR(MonteCarloConfig(seed=42))
    forecast = _forecast("SOL")
    data = _fetch("SOL")
    out_a = a.estimate(data, forecast, position_value_usd=1000.0)
    out_b = b.estimate(data, forecast, position_value_usd=1000.0)
    for h in out_a.estimates:
        assert out_a.at(h).var == out_b.at(h).var
        assert out_a.at(h).cvar == out_b.at(h).cvar


def test_different_seeds_produce_different_estimates() -> None:
    a = MonteCarloCVaR(MonteCarloConfig(seed=1))
    b = MonteCarloCVaR(MonteCarloConfig(seed=2))
    forecast = _forecast("SOL")
    data = _fetch("SOL")
    out_a = a.estimate(data, forecast, position_value_usd=1000.0)
    out_b = b.estimate(data, forecast, position_value_usd=1000.0)
    # Tail estimates should differ given different random draws.
    differs = any(
        out_a.at(h).cvar != out_b.at(h).cvar
        for h in out_a.estimates
    )
    assert differs


# -----------------------------------------------------------------------------
# Statistical sanity (loose tolerances)
# -----------------------------------------------------------------------------


def test_normal_cvar_matches_analytical_formula() -> None:
    """For Z ~ N(mu, sigma^2), the closed-form CVaR_alpha is:
        CVaR = mu - sigma * phi(z_alpha) / alpha
    where alpha = 1 - confidence_level, z_alpha is the alpha-quantile of N(0,1),
    and phi is the standard normal density.

    We compare our Monte Carlo CVaR against this analytical value.
    Tolerance: 5% relative error with 50K samples is comfortable.
    """
    mean = 0.0
    vol = 0.02
    confidence = 0.95

    risk = MonteCarloCVaR(MonteCarloConfig(n_samples=50_000, confidence_level=confidence, seed=123))
    forecast = _synthetic_forecast("TEST", mean=mean, vol=vol, horizons=(60.0,))
    out = risk.estimate(_synthetic_token("TEST"), forecast, position_value_usd=1.0)

    # Analytical CVaR for normal distribution at confidence c:
    #   alpha = 1 - c (e.g., 0.05)
    #   z_alpha = quantile of N(0,1) at alpha (negative number)
    #   phi(z_alpha) = density at that quantile
    #   CVaR = mu - sigma * phi(z_alpha) / alpha
    alpha = 1.0 - confidence
    z_alpha = norm.ppf(alpha)
    analytical_cvar = mean - vol * norm.pdf(z_alpha) / alpha

    sample_cvar = out.at(60.0).cvar
    relative_error = abs(sample_cvar - analytical_cvar) / abs(analytical_cvar)
    assert relative_error < 0.05, (
        f"sample CVaR {sample_cvar:.6f} vs analytical {analytical_cvar:.6f}, "
        f"relative error {relative_error:.4f}"
    )


def test_higher_vol_token_has_more_severe_cvar() -> None:
    """BONK (memecoin) should have more negative CVaR than SOL (major) at every horizon."""
    risk = MonteCarloCVaR()
    sol_out = risk.estimate(_fetch("SOL"), _forecast("SOL"), position_value_usd=1000.0)
    bonk_out = risk.estimate(_fetch("BONK"), _forecast("BONK"), position_value_usd=1000.0)
    for h in (5.0, 30.0, 120.0):
        assert bonk_out.at(h).cvar < sol_out.at(h).cvar


def test_stablecoin_has_near_zero_cvar() -> None:
    """USDC's GARCH forecast has near-zero vol, so CVaR_dollar should be tiny."""
    risk = MonteCarloCVaR()
    out = risk.estimate(_fetch("USDC"), _forecast("USDC"), position_value_usd=1000.0)
    # Over 120s, even at 95% confidence, USDC tail loss should be far under $1.
    assert abs(out.at(120.0).cvar_dollar) < 1.0


# -----------------------------------------------------------------------------
# Cross-horizon monotonicity
# -----------------------------------------------------------------------------


def test_longer_horizons_have_more_severe_cvar() -> None:
    """Variance grows with horizon, so |CVaR| should grow with horizon length."""
    risk = MonteCarloCVaR()
    out = risk.estimate(_fetch("SOL"), _forecast("SOL"), position_value_usd=1000.0)
    cvar_5 = out.at(5.0).cvar
    cvar_30 = out.at(30.0).cvar
    cvar_120 = out.at(120.0).cvar
    # All negative; longer horizons more negative.
    assert cvar_120 < cvar_30 < cvar_5


def test_worst_cvar_dollar_returns_120s() -> None:
    """For a typical token, the 120s horizon is the worst because variance grows with time."""
    risk = MonteCarloCVaR()
    out = risk.estimate(_fetch("SOL"), _forecast("SOL"), position_value_usd=1000.0)
    assert out.worst_cvar_dollar() == out.at(120.0).cvar_dollar


# -----------------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------------


def test_config_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        MonteCarloConfig(n_samples=50)


def test_config_rejects_invalid_confidence_level() -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        MonteCarloConfig(confidence_level=1.5)
    with pytest.raises(ValueError, match="confidence_level"):
        MonteCarloConfig(confidence_level=0.0)


# -----------------------------------------------------------------------------
# End-to-end smoke
# -----------------------------------------------------------------------------


def test_full_universe_estimate_smoke() -> None:
    """All 8 calibrated tokens should produce valid CVaR estimates without errors."""
    risk = MonteCarloCVaR()
    universe = ["SOL", "USDC", "PYTH", "AERO", "JUP", "BRETT", "WIF", "BONK"]
    for sym in universe:
        out = risk.estimate(
            _fetch(sym), _forecast(sym), position_value_usd=1000.0
        )
        for est in out.estimates.values():
            assert math.isfinite(est.cvar)
            assert est.cvar <= est.var + 1e-9
