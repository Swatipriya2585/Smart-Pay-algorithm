"""Verify PricePath and TokenMarketData behave correctly on known inputs."""

import math

import numpy as np

from app.market_data.base import PricePath, TokenMarketData


def test_log_returns_on_known_path() -> None:
    """log_returns of [100, 110, 121] should be [ln(1.1), ln(1.1)] approximately."""
    path = PricePath(
        symbol="TEST",
        prices_usd=np.array([100.0, 110.0, 121.0]),
        interval_seconds=60.0,
    )
    returns = path.log_returns()
    assert len(returns) == 2
    expected = math.log(1.1)
    assert abs(returns[0] - expected) < 1e-12
    assert abs(returns[1] - expected) < 1e-12


def test_realized_volatility_constant_returns_is_zero() -> None:
    """A path with constant log returns should have zero realized vol (sample std)."""
    # Geometric series: each step multiplies by 1.1, so log returns are all equal.
    prices = np.array([100.0 * (1.1 ** i) for i in range(10)])
    path = PricePath(symbol="TEST", prices_usd=prices, interval_seconds=60.0)
    assert path.realized_volatility() < 1e-12


def test_realized_volatility_matches_numpy() -> None:
    """For an arbitrary path, our realized_volatility must equal numpy's std(ddof=1) of log returns."""
    rng = np.random.default_rng(42)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=100)))
    path = PricePath(symbol="TEST", prices_usd=prices, interval_seconds=60.0)
    expected = float(np.std(np.diff(np.log(prices)), ddof=1))
    assert abs(path.realized_volatility() - expected) < 1e-12


def test_token_market_data_current_price() -> None:
    """current_price_usd should return the last element of the path."""
    path = PricePath(
        symbol="TEST",
        prices_usd=np.array([100.0, 200.0, 300.0]),
        interval_seconds=60.0,
    )
    md = TokenMarketData(
        symbol="TEST",
        mint="abc",
        path=path,
        liquidity_depth_usd=1_000_000.0,
        spread_bps=5.0,
    )
    assert md.current_price_usd == 300.0


def test_pricepath_short_path_volatility() -> None:
    """A path with fewer than 2 returns must return 0.0 vol, not crash."""
    path = PricePath(
        symbol="TEST",
        prices_usd=np.array([100.0, 110.0]),
        interval_seconds=60.0,
    )
    # 1 return -> not enough for sample std with ddof=1
    assert path.realized_volatility() == 0.0
