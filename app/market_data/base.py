"""
Abstract contracts for market data sources.

Anything downstream (forecaster, CVaR, cost scorer) only sees these types.
That lets MockMarketData (synthetic), HistoricalMarketData (CSV replay),
and LiveMarketData (Pyth + Jupiter) be substitutable with zero friction.

This module has NO dependency on calibration.py — it's the shared contract
both calibration and mock build on top of.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PricePath:
    """A historical price series for one token.

    Prices are sampled at uniform intervals (default 1 minute) so downstream
    volatility and CVaR models can assume equal-spaced observations.

    Attributes:
        symbol: Token symbol (e.g. "SOL").
        prices_usd: 1D array of prices. prices_usd[-1] is the most recent.
        interval_seconds: Spacing between observations in seconds.
    """

    symbol: str
    prices_usd: np.ndarray
    interval_seconds: float

    def log_returns(self) -> np.ndarray:
        """Natural log returns r_t = ln(p_t / p_{t-1}). Length = len(prices) - 1."""
        return np.diff(np.log(self.prices_usd))

    def realized_volatility(self) -> float:
        """Sample standard deviation of log returns over the full window.

        Uses Bessel's correction (ddof=1). For a path of length N, this
        is computed over N-1 returns.
        """
        returns = self.log_returns()
        if len(returns) < 2:
            return 0.0
        return float(np.std(returns, ddof=1))


@dataclass(frozen=True)
class TokenMarketData:
    """Full market snapshot for one token at the moment of request.

    Bundles the historical price path with point-in-time liquidity and
    spread data, which downstream scorers need but don't fetch themselves.
    """

    symbol: str
    mint: str
    path: PricePath
    liquidity_depth_usd: float
    spread_bps: float

    @property
    def current_price_usd(self) -> float:
        return float(self.path.prices_usd[-1])


class MarketDataSource(Protocol):
    """Protocol every market data source must satisfy.

    Downstream scorers accept a MarketDataSource and call .fetch() only.
    Keeping this Protocol tiny means mock and live implementations are
    substitutable with zero friction.
    """

    def fetch(self, symbols: list[str]) -> list[TokenMarketData]:
        """Return current market data for each requested symbol.

        The returned list preserves input order. If a symbol is unavailable,
        implementations should raise KeyError — do NOT silently skip.
        """
        ...


@dataclass(frozen=True)
class NetworkConditions:
    """Chain-level state used by the cost scorer.

    Kept separate from per-token data because it's shared across tokens.
    """

    priority_fee_lamports: float
    congestion_score: float  # 0.0 = empty, 1.0 = saturated
    slot_time_ms: float
