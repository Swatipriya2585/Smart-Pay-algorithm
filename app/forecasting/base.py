"""
Abstract contracts for multi-horizon forecasters.

Per the RAMHD specification, forecasters predict short-term price stability
over multiple time windows (5s, 30s, 120s) so the algorithm can anticipate
near-future fluctuations rather than reacting only to current prices.

Output shape:
- For each horizon, return predicted_return (mean log-return),
  predicted_volatility (std-dev of log-return), and a 95% confidence band.
- Bundle horizons into a single MultiHorizonForecast keyed by horizon_seconds.

This module has NO dependency on any specific forecasting model — it's the
shared contract any implementation (GARCH, LSTM hybrid, etc.) builds on top of.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.market_data.base import TokenMarketData


# Standard horizons from the original RAMHD specification.
# Adding/removing horizons later is a config-only change for downstream consumers.
DEFAULT_HORIZONS: tuple[float, ...] = (5.0, 30.0, 120.0)


@dataclass(frozen=True)
class HorizonForecast:
    """Forecast for one token at one horizon.

    All values describe a log-return distribution over the horizon window:
    - predicted_return: expected log-return (E[r])
    - predicted_volatility: std-dev of log-return (sqrt(Var[r]))
    - confidence_lower_95 / upper_95: 5th/95th percentile bounds

    Convention: predicted_return is in decimal log-return form (e.g., -0.005
    for a -0.5% expected move). Volatility is the std-dev in the same units.
    Both scale with horizon length per Brownian-motion variance scaling.
    """

    horizon_seconds: float
    predicted_return: float
    predicted_volatility: float
    confidence_lower_95: float
    confidence_upper_95: float

    def __post_init__(self) -> None:
        if self.horizon_seconds <= 0:
            raise ValueError(
                f"horizon_seconds must be positive, got {self.horizon_seconds}"
            )
        if self.predicted_volatility < 0:
            raise ValueError(
                f"predicted_volatility must be non-negative, got {self.predicted_volatility}"
            )
        if self.confidence_lower_95 > self.confidence_upper_95:
            raise ValueError(
                f"confidence_lower_95 ({self.confidence_lower_95}) must be <= "
                f"confidence_upper_95 ({self.confidence_upper_95})"
            )


@dataclass(frozen=True)
class MultiHorizonForecast:
    """All horizons for one token, bundled.

    Lookup pattern:
        mhf.at(30.0)  -> HorizonForecast for 30-second horizon
        mhf.horizons  -> dict keyed by horizon_seconds

    Construction enforces that horizons are positive and unique; consumers
    can rely on the dict keys being a strict subset of valid horizons.
    """

    symbol: str
    horizons: dict[float, HorizonForecast]

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if not self.horizons:
            raise ValueError("horizons must contain at least one forecast")
        for h_seconds, fcast in self.horizons.items():
            if h_seconds != fcast.horizon_seconds:
                raise ValueError(
                    f"horizon key {h_seconds} does not match "
                    f"forecast.horizon_seconds {fcast.horizon_seconds}"
                )

    def at(self, horizon_seconds: float) -> HorizonForecast:
        """Look up the forecast for a specific horizon. Raises KeyError if absent."""
        try:
            return self.horizons[horizon_seconds]
        except KeyError as e:
            raise KeyError(
                f"horizon {horizon_seconds}s not in this forecast. "
                f"Available: {sorted(self.horizons)}"
            ) from e

    def horizon_seconds_list(self) -> list[float]:
        """Sorted list of available horizons. Convenience for iteration."""
        return sorted(self.horizons)


class Forecaster(Protocol):
    """Protocol every forecaster must satisfy.

    Downstream RAMHD components (CVaR estimator, regime detector, scorers)
    depend only on this Protocol — they do not know whether they're using
    GARCH, an LSTM hybrid, or a future neural model.
    """

    def forecast(
        self,
        data: TokenMarketData,
        horizons: tuple[float, ...] = DEFAULT_HORIZONS,
    ) -> MultiHorizonForecast:
        """Produce multi-horizon forecasts for one token.

        Args:
            data: latest market data for the token (path + liquidity context).
            horizons: which forecast horizons (in seconds) to produce.

        Returns:
            MultiHorizonForecast with one HorizonForecast per requested horizon.
        """
        ...
