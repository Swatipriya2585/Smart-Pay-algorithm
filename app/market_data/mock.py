"""
Synthetic market data generator.

Implements the MarketDataSource Protocol with calibrated, deterministic
synthetic price paths. Downstream scorers (forecaster, CVaR, cost)
consume this exactly like they would consume live data — they don't
know it's synthetic.

Math:

- Non-stablecoins: Geometric Brownian Motion (GBM) with calibrated
  vol and blended drift. Discretized as:

    p_{t+1} = p_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

  where Z ~ N(0,1), mu and sigma are per-second values derived from
  the annualized calibration parameters.

- Stablecoins: Ornstein-Uhlenbeck mean reversion to $1.00 with very
  small noise. GBM is wrong for pegged assets — they don't drift, they
  oscillate around the peg.

    p_{t+1} = peg + (p_t - peg)*exp(-kappa*dt) + small_noise

Reproducibility: every MockMarketData(seed=N) produces identical paths
for identical inputs. Critical for unit tests.

References:

- Hull, "Options, Futures, and Other Derivatives", ch. 14 (GBM discretization)

- Vasicek 1977 (mean-reverting OU process for fixed-income; same math)

"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.market_data.base import (
    MarketDataSource,
    PricePath,
    TokenMarketData,
)
from app.market_data.calibration import (
    BlendingConfig,
    Calibration,
    Regime,
    TokenCalibration,
    blended_drift,
)

SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


@dataclass(frozen=True)
class MockConfig:
    """Tuning knobs for the synthetic generator.

    Defaults: 1-minute bars over a 24-hour window (1440 observations),
    enough history for GARCH (typically needs 100+) and CVaR Monte Carlo
    (typically resamples this directly). The regime defaults to 'stress',
    the algorithm's middle/baseline regime.

    """

    interval_seconds: float = 60.0
    n_observations: int = 1440  # 24 hours of 1-minute bars
    regime: Regime = "stress"
    seed: int = 42

    # Stablecoin-specific knobs.
    stablecoin_peg_usd: float = 1.0
    stablecoin_mean_reversion_kappa: float = 5.0  # per year; pulls back to peg
    stablecoin_noise_bps: float = 5.0  # daily-equivalent noise in basis points

    # Defaults for liquidity/spread by regime, used when calibration doesn't
    # carry these (it doesn't — they come from live data in production).
    # These are reasonable starting points, scalable per-token in subclasses.
    default_liquidity_depth_usd: float = 5_000_000.0
    default_spread_bps: float = 8.0

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if self.n_observations < 2:
            raise ValueError("n_observations must be >= 2")
        if self.stablecoin_peg_usd <= 0:
            raise ValueError("stablecoin_peg_usd must be positive")


class MockMarketData(MarketDataSource):
    """Calibrated, deterministic synthetic market data.

    Usage:

        cal = Calibration()
        mock = MockMarketData(calibration=cal)
        snapshots = mock.fetch(["SOL", "USDC", "BONK"])
        # snapshots is a list[TokenMarketData], same order as input

    The fetch() output is fully deterministic given the same seed and
    inputs, so unit tests can assert exact statistical properties.

    """

    def __init__(
        self,
        calibration: Calibration | None = None,
        blending: BlendingConfig | None = None,
        config: MockConfig | None = None,
    ) -> None:
        self.calibration = calibration if calibration is not None else Calibration()
        self.blending = blending if blending is not None else BlendingConfig()
        self.config = config if config is not None else MockConfig()
        self._rng = np.random.default_rng(self.config.seed)

    # -------------------------------------------------------------------
    # Public API: MarketDataSource protocol
    # -------------------------------------------------------------------

    def fetch(self, symbols: list[str]) -> list[TokenMarketData]:
        """Return synthetic market data for each symbol, preserving order.

        Raises KeyError if a symbol is not in the calibration universe —
        same contract as a live MarketDataSource would honor.

        """
        out: list[TokenMarketData] = []
        for symbol in symbols:
            token = self.calibration.get(symbol)
            path = self._generate_path(token)
            out.append(
                TokenMarketData(
                    symbol=token.symbol,
                    mint=token.address,
                    path=path,
                    liquidity_depth_usd=self.config.default_liquidity_depth_usd,
                    spread_bps=self.config.default_spread_bps,
                )
            )
        return out

    # -------------------------------------------------------------------
    # Path generation
    # -------------------------------------------------------------------

    def _generate_path(self, token: TokenCalibration) -> PricePath:
        if token.is_stablecoin:
            prices = self._generate_stablecoin_path(token)
        else:
            prices = self._generate_gbm_path(token)
        return PricePath(
            symbol=token.symbol,
            prices_usd=prices,
            interval_seconds=self.config.interval_seconds,
        )

    def _generate_gbm_path(self, token: TokenCalibration) -> np.ndarray:
        """Geometric Brownian Motion with calibrated annualized parameters.

        Uses the exact-discretization form of GBM (Hull eq. 14.20):

            p_{t+1} = p_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

        which is correct for any dt, not just dt -> 0.

        """
        # Convert annualized params to per-second.
        mu_annual = blended_drift(token, self.config.regime, self.blending)
        sigma_annual = token.annualized_vol

        dt = self.config.interval_seconds / SECONDS_PER_YEAR
        n = self.config.n_observations

        # n_observations price points => n-1 increments.
        # Draw n-1 standard-normal shocks.
        shocks = self._rng.standard_normal(n - 1)

        # Exact-discretization log-step.
        drift_term = (mu_annual - 0.5 * sigma_annual ** 2) * dt
        diffusion_term = sigma_annual * math.sqrt(dt) * shocks
        log_steps = drift_term + diffusion_term

        # Cumulative log price relative to starting price.
        log_relative = np.concatenate(([0.0], np.cumsum(log_steps)))
        prices = token.current_price_usd * np.exp(log_relative)

        return prices

    def _generate_stablecoin_path(self, token: TokenCalibration) -> np.ndarray:
        """Mean-reverting OU around the peg, with small noise.

        We do NOT use GBM for stablecoins — they don't drift, they oscillate.

        OU process: dp = -kappa*(p - peg)*dt + sigma_local*dW

        Discrete form (exact for OU):

            p_{t+1} = peg + (p_t - peg)*exp(-kappa*dt) +
                      sigma_local*sqrt((1-exp(-2*kappa*dt))/(2*kappa))*Z

        """
        peg = self.config.stablecoin_peg_usd
        kappa_annual = self.config.stablecoin_mean_reversion_kappa
        # Convert basis-points-per-day noise to absolute std-per-second.
        # noise_bps is std of the price as a fraction of peg per day.
        noise_per_day = self.config.stablecoin_noise_bps / 10_000.0 * peg
        sigma_annual = noise_per_day * math.sqrt(365.0)

        dt = self.config.interval_seconds / SECONDS_PER_YEAR
        n = self.config.n_observations

        decay = math.exp(-kappa_annual * dt)
        # Stationary noise scaling for the discrete OU.
        if kappa_annual > 0:
            noise_scale = sigma_annual * math.sqrt(
                (1.0 - math.exp(-2.0 * kappa_annual * dt)) / (2.0 * kappa_annual)
            )
        else:
            noise_scale = sigma_annual * math.sqrt(dt)

        shocks = self._rng.standard_normal(n - 1)

        prices = np.empty(n)
        prices[0] = token.current_price_usd
        for i in range(1, n):
            prices[i] = peg + (prices[i - 1] - peg) * decay + noise_scale * shocks[i - 1]

        return prices

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset the RNG. Useful between tests that share a single instance."""
        new_seed = seed if seed is not None else self.config.seed
        self._rng = np.random.default_rng(new_seed)
