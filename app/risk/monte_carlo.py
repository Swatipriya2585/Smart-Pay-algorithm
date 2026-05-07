"""
Monte Carlo CVaR estimator for RAMHD tail-risk modeling.

Given a forecast (predicted return + volatility) for one token at multiple
horizons, generate simulated return outcomes and compute:
- VaR: the 5th-percentile loss (the "loss threshold" in the bad tail)
- CVaR: the average loss conditional on being in that tail

Why Monte Carlo: even though the normal-distribution case has analytical
closed forms for VaR and CVaR, sampling-based estimation gives us a
distribution-agnostic engine. We can swap to Student-t (fat tails),
historical simulation, or jump-diffusion processes by changing the
sampling step alone — the rest of the math stays identical.

Numerical practice (Glasserman, 'Monte Carlo Methods in Financial
Engineering', ch. 9):
- Use a fixed seed so results are reproducible.
- Use at least 10,000 samples; below that, CVaR estimates have ~5%+
  sampling-error relative noise that destabilizes downstream routing.
- For symmetric distributions, the antithetic-variate trick can halve
  variance, but we don't enable it by default — keeping the code path
  simple is more valuable than the marginal speedup at our sample count.

Stablecoin handling:
A token flagged as a stablecoin in calibration produces near-zero
volatility forecasts (the GARCH forecaster already special-cases this).
That naturally produces near-zero CVaR. We don't add another bypass here
— letting the same code path run on a tiny variance is correct and keeps
the test surface uniform.

References:
- Rockafellar & Uryasev (2000) — CVaR foundations.
- Glasserman (2004) — Monte Carlo methods in financial engineering.
- Acerbi & Tasche (2002) — coherence properties of expected shortfall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.forecasting.base import MultiHorizonForecast
from app.market_data.base import TokenMarketData
from app.risk.base import MultiHorizonRiskEstimate, TailRiskEstimate


Distribution = Literal["normal"]


@dataclass(frozen=True)
class MonteCarloConfig:
    """Tuning knobs for Monte Carlo CVaR estimation.

    Defaults reflect production-ready settings:
    - 10,000 samples is the standard for stable tail estimates.
    - 95% confidence is the canonical short-horizon trading threshold.
    - Seed pinned for reproducibility; override per-test if needed.
    """

    n_samples: int = 10_000
    confidence_level: float = 0.95
    distribution: Distribution = "normal"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_samples < 100:
            raise ValueError(
                f"n_samples must be >= 100 for stable tail estimates, got {self.n_samples}"
            )
        if not 0 < self.confidence_level < 1:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )


class MonteCarloCVaR:
    """Sampling-based VaR/CVaR estimator that consumes GARCH forecasts.

    Usage:
        forecaster = GARCHForecaster(calibration=cal)
        risk = MonteCarloCVaR()
        forecast = forecaster.forecast(token_data)
        estimate = risk.estimate(token_data, forecast, position_value_usd=1000.0)
        worst = estimate.worst_cvar_dollar()  # used by risk-adaptive routing

    Stateless: each estimate() call independently samples a fresh batch of
    paths under the configured seed. To get fresh samples across calls
    (e.g., between test cases), construct a new instance with a new seed.
    """

    def __init__(self, config: MonteCarloConfig | None = None) -> None:
        self.config = config if config is not None else MonteCarloConfig()
        self._rng = np.random.default_rng(self.config.seed)

    # -------------------------------------------------------------------
    # Public API: RiskEstimator protocol
    # -------------------------------------------------------------------

    def estimate(
        self,
        data: TokenMarketData,
        forecast: MultiHorizonForecast,
        position_value_usd: float,
    ) -> MultiHorizonRiskEstimate:
        """Produce VaR/CVaR estimates for one token across all forecast horizons."""
        if position_value_usd < 0:
            raise ValueError(
                f"position_value_usd must be non-negative, got {position_value_usd}"
            )

        estimates: dict[float, TailRiskEstimate] = {}
        for horizon_seconds in forecast.horizon_seconds_list():
            hf = forecast.at(horizon_seconds)
            estimates[horizon_seconds] = self._estimate_one_horizon(
                horizon_seconds=horizon_seconds,
                predicted_return=hf.predicted_return,
                predicted_volatility=hf.predicted_volatility,
                position_value_usd=position_value_usd,
            )

        return MultiHorizonRiskEstimate(
            symbol=data.symbol,
            position_value_usd=position_value_usd,
            estimates=estimates,
        )

    # -------------------------------------------------------------------
    # Per-horizon estimation
    # -------------------------------------------------------------------

    def _estimate_one_horizon(
        self,
        horizon_seconds: float,
        predicted_return: float,
        predicted_volatility: float,
        position_value_usd: float,
    ) -> TailRiskEstimate:
        """Sample N return outcomes, compute the (1-c) tail's VaR and CVaR."""
        # Generate samples from the forecast distribution.
        samples = self._sample(
            mean=predicted_return,
            std=predicted_volatility,
            n=self.config.n_samples,
        )

        # Tail cutoff index: the (1 - confidence_level) quantile.
        # E.g., for c=0.95 with 10000 samples, the 500 worst samples form the tail.
        tail_alpha = 1.0 - self.config.confidence_level
        sorted_samples = np.sort(samples)  # ascending: worst losses first
        cutoff_idx = max(1, int(round(tail_alpha * len(sorted_samples))))

        # VaR is the loss threshold at the cutoff. CVaR is the mean of the
        # tail samples (those at indices [0, cutoff_idx)).
        var = float(sorted_samples[cutoff_idx - 1])
        cvar = float(np.mean(sorted_samples[:cutoff_idx]))

        # If the predicted_volatility is exactly zero (e.g. perfectly pegged
        # stable), every sample equals the mean and var == cvar == mean.
        # The contract's "cvar <= var" invariant still holds (with equality).

        return TailRiskEstimate(
            horizon_seconds=horizon_seconds,
            confidence_level=self.config.confidence_level,
            var=var,
            cvar=cvar,
            var_dollar=var * position_value_usd,
            cvar_dollar=cvar * position_value_usd,
            n_samples=self.config.n_samples,
        )

    # -------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------

    def _sample(self, mean: float, std: float, n: int) -> np.ndarray:
        """Draw n samples from the configured distribution.

        Currently only normal is supported; Student-t and historical
        simulation are future extensions.
        """
        if self.config.distribution == "normal":
            return self._rng.normal(loc=mean, scale=max(std, 0.0), size=n)
        # Should be unreachable given the Literal type, but defensive.
        raise ValueError(f"Unsupported distribution: {self.config.distribution}")

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset the RNG. Useful between independent test cases."""
        new_seed = seed if seed is not None else self.config.seed
        self._rng = np.random.default_rng(new_seed)
