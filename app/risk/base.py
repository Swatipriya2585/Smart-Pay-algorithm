"""
Abstract contracts for tail-risk estimators.

Per the RAMHD specification, the system applies Conditional Value-at-Risk
(CVaR) modeling to estimate potential worst-case price swings and
proactively avoids tokens with extreme downside risk.

Core concepts:

- VaR (Value-at-Risk) at confidence c is the loss threshold such that
  losses worse than VaR happen with probability (1 - c). E.g., VaR_95 of
  -2% means there's a 5% chance of losing more than 2% over the horizon.

- CVaR (also called Expected Shortfall) is the AVERAGE loss in the cases
  worse than VaR. CVaR_95 of -3% means: when losses do exceed VaR, they
  average -3%. CVaR is always at least as severe as VaR.

CVaR is preferred over VaR for risk-adaptive routing because it captures
the SEVERITY of bad outcomes, not just their likelihood. Two tokens with
identical VaR can have very different CVaRs if one has fatter tails.

References:

- Rockafellar & Uryasev (2000) — "Optimization of Conditional Value-at-Risk"
- Acerbi & Tasche (2002) — "On the coherence of expected shortfall"
- Basel Committee FRTB (2019) — formal adoption of CVaR for regulatory capital
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.forecasting.base import MultiHorizonForecast
from app.market_data.base import TokenMarketData


@dataclass(frozen=True)
class TailRiskEstimate:
    """Tail-risk estimate for one token at one horizon.

    All return values are in log-return space (consistent with forecaster
    output). Convention: losses are NEGATIVE numbers, so VaR_95 of -0.02
    means the 5th percentile loss is -2%.

    Attributes:
        horizon_seconds: forecast horizon this estimate applies to
        confidence_level: e.g. 0.95 for VaR_95 / CVaR_95
        var: Value-at-Risk — the (1-c) quantile of the loss distribution
        cvar: Conditional Value-at-Risk — average loss conditional on loss <= VaR
        var_dollar: VaR translated to dollars given the position size
        cvar_dollar: CVaR translated to dollars given the position size
        n_samples: number of Monte Carlo paths used (or 0 for analytical estimators)

    Invariants enforced at construction:
        - confidence_level in (0, 1)
        - cvar <= var (CVaR is always at least as severe)
        - var/cvar in dollar form must have same sign as var/cvar in return form
    """

    horizon_seconds: float
    confidence_level: float
    var: float
    cvar: float
    var_dollar: float
    cvar_dollar: float
    n_samples: int

    def __post_init__(self) -> None:
        if self.horizon_seconds <= 0:
            raise ValueError(
                f"horizon_seconds must be positive, got {self.horizon_seconds}"
            )
        if not 0 < self.confidence_level < 1:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )
        if self.n_samples < 0:
            raise ValueError(
                f"n_samples must be non-negative, got {self.n_samples}"
            )
        # CVaR is the average of the worst tail; it must be at least as
        # severe (i.e., as negative) as the VaR threshold.
        # Allow tiny floating-point slack of 1e-9.
        if self.cvar > self.var + 1e-9:
            raise ValueError(
                f"cvar ({self.cvar}) must be <= var ({self.var}) "
                f"(CVaR is the average of the tail beyond VaR)"
            )
        # Dollar values should agree in sign with return values.
        # We allow zero (e.g., a peg-stable token).
        if self.var < 0 and self.var_dollar > 0:
            raise ValueError("var_dollar must be non-positive when var is negative")
        if self.cvar < 0 and self.cvar_dollar > 0:
            raise ValueError("cvar_dollar must be non-positive when cvar is negative")


@dataclass(frozen=True)
class MultiHorizonRiskEstimate:
    """Tail-risk estimates for one token across all forecast horizons."""

    symbol: str
    position_value_usd: float
    estimates: dict[float, TailRiskEstimate]

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if self.position_value_usd < 0:
            raise ValueError(
                f"position_value_usd must be non-negative, got {self.position_value_usd}"
            )
        if not self.estimates:
            raise ValueError("estimates must contain at least one horizon")
        for h_seconds, est in self.estimates.items():
            if h_seconds != est.horizon_seconds:
                raise ValueError(
                    f"horizon key {h_seconds} does not match "
                    f"estimate.horizon_seconds {est.horizon_seconds}"
                )

    def at(self, horizon_seconds: float) -> TailRiskEstimate:
        """Look up the estimate for a specific horizon. Raises KeyError if absent."""
        try:
            return self.estimates[horizon_seconds]
        except KeyError as e:
            raise KeyError(
                f"horizon {horizon_seconds}s not in this estimate. "
                f"Available: {sorted(self.estimates)}"
            ) from e

    def horizon_seconds_list(self) -> list[float]:
        return sorted(self.estimates)

    def worst_cvar_dollar(self) -> float:
        """The most-severe CVaR (in dollars) across all horizons.

        Used by the risk-adaptive router to make routing decisions. A token
        whose worst-case CVaR exceeds a configurable bound gets excluded.
        Returns the most-negative cvar_dollar value, or 0 if none are negative.
        """
        return min((e.cvar_dollar for e in self.estimates.values()), default=0.0)


class RiskEstimator(Protocol):
    """Protocol every tail-risk estimator must satisfy.

    The default implementation is Monte Carlo CVaR over GARCH forecasts,
    but the Protocol leaves room for analytical CVaR (Gaussian closed-form),
    historical-simulation CVaR, or extreme-value-theory based estimators.
    """

    def estimate(
        self,
        data: TokenMarketData,
        forecast: MultiHorizonForecast,
        position_value_usd: float,
    ) -> MultiHorizonRiskEstimate:
        """Produce tail-risk estimates for one token at all horizons in the forecast.

        Args:
            data: market data for the token (path, liquidity context).
            forecast: multi-horizon forecast from the GARCH forecaster.
            position_value_usd: USD value of the proposed payment to scale
                                returns into dollar-denominated VaR/CVaR.

        Returns:
            MultiHorizonRiskEstimate with one TailRiskEstimate per horizon
            in the forecast.
        """
        ...
