"""
Abstract contracts for cost & latency scorers.

Per the RAMHD specification, the dominance-modeling layer compares tokens
across multiple factors — speed, cost, reliability, volatility — and
filters those that are too risky, slow, or expensive. This module owns
the "cost" and "speed" dimensions of that comparison.

Three components compose into one estimate per horizon:

1. Slippage — price impact of swapping the position into/through this
   token. Closed-form from constant-product AMM math: a $1000 swap into
   a $5M-deep pool costs ~$0.02 in slippage; a $1000 swap into a $20K
   pool costs ~$50.

2. Gas / priority fees — Solana base + priority fee, scaled by current
   network congestion. Tiny in absolute terms (~$0.0001 base) but
   priority fees can spike during shocks.

3. Settlement risk — exposure to price movement during the time it takes
   the transaction to confirm. Computed as: predicted_volatility scaled
   to the settlement window × position_value. Reuses the GARCH forecast.

Sign convention: all costs are NEGATIVE numbers (value the user gives up).
A $5 slippage cost means slippage_dollar = -5.0. This keeps the algorithm
math consistent: total_utility = positive_returns + negative_costs +
negative_tail_risk.

References:

- Constant-product AMM math: Adams et al., "Uniswap v2 Core" (2020).
- Solana fee mechanics: Solana docs, "Transaction Fees" (2024).
- Settlement-window risk: Almgren & Chriss, "Optimal Execution of
  Portfolio Transactions" (2000) — same idea applied to crypto payments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.forecasting.base import MultiHorizonForecast
from app.market_data.base import NetworkConditions, TokenMarketData


@dataclass(frozen=True)
class CostBreakdown:
    """Per-horizon cost decomposition for one token.

    All dollar fields are non-positive (zero or negative); they represent
    value the user gives up. The aggregate `total_cost_dollar` is the sum
    of the three components and is also non-positive.

    Attributes:
        horizon_seconds: forecast horizon this estimate applies to
        slippage_dollar: dollar value of expected price impact (<= 0)
        gas_dollar: dollar value of gas + priority fees (<= 0)
        settlement_risk_dollar: dollar exposure during settlement window (<= 0)
        total_cost_dollar: sum of the three (<= 0)
        settlement_seconds: expected settlement time in seconds (used for risk)
    """

    horizon_seconds: float
    slippage_dollar: float
    gas_dollar: float
    settlement_risk_dollar: float
    total_cost_dollar: float
    settlement_seconds: float

    def __post_init__(self) -> None:
        if self.horizon_seconds <= 0:
            raise ValueError(
                f"horizon_seconds must be positive, got {self.horizon_seconds}"
            )
        if self.settlement_seconds <= 0:
            raise ValueError(
                f"settlement_seconds must be positive, got {self.settlement_seconds}"
            )
        # All cost components must be non-positive (sign convention).
        for name, val in (
            ("slippage_dollar", self.slippage_dollar),
            ("gas_dollar", self.gas_dollar),
            ("settlement_risk_dollar", self.settlement_risk_dollar),
            ("total_cost_dollar", self.total_cost_dollar),
        ):
            if val > 1e-9:
                raise ValueError(
                    f"{name} must be non-positive (cost is value given up), got {val}"
                )
        # The aggregate must equal the sum of components (within float slack).
        component_sum = (
            self.slippage_dollar + self.gas_dollar + self.settlement_risk_dollar
        )
        if abs(self.total_cost_dollar - component_sum) > 1e-6:
            raise ValueError(
                f"total_cost_dollar ({self.total_cost_dollar}) must equal "
                f"slippage + gas + settlement_risk ({component_sum})"
            )


@dataclass(frozen=True)
class MultiHorizonCostEstimate:
    """Cost breakdowns for one token across all horizons."""

    symbol: str
    position_value_usd: float
    breakdowns: dict[float, CostBreakdown]

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if self.position_value_usd < 0:
            raise ValueError(
                f"position_value_usd must be non-negative, got {self.position_value_usd}"
            )
        if not self.breakdowns:
            raise ValueError("breakdowns must contain at least one horizon")
        for h_seconds, br in self.breakdowns.items():
            if h_seconds != br.horizon_seconds:
                raise ValueError(
                    f"horizon key {h_seconds} does not match "
                    f"breakdown.horizon_seconds {br.horizon_seconds}"
                )

    def at(self, horizon_seconds: float) -> CostBreakdown:
        """Look up the breakdown for a specific horizon. Raises KeyError if absent."""
        try:
            return self.breakdowns[horizon_seconds]
        except KeyError as e:
            raise KeyError(
                f"horizon {horizon_seconds}s not in this estimate. "
                f"Available: {sorted(self.breakdowns)}"
            ) from e

    def horizon_seconds_list(self) -> list[float]:
        return sorted(self.breakdowns)

    def worst_total_cost_dollar(self) -> float:
        """The most-severe total cost across all horizons (most negative).

        Used by the Pareto filter to compare candidates on the cost dimension.
        Returns the most-negative value, or 0 if all are zero.
        """
        return min(
            (b.total_cost_dollar for b in self.breakdowns.values()),
            default=0.0,
        )


class CostEstimator(Protocol):
    """Protocol every cost & latency estimator must satisfy.

    The default implementation is SolanaCostScorer using closed-form AMM
    math + GARCH-linked settlement risk. Future implementations might
    add fee oracles, MEV-protection routing premiums, or chain-specific
    adapters.
    """

    def estimate(
        self,
        data: TokenMarketData,
        forecast: MultiHorizonForecast,
        network: NetworkConditions,
        position_value_usd: float,
    ) -> MultiHorizonCostEstimate:
        """Produce cost breakdowns for one token at all horizons in the forecast.

        Args:
            data: market data for the token (path + liquidity context).
            forecast: multi-horizon forecast — used to scale settlement risk.
            network: current Solana network state (gas, congestion, slot time).
            position_value_usd: USD value of the proposed payment.

        Returns:
            MultiHorizonCostEstimate with one CostBreakdown per horizon.
        """
        ...
