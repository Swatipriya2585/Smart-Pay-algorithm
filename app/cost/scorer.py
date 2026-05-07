"""
Solana cost & latency scorer.

Implements CostEstimator with closed-form slippage from constant-product
AMM math, deterministic gas calculation from Solana fee mechanics, and
settlement-risk derived from the GARCH forecast.

This is a v1 conservative estimator. It deliberately overestimates cost
(uses single-pool x*y=k slippage rather than multi-route Whirlpool/CLMM
math) so the algorithm errs toward caution. When we wire up live Jupiter
quotes in Step 13, we'll replace the slippage and gas calculations with
actual route costs from Jupiter's API; the Protocol contract stays the
same so nothing downstream changes.

Math:

1. Slippage (constant-product AMM, Uniswap v2 / Raydium classic):

       slippage_fraction = S / (S + D)

       slippage_dollar   = -S * slippage_fraction

   where S is swap size in USD and D is liquidity depth in USD.

2. Gas (Solana fee mechanics):

       total_lamports  = base_fee + priority_fee_per_cu * compute_units

       gas_dollar      = -(total_lamports / 1e9) * sol_price_usd

3. Settlement risk (GARCH-linked):

       settlement_seconds = slot_time_ms/1000 * congestion_multiplier

       per_second_vol     = forecast.at(5s).predicted_volatility / sqrt(5)

       settlement_vol     = per_second_vol * sqrt(settlement_seconds)

       settlement_risk    = -position_value * settlement_vol

References:

- Adams et al., "Uniswap v2 Core" (2020) — constant-product AMM derivation.
- Solana docs, "Transaction Fees" — fee structure and lamport units.
- Almgren & Chriss, "Optimal Execution of Portfolio Transactions" (2000)
  — settlement-window risk is the same idea applied to crypto payments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.cost.base import CostBreakdown, MultiHorizonCostEstimate
from app.forecasting.base import MultiHorizonForecast
from app.market_data.base import NetworkConditions, TokenMarketData


SOL_LAMPORTS = 1_000_000_000.0  # 1 SOL = 10^9 lamports
DEFAULT_BASE_FEE_LAMPORTS = 5_000.0
DEFAULT_COMPUTE_UNITS_PER_SWAP = 200_000.0


@dataclass(frozen=True)
class SolanaCostConfig:
    """Tuning knobs for the Solana cost scorer.

    Defaults reflect production-typical Solana behavior:

    - Base fee 5000 lamports per signature.

    - 200K compute units per swap is conservative; simple transfers use less,
      complex multi-hop routes can use more.

    - Sol price defaults to a recent reference price; in production this
      comes from the live SOL/USD oracle. Override per-call when fresh.

    - Congestion multiplier scales 1.0 (empty) to 5.0 (saturated) for
      settlement time. 5x slowdown at full saturation matches observed
      behavior during 2024-2025 Solana congestion events.
    """

    base_fee_lamports: float = DEFAULT_BASE_FEE_LAMPORTS
    compute_units_per_swap: float = DEFAULT_COMPUTE_UNITS_PER_SWAP
    sol_price_usd: float = 150.0
    max_congestion_settlement_multiplier: float = 5.0
    settlement_risk_horizon_seconds: float = 5.0
    """Which forecast horizon to derive per-second vol from. Default = 5s
    (the shortest in the standard RAMHD horizon set), giving the most
    responsive vol estimate."""

    def __post_init__(self) -> None:
        if self.base_fee_lamports < 0:
            raise ValueError("base_fee_lamports must be non-negative")
        if self.compute_units_per_swap <= 0:
            raise ValueError("compute_units_per_swap must be positive")
        if self.sol_price_usd <= 0:
            raise ValueError("sol_price_usd must be positive")
        if self.max_congestion_settlement_multiplier < 1.0:
            raise ValueError(
                "max_congestion_settlement_multiplier must be >= 1.0 "
                "(congestion never speeds things up)"
            )
        if self.settlement_risk_horizon_seconds <= 0:
            raise ValueError("settlement_risk_horizon_seconds must be positive")


class SolanaCostScorer:
    """Closed-form cost & latency estimator for Solana token swaps.

    Usage:
        scorer = SolanaCostScorer()
        estimate = scorer.estimate(
            data=token_market_data,
            forecast=garch_forecast,
            network=current_network_conditions,
            position_value_usd=1000.0,
        )
        worst = estimate.worst_total_cost_dollar()
    """

    def __init__(self, config: SolanaCostConfig | None = None) -> None:
        self.config = config if config is not None else SolanaCostConfig()

    # -------------------------------------------------------------------
    # Public API: CostEstimator protocol
    # -------------------------------------------------------------------

    def estimate(
        self,
        data: TokenMarketData,
        forecast: MultiHorizonForecast,
        network: NetworkConditions,
        position_value_usd: float,
    ) -> MultiHorizonCostEstimate:
        if position_value_usd < 0:
            raise ValueError(
                f"position_value_usd must be non-negative, got {position_value_usd}"
            )

        # These three are the same across horizons — they depend only on
        # the trade size, network state, and current liquidity, not on the
        # forecast horizon. We compute once and reuse.
        slippage_dollar = self._compute_slippage(
            position_value_usd=position_value_usd,
            liquidity_depth_usd=data.liquidity_depth_usd,
        )
        gas_dollar = self._compute_gas(network=network)
        settlement_seconds = self._compute_settlement_seconds(network=network)

        # Settlement risk derives from the forecast's per-second vol, scaled
        # to the settlement window. Same value across all horizons because
        # we always settle on the same chain regardless of analysis horizon.
        settlement_risk_dollar = self._compute_settlement_risk(
            forecast=forecast,
            settlement_seconds=settlement_seconds,
            position_value_usd=position_value_usd,
        )

        breakdowns: dict[float, CostBreakdown] = {}
        for h_seconds in forecast.horizon_seconds_list():
            total = slippage_dollar + gas_dollar + settlement_risk_dollar
            breakdowns[h_seconds] = CostBreakdown(
                horizon_seconds=h_seconds,
                slippage_dollar=slippage_dollar,
                gas_dollar=gas_dollar,
                settlement_risk_dollar=settlement_risk_dollar,
                total_cost_dollar=total,
                settlement_seconds=settlement_seconds,
            )

        return MultiHorizonCostEstimate(
            symbol=data.symbol,
            position_value_usd=position_value_usd,
            breakdowns=breakdowns,
        )

    # -------------------------------------------------------------------
    # Slippage (constant-product AMM, closed form)
    # -------------------------------------------------------------------

    @staticmethod
    def _compute_slippage(
        position_value_usd: float, liquidity_depth_usd: float
    ) -> float:
        """Closed-form slippage cost for a constant-product AMM swap.

        Returns a non-positive dollar value (cost is value given up).
        Edge case: zero position size produces zero cost. Zero liquidity
        produces ~100% slippage (pathological — caller is responsible for
        not routing through dead pools).
        """
        if position_value_usd <= 0:
            return 0.0
        if liquidity_depth_usd <= 0:
            # Pathological — pool has no liquidity. Conservative: 100% loss.
            return -float(position_value_usd)
        slippage_fraction = position_value_usd / (
            position_value_usd + liquidity_depth_usd
        )
        return -position_value_usd * slippage_fraction

    # -------------------------------------------------------------------
    # Gas (Solana fee mechanics)
    # -------------------------------------------------------------------

    def _compute_gas(self, network: NetworkConditions) -> float:
        """Total transaction cost in USD given current network state.

        Returns a non-positive dollar value.
        """
        priority_lamports = (
            network.priority_fee_lamports * self.config.compute_units_per_swap
        )
        total_lamports = self.config.base_fee_lamports + priority_lamports
        sol_cost = total_lamports / SOL_LAMPORTS
        return -sol_cost * self.config.sol_price_usd

    # -------------------------------------------------------------------
    # Settlement seconds (slot time scaled by congestion)
    # -------------------------------------------------------------------

    def _compute_settlement_seconds(
        self, network: NetworkConditions
    ) -> float:
        """Expected wall-clock settlement time in seconds.

        Linearly interpolates between 1x (calm) and max_congestion_*x at
        full saturation.
        """
        base_seconds = network.slot_time_ms / 1000.0
        congestion_mult = 1.0 + (
            self.config.max_congestion_settlement_multiplier - 1.0
        ) * network.congestion_score
        return base_seconds * congestion_mult

    # -------------------------------------------------------------------
    # Settlement risk (forecast-linked)
    # -------------------------------------------------------------------

    def _compute_settlement_risk(
        self,
        forecast: MultiHorizonForecast,
        settlement_seconds: float,
        position_value_usd: float,
    ) -> float:
        """Dollar exposure during the settlement window (one-sigma estimate).

        The "two-sigma" or "tail" exposure lives in CVaR; this is just the
        baseline volatility-during-settlement number, used for cost ranking.
        """
        if position_value_usd <= 0:
            return 0.0

        # Use the configured horizon to derive per-second vol.
        # Variance scales linearly with time, so std scales with sqrt(time).
        try:
            ref_horizon = self.config.settlement_risk_horizon_seconds
            if ref_horizon not in forecast.horizons:
                # Fall back to the shortest available horizon.
                ref_horizon = forecast.horizon_seconds_list()[0]
            ref_forecast = forecast.at(ref_horizon)
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Cannot derive settlement risk: forecast has no usable horizon. "
                f"Available: {sorted(forecast.horizons)}"
            ) from e

        # Per-second std-dev: vol_h / sqrt(h_seconds)
        per_second_vol = ref_forecast.predicted_volatility / math.sqrt(
            ref_forecast.horizon_seconds
        )
        # Scale to settlement window
        settlement_vol = per_second_vol * math.sqrt(settlement_seconds)
        return -position_value_usd * settlement_vol
