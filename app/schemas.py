"""
Pydantic schemas for the RAMHD context vector x in R^d.

This is Stage 0 of the algorithm pipeline. Every downstream scorer
(forecaster, CVaR, cost, regime) reads from the same context object.
No business logic here — just the data contract.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat


class PaymentIntent(BaseModel):
    """What the user is trying to pay."""

    amount_usd: PositiveFloat = Field(..., description="Payment amount in USD")
    deadline_seconds: PositiveFloat = Field(
        default=120.0,
        description="How long the recommendation is valid. Defaults to 2 minutes.",
    )
    max_slippage_bps: NonNegativeFloat = Field(
        default=50.0,
        description="Max acceptable slippage in basis points (50 = 0.5%).",
    )
    purpose: str = Field(default="payment", description="Free-text purpose tag.")


class TokenMarketSnapshot(BaseModel):
    """Current market state for a single token the user holds."""

    symbol: str
    mint: str = Field(..., description="Solana SPL mint address or 'SOL' for native.")
    price_usd: PositiveFloat
    balance: NonNegativeFloat = Field(..., description="User's balance in token units.")
    balance_usd: NonNegativeFloat = Field(..., description="balance * price_usd.")
    volatility_24h: NonNegativeFloat = Field(
        ..., description="Realized 24h return standard deviation (decimal, not %)."
    )
    liquidity_depth_usd: NonNegativeFloat = Field(
        ..., description="USD available within 1% price impact on primary DEX."
    )
    spread_bps: NonNegativeFloat = Field(
        ..., description="Bid-ask spread in basis points from Jupiter quote."
    )


class NetworkState(BaseModel):
    """Solana chain conditions that affect execution cost and reliability."""

    priority_fee_lamports: NonNegativeFloat = Field(
        ..., description="Current recommended priority fee per compute unit."
    )
    congestion_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized network congestion. 0 = empty, 1 = saturated.",
    )
    slot_time_ms: PositiveFloat = Field(
        default=400.0, description="Observed average slot time."
    )


class HistoryStats(BaseModel):
    """Rolling statistics from the user's recent payment history."""

    last_n_fills: int = Field(default=0, ge=0)
    avg_realized_slippage_bps: NonNegativeFloat = Field(default=0.0)
    success_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Fraction of recent txs that finalized."
    )


class RamhdContext(BaseModel):
    """The full context vector x in R^d passed to every RAMHD stage."""

    intent: PaymentIntent
    tokens: list[TokenMarketSnapshot] = Field(
        ..., min_length=1, description="User's held tokens, each with a market snapshot."
    )
    network: NetworkState
    history: HistoryStats = Field(default_factory=HistoryStats)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
