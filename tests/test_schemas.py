"""Verify the RamhdContext schema accepts a well-formed payload and rejects malformed ones."""

import pytest
from pydantic import ValidationError

from app.schemas import (
    HistoryStats,
    NetworkState,
    PaymentIntent,
    RamhdContext,
    TokenMarketSnapshot,
)


def _valid_context() -> RamhdContext:
    return RamhdContext(
        intent=PaymentIntent(amount_usd=100.0),
        tokens=[
            TokenMarketSnapshot(
                symbol="SOL",
                mint="SOL",
                price_usd=180.0,
                balance=2.0,
                balance_usd=360.0,
                volatility_24h=0.045,
                liquidity_depth_usd=5_000_000.0,
                spread_bps=8.0,
            ),
            TokenMarketSnapshot(
                symbol="USDC",
                mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                price_usd=1.0,
                balance=250.0,
                balance_usd=250.0,
                volatility_24h=0.0005,
                liquidity_depth_usd=50_000_000.0,
                spread_bps=1.0,
            ),
        ],
        network=NetworkState(
            priority_fee_lamports=5000.0,
            congestion_score=0.3,
        ),
        history=HistoryStats(last_n_fills=10, success_rate=1.0),
    )


def test_valid_context_parses() -> None:
    ctx = _valid_context()
    assert ctx.intent.amount_usd == 100.0
    assert len(ctx.tokens) == 2
    assert ctx.tokens[0].symbol == "SOL"
    assert ctx.network.congestion_score == 0.3


def test_empty_token_list_rejected() -> None:
    with pytest.raises(ValidationError):
        RamhdContext(
            intent=PaymentIntent(amount_usd=100.0),
            tokens=[],
            network=NetworkState(priority_fee_lamports=5000.0, congestion_score=0.3),
        )


def test_negative_amount_rejected() -> None:
    with pytest.raises(ValidationError):
        PaymentIntent(amount_usd=-10.0)


def test_congestion_score_bounds() -> None:
    with pytest.raises(ValidationError):
        NetworkState(priority_fee_lamports=1000.0, congestion_score=1.5)
    with pytest.raises(ValidationError):
        NetworkState(priority_fee_lamports=1000.0, congestion_score=-0.1)
