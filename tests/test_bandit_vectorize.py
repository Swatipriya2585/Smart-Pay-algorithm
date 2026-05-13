"""Tests for bandit feature vectorization."""

from __future__ import annotations

import numpy as np
import pytest

from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import FEATURE_DIM
from app.bandit.vectorize import build_feature_vector, get_snapshot_by_symbol
from app.market_data.calibration import Calibration
from app.schemas import (
    HistoryStats,
    NetworkState,
    PaymentIntent,
    RamhdContext,
    TokenMarketSnapshot,
)


def _make_context(
    *,
    amount_usd: float = 1000.0,
    congestion: float = 0.2,
    tokens: list[TokenMarketSnapshot] | None = None,
) -> RamhdContext:
    if tokens is None:
        tokens = [
            TokenMarketSnapshot(
                symbol="SOL",
                mint="So11111111111111111111111111111111111111112",
                price_usd=100.0,
                balance=1.0,
                balance_usd=100.0,
                volatility_24h=0.05,
                liquidity_depth_usd=5_000_000.0,
                spread_bps=10.0,
            )
        ]
    return RamhdContext(
        intent=PaymentIntent(amount_usd=amount_usd),
        tokens=tokens,
        network=NetworkState(
            priority_fee_lamports=1.0,
            congestion_score=congestion,
            slot_time_ms=400.0,
        ),
        history=HistoryStats(),
    )


def test_returns_shape_7_float64() -> None:
    cal = Calibration()
    bcal = BanditCalibration()
    ctx = _make_context()
    v = build_feature_vector(ctx, "SOL", cal, bcal)
    assert v.shape == (FEATURE_DIM,)
    assert v.dtype == np.float64


def test_bias_slot_always_one() -> None:
    cal = Calibration()
    bcal = BanditCalibration()
    for amount in (1.0, 100.0, 10_000.0):
        ctx = _make_context(amount_usd=amount)
        v = build_feature_vector(ctx, "SOL", cal, bcal)
        assert v[6] == 1.0


def test_is_stable_true_for_usdc() -> None:
    cal = Calibration()
    bcal = BanditCalibration()
    ctx = _make_context(
        tokens=[
            TokenMarketSnapshot(
                symbol="USDC",
                mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                price_usd=1.0,
                balance=1000.0,
                balance_usd=1000.0,
                volatility_24h=0.001,
                liquidity_depth_usd=10_000_000.0,
                spread_bps=2.0,
            )
        ]
    )
    v = build_feature_vector(ctx, "USDC", cal, bcal)
    assert v[5] == 1.0


def test_is_stable_false_for_sol() -> None:
    cal = Calibration()
    bcal = BanditCalibration()
    ctx = _make_context()
    v = build_feature_vector(ctx, "SOL", cal, bcal)
    assert v[5] == 0.0


def test_is_stable_zero_for_unknown_symbol() -> None:
    cal = Calibration()
    bcal = BanditCalibration()
    ctx = _make_context(
        tokens=[
            TokenMarketSnapshot(
                symbol="UNKNOWN",
                mint="mint",
                price_usd=1.0,
                balance=1.0,
                balance_usd=1.0,
                volatility_24h=0.1,
                liquidity_depth_usd=1_000_000.0,
                spread_bps=10.0,
            )
        ]
    )
    v = build_feature_vector(ctx, "UNKNOWN", cal, bcal)
    assert v[5] == 0.0


def test_log_amount_monotonic() -> None:
    cal = Calibration()
    bcal = BanditCalibration()
    ctx_small = _make_context(amount_usd=100.0)
    ctx_large = _make_context(amount_usd=100_000.0)
    v_s = build_feature_vector(ctx_small, "SOL", cal, bcal)
    v_l = build_feature_vector(ctx_large, "SOL", cal, bcal)
    assert v_l[0] > v_s[0]


def test_liquidity_ratio_monotonic() -> None:
    cal = Calibration()
    bcal = BanditCalibration()
    snap_fields = dict(
        symbol="SOL",
        mint="So11111111111111111111111111111111111111112",
        price_usd=100.0,
        balance=1.0,
        balance_usd=100.0,
        volatility_24h=0.05,
        spread_bps=10.0,
    )
    low = TokenMarketSnapshot(liquidity_depth_usd=100_000.0, **snap_fields)
    high = TokenMarketSnapshot(liquidity_depth_usd=10_000_000.0, **snap_fields)
    ctx_low = _make_context(tokens=[low])
    ctx_high = _make_context(tokens=[high])
    v_lo = build_feature_vector(ctx_low, "SOL", cal, bcal)
    v_hi = build_feature_vector(ctx_high, "SOL", cal, bcal)
    assert v_hi[3] > v_lo[3]


def test_spread_clipping() -> None:
    cal = Calibration()
    bcal = BanditCalibration()
    ctx = _make_context(
        tokens=[
            TokenMarketSnapshot(
                symbol="SOL",
                mint="So11111111111111111111111111111111111111112",
                price_usd=100.0,
                balance=1.0,
                balance_usd=100.0,
                volatility_24h=0.05,
                liquidity_depth_usd=5_000_000.0,
                spread_bps=9999.0,
            )
        ]
    )
    v = build_feature_vector(ctx, "SOL", cal, bcal)
    assert abs(v[4] - 1.0) < 1e-12


def test_volatility_clipping() -> None:
    cal = Calibration()
    bcal = BanditCalibration()
    ctx = _make_context(
        tokens=[
            TokenMarketSnapshot(
                symbol="SOL",
                mint="So11111111111111111111111111111111111111112",
                price_usd=100.0,
                balance=1.0,
                balance_usd=100.0,
                volatility_24h=5.0,
                liquidity_depth_usd=5_000_000.0,
                spread_bps=10.0,
            )
        ]
    )
    v = build_feature_vector(ctx, "SOL", cal, bcal)
    assert abs(v[2] - 1.0) < 1e-12


def test_congestion_passthrough() -> None:
    cal = Calibration()
    bcal = BanditCalibration()
    ctx = _make_context(congestion=0.7)
    v = build_feature_vector(ctx, "SOL", cal, bcal)
    assert abs(v[1] - 0.7) < 1e-12


def test_symbol_not_in_context_raises_key_error() -> None:
    cal = Calibration()
    bcal = BanditCalibration()
    ctx = _make_context()
    with pytest.raises(KeyError, match="not in context.tokens"):
        build_feature_vector(ctx, "BONK", cal, bcal)


def test_get_snapshot_by_symbol_success() -> None:
    ctx = _make_context()
    snap = get_snapshot_by_symbol(ctx, "SOL")
    assert snap.symbol == "SOL"


def test_all_features_finite() -> None:
    cal = Calibration()
    bcal = BanditCalibration()
    edge = TokenMarketSnapshot(
        symbol="EDGE",
        mint="m",
        price_usd=1.0,
        balance=0.0,
        balance_usd=0.0,
        volatility_24h=0.0,
        liquidity_depth_usd=0.0,
        spread_bps=0.0,
    )
    ctx = _make_context(amount_usd=1.0, tokens=[edge])
    v = build_feature_vector(ctx, "EDGE", cal, bcal)
    assert np.all(np.isfinite(v))
