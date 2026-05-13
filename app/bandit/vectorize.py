"""Feature vectorization for LinUCB (context × candidate → R^7)."""

from __future__ import annotations

import math

import numpy as np

from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import FEATURE_DIM, FEATURE_NAMES
from app.market_data.calibration import Calibration
from app.schemas import RamhdContext, TokenMarketSnapshot


def get_snapshot_by_symbol(
    context: RamhdContext,
    symbol: str,
) -> TokenMarketSnapshot:
    """Find the TokenMarketSnapshot for a given symbol in the context.

    Raises KeyError with a helpful message if not present.
    """
    for snap in context.tokens:
        if snap.symbol == symbol:
            return snap
    available = sorted({t.symbol for t in context.tokens})
    raise KeyError(
        f"symbol {symbol!r} not in context.tokens. Available: {available}"
    )


def build_feature_vector(
    context: RamhdContext,
    symbol: str,
    calibration: Calibration,
    bandit_calibration: BanditCalibration,
) -> np.ndarray:
    """Build the 7-dim feature vector for one (context, candidate) pair.

    Returns:
        np.ndarray of shape (FEATURE_DIM,), dtype float64.

    Raises:
        KeyError: if symbol is not in context.tokens.
        ValueError: if any computed feature is non-finite (NaN/inf).
    """
    snap = get_snapshot_by_symbol(context, symbol)
    amount = float(context.intent.amount_usd)

    log_amount = math.log10(1.0 + amount) / bandit_calibration.log_amount_divisor
    congestion = float(context.network.congestion_score)
    vol = min(float(snap.volatility_24h), bandit_calibration.vol_clip_max)
    volatility = vol / bandit_calibration.vol_clip_max
    ratio = float(snap.liquidity_depth_usd) / amount
    liquidity_ratio = math.log10(1.0 + ratio) / bandit_calibration.liquidity_ratio_log_divisor
    spread_raw = min(
        float(snap.spread_bps), bandit_calibration.spread_clip_max_bps
    )
    spread = spread_raw / bandit_calibration.spread_clip_max_bps
    is_stable = (
        1.0
        if calibration.has(symbol) and calibration.get(symbol).is_stablecoin
        else 0.0
    )
    bias = 1.0

    vector = np.array(
        [
            log_amount,
            congestion,
            volatility,
            liquidity_ratio,
            spread,
            is_stable,
            bias,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(vector)):
        bad_idx = int(np.where(~np.isfinite(vector))[0][0])
        name = FEATURE_NAMES[bad_idx]
        raise ValueError(
            f"non-finite feature at index {bad_idx} ({name}): {vector[bad_idx]}"
        )
    return vector
