"""Tests for LinUCB candidate selector."""

from __future__ import annotations

import numpy as np
import pytest

from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import FEATURE_DIM, LinUCBArmState, LinUCBConfig
from app.bandit.selector import pick_candidate
from app.market_data.calibration import Calibration
from app.schemas import (
    HistoryStats,
    NetworkState,
    PaymentIntent,
    RamhdContext,
    TokenMarketSnapshot,
)


def _snap(
    symbol: str,
    *,
    vol: float = 0.05,
    liq: float = 5_000_000.0,
    spread: float = 10.0,
) -> TokenMarketSnapshot:
    return TokenMarketSnapshot(
        symbol=symbol,
        mint=f"mint-{symbol}",
        price_usd=100.0,
        balance=1.0,
        balance_usd=100.0,
        volatility_24h=vol,
        liquidity_depth_usd=liq,
        spread_bps=spread,
    )


def _ctx(
    symbols: list[str],
    *,
    amount_usd: float = 1000.0,
    congestion: float = 0.2,
) -> RamhdContext:
    tokens = [_snap(s) for s in symbols]
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


@pytest.fixture
def cal() -> Calibration:
    return Calibration()


@pytest.fixture
def bcal() -> BanditCalibration:
    return BanditCalibration()


def test_single_candidate_is_picked(cal: Calibration, bcal: BanditCalibration) -> None:
    ctx = _ctx(["SOL"])
    res = pick_candidate(
        ctx,
        ["SOL"],
        {},
        LinUCBConfig(),
        cal,
        bcal,
    )
    assert res.chosen_symbol == "SOL"
    assert set(res.ucb_scores) == {"SOL"}
    assert len(res.feature_vectors) == 1


def test_empty_candidates_raises(cal: Calibration, bcal: BanditCalibration) -> None:
    ctx = _ctx(["SOL"])
    with pytest.raises(ValueError, match="non-empty"):
        pick_candidate(ctx, [], {}, LinUCBConfig(), cal, bcal)


def test_picks_highest_ucb_among_three(cal: Calibration, bcal: BanditCalibration) -> None:
    ctx = _ctx(["SOL", "BONK", "USDC"])
    e0 = np.zeros(FEATURE_DIM)
    e0[0] = 1.0
    arms = {
        "SOL": LinUCBArmState(A=np.eye(FEATURE_DIM), b=1e6 * e0.copy(), n_updates=5),
        "BONK": LinUCBArmState(A=np.eye(FEATURE_DIM), b=-1e6 * e0.copy(), n_updates=5),
        "USDC": LinUCBArmState(A=np.eye(FEATURE_DIM), b=-1e6 * e0.copy(), n_updates=5),
    }
    res = pick_candidate(
        ctx,
        ["SOL", "BONK", "USDC"],
        arms,
        LinUCBConfig(alpha=0.01),
        cal,
        bcal,
    )
    assert res.chosen_symbol == "SOL"


def test_cold_start_arms_included(cal: Calibration, bcal: BanditCalibration) -> None:
    ctx = _ctx(["SOL", "BONK"])
    res = pick_candidate(
        ctx,
        ["SOL", "BONK"],
        {"SOL": LinUCBArmState.fresh(1.0)},
        LinUCBConfig(),
        cal,
        bcal,
    )
    assert "BONK" in res.ucb_scores
    assert "BONK" in res.feature_vectors


def test_selector_does_not_mutate_arms_dict(cal: Calibration, bcal: BanditCalibration) -> None:
    ctx = _ctx(["SOL", "BONK"])
    arms: dict[str, LinUCBArmState] = {"SOL": LinUCBArmState.fresh(1.0)}
    pick_candidate(
        ctx,
        ["SOL", "BONK"],
        arms,
        LinUCBConfig(),
        cal,
        bcal,
    )
    assert set(arms.keys()) == {"SOL"}


def test_feature_vectors_returned_for_all_candidates(
    cal: Calibration, bcal: BanditCalibration
) -> None:
    ctx = _ctx(["SOL", "BONK", "USDC"])
    res = pick_candidate(
        ctx,
        ["SOL", "BONK", "USDC"],
        {},
        LinUCBConfig(),
        cal,
        bcal,
    )
    assert len(res.feature_vectors) == 3
    for v in res.feature_vectors.values():
        assert v.shape == (FEATURE_DIM,)


def test_chosen_feature_vector_matches_chosen_symbol(
    cal: Calibration, bcal: BanditCalibration
) -> None:
    ctx = _ctx(["SOL", "BONK"])
    res = pick_candidate(ctx, ["SOL", "BONK"], {}, LinUCBConfig(), cal, bcal)
    assert np.array_equal(
        res.chosen_feature_vector,
        res.feature_vectors[res.chosen_symbol],
    )


def test_alphabetical_tiebreaker_propagates(
    cal: Calibration, bcal: BanditCalibration
) -> None:
    # Symbols not in calibration → is_stable=0 for all; identical snapshots → identical x.
    tokens = [
        _snap("ZZZ", vol=0.05, liq=5e6, spread=10.0),
        _snap("AAA", vol=0.05, liq=5e6, spread=10.0),
        _snap("MMM", vol=0.05, liq=5e6, spread=10.0),
    ]
    ctx = RamhdContext(
        intent=PaymentIntent(amount_usd=1000.0),
        tokens=tokens,
        network=NetworkState(
            priority_fee_lamports=1.0,
            congestion_score=0.2,
            slot_time_ms=400.0,
        ),
        history=HistoryStats(),
    )
    res = pick_candidate(
        ctx,
        ["ZZZ", "AAA", "MMM"],
        {},
        LinUCBConfig(),
        cal,
        bcal,
    )
    u = res.ucb_scores["ZZZ"]
    assert abs(u - res.ucb_scores["AAA"]) < 1e-9
    assert abs(u - res.ucb_scores["MMM"]) < 1e-9
    assert res.chosen_symbol == "AAA"


def test_symbol_not_in_context_raises_key_error(
    cal: Calibration, bcal: BanditCalibration
) -> None:
    ctx = _ctx(["SOL"])
    with pytest.raises(KeyError, match="not in context.tokens"):
        pick_candidate(ctx, ["BONK"], {}, LinUCBConfig(), cal, bcal)
