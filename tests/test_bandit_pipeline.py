"""Tests for LinUCB pipeline (Pareto survivors → decision; reward → persistence)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import FEATURE_DIM, LinUCBArmState, LinUCBConfig
from app.bandit.persistence import load_state, save_state
from app.bandit.pipeline import BanditDecision, record_observation, run_bandit_stage
from app.market_data.calibration import Calibration
from app.pareto.contracts import CandidateScore
from app.schemas import (
    HistoryStats,
    NetworkState,
    PaymentIntent,
    RamhdContext,
    TokenMarketSnapshot,
)


def _score(symbol: str) -> CandidateScore:
    return CandidateScore(
        symbol=symbol,
        expected_return_120s=0.012,
        cvar_95_120s=-0.03,
        effective_cost_bps=35.0,
        liquidity_usd=2_500_000.0,
    )


def _ctx(symbols: list[str], *, amount_usd: float = 1000.0) -> RamhdContext:
    tokens = [
        TokenMarketSnapshot(
            symbol=sym,
            mint=f"mint-{sym}",
            price_usd=100.0 if sym != "USDC" else 1.0,
            balance=10.0,
            balance_usd=1000.0 if sym != "USDC" else 10.0,
            volatility_24h=0.04 if sym != "USDC" else 0.002,
            liquidity_depth_usd=5_000_000.0,
            spread_bps=8.0,
        )
        for sym in symbols
    ]
    return RamhdContext(
        intent=PaymentIntent(amount_usd=amount_usd),
        tokens=tokens,
        network=NetworkState(
            priority_fee_lamports=1.0,
            congestion_score=0.15,
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


@pytest.fixture
def cfg() -> LinUCBConfig:
    return LinUCBConfig()


def test_returns_decision_with_chosen_symbol_from_survivors(
    tmp_path: Path, cal: Calibration, bcal: BanditCalibration, cfg: LinUCBConfig
) -> None:
    state = tmp_path / "linucb_state.json"
    ctx = _ctx(["SOL", "USDC", "BONK"])
    survivors = [_score("SOL"), _score("USDC"), _score("BONK")]
    dec = run_bandit_stage(ctx, survivors, cfg, cal, bcal, state_path=state)
    assert dec.chosen_symbol in {"SOL", "USDC", "BONK"}


def test_decision_includes_all_candidates_in_diagnostics(
    tmp_path: Path, cal: Calibration, bcal: BanditCalibration, cfg: LinUCBConfig
) -> None:
    state = tmp_path / "s.json"
    ctx = _ctx(["SOL", "PYTH", "JUP"])
    survivors = [_score("SOL"), _score("PYTH"), _score("JUP")]
    dec = run_bandit_stage(ctx, survivors, cfg, cal, bcal, state_path=state)
    assert len(dec.ucb_scores) == 3
    assert len(dec.feature_vectors) == 3
    assert len(dec.candidates_evaluated) == 3


def test_candidates_evaluated_preserves_input_order(
    tmp_path: Path, cal: Calibration, bcal: BanditCalibration, cfg: LinUCBConfig
) -> None:
    state = tmp_path / "s.json"
    ctx = _ctx(["SOL", "USDC", "BONK"])
    survivors = [_score("SOL"), _score("USDC"), _score("BONK")]
    dec = run_bandit_stage(ctx, survivors, cfg, cal, bcal, state_path=state)
    assert dec.candidates_evaluated == ("SOL", "USDC", "BONK")


def test_empty_survivors_raises_value_error(
    tmp_path: Path, cal: Calibration, bcal: BanditCalibration, cfg: LinUCBConfig
) -> None:
    state = tmp_path / "s.json"
    ctx = _ctx(["SOL"])
    with pytest.raises(ValueError, match="non-empty"):
        run_bandit_stage(ctx, [], cfg, cal, bcal, state_path=state)


def test_cold_start_uses_fresh_arms(
    tmp_path: Path, cal: Calibration, bcal: BanditCalibration, cfg: LinUCBConfig
) -> None:
    state = tmp_path / "s.json"
    ctx = _ctx(["AERO", "WIF", "BRETT"])
    survivors = [_score("AERO"), _score("WIF"), _score("BRETT")]
    dec = run_bandit_stage(ctx, survivors, cfg, cal, bcal, state_path=state)
    for u in dec.ucb_scores.values():
        assert np.isfinite(u)
        assert u > 0


def test_decision_does_not_save_state(
    tmp_path: Path, cal: Calibration, bcal: BanditCalibration, cfg: LinUCBConfig
) -> None:
    state = tmp_path / "linucb_state.json"
    assert not state.exists()
    ctx = _ctx(["SOL", "BONK"])
    run_bandit_stage(
        ctx, [_score("SOL"), _score("BONK")], cfg, cal, bcal, state_path=state
    )
    assert not state.exists()


def test_explicit_now_utc_iso_used_in_decision(
    tmp_path: Path, cal: Calibration, bcal: BanditCalibration, cfg: LinUCBConfig
) -> None:
    state = tmp_path / "s.json"
    ctx = _ctx(["SOL", "JUP"])
    ts = "2026-05-13T12:00:00+00:00"
    dec = run_bandit_stage(
        ctx,
        [_score("SOL"), _score("JUP")],
        cfg,
        cal,
        bcal,
        state_path=state,
        now_utc_iso=ts,
    )
    assert dec.decision_utc == ts


def test_decision_returns_chosen_feature_vector_matching_dict(
    tmp_path: Path, cal: Calibration, bcal: BanditCalibration, cfg: LinUCBConfig
) -> None:
    state = tmp_path / "s.json"
    ctx = _ctx(["SOL", "USDC"])
    dec = run_bandit_stage(ctx, [_score("SOL"), _score("USDC")], cfg, cal, bcal, state_path=state)
    assert np.array_equal(
        dec.chosen_feature_vector,
        dec.feature_vectors[dec.chosen_symbol],
    )


def test_pipeline_integration_after_pareto(
    tmp_path: Path, cal: Calibration, bcal: BanditCalibration, cfg: LinUCBConfig
) -> None:
    state = tmp_path / "s.json"
    ctx = _ctx(["SOL", "PYTH", "AERO"])
    pareto_like = [_score("PYTH"), _score("AERO")]
    dec = run_bandit_stage(ctx, pareto_like, cfg, cal, bcal, state_path=state)
    assert dec.chosen_symbol in {"PYTH", "AERO"}
    assert isinstance(dec, BanditDecision)


def test_record_updates_chosen_arm_and_persists(
    tmp_path: Path, cal: Calibration, bcal: BanditCalibration, cfg: LinUCBConfig
) -> None:
    state = tmp_path / "linucb_state.json"
    ctx = _ctx(["SOL", "BONK", "USDC"])
    dec = run_bandit_stage(
        ctx,
        [_score("SOL"), _score("BONK"), _score("USDC")],
        cfg,
        cal,
        bcal,
        state_path=state,
    )
    assert not state.exists()
    record_observation(
        dec.chosen_symbol,
        dec.chosen_feature_vector,
        0.01,
        cfg,
        state_path=state,
        now_utc_iso="2026-01-01T00:00:00+00:00",
    )
    assert state.exists()
    arms = load_state(cfg, path=state)
    assert dec.chosen_symbol in arms
    assert arms[dec.chosen_symbol].n_updates == 1


def test_record_does_not_update_non_chosen_arms(
    tmp_path: Path, cfg: LinUCBConfig
) -> None:
    state = tmp_path / "st.json"
    bonk = LinUCBArmState(
        A=np.eye(FEATURE_DIM) * 3.0,
        b=np.linspace(0.2, 0.8, FEATURE_DIM),
        n_updates=5,
        last_update_utc="2025-01-01T00:00:00Z",
    )
    sol = LinUCBArmState(
        A=np.eye(FEATURE_DIM) * 1.1,
        b=np.linspace(-0.1, 0.3, FEATURE_DIM),
        n_updates=3,
        last_update_utc=None,
    )
    save_state({"BONK": bonk, "SOL": sol}, cfg, path=state, now_utc_iso="t0")

    x = np.arange(FEATURE_DIM, dtype=np.float64) * 0.01
    record_observation("SOL", x, 0.5, cfg, state_path=state, now_utc_iso="t1")

    arms = load_state(cfg, path=state)
    assert np.allclose(arms["BONK"].A, bonk.A)
    assert np.allclose(arms["BONK"].b, bonk.b)
    assert arms["BONK"].n_updates == bonk.n_updates
    assert arms["SOL"].n_updates == 4


def test_record_creates_arm_for_unknown_chosen_symbol(
    tmp_path: Path, cfg: LinUCBConfig
) -> None:
    state = tmp_path / "x.json"
    assert not state.exists()
    x = np.ones(FEATURE_DIM, dtype=np.float64) * 0.02
    record_observation("NEVER_SEEN", x, 1.0, cfg, state_path=state, now_utc_iso="u1")
    arms = load_state(cfg, path=state)
    assert "NEVER_SEEN" in arms
    assert arms["NEVER_SEEN"].n_updates == 1


def test_record_reward_nan_raises(tmp_path: Path, cfg: LinUCBConfig) -> None:
    state = tmp_path / "n.json"
    x = np.ones(FEATURE_DIM)
    with pytest.raises(ValueError, match="finite"):
        record_observation("SOL", x, float("nan"), cfg, state_path=state)
    assert not state.exists()


def test_record_reward_inf_raises(tmp_path: Path, cfg: LinUCBConfig) -> None:
    state = tmp_path / "i.json"
    x = np.ones(FEATURE_DIM)
    with pytest.raises(ValueError, match="finite"):
        record_observation("SOL", x, float("inf"), cfg, state_path=state)
    assert not state.exists()


def test_record_wrong_shape_feature_vector_raises(tmp_path: Path, cfg: LinUCBConfig) -> None:
    state = tmp_path / "w.json"
    with pytest.raises(ValueError, match="shape"):
        record_observation("SOL", np.zeros(5), 1.0, cfg, state_path=state)
    assert not state.exists()


def test_record_persists_with_correct_config_hash(tmp_path: Path) -> None:
    cfg_a = LinUCBConfig(alpha=1.0)
    cfg_b = LinUCBConfig(alpha=2.0)
    state = tmp_path / "h.json"
    save_state({}, cfg_a, path=state, now_utc_iso="a")
    x = np.ones(FEATURE_DIM) * 0.03
    record_observation("SOL", x, 0.1, cfg_a, state_path=state, now_utc_iso="b")
    load_state(cfg_a, path=state)
    with pytest.raises(ValueError, match="config_hash"):
        load_state(cfg_b, path=state)


def test_record_full_round_trip_two_observations(
    tmp_path: Path, cal: Calibration, bcal: BanditCalibration, cfg: LinUCBConfig
) -> None:
    state = tmp_path / "r.json"
    ctx1 = _ctx(["SOL", "BONK"])
    d1 = run_bandit_stage(ctx1, [_score("SOL"), _score("BONK")], cfg, cal, bcal, state_path=state)
    record_observation(
        d1.chosen_symbol,
        d1.chosen_feature_vector,
        0.02,
        cfg,
        state_path=state,
    )
    ctx2 = _ctx(["SOL", "BONK"], amount_usd=50_000.0)
    d2 = run_bandit_stage(ctx2, [_score("SOL"), _score("BONK")], cfg, cal, bcal, state_path=state)
    record_observation(
        d2.chosen_symbol,
        d2.chosen_feature_vector,
        -0.01,
        cfg,
        state_path=state,
    )
    arms = load_state(cfg, path=state)
    total = sum(a.n_updates for a in arms.values())
    assert total == 2
