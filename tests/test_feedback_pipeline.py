"""Tests for record_decision and the end-to-end feedback loop."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import LinUCBConfig
from app.bandit.persistence import load_state
from app.feedback.contracts import RewardConfig
from app.feedback.outbox import OutboxStore, SQLiteOutboxStore
from app.feedback.outbox_record import OutboxStatus
from app.feedback.outcome_source import MockOutcomeSource
from app.feedback.pipeline import RecordedDecision, record_decision
from app.feedback.processor import run_reward_processor
from app.feedback.reward import compute_reward
from app.forecasting.base import HorizonForecast, MultiHorizonForecast
from app.market_data.calibration import Calibration
from app.pareto.contracts import CandidateScore
from app.schemas import (
    HistoryStats,
    NetworkState,
    PaymentIntent,
    RamhdContext,
    TokenMarketSnapshot,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def build_survivors(symbols: list[str]) -> list[CandidateScore]:
    return [
        CandidateScore(
            symbol=sym,
            expected_return_120s=0.012,
            cvar_95_120s=-0.03,
            effective_cost_bps=35.0,
            liquidity_usd=2_500_000.0,
        )
        for sym in symbols
    ]


def build_context(
    symbols: list[str],
    *,
    amount_usd: float = 1000.0,
    congestion: float = 0.15,
) -> RamhdContext:
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
            congestion_score=congestion,
            slot_time_ms=400.0,
        ),
        history=HistoryStats(),
    )


def load_real_calibrations() -> tuple[Calibration, BanditCalibration]:
    return Calibration(), BanditCalibration()


def _forecast_for(symbol: str, mu: float = 0.005) -> MultiHorizonForecast:
    horizons = {}
    for h in (5.0, 30.0, 120.0):
        horizons[h] = HorizonForecast(
            horizon_seconds=h,
            predicted_return=mu,
            predicted_volatility=0.001,
            confidence_lower_95=mu - 0.01,
            confidence_upper_95=mu + 0.01,
        )
    return MultiHorizonForecast(symbol=symbol, horizons=horizons)


@pytest.fixture
def cfg() -> LinUCBConfig:
    return LinUCBConfig()


@pytest.fixture
def cals() -> tuple[Calibration, BanditCalibration]:
    return load_real_calibrations()


# -----------------------------------------------------------------------------
# record_decision basics
# -----------------------------------------------------------------------------


def test_record_decision_runs_bandit_and_writes_outbox(
    tmp_path: Path,
    cfg: LinUCBConfig,
    cals: tuple[Calibration, BanditCalibration],
) -> None:
    cal, bcal = cals
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        ctx = build_context(["SOL", "USDC", "BONK"])
        survivors = build_survivors(["SOL", "USDC", "BONK"])
        result = record_decision(
            tx_id="t-001",
            context=ctx,
            survivors=survivors,
            config=cfg,
            calibration=cal,
            bandit_calibration=bcal,
            outbox=outbox,
            state_path=tmp_path / "linucb_state.json",
        )
        assert result.tx_id == "t-001"
        assert result.outbox_write_succeeded is True
        assert result.decision.chosen_symbol in {s.symbol for s in survivors}

        rec = outbox.fetch_by_tx_id("t-001")
        assert rec is not None
        assert rec.chosen_symbol == result.decision.chosen_symbol
        assert rec.status == OutboxStatus.PENDING
        assert np.allclose(
            rec.chosen_feature_vector,
            result.decision.chosen_feature_vector,
        )
        assert rec.amount_usd == 1000.0
    finally:
        outbox.close()


def test_record_decision_preserves_decision_fields(
    tmp_path: Path,
    cfg: LinUCBConfig,
    cals: tuple[Calibration, BanditCalibration],
) -> None:
    cal, bcal = cals
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        ctx = build_context(["SOL", "USDC", "JUP"])
        survivors = build_survivors(["SOL", "USDC", "JUP"])
        result = record_decision(
            tx_id="t-fields",
            context=ctx,
            survivors=survivors,
            config=cfg,
            calibration=cal,
            bandit_calibration=bcal,
            outbox=outbox,
            state_path=tmp_path / "state.json",
            now_utc_iso="2026-05-13T12:00:00+00:00",
        )
        d = result.decision
        assert len(d.ucb_scores) == 3
        assert len(d.feature_vectors) == 3
        assert d.candidates_evaluated == ("SOL", "USDC", "JUP")
        assert d.decision_utc == "2026-05-13T12:00:00+00:00"
    finally:
        outbox.close()


def test_record_decision_empty_tx_id_raises(
    tmp_path: Path,
    cfg: LinUCBConfig,
    cals: tuple[Calibration, BanditCalibration],
) -> None:
    cal, bcal = cals
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        with pytest.raises(ValueError, match="tx_id"):
            record_decision(
                tx_id="",
                context=build_context(["SOL"]),
                survivors=build_survivors(["SOL"]),
                config=cfg,
                calibration=cal,
                bandit_calibration=bcal,
                outbox=outbox,
                state_path=tmp_path / "s.json",
            )
    finally:
        outbox.close()


def test_record_decision_empty_survivors_raises(
    tmp_path: Path,
    cfg: LinUCBConfig,
    cals: tuple[Calibration, BanditCalibration],
) -> None:
    cal, bcal = cals
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        with pytest.raises(ValueError, match="non-empty"):
            record_decision(
                tx_id="t-1",
                context=build_context(["SOL"]),
                survivors=[],
                config=cfg,
                calibration=cal,
                bandit_calibration=bcal,
                outbox=outbox,
                state_path=tmp_path / "s.json",
            )
    finally:
        outbox.close()


def test_record_decision_duplicate_tx_id_returns_false_flag(
    tmp_path: Path,
    cfg: LinUCBConfig,
    cals: tuple[Calibration, BanditCalibration],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger="app.feedback.pipeline")
    cal, bcal = cals
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        ctx = build_context(["SOL", "USDC"])
        survivors = build_survivors(["SOL", "USDC"])
        kwargs = dict(
            context=ctx,
            survivors=survivors,
            config=cfg,
            calibration=cal,
            bandit_calibration=bcal,
            outbox=outbox,
            state_path=tmp_path / "s.json",
        )
        r1 = record_decision(tx_id="dup", **kwargs)
        r2 = record_decision(tx_id="dup", **kwargs)
        assert r1.outbox_write_succeeded is True
        assert r2.outbox_write_succeeded is False
        assert r2.decision.chosen_symbol in {s.symbol for s in survivors}
        assert any("dup" in rec.getMessage() for rec in caplog.records)
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# End-to-end closed loop
# -----------------------------------------------------------------------------


def test_full_feedback_loop_updates_bandit(
    tmp_path: Path,
    cfg: LinUCBConfig,
    cals: tuple[Calibration, BanditCalibration],
) -> None:
    cal, bcal = cals
    outbox_path = tmp_path / "outbox.sqlite"
    state_path = tmp_path / "linucb_state.json"
    outbox = SQLiteOutboxStore(path=outbox_path)
    try:
        ctx = build_context(["SOL", "USDC", "JUP"], amount_usd=1000.0)
        survivors = build_survivors(["SOL", "USDC", "JUP"])

        result = record_decision(
            tx_id="tx-001",
            context=ctx,
            survivors=survivors,
            config=cfg,
            calibration=cal,
            bandit_calibration=bcal,
            outbox=outbox,
            state_path=state_path,
        )
        assert result.outbox_write_succeeded is True
        assert not state_path.exists()
        chosen = result.decision.chosen_symbol

        # Mock outcome source with deterministic FILLED behavior.
        src = MockOutcomeSource(
            noise_std=0.0,
            failure_rate=0.0,
            timeout_rate=0.0,
            cost_dollar_per_trade=-50.0,
            rng_seed=42,
        )
        src.register_decision(
            "tx-001",
            result.decision,
            _forecast_for(chosen, mu=0.005),
            amount_usd=1000.0,
        )

        stats = run_reward_processor(
            outbox=outbox,
            outcome_source=src,
            linucb_config=cfg,
            state_path=state_path,
        )
        assert stats.n_processed == 1
        assert stats.n_skipped == 0
        assert stats.n_expired == 0
        assert stats.n_still_pending == 0
        assert stats.n_errors == 0

        arms = load_state(cfg, path=state_path)
        assert chosen in arms
        assert arms[chosen].n_updates == 1
        assert arms[chosen].last_update_utc is not None

        rec = outbox.fetch_by_tx_id("tx-001")
        assert rec is not None
        assert rec.status == OutboxStatus.PROCESSED

        # Hand-computed reward: realized_return=0.005, cost=-50, amount=1000.
        # reward = 0.005 - (50/1000) = -0.045
        expected_reward = 0.005 - (50.0 / 1000.0)
        assert rec.reward == pytest.approx(expected_reward, abs=1e-12)

        # Independent check via compute_reward on the same synthetic outcome.
        from app.feedback.contracts import RealizedOutcome, TradeStatus

        synth_outcome = RealizedOutcome(
            tx_id="tx-001",
            status=TradeStatus.FILLED,
            realized_return=0.005,
            realized_cost_dollar=-50.0,
            fill_fraction=1.0,
            observed_at_utc="2026-05-13T00:01:00+00:00",
        )
        assert compute_reward(synth_outcome, 1000.0, RewardConfig()) == pytest.approx(
            expected_reward, abs=1e-12
        )
    finally:
        outbox.close()


def test_two_decisions_two_observations_both_persist(
    tmp_path: Path,
    cfg: LinUCBConfig,
    cals: tuple[Calibration, BanditCalibration],
) -> None:
    cal, bcal = cals
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    state_path = tmp_path / "state.json"
    try:
        ctx = build_context(["SOL", "USDC", "JUP"])
        survivors = build_survivors(["SOL", "USDC", "JUP"])
        r1 = record_decision(
            tx_id="x1",
            context=ctx,
            survivors=survivors,
            config=cfg,
            calibration=cal,
            bandit_calibration=bcal,
            outbox=outbox,
            state_path=state_path,
        )
        r2 = record_decision(
            tx_id="x2",
            context=build_context(["SOL", "USDC", "JUP"], amount_usd=2000.0),
            survivors=survivors,
            config=cfg,
            calibration=cal,
            bandit_calibration=bcal,
            outbox=outbox,
            state_path=state_path,
        )

        src = MockOutcomeSource(
            noise_std=0.0,
            failure_rate=0.0,
            timeout_rate=0.0,
            cost_dollar_per_trade=-25.0,
            rng_seed=7,
        )
        src.register_decision("x1", r1.decision, _forecast_for(r1.decision.chosen_symbol), 1000.0)
        src.register_decision("x2", r2.decision, _forecast_for(r2.decision.chosen_symbol), 2000.0)

        stats = run_reward_processor(
            outbox=outbox,
            outcome_source=src,
            linucb_config=cfg,
            state_path=state_path,
        )
        assert stats.n_processed == 2

        assert outbox.fetch_by_tx_id("x1").status == OutboxStatus.PROCESSED  # type: ignore[union-attr]
        assert outbox.fetch_by_tx_id("x2").status == OutboxStatus.PROCESSED  # type: ignore[union-attr]
        arms = load_state(cfg, path=state_path)
        total_updates = sum(arm.n_updates for arm in arms.values())
        assert total_updates == 2
    finally:
        outbox.close()


def test_decision_without_observation_stays_pending(
    tmp_path: Path,
    cfg: LinUCBConfig,
    cals: tuple[Calibration, BanditCalibration],
) -> None:
    cal, bcal = cals
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    state_path = tmp_path / "state.json"
    try:
        ctx = build_context(["SOL", "USDC"])
        survivors = build_survivors(["SOL", "USDC"])
        record_decision(
            tx_id="lone",
            context=ctx,
            survivors=survivors,
            config=cfg,
            calibration=cal,
            bandit_calibration=bcal,
            outbox=outbox,
            state_path=state_path,
        )
        src = MockOutcomeSource(rng_seed=1)
        stats = run_reward_processor(
            outbox=outbox,
            outcome_source=src,
            linucb_config=cfg,
            state_path=state_path,
        )
        assert stats.n_still_pending == 1
        rec = outbox.fetch_by_tx_id("lone")
        assert rec is not None
        assert rec.status == OutboxStatus.PENDING
        assert not state_path.exists()
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# Protocol compliance smoke
# -----------------------------------------------------------------------------


def test_outbox_store_protocol_compliance(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        assert isinstance(outbox, OutboxStore)
    finally:
        outbox.close()
