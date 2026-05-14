"""Tests for run_reward_processor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from app.bandit.contracts import FEATURE_DIM, LinUCBConfig
from app.bandit.persistence import load_state
from app.feedback.contracts import RealizedOutcome, TradeStatus
from app.feedback.outbox import SQLiteOutboxStore
from app.feedback.outbox_record import BanditDecisionRecord, OutboxStatus
from app.feedback.processor import ProcessorStats, run_reward_processor


class FakeOutcomeSource:
    """Deterministic stand-in for OutcomeSource."""

    def __init__(self) -> None:
        self.outcomes: dict[str, RealizedOutcome] = {}
        self.raise_for: dict[str, Exception] = {}
        self.calls: list[str] = []

    def fetch_outcome(self, tx_id: str) -> Optional[RealizedOutcome]:
        self.calls.append(tx_id)
        if tx_id in self.raise_for:
            raise self.raise_for[tx_id]
        return self.outcomes.get(tx_id)


def _fv(value: float = 0.1) -> np.ndarray:
    return np.full(FEATURE_DIM, value, dtype=np.float64)


def build_pending_record(
    tx_id: str,
    symbol: str = "SOL",
    decision_utc: str = "2026-05-13T00:00:00+00:00",
    amount_usd: float = 1000.0,
) -> BanditDecisionRecord:
    return BanditDecisionRecord(
        tx_id=tx_id,
        chosen_symbol=symbol,
        chosen_feature_vector=_fv(),
        amount_usd=amount_usd,
        decision_utc=decision_utc,
    )


def _filled(tx_id: str, ret: float = 0.005, cost: float = -50.0) -> RealizedOutcome:
    return RealizedOutcome(
        tx_id=tx_id,
        status=TradeStatus.FILLED,
        realized_return=ret,
        realized_cost_dollar=cost,
        fill_fraction=1.0,
        observed_at_utc="2026-05-13T00:01:00+00:00",
    )


def _data_missing(tx_id: str) -> RealizedOutcome:
    return RealizedOutcome(
        tx_id=tx_id,
        status=TradeStatus.DATA_MISSING,
        realized_return=0.0,
        realized_cost_dollar=0.0,
        fill_fraction=0.0,
        observed_at_utc="2026-05-13T00:01:00+00:00",
    )


def _now() -> datetime:
    return datetime(2026, 5, 13, 0, 1, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg() -> LinUCBConfig:
    return LinUCBConfig()


# -----------------------------------------------------------------------------
# Single-record happy path
# -----------------------------------------------------------------------------


def test_processes_filled_outcome(tmp_path: Path, cfg: LinUCBConfig) -> None:
    outbox_path = tmp_path / "outbox.sqlite"
    state_path = tmp_path / "linucb_state.json"
    outbox = SQLiteOutboxStore(path=outbox_path)
    src = FakeOutcomeSource()
    try:
        outbox.append(build_pending_record("t1", symbol="SOL"))
        src.outcomes["t1"] = _filled("t1", ret=0.005, cost=-50.0)

        stats = run_reward_processor(
            outbox=outbox,
            outcome_source=src,
            linucb_config=cfg,
            state_path=state_path,
            now_utc=_now(),
        )
        assert stats.n_processed == 1
        assert stats.n_skipped == 0
        assert stats.n_expired == 0
        assert stats.n_still_pending == 0
        assert stats.n_errors == 0

        rec = outbox.fetch_by_tx_id("t1")
        assert rec is not None
        assert rec.status == OutboxStatus.PROCESSED
        # reward = 0.005 - 50/1000 = -0.045
        assert rec.reward == pytest.approx(-0.045, abs=1e-12)

        arms = load_state(cfg, path=state_path)
        assert "SOL" in arms
        assert arms["SOL"].n_updates == 1
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# Still-pending case
# -----------------------------------------------------------------------------


def test_outcome_not_ready_leaves_record_pending(
    tmp_path: Path, cfg: LinUCBConfig
) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    src = FakeOutcomeSource()
    state_path = tmp_path / "linucb_state.json"
    try:
        outbox.append(build_pending_record("t1"))
        stats = run_reward_processor(
            outbox=outbox,
            outcome_source=src,
            linucb_config=cfg,
            state_path=state_path,
            now_utc=_now(),
        )
        assert stats.n_still_pending == 1
        assert stats.n_processed == 0
        rec = outbox.fetch_by_tx_id("t1")
        assert rec is not None
        assert rec.status == OutboxStatus.PENDING
        assert not state_path.exists()
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# Skip on data_missing
# -----------------------------------------------------------------------------


def test_data_missing_outcome_skips_and_marks(
    tmp_path: Path, cfg: LinUCBConfig
) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    src = FakeOutcomeSource()
    try:
        outbox.append(build_pending_record("t1"))
        src.outcomes["t1"] = _data_missing("t1")
        stats = run_reward_processor(
            outbox=outbox,
            outcome_source=src,
            linucb_config=cfg,
            state_path=tmp_path / "s.json",
            now_utc=_now(),
        )
        assert stats.n_skipped == 1
        assert stats.n_processed == 0
        rec = outbox.fetch_by_tx_id("t1")
        assert rec is not None
        assert rec.status == OutboxStatus.SKIPPED
        assert rec.error is not None
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# Expiry
# -----------------------------------------------------------------------------


def test_expired_record_marked_expired(tmp_path: Path, cfg: LinUCBConfig) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    src = FakeOutcomeSource()
    try:
        # decision_utc is 1 hour before now; max_age=60s → expired.
        outbox.append(
            build_pending_record(
                "t1", decision_utc="2026-05-12T23:01:00+00:00"
            )
        )
        stats = run_reward_processor(
            outbox=outbox,
            outcome_source=src,
            linucb_config=cfg,
            state_path=tmp_path / "s.json",
            max_age_seconds=60.0,
            now_utc=_now(),
        )
        assert stats.n_expired == 1
        assert "t1" not in src.calls
        rec = outbox.fetch_by_tx_id("t1")
        assert rec is not None
        assert rec.status == OutboxStatus.EXPIRED
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# Multiple records mixed
# -----------------------------------------------------------------------------


def test_mixed_batch(tmp_path: Path, cfg: LinUCBConfig) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    src = FakeOutcomeSource()
    state_path = tmp_path / "state.json"
    try:
        now = _now()
        recent = (now - timedelta(seconds=10)).isoformat()
        old = (now - timedelta(seconds=3600)).isoformat()
        outbox.append(build_pending_record("a", decision_utc=recent))
        outbox.append(build_pending_record("b", decision_utc=recent))
        outbox.append(build_pending_record("c", decision_utc=recent))
        outbox.append(build_pending_record("d", decision_utc=old))
        outbox.append(build_pending_record("e", decision_utc=recent))

        src.outcomes["a"] = _filled("a")
        src.outcomes["b"] = _filled("b")
        # c → still pending (no outcome registered)
        # d → expired (old timestamp)
        src.outcomes["e"] = _data_missing("e")

        stats = run_reward_processor(
            outbox=outbox,
            outcome_source=src,
            linucb_config=cfg,
            state_path=state_path,
            max_age_seconds=600.0,
            now_utc=now,
        )
        assert stats.n_processed == 2
        assert stats.n_still_pending == 1
        assert stats.n_expired == 1
        assert stats.n_skipped == 1
        assert stats.n_errors == 0
        assert stats.n_pending_at_start == 5
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------


def test_outcome_source_exception_leaves_record_pending(
    tmp_path: Path, cfg: LinUCBConfig
) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    src = FakeOutcomeSource()
    try:
        outbox.append(build_pending_record("t1"))
        src.raise_for["t1"] = RuntimeError("oracle exploded")
        stats = run_reward_processor(
            outbox=outbox,
            outcome_source=src,
            linucb_config=cfg,
            state_path=tmp_path / "s.json",
            now_utc=_now(),
        )
        assert stats.n_errors == 1
        assert stats.n_pending_at_start == 1
        rec = outbox.fetch_by_tx_id("t1")
        assert rec is not None
        assert rec.status == OutboxStatus.PENDING
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# Idempotency
# -----------------------------------------------------------------------------


def test_processor_idempotent_when_no_new_data(
    tmp_path: Path, cfg: LinUCBConfig
) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    src = FakeOutcomeSource()
    try:
        for _ in range(2):
            stats = run_reward_processor(
                outbox=outbox,
                outcome_source=src,
                linucb_config=cfg,
                state_path=tmp_path / "s.json",
                now_utc=_now(),
            )
            assert stats.n_processed == 0
            assert stats.n_skipped == 0
            assert stats.n_expired == 0
            assert stats.n_still_pending == 0
            assert stats.n_errors == 0
            assert stats.n_pending_at_start == 0
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# Stats correctness
# -----------------------------------------------------------------------------


def test_stats_elapsed_seconds_is_positive(
    tmp_path: Path, cfg: LinUCBConfig
) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    src = FakeOutcomeSource()
    try:
        stats = run_reward_processor(
            outbox=outbox,
            outcome_source=src,
            linucb_config=cfg,
            state_path=tmp_path / "s.json",
            now_utc=_now(),
        )
        assert stats.elapsed_seconds > 0
        assert stats.elapsed_seconds < 5.0
    finally:
        outbox.close()


def test_stats_n_pending_at_start_includes_all_pending(
    tmp_path: Path, cfg: LinUCBConfig
) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    src = FakeOutcomeSource()
    try:
        now = _now()
        recent = (now - timedelta(seconds=5)).isoformat()
        for i in range(5):
            outbox.append(build_pending_record(f"t{i}", decision_utc=recent))
        src.outcomes["t0"] = _filled("t0")
        src.outcomes["t1"] = _data_missing("t1")
        stats = run_reward_processor(
            outbox=outbox,
            outcome_source=src,
            linucb_config=cfg,
            state_path=tmp_path / "s.json",
            now_utc=now,
        )
        assert stats.n_pending_at_start == 5
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# Bandit state update integration
# -----------------------------------------------------------------------------


def test_processor_updates_existing_arm(
    tmp_path: Path, cfg: LinUCBConfig
) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    src = FakeOutcomeSource()
    state_path = tmp_path / "state.json"
    try:
        now = _now()
        recent = (now - timedelta(seconds=5)).isoformat()
        outbox.append(build_pending_record("t1", symbol="SOL", decision_utc=recent))
        src.outcomes["t1"] = _filled("t1")
        run_reward_processor(
            outbox=outbox,
            outcome_source=src,
            linucb_config=cfg,
            state_path=state_path,
            now_utc=now,
        )
        arms = load_state(cfg, path=state_path)
        assert arms["SOL"].n_updates == 1

        outbox.append(build_pending_record("t2", symbol="SOL", decision_utc=recent))
        src.outcomes["t2"] = _filled("t2")
        run_reward_processor(
            outbox=outbox,
            outcome_source=src,
            linucb_config=cfg,
            state_path=state_path,
            now_utc=now,
        )
        arms = load_state(cfg, path=state_path)
        assert arms["SOL"].n_updates == 2
    finally:
        outbox.close()
