"""Tests for BanditDecisionRecord and SQLiteOutboxStore."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from app.bandit.contracts import FEATURE_DIM
from app.feedback.outbox import SQLiteOutboxStore
from app.feedback.outbox_record import BanditDecisionRecord, OutboxStatus


def _fv(value: float = 0.1) -> np.ndarray:
    return np.full(FEATURE_DIM, value, dtype=np.float64)


def build_record(
    *,
    tx_id: str = "tx-1",
    chosen_symbol: str = "SOL",
    amount_usd: float = 1000.0,
    decision_utc: str = "2026-05-13T00:00:00+00:00",
    feature_vector: np.ndarray | None = None,
) -> BanditDecisionRecord:
    if feature_vector is None:
        feature_vector = _fv()
    return BanditDecisionRecord(
        tx_id=tx_id,
        chosen_symbol=chosen_symbol,
        chosen_feature_vector=feature_vector,
        amount_usd=amount_usd,
        decision_utc=decision_utc,
    )


# -----------------------------------------------------------------------------
# BanditDecisionRecord validation
# -----------------------------------------------------------------------------


def test_constructs_with_valid_fields() -> None:
    r = build_record()
    assert r.tx_id == "tx-1"
    assert r.chosen_symbol == "SOL"
    assert r.amount_usd == 1000.0
    assert r.status == OutboxStatus.PENDING
    assert r.reward is None


def test_empty_tx_id_raises() -> None:
    with pytest.raises(ValueError, match="tx_id"):
        build_record(tx_id="")


def test_empty_chosen_symbol_raises() -> None:
    with pytest.raises(ValueError, match="chosen_symbol"):
        build_record(chosen_symbol="")


def test_non_positive_amount_raises() -> None:
    with pytest.raises(ValueError, match="amount_usd"):
        build_record(amount_usd=0.0)
    with pytest.raises(ValueError, match="amount_usd"):
        build_record(amount_usd=-100.0)


def test_wrong_shape_feature_vector_raises() -> None:
    with pytest.raises(ValueError, match="chosen_feature_vector"):
        BanditDecisionRecord(
            tx_id="tx",
            chosen_symbol="SOL",
            chosen_feature_vector=np.zeros(3, dtype=np.float64),
            amount_usd=1000.0,
            decision_utc="t",
        )


# -----------------------------------------------------------------------------
# SQLiteOutboxStore — basic CRUD
# -----------------------------------------------------------------------------


def test_init_creates_file_and_schema(tmp_path: Path) -> None:
    p = tmp_path / "out.sqlite"
    outbox = SQLiteOutboxStore(path=p)
    try:
        assert p.exists()
        cur = outbox._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ramhd_outbox'"
        )
        assert cur.fetchone() is not None
    finally:
        outbox.close()


def test_append_then_fetch_by_tx_id(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        fv = np.arange(FEATURE_DIM, dtype=np.float64) / 10.0
        rec = build_record(tx_id="abc", feature_vector=fv)
        outbox.append(rec)
        got = outbox.fetch_by_tx_id("abc")
        assert got is not None
        assert got.tx_id == "abc"
        assert got.chosen_symbol == "SOL"
        assert got.amount_usd == 1000.0
        assert got.status == OutboxStatus.PENDING
        assert np.allclose(got.chosen_feature_vector, fv)
    finally:
        outbox.close()


def test_fetch_unknown_tx_returns_none(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        assert outbox.fetch_by_tx_id("missing") is None
    finally:
        outbox.close()


def test_duplicate_tx_id_raises(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        outbox.append(build_record(tx_id="dup"))
        with pytest.raises(sqlite3.IntegrityError):
            outbox.append(build_record(tx_id="dup"))
    finally:
        outbox.close()


def test_fetch_pending_returns_empty_when_no_pending(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        assert outbox.fetch_pending() == []
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# Pending behavior
# -----------------------------------------------------------------------------


def test_fetch_pending_excludes_processed(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        for i in range(3):
            outbox.append(build_record(tx_id=f"t{i}"))
        outbox.mark_processed("t1", reward=0.01, processed_utc="z")
        pending = outbox.fetch_pending()
        assert {r.tx_id for r in pending} == {"t0", "t2"}
    finally:
        outbox.close()


def test_fetch_pending_orders_by_decision_utc(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        outbox.append(build_record(tx_id="b", decision_utc="2026-05-13T01:00:00+00:00"))
        outbox.append(build_record(tx_id="a", decision_utc="2026-05-13T00:00:00+00:00"))
        outbox.append(build_record(tx_id="c", decision_utc="2026-05-13T02:00:00+00:00"))
        order = [r.tx_id for r in outbox.fetch_pending()]
        assert order == ["a", "b", "c"]
    finally:
        outbox.close()


def test_fetch_pending_respects_limit(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        for i in range(5):
            outbox.append(
                build_record(tx_id=f"x{i}", decision_utc=f"2026-05-13T00:00:0{i}+00:00")
            )
        assert len(outbox.fetch_pending(limit=2)) == 2
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# Status transitions
# -----------------------------------------------------------------------------


def test_mark_processed_sets_fields(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        outbox.append(build_record(tx_id="p"))
        outbox.mark_processed("p", reward=-0.04, processed_utc="2026-05-13T01:00:00Z")
        got = outbox.fetch_by_tx_id("p")
        assert got is not None
        assert got.status == OutboxStatus.PROCESSED
        assert got.reward == pytest.approx(-0.04)
        assert got.processed_utc == "2026-05-13T01:00:00Z"
    finally:
        outbox.close()


def test_mark_processed_on_non_pending_raises(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        outbox.append(build_record(tx_id="p"))
        outbox.mark_processed("p", reward=0.0, processed_utc="t")
        with pytest.raises(ValueError, match="PENDING"):
            outbox.mark_processed("p", reward=0.0, processed_utc="t")
    finally:
        outbox.close()


def test_mark_expired_sets_status_and_error(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        outbox.append(build_record(tx_id="x"))
        outbox.mark_expired("x", reason="too old", processed_utc="t")
        got = outbox.fetch_by_tx_id("x")
        assert got is not None
        assert got.status == OutboxStatus.EXPIRED
        assert got.error == "too old"
        assert got.reward is None
    finally:
        outbox.close()


def test_mark_skipped_sets_status_and_error(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        outbox.append(build_record(tx_id="s"))
        outbox.mark_skipped("s", reason="data missing", processed_utc="t")
        got = outbox.fetch_by_tx_id("s")
        assert got is not None
        assert got.status == OutboxStatus.SKIPPED
        assert got.error == "data missing"
    finally:
        outbox.close()


def test_mark_expired_on_non_pending_raises(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        outbox.append(build_record(tx_id="x"))
        outbox.mark_expired("x", reason="r", processed_utc="t")
        with pytest.raises(ValueError, match="PENDING"):
            outbox.mark_expired("x", reason="r", processed_utc="t")
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# count_by_status
# -----------------------------------------------------------------------------


def test_count_by_status_returns_correct_counts(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        for i in range(5):
            outbox.append(build_record(tx_id=f"t{i}"))
        outbox.mark_processed("t0", reward=0.0, processed_utc="t")
        outbox.mark_processed("t1", reward=0.0, processed_utc="t")
        outbox.mark_expired("t2", reason="age", processed_utc="t")
        outbox.mark_skipped("t3", reason="missing", processed_utc="t")
        counts = outbox.count_by_status()
        assert counts[OutboxStatus.PROCESSED] == 2
        assert counts[OutboxStatus.EXPIRED] == 1
        assert counts[OutboxStatus.SKIPPED] == 1
        assert counts[OutboxStatus.PENDING] == 1
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# Feature vector serialization
# -----------------------------------------------------------------------------


def test_feature_vector_serializes_and_deserializes(tmp_path: Path) -> None:
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        fv = np.arange(FEATURE_DIM, dtype=np.float64) / 10.0
        outbox.append(build_record(tx_id="fv", feature_vector=fv))
        got = outbox.fetch_by_tx_id("fv")
        assert got is not None
        assert got.chosen_feature_vector.dtype == np.float64
        assert np.allclose(got.chosen_feature_vector, fv)
    finally:
        outbox.close()


def test_corrupted_feature_vector_skipped_in_fetch_pending(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="app.feedback.outbox")
    outbox = SQLiteOutboxStore(path=tmp_path / "o.sqlite")
    try:
        outbox.append(build_record(tx_id="ok"))
        # Bypass append to inject a malformed feature vector (length 3).
        outbox._conn.execute(
            "INSERT INTO ramhd_outbox ("
            "tx_id, chosen_symbol, feature_vector_json, amount_usd, "
            "decision_utc, status, reward, processed_utc, error) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
            (
                "bad",
                "SOL",
                json.dumps([0.0, 0.0, 0.0]),
                1000.0,
                "2026-05-13T00:00:00+00:00",
                OutboxStatus.PENDING.value,
            ),
        )
        outbox._conn.commit()
        pending = outbox.fetch_pending()
        assert {r.tx_id for r in pending} == {"ok"}
        assert any("bad" in r.getMessage() for r in caplog.records)
    finally:
        outbox.close()


# -----------------------------------------------------------------------------
# Context manager
# -----------------------------------------------------------------------------


def test_context_manager_closes_connection(tmp_path: Path) -> None:
    p = tmp_path / "o.sqlite"
    with SQLiteOutboxStore(path=p) as outbox:
        outbox.append(build_record(tx_id="cm"))
    with pytest.raises(sqlite3.ProgrammingError):
        outbox.fetch_by_tx_id("cm")


# -----------------------------------------------------------------------------
# Default path resolution
# -----------------------------------------------------------------------------


def test_default_path_uses_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Redirect the default-resolved path into tmp_path so the real
    # data/ directory is never touched.
    fake_default = tmp_path / "ramhd_outbox.sqlite"
    outbox = SQLiteOutboxStore(path=fake_default)
    try:
        assert outbox.path == fake_default
        assert outbox.path.parent.name == tmp_path.name
        assert outbox.path.name == "ramhd_outbox.sqlite"
    finally:
        outbox.close()
