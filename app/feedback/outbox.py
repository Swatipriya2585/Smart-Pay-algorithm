"""Outbox interface and SQLite-backed implementation for the reward loop."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from app.bandit.contracts import FEATURE_DIM
from app.feedback.outbox_record import BanditDecisionRecord, OutboxStatus

logger = logging.getLogger(__name__)


@runtime_checkable
class OutboxStore(Protocol):
    """Storage interface for the reward outbox.

    v1: :class:`SQLiteOutboxStore`. Future implementations (e.g.
    ``PostgresOutboxStore``) plug in here without changing the processor.

    Single-process safety only — the v1 reward processor is single-process,
    so cross-process locking is intentionally out of scope.
    """

    def append(self, record: BanditDecisionRecord) -> None:
        ...

    def fetch_pending(self, limit: int = 100) -> list[BanditDecisionRecord]:
        ...

    def fetch_by_tx_id(self, tx_id: str) -> Optional[BanditDecisionRecord]:
        ...

    def mark_processed(self, tx_id: str, reward: float, processed_utc: str) -> None:
        ...

    def mark_expired(self, tx_id: str, reason: str, processed_utc: str) -> None:
        ...

    def mark_skipped(self, tx_id: str, reason: str, processed_utc: str) -> None:
        ...

    def count_by_status(self) -> dict[OutboxStatus, int]:
        ...


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ramhd_outbox (
    tx_id                TEXT PRIMARY KEY,
    chosen_symbol        TEXT NOT NULL,
    feature_vector_json  TEXT NOT NULL,
    amount_usd           REAL NOT NULL,
    decision_utc         TEXT NOT NULL,
    status               TEXT NOT NULL,
    reward               REAL,
    processed_utc        TEXT,
    error                TEXT
)
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_outbox_status_decision
    ON ramhd_outbox(status, decision_utc)
"""


class SQLiteOutboxStore:
    """SQLite-backed outbox for bandit decisions awaiting rewards."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            here = Path(__file__).resolve()
            path = here.parent.parent.parent / "data" / "ramhd_outbox.sqlite"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_schema(self) -> None:
        try:
            self._conn.execute(_CREATE_TABLE_SQL)
            self._conn.execute(_CREATE_INDEX_SQL)
            self._conn.commit()
        except sqlite3.DatabaseError as e:
            raise sqlite3.DatabaseError(
                f"failed to create outbox schema at {self.path}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def append(self, record: BanditDecisionRecord) -> None:
        """Insert one record. Raises ``sqlite3.IntegrityError`` if tx_id exists."""
        fv_json = json.dumps(record.chosen_feature_vector.tolist())
        try:
            self._conn.execute(
                "INSERT INTO ramhd_outbox ("
                "tx_id, chosen_symbol, feature_vector_json, amount_usd, "
                "decision_utc, status, reward, processed_utc, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.tx_id,
                    record.chosen_symbol,
                    fv_json,
                    record.amount_usd,
                    record.decision_utc,
                    record.status.value,
                    record.reward,
                    record.processed_utc,
                    record.error,
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            raise
        except sqlite3.DatabaseError as e:
            raise sqlite3.DatabaseError(
                f"failed to insert outbox row {record.tx_id}: {e}"
            ) from e

    def mark_processed(self, tx_id: str, reward: float, processed_utc: str) -> None:
        """PENDING → PROCESSED with reward + timestamp."""
        self._transition(
            tx_id=tx_id,
            new_status=OutboxStatus.PROCESSED,
            reward=reward,
            processed_utc=processed_utc,
            error=None,
        )

    def mark_expired(self, tx_id: str, reason: str, processed_utc: str) -> None:
        """PENDING → EXPIRED with the reason stored in ``error``."""
        self._transition(
            tx_id=tx_id,
            new_status=OutboxStatus.EXPIRED,
            reward=None,
            processed_utc=processed_utc,
            error=reason,
        )

    def mark_skipped(self, tx_id: str, reason: str, processed_utc: str) -> None:
        """PENDING → SKIPPED with the reason stored in ``error``."""
        self._transition(
            tx_id=tx_id,
            new_status=OutboxStatus.SKIPPED,
            reward=None,
            processed_utc=processed_utc,
            error=reason,
        )

    def _transition(
        self,
        *,
        tx_id: str,
        new_status: OutboxStatus,
        reward: Optional[float],
        processed_utc: str,
        error: Optional[str],
    ) -> None:
        # Only PENDING rows may transition (idempotency guard).
        cur = self._conn.execute(
            "UPDATE ramhd_outbox "
            "SET status = ?, reward = ?, processed_utc = ?, error = ? "
            "WHERE tx_id = ? AND status = ?",
            (
                new_status.value,
                reward,
                processed_utc,
                error,
                tx_id,
                OutboxStatus.PENDING.value,
            ),
        )
        if cur.rowcount == 0:
            existing = self.fetch_by_tx_id(tx_id)
            if existing is None:
                raise ValueError(f"outbox has no record for tx_id {tx_id!r}")
            raise ValueError(
                f"cannot transition tx_id {tx_id!r} from {existing.status.value} "
                f"to {new_status.value}; only PENDING rows are updatable"
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def fetch_pending(self, limit: int = 100) -> list[BanditDecisionRecord]:
        """Return PENDING records ordered by ``decision_utc`` ASC."""
        cur = self._conn.execute(
            "SELECT tx_id, chosen_symbol, feature_vector_json, amount_usd, "
            "decision_utc, status, reward, processed_utc, error "
            "FROM ramhd_outbox WHERE status = ? "
            "ORDER BY decision_utc ASC LIMIT ?",
            (OutboxStatus.PENDING.value, int(limit)),
        )
        results: list[BanditDecisionRecord] = []
        for row in cur.fetchall():
            try:
                rec = self._row_to_record(row)
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    "skipping corrupted outbox row tx_id=%s: %s", row[0], e
                )
                continue
            results.append(rec)
        return results

    def fetch_by_tx_id(self, tx_id: str) -> Optional[BanditDecisionRecord]:
        cur = self._conn.execute(
            "SELECT tx_id, chosen_symbol, feature_vector_json, amount_usd, "
            "decision_utc, status, reward, processed_utc, error "
            "FROM ramhd_outbox WHERE tx_id = ?",
            (tx_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def count_by_status(self) -> dict[OutboxStatus, int]:
        cur = self._conn.execute(
            "SELECT status, COUNT(*) FROM ramhd_outbox GROUP BY status"
        )
        counts: dict[OutboxStatus, int] = {s: 0 for s in OutboxStatus}
        for status_str, n in cur.fetchall():
            try:
                counts[OutboxStatus(status_str)] = int(n)
            except ValueError:
                logger.warning("unknown outbox status in DB: %s", status_str)
        return counts

    # ------------------------------------------------------------------
    # Helpers / lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: tuple) -> BanditDecisionRecord:
        (
            tx_id,
            chosen_symbol,
            fv_json,
            amount_usd,
            decision_utc,
            status_str,
            reward,
            processed_utc,
            error,
        ) = row
        fv_list = json.loads(fv_json)
        fv = np.asarray(fv_list, dtype=np.float64)
        if fv.shape != (FEATURE_DIM,):
            raise ValueError(
                f"feature_vector has wrong shape {fv.shape}, expected ({FEATURE_DIM},)"
            )
        return BanditDecisionRecord(
            tx_id=tx_id,
            chosen_symbol=chosen_symbol,
            chosen_feature_vector=fv,
            amount_usd=float(amount_usd),
            decision_utc=decision_utc,
            status=OutboxStatus(status_str),
            reward=None if reward is None else float(reward),
            processed_utc=processed_utc,
            error=error,
        )

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.DatabaseError as e:
            logger.warning("error closing outbox connection: %s", e)

    def __enter__(self) -> "SQLiteOutboxStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
