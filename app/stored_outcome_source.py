"""SQLite-backed OutcomeSource: /observe writes, reward processor reads."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from app.feedback.contracts import RealizedOutcome, TradeStatus

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ramhd_outcomes (
    tx_id                TEXT PRIMARY KEY,
    status               TEXT NOT NULL,
    realized_return      REAL NOT NULL,
    realized_cost_dollar REAL NOT NULL,
    fill_fraction        REAL NOT NULL,
    observed_at_utc      TEXT NOT NULL
)
"""


class StoredOutcomeSource:
    """Push→pull bridge: executor stores outcomes; processor drains via fetch_outcome."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            here = Path(__file__).resolve()
            path = here.parent.parent / "data" / "ramhd_outcomes.sqlite"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    def store(self, outcome: RealizedOutcome) -> None:
        """Insert or replace the outcome for outcome.tx_id."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO ramhd_outcomes (
                tx_id, status, realized_return, realized_cost_dollar,
                fill_fraction, observed_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                outcome.tx_id,
                outcome.status.value,
                outcome.realized_return,
                outcome.realized_cost_dollar,
                outcome.fill_fraction,
                outcome.observed_at_utc,
            ),
        )
        self._conn.commit()
        logger.debug("stored outcome for tx_id=%s status=%s", outcome.tx_id, outcome.status.value)

    def fetch_outcome(self, tx_id: str) -> Optional[RealizedOutcome]:
        """Return the stored outcome for tx_id, or None if absent."""
        row = self._conn.execute(
            "SELECT * FROM ramhd_outcomes WHERE tx_id = ?",
            (tx_id,),
        ).fetchone()
        if row is None:
            return None
        return RealizedOutcome(
            tx_id=row["tx_id"],
            status=TradeStatus(row["status"]),
            realized_return=float(row["realized_return"]),
            realized_cost_dollar=float(row["realized_cost_dollar"]),
            fill_fraction=float(row["fill_fraction"]),
            observed_at_utc=row["observed_at_utc"],
        )

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.DatabaseError as e:
            logger.warning("error closing outcome store connection: %s", e)

    def __enter__(self) -> "StoredOutcomeSource":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
