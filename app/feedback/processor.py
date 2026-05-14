"""Reward processor: drain pending outbox rows and update the bandit."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.bandit.contracts import LinUCBConfig
from app.bandit.pipeline import record_observation
from app.feedback.contracts import RewardConfig
from app.feedback.outbox import OutboxStore
from app.feedback.outcome_source import OutcomeSource
from app.feedback.reward import compute_reward

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessorStats:
    """Summary of one reward-processor pass."""

    n_pending_at_start: int
    n_processed: int
    n_skipped: int
    n_expired: int
    n_still_pending: int
    n_errors: int
    elapsed_seconds: float


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp, tolerating trailing 'Z'."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def run_reward_processor(
    outbox: OutboxStore,
    outcome_source: OutcomeSource,
    linucb_config: LinUCBConfig,
    reward_config: Optional[RewardConfig] = None,
    state_path: Path | str | None = None,
    max_age_seconds: float = 600.0,
    max_records_per_pass: int = 100,
    now_utc: Optional[datetime] = None,
) -> ProcessorStats:
    """Drain pending outbox rows: fetch outcomes, score, update bandit."""
    start = time.monotonic()
    now = now_utc if now_utc is not None else datetime.now(timezone.utc)
    now_iso = now.isoformat()

    pending = outbox.fetch_pending(limit=max_records_per_pass)
    n_pending_at_start = len(pending)
    logger.info(
        "reward processor starting, %d pending in outbox", n_pending_at_start
    )

    n_processed = 0
    n_skipped = 0
    n_expired = 0
    n_still_pending = 0
    n_errors = 0

    for record in pending:
        try:
            try:
                decision_dt = _parse_iso(record.decision_utc)
            except ValueError as e:
                logger.error(
                    "tx_id=%s has unparseable decision_utc %r: %s",
                    record.tx_id,
                    record.decision_utc,
                    e,
                )
                n_errors += 1
                continue

            age = (now - decision_dt).total_seconds()
            if age > max_age_seconds:
                outbox.mark_expired(
                    record.tx_id,
                    f"exceeded max_age ({age:.1f}s > {max_age_seconds:.1f}s)",
                    now_iso,
                )
                n_expired += 1
                continue

            outcome = outcome_source.fetch_outcome(record.tx_id)
            if outcome is None:
                n_still_pending += 1
                continue

            reward = compute_reward(
                outcome, amount_usd=record.amount_usd, config=reward_config
            )
            if reward is None:
                outbox.mark_skipped(
                    record.tx_id,
                    "compute_reward returned None (DATA_MISSING)",
                    now_iso,
                )
                n_skipped += 1
                continue

            record_observation(
                record.chosen_symbol,
                record.chosen_feature_vector,
                reward,
                linucb_config,
                state_path=state_path,
                now_utc_iso=now_iso,
            )
            outbox.mark_processed(record.tx_id, reward, now_iso)
            n_processed += 1
        except Exception as e:  # noqa: BLE001
            logger.error(
                "error processing tx_id=%s: %s", record.tx_id, e, exc_info=True
            )
            n_errors += 1
            continue

    elapsed = time.monotonic() - start
    logger.info(
        "reward processor finished: processed=%d skipped=%d expired=%d "
        "still_pending=%d errors=%d elapsed=%.3fs",
        n_processed,
        n_skipped,
        n_expired,
        n_still_pending,
        n_errors,
        elapsed,
    )
    return ProcessorStats(
        n_pending_at_start=n_pending_at_start,
        n_processed=n_processed,
        n_skipped=n_skipped,
        n_expired=n_expired,
        n_still_pending=n_still_pending,
        n_errors=n_errors,
        elapsed_seconds=elapsed,
    )
