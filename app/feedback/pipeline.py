"""Decision-time wrapper that runs the bandit and writes the outbox row.

Closes the loop with Step 9's :func:`run_bandit_stage` and Step 10.2's
:class:`OutboxStore`. The observation half is handled by the reward
processor (``run_reward_processor`` in ``app.feedback.processor``).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import LinUCBConfig
from app.bandit.pipeline import BanditDecision, run_bandit_stage
from app.feedback.outbox import OutboxStore
from app.feedback.outbox_record import BanditDecisionRecord
from app.market_data.calibration import Calibration
from app.pareto.contracts import CandidateScore
from app.schemas import RamhdContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordedDecision:
    """The result of :func:`record_decision`.

    Same diagnostics as :class:`BanditDecision`, plus ``tx_id`` (so the
    caller can correlate the decision with the eventual outcome) and a
    flag indicating whether the outbox write succeeded.
    """

    tx_id: str
    decision: BanditDecision
    outbox_write_succeeded: bool


def record_decision(
    tx_id: str,
    context: RamhdContext,
    survivors: list[CandidateScore],
    config: LinUCBConfig,
    calibration: Calibration,
    bandit_calibration: BanditCalibration,
    outbox: OutboxStore,
    state_path: Optional[Path] = None,
    now_utc_iso: Optional[str] = None,
) -> RecordedDecision:
    """Run the bandit decision AND record it to the outbox for later observation.

    A duplicate ``tx_id`` is intentionally non-fatal: the decision was made
    and the user is acting on it, so we surface the audit failure via the
    ``outbox_write_succeeded`` flag rather than raising.

    Raises:
        ValueError: if ``tx_id`` is empty.
        Any error from :func:`run_bandit_stage` (e.g. empty survivors).
        Any non-:class:`sqlite3.IntegrityError` raised by ``outbox.append``.
    """
    if not tx_id:
        raise ValueError("tx_id must be non-empty")

    logger.info(
        "record_decision tx_id=%s chosen among %d survivors",
        tx_id,
        len(survivors),
    )

    decision = run_bandit_stage(
        context=context,
        survivors=survivors,
        config=config,
        calibration=calibration,
        bandit_calibration=bandit_calibration,
        state_path=state_path,
        now_utc_iso=now_utc_iso,
    )

    record = BanditDecisionRecord(
        tx_id=tx_id,
        chosen_symbol=decision.chosen_symbol,
        chosen_feature_vector=decision.chosen_feature_vector,
        amount_usd=float(context.intent.amount_usd),
        decision_utc=decision.decision_utc,
    )

    outbox_write_succeeded = True
    try:
        outbox.append(record)
    except sqlite3.IntegrityError as e:
        logger.error(
            "outbox append failed for tx_id=%s (duplicate/integrity): %s",
            tx_id,
            e,
        )
        outbox_write_succeeded = False
    except Exception as e:
        logger.error("unexpected outbox failure for tx_id=%s: %s", tx_id, e)
        raise

    return RecordedDecision(
        tx_id=tx_id,
        decision=decision,
        outbox_write_succeeded=outbox_write_succeeded,
    )
