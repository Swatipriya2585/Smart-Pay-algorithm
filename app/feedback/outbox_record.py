"""One row of the outbox: a decision waiting for its reward."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from app.bandit.contracts import FEATURE_DIM


class OutboxStatus(str, Enum):
    """Lifecycle states of one outbox row."""

    PENDING = "pending"        # awaiting reward
    PROCESSED = "processed"    # reward applied, bandit updated
    EXPIRED = "expired"        # reward never arrived in time
    SKIPPED = "skipped"        # reward computed as None (data_missing)


@dataclass
class BanditDecisionRecord:
    """One row in the outbox.

    Stores enough information to update the bandit later when the reward
    arrives. The ``chosen_feature_vector`` MUST be preserved byte-exactly —
    recomputing it at reward time would be mathematically wrong because
    the underlying market data has moved on.

    Optional fields (reward, processed_utc, error) are NULL-allowed in
    the SQLite schema and start at ``None`` on a freshly appended row.
    """

    tx_id: str
    chosen_symbol: str
    chosen_feature_vector: np.ndarray
    amount_usd: float
    decision_utc: str
    status: OutboxStatus = OutboxStatus.PENDING
    reward: Optional[float] = None
    processed_utc: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.tx_id:
            raise ValueError("tx_id must be non-empty")
        if not self.chosen_symbol:
            raise ValueError("chosen_symbol must be non-empty")
        if self.amount_usd <= 0:
            raise ValueError(f"amount_usd must be positive, got {self.amount_usd}")
        if self.chosen_feature_vector.shape != (FEATURE_DIM,):
            raise ValueError(
                f"chosen_feature_vector must be shape ({FEATURE_DIM},), "
                f"got {self.chosen_feature_vector.shape}"
            )
