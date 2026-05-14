"""
Contracts for realized-outcome ingestion and reward configuration (Step 10.1).

A RealizedOutcome captures what actually happened to a trade after the reward
window has elapsed. TradeStatus categorizes the outcome so the reward function
can apply branch-appropriate math; DATA_MISSING tells the caller to SKIP the
bandit update (it is not the same as TIMEOUT — TIMEOUT means the trade
happened but didn't settle in time, DATA_MISSING means we cannot observe).

Sign conventions:
- ``realized_return`` is signed log-return (e.g. ``+0.005`` = +0.5%).
- ``realized_cost_dollar`` is non-positive (consistent with ``CostBreakdown``
  in ``app/cost/base.py``); zero or negative.
- ``fill_fraction`` is in [0, 1]; 1.0 for FILLED, 0.0 for FAILED/TIMEOUT,
  in-between for PARTIAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TradeStatus(str, Enum):
    """Outcome classification for one trade."""

    FILLED = "filled"               # fully executed, settlement observed
    PARTIAL = "partial"             # partially filled (e.g. slippage hit)
    FAILED = "failed"               # rejected pre-settlement
    TIMEOUT = "timeout"             # no settlement within reward window
    DATA_MISSING = "data_missing"   # executed but cannot observe outcome


@dataclass(frozen=True)
class RealizedOutcome:
    """What actually happened to a trade after the reward window."""

    tx_id: str
    status: TradeStatus
    realized_return: float
    realized_cost_dollar: float
    fill_fraction: float
    observed_at_utc: str

    def __post_init__(self) -> None:
        if not self.tx_id:
            raise ValueError("tx_id must be non-empty")
        if self.realized_cost_dollar > 1e-9:
            raise ValueError(
                "realized_cost_dollar must be non-positive (cost is value given up), "
                f"got {self.realized_cost_dollar}"
            )
        if not 0.0 <= self.fill_fraction <= 1.0:
            raise ValueError(
                f"fill_fraction must be in [0, 1], got {self.fill_fraction}"
            )
        if self.status == TradeStatus.FILLED and self.fill_fraction < 0.999:
            raise ValueError("FILLED requires fill_fraction >= 0.999")
        if self.status == TradeStatus.FAILED and self.fill_fraction > 1e-9:
            raise ValueError("FAILED requires fill_fraction == 0")
        if self.status == TradeStatus.TIMEOUT and self.fill_fraction > 1e-9:
            raise ValueError("TIMEOUT requires fill_fraction == 0")


@dataclass(frozen=True)
class RewardConfig:
    """Tuning knobs for the reward function.

    Defaults reflect a capped-pessimistic approach for failures: charge the
    actual cost incurred (with a floor), don't invent extra penalties.
    """

    failure_cost_floor_dollar: float = -10.0
    """For FAILED/TIMEOUT, reward uses
    ``max(realized_cost_dollar, failure_cost_floor_dollar) / amount_usd``.
    The floor prevents a single catastrophic-cost row from dominating the
    bandit's learned weights."""

    partial_fill_floor: float = 0.05
    """For PARTIAL, fill_fraction below this is treated as TIMEOUT for
    reward purposes — at that point the trade is closer to ``didn't happen``
    than to ``happened cheaply``."""

    def __post_init__(self) -> None:
        if self.failure_cost_floor_dollar > 0:
            raise ValueError(
                "failure_cost_floor_dollar must be non-positive, "
                f"got {self.failure_cost_floor_dollar}"
            )
        if not 0.0 <= self.partial_fill_floor <= 1.0:
            raise ValueError(
                f"partial_fill_floor must be in [0, 1], got {self.partial_fill_floor}"
            )
