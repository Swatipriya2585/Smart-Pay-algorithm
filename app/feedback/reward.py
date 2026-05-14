"""Reward function that converts a RealizedOutcome into a scalar for LinUCB."""

from __future__ import annotations

import logging
from typing import Optional

from app.feedback.contracts import RealizedOutcome, RewardConfig, TradeStatus

logger = logging.getLogger(__name__)


def compute_reward(
    outcome: RealizedOutcome,
    amount_usd: float,
    config: Optional[RewardConfig] = None,
) -> Optional[float]:
    """Convert a RealizedOutcome into a scalar reward for the bandit.

    Returns ``None`` when the outcome must NOT update the bandit
    (``DATA_MISSING``). The caller should skip ``record_observation`` in
    that case. Otherwise returns a unitless float reward.

    Reward formula by status:

    - FILLED:
        ``reward = realized_return - (-realized_cost_dollar / amount_usd)``
        ``-realized_cost_dollar`` is the positive cost-as-fraction.
    - PARTIAL: if fill_fraction < ``config.partial_fill_floor`` treat as
        TIMEOUT; otherwise costs are charged in full but only the filled
        fraction earns return.
    - FAILED / TIMEOUT: ``reward = cost_charged / amount_usd`` where
        ``cost_charged = max(realized_cost_dollar, floor)``.
    - DATA_MISSING: returns ``None``; caller skips the update.

    Raises:
        ValueError: if ``amount_usd <= 0``.
    """
    if amount_usd <= 0:
        raise ValueError(f"amount_usd must be positive, got {amount_usd}")

    cfg = config if config is not None else RewardConfig()
    status = outcome.status

    if status == TradeStatus.DATA_MISSING:
        # No update — caller must skip record_observation.
        logger.warning(
            "outcome for %s has status DATA_MISSING; skipping bandit update",
            outcome.tx_id,
        )
        return None

    if status == TradeStatus.FILLED:
        # FILLED example: realized_return = +0.005, realized_cost_dollar = -50,
        # amount_usd = 1000  ->  cost_fraction = 50/1000 = 0.05
        # reward = 0.005 - 0.05 = -0.045  (lost money on this trade)
        cost_fraction = (-outcome.realized_cost_dollar) / amount_usd
        return outcome.realized_return - cost_fraction

    if status == TradeStatus.PARTIAL:
        if outcome.fill_fraction < cfg.partial_fill_floor:
            # Below the floor: treat as TIMEOUT (fall through to the
            # failure-style branch below).
            cost_charged = max(
                outcome.realized_cost_dollar, cfg.failure_cost_floor_dollar
            )
            return cost_charged / amount_usd
        # PARTIAL example: fill_fraction = 0.5, return = +0.01,
        # realized_cost_dollar = -50, amount_usd = 1000
        # earned = 0.5 * 0.01 = 0.005
        # cost_fraction = 50/1000 = 0.05
        # reward = 0.005 - 0.05 = -0.045
        earned = outcome.fill_fraction * outcome.realized_return
        cost_fraction = (-outcome.realized_cost_dollar) / amount_usd
        return earned - cost_fraction

    if status in (TradeStatus.FAILED, TradeStatus.TIMEOUT):
        # FAILED/TIMEOUT example: realized_cost_dollar = -50, floor = -10,
        # amount_usd = 1000
        # cost_charged = max(-50, -10) = -10   (less-negative wins)
        # reward = -10 / 1000 = -0.01
        cost_charged = max(
            outcome.realized_cost_dollar, cfg.failure_cost_floor_dollar
        )
        return cost_charged / amount_usd

    # Defensive — unreachable when TradeStatus is exhaustive.
    raise ValueError(f"unsupported TradeStatus: {status}")
