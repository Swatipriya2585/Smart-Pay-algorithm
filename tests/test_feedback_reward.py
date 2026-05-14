"""Tests for compute_reward (FILLED, PARTIAL, FAILED, TIMEOUT, DATA_MISSING)."""

from __future__ import annotations

import logging

import pytest

from app.feedback.contracts import RealizedOutcome, RewardConfig, TradeStatus
from app.feedback.reward import compute_reward


def _outcome(
    *,
    status: TradeStatus,
    realized_return: float,
    realized_cost_dollar: float,
    fill_fraction: float,
    tx_id: str = "tx-test",
) -> RealizedOutcome:
    return RealizedOutcome(
        tx_id=tx_id,
        status=status,
        realized_return=realized_return,
        realized_cost_dollar=realized_cost_dollar,
        fill_fraction=fill_fraction,
        observed_at_utc="2026-05-13T00:00:00+00:00",
    )


# -----------------------------------------------------------------------------
# FILLED branch
# -----------------------------------------------------------------------------


def test_filled_profitable() -> None:
    # return = +0.01, cost = -50, amount = 1000
    # cost_fraction = 50/1000 = 0.05
    # reward = 0.01 - 0.05 = -0.04
    o = _outcome(
        status=TradeStatus.FILLED,
        realized_return=0.01,
        realized_cost_dollar=-50.0,
        fill_fraction=1.0,
    )
    assert compute_reward(o, amount_usd=1000.0) == pytest.approx(-0.04, abs=1e-12)


def test_filled_break_even() -> None:
    # return = +0.05, cost = -50, amount = 1000 -> reward = 0.05 - 0.05 = 0
    o = _outcome(
        status=TradeStatus.FILLED,
        realized_return=0.05,
        realized_cost_dollar=-50.0,
        fill_fraction=1.0,
    )
    assert compute_reward(o, amount_usd=1000.0) == pytest.approx(0.0, abs=1e-12)


def test_filled_winning() -> None:
    # return = +0.10, cost = -50, amount = 1000 -> reward = 0.10 - 0.05 = +0.05
    o = _outcome(
        status=TradeStatus.FILLED,
        realized_return=0.10,
        realized_cost_dollar=-50.0,
        fill_fraction=1.0,
    )
    assert compute_reward(o, amount_usd=1000.0) == pytest.approx(0.05, abs=1e-12)


def test_filled_zero_cost() -> None:
    # return = +0.005, cost = 0, amount = 1000 -> reward = 0.005
    o = _outcome(
        status=TradeStatus.FILLED,
        realized_return=0.005,
        realized_cost_dollar=0.0,
        fill_fraction=1.0,
    )
    assert compute_reward(o, amount_usd=1000.0) == pytest.approx(0.005, abs=1e-12)


# -----------------------------------------------------------------------------
# PARTIAL branch
# -----------------------------------------------------------------------------


def test_partial_above_floor() -> None:
    # fill = 0.5, return = +0.01, cost = -50, amount = 1000
    # earned = 0.5 * 0.01 = 0.005; cost_fraction = 0.05
    # reward = 0.005 - 0.05 = -0.045
    o = _outcome(
        status=TradeStatus.PARTIAL,
        realized_return=0.01,
        realized_cost_dollar=-50.0,
        fill_fraction=0.5,
    )
    assert compute_reward(o, amount_usd=1000.0) == pytest.approx(-0.045, abs=1e-12)


def test_partial_below_floor_treated_as_failure() -> None:
    # fill = 0.01 < default floor 0.05 -> failure branch
    # cost_charged = max(-50, -10) = -10  -> reward = -10/1000 = -0.01
    o = _outcome(
        status=TradeStatus.PARTIAL,
        realized_return=0.0,
        realized_cost_dollar=-50.0,
        fill_fraction=0.01,
    )
    assert compute_reward(o, amount_usd=1000.0) == pytest.approx(-0.01, abs=1e-12)


# -----------------------------------------------------------------------------
# FAILED / TIMEOUT branch
# -----------------------------------------------------------------------------


def test_failed_cost_below_floor_clipped() -> None:
    # cost = -50, floor = -10 -> cost_charged = -10 -> reward = -10/1000 = -0.01
    o = _outcome(
        status=TradeStatus.FAILED,
        realized_return=0.0,
        realized_cost_dollar=-50.0,
        fill_fraction=0.0,
    )
    assert compute_reward(o, amount_usd=1000.0) == pytest.approx(-0.01, abs=1e-12)


def test_failed_cost_above_floor_unclipped() -> None:
    # cost = -3, floor = -10 -> cost_charged = max(-3,-10) = -3 -> -3/1000 = -0.003
    o = _outcome(
        status=TradeStatus.FAILED,
        realized_return=0.0,
        realized_cost_dollar=-3.0,
        fill_fraction=0.0,
    )
    assert compute_reward(o, amount_usd=1000.0) == pytest.approx(-0.003, abs=1e-12)


def test_timeout_uses_same_formula_as_failed() -> None:
    o_fail = _outcome(
        status=TradeStatus.FAILED,
        realized_return=0.0,
        realized_cost_dollar=-50.0,
        fill_fraction=0.0,
    )
    o_to = _outcome(
        status=TradeStatus.TIMEOUT,
        realized_return=0.0,
        realized_cost_dollar=-50.0,
        fill_fraction=0.0,
    )
    a = compute_reward(o_fail, amount_usd=1000.0)
    b = compute_reward(o_to, amount_usd=1000.0)
    assert a is not None and b is not None
    assert a == pytest.approx(b, abs=1e-12)


# -----------------------------------------------------------------------------
# DATA_MISSING
# -----------------------------------------------------------------------------


def test_data_missing_returns_none(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="app.feedback.reward")
    o = _outcome(
        status=TradeStatus.DATA_MISSING,
        realized_return=0.0,
        realized_cost_dollar=0.0,
        fill_fraction=0.0,
    )
    assert compute_reward(o, amount_usd=1000.0) is None
    assert any(
        "DATA_MISSING" in r.getMessage() or "skipping" in r.getMessage().lower()
        for r in caplog.records
    )


# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------


def test_zero_amount_raises() -> None:
    o = _outcome(
        status=TradeStatus.FILLED,
        realized_return=0.0,
        realized_cost_dollar=-1.0,
        fill_fraction=1.0,
    )
    with pytest.raises(ValueError, match="amount_usd"):
        compute_reward(o, amount_usd=0.0)


def test_negative_amount_raises() -> None:
    o = _outcome(
        status=TradeStatus.FILLED,
        realized_return=0.0,
        realized_cost_dollar=-1.0,
        fill_fraction=1.0,
    )
    with pytest.raises(ValueError, match="amount_usd"):
        compute_reward(o, amount_usd=-100.0)
