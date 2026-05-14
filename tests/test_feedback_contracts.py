"""Tests for feedback contracts (RealizedOutcome, TradeStatus, RewardConfig)."""

from __future__ import annotations

import pytest

from app.feedback.contracts import RealizedOutcome, RewardConfig, TradeStatus


# -----------------------------------------------------------------------------
# RealizedOutcome
# -----------------------------------------------------------------------------


def test_realized_outcome_constructs_with_valid_filled() -> None:
    o = RealizedOutcome(
        tx_id="tx-1",
        status=TradeStatus.FILLED,
        realized_return=0.005,
        realized_cost_dollar=-50.0,
        fill_fraction=1.0,
        observed_at_utc="2026-05-13T00:00:00+00:00",
    )
    assert o.tx_id == "tx-1"
    assert o.status == TradeStatus.FILLED
    assert o.realized_return == 0.005
    assert o.realized_cost_dollar == -50.0
    assert o.fill_fraction == 1.0


def test_empty_tx_id_raises() -> None:
    with pytest.raises(ValueError, match="tx_id"):
        RealizedOutcome(
            tx_id="",
            status=TradeStatus.FILLED,
            realized_return=0.0,
            realized_cost_dollar=-1.0,
            fill_fraction=1.0,
            observed_at_utc="t",
        )


def test_positive_realized_cost_raises() -> None:
    with pytest.raises(ValueError, match="non-positive"):
        RealizedOutcome(
            tx_id="tx",
            status=TradeStatus.FILLED,
            realized_return=0.0,
            realized_cost_dollar=50.0,
            fill_fraction=1.0,
            observed_at_utc="t",
        )


def test_fill_fraction_above_one_raises() -> None:
    with pytest.raises(ValueError, match="fill_fraction"):
        RealizedOutcome(
            tx_id="tx",
            status=TradeStatus.PARTIAL,
            realized_return=0.0,
            realized_cost_dollar=-1.0,
            fill_fraction=1.5,
            observed_at_utc="t",
        )


def test_fill_fraction_negative_raises() -> None:
    with pytest.raises(ValueError, match="fill_fraction"):
        RealizedOutcome(
            tx_id="tx",
            status=TradeStatus.PARTIAL,
            realized_return=0.0,
            realized_cost_dollar=-1.0,
            fill_fraction=-0.1,
            observed_at_utc="t",
        )


def test_filled_with_low_fill_fraction_raises() -> None:
    with pytest.raises(ValueError, match="FILLED"):
        RealizedOutcome(
            tx_id="tx",
            status=TradeStatus.FILLED,
            realized_return=0.0,
            realized_cost_dollar=-1.0,
            fill_fraction=0.5,
            observed_at_utc="t",
        )


def test_failed_with_nonzero_fill_raises() -> None:
    with pytest.raises(ValueError, match="FAILED"):
        RealizedOutcome(
            tx_id="tx",
            status=TradeStatus.FAILED,
            realized_return=0.0,
            realized_cost_dollar=-1.0,
            fill_fraction=0.5,
            observed_at_utc="t",
        )


def test_timeout_with_nonzero_fill_raises() -> None:
    with pytest.raises(ValueError, match="TIMEOUT"):
        RealizedOutcome(
            tx_id="tx",
            status=TradeStatus.TIMEOUT,
            realized_return=0.0,
            realized_cost_dollar=-1.0,
            fill_fraction=0.5,
            observed_at_utc="t",
        )


def test_partial_with_intermediate_fill_constructs() -> None:
    o = RealizedOutcome(
        tx_id="tx",
        status=TradeStatus.PARTIAL,
        realized_return=0.01,
        realized_cost_dollar=-50.0,
        fill_fraction=0.7,
        observed_at_utc="t",
    )
    assert o.fill_fraction == 0.7
    assert o.status == TradeStatus.PARTIAL


# -----------------------------------------------------------------------------
# RewardConfig
# -----------------------------------------------------------------------------


def test_default_config_constructs() -> None:
    cfg = RewardConfig()
    assert cfg.failure_cost_floor_dollar == -10.0
    assert cfg.partial_fill_floor == 0.05


def test_positive_failure_cost_floor_raises() -> None:
    with pytest.raises(ValueError, match="failure_cost_floor_dollar"):
        RewardConfig(failure_cost_floor_dollar=5.0)


def test_partial_fill_floor_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="partial_fill_floor"):
        RewardConfig(partial_fill_floor=1.5)
    with pytest.raises(ValueError, match="partial_fill_floor"):
        RewardConfig(partial_fill_floor=-0.1)
