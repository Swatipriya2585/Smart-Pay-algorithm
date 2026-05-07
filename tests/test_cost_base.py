"""Verify the cost data contracts behave correctly on hand-built inputs."""

import pytest

from app.cost.base import CostBreakdown, MultiHorizonCostEstimate


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_breakdown(
    h: float = 30.0,
    slippage: float = -2.0,
    gas: float = -0.001,
    settlement: float = -1.0,
) -> CostBreakdown:
    total = slippage + gas + settlement
    return CostBreakdown(
        horizon_seconds=h,
        slippage_dollar=slippage,
        gas_dollar=gas,
        settlement_risk_dollar=settlement,
        total_cost_dollar=total,
        settlement_seconds=0.8,
    )


# -----------------------------------------------------------------------------
# CostBreakdown validation
# -----------------------------------------------------------------------------


def test_cost_breakdown_constructs_with_valid_inputs() -> None:
    b = _make_breakdown()
    assert b.horizon_seconds == 30.0
    assert b.total_cost_dollar < 0


def test_cost_breakdown_rejects_zero_horizon() -> None:
    with pytest.raises(ValueError, match="horizon_seconds"):
        CostBreakdown(
            horizon_seconds=0.0,
            slippage_dollar=-1.0,
            gas_dollar=-0.001,
            settlement_risk_dollar=-0.5,
            total_cost_dollar=-1.501,
            settlement_seconds=0.8,
        )


def test_cost_breakdown_rejects_zero_settlement_seconds() -> None:
    with pytest.raises(ValueError, match="settlement_seconds"):
        CostBreakdown(
            horizon_seconds=30.0,
            slippage_dollar=-1.0,
            gas_dollar=-0.001,
            settlement_risk_dollar=-0.5,
            total_cost_dollar=-1.501,
            settlement_seconds=0.0,
        )


def test_cost_breakdown_rejects_positive_slippage() -> None:
    """Costs are non-positive; positive slippage means value gained, which is impossible."""
    with pytest.raises(ValueError, match="slippage_dollar"):
        CostBreakdown(
            horizon_seconds=30.0,
            slippage_dollar=2.0,  # positive — invalid
            gas_dollar=-0.001,
            settlement_risk_dollar=-0.5,
            total_cost_dollar=1.499,
            settlement_seconds=0.8,
        )


def test_cost_breakdown_rejects_positive_gas() -> None:
    with pytest.raises(ValueError, match="gas_dollar"):
        CostBreakdown(
            horizon_seconds=30.0,
            slippage_dollar=-1.0,
            gas_dollar=0.001,  # positive — invalid
            settlement_risk_dollar=-0.5,
            total_cost_dollar=-1.499,
            settlement_seconds=0.8,
        )


def test_cost_breakdown_rejects_positive_settlement_risk() -> None:
    with pytest.raises(ValueError, match="settlement_risk_dollar"):
        CostBreakdown(
            horizon_seconds=30.0,
            slippage_dollar=-1.0,
            gas_dollar=-0.001,
            settlement_risk_dollar=0.5,  # positive — invalid
            total_cost_dollar=-0.501,
            settlement_seconds=0.8,
        )


def test_cost_breakdown_rejects_total_not_matching_components() -> None:
    """The aggregate must equal the sum of the three components."""
    with pytest.raises(ValueError, match="total_cost_dollar"):
        CostBreakdown(
            horizon_seconds=30.0,
            slippage_dollar=-2.0,
            gas_dollar=-0.001,
            settlement_risk_dollar=-1.0,
            total_cost_dollar=-5.0,  # doesn't match -3.001
            settlement_seconds=0.8,
        )


def test_cost_breakdown_allows_all_zero() -> None:
    """A perfectly free transaction (synthetic edge case) should be valid."""
    b = CostBreakdown(
        horizon_seconds=30.0,
        slippage_dollar=0.0,
        gas_dollar=0.0,
        settlement_risk_dollar=0.0,
        total_cost_dollar=0.0,
        settlement_seconds=0.4,
    )
    assert b.total_cost_dollar == 0.0


def test_cost_breakdown_allows_zero_components_with_negative_others() -> None:
    """Some components can be zero (e.g., zero gas on free L2) as long as total matches."""
    b = CostBreakdown(
        horizon_seconds=30.0,
        slippage_dollar=-2.0,
        gas_dollar=0.0,
        settlement_risk_dollar=-1.0,
        total_cost_dollar=-3.0,
        settlement_seconds=0.8,
    )
    assert b.gas_dollar == 0.0


# -----------------------------------------------------------------------------
# MultiHorizonCostEstimate construction and lookup
# -----------------------------------------------------------------------------


def test_multihorizon_constructs_with_valid_inputs() -> None:
    breakdowns = {h: _make_breakdown(h=h) for h in (5.0, 30.0, 120.0)}
    mhce = MultiHorizonCostEstimate(
        symbol="SOL",
        position_value_usd=1000.0,
        breakdowns=breakdowns,
    )
    assert mhce.symbol == "SOL"
    assert len(mhce.breakdowns) == 3


def test_multihorizon_at_returns_correct_horizon() -> None:
    breakdowns = {
        5.0: _make_breakdown(h=5.0, slippage=-1.0),
        30.0: _make_breakdown(h=30.0, slippage=-2.0),
        120.0: _make_breakdown(h=120.0, slippage=-3.0),
    }
    mhce = MultiHorizonCostEstimate(
        symbol="SOL", position_value_usd=1000.0, breakdowns=breakdowns
    )
    assert mhce.at(30.0).slippage_dollar == -2.0


def test_multihorizon_at_unknown_horizon_raises() -> None:
    breakdowns = {30.0: _make_breakdown(h=30.0)}
    mhce = MultiHorizonCostEstimate(
        symbol="SOL", position_value_usd=1000.0, breakdowns=breakdowns
    )
    with pytest.raises(KeyError, match="not in this estimate"):
        mhce.at(60.0)


def test_multihorizon_rejects_empty_breakdowns() -> None:
    with pytest.raises(ValueError, match="at least one horizon"):
        MultiHorizonCostEstimate(
            symbol="SOL", position_value_usd=1000.0, breakdowns={}
        )


def test_multihorizon_rejects_empty_symbol() -> None:
    breakdowns = {30.0: _make_breakdown(h=30.0)}
    with pytest.raises(ValueError, match="non-empty string"):
        MultiHorizonCostEstimate(
            symbol="", position_value_usd=1000.0, breakdowns=breakdowns
        )


def test_multihorizon_rejects_negative_position_value() -> None:
    breakdowns = {30.0: _make_breakdown(h=30.0)}
    with pytest.raises(ValueError, match="position_value_usd"):
        MultiHorizonCostEstimate(
            symbol="SOL", position_value_usd=-100.0, breakdowns=breakdowns
        )


def test_multihorizon_rejects_horizon_key_mismatch() -> None:
    breakdowns = {30.0: _make_breakdown(h=60.0)}  # key 30, breakdown says 60
    with pytest.raises(ValueError, match="does not match"):
        MultiHorizonCostEstimate(
            symbol="SOL", position_value_usd=1000.0, breakdowns=breakdowns
        )


def test_horizon_seconds_list_is_sorted() -> None:
    breakdowns = {
        120.0: _make_breakdown(h=120.0),
        5.0: _make_breakdown(h=5.0),
        30.0: _make_breakdown(h=30.0),
    }
    mhce = MultiHorizonCostEstimate(
        symbol="SOL", position_value_usd=1000.0, breakdowns=breakdowns
    )
    assert mhce.horizon_seconds_list() == [5.0, 30.0, 120.0]


def test_worst_total_cost_dollar_returns_most_negative() -> None:
    """The Pareto filter uses this — must return the most-negative total cost."""
    breakdowns = {
        5.0: _make_breakdown(h=5.0, slippage=-0.5, gas=-0.001, settlement=-0.5),  # -1.001
        30.0: _make_breakdown(h=30.0, slippage=-2.0, gas=-0.001, settlement=-1.0),  # -3.001
        120.0: _make_breakdown(h=120.0, slippage=-2.0, gas=-0.001, settlement=-3.0),  # -5.001
    }
    mhce = MultiHorizonCostEstimate(
        symbol="SOL", position_value_usd=1000.0, breakdowns=breakdowns
    )
    assert abs(mhce.worst_total_cost_dollar() - (-5.001)) < 1e-9


def test_worst_total_cost_dollar_zero_for_free_transaction() -> None:
    """If all costs are zero, worst is zero, not None."""
    breakdowns = {
        h: _make_breakdown(h=h, slippage=0.0, gas=0.0, settlement=0.0)
        for h in (5.0, 30.0, 120.0)
    }
    mhce = MultiHorizonCostEstimate(
        symbol="USDC", position_value_usd=1000.0, breakdowns=breakdowns
    )
    assert mhce.worst_total_cost_dollar() == 0.0
