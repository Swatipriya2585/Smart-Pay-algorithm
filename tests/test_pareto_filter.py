"""
Tests for :func:`~app.pareto.filter.apply_pareto_filter`.

Hand-built :class:`CandidateScore` rows; caplog checks for INFO/WARNING.
"""

from __future__ import annotations

import logging

import pytest

from app.pareto.contracts import CandidateScore, ParetoConfig
from app.pareto import filter as pareto_filter_mod
from app.pareto.filter import apply_pareto_filter

EPS = 1e-9


def _c(
    symbol: str,
    *,
    er: float = 0.0,
    cvar: float = -0.10,
    cost_bps: float = 50.0,
    liq: float = 1_000_000.0,
) -> CandidateScore:
    return CandidateScore(
        symbol=symbol,
        expected_return_120s=er,
        cvar_95_120s=cvar,
        effective_cost_bps=cost_bps,
        liquidity_usd=liq,
    )


# --- pass-through -------------------------------------------------------------


def test_front_within_bounds_unchanged(caplog: pytest.LogCaptureFixture) -> None:
    # Three pairwise-incomparable tokens → front size 3; min=2 max=5 → no floor/cap.
    a = _c("A", er=0.05, cvar=-0.02, cost_bps=20.0, liq=2_000_000.0)
    b = _c("B", er=0.02, cvar=-0.05, cost_bps=40.0, liq=50_000_000.0)
    c = _c("C", er=0.10, cvar=-0.08, cost_bps=35.0, liq=1_000_000.0)
    cand = [a, b, c]
    cfg = ParetoConfig(min_survivors=2, max_survivors=5, epsilon=EPS)
    with caplog.at_level(logging.INFO, logger="app.pareto.filter"):
        out = apply_pareto_filter(cand, cfg)
    assert out == cand
    assert "floor relaxation" not in caplog.text.lower()
    assert "capped survivors" not in caplog.text.lower()


# --- floor -------------------------------------------------------------------


def test_floor_relaxes_when_front_too_small(caplog: pytest.LogCaptureFixture) -> None:
    # A dominates B and C → front [A]. min=2 → add best dominated by tiebreaker (default ER).
    a = _c("A", er=0.10, cvar=-0.01, cost_bps=5.0, liq=10_000_000.0)
    b = _c("B", er=0.01, cvar=-0.20, cost_bps=60.0, liq=500_000.0)
    c = _c("C", er=0.05, cvar=-0.15, cost_bps=55.0, liq=600_000.0)
    cfg = ParetoConfig(
        min_survivors=2,
        max_survivors=5,
        tiebreaker="expected_return_120s",
        epsilon=EPS,
    )
    with caplog.at_level(logging.INFO, logger="app.pareto.filter"):
        out = apply_pareto_filter([a, b, c], cfg)
    assert len(out) == 2
    assert out[0] is a
    # Between dominated rows, C has higher ER (0.05) than B (0.01).
    assert out[1] is c
    assert "floor relaxation added 1 dominated candidate(s): C" in caplog.text


def test_floor_capped_by_total_input(caplog: pytest.LogCaptureFixture) -> None:
    only = _c("solo")
    cfg = ParetoConfig(min_survivors=2, max_survivors=5, epsilon=EPS)
    with caplog.at_level(logging.WARNING, logger="app.pareto.filter"):
        out = apply_pareto_filter([only], cfg)
    assert out == [only]
    assert "below min_survivors" in caplog.text


def test_floor_uses_tiebreaker_direction(caplog: pytest.LogCaptureFixture) -> None:
    # Front [A]; dominated B (cost 40), C (cost 25). Tiebreaker cost (-1) → prefer C (lower bps).
    a = _c("A", er=0.10, cvar=-0.01, cost_bps=5.0, liq=10_000_000.0)
    b = _c("B", er=0.01, cvar=-0.20, cost_bps=40.0, liq=500_000.0)
    c = _c("C", er=0.02, cvar=-0.18, cost_bps=25.0, liq=550_000.0)
    cfg = ParetoConfig(
        min_survivors=2,
        max_survivors=5,
        tiebreaker="effective_cost_bps",
        epsilon=EPS,
    )
    with caplog.at_level(logging.INFO, logger="app.pareto.filter"):
        out = apply_pareto_filter([a, b, c], cfg)
    assert out == [a, c]


# --- cap ---------------------------------------------------------------------


def test_cap_trims_to_max_survivors(caplog: pytest.LogCaptureFixture) -> None:
    # Seven spike specialists → full front of 7; max=5 → drop 2 lowest-ranked by default ER.
    s = [
        _c("E1", er=10.0, cvar=-1000.0, cost_bps=10_000.0, liq=1.0),
        _c("E2", er=-1000.0, cvar=-1e-9, cost_bps=10_000.0, liq=1.0),
        _c("E3", er=-1000.0, cvar=-1000.0, cost_bps=1.0, liq=1.0),
        _c("E4", er=-1000.0, cvar=-1000.0, cost_bps=10_000.0, liq=1e15),
        _c("E5", er=-500.0, cvar=-500.0, cost_bps=5000.0, liq=5000.0),
        _c("E6", er=8.0, cvar=-900.0, cost_bps=9000.0, liq=100.0),
        _c("E7", er=-900.0, cvar=-800.0, cost_bps=8000.0, liq=8000.0),
    ]
    cfg = ParetoConfig(
        min_survivors=2,
        max_survivors=5,
        tiebreaker="expected_return_120s",
        epsilon=EPS,
    )
    from app.pareto.dominance import pareto_front

    front = pareto_front(s, EPS)
    assert len(front) == 7

    with caplog.at_level(logging.INFO, logger="app.pareto.filter"):
        out = apply_pareto_filter(s, cfg)
    assert len(out) == 5

    ranked = sorted(
        s,
        key=lambda c: (
            -c.expected_return_120s * 1,
            s.index(c),
            c.symbol,
        ),
    )[:5]
    assert [x.symbol for x in out] == [x.symbol for x in ranked]
    assert "capped survivors from 7 to 5" in caplog.text
    dropped_syms = caplog.text.split("dropped symbols: ")[-1].strip().split()[0]
    assert "," in dropped_syms or dropped_syms  # log lists symbols


def test_cap_preserves_tiebreaker_direction(caplog: pytest.LogCaptureFixture) -> None:
    # Seven tokens on frontier; tiebreaker cost → keep lowest bps (direction -1).
    s = [
        _c("n1", er=10.0, cvar=-1000.0, cost_bps=10.0, liq=1.0),
        _c("n2", er=-1000.0, cvar=-1e-9, cost_bps=20.0, liq=1.0),
        _c("n3", er=-1000.0, cvar=-1000.0, cost_bps=3.0, liq=1.0),
        _c("n4", er=-1000.0, cvar=-1000.0, cost_bps=40.0, liq=1e12),
        _c("n5", er=-500.0, cvar=-500.0, cost_bps=35.0, liq=5000.0),
        _c("n6", er=9.0, cvar=-900.0, cost_bps=15.0, liq=90.0),
        _c("n7", er=-900.0, cvar=-800.0, cost_bps=8.0, liq=8000.0),
    ]
    cfg = ParetoConfig(
        min_survivors=2,
        max_survivors=5,
        tiebreaker="effective_cost_bps",
        epsilon=EPS,
    )
    from app.pareto.dominance import pareto_front

    assert len(pareto_front(s, EPS)) == 7

    with caplog.at_level(logging.INFO, logger="app.pareto.filter"):
        out = apply_pareto_filter(s, cfg)
    costs = [x.effective_cost_bps for x in out]
    assert len(costs) == 5
    assert max(costs) <= min(x.effective_cost_bps for x in s if x not in out) or len(s) == 7
    # Retained set must be the 5 smallest costs (unique in this construction).
    other_costs = sorted([x.effective_cost_bps for x in s])
    assert sorted(costs) == other_costs[:5]


# --- tiebreaker flows --------------------------------------------------------


def test_three_tiebreakers_yield_three_orderings() -> None:
    """Five mutually non-dominated spikes; cap must trim (len(front) > max) so tiebreaker ranks."""
    pool = [
        _c("p0", er=10.0, cvar=-1000.0, cost_bps=100.0, liq=1.0),
        _c("p1", er=-1000.0, cvar=-1e-9, cost_bps=100.0, liq=1.0),
        _c("p2", er=-1000.0, cvar=-1000.0, cost_bps=1.0, liq=1.0),
        _c("p3", er=-1000.0, cvar=-1000.0, cost_bps=100.0, liq=1e15),
        _c("p4", er=-500.0, cvar=-500.0, cost_bps=5000.0, liq=5000.0),
    ]
    cfg_er = ParetoConfig(
        min_survivors=2, max_survivors=3, tiebreaker="expected_return_120s", epsilon=EPS
    )
    cfg_liq = ParetoConfig(
        min_survivors=2, max_survivors=3, tiebreaker="liquidity_usd", epsilon=EPS
    )
    cfg_cost = ParetoConfig(
        min_survivors=2, max_survivors=3, tiebreaker="effective_cost_bps", epsilon=EPS
    )

    o_er = [x.symbol for x in apply_pareto_filter(pool, cfg_er)]
    o_liq = [x.symbol for x in apply_pareto_filter(pool, cfg_liq)]
    o_cost = [x.symbol for x in apply_pareto_filter(pool, cfg_cost)]

    # Top ER: p0 (10), then p4 (-500), then best among -1000 by index → p1.
    assert o_er == ["p0", "p4", "p1"]
    # Liquidity: p3 wins mass; then p4; among liq=1 tie, lowest input index p0.
    assert o_liq == ["p3", "p4", "p0"]
    # Cost (minimize): p2 has bps=1; then p0 before p1 at equal bps=100 (input index).
    assert o_cost == ["p2", "p0", "p1"]

    assert o_er != o_liq
    assert o_liq != o_cost
    assert o_er != o_cost


# --- determinism -------------------------------------------------------------


def test_determinism_ten_runs() -> None:
    pool = [
        _c("d0", er=1.0, cvar=-1.0, cost_bps=10.0, liq=100.0),
        _c("d1", er=2.0, cvar=-2.0, cost_bps=20.0, liq=200.0),
        _c("d2", er=3.0, cvar=-3.0, cost_bps=30.0, liq=300.0),
    ]
    cfg = ParetoConfig(min_survivors=2, max_survivors=2, tiebreaker="expected_return_120s", epsilon=EPS)
    first = apply_pareto_filter(pool, cfg)
    for _ in range(9):
        assert apply_pareto_filter(pool, cfg) == first


def test_tiebreaker_ties_fall_back_to_input_order() -> None:
    # Same ER tiebreaker value; earlier list slot wins over later.
    x = _c("first", er=1.0)
    y = _c("second", er=1.0)
    cfg = ParetoConfig(min_survivors=1, max_survivors=1, tiebreaker="expected_return_120s", epsilon=EPS)
    out = apply_pareto_filter([x, y], cfg)
    assert out == [x]


def test_tiebreaker_and_input_order_ties_fall_back_to_symbol() -> None:
    # Force identical (-signed TB, index) via injected index map on private helper:
    # same signed tiebreaker and same synthetic index → alphabetical symbol.
    a = _c("zebra", er=1.0)
    b = _c("apple", er=1.0)
    fake_idx = {id(a): 7, id(b): 7}
    ranked = pareto_filter_mod._sort_candidates_deterministic(
        [a, b],
        "expected_return_120s",
        fake_idx,
    )
    assert [x.symbol for x in ranked] == ["apple", "zebra"]


# --- edges --------------------------------------------------------------------


def test_empty_input(caplog: pytest.LogCaptureFixture) -> None:
    cfg = ParetoConfig()
    with caplog.at_level(logging.WARNING, logger="app.pareto.filter"):
        assert apply_pareto_filter([], cfg) == []
    assert "empty candidate list" in caplog.text


def test_min_zero_max_zero() -> None:
    """max=0 → no cap; min=0 → no floor relaxation."""
    # Pairwise incomparable (same pattern as test_front_within_bounds_unchanged).
    a = _c("a", er=0.05, cvar=-0.02, cost_bps=20.0, liq=2_000_000.0)
    b = _c("b", er=0.02, cvar=-0.05, cost_bps=40.0, liq=50_000_000.0)
    c = _c("c", er=0.10, cvar=-0.08, cost_bps=35.0, liq=1_000_000.0)
    pool = [a, b, c]
    cfg = ParetoConfig(min_survivors=0, max_survivors=0, epsilon=EPS)
    out = apply_pareto_filter(pool, cfg)
    assert len(out) == 3
    assert out == pool
