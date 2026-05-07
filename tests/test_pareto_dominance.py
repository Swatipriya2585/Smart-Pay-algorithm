"""
Unit tests for Pareto dominance (hand-built :class:`CandidateScore` rows).

Arrange / Act / Assert, with expected values noted in comments.
"""

from __future__ import annotations

from app.pareto.contracts import CandidateScore
from app.pareto.dominance import dominates, pareto_front

EPS = 1e-9  # same order of magnitude as ParetoConfig.epsilon default


def _c(
    symbol: str = "T",
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


# --- dominates: strict cases -------------------------------------------------


def test_strict_dominance_all_four_dims() -> None:
    # A: higher return, less-negative CVaR, lower cost, higher liq.
    a = _c(symbol="A", er=0.02, cvar=-0.02, cost_bps=10.0, liq=2_000_000.0)
    b = _c(symbol="B", er=0.01, cvar=-0.05, cost_bps=40.0, liq=1_000_000.0)
    assert dominates(a, b, EPS) is True
    assert dominates(b, a, EPS) is False


def test_equality_no_dominance() -> None:
    a = _c(symbol="A")
    b = _c(symbol="B")
    assert dominates(a, b, EPS) is False
    assert dominates(b, a, EPS) is False


def test_mixed_no_dominance() -> None:
    # A wins expected_return + effective_cost; B wins CVaR + liquidity → incomparable.
    a = _c(er=0.05, cvar=-0.20, cost_bps=30.0, liq=500_000.0)
    b = _c(er=0.02, cvar=-0.05, cost_bps=60.0, liq=2_000_000.0)
    assert dominates(a, b, EPS) is False
    assert dominates(b, a, EPS) is False


def test_single_dim_strictly_better() -> None:
    # Diff only on expected_return_120s (others default-equal).
    a = _c(er=0.03)
    b = _c(er=0.01)
    assert dominates(a, b, EPS) is True
    assert dominates(b, a, EPS) is False


# --- dominates: sign conventions ---------------------------------------------


def test_cvar_sign_convention() -> None:
    # Less-negative CVaR is better; this test catches the most common bug in this module.
    a = _c(cvar=-0.05)
    b = _c(cvar=-0.10)
    assert dominates(a, b, EPS) is True
    assert dominates(b, a, EPS) is False


def test_cost_sign_convention() -> None:
    # Lower effective_cost_bps is better (direction -1).
    a = _c(cost_bps=30.0)
    b = _c(cost_bps=50.0)
    assert dominates(a, b, EPS) is True
    assert dominates(b, a, EPS) is False


# --- dominates: epsilon -------------------------------------------------------


def test_epsilon_treats_tiny_diff_as_equal() -> None:
    # Delta = 0.5 * EPS on one dimension → no strict improvement → neither dominates.
    base_er = 0.01
    delta = 0.5 * EPS
    a = _c(er=base_er + delta)
    b = _c(er=base_er)
    assert dominates(a, b, EPS) is False
    assert dominates(b, a, EPS) is False


def test_epsilon_does_not_swallow_real_diff() -> None:
    # Delta = 100 * EPS on expected_return → strict improvement → A dominates B.
    base_er = 0.01
    delta = 100.0 * EPS
    a = _c(er=base_er + delta)
    b = _c(er=base_er)
    assert dominates(a, b, EPS) is True
    assert dominates(b, a, EPS) is False


# --- pareto_front -----------------------------------------------------------


def test_empty_input_returns_empty() -> None:
    assert pareto_front([], EPS) == []


def test_single_candidate() -> None:
    only = _c(symbol="solo")
    assert pareto_front([only], EPS) == [only]


def test_two_identical_both_survive() -> None:
    a = _c(symbol="A")
    b = _c(symbol="B")
    assert pareto_front([a, b], EPS) == [a, b]


def test_three_with_one_dominated() -> None:
    # A dominates B; C incomparable with both → frontier {A, C}. Input order [B, C, A] → [C, A].
    a = _c(symbol="A", er=0.05, cvar=-0.02, cost_bps=20.0, liq=2_000_000.0)
    b = _c(symbol="B", er=0.01, cvar=-0.20, cost_bps=60.0, liq=500_000.0)
    c = _c(symbol="C", er=0.10, cvar=-0.08, cost_bps=35.0, liq=1_000_000.0)
    assert dominates(a, b, EPS) is True
    assert dominates(a, c, EPS) is False and dominates(c, a, EPS) is False
    assert pareto_front([b, c, a], EPS) == [c, a]


def test_five_all_on_frontier() -> None:
    # Five spikes — each token wins on a different axis relative to the others; pairwise incomparable.
    c0 = _c(symbol="0", er=10.0, cvar=-1000.0, cost_bps=10_000.0, liq=1.0)
    c1 = _c(symbol="1", er=-1000.0, cvar=-1e-9, cost_bps=10_000.0, liq=1.0)
    c2 = _c(symbol="2", er=-1000.0, cvar=-1000.0, cost_bps=1.0, liq=1.0)
    c3 = _c(symbol="3", er=-1000.0, cvar=-1000.0, cost_bps=10_000.0, liq=1e15)
    c4 = _c(symbol="4", er=-500.0, cvar=-500.0, cost_bps=5000.0, liq=5000.0)
    cand = [c0, c1, c2, c3, c4]
    assert pareto_front(cand, EPS) == cand


def test_five_where_one_dominates_four() -> None:
    winner = _c(symbol="A", er=0.10, cvar=-0.001, cost_bps=1.0, liq=50_000_000.0)
    losers = [
        _c(symbol=f"L{i}", er=0.01, cvar=-0.10, cost_bps=50.0, liq=1_000_000.0)
        for i in range(4)
    ]
    cand = [losers[0], losers[1], winner, losers[2], losers[3]]
    assert pareto_front(cand, EPS) == [winner]


def test_input_order_preserved() -> None:
    # Frontier {E, C, A}; B dominated by A; D dominated by C. Input [E, C, A, B, D] → [E, C, A].
    e = _c(symbol="E", er=0.02, cvar=-0.05, cost_bps=40.0, liq=50_000_000.0)
    c = _c(symbol="C", er=0.10, cvar=-0.08, cost_bps=35.0, liq=1_000_000.0)
    a = _c(symbol="A", er=0.05, cvar=-0.02, cost_bps=20.0, liq=2_000_000.0)
    b = _c(symbol="B", er=0.01, cvar=-0.20, cost_bps=60.0, liq=500_000.0)
    d = _c(symbol="D", er=0.03, cvar=-0.15, cost_bps=55.0, liq=600_000.0)

    assert dominates(a, b, EPS) is True
    assert dominates(c, d, EPS) is True
    assert dominates(a, c, EPS) is False and dominates(c, a, EPS) is False
    assert dominates(a, e, EPS) is False and dominates(e, a, EPS) is False
    assert dominates(c, e, EPS) is False and dominates(e, c, EPS) is False

    ordered = [e, c, a, b, d]
    assert pareto_front(ordered, EPS) == [e, c, a]
