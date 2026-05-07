"""
Pure Pareto dominance and skyline extraction.

No I/O, logging, or configuration reads—only comparisons using
:data:`~app.pareto.contracts.DIMENSION_DIRECTIONS`.
"""

from __future__ import annotations

from app.pareto.contracts import DIMENSION_DIRECTIONS, CandidateScore

__all__ = ["dominates", "pareto_front"]


def dominates(a: CandidateScore, b: CandidateScore, eps: float) -> bool:
    """
    Returns True iff ``a`` weakly dominates ``b`` on every dimension and strictly
    dominates on at least one, where "better" is defined by ``DIMENSION_DIRECTIONS``.

    Implementation: iterate over ``DIMENSION_DIRECTIONS.items()`` rather than
    hardcoding field names. Adding a new dimension should require editing only the
    dict plus the dataclass.

    For each dimension ``d`` with direction sign ``s``::

        a_d = getattr(a, d) * s
        b_d = getattr(b, d) * s

        ``a`` is "at least as good" iff ``a_d >= b_d - eps``

        ``a`` is "strictly better"  iff ``a_d >  b_d + eps``

    Returns True iff at-least-as-good on all dims AND strictly-better on at least one.
    """
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    if not DIMENSION_DIRECTIONS:
        raise ValueError("DIMENSION_DIRECTIONS must not be empty")

    strictly_better_any = False
    for dim, sign in DIMENSION_DIRECTIONS.items():
        a_d = getattr(a, dim) * sign
        b_d = getattr(b, dim) * sign
        if a_d + eps < b_d:
            return False
        if a_d > b_d + eps:
            strictly_better_any = True

    return strictly_better_any


def pareto_front(candidates: list[CandidateScore], eps: float) -> list[CandidateScore]:
    """
    Returns the non-dominated subset of candidates, preserving input order.

    A candidate ``c`` is on the front iff there is no other candidate ``d`` in the
    input such that ``dominates(d, c, eps)`` is True.

    Complexity: ``O(n²)`` pairwise checks — acceptable for ``n <= 30``.
    """
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    if not candidates:
        return []

    survivors: list[CandidateScore] = []
    for i, c in enumerate(candidates):
        dominated = False
        for j, other in enumerate(candidates):
            if i == j:
                continue
            if dominates(other, c, eps):
                dominated = True
                break
        if not dominated:
            survivors.append(c)
    return survivors
