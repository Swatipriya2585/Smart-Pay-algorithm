"""
Contracts for Pareto frontier filtering (RAMHD Step 8).

Upstream stages produce forecasts, CVaR, dollar costs, regime, and routing biases.
The router applies hard exclusions and ``score_bias_bps``; the assembler (later)
folds bias into ``effective_cost_bps`` before candidates reach this layer—there is
no separate regime-fit scalar as a Pareto dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CandidateScore",
    "ParetoConfig",
    "DIMENSION_DIRECTIONS",
]


@dataclass(frozen=True)
class CandidateScore:
    """Four comparable objectives plus identifier for one surviving candidate."""

    symbol: str
    expected_return_120s: float
    cvar_95_120s: float
    effective_cost_bps: float
    liquidity_usd: float


@dataclass(frozen=True)
class ParetoConfig:
    """Limits for survivor counts and tie-breaking (used by the filter in Prompt 2)."""

    min_survivors: int = 2
    max_survivors: int = 5
    tiebreaker: Literal[
        "expected_return_120s",
        "liquidity_usd",
        "effective_cost_bps",
    ] = "expected_return_120s"
    epsilon: float = 1e-9


class _DimensionDirections(dict[str, int]):
    """Maps each numeric :class:`CandidateScore` field (excluding ``symbol``) to its optimization direction.

    Values are ``+1`` if larger raw numbers are better, or ``-1`` if smaller raw numbers are better.

    **CVaR sign convention (most error-prone part):** CVaR is reported as a loss in return space—it is
    zero or **negative** (e.g. ``-0.03`` means worse tail loss than ``-0.01``). A CVaR of ``-0.03`` is
    **better** than a CVaR of ``-0.10`` because it is *less negative* (closer to zero). We therefore use
    direction ``+1`` (**maximize** the stored number): larger values mean safer tails. Flipping the sign
    on stored CVaR without changing this direction will invert dominance logic and silently mis-rank tokens.
    """


DIMENSION_DIRECTIONS: dict[str, int] = _DimensionDirections(
    {
        "expected_return_120s": +1,
        "cvar_95_120s": +1,
        "effective_cost_bps": -1,
        "liquidity_usd": +1,
    }
)
