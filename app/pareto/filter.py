"""
Pareto survivor filtering: skyline plus floor/cap and deterministic tie-breaks.
"""

from __future__ import annotations

import logging
from typing import Iterable

from app.pareto.contracts import (
    DIMENSION_DIRECTIONS,
    CandidateScore,
    ParetoConfig,
)
from app.pareto.dominance import pareto_front

logger = logging.getLogger(__name__)

__all__ = ["apply_pareto_filter"]


def _signed_tiebreaker(c: CandidateScore, tiebreaker_field: str) -> float:
    sign = DIMENSION_DIRECTIONS[tiebreaker_field]
    return getattr(c, tiebreaker_field) * sign


def _sort_tuple_for_ranking(
    c: CandidateScore,
    tiebreaker_field: str,
    original_index: int,
) -> tuple[float, int, str]:
    """Primary: maximize signed tiebreaker; then input index; then symbol (ASCII)."""
    signed = _signed_tiebreaker(c, tiebreaker_field)
    return (-signed, original_index, c.symbol)


def _index_lookup(candidates: list[CandidateScore]) -> dict[int, int]:
    """Map ``id(candidate)`` → position in ``candidates`` (stable for this call)."""
    return {id(c): i for i, c in enumerate(candidates)}


def _sort_candidates_deterministic(
    rows: Iterable[CandidateScore],
    tiebreaker_field: str,
    orig_index: dict[int, int],
) -> list[CandidateScore]:
    """Sort using chained tuple keys only (no single-key sorted shortcut)."""
    decorated = [
        (_sort_tuple_for_ranking(c, tiebreaker_field, orig_index[id(c)]), c) for c in rows
    ]
    decorated.sort(key=lambda pair: pair[0])
    return [pair[1] for pair in decorated]


def apply_pareto_filter(
    candidates: list[CandidateScore],
    config: ParetoConfig,
) -> list[CandidateScore]:
    """
    Wraps :func:`~app.pareto.dominance.pareto_front` with three guarantees:

    1. **Floor:** if the front has fewer than ``min_survivors`` **and**
       ``min_survivors > 0``, and dominated candidates exist, add the next-best
       dominated rows ranked by the configured tiebreaker (sign from
       ``DIMENSION_DIRECTIONS``), then input index, then symbol, until the floor
       is met or dominated candidates are exhausted. Logs at INFO with counts and
       symbols.

    2. **Cap:** if ``max_survivors > 0`` and the working set is larger than
       ``max_survivors``, keep only the top rows by the same tiebreaker tuple sort.
       Logs at INFO with symbols dropped.

    3. **Determinism:** ties use tuple ``(-signed_tiebreaker, original_index,
       symbol)`` ascending — equivalent to maximizing signed tiebreaker, then
       earlier input position, then lexicographic ``symbol`` for backtest
       reproducibility (Step 12).

    **Edge behavior (never raises):**

    * Empty input → ``[]`` with WARNING.

    * ``len(candidates) < min_survivors`` (when ``min_survivors > 0``) → return
      **all** input candidates (WARNING); cap may still apply.

    * ``min_survivors == 0`` → no floor relaxation.

    * ``max_survivors == 0`` → **no cap** (treat as unlimited).

    Returns the surviving list in the order produced by the final ranking step
    when trimming applies; otherwise preserves skyline order then dominated adds.
    """
    eps = config.epsilon
    tb = config.tiebreaker
    min_req = max(0, config.min_survivors)
    max_req = config.max_survivors

    if not candidates:
        logger.warning("pareto_filter: empty candidate list; returning []")
        return []

    if min_req > 0 and len(candidates) < min_req:
        logger.warning(
            "pareto_filter: input size %d below min_survivors=%d; returning all "
            "candidates (cap may still trim)",
            len(candidates),
            min_req,
        )
        working = list(candidates)
        orig_index = _index_lookup(candidates)
        if max_req > 0 and len(working) > max_req:
            ranked = _sort_candidates_deterministic(working, tb, orig_index)
            kept = ranked[:max_req]
            dropped = ranked[max_req:]
            logger.info(
                "pareto_filter: capped shortfall set from %d to %d; dropped symbols: %s",
                len(working),
                max_req,
                ", ".join(c.symbol for c in dropped),
            )
            return kept
        return working

    orig_index = _index_lookup(candidates)

    front = pareto_front(candidates, eps)
    front_ids = {id(c) for c in front}

    dominated = [c for c in candidates if id(c) not in front_ids]

    result: list[CandidateScore] = list(front)

    if min_req > 0 and len(result) < min_req and dominated:
        need = min_req - len(result)
        ranked_dominated = _sort_candidates_deterministic(dominated, tb, orig_index)
        taken = ranked_dominated[:need]
        result.extend(taken)
        logger.info(
            "pareto_filter: floor relaxation added %d dominated candidate(s): %s",
            len(taken),
            ", ".join(c.symbol for c in taken),
        )

    if max_req > 0 and len(result) > max_req:
        ranked_all = _sort_candidates_deterministic(result, tb, orig_index)
        kept = ranked_all[:max_req]
        dropped = ranked_all[max_req:]
        logger.info(
            "pareto_filter: capped survivors from %d to %d; dropped symbols: %s",
            len(result),
            max_req,
            ", ".join(c.symbol for c in dropped),
        )
        return kept

    return result
