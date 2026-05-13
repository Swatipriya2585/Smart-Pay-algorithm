"""LinUCB pipeline: decision time (Pareto survivors) and reward persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import FEATURE_DIM, LinUCBConfig
from app.bandit.linucb import update_arm
from app.bandit.persistence import get_or_create_arm, load_state, save_state
from app.bandit.selector import pick_candidate
from app.market_data.calibration import Calibration
from app.pareto.contracts import CandidateScore
from app.schemas import RamhdContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BanditDecision:
    """Result of one bandit selection at decision time.

    The caller stores this (or its fields) somewhere durable so that
    when a reward arrives later, record_observation can be called
    with the matching context + chosen_feature_vector.
    """

    chosen_symbol: str
    chosen_feature_vector: np.ndarray
    ucb_scores: dict[str, float]
    feature_vectors: dict[str, np.ndarray]
    candidates_evaluated: tuple[str, ...]
    decision_utc: str


def run_bandit_stage(
    context: RamhdContext,
    survivors: list[CandidateScore],
    config: LinUCBConfig,
    calibration: Calibration,
    bandit_calibration: BanditCalibration,
    state_path: Path | str | None = None,
    now_utc_iso: Optional[str] = None,
) -> BanditDecision:
    """Run LinUCB on Pareto's survivors and return the bandit's choice.

    Read-only with respect to disk: loads state, picks one survivor,
    returns a BanditDecision. Does NOT call save_state.
    """
    if not survivors:
        raise ValueError("survivors must be non-empty")

    candidate_symbols = [s.symbol for s in survivors]
    decision_utc = (
        now_utc_iso
        if now_utc_iso is not None
        else datetime.now(timezone.utc).isoformat()
    )

    arms = load_state(config, state_path)
    result = pick_candidate(
        context,
        candidate_symbols,
        arms,
        config,
        calibration,
        bandit_calibration,
    )
    max_ucb = max(result.ucb_scores.values())
    logger.info(
        "bandit chose %s among %d survivors (ucb=%.4f) at %s",
        result.chosen_symbol,
        len(survivors),
        max_ucb,
        decision_utc,
    )
    return BanditDecision(
        chosen_symbol=result.chosen_symbol,
        chosen_feature_vector=np.array(
            result.chosen_feature_vector, dtype=np.float64, copy=True
        ),
        ucb_scores=dict(result.ucb_scores),
        feature_vectors={
            k: np.array(v, dtype=np.float64, copy=True)
            for k, v in result.feature_vectors.items()
        },
        candidates_evaluated=tuple(candidate_symbols),
        decision_utc=decision_utc,
    )


def record_observation(
    chosen_symbol: str,
    chosen_feature_vector: np.ndarray,
    reward: float,
    config: LinUCBConfig,
    state_path: Path | str | None = None,
    now_utc_iso: Optional[str] = None,
) -> Path:
    """Update the bandit's state with one observed reward."""
    if not np.isfinite(reward):
        raise ValueError(f"reward must be finite, got {reward}")
    x = np.asarray(chosen_feature_vector, dtype=np.float64).reshape(-1)
    if x.shape != (FEATURE_DIM,):
        raise ValueError(
            f"chosen_feature_vector must have shape ({FEATURE_DIM},), got {x.shape}"
        )

    update_utc = (
        now_utc_iso
        if now_utc_iso is not None
        else datetime.now(timezone.utc).isoformat()
    )

    arms = load_state(config, state_path)
    arm = get_or_create_arm(arms, chosen_symbol, config)
    new_arm = update_arm(arm, x, float(reward), update_utc)
    arms[chosen_symbol] = new_arm

    final_path = save_state(arms, config, state_path, update_utc)
    logger.info(
        "recorded reward %.4f for %s (arm has %d updates now)",
        reward,
        chosen_symbol,
        new_arm.n_updates,
    )
    return final_path
