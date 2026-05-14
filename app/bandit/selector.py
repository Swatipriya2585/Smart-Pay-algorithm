"""LinUCB candidate selection for one request (read-only; no persistence)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import LinUCBArmState, LinUCBConfig
from app.bandit.linucb import select_arm
from app.bandit.persistence import get_or_create_arm
from app.bandit.vectorize import build_feature_vector
from app.market_data.calibration import Calibration
from app.schemas import RamhdContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectorResult:
    """Result of one bandit selection."""

    chosen_symbol: str
    chosen_feature_vector: np.ndarray
    ucb_scores: dict[str, float]
    feature_vectors: dict[str, np.ndarray]


def pick_candidate(
    context: RamhdContext,
    candidate_symbols: list[str],
    arms: dict[str, LinUCBArmState],
    config: LinUCBConfig,
    calibration: Calibration,
    bandit_calibration: BanditCalibration,
) -> SelectorResult:
    """Pick one candidate among ``candidate_symbols`` using LinUCB (read-only)."""
    if not candidate_symbols:
        raise ValueError("candidate_symbols must be non-empty")

    arms_for_request: dict[str, LinUCBArmState] = {}
    feature_vectors: dict[str, np.ndarray] = {}
    for symbol in candidate_symbols:
        x_s = build_feature_vector(context, symbol, calibration, bandit_calibration)
        arm_s = get_or_create_arm(arms, symbol, config)
        arms_for_request[symbol] = arm_s
        feature_vectors[symbol] = x_s

    chosen_symbol, ucb_scores = select_arm(
        arms_for_request, feature_vectors, config.alpha
    )
    chosen_vec = feature_vectors[chosen_symbol]
    max_ucb = max(ucb_scores.values())
    logger.info(
        "selected %s among %d candidates (ucb=%.4f)",
        chosen_symbol,
        len(candidate_symbols),
        max_ucb,
    )
    return SelectorResult(
        chosen_symbol=chosen_symbol,
        chosen_feature_vector=np.array(chosen_vec, dtype=np.float64, copy=True),
        ucb_scores=dict(ucb_scores),
        feature_vectors={k: np.array(v, dtype=np.float64, copy=True) for k, v in feature_vectors.items()},
    )
