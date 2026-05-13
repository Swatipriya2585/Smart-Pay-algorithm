"""
LinUCB bandit contracts and configuration (RAMHD Step 9).

Math/state types use dataclasses (same convention as app.pareto.contracts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

FEATURE_DIM = 7

FEATURE_NAMES: tuple[str, ...] = (
    "log_amount",
    "congestion",
    "volatility",
    "liquidity_ratio",
    "spread",
    "is_stable",
    "bias",
)


@dataclass(frozen=True)
class LinUCBConfig:
    """Hyperparameters for LinUCB."""

    alpha: float = 1.0
    regularization: float = 1.0
    reward_horizon_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.regularization <= 0:
            raise ValueError(
                f"regularization must be positive, got {self.regularization}"
            )
        if self.reward_horizon_seconds <= 0:
            raise ValueError(
                f"reward_horizon_seconds must be positive, got {self.reward_horizon_seconds}"
            )


@dataclass
class LinUCBArmState:
    """The (A, b) state for one arm.

    A is (FEATURE_DIM, FEATURE_DIM), initially regularization * I.
    b is (FEATURE_DIM,), initially zeros.
    n_updates tracks how many times this arm has been observed.
    last_update_utc is an ISO 8601 string, None until first update.
    """

    A: np.ndarray
    b: np.ndarray
    n_updates: int = 0
    last_update_utc: Optional[str] = None

    @classmethod
    def fresh(cls, regularization: float) -> LinUCBArmState:
        """Create a cold-start arm: A = λI, b = 0."""
        return cls(
            A=regularization * np.eye(FEATURE_DIM),
            b=np.zeros(FEATURE_DIM),
            n_updates=0,
            last_update_utc=None,
        )

    def __post_init__(self) -> None:
        if self.A.shape != (FEATURE_DIM, FEATURE_DIM):
            raise ValueError(
                f"A must be ({FEATURE_DIM},{FEATURE_DIM}), got {self.A.shape}"
            )
        if self.b.shape != (FEATURE_DIM,):
            raise ValueError(f"b must be ({FEATURE_DIM},), got {self.b.shape}")
