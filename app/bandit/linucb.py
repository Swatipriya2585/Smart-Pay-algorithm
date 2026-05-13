"""Pure LinUCB math (no I/O, no logging)."""

from __future__ import annotations

import math

import numpy as np

from app.bandit.contracts import FEATURE_DIM, LinUCBArmState


def ucb_score(
    arm: LinUCBArmState,
    x: np.ndarray,
    alpha: float,
) -> float:
    """Upper confidence bound for one arm given context x.

    Returns: theta_hat @ x + alpha * sqrt(x.T @ A_inv @ x)
    where theta_hat = A_inv @ b.

    Uses np.linalg.solve(A, x) and np.linalg.solve(A, b) — never explicit A_inv.

    Raises:
        ValueError: if x.shape != (FEATURE_DIM,).
        ValueError: if alpha < 0.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if x.shape != (FEATURE_DIM,):
        raise ValueError(f"x must have shape ({FEATURE_DIM},), got {x.shape}")
    x_col = x.reshape(-1, 1)
    theta_hat = np.linalg.solve(arm.A, arm.b.reshape(-1, 1)).reshape(-1)
    s = np.linalg.solve(arm.A, x_col).reshape(-1)
    exploitation = float(np.dot(theta_hat, x))
    bonus = alpha * math.sqrt(float(np.dot(x, s)))
    return exploitation + bonus


def select_arm(
    arms: dict[str, LinUCBArmState],
    contexts: dict[str, np.ndarray],
    alpha: float,
) -> tuple[str, dict[str, float]]:
    """Pick the arm with the highest UCB.

    Tie-breaker: alphabetically lowest symbol wins.

    Raises:
        ValueError: if arms is empty.
        ValueError: if contexts.keys() != arms.keys().
    """
    if not arms:
        raise ValueError("arms must be non-empty")
    if set(contexts.keys()) != set(arms.keys()):
        raise ValueError(
            f"contexts keys {sorted(contexts)} must match arms keys {sorted(arms)}"
        )
    scores: dict[str, float] = {
        sym: ucb_score(arms[sym], contexts[sym], alpha) for sym in arms
    }
    best_ucb = max(scores.values())
    tied = [sym for sym, sc in scores.items() if sc == best_ucb]
    chosen = min(tied)
    return chosen, scores


def update_arm(
    arm: LinUCBArmState,
    x: np.ndarray,
    reward: float,
    now_utc_iso: str,
) -> LinUCBArmState:
    """Return a NEW LinUCBArmState reflecting the observation.

    A_new = A + outer(x, x)
    b_new = b + reward * x

    Does NOT mutate the input arm.

    Raises:
        ValueError: if x.shape != (FEATURE_DIM,).
        ValueError: if reward is NaN or inf.
    """
    if x.shape != (FEATURE_DIM,):
        raise ValueError(f"x must have shape ({FEATURE_DIM},), got {x.shape}")
    if not math.isfinite(reward):
        raise ValueError(f"reward must be finite, got {reward}")
    x_flat = x.reshape(-1)
    A_new = arm.A + np.outer(x_flat, x_flat)
    b_new = arm.b + reward * x_flat
    return LinUCBArmState(
        A=A_new,
        b=b_new,
        n_updates=arm.n_updates + 1,
        last_update_utc=now_utc_iso,
    )
