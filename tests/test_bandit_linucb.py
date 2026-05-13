"""Pure LinUCB math tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.bandit.contracts import FEATURE_DIM, LinUCBArmState
from app.bandit.linucb import select_arm, ucb_score, update_arm


def _exploration_bonus(arm: LinUCBArmState, x: np.ndarray, alpha: float) -> float:
    s = np.linalg.solve(arm.A, x.reshape(-1, 1)).reshape(-1)
    return alpha * math.sqrt(float(np.dot(x, s)))


def test_fresh_arm_ucb_equals_alpha_times_sqrt_inv_reg() -> None:
    lam = 2.0
    alpha = 3.0
    arm = LinUCBArmState.fresh(lam)
    x = np.ones(FEATURE_DIM, dtype=np.float64)
    expected_bonus = alpha * math.sqrt(np.dot(x, x) / lam)
    assert abs(ucb_score(arm, x, alpha) - expected_bonus) < 1e-10


def test_ucb_increases_with_alpha() -> None:
    arm = LinUCBArmState.fresh(1.0)
    x = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    u1 = ucb_score(arm, x, 1.0)
    u5 = ucb_score(arm, x, 5.0)
    assert u5 > u1


def test_ucb_decreases_with_updates() -> None:
    arm = LinUCBArmState.fresh(1.0)
    x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    alpha = 1.0
    bonus_before = _exploration_bonus(arm, x, alpha)
    for _ in range(40):
        arm = update_arm(arm, x, 1.0, "2026-01-01T00:00:00Z")
    bonus_after = _exploration_bonus(arm, x, alpha)
    assert bonus_after < bonus_before


def test_negative_alpha_raises() -> None:
    arm = LinUCBArmState.fresh(1.0)
    x = np.zeros(FEATURE_DIM, dtype=np.float64)
    with pytest.raises(ValueError, match="alpha"):
        ucb_score(arm, x, -0.1)


def test_wrong_shape_x_raises() -> None:
    arm = LinUCBArmState.fresh(1.0)
    x = np.zeros(5, dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        ucb_score(arm, x, 1.0)


def test_picks_highest_ucb() -> None:
    arms = {
        "SOL": LinUCBArmState.fresh(1.0),
        "BONK": LinUCBArmState.fresh(1.0),
        "USDC": LinUCBArmState.fresh(1.0),
    }
    contexts = {
        "SOL": np.array([3.0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        "BONK": np.array([1.0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        "USDC": np.array([2.0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
    }
    chosen, scores = select_arm(arms, contexts, alpha=1.0)
    assert chosen == "SOL"
    assert scores["SOL"] > scores["BONK"]


def test_alphabetical_tiebreaker() -> None:
    x = np.array([1.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    arms = {s: LinUCBArmState.fresh(1.0) for s in ("SOL", "BONK", "USDC")}
    contexts = {s: x.copy() for s in arms}
    chosen, scores = select_arm(arms, contexts, alpha=1.0)
    assert scores["SOL"] == scores["BONK"] == scores["USDC"]
    assert chosen == "BONK"


def test_empty_arms_raises() -> None:
    with pytest.raises(ValueError, match="arms must"):
        select_arm({}, {}, alpha=1.0)


def test_key_mismatch_raises() -> None:
    arms = {"SOL": LinUCBArmState.fresh(1.0), "USDC": LinUCBArmState.fresh(1.0)}
    contexts = {
        "SOL": np.zeros(FEATURE_DIM),
        "BONK": np.zeros(FEATURE_DIM),
    }
    with pytest.raises(ValueError, match="keys"):
        select_arm(arms, contexts, alpha=1.0)


def test_a_increases_correctly() -> None:
    arm = LinUCBArmState.fresh(1.0)
    x = np.array([1.0, 2.0, 0, 0, 0, 0, 0], dtype=np.float64)
    new = update_arm(arm, x, 0.5, "t")
    expected_A = arm.A + np.outer(x, x)
    assert np.allclose(new.A, expected_A)


def test_b_increases_correctly() -> None:
    arm = LinUCBArmState.fresh(1.0)
    x = np.array([1.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    r = 2.5
    new = update_arm(arm, x, r, "t")
    assert np.allclose(new.b, arm.b + r * x)


def test_n_updates_increments() -> None:
    arm = LinUCBArmState.fresh(1.0)
    x = np.ones(FEATURE_DIM, dtype=np.float64)
    for k in range(1, 4):
        arm = update_arm(arm, x, 1.0, "t")
        assert arm.n_updates == k


def test_last_update_utc_set() -> None:
    arm = LinUCBArmState.fresh(1.0)
    ts = "2026-05-12T18:00:00Z"
    new = update_arm(arm, np.ones(FEATURE_DIM), 1.0, ts)
    assert new.last_update_utc == ts


def test_returns_new_instance_not_mutation() -> None:
    arm = LinUCBArmState.fresh(1.0)
    x = np.ones(FEATURE_DIM, dtype=np.float64)
    A_old = arm.A.copy()
    new = update_arm(arm, x, 1.0, "t")
    assert np.allclose(arm.A, A_old)
    assert not np.allclose(new.A, A_old)


def test_nan_reward_raises() -> None:
    arm = LinUCBArmState.fresh(1.0)
    x = np.ones(FEATURE_DIM, dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        update_arm(arm, x, float("nan"), "t")


def test_inf_reward_raises() -> None:
    arm = LinUCBArmState.fresh(1.0)
    x = np.ones(FEATURE_DIM, dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        update_arm(arm, x, float("inf"), "t")


def test_update_wrong_shape_x_raises() -> None:
    arm = LinUCBArmState.fresh(1.0)
    with pytest.raises(ValueError, match="shape"):
        update_arm(arm, np.zeros(3), 1.0, "t")
