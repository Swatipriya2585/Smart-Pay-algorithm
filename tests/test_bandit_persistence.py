"""Tests for LinUCB JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app.bandit.contracts import FEATURE_DIM, LinUCBArmState, LinUCBConfig
from app.bandit.persistence import (
    ArmStateOnDisk,
    LinUCBStateFile,
    config_hash,
    deserialize_arm,
    get_or_create_arm,
    load_state,
    save_state,
    serialize_arm,
)


def test_hash_is_deterministic() -> None:
    cfg = LinUCBConfig()
    assert config_hash(cfg) == config_hash(cfg)


def test_hash_changes_with_alpha() -> None:
    a = LinUCBConfig(alpha=1.0)
    b = LinUCBConfig(alpha=2.0)
    assert config_hash(a) != config_hash(b)


def test_hash_changes_with_regularization() -> None:
    a = LinUCBConfig(regularization=1.0)
    b = LinUCBConfig(regularization=2.0)
    assert config_hash(a) != config_hash(b)


def test_hash_independent_of_reward_horizon() -> None:
    a = LinUCBConfig(reward_horizon_seconds=60.0)
    b = LinUCBConfig(reward_horizon_seconds=999.0)
    assert config_hash(a) == config_hash(b)


def test_round_trip_preserves_a() -> None:
    A = np.eye(FEATURE_DIM, dtype=np.float64) * 2.0
    arm = LinUCBArmState(A=A, b=np.zeros(FEATURE_DIM), n_updates=0)
    disk = serialize_arm(arm)
    back = deserialize_arm(disk)
    assert np.allclose(back.A, A)


def test_round_trip_preserves_b() -> None:
    b = np.arange(FEATURE_DIM, dtype=np.float64) * 0.1
    arm = LinUCBArmState(A=np.eye(FEATURE_DIM), b=b, n_updates=2, last_update_utc=None)
    back = deserialize_arm(serialize_arm(arm))
    assert np.allclose(back.b, b)


def test_round_trip_preserves_n_updates_and_timestamp() -> None:
    arm = LinUCBArmState(
        A=np.eye(FEATURE_DIM),
        b=np.ones(FEATURE_DIM),
        n_updates=42,
        last_update_utc="2026-01-02T03:04:05Z",
    )
    back = deserialize_arm(serialize_arm(arm))
    assert back.n_updates == 42
    assert back.last_update_utc == "2026-01-02T03:04:05Z"


def test_serialize_wrong_shape_a_raises() -> None:
    arm = LinUCBArmState.fresh(1.0)
    arm.A = np.zeros((6, 6), dtype=np.float64)  # type: ignore[misc]
    with pytest.raises(ValueError, match="A must have shape"):
        serialize_arm(arm)


def test_deserialize_wrong_shape_a_raises() -> None:
    bad = ArmStateOnDisk(
        A=[[0.0] * 6 for _ in range(6)],
        b=[0.0] * FEATURE_DIM,
        n_updates=0,
    )
    with pytest.raises(ValueError, match="A on disk must have"):
        deserialize_arm(bad)


def test_load_missing_file_returns_empty_dict(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.WARNING)
    p = tmp_path / "missing.json"
    out = load_state(LinUCBConfig(), path=p)
    assert out == {}
    assert any(
        "cold start" in r.getMessage().lower() or "not found" in r.getMessage().lower()
        for r in caplog.records
    )


def test_load_existing_file_round_trip(tmp_path: Path) -> None:
    cfg = LinUCBConfig()
    arms_in = {
        "SOL": LinUCBArmState(
            A=np.eye(FEATURE_DIM) * 1.5,
            b=np.linspace(0.1, 0.7, FEATURE_DIM),
            n_updates=3,
            last_update_utc="2026-05-01T00:00:00Z",
        )
    }
    path = tmp_path / "state.json"
    save_state(arms_in, cfg, path=path, now_utc_iso="2026-05-01T00:00:00Z")
    arms_out = load_state(cfg, path=path)
    assert set(arms_out) == {"SOL"}
    assert np.allclose(arms_out["SOL"].A, arms_in["SOL"].A)
    assert np.allclose(arms_out["SOL"].b, arms_in["SOL"].b)
    assert arms_out["SOL"].n_updates == 3


def test_load_config_hash_mismatch_raises(tmp_path: Path) -> None:
    cfg_a = LinUCBConfig(alpha=1.0)
    cfg_b = LinUCBConfig(alpha=9.0)
    path = tmp_path / "state.json"
    save_state({}, cfg_a, path=path, now_utc_iso="t")
    with pytest.raises(ValueError, match="config_hash"):
        load_state(cfg_b, path=path)


def test_load_feature_dim_mismatch_raises(tmp_path: Path) -> None:
    cfg = LinUCBConfig()
    bogus = {
        "schema_version": 1,
        "config_hash": config_hash(cfg),
        "feature_dim": 5,
        "generated_at_utc": "2026-01-01T00:00:00Z",
        "arms": {},
    }
    path = tmp_path / "bad_dim.json"
    path.write_text(json.dumps(bogus), encoding="utf-8")
    with pytest.raises(ValueError, match="feature_dim"):
        load_state(cfg, path=path)


def test_load_invalid_schema_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_state(LinUCBConfig(), path=path)


def test_save_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    save_state({}, LinUCBConfig(), path=path, now_utc_iso="2026-01-01T00:00:00Z")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    LinUCBStateFile(**data)


def test_save_atomic_no_tmp_left_behind(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    save_state({}, LinUCBConfig(), path=path, now_utc_iso="t")
    assert not (tmp_path / "out.json.tmp").exists()


def test_save_overwrites_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    cfg = LinUCBConfig()
    save_state({"SOL": LinUCBArmState.fresh(1.0)}, cfg, path=path, now_utc_iso="t1")
    save_state({"BONK": LinUCBArmState.fresh(2.0)}, cfg, path=path, now_utc_iso="t2")
    loaded = load_state(cfg, path=path)
    assert set(loaded) == {"BONK"}
    assert np.allclose(loaded["BONK"].A, 2.0 * np.eye(FEATURE_DIM))


def test_save_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    cfg = LinUCBConfig()
    arms = {"SOL": LinUCBArmState.fresh(1.0)}
    ts = "2026-06-01T12:00:00Z"
    save_state(arms, cfg, path=path, now_utc_iso=ts)
    b1 = path.read_bytes()
    save_state(arms, cfg, path=path, now_utc_iso=ts)
    b2 = path.read_bytes()
    assert b1 == b2


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    cfg = LinUCBConfig(regularization=1.5)
    arms = {
        "SOL": LinUCBArmState.fresh(1.5),
        "BONK": LinUCBArmState(
            A=np.eye(FEATURE_DIM) * 2.0,
            b=np.ones(FEATURE_DIM) * 0.25,
            n_updates=10,
            last_update_utc=None,
        ),
    }
    path = tmp_path / "full.json"
    save_state(arms, cfg, path=path, now_utc_iso="z")
    loaded = load_state(cfg, path=path)
    for sym in arms:
        assert np.allclose(loaded[sym].A, arms[sym].A)
        assert np.allclose(loaded[sym].b, arms[sym].b)
        assert loaded[sym].n_updates == arms[sym].n_updates


def test_returns_existing_arm() -> None:
    cfg = LinUCBConfig()
    arm = LinUCBArmState.fresh(1.0)
    arms = {"SOL": arm}
    got = get_or_create_arm(arms, "SOL", cfg)
    assert got is arm


def test_creates_fresh_arm_for_missing() -> None:
    cfg = LinUCBConfig(regularization=2.0)
    arms: dict[str, LinUCBArmState] = {}
    got = get_or_create_arm(arms, "NEW", cfg)
    assert np.allclose(got.A, 2.0 * np.eye(FEATURE_DIM))
    assert np.allclose(got.b, np.zeros(FEATURE_DIM))
    assert got.n_updates == 0


def test_does_not_mutate_input_dict() -> None:
    cfg = LinUCBConfig()
    arms: dict[str, LinUCBArmState] = {"SOL": LinUCBArmState.fresh(1.0)}
    _ = get_or_create_arm(arms, "BONK", cfg)
    assert set(arms) == {"SOL"}
