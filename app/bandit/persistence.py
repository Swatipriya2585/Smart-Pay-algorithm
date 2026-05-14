"""Atomic JSON persistence for LinUCB arm state."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel

from app.bandit.contracts import FEATURE_DIM, FEATURE_NAMES, LinUCBArmState, LinUCBConfig

logger = logging.getLogger(__name__)


class ArmStateOnDisk(BaseModel):
    """One arm as stored in linucb_state.json."""

    A: list[list[float]]
    b: list[float]
    n_updates: int
    last_update_utc: Optional[str] = None


class LinUCBStateFile(BaseModel):
    """On-disk schema for LinUCB state."""

    schema_version: int = 1
    config_hash: str
    feature_dim: int
    generated_at_utc: str
    arms: dict[str, ArmStateOnDisk]


def config_hash(config: LinUCBConfig) -> str:
    """Stable hash of config fields that affect stored math.

    Hash inputs: alpha, regularization, FEATURE_DIM, FEATURE_NAMES.
    Returns SHA-256 hex digest truncated to 16 characters.
    """
    payload: dict[str, Any] = {
        "alpha": config.alpha,
        "regularization": config.regularization,
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def serialize_arm(arm: LinUCBArmState) -> ArmStateOnDisk:
    """Convert in-memory arm to on-disk model."""
    if arm.A.shape != (FEATURE_DIM, FEATURE_DIM):
        raise ValueError(f"A must have shape ({FEATURE_DIM}, {FEATURE_DIM}), got {arm.A.shape}")
    if arm.b.shape != (FEATURE_DIM,):
        raise ValueError(f"b must have shape ({FEATURE_DIM},), got {arm.b.shape}")
    return ArmStateOnDisk(
        A=arm.A.tolist(),
        b=arm.b.tolist(),
        n_updates=arm.n_updates,
        last_update_utc=arm.last_update_utc,
    )


def deserialize_arm(on_disk: ArmStateOnDisk) -> LinUCBArmState:
    """Restore LinUCBArmState from disk model."""
    if len(on_disk.A) != FEATURE_DIM:
        raise ValueError(
            f"A on disk must have {FEATURE_DIM} rows, got {len(on_disk.A)}"
        )
    for i, row in enumerate(on_disk.A):
        if len(row) != FEATURE_DIM:
            raise ValueError(
                f"A on disk row {i} must have length {FEATURE_DIM}, got {len(row)}"
            )
    A = np.asarray(on_disk.A, dtype=np.float64)
    if A.shape != (FEATURE_DIM, FEATURE_DIM):
        raise ValueError(
            f"A on disk must have shape ({FEATURE_DIM}, {FEATURE_DIM}), got {A.shape}"
        )
    b = np.asarray(on_disk.b, dtype=np.float64).reshape(-1)
    if b.shape != (FEATURE_DIM,):
        raise ValueError(f"b on disk must have length {FEATURE_DIM}, got {b.shape[0]}")
    return LinUCBArmState(
        A=A,
        b=b,
        n_updates=on_disk.n_updates,
        last_update_utc=on_disk.last_update_utc,
    )


def _default_state_path(path: Path | str | None) -> Path:
    if path is not None:
        return Path(path)
    here = Path(__file__).resolve()
    return here.parent.parent.parent / "data" / "linucb_state.json"


def load_state(
    config: LinUCBConfig,
    path: Path | str | None = None,
) -> dict[str, LinUCBArmState]:
    """Load bandit arms from disk. Missing file → empty dict (cold start)."""
    resolved = _default_state_path(path)
    if not resolved.exists():
        logger.warning(
            "LinUCB state file not found at %s — cold start (no arms on disk).",
            resolved,
        )
        return {}
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    file = LinUCBStateFile(**raw)
    expected_hash = config_hash(config)
    if file.config_hash != expected_hash:
        raise ValueError(
            "stored state's config_hash does not match current config; refusing to load. Either "
            "revert config changes or delete linucb_state.json to cold-start."
        )
    if file.feature_dim != FEATURE_DIM:
        raise ValueError(
            f"stored feature_dim {file.feature_dim} does not match runtime FEATURE_DIM {FEATURE_DIM}"
        )
    arms = {sym: deserialize_arm(arm_disk) for sym, arm_disk in file.arms.items()}
    logger.info("Loaded LinUCB state from %s (%d arms).", resolved, len(arms))
    return arms


def save_state(
    arms: dict[str, LinUCBArmState],
    config: LinUCBConfig,
    path: Path | str | None = None,
    now_utc_iso: Optional[str] = None,
) -> Path:
    """Atomically write LinUCB state to JSON."""
    resolved = _default_state_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    ts = now_utc_iso or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    state = LinUCBStateFile(
        schema_version=1,
        config_hash=config_hash(config),
        feature_dim=FEATURE_DIM,
        generated_at_utc=ts,
        arms={sym: serialize_arm(arm) for sym, arm in arms.items()},
    )
    payload = state.model_dump(mode="json")
    text = json.dumps(payload, indent=2, sort_keys=True)
    tmp_path = resolved.with_suffix(resolved.suffix + ".tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, resolved)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    logger.info("Saved LinUCB state to %s (%d arms).", resolved, len(arms))
    return resolved


def get_or_create_arm(
    arms: dict[str, LinUCBArmState],
    symbol: str,
    config: LinUCBConfig,
) -> LinUCBArmState:
    """Return arms[symbol] if present, else a fresh ridge arm."""
    existing = arms.get(symbol)
    if existing is not None:
        return existing
    return LinUCBArmState.fresh(config.regularization)
