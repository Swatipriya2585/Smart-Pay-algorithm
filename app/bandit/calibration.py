"""
Bandit feature normalization constants.

Reads data/bandit_calibration.json (defaults; Step 12.5 recalibrates from backtest).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class BanditCalibrationFile(BaseModel):
    """Top-level shape of bandit_calibration.json."""

    schema_version: int
    generated_at_utc: str
    log_amount_divisor: float
    vol_clip_max: float
    liquidity_ratio_log_divisor: float
    spread_clip_max_bps: float


class BanditCalibration:
    """Loads bandit_calibration.json and exposes normalization constants.

    Caches the parsed result so repeated lookups are free. Re-instantiate
    if you need to pick up a fresh file.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            here = Path(__file__).resolve()
            path = here.parent.parent.parent / "data" / "bandit_calibration.json"
        self.path = Path(path)
        self._file = self._load()

    def _load(self) -> BanditCalibrationFile:
        if not self.path.exists():
            raise FileNotFoundError(
                f"bandit_calibration.json not found at {self.path}. "
                f"Use the default file in data/ or pass an explicit path."
            )
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        file = BanditCalibrationFile(**raw)
        for name in (
            "log_amount_divisor",
            "vol_clip_max",
            "liquidity_ratio_log_divisor",
            "spread_clip_max_bps",
        ):
            val = getattr(file, name)
            if val <= 0:
                raise ValueError(f"{name} must be strictly positive, got {val}")
        return file

    @property
    def log_amount_divisor(self) -> float:
        return self._file.log_amount_divisor

    @property
    def vol_clip_max(self) -> float:
        return self._file.vol_clip_max

    @property
    def liquidity_ratio_log_divisor(self) -> float:
        return self._file.liquidity_ratio_log_divisor

    @property
    def spread_clip_max_bps(self) -> float:
        return self._file.spread_clip_max_bps
