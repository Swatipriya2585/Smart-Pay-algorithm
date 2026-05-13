"""Tests for BanditCalibration loader."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.bandit.calibration import BanditCalibration


def test_loads_default_path_from_data_dir() -> None:
    cal = BanditCalibration()
    assert cal.log_amount_divisor == 4.0
    assert cal.vol_clip_max == 1.0
    assert cal.liquidity_ratio_log_divisor == 6.0
    assert cal.spread_clip_max_bps == 500.0


def test_loads_explicit_path(tmp_path: Path) -> None:
    p = tmp_path / "bc.json"
    p.write_text(
        '{"schema_version":1,"generated_at_utc":"2020-01-01T00:00:00Z",'
        '"log_amount_divisor":2.0,"vol_clip_max":0.5,'
        '"liquidity_ratio_log_divisor":3.0,"spread_clip_max_bps":100.0}',
        encoding="utf-8",
    )
    cal = BanditCalibration(path=p)
    assert cal.log_amount_divisor == 2.0
    assert cal.vol_clip_max == 0.5
    assert cal.liquidity_ratio_log_divisor == 3.0
    assert cal.spread_clip_max_bps == 100.0


def test_missing_file_raises_file_not_found_error() -> None:
    missing = Path("/tmp/nonexistent_bandit_calibration_xyz.json")
    with pytest.raises(FileNotFoundError, match="bandit_calibration.json not found"):
        BanditCalibration(path=missing)


def test_invalid_schema_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValidationError):
        BanditCalibration(path=p)


def test_zero_or_negative_divisor_raises(tmp_path: Path) -> None:
    p = tmp_path / "zero_div.json"
    p.write_text(
        '{"schema_version":1,"generated_at_utc":"2020-01-01T00:00:00Z",'
        '"log_amount_divisor":0.0,"vol_clip_max":1.0,'
        '"liquidity_ratio_log_divisor":6.0,"spread_clip_max_bps":500.0}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="log_amount_divisor"):
        BanditCalibration(path=p)


def test_zero_or_negative_clip_raises(tmp_path: Path) -> None:
    p = tmp_path / "neg_spread.json"
    p.write_text(
        '{"schema_version":1,"generated_at_utc":"2020-01-01T00:00:00Z",'
        '"log_amount_divisor":4.0,"vol_clip_max":1.0,'
        '"liquidity_ratio_log_divisor":6.0,"spread_clip_max_bps":-1.0}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="spread_clip_max_bps"):
        BanditCalibration(path=p)
