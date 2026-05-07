"""Verify the blending formula and Calibration loader.

These tests use HAND-COMPUTED expected values for the shrinkage formula
so any future regression is caught precisely. Computing values by hand
also verifies the test author understood the math correctly.
"""

import math
from pathlib import Path

import pytest

from app.market_data.calibration import (
    BlendingConfig,
    Calibration,
    TokenCalibration,
    blended_drift,
)


def _make_token(
    symbol: str = "SOL",
    is_stablecoin: bool = False,
    annualized_drift: float = -1.59,
    annualized_vol: float = 0.767,
    current_price: float = 84.43,
) -> TokenCalibration:
    """Build a TokenCalibration for testing."""
    return TokenCalibration(
        symbol=symbol,
        coingecko_id=symbol.lower(),
        chain="solana",
        address="testaddr",
        regime="major",
        is_stablecoin=is_stablecoin,
        n_observations=91,
        current_price_usd=current_price,
        min_price_usd=current_price * 0.5,
        max_price_usd=current_price * 1.5,
        daily_log_return_mean=annualized_drift / 365.0,
        daily_log_return_std=annualized_vol / math.sqrt(365.0),
        annualized_drift=annualized_drift,
        annualized_vol=annualized_vol,
    )


# -----------------------------------------------------------------------------
# Blending formula — hand-computed expected values
# -----------------------------------------------------------------------------


def test_stablecoin_always_zero_regardless_of_regime() -> None:
    """Stablecoins must bypass blending entirely. mu = 0 in every regime."""
    usdc = _make_token(symbol="USDC", is_stablecoin=True, annualized_drift=0.05)
    cfg = BlendingConfig()
    for regime in ("calm", "stress", "shock"):
        assert blended_drift(usdc, regime, cfg) == 0.0


def test_realized_drift_cap_applied() -> None:
    """SOL has realized -159% drift. With cap=50%, the effective input must be -50%."""
    sol = _make_token(annualized_drift=-1.59)  # -159%
    cfg = BlendingConfig(alpha=0.0, regime_multipliers={"stress": 1.0, "calm": 0.3, "shock": 2.0})
    # alpha=0 means pure realized (clipped). Stress mult=1.0. Expected: -0.5.
    result = blended_drift(sol, "stress", cfg)
    assert abs(result - (-0.5)) < 1e-12


def test_uncapped_when_realized_within_band() -> None:
    """AERO has realized -8.8% drift, well within +/-50%. Cap should not bite."""
    aero = _make_token(symbol="AERO", annualized_drift=-0.088)
    cfg = BlendingConfig(alpha=0.0)  # pure realized
    # Expected: -0.088 * 1.0 (stress mult) = -0.088
    result = blended_drift(aero, "stress", cfg)
    assert abs(result - (-0.088)) < 1e-12


def test_blending_at_alpha_07_baseline_zero() -> None:
    """SOL with default config: alpha=0.7, baseline=0%, clipped realized=-50%.

    mu_blended = 0.7*0 + 0.3*(-0.5) = -0.15
    stress regime mult = 1.0 -> expected = -0.15
    """
    sol = _make_token(annualized_drift=-1.59)
    cfg = BlendingConfig()  # defaults: alpha=0.7
    result = blended_drift(sol, "stress", cfg)
    assert abs(result - (-0.15)) < 1e-12


def test_regime_multipliers_applied_correctly() -> None:
    """Same blended mu should scale by regime: calm=0.3x, stress=1.0x, shock=2.0x."""
    sol = _make_token(annualized_drift=-1.59)
    cfg = BlendingConfig()  # alpha=0.7 -> mu_blended = -0.15
    calm = blended_drift(sol, "calm", cfg)
    stress = blended_drift(sol, "stress", cfg)
    shock = blended_drift(sol, "shock", cfg)
    assert abs(calm - (-0.15 * 0.3)) < 1e-12
    assert abs(stress - (-0.15 * 1.0)) < 1e-12
    assert abs(shock - (-0.15 * 2.0)) < 1e-12


def test_unknown_regime_raises_keyerror() -> None:
    """Asking for a regime not in the multiplier table should fail loudly."""
    sol = _make_token()
    cfg = BlendingConfig()
    with pytest.raises(KeyError):
        blended_drift(sol, "rally", cfg)  # type: ignore[arg-type]


def test_alpha_validation() -> None:
    """alpha outside [0, 1] should fail at construction."""
    with pytest.raises(ValueError, match="alpha"):
        BlendingConfig(alpha=1.5)
    with pytest.raises(ValueError, match="alpha"):
        BlendingConfig(alpha=-0.1)


def test_negative_drift_cap_rejected() -> None:
    with pytest.raises(ValueError, match="drift_cap"):
        BlendingConfig(drift_cap=-0.1)


# -----------------------------------------------------------------------------
# Calibration loader — uses the real calibration.json from Step 3a
# -----------------------------------------------------------------------------


def test_calibration_loads_from_default_path() -> None:
    """The default-path Calibration() should find data/calibration.json."""
    cal = Calibration()
    assert cal.lookback_days == 90
    assert len(cal.tokens) >= 1


def test_calibration_has_expected_universe() -> None:
    """All 8 tokens from the calibration universe must be present and parseable."""
    cal = Calibration()
    expected = {"SOL", "USDC", "PYTH", "AERO", "JUP", "BRETT", "WIF", "BONK"}
    assert set(cal.symbols) == expected


def test_calibration_get_returns_correct_token() -> None:
    cal = Calibration()
    sol = cal.get("SOL")
    assert sol.symbol == "SOL"
    assert sol.chain == "solana"
    assert sol.is_stablecoin is False


def test_calibration_get_unknown_symbol_raises() -> None:
    cal = Calibration()
    with pytest.raises(KeyError, match="not in calibration universe"):
        cal.get("DOGE")  # not in our 8-token universe


def test_calibration_missing_file_raises() -> None:
    """Loading from a path that doesn't exist must raise a clear error."""
    with pytest.raises(FileNotFoundError, match="calibration.json not found"):
        Calibration(path=Path("/tmp/nonexistent_calibration_xyz.json"))


def test_usdc_is_flagged_as_stablecoin() -> None:
    """Sanity check: the calibration file correctly marks USDC."""
    cal = Calibration()
    usdc = cal.get("USDC")
    assert usdc.is_stablecoin is True


def test_full_pipeline_on_real_sol() -> None:
    """End-to-end: load real SOL calibration, run through blended_drift."""
    cal = Calibration()
    sol = cal.get("SOL")
    cfg = BlendingConfig()  # defaults

    calm = blended_drift(sol, "calm", cfg)
    stress = blended_drift(sol, "stress", cfg)
    shock = blended_drift(sol, "shock", cfg)

    # SOL realized drift was around -159%, well past the cap.
    # Expected mu_blended = 0.7*0 + 0.3*(-0.5) = -0.15
    # So calm = -0.045, stress = -0.15, shock = -0.30
    assert abs(calm - (-0.045)) < 1e-9
    assert abs(stress - (-0.15)) < 1e-9
    assert abs(shock - (-0.30)) < 1e-9

    # Monotonicity: |calm| < |stress| < |shock|
    assert abs(calm) < abs(stress) < abs(shock)
