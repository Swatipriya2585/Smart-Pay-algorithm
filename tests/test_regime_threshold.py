"""ThresholdRegimeDetector tests.

Two tolerance regimes:

- Functional invariants (regime label, confidence in [0,1], shape): tight.

- Statistical claims (realized vol matches expected): loose, ~15% relative.

"""

import math

import numpy as np
import pytest

from app.market_data.base import PricePath, TokenMarketData
from app.market_data.calibration import Calibration
from app.market_data.mock import MockConfig, MockMarketData
from app.regime.threshold import ThresholdConfig, ThresholdRegimeDetector


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _fetch(
    symbol: str,
    n_obs: int = 1440,
    seed: int = 7,
    regime: str = "calm",
) -> TokenMarketData:
    cfg = MockConfig(n_observations=n_obs, seed=seed, regime=regime)  # type: ignore[arg-type]
    mock = MockMarketData(config=cfg)
    return mock.fetch([symbol])[0]


def _detector() -> ThresholdRegimeDetector:
    return ThresholdRegimeDetector(calibration=Calibration())


# -----------------------------------------------------------------------------
# Functional invariants
# -----------------------------------------------------------------------------


def test_classify_returns_estimate_with_correct_symbol() -> None:
    d = _detector()
    out = d.classify(_fetch("SOL"))
    assert out.symbol == "SOL"


def test_classify_returns_valid_regime_label() -> None:
    d = _detector()
    out = d.classify(_fetch("SOL"))
    assert out.regime in ("calm", "stress", "shock")


def test_classify_confidence_in_unit_interval() -> None:
    d = _detector()
    for sym in ("SOL", "PYTH", "BONK", "BRETT", "WIF", "JUP", "AERO"):
        out = d.classify(_fetch(sym))
        assert 0.0 <= out.confidence <= 1.0, f"{sym}: confidence {out.confidence}"


def test_classify_returns_finite_volatilities() -> None:
    d = _detector()
    out = d.classify(_fetch("SOL"))
    assert math.isfinite(out.realized_volatility)
    assert math.isfinite(out.baseline_volatility)
    assert math.isfinite(out.ratio)
    assert out.realized_volatility >= 0
    assert out.baseline_volatility > 0


def test_classify_ratio_matches_realized_over_baseline() -> None:
    d = _detector()
    out = d.classify(_fetch("SOL"))
    expected_ratio = out.realized_volatility / out.baseline_volatility
    assert abs(out.ratio - expected_ratio) < 1e-12


# -----------------------------------------------------------------------------
# Stablecoin handling
# -----------------------------------------------------------------------------


def test_stablecoin_always_classifies_as_calm() -> None:
    d = _detector()
    out = d.classify(_fetch("USDC"))
    assert out.regime == "calm"


def test_stablecoin_classification_confidence_is_one() -> None:
    d = _detector()
    out = d.classify(_fetch("USDC"))
    assert out.confidence == 1.0


def test_stablecoin_in_any_regime_still_calm() -> None:
    """USDC in shock regime should still classify as calm — peg dynamics override."""
    d = _detector()
    out = d.classify(_fetch("USDC", regime="shock"))
    assert out.regime == "calm"
    assert out.confidence == 1.0


# -----------------------------------------------------------------------------
# Cross-regime sensitivity (mock-driven)
# -----------------------------------------------------------------------------


def test_calm_simulation_produces_calm_or_stress_label() -> None:
    """A token simulated under 'calm' regime should not be classified as shock."""
    d = _detector()
    out = d.classify(_fetch("SOL", regime="calm"))
    assert out.regime in ("calm", "stress")


# -----------------------------------------------------------------------------
# Threshold boundary behavior
# -----------------------------------------------------------------------------


def test_classify_ratio_below_calm_boundary_is_calm() -> None:
    """Direct method test: ratio of 0.5 with default thresholds -> calm."""
    cfg = ThresholdConfig()
    d = ThresholdRegimeDetector(calibration=Calibration(), config=cfg)
    assert d._classify_ratio(0.5) == "calm"


def test_classify_ratio_at_calm_boundary_is_stress() -> None:
    """ratio = 1.0 with calm_max=1.0 -> stress (boundary is exclusive for calm)."""
    cfg = ThresholdConfig()
    d = ThresholdRegimeDetector(calibration=Calibration(), config=cfg)
    assert d._classify_ratio(1.0) == "stress"


def test_classify_ratio_at_shock_boundary_is_shock() -> None:
    """ratio = 1.8 with shock_min=1.8 -> shock (boundary inclusive for shock)."""
    cfg = ThresholdConfig()
    d = ThresholdRegimeDetector(calibration=Calibration(), config=cfg)
    assert d._classify_ratio(1.8) == "shock"


def test_classify_ratio_in_stress_band_is_stress() -> None:
    cfg = ThresholdConfig()
    d = ThresholdRegimeDetector(calibration=Calibration(), config=cfg)
    assert d._classify_ratio(1.4) == "stress"


def test_classify_ratio_above_shock_is_shock() -> None:
    cfg = ThresholdConfig()
    d = ThresholdRegimeDetector(calibration=Calibration(), config=cfg)
    assert d._classify_ratio(5.0) == "shock"


# -----------------------------------------------------------------------------
# Confidence semantics (hand-computed)
# -----------------------------------------------------------------------------


def test_confidence_at_zero_ratio_is_one_for_calm() -> None:
    """ratio=0 is as deep into calm as you can get."""
    cfg = ThresholdConfig()
    d = ThresholdRegimeDetector(calibration=Calibration(), config=cfg)
    # ratio=0, regime=calm: (1.0 - 0) / 1.0 = 1.0
    assert abs(d._compute_confidence(0.0, "calm") - 1.0) < 1e-12


def test_confidence_just_below_calm_boundary_is_low() -> None:
    """ratio=0.95 with calm_max=1.0: (1.0 - 0.95)/1.0 = 0.05 — barely calm."""
    cfg = ThresholdConfig()
    d = ThresholdRegimeDetector(calibration=Calibration(), config=cfg)
    assert abs(d._compute_confidence(0.95, "calm") - 0.05) < 1e-12


def test_confidence_at_stress_midpoint_is_one() -> None:
    """Midpoint of stress band (1.4 with bounds 1.0/1.8) — equidistant from
    both boundaries — confidence should be 1.0 (max stress confidence)."""
    cfg = ThresholdConfig()
    d = ThresholdRegimeDetector(calibration=Calibration(), config=cfg)
    # Distance from each boundary: 0.4. Half-width: 0.4. Confidence = 0.4/0.4 = 1.0
    assert abs(d._compute_confidence(1.4, "stress") - 1.0) < 1e-12


def test_confidence_just_inside_stress_boundary_is_low() -> None:
    """ratio=1.05, just past calm boundary. Distance to calm boundary: 0.05."""
    cfg = ThresholdConfig()
    d = ThresholdRegimeDetector(calibration=Calibration(), config=cfg)
    # half_width = 0.4. dist = min(0.05, 0.75) = 0.05. confidence = 0.05/0.4 = 0.125
    assert abs(d._compute_confidence(1.05, "stress") - 0.125) < 1e-12


def test_confidence_just_into_shock_is_low() -> None:
    """ratio=1.85, just past shock boundary."""
    cfg = ThresholdConfig()
    d = ThresholdRegimeDetector(calibration=Calibration(), config=cfg)
    # (1.85 - 1.8) / (2.8 - 1.8) = 0.05 / 1.0 = 0.05
    assert abs(d._compute_confidence(1.85, "shock") - 0.05) < 1e-12


def test_confidence_deep_in_shock_is_one() -> None:
    """Beyond shock_full_confidence_ratio, confidence saturates at 1.0."""
    cfg = ThresholdConfig()
    d = ThresholdRegimeDetector(calibration=Calibration(), config=cfg)
    assert d._compute_confidence(5.0, "shock") == 1.0


# -----------------------------------------------------------------------------
# Insufficient data handling
# -----------------------------------------------------------------------------


def test_short_path_reduces_confidence() -> None:
    """A path shorter than lookback_n produces classification with reduced confidence."""
    d = ThresholdRegimeDetector(
        calibration=Calibration(),
        config=ThresholdConfig(lookback_n=120, min_observations=20),
    )
    short_data = _fetch("SOL", n_obs=50)  # 49 returns < 120 lookback
    out = d.classify(short_data)
    # Data quality factor = 49/120 ≈ 0.408. Confidence is scaled down by this.
    assert out.confidence <= 0.5  # significantly reduced


def test_path_at_min_observations_classifies() -> None:
    """At exactly min_observations, we still get a classification (not error)."""
    d = ThresholdRegimeDetector(
        calibration=Calibration(),
        config=ThresholdConfig(lookback_n=60, min_observations=20),
    )
    # 21 prices = 20 returns = min_observations
    data = _fetch("SOL", n_obs=21)
    out = d.classify(data)
    # Should not error; regime label should be valid.
    assert out.regime in ("calm", "stress", "shock")


# -----------------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------------


def test_config_rejects_too_small_lookback() -> None:
    with pytest.raises(ValueError, match="lookback_n"):
        ThresholdConfig(lookback_n=1)


def test_config_rejects_negative_calm_max() -> None:
    with pytest.raises(ValueError, match="calm_max_ratio"):
        ThresholdConfig(calm_max_ratio=-0.5)


def test_config_rejects_shock_min_below_calm_max() -> None:
    with pytest.raises(ValueError, match="shock_min_ratio"):
        ThresholdConfig(calm_max_ratio=1.5, shock_min_ratio=1.0)


def test_config_rejects_shock_full_confidence_below_shock_min() -> None:
    with pytest.raises(ValueError, match="shock_full_confidence_ratio"):
        ThresholdConfig(shock_min_ratio=1.8, shock_full_confidence_ratio=1.5)


def test_config_rejects_too_small_min_observations() -> None:
    with pytest.raises(ValueError, match="min_observations"):
        ThresholdConfig(min_observations=1)


# -----------------------------------------------------------------------------
# End-to-end smoke
# -----------------------------------------------------------------------------


def test_full_universe_classify_smoke() -> None:
    """All 8 calibrated tokens should classify cleanly without errors."""
    d = _detector()
    universe = ["SOL", "USDC", "PYTH", "AERO", "JUP", "BRETT", "WIF", "BONK"]
    for sym in universe:
        out = d.classify(_fetch(sym))
        assert out.symbol == sym
        assert out.regime in ("calm", "stress", "shock")
        assert 0.0 <= out.confidence <= 1.0


def test_unknown_symbol_raises_keyerror() -> None:
    """A symbol not in the calibration universe should fail loud."""
    d = _detector()
    # Build a TokenMarketData with an unknown symbol.
    fake_data = TokenMarketData(
        symbol="DOGE",  # not in our 8-token calibration
        mint="fake",
        path=PricePath(
            symbol="DOGE",
            prices_usd=np.array([1.0] * 100),
            interval_seconds=60.0,
        ),
        liquidity_depth_usd=1_000_000.0,
        spread_bps=5.0,
    )
    with pytest.raises(KeyError):
        d.classify(fake_data)
