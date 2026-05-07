"""
Threshold-based regime detector.

Computes realized rolling volatility from a token's recent price path,

compares it against the token's calibrated baseline volatility, and

classifies the result as calm / stress / shock based on the ratio.

Algorithm:

1. Compute log returns over the last `lookback_n` observations.

2. Compute realized per-step std-dev (sample std with Bessel correction).

3. Convert calibrated annualized_vol to per-step using the path's

   interval_seconds.

4. ratio = realized / baseline.

5. Classify:

       ratio < calm_max               -> calm

       calm_max <= ratio < shock_min  -> stress

       ratio >= shock_min             -> shock

6. Compute confidence based on distance from regime boundary.

Stablecoin handling:

Stablecoins (per Calibration.is_stablecoin) always classify as calm with

confidence 1.0. The regime concept doesn't apply to pegged assets — they

oscillate around the peg by design and have no meaningful "stress" mode

in the volatility-clustering sense.

Insufficient data handling:

If the path has fewer observations than the lookback window, we compute

on what's available but reduce confidence proportionally. Consumers

always get a usable classification, with confidence reflecting input

quality.

References:

- Per-token thresholds rationale: each token's "stress" looks different

  from another's. Universal thresholds would put BONK always in shock and

  USDC always in calm, providing no current-moment signal.

- The 1.8x shock threshold corresponds roughly to the 95th percentile

  of vol-to-baseline ratios in normal markets. Backtesting (Step 12)

  will validate or adjust.

"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.market_data.base import TokenMarketData
from app.market_data.calibration import Calibration, Regime
from app.regime.base import RegimeEstimate


SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


@dataclass(frozen=True)
class ThresholdConfig:
    """Tuning knobs for the threshold regime detector.

    Defaults reflect a conservative calibration:

    - 60-min lookback at 1-min bars gives ~60 samples (sample std relative

      noise ~13%), enough to be stable but reactive within the hour.

    - Calm/stress boundary at 1.0x baseline: any vol above normal triggers

      stress. Conservative — could be loosened to 1.2x if backtesting

      shows too-frequent stress classifications.

    - Stress/shock boundary at 1.8x baseline: roughly 95th-percentile of

      ratios in normal markets, appropriately rare for "shock" semantics.

    """

    lookback_n: int = 60

    """Number of past observations to use for realized vol computation."""

    calm_max_ratio: float = 1.0

    """Upper bound for calm regime (exclusive). ratio < this -> calm."""

    shock_min_ratio: float = 1.8

    """Lower bound for shock regime (inclusive). ratio >= this -> shock."""

    shock_full_confidence_ratio: float = 2.8

    """Ratio at which shock confidence saturates to 1.0. Beyond this, deeper

    in shock doesn't add confidence. Default 2.8 means width-of-1.0 above

    shock_min_ratio for confidence scaling."""

    min_observations: int = 20

    """Minimum observations needed for any classification at all. Below

    this, we still classify but confidence is heavily penalized."""

    def __post_init__(self) -> None:
        if self.lookback_n < 2:
            raise ValueError(f"lookback_n must be >= 2, got {self.lookback_n}")
        if self.calm_max_ratio <= 0:
            raise ValueError(
                f"calm_max_ratio must be positive, got {self.calm_max_ratio}"
            )
        if self.shock_min_ratio <= self.calm_max_ratio:
            raise ValueError(
                f"shock_min_ratio ({self.shock_min_ratio}) must be greater than "
                f"calm_max_ratio ({self.calm_max_ratio})"
            )
        if self.shock_full_confidence_ratio <= self.shock_min_ratio:
            raise ValueError(
                f"shock_full_confidence_ratio ({self.shock_full_confidence_ratio}) "
                f"must be greater than shock_min_ratio ({self.shock_min_ratio})"
            )
        if self.min_observations < 2:
            raise ValueError(
                f"min_observations must be >= 2, got {self.min_observations}"
            )


class ThresholdRegimeDetector:
    """Per-token threshold regime detector.

    Each token's classification compares its current realized vol against

    its own calibrated baseline. This means "stress for SOL" and "stress

    for BONK" are both meaningful current-moment signals, not just the

    long-run nature of each token.

    Usage:

        cal = Calibration()

        detector = ThresholdRegimeDetector(calibration=cal)

        estimate = detector.classify(token_market_data)

        if estimate.regime == "shock" and estimate.confidence > 0.7:

            # high-confidence shock — restrict to stablecoins

            ...

    """

    def __init__(
        self,
        calibration: Calibration,
        config: ThresholdConfig | None = None,
    ) -> None:
        self.calibration = calibration
        self.config = config if config is not None else ThresholdConfig()

    # -------------------------------------------------------------------
    # Public API: RegimeDetector protocol
    # -------------------------------------------------------------------

    def classify(self, data: TokenMarketData) -> RegimeEstimate:
        """Classify the current regime for one token."""
        token_cal = self.calibration.get(data.symbol)

        # Stablecoin bypass: always calm with full confidence.
        if token_cal.is_stablecoin:
            return self._stablecoin_estimate(data, token_cal.annualized_vol)

        # Compute baseline per-step volatility from calibration.
        # calibrated annualized_vol scales by sqrt(dt) to per-step.
        dt = data.path.interval_seconds / SECONDS_PER_YEAR
        baseline_per_step = token_cal.annualized_vol * math.sqrt(dt)

        # Compute realized per-step volatility from recent path.
        realized_per_step, n_used = self._compute_realized_volatility(data)

        # Edge case: zero baseline (shouldn't happen for non-stablecoins,
        # but defensive). Treat as calm with low confidence.
        if baseline_per_step <= 0:
            return RegimeEstimate(
                symbol=data.symbol,
                regime="calm",
                confidence=0.0,
                realized_volatility=realized_per_step,
                baseline_volatility=max(baseline_per_step, 1e-12),
                ratio=0.0,
            )

        ratio = realized_per_step / baseline_per_step

        regime = self._classify_ratio(ratio)
        confidence = self._compute_confidence(ratio, regime)

        # Reduce confidence if we used fewer than the configured lookback.
        data_quality_factor = min(1.0, n_used / self.config.lookback_n)
        confidence = confidence * data_quality_factor

        return RegimeEstimate(
            symbol=data.symbol,
            regime=regime,
            confidence=confidence,
            realized_volatility=realized_per_step,
            baseline_volatility=baseline_per_step,
            ratio=ratio,
        )

    # -------------------------------------------------------------------
    # Stablecoin special case
    # -------------------------------------------------------------------

    def _stablecoin_estimate(
        self, data: TokenMarketData, annualized_vol: float
    ) -> RegimeEstimate:
        """Stablecoins are always calm at full confidence."""
        dt = data.path.interval_seconds / SECONDS_PER_YEAR
        baseline_per_step = max(annualized_vol * math.sqrt(dt), 1e-12)
        realized_per_step, _ = self._compute_realized_volatility(data)
        ratio = realized_per_step / baseline_per_step if baseline_per_step > 0 else 0.0
        return RegimeEstimate(
            symbol=data.symbol,
            regime="calm",
            confidence=1.0,
            realized_volatility=realized_per_step,
            baseline_volatility=baseline_per_step,
            ratio=ratio,
        )

    # -------------------------------------------------------------------
    # Realized volatility from path
    # -------------------------------------------------------------------

    def _compute_realized_volatility(
        self, data: TokenMarketData
    ) -> tuple[float, int]:
        """Sample std-dev of log returns over the last lookback_n observations.

        Returns (realized_per_step_std, n_observations_used).

        If the path has fewer than min_observations, returns (0.0, n_available).

        """
        prices = data.path.prices_usd
        # We need lookback_n + 1 prices to get lookback_n returns. Take
        # whatever is available, capped at the configured lookback.
        max_returns_available = max(0, len(prices) - 1)
        n_returns = min(max_returns_available, self.config.lookback_n)

        if n_returns < self.config.min_observations:
            return 0.0, n_returns

        # Last n_returns + 1 prices, compute returns from those.
        recent_prices = prices[-(n_returns + 1) :]
        log_returns = np.diff(np.log(recent_prices))
        realized = float(np.std(log_returns, ddof=1))

        return realized, n_returns

    # -------------------------------------------------------------------
    # Classification + confidence
    # -------------------------------------------------------------------

    def _classify_ratio(self, ratio: float) -> Regime:
        if ratio < self.config.calm_max_ratio:
            return "calm"
        if ratio < self.config.shock_min_ratio:
            return "stress"
        return "shock"

    def _compute_confidence(self, ratio: float, regime: Regime) -> float:
        """Confidence based on distance from nearest regime boundary.

        - calm: width = calm_max_ratio. confidence = (calm_max - ratio) / calm_max.

        - stress: bounded on both sides. confidence = min(distances) / half_width.

        - shock: unbounded above. confidence = (ratio - shock_min) / (full_conf - shock_min), capped at 1.

        """
        cmax = self.config.calm_max_ratio
        smin = self.config.shock_min_ratio
        sfull = self.config.shock_full_confidence_ratio

        if regime == "calm":
            # ratio in [0, cmax). Distance from boundary = cmax - ratio.
            # Width of regime = cmax. Confidence = distance / width.
            return min(1.0, max(0.0, (cmax - ratio) / cmax))

        if regime == "stress":
            # ratio in [cmax, smin). Two boundaries.
            stress_width = smin - cmax
            half_width = stress_width / 2.0
            # Distance to nearer boundary.
            dist = min(ratio - cmax, smin - ratio)
            return min(1.0, max(0.0, dist / half_width))

        # shock: ratio >= smin. Confidence = how far past shock_min, capped.
        return min(1.0, max(0.0, (ratio - smin) / (sfull - smin)))
