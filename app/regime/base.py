"""
Abstract contracts for market regime detectors.

Per the RAMHD specification, the system must classify the current market
into one of three regimes (calm, stress, shock) so risk-adaptive routing
can adjust its behavior accordingly. Step 7.5 (risk-adaptive routing)
will consume the output of this module to bias token scoring.

Why the regime label matters:
- In CALM markets, RAMHD optimizes for cost — slippage and gas dominate.
- In STRESS markets, RAMHD biases toward stablecoins — reliability over cost.
- In SHOCK markets, RAMHD restricts to stablecoins — preserve principal.

We reuse the existing Regime type from market_data.calibration (where it's
defined for the simulator) so there's a single source of truth across the
codebase.

Confidence semantics:
A confidence of 1.0 means "deep inside this regime, far from any boundary."
A confidence of 0.0 means "right on the boundary, could go either way."
The risk-adaptive router (Step 7.5) can use confidence to soften decisions
near boundaries — e.g., a low-confidence "stress" classification might
warrant a lighter stablecoin bias than a high-confidence "stress."

References:
- Hamilton, "Time Series Analysis", ch. 22 — regime-switching models.
- Ang & Bekaert (2002) — regime classification in financial returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.market_data.base import TokenMarketData
from app.market_data.calibration import Regime


@dataclass(frozen=True)
class RegimeEstimate:
    """Regime classification for one token at the current moment.

    Attributes:
        symbol: token symbol
        regime: one of "calm", "stress", "shock"
        confidence: 0.0 (boundary) to 1.0 (deep inside regime)
        realized_volatility: measured per-step std dev (used as input)
        baseline_volatility: the calibrated reference this was compared against
        ratio: realized / baseline (for diagnostics — what drove the classification)

    Invariants enforced at construction:
        - regime is one of the valid values
        - confidence in [0, 1]
        - volatilities are non-negative
        - baseline > 0 (so ratio is defined)
    """

    symbol: str
    regime: Regime
    confidence: float
    realized_volatility: float
    baseline_volatility: float
    ratio: float

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if self.regime not in ("calm", "stress", "shock"):
            raise ValueError(
                f"regime must be 'calm', 'stress', or 'shock'; got {self.regime!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        if self.realized_volatility < 0:
            raise ValueError(
                f"realized_volatility must be non-negative, got {self.realized_volatility}"
            )
        if self.baseline_volatility <= 0:
            raise ValueError(
                f"baseline_volatility must be positive, got {self.baseline_volatility}"
            )
        if self.ratio < 0:
            raise ValueError(
                f"ratio must be non-negative, got {self.ratio}"
            )


class RegimeDetector(Protocol):
    """Protocol every regime detector must satisfy.

    The default implementation is ThresholdRegimeDetector using per-token
    thresholds derived from calibration. Future implementations might
    use HMMs, regime-switching GARCH, or hybrid approaches with engineered
    features (skewness, kurtosis, jump indicators).
    """

    def classify(self, data: TokenMarketData) -> RegimeEstimate:
        """Classify the current regime for one token.

        Args:
            data: market data for the token (path + liquidity context).
                  The detector reads recent realized volatility from
                  data.path and compares it against its calibrated baseline.

        Returns:
            RegimeEstimate with the classification, confidence, and
            diagnostic values.
        """
        ...
