"""
Calibration loader and blending math.

Reads ramhd-service/data/calibration.json (produced by scripts.recalibrate)
and applies the shrinkage-blending formula to convert short-window realized
drift into a stable simulator parameter:

    mu_baseline       = config.baseline_drift  (default 0%)
    mu_realized_clip  = clip(token.annualized_drift, -cap, +cap)
    mu_blended        = alpha * mu_baseline + (1 - alpha) * mu_realized_clip
    mu_for_regime     = mu_blended * regime_multiplier

where regime_multiplier is one of:
    calm   = 0.3   (gentle drift, low-vol regime)
    stress = 1.0   (full calibrated drift, baseline regime)
    shock  = 2.0   (amplified drift, crisis regime)

Stablecoins (is_stablecoin=True) bypass this entirely — they always get
mu = 0 regardless of regime, because GBM with non-zero drift is wrong
for pegged assets.

References:
- Shrinkage estimators: Stein 1956; James-Stein 1961
- Black-Litterman portfolio model: Black & Litterman 1992
- Hull, "Options, Futures, and Other Derivatives", ch. 14 (GBM calibration)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Three named regimes the simulator can produce.
Regime = Literal["calm", "stress", "shock"]


# -----------------------------------------------------------------------------
# Schema for calibration.json (validates on load)
# -----------------------------------------------------------------------------


class TokenCalibration(BaseModel):
    """One token's calibration record from calibration.json."""

    symbol: str
    coingecko_id: str
    chain: str
    address: str
    regime: str
    is_stablecoin: bool
    n_observations: int
    current_price_usd: float
    min_price_usd: float
    max_price_usd: float
    daily_log_return_mean: float
    daily_log_return_std: float
    annualized_drift: float
    annualized_vol: float


class CalibrationFile(BaseModel):
    """Top-level shape of calibration.json."""

    schema_version: int
    generated_at_utc: str
    source: str
    lookback_days: int
    universe_size: int
    tokens: list[TokenCalibration] = Field(..., min_length=1)


# -----------------------------------------------------------------------------
# Blending configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BlendingConfig:
    """Tuning knobs for the drift-blending formula.

    Defaults reflect the product-owner decision: lean toward baseline (alpha=0.7),
    cap realized drift at +/- 50% annualized, three named regimes with
    monotone-increasing multipliers.

    Override any field per-test or per-environment without subclassing.
    """

    alpha: float = 0.7
    """Blending weight on baseline. 0 = trust realized fully, 1 = trust baseline fully."""

    drift_cap: float = 0.5
    """Symmetric cap on realized annualized drift (decimal, not %)."""

    baseline_drift: float = 0.0
    """Long-run drift assumption for non-stablecoins. Conservative default = 0%."""

    regime_multipliers: dict[Regime, float] = field(
        default_factory=lambda: {"calm": 0.3, "stress": 1.0, "shock": 2.0}
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.drift_cap < 0:
            raise ValueError(f"drift_cap must be non-negative, got {self.drift_cap}")
        for regime, mult in self.regime_multipliers.items():
            if mult < 0:
                raise ValueError(f"regime_multipliers[{regime}] must be non-negative, got {mult}")


# -----------------------------------------------------------------------------
# Blending math
# -----------------------------------------------------------------------------


def blended_drift(
    token: TokenCalibration,
    regime: Regime,
    config: BlendingConfig | None = None,
) -> float:
    """Compute the simulator's drift parameter for one token in one regime.

    Implements the shrinkage formula:
        mu = (alpha * baseline + (1 - alpha) * clip(realized, +/- cap)) * multiplier

    Stablecoins bypass entirely and return 0.0 regardless of regime —
    a pegged asset has no meaningful drift, and GBM with mu != 0 would
    produce paths that walk away from the peg.

    Args:
        token: a TokenCalibration record loaded from calibration.json
        regime: one of "calm", "stress", "shock"
        config: blending parameters; defaults to BlendingConfig()

    Returns:
        Annualized drift in decimal form (e.g. -0.15 for -15% per year)
        suitable for direct use as the mu parameter in GBM simulation.

    Raises:
        KeyError: if regime is not in config.regime_multipliers.
    """
    cfg = config if config is not None else BlendingConfig()

    # Stablecoin bypass.
    if token.is_stablecoin:
        return 0.0

    # Cap realized drift to a sensible band.
    cap = cfg.drift_cap
    realized_clipped = max(-cap, min(cap, token.annualized_drift))

    # Blend baseline and clipped realized.
    mu_blended = cfg.alpha * cfg.baseline_drift + (1.0 - cfg.alpha) * realized_clipped

    # Apply regime multiplier.
    multiplier = cfg.regime_multipliers[regime]
    return mu_blended * multiplier


# -----------------------------------------------------------------------------
# Calibration class — loads and caches the JSON
# -----------------------------------------------------------------------------


class Calibration:
    """Loads calibration.json and provides token lookup.

    Caches the parsed result so repeated lookups are free. Re-instantiate
    if you need to pick up a fresh calibration file (typically only after
    re-running scripts.recalibrate).
    """

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            # Default: ramhd-service/data/calibration.json relative to this file.
            here = Path(__file__).resolve()
            path = here.parent.parent.parent / "data" / "calibration.json"
        self.path = Path(path)
        self._file = self._load()
        self._by_symbol: dict[str, TokenCalibration] = {
            t.symbol: t for t in self._file.tokens
        }

    def _load(self) -> CalibrationFile:
        if not self.path.exists():
            raise FileNotFoundError(
                f"calibration.json not found at {self.path}. "
                f"Run: python -m scripts.recalibrate"
            )
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return CalibrationFile(**raw)

    @property
    def tokens(self) -> list[TokenCalibration]:
        """All calibrated tokens, in original order."""
        return list(self._file.tokens)

    @property
    def symbols(self) -> list[str]:
        """List of symbols available in this calibration."""
        return [t.symbol for t in self._file.tokens]

    @property
    def lookback_days(self) -> int:
        return self._file.lookback_days

    @property
    def generated_at_utc(self) -> str:
        return self._file.generated_at_utc

    def get(self, symbol: str) -> TokenCalibration:
        """Look up calibration for one symbol. Raises KeyError if not found."""
        try:
            return self._by_symbol[symbol]
        except KeyError as e:
            raise KeyError(
                f"{symbol} not in calibration universe. "
                f"Available: {sorted(self._by_symbol)}"
            ) from e

    def has(self, symbol: str) -> bool:
        return symbol in self._by_symbol
