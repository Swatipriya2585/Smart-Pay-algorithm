"""
GARCH(1,1) forecaster for RAMHD multi-horizon volatility prediction.

GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) is the
industry-standard model for short-term volatility forecasting in financial
markets. It captures *volatility clustering* — the empirical fact that
high-volatility periods tend to be followed by more high-volatility, and
calm periods by more calm. This pattern is dominant in crypto on the
seconds-to-minutes timescales RAMHD operates at.

Model specification:

    r_t = mu + epsilon_t

    epsilon_t = sigma_t * z_t,   z_t ~ N(0, 1)

    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

We fit on log-returns of the price path, then produce h-step-ahead variance
forecasts using the analytic recursion:

    E[sigma_{t+h}^2 | F_t] = omega/(1 - alpha - beta) +
                              (alpha + beta)^h * (sigma_t^2 - omega/(1 - alpha - beta))

For multi-period horizons (e.g., a 120s forecast from 1-second steps), we
sum the per-step variance forecasts — this is exact for GARCH because the
process is conditionally Gaussian.

Stablecoin handling:

    Stablecoins have near-zero, near-flat realized volatility. GARCH can
    technically fit them but the parameters become near-degenerate and the
    fit adds no information. We short-circuit: if calibration says the
    token is a stablecoin (or realized vol falls below a threshold), we
    return forecasts with the realized per-step sigma scaled to each
    horizon and confidence bounds straight off that.

References:

    - Bollerslev (1986) — original GARCH paper.
    - Engle (1982) — ARCH foundation.
    - Hamilton, "Time Series Analysis", ch. 21 (multi-step GARCH forecasts).
    - arch library: https://arch.readthedocs.io
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from scipy.stats import norm

from app.forecasting.base import (
    DEFAULT_HORIZONS,
    HorizonForecast,
    MultiHorizonForecast,
)
from app.market_data.base import TokenMarketData
from app.market_data.calibration import Calibration


@dataclass(frozen=True)
class GARCHConfig:
    """Tuning knobs for GARCH fitting and stablecoin detection."""

    # Stablecoin detection threshold: per-step (e.g., per-minute) realized
    # std-dev below this short-circuits to the stablecoin path.
    # 1% annualized ~= 0.01 / sqrt(525600 mins/yr) ~= 1.4e-5 per minute.
    # Set generously: anything tighter than 5% annualized counts as stable.
    stablecoin_per_step_threshold: float = 5e-5

    # Confidence level for the prediction band (95% by default).
    confidence_level: float = 0.95

    # Suppress non-fatal arch convergence warnings — they're noisy and
    # don't indicate a fitting failure for our use case.
    suppress_warnings: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )
        if self.stablecoin_per_step_threshold < 0:
            raise ValueError(
                "stablecoin_per_step_threshold must be non-negative"
            )


class GARCHForecaster:
    """Multi-horizon volatility forecaster using GARCH(1,1).

    Usage:

        cal = Calibration()
        forecaster = GARCHForecaster(calibration=cal)
        mhf = forecaster.forecast(token_market_data)
        f30 = mhf.at(30.0)  # 30-second forecast

    Stateless: each forecast() call refits GARCH on the supplied path. This
    costs ~50ms per token and is the right trade-off for our use case
    (payment recommendations, not high-frequency trading).

    """

    def __init__(
        self,
        calibration: Optional[Calibration] = None,
        config: Optional[GARCHConfig] = None,
    ) -> None:
        self.calibration = calibration
        self.config = config if config is not None else GARCHConfig()

    # -------------------------------------------------------------------
    # Public API: Forecaster protocol
    # -------------------------------------------------------------------

    def forecast(
        self,
        data: TokenMarketData,
        horizons: tuple[float, ...] = DEFAULT_HORIZONS,
    ) -> MultiHorizonForecast:
        """Produce multi-horizon GARCH forecasts for one token."""
        if not horizons:
            raise ValueError("horizons must contain at least one entry")

        prices = data.path.prices_usd
        interval = data.path.interval_seconds

        # Compute log returns. We need at least 50 observations for a
        # reasonable GARCH fit; 1440 (24h of 1-min bars) is what MockMarketData
        # produces by default, well above the minimum.
        log_returns = np.diff(np.log(prices))
        if len(log_returns) < 50:
            raise ValueError(
                f"GARCH needs >= 50 observations, got {len(log_returns)}"
            )

        # Decide stablecoin vs full GARCH path.
        if self._is_stablecoin(data.symbol, log_returns):
            forecasts = self._forecast_stablecoin(data.symbol, log_returns, interval, horizons)
        else:
            forecasts = self._forecast_garch(data.symbol, log_returns, interval, horizons)

        return MultiHorizonForecast(symbol=data.symbol, horizons=forecasts)

    # -------------------------------------------------------------------
    # Stablecoin detection
    # -------------------------------------------------------------------

    def _is_stablecoin(self, symbol: str, log_returns: np.ndarray) -> bool:
        """Decide if this token should bypass GARCH.

        Priority: calibration flag wins. Fallback: per-step realized vol
        below the threshold.
        """
        if self.calibration is not None and self.calibration.has(symbol):
            return self.calibration.get(symbol).is_stablecoin

        per_step_std = float(np.std(log_returns, ddof=1))
        return per_step_std < self.config.stablecoin_per_step_threshold

    # -------------------------------------------------------------------
    # GARCH fit + forecast (non-stablecoin path)
    # -------------------------------------------------------------------

    def _forecast_garch(
        self,
        symbol: str,
        log_returns: np.ndarray,
        interval_seconds: float,
        horizons: tuple[float, ...],
    ) -> dict[float, HorizonForecast]:
        """Fit GARCH(1,1) and produce variance forecasts at each horizon."""
        # arch expects returns in percentage terms for numerical stability.
        # Scale by 100, fit, descale at the end.
        scaled = log_returns * 100.0

        with warnings.catch_warnings():
            if self.config.suppress_warnings:
                warnings.simplefilter("ignore")
            model = arch_model(
                scaled,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="normal",
                rescale=False,
            )
            fit: ARCHModelResult = model.fit(disp="off", show_warning=False)

        mu_per_step_scaled = float(fit.params["mu"])
        # Rescale mean back to log-return space.
        mu_per_step = mu_per_step_scaled / 100.0

        # Compute the maximum number of forward steps we need (in path-step units).
        max_horizon = max(horizons)
        max_steps = max(1, int(math.ceil(max_horizon / interval_seconds)))

        # Forecast variance at each step ahead.
        # arch returns variance in scaled-units squared, i.e., (%-return)^2.
        # We divide by 100^2 to get back to (log-return)^2 = log-return variance.
        forecasts = fit.forecast(horizon=max_steps, reindex=False)
        var_path_scaled = np.asarray(forecasts.variance.values).flatten()
        var_per_step = var_path_scaled / (100.0**2)

        return self._aggregate_variance_to_horizons(
            mu_per_step=mu_per_step,
            var_per_step=var_per_step,
            interval_seconds=interval_seconds,
            horizons=horizons,
        )

    # -------------------------------------------------------------------
    # Stablecoin path — use realized vol directly
    # -------------------------------------------------------------------

    def _forecast_stablecoin(
        self,
        symbol: str,
        log_returns: np.ndarray,
        interval_seconds: float,
        horizons: tuple[float, ...],
    ) -> dict[float, HorizonForecast]:
        """For stablecoins, use realized per-step variance directly.

        Stablecoins don't have meaningful volatility clustering — they
        oscillate around the peg. The realized per-step std is a better
        forecast than fitting a degenerate GARCH model.
        """
        # Mean is essentially zero for a stablecoin path; use the empirical
        # mean to be honest, but expect it to be tiny.
        mu_per_step = float(np.mean(log_returns))
        # Constant per-step variance (no clustering).
        per_step_var = float(np.var(log_returns, ddof=1))
        var_per_step = np.full(
            shape=int(max(1, math.ceil(max(horizons) / interval_seconds))),
            fill_value=per_step_var,
        )

        return self._aggregate_variance_to_horizons(
            mu_per_step=mu_per_step,
            var_per_step=var_per_step,
            interval_seconds=interval_seconds,
            horizons=horizons,
        )

    # -------------------------------------------------------------------
    # Variance aggregation across multiple steps
    # -------------------------------------------------------------------

    def _aggregate_variance_to_horizons(
        self,
        mu_per_step: float,
        var_per_step: np.ndarray,
        interval_seconds: float,
        horizons: tuple[float, ...],
    ) -> dict[float, HorizonForecast]:
        """Aggregate per-step variance to arbitrary horizons via fractional-step interpolation.

        For sub-step horizons (e.g., 5s when interval_seconds=60) we cannot
        observe finer resolution than the path's sampling rate, so we
        fractionally scale the first step's variance: Var[r over h seconds]
        = (h / interval) * var_per_step[0]. This is exact for any Gaussian
        i.i.d. or GARCH process under the standard variance-scaling rule.

        For multi-step horizons that fall between integer step boundaries,
        we linearly interpolate the cumulative variance series.

        For Gaussian (or conditionally Gaussian) processes:

            Var[r_{t+1} + r_{t+2} + ... + r_{t+k}] = sum_{i=1..k} Var[r_{t+i}]
            E[r_{t+1} + ... + r_{t+k}] = k * mu

        which is exact for GARCH conditional on F_t. We extend this to
        non-integer k by interpolation.
        """
        cum_var = np.cumsum(var_per_step)
        max_total_seconds = len(cum_var) * interval_seconds

        z = float(norm.ppf(0.5 + self.config.confidence_level / 2.0))

        out: dict[float, HorizonForecast] = {}
        for h_seconds in horizons:
            # Fractional number of steps spanned by this horizon.
            steps_float = h_seconds / interval_seconds

            # Defensive clamp for callers asking beyond what we forecasted.
            if h_seconds >= max_total_seconds:
                horizon_var = float(cum_var[-1]) * (h_seconds / max_total_seconds)

            elif steps_float <= 1.0:
                # Sub-step horizon: scale the first step's variance fractionally.
                horizon_var = float(var_per_step[0]) * steps_float

            else:
                # Between integer steps: linearly interpolate cumulative variance.
                # cum_var[k-1] corresponds to k full steps. We want the value
                # at steps_float, which lies between floor(steps_float) and
                # ceil(steps_float).
                lower_idx = int(math.floor(steps_float)) - 1  # 0-indexed
                upper_idx = lower_idx + 1
                lower_var = float(cum_var[lower_idx])
                upper_var = float(cum_var[upper_idx])
                fraction = steps_float - math.floor(steps_float)
                horizon_var = lower_var + fraction * (upper_var - lower_var)

            horizon_std = math.sqrt(max(horizon_var, 0.0))
            horizon_mean = mu_per_step * steps_float

            out[h_seconds] = HorizonForecast(
                horizon_seconds=h_seconds,
                predicted_return=horizon_mean,
                predicted_volatility=horizon_std,
                confidence_lower_95=horizon_mean - z * horizon_std,
                confidence_upper_95=horizon_mean + z * horizon_std,
            )

        return out
