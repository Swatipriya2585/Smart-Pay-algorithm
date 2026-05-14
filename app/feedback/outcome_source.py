"""OutcomeSource Protocol and a synthetic MockOutcomeSource for offline use."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from app.bandit.pipeline import BanditDecision
from app.feedback.contracts import RealizedOutcome, TradeStatus
from app.forecasting.base import MultiHorizonForecast

logger = logging.getLogger(__name__)


@runtime_checkable
class OutcomeSource(Protocol):
    """Where realized outcomes come from.

    v1 implementation is :class:`MockOutcomeSource`. Future implementations
    (e.g. HttpOutcomeSource, PostgresOutcomeSource) will wrap the real
    Solana executor or the monorepo's RDS without changing the bandit.
    """

    def fetch_outcome(self, tx_id: str) -> Optional[RealizedOutcome]:
        """Return the realized outcome for this transaction.

        Returns ``None`` if the outcome is not yet available (caller should
        retry later). For executed-but-unobservable trades return a
        :class:`RealizedOutcome` with status ``DATA_MISSING`` instead.
        """
        ...


@dataclass
class MockOutcomeSource:
    """Generates synthetic outcomes for offline / pre-execution-layer use.

    Strategy: the forecaster's 120s predicted return is the mean for FILLED
    outcomes; configurable Gaussian noise is added. Failures and timeouts
    are sampled at configurable rates. Deterministic given the seed.

    ``forecasts_by_tx[tx_id]`` is populated at decision time by the pipeline
    (Step 10.3) via :meth:`register_decision`. Each value is a triple of
    ``(decision, forecast_for_chosen_symbol, amount_usd)``.
    """

    forecasts_by_tx: dict[str, tuple[BanditDecision, MultiHorizonForecast, float]] = field(
        default_factory=dict
    )

    noise_std: float = 0.005
    failure_rate: float = 0.02
    timeout_rate: float = 0.01
    cost_dollar_per_trade: float = -50.0
    failure_cost_dollar: float = -50.0
    timeout_cost_dollar: float = -5.0
    """Cost charged for TIMEOUT: lower magnitude than ``failure_cost_dollar``
    because in practice the trade may not have been submitted at all (e.g.
    pre-submission queueing timed out); only minimal RPC/gas spend is sunk."""

    rng_seed: int = 42

    def __post_init__(self) -> None:
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {self.noise_std}")
        if not 0.0 <= self.failure_rate <= 1.0:
            raise ValueError(
                f"failure_rate must be in [0, 1], got {self.failure_rate}"
            )
        if not 0.0 <= self.timeout_rate <= 1.0:
            raise ValueError(
                f"timeout_rate must be in [0, 1], got {self.timeout_rate}"
            )
        if self.failure_rate + self.timeout_rate > 1.0:
            raise ValueError("failure_rate + timeout_rate must be <= 1.0")
        if self.cost_dollar_per_trade > 1e-9:
            raise ValueError(
                "cost_dollar_per_trade must be non-positive, "
                f"got {self.cost_dollar_per_trade}"
            )
        if self.failure_cost_dollar > 1e-9:
            raise ValueError(
                "failure_cost_dollar must be non-positive, "
                f"got {self.failure_cost_dollar}"
            )
        if self.timeout_cost_dollar > 1e-9:
            raise ValueError(
                "timeout_cost_dollar must be non-positive, "
                f"got {self.timeout_cost_dollar}"
            )
        self._rng = np.random.default_rng(self.rng_seed)

    def fetch_outcome(self, tx_id: str) -> Optional[RealizedOutcome]:
        """Generate a synthetic outcome for ``tx_id`` or ``None`` if unknown."""
        if tx_id not in self.forecasts_by_tx:
            return None
        _decision, forecast, _amount_usd = self.forecasts_by_tx[tx_id]

        r = float(self._rng.random())
        if r < self.failure_rate:
            return self._make_failure(tx_id)
        if r < self.failure_rate + self.timeout_rate:
            return self._make_timeout(tx_id)

        mu = forecast.at(120.0).predicted_return
        noise = float(self._rng.normal(0.0, self.noise_std))
        realized_return = mu + noise
        return RealizedOutcome(
            tx_id=tx_id,
            status=TradeStatus.FILLED,
            realized_return=realized_return,
            realized_cost_dollar=self.cost_dollar_per_trade,
            fill_fraction=1.0,
            observed_at_utc=_now_iso(),
        )

    def _make_failure(self, tx_id: str) -> RealizedOutcome:
        # Trade was submitted and rejected pre-settlement; gas/fees still
        # incurred. Zero realized return because no fill happened.
        return RealizedOutcome(
            tx_id=tx_id,
            status=TradeStatus.FAILED,
            realized_return=0.0,
            realized_cost_dollar=self.failure_cost_dollar,
            fill_fraction=0.0,
            observed_at_utc=_now_iso(),
        )

    def _make_timeout(self, tx_id: str) -> RealizedOutcome:
        # Trade did not settle within the reward window. Treat costs as
        # lower than full-fail: in practice many timeouts never make it
        # past the submission queue, so only minimal RPC/gas was sunk.
        return RealizedOutcome(
            tx_id=tx_id,
            status=TradeStatus.TIMEOUT,
            realized_return=0.0,
            realized_cost_dollar=self.timeout_cost_dollar,
            fill_fraction=0.0,
            observed_at_utc=_now_iso(),
        )

    def register_decision(
        self,
        tx_id: str,
        decision: BanditDecision,
        forecast_for_chosen: MultiHorizonForecast,
        amount_usd: float,
    ) -> None:
        """Pipeline hook: tell the source what was decided for ``tx_id``."""
        self.forecasts_by_tx[tx_id] = (decision, forecast_for_chosen, amount_usd)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
