"""Tests for OutcomeSource Protocol and MockOutcomeSource."""

from __future__ import annotations

import numpy as np
import pytest

from app.bandit.contracts import FEATURE_DIM
from app.bandit.pipeline import BanditDecision
from app.feedback.contracts import TradeStatus
from app.feedback.outcome_source import MockOutcomeSource, OutcomeSource
from app.forecasting.base import HorizonForecast, MultiHorizonForecast


def _decision(symbol: str = "SOL") -> BanditDecision:
    x = np.zeros(FEATURE_DIM, dtype=np.float64)
    x[-1] = 1.0
    return BanditDecision(
        chosen_symbol=symbol,
        chosen_feature_vector=x,
        ucb_scores={symbol: 1.0},
        feature_vectors={symbol: x},
        candidates_evaluated=(symbol,),
        decision_utc="2026-05-13T00:00:00+00:00",
    )


def _forecast(symbol: str = "SOL", mu: float = 0.005) -> MultiHorizonForecast:
    horizons = {}
    for h in (5.0, 30.0, 120.0):
        horizons[h] = HorizonForecast(
            horizon_seconds=h,
            predicted_return=mu,
            predicted_volatility=0.001,
            confidence_lower_95=mu - 0.01,
            confidence_upper_95=mu + 0.01,
        )
    return MultiHorizonForecast(symbol=symbol, horizons=horizons)


# -----------------------------------------------------------------------------
# Configuration validation
# -----------------------------------------------------------------------------


def test_constructs_with_defaults() -> None:
    src = MockOutcomeSource()
    assert src.forecasts_by_tx == {}
    assert src.noise_std == 0.005
    assert src.failure_rate == 0.02
    assert src.timeout_rate == 0.01


def test_negative_noise_std_raises() -> None:
    with pytest.raises(ValueError, match="noise_std"):
        MockOutcomeSource(noise_std=-0.001)


def test_failure_plus_timeout_over_one_raises() -> None:
    with pytest.raises(ValueError, match="failure_rate \\+ timeout_rate"):
        MockOutcomeSource(failure_rate=0.6, timeout_rate=0.5)


def test_positive_cost_per_trade_raises() -> None:
    with pytest.raises(ValueError, match="cost_dollar_per_trade"):
        MockOutcomeSource(cost_dollar_per_trade=10.0)


# -----------------------------------------------------------------------------
# fetch_outcome behavior
# -----------------------------------------------------------------------------


def test_fetch_unknown_tx_returns_none() -> None:
    src = MockOutcomeSource()
    assert src.fetch_outcome("never-seen") is None


def test_filled_outcome_for_registered_tx() -> None:
    src = MockOutcomeSource(
        noise_std=0.0,
        failure_rate=0.0,
        timeout_rate=0.0,
        cost_dollar_per_trade=-25.0,
        rng_seed=1,
    )
    src.register_decision(
        "tx-1",
        _decision("SOL"),
        _forecast("SOL", mu=0.005),
        amount_usd=1000.0,
    )
    out = src.fetch_outcome("tx-1")
    assert out is not None
    assert out.status == TradeStatus.FILLED
    assert out.realized_return == pytest.approx(0.005, abs=1e-9)
    assert out.fill_fraction == 1.0
    assert out.realized_cost_dollar == -25.0


def test_gaussian_noise_with_seed_is_deterministic() -> None:
    a = MockOutcomeSource(noise_std=0.01, failure_rate=0.0, timeout_rate=0.0, rng_seed=7)
    b = MockOutcomeSource(noise_std=0.01, failure_rate=0.0, timeout_rate=0.0, rng_seed=7)
    a.register_decision("tx", _decision(), _forecast(mu=0.005), 1000.0)
    b.register_decision("tx", _decision(), _forecast(mu=0.005), 1000.0)
    oa = a.fetch_outcome("tx")
    ob = b.fetch_outcome("tx")
    assert oa is not None and ob is not None
    assert oa.realized_return == ob.realized_return


def test_different_seeds_yield_different_returns() -> None:
    a = MockOutcomeSource(noise_std=0.01, failure_rate=0.0, timeout_rate=0.0, rng_seed=1)
    b = MockOutcomeSource(noise_std=0.01, failure_rate=0.0, timeout_rate=0.0, rng_seed=99)
    a.register_decision("tx", _decision(), _forecast(mu=0.005), 1000.0)
    b.register_decision("tx", _decision(), _forecast(mu=0.005), 1000.0)
    oa = a.fetch_outcome("tx")
    ob = b.fetch_outcome("tx")
    assert oa is not None and ob is not None
    assert oa.realized_return != ob.realized_return


def test_failure_rate_drives_failures() -> None:
    src = MockOutcomeSource(failure_rate=1.0, timeout_rate=0.0, rng_seed=3)
    src.register_decision("tx", _decision(), _forecast(), 1000.0)
    out = src.fetch_outcome("tx")
    assert out is not None
    assert out.status == TradeStatus.FAILED
    assert out.fill_fraction == 0.0
    assert out.realized_cost_dollar <= 0.0


def test_timeout_rate_drives_timeouts() -> None:
    src = MockOutcomeSource(failure_rate=0.0, timeout_rate=1.0, rng_seed=4)
    src.register_decision("tx", _decision(), _forecast(), 1000.0)
    out = src.fetch_outcome("tx")
    assert out is not None
    assert out.status == TradeStatus.TIMEOUT
    assert out.fill_fraction == 0.0


def test_register_decision_adds_to_dict() -> None:
    src = MockOutcomeSource()
    assert "tx-1" not in src.forecasts_by_tx
    src.register_decision("tx-1", _decision(), _forecast(), 1000.0)
    assert "tx-1" in src.forecasts_by_tx
    decision, forecast, amount = src.forecasts_by_tx["tx-1"]
    assert decision.chosen_symbol == "SOL"
    assert forecast.symbol == "SOL"
    assert amount == 1000.0


def test_outcome_source_protocol_compliance() -> None:
    src = MockOutcomeSource()
    assert isinstance(src, OutcomeSource)
