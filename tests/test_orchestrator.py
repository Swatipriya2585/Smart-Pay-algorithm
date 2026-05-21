"""Integration tests for the RAMHD orchestrator (Step 11)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.bandit.calibration import BanditCalibration
from app.bandit.persistence import load_state
from app.cost.scorer import SolanaCostScorer
from app.feedback.outbox import SQLiteOutboxStore
from app.feedback.outbox_record import OutboxStatus
from app.forecasting.garch import GARCHForecaster
from app.market_data.base import NetworkConditions, TokenMarketData
from app.market_data.calibration import Calibration
from app.market_data.mock import MockConfig, MockMarketData
from app.orchestrator import (
    OrchestratorConfig,
    apply_live_snapshot,
    run_orchestration,
    select_regime_reference_symbol,
    to_network_conditions,
)
from app.regime.threshold import ThresholdRegimeDetector
from app.risk.monte_carlo import MonteCarloCVaR
from app.routing.risk_adaptive import RuleBasedRiskAdaptiveRouter
from app.schemas import (
    HistoryStats,
    NetworkState,
    PaymentIntent,
    RamhdContext,
    TokenMarketSnapshot,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def build_context(
    symbols: list[str],
    *,
    amount_usd: float = 1000.0,
    congestion: float = 0.15,
    liquidity_depth_usd: float = 5_000_000.0,
    spread_bps: float = 8.0,
) -> RamhdContext:
    tokens = [
        TokenMarketSnapshot(
            symbol=sym,
            mint=f"mint-{sym}",
            price_usd=100.0 if sym != "USDC" else 1.0,
            balance=10.0,
            balance_usd=1000.0 if sym != "USDC" else 10.0,
            volatility_24h=0.04 if sym != "USDC" else 0.002,
            liquidity_depth_usd=liquidity_depth_usd,
            spread_bps=spread_bps,
        )
        for sym in symbols
    ]
    return RamhdContext(
        intent=PaymentIntent(amount_usd=amount_usd),
        tokens=tokens,
        network=NetworkState(
            priority_fee_lamports=1.0,
            congestion_score=congestion,
            slot_time_ms=400.0,
        ),
        history=HistoryStats(),
    )


@dataclass
class OrchestratorDeps:
    market_data_source: MockMarketData
    calibration: Calibration
    bandit_calibration: BanditCalibration
    forecaster: GARCHForecaster
    risk_estimator: MonteCarloCVaR
    cost_scorer: SolanaCostScorer
    regime_detector: ThresholdRegimeDetector
    router: RuleBasedRiskAdaptiveRouter
    outbox: SQLiteOutboxStore
    config: OrchestratorConfig
    state_path: Path


def make_orchestrator_deps(tmp_path: Path) -> OrchestratorDeps:
    cal = Calibration()
    bcal = BanditCalibration()
    mock_cfg = MockConfig(seed=42)
    return OrchestratorDeps(
        market_data_source=MockMarketData(calibration=cal, config=mock_cfg),
        calibration=cal,
        bandit_calibration=bcal,
        forecaster=GARCHForecaster(calibration=cal),
        risk_estimator=MonteCarloCVaR(),
        cost_scorer=SolanaCostScorer(),
        regime_detector=ThresholdRegimeDetector(calibration=cal),
        router=RuleBasedRiskAdaptiveRouter(),
        outbox=SQLiteOutboxStore(path=tmp_path / "outbox.sqlite"),
        config=OrchestratorConfig(),
        state_path=tmp_path / "linucb_state.json",
    )


# -----------------------------------------------------------------------------
# Adapters
# -----------------------------------------------------------------------------


def test_to_network_conditions_maps_all_fields() -> None:
    ns = NetworkState(
        priority_fee_lamports=12.5,
        congestion_score=0.42,
        slot_time_ms=380.0,
    )
    nc = to_network_conditions(ns)
    assert isinstance(nc, NetworkConditions)
    assert nc.priority_fee_lamports == 12.5
    assert nc.congestion_score == 0.42
    assert nc.slot_time_ms == 380.0


def test_apply_live_snapshot_overrides_liquidity_and_spread() -> None:
    cal = Calibration()
    mock = MockMarketData(calibration=cal, config=MockConfig(seed=7))
    data = mock.fetch(["SOL"])[0]
    original_path = data.path
    snapshot = TokenMarketSnapshot(
        symbol="SOL",
        mint="mint-SOL",
        price_usd=150.0,
        balance=1.0,
        balance_usd=150.0,
        volatility_24h=0.05,
        liquidity_depth_usd=9_999_999.0,
        spread_bps=42.0,
    )
    updated = apply_live_snapshot(data, snapshot)
    assert updated.liquidity_depth_usd == 9_999_999.0
    assert updated.spread_bps == 42.0
    assert updated.path is original_path


# -----------------------------------------------------------------------------
# Regime reference selection
# -----------------------------------------------------------------------------


def test_regime_ref_prefers_non_stablecoin() -> None:
    cal = Calibration()
    assert select_regime_reference_symbol(["USDC", "SOL"], cal) == "SOL"


def test_regime_ref_all_stablecoins_uses_first() -> None:
    cal = Calibration()
    assert select_regime_reference_symbol(["USDC"], cal) == "USDC"


def test_regime_ref_empty_raises() -> None:
    cal = Calibration()
    with pytest.raises(ValueError, match="eligible_symbols"):
        select_regime_reference_symbol([], cal)


# -----------------------------------------------------------------------------
# Full orchestration
# -----------------------------------------------------------------------------


def test_run_orchestration_end_to_end(tmp_path: Path) -> None:
    deps = make_orchestrator_deps(tmp_path)
    try:
        result = run_orchestration(
            tx_id="tx-1",
            context=build_context(["SOL", "USDC", "BONK"]),
            market_data_source=deps.market_data_source,
            calibration=deps.calibration,
            bandit_calibration=deps.bandit_calibration,
            forecaster=deps.forecaster,
            risk_estimator=deps.risk_estimator,
            cost_scorer=deps.cost_scorer,
            regime_detector=deps.regime_detector,
            router=deps.router,
            outbox=deps.outbox,
            config=deps.config,
            state_path=deps.state_path,
            now_utc_iso="2026-05-20T12:00:00+00:00",
        )
        assert result.tx_id == "tx-1"
        assert result.chosen_symbol in ("SOL", "USDC", "BONK")
        assert 2 <= len(result.survivors) <= 5
        assert result.recorded_decision.outbox_write_succeeded is True

        rec = deps.outbox.fetch_by_tx_id("tx-1")
        assert rec is not None
        assert rec.status == OutboxStatus.PENDING
        assert rec.chosen_symbol == result.chosen_symbol
    finally:
        deps.outbox.close()


def test_skipped_symbols_excluded(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    deps = make_orchestrator_deps(tmp_path)
    caplog.set_level(logging.WARNING)
    try:
        result = run_orchestration(
            tx_id="tx-skip",
            context=build_context(["FAKECOIN", "SOL", "USDC"]),
            market_data_source=deps.market_data_source,
            calibration=deps.calibration,
            bandit_calibration=deps.bandit_calibration,
            forecaster=deps.forecaster,
            risk_estimator=deps.risk_estimator,
            cost_scorer=deps.cost_scorer,
            regime_detector=deps.regime_detector,
            router=deps.router,
            outbox=deps.outbox,
            config=deps.config,
            state_path=deps.state_path,
            now_utc_iso="2026-05-20T12:00:00+00:00",
        )
        assert "FAKECOIN" in result.skipped_symbols
        assert "FAKECOIN" not in result.eligible_symbols
        assert result.chosen_symbol in ("SOL", "USDC")
        assert any("FAKECOIN" in r.message for r in caplog.records)
    finally:
        deps.outbox.close()


def test_all_symbols_skipped_raises(tmp_path: Path) -> None:
    deps = make_orchestrator_deps(tmp_path)
    try:
        with pytest.raises(ValueError, match="no eligible symbols"):
            run_orchestration(
                tx_id="tx-none",
                context=build_context(["FAKECOIN1", "FAKECOIN2"]),
                market_data_source=deps.market_data_source,
                calibration=deps.calibration,
                bandit_calibration=deps.bandit_calibration,
                forecaster=deps.forecaster,
                risk_estimator=deps.risk_estimator,
                cost_scorer=deps.cost_scorer,
                regime_detector=deps.regime_detector,
                router=deps.router,
                outbox=deps.outbox,
                config=deps.config,
                state_path=deps.state_path,
            )
    finally:
        deps.outbox.close()


def test_empty_tx_id_raises(tmp_path: Path) -> None:
    deps = make_orchestrator_deps(tmp_path)
    try:
        with pytest.raises(ValueError, match="tx_id"):
            run_orchestration(
                tx_id="",
                context=build_context(["SOL"]),
                market_data_source=deps.market_data_source,
                calibration=deps.calibration,
                bandit_calibration=deps.bandit_calibration,
                forecaster=deps.forecaster,
                risk_estimator=deps.risk_estimator,
                cost_scorer=deps.cost_scorer,
                regime_detector=deps.regime_detector,
                router=deps.router,
                outbox=deps.outbox,
                config=deps.config,
                state_path=deps.state_path,
            )
    finally:
        deps.outbox.close()


def test_empty_tokens_raises(tmp_path: Path) -> None:
    deps = make_orchestrator_deps(tmp_path)
    ctx = RamhdContext.model_construct(
        intent=PaymentIntent(amount_usd=1000.0),
        tokens=[],
        network=NetworkState(
            priority_fee_lamports=1.0,
            congestion_score=0.1,
            slot_time_ms=400.0,
        ),
        history=HistoryStats(),
    )
    try:
        with pytest.raises(ValueError, match="context.tokens"):
            run_orchestration(
                tx_id="tx-empty",
                context=ctx,
                market_data_source=deps.market_data_source,
                calibration=deps.calibration,
                bandit_calibration=deps.bandit_calibration,
                forecaster=deps.forecaster,
                risk_estimator=deps.risk_estimator,
                cost_scorer=deps.cost_scorer,
                regime_detector=deps.regime_detector,
                router=deps.router,
                outbox=deps.outbox,
                config=deps.config,
                state_path=deps.state_path,
            )
    finally:
        deps.outbox.close()


def test_determinism(tmp_path: Path) -> None:
    chosen: list[str] = []
    for i in range(2):
        sub = tmp_path / f"run{i}"
        sub.mkdir()
        deps = make_orchestrator_deps(sub)
        try:
            result = run_orchestration(
                tx_id=f"tx-det-{i}",
                context=build_context(["SOL", "USDC", "BONK"]),
                market_data_source=deps.market_data_source,
                calibration=deps.calibration,
                bandit_calibration=deps.bandit_calibration,
                forecaster=deps.forecaster,
                risk_estimator=deps.risk_estimator,
                cost_scorer=deps.cost_scorer,
                regime_detector=deps.regime_detector,
                router=deps.router,
                outbox=deps.outbox,
                config=deps.config,
                state_path=deps.state_path,
                now_utc_iso="2026-05-20T12:00:00+00:00",
            )
            chosen.append(result.chosen_symbol)
        finally:
            deps.outbox.close()
    assert chosen[0] == chosen[1]


def test_regime_present_in_result(tmp_path: Path) -> None:
    deps = make_orchestrator_deps(tmp_path)
    try:
        result = run_orchestration(
            tx_id="tx-regime",
            context=build_context(["SOL", "USDC"]),
            market_data_source=deps.market_data_source,
            calibration=deps.calibration,
            bandit_calibration=deps.bandit_calibration,
            forecaster=deps.forecaster,
            risk_estimator=deps.risk_estimator,
            cost_scorer=deps.cost_scorer,
            regime_detector=deps.regime_detector,
            router=deps.router,
            outbox=deps.outbox,
            config=deps.config,
            state_path=deps.state_path,
            now_utc_iso="2026-05-20T12:00:00+00:00",
        )
        assert result.regime.regime in ("calm", "stress", "shock")
        assert 0.0 <= result.regime.confidence <= 1.0
    finally:
        deps.outbox.close()


def test_survivors_subset_of_eligible(tmp_path: Path) -> None:
    deps = make_orchestrator_deps(tmp_path)
    try:
        result = run_orchestration(
            tx_id="tx-sub",
            context=build_context(["SOL", "USDC", "BONK"]),
            market_data_source=deps.market_data_source,
            calibration=deps.calibration,
            bandit_calibration=deps.bandit_calibration,
            forecaster=deps.forecaster,
            risk_estimator=deps.risk_estimator,
            cost_scorer=deps.cost_scorer,
            regime_detector=deps.regime_detector,
            router=deps.router,
            outbox=deps.outbox,
            config=deps.config,
            state_path=deps.state_path,
            now_utc_iso="2026-05-20T12:00:00+00:00",
        )
        eligible_set = set(result.eligible_symbols)
        for s in result.survivors:
            assert s.symbol in eligible_set
    finally:
        deps.outbox.close()


def test_bandit_not_run_twice(tmp_path: Path) -> None:
    """record_decision loads bandit state read-only; weights update only via reward processor."""
    deps = make_orchestrator_deps(tmp_path)
    cfg = deps.config.linucb_config
    try:
        result = run_orchestration(
            tx_id="tx-bandit-once",
            context=build_context(["SOL", "USDC", "BONK"]),
            market_data_source=deps.market_data_source,
            calibration=deps.calibration,
            bandit_calibration=deps.bandit_calibration,
            forecaster=deps.forecaster,
            risk_estimator=deps.risk_estimator,
            cost_scorer=deps.cost_scorer,
            regime_detector=deps.regime_detector,
            router=deps.router,
            outbox=deps.outbox,
            config=deps.config,
            state_path=deps.state_path,
            now_utc_iso="2026-05-20T12:00:00+00:00",
        )
        chosen = result.chosen_symbol
        if deps.state_path.exists():
            arms = load_state(cfg, path=deps.state_path)
            if chosen in arms:
                assert arms[chosen].n_updates == 0
        else:
            assert not deps.state_path.exists()
    finally:
        deps.outbox.close()
