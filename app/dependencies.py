"""Application singletons and dependency injection for FastAPI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import LinUCBConfig
from app.cost.scorer import SolanaCostScorer
from app.feedback.contracts import RewardConfig
from app.feedback.outbox import SQLiteOutboxStore
from app.forecasting.garch import GARCHForecaster
from app.market_data.base import MarketDataSource
from app.market_data.calibration import Calibration
from app.market_data.mock import MockMarketData
from app.orchestrator import OrchestratorConfig
from app.regime.threshold import ThresholdRegimeDetector
from app.risk.monte_carlo import MonteCarloCVaR
from app.routing.risk_adaptive import RuleBasedRiskAdaptiveRouter
from app.stored_outcome_source import StoredOutcomeSource


@dataclass
class AppDependencies:
    calibration: Calibration
    bandit_calibration: BanditCalibration
    market_data_source: MarketDataSource
    forecaster: GARCHForecaster
    risk_estimator: MonteCarloCVaR
    cost_scorer: SolanaCostScorer
    regime_detector: ThresholdRegimeDetector
    router: RuleBasedRiskAdaptiveRouter
    outbox: SQLiteOutboxStore
    outcome_source: StoredOutcomeSource
    orchestrator_config: OrchestratorConfig
    linucb_config: LinUCBConfig
    reward_config: RewardConfig
    state_path: str


def build_dependencies(
    outbox_path: str,
    state_path: str,
    outcome_store_path: str,
) -> AppDependencies:
    """Construct all stage singletons once (called from lifespan startup)."""
    calibration = Calibration()
    bandit_calibration = BanditCalibration()
    market_data_source = MockMarketData(calibration=calibration)
    forecaster = GARCHForecaster(calibration=calibration)
    risk_estimator = MonteCarloCVaR()
    cost_scorer = SolanaCostScorer()
    regime_detector = ThresholdRegimeDetector(calibration=calibration)
    router = RuleBasedRiskAdaptiveRouter()
    outbox = SQLiteOutboxStore(path=outbox_path)
    outcome_source = StoredOutcomeSource(path=outcome_store_path)

    return AppDependencies(
        calibration=calibration,
        bandit_calibration=bandit_calibration,
        market_data_source=market_data_source,
        forecaster=forecaster,
        risk_estimator=risk_estimator,
        cost_scorer=cost_scorer,
        regime_detector=regime_detector,
        router=router,
        outbox=outbox,
        outcome_source=outcome_source,
        orchestrator_config=OrchestratorConfig(),
        linucb_config=LinUCBConfig(),
        reward_config=RewardConfig(),
        state_path=state_path,
    )


def close_dependencies(deps: AppDependencies) -> None:
    """Close SQLite connections on application shutdown."""
    deps.outbox.close()
    deps.outcome_source.close()
