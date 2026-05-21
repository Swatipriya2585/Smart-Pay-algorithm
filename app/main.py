"""
RAMHD FastAPI entry point.

Exposes health, token selection (/decide), outcome ingestion (/observe),
and reward processing (/admin/process-rewards).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request

from app.api_models import (
    DecideRequest,
    DecideResponse,
    ObserveRequest,
    ObserveResponse,
    ProcessRewardsResponse,
)
from app.config import settings
from app.dependencies import AppDependencies, build_dependencies, close_dependencies
from app.feedback.contracts import RealizedOutcome, TradeStatus
from app.feedback.processor import run_reward_processor
from app.orchestrator import run_orchestration
from app.schemas import HealthResponse

logger = logging.getLogger(__name__)


def _runtime_paths() -> tuple[str, str, str]:
    """Production paths from settings; pytest uses a temp dir (not data/)."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        base = Path(os.environ.get("RAMHD_TEST_DATA_DIR", "/tmp/ramhd_pytest"))
        base.mkdir(parents=True, exist_ok=True)
        return (
            str(base / "ramhd_outbox.sqlite"),
            str(base / "linucb_state.json"),
            str(base / "ramhd_outcomes.sqlite"),
        )
    return settings.outbox_path, settings.state_path, settings.outcome_store_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    outbox_path, state_path, outcome_path = _runtime_paths()
    app.state.deps = build_dependencies(outbox_path, state_path, outcome_path)
    yield
    close_dependencies(app.state.deps)


app = FastAPI(
    title="RAMHD Service",
    description="Risk-Adaptive Multi-Horizon Dominance — Smart Pay token selection.",
    version=settings.version,
    lifespan=lifespan,
)


def get_deps(request: Request) -> AppDependencies:
    return request.app.state.deps


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Liveness probe. Returns 200 when the service is up."""
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.version,
    )


@app.post("/decide", response_model=DecideResponse, tags=["routing"])
async def decide(
    body: DecideRequest,
    deps: AppDependencies = Depends(get_deps),
) -> DecideResponse:
    """Run the full RAMHD pipeline and return the chosen token."""
    try:
        result = run_orchestration(
            tx_id=body.tx_id,
            context=body.context,
            market_data_source=deps.market_data_source,
            calibration=deps.calibration,
            bandit_calibration=deps.bandit_calibration,
            forecaster=deps.forecaster,
            risk_estimator=deps.risk_estimator,
            cost_scorer=deps.cost_scorer,
            regime_detector=deps.regime_detector,
            router=deps.router,
            outbox=deps.outbox,
            config=deps.orchestrator_config,
            state_path=deps.state_path,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("decide failed for tx_id=%s", body.tx_id)
        raise HTTPException(status_code=500, detail="internal server error") from e

    return DecideResponse(
        tx_id=result.tx_id,
        chosen_symbol=result.chosen_symbol,
        survivors=[s.symbol for s in result.survivors],
        regime=result.regime.regime,
        excluded_symbols=list(result.excluded_symbols),
        eligible_symbols=list(result.eligible_symbols),
        skipped_symbols=list(result.skipped_symbols),
        outbox_write_succeeded=result.recorded_decision.outbox_write_succeeded,
    )


@app.post("/observe", response_model=ObserveResponse, tags=["feedback"])
async def observe(
    body: ObserveRequest,
    deps: AppDependencies = Depends(get_deps),
) -> ObserveResponse:
    """Store a realized outcome for later reward processing (push side)."""
    try:
        trade_status = TradeStatus(body.status)
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"invalid status {body.status!r}; must be one of "
            f"{[s.value for s in TradeStatus]}",
        ) from e

    observed_at = (
        body.observed_at_utc
        if body.observed_at_utc is not None
        else datetime.now(timezone.utc).isoformat()
    )

    try:
        outcome = RealizedOutcome(
            tx_id=body.tx_id,
            status=trade_status,
            realized_return=body.realized_return,
            realized_cost_dollar=body.realized_cost_dollar,
            fill_fraction=body.fill_fraction,
            observed_at_utc=observed_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    deps.outcome_source.store(outcome)
    logger.info("observe stored tx_id=%s status=%s", body.tx_id, trade_status.value)
    return ObserveResponse(tx_id=body.tx_id, stored=True)


@app.post(
    "/admin/process-rewards",
    response_model=ProcessRewardsResponse,
    tags=["admin"],
)
async def process_rewards(
    max_age_seconds: float = 600.0,
    deps: AppDependencies = Depends(get_deps),
) -> ProcessRewardsResponse:
    """Drain pending outbox rows and apply bandit updates (pull side)."""
    stats = run_reward_processor(
        outbox=deps.outbox,
        outcome_source=deps.outcome_source,
        linucb_config=deps.linucb_config,
        reward_config=deps.reward_config,
        state_path=deps.state_path,
        max_age_seconds=max_age_seconds,
    )
    logger.info(
        "process-rewards: pending_start=%d processed=%d skipped=%d expired=%d "
        "still_pending=%d errors=%d elapsed=%.3fs",
        stats.n_pending_at_start,
        stats.n_processed,
        stats.n_skipped,
        stats.n_expired,
        stats.n_still_pending,
        stats.n_errors,
        stats.elapsed_seconds,
    )
    return ProcessRewardsResponse(
        n_pending_at_start=stats.n_pending_at_start,
        n_processed=stats.n_processed,
        n_skipped=stats.n_skipped,
        n_expired=stats.n_expired,
        n_still_pending=stats.n_still_pending,
        n_errors=stats.n_errors,
        elapsed_seconds=stats.elapsed_seconds,
    )
