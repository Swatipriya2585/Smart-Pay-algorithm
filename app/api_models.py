"""Pydantic models for the RAMHD HTTP API boundary."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from app.schemas import RamhdContext


class DecideRequest(BaseModel):
    tx_id: str
    context: RamhdContext


class DecideResponse(BaseModel):
    tx_id: str
    chosen_symbol: str
    survivors: list[str]
    regime: str
    excluded_symbols: list[str]
    eligible_symbols: list[str]
    skipped_symbols: list[str]
    outbox_write_succeeded: bool


class ObserveRequest(BaseModel):
    tx_id: str
    status: str
    realized_return: float
    realized_cost_dollar: float
    fill_fraction: float
    observed_at_utc: Optional[str] = None


class ObserveResponse(BaseModel):
    tx_id: str
    stored: bool


class ProcessRewardsResponse(BaseModel):
    n_pending_at_start: int
    n_processed: int
    n_skipped: int
    n_expired: int
    n_still_pending: int
    n_errors: int
    elapsed_seconds: float
