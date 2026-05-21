"""HTTP API tests for RAMHD FastAPI layer (Step 11b)."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api_models import DecideResponse, ObserveResponse, ProcessRewardsResponse
from app.bandit.persistence import load_state
from app.dependencies import AppDependencies, build_dependencies
from app.feedback.contracts import RealizedOutcome, TradeStatus
from app.feedback.outcome_source import OutcomeSource
from app.feedback.outbox_record import OutboxStatus
from app.main import app, get_deps
from app.stored_outcome_source import StoredOutcomeSource


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def build_decide_payload(
    symbols: list[str],
    tx_id: str = "tx-1",
    amount_usd: float = 1000.0,
) -> dict[str, Any]:
    tokens = [
        {
            "symbol": sym,
            "mint": f"mint-{sym}",
            "price_usd": 100.0 if sym != "USDC" else 1.0,
            "balance": 10.0,
            "balance_usd": 1000.0 if sym != "USDC" else 10.0,
            "volatility_24h": 0.04 if sym != "USDC" else 0.002,
            "liquidity_depth_usd": 5_000_000.0,
            "spread_bps": 8.0,
        }
        for sym in symbols
    ]
    return {
        "tx_id": tx_id,
        "context": {
            "intent": {"amount_usd": amount_usd},
            "tokens": tokens,
            "network": {
                "priority_fee_lamports": 1.0,
                "congestion_score": 0.15,
                "slot_time_ms": 400.0,
            },
            "history": {},
        },
    }


def build_observe_payload(
    tx_id: str,
    *,
    status: str = "filled",
    realized_return: float = 0.005,
    realized_cost_dollar: float = -50.0,
    fill_fraction: float = 1.0,
    observed_at_utc: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "tx_id": tx_id,
        "status": status,
        "realized_return": realized_return,
        "realized_cost_dollar": realized_cost_dollar,
        "fill_fraction": fill_fraction,
    }
    if observed_at_utc is not None:
        body["observed_at_utc"] = observed_at_utc
    return body


@pytest.fixture
def test_deps(tmp_path: Path) -> AppDependencies:
    return build_dependencies(
        str(tmp_path / "outbox.sqlite"),
        str(tmp_path / "linucb_state.json"),
        str(tmp_path / "outcomes.sqlite"),
    )


@pytest.fixture
def client(
    test_deps: AppDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    monkeypatch.setattr(
        "app.dependencies.build_dependencies",
        lambda _outbox, _state, _outcomes: test_deps,
    )
    app.dependency_overrides[get_deps] = lambda: test_deps
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# -----------------------------------------------------------------------------
# Health regression
# -----------------------------------------------------------------------------


def test_health_still_works(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "ramhd"


# -----------------------------------------------------------------------------
# /decide
# -----------------------------------------------------------------------------


def test_decide_returns_chosen_symbol(client: TestClient) -> None:
    response = client.post("/decide", json=build_decide_payload(["SOL", "USDC", "BONK"]))
    assert response.status_code == 200
    body = response.json()
    assert body["chosen_symbol"] in ("SOL", "USDC", "BONK")
    assert len(body["survivors"]) >= 1
    assert body["outbox_write_succeeded"] is True


def test_decide_writes_outbox_pending_row(
    client: TestClient,
    test_deps: AppDependencies,
) -> None:
    tx_id = "tx-pending"
    response = client.post(
        "/decide",
        json=build_decide_payload(["SOL", "USDC"], tx_id=tx_id),
    )
    assert response.status_code == 200
    rec = test_deps.outbox.fetch_by_tx_id(tx_id)
    assert rec is not None
    assert rec.status == OutboxStatus.PENDING


def test_decide_skipped_symbols_reported(client: TestClient) -> None:
    response = client.post(
        "/decide",
        json=build_decide_payload(["FAKECOIN", "SOL", "USDC"]),
    )
    assert response.status_code == 200
    body = response.json()
    assert "FAKECOIN" in body["skipped_symbols"]
    assert "FAKECOIN" not in body["eligible_symbols"]


def test_decide_all_skipped_returns_422(client: TestClient) -> None:
    response = client.post(
        "/decide",
        json=build_decide_payload(["FAKECOIN1", "FAKECOIN2"]),
    )
    assert response.status_code == 422


def test_decide_empty_tx_id_returns_422(client: TestClient) -> None:
    payload = build_decide_payload(["SOL"])
    payload["tx_id"] = ""
    response = client.post("/decide", json=payload)
    assert response.status_code == 422


def test_decide_response_shape(client: TestClient) -> None:
    response = client.post("/decide", json=build_decide_payload(["SOL", "BONK"]))
    assert response.status_code == 200
    parsed = DecideResponse.model_validate(response.json())
    assert isinstance(parsed.survivors, list)
    assert isinstance(parsed.regime, str)
    assert isinstance(parsed.excluded_symbols, list)
    assert isinstance(parsed.eligible_symbols, list)
    assert isinstance(parsed.skipped_symbols, list)
    assert isinstance(parsed.outbox_write_succeeded, bool)


# -----------------------------------------------------------------------------
# /observe
# -----------------------------------------------------------------------------


def test_observe_stores_outcome(client: TestClient, test_deps: AppDependencies) -> None:
    payload = build_observe_payload("tx-obs-1")
    response = client.post("/observe", json=payload)
    assert response.status_code == 200
    body = ObserveResponse.model_validate(response.json())
    assert body.stored is True

    outcome = test_deps.outcome_source.fetch_outcome("tx-obs-1")
    assert outcome is not None
    assert outcome.tx_id == "tx-obs-1"
    assert outcome.status == TradeStatus.FILLED
    assert outcome.realized_return == pytest.approx(0.005)
    assert outcome.realized_cost_dollar == pytest.approx(-50.0)
    assert outcome.fill_fraction == pytest.approx(1.0)


def test_observe_invalid_status_returns_422(client: TestClient) -> None:
    response = client.post(
        "/observe",
        json=build_observe_payload("tx-bad", status="bogus"),
    )
    assert response.status_code == 422


def test_observe_positive_cost_returns_422(client: TestClient) -> None:
    response = client.post(
        "/observe",
        json=build_observe_payload("tx-cost", realized_cost_dollar=50.0),
    )
    assert response.status_code == 422


def test_observe_fill_out_of_range_returns_422(client: TestClient) -> None:
    response = client.post(
        "/observe",
        json=build_observe_payload("tx-fill", fill_fraction=1.5),
    )
    assert response.status_code == 422


# -----------------------------------------------------------------------------
# /admin/process-rewards
# -----------------------------------------------------------------------------


def test_process_rewards_empty_outbox(client: TestClient) -> None:
    response = client.post("/admin/process-rewards")
    assert response.status_code == 200
    body = ProcessRewardsResponse.model_validate(response.json())
    assert body.n_pending_at_start == 0
    assert body.n_processed == 0
    assert body.n_skipped == 0
    assert body.n_expired == 0
    assert body.n_still_pending == 0
    assert body.n_errors == 0


def test_full_loop_over_http(client: TestClient, test_deps: AppDependencies) -> None:
    decide_resp = client.post(
        "/decide",
        json=build_decide_payload(["SOL", "USDC", "BONK"], tx_id="tx-1"),
    )
    assert decide_resp.status_code == 200
    chosen = decide_resp.json()["chosen_symbol"]

    observe_resp = client.post(
        "/observe",
        json=build_observe_payload(
            "tx-1",
            realized_return=0.005,
            realized_cost_dollar=-50.0,
            fill_fraction=1.0,
        ),
    )
    assert observe_resp.status_code == 200

    process_resp = client.post("/admin/process-rewards")
    assert process_resp.status_code == 200
    stats = ProcessRewardsResponse.model_validate(process_resp.json())
    assert stats.n_processed == 1

    state_path = Path(test_deps.state_path)
    assert state_path.exists()
    arms = load_state(test_deps.linucb_config, path=state_path)
    assert chosen in arms
    assert arms[chosen].n_updates == 1

    rec = test_deps.outbox.fetch_by_tx_id("tx-1")
    assert rec is not None
    assert rec.status == OutboxStatus.PROCESSED


# -----------------------------------------------------------------------------
# StoredOutcomeSource unit coverage
# -----------------------------------------------------------------------------


def test_stored_outcome_source_protocol_compliance(tmp_path: Path) -> None:
    with StoredOutcomeSource(path=tmp_path / "o.sqlite") as src:
        assert isinstance(src, OutcomeSource)


def test_stored_outcome_round_trip(tmp_path: Path) -> None:
    with StoredOutcomeSource(path=tmp_path / "o.sqlite") as src:
        original = RealizedOutcome(
            tx_id="tx-rt",
            status=TradeStatus.FILLED,
            realized_return=0.01,
            realized_cost_dollar=-25.0,
            fill_fraction=1.0,
            observed_at_utc="2026-05-20T12:00:00+00:00",
        )
        src.store(original)
        fetched = src.fetch_outcome("tx-rt")
        assert fetched == original


def test_stored_outcome_fetch_unknown_returns_none(tmp_path: Path) -> None:
    with StoredOutcomeSource(path=tmp_path / "o.sqlite") as src:
        assert src.fetch_outcome("missing") is None
