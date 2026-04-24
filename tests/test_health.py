"""Sanity tests for the scaffold. Verifies the service starts and health check works."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "ramhd"
    assert "version" in body


def test_docs_available() -> None:
    """FastAPI auto-docs should be reachable."""
    response = client.get("/docs")
    assert response.status_code == 200
