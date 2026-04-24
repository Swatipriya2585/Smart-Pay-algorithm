"""
RAMHD FastAPI entry point.

Step 1: exposes only a health check. The /recommend endpoint will be
added in a later step once the algorithm stages are implemented.
"""

from fastapi import FastAPI

from app.config import settings
from app.schemas import HealthResponse

app = FastAPI(
    title="RAMHD Service",
    description="Risk-Adaptive Multi-Horizon Dominance — Smart Pay token selection.",
    version=settings.version,
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Liveness probe. Returns 200 when the service is up."""
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.version,
    )
