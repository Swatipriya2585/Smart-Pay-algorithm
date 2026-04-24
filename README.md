# RAMHD Service

Risk-Adaptive Multi-Horizon Dominance — the new Smart Pay token-selection
algorithm for CryptoChain, running as an isolated Python FastAPI service.

## What this service does
Given a payment intent (amount, user holdings, market snapshot), RAMHD
returns a ranked recommendation of which held token to spend. It will
eventually replace the heuristic `analyzeCryptoPortfolio` logic with a
principled algorithm: GARCH volatility forecasting, CVaR tail-risk
estimation, regime-adaptive weighting, Pareto frontier filtering, and
LinUCB contextual bandit learning.

## Isolation contract
This service runs in its own process. It does not share code, database,
or dependencies with the existing stack. The existing crypto-prediction
endpoints continue to run unchanged.

## Files that must never be modified by RAMHD work
- Smart-pay/Stock-crypto-portfolio-optimizer/pages/api/crypto-prediction.ts
- Smart-pay/Stock-crypto-portfolio-optimizer/pages/api/crypto-prediction-enhanced.ts
- Smart-pay/Stock-crypto-portfolio-optimizer/lib/bots/BotOrchestrator.ts
- Smart-pay/Stock-crypto-portfolio-optimizer/lib/bots/base.ts
- All three services/ai-recommendation.ts (root, CryptoChain-Complete, Stock optimizer)
- cryptochain_mobile/lib/services/api_service.dart
- cryptochain_mobile/lib/screens/home_screen.dart
- All prisma/schema.prisma files

## Running locally (do not run yet — later step)
    cd ramhd-service
    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    uvicorn app.main:app --reload --port 8100

## Project layout
- app/main.py — FastAPI app, health check only (Step 1)
- app/schemas.py — Pydantic models for the context vector x in R^d
- app/config.py — runtime settings
- tests/ — pytest-based tests
- pyproject.toml — dependencies
