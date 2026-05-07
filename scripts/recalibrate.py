"""
Step 3a — calibration data downloader.

Downloads ~90 days of historical daily prices for the RAMHD calibration
universe from CoinGecko's free public API, computes summary statistics,
and writes ramhd-service/data/calibration.json.

Why this exists:
  The synthetic market data generator (Step 3b, MockMarketData) needs
  realistic volatility and drift parameters per token. Hardcoding "50%
  annual vol" is fantasy — calibrating from real history grounds the
  simulator. The output file is committed to git so tests are
  deterministic; anyone can re-run this script to refresh the calibration.

Calibration universe (8 tokens, multi-chain, regime-stratified):
  1. SOL    Solana  major / calm anchor
  2. USDC   both    stablecoin
  3. PYTH   Solana  mid-cap infrastructure
  4. AERO   Base    DeFi mid-cap (ve-tokenomics)
  5. JUP    Solana  DeFi mid-cap (aggregator)
  6. BRETT  Base    memecoin
  7. WIF    Solana  memecoin (narrative-driven)
  8. BONK   Solana  memecoin (micro-price)

Run from the ramhd-service/ folder:
  python -m scripts.recalibrate

API notes:
  - Uses CoinGecko's free public tier (no API key, ~5-15 calls/min).
  - 8 tokens × 1 call each = 8 requests, 12-second throttle, ~96 sec total.
  - Endpoint: /coins/{id}/market_chart?vs_currency=usd&days=90&interval=daily
  - Documentation: https://docs.coingecko.com/v3.0.1/reference/coins-id-market-chart

Statistics computed per token:
  - daily_log_return_mean: average of ln(p_t / p_{t-1}) across the window
  - daily_log_return_std:  standard deviation (this is daily vol)
  - annualized_vol:        daily_std * sqrt(365), our simulator's sigma
  - annualized_drift:      daily_mean * 365, our simulator's mu
  - min_price_usd / max_price_usd / current_price_usd
  - n_observations:        number of daily bars actually returned
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# CoinGecko coin IDs for the 8-token calibration universe.
# Verified against https://api.coingecko.com/api/v3/coins/list at script-write time.
# If a token's ID has changed, the fetch will 404 and that token will be
# reported as failed — re-run after correcting the ID below.
TOKEN_IDS: dict[str, str] = {
    "SOL": "solana",
    "USDC": "usd-coin",
    "PYTH": "pyth-network",
    "AERO": "aerodrome-finance",
    "JUP": "jupiter-exchange-solana",
    "BRETT": "based-brett",
    "WIF": "dogwifcoin",
    "BONK": "bonk",
}

# Native-chain addresses. Solana tokens use SPL mint addresses; Base tokens
# use ERC-20 contract addresses. USDC is multi-chain — we record the Solana
# mint here since Solana is the primary RAMHD chain.
TOKEN_ADDRESSES: dict[str, dict[str, str]] = {
    "SOL": {"chain": "solana", "address": "So11111111111111111111111111111111111111112"},
    "USDC": {"chain": "solana", "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"},
    "PYTH": {"chain": "solana", "address": "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3"},
    "AERO": {"chain": "base", "address": "0x940181a94A35A4569E4529A3CDfB74e38FD98631"},
    "JUP": {"chain": "solana", "address": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"},
    "BRETT": {"chain": "base", "address": "0x532f27101965dd16442E59d40670FaF5eBB142E4"},
    "WIF": {"chain": "solana", "address": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"},
    "BONK": {"chain": "solana", "address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"},
}

STABLECOINS: set[str] = {"USDC"}

# Free-form regime label, used by the simulator to route synthetic-data
# generation. NOT used by the runtime algorithm — this is for development.
TOKEN_REGIME: dict[str, str] = {
    "SOL": "major",
    "USDC": "stablecoin",
    "PYTH": "mid_cap_infrastructure",
    "AERO": "defi_mid_cap",
    "JUP": "defi_mid_cap",
    "BRETT": "memecoin",
    "WIF": "memecoin",
    "BONK": "memecoin_micro",
}

API_BASE = "https://api.coingecko.com/api/v3"
DAYS = 90
SLEEP_BETWEEN_CALLS_SEC = 12.0  # 5 calls/min — below CoinGecko anonymous rate-limit floor


@dataclass
class TokenCalibration:
    """Calibration stats for one token, written to calibration.json."""

    symbol: str
    coingecko_id: str
    chain: str
    address: str
    regime: str
    is_stablecoin: bool
    n_observations: int
    current_price_usd: float
    min_price_usd: float
    max_price_usd: float
    daily_log_return_mean: float
    daily_log_return_std: float
    annualized_drift: float
    annualized_vol: float


def _fetch_market_chart(coin_id: str, days: int = DAYS) -> list[tuple[int, float]]:
    """Hit CoinGecko's market_chart endpoint. Returns list of (timestamp_ms, price_usd)."""
    url = (
        f"{API_BASE}/coins/{coin_id}/market_chart"
        f"?vs_currency=usd&days={days}&interval=daily"
    )
    req = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ramhd-recalibrate/0.1",
        },
    )
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    prices = data.get("prices", [])
    if not prices:
        raise RuntimeError(f"CoinGecko returned no price data for {coin_id}")
    return [(int(ts), float(price)) for ts, price in prices]


def _compute_stats(symbol: str, coin_id: str, prices: list[tuple[int, float]]) -> TokenCalibration:
    """Compute return-based summary statistics for one token."""
    if len(prices) < 2:
        raise RuntimeError(f"{symbol}: need at least 2 observations, got {len(prices)}")

    price_values = [p for _, p in prices]
    log_returns: list[float] = []
    for i in range(1, len(price_values)):
        prev_p = price_values[i - 1]
        curr_p = price_values[i]
        if prev_p <= 0 or curr_p <= 0:
            continue
        log_returns.append(math.log(curr_p / prev_p))

    if not log_returns:
        raise RuntimeError(f"{symbol}: no valid log returns could be computed")

    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / max(len(log_returns) - 1, 1)
    std = math.sqrt(variance)

    annualized_drift = mean * 365.0
    annualized_vol = std * math.sqrt(365.0)

    addr_info = TOKEN_ADDRESSES[symbol]

    return TokenCalibration(
        symbol=symbol,
        coingecko_id=coin_id,
        chain=addr_info["chain"],
        address=addr_info["address"],
        regime=TOKEN_REGIME[symbol],
        is_stablecoin=symbol in STABLECOINS,
        n_observations=len(price_values),
        current_price_usd=price_values[-1],
        min_price_usd=min(price_values),
        max_price_usd=max(price_values),
        daily_log_return_mean=mean,
        daily_log_return_std=std,
        annualized_drift=annualized_drift,
        annualized_vol=annualized_vol,
    )


def main() -> int:
    out_path = Path(__file__).resolve().parent.parent / "data" / "calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    calibrations: list[TokenCalibration] = []
    errors: list[str] = []

    for i, (symbol, coin_id) in enumerate(TOKEN_IDS.items()):
        if i > 0:
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)
        print(f"[{i+1}/{len(TOKEN_IDS)}] Fetching {symbol} ({coin_id})...", flush=True)
        try:
            prices = _fetch_market_chart(coin_id)
            cal = _compute_stats(symbol, coin_id, prices)
            calibrations.append(cal)
            print(
                f"  -> {cal.n_observations} obs, "
                f"price ${cal.current_price_usd:,.6f}, "
                f"annual vol {cal.annualized_vol*100:.1f}%, "
                f"annual drift {cal.annualized_drift*100:+.1f}%",
                flush=True,
            )
        except (HTTPError, URLError) as e:
            errors.append(f"{symbol} ({coin_id}): network error: {e}")
        except Exception as e:  # noqa: BLE001 — keep going for other tokens
            errors.append(f"{symbol} ({coin_id}): {type(e).__name__}: {e}")

    if errors:
        print("\nERRORS:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)

    if not calibrations:
        print("No calibrations succeeded; aborting write.", file=sys.stderr)
        return 1

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "coingecko_public_v3",
        "lookback_days": DAYS,
        "universe_size": len(calibrations),
        "tokens": [asdict(c) for c in calibrations],
    }

    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path} ({len(calibrations)}/{len(TOKEN_IDS)} tokens)")
    if errors:
        print(f"NOTE: {len(errors)} token(s) failed; calibration is partial.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
