# RAMHD scripts

One-off utilities. Not part of the runtime package — these are tools for
operators, not code the FastAPI service imports.

## recalibrate.py

Downloads 90 days of daily prices for the RAMHD calibration universe from
CoinGecko's free public tier and writes `data/calibration.json`. Run this
script when calibration gets stale (every few months) or when the
calibration universe changes.

### Calibration universe (8 tokens)

| Symbol | Chain   | Regime                    |
|--------|---------|---------------------------|
| SOL    | Solana  | Major / calm anchor       |
| USDC   | Solana  | Stablecoin                |
| PYTH   | Solana  | Mid-cap infrastructure    |
| AERO   | Base    | DeFi mid-cap (ve-tokens)  |
| JUP    | Solana  | DeFi mid-cap (aggregator) |
| BRETT  | Base    | Memecoin                  |
| WIF    | Solana  | Memecoin                  |
| BONK   | Solana  | Memecoin (micro-price)    |

Multi-chain, regime-stratified, spans 7 orders of magnitude in price.
This set teaches the simulator (Step 3b) what realistic price behavior
looks like across all four regimes RAMHD must handle. It is NOT the
production token universe — actual user holdings come from wallet/DB
data at runtime and can include any token.

### Run

From the `ramhd-service/` folder, with the venv activated:

```
python -m scripts.recalibrate
```

### What it produces

`ramhd-service/data/calibration.json` — committed to git so unit tests
are deterministic. The file looks like:

```json
{
  "schema_version": 1,
  "generated_at_utc": "2026-04-27T...",
  "source": "coingecko_public_v3",
  "lookback_days": 90,
  "universe_size": 8,
  "tokens": [
    {
      "symbol": "SOL",
      "coingecko_id": "solana",
      "chain": "solana",
      "address": "So111...",
      "regime": "major",
      "is_stablecoin": false,
      "n_observations": 91,
      "current_price_usd": 180.42,
      "min_price_usd": 122.10,
      "max_price_usd": 211.05,
      "daily_log_return_mean": 0.001,
      "daily_log_return_std": 0.045,
      "annualized_drift": 0.365,
      "annualized_vol": 0.86
    },
    ...
  ]
}
```

### Notes

- Uses CoinGecko's anonymous free tier — no API key required.
- 8 tokens × 1 call each = 8 calls total, with 6-second polite throttle
  between calls. Total runtime ~96 seconds (1m 36s).
- If the script partially succeeds (some tokens fail due to a network
  blip or a stale CoinGecko ID), it writes whatever calibrations it got
  and exits with code 2. Re-run to fill in gaps.
- Stablecoins (USDC) get the same calibration treatment as volatile
  tokens for uniformity, but `is_stablecoin: true` flags them so
  `MockMarketData` can generate flat-with-noise paths instead of GBM.
- If a CoinGecko coin ID changes (rare but happens), edit `TOKEN_IDS` in
  `recalibrate.py` and re-run.

### Updating the universe

To add or remove tokens:
1. Edit `TOKEN_IDS`, `TOKEN_ADDRESSES`, `TOKEN_REGIME`, and `STABLECOINS`
   in `recalibrate.py`
2. Update the universe table in this README
3. Re-run `python -m scripts.recalibrate`
4. Commit the new `data/calibration.json`
