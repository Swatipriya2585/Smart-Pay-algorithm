"""
Synthetic backtest episodes (Step 12.1).

**Honesty constraint:** Each episode pre-generates counterfactual
:class:`~app.feedback.contracts.RealizedOutcome` values per eligible token using
GARCH 120s forecasts and Gaussian noise. Policies are compared against the *same*
frozen world, but that world is model-generated — not live execution data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from app.feedback.contracts import RealizedOutcome, TradeStatus
from app.forecasting.base import MultiHorizonForecast
from app.forecasting.garch import GARCHForecaster
from app.market_data.calibration import Calibration, Regime
from app.market_data.mock import MockConfig, MockMarketData
from app.orchestrator import apply_live_snapshot
from app.schemas import (
    HistoryStats,
    NetworkState,
    PaymentIntent,
    RamhdContext,
    TokenMarketSnapshot,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestEpisode:
    """One synthetic payment scenario to replay.

    Holds everything needed to run any policy and score counterfactual outcomes:
    the request context plus a per-symbol map of what *would* happen if that
    token were chosen. Outcomes are pre-generated so every policy faces the
    same world (apples-to-apples comparison).
    """

    episode_id: int
    context: RamhdContext
    outcomes_by_symbol: dict[str, RealizedOutcome]

    def eligible_symbols(self) -> tuple[str, ...]:
        """Symbols from the context that appear in the outcome map (calibration-eligible)."""
        return tuple(t.symbol for t in self.context.tokens if t.symbol in self.outcomes_by_symbol)

    def outcome_if_chosen(self, symbol: str) -> RealizedOutcome:
        """Return the pre-generated outcome for a counterfactual choice of ``symbol``."""
        try:
            return self.outcomes_by_symbol[symbol]
        except KeyError as e:
            raise KeyError(
                f"{symbol} not in episode {self.episode_id} outcomes. "
                f"Available: {sorted(self.outcomes_by_symbol)}"
            ) from e


@dataclass(frozen=True)
class EpisodeConfig:
    """Knobs for synthetic episode generation."""

    n_episodes: int = 50
    seed: int = 42
    symbols: tuple[str, ...] = ("SOL", "USDC", "BONK", "JUP")
    min_tokens_per_episode: int = 2
    max_tokens_per_episode: int = 4
    amount_usd: float = 1000.0
    noise_std: float = 0.005
    cost_dollar_per_trade: float = -50.0
    mock_seed: int = 42
    mock_regime: Regime = "stress"

    def __post_init__(self) -> None:
        if self.n_episodes < 1:
            raise ValueError(f"n_episodes must be >= 1, got {self.n_episodes}")
        if not self.symbols:
            raise ValueError("symbols must be non-empty")
        if self.min_tokens_per_episode < 1:
            raise ValueError(
                f"min_tokens_per_episode must be >= 1, got {self.min_tokens_per_episode}"
            )
        if self.max_tokens_per_episode < self.min_tokens_per_episode:
            raise ValueError(
                "max_tokens_per_episode must be >= min_tokens_per_episode"
            )
        if self.amount_usd <= 0:
            raise ValueError(f"amount_usd must be positive, got {self.amount_usd}")
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {self.noise_std}")
        if self.cost_dollar_per_trade > 1e-9:
            raise ValueError(
                "cost_dollar_per_trade must be non-positive, "
                f"got {self.cost_dollar_per_trade}"
            )


def _episode_rng(config: EpisodeConfig, episode_id: int) -> np.random.Generator:
    return np.random.default_rng(config.seed + episode_id * 10_007)


def _outcome_rng(config: EpisodeConfig, episode_id: int, symbol: str) -> np.random.Generator:
    sym_seed = hash(symbol) & 0xFFFFFFFF
    return np.random.default_rng(config.seed + episode_id * 10_007 + sym_seed)


def _build_context(
    episode_id: int,
    symbols: list[str],
    config: EpisodeConfig,
) -> RamhdContext:
    tokens = [
        TokenMarketSnapshot(
            symbol=sym,
            mint=f"mint-{sym}",
            price_usd=100.0 if sym != "USDC" else 1.0,
            balance=10.0,
            balance_usd=config.amount_usd if sym != "USDC" else 10.0,
            volatility_24h=0.04 if sym != "USDC" else 0.002,
            liquidity_depth_usd=5_000_000.0,
            spread_bps=8.0,
        )
        for sym in symbols
    ]
    congestion = float(_episode_rng(config, episode_id).uniform(0.05, 0.35))
    return RamhdContext(
        intent=PaymentIntent(amount_usd=config.amount_usd),
        tokens=tokens,
        network=NetworkState(
            priority_fee_lamports=1.0,
            congestion_score=congestion,
            slot_time_ms=400.0,
        ),
        history=HistoryStats(),
    )


def _synthetic_filled_outcome(
    *,
    tx_id: str,
    symbol: str,
    forecast: MultiHorizonForecast,
    config: EpisodeConfig,
    episode_id: int,
) -> RealizedOutcome:
    """FILLED outcome: GARCH 120s mean + per-(episode, symbol) Gaussian noise."""
    mu = forecast.at(120.0).predicted_return
    noise = float(_outcome_rng(config, episode_id, symbol).normal(0.0, config.noise_std))
    observed_at = datetime.now(timezone.utc).isoformat()
    return RealizedOutcome(
        tx_id=tx_id,
        status=TradeStatus.FILLED,
        realized_return=mu + noise,
        realized_cost_dollar=config.cost_dollar_per_trade,
        fill_fraction=1.0,
        observed_at_utc=observed_at,
    )


def generate_episodes(
    config: EpisodeConfig,
    calibration: Calibration | None = None,
    forecaster: GARCHForecaster | None = None,
    market_data_source: MockMarketData | None = None,
) -> list[BacktestEpisode]:
    """Generate a list of synthetic backtest episodes.

    Each episode samples a token subset from ``config.symbols`` (calibration
    universe), builds a :class:`~app.schemas.RamhdContext`, fetches mock market
    data, runs GARCH, and pre-computes a FILLED :class:`RealizedOutcome` per
    eligible symbol.
    """
    cal = calibration if calibration is not None else Calibration()
    available = [s for s in config.symbols if cal.has(s)]
    if len(available) < config.min_tokens_per_episode:
        raise ValueError(
            f"need at least {config.min_tokens_per_episode} calibrated symbols "
            f"from {config.symbols}; only {available} available"
        )

    mock_cfg = MockConfig(seed=config.mock_seed, regime=config.mock_regime)
    mock = (
        market_data_source
        if market_data_source is not None
        else MockMarketData(calibration=cal, config=mock_cfg)
    )
    garch = forecaster if forecaster is not None else GARCHForecaster(calibration=cal)

    episodes: list[BacktestEpisode] = []
    for episode_id in range(config.n_episodes):
        rng = _episode_rng(config, episode_id)
        max_n = min(config.max_tokens_per_episode, len(available))
        min_n = min(config.min_tokens_per_episode, max_n)
        n_tokens = int(rng.integers(min_n, max_n + 1))
        chosen_symbols = list(rng.choice(available, size=n_tokens, replace=False))
        context = _build_context(episode_id, chosen_symbols, config)
        tx_id = f"bt-{episode_id}"

        eligible = [t.symbol for t in context.tokens if cal.has(t.symbol)]
        md_list = mock.fetch(eligible)
        snapshot_by_symbol = {t.symbol: t for t in context.tokens}

        outcomes: dict[str, RealizedOutcome] = {}
        for data in md_list:
            live = apply_live_snapshot(data, snapshot_by_symbol[data.symbol])
            forecast = garch.forecast(live)
            outcomes[data.symbol] = _synthetic_filled_outcome(
                tx_id=tx_id,
                symbol=data.symbol,
                forecast=forecast,
                config=config,
                episode_id=episode_id,
            )

        episodes.append(
            BacktestEpisode(
                episode_id=episode_id,
                context=context,
                outcomes_by_symbol=outcomes,
            )
        )

    logger.info(
        "generated %d backtest episodes (symbols pool=%s)",
        len(episodes),
        config.symbols,
    )
    return episodes
