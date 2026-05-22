"""
Synthetic backtest package for RAMHD (Step 12).

**Honesty constraint:** Episodes and outcomes here are *synthetic*. Filled-trade
returns are drawn from GARCH 120s forecasts plus Gaussian noise — the same model
family that powers the live pipeline and :class:`~app.feedback.outcome_source.MockOutcomeSource`.
This backtest validates *learning mechanics* and *relative performance vs naive
baselines*; it does **not** prove real-world edge. That requires real executor
outcomes (Step 13).

Submodules:
- :mod:`app.backtest.episode` — :class:`BacktestEpisode` and episode generation
- :mod:`app.backtest.policies` — :class:`BacktestPolicy` and baseline / bandit policies
"""

from app.backtest.episode import BacktestEpisode, EpisodeConfig, generate_episodes
from app.backtest.policies import (
    BacktestPolicy,
    CheapestRawSpreadPolicy,
    HighestCvarPolicy,
    HighestReturnPolicy,
    LargestBalancePolicy,
    LinUCBPolicy,
    LowestCostPolicy,
    OraclePolicy,
    RandomPolicy,
    StablecoinFirstPolicy,
)

__all__ = [
    "BacktestEpisode",
    "EpisodeConfig",
    "generate_episodes",
    "BacktestPolicy",
    "CheapestRawSpreadPolicy",
    "HighestCvarPolicy",
    "HighestReturnPolicy",
    "LargestBalancePolicy",
    "LinUCBPolicy",
    "LowestCostPolicy",
    "OraclePolicy",
    "RandomPolicy",
    "StablecoinFirstPolicy",
]
