"""
Backtest policies for synthetic episode replay (Step 12.1).

**Honesty constraint:** Rewards are scored against pre-generated synthetic
outcomes in :mod:`app.backtest.episode` (GARCH + noise), not live execution.
This validates learning mechanics and relative rankings — not real-world edge
(Step 13).

**Policy categories** (for honest reporting in Step 12.3):

- ``naive_baseline`` — raw fields only (``spread_bps``, ``balance_usd``,
  stablecoin flag, random). No ML / Pareto dimensions.
- ``ml_ablation`` — one dimension of Pareto-scored survivors (cost, return,
  CVaR). Uses the ML pipeline; **not** naive.
- ``bandit`` — LinUCB contextual selection (RAMHD under test).
- ``oracle`` — counterfactual ceiling via :func:`~app.feedback.reward.compute_reward`.
  Backtest-only; cannot run in production.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Protocol, runtime_checkable

import numpy as np

from app.backtest.episode import BacktestEpisode
from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import LinUCBArmState, LinUCBConfig
from app.bandit.selector import pick_candidate
from app.feedback.contracts import RewardConfig
from app.feedback.reward import compute_reward
from app.market_data.calibration import Calibration
from app.pareto.contracts import CandidateScore
from app.schemas import RamhdContext, TokenMarketSnapshot

logger = logging.getLogger(__name__)

PolicyCategory = Literal["naive_baseline", "ml_ablation", "bandit", "oracle"]

VALID_POLICY_CATEGORIES: frozenset[str] = frozenset(
    {"naive_baseline", "ml_ablation", "bandit", "oracle"}
)


def _require_survivors(survivors: list[CandidateScore]) -> None:
    if not survivors:
        raise ValueError("survivors must be non-empty")


def _tiebreak_symbol(candidates: list[CandidateScore], key_fn) -> str:
    """Pick the best survivor; ties broken by lexicographic symbol (Step 12)."""
    best = min(candidates, key=lambda c: (key_fn(c), c.symbol))
    return best.symbol


def _snapshot_by_symbol(context: RamhdContext) -> dict[str, TokenMarketSnapshot]:
    """Map token symbol → its :class:`~app.schemas.TokenMarketSnapshot` in the context."""
    return {t.symbol: t for t in context.tokens}


def _snapshot_for_symbol(episode: BacktestEpisode, symbol: str) -> TokenMarketSnapshot:
    try:
        return _snapshot_by_symbol(episode.context)[symbol]
    except KeyError as e:
        available = sorted(t.symbol for t in episode.context.tokens)
        raise KeyError(
            f"symbol {symbol!r} not in episode.context.tokens. Available: {available}"
        ) from e


@runtime_checkable
class BacktestPolicy(Protocol):
    """Select one survivor symbol for a backtest episode."""

    @property
    def name(self) -> str:
        """Human-readable policy label for reports."""
        ...

    @property
    def category(self) -> PolicyCategory:
        """One of: naive_baseline, ml_ablation, bandit, oracle."""
        ...

    def choose(
        self,
        episode: BacktestEpisode,
        survivors: list[CandidateScore],
    ) -> str:
        """Return the symbol this policy would spend for ``episode``."""
        ...


class RandomPolicy:
    """Uniform random choice among survivors (deterministic per episode_id + seed)."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        return "random"

    @property
    def category(self) -> PolicyCategory:
        return "naive_baseline"

    def choose(
        self,
        episode: BacktestEpisode,
        survivors: list[CandidateScore],
    ) -> str:
        _require_survivors(survivors)
        rng = np.random.default_rng(self._seed + episode.episode_id)
        idx = int(rng.integers(0, len(survivors)))
        chosen = survivors[idx].symbol
        logger.debug("episode %d random chose %s", episode.episode_id, chosen)
        return chosen


class CheapestRawSpreadPolicy:
    """Naive baseline: lowest raw ``spread_bps`` from context (not ML cost)."""

    @property
    def name(self) -> str:
        return "cheapest_raw_spread"

    @property
    def category(self) -> PolicyCategory:
        return "naive_baseline"

    def choose(
        self,
        episode: BacktestEpisode,
        survivors: list[CandidateScore],
    ) -> str:
        _require_survivors(survivors)

        def spread_key(s: CandidateScore) -> tuple[float, str]:
            snap = _snapshot_for_symbol(episode, s.symbol)
            return (float(snap.spread_bps), s.symbol)

        return min(survivors, key=spread_key).symbol


class LargestBalancePolicy:
    """Naive baseline: largest raw ``balance_usd`` from context."""

    @property
    def name(self) -> str:
        return "largest_balance"

    @property
    def category(self) -> PolicyCategory:
        return "naive_baseline"

    def choose(
        self,
        episode: BacktestEpisode,
        survivors: list[CandidateScore],
    ) -> str:
        _require_survivors(survivors)

        def balance_key(s: CandidateScore) -> tuple[float, str]:
            snap = _snapshot_for_symbol(episode, s.symbol)
            return (-float(snap.balance_usd), s.symbol)

        return min(survivors, key=balance_key).symbol


class StablecoinFirstPolicy:
    """Prefer any stablecoin survivor; otherwise lowest ML effective cost."""

    def __init__(self, calibration: Calibration | None = None) -> None:
        self._calibration = calibration if calibration is not None else Calibration()

    @property
    def name(self) -> str:
        return "stablecoin_first"

    @property
    def category(self) -> PolicyCategory:
        return "naive_baseline"

    def choose(
        self,
        episode: BacktestEpisode,
        survivors: list[CandidateScore],
    ) -> str:
        _require_survivors(survivors)
        for s in survivors:
            if self._calibration.get(s.symbol).is_stablecoin:
                return s.symbol
        return _tiebreak_symbol(survivors, lambda c: c.effective_cost_bps)


class LowestCostPolicy:
    """ML-ablation: minimum ``effective_cost_bps`` among Pareto survivors (not naive)."""

    @property
    def name(self) -> str:
        return "lowest_cost"

    @property
    def category(self) -> PolicyCategory:
        return "ml_ablation"

    def choose(
        self,
        episode: BacktestEpisode,
        survivors: list[CandidateScore],
    ) -> str:
        _require_survivors(survivors)
        return _tiebreak_symbol(survivors, lambda c: c.effective_cost_bps)


class HighestReturnPolicy:
    """ML-ablation: maximum ``expected_return_120s`` (not naive)."""

    @property
    def name(self) -> str:
        return "highest_return"

    @property
    def category(self) -> PolicyCategory:
        return "ml_ablation"

    def choose(
        self,
        episode: BacktestEpisode,
        survivors: list[CandidateScore],
    ) -> str:
        _require_survivors(survivors)
        return _tiebreak_symbol(survivors, lambda c: -c.expected_return_120s)


class HighestCvarPolicy:
    """ML-ablation: safest tail via max ``cvar_95_120s`` (not naive)."""

    @property
    def name(self) -> str:
        return "highest_cvar"

    @property
    def category(self) -> PolicyCategory:
        return "ml_ablation"

    def choose(
        self,
        episode: BacktestEpisode,
        survivors: list[CandidateScore],
    ) -> str:
        _require_survivors(survivors)
        return _tiebreak_symbol(survivors, lambda c: -c.cvar_95_120s)


class LinUCBPolicy:
    """RAMHD LinUCB selection among survivors (read-only on ``arms``)."""

    def __init__(
        self,
        arms: dict[str, LinUCBArmState],
        config: LinUCBConfig,
        calibration: Calibration,
        bandit_calibration: BanditCalibration,
    ) -> None:
        self._arms = arms
        self._config = config
        self._calibration = calibration
        self._bandit_calibration = bandit_calibration

    @property
    def name(self) -> str:
        return "linucb"

    @property
    def category(self) -> PolicyCategory:
        return "bandit"

    def choose(
        self,
        episode: BacktestEpisode,
        survivors: list[CandidateScore],
    ) -> str:
        _require_survivors(survivors)
        symbols = [s.symbol for s in survivors]
        result = pick_candidate(
            episode.context,
            symbols,
            self._arms,
            self._config,
            self._calibration,
            self._bandit_calibration,
        )
        return result.chosen_symbol


class OraclePolicy:
    """Backtest-only ceiling: pick the survivor with highest counterfactual reward.

    Uses pre-generated outcomes via :meth:`BacktestEpisode.outcome_if_chosen` and
    :func:`~app.feedback.reward.compute_reward`. Requires outcome access, so this
    policy cannot run in production — it exists for regret metrics and learning-curve
    upper bounds in synthetic backtests only.
    """

    def __init__(self, reward_config: Optional[RewardConfig] = None) -> None:
        self._reward_config = reward_config if reward_config is not None else RewardConfig()

    @property
    def name(self) -> str:
        return "oracle"

    @property
    def category(self) -> PolicyCategory:
        return "oracle"

    def choose(
        self,
        episode: BacktestEpisode,
        survivors: list[CandidateScore],
    ) -> str:
        _require_survivors(survivors)
        amount = float(episode.context.intent.amount_usd)

        def reward_key(s: CandidateScore) -> tuple[float, str]:
            outcome = episode.outcome_if_chosen(s.symbol)
            reward = compute_reward(outcome, amount, self._reward_config)
            score = float("-inf") if reward is None else float(reward)
            return (-score, s.symbol)

        return min(survivors, key=reward_key).symbol
