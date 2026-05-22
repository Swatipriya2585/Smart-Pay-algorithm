"""Tests for synthetic backtest episodes (Step 12.1)."""

from __future__ import annotations

import pytest

from app.backtest.episode import BacktestEpisode, EpisodeConfig, generate_episodes
from app.feedback.contracts import TradeStatus
from app.market_data.calibration import Calibration


def test_episode_config_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="n_episodes"):
        EpisodeConfig(n_episodes=0)
    with pytest.raises(ValueError, match="max_tokens_per_episode"):
        EpisodeConfig(min_tokens_per_episode=3, max_tokens_per_episode=2)


def test_generate_episodes_count_and_ids() -> None:
    config = EpisodeConfig(n_episodes=5, seed=99)
    episodes = generate_episodes(config)
    assert len(episodes) == 5
    assert [e.episode_id for e in episodes] == [0, 1, 2, 3, 4]


def test_generate_episodes_deterministic() -> None:
    cfg = EpisodeConfig(n_episodes=8, seed=123, symbols=("SOL", "USDC", "BONK"))
    a = generate_episodes(cfg)
    b = generate_episodes(cfg)
    assert len(a) == len(b)
    for ep_a, ep_b in zip(a, b, strict=True):
        assert ep_a.episode_id == ep_b.episode_id
        assert ep_a.context.model_dump() == ep_b.context.model_dump()
        assert set(ep_a.outcomes_by_symbol) == set(ep_b.outcomes_by_symbol)
        for sym in ep_a.outcomes_by_symbol:
            oa = ep_a.outcomes_by_symbol[sym]
            ob = ep_b.outcomes_by_symbol[sym]
            assert oa.realized_return == ob.realized_return
            assert oa.realized_cost_dollar == ob.realized_cost_dollar


def test_outcomes_cover_eligible_symbols() -> None:
    episodes = generate_episodes(EpisodeConfig(n_episodes=3, seed=7))
    for ep in episodes:
        eligible = ep.eligible_symbols()
        assert len(eligible) >= 2
        assert set(eligible) == set(ep.outcomes_by_symbol)
        for sym in eligible:
            out = ep.outcomes_by_symbol[sym]
            assert out.status == TradeStatus.FILLED
            assert out.fill_fraction == pytest.approx(1.0)
            assert out.tx_id == f"bt-{ep.episode_id}"
            assert out.realized_cost_dollar <= 0


def test_outcome_if_chosen_lookup() -> None:
    episodes = generate_episodes(EpisodeConfig(n_episodes=1, seed=1))
    ep = episodes[0]
    sym = ep.eligible_symbols()[0]
    assert ep.outcome_if_chosen(sym) is ep.outcomes_by_symbol[sym]
    with pytest.raises(KeyError, match="NOPE"):
        ep.outcome_if_chosen("NOPE")


def test_generate_episodes_requires_calibrated_pool() -> None:
    with pytest.raises(ValueError, match="calibrated symbols"):
        generate_episodes(
            EpisodeConfig(
                n_episodes=1,
                symbols=("FAKECOIN", "OTHER"),
                min_tokens_per_episode=2,
            )
        )


def test_backtest_episode_frozen() -> None:
    episodes = generate_episodes(EpisodeConfig(n_episodes=1))
    ep = episodes[0]
    assert isinstance(ep, BacktestEpisode)
    with pytest.raises(AttributeError):
        ep.episode_id = 99  # type: ignore[misc]


def test_context_tokens_subset_of_calibration() -> None:
    cal = Calibration()
    episodes = generate_episodes(
        EpisodeConfig(n_episodes=2, seed=42, symbols=tuple(cal.symbols[:5])),
        calibration=cal,
    )
    for ep in episodes:
        for token in ep.context.tokens:
            assert cal.has(token.symbol)
