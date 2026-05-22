"""Tests for backtest policies (Step 12.1)."""

from __future__ import annotations

import pytest

from app.backtest.episode import BacktestEpisode, EpisodeConfig, generate_episodes
from app.backtest.policies import (
    VALID_POLICY_CATEGORIES,
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
from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import LinUCBConfig
from app.feedback.contracts import RealizedOutcome, RewardConfig, TradeStatus
from app.feedback.reward import compute_reward
from app.market_data.calibration import Calibration
from app.pareto.contracts import CandidateScore
from app.schemas import (
    HistoryStats,
    NetworkState,
    PaymentIntent,
    RamhdContext,
    TokenMarketSnapshot,
)


def _survivors(rows: list[tuple[str, float, float, float, float]]) -> list[CandidateScore]:
    return [
        CandidateScore(
            symbol=sym,
            expected_return_120s=ret,
            cvar_95_120s=cvar,
            effective_cost_bps=cost,
            liquidity_usd=liq,
        )
        for sym, ret, cvar, cost, liq in rows
    ]


def _episode_with_tokens(
    tokens: list[TokenMarketSnapshot],
    *,
    episode_id: int = 0,
    amount_usd: float = 1000.0,
    outcomes: dict[str, RealizedOutcome] | None = None,
) -> BacktestEpisode:
    ctx = RamhdContext(
        intent=PaymentIntent(amount_usd=amount_usd),
        tokens=tokens,
        network=NetworkState(
            priority_fee_lamports=1.0,
            congestion_score=0.1,
            slot_time_ms=400.0,
        ),
        history=HistoryStats(),
    )
    if outcomes is None:
        outcomes = {
            t.symbol: RealizedOutcome(
                tx_id=f"bt-{episode_id}",
                status=TradeStatus.FILLED,
                realized_return=0.0,
                realized_cost_dollar=-10.0,
                fill_fraction=1.0,
                observed_at_utc="2026-01-01T00:00:00+00:00",
            )
            for t in tokens
        }
    return BacktestEpisode(
        episode_id=episode_id,
        context=ctx,
        outcomes_by_symbol=outcomes,
    )


def _minimal_episode(episode_id: int = 0) -> BacktestEpisode:
    return _episode_with_tokens(
        [
            TokenMarketSnapshot(
                symbol="SOL",
                mint="m1",
                price_usd=100.0,
                balance=1.0,
                balance_usd=100.0,
                volatility_24h=0.04,
                liquidity_depth_usd=1e6,
                spread_bps=10.0,
            ),
            TokenMarketSnapshot(
                symbol="USDC",
                mint="m2",
                price_usd=1.0,
                balance=1000.0,
                balance_usd=1000.0,
                volatility_24h=0.002,
                liquidity_depth_usd=1e6,
                spread_bps=5.0,
            ),
        ],
        episode_id=episode_id,
        outcomes={
            "SOL": RealizedOutcome(
                tx_id="bt-0",
                status=TradeStatus.FILLED,
                realized_return=0.01,
                realized_cost_dollar=-50.0,
                fill_fraction=1.0,
                observed_at_utc="2026-01-01T00:00:00+00:00",
            ),
            "USDC": RealizedOutcome(
                tx_id="bt-0",
                status=TradeStatus.FILLED,
                realized_return=0.001,
                realized_cost_dollar=-10.0,
                fill_fraction=1.0,
                observed_at_utc="2026-01-01T00:00:00+00:00",
            ),
        },
    )


def _all_policies() -> list[BacktestPolicy]:
    cal = Calibration()
    bcal = BanditCalibration()
    return [
        RandomPolicy(seed=1),
        CheapestRawSpreadPolicy(),
        LargestBalancePolicy(),
        StablecoinFirstPolicy(calibration=cal),
        LowestCostPolicy(),
        HighestReturnPolicy(),
        HighestCvarPolicy(),
        LinUCBPolicy({}, LinUCBConfig(), cal, bcal),
        OraclePolicy(),
    ]


@pytest.fixture
def three_way_survivors() -> list[CandidateScore]:
    return _survivors(
        [
            ("SOL", 0.02, -0.05, 40.0, 2_000_000.0),
            ("USDC", 0.001, -0.001, 15.0, 5_000_000.0),
            ("BONK", 0.03, -0.08, 55.0, 500_000.0),
        ]
    )


def test_all_policies_satisfy_protocol() -> None:
    ep = _minimal_episode()
    survivors = _survivors(
        [
            ("SOL", 0.02, -0.05, 40.0, 2_000_000.0),
            ("USDC", 0.001, -0.001, 15.0, 5_000_000.0),
        ]
    )
    for policy in _all_policies():
        assert isinstance(policy, BacktestPolicy)
        chosen = policy.choose(ep, survivors)
        assert chosen in {s.symbol for s in survivors}


def test_every_policy_has_valid_category() -> None:
    for policy in _all_policies():
        assert policy.category in VALID_POLICY_CATEGORIES


def test_lowest_cost_picks_cheapest(three_way_survivors: list[CandidateScore]) -> None:
    ep = _minimal_episode()
    assert LowestCostPolicy().choose(ep, three_way_survivors) == "USDC"
    assert LowestCostPolicy().category == "ml_ablation"


def test_highest_return_picks_best_return(three_way_survivors: list[CandidateScore]) -> None:
    ep = _minimal_episode()
    assert HighestReturnPolicy().choose(ep, three_way_survivors) == "BONK"
    assert HighestReturnPolicy().category == "ml_ablation"


def test_highest_cvar_picks_safest_tail(three_way_survivors: list[CandidateScore]) -> None:
    ep = _minimal_episode()
    assert HighestCvarPolicy().choose(ep, three_way_survivors) == "USDC"
    assert HighestCvarPolicy().category == "ml_ablation"


def test_stablecoin_first_prefers_usdc(three_way_survivors: list[CandidateScore]) -> None:
    ep = _minimal_episode()
    assert StablecoinFirstPolicy().choose(ep, three_way_survivors) == "USDC"
    assert StablecoinFirstPolicy().category == "naive_baseline"


def test_stablecoin_first_falls_back_to_lowest_cost() -> None:
    ep = _minimal_episode()
    survivors = _survivors(
        [
            ("SOL", 0.02, -0.05, 30.0, 2e6),
            ("BONK", 0.03, -0.08, 20.0, 1e6),
        ]
    )
    assert StablecoinFirstPolicy().choose(ep, survivors) == "BONK"


def test_random_policy_deterministic_per_episode(
    three_way_survivors: list[CandidateScore],
) -> None:
    policy = RandomPolicy(seed=42)
    ep0 = _minimal_episode(0)
    ep1 = _minimal_episode(1)
    assert policy.choose(ep0, three_way_survivors) == policy.choose(ep0, three_way_survivors)
    assert policy.choose(ep0, three_way_survivors) != policy.choose(ep1, three_way_survivors) or len(
        three_way_survivors
    ) == 1


def test_linucb_chooses_from_survivors() -> None:
    ep = _minimal_episode()
    survivors = _survivors(
        [
            ("SOL", 0.02, -0.05, 40.0, 2_000_000.0),
            ("USDC", 0.001, -0.001, 15.0, 5_000_000.0),
        ]
    )
    cal = Calibration()
    bcal = BanditCalibration()
    policy = LinUCBPolicy({}, LinUCBConfig(), cal, bcal)
    chosen = policy.choose(ep, survivors)
    assert chosen in {"SOL", "USDC"}
    assert policy.category == "bandit"


def test_empty_survivors_raises() -> None:
    ep = _minimal_episode()
    with pytest.raises(ValueError, match="survivors"):
        LowestCostPolicy().choose(ep, [])


def test_policy_names() -> None:
    assert RandomPolicy().name == "random"
    assert CheapestRawSpreadPolicy().name == "cheapest_raw_spread"
    assert LargestBalancePolicy().name == "largest_balance"
    assert LowestCostPolicy().name == "lowest_cost"
    assert HighestReturnPolicy().name == "highest_return"
    assert HighestCvarPolicy().name == "highest_cvar"
    assert StablecoinFirstPolicy().name == "stablecoin_first"
    assert LinUCBPolicy({}, LinUCBConfig(), Calibration(), BanditCalibration()).name == "linucb"
    assert OraclePolicy().name == "oracle"


def test_policies_on_generated_episode() -> None:
    episodes = generate_episodes(EpisodeConfig(n_episodes=1, seed=5))
    ep = episodes[0]
    symbols = ep.eligible_symbols()
    survivors = _survivors(
        [
            (symbols[0], 0.01, -0.03, 30.0, 1e6),
            (symbols[1], 0.02, -0.04, 25.0, 2e6),
        ]
    )
    chosen = RandomPolicy(seed=0).choose(ep, survivors)
    assert chosen in symbols


def test_naive_baselines_do_not_read_ml_fields() -> None:
    """RAW spread ordering differs from ML effective_cost ordering."""
    ep = _episode_with_tokens(
        [
            TokenMarketSnapshot(
                symbol="AAA",
                mint="m1",
                price_usd=10.0,
                balance=1.0,
                balance_usd=500.0,
                volatility_24h=0.04,
                liquidity_depth_usd=1e6,
                spread_bps=20.0,
            ),
            TokenMarketSnapshot(
                symbol="BBB",
                mint="m2",
                price_usd=10.0,
                balance=1.0,
                balance_usd=2000.0,
                volatility_24h=0.04,
                liquidity_depth_usd=1e6,
                spread_bps=5.0,
            ),
        ]
    )
    survivors = _survivors(
        [
            # AAA: lowest ML cost, highest raw spread
            ("AAA", 0.01, -0.02, 10.0, 1e6),
            # BBB: higher ML cost, lowest raw spread
            ("BBB", 0.01, -0.02, 50.0, 1e6),
        ]
    )
    assert LowestCostPolicy().choose(ep, survivors) == "AAA"
    assert CheapestRawSpreadPolicy().choose(ep, survivors) == "BBB"
    assert LargestBalancePolicy().choose(ep, survivors) == "BBB"


def test_cheapest_raw_spread_picks_lowest_raw_spread() -> None:
    ep = _minimal_episode()
    survivors = _survivors(
        [
            ("SOL", 0.0, 0.0, 99.0, 1e6),
            ("USDC", 0.0, 0.0, 1.0, 1e6),
        ]
    )
    assert CheapestRawSpreadPolicy().choose(ep, survivors) == "USDC"


def test_cheapest_raw_spread_alphabetical_tiebreak() -> None:
    ep = _episode_with_tokens(
        [
            TokenMarketSnapshot(
                symbol="ZZZ",
                mint="m1",
                price_usd=1.0,
                balance=1.0,
                balance_usd=100.0,
                volatility_24h=0.01,
                liquidity_depth_usd=1e6,
                spread_bps=7.0,
            ),
            TokenMarketSnapshot(
                symbol="AAA",
                mint="m2",
                price_usd=1.0,
                balance=1.0,
                balance_usd=100.0,
                volatility_24h=0.01,
                liquidity_depth_usd=1e6,
                spread_bps=7.0,
            ),
        ]
    )
    survivors = _survivors(
        [
            ("ZZZ", 0.0, 0.0, 1.0, 1e6),
            ("AAA", 0.0, 0.0, 99.0, 1e6),
        ]
    )
    assert CheapestRawSpreadPolicy().choose(ep, survivors) == "AAA"


def test_largest_balance_picks_max_raw_balance() -> None:
    ep = _minimal_episode()
    survivors = _survivors(
        [
            ("SOL", 0.0, 0.0, 1.0, 1e6),
            ("USDC", 0.0, 0.0, 99.0, 1e6),
        ]
    )
    assert LargestBalancePolicy().choose(ep, survivors) == "USDC"


def test_largest_balance_alphabetical_tiebreak() -> None:
    ep = _episode_with_tokens(
        [
            TokenMarketSnapshot(
                symbol="ZZZ",
                mint="m1",
                price_usd=1.0,
                balance=1.0,
                balance_usd=5000.0,
                volatility_24h=0.01,
                liquidity_depth_usd=1e6,
                spread_bps=1.0,
            ),
            TokenMarketSnapshot(
                symbol="AAA",
                mint="m2",
                price_usd=1.0,
                balance=1.0,
                balance_usd=5000.0,
                volatility_24h=0.01,
                liquidity_depth_usd=1e6,
                spread_bps=1.0,
            ),
        ]
    )
    survivors = _survivors(
        [
            ("ZZZ", 0.0, 0.0, 1.0, 1e6),
            ("AAA", 0.0, 0.0, 99.0, 1e6),
        ]
    )
    assert LargestBalancePolicy().choose(ep, survivors) == "AAA"


def test_oracle_picks_highest_reward() -> None:
    # FILLED rewards at amount_usd=1000:
    #   GOOD: return=0.05, cost=-10  -> 0.05 - 0.01 = 0.04
    #   BAD:  return=0.01, cost=-50  -> 0.01 - 0.05 = -0.04
    ep = _episode_with_tokens(
        [
            TokenMarketSnapshot(
                symbol="GOOD",
                mint="m1",
                price_usd=1.0,
                balance=1.0,
                balance_usd=100.0,
                volatility_24h=0.01,
                liquidity_depth_usd=1e6,
                spread_bps=1.0,
            ),
            TokenMarketSnapshot(
                symbol="BAD",
                mint="m2",
                price_usd=1.0,
                balance=1.0,
                balance_usd=100.0,
                volatility_24h=0.01,
                liquidity_depth_usd=1e6,
                spread_bps=1.0,
            ),
        ],
        outcomes={
            "GOOD": RealizedOutcome(
                tx_id="bt-0",
                status=TradeStatus.FILLED,
                realized_return=0.05,
                realized_cost_dollar=-10.0,
                fill_fraction=1.0,
                observed_at_utc="2026-01-01T00:00:00+00:00",
            ),
            "BAD": RealizedOutcome(
                tx_id="bt-0",
                status=TradeStatus.FILLED,
                realized_return=0.01,
                realized_cost_dollar=-50.0,
                fill_fraction=1.0,
                observed_at_utc="2026-01-01T00:00:00+00:00",
            ),
        },
    )
    survivors = _survivors(
        [
            ("GOOD", 0.0, 0.0, 99.0, 1e6),
            ("BAD", 0.0, 0.0, 1.0, 1e6),
        ]
    )
    cfg = RewardConfig()
    assert compute_reward(ep.outcomes_by_symbol["GOOD"], 1000.0, cfg) == pytest.approx(0.04)
    assert compute_reward(ep.outcomes_by_symbol["BAD"], 1000.0, cfg) == pytest.approx(-0.04)
    assert OraclePolicy(cfg).choose(ep, survivors) == "GOOD"


def test_oracle_treats_data_missing_as_worst() -> None:
    ep = _episode_with_tokens(
        [
            TokenMarketSnapshot(
                symbol="OK",
                mint="m1",
                price_usd=1.0,
                balance=1.0,
                balance_usd=100.0,
                volatility_24h=0.01,
                liquidity_depth_usd=1e6,
                spread_bps=1.0,
            ),
            TokenMarketSnapshot(
                symbol="MISSING",
                mint="m2",
                price_usd=1.0,
                balance=1.0,
                balance_usd=100.0,
                volatility_24h=0.01,
                liquidity_depth_usd=1e6,
                spread_bps=1.0,
            ),
        ],
        outcomes={
            "OK": RealizedOutcome(
                tx_id="bt-0",
                status=TradeStatus.FILLED,
                realized_return=0.001,
                realized_cost_dollar=-50.0,
                fill_fraction=1.0,
                observed_at_utc="2026-01-01T00:00:00+00:00",
            ),
            "MISSING": RealizedOutcome(
                tx_id="bt-0",
                status=TradeStatus.DATA_MISSING,
                realized_return=0.0,
                realized_cost_dollar=0.0,
                fill_fraction=0.0,
                observed_at_utc="2026-01-01T00:00:00+00:00",
            ),
        },
    )
    survivors = _survivors(
        [
            ("OK", 0.0, 0.0, 50.0, 1e6),
            ("MISSING", 0.0, 0.0, 1.0, 1e6),
        ]
    )
    assert OraclePolicy().choose(ep, survivors) == "OK"


def test_oracle_deterministic() -> None:
    ep = _minimal_episode()
    survivors = _survivors(
        [
            ("SOL", 0.0, 0.0, 10.0, 1e6),
            ("USDC", 0.0, 0.0, 20.0, 1e6),
        ]
    )
    policy = OraclePolicy()
    assert policy.choose(ep, survivors) == policy.choose(ep, survivors)


def test_oracle_category_is_oracle() -> None:
    assert OraclePolicy().category == "oracle"
