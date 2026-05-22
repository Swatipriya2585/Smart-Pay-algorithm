#!/usr/bin/env python3
"""
Run an end-to-end synthetic RAMHD backtest and write metrics artifacts.

**Honesty constraint:** Outcomes are model-generated (GARCH + Gaussian noise).
This validates learning mechanics and relative performance vs naive baselines —
it does NOT prove real-world edge. See plot caption and CSV for the same caveat.

Install plotting support (optional for --no-plot):

    pip install -e ".[analysis]"

Usage:

    python -m scripts.run_backtest
    python -m scripts.run_backtest --n-episodes 200 --seed 42 --out-dir backtest_out
    python -m scripts.run_backtest --no-plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.backtest.episode import EpisodeConfig, generate_episodes
from app.backtest.harness import HarnessConfig, run_backtest
from app.backtest.metrics import compute_metrics, learning_lift
from app.backtest.policies import (
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
from app.backtest.report import metrics_to_csv, plot_learning_curve
from app.bandit.calibration import BanditCalibration
from app.bandit.contracts import LinUCBConfig
from app.cost.scorer import SolanaCostScorer
from app.forecasting.garch import GARCHForecaster
from app.market_data.calibration import Calibration
from app.market_data.mock import MockConfig, MockMarketData
from app.pareto.contracts import ParetoConfig
from app.regime.threshold import ThresholdRegimeDetector
from app.risk.monte_carlo import MonteCarloCVaR
from app.routing.risk_adaptive import RuleBasedRiskAdaptiveRouter


def _build_stages(seed: int) -> dict:
    cal = Calibration()
    bcal = BanditCalibration()
    mock = MockMarketData(calibration=cal, config=MockConfig(seed=seed))
    return {
        "calibration": cal,
        "bandit_calibration": bcal,
        "market_data_source": mock,
        "forecaster": GARCHForecaster(calibration=cal),
        "risk_estimator": MonteCarloCVaR(),
        "cost_scorer": SolanaCostScorer(),
        "regime_detector": ThresholdRegimeDetector(calibration=cal),
        "router": RuleBasedRiskAdaptiveRouter(),
        "pareto_config": ParetoConfig(),
    }


def _build_policies(stages: dict, seed: int) -> tuple[list, dict]:
    cal = stages["calibration"]
    bcal = stages["bandit_calibration"]
    bandit_arms: dict = {}
    policies = [
        RandomPolicy(seed=seed),
        CheapestRawSpreadPolicy(),
        LargestBalancePolicy(),
        StablecoinFirstPolicy(calibration=cal),
        LowestCostPolicy(),
        HighestReturnPolicy(),
        HighestCvarPolicy(),
        LinUCBPolicy(bandit_arms, LinUCBConfig(), cal, bcal),
        OraclePolicy(),
    ]
    return policies, bandit_arms


def _print_summary(metrics) -> None:
    bandit_name = metrics.bandit_policy_name
    bandit_pm = metrics.per_policy[bandit_name]
    print("\n=== RAMHD Backtest Summary (synthetic) ===\n")
    print(f"Episodes: {metrics.n_episodes}  Skipped: {metrics.n_skipped}\n")
    print(f"{'Policy':<22} {'Category':<16} {'Mean reward':>12}")
    print("-" * 52)
    for name in sorted(metrics.per_policy.keys()):
        pm = metrics.per_policy[name]
        print(f"{name:<22} {pm.policy_category:<16} {pm.mean_reward:>12.6f}")

    print(f"\nBandit ({bandit_name}) mean reward: {bandit_pm.mean_reward:.6f}")
    print(f"Learning lift (2nd half - 1st half): {learning_lift(metrics):+.6f}")
    print(f"Bandit total regret vs oracle: {metrics.total_regret[bandit_name]:.6f}")

    print("\nBandit win-rate vs comparators:")
    for comp, rate in sorted(metrics.bandit_winrate_vs.items()):
        print(f"  vs {comp:<20}: {rate:.1%}")

    naive = [
        n
        for n, pm in metrics.per_policy.items()
        if pm.policy_category == "naive_baseline"
    ]
    if naive:
        print("\nBandit win-rate vs naive baselines:")
        for comp in sorted(naive):
            if comp in metrics.bandit_winrate_vs:
                print(f"  vs {comp:<20}: {metrics.bandit_winrate_vs[comp]:.1%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RAMHD synthetic backtest")
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-dir", type=str, default="backtest_out")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG (no matplotlib required)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = _build_stages(args.seed)
    policies, _ = _build_policies(stages, args.seed)

    print(
        f"Generating {args.n_episodes} episodes (seed={args.seed})...",
        flush=True,
    )
    episodes = generate_episodes(
        EpisodeConfig(n_episodes=args.n_episodes, seed=args.seed),
        calibration=stages["calibration"],
        market_data_source=stages["market_data_source"],
        forecaster=stages["forecaster"],
    )

    print("Running backtest (bandit learns in-memory)...", flush=True)
    record = run_backtest(
        episodes,
        policies,
        market_data_source=stages["market_data_source"],
        calibration=stages["calibration"],
        bandit_calibration=stages["bandit_calibration"],
        forecaster=stages["forecaster"],
        risk_estimator=stages["risk_estimator"],
        cost_scorer=stages["cost_scorer"],
        regime_detector=stages["regime_detector"],
        router=stages["router"],
        pareto_config=stages["pareto_config"],
        config=HarnessConfig(bandit_learns=True),
        seed=args.seed,
    )

    metrics = compute_metrics(record)
    _print_summary(metrics)

    csv_path = metrics_to_csv(metrics, out_dir / "backtest_metrics.csv")
    print(f"\nWrote CSV: {csv_path.resolve()}")

    if not args.no_plot:
        png_path = plot_learning_curve(metrics, out_dir / "learning_curve.png")
        print(f"Wrote plot: {png_path.resolve()}")
    else:
        print("Skipped plot (--no-plot).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
