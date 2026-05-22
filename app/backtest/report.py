"""
Backtest reporting: CSV export and learning-curve plot (Step 12.3).

**Honesty constraint:** Reports and plots describe synthetic outcomes (GARCH +
noise). They validate learning mechanics and relative performance — not
real-world edge. Captions and titles must carry this caveat.

matplotlib is imported lazily inside :func:`plot_learning_curve` only.
"""

from __future__ import annotations

import csv
from pathlib import Path

from app.backtest.metrics import BacktestMetrics

HONESTY_CAPTION = (
    "Synthetic outcomes (GARCH + noise) — validates learning, not real-world edge."
)

CSV_COLUMNS_BASE = [
    "policy_name",
    "policy_category",
    "n_choices",
    "n_missing",
    "mean_reward",
    "mean_reward_first_half",
    "mean_reward_second_half",
    "total_regret",
]


def metrics_to_csv(metrics: BacktestMetrics, path: str | Path) -> Path:
    """Write per-policy metrics to CSV (stdlib ``csv`` only)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    winrate_cols = sorted(metrics.bandit_winrate_vs.keys())
    fieldnames = CSV_COLUMNS_BASE + [f"winrate_vs_{c}" for c in winrate_cols]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name in sorted(metrics.per_policy.keys()):
            pm = metrics.per_policy[name]
            row: dict[str, str | int | float] = {
                "policy_name": pm.policy_name,
                "policy_category": pm.policy_category,
                "n_choices": pm.n_choices,
                "n_missing": pm.n_missing,
                "mean_reward": pm.mean_reward,
                "mean_reward_first_half": pm.mean_reward_first_half,
                "mean_reward_second_half": pm.mean_reward_second_half,
                "total_regret": metrics.total_regret.get(name, 0.0),
            }
            for comp in winrate_cols:
                key = f"winrate_vs_{comp}"
                if name == metrics.bandit_policy_name:
                    row[key] = metrics.bandit_winrate_vs[comp]
                else:
                    row[key] = ""
            writer.writerow(row)

    return out


def plot_learning_curve(
    metrics: BacktestMetrics,
    path: str | Path,
    title: str = "RAMHD Backtest — Cumulative Reward by Policy",
) -> Path:
    """Save a PNG of cumulative reward per policy (headless Agg backend)."""
    try:
        import matplotlib
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install -e '.[analysis]'"
        ) from e

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    category_colors = {
        "naive_baseline": "#4C72B0",
        "ml_ablation": "#55A868",
        "bandit": "#C44E52",
        "oracle": "#8172B2",
    }
    category_styles = {
        "naive_baseline": "-",
        "ml_ablation": "-",
        "bandit": "-",
        "oracle": "--",
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for policy_name in metrics.per_policy.keys():
        pm = metrics.per_policy[policy_name]
        series = metrics.cumulative_reward.get(policy_name, [])
        if not series:
            continue
        x = list(range(len(series)))
        color = category_colors.get(pm.policy_category, "#333333")
        linestyle = category_styles.get(pm.policy_category, "-")
        linewidth = 2.5 if policy_name == metrics.bandit_policy_name else 1.2
        alpha = 1.0 if policy_name in (
            metrics.bandit_policy_name,
            metrics.oracle_policy_name,
        ) else 0.85
        ax.plot(
            x,
            series,
            label=f"{policy_name} ({pm.policy_category})",
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.set_xlabel("Episode index (acted episodes, ascending order)")
    ax.set_ylabel("Cumulative reward (non-None rewards only)")
    ax.set_title(f"{title}\n{HONESTY_CAPTION}", fontsize=11)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)

    return out
