"""Tests for backtest report CSV and plot (Step 12.3)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.backtest.harness import BacktestRecord, PolicyChoice
from app.backtest.metrics import compute_metrics
from app.backtest.report import HONESTY_CAPTION, metrics_to_csv, plot_learning_curve


def _minimal_metrics():
    choices = []
    for ep_id in range(4):
        choices.append(PolicyChoice(ep_id, "linucb", "bandit", "SOL", 0.05 * ep_id))
        choices.append(PolicyChoice(ep_id, "oracle", "oracle", "SOL", 0.2))
        choices.append(PolicyChoice(ep_id, "random", "naive_baseline", "SOL", 0.01))
    record = BacktestRecord(
        choices=choices,
        n_episodes=4,
        n_skipped_episodes=0,
        policy_names=("linucb", "oracle", "random"),
        policy_categories={
            "linucb": "bandit",
            "oracle": "oracle",
            "random": "naive_baseline",
        },
        seed=1,
    )
    return compute_metrics(record)


def test_csv_written_and_parseable(tmp_path: Path) -> None:
    metrics = _minimal_metrics()
    path = metrics_to_csv(metrics, tmp_path / "metrics.csv")
    assert path.exists()
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(metrics.per_policy)
    assert "policy_name" in rows[0]
    assert "policy_category" in rows[0]
    assert "mean_reward" in rows[0]
    assert "total_regret" in rows[0]
    linucb_row = next(r for r in rows if r["policy_name"] == "linucb")
    assert linucb_row["policy_category"] == "bandit"
    assert float(linucb_row["mean_reward"]) == pytest.approx(
        metrics.per_policy["linucb"].mean_reward
    )


def test_csv_has_policy_category_column(tmp_path: Path) -> None:
    metrics = _minimal_metrics()
    path = metrics_to_csv(metrics, tmp_path / "m.csv")
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        categories = {row["policy_category"] for row in reader}
    assert "bandit" in categories
    assert "oracle" in categories
    assert "naive_baseline" in categories


def test_plot_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    metrics = _minimal_metrics()
    out = plot_learning_curve(metrics, tmp_path / "curve.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_uses_agg_backend_no_display(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    metrics = _minimal_metrics()
    plot_learning_curve(
        metrics,
        tmp_path / "headless.png",
        title="Test Plot",
    )


def test_plot_title_includes_honesty_caption(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    assert "Synthetic" in HONESTY_CAPTION
    assert "real-world" in HONESTY_CAPTION
