from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from visualization.utils import _categorize, validate_metrics_csv
from visualization.plots.pareto import plot_pareto
from visualization.plots.dimensions import plot_dimensions
from visualization.plots.distribution import plot_distribution
from visualization.plots.category_heatmap import plot_category_heatmap
from runtime.global_config import GlobalConfig, MetricsConfig, set_global_config_for_testing, reset_global_config


@pytest.fixture(autouse=True)
def _cfg():
    set_global_config_for_testing(GlobalConfig(metrics=MetricsConfig(primary="dist_worst_ratio")))
    yield
    reset_global_config()


def _make_df(n: int = 5, category: str = "yandex") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "algorithm": ["hgs_simple"] * n,
        "instance": [str(i) for i in range(n)],
        "instance_kind": ["yandex"] * n,
        "category": [category] * n,
        "feasible": ["True"] * n,
        "total_distance": rng.uniform(400_000, 800_000, n),
        "num_routes": rng.integers(10, 15, n),
        "dist_worst_ratio": rng.uniform(1.0, 2.0, n),
        "dist_gini": rng.uniform(0.0, 0.3, n),
        "load_worst_ratio": rng.uniform(1.0, 1.5, n),
        "load_gini": rng.uniform(0.0, 0.2, n),
        "clients_worst_ratio": rng.uniform(1.0, 1.5, n),
        "clients_gini": rng.uniform(0.0, 0.2, n),
    })


def test_pareto_smoke(tmp_path):
    df = _make_df()
    out = tmp_path / "pareto.png"
    plot_pareto(df, out)
    assert out.exists() and out.stat().st_size > 0


def test_dimensions_smoke(tmp_path):
    df = _make_df()
    out = tmp_path / "dims.png"
    plot_dimensions(df, out)
    assert out.exists() and out.stat().st_size > 0


def test_distribution_smoke(tmp_path):
    df = _make_df()
    out = tmp_path / "dist.png"
    plot_distribution(df, out)
    assert out.exists() and out.stat().st_size > 0


def test_distribution_with_split_off(tmp_path):
    df = _make_df()
    out = tmp_path / "dist_nosplit.png"
    plot_distribution(df, out, split_by=None)
    assert out.exists() and out.stat().st_size > 0


def test_distribution_grouped_by_category(tmp_path):
    rng = np.random.default_rng(2)
    rows = []
    for algo in ["hgs_simple", "alns"]:
        for cat in ["yandex", "R1"]:
            for _ in range(3):
                rows.append({
                    "algorithm": algo, "category": cat,
                    "instance_kind": "yandex", "feasible": "True",
                    "total_distance": rng.uniform(4e5, 8e5),
                    "dist_worst_ratio": rng.uniform(1.0, 2.0),
                })
    df = pd.DataFrame(rows)
    out = tmp_path / "dist_by_cat.png"
    plot_distribution(df, out, group_col="category", split_by="algorithm")
    assert out.exists() and out.stat().st_size > 0


def test_heatmap_smoke(tmp_path):
    rng = np.random.default_rng(1)
    # Need >= 3 rows per (algorithm, category) for min_n
    rows = []
    for algo in ["hgs_simple", "hgs_rebalance"]:
        for cat in ["yandex", "R1"]:
            for _ in range(4):
                rows.append({
                    "algorithm": algo, "category": cat,
                    "instance_kind": "yandex" if cat == "yandex" else "solomon",
                    "feasible": "True",
                    "total_distance": rng.uniform(400_000, 800_000),
                    "dist_worst_ratio": rng.uniform(1.0, 2.0),
                })
    df = pd.DataFrame(rows)
    out = tmp_path / "heatmap.png"
    plot_category_heatmap(df, out, min_n=3)
    assert out.exists() and out.stat().st_size > 0


def test_validate_missing_columns():
    df = pd.DataFrame({"algorithm": ["x"], "feasible": ["True"]})
    with pytest.raises(ValueError, match="Missing columns"):
        validate_metrics_csv(df, ["dist_worst_ratio"])


def test_validate_missing_columns_message():
    df = pd.DataFrame({"algorithm": ["x"]})
    with pytest.raises(ValueError, match="pre-refactor"):
        validate_metrics_csv(df, ["total_distance"])


def test_categorize_yandex_returns_yandex():
    assert _categorize("3", "yandex") == "yandex"
    assert _categorize("03", "yandex") == "yandex"


def test_categorize_solomon_R101_returns_R1():
    assert _categorize("R101", "solomon") == "R1"


def test_categorize_solomon_C201_returns_C2():
    assert _categorize("C201", "solomon") == "C2"


def test_categorize_solomon_RC105_returns_RC1():
    assert _categorize("RC105", "solomon") == "RC1"
