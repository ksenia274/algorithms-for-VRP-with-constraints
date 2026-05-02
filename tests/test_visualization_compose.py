from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from visualization.compose import plot_all
from runtime.global_config import GlobalConfig, MetricsConfig, set_global_config_for_testing, reset_global_config


@pytest.fixture(autouse=True)
def _cfg():
    set_global_config_for_testing(GlobalConfig(metrics=MetricsConfig(primary="dist_worst_ratio")))
    yield
    reset_global_config()


def _make_df(algorithms=("hgs_simple",), n_per_alg: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for algo in algorithms:
        for i in range(n_per_alg):
            rows.append({
                "algorithm": algo,
                "benchmark_name": f"bench_{algo[:4]}",
                "instance": str(i),
                "instance_kind": "yandex",
                "category": "yandex",
                "feasible": "True",
                "total_distance": rng.uniform(400_000, 800_000),
                "dist_worst_ratio": rng.uniform(1.0, 2.0),
                "dist_gini": rng.uniform(0.0, 0.3),
                "load_worst_ratio": rng.uniform(1.0, 1.5),
                "load_gini": rng.uniform(0.0, 0.2),
                "clients_worst_ratio": rng.uniform(1.0, 1.5),
                "clients_gini": rng.uniform(0.0, 0.2),
            })
    return pd.DataFrame(rows)


def test_plot_all_creates_four_files(tmp_path):
    df = _make_df()
    paths = plot_all(df, tmp_path / "plots")
    assert len(paths) == 4
    for name, path in paths.items():
        assert path.exists(), f"{name} not created at {path}"
        assert path.stat().st_size > 0


def test_plot_all_with_compare_group_col(tmp_path):
    df = _make_df(algorithms=["hgs_simple", "hgs_rebalance"], n_per_alg=4)
    paths = plot_all(df, tmp_path / "compare_plots", group_col="benchmark_name")
    assert len(paths) == 4
    for path in paths.values():
        assert path.exists()
