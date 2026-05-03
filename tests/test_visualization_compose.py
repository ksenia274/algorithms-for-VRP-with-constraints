from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from visualization.compose import plot_all, plot_benchmark, plot_single_run
from runtime.global_config import (
    GlobalConfig, MetricsConfig,
    set_global_config_for_testing, reset_global_config,
)


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
                "total_distance": float(rng.uniform(400_000, 800_000)),
                "dist_worst_ratio": float(rng.uniform(1.0, 2.0)),
                "dist_gini": float(rng.uniform(0.0, 0.3)),
                "dist_cv": float(rng.uniform(0.0, 0.5)),
                "load_worst_ratio": float(rng.uniform(1.0, 1.5)),
                "load_gini": float(rng.uniform(0.0, 0.2)),
                "clients_worst_ratio": float(rng.uniform(1.0, 1.5)),
                "clients_gini": float(rng.uniform(0.0, 0.2)),
            })
    return pd.DataFrame(rows)


# ── Legacy plot_all ────────────────────────────────────────────────────────────

def test_plot_all_creates_files(tmp_path):
    df = _make_df()
    paths = plot_all(df, tmp_path / "plots")
    assert len(paths) == 7
    for name, path in paths.items():
        assert path.exists(), f"{name} not created at {path}"
        if path.is_file():
            assert path.stat().st_size > 0
        else:
            assert any(path.glob("*.png")), f"{name}: no PNG files in {path}"


def test_plot_all_with_compare_group_col(tmp_path):
    df = _make_df(algorithms=["hgs_simple", "hgs_rebalance"], n_per_alg=4)
    paths = plot_all(df, tmp_path / "compare_plots", group_col="benchmark_name")
    assert len(paths) == 7
    for name, path in paths.items():
        assert path.exists(), f"{name} not created at {path}"


# ── New plot_benchmark ─────────────────────────────────────────────────────────

def test_plot_benchmark_creates_aggregated_and_per_instance(tmp_path):
    df = _make_df(algorithms=("alg_a", "alg_b"), n_per_alg=3)
    paths = plot_benchmark(df, tmp_path / "bench_plots")
    assert "aggregated"   in paths
    assert "per_instance" in paths
    assert paths["aggregated"].is_dir()
    assert paths["per_instance"].is_dir()


def test_plot_benchmark_aggregated_has_four_pngs(tmp_path):
    df = _make_df(algorithms=("alg_a", "alg_b"), n_per_alg=3)
    paths = plot_benchmark(df, tmp_path / "bench_plots")
    agg_pngs = list(paths["aggregated"].glob("*.png"))
    # 01/02/03/04 are single PNGs; 05 may or may not exist (no bench_dir passed → uses dir)
    assert len(agg_pngs) >= 4, f"Expected ≥4 aggregated PNGs, got {[p.name for p in agg_pngs]}"


def test_plot_benchmark_per_instance_has_subdirs(tmp_path):
    df = _make_df(algorithms=("alg_a", "alg_b"), n_per_alg=3)
    paths = plot_benchmark(df, tmp_path / "bench_plots")
    pi_dirs = [p for p in paths["per_instance"].iterdir() if p.is_dir()]
    assert len(pi_dirs) >= 3, f"Expected ≥3 per_instance subdirs, got {[p.name for p in pi_dirs]}"


def test_plot_benchmark_with_bench_dir_aggregated_05(tmp_path):
    """When bench_dir is supplied with result.json files, 05 aggregated PNG must be created."""
    # Build a fake benchmark dir structure
    bench_dir = tmp_path / "bench"
    runs_dir  = bench_dir / "runs"

    for i, algo in enumerate(["alg_a", "alg_b"]):
        run_dir = runs_dir / f"run_{i:02d}"
        run_dir.mkdir(parents=True)
        result = {
            "feasible": True,
            "routes": [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
            "config": {"algorithm": algo, "instance": f"inst_{i}"},
        }
        (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    df = _make_df(algorithms=("alg_a", "alg_b"), n_per_alg=2)
    paths = plot_benchmark(df, bench_dir / "plots", bench_dir=bench_dir)
    out05 = paths["aggregated"] / "05_route_profile.png"
    assert out05.exists() and out05.stat().st_size > 0


def test_plot_benchmark_empty_feasible_no_crash(tmp_path):
    """All-infeasible DataFrame must not raise."""
    df = _make_df(n_per_alg=3)
    df["feasible"] = "False"
    plot_benchmark(df, tmp_path / "bench_plots")  # should not raise


# ── New plot_single_run ────────────────────────────────────────────────────────

class _FakeDim:
    def __init__(self):
        self.gini = 0.1
        self.worst_ratio = 1.3
        self.cv = 0.2


class _FakeFairness:
    def __init__(self):
        self.distance = _FakeDim()


class _FakeConfig:
    algorithm = "fake_algo"
    instance  = "inst_0"


class _FakeResult:
    feasible       = True
    routes         = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    total_distance = 500_000.0
    fairness       = _FakeFairness()
    config         = _FakeConfig()


def test_plot_single_run_creates_four_pngs(tmp_path):
    result = _FakeResult()
    paths = plot_single_run(result, tmp_path / "plots")
    assert len(paths) == 4
    for name, p in paths.items():
        assert p.exists() and p.stat().st_size > 0, f"{name} not created"


def test_plot_single_run_correct_names(tmp_path):
    paths = plot_single_run(_FakeResult(), tmp_path / "plots")
    names = set(paths.keys())
    assert "01_pareto_gini"       in names
    assert "02_pareto_worst_mean" in names
    assert "03_pareto_cv"         in names
    assert "05_route_profile"     in names
