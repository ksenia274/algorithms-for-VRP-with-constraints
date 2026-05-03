from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from visualization.utils import _categorize, validate_metrics_csv
from visualization.plots.pareto import plot_pareto
from visualization.plots.dimensions import plot_dimensions
from visualization.plots.category_heatmap import plot_category_heatmap
from visualization.plots.pareto_gini import (
    plot_pareto_gini,
    plot_pareto_gini_aggregated,
    plot_pareto_gini_single,
)
from visualization.plots.pareto_worst_mean import (
    plot_pareto_worst_mean,
    plot_pareto_worst_mean_aggregated,
    plot_pareto_worst_mean_single,
)
from visualization.plots.pareto_cv import (
    plot_pareto_cv,
    plot_pareto_cv_aggregated,
    plot_pareto_cv_single,
)
from visualization.plots.route_profile import (
    plot_route_profile,
    plot_route_profile_single,
)
from visualization.plots.heatmap_2x2 import plot_heatmap_2x2
from runtime.global_config import (
    GlobalConfig, MetricsConfig,
    set_global_config_for_testing, reset_global_config,
)


@pytest.fixture(autouse=True)
def _cfg():
    set_global_config_for_testing(GlobalConfig(metrics=MetricsConfig(primary="dist_worst_ratio")))
    yield
    reset_global_config()


def _make_df(n: int = 5, category: str = "yandex",
             algorithms=("hgs_simple",)) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for algo in algorithms:
        for i in range(n):
            rows.append({
                "algorithm": algo,
                "instance": str(i),
                "instance_kind": "yandex",
                "category": category,
                "feasible": "True",
                "total_distance": float(rng.uniform(400_000, 800_000)),
                "num_routes": int(rng.integers(10, 15)),
                "dist_worst_ratio": float(rng.uniform(1.0, 2.0)),
                "dist_gini": float(rng.uniform(0.0, 0.3)),
                "dist_cv": float(rng.uniform(0.0, 0.5)),
                "load_worst_ratio": float(rng.uniform(1.0, 1.5)),
                "load_gini": float(rng.uniform(0.0, 0.2)),
                "clients_worst_ratio": float(rng.uniform(1.0, 1.5)),
                "clients_gini": float(rng.uniform(0.0, 0.2)),
            })
    return pd.DataFrame(rows)


# ── legacy plots (unchanged) ───────────────────────────────────────────────────

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


def test_heatmap_smoke(tmp_path):
    rng = np.random.default_rng(1)
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


# ── per-instance Pareto plots ──────────────────────────────────────────────────

def test_pareto_gini_smoke(tmp_path):
    df = _make_df()
    out = tmp_path / "pareto_gini"
    plot_pareto_gini(df, out)
    pngs = list(out.glob("*.png"))
    assert len(pngs) > 0 and all(p.stat().st_size > 0 for p in pngs)


def test_pareto_worst_mean_smoke(tmp_path):
    df = _make_df()
    out = tmp_path / "pareto_wm"
    plot_pareto_worst_mean(df, out)
    pngs = list(out.glob("*.png"))
    assert len(pngs) > 0 and all(p.stat().st_size > 0 for p in pngs)


def test_pareto_cv_smoke(tmp_path):
    df = _make_df()
    out = tmp_path / "pareto_cv"
    plot_pareto_cv(df, out)
    pngs = list(out.glob("*.png"))
    assert len(pngs) > 0 and all(p.stat().st_size > 0 for p in pngs)


def test_route_profile_smoke(tmp_path):
    df = _make_df()
    out = tmp_path / "profile"
    plot_route_profile(df, out)
    pngs = list(out.glob("*.png"))
    assert len(pngs) > 0 and all(p.stat().st_size > 0 for p in pngs)


# ── aggregated Pareto plots ────────────────────────────────────────────────────

def test_pareto_gini_aggregated_smoke(tmp_path):
    df = _make_df(n=3, algorithms=("alg_a", "alg_b"))
    out = tmp_path / "agg_gini.png"
    plot_pareto_gini_aggregated(df, out)
    assert out.exists() and out.stat().st_size > 0


def test_pareto_worst_mean_aggregated_smoke(tmp_path):
    df = _make_df(n=3, algorithms=("alg_a", "alg_b"))
    out = tmp_path / "agg_wm.png"
    plot_pareto_worst_mean_aggregated(df, out)
    assert out.exists() and out.stat().st_size > 0


def test_pareto_cv_aggregated_smoke(tmp_path):
    df = _make_df(n=3, algorithms=("alg_a", "alg_b"))
    out = tmp_path / "agg_cv.png"
    plot_pareto_cv_aggregated(df, out)
    assert out.exists() and out.stat().st_size > 0


def test_aggregated_pareto_empty_df_no_crash(tmp_path):
    """Empty DataFrame (all infeasible) must not raise."""
    df = _make_df(n=3)
    df["feasible"] = "False"
    out = tmp_path / "agg_gini_empty.png"
    plot_pareto_gini_aggregated(df, out)  # should not raise


# ── 2×2 heatmap ───────────────────────────────────────────────────────────────

def test_heatmap_2x2_smoke(tmp_path):
    rng = np.random.default_rng(2)
    rows = []
    for algo in ["alg_a", "alg_b"]:
        for cat in ["yandex", "R1"]:
            for _ in range(3):
                rows.append({
                    "algorithm": algo, "category": cat,
                    "feasible": "True",
                    "total_distance": float(rng.uniform(400_000, 800_000)),
                    "dist_worst_ratio": float(rng.uniform(1.0, 2.0)),
                    "dist_gini": float(rng.uniform(0.0, 0.3)),
                    "dist_cv": float(rng.uniform(0.0, 0.5)),
                })
    df = pd.DataFrame(rows)
    out = tmp_path / "heatmap2x2.png"
    plot_heatmap_2x2(df, out)
    assert out.exists() and out.stat().st_size > 0


# ── single-run plot stubs ──────────────────────────────────────────────────────

class _FakeDim:
    def __init__(self, gini=0.1, worst_ratio=1.3, cv=0.2):
        self.gini = gini
        self.worst_ratio = worst_ratio
        self.cv = cv


class _FakeFairness:
    def __init__(self):
        self.distance = _FakeDim()


class _FakeConfig:
    algorithm = "fake_algo"
    instance  = "inst_0"


class _FakeResult:
    feasible   = True
    routes     = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    total_distance = 500_000.0
    fairness   = _FakeFairness()
    config     = _FakeConfig()


def test_pareto_gini_single_smoke(tmp_path):
    out = tmp_path / "single_gini.png"
    plot_pareto_gini_single(_FakeResult(), out)
    assert out.exists() and out.stat().st_size > 0


def test_pareto_worst_mean_single_smoke(tmp_path):
    out = tmp_path / "single_wm.png"
    plot_pareto_worst_mean_single(_FakeResult(), out)
    assert out.exists() and out.stat().st_size > 0


def test_pareto_cv_single_smoke(tmp_path):
    out = tmp_path / "single_cv.png"
    plot_pareto_cv_single(_FakeResult(), out)
    assert out.exists() and out.stat().st_size > 0


def test_route_profile_single_smoke(tmp_path):
    out = tmp_path / "single_profile.png"
    plot_route_profile_single(_FakeResult(), out)
    assert out.exists() and out.stat().st_size > 0


def test_single_infeasible_no_file(tmp_path):
    class _Infeasible(_FakeResult):
        feasible = False
        fairness = None
        routes   = []
    out = tmp_path / "infeasible.png"
    plot_pareto_gini_single(_Infeasible(), out)
    assert not out.exists(), "No file should be created for infeasible result"


# ── validation helpers ─────────────────────────────────────────────────────────

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
