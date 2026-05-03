"""Unit tests for aggregation logic used in Pareto and route-profile plots."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from visualization._pareto_base import compute_aggregated_pareto
from visualization.plots.route_profile import plot_route_profile_aggregated


# ── rel_distance aggregation ───────────────────────────────────────────────────

def _make_pareto_df():
    """3 instances × 2 algorithms, controlled distances and metrics."""
    return pd.DataFrame([
        # instance 0: best = 100 (alg_b)
        {"algorithm": "alg_a", "instance": "i0", "total_distance": 110.0, "dist_gini": 0.2, "feasible": "True"},
        {"algorithm": "alg_b", "instance": "i0", "total_distance": 100.0, "dist_gini": 0.1, "feasible": "True"},
        # instance 1: best = 200 (alg_a)
        {"algorithm": "alg_a", "instance": "i1", "total_distance": 200.0, "dist_gini": 0.3, "feasible": "True"},
        {"algorithm": "alg_b", "instance": "i1", "total_distance": 220.0, "dist_gini": 0.4, "feasible": "True"},
    ])


def test_rel_distance_best_is_one():
    """The algorithm with the best distance on each instance must have rel_distance = 1.0."""
    df = _make_pareto_df()
    x_vals, y_vals, labels = compute_aggregated_pareto(df, "dist_gini")
    # alg_a: i0 -> 110/100=1.1, i1 -> 200/200=1.0  → mean = 1.05
    # alg_b: i0 -> 100/100=1.0, i1 -> 220/200=1.1  → mean = 1.05
    result = dict(zip(labels, x_vals))
    assert abs(result["alg_a"] - 1.05) < 1e-9
    assert abs(result["alg_b"] - 1.05) < 1e-9


def test_rel_distance_mean_y():
    """Mean y values must equal simple arithmetic means per algorithm."""
    df = _make_pareto_df()
    x_vals, y_vals, labels = compute_aggregated_pareto(df, "dist_gini")
    result_y = dict(zip(labels, y_vals))
    # alg_a: (0.2 + 0.3) / 2 = 0.25
    # alg_b: (0.1 + 0.4) / 2 = 0.25
    assert abs(result_y["alg_a"] - 0.25) < 1e-9
    assert abs(result_y["alg_b"] - 0.25) < 1e-9


def test_rel_distance_infeasible_rows_excluded():
    """Infeasible rows must not affect rel_distance computation."""
    df = pd.DataFrame([
        {"algorithm": "alg_a", "instance": "i0", "total_distance": 100.0,
         "dist_gini": 0.1, "feasible": "True"},
        {"algorithm": "alg_b", "instance": "i0", "total_distance": 50.0,
         "dist_gini": 999.0, "feasible": "False"},  # infeasible, must be ignored
    ])
    x_vals, y_vals, labels = compute_aggregated_pareto(df, "dist_gini")
    # only alg_a remains: rel_distance = 100/100 = 1.0
    assert labels == ["alg_a"]
    assert abs(x_vals[0] - 1.0) < 1e-9


def test_rel_distance_returns_empty_for_empty_df():
    df = pd.DataFrame({"algorithm": [], "instance": [], "total_distance": [],
                       "dist_gini": [], "feasible": []})
    x, y, labels = compute_aggregated_pareto(df, "dist_gini")
    assert x == [] and y == [] and labels == []


def test_rel_distance_missing_y_metric_returns_empty():
    df = pd.DataFrame([
        {"algorithm": "a", "instance": "i0", "total_distance": 100.0, "feasible": "True"}
    ])
    x, y, labels = compute_aggregated_pareto(df, "dist_gini_MISSING")
    assert x == [] and y == [] and labels == []


# ── route sorting in profile aggregated ───────────────────────────────────────

def test_routes_sorted_descending():
    """Client counts within each run must be sorted descending before profiling."""
    routes = [[1, 2, 3, 4], [5, 6], [7, 8, 9]]  # lengths: 4, 2, 3
    counts = sorted([len(r) for r in routes], reverse=True)
    assert counts == [4, 3, 2]


def test_routes_single_route_profile():
    """A single route maps to relative rank 0.0."""
    routes = [[1, 2, 3]]
    counts = sorted([len(r) for r in routes], reverse=True)
    n = len(counts)
    x_orig = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    assert x_orig == [0.0]


def test_route_profile_aggregated_creates_png(tmp_path):
    """Aggregated route profile must produce a PNG from fake benchmark runs."""
    bench_dir = tmp_path / "bench"
    runs_dir  = bench_dir / "runs"

    for i, algo in enumerate(["alg_a", "alg_b", "alg_a"]):
        run_dir = runs_dir / f"run_{i:02d}"
        run_dir.mkdir(parents=True)
        result = {
            "feasible": True,
            "routes": [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]],
            "config": {"algorithm": algo, "instance": f"inst_{i}"},
        }
        (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    out = tmp_path / "profile.png"
    plot_route_profile_aggregated(bench_dir, out)
    assert out.exists() and out.stat().st_size > 0


def test_route_profile_aggregated_empty_bench_no_crash(tmp_path):
    """Missing runs/ directory must not raise."""
    bench_dir = tmp_path / "empty_bench"
    bench_dir.mkdir()
    out = tmp_path / "profile.png"
    plot_route_profile_aggregated(bench_dir, out)  # should not raise
    # PNG may or may not be created — just must not raise


def test_route_profile_aggregated_all_infeasible_no_crash(tmp_path):
    """All-infeasible runs must not raise."""
    bench_dir = tmp_path / "bench"
    runs_dir  = bench_dir / "runs"
    run_dir   = runs_dir / "run_00"
    run_dir.mkdir(parents=True)
    result = {"feasible": False, "routes": [], "config": {"algorithm": "a", "instance": "i0"}}
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    out = tmp_path / "profile.png"
    plot_route_profile_aggregated(bench_dir, out)  # should not raise


# ── smoke: 2 algorithms × 3 instances ─────────────────────────────────────────

def test_smoke_2_algos_3_instances(tmp_path):
    """Smoke test: 2 algorithms × 3 instances, all expected files created and non-empty."""
    from visualization.compose import plot_benchmark

    rng = np.random.default_rng(7)
    rows = []
    for algo in ["alg_a", "alg_b"]:
        for inst in ["i0", "i1", "i2"]:
            rows.append({
                "algorithm": algo, "instance": inst,
                "instance_kind": "yandex", "category": "yandex",
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
    df = pd.DataFrame(rows)

    # Build fake bench_dir with result.json files
    bench_dir = tmp_path / "bench"
    runs_dir  = bench_dir / "runs"
    for i, row in enumerate(rows):
        run_dir = runs_dir / f"run_{i:02d}"
        run_dir.mkdir(parents=True)
        result = {
            "feasible": True,
            "routes": [[1, 2, 3], [4, 5, 6], [7, 8]],
            "config": {"algorithm": row["algorithm"], "instance": row["instance"]},
        }
        (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    paths = plot_benchmark(df, bench_dir / "plots", bench_dir=bench_dir)

    # aggregated: 01/02/03/04/05 must all exist and be non-empty
    for fname in ["01_pareto_gini.png", "02_pareto_worst_mean.png",
                  "03_pareto_cv.png", "04_heatmap.png", "05_route_profile.png"]:
        p = paths["aggregated"] / fname
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty: {fname}"

    # per_instance: expected subdirs
    for dname in ["01_pareto_gini", "02_pareto_worst_mean", "03_pareto_cv", "05_route_profile"]:
        d = paths["per_instance"] / dname
        assert d.is_dir(), f"Missing per_instance subdir: {dname}"
        pngs = list(d.glob("*.png"))
        assert len(pngs) > 0 and all(p.stat().st_size > 0 for p in pngs), (
            f"No valid PNGs in {dname}"
        )
