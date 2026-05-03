from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from visualization._style import group_colors, save_fig

_PROFILE_METRICS = ["total_distance", "dist_gini", "dist_cv", "dist_worst_ratio"]
_PROFILE_LABELS  = ["distance",       "gini",       "cv",      "worst_mean"]


def plot_route_profile(
    df: pd.DataFrame,
    output_path: Path,
    *,
    primary_metric: str | None = None,
    group_col: str = "algorithm",
) -> None:
    """MinMax-normalized metric profile lines per instance. output_path is a directory."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    feas = df[df["feasible"].astype(str).str.lower() == "true"].copy()

    for instance in sorted(feas["instance"].unique()):
        sub = feas[feas["instance"] == instance].copy()

        available = [c for c in _PROFILE_METRICS if c in sub.columns]
        labels    = [_PROFILE_LABELS[_PROFILE_METRICS.index(c)] for c in available]
        if not available:
            continue

        vals = sub[available].values.astype(float)
        if vals.shape[0] >= 2:
            sub[available] = MinMaxScaler().fit_transform(vals)

        fig, ax = plt.subplots()

        for method in sorted(sub[group_col].unique()):
            d = sub[sub[group_col] == method]
            row_vals = d[available].mean().values
            ax.plot(labels, row_vals, marker="o", label=str(method))

        ax.set_title(f"Profile plot (instance {instance})")
        ax.set_ylabel("Normalized value")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        save_fig(fig, output_path / f"instance_{instance}.png")


def plot_route_profile_single(result, output_path: Path) -> None:
    """Bar chart: clients per route sorted descending, for one SolverResult."""
    output_path = Path(output_path)
    if not result.feasible or not result.routes:
        return
    counts = sorted([len(r) for r in result.routes], reverse=True)
    fig, ax = plt.subplots()
    ax.bar(range(len(counts)), counts, color="#4A6FA5")
    ax.set_xlabel("Route (sorted by size, descending)")
    ax.set_ylabel("Clients per route")
    ax.set_title(f"Route profile — {result.config.instance}")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_route_profile_aggregated(bench_dir: Path, output_path: Path) -> None:
    """Line per algorithm: average sorted client-count profile across all benchmark runs.

    Routes are sorted by descending client count within each solution, position
    normalised to [0, 1] (relative rank). Profiles are interpolated to a common
    50-point grid then averaged across all runs of the same algorithm.
    """
    bench_dir = Path(bench_dir)
    runs_dir  = bench_dir / "runs"
    if not runs_dir.exists():
        return

    algo_profiles: dict[str, list[list[int]]] = {}
    for run_dir in sorted(runs_dir.iterdir()):
        result_json = run_dir / "result.json"
        if not result_json.exists():
            continue
        with open(result_json, encoding="utf-8") as fh:
            data = json.load(fh)
        if not data.get("feasible", False):
            continue
        algo   = data["config"]["algorithm"]
        routes = data.get("routes", [])
        if not routes:
            continue
        counts = sorted([len(r) for r in routes], reverse=True)
        algo_profiles.setdefault(algo, []).append(counts)

    if not algo_profiles:
        return

    grid   = np.linspace(0, 1, 50)
    colors = group_colors(sorted(algo_profiles.keys()))
    fig, ax = plt.subplots()

    for algo in sorted(algo_profiles):
        interp_profiles = []
        for counts in algo_profiles[algo]:
            n = len(counts)
            x_orig = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
            interp_profiles.append(np.interp(grid, x_orig, counts))
        mean_profile = np.mean(interp_profiles, axis=0)
        ax.plot(grid, mean_profile, label=algo, color=colors.get(algo))

    ax.set_xlabel("Relative rank (0 = largest route, 1 = smallest)")
    ax.set_ylabel("Clients per route")
    ax.set_title("Route profile (aggregated)")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, output_path)
