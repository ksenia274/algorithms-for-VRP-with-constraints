from __future__ import annotations

from pathlib import Path

import pandas as pd

from visualization.plots.pareto_cv import (
    plot_pareto_cv,
    plot_pareto_cv_aggregated,
    plot_pareto_cv_single,
)
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
from visualization.plots.route_profile import (
    plot_route_profile,
    plot_route_profile_aggregated,
    plot_route_profile_single,
)
from visualization.plots.heatmap_2x2 import plot_heatmap_2x2


# ── Main new API ───────────────────────────────────────────────────────────────

def plot_benchmark(
    df: pd.DataFrame,
    plots_dir: Path,
    *,
    group_col: str = "algorithm",
    bench_dir: Path | None = None,
) -> dict[str, Path]:
    """Build aggregated/ and per_instance/ plots for a benchmark.

    aggregated/ : 01/02/03/04/05 as single PNGs
    per_instance/ : 01/02/03/05 as directories of per-instance PNGs
    """
    plots_dir = Path(plots_dir)
    agg_dir = plots_dir / "aggregated"
    pi_dir  = plots_dir / "per_instance"
    agg_dir.mkdir(parents=True, exist_ok=True)
    pi_dir.mkdir(parents=True, exist_ok=True)

    # ── aggregated ────────────────────────────────────────────────────────
    plot_pareto_gini_aggregated(df, agg_dir / "01_pareto_gini.png", group_col=group_col)
    plot_pareto_worst_mean_aggregated(df, agg_dir / "02_pareto_worst_mean.png", group_col=group_col)
    plot_pareto_cv_aggregated(df, agg_dir / "03_pareto_cv.png", group_col=group_col)
    plot_heatmap_2x2(df, agg_dir / "04_heatmap.png", group_col=group_col)
    if bench_dir is not None:
        plot_route_profile_aggregated(bench_dir, agg_dir / "05_route_profile.png")
    else:
        plot_route_profile(df, agg_dir / "05_route_profile", group_col=group_col)

    # ── per instance ──────────────────────────────────────────────────────
    plot_pareto_gini(df, pi_dir / "01_pareto_gini", group_col=group_col)
    plot_pareto_worst_mean(df, pi_dir / "02_pareto_worst_mean", group_col=group_col)
    plot_pareto_cv(df, pi_dir / "03_pareto_cv", group_col=group_col)
    plot_route_profile(df, pi_dir / "05_route_profile", group_col=group_col)

    return {"aggregated": agg_dir, "per_instance": pi_dir}


def plot_single_run(result, plots_dir: Path) -> dict[str, Path]:
    """Build 01/02/03/05 PNGs for a single SolverResult.

    Skips fairness plots when result is infeasible or has no fairness data.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "01_pareto_gini":       plots_dir / "01_pareto_gini.png",
        "02_pareto_worst_mean": plots_dir / "02_pareto_worst_mean.png",
        "03_pareto_cv":         plots_dir / "03_pareto_cv.png",
        "05_route_profile":     plots_dir / "05_route_profile.png",
    }
    plot_pareto_gini_single(result, paths["01_pareto_gini"])
    plot_pareto_worst_mean_single(result, paths["02_pareto_worst_mean"])
    plot_pareto_cv_single(result, paths["03_pareto_cv"])
    plot_route_profile_single(result, paths["05_route_profile"])

    return {k: v for k, v in paths.items() if v.exists()}


# ── Backward-compat: old plot_all ─────────────────────────────────────────────

def plot_all(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    group_col: str = "algorithm",
) -> dict[str, Path]:
    """Legacy orchestrator kept for backward compatibility.

    Produces the old 01–07 flat layout under output_dir.
    """
    from visualization.plots.pareto import plot_pareto
    from visualization.plots.dimensions import plot_dimensions
    from visualization.plots.category_heatmap import plot_category_heatmap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {
        "01_pareto":            output_dir / "01_pareto.png",
        "02_dimensions":        output_dir / "02_dimensions.png",
        "03_category_heatmap":  output_dir / "03_category_heatmap.png",
        "04_pareto_gini":       output_dir / "04_pareto_gini",
        "05_pareto_worst_mean": output_dir / "05_pareto_worst_mean",
        "06_pareto_cv":         output_dir / "06_pareto_cv",
        "07_route_profile":     output_dir / "07_route_profile",
    }
    plot_pareto(df, paths["01_pareto"], group_col=group_col)
    plot_dimensions(df, paths["02_dimensions"], group_col=group_col)
    plot_category_heatmap(df, paths["03_category_heatmap"], group_col=group_col)
    plot_pareto_gini(df, paths["04_pareto_gini"], group_col=group_col)
    plot_pareto_worst_mean(df, paths["05_pareto_worst_mean"], group_col=group_col)
    plot_pareto_cv(df, paths["06_pareto_cv"], group_col=group_col)
    plot_route_profile(df, paths["07_route_profile"], group_col=group_col)
    return paths
