from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from visualization._pareto_base import _plot_pareto, compute_aggregated_pareto
from visualization._style import group_colors, save_fig

_Y_METRIC = "dist_cv"
_Y_LABEL  = "Coefficient of variation (CV)"


def plot_pareto_cv(
    df: pd.DataFrame,
    output_path: Path,
    *,
    primary_metric: str | None = None,
    group_col: str = "algorithm",
) -> None:
    """Scatter total_distance vs dist_cv per instance. output_path is a directory."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    feas = df[df["feasible"].astype(str).str.lower() == "true"].copy()
    colors = group_colors(sorted(feas[group_col].unique()) if group_col in feas.columns else [])

    for instance in sorted(feas["instance"].unique()):
        sub = feas[feas["instance"] == instance]
        fig, ax = plt.subplots()
        _plot_pareto(
            ax,
            list(sub["total_distance"]),
            list(sub[_Y_METRIC]) if _Y_METRIC in sub.columns else [0.0] * len(sub),
            list(sub[group_col]),
            colors,
            y_label=_Y_LABEL,
        )
        ax.set_title(f"Pareto: instance {instance}")
        fig.tight_layout()
        save_fig(fig, output_path / f"instance_{instance}.png")


def plot_pareto_cv_aggregated(
    df: pd.DataFrame,
    output_path: Path,
    *,
    group_col: str = "algorithm",
) -> None:
    """Single PNG: mean rel_distance vs mean CV, one point per algorithm."""
    output_path = Path(output_path)
    feas = df[df["feasible"].astype(str).str.lower() == "true"]
    colors = group_colors(sorted(feas[group_col].unique()) if group_col in feas.columns else [])
    x_vals, y_vals, labels = compute_aggregated_pareto(df, _Y_METRIC, group_col)
    fig, ax = plt.subplots()
    _plot_pareto(
        ax, x_vals, y_vals, labels, colors,
        x_label="Relative distance (vs best per instance)",
        y_label=_Y_LABEL,
        annotate_labels=True,
    )
    ax.set_title("Pareto: CV (aggregated)")
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_pareto_cv_single(result, output_path: Path) -> None:
    """Single PNG for one SolverResult."""
    output_path = Path(output_path)
    if not result.feasible or result.fairness is None:
        return
    algo = result.config.algorithm
    colors = group_colors([algo])
    fig, ax = plt.subplots()
    _plot_pareto(
        ax,
        [result.total_distance],
        [result.fairness.distance.cv],
        [algo],
        colors,
        y_label=_Y_LABEL,
    )
    ax.set_title(f"Pareto — {result.config.instance}")
    fig.tight_layout()
    save_fig(fig, output_path)
