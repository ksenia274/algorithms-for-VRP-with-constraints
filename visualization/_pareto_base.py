from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pandas as pd


def _plot_pareto(
    ax,
    x_vals: list,
    y_vals: list,
    labels: list,
    colors: dict[str, str],
    *,
    x_label: str = "Total distance (cost)",
    y_label: str = "",
    annotate_labels: bool = False,
) -> None:
    """Draw Pareto scatter on ax. One series per unique label."""
    for lbl in sorted(set(labels)):
        xs = [x for x, l in zip(x_vals, labels) if l == lbl]
        ys = [y for y, l in zip(y_vals, labels) if l == lbl]
        ax.scatter(xs, ys, label=lbl, color=colors.get(lbl), s=80, alpha=0.85)
        if annotate_labels:
            for x, y in zip(xs, ys):
                ax.annotate(lbl, (x, y), textcoords="offset points",
                            xytext=(5, 5), fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.4)
    if labels:
        ax.legend()


def compute_aggregated_pareto(
    df: pd.DataFrame,
    y_metric: str,
    group_col: str = "algorithm",
) -> tuple[list, list, list]:
    """Compute per-group mean rel_distance and mean y_metric.

    rel_distance = distance / best_distance_for_instance,
    where best = min total_distance across all runs of that instance.

    Returns (x_vals, y_vals, labels).
    """
    feas = df[df["feasible"].astype(str).str.lower() == "true"].copy()
    if feas.empty or y_metric not in feas.columns:
        return [], [], []
    best = feas.groupby("instance")["total_distance"].transform("min")
    feas["_rel_dist"] = feas["total_distance"] / best
    agg = (
        feas.groupby(group_col, sort=True)
        .agg(mean_rel_dist=("_rel_dist", "mean"), mean_y=(y_metric, "mean"))
        .reset_index()
    )
    return list(agg["mean_rel_dist"]), list(agg["mean_y"]), list(agg[group_col])
