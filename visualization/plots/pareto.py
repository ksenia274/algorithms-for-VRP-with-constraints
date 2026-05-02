from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from visualization._style import COLORS, save_fig, style_ax, group_colors
from visualization.utils import validate_metrics_csv

_REQUIRED = ["total_distance", "feasible"]


def _pareto_frontier(xs, ys):
    pts = sorted(zip(xs, ys))
    frontier, best_y = [], float("inf")
    for x, y in pts:
        if y <= best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def plot_pareto(
    df: pd.DataFrame,
    output_path: Path,
    *,
    primary_metric: Optional[str] = None,
    group_col: str = "algorithm",
) -> None:
    """Scatter total_distance vs primary_metric, one point per group value."""
    if primary_metric is None:
        from runtime.global_config import get_global_config
        primary_metric = get_global_config().metrics.primary

    validate_metrics_csv(df, _REQUIRED + [primary_metric])

    feas = df[df["feasible"].astype(str).str.lower() == "true"].copy()
    groups = sorted(feas[group_col].unique()) if group_col in feas.columns else ["all"]
    colors = group_colors(groups)

    fig, ax = plt.subplots(figsize=(8, 6))
    style_ax(
        ax,
        title=f"Cost vs. {primary_metric}  (lower-left = better)",
        ylabel=primary_metric,
        xlabel="Median total distance",
    )
    ax.axhline(1.0, color=COLORS["ref"], linewidth=1.0, linestyle="--",
               alpha=0.7, zorder=1, label="ideal = 1.0")

    xs_all, ys_all = [], []
    for group in groups:
        sub = feas[feas[group_col] == group] if group_col in feas.columns else feas
        med_d = sub["total_distance"].median()
        med_m = sub[primary_metric].median()
        xs_all.append(med_d)
        ys_all.append(med_m)
        color = colors[group]
        ax.scatter(med_d, med_m, s=80 + len(sub) * 4, color=color,
                   zorder=5, edgecolors="white", linewidths=1.2)
        ax.annotate(f"{group}", (med_d, med_m), xytext=(8, 4),
                    textcoords="offset points", fontsize=9, color=color)

    if len(xs_all) > 1:
        frontier = _pareto_frontier(xs_all, ys_all)
        if len(frontier) > 1:
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, color=COLORS["muted"], linewidth=1.2,
                    linestyle=":", alpha=0.7, zorder=3, label="Pareto frontier")

    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save_fig(fig, output_path)
