from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._style import CATEGORY_COLORS, CATEGORY_ORDER, COLORS, save_fig, style_ax
from visualization.utils import add_category_column, validate_metrics_csv

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False


def plot_distribution(
    df: pd.DataFrame,
    output_path: Path,
    *,
    primary_metric: Optional[str] = None,
    metric_names: Optional[list[str]] = None,
    group_col: str = "algorithm",
    split_by: Optional[str] = "category",
) -> None:
    """Violin + jitter per group_col value, optionally split/coloured by split_by.

    Default (group_col='algorithm', split_by='category'):
        x = algorithm, colour = instance category — shows how each algorithm
        handles different problem types.
    group_col='category', split_by='algorithm':
        x = category, colour = algorithm — comparison across problem types.
    split_by=None:
        x = group_col, no colour split — plain distribution per group.
    """
    if primary_metric is None:
        from runtime.global_config import get_global_config
        cfg = get_global_config()
        primary_metric = cfg.metrics.primary
        metrics = metric_names if metric_names is not None else ([primary_metric] + cfg.metrics.secondary)
    else:
        metrics = metric_names if metric_names is not None else [primary_metric]

    validate_metrics_csv(df, [m for m in metrics if m in df.columns or m == primary_metric])

    feas = df[df["feasible"].astype(str).str.lower() == "true"].copy()
    if "category" not in feas.columns:
        feas = add_category_column(feas)

    groups = sorted(feas[group_col].unique()) if group_col in feas.columns else ["all"]

    if split_by is not None and split_by in feas.columns:
        splits = sorted(feas[split_by].unique())
    else:
        splits = None

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(max(7, 4 * n_metrics), 6), squeeze=False)
    fig.set_facecolor("white")
    rng = np.random.default_rng(0)

    for mi, metric in enumerate(metrics):
        ax = axes[0, mi]
        style_ax(ax, title=metric, ylabel=metric if mi == 0 else "")

        if metric not in feas.columns:
            ax.text(0.5, 0.5, f"{metric}\nnot in data",
                    transform=ax.transAxes, ha="center", va="center", color=COLORS["muted"])
            continue

        if _HAS_SNS and groups:
            rows = []
            for g in groups:
                sub_g = feas[feas[group_col] == g] if group_col in feas.columns else feas
                for _, row in sub_g.iterrows():
                    entry = {group_col: g, "value": row[metric]}
                    if splits is not None:
                        entry[split_by] = row.get(split_by, "other")
                    rows.append(entry)
            if rows:
                tidy = pd.DataFrame(rows)
                hue = split_by if splits is not None else None
                palette = None
                if hue == "category":
                    palette = {c: CATEGORY_COLORS.get(c, COLORS["muted"]) for c in splits}
                sns.violinplot(
                    data=tidy, x=group_col, y="value",
                    hue=hue, palette=palette,
                    inner=None, ax=ax, order=groups,
                    cut=0, linewidth=1.0, alpha=0.6, legend=False,
                )
                for xi, g in enumerate(groups):
                    sub_g = tidy[tidy[group_col] == g]
                    if splits is not None:
                        for sp in splits:
                            vals = sub_g[sub_g[split_by] == sp]["value"].values
                            color = (CATEGORY_COLORS.get(sp, COLORS["muted"])
                                     if hue == "category" else COLORS["muted"])
                            jitter = rng.uniform(-0.12, 0.12, len(vals))
                            ax.scatter(xi + jitter, vals, s=20, color=color,
                                       alpha=0.7, zorder=4, linewidths=0)
                    else:
                        vals = sub_g["value"].values
                        jitter = rng.uniform(-0.12, 0.12, len(vals))
                        ax.scatter(xi + jitter, vals, s=20, color=COLORS["muted"],
                                   alpha=0.7, zorder=4, linewidths=0)
        else:
            for xi, g in enumerate(groups):
                sub_g = feas[feas[group_col] == g] if group_col in feas.columns else feas
                vals = sub_g[metric].dropna().values
                if len(vals) < 2:
                    continue
                color = CATEGORY_COLORS.get(g, COLORS["muted"])
                vp = ax.violinplot(vals, positions=[xi], widths=0.6,
                                   showmedians=False, showextrema=False)
                for pc in vp["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.55)
                jitter = rng.uniform(-0.1, 0.1, len(vals))
                ax.scatter(xi + jitter, vals, s=20, color=color,
                           alpha=0.7, zorder=4, linewidths=0)

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=15, ha="right")
        ax.set_xlabel(group_col)

    if splits is not None and split_by == "category":
        handles = [
            mpatches.Patch(color=CATEGORY_COLORS.get(c, COLORS["muted"]), label=c)
            for c in splits if c in CATEGORY_COLORS
        ]
        if handles:
            axes[0, -1].legend(handles=handles, title=split_by,
                               loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    save_fig(fig, output_path)
