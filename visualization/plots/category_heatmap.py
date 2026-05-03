from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from visualization._style import CATEGORY_ORDER, COLORS, save_fig
from visualization.utils import add_category_column, validate_metrics_csv


def plot_category_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    *,
    primary_metric: Optional[str] = None,
    group_col: str = "algorithm",
    min_n: int = 3,
) -> None:
    """Matrix: group_col × category, cells = mean(primary_metric)."""
    if primary_metric is None:
        from runtime.global_config import get_global_config
        primary_metric = get_global_config().metrics.primary

    validate_metrics_csv(df, [primary_metric])

    feas = df[df["feasible"].astype(str).str.lower() == "true"].copy()
    if "category" not in feas.columns:
        feas = add_category_column(feas)

    groups = sorted(feas[group_col].unique()) if group_col in feas.columns else ["all"]
    present_cats = set(feas["category"].astype(str).unique())
    cats = [c for c in CATEGORY_ORDER if c in present_cats] or sorted(present_cats)

    n_grp, n_cat = len(groups), len(cats)
    data = np.full((n_grp, n_cat), np.nan)
    mask = np.zeros((n_grp, n_cat), dtype=bool)

    for gi, group in enumerate(groups):
        sub_g = feas[feas[group_col] == group] if group_col in feas.columns else feas
        for ci, cat in enumerate(cats):
            sub = sub_g[sub_g["category"].astype(str) == cat][primary_metric].dropna()
            if len(sub) < min_n:
                mask[gi, ci] = True
            else:
                data[gi, ci] = sub.mean()

    fig, ax = plt.subplots(figsize=(max(6, n_cat * 1.5 + 2), max(3, n_grp * 0.9 + 1.8)))
    ax.set_facecolor(COLORS["bg"])
    ax.figure.set_facecolor("white")

    valid = data[~np.isnan(data) & ~mask]
    vmin = float(valid.min()) if len(valid) else 1.0
    vmax = float(valid.max()) if len(valid) else 2.0
    if vmax <= vmin:
        vmax = vmin + 0.1
    cmap = plt.cm.RdYlGn_r
    norm = Normalize(vmin=vmin, vmax=vmax)

    for gi in range(n_grp):
        for ci in range(n_cat):
            if mask[gi, ci] or np.isnan(data[gi, ci]):
                fc, txt, txt_color = "#D0D0D0", f"n<{min_n}", "#999"
            else:
                fc = cmap(norm(data[gi, ci]))
                txt = f"{data[gi, ci]:.3f}"
                lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                txt_color = "white" if lum < 0.5 else COLORS["text"]
            rect = mpatches.FancyBboxPatch(
                (ci - 0.45, gi - 0.4), 0.9, 0.8,
                boxstyle="round,pad=0.02",
                facecolor=fc, edgecolor="white", linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(ci, gi, txt, ha="center", va="center",
                    fontsize=10, color=txt_color, fontweight="500")

    ax.set_xlim(-0.6, n_cat - 0.4)
    ax.set_ylim(-0.6, n_grp - 0.4)
    ax.set_xticks(range(n_cat))
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_yticks(range(n_grp))
    ax.set_yticklabels(groups, fontsize=11)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(f"Mean {primary_metric}", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        f"{primary_metric} by {group_col} × category\n(gray = fewer than {min_n} instances)",
        fontweight="500", color=COLORS["text"], pad=16,
    )
    fig.tight_layout()
    save_fig(fig, output_path)
