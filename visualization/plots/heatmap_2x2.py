from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from visualization._style import CATEGORY_ORDER, COLORS, save_fig
from visualization.utils import add_category_column

_METRICS = [
    ("dist_gini",        "Gini"),
    ("dist_worst_ratio", "Worst ratio"),
    ("dist_cv",          "CV"),
    ("total_distance",   "Distance"),
]


def plot_heatmap_2x2(
    df: pd.DataFrame,
    output_path: Path,
    *,
    group_col: str = "algorithm",
    min_n: int = 1,
) -> None:
    """2×2 heatmap subplots — Gini, worst_ratio, CV, distance — algorithm × category."""
    output_path = Path(output_path)

    feas = df[df["feasible"].astype(str).str.lower() == "true"].copy()
    if "category" not in feas.columns:
        feas = add_category_column(feas)

    groups = sorted(feas[group_col].unique()) if group_col in feas.columns else ["all"]
    present_cats = set(feas["category"].astype(str).unique())
    cats = [c for c in CATEGORY_ORDER if c in present_cats] or sorted(present_cats)
    n_grp, n_cat = len(groups), len(cats)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(max(8, n_cat * 2.0 + 3), max(6, n_grp * 1.2 + 3)),
    )
    fig.set_facecolor("white")

    for idx, (metric, subtitle) in enumerate(_METRICS):
        ax = axes[idx // 2][idx % 2]
        ax.set_facecolor(COLORS["bg"])
        ax.set_title(subtitle, fontsize=11, fontweight="500", pad=8)

        if metric not in feas.columns:
            ax.text(0.5, 0.5, f"no column: {metric}",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        data = np.full((n_grp, n_cat), np.nan)
        for gi, grp in enumerate(groups):
            sub_g = feas[feas[group_col] == grp] if group_col in feas.columns else feas
            for ci, cat in enumerate(cats):
                vals = sub_g[sub_g["category"].astype(str) == cat][metric].dropna()
                if len(vals) >= min_n:
                    data[gi, ci] = vals.mean()

        valid = data[~np.isnan(data)]
        vmin = float(valid.min()) if len(valid) else 0.0
        vmax = float(valid.max()) if len(valid) else 1.0
        if vmax <= vmin:
            vmax = vmin + 0.1
        cmap = plt.cm.RdYlGn_r
        norm = Normalize(vmin=vmin, vmax=vmax)

        for gi in range(n_grp):
            for ci in range(n_cat):
                if np.isnan(data[gi, ci]):
                    fc, txt, tc = "#D0D0D0", "—", "#999"
                else:
                    fc = cmap(norm(data[gi, ci]))
                    if metric == "total_distance":
                        txt = f"{data[gi, ci]:.0f}"
                    else:
                        txt = f"{data[gi, ci]:.3f}"
                    lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                    tc = "white" if lum < 0.5 else COLORS["text"]
                rect = mpatches.FancyBboxPatch(
                    (ci - 0.45, gi - 0.4), 0.9, 0.8,
                    boxstyle="round,pad=0.02",
                    facecolor=fc, edgecolor="white", linewidth=1.0,
                )
                ax.add_patch(rect)
                ax.text(ci, gi, txt, ha="center", va="center",
                        fontsize=9, color=tc)

        ax.set_xlim(-0.6, n_cat - 0.4)
        ax.set_ylim(-0.6, n_grp - 0.4)
        ax.set_xticks(range(n_cat))
        ax.set_xticklabels(cats, fontsize=9)
        ax.set_yticks(range(n_grp))
        ax.set_yticklabels(groups, fontsize=9)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"Metrics by {group_col} × category", fontsize=12, fontweight="500", y=1.01)
    fig.tight_layout()
    save_fig(fig, output_path)
