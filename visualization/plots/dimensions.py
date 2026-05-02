from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._style import COLORS, DIMENSION_COLORS, save_fig, style_ax
from visualization.utils import validate_metrics_csv

_DIMS = [
    ("Distance", "dist_worst_ratio"),
    ("Load",     "load_worst_ratio"),
    ("Clients",  "clients_worst_ratio"),
]
_REQUIRED = [col for _, col in _DIMS]


def plot_dimensions(
    df: pd.DataFrame,
    output_path: Path,
    *,
    primary_metric: Optional[str] = None,
    group_col: str = "algorithm",
) -> None:
    """Grouped bars: worst_ratio for dist/load/clients per group value."""
    validate_metrics_csv(df, _REQUIRED)

    feas = df[df["feasible"].astype(str).str.lower() == "true"].copy()
    groups = sorted(feas[group_col].unique()) if group_col in feas.columns else ["all"]
    n_alg = len(groups)
    n_dim = len(_DIMS)
    bar_w = 0.7 / n_dim

    fig, ax = plt.subplots(figsize=(max(6, n_alg * 2.5), 5))
    style_ax(
        ax,
        title="Route balance by dimension  (median worst / average, 1.0 = ideal)",
        ylabel="worst route / average route",
    )
    ax.axhline(1.0, color=COLORS["ref"], linewidth=1.2, linestyle="--",
               alpha=0.8, zorder=1, label="ideal (1.0)")

    xs = np.arange(n_alg, dtype=float)
    for di, (dim_name, col) in enumerate(_DIMS):
        offset = (di - (n_dim - 1) / 2) * bar_w
        vals = []
        for group in groups:
            sub = feas[feas[group_col] == group] if group_col in feas.columns else feas
            vals.append(sub[col].median() if col in sub.columns else np.nan)
        bars = ax.bar(xs + offset, vals, bar_w * 0.88,
                      color=DIMENSION_COLORS[dim_name], label=dim_name, zorder=3, alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                        color=COLORS["text"])

    ax.set_xticks(xs)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save_fig(fig, output_path)
