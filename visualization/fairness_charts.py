"""
Fairness benchmark visualisation — multi-algorithm support.

Public API
----------
plot_all(csv_paths, labels, output_dir)
    csv_paths : list[str] | str   one CSV per algorithm (or single string — backward compat)
    labels    : list[str] | None  algorithm display names; defaults to filename stems
    output_dir: str               destination directory

Backward-compatible call (from run_visualise.py):
    plot_all(csv_path="results/x.csv", output_dir="visualization/output")

CLI
---
python visualization/fairness_charts.py results/hgs.csv results/alns.csv \\
    --labels "HGS baseline" "ALNS" --output visualization/output/

Charts produced
---------------
01_pareto.png           — cost vs. route balance trade-off (one point per algorithm)
02_dimensions.png       — worst_ratio by dimension (distance / load / clients) per algorithm
03_gini_distribution.png — distribution of dist_gini across instances, per algorithm
04_category_heatmap.png — median dist_worst_ratio by algorithm × instance category

Metric definitions (columns read from CSV)
------------------------------------------
dist_worst_ratio_after   — max route distance / mean route distance  (1.0 = ideal)
dist_gini_after          — Gini coefficient of route distances        (0.0 = equal)
load_worst_ratio_after   — max route load / mean route load           (1.0 = ideal)
clients_worst_ratio_after — max clients per route / mean             (1.0 = ideal)
total_distance           — sum of all route distances
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False


# ─────────────────────────── colour palettes ──────────────────────────────────

COLORS = {
    "bg":   "#FAFAF8",
    "grid": "#E8E6E0",
    "text": "#2C2C2A",
    "muted": "#888780",
    "ref":  "#B0B0A8",  # reference line (ideal = 1.0)
}

CATEGORY_COLORS = {
    "R1":     "#378ADD",
    "R2":     "#85B7EB",
    "C1":     "#1D9E75",
    "C2":     "#5DCAA5",
    "RC1":    "#D85A30",
    "RC2":    "#F0997B",
    "yandex": "#A064C8",
}

CATEGORY_ORDER = ["R1", "R2", "C1", "C2", "RC1", "RC2", "yandex"]

DIMENSION_COLORS = {
    "Distance": "#4A6FA5",
    "Load":     "#E66F51",
    "Clients":  "#3CAA64",
}

_ALG_PALETTE = [
    "#4A6FA5",
    "#E66F51",
    "#3CAA64",
    "#A064C8",
    "#DCBA32",
    "#E06090",
    "#60C8C8",
    "#8B4513",
]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


# ─────────────────────────── helpers ──────────────────────────────────────────

def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if "feasible" in df.columns:
        df = df[df["feasible"].astype(str).str.lower() == "true"].copy()
    return df


def _style_ax(ax, title: str = "", ylabel: str = "", xlabel: str = ""):
    ax.set_facecolor(COLORS["bg"])
    ax.figure.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["text"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontweight="500", color=COLORS["text"], pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=COLORS["muted"])
    if xlabel:
        ax.set_xlabel(xlabel, color=COLORS["muted"])


def _alg_colors(n: int) -> list[str]:
    return [_ALG_PALETTE[i % len(_ALG_PALETTE)] for i in range(n)]


def _norm_inputs(csv_paths, labels, csv_path_kw) -> tuple[list[str], list[str]]:
    if csv_path_kw is not None:
        csv_paths = [csv_path_kw]
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    if not csv_paths:
        raise ValueError("No CSV paths provided.")
    if labels is None:
        labels = [Path(p).stem for p in csv_paths]
    if len(labels) != len(csv_paths):
        raise ValueError("len(labels) must equal len(csv_paths).")
    return list(csv_paths), list(labels)


def _pareto_frontier(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    """Lower is better on both axes."""
    pts = sorted(zip(xs, ys))
    frontier: list[tuple[float, float]] = []
    best_y = float("inf")
    for x, y in pts:
        if y <= best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


# ─────────────────────── Chart 1 — Pareto scatter ─────────────────────────────

def plot_pareto(
    dfs: list[pd.DataFrame],
    labels: list[str],
    colors: list[str],
    output_path: str,
    dfs_raw: list[pd.DataFrame],
):
    """
    Cost–fairness trade-off: median total_distance vs median dist_worst_ratio_after.
    One point per algorithm. Lower-left is better on both axes.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    _style_ax(
        ax,
        title="Cost vs. route balance trade-off  (lower-left = better)",
        ylabel="Worst route length / average route length  (1.0 = ideal)",
        xlabel="Median total route distance  (lower = shorter routes)",
    )

    # reference line at y=1.0 (ideal balance)
    ax.axhline(1.0, color=COLORS["ref"], linewidth=1.0, linestyle="--",
               alpha=0.7, zorder=1, label="Ideal balance (y = 1.0)")

    xs, ys = [], []
    for i, (df, label, color) in enumerate(zip(dfs, labels, colors)):
        if "dist_worst_ratio_after" not in df.columns or "total_distance" not in df.columns:
            continue
        med_dist = df["total_distance"].median()
        med_wr = df["dist_worst_ratio_after"].median()
        n_feas = len(df)
        n_total = len(dfs_raw[i]) if dfs_raw else n_feas
        pct_inf = (1 - n_feas / n_total) * 100 if n_total > 0 else 0.0

        xs.append(med_dist)
        ys.append(med_wr)

        ax.scatter(med_dist, med_wr,
                   s=80 + n_feas * 4, color=color, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(
            f"{label}\n({pct_inf:.0f}% infeasible)",
            (med_dist, med_wr),
            xytext=(10, 6), textcoords="offset points",
            fontsize=9, color=color,
        )

    if len(xs) > 1:
        frontier = _pareto_frontier(xs, ys)
        if len(frontier) > 1:
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, color=COLORS["muted"], linewidth=1.2,
                    linestyle=":", alpha=0.7, zorder=3, label="Pareto frontier")

    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────── Chart 2 — Dimensions bar chart ───────────────────────

def plot_dimensions(
    dfs: list[pd.DataFrame],
    labels: list[str],
    output_path: str,
):
    """
    Grouped bars: for each algorithm, three bars — dist / load / clients worst_ratio.
    Median over all feasible instances. Reference line at 1.0.
    """
    dims = [
        ("Distance", "dist_worst_ratio_after"),
        ("Load",     "load_worst_ratio_after"),
        ("Clients",  "clients_worst_ratio_after"),
    ]
    n_alg = len(labels)
    n_dim = len(dims)
    group_w = 0.7
    bar_w = group_w / n_dim

    fig, ax = plt.subplots(figsize=(max(6, n_alg * 2.5), 5))
    _style_ax(
        ax,
        title="Route balance by dimension  (median worst route / average route)",
        ylabel="Worst route / average route  (1.0 = ideal balance)",
    )

    ax.axhline(1.0, color=COLORS["ref"], linewidth=1.2, linestyle="--",
               alpha=0.8, zorder=1, label="Ideal (1.0)")

    xs = np.arange(n_alg, dtype=float)

    for di, (dim_name, col) in enumerate(dims):
        dim_color = DIMENSION_COLORS[dim_name]
        offset = (di - (n_dim - 1) / 2) * bar_w
        vals = []
        for df in dfs:
            if col in df.columns:
                vals.append(df[col].median())
            else:
                vals.append(np.nan)
        bars = ax.bar(xs + offset, vals, bar_w * 0.88,
                      color=dim_color, label=dim_name, zorder=3, alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=8, color=COLORS["text"])

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_xlim(-0.6, n_alg - 0.4)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────── Chart 3 — Gini distribution ─────────────────────────

def plot_gini_distribution(
    dfs: list[pd.DataFrame],
    labels: list[str],
    colors: list[str],
    output_path: str,
):
    """
    Violin + jitter: distribution of dist_gini_after across instances per algorithm.
    Points coloured by instance category.
    """
    col = "dist_gini_after"
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.8), 6))
    _style_ax(
        ax,
        title="Distribution of route distance Gini coefficient across instances",
        ylabel="Gini coefficient of route distances  (0.0 = equal, 1.0 = maximally unequal)",
    )

    rng = np.random.default_rng(0)

    if _HAS_SNS:
        rows = []
        for label, df in zip(labels, dfs):
            if col not in df.columns:
                continue
            for _, row in df.iterrows():
                rows.append({
                    "algorithm": label,
                    col: row[col],
                    "category": row.get("category", "other"),
                })
        if not rows:
            plt.close(fig)
            return
        tidy = pd.DataFrame(rows)
        palette = dict(zip(labels, colors))
        sns.violinplot(
            data=tidy, x="algorithm", y=col,
            hue="algorithm", palette=palette, inner=None, ax=ax,
            cut=0, linewidth=1.2, alpha=0.65, legend=False,
        )
        cat_set = list(tidy["category"].unique())
        for xi, label in enumerate(labels):
            sub = tidy[tidy["algorithm"] == label]
            for cat in cat_set:
                vals = sub[sub["category"] == cat][col].values
                jitter = rng.uniform(-0.08, 0.08, len(vals))
                cat_color = CATEGORY_COLORS.get(cat, COLORS["muted"])
                ax.scatter(xi + jitter, vals, s=22, color=cat_color,
                           alpha=0.75, zorder=4, linewidths=0)
            med = tidy[tidy["algorithm"] == label][col].median()
            ax.hlines(med, xi - 0.3, xi + 0.3, color="white", linewidth=2.5, zorder=5)
            ax.hlines(med, xi - 0.3, xi + 0.3, color=colors[xi],
                      linewidth=1.5, linestyle="--", zorder=6)
    else:
        for xi, (df, label, color) in enumerate(zip(dfs, labels, colors)):
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) < 2:
                continue
            vp = ax.violinplot(vals, positions=[xi], widths=0.6,
                               showmedians=False, showextrema=True)
            for pc in vp["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.55)
            for part in ("cmins", "cmaxes", "cbars"):
                if part in vp:
                    vp[part].set_color(COLORS["muted"])
                    vp[part].set_linewidth(0.8)
            med = float(np.median(vals))
            ax.hlines(med, xi - 0.25, xi + 0.25, color=color, linewidth=2, zorder=5)
            jitter = rng.uniform(-0.08, 0.08, len(vals))
            cats = df["category"].values if "category" in df.columns else ["other"] * len(vals)
            for v, j, cat in zip(vals, jitter, cats):
                ax.scatter(xi + j, v, s=22,
                           color=CATEGORY_COLORS.get(str(cat), COLORS["muted"]),
                           alpha=0.75, zorder=4, linewidths=0)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")

    present_cats: set[str] = set()
    for df in dfs:
        if "category" in df.columns:
            present_cats.update(df["category"].astype(str).unique())
    handles = [mpatches.Patch(color=CATEGORY_COLORS.get(c, COLORS["muted"]), label=c)
               for c in CATEGORY_ORDER if c in present_cats]
    if handles:
        ax.legend(handles=handles, title="Category", loc="upper right",
                  frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────── Chart 4 — Category heatmap ──────────────────────────

def plot_category_heatmap(
    dfs: list[pd.DataFrame],
    labels: list[str],
    output_path: str,
    min_n: int = 3,
):
    """
    Rows: algorithms. Cols: instance categories.
    Cell value: median(dist_worst_ratio_after).
    Color: RdYlGn_r with vmin=1.0 (ideal) — green = balanced, red = imbalanced.
    Cells with fewer than min_n instances are shown in gray.
    """
    col = "dist_worst_ratio_after"

    present_cats: set[str] = set()
    for df in dfs:
        if "category" in df.columns:
            present_cats.update(df["category"].astype(str).unique())
    cats = [c for c in CATEGORY_ORDER if c in present_cats]
    if not cats:
        cats = sorted(present_cats)

    n_alg = len(labels)
    n_cat = len(cats)
    data = np.full((n_alg, n_cat), np.nan)
    mask = np.zeros((n_alg, n_cat), dtype=bool)

    for ai, df in enumerate(dfs):
        for ci, cat in enumerate(cats):
            if "category" not in df.columns or col not in df.columns:
                mask[ai, ci] = True
                continue
            sub = df[df["category"].astype(str) == cat][col].dropna()
            if len(sub) < min_n:
                mask[ai, ci] = True
            else:
                data[ai, ci] = sub.median()

    fig, ax = plt.subplots(
        figsize=(max(6, n_cat * 1.5 + 2), max(3, n_alg * 0.9 + 1.8))
    )
    ax.set_facecolor(COLORS["bg"])
    ax.figure.set_facecolor("white")

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    valid = data[~np.isnan(data) & ~mask]
    vmin = 1.0  # worst_ratio is always >= 1.0
    vmax = float(valid.max()) if len(valid) > 0 else 2.0
    if vmax <= vmin:
        vmax = vmin + 0.1
    cmap = plt.cm.RdYlGn_r  # green = low (fair), red = high (imbalanced)
    norm = Normalize(vmin=vmin, vmax=vmax)

    for ai in range(n_alg):
        for ci in range(n_cat):
            if mask[ai, ci] or np.isnan(data[ai, ci]):
                fc = "#D0D0D0"
                txt = "n<3"
                txt_color = "#999"
            else:
                fc = cmap(norm(data[ai, ci]))
                txt = f"{data[ai, ci]:.3f}"
                luminance = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                txt_color = "white" if luminance < 0.5 else COLORS["text"]

            rect = mpatches.FancyBboxPatch(
                (ci - 0.45, ai - 0.4), 0.9, 0.8,
                boxstyle="round,pad=0.02",
                facecolor=fc, edgecolor="white", linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(ci, ai, txt, ha="center", va="center",
                    fontsize=10, color=txt_color, fontweight="500")

    ax.set_xlim(-0.6, n_cat - 0.4)
    ax.set_ylim(-0.6, n_alg - 0.4)
    ax.set_xticks(range(n_cat))
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_yticks(range(n_alg))
    ax.set_yticklabels(labels, fontsize=11)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Median  worst route / average route\n(1.0 = ideal, green = balanced)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        "Route distance balance by algorithm × instance category\n"
        "(cell = median dist_worst_ratio_after,  n<3 instances = gray)",
        fontweight="500", color=COLORS["text"], pad=16,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────── main entry point ─────────────────────────────────────

def plot_all(
    csv_paths=None,
    labels: list[str] | None = None,
    output_dir: str = "visualization/output",
    **kwargs,
):
    """
    Build all 4 fairness charts.

    Parameters
    ----------
    csv_paths : list[str] | str
        Paths to per-algorithm CSV files (or single string).
    labels : list[str] | None
        Display names for each algorithm. Defaults to filename stems.
    output_dir : str
        Directory to write PNG files.
    **kwargs
        Accepts legacy `csv_path=` keyword for backward compatibility.
    """
    csv_paths, labels = _norm_inputs(csv_paths, labels, kwargs.get("csv_path"))

    os.makedirs(output_dir, exist_ok=True)
    colors = _alg_colors(len(csv_paths))

    dfs = [_load(p) for p in csv_paths]
    dfs_raw = []
    for p in csv_paths:
        r = pd.read_csv(p)
        r.columns = r.columns.str.strip()
        dfs_raw.append(r)

    print(f"\nBuilding charts  ({len(csv_paths)} algorithm(s))")
    for label, df in zip(labels, dfs):
        print(f"  {label}: {len(df)} feasible rows")
    print(f"Output: {output_dir}\n")

    plot_pareto(
        dfs, labels, colors,
        os.path.join(output_dir, "01_pareto.png"),
        dfs_raw=dfs_raw,
    )
    plot_dimensions(
        dfs, labels,
        os.path.join(output_dir, "02_dimensions.png"),
    )
    plot_gini_distribution(
        dfs, labels, colors,
        os.path.join(output_dir, "03_gini_distribution.png"),
    )
    plot_category_heatmap(
        dfs, labels,
        os.path.join(output_dir, "04_category_heatmap.png"),
    )

    print(f"\nAll charts saved to {output_dir}/")


# ─────────────────────── CLI ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate fairness benchmark charts from CSV results."
    )
    parser.add_argument(
        "csv_paths", nargs="+", metavar="CSV",
        help="Per-algorithm result CSVs",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Algorithm display names (same order as CSVs)",
    )
    parser.add_argument(
        "--output", default="visualization/output",
        help="Output directory (default: visualization/output)",
    )
    args = parser.parse_args()

    plot_all(
        csv_paths=args.csv_paths,
        labels=args.labels,
        output_dir=args.output,
    )
