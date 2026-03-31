"""
Основной API:
    plot_all(csv_path, output_dir)   — строит все графики и сохраняет в output_dir
    plot_gini_improvement(...)       — гистограмма Gini before/after по инстансам
    plot_category_summary(...)       — grouped bar по категориям
    plot_metric_heatmap(...)         — heatmap метрик по категориям
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


COLORS = {
    "before": "#B4B2A9",
    "after": "#1D9E75",
    "accent": "#534AB7",
    "cost": "#D85A30",
    "bg": "#FAFAF8",
    "grid": "#E8E6E0",
    "text": "#2C2C2A",
    "muted": "#888780",
}

CATEGORY_COLORS = {
    "R1": "#378ADD",
    "R2": "#85B7EB",
    "C1": "#1D9E75",
    "C2": "#5DCAA5",
    "RC1": "#D85A30",
    "RC2": "#F0997B",
}

CATEGORY_ORDER = ["R1", "R2", "C1", "C2", "RC1", "RC2"]


def _style_ax(ax, title: str = "", ylabel: str = ""):
    ax.set_facecolor(COLORS["bg"])
    ax.figure.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["text"], labelsize=9)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=13, fontweight="500", color=COLORS["text"], pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=COLORS["muted"])


def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if "feasible" in df.columns:
        df = df[df["feasible"].astype(str).str.lower() == "true"].copy()
    return df


def plot_gini_improvement(csv_path: str, output_path: str):
    df = _load(csv_path)
    df = df.sort_values(["category", "instance"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    _style_ax(ax, "Gini coefficient: before vs after rebalancing", "Gini (lower = fairer)")

    x = np.arange(len(df))
    w = 0.35

    ax.bar(x - w / 2, df["gini_before"], w, color=COLORS["before"], label="Before", zorder=3)
    ax.bar(x + w / 2, df["gini_after"], w, color=COLORS["after"], label="After", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(df["instance"], rotation=55, ha="right", fontsize=7)
    ax.set_ylim(0, df["gini_before"].max() * 1.2)
    ax.legend(fontsize=9, frameon=False)

    cats = df["category"].values
    for i in range(1, len(cats)):
        if cats[i] != cats[i - 1]:
            ax.axvline(i - 0.5, color=COLORS["muted"], linewidth=0.5, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")



def plot_category_summary(csv_path: str, output_path: str):
    df = _load(csv_path)

    metrics = {
        "Gini (distance)": ("gini_before", "gini_after"),
        "CV (distance)": ("cv_before", "cv_after"),
        "Fairness score": ("score_before", "score_after"),
        "Gini (load)": ("load_gini_before", "load_gini_after"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (title, (col_b, col_a)) in zip(axes.flat, metrics.items()):
        _style_ax(ax, title)

        cats = [c for c in CATEGORY_ORDER if c in df["category"].values]
        means_b = [df[df["category"] == c][col_b].mean() for c in cats]
        means_a = [df[df["category"] == c][col_a].mean() for c in cats]

        x = np.arange(len(cats))
        w = 0.32

        ax.bar(x - w / 2, means_b, w, color=COLORS["before"], label="Before", zorder=3)
        ax.bar(x + w / 2, means_a, w, color=COLORS["after"], label="After", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=10)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Fairness metrics by Solomon category (mean)", fontsize=14,
                 fontweight="500", color=COLORS["text"], y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_metric_heatmap(csv_path: str, output_path: str):
    df = _load(csv_path)
    df = df.sort_values(["category", "instance"]).reset_index(drop=True)

    cols = ["gini_before", "gini_after", "cv_before", "cv_after",
            "score_before", "score_after"]
    labels = ["Gini\nbefore", "Gini\nafter", "CV\nbefore", "CV\nafter",
              "Score\nbefore", "Score\nafter"]

    data = df[cols].values

    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.25)))
    _style_ax(ax, "Fairness metrics heatmap")

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0,
                   vmax=max(0.5, data.max()))

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["instance"], fontsize=7)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val > data.max() * 0.6 else COLORS["text"]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=6, color=color)

    fig.colorbar(im, ax=ax, shrink=0.6, label="Value (lower = fairer)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_rebalance_moves(csv_path: str, output_path: str):
    df = _load(csv_path)

    fig, ax = plt.subplots(figsize=(10, 4))
    _style_ax(ax, "Rebalance moves per instance", "Moves applied")

    df_sorted = df.sort_values(["category", "instance"]).reset_index(drop=True)
    x = np.arange(len(df_sorted))

    bar_colors = [CATEGORY_COLORS.get(c, COLORS["muted"]) for c in df_sorted["category"]]
    ax.bar(x, df_sorted["rebalance_moves"], color=bar_colors, zorder=3, width=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["instance"], rotation=55, ha="right", fontsize=7)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in CATEGORY_ORDER
               if c in df_sorted["category"].values]
    ax.legend(handles=handles, fontsize=8, frameon=False, ncol=6, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")



def plot_all(csv_path: str, output_dir: str = "visualization/output"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nBuilding charts from: {csv_path}")
    print(f"Output directory: {output_dir}\n")

    plot_gini_improvement(csv_path, os.path.join(output_dir, "01_gini_before_after.png"))
    plot_category_summary(csv_path, os.path.join(output_dir, "02_category_summary.png"))
    plot_metric_heatmap(csv_path, os.path.join(output_dir, "03_metric_heatmap.png"))
    plot_rebalance_moves(csv_path, os.path.join(output_dir, "04_rebalance_moves.png"))

    print(f"\nAll charts saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/fairness_benchmark.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "visualization/output"
    plot_all(csv_path, output_dir)