"""Visualise per-iteration trace of the adaptive HGS solver.

Single-trace mode (one file, 3-4 panels):
    python visualization/trace_plot.py trace.csv.gz --output trace.png

Multi-trace mode (compare two strategies across instances):
    python visualization/trace_plot.py --multi
        --instances 3 5 11
        --linear-pattern "results/runs/*linear*{N}*/trace.csv.gz"
        --fs-pattern     "results/runs/*fs*{N}*/trace.csv.gz"
        --output trace_grid.png
"""
from __future__ import annotations

import argparse
import pathlib
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_BG = "#FAFAF8"
_GRID = "#E8E6E0"
_TEXT = "#2C2C2A"
_COLOR_LINEAR = "#4A6FA5"
_COLOR_FS = "#E66F51"


def _style_ax(ax) -> None:
    ax.set_facecolor(_BG)
    ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(_GRID)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)


def _smooth(series: pd.Series) -> pd.Series:
    w = min(200, max(50, len(series) // 100))
    return series.rolling(window=w, min_periods=1, center=True).mean()


def _save(fig, output: str | Path) -> None:
    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"Saved: {output}")


def _load_trace(path: str | Path) -> pd.DataFrame:
    p = str(path)
    compression = "gzip" if p.endswith(".gz") else "infer"
    return pd.read_csv(p, compression=compression)


def plot_trace(
    trace_csv_path: str | Path,
    output_path: str | Path,
    *,
    show_range_pct: bool = False,
) -> None:
    """Plot adaptive solver trace. Three panels (optionally four).

    Panel 1: route_cv (std/mean) per iteration.
    Panel 2: weight_route_balance in log-scale.
    Panel 3: max/min route distance (mean unavailable in trace).
    Panel 4 (optional, show_range_pct=True): route_range_pct vs route_cv overlay.
    """
    df = _load_trace(trace_csv_path)

    n_panels = 4 if show_range_pct else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels))
    fig.patch.set_facecolor(_BG)
    color = _COLOR_LINEAR

    iters = df["iteration"] if "iteration" in df.columns else df.index

    # Panel 1 — route_cv
    ax = axes[0]
    if "route_cv" in df.columns:
        ax.plot(iters, df["route_cv"], color=color, alpha=0.2, lw=0.5)
        ax.plot(iters, _smooth(df["route_cv"]), color=color, lw=1.5, label="route_cv")
    else:
        ax.text(0.5, 0.5, "route_cv not available in trace",
                transform=ax.transAxes, ha="center", va="center", color=_TEXT)
    ax.set_ylabel("route CV  (std / mean,  lower = more balanced)")
    ax.legend(fontsize=9)
    _style_ax(ax)

    # Panel 2 — weight_route_balance log-scale
    ax = axes[1]
    if "weight_route_balance" in df.columns:
        ax.plot(iters, df["weight_route_balance"], color=color, lw=1.2)
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "weight_route_balance not available",
                transform=ax.transAxes, ha="center", va="center", color=_TEXT)
    ax.set_ylabel("w_rb in objective  (log scale)")
    _style_ax(ax)

    # Panel 3 — max / min route distance (mean unavailable in trace)
    ax = axes[2]
    has_dist = "max_route_dist" in df.columns and "min_route_dist" in df.columns
    if has_dist:
        ax.plot(iters, _smooth(df["max_route_dist"]), color=color, lw=1.2, label="max")
        ax.plot(iters, _smooth(df["min_route_dist"]), color=color, lw=0.8,
                linestyle="--", alpha=0.7, label="min")
        ax.fill_between(iters, _smooth(df["min_route_dist"]),
                        _smooth(df["max_route_dist"]), alpha=0.12, color=color)
        ax.text(0.02, 0.95, "mean unavailable in trace", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=_TEXT, alpha=0.6)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "mean unavailable in trace",
                transform=ax.transAxes, ha="center", va="center", color=_TEXT)
    ax.set_ylabel("Route distance range")
    ax.set_xlabel("Iteration")
    _style_ax(ax)

    # Panel 4 (optional) — route_range_pct vs route_cv
    if show_range_pct:
        ax = axes[3]
        if "route_cv" in df.columns:
            ax.plot(iters, _smooth(df["route_cv"]), color=color, lw=1.5, label="route_cv (std/mean)")
        if "route_range_pct" in df.columns:
            ax.plot(iters, _smooth(df["route_range_pct"]), color=_COLOR_FS, lw=1.2,
                    linestyle="--", label="route_range_pct (max-min)/mean")
        ax.set_ylabel("CV comparison")
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=9)
        _style_ax(ax)

    fig.tight_layout(pad=2.0)
    _save(fig, output_path)


def plot_multi(
    instances: list[str | int],
    linear_pattern: str,
    fs_pattern: str,
    output: str | Path,
) -> None:
    """Compare route_cv across multiple instances, two strategies side by side."""
    n = len(instances)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    fig.patch.set_facecolor(_BG)

    for j, inst in enumerate(instances):
        lin_path = linear_pattern.replace("{N}", str(inst))
        fs_path = fs_pattern.replace("{N}", str(inst))

        lin = fs = None
        for path, name in [(lin_path, "linear"), (fs_path, "fs")]:
            try:
                df = _load_trace(path)
                if name == "linear":
                    lin = df
                else:
                    fs = df
            except FileNotFoundError:
                print(f"[warn] not found: {path}")

        ax = axes[0, j]
        col = "route_cv" if (lin is not None and "route_cv" in lin.columns) else "route_range_pct"

        for df, color, label in [(lin, _COLOR_LINEAR, "Linear"), (fs, _COLOR_FS, "FairnessSignal")]:
            if df is not None and col in df.columns:
                iters = df["iteration"] if "iteration" in df.columns else df.index
                ax.plot(iters, df[col], color=color, alpha=0.15, lw=0.5)
                ax.plot(iters, _smooth(df[col]), color=color, lw=1.5, label=label)

        ax.set_title(f"Instance {inst}", fontsize=11)
        ax.set_xlabel("Iteration")
        if j == 0:
            ax.set_ylabel(f"{col}  (lower = more balanced)")
            ax.legend(fontsize=9)
        _style_ax(ax)

    fig.suptitle(f"Route balance: Linear vs FairnessSignal  ({col})",
                 fontsize=13, color=_TEXT, y=1.02)
    fig.tight_layout(pad=2.0)
    _save(fig, output)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--multi", action="store_true")
    p.add_argument("trace_csv", nargs="?")
    p.add_argument("--output", default="visualization/output/trace.png")
    p.add_argument("--show-range-pct", action="store_true")
    p.add_argument("--instances", nargs="+")
    p.add_argument("--linear-pattern", default="results/runs/*linear*{N}*/trace.csv.gz")
    p.add_argument("--fs-pattern", default="results/runs/*fs*{N}*/trace.csv.gz")
    a = p.parse_args()

    if a.multi:
        if not a.instances:
            p.error("--multi requires --instances")
        plot_multi(a.instances, a.linear_pattern, a.fs_pattern, a.output)
    else:
        if not a.trace_csv:
            p.error("Provide trace_csv path or use --multi")
        plot_trace(a.trace_csv, a.output, show_range_pct=a.show_range_pct)


if __name__ == "__main__":
    main()
