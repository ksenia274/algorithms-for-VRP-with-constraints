"""Shared matplotlib style constants and helpers."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "bg":    "#FAFAF8",
    "grid":  "#E8E6E0",
    "text":  "#2C2C2A",
    "muted": "#888780",
    "ref":   "#B0B0A8",
}

CATEGORY_COLORS = {
    "R1": "#378ADD", "R2": "#85B7EB",
    "C1": "#1D9E75", "C2": "#5DCAA5",
    "RC1": "#D85A30", "RC2": "#F0997B",
    "yandex": "#A064C8",
}

CATEGORY_ORDER = ["R1", "R2", "C1", "C2", "RC1", "RC2", "yandex"]

DIMENSION_COLORS = {
    "Distance": "#4A6FA5",
    "Load":     "#E66F51",
    "Clients":  "#3CAA64",
}

_ALG_PALETTE = [
    "#4A6FA5", "#E66F51", "#3CAA64", "#A064C8",
    "#DCBA32", "#E06090", "#60C8C8", "#8B4513",
]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})


def style_ax(ax, title: str = "", ylabel: str = "", xlabel: str = "") -> None:
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


def group_colors(groups: list[str]) -> dict[str, str]:
    return {g: _ALG_PALETTE[i % len(_ALG_PALETTE)] for i, g in enumerate(groups)}


def save_fig(fig, path) -> None:
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
