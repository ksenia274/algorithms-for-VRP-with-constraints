"""Визуализация per-iteration трассировки адаптивных стратегий HGS.

Single-режим -- три панели «причина -> реакция -> результат»:
    python visualization/trace_plot.py \\
        results/trace_linear_instance11.csv \\
        results/trace_fairness_signal_instance11.csv \\
        --output visualization/output/trace_instance11_v2.png \\
        --target-cv 0.2 --hold-band 0.05 --max-weight 1e9

Multi-режим -- сравнение дисбаланса по нескольким инстансам:
    python visualization/trace_plot.py --multi \\
        --instances 5 11 18 \\
        --linear-pattern "results/trace_linear_instance{N}.csv" \\
        --fs-pattern     "results/calibrated/trace_fairness_signal_instance{N}.csv" \\
        --output visualization/output/trace_calibrated_grid.png \\
        --target-cv 0.15 --hold-band 0.03
"""
import argparse
import pathlib

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_BG           = "#FAFAF8"
_GRID         = "#E8E6E0"
_TEXT         = "#2C2C2A"
_MUTED        = "#888780"
_COLOR_LINEAR = "#4A6FA5"
_COLOR_FS     = "#E66F51"


def _style_ax(ax):
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


def _first_plateau_start(mask_values, min_run: int = 1000):
    """Первый индекс начала серии >= min_run последовательных True, или None."""
    count = 0
    for i, v in enumerate(mask_values):
        if v:
            count += 1
            if count >= min_run:
                return i - min_run + 1
        else:
            count = 0
    return None


def _save(fig, output: str) -> None:
    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"Saved: {output}")


def plot_single(
    linear_csv: str,
    fs_csv: str,
    output: str,
    target_cv: float = 0.2,
    hold_band: float = 0.05,
    max_weight: float = 1e9,
    min_weight: float = 0.0,
) -> None:
    lin = pd.read_csv(linear_csv)
    fs  = pd.read_csv(fs_csv)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.patch.set_facecolor(_BG)

    # ── Панель 1: CV числа клиентов по маршрутам ──
    ax = axes[0]
    for df, color, label in [
        (lin, _COLOR_LINEAR, "Linear"),
        (fs,  _COLOR_FS,     "FairnessSignal"),
    ]:
        ax.plot(df["iteration"], df["route_balance"],
                color=color, alpha=0.2, lw=0.5)
        ax.plot(df["iteration"], _smooth(df["route_balance"]),
                color=color, lw=1.5, label=label)

    ax.set_ylabel("CV числа клиентов по маршрутам\n(std / mean, ниже = равномернее)")
    ax.legend(fontsize=9)
    _style_ax(ax)

    # ── Панель 2: Адаптивный вес w_rb ──
    ax = axes[1]
    for df, color, label in [
        (lin, _COLOR_LINEAR, "Linear"),
        (fs,  _COLOR_FS,     "FairnessSignal"),
    ]:
        ax.plot(df["iteration"], df["weight_route_balance"],
                color=color, lw=1.2, label=label)

    ax.set_yscale("log")
    ax.set_ylabel("Вес w_rb в целевой функции (log)\n"
                  "cost += w_rb * CV_клиентов")
    ax.legend(fontsize=9)
    _style_ax(ax)

    # ── Панель 3: Средняя длина маршрута текущего решения ──
    # mean_route_dist недоступен напрямую; используем (max + min) / 2
    ax = axes[2]
    for df, color, label in [
        (lin, _COLOR_LINEAR, "Linear"),
        (fs,  _COLOR_FS,     "FairnessSignal"),
    ]:
        mean_approx = (df["max_route_dist"] + df["min_route_dist"]) / 2
        ax.plot(df["iteration"], _smooth(mean_approx),
                color=color, lw=1.5, label=label)

    ax.set_ylabel("Средняя длина маршрута\n(приближение: (max + min) / 2)")
    ax.set_xlabel("Итерация")
    ax.legend(fontsize=9)
    _style_ax(ax)

    fig.tight_layout(pad=2.5)
    _save(fig, output)


def plot_multi(
    instances,
    linear_pattern: str,
    fs_pattern: str,
    output: str,
    target_cv: float = 0.2,
    hold_band: float = 0.05,
) -> None:
    n = len(instances)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    fig.patch.set_facecolor(_BG)

    for j, inst in enumerate(instances):
        lin_path = linear_pattern.replace("{N}", str(inst))
        fs_path  = fs_pattern.replace("{N}", str(inst))
        try:
            lin = pd.read_csv(lin_path)
        except FileNotFoundError:
            lin = None
            print(f"[warn] not found: {lin_path}")
        try:
            fs = pd.read_csv(fs_path)
        except FileNotFoundError:
            fs = None
            print(f"[warn] not found: {fs_path}")

        ax = axes[0, j]
        for df, color, label in [
            (lin, _COLOR_LINEAR, "Linear"),
            (fs,  _COLOR_FS,     "FairnessSignal"),
        ]:
            if df is not None:
                ax.plot(df["iteration"], df["route_balance"],
                        color=color, alpha=0.15, lw=0.5)
                ax.plot(df["iteration"], _smooth(df["route_balance"]),
                        color=color, lw=1.5, label=label)

        ax.set_title(f"Инстанс {inst}", fontsize=11)
        ax.set_xlabel("Итерация")
        if j == 0:
            ax.set_ylabel("CV числа клиентов по маршрутам\n(ниже = равномернее)")
            ax.legend(fontsize=9)
        _style_ax(ax)

    fig.suptitle("Дисбаланс маршрутов: Linear vs FairnessSignal",
                 fontsize=13, color=_TEXT, y=1.02)
    fig.tight_layout(pad=2.0)
    _save(fig, output)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--multi", action="store_true", help="Режим сравнения инстансов")
    p.add_argument("linear_csv", nargs="?", help="Trace CSV: linear стратегия")
    p.add_argument("fs_csv",     nargs="?", help="Trace CSV: fairness_signal стратегия")
    p.add_argument("--instances",      nargs="+", help="ID инстансов для --multi режима")
    p.add_argument("--linear-pattern",
                   default="results/trace_linear_instance{N}.csv")
    p.add_argument("--fs-pattern",
                   default="results/calibrated/trace_fairness_signal_instance{N}.csv")
    p.add_argument("--output",     default="visualization/output/trace.png")
    p.add_argument("--target-cv",  type=float, default=0.2)
    p.add_argument("--hold-band",  type=float, default=0.05)
    p.add_argument("--max-weight", type=float, default=1e9)
    p.add_argument("--min-weight", type=float, default=0.0)
    a = p.parse_args()

    if a.multi:
        if not a.instances:
            p.error("--multi требует --instances")
        plot_multi(
            a.instances, a.linear_pattern, a.fs_pattern,
            a.output, target_cv=a.target_cv, hold_band=a.hold_band,
        )
    else:
        if not a.linear_csv or not a.fs_csv:
            p.error("Укажи linear_csv и fs_csv, или используй --multi режим")
        plot_single(
            a.linear_csv, a.fs_csv, a.output,
            target_cv=a.target_cv,
            hold_band=a.hold_band,
            max_weight=a.max_weight,
            min_weight=a.min_weight,
        )


if __name__ == "__main__":
    main()
