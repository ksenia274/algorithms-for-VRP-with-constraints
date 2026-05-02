from __future__ import annotations

from pathlib import Path


def _is_benchmark_dir(path: Path) -> bool:
    return (path / "runs").is_dir()


def _is_run_dir(path: Path) -> bool:
    return (path / "result.json").is_file()


def cmd_visualize(target_path: Path) -> None:
    """Generate plots for a run or benchmark directory (auto-detected)."""
    target_path = Path(target_path)

    if _is_benchmark_dir(target_path):
        _visualize_benchmark(target_path)
    elif _is_run_dir(target_path):
        _visualize_run(target_path)
    else:
        raise ValueError(
            f"Cannot detect type for '{target_path}'. "
            "Expected a run directory (contains result.json) "
            "or a benchmark directory (contains runs/)."
        )


def _visualize_run(run_dir: Path) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    trace_gz = run_dir / "trace.csv.gz"
    if trace_gz.exists():
        from visualization.trace_plot import plot_trace
        out = plots_dir / "trace.png"
        plot_trace(trace_gz, out)
        print(f"trace plot : {out}")


def _visualize_benchmark(bench_dir: Path) -> None:
    import pandas as pd
    from visualization.compose import plot_all

    plots_dir = bench_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    metrics_csv = bench_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(
            f"metrics.csv not found in {bench_dir}. Run aggregate_metrics first."
        )

    df = pd.read_csv(metrics_csv)
    paths = plot_all(df, plots_dir, group_col="algorithm")
    for name, path in paths.items():
        print(f"{name} : {path}")
