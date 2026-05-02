from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


def _iso_now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def cmd_compare(
    bench_paths: list[Path],
    *,
    output: Optional[Path] = None,
) -> Path:
    """Combine metrics from multiple benchmarks and generate comparison plots."""
    import pandas as pd
    from runtime.global_config import get_global_config

    bench_paths = [Path(p) for p in bench_paths]

    if output is not None:
        compare_dir = Path(output)
    else:
        cfg = get_global_config()
        names = "_vs_".join(p.name[:16] for p in bench_paths)
        compare_dir = Path(cfg.paths.results) / "comparisons" / f"{_iso_now()}_{names}"

    compare_dir.mkdir(parents=True, exist_ok=True)
    (compare_dir / "sources").mkdir(exist_ok=True)

    frames = []
    for bench_path in bench_paths:
        metrics_csv = bench_path / "metrics.csv"
        if not metrics_csv.exists():
            print(f"[warn] no metrics.csv in {bench_path}, skipping")
            continue
        df = pd.read_csv(metrics_csv)
        df["benchmark_name"] = bench_path.name
        frames.append(df)

        for yml in bench_path.glob("*.yaml"):
            shutil.copy(yml, compare_dir / "sources" / f"{bench_path.name}_{yml.name}")

    if not frames:
        raise ValueError("No metrics.csv found in any of the provided benchmark paths.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(compare_dir / "metrics.csv", index=False)

    plots_dir = compare_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    from visualization.compose import plot_all
    paths = plot_all(combined, plots_dir, group_col="benchmark_name")
    for name, path in paths.items():
        print(f"{name} : {path}")

    print(f"compare_dir : {compare_dir}")
    print(f"metrics     : {compare_dir / 'metrics.csv'}")
    return compare_dir
