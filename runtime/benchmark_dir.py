from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


def _iso_now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def _default_base() -> Path:
    from runtime.global_config import get_global_config
    cfg = get_global_config()
    return Path(cfg.paths.results) / "benchmarks"


def create_benchmark_dir(
    name: str,
    *,
    base_dir: Optional[Path] = None,
) -> Path:
    """Create benchmark directory with runs/ and plots/ subdirectories.

    Name: {ISO-timestamp}_{name}
    """
    base = base_dir if base_dir is not None else _default_base()
    benchmark_dir = base / f"{_iso_now()}_{name}"
    (benchmark_dir / "runs").mkdir(parents=True, exist_ok=False)
    (benchmark_dir / "plots").mkdir(parents=True, exist_ok=False)
    return benchmark_dir


def aggregate_metrics(benchmark_dir: Path) -> Path:
    """Concatenate all runs/*/metrics.csv into benchmark_dir/metrics.csv.

    Returns the path to the aggregated file.
    """
    out_path = benchmark_dir / "metrics.csv"
    header_written = False
    with open(out_path, "w", newline="", encoding="utf-8") as out_fh:
        writer: Optional[csv.DictWriter] = None
        for metrics_csv in sorted((benchmark_dir / "runs").rglob("metrics.csv")):
            with open(metrics_csv, newline="", encoding="utf-8") as in_fh:
                reader = csv.DictReader(in_fh)
                if not header_written:
                    writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    header_written = True
                for row in reader:
                    writer.writerow(row)
    return out_path
