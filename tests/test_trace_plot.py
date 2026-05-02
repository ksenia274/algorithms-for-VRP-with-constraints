from __future__ import annotations

import gzip
import io
import pytest
import pandas as pd
from pathlib import Path

from visualization.trace_plot import plot_trace


def _make_trace_gz(path: Path, n: int = 200, has_dist: bool = True) -> None:
    rows = []
    import numpy as np
    rng = np.random.default_rng(0)
    for i in range(1, n + 1):
        row = {
            "iteration": i,
            "current_cost": rng.uniform(1e5, 2e5),
            "best_cost": rng.uniform(1e5, 1.5e5),
            "route_cv": rng.uniform(0.05, 0.4),
            "route_range_pct": rng.uniform(0.1, 0.8),
            "weight_route_balance": max(1.0, 500.0 * (0.99999 ** i)),
            "event": "decay",
        }
        if has_dist:
            row["max_route_dist"] = rng.uniform(60_000, 100_000)
            row["min_route_dist"] = rng.uniform(20_000, 50_000)
        rows.append(row)
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(df.to_csv(index=False).encode())
    path.write_bytes(buf.getvalue())


def test_plot_trace_smoke(tmp_path):
    trace_gz = tmp_path / "trace.csv.gz"
    _make_trace_gz(trace_gz)
    out = tmp_path / "trace.png"
    plot_trace(trace_gz, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_trace_with_range_pct(tmp_path):
    trace_gz = tmp_path / "trace.csv.gz"
    _make_trace_gz(trace_gz)
    out = tmp_path / "trace_range.png"
    plot_trace(trace_gz, out, show_range_pct=True)
    assert out.exists() and out.stat().st_size > 0


def test_plot_trace_no_mean_data(tmp_path):
    trace_gz = tmp_path / "trace_nodist.csv.gz"
    _make_trace_gz(trace_gz, has_dist=False)
    out = tmp_path / "trace_nodist.png"
    # Should not raise; panel 3 shows "mean unavailable in trace"
    plot_trace(trace_gz, out)
    assert out.exists() and out.stat().st_size > 0
