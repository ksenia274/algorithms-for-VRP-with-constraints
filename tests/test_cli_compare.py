from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path

from data.instance_resolver import resolve_yandex_path
from runtime.cli.benchmark_command import cmd_benchmark
from runtime.cli.compare_command import cmd_compare


def _find_yandex_instance_name() -> str | None:
    for name in ("3", "0", "1"):
        if resolve_yandex_path(name) is not None:
            return name
    return None


def _make_mini_bench(base: Path, name: str, inst: str) -> Path:
    cfg = base / f"{name}.yaml"
    cfg.write_text(f"""
name: {name}
instances:
  - {{name: "{inst}", kind: yandex}}
algorithms:
  - name: simple
    type: hgs_simple
    algorithm_params: {{}}
shared:
  time_limit: 3
  seed: 42
  capacity: 200
  num_vehicles: 25
""")
    return cmd_benchmark(cfg, output=base / name)


def test_cmd_compare_combines_metrics(tmp_results_dir):
    inst = _find_yandex_instance_name()
    if inst is None:
        pytest.skip("No yandex instances found on disk")

    bench1 = _make_mini_bench(tmp_results_dir, "bench_a", inst)
    bench2 = _make_mini_bench(tmp_results_dir, "bench_b", inst)

    compare_dir = cmd_compare([bench1, bench2], output=tmp_results_dir / "compare_out")

    assert (compare_dir / "metrics.csv").exists()
    df = pd.read_csv(compare_dir / "metrics.csv")
    assert "benchmark_name" in df.columns
    bench_names = set(df["benchmark_name"].unique())
    assert "bench_a" in bench_names
    assert "bench_b" in bench_names
