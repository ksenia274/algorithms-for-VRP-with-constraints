from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data.instance_resolver import resolve_yandex_path
from runtime.cli.benchmark_command import cmd_benchmark


def _find_yandex_instance_name() -> str | None:
    for name in ("3", "0", "1"):
        if resolve_yandex_path(name) is not None:
            return name
    return None


def _write_bench_config(path: Path, instance: str, time_limit: int = 3) -> None:
    path.write_text(f"""
name: test_bench
instances:
  - {{name: "{instance}", kind: yandex}}
algorithms:
  - name: simple
    type: hgs_simple
    algorithm_params: {{}}
shared:
  time_limit: {time_limit}
  seed: 42
  capacity: 200
  num_vehicles: 25
""")


def test_cmd_benchmark_creates_structure(tmp_results_dir):
    inst = _find_yandex_instance_name()
    if inst is None:
        pytest.skip("No yandex instances found on disk")

    cfg = tmp_results_dir / "bench.yaml"
    _write_bench_config(cfg, inst)
    out = tmp_results_dir / "bench_out"

    bench_dir = cmd_benchmark(cfg, output=out)

    assert (bench_dir / "runs").is_dir()
    assert (bench_dir / "metrics.csv").exists()
    run_dirs = list((bench_dir / "runs").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "result.json").exists()


def test_cmd_benchmark_aggregates_metrics(tmp_results_dir):
    from data.instance_resolver import resolve_yandex_path
    inst0 = "0" if resolve_yandex_path("0") else None
    inst1 = "1" if resolve_yandex_path("1") else None
    if inst0 is None or inst1 is None:
        pytest.skip("Need both yandex instances '0' and '1' on disk")

    cfg = tmp_results_dir / "bench2.yaml"
    cfg.write_text(f"""
name: two_instances
instances:
  - {{name: "{inst0}", kind: yandex}}
  - {{name: "{inst1}", kind: yandex}}
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
    bench_dir = cmd_benchmark(cfg, output=tmp_results_dir / "two_out")
    df = pd.read_csv(bench_dir / "metrics.csv")
    assert len(df) == 2  # two different instances → two distinct runs → two rows


def test_cmd_benchmark_continues_on_failure(tmp_results_dir):
    inst = _find_yandex_instance_name()
    if inst is None:
        pytest.skip("No yandex instances found on disk")

    cfg = tmp_results_dir / "fail_bench.yaml"
    _write_bench_config(cfg, inst)
    out = tmp_results_dir / "fail_out"

    with patch("runtime.cli.benchmark_command.build_solver", side_effect=RuntimeError("boom")):
        bench_dir = cmd_benchmark(cfg, output=out)

    assert (bench_dir / "log.txt").exists()
    log = (bench_dir / "log.txt").read_text()
    assert "boom" in log
    # metrics.csv should still exist (failure row written)
    assert (bench_dir / "metrics.csv").exists()
