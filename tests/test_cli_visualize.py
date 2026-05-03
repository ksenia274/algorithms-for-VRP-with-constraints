from __future__ import annotations

import pytest
from pathlib import Path

from data.instance_resolver import resolve_yandex_path
from runtime.cli.benchmark_command import cmd_benchmark
from runtime.cli.run_command import cmd_run
from runtime.cli.visualize_command import cmd_visualize
from runtime.serialization import save_config_yaml
from algorithms.solver_result import SolverConfig
from algorithms.algorithm_params import HgsSimpleParams


def _find_yandex_instance_name() -> str | None:
    for name in ("3", "0", "1"):
        if resolve_yandex_path(name) is not None:
            return name
    return None


def _simple_run(base: Path, inst: str) -> Path:
    cfg_path = base / "config.yaml"
    config = SolverConfig(
        schema_version="1.0",
        algorithm="hgs_simple",
        instance=inst,
        instance_kind="yandex",
        time_limit=3,
        seed=42,
        capacity=200,
        num_vehicles=25,
        algorithm_params=HgsSimpleParams(),
    )
    save_config_yaml(config, cfg_path)
    return cmd_run(cfg_path, output=base / "run")


def _simple_bench(base: Path, inst: str) -> Path:
    cfg = base / "bench.yaml"
    cfg.write_text(f"""
name: vis_test
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
    return cmd_benchmark(cfg, output=base / "bench")


def test_cmd_visualize_run_smoke(tmp_results_dir):
    inst = _find_yandex_instance_name()
    if inst is None:
        pytest.skip("No yandex instances found on disk")

    run_dir = _simple_run(tmp_results_dir, inst)
    # Should not raise
    cmd_visualize(run_dir)


def test_cmd_visualize_run_creates_plots(tmp_results_dir):
    inst = _find_yandex_instance_name()
    if inst is None:
        pytest.skip("No yandex instances found on disk")

    run_dir = _simple_run(tmp_results_dir, inst)
    cmd_visualize(run_dir)

    plots_dir = run_dir / "plots"
    pngs = list(plots_dir.glob("*.png"))
    assert len(pngs) > 0, "No PNGs created in single-run plots/"


def test_cmd_visualize_benchmark_smoke(tmp_results_dir):
    inst = _find_yandex_instance_name()
    if inst is None:
        pytest.skip("No yandex instances found on disk")

    bench_dir = _simple_bench(tmp_results_dir, inst)
    cmd_visualize(bench_dir)


def test_cmd_visualize_benchmark_creates_structure(tmp_results_dir):
    inst = _find_yandex_instance_name()
    if inst is None:
        pytest.skip("No yandex instances found on disk")

    bench_dir = _simple_bench(tmp_results_dir, inst)
    cmd_visualize(bench_dir)

    plots_dir = bench_dir / "plots"
    agg_dir   = plots_dir / "aggregated"
    pi_dir    = plots_dir / "per_instance"

    assert agg_dir.is_dir(), "aggregated/ not created"
    assert pi_dir.is_dir(),  "per_instance/ not created"

    # aggregated must have at least 01/02/03/04
    agg_pngs = list(agg_dir.glob("*.png"))
    assert len(agg_pngs) >= 4, f"Expected ≥4 PNGs in aggregated/, got {[p.name for p in agg_pngs]}"

    # per_instance must have subdirectories for each Pareto/profile plot
    pi_subdirs = [p for p in pi_dir.iterdir() if p.is_dir()]
    assert len(pi_subdirs) >= 3, (
        f"Expected ≥3 per_instance/ subdirs, got {[p.name for p in pi_subdirs]}"
    )


def test_cmd_visualize_invalid_path(tmp_results_dir):
    empty = tmp_results_dir / "empty_dir"
    empty.mkdir()
    with pytest.raises(ValueError, match="Cannot detect type"):
        cmd_visualize(empty)
