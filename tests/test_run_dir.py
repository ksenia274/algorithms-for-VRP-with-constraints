import pytest
from pathlib import Path

from algorithms.algorithm_params import HgsSimpleParams
from algorithms.solver_result import SolverConfig, SolverDiagnostics, SolverResult
from runtime.run_dir import create_run_dir, save_run, load_run, list_runs


def _config(algorithm: str = "hgs_simple", instance: str = "03") -> SolverConfig:
    return SolverConfig(
        schema_version="1.0",
        algorithm=algorithm,
        instance=instance,
        instance_kind="yandex",
        time_limit=30,
        seed=42,
        capacity=200,
        num_vehicles=25,
        algorithm_params=HgsSimpleParams(),
    )


def _result(config: SolverConfig) -> SolverResult:
    return SolverResult.from_routes(
        routes=[[1, 2], [3]],
        distance_matrix=[[0, 10, 20, 30], [10, 0, 15, 25], [20, 15, 0, 12], [30, 25, 12, 0]],
        loc_loads=[0.0, 5.0, 8.0, 3.0],
        feasible=True,
        config=config,
        diagnostics=SolverDiagnostics(solve_time_s=2.5),
    )


def test_create_run_dir_unique_naming(tmp_results_dir: Path):
    cfg = _config()
    base = tmp_results_dir / "runs"
    base.mkdir(parents=True, exist_ok=True)
    d1 = create_run_dir(cfg, base_dir=base)
    d2 = create_run_dir(cfg, base_dir=base)
    assert d1 != d2
    assert d1.exists()
    assert d2.exists()


def test_save_load_run_roundtrip(tmp_results_dir: Path):
    cfg = _config()
    base = tmp_results_dir / "runs"
    base.mkdir(parents=True, exist_ok=True)
    run_dir = create_run_dir(cfg, base_dir=base)
    original = _result(cfg)
    save_run(original, run_dir)

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "result.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "routes.txt").exists()

    loaded = load_run(run_dir)
    assert loaded.total_distance == pytest.approx(original.total_distance)
    assert loaded.routes == original.routes


def test_list_runs_with_filters(tmp_results_dir: Path):
    base = tmp_results_dir / "runs"
    base.mkdir(parents=True, exist_ok=True)

    cfg_a = _config(algorithm="hgs_simple", instance="01")
    cfg_b = _config(algorithm="alns", instance="03")

    for cfg in [cfg_a, cfg_b]:
        run_dir = create_run_dir(cfg, base_dir=base)
        save_run(_result(cfg), run_dir)

    all_runs = list_runs(base_dir=base)
    assert len(all_runs) == 2

    hgs_runs = list_runs(base_dir=base, algorithm="hgs_simple")
    assert len(hgs_runs) == 1
    assert "hgs_simple" in hgs_runs[0].name

    inst_runs = list_runs(base_dir=base, instance="03")
    assert len(inst_runs) == 1
    assert "_03_" in inst_runs[0].name
