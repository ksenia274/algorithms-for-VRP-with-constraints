from __future__ import annotations

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

from algorithms.algorithm_params import ALGORITHM_PARAMS_REGISTRY
from algorithms.factory import build_solver
from algorithms.solver_result import SolverConfig, SolverDiagnostics, SolverResult
from data.instance_resolver import resolve_instance
from runtime.benchmark_dir import aggregate_metrics, create_benchmark_dir
from runtime.cli.benchmark_config import BenchmarkConfig
from runtime.run_dir import save_run
from runtime.serialization import append_metrics_row

logger = logging.getLogger(__name__)


def _load_benchmark_config(config_path: Path) -> BenchmarkConfig:
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return BenchmarkConfig.model_validate(data)


def _build_solver_config(
    algo_spec,
    inst_spec,
    shared,
    *,
    seed_override: Optional[int],
    time_limit_override: Optional[int],
) -> SolverConfig:
    params_cls = ALGORITHM_PARAMS_REGISTRY.get(algo_spec.type)
    if params_cls is None:
        raise ValueError(f"Unknown algorithm type '{algo_spec.type}'")
    params = params_cls.model_validate(algo_spec.algorithm_params)

    return SolverConfig(
        schema_version="1.0",
        algorithm=algo_spec.name,
        algorithm_type=algo_spec.type,
        instance=inst_spec.name,
        instance_kind=inst_spec.kind,
        time_limit=time_limit_override if time_limit_override is not None else shared.time_limit,
        seed=seed_override if seed_override is not None else shared.seed,
        capacity=shared.capacity,
        num_vehicles=shared.num_vehicles,
        algorithm_params=params,
    )


def _run_dir_name(algo_name: str, inst_name: str, config_hash: str) -> str:
    return f"{algo_name}_{inst_name}_{config_hash}"


def cmd_benchmark(
    config_path: Path,
    *,
    seed: Optional[int] = None,
    time_limit: Optional[int] = None,
    instance: Optional[str] = None,
    output: Optional[Path] = None,
) -> Path:
    """Run all algorithm × instance combinations from a BenchmarkConfig YAML."""
    bench_cfg = _load_benchmark_config(config_path)

    instances = bench_cfg.instances
    if instance is not None:
        instances = [i for i in instances if i.name == instance]

    if output is not None:
        bench_dir = Path(output)
        (bench_dir / "runs").mkdir(parents=True, exist_ok=True)
        (bench_dir / "plots").mkdir(parents=True, exist_ok=True)
    else:
        bench_dir = create_benchmark_dir(bench_cfg.name)

    log_path = bench_dir / "log.txt"
    total = len(instances) * len(bench_cfg.algorithms)

    with tqdm(total=total, unit="run") as pbar:
        for inst_spec in instances:
            for algo_spec in bench_cfg.algorithms:
                pbar.set_description(f"{algo_spec.name} on {inst_spec.name}")

                try:
                    config = _build_solver_config(
                        algo_spec, inst_spec, bench_cfg.shared,
                        seed_override=seed, time_limit_override=time_limit,
                    )
                    run_dir = bench_dir / "runs" / _run_dir_name(
                        algo_spec.name, inst_spec.name, config.content_hash()
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)

                    instance_path = resolve_instance(inst_spec.name, inst_spec.kind)
                    solver = build_solver(config)

                    if config.algorithm == "hgs_adaptive":
                        result = solver.solve(instance_path, run_dir=run_dir)
                    else:
                        result = solver.solve(instance_path)

                    save_run(result, run_dir)

                except Exception as exc:
                    msg = f"FAILED {algo_spec.name} on {inst_spec.name}: {exc}\n{traceback.format_exc()}"
                    logger.error(msg)
                    with open(log_path, "a", encoding="utf-8") as lf:
                        lf.write(f"[{datetime.now().isoformat()}] {msg}\n")

                    _write_failure_row(bench_dir / "runs", algo_spec, inst_spec, bench_cfg.shared)

                pbar.update(1)

    aggregate_metrics(bench_dir)
    print(f"benchmark : {bench_dir}")
    print(f"metrics   : {bench_dir / 'metrics.csv'}")
    return bench_dir


def _write_failure_row(
    runs_dir: Path,
    algo_spec,
    inst_spec,
    shared,
) -> None:
    from algorithms.algorithm_params import ALGORITHM_PARAMS_REGISTRY
    params_cls = ALGORITHM_PARAMS_REGISTRY.get(algo_spec.type)
    params = params_cls() if params_cls else ALGORITHM_PARAMS_REGISTRY["hgs_simple"]()

    config = SolverConfig(
        schema_version="1.0",
        algorithm=algo_spec.name,
        algorithm_type=algo_spec.type,
        instance=inst_spec.name,
        instance_kind=inst_spec.kind,
        time_limit=shared.time_limit,
        seed=shared.seed,
        capacity=shared.capacity,
        num_vehicles=shared.num_vehicles,
        algorithm_params=params,
    )
    diagnostics = SolverDiagnostics(solve_time_s=0.0)
    result = SolverResult.infeasible(config=config, diagnostics=diagnostics)

    run_dir = runs_dir / _run_dir_name(algo_spec.name, inst_spec.name, config.content_hash())
    run_dir.mkdir(parents=True, exist_ok=True)
    append_metrics_row(result, run_dir / "metrics.csv")
