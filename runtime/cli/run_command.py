from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

from algorithms.factory import build_solver
from algorithms.solver_result import SolverConfig
from data.instance_resolver import resolve_instance
from runtime.run_dir import create_run_dir, save_run
from runtime.serialization import load_config_yaml


def _apply_overrides(
    config: SolverConfig,
    *,
    seed: Optional[int],
    time_limit: Optional[int],
    instance: Optional[str],
) -> SolverConfig:
    if seed is None and time_limit is None and instance is None:
        return config
    return dataclasses.replace(
        config,
        seed=seed if seed is not None else config.seed,
        time_limit=time_limit if time_limit is not None else config.time_limit,
        instance=instance if instance is not None else config.instance,
    )


def cmd_run(
    config_path: Path,
    *,
    seed: Optional[int] = None,
    time_limit: Optional[int] = None,
    instance: Optional[str] = None,
    output: Optional[Path] = None,
) -> Path:
    """Load SolverConfig from YAML, apply overrides, run solve(), save result."""
    config = load_config_yaml(config_path)
    config = _apply_overrides(config, seed=seed, time_limit=time_limit, instance=instance)

    if output is not None:
        run_dir = Path(output)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(config)

    instance_path = resolve_instance(config.instance, config.instance_kind)
    solver = build_solver(config)

    if config.algorithm == "hgs_adaptive":
        result = solver.solve(instance_path, run_dir=run_dir)
    else:
        result = solver.solve(instance_path)

    save_run(result, run_dir)

    primary = result.primary_metric()
    print(f"run_dir   : {run_dir}")
    print(f"feasible  : {result.feasible}")
    print(f"distance  : {result.total_distance:.1f}")
    print(f"primary   : {primary:.4f}")

    return run_dir
