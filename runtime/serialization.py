from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

if TYPE_CHECKING:
    from algorithms.solver_result import SolverConfig, SolverResult

_METRICS_CSV_HEADER = [
    "schema_version",
    "algorithm",
    "instance",
    "instance_kind",
    "seed",
    "feasible",
    "total_distance",
    "num_routes",
    "solve_time_s",
    "rebalance_moves",
    "cost_delta_pct",
    "weight_applied_count",
    "iterations",
    "dist_worst_ratio",
    "dist_gini",
    "dist_cv",
    "load_worst_ratio",
    "load_gini",
    "load_cv",
    "clients_worst_ratio",
    "clients_gini",
    "clients_cv",
    "dist_worst_ratio_initial",
    "dist_gini_initial",
    "load_worst_ratio_initial",
    "load_gini_initial",
    "clients_worst_ratio_initial",
    "clients_gini_initial",
    "config_hash",
    "run_dir",
]


def save_result_json(result: SolverResult, path: Path) -> None:
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def load_result_json(path: Path) -> SolverResult:
    from algorithms.solver_result import SolverResult
    data = json.loads(path.read_text(encoding="utf-8"))
    return SolverResult.from_dict(data)


def save_config_yaml(config: SolverConfig, path: Path) -> None:
    path.write_text(config.to_yaml(), encoding="utf-8")


def load_config_yaml(path: Path) -> SolverConfig:
    from algorithms.solver_result import SolverConfig
    return SolverConfig.from_yaml(path.read_text(encoding="utf-8"))


def append_metrics_row(
    result: SolverResult,
    csv_path: Path,
    *,
    run_dir: Optional[Path] = None,
) -> None:
    """Append one metrics row. Creates file with header if missing."""
    write_header = not csv_path.exists()
    f = result.fairness
    fi = result.initial.fairness if result.initial is not None else None
    diag = result.diagnostics

    row = {
        "schema_version": result.schema_version,
        "algorithm": result.config.algorithm,
        "instance": result.config.instance,
        "instance_kind": result.config.instance_kind,
        "seed": result.config.seed,
        "feasible": result.feasible,
        "total_distance": result.total_distance,
        "num_routes": result.num_routes,
        "solve_time_s": diag.solve_time_s,
        "rebalance_moves": diag.rebalance_moves,
        "cost_delta_pct": diag.cost_delta_pct,
        "weight_applied_count": diag.weight_applied_count,
        "iterations": diag.iterations,
        "dist_worst_ratio": f.distance.worst_ratio if f else None,
        "dist_gini": f.distance.gini if f else None,
        "dist_cv": f.distance.cv if f else None,
        "load_worst_ratio": f.load.worst_ratio if f else None,
        "load_gini": f.load.gini if f else None,
        "load_cv": f.load.cv if f else None,
        "clients_worst_ratio": f.clients.worst_ratio if f else None,
        "clients_gini": f.clients.gini if f else None,
        "clients_cv": f.clients.cv if f else None,
        "dist_worst_ratio_initial": fi.distance.worst_ratio if fi else None,
        "dist_gini_initial": fi.distance.gini if fi else None,
        "load_worst_ratio_initial": fi.load.worst_ratio if fi else None,
        "load_gini_initial": fi.load.gini if fi else None,
        "clients_worst_ratio_initial": fi.clients.worst_ratio if fi else None,
        "clients_gini_initial": fi.clients.gini if fi else None,
        "config_hash": result.config.content_hash(),
        "run_dir": str(run_dir) if run_dir is not None else None,
    }
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_METRICS_CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
