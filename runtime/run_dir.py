from __future__ import annotations

import itertools
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from algorithms.solver_result import SolverConfig, SolverResult

_counter = itertools.count()


def _iso_now() -> str:
    return f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}-{next(_counter):04d}"


def _default_base() -> Path:
    from runtime.global_config import get_global_config
    cfg = get_global_config()
    return Path(cfg.paths.results) / "runs"


def create_run_dir(
    config: SolverConfig,
    *,
    base_dir: Optional[Path] = None,
) -> Path:
    """Create and return a new unique run directory.

    Name: {ISO-timestamp}_{algorithm}_{instance}_{config_hash}
    """
    base = base_dir if base_dir is not None else _default_base()
    name = f"{_iso_now()}_{config.algorithm}_{config.instance}_{config.content_hash()}"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run(result: SolverResult, run_dir: Path) -> None:
    """Write config.yaml, result.json, metrics.csv, and routes.txt into run_dir."""
    from runtime.serialization import (
        append_metrics_row,
        save_config_yaml,
        save_result_json,
    )

    save_config_yaml(result.config, run_dir / "config.yaml")
    save_result_json(result, run_dir / "result.json")
    append_metrics_row(result, run_dir / "metrics.csv", run_dir=run_dir)
    _write_routes_txt(result, run_dir / "routes.txt")


def _write_routes_txt(result: SolverResult, path: Path) -> None:
    lines = [
        f"Algorithm : {result.config.algorithm}",
        f"Instance  : {result.config.instance} ({result.config.instance_kind})",
        f"Feasible  : {result.feasible}",
        f"Distance  : {result.total_distance:.1f}",
        f"Routes    : {result.num_routes}",
        "",
    ]
    for i, route in enumerate(result.routes, 1):
        lines.append(f"Route {i:>3}: {' -> '.join(str(c) for c in route)}")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_run(run_dir: Path) -> SolverResult:
    """Load a SolverResult from a run directory (reads result.json)."""
    from runtime.serialization import load_result_json
    return load_result_json(run_dir / "result.json")


def list_runs(
    base_dir: Optional[Path] = None,
    algorithm: Optional[str] = None,
    instance: Optional[str] = None,
) -> list[Path]:
    """Return sorted list of run directories matching optional filters."""
    base = base_dir if base_dir is not None else _default_base()
    if not base.exists():
        return []
    results = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if algorithm and f"_{algorithm}_" not in p.name:
            continue
        if instance and f"_{instance}_" not in p.name:
            continue
        results.append(p)
    return results
