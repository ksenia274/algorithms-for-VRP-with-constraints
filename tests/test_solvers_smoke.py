from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from algorithms.algorithm_params import (
    AlnsParams,
    HgsAdaptiveParams,
    HgsPenaltyParams,
    HgsRebalanceParams,
    HgsSimpleParams,
    ALGORITHM_PARAMS_REGISTRY,
)
from algorithms.factory import build_solver
from algorithms.solver_result import SolverConfig, SolverResult
from data.instance_resolver import resolve_yandex_path


def _find_yandex_instance() -> Optional[str]:
    """Return path to first available yandex instance, or None."""
    for name in ("3", "0", "1"):
        path = resolve_yandex_path(name)
        if path is not None:
            return path
    return None


def _make_config(algorithm: str, time_limit: int = 5) -> SolverConfig:
    params_cls = ALGORITHM_PARAMS_REGISTRY[algorithm]
    if algorithm == "hgs_adaptive":
        params = HgsAdaptiveParams(initial_route_balance=500.0, strategy="linear", trace=False)
    else:
        params = params_cls()
    return SolverConfig(
        schema_version="1.0",
        algorithm=algorithm,
        instance="3",
        instance_kind="yandex",
        time_limit=time_limit,
        seed=42,
        capacity=200,
        num_vehicles=25,
        algorithm_params=params,
    )


@pytest.mark.parametrize("algorithm", [
    "hgs_simple",
    "hgs_rebalance",
    "hgs_penalty",
    "hgs_adaptive",
    "alns",
])
def test_solver_returns_valid_result(algorithm, tmp_results_dir):
    instance = _find_yandex_instance()
    if instance is None:
        pytest.skip("No yandex instances found on disk")

    config = _make_config(algorithm)
    solver = build_solver(config)
    result = solver.solve(instance)

    assert isinstance(result, SolverResult)
    assert result.config.algorithm == algorithm
    assert result.diagnostics.solve_time_s > 0
    if result.feasible:
        assert len(result.routes) > 0
        assert result.fairness is not None


def test_hgs_simple_no_initial(tmp_results_dir):
    instance = _find_yandex_instance()
    if instance is None:
        pytest.skip("No yandex instances found on disk")

    config = _make_config("hgs_simple")
    result = build_solver(config).solve(instance)
    assert result.initial is None


def test_hgs_rebalance_has_initial(tmp_results_dir):
    instance = _find_yandex_instance()
    if instance is None:
        pytest.skip("No yandex instances found on disk")

    config = _make_config("hgs_rebalance")
    result = build_solver(config).solve(instance)
    if result.feasible:
        assert result.initial is not None
        assert result.initial.initial is None


def test_adaptive_writes_artifacts(tmp_results_dir):
    instance = _find_yandex_instance()
    if instance is None:
        pytest.skip("No yandex instances found on disk")

    run_dir = tmp_results_dir / "adaptive_run"
    run_dir.mkdir()

    params = HgsAdaptiveParams(initial_route_balance=500.0, strategy="linear", trace=True)
    config = SolverConfig(
        schema_version="1.0",
        algorithm="hgs_adaptive",
        instance="3",
        instance_kind="yandex",
        time_limit=5,
        seed=42,
        capacity=200,
        num_vehicles=25,
        algorithm_params=params,
    )
    from algorithms.hgs_solver_adaptive import HGSSolverAdaptive
    result = HGSSolverAdaptive(config).solve(instance, run_dir=run_dir)

    assert isinstance(result, SolverResult)
    if result.feasible and result.artifacts:
        for key, rel_path in result.artifacts.items():
            assert (run_dir / rel_path).exists(), f"Artifact '{key}' not found at {run_dir / rel_path}"
