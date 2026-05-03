import pytest
from algorithms.algorithm_params import HgsSimpleParams
from algorithms.solver_result import SolverConfig, SolverDiagnostics, SolverResult
from runtime.global_config import GlobalConfig, MetricsConfig, set_global_config_for_testing, reset_global_config


def _make_config(algorithm: str = "hgs_simple") -> SolverConfig:
    return SolverConfig(
        schema_version="1.0",
        algorithm=algorithm,
        instance="03",
        instance_kind="yandex",
        time_limit=30,
        seed=42,
        capacity=200,
        num_vehicles=25,
        algorithm_params=HgsSimpleParams(),
    )


def _make_diagnostics() -> SolverDiagnostics:
    return SolverDiagnostics(solve_time_s=1.23)


def _make_feasible_result() -> SolverResult:
    dm = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
    loads = [0.0, 5.0, 8.0]
    return SolverResult.from_routes(
        routes=[[1], [2]],
        distance_matrix=dm,
        loc_loads=loads,
        feasible=True,
        config=_make_config(),
        diagnostics=_make_diagnostics(),
    )


@pytest.fixture(autouse=True)
def reset_cfg():
    yield
    reset_global_config()


def test_infeasible_constructor():
    r = SolverResult.infeasible(config=_make_config(), diagnostics=_make_diagnostics())
    assert not r.feasible
    assert r.total_distance == float("inf")
    assert r.routes == []
    assert r.fairness is None


def test_initial_must_not_nest():
    inner = _make_feasible_result()
    outer = SolverResult.from_routes(
        routes=[[1], [2]],
        distance_matrix=[[0, 10, 20], [10, 0, 15], [20, 15, 0]],
        loc_loads=[0.0, 5.0, 8.0],
        feasible=True,
        config=_make_config(),
        diagnostics=_make_diagnostics(),
        initial=inner,
    )
    assert outer.initial is inner
    assert inner.initial is None


def test_solver_result_serialization_roundtrip():
    original = _make_feasible_result()
    data = original.to_dict()
    restored = SolverResult.from_dict(data)

    assert restored.total_distance == pytest.approx(original.total_distance)
    assert restored.routes == original.routes
    assert restored.feasible == original.feasible
    assert restored.config.algorithm == original.config.algorithm
    assert restored.diagnostics.solve_time_s == pytest.approx(original.diagnostics.solve_time_s)
    assert restored.fairness.distance.gini == pytest.approx(original.fairness.distance.gini)


def test_primary_metric_reads_global_config():
    set_global_config_for_testing(
        GlobalConfig(metrics=MetricsConfig(primary="dist_gini"))
    )
    r = _make_feasible_result()
    val = r.primary_metric()
    assert val == pytest.approx(r.fairness.distance.gini)


def test_primary_metric_infeasible_returns_inf():
    r = SolverResult.infeasible(config=_make_config(), diagnostics=_make_diagnostics())
    set_global_config_for_testing(
        GlobalConfig(metrics=MetricsConfig(primary="dist_worst_ratio"))
    )
    assert r.primary_metric() == float("inf")
