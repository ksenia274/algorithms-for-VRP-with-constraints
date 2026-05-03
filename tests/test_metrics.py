import pytest
from metrics.fairness import (
    FairnessReport,
    DimensionMetrics,
    coefficient_of_variation,
    compute_fairness,
)


def _equal_routes_fairness() -> FairnessReport:
    return compute_fairness(
        route_distances=[100.0, 100.0, 100.0],
        route_loads=[50.0, 50.0, 50.0],
        route_clients=[5, 5, 5],
    )


def test_compute_fairness_equal_routes():
    f = _equal_routes_fairness()
    assert f.distance.gini == pytest.approx(0.0)
    assert f.distance.worst_ratio == pytest.approx(1.0)
    assert f.distance.cv == pytest.approx(0.0)


def test_compute_fairness_one_dominant_route():
    f = compute_fairness(
        route_distances=[200.0, 50.0, 50.0],
        route_loads=[100.0, 50.0, 50.0],
        route_clients=[10, 3, 3],
    )
    assert f.distance.worst_ratio > 1.0
    assert f.distance.gini > 0.0


def test_coefficient_of_variation_zero_mean():
    assert coefficient_of_variation([0.0, 0.0, 0.0]) == 0.0


def test_coefficient_of_variation_empty():
    assert coefficient_of_variation([]) == 0.0


def test_value_valid_metric():
    f = _equal_routes_fairness()
    val = f.value("dist_worst_ratio")
    assert val == pytest.approx(1.0)


def test_value_invalid_metric_name_raises():
    f = _equal_routes_fairness()
    with pytest.raises(ValueError, match="Unknown metric"):
        f.value("nonexistent_metric")
