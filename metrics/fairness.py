from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

_DIM_PREFIX = {"dist": "distance", "load": "load", "clients": "clients"}

_VALID_METRIC_NAMES = frozenset(
    f"{prefix}_{attr}"
    for prefix in ("dist", "load", "clients")
    for attr in ("mean", "std", "min", "max", "worst_ratio", "gini", "cv")
)


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Standard CV: std / mean. Returns 0.0 if mean is zero or sequence is empty."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    if mean == 0.0:
        return 0.0
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return std / mean


@dataclass
class DimensionMetrics:
    """Fairness statistics for one route dimension (distance, load, or clients)."""

    mean: float
    std: float
    min: float
    max: float
    worst_ratio: float  # max / mean — 1.0 = ideal
    gini: float         # 0.0 = equal, 1.0 = maximally unequal
    cv: float           # std / mean — standard coefficient of variation


@dataclass
class FairnessReport:
    """Per-dimension fairness statistics for a VRP solution."""

    distance: DimensionMetrics
    load: DimensionMetrics
    clients: DimensionMetrics

    def summary(self) -> str:
        header = f"  {'Dimension':<10} {'worst_ratio':>12} {'gini':>8} {'cv':>8} {'mean':>10} {'std':>8}"
        sep = "  " + "-" * 70
        lines = ["=== Fairness Report ===", header, sep]
        for name, dm in [("Distance", self.distance), ("Load", self.load), ("Clients", self.clients)]:
            lines.append(
                f"  {name:<10} {dm.worst_ratio:>12.4f} {dm.gini:>8.4f}"
                f" {dm.cv:>8.4f} {dm.mean:>10.1f} {dm.std:>8.1f}"
            )
        return "\n".join(lines)

    def value(self, metric_name: str) -> float:
        """Return metric by name, e.g. 'dist_worst_ratio', 'load_gini', 'clients_cv'.

        Raises ValueError for unknown names.
        """
        if metric_name not in _VALID_METRIC_NAMES:
            raise ValueError(
                f"Unknown metric '{metric_name}'. Valid: {sorted(_VALID_METRIC_NAMES)}"
            )
        prefix, attr = metric_name.split("_", 1)
        dim: DimensionMetrics = getattr(self, _DIM_PREFIX[prefix])
        return float(getattr(dim, attr))


def _dim(vals: list[float]) -> DimensionMetrics:
    n = len(vals)
    if n == 0:
        return DimensionMetrics(mean=0.0, std=0.0, min=0.0, max=0.0, worst_ratio=1.0, gini=0.0, cv=0.0)
    mean = sum(vals) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
    lo, hi = min(vals), max(vals)
    worst_ratio = hi / mean if mean > 0 else 1.0
    total = sum(vals)
    if total == 0 or n < 2:
        gini = 0.0
    else:
        abs_diff_sum = sum(abs(vals[i] - vals[j]) for i in range(n) for j in range(n))
        gini = abs_diff_sum / (2 * n * total)
    cv = std / mean if mean > 0 else 0.0
    return DimensionMetrics(mean=mean, std=std, min=lo, max=hi, worst_ratio=worst_ratio, gini=gini, cv=cv)


def compute_fairness(
    route_distances: list[float],
    route_loads: list[float],
    route_clients: list[int],
) -> FairnessReport:
    """Compute per-dimension fairness metrics from route measurement vectors."""
    return FairnessReport(
        distance=_dim([float(v) for v in route_distances]),
        load=_dim([float(v) for v in route_loads]),
        clients=_dim([float(v) for v in route_clients]),
    )


def compute_fairness_for_routes(
    routes: list[list[int]],
    distance_matrix: Sequence[Sequence[float]],
    loc_loads: Sequence[float],
) -> FairnessReport:
    """Compute fairness directly from routes, distance matrix, and load vector."""
    depot = 0
    distances, loads, clients_count = [], [], []
    for route in routes:
        d, prev = 0.0, depot
        for c in route:
            d += distance_matrix[prev][c]
            prev = c
        d += distance_matrix[prev][depot]
        distances.append(d)
        loads.append(float(sum(loc_loads[c] for c in route)))
        clients_count.append(len(route))
    return compute_fairness(
        route_distances=distances,
        route_loads=loads,
        route_clients=clients_count,
    )
