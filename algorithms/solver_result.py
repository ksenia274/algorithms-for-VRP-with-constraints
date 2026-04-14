from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from pyvrp import ProblemData

from algorithms.fairness_metrics import FairnessReport, compute_fairness


@dataclass
class SolverResult:
    routes: list[list[int]]
    total_distance: float
    num_routes: int
    feasible: bool
    fairness: FairnessReport | None
    metadata: dict[str, Any]

    @classmethod
    def infeasible(cls, **kwargs) -> SolverResult:
        return cls(
            routes=[],
            total_distance=float("inf"),
            num_routes=0,
            feasible=False,
            fairness=None,
            metadata=kwargs,
        )

    @classmethod
    def from_routes(
        cls,
        routes: list[list[int]],
        distance_matrix: Sequence[Sequence[float]],
        loc_loads: Sequence[float],
        **kwargs,
    ) -> SolverResult:
        depot = 0
        distances, loads, clients_count, durations = [], [], [], []

        for route in routes:
            d, prev = 0, depot
            for c in route:
                d += distance_matrix[prev][c]
                prev = c
            d += distance_matrix[prev][depot]
            distances.append(float(d))
            loads.append(float(sum(loc_loads[c] for c in route)))
            clients_count.append(len(route))
            durations.append(float(d))

        fairness = compute_fairness(
            route_distances=distances,
            route_loads=loads,
            route_clients=clients_count,
            route_durations=durations,
        )
        return cls(
            routes=routes,
            total_distance=sum(distances),
            num_routes=len(routes),
            feasible=True,
            fairness=fairness,
            metadata=kwargs,
        )

    @classmethod
    def from_routes_pyvrp_adapter(
        cls, routes: list[list[int]], data: ProblemData, **kwargs
    ) -> SolverResult:
        dm = data.distance_matrix(0)
        loc_loads = [0.0] * data.num_locations
        for j in range(data.num_clients):
            client = data.client(j)
            if client.delivery:
                loc_loads[client.location] = float(client.delivery[0])
        return cls.from_routes(routes=routes, distance_matrix=dm, loc_loads=loc_loads, **kwargs)
