from __future__ import annotations

import pyvrp
import pyvrp.stop

from algorithms.hgs_base import HGSBase
from algorithms.solver_result import SolverResult
from data.vrp_instance import VRPInstanceInput


class HGSSolverSimple(HGSBase):
    def __init__(
        self,
        time_limit: int = 60,
        seed: int = 0,
        vehicle_capacity: int = 100,
        num_vehicles: int = 25,
        max_distance: float | None = None,
    ):
        super().__init__(time_limit, seed, vehicle_capacity, num_vehicles)
        self.max_distance = max_distance

    def solve(self, instance: str | VRPInstanceInput) -> tuple[SolverResult, SolverResult]:
        max_dist_int = int(self.max_distance) if self.max_distance else None
        m, data, _, _ = self._build_model(instance, max_distance=max_dist_int)

        result = m.solve(
            stop=pyvrp.stop.MaxRuntime(self.time_limit),
            seed=self.seed,
        )
        best = result.best

        if not best.is_feasible():
            inf = SolverResult.infeasible()
            return inf, inf

        all_routes_obj = list(best.routes())
        routes = self._extract_routes(best)

        actual_max_duration = max(float(r.duration()) for r in all_routes_obj) if all_routes_obj else 0.0
        actual_max_distance = max(float(r.distance()) for r in all_routes_obj) if all_routes_obj else 0.0

        sol = SolverResult.from_routes_pyvrp_adapter(
            routes, data,
            max_duration=actual_max_duration,
            max_distance=actual_max_distance,
        )
        return sol, sol
