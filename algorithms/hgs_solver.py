from __future__ import annotations

import pyvrp
import pyvrp.stop

from algorithms.fairness_rebalancer import rebalance
from algorithms.hgs_base import HGSBase
from algorithms.solver_result import SolverResult
from data.vrp_instance import VRPInstanceInput


class HGSSolver(HGSBase):
    def __init__(
        self,
        time_limit: int = 60,
        seed: int = 0,
        vehicle_capacity: int = 100,
        num_vehicles: int = 25,
        enable_fairness: bool = True,
        max_cost_increase_pct: float = 5.0,
        rebalance_iterations: int = 3000,
        use_prizes: bool = False,
        display: bool = True,
    ):
        super().__init__(time_limit, seed, vehicle_capacity, num_vehicles)
        self.enable_fairness = enable_fairness
        self.max_cost_increase_pct = max_cost_increase_pct
        self.rebalance_iterations = rebalance_iterations
        self.use_prizes = use_prizes
        self.display = display

    def solve(self, instance: str | VRPInstanceInput) -> tuple[SolverResult, SolverResult]:
        m, data, _, _ = self._build_model(instance, use_prizes=self.use_prizes)

        result = m.solve(stop=pyvrp.stop.MaxRuntime(self.time_limit), seed=self.seed, display=self.display)
        best = result.best

        if not best.is_feasible():
            inf = SolverResult.infeasible(rebalance_moves=0, cost_delta_pct=0.0)
            return inf, inf

        raw_routes = self._extract_routes(best)
        hgs_distance = best.distance()
        before = SolverResult.from_routes_pyvrp_adapter(raw_routes, data)

        if self.enable_fairness and len(raw_routes) >= 2:
            rb = rebalance(
                data,
                raw_routes,
                max_cost_increase_pct=self.max_cost_increase_pct,
                max_iterations=self.rebalance_iterations,
                seed=self.seed,
            )
            final_routes = rb.routes
            final_distance = rb.total_distance
            rebalance_moves = rb.moves_applied
        else:
            final_routes = raw_routes
            final_distance = float(hgs_distance)
            rebalance_moves = 0

        cost_delta = (
            (final_distance - hgs_distance) / hgs_distance * 100.0
            if hgs_distance > 0
            else 0.0
        )

        after = SolverResult.from_routes_pyvrp_adapter(
            final_routes, data,
            rebalance_moves=rebalance_moves,
            cost_delta_pct=cost_delta,
        )
        return before, after
