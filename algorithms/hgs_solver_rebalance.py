from __future__ import annotations

import time

import pyvrp
import pyvrp.stop

from algorithms.algorithm_params import HgsRebalanceParams
from algorithms.fairness_rebalancer import rebalance
from algorithms.hgs_base import HGSBase
from algorithms.solver_result import SolverConfig, SolverDiagnostics, SolverResult
from data.vrp_instance import VRPInstanceInput


class HGSSolverRebalance(HGSBase):
    """HGS with optional fairness rebalancing post-processing (relocate/swap)."""

    def __init__(self, config: SolverConfig) -> None:
        if not isinstance(config.algorithm_params, HgsRebalanceParams):
            raise TypeError(f"Expected HgsRebalanceParams, got {type(config.algorithm_params)}")
        super().__init__(config)
        self.params: HgsRebalanceParams = config.algorithm_params

    def solve(self, instance: str | VRPInstanceInput) -> SolverResult:
        t0 = time.perf_counter()
        m, data, _, _ = self._build_model(instance, use_prizes=self.params.use_prizes)

        result = m.solve(
            stop=pyvrp.stop.MaxRuntime(self.time_limit),
            seed=self.seed,
            display=False,
        )
        best = result.best

        if not best.is_feasible():
            diagnostics = SolverDiagnostics(
                solve_time_s=time.perf_counter() - t0,
                rebalance_moves=0,
                cost_delta_pct=0.0,
            )
            return SolverResult.infeasible(config=self.config, diagnostics=diagnostics)

        raw_routes = self._extract_routes(best)
        hgs_distance = float(best.distance())

        before_diag = SolverDiagnostics(solve_time_s=0.0)
        before = self._make_result(raw_routes, data, diagnostics=before_diag)

        if self.params.enable_fairness and len(raw_routes) >= 2:
            rb = rebalance(
                data,
                raw_routes,
                max_cost_increase_pct=self.params.max_cost_increase_pct,
                max_iterations=self.params.rebalance_iterations,
                seed=self.seed,
            )
            final_routes = rb.routes
            final_distance = rb.total_distance
            rebalance_moves = rb.moves_applied
        else:
            final_routes = raw_routes
            final_distance = hgs_distance
            rebalance_moves = 0

        cost_delta = (
            (final_distance - hgs_distance) / hgs_distance * 100.0
            if hgs_distance > 0 else 0.0
        )

        diagnostics = SolverDiagnostics(
            solve_time_s=time.perf_counter() - t0,
            rebalance_moves=rebalance_moves,
            cost_delta_pct=cost_delta,
        )
        return self._make_result(final_routes, data, diagnostics=diagnostics, initial=before)
