from __future__ import annotations

import time

import pyvrp
import pyvrp.stop

from algorithms.algorithm_params import HgsSimpleParams
from algorithms.hgs_base import HGSBase
from algorithms.solver_result import SolverConfig, SolverDiagnostics, SolverResult
from data.vrp_instance import VRPInstanceInput


class HGSSolverSimple(HGSBase):
    """Vanilla HGS with optional max-distance constraint. No fairness post-processing."""

    def __init__(self, config: SolverConfig) -> None:
        if not isinstance(config.algorithm_params, HgsSimpleParams):
            raise TypeError(f"Expected HgsSimpleParams, got {type(config.algorithm_params)}")
        super().__init__(config)
        self.params: HgsSimpleParams = config.algorithm_params

    def solve(self, instance: str | VRPInstanceInput) -> SolverResult:
        t0 = time.perf_counter()
        max_dist_int = int(self.params.max_distance) if self.params.max_distance else None
        m, data, _, _ = self._build_model(instance, max_distance=max_dist_int)

        result = m.solve(
            stop=pyvrp.stop.MaxRuntime(self.time_limit),
            seed=self.seed,
        )
        solve_time = time.perf_counter() - t0
        best = result.best

        diagnostics = SolverDiagnostics(solve_time_s=solve_time)

        if not best.is_feasible():
            return SolverResult.infeasible(config=self.config, diagnostics=diagnostics)

        routes = self._extract_routes(best)
        return self._make_result(routes, data, diagnostics=diagnostics)
