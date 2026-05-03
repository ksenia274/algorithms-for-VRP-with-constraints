from __future__ import annotations

import time

import numpy as np
import pyvrp
import pyvrp.stop

from pyvrp import ActivityType, Route, Solution

from algorithms.algorithm_params import HgsPenaltyParams
from algorithms.hgs_base import HGSBase
from algorithms.solver_result import SolverConfig, SolverDiagnostics, SolverResult
from data.vrp_instance import VRPInstanceInput


class HGSSolverPenalty(HGSBase):
    """HGS with iterative distance-matrix penalty to improve route-length fairness.

    before — first clean run without penalty.
    after  — best solution by Gini across all restarts within cost budget.
    """

    def __init__(self, config: SolverConfig) -> None:
        if not isinstance(config.algorithm_params, HgsPenaltyParams):
            raise TypeError(f"Expected HgsPenaltyParams, got {type(config.algorithm_params)}")
        super().__init__(config)
        self.params: HgsPenaltyParams = config.algorithm_params

    def solve(self, instance: str | VRPInstanceInput) -> SolverResult:
        t0 = time.perf_counter()
        _, base_data, orig_dm, _ = self._build_model(instance)
        time_per_restart = max(1, self.time_limit // self.params.num_restarts)

        result = self._solve_data(base_data, time_per_restart, self.seed)

        if not result.best.is_feasible():
            diagnostics = SolverDiagnostics(
                solve_time_s=time.perf_counter() - t0,
                rebalance_moves=0,
                cost_delta_pct=0.0,
            )
            return SolverResult.infeasible(config=self.config, diagnostics=diagnostics)

        best_solution = result.best
        before = self._make_result(
            self._extract_routes(best_solution),
            base_data,
            diagnostics=SolverDiagnostics(solve_time_s=0.0),
        )

        best_gini = before.fairness.distance.gini if before.fairness else float("inf")
        cost_budget = before.total_distance * (1 + self.params.max_cost_increase_pct / 100.0)

        for iteration in range(self.params.num_restarts - 1):
            routes = self._extract_routes(best_solution)
            penalty = self._compute_penalty_matrix(orig_dm, base_data.num_locations, routes)
            penalized_data = base_data.replace(distance_matrices=[orig_dm + penalty])

            initial = self._convert_solution(best_solution, penalized_data)
            result = self._solve_data(
                penalized_data,
                time_per_restart,
                self.seed + iteration + 1,
                initial_solution=initial,
            )

            if not result.best.is_feasible():
                continue

            candidate_routes = self._extract_routes(result.best)
            candidate_dist = self._routes_distance(candidate_routes, orig_dm)
            candidate = self._make_result(
                candidate_routes, base_data,
                diagnostics=SolverDiagnostics(solve_time_s=0.0),
            )
            candidate_gini = candidate.fairness.distance.gini if candidate.fairness else float("inf")

            if candidate_gini < best_gini and candidate_dist <= cost_budget:
                best_solution = result.best
                best_gini = candidate_gini

        final_routes = self._extract_routes(best_solution)
        final_dist = self._routes_distance(final_routes, orig_dm)
        cost_delta = (
            (final_dist / before.total_distance * 100.0 - 100.0)
            if before.total_distance > 0 else 0.0
        )
        diagnostics = SolverDiagnostics(
            solve_time_s=time.perf_counter() - t0,
            rebalance_moves=self.params.num_restarts - 1,
            cost_delta_pct=cost_delta,
        )
        return self._make_result(final_routes, base_data, diagnostics=diagnostics, initial=before)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_data(data, time_limit: int, seed: int, initial_solution=None):
        model = pyvrp.Model.from_data(data)
        return model.solve(
            stop=pyvrp.stop.MaxRuntime(time_limit),
            seed=seed,
            display=False,
            initial_solution=initial_solution,
        )

    def _compute_penalty_matrix(
        self, dm: np.ndarray, n: int, routes: list[list[int]]
    ) -> np.ndarray:
        route_lengths = [self._route_length(r, dm) for r in routes]
        if not route_lengths:
            return np.zeros((n, n), dtype=np.int64)
        mean_len = sum(route_lengths) / len(route_lengths)
        penalty = np.zeros((n, n), dtype=np.float64)
        for route, length in zip(routes, route_lengths):
            if length <= mean_len:
                continue
            scale = self.params.fairness_weight * (length - mean_len) / mean_len
            prev = 0
            for c in route:
                penalty[prev, c] += dm[prev, c] * scale
                penalty[c, prev] += dm[c, prev] * scale
                prev = c
            penalty[prev, 0] += dm[prev, 0] * scale
            penalty[0, prev] += dm[0, prev] * scale
        return penalty.astype(np.int64)

    @staticmethod
    def _convert_solution(solution, new_data):
        try:
            routes = [
                Route(new_data, visits=[act.idx for act in r if act.type == ActivityType.CLIENT], vehicle_type=0)
                for r in solution.routes()
            ]
            return Solution(new_data, routes)
        except Exception:
            return None

    @staticmethod
    def _route_length(route: list[int], dm: np.ndarray) -> float:
        d, prev = 0.0, 0
        for c in route:
            d += dm[prev, c]
            prev = c
        return d + dm[prev, 0]

    @classmethod
    def _routes_distance(cls, routes: list[list[int]], dm: np.ndarray) -> float:
        return sum(cls._route_length(r, dm) for r in routes)
