from __future__ import annotations

import numpy as np
import pyvrp
import pyvrp.stop
from pyvrp import Solution
from pyvrp._pyvrp import ActivityType, Route

from algorithms.hgs_base import HGSBase
from algorithms.solver_result import SolverResult
from data.vrp_instance import VRPInstanceInput


class HGSSolverPenalty(HGSBase):
    """
    HGS с fairness через итеративную модификацию матрицы расстояний.

    На каждой итерации:
    1. Смотрим текущее лучшее решение (по оригинальным расстояниям)
    2. Считаем длину каждого маршрута
    3. Добавляем штраф к рёбрам маршрутов, длина которых > mean
    4. Перезапускаем HGS с warm start и модифицированной матрицей

    before — лучшее решение первого (чистого) запуска без штрафа.
    after  — лучшее решение по оригинальным расстояниям среди всех рестартов.
    """

    def __init__(
        self,
        time_limit: int = 60,
        seed: int = 0,
        vehicle_capacity: int = 100,
        num_vehicles: int = 25,
        fairness_weight: float = 0.3,
        num_restarts: int = 5,
        max_cost_increase_pct: float = 5.0,
        display: bool = True,
    ):
        super().__init__(time_limit, seed, vehicle_capacity, num_vehicles)
        self.fairness_weight = fairness_weight
        self.num_restarts = num_restarts
        self.max_cost_increase_pct = max_cost_increase_pct
        self.display = display

    def solve(self, instance: str | VRPInstanceInput) -> tuple[SolverResult, SolverResult]:
        _, base_data, orig_dm, _ = self._build_model(instance)
        time_per_restart = max(1, self.time_limit // self.num_restarts)

        result = self._solve_data(base_data, time_per_restart, self.seed, display=self.display)

        if not result.best.is_feasible():
            inf = SolverResult.infeasible(rebalance_moves=0, cost_delta_pct=0.0)
            return inf, inf

        best_solution = result.best
        before = SolverResult.from_routes_pyvrp_adapter(
            self._extract_routes(best_solution),
            base_data,
        )

        best_gini = before.fairness.distance.gini if before.fairness else float("inf")
        cost_budget = before.total_distance * (1 + self.max_cost_increase_pct / 100.0)

        if self.display:
            f = before.fairness
            gini_str = f"{f.distance.gini:.4f}" if f else "n/a"
            print(f"[restart 0/{self.num_restarts - 1}] "
                  f"dist: {before.total_distance:.1f} | routes: {before.num_routes} | gini: {gini_str}")

        for iteration in range(self.num_restarts - 1):
            routes = self._extract_routes(best_solution)
            penalty = self._compute_penalty_matrix(orig_dm, base_data.num_locations, routes)
            penalized_data = base_data.replace(
                distance_matrices=[orig_dm + penalty],
            )

            initial = self._convert_solution(best_solution, penalized_data)
            result = self._solve_data(
                penalized_data,
                time_per_restart,
                self.seed + iteration + 1,
                display=False,
                initial_solution=initial,
            )

            if not result.best.is_feasible():
                continue

            candidate_routes = self._extract_routes(result.best)
            candidate_dist = self._routes_distance(candidate_routes, orig_dm)
            candidate = SolverResult.from_routes_pyvrp_adapter(candidate_routes, base_data)
            candidate_gini = candidate.fairness.distance.gini if candidate.fairness else float("inf")

            if self.display:
                marker = " *" if (candidate_gini < best_gini and candidate_dist <= cost_budget) else ""
                print(f"[restart {iteration + 1}/{self.num_restarts - 1}] "
                      f"dist: {candidate_dist:.1f} | routes: {len(candidate_routes)} | "
                      f"gini: {candidate_gini:.4f}{marker}")

            if candidate_gini < best_gini and candidate_dist <= cost_budget:
                best_solution = result.best
                best_gini = candidate_gini

        final_routes = self._extract_routes(best_solution)
        after = SolverResult.from_routes_pyvrp_adapter(
            final_routes,
            base_data,
            rebalance_moves=self.num_restarts - 1,
            cost_delta_pct=(
                self._routes_distance(final_routes, orig_dm) / before.total_distance * 100.0 - 100.0
                if before.total_distance > 0 else 0.0
            ),
        )
        return before, after

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_data(data, time_limit: int, seed: int, display: bool = True,
                    initial_solution=None):
        model = pyvrp.Model.from_data(data)
        return model.solve(
            stop=pyvrp.stop.MaxRuntime(time_limit),
            seed=seed,
            display=display,
            initial_solution=initial_solution,
        )

    def _compute_penalty_matrix(self, dm: np.ndarray, n: int,
                                routes: list[list[int]]) -> np.ndarray:
        route_lengths = [self._route_length(r, dm) for r in routes]

        if not route_lengths:
            return np.zeros((n, n), dtype=np.int64)

        mean_len = sum(route_lengths) / len(route_lengths)
        penalty = np.zeros((n, n), dtype=np.float64)

        for route, length in zip(routes, route_lengths):
            if length <= mean_len:
                continue
            scale = self.fairness_weight * (length - mean_len) / mean_len
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
        d += dm[prev, 0]
        return d

    @classmethod
    def _routes_distance(cls, routes: list[list[int]], dm: np.ndarray) -> float:
        return sum(cls._route_length(r, dm) for r in routes)
