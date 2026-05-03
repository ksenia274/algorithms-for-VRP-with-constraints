from __future__ import annotations

import math

import numpy as np
import pyvrp
import pyvrp.stop
from pyvrp import Solution
from pyvrp._pyvrp import ActivityType, Route

from algorithms.solver_result import SolverResult
from data.vrp_instance import VRPInstanceInput, load_instance_input


class HGSSolverPenalty:
    """
    HGS с fairness прямо в solve-цикле через итеративную модификацию
    матрицы расстояний.

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
        self.time_limit = time_limit
        self.seed = seed
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.fairness_weight = fairness_weight
        self.num_restarts = num_restarts
        self.max_cost_increase_pct = max_cost_increase_pct
        self.display = display

    def solve(self, instance: str | VRPInstanceInput) -> tuple[SolverResult, SolverResult]:
        inp = load_instance_input(instance)
        time_per_restart = max(1, self.time_limit // self.num_restarts)

        base_data, orig_dm, dur_dm = self._build_data(inp)

        result = self._solve_data(base_data, time_per_restart, self.seed, display=self.display)

        if not result.best.is_feasible():
            inf = SolverResult.infeasible(rebalance_moves=0, cost_delta_pct=0.0)
            return inf, inf

        best_solution = result.best
        before = SolverResult.from_routes_pyvrp_adapter(
            self._extract_routes(best_solution),
            base_data,
        )

        best_gini = before.fairness.dist_gini if before.fairness else float("inf")
        cost_budget = before.total_distance * (1 + self.max_cost_increase_pct / 100.0)

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
            candidate_gini = candidate.fairness.dist_gini if candidate.fairness else float("inf")

            if candidate_gini < best_gini and candidate_dist <= cost_budget:
                best_solution = result.best
                best_gini = candidate_gini

        final_routes = self._extract_routes(best_solution)
        after = SolverResult.from_routes_pyvrp_adapter(
            final_routes,
            base_data,
            rebalance_moves=self.num_restarts - 1,
            cost_delta_pct=self._routes_distance(final_routes, orig_dm) / before.total_distance * 100.0 - 100.0
            if before.total_distance > 0 else 0.0,
        )
        return before, after

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_data(self, inp: VRPInstanceInput):
        """
        Builds ProblemData once. Returns (data, orig_dm, dur_dm) where
        orig_dm and dur_dm are numpy int64 arrays used for fast matrix swaps.
        """
        df = inp.df.copy()
        df.columns = df.columns.str.strip()

        m = pyvrp.Model()

        depot_row = df.iloc[0]
        depot_loc = m.add_location(x=int(depot_row["XCOORD."]), y=int(depot_row["YCOORD."]))
        depot = m.add_depot(
            depot_loc,
            tw_early=int(depot_row["READY TIME"]),
            tw_late=int(depot_row["DUE DATE"]),
            name="Depot",
        )
        m.add_vehicle_type(
            num_available=self.num_vehicles,
            capacity=[self.vehicle_capacity],
            start_depot=depot,
            end_depot=depot,
        )
        for _, row in df.iloc[1:].iterrows():
            loc = m.add_location(x=int(row["XCOORD."]), y=int(row["YCOORD."]))
            m.add_client(
                loc,
                delivery=[int(row["DEMAND"])],
                tw_early=int(row["READY TIME"]),
                tw_late=int(row["DUE DATE"]),
                service_duration=int(row["SERVICE TIME"]),
                name=f"Client {int(row['CUST NO.'])}",
            )

        locs = list(m.locations)
        n = len(locs)

        if inp.dist_matrix is not None:
            orig_dm = np.array(inp.dist_matrix, dtype=np.int64)
        else:
            coords = np.array([(loc.x, loc.y) for loc in locs], dtype=np.float64)
            dx = coords[:, 0:1] - coords[:, 0]
            dy = coords[:, 1:2] - coords[:, 1]
            orig_dm = np.hypot(dx, dy).astype(np.int64)

        dur_dm = np.array(inp.time_matrix, dtype=np.int64) if inp.time_matrix is not None else orig_dm

        # Add edges once using the original matrix
        for i, frm in enumerate(locs):
            for j, to in enumerate(locs):
                m.add_edge(frm, to, distance=int(orig_dm[i, j]), duration=int(dur_dm[i, j]))

        data = m.data()
        return data, orig_dm, dur_dm

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
        """
        Returns an additive int64 penalty matrix. Edges belonging to routes
        longer than the mean get a penalty proportional to their excess length.
        """
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
    def _extract_routes(solution) -> list[list[int]]:
        return [
            [act.idx + 1 for act in route if act.type == ActivityType.CLIENT]
            for route in solution.routes()
        ]

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
