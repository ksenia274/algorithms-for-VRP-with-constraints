from __future__ import annotations

import math

import pandas as pd
import pyvrp
import pyvrp.stop

from algorithms.fairness_metrics import FairnessReport, compute_fairness
from algorithms.fairness_rebalancer import rebalance


class HGSSolver:
    def __init__(
        self,
        time_limit: int = 60,
        seed: int = 0,
        vehicle_capacity: int = 100,
        num_vehicles: int = 25,
        enable_fairness: bool = True,
        max_cost_increase_pct: float = 5.0,
        rebalance_iterations: int = 3000,
    ):
        self.time_limit = time_limit
        self.seed = seed
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.enable_fairness = enable_fairness
        self.max_cost_increase_pct = max_cost_increase_pct
        self.rebalance_iterations = rebalance_iterations


    def solve(self, instance_path: str) -> dict:
    
        model, data = self._build_model(instance_path)

        result = model.solve(
            stop=pyvrp.stop.MaxRuntime(self.time_limit),
            seed=self.seed,
        )

        best = result.best
        if not best.is_feasible():
            return {
                "routes": [],
                "total_distance": float("inf"),
                "num_routes": 0,
                "feasible": False,
                "fairness": None,
                "fairness_before": None,
                "rebalance_moves": 0,
                "cost_delta_pct": 0.0,
            }

        raw_routes = [route.visits() for route in best.routes()]
        hgs_distance = best.distance()

        fairness_before = self._compute_fairness_for_routes(data, raw_routes)

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

        fairness_after = self._compute_fairness_for_routes(data, final_routes)

        cost_delta = (
            (final_distance - hgs_distance) / hgs_distance * 100.0
            if hgs_distance > 0
            else 0.0
        )

        return {
            "routes": final_routes,
            "total_distance": final_distance,
            "num_routes": len(final_routes),
            "feasible": True,
            "fairness": fairness_after,
            "fairness_before": fairness_before,
            "rebalance_moves": rebalance_moves,
            "cost_delta_pct": cost_delta,
        }


    def _build_model(self, instance_path: str):
        df = pd.read_csv(instance_path)
        df.columns = df.columns.str.strip()

        m = pyvrp.Model()

        depot_row = df.iloc[0]
        depot = m.add_depot(
            x=int(depot_row["XCOORD."]),
            y=int(depot_row["YCOORD."]),
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
            m.add_client(
                x=int(row["XCOORD."]),
                y=int(row["YCOORD."]),
                delivery=[int(row["DEMAND"])],
                tw_early=int(row["READY TIME"]),
                tw_late=int(row["DUE DATE"]),
                service_duration=int(row["SERVICE TIME"]),
                name=f"Client {int(row['CUST NO.'])}",
            )

        for frm in m.locations:
            for to in m.locations:
                dist = int(math.hypot(frm.x - to.x, frm.y - to.y))
                m.add_edge(frm, to, distance=dist, duration=dist)

        return m, m.data()

    def _compute_fairness_for_routes(
        self,
        data,
        routes: list[list[int]],
    ) -> FairnessReport:
        dm = data.distance_matrix(0)
        depot = 0

        distances = []
        loads = []
        clients_count = []
        durations = []

        for route in routes:
            d = 0
            prev = depot
            for c in route:
                d += dm[prev, c]
                prev = c
            d += dm[prev, depot]
            distances.append(float(d))

            ld = 0
            for c in route:
                loc = data.location(c)
                if hasattr(loc, "delivery") and loc.delivery:
                    ld += loc.delivery[0]
            loads.append(float(ld))

            clients_count.append(len(route))

            durations.append(float(d))

        return compute_fairness(
            route_distances=distances,
            route_loads=loads,
            route_clients=clients_count,
            route_durations=durations,
        )