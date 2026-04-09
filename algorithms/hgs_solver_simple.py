import math

import pandas as pd
import pyvrp
import pyvrp.stop

from algorithms.solver_result import SolverResult
from data.load_solomon import load_instance


class HGSSolver:
    def __init__(self, time_limit=60, seed=0, vehicle_capacity=100, num_vehicles=25):
        self.time_limit = time_limit
        self.seed = seed
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self._cache_instance_name = None
        self._cache_df = None

    def _get_df(self, instance_name: str) -> pd.DataFrame:
        if self._cache_df is None or self._cache_instance_name != instance_name:
            df = load_instance(instance_name)
            df.columns = df.columns.str.strip()
            self._cache_df = df
            self._cache_instance_name = instance_name
        return self._cache_df

    def solve(self, instance_name, max_distance: float = math.inf) -> SolverResult:
        df = self._get_df(instance_name)

        m = pyvrp.Model()

        depot_row = df.iloc[0]
        depot = m.add_depot(
            x=int(depot_row["XCOORD."]),
            y=int(depot_row["YCOORD."]),
            tw_early=int(depot_row["READY TIME"]),
            tw_late=int(depot_row["DUE DATE"]),
            name="Depot",
        )

        if max_distance and max_distance < math.inf:
            m.add_vehicle_type(
                num_available=self.num_vehicles,
                capacity=[self.vehicle_capacity],
                start_depot=depot,
                end_depot=depot,
                max_distance=int(max_distance),
            )
        else: 
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

        result = m.solve(
            stop=pyvrp.stop.MaxRuntime(self.time_limit),
            seed=self.seed,
        )

        best = result.best

        if best.is_feasible():
            routes = [route.visits() for route in best.routes()]
            actual_max_duration = max([float(route.duration()) for route in best.routes()]) if routes else 0.0
            actual_max_distance = max([float(route.distance()) for route in best.routes()]) if routes else 0.0
            return SolverResult.from_routes_pyvrp_adapter(routes, m.data(), max_duration=actual_max_duration, max_distance=actual_max_distance)
        else:
            return SolverResult.infeasible()
