import math

import pandas as pd
import pyvrp
import pyvrp.stop

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
        if self._cache_instance_name != instance_name:
            df = load_instance(instance_name)
            df.columns = df.columns.str.strip()
            self._cache_df = df
            self._cache_instance_name = instance_name
        return self._cache_df
    
    def set_vehicle_capacity(self, vehicle_capacity: int):
        self.vehicle_capacity = vehicle_capacity

    def solve(self, instance_name):
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
            total_distance = best.distance()
            routes = [route.visits() for route in best.routes()]
            actual_max_duration = max([float(route.duration()) for route in best.routes()]) if routes else 0.0
        else:
            total_distance = float("inf")
            routes = []
            actual_max_duration = float("inf")

        return {
            "routes": routes,
            "total_distance": total_distance,
            "num_routes": len(routes),
            "feasible": best.is_feasible(),
            "max_duration": actual_max_duration,
        }