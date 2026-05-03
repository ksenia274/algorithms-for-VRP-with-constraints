from __future__ import annotations

import math
from typing import Literal

import pyvrp
import pyvrp.stop
from pyvrp._pyvrp import ActivityType
from pyvrp.adaptive_objective import (
    AdaptiveAdjustment,
    AdaptiveObjective,
    LinearDecay,
    ObjectiveWeights,
)
from pyvrp.IteratedLocalSearch import IteratedLocalSearchParams
from pyvrp.solve import SolveParams

from algorithms.solver_result import SolverResult
from data.vrp_instance import VRPInstanceInput, load_instance_input


class HGSSolverAdaptive:

    def __init__(
        self,
        time_limit: int = 60,
        seed: int = 0,
        vehicle_capacity: int = 100,
        num_vehicles: int = 25,
        initial_route_balance: float = 500.0,
        strategy: Literal["linear", "adaptive"] = "linear",
        decay: float = 0.99999,
        target_feasibility: float = 0.5,
        display: bool = True,
    ):
        self.time_limit = time_limit
        self.seed = seed
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.initial_route_balance = initial_route_balance
        self.strategy = strategy
        self.decay = decay
        self.target_feasibility = target_feasibility
        self.display = display

    def solve(
        self, instance: str | VRPInstanceInput
    ) -> tuple[tuple[SolverResult, SolverResult], AdaptiveObjective]:
        model, data = self._build_model(instance)

        if self.strategy == "adaptive":
            strat = AdaptiveAdjustment(target_feasibility=self.target_feasibility)
        else:
            strat = LinearDecay(decay=self.decay)

        objective = AdaptiveObjective(
            initial_weights=ObjectiveWeights(route_balance=self.initial_route_balance),
            strategy=strat,
        )

        result = model.solve(
            stop=pyvrp.stop.MaxRuntime(self.time_limit),
            seed=self.seed,
            display=self.display,
            params=SolveParams(
                ils=IteratedLocalSearchParams(callbacks=objective.as_callback())
            ),
        )
        best = result.best

        if not best.is_feasible():
            inf = SolverResult.infeasible(rebalance_moves=0, cost_delta_pct=0.0)
            return (inf, inf), objective

        raw_routes = [
            [act.idx + 1 for act in route if act.type == ActivityType.CLIENT]
            for route in best.routes()
        ]
        solver_result = SolverResult.from_routes_pyvrp_adapter(raw_routes, data)

        return (solver_result, solver_result), objective

    def _build_model(self, instance: str | VRPInstanceInput):
        inp = load_instance_input(instance)
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
        for i, frm in enumerate(locs):
            for j, to in enumerate(locs):
                dist = (
                    int(inp.dist_matrix[i][j])
                    if inp.dist_matrix is not None
                    else int(math.hypot(frm.x - to.x, frm.y - to.y))
                )
                duration = int(inp.time_matrix[i][j]) if inp.time_matrix is not None else dist
                m.add_edge(frm, to, distance=dist, duration=duration)

        return m, m.data()
