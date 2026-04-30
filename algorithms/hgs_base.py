from __future__ import annotations

import numpy as np
import pyvrp
from pyvrp._pyvrp import ActivityType

from data.vrp_instance import VRPInstanceInput, load_instance_input


class HGSBase:
    """
    Базовый класс для всех HGS-солверов.
    Содержит общую логику построения модели PyVRP и извлечения маршрутов.
    """

    def __init__(
        self,
        time_limit: int = 60,
        seed: int = 0,
        vehicle_capacity: int = 100,
        num_vehicles: int = 25,
    ):
        self.time_limit = time_limit
        self.seed = seed
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles

    @staticmethod
    def _extract_routes(solution) -> list[list[int]]:
        """Извлекает маршруты как списки location-индексов (1-based, depot=0)."""
        return [
            [act.idx + 1 for act in route if act.type == ActivityType.CLIENT]
            for route in solution.routes()
        ]

    def _build_model(
        self,
        instance: str | VRPInstanceInput,
        *,
        use_prizes: bool = False,
        max_distance: int | None = None,
    ) -> tuple[pyvrp.Model, object, np.ndarray, np.ndarray]:
        """
        Строит PyVRP-модель из инстанса.

        Возвращает (model, data, orig_dm, dur_dm), где:
          - model   — pyvrp.Model (нужен для model.solve())
          - data    — ProblemData (нужен для data.replace(), Route(), Solution())
          - orig_dm — numpy int64 матрица расстояний (нужна penalty-солверу)
          - dur_dm  — numpy int64 матрица длительностей
        """
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

        vt_kwargs: dict = dict(
            num_available=self.num_vehicles,
            capacity=[self.vehicle_capacity],
            start_depot=depot,
            end_depot=depot,
        )
        if max_distance is not None:
            vt_kwargs["max_distance"] = max_distance
        m.add_vehicle_type(**vt_kwargs)

        scores = inp.point_scores if use_prizes else None
        for idx, (_, row) in enumerate(df.iloc[1:].iterrows()):
            loc = m.add_location(x=int(row["XCOORD."]), y=int(row["YCOORD."]))
            client_kwargs: dict = dict(
                delivery=[int(row["DEMAND"])],
                tw_early=int(row["READY TIME"]),
                tw_late=int(row["DUE DATE"]),
                service_duration=int(row["SERVICE TIME"]),
                name=f"Client {int(row['CUST NO.'])}",
            )
            if scores is not None:
                client_kwargs["required"] = False
                client_kwargs["prize"] = int(scores[idx]) * 100 if idx < len(scores) else 0
            m.add_client(loc, **client_kwargs)

        locs = list(m.locations)

        if inp.dist_matrix is not None:
            orig_dm = np.array(inp.dist_matrix, dtype=np.int64)
        else:
            coords = np.array([(loc.x, loc.y) for loc in locs], dtype=np.float64)
            dx = coords[:, 0:1] - coords[:, 0]
            dy = coords[:, 1:2] - coords[:, 1]
            orig_dm = np.hypot(dx, dy).astype(np.int64)

        dur_dm = (
            np.array(inp.time_matrix, dtype=np.int64)
            if inp.time_matrix is not None
            else orig_dm
        )

        for i, frm in enumerate(locs):
            for j, to in enumerate(locs):
                m.add_edge(frm, to, distance=int(orig_dm[i, j]), duration=int(dur_dm[i, j]))

        return m, m.data(), orig_dm, dur_dm
