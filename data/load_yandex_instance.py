from __future__ import annotations

import json
import os

import pandas as pd

from data.vrp_instance import VRPInstanceInput


def load_yandex_instance(path: str) -> VRPInstanceInput:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    n_points: int = raw["points_count"]
    dist_matrix: list[list[int]] = raw["distance_matrix"]
    max_time: int = int(raw.get("max_time", 36_000))
    service_times: list[int] = raw.get("point_service_times", [0] * n_points)
    max_load: int = int(raw.get("max_load", 35))

    raw_time = raw.get("time_matrix")
    if raw_time and isinstance(raw_time[0][0], list):
        n_periods = len(raw_time)
        time_matrix = [
            [int(sum(raw_time[p][i][j] for p in range(n_periods)) / n_periods)
             for j in range(n_points)]
            for i in range(n_points)
        ]
    elif raw_time:
        time_matrix = raw_time
    else:
        time_matrix = None

    rows = []
    for i in range(n_points):
        rows.append({
            "CUST NO.": i,
            "XCOORD.": 0,
            "YCOORD.": 0,
            "DEMAND": 0 if i == 0 else 1,
            "READY TIME": 0,
            "DUE DATE": max_time,
            "SERVICE TIME": int(service_times[i]) if i < len(service_times) else 0,
        })

    df = pd.DataFrame(rows)

    raw_coords = raw.get("coordinates")
    coordinates = [tuple(c) for c in raw_coords] if raw_coords else None

    return VRPInstanceInput(
        df=df,
        dist_matrix=dist_matrix,
        time_matrix=time_matrix,
        recommended_capacity=max_load,
        coordinates=coordinates,
    )


def collect_yandex_instances(directory: str) -> list[tuple[str, str]]:
    results = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".json"):
            name = os.path.splitext(fname)[0]
            results.append((name, os.path.join(directory, fname)))
    return results
