from __future__ import annotations

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_COORDS = (60.00771529992149, 30.370180423873254)
DISTRICT_NAME = "Kalininsky District, Saint Petersburg, Russia"


def generate_instance(n_points: int, capacity: int, seed: int) -> dict:
    import numpy as np
    import networkx as nx
    import osmnx as ox
    from shapely.geometry import Point

    print(f"Downloading road network for '{DISTRICT_NAME}' from OpenStreetMap...")
    graph = ox.graph_from_place(DISTRICT_NAME, network_type="drive")
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    print(f"Graph ready: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    print("Getting district boundary...")
    boundary = ox.geocode_to_gdf(DISTRICT_NAME).geometry.iloc[0]

    print(f"Generating {n_points} client points (normal distribution around depot)...")
    rng = np.random.default_rng(seed)
    std_dev = 0.015
    points: list[tuple[float, float]] = []
    while len(points) < n_points:
        lat = float(rng.normal(BASE_COORDS[0], std_dev))
        lon = float(rng.normal(BASE_COORDS[1], std_dev))
        if boundary.contains(Point(lon, lat)):
            points.append((lat, lon))

    coords_list = [BASE_COORDS] + points
    n_total = len(coords_list)

    print(f"Computing {n_total}×{n_total} travel-time matrix (Dijkstra on road graph)...")
    osm_nodes = ox.nearest_nodes(
        graph,
        [p[1] for p in coords_list],
        [p[0] for p in coords_list],
    )
    dist_matrix: list[list[int]] = []
    for i, source in enumerate(osm_nodes):
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="travel_time")
        row = [int(lengths.get(target, 1_000_000)) for target in osm_nodes]
        dist_matrix.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_total} rows done")

    return {
        "points_count": n_total,
        "min_load": 10,
        "max_load": capacity,
        "max_time": 36_000,
        "max_distance": 1_000_000,
        "distance_matrix": dist_matrix,
        "point_scores": [100] * (n_total - 1),
        "point_service_times": [300] * n_total,
        "coordinates": [[lat, lon] for lat, lon in coords_list],
    }


def problem_to_instance_input(problem: dict, capacity: int):
    from data.vrp_instance import VRPInstanceInput

    n_total: int = problem["points_count"]
    dist_matrix = problem["distance_matrix"]
    max_time: int = int(problem.get("max_time", 36_000))
    service_times: list[int] = problem.get("point_service_times", [0] * n_total)
    raw_coords = problem.get("coordinates")
    coordinates = [tuple(c) for c in raw_coords] if raw_coords else None

    rows = []
    for i in range(n_total):
        rows.append({
            "CUST NO.": i,
            "XCOORD.": 0,
            "YCOORD.": 0,
            "DEMAND": 0 if i == 0 else 1,
            "READY TIME": 0,
            "DUE DATE": max_time,
            "SERVICE TIME": int(service_times[i]) if i < len(service_times) else 0,
        })

    raw_scores = problem.get("point_scores")
    point_scores = [int(s) for s in raw_scores] if raw_scores else None

    return VRPInstanceInput(
        df=pd.DataFrame(rows),
        dist_matrix=dist_matrix,
        recommended_capacity=capacity,
        coordinates=coordinates,
        point_scores=point_scores,
    )
