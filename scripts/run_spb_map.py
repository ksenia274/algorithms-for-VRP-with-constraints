"""Generate and solve a Saint Petersburg VRP instance, then plot a Folium map.

Usage:
    # Load existing JSON, run hgs_rebalance, draw map
    python scripts/run_spb_map.py --load-json path/to/problem.json --algorithm hgs_rebalance

    # Generate synthetic instance, save JSON, draw map
    python scripts/run_spb_map.py --points 50 --save-json my_problem.json --algorithm hgs_adaptive

    # Load, run, save run-directory with result.json + metrics.csv
    python scripts/run_spb_map.py --load-json problem.json --save-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_COORDS = (60.00771529992149, 30.370180423873254)
DISTRICT_NAME = "Kalininsky District, Saint Petersburg, Russia"


def _generate_instance(n_points: int, capacity: int, seed: int) -> dict:
    import numpy as np
    import networkx as nx
    import osmnx as ox
    from shapely.geometry import Point

    print(f"Downloading road network for '{DISTRICT_NAME}'...")
    graph = ox.graph_from_place(DISTRICT_NAME, network_type="drive")
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    boundary = ox.geocode_to_gdf(DISTRICT_NAME).geometry.iloc[0]

    rng = np.random.default_rng(seed)
    points: list[tuple[float, float]] = []
    while len(points) < n_points:
        lat, lon = float(rng.normal(BASE_COORDS[0], 0.015)), float(rng.normal(BASE_COORDS[1], 0.015))
        if boundary.contains(Point(lon, lat)):
            points.append((lat, lon))

    coords_list = [BASE_COORDS] + points
    n_total = len(coords_list)
    osm_nodes = ox.nearest_nodes(graph, [p[1] for p in coords_list], [p[0] for p in coords_list])

    print(f"Computing {n_total}×{n_total} travel-time matrix...")
    dist_matrix: list[list[int]] = []
    for i, source in enumerate(osm_nodes):
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="travel_time")
        dist_matrix.append([int(lengths.get(t, 1_000_000)) for t in osm_nodes])
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_total} rows done")

    return {
        "points_count": n_total,
        "min_load": 10,
        "max_load": capacity,
        "max_time": 36_000,
        "distance_matrix": dist_matrix,
        "point_scores": [100] * (n_total - 1),
        "point_service_times": [300] * n_total,
        "coordinates": [[lat, lon] for lat, lon in coords_list],
    }


def cmd_spb_map(
    *,
    load_json=None,
    save_json: str | None = None,
    points: int = 50,
    algorithm: str = "hgs_rebalance",
    time_limit: int = 30,
    vehicles: int = 10,
    capacity: int | None = None,
    seed: int = 42,
    output_map: str = "map_results.html",
    save_run: bool = False,
) -> None:
    """Solve an SPb VRP instance and draw a Folium route map."""
    if load_json:
        print(f"Loading problem from {load_json} ...")
        with open(load_json, encoding="utf-8") as fh:
            problem = json.load(fh)
        cap = capacity if capacity is not None else int(problem.get("max_load", 35))
    else:
        cap = capacity if capacity is not None else 35
        problem = _generate_instance(points, cap, seed)
        if save_json:
            with open(save_json, "w", encoding="utf-8") as fh:
                json.dump(problem, fh)
            print(f"Problem saved → {save_json}")

    tmp_path = load_json
    if tmp_path is None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
            json.dump(problem, tf)
            tmp_path = tf.name

    from data.load_yandex_instance import load_yandex_instance
    inp = load_yandex_instance(str(tmp_path))

    from algorithms.algorithm_params import ALGORITHM_PARAMS_REGISTRY
    from algorithms.factory import build_solver
    from algorithms.solver_result import SolverConfig

    params_cls = ALGORITHM_PARAMS_REGISTRY[algorithm]
    config = SolverConfig(
        schema_version="1.0",
        algorithm=algorithm,
        instance="spb",
        instance_kind="yandex",
        time_limit=time_limit,
        seed=seed,
        capacity=cap,
        num_vehicles=vehicles,
        algorithm_params=params_cls(),
    )
    solver = build_solver(config)

    print(f"\nRunning {algorithm}  (time={time_limit}s  vehicles={vehicles}  capacity={cap}) ...")
    result = solver.solve(inp)

    print(f"\nFeasible:        {result.feasible}")
    print(f"Total distance:  {result.total_distance:.0f}")
    print(f"Num routes:      {result.num_routes}")
    if result.diagnostics.rebalance_moves is not None:
        print(f"Rebalance moves: {result.diagnostics.rebalance_moves}")
    if result.diagnostics.cost_delta_pct is not None:
        print(f"Cost delta:      {result.diagnostics.cost_delta_pct:+.2f}%")
    if result.fairness:
        print(f"Gini:            {result.fairness.distance.gini:.4f}")
        print(f"Worst ratio:     {result.fairness.distance.worst_ratio:.4f}")

    if save_run:
        from runtime.run_dir import create_run_dir, save_run as _save_run
        run_dir = create_run_dir(config)
        _save_run(result, run_dir)
        print(f"\nRun saved → {run_dir}")

    coords = inp.coordinates
    coords = inp.coordinates
    if coords is None:
        print("\nNo coordinates in instance — skipping map.")
        return

    from visualization.map_routes import plot_routes_on_map
    print(f"Drawing map → {output_map} ...")
    plot_routes_on_map(
        routes=result.routes,
        coordinates=coords,
        output_path=str(output_map),
        center=BASE_COORDS,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--load-json", default=None, metavar="PATH")
    p.add_argument("--save-json", default=None, metavar="PATH")
    p.add_argument("--points", type=int, default=50)
    p.add_argument("--algorithm", default="hgs_rebalance",
                   choices=["hgs_simple", "hgs_rebalance", "hgs_penalty", "hgs_adaptive", "alns"])
    p.add_argument("--time", type=int, default=30)
    p.add_argument("--vehicles", type=int, default=10)
    p.add_argument("--capacity", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-map", default="map_results.html")
    p.add_argument("--save-run", action="store_true")
    args = p.parse_args()

    cmd_spb_map(
        load_json=args.load_json,
        save_json=args.save_json,
        points=args.points,
        algorithm=args.algorithm,
        time_limit=args.time,
        vehicles=args.vehicles,
        capacity=args.capacity,
        seed=args.seed,
        output_map=args.output_map,
        save_run=args.save_run,
    )


if __name__ == "__main__":
    main()
