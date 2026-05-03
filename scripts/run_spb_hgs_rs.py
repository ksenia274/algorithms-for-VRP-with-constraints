from __future__ import annotations

import argparse
import json
import os
import sys

from spb_utils import generate_instance, problem_to_instance_input

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_COORDS = (60.00771529992149, 30.370180423873254)
DISTRICT_NAME = "Kalininsky District, Saint Petersburg, Russia"

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--load-json", default=None, metavar="PATH",
        help="Skip generation, load an existing problem JSON (must contain 'coordinates')",
    )
    parser.add_argument(
        "--save-json", default=None, metavar="PATH",
        help="Save the generated problem JSON to this path (for reuse with --load-json)",
    )
    parser.add_argument("--points", type=int, default=50,
                        help="Number of client points to generate (default: 50)")
    parser.add_argument("--time", type=int, default=30,
                        help="HGS solver time limit in seconds (default: 30)")
    parser.add_argument("--vehicles", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=None,
                        help="Max stops per vehicle (overrides max_load from JSON; default: 35)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-map", default="map_results.html",
                        help="Output HTML map file (default: map_results.html)")
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()

    if args.load_json:
        print(f"Loading problem from {args.load_json} ...")
        with open(args.load_json, encoding="utf-8") as fh:
            problem = json.load(fh)
        capacity = args.capacity if args.capacity is not None else int(problem.get("max_load", 35))
    else:
        capacity = args.capacity if args.capacity is not None else 35
        problem = generate_instance(args.points, capacity, args.seed)
        if args.save_json:
            with open(args.save_json, "w", encoding="utf-8") as fh:
                json.dump(problem, fh)
            print(f"Problem saved → {args.save_json}")

    inp = problem_to_instance_input(problem, capacity)

    from algorithms.rectangle_splitting import RSSolver
    from algorithms.hgs_solver_simple import HGSSolver
    from algorithms.solver_result import SolverResult
    from run_hgs_rs import DurationAdapter

    print(f"\nRunning HGS with RS  (time={args.time}s  vehicles={args.vehicles}"
          f"  capacity={capacity}  num-threads={args.num_threads}) ...")

    hgs_simple = HGSSolver(
        time_limit=max(1, int(args.time / 10)),
        seed=args.seed,
        vehicle_capacity=capacity,
        num_vehicles=args.vehicles,
    )

    rs_solver = RSSolver[SolverResult](DurationAdapter(hgs_simple), time_limit=args.time, max_workers=args.num_threads)
    max_total_distance = 10000
    pareto_frontier = rs_solver.solve(inp, max_obj1=max_total_distance, min_obj2=0)
    sol = min(pareto_frontier, key=lambda s: s.fairness.fairness_score)

    print(f"\nFeasible:        {sol.feasible}")
    print(f"Total distance:  {sol.total_distance}")
    print(f"Num routes:      {sol.num_routes}")
    print(f"Rebalance moves: {sol.metadata.get('rebalance_moves', 0)}")
    print(f"Cost delta:      {sol.metadata.get('cost_delta_pct', 0.0):+.2f}%")
    if sol.fairness:
        r = sol.fairness
        print(f"Gini (after):    {r.dist_gini:.4f}")
        print(f"CV   (after):    {r.dist_cv:.4f}")

    if not sol.feasible:
        print("\nWARNING: infeasible — try --vehicles or --time.")

    n_clients = len(inp.df) - 1
    in_routes = set(c for r in sol.routes for c in r)
    missing = [i for i in range(1, n_clients + 1) if i not in in_routes]
    if missing:
        print(f"\nWARNING: {len(missing)} clients NOT in any route: {missing}")
    else:
        print(f"\nAll {n_clients} clients are in routes.")

    if inp.coordinates is None:
        print("\nNo coordinates in instance — skipping map.")
        print("Tip: use --save-json when generating to store coordinates,")
        print("     then --load-json to reload with visualization support.")
        return

    from visualization.map_routes import plot_routes_on_map

    print("Drawing map...")
    plot_routes_on_map(
        routes=sol.routes,
        coordinates=inp.coordinates,
        output_path=args.output_map,
        center=BASE_COORDS,
    )


if __name__ == "__main__":
    main()
