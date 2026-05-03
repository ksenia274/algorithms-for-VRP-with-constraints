from __future__ import annotations

import argparse
import json
from spb_utils import generate_instance, problem_to_instance_input

BASE_COORDS = (60.00771529992149, 30.370180423873254)
DISTRICT_NAME = "Kalininsky District, Saint Petersburg, Russia"

def main():
    parser = argparse.ArgumentParser(
        description="ALNS solver for SPb road-network VRP with optional fairness + map output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--load-json", default=None, metavar="PATH",
        help="Skip generation, load an existing problem JSON",
    )
    parser.add_argument(
        "--save-json", default=None, metavar="PATH",
        help="Save the generated problem JSON to this path",
    )
    parser.add_argument("--points", type=int, default=50,
                        help="Number of client points to generate (default: 50)")
    parser.add_argument("--time", type=int, default=30,
                        help="ALNS time limit in seconds (default: 30)")
    parser.add_argument("--vehicles", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=None,
                        help="Max stops per vehicle (overrides max_load from JSON; default: 35)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-map", default="map_results.html",
                        help="Output HTML map file (default: map_results.html)")
    parser.add_argument("--fairness", action="store_true",
                        help="Enable fairness objective in ALNS")
    parser.add_argument("--fairness-weight", type=float, default=100.0,
                        help="Weight of fairness penalty in objective (default: 100)")
    parser.add_argument("--alns-iterations", type=int, default=25000,
                        help="Max ALNS iterations (default: 25000)")
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

    from algorithms.alns_solver import ALNSSolver

    print(f"\nRunning ALNS  (time={args.time}s  vehicles={args.vehicles}"
          f"  capacity={capacity}  fairness={'on' if args.fairness else 'off'}) ...")

    solver = ALNSSolver(
        time_limit=args.time,
        seed=args.seed,
        vehicle_capacity=capacity,
        num_vehicles=args.vehicles,
        enable_fairness=args.fairness,
        fairness_weight=args.fairness_weight,
        max_iterations=args.alns_iterations,
    )
    sol = solver.solve(inp)

    print(f"\nFeasible:       {sol.feasible}")
    print(f"Total distance: {sol.total_distance}")
    print(f"Num routes:     {sol.num_routes}")
    if sol.fairness:
        r = sol.fairness
        print(f"Gini:           {r.dist_gini:.4f}")
        print(f"CV:             {r.dist_cv:.4f}")

    if not sol.feasible:
        print("\nWARNING: infeasible — try more --vehicles or --time.")

    n_clients = len(inp.df) - 1
    in_routes = set(c for r in sol.routes for c in r)
    missing = [i for i in range(1, n_clients + 1) if i not in in_routes]
    if missing:
        print(f"\nWARNING: {len(missing)} clients NOT in any route: {missing}")
    else:
        print(f"\nAll {n_clients} clients are in routes.")

    if inp.coordinates is None:
        print("\nNo coordinates in instance — skipping map.")
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
