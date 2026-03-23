"""
Режимы:
    python main.py                           # простой HGS на одном инстансе
    python main.py --fairness                # HGS + fairness rebalancing
    python main.py visualise                 # построить графики из results CSV
    python main.py visualise --csv results/fairness_benchmark.csv
"""

import argparse
import sys
from visualization.fairness_charts import plot_all


def run_simple(args):
    from algorithms.hgs_solver_simple import HGSSolver

    hgs = HGSSolver(
        time_limit=args.time,
        seed=args.seed,
        vehicle_capacity=args.capacity,
        num_vehicles=args.vehicles,
    )
    solution = hgs.solve(args.instance)

    print(f"Feasible:       {solution['feasible']}")
    print(f"Total distance: {solution['total_distance']}")
    print(f"Num routes:     {solution['num_routes']}")
    print()
    for i, route in enumerate(solution["routes"]):
        print(f"  Route {i + 1}: {route}")


def run_fairness(args):
    from algorithms.hgs_solver import HGSSolver

    hgs = HGSSolver(
        time_limit=args.time,
        seed=args.seed,
        vehicle_capacity=args.capacity,
        num_vehicles=args.vehicles,
        enable_fairness=True,
        max_cost_increase_pct=args.max_cost_increase,
        rebalance_iterations=args.rebalance_iters,
    )
    solution = hgs.solve(args.instance)

    print(f"Feasible:        {solution['feasible']}")
    print(f"Total distance:  {solution['total_distance']}")
    print(f"Num routes:      {solution['num_routes']}")
    print(f"Rebalance moves: {solution['rebalance_moves']}")
    print(f"Cost delta:      {solution['cost_delta_pct']:+.2f}%")
    print()

    for i, route in enumerate(solution["routes"]):
        print(f"  Route {i + 1}: {route}")

    print()
    print("=== BEFORE rebalancing ===")
    print(solution["fairness_before"].summary())
    print()
    print("=== AFTER rebalancing ===")
    print(solution["fairness"].summary())


def run_visualise(args):
    from visualization.fairness_charts import plot_all
    plot_all(csv_path=args.csv, output_dir=args.output)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command")

    bench_p = subparsers.add_parser("benchmark", help="Run full Solomon benchmark")
    bench_p.add_argument("--data-root", default="data/solomon")
    bench_p.add_argument("--time", type=int, default=30, help="HGS time limit per instance")
    bench_p.add_argument("--max-cost-increase", type=float, default=8.0)
    bench_p.add_argument("--rebalance-iters", type=int, default=5000)
    bench_p.add_argument("--seed", type=int, default=42)
    bench_p.add_argument("--output", default="results")

    vis_p = subparsers.add_parser("visualise", help="Build charts from benchmark CSV")
    vis_p.add_argument(
        "--csv",
        default="results/fairness_benchmark.csv",
        help="Path to benchmark results CSV",
    )
    vis_p.add_argument("--output", default="visualization/output")

    parser.add_argument("--instance", default="data/solomon/R1/R101.csv")
    parser.add_argument("--time", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--vehicles", type=int, default=25)
    parser.add_argument("--fairness", action="store_true")
    parser.add_argument("--max-cost-increase", type=float, default=5.0)
    parser.add_argument("--rebalance-iters", type=int, default=3000)

    args = parser.parse_args()

    if args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "visualise":
        run_visualise(args)
    elif args.fairness:
        run_fairness(args)
    else:
        run_simple(args)


if __name__ == "__main__":
    main()