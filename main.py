"""
Режимы:
    python main.py                           # простой HGS на одном инстансе
    python main.py --fairness                # HGS + fairness rebalancing
    python main.py benchmark                 # прогон на всех solomon инстансах и вывод в CSV
    python main.py visualise                 # построить графики из results CSV
    python main.py visualise --csv results/fairness_benchmark.csv
"""

import argparse
import os
import time
from enum import Enum

from data.load_solomon import get_solomon_path


def run_simple(args):
    from algorithms.hgs_solver_simple import HGSSolver

    sol = HGSSolver(
        time_limit=args.time,
        seed=args.seed,
        vehicle_capacity=args.capacity,
        num_vehicles=args.vehicles,
    ).solve(args.instance)

    print(f"Feasible:       {sol.feasible}")
    print(f"Total distance: {sol.total_distance}")
    print(f"Num routes:     {sol.num_routes}")
    print()
    for i, route in enumerate(sol.routes):
        print(f"  Route {i + 1}: {route}")
    print(sol.fairness.summary())

def run_simple_rectangle_splitting(args):
    from algorithms.rectangle_splitting import RSSolver, GenericSolution
    from algorithms.hgs_solver_simple import HGSSolver
    
    print("Running Rectangle Splitting with Simple HGS...")

    hgs_simple = HGSSolver(
        time_limit=max(1, int(args.time / 5)),
        seed=args.seed,
        vehicle_capacity=args.capacity,
        num_vehicles=args.vehicles,
    )

    class DurationAdapter:
        def __init__(self, base_solver: HGSSolver):
            self.base_solver = base_solver
            
        def optimize(self, instance_path: str, max_obj2: float) -> GenericSolution:
            self.base_solver.set_vehicle_capacity(int(max_obj2))
            res = self.base_solver.solve(instance_path)
            return GenericSolution(
                obj1=res.total_distance,
                obj2=res.metadata["max_duration"] if res.feasible else max_obj2,
                is_feasible=res.feasible,
                payload=res
            )

    rs_solver = RSSolver(DurationAdapter(hgs_simple), time_limit=args.time)
    
    sol = rs_solver.solve(args.instance, min_obj1=0, max_obj2=args.capacity)
    
    print(f"Feasible:       {sol.feasible}")
    print(f"Total distance: {sol.total_distance}")
    print(f"Max Duration:   {sol.metadata["max_duration"]}")
    print(f"Num routes:     {sol.num_routes}")
    print()
    for i, route in enumerate(sol.routes):
        print(f"  Route {i + 1}: {route}")
    print(sol.fairness.summary())


def run_fairness_rebalance(args):
    from algorithms.hgs_solver import HGSSolver

    sol_before_rebalance, sol = HGSSolver(
        time_limit=args.time,
        seed=args.seed,
        vehicle_capacity=args.capacity,
        num_vehicles=args.vehicles,
        enable_fairness=True,
        max_cost_increase_pct=args.max_cost_increase,
        rebalance_iterations=args.rebalance_iters,
    ).solve(args.instance)

    print(f"Feasible:        {sol.feasible}")
    print(f"Total distance:  {sol.total_distance}")
    print(f"Num routes:      {sol.num_routes}")
    print(f"Rebalance moves: {sol.metadata['rebalance_moves']}")
    print(f"Cost delta:      {sol.metadata['cost_delta_pct']:+.2f}%")
    print()

    for i, route in enumerate(sol.routes):
        print(f"  Route {i + 1}: {route}")

    print("\n=== BEFORE rebalancing ===")
    print(sol_before_rebalance.fairness.summary())
    print("\n=== AFTER rebalancing ===")
    print(sol.fairness.summary())


def _detect_category(name):
    for prefix in ("RC", "R", "C"):
        if name.upper().startswith(prefix):
            digit = name[len(prefix):len(prefix) + 1]
            if digit in ("1", "2"):
                return f"{prefix}{digit}"
    return "other"


def _collect_instances():
    base = get_solomon_path()
    found = []
    for root, _, files in os.walk(base):
        for f in sorted(files):
            if f.endswith(".csv"):
                name = os.path.splitext(f)[0]
                found.append((name, os.path.join(root, f)))
    found.sort(key=lambda x: x[0])
    return found


def _extract_metrics(report):
    if report is None:
        return {}
    return {k: getattr(report, k, None) for k in (
        "dist_gini", "dist_jain", "dist_cv", "dist_std",
        "dist_range", "load_gini", "fairness_score",
    )}


def run_benchmark(args):
    import pandas as pd
    from algorithms.hgs_solver import HGSSolver

    instance_paths = _collect_instances()
    if not instance_paths:
        print(f"No Solomon CSV files found")
        return

    print(f"Instances found: {len(instance_paths)}")
    print(f"time_limit={args.time}s  max_cost_increase={args.max_cost_increase}%"
          f"  rebalance_iters={args.rebalance_iters}  seed={args.seed}")
    print("=" * 72)

    rows = []

    for name, path in instance_paths:
        category = _detect_category(name)
        print(f"\n[{category}] {name} ... ", end="", flush=True)

        solver = HGSSolver(
            time_limit=args.time,
            seed=args.seed,
            vehicle_capacity=args.capacity,
            num_vehicles=args.vehicles,
            enable_fairness=True,
            max_cost_increase_pct=args.max_cost_increase,
            rebalance_iterations=args.rebalance_iters,
        )

        t0 = time.time()
        try:
            sol_before_rebalance, sol = solver.solve(name)
        except Exception as exc:
            print(f"ERROR: {exc}")
            rows.append({"instance": name, "category": category, "error": str(exc)})
            continue
        elapsed = time.time() - t0

        before = _extract_metrics(sol_before_rebalance.fairness)
        after = _extract_metrics(sol.fairness)

        row = {
            "instance": name,
            "category": category,
            "feasible": sol.feasible,
            "total_distance": sol.total_distance,
            "num_routes": sol.num_routes,
            "rebalance_moves": sol.metadata.get("rebalance_moves", 0),
            "cost_delta_pct": sol.metadata.get("cost_delta_pct", 0.0),
            "solve_time_s": round(elapsed, 2),
            **{f"{k}_before": v for k, v in before.items()},
            **{f"{k}_after": v for k, v in after.items()},
        }
        rows.append(row)

        gini_b = before.get("dist_gini")
        gini_a = after.get("dist_gini")
        gini_str = ""
        if gini_b is not None and gini_a is not None:
            gini_str = f" | Gini {gini_b:.3f} -> {gini_a:.3f} ({gini_a - gini_b:+.3f})"

        status = "OK" if sol.feasible else "INFEASIBLE"
        print(f"{status} | {elapsed:.1f}s | dist={sol.total_distance}"
              f" | moves={sol.metadata.get('rebalance_moves', 0)}{gini_str}")

    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, "fairness_benchmark.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    feasible_n = sum(1 for r in rows if r.get("feasible"))
    print(f"\n{'=' * 72}")
    print(f"Saved to {csv_path}  ({len(rows)} instances, {feasible_n} feasible)")

    if "dist_gini_before" in df.columns:
        print("\nMean Gini by category:")
        print(df.groupby("category").agg(
            gini_before=("dist_gini_before", "mean"),
            gini_after=("dist_gini_after", "mean"),
            avg_moves=("rebalance_moves", "mean"),
        ).round(4).to_string())


def run_visualise(args):
    from visualization.fairness_charts import plot_all
    plot_all(csv_path=args.csv, output_dir=args.output)

class Algorithm(Enum):
    HGS_SIMPLE = 'hgs_simple'
    HGS_REBALANCE = 'hgs_rebalance'
    HGS_RECTANGLE_SPLITTING = 'hgs_rs'
    
    def __str__(self):
        return self.value

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    bench = subparsers.add_parser("benchmark", help="Run full Solomon benchmark")
    bench.add_argument("--time", type=int, default=30)
    bench.add_argument("--capacity", type=int, default=200)
    bench.add_argument("--vehicles", type=int, default=25)
    bench.add_argument("--max-cost-increase", type=float, default=8.0)
    bench.add_argument("--rebalance-iters", type=int, default=5000)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--output", default="results")

    vis = subparsers.add_parser("visualise", help="Build charts from benchmark CSV")
    vis.add_argument("--csv", default="results/fairness_benchmark.csv")
    vis.add_argument("--output", default="visualization/output")

    parser.add_argument("--instance", default="R101")
    parser.add_argument("--time", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--vehicles", type=int, default=25)
    parser.add_argument("--algorithm", type=Algorithm, choices=list(Algorithm))
    parser.add_argument("--max-cost-increase", type=float, default=5.0)
    parser.add_argument("--rebalance-iters", type=int, default=3000)

    args = parser.parse_args()

    if args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "visualise":
        run_visualise(args)
    elif args.algorithm == Algorithm.HGS_REBALANCE:
        run_fairness_rebalance(args)
    elif args.algorithm == Algorithm.HGS_RECTANGLE_SPLITTING:
        run_simple_rectangle_splitting(args)
    elif args.algorithm == Algorithm.HGS_SIMPLE:
        run_simple(args)
    else:
        raise Exception("Neither the algorithm nor the command was specified")


if __name__ == "__main__":
    main()