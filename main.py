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

from data.load_solomon import get_solomon_path, load_instance


def run_simple(args):
    from algorithms.hgs_solver_simple import HGSSolver

    sol = HGSSolver(
        time_limit=args.time,
        seed=args.seed,
        vehicle_capacity=args.capacity,
        num_vehicles=args.vehicles,
    ).solve(args.instance)

    print(f"Feasible:       {sol['feasible']}")
    print(f"Total distance: {sol['total_distance']}")
    print(f"Num routes:     {sol['num_routes']}")
    print()
    for i, route in enumerate(sol["routes"]):
        print(f"  Route {i + 1}: {route}")


def run_fairness(args):
    from algorithms.hgs_solver import HGSSolver

    sol = HGSSolver(
        time_limit=args.time,
        seed=args.seed,
        vehicle_capacity=args.capacity,
        num_vehicles=args.vehicles,
        enable_fairness=True,
        max_cost_increase_pct=args.max_cost_increase,
        rebalance_iterations=args.rebalance_iters,
    ).solve(args.instance)

    print(f"Feasible:        {sol['feasible']}")
    print(f"Total distance:  {sol['total_distance']}")
    print(f"Num routes:      {sol['num_routes']}")
    print(f"Rebalance moves: {sol['rebalance_moves']}")
    print(f"Cost delta:      {sol['cost_delta_pct']:+.2f}%")
    print()

    for i, route in enumerate(sol["routes"]):
        print(f"  Route {i + 1}: {route}")

    print("\n=== BEFORE rebalancing ===")
    print(sol["fairness_before"].summary())
    print("\n=== AFTER rebalancing ===")
    print(sol["fairness"].summary())


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
            sol = solver.solve(name)
        except Exception as exc:
            print(f"ERROR: {exc}")
            rows.append({"instance": name, "category": category, "error": str(exc)})
            continue
        elapsed = time.time() - t0

        before = _extract_metrics(sol.get("fairness_before"))
        after = _extract_metrics(sol.get("fairness"))

        row = {
            "instance": name,
            "category": category,
            "feasible": sol["feasible"],
            "total_distance": sol["total_distance"],
            "num_routes": sol["num_routes"],
            "rebalance_moves": sol.get("rebalance_moves", 0),
            "cost_delta_pct": sol.get("cost_delta_pct", 0.0),
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

        status = "OK" if sol["feasible"] else "INFEASIBLE"
        print(f"{status} | {elapsed:.1f}s | dist={sol['total_distance']}"
              f" | moves={sol.get('rebalance_moves', 0)}{gini_str}")

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