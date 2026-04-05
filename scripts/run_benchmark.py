import os
import time

from data.load_solomon import get_solomon_path
from data.load_yandex_instance import collect_yandex_instances, load_yandex_instance


def _collect_solomon_instances():
    base = get_solomon_path()
    found = []
    for root, _, files in os.walk(base):
        for f in sorted(files):
            if f.endswith(".csv"):
                name = os.path.splitext(f)[0]
                found.append((name, os.path.join(root, f)))
    found.sort(key=lambda x: x[0])
    return found


def _detect_category(name):
    for prefix in ("RC", "R", "C"):
        if name.upper().startswith(prefix):
            digit = name[len(prefix):len(prefix) + 1]
            if digit in ("1", "2"):
                return f"{prefix}{digit}"
    return "other"


def _extract_metrics(report):
    if report is None:
        return {}
    return {
        "gini": getattr(report, "dist_gini", None),
        "jain": getattr(report, "dist_jain", None),
        "cv": getattr(report, "dist_cv", None),
        "dist_std": getattr(report, "dist_std", None),
        "dist_range": getattr(report, "dist_range", None),
        "load_gini": getattr(report, "load_gini", None),
        "score": getattr(report, "fairness_score", None),
    }


def run_benchmark(args):
    import pandas as pd
    from algorithms.hgs_solver import HGSSolver
    from algorithms.alns_solver import ALNSSolver

    from main import Algorithm
    algorithm = getattr(args, "algorithm", Algorithm.HGS_REBALANCE)
    dataset = getattr(args, "dataset", "solomon")

    if dataset == "yandex":
        yandex_dir = getattr(args, "yandex_path", "vrp_problems")
        raw_instances = collect_yandex_instances(yandex_dir)
        if not raw_instances:
            print(f"No Yandex JSON files found in {yandex_dir}")
            return
        instances = []
        for name, path in raw_instances:
            inp = load_yandex_instance(path)
            instances.append((name, inp))
    else:
        raw_instances = _collect_solomon_instances()
        if not raw_instances:
            print("No Solomon CSV files found")
            return
        instances = [(name, path) for name, path in raw_instances]

    print(f"Dataset: {dataset}  instances found: {len(instances)}")
    print(f"time_limit={args.time}s  max_cost_increase={args.max_cost_increase}%"
          f"  rebalance_iters={args.rebalance_iters}  seed={args.seed}")
    print("=" * 72)

    rows = []

    for name, instance_or_path in instances:
        if dataset == "yandex":
            category = "yandex"
            capacity = (
                instance_or_path.recommended_capacity
                if instance_or_path.recommended_capacity is not None
                else args.capacity
            )
            instance_arg = instance_or_path
        else:
            category = _detect_category(name)
            capacity = args.capacity
            instance_arg = instance_or_path

        print(f"\n[{category}] {name} ... ", end="", flush=True)

        if algorithm == Algorithm.ALNS:
            solver = ALNSSolver(
                time_limit=args.time,
                seed=args.seed,
                vehicle_capacity=capacity,
                num_vehicles=args.vehicles,
                enable_fairness=True,
                fairness_weight=getattr(args, "fairness_weight", 100.0),
                max_iterations=getattr(args, "alns_iterations", 25000),
            )
        else:
            solver = HGSSolver(
                time_limit=args.time,
                seed=args.seed,
                vehicle_capacity=capacity,
                num_vehicles=args.vehicles,
                enable_fairness=(algorithm == Algorithm.HGS_REBALANCE),
                max_cost_increase_pct=args.max_cost_increase,
                rebalance_iterations=args.rebalance_iters,
            )

        t0 = time.time()
        try:
            result = solver.solve(instance_arg)
        except Exception as exc:
            print(f"ERROR: {exc}")
            rows.append({"instance": name, "category": category, "error": str(exc)})
            continue
        elapsed = time.time() - t0

        if isinstance(result, tuple):
            sol_before, sol = result
        else:
            sol_before, sol = None, result

        before = _extract_metrics(sol_before.fairness if sol_before is not None else None)
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

        gini_b = before.get("gini")
        gini_a = after.get("gini")
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

    if "gini_before" in df.columns:
        print("\nMean Gini by category:")
        print(df.groupby("category").agg(
            gini_before=("gini_before", "mean"),
            gini_after=("gini_after", "mean"),
            avg_moves=("rebalance_moves", "mean"),
        ).round(4).to_string())