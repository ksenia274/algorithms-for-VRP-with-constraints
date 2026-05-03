def run_hgs_adaptive(args):
    from algorithms.hgs_solver_adaptive import HGSSolverAdaptive

    (before, after), objective = HGSSolverAdaptive(
        time_limit=args.time,
        seed=args.seed,
        vehicle_capacity=args.capacity,
        num_vehicles=args.vehicles,
        initial_route_balance=getattr(args, "route_balance", 500.0),
        strategy=getattr(args, "strategy", "adaptive"),
        decay=getattr(args, "decay", 0.9999),
        target_feasibility=getattr(args, "target_feasibility", 0.5),
    ).solve(args.instance)

    print(f"Feasible:       {after.feasible}")
    print(f"Total distance: {after.total_distance:.1f}")
    print(f"Num routes:     {after.num_routes}")
    print()

    for i, route in enumerate(after.routes):
        print(f"  Route {i + 1}: {route}")

    print("\n=== Fairness metrics ===")
    if after.feasible and after.fairness:
        r = after.fairness
        print(f"  Gini  dist={r.dist_gini:.4f}  load={r.load_gini:.4f}")
        print(f"  CV    dist={r.dist_cv:.4f}  load={r.load_cv:.4f}")
        print(f"  Jain  dist={r.dist_jain:.4f}  load={r.load_jain:.4f}")
        print(f"  Fairness score: {r.fairness_score:.4f}")
    else:
        print("Not feasible")

    try:
        history = objective.get_history_dataframe()
        if history is not None and not history.empty:
            print(f"\n=== AdaptiveObjective history ({len(history)} iterations) ===")
            print(history.tail(5).to_string())
    except Exception:
        pass
