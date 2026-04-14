def run_hgs_penalty(args):
    from algorithms.hgs_solver_penalty import HGSSolverPenalty

    before, after = HGSSolverPenalty(
        time_limit=args.time,
        seed=args.seed,
        vehicle_capacity=args.capacity,
        num_vehicles=args.vehicles,
        fairness_weight=args.fairness_weight,
        num_restarts=getattr(args, "fair_restarts", 5),
    ).solve(args.instance)

    print(f"Feasible:       {after.feasible}")
    print(f"Total distance: {after.total_distance:.1f}")
    print(f"Num routes:     {after.num_routes}")
    print(f"Cost delta:     {after.metadata.get('cost_delta_pct', 0):+.2f}%")
    print()

    for i, route in enumerate(after.routes):
        print(f"  Route {i + 1}: {route}")

    print("\n=== BEFORE fairness restarts ===")
    if before.feasible:
        print(before.fairness.summary())
    else:
        print("Not feasible")

    print("\n=== AFTER fairness restarts ===")
    if after.feasible:
        print(after.fairness.summary())
    else:
        print("Not feasible")
