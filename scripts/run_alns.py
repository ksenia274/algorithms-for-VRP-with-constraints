def run_alns(args):
    from algorithms.alns_solver import ALNSSolver

    sol = ALNSSolver(
        time_limit=args.time,
        seed=args.seed,
        vehicle_capacity=args.capacity,
        num_vehicles=args.vehicles,
        enable_fairness=args.fairness,
        fairness_weight=args.fairness_weight,
        max_iterations=args.alns_iterations,
    ).solve(args.instance)

    print(f"Feasible:       {sol['feasible']}")
    print(f"Total distance: {sol['total_distance']}")
    print(f"Num routes:     {sol['num_routes']}")
    print()
    for i, route in enumerate(sol["routes"]):
        print(f"  Route {i + 1}: {route}")

    if sol["fairness"]:
        print("\n=== Fairness ===")
        print(sol["fairness"].summary())