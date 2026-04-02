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

    if sol["fairness_before"]:
        print("\n=== BEFORE rebalancing ===")
        print(sol["fairness_before"].summary())
    if sol["fairness"]:
        print("\n=== AFTER rebalancing ===")
        print(sol["fairness"].summary())
