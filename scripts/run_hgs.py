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
    if sol_before_rebalance.feasible:
        print(sol_before_rebalance.fairness.summary())
    else:
        print("Is Not Feasible")

    print("\n=== AFTER rebalancing ===")
    if sol.feasible:
        print(sol.fairness.summary())
    else:
        print("Is Not Feasible")
