def run_simple(args):
    from algorithms.hgs_solver_simple import HGSSolverSimple

    _, sol = HGSSolverSimple(
        time_limit=args.time,
        seed=args.seed,
        vehicle_capacity=args.capacity,
        num_vehicles=args.vehicles,
    ).solve(args.instance)

    print(sol.summary("HGS Simple"))


def run_fairness_rebalance(args):
    from algorithms.hgs_solver import HGSSolver

    before, after = HGSSolver(
        time_limit=args.time,
        seed=args.seed,
        vehicle_capacity=args.capacity,
        num_vehicles=args.vehicles,
        enable_fairness=True,
        max_cost_increase_pct=args.max_cost_increase,
        rebalance_iterations=args.rebalance_iters,
    ).solve(args.instance)

    print(before.summary("Before rebalancing"))
    print()
    print(after.summary("After rebalancing"))
