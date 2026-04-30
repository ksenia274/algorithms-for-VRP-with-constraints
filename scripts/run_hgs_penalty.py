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

    print()
    print(before.summary("Before penalty restarts"))
    print()
    print(after.summary("After penalty restarts"))
