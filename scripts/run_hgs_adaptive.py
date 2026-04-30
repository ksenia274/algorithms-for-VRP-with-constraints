def run_hgs_adaptive(args):
    import os
    from algorithms.hgs_solver_adaptive import HGSSolverAdaptive

    trace_path = None
    if getattr(args, "trace", False):
        inst_name = os.path.splitext(os.path.basename(str(args.instance)))[0]
        strat = getattr(args, "strategy", "linear")
        trace_dir = getattr(args, "trace_dir", "results")
        trace_path = f"{trace_dir}/trace_{strat}_instance{inst_name}.csv"

    (before, after), objective = HGSSolverAdaptive(
        time_limit=args.time,
        seed=args.seed,
        vehicle_capacity=args.capacity,
        num_vehicles=args.vehicles,
        initial_route_balance=getattr(args, "route_balance", 500.0),
        strategy=getattr(args, "strategy", "linear"),
        decay=getattr(args, "decay", 0.9999),
        target_cv=getattr(args, "target_cv", 0.2),
        hold_band=getattr(args, "hold_band", 0.05),
        boost_factor=getattr(args, "boost_multiplier", 1.05),
        fs_decay_factor=getattr(args, "decay_multiplier", 0.995),
        min_weight=getattr(args, "min_weight", 0.0),
        max_weight=getattr(args, "max_weight", 1e9),
    ).solve(args.instance, trace_path=trace_path)

    print(after.summary("HGS Adaptive"))

    try:
        history = objective.get_history_dataframe()
        if history is not None and not history.empty:
            print(f"\n=== AdaptiveObjective history ({len(history)} iterations) ===")
            print(history.tail(5).to_string())
    except Exception:
        pass
