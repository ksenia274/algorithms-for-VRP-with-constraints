from algorithms.rectangle_splitting import GenericSolution
from algorithms.hgs_solver_simple import HGSSolver

class DurationAdapter:
        def __init__(self, base_solver: HGSSolver):
            self.base_solver = base_solver

        def optimize(self, instance_path: str, max_obj2: float) -> GenericSolution:
            res = self.base_solver.solve(instance_path, max_distance=max_obj2)
            return GenericSolution(
                obj1=res.total_distance,
                obj2=res.metadata["max_distance"] if res.feasible else max_obj2,
                is_feasible=res.feasible,
                payload=res
            )

def run_hgs_rs(args):
    from algorithms.rectangle_splitting import RSSolver
    from algorithms.hgs_solver_simple import HGSSolver
    from algorithms.solver_result import SolverResult

    hgs_simple = HGSSolver(
        time_limit=max(1, int(args.time / 10)),
        seed=args.seed,
        vehicle_capacity=args.capacity,
        num_vehicles=args.vehicles,
    )

    rs_solver = RSSolver[SolverResult](DurationAdapter(hgs_simple), time_limit=args.time, max_workers=4)
    max_total_distance = 10000
    pareto_frontier = rs_solver.solve(args.instance, max_obj1=max_total_distance, min_obj2=0)
    sol = min(pareto_frontier, key=lambda s: s.fairness.fairness_score)

    print(f"Feasible:       {sol.feasible}")
    print(f"Total distance: {sol.total_distance}")
    print(f"Max Duration:   {sol.metadata['max_duration']}")
    print(f"Num routes:     {sol.num_routes}")
    print()
    for i, route in enumerate(sol.routes):
        print(f"  Route {i + 1}: {route}")
    print(sol.fairness.summary())
