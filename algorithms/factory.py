from __future__ import annotations

from algorithms.solver_result import SolverConfig


def build_solver(config: SolverConfig):
    """Return an instantiated solver for config.algorithm.

    Raises ValueError for unknown algorithm names.
    """
    match config.algorithm:
        case "hgs_simple":
            from algorithms.hgs_solver_simple import HGSSolverSimple
            return HGSSolverSimple(config)
        case "hgs_rebalance":
            from algorithms.hgs_solver_rebalance import HGSSolverRebalance
            return HGSSolverRebalance(config)
        case "hgs_penalty":
            from algorithms.hgs_solver_penalty import HGSSolverPenalty
            return HGSSolverPenalty(config)
        case "hgs_adaptive":
            from algorithms.hgs_solver_adaptive import HGSSolverAdaptive
            return HGSSolverAdaptive(config)
        case "alns":
            from algorithms.alns_solver import ALNSSolver
            return ALNSSolver(config)
        case _:
            raise ValueError(f"Unknown algorithm: '{config.algorithm}'")
