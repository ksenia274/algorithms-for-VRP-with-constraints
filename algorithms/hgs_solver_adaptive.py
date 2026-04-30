from __future__ import annotations

import logging
from typing import Literal

import pyvrp.stop
from pyvrp.adaptive_objective import (
    AdaptiveObjective,
    LinearDecay,
    ObjectiveWeights,
)
from pyvrp.IteratedLocalSearch import (
    IteratedLocalSearch,
    IteratedLocalSearchCallbacks,
    IteratedLocalSearchParams,
)
from pyvrp.PenaltyManager import PenaltyManager
from pyvrp._pyvrp import RandomNumberGenerator, Solution
from pyvrp.search import LocalSearch, PerturbationManager, compute_neighbours
from pyvrp.solve import SolveParams

from algorithms.fairness_signal_strategy import FairnessSignalAdjustment
from algorithms.hgs_base import HGSBase
from algorithms.solver_result import SolverResult
from data.vrp_instance import VRPInstanceInput

logger = logging.getLogger(__name__)


def _save_trace(objective: AdaptiveObjective, path: str) -> None:
    import os
    import pandas as pd

    df = objective.get_history_dataframe()
    if df is None or df.empty:
        return

    df = df.copy()
    w = df["weight_route_balance"]
    nxt = w.shift(-1)
    eps = 1e-9
    event = pd.Series("hold", index=df.index, dtype=str)
    event[nxt > w + eps] = "boost"
    event[nxt < w - eps] = "decay"
    event.iloc[-1] = "hold"  # last row has no next weight to compare
    df["event"] = event

    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Trace saved → %s (%d rows)", path, len(df))


def _solve_with_weighted_objective(
    data,
    stop,
    seed: int,
    collect_stats: bool,
    display: bool,
    params: SolveParams,
    objective: AdaptiveObjective,
):
    """
    Mirrors pyvrp.solve.solve() but patches PenaltyManager.cost_evaluator()
    to inject the adaptive weights on every call.

    Without this patch, the ILS creates a fresh CostEvaluator (with zero
    custom weights) at the start of every iteration, discarding any weights
    applied by the on_iteration callback.  The patch ensures that every CE
    returned by the PM already carries the current objective.weights, so the
    search actually optimises for route balance.
    """
    rng = RandomNumberGenerator(seed=seed)
    neighbours = compute_neighbours(data, params.neighbourhood)
    perturbation = PerturbationManager(params.perturbation)
    ls = LocalSearch(data, rng, neighbours, perturbation)

    for op in params.operators:
        if op.supports(data):
            ls.add_operator(op(data))

    penalties = params.penalty.midpoint_penalties(data)
    pm = PenaltyManager(penalties, params.penalty)

    # THE FIX: wrap cost_evaluator() so every new CE gets current weights.
    # The ILS calls pm.cost_evaluator() at the start of every iteration,
    # creating a brand-new CostEvaluator with zero custom weights.  Any
    # weights applied by the on_iteration callback to the previous CE are
    # silently discarded.  This patch propagates the current objective weights
    # into each freshly created CE before the search step uses it.
    _orig_ce = pm.cost_evaluator

    def _weighted_ce():
        ce = _orig_ce()
        objective.weights.apply_to(ce)
        return ce

    pm.cost_evaluator = _weighted_ce

    random = Solution.make_random(data, rng)
    init = ls(random, pm.max_cost_evaluator(), exhaustive=True)

    algo = IteratedLocalSearch(data, pm, ls, init, params.ils)
    return algo.run(stop, collect_stats, display, params.display_interval)


class HGSSolverAdaptive(HGSBase):

    def __init__(
        self,
        time_limit: int = 60,
        seed: int = 0,
        vehicle_capacity: int = 100,
        num_vehicles: int = 25,
        initial_route_balance: float = 500.0,
        strategy: Literal["linear", "fairness_signal"] = "linear",
        # linear strategy
        decay: float = 0.99999,
        min_weight: float = 0.0,
        max_weight: float = 1e9,
        # fairness_signal strategy
        target_cv: float = 0.2,
        hold_band: float = 0.05,
        boost_factor: float = 1.05,
        fs_decay_factor: float = 0.995,
        fs_ma_window: int = 20,
        # objective update cadence (iterations between weight updates)
        update_every: int = 1,
        display: bool = True,
    ):
        super().__init__(time_limit, seed, vehicle_capacity, num_vehicles)
        self.initial_route_balance = initial_route_balance
        self.strategy = strategy
        self.decay = decay
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.target_cv = target_cv
        self.hold_band = hold_band
        self.boost_factor = boost_factor
        self.fs_decay_factor = fs_decay_factor
        self.fs_ma_window = fs_ma_window
        self.update_every = update_every
        self.display = display

    def solve(
        self, instance: str | VRPInstanceInput, trace_path: str | None = None
    ) -> tuple[tuple[SolverResult, SolverResult], AdaptiveObjective]:
        _, data, _, _ = self._build_model(instance)

        if self.strategy == "fairness_signal":
            strat = FairnessSignalAdjustment(
                target_cv=self.target_cv,
                hold_band=self.hold_band,
                boost_factor=self.boost_factor,
                decay_factor=self.fs_decay_factor,
                ma_window=self.fs_ma_window,
                min_weight=self.min_weight if self.min_weight > 0 else 1.0,
                max_weight=self.max_weight,
            )
        else:
            strat = LinearDecay(decay=self.decay)

        objective = AdaptiveObjective(
            initial_weights=ObjectiveWeights(route_balance=self.initial_route_balance),
            strategy=strat,
            update_every=self.update_every,
        )

        adaptive_cb = objective.as_callback()
        _log_every = 100

        class _LoggingCallback(IteratedLocalSearchCallbacks):
            def on_iteration(self, current, candidate, best, cost_evaluator):
                adaptive_cb.on_iteration(current, candidate, best, cost_evaluator)
                n = objective.iteration
                if n == 1:
                    logger.debug("Adaptive callback: first iteration reached")
                if n % _log_every == 0:
                    hist = objective.get_history()
                    feas = hist[-1].feasibility_rate if hist else float("nan")
                    logger.debug(
                        "iter=%d  route_balance_weight=%.4f  feasibility_rate=%.3f",
                        n, objective.weights.route_balance, feas,
                    )

            def on_restart(self, best):
                adaptive_cb.on_restart(best)
                logger.debug("Adaptive callback: restart at iter=%d", objective.iteration)

        result = _solve_with_weighted_objective(
            data=data,
            stop=pyvrp.stop.MaxRuntime(self.time_limit),
            seed=self.seed,
            collect_stats=True,
            display=self.display,
            params=SolveParams(
                ils=IteratedLocalSearchParams(callbacks=_LoggingCallback())
            ),
            objective=objective,
        )
        if objective.iteration == 0:
            logger.warning("Adaptive callback was NEVER called — check ILS params registration")
        else:
            logger.info("Adaptive callback: total iterations=%d", objective.iteration)

        weight_applied_count = objective.iteration // self.update_every
        logger.debug("weight_applied_count=%d", weight_applied_count)

        best = result.best

        if trace_path is not None:
            _save_trace(objective, trace_path)

        if not best.is_feasible():
            inf = SolverResult.infeasible(
                rebalance_moves=weight_applied_count,
                cost_delta_pct=0.0,
            )
            return (inf, inf), objective

        raw_routes = self._extract_routes(best)
        solver_result = SolverResult.from_routes_pyvrp_adapter(
            raw_routes, data, rebalance_moves=weight_applied_count
        )

        return (solver_result, solver_result), objective
