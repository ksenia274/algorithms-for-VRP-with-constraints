from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

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

from algorithms.algorithm_params import HgsAdaptiveParams
from algorithms.fairness_signal_strategy import FairnessSignalAdjustment
from algorithms.hgs_base import HGSBase
from algorithms.solver_result import SolverConfig, SolverDiagnostics, SolverResult
from data.vrp_instance import VRPInstanceInput

logger = logging.getLogger(__name__)

_LOG_EVERY = 100


def _solve_with_weighted_objective(
    data,
    stop,
    seed: int,
    collect_stats: bool,
    display: bool,
    params: SolveParams,
    objective: AdaptiveObjective,
):
    """Mirror of pyvrp.solve.solve() with a monkey-patched PenaltyManager.

    # TODO(fork): replace monkey-patch when PyVRP fork exposes set_custom_weights publicly
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
    # TODO(fork): replace monkey-patch when PyVRP fork exposes set_custom_weights publicly
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


def _save_artifacts(
    objective: AdaptiveObjective,
    run_dir: Path,
) -> dict[str, str]:
    """Save weight_history.csv.gz and trace.csv.gz to run_dir.

    Returns dict of relative artifact paths.
    """
    import pandas as pd

    df = objective.get_history_dataframe()
    if df is None or df.empty:
        return {}

    df = df.copy()

    # Compute event column from weight shifts
    w = df["weight_route_balance"]
    nxt = w.shift(-1)
    eps = 1e-9
    event = pd.Series("hold", index=df.index, dtype=str)
    event[nxt > w + eps] = "boost"
    event[nxt < w - eps] = "decay"
    event.iloc[-1] = "hold"
    df["event"] = event

    # Add both CV variants; rename old column
    if "route_balance" in df.columns:
        df = df.rename(columns={"route_balance": "route_range_pct"})
    elif "route_range_pct" not in df.columns:
        if "max_route_dist" in df.columns and "min_route_dist" in df.columns:
            mean_ = (df["max_route_dist"] + df["min_route_dist"]) / 2.0
            df["route_range_pct"] = (df["max_route_dist"] - df["min_route_dist"]) / mean_.replace(0, float("nan"))
        else:
            df["route_range_pct"] = float("nan")

    # route_cv: proper std/mean — not reconstructible from max/min alone, set NaN
    # A future PyVRP fork version could expose per-route distances in IterationMetrics
    if "route_cv" not in df.columns:
        df["route_cv"] = float("nan")

    artifacts: dict[str, str] = {}

    trace_path = run_dir / "trace.csv.gz"
    df.to_csv(trace_path, index=False, compression="gzip")
    artifacts["trace"] = "trace.csv.gz"

    weight_cols = [c for c in df.columns if c.startswith("weight_")]
    weight_history_path = run_dir / "weight_history.csv.gz"
    df[["iteration"] + weight_cols + ["event"]].to_csv(
        weight_history_path, index=False, compression="gzip"
    )
    artifacts["weight_history"] = "weight_history.csv.gz"

    logger.info("Artifacts saved to %s (%d rows)", run_dir, len(df))
    return artifacts


class HGSSolverAdaptive(HGSBase):
    """HGS with adaptive route-balance objective weight."""

    def __init__(self, config: SolverConfig) -> None:
        if not isinstance(config.algorithm_params, HgsAdaptiveParams):
            raise TypeError(f"Expected HgsAdaptiveParams, got {type(config.algorithm_params)}")
        super().__init__(config)
        self.params: HgsAdaptiveParams = config.algorithm_params

    def solve(
        self,
        instance: str | VRPInstanceInput,
        *,
        run_dir: Optional[Path] = None,
    ) -> SolverResult:
        t0 = time.perf_counter()
        _, data, _, _ = self._build_model(instance)
        p = self.params

        if p.strategy == "fairness_signal":
            strat = FairnessSignalAdjustment(
                target_cv=p.target_cv,
                hold_band=p.hold_band,
                boost_factor=p.boost_factor,
                decay_factor=p.fs_decay_factor,
                ma_window=p.fs_ma_window,
                min_weight=p.min_weight if p.min_weight > 0 else 1.0,
                max_weight=p.max_weight,
            )
        else:
            strat = LinearDecay(decay=p.decay)

        objective = AdaptiveObjective(
            initial_weights=ObjectiveWeights(route_balance=p.initial_route_balance),
            strategy=strat,
            update_every=p.update_every,
        )

        adaptive_cb = objective.as_callback()

        class _Callback(IteratedLocalSearchCallbacks):
            def on_iteration(self, current, candidate, best, cost_evaluator):
                adaptive_cb.on_iteration(current, candidate, best, cost_evaluator)
                n = objective.iteration
                if n % _LOG_EVERY == 0:
                    hist = objective.get_history()
                    feas = hist[-1].feasibility_rate if hist else float("nan")
                    logger.debug("iter=%d  rb_weight=%.4f  feas=%.3f", n, objective.weights.route_balance, feas)

            def on_restart(self, best):
                adaptive_cb.on_restart(best)

        result = _solve_with_weighted_objective(
            data=data,
            stop=pyvrp.stop.MaxRuntime(self.time_limit),
            seed=self.seed,
            collect_stats=True,
            display=False,
            params=SolveParams(ils=IteratedLocalSearchParams(callbacks=_Callback())),
            objective=objective,
        )

        weight_applied_count = objective.iteration // p.update_every if p.update_every else 0
        solve_time = time.perf_counter() - t0

        artifacts: dict[str, str] = {}
        if p.trace and run_dir is not None:
            try:
                artifacts = _save_artifacts(objective, run_dir)
            except Exception:
                logger.exception("Failed to save adaptive artifacts")

        best = result.best
        diagnostics = SolverDiagnostics(
            solve_time_s=solve_time,
            weight_applied_count=weight_applied_count,
        )

        if not best.is_feasible():
            return SolverResult.infeasible(config=self.config, diagnostics=diagnostics)

        routes = self._extract_routes(best)
        return self._make_result(routes, data, diagnostics=diagnostics, artifacts=artifacts or None)
