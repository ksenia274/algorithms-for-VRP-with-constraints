from __future__ import annotations

import logging
from collections import deque
from enum import Enum, auto

from pyvrp.adaptive_objective import AdaptationStrategy, IterationMetrics, ObjectiveWeights

logger = logging.getLogger(__name__)


class _State(Enum):
    BOOST = auto()
    DECAY = auto()
    HOLD = auto()


class FairnessSignalAdjustment(AdaptationStrategy):
    def __init__(
        self,
        target_cv: float = 0.2,
        hold_band: float = 0.05,
        boost_factor: float = 1.05,
        decay_factor: float = 0.995,
        ma_window: int = 20,
        min_weight: float = 1.0,
        max_weight: float = 1e8,
    ):
        if boost_factor <= 1:
            raise ValueError("boost_factor must be > 1.")
        if not (0 < decay_factor < 1):
            raise ValueError("decay_factor must be in (0, 1).")
        self._target = target_cv
        self._band = hold_band
        self._boost = boost_factor
        self._decay = decay_factor
        self._ma_window = ma_window
        self._min = min_weight
        self._max = max_weight
        self._cv_history: deque[float] = deque(maxlen=ma_window)
        self._state = _State.HOLD
        self._state_history: list[tuple[int, str]] = []

    @property
    def state(self) -> str:
        return self._state.name

    @property
    def state_history(self) -> list[tuple[int, str]]:
        return list(self._state_history)

    def fairness_delta_ma(self, k: int | None = None) -> float:
        window = list(self._cv_history)
        if k is not None:
            window = window[-k:]
        if len(window) < 2:
            return 0.0
        deltas = [window[i] - window[i - 1] for i in range(1, len(window))]
        return sum(deltas) / len(deltas)

    def _compute_cv(self, metrics: IterationMetrics) -> float:
        dists = metrics.per_route_distances
        if dists and len(dists) >= 2:
            mean = sum(dists) / len(dists)
            if mean > 0:
                return (max(dists) - min(dists)) / mean
        return metrics.route_balance

    def _decide_state(self, cv: float) -> _State:
        delta_ma = self.fairness_delta_ma()
        if cv > self._target + self._band:
            return _State.BOOST if delta_ma >= 0 else _State.HOLD
        if cv < self._target - self._band:
            return _State.DECAY
        return _State.HOLD

    def update(self, weights: ObjectiveWeights, metrics: IterationMetrics) -> ObjectiveWeights:
        cv = self._compute_cv(metrics)
        self._cv_history.append(cv)

        new_state = self._decide_state(cv)
        if new_state != self._state:
            logger.debug(
                "FairnessSignal iter=%d  %s→%s  cv=%.4f  delta_ma=%.6f",
                metrics.iteration, self._state.name, new_state.name,
                cv, self.fairness_delta_ma(),
            )
        self._state = new_state
        self._state_history.append((metrics.iteration, new_state.name))

        factor = {_State.BOOST: self._boost, _State.DECAY: self._decay, _State.HOLD: 1.0}[self._state]

        def _adj(v: float) -> float:
            return max(self._min, min(v * factor, self._max))

        new_w = ObjectiveWeights(
            vehicle_count=weights.vehicle_count,
            route_balance=_adj(weights.route_balance),
            dist=weights.dist,
            time=weights.time,
        )
        logger.debug(
            "iter=%d  state=%s  cv=%.4f  w_rb: %.1f→%.1f",
            metrics.iteration, self._state.name, cv,
            weights.route_balance, new_w.route_balance,
        )
        return new_w
