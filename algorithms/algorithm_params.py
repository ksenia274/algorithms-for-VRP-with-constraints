from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class HgsSimpleParams(BaseModel, extra="forbid"):
    max_distance: Optional[float] = None


class HgsRebalanceParams(BaseModel, extra="forbid"):
    enable_fairness: bool = True
    max_cost_increase_pct: float = 5.0
    rebalance_iterations: int = 3000
    use_prizes: bool = False


class HgsPenaltyParams(BaseModel, extra="forbid"):
    fairness_weight: float = 0.3
    num_restarts: int = 5
    max_cost_increase_pct: float = 5.0


class HgsAdaptiveParams(BaseModel, extra="forbid"):
    initial_route_balance: float = 500.0
    strategy: Literal["linear", "fairness_signal"] = "linear"
    decay: float = 0.99999
    min_weight: float = 0.0
    max_weight: float = 1e9
    target_cv: float = 0.2
    hold_band: float = 0.05
    boost_factor: float = 1.05
    fs_decay_factor: float = 0.995
    fs_ma_window: int = 20
    update_every: int = 1
    trace: bool = False


class AlnsParams(BaseModel, extra="forbid"):
    enable_fairness: bool = False
    fairness_weight: float = 100.0
    max_iterations: int = 25000
    sa_start_temp: float = 100.0
    sa_end_temp: float = 0.5


ALGORITHM_PARAMS_REGISTRY: dict[str, type[BaseModel]] = {
    "hgs_simple": HgsSimpleParams,
    "hgs_rebalance": HgsRebalanceParams,
    "hgs_penalty": HgsPenaltyParams,
    "hgs_adaptive": HgsAdaptiveParams,
    "alns": AlnsParams,
}
