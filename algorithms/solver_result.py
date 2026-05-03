from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

import yaml
from pydantic import BaseModel

from metrics.fairness import FairnessReport, compute_fairness

if TYPE_CHECKING:
    from pyvrp import ProblemData


@dataclass(frozen=True)
class SolverConfig:
    """Immutable description of a single solver run."""

    schema_version: str
    algorithm: str       # display name for CSV/plots (may equal algorithm_type)
    instance: str
    instance_kind: str   # "yandex" | "solomon"
    time_limit: int
    seed: int
    capacity: int
    num_vehicles: int
    algorithm_params: BaseModel
    algorithm_type: str = ""  # type key for factory dispatch; defaults to algorithm

    def __post_init__(self) -> None:
        if not self.algorithm_type:
            object.__setattr__(self, "algorithm_type", self.algorithm)

    def to_yaml(self) -> str:
        d = {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "instance": self.instance,
            "instance_kind": self.instance_kind,
            "time_limit": self.time_limit,
            "seed": self.seed,
            "capacity": self.capacity,
            "num_vehicles": self.num_vehicles,
            "algorithm_params": self.algorithm_params.model_dump(),
        }
        if self.algorithm_type != self.algorithm:
            d["algorithm_type"] = self.algorithm_type
        return yaml.dump(d, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> SolverConfig:
        from algorithms.algorithm_params import ALGORITHM_PARAMS_REGISTRY
        d = yaml.safe_load(text)
        algorithm = d["algorithm"]
        algorithm_type = d.get("algorithm_type") or algorithm
        params_cls = ALGORITHM_PARAMS_REGISTRY.get(algorithm_type)
        if params_cls is None:
            raise ValueError(f"Unknown algorithm_type '{algorithm_type}' in ALGORITHM_PARAMS_REGISTRY")
        params = params_cls.model_validate(d.get("algorithm_params") or {})
        return cls(
            schema_version=d.get("schema_version", "1.0"),
            algorithm=algorithm,
            instance=d["instance"],
            instance_kind=d["instance_kind"],
            time_limit=int(d["time_limit"]),
            seed=int(d["seed"]),
            capacity=int(d["capacity"]),
            num_vehicles=int(d["num_vehicles"]),
            algorithm_params=params,
            algorithm_type=algorithm_type,
        )

    def content_hash(self) -> str:
        digest = hashlib.sha256(self.to_yaml().encode()).hexdigest()
        return digest[:6]


@dataclass(frozen=True)
class SolverDiagnostics:
    """Timing and move-count diagnostics for a solver run."""

    solve_time_s: float
    rebalance_moves: Optional[int] = None
    cost_delta_pct: Optional[float] = None
    weight_applied_count: Optional[int] = None
    iterations: Optional[int] = None


@dataclass
class SolverResult:
    """Complete result of a single solver run including fairness metrics."""

    schema_version: str
    routes: list[list[int]]
    total_distance: float
    num_routes: int
    feasible: bool
    fairness: Optional[FairnessReport]
    config: SolverConfig
    diagnostics: SolverDiagnostics
    initial: Optional[SolverResult] = None  # initial.initial is always None
    artifacts: dict[str, str] = field(default_factory=dict)  # relative paths to run-dir

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def infeasible(
        cls,
        *,
        config: SolverConfig,
        diagnostics: SolverDiagnostics,
    ) -> SolverResult:
        return cls(
            schema_version="1.0",
            routes=[],
            total_distance=float("inf"),
            num_routes=0,
            feasible=False,
            fairness=None,
            config=config,
            diagnostics=diagnostics,
        )

    @classmethod
    def from_routes(
        cls,
        *,
        routes: list[list[int]],
        distance_matrix: Sequence[Sequence[float]],
        loc_loads: Sequence[float],
        feasible: bool,
        config: SolverConfig,
        diagnostics: SolverDiagnostics,
        initial: Optional[SolverResult] = None,
        artifacts: Optional[dict[str, str]] = None,
    ) -> SolverResult:
        depot = 0
        distances, loads, clients_count = [], [], []
        for route in routes:
            d, prev = 0.0, depot
            for c in route:
                d += distance_matrix[prev][c]
                prev = c
            d += distance_matrix[prev][depot]
            distances.append(d)
            loads.append(float(sum(loc_loads[c] for c in route)))
            clients_count.append(len(route))
        fairness = compute_fairness(
            route_distances=distances,
            route_loads=loads,
            route_clients=clients_count,
        ) if routes else None
        return cls(
            schema_version="1.0",
            routes=routes,
            total_distance=sum(distances),
            num_routes=len(routes),
            feasible=feasible,
            fairness=fairness,
            config=config,
            diagnostics=diagnostics,
            initial=initial,
            artifacts=artifacts or {},
        )

    @classmethod
    def from_pyvrp_solution(
        cls,
        *,
        solution: Any,
        data: Any,  # pyvrp.ProblemData
        config: SolverConfig,
        diagnostics: SolverDiagnostics,
        initial: Optional[SolverResult] = None,
        artifacts: Optional[dict[str, str]] = None,
    ) -> SolverResult:
        from pyvrp._pyvrp import ActivityType

        routes: list[list[int]] = []
        for route in solution.routes():
            clients = [
                visit.location_idx()
                for visit in route.visits()
                if visit.activity_type() == ActivityType.CLIENT
            ]
            if clients:
                routes.append(clients)

        dm = data.distance_matrix(0)
        loc_loads = [0.0] * data.num_locations
        for j in range(data.num_clients):
            client = data.client(j)
            if client.delivery:
                loc_loads[client.location] = float(client.delivery[0])

        return cls.from_routes(
            routes=routes,
            distance_matrix=dm,
            loc_loads=loc_loads,
            feasible=solution.is_feasible(),
            config=config,
            diagnostics=diagnostics,
            initial=initial,
            artifacts=artifacts,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        def _dim_to_dict(dm: Any) -> dict[str, float]:
            return {
                "mean": dm.mean, "std": dm.std, "min": dm.min, "max": dm.max,
                "worst_ratio": dm.worst_ratio, "gini": dm.gini, "cv": dm.cv,
            }

        fairness_dict: Optional[dict] = None
        if self.fairness is not None:
            fairness_dict = {
                "distance": _dim_to_dict(self.fairness.distance),
                "load": _dim_to_dict(self.fairness.load),
                "clients": _dim_to_dict(self.fairness.clients),
            }

        diag = self.diagnostics
        return {
            "schema_version": self.schema_version,
            "routes": self.routes,
            "total_distance": self.total_distance,
            "num_routes": self.num_routes,
            "feasible": self.feasible,
            "fairness": fairness_dict,
            "config": {
                "schema_version": self.config.schema_version,
                "algorithm": self.config.algorithm,
                "instance": self.config.instance,
                "instance_kind": self.config.instance_kind,
                "time_limit": self.config.time_limit,
                "seed": self.config.seed,
                "capacity": self.config.capacity,
                "num_vehicles": self.config.num_vehicles,
                "algorithm_params": self.config.algorithm_params.model_dump(),
            },
            "diagnostics": {
                "solve_time_s": diag.solve_time_s,
                "rebalance_moves": diag.rebalance_moves,
                "cost_delta_pct": diag.cost_delta_pct,
                "weight_applied_count": diag.weight_applied_count,
                "iterations": diag.iterations,
            },
            "initial": self.initial.to_dict() if self.initial is not None else None,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolverResult:
        from algorithms.algorithm_params import ALGORITHM_PARAMS_REGISTRY
        from metrics.fairness import DimensionMetrics, FairnessReport

        cfg_d = data["config"]
        params_cls = ALGORITHM_PARAMS_REGISTRY.get(cfg_d["algorithm"])
        if params_cls is None:
            raise ValueError(f"Unknown algorithm '{cfg_d['algorithm']}'")
        config = SolverConfig(
            schema_version=cfg_d.get("schema_version", "1.0"),
            algorithm=cfg_d["algorithm"],
            instance=cfg_d["instance"],
            instance_kind=cfg_d["instance_kind"],
            time_limit=cfg_d["time_limit"],
            seed=cfg_d["seed"],
            capacity=cfg_d["capacity"],
            num_vehicles=cfg_d["num_vehicles"],
            algorithm_params=params_cls.model_validate(cfg_d.get("algorithm_params") or {}),
        )
        diag_d = data["diagnostics"]
        diagnostics = SolverDiagnostics(
            solve_time_s=diag_d["solve_time_s"],
            rebalance_moves=diag_d.get("rebalance_moves"),
            cost_delta_pct=diag_d.get("cost_delta_pct"),
            weight_applied_count=diag_d.get("weight_applied_count"),
            iterations=diag_d.get("iterations"),
        )
        fairness: Optional[FairnessReport] = None
        if data.get("fairness") is not None:
            fd = data["fairness"]

            def _dm(d: dict) -> DimensionMetrics:
                return DimensionMetrics(**d)

            fairness = FairnessReport(
                distance=_dm(fd["distance"]),
                load=_dm(fd["load"]),
                clients=_dm(fd["clients"]),
            )
        initial: Optional[SolverResult] = None
        if data.get("initial") is not None:
            initial = cls.from_dict(data["initial"])
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            routes=data["routes"],
            total_distance=data["total_distance"],
            num_routes=data["num_routes"],
            feasible=data["feasible"],
            fairness=fairness,
            config=config,
            diagnostics=diagnostics,
            initial=initial,
            artifacts=data.get("artifacts") or {},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def primary_metric(self) -> float:
        """Return the primary fairness metric defined in global_config.metrics.primary."""
        from runtime.global_config import get_global_config
        if self.fairness is None:
            return float("inf")
        cfg = get_global_config()
        return self.fairness.value(cfg.metrics.primary)

    def summary(self, title: str = "Result") -> str:
        bar = "-" * 46
        if not self.feasible:
            return f"[{title}]\n  INFEASIBLE\n{bar}"

        lines = [f"[{title}]", f"  Distance : {self.total_distance:.1f}   Routes: {self.num_routes}"]
        d = self.diagnostics
        if d.cost_delta_pct is not None:
            lines.append(f"  Cost delta: {d.cost_delta_pct:+.2f}%")
        if d.rebalance_moves is not None:
            lines.append(f"  Moves     : {d.rebalance_moves}")

        if self.fairness and self.num_routes >= 2:
            f = self.fairness
            lines.append("  --- Fairness ---")
            lines.append(f"  Distance : worst_ratio={f.distance.worst_ratio:.4f}  gini={f.distance.gini:.4f}  cv={f.distance.cv:.4f}")
            if f.load.mean > 0:
                lines.append(f"  Load     : worst_ratio={f.load.worst_ratio:.4f}  gini={f.load.gini:.4f}  cv={f.load.cv:.4f}")
            lines.append(f"  Clients  : worst_ratio={f.clients.worst_ratio:.4f}  gini={f.clients.gini:.4f}  cv={f.clients.cv:.4f}")

        lines.append(bar)
        return "\n".join(lines)
