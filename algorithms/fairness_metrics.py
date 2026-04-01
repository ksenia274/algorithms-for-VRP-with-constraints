from __future__ import annotations

import math
from dataclasses import dataclass, field

from pyvrp import ProblemData


@dataclass
class FairnessReport:
    route_distances: list[float] = field(default_factory=list)
    route_loads: list[float] = field(default_factory=list)
    route_clients: list[int] = field(default_factory=list)
    route_durations: list[float] = field(default_factory=list)

    dist_mean: float = 0.0
    dist_std: float = 0.0
    dist_range: float = 0.0
    dist_min: float = 0.0
    dist_max: float = 0.0
    dist_cv: float = 0.0          
    dist_jain: float = 0.0        
    dist_gini: float = 0.0        

    load_mean: float = 0.0
    load_std: float = 0.0
    load_range: float = 0.0
    load_cv: float = 0.0
    load_jain: float = 0.0
    load_gini: float = 0.0

    clients_mean: float = 0.0
    clients_std: float = 0.0
    clients_range: float = 0.0
    clients_jain: float = 0.0

    fairness_score: float = 0.0

    def summary(self) -> str:
        lines = [
            "═══ Fairness Report ═══",
            f"  Routes: {len(self.route_distances)}",
            "",
            "  Distance:",
            f"    mean={self.dist_mean:.1f}  std={self.dist_std:.1f}  "
            f"range={self.dist_range:.1f}  CV={self.dist_cv:.3f}",
            f"    Jain={self.dist_jain:.4f}  Gini={self.dist_gini:.4f}",
            "",
            "  Load:",
            f"    mean={self.load_mean:.1f}  std={self.load_std:.1f}  "
            f"range={self.load_range:.1f}  CV={self.load_cv:.3f}",
            f"    Jain={self.load_jain:.4f}  Gini={self.load_gini:.4f}",
            "",
            "  Clients per route:",
            f"    mean={self.clients_mean:.1f}  std={self.clients_std:.1f}  "
            f"range={self.clients_range:.1f}  Jain={self.clients_jain:.4f}",
            "",
            f"  * Composite fairness score: {self.fairness_score:.4f}",
            f"    (0 = perfectly fair, lower is better)",
        ]
        return "\n".join(lines)



def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _cv(vals: list[float]) -> float:
    """Coefficient of Variation: std / mean."""
    m = _mean(vals)
    return _std(vals) / m if m > 0 else 0.0


def _jain(vals: list[float]) -> float:
    """
    Jain's Fairness Index:  (Σ xi)² / (n · Σ xi²)
    Возвращает 1.0 при полном равенстве.
    """
    n = len(vals)
    if n == 0:
        return 1.0
    s = sum(vals)
    ss = sum(v * v for v in vals)
    if ss == 0:
        return 1.0
    return (s * s) / (n * ss)


def _gini(vals: list[float]) -> float:
    """
    Gini coefficient: 0 = полное равенство, 1 = полное неравенство.
    """
    n = len(vals)
    if n < 2:
        return 0.0
    s = sorted(vals)
    total = sum(s)
    if total == 0:
        return 0.0
    cum = 0.0
    area = 0.0
    for v in s:
        cum += v
        area += cum
    return 1.0 - (2.0 * area) / (n * total) + 1.0 / n



def compute_fairness(
    route_distances: list[float],
    route_loads: list[float],
    route_clients: list[int],
    route_durations: list[float] | None = None,
    *,
    weight_dist: float = 0.5,
    weight_load: float = 0.3,
    weight_clients: float = 0.2,
) -> FairnessReport:
    rd = [float(v) for v in route_distances]
    rl = [float(v) for v in route_loads]
    rc = [int(v) for v in route_clients]
    rdu = [float(v) for v in (route_durations or [])]

    report = FairnessReport(
        route_distances=rd,
        route_loads=rl,
        route_clients=rc,
        route_durations=rdu,
    )

    if len(rd) < 2:
        report.dist_jain = 1.0
        report.load_jain = 1.0
        report.clients_jain = 1.0
        return report

    report.dist_mean = _mean(rd)
    report.dist_std = _std(rd)
    report.dist_min = min(rd)
    report.dist_max = max(rd)
    report.dist_range = report.dist_max - report.dist_min
    report.dist_cv = _cv(rd)
    report.dist_jain = _jain(rd)
    report.dist_gini = _gini(rd)

    report.load_mean = _mean(rl)
    report.load_std = _std(rl)
    report.load_range = max(rl) - min(rl) if rl else 0.0
    report.load_cv = _cv(rl)
    report.load_jain = _jain(rl)
    report.load_gini = _gini(rl)

    rc_f = [float(v) for v in rc]
    report.clients_mean = _mean(rc_f)
    report.clients_std = _std(rc_f)
    report.clients_range = max(rc_f) - min(rc_f) if rc_f else 0.0
    report.clients_jain = _jain(rc_f)

    report.fairness_score = (
        weight_dist * report.dist_cv
        + weight_load * report.load_cv
        + weight_clients * _cv(rc_f)
    )

    return report

def compute_fairness_for_routes(
        data: ProblemData,
        routes: list[list[int]],
    ) -> FairnessReport:
        dm = data.distance_matrix(0)
        depot = 0

        distances = []
        loads = []
        clients_count = []
        durations = []

        for route in routes:
            d = 0
            prev = depot
            for c in route:
                d += dm[prev, c]
                prev = c
            d += dm[prev, depot]
            distances.append(float(d))

            ld = 0
            for c in route:
                loc = data.location(c)
                if hasattr(loc, "delivery") and loc.delivery:
                    ld += loc.delivery[0]
            loads.append(float(ld))

            clients_count.append(len(route))
            durations.append(float(d))

        return compute_fairness(
            route_distances=distances,
            route_loads=loads,
            route_clients=clients_count,
            route_durations=durations,
        )
