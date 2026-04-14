from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pyvrp._pyvrp as _pyvrp


def _route_distances(data: _pyvrp.ProblemData, routes: list[list[int]]) -> list[float]:
    dm = data.distance_matrix(0)
    dists = []
    for route in routes:
        d, prev = 0, 0
        for c in route:
            d += dm[prev, c]
            prev = c
        d += dm[prev, 0]
        dists.append(float(d))
    return dists


def _is_feasible(data: _pyvrp.ProblemData, routes: list[list[int]]) -> bool:
    try:
        pyvrp_routes = [
            _pyvrp.Route(data, visits=[c - 1 for c in r], vehicle_type=0)
            for r in routes if r
        ]
        return _pyvrp.Solution(data, pyvrp_routes).is_feasible()
    except Exception:
        return False


def _total_distance(data: _pyvrp.ProblemData, routes: list[list[int]]) -> float:
    return sum(_route_distances(data, routes))


def _fairness_obj(dists: list[float]) -> float:
    if len(dists) < 2:
        return 0.0
    m = sum(dists) / len(dists)
    return math.sqrt(sum((d - m) ** 2 for d in dists) / len(dists))


@dataclass
class RebalanceResult:
    routes: list[list[int]]
    total_distance: float
    fairness_std_before: float
    fairness_std_after: float
    moves_applied: int


def rebalance(
    data: _pyvrp.ProblemData,
    routes: list[list[int]],
    *,
    max_cost_increase_pct: float = 5.0,
    max_iterations: int = 2000,
    seed: int = 0,
) -> RebalanceResult:
    rng = random.Random(seed)
    current = [list(r) for r in routes]

    base_cost = _total_distance(data, current)
    cost_limit = base_cost * (1.0 + max_cost_increase_pct / 100.0)

    fairness_before = _fairness_obj(_route_distances(data, current))

    best_routes = [list(r) for r in current]
    best_fairness = fairness_before
    moves = 0

    for _ in range(max_iterations):
        if len(current) < 2:
            break

        candidate = (
            _try_relocate(data, current, rng)
            if rng.random() < 0.7
            else _try_swap(data, current, rng)
        )

        if candidate is None or not _is_feasible(data, candidate):
            continue

        if _total_distance(data, candidate) > cost_limit:
            continue

        new_fairness = _fairness_obj(_route_distances(data, candidate))
        if new_fairness < best_fairness:
            best_routes = [list(r) for r in candidate]
            best_fairness = new_fairness
            current = [list(r) for r in candidate]
            moves += 1

    return RebalanceResult(
        routes=best_routes,
        total_distance=_total_distance(data, best_routes),
        fairness_std_before=fairness_before,
        fairness_std_after=_fairness_obj(_route_distances(data, best_routes)),
        moves_applied=moves,
    )


def _try_relocate(
    data: _pyvrp.ProblemData,
    routes: list[list[int]],
    rng: random.Random,
) -> list[list[int]] | None:
    dists = _route_distances(data, routes)
    order = sorted(range(len(routes)), key=lambda i: dists[i])
    long_idx, short_idx = order[-1], order[0]

    if long_idx == short_idx or len(routes[long_idx]) <= 1:
        return None

    pos = rng.randrange(len(routes[long_idx]))
    client = routes[long_idx][pos]

    candidate = [list(r) for r in routes]
    candidate[long_idx].pop(pos)

    if not candidate[long_idx]:
        candidate.pop(long_idx)
        if short_idx > long_idx:
            short_idx -= 1

    candidate[short_idx].insert(_best_insert_pos(data, candidate[short_idx], client), client)
    return candidate


def _try_swap(
    data: _pyvrp.ProblemData,
    routes: list[list[int]],
    rng: random.Random,
) -> list[list[int]] | None:
    dists = _route_distances(data, routes)
    order = sorted(range(len(routes)), key=lambda i: dists[i])
    long_idx, short_idx = order[-1], order[0]

    if long_idx == short_idx or not routes[long_idx] or not routes[short_idx]:
        return None

    pos_long = rng.randrange(len(routes[long_idx]))
    pos_short = rng.randrange(len(routes[short_idx]))

    candidate = [list(r) for r in routes]
    candidate[long_idx][pos_long], candidate[short_idx][pos_short] = (
        candidate[short_idx][pos_short],
        candidate[long_idx][pos_long],
    )
    return candidate


def _best_insert_pos(data: _pyvrp.ProblemData, route: list[int], client: int) -> int:
    dm = data.distance_matrix(0)
    best_cost, best_pos = float("inf"), 0

    for pos in range(len(route) + 1):
        prev = 0 if pos == 0 else route[pos - 1]
        nxt = 0 if pos == len(route) else route[pos]
        cost = dm[prev, client] + dm[client, nxt] - dm[prev, nxt]
        if cost < best_cost:
            best_cost = cost
            best_pos = pos

    return best_pos
