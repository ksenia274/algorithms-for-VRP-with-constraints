from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import pyvrp._pyvrp as _pyvrp



def _route_distances(data: _pyvrp.ProblemData, routes: list[list[int]]) -> list[float]:
    dm = data.distance_matrix(0)           
    depot = 0                               
    dists = []
    for route in routes:
        d = 0
        prev = depot
        for c in route:
            d += dm[prev, c]
            prev = c
        d += dm[prev, depot]
        dists.append(float(d))
    return dists


def _route_loads(data: _pyvrp.ProblemData, routes: list[list[int]]) -> list[float]:
    loads = []
    for route in routes:
        ld = 0
        for c in route:
            loc = data.location(c)
            ld += loc.delivery[0] if hasattr(loc, "delivery") and loc.delivery else 0
        loads.append(float(ld))
    return loads


def _is_feasible(data: _pyvrp.ProblemData, routes: list[list[int]]) -> bool:
    try:
        sol = _pyvrp.Solution(data, routes)
        return sol.is_feasible()
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

    dists_before = _route_distances(data, current)
    fairness_before = _fairness_obj(dists_before)

    best_routes = [list(r) for r in current]
    best_fairness = fairness_before
    moves = 0

    for _ in range(max_iterations):
        if len(current) < 2:
            break

        if rng.random() < 0.7:
            candidate = _try_relocate(data, current, rng)
        else:
            candidate = _try_swap(data, current, rng)

        if candidate is None:
            continue

        if not _is_feasible(data, candidate):
            continue

        new_cost = _total_distance(data, candidate)
        if new_cost > cost_limit:
            continue

        new_dists = _route_distances(data, candidate)
        new_fairness = _fairness_obj(new_dists)

        if new_fairness < best_fairness:
            best_routes = [list(r) for r in candidate]
            best_fairness = new_fairness
            current = [list(r) for r in candidate]
            moves += 1

    dists_after = _route_distances(data, best_routes)

    return RebalanceResult(
        routes=best_routes,
        total_distance=_total_distance(data, best_routes),
        fairness_std_before=fairness_before,
        fairness_std_after=_fairness_obj(dists_after),
        moves_applied=moves,
    )



def _try_relocate(
    data: _pyvrp.ProblemData,
    routes: list[list[int]],
    rng: random.Random,
) -> list[list[int]] | None:
    dists = _route_distances(data, routes)
    order = sorted(range(len(routes)), key=lambda i: dists[i])
    long_idx = order[-1]
    short_idx = order[0]

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

    best_ins = _best_insert_pos(data, candidate[short_idx], client)
    candidate[short_idx].insert(best_ins, client)

    return candidate


def _try_swap(
    data: _pyvrp.ProblemData,
    routes: list[list[int]],
    rng: random.Random,
) -> list[list[int]] | None:
    dists = _route_distances(data, routes)
    order = sorted(range(len(routes)), key=lambda i: dists[i])
    long_idx = order[-1]
    short_idx = order[0]

    if long_idx == short_idx:
        return None
    if not routes[long_idx] or not routes[short_idx]:
        return None

    pos_long = rng.randrange(len(routes[long_idx]))
    pos_short = rng.randrange(len(routes[short_idx]))

    candidate = [list(r) for r in routes]
    candidate[long_idx][pos_long], candidate[short_idx][pos_short] = (
        candidate[short_idx][pos_short],
        candidate[long_idx][pos_long],
    )

    return candidate


def _best_insert_pos(
    data: _pyvrp.ProblemData,
    route: list[int],
    client: int,
) -> int:
    dm = data.distance_matrix(0)
    depot = 0
    best_cost = float("inf")
    best_pos = 0

    for pos in range(len(route) + 1):
        prev = depot if pos == 0 else route[pos - 1]
        nxt = depot if pos == len(route) else route[pos]
        cost = dm[prev, client] + dm[client, nxt] - dm[prev, nxt]
        if cost < best_cost:
            best_cost = cost
            best_pos = pos

    return best_pos