import math

import numpy as np
import pandas as pd
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.State import State
from alns.stop import MaxRuntime

from algorithms.fairness_metrics import FairnessReport, compute_fairness
from data.load_solomon import load_instance
from data.vrp_instance import VRPInstanceInput


UNASSIGNED_PENALTY = 1_000_000


class InstanceData:

    def __init__(self, n_clients, coords, demands, tw_early, tw_late,
                 service_time, capacity, num_vehicles, dist_matrix,
                 time_matrix=None,
                 enable_fairness=False, fairness_weight=0.0):
        self.n_clients = n_clients
        self.coords = coords
        self.demands = demands
        self.tw_early = tw_early
        self.tw_late = tw_late
        self.service_time = service_time
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        self.dm = dist_matrix           # used for cost (objective)
        self.tm = time_matrix if time_matrix is not None else dist_matrix  # used for TW checks
        self.enable_fairness = enable_fairness
        self.fairness_weight = fairness_weight


def _parse_instance(instance, capacity, num_vehicles,
                    enable_fairness, fairness_weight):
    if isinstance(instance, VRPInstanceInput):
        inp = instance
        df = inp.df.copy()
        df.columns = df.columns.str.strip()
    elif isinstance(instance, str) and instance.endswith(".json"):
        from data.load_yandex_instance import load_yandex_instance
        inp = load_yandex_instance(instance)
        if capacity == 200 and inp.recommended_capacity is not None:
            capacity = inp.recommended_capacity
        df = inp.df.copy()
        df.columns = df.columns.str.strip()
    elif instance.endswith(".csv") or ("/" in instance) or ("\\" in instance):
        df = pd.read_csv(instance)
        df.columns = df.columns.str.strip()
        inp = VRPInstanceInput(df=df)
    else:
        df = load_instance(instance)
        df.columns = df.columns.str.strip()
        inp = VRPInstanceInput(df=df)

    n = len(df) - 1
    # ensure enough vehicles so every client can be assigned
    num_vehicles = max(num_vehicles, math.ceil(n / max(capacity, 1)))
    coords = []
    demands = []
    tw_early = []
    tw_late = []
    service_time = []

    for _, row in df.iterrows():
        coords.append((int(row["XCOORD."]), int(row["YCOORD."])))
        demands.append(int(row["DEMAND"]))
        tw_early.append(int(row["READY TIME"]))
        tw_late.append(int(row["DUE DATE"]))
        service_time.append(int(row["SERVICE TIME"]))

    size = n + 1
    if inp.dist_matrix is not None:
        dm = [[int(inp.dist_matrix[i][j]) for j in range(size)]
              for i in range(size)]
    else:
        dm = [[0] * size for _ in range(size)]
        for i in range(size):
            xi, yi = coords[i]
            for j in range(i + 1, size):
                xj, yj = coords[j]
                d = int(math.hypot(xi - xj, yi - yj))
                dm[i][j] = d
                dm[j][i] = d

    tm = None
    if inp.time_matrix is not None:
        raw_tm = inp.time_matrix
        if isinstance(raw_tm[0][0], list):
            # 3D time-dependent: average over periods
            n_periods = len(raw_tm)
            tm = [[int(sum(raw_tm[p][i][j] for p in range(n_periods)) / n_periods)
                   for j in range(size)]
                  for i in range(size)]
        else:
            tm = [[int(raw_tm[i][j]) for j in range(size)]
                  for i in range(size)]

    return InstanceData(
        n_clients=n, coords=coords, demands=demands,
        tw_early=tw_early, tw_late=tw_late, service_time=service_time,
        capacity=capacity, num_vehicles=num_vehicles, dist_matrix=dm,
        time_matrix=tm,
        enable_fairness=enable_fairness, fairness_weight=fairness_weight,
    )


class VRPState(State):
    def __init__(self, routes, unassigned, data):
        self.routes = routes
        self.unassigned = unassigned
        self.data = data

    def copy(self):
        return VRPState(
            [list(r) for r in self.routes],
            list(self.unassigned),
            self.data,
        )

    def objective(self):
        total = self._total_distance()
        if self.unassigned:
            total += len(self.unassigned) * UNASSIGNED_PENALTY
        if self.data.enable_fairness and len(self.routes) >= 2:
            total += self.data.fairness_weight * self._fairness_penalty()
        return float(total)

    def _total_distance(self):
        dm = self.data.dm
        total = 0
        for route in self.routes:
            if not route:
                continue
            prev = 0
            for c in route:
                total += dm[prev][c]
                prev = c
            total += dm[prev][0]
        return total

    def _fairness_penalty(self):
        dists = self._route_distances()
        if len(dists) < 2:
            return 0.0
        mean = sum(dists) / len(dists)
        return math.sqrt(sum((d - mean) ** 2 for d in dists) / len(dists))

    def _route_distances(self):
        dm = self.data.dm
        result = []
        for route in self.routes:
            if not route:
                continue
            d = 0
            prev = 0
            for c in route:
                d += dm[prev][c]
                prev = c
            d += dm[prev][0]
            result.append(d)
        return result



def _compute_departures(route, data):
    tm = data.tm  # travel-time matrix for TW feasibility
    deps = []
    time = data.tw_early[0]
    prev = 0
    for c in route:
        arrival = time + tm[prev][c]
        if arrival > data.tw_late[c]:
            return None
        time = max(arrival, data.tw_early[c]) + data.service_time[c]
        deps.append(time)
        prev = c
    return deps


def _can_insert_at(route, pos, client, data, deps):
    tm = data.tm  # travel-time matrix for TW feasibility
    tw_late = data.tw_late
    tw_early = data.tw_early
    svc = data.service_time

    prev = 0 if pos == 0 else route[pos - 1]
    prev_dep = data.tw_early[0] if pos == 0 else deps[pos - 1]

    arr = prev_dep + tm[prev][client]
    if arr > tw_late[client]:
        return False
    time = (arr if arr >= tw_early[client] else tw_early[client]) + svc[client]

    prev_node = client
    for i in range(pos, len(route)):
        c = route[i]
        arr = time + tm[prev_node][c]
        if arr > tw_late[c]:
            return False
        new_dep = (arr if arr >= tw_early[c] else tw_early[c]) + svc[c]
        if new_dep <= deps[i]:
            return True
        time = new_dep
        prev_node = c

    last = client if pos == len(route) else route[-1]
    if time + tm[last][0] > tw_late[0]:
        return False
    return True


def _best_insertion(route, client, data, load=None, deps=None):
    if load is None:
        load = sum(data.demands[c] for c in route)
    if load + data.demands[client] > data.capacity:
        return None

    if deps is None:
        deps = _compute_departures(route, data)
    if deps is None and route:
        return None
    if not route:
        deps = []

    dm = data.dm
    best_pos = None
    best_cost = float("inf")

    for pos in range(len(route) + 1):
        if not _can_insert_at(route, pos, client, data, deps):
            continue
        prev = 0 if pos == 0 else route[pos - 1]
        nxt = 0 if pos == len(route) else route[pos]
        cost = dm[prev][client] + dm[client][nxt] - dm[prev][nxt]
        if cost < best_cost:
            best_cost = cost
            best_pos = pos

    return (best_pos, best_cost) if best_pos is not None else None


def _build_initial_solution(data):
    clients = list(range(1, data.n_clients + 1))
    clients.sort(key=lambda c: data.tw_early[c])

    routes = []
    route_loads = []
    route_deps = []
    unassigned = []

    for client in clients:
        best_r = None
        best_pos = None
        best_cost = float("inf")

        for ri, route in enumerate(routes):
            result = _best_insertion(
                route, client, data,
                load=route_loads[ri], deps=route_deps[ri],
            )
            if result is not None:
                pos, cost = result
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
                    best_r = ri

        if best_r is not None:
            routes[best_r].insert(best_pos, client)
            route_loads[best_r] += data.demands[client]
            route_deps[best_r] = _compute_departures(routes[best_r], data)
        elif len(routes) < data.num_vehicles:
            routes.append([client])
            route_loads.append(data.demands[client])
            route_deps.append(_compute_departures([client], data))
        else:
            unassigned.append(client)

    return VRPState(routes, unassigned, data)


def _n_to_remove(state, rng):
    total = sum(len(r) for r in state.routes)
    if total <= 1:
        return 1
    lo = max(1, int(total * 0.1))
    hi = max(lo + 1, int(total * 0.4))
    return int(rng.integers(lo, hi))


def random_removal(state, rng):
    state = state.copy()
    n_remove = _n_to_remove(state, rng)
    all_clients = [c for route in state.routes for c in route]
    if not all_clients:
        return state
    n_remove = min(n_remove, len(all_clients))
    indices = rng.choice(len(all_clients), size=n_remove, replace=False)
    to_remove = set(all_clients[i] for i in indices)
    state.routes = [[c for c in r if c not in to_remove] for r in state.routes]
    state.routes = [r for r in state.routes if r]
    state.unassigned.extend(to_remove)
    return state


def worst_removal(state, rng):
    state = state.copy()
    n_remove = _n_to_remove(state, rng)
    dm = state.data.dm
    savings = []
    for route in state.routes:
        for i, client in enumerate(route):
            prev = 0 if i == 0 else route[i - 1]
            nxt = 0 if i == len(route) - 1 else route[i + 1]
            savings.append((dm[prev][client] + dm[client][nxt] - dm[prev][nxt],
                            client))
    savings.sort(reverse=True)
    to_remove = set()
    for _, client in savings:
        if len(to_remove) >= n_remove:
            break
        to_remove.add(client)
    state.routes = [[c for c in r if c not in to_remove] for r in state.routes]
    state.routes = [r for r in state.routes if r]
    state.unassigned.extend(to_remove)
    return state


def shaw_removal(state, rng):
    state = state.copy()
    n_remove = _n_to_remove(state, rng)
    dm = state.data.dm
    data = state.data

    all_clients = [c for route in state.routes for c in route]
    if not all_clients:
        return state

    seed_client = all_clients[int(rng.integers(0, len(all_clients)))]

    max_dist = max(dm[seed_client]) or 1
    max_demand = max(data.demands) or 1
    tw_range = max(1, max(data.tw_late) - min(data.tw_early))

    relatedness = []
    for c in all_clients:
        if c == seed_client:
            continue
        r = (dm[seed_client][c] / max_dist
             + abs(data.tw_early[c] - data.tw_early[seed_client]) / tw_range
             + abs(data.demands[c] - data.demands[seed_client]) / max_demand)
        relatedness.append((r, c))
    relatedness.sort()

    to_remove = {seed_client}
    for _, client in relatedness:
        if len(to_remove) >= n_remove:
            break
        to_remove.add(client)

    state.routes = [[c for c in r if c not in to_remove] for r in state.routes]
    state.routes = [r for r in state.routes if r]
    state.unassigned.extend(to_remove)
    return state


def greedy_repair(state, rng):
    state = state.copy()
    data = state.data
    order = list(range(len(state.unassigned)))
    rng.shuffle(order)
    shuffled = [state.unassigned[i] for i in order]

    loads = [sum(data.demands[c] for c in r) for r in state.routes]
    deps = [_compute_departures(r, data) for r in state.routes]

    still_unassigned = []
    for client in shuffled:
        best_r = None
        best_pos = None
        best_cost = float("inf")

        for ri, route in enumerate(state.routes):
            result = _best_insertion(route, client, data,
                                    load=loads[ri], deps=deps[ri])
            if result is not None:
                pos, cost = result
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
                    best_r = ri

        if best_r is not None:
            state.routes[best_r].insert(best_pos, client)
            loads[best_r] += data.demands[client]
            deps[best_r] = _compute_departures(state.routes[best_r], data)
        elif len(state.routes) < data.num_vehicles:
            state.routes.append([client])
            loads.append(data.demands[client])
            deps.append(_compute_departures([client], data))
        else:
            still_unassigned.append(client)

    state.unassigned = still_unassigned
    return state


def regret_2_repair(state, rng):
    state = state.copy()
    data = state.data

    loads = [sum(data.demands[c] for c in r) for r in state.routes]
    deps = [_compute_departures(r, data) for r in state.routes]

    while state.unassigned:
        best_client = None
        best_route_idx = None
        best_pos = None
        max_regret = -float("inf")

        for client in state.unassigned:
            insertions = []
            for ri, route in enumerate(state.routes):
                result = _best_insertion(route, client, data,
                                        load=loads[ri], deps=deps[ri])
                if result is not None:
                    pos, cost = result
                    insertions.append((cost, ri, pos))

            if len(state.routes) < data.num_vehicles:
                insertions.append((0, -1, 0))

            if not insertions:
                continue

            insertions.sort()
            regret = (UNASSIGNED_PENALTY if len(insertions) == 1
                      else insertions[1][0] - insertions[0][0])

            if regret > max_regret:
                max_regret = regret
                best_client = client
                _, best_route_idx, best_pos = insertions[0]

        if best_client is None:
            break

        state.unassigned.remove(best_client)
        if best_route_idx == -1:
            state.routes.append([best_client])
            loads.append(data.demands[best_client])
            deps.append(_compute_departures([best_client], data))
        else:
            state.routes[best_route_idx].insert(best_pos, best_client)
            loads[best_route_idx] += data.demands[best_client]
            deps[best_route_idx] = _compute_departures(
                state.routes[best_route_idx], data)

    return state


class ALNSSolver:
    def __init__(
        self,
        time_limit: int = 60,
        seed: int = 0,
        vehicle_capacity: int = 200,
        num_vehicles: int = 25,
        enable_fairness: bool = False,
        fairness_weight: float = 100.0,
        max_iterations: int = 25000,
        sa_start_temp: float = 100.0,
        sa_end_temp: float = 0.5,
    ):
        self.time_limit = time_limit
        self.seed = seed
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.enable_fairness = enable_fairness
        self.fairness_weight = fairness_weight
        self.max_iterations = max_iterations
        self.sa_start_temp = sa_start_temp
        self.sa_end_temp = sa_end_temp

    def solve(self, instance_name: str | VRPInstanceInput) -> dict:
        data = _parse_instance(
            instance_name,
            self.vehicle_capacity,
            self.num_vehicles,
            self.enable_fairness,
            self.fairness_weight,
        )

        initial = _build_initial_solution(data)

    
        init_obj = initial._total_distance() or 1.0
        scale = init_obj / 10_000.0 
        start_temp = max(self.sa_start_temp, self.sa_start_temp * scale)
        end_temp = max(self.sa_end_temp, self.sa_end_temp * scale)

        rng = np.random.default_rng(self.seed)
        alns = ALNS(rng)

        alns.add_destroy_operator(random_removal, name="random_removal")
        alns.add_destroy_operator(worst_removal, name="worst_removal")
        alns.add_destroy_operator(shaw_removal, name="shaw_removal")
        alns.add_repair_operator(greedy_repair, name="greedy_repair")
        alns.add_repair_operator(regret_2_repair, name="regret_2_repair")

        sa_step = (end_temp / start_temp) ** (1.0 / self.max_iterations)

        select = RouletteWheel(
            scores=[5, 2, 1, 0.5],
            decay=0.8,
            num_destroy=3,
            num_repair=2,
        )

        accept = SimulatedAnnealing(
            start_temperature=start_temp,
            end_temperature=end_temp,
            step=sa_step,
        )

        stop = MaxRuntime(self.time_limit)

        result = alns.iterate(initial, select, accept, stop)
        best = result.best_state

        is_feasible = len(best.unassigned) == 0
        total_dist = best._total_distance() if is_feasible else float("inf")
        fairness = self._fairness_report(best) if is_feasible else None

        return {
            "routes": [list(r) for r in best.routes],
            "total_distance": total_dist,
            "num_routes": len(best.routes),
            "feasible": is_feasible,
            "fairness": fairness,
            "fairness_before": None,
            "rebalance_moves": 0,
            "cost_delta_pct": 0.0,
        }

    def _fairness_report(self, state: VRPState) -> FairnessReport:
        data = state.data
        dm = data.dm
        distances, loads, counts = [], [], []

        for route in state.routes:
            d = 0
            prev = 0
            for c in route:
                d += dm[prev][c]
                prev = c
            d += dm[prev][0]
            distances.append(float(d))
            loads.append(float(sum(data.demands[c] for c in route)))
            counts.append(len(route))

        return compute_fairness(
            route_distances=distances,
            route_loads=loads,
            route_clients=counts,
            route_durations=distances,
        )