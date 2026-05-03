import math
import time
from typing import Protocol
from dataclasses import dataclass

import concurrent.futures

from data.vrp_instance import VRPInstanceInput


@dataclass(frozen=True)
class _Point:
    x: float
    y: float

    def dominates(self, other: _Point) -> bool:
        return self != other and (self.x <= other.x and self.y <= other.y)


@dataclass(frozen=True)
class _Rectangle:
    z1: _Point
    z2: _Point

    def area(self) -> float:
        return abs((self.z1.x - self.z2.x) * (self.z1.y - self.z2.y))


@dataclass
class _OptimizationResult[T]:
    point: _Point
    is_feasible: bool
    payload: T


@dataclass
class GenericSolution[T]:
    obj1: float
    obj2: float
    is_feasible: bool
    payload: T


class ConstrainedSolver[T](Protocol):
    def optimize(self, instance: str | VRPInstanceInput, max_obj2: float) -> GenericSolution[T]:
        """Minimizes obj1 subject to obj2 <= max_obj2"""
        ...

class RSSolver[T]:
    def __init__(
        self,
        solver: ConstrainedSolver[T],
        time_limit: int = 60,
        max_workers: int = 1,
    ):
        self.solver = solver
        self.time_limit = time_limit
        self.max_workers = max_workers

        self.points_history = []
        self.final_rectangles = set()
        self.pareto_set = []

    def solve(
        self,
        instance_path: str | VRPInstanceInput,
        max_obj1: float = 10000,
        min_obj2: float = 0,
    ) -> list[T]:
        print(f"Running on {self.max_workers} threads")

        start_time = time.perf_counter()
        generation = 0

        optimization_result = self._optimize_with_constraint(
            instance_path, max_obj2=math.inf
        )
        x, is_feasible = optimization_result.point, optimization_result.is_feasible

        if not is_feasible: 
            return []
        
        self.points_history.append((x, generation))

        pareto_set = [optimization_result]
        rectangles = {_Rectangle(x, _Point(max_obj1, min_obj2))}

        running_tasks: dict[concurrent.futures.Future, _Rectangle] = {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            while time.perf_counter() - start_time < self.time_limit:
                while len(running_tasks) < self.max_workers and rectangles:
                    max_rectangle = max(rectangles, key=lambda r: r.area())
                    rectangles.remove(max_rectangle)

                    y1 = max_rectangle.z1
                    y2 = max_rectangle.z2
                    c = 0.5 * (y1.y + y2.y)

                    future = executor.submit(self._optimize_with_constraint, instance_path, c)
                    running_tasks[future] = max_rectangle
                
                if not running_tasks:
                    break

                timeout = self.time_limit - (time.perf_counter() - start_time)
                done, _ = concurrent.futures.wait(
                    running_tasks.keys(),
                    return_when=concurrent.futures.ALL_COMPLETED,
                    timeout=timeout if timeout > 0 else 0
                )

                generation += 1
                for fut in done:
                    max_rectangle = running_tasks.pop(fut)
                    rectangles.add(max_rectangle)
                    optimization_result = fut.result()

                    y1 = max_rectangle.z1
                    y2 = max_rectangle.z2
                    c = 0.5 * (y1.y + y2.y)

                    x, is_feasible = optimization_result.point, optimization_result.is_feasible
                    is_dominated = any(p.point.dominates(x) or p.point == x for p in pareto_set)

                    if is_feasible:
                        self.points_history.append((x, generation))

                    if not is_feasible or is_dominated:
                        rectangles.remove(max_rectangle)
                        rectangles.add(_Rectangle(y1, _Point(y2.x, c)))
                    else:
                        pareto_set = [p for p in pareto_set if not x.dominates(p.point)]
                        pareto_set.append(optimization_result)
                        r1 = None
                        for rect in rectangles:
                            if rect.z1.x < x.x <= rect.z2.x:
                                r1 = rect
                                break

                        r2 = None
                        for rect in rectangles:
                            if rect.z1.y >= x.y >= rect.z2.y:
                                r2 = rect
                                break

                        rectangles_to_remove = [
                            rect for rect in rectangles if x.dominates(rect.z1)
                        ]
                        rectangles.difference_update(rectangles_to_remove)

                        if r1 is not None:
                            rectangles.discard(r1)
                            rectangles.add(_Rectangle(r1.z1, _Point(x.x, max(r1.z2.y, c))))

                        if r2 is not None:
                            rectangles.discard(r2)
                            rectangles.add(_Rectangle(_Point(max(x.x, r2.z1.x), x.y), r2.z2))

        for fut in running_tasks:
            fut.cancel()

        self.final_rectangles = rectangles
        self.pareto_set = pareto_set

        return list(map(lambda result: result.payload, pareto_set))

    def _optimize_with_constraint(
        self, instance_path: str | VRPInstanceInput, max_obj2
    ) -> _OptimizationResult[T]:
        result = self.solver.optimize(instance_path, max_obj2=max_obj2)
        point = _Point(result.obj1, result.obj2)
        is_feasible = result.is_feasible
        return _OptimizationResult(
            point=point, is_feasible=is_feasible, payload=result.payload
        )
