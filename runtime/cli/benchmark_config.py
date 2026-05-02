from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InstanceSpec(BaseModel):
    name: str
    kind: str  # "yandex" | "solomon"


class AlgorithmSpec(BaseModel):
    name: str
    type: str
    algorithm_params: dict[str, Any] = Field(default_factory=dict)


class SharedParams(BaseModel):
    time_limit: int = 30
    seed: int = 42
    capacity: int = 200
    num_vehicles: int = 25


class BenchmarkConfig(BaseModel):
    name: str
    description: str = ""
    instances: list[InstanceSpec]
    algorithms: list[AlgorithmSpec]
    shared: SharedParams = Field(default_factory=SharedParams)
