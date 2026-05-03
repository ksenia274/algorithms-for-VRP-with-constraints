from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "global.yaml"
_cached: Optional["GlobalConfig"] = None


class MetricsConfig(BaseModel):
    primary: str
    secondary: list[str] = []


class PathsConfig(BaseModel):
    results: str = "results"
    archive: str = "results/archive"
    data_yandex: str = "data/yandex"
    data_yandex_external: str = "../Fair_VRP_ala_Yandex/vrp_problems"
    data_solomon: str = "data/solomon"


class DefaultsConfig(BaseModel):
    time_limit: int = 30
    seed: int = 42
    capacity: int = 200
    num_vehicles: int = 25


class GlobalConfig(BaseModel):
    metrics: MetricsConfig
    paths: PathsConfig = PathsConfig()
    defaults: DefaultsConfig = DefaultsConfig()


def get_global_config() -> GlobalConfig:
    """Return cached GlobalConfig loaded from configs/global.yaml."""
    global _cached
    if _cached is None:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _cached = GlobalConfig.model_validate(data)
    return _cached


def set_global_config_for_testing(cfg: GlobalConfig) -> None:
    """Override the singleton for test isolation."""
    global _cached
    _cached = cfg


def reset_global_config() -> None:
    """Clear the cached singleton so the next call reloads from disk."""
    global _cached
    _cached = None
