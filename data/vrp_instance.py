from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class VRPInstanceInput:
    df: pd.DataFrame
    dist_matrix: list[list[int]] | None = None
    time_matrix: list[list[int]] | None = None
    recommended_capacity: int | None = None
    coordinates: list[tuple[float, float]] | None = None


def load_instance_input(instance: str | VRPInstanceInput) -> VRPInstanceInput:
    if isinstance(instance, VRPInstanceInput):
        return instance
    if instance.endswith(".json"):
        from data.load_yandex_instance import load_yandex_instance
        return load_yandex_instance(instance)
    if instance.endswith(".csv") or ("/" in instance) or ("\\" in instance):
        return VRPInstanceInput(df=pd.read_csv(instance))
    from data.load_solomon import load_instance
    return VRPInstanceInput(df=load_instance(instance))
