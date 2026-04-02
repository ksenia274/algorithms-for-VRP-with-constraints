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
