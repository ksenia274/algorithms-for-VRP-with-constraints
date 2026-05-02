from __future__ import annotations

import re

import pandas as pd

_SOLOMON_PATTERN = re.compile(r"^(RC|R|C)([12])", re.IGNORECASE)


def _categorize(instance: str, instance_kind: str) -> str:
    """Return display category: 'yandex', 'R1', 'R2', 'C1', 'C2', 'RC1', 'RC2'."""
    if instance_kind != "solomon":
        return instance_kind
    m = _SOLOMON_PATTERN.match(instance)
    if m:
        return m.group(1).upper() + m.group(2)
    return instance[:2].upper()


def add_category_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'category' column derived from instance and instance_kind."""
    df = df.copy()
    df["category"] = df.apply(
        lambda r: _categorize(str(r["instance"]), str(r.get("instance_kind", "yandex"))),
        axis=1,
    )
    return df


def validate_metrics_csv(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError with clear message if any required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. CSV may use old schema (pre-refactor)."
        )
