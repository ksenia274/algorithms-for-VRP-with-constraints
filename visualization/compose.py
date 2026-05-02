from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from visualization.plots.category_heatmap import plot_category_heatmap
from visualization.plots.dimensions import plot_dimensions
from visualization.plots.distribution import plot_distribution
from visualization.plots.pareto import plot_pareto


def plot_all(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    group_col: str = "algorithm",
    primary_metric: Optional[str] = None,
) -> dict[str, Path]:
    """Generate all 4 fairness charts. Returns mapping of chart name to output path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "01_pareto":            output_dir / "01_pareto.png",
        "02_dimensions":        output_dir / "02_dimensions.png",
        "03_distribution":      output_dir / "03_distribution.png",
        "04_category_heatmap":  output_dir / "04_category_heatmap.png",
    }

    plot_pareto(df, paths["01_pareto"], primary_metric=primary_metric, group_col=group_col)
    plot_dimensions(df, paths["02_dimensions"], primary_metric=primary_metric, group_col=group_col)
    plot_distribution(df, paths["03_distribution"], primary_metric=primary_metric, group_col=group_col)
    plot_category_heatmap(df, paths["04_category_heatmap"], primary_metric=primary_metric, group_col=group_col)

    return paths
