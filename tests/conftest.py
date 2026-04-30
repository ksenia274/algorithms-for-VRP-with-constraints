import pytest
from pathlib import Path

from runtime.global_config import GlobalConfig, MetricsConfig, PathsConfig, DefaultsConfig, set_global_config_for_testing, reset_global_config


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Configures global_config to use tmp_path as results root and resets after test."""
    cfg = GlobalConfig(
        metrics=MetricsConfig(primary="dist_worst_ratio", secondary=["dist_gini"]),
        paths=PathsConfig(
            results=str(tmp_path / "results"),
            archive=str(tmp_path / "results" / "archive"),
            data_yandex=str(tmp_path / "data" / "yandex"),
            data_solomon=str(tmp_path / "data" / "solomon"),
        ),
        defaults=DefaultsConfig(time_limit=30, seed=42, capacity=200, num_vehicles=25),
    )
    set_global_config_for_testing(cfg)
    yield tmp_path
    reset_global_config()
