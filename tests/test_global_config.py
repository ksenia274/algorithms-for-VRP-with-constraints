import pytest
from runtime.global_config import (
    GlobalConfig,
    MetricsConfig,
    PathsConfig,
    DefaultsConfig,
    get_global_config,
    set_global_config_for_testing,
    reset_global_config,
)


@pytest.fixture(autouse=True)
def reset_after():
    yield
    reset_global_config()


def test_singleton_caching(tmp_results_dir):
    cfg1 = get_global_config()
    cfg2 = get_global_config()
    assert cfg1 is cfg2


def test_set_for_testing_overrides():
    custom = GlobalConfig(
        metrics=MetricsConfig(primary="load_gini"),
        paths=PathsConfig(),
        defaults=DefaultsConfig(),
    )
    set_global_config_for_testing(custom)
    assert get_global_config().metrics.primary == "load_gini"


def test_reset_clears_cache(tmp_results_dir):
    cfg1 = get_global_config()
    reset_global_config()
    cfg2 = get_global_config()
    assert cfg1 is not cfg2
