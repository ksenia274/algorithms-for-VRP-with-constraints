from __future__ import annotations

import pytest
from pathlib import Path

from algorithms.algorithm_params import HgsSimpleParams
from algorithms.solver_result import SolverConfig
from data.instance_resolver import resolve_yandex_path
from runtime.cli.run_command import cmd_run
from runtime.serialization import load_config_yaml, save_config_yaml


def _find_yandex_instance_name() -> str | None:
    for name in ("3", "0", "1"):
        if resolve_yandex_path(name) is not None:
            return name
    return None


def _write_simple_config(path: Path, instance: str, time_limit: int = 3, seed: int = 42) -> None:
    config = SolverConfig(
        schema_version="1.0",
        algorithm="hgs_simple",
        instance=instance,
        instance_kind="yandex",
        time_limit=time_limit,
        seed=seed,
        capacity=200,
        num_vehicles=25,
        algorithm_params=HgsSimpleParams(),
    )
    save_config_yaml(config, path)


def test_cmd_run_creates_run_dir(tmp_results_dir):
    inst = _find_yandex_instance_name()
    if inst is None:
        pytest.skip("No yandex instances found on disk")

    cfg_path = tmp_results_dir / "config.yaml"
    _write_simple_config(cfg_path, inst)

    run_dir = cmd_run(cfg_path, output=tmp_results_dir / "my_run")

    assert run_dir.is_dir()
    assert (run_dir / "result.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "config.yaml").exists()


def test_cmd_run_overrides(tmp_results_dir):
    inst = _find_yandex_instance_name()
    if inst is None:
        pytest.skip("No yandex instances found on disk")

    cfg_path = tmp_results_dir / "config.yaml"
    _write_simple_config(cfg_path, inst, seed=42)

    run_dir = cmd_run(cfg_path, seed=99, output=tmp_results_dir / "override_run")

    saved_config = load_config_yaml(run_dir / "config.yaml")
    assert saved_config.seed == 99


def test_cmd_run_invalid_config(tmp_results_dir):
    bad_cfg = tmp_results_dir / "bad.yaml"
    bad_cfg.write_text("algorithm: hgs_simple\nalgorithm_params:\n  unknown_key: 123\n")

    with pytest.raises(Exception):
        cmd_run(bad_cfg, output=tmp_results_dir / "bad_run")
