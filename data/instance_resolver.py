from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_yandex_path(name: str) -> Optional[str]:
    """Return absolute path to a Yandex JSON instance file, or None if not found.

    Searches data_yandex_external first, then data_yandex from global_config.
    Instance name '03' resolves to file '3.json' (leading zeros stripped).
    """
    from runtime.global_config import get_global_config
    cfg = get_global_config()

    try:
        idx = str(int(name))
    except ValueError:
        idx = name

    candidates = [
        Path(cfg.paths.data_yandex_external) / f"{idx}.json",
        Path(cfg.paths.data_yandex) / f"{idx}.json",
        Path(cfg.paths.data_yandex) / f"{name}.json",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def resolve_instance(name: str, kind: str) -> str:
    """Resolve instance name + kind to a path or Solomon name.

    Raises FileNotFoundError for yandex instances not found on disk.
    """
    if kind == "yandex":
        path = resolve_yandex_path(name)
        if path is None:
            raise FileNotFoundError(
                f"Yandex instance '{name}' not found. "
                f"Searched data_yandex and data_yandex_external paths."
            )
        return path
    return name
