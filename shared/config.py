"""
shared/config.py
Load and validate config.yaml.  Returns a plain dict so callers need no
extra imports.  Call load_config() once at startup and pass the result down.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load and return the config as a plain dict (cached after first call)."""
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    """Raise ValueError for obviously missing required keys."""
    required = [
        "llama_server_exe",
        "models",
        "server",
        "gateway",
        "telemetry",
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f"config.yaml is missing required key: '{key}'")

    exe = cfg["llama_server_exe"]
    if not Path(exe).exists():
        raise FileNotFoundError(
            f"llama_server_exe not found: {exe}\n"
            "Update llama_server_exe in config.yaml."
        )

    for tier in ("lite", "mid", "full"):
        if tier not in cfg["models"]:
            raise ValueError(f"config.yaml models section is missing tier: '{tier}'")
