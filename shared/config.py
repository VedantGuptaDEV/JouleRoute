"""Load config.yaml from the repo root."""
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("seed", "models", "server", "router"):
        if key not in cfg:
            raise ValueError(f"config.yaml is missing required section '{key}'")
    return cfg


def repo_path(rel: str) -> Path:
    """Resolve a path from config.yaml relative to the repo root."""
    return REPO_ROOT / rel
