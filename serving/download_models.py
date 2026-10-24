"""
serving/download_models.py
Download the three GGUF model files into models/ using huggingface_hub.

Usage:
    python serving/download_models.py [--config config.yaml] [--tier lite|mid|full]

Requires:
    pip install huggingface_hub>=0.23.0
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("serving.download_models")


def download_tier(tier_name: str, cfg: dict, models_dir: Path) -> Path:
    """Download a single GGUF file and return its local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.error(
            "huggingface_hub is not installed.\n"
            "Run: pip install 'huggingface_hub>=0.23.0'"
        )
        sys.exit(1)

    m = cfg["models"][tier_name]
    repo_id: str = m["repo"]
    filename: str = m["file"]
    dest: Path = models_dir / filename

    if dest.exists():
        log.info("[%s] Already exists: %s (skipping download)", tier_name.upper(), dest)
        return dest

    log.info(
        "[%s] Downloading %s from %s ...",
        tier_name.upper(), filename, repo_id,
    )
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(models_dir),
        local_dir_use_symlinks=False,
    )
    dest_actual = Path(local_path)
    log.info("[%s] Saved to %s", tier_name.upper(), dest_actual)
    return dest_actual


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GGUF model files for JouleRoute.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config.yaml"),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--tier",
        choices=["lite", "mid", "full", "all"],
        default="all",
        help="Which tier to download (default: all).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    models_dir = _REPO_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    tiers = ["lite", "mid", "full"] if args.tier == "all" else [args.tier]

    results: dict[str, Path] = {}
    for tier_name in tiers:
        results[tier_name] = download_tier(tier_name, cfg, models_dir)

    print("\n── Download summary ──────────────────────────────────")
    for tier_name, path in results.items():
        size_mb = path.stat().st_size / 1_048_576
        print(f"  {tier_name.upper():4s}  {size_mb:7.0f} MB  {path}")
    print("─────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
