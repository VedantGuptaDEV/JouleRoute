"""Download the GGUF file for each tier into models/ (repos and filenames from config.yaml).

Files that are already present are skipped.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import load_config, repo_path  # noqa: E402
from shared.tiers import TIER_KEYS  # noqa: E402


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiers", nargs="+", choices=TIER_KEYS, default=TIER_KEYS)
    parser.add_argument("--models-dir", type=Path, default=repo_path(cfg["server"]["models_dir"]))
    args = parser.parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)

    for tier in args.tiers:
        m = cfg["models"][tier]
        path = args.models_dir / m["file"]
        if path.exists():
            print(f"{tier}: {path.name} already present")
            continue
        print(f"{tier}: downloading {m['repo']}/{m['file']}")
        hf_hub_download(m["repo"], m["file"], local_dir=args.models_dir)

    print("\nmodels:")
    for tier in args.tiers:
        path = args.models_dir / cfg["models"][tier]["file"]
        print(f"  {tier:<5} {path.name}  {path.stat().st_size / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
