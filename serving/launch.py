"""Start one llama-server per tier (ports from config.yaml), wait for health, stop on Ctrl-C.

FULL is started with LITE's GGUF as its draft model for speculative decoding
(--spec-draft-model, checked against `llama-server --help` for build b11840) unless
--no-spec is given or server.speculative is false. Every tier gets the same context,
thread and batch settings.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import load_config, repo_path  # noqa: E402
from shared.tiers import TIER_KEYS  # noqa: E402


def server_cmd(cfg, tier: str, speculative: bool, server_bin: Path) -> list[str]:
    srv = cfg["server"]
    models_dir = repo_path(srv["models_dir"])
    model = models_dir / cfg["models"][tier]["file"]
    if not model.exists():
        sys.exit(f"missing model file {model}; download it first")
    cmd = [
        str(server_bin),
        "-m", str(model), "--port", str(cfg["models"][tier]["port"]), "--host", "127.0.0.1",
        "-c", str(srv["ctx_size"]), "-np", "1",
        "-t", str(srv["threads"]), "-b", str(srv["batch_size"]),
        "--jinja", "--seed", str(cfg["seed"]),
    ]
    if tier == "full" and speculative:
        cmd += ["--spec-draft-model", str(models_dir / cfg["models"]["lite"]["file"])]
    return cmd


def wait_healthy(proc: subprocess.Popen, port: int, tier: str, log_path: Path, timeout_s: int = 300):
    for _ in range(timeout_s):
        if proc.poll() is not None:
            sys.exit(f"llama-server for {tier} exited with code {proc.returncode}; see {log_path}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    proc.kill()
    sys.exit(f"llama-server for {tier} did not become healthy in {timeout_s}s; see {log_path}")


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiers", nargs="+", choices=TIER_KEYS, default=TIER_KEYS)
    parser.add_argument("--no-spec", action="store_true", help="start FULL without speculative decoding")
    parser.add_argument("--log-dir", type=Path, default=repo_path("data/logs"))
    parser.add_argument("--llama-server", type=Path, default=repo_path(cfg["server"]["llama_server_bin"]),
                        help="llama-server binary (e.g. the CUDA build from TRAINING.md)")
    args = parser.parse_args()
    speculative = cfg["server"]["speculative"] and not args.no_spec
    args.log_dir.mkdir(parents=True, exist_ok=True)
    server_bin = args.llama_server.resolve()
    # Release tarballs ship their shared libraries (CUDA runtime, ggml backends) beside the binary.
    env = {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(
        p for p in [str(server_bin.parent), os.environ.get("LD_LIBRARY_PATH", "")] if p)}

    procs = {}
    try:
        for tier in args.tiers:
            log_path = args.log_dir / f"llama-{tier}.log"
            cmd = server_cmd(cfg, tier, speculative, server_bin)
            print(f"starting {tier} on :{cfg['models'][tier]['port']} (log: {log_path})", flush=True)
            procs[tier] = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env)
            wait_healthy(procs[tier], cfg["models"][tier]["port"], tier, log_path)
        spec = "on" if speculative and "full" in procs else "off"
        ports = ", ".join(f"{t}:{cfg['models'][t]['port']}" for t in procs)
        print(f"\nall healthy: {ports} (FULL speculative decoding {spec}). Ctrl-C to stop.", flush=True)
        while all(p.poll() is None for p in procs.values()):
            time.sleep(1)
        dead = [t for t, p in procs.items() if p.poll() is not None]
        print(f"llama-server exited unexpectedly: {', '.join(dead)}")
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            p.wait(timeout=30)


if __name__ == "__main__":
    main()
