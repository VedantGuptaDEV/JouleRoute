"""
serving/launch.py
Start, health-check, and stop the three llama-server processes
(LITE :8081, MID :8082, FULL :8083).

Usage:
    python serving/launch.py [--no-spec] [--config config.yaml]

Flags:
    --no-spec   Start FULL without speculative decoding (for benchmarking).
    --config    Path to config.yaml (default: config.yaml in repo root).

The script blocks until you press Ctrl-C, then shuts down all servers cleanly.
"""
from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

# Ensure repo root is on sys.path when run directly
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.config import load_config
from shared.tiers import Tier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("serving.launch")

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_processes: dict[Tier, subprocess.Popen] = {}


# ---------------------------------------------------------------------------
# Build command helpers
# ---------------------------------------------------------------------------

def _base_cmd(exe: str, model_path: str, port: int, cfg: dict) -> list[str]:
    """Return the common flags shared by all three tiers."""
    srv = cfg["server"]
    cmd = [
        exe,
        "--model",        str(Path(model_path).resolve()),
        "--host",         cfg["gateway"]["host"],
        "--port",         str(port),
        "--ctx-size",     str(srv["ctx_size"]),
        "--batch-size",   str(srv["batch_size"]),
        "--ubatch-size",  str(srv["ubatch_size"]),
        "--seed",         str(cfg["seed"]),
        "--reasoning",    "off",   # thinking mode off for all tiers
        "--no-webui",
        "--log-disable",           # keep stdout clean; we log health ourselves
    ]
    threads = srv.get("threads", -1)
    if threads and threads > 0:
        cmd += ["--threads", str(threads)]
    return cmd


def _lite_cmd(exe: str, cfg: dict) -> list[str]:
    m = cfg["models"]["lite"]
    return _base_cmd(exe, m["path"], m["port"], cfg)


def _mid_cmd(exe: str, cfg: dict) -> list[str]:
    m = cfg["models"]["mid"]
    return _base_cmd(exe, m["path"], m["port"], cfg)


def _full_cmd(exe: str, cfg: dict, no_spec: bool = False, force_spec: bool = False) -> list[str]:
    m_full = cfg["models"]["full"]
    m_lite = cfg["models"]["lite"]
    srv = cfg["server"]
    cmd = _base_cmd(exe, m_full["path"], m_full["port"], cfg)

    # Resolution order: --spec (force_spec) > --no-spec (no_spec) > config default
    if force_spec:
        spec_on = True
    elif no_spec:
        spec_on = False
    else:
        spec_on = srv.get("speculative", False)

    if spec_on:
        draft_path = str(Path(m_lite["path"]).resolve())
        cmd += [
            "--spec-draft-model", draft_path,
            "--spec-type",        "draft-simple",
            "--spec-draft-n-max", str(srv.get("spec_draft_n_max", 8)),
        ]
        log.info("FULL: speculative decoding: ON (draft=%s)", draft_path)
    else:
        log.info("FULL: speculative decoding: OFF")

    return cmd


# ---------------------------------------------------------------------------
# Process lifecycle
# ---------------------------------------------------------------------------

def start_server(tier: Tier, cmd: list[str]) -> subprocess.Popen:
    """Spawn a llama-server process and return the Popen handle."""
    log.info("Starting %s: %s", tier.name, " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    _processes[tier] = proc
    return proc


def wait_for_health(tier: Tier, host: str, port: int, timeout_s: int = 120) -> None:
    """Poll GET /health until 200 or timeout."""
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout_s
    last_err: Optional[str] = None

    log.info("Waiting for %s health at %s (timeout %ds) ...", tier.name, url, timeout_s)
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                log.info("%s is healthy (port %d).", tier.name, port)
                return
            last_err = f"HTTP {r.status_code}"
        except httpx.RequestError as exc:
            last_err = str(exc)
        time.sleep(1.0)

    raise RuntimeError(
        f"{tier.name} did not become healthy within {timeout_s}s "
        f"(last error: {last_err})"
    )


def stop_all() -> None:
    """Terminate all running server processes."""
    for tier, proc in list(_processes.items()):
        if proc.poll() is None:
            log.info("Stopping %s (pid %d) ...", tier.name, proc.pid)
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(proc.pid)],
                    capture_output=True,
                )
            else:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log.warning("%s did not stop in time; killing.", tier.name)
                proc.kill()
        else:
            log.info("%s already stopped.", tier.name)
    _processes.clear()


# ---------------------------------------------------------------------------
# Public API used by gateway / tests
# ---------------------------------------------------------------------------

def start_all(cfg: dict, no_spec: bool = False, force_spec: bool = False) -> None:
    """Start LITE, MID, FULL and wait for all health checks."""
    exe = cfg["llama_server_exe"]
    host = cfg["gateway"]["host"]
    timeout = cfg["server"].get("health_timeout_s", 120)

    _verify_model_files(cfg)

    tiers_cmds = [
        (Tier.LITE, _lite_cmd(exe, cfg)),
        (Tier.MID,  _mid_cmd(exe, cfg)),
        (Tier.FULL, _full_cmd(exe, cfg, no_spec=no_spec, force_spec=force_spec)),
    ]

    for tier, cmd in tiers_cmds:
        start_server(tier, cmd)

    # Health-check in order
    for tier, _ in tiers_cmds:
        port = cfg["models"][tier.name.lower()]["port"]
        wait_for_health(tier, host, port, timeout_s=timeout)

    log.info(
        "All servers ready — LITE:%d  MID:%d  FULL:%d",
        cfg["models"]["lite"]["port"],
        cfg["models"]["mid"]["port"],
        cfg["models"]["full"]["port"],
    )


def _verify_model_files(cfg: dict) -> None:
    """Raise FileNotFoundError if any GGUF file is missing."""
    for tier_name in ("lite", "mid", "full"):
        p = Path(cfg["models"][tier_name]["path"])
        if not p.exists():
            raise FileNotFoundError(
                f"Model file for {tier_name.upper()} not found: {p}\n"
                f"Run:  python serving/download_models.py"
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the three llama-server processes for JouleRoute.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    spec_group = parser.add_mutually_exclusive_group()
    spec_group.add_argument(
        "--spec",
        action="store_true",
        help="Force speculative decoding ON for FULL, overriding config.",
    )
    spec_group.add_argument(
        "--no-spec",
        action="store_true",
        help="Force speculative decoding OFF for FULL, overriding config.",
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config.yaml"),
        help="Path to config.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    # Register cleanup on Ctrl-C / SIGTERM
    def _shutdown(signum, frame):  # noqa: ANN001
        log.info("Signal %s received, shutting down ...", signum)
        stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    try:
        start_all(cfg, no_spec=args.no_spec, force_spec=args.spec)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        stop_all()
        sys.exit(1)

    log.info("Servers running. Press Ctrl-C to stop.")
    while True:
        for tier, proc in list(_processes.items()):
            if proc.poll() is not None:
                log.error(
                    "%s exited unexpectedly (returncode=%d). Shutting down all.",
                    tier.name,
                    proc.returncode,
                )
                stop_all()
                sys.exit(1)
        time.sleep(2.0)


if __name__ == "__main__":
    main()
