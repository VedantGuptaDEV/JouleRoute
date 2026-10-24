"""
serving/start_servers_bg.py
Start the three llama-servers as background processes (Windows-friendly).
Does NOT block — prints PIDs and exits once all servers are healthy.
Use serving/stop_servers_bg.py to stop them later.

Usage:
    python serving/start_servers_bg.py [--no-spec] [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.config import load_config
from shared.tiers import Tier
from serving.launch import (
    _lite_cmd, _mid_cmd, _full_cmd,
    _verify_model_files, wait_for_health,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("start_servers_bg")

_PID_FILE = _REPO_ROOT / "serving" / ".server_pids.json"


def start_background(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
        ) if sys.platform == "win32" else 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start all three llama-servers in the background.",
    )
    parser.add_argument("--no-spec", action="store_true")
    parser.add_argument("--config", default=str(_REPO_ROOT / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    _verify_model_files(cfg)

    exe = cfg["llama_server_exe"]
    host = cfg["gateway"]["host"]
    timeout = cfg["server"].get("health_timeout_s", 120)

    cmds = [
        (Tier.LITE, _lite_cmd(exe, cfg)),
        (Tier.MID,  _mid_cmd(exe, cfg)),
        (Tier.FULL, _full_cmd(exe, cfg, no_spec=args.no_spec)),
    ]

    pids: dict[str, int] = {}
    for tier, cmd in cmds:
        log.info("Spawning %s ...", tier.name)
        proc = start_background(cmd)
        pids[tier.name] = proc.pid
        log.info("%s  pid=%d", tier.name, proc.pid)

    # Save PIDs so stop script can find them
    _PID_FILE.write_text(json.dumps(pids, indent=2))
    log.info("PIDs saved to %s", _PID_FILE)

    # Health checks
    for tier, _ in cmds:
        port = cfg["models"][tier.name.lower()]["port"]
        wait_for_health(tier, host, port, timeout_s=timeout)

    log.info(
        "All servers healthy — LITE:%d  MID:%d  FULL:%d",
        cfg["models"]["lite"]["port"],
        cfg["models"]["mid"]["port"],
        cfg["models"]["full"]["port"],
    )
    log.info("PIDs: %s", pids)


if __name__ == "__main__":
    main()
