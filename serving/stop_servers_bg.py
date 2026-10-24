"""
serving/stop_servers_bg.py
Stop the background llama-server processes started by start_servers_bg.py.

Usage:
    python serving/stop_servers_bg.py
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_PID_FILE = Path(__file__).parent / ".server_pids.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stop_servers_bg")


def main() -> None:
    if not _PID_FILE.exists():
        log.warning("No PID file found at %s — servers may not be running.", _PID_FILE)
        return

    pids: dict[str, int] = json.loads(_PID_FILE.read_text())
    for tier_name, pid in pids.items():
        try:
            if sys.platform == "win32":
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            else:
                os.kill(pid, signal.SIGTERM)
            log.info("Sent stop signal to %s (pid=%d).", tier_name, pid)
        except OSError as exc:
            log.warning("Could not stop %s pid=%d: %s", tier_name, pid, exc)

    _PID_FILE.unlink(missing_ok=True)
    log.info("Done.")


if __name__ == "__main__":
    main()
