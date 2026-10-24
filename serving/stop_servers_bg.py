"""
serving/stop_servers_bg.py
Stop the background llama-server processes started by start_servers_bg.py.

On Windows, CTRL_BREAK_EVENT does not work on DETACHED_PROCESS children
(raises OSError WinError 87).  This script uses:
  - Windows: taskkill /F /PID <pid>   (always works)
  - Other:   SIGTERM, falling back to SIGKILL after 10 s

Usage:
    python serving/stop_servers_bg.py
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

_PID_FILE = Path(__file__).parent / ".server_pids.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stop_servers_bg")


def _kill_windows(tier_name: str, pid: int) -> None:
    """Kill a process on Windows using taskkill /F /PID."""
    result = subprocess.run(
        ["taskkill", "/F", "/PID", str(pid)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        log.info("Stopped %s (pid=%d).", tier_name, pid)
    else:
        # returncode 128 = process not found (already dead)
        stderr = result.stderr.strip()
        if "not found" in stderr.lower() or result.returncode == 128:
            log.info("%s (pid=%d) was already stopped.", tier_name, pid)
        else:
            log.warning(
                "taskkill for %s (pid=%d) returned %d: %s",
                tier_name, pid, result.returncode, stderr,
            )


def _kill_posix(tier_name: str, pid: int) -> None:
    """Send SIGTERM on POSIX, force-kill after 10 s if still alive."""
    import os
    import signal as _signal

    try:
        os.kill(pid, _signal.SIGTERM)
        log.info("Sent SIGTERM to %s (pid=%d).", tier_name, pid)
    except ProcessLookupError:
        log.info("%s (pid=%d) was already stopped.", tier_name, pid)
        return
    except OSError as exc:
        log.warning("Could not signal %s pid=%d: %s", tier_name, pid, exc)
        return

    # Wait up to 10 s then SIGKILL
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        time.sleep(0.5)
        try:
            os.kill(pid, 0)   # probe: raises if dead
        except ProcessLookupError:
            log.info("%s (pid=%d) stopped cleanly.", tier_name, pid)
            return

    log.warning("%s (pid=%d) did not stop; sending SIGKILL.", tier_name, pid)
    try:
        os.kill(pid, _signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> None:
    if not _PID_FILE.exists():
        log.warning("No PID file at %s — servers may not be running.", _PID_FILE)
        return

    pids: dict[str, int] = json.loads(_PID_FILE.read_text())
    log.info("Stopping %d server(s): %s", len(pids), pids)

    for tier_name, pid in pids.items():
        if sys.platform == "win32":
            _kill_windows(tier_name, pid)
        else:
            _kill_posix(tier_name, pid)

    _PID_FILE.unlink(missing_ok=True)
    log.info("PID file removed. Done.")


if __name__ == "__main__":
    main()
