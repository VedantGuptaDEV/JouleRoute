"""SQLite telemetry writer (Readme §5.I). One row per request, including failures.

Query text is never stored when the PII flag is set, whatever the config says.
"""
import sqlite3
import threading
import time
from pathlib import Path

COLUMNS = [
    ("ts", "REAL"), ("query_hash", "TEXT"), ("query_text", "TEXT"), ("category_guess", "TEXT"),
    ("source", "TEXT"), ("pii_flag", "INTEGER"), ("predicted_tier", "TEXT"), ("tier", "TEXT"),
    ("confidence", "REAL"), ("fallback", "INTEGER"), ("escalated", "INTEGER"), ("tau", "REAL"),
    ("mode", "TEXT"), ("router_ms", "REAL"), ("ttft_ms", "REAL"), ("total_ms", "REAL"),
    ("tokens_in", "INTEGER"), ("tokens_out", "INTEGER"), ("joules", "REAL"), ("joules_est_full", "REAL"),
    ("battery_pct_real", "REAL"), ("battery_pct_sim", "REAL"), ("energy_backend", "TEXT"),
    ("is_estimate", "INTEGER"), ("error", "TEXT"),
]
NAMES = [name for name, _ in COLUMNS]


class Telemetry:
    def __init__(self, db_path: Path, store_query_text: bool):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_query_text = store_query_text
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        cols = ", ".join(f"{n} {t}" for n, t in COLUMNS)
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
        self._conn.commit()

    def write(self, row: dict) -> int:
        row = {**row, "ts": row.get("ts") or time.time()}
        if row.get("pii_flag") or not self.store_query_text:
            row["query_text"] = None
        values = [row.get(n) for n in NAMES]
        with self._lock:
            cur = self._conn.execute(
                f"INSERT INTO requests ({', '.join(NAMES)}) VALUES ({', '.join('?' * len(NAMES))})", values)
            self._conn.commit()
            return cur.lastrowid

    def recent(self, n: int = 20) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(f"SELECT id, {', '.join(NAMES)} FROM requests ORDER BY id DESC LIMIT ?", (n,))
            return [dict(zip(["id", *NAMES], r)) for r in cur.fetchall()]
