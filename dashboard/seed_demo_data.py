"""
dashboard/seed_demo_data.py — Writes ~200 demo rows into data/telemetry_demo.sqlite.

The demo DB is SEPARATE from the real telemetry DB (data/telemetry.sqlite).
Every row has energy_backend='simulated' so the dashboard can detect and banner it.

Usage:
    python dashboard/seed_demo_data.py            # write 200 rows
    python dashboard/seed_demo_data.py --clear     # delete demo DB and exit
    python dashboard/seed_demo_data.py --n 50      # write 50 rows
    python dashboard/seed_demo_data.py --help

The dashboard reads this file when launched with --demo or JOULE_DEMO=1.
"""
from __future__ import annotations

import argparse
import math
import random
import sqlite3
import sys
import time
from pathlib import Path

# ── repo root on sys.path ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gateway.telemetry import COLUMNS, NAMES   # reuse exact column definitions

DEMO_DB = _REPO_ROOT / "data" / "telemetry_demo.sqlite"
SEED    = 42


# ── representative query bank ─────────────────────────────────────────────────
# Queries are placeholders; none contain PII.
_QUERIES_BY_TIER: dict[str, list[tuple[str, str]]] = {
    "lite": [
        ("what is 17 * 23?",                    "arithmetic"),
        ("set a timer for 10 minutes",           "device_control"),
        ("convert 5 miles to km",                "conversion"),
        ("what is the capital of France?",       "factoid"),
        ("spell the word 'necessary'",           "language"),
        ("what day is Christmas this year?",     "factoid"),
        ("translate hello to Spanish",           "translation"),
        ("how many days in a leap year?",        "arithmetic"),
        ("what is 2 to the power of 10?",        "arithmetic"),
        ("remind me to drink water",             "device_control"),
    ],
    "mid": [
        ("explain photosynthesis in 2 sentences",  "explanation"),
        ("what is the difference between RAM and ROM?", "explanation"),
        ("summarise the water cycle",              "summary"),
        ("what causes thunder?",                   "explanation"),
        ("give me a recipe for scrambled eggs",    "creative"),
        ("what is machine learning?",              "explanation"),
        ("how does GPS work?",                     "explanation"),
        ("what is the Doppler effect?",            "explanation"),
        ("compare SQL and NoSQL briefly",          "comparison"),
        ("what are three benefits of exercise?",   "list"),
    ],
    "full": [
        ("write a Python function to merge two sorted lists", "code"),
        ("explain the difference between TCP and UDP in depth", "explanation"),
        ("summarise the causes of World War I",    "summary"),
        ("what is gradient descent and why does it work?", "explanation"),
        ("write a haiku about artificial intelligence", "creative"),
        ("explain how transformers work",          "explanation"),
        ("debug this: for i in range(10): print(i) — why does it stop at 9?", "code"),
        ("what are the tradeoffs between microservices and monoliths?", "comparison"),
        ("explain reinforcement learning with an example", "explanation"),
        ("write unit tests for a function that reverses a string", "code"),
    ],
}

# tier weights: lite appears often, full appears rarely (realistic routing mix)
_TIER_WEIGHTS = {"lite": 0.45, "mid": 0.35, "full": 0.20}

# source mix
_SOURCES = ["model", "model", "model", "model", "cache", "rules"]

# mode presets
_MODES = ["balanced", "balanced", "saver", "performance"]


def _sample_row(rng: random.Random, ts: float) -> dict:
    """Return one synthetic telemetry row as a dict matching NAMES."""
    tier_key = rng.choices(
        list(_TIER_WEIGHTS.keys()),
        weights=list(_TIER_WEIGHTS.values()),
    )[0]
    source = rng.choice(_SOURCES)

    # pull a query from the tier's bank
    query_text, category = rng.choice(_QUERIES_BY_TIER[tier_key])

    # plausible confidence: FULL gets lower confidence (that's why it went there)
    if tier_key == "lite":
        confidence = rng.uniform(0.72, 0.98)
        predicted_tier = "lite"
        fallback = 0
    elif tier_key == "mid":
        confidence = rng.uniform(0.55, 0.80)
        predicted_tier = rng.choice(["mid", "lite"])
        fallback = 1 if predicted_tier == "lite" else 0
    else:  # full
        confidence = rng.uniform(0.28, 0.55)
        predicted_tier = rng.choice(["mid", "full"])
        fallback = 1

    # escalation happens ~5% of non-full queries
    escalated = 0
    if tier_key in ("lite", "mid") and rng.random() < 0.05:
        escalated = 1

    tau = round(rng.uniform(0.30, 0.70), 3)
    mode = rng.choice(_MODES)

    # latency: realistic ranges per tier (ms)
    ttft_ranges = {"lite": (200, 800),   "mid": (600, 1800),  "full": (900, 2500)}
    total_ranges = {"lite": (800, 4000), "mid": (3000, 12000), "full": (10000, 30000)}
    ttft_ms  = rng.uniform(*ttft_ranges[tier_key])
    total_ms = rng.uniform(*total_ranges[tier_key])
    router_ms = rng.uniform(0.5, 3.0)

    tokens_out = int(rng.uniform(20, 200))
    # tokens_in: not written by the gateway (always None in real data)
    tokens_in = None

    # Energy: placeholder ratios from config.yaml energy_weights_placeholder
    # lite=1, mid=2.5, full=5 (arbitrary relative units scaled to ~joules range)
    _energy_scale = {"lite": 1.0, "mid": 2.5, "full": 5.0}
    base_j = _energy_scale[tier_key] * rng.uniform(0.8, 1.2)
    joules = round(base_j, 4)
    joules_est_full = round(5.0 * rng.uniform(0.8, 1.2), 4)

    # battery: simulated draining from 80% down
    # We'll compute this outside; for now store None, set by caller
    battery_pct_sim  = None  # set by caller
    battery_pct_real = None  # not available

    pii_flag = 0  # no PII in demo queries
    query_hash = str(hash(query_text) & 0xFFFFFFFF)

    return {
        "ts":              ts,
        "query_hash":      query_hash,
        "query_text":      query_text,
        "category_guess":  category,
        "source":          source,
        "pii_flag":        pii_flag,
        "predicted_tier":  predicted_tier,
        "tier":            tier_key,
        "confidence":      round(confidence, 4),
        "fallback":        fallback,
        "escalated":       escalated,
        "tau":             tau,
        "mode":            mode,
        "router_ms":       round(router_ms, 3),
        "ttft_ms":         round(ttft_ms, 2),
        "total_ms":        round(total_ms, 2),
        "tokens_in":       tokens_in,
        "tokens_out":      tokens_out,
        "joules":          joules,
        "joules_est_full": joules_est_full,
        "battery_pct_real": battery_pct_real,
        "battery_pct_sim": battery_pct_sim,
        "energy_backend":  "simulated",   # ← marks these rows as demo data
        "is_estimate":     1,
        "error":           None,
    }


def seed(n: int = 200, clear_first: bool = False) -> None:
    """Write n demo rows to DEMO_DB.

    If clear_first, the DB is deleted before writing.
    Rows are spread over the last hour, with a simulated battery starting at 80%.
    """
    if clear_first and DEMO_DB.exists():
        DEMO_DB.unlink()
        print(f"Cleared {DEMO_DB}")
        return

    DEMO_DB.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    conn = sqlite3.connect(DEMO_DB, check_same_thread=False)
    cols = ", ".join(f"{name} {typ}" for name, typ in COLUMNS)
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS requests "
        f"(id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})"
    )
    conn.commit()

    now = time.time()
    start_ts = now - 3600  # spread over last hour

    # simulated battery draining from 80%
    battery_pct_sim = 80.0
    typical_watts   = 8.0        # from config.yaml battery.typical_watts
    capacity_wh     = 50.0       # from config.yaml battery.capacity_wh

    rows = []
    for i in range(n):
        # linearly space timestamps, with small jitter
        frac = i / max(n - 1, 1)
        ts = start_ts + frac * 3600 + rng.uniform(-10, 10)
        row = _sample_row(rng, ts)

        # update simulated battery
        joules = row["joules"] or 0.0
        wh_used = joules / 3600.0
        # also drain idle power proportional to gap between requests
        if i > 0:
            gap_s = ts - rows[-1]["ts"]
            idle_wh = (typical_watts * 0.3) * gap_s / 3600.0
            wh_used += idle_wh
        battery_pct_sim -= (wh_used / capacity_wh) * 100.0
        battery_pct_sim = max(battery_pct_sim, 0.0)
        row["battery_pct_sim"] = round(battery_pct_sim, 2)

        rows.append(row)

    conn.executemany(
        f"INSERT INTO requests ({', '.join(NAMES)}) VALUES ({', '.join('?' * len(NAMES))})",
        [[r[name] for name in NAMES] for r in rows],
    )
    conn.commit()
    conn.close()

    print(f"Seeded {n} demo rows -> {DEMO_DB}")
    print(f"Battery simulated: 80% -> {battery_pct_sim:.1f}%")
    print(f"Tier distribution: {_describe_tiers(rows)}")


def _describe_tiers(rows: list[dict]) -> str:
    from collections import Counter
    counts = Counter(r["tier"] for r in rows)
    return ", ".join(f"{k.upper()}={v}" for k, v in sorted(counts.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed demo telemetry data into data/telemetry_demo.sqlite"
    )
    parser.add_argument("--clear", action="store_true",
                        help="Delete the demo database and exit")
    parser.add_argument("--n", type=int, default=200, metavar="N",
                        help="Number of demo rows to write (default: 200)")
    args = parser.parse_args()

    if args.clear:
        if DEMO_DB.exists():
            DEMO_DB.unlink()
            print(f"Cleared {DEMO_DB}")
        else:
            print(f"Nothing to clear — {DEMO_DB} does not exist")
    else:
        seed(n=args.n)
