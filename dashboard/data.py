"""
dashboard/data.py — pandas helpers for reading the JouleRoute telemetry SQLite database.

No raw SQL lives in page files; all queries are here.

Columns in `requests` table (from gateway/telemetry.py):
  id, ts, query_hash, query_text, category_guess, source, pii_flag,
  predicted_tier, tier, confidence, fallback, escalated, tau, mode,
  router_ms, ttft_ms, total_ms, tokens_in, tokens_out,
  joules, joules_est_full, battery_pct_real, battery_pct_sim,
  energy_backend, is_estimate, error

Missing / null in current gateway (energy meter not built yet):
  joules, joules_est_full, battery_pct_real, battery_pct_sim,
  energy_backend, is_estimate, tokens_in
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

# ── Tier colour palette (shared with Home.py) ─────────────────────────────────
TIER_COLORS: dict[str, str] = {
    "lite": "#4CAF50",   # green
    "mid":  "#FFC107",   # amber/yellow
    "full": "#FF9800",   # orange
}

# ── Placeholder energy weights used when joules are not measured ──────────────
# These come from config.yaml router.energy_weights_placeholder.
# They are relative only and used to produce *estimates*. Labelled as such in UI.
ENERGY_PLACEHOLDER_J: dict[str, float] = {
    "lite": 1.0,
    "mid":  2.5,
    "full": 5.0,
}

_ALL_COLUMNS = [
    "id", "ts", "query_hash", "query_text", "category_guess", "source",
    "pii_flag", "predicted_tier", "tier", "confidence", "fallback", "escalated",
    "tau", "mode", "router_ms", "ttft_ms", "total_ms", "tokens_in", "tokens_out",
    "joules", "joules_est_full", "battery_pct_real", "battery_pct_sim",
    "energy_backend", "is_estimate", "error",
]


def _connect(db_path: Path) -> sqlite3.Connection:
    """Return a read-only connection to the given SQLite file."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)


def load_telemetry(db_path: Path) -> pd.DataFrame:
    """
    Load every row from the `requests` table into a DataFrame.

    Returns an empty DataFrame with correct dtypes when the file is missing,
    the table is empty, or the query fails.
    """
    if not db_path.exists():
        return _empty_df()

    try:
        conn = _connect(db_path)
        df = pd.read_sql_query(
            "SELECT id, ts, query_hash, query_text, category_guess, source,"
            " pii_flag, predicted_tier, tier, confidence, fallback, escalated,"
            " tau, mode, router_ms, ttft_ms, total_ms, tokens_in, tokens_out,"
            " joules, joules_est_full, battery_pct_real, battery_pct_sim,"
            " energy_backend, is_estimate, error"
            " FROM requests ORDER BY id ASC",
            conn,
        )
        conn.close()
    except Exception:
        return _empty_df()

    if df.empty:
        return _empty_df()

    # ── type coercions ────────────────────────────────────────────────────────
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    for col in ("pii_flag", "fallback", "escalated", "is_estimate"):
        df[col] = df[col].astype("boolean")
    for col in ("confidence", "router_ms", "ttft_ms", "total_ms", "joules",
                "joules_est_full", "battery_pct_real", "battery_pct_sim", "tau"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("tokens_in", "tokens_out"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # ── derived columns ───────────────────────────────────────────────────────
    # simulated flag: row is from demo seed data when energy_backend == 'simulated'
    df["_simulated"] = df["energy_backend"] == "simulated"

    # joules_effective: use measured joules where present; fall back to placeholder estimate
    df["joules_effective"] = df["joules"]
    mask_missing = df["joules"].isna()
    if mask_missing.any():
        df.loc[mask_missing, "joules_effective"] = df.loc[mask_missing, "tier"].map(
            ENERGY_PLACEHOLDER_J
        )
        # joules_effective is an estimate wherever joules was null
    df["joules_is_estimate"] = mask_missing | df["is_estimate"].fillna(False)

    # joules_est_full fallback from placeholder when not measured
    df["joules_est_full_eff"] = df["joules_est_full"]
    mask_full_missing = df["joules_est_full"].isna()
    if mask_full_missing.any():
        df.loc[mask_full_missing, "joules_est_full_eff"] = ENERGY_PLACEHOLDER_J["full"]

    return df


def _empty_df() -> pd.DataFrame:
    """DataFrame with all expected columns, zero rows."""
    return pd.DataFrame(columns=_ALL_COLUMNS + [
        "_simulated", "joules_effective", "joules_is_estimate", "joules_est_full_eff"
    ])


# ── Metric helpers ────────────────────────────────────────────────────────────

def count_queries(df: pd.DataFrame) -> int:
    return len(df)


def tier_split(df: pd.DataFrame) -> dict[str, int]:
    """Return {tier_key: count}. Only tiers that appear are included."""
    if df.empty or "tier" not in df.columns:
        return {}
    return df["tier"].value_counts().to_dict()


def tier_share(df: pd.DataFrame) -> dict[str, float]:
    """Return fraction 0–1 for each tier."""
    counts = tier_split(df)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def energy_saved_joules(df: pd.DataFrame) -> tuple[float, bool]:
    """
    Total energy saved vs. always-FULL, in joules.

    Returns (joules_saved, is_estimate).
    Savings can be negative when escalation caused extra work.
    always-full cost = sum(joules_est_full_eff)
    actual cost      = sum(joules_effective)
    """
    if df.empty:
        return 0.0, True
    full_cost = df["joules_est_full_eff"].sum()
    actual_cost = df["joules_effective"].sum()
    is_est = bool(df["joules_is_estimate"].any())
    return float(full_cost - actual_cost), is_est


def battery_minutes_saved(df: pd.DataFrame, typical_watts: float = 8.0) -> tuple[float, bool]:
    """
    battery_minutes_saved = energy_saved_J / (typical_watts × 60).
    Derived from joules, so always estimated.
    """
    saved_j, is_est = energy_saved_joules(df)
    minutes = saved_j / (typical_watts * 60.0)
    return float(minutes), True   # always estimated


def time_saved_seconds(df: pd.DataFrame) -> tuple[float, bool]:
    """
    Total wall-clock seconds saved vs. always-FULL.
    Estimated from median total_ms per tier (if available).
    Falls back to a placeholder ratio (full 5×, mid 2×, lite 1× of lite).
    """
    if df.empty:
        return 0.0, True

    tier_median_ms: dict[str, float] = {}
    for tier in ("lite", "mid", "full"):
        sub = df[df["tier"] == tier]["total_ms"].dropna()
        if not sub.empty:
            tier_median_ms[tier] = float(sub.median())

    # placeholder fallback from spec: config.yaml energy_weights_placeholder
    placeholder_ms = {"lite": 5000.0, "mid": 12000.0, "full": 25000.0}

    saved_ms = 0.0
    for _, row in df.iterrows():
        tier = row["tier"]
        if not tier:
            continue
        actual = row["total_ms"] if pd.notna(row["total_ms"]) else tier_median_ms.get(tier, placeholder_ms.get(tier, 0))
        full_est = tier_median_ms.get("full", placeholder_ms["full"])
        saved_ms += full_est - actual

    return float(saved_ms / 1000.0), True


def cumulative_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      ts, cumulative_actual_j, cumulative_full_j
    Suitable for a "JouleRoute vs Always-FULL" area chart.
    """
    if df.empty:
        return pd.DataFrame(columns=["ts", "cumulative_actual_j", "cumulative_full_j"])

    tmp = df[["ts", "joules_effective", "joules_est_full_eff"]].copy()
    tmp = tmp.sort_values("ts").reset_index(drop=True)
    tmp["cumulative_actual_j"] = tmp["joules_effective"].cumsum()
    tmp["cumulative_full_j"]   = tmp["joules_est_full_eff"].cumsum()
    return tmp[["ts", "cumulative_actual_j", "cumulative_full_j"]]


def median_latency_per_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with columns: tier, ttft_ms_median, total_ms_median, n.
    Includes only tiers with at least one non-null measurement.
    """
    if df.empty:
        return pd.DataFrame(columns=["tier", "ttft_ms_median", "total_ms_median", "n"])

    rows = []
    for tier in ("lite", "mid", "full"):
        sub = df[df["tier"] == tier]
        if sub.empty:
            continue
        rows.append({
            "tier": tier,
            "ttft_ms_median": sub["ttft_ms"].dropna().median() if not sub["ttft_ms"].dropna().empty else None,
            "total_ms_median": sub["total_ms"].dropna().median() if not sub["total_ms"].dropna().empty else None,
            "n": len(sub),
        })
    return pd.DataFrame(rows)


def recent_requests(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Return the n most recent rows, with PII query text redacted.
    """
    if df.empty:
        return df
    recent = df.sort_values("id", ascending=False).head(n).copy()
    # Mask query text for PII rows
    pii_mask = recent["pii_flag"].fillna(False).astype(bool)
    recent.loc[pii_mask, "query_text"] = "[hidden — PII]"
    return recent


def fallback_rate(df: pd.DataFrame) -> float | None:
    if df.empty or "fallback" not in df.columns:
        return None
    valid = df["fallback"].dropna()
    if valid.empty:
        return None
    return float(valid.astype(bool).mean())


def escalation_rate(df: pd.DataFrame) -> float | None:
    if df.empty or "escalated" not in df.columns:
        return None
    valid = df["escalated"].dropna()
    if valid.empty:
        return None
    return float(valid.astype(bool).mean())


def confidence_distribution(df: pd.DataFrame) -> pd.Series:
    """Return the confidence column (for histogram)."""
    if df.empty:
        return pd.Series(dtype=float)
    return df["confidence"].dropna()


def is_demo_data(df: pd.DataFrame) -> bool:
    """True if any row carries the simulated flag (seeded demo data)."""
    if df.empty or "_simulated" not in df.columns:
        return False
    return bool(df["_simulated"].any())


# ── Results CSV helpers ───────────────────────────────────────────────────────

def load_spec_decoding(results_dir: Path) -> pd.DataFrame | None:
    """
    Load results/spec_decoding.csv if it exists.
    Columns: spec, runs, ttft_ms_median, total_ms_median, tokens_median, tok_s_median, ...
    Returns None if the file is missing.
    """
    path = results_dir / "spec_decoding.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_bench_summary(results_dir: Path) -> pd.DataFrame | None:
    path = results_dir / "bench_summary.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_energy_profile(results_dir: Path) -> dict | None:
    import json
    path = results_dir / "energy_profile.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
