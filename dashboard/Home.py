"""
dashboard/Home.py — JouleRoute Live View (Streamlit).

Launch normally:
    streamlit run dashboard/Home.py

Launch with demo data (reads data/telemetry_demo.sqlite instead):
    streamlit run dashboard/Home.py -- --demo
    # or set env var:  set JOULE_DEMO=1

Auto-refreshes every 2 seconds.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── resolve repo root so shared/ is importable ───────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard.data import (
    TIER_COLORS,
    battery_minutes_saved,
    confidence_distribution,
    count_queries,
    cumulative_energy,
    energy_saved_joules,
    escalation_rate,
    fallback_rate,
    is_demo_data,
    load_bench_summary,
    load_spec_decoding,
    load_telemetry,
    median_latency_per_tier,
    recent_requests,
    tier_split,
    time_saved_seconds,
)
from shared.config import load_config as load_cfg
from shared.tiers import Tier

# ── CLI / env-var flag ────────────────────────────────────────────────────────
def _is_demo() -> bool:
    if os.environ.get("JOULE_DEMO", "").strip() in ("1", "true", "yes"):
        return True
    # Streamlit passes extra args after "--"
    try:
        idx = sys.argv.index("--")
        extra = sys.argv[idx + 1:]
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--demo", action="store_true")
        ns, _ = p.parse_known_args(extra)
        return ns.demo
    except (ValueError, SystemExit):
        return False


DEMO_MODE = _is_demo()

# ── paths ─────────────────────────────────────────────────────────────────────
_CFG = load_cfg()
_TYPICAL_WATTS: float = _CFG.get("battery", {}).get("typical_watts", 8.0)

_REAL_DB   = _REPO_ROOT / _CFG.get("telemetry", {}).get("db", "data/telemetry.sqlite")
_DEMO_DB   = _REPO_ROOT / "data" / "telemetry_demo.sqlite"
_RESULTS   = _REPO_ROOT / "results"

DB_PATH = _DEMO_DB if DEMO_MODE else _REAL_DB

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JouleRoute — Live",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── auto-refresh every 2 seconds ─────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    st_autorefresh(interval=2000, limit=None, key="live_refresh")
except ImportError:
    # graceful degradation: user must refresh manually
    st.sidebar.info("Install streamlit-autorefresh for live updates.")

# ── load data ─────────────────────────────────────────────────────────────────
df = load_telemetry(DB_PATH)

# ── demo banner ───────────────────────────────────────────────────────────────
if DEMO_MODE or is_demo_data(df):
    st.warning(
        "⚠️ **Demo data — not measured results.** "
        "Run the gateway to collect real telemetry.",
        icon="⚠️",
    )

# ── page header ───────────────────────────────────────────────────────────────
st.title("⚡ JouleRoute — Live")
st.caption(f"Database: `{DB_PATH}`  ·  {'demo' if DEMO_MODE else 'live'} mode")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Summary cards
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Session summary")

n_queries = count_queries(df)
splits    = tier_split(df)
saved_j, j_is_est   = energy_saved_joules(df, )
saved_min, _         = battery_minutes_saved(df, _TYPICAL_WATTS)
saved_s,  s_is_est   = time_saved_seconds(df)

def _badge(label: str) -> str:
    colors = {"measured": "#4CAF50", "estimated": "#FF9800", "simulated": "#9E9E9E"}
    c = colors.get(label.lower(), "#9E9E9E")
    return (
        f'<span style="background:{c};color:white;border-radius:4px;'
        f'padding:2px 7px;font-size:0.72em;font-weight:600;'
        f'margin-left:4px">{label.upper()}</span>'
    )


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Queries served", n_queries)
    st.markdown(_badge("measured"), unsafe_allow_html=True)

with col2:
    split_str = " / ".join(
        f"{k.upper()} {v}" for k, v in sorted(splits.items(), key=lambda x: ["lite","mid","full"].index(x[0]) if x[0] in ["lite","mid","full"] else 99)
    ) if splits else "—"
    st.metric("Tier split", split_str)
    st.markdown(_badge("measured"), unsafe_allow_html=True)

with col3:
    est_label = "simulated" if (DEMO_MODE or is_demo_data(df)) else "estimated"
    st.metric("Energy saved", f"{saved_j:+.2f} J")
    st.markdown(_badge(est_label), unsafe_allow_html=True)
    if saved_j < 0:
        st.caption("Negative = escalation overhead cost more than routing saved")

with col4:
    st.metric("Time saved", f"{saved_s:+.1f} s")
    st.markdown(_badge(est_label), unsafe_allow_html=True)
    if saved_s < 0:
        st.caption("Negative = escalation added latency")

with col5:
    st.metric("Battery saved", f"{saved_min:+.2f} min")
    st.markdown(_badge(est_label), unsafe_allow_html=True)
    st.caption(f"@{_TYPICAL_WATTS} W typical (config)")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Charts
# ─────────────────────────────────────────────────────────────────────────────

chart_col1, chart_col2, chart_col3 = st.columns([2, 1, 1])

# Chart 1 — Cumulative energy: JouleRoute vs Always-FULL
with chart_col1:
    st.subheader("Chart 1 · Cumulative energy")

    cum = cumulative_energy(df)
    if cum.empty:
        st.info("No data yet.")
    else:
        fig = go.Figure()

        # Shaded gap between the two lines
        fig.add_trace(go.Scatter(
            x=pd.concat([cum["ts"], cum["ts"].iloc[::-1]]),
            y=pd.concat([cum["cumulative_full_j"], cum["cumulative_actual_j"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(76, 175, 80, 0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            name="gap",
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=cum["ts"],
            y=cum["cumulative_full_j"],
            mode="lines",
            name="Always FULL",
            line=dict(color=TIER_COLORS["full"], width=2, dash="dash"),
        ))

        fig.add_trace(go.Scatter(
            x=cum["ts"],
            y=cum["cumulative_actual_j"],
            mode="lines",
            name="JouleRoute",
            line=dict(color="#2196F3", width=2),
        ))

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Joules (estimated)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ *Estimated* — joules are placeholder values until the energy meter is built.")

# Chart 2 — Tier share pie
with chart_col2:
    st.subheader("Chart 2 · Tier share")

    if not splits:
        st.info("No data yet.")
    else:
        labels = [k.upper() for k in splits.keys()]
        values = list(splits.values())
        colors = [TIER_COLORS.get(k, "#9E9E9E") for k in splits.keys()]

        fig2 = go.Figure(go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            textinfo="label+percent",
            hole=0.4,
        ))
        fig2.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(_badge("measured"), unsafe_allow_html=True)

# Chart 3 — Median latency per tier
with chart_col3:
    st.subheader("Chart 3 · Median latency")

    lat = median_latency_per_tier(df)
    if lat.empty:
        st.info("No data yet.")
    else:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=[t.upper() for t in lat["tier"]],
            y=lat["ttft_ms_median"],
            name="TTFT",
            marker_color="#42A5F5",
        ))
        fig3.add_trace(go.Bar(
            x=[t.upper() for t in lat["tier"]],
            y=lat["total_ms_median"],
            name="Total",
            marker_color=[TIER_COLORS.get(t, "#9E9E9E") for t in lat["tier"]],
            opacity=0.7,
        ))
        fig3.update_layout(
            barmode="overlay",
            xaxis_title="Tier",
            yaxis_title="ms",
            legend=dict(orientation="h", y=-0.3),
            margin=dict(t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(_badge("measured"), unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Live request table
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Latest 20 requests")

recent = recent_requests(df, n=20)

if recent.empty:
    st.info("No requests recorded yet.")
else:
    display_cols = [
        "id", "ts", "query_text", "tier", "predicted_tier", "confidence",
        "fallback", "escalated", "ttft_ms", "total_ms", "joules", "source", "error",
    ]
    # only keep columns that actually exist in the dataframe
    display_cols = [c for c in display_cols if c in recent.columns]
    display = recent[display_cols].copy()

    # format booleans for readability
    for col in ("fallback", "escalated"):
        if col in display.columns:
            display[col] = display[col].map(lambda v: "✓" if v is True or v == 1 else ("" if pd.isna(v) else ""))

    # round floats
    for col in ("confidence",):
        if col in display.columns:
            display[col] = display[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    for col in ("ttft_ms", "total_ms"):
        if col in display.columns:
            display[col] = display[col].map(lambda v: f"{v:.0f}" if pd.notna(v) else "—")
    for col in ("joules",):
        if col in display.columns:
            display[col] = display[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")

    # tier color display
    if "tier" in display.columns:
        display["tier"] = display["tier"].str.upper()
    if "predicted_tier" in display.columns:
        display["predicted_tier"] = display["predicted_tier"].str.upper()

    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Query text hidden for rows with PII flag set.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Session stats (fallback rate, escalation rate, confidence histogram)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Session stats")

stat_col1, stat_col2, stat_col3 = st.columns(3)

with stat_col1:
    fb = fallback_rate(df)
    if fb is None:
        st.metric("Fallback rate", "—")
    else:
        st.metric("Fallback rate", f"{fb:.1%}")
    st.markdown(_badge("measured"), unsafe_allow_html=True)

with stat_col2:
    esc = escalation_rate(df)
    if esc is None:
        st.metric("Escalation rate", "—")
    else:
        st.metric("Escalation rate", f"{esc:.1%}")
    st.markdown(_badge("measured"), unsafe_allow_html=True)

with stat_col3:
    conf = confidence_distribution(df)
    if conf.empty:
        st.info("No confidence data yet.")
    else:
        fig_conf = go.Figure(go.Histogram(
            x=conf,
            nbinsx=20,
            marker_color="#42A5F5",
        ))
        fig_conf.update_layout(
            xaxis_title="Confidence",
            yaxis_title="Count",
            margin=dict(t=10, b=10),
            height=180,
            title_text="Confidence distribution",
            title_font_size=12,
        )
        st.plotly_chart(fig_conf, use_container_width=True)
        st.markdown(_badge("measured"), unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Measured results panel (reads results/*.csv)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Measured results (from results/)")

spec_df   = load_spec_decoding(_RESULTS)
bench_df  = load_bench_summary(_RESULTS)

if spec_df is None and bench_df is None:
    st.info(
        "No results yet. Run `bench/run_bench.py` and `bench/build_energy_profile.py` "
        "to populate this section."
    )
else:
    if spec_df is not None:
        st.markdown("**Speculative decoding: speed comparison**")
        st.caption(
            "Source: `results/spec_decoding.csv` · Measured on FULL tier. "
            "tok/s = decode rate excluding first token."
        )

        # Build a clean table for display
        show_cols = [c for c in ("spec", "runs", "ttft_ms_median", "total_ms_median",
                                  "tokens_median", "tok_s_median") if c in spec_df.columns]
        st.dataframe(spec_df[show_cols], use_container_width=True, hide_index=True)

        if "tok_s_median" in spec_df.columns and len(spec_df) >= 2:
            on_row  = spec_df[spec_df["spec"] == "on"]
            off_row = spec_df[spec_df["spec"] == "off"]
            if not on_row.empty and not off_row.empty:
                tok_on  = float(on_row["tok_s_median"].iloc[0])
                tok_off = float(off_row["tok_s_median"].iloc[0])
                if tok_off > 0:
                    speedup = tok_on / tok_off
                    # spec_decoding.csv shows spec=on is SLOWER (1.56 tok/s vs 7.73)
                    # This is expected when draft model is same size; report factually.
                    direction = "faster" if speedup > 1 else "slower"
                    st.info(
                        f"Spec ON: **{tok_on:.2f} tok/s** vs Spec OFF: **{tok_off:.2f} tok/s** "
                        f"→ spec decoding is **{abs(speedup):.1f}× {direction}** on this hardware. "
                        f"(See `results/spec_decoding.csv` for all runs.)"
                    )
                    st.markdown(_badge("measured"), unsafe_allow_html=True)

    if bench_df is not None:
        st.markdown("**Benchmark summary**")
        st.caption("Source: `results/bench_summary.csv`")
        st.dataframe(bench_df, use_container_width=True, hide_index=True)
        st.markdown(_badge("measured"), unsafe_allow_html=True)

st.divider()
st.caption(
    "JouleRoute dashboard · "
    "Badges: MEASURED = from live telemetry; ESTIMATED = derived from placeholder energy weights "
    f"(config.yaml `energy_weights_placeholder`); SIMULATED = seeded demo data. "
    f"Typical power: {_TYPICAL_WATTS} W (config `battery.typical_watts`)."
)
