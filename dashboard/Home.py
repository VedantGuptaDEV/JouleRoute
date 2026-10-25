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
import requests
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
    current_tau,
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
from dashboard.styles import (
    ACCENT,
    ACCENT_AMBER,
    ACCENT_TEAL,
    BORDER,
    CHART_CONFIG,
    TEXT_MUTED,
    TIER_PALETTE,
    badge,
    inject_css,
    plotly_template,
    section_title,
    style_fig,
)
from shared.config import load_config as load_cfg
from shared.tiers import Tier
from shared.prompts import parse_tier_override

# ── CLI / env-var flag ────────────────────────────────────────────────────────
def _is_demo() -> bool:
    if os.environ.get("JOULE_DEMO", "").strip() in ("1", "true", "yes"):
        return True
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

# ── clear-telemetry helper ────────────────────────────────────────────────────
def _clear_telemetry(db_path: Path) -> int:
    """Delete all rows from the requests table. Returns number of rows deleted."""
    import sqlite3
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(db_path)
    cur = conn.execute("DELETE FROM requests")
    conn.execute("DELETE FROM sqlite_sequence WHERE name='requests'")
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return deleted

# ── paths ─────────────────────────────────────────────────────────────────────
_CFG = load_cfg()
_TYPICAL_WATTS: float = _CFG.get("battery", {}).get("typical_watts", 8.0)

_REAL_DB  = _REPO_ROOT / _CFG.get("telemetry", {}).get("db", "data/telemetry.sqlite")
_DEMO_DB  = _REPO_ROOT / "data" / "telemetry_demo.sqlite"
_RESULTS  = _REPO_ROOT / "results"

DB_PATH = _DEMO_DB if DEMO_MODE else _REAL_DB

# ── gateway base URL (from config) ───────────────────────────────────────────
_GW = _CFG.get("gateway", {})
_GATEWAY_URL = f"http://{_GW.get('host', '127.0.0.1')}:{_GW.get('port', 8000)}"

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="JouleRoute — Live",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── inject shared stylesheet ──────────────────────────────────────────────────
inject_css()

# ── auto-refresh every 2 seconds ─────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    st_autorefresh(interval=2000, limit=None, key="live_refresh")
except ImportError:
    st.sidebar.info("Install streamlit-autorefresh for live updates.")

# ── sidebar — clear telemetry ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗑️ Telemetry")
    st.caption(f"DB: `{DB_PATH.name}`")
    if DEMO_MODE:
        st.info("Clear is disabled in demo mode.", icon="ℹ️")
    else:
        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = False

        if not st.session_state.confirm_clear:
            if st.button("Clear all rows", type="secondary", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("This will delete every row in the telemetry database. Are you sure?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, clear", type="primary", use_container_width=True):
                    deleted = _clear_telemetry(DB_PATH)
                    st.session_state.confirm_clear = False
                    st.success(f"Cleared {deleted} row(s).")
                    st.rerun()
            with col_no:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()

# ── load data  (unchanged logic) ─────────────────────────────────────────────
df = load_telemetry(DB_PATH)

# ── demo banner ───────────────────────────────────────────────────────────────
if DEMO_MODE or is_demo_data(df):
    st.warning(
        "⚠️ **Demo data — not measured results.**  "
        "Run the gateway to collect real telemetry.",
        icon="⚠️",
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
title_col, meta_col = st.columns([3, 1])
with title_col:
    st.markdown("# ⚡ JouleRoute")
with meta_col:
    mode_label = "demo" if DEMO_MODE else "live"
    st.markdown(
        f'<div style="text-align:right;padding-top:14px;color:{TEXT_MUTED};font-size:0.78rem;">'
        f'<span style="color:{ACCENT_TEAL};font-weight:600;">● {mode_label.upper()}</span>'
        f"&nbsp;&nbsp;`{DB_PATH.name}`</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Summary metric cards
# ─────────────────────────────────────────────────────────────────────────────
# All calculations identical to original; only presentation changes.
n_queries               = count_queries(df)
splits                  = tier_split(df)
saved_j, j_is_est       = energy_saved_joules(df)
saved_min, _            = battery_minutes_saved(df, _TYPICAL_WATTS)
saved_s, s_is_est       = time_saved_seconds(df)
est_label               = "simulated" if (DEMO_MODE or is_demo_data(df)) else "estimated"

st.markdown(section_title("📊", "Session Summary"), unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Queries served", n_queries)
    st.markdown(badge("measured"), unsafe_allow_html=True)

with col2:
    split_str = (
        " / ".join(
            f"{k.upper()} {v}"
            for k, v in sorted(
                splits.items(),
                key=lambda x: ["lite", "mid", "full"].index(x[0])
                if x[0] in ["lite", "mid", "full"]
                else 99,
            )
        )
        if splits
        else "—"
    )
    st.metric("Tier split", split_str)
    st.markdown(badge("measured"), unsafe_allow_html=True)

with col3:
    st.metric("Energy saved", f"{saved_j:+.2f} J")
    st.markdown(badge(est_label), unsafe_allow_html=True)
    if saved_j < 0:
        st.caption("Negative = escalation overhead cost more than routing saved")

with col4:
    st.metric("Time saved", f"{saved_s:+.1f} s")
    st.markdown(badge(est_label), unsafe_allow_html=True)
    if saved_s < 0:
        st.caption("Negative = escalation added latency")

with col5:
    st.metric("Battery saved", f"{saved_min:+.2f} min")
    st.markdown(badge(est_label), unsafe_allow_html=True)
    st.caption(f"@ {_TYPICAL_WATTS} W typical (config)")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Charts
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(section_title("📈", "Routing Insights"), unsafe_allow_html=True)

chart_col1, chart_col2, chart_col3 = st.columns([2, 1, 1])

# ── Chart 1 · Cumulative energy ───────────────────────────────────────────────
with chart_col1:
    with st.container():
        st.markdown("**Cumulative energy — JouleRoute vs Always-FULL**")
        cum = cumulative_energy(df)
        if cum.empty:
            st.info("No data yet — send some queries through the gateway.")
        else:
            fig = go.Figure()
            # Shaded savings gap — no tooltip, no legend entry
            fig.add_trace(go.Scatter(
                x=pd.concat([cum["ts"], cum["ts"].iloc[::-1]]),
                y=pd.concat([cum["cumulative_full_j"], cum["cumulative_actual_j"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(99,102,241,0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))
            # Difference column used by the JouleRoute trace tooltip
            diff_j = cum["cumulative_full_j"] - cum["cumulative_actual_j"]
            req_num = list(range(1, len(cum) + 1))
            fig.add_trace(go.Scatter(
                x=cum["ts"],
                y=cum["cumulative_full_j"],
                mode="lines",
                name="Always FULL",
                line=dict(color=ACCENT_AMBER, width=2, dash="dot"),
                hovertemplate=(
                    "<b>Always FULL</b><br>"
                    "Request #%{customdata}<br>"
                    "Cumulative: %{y:.2f} J<extra></extra>"
                ),
                customdata=req_num,
            ))
            fig.add_trace(go.Scatter(
                x=cum["ts"],
                y=cum["cumulative_actual_j"],
                mode="lines",
                name="JouleRoute",
                line=dict(color=ACCENT, width=2.5),
                # customdata columns: [req_num, diff_j]
                customdata=list(zip(req_num, diff_j)),
                hovertemplate=(
                    "<b>JouleRoute</b><br>"
                    "Request #%{customdata[0]}<br>"
                    "Cumulative: %{y:.2f} J<br>"
                    "Saved: <b>%{customdata[1]:.2f} J</b> (estimated)<extra></extra>"
                ),
            ))
            # hovermode="x unified" shows both lines' values in one tooltip on the same x
            style_fig(fig, yaxis_title="Joules (estimated)", xaxis_title="",
                      hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        st.caption(
            f"{badge('estimated')} Joules are placeholder values until the energy meter is built.",
            unsafe_allow_html=True,
        )

# ── Chart 2 · Tier share donut ────────────────────────────────────────────────
with chart_col2:
    with st.container():
        st.markdown("**Tier share**")
        if not splits:
            st.info("No data yet.")
        else:
            labels = [k.upper() for k in splits.keys()]
            values = list(splits.values())
            colors = [TIER_PALETTE.get(k, TEXT_MUTED) for k in splits.keys()]
            fig2 = go.Figure(go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(color="#0F172A", width=2)),
                textinfo="label+percent",
                textposition="inside",
                textfont=dict(family="Inter, sans-serif", size=11, color="#F1F5F9"),
                hole=0.55,
                insidetextorientation="radial",
                hovertemplate="<b>%{label}</b><br>Queries: %{value}<br>Share: %{percent}<extra></extra>",
            ))
            style_fig(fig2, show_legend=False)
            st.plotly_chart(fig2, use_container_width=True, config=CHART_CONFIG)
        st.markdown(badge("measured"), unsafe_allow_html=True)

# ── Chart 3 · Median latency per tier ────────────────────────────────────────
with chart_col3:
    with st.container():
        st.markdown("**Median latency**")
        lat = median_latency_per_tier(df)
        if lat.empty:
            st.info("No data yet.")
        else:
            tier_colors_bar = [TIER_PALETTE.get(t, TEXT_MUTED) for t in lat["tier"]]
            # Convert ms → s so LITE (~0.5 s) and MID/FULL (~10–22 s) are both readable.
            # Keep original ms values (median + p95) in customdata for the tooltip.
            ttft_s  = lat["ttft_ms_median"] / 1000
            total_s = lat["total_ms_median"] / 1000
            tiers_upper = [t.upper() for t in lat["tier"]]
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=tiers_upper,
                y=ttft_s,
                name="TTFT",
                marker=dict(color=ACCENT, opacity=0.75),
                # customdata columns: [median_ms, p95_ms]
                customdata=list(zip(
                    lat["ttft_ms_median"].fillna(0),
                    lat["ttft_ms_p95"].fillna(0),
                )),
                hovertemplate=(
                    "<b>%{x} — TTFT</b><br>"
                    "Median: %{customdata[0]:.0f} ms<br>"
                    "p95:&nbsp;&nbsp;&nbsp;&nbsp;%{customdata[1]:.0f} ms<extra></extra>"
                ),
            ))
            fig3.add_trace(go.Bar(
                x=tiers_upper,
                y=total_s,
                name="Total",
                marker=dict(color=tier_colors_bar),
                opacity=0.85,
                customdata=list(zip(
                    lat["total_ms_median"].fillna(0),
                    lat["total_ms_p95"].fillna(0),
                )),
                hovertemplate=(
                    "<b>%{x} — Total</b><br>"
                    "Median: %{customdata[0]:.0f} ms<br>"
                    "p95:&nbsp;&nbsp;&nbsp;&nbsp;%{customdata[1]:.0f} ms<extra></extra>"
                ),
            ))
            style_fig(fig3, yaxis_title="seconds", xaxis_title="Tier")
            fig3.update_layout(barmode="overlay")
            st.plotly_chart(fig3, use_container_width=True, config=CHART_CONFIG)
        st.markdown(badge("measured"), unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Query box
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(section_title("💬", "Send a Query"), unsafe_allow_html=True)

if DEMO_MODE:
    st.info("Query box is disabled in demo mode — the gateway is not running.", icon="ℹ️")
else:
    qbox_col, tier_col = st.columns([4, 1])
    with qbox_col:
        user_query = st.text_input(
            "Query",
            placeholder='e.g. "use full — explain gradient descent" or just ask anything',
            label_visibility="collapsed",
            key="query_input",
        )
    with tier_col:
        tier_choice = st.selectbox(
            "Tier",
            options=["auto", "lite", "mid", "full"],
            index=0,
            key="tier_select",
            label_visibility="collapsed",
        )

    send_col, _ = st.columns([1, 5])
    with send_col:
        send_clicked = st.button("Send ⚡", type="primary", use_container_width=True)

    if send_clicked and user_query.strip():
        # Natural-language tier override takes priority over the dropdown.
        nl_tier, clean_query = parse_tier_override(user_query.strip())
        effective_tier = nl_tier if nl_tier else (tier_choice if tier_choice != "auto" else "auto")
        if nl_tier:
            st.caption(f"Natural-language directive detected → forcing **{nl_tier.upper()}** for this query.")

        with st.spinner(f"Routing to {effective_tier.upper() if effective_tier != 'auto' else 'auto'} …"):
            try:
                resp = requests.post(
                    f"{_GATEWAY_URL}/v1/chat/completions",
                    json={
                        "model": effective_tier,
                        "messages": [{"role": "user", "content": clean_query}],
                        "stream": False,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
                xr = data.get("x_routing", {})

                # Answer
                st.markdown(f"**Answer** ({xr.get('tier', '?').upper()})")
                st.markdown(answer)

                # Energy receipt
                with st.expander("⚡ Energy receipt", expanded=False):
                    receipt_cols = st.columns(4)
                    with receipt_cols[0]:
                        st.metric("Tier", xr.get("tier", "?").upper())
                    with receipt_cols[1]:
                        conf = xr.get("confidence")
                        st.metric("Confidence", f"{conf:.2f}" if conf is not None else "—")
                    with receipt_cols[2]:
                        j = xr.get("joules")
                        st.metric("Energy", f"{j:.3f} J" if j is not None else "—")
                        st.markdown(badge("estimated" if xr.get("is_estimate") else "measured"),
                                    unsafe_allow_html=True)
                    with receipt_cols[3]:
                        ms = xr.get("total_ms")
                        st.metric("Latency", f"{ms:,.0f} ms" if ms is not None else "—")

                    detail_rows = []
                    if xr.get("fallback"):
                        detail_rows.append("🔁 Confidence gate triggered fallback to FULL")
                    if xr.get("escalated"):
                        detail_rows.append("⬆️ In-generation escalation to FULL")
                    if xr.get("source") == "cache":
                        detail_rows.append("💾 Answered from cache")
                    if xr.get("pii_flag"):
                        detail_rows.append("🔒 PII detected — kept on-device")
                    for row in detail_rows:
                        st.caption(row)

            except requests.ConnectionError:
                st.error(f"Cannot reach the gateway at {_GATEWAY_URL}. Start it with `python gateway/app.py`.")
            except requests.HTTPError as e:
                st.error(f"Gateway returned an error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Live request feed
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(section_title("🔄", "Latest 20 Requests"), unsafe_allow_html=True)

recent = recent_requests(df, n=20)

if recent.empty:
    st.info("No requests recorded yet. Start the gateway and send a query.")
else:
    display_cols = [
        "id", "ts", "query_text", "tier", "predicted_tier", "confidence",
        "fallback", "escalated", "ttft_ms", "total_ms", "joules", "source", "error",
    ]
    display_cols = [c for c in display_cols if c in recent.columns]
    display = recent[display_cols].copy()

    for col in ("fallback", "escalated"):
        if col in display.columns:
            display[col] = display[col].map(
                lambda v: "✓" if v is True or v == 1 else ("" if pd.isna(v) else "")
            )

    if "confidence" in display.columns:
        display["confidence"] = display["confidence"].map(
            lambda v: f"{v:.2f}" if pd.notna(v) else ""
        )
    for col in ("ttft_ms", "total_ms"):
        if col in display.columns:
            display[col] = display[col].map(
                lambda v: f"{v:,.0f}" if pd.notna(v) else "—"
            )
    if "joules" in display.columns:
        display["joules"] = display["joules"].map(
            lambda v: f"{v:.3f}" if pd.notna(v) else "—"
        )
    if "tier" in display.columns:
        display["tier"] = display["tier"].str.upper()
    if "predicted_tier" in display.columns:
        display["predicted_tier"] = display["predicted_tier"].str.upper()

    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Query text is hidden for rows where the PII flag is set.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Session stats
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(section_title("📉", "Session Stats"), unsafe_allow_html=True)

stat_col1, stat_col2, stat_col3 = st.columns(3)

with stat_col1:
    fb = fallback_rate(df)
    st.metric("Fallback rate", f"{fb:.1%}" if fb is not None else "—")
    st.markdown(badge("measured"), unsafe_allow_html=True)

with stat_col2:
    esc = escalation_rate(df)
    st.metric("Escalation rate", f"{esc:.1%}" if esc is not None else "—")
    st.markdown(badge("measured"), unsafe_allow_html=True)

with stat_col3:
    conf = confidence_distribution(df)
    if conf.empty:
        st.info("No confidence data yet.")
    else:
        fig_conf = go.Figure(go.Histogram(
            x=conf,
            nbinsx=20,
            marker=dict(color=ACCENT, opacity=0.80),
            # %{x} is the bin start; Plotly has no %{x_end} for histograms.
            # Show bin start and count — clear and unambiguous.
            hovertemplate="Confidence ≥ %{x:.2f}: %{y} queries<extra></extra>",
        ))
        style_fig(fig_conf, height=190, xaxis_title="Confidence",
                  yaxis_title="Count", show_legend=False, hovermode="x")
        # Tau threshold line — read from most recent telemetry row
        tau_val = current_tau(df)
        if tau_val is not None:
            fig_conf.add_vline(
                x=tau_val,
                line=dict(color=ACCENT_AMBER, width=2, dash="dash"),
                annotation_text=f"τ = {tau_val:.2f}",
                annotation_position="top right",
                annotation_font=dict(color=ACCENT_AMBER, size=11),
            )
        st.plotly_chart(fig_conf, use_container_width=True, config=CHART_CONFIG)
        st.markdown(
            f"<small style='color:{TEXT_MUTED}'>Confidence distribution</small> "
            + badge("measured"),
            unsafe_allow_html=True,
        )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Measured results panel (results/*.csv)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(section_title("🧪", "Measured Results"), unsafe_allow_html=True)

spec_df  = load_spec_decoding(_RESULTS)
bench_df = load_bench_summary(_RESULTS)

if spec_df is None and bench_df is None:
    st.info(
        "No results yet. Run `bench/run_bench.py` and `bench/build_energy_profile.py` "
        "to populate this section."
    )
else:
    if spec_df is not None:
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("**Speculative decoding · speed comparison**")
            st.caption("Source: `results/spec_decoding.csv` · FULL tier · tok/s = output tokens / total time (end to end, includes time to first token).")
            show_cols = [
                c for c in ("spec", "runs", "ttft_ms_median", "total_ms_median",
                            "tokens_median", "tok_s_median")
                if c in spec_df.columns
            ]
            disp_spec = spec_df[show_cols].copy()
            for col in ("ttft_ms_median", "total_ms_median"):
                if col in disp_spec.columns:
                    disp_spec[col] = disp_spec[col].apply(
                        lambda v: f"{v:,.1f}" if pd.notna(v) else "—"
                    )
            for col in ("tokens_median",):
                if col in disp_spec.columns:
                    disp_spec[col] = disp_spec[col].apply(
                        lambda v: f"{int(v):,}" if pd.notna(v) else "—"
                    )
            st.dataframe(disp_spec, use_container_width=True, hide_index=True)

        with res_col2:
            if "tok_s_median" in spec_df.columns and len(spec_df) >= 2:
                on_row  = spec_df[spec_df["spec"] == "on"]
                off_row = spec_df[spec_df["spec"] == "off"]
                if not on_row.empty and not off_row.empty:
                    tok_on  = float(on_row["tok_s_median"].iloc[0])
                    tok_off = float(off_row["tok_s_median"].iloc[0])
                    if tok_off > 0:
                        speedup   = tok_on / tok_off
                        direction = "faster" if speedup > 1 else "slower"

                        fig_spec = go.Figure(go.Bar(
                            x=["Spec ON", "Spec OFF"],
                            y=[tok_on, tok_off],
                            marker=dict(
                                color=[ACCENT_AMBER if speedup < 1 else ACCENT_TEAL, ACCENT_TEAL],
                                opacity=0.85,
                            ),
                            text=[f"{tok_on:.2f}", f"{tok_off:.2f}"],
                            textposition="outside",
                            textfont=dict(color=TEXT_MUTED, size=11),
                        ))
                        style_fig(fig_spec, height=200, yaxis_title="tok/s",
                                  show_legend=False)
                        st.plotly_chart(fig_spec, use_container_width=True,
                                        config=CHART_CONFIG)
                        st.markdown(
                            f"{badge('measured')} &nbsp; Spec ON is "
                            f"**{(1/speedup if speedup < 1 else speedup):.1f}× {direction}** on this hardware.",
                            unsafe_allow_html=True,
                        )

    if bench_df is not None:
        st.markdown("**Benchmark summary**")
        st.caption("Source: `results/bench_summary.csv`")
        st.dataframe(bench_df, use_container_width=True, hide_index=True)
        st.markdown(badge("measured"), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f'<p style="color:{TEXT_MUTED};font-size:0.75rem;text-align:center;">'
    f"JouleRoute dashboard &nbsp;·&nbsp; "
    f"{badge('measured')} from live telemetry &nbsp; "
    f"{badge('estimated')} placeholder energy weights (config <code>energy_weights_placeholder</code>) &nbsp; "
    f"{badge('simulated')} seeded demo data &nbsp;·&nbsp; "
    f"Typical power: {_TYPICAL_WATTS} W (config <code>battery.typical_watts</code>)"
    f"</p>",
    unsafe_allow_html=True,
)
