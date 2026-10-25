"""
dashboard/pages/Evaluation.py — JouleRoute Evaluation View (Streamlit).

Reads ONLY from results/. Never runs benchmarks or touches gateway/serving.
Every section degrades gracefully: if its source file is missing it shows a
"not measured yet" card instead of fake data.

Launch: streamlit run dashboard/Home.py   (this page appears automatically
in the sidebar via Streamlit's multi-page convention).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── resolve repo root ─────────────────────────────────────────────────────────
import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard.styles import (
    ACCENT,
    ACCENT_AMBER,
    ACCENT_TEAL,
    BORDER,
    CHART_CONFIG,
    TEXT_MUTED,
    TEXT_PRIMARY,
    BG_SURFACE,
    BG_ELEVATED,
    TIER_PALETTE,
    badge,
    inject_css,
    section_title,
    style_fig,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JouleRoute — Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

_RESULTS = _REPO_ROOT / "results"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(name: str) -> pd.DataFrame | None:
    p = _RESULTS / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _load_json(name: str) -> dict | None:
    p = _RESULTS / name
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _not_measured(label: str = "") -> None:
    """Standard 'not measured yet' card."""
    msg = f"Not measured yet{' — ' + label if label else ''}."
    st.info(f"🔬 {msg}  \nRun the benchmark scripts to populate this section.", icon="🔬")


def _preliminary_badge(data: dict | None) -> str:
    """Return 'preliminary' badge if the data dict says so, else 'measured'."""
    if data and data.get("preliminary", False):
        return badge("preliminary")
    return badge("measured")


# ─────────────────────────────────────────────────────────────────────────────
# TIER COLOUR HELPERS  (LITE green, MID yellow, FULL orange — matches Home.py)
# ─────────────────────────────────────────────────────────────────────────────
_TIER_COLOR = {
    "lite": "#4CAF50",   # green  (TIER_COLORS from data.py)
    "mid":  "#FFC107",   # yellow
    "full": "#FF9800",   # orange
}


def _tier_color(tier: str) -> str:
    return _TIER_COLOR.get(str(tier).lower(), TEXT_MUTED)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
title_col, meta_col = st.columns([3, 1])
with title_col:
    st.markdown("# 📊 JouleRoute — Evaluation")
with meta_col:
    st.markdown(
        f'<div style="text-align:right;padding-top:14px;color:{TEXT_MUTED};font-size:0.78rem;">'
        f'<span style="color:{ACCENT_TEAL};font-weight:600;">● EVALUATION</span>'
        f"&nbsp;&nbsp;<code>results/</code></div>",
        unsafe_allow_html=True,
    )

st.caption(
    "All numbers on this page come from files in `results/`. "
    "Sections without a source file show a 'not measured yet' card."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Scorecard
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(section_title("🏆", "Problem-Statement Scorecard"), unsafe_allow_html=True)

_tier_speeds   = _load_csv("tier_speeds.csv")
_spec_df       = _load_csv("spec_decoding.csv")
_acc_summary   = _load_json("accuracy_summary.json")
_routed_summ   = _load_json("routed_summary.json")
_router_lat    = _load_json("router_latency.json")
_energy_est    = _load_json("energy_estimate.json")
_battery_drain = _load_csv("battery_drain.csv")

def _scorecard_row(
    col,
    letter: str,
    title: str,
    desc: str,
    done: bool,
    partial: bool,
    evidence: str,
) -> None:
    if done:
        status_color = ACCENT_TEAL
        status_icon  = "✓ Done"
    elif partial:
        status_color = ACCENT_AMBER
        status_icon  = "◑ Partial"
    else:
        status_color = TEXT_MUTED
        status_icon  = "○ Pending"

    col.markdown(
        f'<div style="background:{BG_SURFACE};border:1px solid {BORDER};'
        f'border-radius:12px;padding:16px 18px;height:100%;">'
        f'<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;'
        f'text-transform:uppercase;color:{TEXT_MUTED};">Req {letter}</div>'
        f'<div style="font-size:1rem;font-weight:600;color:{TEXT_PRIMARY};margin:4px 0 2px;">'
        f'{title}</div>'
        f'<div style="font-size:0.80rem;color:{TEXT_MUTED};margin-bottom:10px;">{desc}</div>'
        f'<div style="font-size:0.78rem;font-weight:600;color:{status_color};">'
        f'{status_icon}</div>'
        f'<div style="font-size:0.75rem;color:{TEXT_MUTED};margin-top:6px;">'
        f'<em>{evidence}</em></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

sc1, sc2, sc3, sc4 = st.columns(4)

# (a) Two or more tiers + routing
_a_done    = _tier_speeds is not None and _routed_summ is not None
_a_partial = _tier_speeds is not None or _routed_summ is not None
if _tier_speeds is not None:
    n_tiers = _tier_speeds["tier"].nunique() if "tier" in _tier_speeds.columns else 0
    _a_ev = f"{n_tiers} tier(s) measured in tier_speeds.csv; routed_summary.json {'present' if _routed_summ else 'missing'}."
else:
    _a_ev = "tier_speeds.csv missing; routed_summary.json missing."
_scorecard_row(sc1, "A", "≥ 2 tiers + routing",
               "Multi-tier inference with a routing/gating mechanism",
               _a_done, _a_partial, _a_ev)

# (b) Battery / latency savings vs always-full
_b_done    = (_routed_summ is not None and
              all(k in _routed_summ for k in ("time_saved_pct", "accuracy_loss_pp")))
_b_partial = _routed_summ is not None or _energy_est is not None
if _routed_summ is not None:
    ts = _routed_summ.get("time_saved_pct")
    _b_ev = f"time_saved_pct = {ts}%" if ts is not None else "routed_summary.json present but time_saved_pct missing."
elif _energy_est is not None:
    _b_ev = "energy_estimate.json present; routed_summary.json missing."
else:
    _b_ev = "routed_summary.json and energy_estimate.json both missing."
_scorecard_row(sc2, "B", "Battery & latency savings",
               "Measured energy and latency reduction vs always-FULL",
               _b_done, _b_partial, _b_ev)

# (c) Cheap routing decision
_c_done    = (_router_lat is not None and
              "latency_ms_median" in _router_lat and
              "backend" in _router_lat)
_c_partial = _router_lat is not None
if _router_lat is not None:
    med = _router_lat.get("latency_ms_median")
    _c_ev = f"Router latency median = {med} ms ({_router_lat.get('backend','?')})."
else:
    _c_ev = "router_latency.json missing."
_scorecard_row(sc3, "C", "Cheap routing",
               "BERT-Tiny router ~1 ms, measured overhead < 1 % of LITE",
               _c_done, _c_partial, _c_ev)

# (d) Accuracy trade-off
_d_done    = (_acc_summary is not None and
              "accuracy" in _acc_summary and
              "harmful_downgrades" in _acc_summary)
_d_partial = _acc_summary is not None
if _acc_summary is not None:
    acc = _acc_summary.get("accuracy", {})
    _d_ev = f"accuracy keys: {list(acc.keys()) if isinstance(acc, dict) else acc}; harmful_downgrades present: {'harmful_downgrades' in _acc_summary}."
else:
    _d_ev = "accuracy_summary.json missing."
_scorecard_row(sc4, "D", "Accuracy trade-off",
               "Per-tier accuracy, harmful-downgrade analysis, τ sweep",
               _d_done, _d_partial, _d_ev)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Tier speeds
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(section_title("⚡", "Tier Speeds"), unsafe_allow_html=True)

if _tier_speeds is None:
    _not_measured("run bench/build_energy_profile.py to generate tier_speeds.csv")
else:
    st.markdown(badge("measured"), unsafe_allow_html=True)
    st.caption("Source: `results/tier_speeds.csv` · median over 3 runs · prompt b = 256-token generation task")

    # We only want the 256-token generation rows (prompt == "b") for tok/s;
    # fall back to all rows if "prompt" column is absent.
    if "prompt" in _tier_speeds.columns:
        df_b = _tier_speeds[_tier_speeds["prompt"] == "b"].copy()
        df_a = _tier_speeds[_tier_speeds["prompt"] == "a"].copy()
    else:
        df_b = _tier_speeds.copy()
        df_a = pd.DataFrame()

    tier_order = ["lite", "mid", "full"]

    ts_col1, ts_col2 = st.columns(2)

    # Chart A — median tok/s (generation speed)
    with ts_col1:
        st.markdown("**Decode speed — median tok/s** (256-token task)")
        df_plot = df_b.copy() if not df_b.empty else _tier_speeds.copy()
        if "tok_s_median" in df_plot.columns and "tier" in df_plot.columns:
            df_plot["tier_l"] = df_plot["tier"].str.lower()
            df_plot = df_plot[df_plot["tier_l"].isin(tier_order)]
            df_plot = df_plot.set_index("tier_l").reindex(tier_order).reset_index()
            df_plot = df_plot.dropna(subset=["tok_s_median"])

            fig_tok = go.Figure(go.Bar(
                x=[t.upper() for t in df_plot["tier_l"]],
                y=df_plot["tok_s_median"],
                marker=dict(color=[_tier_color(t) for t in df_plot["tier_l"]], opacity=0.88),
                text=[f"{v:.1f}" for v in df_plot["tok_s_median"]],
                textposition="outside",
                textfont=dict(color=TEXT_MUTED, size=11),
                customdata=list(zip(
                    df_plot["tok_s_median"],
                    df_plot.get("tok_s_min", [None]*len(df_plot)).fillna(0),
                    df_plot.get("tok_s_max", [None]*len(df_plot)).fillna(0),
                )),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Median: %{customdata[0]:.2f} tok/s<br>"
                    "Min: %{customdata[1]:.2f} tok/s<br>"
                    "Max: %{customdata[2]:.2f} tok/s<extra></extra>"
                ),
            ))
            style_fig(fig_tok, height=280, yaxis_title="tok/s", xaxis_title="Tier",
                      show_legend=False)
            st.plotly_chart(fig_tok, use_container_width=True, config=CHART_CONFIG)
        else:
            st.info("tok_s_median column not found in tier_speeds.csv.")

    # Chart B — total_ms_median
    with ts_col2:
        st.markdown("**Total generation time — median ms** (256-token task)")
        df_plot2 = df_b.copy() if not df_b.empty else _tier_speeds.copy()
        if "total_ms_median" in df_plot2.columns and "tier" in df_plot2.columns:
            df_plot2["tier_l"] = df_plot2["tier"].str.lower()
            df_plot2 = df_plot2[df_plot2["tier_l"].isin(tier_order)]
            df_plot2 = df_plot2.set_index("tier_l").reindex(tier_order).reset_index()
            df_plot2 = df_plot2.dropna(subset=["total_ms_median"])

            fig_total = go.Figure(go.Bar(
                x=[t.upper() for t in df_plot2["tier_l"]],
                y=df_plot2["total_ms_median"],
                marker=dict(color=[_tier_color(t) for t in df_plot2["tier_l"]], opacity=0.88),
                text=[f"{v/1000:.1f} s" for v in df_plot2["total_ms_median"]],
                textposition="outside",
                textfont=dict(color=TEXT_MUTED, size=11),
                customdata=list(zip(
                    df_plot2["total_ms_median"],
                    df_plot2.get("total_ms_min", [None]*len(df_plot2)).fillna(0),
                    df_plot2.get("total_ms_max", [None]*len(df_plot2)).fillna(0),
                    df_plot2.get("ttft_ms_median", [None]*len(df_plot2)).fillna(0),
                )),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Total median: %{customdata[0]:.0f} ms<br>"
                    "Min: %{customdata[1]:.0f} ms<br>"
                    "Max: %{customdata[2]:.0f} ms<br>"
                    "TTFT median: %{customdata[3]:.0f} ms<extra></extra>"
                ),
            ))
            style_fig(fig_total, height=280, yaxis_title="ms", xaxis_title="Tier",
                      show_legend=False)
            st.plotly_chart(fig_total, use_container_width=True, config=CHART_CONFIG)
        else:
            st.info("total_ms_median column not found in tier_speeds.csv.")

    # Raw table for all prompts
    with st.expander("Raw tier_speeds.csv"):
        show_cols = [c for c in (
            "tier", "prompt", "ttft_ms_median", "total_ms_median",
            "tokens_median", "tok_s_median", "tok_s_min", "tok_s_max",
        ) if c in _tier_speeds.columns]
        st.dataframe(_tier_speeds[show_cols], use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Accuracy per tier and category
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(section_title("🎯", "Accuracy per Tier & Category"), unsafe_allow_html=True)

if _acc_summary is None:
    _not_measured("run bench/analyze.py to generate accuracy_summary.json")
else:
    st.markdown(badge("measured"), unsafe_allow_html=True)
    st.caption("Source: `results/accuracy_summary.json`")

    acc_data = _acc_summary.get("accuracy", {})

    # ── Per-tier accuracy bar ──────────────────────────────────────────────
    if isinstance(acc_data, dict) and acc_data:
        tier_keys = [k for k in tier_order if k in acc_data]
        if tier_keys:
            acc_col1, acc_col2 = st.columns([1, 1])
            with acc_col1:
                st.markdown("**Per-tier accuracy**")
                acc_vals = [acc_data[k] for k in tier_keys]
                fig_acc = go.Figure(go.Bar(
                    x=[k.upper() for k in tier_keys],
                    y=acc_vals,
                    marker=dict(color=[_tier_color(k) for k in tier_keys], opacity=0.88),
                    text=[f"{v:.1%}" for v in acc_vals],
                    textposition="outside",
                    textfont=dict(color=TEXT_MUTED, size=11),
                    customdata=acc_vals,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Accuracy: %{customdata:.1%}<extra></extra>"
                    ),
                ))
                style_fig(fig_acc, height=260, yaxis_title="Accuracy",
                          xaxis_title="Tier", show_legend=False)
                fig_acc.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1.15]))
                st.plotly_chart(fig_acc, use_container_width=True, config=CHART_CONFIG)

            # ── Harmful downgrade callout ──────────────────────────────────
            with acc_col2:
                hd = _acc_summary.get("harmful_downgrades")
                if hd is not None:
                    st.markdown("**Harmful downgrades** (LITE/MID wrong, FULL right)")
                    st.markdown(
                        f'<div style="background:{BG_SURFACE};border:1px solid {ACCENT_AMBER};'
                        f'border-radius:12px;padding:20px 24px;text-align:center;">'
                        f'<div style="font-size:2.4rem;font-weight:700;color:{ACCENT_AMBER};">'
                        f'{hd}</div>'
                        f'<div style="font-size:0.82rem;color:{TEXT_MUTED};margin-top:4px;">'
                        f'queries where routing chose LITE/MID but only FULL answered correctly'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "A harmful downgrade is a query the cheaper tier got wrong "
                        "that FULL would have answered correctly. Lower is better."
                    )
                else:
                    st.info("harmful_downgrades key not found in accuracy_summary.json.")

    # ── Per-category accuracy ──────────────────────────────────────────────
    cat_data = _acc_summary.get("per_category", {})
    if cat_data and isinstance(cat_data, dict):
        st.markdown("**Per-category accuracy**")
        # cat_data expected as {category: {tier: accuracy, ...}, ...}
        # or {category: accuracy} — handle both
        first_val = next(iter(cat_data.values()))
        if isinstance(first_val, dict):
            # Stacked grouped bar per category, one bar per tier
            cats = list(cat_data.keys())
            fig_cat = go.Figure()
            for tier in tier_order:
                vals = [cat_data[c].get(tier) for c in cats]
                if any(v is not None for v in vals):
                    fig_cat.add_trace(go.Bar(
                        name=tier.upper(),
                        x=cats,
                        y=[v if v is not None else 0 for v in vals],
                        marker=dict(color=_tier_color(tier), opacity=0.85),
                        customdata=vals,
                        hovertemplate=(
                            f"<b>%{{x}} — {tier.upper()}</b><br>"
                            "Accuracy: %{y:.1%}<extra></extra>"
                        ),
                    ))
            style_fig(fig_cat, height=300, yaxis_title="Accuracy",
                      xaxis_title="Category", hovermode="x unified")
            fig_cat.update_layout(barmode="group",
                                  yaxis=dict(tickformat=".0%", range=[0, 1.15]))
            st.plotly_chart(fig_cat, use_container_width=True, config=CHART_CONFIG)
        else:
            # Flat {category: accuracy}
            cats  = list(cat_data.keys())
            vals  = list(cat_data.values())
            fig_cat2 = go.Figure(go.Bar(
                x=cats,
                y=vals,
                marker=dict(color=ACCENT, opacity=0.85),
                customdata=vals,
                hovertemplate="<b>%{x}</b><br>Accuracy: %{customdata:.1%}<extra></extra>",
            ))
            style_fig(fig_cat2, height=280, yaxis_title="Accuracy",
                      xaxis_title="Category", show_legend=False)
            fig_cat2.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1.15]))
            st.plotly_chart(fig_cat2, use_container_width=True, config=CHART_CONFIG)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Routed vs always-full
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(section_title("🔀", "Routed vs Always-FULL"), unsafe_allow_html=True)

if _routed_summ is None:
    _not_measured("run bench/analyze.py to generate routed_summary.json")
else:
    prelim = _routed_summ.get("preliminary", False)
    _badge_label = "preliminary" if prelim else "measured"
    st.markdown(_preliminary_badge(_routed_summ), unsafe_allow_html=True)
    if prelim:
        st.caption("⚠️ Marked preliminary — results may change when full dataset is used.")
    st.caption("Source: `results/routed_summary.json`")

    rv_col1, rv_col2, rv_col3, rv_col4 = st.columns(4)

    acc_loss = _routed_summ.get("accuracy_loss_pp")
    with rv_col1:
        st.metric(
            "Accuracy loss",
            f"{acc_loss:+.2f} pp" if acc_loss is not None else "—",
        )
        st.caption("percentage points vs always-FULL")

    time_saved = _routed_summ.get("time_saved_pct")
    with rv_col2:
        st.metric(
            "Time saved",
            f"{time_saved:.1f}%" if time_saved is not None else "—",
        )
        st.caption("vs always-FULL (median total_ms)")

    fb_rate = _routed_summ.get("fallback_rate_pct")
    with rv_col3:
        st.metric(
            "Fallback rate",
            f"{fb_rate:.1f}%" if fb_rate is not None else "—",
        )
        st.caption("queries escalated to FULL by router")

    n_q = _routed_summ.get("n_queries")
    with rv_col4:
        st.metric("Queries evaluated", n_q if n_q is not None else "—")
        st.caption("total queries in this run")

    # Tier distribution donut
    tier_dist = _routed_summ.get("tier_distribution", {})
    if tier_dist:
        st.markdown("**Tier distribution (routed policy)**")
        td_labels = [k.upper() for k in tier_dist.keys()]
        td_values = list(tier_dist.values())
        td_colors = [_tier_color(k) for k in tier_dist.keys()]
        fig_td = go.Figure(go.Pie(
            labels=td_labels,
            values=td_values,
            marker=dict(colors=td_colors, line=dict(color="#0F172A", width=2)),
            textinfo="label+percent",
            textposition="inside",
            textfont=dict(size=11, color="#F1F5F9"),
            hole=0.55,
            insidetextorientation="radial",
            hovertemplate="<b>%{label}</b><br>Queries: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        style_fig(fig_td, height=260, show_legend=False)
        st.plotly_chart(fig_td, use_container_width=True, config=CHART_CONFIG)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Router cost
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(section_title("🧭", "Router Cost"), unsafe_allow_html=True)

if _router_lat is None:
    _not_measured("run bench/analyze.py to generate router_latency.json")
else:
    st.markdown(badge("measured"), unsafe_allow_html=True)
    st.caption("Source: `results/router_latency.json`")

    rl_col1, rl_col2, rl_col3, rl_col4 = st.columns(4)

    med_ms  = _router_lat.get("latency_ms_median")
    p95_ms  = _router_lat.get("latency_ms_p95")
    backend = _router_lat.get("backend", "—")
    pct_of_lite = _router_lat.get("pct_of_lite_request")

    with rl_col1:
        st.metric("Median latency", f"{med_ms:.2f} ms" if med_ms is not None else "—")
        st.caption("router inference only")
    with rl_col2:
        st.metric("p95 latency", f"{p95_ms:.2f} ms" if p95_ms is not None else "—")
        st.caption("router inference only")
    with rl_col3:
        st.metric("Backend", backend)
        st.caption("ONNX / CPU / GPU")
    with rl_col4:
        st.metric(
            "% of one LITE request",
            f"{pct_of_lite:.2f}%" if pct_of_lite is not None else "—",
        )
        st.caption("overhead vs cheapest tier")

    # Bar: median vs p95
    if med_ms is not None or p95_ms is not None:
        fig_rl = go.Figure()
        if med_ms is not None:
            fig_rl.add_trace(go.Bar(
                x=["Median"],
                y=[med_ms],
                name="Median",
                marker=dict(color=ACCENT, opacity=0.88),
                text=[f"{med_ms:.2f} ms"],
                textposition="outside",
                textfont=dict(color=TEXT_MUTED, size=11),
                hovertemplate="<b>Router median</b><br>%{y:.2f} ms<extra></extra>",
            ))
        if p95_ms is not None:
            fig_rl.add_trace(go.Bar(
                x=["p95"],
                y=[p95_ms],
                name="p95",
                marker=dict(color=ACCENT_AMBER, opacity=0.88),
                text=[f"{p95_ms:.2f} ms"],
                textposition="outside",
                textfont=dict(color=TEXT_MUTED, size=11),
                hovertemplate="<b>Router p95</b><br>%{y:.2f} ms<extra></extra>",
            ))
        style_fig(fig_rl, height=220, yaxis_title="ms", xaxis_title="",
                  show_legend=False)
        st.plotly_chart(fig_rl, use_container_width=True, config=CHART_CONFIG)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Energy and battery
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(section_title("🔋", "Energy & Battery"), unsafe_allow_html=True)

en_col1, en_col2 = st.columns(2)

# energy_estimate.json — always ESTIMATED
with en_col1:
    st.markdown("**Energy estimate** (per-tier, per-request)")
    if _energy_est is None:
        _not_measured("run bench/build_energy_profile.py to generate energy_estimate.json")
    else:
        st.markdown(badge("estimated"), unsafe_allow_html=True)
        st.caption(
            "Source: `results/energy_estimate.json` · "
            "Values are placeholder estimates until the hardware energy meter is built."
        )
        # Expected shape: {tier: {joules_per_request: float, joules_per_token: float, ...}}
        # or flat {tier: joules}
        if isinstance(_energy_est, dict):
            first = next(iter(_energy_est.values())) if _energy_est else None
            if isinstance(first, dict):
                tiers_e = [k for k in tier_order if k in _energy_est]
                j_req   = [_energy_est[t].get("joules_per_request", 0) for t in tiers_e]
                j_tok   = [_energy_est[t].get("joules_per_token", 0)   for t in tiers_e]
                fig_en = go.Figure()
                fig_en.add_trace(go.Bar(
                    name="J / request",
                    x=[t.upper() for t in tiers_e],
                    y=j_req,
                    marker=dict(color=[_tier_color(t) for t in tiers_e], opacity=0.88),
                    customdata=list(zip(j_req, j_tok)),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "J/request: %{customdata[0]:.3f} J<br>"
                        "J/token: %{customdata[1]:.4f} J<extra></extra>"
                    ),
                ))
                style_fig(fig_en, height=240, yaxis_title="Joules",
                          xaxis_title="Tier", show_legend=False)
                st.plotly_chart(fig_en, use_container_width=True, config=CHART_CONFIG)
            else:
                # flat {tier: joules}
                tiers_e = [k for k in tier_order if k in _energy_est]
                j_vals  = [_energy_est[k] for k in tiers_e]
                fig_en2 = go.Figure(go.Bar(
                    x=[t.upper() for t in tiers_e],
                    y=j_vals,
                    marker=dict(color=[_tier_color(t) for t in tiers_e], opacity=0.88),
                    customdata=j_vals,
                    hovertemplate="<b>%{x}</b><br>Energy: %{customdata:.3f} J<extra></extra>",
                ))
                style_fig(fig_en2, height=240, yaxis_title="Joules",
                          xaxis_title="Tier", show_legend=False)
                st.plotly_chart(fig_en2, use_container_width=True, config=CHART_CONFIG)
        else:
            st.info("Unexpected format in energy_estimate.json.")

# battery_drain.csv — MEASURED (coarse)
with en_col2:
    st.markdown("**Battery drain** (measured coarse)")
    if _battery_drain is None:
        _not_measured("run bench/build_energy_profile.py to generate battery_drain.csv")
    else:
        st.markdown(badge("measured"), unsafe_allow_html=True)
        st.caption(
            "Source: `results/battery_drain.csv` · "
            "Coarse battery % readings; resolution depends on hardware."
        )
        if "policy" in _battery_drain.columns and "drain_pct" in _battery_drain.columns:
            fig_bd = go.Figure(go.Bar(
                x=_battery_drain["policy"],
                y=_battery_drain["drain_pct"],
                marker=dict(color=ACCENT_TEAL, opacity=0.85),
                customdata=list(zip(
                    _battery_drain["policy"],
                    _battery_drain["drain_pct"],
                    _battery_drain.get("duration_s", [None]*len(_battery_drain)).fillna(0),
                )),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Drain: %{customdata[1]:.2f}%<br>"
                    "Duration: %{customdata[2]:.0f} s<extra></extra>"
                ),
            ))
            style_fig(fig_bd, height=240, yaxis_title="Battery drain (%)",
                      xaxis_title="Policy", show_legend=False)
            st.plotly_chart(fig_bd, use_container_width=True, config=CHART_CONFIG)
        else:
            st.dataframe(_battery_drain, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Speculative decoding
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(section_title("🚀", "Speculative Decoding"), unsafe_allow_html=True)

if _spec_df is None:
    _not_measured("run bench/run_bench.py --spec to generate spec_decoding.csv")
else:
    st.markdown(badge("measured"), unsafe_allow_html=True)
    # Caption: tok/s in this CSV = output_tokens / total_ms * 1000 (end-to-end,
    # TTFT included).  Verified: 180 / 23282.8 ms × 1000 = 7.73 tok/s (spec OFF) ✓
    st.caption(
        "Source: `results/spec_decoding.csv` · FULL tier only · "
        "tok/s = output tokens / total time (end to end, includes time to first token) · "
        "median over 3 runs."
    )

    sp_col1, sp_col2 = st.columns([1, 1])

    with sp_col1:
        show_cols = [c for c in (
            "spec", "runs", "ttft_ms_median", "total_ms_median",
            "tokens_median", "tok_s_median",
        ) if c in _spec_df.columns]
        disp = _spec_df[show_cols].copy()
        # Thousands separators on large ms values; keep decimal precision
        for col in ("ttft_ms_median", "total_ms_median"):
            if col in disp.columns:
                disp[col] = disp[col].apply(
                    lambda v: f"{v:,.1f}" if pd.notna(v) else "—"
                )
        for col in ("tokens_median",):
            if col in disp.columns:
                disp[col] = disp[col].apply(
                    lambda v: f"{int(v):,}" if pd.notna(v) else "—"
                )
        for col in ("tok_s_median",):
            if col in disp.columns:
                disp[col] = disp[col].apply(
                    lambda v: f"{v:.2f}" if pd.notna(v) else "—"
                )
        st.dataframe(disp, use_container_width=True, hide_index=True)

    with sp_col2:
        if "tok_s_median" in _spec_df.columns and len(_spec_df) >= 2:
            on_row  = _spec_df[_spec_df["spec"] == "on"]
            off_row = _spec_df[_spec_df["spec"] == "off"]
            if not on_row.empty and not off_row.empty:
                tok_on  = float(on_row["tok_s_median"].iloc[0])
                tok_off = float(off_row["tok_s_median"].iloc[0])

                def _parse_list(val):
                    try:
                        import ast as _ast
                        return _ast.literal_eval(str(val))
                    except Exception:
                        return []

                on_all  = _parse_list(on_row["tok_s_all"].iloc[0])  if "tok_s_all" in on_row.columns  else []
                off_all = _parse_list(off_row["tok_s_all"].iloc[0]) if "tok_s_all" in off_row.columns else []

                # min/max from individual runs for hover text
                on_min  = min(on_all)  if on_all  else tok_on
                on_max  = max(on_all)  if on_all  else tok_on
                off_min = min(off_all) if off_all else tok_off
                off_max = max(off_all) if off_all else tok_off

                speedup   = tok_on / tok_off if tok_off > 0 else float("nan")
                direction = "faster" if speedup > 1 else "slower"

                # Raise y-axis ceiling so the "outside" text labels are not clipped
                y_max = max(tok_on, tok_off) * 1.30

                fig_sp = go.Figure()
                fig_sp.add_trace(go.Bar(
                    x=["Spec ON", "Spec OFF"],
                    y=[tok_on, tok_off],
                    marker=dict(
                        color=[ACCENT_AMBER if speedup < 1 else ACCENT_TEAL, ACCENT_TEAL],
                        opacity=0.88,
                    ),
                    text=[f"{tok_on:.2f}", f"{tok_off:.2f}"],
                    textposition="outside",
                    textfont=dict(color=TEXT_MUTED, size=11),
                    # customdata columns: [median, min, max]
                    customdata=[
                        [tok_on,  on_min,  on_max],
                        [tok_off, off_min, off_max],
                    ],
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Median: %{customdata[0]:.2f} tok/s<br>"
                        "Min: %{customdata[1]:.2f} tok/s<br>"
                        "Max: %{customdata[2]:.2f} tok/s<extra></extra>"
                    ),
                ))
                style_fig(fig_sp, height=260, yaxis_title="tok/s", show_legend=False)
                # update_yaxes runs AFTER style_fig so it wins over style_fig's defaults
                fig_sp.update_yaxes(range=[0, y_max])
                st.plotly_chart(fig_sp, use_container_width=True, config=CHART_CONFIG)

                # speedup = tok_on / tok_off; when < 1 ON is slower, ratio to show = 1/speedup
                ratio = (1 / speedup) if speedup < 1 else speedup
                st.markdown(
                    f"{badge('measured')} &nbsp; Spec ON is "
                    f"**{ratio:.1f}× {direction}** on this hardware.  \n"
                    f"Spec ON: {tok_on:.2f} tok/s &nbsp;·&nbsp; "
                    f"Spec OFF: {tok_off:.2f} tok/s",
                    unsafe_allow_html=True,
                )

    # TTFT comparison
    if "ttft_ms_median" in _spec_df.columns and len(_spec_df) >= 2:
        st.markdown("**Time-to-first-token (TTFT)**")
        on_row2  = _spec_df[_spec_df["spec"] == "on"]
        off_row2 = _spec_df[_spec_df["spec"] == "off"]
        if not on_row2.empty and not off_row2.empty:
            ttft_on  = float(on_row2["ttft_ms_median"].iloc[0])
            ttft_off = float(off_row2["ttft_ms_median"].iloc[0])
            total_on  = float(on_row2["total_ms_median"].iloc[0]) if "total_ms_median" in on_row2.columns else None
            total_off = float(off_row2["total_ms_median"].iloc[0]) if "total_ms_median" in off_row2.columns else None

            fig_ttft = go.Figure()
            fig_ttft.add_trace(go.Bar(
                name="TTFT",
                x=["Spec ON", "Spec OFF"],
                y=[ttft_on, ttft_off],
                marker=dict(color=ACCENT, opacity=0.80),
                customdata=[[ttft_on, total_on], [ttft_off, total_off]],
                hovertemplate=(
                    "<b>%{x} — TTFT</b><br>"
                    "TTFT: %{customdata[0]:.0f} ms<br>"
                    "Total: %{customdata[1]:.0f} ms<extra></extra>"
                ),
            ))
            if total_on is not None and total_off is not None:
                fig_ttft.add_trace(go.Bar(
                    name="Total",
                    x=["Spec ON", "Spec OFF"],
                    y=[total_on, total_off],
                    marker=dict(color=ACCENT_AMBER, opacity=0.55),
                    customdata=[[ttft_on, total_on], [ttft_off, total_off]],
                    hovertemplate=(
                        "<b>%{x} — Total</b><br>"
                        "TTFT: %{customdata[0]:.0f} ms<br>"
                        "Total: %{customdata[1]:.0f} ms<extra></extra>"
                    ),
                ))
            style_fig(fig_ttft, height=220, yaxis_title="ms",
                      hovermode="x unified")
            fig_ttft.update_layout(barmode="overlay")
            st.plotly_chart(fig_ttft, use_container_width=True, config=CHART_CONFIG)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<p style="color:{TEXT_MUTED};font-size:0.75rem;text-align:center;">'
    f"JouleRoute · Evaluation page &nbsp;·&nbsp; "
    f"{badge('measured')} from benchmark scripts &nbsp; "
    f"{badge('estimated')} placeholder energy weights &nbsp; "
    f"{badge('preliminary')} early run, may change"
    f"</p>",
    unsafe_allow_html=True,
)
