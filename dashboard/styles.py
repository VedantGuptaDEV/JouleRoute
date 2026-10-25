"""
dashboard/styles.py — Single source of truth for all visual styling.

Call inject_css() at the top of every page (after st.set_page_config).
Call plotly_template() to get the shared Plotly layout dict for every chart.

DESIGN TOKENS — edit here and in .streamlit/config.toml to retheme:
  BG_BASE        #0F172A   deep navy, main page background
  BG_SURFACE     #1E293B   card and sidebar surface
  BG_ELEVATED    #263548   inputs, hover state
  ACCENT         #6366F1   indigo, primary interactive / focus
  ACCENT_TEAL    #14B8A6   teal, positive delta / success
  ACCENT_AMBER   #F59E0B   amber, warning
  TEXT_PRIMARY   #F1F5F9   headings, key values
  TEXT_MUTED     #94A3B8   captions, labels, secondary text
  BORDER         #334155   subtle borders / dividers
  RADIUS         12px      corner radius everywhere
  SHADOW         0 4px 24px rgba(0,0,0,0.35)

CHART PALETTE (fixed order, used in plotly_template):
  0  #6366F1  indigo  (LITE)
  1  #14B8A6  teal    (MID)
  2  #F59E0B  amber   (FULL)
  3  #94A3B8  slate   (other / neutral)
"""
from __future__ import annotations

import streamlit as st

# ── palette (mirrors .streamlit/config.toml) ─────────────────────────────────
BG_BASE      = "#0F172A"
BG_SURFACE   = "#1E293B"
BG_ELEVATED  = "#263548"
ACCENT       = "#6366F1"
ACCENT_TEAL  = "#14B8A6"
ACCENT_AMBER = "#F59E0B"
TEXT_PRIMARY = "#F1F5F9"
TEXT_MUTED   = "#94A3B8"
BORDER       = "#334155"

# Tier colour map — used by charts and badges
TIER_PALETTE: dict[str, str] = {
    "lite": ACCENT,        # indigo
    "mid":  ACCENT_TEAL,   # teal
    "full": ACCENT_AMBER,  # amber
}

# Chart colour sequence (Plotly colorway)
CHART_COLORS = [ACCENT, ACCENT_TEAL, ACCENT_AMBER, TEXT_MUTED, "#EC4899", "#8B5CF6"]

_CSS = f"""
<style>
/* ── Google Fonts: Inter ───────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── CSS design tokens ─────────────────────────────────────────────────────── */
:root {{
  --bg-base:      {BG_BASE};
  --bg-surface:   {BG_SURFACE};
  --bg-elevated:  {BG_ELEVATED};
  --accent:       {ACCENT};
  --accent-teal:  {ACCENT_TEAL};
  --accent-amber: {ACCENT_AMBER};
  --text-primary: {TEXT_PRIMARY};
  --text-muted:   {TEXT_MUTED};
  --border:       {BORDER};
  --radius:       12px;
  --shadow:       0 4px 24px rgba(0,0,0,0.35);
  --space-1:      8px;
  --space-2:      16px;
  --space-3:      24px;
  --space-4:      32px;
}}

/* ── Base typography ───────────────────────────────────────────────────────── */
html, body, [class*="css"] {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  color: var(--text-primary);
  background-color: var(--bg-base);
}}

/* ── Hide Streamlit chrome ─────────────────────────────────────────────────── */
#MainMenu, footer, header {{ visibility: hidden; height: 0; }}
.stDeployButton {{ display: none; }}
div[data-testid="stToolbar"] {{ display: none; }}

/* ── Main content area ─────────────────────────────────────────────────────── */
.main .block-container {{
  padding: var(--space-3) var(--space-4) var(--space-4);
  max-width: 1400px;
}}

/* ── Page title ────────────────────────────────────────────────────────────── */
h1 {{
  font-size: 2rem;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.02em;
  margin-bottom: 4px;
}}
h2 {{
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: -0.01em;
  margin-bottom: var(--space-2);
}}
h3 {{
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: var(--space-1);
}}

/* ── Caption / small text ──────────────────────────────────────────────────── */
.stCaption, small, [data-testid="stCaption"] {{
  color: var(--text-muted) !important;
  font-size: 0.78rem;
}}

/* ── Metric cards ──────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: var(--space-2) var(--space-2) var(--space-1);
  box-shadow: var(--shadow);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}}
[data-testid="stMetric"]:hover {{
  border-color: var(--accent);
  box-shadow: 0 0 0 2px rgba(99,102,241,0.25), var(--shadow);
}}
[data-testid="stMetricLabel"] > div {{
  font-size: 0.72rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted) !important;
}}
[data-testid="stMetricValue"] > div {{
  font-size: 1.65rem;
  font-weight: 700;
  color: var(--text-primary) !important;
  line-height: 1.2;
}}
[data-testid="stMetricDelta"] > div {{
  font-size: 0.82rem;
  font-weight: 500;
}}

/* ── Section containers / cards ────────────────────────────────────────────── */
.jr-card {{
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: var(--space-2) var(--space-3);
  box-shadow: var(--shadow);
  margin-bottom: var(--space-2);
}}
.jr-card-header {{
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  margin-bottom: var(--space-1);
  display: flex;
  align-items: center;
  gap: 6px;
}}

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] .block-container {{
  padding: var(--space-2);
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
  color: var(--text-primary);
}}
[data-testid="stSidebarNav"] {{
  padding-top: var(--space-1);
}}
[data-testid="stSidebarNav"] a {{
  border-radius: 8px;
  padding: 6px 12px;
  font-weight: 500;
  font-size: 0.88rem;
  color: var(--text-muted) !important;
  transition: background 0.15s, color 0.15s;
}}
[data-testid="stSidebarNav"] a:hover {{
  background: var(--bg-elevated);
  color: var(--text-primary) !important;
}}
[data-testid="stSidebarNav"] a[aria-current="page"] {{
  background: rgba(99,102,241,0.15);
  color: var(--accent) !important;
  font-weight: 600;
}}

/* ── Divider ───────────────────────────────────────────────────────────────── */
hr {{
  border: none;
  border-top: 1px solid var(--border);
  margin: var(--space-3) 0;
}}

/* ── Buttons ───────────────────────────────────────────────────────────────── */
.stButton > button {{
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: var(--radius);
  padding: 8px 20px;
  font-family: 'Inter', sans-serif;
  font-size: 0.88rem;
  font-weight: 600;
  letter-spacing: 0.01em;
  transition: opacity 0.2s, box-shadow 0.2s;
  box-shadow: 0 2px 8px rgba(99,102,241,0.30);
}}
.stButton > button:hover {{
  opacity: 0.88;
  box-shadow: 0 4px 16px rgba(99,102,241,0.45);
}}
.stButton > button:active {{
  opacity: 0.75;
}}

/* ── Text inputs, number inputs, text areas ────────────────────────────────── */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {{
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif;
  font-size: 0.88rem;
  transition: border-color 0.2s;
}}
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.20) !important;
}}

/* ── Selectbox ─────────────────────────────────────────────────────────────── */
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {{
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-primary) !important;
}}
.stSelectbox div[data-baseweb="select"] > div:focus-within,
.stMultiSelect div[data-baseweb="select"] > div:focus-within {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.20) !important;
}}

/* ── Tabs ──────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
  background: var(--bg-surface);
  border-radius: var(--radius);
  padding: 4px;
  gap: 4px;
  border-bottom: none !important;
}}
.stTabs [data-baseweb="tab"] {{
  border-radius: 8px;
  padding: 6px 16px;
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text-muted) !important;
  background: transparent;
  border: none !important;
  transition: background 0.15s, color 0.15s;
}}
.stTabs [aria-selected="true"] {{
  background: var(--accent) !important;
  color: #fff !important;
  font-weight: 600 !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* ── Dataframe ─────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden;
}}
[data-testid="stDataFrame"] th {{
  background: var(--bg-elevated) !important;
  color: var(--text-muted) !important;
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  border-bottom: 1px solid var(--border) !important;
}}
[data-testid="stDataFrame"] td {{
  font-size: 0.82rem;
  color: var(--text-primary) !important;
  border-bottom: 1px solid rgba(51,65,85,0.5) !important;
}}
[data-testid="stDataFrame"] tr:hover td {{
  background: var(--bg-elevated) !important;
}}

/* ── Info / warning / error alerts ────────────────────────────────────────── */
.stAlert {{
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  font-size: 0.85rem;
}}
div[data-baseweb="notification"] {{
  border-radius: var(--radius) !important;
}}

/* ── Spinner ───────────────────────────────────────────────────────────────── */
.stSpinner > div > div {{
  border-top-color: var(--accent) !important;
}}

/* ── Plotly chart containers ───────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {{
  border-radius: var(--radius);
  overflow: hidden;
}}

/* ── Badge helper (used by _badge() in Home.py) ────────────────────────────── */
.jr-badge {{
  display: inline-flex;
  align-items: center;
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  line-height: 1.6;
}}
.jr-badge-measured  {{ background: rgba(20,184,166,0.18); color: {ACCENT_TEAL}; border: 1px solid rgba(20,184,166,0.35); }}
.jr-badge-estimated {{ background: rgba(245,158,11,0.15); color: {ACCENT_AMBER}; border: 1px solid rgba(245,158,11,0.30); }}
.jr-badge-simulated {{ background: rgba(148,163,184,0.12); color: {TEXT_MUTED}; border: 1px solid rgba(148,163,184,0.25); }}
.jr-badge-preliminary {{ background: rgba(236,72,153,0.12); color: #EC4899; border: 1px solid rgba(236,72,153,0.28); }}
.jr-badge-computed    {{ background: rgba(99,102,241,0.14); color: {ACCENT}; border: 1px solid rgba(99,102,241,0.30); }}

/* ── Section header row ────────────────────────────────────────────────────── */
.jr-section-title {{
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.10em;
  color: var(--text-muted);
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 8px;
}}
.jr-section-title::before {{
  content: '';
  display: inline-block;
  width: 3px;
  height: 14px;
  background: var(--accent);
  border-radius: 2px;
  flex-shrink: 0;
}}

/* ── Scrollbar ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg-base); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #475569; }}

/* ── Responsive: tighten padding on narrow screens ─────────────────────────── */
@media (max-width: 768px) {{
  .main .block-container {{
    padding: var(--space-2) var(--space-1);
  }}
  h1 {{ font-size: 1.5rem; }}
}}
</style>
"""


def inject_css() -> None:
    """Inject the shared stylesheet. Call once at the top of every page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def badge(label: str) -> str:
    """
    Return an HTML badge string for inline rendering.
    label: 'measured' | 'estimated' | 'simulated' | 'preliminary' | 'computed'
    Usage:  st.markdown(badge('measured'), unsafe_allow_html=True)
    """
    cls_map = {
        "measured":    "jr-badge-measured",
        "estimated":   "jr-badge-estimated",
        "simulated":   "jr-badge-simulated",
        "preliminary": "jr-badge-preliminary",
        "computed":    "jr-badge-computed",
    }
    cls = cls_map.get(label.lower(), "jr-badge-simulated")
    return f'<span class="jr-badge {cls}">{label}</span>'


def section_title(icon: str, text: str) -> str:
    """Styled section header rendered via st.markdown."""
    return f'<div class="jr-section-title">{icon}&nbsp;{text}</div>'


def plotly_template() -> dict:
    """
    Base Plotly layout dict with palette, fonts, and background only.
    Does NOT include legend, margin, height, or axis titles so callers
    can pass those without getting a duplicate-keyword error.

    Prefer style_fig() over calling this directly — it applies everything
    including the standard legend position, margins, and gridline rules.
    """
    _grid_h = "rgba(148,163,184,0.15)"   # faint horizontal gridlines
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=12),
        title_font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=13),
        colorway=CHART_COLORS,
        xaxis=dict(
            showgrid=False,                     # no vertical gridlines
            linecolor=BORDER,
            tickfont=dict(color=TEXT_MUTED, size=11),
            title_font=dict(color=TEXT_MUTED, size=11),
            zeroline=False,
            automargin=True,
        ),
        yaxis=dict(
            gridcolor=_grid_h,                  # faint horizontal lines only
            gridwidth=1,
            griddash="dot",
            showgrid=True,
            linecolor=BORDER,
            tickfont=dict(color=TEXT_MUTED, size=11),
            title_font=dict(color=TEXT_MUTED, size=11),
            zeroline=False,
            automargin=True,
        ),
        hoverlabel=dict(
            bgcolor=BG_ELEVATED,
            bordercolor=BORDER,
            font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=12),
        ),
    )


# Pass to every st.plotly_chart() call.
# displayModeBar=True keeps zoom, pan, download-PNG, and legend toggles visible,
# matching the original committed behaviour (no config arg = modebar shown).
CHART_CONFIG: dict = {"displayModeBar": True}

# Standard legend position: horizontal row above the plot area.
_LEGEND_ABOVE: dict = dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="left",
    x=0,
    title_text="",
    font=dict(color=TEXT_MUTED, size=11),
    bgcolor="rgba(0,0,0,0)",
)

# Standard chart margins: room at top for the above-plot legend.
_MARGIN_STD: dict = dict(l=10, r=10, t=40, b=10)


def style_fig(fig, height: int = 320, *, yaxis_title: str = "",
              xaxis_title: str = "", show_legend: bool = True,
              hovermode: str = "closest") -> None:
    """
    Apply the full shared style to a Plotly Figure in-place.

    Parameters
    ----------
    fig         : go.Figure to mutate
    height      : chart height in pixels (default 320)
    yaxis_title : optional y-axis label
    xaxis_title : optional x-axis label (pass "" to omit)
    show_legend : set False for charts that don't need a legend (e.g. donut)
    hovermode   : Plotly hovermode; use "x unified" for multi-line time charts,
                  "closest" (default) for bar/scatter, "x" for histograms.
                  Never pass staticPlot or any setting that disables interaction.

    Usage
    -----
        fig = go.Figure(...)
        # add traces ...
        style_fig(fig, yaxis_title="Joules", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
    """
    legend_cfg = _LEGEND_ABOVE if show_legend else dict(visible=False)

    fig.update_layout(
        **plotly_template(),
        title_text="",          # never show an internal Plotly title
        height=height,
        margin=_MARGIN_STD,
        legend=legend_cfg,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode=hovermode,
        # hoverlabel is already supplied by plotly_template(); do not repeat it here
        # or update_layout raises "multiple values for keyword argument 'hoverlabel'".
    )
