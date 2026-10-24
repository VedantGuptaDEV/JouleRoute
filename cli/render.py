"""Shared look for the full-screen app and `-p` print mode.

Each reply has the same parts in the same order:
  header   model name, tier pill, how it got there (routed / fallback / forced / cached)
  trace    one line: cache, privacy, router, gate (expands to all six pipeline rows)
  answer   the streamed markdown
  meta     tokens, speed, latency
Stages the gateway did not run are shown as skipped, so every reply reads the same way.
"""
from dataclasses import dataclass, field

from rich.table import Table
from rich.text import Text

# Neutral zinc greys, one indigo accent for focus and activity, tier colour only inside pills.
FG = "#e4e4e7"
MUTED = "#a1a1aa"
FAINT = "#71717a"
DIM = "#52525b"
LINE = "#27272a"
PILL_BG = "#202024"
ACCENT = "#818cf8"
OK = "#34d399"
WARN = "#fbbf24"
ERR = "#f87171"
TIER_STYLE = {"lite": "#34d399", "mid": "#fbbf24", "full": "#a78bfa"}
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

ROWS = ["cache", "privacy", "router", "gate", "model", "log"]
GATEWAY_TO_ROW = {"cache": "cache", "pii": "privacy", "router": "router", "gate": "gate",
                  "generate": "model", "telemetry": "log"}
VERBS = {"cache": "Checking cache", "privacy": "Scanning for PII", "router": "Routing", "gate": "Routing",
         "model": "Generating", "log": "Finishing"}


@dataclass
class Row:
    status: str = "pending"          # pending | running | done | skip | error
    detail: Text = field(default_factory=Text)
    ms: float | None = None
    started: float | None = None
    data: dict = field(default_factory=dict)


def spinner(now: float) -> str:
    return SPINNER[int(now * 12) % len(SPINNER)]


def tier_pill(key: str | None) -> Text:
    if not key:
        return Text("")
    return Text(f" {key.upper()} ", f"bold {TIER_STYLE.get(key, FG)} on {PILL_BG}")


def tier_word(key: str | None) -> Text:
    return Text(key.upper(), TIER_STYLE.get(key, FG)) if key else Text("—", FAINT)


def prob_bars(probs: list[float], predicted: str) -> Text:
    out = Text()
    for i, (key, p) in enumerate(zip(["lite", "mid", "full"], probs)):
        hot = key == predicted
        filled = round(p * 10)
        out.append("   " if i else "")
        out.append(f"{key} ", FG if hot else FAINT)
        out.append("▬" * filled, TIER_STYLE[key] if hot else FAINT)
        out.append("▬" * (10 - filled), LINE)
        out.append(f" {p:.2f}", FG if hot else FAINT)
    return out


def skip_reason(step: dict) -> str:
    detail = step.get("detail", "")
    return detail.split("bypassed: ", 1)[-1] if detail else "skipped"


def step_detail(step: dict) -> Text:
    """One-line description of a gateway step for the expanded trace."""
    name, status, d = step["name"], step["status"], step.get("data") or {}
    if status == "skip":
        return Text(f"skipped · {skip_reason(step)}", FAINT)
    if status == "error":
        return Text(step.get("detail") or "failed", ERR)
    if name == "cache":
        return (Text.assemble(("hit", OK), (" · answered earlier by ", FAINT), tier_word(d.get("tier")))
                if d.get("hit") else Text("miss", MUTED))
    if name == "pii":
        if d.get("flag"):
            return Text.assemble((", ".join(d["types"]), WARN), ("  kept on-device", FAINT))
        return Text("clean", MUTED)
    if name == "router":
        return prob_bars(d["probs"], d["predicted"])
    if name == "gate":
        op = "<" if d["fallback"] else "≥"
        out = Text(f"{d['confidence']:.2f} {op} τ {d['tau']:.2f}", MUTED)
        out.append(" manual" if d.get("tau_source") == "manual" else "", FAINT)
        out.append("   →  ", DIM)
        out.append_text(Text("fallback to FULL", WARN) if d["fallback"] else tier_word(d["tier"]))
        return out
    if name == "generate":
        out = Text(d.get("model", ""), MUTED)
        out.append(f"  :{d['port']}" if d.get("port") else "", FAINT)
        out.append("  speculative" if d.get("speculative") else "", FAINT)
        return out
    if name == "telemetry":
        return Text(f"row {d.get('row', '?')}  {d.get('db', '')}", FAINT)
    return Text(step.get("detail", ""), FAINT)


class TurnState:
    """Pipeline rows for one reply, updated from gateway events."""

    def __init__(self):
        self.rows = {name: Row() for name in ROWS}
        self.model_head: dict | None = None   # generate "start" step, merged into the done one
        self.forced = False
        self.cache_hit_tier: str | None = None

    def start(self, now: float):
        self._advance(now)

    def apply(self, step: dict, now: float):
        if step["name"] == "error":
            target = next((r for r in self.rows.values() if r.status in ("running", "pending")), self.rows["model"])
            target.status, target.detail, target.ms = "error", step_detail(step), None
            return
        row = self.rows.get(GATEWAY_TO_ROW.get(step["name"], ""))
        if row is None:
            return
        if step["name"] == "generate":
            if step["status"] == "start":
                self.model_head = step
                row.status, row.detail, row.started, row.data = "running", step_detail(step), now, step["data"]
                return
            step = {**step, "data": {**(self.model_head or {}).get("data", {}), **step.get("data", {})}}
        row.status = {"done": "done", "skip": "skip", "error": "error"}.get(step["status"], "done")
        row.detail, row.ms, row.data = step_detail(step), step.get("ms"), step.get("data") or {}
        if step["name"] == "router" and step["status"] == "skip":
            self.forced = True
            self._skip("gate", skip_reason(step))
        if step["name"] == "cache" and row.data.get("hit"):
            self.cache_hit_tier = row.data.get("tier")
            for name in ("privacy", "router", "gate", "model"):
                self._skip(name, "cache hit")
        self._advance(now)

    def finish(self, now: float):
        for row in self.rows.values():
            if row.status == "running":
                row.status, row.detail = "error", Text("interrupted", ERR)
            elif row.status == "pending":
                row.status, row.detail = "skip", Text("skipped", FAINT)

    @property
    def tier(self) -> str | None:
        return (self.model_head or {}).get("data", {}).get("tier") or self.cache_hit_tier

    @property
    def running(self) -> str | None:
        return next((name for name, r in self.rows.items() if r.status == "running"), None)

    def _skip(self, name: str, reason: str):
        row = self.rows[name]
        if row.status in ("pending", "running"):
            row.status, row.detail, row.ms = "skip", Text(f"skipped · {reason}", FAINT), None

    def _advance(self, now: float):
        """Mark the stage currently in progress (the first pending row) as running."""
        if self.running:
            return
        for row in self.rows.values():
            if row.status == "pending":
                row.status, row.started = "running", now
                break


def fmt_ms(ms: float | None) -> str:
    if ms is None:
        return ""
    if ms >= 1000:
        return f"{ms / 1000:.1f} s"
    return f"{ms:.1f} ms" if ms < 10 else f"{ms:.0f} ms"


def header(state: TurnState, models: dict[str, str], now: float, error: bool = False) -> Text:
    """`Qwen3.5-2B  MID  routed`, or a spinner with the current stage before a model is chosen."""
    tier = state.tier
    if tier is None:
        if error:
            return Text("Request failed", f"bold {ERR}")
        stage = state.running
        return Text.assemble((spinner(now) + " ", ACCENT), (VERBS.get(stage, "Waiting") + "…", MUTED))
    gate = state.rows["gate"].data
    how = ("cached" if state.cache_hit_tier else "forced" if state.forced
           else "fallback" if gate.get("fallback") else "routed")
    name = (state.model_head or {}).get("data", {}).get("model") or models.get(tier, tier.upper())
    out = Text.assemble((name, f"bold {FG}"), "  ", tier_pill(tier), ("  " + how, WARN if how == "fallback" else FAINT))
    if state.running == "model":
        out.append(f"   {spinner(now)}", ACCENT)
    return out


def trace(state: TurnState, now: float, open_: bool) -> Text:
    """One line: `▸ 3.1 ms   cache miss   privacy clean   router MID 0.37   gate pass`."""
    out = Text("▾ " if open_ else "▸ ", DIM)
    routing_ms = sum(state.rows[n].ms or 0 for n in ("cache", "privacy", "router"))
    out.append(f"{fmt_ms(routing_ms) or '—':>7}", FAINT)
    for name in ("cache", "privacy", "router", "gate"):
        row = state.rows[name]
        out.append(f"   {name} ", DIM)
        if row.status == "running":
            out.append(spinner(now), ACCENT)
        elif row.status in ("pending", "skip"):
            out.append("–", DIM)
        elif row.status == "error":
            out.append("error", ERR)
        elif name == "cache":
            out.append("hit" if row.data.get("hit") else "miss", OK if row.data.get("hit") else FAINT)
        elif name == "privacy":
            out.append(", ".join(row.data["types"]) if row.data.get("flag") else "clean",
                       WARN if row.data.get("flag") else FAINT)
        elif name == "router":
            out.append_text(Text.assemble(tier_word(row.data["predicted"]), (f" {max(row.data['probs']):.2f}", FAINT)))
        elif name == "gate":
            out.append("fallback" if row.data.get("fallback") else "pass", WARN if row.data.get("fallback") else FAINT)
    return out


def pipeline_table(state: TurnState, now: float) -> Table:
    """All six stages, for the expanded trace and print mode."""
    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column(width=1, no_wrap=True)
    grid.add_column(width=7, no_wrap=True)
    grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    grid.add_column(width=7, justify="right", no_wrap=True)
    for name, row in state.rows.items():
        icon = {"pending": Text("○", DIM), "running": Text(spinner(now), ACCENT), "done": Text("●", OK),
                "skip": Text("○", DIM), "error": Text("●", ERR)}[row.status]
        ms = fmt_ms(row.ms)
        if row.status == "running" and row.started is not None:
            ms = f"{now - row.started:.1f} s"
        grid.add_row(icon, Text(name, MUTED if row.status != "skip" else DIM), row.detail, Text(ms, FAINT))
    return grid


def meta(r: dict, state: TurnState) -> Text:
    """Line under the answer: same fields, same order, every reply."""
    parts: list[tuple[str, str]] = []
    if n := r.get("tokens_out"):
        parts.append((f"{n} token{'s' * (n != 1)}", FAINT))
    if tps := state.rows["model"].data.get("tok_s"):
        parts.append((f"{tps:.1f} tok/s", FAINT))
    if r.get("ttft_ms") is not None:
        parts.append((f"{fmt_ms(r['ttft_ms'])} to first token", FAINT))
    if r.get("total_ms") is not None:
        parts.append((f"{fmt_ms(r['total_ms'])} total", FAINT))
    parts.append(("energy not measured" if r.get("joules") is None else f"{r['joules']:.1f} J", DIM))
    if r.get("finish_reason") == "length":
        why = f"the {r['max_tokens']}-token limit" if r.get("max_tokens") else "a full context window"
        parts.append((f"cut off at {why}", WARN))
    out = Text()
    for i, (s, style) in enumerate(parts):
        out.append("  ·  " if i else "", DIM)
        out.append(s, style)
    return out
