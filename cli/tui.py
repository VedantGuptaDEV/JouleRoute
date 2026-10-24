"""Full-screen JouleRoute chat (Textual).

A centred conversation column: your messages on the right, replies on the left headed
by the model that answered. Each reply carries a one-line routing trace (click it, or
ctrl+o, to see all six pipeline stages), the streamed answer, and a metadata line.
"""
import re
import threading
import time

import requests
from rich.table import Table
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.suggester import SuggestFromList
from textual.theme import Theme
from textual.widgets import Input, Markdown, Static

from cli.client import GatewayClient
from cli.render import (ACCENT, DIM, ERR, FAINT, FG, LINE, MUTED, OK, TIER_STYLE, WARN, TurnState, header,
                        meta, pipeline_table, trace)
from shared.prompts import CATEGORIES

MODELS = ["auto", "lite", "mid", "full"]
MODES = ["saver", "balanced", "performance"]
COMMANDS = {
    "/model": "auto, lite, mid or full. Route automatically or force a tier",
    "/mode": "saver, balanced or performance. Gateway τ preset",
    "/category": "a prompt template name, or none",
    "/context": "on or off. Send the whole conversation each turn",
    "/tokens": "a number, or off. Cap answer length",
    "/status": "gateway, router and model health",
    "/clear": "clear the conversation",
    "/help": "this list",
    "/exit": "quit",
}
KEYS = [("enter", "send"), ("↑ ↓", "previous prompts"), ("→", "accept suggestion"), ("esc", "stop the answer"),
        ("ctrl+o", "expand the last trace"), ("ctrl+l", "clear"), ("ctrl+c twice", "quit")]
SUGGESTIONS = (list(COMMANDS) + [f"/model {m}" for m in MODELS] + [f"/mode {m}" for m in MODES]
               + [f"/category {c}" for c in [*CATEGORIES, "none"]] + ["/context on", "/context off", "/tokens off"])

THEME = Theme(
    name="joule", dark=True,
    primary=ACCENT, secondary=ACCENT, accent=ACCENT,
    foreground=FG, background="#09090b", surface="#141416", panel=LINE,
    success=OK, warning=WARN, error=ERR,
)


def keep_line_breaks(text: str) -> str:
    """Markdown joins single newlines; LLM answers (lists, poems) mean them. Code fences are left alone."""
    parts = re.split(r"(```.*?(?:```|$))", text, flags=re.DOTALL)
    return "".join(p if p.startswith("```") else re.sub(r"(?<!\n)\n(?!\n)", "  \n", p) for p in parts)


def two_columns(rows: list[tuple], key_style: str = MUTED) -> Table:
    t = Table.grid(padding=(0, 3))
    t.add_column(style=key_style, no_wrap=True)
    t.add_column(style=FAINT)
    for row in rows:
        t.add_row(*row)
    return t


class Prompt(Input):
    """Input with ↑/↓ history."""

    BINDINGS = [Binding("up", "history(-1)", show=False), Binding("down", "history(1)", show=False)]

    def __init__(self, **kw):
        super().__init__(**kw)
        self.history: list[str] = []
        self.pos = 0

    def remember(self, line: str):
        if line and (not self.history or self.history[-1] != line):
            self.history.append(line)
        self.pos = len(self.history)

    def action_history(self, step: int):
        if not self.history:
            return
        self.pos = max(0, min(len(self.history), self.pos + step))
        self.value = self.history[self.pos] if self.pos < len(self.history) else ""
        self.cursor_position = len(self.value)


class Trace(Static):
    """The one-line routing trace; click to expand the full pipeline."""

    def on_click(self, event: events.Click):
        self.parent.toggle()


class Reply(Vertical):
    """Assistant side of one turn: header, trace, (detail), answer, meta."""

    def __init__(self, models: dict[str, str]):
        super().__init__(classes="reply")
        self.models = models
        self.state = TurnState()
        self.parts: list[str] = []
        self.routing: dict | None = None
        self.error: str | None = None
        self.open = False
        self.dirty = False
        self.done = False

    def compose(self) -> ComposeResult:
        yield Static(classes="head")
        yield Trace(classes="trace")
        yield Static(classes="detail")
        yield Markdown(classes="md")
        yield Static(classes="meta")

    def on_mount(self):
        self.head = self.query_one(".head", Static)
        self.trace_line = self.query_one(".trace", Trace)
        self.detail = self.query_one(".detail", Static)
        self.md = self.query_one(".md", Markdown)
        self.meta_line = self.query_one(".meta", Static)
        self.detail.display = self.md.display = self.meta_line.display = False
        self.state.start(time.monotonic())
        self.ticker = self.set_interval(1 / 12, self.tick)
        self.tick()

    def tick(self):
        now = time.monotonic()
        self.head.update(header(self.state, self.models, now, error=bool(self.error)))
        self.trace_line.update(trace(self.state, now, self.open))
        if self.open:
            self.detail.update(pipeline_table(self.state, now))
        if self.dirty:
            self.dirty = False
            self.md.update(keep_line_breaks("".join(self.parts)))

    def toggle(self):
        self.open = not self.open
        self.detail.display = self.open
        self.tick()

    def handle(self, kind: str, payload):
        if self.done:  # stopped by the user; late events from the network thread are dropped
            return
        if kind == "step":
            self.state.apply(payload, time.monotonic())
        elif kind == "token":
            self.parts.append(payload)
            self.md.display = True
            self.dirty = True
        elif kind == "routing":
            self.routing = payload
            self.error = payload.get("error")

    def finish(self, note: Text | None = None):
        self.done = True
        self.state.finish(time.monotonic())
        self.ticker.stop()
        self.tick()
        if note is None and self.routing and not self.error:
            self.meta_line.update(meta(self.routing, self.state))
        else:
            self.meta_line.update(note or Text(self.error or "no response", ERR))
        self.meta_line.display = True

    @property
    def ok(self) -> bool:
        return bool(self.routing and not self.error and self.parts)


class JouleApp(App):
    TITLE = "JouleRoute"
    CSS = """
    Screen { background: $background; }

    #topbar { dock: top; height: 2; padding: 0 2; border-bottom: solid $panel; background: $background; }
    #brand { width: 1fr; }
    #health { width: auto; }

    #chat { padding: 0 2; align-horizontal: center; scrollbar-size-vertical: 1;
            scrollbar-color: $panel; scrollbar-color-hover: #3f3f46; scrollbar-background: $background; }
    #chat.-empty { align-vertical: middle; }

    #empty { height: auto; width: 100%; align-horizontal: center; }
    #empty > Static { width: auto; height: auto; margin-bottom: 1; }

    .user-row { height: auto; align-horizontal: right; margin: 1 0; }
    .bubble { width: auto; max-width: 80%; height: auto; padding: 1 2; background: $surface; }

    .reply { height: auto; margin-bottom: 1; }
    .head { height: 1; }
    .trace { height: 1; margin-bottom: 1; }
    .trace:hover { background: $surface; }
    .detail { height: auto; margin: 0 0 1 2; padding: 0 2; border-left: solid $panel; }
    .md { height: auto; margin: 0; padding: 0; background: $background; }
    .md > *:last-child { margin-bottom: 0; }
    .md MarkdownH1, .md MarkdownH2, .md MarkdownH3, .md MarkdownH4 {
        color: $foreground; background: $background; text-style: bold; content-align: left top;
        border: none; padding: 0; margin: 1 0 1 0; }
    .md MarkdownFence { background: $surface; margin: 1 0; }
    .meta { height: auto; margin-top: 1; }

    .note { height: auto; margin: 1 0; padding: 0 2; border: round $panel; }
    .line { height: auto; margin: 0 0 1 0; }

    #composer { dock: bottom; height: auto; padding: 0 2 0 2; align-horizontal: center; background: $background; }
    #prompt { border: round $panel; background: $background; padding: 0 1; }
    #prompt:focus { border: round $primary; }
    #footer { height: 1; padding: 0 1; margin-bottom: 0; }
    #footer-left { width: 1fr; }
    #footer-right { width: auto; }
    """
    BINDINGS = [
        Binding("ctrl+c", "ctrl_c", priority=True, show=False),
        Binding("escape", "interrupt", show=False),
        Binding("ctrl+o", "toggle_trace", show=False),
        Binding("ctrl+l", "clear", show=False),
        Binding("ctrl+d", "quit", show=False),
    ]

    def __init__(self, url: str, model: str = "auto", category: str | None = None, context: bool = False,
                 max_tokens: int | None = None):
        super().__init__()
        self.client = GatewayClient(url)
        self.model = model
        self.category = category
        self.context = context
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self.status: dict | None = None
        self.status_loaded = False
        self.busy = False
        self.turn: tuple[Reply, threading.Event] | None = None  # the reply in flight and its cancel flag
        self.last_ctrl_c = 0.0

    # ------------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static(id="brand")
            yield Static(id="health")
        with VerticalScroll(id="chat"):
            with Vertical(id="empty"):
                yield Static(id="empty-title")
                yield Static(id="empty-models")
                yield Static(id="empty-foot")
        with Vertical(id="composer"):
            yield Prompt(placeholder="Message JouleRoute", id="prompt",
                         suggester=SuggestFromList(SUGGESTIONS, case_sensitive=False))
            with Horizontal(id="footer"):
                yield Static(id="footer-left")
                yield Static(id="footer-right")

    def on_mount(self):
        self.register_theme(THEME)
        self.theme = "joule"
        self.chat = self.query_one("#chat", VerticalScroll)
        self.show_empty(True)
        self.prompt = self.query_one("#prompt", Prompt)
        self.prompt.focus()
        self.render_chrome()
        self.load_status()

    def models(self) -> dict[str, str]:
        return {k: t["model"] for k, t in (self.status or {}).get("tiers", {}).items()}

    def render_chrome(self):
        """Top bar, empty state and footer, all derived from the last /v1/status."""
        s = self.status
        self.query_one("#brand", Static).update(Text.assemble(("JouleRoute", f"bold {FG}"),
                                                              ("   local model router", FAINT)))
        health = Text()
        if s:
            for key, t in s["tiers"].items():
                health.append("● ", OK if t["healthy"] else ERR)
                health.append(f"{key}   ", FAINT)
            health.append(self.client.url.split("://")[-1], DIM)
        elif self.status_loaded:
            health.append("● ", ERR)
            health.append("gateway offline", FAINT)
        self.query_one("#health", Static).update(health)

        title = Text(justify="center")
        title.append("JouleRoute\n", f"bold {FG}")
        title.append("Each prompt goes to the smallest local model that can answer it.", FAINT)
        self.query_one("#empty-title", Static).update(title)

        models = Table.grid(padding=(0, 3))
        for _ in range(4):
            models.add_column(no_wrap=True)
        foot = Text(justify="center")
        if s:
            for key, t in s["tiers"].items():
                models.add_row(Text(key, TIER_STYLE[key]), Text(t["model"], FG), Text(f":{t['port']}", DIM),
                               Text("● ready", OK) if t["healthy"] else Text("● down", ERR))
            rt = s["router"]
            manual = s.get("tau_source") == "manual"
            foot.append(f"router {rt['name']}  ·  τ {s['tau']:.2f}", FAINT)
            foot.append(" manual" if manual else f" {s['mode']}", WARN if manual else FAINT)
            if manual:
                foot.append("\nτ is set by hand in config.yaml, so the accuracy-loss bound does not apply", DIM)
            if not all(t["healthy"] for t in s["tiers"].values()):
                foot.append("\nstart the models with .venv/bin/python serving/launch.py", WARN)
        elif self.status_loaded:
            foot.append(f"Can't reach the gateway at {self.client.url}\n", ERR)
            foot.append("start it with .venv/bin/python gateway/app.py, then /status", FAINT)
        else:
            foot.append("connecting…", FAINT)
        self.query_one("#empty-models", Static).update(models)
        self.query_one("#empty-foot", Static).update(foot)

        left = Text()
        left.append(self.model, TIER_STYLE.get(self.model, FG))
        if s:
            manual = s.get("tau_source") == "manual"
            left.append(f"   τ {s['tau']:.2f}", FAINT)
            left.append(" manual" if manual else f" {s['mode']}", FAINT)
        left.append("   " + (f"max {self.max_tokens} tokens" if self.max_tokens else "no length cap"), FAINT)
        left.append(f"   context {'on' if self.context else 'off'}", FAINT)
        if self.category:
            left.append(f"   {self.category}", FAINT)
        self.query_one("#footer-left", Static).update(left)
        self.query_one("#footer-right", Static).update(
            Text("esc stop   ctrl+o trace   /help" if not self.busy else "esc to stop", DIM))

    @work(thread=True, exclusive=True, group="status")
    def load_status(self, announce: bool = False):
        s = self.client.status()
        self.call_from_thread(self.apply_status, s, announce)

    def apply_status(self, s: dict | None, announce: bool = False):
        self.status, self.status_loaded = s, True
        self.render_chrome()
        if announce:
            self.show_status()

    def show_empty(self, empty: bool):
        """Centre the intro while the chat is empty; afterwards stick the view to the newest message."""
        self.query_one("#empty").display = empty
        self.chat.set_class(empty, "-empty")
        if not empty:
            self.chat.anchor()

    def note(self, renderable, boxed: bool = True):
        self.show_empty(False)
        self.chat.mount(Static(renderable, classes="note" if boxed else "line"))

    def say(self, msg: str, style: str = FAINT):
        self.note(Text(msg, style), boxed=False)

    # ------------------------------------------------------------------ chat

    async def on_input_submitted(self, event: Input.Submitted):
        line = event.value.strip()
        if not line:
            return
        if self.busy:
            self.notify("Still answering. Press esc to stop.", severity="warning", timeout=2)
            return
        self.prompt.remember(line)
        self.prompt.value = ""
        if line.startswith("/"):
            self.command(line)
            return
        self.show_empty(False)
        if not self.context:
            self.messages.clear()
        self.messages.append({"role": "user", "content": line})
        reply = Reply(self.models())
        await self.chat.mount(Horizontal(Static(line, classes="bubble"), classes="user-row"))
        await self.chat.mount(reply)
        cancel = threading.Event()
        self.turn = (reply, cancel)
        self.busy = True
        self.render_chrome()
        self.run_turn(reply, {"model": self.model, "messages": list(self.messages), "x_category": self.category,
                              "max_tokens": self.max_tokens}, cancel)

    @work(thread=True, group="chat")
    def run_turn(self, reply: Reply, body: dict, cancel: threading.Event):
        error = None
        try:
            for kind, payload in self.client.chat(body, cancel):
                if cancel.is_set():
                    break
                self.call_from_thread(reply.handle, kind, payload)
        except Exception as e:  # an aborted stream raises from inside requests; that is the interrupt
            if not cancel.is_set():
                error = f"Gateway request failed: {e}"
        if not cancel.is_set():
            self.call_from_thread(self.turn_done, reply, error)

    def turn_done(self, reply: Reply, error: str | None, stopped: bool = False):
        if reply.done:
            return
        if stopped:
            reply.finish(Text("Stopped. Not kept in the conversation.", FAINT))
        elif error:
            reply.error = error
            reply.finish(Text(error, ERR))
        else:
            reply.finish()
        if reply.ok and not stopped:
            self.messages.append({"role": "assistant", "content": "".join(reply.parts)})
        elif self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()
        self.turn = None
        self.busy = False
        self.render_chrome()

    # ------------------------------------------------------------------ commands

    def command(self, line: str):
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        if cmd in ("/exit", "/quit"):
            self.exit()
        elif cmd == "/help":
            self.note(two_columns([*COMMANDS.items(), ("", ""), *KEYS], FG))
        elif cmd == "/model":
            if arg in MODELS:
                self.model = arg
                self.say("Routing automatically." if arg == "auto" else f"Every prompt now goes to {arg.upper()}.")
            else:
                self.say(f"Model is {self.model}. Options: {', '.join(MODELS)}.")
        elif cmd == "/mode":
            if arg in MODES:
                try:
                    r = self.client.set_mode(arg)
                    extra = " The manual τ in config.yaml still overrides presets." if r.get("tau_source") == "manual" else ""
                    self.say(f"Mode {r['mode']}, τ {r['tau']:.2f}.{extra}")
                    self.load_status()
                except requests.RequestException as e:
                    self.say(f"Couldn't set the mode: {e}", ERR)
            else:
                self.say(f"Mode is {self.status['mode'] if self.status else 'unknown'}. Options: {', '.join(MODES)}.")
        elif cmd == "/category":
            if arg in CATEGORIES or arg == "none":
                self.category = None if arg == "none" else arg
                self.say(f"Category {arg}.")
            else:
                self.say(f"Category is {self.category or 'none'}. Options: {', '.join(CATEGORIES)}, none.")
        elif cmd == "/context":
            if arg in ("on", "off"):
                self.context = arg == "on"
                self.messages.clear()
                self.say("The whole conversation is now sent each turn." if self.context
                         else "Each prompt now stands alone, routed and cached on its own.")
            else:
                self.say(f"Context is {'on' if self.context else 'off'}. Options: on, off.")
        elif cmd == "/tokens":
            if arg == "off":
                self.max_tokens = None
                self.say("No length cap. Answers end when the model is done or the context window is full.")
            elif arg.isdigit() and int(arg) >= 1:
                self.max_tokens = int(arg)
                self.say(f"Answers capped at {self.max_tokens} tokens.")
            else:
                self.say("Usage: /tokens <n> or /tokens off.")
        elif cmd == "/status":
            self.load_status(announce=True)
        elif cmd == "/clear":
            self.action_clear()
        else:
            self.say(f"Unknown command {cmd}. Try /help.", ERR)
        self.render_chrome()

    def show_status(self):
        s = self.status
        if s is None:
            self.say(f"Can't reach the gateway at {self.client.url}.", ERR)
            return
        rt = s["router"]
        manual = s.get("tau_source") == "manual"
        rows = [("gateway", Text(self.client.url, FG)),
                ("router", Text(f"{rt['name']}  {rt['backend']}  loaded in {rt['load_ms']:.0f} ms", FG)),
                ("threshold", Text(f"τ {s['tau']:.2f}  " + ("manual override, presets ignored" if manual
                                                          else f"{s['mode']}, τ_min {s['tau_min']:.2f}"), FG)),
                ("presets", Text("   ".join(f"{k} {v:.2f}" for k, v in s["mode_presets"].items()), FG))]
        for key, t in s["tiers"].items():
            rows.append((Text(key, TIER_STYLE[key]), Text.assemble((f"{t['model']}  :{t['port']}  ", FG),
                                                                  ("● ready", OK) if t["healthy"] else ("● down", ERR))))
        self.note(two_columns(rows))

    # ------------------------------------------------------------------ actions

    def action_interrupt(self):
        """Stop the reply now: the UI finishes it immediately, the network thread is woken
        without blocking, and the gateway (seeing the disconnect) stops the model."""
        if self.turn is None:
            return
        reply, cancel = self.turn
        cancel.set()
        self.client.abort()
        self.turn_done(reply, None, stopped=True)

    def action_toggle_trace(self):
        replies = self.chat.query(Reply)
        if replies:
            replies.last().toggle()

    def action_clear(self):
        if self.busy:
            return
        self.messages.clear()
        for w in self.chat.query(".reply, .user-row, .note, .line"):
            w.remove()
        self.show_empty(True)

    def action_ctrl_c(self):
        if self.busy:
            self.action_interrupt()
        elif self.prompt.value:
            self.prompt.value = ""
        elif time.monotonic() - self.last_ctrl_c < 1.5:
            self.exit()
        else:
            self.last_ctrl_c = time.monotonic()
            self.notify("Press ctrl+c again to quit", timeout=1.5)
