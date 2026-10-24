"""JouleRoute chat: a full-screen, Claude Code style front end for the gateway.

Every message goes to the gateway URL (POST /v1/chat/completions, streaming). The six
pipeline stages it passes through (cache, privacy, router, gate, model, log) fill in live,
then the answer streams in, then a one-line receipt.

  .venv/bin/python cli/joule.py                       # full-screen chat
  .venv/bin/python cli/joule.py --url http://host:8000
  .venv/bin/python cli/joule.py -p "what is 17 * 23"  # one-shot: same layout, printed, then exit

Answers are not length-capped by default (they end when the model finishes, or when the
context window in server.ctx_size fills up); --max-tokens sets a cap.
"""
import argparse
import sys
import time
from pathlib import Path

import requests
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cli.client import GatewayClient  # noqa: E402
from cli.render import ERR, FG, TurnState, header, meta, pipeline_table, trace  # noqa: E402
from cli.tui import MODELS, JouleApp, keep_line_breaks  # noqa: E402
from shared.config import load_config  # noqa: E402
from shared.prompts import CATEGORIES  # noqa: E402

def print_once(url: str, body: dict, prompt: str) -> int:
    """Run one query and print it in the same layout as the full-screen app (trace expanded)."""
    console = Console(highlight=False)
    client = GatewayClient(url)
    status = client.status() or {}
    models = {k: t["model"] for k, t in status.get("tiers", {}).items()}
    state, parts, routing, error = TurnState(), [], None, None

    def view(final: bool = False):
        now = time.monotonic()
        rows = [Text(prompt, FG, justify="right"), Text(),
                header(state, models, now, error=bool(error or (routing or {}).get("error"))),
                trace(state, now, open_=True), Padding(pipeline_table(state, now), (1, 0, 1, 4))]
        if parts:
            rows.append(Markdown(keep_line_breaks("".join(parts))))
        if final:
            err = error or (routing or {}).get("error")
            rows += [Text(), Text(err, ERR) if err else meta(routing or {}, state)]
        return Group(*rows)

    state.start(time.monotonic())
    with Live(view(), console=console, refresh_per_second=12, vertical_overflow="visible") as live:
        try:
            for kind, payload in client.chat({**body, "messages": [{"role": "user", "content": prompt}]}):
                if kind == "step":
                    state.apply(payload, time.monotonic())
                elif kind == "token":
                    parts.append(payload)
                else:
                    routing = payload
                live.update(view())
        except requests.RequestException as e:
            error = f"Gateway request failed: {e}"
        except KeyboardInterrupt:
            error = "Stopped."
        state.finish(time.monotonic())
        live.update(view(final=True))
    console.print()
    return 1 if error or (routing or {}).get("error") else 0


def main():
    cfg = load_config()
    gw = cfg["gateway"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=f"http://{gw['host']}:{gw['port']}", help="gateway base URL")
    parser.add_argument("--model", choices=MODELS, default="auto", help="auto routes; others force a tier")
    parser.add_argument("--category", choices=CATEGORIES, help="prompt template to wrap queries in")
    parser.add_argument("--context", action="store_true", help="send the whole conversation each turn")
    parser.add_argument("--max-tokens", type=int, default=None, help="cap answer length (default: no cap)")
    parser.add_argument("-p", "--print", dest="prompt", help="send one prompt, print the result, and exit")
    args = parser.parse_args()

    if args.prompt:
        body = {"model": args.model, "x_category": args.category, "max_tokens": args.max_tokens}
        sys.exit(print_once(args.url, body, args.prompt))
    JouleApp(args.url, args.model, args.category, args.context, args.max_tokens).run()


if __name__ == "__main__":
    main()
