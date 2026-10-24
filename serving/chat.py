"""
serving/chat.py
Interactive chat loop for testing a single llama-server tier.

Usage:
    python -m serving.chat <lite|mid|full> [--max-tokens N] [--config PATH]

Type "exit" or "quit" to quit.  Ctrl+C exits cleanly.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.config import load_config
from shared.tiers import Tier
from serving.llm_client import stream

# Reconfigure stdout for UTF-8 so Windows cmd/PowerShell don't crash on
# special characters in model output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

log = logging.getLogger("serving.chat")


def chat_loop(tier: Tier, max_tokens: int, cfg: dict) -> None:
    """Read-stream-print loop until the user quits."""
    print(f"Chat with {tier.name}  (max_tokens={max_tokens if max_tokens != -1 else 'unlimited'})")
    print('Type "exit" or "quit" to quit.\n')

    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            return

        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        sys.stdout.write(f"{tier.name}: ")
        sys.stdout.flush()

        assistant_text: list[str] = []
        finish_reason: str | None = None
        tokens_out: int = 0
        t_start = time.perf_counter()
        t_first: float | None = None
        t_end = t_start

        try:
            for ev in stream(tier, history, max_tokens=max_tokens, cfg=cfg):
                if ev.finish_reason is not None:
                    finish_reason = ev.finish_reason
                    t_end = time.perf_counter()
                    continue
                if ev.text:
                    if t_first is None:
                        t_first = time.perf_counter()
                    tokens_out += 1
                sys.stdout.write(ev.text)
                sys.stdout.flush()
                assistant_text.append(ev.text)
        except httpx.ConnectError:
            print(
                f"\nError: cannot connect to {tier.name} server. "
                f"Is it running on port {cfg['models'][tier.name.lower()]['port']}?"
            )
            history.pop()
            continue
        except httpx.HTTPStatusError as exc:
            print(f"\nError: server returned {exc.response.status_code}.")
            history.pop()
            continue

        sys.stdout.write("\n")

        # Summary line
        total_s = t_end - t_start
        ttft_s = (t_first - t_start) if t_first else 0.0
        tok_s = tokens_out / total_s if total_s > 0 and tokens_out > 0 else 0.0
        print(
            f"[{tier.name} | {tokens_out} tokens (streamed) | {tok_s:.1f} tok/s"
            f" | first token {ttft_s:.1f} s | total {total_s:.1f} s"
            f" | finish: {finish_reason}]"
        )

        if finish_reason == "length":
            print("(Warning: response was cut off — max_tokens cap hit.)")

        print()

        # Add assistant reply to history for multi-turn context
        history.append({"role": "assistant", "content": "".join(assistant_text)})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive chat with a single llama-server tier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "tier",
        choices=["lite", "mid", "full"],
        help="Which tier to chat with.",
    )
    parser.add_argument(
        "--max-tokens", dest="max_tokens", type=int, default=None,
        metavar="N",
        help="Max tokens per response (-1 = no cap). Default: server.max_tokens from config.yaml.",
    )
    parser.add_argument(
        "--config", default=str(_REPO_ROOT / "config.yaml"),
        help="Path to config.yaml.",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading config: {exc}")
        sys.exit(1)

    _tier = Tier[args.tier.upper()]
    _max_tokens: int = (
        args.max_tokens
        if args.max_tokens is not None
        else cfg["server"].get("max_tokens", -1)
    )

    try:
        chat_loop(_tier, _max_tokens, cfg)
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
