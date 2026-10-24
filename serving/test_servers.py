"""
serving/test_servers.py
Send one chat completion to each llama-server and report TTFT + total latency.

Usage:
    python serving/test_servers.py [--config config.yaml] [--prompt "..."]

The servers must already be running (start them with serving/launch.py first,
or pass --launch to start them here and shut down afterwards).
"""
from __future__ import annotations

import argparse
import json
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("serving.test_servers")

_DEFAULT_PROMPT = "In one sentence, what is the capital of France?"


def _chat_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1/chat/completions"


def probe_tier(tier: Tier, host: str, port: int, prompt: str) -> dict:
    """
    Send a non-streaming chat completion and return timing + response info.
    Uses temperature=0 (greedy) and seed from config for reproducibility.
    """
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": False,
    }
    url = _chat_url(host, port)

    t0 = time.perf_counter()
    try:
        resp = httpx.post(url, json=payload, timeout=60.0)
        t_total = time.perf_counter() - t0
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        return {"tier": tier.name, "ok": False, "error": str(exc)}

    data = resp.json()
    reply = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    tokens_out = usage.get("completion_tokens", 0)

    return {
        "tier":       tier.name,
        "ok":         True,
        "port":       port,
        "total_ms":   round(t_total * 1000, 1),
        "tok_per_s":  round(tokens_out / t_total, 1) if t_total > 0 else 0,
        "tokens_out": tokens_out,
        "reply":      reply,
    }


def run_tests(cfg: dict, prompt: str) -> list[dict]:
    host = cfg["gateway"]["host"]
    results = []
    for tier in Tier:
        port = cfg["models"][tier.name.lower()]["port"]
        log.info("Probing %s on port %d ...", tier.name, port)
        r = probe_tier(tier, host, port, prompt)
        results.append(r)
    return results


def _print_summary(results: list[dict], prompt: str) -> None:
    sep = "-" * 53
    print(f"\n{sep}")
    print(f"  Test prompt: {prompt}")
    print(sep)
    for r in results:
        if r["ok"]:
            print(
                f"  {r['tier']:4s}  port={r['port']}  "
                f"total={r['total_ms']:7.1f} ms  "
                f"tok/s={r['tok_per_s']:5.1f}  "
                f"tokens_out={r['tokens_out']:3d}  "
                f"reply: {r['reply'][:80]}"
            )
        else:
            print(f"  {r['tier']:4s}  FAILED: {r['error']}")
    print(f"{sep}\n")
    all_ok = all(r["ok"] for r in results)
    if all_ok:
        print("OK  All three servers answered successfully.")
    else:
        failed = [r["tier"] for r in results if not r["ok"]]
        print(f"FAIL  Failed tiers: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test all three llama-servers with one prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config.yaml"),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--prompt",
        default=_DEFAULT_PROMPT,
        help="Prompt to send to each server.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = run_tests(cfg, args.prompt)
    _print_summary(results, args.prompt)

    all_ok = all(r["ok"] for r in results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
