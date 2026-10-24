"""
serving/llm_client.py
Streaming client for the three llama-servers.

Public API
----------
stream(tier, messages, max_tokens, cfg) -> Iterator[TokenEvent]

    Yields one TokenEvent per token.  The final event is a sentinel:
        text=""  finish_reason="stop"|"length"
    All non-sentinel events have finish_reason=None.
    Existing callers that only read .text are unaffected by the sentinel.

token_entropy(top_logprobs) -> float
    Normalized entropy (0–1) over the returned top-k probabilities.

complete(tier, messages, max_tokens, cfg) -> str
    Blocking; returns full reply text.

Logprobs field names (confirmed from raw server response):
    choices[0].logprobs.content[0].top_logprobs
    Each entry: {"id": int, "token": str, "logprob": float,
                 "bytes": list[int], "top_logprobs": [...]}

finish_reason field (confirmed from raw server response):
    Final SSE chunk: choices[0].finish_reason = "stop" | "length"
    Final chunk has choices[0].delta = {} (no content).

Timeout behaviour:
    connect/write/pool timeouts are set to 10 s each (fast fail if server
    is not reachable).  The per-read timeout is disabled so a slow but
    still-streaming answer is never cut off mid-token.  A separate
    wall-clock deadline (server.request_timeout_s, default 180 s) kills
    the whole request if no [DONE] arrives within that window.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import httpx

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.config import load_config
from shared.tiers import Tier

log = logging.getLogger("serving.llm_client")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class LogProbEntry:
    token: str
    logprob: float
    bytes: list[int] | None = field(default=None)


@dataclass
class TokenEvent:
    """One streamed token with its top-k log-probabilities.

    finish_reason is None for every normal token.  The iterator always
    appends one final sentinel event with text="" and finish_reason set to
    the server's value ("stop" or "length").  Callers that only read .text
    see an empty string for the sentinel and are unaffected.
    """
    text: str
    top_logprobs: list[LogProbEntry] = field(default_factory=list)
    finish_reason: str | None = field(default=None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TOP_K = 5          # number of alternative tokens to request


def _base_url(tier: Tier, cfg: dict) -> str:
    host = cfg["gateway"]["host"]
    port = cfg["models"][tier.name.lower()]["port"]
    return f"http://{host}:{port}"


def _parse_logprobs(raw: list[dict[str, Any]] | None) -> list[LogProbEntry]:
    """Convert the raw top_logprobs list to LogProbEntry objects."""
    if not raw:
        return []
    result: list[LogProbEntry] = []
    for entry in raw:
        result.append(
            LogProbEntry(
                token=entry.get("token", ""),
                logprob=float(entry.get("logprob", float("-inf"))),
                bytes=entry.get("bytes"),
            )
        )
    return result


def _build_payload(
    messages: list[dict],
    max_tokens: int,
    seed: int,
    stream: bool,
) -> dict:
    return {
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,   # greedy; reproducible
        "seed": seed,
        "stream": stream,
        # Request top-k logprobs per token (OpenAI-compatible field names)
        "logprobs": True,
        "top_logprobs": _TOP_K,
    }


def _check_ctx_size(cfg: dict) -> None:
    """
    Warn if ctx_size is too small to reach the max_tokens cap.

    Rule: ctx_size should be >= max_tokens + 512 (rough room for the prompt).
    If not, the context fills before the cap is hit, so the cap is meaningless.
    Logs a WARNING and prints a recommendation but does NOT change any value.
    """
    srv = cfg.get("server", {})
    ctx_size: int = srv.get("ctx_size", 0)
    max_tokens: int = srv.get("max_tokens", -1)

    if ctx_size <= 0 or max_tokens <= 0:
        return   # can't evaluate; skip

    min_ctx = max_tokens + 512
    if ctx_size < min_ctx:
        # Recommend the next round power-of-two above min_ctx
        recommended = 1
        while recommended < min_ctx:
            recommended *= 2

        # Extra RAM per server: KV cache size ≈ ctx_size × layers × 2 × dtype_bytes
        # Q4_K_M Qwen3.5 has 28 layers; f16 KV = 2 bytes per element × 2 (K+V)
        # Each element is head_dim (128) × n_heads (varies, ~14 for 0.8B, ~16 for 4B)
        # Rough upper bound: 4 bytes/token/layer × 28 layers × Δctx tokens ÷ 1024² MB
        delta_ctx = recommended - ctx_size
        extra_mb_per_server = delta_ctx * 28 * 4 / (1024 * 1024)

        msg = (
            f"ctx_size ({ctx_size}) < max_tokens ({max_tokens}) + 512 = {min_ctx}. "
            f"The {max_tokens}-token cap will not be reached because the context fills "
            f"first. Recommended ctx_size: {recommended}. "
            f"Extra RAM per server: ~{extra_mb_per_server:.1f} MB "
            f"(+{extra_mb_per_server * 3:.0f} MB total for three servers)."
        )
        log.warning(msg)
        # Also print so it's visible in the self-test even without logging
        print(f"WARNING: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def token_entropy(top_logprobs: list[LogProbEntry]) -> float:
    """
    Normalized entropy over the returned top-k log-probabilities.

    Steps:
      1. Convert each logprob to a probability with exp().
      2. Renormalize so they sum to 1 (top-k may not cover the full distribution).
      3. Compute Shannon entropy: H = -sum(p * log(p)).
      4. Divide by log(k) to normalize to [0, 1].

    Returns 0.0 if fewer than 2 entries are present.
    """
    k = len(top_logprobs)
    if k < 2:
        return 0.0

    probs = [math.exp(e.logprob) for e in top_logprobs]
    total = sum(probs)
    if total <= 0.0:
        return 0.0

    probs = [p / total for p in probs]
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    return entropy / math.log(k)


def stream(
    tier: Tier,
    messages: list[dict],
    max_tokens: int | None = None,
    cfg: dict | None = None,
) -> Iterator[TokenEvent]:
    """
    Stream tokens from the given tier's llama-server.

    Yields one TokenEvent per token.  The final event is always a sentinel:
        text=""  finish_reason="stop"|"length"
    All prior events have finish_reason=None.

    Raises StreamTimeoutError if no [DONE] arrives within request_timeout_s.

    Parameters
    ----------
    tier        Which server to target.
    messages    OpenAI-format messages list.
    max_tokens  Override config default (-1 = no cap).
    cfg         Config dict; loaded from config.yaml if None.
    """
    if cfg is None:
        cfg = load_config()

    if max_tokens is None:
        max_tokens = cfg["server"].get("max_tokens", -1)

    seed = cfg.get("seed", 42)
    timeout_s: float = cfg.get("server", {}).get("request_timeout_s", 180)

    payload = _build_payload(messages, max_tokens, seed, stream=True)
    url = _base_url(tier, cfg) + "/v1/chat/completions"

    log.debug("stream -> %s  max_tokens=%d  timeout=%ss", url, max_tokens, timeout_s)

    # Connect/write/pool have a 10-s hard timeout (fast fail if not reachable).
    # Per-read timeout is disabled so a slow but still-streaming answer is never
    # killed mid-token.  We enforce the total wall-clock limit ourselves.
    http_timeout = httpx.Timeout(
        connect=10.0,
        read=None,      # no per-read timeout — tokens can arrive slowly
        write=10.0,
        pool=10.0,
    )

    finish_reason: str | None = None
    tokens_received: int = 0
    deadline: float = time.monotonic() + timeout_s if timeout_s > 0 else float("inf")

    with httpx.Client(timeout=http_timeout) as client:
        with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                # Wall-clock deadline check
                if time.monotonic() > deadline:
                    raise StreamTimeoutError(
                        tier=tier,
                        tokens_received=tokens_received,
                        timeout_s=timeout_s,
                    )

                line = raw_line.strip()
                if not line:
                    continue
                if line == "data: [DONE]":
                    break
                if not line.startswith("data: "):
                    continue

                json_str = line[len("data: "):]
                try:
                    chunk = json.loads(json_str)
                except json.JSONDecodeError:
                    log.warning("Could not decode SSE chunk: %s", json_str[:120])
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]

                # Final chunk: choices[0].finish_reason = "stop" | "length"
                # choices[0].delta = {} (confirmed from raw server response)
                fr = choice.get("finish_reason")
                if fr is not None:
                    finish_reason = fr
                    break

                delta = choice.get("delta", {})
                text: str = delta.get("content") or ""

                # Parse logprobs
                lp_data = choice.get("logprobs") or {}
                content_lp = lp_data.get("content") or []
                top_lp: list[LogProbEntry] = []
                if content_lp:
                    top_lp = _parse_logprobs(
                        content_lp[0].get("top_logprobs") if content_lp else None
                    )

                if text:
                    tokens_received += 1

                yield TokenEvent(text=text, top_logprobs=top_lp)

    # Always yield a sentinel so callers can read finish_reason
    yield TokenEvent(text="", top_logprobs=[], finish_reason=finish_reason)


def complete(
    tier: Tier,
    messages: list[dict],
    max_tokens: int | None = None,
    cfg: dict | None = None,
) -> str:
    """
    Non-streaming convenience wrapper.  Returns the full reply text.
    Logprobs and finish_reason are discarded; use stream() if you need them.
    """
    if cfg is None:
        cfg = load_config()

    if max_tokens is None:
        max_tokens = cfg["server"].get("max_tokens", -1)

    seed = cfg.get("seed", 42)
    payload = _build_payload(messages, max_tokens, seed, stream=False)
    url = _base_url(tier, cfg) + "/v1/chat/completions"

    timeout_s: float = cfg.get("server", {}).get("request_timeout_s", 180)
    http_timeout = httpx.Timeout(connect=10.0, read=timeout_s or None,
                                  write=10.0, pool=10.0)

    log.debug("complete -> %s  max_tokens=%d", url, max_tokens)

    resp = httpx.post(url, json=payload, timeout=http_timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StreamTimeoutError(Exception):
    """Raised when the total streaming wall-clock deadline is exceeded."""

    def __init__(self, tier: Tier, tokens_received: int, timeout_s: float) -> None:
        self.tier = tier
        self.tokens_received = tokens_received
        self.timeout_s = timeout_s
        super().__init__(
            f"Stream from {tier.name} timed out after {timeout_s:.0f}s "
            f"({tokens_received} tokens received so far)."
        )


# ---------------------------------------------------------------------------
# Quick smoke test (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os

    # UTF-8 stdout so Windows cmd does not crash on special characters
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Quick smoke-test for llm_client.py")
    parser.add_argument("--tier", choices=["lite", "mid", "full"], default="lite")
    parser.add_argument(
        "--prompt", default="In one sentence, what is the capital of France?",
    )
    parser.add_argument("--config", default=str(_REPO_ROOT / "config.yaml"))
    parser.add_argument(
        "--max-tokens", dest="max_tokens", type=int, default=None,
        metavar="N",
        help="Max tokens (-1 = no cap). Default: server.max_tokens from config.yaml.",
    )
    parser.add_argument(
        "--show-probs", action="store_true",
        help="Print top-1 logprob after each token.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _verbose = args.verbose or bool(os.environ.get("LLMCLIENT_DEBUG"))
    logging.getLogger("serving.llm_client").setLevel(
        logging.DEBUG if _verbose else logging.INFO
    )

    _cfg = load_config(args.config)

    # Context-size check (prints a WARNING if ctx_size is too small)
    _check_ctx_size(_cfg)

    _tier = Tier[args.tier.upper()]
    _messages = [{"role": "user", "content": args.prompt}]
    _max_tokens: int = (
        args.max_tokens
        if args.max_tokens is not None
        else _cfg["server"].get("max_tokens", -1)
    )

    # Show the exact request body
    _payload = _build_payload(
        _messages, max_tokens=_max_tokens, seed=_cfg.get("seed", 42), stream=True,
    )
    print("\nRequest body sent to server:")
    print(json.dumps(_payload, indent=2))
    print(f"\nStreaming from {_tier.name} (prompt: {args.prompt!r}):\n")

    lp_counts: list[int] = []
    _finish_reason: str | None = None
    _t_start = time.perf_counter()
    _t_first: float | None = None
    _t_end: float = _t_start

    try:
        for ev in stream(_tier, _messages, max_tokens=_max_tokens, cfg=_cfg):
            if ev.finish_reason is not None:
                _finish_reason = ev.finish_reason
                _t_end = time.perf_counter()
                continue

            sys.stdout.write(ev.text)
            sys.stdout.flush()

            if ev.text and _t_first is None:
                _t_first = time.perf_counter()

            if args.show_probs and ev.top_logprobs:
                best = ev.top_logprobs[0]
                sys.stdout.write(f"  [top={best.token!r} lp={best.logprob:.3f}]\n")
                sys.stdout.flush()

            if ev.top_logprobs:
                lp_counts.append(len(ev.top_logprobs))
            elif ev.text:
                lp_counts.append(0)

    except StreamTimeoutError as exc:
        sys.stdout.write("\n")
        print(f"\nERROR: {exc}")
        sys.exit(1)

    sys.stdout.write("\n")

    # --- Summary line ---
    _tokens = len(lp_counts)
    _total_s = _t_end - _t_start
    _ttft_s = (_t_first - _t_start) if _t_first else 0.0
    _tok_s = _tokens / _total_s if _total_s > 0 and _tokens > 0 else 0.0
    print(
        f"\n[{_tier.name} | {_tokens} tokens (streamed) | {_tok_s:.1f} tok/s"
        f" | first token {_ttft_s:.1f} s | total {_total_s:.1f} s"
        f" | finish: {_finish_reason}]"
    )

    if _finish_reason == "length":
        print("WARNING: output was cut off — max_tokens cap was hit.")

    if lp_counts:
        mn, mx = min(lp_counts), max(lp_counts)
        print(f"top_logprobs per token: min={mn} max={mx}  (over {_tokens} tokens)")
        if mn < _TOP_K:
            print(f"WARNING: expected {_TOP_K} logprobs per token but got as few as {mn}.")
    else:
        print("WARNING: no logprob data received from server.")
