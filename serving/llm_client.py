"""
serving/llm_client.py
Streaming client for the three llama-servers.

Public API
----------
stream(tier, messages, max_tokens, cfg) -> Iterator[TokenEvent]

TokenEvent has:
    text        str       – decoded text of this token
    top_logprobs list[LogProbEntry]  – up to k=5 entries, may be empty

Also exposes:
    complete(tier, messages, max_tokens, cfg) -> str
    (blocking, returns full reply text; useful for simple tests)

Logprobs field names follow the OpenAI streaming format:
  choices[0].logprobs.content[0].top_logprobs
Each entry: {"token": str, "logprob": float, "bytes": list[int] | null}

llama-server (>= b3447) returns logprobs when:
  "logprobs": true, "top_logprobs": k  (k 0..20)

References:
  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
"""
from __future__ import annotations

import json
import logging
import sys
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
    """One streamed token with its top-k log-probabilities."""
    text: str
    top_logprobs: list[LogProbEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TOP_K = 5          # number of alternative tokens to request


def _base_url(tier: Tier, cfg: dict) -> str:
    host = cfg["gateway"]["host"]
    port = cfg["models"][tier.name.lower()]["port"]
    return f"http://{host}:{port}"


def _parse_logprobs(raw: list[dict[str, Any]] | None) -> list[LogProbEntry]:
    """Convert the raw OpenAI top_logprobs list to LogProbEntry objects."""
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stream(
    tier: Tier,
    messages: list[dict],
    max_tokens: int | None = None,
    cfg: dict | None = None,
) -> Iterator[TokenEvent]:
    """
    Stream tokens from the given tier's llama-server.

    Yields a TokenEvent for each SSE data chunk that contains a token.
    The last chunk (finish_reason != null) yields an event with empty text
    and no logprobs if the server does not include a token there.

    Parameters
    ----------
    tier : Tier
        Which server to target.
    messages : list[dict]
        OpenAI-format messages, e.g. [{"role":"user","content":"hello"}].
    max_tokens : int | None
        Override config default.
    cfg : dict | None
        Config dict from load_config().  Loaded from config.yaml if None.
    """
    if cfg is None:
        cfg = load_config()

    if max_tokens is None:
        max_tokens = cfg["server"].get("max_tokens", 256)

    seed = cfg.get("seed", 42)
    payload = _build_payload(messages, max_tokens, seed, stream=True)
    url = _base_url(tier, cfg) + "/v1/chat/completions"

    log.debug("stream → %s  max_tokens=%d", url, max_tokens)

    with httpx.Client(timeout=None) as client:
        with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
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
                delta = choice.get("delta", {})
                text: str = delta.get("content") or ""

                # Parse logprobs if present
                lp_data = choice.get("logprobs") or {}
                content_lp = lp_data.get("content") or []
                top_lp: list[LogProbEntry] = []
                if content_lp:
                    # content is a list of per-token objects; we only care about
                    # the first one (this delta is exactly one token)
                    top_lp = _parse_logprobs(
                        content_lp[0].get("top_logprobs") if content_lp else None
                    )

                yield TokenEvent(text=text, top_logprobs=top_lp)


def complete(
    tier: Tier,
    messages: list[dict],
    max_tokens: int | None = None,
    cfg: dict | None = None,
) -> str:
    """
    Non-streaming convenience wrapper.  Returns the full reply text.
    Logprobs are discarded; use stream() if you need them.
    """
    if cfg is None:
        cfg = load_config()

    if max_tokens is None:
        max_tokens = cfg["server"].get("max_tokens", 256)

    seed = cfg.get("seed", 42)
    payload = _build_payload(messages, max_tokens, seed, stream=False)
    url = _base_url(tier, cfg) + "/v1/chat/completions"

    log.debug("complete → %s  max_tokens=%d", url, max_tokens)

    resp = httpx.post(url, json=payload, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Quick smoke test (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Quick smoke-test for llm_client.py")
    parser.add_argument(
        "--tier", choices=["lite", "mid", "full"], default="lite",
        help="Which tier to test.",
    )
    parser.add_argument(
        "--prompt", default="In one sentence, what is the capital of France?",
    )
    parser.add_argument("--config", default=str(_REPO_ROOT / "config.yaml"))
    args = parser.parse_args()

    _cfg = load_config(args.config)
    _tier = Tier[args.tier.upper()]
    _messages = [{"role": "user", "content": args.prompt}]

    print(f"\nStreaming from {_tier.name}:")
    for ev in stream(_tier, _messages, max_tokens=64, cfg=_cfg):
        print(ev.text, end="", flush=True)
        if ev.top_logprobs:
            best = ev.top_logprobs[0]
            print(
                f"  [top={best.token!r} lp={best.logprob:.3f}]",
                end="\n",
                flush=True,
            )
    print()
