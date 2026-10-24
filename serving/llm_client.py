"""Streaming client for the per-tier llama-server processes (Readme §5.B).

Uses llama-server's OpenAI-compatible /v1/chat/completions with stream=true and
top-5 logprobs per token (for in-generation escalation). Greedy decoding, thinking off.
"""
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Iterator

import requests

from shared.tiers import Tier

log = logging.getLogger(__name__)
TOP_K = 5


@dataclass
class TokenEvent:
    text: str
    top_logprobs: list[float] = field(default_factory=list)  # one list per streamed token piece
    finish_reason: str | None = None                          # "stop", or "length" if max_tokens cut it off


class LLMClient:
    def __init__(self, cfg: dict, host: str = "127.0.0.1"):
        self.cfg = cfg
        self.host = host

    def url(self, tier: Tier) -> str:
        return f"http://{self.host}:{self.cfg['models'][tier.key]['port']}"

    def healthy(self, tier: Tier) -> bool:
        try:
            return requests.get(f"{self.url(tier)}/health", timeout=1).status_code == 200
        except requests.RequestException:
            return False

    def stream(self, tier: Tier, messages: list[dict], max_tokens: int | None = None,
               cancel: threading.Event | None = None) -> Iterator[TokenEvent]:
        """max_tokens=None: no output cap; the model stops at end-of-answer or when the context is full.

        Setting `cancel` ends the stream at the next token and closes the connection, which makes
        llama-server stop generating (it otherwise keeps going until the answer is finished).
        """
        srv = self.cfg["server"]
        body = {
            "messages": messages,
            "stream": True,
            "temperature": srv["temperature"],
            "seed": self.cfg["seed"],
            "logprobs": True,
            "top_logprobs": TOP_K,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        with requests.post(f"{self.url(tier)}/v1/chat/completions", json=body, stream=True, timeout=600) as r:
            r.raise_for_status()
            # Split raw bytes on \n only: decoded str.splitlines() would also break lines at U+2028,
            # U+0085 etc., which llama-server emits unescaped inside JSON strings.
            for raw in r.iter_lines():
                if cancel is not None and cancel.is_set():
                    return
                line = raw.decode("utf-8")
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if not chunk.get("choices"):
                    continue
                choice = chunk["choices"][0]
                text = (choice.get("delta") or {}).get("content") or ""
                logprobs = [[t["logprob"] for t in c.get("top_logprobs", [])]
                            for c in ((choice.get("logprobs") or {}).get("content") or [])]
                if text or choice.get("finish_reason"):
                    yield TokenEvent(text, logprobs, choice.get("finish_reason"))
