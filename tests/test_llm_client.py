"""
tests/test_llm_client.py
Unit tests for serving.llm_client.

Covers (no real server needed):
  - token_entropy: near-0 for one-hot, near-1 for uniform, 0.0 for empty/single
  - finish_reason parsing: "stop" and "length" via fake SSE response
  - max_tokens=-1 reaching the request body
  - ctx_size warning triggering when ctx_size < max_tokens + 512
"""
from __future__ import annotations

import json
import logging
import math
import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from serving.llm_client import (
    LogProbEntry,
    TokenEvent,
    StreamTimeoutError,
    _build_payload,
    _check_ctx_size,
    _TOP_K,
    stream,
    token_entropy,
)
from shared.tiers import Tier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entries(logprobs: list[float]) -> list[LogProbEntry]:
    return [LogProbEntry(token=f"t{i}", logprob=lp) for i, lp in enumerate(logprobs)]


def _sse_lines(*chunks: dict) -> list[str]:
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append("data: [DONE]")
    return lines


def _token_chunk(text: str) -> dict:
    return {
        "choices": [
            {
                "finish_reason": None,
                "index": 0,
                "delta": {"content": text},
                "logprobs": None,
            }
        ]
    }


def _final_chunk(finish_reason: str) -> dict:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "index": 0,
                "delta": {},
            }
        ]
    }


@contextmanager
def _fake_stream(sse_lines: list[str]) -> Iterator[MagicMock]:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_lines = MagicMock(return_value=iter(sse_lines))

    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_response)
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_cm)

    mock_client_cm = MagicMock()
    mock_client_cm.__enter__ = MagicMock(return_value=mock_client)
    mock_client_cm.__exit__ = MagicMock(return_value=False)

    with patch("serving.llm_client.httpx.Client", return_value=mock_client_cm):
        yield mock_client


_FAKE_CFG = {
    "gateway": {"host": "127.0.0.1"},
    "models": {
        "lite": {"port": 8081},
        "mid":  {"port": 8082},
        "full": {"port": 8083},
    },
    "server": {"max_tokens": -1, "request_timeout_s": 180},
    "seed": 42,
}


# ---------------------------------------------------------------------------
# token_entropy
# ---------------------------------------------------------------------------

def test_token_entropy_near_zero_for_near_one_hot() -> None:
    entries = _make_entries([-0.0001, -20.0, -20.0, -20.0, -20.0])
    result = token_entropy(entries)
    assert result < 0.05, f"Expected near-zero entropy, got {result:.4f}"


def test_token_entropy_near_one_for_uniform() -> None:
    k = 5
    lp = math.log(1.0 / k)
    entries = _make_entries([lp] * k)
    result = token_entropy(entries)
    assert abs(result - 1.0) < 1e-9, f"Expected 1.0, got {result:.9f}"


def test_token_entropy_empty_returns_zero() -> None:
    assert token_entropy([]) == 0.0


def test_token_entropy_single_entry_returns_zero() -> None:
    assert token_entropy(_make_entries([-1.0])) == 0.0


# ---------------------------------------------------------------------------
# finish_reason (fake SSE)
# ---------------------------------------------------------------------------

def test_finish_reason_stop() -> None:
    lines = _sse_lines(
        _token_chunk("Hello"),
        _token_chunk(" world"),
        _final_chunk("stop"),
    )
    with _fake_stream(lines):
        events = list(stream(Tier.LITE, [{"role": "user", "content": "hi"}], cfg=_FAKE_CFG))

    sentinel = events[-1]
    assert sentinel.finish_reason == "stop"
    assert sentinel.text == ""
    for ev in events[:-1]:
        assert ev.finish_reason is None


def test_finish_reason_length() -> None:
    lines = _sse_lines(
        _token_chunk("Cut"),
        _final_chunk("length"),
    )
    with _fake_stream(lines):
        events = list(stream(Tier.LITE, [{"role": "user", "content": "hi"}], cfg=_FAKE_CFG))

    sentinel = events[-1]
    assert sentinel.finish_reason == "length"
    assert sentinel.text == ""


# ---------------------------------------------------------------------------
# max_tokens=-1 in request body
# ---------------------------------------------------------------------------

def test_max_tokens_minus_one_in_payload() -> None:
    payload = _build_payload(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=-1,
        seed=42,
        stream=True,
    )
    assert payload["max_tokens"] == -1


def test_max_tokens_minus_one_sent_to_server() -> None:
    lines = _sse_lines(_token_chunk("Hi"), _final_chunk("stop"))
    with _fake_stream(lines) as mock_client:
        list(stream(Tier.LITE, [{"role": "user", "content": "hi"}], cfg=_FAKE_CFG))

    call_kwargs = mock_client.stream.call_args
    sent_json: dict = call_kwargs.kwargs.get("json") or call_kwargs.args[2]
    assert sent_json["max_tokens"] == -1
    assert sent_json["logprobs"] is True
    assert sent_json["top_logprobs"] == _TOP_K


# ---------------------------------------------------------------------------
# ctx_size warning
# ---------------------------------------------------------------------------

def test_ctx_size_warning_fires_when_too_small(caplog: pytest.LogCaptureFixture) -> None:
    """
    _check_ctx_size must log a WARNING when ctx_size < max_tokens + 512.
    Example: ctx_size=512, max_tokens=1024 → threshold=1536 → warning.
    """
    cfg = {
        "server": {
            "ctx_size": 512,
            "max_tokens": 1024,
        }
    }
    with caplog.at_level(logging.WARNING, logger="serving.llm_client"):
        # _check_ctx_size also writes to stderr; redirect to avoid noise in test output
        with patch("sys.stderr", new_callable=StringIO):
            _check_ctx_size(cfg)

    assert any("ctx_size" in rec.message for rec in caplog.records), (
        "Expected a WARNING mentioning ctx_size"
    )


def test_ctx_size_warning_does_not_fire_when_adequate(caplog: pytest.LogCaptureFixture) -> None:
    """No warning when ctx_size >= max_tokens + 512."""
    cfg = {
        "server": {
            "ctx_size": 4096,
            "max_tokens": 2048,
        }
    }
    # 4096 >= 2048+512=2560 → no warning
    with caplog.at_level(logging.WARNING, logger="serving.llm_client"):
        _check_ctx_size(cfg)

    assert not any("ctx_size" in rec.message for rec in caplog.records), (
        "No warning expected when ctx_size is adequate"
    )


def test_ctx_size_warning_recommends_larger_value(caplog: pytest.LogCaptureFixture) -> None:
    """The warning message must include a recommended ctx_size larger than the threshold."""
    cfg = {
        "server": {
            "ctx_size": 512,
            "max_tokens": 1024,
        }
    }
    with caplog.at_level(logging.WARNING, logger="serving.llm_client"):
        with patch("sys.stderr", new_callable=StringIO):
            _check_ctx_size(cfg)

    warning_text = " ".join(rec.message for rec in caplog.records)
    # threshold = 1024+512 = 1536; recommended next power-of-two = 2048
    assert "2048" in warning_text, (
        f"Expected recommended value 2048 in warning; got: {warning_text}"
    )
