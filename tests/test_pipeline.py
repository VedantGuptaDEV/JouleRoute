import sqlite3

import pytest
import requests

from gateway.pipeline import Pipeline
from gateway.telemetry import Telemetry
from precheck.cache import AnswerCache, normalize
from privacy.pii import PIIDetector, luhn_valid, verhoeff_check_digit, verhoeff_valid
from router.runtime.router import RouteDecision
from serving.llm_client import TokenEvent
from shared.config import load_config
from shared.schemas import ChatRequest, XRouting
from shared.tiers import Tier

# ------------------------------------------------------------------ PII


def test_verhoeff_round_trip_and_rejects_typo():
    body = "23456789012"
    valid = body + verhoeff_check_digit(body)
    assert verhoeff_valid(valid)
    assert not verhoeff_valid(body + str((int(valid[-1]) + 1) % 10))


def test_luhn():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")


@pytest.mark.parametrize("text, kind", [
    ("mail a.b@example.com", "email"),
    ("call +91 98765 43210", "phone"),
    ("card 4111 1111 1111 1111", "card"),
    ("PAN ABCDE1234F", "pan"),
    ("ifsc SBIN0001234", "ifsc"),
    ("server 192.168.1.1", "ipv4"),
    ("my otp is 4821", "sensitive_keyword"),
])
def test_pii_flags(text, kind):
    result = PIIDetector().detect(text)
    assert result.flag and kind in result.types


def test_aadhaar_needs_valid_checksum():
    body = "23456789012"
    good, bad = body + verhoeff_check_digit(body), body + str((int(verhoeff_check_digit(body)) + 1) % 10)
    assert "aadhaar" in PIIDetector().detect(f"id {good}").types
    assert not PIIDetector().detect(f"id {bad}").flag


@pytest.mark.parametrize("text", ["what is the capital of france", "solve 12 + 5", "card 4111 1111 1111 1112"])
def test_pii_clean(text):
    assert not PIIDetector().detect(text).flag


# ------------------------------------------------------------------ cache


def test_cache_normalizes_and_evicts():
    cache = AnswerCache(size=1, ttl_s=60)
    cache.put("What is  2+2?", "4", Tier.LITE, 1)
    assert normalize("What is  2+2?") == "what is 22"
    assert cache.get("what is 2+2").text == "4"
    cache.put("another", "x", Tier.LITE, 1)
    assert cache.get("what is 2+2") is None


# ------------------------------------------------------------------ pipeline


class FakeRouter:
    def __init__(self, decision: RouteDecision):
        self.decision = decision
        self.thresholds = {"mode_presets": {"saver": 0.5, "balanced": 0.7, "performance": 0.9}}
        self.backend = "fake"
        self.dir = type("D", (), {"name": "fake"})()

    def clamp_tau(self, tau):
        return tau

    def route(self, text, tau=None):
        return self.decision


class FakeLLM:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = []

    def stream(self, tier, messages, max_tokens=None, cancel=None):
        self.calls.append(tier)
        if self.fail:
            raise requests.ConnectionError("down")
        yield TokenEvent("Hello", [[-0.1] * 5])
        yield TokenEvent(" world", [[-0.2] * 5])


@pytest.fixture
def make_pipeline(tmp_path):
    def make(decision=None, fail=False):
        cfg = load_config()
        decision = decision or RouteDecision(Tier.MID, Tier.MID, 0.9, [0.05, 0.9, 0.05], False, 0.7, 1.0)
        tel = Telemetry(tmp_path / "t.sqlite", store_query_text=True)
        return Pipeline(cfg, router=FakeRouter(decision), llm=FakeLLM(fail), telemetry=tel)
    return make


def run(pipeline, text, **kw):
    events = list(pipeline.run(ChatRequest(messages=[{"role": "user", "content": text}], **kw)))
    steps = [(e.step.name, e.step.status) for e in events if e.kind == "step"]
    done = events[-1]
    assert done.kind == "done" and isinstance(done.routing, XRouting)
    return steps, done


def test_stages_in_order_and_answer_streamed(make_pipeline):
    steps, done = run(make_pipeline(), "summarize this paragraph")
    assert steps == [("cache", "done"), ("pii", "done"), ("router", "done"), ("gate", "done"),
                     ("generate", "start"), ("generate", "done"), ("telemetry", "done")]
    assert done.text == "Hello world"
    assert done.routing.tier == "mid" and done.routing.tokens_out == 2 and done.routing.source == "model"


def test_second_identical_query_hits_cache(make_pipeline):
    p = make_pipeline()
    run(p, "What is 2+2?")
    steps, done = run(p, "what is 2 + 2")
    steps2, done2 = run(p, "what is 2+2")
    assert done2.routing.source == "cache" and ("router", "done") not in steps2
    assert p.llm.calls == [Tier.MID, Tier.MID]


def test_low_confidence_falls_back_to_full(make_pipeline):
    d = RouteDecision(Tier.FULL, Tier.LITE, 0.4, [0.4, 0.3, 0.3], True, 0.7, 1.0)
    _, done = run(make_pipeline(d), "ambiguous")
    assert done.routing.fallback and done.routing.tier == "full" and done.routing.predicted_tier == "lite"


def test_forced_tier_skips_router(make_pipeline):
    steps, done = run(make_pipeline(), "hi", model="lite")
    assert ("router", "skip") in steps and ("gate", "done") not in steps
    assert done.routing.tier == "lite"


def test_pii_query_text_never_stored_and_not_cached(make_pipeline, tmp_path):
    p = make_pipeline()
    run(p, "my email is a@b.com")
    steps, done = run(p, "my email is a@b.com")
    assert done.routing.pii_flag and done.routing.source == "model"
    rows = sqlite3.connect(tmp_path / "t.sqlite").execute("SELECT query_text, pii_flag FROM requests").fetchall()
    assert rows == [(None, 1), (None, 1)]


def test_llm_down_still_writes_telemetry(make_pipeline, tmp_path):
    steps, done = run(make_pipeline(fail=True), "hello")
    assert ("error", "error") in steps and steps[-1] == ("telemetry", "done")
    assert "not reachable" in done.routing.error
    (err,) = sqlite3.connect(tmp_path / "t.sqlite").execute("SELECT error FROM requests").fetchone()
    assert "not reachable" in err


# ------------------------------------------------------------------ manual τ override

ARTIFACTS = load_config()["router"]["artifacts_dir"]


@pytest.mark.skipif(not __import__("pathlib").Path(ARTIFACTS).exists(), reason="router artifacts not present")
def test_tau_override_replaces_risk_controlled_tau():
    from router.runtime.router import Router
    r = Router(ARTIFACTS, tau_override=0.32)
    assert r.clamp_tau(None) == r.clamp_tau(0.99) == 0.32
    with pytest.raises(ValueError):
        Router(ARTIFACTS, tau_override=3.2)


# ------------------------------------------------------------------ streaming client


def test_llm_stream_survives_unicode_line_separators(monkeypatch):
    """U+2028 inside a JSON string must not split the SSE line (it did with decode_unicode=True)."""
    import io
    import json as _json

    from serving.llm_client import LLMClient

    def sse(obj):
        return b"data: " + _json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"

    body = (sse({"choices": [{"delta": {"content": "a b\x85c"}}]})
            + sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}) + b"data: [DONE]\n\n")

    def fake_post(*args, **kwargs):
        resp = requests.models.Response()
        resp.status_code, resp.raw = 200, io.BytesIO(body)
        return resp

    monkeypatch.setattr(requests, "post", fake_post)
    events = list(LLMClient(load_config()).stream(Tier.LITE, [{"role": "user", "content": "x"}]))
    assert "".join(e.text for e in events) == "a b\x85c"
    assert events[-1].finish_reason == "stop"


def test_cancel_stops_generation_and_still_logs(make_pipeline, tmp_path):
    import threading
    cancel = threading.Event()

    class SlowLLM(FakeLLM):
        def stream(self, tier, messages, max_tokens=None, cancel=None):
            for i in range(100):
                if cancel is not None and cancel.is_set():
                    return
                yield TokenEvent(f"t{i} ", [[-0.1] * 5])

    p = make_pipeline()
    p.llm = SlowLLM()
    events = p.run(ChatRequest(messages=[{"role": "user", "content": "long answer"}]), cancel)
    tokens = 0
    for ev in events:
        if ev.kind == "token":
            tokens += 1
            if tokens == 3:
                cancel.set()
    assert tokens == 3
    assert ev.kind == "done" and ev.routing.error == "interrupted by client"
    assert p.cache.get("long answer") is None  # a cut-short answer is never cached
    (err,) = sqlite3.connect(tmp_path / "t.sqlite").execute("SELECT error FROM requests").fetchone()
    assert err == "interrupted by client"
