"""The request pipeline (Readme §2), one timed stage at a time.

  1 answer cache  →  3 PII detector  →  4 router  →  5 confidence gate  →  6 generate  →  8 telemetry

`Pipeline.run` is a generator of events so the gateway can stream each stage to the
client as it finishes. Not built yet, and therefore not shown as stages: rule engine (2),
in-generation escalation, energy meter / receipt (7), battery controller.
"""
import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Literal

import requests

from gateway.telemetry import Telemetry
from precheck.cache import AnswerCache, query_hash
from privacy.pii import PIIDetector
from router.runtime.router import Router
from serving.llm_client import LLMClient
from shared.config import repo_path
from shared.prompts import build_messages
from shared.schemas import ChatRequest, Step, XRouting
from shared.tiers import Tier

log = logging.getLogger(__name__)


@dataclass
class Event:
    kind: Literal["step", "token", "done"]
    step: Step | None = None
    text: str = ""
    routing: XRouting | None = None


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


class Pipeline:
    def __init__(self, cfg: dict, router: Router | None = None, llm: LLMClient | None = None,
                 telemetry: Telemetry | None = None):
        self.cfg = cfg
        self.router = router or Router(repo_path(cfg["router"]["artifacts_dir"]), cfg["router"]["max_length"],
                                       cfg["router"].get("tau_override"))
        self.llm = llm or LLMClient(cfg)
        self.telemetry = telemetry or Telemetry(repo_path(cfg["telemetry"]["db"]),
                                                cfg["telemetry"]["store_query_text"])
        self.pii = PIIDetector(cfg["pii"]["ner_enabled"])
        c = cfg["cache"]
        self.cache = AnswerCache(c["size"], c["ttl_s"]) if c["enabled"] else None
        self.mode = "balanced"

    @property
    def tau_source(self) -> str:
        return "manual" if getattr(self.router, "tau_override", None) is not None else self.mode

    @property
    def tau(self) -> float:
        return self.router.clamp_tau(self.router.thresholds["mode_presets"][self.mode])

    def model_name(self, tier: Tier) -> str:
        return self.cfg["models"][tier.key]["hf_id"].split("/")[-1]

    def run(self, req: ChatRequest, cancel: threading.Event | None = None) -> Iterator[Event]:
        """Run every stage for one request. Setting `cancel` (client went away) stops generation early."""
        t0 = time.perf_counter()
        users = [m for m in req.messages if m.role == "user"]
        query = users[-1].content if users else ""
        r = XRouting(mode=self.mode)
        answer: list[str] = []
        state = {"completed": False}
        try:
            yield from self._stages(req, query, r, answer, cancel)
            state["completed"] = True
        except Exception as e:  # every request gets a telemetry row, failures included
            log.exception("pipeline failed")
            r.error = r.error or f"{type(e).__name__}: {e}"
            yield Event("step", Step(name="error", status="error", title="Pipeline error", detail=r.error))
        finally:
            r.total_ms = _ms(t0)
            if not state["completed"] and r.error is None:
                r.error = "interrupted by client"
            t1 = time.perf_counter()
            row_id = self._log(query, req.x_category, r)
        yield Event("step", Step(name="telemetry", status="done", title="Telemetry",
                                 detail=f"row #{row_id} → {self.cfg['telemetry']['db']}", ms=_ms(t1),
                                 data={"row": row_id, "db": self.cfg["telemetry"]["db"]}))
        yield Event("done", text="".join(answer), routing=r)

    def _stages(self, req: ChatRequest, query: str, r: XRouting, answer: list[str],
                cancel: threading.Event | None) -> Iterator[Event]:
        # 1. Answer cache: single-turn, routed queries only.
        single_turn = sum(m.role != "system" for m in req.messages) == 1
        cacheable = self.cache is not None and req.model == "auto" and single_turn and not req.x_category
        if self.cache is None:
            yield Event("step", Step(name="cache", status="skip", title="Answer cache", detail="disabled in config"))
        elif not cacheable:
            reason = "forced tier" if req.model != "auto" else "multi-turn conversation"
            yield Event("step", Step(name="cache", status="skip", title="Answer cache", detail=f"bypassed: {reason}"))
        else:
            t = time.perf_counter()
            hit = self.cache.get(query)  # PII queries are never stored, so never hit
            if hit:
                r.source, r.tier, r.tokens_out = "cache", hit.tier.key, hit.tokens_out
                yield Event("step", Step(name="cache", status="done", title="Answer cache",
                                         detail=f"hit · answered earlier by {hit.tier.name}", ms=_ms(t),
                                         data={"hit": True, "tier": hit.tier.key}))
                answer.append(hit.text)
                yield Event("token", text=hit.text)
                return
            yield Event("step", Step(name="cache", status="done", title="Answer cache", detail="miss",
                                     ms=_ms(t), data={"hit": False}))

        # 3. PII detector. With the cloud tier off the flag only affects caching and telemetry.
        t = time.perf_counter()
        pii = self.pii.detect(query)
        r.pii_flag, r.pii_types = pii.flag, pii.types
        where = "never sent to cloud" if self.cfg["cloud"]["enabled"] else "kept on-device (cloud tier off)"
        detail = f"flagged: {', '.join(pii.types)} · {where}" if pii.flag else "clean"
        yield Event("step", Step(name="pii", status="done", title="PII scan", detail=detail, ms=_ms(t),
                                 data={"flag": pii.flag, "types": pii.types,
                                       "spans": [[s, e, k] for s, e, k in pii.spans]}))

        # 4–5. Router + confidence gate, unless a tier was forced.
        if req.model == "auto":
            d = self.router.route(query, self.tau)
            r.tier, r.predicted_tier = d.tier.key, d.predicted_tier.key
            r.confidence, r.probs, r.fallback, r.tau = d.confidence, d.probs, d.fallback, d.tau
            r.router_ms, r.router_backend = round(d.latency_ms, 2), self.router.backend
            yield Event("step", Step(
                name="router", status="done", title=f"Router · {self.router.dir.name}",
                detail=f"predicted {d.predicted_tier.name} · confidence {d.confidence:.2f}",
                ms=r.router_ms, data={"probs": d.probs, "predicted": d.predicted_tier.key,
                                      "backend": self.router.backend}))
            op = "<" if d.fallback else "≥"
            outcome = "fallback → FULL" if d.fallback else f"→ {d.tier.name}"
            yield Event("step", Step(
                name="gate", status="done", title="Confidence gate",
                detail=f"{d.confidence:.2f} {op} τ {d.tau:.2f} ({self.tau_source}) {outcome}",
                data={"confidence": d.confidence, "tau": d.tau, "fallback": d.fallback, "tier": d.tier.key,
                      "predicted": d.predicted_tier.key, "tau_source": self.tau_source}))
            tier = d.tier
        else:
            tier = Tier[req.model.upper()]
            r.tier = tier.key
            yield Event("step", Step(name="router", status="skip", title="Router",
                                     detail=f"bypassed: tier forced to {tier.name}", data={"tier": tier.key}))

        # 6. Generate on the chosen tier (streaming).
        r.source = "model"
        port = self.cfg["models"][tier.key]["port"]
        spec = tier == Tier.FULL and self.cfg["server"]["speculative"]
        title = f"{tier.name} · {self.model_name(tier)}"
        yield Event("step", Step(name="generate", status="start", title=title,
                                 detail=f":{port}" + (f" · speculative (draft {self.model_name(Tier.LITE)})" if spec else ""),
                                 data={"tier": tier.key, "model": self.model_name(tier), "port": port,
                                       "speculative": spec}))
        messages = [m.model_dump() for m in req.messages]
        if req.x_category:
            messages = build_messages(req.x_category, query)
        r.max_tokens = req.max_tokens  # None = no cap
        t = time.perf_counter()
        try:
            for ev in self.llm.stream(tier, messages, r.max_tokens, cancel):
                r.finish_reason = ev.finish_reason or r.finish_reason
                if not ev.text:
                    continue
                if r.ttft_ms is None:
                    r.ttft_ms = _ms(t)
                r.tokens_out += max(len(ev.top_logprobs), 1)
                answer.append(ev.text)
                yield Event("token", text=ev.text)
        except requests.ConnectionError:
            r.error = f"llama-server for {tier.name} is not reachable on :{port}; start it with serving/launch.py"
            raise
        if cancel is not None and cancel.is_set():
            r.error = "interrupted by client"
            return
        gen_ms = _ms(t)
        # Decode rate excludes the first token (prompt processing is in ttft).
        decode_s = (gen_ms - (r.ttft_ms or 0)) / 1000
        tps = (r.tokens_out - 1) / decode_s if r.tokens_out > 1 and decode_s > 0 else None
        rate = f" · {tps:.1f} tok/s" if tps else ""
        yield Event("step", Step(name="generate", status="done", title=title,
                                 detail=f"{r.tokens_out} tokens{rate} · ttft {r.ttft_ms or 0:.0f} ms",
                                 ms=gen_ms, data={"tier": tier.key, "tokens_out": r.tokens_out, "tok_s": tps,
                                       "finish_reason": r.finish_reason, "max_tokens": r.max_tokens}))

        # Cache only routed, non-escalated answers without PII.
        if cacheable and not r.pii_flag and not r.escalated and answer:
            self.cache.put(query, "".join(answer), tier, r.tokens_out)

    def _log(self, query: str, category: str | None, r: XRouting) -> int:
        return self.telemetry.write({
            "query_hash": query_hash(query), "query_text": query, "category_guess": category,
            "source": r.source, "pii_flag": int(r.pii_flag), "predicted_tier": r.predicted_tier,
            "tier": r.tier, "confidence": r.confidence, "fallback": int(r.fallback),
            "escalated": int(r.escalated), "tau": r.tau, "mode": r.mode, "router_ms": r.router_ms,
            "ttft_ms": r.ttft_ms, "total_ms": r.total_ms, "tokens_out": r.tokens_out,
            "joules": r.joules, "joules_est_full": r.joules_est_full, "is_estimate": r.is_estimate,
            "error": r.error,
        })
