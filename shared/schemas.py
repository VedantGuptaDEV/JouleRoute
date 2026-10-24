"""Pydantic models for the gateway API (Readme §5.H)."""
from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: Literal["auto", "lite", "mid", "full"] = "auto"
    messages: list[Message]
    stream: bool = False
    # None = no output cap (bounded only by server.ctx_size). Benchmarks pass server.max_tokens explicitly.
    max_tokens: int | None = None
    # Non-standard: wrap the last user message in this category's prompt template
    # (shared/prompts.py). Benchmarks pass the true category; chat leaves it unset.
    x_category: str | None = None


class Step(BaseModel):
    """One pipeline stage, streamed to clients as `x_step` so a UI can show progress."""
    name: str                                  # cache | pii | router | gate | generate | telemetry
    status: Literal["start", "done", "skip", "error"]
    title: str
    detail: str = ""
    ms: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class XRouting(BaseModel):
    tier: str | None = None
    predicted_tier: str | None = None
    confidence: float | None = None
    probs: list[float] | None = None
    fallback: bool = False
    escalated: bool = False
    pii_flag: bool = False
    pii_types: list[str] = Field(default_factory=list)
    source: Literal["cache", "rules", "model"] | None = None
    router_backend: str | None = None
    router_ms: float | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    tokens_out: int = 0
    max_tokens: int | None = None
    finish_reason: str | None = None          # "length": cut off at max_tokens, or the context filled up
    # Energy fields stay null until energy/meter.py and results/energy_profile.json exist.
    joules: float | None = None
    joules_est_full: float | None = None
    battery_minutes_saved: float | None = None
    tau: float | None = None
    mode: str | None = None
    is_estimate: bool | None = None
    error: str | None = None


class ModeRequest(BaseModel):
    mode: Literal["saver", "balanced", "performance"]
