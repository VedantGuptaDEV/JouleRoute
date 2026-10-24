"""FastAPI gateway, OpenAI-compatible (Readme §5.H).

  POST /v1/chat/completions   model "auto" (routed) or "lite" | "mid" | "full" (forced), stream or not
  GET  /v1/status             mode, τ, router info, warnings
  POST /v1/mode               set a mode preset
  GET  /health

Non-standard fields: `x_routing` on the response (and on the last stream chunk), and
`x_step` chunks with empty `choices` while streaming, one per pipeline stage. Standard
OpenAI clients ignore both.

Run: .venv/bin/python gateway/app.py   (host/port from config.yaml)
"""
import argparse
import json
import logging
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gateway.pipeline import Pipeline  # noqa: E402
from shared.config import load_config  # noqa: E402
from shared.schemas import ChatRequest, ModeRequest  # noqa: E402
from shared.tiers import Tier  # noqa: E402


class CancellableStream(StreamingResponse):
    """Sets `cancel` as soon as the client disconnects.

    Starlette stops reading the body on disconnect but never closes it, which would leave
    the pipeline parked mid-answer and llama-server generating to the end.
    """

    def __init__(self, content, cancel: threading.Event, **kw):
        super().__init__(content, **kw)
        self.cancel = cancel

    async def listen_for_disconnect(self, receive) -> None:
        await super().listen_for_disconnect(receive)
        self.cancel.set()


def run_in_background(pipeline: Pipeline, req: ChatRequest, cancel: threading.Event) -> queue.Queue:
    """Run the pipeline in its own thread so it always finishes (and logs telemetry), even if
    nobody is reading any more. Events arrive on the queue; None marks the end."""
    q: queue.Queue = queue.Queue()

    def produce():
        try:
            for ev in pipeline.run(req, cancel):
                q.put(ev)
        finally:
            q.put(None)

    threading.Thread(target=produce, daemon=True, name="pipeline").start()
    return q


def create_app(pipeline: Pipeline) -> FastAPI:
    app = FastAPI(title="JouleRoute gateway")

    def chunk(cid: str, model: str, **fields) -> str:
        body = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                "choices": [], **fields}
        return f"data: {json.dumps(body)}\n\n"

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        if not req.stream:
            events = pipeline.run(req)
            steps, text, routing = [], "", None
            for ev in events:
                if ev.kind == "step":
                    steps.append(ev.step.model_dump())
                elif ev.kind == "done":
                    text, routing = ev.text, ev.routing
            return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": req.model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": "error" if routing.error else "stop"}],
                    "usage": {"completion_tokens": routing.tokens_out},
                    "x_routing": routing.model_dump(), "x_steps": steps}

        cancel = threading.Event()
        events = run_in_background(pipeline, req, cancel)

        async def sse():
            while (ev := await run_in_threadpool(events.get)) is not None:
                if ev.kind == "step":
                    yield chunk(cid, req.model, x_step=ev.step.model_dump())
                elif ev.kind == "token":
                    yield chunk(cid, req.model, choices=[{"index": 0, "delta": {"content": ev.text},
                                                          "finish_reason": None}])
                else:
                    reason = "error" if ev.routing.error else "stop"
                    yield chunk(cid, req.model, choices=[{"index": 0, "delta": {}, "finish_reason": reason}],
                                x_routing=ev.routing.model_dump())
            yield "data: [DONE]\n\n"

        return CancellableStream(sse(), cancel, media_type="text/event-stream")

    @app.get("/v1/status")
    def status():
        router = pipeline.router
        manual = router.tau_override is not None
        warnings = ([f"manual τ = {router.tau_override:.2f} from config.yaml (router.tau_override): "
                     "not risk-controlled, so the accuracy-loss guarantee does not apply"] if manual
                    else [w for w in [router.thresholds.get("warning")] if w])
        return {
            "mode": pipeline.mode, "tau": pipeline.tau, "tau_min": router.tau_min, "tau_source": pipeline.tau_source,
            "mode_presets": router.thresholds["mode_presets"],
            "router": {"name": router.dir.name, "backend": router.backend, "load_ms": round(router.load_ms, 1)},
            "tiers": {t.key: {"model": pipeline.model_name(t), "port": pipeline.cfg["models"][t.key]["port"],
                              "healthy": pipeline.llm.healthy(t)} for t in Tier},
            "cache": pipeline.cache is not None, "cloud": pipeline.cfg["cloud"]["enabled"],
            "warnings": warnings,
        }

    @app.post("/v1/mode")
    def set_mode(req: ModeRequest):
        pipeline.mode = req.mode
        return {"mode": pipeline.mode, "tau": pipeline.tau, "tau_source": pipeline.tau_source}

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=cfg["gateway"]["host"])
    parser.add_argument("--port", type=int, default=cfg["gateway"]["port"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    pipeline = Pipeline(cfg)
    print(f"router {pipeline.router.dir.name} ({pipeline.router.backend}) loaded in "
          f"{pipeline.router.load_ms:.0f} ms; gateway on http://{args.host}:{args.port}")
    uvicorn.run(create_app(pipeline), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
