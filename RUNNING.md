# Running the app

```bash
PY=.venv/bin/python
$PY serving/launch.py        # terminal 1: LITE :8081, MID :8082, FULL :8083 (FULL drafts with LITE); --no-spec to disable
$PY gateway/app.py           # terminal 2: gateway on http://127.0.0.1:8000
$PY cli/joule.py             # terminal 3: full-screen chat; --url to point at another gateway
$PY cli/joule.py -p "what is 17 * 23"   # one-shot, same layout, printed
```

Each message goes `cli → POST /v1/chat/completions (stream) → answer cache → PII scan → router →
confidence gate → llama-server for the chosen tier → telemetry (data/telemetry.sqlite)`, and every reply has
the same parts: the model that answered with its tier and how it got there (routed, fallback, forced,
cached), a one-line routing trace (click it or press ctrl+o for all six stages, filled live from the
gateway's `x_step` chunks), the streamed answer, and a metadata line. Skipped stages are shown as skipped.

CLI commands: `/model auto|lite|mid|full`, `/mode saver|balanced|performance`, `/category <name>|none`,
`/context on|off`, `/tokens <n>|off`, `/status`, `/clear`, `/help`, `/exit`. Keys: esc interrupts, ↑/↓ history,
→ accepts a suggestion, ctrl+o expands the last trace, ctrl+l clears, ctrl+c twice quits.

Answers have no length cap by default: the gateway omits `max_tokens` unless the request sets one,
so the model stops when it is done. The only hard bound is the context window (`server.ctx_size`,
4096 tokens for prompt + answer); a receipt ending in "cut off at a full context window" means that
was hit. `/tokens <n>` or `--max-tokens` sets a cap. Labels and benchmarks use `server.max_tokens`
(256): `generate_answers.py` passes it directly, and the benchmark harness must pass it in each request.

`router.tau_override` in `config.yaml` is a manual confidence threshold (currently 0.32) that replaces
the risk-controlled one from `thresholds.json`; set it to `null` to go back.

Not built yet (so not shown as stages): rule engine, in-generation escalation, energy meter and
receipt, battery controller. The router runs the fp32 PyTorch checkpoint until
`router/artifacts/bert_tiny/model.int8.onnx` is exported.

Training the router on a GPU machine: see `TRAINING.md`.
