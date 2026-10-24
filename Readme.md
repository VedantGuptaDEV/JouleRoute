# CLAUDE.md: One App, Many Modes (full project)

Context for building the whole project. Read this file fully before writing code.
The router component has its own detailed spec in `router/CLAUDE.md`; this file
covers everything else and how the parts fit together.

---

## 1. What we are building

**One App, Many Modes** is an on-device AI inference layer. It runs three local LLMs of
the same family and sends each user query to the *cheapest model that can answer it
correctly*, so the device saves battery on easy queries and still uses the strong
model on hard ones.

Hackathon: Cognition 2026 (VIT), Edge AI / On-Device Intelligence track.

**Pitch line:** "A drop-in on-device inference layer that routes every query to the
cheapest model that can answer it, with a guaranteed accuracy bound, and falls back
to the strongest model whenever it is unsure."

### Judging requirements and where each one is met

| Requirement from the brief | Met by |
|---|---|
| Two or more model tiers, with a routing/gating mechanism | Three tiers (§5.B) + router + confidence gate (§5.A, §5.E) |
| Battery/latency benchmarks vs. always-full | Energy meter (§5.G) + benchmark harness (§5.J) |
| Routing decision must be cheap | BERT-Tiny int8 router, ~1 ms, measured (§5.A) |
| Accuracy trade-off analysis | Threshold sweep, Pareto curve, harmful-downgrade analysis (§5.A, §5.J, §5.K) |

Every component below must trace back to one of these four rows or to the demo.
If it doesn't, it is a stretch goal.

---

## 2. Architecture

```
                 ┌──────────────────────── Gateway (FastAPI, OpenAI-compatible) ────────────────────────┐
User / any app → │ 1 Answer cache ─hit──────────────────────────────────────────────────────┐          │
                 │ 2 Rule engine  ─match────────────────────────────────────────────────────┤          │
                 │ 3 PII detector → privacy flag (only matters if cloud tier enabled)       │          │
                 │ 4 Router (BERT-Tiny int8) → probs                                         │          │
                 │ 5 Confidence gate: conf ≥ τ → predicted tier, else FULL (fallback)        │          │
                 │        ▲ τ from Battery controller (uses battery %, target, energy rate)  │          │
                 │ 6 Generate on chosen tier (streaming)                                     │          │
                 │      LITE / MID: watch token entropy → escalate to FULL if it spikes      │          │
                 │      FULL: speculative decoding (0.8B drafts, 4B verifies)                │          │
                 │ 7 Response + energy receipt ◄─────────────────────────────────────────────┘          │
                 │ 8 Telemetry → SQLite                                                                  │
                 └───────────────────────────────────────────────────────────────────────────────────────┘
   llama-server ×3 (LITE :8081, MID :8082, FULL :8083)      Energy meter samples power during each request
   Benchmark harness → results/*.csv, energy_profile.json   Dashboard (Streamlit) reads SQLite + results/
```

Two confidence checks, by design:
- **Pre-routing confidence** (router, before generation): low → fall back to FULL.
- **In-generation confidence** (LLM token entropy, during generation): spike → escalate to FULL.

---

## 3. Fixed decisions (do not change without asking)

| Area | Decision |
|---|---|
| Language | Python 3.10+ for everything |
| Models | `LITE` Qwen3.5-0.8B, `MID` Qwen3.5-2B, `FULL` Qwen3.5-4B (same family, shared tokenizer) |
| Model format | GGUF `Q4_K_M` for all tiers by default. Measure LITE at `Q8_0` too (small models lose more to 4-bit); keep whichever is better, same format in benchmarks and demo |
| Thinking mode | Off for all tiers |
| Inference server | llama.cpp `llama-server`, one process per tier |
| Speculative decoding | FULL tier uses LITE as the draft model |
| Router | BERT-Tiny, 3-class, int8 ONNX (spec: `router/CLAUDE.md`) |
| Gateway | FastAPI + uvicorn, OpenAI-compatible `/v1/chat/completions` with streaming |
| Telemetry store | SQLite (single file) |
| Dashboard | Streamlit, two pages: Live and Evaluation |
| Config | One file: `config.yaml` at repo root; no magic numbers in code |
| Seeds | 42 everywhere |

**Hard rules**
- The router must stay cheap: no LLM, no large embedding model, no network call in the routing path.
- Everything runs offline on the device. The optional cloud tier is off by default.
- Never report a number not produced by a script in this repo. Estimates and placeholders
  must be labelled as such everywhere they appear (code, JSON, UI).

---

## 4. Repository layout

```
one-app-many-modes/
├── CLAUDE.md                  # this file
├── config.yaml                # all ports, paths, thresholds, constants
├── requirements.txt           # pinned; GPU-only deps in requirements-gpu.txt
├── shared/
│   ├── tiers.py               # Tier enum (LITE=0, MID=1, FULL=2), single source of truth
│   ├── config.py              # loads + validates config.yaml
│   └── schemas.py             # pydantic models for API, telemetry, energy profile
├── router/                    # see router/CLAUDE.md (data, training, calibration, runtime)
├── serving/
│   ├── download_models.py     # fetch GGUF files
│   ├── launch.py              # start/stop the 3 llama-server processes, health checks
│   └── llm_client.py          # streaming client returning tokens + top-k logprobs
├── precheck/
│   ├── cache.py               # answer cache (Tier −1)
│   └── rules.py               # rule engine (Tier 0)
├── privacy/
│   └── pii.py                 # PII detector
├── control/
│   ├── escalation.py          # in-generation entropy monitor
│   ├── battery.py             # battery reader (real + simulated)
│   └── controller.py          # battery controller that sets τ
├── energy/
│   ├── meter.py               # power sampling backends + per-request attribution
│   └── calibrate_idle.py      # measure idle power baseline
├── gateway/
│   ├── app.py                 # FastAPI app, endpoints
│   ├── pipeline.py            # the request pipeline (§2 steps 1–8)
│   └── telemetry.py           # SQLite writer
├── bench/
│   ├── run_bench.py           # run every policy over the test set on-device
│   ├── build_energy_profile.py
│   └── analyze.py             # final tables + plots for slides
├── dashboard/
│   ├── Home.py                # Live view
│   └── pages/Evaluation.py    # Evaluation view
├── results/                   # outputs consumed by dashboard and slides
├── data/                      # datasets (gitignored if large)
├── models/                    # GGUF files (gitignored)
└── tests/
```

---

## 5. Component specs

### 5.A Router and confidence gate
Fully specified in `router/CLAUDE.md`. Summary of what the rest of the system relies on:
- `router.runtime.router.Router(artifacts_dir).route(text, tau) -> RouteDecision`
  with fields `tier, predicted_tier, confidence, probs, fallback, tau, latency_ms`.
- `artifacts/thresholds.json` gives `tau_min`, `default_tau`, and mode presets.
- The gate is inside `Router.route`; τ below `tau_min` is clamped up.

`shared/tiers.py` must be the only definition of `Tier`; the router imports it.

### 5.B Serving layer (`serving/`)
- `download_models.py`: download GGUF files for the three tiers into `models/`
  (repos and filenames come from `config.yaml`; verify they exist on Hugging Face).
- `launch.py`: start one `llama-server` per tier on ports from config
  (defaults LITE 8081, MID 8082, FULL 8083), wait for health, stop cleanly on exit.
  - FULL is started with LITE's GGUF as its draft model for speculative decoding.
    Check exact flag names with `llama-server --help` (draft model, draft length);
    do not assume them.
  - Same context size, thread count and batch settings for every tier, set in config.
  - `--no-spec` option launches FULL without speculation, so we can measure its effect.
- `llm_client.py`:
  - `stream(tier, messages, max_tokens) -> iterator of TokenEvent(text, top_logprobs)`.
  - Request top-k token probabilities (k=5) from the server so escalation can compute entropy.
    Verify the request/response field names against the llama-server docs.
  - Greedy decoding (temperature 0) by default; reproducibility matters for benchmarks.
- **Model residency:** LITE always loaded. MID and FULL may be lazy-started and stopped
  after an idle period (config). If enabled, the cold-start time and energy must be
  measured and reported, not hidden. Default for the demo: all three resident.
- Prompt templates per query category live in one module and are shared with
  `router/data/generate_answers.py`. Labels are only valid if the app uses the same templates.

### 5.C Pre-checks (`precheck/`)
Both run before the router and are optional speed-ups; they must never return a wrong answer.
- **Answer cache:** key = hash of normalized query (lowercase, collapse whitespace, strip
  punctuation). LRU, size and TTL from config. Only cache answers from FULL or answers that
  were not escalated. Never cache queries with the PII flag.
- **Rule engine:** regex intents for set alarm/timer/reminder, unit conversion, and plain
  arithmetic (evaluate safely with a tiny parser; never use `eval`). Returns a structured
  action + a short text reply. Anything not clearly matched goes to the router.

### 5.D PII detector (`privacy/pii.py`)
- `detect(text) -> PIIResult(flag: bool, types: list[str], spans: list)`.
- Layer 1 regex with validation: email, Indian mobile (`+91`, 10 digits starting 6–9),
  Aadhaar (12 digits, first not 0/1, **Verhoeff checksum**), PAN (`[A-Z]{5}[0-9]{4}[A-Z]`),
  card numbers (13–19 digits, **Luhn checksum**), IFSC, IPv4.
- Layer 2 (optional, config flag): spaCy small English NER for PERSON / GPE / LOC.
  Only run it if layer 1 found nothing.
- Layer 3: keyword list for sensitive intent ("password", "OTP", "PIN", medical words).
- Tune toward flagging when in doubt (false negatives are worse than false positives).
- Effect: if `flag` is true, the query may not go to the cloud tier (§5.L). With the cloud
  tier disabled, the flag is still computed and logged so the dashboard can show it.
- Unit tests with valid and invalid checksum examples for Aadhaar and cards.

### 5.E In-generation escalation (`control/escalation.py`)
Applies only when the routed tier is LITE or MID.
- For each generated token, compute normalized entropy over the returned top-k probabilities.
- Keep a rolling mean over the last N tokens (config, default 16). After a minimum of
  M tokens (default 8), if the rolling mean exceeds `h_max`, escalate.
- On escalation: stop the stream, discard the partial answer, regenerate on FULL.
  Record `escalated=True`, the wasted tokens, and their energy in telemetry.
- Tune `h_max` offline: replay LITE/MID generations on the router's **val** split and choose
  the value that best separates wrong answers from right ones. Save to config with the
  achieved precision/recall. Never tune it on test.
- The UI shows that an escalation happened (it adds latency; be honest about it).

### 5.F Battery controller (`control/battery.py`, `control/controller.py`)
**Battery reader**
- Real: `psutil.sensors_battery()` (percent, plugged in).
- **Simulated battery** for the demo: capacity in Wh from config, drained by the *measured*
  joules of each request plus idle power over time. Real battery percentage barely moves
  during a short demo, so the simulated battery is what the live view shows. It must be
  labelled "simulated" in the UI.

**Controller**
- Inputs: target end time (or a mode preset), current battery energy, recent average power
  (from telemetry, moving window from config).
- Every K requests or T seconds: required power = remaining energy / time left.
  If measured power > required, lower τ by a step; if well below, raise τ by a step.
  Proportional step with a max change per update (config) to avoid oscillation.
- Always clamp τ to `[tau_min, 0.99]`. If the target cannot be met even at `tau_min`,
  keep `tau_min` and raise an "infeasible target" warning for the UI. Accuracy guarantee
  wins over battery target.
- Mode presets (Saver / Balanced / Performance) are fixed τ values from `thresholds.json`;
  a target time switches to automatic control.
- Device state inputs (charging, low-power mode, thermal) shift the step direction:
  charging → may raise τ; low-power mode → treat as Saver. Thermal only if readable on the
  platform; skip silently otherwise.

### 5.G Energy measurement (`energy/`)
`meter.py` exposes:
```python
with EnergyMeter() as m:        # samples power in a background thread
    ...run one request...
m.joules, m.seconds, m.avg_watts, m.backend, m.is_estimate
```
Backends, chosen automatically, in order:
1. Linux Intel/AMD: RAPL energy counters under `/sys/class/powercap` (package + DRAM if present).
2. NVIDIA GPU: NVML (`pynvml`) power readings, integrated over time; add to CPU energy.
3. macOS: `powermetrics` (requires sudo; document how to run it).
4. Fallback: time × calibrated average active power; `is_estimate=True`.

Rules:
- `calibrate_idle.py` measures idle power for 60 s; meters report **energy above idle**
  as well as total. Benchmarks use energy above idle.
- Handle counter wrap-around for RAPL.
- Record the backend in every result. Never mix backends within one comparison.

### 5.H Gateway (`gateway/`)
Endpoints:
- `POST /v1/chat/completions`: OpenAI-compatible, supports `stream: true` (SSE).
  `model` may be `"auto"` (routed) or `"lite" | "mid" | "full"` (forced; used by benchmarks).
  Non-standard info goes in an extra top-level field `x_routing`:
  `{tier, predicted_tier, confidence, fallback, escalated, pii_flag, source (cache|rules|model),
    router_ms, ttft_ms, total_ms, tokens_out, joules, joules_est_full, battery_minutes_saved,
    tau, is_estimate}`.
- `GET /v1/status`: battery (real and simulated), mode, current τ, controller state, warnings.
- `POST /v1/mode`: set mode preset or target end time.
- `GET /health`.

`pipeline.py` implements §2 steps 1–8 in order, with each step timed.
Every request writes exactly one telemetry row, including failures.

**Energy receipt** (shown under each answer):
- `joules`: measured for this request.
- `joules_est_full`: estimated energy if FULL had answered, from `energy_profile.json`
  (FULL joules per output token × tokens, plus FULL fixed overhead). Labelled estimate.
- `battery_minutes_saved = (joules_est_full − joules) / (typical_watts × 60)`,
  `typical_watts` from config. Labelled estimate.

### 5.I Telemetry (`gateway/telemetry.py`)
SQLite table `requests`:
```
id, ts, query_hash, query_text (optional, config flag), category_guess,
source, pii_flag, predicted_tier, tier, confidence, fallback, escalated, tau, mode,
router_ms, ttft_ms, total_ms, tokens_in, tokens_out, joules, joules_est_full,
battery_pct_real, battery_pct_sim, energy_backend, is_estimate, error
```
Never store query text when `pii_flag` is true, whatever the config flag says.

### 5.J Benchmark harness (`bench/`)
- `run_bench.py`: runs each policy over the router's **test** split *through the gateway*
  with energy measurement, on the demo device:
  always-full, always-lite, random (matched tier distribution), baseline router,
  BERT-Tiny router at each mode preset, oracle (cheapest correct tier from labels).
  Also FULL with and without speculative decoding.
  - Repeat each policy R times (config, default 3); report mean ± std.
  - Shuffle query order per run with a fixed seed; cool-down pause between policies.
  - Same settings (threads, context, max_tokens) for every policy.
  - Output: `results/bench_raw.csv` (one row per query per run) and `results/bench_summary.csv`.
- `build_energy_profile.py`: from forced-tier runs, compute per tier: joules per output
  token, fixed joules per request, ttft, tokens/s. Writes `results/energy_profile.json`.
  This replaces the placeholder energy weights used in `router/select_threshold.py`;
  rerun the threshold sweep after it exists.
- `analyze.py`: produces the slide figures and tables:
  - energy per query and latency (median, p95, ttft) for every policy
  - router overhead as % of one LITE request
  - accuracy vs. energy Pareto curve (τ sweep) with baselines, oracle and presets marked
  - joules per correct answer per policy
  - harmful downgrades vs. wasted escalations; per-category breakdown
  - speculative decoding speed-up and energy change on FULL
  - fallback rate and escalation rate, and accuracy recovered by each

### 5.K Dashboard (`dashboard/`)
Streamlit, two pages. Every number shows whether it is **measured**, **estimated**, or
**simulated** (small label or badge). Auto-refresh the Live page every 1–2 s.

**Live (`Home.py`)**
- Header: battery % (simulated, with real % shown smaller), mode selector / target time,
  current τ, controller warnings.
- Predicted battery life chart: projected battery % over time under current routing vs.
  always-full vs. always-lite, plus target line. Formula:
  `P = P_idle + query_rate × Σ_tier share_tier × J_tier`;
  `time_left = battery_fraction × capacity_Wh × 3600 / P`.
  `query_rate` and `share_tier` from a moving window of telemetry; `J_tier` from
  `energy_profile.json`.
- Query box to send a prompt through the gateway, showing the answer and its energy receipt.
- Live feed table of recent requests: query (hidden if PII), tier, confidence, fallback,
  escalated, latency, joules.
- Session stats: tier split, fallback rate, escalation rate, total joules saved,
  battery minutes saved, confidence histogram.

**Evaluation (`pages/Evaluation.py`)**, reading only from `results/`:
- Energy and latency bars per policy, router overhead %, speculative decoding effect.
- Pareto curve with presets, baselines and oracle.
- Router confusion matrix in plain words; per-category accuracy loss.
- Reliability diagram and ECE (from router calibration).
- Risk-control statement: "accuracy loss ≤ α with probability ≥ 1−δ", with the actual values.
- Failure gallery: test queries that were harmful downgrades, with routed answer vs. reference.

### 5.L Optional cloud tier (stretch, off by default)
- Config flag `cloud.enabled: false`. When enabled, a fourth tier above FULL for queries
  the router sends to FULL with very low confidence (config threshold).
- Queries with `pii_flag` are **never** sent to the cloud. Show "kept on-device: PII" in the UI.
- Stretch-on-stretch: redaction (replace PII spans with placeholders, send, restore locally).
- Cloud energy is not measurable on-device; report it separately and never mix it into
  on-device energy comparisons.

### 5.M Other stretch features
Only after all P0/P1 work is done:
- Explainable routing: show top contributing features from the logistic-regression baseline
  for the current query ("contains code", "math operators", "why-question").
- Implicit feedback: if the user rephrases or retries within 60 s of a LITE/MID answer,
  log it as a likely failure; offline script to fine-tune the router head on these.

---

## 6. Shared config (`config.yaml`) must include at least

```yaml
seed: 42
models: {lite: {repo, file, port: 8081}, mid: {...port: 8082}, full: {...port: 8083}}
server: {ctx_size, threads, batch_size, max_tokens: 256, speculative: true}
router: {artifacts_dir: router/artifacts}
cache: {enabled: true, size: 512, ttl_s: 3600}
pii: {ner_enabled: false}
escalation: {enabled: true, window: 16, min_tokens: 8, h_max: null}   # set after tuning
controller: {update_every_requests: 5, max_tau_step: 0.03, power_window_s: 120}
battery: {mode: simulated, capacity_wh: 50, typical_watts: 8}
energy: {backend: auto, idle_watts: null}                              # set by calibrate_idle
gateway: {host: 127.0.0.1, port: 8000}
telemetry: {db: data/telemetry.sqlite, store_query_text: true}
cloud: {enabled: false}
```
Values marked `null` must be filled by the scripts that measure them; code must refuse to
start features that need them until they are set, with a clear message.

---

## 7. Build order and priorities

**P0: the brief's requirements (must exist for judging)**
1. `shared/` + `config.yaml`.
2. Serving: download models, launch three servers, streaming client. Verify all three answer.
3. Router pipeline on a 100-query subset end to end (see `router/CLAUDE.md` §10).
4. Gateway with routing + confidence gate + telemetry (no cache/rules/PII yet).
5. Energy meter + idle calibration on the demo device.
6. Benchmark harness on the subset → energy profile → rerun threshold sweep.
7. Scale the router dataset to full size; retrain; rerun benchmarks.
8. Minimal Evaluation page: energy/latency bars, Pareto curve, harmful-downgrade analysis.

**P1: differentiators**
9. Speculative decoding on FULL, with on/off benchmark.
10. Live dashboard page with energy receipt and predicted battery life.
11. Battery controller with simulated battery and mode presets.
12. In-generation escalation, tuned on val.
13. Cache + rule engine.
14. Risk-control statement, reliability diagram, failure gallery on the Evaluation page.

**P2: stretch**
15. PII detector (P1 if the cloud tier is built).
16. Cloud tier, redaction, explainable routing, implicit feedback.

Rule: never start a P1 item while a P0 item is broken. Keep the main branch demo-able.

---

## 8. Demo script (what the final build must support)

1. Open the Live page; Balanced mode; simulated battery at 60%.
2. Send an easy query ("set a timer for 10 minutes") → rule engine, near-zero energy.
3. Send a short classification query → LITE, energy receipt shows minutes saved.
4. Send a summary → MID.
5. Send a math word problem → FULL with speculative decoding.
6. Send an ambiguous query → low confidence → visible fallback to FULL.
7. Set target "last until 22:00" → τ adjusts; projected battery line moves toward target.
8. Switch to the Evaluation page: Pareto curve, always-full vs. router energy, router
   overhead %, accuracy loss bound, one failure example.

Have a pre-recorded fallback video of this flow in case live inference fails on stage.

---

## 9. Rules for working on this code

- Read `router/CLAUDE.md` before touching anything in `router/`.
- The router's **test split is used only for final evaluation and benchmarks**. Tune nothing on it.
- Simple, readable code; it will be explained to judges.
- Every script: `argparse` CLI with `--help`, sensible defaults from `config.yaml`, prints a
  short summary at the end.
- Use `logging`, not bare prints, inside library modules.
- Type hints and pydantic schemas for anything crossing a module boundary.
- Pin dependencies; keep GPU-only packages (vllm) in `requirements-gpu.txt`.
- Tests (pytest) for: Tier enum consistency, gate and τ clamping, PII checksum validators,
  rule-engine parser safety, escalation trigger on synthetic entropy sequences,
  controller clamping and infeasible-target warning, energy-meter counter wrap-around,
  telemetry never storing PII text, and the gateway `x_routing` schema.
- Do not hide costs: cold starts, escalation waste, speculative-decoding overhead and
  router load time are all measured and reported.
- When a design choice is not covered here, ask instead of guessing.
