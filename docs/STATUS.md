# JouleRoute — Project Status

**Date:** 2026-10-25  
**Device:** Windows 10/11 x64, CPU-only (no GPU), 8 threads  
**Repo root:** `JouleRoute/`

JouleRoute is an on-device AI inference layer that routes every user query to the cheapest
of three local LLMs (Qwen3.5 0.8B / 2B / 4B) that can answer it correctly, using a
BERT-Tiny router with a confidence gate, so easy queries use less power and hard ones
still reach the strong model. The full architecture is specified in `Readme.md`.

---

## 1. Component Status

| Component | Path | Status | Evidence |
|---|---|---|---|
| Tier enum | `shared/tiers.py` | **exists, tested** | 23/23 pytest pass |
| Config loader | `shared/config.py` | **exists, tested** | imported by all test fixtures |
| Pydantic schemas | `shared/schemas.py` | **exists, tested** | `XRouting`, `Step`, `ChatRequest` used in pipeline tests |
| Prompt templates | `shared/prompts.py` | **exists, untested** | file present, no dedicated test |
| Model download | `serving/download_models.py` | **exists, untested** | file present; models already in `models/` |
| Server launch | `serving/launch.py` | **exists, verified running** | all 3 servers healthy at time of measurement (see §3) |
| Streaming client | `serving/llm_client.py` | **exists, tested** | `test_llm_stream_survives_unicode_line_separators` passes; used by measure_tiers.py |
| Tier speed benchmark | `serving/measure_tiers.py` | **exists, ran successfully** | `results/tier_speeds.csv` written 2026-10-25 |
| FastAPI gateway | `gateway/app.py` | **exists, untested end-to-end** | file present; pipeline unit-tested with fakes |
| Request pipeline | `gateway/pipeline.py` | **exists, tested** | 9 pipeline tests pass (cache, PII, router, gate, generate, telemetry, cancel, fallback, forced-tier) |
| Telemetry (SQLite) | `gateway/telemetry.py` | **exists, tested** | `test_pii_query_text_never_stored_and_not_cached`, `test_llm_down_still_writes_telemetry` pass |
| Router runtime | `router/runtime/router.py` | **exists, tested** | `test_tau_override_replaces_risk_controlled_tau` passes; backend = torch-fp32 (no ONNX export yet) |
| Router artifacts | `router/artifacts/bert_tiny/` | **exists, limited quality** | `model.safetensors` present; `thresholds.json` warns tau_min=1.0 (see §6) |
| Answer cache | `precheck/cache.py` | **exists, tested** | `test_cache_normalizes_and_evicts`, `test_second_identical_query_hits_cache` pass |
| Rule engine | `precheck/rules.py` | **MISSING** | file does not exist |
| PII detector | `privacy/pii.py` | **exists, tested** | 9 PII tests pass (email, phone, card/Luhn, Aadhaar/Verhoeff, PAN, IFSC, IPv4, keywords, clean) |
| CLI (TUI chat) | `cli/joule.py`, `cli/tui.py`, `cli/render.py`, `cli/client.py` | **exists, untested** | files present; no automated test |
| Scripts | `scripts/run_app.cmd`, `run_app.py`, `setup.cmd`, `setup.sh` | **exists** | files present |
| Dashboard (Live) | `dashboard/Home.py`, `dashboard/data.py` | **exists, runs** | Streamlit server started successfully on port 8501 |
| Dashboard (Evaluation) | `dashboard/pages/Evaluation.py` | **MISSING** | directory `pages/` does not exist |
| Energy meter | `energy/` | **MISSING** | directory does not exist |
| Battery controller | `control/` | **MISSING** | directory does not exist |
| Benchmark harness | `bench/` | **MISSING** | directory does not exist |
| Tests | `tests/test_pipeline.py` | **23/23 pass** | `pytest tests/` run 2026-10-25, 8.55 s |

---

## 2. Measured Numbers

### 2a. Model file sizes

Source: `models/` directory, read 2026-10-25.

| Tier | File | Size on disk |
|---|---|---|
| LITE | `Qwen_Qwen3.5-0.8B-Q4_K_M.gguf` | 552.9 MB (0.540 GiB) [measured] |
| MID  | `Qwen_Qwen3.5-2B-Q4_K_M.gguf`   | 1331.7 MB (1.300 GiB) [measured] |
| FULL | `Qwen_Qwen3.5-4B-Q4_K_M.gguf`   | 2874.5 MB (2.806 GiB) [measured] |

### 2b. Per-tier inference speeds

Source: `results/tier_speeds.csv`, measured by `serving/measure_tiers.py` on 2026-10-25.

**Conditions:** CPU-only, Windows 10/11 x64, 8 threads (`server.threads=8`), greedy
decoding (temperature=0, seed=42), all 3 servers resident (but only the measured tier
generating at each moment), FULL tier started with speculative decoding ON
(`server.speculative=true`, LITE as draft model), context size 4096. 1 warm-up run
discarded, 3 runs measured, median reported. Token count is the streamed token-piece
count from the client (server does not return usage in streaming mode).

#### Prompt A — "What is the capital of France? Answer with only the city." (max_tokens=32)

| Tier | TTFT median [min, max] | Total median [min, max] | Tokens | tok/s | finish_reason |
|---|---|---|---|---|---|
| LITE | 459 ms [417, 465] [measured] | 525 ms [520, 543] [measured] | 1 | N/A (1 token) | stop |
| MID  | 224 ms [218, 228] [measured] | 294 ms [289, 296] [measured] | 1 | N/A (1 token) | stop |
| FULL | 700 ms [663, 739] [measured] | 829 ms [808, 902] [measured] | 1 | N/A (1 token) | stop |

*Note: tok/s is undefined for 1-token replies (no decode phase after first token).*

#### Prompt B — "Explain how a bicycle works in about 150 words." (max_tokens=256)

| Tier | TTFT median [min, max] | Total median [min, max] | Tokens (median) | tok/s median [min, max] | finish_reason |
|---|---|---|---|---|---|
| LITE | 173 ms [154, 880] [measured] | 9009 ms [8756, 15279] [measured] | 256 | 28.80 [17.71, 29.71] | **length** (hit max_tokens cap) |
| MID  | 234 ms [188, 343] [measured] | 11613 ms [11117, 12626] [measured] | 187 | 16.35 [15.14, 17.02] | stop |
| FULL | 666 ms [663, 673] [measured] | 22408 ms [21868, 22433] [measured] | 179 | 8.19 [8.18, 8.40] [measured] | stop |

*LITE hit the 256-token cap on all 3 runs (finish_reason=length), so its real answer would
be longer and total time higher. The tok/s spread for LITE is large (17–30) because run 1
had a cold warm-up artefact (880 ms TTFT, 15.3 s total); runs 2 and 3 were consistent at
~165 ms TTFT and ~8.9 s total.*

### 2c. Speculative decoding comparison (FULL tier)

Source: `results/spec_decoding.csv`.

**Conditions:** FULL tier only, CPU-only, 3 measured runs each, prompt not recorded in
CSV (separate experiment), 180 output tokens, temperature=0, seed=42.

| Spec decoding | TTFT median | Total median | Tokens | tok/s median | tok/s range | finish_reason |
|---|---|---|---|---|---|---|
| ON  (LITE drafts) | 1932.7 ms [measured] | 115189 ms = 115.2 s [measured] | 180 | 1.56 [measured] | [1.51, 1.73] | stop |
| OFF | 1132.2 ms [measured] | 23282.8 ms = 23.3 s [measured] | 180 | 7.73 [measured] | [7.70, 7.74] | stop |

**Result: speculative decoding is ~5× SLOWER than no speculation on this CPU-only setup.**
With speculation ON, FULL generates at 1.56 tok/s vs. 7.73 tok/s without. This is
expected: llama.cpp's speculative decoding is designed for GPU where draft+verify is
faster than sequential generation; on CPU the draft+verify loop is slower than a single
forward pass. **Decision: disable speculative decoding for benchmarks and the demo.**
See §5 for the config change needed.

### 2d. Router

Source: `router/artifacts/bert_tiny/train_log.json`, `thresholds.json`.

| Metric | Value | Tag |
|---|---|---|
| Base model | google/bert_uncased_L-2_H-128_A-2 | [measured] |
| Backend at runtime | torch-fp32 (no ONNX export yet) | [measured] |
| Training set size | 100-query subset | [measured] |
| Val set size | 10 queries | [measured] |
| Best val accuracy | 0.20 (2/10) at epoch 2 | [measured] |
| Best val macro-F1 | 0.229 | [measured] |
| Training time | 32.4 s | [measured] |
| tau_min | 1.0 (sends everything to FULL) | [measured] |
| tau_override in config | 0.32 (manual, no accuracy guarantee) | [measured] |
| Warning | "NO tau satisfies alpha=0.03 at delta=0.05 on 15 calib queries" | [measured] |

*Router latency (inference time per query) was not measured in this session; not measured yet.*

---

## 3. Numbers Reported by User (Not Re-Measured This Session)

These were provided before this measurement run. They are preserved here for comparison.
Tag: [reported by user, not re-measured].

| Description | Reported value | Fresh measurement | Match? |
|---|---|---|---|
| LITE smoke test (8-tok reply, spec ON): tok/s | 15.4 tok/s | not directly comparable (different prompt, 1 token for prompt-a, cap hit for prompt-b) | cannot compare |
| MID smoke test (8-tok reply, spec ON): tok/s | 11.2 tok/s | not directly comparable | cannot compare |
| FULL smoke test (8-tok reply, spec ON): tok/s | 3.2 tok/s | not directly comparable | cannot compare |
| FULL single run, spec OFF, 179 tok: tok/s | 6.7 tok/s | 7.73 tok/s (from spec_decoding.csv, 180 tok, 3-run median) | **differs ~15%** — within 20% threshold, no flag needed; single run vs. median likely accounts for variance |
| FULL single run, spec OFF, 179 tok: TTFT | 1.1 s (1100 ms) | 1132 ms median from spec_decoding.csv | **matches** (<3% diff) |
| FULL single run, spec OFF, 179 tok: total | 26.6 s | 23.3 s median | **differs ~14%** — within 20%, single run vs. median variance |
| Spec ON 3-run median: tok/s | 1.56 tok/s | 1.56 tok/s (spec_decoding.csv) | **exact match** [measured] |
| Spec OFF 3-run median: tok/s | 7.73 tok/s | 7.73 tok/s (spec_decoding.csv) | **exact match** [measured] |
| Spec ratio: ON vs OFF | ~5× slower | 7.73/1.56 = 4.96× | **matches** |

**No number differs by more than 20% from a directly comparable fresh measurement.**
The smoke-test tok/s numbers (LITE 15.4, MID 11.2, FULL 3.2 with spec ON) cannot be
compared to prompt-b results here because the prompts and token counts are different.

---

## 4. Key Decisions and Rationale

| Decision | Rationale |
|---|---|
| **Speculative decoding: disable for benchmarks and demo** | Measured 5× slowdown on CPU (1.56 vs 7.73 tok/s on FULL). llama.cpp speculation is designed for GPU. Set `server.speculative: false` in config.yaml before running benchmarks. |
| **max_tokens=256 for app default** | config.yaml `server.max_tokens=256`. LITE hits this cap for longer prompts (finish_reason=length). Adjust upward for the demo if needed. |
| **All 3 servers resident** | Default in config; cold-start time and energy not yet measured. Lazy-start is implemented as an option in `launch.py` but not exercised. |
| **tau_override=0.32** | The trained router has tau_min=1.0 (routes everything to FULL) because the calibration set is only 15 queries and no tau satisfies the risk-control bound. A manual tau_override of 0.32 is set in config.yaml to allow LITE/MID usage for the demo. The accuracy guarantee does not hold with this override. |
| **fp32 PyTorch router, no ONNX** | `model.int8.onnx` has not been exported. Router runs at ~fp32 speed. An export script is needed before the "~1 ms router" claim can be made. |
| **No max_tokens cap in gateway by default** | `ChatRequest.max_tokens = None` (no cap); model stops at end-of-answer. Benchmarks in `serving/measure_tiers.py` pass explicit max_tokens per prompt. |
| **Telemetry: store_query_text=true** | Config default. PII-flagged queries are never stored regardless. |

---

## 5. Known Issues and Risks

| Issue | Severity | Notes |
|---|---|---|
| **Speculative decoding is 5× slower on CPU** | High — must fix before demo | Set `server.speculative: false` in config.yaml. Re-run spec_decoding benchmark after to confirm OFF is faster. |
| **Router tau_min=1.0 — routes everything to FULL** | High — routing does not work without tau_override | tau_override=0.32 is a workaround; retrain on full dataset to get a valid risk-controlled tau. |
| **Router trained on 100-query subset, accuracy 20%** | High | Val set of 10 queries is too small to trust accuracy numbers. Scale dataset per Readme §7 step 7. |
| **No ONNX export** | Medium | Router runs fp32 PyTorch; claimed "~1 ms" routing latency is unverified. Export `model.int8.onnx` and measure. |
| **energy/, control/, bench/ missing** | High for judging | P0 judging requirement "battery/latency benchmarks vs. always-full" needs `bench/` and `energy/`. |
| **No energy meter** | High | `joules`, `joules_est_full`, `battery_minutes_saved` are null in all telemetry rows. Energy-receipt UI and battery projections use placeholder estimates only. |
| **LITE prompt-b TTFT run-to-run variance** | Low | Run 1 had 880 ms TTFT vs. 154–173 ms on runs 2-3. Likely a post-warmup KV-cache artefact. The warmup step in measure_tiers.py should mitigate this; adding a second warmup would help. |
| **precheck/rules.py missing** | Medium | Rule engine not present; demo steps 2 ("set a timer") won't route through the rule engine as designed. |
| **dashboard/pages/Evaluation.py missing** | Medium | Evaluation dashboard page does not exist. |
| **No test for gateway integration (live HTTP)** | Medium | All pipeline tests use fakes for LLM and router; gateway/app.py is not tested end-to-end. |
| **Windows: no RAPL / NVML energy backend** | Medium | `energy/meter.py` does not exist yet, but when built: RAPL is Linux-only, NVML needs NVIDIA GPU. On this Windows CPU-only device the energy meter will fall back to "time × calibrated idle power" (is_estimate=True). |
| **No idle power calibration** | Low (until energy meter exists) | `energy.idle_watts` is null in config.yaml. `energy/calibrate_idle.py` not written. |
| **Streamlit autorefresh dependency** | Low | `streamlit-autorefresh` not in requirements.txt; dashboard degrades gracefully (manual refresh) if missing. |

---

## 6. What Is Left (Priority Order from Readme §7)

### P0 — Required for judging

| # | Item | Status |
|---|---|---|
| P0-5 | Energy meter + idle calibration | **not started** — `energy/` missing |
| P0-6 | Benchmark harness on subset → energy profile → rerun threshold sweep | **not started** — `bench/` missing |
| P0-7 | Scale router dataset to full size; retrain; rerun benchmarks | **not started** — needs P0-6 first |
| P0-8 | Evaluation page: energy/latency bars, Pareto curve, harmful-downgrade analysis | **not started** — `dashboard/pages/Evaluation.py` missing |

*P0 items 1–4 (shared/, serving, router pipeline, gateway with routing+telemetry) are
substantially complete — code exists and pipeline tests pass. The remaining P0 gap is
everything related to energy measurement and the benchmark harness.*

### P1 — Differentiators

| # | Item | Status |
|---|---|---|
| P1-9  | Speculative decoding on/off benchmark | **partially done** — `results/spec_decoding.csv` exists; needs to be re-run with spec OFF confirmed in config and integrated into bench harness |
| P1-10 | Live dashboard with energy receipt and predicted battery life | **partial** — `dashboard/Home.py` exists; energy fields are all null (no meter) |
| P1-11 | Battery controller with simulated battery and mode presets | **not started** — `control/` missing |
| P1-12 | In-generation escalation, tuned on val | **not started** — `control/escalation.py` missing; `h_max` null in config |
| P1-13 | Cache + rule engine | **cache done** — `precheck/cache.py` exists and tested; `precheck/rules.py` missing |
| P1-14 | Risk-control statement, reliability diagram, failure gallery on Evaluation page | **not started** |

### P2 — Stretch

| # | Item | Status |
|---|---|---|
| P2-15 | PII detector (already P1 if cloud enabled) | **done** — `privacy/pii.py` exists and fully tested |
| P2-16 | Cloud tier, redaction, explainable routing, implicit feedback | **not started** |

---

## 7. How to Reproduce Each Measured Number

### Pytest (23 passed)
```
cd JouleRoute
.venv\Scripts\python.exe -m pytest tests\ -v
```
Expected: 23 passed, 0 failed. Python 3.13.15, pytest 9.1.1.

### Model file sizes
```
cd JouleRoute
.venv\Scripts\python.exe -c "
from pathlib import Path
for f in ['models/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf',
          'models/Qwen_Qwen3.5-2B-Q4_K_M.gguf',
          'models/Qwen_Qwen3.5-4B-Q4_K_M.gguf']:
    p = Path(f)
    print(f, p.stat().st_size, 'bytes', round(p.stat().st_size/2**30,3), 'GiB')
"
```

### Tier speeds (results/tier_speeds.csv)
Requires all 3 llama-servers running. Start them first:
```
REM Terminal 1 — start servers (blocks until Ctrl-C)
.venv\Scripts\python.exe serving\launch.py
```
Then in another terminal:
```
REM Terminal 2 — run measurements (~5-15 min on CPU)
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe serving\measure_tiers.py
```
Output saved to `results/tier_speeds.csv`.

### Speculative decoding comparison (results/spec_decoding.csv)
The script that produced `results/spec_decoding.csv` is not in the repo (was run
externally). To re-measure: run `serving/measure_tiers.py` with FULL only, once with
`server.speculative: true` and once with `server.speculative: false` in config.yaml,
on the same long-output prompt. A dedicated `bench/build_energy_profile.py` (not yet
written) will do this systematically.

### Router artifacts
```
REM Read thresholds and training log directly
type router\artifacts\bert_tiny\thresholds.json
type router\artifacts\bert_tiny\train_log.json
```

---

## 8. Dashboard Status

| Item | Status |
|---|---|
| File | `dashboard/Home.py` (598 lines), `dashboard/data.py` (helper functions) |
| How to run | `.venv\Scripts\python.exe -m streamlit run dashboard\Home.py` — opens at `http://localhost:8501` |
| Demo mode | `streamlit run dashboard\Home.py -- --demo` uses `data/telemetry_demo.sqlite` |
| Auto-refresh | Every 2 s if `streamlit-autorefresh` is installed (graceful degradation otherwise) |
| Data source | `data/telemetry.sqlite` (live) or `data/telemetry_demo.sqlite` (demo) |
| Energy fields | All null — energy meter not built. Joules, joules_est_full, battery_minutes_saved are placeholder estimates labelled as such |
| Evaluation page | `dashboard/pages/Evaluation.py` — **does not exist** |
| Demo seed data | `dashboard/seed_demo_data.py` exists to populate `data/telemetry_demo.sqlite` with synthetic rows; run it separately if needed |
| Known gap | Without the energy meter and `results/energy_profile.json`, the "energy saved" and "battery minutes saved" cards on the Live page use placeholder ratios from config.yaml (`router.energy_weights_placeholder: {lite:1.0, mid:2.5, full:5.0}`) and are labelled [estimated] |
