# CLAUDE.md: Router / Classifier Component

Context for building the **query router** of "One App, Many Modes", a hackathon project
(Cognition 2026, Edge AI track). Read this whole file before writing code.

---

## 1. Project in one paragraph

An on-device assistant runs three local LLM tiers of the same family. Running the biggest model
for every query wastes battery, so a tiny classifier (the **router**) predicts the *cheapest tier
that will answer each query correctly*. A **confidence gate** checks how sure the router is; if
confidence is below a threshold τ, the query falls back to the full model. The judges require:
(1) two or more real model tiers with a routing mechanism, (2) battery/latency benchmarks vs.
always-full, (3) a router that is cheap to run, (4) a quantified accuracy trade-off.

**This file covers only the router component:** data generation, labelling, training,
calibration, threshold selection, export, evaluation, and a runtime `Router` class.
Out of scope here: the LLM serving layer, energy measurement, the battery controller,
the PII detector, and the dashboard UI. They consume this component's outputs.

---

## 2. Fixed design decisions (do not change without asking)

| Decision | Value |
|---|---|
| Tiers | `LITE=0` Qwen3.5-0.8B, `MID=1` Qwen3.5-2B, `FULL=2` Qwen3.5-4B |
| Router task | 3-class classification: cheapest tier that answers correctly |
| Main router model | BERT-Tiny `google/bert_uncased_L-2_H-128_A-2` (~4.4M params) |
| Baseline router | Logistic regression on hashed n-grams + handcrafted features |
| Upper-bound router | `sentence-transformers/all-MiniLM-L6-v2` fine-tuned the same way |
| Max query length | 64 tokens |
| Confidence score | max of temperature-scaled softmax probabilities |
| Gate rule | `confidence >= τ` → predicted tier; else → `FULL` (fallback) |
| Threshold τ | passed in at runtime (battery controller sets it); calibration gives a floor `τ_min` |
| Runtime format | ONNX, int8 dynamic quantization, onnxruntime CPU, 1 thread |
| Language | Python 3.10+ |
| Seeds | 42 everywhere; results must be reproducible |

**Hard constraint:** the router must stay cheap. Never use an LLM, a large embedding model,
or any network call as the router. Target: median < 2 ms per query (tokenize + infer) on a
laptop CPU, single thread.

---

## 3. Repository layout to create

```
router/
├── data/
│   ├── build_dataset.py      # download + sample queries into one CSV
│   ├── generate_answers.py   # run 3 LLMs on every query (vLLM on GPU)
│   ├── grade.py              # mark each tier's answer correct / incorrect
│   └── make_labels.py        # cheapest-correct-tier labels + splits
├── train/
│   ├── features.py           # handcrafted + hashed n-gram features (baseline)
│   ├── train_baseline.py     # logistic regression router
│   ├── train_bert.py         # BERT-Tiny / MiniLM fine-tuning (3-class)
│   ├── calibrate.py          # temperature scaling + reliability diagram
│   └── select_threshold.py   # τ sweep + risk-controlled τ_min
├── export/
│   └── export_onnx.py        # ONNX export, int8 quantization, parity check
├── runtime/
│   └── router.py             # Router class used by the app (see §8)
├── eval/
│   ├── evaluate.py           # all metrics + baselines (see §9)
│   └── bench_latency.py      # router latency benchmark
├── artifacts/                # model files, temperature, thresholds (gitignored if large)
├── results/                  # JSON/CSV outputs consumed by the dashboard
└── tests/
```

A binary prototype `train_router.py` (lite vs. not-lite) already exists. Reuse its patterns
(weighted Trainer, ONNX export, quantize_dynamic, latency loop) but implement the 3-class
design described here.

---

## 4. Data pipeline

### 4.1 Build the query set (`build_dataset.py`)
Target **3,000 to 4,000 queries**, each with a reference answer, mixed across difficulty.
Suggested sources (verify dataset IDs on Hugging Face before using):

| Category | Source idea | Grading type |
|---|---|---|
| `intent` | CLINC150, MASSIVE | exact label match |
| `classification` | SST-2, SMS spam | exact label match |
| `factual_qa` | TriviaQA (short answers) | normalized match / alias list |
| `rewrite` | hand-written grammar fixes, tone changes | judge |
| `summary` | XSum (short articles only) | ROUGE-L or judge |
| `math` | GSM8K | final-number match |
| `code` | MBPP-style questions | judge or test execution |

Output: `data/queries.csv` with columns
`query_id, category, text, reference, grading_type`.

Keep the category balance roughly realistic: many easy queries, fewer hard ones. Log the
final per-category counts.

### 4.2 Generate answers (`generate_answers.py`)
- Run all three tiers on every query. Use **vLLM** on a GPU (Colab) for batching.
- Model IDs: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B` (verify on Hugging Face).
- Greedy decoding (temperature 0), `max_tokens=256`, **thinking/reasoning mode off**.
- One shared prompt template per category (e.g. "Answer with only the final number" for math)
  so grading is reliable. The same templates must be used later in the app.
- Make the script resumable: write results incrementally, skip query_ids already done.
- Output: `data/answers.csv` with `query_id, tier, answer, tokens_out, gen_seconds`.

Note: correctness labels come from this step. Energy is measured separately on the demo
device and is NOT part of this component.

### 4.3 Grade (`grade.py`)
- Exact-match types: normalize case, whitespace, punctuation; for math, extract the last number.
- Open-ended types: ROUGE-L threshold or an LLM judge. **Never use Qwen3.5-4B as the judge**,
  because it would be grading itself. Use reference answers or a larger external model, and record
  which one in the output metadata.
- Output: `data/graded.csv` with `query_id, lite_correct, mid_correct, full_correct`.

### 4.4 Labels and splits (`make_labels.py`)
```
label = 0 if lite_correct else 1 if mid_correct else 2
```
Queries no tier answers correctly get label 2 (spend the battery anyway).

Split **stratified by label**, seed 42:

| Split | Share | Used for |
|---|---|---|
| train | 60% | fitting router weights |
| val | 10% | early stopping + temperature scaling |
| calib | 15% | threshold selection / risk control only |
| test | 15% | final reported numbers only, touched once |

Output: `data/labeled.csv` with all graded columns + `label, split`.
Print label distribution per split.

---

## 5. Training

### 5.1 Baseline (`train_baseline.py`)
Features: query length (chars, tokens), digit count, operator symbols, code-like tokens
(`def`, `{`, `()`, `;`), question words (`why`, `how`, `explain`, `prove`, `calculate`),
plus hashed word 1–2-grams and char 3–5-grams (`HashingVectorizer`, 2^18 features).
Model: multinomial logistic regression with class weights. Save with joblib.
This baseline must be included in every comparison.

### 5.2 BERT-Tiny (`train_bert.py`)
- `AutoModelForSequenceClassification`, `num_labels=3`, max_length 64.
- Learning rate grid `{3e-4, 1e-4, 5e-5}`, batch 32, 4–8 epochs, warmup 10%, weight decay 0.01.
- Select the best checkpoint by validation **expected-cost** (see below), not accuracy.
- Same script must also train MiniLM via a `--model` flag.

**Cost-sensitive loss.** Under-routing (predicting a cheaper tier than needed) hurts accuracy;
over-routing only wastes energy. Use a cost matrix `C[true][pred]`, configurable via CLI:

```
            pred=LITE  pred=MID  pred=FULL
true=LITE      0          1         2        # wasted energy
true=MID       4          0         1
true=FULL      6          4         0        # harmful downgrades
```

Implement as weighted cross-entropy (class weights from label frequency) **plus** an
expected-cost term: `loss = CE + λ * sum_j p_j * C[y, j]`, λ default 0.5.
Report validation expected cost per epoch.

### 5.3 Temperature scaling (`calibrate.py`)
- Fit a single scalar `T > 0` on **val** logits by minimizing NLL (LBFGS or grid search).
- Save `artifacts/temperature.json`.
- Produce a reliability diagram (10 bins) and ECE before/after, saved to `results/`.

---

## 6. Threshold selection (`select_threshold.py`)

Use the **calib** split only.

Per-query loss for a given τ (this is exactly "accuracy lost vs. always-full"):
```
loss_i(τ) = 1  if routed tier answered incorrectly AND full_correct
            0  otherwise
```
Raising τ only moves queries to FULL, so loss is non-increasing in τ. This monotonicity
is what makes risk control valid.

1. **Sweep** τ over `[0.30, 0.99]` in steps of 0.01. For each τ, record: tier distribution,
   fallback rate, accuracy loss, expected relative energy (using placeholder per-tier energy
   weights `{LITE:1, MID:2.5, FULL:5}` from config, to be replaced by measured joules).
   Save to `results/threshold_sweep.csv`. This drives the Pareto curve.
2. **Risk-controlled floor τ_min:** given target `α` (default 0.03) and `δ` (default 0.05),
   choose the smallest τ whose **binomial (Clopper–Pearson) upper confidence bound** on
   calib accuracy loss is ≤ α. Scan τ from high to low and stop at the first violation
   (fixed-sequence testing). Statement this supports: "accuracy loss ≤ α with probability ≥ 1−δ".
3. Also compute the simpler conformal-risk-control value
   `inf{τ : (n·R̂(τ) + 1)/(n + 1) ≤ α}`, which bounds *expected* loss. Report both.
4. Save `artifacts/thresholds.json`:
   `{ "alpha", "delta", "tau_min", "tau_crc", "default_tau", "mode_presets": {saver, balanced, performance} }`.
   Presets must all be ≥ τ_min. If no τ satisfies α, say so loudly; do not silently pick one.

---

## 7. Export (`export_onnx.py`)
- Export BERT-Tiny (and MiniLM) to ONNX, opset 17, dynamic batch and sequence axes,
  inputs `input_ids, attention_mask, token_type_ids` (MiniLM may omit `token_type_ids`;
  handle both).
- int8 via `onnxruntime.quantization.quantize_dynamic`.
- **Parity check** on the test split: agreement rate between fp32 and int8 routed tiers
  (target ≥ 99%) and max abs probability difference. Fail loudly if agreement < 97%.
- Report file sizes for fp32 and int8.

---

## 8. Runtime interface (`runtime/router.py`)

This is the contract the rest of the app depends on. Keep it stable.
`Tier` must be imported from `shared/tiers.py` (see the root `CLAUDE.md`), not redefined;
it is shown below only for reference.

```python
from dataclasses import dataclass
from enum import IntEnum

class Tier(IntEnum):
    LITE = 0
    MID = 1
    FULL = 2

@dataclass
class RouteDecision:
    tier: Tier               # final tier after the gate
    predicted_tier: Tier     # router's raw prediction
    confidence: float        # calibrated max probability, 0..1
    probs: list[float]       # calibrated probabilities [lite, mid, full]
    fallback: bool           # True if gate overrode the prediction
    tau: float               # threshold actually used
    latency_ms: float        # tokenize + infer + gate

class Router:
    def __init__(self, artifacts_dir: str): ...      # load int8 ONNX, tokenizer, T, thresholds; 1 thread
    def route(self, text: str, tau: float | None = None) -> RouteDecision: ...
        # tau=None → default_tau; tau below tau_min is clamped up to tau_min
```

Requirements: load everything once in `__init__`; no network; deterministic; never raises on
empty or very long input (empty → FULL with fallback=True; long → truncate to 64 tokens).

---

## 9. Evaluation (`evaluate.py`, `bench_latency.py`)

Run on the **test** split only. Policies to compare:

| Policy | Description |
|---|---|
| always-full | every query → FULL |
| always-lite | every query → LITE |
| random | random tiers matching the router's tier distribution |
| baseline | logistic regression router + same gate |
| bert-tiny | main router + gate (at default τ and each preset) |
| minilm | upper-bound router + gate |
| oracle | cheapest correct tier (perfect hindsight) |

Metrics per policy:
- end-task accuracy, and accuracy loss vs. always-full
- tier distribution and fallback rate
- relative energy (placeholder weights from config until real joules arrive)
- **joules (relative) per correct answer**
- confusion matrix of predicted vs. true label; separately count *harmful downgrades*
  (routed cheaper than needed and answer wrong) and *wasted escalations*
- per-category breakdown of all of the above
- router accuracy, macro-F1, ECE

Latency (`bench_latency.py`): int8 ONNX, 1 thread, 20 warm-up calls, then median and p95
over ≥ 300 test queries, including tokenization and the gate. Report one-time load time
separately. Do the same for the baseline and MiniLM.

### Output files for the dashboard (schemas are a contract)
- `results/router_eval.json`: all metrics above, keyed by policy, plus metadata
  (model ids, quantization, judge used, α, δ, τ values, date).
- `results/test_predictions.csv`: one row per test query:
  `query_id, category, text, label, predicted_tier, routed_tier, confidence, fallback,
   lite_correct, mid_correct, full_correct, routed_correct, harmful_downgrade`.
- `results/threshold_sweep.csv`: from §6.
- `results/calibration.json` and the reliability diagram image.
- `results/latency.json`: per router, `median_ms, p95_ms, load_ms`.

---

## 10. Build order

1. `build_dataset.py` on a small subset (100 queries) → verify end to end first.
2. `generate_answers.py` + `grade.py` + `make_labels.py` on the subset.
3. `train_baseline.py` → first numbers.
4. `train_bert.py` → `calibrate.py` → `select_threshold.py`.
5. `export_onnx.py` → `runtime/router.py` → `bench_latency.py`.
6. `evaluate.py`.
7. Scale to the full dataset and rerun everything.
8. MiniLM comparison last, only if time allows.

Always get the pipeline working on small data before scaling up.

---

## 11. Rules for working on this code

- The **test split is used once**, for final numbers. Never tune anything on it.
- Never report a number that wasn't produced by a script in this repo.
- Keep placeholder energy weights clearly labelled as placeholders everywhere they appear.
- Prefer simple, readable code over clever code. This will be explained to judges.
- Every script: `argparse` CLI, `--help` text, sensible defaults, prints a short summary at the end.
- Pin dependencies in `requirements.txt` (transformers, datasets, torch, scikit-learn,
  onnx, onnxruntime, pandas, numpy, scipy, matplotlib; vllm in a separate GPU requirements file).
- Add small unit tests for: label creation, the gate rule, τ clamping, risk-control threshold
  selection on synthetic data, and the `Router` contract.
- When unsure about a design choice not covered here, ask instead of guessing.
