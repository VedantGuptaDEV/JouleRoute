"""Measure per-tier speed: TTFT, total time, tokens/s for two prompts.

For each tier (one at a time, other servers idle):
  - 1 warm-up run (discarded)
  - 3 measured runs
  Reports median, min, max for each metric. Saves raw rows to
  results/tier_speeds.csv and prints a summary table.

If a server is not reachable the tier is reported as "not measured: server
down" and the script moves on.

Run:
    .venv\\Scripts\\python.exe serving\\measure_tiers.py
    .venv\\Scripts\\python.exe serving\\measure_tiers.py --tiers lite mid
    .venv\\Scripts\\python.exe serving\\measure_tiers.py --help

Estimated time (all 3 tiers, all prompts, with warm-ups):
  - prompt-a (max_tokens=32):  ~10-30 s per tier depending on speed
  - prompt-b (max_tokens=256): ~60-300 s per tier depending on speed
  Plan for roughly 5-15 minutes total on CPU, less if servers are fast.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

# ── make repo root importable ─────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from serving.llm_client import LLMClient  # noqa: E402
from shared.config import load_config, repo_path  # noqa: E402
from shared.tiers import Tier  # noqa: E402

# ── prompts ───────────────────────────────────────────────────────────────────
PROMPTS: list[dict] = [
    {
        "label": "a",
        "max_tokens": 32,
        "text": "What is the capital of France? Answer with only the city.",
    },
    {
        "label": "b",
        "max_tokens": 256,
        "text": "Explain how a bicycle works in about 150 words.",
    },
]

MEASURED_RUNS = 3
WARMUP_RUNS = 1


def _messages(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def measure_one(client: LLMClient, tier: Tier, prompt: dict) -> dict:
    """Run one measured pass. Returns a dict of raw metrics."""
    messages = _messages(prompt["text"])
    max_tokens = prompt["max_tokens"]
    t_start = time.perf_counter()
    ttft_ms: float | None = None
    token_count = 0
    finish_reason: str | None = None

    for ev in client.stream(tier, messages, max_tokens=max_tokens):
        if ev.text:
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t_start) * 1000.0
            token_count += max(len(ev.top_logprobs), 1)
        if ev.finish_reason:
            finish_reason = ev.finish_reason

    total_ms = (time.perf_counter() - t_start) * 1000.0

    # Decode rate excludes the first token (prompt processing is in TTFT)
    decode_s = (total_ms - (ttft_ms or 0)) / 1000.0
    tok_s = (token_count - 1) / decode_s if token_count > 1 and decode_s > 0 else None

    return {
        "ttft_ms": round(ttft_ms or 0.0, 1),
        "total_ms": round(total_ms, 1),
        "tokens": token_count,
        "tok_s": round(tok_s, 2) if tok_s is not None else None,
        "finish_reason": finish_reason or "unknown",
        "tok_source": "streamed_count",  # llama-server doesn't return usage in streaming
    }


def run_tier_prompt(
    client: LLMClient,
    tier: Tier,
    prompt: dict,
) -> dict | None:
    """
    Run warm-ups then MEASURED_RUNS for one tier+prompt.
    Returns aggregated result dict, or None if server is unreachable.
    """
    # health check first
    if not client.healthy(tier):
        return None

    print(
        f"  [{tier.key}] prompt-{prompt['label']}: "
        f"{WARMUP_RUNS} warm-up, {MEASURED_RUNS} measured ...",
        flush=True,
    )

    # warm-up (discarded)
    for i in range(WARMUP_RUNS):
        print(f"    warm-up {i+1}/{WARMUP_RUNS} ...", flush=True)
        measure_one(client, tier, prompt)

    # measured runs
    results = []
    for i in range(MEASURED_RUNS):
        print(f"    run {i+1}/{MEASURED_RUNS} ...", end=" ", flush=True)
        r = measure_one(client, tier, prompt)
        results.append(r)
        print(
            f"ttft={r['ttft_ms']:.0f}ms  total={r['total_ms']:.0f}ms  "
            f"tokens={r['tokens']}  tok/s={r['tok_s']}",
            flush=True,
        )

    ttfts  = [r["ttft_ms"] for r in results]
    totals = [r["total_ms"] for r in results]
    tokens_list = [r["tokens"] for r in results]
    tok_s_list  = [r["tok_s"] for r in results if r["tok_s"] is not None]

    return {
        "ttft_ms_median":  round(statistics.median(ttfts), 1),
        "ttft_ms_min":     round(min(ttfts), 1),
        "ttft_ms_max":     round(max(ttfts), 1),
        "total_ms_median": round(statistics.median(totals), 1),
        "total_ms_min":    round(min(totals), 1),
        "total_ms_max":    round(max(totals), 1),
        "tokens_median":   statistics.median(tokens_list),
        "tok_s_median":    round(statistics.median(tok_s_list), 2) if tok_s_list else None,
        "tok_s_min":       round(min(tok_s_list), 2) if tok_s_list else None,
        "tok_s_max":       round(max(tok_s_list), 2) if tok_s_list else None,
        "finish_reasons":  str([r["finish_reason"] for r in results]),
        "tok_source":      "streamed_count",
        # raw arrays for the CSV
        "ttft_ms_all":     str([r["ttft_ms"] for r in results]),
        "total_ms_all":    str([r["total_ms"] for r in results]),
        "tok_s_all":       str([r["tok_s"] for r in results]),
        "tokens_all":      str([r["tokens"] for r in results]),
    }


CSV_COLUMNS = [
    "tier", "prompt", "max_tokens",
    "ttft_ms_median", "ttft_ms_min", "ttft_ms_max",
    "total_ms_median", "total_ms_min", "total_ms_max",
    "tokens_median", "tok_s_median", "tok_s_min", "tok_s_max",
    "finish_reasons", "tok_source",
    "ttft_ms_all", "total_ms_all", "tok_s_all", "tokens_all",
]


def print_summary_table(rows: list[dict]) -> None:
    print()
    print("=" * 90)
    print(f"{'Tier':<6} {'Prompt':<8} {'MaxTok':<7} {'TTFT(ms)':<12} {'Total(ms)':<14} {'Tokens':<8} {'tok/s'}")
    print(f"{'':6} {'':8} {'':7} {'med[min,max]':<12} {'med[min,max]':<14} {'median':<8} {'med[min,max]'}")
    print("-" * 90)
    for r in rows:
        if r.get("_down"):
            print(f"{r['tier']:<6} {r['prompt']:<8} {'---':<7}  not measured: server down")
            continue
        tok_s_str = (
            f"{r['tok_s_median']:.2f}[{r['tok_s_min']:.2f},{r['tok_s_max']:.2f}]"
            if r["tok_s_median"] is not None
            else "N/A"
        )
        print(
            f"{r['tier']:<6} {r['prompt']:<8} {r['max_tokens']:<7}"
            f"  {r['ttft_ms_median']:.0f}[{r['ttft_ms_min']:.0f},{r['ttft_ms_max']:.0f}]"
            f"     {r['total_ms_median']:.0f}[{r['total_ms_min']:.0f},{r['total_ms_max']:.0f}]"
            f"      {r['tokens_median']:<8.0f}"
            f"  {tok_s_str}"
        )
    print("=" * 90)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=["lite", "mid", "full"],
        default=["lite", "mid", "full"],
        help="which tiers to measure (default: all)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=["a", "b"],
        default=["a", "b"],
        help="which prompts to run: a=short factual (max 32), b=explanation (max 256)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_path("results/tier_speeds.csv"),
        help="path to output CSV (default: results/tier_speeds.csv)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="llama-server host (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    client = LLMClient(cfg, host=args.host)
    selected_prompts = [p for p in PROMPTS if p["label"] in args.prompts]

    # Estimate time
    n_tiers   = len(args.tiers)
    n_prompts = len(selected_prompts)
    total_runs = n_tiers * n_prompts * (WARMUP_RUNS + MEASURED_RUNS)
    print(
        f"\nPlanning {n_tiers} tier(s) × {n_prompts} prompt(s) × "
        f"({WARMUP_RUNS} warm-up + {MEASURED_RUNS} measured) = {total_runs} total runs."
    )
    print(
        "Estimated time: ~5–20 min on CPU (mainly depends on FULL tier speed). "
        "Servers must already be running; this script does not start them.\n"
    )

    rows: list[dict] = []
    t_overall = time.perf_counter()

    for tier_key in args.tiers:
        tier = Tier[tier_key.upper()]
        print(f"\n-- {tier.name} (port {cfg['models'][tier_key]['port']}) --")
        for prompt in selected_prompts:
            agg = run_tier_prompt(client, tier, prompt)
            if agg is None:
                print(
                    f"  [{tier.key}] prompt-{prompt['label']}: "
                    "not measured: server down"
                )
                rows.append({"tier": tier.key, "prompt": prompt["label"],
                              "max_tokens": prompt["max_tokens"], "_down": True})
            else:
                row = {
                    "tier": tier.key,
                    "prompt": prompt["label"],
                    "max_tokens": prompt["max_tokens"],
                    **agg,
                }
                rows.append(row)

    elapsed = (time.perf_counter() - t_overall) / 60.0
    print(f"\nDone in {elapsed:.1f} min.")

    # ── print summary table ───────────────────────────────────────────────────
    print_summary_table(rows)

    # ── save CSV ─────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = [r for r in rows if not r.get("_down")]
    down_rows = [r for r in rows if r.get("_down")]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
        for r in down_rows:
            writer.writerow({
                "tier": r["tier"], "prompt": r["prompt"],
                "max_tokens": r["max_tokens"],
                "ttft_ms_median": "not measured: server down",
            })

    print(f"\nRaw results saved to: {args.out}")
    print(
        "\nConditions: CPU-only inference, temperature=0 (greedy), seed=42, "
        f"all 3 servers resident (but only the measured tier generating), "
        f"FULL speculative={'on' if cfg['server']['speculative'] else 'off'} "
        "(per config.yaml server.speculative)."
    )


if __name__ == "__main__":
    main()
