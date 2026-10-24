"""
serving/bench_spec.py
Measure FULL-tier generation speed with and without speculative decoding.

Runs 1 warm-up (discarded) + N measured runs (default 3) and reports:
  - TTFT  (time to first token, ms)
  - total (wall-clock from request to final token, ms)
  - tokens_out  (from server usage if available, else streamed-event count)
  - tok/s
  - finish_reason

Reports the median of the measured runs.

Usage:
    python serving/bench_spec.py [--spec on|off] [--runs 3] [--config PATH]
    python serving/bench_spec.py --spec on   # must already be running with spec
    python serving/bench_spec.py --spec off  # must already be running without spec

The --spec flag is ONLY used to label the result; it does not start/stop servers.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.config import load_config
from shared.tiers import Tier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("bench_spec")

# ---------------------------------------------------------------------------
# Benchmark prompt (fixed; do not change between runs)
# ---------------------------------------------------------------------------
_PROMPT = "Explain how a bicycle works in about 150 words."
_MESSAGES = [{"role": "user", "content": _PROMPT}]


def _bench_url(cfg: dict) -> str:
    host = cfg["gateway"]["host"]
    port = cfg["models"]["full"]["port"]
    return f"http://{host}:{port}/v1/chat/completions"


def run_one(cfg: dict, max_tokens: int) -> dict:
    """
    Send one streaming request to FULL and return timing + metadata.

    Token count: prefer usage.completion_tokens reported in the final chunk's
    timings (llama-server puts it in `timings.predicted_n`); fall back to
    counting streamed events.
    """
    seed = cfg.get("seed", 42)
    payload = {
        "model": "local",
        "messages": _MESSAGES,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
    }
    url = _bench_url(cfg)

    http_timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)

    t_start = time.perf_counter()
    t_first: float | None = None
    tokens_streamed = 0
    finish_reason: str | None = None
    tokens_from_server: int | None = None
    t_last = t_start

    with httpx.Client(timeout=http_timeout) as client:
        with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                if line == "data: [DONE]":
                    break
                if not line.startswith("data: "):
                    continue

                json_str = line[len("data: "):]
                try:
                    chunk = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]

                fr = choice.get("finish_reason")
                if fr is not None:
                    finish_reason = fr
                    # llama-server puts token counts in timings on the final chunk
                    timings = chunk.get("timings", {})
                    if timings:
                        tokens_from_server = timings.get("predicted_n")
                    break

                delta = choice.get("delta", {})
                text: str = delta.get("content") or ""
                if text:
                    if t_first is None:
                        t_first = time.perf_counter()
                    tokens_streamed += 1
                    t_last = time.perf_counter()

    t_end = time.perf_counter()

    # Resolve token count: prefer server-reported value
    if tokens_from_server is not None:
        tokens_out = tokens_from_server
        tok_source = "server"
    else:
        tokens_out = tokens_streamed
        tok_source = "streamed-events"

    ttft_ms = round((t_first - t_start) * 1000, 1) if t_first else None
    total_ms = round((t_end - t_start) * 1000, 1)
    tok_s = round(tokens_out / (t_end - t_start), 2) if tokens_out else 0.0

    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "tokens_out": tokens_out,
        "tok_s": tok_s,
        "finish_reason": finish_reason,
        "tok_source": tok_source,
    }


def bench(cfg: dict, n_runs: int, max_tokens: int, spec_label: str) -> dict:
    """Run 1 warm-up + n_runs measured, return median stats."""
    log.info(
        "Warm-up run (discarded) — spec=%s  max_tokens=%d  prompt=%r",
        spec_label, max_tokens, _PROMPT,
    )
    run_one(cfg, max_tokens)  # discard

    results = []
    for i in range(1, n_runs + 1):
        log.info("Measured run %d/%d ...", i, n_runs)
        r = run_one(cfg, max_tokens)
        log.info(
            "  TTFT=%.1f ms  total=%.1f ms  tokens=%d  tok/s=%.2f  finish=%s  source=%s",
            r["ttft_ms"] or 0, r["total_ms"], r["tokens_out"],
            r["tok_s"], r["finish_reason"], r["tok_source"],
        )
        results.append(r)

    ttfts   = [r["ttft_ms"]  for r in results if r["ttft_ms"] is not None]
    totals  = [r["total_ms"] for r in results]
    toks    = [r["tokens_out"] for r in results]
    tok_ss  = [r["tok_s"]    for r in results]

    return {
        "spec":           spec_label,
        "runs":           n_runs,
        "ttft_ms_median": round(statistics.median(ttfts),  1) if ttfts  else None,
        "total_ms_median":round(statistics.median(totals), 1),
        "tokens_median":  round(statistics.median(toks),   1),
        "tok_s_median":   round(statistics.median(tok_ss), 2),
        "ttft_ms_all":    ttfts,
        "total_ms_all":   totals,
        "tok_s_all":      tok_ss,
        "finish_reasons": [r["finish_reason"] for r in results],
        "tok_source":     results[0]["tok_source"],
    }


def _save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "spec", "runs",
        "ttft_ms_median", "total_ms_median", "tokens_median", "tok_s_median",
        "ttft_ms_all", "total_ms_all", "tok_s_all",
        "finish_reasons", "tok_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    log.info("Saved results to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FULL-tier speed (spec vs no-spec).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--spec", choices=["on", "off"], required=True,
        help="Label for this run: 'on' if spec decoding is active, 'off' if not.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--config", default=str(_REPO_ROOT / "config.yaml"))
    parser.add_argument(
        "--out", default=str(_REPO_ROOT / "results" / "spec_decoding.csv"),
        help="Path to output CSV (appended if it exists).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    max_tokens: int = cfg.get("bench", {}).get("max_tokens", 256)

    log.info(
        "Starting benchmark: spec=%s  runs=%d  max_tokens=%d",
        args.spec, args.runs, max_tokens,
    )

    # Verify FULL is reachable before starting
    url = _bench_url(cfg)
    try:
        health = httpx.get(
            f"http://{cfg['gateway']['host']}:{cfg['models']['full']['port']}/health",
            timeout=5.0,
        )
        health.raise_for_status()
    except Exception as exc:
        print(f"ERROR: FULL server not reachable at {url}: {exc}")
        sys.exit(1)

    result = bench(cfg, n_runs=args.runs, max_tokens=max_tokens, spec_label=args.spec)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  Speculative decoding: {args.spec.upper()}")
    print(f"  Prompt:   {_PROMPT!r}")
    print(f"  Max tokens: {max_tokens}   Runs (measured): {args.runs}")
    print(f"{'='*60}")
    print(f"  TTFT (median):    {result['ttft_ms_median']} ms")
    print(f"  Total (median):   {result['total_ms_median']} ms")
    print(f"  Tokens (median):  {result['tokens_median']}")
    print(f"  Tok/s (median):   {result['tok_s_median']}")
    print(f"  Finish reasons:   {result['finish_reasons']}")
    print(f"  Token count from: {result['tok_source']}")
    print(f"  All total_ms:     {result['total_ms_all']}")
    print(f"  All tok/s:        {result['tok_s_all']}")
    print(f"{'='*60}\n")

    # Append to CSV
    out_path = Path(args.out)
    # Read existing rows if file already present
    existing: list[dict] = []
    if out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    # Remove a prior row with the same spec label so re-runs replace it
    existing = [r for r in existing if r.get("spec") != args.spec]
    existing.append(result)
    _save_csv(existing, out_path)


if __name__ == "__main__":
    main()
