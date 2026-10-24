"""Run every tier on every query and save the answers (resumable).

Backends:
  llamacpp  starts one llama-server per tier (GGUF from config.yaml). With the CPU build this
            suits the 100-query subset; with the CUDA build (--llama-server, see TRAINING.md)
            it runs the full dataset on a GPU using the same quantised files the app serves.
  vllm      batched generation on a GPU (e.g. Colab) with the HF checkpoints; use for the
            full dataset. Needs `pip install vllm` (requirements-gpu.txt).

Greedy decoding, max_tokens from config, thinking mode off, prompt templates from
shared/prompts.py. Output: answers.csv with query_id, tier, answer, tokens_out, gen_seconds.
Rows are appended as they finish; (query_id, tier) pairs already present are skipped.
"""
import argparse
import csv
import logging
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.config import load_config, repo_path  # noqa: E402
from shared.prompts import build_messages  # noqa: E402
from shared.tiers import TIER_KEYS  # noqa: E402

log = logging.getLogger("generate_answers")
FIELDS = ["query_id", "tier", "answer", "tokens_out", "gen_seconds"]
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean(answer: str) -> str:
    return THINK_RE.sub("", answer or "").strip()


class AnswerWriter:
    """Thread-safe incremental CSV writer."""

    def __init__(self, path: Path):
        new = not path.exists()
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=FIELDS)
        if new:
            self.w.writeheader()
            self.f.flush()
        self.lock = threading.Lock()

    def write(self, row: dict):
        with self.lock:
            self.w.writerow(row)
            self.f.flush()

    def close(self):
        self.f.close()


def done_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["query_id", "tier"])
    return set(zip(df.query_id, df.tier))


# ---------------------------------------------------------------- llama.cpp backend

def start_llama_server(cfg, tier: str, parallel: int, server_bin: Path) -> subprocess.Popen:
    srv = cfg["server"]
    model = repo_path(srv["models_dir"]) / cfg["models"][tier]["file"]
    if not model.exists():
        sys.exit(f"missing model file {model}; download it first")
    port = cfg["models"][tier]["port"]
    try:  # never silently reuse a server someone else started (it may be a different build or model)
        requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
        sys.exit(f"port {port} is already in use (serving/launch.py or the app running?); stop it first")
    except requests.ConnectionError:
        pass
    cmd = [
        str(server_bin),
        "-m", str(model), "--port", str(port), "--host", "127.0.0.1",
        "-c", str(srv["ctx_size"] * parallel), "-np", str(parallel),
        "-t", str(srv["threads"]), "-b", str(srv["batch_size"]),
        "--jinja", "--seed", str(cfg["seed"]),
    ]
    log.info("starting %s: %s", tier, " ".join(cmd))
    # Release tarballs ship their shared libraries (CUDA runtime, ggml backends) beside the binary.
    env = {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(
        p for p in [str(server_bin.parent), os.environ.get("LD_LIBRARY_PATH", "")] if p)}
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(300):
        if proc.poll() is not None:
            sys.exit(f"llama-server for {tier} exited with code {proc.returncode}")
        try:
            if requests.get(url, timeout=1).status_code == 200:
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(1)
    proc.kill()
    sys.exit(f"llama-server for {tier} did not become healthy")


def llamacpp_one(cfg, tier: str, row) -> dict:
    port = cfg["models"][tier]["port"]
    body = {
        "messages": build_messages(row.category, row.text),
        "temperature": cfg["server"]["temperature"],
        "max_tokens": cfg["server"]["max_tokens"],
        "seed": cfg["seed"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    out = r.json()
    return dict(query_id=row.query_id, tier=tier,
                answer=clean(out["choices"][0]["message"]["content"]),
                tokens_out=out["usage"]["completion_tokens"],
                gen_seconds=round(time.perf_counter() - t0, 3))


def run_llamacpp(cfg, tier, todo: pd.DataFrame, writer: AnswerWriter, parallel: int, server_bin: Path):
    proc = start_llama_server(cfg, tier, parallel, server_bin)
    try:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(llamacpp_one, cfg, tier, row) for row in todo.itertuples()]
            for i, fut in enumerate(as_completed(futures), 1):
                writer.write(fut.result())
                if i % 10 == 0 or i == len(futures):
                    log.info("%s: %d/%d", tier, i, len(futures))
    finally:
        proc.terminate()
        proc.wait(timeout=30)


# ---------------------------------------------------------------- vLLM backend

def run_vllm(cfg, tier, todo: pd.DataFrame, writer: AnswerWriter, chunk: int):
    from vllm import LLM, SamplingParams  # GPU only

    llm = LLM(model=cfg["models"][tier]["hf_id"], seed=cfg["seed"], max_model_len=cfg["server"]["ctx_size"])
    params = SamplingParams(temperature=cfg["server"]["temperature"], max_tokens=cfg["server"]["max_tokens"])
    rows = list(todo.itertuples())
    for start in range(0, len(rows), chunk):
        batch = rows[start:start + chunk]
        t0 = time.perf_counter()
        outs = llm.chat([build_messages(r.category, r.text) for r in batch], params,
                        chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        # Batched generation has no per-query time; this is the batch average.
        per_query = (time.perf_counter() - t0) / len(batch)
        for r, o in zip(batch, outs):
            writer.write(dict(query_id=r.query_id, tier=tier, answer=clean(o.outputs[0].text),
                              tokens_out=len(o.outputs[0].token_ids), gen_seconds=round(per_query, 3)))
        log.info("%s: %d/%d", tier, start + len(batch), len(rows))
    del llm


def main():
    cfg = load_config()
    data_dir = repo_path(cfg["router"]["data_dir"])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["llamacpp", "vllm"], default="llamacpp")
    parser.add_argument("--queries", type=Path, default=data_dir / "queries.csv")
    parser.add_argument("--out", type=Path, default=data_dir / "answers.csv")
    parser.add_argument("--tiers", nargs="+", choices=TIER_KEYS, default=TIER_KEYS)
    parser.add_argument("--parallel", type=int, default=4, help="llamacpp: concurrent slots (8 suits a 12 GB GPU)")
    parser.add_argument("--llama-server", type=Path, default=repo_path(cfg["server"]["llama_server_bin"]),
                        help="llamacpp: llama-server binary (e.g. the CUDA build from TRAINING.md)")
    parser.add_argument("--chunk", type=int, default=256, help="vllm: queries per batch between checkpoints")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    queries = pd.read_csv(args.queries)
    writer = AnswerWriter(args.out)
    t0 = time.perf_counter()
    try:
        for tier in args.tiers:
            done = done_pairs(args.out)
            todo = queries[[(q, tier) not in done for q in queries.query_id]]
            log.info("%s: %d to generate, %d already done", tier, len(todo), len(queries) - len(todo))
            if todo.empty:
                continue
            if args.backend == "llamacpp":
                run_llamacpp(cfg, tier, todo, writer, args.parallel, args.llama_server.resolve())
            else:
                run_vllm(cfg, tier, todo, writer, args.chunk)
    finally:
        writer.close()

    answers = pd.read_csv(args.out)
    print(f"\nDone in {time.perf_counter() - t0:.0f}s. {args.out} has {len(answers)} rows.")
    print(answers.groupby("tier").agg(n=("query_id", "size"), mean_tokens=("tokens_out", "mean"),
                                      mean_seconds=("gen_seconds", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
