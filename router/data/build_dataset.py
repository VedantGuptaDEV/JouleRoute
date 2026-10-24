"""Download and sample router queries into one CSV.

Output: queries.csv with columns query_id, category, text, source, reference, grading_type.
`text` is what the user types (and what the router sees); the per-category system
prompt from shared/prompts.py is added only at generation time.

Data-quality choices:
  - Natural phrasing: raw utterances/questions where users would type them that way, and
    several rotating phrasings elsewhere, so the router cannot key on a fixed prefix.
  - Difficulty spread inside each category (easy + hard source), so the router has to
    learn difficulty, not just category.
  - Whitespace-tokenized sources are detokenized; duplicates and degenerate texts are dropped.
"""
import argparse
import hashlib
import json
import logging
import random
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.config import load_config, repo_path  # noqa: E402
from shared.prompts import INTENT_LABELS  # noqa: E402

log = logging.getLogger("build_dataset")

# Realistic mix: many easy queries, fewer hard ones.
CATEGORY_SHARES = {
    "intent": 0.18,
    "classification": 0.18,
    "factual_qa": 0.16,
    "rewrite": 0.10,
    "summary": 0.10,
    "math": 0.16,
    "code": 0.12,
}
MIN_SUMMARY_WORDS, MAX_SUMMARY_WORDS = 60, 220

PHRASINGS = {
    "sentiment": [
        'Is this movie review positive or negative? "{x}"',
        "Positive or negative review: {x}",
        'What\'s the sentiment of this review, positive or negative?\n"{x}"',
        'Would you call this review positive or negative? {x}',
        "Classify the sentiment (positive/negative): {x}",
    ],
    "spam": [
        'Is this text message spam or not spam? "{x}"',
        "Spam or not spam: {x}",
        'I got this SMS, is it spam or not spam?\n"{x}"',
        'Check if this message is spam or not spam: "{x}"',
    ],
    "rewrite": [
        "Fix the grammar and spelling: {x}",
        "Correct this sentence: {x}",
        "Can you fix the grammar in this? {x}",
        "Proofread this: {x}",
        "Rewrite this with correct grammar:\n{x}",
    ],
    "summary": [
        "Summarize this article in one sentence:\n\n{x}",
        "TL;DR this in one sentence:\n\n{x}",
        "Give me a one-line summary of this:\n\n{x}",
        "{x}\n\nSummarize the above in one sentence.",
    ],
    "humaneval": [
        "Complete this Python function:\n\n{x}",
        "Implement this function in Python:\n\n{x}",
    ],
}


def detokenize(s: str) -> str:
    """Undo PTB-style spacing: 'it , is n't " good "' -> 'it, isn't "good"'."""
    s = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", s)
    s = re.sub(r"([(\[{$])\s+", r"\1", s)
    s = re.sub(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", s)
    s = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', s)
    s = re.sub(r"\s+-\s+", " - ", s)
    s = " ".join(s.split())
    # Rotten Tomatoes wraps some snippets in stray single quotes; keep possessives like "films'".
    if s.startswith("'"):
        s = s[1:].lstrip()
    if s.endswith("'") and not s.endswith("s'"):
        s = s[:-1].rstrip()
    return s


INVISIBLE_RE = re.compile(r"[​-‏  ⁠﻿]")


def sample(ds, n: int, seed: int):
    if n > len(ds):
        log.warning("requested %d rows but only %d available; using all", n, len(ds))
    return ds.shuffle(seed=seed).select(range(min(n, len(ds))))


# ---------------------------------------------------------------- sources

def clinc(n, seed, rng):
    ds = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = ds.features["intent"].names
    keep = {names.index(label) for label in INTENT_LABELS}
    ds = ds.filter(lambda r: r["intent"] in keep)
    for r in sample(ds, n, seed):
        yield dict(category="intent", source="clinc150", grading_type="exact",
                   text=r["text"].strip(), reference=names[r["intent"]])


def rotten_tomatoes(n, seed, rng):
    ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
    ds = ds.filter(lambda r: 8 <= len(r["text"].split()) <= 50)
    for r in sample(ds, n, seed):
        yield dict(category="classification", source="rotten_tomatoes", grading_type="exact",
                   text=rng.choice(PHRASINGS["sentiment"]).format(x=detokenize(r["text"])),
                   reference="positive" if r["label"] == 1 else "negative")


def sms_spam(n, seed, rng):
    ds = load_dataset("ucirvine/sms_spam", split="train")
    ds = ds.filter(lambda r: len(r["sms"].split()) >= 4)
    for r in sample(ds, n, seed):
        yield dict(category="classification", source="sms_spam", grading_type="exact",
                   text=rng.choice(PHRASINGS["spam"]).format(x=" ".join(r["sms"].split())),
                   reference="spam" if r["label"] == 1 else "ham")


# Answers to these changed since the dataset was written, so their references are stale.
TIME_SENSITIVE_RE = re.compile(r"\b(current|currently|now|latest|recent|newest|present|this year|last time|next)\b", re.I)


def nq_open(n, seed, rng):
    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    ds = ds.filter(lambda r: not TIME_SENSITIVE_RE.search(r["question"]))
    for r in sample(ds, n, seed):
        q = r["question"].strip()
        yield dict(category="factual_qa", source="nq_open", grading_type="alias",
                   text=q[0].upper() + q[1:] + ("" if q.endswith("?") else "?"),
                   reference=json.dumps(sorted(set(r["answer"]))))


def trivia_qa(n, seed, rng):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    for r in sample(ds, n, seed):
        aliases = sorted(set(r["answer"]["normalized_aliases"]) | {r["answer"]["normalized_value"]})
        yield dict(category="factual_qa", source="trivia_qa", grading_type="alias",
                   text=r["question"].strip(), reference=json.dumps(aliases))


def jfleg(n, seed, rng):
    ds = load_dataset("jhu-clsp/jfleg", split="validation")
    for r in sample(ds, n, seed):
        original = detokenize(r["sentence"])
        refs = [detokenize(c) for c in r["corrections"] if c.strip()]
        if len(original.split()) < 4 or not refs:
            continue
        yield dict(category="rewrite", source="jfleg", grading_type="rouge_l",
                   text=rng.choice(PHRASINGS["rewrite"]).format(x=original),
                   reference=json.dumps({"refs": refs, "original": original}))


def xsum(n, seed, rng):
    ds = load_dataset("EdinburghNLP/xsum", split="validation")
    ds = ds.filter(lambda r: MIN_SUMMARY_WORDS <= len(r["document"].split()) <= MAX_SUMMARY_WORDS)
    for r in sample(ds, n, seed):
        doc = " ".join(r["document"].split())
        yield dict(category="summary", source="xsum", grading_type="rouge_l",
                   text=rng.choice(PHRASINGS["summary"]).format(x=doc),
                   reference=json.dumps({"refs": [r["summary"].strip()], "original": doc}))


def svamp(n, seed, rng):
    ds = load_dataset("ChilleD/SVAMP", split="train")
    for r in sample(ds, n, seed):
        body = r["Body"].strip().rstrip(".")
        yield dict(category="math", source="svamp", grading_type="number",
                   text=f"{body}. {r['Question'].strip()}",
                   reference=str(float(r["Answer"])).removesuffix(".0"))


def gsm8k(n, seed, rng):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    for r in sample(ds, n, seed):
        yield dict(category="math", source="gsm8k", grading_type="number",
                   text=r["question"].strip(), reference=r["answer"].split("####")[-1].strip().replace(",", ""))


def mbpp(n, seed, rng):
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    for r in sample(ds, n, seed):
        yield dict(category="code", source="mbpp", grading_type="code_exec",
                   text=f"{r['text'].strip()}\nYour code should pass this test:\n{r['test_list'][0]}",
                   reference=json.dumps({"setup": r["test_setup_code"], "tests": r["test_list"]}))


def humaneval(n, seed, rng):
    ds = load_dataset("openai/openai_humaneval", split="test")
    for r in sample(ds, n, seed):
        imports = "\n".join(l for l in r["prompt"].splitlines() if l.startswith(("import ", "from ")))
        yield dict(category="code", source="humaneval", grading_type="code_exec",
                   text=rng.choice(PHRASINGS["humaneval"]).format(x=r["prompt"].strip()),
                   reference=json.dumps({"setup": imports, "tests": [r["test"], f"check({r['entry_point']})"]}))


# category -> [(source builder, share of the category)]; first source is the easier one.
SOURCES = {
    "intent": [(clinc, 1.0)],
    "classification": [(rotten_tomatoes, 0.5), (sms_spam, 0.5)],
    "factual_qa": [(nq_open, 0.5), (trivia_qa, 0.5)],
    "rewrite": [(jfleg, 1.0)],
    "summary": [(xsum, 1.0)],
    "math": [(svamp, 0.4), (gsm8k, 0.6)],
    "code": [(mbpp, 0.7), (humaneval, 0.3)],
}


def query_id(source: str, text: str) -> str:
    """Stable id from content, so saved answers stay attached to the right query across rebuilds."""
    return "q" + hashlib.sha1(f"{source}\n{text}".encode()).hexdigest()[:12]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.assign(text=df.text.str.replace(INVISIBLE_RE, "", regex=True).str.replace(r"\s+\?$", "?", regex=True))
    df = df[df.text.str.split().str.len() >= 2]
    key = df.text.str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    df = df[~key.duplicated()]
    if len(df) < before:
        log.info("dropped %d duplicate or degenerate queries", before - len(df))
    return df


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=3500, help="total number of queries (use 100 for the first end-to-end run)")
    parser.add_argument("--out", type=Path, default=repo_path(cfg["router"]["data_dir"]) / "queries.csv")
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    rng = random.Random(args.seed)

    rows = []
    for category, share in CATEGORY_SHARES.items():
        n_cat = max(1, round(args.n * share))
        for builder, src_share in SOURCES[category]:
            n_src = max(1, round(n_cat * src_share))
            log.info("sampling %d %s queries from %s", n_src, category, builder.__name__)
            rows.extend(builder(n_src, args.seed, rng))

    df = clean(pd.DataFrame(rows)).reset_index(drop=True)
    df.insert(0, "query_id", [query_id(r.source, r.text) for r in df.itertuples()])
    df = df[["query_id", "category", "text", "source", "reference", "grading_type"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nWrote {len(df)} queries to {args.out}")
    print(df.groupby(["category", "source"]).size().to_string())
    words = df.text.str.split().str.len()
    print(f"\nWords per query: median {words.median():.0f}, p95 {words.quantile(0.95):.0f}")


if __name__ == "__main__":
    main()
