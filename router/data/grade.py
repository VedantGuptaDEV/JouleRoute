"""Mark each tier's answer correct or incorrect.

Grading per query type (no LLM judge; Qwen3.5-4B must never grade itself):
  exact      normalized label match (intent, classification)
  alias      normalized answer equals, or contains, a TriviaQA alias
  number     number after 'Answer:' (else the last number) equals the reference
  rouge_l    ROUGE-L F1 vs. human references >= per-category threshold from config;
             rewrite must also be closer to a reference than the uncorrected input was
  code_exec  extracted code + MBPP asserts run in a subprocess with a timeout

Output: graded.csv with query_id, lite_correct, mid_correct, full_correct,
and graded_meta.json recording how grading was done.
"""
import argparse
import json
import logging
import re
import resource
import string
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.config import load_config, repo_path  # noqa: E402
from shared.tiers import TIER_KEYS  # noqa: E402

log = logging.getLogger("grade")
ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
LABEL_SYNONYMS = {"not_spam": "ham", "legitimate": "ham", "not spam": "ham"}


def normalize_text(s: str) -> str:
    """SQuAD-style: lowercase, drop punctuation and articles, collapse whitespace."""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def normalize_label(s: str) -> str:
    s = str(s).strip().splitlines()[0] if str(s).strip() else ""
    s = s.lower().strip(" .\"'`*:")
    s = s.removeprefix("intent:").removeprefix("label:").strip()
    s = LABEL_SYNONYMS.get(s, s)
    return s.replace(" ", "_")


def grade_exact(answer, reference) -> bool:
    return normalize_label(answer) == normalize_label(reference)


def grade_alias(answer, reference) -> bool:
    ans = normalize_text(answer)
    if not ans:
        return False
    aliases = [normalize_text(a) for a in json.loads(reference)]
    if ans in aliases:
        return True
    # Accept short answers that contain an alias as whole words ("the eiffel tower in paris").
    return len(ans.split()) <= 10 and any(a and re.search(rf"\b{re.escape(a)}\b", ans) for a in aliases)


NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(answer: str):
    answer = str(answer)
    m = re.search(r"answer\s*[:=]\s*\$?\s*(-?\d[\d,]*\.?\d*)", answer, re.IGNORECASE)
    raw = m.group(1) if m else (NUMBER_RE.findall(answer) or [None])[-1]
    if raw is None:
        return None
    try:
        return float(raw.replace(",", "").rstrip("."))
    except ValueError:
        return None


def grade_number(answer, reference) -> bool:
    got = extract_number(answer)
    return got is not None and abs(got - float(reference)) < 1e-6


def rouge_l(answer: str, refs: list[str]) -> float:
    return max(ROUGE.score(r, str(answer))["rougeL"].fmeasure for r in refs)


def grade_rouge(answer, reference, category, thresholds) -> bool:
    ref = json.loads(reference)
    score = rouge_l(answer, ref["refs"])
    if score < thresholds[category]:
        return False
    if category == "rewrite":
        return score >= 0.999 or score > rouge_l(ref["original"], ref["refs"])
    return True


CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(answer: str) -> str:
    blocks = CODE_BLOCK_RE.findall(str(answer))
    return max(blocks, key=len) if blocks else str(answer)


def _limit_resources():
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))  # 1 GB memory
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def grade_code(answer, reference, timeout) -> bool:
    ref = json.loads(reference)
    program = "\n".join([ref["setup"] or "", extract_code(answer), *ref["tests"]])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(program)
        try:
            res = subprocess.run([sys.executable, "-I", str(path)], cwd=tmp, capture_output=True,
                                 timeout=timeout, preexec_fn=_limit_resources, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False
    return res.returncode == 0


def grade_one(row, cfg) -> bool:
    r = cfg["router"]
    if pd.isna(row.answer):
        return False
    if row.grading_type == "exact":
        return grade_exact(row.answer, row.reference)
    if row.grading_type == "alias":
        return grade_alias(row.answer, row.reference)
    if row.grading_type == "number":
        return grade_number(row.answer, row.reference)
    if row.grading_type == "rouge_l":
        return grade_rouge(row.answer, row.reference, row.category, r["rouge_l_threshold"])
    if row.grading_type == "code_exec":
        return grade_code(row.answer, row.reference, r["code_test_timeout_s"])
    raise ValueError(f"unknown grading_type {row.grading_type}")


def main():
    cfg = load_config()
    data_dir = repo_path(cfg["router"]["data_dir"])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, default=data_dir / "queries.csv")
    parser.add_argument("--answers", type=Path, default=data_dir / "answers.csv")
    parser.add_argument("--out", type=Path, default=data_dir / "graded.csv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    queries = pd.read_csv(args.queries)
    answers = pd.read_csv(args.answers).drop_duplicates(["query_id", "tier"], keep="last")
    df = answers.merge(queries, on="query_id", how="inner")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        df["correct"] = list(pool.map(lambda row: grade_one(row, cfg), df.itertuples()))

    wide = df.pivot(index="query_id", columns="tier", values="correct")
    missing = wide.isna().any(axis=1)
    if missing.any():
        log.warning("%d queries lack answers from some tier; they are dropped", missing.sum())
    wide = wide[~missing].astype(int)
    graded = wide.rename(columns={t: f"{t}_correct" for t in TIER_KEYS}).reset_index()
    graded = graded[["query_id"] + [f"{t}_correct" for t in TIER_KEYS]]
    graded.to_csv(args.out, index=False)

    meta = {
        "date": str(date.today()),
        "judge": "none (reference-based only; no LLM judge)",
        "methods": {
            "exact": "normalized label match",
            "alias": "normalized TriviaQA alias match or whole-word containment (answers <= 10 words)",
            "number": "number after 'Answer:' or last number, exact float match",
            "rouge_l": f"ROUGE-L F1 >= {cfg['router']['rouge_l_threshold']} vs. references; "
                       "rewrite must also beat the uncorrected input's ROUGE-L",
            "code_exec": f"MBPP asserts in a subprocess, timeout {cfg['router']['code_test_timeout_s']}s, 1 GB memory",
        },
        "models": {t: cfg["models"][t]["hf_id"] for t in TIER_KEYS},
    }
    args.out.with_name("graded_meta.json").write_text(json.dumps(meta, indent=2))

    summary = graded.merge(queries[["query_id", "category"]]).groupby("category")[
        [f"{t}_correct" for t in TIER_KEYS]].mean().round(2)
    print(f"\nWrote {len(graded)} graded queries to {args.out}")
    print("Accuracy per tier and category:")
    print(summary.to_string())
    print("Overall:", graded[[f"{t}_correct" for t in TIER_KEYS]].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
