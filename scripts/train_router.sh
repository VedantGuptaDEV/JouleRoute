#!/usr/bin/env bash
# Full router pipeline on a GPU machine (see TRAINING.md). Safe to rerun: answer generation
# resumes where it stopped, and every later step rebuilds its outputs from scratch.
#   N=3500        number of queries
#   PARALLEL=8    concurrent generations per model
#   LLAMA_SERVER  llama-server binary (default: the CUDA build from setup_gpu.sh)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
N=${N:-3500}
PARALLEL=${PARALLEL:-8}
LLAMA_SERVER=${LLAMA_SERVER:-tools/llama-b11840-cuda/llama-server}
step() { echo; echo "== $*"; }

if [ ! -f router/data/queries.csv ]; then
    step "1/8 build dataset ($N queries)"
    $PY router/data/build_dataset.py --n "$N"
else
    step "1/8 dataset: keeping existing router/data/queries.csv ($(($(wc -l < router/data/queries.csv) - 1)) lines)"
fi
step "2/8 generate answers on all three tiers (resumable)"
$PY router/data/generate_answers.py --llama-server "$LLAMA_SERVER" --parallel "$PARALLEL"
step "3/8 grade";              $PY router/data/grade.py
step "4/8 labels and splits";  $PY router/data/make_labels.py
step "5/8 baseline router";    $PY router/train/train_baseline.py
step "6/8 BERT-Tiny router";   $PY router/train/train_bert.py
step "7/8 calibrate";          $PY router/train/calibrate.py --name bert_tiny
                               $PY router/train/calibrate.py --name baseline
step "8/8 threshold";          $PY router/train/select_threshold.py --name bert_tiny
step "tests";                  $PY -m pytest router/tests tests -q

echo
echo "Done. Trained router: router/artifacts/bert_tiny (thresholds.json has tau_min and the presets)."
echo "Send it back with the commands in TRAINING.md ('Bring the results back')."
