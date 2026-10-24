#!/usr/bin/env bash
# Full router pipeline (see TRAINING.md). Runs on an NVIDIA GPU machine or CPU-only: it picks the
# CUDA llama.cpp build when nvidia-smi sees a GPU and that build is installed, else the CPU build.
# Safe to rerun: answer generation resumes where it stopped, and every later step rebuilds its
# outputs from scratch.
#   N=...         number of queries (default 3500 on GPU, 1000 on CPU: ~11 h for 3500 on a laptop CPU)
#   PARALLEL=...  concurrent generations per model (default 8 on GPU, 4 on CPU)
#   LLAMA_SERVER  llama-server binary (default: CUDA build from setup_gpu.sh, else config.yaml's CPU build)
#   TRAIN_ONLY=1  skip dataset and answer generation; regrade and retrain on the answers already there
#   MODEL=...     HF model to fine-tune (default: BERT-Tiny from config.yaml), e.g.
#                 sentence-transformers/all-MiniLM-L6-v2
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
CUDA_SERVER=tools/llama-b11840-cuda/llama-server
CPU_SERVER=tools/llama-b11840/llama-server
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && [ -x "$CUDA_SERVER" ]; then
    HW=gpu; N=${N:-3500}; PARALLEL=${PARALLEL:-8}; LLAMA_SERVER=${LLAMA_SERVER:-$CUDA_SERVER}
else
    HW=cpu; N=${N:-1000}; PARALLEL=${PARALLEL:-4}; LLAMA_SERVER=${LLAMA_SERVER:-$CPU_SERVER}
fi
TRAIN_ONLY=${TRAIN_ONLY:-0}
MODEL_ARGS=()
NAME=bert_tiny
if [ -n "${MODEL:-}" ]; then
    MODEL_ARGS=(--model "$MODEL")
    case "$MODEL" in *MiniLM*) NAME=minilm ;; esac
fi
step() { echo; echo "== $*"; }
query_count() { $PY -c "import pandas as pd; print(len(pd.read_csv('router/data/queries.csv')))"; }
# build_dataset.py drops duplicates, so the file can hold slightly fewer than N; remember the N asked for.
N_FILE=router/data/queries.n
dataset_n() { if [ -f "$N_FILE" ]; then cat "$N_FILE"; else query_count; fi; }

[ -x "$PY" ] || { echo "$PY not found: run scripts/setup_gpu.sh (GPU) or the setup in router/RUNNING.md (CPU)"; exit 1; }
echo "Hardware: $HW | llama-server: $LLAMA_SERVER | torch device: $($PY -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"

if [ "$TRAIN_ONLY" = 1 ]; then
    [ -f router/data/answers.csv ] || { echo "TRAIN_ONLY=1 needs router/data/answers.csv; run without it first"; exit 1; }
    step "1-2/8 skipped (TRAIN_ONLY=1): using $(query_count) queries in router/data/queries.csv"
else
    [ -x "$LLAMA_SERVER" ] || { echo "llama-server not found at $LLAMA_SERVER"; exit 1; }
    if [ -f router/data/queries.csv ] && [ "$(dataset_n)" != "$N" ]; then
        # A dataset of a different size: set it aside (with its answers) instead of mixing the two.
        old=router/data/previous_$(date +%Y%m%d_%H%M%S)
        step "1/8 existing dataset was built with N=$(dataset_n), not N=$N: moving it to $old"
        mkdir -p "$old"
        mv router/data/*.csv router/data/graded_meta.json "$N_FILE" "$old"/ 2>/dev/null || true
    fi
    if [ ! -f router/data/queries.csv ]; then
        step "1/8 build dataset ($N queries)"
        $PY router/data/build_dataset.py --n "$N"
        echo "$N" > "$N_FILE"
    else
        step "1/8 dataset: keeping existing router/data/queries.csv ($(query_count) queries, N=$N)"
    fi
    if [ "$HW" = cpu ]; then
        echo "Note: CPU answer generation is slow (~11 h for 3500 queries on an i7-13620H)."
        echo "Run it inside tmux or with nohup; rerunning resumes where it stopped."
    fi
    step "2/8 generate answers on all three tiers ($HW, resumable)"
    $PY router/data/generate_answers.py --llama-server "$LLAMA_SERVER" --parallel "$PARALLEL"
fi
step "3/8 grade";              $PY router/data/grade.py
step "4/8 labels and splits";  $PY router/data/make_labels.py
step "5/8 baseline router";    $PY router/train/train_baseline.py
step "6/8 fine-tune $NAME";    $PY router/train/train_bert.py "${MODEL_ARGS[@]}" --name "$NAME"
step "7/8 calibrate";          $PY router/train/calibrate.py --name "$NAME"
                               $PY router/train/calibrate.py --name baseline
step "8/8 threshold";          $PY router/train/select_threshold.py --name "$NAME"
step "tests";                  $PY -m pytest router/tests tests -q

echo
echo "Done. Trained router: router/artifacts/$NAME (thresholds.json has tau_min and the presets)."
echo "Send it back with the commands in TRAINING.md ('Bring the results back')."
