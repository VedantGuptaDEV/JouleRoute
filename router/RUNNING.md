# Running the router pipeline

Setup (CPU, Python 3.12):

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Local answer generation needs `tools/llama-b11840/llama-server` (llama.cpp CPU release b11840)
and the three `Q4_K_M` GGUFs from `config.yaml` in `models/`.

## Steps

```bash
PY=.venv/bin/python
$PY router/data/build_dataset.py --n 100          # 3500 for the full run
$PY router/data/generate_answers.py               # resumable; --backend vllm on a GPU
$PY router/data/grade.py
$PY router/data/make_labels.py
$PY router/train/train_baseline.py
$PY router/train/train_bert.py                    # --model sentence-transformers/all-MiniLM-L6-v2 for the upper bound
$PY router/train/calibrate.py --name bert_tiny    # also: --name baseline
$PY router/train/select_threshold.py --name bert_tiny
$PY -m pytest router/tests
```

## Full-scale answer generation on Colab (GPU)

CPU generation of 3,500 queries x 3 tiers takes many hours. On a Colab GPU:

```bash
git clone <this repo> && cd JouleRoute
pip install -r requirements-gpu.txt pandas pyyaml requests
python router/data/build_dataset.py --n 3500      # or upload router/data/queries.csv
python router/data/generate_answers.py --backend vllm
```

Download `router/data/answers.csv` and continue from `grade.py` locally. The script appends
rows as it goes, so a disconnected Colab session can be resumed with the same command.

## Notes

- Risk control needs data: with zero observed losses, the Clopper-Pearson 95% bound is
  1 - 0.05^(1/n), so `alpha = 0.03` needs at least ~100 **calib** queries (~670 total) even
  for a perfect router. Subset runs will report "no tau satisfies alpha"; that is expected.
- Grading is reference-based (no LLM judge). Summary uses ROUGE-L >= 0.25 against the XSum
  reference, which is a noisy proxy; an external judge (not Qwen3.5-4B) would give cleaner labels.
- Energy numbers in `threshold_sweep.csv` use placeholder tier weights until
  `results/energy_profile.json` is measured on the demo device.
