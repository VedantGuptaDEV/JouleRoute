# Training the router on a GPU machine (Fedora + NVIDIA)

Written for a Fedora PC with an RTX 3060 12 GB and 16 GB RAM, but works on any Linux box with
an NVIDIA GPU. The GPU matters for one step: running all three models on every query to
produce the labels. Training BERT-Tiny itself takes about 1.5 minutes on a laptop CPU
(measured: 55 ms per step, 3 learning rates x 8 epochs on 2,100 queries).

| Step | Laptop CPU (i7-13620H, measured) | RTX 3060 (estimate, not yet measured) |
|---|---|---|
| Answers from 3 models x 3,500 queries | ~11 h | ~30-90 min |
| Grade, label, train, calibrate, choose τ | a few minutes | a few minutes |

Labels are generated with llama.cpp and the **same Q4_K_M GGUF files the app serves**, so
"LITE answered this correctly" means the quantised LITE the app runs did, not the full-precision
model. (vLLM on Colab would label with full-precision weights instead.)

## 1. NVIDIA driver (once)

```bash
sudo dnf install \
  https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
  https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda
# wait a few minutes for the kernel module to build, then reboot
nvidia-smi        # must list the RTX 3060
```

With Secure Boot on, the module has to be signed and its key enrolled first (RPM Fusion's
"Secure Boot" howto). No CUDA toolkit is needed: the prebuilt llama.cpp bundles the CUDA
runtime. Its CUDA 12.8 build needs driver 570 or newer, which current RPM Fusion drivers are.

## 2. Code and setup (once)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv is not installed
git clone https://github.com/VedantGuptaDEV/JouleRoute && cd JouleRoute
git checkout gpu-training-and-chat-ui
scripts/setup_gpu.sh
```

`setup_gpu.sh` creates `.venv` (Python 3.12, CPU PyTorch), installs llama.cpp b11840 CUDA 12.8
into `tools/llama-b11840-cuda/` (prints the GPU it sees), and downloads the three GGUF models
(about 4.2 GiB) into `models/`.

## 3. Train

```bash
scripts/train_router.sh            # N=3500 PARALLEL=8 by default
```

It runs: build dataset → answers from all three tiers on the GPU → grade → labels and splits →
baseline router → BERT-Tiny → calibration → risk-controlled τ → tests. If it stops (closed
terminal, crash), run it again: answers resume where they stopped. For a long run, start it
inside `tmux` or with `nohup scripts/train_router.sh > train.log 2>&1 &`.

Stop `serving/launch.py` (or anything else on ports 8081-8083) first; answer generation refuses
to reuse a server it did not start.

Check the result:
- `router/artifacts/bert_tiny/train_log.json`: validation accuracy, macro-F1, expected cost
- `router/artifacts/bert_tiny/thresholds.json`: `tau_min` and the saver/balanced/performance
  presets; a `warning` field means no τ met the accuracy-loss bound
- `router/results/calibration.json`, `reliability_bert_tiny.png`, `threshold_sweep.csv`

Optional: with a CUDA build of PyTorch in `.venv` (see pytorch.org for the wheel matching your
driver), `train_bert.py` uses the GPU automatically (`--device auto`). Worth it mainly for the
larger MiniLM router (`train_bert.py --model sentence-transformers/all-MiniLM-L6-v2`).

## 4. Bring the results back

The trained router and data are gitignored, so add them explicitly:

```bash
git add -f router/artifacts/bert_tiny router/artifacts/baseline router/data/*.csv \
           router/data/graded_meta.json router/results
git commit -m "Router trained on the full dataset"
git push
```

On the demo laptop: `git pull`, set `router.tau_override: null` in `config.yaml` so the app
uses the trained τ and presets, and restart `gateway/app.py`.

## Running the app on the GPU machine

```bash
.venv/bin/python serving/launch.py --llama-server tools/llama-b11840-cuda/llama-server
.venv/bin/python gateway/app.py
.venv/bin/python cli/joule.py
```

## Known limit

The dataset covers seven categories (intent, classification, factual QA, rewrite, summary,
math, code). Chat-style prompts ("hey, how are you?", "explain X", "build me a website") are
not in it, so the router will still be unreliable on them until those categories and an
LLM-judge grading mode are added.
