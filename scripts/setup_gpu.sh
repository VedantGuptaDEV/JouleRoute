#!/usr/bin/env bash
# One-time setup of a Linux machine with an NVIDIA GPU for training the router (see TRAINING.md):
#   1. Python 3.12 venv with requirements.txt (CPU PyTorch; the classifier trains in minutes on CPU)
#   2. llama.cpp b11840 prebuilt CUDA 12.8 build + CUDA runtime -> tools/llama-b11840-cuda/
#   3. the three GGUF models -> models/
# Needs: the NVIDIA driver (nvidia-smi works), uv, curl, tar. No CUDA toolkit needed.
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD=b11840
CUDA=12.8
DEST=tools/llama-$BUILD-cuda
BASE=https://github.com/ggml-org/llama.cpp/releases/download/$BUILD

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver first (TRAINING.md step 1)"; exit 1; }
command -v uv >/dev/null || { echo "uv not found: install it with 'curl -LsSf https://astral.sh/uv/install.sh | sh'"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== Python environment"
[ -d .venv ] || uv venv -p 3.12 .venv
uv pip install -p .venv --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

echo "== llama.cpp $BUILD (CUDA $CUDA)"
if [ -x "$DEST/llama-server" ]; then
    echo "already installed in $DEST"
else
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    for f in "llama-$BUILD-bin-ubuntu-cuda-$CUDA-x64.tar.gz" "cudart-llama-$BUILD-bin-ubuntu-cuda-$CUDA-x64.tar.gz"; do
        echo "downloading $f"
        curl -fL --progress-bar -o "$tmp/$f" "$BASE/$f"
        tar -xzf "$tmp/$f" -C "$tmp"
    done
    mkdir -p "$DEST"
    # Both tarballs unpack into their own folder; the binary and every library go side by side.
    cp -a "$tmp"/llama-$BUILD/. "$DEST"/
    cp -a "$tmp"/cudart-llama-$BUILD-bin-ubuntu-cuda-$CUDA-x64/. "$DEST"/
fi
LD_LIBRARY_PATH="$DEST" "$DEST/llama-server" --list-devices

echo "== models"
.venv/bin/python serving/download_models.py

echo
echo "Setup done. Next: scripts/train_router.sh"
