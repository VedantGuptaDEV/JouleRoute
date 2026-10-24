#!/usr/bin/env bash
# One-time setup after cloning, on any Linux x86-64 machine (CPU-only or NVIDIA GPU):
#   1. Python 3.12 venv with requirements.txt
#   2. llama.cpp b11840 CPU build -> tools/llama-b11840/ (what config.yaml serves with), and on a
#      GPU machine also the CUDA 12.8 build + CUDA runtime -> tools/llama-b11840-cuda/
#   3. the three GGUF models -> models/ (about 4.2 GiB; files already there are kept)
# Needs: uv, curl, tar. No CUDA toolkit needed. Safe to rerun: finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD=b11840
CUDA=12.8
BASE=https://github.com/ggml-org/llama.cpp/releases/download/$BUILD

command -v uv >/dev/null || { echo "uv not found: install it with 'curl -LsSf https://astral.sh/uv/install.sh | sh'"; exit 1; }
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    GPU=1
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    GPU=0
    echo "No NVIDIA GPU found: CPU-only setup"
fi

echo "== Python environment"
[ -d .venv ] || uv venv -p 3.12 .venv
uv pip install -p .venv --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# install_llama <dest> <tarball>...: unpack release tarballs side by side into <dest>
install_llama() {
    local dest=$1; shift
    if [ -x "$dest/llama-server" ]; then
        echo "already installed in $dest"
    else
        local tmp
        tmp=$(mktemp -d)
        for f in "$@"; do
            echo "downloading $f"
            curl -fL --progress-bar -o "$tmp/$f" "$BASE/$f"
            tar -xzf "$tmp/$f" -C "$tmp"
            rm "$tmp/$f"
        done
        mkdir -p "$dest"
        # Each tarball unpacks into its own folder; the binary and every library go side by side.
        for d in "$tmp"/*/; do cp -a "$d". "$dest"/; done
        rm -rf "$tmp"
    fi
    LD_LIBRARY_PATH="$dest" "$dest/llama-server" --list-devices
}

echo "== llama.cpp $BUILD (CPU)"
install_llama "tools/llama-$BUILD" "llama-$BUILD-bin-ubuntu-x64.tar.gz"
if [ "$GPU" = 1 ]; then
    echo "== llama.cpp $BUILD (CUDA $CUDA)"
    install_llama "tools/llama-$BUILD-cuda" "llama-$BUILD-bin-ubuntu-cuda-$CUDA-x64.tar.gz" \
        "cudart-llama-$BUILD-bin-ubuntu-cuda-$CUDA-x64.tar.gz"
fi

echo "== models"
.venv/bin/python serving/download_models.py

echo
echo "Setup done. Run the app: scripts/run_app.sh"
