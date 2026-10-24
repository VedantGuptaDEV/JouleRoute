#!/usr/bin/env bash
# setup.sh on a machine that must have an NVIDIA GPU (see TRAINING.md): stops early if the driver
# is missing instead of falling back to a CPU-only setup.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
    || { echo "nvidia-smi not found or sees no GPU: install the NVIDIA driver first (TRAINING.md step 1)"; exit 1; }
exec scripts/setup.sh
