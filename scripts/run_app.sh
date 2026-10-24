#!/usr/bin/env bash
# Start the whole app with one command: the three llama-servers, the gateway, then the chat.
# See scripts/run_app.py (shared with scripts\run_app.cmd on Windows). Arguments go to
# cli/joule.py, e.g. scripts/run_app.sh -p "what is 17 * 23".
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || { echo ".venv not found: run scripts/setup.sh first"; exit 1; }
exec .venv/bin/python scripts/run_app.py "$@"
