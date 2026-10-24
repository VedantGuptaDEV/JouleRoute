#!/usr/bin/env bash
# Start the whole app with one command: the three llama-servers, the gateway, then the chat.
# Anything already running on its port is reused; whatever this script started is stopped when
# the chat exits. Arguments go to cli/joule.py, e.g. scripts/run_app.sh -p "what is 17 * 23".
#   LLAMA_SERVER  llama-server binary (default: the CUDA build from setup_gpu.sh if there is a GPU,
#                 else config.yaml's CPU build)
# Logs: data/logs/launch.log, data/logs/gateway.log, data/logs/llama-<tier>.log
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
CUDA_SERVER=tools/llama-b11840-cuda/llama-server
if [ -z "${LLAMA_SERVER:-}" ] && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
        && [ -x "$CUDA_SERVER" ]; then
    LLAMA_SERVER=$CUDA_SERVER
fi
read -r -a TIER_PORTS <<< "$($PY -c "from shared.config import load_config; c = load_config(); print(*(m['port'] for m in c['models'].values()))")"
GATEWAY_PORT=$($PY -c "from shared.config import load_config; print(load_config()['gateway']['port'])")
mkdir -p data/logs

healthy() { curl -sf -o /dev/null "http://127.0.0.1:$1/health"; }
pids=()
cleanup() {
    [ ${#pids[@]} -eq 0 ] && return
    echo "stopping the gateway and model servers"
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT

# wait_for <pid> <name> <log> <port>...: until every port is healthy, or fail if the process exits
wait_for() {
    local pid=$1 name=$2 log=$3; shift 3
    for port in "$@"; do
        until healthy "$port"; do
            kill -0 "$pid" 2>/dev/null || { echo "$name exited; last lines of $log:"; tail -n 20 "$log"; exit 1; }
            sleep 1
        done
    done
}

if healthy "${TIER_PORTS[0]}" && healthy "${TIER_PORTS[1]}" && healthy "${TIER_PORTS[2]}"; then
    echo "model servers: already running, reusing them"
else
    echo "starting the model servers (first start loads ~4 GiB of models; log: data/logs/launch.log)"
    $PY serving/launch.py ${LLAMA_SERVER:+--llama-server "$LLAMA_SERVER"} > data/logs/launch.log 2>&1 &
    pids+=($!)
    wait_for $! "serving/launch.py" data/logs/launch.log "${TIER_PORTS[@]}"
fi

if healthy "$GATEWAY_PORT"; then
    echo "gateway: already running on :$GATEWAY_PORT, reusing it"
else
    echo "starting the gateway (log: data/logs/gateway.log)"
    $PY gateway/app.py > data/logs/gateway.log 2>&1 &
    pids+=($!)
    wait_for $! "gateway/app.py" data/logs/gateway.log "$GATEWAY_PORT"
fi

$PY cli/joule.py --url "http://127.0.0.1:$GATEWAY_PORT" "$@"
