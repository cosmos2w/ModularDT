#!/usr/bin/env bash

# whenever want to work on or show off the demo, 
# only need to open one single terminal and run these two commands:
# conda activate ModularDT
# ./start_demo.sh

set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "ModularDT" ]]; then
    echo "Warning: Conda environment 'ModularDT' does not appear to be active."
    echo "For the backend to work, you may need to run 'conda activate ModularDT' first."
    echo ""
fi

cleanup() {
    echo ""
    echo "Shutting down the ChannelThermal demo..."
    local job_pids
    job_pids="$(jobs -p)"
    if [[ -n "$job_pids" ]]; then
        kill $job_pids 2>/dev/null || true
        wait $job_pids 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT TERM

require_free_port() {
    local port="$1"
    local label="$2"
    if python - "$port" >/dev/null 2>&1 <<'PY'
import errno
import socket
import sys

port = int(sys.argv[1])
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", port))
except PermissionError:
    raise SystemExit(0)
except OSError as exc:
    if exc.errno == errno.EPERM:
        raise SystemExit(0)
    raise SystemExit(1)
finally:
    if "sock" in locals():
        sock.close()
PY
    then
        return
    fi
    echo "Port $port is already in use for $label."
    echo "Stop the existing process or change the $label port before running this script."
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/web_demo"

require_free_port 8001 "backend"
require_free_port 5174 "frontend"

./run_backend.sh &
./run_frontend.sh &

echo "ChannelThermal demo is starting:"
echo "  backend  http://127.0.0.1:8001"
echo "  frontend http://127.0.0.1:5174"
echo "Press Ctrl+C to stop both servers."

echo "Waiting 4 seconds for servers to initialize..."
sleep 4

echo "Opening ChannelThermal dashboard..."
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open http://127.0.0.1:5174 >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
    open http://127.0.0.1:5174 >/dev/null 2>&1 || true
elif command -v start >/dev/null 2>&1; then
    start http://127.0.0.1:5174 >/dev/null 2>&1 || true
else
    echo "Please manually open: http://127.0.0.1:5174"
fi

wait
