#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "ModularDT" ]]; then
    echo "Warning: Conda environment 'ModularDT' does not appear to be active."
    echo "For the backend to work, you may need to run 'conda activate ModularDT' first."
    echo ""
fi

cleanup() {
    echo ""
    echo "Shutting down the ChannelThermal demo..."
    kill $(jobs -p) 2>/dev/null || true
    wait $(jobs -p) 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/web_demo"

./run_backend.sh &
./run_frontend.sh &

echo "ChannelThermal demo is starting:"
echo "  backend  http://127.0.0.1:8001"
echo "  frontend http://127.0.0.1:5174"
echo "Press Ctrl+C to stop both servers."
wait
