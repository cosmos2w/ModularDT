#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/backend"

echo "Starting ChannelThermal web demo backend at http://127.0.0.1:8001"
echo "Use your preferred Python environment. If needed: pip install -r requirements.txt"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-channelthermal-web}"
mkdir -p "$MPLCONFIGDIR"
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8001
