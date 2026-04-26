#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/frontend"

if [ ! -d node_modules ]; then
  echo "node_modules not found; running npm install"
  npm install
fi

echo "Starting ModularDT web demo frontend at http://127.0.0.1:5173"
npm run dev -- --host 127.0.0.1 --port 5173
