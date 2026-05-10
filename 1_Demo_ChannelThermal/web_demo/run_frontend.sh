#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/frontend"

if [ ! -d node_modules ]; then
  SHARED_NODE_MODULES="../../../0_Demo_MultiCylinder/web_demo/frontend/node_modules"
  if [ -d "$SHARED_NODE_MODULES" ]; then
    echo "node_modules not found; reusing local dependency cache from 0_Demo_MultiCylinder"
    ln -s "$SHARED_NODE_MODULES" node_modules
  else
    echo "node_modules not found; running npm install"
    npm install
  fi
fi

echo "Starting ChannelThermal web demo frontend at http://127.0.0.1:5174"
if [[ "${CONDA_DEFAULT_ENV:-}" != "ModularDT" ]] && command -v conda >/dev/null 2>&1; then
  conda run --no-capture-output -n ModularDT npm run dev
else
  npm run dev
fi
