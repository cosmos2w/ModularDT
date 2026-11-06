#!/usr/bin/env bash
# Run from the active VS Code Remote integrated terminal.
set -euo pipefail

demo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
port="${HONF_DEMO_PORT:-8765}"
python_bin="${HONF_DEMO_PYTHON:-/home/wanglz/miniconda3/envs/ModularDT/bin/python}"
open_browser=true

usage() {
  cat <<'HELP'
Usage: ./open.sh [--port PORT] [--no-open]

Start or reuse Wind Lab, then request opening in your local browser through
the active VS Code Remote terminal. Default port: 8765.

  --port PORT  Override HONF_DEMO_PORT.
  --no-open    Start/reuse the server and print the URL without opening it.
  --help       Show this help.
HELP
}

while (( $# )); do
  case "$1" in
    --port)
      if (( $# < 2 )); then usage >&2; exit 2; fi
      port="$2"; shift 2 ;;
    --no-open) open_browser=false; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if [[ ! "$port" =~ ^[0-9]{1,5}$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
  echo 'Use a port between 1 and 65535.' >&2
  exit 2
fi
port="$((10#$port))"
url="http://127.0.0.1:$port"
session="honf-wind-demo-$port"
if [[ ! -x "$python_bin" ]]; then
  echo 'Set HONF_DEMO_PYTHON to the existing project Python environment.' >&2
  exit 1
fi

healthy() {
  "$python_bin" - "$url/api/meta" <<'PY' >/dev/null 2>&1
import json, sys
from urllib.request import urlopen
try:
    with urlopen(sys.argv[1], timeout=2) as response:
        checkpoint = json.load(response)["checkpoint"]
    if not (checkpoint["run"].startswith("wind_") and isinstance(checkpoint["epoch"], int)):
        sys.exit(1)
except Exception:
    sys.exit(1)
PY
}

if healthy; then
  echo 'Reusing the running Wind Lab server.'
else
  if ! command -v tmux >/dev/null; then
    echo "tmux is required for background startup. Run $demo_dir/run.sh in another terminal." >&2
    exit 1
  fi
  if ! tmux has-session -t "=$session" 2>/dev/null; then
    if "$python_bin" - "$port" <<'PY'
import socket, sys
try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
        pass
except OSError:
    sys.exit(1)
PY
    then
      echo "Port $port is occupied by a service that is not a ready Wind Lab. Choose --port PORT or inspect that service." >&2
      exit 1
    fi
    tmux new-session -d -s "$session" -c "$demo_dir" env \
      "HONF_DEMO_PORT=$port" \
      "HONF_DEMO_PYTHON=$python_bin" \
      "HONF_DEMO_NODE=${HONF_DEMO_NODE:-/home/wanglz/.nvm/versions/node/v26.12.1/bin/node}" \
      "HONF_DEMO_OUTPUT_ROOT=${HONF_DEMO_OUTPUT_ROOT:-/data2/wanglz/ModularDT/windfarm_sponsor_demo_20261105}" \
      ./run.sh
  fi
  echo "Starting Wind Lab in tmux session $session..."
  deadline=$((SECONDS + 60))
  until healthy; do
    if ! tmux has-session -t "=$session" 2>/dev/null; then
      echo "Startup exited. Run $demo_dir/run.sh in the foreground to see the error." >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Startup is still pending. Inspect it with: tmux attach -t $session" >&2
      exit 1
    fi
    sleep 0.5
  done
fi

printf '\nWind Lab: %s\n' "$url"
printf 'VS Code Ports: forward port %s, then use Open in Browser on its displayed Forwarded Address.\n' "$port"
if "$open_browser"; then
  if [[ -n "${VSCODE_IPC_HOOK_CLI:-}" && -S "$VSCODE_IPC_HOOK_CLI" ]]; then
    if [[ "${BROWSER:-}" == */bin/helpers/browser.sh && -x "${BROWSER:-}" ]]; then
      if "$BROWSER" "$url"; then
        echo 'Opening requested through VS Code.'
        exit 0
      fi
    fi
    remote_code="$(command -v code || true)"
    if [[ "$remote_code" == */bin/remote-cli/code ]]; then
      if "$remote_code" --openExternal "$url"; then
        echo 'Opening requested through VS Code.'
        exit 0
      fi
    fi
  fi
  echo 'For automatic browser opening, run this command in the active VS Code Remote integrated terminal.'
fi
