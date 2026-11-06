#!/usr/bin/env bash
set -euo pipefail

demo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_root="${HONF_DEMO_OUTPUT_ROOT:-/data2/wanglz/ModularDT/windfarm_sponsor_demo_20261105}"
output_root="$(realpath -m -- "$output_root")"
python_bin="${HONF_DEMO_PYTHON:-/home/wanglz/miniconda3/envs/ModularDT/bin/python}"
node_bin="${HONF_DEMO_NODE:-/home/wanglz/.nvm/versions/node/v26.12.1/bin/node}"
port="${HONF_DEMO_PORT:-8765}"

if ! mountpoint -q /data2; then
  echo 'Mount /data2 before starting Wind Lab.' >&2
  exit 1
fi
case "$output_root" in /data2/wanglz/ModularDT/*) ;; *) echo 'Use an output directory beneath /data2/wanglz/ModularDT/.' >&2; exit 1;; esac
if [[ ! -x "$node_bin" || ! -x "$python_bin" ]]; then
  echo 'Set HONF_DEMO_NODE and HONF_DEMO_PYTHON to Node >= 20 and the project Python environment.' >&2
  exit 1
fi
if [[ "$($node_bin -p 'Number(process.versions.node.split(".")[0])')" -lt 20 ]]; then
  echo 'Wind Lab requires Node 20 or later.' >&2
  exit 1
fi
available_bytes="$(df -B1 --output=avail /data2 | tail -1 | tr -d ' ')"
if (( available_bytes < 500000000000 )); then
  echo 'Wind Lab keeps at least 500 GB free on /data2.' >&2
  exit 1
fi
mkdir -p "$output_root" "$output_root/dist" "$output_root/cache"
if [[ "$(stat -c %d /data2)" != "$(stat -c %d "$output_root")" ]]; then
  echo 'The output directory must reside on the mounted /data2 filesystem.' >&2
  exit 1
fi
export PATH="$(dirname "$node_bin"):$PATH"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MPLCONFIGDIR="$output_root/cache/matplotlib"
if [[ ! -f "$output_root/node_modules/vite/bin/vite.js" ]]; then
  cp "$demo_dir/package.json" "$demo_dir/package-lock.json" "$output_root/"
  npm ci --prefix "$output_root" --cache "$output_root/cache/npm" --no-audit --no-fund
fi
for item in node_modules dist; do
  if [[ -L "$demo_dir/$item" ]]; then
    if [[ "$(readlink -f "$demo_dir/$item")" != "$(readlink -f "$output_root/$item")" ]]; then
      echo "Existing $item symlink points elsewhere; use its output directory or update the binding." >&2
      exit 1
    fi
  elif [[ -e "$demo_dir/$item" ]]; then
    echo "Move the existing $item directory to /data2 before starting." >&2
    exit 1
  else
    ln -s "$output_root/$item" "$demo_dir/$item"
  fi
done
cd "$demo_dir"
"$node_bin" "$output_root/node_modules/vite/bin/vite.js" build
echo "Wind Lab: http://127.0.0.1:$port"
exec "$python_bin" -m uvicorn backend.app:app --host 127.0.0.1 --port "$port"
