#!/usr/bin/env python3
"""Compatibility wrapper; maintained source moved to tools/diagnostics."""
from pathlib import Path
import runpy

globals().update(runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "diagnostics" / Path(__file__).name)))
if __name__ == "__main__":
    raise SystemExit(main())
