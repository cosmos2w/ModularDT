#!/usr/bin/env python3
"""Relation-guided capability entrypoint using the maintained HONF runner."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src"):
    sys.path.insert(0, str(directory))

from honf_runtime.guided_member_routing_training import main

if __name__ == "__main__":
    raise SystemExit(main())
