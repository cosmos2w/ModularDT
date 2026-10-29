#!/usr/bin/env python3
"""Compatibility entry point for the inspection-only preprocessing pass.

This command is intentionally an alias for :mod:`inspect_windfarm`: it writes
validated metadata, case tables, and group-safe split indices, but no model,
training artifact, or copied dataset.
"""

from __future__ import annotations

from inspect_windfarm import main

if __name__ == "__main__":
    raise SystemExit(main())
