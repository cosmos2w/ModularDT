#!/usr/bin/env python
"""Evaluate an R13-7 U or R13-8 A/L native checkpoint without fitting."""
from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run maintained full/restricted native evaluation for U/A/L and bounded "
            "E-organization, same-state intervention, and live-boundary diagnostics for A/L."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--representative-case", action="append", dest="representative_cases",
        help=(
            "pin one input-selected validation detail case ID; repeat to reuse the same "
            "panel across checkpoints with different validation populations"
        ),
    )
    args = parser.parse_args()

    from honf_runtime.r13_8_anchored_organization_evaluation import evaluate

    result = evaluate(
        args.checkpoint, args.device, args.output,
        representative_case_ids=args.representative_cases,
    )
    print(json.dumps(result, sort_keys=True, default=str, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
