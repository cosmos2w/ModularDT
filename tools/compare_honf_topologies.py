#!/usr/bin/env python3
"""Compare two unordered HONF topology-signature NPZ artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from honf_forward_core.evaluation.topology_signature import (
    compare_topology_signatures,
    load_topology_signature,
    summarize_topology_signature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--relation-weight", type=float, default=0.25)
    parser.add_argument("--unmatched-cost", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.relation_weight < 0 or args.unmatched_cost < 0:
        parser.error("relation-weight and unmatched-cost must be nonnegative")
    return args


def main() -> int:
    args = parse_args()
    first = load_topology_signature(args.first)
    second = load_topology_signature(args.second)
    result = {
        "first": {"path": str(args.first), "summary": summarize_topology_signature(first)},
        "second": {"path": str(args.second), "summary": summarize_topology_signature(second)},
        "comparison": compare_topology_signatures(
            first,
            second,
            relation_weight=float(args.relation_weight),
            unmatched_cost=float(args.unmatched_cost),
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
