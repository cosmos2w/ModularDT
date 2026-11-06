#!/usr/bin/env python3
"""Derive R16-5 P64 fork recipes or run their maintained train/resume CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / 'src', PROJECT / 'Case_ThermalChannel/src', PROJECT / 'Case_WindFarm/src'):
    sys.path.insert(0, str(directory))

from channelthermal.training.r16_5_warm_fork import (
    H0_IDENTITY,
    P64_IDENTITY,
    _read_identity,
    make_research_metadata,
    make_warm_fork_recipe,
)
from relation_anchored_train import main as training_main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('inspect', 'fit', 'evaluate'):
        return training_main(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('metadata', help='Compute the one shared original-FIT research receipt')
    recipe_parser = sub.add_parser('recipe', help='Derive one fork using the saved shared receipt')
    recipe_parser.add_argument('--research-metadata', required=True)
    recipe_parser.add_argument('--arm', choices=('PDelta', 'PDeltaD'), required=True)
    recipe_parser.add_argument('--name', required=True)
    recipe_parser.add_argument('--output-root', required=True)
    recipe_parser.add_argument('--membership', choices=('fixed25_train150_dev22', 'full_train600'), required=True)
    recipe_parser.add_argument('--local-horizon', type=int, required=True)
    recipe_parser.add_argument('--first-segment-epochs', type=int, required=True)
    recipe_parser.add_argument('--wall-clock-budget-seconds', type=int, default=86400)
    recipe_parser.add_argument('--clean-save-reserve-seconds', type=int, default=600)
    args = parser.parse_args(argv)
    p64, h0 = _read_identity(P64_IDENTITY), _read_identity(H0_IDENTITY)
    if args.command == 'metadata':
        print(json.dumps(make_research_metadata(p64, h0), indent=2, sort_keys=True))
        return 0
    recipe = make_warm_fork_recipe(p64, h0, _read_identity(args.research_metadata), arm=args.arm,
        name=args.name, output_root=args.output_root, membership=args.membership,
        local_horizon=args.local_horizon, first_segment_epochs=args.first_segment_epochs,
        wall_clock_budget_seconds=args.wall_clock_budget_seconds,
        clean_save_reserve_seconds=args.clean_save_reserve_seconds)
    print(json.dumps(recipe, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
