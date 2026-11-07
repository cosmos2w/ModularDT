#!/usr/bin/env python3
"""Calibrate and train the R16-6 balanced source-initialized forks."""
import argparse
import json
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
for path in (PROJECT / 'src', PROJECT / 'Case_ThermalChannel/src', PROJECT / 'Case_WindFarm/src'):
    sys.path.insert(0, str(path))
from relation_anchored_train import build, engine_for, main as training_main
from channelthermal.training.r16_6_balanced_task import calibrate_source


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('inspect', 'fit', 'evaluate'):
        return training_main(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['calibrate'])
    parser.add_argument('--recipe', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    recipe = json.loads(args.recipe.read_text())
    model, provider = build(recipe, device=args.device)
    result = calibrate_source(model, provider, engine_for(recipe, device=args.device))
    content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.write_text(content)
    else:
        print(content, end='')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
