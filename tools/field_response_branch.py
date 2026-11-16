#!/usr/bin/env python3
"""Initialize a new paired-field lineage from its literal native epoch-ten state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .relation_anchored_train import build, engine_for
except ImportError:
    from relation_anchored_train import build, engine_for
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.native_prefix_branch import initialize_native_prefix_child


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    recipe = json.loads(args.recipe.read_text())
    calibration = json.loads(args.calibration.read_text())
    source = load_trusted_checkpoint(args.source, map_location='cpu')
    model, provider = build(recipe, device='cpu', resume_payload=source)
    engine = engine_for(recipe, device='cpu')
    # Verify the declared storage binding and allowance before initializing.
    # This records no training and changes no source-run artifact.
    from honf_runtime.state_response_storage import prepare_state_response_storage
    prepare_state_response_storage(recipe, args.output, model, completed_epoch=0,
        stop_after=recipe['training']['local_horizon'], checkpoint_epochs=engine.config.checkpoint_epochs)
    receipt = initialize_native_prefix_child(args.source, args.output, recipe, model,
                                             provider, engine, calibration=calibration)
    print(json.dumps({'output': str(args.output.resolve()), **receipt}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
