#!/usr/bin/env python3
"""Create a declared state-preserving Thermal input fork; execute no training epochs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from relation_anchored_train import build, engine_for
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.run_layout import RunLayout
from honf_runtime.unified_training import (
    _atomic_json, _atomic_torch_save, _write_training_history, _render_loss_curves,
)
from channelthermal.training.state_response_fork import make_fork_payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    recipe = json.loads(args.recipe.read_text())
    source_path = Path(recipe['checkpoint_fork']['source_checkpoint']).resolve()
    output = args.output.resolve()
    if output == source_path.parent.parent or (output.exists() and any(output.iterdir())):
        raise ValueError('An input fork requires a distinct empty output directory.')
    before = source_path.stat()
    source = load_trusted_checkpoint(source_path, map_location='cpu')
    scales = source['experiment_identity']['provider_identity']['response_scales']
    model, provider = build(recipe, device='cpu', response_scales=scales)
    engine = engine_for(recipe, device='cpu')
    payload, receipt = make_fork_payload(source, recipe, model, provider, engine, source_path)
    after = source_path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError('The source checkpoint changed during initialization.')
    layout = RunLayout(output); layout.ensure()
    _atomic_json(output / 'artifact_layout.json', {'artifact_layout_version': 1, 'run_dir': str(output)})
    _atomic_json(layout.write_path('experiment_identity.json'), payload['experiment_identity'])
    _atomic_json(layout.write_path('input_fork.json', category='configs'), receipt)
    _write_training_history(layout.write_path('history.json'), payload['history'], compact=False)
    _atomic_torch_save(layout.write_path('latest_model.pt'), payload)
    _render_loss_curves(payload['history'], layout.write_path('loss_curves.png').parent,
                        engine.selection.field_metric,
                        provider_identity=payload['experiment_identity']['provider_identity'])
    print(json.dumps({'output': str(output), **receipt}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
