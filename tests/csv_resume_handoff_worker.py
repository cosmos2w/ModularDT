"""CPU subprocess worker for the shared CSV resume contract regression."""
from __future__ import annotations

import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

import honf_runtime.unified_training as runtime
from honf_runtime.unified_training import LossTerm, TrainingEngine
from tests.test_unified_training import _ToyProvider, _config


class OrderedLossProvider(_ToyProvider):
    training_output_policy = 'csv_v3'
    training_stream_hash_history = False

    def __init__(self, config):
        super().__init__()
        self.terms = config['terms']
        self.completed = 0

    def completed_training_epoch(self, model, engine, epoch, output, row):
        self.completed += 1

    def training_state_dict(self):
        return {'completed': self.completed}

    def load_training_state_dict(self, state):
        self.completed = state['completed']

    def training_history_columns(self):
        return tuple('train_losses.' + name for name in self.terms) + ('work_counts.query_rows',)

    def loss_metadata(self):
        return {name: {'unit': definition['unit'], 'meaning': definition['meaning']}
                for name, definition in self.terms.items()}

    def training_column_definitions(self):
        return {'work_counts.query_rows': {'dtype': 'float', 'unit': 'rows',
                                            'meaning': 'Native query rows processed in the epoch'}}

    def loss_denominators(self, batches, phase, arm):
        count = sum(batch.targets.numel() for batch in batches)
        return {name: count for name in self.terms}

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        # Three distinct, nonzero terms make ordering and weighted aggregation real.
        residual = predictions - targets
        values = {'zeta': residual.square().sum(),
                  'middle': (residual - 0.25).square().sum(),
                  'alpha': (residual + 0.5).square().sum()}
        return {name: LossTerm(values[name], targets.numel(), weight=entry['weight'])
                for name, entry in self.terms.items()}

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch, arm):
        return self.loss_terms(predictions, targets, 'joint', auxiliary_state)

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch, temperature):
        # Exercise all three restored RNG families during actual training.
        jitter = (random.random() + float(np.random.random()) + float(torch.rand(()))) * 1e-4
        return model(scene) + jitter, None


def main():
    config_path, output, stop, resume = sys.argv[1:]
    config = json.loads(Path(config_path).read_text())
    if resume == '-':
        runtime._atomic_json(Path(output) / 'configs/resolved_config.json', config)
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    model = nn.Linear(1, 1)
    provider = OrderedLossProvider(config)
    settings = replace(_config(), total_epochs=4, monitor_every=1, checkpoint_epochs=(2, 4),
                       training_mode='joint', progress_enabled=False)
    # Plot rendering is unrelated to the CSV/checkpoint contract being tested.
    runtime._render_loss_curves = lambda *args, **kwargs: None
    TrainingEngine(settings, device='cpu').fit(
        model, provider, Path(output), identity={'run': 'handoff', 'recipe': config},
        arm='adaptive_detail', stop_after=int(stop),
        resume_checkpoint=(Path(resume) if resume != '-' else None))


if __name__ == '__main__':
    main()
