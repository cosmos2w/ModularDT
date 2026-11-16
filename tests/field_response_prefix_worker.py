"""Fresh-process CPU worker for native-prefix child recording and recovery."""
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

import honf_runtime.unified_training as runtime
from honf_runtime.native_prefix_branch import initialize_native_prefix_child
from honf_runtime.run_layout import RunLayout
from honf_runtime.unified_training import LossTerm, SamplingKey, TrainingEngine
from tests.csv_resume_handoff_worker import OrderedLossProvider
from tests.test_unified_training import _config


class PrefixProvider(OrderedLossProvider):
    def __init__(self, recipe):
        super().__init__(recipe)
        self.variant = recipe['thermal_field_response']['variant']
        self.calibration = None
        self.auxiliary_calls = 0
        names = {'field_only': (), 'field_anchors': ('absolute',),
                 'field_response': ('absolute', 'difference')}[self.variant]
        self.auxiliary_loss_names = tuple('field_response/' + name for name in names)
        # A native-only provider must not expose an empty auxiliary hook.
        if not names:
            self.auxiliary_update_terms = None

    def training_state_dict(self):
        return {'native': {'completed': self.completed, 'auxiliary_calls': self.auxiliary_calls},
                'field_response': {'calibration': self.calibration}}

    def load_training_state_dict(self, state):
        self.completed = state['native']['completed']
        self.auxiliary_calls = state['native']['auxiliary_calls']
        self.calibration = state['field_response']['calibration']

    def training_history_columns(self):
        return super().training_history_columns() + tuple('train_losses.' + name for name in self.auxiliary_loss_names)

    def loss_metadata(self):
        return {**super().loss_metadata(), **{name: {'unit': 'scaled', 'meaning': name}
                                               for name in self.auxiliary_loss_names}}

    def auxiliary_update_terms(self, model, *, epoch, update_index, update_count):
        if update_index == 0:
            self.auxiliary_calls += 1
            jitter = (random.random() + float(np.random.random()) + float(torch.rand(()))) * 1e-4
            yield ({name: LossTerm((model.weight + jitter + index).square().sum(), 1,
                                   self.calibration['coefficients'][name] * update_count)
                    for index, name in enumerate(self.auxiliary_loss_names)}, {'pair_queries': 1})


def main():
    action, recipe_path, output, argument = sys.argv[1:]
    recipe = json.loads(Path(recipe_path).read_text())
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    model, provider = nn.Linear(1, 1), PrefixProvider(recipe)
    settings = replace(_config(), total_epochs=12, monitor_every=1, checkpoint_epochs=(10, 12),
                       training_mode='joint', progress_enabled=False, sampling_version=SamplingKey.CASE_EPOCH_VERSION)
    engine = TrainingEngine(settings, device='cpu')
    runtime._render_loss_curves = lambda *args, **kwargs: None
    if action == 'branch':
        calibration = {'coefficients': {'field_response/absolute': .25, 'field_response/difference': .10},
                       'source_epoch': 10}
        initialize_native_prefix_child(argument, output, recipe, model, provider, engine,
                                       calibration=calibration, render_curves=False)
    else:
        if action == 'parent':
            runtime._atomic_json(Path(output) / 'configs/resolved_config.json', recipe)
        engine.fit(model, provider, output, identity={'workflow': 'prefix-toy', 'recipe': recipe},
                   arm='adaptive_detail', stop_after=int(argument),
                   resume_checkpoint=None if action == 'parent' else RunLayout(Path(output)).read_path('latest_model.pt'))


if __name__ == '__main__':
    main()
