"""Small fresh-process recovery fixture with actual conditioned response losses."""
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

import honf_runtime.unified_training as runtime
from honf_runtime.local_response_contrasts import (CommonObservations, build_local_contrast,
                                                  build_uniform_contrast, neighborhood_field_losses)
from honf_runtime.native_prefix_branch import initialize_native_prefix_child
from honf_runtime.run_layout import RunLayout
from honf_runtime.unified_training import LossTerm, SamplingKey, TrainingEngine
from tests.csv_resume_handoff_worker import OrderedLossProvider
from tests.field_response_prefix_worker import PrefixProvider
from tests.test_unified_training import _config


class LocalProvider(OrderedLossProvider):
    def __init__(self, recipe):
        super().__init__(recipe)
        self.variant = recipe['thermal_local_response']['variant']
        local_config = recipe['thermal_local_response']
        self.response_weight = local_config.get('objective', {}).get('response_weight', 1.)
        if local_config.get('state_priority', False):
            self.state_priority_response_terms = ('local_response/response',)
        self.calibration = None
        self.episodes = []
        self.auxiliary_loss_names = ('local_response/absolute', 'local_response/response')

    def validate_local_calibration(self, record, *, source_checkpoint):
        assert record['source_epoch'] == 10 and Path(source_checkpoint).name == 'epoch_0010_model.pt'
        return record

    def initialize_local_response_state(self, native_state, calibration):
        return {'native': native_state['native'], 'local_response': {'calibration': calibration, 'episodes': []}}

    def training_state_dict(self):
        return {'native': {'completed': self.completed, 'auxiliary_calls': 0},
                'local_response': {'calibration': self.calibration, 'episodes': self.episodes}}

    def load_training_state_dict(self, state):
        self.completed = state['native']['completed']
        self.calibration = state['local_response']['calibration']
        self.episodes = state['local_response']['episodes']

    def training_history_columns(self):
        return super().training_history_columns() + tuple('train_losses.' + n for n in self.auxiliary_loss_names)

    def loss_metadata(self):
        return {**super().loss_metadata(), **{n: {'unit': 'scaled', 'meaning': n} for n in self.auxiliary_loss_names}}

    def auxiliary_update_terms(self, model, *, epoch, update_index, update_count):
        if update_index:
            return
        draw = random.randrange(1000)
        chart = torch.tensor([[0., 0.], [1., 0.], [0., 0.2]], dtype=torch.float64)
        contrast = (build_uniform_contrast(3) if self.variant == 'uniform_response'
                    else build_local_contrast(chart))
        self.episodes.append({'epoch': epoch, 'center': draw, 'operator': contrast.operator.tolist()})
        scene = torch.tensor([[0.], [1.], [2.]]) + torch.rand(()) * 0.01 + float(np.random.random()) * 0.01
        prediction = model(scene).reshape(1, 3, 1, 1)
        observation = CommonObservations(torch.tensor([[[0.]], [[0.4]], [[1.3]]]),
                                         torch.ones((1, 1), dtype=torch.bool),
                                         torch.ones((1, 1)))
        losses = neighborhood_field_losses(prediction, observation, contrast, [1.], [1.])
        yield ({'local_response/absolute': LossTerm(losses.absolute, 1, 0.25 * update_count),
                'local_response/response': LossTerm(losses.response, 1, 0.1 * self.response_weight * update_count)}, {})


def main():
    action, recipe_path, output, argument = sys.argv[1:]
    recipe = json.loads(Path(recipe_path).read_text())
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    model = nn.Linear(1, 1)
    provider = PrefixProvider(recipe) if action == 'parent' else LocalProvider(recipe)
    settings = replace(_config(), total_epochs=12, monitor_every=1, checkpoint_epochs=(10, 12),
                       training_mode='joint', progress_enabled=False, sampling_version=SamplingKey.CASE_EPOCH_VERSION)
    engine = TrainingEngine(settings, device='cpu')
    runtime._render_loss_curves = lambda *args, **kwargs: None
    if action == 'branch':
        initialize_native_prefix_child(argument, output, recipe, model, provider, engine,
            calibration={'source_epoch': 10, 'coefficients': {'absolute': .25, 'response': .1}}, render_curves=False)
    else:
        if action == 'parent':
            runtime._atomic_json(Path(output) / 'configs/resolved_config.json', recipe)
        engine.fit(model, provider, output, identity={'workflow': 'prefix-toy', 'recipe': recipe},
                   arm='adaptive_detail', stop_after=int(argument),
                   resume_checkpoint=None if action == 'parent' else RunLayout(Path(output)).read_path('latest_model.pt'))


if __name__ == '__main__':
    main()
