"""The fit loop passes both source-read and epoch-normalized auxiliary inputs."""
from dataclasses import replace

import torch
from test_unified_training import _config, _ToyProvider
from torch import nn

from honf_runtime.unified_training import LossTerm, TrainingEngine


class SourceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[.5]]))
        self.bias = nn.Parameter(torch.tensor([.1]))
        self.core = nn.Module()
        self.core.direct_reader = nn.Module()
        self.core.direct_reader.source_read = nn.Module()
        self.core.direct_reader.source_read.scorer = nn.Parameter(torch.tensor(.25))

    def forward(self, values):
        return values @ self.weight.T + self.bias


class Manager:
    last_primary_fine_rows = 1
    auxiliary_coefficient = 1.
    initial_teacher_rows = initial_teacher_present_rows = 0
    coadaptation_teacher_rows = coadaptation_teacher_present_rows = 0
    def __init__(self):
        self.calls = []
        self.native_norms = []
        self.last_loss_components = {'scorer': .1, 'completion': 0.}
        self.last_refresh = {'refreshed': False}

    def before_update(self, model, provider, **kwargs):
        self.calls.append(kwargs)

    def auxiliary_loss(self, model):
        return model.core.direct_reader.source_read.scorer.square()

    def record_gradient_balance(self, *, native_norm, weighted_auxiliary_norm):
        self.native_norms.append(native_norm)
        self.last_gradient_balance = {
            'native_all_parameter_gradient_l2_norm': native_norm,
            'weighted_auxiliary_new_parameter_gradient_l2_norm': weighted_auxiliary_norm,
            'weighted_auxiliary_to_native_gradient_share': weighted_auxiliary_norm / native_norm if native_norm else None}

    def record_primary_fine_rows(self, value):
        self.last_primary_fine_rows = value

    def state_dict(self):
        return {'last_primary_fine_rows': self.last_primary_fine_rows}


class Provider(_ToyProvider):
    auxiliary_loss_names = ('response_test',)

    def __init__(self):
        super().__init__()
        self.source_read_auxiliary_manager = Manager()
        self.calls = []
        self.native_norms = []

    def optimizer_groups(self, model, arm, stage):
        group = super().optimizer_groups(model, arm, stage)[0]
        return (replace(group, parameter_names=group.parameter_names + ('core.direct_reader.source_read.scorer',)),)

    def work_counts(self, batch, predictions, auxiliary_state):
        return {'core_decode_primary_source_read_fine_executed_rows': len(batch.case_keys)}

    def needs_native_gradient_norm(self, *, epoch, update_index):
        return epoch == 1 and update_index == 0

    def record_native_gradient_norm(self, norm, *, case_keys):
        self.native_norms.append(norm)
        self.native_case_keys = case_keys

    def auxiliary_update_terms(self, model, *, epoch, update_index, update_count):
        self.calls.append((epoch, update_index, update_count))
        yield {'response_test': LossTerm(model.weight.square().sum(), 1, .1)}, {'response_actions': 1}


def test_fit_combines_source_read_and_epoch_auxiliary(tmp_path):
    torch.manual_seed(4)
    model, provider = SourceModel(), Provider()
    engine = TrainingEngine(replace(_config(), training_mode='joint', progress_enabled=False,
                                    curve_every=None), device='cpu')
    engine.fit(model, provider, tmp_path, identity={'workflow': 'combined_auxiliary_test'},
               arm='toy', stop_after=1)
    assert provider.calls == [(1, 0, 2), (1, 1, 2)]
    assert [call['ordinary_update_index'] for call in provider.source_read_auxiliary_manager.calls] == [1, 2]
    assert [call['primary_fine_rows'] for call in provider.source_read_auxiliary_manager.calls] == [1, 5]
    assert provider.native_norms == [provider.source_read_auxiliary_manager.native_norms[0]]
    assert len(provider.native_case_keys) == 4
    assert model.core.direct_reader.source_read.scorer.item() != .25
