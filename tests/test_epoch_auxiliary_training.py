"""Epoch-normalized auxiliary work shares the maintained native optimizer."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from honf_runtime.unified_training import LossTerm, TrainingEngine, _mean_mapping
from test_unified_training import _ToyProvider, _config


class AuxiliaryProvider(_ToyProvider):
    auxiliary_loss_names = ('response_test',)

    def __init__(self):
        super().__init__()
        self.calls = []

    def auxiliary_update_terms(self, model, *, epoch, update_index, update_count):
        self.calls.append((epoch, update_index, update_count))
        # One bounded family batch, with the epoch weight independent of the
        # number of native macro updates. Other updates contribute exact zero.
        if update_index == 0:
            yield {'response_test': LossTerm(model.weight.square().sum(), 1,
                                            0.1 * update_count)}, {'auxiliary_actions': 1}


@pytest.mark.parametrize('effective', [2, 4, 8])
def test_auxiliary_epoch_mean_and_gradient(effective):
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(2)
        model.bias.fill_(0.5)
    provider = AuxiliaryProvider()
    engine = TrainingEngine(replace(_config(), effective_cases=effective,
                                    training_mode='joint'), device='cpu')
    # Zero lr isolates normalization and gradient accumulation from changing
    # parameter values across the native update groups.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0, weight_decay=0)
    count = len(engine.epoch_update_slices(5))
    rows, gradients = [], []
    for index, group_slice in enumerate(engine.epoch_update_slices(5)):
        keys = list(range(5))[group_slice]
        batches = engine.prepare_update_microbatches(provider, arm='toy', phase='joint',
                   epoch=1, update_index=index, update_cases=keys, sampling_dataset_id='toy')
        losses, work, _, _ = engine._run_update(model, provider, optimizer, batches,
                   phase='joint', arm='toy', epoch=1, update_index=index,
                   epoch_update_count=count)
        rows.append(losses)
        gradients.append(float(model.weight.grad) + sum(keys) / len(keys))
        assert work.get('auxiliary_actions', 0) == (1 if index == 0 else 0)
    assert _mean_mapping(rows)['response_test'] == pytest.approx(0.4)
    assert sum(gradients) / count == pytest.approx(0.4)
    assert provider.calls == [(1, index, count) for index in range(count)]


def test_auxiliary_uses_ordinary_save_and_resume(tmp_path):
    def run(path, stop, resume=None):
        torch.manual_seed(13)
        model = nn.Linear(1, 1)
        provider = AuxiliaryProvider()
        engine = TrainingEngine(replace(_config(), training_mode='joint',
                                progress_enabled=False), device='cpu')
        engine.fit(model, provider, path, identity={'workflow': 'auxiliary_test'},
                   arm='toy', stop_after=stop, resume_checkpoint=resume)
        return model
    full = run(tmp_path / 'full', 3)
    run(tmp_path / 'resumed', 2)
    resumed = run(tmp_path / 'resumed', 3, 'latest')
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
