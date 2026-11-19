import math

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from honf_runtime.state_priority import combine_state_response
from honf_runtime.unified_training import LossTerm, SamplingKey, TrainingEngine
from tests.test_unified_training import _ToyProvider, _config


@pytest.mark.parametrize('response, expected, projected', [([1., 1.], [2., 1.], 0), ([-1., 1.], [1., 1.], 1), ([0., 1.], [1., 1.], 0), ([0., 0.], [1., 0.], 0)])
def test_projection(response, expected, projected):
    parameter = nn.Parameter(torch.tensor([1., 0.]))
    parameter.grad = torch.tensor([1., 0.])
    _, stats = combine_state_response([parameter], [torch.tensor(response)])
    assert parameter.grad.tolist() == pytest.approx(expected)
    assert stats['projected_count'] == projected
    assert stats.get('retained_fraction_sum', 0.) <= 1. + 1e-7


def test_tiny_state_suppresses_response_and_preserves_unused():
    first, unused = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
    first.grad = torch.tensor([1e-14])
    _, stats = combine_state_response([first, unused], [torch.ones(1), None])
    assert first.grad.item() == pytest.approx(1e-14)
    assert unused.grad is None
    assert stats['suppressed_count'] == 1


class PriorityProvider(_ToyProvider):
    auxiliary_loss_names = ('aux/state', 'aux/response')
    state_priority_response_terms = ('aux/response',)

    def auxiliary_update_terms(self, model, *, epoch, update_index, update_count):
        if update_index:
            return
        # Activation checkpoint forces recomputation for each backward traversal.
        y = checkpoint(lambda x: model(x), torch.ones(1, 1), use_reentrant=False)
        yield {'aux/state': LossTerm(y.square().sum(), 1),
               'aux/response': LossTerm(-y.sum(), 1)}, {}


def test_trainer_sparse_episode_microbatches_and_checkpoint():
    torch.manual_seed(3)
    model = nn.Linear(1, 1)
    provider = PriorityProvider()
    engine = TrainingEngine(_config(), device='cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
    batches = [provider.make_batch((x,), SamplingKey(0, 1, 0, 0, 'warmup', 'warmup')) for x in (1, 2)]
    losses, work, _, _ = engine._run_update(model, provider, optimizer, batches, phase='joint', arm='joint', epoch=100, update_index=0, epoch_update_count=2)
    assert work['state_priority_active_count'] == 1
    assert math.isfinite(work['state_priority_displacement_dot_sum'])
    assert 'aux/state' in losses and 'aux/response' in losses
    _, work, _, _ = engine._run_update(model, provider, optimizer, batches, phase='joint', arm='joint', epoch=100, update_index=1, epoch_update_count=2)
    assert work['state_priority_absent_count'] == 1
    assert 'state_priority_active_count' not in work


def test_optout_keeps_historical_sum():
    import copy
    model = nn.Linear(1, 1)
    other = copy.deepcopy(model)
    provider = PriorityProvider()
    provider.state_priority_response_terms = None
    engine = TrainingEngine(_config(), device='cpu')
    batches = [provider.make_batch((x,), SamplingKey(0, 1, 0, 0, 'warmup', 'warmup')) for x in (1, 2)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    expected_optimizer = torch.optim.AdamW(other.parameters(), lr=.001)
    engine._run_update(model, provider, optimizer, batches, phase='joint', arm='joint', epoch=1, update_index=0, epoch_update_count=2)
    expected_optimizer.zero_grad(set_to_none=True)
    for batch in batches:
        ((other(batch.scene_inputs) - batch.targets).square().sum() / 2).backward()
    y = other(torch.ones(1, 1))
    (y.square().sum() - y.sum()).backward()
    torch.nn.utils.clip_grad_norm_(other.parameters(), engine.config.gradient_clip)
    expected_optimizer.step()
    for actual, expected in zip(model.parameters(), other.parameters()):
        assert torch.equal(actual, expected)


from tests.test_unified_csv_recording import CsvToyProvider, _config_for_test


class PriorityCsvProvider(PriorityProvider, CsvToyProvider):
    training_output_policy = 'csv_v3'
    training_stream_hash_history = False

    def __init__(self):
        CsvToyProvider.__init__(self)
        self.selected = []

    def identity_payload(self):
        return {'dataset': 'priority-toy', 'target': 'fixed-control'}

    def training_history_columns(self):
        return (*CsvToyProvider.training_history_columns(self), 'train_losses.aux/state', 'train_losses.aux/response')

    def auxiliary_update_terms(self, model, *, epoch, update_index, update_count):
        if update_index == 0:
            self.selected.append(epoch)
        yield from super().auxiliary_update_terms(model, epoch=epoch, update_index=update_index, update_count=update_count)

    def training_state_dict(self):
        return {'completed': self.completed, 'selected': self.selected}

    def load_training_state_dict(self, state):
        self.completed, self.selected = state['completed'], list(state['selected'])


def _recovery_worker(output, config_path, stop, resume):
    import json
    import random
    from dataclasses import replace
    from pathlib import Path
    import numpy as np
    import honf_runtime.unified_training as runtime
    from honf_runtime.run_layout import RunLayout
    config = json.loads(Path(config_path).read_text())
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    runtime._render_loss_curves = lambda *args, **kwargs: None
    model = nn.Linear(1, 1)
    engine = TrainingEngine(replace(_config_for_test(), total_epochs=config['total_epochs'], checkpoint_epochs=(2,)), device='cpu')
    layout = RunLayout(Path(output))
    engine.fit(model, PriorityCsvProvider(), Path(output), identity={'run': 'priority-recovery'}, arm='joint',
               stop_after=int(stop), resume_checkpoint=layout.read_path('latest_model.pt') if resume else None)


def test_fresh_process_csv_recovery(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from honf_runtime.run_layout import RunLayout
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'seed': 123, 'total_epochs': 2}))
    code = ('import sys; from tests.test_state_priority import _recovery_worker; '
            '_recovery_worker(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1")')
    def run(name, stop, resume):
        subprocess.run([sys.executable, '-c', code, str(tmp_path / name), str(config), str(stop), str(int(resume))],
                       check=True, env=os.environ.copy(), stdout=subprocess.DEVNULL)
    run('full', 2, False)
    run('resumed', 1, False)
    run('resumed', 2, True)
    full = RunLayout(tmp_path / 'full')
    resumed = RunLayout(tmp_path / 'resumed')
    left = torch.load(full.read_path('latest_model.pt'), weights_only=False)
    right = torch.load(resumed.read_path('latest_model.pt'), weights_only=False)
    for name in left['model_state_dict']:
        assert torch.equal(left['model_state_dict'][name], right['model_state_dict'][name])
    for name, state in left['optimizer_state_by_name']['state_by_name'].items():
        for key, value in state.items():
            assert torch.equal(value, right['optimizer_state_by_name']['state_by_name'][name][key])
    assert torch.equal(left['rng_state']['torch_cpu'], right['rng_state']['torch_cpu'])
    assert left['rng_state']['python'] == right['rng_state']['python']
    assert left['sampler_state'] == right['sampler_state']
    assert left['provider_training_state'] == right['provider_training_state'] == {'completed': 2, 'selected': [1, 2]}
    from honf_runtime.scalar_history import read_scalar_rows
    left_rows, right_rows = read_scalar_rows(full.read_path('history.csv')), read_scalar_rows(resumed.read_path('history.csv'))
    for left_row, right_row in zip(left_rows, right_rows):
        left_row.pop('train_seconds'); right_row.pop('train_seconds')
        assert left_row == right_row
    assert json.loads((tmp_path / 'full/configs/metric_definitions.json').read_text()) == json.loads((tmp_path / 'resumed/configs/metric_definitions.json').read_text())
