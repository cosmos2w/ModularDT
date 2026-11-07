import copy
import pytest
import torch

from channelthermal.training.r16_7_prescribed_task import (initialize_prescribed_weights,
    make_prescribed_recipe, validate_prescribed_recipe, COEFFICIENT)
from test_r16_5_warm_fork import identities
from relation_anchored_train import engine_for


class Encoder(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.environment_encoder = torch.nn.Sequential(torch.nn.Linear(width, 128), torch.nn.Tanh())
        self.other = torch.nn.Linear(4, 3)


def test_p64_mapping_parity_and_active_new_columns():
    torch.manual_seed(4)
    source, target = Encoder(22), Encoder(358)
    state = copy.deepcopy(source.state_dict())
    initialize_prescribed_weights(target, state)
    assert torch.equal(target.other.weight, source.other.weight)
    assert torch.equal(target.environment_encoder[0].bias, source.environment_encoder[0].bias)
    w = target.environment_encoder[0].weight
    assert torch.equal(w[:, :6], state['environment_encoder.0.weight'][:, :6])
    assert torch.count_nonzero(w[:, 6:342]) == 0
    assert torch.equal(w[:, 342:], state['environment_encoder.0.weight'][:, 6:])
    old = torch.randn(5, 22)
    patch = torch.randn(5, 336)
    expanded = torch.cat((old[:, :6], patch, old[:, 6:]), dim=1)
    torch.testing.assert_close(source.environment_encoder(old), target.environment_encoder(expanded), rtol=1e-5, atol=1e-6)
    target.environment_encoder(expanded).sum().backward()
    assert torch.count_nonzero(w.grad[:, 6:342]) > 0


def test_formal_recipe_and_strict_identity():
    p64, h0, _ = identities()
    def build(arm):
        return make_prescribed_recipe(p64, h0, arm=arm, name=arm, output_root='/tmp',
            membership='full_train600', local_horizon=500, first_segment_epochs=100)
    for arm in ('P0', 'P1'):
        recipe = build(arm)
        validate_prescribed_recipe(None, recipe, p64, h0)
        assert recipe['r16_7_prescribed']['coefficient'] == COEFFICIENT
        assert recipe['member_routing']['model']['base_model_config']['environment_input_dim'] == 342
        engine = engine_for(recipe, device='cpu')
        assert engine.config.total_epochs == 500
        assert engine.config.checkpoint_epochs == (100, 200, 300, 400, 500)
        recipe['training']['monitor_every'] = 1
        with pytest.raises(ValueError, match='differs'):
            validate_prescribed_recipe(None, recipe, p64, h0)
    with pytest.raises(ValueError):
        make_prescribed_recipe(p64, h0, arm='P1', name='x', output_root='/tmp',
            membership='full_train600', local_horizon=100, first_segment_epochs=100)


def test_prescribed_input_checkpoint_resume_preserves_sampling_clocks(tmp_path):
    from test_r16_5_warm_fork import ToyProvider
    from channelthermal.training.r16_5_warm_fork import ThermalR165WarmForkEngine
    from honf_runtime.run_layout import resolve_checkpoint
    from honf_runtime.unified_training import EngineConfig, SamplingKey, TaskBatch

    class PatchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.environment_encoder = torch.nn.Sequential(torch.nn.Linear(358, 128), torch.nn.Tanh())
            self.output = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(128, 1))
        def forward(self, inputs):
            return self.output(self.environment_encoder(inputs))

    class PatchProvider(ToyProvider):
        def make_batch(self, cases, key):
            # Fixed case/epoch-derived prescribed patch inputs retain the input
            # clock across a real engine checkpoint/resume.
            base = torch.tensor(cases, dtype=torch.float32).reshape(-1, 1)
            patch = torch.arange(336, dtype=torch.float32).reshape(1, -1).expand(len(cases), -1)
            patch = (patch + base + key.epoch) / 6000
            inputs = torch.cat((base.expand(-1, 6) / 10, patch,
                                base.expand(-1, 16) / 10), dim=1)
            return TaskBatch(inputs, None, 2 * base + 1, {}, tuple(cases))

    config = EngineConfig(seed=0, microbatch_cases=2, effective_cases=3, total_epochs=2,
        training_mode='joint', sampling_version=SamplingKey.CASE_EPOCH_VERSION, monitor_every=100,
        checkpoint_epochs=(2,), latest_every=1, curve_every=None, write_initial_artifacts=False,
        progress_enabled=False, gradient_clip=1.)
    engine = ThermalR165WarmForkEngine(config, device='cpu')
    torch.manual_seed(14)
    initial = PatchModel()
    full, resumed = copy.deepcopy(initial), copy.deepcopy(initial)
    identity = {'workflow': 'test_r16_7', 'recipe': {'r16_7_prescribed': {'arm': 'P1'}}}
    torch.manual_seed(91)
    engine.fit(full, PatchProvider(), tmp_path / 'full', identity=identity, arm='A', stop_after=2)
    torch.manual_seed(91)
    engine.fit(resumed, PatchProvider(), tmp_path / 'resumed', identity=identity, arm='A', stop_after=1)
    engine.fit(resumed, PatchProvider(), tmp_path / 'resumed', identity=identity, arm='A', stop_after=2,
               resume_checkpoint='latest')
    saved = torch.load(resolve_checkpoint(tmp_path / 'resumed', 'latest'), map_location='cpu', weights_only=False)
    assert [row['sampler_epoch'] for row in saved['history']] == [5101, 5102]
    assert [row['fit_sampler_epoch'] for row in saved['history']] == [5101, 5102]
    assert saved['sampler_state']['next_fit_sampling_epoch'] == 5103
    assert all(torch.equal(full.state_dict()[name], resumed.state_dict()[name]) for name in full.state_dict())
