"""Post-training preserves the parent state while explicitly changing supervision."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from honf_runtime.unified_training import (
    EngineConfig, OptimizerGroupSpec, SamplingKey, ScheduleSpec, TrainingEngine,
    named_optimizer_state,
)
from channelthermal.training.r16_10_posttrain import (
    PosttrainEngine, PosttrainNativeTask, attach_posttrain_clock, make_recipe, validate_parent,
)


def parent():
    recipe = json.loads(Path('configs/research/thermal_r16_10_b0_train600.json').read_text())
    return {'epoch': 3000, 'experiment_identity': {'recipe': recipe,
            'provider_identity': {'training_case_ids': list(range(600))}},
            'history': [{'epoch': i} for i in range(1, 3001)],
            'model_state_dict': {'weight': torch.ones(1)}, 'optimizer_state_by_name': {'state': 1},
            'rng_state': {'present': True}, 'provider_training_state': {'training_elapsed_seconds': 1234.5}}


def test_recipe_retains_model_native_loss_data_and_calibration():
    source = parent()
    original = copy.deepcopy(source['experiment_identity']['recipe'])
    result = make_recipe(source, '/data2/parent/checkpoints/epoch_3000_model.pt',
                         name='b0_to_r', output_root='/data2/child')
    assert source['experiment_identity']['recipe'] == original
    for key in ('dataset', 'member_routing', 'objective', 'evaluation'):
        assert result[key] == original[key]
    assert result['r16_10_family'] == {**original['r16_10_family'], 'arm': 'R'}
    assert result['initialization']['pretrained_weights'] is True
    assert result['initialization']['source_parent_age'] == 3000
    assert result['optimizer']['lr'] == 3e-5
    assert result['optimizer']['kind'] == 'inherited_named_adamw_from_parent'
    assert result['training']['local_horizon'] == 1000
    assert result['training']['checkpoint_epochs'] == list(range(100, 1001, 100))
    assert result['training']['wall_clock_budget_seconds'] == 181235
    assert result['r16_10_posttrain']['optimizer_state'] == 'inherited_named_adamw'


@pytest.mark.parametrize('change', ['epoch', 'arm', 'access', 'membership', 'history', 'optimizer'])
def test_parent_rejects_wrong_lineage_or_incomplete_state(change):
    source = parent()
    recipe = source['experiment_identity']['recipe']
    if change == 'epoch': source['epoch'] = 2999
    if change == 'arm': recipe['r16_10_family']['arm'] = 'R'
    if change == 'access': recipe['member_routing']['model']['access_measure'] = 'active_domain'
    if change == 'membership': source['experiment_identity']['provider_identity']['training_case_ids'].pop()
    if change == 'history': source['history'].pop(123)
    if change == 'optimizer': source.pop('optimizer_state_by_name')
    with pytest.raises(ValueError): validate_parent(source)


def test_local_posttrain_epoch_continues_absolute_native_case_order():
    class Cases:
        def epoch_cases(self, epoch, seed): return tuple(range(600))
        def identity_payload(self): return {'dataset': 'test', 'training_case_ids': list(range(600))}
    config = EngineConfig(seed=0, microbatch_cases=24, effective_cases=48, total_epochs=4000,
                          sampling_version=SamplingKey.CASE_EPOCH_VERSION, training_mode='joint')
    child = PosttrainEngine(config, device='cpu')
    baseline = TrainingEngine(config, device='cpu')
    assert child.sampler_epoch(1) == 3001 and child.sampler_epoch(1000) == 4000
    assert child.epoch_case_order(Cases(), 1) == baseline.epoch_case_order(Cases(), 3001)


def test_terminal_learning_rate_and_named_moments_are_preserved():
    schedule = ScheduleSpec(**PosttrainNativeTask._schedule_payload(None))
    assert schedule.value(1) == schedule.value(40) == schedule.value(1000) == 3e-5
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW([{'params': model.parameters(), 'group_name': 'joint'}], lr=3e-5)
    model(torch.ones(1, 2)).sum().backward(); optimizer.step()
    seed = named_optimizer_state(model, optimizer)
    class Provider:
        def optimizer_groups(self, model, arm, phase):
            return (OptimizerGroupSpec('joint', tuple(n for n, _ in model.named_parameters()), schedule),)
    engine = PosttrainEngine(EngineConfig(seed=0, microbatch_cases=24, effective_cases=48,
                                        total_epochs=1000, training_mode='joint'), device='cpu')
    restored, _ = engine._make_optimizer(model, Provider(), 'joint', 1, seed)
    result = named_optimizer_state(model, restored)
    assert result['groups'] == seed['groups']
    for name, values in seed['state_by_name'].items():
        for key, value in values.items(): assert torch.equal(value, result['state_by_name'][name][key])


def test_auxiliary_ramp_uses_first_epoch_and_resumes_family_cursor():
    from channelthermal.training.r16_10_corpus import EligibleFamilySchedule
    from channelthermal.training.r16_10_family_task import attach_family_objective
    def provider():
        obj = SimpleNamespace(schedule=EligibleFamilySchedule(
            list(map(str, range(48))), list(map(str, range(96))), list(map(str, range(600)))),
            response_scales={}, sensitivity={}, research_scales={})
        obj.family_components = lambda model, source, family: (
            {'endpoint': model.weight.square().sum(), 'response': 4 * model.weight.square().sum()}, {})
        p = SimpleNamespace(identity_payload=lambda: {})
        attach_family_objective(p, obj, arm='R', coefficients={'endpoint': .2, 'response': .01},
                                families_per_epoch={'original': 4, 'new': 8})
        return attach_posttrain_clock(p, {'auxiliary_ramp': 'min(local_epoch / 40, 1)'})
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad(): model.weight.fill_(1.)
    p = provider()
    rows = [r for i in range(13) for r in p.auxiliary_update_terms(
        model, epoch=1, update_index=i, update_count=13)]
    assert len(rows) == 12
    loss = sum(term.numerator * term.weight / term.denominator for terms, _ in rows
               for term in terms.values()) / 13
    assert loss.item() == pytest.approx(.025 * (.2 + .04))
    loss.backward(); assert model.weight.grad.abs().sum() > 0
    state = p.training_state_dict()
    assert state['r16_10_family']['cursor'] == {'original': 4, 'new': 8}
    restored = provider(); restored.load_training_state_dict(state)
    for i in range(13):
        list(restored.auxiliary_update_terms(model, epoch=2, update_index=i, update_count=13))
    assert restored.training_state_dict()['r16_10_family']['cursor'] == {'original': 8, 'new': 16}


def test_launcher_binds_budget_on_resume_and_retains_clean_save_reserve(monkeypatch, tmp_path):
    from tools import thermal_r16_10_posttrain as cli
    from channelthermal.training.contextual_joint_task import ThermalContextualJointTask
    from channelthermal.training import contextual_joint_task as contextual

    recipe = make_recipe(parent(), tmp_path / 'parent/checkpoints/epoch_3000_model.pt',
                         name='child', output_root=tmp_path)
    recipe['training']['wall_clock_budget_seconds'] = 10
    recipe['training']['clean_save_reserve_seconds'] = 2
    recipe_path = tmp_path / 'recipe.json'
    recipe_path.write_text(json.dumps(recipe))
    output = tmp_path / 'child'
    output.mkdir()
    provider = ThermalContextualJointTask.__new__(ThermalContextualJointTask)
    provider.state = {'training_elapsed_seconds': 0.}
    clock = [0.]
    monkeypatch.setattr(contextual.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(cli, 'resolve_checkpoint', lambda *args: Path('unused'))
    monkeypatch.setattr(cli, 'load_trusted_checkpoint', lambda *args, **kwargs: {'epoch': 1})
    monkeypatch.setattr(cli, 'build_posttrain_task', lambda *args, **kwargs:
                        (SimpleNamespace(architecture='R14-Detail-LocalECover'), provider))

    class Engine:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, model, actual_provider, actual_output, **kwargs):
            assert actual_provider is provider
            assert provider._budget_binding == (output, 10., 2.)
            assert kwargs['resume_checkpoint'] == 'latest'
            clock[0] = 7.
            provider._check_wall_clock_budget()
            assert not (output / 'CLEAN_STOP_REQUEST.json').exists()
            clock[0] = 8.
            provider._check_wall_clock_budget()
            request = json.loads((output / 'CLEAN_STOP_REQUEST.json').read_text())
            assert request['reason'] == 'finite training wall budget reached'
            return {'completed_epoch': 1}

    monkeypatch.setattr(cli, 'PosttrainEngine', Engine)
    assert cli.main(['--recipe', str(recipe_path), '--output', str(output),
                     '--device', 'cpu', '--resume', 'latest', '--stop-after', '1000']) == 0
