"""Recommended profile dispatch, physical identity and configured clean-save budgets."""
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('entrypoint_integration', PROJECT / 'tools/relation_anchored_train.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def thermal_recipe():
    return {'training_profile': 'state_response', 'dataset': {'name': 'thermal'},
        'initialization': {'seed': 0}, 'member_routing': {'arm': 'R14-Detail-LocalECover'},
        'training': {'wall_clock_budget_seconds': 10, 'clean_save_reserve_seconds': 2}}


def test_recommended_profile_dispatches_without_changing_saved_recipe(monkeypatch):
    module = ModuleType('channelthermal.training.thermal_joint_response_task')
    expected = object(), object()
    recipe = thermal_recipe()
    original = deepcopy(recipe)
    saved = {'epoch': 9}
    def build(actual, *, device, resume_payload):
        assert actual == original and device == 'cpu' and resume_payload is saved
        return expected
    module.build_thermal_joint_response_task = build
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert cli.build(recipe, device='cpu', resume_payload=saved) == expected
    assert recipe == original


def test_unsupported_case_cannot_silently_drop_default_response():
    recipe = {'dataset': 'wind', 'seed': 0, 'training_profile': 'state_response'}
    with pytest.raises(ValueError, match='no state-response training adapter'):
        cli.build(recipe, device='cpu')


def test_native_only_optout_reuses_existing_case_adapter(monkeypatch):
    module = ModuleType('windfarm.training.relation_anchored_task')
    expected = object(), object()
    module.build_wind_relation_anchored_task = lambda recipe, *, device: expected
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert cli.build({'dataset': 'wind', 'seed': 0, 'training_profile': 'state_response',
                      'state_response': {'enabled': False}}, device='cpu') == expected


@pytest.mark.parametrize('budget,reserve', [(0, 0), (10, 10), (True, 0), (float('nan'), 0), (10, -1)])
def test_invalid_budget_fails_explicitly(budget, reserve):
    with pytest.raises(ValueError, match='budget'):
        cli.bind_training_budget(object(), {'training': {'wall_clock_budget_seconds': budget,
            'clean_save_reserve_seconds': reserve}}, Path('/unused'))


@pytest.mark.parametrize('resume', [False, True])
def test_standard_fit_binds_budget_and_requests_clean_save(monkeypatch, tmp_path, resume):
    from channelthermal.training import contextual_joint_task as contextual
    from channelthermal.training.contextual_joint_task import ThermalContextualJointTask
    recipe = thermal_recipe()
    # Test the ordinary CLI without opening actual physical data.
    recipe.pop('training_profile')
    path = tmp_path / 'recipe.json'; path.write_text(json.dumps(recipe))
    output = tmp_path / 'run'; output.mkdir()
    provider = ThermalContextualJointTask.__new__(ThermalContextualJointTask)
    provider.state = {'training_elapsed_seconds': 3. if resume else 0.}
    clock = [0.]
    monkeypatch.setattr(contextual.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(cli, 'build', lambda *a, **kw: (SimpleNamespace(architecture='R14-Detail-LocalECover'), provider))
    class Engine:
        def fit(self, model, actual, out, **kwargs):
            assert actual._budget_binding == (output, 10., 2.)
            assert kwargs['resume_checkpoint'] == ('latest' if resume else None)
            elapsed = provider.state['training_elapsed_seconds']
            clock[0] = 7. - elapsed
            provider._check_wall_clock_budget()
            assert not (output / 'CLEAN_STOP_REQUEST.json').exists()
            clock[0] = 8. - elapsed
            provider._check_wall_clock_budget()
            assert json.loads((output / 'CLEAN_STOP_REQUEST.json').read_text())['schema'] == 'ordinary_clean_stop_request_v1'
            return {'clean_stop_requested': True, 'completed_epoch': 1}
    monkeypatch.setattr(cli, 'engine_for', lambda *a, **kw: Engine())
    args = ['fit', '--recipe', str(path), '--output', str(output), '--device', 'cpu', '--stop-after', '1']
    if resume: args += ['--resume', 'latest']
    assert cli.main(args) == 0


def test_truth_join_rejects_changed_thermal_control_with_same_geometry_re_and_heat():
    from channelthermal.physical_provenance import validate_thermal_endpoint_config
    source = {'layout': {'num_modules': 1, 'centers': [[3., 2.]], 'heat_powers': [1.]}}
    changed = deepcopy(source); changed['thermal'] = {'solid_k': 123.}
    with pytest.raises(ValueError, match='thermal'):
        validate_thermal_endpoint_config(source, changed, centers=[[3., 2.]])


def test_recommended_example_enables_response_and_s_access():
    from honf_runtime.state_response import ResponseObjectiveConfig
    recipe = json.loads((PROJECT / 'configs/examples/thermal_relation_anchored_state_response.json').read_text())
    constraints = ResponseObjectiveConfig.from_mapping(recipe['state_response'])
    assert recipe['training_profile'] == 'state_response' and constraints.enabled
    assert recipe['member_routing']['model']['access_measure'] == 'inventory_unit_total'
    assert recipe['training']['curve_every'] == 10
    assert recipe['thermal_joint_response']['native_ports']['angle_count'] == 64
    assert recipe['evaluation']['sample_fraction'] == .1
    assert recipe['training']['local_horizon'] == 2000
    assert not any(key.startswith('r16_') for key in recipe)


def test_configured_thermal_gradient_clip_reaches_engine():
    recipe = json.loads((PROJECT / 'configs/examples/thermal_relation_anchored_state_response.json').read_text())
    recipe['optimizer']['gradient_clip'] = .375
    assert cli.engine_for(recipe, device='cpu').config.gradient_clip == .375


@pytest.mark.parametrize('dry', [False, True])
def test_legacy_cli_forwards_new_profile_instead_of_native_only(monkeypatch, tmp_path, dry):
    import honf_research
    import relation_anchored_train as supported
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps({'training_profile': 'state_response', 'training': {'first_segment_epochs': 100}}))
    calls = []
    monkeypatch.setattr(supported, 'main', lambda args: calls.append(args) or 0)
    args = ['train-member-routing', '--config', str(path), '--device', 'cpu', '--output', str(tmp_path / 'run')]
    if dry: args.append('--dry-run')
    assert honf_research.main(args) == 0
    assert calls[0][0] == ('inspect' if dry else 'fit')
    assert '--recipe' in calls[0]
    if not dry: assert calls[0][calls[0].index('--stop-after') + 1] == '100'


@pytest.mark.parametrize('command,extra', [
    ('resume', ['--stop-after-epoch', '200']),
    ('resume-member-routing', ['--stop-after-epoch', '200']),
    ('evaluate', []),
])
def test_legacy_wrappers_do_not_rebuild_profile_with_native_only_provider(tmp_path, command, extra):
    import honf_research
    directory = tmp_path / 'run' / 'configs'
    directory.mkdir(parents=True)
    (directory / 'experiment_identity.json').write_text(json.dumps({
        'recipe': {'training_profile': 'state_response'}}))
    with pytest.raises(ValueError, match='relation_anchored_train.py'):
        honf_research.main([command, '--run', str(directory.parent), '--device', 'cpu', *extra])


def test_ordinary_research_constructor_cannot_silently_drop_profile(tmp_path):
    import honf_research
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps({'training_profile': 'state_response'}))
    with pytest.raises(ValueError, match='train-member-routing'):
        honf_research.main(['train', '--config', str(path), '--device', 'cpu',
                           '--output', str(tmp_path / 'run')])


@pytest.mark.parametrize('resume', [False, True])
def test_new_profile_records_storage_and_binds_same_run_epoch(monkeypatch, tmp_path, resume):
    from honf_runtime import state_response_storage
    recipe = thermal_recipe()
    recipe['storage'] = {'formal_forward_run_alias': False}
    recipe['training'].pop('wall_clock_budget_seconds')
    path = tmp_path / 'recipe.json'; path.write_text(json.dumps(recipe))
    output = tmp_path / 'run'
    saved = {'epoch': 100, 'experiment_identity': {'recipe': recipe}}
    calls = []
    def prepare(actual, out, model, **kwargs):
        assert actual == recipe and out == output
        assert kwargs == {'completed_epoch': 100 if resume else 0, 'stop_after': 200,
                          'checkpoint_epochs': (100, 200)}
        calls.append('storage')
        return {'run_output': str(output), 'estimated_new_bytes': 100}
    monkeypatch.setattr(state_response_storage, 'prepare_state_response_storage', prepare)
    monkeypatch.setattr(cli, 'load_trusted_checkpoint', lambda *a, **kw: saved)
    monkeypatch.setattr(cli, 'resolve_checkpoint', lambda *a: Path('unused'))
    monkeypatch.setattr(cli, 'build', lambda *a, **kw: (object(), object()))
    class Engine:
        config = SimpleNamespace(checkpoint_epochs=(100, 200), total_epochs=200)
        def fit(self, *args, **kwargs):
            calls.append('fit')
            record = 'storage_preparation_resume.json' if resume else 'storage_preparation.json'
            assert json.loads((output / 'logs' / record).read_text())['estimated_new_bytes'] == 100
            if not resume:
                assert json.loads((output / 'configs/resolved_config.json').read_text()) == recipe
            return {'completed_epoch': 200}
    monkeypatch.setattr(cli, 'engine_for', lambda *a, **kw: Engine())
    args = ['fit', '--recipe', str(path), '--output', str(output), '--stop-after', '200']
    if resume: args += ['--resume', 'latest']
    assert cli.main(args) == 0
    assert calls == ['storage', 'fit']


def test_invalid_profile_stop_writes_no_storage_or_alias(monkeypatch, tmp_path):
    from honf_runtime import state_response_storage
    monkeypatch.setattr(state_response_storage, 'prepare_state_response_storage',
                        lambda *a, **kw: pytest.fail('Storage preparation must not run'))
    engine = SimpleNamespace(config=SimpleNamespace(total_epochs=100))
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='epoch horizon'):
        cli.prepare_profile_fit({}, output, object(), engine, source_payload=None, stop_after=101)
    assert not output.exists()
