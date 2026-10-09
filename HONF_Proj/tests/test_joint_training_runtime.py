"""Joint lifecycle preserves ordinary updates, validation, and exact resume."""
import copy
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_unified_training import _config, _ToyProvider

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.run_layout import resolve_checkpoint
from honf_runtime.unified_training import (
    TrainingEngine,
    _branch_joint_locality_amendment,
    _engine_config_payload,
    capture_rng_state,
    named_optimizer_state,
)


def _joint_config():
    return replace(_config(), training_mode="joint", warmup_epochs=0,
                   open_through_epoch=0, soft_through_epoch=0,
                   sampling_version="case_epoch_v1")


def test_joint_exact_resume_and_validation_keep_all_heads_active(tmp_path, monkeypatch):
    import honf_runtime.unified_training as runtime
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args, **kwargs: None)
    torch.manual_seed(71)
    initial = torch.nn.Linear(1, 1).state_dict()

    class JointProvider(_ToyProvider):
        def __init__(self):
            super().__init__()
            self.dev_grad_states = []

        def predict_native(self, model, *args, **kwargs):
            if not model.training:
                self.dev_grad_states.append(torch.is_grad_enabled())
            return super().predict_native(model, *args, **kwargs)

    def run(directory, stop, resume=False):
        model = torch.nn.Linear(1, 1)
        model.load_state_dict(initial)
        provider = JointProvider()
        engine = TrainingEngine(_joint_config(), device="cpu")
        engine.fit(model, provider, directory, identity={"workflow": "joint_test"},
                   arm="J-H", stop_after=stop,
                   resume_checkpoint=resolve_checkpoint(directory, "latest") if resume else None)
        assert all(phase[0] == "joint" for phase in provider.validation_phases)
        assert provider.dev_grad_states and not any(provider.dev_grad_states)
        assert torch.is_grad_enabled()
        return load_trusted_checkpoint(resolve_checkpoint(directory, "latest"), map_location="cpu")

    complete = run(tmp_path / "complete", 10)
    run(tmp_path / "resumed", 4)
    resumed = run(tmp_path / "resumed", 10, True)
    assert complete["workflow"] == "joint_test"
    for name, value in complete["model_state_dict"].items():
        torch.testing.assert_close(value, resumed["model_state_dict"][name], rtol=0, atol=0)
    for name, state in complete["optimizer_state_by_name"]["state_by_name"].items():
        for key, value in state.items():
            other = resumed["optimizer_state_by_name"]["state_by_name"][name][key]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
    assert sum(row["optimizer_updates"] for row in resumed["history"]) == 20
    assert all(row["phase"] == "joint" for row in resumed["history"])


def test_joint_mode_is_explicit_and_legacy_resume_identity_unchanged():
    legacy = _engine_config_payload(_config())
    assert "training_mode" not in legacy
    assert _engine_config_payload(_joint_config())["training_mode"] == "joint"
    with pytest.raises(ValueError, match="Training mode"):
        replace(_config(), training_mode="unknown")


def _cli():
    path = Path(__file__).parents[1] / "tools/joint_regional_train.py"
    spec = importlib.util.spec_from_file_location("joint_regional_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task", ["thermal", "wind"])
def test_formal_recipe_rejects_external_models_and_implicit_launch(task):
    module = _cli()
    path = Path(__file__).parents[1] / f"src/config_core/forward/joint_regional/{task}_full5000_v1.json"
    recipe = module.read_recipe(path)
    assert recipe["total_epochs"] == 5000 and recipe["initialization"] == "fresh_all_trainable"
    with pytest.raises(ValueError, match="unsupported"):
        module.validate_recipe({**recipe, "flow_checkpoint": "old_D_sep.pt"})
    with pytest.raises(SystemExit):
        module.main(["start", "--recipe-json", str(path), "--output-dir", "/unused", "--stop-after", "5000"])


def _revision_declaration(tmp_path):
    evidence = tmp_path / 'prefit_revision.json'
    evidence.write_text(json.dumps({'source_epoch': 100, 'stopping_boundary': 200,
        'change': 'registered_gaussian_attention_prior', 'prefit_optimizer_updates': 0,
        'solver_attempts': 0, 'WindTEST_targets': 'locked'}))
    return {'kind': 'registered_locality_prior_branch', 'source_epoch': 100, 'review_stop': 200,
        'from_strength': 0.0, 'to_strength': 1.0, 'evidence_path': str(evidence),
        'evidence_sha256': hashlib.sha256(evidence.read_bytes()).hexdigest(),
        'reason': 'A diagnosed locality deficit; keep every other scientific contract.'}


def test_joint_locality_child_preserves_age_moments_rng_and_resets_selection(tmp_path, monkeypatch):
    import honf_runtime.unified_training as runtime
    monkeypatch.setattr(runtime, '_render_loss_curves', lambda *args, **kwargs: None)
    config = replace(_joint_config(), total_epochs=200, monitor_every=100, latest_every=100, curve_every=100)

    class Provider(_ToyProvider):
        def __init__(self, strength):
            super().__init__()
            self.strength = strength
            self.first_weights = None
            self.first_rng = None

        def identity_payload(self):
            model_config = {} if self.strength == 0 else {'locality_prior_strength': self.strength}
            return {**super().identity_payload(), 'model_config': model_config}

        def predict_native(self, model, *args, **kwargs):
            if model.training and self.first_weights is None:
                self.first_weights = copy.deepcopy(model.state_dict())
                self.first_rng = capture_rng_state()
            result, auxiliary = super().predict_native(model, *args, **kwargs)
            return result * (1 + self.strength), auxiliary

        def reduce_native_metrics(self, records):
            # The parent's deliberately better score must not select a state
            # evaluated under the previous function for this child.
            return {**super().reduce_native_metrics(records), 'field_score': 100 * self.strength}

    def identity(strength):
        recipe = {'mode': 'J-H', 'formal_full': False}
        if strength:
            recipe['locality_prior_strength'] = strength
        return {'workflow': 'joint_test', 'recipe': recipe, 'initial_model_state_sha256': 'same-fresh-seed'}

    parent_dir, child_dir = tmp_path / 'parent', tmp_path / 'child'
    TrainingEngine(config, device='cpu').fit(torch.nn.Linear(1, 1), Provider(0), parent_dir,
        identity=identity(0), arm='J-H', stop_after=100)
    parent_path = resolve_checkpoint(parent_dir, 'latest')
    parent_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    parent = load_trusted_checkpoint(parent_path, map_location='cpu')
    engine = TrainingEngine(config, device='cpu')
    make_optimizer = engine._make_optimizer
    restored_moments = {}

    def capture_optimizer(*args, **kwargs):
        optimizer, specs = make_optimizer(*args, **kwargs)
        restored_moments.update(named_optimizer_state(args[0], optimizer))
        return optimizer, specs

    monkeypatch.setattr(engine, '_make_optimizer', capture_optimizer)
    provider = Provider(1.0)
    engine.fit(torch.nn.Linear(1, 1), provider, child_dir, identity=identity(1.0), arm='J-H',
        stop_after=200, branch_from_checkpoint=parent_path,
        joint_core_revision=_revision_declaration(tmp_path))
    for name, value in parent['model_state_dict'].items():
        torch.testing.assert_close(provider.first_weights[name], value, rtol=0, atol=0)
    for name, state in parent['optimizer_state_by_name']['state_by_name'].items():
        for key, value in state.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(restored_moments['state_by_name'][name][key], value, rtol=0, atol=0)
    torch.testing.assert_close(provider.first_rng['torch_cpu'], parent['rng_state']['torch_cpu'], rtol=0, atol=0)
    child = load_trusted_checkpoint(resolve_checkpoint(child_dir, 'latest'), map_location='cpu')
    assert child['epoch'] == 200 and len(child['history']) == 200
    assert sum(row['optimizer_updates'] for row in child['history']) == 400
    assert sum(row['case_visits'] for row in child['history']) == 1000
    assert child['branch_parent']['inherited_optimizer_updates'] == 200
    assert child['branch_parent']['inherited_case_visits'] == 500
    assert child['history'][:100] == parent['history']
    assert load_trusted_checkpoint(resolve_checkpoint(child_dir, 'best_field'), map_location='cpu')['epoch'] == 200
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == parent_hash
    with pytest.raises(ValueError, match='identity'):
        TrainingEngine(config, device='cpu').fit(torch.nn.Linear(1, 1), Provider(1.0), parent_dir,
            identity=identity(1.0), arm='J-H', stop_after=200, resume_checkpoint=parent_path)


@pytest.mark.parametrize('binding', ['dataset', 'loss', 'schedule', 'stream', 'width'])
def test_joint_locality_branch_rejects_other_changed_bindings(tmp_path, binding):
    saved = {'engine_config': {'training_mode': 'joint', 'seed': 17},
        'recipe': {'mode': 'J-H', 'formal_full': False},
        'provider_identity': {'model_config': {'hidden': 16}, 'dataset': 'fixed', 'loss': 1.0}}
    current = copy.deepcopy(saved)
    current['recipe']['locality_prior_strength'] = 1.0
    current['provider_identity']['model_config']['locality_prior_strength'] = 1.0
    if binding == 'schedule':
        current['engine_config']['total_epochs'] = 1000
    elif binding == 'stream':
        current['engine_config']['seed'] = 18
    elif binding == 'width':
        current['provider_identity']['model_config']['hidden'] = 32
    else:
        current['provider_identity'][binding] = 'changed'
    with pytest.raises(ValueError, match='other experiment binding'):
        _branch_joint_locality_amendment(saved, current, _revision_declaration(tmp_path))
