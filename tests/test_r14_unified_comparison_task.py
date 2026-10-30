"""Native comparison uses the same case-epoch draws as the anchored provider."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from channelthermal.data.datasets import fit_global_normalizer
from channelthermal.data.development_split import development_case_ids
from channelthermal.training.contextual_joint_task import ThermalContextualJointTask
from channelthermal.training.joint_task import _load_data_binding, _read_selected_cases
from channelthermal.training.r14_unified_comparison_task import ThermalR14UnifiedComparisonTask
from honf_runtime.unified_training import SamplingKey


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5')
MANIFEST = Path('/data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json')


class _ClockCore:
    schema = "unified_continuous_joint_native_v1"

    def set_training_clock(self, epoch, update):
        self.clock = (epoch, update)

    def set_organization_clock(self, epoch, update):
        self.clock = (epoch, update)


class _Model:
    architecture = "U-H"
    def __init__(self):
        self.core = _ClockCore()
    def state_payload(self):
        return {"schema": "test_adapter"}


@pytest.mark.skipif(not DATA.is_file() or not MANIFEST.is_file(), reason="local Thermal data unavailable")
def test_actual_case_draws_and_native_targets_match_anchored_sampler():
    manifest, path, manifest_path = _load_data_binding(MANIFEST, DATA)
    ids = development_case_ids(manifest, "train")[:2]
    normalizer = fit_global_normalizer(path, ids)
    cases = _read_selected_cases(path, ids, split="train", normalizer=normalizer, include_grid=False)
    recipe = json.loads((ROOT / 'configs/research/thermal_r13_11s_e_anchor_train150_v1.json').read_text())
    recipe['member_routing'].pop('health', None)
    kwargs = dict(training_cases=cases, validation_cases=(), manifest=manifest, data_path=path,
                  manifest_path=manifest_path, recipe=recipe, device='cpu')
    baseline = ThermalR14UnifiedComparisonTask(model=_Model(), **kwargs)
    anchored = ThermalContextualJointTask(model=_Model(), native_only=True, **kwargs)
    key = SamplingKey(0, 3, 1, 0, 'joint', 'R13-3-Relative',
                      sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id=baseline.sampling_dataset_id)
    left, right = baseline.make_batch(ids, key), anchored.make_batch(ids, key)
    assert left.auxiliary == right.auxiliary
    assert baseline.model.core.clock == (3003, 1)
    for name in ('fluid_xy', 'local_query_points', 'heat'):
        torch.testing.assert_close(getattr(left.receivers, name), getattr(right.receivers, name), rtol=0, atol=0)
    for name in ('field_targets', 'point_weights', 'interface_target', 'interface_valid_mask', 'material_targets'):
        torch.testing.assert_close(getattr(left.targets, name), getattr(right.targets, name), rtol=0, atol=0)
    assert left.receivers.fluid_xy.shape[1] == 1024
    assert left.receivers.local_query_points.shape[-2] == 32
    assert left.receivers.ntheta == 16
    identity = baseline.identity_payload()
    assert identity['objective']['affine_temperature_constraint'] is True
    assert identity['objective']['heat_independent_flow_constraint'] is True
    assert identity['objective']['native_weights'] == recipe['objective']['native_weights']
    assert identity['response_supervision'] is None


def test_unified_prediction_clears_joint_entries_before_prepare():
    events = []
    class Core:
        def joint_begin_microbatch(self): events.append('clear')
    class Model:
        architecture = 'U-H'
        core = Core()
        def prepare_native(self, *args, **kwargs):
            events.append('prepare')
            return SimpleNamespace(union=SimpleNamespace(unique_counts=torch.tensor([1])),
                query_batch=SimpleNamespace(local_weights=torch.ones(1)),
                organization=SimpleNamespace(work_counts={}))
        def apply_native_with_work_counts(self, prepared, heat):
            return object(), {}
    task = object.__new__(ThermalR14UnifiedComparisonTask)
    task.model = Model()
    scene = SimpleNamespace(structure={}, case_ids=('x',))
    receivers = SimpleNamespace(fluid_xy=torch.zeros(1, 1, 2), local_query_points=None, ntheta=1, heat=torch.ones(1))
    for _ in range(2):
        task.predict_native(task.model, scene, receivers, 'U-H', 'joint')
    assert events == ['clear', 'prepare', 'clear', 'prepare']


def test_macro_optout_preserves_native_gradient_and_default_hooks(tmp_path, monkeypatch):
    import copy
    from dataclasses import replace
    from torch import nn
    import honf_runtime.unified_training as runtime
    from honf_runtime.joint_native_objective import JointMacroObjective
    from honf_runtime.run_layout import RunLayout
    from tests.test_unified_training import _ToyProvider, _config

    class HookedLinear(nn.Linear):
        def __init__(self):
            super().__init__(1, 1)
            self.calls = []
        def joint_begin_update(self, epoch, update):
            self.calls.append('begin')
            return {'absolute_epoch': epoch, 'constraints_enabled': False}
        def joint_begin_microbatch(self): self.calls.append('microbatch')
        def joint_microbatch_result(self): raise AssertionError('unexpected constraint observation')
        def joint_constraint_duals(self): raise AssertionError('unexpected constraint dual')
        def joint_after_update(self, epoch, update, residuals): self.calls.append('after')

    model = HookedLinear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(TypeError, match='bool'):
        JointMacroObjective(model, optimizer, 1, 0, enabled=0)
    disabled = JointMacroObjective(model, optimizer, 1, 0, enabled=False)
    disabled.begin_microbatch(); disabled.observe(1); disabled.apply(); disabled.after_update()
    assert not disabled.capable and model.calls == []
    default = JointMacroObjective(model, optimizer, 1, 0)
    default.begin_microbatch(); default.apply(); default.after_update()
    assert default.capable and model.calls == ['begin', 'after']

    class OneCaseProvider(_ToyProvider):
        joint_work_objective_enabled = False
        def epoch_cases(self, epoch, seed): return (1,)
        def validation_batches(self):
            yield self.make_batch((1,), SamplingKey(0, 1, 0, 0, 'joint', 'test'))

    monkeypatch.setattr(runtime, '_render_loss_curves', lambda *args, **kwargs: None)
    baseline = nn.Linear(1, 1)
    candidate = HookedLinear()
    candidate.load_state_dict(copy.deepcopy(baseline.state_dict()))
    config = replace(_config(), total_epochs=1, microbatch_cases=1, effective_cases=1,
                     training_mode='joint', checkpoint_epochs=(1,), progress_enabled=False)
    class PlainProvider(OneCaseProvider): pass
    for m, name, provider in ((baseline, 'plain', PlainProvider()), (candidate, 'hooked', OneCaseProvider())):
        runtime.TrainingEngine(config, device='cpu').fit(m, provider, tmp_path / name,
            identity={'run': name}, arm='test', stop_after=1)
        artifacts = RunLayout(tmp_path / name)
        assert artifacts.read_path('latest_model.pt').is_file()
        assert artifacts.read_path('validation_epoch_0001.json').is_file()
    assert candidate.calls == []
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
