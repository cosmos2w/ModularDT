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
