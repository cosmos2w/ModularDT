"""Native port evaluation respects the caller's selected member access."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from channelthermal.training.thermal_port_response_task import ThermalPortResponseTask
from channelthermal.training.r13_8_anchored_task import ThermalR138AnchoredMemberRoutingTask
from channelthermal.training.joint_member_routing_task import ThermalJointPairPredictions


class Core:
    _read_execution_override = None

    def set_training_clock(self, *args):
        pass

    @contextmanager
    def read_execution(self, mode):
        previous = self._read_execution_override
        self._read_execution_override = mode
        try:
            yield
        finally:
            self._read_execution_override = previous


class Model:
    architecture = 'detail'

    def __init__(self, training):
        self.training = training
        self.core = Core()
        self.modes = []

    def prepare_native(self, *args, **kwargs):
        return SimpleNamespace(organization=SimpleNamespace(work_counts={'directory': 2}))

    def apply_native_with_work_counts(self, prepared, heat):
        mode = self.core._read_execution_override
        self.modes.append(mode)
        value = 2. if mode == 'full' else 1.
        return {'pred_field': value}, {'pairs': int(value)}, SimpleNamespace(joint_fields=torch.ones(1, 4) * value)

    def _extract_native_roles(self, prepared, temperature, flow):
        return {'pred_interface': temperature}


@pytest.mark.parametrize('requested,expected', [(None, 'restricted'), ('restricted', 'restricted'), ('full', 'full')])
def test_eval_honors_read_context_and_uses_matching_local_values(requested, expected):
    model = Model(False)
    task = object.__new__(ThermalPortResponseTask)
    task.model, task.source_parent_age, task.local_operator = model, 100, None
    scene = SimpleNamespace(structure={}, case_ids=('case',))
    receivers = SimpleNamespace(fluid_xy=None, heat=None, local_query_points=None, ntheta=64)
    with patch('channelthermal.training.thermal_port_response_task.native_angle_view', side_effect=lambda p, n: p), \
         patch('channelthermal.training.thermal_port_response_task.outside_from_native_output', side_effect=lambda m, p, raw: raw.joint_fields[0, 0]), \
         patch('channelthermal.training.thermal_port_response_task.local_response_from_outside', side_effect=lambda m, l, p, h, q, outside: {'interface_pred': torch.ones(1, 64, 2) * outside, 'internal_temperature': outside}), \
         model.core.read_execution(requested):
        prediction, _ = task.predict_native(model, scene, receivers, 'detail', 'joint')
    assert model.modes == [expected]
    assert prediction.read_mode == expected
    assert set(prediction.native64) == set(prediction.local1) == set(prediction.outside64) == {expected}
    assert prediction.work[f'{expected}_decode_pairs'] == (2 if expected == 'full' else 1)
    targets = SimpleNamespace(interface_target=torch.zeros(1, 64, 2))
    with patch.object(ThermalR138AnchoredMemberRoutingTask, 'validation_metrics', side_effect=lambda p, t, a: p.native_main):
        assembled = task.validation_metrics(prediction, targets, None)
    assert assembled['pred_interface'][0, 0, 0].item() == (2 if expected == 'full' else 1)


def test_training_stays_paired_even_in_full_context():
    model = Model(True)
    task = object.__new__(ThermalPortResponseTask)
    task.model, task.source_parent_age, task.local_operator = model, 100, None
    scene = SimpleNamespace(structure={}, case_ids=('case',))
    receivers = SimpleNamespace(fluid_xy=None, heat=None, local_query_points=None, ntheta=64)
    with patch('channelthermal.training.thermal_port_response_task.native_angle_view', side_effect=lambda p, n: p), \
         patch('channelthermal.training.thermal_port_response_task.outside_from_native_output', return_value=None), \
         patch('channelthermal.training.thermal_port_response_task.local_response_from_outside', return_value={}), \
         model.core.read_execution('full'):
        prediction, _ = task.predict_native(model, scene, receivers, 'detail', 'joint')
    assert isinstance(prediction, ThermalJointPairPredictions)
    assert model.modes == ['full', 'restricted']
