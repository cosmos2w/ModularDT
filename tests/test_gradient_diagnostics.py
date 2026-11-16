"""Calibration diagnostics remain portable and leave the caller's state intact."""
import copy
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from honf_runtime.gradient_diagnostics import gradient_diagnostics, preserved_state


def test_diagnostics_and_exception_restore_model_optimizer_and_rng():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    model.weight.grad = torch.tensor([[3.]])
    initial_model = copy.deepcopy(model.state_dict())
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    random.seed(18); np.random.seed(18); torch.manual_seed(18)
    expected_random = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    result = gradient_diagnostics(model, {'native': lambda: model.weight.square().sum(),
        'twice': lambda: [model.weight.square().sum(), model.weight.square().sum()]},
        {'weight': ['weight']}, require_nonzero=True)
    assert result['losses'] == {'native': 4., 'twice': 8.}
    assert result['groups']['global']['norms'] == {'native': 4., 'twice': 8.}
    assert result['groups']['weight']['cosines']['native:twice'] == pytest.approx(1.)
    with pytest.raises(RuntimeError, match='deliberate'):
        with preserved_state(model, optimizer):
            model.train(False); model.weight.grad.fill_(12.); optimizer.step()
            random.random(); np.random.rand(); torch.rand(1)
            raise RuntimeError('deliberate')
    assert torch.equal(model.weight, initial_model['weight'])
    assert model.training and model.weight.grad.item() == 3.
    assert optimizer.state_dict() == initial_optimizer
    assert random.getstate() == expected_random[0]
    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
               for a, b in zip(np.random.get_state(), expected_random[1]))
    assert torch.equal(torch.get_rng_state(), expected_random[2])


def test_builders_import_without_ignored_r16_module():
    root = Path(__file__).resolve().parents[1]
    script = r'''
import importlib.abc, sys
class BlockIgnoredR16(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.endswith('.r16_10_diagnostics'):
            raise ImportError('blocked ignored local R16 helper')
sys.meta_path.insert(0, BlockIgnoredR16())
from channelthermal.training.thermal_native_pair_task import build_thermal_native_pair_task
from channelthermal.training.thermal_field_response import build_thermal_field_response_task
for builder, expected in ((build_thermal_native_pair_task, 'Choose a declared native-pair variant.'),
                          (build_thermal_field_response_task, 'Unknown Thermal field-response variant.')):
    try:
        builder({})
    except ValueError as error:
        assert str(error) == expected
    else:
        raise AssertionError('Expected configuration validation')
assert not any(name.endswith('.r16_10_diagnostics') for name in sys.modules)
'''
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(str(root / p) for p in ('src', 'Case_ThermalChannel/src', 'Case_WindFarm/src', 'tools'))}
    subprocess.run([sys.executable, '-c', script], cwd=root, env=env, text=True, capture_output=True, check=True)
