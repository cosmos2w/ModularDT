"""Native joint-model derivatives preserve original source axes and model state."""

import copy

import numpy as np
import pytest
import torch

from honf_runtime.contextual_frontier_derivatives import live_derivative_probe, source_law_and_reuse_probe
from tests.test_thermal_contextual_joint import _model
from tests.test_thermal_geometry_supervision import _Module, _record


def test_live_native_derivatives_and_restore():
    torch.set_num_threads(1)
    model = _model().double().train()
    record = _record((_Module("M:31", (2.0, 2.0), 1.0), _Module("M:47", (7.5, 4.0), 2.0)))
    weights = copy.deepcopy(model.state_dict())
    result = live_derivative_probe(
        model, record, np.array([[0.75, 0.5], [3.75, 0.5], [9.75, 5.5]]), np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]])
    )
    assert result["source_module_ids"] == ["M:31", "M:47"]
    assert result["source_columns"] == [31, 47]
    assert np.asarray(result["source_jacobian_AD"]).shape == (8, 2)
    assert result["native_forward_calls"] == 7
    assert result["optimizer_updates"] == result["solver_calls"] == 0
    for probe in result["probes"]:
        assert np.isfinite(probe["AD"]).all() and np.isfinite(probe["FD"]).all()
        assert "source_read_support_changed" in probe["plus_seams"]
        if not any(flag for side in ("plus_seams", "minus_seams") for flag in probe[side].values()):
            np.testing.assert_allclose(probe["AD"], probe["FD"], rtol=1e-2, atol=1e-4)
    assert model.training
    for name, value in model.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        else:
            assert value == weights[name]
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("step", [float("nan"), float("inf"), 0.0])
def test_reject_nonfinite_or_zero_probe_steps(step):
    with pytest.raises(ValueError, match="positive finite"):
        live_derivative_probe(_model(), None, None, None, step_fraction=step)


def test_fixed_prepare_reuse_reenters_readers_and_controls_reprepare():
    torch.set_num_threads(1)
    model = _model().double().train()
    record = _record((_Module("M:31", (2.0, 2.0), 1.0), _Module("M:47", (7.5, 4.0), 2.0)))
    result = source_law_and_reuse_probe(
        model, record, np.array([[0.75, 0.5], [3.75, 0.5], [9.75, 5.5]]), np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]])
    )
    assert result["partition_is_nontrivial"]
    assert result["native_forward_calls"] == 6
    assert result["total_observed_neural_calls"]["module_encoder"] == 5
    assert result["cached_apply_neural_calls"]["module_encoder"] == 0
    assert result["cached_apply_neural_calls"]["environment_value"] == 1
    for row in result["per_component"].values():
        assert row["fixed_prepare_reuse_difference_rms"] == 0
        assert row["fresh_prepare_difference_rms"] == 0
        assert np.isfinite(row["heat_additivity_residual_rms"])
    assert model.training
    assert all(not module._forward_hooks for module in model.modules())
    assert all(parameter.grad is None for parameter in model.parameters())
