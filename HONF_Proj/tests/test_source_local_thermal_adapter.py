"""Source-local initial heat features and train-only fixed scale fitting."""

import h5py
import numpy as np
import pytest
import torch

from channelthermal.input_adapter import ChannelThermalInputAdapter, fit_source_local_heat_scale


def _inputs(heat):
    return dict(re=torch.tensor([90.]), u_in=torch.tensor([.1]),
        module_centers=torch.tensor([[[2., 1.], [4., 2.], [8., 3.]]]),
        heat_powers=heat, module_present=torch.ones(1, 3),
        material_params=torch.tensor([.001, .02, .03, 1., 2., .4]))


def test_changing_one_heat_has_no_background_or_other_initial_token_path():
    adapter = ChannelThermalInputAdapter(global_feature_schema="source_local_v3", fixed_heat_scale=3.)
    first_heat = torch.tensor([[.2, -.7, 1.3]], requires_grad=True)
    first = adapter(**_inputs(first_heat))
    second = adapter(**_inputs(torch.tensor([[2., -.7, 1.3]])))
    torch.testing.assert_close(first.global_context, second.global_context, atol=0, rtol=0)
    torch.testing.assert_close(first.module_features[:, 1:], second.module_features[:, 1:], atol=0, rtol=0)
    assert first.global_context.shape[-1] == len(adapter.global_context_names) == 18
    assert first.module_features.shape[-1] == len(adapter.feature_names) == 10
    assert adapter.feature_names[2] == "heat_train_scaled"
    gradient, = torch.autograd.grad(first.global_context.sum() + first.module_features[:, 1:].sum(), first_heat)
    assert gradient[0, 0] == 0 and gradient[0, 1:].abs().sum() > 0
    # Physical heat used by local Stage-A remains caller-owned and unchanged.
    torch.testing.assert_close(first.heat_powers, first_heat, atol=0, rtol=0)


@pytest.mark.parametrize("scale", [None, 0, -1, float("nan"), float("inf")])
def test_source_local_mode_requires_explicit_finite_training_scale(scale):
    with pytest.raises(ValueError, match="training-fitted"):
        ChannelThermalInputAdapter(global_feature_schema="source_local_v3", fixed_heat_scale=scale)


def test_heat_scale_fit_never_reads_excluded_validation_or_inactive_heat(tmp_path):
    path = tmp_path / "cases.h5"
    with h5py.File(path, "w") as handle:
        cases = handle.create_group("cases")
        for case_id, heat, present in (("train", [3., -5., 1000.], [1, 1, 0]),
            ("validation", [1e9], [1])):
            group = cases.create_group(case_id)
            group["heat_powers"] = np.asarray(heat)
            group["module_present"] = np.asarray(present)

    class TrainingNormalizer:
        def normalize_heat_power(self, value):
            return (value - 1.) / 2.

    assert fit_source_local_heat_scale(path, ["train"]) == 5.
    assert fit_source_local_heat_scale(path, ["train"], TrainingNormalizer()) == 3.
