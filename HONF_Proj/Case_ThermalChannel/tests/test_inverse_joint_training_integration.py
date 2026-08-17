"""One sparse stage-four GPU batch through frozen HONF.

The batch carries `R,c`, planned `G`, and generated endpoint `D`; autonomous
predicted ports produce differentiable `G_hat`. The test is diagnostic only
and performs one forward/backward pass, not training or optimization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import pytest
import torch
from torch.utils.data import DataLoader

from honf_inverse_core.models.hierarchical_inverse import HierarchicalInverseDesigner
from honf_inverse_core.training.trainer import recursive_to_device
from channelthermal.inverse.dataset_io import InverseH5Dataset
from channelthermal.inverse.differentiable_verifier import DifferentiableThermalChannelVerifier
from channelthermal.inverse.verifier import FrozenThermalChannelVerifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = PROJECT_ROOT / "Trained_Results/ThermalChannel/Inverse_Resources/best_predicted_model_snapshot.pt"
DATASET = PROJECT_ROOT / "Trained_Results/ThermalChannel/Inverse_Dataset_Builds/Build_0003_diagnostic_inverse_data_v1/inverse_dataset_v1.h5"


@pytest.mark.skipif(not CHECKPOINT.is_file() or not DATASET.is_file(), reason="local inverse integration artifacts unavailable")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_one_joint_batch_has_finite_inverse_gradients() -> None:
    expected_physical_gpu = os.environ.get("HONF_TEST_PHYSICAL_GPU")
    if expected_physical_gpu is not None:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == expected_physical_gpu
    dataset = InverseH5Dataset(DATASET, split="train")
    batch = recursive_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cuda:0"))
    with h5py.File(DATASET, "r") as h5:
        heat_mean = float(h5["normalization/active_heat_power_mean"][()])
        heat_std = float(h5["normalization/active_heat_power_std"][()])
        functional_mean = h5["normalization/functional_mean"][...]
        functional_std = h5["normalization/functional_std"][...]
    frozen = FrozenThermalChannelVerifier(CHECKPOINT, device="cuda:0")
    adapter = DifferentiableThermalChannelVerifier(
        frozen,
        inverse_heat_mean=heat_mean,
        inverse_heat_std=heat_std,
        functional_mean=functional_mean,
        functional_std=functional_std,
        query_grid=(12, 8),
        local_grid_size=10,
    )
    designer = HierarchicalInverseDesigner.from_config(
        {
            "num_edges": 6, "max_modules": 12, "request_hidden_dim": 32,
            "plan_hidden_dim": 48, "plan_layers": 1, "layout_hidden_dim": 48,
            "layout_layers": 1, "corrector_enabled": True,
            "corrector_hidden_dim": 48, "corrector_blocks": 1, "dropout": 0.0,
        }
    ).to("cuda:0")
    encoding = designer.encode_request(
        batch["request"], batch["context"], batch["geometry_constraints"], batch["geometry_constraint_mask"]
    )
    endpoint = batch["layout"].clone().requires_grad_(True)
    losses = adapter.joint_loss_hook(designer, batch, encoding, batch["plan"], endpoint)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert endpoint.grad is not None and torch.isfinite(endpoint.grad).all()
    corrector_gradients = [
        parameter.grad for parameter in designer.corrector.parameters() if parameter.grad is not None
    ]
    assert corrector_gradients and all(torch.isfinite(gradient).all() for gradient in corrector_gradients)
    assert not any(parameter.requires_grad for parameter in frozen.model.parameters())
    dataset.close()
