"""Real-compute contract for the Run-1501 WindFarm adapter."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from honf_runtime.config_loader import load_config_bundle
from honf_runtime.registry import load_case_plugin

from windfarm.data import WindFarmNativeView, case_batch
from windfarm.model import WindFarmForwardModel, build_windfarm_forward_config
from windfarm.normalization import VelocityNormalizer
from windfarm.splits import make_group_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE = PROJECT_ROOT / "src/config_core/forward/windfarm_sparse_incidence_group_control_b16_q8192.json"
VOLUME_ROOT = PROJECT_ROOT / "Case_WindFarm/Dataset/links/wind_farm"
NORMALIZATION = PROJECT_ROOT / "Case_WindFarm/Dataset/derived/forward_velocity_v1/normalization.json"


def test_run1501_windfarm_adapter_performs_real_3d_update() -> None:
    if not VOLUME_ROOT.exists() or not NORMALIZATION.is_file():
        pytest.skip("native WindFarm resources are unavailable")
    bundle = load_config_bundle(PROFILE)
    plugin = load_case_plugin(bundle.case["plugin"])
    plugin.validate_config(bundle)
    dataset = bundle.case["dataset"]
    assert dataset["batch_size"] == 16
    assert dataset["q_train"] == 8192
    assert dataset["env_token_shape"] == [16, 8, 4]
    assert dataset["receiver_chunk_size"] == 512

    payload = copy.deepcopy(bundle.effective["model"]["core_honf"])
    payload["hidden_dim"] = 32
    payload["query_fourier_frequencies"] = 2
    payload["position_fourier_frequencies"] = 2
    payload["interface_model"].update(
        {
            "message_hidden_dim": 32,
            "attention_heads": 4,
            "relative_fourier_frequencies": 2,
            "receiver_chunk_size": 8,
            "activation_checkpointing": False,
        }
    )
    config = build_windfarm_forward_config(payload)
    assert config.forward_architecture == "sparse_incidence_group_control_honf"
    assert config.spatial_dim == 3
    assert config.spatial_scale() == (50.0, 38.0, 6.25)
    restored = build_windfarm_forward_config(config.to_dict())
    assert restored.to_dict() == config.to_dict()

    view = WindFarmNativeView(VOLUME_ROOT)
    split = make_group_split(np.asarray(view.volume.array("layout_index")), seed=42)
    case = view.run(int(split.train[0]))
    normalizer = VelocityNormalizer.from_dict(json.loads(NORMALIZATION.read_text(encoding="utf-8")))
    sample = case.sample_queries(16, np.random.default_rng(2104), mode="volume")
    batch = case_batch(
        case,
        sample.coords_D,
        normalizer=normalizer,
        velocity_mps=sample.velocity_mps,
    )
    assert tuple(batch.env_coords.shape) == (1, 512, 3)
    assert float(batch.env_weights.sum()) == pytest.approx(case.support.volume_D3, rel=1.0e-6)

    torch.manual_seed(2104)
    model = WindFarmForwardModel(config, velocity_transform=normalizer).cpu()
    model.materialize(case_batch(case, case.module_centers))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-5)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch)["pred_field"]
    assert tuple(prediction.shape) == (1, 16, 3)
    assert torch.isfinite(prediction).all()
    loss = (prediction - batch.target_field).square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    tracked = next(parameter for parameter in model.parameters() if parameter.grad is not None)
    before = tracked.detach().clone()
    optimizer.step()
    assert torch.linalg.vector_norm(tracked.detach() - before).item() > 0.0


__all__ = ["test_run1501_windfarm_adapter_performs_real_3d_update"]
