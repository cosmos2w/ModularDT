from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn.parameter import UninitializedParameter

from windfarm.data import NativeCase, case_batch
from windfarm.geometry import (
    environment_representation,
    global_geometry_features,
    module_geometry,
    support_geometry,
)
from windfarm.model import WindFarmForwardModel, build_windfarm_forward_config


def _native_geometry_batch(device: torch.device) -> object:
    x_m = np.asarray([-100.0, 0.0, 100.0], dtype=np.float32)
    y_m = np.asarray([-80.0, 0.0, 80.0], dtype=np.float32)
    z_m = np.asarray([20.0, 70.0, 120.0], dtype=np.float32)
    support = support_geometry(x_m, y_m, z_m)
    environment = environment_representation(support, token_shape=(16, 8, 4))
    turbine_xy = np.asarray(
        [[-0.8, 0.0], [-0.48, 0.0], [-0.16, 0.0], [0.16, 0.0], [0.48, 0.0], [0.8, 0.0]],
        dtype=np.float32,
    )
    centers, present, features = module_geometry(turbine_xy, 6)
    case = NativeCase(
        index=0,
        case="native-wind-test",
        layout="layout-0",
        layout_index=0,
        wind_direction_deg=270.0,
        n_turbines=6,
        run=SimpleNamespace(
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            shape_nxyz=(x_m.size, y_m.size, z_m.size),
        ),
        support=support,
        environment=environment,
        module_centers=centers,
        module_present=present,
        module_features=features,
        global_context=global_geometry_features(support, 270.0, 6),
    )
    query_coords = np.asarray([[-0.1, 0.0, 0.875], [0.1, 0.0, 0.875]], dtype=np.float32)
    return case_batch(case, query_coords, include_receiver_anchors=True).to(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Run this native materialization check on allocated GPU 2")
def test_adaptive_cover_materializes_native_3d_geometry_on_gpu() -> None:
    config = build_windfarm_forward_config(
        {
            "forward_architecture": "adaptive_interaction_cover_honf",
            "field_dim": 3,
            "spatial_dim": 3,
            "coordinate_scale": [50.0, 38.0, 6.25],
            "periodic_axes": [],
            "geometry_mode": "nonperiodic",
            "boundary_feature_mode": "none",
            "hidden_dim": 32,
            "module_radius": 0.5,
            "num_env_tokens_x": 16,
            "num_env_tokens_y": 8,
            "query_fourier_frequencies": 2,
            "position_fourier_frequencies": 2,
            "interface_model": {
                "message_hidden_dim": 16,
                "attention_heads": 4,
                "relative_fourier_frequencies": 2,
                "receiver_chunk_size": 8,
            },
        }
    )
    device = torch.device("cuda:0")
    batch = _native_geometry_batch(device)
    model = WindFarmForwardModel(config).to(device).eval()

    prepared = model.materialize(batch)

    assert model.architecture == "adaptive_interaction_cover_honf"
    assert batch.target_field is None
    assert tuple(prepared.encoded.env_coords.shape) == (1, 512, 3)
    assert model.core.backend.cover_mode == "full_access"
    backend_state = prepared.dense_prepared.backend_state
    assert len(backend_state["cover_trees"]) == 1
    assert backend_state["cover_trees"][0].universe.coordinates.shape == (614, 3)
    assert len(backend_state["cover_plans"]) == 1
    assert not any(isinstance(parameter, UninitializedParameter) for parameter in model.parameters())
    with torch.no_grad():
        prediction = model.predict_standardized(
            prepared,
            batch.query_xy,
            query_features=batch.query_features,
            receiver_chunk_size=8,
        )
    assert prediction.shape == (1, 2, 3)
    assert torch.isfinite(prediction).all()
