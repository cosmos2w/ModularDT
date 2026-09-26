"""Physical ThermalChannel wrapper keeps the adaptive full-access identity."""

from __future__ import annotations

import torch
from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.model import ChannelThermalHONFModel


def _config(architecture: str) -> ChannelThermalHONFConfig:
    return ChannelThermalHONFConfig.from_dict({
        "core_honf": {
            "forward_architecture": architecture,
            "interface_model": {
                "message_hidden_dim": 8,
                "attention_heads": 2,
                "relative_fourier_frequencies": 2,
                "receiver_chunk_size": 3,
            },
            "field_dim": 5,
            "hidden_dim": 16,
            "domain_length_x": 12.0,
            "domain_length_y": 6.0,
            "coordinate_scale": [12.0, 6.0],
            "module_radius": 0.45,
            "num_env_tokens_x": 4,
            "num_env_tokens_y": 2,
            "dropout": 0.0,
            "boundary_feature_mode": "none",
            "position_fourier_frequencies": 2,
            "query_fourier_frequencies": 2,
        },
        "channelthermal": {
            "use_local_surrogate": False,
            "internal_prediction_mode": "global_head",
            "default_num_interface_points": 8,
        },
    })


def _inputs(device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    structure = {
        "re": torch.tensor([[100.0]], device=device),
        "u_in": torch.tensor([[2.0]], device=device),
        "module_centers": torch.tensor([[[2.0, 2.0], [8.0, 4.0], [0.0, 0.0]]], device=device),
        "heat_powers": torch.tensor([[1.0, 2.0, 0.0]], device=device),
        "module_present": torch.tensor([[1.0, 1.0, 0.0]], device=device),
        "material_params": torch.tensor([[1.0e-3, 0.1, 0.01, 10.0, 1.0, 0.45]], device=device),
        "domain_length_x": torch.tensor([[12.0]], device=device),
        "domain_length_y": torch.tensor([[6.0]], device=device),
    }
    query = torch.tensor([[[0.5, 0.5], [4.0, 3.0], [11.0, 5.5]]], device=device)
    return structure, query


def test_physical_wrapper_anchor_universe_full_access_and_design_gradient() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "this integration test is assigned to physical GPU 2"
    torch.manual_seed(523)
    parent = ChannelThermalHONFModel(
        _config("three_term_full_access_honf"), attach_local_from_checkpoint=False
    ).to(device).eval()
    candidate = ChannelThermalHONFModel(
        _config("adaptive_interaction_cover_honf"), attach_local_from_checkpoint=False
    ).to(device).eval()
    structure, query = _inputs(device)
    with torch.no_grad():
        parent(structure, query)
        candidate(structure, query)
    candidate.load_state_dict(parent.state_dict(), strict=True)
    with torch.no_grad():
        reference = parent(structure, query)
        adaptive = candidate(structure, query, return_prepared_state=True)
    torch.testing.assert_close(adaptive["pred_field"], reference["pred_field"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        adaptive["pred_port_condition"], reference["pred_port_condition"], rtol=0.0, atol=0.0
    )
    candidate.core.backend.set_cover_mode("external")
    with torch.no_grad():
        externally_compiled_full = candidate(structure, query)
    torch.testing.assert_close(
        externally_compiled_full["pred_field"], reference["pred_field"], rtol=1.0e-5, atol=1.0e-6
    )
    torch.testing.assert_close(
        externally_compiled_full["pred_port_condition"],
        reference["pred_port_condition"],
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    prepared = adaptive["prepared_state"].prepared
    anchors = prepared.encoded.receiver_anchor_coords
    weights = prepared.encoded.receiver_anchor_weights
    roles = prepared.encoded.receiver_anchor_roles
    assert anchors is not None and weights is not None and roles is not None
    assert {0, 1, 2, 3, 4}.issubset(set(roles.flatten().tolist()))
    assert int((weights == 0).sum()) > 0  # padded module/port anchors
    assert len(prepared.backend_state["cover_trees"][0].universe.coordinates) == int((weights > 0).sum())

    centers = structure["module_centers"].clone().requires_grad_(True)
    current = {**structure, "module_centers": centers}
    prediction = candidate(current, query)["pred_field"]
    derivative = torch.autograd.grad(prediction.square().sum(), centers)[0]
    assert torch.isfinite(derivative).all()
    assert float(derivative[:, :2].abs().sum()) > 0.0
