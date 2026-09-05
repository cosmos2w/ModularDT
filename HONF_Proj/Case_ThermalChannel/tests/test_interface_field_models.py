"""Focused tests for the matched Stage-1 interface-field adaptations."""

from __future__ import annotations

import pytest
import torch

from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.model import ChannelThermalHONFModel


def _model(architecture: str) -> ChannelThermalHONFModel:
    config = ChannelThermalHONFConfig.from_dict(
        {
            "core_honf": {
                "forward_architecture": architecture,
                "interface_model": {
                    "message_hidden_dim": 24,
                    "attention_heads": 4,
                    "coarse_latent_count": 4,
                    "coarse_blocks": 1,
                    "main_latent_count": 4,
                    "main_latent_blocks": 2,
                    "local_radius_factor": 2.5,
                    "relative_fourier_frequencies": 2,
                    "receiver_chunk_size": 3,
                    "activation_checkpointing": False,
                },
                "hidden_dim": 32,
                "field_dim": 5,
                "domain_length_x": 12.0,
                "domain_length_y": 4.0,
                "coordinate_scale": [12.0, 4.0],
                "module_radius": 0.45,
                "num_env_tokens_x": 4,
                "num_env_tokens_y": 3,
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
        }
    )
    return ChannelThermalHONFModel(config, attach_local_from_checkpoint=False).eval()


def _inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(4)
    return {
        "query_xy": torch.rand(2, 7, 2) * torch.tensor([12.0, 4.0]),
        "re": torch.tensor([[100.0], [120.0]]),
        "u_in": torch.tensor([[1.0], [1.2]]),
        "module_centers": torch.tensor(
            [[[2.0, 1.0], [5.0, 2.0], [9.0, 3.0]], [[3.0, 1.0], [7.0, 2.5], [0.0, 0.0]]]
        ),
        "heat_powers": torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]]),
        "module_present": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
        "material_params": torch.tensor(
            [[0.01, 0.02, 0.03, 1.0, 0.5, 0.45], [0.01, 0.02, 0.03, 1.0, 0.5, 0.45]]
        ),
    }


@pytest.mark.parametrize("architecture", ["dense_pairwise_field", "geometry_latent_field"])
def test_interface_fields_are_finite_chunk_independent_and_coordinate_differentiable(architecture: str) -> None:
    model = _model(architecture)
    inputs = _inputs()
    output = model(**inputs, return_prepared_state=True, return_routing_maps=True)
    assert torch.isfinite(output["pred_field"]).all()
    assert output["organizer_aux"] == {}
    assert "local_neighbor_count" in output["interaction_aux"]
    routing_key = (
        "dense_environment_attention"
        if architecture == "dense_pairwise_field"
        else "latent_query_attention"
    )
    assert routing_key in output["routing_aux"]
    assert output["routing_aux"][routing_key].shape[:3] == (2, 4, 7)
    prepared = output["prepared_state"]
    whole = model.decode_prepared(prepared, inputs["query_xy"])["pred_field"]
    split = torch.cat(
        [
            model.decode_prepared(prepared, inputs["query_xy"][:, :2])["pred_field"],
            model.decode_prepared(prepared, inputs["query_xy"][:, 2:])["pred_field"],
        ],
        dim=1,
    )
    torch.testing.assert_close(whole, split, rtol=2.0e-5, atol=2.0e-6)
    coordinates = inputs["query_xy"].clone().requires_grad_(True)
    values = model.decode_prepared(prepared, coordinates)["pred_field"]
    gradient = torch.autograd.grad(values.sum(), coordinates)[0]
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("architecture", ["dense_pairwise_field", "geometry_latent_field"])
def test_interface_fields_preserve_joint_module_permutation(architecture: str) -> None:
    model = _model(architecture)
    inputs = _inputs()
    original = model(**inputs)
    permutation = torch.tensor([2, 0, 1])
    permuted_inputs = dict(inputs)
    for key in ("module_centers", "heat_powers", "module_present"):
        permuted_inputs[key] = inputs[key].index_select(1, permutation)
    permuted = model(**permuted_inputs)
    torch.testing.assert_close(original["pred_field"], permuted["pred_field"], rtol=3.0e-5, atol=3.0e-6)
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(
        original["pred_port_condition"],
        permuted["pred_port_condition"].index_select(1, inverse),
        rtol=3.0e-5,
        atol=3.0e-6,
    )


def test_new_family_config_does_not_serialize_legacy_organizer_settings() -> None:
    config = _model("dense_pairwise_field").config.core_honf.to_dict()
    assert config["forward_architecture"] == "dense_pairwise_field"
    assert "interface_model" in config
    assert "organizer_mode" not in config
    assert "decoder_mode" not in config
