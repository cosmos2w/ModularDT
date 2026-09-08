"""Focused tests for the matched Stage-1 interface-field adaptations."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.evaluation import loading as evaluation_loading
from channelthermal.model import ChannelThermalHONFModel


def _model(architecture: str) -> ChannelThermalHONFModel:
    interface_model = {
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
    }
    if architecture == "sparse_interface_honf":
        interface_model["support_spacing_factor"] = 4.0
    config = ChannelThermalHONFConfig.from_dict(
        {
            "core_honf": {
                "forward_architecture": architecture,
                "interface_model": interface_model,
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


@pytest.mark.parametrize(
    "architecture", ["dense_pairwise_field", "geometry_latent_field", "sparse_interface_honf"]
)
def test_interface_fields_are_finite_chunk_independent_and_coordinate_differentiable(architecture: str) -> None:
    model = _model(architecture)
    inputs = _inputs()
    output = model(**inputs, return_prepared_state=True, return_routing_maps=True)
    assert torch.isfinite(output["pred_field"]).all()
    assert output["organizer_aux"] == {}
    assert "local_neighbor_count" in output["interaction_aux"]
    routing_key = {
        "dense_pairwise_field": "dense_environment_attention",
        "geometry_latent_field": "latent_query_attention",
        "sparse_interface_honf": "group_read_group_index",
    }[architecture]
    assert routing_key in output["routing_aux"]
    if architecture == "sparse_interface_honf":
        assert output["routing_aux"][routing_key].shape == (2, 7, 16)
    else:
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


@pytest.mark.parametrize(
    "architecture", ["dense_pairwise_field", "geometry_latent_field", "sparse_interface_honf"]
)
def test_routing_maps_are_opt_in_and_inference_chunk_override_preserves_outputs(architecture: str) -> None:
    model = _model(architecture)
    inputs = _inputs()
    with torch.no_grad():
        summary_output = model(**inputs, return_prepared_state=True, return_routing_maps=False)
        detailed_output = model(**inputs, return_prepared_state=True, return_routing_maps=True)

    routing_key = {
        "dense_pairwise_field": "dense_environment_attention",
        "geometry_latent_field": "latent_query_attention",
        "sparse_interface_honf": "group_read_group_index",
    }[architecture]
    assert routing_key not in summary_output["routing_aux"]
    assert not any(routing_key in key for key in summary_output["interaction_aux"])
    torch.testing.assert_close(
        summary_output["pred_field"],
        detailed_output["pred_field"],
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    prepared = summary_output["prepared_state"]
    if architecture == "geometry_latent_field":
        assert "module_attention" not in prepared.prepared.backend_state
        assert "environment_attention" not in prepared.prepared.backend_state
        detailed_backend_state = detailed_output["prepared_state"].prepared.backend_state
        assert "module_attention" in detailed_backend_state
        assert "environment_attention" in detailed_backend_state

    with torch.no_grad():
        configured = model.decode_prepared(prepared, inputs["query_xy"])
        large_chunk = model.decode_prepared(
            prepared,
            inputs["query_xy"],
            receiver_chunk_size=2048,
        )
    assert model.core.receiver_chunk_size == 3
    torch.testing.assert_close(
        configured["pred_field"],
        large_chunk["pred_field"],
        rtol=2.0e-5,
        atol=2.0e-6,
    )

    configured_coordinates = inputs["query_xy"].clone().requires_grad_(True)
    large_chunk_coordinates = inputs["query_xy"].clone().requires_grad_(True)
    configured_values = model.decode_prepared(
        prepared,
        configured_coordinates,
        return_routing_maps=False,
    )["pred_field"]
    large_chunk_values = model.decode_prepared(
        prepared,
        large_chunk_coordinates,
        receiver_chunk_size=2048,
        return_routing_maps=False,
    )["pred_field"]
    configured_gradient = torch.autograd.grad(configured_values.sum(), configured_coordinates)[0]
    large_chunk_gradient = torch.autograd.grad(large_chunk_values.sum(), large_chunk_coordinates)[0]
    torch.testing.assert_close(configured_values, large_chunk_values, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(configured_gradient, large_chunk_gradient, rtol=3.0e-5, atol=3.0e-6)
    assert torch.isfinite(configured_gradient).all()
    assert torch.isfinite(large_chunk_gradient).all()

    summary_coordinates = inputs["query_xy"].clone().requires_grad_(True)
    detailed_coordinates = inputs["query_xy"].clone().requires_grad_(True)
    summary_values = model.decode_prepared(
        prepared,
        summary_coordinates,
        receiver_chunk_size=2048,
        return_routing_maps=False,
    )["pred_field"]
    detailed_values = model.decode_prepared(
        detailed_output["prepared_state"],
        detailed_coordinates,
        receiver_chunk_size=2048,
        return_routing_maps=True,
    )["pred_field"]
    summary_gradient = torch.autograd.grad(summary_values.sum(), summary_coordinates)[0]
    detailed_gradient = torch.autograd.grad(detailed_values.sum(), detailed_coordinates)[0]
    torch.testing.assert_close(summary_values, detailed_values, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(summary_gradient, detailed_gradient, rtol=3.0e-5, atol=3.0e-6)
    assert torch.isfinite(summary_gradient).all()
    assert torch.isfinite(detailed_gradient).all()


@pytest.mark.parametrize(
    "architecture", ["dense_pairwise_field", "geometry_latent_field", "sparse_interface_honf"]
)
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


def test_sparse_environment_work_is_cached_while_group_states_refresh() -> None:
    model = _model("sparse_interface_honf")
    output = model(**_inputs(), return_prepared_state=True)
    prepared = output["prepared_state"].prepared
    cache = prepared.backend_state.cache
    cache.environment = None
    calls = {"membership": 0, "message": 0}

    def count_membership(_module, _inputs, _output) -> None:
        calls["membership"] += 1

    def count_message(_module, _inputs, _output) -> None:
        calls["message"] += 1

    membership_hook = model.core.backend.environment_membership.register_forward_hook(count_membership)
    message_hook = model.core.backend.environment_message.register_forward_hook(count_message)
    try:
        states = []
        for offset in (0.0, 0.1, 0.2):
            state = model.core.prepare(
                prepared.encoded,
                prepared.encoded.module_tokens + offset * prepared.encoded.module_present[..., None],
                layout_cache=cache,
            ).backend_state
            states.append(state.group_state)
    finally:
        membership_hook.remove()
        message_hook.remove()

    assert calls == {"membership": 1, "message": 1}
    assert cache.environment is not None
    assert not torch.equal(states[0], states[1])
    assert not torch.equal(states[1], states[2])


def test_sparse_interface_checkpoint_reconstructs_strictly_with_bit_exact_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _model("sparse_interface_honf")
    inputs = _inputs()
    with torch.no_grad():
        expected = source(**inputs)
    checkpoint = {
        "checkpoint_schema_version": 1,
        "case_id": "ThermalChannel",
        "model_family": "honf_forward",
        "workflow": "forward",
        "epoch": 7,
        "model_config": source.config.to_dict(),
        "model_state_dict": copy.deepcopy(source.state_dict()),
        "train_config": {"training": {"epochs": 500}},
    }
    monkeypatch.setattr(
        evaluation_loading,
        "load_trusted_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )

    restored, loaded = evaluation_loading.load_model(Path("unused.pt"), torch.device("cpu"))
    with torch.no_grad():
        actual = restored(**inputs)

    assert loaded is checkpoint
    assert restored.config.core_honf.forward_architecture == "sparse_interface_honf"
    assert restored.config.core_honf.interface_model is not None
    assert restored.config.core_honf.interface_model.support_spacing_factor == pytest.approx(4.0)
    assert tuple(restored.state_dict()) == tuple(source.state_dict())
    torch.testing.assert_close(actual["pred_field"], expected["pred_field"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual["pred_port_condition"], expected["pred_port_condition"], rtol=0.0, atol=0.0
    )
