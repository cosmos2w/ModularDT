"""Focused Run-1405 configuration and facade compatibility checks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from honf_forward_core.config import BatchData, InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields.core import InterfaceFieldCore, _merge_fixed_group_maps
from honf_runtime.config_loader import load_config_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_URI = "project://src/config_core/forward/fixed_group_pairwise_honf_context.json"


def _fixed_payload(**overrides: object) -> dict[str, object]:
    interface = {
        "message_hidden_dim": 16,
        "attention_heads": 4,
        "relative_fourier_frequencies": 2,
        "receiver_chunk_size": 8,
        "group_count": 6,
        "source_normalizer": "entmax15",
        "query_normalizer": "entmax15",
        "module_temperature": 1.0,
        "environment_temperature": 1.0,
        "query_temperature": 1.0,
        "group_code_dim": 32,
    }
    interface.update(overrides.pop("interface_model", {}))  # type: ignore[arg-type]
    return {
        "forward_architecture": "fixed_group_pairwise_honf",
        "field_dim": 3,
        "hidden_dim": 16,
        "coordinate_scale": [8.0, 4.0],
        "boundary_feature_mode": "none",
        "interface_model": interface,
        **overrides,
    }


def _fixed_core(*, field_dim: int = 5, hidden_dim: int = 16) -> UnifiedForwardConfig:
    payload = _fixed_payload(field_dim=field_dim, hidden_dim=hidden_dim)
    return UnifiedForwardConfig.from_dict(payload)


def _batch(*, seed: int = 41, query_count: int = 9) -> BatchData:
    generator = torch.Generator().manual_seed(seed)
    extent = torch.tensor([8.0, 4.0])
    return BatchData(
        module_centers=torch.rand(2, 4, 2, generator=generator) * extent,
        module_present=torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
        module_features=torch.randn(2, 4, 3, generator=generator),
        global_context=torch.randn(2, 5, generator=generator),
        query_xy=torch.rand(2, query_count, 2, generator=generator) * extent,
        query_time=None,
        target_field=torch.randn(2, query_count, 5, generator=generator),
        case_name="run1405-fixed-group-test",
        metadata={},
        env_coords=torch.rand(2, 8, 2, generator=generator) * extent,
        env_features=torch.randn(2, 8, 2, generator=generator),
        env_weights=torch.rand(2, 8, generator=generator) + 0.2,
    )


def test_run1405_profile_is_registered_and_serializes_compact_fixed_group_settings() -> None:
    payload = json.loads(
        (PROJECT_ROOT / "src/config_core/forward/fixed_group_pairwise_honf_context.json").read_text(
            encoding="utf-8"
        )
    )
    bundle = load_config_bundle(PROFILE_URI)
    core = bundle.effective["model"]["core_honf"]
    interface = core["interface_model"]

    assert payload["profile_name"] == "fixed_group_pairwise_honf_context"
    assert payload["run"] == {
        "id": "1405",
        "name": "fixed_group_conditioned_interaction",
        "output_root": "project://Trained_Results",
    }
    assert core["forward_architecture"] == "fixed_group_pairwise_honf"
    assert interface == {
        "message_hidden_dim": 128,
        "attention_heads": 4,
        "relative_fourier_frequencies": 4,
        "receiver_chunk_size": 128,
        "activation_checkpointing": True,
        "group_count": 6,
        "source_normalizer": "entmax15",
        "query_normalizer": "entmax15",
        "module_temperature": 1.0,
        "environment_temperature": 1.0,
        "query_temperature": 1.0,
        "group_code_dim": 32,
    }

    registry = json.loads(
        (PROJECT_ROOT / "src/config_core/forward/profile_registry.json").read_text(encoding="utf-8")
    )
    entry = next(item for item in registry["profiles"] if item["name"] == payload["profile_name"])
    assert entry["status"] == "candidate"
    assert entry["base"] is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("group_count", 5, "exactly 6"),
        ("source_normalizer", "softmax", "source_normalizer"),
        ("query_normalizer", "softmax", "query_normalizer"),
        ("module_temperature", 0.5, "module_temperature"),
        ("environment_temperature", 0.5, "environment_temperature"),
        ("query_temperature", 0.5, "query_temperature"),
        ("group_code_dim", 16, "exactly 32"),
    ],
)
def test_run1405_fixed_group_contract_is_exact(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        UnifiedForwardConfig.from_dict(_fixed_payload(interface_model={field: value}))


def test_historical_interface_configs_do_not_gain_fixed_group_serialized_keys() -> None:
    historical = UnifiedForwardConfig.from_dict(
        {
            "forward_architecture": "dense_pairwise_field",
            "hidden_dim": 16,
            "interface_model": {
                "message_hidden_dim": 16,
                "attention_heads": 4,
                "relative_fourier_frequencies": 2,
                "receiver_chunk_size": 8,
            },
        }
    )
    serialized = historical.to_dict()
    assert serialized["forward_architecture"] == "dense_pairwise_field"
    assert not {
        "group_count",
        "source_normalizer",
        "query_normalizer",
        "module_temperature",
        "environment_temperature",
        "query_temperature",
        "group_code_dim",
    } & set(serialized["interface_model"])

    # Positional construction remains valid after appending the new fields.
    assert InterfaceFieldConfig(
        16,
        4,
        2,
        1,
        16,
        2,
        2.5,
        None,
        "null_softmax",
        2,
        8,
        False,
        [2, 2],
        [1.0, 2.0],
        "module_states",
        None,
    ).group_count == 6


def test_fixed_group_core_uses_three_terms_and_exposes_exact_route_supports() -> None:
    torch.manual_seed(17)
    core = InterfaceFieldCore(_fixed_core()).eval()
    batch = _batch()
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens, return_routing_maps=True)

    assert core.common.__class__.__name__ == "ThreeTermInterfaceContext"
    assert not hasattr(core.common, "coarse_seeds")
    assert not hasattr(core.common, "local_message")
    assert prepared.interaction_aux["coarse_latent_count"] == 0
    state = prepared.backend_state["router_state"]
    assert state.module_membership.shape == (2, 4, 6)
    assert state.environment_membership.shape == (2, 8, 6)
    assert torch.isfinite(state.group_state).all()
    assert torch.all(state.module_membership >= 0)
    assert torch.all(state.environment_membership >= 0)
    torch.testing.assert_close(
        state.module_membership.sum(dim=-1),
        batch.module_present,
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        state.environment_membership.sum(dim=-1),
        torch.ones_like(state.environment_membership[..., 0]),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert torch.count_nonzero(state.module_membership == 0) > 0
    assert "fixed_group_module_incidence" in prepared.interaction_aux
    assert "fixed_group_environment_centres" in prepared.interaction_aux

    read = core.read(prepared, batch.query_xy, return_routing_maps=True)
    assert torch.isfinite(read.context).all()
    torch.testing.assert_close(
        read.interaction_aux["local_neighbor_count"],
        torch.zeros_like(read.interaction_aux["local_neighbor_count"]),
        rtol=0.0,
        atol=0.0,
    )
    route = read.interaction_aux["fixed_group_query_assignment"]
    torch.testing.assert_close(route.sum(dim=-1), torch.ones_like(route[..., 0]), atol=1.0e-6, rtol=0.0)

    first = core.decode_queries(
        prepared,
        batch.query_xy,
        query_features=torch.zeros(2, batch.query_xy.shape[1], 3),
    )["pred_field"]
    second = core.decode_queries(
        prepared,
        batch.query_xy,
        query_features=torch.randn(2, batch.query_xy.shape[1], 3),
    )["pred_field"]
    # ThreeTermInterfaceContext intentionally has no query-feature bypass.
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    loss = first.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in core.backend.parameters()
        if parameter.requires_grad
    )


def test_fixed_group_core_is_invariant_to_module_and_environment_permutations() -> None:
    torch.manual_seed(19)
    core = InterfaceFieldCore(_fixed_core()).eval()
    batch = _batch(seed=53)

    def predict(case: BatchData) -> torch.Tensor:
        encoded = core.encode_case(case)
        prepared = core.prepare(encoded, encoded.module_tokens)
        return core.decode_queries(prepared, case.query_xy)["pred_field"]

    with torch.no_grad():
        reference = predict(batch)
        module_order = torch.tensor([3, 1, 0, 2])
        module_permuted = replace(
            batch,
            module_centers=batch.module_centers[:, module_order],
            module_present=batch.module_present[:, module_order],
            module_features=batch.module_features[:, module_order],
        )
        environment_order = torch.tensor([7, 2, 0, 6, 4, 1, 5, 3])
        environment_permuted = replace(
            batch,
            env_coords=batch.env_coords[:, environment_order],
            env_features=batch.env_features[:, environment_order],
            env_weights=batch.env_weights[:, environment_order],
        )
        torch.testing.assert_close(predict(module_permuted), reference, rtol=2.0e-5, atol=2.0e-6)
        torch.testing.assert_close(predict(environment_permuted), reference, rtol=2.0e-5, atol=2.0e-6)


def test_fixed_group_chunk_merge_preserves_dense_attention_and_ragged_triples() -> None:
    first = {
        "fixed_group_environment_attention": torch.zeros(1, 6, 4, 2, 8),
        "fixed_group_environment_attention_batch": torch.tensor([0, 0]),
        "fixed_group_environment_attention_query": torch.tensor([0, 1]),
        "fixed_group_environment_attention_group": torch.tensor([1, 2]),
        "fixed_group_environment_attention_source": torch.tensor([3, 4]),
        "fixed_group_query_assignment": torch.zeros(1, 2, 6),
        "fixed_group_module_triple_query": torch.tensor([0, 1]),
        "fixed_group_module_triple_batch": torch.tensor([0, 0]),
        "fixed_group_module_triple_group": torch.tensor([1, 2]),
        "fixed_group_module_triple_source": torch.tensor([3, 4]),
    }
    second = {
        "fixed_group_environment_attention": torch.ones(1, 6, 4, 3, 8),
        "fixed_group_environment_attention_batch": torch.tensor([0, 0]),
        "fixed_group_environment_attention_query": torch.tensor([0, 2]),
        "fixed_group_environment_attention_group": torch.tensor([0, 5]),
        "fixed_group_environment_attention_source": torch.tensor([1, 2]),
        "fixed_group_query_assignment": torch.ones(1, 3, 6),
        "fixed_group_module_triple_query": torch.tensor([0, 2]),
        "fixed_group_module_triple_batch": torch.tensor([0, 0]),
        "fixed_group_module_triple_group": torch.tensor([0, 5]),
        "fixed_group_module_triple_source": torch.tensor([1, 2]),
    }
    merged = _merge_fixed_group_maps([(first, 2), (second, 3)])
    assert merged["fixed_group_environment_attention"].shape == (1, 6, 4, 5, 8)
    assert merged["fixed_group_query_assignment"].shape == (1, 5, 6)
    torch.testing.assert_close(
        merged["fixed_group_environment_attention_query"],
        torch.tensor([0, 1, 2, 4]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        merged["fixed_group_environment_attention_source"],
        torch.tensor([3, 4, 1, 2]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        merged["fixed_group_module_triple_query"],
        torch.tensor([0, 1, 2, 4]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        merged["fixed_group_module_triple_source"],
        torch.tensor([3, 4, 1, 2]),
        rtol=0.0,
        atol=0.0,
    )


def test_fixed_group_chunked_read_keeps_full_q8192_routing_evidence() -> None:
    torch.manual_seed(1405)
    core = InterfaceFieldCore(_fixed_core(field_dim=3, hidden_dim=8)).eval()
    encoded = core.encode_case(_batch(query_count=9))
    prepared = core.prepare(encoded, encoded.module_tokens, return_routing_maps=True)
    read = core.read(
        prepared,
        torch.rand(2, 8192, 2),
        receiver_chunk_size=2048,
        return_routing_maps=True,
    )
    assert read.context.shape == (2, 8192, 8)
    assignment = read.interaction_aux["fixed_group_query_assignment"]
    assert assignment.shape == (2, 8192, 6)
    module_queries = read.interaction_aux["fixed_group_module_triple_query"]
    environment_queries = read.interaction_aux["fixed_group_environment_triple_query"]
    assert int(module_queries.min()) == 0
    assert int(environment_queries.min()) == 0
    assert int(module_queries.max()) == 8191
    assert int(environment_queries.max()) == 8191


def test_fixed_group_uses_ordinary_thermalchannel_path_without_architecture_branch() -> None:
    coupling = PROJECT_ROOT / "Case_ThermalChannel/src/channelthermal/interface_field_coupling.py"
    assert "fixed_group_pairwise_honf" not in coupling.read_text(encoding="utf-8")


@pytest.mark.skipif(
    not (PROJECT_ROOT / "Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt").is_file(),
    reason="the established frozen ThermalChannel local surrogate is unavailable",
)
def test_fixed_group_predicted_port_path_refreshes_p0_p1_p2_backend_on_cpu() -> None:
    from channelthermal.config import ChannelThermalHONFConfig
    from channelthermal.model import ChannelThermalHONFModel

    local_checkpoint = PROJECT_ROOT / "Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt"
    config = ChannelThermalHONFConfig.from_dict(
        {
            "core_honf": {
                **_fixed_payload(field_dim=5, hidden_dim=32),
                "domain_length_x": 12.0,
                "domain_length_y": 4.0,
                "coordinate_scale": [12.0, 6.0],
                "module_radius": 0.45,
                "num_env_tokens_x": 4,
                "num_env_tokens_y": 3,
                "interface_model": {
                    **_fixed_payload(field_dim=5, hidden_dim=32)["interface_model"],
                    "message_hidden_dim": 24,
                    "attention_heads": 4,
                    "receiver_chunk_size": 8,
                },
            },
            "channelthermal": {
                "use_local_surrogate": True,
                "freeze_local_surrogate": True,
                "local_surrogate_checkpoint_path": str(local_checkpoint),
                "internal_prediction_mode": "local_surrogate",
                "interaction_refinement_steps": 1,
                "default_num_interface_points": 8,
            },
        }
    )
    model = ChannelThermalHONFModel(config).eval()
    original_prepare = model.core.backend.prepare
    preparation_roles: list[str | None] = []

    def capture(*args: object, **kwargs: object):
        preparation_roles.append(getattr(model.core, "_interface_read_role", None))
        return original_prepare(*args, **kwargs)

    model.core.backend.prepare = capture  # type: ignore[method-assign]
    output = model(
        query_xy=torch.tensor([[[1.0, 1.0], [6.0, 2.0], [11.0, 3.0]]]),
        re=torch.tensor([[100.0]]),
        u_in=torch.tensor([[1.0]]),
        module_centers=torch.tensor([[[3.0, 1.5], [8.0, 2.5]]]),
        heat_powers=torch.tensor([[1.0, 2.0]]),
        module_present=torch.ones(1, 2),
        material_params=torch.tensor([[0.01, 0.02, 0.03, 1.0, 0.5, 0.45]]),
        local_port_condition_mode="predicted",
        return_prepared_state=True,
    )
    assert preparation_roles == ["p0_port", "p1_refinement", "p2_field"]
    assert torch.isfinite(output["pred_field"]).all()
    assert output["prepared_state"].architecture == "fixed_group_pairwise_honf"
