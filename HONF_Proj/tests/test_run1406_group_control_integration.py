"""Focused Run-1406 configuration and shared-facade compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from honf_forward_core.config import BatchData, InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields import (
    GroupControlPairwiseField,
    InterfaceFieldCore,
    ThreeTermInterfaceContext,
)
from honf_forward_core.interface_fields.core import _merge_group_control_maps
from honf_runtime.config_loader import load_config_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_URI = "project://src/config_core/forward/group_control_pairwise_honf_context.json"


def _payload(**interface_overrides: object) -> dict[str, object]:
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
        "group_control_dim": 16,
    }
    interface.update(interface_overrides)
    return {
        "forward_architecture": "group_control_pairwise_honf",
        "field_dim": 3,
        "hidden_dim": 16,
        "coordinate_scale": [8.0, 4.0],
        "boundary_feature_mode": "none",
        "interface_model": interface,
    }


def _batch(*, seed: int = 1406, query_count: int = 7) -> BatchData:
    generator = torch.Generator().manual_seed(seed)
    extent = torch.tensor([8.0, 4.0])
    return BatchData(
        module_centers=torch.rand(2, 4, 2, generator=generator) * extent,
        module_present=torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
        module_features=torch.randn(2, 4, 3, generator=generator),
        global_context=torch.randn(2, 5, generator=generator),
        query_xy=torch.rand(2, query_count, 2, generator=generator) * extent,
        query_time=None,
        target_field=torch.randn(2, query_count, 3, generator=generator),
        case_name="run1406-group-control-test",
        metadata={},
        env_coords=torch.rand(2, 8, 2, generator=generator) * extent,
        env_features=torch.randn(2, 8, 2, generator=generator),
        env_weights=torch.rand(2, 8, generator=generator) + 0.2,
    )


def test_run1406_profile_is_registered_and_compact() -> None:
    payload = json.loads(
        (PROJECT_ROOT / "src/config_core/forward/group_control_pairwise_honf_context.json").read_text(
            encoding="utf-8"
        )
    )
    bundle = load_config_bundle(PROFILE_URI)
    core_payload = bundle.effective["model"]["core_honf"]
    interface = core_payload["interface_model"]

    assert payload["profile_name"] == "group_control_pairwise_honf_context"
    assert payload["run"] == {
        "id": "1406",
        "name": "low_dimensional_group_control",
        "output_root": "project://Trained_Results",
    }
    assert payload["training"]["epochs"] == 50
    assert payload["checkpointing"]["save_epoch_milestones"] == [10, 50, 100, 250, 500, 2500, 5000]
    assert core_payload["forward_architecture"] == "group_control_pairwise_honf"
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
        "group_control_dim": 16,
    }
    registry = json.loads(
        (PROJECT_ROOT / "src/config_core/forward/profile_registry.json").read_text(encoding="utf-8")
    )
    entry = next(item for item in registry["profiles"] if item["name"] == payload["profile_name"])
    assert entry["status"] == "candidate"
    assert entry["base"] is None


def test_run1406_serialization_keeps_only_new_control_width() -> None:
    config = UnifiedForwardConfig.from_dict(_payload())
    interface = config.to_dict()["interface_model"]
    assert interface["group_count"] == 6
    assert interface["group_control_dim"] == 16
    assert "group_code_dim" not in interface
    assert "coarse_latent_count" not in interface
    assert "local_radius_factor" not in interface

    # K remains an ordinary reusable capacity setting; the formal profile is
    # the place that fixes it at six.
    configurable = UnifiedForwardConfig.from_dict(_payload(group_count=4, group_control_dim=8))
    assert configurable.interface_model.group_count == 4
    assert configurable.interface_model.group_control_dim == 8


@pytest.mark.parametrize("value", [0, -1, True, False])
def test_run1406_control_width_must_be_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="group_control_dim"):
        UnifiedForwardConfig.from_dict(_payload(group_control_dim=value))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_normalizer", "softmax", "source_normalizer"),
        ("query_normalizer", "softmax", "query_normalizer"),
        ("module_temperature", 0.5, "module_temperature"),
        ("environment_temperature", 0.5, "environment_temperature"),
        ("query_temperature", 0.5, "query_temperature"),
    ],
)
def test_run1406_control_router_contract_is_entmax15_at_unit_temperature(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        UnifiedForwardConfig.from_dict(_payload(**{field: value}))


def test_run1406_reuses_verified_two_dimensional_core_path() -> None:
    with pytest.raises(ValueError, match="spatial_dim=3"):
        UnifiedForwardConfig.from_dict(
            {
                **_payload(),
                "spatial_dim": 3,
                "coordinate_scale": [8.0, 4.0, 2.0],
            }
        )


def test_historical_positional_and_serialized_config_remain_unchanged() -> None:
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
    interface = historical.to_dict()["interface_model"]
    assert "group_control_dim" not in interface
    assert "group_code_dim" not in interface
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
    ).group_control_dim == 16


def test_group_control_chunk_reducer_sums_work_counts_and_preserves_query_maps() -> None:
    first = {
        "group_control_module_pair_count_numerator": torch.tensor(12.0),
        "group_control_module_pair_count_denominator": torch.tensor(8.0),
        "group_control_query_routing": torch.zeros(1, 2, 6),
        "group_control_source_control": torch.zeros(1, 6, 16),
        "group_control_module_query": torch.tensor([0, 1]),
    }
    second = {
        "group_control_module_pair_count_numerator": torch.tensor(18.0),
        "group_control_module_pair_count_denominator": torch.tensor(10.0),
        "group_control_query_routing": torch.ones(1, 3, 6),
        "group_control_source_control": torch.ones(1, 6, 16),
        "group_control_module_query": torch.tensor([0, 2]),
    }
    merged = _merge_group_control_maps([(first, 2), (second, 3)])
    assert merged["group_control_module_pair_count_numerator"].item() == 30.0
    assert merged["group_control_module_pair_count_denominator"].item() == 18.0
    assert merged["group_control_query_routing"].shape == (1, 5, 6)
    assert merged["group_control_source_control"].shape == (1, 6, 16)
    torch.testing.assert_close(
        merged["group_control_module_query"],
        torch.tensor([0, 1, 2, 4]),
        rtol=0.0,
        atol=0.0,
    )


def test_group_control_core_uses_three_terms_and_chunked_predicted_queries() -> None:
    torch.manual_seed(1406)
    core = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_payload())).eval()
    batch = _batch()
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens, return_routing_maps=True)

    assert isinstance(core.common, ThreeTermInterfaceContext)
    assert isinstance(core.backend, GroupControlPairwiseField)
    assert prepared.backend_state["group_control_state"].group_control.shape[-1] == 16
    assert prepared.coarse_state.shape == (2, 0, 16)
    assert prepared.interaction_aux["coarse_latent_count"] == 0

    read = core.read(
        prepared,
        batch.query_xy,
        receiver_chunk_size=3,
        return_routing_maps=True,
    )
    assert read.context.shape == (2, batch.query_xy.shape[1], 16)
    assert torch.isfinite(read.context).all()
    torch.testing.assert_close(
        read.interaction_aux["local_neighbor_count"],
        torch.zeros_like(read.interaction_aux["local_neighbor_count"]),
        rtol=0.0,
        atol=0.0,
    )
    assignment = read.interaction_aux["group_control_query_routing"]
    assert assignment.shape == (2, batch.query_xy.shape[1], 6)
    torch.testing.assert_close(
        assignment.sum(dim=-1),
        torch.ones_like(assignment[..., 0]),
        rtol=0.0,
        atol=1.0e-5,
    )
    full_read = core.read(
        prepared,
        batch.query_xy,
        receiver_chunk_size=batch.query_xy.shape[1],
        return_routing_maps=True,
    )
    for prefix in ("group_control_module", "group_control_environment"):
        query = read.interaction_aux[f"{prefix}_pair_query"]
        assert int(query.min()) == 0
        assert int(query.max()) == batch.query_xy.shape[1] - 1
        assert read.interaction_aux[f"{prefix}_pair_batch"].numel() == query.numel()
        for suffix in ("unique_pairs", "logical_paths", "fine_rows"):
            torch.testing.assert_close(
                read.interaction_aux[f"{prefix}_{suffix}"],
                full_read.interaction_aux[f"{prefix}_{suffix}"],
                rtol=0.0,
                atol=0.0,
            )
    expected_module_valid = batch.module_present.sum() * batch.query_xy.shape[1]
    expected_module_padded = batch.module_present.numel() * batch.query_xy.shape[1] - expected_module_valid
    torch.testing.assert_close(
        read.interaction_aux["group_control_module_valid_pair_denominator"],
        expected_module_valid,
    )
    torch.testing.assert_close(
        read.interaction_aux["group_control_module_padded_pair_denominator"],
        expected_module_padded,
    )
    expected_environment_valid = batch.env_weights.numel() * batch.query_xy.shape[1]
    assert read.interaction_aux["group_control_environment_valid_pair_denominator"].item() == expected_environment_valid
    assert read.interaction_aux["group_control_environment_padded_pair_denominator"].item() == 0.0

    zeros = torch.zeros(2, batch.query_xy.shape[1], 3)
    first = core.decode_queries(
        prepared,
        batch.query_xy,
        query_features=zeros,
        receiver_chunk_size=3,
    )["pred_field"]
    second = core.decode_queries(
        prepared,
        batch.query_xy,
        query_features=torch.randn_like(zeros),
        receiver_chunk_size=3,
    )["pred_field"]
    # ThreeTermInterfaceContext intentionally ignores the legacy query bypass.
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert torch.isfinite(first).all()
    loss = first.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in core.backend.parameters()
        if parameter.requires_grad
    )


def test_group_control_uses_ordinary_thermalchannel_interface_path() -> None:
    coupling = PROJECT_ROOT / "Case_ThermalChannel/src/channelthermal/interface_field_coupling.py"
    assert "group_control_pairwise_honf" not in coupling.read_text(encoding="utf-8")


@pytest.mark.skipif(
    not (PROJECT_ROOT / "Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt").is_file(),
    reason="the established frozen ThermalChannel local surrogate is unavailable",
)
def test_group_control_predicted_port_path_refreshes_p0_p1_p2_on_cpu() -> None:
    from channelthermal.config import ChannelThermalHONFConfig
    from channelthermal.model import ChannelThermalHONFModel

    local_checkpoint = PROJECT_ROOT / "Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt"
    core_payload = _payload()
    core_payload.update(
        {
            "field_dim": 5,
            "hidden_dim": 32,
            "domain_length_x": 12.0,
            "domain_length_y": 4.0,
            "coordinate_scale": [12.0, 6.0],
            "module_radius": 0.45,
            "num_env_tokens_x": 4,
            "num_env_tokens_y": 3,
            "interface_model": {
                **core_payload["interface_model"],
                "message_hidden_dim": 24,
                "attention_heads": 4,
                "receiver_chunk_size": 8,
            },
        }
    )
    config = ChannelThermalHONFConfig.from_dict(
        {
            "core_honf": core_payload,
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
        return_routing_maps=True,
        return_port_global_consistency=True,
        return_prepared_state=True,
    )
    assert preparation_roles == ["p0_port", "p1_refinement", "p2_field"]
    assert torch.isfinite(output["pred_field"]).all()
    assert output["prepared_state"].architecture == "group_control_pairwise_honf"
    aux = output["interaction_aux"]
    for prefix in ("initial_port_", "provisional_", "", "port_global_"):
        assert f"{prefix}group_control_module_logical_paths" in aux
        assert f"{prefix}group_control_environment_logical_paths" in aux
    assert "initial_port_group_control_module_source_projection_rows" in aux
    assert "provisional_group_control_module_source_projection_rows" in aux
    assert "group_control_module_source_projection_rows" in aux
    # The consistency read reuses the already prepared P2 state.
    assert "port_global_group_control_module_source_projection_rows" not in aux
