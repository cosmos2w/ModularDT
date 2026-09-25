"""CPU integration checks for the static Run-1507 specialization."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from channelthermal.training.source_conditioned_conversion import (
    REMOVED_STATE_KEY_SUFFIXES,
    convert_task_trained_checkpoint,
    convert_task_trained_state_dict,
    expected_removed_state_keys,
)

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields import InterfaceFieldCore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_PROFILE = (
    PROJECT_ROOT
    / "src/config_core/forward/source_conditioned_pairwise_honf_context.json"
)
TASK_TRAINED_PROFILE = (
    PROJECT_ROOT
    / "src/config_core/forward/task_trained_functional_coalescence_honf_context.json"
)


def _config_payload(profile: Path, architecture: str) -> dict[str, object]:
    payload = copy.deepcopy(json.loads(profile.read_text(encoding="utf-8"))["model"]["core_honf"])
    payload.update(
        {
            "forward_architecture": architecture,
            "field_dim": 5,
            "hidden_dim": 16,
            "domain_length_x": 12.0,
            "domain_length_y": 6.0,
            "module_radius": 0.45,
            "query_fourier_frequencies": 2,
            "position_fourier_frequencies": 2,
        }
    )
    payload["interface_model"].update(
        {
            "message_hidden_dim": 8,
            "attention_heads": 2,
            "relative_fourier_frequencies": 2,
            "receiver_chunk_size": 8,
            "activation_checkpointing": False,
            "environment_refinement_normalizer": "sparsemax",
        }
    )
    return payload


def _core(architecture: str) -> InterfaceFieldCore:
    profile = (
        TASK_TRAINED_PROFILE
        if architecture == "task_trained_functional_coalescence_honf"
        else STATIC_PROFILE
    )
    return InterfaceFieldCore(
        UnifiedForwardConfig.from_dict(_config_payload(profile, architecture))
    ).cpu().eval()


def _batch(seed: int = 15080001, queries: int = 5) -> BatchData:
    generator = torch.Generator().manual_seed(seed)
    extent = torch.tensor([12.0, 6.0])
    return BatchData(
        module_centers=torch.rand(1, 12, 2, generator=generator) * extent,
        module_present=torch.ones(1, 12),
        module_features=torch.randn(1, 12, 3, generator=generator),
        global_context=torch.randn(1, 4, generator=generator),
        query_xy=torch.rand(1, queries, 2, generator=generator) * extent,
        query_time=None,
        target_field=torch.randn(1, queries, 5, generator=generator),
        case_name="source-conditioned-core-integration",
        metadata={},
        env_coords=torch.rand(1, 9, 2, generator=generator) * extent,
        env_features=torch.randn(1, 9, 2, generator=generator),
        env_weights=torch.rand(1, 9, generator=generator) + 0.5,
    )


def _materialize(core: InterfaceFieldCore, batch: BatchData) -> None:
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens, return_routing_maps=True)
    core.decode_queries(prepared, batch.query_xy, return_routing_maps=True)


def test_static_profile_has_source_capacity_without_query_router_config() -> None:
    profile = json.loads(STATIC_PROFILE.read_text(encoding="utf-8"))
    raw_core = profile["model"]["core_honf"]
    interface = raw_core["interface_model"]
    assert raw_core["forward_architecture"] == "source_conditioned_pairwise_honf"
    assert interface["group_count"] == 12
    assert interface["group_control_dim"] == 16
    assert interface["source_normalizer"] == "entmax15"
    assert interface["environment_refinement_normalizer"] == "sparsemax"
    assert "query_normalizer" not in interface
    assert "query_temperature" not in interface
    assert "functional_tree_subsets" not in interface
    assert "functional_detail_complexity_weight" not in profile.get("training", {})
    assert "id" not in profile.get("run", {})
    assert profile["training"]["epochs"] == 500
    assert profile["training"]["device"] == "cuda:2"

    config = UnifiedForwardConfig.from_dict(raw_core)
    serialized = config.to_dict()
    assert serialized["forward_architecture"] == "source_conditioned_pairwise_honf"
    assert "query_normalizer" not in serialized["interface_model"]
    assert "query_temperature" not in serialized["interface_model"]
    assert serialized["interface_model"]["environment_refinement_normalizer"] == "sparsemax"


def test_run1507_all_closed_conversion_is_strict_and_keeps_physical_query_gradients() -> None:
    torch.manual_seed(15080002)
    historical = _core("task_trained_functional_coalescence_honf")
    historical.set_training_progress(epoch=150, total_epochs=500)
    torch.manual_seed(15080003)
    static = _core("source_conditioned_pairwise_honf")
    batch = _batch()
    _materialize(historical, batch)
    _materialize(static, batch)

    converted, inventory = convert_task_trained_state_dict(
        historical.state_dict(),
        static.state_dict(),
        backend_prefix="backend.",
    )
    assert tuple(inventory["removed"]) == tuple(sorted(expected_removed_state_keys(backend_prefix="backend.")))
    assert len(inventory["removed"]) == 14 == len(REMOVED_STATE_KEY_SUFFIXES)
    with pytest.raises(ValueError, match="removed-state inventory"):
        convert_task_trained_state_dict(
            {**historical.state_dict(), "unexpected_controller_state": torch.zeros(1)},
            static.state_dict(),
            backend_prefix="backend.",
        )
    incompatibility = static.load_state_dict(converted, strict=True)
    assert incompatibility.missing_keys == []
    assert incompatibility.unexpected_keys == []
    assert not hasattr(static.backend.router, "query_projection")
    assert not hasattr(static.backend.router, "query_group_projection")

    with torch.no_grad():
        static.backend.router.group_codes.zero_()
        historical.backend.router.group_codes.zero_()
        historical.backend.router.geometry_fraction = 1.0e6
        static.backend.router.geometry_fraction = 1.0e6
        historical.backend.functional_detail_controller.network[-1].weight.zero_()
        historical.backend.functional_detail_controller.network[-1].bias.fill_(-8.0)

    batch = replace(
        batch,
        module_centers=batch.module_centers.clone().requires_grad_(),
        env_coords=batch.env_coords.clone().requires_grad_(),
        query_xy=batch.query_xy.clone().requires_grad_(),
    )
    historical_encoded = historical.encode_case(batch)
    static_encoded = static.encode_case(batch)
    historical_prepared = historical.prepare(
        historical_encoded, historical_encoded.module_tokens
    )
    static_prepared = static.prepare(static_encoded, static_encoded.module_tokens)
    assert historical_prepared.backend_state["functional_detail_plan"].detail.actual_R.tolist() == [1]
    assert int(static_prepared.backend_state["source_conditioned_controls"].k_active.item()) == 12

    historical_output = historical.decode_queries(historical_prepared, batch.query_xy)["pred_field"]
    static_output = static.decode_queries(static_prepared, batch.query_xy)["pred_field"]
    torch.testing.assert_close(static_output, historical_output, atol=3.0e-6, rtol=3.0e-6)
    gradients = torch.autograd.grad(
        static_output.square().mean(),
        (batch.query_xy, batch.module_centers, batch.env_coords),
        allow_unused=True,
    )
    for gradient in gradients:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_checkpoint_conversion_updates_identity_and_disables_resume_state() -> None:
    torch.manual_seed(15080004)
    historical = _core("task_trained_functional_coalescence_honf")
    historical.set_training_progress(epoch=150, total_epochs=500)
    torch.manual_seed(15080005)
    static = _core("source_conditioned_pairwise_honf")
    batch = _batch(15080006)
    _materialize(historical, batch)
    _materialize(static, batch)

    source_payload = {
        "checkpoint_schema_version": 1,
        "case_id": "ThermalChannel",
        "model_family": "honf_forward",
        "workflow": "forward",
        "model_config": {
            "core_honf": {
                "forward_architecture": "task_trained_functional_coalescence_honf"
            }
        },
        "model_state_dict": historical.state_dict(),
        "optimizer_state_dict": {"stale": True},
        "optimizer_group_inventory": {"stale": True},
        "scaler_state_dict": {"stale": True},
        "dataset_id": "thermal_channel_global_v1",
        "dataset_schema": 7,
        "dataset_fingerprint": "fingerprint-fixture",
        "global_normalization_config": {"normalize_inputs": False, "normalize_targets": True},
        "global_normalization_stats": {"field": [1.0, 2.0]},
        "local_normalization_config": {"normalize_inputs": True, "normalize_targets": False},
        "local_normalization_stats": {"port_tokens_mean": [3.0]},
    }

    class _TargetModel:
        config = SimpleNamespace(
            core_honf=static.config,
            to_dict=lambda: {"core_honf": static.config.to_dict()},
        )

        def state_dict(self):
            return static.state_dict()

        def load_state_dict(self, payload, *, strict):
            return static.load_state_dict(payload, strict=strict)

        def selection_state(self):
            return static.selection_state()

    # The full ChannelThermal model prefixes the reusable core with core.;
    # this focused facade uses the core's own state-dict prefix.
    converted, inventory = convert_task_trained_checkpoint(
        source_payload,
        _TargetModel(),
        backend_prefix="backend.",
    )
    assert converted["architecture_conversion"]["target_architecture"] == (
        "source_conditioned_pairwise_honf"
    )
    assert converted["architecture_conversion"]["removed_state_keys"] == inventory["removed"]
    assert converted["optimizer_state_dict"] is None
    assert converted["optimizer_group_inventory"] is None
    assert converted["scaler_state_dict"] is None
    for key in (
        "dataset_id",
        "dataset_schema",
        "dataset_fingerprint",
        "global_normalization_config",
        "global_normalization_stats",
        "local_normalization_config",
        "local_normalization_stats",
    ):
        assert converted[key] == source_payload[key]


@pytest.mark.parametrize(
    "key,value",
    [
        ("checkpoint_schema_version", 2),
        ("case_id", "OtherCase"),
        ("model_family", "other_family"),
        ("workflow", "local"),
    ],
)
def test_checkpoint_conversion_rejects_wrong_identity(key: str, value: object) -> None:
    torch.manual_seed(15080007)
    historical = _core("task_trained_functional_coalescence_honf")
    historical.set_training_progress(epoch=150, total_epochs=500)
    torch.manual_seed(15080008)
    static = _core("source_conditioned_pairwise_honf")
    batch = _batch(15080009)
    _materialize(historical, batch)
    _materialize(static, batch)
    payload = {
        "checkpoint_schema_version": 1,
        "case_id": "ThermalChannel",
        "model_family": "honf_forward",
        "workflow": "forward",
        "model_config": {
            "core_honf": {
                "forward_architecture": "task_trained_functional_coalescence_honf"
            }
        },
        "model_state_dict": historical.state_dict(),
    }
    payload[key] = value

    class _TargetModel:
        config = SimpleNamespace(
            core_honf=static.config,
            to_dict=lambda: {"core_honf": static.config.to_dict()},
        )

        def state_dict(self):
            return static.state_dict()

        def load_state_dict(self, state, *, strict):
            return static.load_state_dict(state, strict=strict)

        def selection_state(self):
            return static.selection_state()

    with pytest.raises(ValueError):
        convert_task_trained_checkpoint(payload, _TargetModel(), backend_prefix="backend.")
