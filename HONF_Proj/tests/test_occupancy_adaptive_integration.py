"""CPU integration contracts for the opt-in Run-1409 ThermalChannel reader."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields import (
    InterfaceFieldCore,
    OccupancyAdaptiveGroupControlPairwiseField,
    OccupancyGroupPlan,
)
from honf_runtime.config_loader import load_config_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = PROJECT_ROOT / "src/config_core/forward/occupancy_adaptive_group_control_honf_context.json"


def _small_payload() -> dict[str, object]:
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["model"]["core_honf"]
    payload = copy.deepcopy(payload)
    payload.update(
        {
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
            "receiver_chunk_size": 3,
            "activation_checkpointing": False,
        }
    )
    return payload


def _batch(*, seed: int = 1409, query_count: int = 5) -> BatchData:
    generator = torch.Generator().manual_seed(seed)
    extent = torch.tensor([12.0, 6.0])
    return BatchData(
        module_centers=torch.rand(1, 3, 2, generator=generator) * extent,
        module_present=torch.tensor([[1.0, 1.0, 0.0]]),
        module_features=torch.randn(1, 3, 3, generator=generator),
        global_context=torch.randn(1, 4, generator=generator),
        query_xy=torch.rand(1, query_count, 2, generator=generator) * extent,
        query_time=None,
        target_field=torch.randn(1, query_count, 5, generator=generator),
        case_name="run1409-occupancy-test",
        metadata={},
        env_coords=torch.rand(1, 8, 2, generator=generator) * extent,
        env_features=torch.randn(1, 8, 2, generator=generator),
        env_weights=torch.rand(1, 8, generator=generator) + 0.2,
    )


def test_run1409_profile_is_opt_in_and_has_no_budget_or_schedule() -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/occupancy_adaptive_group_control_honf_context.json"
    )
    profile = bundle.effective["model"]["core_honf"]
    config = UnifiedForwardConfig.from_dict(profile)
    serialized = config.to_dict()

    assert profile["forward_architecture"] == "occupancy_adaptive_group_control_honf"
    assert profile["interface_model"]["group_count"] == 12
    assert profile["interface_model"]["group_control_dim"] == 16
    assert config.interface_model.case_group_budget is None
    assert "case_group_budget" not in serialized["interface_model"]
    assert UnifiedForwardConfig.from_dict(serialized).to_dict() == serialized
    assert bundle.effective["training"]["seed"] == 0
    assert bundle.effective["training"]["port_curriculum"]["schedule"] == "none"
    assert bundle.effective["loss"]["case_group_budget_weight"] == 0.0
    assert bundle.effective["loss"]["organizer_regularization"]["enabled"] is False
    assert bundle.effective["run"] == {
        "id": "1409",
        "name": "occupancy_adaptive_geometric_group_control",
        "output_root": "project://Trained_Results",
    }


def test_run1409_core_uses_geometry_plan_and_exact_rectangular_reader() -> None:
    torch.manual_seed(1409001)
    core = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_small_payload())).eval()
    assert isinstance(core.backend, OccupancyAdaptiveGroupControlPairwiseField)
    assert core.backend.executor_policy == "rectangular_reference"
    assert core.backend.ledger_rectangular_rows is True
    assert core.backend.execution_mode == "full_width"

    batch = _batch()
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens, return_routing_maps=True)
    assert isinstance(prepared.phase_shared_state, OccupancyGroupPlan)
    assert prepared.phase_shared_state.Kmax == 12
    assert prepared.interaction_aux["occupancy_group_plan_active_mask"].shape == (1, 12)
    assert prepared.interaction_aux["occupancy_group_plan_prototype_ids"].shape == (1, 12)

    plain = core.read(prepared, batch.query_xy, return_routing_maps=False)
    mapped = core.read(prepared, batch.query_xy, return_routing_maps=True)
    torch.testing.assert_close(mapped.context, plain.context, rtol=3.0e-5, atol=3.0e-6)
    assert mapped.interaction_aux["group_control_module_fine_rows"].item() == mapped.interaction_aux[
        "group_control_module_fine_rows_forward"
    ].item()
    assert mapped.interaction_aux["group_control_environment_fine_rows"].item() == mapped.interaction_aux[
        "group_control_environment_fine_rows_forward"
    ].item()


def test_run1409_phase_plan_and_checkpoint_round_trip() -> None:
    torch.manual_seed(1409002)
    config = UnifiedForwardConfig.from_dict(_small_payload())
    batch = _batch(seed=1409002, query_count=4)
    source = InterfaceFieldCore(config).eval()
    encoded = source.encode_case(batch)
    p0 = source.prepare(encoded, encoded.module_tokens)
    p1 = source.prepare(
        encoded,
        encoded.module_tokens + 0.125,
        phase_shared_state=p0.phase_shared_state,
    )
    assert p1.phase_shared_state is p0.phase_shared_state
    source_output = source.read(p1, batch.query_xy).context
    checkpoint = copy.deepcopy(source.state_dict())

    restored = InterfaceFieldCore(config).eval()
    restored.load_state_dict(checkpoint, strict=True)
    # The P0 phase plan is derived from model outputs, so build it after the
    # checkpoint load instead of retaining the plan made by the random target
    # initialization.
    restored_encoded = restored.encode_case(batch)
    restored_p0 = restored.prepare(restored_encoded, restored_encoded.module_tokens)
    restored_p1 = restored.prepare(
        restored_encoded,
        restored_encoded.module_tokens + 0.125,
        phase_shared_state=restored_p0.phase_shared_state,
    )
    restored_output = restored.read(restored_p1, batch.query_xy).context
    torch.testing.assert_close(restored_output, source_output, rtol=2.0e-5, atol=2.0e-6)


def test_historical_group_control_profile_still_resolves() -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/group_control_pairwise_honf_context.json"
    )
    config = UnifiedForwardConfig.from_dict(bundle.core["model"]["core_honf"])
    assert config.forward_architecture == "group_control_pairwise_honf"
    assert config.to_dict()["forward_architecture"] == "group_control_pairwise_honf"
