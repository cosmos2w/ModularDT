"""Focused CPU contracts for the Run-1407 phase-shared controller."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields import (
    InterfaceFieldCore,
    PhaseSharedGroupControl,
    PhaseSharedGroupControlPairwiseField,
    PrototypeAnchoredGroupRouter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _payload(**interface_overrides: object) -> dict[str, object]:
    interface = {
        "message_hidden_dim": 12,
        "attention_heads": 2,
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
        "forward_architecture": "phase_shared_group_control_honf",
        "field_dim": 3,
        "hidden_dim": 8,
        "coordinate_scale": [8.0, 4.0],
        "boundary_feature_mode": "none",
        "interface_model": interface,
    }


def _batch(*, seed: int = 1407, query_count: int = 7) -> BatchData:
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
        case_name="run1407-phase-shared-test",
        metadata={},
        env_coords=torch.rand(2, 8, 2, generator=generator) * extent,
        env_features=torch.randn(2, 8, 2, generator=generator),
        env_weights=torch.rand(2, 8, generator=generator) + 0.2,
    )


def test_run1407_profile_and_fixed_controller_contract() -> None:
    payload = json.loads(
        (PROJECT_ROOT / "src/config_core/forward/phase_shared_group_control_honf.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["run"] == {
        "id": "1407",
        "name": "phase_shared_prototype_group_control",
        "output_root": "project://Trained_Results",
    }
    config = UnifiedForwardConfig.from_dict(_payload())
    assert config.forward_architecture == "phase_shared_group_control_honf"
    assert config.interface_model.group_count == 6
    assert config.interface_model.group_control_dim == 16
    assert config.to_dict()["interface_model"]["group_control_dim"] == 16


def test_phase_shared_controller_reuses_p0_banks_and_refreshes_fine_values() -> None:
    torch.manual_seed(1407001)
    config = UnifiedForwardConfig.from_dict(_payload())
    core = InterfaceFieldCore(config).eval()
    batch = _batch()
    encoded = core.encode_case(batch)
    p0 = core.prepare(encoded, encoded.module_tokens, return_routing_maps=True)
    shared = p0.phase_shared_state
    assert isinstance(shared, PhaseSharedGroupControl)
    assert isinstance(core.backend, PhaseSharedGroupControlPairwiseField)
    assert isinstance(core.backend.router, PrototypeAnchoredGroupRouter)
    assert core.backend.router.query_group_projection is not None

    p1 = core.prepare(
        encoded,
        encoded.module_tokens + 0.125,
        phase_shared_state=shared,
        return_routing_maps=True,
    )
    assert p1.phase_shared_state is shared
    assert p1.backend_state["phase_shared_group_control"] is shared
    assert p1.backend_state["group_control_state"] is shared.controls
    assert p1.backend_state["module_control_bank"] is shared.module_control_bank
    assert p1.backend_state["environment_head_source_control"] is shared.environment_head_source_control
    assert p0.backend_state["environment_value_gain"] is shared.environment_value_gain
    assert p1.backend_state["environment_value_gain"] is shared.environment_value_gain
    route0 = core.backend._route(p0.backend_state, encoded, batch.query_xy)
    route1 = core.backend._route(p1.backend_state, encoded, batch.query_xy)
    assert route0.query_keys is shared.normalized_query_keys
    assert route1.query_keys is shared.normalized_query_keys
    assert not torch.equal(
        p0.backend_state["module_first_affine"],
        p1.backend_state["module_first_affine"],
    )
    assert not torch.equal(
        p0.backend_state["environment_keys"],
        p1.backend_state["environment_keys"],
    )
    assert p1.interaction_aux["phase_shared_group_control_controller_reused"].eq(1).all()


def test_prototype_plus_case_keys_use_parameter_free_rms_before_entmax() -> None:
    torch.manual_seed(1407002)
    config = UnifiedForwardConfig.from_dict(_payload())
    core = InterfaceFieldCore(config).eval()
    batch = _batch(query_count=5)
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens)
    router = core.backend.router
    controls = prepared.backend_state["group_control_state"]
    receivers = batch.query_xy
    features = router.query_fourier(receivers / encoded.coordinate_scale)
    route = router.route_queries(encoded, controls, receivers, features)
    expected_keys = router.query_key_bank(controls)
    normalized_keys = expected_keys / torch.sqrt(
        expected_keys.square().mean(dim=-1, keepdim=True) + 1.0e-6
    )
    torch.testing.assert_close(route.query_keys, normalized_keys, rtol=0.0, atol=0.0)
    query = router.query_projection(
        torch.cat(
            [
                features,
                controls.global_control[:, None, :].expand(-1, receivers.shape[1], -1),
            ],
            dim=-1,
        )
    )
    normalized_query = query / torch.sqrt(
        query.square().mean(dim=-1, keepdim=True) + 1.0e-6
    )
    expected_logits = torch.einsum(
        "bqd,bkd->bqk", normalized_query, normalized_keys
    ) / (16.0**0.5)
    torch.testing.assert_close(route.logits, expected_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(route.assignment.sum(dim=-1), torch.ones_like(route.assignment[..., 0]), atol=1.0e-5, rtol=0.0)


def test_prototype_key_rms_scaling_is_finite_for_zero_and_near_zero_rows() -> None:
    zero = torch.zeros(2, 16)
    near = torch.full((2, 16), 1.0e-12)
    zero_scaled = PrototypeAnchoredGroupRouter._rms_scale(zero)
    near_scaled = PrototypeAnchoredGroupRouter._rms_scale(near)
    assert torch.equal(zero_scaled, zero)
    assert torch.isfinite(near_scaled).all()
    expected = near / torch.sqrt(near.square().mean(dim=-1, keepdim=True) + 1.0e-6)
    torch.testing.assert_close(near_scaled, expected, rtol=0.0, atol=0.0)


def test_phase_shared_rectangular_executor_is_independent_of_diagnostic_flag() -> None:
    torch.manual_seed(1407003)
    config = UnifiedForwardConfig.from_dict(_payload())
    core = InterfaceFieldCore(config).eval()
    batch = _batch(query_count=6)
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens)
    eager = core.read(prepared, batch.query_xy, receiver_chunk_size=6, return_routing_maps=False)
    mapped = core.read(prepared, batch.query_xy, receiver_chunk_size=6, return_routing_maps=True)
    torch.testing.assert_close(mapped.context, eager.context, rtol=3.0e-5, atol=3.0e-6)
    assert mapped.interaction_aux["group_control_module_complete_support"].eq(1).all()
    assert mapped.interaction_aux["group_control_environment_complete_support"].eq(1).all()


def test_phase_shared_controller_remains_live_for_backward() -> None:
    torch.manual_seed(1407004)
    config = UnifiedForwardConfig.from_dict(_payload())
    core = InterfaceFieldCore(config).train()
    batch = _batch(query_count=5)
    encoded = core.encode_case(batch)
    p0 = core.prepare(encoded, encoded.module_tokens)
    p2 = core.prepare(
        encoded,
        encoded.module_tokens + 0.05,
        phase_shared_state=p0.phase_shared_state,
    )
    read = core.read(p2, batch.query_xy, receiver_chunk_size=5).context
    loss = read.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert core.backend.router.group_codes.grad is not None
    assert torch.isfinite(core.backend.router.group_codes.grad).all()
