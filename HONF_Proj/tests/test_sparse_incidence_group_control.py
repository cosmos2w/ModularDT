"""Focused algebra and integration contracts for Run 1501."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields import (
    InterfaceFieldCore,
    SparseIncidenceGroupControlPairwiseField,
    SparseIncidenceGroupRouter,
)
from honf_forward_core.interface_fields.occupancy_group_router import (
    direct_mask_intersection,
    occupancy_support,
    pack_16bit_mask,
)
from honf_forward_core.interface_fields.routing_index.sparse_projection import (
    masked_sparsemax,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase
from honf_forward_core.routing import entmax15
from honf_runtime.config_loader import load_config_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = (
    PROJECT_ROOT
    / "src/config_core/forward/sparse_incidence_group_control_honf_context.json"
)


def test_sparse_incidence_router_rejects_nonunit_query_temperature() -> None:
    with pytest.raises(ValueError, match="fixed unit temperature"):
        SparseIncidenceGroupRouter(hidden_dim=16, query_temperature=0.5)


def _small_payload() -> dict[str, object]:
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["model"][
        "core_honf"
    ]
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


def _encoded(seed: int = 1501) -> tuple[EncodedInterfaceCase, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    batch, modules, environments, hidden = 2, 5, 13, 16
    extent = torch.tensor([12.0, 6.0])
    module_present = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
    )
    encoded = EncodedInterfaceCase(
        module_tokens=torch.randn(batch, modules, hidden, generator=generator),
        env_tokens=torch.randn(batch, environments, hidden, generator=generator),
        global_token=torch.randn(batch, hidden, generator=generator),
        module_centers=torch.rand(batch, modules, 2, generator=generator) * extent,
        env_coords=torch.rand(batch, environments, 2, generator=generator) * extent,
        module_present=module_present,
        module_features=None,
        env_features=None,
        env_weights=torch.rand(batch, environments, generator=generator) + 0.2,
        coordinate_scale=extent,
    )
    return (
        encoded,
        torch.randn(batch, modules, hidden, generator=generator),
        torch.randn(batch, environments, hidden, generator=generator),
    )


def _batch(seed: int = 1501, queries: int = 7) -> BatchData:
    generator = torch.Generator().manual_seed(seed)
    extent = torch.tensor([12.0, 6.0])
    return BatchData(
        module_centers=torch.rand(2, 5, 2, generator=generator) * extent,
        module_present=torch.tensor(
            [[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
        ),
        module_features=torch.randn(2, 5, 3, generator=generator),
        global_context=torch.randn(2, 4, generator=generator),
        query_xy=torch.rand(2, queries, 2, generator=generator) * extent,
        query_time=None,
        target_field=torch.randn(2, queries, 5, generator=generator),
        case_name="run1501-sparse-incidence-test",
        metadata={},
        env_coords=torch.rand(2, 13, 2, generator=generator) * extent,
        env_features=torch.randn(2, 13, 2, generator=generator),
        env_weights=torch.rand(2, 13, generator=generator) + 0.2,
    )


def test_masked_query_sparsemax_normalizes_zeros_and_gradients() -> None:
    logits = torch.tensor(
        [[0.8, 0.2, -0.4, 1.1], [2.0, -3.0, 0.5, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    valid = torch.tensor(
        [[True, True, False, True], [False, False, True, False]]
    )
    assignment = masked_sparsemax(logits, valid)
    torch.testing.assert_close(
        assignment.sum(dim=-1), torch.ones(2, dtype=torch.float64)
    )
    assert torch.all(assignment.masked_select(~valid) == 0.0)
    assert assignment[1, 2].item() == 1.0
    assert torch.count_nonzero(assignment[0]) < int(valid[0].sum())
    assignment.square().sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[0]) > 0


def test_router_is_phase_local_and_uses_fixed_capacity_without_case_plan() -> None:
    encoded, module_states, environment_states = _encoded()
    router = SparseIncidenceGroupRouter(
        16, group_count=12, control_dim=16, fourier_frequencies=2
    )
    p0 = router.prepare(encoded, module_states, environment_states)
    assert not hasattr(p0, "plan")
    assert p0.module_membership.shape[-1] == 12
    assert p0.environment_membership.shape[-1] == 12
    module_rows = p0.module_membership.sum(dim=-1)
    torch.testing.assert_close(
        module_rows[encoded.module_present > 0.5],
        torch.ones_like(module_rows[encoded.module_present > 0.5]),
        atol=2.0e-5,
        rtol=0.0,
    )
    assert torch.all(module_rows[encoded.module_present <= 0.5] == 0.0)
    torch.testing.assert_close(
        p0.environment_membership.sum(dim=-1),
        torch.ones_like(p0.environment_membership[..., 0]),
        atol=2.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        p0.pi.sum(dim=-1), torch.ones_like(p0.kappa), atol=2.0e-5, rtol=0.0
    )
    torch.testing.assert_close(
        p0.kappa,
        p0.pi.square().sum(dim=-1).reciprocal(),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert torch.all(
        (p0.proposal_module_mass + p0.proposal_environment_mass).masked_select(
            ~p0.proposal_occupied
        )
        == 0.0
    )

    p1 = router.prepare(
        encoded, module_states + 0.75, environment_states - 0.5
    )
    assert p1 is not p0
    assert p1.phase_occupied.data_ptr() != p0.phase_occupied.data_ptr()
    route = router.route_queries(encoded, p1, encoded.env_coords[:, :6])
    torch.testing.assert_close(
        route.assignment.sum(dim=-1),
        torch.ones_like(route.assignment[..., 0]),
        atol=2.0e-5,
        rtol=0.0,
    )
    assert torch.isfinite(route.assignment).all()
    assert torch.all(
        route.assignment.masked_select(
            ~p1.phase_occupied[:, None, :].expand_as(route.assignment)
        )
        == 0.0
    )
    key_rms = route.query_keys.square().mean(dim=-1)
    torch.testing.assert_close(key_rms, torch.ones_like(key_rms), atol=3.0e-5, rtol=0.0)
    expected_keys = router._rms_scale(
        router.group_codes[None, :, :]
        + router.query_group_projection(p1.group_control)
    )
    torch.testing.assert_close(route.query_keys, expected_keys)


def test_source_content_proposal_receives_one_quarter_diagonal_geometry_refinement() -> None:
    encoded, module_states, environment_states = _encoded(seed=1501002)
    router = SparseIncidenceGroupRouter(
        16, group_count=12, control_dim=16, fourier_frequencies=2
    )
    (
        _,
        module_control,
        environment_control,
        active_modules,
        module_measure,
        environment_measure,
    ) = router._source_controls_and_measures(
        encoded, module_states, environment_states
    )
    codes = router.group_codes
    valid = torch.ones((2, 12), dtype=torch.bool)
    module_logits = router._content_logits(module_control, codes)
    environment_logits = router._content_logits(environment_control, codes)
    module_proposal = entmax15(
        module_logits,
        dim=-1,
        mask=valid[:, None, :] & active_modules[..., None],
    ) * active_modules[..., None]
    environment_proposal = entmax15(
        environment_logits,
        dim=-1,
        mask=valid[:, None, :],
    )
    final_module, final_environment, *_ = router._build_assignments(
        encoded,
        module_control,
        environment_control,
        active_modules,
        module_measure,
        environment_measure,
        codes,
        valid,
    )
    diagonal = torch.linalg.vector_norm(encoded.coordinate_scale)
    torch.testing.assert_close(
        router._geometry_scale(encoded),
        torch.full((2,), 0.25 * float(diagonal)),
    )
    assert torch.max(torch.abs(final_module - module_proposal)) > 1.0e-5
    assert torch.max(torch.abs(final_environment - environment_proposal)) > 1.0e-5


def test_query_routing_is_safe_when_some_phase_groups_are_empty() -> None:
    encoded, module_states, environment_states = _encoded(seed=1501003)
    router = SparseIncidenceGroupRouter(
        16, group_count=12, control_dim=16, fourier_frequencies=2
    )
    state = router.prepare(encoded, module_states, environment_states)
    keep = state.phase_occupied.clone()
    keep[:, -3:] = False
    module_membership = state.module_membership.clone()
    environment_membership = state.environment_membership.clone()
    module_membership[..., -3:] = 0.0
    environment_membership[..., -3:] = 0.0
    masked = replace(
        state,
        module_membership=module_membership,
        environment_membership=environment_membership,
        module_mass=state.module_mass * keep,
        environment_mass=state.environment_mass * keep,
        group_control=state.group_control * keep[..., None],
        phase_occupied=keep,
    )
    route = router.route_queries(encoded, masked, encoded.env_coords[:, :6])
    assert torch.isfinite(route.assignment).all()
    torch.testing.assert_close(
        route.assignment.sum(dim=-1), torch.ones_like(route.assignment[..., 0])
    )
    assert torch.count_nonzero(route.assignment[..., -3:]) == 0


def test_bitmask_support_matches_dense_boolean_intersection_and_permutation() -> None:
    generator = torch.Generator().manual_seed(1501003)
    query = torch.rand(2, 9, 12, generator=generator)
    source = torch.rand(2, 7, 12, generator=generator)
    query = query * (query > 0.68)
    source = source * (source > 0.72)
    direct = (query.gt(0)[:, :, None, :] & source.gt(0)[:, None, :, :]).any(
        dim=-1
    )
    torch.testing.assert_close(occupancy_support(query, source), direct)
    torch.testing.assert_close(
        direct_mask_intersection(pack_16bit_mask(query), pack_16bit_mask(source)),
        direct,
    )
    permutation = torch.randperm(12, generator=generator)
    torch.testing.assert_close(
        occupancy_support(query[..., permutation], source[..., permutation]), direct
    )


def test_core_is_rectangular_phase_local_and_backward_trainable() -> None:
    torch.manual_seed(1501004)
    core = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_small_payload())).train()
    assert isinstance(core.backend, SparseIncidenceGroupControlPairwiseField)
    assert core.backend.executor_policy == "rectangular_reference"
    assert core.backend.ledger_rectangular_rows is True
    batch = _batch()
    encoded = core.encode_case(batch)
    p0 = core.prepare(encoded, encoded.module_tokens, return_routing_maps=True)
    p1 = core.prepare(
        encoded, encoded.module_tokens + 0.125, return_routing_maps=True
    )
    assert p0.phase_shared_state is None
    assert p1.phase_shared_state is None
    assert "sparse_incidence_module_incidence" in p0.interaction_aux
    result = core.read(p1, batch.query_xy, return_routing_maps=True)
    assert torch.isfinite(result.context).all()
    assert "sparse_incidence_module_mask_support" in result.interaction_aux
    assert result.interaction_aux["group_control_module_fine_rows"].item() == result.interaction_aux[
        "group_control_module_fine_rows_forward"
    ].item()
    assert result.interaction_aux["group_control_environment_fine_rows"].item() == result.interaction_aux[
        "group_control_environment_fine_rows_forward"
    ].item()
    result.context.square().mean().backward()
    gradient = core.backend.router.group_codes.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_selected_executor_runs_each_supported_pair_once_and_matches_reference() -> None:
    torch.manual_seed(1501006)
    core = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_small_payload())).eval()
    batch = _batch(seed=1501006, queries=11)
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens, return_routing_maps=True)
    backend = core.backend
    backend.executor_policy = "rectangular_reference"
    backend.diagnostic_executor_independent = True
    backend.ledger_rectangular_rows = True
    with torch.no_grad():
        reference = core.read(prepared, batch.query_xy, return_routing_maps=True)
    backend.executor_policy = "diagnostic_support"
    backend.diagnostic_executor_independent = False
    backend.ledger_rectangular_rows = False
    with torch.no_grad():
        selected = core.read(prepared, batch.query_xy, return_routing_maps=True)
    torch.testing.assert_close(selected.context, reference.context, atol=5.0e-5, rtol=5.0e-5)
    aux = selected.interaction_aux
    assert aux["group_control_module_fine_rows"].item() == aux[
        "group_control_module_unique_pairs"
    ].item()
    assert aux["group_control_environment_fine_rows"].item() == aux[
        "group_control_environment_unique_pairs"
    ].item()
    assert aux["group_control_module_fine_rows"].item() < reference.interaction_aux[
        "group_control_module_fine_rows"
    ].item()
    assert aux["group_control_environment_fine_rows"].item() <= reference.interaction_aux[
        "group_control_environment_fine_rows"
    ].item()


def test_run1501_profile_identity_and_historical_strict_round_trip() -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/sparse_incidence_group_control_honf_context.json"
    )
    profile = bundle.effective
    interface = profile["model"]["core_honf"]["interface_model"]
    assert profile["model"]["core_honf"]["forward_architecture"] == (
        "sparse_incidence_group_control_honf"
    )
    assert interface["group_count"] == 12
    assert interface["group_control_dim"] == 16
    assert interface["source_normalizer"] == "entmax15"
    assert interface["query_normalizer"] == "sparsemax"
    assert profile["loss"]["case_group_budget_weight"] == 0.0
    assert profile["loss"]["organizer_regularization"]["enabled"] is False
    assert profile["run"] == {
        "id": "1501",
        "name": "sparse_incidence_adaptive_honf",
        "output_root": "project://Trained_Results",
    }

    historical = copy.deepcopy(_small_payload())
    historical["forward_architecture"] = "occupancy_adaptive_group_control_honf"
    historical["interface_model"]["query_normalizer"] = "entmax15"
    config = UnifiedForwardConfig.from_dict(historical)
    source = InterfaceFieldCore(config).eval()
    batch = _batch(seed=1501005, queries=3)
    encoded = source.encode_case(batch)
    source.prepare(encoded, encoded.module_tokens)
    state = copy.deepcopy(source.state_dict())
    restored = InterfaceFieldCore(config).eval()
    restored.load_state_dict(state, strict=True)
