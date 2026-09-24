"""Parent identity and virtual-quotient tests for Run 1503-v3."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path
from types import MethodType

import pytest
import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields import InterfaceFieldCore
from honf_forward_core.interface_fields.coalesced_sparse_incidence import (
    ConvergedCoalescedPreparedGroupControl,
)
from honf_forward_core.interface_fields.group_control_router import GroupQueryRoute
from honf_forward_core.interface_fields.sparse_incidence_group_control import (
    SparseIncidenceGroupControlPairwiseField,
)
from honf_forward_core.interface_fields.sparse_incidence_router import (
    SparseIncidencePreparedGroupControl,
)
from honf_forward_core.organization.group_fusion import GroupFusionPlan, effective_query_centers

V3_ARCHITECTURE = "converged_identity_preserving_coalescence_honf"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_PROFILE = (
    PROJECT_ROOT
    / "src/config_core/forward/sparse_incidence_group_control_honf_context.json"
)


def _payload(architecture: str) -> dict[str, object]:
    payload = json.loads(PARENT_PROFILE.read_text(encoding="utf-8"))["model"][
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
            "forward_architecture": architecture,
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
    if architecture in {
        "sparse_incidence_group_control_honf",
        "coalesced_sparse_incidence_honf",
        V3_ARCHITECTURE,
    }:
        payload["interface_model"]["environment_refinement_normalizer"] = "sparsemax"
    return payload


def _batch(seed: int, queries: int) -> BatchData:
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
        case_name="run1503-v3-identity-test",
        metadata={},
        env_coords=torch.rand(2, 13, 2, generator=generator) * extent,
        env_features=torch.randn(2, 13, 2, generator=generator),
        env_weights=torch.rand(2, 13, generator=generator) + 0.2,
    )


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:2" if torch.cuda.device_count() > 2 else "cuda:0")
    return torch.device("cpu")


def _batch_on_device(seed: int, queries: int, device: torch.device):
    batch = _batch(seed=seed, queries=queries)
    updates = {
        field.name: value.to(device)
        for field in fields(batch)
        if torch.is_tensor(value := getattr(batch, field.name))
    }
    return replace(batch, **updates)


def _v3_payload() -> dict[str, object]:
    payload = _payload("sparse_incidence_group_control_honf")
    payload["forward_architecture"] = V3_ARCHITECTURE
    payload["interface_model"].update(
        {
            "environment_refinement_normalizer": "sparsemax",
            "fusion_max_iterations": 512,
            "fusion_eps_abs": 1.0e-9,
            "fusion_eps_rel": 1.0e-8,
            "fusion_eta_final": 0.5,
            "fusion_ramp_epochs": 150,
        }
    )
    return payload


def _paired_cores(
    seed: int = 15033001,
    *,
    device: torch.device | None = None,
) -> tuple[InterfaceFieldCore, InterfaceFieldCore]:
    device = device or _device()
    torch.manual_seed(seed)
    parent = InterfaceFieldCore(
        UnifiedForwardConfig.from_dict(_payload("sparse_incidence_group_control_honf"))
    ).to(device)
    candidate = InterfaceFieldCore(
        UnifiedForwardConfig.from_dict(_v3_payload())
    ).to(device)
    parent.eval()
    candidate.eval()
    candidate.set_training_progress(epoch=0, total_epochs=500)
    warmup = _batch_on_device(seed + 1, 2, device)
    with torch.no_grad():
        for core in (parent, candidate):
            encoded = core.encode_case(warmup)
            prepared = core.prepare(encoded, encoded.module_tokens)
            core.decode_queries(prepared, warmup.query_xy)
    candidate.load_state_dict(parent.state_dict(), strict=True)
    return parent, candidate


def _patch_converged_plan(
    monkeypatch: pytest.MonkeyPatch,
    transform: Callable[[GroupFusionPlan], GroupFusionPlan],
) -> None:
    original = GroupFusionPlan.build_converged

    def wrapped(cls, *args, **kwargs):
        return transform(original(*args, **kwargs))

    monkeypatch.setattr(
        GroupFusionPlan, "build_converged", classmethod(wrapped)
    )


def _force_all_singleton(plan: GroupFusionPlan) -> GroupFusionPlan:
    batch = int(plan.active.shape[0])
    groups = int(plan.active.shape[1])
    ids = torch.arange(groups, device=plan.active.device)[None, :].expand(batch, -1)
    class_index = torch.where(plan.active, ids, torch.full_like(ids, -1))
    multiplicity = plan.active.to(torch.long)
    return replace(
        plan,
        class_index=class_index,
        multiplicity=multiplicity,
        valid=plan.active,
        pair_fused=torch.zeros_like(plan.pair_fused),
    )


def _force_first_pair_merge(plan: GroupFusionPlan) -> GroupFusionPlan:
    batch = int(plan.active.shape[0])
    class_index = torch.full_like(plan.class_index, -1)
    multiplicity = torch.zeros_like(plan.multiplicity)
    valid = torch.zeros_like(plan.valid)
    pair_fused = torch.zeros_like(plan.pair_fused)
    for batch_index in range(batch):
        active_ids = torch.nonzero(plan.active[batch_index], as_tuple=False).flatten()
        if batch_index == 0:
            assert int(active_ids.numel()) >= 3
            class_index[batch_index, active_ids[:2]] = 0
            class_index[batch_index, active_ids[2:]] = torch.arange(
                1, int(active_ids.numel()) - 1, device=active_ids.device
            )
            multiplicity[batch_index, 0] = 2
            multiplicity[batch_index, 1 : int(active_ids.numel()) - 1] = 1
            valid[batch_index, : int(active_ids.numel()) - 1] = True
            pair_fused[batch_index, active_ids[0], active_ids[1]] = True
            pair_fused[batch_index, active_ids[1], active_ids[0]] = True
        else:
            class_index[batch_index, active_ids] = active_ids
            multiplicity[batch_index, active_ids] = 1
            valid[batch_index] = plan.active[batch_index]
    return replace(
        plan,
        class_index=class_index,
        multiplicity=multiplicity,
        valid=valid,
        pair_fused=pair_fused,
    )


def _install_last_row_occupancy_hole(core: InterfaceFieldCore) -> int:
    router = core.backend.router
    original_prepare = router.prepare
    chosen_hole: list[int] = []

    def wrapped(self, encoded, module_states, environment_states):
        controls: SparseIncidencePreparedGroupControl = original_prepare(
            encoded, module_states, environment_states
        )
        target_row = int(controls.phase_occupied.shape[0]) - 1
        active_ids = torch.nonzero(
            controls.phase_occupied[target_row], as_tuple=False
        ).flatten()
        assert int(active_ids.numel()) >= 3
        hole = int(active_ids[-1].item())
        chosen_hole[:] = [hole]
        module_membership = controls.module_membership.clone()
        environment_membership = controls.environment_membership.clone()
        module_mass = controls.module_mass.clone()
        environment_mass = controls.environment_mass.clone()
        phase_occupied = controls.phase_occupied.clone()
        module_membership[target_row, :, hole] = 0.0
        environment_membership[target_row, :, hole] = 0.0
        module_mass[target_row, hole] = 0.0
        environment_mass[target_row, hole] = 0.0
        phase_occupied[target_row, hole] = False
        return replace(
            controls,
            module_membership=module_membership,
            environment_membership=environment_membership,
            module_mass=module_mass,
            environment_mass=environment_mass,
            phase_occupied=phase_occupied,
        )

    router.prepare = MethodType(wrapped, router)
    return chosen_hole


def _slice_encoded(encoded, row: int):
    updates = {}
    for field in fields(encoded):
        value = getattr(encoded, field.name)
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) > row:
            updates[field.name] = value[row : row + 1]
    return replace(encoded, **updates)


def _rho_and_moment(
    route: GroupQueryRoute,
    controls: SparseIncidencePreparedGroupControl,
    source: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source == "module":
        incidence = controls.module_membership
    else:
        incidence = controls.environment_membership
    rho = torch.bmm(route.assignment, incidence.transpose(1, 2))
    source_moment = incidence[..., None] * controls.group_control[:, None, :, :]
    moment = torch.einsum("bqk,bskd->bqsd", route.assignment, source_moment)
    return rho, moment


def _selected_parameters(core: InterfaceFieldCore) -> dict[str, torch.nn.Parameter]:
    parameters = dict(core.named_parameters())
    selected = {}
    fragments = (
        "backend.router.group_codes",
        "backend.router.query_projection",
        "backend.router.group_control",
        "backend.module_control_gain.weight",
        "backend.environment_value_control.weight",
        "backend.environment_score_control.weight",
    )
    for fragment in fragments:
        matches = [
            (name, parameter)
            for name, parameter in parameters.items()
            if fragment in name and (not fragment.endswith("query_projection") or name.endswith("weight"))
        ]
        assert matches, f"no parameter found for {fragment}"
        selected[matches[0][0]] = matches[0][1]
    return selected


def _assert_same_gradients(
    left_core: InterfaceFieldCore,
    right_core: InterfaceFieldCore,
    left_loss: torch.Tensor,
    right_loss: torch.Tensor,
    input_tensors: tuple[torch.Tensor, ...],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    left_parameters = _selected_parameters(left_core)
    right_parameters = _selected_parameters(right_core)
    assert left_parameters.keys() == right_parameters.keys()
    left_values = (*input_tensors, *left_parameters.values())
    right_values = (*input_tensors, *right_parameters.values())
    left_gradients = torch.autograd.grad(
        left_loss, left_values, allow_unused=True, retain_graph=True
    )
    right_gradients = torch.autograd.grad(
        right_loss, right_values, allow_unused=True, retain_graph=True
    )
    for left, right in zip(left_gradients, right_gradients, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            assert torch.isfinite(left).all()
            assert torch.isfinite(right).all()
            torch.testing.assert_close(left, right, atol=atol, rtol=rtol)


def test_nonzero_eta_singleton_operator_is_exact_parent_with_first_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _device()
    parent, candidate = _paired_cores()
    candidate.set_training_progress(epoch=50, total_epochs=500)
    batch = _batch_on_device(seed=15033003, queries=5, device=device)
    batch = replace(
        batch,
        query_xy=batch.query_xy.clone().requires_grad_(),
        module_centers=batch.module_centers.clone().requires_grad_(),
        env_coords=batch.env_coords.clone().requires_grad_(),
    )
    assert candidate.backend._fusion_strength() > 0.0
    _patch_converged_plan(monkeypatch, _force_all_singleton)

    parent_encoded = parent.encode_case(batch)
    candidate_encoded = candidate.encode_case(batch)
    parent_prepared = parent.prepare(
        parent_encoded, parent_encoded.module_tokens, return_routing_maps=True
    )
    candidate_prepared = candidate.prepare(
        candidate_encoded, candidate_encoded.module_tokens, return_routing_maps=True
    )
    candidate_plan = candidate_prepared.backend_state["coalescence_plan"]
    assert candidate_plan.fusion_strength[0].item() == pytest.approx(1.0 / 6.0)
    assert candidate_prepared.backend_state["coalescence_all_singleton"]
    assert torch.equal(candidate_plan.class_index, candidate_plan.active.long().cumsum(-1) - 1)
    assert torch.equal(candidate_plan.multiplicity, candidate_plan.active.long())

    parent_route = parent.backend._route(
        parent_prepared.backend_state,
        parent_encoded,
        batch.query_xy,
        parent._receiver_features(parent_prepared, batch.query_xy.float()),
    )
    candidate_route = candidate.backend._route(
        candidate_prepared.backend_state,
        candidate_encoded,
        batch.query_xy,
        candidate._receiver_features(candidate_prepared, batch.query_xy.float()),
    )
    for left, right in (
        (parent_route.logits, candidate_route.logits),
        (parent_route.assignment, candidate_route.assignment),
        (parent_route.query_keys, candidate_route.query_keys),
    ):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)

    for source in ("module", "environment"):
        parent_rho, parent_moment = _rho_and_moment(
            parent_route, parent_prepared.backend_state["group_control_state"], source
        )
        candidate_rho, candidate_moment = _rho_and_moment(
            candidate_route,
            candidate_prepared.backend_state["group_control_state"],
            source,
        )
        torch.testing.assert_close(parent_rho, candidate_rho, atol=0.0, rtol=0.0)
        torch.testing.assert_close(parent_moment, candidate_moment, atol=0.0, rtol=0.0)

    query_features_parent = parent._receiver_features(
        parent_prepared, batch.query_xy.float()
    )
    query_features_candidate = candidate._receiver_features(
        candidate_prepared, batch.query_xy.float()
    )
    parent_qm, _ = parent.backend._read_module(
        parent_prepared.backend_state,
        parent_encoded,
        batch.query_xy,
        parent_route,
        include_diagnostics=False,
    )
    parent_qe, _ = parent.backend._read_environment(
        parent_prepared.backend_state,
        parent_encoded,
        batch.query_xy,
        query_features_parent,
        parent_route,
        include_diagnostics=False,
    )
    candidate_parent_read_state = dict(candidate_prepared.backend_state)
    candidate_parent_read_state.pop("coalescence_plan")
    candidate_qm, _ = candidate.backend._read_module(
        candidate_parent_read_state,
        candidate_encoded,
        batch.query_xy,
        candidate_route,
        include_diagnostics=False,
    )
    candidate_qe, _ = candidate.backend._read_environment(
        candidate_parent_read_state,
        candidate_encoded,
        batch.query_xy,
        query_features_candidate,
        candidate_route,
        include_diagnostics=False,
    )
    torch.testing.assert_close(parent_qm, candidate_qm, atol=0.0, rtol=0.0)
    torch.testing.assert_close(parent_qe, candidate_qe, atol=0.0, rtol=0.0)

    parent_output = parent.decode_queries(
        parent_prepared,
        batch.query_xy,
        return_routing_maps=True,
    )
    candidate_output = candidate.decode_queries(
        candidate_prepared,
        batch.query_xy,
        return_routing_maps=True,
    )
    torch.testing.assert_close(
        parent_output["pred_field"], candidate_output["pred_field"], atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        parent_output["sparse_incidence_query_logits"],
        candidate_output["sparse_incidence_query_logits"],
        atol=0.0,
        rtol=0.0,
    )
    left_weights = torch.linspace(
        0.7, 1.3, parent_output["pred_field"].numel(), device=device
    ).reshape_as(parent_output["pred_field"])
    _assert_same_gradients(
        parent,
        candidate,
        (parent_output["pred_field"] * left_weights).sum(),
        (candidate_output["pred_field"] * left_weights).sum(),
        (batch.query_xy, batch.module_centers, batch.env_coords),
    )


def test_mixed_merge_keeps_singleton_hole_parent_exact_and_quotient_conserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _device()
    parent, candidate = _paired_cores(seed=15033011)
    candidate.set_training_progress(epoch=50, total_epochs=500)
    batch = _batch_on_device(seed=15033012, queries=5, device=device)
    batch = replace(
        batch,
        query_xy=batch.query_xy.clone().requires_grad_(),
        module_centers=batch.module_centers.clone().requires_grad_(),
        env_coords=batch.env_coords.clone().requires_grad_(),
    )
    _install_last_row_occupancy_hole(parent)
    hole_box = _install_last_row_occupancy_hole(candidate)
    _patch_converged_plan(monkeypatch, _force_first_pair_merge)

    parent_encoded = parent.encode_case(batch)
    candidate_encoded = candidate.encode_case(batch)
    parent_full = parent.prepare(
        parent_encoded, parent_encoded.module_tokens, return_routing_maps=True
    )
    candidate_prepared = candidate.prepare(
        candidate_encoded, candidate_encoded.module_tokens, return_routing_maps=True
    )
    plan: GroupFusionPlan = candidate_prepared.backend_state["coalescence_plan"]
    controls: ConvergedCoalescedPreparedGroupControl = candidate_prepared.backend_state[
        "group_control_state"
    ]
    assert plan.case_group_count[0] < plan.active_proposal_count[0]
    assert plan.case_group_count[1] == plan.active_proposal_count[1]
    assert int(plan.multiplicity[0].max()) == 2
    assert hole_box and hole_box[0] >= 0
    assert not plan.active[1, hole_box[0]]
    assert torch.equal(plan.class_index[1, plan.active[1]], torch.arange(
        plan.active.shape[1], device=device
    )[plan.active[1]])
    assert torch.equal(
        controls.module_membership[1],
        parent_full.backend_state["group_control_state"].module_membership[1],
    )
    assert torch.equal(
        controls.environment_membership[1],
        parent_full.backend_state["group_control_state"].environment_membership[1],
    )

    candidate_route = candidate.backend._route(
        candidate_prepared.backend_state,
        candidate_encoded,
        batch.query_xy,
        candidate._receiver_features(candidate_prepared, batch.query_xy.float()),
    )
    parent_full_route = parent.backend._route(
        parent_full.backend_state,
        parent_encoded,
        batch.query_xy,
        parent._receiver_features(parent_full, batch.query_xy.float()),
    )
    torch.testing.assert_close(
        candidate_route.logits[1], parent_full_route.logits[1], atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        candidate_route.assignment[1], parent_full_route.assignment[1], atol=0.0, rtol=0.0
    )
    for source in ("module", "environment"):
        candidate_rho, candidate_moment = _rho_and_moment(
            candidate_route, controls, source
        )
        parent_controls = parent_full.backend_state["group_control_state"]
        parent_rho, parent_moment = _rho_and_moment(
            parent_full_route, parent_controls, source
        )
        torch.testing.assert_close(
            candidate_rho[1], parent_rho[1], atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            candidate_moment[1], parent_moment[1], atol=0.0, rtol=0.0
        )
    mixed_qm, _ = candidate.backend._read_module(
        candidate_prepared.backend_state,
        candidate_encoded,
        batch.query_xy,
        candidate_route,
        include_diagnostics=False,
    )
    parent_qm, _ = parent.backend._read_module(
        parent_full.backend_state,
        parent_encoded,
        batch.query_xy,
        parent_full_route,
        include_diagnostics=False,
    )
    features = candidate._receiver_features(candidate_prepared, batch.query_xy.float())
    mixed_qe, _ = candidate.backend._read_environment(
        candidate_prepared.backend_state,
        candidate_encoded,
        batch.query_xy,
        features,
        candidate_route,
        include_diagnostics=False,
    )
    parent_qe, _ = parent.backend._read_environment(
        parent_full.backend_state,
        parent_encoded,
        batch.query_xy,
        parent._receiver_features(parent_full, batch.query_xy.float()),
        parent_full_route,
        include_diagnostics=False,
    )
    torch.testing.assert_close(mixed_qm[1], parent_qm[1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(mixed_qe[1], parent_qe[1], atol=0.0, rtol=0.0)

    # Verify conserved Abar/B, direct singleton keys, and geometry rebuilt from
    # the merged fine-source incidence on the actually merged row.
    provisional = controls.parent_controls
    assert provisional is not None
    expected_module = torch.zeros_like(controls.module_membership[0:1])
    expected_module_moment = torch.zeros_like(controls.module_source_moment[0:1])
    expected_environment = torch.zeros_like(controls.environment_membership[0:1])
    expected_environment_moment = torch.zeros_like(
        controls.environment_source_moment[0:1]
    )
    for group in range(plan.active.shape[1]):
        packed = int(plan.class_index[0, group].item())
        if packed >= 0:
            expected_module[:, :, packed] += provisional.module_membership[0:1, :, group]
            expected_module_moment[:, :, packed] += (
                provisional.module_membership[0:1, :, group, None]
                * provisional.group_control[0:1, None, group, :]
            )
            expected_environment[:, :, packed] += provisional.environment_membership[
                0:1, :, group
            ]
            expected_environment_moment[:, :, packed] += (
                provisional.environment_membership[0:1, :, group, None]
                * provisional.group_control[0:1, None, group, :]
            )
    torch.testing.assert_close(
        controls.module_membership[0:1], expected_module, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        controls.module_source_moment[0:1], expected_module_moment, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        controls.environment_membership[0:1], expected_environment, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        controls.environment_source_moment[0:1],
        expected_environment_moment,
        atol=0.0,
        rtol=0.0,
    )
    parent_keys = candidate.backend.router.normalized_query_key_bank(provisional)
    merged_members = torch.nonzero(
        plan.class_index[0] == 0, as_tuple=False
    ).flatten()
    expected_key = candidate.backend.router._rms_scale(
        parent_keys[0:1, merged_members].mean(dim=1)
    )
    torch.testing.assert_close(
        candidate_prepared.backend_state["coalescence_query_keys"][0:1, 0],
        expected_key,
        atol=0.0,
        rtol=0.0,
    )
    (
        merged_module_mass,
        merged_environment_mass,
        merged_module_centers,
        merged_environment_centers,
        _,
    ) = candidate.backend.router._joint_centres(
        expected_module,
        expected_environment,
        provisional.module_measure[0:1],
        provisional.environment_measure[0:1],
        candidate_encoded.module_centers[0:1],
        candidate_encoded.env_coords[0:1],
    )
    expected_module_centers, expected_environment_centers = effective_query_centers(
        merged_module_centers,
        merged_environment_centers,
        merged_module_mass,
        merged_environment_mass,
    )
    torch.testing.assert_close(
        candidate_prepared.backend_state["coalescence_module_query_centers"][0:1, 0],
        expected_module_centers[:, 0],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        candidate_prepared.backend_state[
                "coalescence_environment_query_centers"
            ][0:1, 0],
        expected_environment_centers[:, 0],
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    # Compare the no-merge row to a standalone parent evaluation of that case.
    parent_single_encoded = _slice_encoded(parent_encoded, 1)
    parent_single_prepared = parent.prepare(
        parent_single_encoded,
        parent_single_encoded.module_tokens,
        return_routing_maps=True,
    )
    candidate_single_route = candidate.backend._route(
        candidate_prepared.backend_state,
        candidate_encoded,
        batch.query_xy,
        candidate._receiver_features(candidate_prepared, batch.query_xy.float()),
    )
    parent_single_route = parent.backend._route(
        parent_single_prepared.backend_state,
        parent_single_encoded,
        batch.query_xy[1:2],
        parent._receiver_features(
            parent_single_prepared, batch.query_xy[1:2].float()
        ),
    )
    torch.testing.assert_close(
        candidate_single_route.logits[1:2], parent_single_route.logits, atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        candidate_single_route.assignment[1:2],
        parent_single_route.assignment,
        atol=2e-6,
        rtol=2e-6,
    )
    for source in ("module", "environment"):
        candidate_rho, candidate_moment = _rho_and_moment(
            candidate_single_route, controls, source
        )
        parent_rho, parent_moment = _rho_and_moment(
            parent_single_route,
            parent_single_prepared.backend_state["group_control_state"],
            source,
        )
        torch.testing.assert_close(
            candidate_rho[1:2], parent_rho, atol=2e-6, rtol=2e-6
        )
        torch.testing.assert_close(
            candidate_moment[1:2], parent_moment, atol=2e-6, rtol=2e-6
        )
    candidate_output = candidate.decode_queries(
        candidate_prepared, batch.query_xy, return_routing_maps=True
    )
    parent_single_output = parent.decode_queries(
        parent_single_prepared, batch.query_xy[1:2], return_routing_maps=True
    )
    torch.testing.assert_close(
        candidate_output["pred_field"][1:2],
        parent_single_output["pred_field"],
        atol=3e-6,
        rtol=3e-6,
    )
    singleton_weights = torch.linspace(
        0.8,
        1.2,
        parent_single_output["pred_field"].numel(),
        device=device,
    ).reshape_as(parent_single_output["pred_field"])
    _assert_same_gradients(
        parent,
        candidate,
        (parent_single_output["pred_field"] * singleton_weights).sum(),
        (candidate_output["pred_field"][1:2] * singleton_weights).sum(),
        (batch.query_xy, batch.module_centers, batch.env_coords),
        atol=3e-6,
        rtol=3e-6,
    )

    # The compact quotient and a literal virtual expansion must preserve QM,
    # QE, and first gradients for the deliberately fused two-member class.
    compact_route = candidate.backend._route(
        candidate_prepared.backend_state,
        candidate_encoded,
        batch.query_xy,
        candidate._receiver_features(candidate_prepared, batch.query_xy.float()),
    )
    group_ids = plan.class_index.clamp_min(0)
    virtual_assignment = compact_route.query_density.gather(
        -1, group_ids[:, None, :].expand(-1, batch.query_xy.shape[1], -1)
    ) * plan.active[:, None, :].to(batch.query_xy.dtype)
    virtual_route = GroupQueryRoute(
        query_control=compact_route.query_control,
        assignment=virtual_assignment,
        logits=compact_route.logits.gather(
            -1, group_ids[:, None, :].expand(-1, batch.query_xy.shape[1], -1)
        ),
        query_keys=parent_keys,
    )
    virtual_parent_state = SparseIncidenceGroupControlPairwiseField.prepare(
        candidate.backend,
        candidate_encoded,
        candidate_encoded.module_tokens,
        return_routing_maps=False,
    )
    features = candidate._receiver_features(
        candidate_prepared, batch.query_xy.float()
    )
    compact_qm, _ = candidate.backend._read_module(
        candidate_prepared.backend_state,
        candidate_encoded,
        batch.query_xy,
        compact_route,
        include_diagnostics=False,
    )
    compact_qe, _ = candidate.backend._read_environment(
        candidate_prepared.backend_state,
        candidate_encoded,
        batch.query_xy,
        candidate._receiver_features(candidate_prepared, batch.query_xy.float()),
        compact_route,
        include_diagnostics=False,
    )
    virtual_qm, _ = candidate.backend._read_module(
        virtual_parent_state,
        candidate_encoded,
        batch.query_xy,
        virtual_route,
        include_diagnostics=False,
    )
    virtual_qe, _ = candidate.backend._read_environment(
        virtual_parent_state,
        candidate_encoded,
        batch.query_xy,
        features,
        virtual_route,
        include_diagnostics=False,
    )
    torch.testing.assert_close(compact_qm[0:1], virtual_qm[0:1], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(compact_qe[0:1], virtual_qe[0:1], atol=2e-5, rtol=2e-5)
    compact_loss = (compact_qm[0:1] + compact_qe[0:1]).square().sum()
    virtual_loss = (virtual_qm[0:1] + virtual_qe[0:1]).square().sum()
    _assert_same_gradients(
        candidate,
        candidate,
        compact_loss,
        virtual_loss,
        (batch.query_xy, batch.module_centers, batch.env_coords),
        atol=5e-5,
        rtol=5e-4,
    )
