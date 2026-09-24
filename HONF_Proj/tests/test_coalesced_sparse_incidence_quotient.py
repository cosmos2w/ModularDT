"""Controlled-merge parity for the Run 1503-v2 physical quotient reader."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields import InterfaceFieldCore
from honf_forward_core.interface_fields.coalesced_sparse_incidence import (
    CoalescedGroupQueryRoute,
    CoalescedPreparedGroupControl,
)
from honf_forward_core.interface_fields.group_control_router import GroupQueryRoute
from honf_forward_core.interface_fields.sparse_incidence_group_control import (
    SparseIncidenceGroupControlPairwiseField,
)
from honf_forward_core.interface_fields.sparse_incidence_router import (
    SparseIncidencePreparedGroupControl,
)
from honf_forward_core.organization.group_fusion import GroupFusionPlan

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_PROFILE = (
    PROJECT_ROOT
    / "src/config_core/forward/sparse_incidence_group_control_honf_context.json"
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:2" if torch.cuda.device_count() > 2 else "cuda:0")
    return torch.device("cpu")


def _payload() -> dict[str, object]:
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
            "forward_architecture": "coalesced_sparse_incidence_honf",
        }
    )
    payload["interface_model"].update(
        {
            "message_hidden_dim": 8,
            "attention_heads": 2,
            "relative_fourier_frequencies": 2,
            "receiver_chunk_size": 4,
            "activation_checkpointing": False,
            "environment_refinement_normalizer": "sparsemax",
            "fusion_max_iterations": 64,
            "fusion_eta_final": 0.5,
            "fusion_ramp_epochs": 150,
        }
    )
    return payload


def _batch(device: torch.device) -> BatchData:
    generator = torch.Generator(device="cpu").manual_seed(15030301)
    extent = torch.tensor([12.0, 6.0])
    return BatchData(
        module_centers=torch.rand(2, 5, 2, generator=generator).to(device) * extent.to(device),
        module_present=torch.tensor(
            [[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
            device=device,
        ),
        module_features=torch.randn(2, 5, 3, generator=generator).to(device),
        global_context=torch.randn(2, 4, generator=generator).to(device),
        query_xy=torch.rand(2, 5, 2, generator=generator).to(device) * extent.to(device),
        query_time=None,
        target_field=torch.randn(2, 5, 5, generator=generator).to(device),
        case_name="run1503-v2-controlled-quotient-test",
        metadata={},
        env_coords=torch.rand(2, 13, 2, generator=generator).to(device) * extent.to(device),
        env_features=torch.randn(2, 13, 2, generator=generator).to(device),
        env_weights=torch.rand(2, 13, generator=generator).to(device) + 0.2,
    )


def _force_first_pair_quotient(plan: GroupFusionPlan) -> GroupFusionPlan:
    """Merge the first two active inputs, keeping every other active singleton."""

    batch, groups = plan.active.shape
    class_index = torch.full_like(plan.class_index, -1)
    for batch_index in range(batch):
        active_indices = torch.nonzero(plan.active[batch_index], as_tuple=False).flatten()
        assert int(active_indices.numel()) >= 2
        for rank, input_index in enumerate(active_indices.tolist()):
            class_index[batch_index, input_index] = max(0, rank - 1)
    membership = F.one_hot(class_index.clamp_min(0), num_classes=groups).to(
        plan.input_descriptors.dtype
    )
    membership = membership * plan.active[..., None].to(membership.dtype)
    multiplicity = membership.sum(dim=1).to(torch.long)
    valid = multiplicity > 0
    fused = torch.einsum("bkr,bkd->brd", membership, plan.input_descriptors)
    fused = fused / multiplicity.clamp_min(1).to(fused.dtype)[..., None]
    fused = fused * valid[..., None].to(fused.dtype)
    projection = torch.linalg.vector_norm(
        plan.input_descriptors
        - torch.einsum("bkr,brd->bkd", membership, fused),
        dim=-1,
    )
    same_class = (
        (class_index[:, :, None] == class_index[:, None, :])
        & plan.active[:, :, None]
        & plan.active[:, None, :]
    )
    same_class = same_class & ~torch.eye(
        groups, device=plan.active.device, dtype=torch.bool
    )[None, :, :]
    return replace(
        plan,
        fused_descriptors=fused,
        class_index=class_index,
        multiplicity=multiplicity,
        valid=valid,
        projection_displacement=projection,
        pair_fused=same_class,
    )


def _literal_sparsemax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Independent ordinary sparsemax oracle for repeated virtual logits."""

    masked = logits.masked_fill(~valid, -torch.inf)
    maximum = masked.amax(dim=-1, keepdim=True)
    shifted = torch.where(valid, logits - maximum, torch.full_like(logits, -torch.inf))
    sorted_scores, _ = torch.sort(shifted, dim=-1, descending=True)
    rank = torch.arange(
        1, logits.shape[-1] + 1, device=logits.device, dtype=logits.dtype
    )
    cumulative = sorted_scores.cumsum(dim=-1) - 1.0
    support = (rank * sorted_scores > cumulative) & torch.isfinite(sorted_scores)
    support_count = support.sum(dim=-1, keepdim=True).clamp_min(1)
    tau = torch.gather(cumulative, -1, support_count - 1) / support_count.to(
        logits.dtype
    )
    return torch.where(valid, torch.relu(shifted - tau), torch.zeros_like(logits))


def _packed_state(
    backend: torch.nn.Module,
    parent_state: dict[str, object],
    plan: GroupFusionPlan,
    encoded: object,
) -> tuple[dict[str, object], CoalescedPreparedGroupControl]:
    provisional: SparseIncidencePreparedGroupControl = parent_state[
        "group_control_state"
    ]
    module_incidence, module_moment = plan.reduce_source_banks(
        provisional.module_membership, provisional.group_control
    )
    environment_incidence, environment_moment = plan.reduce_source_banks(
        provisional.environment_membership, provisional.group_control
    )
    (
        module_mass,
        environment_mass,
        module_centers,
        environment_centers,
        joint_centers,
    ) = backend.router._joint_centres(
        module_incidence,
        environment_incidence,
        provisional.module_measure,
        provisional.environment_measure,
        encoded.module_centers,
        encoded.env_coords,
    )
    mean_control = backend._mean_group_controls(provisional, plan)
    packed_base = replace(
        provisional,
        module_membership=module_incidence,
        environment_membership=environment_incidence,
        module_mass=module_mass,
        environment_mass=environment_mass,
        group_control=mean_control,
        phase_occupied=plan.valid,
        module_centres=module_centers,
        environment_centres=environment_centers,
        joint_centres=joint_centers,
    )
    packed_controls = CoalescedPreparedGroupControl(
        **{
            name: getattr(packed_base, name)
            for name in SparseIncidencePreparedGroupControl.__dataclass_fields__
        },
        fusion_plan=plan,
        module_source_moment=module_moment,
        environment_source_moment=environment_moment,
    )
    packed_state = dict(parent_state)
    packed_state["group_control_state"] = packed_controls
    packed_state["module_control_bank"] = module_moment.permute(0, 2, 1, 3).reshape(
        module_moment.shape[0],
        module_moment.shape[2],
        module_moment.shape[1] * module_moment.shape[-1],
    )
    packed_state["module_control_bank_membership"] = module_incidence
    packed_state["module_control_bank_group_control"] = mean_control
    packed_state["module_control_bank_source_moment"] = module_moment
    (
        environment_key,
        environment_value,
        source_group_control,
        head_control,
        head_source_control,
        value_gain,
    ) = backend._prepare_environment_bank_with_gain(parent_state["env_tokens"], packed_controls)
    packed_state.update(
        {
            "environment_keys": environment_key,
            "environment_values": environment_value,
            "environment_source_group_control": source_group_control,
            "environment_head_control": head_control,
            "environment_head_source_control": head_source_control,
            "environment_value_gain": value_gain,
            "coalescence_plan": plan,
        }
    )
    return packed_state, packed_controls


def _constituent_sum(
    incidence: torch.Tensor, controls: torch.Tensor, plan: GroupFusionPlan
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent loop oracle for Abar and source-resolved B."""

    batch, sources, groups = incidence.shape
    width = controls.shape[-1]
    expected_incidence = incidence.new_zeros(batch, sources, groups)
    expected_moment = incidence.new_zeros(batch, sources, groups, width)
    for batch_index in range(batch):
        for group in range(groups):
            packed = int(plan.class_index[batch_index, group].item())
            if packed >= 0:
                expected_incidence[batch_index, :, packed] += incidence[
                    batch_index, :, group
                ]
                expected_moment[batch_index, :, packed] += (
                    incidence[batch_index, :, group, None]
                    * controls[batch_index, group]
                )
    return expected_incidence, expected_moment


def test_forced_quotient_reader_matches_literal_virtual_qm_qe_and_gradients() -> None:
    device = _device()
    torch.manual_seed(15030302)
    core = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_payload())).to(device).train()
    core.set_training_progress(epoch=50, total_epochs=500)
    batch = _batch(device)
    encoded = core.encode_case(batch)
    backend = core.backend

    # Produce both paths from the same live parent source-controller preparation.
    provisional_state = SparseIncidenceGroupControlPairwiseField.prepare(
        backend,
        encoded,
        encoded.module_tokens,
        return_routing_maps=False,
    )
    candidate_state = backend.prepare(
        encoded,
        encoded.module_tokens,
        return_routing_maps=True,
    )
    generated_plan: GroupFusionPlan = candidate_state["coalescence_plan"]
    plan = _force_first_pair_quotient(generated_plan)
    assert (plan.case_group_count < plan.active_proposal_count).all()
    assert (plan.multiplicity == 2).sum(dim=-1).eq(1).all()
    assert (plan.multiplicity == 1).sum(dim=-1).eq(plan.active_proposal_count - 2).all()

    provisional: SparseIncidencePreparedGroupControl = provisional_state[
        "group_control_state"
    ]

    query = plan.module_query_centers[:, :1, :].expand(-1, 5, -1).contiguous()
    receiver_features = backend.router.query_fourier(
        query / backend.router._scale(encoded)
    )
    query_input = torch.cat(
        [
            receiver_features,
            provisional.global_control[:, None, :].expand(-1, query.shape[1], -1),
        ],
        dim=-1,
    )
    scaled_query = backend.router._rms_scale(
        backend.router.query_projection(query_input)
    )
    geometry = -0.5 * (
        torch.linalg.vector_norm(
            query[:, :, None, :] - plan.module_query_centers[:, None, :, :], dim=-1
        )
        + torch.linalg.vector_norm(
            query[:, :, None, :] - plan.environment_query_centers[:, None, :, :],
            dim=-1,
        )
    ) / plan.geometry_scale[:, None, None]
    # Give the merged class and one singleton equal, live sparsemax support.
    # This controls only this algebra fixture's access logits; source moments,
    # fine QM/QE rows, nonlinear readers, and gradients remain model-generated.
    target_logits = torch.full_like(geometry[:, 0, :], -2.0)
    target_logits[:, :2] = 0.5
    key_direction = scaled_query[:, 0, :].detach()
    key_norm = key_direction.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    key_block = key_direction[:, None, :] * (
        (target_logits - geometry[:, 0, :].detach()) / key_norm
    )[..., None]
    controlled_descriptors = plan.fused_descriptors.clone()
    controlled_descriptors[..., : plan.control_dim] = key_block
    plan = replace(plan, fused_descriptors=controlled_descriptors)

    packed_state, packed_controls = _packed_state(
        backend, provisional_state, plan, encoded
    )
    # The controller's parent calibration is carried through coalescence.
    assert torch.equal(packed_controls.kappa, provisional.kappa)
    assert torch.equal(packed_controls.pi, provisional.pi)

    expected_m, expected_bm = _constituent_sum(
        provisional.module_membership, provisional.group_control, plan
    )
    expected_e, expected_be = _constituent_sum(
        provisional.environment_membership, provisional.group_control, plan
    )
    torch.testing.assert_close(packed_controls.module_membership, expected_m)
    torch.testing.assert_close(packed_controls.module_source_moment, expected_bm)
    torch.testing.assert_close(packed_controls.environment_membership, expected_e)
    torch.testing.assert_close(packed_controls.environment_source_moment, expected_be)

    compact_route = backend._route(
        packed_state, encoded, query, receiver_features
    )
    assert isinstance(compact_route, CoalescedGroupQueryRoute)
    assert torch.equal(compact_route.query_density, compact_route.assignment)
    torch.testing.assert_close(
        compact_route.query_mass.sum(dim=-1),
        torch.ones_like(compact_route.query_mass[..., 0]),
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    assert (compact_route.query_mass[..., 0] > 0.0).all()
    assert (compact_route.query_mass[..., 1] > 0.0).all()
    torch.testing.assert_close(
        compact_route.query_mass[..., 0],
        2.0 * compact_route.query_density[..., 0],
        atol=2.0e-6,
        rtol=2.0e-6,
    )

    # Literal virtual model: repeat each fused class logit over its original
    # constituents, then use ordinary sparsemax and the unchanged parent reader.
    virtual_logits = compact_route.logits.gather(
        -1,
        plan.class_index.clamp_min(0)[:, None, :].expand(
            -1, query.shape[1], -1
        ),
    )
    virtual_assignment = _literal_sparsemax(
        virtual_logits, provisional.phase_occupied[:, None, :]
    )
    expected_class_mass = virtual_assignment.new_zeros(
        plan.valid.shape[0], query.shape[1], plan.valid.shape[1]
    ).scatter_add(
        2,
        plan.class_index.clamp_min(0)[:, None, :].expand(
            -1, query.shape[1], -1
        ),
        virtual_assignment,
    )
    torch.testing.assert_close(
        compact_route.query_mass, expected_class_mass, atol=3.0e-6, rtol=3.0e-6
    )
    virtual_route = GroupQueryRoute(
        query_control=compact_route.query_control,
        assignment=virtual_assignment,
        logits=virtual_logits,
        query_keys=compact_route.query_keys,
    )

    compact_context, _ = backend.read(
        packed_state,
        encoded,
        query,
        receiver_features,
        return_routing_maps=False,
    )
    virtual_module, _ = backend._read_module(
        provisional_state,
        encoded,
        query,
        virtual_route,
        include_diagnostics=False,
    )
    virtual_environment, _ = backend._read_environment(
        provisional_state,
        encoded,
        query,
        receiver_features,
        virtual_route,
        include_diagnostics=False,
    )
    virtual_context = virtual_module + virtual_environment
    torch.testing.assert_close(
        compact_context, virtual_context, atol=2.0e-5, rtol=2.0e-5
    )

    parameters = [
        backend.router.query_projection[2].weight,
        backend.router.group_control.net[0].weight,
        backend.module_control_gain.weight,
        backend.environment_value_control.weight,
        backend.environment_score_control.weight,
    ]
    compact_gradients = torch.autograd.grad(
        compact_context.square().sum(), parameters, retain_graph=True
    )
    virtual_gradients = torch.autograd.grad(
        virtual_context.square().sum(), parameters, retain_graph=True
    )
    for compact_gradient, virtual_gradient in zip(
        compact_gradients, virtual_gradients, strict=True
    ):
        assert torch.isfinite(compact_gradient).all()
        assert torch.isfinite(virtual_gradient).all()
        torch.testing.assert_close(
            compact_gradient,
            virtual_gradient,
            atol=5.0e-5,
            rtol=5.0e-4,
        )
