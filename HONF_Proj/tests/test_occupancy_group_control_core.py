"""Focused core checks for the opt-in Run-1409 occupancy router."""

from __future__ import annotations

import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.interface_fields.occupancy_group_router import (
    OccupancyAdaptiveGroupRouter,
    OccupancyGroupPlan,
    direct_mask_intersection,
    pack_16bit_mask,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _config() -> UnifiedForwardConfig:
    return UnifiedForwardConfig.from_dict(
        {
            "forward_architecture": "occupancy_adaptive_group_control_honf",
            "field_dim": 3,
            "hidden_dim": 16,
            "coordinate_scale": [8.0, 4.0],
            "boundary_feature_mode": "none",
            "interface_model": {
                "message_hidden_dim": 16,
                "attention_heads": 4,
                "relative_fourier_frequencies": 2,
                "receiver_chunk_size": 4,
                "group_count": 12,
                "source_normalizer": "entmax15",
                "query_normalizer": "entmax15",
                "module_temperature": 1.0,
                "environment_temperature": 1.0,
                "query_temperature": 1.0,
                "group_control_dim": 16,
            },
        }
    )


def _batch(seed: int = 1409, queries: int = 5) -> BatchData:
    generator = torch.Generator().manual_seed(seed)
    extent = torch.tensor([8.0, 4.0])
    return BatchData(
        module_centers=torch.rand(2, 4, 2, generator=generator) * extent,
        module_present=torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
        module_features=torch.randn(2, 4, 3, generator=generator),
        global_context=torch.randn(2, 5, generator=generator),
        query_xy=torch.rand(2, queries, 2, generator=generator) * extent,
        query_time=None,
        target_field=torch.randn(2, queries, 3, generator=generator),
        case_name="run1409-occupancy-test",
        metadata={},
        env_coords=torch.rand(2, 8, 2, generator=generator) * extent,
        env_features=torch.randn(2, 8, 2, generator=generator),
        env_weights=torch.rand(2, 8, generator=generator) + 0.2,
    )


def _encoded() -> tuple[EncodedInterfaceCase, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    batch, modules, environments, hidden = 1, 4, 8, 16
    encoded = EncodedInterfaceCase(
        module_tokens=torch.randn(batch, modules, hidden, generator=generator),
        env_tokens=torch.randn(batch, environments, hidden, generator=generator),
        global_token=torch.randn(batch, hidden, generator=generator),
        module_centers=torch.rand(batch, modules, 2, generator=generator) * torch.tensor([8.0, 4.0]),
        env_coords=torch.rand(batch, environments, 2, generator=generator) * torch.tensor([8.0, 4.0]),
        module_present=torch.ones(batch, modules),
        module_features=None,
        env_features=None,
        env_weights=torch.rand(batch, environments, generator=generator) + 0.2,
        coordinate_scale=torch.tensor([8.0, 4.0]),
    )
    return (
        encoded,
        torch.randn(batch, modules, hidden, generator=generator),
        torch.randn(batch, environments, hidden, generator=generator),
    )


def _mixed_encoded_and_plan() -> tuple[
    EncodedInterfaceCase,
    torch.Tensor,
    torch.Tensor,
    OccupancyAdaptiveGroupRouter,
    OccupancyGroupPlan,
]:
    """Build two cases whose P0 plans use different, nonconsecutive IDs."""

    generator = torch.Generator().manual_seed(1420)
    batch, modules, environments, hidden = 2, 4, 8, 16
    encoded = EncodedInterfaceCase(
        module_tokens=torch.randn(batch, modules, hidden, generator=generator),
        env_tokens=torch.randn(batch, environments, hidden, generator=generator),
        global_token=torch.randn(batch, hidden, generator=generator),
        module_centers=torch.rand(batch, modules, 2, generator=generator)
        * torch.tensor([8.0, 4.0]),
        env_coords=torch.rand(batch, environments, 2, generator=generator)
        * torch.tensor([8.0, 4.0]),
        module_present=torch.ones(batch, modules),
        module_features=None,
        env_features=None,
        env_weights=torch.ones(batch, environments),
        coordinate_scale=torch.tensor([8.0, 4.0]),
    )
    module_states = torch.randn(batch, modules, hidden, generator=generator)
    environment_states = torch.randn(batch, environments, hidden, generator=generator)
    router = OccupancyAdaptiveGroupRouter(
        hidden,
        group_count=12,
        control_dim=16,
        fourier_frequencies=2,
    )

    module_assignment = torch.zeros(batch, modules, 12)
    environment_assignment = torch.zeros(batch, environments, 12)
    ids_per_case = ([2, 9], [1, 7, 11])
    for case_index, ids in enumerate(ids_per_case):
        for source_index in range(modules):
            module_assignment[case_index, source_index, ids[source_index % len(ids)]] = 1.0
        for source_index in range(environments):
            environment_assignment[case_index, source_index, ids[source_index % len(ids)]] = 1.0
    plan = router._make_plan(
        module_assignment,
        environment_assignment,
        torch.full((batch, modules), 1.0 / modules),
        torch.full((batch, environments), 1.0 / environments),
        encoded,
        torch.ones(batch),
    )
    return encoded, module_states, environment_states, router, plan


def test_occupancy_masks_are_exact_and_do_not_need_a_lookup_table() -> None:
    source = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]]])
    query = torch.tensor([[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]])
    source_mask = pack_16bit_mask(source)
    query_mask = pack_16bit_mask(query)
    assert source_mask.tolist() == [[1, 2]]
    assert query_mask.tolist() == [[2, 8]]
    assert direct_mask_intersection(query_mask, source_mask).tolist() == [[[False, True], [False, False]]]


def test_router_derives_kappa_and_keeps_original_ids_for_packed_path() -> None:
    encoded, module_states, environment_states = _encoded()
    router = OccupancyAdaptiveGroupRouter(16, group_count=12, control_dim=16, fourier_frequencies=2)
    module_assignment = torch.zeros(1, 4, 12)
    environment_assignment = torch.zeros(1, 8, 12)
    module_assignment[0, 0, 0] = 1.0
    module_assignment[0, 1, 1] = 1.0
    module_assignment[0, 2, 0] = 1.0
    module_assignment[0, 3, 1] = 1.0
    environment_assignment[0, :4, 0] = 1.0
    environment_assignment[0, 4:, 1] = 1.0
    plan = router._make_plan(
        module_assignment,
        environment_assignment,
        torch.full((1, 4), 0.25),
        torch.full((1, 8), 0.125),
        encoded,
        torch.tensor([2.0]),
    )
    assert plan.k_plan.tolist() == [2]
    assert plan.prototype_ids.tolist()[0][:3] == [0, 1, -1]
    assert plan.packed_prototype_ids.tolist() == [[0, 1]]
    assert plan.packed_valid.tolist() == [[True, True]]

    full = router.prepare(encoded, module_states, environment_states, plan=plan)
    packed = router.prepare(encoded, module_states, environment_states, plan=plan, compact=True)
    ids = plan.packed_prototype_ids
    valid = plan.packed_valid
    full_membership = full.module_membership.gather(
        -1, ids[:, None, :].expand(1, full.module_membership.shape[1], -1)
    ) * valid[:, None, :]
    torch.testing.assert_close(full_membership, packed.module_membership, rtol=0.0, atol=1.0e-6)
    full_control = full.group_control.gather(1, ids[:, :, None].expand(1, -1, 16)) * valid[:, :, None]
    torch.testing.assert_close(full_control, packed.group_control, rtol=0.0, atol=1.0e-6)
    expected_kappa = 1.0 / (0.5 * (full.module_mass + full.environment_mass)).square().sum(dim=-1)
    torch.testing.assert_close(full.kappa, expected_kappa, rtol=0.0, atol=1.0e-6)


def test_core_forward_backward_and_p0_plan_reuse() -> None:
    torch.manual_seed(1409)
    core = InterfaceFieldCore(_config()).train()
    batch = _batch()
    encoded = core.encode_case(batch)
    p0 = core.prepare(encoded, encoded.module_tokens, return_routing_maps=True)
    p1 = core.prepare(
        encoded,
        encoded.module_tokens + 0.1,
        phase_shared_state=p0.phase_shared_state,
        return_routing_maps=True,
    )
    assert p0.phase_shared_state is p1.phase_shared_state
    assert torch.equal(
        p0.interaction_aux["occupancy_group_plan_prototype_ids"],
        p1.interaction_aux["occupancy_group_plan_prototype_ids"],
    )
    assert p0.backend_state["group_control_state"].module_membership.shape[-1] == 12
    assert p0.interaction_aux["occupancy_group_k_plan_per_case"].ge(1).all()
    assert torch.isfinite(p0.interaction_aux["occupancy_group_kappa"]).all()

    read = core.read(p1, batch.query_xy, return_routing_maps=True)
    assert torch.isfinite(read.context).all()
    query_assignment = read.interaction_aux["group_control_query_routing"]
    torch.testing.assert_close(
        query_assignment.sum(dim=-1),
        torch.ones_like(query_assignment[..., 0]),
        rtol=0.0,
        atol=1.0e-5,
    )
    assert "occupancy_group_module_mask_support" in read.interaction_aux
    loss = read.context.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in core.backend.parameters()
        if parameter.requires_grad
    )


def test_field_full_width_and_packed_readers_have_same_context() -> None:
    core = InterfaceFieldCore(_config()).eval()
    batch = _batch(seed=1410, queries=4)
    encoded = core.encode_case(batch)
    p0 = core.prepare(encoded, encoded.module_tokens)
    full = core.read(p0, batch.query_xy).context
    core.backend.set_execution_mode("packed")
    packed_state = core.prepare(
        encoded,
        encoded.module_tokens,
        phase_shared_state=p0.phase_shared_state,
    )
    packed = core.read(packed_state, batch.query_xy).context
    torch.testing.assert_close(full, packed, rtol=1.0e-5, atol=1.0e-6)


def test_mixed_batch_nonconsecutive_plan_ids_preserve_full_packed_values_and_gradients() -> None:
    (
        encoded,
        module_states,
        environment_states,
        router,
        plan,
    ) = _mixed_encoded_and_plan()
    assert plan.prototype_ids.tolist() == [
        [-1, -1, 2, -1, -1, -1, -1, -1, -1, 9, -1, -1],
        [-1, 1, -1, -1, -1, -1, -1, 7, -1, -1, -1, 11],
    ]
    assert plan.packed_prototype_ids.tolist() == [[2, 9, 0], [1, 7, 11]]
    assert plan.packed_valid.tolist() == [[True, True, False], [True, True, True]]

    full = router.prepare(encoded, module_states, environment_states, plan=plan)
    packed = router.prepare(
        encoded,
        module_states,
        environment_states,
        plan=plan,
        compact=True,
    )
    ids = plan.packed_prototype_ids
    valid = plan.packed_valid
    full_membership = full.module_membership.gather(
        -1, ids[:, None, :].expand(2, module_states.shape[1], -1)
    ) * valid[:, None, :]
    full_control = full.group_control.gather(
        1, ids[:, :, None].expand(2, -1, full.group_control.shape[-1])
    ) * valid[:, :, None]
    torch.testing.assert_close(full_membership, packed.module_membership, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(full_control, packed.group_control, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(full.kappa, packed.kappa, rtol=0.0, atol=1.0e-6)

    receivers = encoded.module_centers[:, :2]
    full_route = router.route_queries(encoded, full, receivers)
    packed_route = router.route_queries(encoded, packed, receivers)
    full_assignment = full_route.assignment.gather(
        -1, ids[:, None, :].expand(2, receivers.shape[1], -1)
    ) * valid[:, None, :]
    torch.testing.assert_close(full_assignment, packed_route.assignment, rtol=1.0e-5, atol=1.0e-6)

    group_weights = torch.arange(
        1.0,
        float(packed.group_control.shape[1]) + 1.0,
        device=packed.group_control.device,
    )[None, :, None]
    route_weights = torch.arange(
        1.0,
        float(packed_route.assignment.shape[1]) + 1.0,
        device=packed_route.assignment.device,
    )[None, :, None]
    full_loss = (
        (full_control * group_weights).sum()
        + (full_assignment * route_weights).sum()
    )
    packed_loss = (
        (packed.group_control * group_weights).sum()
        + (packed_route.assignment * route_weights).sum()
    )
    full_grad = torch.autograd.grad(full_loss, router.group_codes, retain_graph=True)[0]
    packed_grad = torch.autograd.grad(packed_loss, router.group_codes)[0]
    torch.testing.assert_close(full_grad, packed_grad, rtol=1.0e-5, atol=1.0e-6)

    active_ids = torch.tensor([1, 2, 7, 9, 11])
    active_grad = full_grad[active_ids]
    assert torch.isfinite(active_grad).all()
    assert not torch.allclose(active_grad[0], active_grad[1])


def test_registered_capacity_padding_is_closed_under_inactive_prototype_changes() -> None:
    encoded, module_states, environment_states, router, plan = _mixed_encoded_and_plan()
    before = router.prepare(encoded, module_states, environment_states, plan=plan)
    receivers = encoded.module_centers[:, :2]
    before_route = router.route_queries(encoded, before, receivers)

    inactive_ids = torch.tensor([0, 3, 4, 5, 6, 8, 10])
    with torch.no_grad():
        router.group_codes[inactive_ids] += torch.linspace(
            0.5,
            3.5,
            inactive_ids.numel(),
            device=router.group_codes.device,
        )[:, None]
    after = router.prepare(encoded, module_states, environment_states, plan=plan)
    after_route = router.route_queries(encoded, after, receivers)

    ids = plan.packed_prototype_ids
    valid = plan.packed_valid
    before_control = before.group_control.gather(
        1, ids[:, :, None].expand(2, -1, before.group_control.shape[-1])
    ) * valid[:, :, None]
    after_control = after.group_control.gather(
        1, ids[:, :, None].expand(2, -1, after.group_control.shape[-1])
    ) * valid[:, :, None]
    before_assignment = before_route.assignment.gather(
        -1, ids[:, None, :].expand(2, receivers.shape[1], -1)
    ) * valid[:, None, :]
    after_assignment = after_route.assignment.gather(
        -1, ids[:, None, :].expand(2, receivers.shape[1], -1)
    ) * valid[:, None, :]
    torch.testing.assert_close(before_control, after_control, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(before_assignment, after_assignment, rtol=0.0, atol=1.0e-6)
    assert not bool(plan.planned_mask[:, inactive_ids].any())
    assert torch.equal(before.phase_occupied, after.phase_occupied)
    assert torch.equal(before.packed_valid, plan.planned_mask)
