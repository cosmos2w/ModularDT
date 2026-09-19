"""CPU contracts for the Run-1406 low-dimensional group-control backend."""

from __future__ import annotations

import inspect
from dataclasses import replace

import torch

from honf_forward_core.interface_fields.dense_pairwise import DensePairwiseField
from honf_forward_core.interface_fields.group_control_pairwise import (
    GroupControlPairwiseField,
)
from honf_forward_core.interface_fields.group_control_router import (
    GroupQueryRoute,
    LowDimensionalGroupRouter,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase

HIDDEN = 8
MESSAGE = 12
HEADS = 2
FOURIER = 2
GROUPS = 6
CONTROL = 4


def _encoded(
    *,
    batch: int = 2,
    modules: int = 4,
    environments: int = 5,
    all_active: bool = False,
) -> EncodedInterfaceCase:
    generator = torch.Generator().manual_seed(1406001 + batch + modules + environments)
    module_present = torch.ones(batch, modules)
    if not all_active:
        module_present[0, -1] = 0.0
        if batch > 1:
            module_present[1, 1::2] = 0.0
    return EncodedInterfaceCase(
        module_tokens=torch.randn(batch, modules, HIDDEN, generator=generator),
        env_tokens=torch.randn(batch, environments, HIDDEN, generator=generator),
        global_token=torch.randn(batch, HIDDEN, generator=generator),
        module_centers=torch.randn(batch, modules, 2, generator=generator),
        env_coords=torch.randn(batch, environments, 2, generator=generator),
        module_present=module_present,
        module_features=torch.randn(batch, modules, 3, generator=generator),
        env_features=torch.randn(batch, environments, 3, generator=generator),
        env_weights=torch.rand(batch, environments, generator=generator) + 0.2,
        coordinate_scale=torch.ones(1, 1, 2),
    )


def _field(*, query_tile_size: int = 3) -> GroupControlPairwiseField:
    return GroupControlPairwiseField(
        HIDDEN,
        MESSAGE,
        HEADS,
        FOURIER,
        group_count=GROUPS,
        group_control_dim=CONTROL,
        query_tile_size=query_tile_size,
        source_tile_size=4,
    )


def _route(field: GroupControlPairwiseField, batch: int, queries: int, group: int = 0) -> GroupQueryRoute:
    assignment = torch.zeros(batch, queries, GROUPS)
    assignment[..., group] = 1.0
    return GroupQueryRoute(
        query_control=torch.zeros(batch, queries, field.group_control_dim),
        assignment=assignment,
        logits=torch.zeros_like(assignment),
    )


def test_router_entmax_support_empty_modules_and_bounded_controls() -> None:
    encoded = _encoded()
    router = LowDimensionalGroupRouter(
        HIDDEN,
        group_count=GROUPS,
        control_dim=CONTROL,
        fourier_frequencies=FOURIER,
    )
    module_states = torch.randn_like(encoded.module_tokens)
    prepared = router.prepare(encoded, module_states, encoded.env_tokens)
    route = router.route_queries(encoded, prepared, torch.randn(2, 7, 2))

    active = encoded.module_present > 0.5
    torch.testing.assert_close(
        prepared.module_membership[~active],
        torch.zeros_like(prepared.module_membership[~active]),
        rtol=0.0,
        atol=0.0,
    )
    active_sums = prepared.module_membership[active].sum(dim=-1)
    torch.testing.assert_close(
        active_sums,
        torch.ones_like(active_sums),
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        prepared.environment_membership.sum(dim=-1),
        torch.ones(2, encoded.env_coords.shape[1]),
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        route.assignment.sum(dim=-1),
        torch.ones(2, 7),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert torch.all(prepared.group_control.abs() <= 1.0)
    assert torch.isfinite(prepared.group_control).all()


def test_collapsed_moments_match_explicit_group_sum_and_have_finite_gradients() -> None:
    torch.manual_seed(1406002)
    encoded = _encoded(all_active=True)
    router = LowDimensionalGroupRouter(
        HIDDEN,
        group_count=GROUPS,
        control_dim=CONTROL,
        fourier_frequencies=FOURIER,
    )
    prepared = router.prepare(encoded, encoded.module_tokens, encoded.env_tokens)
    route = router.route_queries(encoded, prepared, torch.randn(2, 6, 2))
    alpha = route.assignment
    h = prepared.group_control
    for membership in (
        prepared.module_membership,
        prepared.environment_membership,
    ):
        rho = torch.einsum("bqk,bsk->bqs", alpha, membership)
        moment = torch.einsum("bqk,bsk,bkd->bqsd", alpha, membership, h)
        explicit = sum(
            alpha[..., group, None, None]
            * membership[:, None, :, group, None]
            * h[:, None, None, group, :]
            for group in range(GROUPS)
        )
        torch.testing.assert_close(moment, explicit, rtol=1.0e-5, atol=1.0e-6)
        assert torch.all(moment.abs() <= rho[..., None] + 1.0e-6)

    loss = route.assignment.square().sum() + prepared.group_control.square().sum()
    loss.backward()
    gradients = [parameter.grad for parameter in router.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_environment_head_contraction_and_partial_multi_tile_parity() -> None:
    torch.manual_seed(1406003)
    encoded = _encoded()
    field = _field(query_tile_size=3).eval()
    module_states = torch.randn_like(encoded.module_tokens)
    state = field.prepare(encoded, module_states)
    queries = 11
    receivers = torch.randn(2, queries, 2)
    receiver_features = torch.randn(2, queries, HIDDEN)
    route = _route(field, 2, queries)
    controls = state["group_control_state"]
    rho, zeta = field._environment_control_tile(state, route, 0, queries, 0, 5)
    expected_rho = torch.einsum(
        "bqk,bek->bqe", route.assignment, controls.environment_membership
    )
    expected_zeta = torch.einsum(
        "bqk,bek,bkh->bqeh",
        route.assignment,
        controls.environment_membership,
        state["environment_head_control"],
    ).permute(0, 3, 1, 2)
    torch.testing.assert_close(rho, expected_rho, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(zeta, expected_zeta, rtol=1.0e-5, atol=1.0e-6)

    field.query_tile_size = queries
    full, full_aux = field._read_environment(
        state, encoded, receivers, receiver_features, route
    )
    field.query_tile_size = 3
    tiled, tiled_aux = field._read_environment(
        state, encoded, receivers, receiver_features, route
    )
    torch.testing.assert_close(full, tiled, rtol=2.0e-5, atol=2.0e-6)
    assert full_aux["group_control_environment_complete_support"].item() == 0.0
    torch.testing.assert_close(
        full_aux["group_control_environment_unique_pairs"],
        tiled_aux["group_control_environment_unique_pairs"],
        rtol=0.0,
        atol=0.0,
    )

    module_context, module_aux = field._read_module(state, encoded, receivers, route)
    assert module_context.shape == (2, queries, HIDDEN)
    assert module_aux["group_control_module_complete_support"].item() == 0.0
    torch.testing.assert_close(
        module_aux["group_control_module_fine_forward_rows"],
        module_aux["group_control_module_unique_pairs"],
        rtol=0.0,
        atol=0.0,
    )
    assert module_aux["group_control_module_padded_rows"].item() == 0.0


def test_first_affine_split_matches_unsplit_dense_qm() -> None:
    torch.manual_seed(1406004)
    encoded = _encoded(all_active=True)
    field = _field().eval()
    module_states = torch.randn_like(encoded.module_tokens)
    state = field.prepare(encoded, module_states)
    receivers = torch.randn(2, 5, 2)
    relative = (receivers[:, :, None, :] - encoded.module_centers[:, None, :, :]) / encoded.coordinate_scale
    relative_features = field.relative_fourier(relative)
    source = state["module_tokens"][:, None, :, :].expand(-1, receivers.shape[1], -1, -1)
    global_features = encoded.global_token[:, None, None, :].expand_as(source)
    unsplit = field.query_module_message(
        torch.cat([source, relative_features, global_features], dim=-1)
    )
    relative_weight, _ = field._module_first_slices()
    split_hidden = torch.nn.functional.gelu(
        state["module_first_affine"][:, None, :, :]
        + torch.nn.functional.linear(relative_features, relative_weight, None)
    )
    split = field._module_tail(split_hidden)
    torch.testing.assert_close(unsplit, split, rtol=2.0e-5, atol=2.0e-6)


def _copy_dense_parameters(dense: DensePairwiseField, grouped: GroupControlPairwiseField) -> None:
    grouped_parameters = dict(grouped.named_parameters())
    for name, source in dense.named_parameters():
        target = grouped_parameters.get(name)
        if target is None:
            continue
        try:
            shape = target.shape
        except RuntimeError:
            continue
        if shape == source.shape:
            target.data.copy_(source.data)


def test_uniform_control_off_reproduces_dense_main_readers_including_biases() -> None:
    torch.manual_seed(1406005)
    encoded = _encoded(batch=1, all_active=True)
    module_states = torch.randn_like(encoded.module_tokens)
    receivers = torch.randn(1, 6, 2)
    receiver_features = torch.randn(1, 6, HIDDEN)
    dense = DensePairwiseField(HIDDEN, MESSAGE, HEADS, FOURIER).eval()
    dense_state = dense.prepare(encoded, module_states)
    dense.read(dense_state, encoded, receivers, receiver_features)
    grouped = _field(query_tile_size=3).eval()
    grouped_state = grouped.prepare(encoded, module_states)
    grouped.read(grouped_state, encoded, receivers, receiver_features)
    _copy_dense_parameters(dense, grouped)
    with torch.no_grad():
        grouped.module_control_gain.weight.zero_()
        grouped.environment_value_control.weight.zero_()
        grouped.environment_score_control.weight.zero_()
    dense_state = dense.prepare(encoded, module_states)
    grouped_state = grouped.prepare(encoded, module_states)
    controls = grouped_state["group_control_state"]
    uniform_m = torch.full_like(controls.module_membership, 1.0 / GROUPS)
    uniform_e = torch.full_like(controls.environment_membership, 1.0 / GROUPS)
    controls = replace(
        controls,
        module_membership=uniform_m,
        environment_membership=uniform_e,
        module_mass=torch.full_like(controls.module_mass, 1.0 / GROUPS),
        environment_mass=torch.full_like(controls.environment_mass, 1.0 / GROUPS),
        group_control=torch.zeros_like(controls.group_control),
    )
    grouped_state["group_control_state"] = controls
    key, value, source_control, head_control, head_source_control = grouped._prepare_environment_bank(
        grouped_state["env_tokens"], controls
    )
    grouped_state.update(
        environment_keys=key,
        environment_values=value,
        environment_source_group_control=source_control,
        environment_head_control=head_control,
        environment_head_source_control=head_source_control,
    )
    uniform_route = GroupQueryRoute(
        torch.zeros(1, receivers.shape[1], CONTROL),
        torch.full((1, receivers.shape[1], GROUPS), 1.0 / GROUPS),
        torch.zeros(1, receivers.shape[1], GROUPS),
    )
    dense_module = dense.read_module(dense_state, encoded, receivers, receiver_features)
    dense_environment, _ = dense.read_environment(
        dense_state, encoded, receivers, receiver_features
    )
    grouped_module, _ = grouped._read_module(
        grouped_state, encoded, receivers, uniform_route
    )
    grouped_environment, _ = grouped._read_environment(
        grouped_state, encoded, receivers, receiver_features, uniform_route
    )
    torch.testing.assert_close(grouped_module, dense_module, rtol=3.0e-5, atol=3.0e-6)
    torch.testing.assert_close(grouped_environment, dense_environment, rtol=3.0e-5, atol=3.0e-6)


def test_group_control_changes_fine_module_and_environment_content() -> None:
    torch.manual_seed(1406006)
    encoded = _encoded(all_active=True)
    field = _field().eval()
    state = field.prepare(encoded, torch.randn_like(encoded.module_tokens))
    controls = state["group_control_state"]
    route = field._route(state, encoded, torch.randn(2, 4, 2))
    changed_h = controls.group_control + torch.nn.functional.one_hot(
        torch.tensor(0), num_classes=CONTROL
    ).to(controls.group_control)[None, None, :] * 0.35
    changed = replace(controls, group_control=changed_h)
    source_affine = state["module_first_affine"][:, :1]
    relative = (
        torch.zeros(2, 1, 2) - encoded.module_centers[:, :1]
    ) / encoded.coordinate_scale
    relative_features = field.relative_fourier(relative)
    moment = torch.einsum(
        "bqk,bsk,bkd->bqsd",
        route.assignment[:, :1],
        controls.module_membership[:, :1],
        controls.group_control,
    )[:, :, :1]
    changed_moment = torch.einsum(
        "bqk,bsk,bkd->bqsd",
        route.assignment[:, :1],
        controls.module_membership[:, :1],
        changed_h,
    )[:, :, :1]
    psi = field._module_psi(source_affine, relative_features, moment)
    changed_psi = field._module_psi(source_affine, relative_features, changed_moment)
    assert not torch.allclose(psi, changed_psi)
    _, values, _, old_head, _ = field._prepare_environment_bank(
        state["env_tokens"], controls
    )
    _, changed_values, _, changed_head, _ = field._prepare_environment_bank(
        state["env_tokens"], changed
    )
    assert not torch.allclose(values, changed_values)
    assert not torch.allclose(old_head, changed_head)


def test_empty_support_is_exact_zero_and_read_gradients_are_finite() -> None:
    torch.manual_seed(1406007)
    encoded = _encoded()
    field = _field().train()
    module_states = torch.randn_like(encoded.module_tokens, requires_grad=True)
    state = field.prepare(encoded, module_states)
    receivers = torch.randn(2, 8, 2, requires_grad=True)
    receiver_features = torch.randn(2, 8, HIDDEN, requires_grad=True)
    empty = GroupQueryRoute(
        torch.zeros(2, 8, CONTROL),
        torch.zeros(2, 8, GROUPS),
        torch.zeros(2, 8, GROUPS),
    )
    module, _ = field._read_module(state, encoded, receivers, empty)
    environment, _ = field._read_environment(
        state, encoded, receivers, receiver_features, empty
    )
    assert torch.equal(module, torch.zeros_like(module))
    assert torch.equal(environment, torch.zeros_like(environment))

    output, _ = field.read(state, encoded, receivers, receiver_features)
    output.square().mean().backward()
    assert module_states.grad is not None and torch.isfinite(module_states.grad).all()
    assert receivers.grad is not None and torch.isfinite(receivers.grad).all()
    assert receiver_features.grad is not None and torch.isfinite(receiver_features.grad).all()


def test_module_environment_permutations_and_receiver_chunking_preserve_read() -> None:
    torch.manual_seed(1406008)
    encoded = _encoded(all_active=True)
    field = _field(query_tile_size=3).eval()
    module_states = torch.randn_like(encoded.module_tokens)
    receivers = torch.randn(2, 10, 2)
    receiver_features = torch.randn(2, 10, HIDDEN)
    state = field.prepare(encoded, module_states)
    reference, _ = field.read(state, encoded, receivers, receiver_features)

    module_perm = torch.tensor([2, 0, 3, 1])
    module_encoded = replace(
        encoded,
        module_tokens=encoded.module_tokens[:, module_perm],
        module_features=encoded.module_features[:, module_perm],
        module_centers=encoded.module_centers[:, module_perm],
        module_present=encoded.module_present[:, module_perm],
    )
    module_state = module_states[:, module_perm]
    module_read, _ = field.read(
        field.prepare(module_encoded, module_state),
        module_encoded,
        receivers,
        receiver_features,
    )
    torch.testing.assert_close(reference, module_read, rtol=3.0e-5, atol=3.0e-6)

    env_perm = torch.tensor([3, 0, 4, 1, 2])
    env_encoded = replace(
        encoded,
        env_tokens=encoded.env_tokens[:, env_perm],
        env_features=encoded.env_features[:, env_perm],
        env_coords=encoded.env_coords[:, env_perm],
        env_weights=encoded.env_weights[:, env_perm],
    )
    env_read, _ = field.read(
        field.prepare(env_encoded, module_states),
        env_encoded,
        receivers,
        receiver_features,
    )
    torch.testing.assert_close(reference, env_read, rtol=3.0e-5, atol=3.0e-6)

    chunks = []
    for start, stop in ((0, 3), (3, 7), (7, 10)):
        chunk, _ = field.read(
            state,
            encoded,
            receivers[:, start:stop],
            receiver_features[:, start:stop],
        )
        chunks.append(chunk)
    torch.testing.assert_close(
        reference,
        torch.cat(chunks, dim=1),
        rtol=3.0e-5,
        atol=3.0e-6,
    )


def test_environment_quadrature_duplication_and_vanishing_support() -> None:
    torch.manual_seed(1406009)
    encoded = _encoded(batch=1, environments=4, all_active=True)
    duplicate = replace(
        encoded,
        env_tokens=torch.cat([encoded.env_tokens, encoded.env_tokens[:, :1]], dim=1),
        env_features=torch.cat([encoded.env_features, encoded.env_features[:, :1]], dim=1),
        env_coords=torch.cat([encoded.env_coords, encoded.env_coords[:, :1]], dim=1),
        env_weights=torch.cat(
            [encoded.env_weights[:, :1] * 0.5, encoded.env_weights[:, 1:], encoded.env_weights[:, :1] * 0.5],
            dim=1,
        ),
    )
    field = _field().eval()
    module_states = torch.randn_like(encoded.module_tokens)
    receivers = torch.randn(1, 7, 2)
    receiver_features = torch.randn(1, 7, HIDDEN)
    original, _ = field.read(
        field.prepare(encoded, module_states), encoded, receivers, receiver_features
    )
    duplicated, _ = field.read(
        field.prepare(duplicate, module_states), duplicate, receivers, receiver_features
    )
    torch.testing.assert_close(original, duplicated, rtol=4.0e-5, atol=4.0e-6)

    state = field.prepare(encoded, module_states)
    route_one = _route(field, 1, receivers.shape[1])
    norms = []
    for amount in (1.0, 1.0e-3, 1.0e-5, 0.0):
        route = replace(route_one, assignment=route_one.assignment * amount)
        module, _ = field._read_module(state, encoded, receivers, route)
        environment, _ = field._read_environment(
            state, encoded, receivers, receiver_features, route
        )
        norms.append(torch.cat([module, environment], dim=-1).norm())
    assert norms[-1].item() == 0.0
    assert norms[0] > norms[1] > norms[2] >= 0.0
    assert norms[1] <= norms[0] * 2.0e-3
    assert norms[2] <= norms[0] * 2.0e-5


def _uniform_control_state(field: GroupControlPairwiseField, state: dict[str, object]):
    controls = state["group_control_state"]
    module_membership = torch.full_like(controls.module_membership, 1.0 / GROUPS).requires_grad_()
    environment_membership = torch.full_like(
        controls.environment_membership, 1.0 / GROUPS
    ).requires_grad_()
    module_measure = controls.module_measure.detach().clone().requires_grad_()
    environment_measure = controls.environment_measure.detach().clone().requires_grad_()
    group_control = controls.group_control.detach().clone().requires_grad_()
    controls = replace(
        controls,
        module_membership=module_membership,
        environment_membership=environment_membership,
        module_measure=module_measure,
        environment_measure=environment_measure,
        module_mass=torch.full_like(controls.module_mass, 1.0 / GROUPS),
        environment_mass=torch.full_like(controls.environment_mass, 1.0 / GROUPS),
        group_control=group_control,
    )
    state = dict(state)
    state["group_control_state"] = controls
    key, value, source_control, head_control, head_source_control = field._prepare_environment_bank(
        state["env_tokens"], controls
    )
    state.update(
        environment_keys=key,
        environment_values=value,
        environment_source_group_control=source_control,
        environment_head_control=head_control,
        environment_head_source_control=head_source_control,
        module_first_affine=state["module_first_affine"].detach().clone().requires_grad_(),
    )
    return state


def test_sparse_and_complete_paths_match_outputs_and_gradients() -> None:
    torch.manual_seed(1406010)
    encoded = _encoded(batch=1, all_active=True)
    field = _field(query_tile_size=2).eval()
    state = _uniform_control_state(field, field.prepare(encoded, torch.randn_like(encoded.module_tokens)))
    receivers = torch.randn(1, 7, 2, requires_grad=True)
    receiver_features = torch.randn(1, 7, HIDDEN)
    route = GroupQueryRoute(
        torch.zeros(1, 7, CONTROL),
        torch.full((1, 7, GROUPS), 1.0 / GROUPS, requires_grad=True),
        torch.zeros(1, 7, GROUPS),
    )
    controls = state["group_control_state"]
    overlap_m = field._overlap(route.assignment, controls.module_membership)
    overlap_e = field._overlap(route.assignment, controls.environment_membership)
    active_count = encoded.module_present.sum(dim=-1)
    module_complete, _ = field._read_module_complete(
        state,
        encoded,
        receivers,
        route,
        overlap_m,
        source_measure=controls.module_measure,
        active_module_count=active_count,
    )
    module_sparse, _ = field._read_module_partial(
        state,
        encoded,
        receivers,
        route,
        overlap_m,
        source_measure=controls.module_measure,
        active_module_count=active_count,
    )
    environment_complete, _ = field._read_environment_complete(
        state,
        encoded,
        receivers,
        receiver_features,
        route,
        overlap_e,
        source_measure=controls.environment_measure,
    )
    environment_sparse, _ = field._read_environment_partial(
        state,
        encoded,
        receivers,
        receiver_features,
        route,
        overlap_e,
        source_measure=controls.environment_measure,
    )
    complete = module_complete + environment_complete
    sparse = module_sparse + environment_sparse
    torch.testing.assert_close(complete, sparse, rtol=3.0e-5, atol=3.0e-6)
    targets = [
        route.assignment,
        controls.group_control,
        controls.module_membership,
        controls.environment_membership,
        controls.module_measure,
        controls.environment_measure,
        state["module_first_affine"],
        state["environment_values"],
        receivers,
    ]
    complete_grads = torch.autograd.grad(
        complete.square().sum(), targets, retain_graph=True, allow_unused=True
    )
    sparse_grads = torch.autograd.grad(
        sparse.square().sum(), targets, allow_unused=True
    )
    for complete_grad, sparse_grad in zip(complete_grads, sparse_grads):
        assert complete_grad is not None and sparse_grad is not None
        torch.testing.assert_close(complete_grad, sparse_grad, rtol=5.0e-4, atol=5.0e-5)

    support, diagnostics = field._read_module(state, encoded, receivers.detach(), route)
    del support
    assert diagnostics["group_control_module_logical_paths"].item() == (
        GROUPS * diagnostics["group_control_module_unique_pairs"].item()
    )
    assert diagnostics["group_control_module_fine_forward_rows"].item() == diagnostics[
        "group_control_module_unique_pairs"
    ].item()


def test_backend_has_no_direct_group_value_field_branch() -> None:
    module_source = inspect.getsource(GroupControlPairwiseField._module_finalize)
    environment_source = inspect.getsource(GroupControlPairwiseField._read_environment_complete)
    assert "group_value" not in inspect.getsource(GroupControlPairwiseField)
    assert "group_control" not in module_source
    assert "group_control" not in environment_source
