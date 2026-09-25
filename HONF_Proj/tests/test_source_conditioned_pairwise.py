"""CPU algebra checks for the static source-conditioned fine reader."""

from __future__ import annotations

from dataclasses import replace

import torch

from honf_forward_core.interface_fields.group_control_pairwise import (
    GroupControlPairwiseField,
)
from honf_forward_core.interface_fields.group_control_router import GroupQueryRoute
from honf_forward_core.interface_fields.source_conditioned_pairwise import (
    SourceConditionedPairwiseField,
)
from honf_forward_core.interface_fields.sparse_incidence_group_control import (
    SparseIncidenceGroupControlPairwiseField,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase

HIDDEN = 8
MESSAGE = 12
HEADS = 2
FOURIER = 2
GROUPS = 12
CONTROL = 16


def _encoded(*, batch: int = 2, modules: int = 4, environments: int = 7) -> EncodedInterfaceCase:
    generator = torch.Generator().manual_seed(150701 + batch + modules + environments)
    present = torch.ones(batch, modules)
    if modules > 1:
        present[0, -1] = 0.0
    return EncodedInterfaceCase(
        module_tokens=torch.randn(batch, modules, HIDDEN, generator=generator),
        env_tokens=torch.randn(batch, environments, HIDDEN, generator=generator),
        global_token=torch.randn(batch, HIDDEN, generator=generator),
        module_centers=torch.randn(batch, modules, 2, generator=generator),
        env_coords=torch.randn(batch, environments, 2, generator=generator),
        module_present=present,
        module_features=torch.randn(batch, modules, 3, generator=generator),
        env_features=torch.randn(batch, environments, 3, generator=generator),
        env_weights=torch.rand(batch, environments, generator=generator) + 0.2,
        coordinate_scale=torch.ones(1, 1, 2),
    )


def _source_field() -> SourceConditionedPairwiseField:
    return SourceConditionedPairwiseField(
        HIDDEN,
        MESSAGE,
        HEADS,
        FOURIER,
        group_count=GROUPS,
        group_control_dim=CONTROL,
        spatial_dim=2,
        query_tile_size=3,
        source_tile_size=4,
    )


def _reference_field() -> SparseIncidenceGroupControlPairwiseField:
    return SparseIncidenceGroupControlPairwiseField(
        HIDDEN,
        MESSAGE,
        HEADS,
        FOURIER,
        group_count=GROUPS,
        group_control_dim=CONTROL,
        spatial_dim=2,
        query_tile_size=3,
        source_tile_size=4,
    )


def _forced_uniform_active_read(
    field: SparseIncidenceGroupControlPairwiseField,
    state: dict[str, object],
    encoded: EncodedInterfaceCase,
    receivers: torch.Tensor,
    receiver_features: torch.Tensor,
) -> torch.Tensor:
    controls = state["group_control_state"]
    active = controls.phase_occupied
    k_active = active.sum(dim=-1).to(receivers.dtype).clamp_min(1.0)
    assignment = active[:, None, :].to(receivers.dtype) / k_active[:, None, None]
    route = GroupQueryRoute(
        query_control=receivers.new_zeros(
            receivers.shape[0], receivers.shape[1], field.group_control_dim
        ),
        assignment=assignment.expand(-1, receivers.shape[1], -1),
        logits=receivers.new_zeros(
            receivers.shape[0], receivers.shape[1], field.group_count
        ),
    )
    module, _ = GroupControlPairwiseField._read_module(
        field, state, encoded, receivers, route, include_diagnostics=False
    )
    environment, _ = GroupControlPairwiseField._read_environment(
        field,
        state,
        encoded,
        receivers,
        receiver_features,
        route,
        include_diagnostics=False,
    )
    return (module + environment) * (
        controls.kappa.to(receivers.dtype) / float(field.group_count)
    )[:, None, None]


def _copy_shared_weights(
    reference: SparseIncidenceGroupControlPairwiseField,
    source: SourceConditionedPairwiseField,
) -> None:
    reference_state = reference.state_dict()
    source_state = source.state_dict()
    omitted = set(reference_state) - set(source_state)
    assert omitted
    assert all(
        key.startswith(("router.query_projection.", "router.query_group_projection."))
        for key in omitted
    )
    assert set(source_state).issubset(reference_state)
    source.load_state_dict({key: reference_state[key] for key in source_state}, strict=True)


def _materialize_pair(
    reference: SparseIncidenceGroupControlPairwiseField,
    source: SourceConditionedPairwiseField,
    encoded: EncodedInterfaceCase,
    module_states: torch.Tensor,
    receivers: torch.Tensor,
    receiver_features: torch.Tensor,
) -> tuple[dict[str, object], dict[str, object]]:
    reference_state = reference.prepare(encoded, module_states)
    source_state = source.prepare(encoded, module_states)
    # Materialize the reference's query-only LazyLinear so the state-dict
    # comparison enumerates every removed parameter explicitly.
    reference.router.route_queries(encoded, reference_state["group_control_state"], receivers)
    reference.read(reference_state, encoded, receivers, receiver_features)
    source.read(source_state, encoded, receivers, receiver_features)
    _copy_shared_weights(reference, source)
    return (
        reference.prepare(encoded, module_states),
        source.prepare(encoded, module_states),
    )


def test_static_backend_has_no_query_router_and_prepares_source_local_controls() -> None:
    torch.manual_seed(150701)
    encoded = _encoded()
    field = _source_field().eval()
    state = field.prepare(encoded, encoded.module_tokens)
    controls = state["source_conditioned_controls"]

    assert not hasattr(field.router, "query_fourier")
    assert not hasattr(field.router, "query_projection")
    assert not hasattr(field.router, "query_group_projection")
    assert not any("router.query_" in name for name, _ in field.named_parameters())
    torch.testing.assert_close(
        state["module_source_u"],
        torch.einsum("bmk,bkd->bmd", controls.module_membership, controls.group_control),
    )
    torch.testing.assert_close(
        state["environment_source_u"],
        torch.einsum(
            "bek,bkd->bed", controls.environment_membership, controls.group_control
        ),
    )
    expected_active = controls.phase_occupied.sum(dim=-1).to(encoded.global_token.dtype)
    torch.testing.assert_close(controls.k_active, expected_active)
    torch.testing.assert_close(
        state["module_source_rho"],
        controls.module_membership.sum(dim=-1) / expected_active[:, None],
    )
    torch.testing.assert_close(
        state["module_source_n"],
        state["module_source_u"] / expected_active[:, None, None],
    )
    assert torch.isfinite(controls.kappa).all()


def test_source_local_reader_matches_forced_uniform_active_reference_and_gradients() -> None:
    torch.manual_seed(150702)
    encoded = _encoded(batch=1, modules=2, environments=2)
    encoded = replace(
        encoded,
        global_token=encoded.global_token.detach().clone().requires_grad_(True),
        module_centers=encoded.module_centers.detach().clone().requires_grad_(True),
        env_coords=encoded.env_coords.detach().clone().requires_grad_(True),
    )
    module_states = encoded.module_tokens.detach().clone().requires_grad_(True)
    receivers = torch.randn(1, 6, 2, requires_grad=True)
    receiver_features = torch.randn(1, 6, HIDDEN, requires_grad=True)
    reference = _reference_field().eval()
    source = _source_field().eval()
    reference_state, source_state = _materialize_pair(
        reference,
        source,
        encoded,
        module_states,
        receivers,
        receiver_features,
    )

    expected = _forced_uniform_active_read(
        reference, reference_state, encoded, receivers, receiver_features
    )
    actual, _ = source.read(source_state, encoded, receivers, receiver_features)
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
    assert source_state["source_conditioned_controls"].k_active.item() < GROUPS
    assert torch.unique(encoded.env_weights).numel() > 1
    assert bool((encoded.module_present == 0.0).any())

    weights = torch.linspace(0.2, 1.1, actual.numel()).reshape_as(actual)
    inputs = (
        module_states,
        encoded.global_token,
        encoded.module_centers,
        encoded.env_coords,
        receivers,
        receiver_features,
    )
    reference_loss = (
        _forced_uniform_active_read(
            reference, reference_state, encoded, receivers, receiver_features
        )
        * weights
    ).sum()
    source_loss = (
        source.read(source_state, encoded, receivers, receiver_features)[0] * weights
    ).sum()
    selected_names = (
        "query_module_message.net.0.weight",
        "query_module_output.bias",
        "module_control_gain.weight",
        "environment_value_control.weight",
        "environment_score_control.weight",
        "env_attention.query.weight",
        "env_attention.key.weight",
        "env_attention.output.bias",
        "router.group_codes",
        "router.group_control.net.0.weight",
    )
    reference_params = dict(reference.named_parameters())
    source_params = dict(source.named_parameters())
    assert all(name in reference_params and name in source_params for name in selected_names)
    reference_targets = inputs + tuple(reference_params[name] for name in selected_names)
    source_targets = inputs + tuple(source_params[name] for name in selected_names)
    reference_grads = torch.autograd.grad(reference_loss, reference_targets, retain_graph=True)
    source_grads = torch.autograd.grad(source_loss, source_targets)
    for reference_grad, source_grad in zip(reference_grads, source_grads, strict=True):
        assert reference_grad is not None and source_grad is not None
        torch.testing.assert_close(source_grad, reference_grad, rtol=5.0e-4, atol=3.0e-6)


def test_source_gains_fold_after_projection_and_empty_module_type_stays_zero() -> None:
    torch.manual_seed(150703)
    encoded = _encoded(batch=1, modules=3, environments=5)
    field = _source_field().eval()
    state = field.prepare(encoded, encoded.module_tokens)
    expected_keys = state["environment_keys_unfolded"] * state[
        "environment_source_score_gain"
    ].permute(0, 2, 1)[..., None]
    torch.testing.assert_close(state["environment_keys"], expected_keys, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        state["environment_source_value_gain"],
        1.0 + torch.tanh(field.environment_value_control(state["environment_source_u"])),
        rtol=0.0,
        atol=0.0,
    )

    empty_encoded = EncodedInterfaceCase(
        **{
            **encoded.__dict__,
            "module_present": torch.zeros_like(encoded.module_present),
        }
    )
    empty_state = field.prepare(empty_encoded, empty_encoded.module_tokens)
    receivers = torch.randn(1, 4, 2)
    features = torch.randn(1, 4, HIDDEN)
    module_context = field._read_module_source_conditioned(
        empty_state, empty_encoded, receivers
    )
    assert torch.count_nonzero(empty_state["module_source_rho"]) == 0
    assert torch.count_nonzero(empty_state["source_conditioned_module_sigma"]) == 0
    assert torch.count_nonzero(module_context) == 0
    output, _ = field.read(empty_state, empty_encoded, receivers, features)
    assert torch.isfinite(output).all()

    empty_environment = replace(
        encoded,
        env_tokens=encoded.env_tokens[:, :0],
        env_coords=encoded.env_coords[:, :0],
        env_features=encoded.env_features[:, :0],
        env_weights=encoded.env_weights[:, :0],
    )
    empty_environment_state = field.prepare(
        empty_environment, empty_environment.module_tokens
    )
    empty_environment_context = field._read_environment_source_conditioned(
        empty_environment_state,
        empty_environment,
        receivers,
        features,
    )
    assert torch.count_nonzero(
        empty_environment_state["source_conditioned_environment_sigma"]
    ) == 0
    assert torch.count_nonzero(empty_environment_context) == 0


def test_query_chunk_boundaries_do_not_change_static_reader() -> None:
    torch.manual_seed(150704)
    encoded = _encoded(batch=1, modules=4, environments=6)
    field = _source_field().eval()
    state = field.prepare(encoded, encoded.module_tokens)
    receivers = torch.randn(1, 11, 2)
    features = torch.randn(1, 11, HIDDEN)
    full, _ = field.read(state, encoded, receivers, features)
    parts = [
        field.read(state, encoded, receivers[:, start:stop], features[:, start:stop])[0]
        for start, stop in ((0, 4), (4, 9), (9, 11))
    ]
    torch.testing.assert_close(torch.cat(parts, dim=1), full, rtol=2.0e-5, atol=2.0e-6)
