"""Focused checks for the NStage2 group-mediated coarse source."""

from __future__ import annotations

from dataclasses import replace

import torch

from honf_forward_core.interface_fields.common import SharedInterfaceContext
from honf_forward_core.interface_fields.group_operator import (
    PackedCoarseGroupSources,
    SparseInterfaceHONF,
    packed_coarse_group_sources,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase

HIDDEN = 8


def _case_fixture(dtype: torch.dtype = torch.float64):
    generator = torch.Generator().manual_seed(1901)
    centers = torch.tensor(
        [
            [[0.20, 0.25], [1.45, 0.75], [2.60, 1.25]],
            [[0.20, 0.25], [1.45, 0.75], [2.60, 1.25]],
        ],
        dtype=dtype,
    )
    offsets = torch.tensor(
        [[[-0.12, -0.08], [0.12, -0.08], [0.0, 0.12]]], dtype=dtype
    )
    ports = centers[:, :, None, :] + offsets
    present = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype)
    features = torch.randn((2, 3, 3), generator=generator, dtype=dtype)
    env_coords = torch.tensor(
        [
            [[-0.25, 0.10], [0.40, 0.35], [1.20, 0.80], [2.20, 1.35]],
            [[-0.25, 0.10], [0.40, 0.35], [1.20, 0.80], [2.20, 1.35]],
        ],
        dtype=dtype,
    )
    env_tokens = torch.randn((2, 4, HIDDEN), generator=generator, dtype=dtype)
    env_weights = torch.tensor(
        [[0.20, 0.30, 0.35, 0.15], [0.15, 0.35, 0.30, 0.20]], dtype=dtype
    )
    encoded = EncodedInterfaceCase(
        module_tokens=torch.randn((2, 3, HIDDEN), generator=generator, dtype=dtype),
        env_tokens=env_tokens,
        global_token=torch.randn((2, HIDDEN), generator=generator, dtype=dtype),
        module_centers=centers,
        env_coords=env_coords,
        module_present=present,
        module_features=features,
        env_features=None,
        env_weights=env_weights,
        coordinate_scale=torch.tensor([[[3.0, 2.0]], [[3.0, 2.0]]], dtype=dtype),
    )
    return encoded, ports, present


def _operator_and_state():
    encoded, ports, present = _case_fixture()
    operator = SparseInterfaceHONF(
        hidden_dim=HIDDEN,
        message_hidden_dim=12,
        fourier_frequencies=2,
        support_spacing_factor=1.0,
        group_read_mode="geometry_envelope_attention",
    ).double()
    cache = operator.build_layout(encoded, ports, module_radius=1.0)
    module_states = torch.randn((2, 3, HIDDEN), dtype=torch.float64, requires_grad=True)
    state = operator.prepare(encoded, module_states, cache)
    return encoded, ports, present, operator, module_states, state


def _coarse_context(*, source: str) -> SharedInterfaceContext:
    return SharedInterfaceContext(
        hidden_dim=HIDDEN,
        field_dim=5,
        num_heads=2,
        coarse_latent_count=3,
        coarse_blocks=1,
        local_radius_factor=2.5,
        fourier_frequencies=2,
        coarse_module_source=source,
    ).double()


def test_historical_coarse_source_keeps_ownership_and_arithmetic() -> None:
    torch.manual_seed(1902)
    context = _coarse_context(source="module_states").eval()
    assert hasattr(context, "coarse_module_attention")
    assert not hasattr(context, "coarse_group_attention")
    state_keys = set(context.state_dict())
    assert any(key.startswith("coarse_module_attention.") for key in state_keys)
    assert not any(key.startswith("coarse_group_attention.") for key in state_keys)

    modules = torch.randn((2, 3, HIDDEN), dtype=torch.float64)
    env = torch.randn((2, 4, HIDDEN), dtype=torch.float64)
    present = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    weights = torch.tensor(
        [[0.20, 0.30, 0.35, 0.15], [0.15, 0.35, 0.30, 0.20]], dtype=torch.float64
    )
    expected = context.prepare_coarse(modules, env, present, weights)
    ignored = context.prepare_coarse(
        modules,
        env,
        present,
        weights,
        torch.randn((2, 7, HIDDEN), dtype=torch.float64),
        torch.ones((2, 7), dtype=torch.float64),
        torch.ones((2, 7), dtype=torch.bool),
    )
    torch.testing.assert_close(expected, ignored, rtol=0.0, atol=0.0)


def test_group_sources_pack_batches_and_preserve_footprint_mass_under_permutation() -> None:
    encoded, ports, present, operator, module_states, state = _operator_and_state()
    payload = packed_coarse_group_sources(state)
    method_payload = operator.packed_coarse_group_sources(state)
    assert isinstance(payload, PackedCoarseGroupSources)
    torch.testing.assert_close(payload.group_states, method_payload.group_states)
    torch.testing.assert_close(payload.occupancy, method_payload.occupancy)
    assert payload.group_states.requires_grad
    assert payload.group_states.device == state.group_state.device
    assert payload.valid.dtype == torch.bool

    for batch_index in range(2):
        group_count = int(state.cache.layout.case_group_offsets[batch_index + 1]) - int(
            state.cache.layout.case_group_offsets[batch_index]
        )
        assert int(payload.valid[batch_index].sum()) == group_count
        torch.testing.assert_close(
            payload.occupancy[batch_index, payload.valid[batch_index]].sum(),
            present[batch_index].sum(),
            rtol=2.0e-6,
            atol=2.0e-6,
        )
        assert torch.equal(
            payload.occupancy[batch_index, ~payload.valid[batch_index]],
            torch.zeros_like(payload.occupancy[batch_index, ~payload.valid[batch_index]]),
        )

    # Slot permutation changes no physical source: support keys and the
    # pre-membership occupancy remain the same after carrying all module data
    # along with the permutation.
    permutation = torch.tensor([1, 0, 2])
    permuted_encoded = replace(
        encoded,
        module_tokens=encoded.module_tokens[:, permutation],
        module_centers=encoded.module_centers[:, permutation],
        module_present=encoded.module_present[:, permutation],
        module_features=encoded.module_features[:, permutation],
    )
    permuted_ports = ports[:, permutation]
    permuted_states = module_states[:, permutation].detach().requires_grad_(True)
    permuted_cache = operator.build_layout(permuted_encoded, permuted_ports, module_radius=1.0)
    permuted_state = operator.prepare(permuted_encoded, permuted_states, permuted_cache)
    torch.testing.assert_close(state.cache.layout.lattice_keys, permuted_state.cache.layout.lattice_keys)
    torch.testing.assert_close(state.cache.layout.occupancy, permuted_state.cache.layout.occupancy)


def test_group_source_gradient_is_conditional_on_live_group_states() -> None:
    encoded, _ports, _present, _operator, module_states, state = _operator_and_state()
    payload = packed_coarse_group_sources(state)
    context = _coarse_context(source="group_states").eval()
    modules = module_states
    env = encoded.env_tokens.detach()
    present = encoded.module_present
    weights = encoded.env_weights
    coefficients = torch.randn((2, 3, HIDDEN), dtype=torch.float64)

    live = context.prepare_coarse(
        modules,
        env,
        present,
        weights,
        payload.group_states,
        payload.occupancy,
        payload.valid,
    )
    live_gradient = torch.autograd.grad((live * coefficients).sum(), modules, retain_graph=True)[0]
    assert torch.isfinite(live_gradient).all()
    assert torch.count_nonzero(live_gradient) > 0

    detached_payload = PackedCoarseGroupSources(
        group_states=payload.group_states.detach(),
        occupancy=payload.occupancy,
        valid=payload.valid,
    )
    detached = context.prepare_coarse(
        modules,
        env,
        present,
        weights,
        detached_payload.group_states,
        detached_payload.occupancy,
        detached_payload.valid,
    )
    detached_gradient = torch.autograd.grad(
        (detached * coefficients).sum(), modules, allow_unused=True
    )[0]
    assert detached_gradient is None or torch.equal(
        detached_gradient, torch.zeros_like(detached_gradient)
    )


def test_group_source_intervention_preserves_environmental_background_and_local_empty_read() -> None:
    encoded, _ports, _present, operator, module_states, state = _operator_and_state()
    payload = packed_coarse_group_sources(state)
    context = _coarse_context(source="group_states").eval()
    modules = module_states.detach()
    env = encoded.env_tokens.detach().clone().requires_grad_(True)

    without_groups = context.prepare_coarse(
        modules,
        env,
        encoded.module_present,
        encoded.env_weights,
        group_source_enabled=False,
    )
    with_groups = context.prepare_coarse(
        modules,
        env,
        encoded.module_present,
        encoded.env_weights,
        payload.group_states,
        payload.occupancy,
        payload.valid,
    )
    assert not torch.allclose(without_groups, with_groups)
    env_gradient = torch.autograd.grad(without_groups.square().sum(), env)[0]
    assert torch.isfinite(env_gradient).all()
    assert torch.count_nonzero(env_gradient) > 0

    receivers = torch.full((2, 1, 2), 100.0, dtype=torch.float64)
    receiver_features = torch.cat([receivers, receivers.square(), torch.sin(receivers)], dim=-1)
    context_read, aux = operator.read(state, encoded, receivers, receiver_features)
    assert torch.equal(context_read, torch.zeros_like(context_read))
    assert torch.equal(
        aux["group_read_geometric_availability"],
        torch.zeros_like(aux["group_read_geometric_availability"]),
    )
