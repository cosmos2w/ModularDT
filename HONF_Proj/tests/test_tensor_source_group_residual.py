"""Measured small-native algebra and control-information invariance tests."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from honf_forward_core.config import InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.interface_fields.tensor_source_group_residual import TensorSourceGroupResidualField
from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def test_disabled_tensor_factory_preserves_historical_backend_and_parameter_inventory():
    def config(settings):
        return UnifiedForwardConfig(
            forward_architecture='native_context_global_control_honf', hidden_dim=8,
            field_dim=3, spatial_dim=2,
            interface_model=InterfaceFieldConfig(
                message_hidden_dim=12, attention_heads=2,
                hypergraph_options={'organizer_dim': 16, 'global_fast_reader': True, **settings}),
        )

    absent = InterfaceFieldCore(config({}))
    disabled_config = config({'tensor_source_residual': False})
    disabled = InterfaceFieldCore(disabled_config)
    assert type(disabled.backend) is TypedHypergraphField
    assert absent.state_dict().keys() == disabled.state_dict().keys()
    assert disabled_config.interface_model.hypergraph_options['tensor_source_residual'] is False
    enabled = InterfaceFieldCore(config({'tensor_source_residual': True}))
    assert isinstance(enabled.backend, TensorSourceGroupResidualField)
    wrong = replace(config({'tensor_source_residual': True}), forward_architecture='native_context_tree_honf')
    with pytest.raises(ValueError):
        InterfaceFieldCore(wrong)


def record(modules=4, environments=7):
    generator = torch.Generator().manual_seed(62)
    def rand(*shape): return torch.rand(*shape, generator=generator, dtype=torch.float64)
    present = torch.ones(2, modules, dtype=torch.float64)
    if modules > 1: present[1, -1] = 0
    coords, weights = rand(2, environments, 2), rand(2, environments)
    return EncodedInterfaceCase(rand(2, modules, 8), rand(2, environments, 8), rand(2, 8),
        rand(2, modules, 2), coords, present, rand(2, modules, 3), None, weights,
        torch.ones(1, 1, 2, dtype=torch.float64), receiver_anchor_coords=coords,
        receiver_anchor_weights=weights, receiver_anchor_roles=torch.zeros_like(weights, dtype=torch.long))


def field(tensor=True):
    cls = TensorSourceGroupResidualField if tensor else TypedHypergraphField
    return cls(8, 12, 2, 2, architecture='native_context_global_control_honf', spatial_dim=2,
        module_characteristic_length=.1, options={'organizer_dim': 16, 'global_fast_reader': True,
            **({'tensor_source_residual': True} if tensor else {})}).double()


def read(model, encoded, query):
    return model.read(model.prepare(encoded, encoded.module_tokens), encoded, query,
                      torch.cat((query, query.square(), query.sin()), -1))[0]


@pytest.mark.parametrize('modules', [1, 9])
def test_identity_common_output_gradients_and_actual_adam_steps(modules):
    torch.manual_seed(7)
    encoded = record(modules)
    query = torch.rand(2, 5, 2, dtype=torch.float64)
    parent, child = field(False), field()
    read(parent, encoded, query); read(child, encoded, query)
    with torch.no_grad():
        for gain in parent.control_gain.values(): gain.weight.normal_(std=.05)
        for head in parent.organizer.control_heads.values(): head[-1].weight.normal_(std=.05)
    child.load_state_dict(parent.state_dict(), strict=False)
    results = []
    for model in (parent, child):
        q = query.clone().requires_grad_()
        tokens = encoded.module_tokens.clone().requires_grad_()
        output = read(model, replace(encoded, module_tokens=tokens), q)
        output.square().mean().backward()
        results.append((output.detach(), q.grad, tokens.grad))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    for name, parameter in parent.named_parameters():
        candidate = dict(child.named_parameters())[name]
        if parameter.grad is not None:
            torch.testing.assert_close(candidate.grad, parameter.grad, atol=1e-12, rtol=1e-10)
    optimizer = torch.optim.Adam(child.parameters(), lr=.01)
    initial = {name: p.detach().clone() for name, p in child.tensor_residual.named_parameters()}
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        read(child, encoded, query).square().mean().backward()
        optimizer.step()
    changed = {name for name, p in child.tensor_residual.named_parameters()
               if not torch.equal(initial[name], p)}
    assert any(n.startswith('gamma') for n in changed)
    assert any(n.startswith('collective') for n in changed)
    assert any(n.startswith('admission_head') for n in changed)
    assert any(n.startswith('donor_scorer') for n in changed)


@pytest.mark.parametrize('age', [50, 150, 201])
def test_permutation_unequal_atom_split_and_query_chunking(age):
    torch.manual_seed(19)
    model = field().eval(); model.set_training_progress(epoch=500 + age)
    with torch.no_grad():
        for gamma in model.tensor_residual.gamma.values(): gamma.weight.normal_(std=.1)
    encoded, query = record(), torch.rand(2, 9, 2, dtype=torch.float64)
    result = read(model, encoded, query)
    perm = torch.tensor([2, 0, 3, 1])
    permuted = replace(encoded, module_tokens=encoded.module_tokens[:, perm],
        module_centers=encoded.module_centers[:, perm], module_present=encoded.module_present[:, perm],
        module_features=encoded.module_features[:, perm])
    torch.testing.assert_close(read(model, permuted, query), result, atol=2e-12, rtol=2e-10)
    fractions = query.new_tensor([.3, .7]).repeat(encoded.env_coords.shape[1])
    refined = replace(encoded, env_coords=encoded.env_coords.repeat_interleave(2, 1),
        env_tokens=encoded.env_tokens.repeat_interleave(2, 1),
        env_weights=encoded.env_weights.repeat_interleave(2, 1) * fractions,
        receiver_anchor_coords=encoded.env_coords.repeat_interleave(2, 1),
        receiver_anchor_weights=encoded.env_weights.repeat_interleave(2, 1) * fractions,
        receiver_anchor_roles=encoded.receiver_anchor_roles.repeat_interleave(2, 1))
    refined_result = read(model, refined, query)
    torch.testing.assert_close(refined_result, result, atol=2e-12, rtol=2e-10)
    original = model.tensor_residual.build_plan(encoded); children = model.tensor_residual.build_plan(refined)
    torch.testing.assert_close(original.density['E'], children.density['E'][:, :, ::2], atol=2e-12, rtol=2e-10)
    state = model.prepare(encoded, encoded.module_tokens)
    order = torch.randperm(9)
    pieces = []
    for chunk in query[:, order].split(3, 1):
        pieces.append(model.read(state, encoded, chunk, torch.cat((chunk, chunk.square(), chunk.sin()), -1))[0])
    torch.testing.assert_close(torch.cat(pieces, 1)[:, order.argsort()], result, atol=2e-12, rtol=2e-10)


def test_once_per_wrapper_plan_and_empty_routes():
    model, encoded = field(), record(environments=0)
    state0 = model.prepare(encoded, encoded.module_tokens)
    shared = state0['phase_shared_group_control']
    for p in (1, 2):
        state = model.prepare(encoded, encoded.module_tokens + p, interaction_context=SimpleNamespace(phase=f'P{p}'),
                              phase_shared_state=shared)
        assert state['phase_shared_group_control'] is shared
    with pytest.raises(ValueError, match='P0-owned'):
        model.prepare(encoded, encoded.module_tokens, interaction_context=SimpleNamespace(phase='P1'))
    output = read(model, encoded, torch.rand(2, 3, 2, dtype=torch.float64))
    assert torch.isfinite(output).all()


def test_ordinary_autograd_finite_difference_and_sparse_admission_continuity():
    torch.manual_seed(41)
    model, encoded = field().eval(), record()
    model.set_training_progress(epoch=750)
    with torch.no_grad():
        for gamma in model.tensor_residual.gamma.values(): gamma.weight.normal_(std=.05)
    query = torch.rand(2, 3, 2, dtype=torch.float64, requires_grad=True)
    output = read(model, encoded, query).square().sum()
    gradient, = torch.autograd.grad(output, query)
    epsilon = 1e-6
    delta = torch.zeros_like(query); delta[0, 0, 0] = epsilon
    finite = (read(model, encoded, query + delta).square().sum() -
              read(model, encoded, query - delta).square().sum()) / (2 * epsilon)
    torch.testing.assert_close(finite, gradient[0, 0, 0], atol=1e-7, rtol=1e-5)
    from honf_forward_core.interface_fields.routing_index.sparse_projection import masked_sparsemax
    for offset in (-1e-6, 0., 1e-6):
        pi = masked_sparsemax(torch.tensor([[0., 1. + offset]], dtype=torch.float64))
        plan = model.tensor_residual.build_plan(encoded)
        # Explicit mathematical two-group boundary test of the bounded-access
        # action, independent of a target-selected admission recipe.
        reduced = replace(plan, keys=plan.keys[:, :2], admission=pi.expand(2, -1),
                          density={k: v[:, :2] for k, v in plan.density.items()})
        gamma = torch.ones(2, 2, 1, dtype=torch.float64)
        value = model.tensor_residual.raw_action(reduced, gamma, query, 'QM')
        if offset == -1e-6: before = value
        elif offset == 1e-6:
            assert (value - before).abs().max() < 1e-4


def test_exported_physical_support_and_fixed_membership_excluded_content():
    model, encoded = field().eval(), record()
    model.set_training_progress(epoch=750)
    state = model.prepare(encoded, encoded.module_tokens)
    exported = model.export_typed_state(state)
    query = torch.rand(2, 3, 2, dtype=torch.float64)
    access = exported['receiver_access'](query, 'QE')
    assert access.support.all()  # Physical values remain full native access.
    assert access.edge_access.shape[-1] == encoded.module_tokens.shape[1] + 1
    shared = state['phase_shared_group_control']
    excluded = shared.density['E'] == 0
    assert excluded.any()  # Constructed geometry produces actual sparse donors.
    tokens = encoded.env_tokens.clone().requires_grad_()
    # Freeze memberships to isolate h-content, deliberately removing planner
    # derivatives here only. Full model export names those global paths.
    mass = (shared.density['E'] * shared.measures['E'][:, None]).detach()
    content = torch.bmm(mass, model.tensor_residual.content_encoder['E'](tokens))
    chosen = torch.nonzero(excluded, as_tuple=False)[0]
    case, group, source = chosen.tolist()
    gradient, = torch.autograd.grad(content[case, group].sum(), tokens)
    assert torch.count_nonzero(gradient[case, source]) == 0
    assert 'planning' in exported['dependency_provenance']['global_paths']
    assert 'centering' in exported['dependency_provenance']['global_paths']
    phase = state['hypergraph_plan'].strategy_data['tensor_phase_actions']
    for tau in ('MM', 'ME', 'EM', 'QM', 'QE'):
        assert phase.centering[tau].isfinite().all()


def test_pre_tanh_centering_native_eligibility_and_empty_capacity():
    model = field().eval(); model.set_training_progress(epoch=750)
    with torch.no_grad():
        for gamma in model.tensor_residual.gamma.values(): gamma.weight.normal_(std=.1)
    for encoded in (record(), record(modules=1, environments=0), record(modules=0, environments=0)):
        state = model.prepare(encoded, encoded.module_tokens)
        phase = state['hypergraph_plan'].strategy_data['tensor_phase_actions']
        plan = phase.plan
        assert plan.valid.shape[-1] == encoded.module_tokens.shape[1] + 1
        for tau in ('MM', 'ME', 'EM', 'QM', 'QE'):
            receiver_kind = 'M' if tau in ('MM', 'ME') else 'E' if tau == 'EM' else 'Q'
            source_kind = 'E' if tau in ('ME', 'QE') else 'M'
            raw = model.tensor_residual.raw_action(plan, phase.gamma[tau], plan.reference_coords[receiver_kind], tau)
            mass = plan.reference_weights[receiver_kind][:, :, None] * plan.measures[source_kind][:, None]
            if tau == 'MM': mass = mass * (~torch.eye(mass.shape[-1], dtype=torch.bool))[None]
            centered_integral = ((raw - phase.centering[tau][:, None, None]) * mass[..., None]).sum((1, 2))
            torch.testing.assert_close(centered_integral, torch.zeros_like(centered_integral), atol=1e-12, rtol=0)
            if not mass.any(): assert torch.count_nonzero(phase.centering[tau]) == 0


def test_qe_omits_arbitrary_frozen_constant_score_with_live_residual():
    torch.manual_seed(88)
    model, encoded = field().eval(), record()
    model.set_training_progress(epoch=750)
    with torch.no_grad():
        for gamma in model.tensor_residual.gamma.values(): gamma.weight.normal_(std=.1)
    query = torch.rand(2, 5, 2, dtype=torch.float64)
    reference = read(model, encoded, query)
    state = model.prepare(encoded, encoded.module_tokens)
    actions = model._prepare_actions(state['hypergraph_plan'])
    access = model._numerical_access(state['hypergraph_plan'], query, 'QE', actions)
    assert access.residual_projected[..., :2].abs().max() > 0
    keys_before = set(model.state_dict())
    assert not model.control_score.weight.requires_grad
    with torch.no_grad():
        model.control_score.weight.normal_(mean=100., std=50.)
        model.control_score.bias.copy_(torch.tensor([1e6, -1e6], dtype=torch.float64))
    candidate = read(model, encoded, query)
    torch.testing.assert_close(candidate, reference, atol=0, rtol=0)
    assert set(model.state_dict()) == keys_before
    state = model.prepare(encoded, encoded.module_tokens)
    access = model._numerical_access(state['hypergraph_plan'], query, 'QE',
                                     model._prepare_actions(state['hypergraph_plan']))
    torch.testing.assert_close(access.projected[..., :2], access.residual_projected[..., :2], atol=0, rtol=0)
