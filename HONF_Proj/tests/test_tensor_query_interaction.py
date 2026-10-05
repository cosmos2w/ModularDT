"""Weighted query-interaction identities and frozen native-reader contracts."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from honf_forward_core.config import InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.interface_fields.tensor_query_interaction import (
    TensorQueryInteractionField,
    freeze_query_interaction_backbone,
    stable_tanh_contrast,
)
from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
from tests.test_tensor_source_group_residual import field as parent_field
from tests.test_tensor_source_group_residual import read, record


def field(mode='joint'):
    return TensorQueryInteractionField(8, 12, 2, 2,
        architecture='native_context_global_control_honf', spatial_dim=2,
        module_characteristic_length=.1, options={'organizer_dim': 16,
            'global_fast_reader': True, 'query_interaction_mode': mode}).double()


def nonzero(model):
    with torch.no_grad():
        for gamma in model.tensor_query_interaction.gamma.values():
            gamma.weight.normal_(std=.1)


@pytest.mark.parametrize('age', [1, 150, 201])
def test_weighted_decomposition_reconstruction_and_both_joint_marginals(age):
    model, encoded = field(), record()
    nonzero(model); model.set_training_progress(epoch=age)
    state = model.prepare(encoded, encoded.module_tokens)
    phase = state['hypergraph_plan'].strategy_data['tensor_phase_actions']
    plan = phase.plan
    assert plan.reference_coords['Q'].shape == encoded.receiver_anchor_coords.shape
    assert torch.equal(plan.reference_roles, encoded.receiver_anchor_roles)
    for tau, kind in [('QM', 'M'), ('QE', 'E')]:
        parts, correction = model.tensor_query_interaction.components(phase, plan.reference_coords['Q'], tau)
        raw = model.tensor_query_interaction.raw_action(plan, phase.gamma[tau], plan.reference_coords['Q'], tau)
        torch.testing.assert_close(raw, parts['C'] + parts['S'] + parts['R'] + parts['I'], atol=2e-12, rtol=2e-10)
        torch.testing.assert_close(correction, parts['S'] + parts['R'] + parts['I'], atol=0, rtol=0)
        for marginal in [(parts['I'] * plan.reference_weights['Q'][..., None, None]).sum(1),
                         (parts['I'] * plan.measures[kind][:, None, :, None]).sum(2)]:
            torch.testing.assert_close(marginal, torch.zeros_like(marginal), atol=2e-12, rtol=0)
        uniform = replace(plan, density={**plan.density, kind: torch.ones_like(plan.density[kind])},
                          mean_density={**plan.mean_density, kind: torch.ones_like(plan.mean_density[kind])})
        flat = model.tensor_query_interaction.phase_actions(uniform, encoded, encoded.module_tokens,
                                                          state['hypergraph_plan'])
        assert torch.count_nonzero(model.tensor_query_interaction.components(flat, encoded.env_coords, tau)[0]['I']) == 0


def test_equal_parameters_zero_attachment_and_full_host_freeze_input_autograd():
    encoded, query = record(), torch.rand(2, 5, 2, dtype=torch.float64)
    torch.manual_seed(5); additive = field('add')
    torch.manual_seed(5); joint = field('joint')
    parent = parent_field(False)
    for model in (additive, joint, parent): read(model, encoded, query)
    for a, b in zip(additive.tensor_query_interaction.parameters(), joint.tensor_query_interaction.parameters()):
        assert torch.equal(a, b)
    for model in (additive, joint):
        model.load_state_dict(parent.state_dict(), strict=False)
        torch.testing.assert_close(read(model, encoded, query), read(parent, encoded, query), atol=2e-12, rtol=2e-10)
    wrapper = nn.Module(); wrapper.core = nn.Module(); wrapper.core.backend = joint
    wrapper.adapter = nn.Linear(8, 8).double()
    trainable = freeze_query_interaction_backbone(wrapper)
    assert trainable and all(n.startswith('core.backend.tensor_query_interaction.') for n in trainable)
    before = {n: p.detach().clone() for n, p in wrapper.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=.01)
    token = encoded.module_tokens.clone().requires_grad_()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        read(joint, replace(encoded, module_tokens=wrapper.adapter(token)), query).square().mean().backward()
        optimizer.step()
    assert token.grad.abs().max() > 0
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in joint.tensor_query_interaction.gamma.parameters())
    assert all(torch.equal(before[n], p) for n, p in wrapper.named_parameters() if n in before)


def test_query_split_permutation_chunking_and_pullback_gradient():
    torch.manual_seed(8); model = field().eval(); nonzero(model)
    model.set_training_progress(epoch=150)
    encoded, query = record(), torch.rand(2, 7, 2, dtype=torch.float64)
    tokens = encoded.env_tokens.clone().requires_grad_()
    encoded = replace(encoded, env_tokens=tokens)
    original = read(model, encoded, query)
    fractions = query.new_tensor([.3, .7]).repeat(encoded.env_coords.shape[1])
    refined = replace(encoded, env_tokens=tokens.repeat_interleave(2, 1),
        env_coords=encoded.env_coords.repeat_interleave(2, 1),
        env_weights=encoded.env_weights.repeat_interleave(2, 1) * fractions,
        receiver_anchor_coords=encoded.receiver_anchor_coords.repeat_interleave(2, 1),
        receiver_anchor_weights=encoded.receiver_anchor_weights.repeat_interleave(2, 1) * fractions,
        receiver_anchor_roles=encoded.receiver_anchor_roles.repeat_interleave(2, 1))
    children = read(model, refined, query)
    torch.testing.assert_close(original, children, atol=2e-12, rtol=2e-10)
    for left, right in zip(torch.autograd.grad(original.square().sum(), (tokens,), retain_graph=True),
                           torch.autograd.grad(children.square().sum(), (tokens,))):
        torch.testing.assert_close(left, right, atol=2e-12, rtol=2e-10)
    perm = torch.tensor([2, 0, 3, 1])
    permuted = replace(encoded, module_tokens=encoded.module_tokens[:, perm], module_centers=encoded.module_centers[:, perm],
                       module_present=encoded.module_present[:, perm], module_features=encoded.module_features[:, perm])
    torch.testing.assert_close(read(model, permuted, query), original, atol=2e-12, rtol=2e-10)
    state = model.prepare(encoded, encoded.module_tokens)
    order = torch.randperm(query.shape[1]); chunks = []
    for chunk in query[:, order].split(2, 1):
        chunks.append(model.read(state, encoded, chunk, torch.cat((chunk, chunk.square(), chunk.sin()), -1))[0])
    torch.testing.assert_close(torch.cat(chunks, 1)[:, order.argsort()], original, atol=2e-12, rtol=2e-10)


def test_once_wrapper_reference_refresh_unchanged_physical_routes_and_interventions():
    torch.manual_seed(18); model = field().eval(); nonzero(model)
    encoded, query = record(), torch.rand(2, 4, 2, dtype=torch.float64)
    state = model.prepare(encoded, encoded.module_tokens)
    shared = state['phase_shared_group_control']
    for index in (1, 2):
        child = model.prepare(encoded, encoded.module_tokens + index,
            interaction_context=SimpleNamespace(phase=f'P{index}'), phase_shared_state=shared)
        assert child['phase_shared_group_control'] is shared
        assert child['phase_shared_group_control'].mean_access is shared.mean_access
    changed = model.prepare(replace(encoded, module_centers=encoded.module_centers + .2), encoded.module_tokens)
    assert changed['phase_shared_group_control'] is not shared
    base = state['hypergraph_plan']; actions = model._prepare_actions(base)
    for tau in ('MM', 'ME', 'EM'):
        receivers = encoded.env_coords if tau == 'EM' else encoded.module_centers
        actual = model._numerical_access(base, receivers, tau, actions)
        expected = TypedHypergraphField._numerical_access(model, base, receivers, tau, actions)
        for name in ('projected', 'weight', 'support'):
            assert torch.equal(getattr(actual, name), getattr(expected, name))
    read(model, encoded, query)
    parent = parent_field(False); read(parent, encoded, query); parent.load_state_dict(
        {n: t for n, t in model.state_dict().items() if not n.startswith('tensor_query_interaction.')})
    model.tensor_query_interaction.intervention = 'zero_corrections'
    torch.testing.assert_close(read(model, encoded, query), read(parent, encoded, query), atol=2e-12, rtol=2e-10)
    model.tensor_query_interaction.intervention = 'normal'
    baseline = read(model, encoded, query)
    with torch.no_grad(): model.control_score.bias.fill_(1e6)
    assert torch.equal(read(model, encoded, query), baseline)
    phase = base.strategy_data['tensor_phase_actions']
    parts, _ = model.tensor_query_interaction.components(phase, query, 'QE')
    model.tensor_query_interaction.intervention = 'zero_joint'
    changed_parts, correction = model.tensor_query_interaction.components(phase, query, 'QE')
    assert torch.equal(parts['S'], changed_parts['S']) and torch.equal(parts['R'], changed_parts['R'])
    torch.testing.assert_close(correction, parts['S'] + parts['R'], atol=0, rtol=0)
    model.tensor_query_interaction.intervention = 'remove_joint_group'
    changed_parts, correction = model.tensor_query_interaction.components(phase, query, 'QE')
    selected = shared.admission.argmax(-1)
    selected_a = (parts['access'] - shared.mean_access[:, None]).gather(
        2, selected[:, None, None].expand(-1, query.shape[1], 1))
    selected_b = (shared.density['E'] - shared.mean_density['E'][..., None])[
        torch.arange(selected.shape[0]), selected]
    selected_gamma = phase.gamma['QE'][torch.arange(selected.shape[0]), selected]
    removed = selected_a[..., None] * selected_b[:, None, :, None] * selected_gamma[:, None, None]
    torch.testing.assert_close(changed_parts['I_used'], parts['I'] - removed, atol=2e-12, rtol=2e-10)
    assert torch.equal(parts['S'], changed_parts['S']) and torch.equal(parts['R'], changed_parts['R'])
    exported = model.export_typed_state(state)
    access = exported['receiver_access'](query, 'QM')
    assert torch.equal(access.support, (encoded.module_present > .5)[:, None].expand_as(access.support))
    assert access.query_components['I'].shape[:3] == access.support.shape
    assert exported['typed_admission']['MM'].shape[-1] == 1


def test_empty_catalogues_positive_reference_fallback_and_factory():
    model = field()
    for encoded in (record(modules=0), replace(record(environments=0),
        receiver_anchor_coords=torch.zeros(2, 1, 2, dtype=torch.float64),
        receiver_anchor_weights=torch.ones(2, 1, dtype=torch.float64),
        receiver_anchor_roles=torch.zeros(2, 1, dtype=torch.long))):
        assert torch.isfinite(read(model, encoded, torch.rand(2, 2, 2, dtype=torch.float64))).all()
    fallback = model.tensor_query_interaction.build_plan(replace(record(), receiver_anchor_coords=None))
    assert 'environmental' in fallback.reference_origin
    with pytest.raises(ValueError, match='positive'):
        model.tensor_query_interaction.build_plan(record(environments=0))
    def config(options):
        return UnifiedForwardConfig(forward_architecture='native_context_global_control_honf', hidden_dim=8,
            field_dim=3, spatial_dim=2, interface_model=InterfaceFieldConfig(message_hidden_dim=12,
            attention_heads=2, hypergraph_options={'organizer_dim': 16, 'global_fast_reader': True, **options}))
    absent, disabled = InterfaceFieldCore(config({})), InterfaceFieldCore(config({'tensor_query_interaction': False}))
    assert type(disabled.backend) is TypedHypergraphField and absent.state_dict().keys() == disabled.state_dict().keys()
    assert isinstance(InterfaceFieldCore(config({'tensor_query_interaction': True})).backend, TensorQueryInteractionField)
    with pytest.raises(ValueError, match='only one'):
        InterfaceFieldCore(config({'tensor_query_interaction': True, 'tensor_source_residual': True}))


@pytest.mark.parametrize('age, fraction', [(0, 0.), (100, 0.), (150, .5), (200, 1.), (201, 1.)])
def test_checkpoint_selection_progress_preserves_fit_age_and_frozen_host(age, fraction):
    model = field(); model.set_training_progress(epoch=age, total_epochs=1000)
    saved = model.selection_state()
    restored = field(); restored.set_training_progress(**saved)
    assert restored.selection_state() == {'epoch': age, 'total_epochs': 1000}
    assert restored.epoch == 1000  # Global organizer itself is stateless in age.
    plan = restored.tensor_query_interaction.build_plan(record())
    assert plan.additional_age == age and plan.sparse_fraction == fraction


def test_actual_component_recording_and_statistics_without_extra_physical_reads():
    from honf_forward_core.evaluation.organization_statistics import TypedOrganizationStatistics
    from honf_forward_core.evaluation.typed_work_evidence import TypedWorkEvidenceRecorder
    model, encoded = field().eval(), record()
    nonzero(model)
    query = torch.rand(2, 3, 2, dtype=torch.float64)
    reference = read(model, encoded, query)
    with TypedWorkEvidenceRecorder(model) as recorder:
        result = read(model, encoded, query)
    assert torch.equal(result, reference)
    assert recorder.arrays['phase/P0/query_interaction/reference_ids'].shape == encoded.receiver_anchor_weights.shape
    assert recorder.arrays['phase/P0/query_interaction/mode'].item() == 'joint'
    prefixes = [name.removesuffix('/preactivation') for name in recorder.arrays if name.endswith('/preactivation')]
    assert prefixes and all('/QM/' in p or '/QE/' in p for p in prefixes)
    for prefix in prefixes:
        assert prefix + '/query_components/I' in recorder.arrays
        assert prefix + '/effective_gain' in recorder.arrays
    with TypedOrganizationStatistics(model) as statistics:
        result = read(model, encoded, query)
    assert torch.equal(result, reference)
    summary = statistics.summary()
    assert summary['phases']['P0']['query_interaction']['reference_rows'] == encoded.env_coords.shape[1]
    assert any(row.get('query_interaction', {}).get('rms', {}).get('I', 0) > 0
               for row in summary['native_routes'].values())


def test_constant_access_has_no_joint_and_additive_uses_exactly_separable_preactivation():
    model, encoded = field('add'), record()
    nonzero(model)
    residual = model.tensor_query_interaction
    residual.distance_alpha = 0.
    with torch.no_grad():
        for parameter in residual.receiver_key.parameters(): parameter.zero_()
    state = model.prepare(encoded, encoded.module_tokens)
    phase = state['hypergraph_plan'].strategy_data['tensor_phase_actions']
    parts, correction = residual.components(phase, torch.rand(2, 5, 2, dtype=torch.float64), 'QE')
    torch.testing.assert_close(parts['I'], torch.zeros_like(parts['I']), atol=2e-12, rtol=0)
    assert torch.count_nonzero(parts['I_used']) == 0
    torch.testing.assert_close(correction, parts['S'] + parts['R'], atol=0, rtol=0)


def test_query_geometry_and_fresh_input_reference_gradients_match_finite_difference():
    torch.manual_seed(34); model = field().eval(); nonzero(model)
    encoded = record(); model.set_training_progress(epoch=50)
    query = torch.rand(2, 3, 2, dtype=torch.float64, requires_grad=True)
    centres = encoded.module_centers.clone().requires_grad_()
    output = read(model, replace(encoded, module_centers=centres), query).square().sum()
    q_grad, c_grad = torch.autograd.grad(output, (query, centres))
    epsilon = 1e-6
    dq, dc = torch.zeros_like(query), torch.zeros_like(centres)
    dq[0, 1, 0] = epsilon; dc[0, 0, 1] = epsilon
    finite_q = (read(model, replace(encoded, module_centers=centres), query + dq).square().sum() -
                read(model, replace(encoded, module_centers=centres), query - dq).square().sum()) / (2 * epsilon)
    finite_c = (read(model, replace(encoded, module_centers=centres + dc), query).square().sum() -
                read(model, replace(encoded, module_centers=centres - dc), query).square().sum()) / (2 * epsilon)
    torch.testing.assert_close(finite_q, q_grad[0, 1, 0], atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(finite_c, c_grad[0, 0, 1], atol=1e-7, rtol=1e-5)


def test_stable_outer_tanh_contrast_formula_gradients_and_exact_float32_zero():
    torch.manual_seed(91)
    base = torch.randn(2, 1, 1, 3, dtype=torch.float64, requires_grad=True)
    residual = torch.tanh(torch.randn(2, 7, 9, 3, dtype=torch.float64)).requires_grad_()
    stable = stable_tanh_contrast(torch.tanh(base), residual)
    reference = torch.tanh(base + residual) - torch.tanh(base)
    torch.testing.assert_close(stable, reference, atol=5e-16, rtol=2e-13)
    for actual, expected in zip(torch.autograd.grad(stable.square().sum(), (base, residual), retain_graph=True),
                                torch.autograd.grad(reference.square().sum(), (base, residual))):
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-12)
    b32 = base.detach().float()
    zero = torch.zeros(2, 1024, 12, 3, dtype=torch.float32, requires_grad=True)
    actual = stable_tanh_contrast(torch.tanh(b32), zero)
    assert torch.count_nonzero(actual) == 0
    derivative, = torch.autograd.grad(actual.sum(), zero)
    expected = (1 - torch.tanh(b32).square()).expand_as(zero)
    assert torch.equal(derivative, expected) and derivative.abs().max() > 0
    # The same live derivative also survives actual native query reductions.
    model, encoded = field(), record()
    state = model.prepare(encoded, encoded.module_tokens)
    actions = model._prepare_actions(state['hypergraph_plan'])
    access = model._numerical_access(state['hypergraph_plan'], encoded.env_coords, 'QM', actions)
    assert torch.count_nonzero(access.executed_gain_contrast) == 0
    messages = torch.randn(*access.support.shape, 8, dtype=torch.float64)
    result = model._reduce_messages(messages, access, 'QM')
    grads = torch.autograd.grad(result.square().sum(), tuple(model.tensor_query_interaction.gamma['QM'].parameters()))
    assert any(g.abs().max() > 0 for g in grads)
