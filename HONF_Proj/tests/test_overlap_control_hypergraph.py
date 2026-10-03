"""Typed overlap/local organizer algebra, freshness and restoration gradients."""

from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.dense_pairwise import DensePairwiseField
from honf_forward_core.interface_fields.overlap_control_hypergraph import OverlapControlHypergraph, _source_density
from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
from honf_forward_core.interface_fields.typed_hypergraph_state import (
    masked_max,
    masked_mean,
    pad_geometry,
    smooth_near_envelope,
    source_moments,
    structural_cost,
    structural_pressure,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase
from honf_forward_core.training.hypergraph_shadow import hard_value_soft_hypergraph_forward


def encoded(dimension=2, modules=4, environment=5, batch=2):
    torch.manual_seed(38)
    hidden = 8
    present = torch.ones(batch, modules)
    if modules:
        present[0, -1] = 0
    return EncodedInterfaceCase(
        module_tokens=torch.randn(batch, modules, hidden),
        env_tokens=torch.randn(batch, environment, hidden),
        global_token=torch.randn(batch, hidden),
        module_centers=torch.rand(batch, modules, dimension) * 4,
        env_coords=torch.rand(batch, environment, dimension) * 4,
        module_present=present,
        module_features=torch.randn(batch, modules, 3),
        env_features=torch.randn(batch, environment, 2),
        env_weights=torch.rand(batch, environment) + 0.1,
        coordinate_scale=torch.ones(1, 1, 1, dimension) * 4,
    )


def test_paths_deduplicate_without_k_amplitude_or_value_collapse():
    source = torch.tensor([[[1., 3.], [5., 7.], [11., 13.]]], requires_grad=True)
    measures = torch.tensor([[1., 2., 3.]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    outputs, gradients = [], []
    for groups in (1, 2, 8):
        access = torch.full((1, 2, groups), 1 / groups)
        member = torch.full((1, groups, 3), 1 / 3)
        controls = torch.full((1, groups, 16), 2.)
        result = source_moments(access, member, controls, measures, valid)
        torch.testing.assert_close(result.weight, torch.ones(1, 2, 3))
        torch.testing.assert_close(result.control, torch.full((1, 2, 3, 16), 2.))
        assert int(result.diagnostics['unique_pairs']) == 6
        assert int(result.diagnostics['repeated_paths_removed']) == 6 * (groups - 1)
        # Original source-local values remain in the physical additive read.
        output = (source[:, None] * measures[:, None, :, None] * result.weight[..., None]).sum(dim=2)
        outputs.append(output)
        gradients.append(torch.autograd.grad(output.square().sum(), source)[0])
    for output, gradient in zip(outputs[1:], gradients[1:]):
        torch.testing.assert_close(output, outputs[0])
        torch.testing.assert_close(gradient, gradients[0])


def test_pair_validity_normalization_and_empty_type():
    access = torch.ones(1, 3, 1)
    member = torch.ones(1, 1, 3)
    control = torch.randn(1, 1, 16)
    valid = torch.ones(1, 3, dtype=torch.bool)
    pair_valid = ~torch.eye(3, dtype=torch.bool)[None]
    result = source_moments(access, member, control, torch.ones(1, 3), valid, pair_valid=pair_valid)
    torch.testing.assert_close(result.weight, pair_valid.float())
    assert result.diagnostics['eligible_pairs'] == 6
    empty = source_moments(access, member[..., :0], control, torch.empty(1, 0), valid[..., :0])
    assert empty.weight.shape == (1, 3, 0)
    assert empty.control.shape == (1, 3, 0, 16)
    assert int(empty.diagnostics['unique_pairs']) == 0


def test_masked_reductions_keep_negative_population_and_empty_flags():
    values = torch.tensor([[[-9., -8.], [-4., -5.], [99., 99.]], [[1., 2.], [3., 4.], [5., 6.]]])
    valid = torch.tensor([[True, True, False], [False, False, False]])
    maximum, present = masked_max(values, valid)
    torch.testing.assert_close(maximum, torch.tensor([[-4., -5.], [0., 0.]]))
    assert present.tolist() == [True, False]
    torch.testing.assert_close(masked_mean(values, valid), torch.tensor([[-6.5, -6.5], [0., 0.]]))


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('modules,environment', [(0, 0), (0, 3), (1, 0), (4, 5)])
def test_zero_one_many_sources_finite_and_query_chunk_invariant(dimension, modules, environment):
    case = encoded(dimension, modules, environment)
    model = OverlapControlHypergraph(8, dimension).eval()
    state = model.prepare(case, case.module_tokens, phase='P1')
    queries = torch.rand(2, 7, dimension)
    for mechanism in ('MM', 'ME', 'EM', 'QM', 'QE'):
        result = model.access(state, queries, mechanism)
        chunks = [model.access(state, chunk, mechanism) for chunk in queries.split(3, dim=1)]
        torch.testing.assert_close(result.weight, torch.cat([chunk.weight for chunk in chunks], dim=1))
        torch.testing.assert_close(result.control, torch.cat([chunk.control for chunk in chunks], dim=1))
        assert torch.isfinite(result.weight).all() and torch.isfinite(result.control).all()
        invalid = ~state.source_valid['E' if mechanism in ('ME', 'QE') else 'M']
        assert not result.support.masked_select(invalid[:, None].expand_as(result.support)).any()


def test_module_permutation_and_physical_ids_track_sources():
    case = encoded(batch=1)
    case = replace(case, module_source_ids=torch.tensor([[101, 202, 303, 404]]))
    model = OverlapControlHypergraph(8, 2).eval()
    state = model.prepare(case, case.module_tokens)
    permutation = torch.tensor([2, 0, 3, 1])
    moved = replace(case, module_tokens=case.module_tokens[:, permutation], module_centers=case.module_centers[:, permutation], module_present=case.module_present[:, permutation], module_features=case.module_features[:, permutation], module_source_ids=case.module_source_ids[:, permutation])
    permuted = model.prepare(moved, moved.module_tokens)
    torch.testing.assert_close(state.centres, permuted.centres)
    torch.testing.assert_close(state.admission, permuted.admission)
    for mechanism in ('MM', 'EM', 'QM'):
        torch.testing.assert_close(state.memberships[mechanism][:, :, permutation], permuted.memberships[mechanism])
        torch.testing.assert_close(state.controls[mechanism], permuted.controls[mechanism])
    torch.testing.assert_close(state.source_ids['M'][:, permutation], permuted.source_ids['M'])


def test_padded_tokens_coordinates_and_lengths_do_not_change_valid_actions():
    case = encoded(batch=1)
    model = OverlapControlHypergraph(8, 2, local_access=True).eval()
    state = model.prepare(case, case.module_tokens)
    tokens, centres = case.module_tokens.clone(), case.module_centers.clone()
    tokens[:, -1] = torch.nan
    centres[:, -1] = torch.nan
    lengths = torch.full((1, 4), model.module_characteristic_length)
    lengths[:, -1] = 0
    poisoned = replace(case, module_tokens=tokens, module_centers=centres, module_characteristic_lengths=lengths)
    masked = model.prepare(poisoned, poisoned.module_tokens)
    queries = case.env_coords
    for mechanism in ('QM', 'QE'):
        original = model.access(state, queries, mechanism)
        current = model.access(masked, queries, mechanism)
        torch.testing.assert_close(original.weight, current.weight)
        torch.testing.assert_close(original.control, current.control)


def test_phase_state_freshness_and_target_free_export():
    case = encoded()
    model = OverlapControlHypergraph(8, 2).eval()
    p0 = model.prepare(case, case.module_tokens, phase=0)
    p1 = model.prepare(case, case.module_tokens + 1, phase=1)
    assert p0.phase == 0 and p1.phase == 1
    assert not torch.allclose(p0.memberships['QM'], p1.memberships['QM'])
    exported = model.export(p1)
    assert 'receiver_access' in exported and callable(exported['receiver_access'])
    assert not {'target_field', 'case_id', 'reference', 'hidden_heat'} & exported.keys()
    access = exported['receiver_access'](case.module_centers, 'QM')
    assert access.weight.shape == (2, 4, 4)
    with pytest.raises(ValueError, match='hard/soft mode'):
        model.access(p0, case.module_centers, 'QM', soft=True)


def test_gate_rescue_uses_highest_score_and_is_reported():
    case = encoded(batch=1)
    model = OverlapControlHypergraph(8, 2).eval()
    with torch.no_grad():
        model.admission_gate.network.net[-1].weight.zero_()
        model.admission_gate.network.net[-1].bias.fill_(-100)
    state = model.prepare(case, case.module_tokens)
    assert bool(state.diagnostics['admission_rescue'][0])
    assert int((state.admission > 0).sum()) == 1
    assert state.admission[0, state.strategy_data['gate_logits'][0].argmax()] == 1
    assert torch.isfinite(model.access(state, case.module_centers, 'QE').weight).all()


def test_soft_shadow_restores_omitted_sources_with_useful_task_gradient():
    case = encoded(batch=1)
    model = OverlapControlHypergraph(8, 2).eval()
    hard = model.prepare(case, case.module_tokens)
    soft = model.prepare(case, case.module_tokens, soft=True)
    query = case.module_centers[:, :2]
    hard_access = model.access(hard, query, 'QE')
    assert torch.isfinite(hard_access.weight).all()
    soft_access = model.access(soft, query, 'QE', soft=True)
    # Initial supports can be dense; do not force a K/support histogram.
    assert bool(soft_access.support.all())
    values = torch.arange(1., 6.)[None, None]
    prediction = (soft_access.weight * values * case.env_weights[:, None]).sum(dim=-1)
    gradients = torch.autograd.grad(prediction.square().mean(), (model.source_scores['QE'].net[-1].weight, model.receiver_scores['QE'].net[-1].weight, model.admission_gate.network.net[-1].weight))
    for gradient in gradients:
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    logits = torch.tensor([[[0., -8., -8.]]], requires_grad=True)
    probability = logits.softmax(dim=-1)
    result = source_moments(torch.ones(1, 1, 1), probability, torch.zeros(1, 1, 16), torch.ones(1, 3), torch.ones(1, 3, dtype=torch.bool))
    value = (result.weight * torch.tensor([[[1., 4., 7.]]])).mean()
    derivative = torch.autograd.grad(value, logits)[0][0, 0, 2]
    restored = logits.detach().clone()
    restored[..., 2] += 0.01
    restore = source_moments(torch.ones(1, 1, 1), restored.softmax(dim=-1), torch.zeros(1, 1, 16), torch.ones(1, 3), torch.ones(1, 3, dtype=torch.bool))
    restored_value = (restore.weight * torch.tensor([[[1., 4., 7.]]])).mean()
    assert derivative > 0 and restored_value > value.detach()


def test_local_envelope_preserves_near_fine_access_and_one_union():
    receivers = torch.tensor([[[0., 0.], [1.5, 0.], [2., 0.]]])
    sources = torch.tensor([[[0., 0.]]])
    near = smooth_near_envelope(receivers, sources, torch.ones(1, 1))
    torch.testing.assert_close(near, torch.tensor([[[1.], [.5], [0.]]]))
    result = source_moments(torch.ones(1, 3, 1), torch.zeros(1, 1, 1), torch.ones(1, 1, 16), torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool), near=near)
    torch.testing.assert_close(result.weight, near)
    assert int(result.diagnostics['near_mandatory_pairs']) == 2
    assert result.control.abs().sum() == 0
    with pytest.raises(ValueError, match='positive and finite'):
        smooth_near_envelope(receivers, sources, torch.zeros(1, 1))


def test_deliberately_omitted_fine_environment_source_has_finite_restore_derivative():
    case = encoded(batch=1)
    source_states = case.env_tokens.clone()
    source_states[0, :, 0] = torch.arange(5.)
    case = replace(case, env_tokens=source_states)
    model = OverlapControlHypergraph(8, 2).eval()
    with torch.no_grad():
        model.source_projection['E'].weight.zero_()
        model.source_projection['E'].bias.zero_()
        model.source_projection['E'].weight[0, 0] = 1
        model.proposals.zero_()
        model.proposals[:, 0] = 10
        model.source_scores['QE'].net[-1].weight.zero_()
        model.source_scores['QE'].net[-1].bias.zero_()
    hard = model.prepare(case, case.module_tokens)
    soft = model.prepare(case, case.module_tokens, soft=True)
    queries = case.module_centers[:, :2]
    hard_access = model.access(hard, queries, 'QE')
    soft_access = model.access(soft, queries, 'QE', soft=True)
    assert not hard_access.support[..., 0].any()
    assert soft_access.support[..., 0].all()
    values = torch.tensor([[[7., 4., 3., 2., 1.]]])
    # One union normalization using retained fine-source values and measures.
    prediction = (soft_access.weight * case.env_weights[:, None] * values).sum(-1) / case.env_weights.sum(-1, keepdim=True)
    logits = soft.strategy_data['source_logits']['QE']
    derivative = torch.autograd.grad(prediction.mean(), logits)[0][..., 0].sum()
    assert derivative > 0
    restored_logits = logits.detach().clone()
    restored_logits[..., 0] += .01
    restored_membership = _source_density(restored_logits, case.env_weights, soft=True, temperature=1.)
    restored = source_moments(soft_access.edge_access.detach(), restored_membership, soft.controls['QE'].detach(), case.env_weights, soft.source_valid['E'])
    restored_prediction = (restored.weight * case.env_weights[:, None] * values).sum(-1) / case.env_weights.sum(-1, keepdim=True)
    assert restored_prediction.mean() > prediction.detach().mean()
    torch.testing.assert_close((restored_prediction.mean() - prediction.detach().mean()) / .01, derivative, rtol=.02, atol=1e-6)


@pytest.mark.parametrize('soft', [False, True])
def test_identical_environment_atom_split_preserves_density_controls_and_read(soft):
    case = encoded(batch=1)
    model = OverlapControlHypergraph(8, 2).eval()
    state = model.prepare(case, case.module_tokens, soft=soft)
    index = torch.tensor([0, 1, 2, 3, 4, 4])
    measures = case.env_weights[:, index].clone()
    measures[:, -2:] *= .5
    split = replace(case, env_tokens=case.env_tokens[:, index], env_coords=case.env_coords[:, index], env_weights=measures, env_features=case.env_features[:, index])
    split_state = model.prepare(split, split.module_tokens, soft=soft)
    torch.testing.assert_close(state.admission, split_state.admission)
    torch.testing.assert_close(state.controls['QE'], split_state.controls['QE'])
    original = model.access(state, case.module_centers, 'QE', soft=soft)
    divided = model.access(split_state, case.module_centers, 'QE', soft=soft)
    torch.testing.assert_close(original.weight[:, :, index], divided.weight)
    values = case.env_tokens[..., :2]
    reference = (original.weight[..., None] * case.env_weights[:, None, :, None] * values[:, None]).sum(2)
    observed = (divided.weight[..., None] * split.env_weights[:, None, :, None] * values[:, None, index]).sum(2)
    torch.testing.assert_close(reference, observed)


def test_axis_identity_and_schedule_resume_continuity():
    two = pad_geometry(torch.tensor([[2., 3.]]))
    three = pad_geometry(torch.tensor([[2., 3., 0.]]))
    torch.testing.assert_close(two[:, :3], three[:, :3])
    assert two.tolist() == [[2., 3., 0., 1., 1., 0.]]
    assert three.tolist() == [[2., 3., 0., 1., 1., 1.]]
    assert structural_pressure(25) == 0 and structural_pressure(100) == 1
    model = OverlapControlHypergraph(8, 2)
    model.set_epoch(80)
    resumed = OverlapControlHypergraph(8, 2)
    resumed.load_state_dict(model.state_dict())
    assert resumed.training_epoch == 80
    assert structural_pressure(int(resumed.training_epoch)) == structural_pressure(80)


def test_structural_objective_does_not_reward_dense_root_or_padding():
    case = encoded(batch=1)
    model = OverlapControlHypergraph(8, 2).eval()
    state = model.prepare(case, case.module_tokens, soft=True)
    access = model.access(state, case.module_centers, 'MM', soft=True, pair_valid=(~torch.eye(4, dtype=torch.bool))[None])
    cost, metrics = structural_cost({'MM': access}, state)
    assert torch.isfinite(cost)
    assert metrics['MM_unique_pairs'] == 9  # One padded source, diagonal excluded.
    assert access.diagnostics['eligible_pairs'] == 9
    # A source-uniform universal root has no pair sparsity.
    root = source_moments(torch.ones(1, 4, 1), torch.ones(1, 1, 4), torch.zeros(1, 1, 16), torch.ones(1, 4), torch.ones(1, 4, dtype=torch.bool))
    _, root_metrics = structural_cost({'MM': root}, replace(state, source_valid={'M': torch.ones(1, 4, dtype=torch.bool), 'E': state.source_valid['E']}))
    assert root_metrics['MM_smooth_pair_fraction'] > .95


def test_dense_unit_density_structural_proxy_has_nonzero_gradient_at_qe192():
    case = encoded(batch=1, environment=192)
    model = OverlapControlHypergraph(8, 2).eval()
    state = model.prepare(case, case.module_tokens, soft=True)
    access = source_moments(torch.ones(1, 2, 1), torch.ones(1, 1, 192), torch.zeros(1, 1, 16), case.env_weights, state.source_valid['E'])
    density = torch.ones_like(access.density, requires_grad=True)
    access = replace(access, density=density)
    cost, metrics = structural_cost({'QE': access}, state)
    derivative = torch.autograd.grad(cost, density)[0]
    assert derivative.min() > 1e-5
    expected = torch.full_like(derivative, 3 * torch.exp(torch.tensor(-3.)) / density.numel())
    torch.testing.assert_close(derivative, expected)
    assert metrics['QE_smooth_pair_fraction'] > .95


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('architecture', ['overlap_control_hypergraph_honf', 'local_overlap_hypergraph_honf'])
def test_overlap_local_native_fine_identity_outputs_and_first_gradients(dimension, architecture):
    case = encoded(dimension)
    query, features = torch.rand(2, 3, dimension), torch.rand(2, 3, 6)
    dense = DensePairwiseField(8, 12, 2, 2).eval()
    model = TypedHypergraphField(8, 12, 2, 2, architecture=architecture, spatial_dim=dimension, module_characteristic_length=.2).eval()
    model.plan_intervention = 'full_access'
    dense.read(dense.prepare(case, case.module_tokens), case, query, features)
    model.read(model.prepare(case, case.module_tokens), case, query, features)
    model.load_state_dict({**model.state_dict(), **dense.state_dict()})
    values, gradients = [], []
    for current in (dense, model):
        source = case.module_tokens.clone().requires_grad_()
        environment = case.env_tokens.clone().requires_grad_()
        coordinates = query.clone().requires_grad_()
        inputs = replace(case, module_tokens=source, env_tokens=environment)
        output, _ = current.read(current.prepare(inputs, source), inputs, coordinates, features)
        values.append(output)
        gradients.append(torch.autograd.grad(output.square().sum(), (source, environment, coordinates)))
    torch.testing.assert_close(values[1], values[0], rtol=2e-5, atol=2e-6)
    for observed, expected in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(observed, expected, rtol=3e-5, atol=3e-6)


def test_overlap_collective_control_heads_learn_after_identity_gain_first_step():
    case = encoded()
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = torch.nn.Module()
            self.core.backend = TypedHypergraphField(8, 12, 2, 2, architecture='overlap_control_hypergraph_honf', spatial_dim=2, module_characteristic_length=.2)
        def forward(self, encoded, queries, features):
            prepared = self.core.backend.prepare(encoded, encoded.module_tokens)
            context, auxiliary = self.core.backend.read(prepared, encoded, queries, features)
            return {'pred_field': context, 'interaction_aux': auxiliary}
    model = Wrapper().train()
    model.core.backend.organizer.eval()
    queries, features = torch.rand(2, 3, 2), torch.rand(2, 3, 6)
    model(case, queries, features)
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    gradients = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        result = hard_value_soft_hypergraph_forward(model, case, queries, features)
        result['pred_field'].square().mean().backward()
        gradient = model.core.backend.organizer.control_heads['QE'].net[-1].weight.grad
        gradients.append(0 if gradient is None else float(gradient.abs().sum()))
        optimizer.step()
    assert max(gradients[1:]) > 1e-8
