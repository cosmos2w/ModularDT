"""Focused contracts for the opt-in interaction-preserving joint core."""
from __future__ import annotations

import io

import pytest
import torch

from honf_forward_core.interface_fields.interaction_core import NonlinearFieldReadout, _geometry
from honf_forward_core.interface_fields.interaction_preserving_joint import (
    InteractionPreservingJointCore,
)


def _inputs(*, fractional: bool = True):
    generator = torch.Generator().manual_seed(22103)
    random = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    batch, modules, environments, dimension, query_width = 1, 4, 5, 2, 2
    centers = torch.rand(batch, modules, dimension, generator=generator, dtype=torch.float64)
    source_present = torch.tensor([[1.0, 0.5, 1.0, 0.0]], dtype=torch.float64)
    environment_present = torch.tensor([[1.0, 0.5, 1.0, 1.0, 0.0]], dtype=torch.float64)
    if not fractional:
        source_present = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
        environment_present = torch.ones(batch, environments, dtype=torch.float64)
    receivers = torch.rand(batch, 7, dimension, generator=generator, dtype=torch.float64)
    return {
        'sources': random(batch, modules, 4),
        'context': random(batch, 3),
        'centers': centers,
        'present': source_present,
        'lengths': torch.tensor([[1.2, 0.9]], dtype=torch.float64),
        'source_lengths': torch.full((batch, modules), 0.18, dtype=torch.float64),
        'source_measures': (torch.ones(batch, modules, dtype=torch.float64) if not fractional else
                            torch.tensor([[0.5, 2.0, 1.0, 0.0]], dtype=torch.float64)),
        'environment_tokens': random(batch, environments, 5),
        'environment_coords': torch.rand(batch, environments, dimension,
                                          generator=generator, dtype=torch.float64),
        'environment_present': environment_present,
        'environment_measures': (torch.ones(batch, environments, dtype=torch.float64)
                                 if not fractional else
                                 torch.tensor([[0.5, 2.0, 1.0, 0.75, 0.0]], dtype=torch.float64)),
        'source_ids': torch.tensor([[71, 19, 44, -1 if fractional else 58]], dtype=torch.long),
        'environment_ids': torch.tensor([[93, 14, 52, 38, -1 if fractional else 85]],
                                        dtype=torch.long),
        'domain_origin': torch.tensor([[-0.1, 0.2]], dtype=torch.float64),
        'receivers': receivers,
        'receiver_features': random(batch, receivers.shape[1], query_width),
    }


def _core(mode: str = 'P', *, affine_outputs: int = 1, seed: int = 703):
    return InteractionPreservingJointCore(
        4, 3, 5, spatial_dim=2, hidden=12, message=8, mode=mode,
        collective_width=6, regional_anchors=0 if mode == 'P' else 3,
        field_outputs=3, affine_outputs=affine_outputs, query_width=2,
        initialization_seed=seed,
        locality_prior_strength=None if mode == 'P' else 1.0,
    ).double()


def _prepare(model, values, **kwargs):
    arguments = {name: values[name] for name in (
        'sources', 'context', 'centers', 'present', 'lengths', 'source_lengths',
        'source_measures', 'environment_tokens', 'environment_coords',
        'environment_present', 'environment_measures', 'source_ids',
        'environment_ids', 'domain_origin')}
    return model.prepare_context(**arguments, **kwargs)


def test_wind_p_is_exact_maintained_nonlinear_readout_and_loads_its_weights():
    values = _inputs()
    values['centers'] = values['centers'].detach().requires_grad_(True)
    torch.manual_seed(17)
    legacy = NonlinearFieldReadout(4, 3, 5, spatial_dim=2, hidden=12,
                                   message=8, output_width=3, query_width=2).double()
    recovered = _core('P', affine_outputs=0)
    recovered.load_state_dict(legacy.state_dict(), strict=True)
    old_context = legacy.prepare_context(**{name: values[name] for name in (
        'sources', 'context', 'centers', 'present', 'lengths', 'source_lengths',
        'source_measures', 'environment_tokens', 'environment_coords',
        'environment_present', 'environment_measures', 'source_ids')})
    new_context = _prepare(recovered, values)
    old_value = legacy.predict(old_context, values['receivers'], values['receiver_features'])
    new_value = recovered.predict_fields(new_context, values['receivers'], values['receiver_features'])
    torch.testing.assert_close(new_value, old_value, rtol=0, atol=0)

    old_grad = torch.autograd.grad(old_value.sum(), values['centers'], retain_graph=True)[0]
    new_grad = torch.autograd.grad(new_value.sum(), values['centers'])[0]
    torch.testing.assert_close(new_grad, old_grad, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['P-G', 'P-H'])
def test_zero_collective_outputs_recover_p_exactly_with_fractional_presence(mode):
    values = _inputs(fractional=True)
    direct, organized = _core('P'), _core(mode)
    direct_context = _prepare(direct, values)
    organized_context = _prepare(organized, values)
    for name, parameter in direct.named_parameters():
        torch.testing.assert_close(parameter, dict(organized.named_parameters())[name],
                                   rtol=0, atol=0)
    for name in ('source_states', 'environment_states', 'global_state'):
        torch.testing.assert_close(getattr(organized_context, name),
                                   getattr(direct_context, name), rtol=0, atol=0)
    direct_fields = direct.predict_fields(
        direct_context, values['receivers'], values['receiver_features'])
    organized_fields = organized.predict_fields(
        organized_context, values['receivers'], values['receiver_features'])
    torch.testing.assert_close(organized_fields, direct_fields, rtol=0, atol=0)

    direct_response = direct.prepare_receivers(
        direct_context, values['receivers'], values['receiver_features'])
    organized_response = organized.prepare_receivers(
        organized_context, values['receivers'], values['receiver_features'])
    torch.testing.assert_close(organized_response.dense_kernel(),
                               direct_response.dense_kernel(), rtol=0, atol=0)
    torch.testing.assert_close(organized_context.source_membership,
                               _prepare(_core('P-G'), values).source_membership,
                               rtol=0, atol=0)


def test_initial_graph_branches_share_content_and_geometric_scores():
    values = _inputs()
    geometry = _core('P-G')
    learned = _core('P-H')
    geometry_context = _prepare(geometry, values)
    learned_context = _prepare(learned, values)
    for name in ('source_states', 'environment_states', 'global_state', 'group_states',
                 'source_membership', 'environment_membership', 'source_update_delta',
                 'environment_update_delta'):
        torch.testing.assert_close(getattr(learned_context, name),
                                   getattr(geometry_context, name), rtol=0, atol=0)
    assert learned_context.source_measures_raw is values['source_measures']
    assert learned_context.environment_measures_raw is values['environment_measures']
    assert learned_context._intervention_receipt['return_rule'].startswith('normalized transpose')


def test_pair_message_path_executes_two_rounds_before_and_after_collective():
    values = _inputs()
    model = _core('P-H', affine_outputs=0)
    seen_mm, seen_me_em = [], []
    handles = []
    for module in model.module_messages:
        handles.append(module.register_forward_pre_hook(
            lambda _module, args: seen_mm.append(tuple(args[0].shape[:3]))))
    for module in model.environment_messages:
        handles.append(module.register_forward_pre_hook(
            lambda _module, args: seen_me_em.append(tuple(args[0].shape[:3]))))
    prepared = _prepare(model, values)
    for handle in handles:
        handle.remove()
    assert seen_mm == [(1, 4, 4), (1, 4, 4)]
    assert seen_me_em == [(1, 4, 5), (1, 5, 4), (1, 4, 5), (1, 5, 4)]
    assert prepared.source_membership.shape == (1, 7, 4)
    assert prepared.environment_membership.shape == (1, 7, 5)
    assert prepared._intervention_receipt['incidence'] == 'learned'


def test_first_step_trains_zero_outputs_then_reaches_collective_values_and_scores():
    values = _inputs(fractional=False)
    model = _core('P-H', affine_outputs=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    projections = (model.collective_to_source, model.collective_to_environment,
                   model.collective_to_receiver)
    assert all(torch.count_nonzero(layer.weight) == 0 for layer in projections)
    for _step in range(3):
        optimizer.zero_grad(set_to_none=True)
        prepared = _prepare(model, values)
        prediction = model.predict_fields(prepared, values['receivers'],
                                          values['receiver_features'])
        response = model.prepare_receivers(prepared, values['receivers'],
                                          values['receiver_features'])
        affine = model.apply_forcing(response, torch.tensor([[0.2, -0.5, 0.7, 0.1]],
                                                             dtype=torch.float64))
        (prediction.square().mean() + affine.square().mean()).backward()
        if _step == 0:
            assert all(layer.weight.grad is not None and layer.weight.grad.norm() > 0
                       for layer in projections)
            assert model.affine_head[-1].weight.grad is not None
            assert model.affine_head[-1].weight.grad.norm() > 0
            assert model.affine_source_read[0].weight.grad is not None
            assert model.affine_source_read[0].weight.grad.norm() > 0
        if _step == 1:
            assert model.collective_source_value[0].weight.grad is not None
            assert model.collective_source_value[0].weight.grad.norm() > 0
            assert model.source_membership_score[-1].weight.grad is not None
            assert model.source_membership_score[-1].weight.grad.norm() > 0
        if _step == 2:
            assert model.source_membership_score[0].weight.grad is not None
            assert model.source_membership_score[0].weight.grad.norm() > 0
        optimizer.step()


def test_graph_interventions_recompute_pair_context_restore_and_keep_adapter_sidecars():
    values = _inputs(fractional=False)
    model = _core('P-H', affine_outputs=0)
    with torch.no_grad():
        model.source_membership_score[-1].weight.fill_(0.15)
        model.environment_membership_score[-1].weight.fill_(-0.1)
        model.collective_to_source.weight.normal_(0, 0.04)
        model.collective_to_environment.weight.normal_(0, 0.04)
        model.collective_to_receiver.weight.normal_(0, 0.04)
    learned = _prepare(model, values)
    learned.adapter_binding = {'catalogue': 'held-reference'}
    geometric = model.intervene_context(
        learned, incidence_mode='geometry', receiver_access='geometry')
    shuffled = model.intervene_context(learned, matched_mass_shuffle=True)
    disabled = model.intervene_context(learned, collective_enabled=False)
    restored = model.restore(geometric)

    assert geometric.adapter_binding == learned.adapter_binding
    assert geometric._incidence_mode == 'geometry'
    assert not torch.equal(geometric.source_states, learned.source_states)
    assert not torch.equal(shuffled.source_membership, learned.source_membership)
    torch.testing.assert_close(torch.sort(shuffled.source_membership, dim=-1).values,
                               torch.sort(learned.source_membership, dim=-1).values,
                               rtol=0, atol=0)
    assert not torch.equal(disabled.source_states, learned.source_states)
    torch.testing.assert_close(restored.source_states, learned.source_states, rtol=0, atol=0)
    torch.testing.assert_close(restored.group_states, learned.group_states, rtol=0, atol=0)

    direct = _core('P', affine_outputs=0)
    direct_context = _prepare(direct, values)
    disabled_fields = model.predict_fields(disabled, values['receivers'],
                                          values['receiver_features'])
    direct_fields = direct.predict_fields(direct_context, values['receivers'],
                                          values['receiver_features'])
    torch.testing.assert_close(disabled_fields, direct_fields, rtol=0, atol=0)


def test_native_affine_response_is_source_resolved_and_uses_fp64_application():
    values = _inputs()
    model = _core('P-H')
    prepared = _prepare(model, values)
    response = model.prepare_receivers(prepared, values['receivers'],
                                       values['receiver_features'])
    kernel = response.dense_kernel()
    assert kernel.shape == (1, 7, 4, 1)
    assert torch.count_nonzero(kernel[:, :, 3]) == 0
    heat = torch.tensor([[0.0, 0.4, -0.2, 0.0]], dtype=torch.float64)
    applied = model.apply_forcing(response, heat, accumulation_dtype=torch.float64)
    expected = response.offset + torch.einsum('bqma,bm->bqa',
                                               kernel.double(), heat.double())
    torch.testing.assert_close(applied, expected[..., 0], rtol=0, atol=0)
    increment = model.apply_increment(response, heat, accumulation_dtype=torch.float64)
    torch.testing.assert_close(increment, torch.einsum('bqma,bm->bqa',
        kernel.double(), heat.double())[..., 0], rtol=0, atol=0)
    torch.testing.assert_close(model.predict_responses(prepared, values['receivers'],
        values['receiver_features']), kernel[..., 0], rtol=0, atol=0)


def test_response_applications_reject_a_context_owned_by_another_core():
    values = _inputs()
    owner = _core('P-H')
    other = _core('P-H')
    response = owner.prepare_receivers(_prepare(owner, values), values['receivers'],
                                      values['receiver_features'])
    forcing = torch.zeros_like(values['present'])
    with pytest.raises(ValueError, match='belongs to another model'):
        other.apply_forcing(response, forcing)
    with pytest.raises(ValueError, match='belongs to another model'):
        other.apply_increment(response, forcing)


def test_absent_or_ambiguous_source_ids_and_wind_affine_increment_are_rejected():
    values = _inputs()
    model = _core('P')
    arguments = {name: values[name] for name in (
        'sources', 'context', 'centers', 'present', 'lengths', 'source_lengths',
        'source_measures', 'environment_tokens', 'environment_coords',
        'environment_present', 'environment_measures')}
    with pytest.raises(ValueError, match='explicit physical source_ids'):
        model.prepare_context(**arguments)
    duplicate = dict(values)
    duplicate['source_ids'] = torch.tensor([[71, 71, 44, -1]], dtype=torch.long)
    with pytest.raises(ValueError, match='uniquely identify'):
        _prepare(model, duplicate)

    wind = _core('P', affine_outputs=0)
    with pytest.raises(ValueError, match='no exact affine finite-increment'):
        wind.apply_increment(None, torch.ones(1, 4, dtype=torch.float64))


def test_prepared_context_rejects_changed_inputs_weights_locality_and_access_semantics():
    values = _inputs()
    model = _core('P-H')
    prepared = _prepare(model, values)
    model.locality_prior_strength = 1.5
    with pytest.raises(ValueError, match='semantics changed'):
        prepared.assert_fresh()

    model.locality_prior_strength = 1.0
    prepared = _prepare(model, values)
    prepared._access_mode = 'geometry'
    with pytest.raises(ValueError, match='semantics changed'):
        prepared.assert_fresh()

    prepared = _prepare(model, values)
    with torch.no_grad():
        model.source_encoder[0].weight.add_(0.01)
    with pytest.raises(ValueError, match='model weights changed'):
        prepared.assert_fresh()

    prepared = _prepare(model, values)
    with torch.no_grad():
        values['domain_origin'].add_(0.01)
    with pytest.raises(ValueError, match='tensors changed'):
        prepared.assert_fresh()


def test_learned_receiver_query_uses_the_inherited_absolute_query_geometry():
    values = _inputs()
    model = _core('P-H')
    prepared = _prepare(model, values)
    captured = []
    handle = model.receiver_query[0].register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0]))
    model._receiver_access(prepared, values['receivers'], values['receiver_features'], 'learned')
    handle.remove()
    geometry_width = model.spatial_dim * 9
    expected = _geometry(values['receivers'] / prepared.lengths[:, None])
    torch.testing.assert_close(captured[0][..., :geometry_width], expected, rtol=0, atol=0)


def test_query_chunking_source_packing_and_same_recipe_checkpoint_are_invariant():
    values = _inputs(fractional=False)
    model = _core('P-H')
    with torch.no_grad():
        model.collective_to_source.weight.normal_(0, 0.03)
        model.collective_to_environment.weight.normal_(0, 0.03)
        model.collective_to_receiver.weight.normal_(0, 0.03)
        model.source_membership_score[-1].weight.normal_(0, 0.03)
        model.environment_membership_score[-1].weight.normal_(0, 0.03)
        model.receiver_query[-1].weight.normal_(0, 0.03)
        model.edge_key.weight.normal_(0, 0.03)
        model.access_geometry[-1].weight.normal_(0, 0.03)
    prepared = _prepare(model, values)
    full = model.predict_fields(prepared, values['receivers'], values['receiver_features'])
    chunked = model.predict_fields(prepared, values['receivers'], values['receiver_features'],
                                   chunk_size=2)
    torch.testing.assert_close(chunked, full, rtol=1e-12, atol=1e-12)
    full_kernel = model.prepare_receivers(prepared, values['receivers'],
                                          values['receiver_features']).dense_kernel()
    chunked_kernel = model.prepare_receivers(prepared, values['receivers'],
        values['receiver_features'], chunk_size=2).dense_kernel()
    torch.testing.assert_close(chunked_kernel, full_kernel, rtol=1e-12, atol=1e-12)

    order = torch.tensor([2, 0, 3, 1])
    repacked = dict(values)
    for key in ('sources', 'centers', 'present', 'source_lengths', 'source_measures', 'source_ids'):
        repacked[key] = values[key].index_select(1, order)
    repacked_context = _prepare(model, repacked)
    repacked_output = model.predict_fields(repacked_context, values['receivers'],
                                            values['receiver_features'])
    torch.testing.assert_close(repacked_output, full, rtol=1e-12, atol=1e-12)

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = _core('P-H')
    restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True),
                             strict=True)
    restored_context = _prepare(restored, values)
    restored_output = restored.predict_fields(restored_context, values['receivers'],
                                               values['receiver_features'])
    torch.testing.assert_close(restored_output, full, rtol=0, atol=0)


def test_learned_collective_recipe_requires_explicit_prior_and_widths_are_bound():
    with pytest.raises(ValueError, match='explicit finite nonnegative locality prior'):
        InteractionPreservingJointCore(4, 3, 5, mode='P-H', regional_anchors=3)
    model = _core('P-H')
    assert model.config['mode'] == 'P-H'
    assert model.config['collective_width'] == 6
    assert model.config['locality_prior_strength'] == 1.0


@pytest.mark.parametrize('mode', ['P', 'P-G', 'P-H'])
def test_native_curl_three_output_head_is_exact_four_output_initialization_trim(mode):
    anchors = 0 if mode == 'P' else 3
    prior = None if mode == 'P' else 1.0
    common = {
        "source_width": 4,
        "context_width": 3,
        "environment_width": 5,
        "spatial_dim": 2,
        "hidden": 12,
        "message": 8,
        "mode": mode,
        "collective_width": 6,
        "regional_anchors": anchors,
        "affine_outputs": 1,
        "query_width": 2,
        "initialization_seed": 703,
        "locality_prior_strength": prior,
    }
    torch.manual_seed(9927)
    before = torch.random.get_rng_state().clone()
    reference = InteractionPreservingJointCore(**common, field_outputs=4).double()
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    trimmed = InteractionPreservingJointCore(
        **common, field_outputs=3, initialization_reference_field_outputs=4
    ).double()
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    assert reference.field_outputs == 4 and trimmed.field_outputs == 3
    assert trimmed.output_width == trimmed.config['output_width'] == 3
    assert trimmed.config['initialization_reference_field_outputs'] == 4
    ref_state, trim_state = reference.state_dict(), trimmed.state_dict()
    assert set(ref_state) == set(trim_state)
    for name, ref_tensor in ref_state.items():
        if name in {'field_head.2.weight', 'field_head.2.bias'}:
            torch.testing.assert_close(trim_state[name], ref_tensor[:3], rtol=0, atol=0)
        else:
            torch.testing.assert_close(trim_state[name], ref_tensor, rtol=0, atol=0)
    values = _inputs()
    reference_value = reference.predict_fields(_prepare(reference, values), values['receivers'],
                                              values['receiver_features'])
    trimmed_value = trimmed.predict_fields(_prepare(trimmed, values), values['receivers'],
                                           values['receiver_features'])
    torch.testing.assert_close(trimmed_value, reference_value[..., :3], rtol=1e-14, atol=2e-16)

def test_initialization_reference_width_cannot_be_narrower_than_actual_head():
    with pytest.raises(ValueError, match='at least as wide as the actual field head'):
        InteractionPreservingJointCore(4, 3, 5, field_outputs=4,
                                       initialization_reference_field_outputs=3)
