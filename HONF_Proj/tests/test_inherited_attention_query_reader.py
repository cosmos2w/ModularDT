from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from honf_forward_core.inherited_attention.contracts import ReceiverCover, TypedAssociationAtlas, TypedTokenValues
from honf_forward_core.inherited_attention.normalization import measure_relative_entmax15, measure_relative_softmax
from honf_forward_core.inherited_attention.query_reader import build_cell_query_plan, query_conditioned_gather

from .test_inherited_attention_model import _core, _many_anchor_scene, _query, _select_query, _slice_scene


def _primitive(executor: str, *, sparse: bool):
    generator = torch.Generator().manual_seed(39)
    logits = (3 * torch.randn(2, 2, 4, 5, generator=generator, dtype=torch.float64)).requires_grad_()
    measure = torch.tensor([[.1, .2, .3, .4, 0.], [.3, .2, .1, .4, 0.]], dtype=torch.float64, requires_grad=True)
    valid = measure.detach() > 0
    function = measure_relative_entmax15 if sparse else measure_relative_softmax
    probabilities = function(logits, measure, valid, torch.ones(2, 4, dtype=torch.bool))
    ids = torch.arange(5)[None].expand(2, -1)
    atlas = TypedAssociationAtlas('module', logits, probabilities, torch.arange(4)[None].expand(2, -1),
                                  ids, valid, torch.ones(2, 4, dtype=torch.bool), measure / measure.sum(-1, keepdim=True),
                                  'entmax15' if sparse else 'softmax')
    value = torch.randn(2, 2, 5, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    keys = torch.randn(2, 2, 5, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    qs = torch.randn(2, 2, 6, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    raw_weights = torch.tensor([.2, .3, .7, .4, .5, .9], dtype=torch.float64, requires_grad=True)
    weights = torch.stack((raw_weights, 1 - raw_weights), -1)[None].expand(2, -1, -1)
    offsets = torch.tensor([[[0, 1], [0, 1], [2, 3], [2, 3], [0, 1], [2, 3]]]).expand(2, -1, -1)
    cover = ReceiverCover(offsets, offsets.clone(), weights, torch.ones_like(offsets, dtype=torch.bool),
                          torch.arange(6)[None].expand(2, -1), 'fixture-frame')
    values = TypedTokenValues('module', value, ids, valid, measure)
    result, work, trace = query_conditioned_gather(atlas, values, qs, keys, cover, executor=executor, return_trace=True)
    return result, work, trace, (logits, measure, value, keys, qs, raw_weights), atlas, values, cover


@pytest.mark.parametrize('sparse', [False, True])
def test_cell_attention_is_literal_beta_times_exp_and_matches_all_gradients(sparse):
    indexed, work, trace, inputs, atlas, values, cover = _primitive('cell_selected', sparse=sparse)
    dense, _, _, dense_inputs, _, _, _ = _primitive('dense_reference', sparse=sparse)
    beta = torch.stack([
        sum(cover.weights[b, q, s] * atlas.probabilities[b, :, int(cover.anchor_indices[b, q, s])]
            for s in range(cover.slots))
        for b in range(2) for q in range(6)
    ]).reshape(2, 6, 2, 5).transpose(1, 2)
    score = inputs[4] @ inputs[3].transpose(-1, -2)
    scalar = torch.where(beta > 0, beta, torch.ones_like(beta)).log() + score
    scalar = scalar.masked_fill(beta == 0, -torch.inf)
    explicit_weights = torch.softmax(scalar, -1)
    expected = (explicit_weights @ values.values).transpose(1, 2)
    torch.testing.assert_close(indexed, expected, rtol=3e-13, atol=3e-13)
    torch.testing.assert_close(indexed, dense, rtol=3e-13, atol=3e-13)
    ga = torch.autograd.grad(indexed.square().sum(), inputs)
    gb = torch.autograd.grad(dense.square().sum(), dense_inputs)
    for actual, target in zip(ga, gb):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, target, rtol=2e-10, atol=2e-10)
    assert work['query_padded_candidate_dot_products'] >= work['query_candidate_dot_products']
    assert trace is not None and trace['probabilities'].shape[:3] == (2, 6, 2)
    if sparse:
        assert (atlas.probabilities == 0).any()


def _qcore(**options):
    return _core(architecture='IAG-Q', thermal_response=False, scene_executor='cached_tokens',
                 query_executor=options.pop('query_executor', 'cell_selected'), **options)


def test_shared_final_route_modules_and_no_scene_value_aggregation(monkeypatch):
    import honf_forward_core.inherited_attention.model as model_module

    q = _qcore().eval()
    c = _core(thermal_response=False).eval()
    assert tuple(q.state_dict()) == tuple(c.state_dict())
    for name, value in q.state_dict().items():
        torch.testing.assert_close(value, c.state_dict()[name], rtol=0, atol=0)
    assert len(tuple(q.named_parameters())) == len(tuple(q.named_parameters(remove_duplicate=False)))
    monkeypatch.setattr(model_module, 'prepare_anchor_values', lambda *a, **k: pytest.fail('Q must not compute P@V'))
    scene = _many_anchor_scene()
    org = q.prepare(scene)
    assert org.module_anchor_values is None and org.environment_anchor_values is None
    route = q.context_blocks[-1].em
    expected = q._route_keys(route, org.module_states)
    torch.testing.assert_close(org.module_query_keys, expected, rtol=0, atol=0)
    assert org.module_query_keys is not org.module_score_snapshot.key_normalized
    result = q.decode(org, _query(scene), return_trace=True)
    loss = result.nonlinear_field.square().sum()
    params = dict(q.named_parameters())
    names = ('context_blocks.0.em.query.weight', 'context_blocks.0.em.key.weight',
             'context_blocks.0.em.raw_temperature', 'module_value_projection.weight',
             'context_blocks.0.ee.query.weight', 'context_blocks.0.ee.key.weight')
    gradients = torch.autograd.grad(loss, tuple(params[n] for n in names))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    assert org.work_counts['scene_value_executed_associations'] == 0
    assert result.work_counts['far_pair_value_rows'] == 0
    assert result.work_counts['cover_executed_anchor_reads'] == 0
    assert result.work_counts['query_geometry_grouping_operations'] == 1


def test_core_dense_reference_matches_parameters_and_geometry_derivatives():
    selected, dense = _qcore().eval(), _qcore(query_executor='dense_reference').eval()
    dense.load_state_dict(selected.state_dict())
    sa, sb = _many_anchor_scene(), _many_anchor_scene()
    sa.module_coordinates.requires_grad_()
    sb.module_coordinates.requires_grad_()
    qa, qb = _query(sa, requires_grad=True), _query(sb, requires_grad=True)
    oa, ob = selected.prepare(sa), dense.prepare(sb)
    ya, yb = selected.decode(oa, qa).nonlinear_field, dense.decode(ob, qb).nonlinear_field
    torch.testing.assert_close(ya, yb, rtol=3e-5, atol=3e-6)
    names = ('module_value_projection.weight', 'context_blocks.0.em.query.weight',
             'context_blocks.0.ee.key.weight', 'context_blocks.0.em.geometry_score.0.weight')
    pa, pb = dict(selected.named_parameters()), dict(dense.named_parameters())
    ga = torch.autograd.grad(ya.square().sum(), tuple(pa[n] for n in names) + (qa.coordinates, sa.module_coordinates))
    gb = torch.autograd.grad(yb.square().sum(), tuple(pb[n] for n in names) + (qb.coordinates, sb.module_coordinates))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_query_order_subset_batching_masks_empty_types_and_cache_mutation():
    core = _qcore().eval()
    scene = _many_anchor_scene()
    query = _query(scene)
    with torch.no_grad():
        org = core.prepare(scene)
        full = core.decode(org, query).nonlinear_field
        order = torch.tensor([4, 0, 2])
        subset = core.decode(org, _select_query(query, order)).nonlinear_field
        torch.testing.assert_close(subset, full[:, order], rtol=3e-5, atol=3e-6)
        single_scene = _slice_scene(scene, 0)
        single_query = _query(single_scene, coordinates=query.coordinates[:1].clone())
        # Preserve feature/query IDs from the original batch for this control.
        single_query = replace(single_query, features=query.features[:1].clone())
        single = core.decode(core.prepare(single_scene), single_query).nonlinear_field
        torch.testing.assert_close(single, full[:1], rtol=3e-5, atol=3e-6)
        empty = replace(scene, module_mask=torch.zeros_like(scene.module_mask), module_measure=torch.zeros_like(scene.module_measure))
        empty_result = core.decode(core.prepare(empty), _query(empty))
        assert torch.isfinite(empty_result.nonlinear_field).all()
        mask = query.query_mask.clone()
        mask[:, -1] = False
        masked = _query(scene, query_mask=mask)
        assert (core.decode(org, masked).nonlinear_field[:, -1] == 0).all()
        assert org.module_query_keys is not None
        org.module_query_keys.add_(1)
        with pytest.raises(ValueError, match='factors changed'):
            core.decode(org, query)


def test_plan_mutation_and_exact_zero_support_are_not_resurrected():
    result, _, _, inputs, atlas, values, cover = _primitive('cell_selected', sparse=True)
    plan = build_cell_query_plan(cover)
    plan.query_indices[0, 0] = plan.query_indices[0, 0] + 1
    with pytest.raises(ValueError, match='plan'):
        query_conditioned_gather(atlas, values, inputs[4], inputs[3], cover, executor='cell_selected', plan=plan)
    empty_atlas = replace(atlas, probabilities=atlas.probabilities * 0)
    empty, work, _ = query_conditioned_gather(empty_atlas, values, inputs[4], inputs[3], cover, executor='cell_selected')
    assert torch.isfinite(result).all() and (empty == 0).all()
    assert work['query_candidate_dot_products'] == 0


def test_q_same_backend_checkpoint_resume_preserves_adam_and_next_step():
    core = _qcore()
    optimizer = torch.optim.AdamW(core.parameters(), lr=3e-4)
    scene = _many_anchor_scene()
    query = _query(scene)

    def step(model, optim):
        optim.zero_grad(set_to_none=True)
        model.decode(model.prepare(scene), query).nonlinear_field.square().mean().backward()
        optim.step()

    step(core, optimizer)
    model_state, optimizer_state = deepcopy(core.state_dict()), deepcopy(optimizer.state_dict())
    step(core, optimizer)
    resumed = _qcore()
    resumed.load_state_dict(model_state)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=3e-4)
    resumed_optimizer.load_state_dict(optimizer_state)
    step(resumed, resumed_optimizer)
    for name, value in core.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    for key, state in optimizer.state_dict()['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, resumed_optimizer.state_dict()['state'][key][name], rtol=0, atol=0)
