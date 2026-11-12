from dataclasses import replace
import copy

import torch

from honf_forward_core.interaction_reused_member_routing import anchored_core_from_config
from honf_forward_core.interaction_reused_member_routing.detail import DetailAnchoredMemberRoutingCore
from honf_forward_core.interaction_reused_member_routing.anchored import anchored_row_read
from honf_forward_core.interaction_reused_member_routing.primitives import build_joint_member_measure
from tests.test_r14_query_relative_values import _model_scene_query


def _variant(model, capability):
    config = model.get_config()
    config['reader_capability'] = capability
    other = anchored_core_from_config(config).double().eval()
    missing, unexpected = other.load_state_dict(model.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == ({'evidence_strength.M', 'evidence_strength.E'} if capability == 'evidence_strength' else set())
    return other


def test_h1_zero_init_equivalence_and_gradients():
    h0, scene, query = _model_scene_query()
    h1 = _variant(h0, 'evidence_strength')
    assert sum(p.numel() for p in h1.evidence_strength.parameters()) == 6
    for mode in ('full', 'restricted'):
        old = h0.decode(h0.prepare(scene), query, intervention={'read_mode': mode})
        new = h1.decode(h1.prepare(scene), query, intervention={'read_mode': mode}, return_trace=True)
        torch.testing.assert_close(old.joint_fields, new.joint_fields, rtol=0, atol=0)
        assert torch.isfinite(new.trace['anchor_read_strength_evidence']).all()
    old_params = dict(h0.named_parameters())
    new_params = dict(h1.named_parameters())
    names = tuple(old_params)
    old_grad = torch.autograd.grad(old.joint_fields.square().sum(),
                                   [old_params[n] for n in names] + [query.coordinates],
                                   allow_unused=True, retain_graph=True)
    new_grad = torch.autograd.grad(new.joint_fields.square().sum(),
                                   [new_params[n] for n in names] + [query.coordinates],
                                   allow_unused=True, retain_graph=True)
    for a, b in zip(old_grad, new_grad):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
    new.joint_fields.square().sum().backward()
    for p in h1.evidence_strength.values():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    restored = anchored_core_from_config(h1.get_config()).double()
    restored.load_state_dict(h1.state_dict(), strict=True)
    assert restored.reader_capability == 'evidence_strength'


def _explicit_all_native(model, prepared, query):
    state = model._query_states(prepared, query)
    qnorm = model._routing_features(state)
    mu = build_joint_member_measure(prepared.scene.module_mask, prepared.scene.environment_mask,
                                     prepared.scene.environment_measure, dtype=state.dtype)
    contexts = {role: [] for role in ('M', 'E')}
    for b in range(query.batch_size):
        rows = {role: [] for role in contexts}
        for q in range(query.num_queries):
            for role in contexts:
                start = 0 if role == 'M' else prepared.scene.num_modules
                end = prepared.scene.num_modules if role == 'M' else mu.shape[-1]
                slots = (mu[b, start:end] > 0).nonzero().flatten()
                indices = torch.stack((torch.full_like(slots,b), torch.full_like(slots,q), slots), -1)
                keys = prepared.member_keys_M if role == 'M' else prepared.member_keys_E
                scorer = model.member_router.member_M if role == 'M' else model.member_router.member_E
                geo = model._pair_geometry(prepared.scene, query, role, indices)
                raw = scorer(torch.cat((qnorm[b,q].expand(len(slots),-1), model._routing_features(keys)[b,slots], geo), -1)).squeeze(-1)
                values, _ = model._selected_member_values(prepared, query, state, role, indices[:,0], indices[:,1], slots)
                result = anchored_row_read(raw[None], mu[b,start:end][slots], torch.ones_like(slots,dtype=torch.bool),
                    values, qnorm[b,q][None], model.typed_read_gates[role], len(slots), torch.ones(1,dtype=torch.bool))
                rows[role].append(result.context[0])
        for role in contexts:
            contexts[role].append(torch.stack(rows[role]))
    fused = torch.cat((state, *(torch.stack(contexts[r]) for r in ('M','E'))), -1)
    return model.member_router.joint_head(model.member_router.joint_trunk(fused))


def test_all_native_matches_explicit_and_query_tiling_gradients():
    base, scene, query = _model_scene_query()
    model = _variant(base, 'all_native')
    ref = copy.deepcopy(model)
    prepared = model.prepare(scene)
    output = model.decode(prepared, query, return_trace=True)
    for role, expected_count in (('M', scene.module_mask[0].sum()),
                                 ('E', (scene.environment_mask[0] & (scene.environment_measure[0] > 0)).sum())):
        assert output.trace[f'candidate_mask_{role}'][0, 0].sum() == expected_count
    expected = _explicit_all_native(ref, ref.prepare(scene), query)
    torch.testing.assert_close(output.joint_fields, expected, rtol=1e-10, atol=1e-11)
    a,b,shared = model.decode_paired(prepared, query)
    assert a is b and shared == {}
    tiled = copy.deepcopy(model); tiled.query_tile_size = 1
    perm = torch.arange(query.num_queries - 1, -1, -1)
    pq = replace(query, coordinates=query.coordinates[:,perm], anchor_indices=query.anchor_indices[:,perm],
                 anchor_weights=query.anchor_weights[:,perm], query_mask=query.query_mask[:,perm] if query.query_mask is not None else None)
    out2 = tiled.decode(tiled.prepare(scene), pq)
    torch.testing.assert_close(output.joint_fields[:,perm], out2.joint_fields, rtol=1e-10, atol=1e-11)
    g1 = torch.autograd.grad(output.joint_fields.square().sum(), tuple(model.parameters()), allow_unused=True)
    g2 = torch.autograd.grad(out2.joint_fields.square().sum(), tuple(tiled.parameters()), allow_unused=True)
    for a,b in zip(g1,g2):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a,b,rtol=1e-7,atol=1e-9)


def test_evidence_matches_live_selected_priors_and_scores():
    base, scene, query = _model_scene_query()
    model = _variant(base, 'evidence_strength')
    with torch.no_grad():
        model.evidence_strength['M'].copy_(torch.tensor([0.2, -0.3, 0.4], dtype=torch.double))
    output = model.decode(model.prepare(scene), query, return_trace=True)
    trace = output.trace
    read = trace['anchor_read_member_read_index']
    group = read * 2 + (~trace['anchor_read_is_module']).long()
    prior = trace['anchor_read_access_prior']
    keys = trace['anchor_read_coalesced_union_keys']
    pairs = ((trace['anchor_read_indices'][read, 0] * query.num_queries
              + trace['anchor_read_indices'][read, 1]) * (scene.num_modules + scene.num_environment_tokens)
             + trace['anchor_read_member_index'])
    scores = torch.zeros_like(prior)
    for role in ('M', 'E'):
        mask = trace['anchor_read_is_module'] == (role == 'M')
        rolekeys = trace[f'member_score_union_indices_{role}']
        scores[mask] = trace[f'member_score_union_values_{role}'][torch.searchsorted(rolekeys, pairs[mask])]
    evidence = trace['anchor_read_strength_evidence'].reshape(-1, 3)
    for g in range(evidence.shape[0]):
        use = group == g
        if not use.any():
            torch.testing.assert_close(evidence[g], torch.zeros(3, dtype=prior.dtype))
            continue
        mass = prior[use].sum()
        mean = (prior[use] * scores[use]).sum() / mass
        var = (prior[use] * (scores[use] - mean).square()).sum() / mass
        torch.testing.assert_close(evidence[g], torch.stack((mass.log1p(), mean.asinh(), var.log1p())))


def test_h0_prior_scale_and_singleton_invariance():
    gate = torch.nn.Linear(4, 2).double()
    scores = torch.tensor([[1., 3.], [7., -2.]], dtype=torch.double)
    prior = torch.tensor([.2, .6], dtype=torch.double)
    kwargs = dict(candidate=torch.ones(2, dtype=torch.bool), values=torch.ones(2, 3, dtype=torch.double),
                  normalized_query=torch.zeros(2, 2, dtype=torch.double), typed_gate=gate,
                  scene_eligible_count=2, query_mask=torch.ones(2, dtype=torch.bool))
    first = anchored_row_read(scores, prior, **kwargs)
    scaled = anchored_row_read(scores, prior * 4, **kwargs)
    torch.testing.assert_close(first.strength, scaled.strength)
    singleton = anchored_row_read(scores, prior, **{**kwargs, 'candidate': torch.tensor([True, False])})
    assert torch.equal(singleton.expectation, torch.zeros(2, dtype=torch.double))
    assert torch.equal(singleton.normalized_entropy, torch.zeros(2, dtype=torch.double))
    empty = anchored_row_read(scores, prior, **{**kwargs, 'candidate': torch.zeros(2, dtype=torch.bool)})
    assert torch.equal(empty.strength, torch.zeros(2, dtype=torch.double))


def test_q0_native_extraction_and_work_are_not_counted_twice():
    from types import SimpleNamespace
    from channelthermal.training.r13_6_member_routing_task import _AnchoredPairedMemberRoutingTask
    from honf_forward_core.interaction_reused_member_routing.core import InteractionJointOutput
    fields = torch.zeros(1, 2, 4)
    result = InteractionJointOutput(fields, None, {'reader_pairs': 12})
    class Adapter:
        def __init__(self):
            self.calls = 0
            self.core = SimpleNamespace(decode_paired=lambda *args: (result, result, {}))
        def validate_prepared(self, prepared):
            pass
        def _extract_native_roles(self, prepared, temperature, flow):
            self.calls += 1
            return {'temperature': temperature, 'flow': flow}
    model = Adapter()
    task = SimpleNamespace(model=model, LINEAGE='Q0')
    prepared = SimpleNamespace(organization=SimpleNamespace(work_counts={}), query_batch=SimpleNamespace(local_weights=torch.zeros(1)),
        fluid_xy=torch.zeros(1, 2, 2), union=SimpleNamespace(unique_counts=torch.tensor([2])))
    pair = _AnchoredPairedMemberRoutingTask._paired_predictions(task, prepared, None)
    assert model.calls == 1
    assert pair.full.native_main is pair.restricted.native_main
    assert pair.work['core_decode_joint_full_reader_pairs'] == 12
    assert 'core_decode_joint_restricted_reader_pairs' not in pair.work
