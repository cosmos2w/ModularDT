"""Directed composition and execution contracts for the mediated reader."""
from dataclasses import replace
import copy
import pytest
import torch
from tests.test_e_anchored_packed_execution import fixture
from honf_forward_core.interaction_reused_member_routing.anchored import AnchoredMemberRoutingCore


def model_fixture(mode='P', locator='e_anchor', dtype=torch.float64):
    old, scene, query = fixture(locator)
    model = AnchoredMemberRoutingCore(old.get_config()['base_model_config'], locator_mode=locator,
        mediated_mode=mode, query_tile_size=4).to(dtype=dtype)
    if dtype == torch.float32:
        from dataclasses import fields
        scene = replace(scene, **{f.name: v.float() for f in fields(scene)
            if torch.is_tensor(v := getattr(scene, f.name)) and v.is_floating_point()})
        query = replace(query, coordinates=query.coordinates.float(), anchor_weights=query.anchor_weights.float())
    return model, scene, query


@pytest.mark.parametrize('mode', ['P', 'G'])
@pytest.mark.parametrize('execution', ['full', 'all_e95', 'source95_all', 'restricted'])
def test_composition_and_true_me(mode, execution):
    model, scene, query = model_fixture(mode)
    prepared = model.prepare(scene)
    result = model.decode(prepared, query, intervention={'read_mode': execution}, return_trace=True)
    t = result.trace
    for (rids, direct), (rids2, conditional, mass) in zip(t['mediated_direct_rows'], t['mediated_conditional_rows']):
        assert torch.equal(rids, rids2)
        rid = int(rids[0]); bi = int(t['anchor_read_indices'][rid, 0]); ai = int(t['anchor_read_E_slots'][rid])
        slots = (prepared.anchored_applied_EM[bi, ai] > 0).nonzero(as_tuple=False).flatten()
        alpha = t['mediated_source_alpha'][rids]
        selected = t['mediated_source_selected'][rids]
        expected = alpha[:, :1] * direct + (alpha[:, slots + 1] * selected[:, slots + 1]) @ conditional
        torch.testing.assert_close(t['mediated_environment_prior'][rids], expected)
        assert torch.all(conditional[mass == 0] == 0)
        rid = int(rids[0]); bi = int(t['anchor_read_indices'][rid, 0]); ai = int(t['anchor_read_E_slots'][rid])
        row = prepared.e_access_directory.full.row(bi, ai)
        slots = row.member_index[row.is_module].sort().values
        from honf_forward_core.interaction_reused_member_routing.anchored import build_e_access_measure
        mu = build_e_access_measure(scene.module_mask, scene.environment_mask, scene.environment_measure,
                                    dtype=conditional.dtype, access_measure=model.access_measure)[bi, scene.num_modules:]
        actual_mass = prepared.mediated_applied_ME[bi, slots] * mu
        torch.testing.assert_close(mass, actual_mass.sum(-1))
        weighted = actual_mass
        if mode == 'G':
            weighted = weighted * prepared.mediated_ee_geometry[bi, ai].exp()
        denom = weighted.sum(-1, keepdim=True)
        reference = torch.where(denom > 0, weighted / denom.clamp_min(torch.finfo(weighted.dtype).tiny), torch.zeros_like(weighted))
        consulted = selected[:, slots + 1].any(0)
        torch.testing.assert_close(conditional[consulted], reference[consulted])
    assert prepared.mediated_applied_ME.shape == (scene.batch_size, scene.num_modules, scene.num_environment_tokens)
    assert prepared.mediated_me_stage.endswith('/me')
    assert torch.isfinite(result.joint_fields).all()


@pytest.mark.parametrize('mode', ['P', 'G'])
def test_live_e_path_gradient_and_interventions(mode):
    model, scene, query = model_fixture(mode)
    prepared = model.prepare(scene)
    normal = model.decode(prepared, query, return_trace=True)
    t = normal.trace
    grad = torch.autograd.grad(t['context_E'].square().sum(), (model.member_router.member_M[0].weight, prepared.mediated_applied_ME), retain_graph=True, allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grad)
    blocked = model.decode(prepared, query, intervention={'remove_module_mediated_environment': True}, return_trace=True)
    muted = model.decode(prepared, query, intervention={'suppress_module_values': True}, return_trace=True)
    assert torch.count_nonzero(muted.trace['context_M']) == 0
    torch.testing.assert_close(muted.trace['context_E'], t['context_E'])
    assert not torch.allclose(blocked.trace['context_E'], t['context_E'])


def test_float32_paired_checkpoint_and_config_roundtrip():
    model, scene, query = model_fixture('G', 'local_e_cover', torch.float32)
    clone = AnchoredMemberRoutingCore.from_config(model.get_config())
    clone.load_state_dict(model.state_dict())
    model.train(); clone.train()
    p = model.prepare(scene); pc = clone.prepare(scene)
    first, second, _ = model.decode_paired(p, query)
    cf, cs, _ = clone.decode_paired(pc, query)
    torch.testing.assert_close(first.joint_fields, cf.joint_fields)
    torch.testing.assert_close(second.joint_fields, cs.joint_fields)
    (first.joint_fields.square().sum() + second.joint_fields.square().sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_no_m_selected_and_empty_me_still_reads_own():
    model, scene, query = model_fixture()
    model.source_retained_mass = 0.000001
    prepared = model.prepare(scene)
    prepared.mediated_applied_ME = torch.zeros_like(prepared.mediated_applied_ME)
    result = model.decode(prepared, query, intervention={'read_mode': 'restricted'}, return_trace=True)
    t = result.trace
    assert torch.isfinite(result.joint_fields).all()
    for rid, (_, _, _) in enumerate(zip(t['anchor_read_indices'], t['anchor_read_E_slots'], t['mediated_environment_prior'])):
        ai = int(t['anchor_read_E_slots'][rid])
        assert t['mediated_environment_prior'][rid, ai] > 0
    assert all(torch.count_nonzero(rows) == 0 for _, rows, _ in t['mediated_conditional_rows'])


def test_geometry_prior_has_live_gradient():
    model, scene, query = model_fixture('G')
    prepared = model.prepare(scene)
    result = model.decode(prepared, query, return_trace=True)
    grad, = torch.autograd.grad(result.trace['context_E'].square().sum(), prepared.mediated_ee_geometry)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0

@pytest.mark.parametrize('mode', ['P', 'G'])
def test_dense_reference_fields_and_gradients(mode):
    from honf_forward_core.interaction_reused_member_routing.anchored import anchored_row_read, build_e_access_measure
    from honf_forward_core.interaction_reused_member_routing.guided import prior_weighted_scores
    from honf_forward_core.interaction_reused_member_routing.relative import read_probability
    model, scene, query = model_fixture(mode, 'local_e_cover')
    oracle = copy.deepcopy(model)
    model.eval(); oracle.eval()
    actual = model.decode(model.prepare(scene), query).joint_fields
    p = oracle.prepare(scene); qs = oracle._query_states(p, query); qn = oracle._routing_features(qs)
    mu = build_e_access_measure(scene.module_mask, scene.environment_mask, scene.environment_measure, dtype=qs.dtype)
    mm, em = mu[:, :scene.num_modules], mu[:, scene.num_modules:]
    batches = []
    for b in range(query.batch_size):
        fields = []
        for q in range(query.num_queries):
            cm, ce = qs.new_zeros(oracle.context_dim), qs.new_zeros(oracle.context_dim)
            for k in range(query.anchor_indices.shape[-1]):
                a = int(query.anchor_indices[b, q, k]); aw = query.anchor_weights[b, q, k]
                dm = p.anchored_applied_EE[b, a] * em[b]
                dm = dm + torch.nn.functional.one_hot(torch.tensor(a), scene.num_environment_tokens).to(dm) * em[b, a]
                mp = p.anchored_applied_EM[b, a] * mm[b]
                mids = (mp > 0).nonzero(as_tuple=False).flatten()
                def scores(role, ids):
                    rows = torch.stack((torch.full_like(ids, b), torch.full_like(ids, q), ids), -1)
                    keys = p.member_keys_M if role == 'M' else p.member_keys_E
                    scorer = oracle.member_router.member_M if role == 'M' else oracle.member_router.member_E
                    return scorer(torch.cat((qn[b,q].expand(len(ids),-1), oracle._routing_features(keys)[b,ids], oracle._pair_geometry(scene, query, role, rows)), -1)).squeeze(-1)
                own = torch.tensor([a]); source_raw = torch.cat((scores('E', own), scores('M', mids)))
                source_prior = torch.cat((dm.sum()[None], mp[mids]))
                source_ell = prior_weighted_scores(source_raw[None], source_prior[None], torch.ones_like(source_prior[None],dtype=torch.bool))[0]
                alpha = torch.softmax(source_prior.log() + source_ell, -1)
                rows = p.mediated_applied_ME[b, mids] * em[b]
                if mode == 'G': rows = rows * p.mediated_ee_geometry[b, a].exp()
                mass = rows.sum(-1,keepdim=True)
                rows = torch.where(mass > 0, rows / mass.clamp_min(torch.finfo(rows.dtype).tiny), torch.zeros_like(rows))
                environmental = alpha[0] * dm / dm.sum() + alpha[1:] @ rows
                eids = (environmental > 0).nonzero(as_tuple=False).flatten()
                er = anchored_row_read(scores('E',eids)[None], environmental[eids], torch.ones_like(eids,dtype=torch.bool), p.member_values_E[b,eids], qn[b,q][None], oracle.typed_read_gates['E'], (scene.environment_mask[b] & (scene.environment_measure[b]>0)).sum(), torch.ones(1,dtype=torch.bool))
                if len(mids):
                    pm = alpha[1:] / alpha[1:].sum()
                    ent = -(pm * pm.log()).sum()
                    eligible = scene.module_mask[b].sum()
                    ent = torch.where(eligible > 1, ent / eligible.clamp_min(2).to(ent).log(), torch.zeros_like(ent))
                    gate = oracle.typed_read_gates['M'](torch.cat((qn[b,q], (pm*source_ell[1:]).sum()[None], ent[None]))[None])[0]
                    cm = cm + aw * read_probability(gate[0]-gate[1]) * (pm[:,None]*p.member_values_M[b,mids]).sum(0)
                ce = ce + aw * er.context[0]
            fields.append(oracle.member_router.joint_head(oracle.member_router.joint_trunk(torch.cat((qs[b,q],cm,ce)))))
        batches.append(torch.stack(fields))
    expected = torch.stack(batches)
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
    names=list(dict(model.named_parameters())); left=dict(model.named_parameters());right=dict(oracle.named_parameters())
    ga=torch.autograd.grad(actual.square().sum(),[left[n] for n in names]+[query.coordinates],allow_unused=True,retain_graph=True)
    gb=torch.autograd.grad(expected.square().sum(),[right[n] for n in names]+[query.coordinates],allow_unused=True)
    for name,a,b in zip(names+['query'],ga,gb):
        assert (a is None)==(b is None),name
        if a is not None: torch.testing.assert_close(a,b,rtol=3e-7,atol=3e-9,msg=name)


@pytest.mark.parametrize('mode',['P','G'])
def test_empty_physical_m_inventory(mode):
    from dataclasses import fields
    from tests.test_r13_8_anchored_core_helpers import _scene, _base_config
    from tests.test_faithful_member_routing import _query
    model = AnchoredMemberRoutingCore(_base_config(), mediated_mode=mode).eval()
    scene = _scene(0,4); query = _query()
    query = replace(query, **{f.name: v[:1] for f in fields(query) if torch.is_tensor(v := getattr(query,f.name)) and v.ndim and v.shape[0] == 2})
    query = replace(query, scene_key=scene.scene_key, coordinate_frame_key=scene.coordinate_frame_key, anchor_indices=torch.zeros((1,query.num_queries,1),dtype=torch.long), anchor_weights=torch.ones((1,query.num_queries,1)))
    prepared=model.prepare(scene)
    for read_mode in ('full','restricted'):
        output=model.decode(prepared,query,intervention={'read_mode':read_mode},return_trace=True)
        assert torch.isfinite(output.joint_fields).all()
        assert torch.count_nonzero(output.trace['context_M']) == 0
        assert torch.all(output.trace['mediated_environment_prior'][:,0] > 0)

@pytest.mark.parametrize('mode',['P','G'])
def test_native_fp32_high_contrast_own_access(mode):
    model, scene, query = model_fixture(mode, dtype=torch.float32)
    measure = scene.environment_measure.clone(); measure[:,0] = 1e-35
    scene = replace(scene, environment_measure=measure)
    prepared = model.prepare(scene)
    # Explicitly exercise a legal learned-EE-empty row with extremely weak
    # positive structural own mass and much larger active module priors.
    prepared.anchored_applied_EE = torch.zeros_like(prepared.anchored_applied_EE)
    output,_ = model._decode_tile(prepared,query,('full','restricted'),return_trace=True)
    for fields, trace, _ in output.values():
        assert torch.isfinite(fields).all()
        assert torch.all(trace['mediated_environment_prior'][:,0] > 0)
        use = (~trace['anchor_read_is_module']) & (trace['anchor_read_member_index'] == scene.num_modules)
        assert int(use.sum()) == len(trace['anchor_read_indices'])
        assert torch.all(trace['anchor_read_access_prior'][use] > 0)

@pytest.mark.parametrize('mode',['P','G'])
def test_padded_noncontiguous_original_ids_and_zero_area(mode):
    model, scene, query = model_fixture(mode)
    mids=scene.module_ids.clone(); mids[0]=torch.tensor([81,7,450]); emids=scene.environment_ids.clone(); emids[0]=torch.tensor([103,990,52,701]); measure=scene.environment_measure.clone(); measure[0,2]=0
    mask=scene.module_mask.clone(); mask[0,1]=False
    scene=replace(scene,module_ids=mids,environment_ids=emids,environment_measure=measure,module_mask=mask)
    out=model.decode(model.prepare(scene),query,return_trace=True)
    trace=out.trace; selected=trace['anchor_read_member_read_index']; batch=trace['anchor_read_indices'][selected,0]
    assert not bool(((batch == 0) & (trace['anchor_read_original_id'] == 7) & trace['anchor_read_is_module']).any())
    assert not bool(((batch == 0) & (trace['anchor_read_original_id'] == 52) & ~trace['anchor_read_is_module']).any())
    assert torch.isfinite(out.joint_fields).all()


def test_standalone_restricted_omits_unselected_me_conditioning():
    model,scene,query=model_fixture('G')
    model.source_retained_mass=0.01
    prepared=model.prepare(scene)
    prepared.anchored_applied_EM=prepared.anchored_applied_EM*1e-8
    output,_=model._decode_tile(prepared,query,('restricted',),return_trace=True)
    trace=output['restricted'][1]
    assert not trace['mediated_source_selected'][:,1:].any()
    assert trace['mediated_conditioned_anchor_module_rows'].numel()==0
    assert torch.count_nonzero(trace['context_M'])==0
    assert torch.isfinite(output['restricted'][0]).all()
