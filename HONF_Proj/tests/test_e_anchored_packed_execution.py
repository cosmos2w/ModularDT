"""Independent dense row reads versus compiled union execution."""
from dataclasses import fields, replace
import copy
import pytest
import torch
from tests.test_faithful_member_routing import _base_config, _scene, _query
from honf_forward_core.interaction_reused_member_routing.anchored import AnchoredMemberRoutingCore, anchored_row_read
from honf_forward_core.interaction_reused_member_routing.single_distribution_core import SingleDistributionMemberRoutingCore


def fixture(mode):
    scene = _scene()
    scene = replace(scene, **{f.name: v.double() for f in fields(scene)
                              if torch.is_tensor(v := getattr(scene, f.name)) and v.is_floating_point()})
    query = _query()
    coordinates = query.coordinates.double().detach().requires_grad_(True)
    shape = coordinates.shape[:2]
    if mode == "e_anchor":
        indices = torch.zeros((*shape, 1), dtype=torch.long)
        weights = torch.ones((*shape, 1), dtype=torch.double)
    else:
        indices = torch.stack((torch.zeros(shape, dtype=torch.long), torch.ones(shape, dtype=torch.long)), -1)
        w = coordinates[..., 0].sigmoid()
        weights = torch.stack((w, 1-w), -1)
    query = replace(query, coordinates=coordinates, anchor_indices=indices, anchor_weights=weights)
    model = AnchoredMemberRoutingCore(_base_config(), locator_mode=mode, query_tile_size=4).double()
    return model, scene, query


def dense_oracle(model, prepared, query, mode):
    qstates = model._query_states(prepared, query)
    qnorm = model._routing_features(qstates)
    table = getattr(prepared.e_access_directory, mode)
    contexts = {role: [] for role in ("M", "E")}
    for b in range(query.batch_size):
        outputs = {role: [] for role in contexts}
        for q in range(query.num_queries):
            local = {role: qstates.new_zeros(model.context_dim) for role in contexts}
            for k in range(query.anchor_indices.shape[-1]):
                anchor = int(query.anchor_indices[b,q,k])
                row = table.row(b, anchor)
                for role in contexts:
                    choose = row.is_module if role == "M" else ~row.is_module
                    slots = row.member_index[choose] - (prepared.scene.num_modules if role == "E" else 0)
                    keys = prepared.member_keys_M if role == "M" else prepared.member_keys_E
                    values = prepared.member_values_M if role == "M" else prepared.member_values_E
                    scorer = model.member_router.member_M if role == "M" else model.member_router.member_E
                    rows = torch.stack((torch.full_like(slots,b), torch.full_like(slots,q), slots), -1)
                    geometry = model._pair_geometry(prepared.scene, query, role, rows)
                    raw = scorer(torch.cat((qnorm[b,q].expand(len(slots),-1),
                                            model._routing_features(keys)[b,slots], geometry), -1)).squeeze(-1)
                    eligible = (prepared.scene.module_mask[b].sum() if role == "M" else
                                (prepared.scene.environment_mask[b] & (prepared.scene.environment_measure[b]>0)).sum())
                    result = anchored_row_read(raw[None], row.access_weight[choose],
                        torch.ones_like(slots,dtype=torch.bool), values[b,slots], qnorm[b,q][None],
                        model.typed_read_gates[role], eligible, torch.ones(1,dtype=torch.bool),
                        score_calibration=model.score_calibration)
                    local[role] = local[role] + query.anchor_weights[b,q,k]*result.context[0]
            for role in contexts:
                outputs[role].append(local[role])
        for role in contexts:
            contexts[role].append(torch.stack(outputs[role]))
    contexts = {role: torch.stack(rows) for role,rows in contexts.items()}
    fused = torch.cat((qstates,contexts["M"],contexts["E"]),-1)
    output = model.member_router.joint_head(model.member_router.joint_trunk(fused))
    return output, contexts


@pytest.mark.parametrize("locator", ["e_anchor", "local_e_cover"])
@pytest.mark.parametrize("mode", ["full", "restricted"])
def test_packed_matches_independent_dense_fields_contexts_and_live_gradients(locator, mode):
    model, scene, query = fixture(locator)
    oracle = copy.deepcopy(model)
    model.eval(); oracle.eval()
    prepared = model.prepare(scene)
    actual = model.decode(prepared,query,intervention={"read_mode":mode},return_trace=True)
    expected, contexts = dense_oracle(oracle,oracle.prepare(scene),query,mode)
    torch.testing.assert_close(actual.joint_fields,expected,rtol=1e-10,atol=1e-11)
    for role in contexts:
        torch.testing.assert_close(actual.trace[f"context_{role}"],contexts[role],rtol=1e-10,atol=1e-11)
    names = list(dict(model.named_parameters()))
    p = dict(model.named_parameters()); op = dict(oracle.named_parameters())
    actual_g = torch.autograd.grad(actual.joint_fields.square().sum(),[p[n] for n in names]+[query.coordinates],allow_unused=True,retain_graph=True)
    expected_g = torch.autograd.grad(expected.square().sum(),[op[n] for n in names]+[query.coordinates],allow_unused=True)
    for name,a,e in zip(names+["coordinates"],actual_g,expected_g):
        assert (a is None)==(e is None),name
        if a is not None:
            torch.testing.assert_close(a,e,rtol=2e-8,atol=2e-9,msg=name)


def test_common_initial_parameters_equal_u_without_entrance_parameters():
    u = SingleDistributionMemberRoutingCore(_base_config())
    a = AnchoredMemberRoutingCore(_base_config())
    assert all("proposal" not in name for name in a.state_dict())
    for name,value in a.state_dict().items():
        torch.testing.assert_close(value,u.state_dict()[name],rtol=0,atol=0,msg=name)


def test_paired_checkpoint_replay_matches_separate_reads_and_shares_raw_scores():
    model,scene,query = fixture("local_e_cover")
    separate=copy.deepcopy(model)
    model.query_tile_size=2; separate.query_tile_size=2
    prepared=model.prepare(scene)
    full,restricted,shared=model.decode_paired(prepared,query)
    reference=separate.prepare(scene)
    outputs=[separate.decode(reference,query,intervention={"read_mode":mode}) for mode in ("full","restricted")]
    for a,e in zip((full,restricted),outputs):
        torch.testing.assert_close(a.joint_fields,e.joint_fields,rtol=0,atol=0)
    (full.joint_fields.square().sum()+restricted.joint_fields.square().sum()).backward(retain_graph=True)
    sum(o.joint_fields.square().sum() for o in outputs).backward()
    for (name,p),(_,other) in zip(model.named_parameters(),separate.named_parameters()):
        assert (p.grad is None)==(other.grad is None),name
        if p.grad is not None:
            torch.testing.assert_close(p.grad,other.grad,rtol=2e-8,atol=2e-9,msg=name)
    assert shared["query_unique_union_raw_score_pairs_E"]==outputs[0].work_counts["query_unique_union_raw_score_pairs_E"]


@pytest.mark.parametrize("locator", ["e_anchor", "local_e_cover"])
@pytest.mark.parametrize("calibration", ["inherited", "bounded_tanh"])
@pytest.mark.parametrize("reference_change", ["raw", "directory"])
def test_live_raw_projection_reuse_fp32_paired_replay_and_gradients(locator, calibration, reference_change):
    model, scene, query = fixture(locator)
    model = model.float().train()
    model.score_calibration = calibration
    reference = copy.deepcopy(model)
    if reference_change == "raw":
        reference._reuse_raw_projections = False
    elif reference_change == "directory":
        reference._prepare_legacy_directory = True
    def live_scene():
        changes = {f.name: v.float().detach().clone().requires_grad_(True)
                   for f in fields(scene) if torch.is_tensor(v := getattr(scene, f.name))
                   and v.is_floating_point()}
        return replace(scene, **changes), changes
    scenes = [live_scene(), live_scene()]
    queries = []
    for _ in range(2):
        coords = query.coordinates.float().detach().clone().requires_grad_(True)
        weights = (torch.stack((coords[..., 0].sigmoid(), 1 - coords[..., 0].sigmoid()), -1)
                   if locator == "local_e_cover" else query.anchor_weights.float())
        queries.append(replace(query, coordinates=coords, anchor_weights=weights))
    outputs, grads = [], []
    for current, (s, inputs), q in zip((model, reference), scenes, queries):
        full, restricted, _ = current.decode_paired(current.prepare(s), q)
        outputs.append(torch.stack((full.joint_fields, restricted.joint_fields)))
        targets = list(inputs.values()) + [q.coordinates] + list(current.parameters())
        grads.append(torch.autograd.grad(outputs[-1].square().sum(), targets, allow_unused=True))
    output_diff = (outputs[0] - outputs[1]).abs().max().item()
    grad_diffs = []
    for actual, expected in zip(*grads):
        assert (actual is None) == (expected is None)
        if actual is not None:
            grad_diffs.append((actual - expected).abs().max().item() if actual.numel() else 0.)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-5, atol=2e-7)
    print(f"FP32 {reference_change} {locator} {calibration}: output max {output_diff:.9g}, gradient max {max(grad_diffs):.9g}")


def test_anchored_preparation_omits_legacy_directory_and_keeps_actual_stages():
    model, scene, query = fixture("e_anchor")
    prepared = model.prepare(scene, trace_request={"include": []})
    assert not prepared.has_legacy_member_directory
    assert not hasattr(prepared, "B_M") and not hasattr(prepared, "B_E")
    assert all(prepared.final_applied_context_stages[route]["snapshot"] is not None
               for route in ("mm", "em", "ee", "me"))
    output = model.decode(prepared, query, return_trace=True)
    assert output.trace["anchored_applied_EM"] is prepared.anchored_applied_EM
    assert output.trace["anchored_applied_EE"] is prepared.anchored_applied_EE
    with pytest.raises(ValueError, match="E-directory"):
        model.member_routing_intervention("b_fixed_candidates_scores", prepared, query)
    _, options = model.member_routing_intervention("remove_contextual_message_recompute", prepared, query)
    removed = model.prepare(scene, **options)
    assert removed.e_access_directory is not None


@pytest.mark.parametrize("locator", ["e_anchor", "local_e_cover"])
@pytest.mark.parametrize("mode", ["full", "restricted"])
def test_bounded_tanh_packed_matches_independent_row_execution(locator, mode):
    model, scene, query = fixture(locator)
    model.score_calibration = "bounded_tanh"
    model.eval()
    prepared = model.prepare(scene)
    actual = model.decode(prepared, query, intervention={"read_mode": mode}, return_trace=True)
    expected, contexts = dense_oracle(model, prepared, query, mode)
    torch.testing.assert_close(actual.joint_fields, expected, rtol=1e-10, atol=1e-11)
    for role in contexts:
        torch.testing.assert_close(actual.trace[f"context_{role}"], contexts[role], rtol=1e-10, atol=1e-11)


def test_singleton_calibration_and_raw_score_trace_contract():
    model, scene, query = fixture("e_anchor")
    gate = model.typed_read_gates["E"]
    raw = torch.tensor([[2.]], dtype=torch.double)
    normal = torch.tensor([torch.finfo(torch.double).tiny], dtype=torch.double)
    subnormal = normal * .5
    arguments = (torch.ones(1, dtype=torch.bool), torch.ones(1, model.context_dim, dtype=torch.double),
                 torch.zeros(1, model.context_dim, dtype=torch.double), gate, 2,
                 torch.ones(1, dtype=torch.bool))
    regular = anchored_row_read(raw, normal, *arguments)
    underflow = anchored_row_read(raw, subnormal, *arguments)
    bounded = anchored_row_read(raw, normal, *arguments, score_calibration="bounded_tanh")
    assert regular.calibrated_scores.item() == regular.expectation.item() == regular.normalized_entropy.item() == 0.
    assert regular.probabilities.item() == 1.
    assert underflow.calibrated_scores.item() != 0.
    assert bounded.calibrated_scores.item() != 0.
    traced = model.decode(model.prepare(scene), query, return_trace=True).trace
    assert traced["member_score_union_values_E"].numel() > 0


@pytest.mark.parametrize("locator", ["e_anchor", "local_e_cover"])
def test_packed_singleton_and_empty_role_with_traces(locator):
    model, scene, query = fixture(locator)
    model.eval()
    output = model.decode(model.prepare(scene), query,
        intervention={"read_mode": "restricted", "access_intervention": "self_only"}, return_trace=True)
    assert torch.count_nonzero(output.trace["context_M"]) == 0
    assert torch.count_nonzero(output.trace["anchor_read_calibrated_scores"]) == 0
    assert output.trace["member_score_union_values_E"].numel() > 0


def test_one_anchor_local_cover_equals_a_and_row_only_m_fades_with_lambda():
    from honf_forward_core.interaction_reused_member_routing.anchored import compile_e_access_directory
    from honf_forward_core.interaction_reused_member_routing.joint import build_joint_member_measure
    a,scene,query=fixture('e_anchor')
    local=AnchoredMemberRoutingCore(_base_config(),locator_mode='local_e_cover',query_tile_size=4).double().eval()
    a.eval();local.load_state_dict(a.state_dict(),strict=True)
    actual=a.decode(a.prepare(scene),query,return_trace=True)
    one=local.decode(local.prepare(scene),query,return_trace=True)
    torch.testing.assert_close(actual.joint_fields,one.joint_fields,rtol=0,atol=0)
    prepared=local.prepare(scene)
    em=prepared.anchored_applied_EM.clone()
    em[:,1]=0
    mu=build_joint_member_measure(scene.module_mask,scene.environment_mask,scene.environment_measure)
    # A fixed-state access intervention fixture: one row has no M permission.
    directory=compile_e_access_directory(em,prepared.anchored_applied_EE,mu[:,:scene.num_modules],
        mu[:,scene.num_modules:],scene.module_mask,scene.environment_mask,scene.module_ids,scene.environment_ids)
    prepared.e_access_directory=directory
    prepared._anchored_directory_versions=local._directory_versions(directory)
    indices=torch.stack((torch.zeros(query.coordinates.shape[:2],dtype=torch.long),
                         torch.ones(query.coordinates.shape[:2],dtype=torch.long)),-1)
    contexts=[]
    for weight in (1.,.25,0.):
        weights=query.coordinates.new_empty((*query.coordinates.shape[:2],2))
        weights[...,0]=weight;weights[...,1]=1-weight
        q=replace(query,anchor_indices=indices,anchor_weights=weights)
        contexts.append(local.decode(prepared,q,return_trace=True).trace['context_M'])
    torch.testing.assert_close(contexts[1],contexts[0]*.25,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(contexts[2],torch.zeros_like(contexts[2]),rtol=0,atol=0)
