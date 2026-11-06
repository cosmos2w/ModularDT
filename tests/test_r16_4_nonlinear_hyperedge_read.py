"""Direct row equations and strict C/G integration for the R16-4 read."""
import copy
from dataclasses import fields, replace

import pytest
import torch
from torch.nn import functional as F

from channelthermal.member_routing import (
    ThermalDetailAnchoredMemberRoutingAdapter, ThermalHyperedgeNonlinearReadAdapter,
)
from honf_forward_core.interaction_reused_member_routing import (
    DetailAnchoredMemberRoutingCore, HyperedgeNonlinearReadCore, anchored_core_from_config,
    anchored_row_read, compile_e_access_directory,
)
from honf_forward_core.interaction_reused_member_routing.primitives import build_joint_member_measure
from tests.anchored_fixtures import _base_config
from tests.test_e_anchored_packed_execution import fixture
from tests.test_r14_runtime_integration import STATS


def _case(mode):
    _, scene, query = fixture("local_e_cover")
    old = DetailAnchoredMemberRoutingCore(_base_config(), query_tile_size=4).double().eval()
    config = HyperedgeNonlinearReadCore.from_detail_config(old.get_config(), mode)
    model = HyperedgeNonlinearReadCore.from_config(config).double().eval()
    model.initialize_detail_weights(old.state_dict())
    return old, model, scene, query


def _direct_rows(model, prepared, query, read_mode, *, capture_row_variables=False):
    """Literal original-member reads, independent of packed coalescing."""
    states = model._query_states(prepared, query)
    normalized = model._routing_features(states)
    table = getattr(prepared.e_access_directory, read_mode)
    batch_fields, row_variables = [], []
    for b in range(query.batch_size):
        fields = []
        for q in range(query.num_queries):
            blended = {r: states.new_zeros(model.context_dim) for r in ("M", "E")}
            row_hidden = states.new_zeros(32)
            for k in range(query.anchor_indices.shape[-1]):
                weight = query.anchor_weights[b, q, k]
                if not bool(weight.detach() > 0):
                    continue
                row = table.row(b, int(query.anchor_indices[b, q, k]))
                row_context = {}
                for role in ("M", "E"):
                    selected = row.is_module if role == "M" else ~row.is_module
                    slots = row.member_index[selected] - (prepared.scene.num_modules if role == "E" else 0)
                    bb, qq = torch.full_like(slots, b), torch.full_like(slots, q)
                    geometry = model._pair_geometry(prepared.scene, query, role, torch.stack((bb, qq, slots), -1))
                    keys = prepared.member_keys_M if role == "M" else prepared.member_keys_E
                    scorer = model.member_router.member_M if role == "M" else model.member_router.member_E
                    scores = scorer(torch.cat((normalized[b, q].expand(len(slots), -1),
                                               model._routing_features(keys)[b, slots], geometry), -1)).squeeze(-1)
                    values, _ = model._selected_member_values(prepared, query, states, role, bb, qq, slots)
                    count = (prepared.scene.module_mask[b].sum() if role == "M" else
                             (prepared.scene.environment_mask[b] & (prepared.scene.environment_measure[b] > 0)).sum())
                    result = anchored_row_read(scores[None], row.access_weight[selected],
                                               torch.ones_like(slots, dtype=torch.bool), values,
                                               normalized[b, q][None], model.typed_read_gates[role], count,
                                               torch.ones(1, dtype=torch.bool),
                                               score_calibration=model.score_calibration)
                    row_variables.append((result.strength, result.probabilities, result.context[0]))
                    row_context[role] = result.context[0]
                    blended[role] = blended[role] + weight * row_context[role]
                residual = model.hyperedge_read
                h = F.silu(F.linear(row_context["M"], residual.weight_M)
                           + F.linear(row_context["E"], residual.weight_E) + residual.bias) - F.silu(residual.bias)
                row_hidden = row_hidden + weight * h
            if model.hyperedge_mode == "C":
                residual = model.hyperedge_read
                row_hidden = F.silu(F.linear(blended["M"], residual.weight_M)
                                    + F.linear(blended["E"], residual.weight_E) + residual.bias) - F.silu(residual.bias)
            fused = torch.cat((states[b, q], blended["M"], blended["E"]))
            trunk = model.member_router.joint_trunk(fused) + F.linear(row_hidden, model.hyperedge_read.output.weight)
            fields.append(model.member_router.joint_head(trunk) * query.query_mask[b, q])
        batch_fields.append(torch.stack(fields))
    fields = torch.stack(batch_fields)
    return (fields, row_variables) if capture_row_variables else fields


@pytest.mark.parametrize("mode", ["C", "G"])
@pytest.mark.parametrize("read_mode", ["full", "restricted"])
def test_nonzero_residual_matches_direct_row_equation_and_gradients(mode, read_mode):
    _, model, scene, query = _case(mode)
    # Force several row accumulation chunks even in this small scene.
    model._row_reduction_chunk_size = 3
    with torch.no_grad():
        model.hyperedge_read.output.weight.copy_(
            torch.linspace(-0.08, 0.07, model.hyperedge_read.output.weight.numel()).reshape_as(
                model.hyperedge_read.output.weight))
        model.hyperedge_read.bias.copy_(torch.linspace(-0.2, 0.15, 32))
    reference = copy.deepcopy(model)
    def local_query():
        return replace(query, coordinates=query.coordinates.detach().clone().requires_grad_(),
                       anchor_weights=query.anchor_weights.detach().clone().requires_grad_())
    actual_query, direct_query = local_query(), local_query()
    prepared, direct_prepared = model.prepare(scene), reference.prepare(scene)
    actual = model.decode(prepared, actual_query, intervention={"read_mode": read_mode}, return_trace=True)
    expected, row_variables = _direct_rows(reference, direct_prepared, direct_query, read_mode,
                                           capture_row_variables=True)
    torch.testing.assert_close(actual.joint_fields, expected, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(actual.joint_fields.square().sum(), expected.square().sum(), rtol=1e-10, atol=1e-11)
    if mode == "G":
        assert actual.trace["hyperedge_row_context_space"] == "role_value_projection_before_silu"
        for role_index, role in enumerate(("M", "E")):
            weight = reference.hyperedge_read.weight_M if role == "M" else reference.hyperedge_read.weight_E
            expected_rows = torch.stack([F.linear(value[2], weight) for value in row_variables[role_index::2]])
            torch.testing.assert_close(actual.trace[f"hyperedge_row_projected_{role}"],
                                       expected_rows, rtol=1e-10, atol=1e-11)
    names = ("hyperedge_read.weight_M", "hyperedge_read.weight_E", "hyperedge_read.bias",
             "hyperedge_read.output.weight", "typed_read_gates.M.0.weight", "member_router.member_E.0.weight",
             "query_value.M.weight", "relative_value.E.weight")
    parameters, direct_parameters = dict(model.named_parameters()), dict(reference.named_parameters())
    source = getattr(prepared.e_access_directory, read_mode).access_weight
    source_direct = getattr(direct_prepared.e_access_directory, read_mode).access_weight
    targets = [parameters[n] for n in names] + [actual_query.coordinates, actual_query.anchor_weights, source]
    expected_targets = [direct_parameters[n] for n in names] + [direct_query.coordinates,
                                                                direct_query.anchor_weights, source_direct]
    probabilities = [(index, value[1]) for index, value in enumerate(row_variables) if value[1].requires_grad]
    gradients = torch.autograd.grad(actual.joint_fields.square().sum(), targets + [
        actual.trace["anchor_read_strength"], actual.trace["anchor_read_probabilities"]], allow_unused=True)
    direct_gradients = torch.autograd.grad(expected.square().sum(), expected_targets + [
        value[0] for value in row_variables] + [value for _, value in probabilities], allow_unused=True)
    base_count = len(targets)
    for name, got, wanted in zip(names + ("coordinates", "lambda", "prior"),
                                  gradients[:base_count], direct_gradients[:base_count]):
        assert (got is None) == (wanted is None), name
        if got is not None:
            torch.testing.assert_close(got, wanted, rtol=2e-8, atol=2e-9, msg=name)
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients[:base_count])
    direct_strengths = torch.stack([g.reshape(()) for g in
                                    direct_gradients[base_count:base_count + len(row_variables)]]).reshape(-1, 2)
    torch.testing.assert_close(gradients[base_count], direct_strengths, rtol=2e-8, atol=2e-9)
    direct_probabilities = torch.zeros_like(gradients[base_count + 1])
    rid = actual.trace["anchor_read_member_read_index"]
    is_module = actual.trace["anchor_read_is_module"]
    for (index, _), grad in zip(probabilities, direct_gradients[base_count + len(row_variables):]):
        mask = (rid == index // 2) & (is_module == (index % 2 == 0))
        direct_probabilities[mask] = grad.reshape(-1)
    torch.testing.assert_close(gradients[base_count + 1], direct_probabilities, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("read_mode", ["full", "restricted"])
def test_float32_projected_rows_match_canonical_fields_loss_and_gradients(read_mode):
    _, model, scene, query = _case("G")
    model = model.float()
    model._row_reduction_chunk_size = 3
    with torch.no_grad():
        model.hyperedge_read.output.weight.copy_(
            torch.linspace(-.08, .07, model.hyperedge_read.output.weight.numel()).reshape_as(
                model.hyperedge_read.output.weight))
        model.hyperedge_read.bias.copy_(torch.linspace(-.2, .15, 32))
    reference = copy.deepcopy(model)
    scene = replace(scene, **{f.name: value.float() for f in fields(scene)
                              if torch.is_tensor(value := getattr(scene, f.name)) and value.is_floating_point()})
    def local_query():
        return replace(query, coordinates=query.coordinates.detach().float().requires_grad_(),
                       anchor_weights=query.anchor_weights.detach().float().requires_grad_())
    actual_query, direct_query = local_query(), local_query()
    prepared, direct_prepared = model.prepare(scene), reference.prepare(scene)
    actual = model.decode(prepared, actual_query, intervention={"read_mode": read_mode}).joint_fields
    expected = _direct_rows(reference, direct_prepared, direct_query, read_mode)
    actual_loss, expected_loss = actual.square().sum(), expected.square().sum()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-7)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-5, atol=3e-7)
    names = ("hyperedge_read.weight_M", "hyperedge_read.weight_E", "hyperedge_read.bias",
             "hyperedge_read.output.weight", "typed_read_gates.M.0.weight", "query_value.M.weight")
    params, ref_params = dict(model.named_parameters()), dict(reference.named_parameters())
    targets = [params[n] for n in names] + [actual_query.coordinates, actual_query.anchor_weights,
                                           getattr(prepared.e_access_directory, read_mode).access_weight]
    ref_targets = [ref_params[n] for n in names] + [direct_query.coordinates, direct_query.anchor_weights,
                                                    getattr(direct_prepared.e_access_directory, read_mode).access_weight]
    gradients = torch.autograd.grad(actual_loss, targets)
    expected_gradients = torch.autograd.grad(expected_loss, ref_targets)
    for name, actual_grad, expected_grad in zip(names + ("coordinates", "lambda", "prior"),
                                                gradients, expected_gradients):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=2e-6, msg=name)


@pytest.mark.parametrize("mode", ["C", "G"])
def test_zero_parity_strict_historical_and_thermal_loading(mode):
    old, model, scene, query = _case(mode)
    other = anchored_core_from_config(model.get_config()).double().eval()
    other.load_state_dict(model.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        other.load_state_dict(old.state_dict(), strict=True)
    with pytest.raises(ValueError):
        model.initialize_detail_weights({k: v for k, v in old.state_dict().items() if k != next(iter(old.state_dict()))})
    historical = DetailAnchoredMemberRoutingCore.from_config(old.get_config()).double().eval()
    historical.load_state_dict(old.state_dict(), strict=True)
    old_adapter = ThermalDetailAnchoredMemberRoutingAdapter(STATS, core=old)
    assert ThermalDetailAnchoredMemberRoutingAdapter.from_state_payload(
        old_adapter.state_payload(), state_dict=old_adapter.state_dict()).get_config() == old_adapter.get_config()
    adapter = ThermalHyperedgeNonlinearReadAdapter(STATS, core=model)
    restored = ThermalHyperedgeNonlinearReadAdapter.from_state_payload(
        adapter.state_payload(), state_dict=adapter.state_dict()).double().eval()
    assert restored.get_config() == adapter.get_config()
    for read_mode in ("full", "restricted"):
        base = old.decode(old.prepare(scene), query, intervention={"read_mode": read_mode})
        actual = restored.core.decode(restored.core.prepare(scene), query, intervention={"read_mode": read_mode})
        torch.testing.assert_close(actual.joint_fields, base.joint_fields, rtol=0, atol=0)
        ga = torch.autograd.grad(actual.joint_fields.sum(), query.coordinates, retain_graph=True)[0]
        gb = torch.autograd.grad(base.joint_fields.sum(), query.coordinates, retain_graph=True)[0]
        torch.testing.assert_close(ga, gb, rtol=0, atol=0)
    names = [model.hyperedge_read.output.weight, model.hyperedge_read.weight_M,
             model.hyperedge_read.weight_E, model.hyperedge_read.bias]
    prediction = model.decode(model.prepare(scene), query).joint_fields
    gradients = torch.autograd.grad(prediction.square().sum(), names)
    assert gradients[0].abs().sum() > 0
    assert all(g.eq(0).all() for g in gradients[1:])


def test_same_initialization_parameter_count_and_rng_isolation():
    old, c, _, _ = _case("C")
    state = torch.random.get_rng_state().clone()
    g = HyperedgeNonlinearReadCore.from_config(HyperedgeNonlinearReadCore.from_detail_config(old.get_config(), "G"))
    assert torch.equal(state, torch.random.get_rng_state())
    for key in HyperedgeNonlinearReadCore.NEW_STATE_KEYS:
        torch.testing.assert_close(c.state_dict()[key], g.state_dict()[key].double(), rtol=0, atol=0)
    base = _base_config()
    base["context_dim"] = 128
    wide = HyperedgeNonlinearReadCore(base, hyperedge_mode="G")
    assert sum(p.numel() for p in wide.hyperedge_read.parameters()) == 12320
    assert torch.count_nonzero(wide.hyperedge_read.output.weight) == 0
    assert torch.count_nonzero(wide.hyperedge_read.bias) == 0


def test_grouped_permutation_lambda_empty_roles_masks_and_collapse():
    _, model, scene, query = _case("G")
    with torch.no_grad():
        model.hyperedge_read.output.weight.fill_(0.07)
    prepared = model.prepare(scene)
    original = model.decode(prepared, query, return_trace=True)
    permuted = replace(query, anchor_indices=query.anchor_indices.flip(-1), anchor_weights=query.anchor_weights.flip(-1))
    reordered = model.decode(prepared, permuted, return_trace=True)
    torch.testing.assert_close(original.joint_fields, reordered.joint_fields, rtol=1e-12, atol=1e-12)
    mask = query.query_mask.clone(); mask[0, 0] = False
    anchors = query.anchor_indices.clone(); anchors[0, 0] = -1
    weights = query.anchor_weights.clone(); weights[0, 0] = 0
    masked = replace(query, query_mask=mask, anchor_indices=anchors, anchor_weights=weights)
    output = model.decode(prepared, masked, intervention={"access_intervention": "self_only"}, return_trace=True)
    assert output.joint_fields[0, 0].eq(0).all()
    assert output.trace["hyperedge_row_projected_M"].eq(0).all()
    assert output.trace["hyperedge_row_context_space"] == "role_value_projection_before_silu"
    assert output.work_counts["query_hyperedge_row_reduction_pairs_M"] == 0
    for role in ("M", "E"):
        assert output.work_counts[f"query_hyperedge_row_reduction_channels_{role}"] == (
            output.work_counts[f"query_hyperedge_row_reduction_pairs_{role}"] * 32)
    assert output.work_counts["query_hyperedge_active_address_rows"] == int((weights > 0).sum())
    removed_id = int(scene.module_ids[0, 0])
    removed = model.decode(prepared, query, intervention={"access_intervention": "remove_member",
                           "member_role": "M", "member_id": removed_id}, return_trace=True)
    assert not ((removed.trace["anchor_read_original_id"] == removed_id)
                & removed.trace["anchor_read_is_module"]).any()
    collapsed = model.decode(prepared, query, intervention={"collapse_hyperedge_rows": True}, return_trace=True)
    assert collapsed.trace["hyperedge_rows_collapsed"]
    assert not torch.equal(collapsed.joint_fields, original.joint_fields)
    corrections = []
    for first in (1., 0., .25):
        w = torch.empty_like(query.anchor_weights)
        w[..., 0] = first; w[..., 1] = 1 - first
        corrections.append(model.decode(prepared, replace(query, anchor_weights=w), return_trace=True)
                           .trace["hyperedge_correction"])
    torch.testing.assert_close(corrections[2], .25 * corrections[0] + .75 * corrections[1],
                               rtol=1e-11, atol=1e-12)


def test_empty_actual_rows_are_exactly_zero_with_nonzero_bias():
    _, model, scene, query = _case("G")
    with torch.no_grad():
        model.hyperedge_read.output.weight.fill_(.2)
        model.hyperedge_read.bias.fill_(.4)
    prepared = model.prepare(scene)
    directory = prepared.e_access_directory
    def empty(table):
        return replace(table, offsets=torch.zeros_like(table.offsets),
                       member_index=table.member_index[:0], original_id=table.original_id[:0],
                       is_module=table.is_module[:0], access_weight=table.access_weight[:0],
                       applied_exchange=table.applied_exchange[:0], structural_self=table.structural_self[:0],
                       measure=table.measure[:0])
    prepared.e_access_directory = replace(directory, full=empty(directory.full), restricted=empty(directory.restricted))
    prepared._anchored_directory_versions = model._directory_versions(prepared.e_access_directory)
    out = model.decode(prepared, query, return_trace=True)
    assert out.trace["hyperedge_row_projected_M"].eq(0).all()
    assert out.trace["hyperedge_row_projected_E"].eq(0).all()
    assert out.trace["hyperedge_correction"].eq(0).all()
    assert out.work_counts["query_hyperedge_row_reduction_pairs_M"] == 0
    assert out.work_counts["query_hyperedge_row_reduction_pairs_E"] == 0


def test_row_values_do_not_disappear_when_lambda_blend_underflows():
    _, model, scene, query = _case("G")
    prepared = model.prepare(scene)
    measure = build_joint_member_measure(scene.module_mask, scene.environment_mask,
                                         scene.environment_measure, dtype=query.coordinates.dtype)
    directory = compile_e_access_directory(
        torch.zeros_like(prepared.anchored_applied_EM), torch.zeros_like(prepared.anchored_applied_EE),
        measure[:, :scene.num_modules], measure[:, scene.num_modules:], scene.module_mask,
        scene.environment_mask, scene.module_ids, scene.environment_ids)
    prepared.e_access_directory = directory
    prepared._anchored_directory_versions = model._directory_versions(directory)
    results = []
    for weight in (.5, torch.nextafter(query.coordinates.new_zeros(()), query.coordinates.new_ones(())).item()):
        weights = torch.empty_like(query.anchor_weights)
        weights[..., 0] = 1 - weight; weights[..., 1] = weight
        out = model.decode(prepared, replace(query, anchor_weights=weights), return_trace=True)
        second = out.trace["anchor_read_indices"][:, 2] == 1
        results.append(out.trace["hyperedge_row_projected_E"][second])
    assert results[0].abs().sum() > 0
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    assert out.work_counts["query_relative_value_pairs_E"] > out.work_counts["query_anchor_coalesced_value_pairs_E"]


def test_collapsed_grouped_read_matches_matched_capacity_control():
    _, grouped, scene, query = _case("G")
    _, control, _, _ = _case("C")
    with torch.no_grad():
        grouped.hyperedge_read.output.weight.copy_(
            torch.linspace(-.3, .2, grouped.hyperedge_read.output.weight.numel()).reshape_as(
                grouped.hyperedge_read.output.weight))
    control.load_state_dict(grouped.state_dict(), strict=True)
    g = grouped.decode(grouped.prepare(scene), query, intervention={"collapse_hyperedge_rows": True})
    c = control.decode(control.prepare(scene), query)
    torch.testing.assert_close(g.joint_fields, c.joint_fields, rtol=1e-11, atol=1e-12)


def test_paired_mode_reuses_unique_values_and_matches_separate_gradient():
    _, model, scene, query = _case("G")
    model.train(); model.query_tile_size = 2
    with torch.no_grad():
        model.hyperedge_read.output.weight.fill_(0.04)
    reference = copy.deepcopy(model)
    reference._share_paired_selected_values = False
    actual = model.decode_paired(model.prepare(scene), query)
    direct = reference.decode_paired(reference.prepare(scene), query)
    for a, b in zip(actual[:2], direct[:2]):
        torch.testing.assert_close(a.joint_fields, b.joint_fields, rtol=1e-12, atol=1e-12)
    parameters, other_parameters = dict(model.named_parameters()), dict(reference.named_parameters())
    names = tuple(parameters)
    loss = actual[0].joint_fields.square().sum() + .37 * actual[1].joint_fields.square().sum()
    other_loss = direct[0].joint_fields.square().sum() + .37 * direct[1].joint_fields.square().sum()
    ga = torch.autograd.grad(loss, [parameters[n] for n in names], allow_unused=True)
    gb = torch.autograd.grad(other_loss, [other_parameters[n] for n in names], allow_unused=True)
    for name, a, b in zip(names, ga, gb):
        assert (a is None) == (b is None), name
        if a is not None:
            torch.testing.assert_close(a, b, rtol=2e-8, atol=2e-9, msg=name)
    for role in ("M", "E"):
        key = f"query_relative_value_pairs_{role}"
        assert actual[2][key] == direct[0].work_counts[key]
        assert actual[0].work_counts[key] == actual[1].work_counts[key] == 0
        projected = f"query_hyperedge_value_projection_pairs_{role}"
        macs = f"query_hyperedge_value_projection_macs_{role}"
        assert actual[2][projected] == actual[2][key]
        assert actual[2][macs] == actual[2][projected] * model.context_dim * 32
        assert actual[0].work_counts[projected] == actual[1].work_counts[projected] == 0
        assert direct[0].work_counts[projected] == direct[0].work_counts[key]
        assert direct[1].work_counts[projected] == direct[1].work_counts[key]


def test_grouped_paired_checkpoint_supports_spatial_derivative_gradients():
    _, model, scene, query = _case("G")
    model.train(); model.query_tile_size = 2
    model._row_reduction_chunk_size = 3
    with torch.no_grad():
        model.hyperedge_read.output.weight.fill_(.04)
    reference = copy.deepcopy(model)
    reference._share_paired_selected_values = False
    def objective(net):
        coords = query.coordinates.detach().clone().requires_grad_()
        weights = coords[..., 0].sigmoid()
        current = replace(query, coordinates=coords, anchor_weights=torch.stack((weights, 1 - weights), -1))
        full, restricted, _ = net.decode_paired(net.prepare(scene), current)
        loss = coords.new_zeros(())
        for scale, result in ((1., full), (.37, restricted)):
            gradient = torch.autograd.grad(result.joint_fields[..., 3].sum(), coords, create_graph=True)[0]
            loss = loss + scale * gradient.square().sum()
        return loss
    actual, expected = objective(model), objective(reference)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    left, right = dict(model.named_parameters()), dict(reference.named_parameters())
    names = tuple(left)
    ga = torch.autograd.grad(actual, [left[n] for n in names], allow_unused=True)
    gb = torch.autograd.grad(expected, [right[n] for n in names], allow_unused=True)
    for name, a, b in zip(names, ga, gb):
        assert (a is None) == (b is None), name
        if a is not None:
            torch.testing.assert_close(a, b, rtol=5e-8, atol=5e-9, msg=name)


def test_multichunk_checkpoint_spatial_derivatives_match_direct_rows():
    _, model, scene, query = _case("G")
    model.train(); model.query_tile_size = 2; model._row_reduction_chunk_size = 3
    with torch.no_grad():
        model.hyperedge_read.output.weight.fill_(.04)
    reference = copy.deepcopy(model)

    def objective(net, direct):
        coordinates = query.coordinates.detach().clone().requires_grad_()
        weight = coordinates[..., 0].sigmoid()
        local = replace(query, coordinates=coordinates,
                        anchor_weights=torch.stack((weight, 1 - weight), -1))
        prepared = net.prepare(scene)
        fields = (_direct_rows(net, prepared, local, "restricted") if direct else
                  net.decode(prepared, local, intervention={"read_mode": "restricted"}).joint_fields)
        spatial = torch.autograd.grad(fields[..., 3].sum(), coordinates, create_graph=True)[0]
        return spatial.square().sum(), coordinates

    actual, coordinates = objective(model, False)
    expected, direct_coordinates = objective(reference, True)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    params, ref_params = dict(model.named_parameters()), dict(reference.named_parameters())
    names = tuple(params)
    gradients = torch.autograd.grad(actual, [params[n] for n in names] + [coordinates], allow_unused=True)
    expected_gradients = torch.autograd.grad(expected, [ref_params[n] for n in names] + [direct_coordinates],
                                             allow_unused=True)
    for name, left, right in zip(names + ("coordinates",), gradients, expected_gradients):
        assert (left is None) == (right is None), name
        if left is not None:
            torch.testing.assert_close(left, right, rtol=5e-8, atol=5e-9, msg=name)
