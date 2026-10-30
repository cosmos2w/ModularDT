"""R14 selected-message and checkpoint contracts."""
import copy
import math
import types
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from honf_forward_core.interaction_reused_member_routing.anchored import anchored_row_read
from honf_forward_core.interaction_reused_member_routing.r14 import DetailAnchoredMemberRoutingCore
from tests.anchored_fixtures import _base_config
from tests.test_e_anchored_packed_execution import fixture


def _model_scene_query(measure="inventory_unit_total"):
    old, scene, query = fixture("local_e_cover")
    model = DetailAnchoredMemberRoutingCore(_base_config(), query_tile_size=4, access_measure=measure).double().eval()
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    return model, scene, query


def _row_oracle(model, prepared, query, mode):
    state = model._query_states(prepared, query)
    norm = model._routing_features(state)
    table = getattr(prepared.e_access_directory, mode)
    result = {role: [] for role in ("M", "E")}
    for b in range(query.batch_size):
        scene_results = {role: [] for role in result}
        for q in range(query.num_queries):
            sums = {role: state.new_zeros(model.context_dim) for role in result}
            for k in range(query.anchor_indices.shape[-1]):
                row = table.row(b, int(query.anchor_indices[b, q, k]))
                for role in result:
                    chosen = row.is_module if role == "M" else ~row.is_module
                    slots = row.member_index[chosen] - (prepared.scene.num_modules if role == "E" else 0)
                    batch = torch.full_like(slots, b)
                    qi = torch.full_like(slots, q)
                    geom = model._pair_geometry(prepared.scene, query, role, torch.stack((batch, qi, slots), -1))
                    keys = prepared.member_keys_M if role == "M" else prepared.member_keys_E
                    scorer = model.member_router.member_M if role == "M" else model.member_router.member_E
                    raw = scorer(torch.cat((norm[b, q].expand(len(slots), -1),
                                            model._routing_features(keys)[b, slots], geom), -1)).squeeze(-1)
                    values, _ = model._selected_member_values(prepared, query, state, role, batch, qi, slots)
                    eligible = (prepared.scene.module_mask[b].sum() if role == "M" else
                                (prepared.scene.environment_mask[b] & (prepared.scene.environment_measure[b] > 0)).sum())
                    read = anchored_row_read(raw[None], row.access_weight[chosen], torch.ones_like(slots, dtype=torch.bool),
                                             values, norm[b, q][None], model.typed_read_gates[role], eligible,
                                             torch.ones(1, dtype=torch.bool), score_calibration="inherited")
                    sums[role] = sums[role] + query.anchor_weights[b, q, k] * read.context[0]
            for role in result:
                scene_results[role].append(sums[role])
        for role in result:
            result[role].append(torch.stack(scene_results[role]))
    contexts = {role: torch.stack(values) for role, values in result.items()}
    fused = torch.cat((state, contexts["M"], contexts["E"]), -1)
    return model.member_router.joint_head(model.member_router.joint_trunk(fused)), contexts


@pytest.mark.parametrize("mode", ["full", "restricted"])
@pytest.mark.parametrize("measure", ["inventory_unit_total", "active_domain"])
def test_selected_messages_match_independent_row_reductions(mode, measure):
    model, scene, query = _model_scene_query(measure)
    reference = copy.deepcopy(model)
    output = model.decode(model.prepare(scene), query, intervention={"read_mode": mode}, return_trace=True)
    expected, contexts = _row_oracle(reference, reference.prepare(scene), query, mode)
    torch.testing.assert_close(output.joint_fields, expected, atol=1e-11, rtol=1e-10)
    for role in ("M", "E"):
        torch.testing.assert_close(output.trace[f"context_{role}"], contexts[role], atol=1e-11, rtol=1e-10)
        assert output.work_counts[f"query_relative_value_pairs_{role}"] == output.work_counts[f"query_anchor_value_reduction_pairs_{role}"]
        assert output.trace[f"anchor_read_original_id"].ge(0).all()
    gradients = torch.autograd.grad(output.joint_fields.square().sum(),
                                    [query.coordinates, model.relative_value["M"].weight,
                                     model.local_value["E"].weight], retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)


def test_schema_strict_roundtrip_masks_and_paired_counts():
    model, scene, query = _model_scene_query()
    cloned = DetailAnchoredMemberRoutingCore.from_config(model.get_config()).double().eval()
    cloned.load_state_dict(model.state_dict(), strict=True)
    bad = dict(model.get_config(), locator_mode="e_anchor")
    with pytest.raises(ValueError):
        DetailAnchoredMemberRoutingCore.from_config(bad)
    mask = query.query_mask.clone()
    mask[0, 0] = False
    weights = query.anchor_weights.clone()
    weights[0, 0] = 0
    indices = query.anchor_indices.clone()
    indices[0, 0] = -1
    query = replace(query, query_mask=mask, anchor_weights=weights, anchor_indices=indices)
    prepared = model.prepare(scene)
    full, restricted, _ = model.decode_paired(prepared, query)
    for mode, output in (("full", full), ("restricted", restricted)):
        other = cloned.decode(cloned.prepare(scene), query, intervention={"read_mode": mode})
        torch.testing.assert_close(output.joint_fields, other.joint_fields, atol=0, rtol=0)
        assert output.joint_fields[0, 0].eq(0).all()
        for role in ("M", "E"):
            assert output.work_counts[f"query_relative_value_pairs_{role}"] == output.work_counts[f"query_anchor_value_reduction_pairs_{role}"]


def test_continuity_as_a_cover_row_enters_and_original_id_is_removed():
    model, scene, query = _model_scene_query()
    prepared = model.prepare(scene)
    outputs = []
    for offset in (-1e-5, 0., 1e-5):
        coords = query.coordinates.detach().clone()
        coords[:, :, 0] = 0.5 + offset
        # A smoothstep cover centered on the first row: the adjacent row's
        # mass vanishes continuously at the row transition.
        t = (coords[..., 0] - 0.5).clamp(0, 1)
        w = 3 * t.square() - 2 * t.pow(3)
        weights = torch.stack((1 - w, w), -1)
        indices = query.anchor_indices.clone()
        indices[..., 1] = torch.where(w > 0, indices[..., 1], -1)
        current = replace(query, coordinates=coords, anchor_indices=indices, anchor_weights=weights)
        output = model.decode(prepared, current, intervention={"read_mode": "restricted"}, return_trace=True)
        outputs.append(output.joint_fields)
    assert (outputs[0] - outputs[2]).abs().max() < 1e-3
    removed_id = int(scene.module_ids[0, 0])
    out = model.decode(prepared, current, intervention={"read_mode": "restricted", "access_intervention": "remove_member", "member_role": "M", "member_id": removed_id}, return_trace=True)
    ids = out.trace["anchor_read_original_id"]
    roles = out.trace["anchor_read_is_module"]
    assert not ((ids == removed_id) & roles).any()
    assert out.work_counts["query_relative_value_pairs_M"] == out.work_counts["query_anchor_value_reduction_pairs_M"]


@pytest.mark.parametrize("mode", ["full", "restricted"])
def test_factored_query_projection_matches_literal_pair_outputs_and_gradients(mode):
    model, scene, query = _model_scene_query()
    reference = copy.deepcopy(model)

    def literal_pair_values(self, prepared, query, qstates, role, batch, qi, slot):
        values = prepared.member_values_M if role == "M" else prepared.member_values_E
        geometry = self._pair_geometry(prepared.scene, query, role, torch.stack((batch, qi, slot), -1))
        relative = geometry[:, :self.coordinate_dim]
        gamma = [relative]
        for frequency in (1., 2., 4.):
            phase = 2 * math.pi * frequency * relative
            gamma.extend((phase.sin(), phase.cos()))
        message = F.silu(values[batch, slot] + self.query_value[role](qstates[batch, qi])
                         + self.relative_value[role](torch.cat(gamma, -1))
                         + self.local_value[role](geometry[:, -self.coordinate_dim:]))
        return message, {}

    reference._selected_member_values = types.MethodType(literal_pair_values, reference)
    out = model.decode(model.prepare(scene), query, intervention={"read_mode": mode})
    expected = reference.decode(reference.prepare(scene), query, intervention={"read_mode": mode})
    torch.testing.assert_close(out.joint_fields, expected.joint_fields, rtol=1e-10, atol=1e-11)
    names = tuple(dict(model.named_parameters()))
    params, ref_params = dict(model.named_parameters()), dict(reference.named_parameters())
    grads = torch.autograd.grad(out.joint_fields.square().sum(),
                                [params[n] for n in names] + [query.coordinates], allow_unused=True, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.joint_fields.square().sum(),
                                         [ref_params[n] for n in names] + [query.coordinates], allow_unused=True)
    for name, actual, wanted in zip(names + ("coordinates",), grads, expected_grads):
        assert (actual is None) == (wanted is None), name
        if actual is not None:
            torch.testing.assert_close(actual, wanted, rtol=2e-8, atol=2e-9, msg=name)
    for role in ("M", "E"):
        assert out.work_counts[f"query_value_projection_rows_{role}"] == query.batch_size * query.num_queries
        assert out.work_counts[f"query_relative_value_pairs_{role}"] == out.work_counts[f"query_anchor_value_reduction_pairs_{role}"]
