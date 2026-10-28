from __future__ import annotations

import torch
from torch import nn
from dataclasses import replace

import pytest

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2, QueryBatch, SceneBatch
from honf_forward_core.interaction_reused_member_routing import AnchoredMemberRoutingCore
from honf_forward_core.interaction_reused_member_routing.anchored import (
    anchored_row_read,
    blend_anchored_contexts,
    compile_e_access_directory,
)
from honf_forward_core.interaction_reused_member_routing.joint import build_joint_member_measure
from honf_forward_core.interaction_reused_member_routing.guided import prior_weighted_scores
from honf_forward_core.interaction_reused_member_routing.single_distribution_core import (
    SingleDistributionMemberRoutingCore,
)
from honf_forward_core.interaction_reused_member_routing.faithful import (
    original_role_conditional_likelihood,
)


def _base_config() -> dict:
    return ContentHypergraphCoreV2(
        module_input_dim=3,
        environment_input_dim=2,
        coordinate_dim=2,
        query_input_dim=0,
        operating_input_dim=4,
        output_dim=4,
        architecture="A2-D",
        context_dim=8,
        num_heads=2,
        context_depth=2,
        local_hidden_dim=8,
        dropout=0.0,
        init_seed=301,
    ).get_config()


def _scene(module_count: int = 3, environment_count: int = 4) -> SceneBatch:
    module_x = torch.linspace(0.0, 1.0, module_count)
    environment_x = torch.linspace(0.0, 1.0, environment_count)
    return SceneBatch(
        module_features=torch.arange(module_count * 3, dtype=torch.float32).reshape(1, module_count, 3) / 10,
        environment_features=torch.arange(environment_count * 2, dtype=torch.float32).reshape(1, environment_count, 2) / 10,
        module_coordinates=torch.stack((module_x, torch.zeros_like(module_x)), dim=-1)[None],
        environment_coordinates=torch.stack((environment_x, torch.ones_like(environment_x) * 0.1), dim=-1)[None],
        module_mask=torch.ones((1, module_count), dtype=torch.bool),
        environment_mask=torch.ones((1, environment_count), dtype=torch.bool),
        module_measure=torch.tensor([[0.2 + i for i in range(module_count)]], dtype=torch.float32),
        environment_measure=torch.arange(1, environment_count + 1, dtype=torch.float32)[None],
        module_ids=torch.arange(30, 30 + module_count)[None],
        environment_ids=torch.arange(200, 200 + environment_count)[None],
        local_radius=torch.full((1, module_count), 0.4),
        operating_features=torch.ones((1, 4)),
        coordinate_origin=torch.zeros((1, 2)),
        coordinate_scale=torch.ones((1, 2)),
        module_measure_reference=torch.ones(1),
        environment_measure_reference=torch.ones(1),
        scene_key="r13-8-anchored-test-scene",
        normalizer_key="r13-8-anchored-test-normalizer",
        coordinate_frame_key="r13-8-anchored-test-frame",
        module_characteristic_lengths=torch.full((1, module_count, 2), 0.2),
        environment_characteristic_lengths=torch.full((1, environment_count, 2), 0.2),
    )


def test_compiler_uses_normalized_joint_row_mass_and_appends_structural_self() -> None:
    # The unnormalized receiver row totals 49.11; the actual prefix target is
    # 95% of the normalized row mass (one), not 95% of that raw total.
    em = torch.tensor([[[98.0, 0.1], [0.0, 0.0]]], requires_grad=True)
    ee = torch.tensor([[[2.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    module_measure = torch.tensor([[0.5, 0.5]])
    environment_measure = torch.tensor([[0.02, 0.98]])
    module_mask = torch.tensor([[True, True]])
    environment_mask = torch.tensor([[True, True]])
    module_ids = torch.tensor([[20, 10]])
    environment_ids = torch.tensor([[101, 100]])

    directory = compile_e_access_directory(
        em, ee, module_measure, environment_measure, module_mask, environment_mask,
        module_ids, environment_ids,
    )
    full = directory.full.row(0, 0)
    restricted = directory.restricted.row(0, 0)
    assert full.member_index.tolist() == [0, 2, 1]
    # The .95 prefix keeps the dominant M record; structural self is appended
    # even though it falls outside the learned prefix.
    assert restricted.member_index.tolist() == [0, 2]
    assert restricted.original_id.tolist() == [20, 101]
    self_position = restricted.member_index == 2
    assert restricted.applied_exchange[self_position].item() == 2.0
    assert restricted.structural_self[self_position].item() == 1.0
    torch.testing.assert_close(full.access_weight.sum(), torch.tensor(1.0))
    torch.testing.assert_close(restricted.access_weight.sum(), full.access_weight[:2].sum())

    weighted_identity = (restricted.access_weight * restricted.member_index.to(torch.float32)).sum()
    grad_em, grad_ee = torch.autograd.grad(weighted_identity, (em, ee))
    assert torch.isfinite(grad_em).all() and torch.isfinite(grad_ee).all()
    assert grad_em.abs().sum() > 0
    assert grad_ee.abs().sum() > 0


def test_compiler_gates_invalid_receivers_and_uses_role_then_id_ties() -> None:
    em = torch.tensor([[[1.0, 1.0], [5.0, 7.0]]])
    ee = torch.zeros((1, 2, 2))
    module_measure = torch.tensor([[0.5, 0.5]])
    environment_measure = torch.tensor([[0.25, 0.75]])
    module_mask = torch.tensor([[True, True]])
    environment_mask = torch.tensor([[True, False]])
    module_ids = torch.tensor([[20, 10]])
    environment_ids = torch.tensor([[7, 8]])
    directory = compile_e_access_directory(
        em, ee, module_measure, environment_measure, module_mask, environment_mask,
        module_ids, environment_ids,
    )

    valid = directory.full.row(0, 0)
    assert valid.member_index.tolist()[:2] == [1, 0]  # tied M mass sorts by original ID
    assert valid.original_id.tolist()[:2] == [10, 20]
    invalid = directory.full.row(0, 1)
    assert invalid.member_index.numel() == 0
    assert directory.restricted.row(0, 1).member_index.numel() == 0


def test_compiler_vectorized_packing_matches_dense_rows_and_live_gradients() -> None:
    dtype = torch.float64
    em = torch.tensor(
        [
            [[5.0, 5.0, 0.0], [0.0, 3.0, 3.0], [0.0, 0.0, 0.0], [2.0, 1.0, 7.0]],
            [[9.0, 0.0, 1.0], [2.0, 0.0, 2.0], [0.0, 0.0, 0.0], [1.0, 1.0, 3.0]],
        ], dtype=dtype, requires_grad=True,
    )
    ee = torch.tensor(
        [
            [[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]],
            [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
        ], dtype=dtype, requires_grad=True,
    )
    module_mask = torch.tensor([[True, True, False], [True, False, True]])
    environment_mask = torch.tensor([[True, True, True, False], [False, True, True, True]])
    module_measure = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]], dtype=dtype)
    environment_measure = torch.tensor(
        [[0.25, 0.5, 0.25, 0.0], [0.0, 0.5, 0.2, 0.3]], dtype=dtype,
    )
    module_ids = torch.tensor([[9, 2, 99], [88, 77, 1]])
    environment_ids = torch.tensor([[40, 30, 20, 10], [99, 9, 3, 1]])
    own = torch.tensor([[2, 0, 1, 0], [0, 3, 1, 2]])
    retained_mass = 0.75
    directory = compile_e_access_directory(
        em, ee, module_measure, environment_measure, module_mask, environment_mask,
        module_ids, environment_ids, own_e_member_index=own, retained_mass=retained_mass,
    )

    # Reconstruct dense access independently, then use simple host sorting to
    # form oracle rows. This checks multi-scene offsets, padded/empty rows,
    # non-identity self mapping, the prefix, and its self append.
    members = em.shape[-1] + ee.shape[-1]
    measure = torch.cat((module_measure, environment_measure), dim=-1)
    valid_member = torch.cat((module_mask, environment_mask), dim=-1)
    ids = torch.cat((module_ids, environment_ids), dim=-1)
    exchange = torch.cat((em, ee), dim=-1)
    exchange = exchange * valid_member[:, None, :].to(dtype)
    receiver_valid = environment_mask & (environment_measure > 0)
    exchange = exchange * receiver_valid[..., None].to(dtype)
    self_access = torch.zeros_like(exchange)
    for batch in range(em.shape[0]):
        for anchor in range(ee.shape[1]):
            if receiver_valid[batch, anchor]:
                self_access[batch, anchor, em.shape[-1] + int(own[batch, anchor])] = 1.0
    weighted = (exchange + self_access) * measure[:, None, :]
    row_mass = weighted.sum(-1, keepdim=True)
    access = torch.where(row_mass > 0, weighted / row_mass.clamp_min(torch.finfo(dtype).tiny),
                         torch.zeros_like(weighted))
    is_module = torch.arange(members) < em.shape[-1]

    expected_rows = {"full": [], "restricted": []}
    expected_offsets = {"full": [], "restricted": []}
    for batch in range(em.shape[0]):
        batch_offsets = {"full": [0], "restricted": [0]}
        for anchor in range(ee.shape[1]):
            positive = [i for i in range(members) if float(access[batch, anchor, i].detach()) > 0]
            ordered = sorted(
                positive,
                key=lambda i: (-float(access[batch, anchor, i].detach()),
                               0 if i < em.shape[-1] else 1, int(ids[batch, i])),
            )
            prefix = []
            if ordered:
                ordered_tensor = torch.tensor(ordered, dtype=torch.long)
                cumulative = access[batch, anchor, ordered_tensor].detach().cumsum(0)
                target = access[batch, anchor].detach().sum() * retained_mass
                count = int(torch.searchsorted(cumulative, target, right=False)) + 1
                prefix = ordered[:count]
            own_global = em.shape[-1] + int(own[batch, anchor])
            restricted = list(prefix)
            if receiver_valid[batch, anchor] and own_global not in restricted:
                restricted.append(own_global)
            for name, indices in (("full", ordered), ("restricted", restricted)):
                index = torch.tensor(indices, dtype=torch.long)
                row = {
                    "member_index": index,
                    "original_id": ids[batch, index],
                    "is_module": is_module[index],
                    "access_weight": access[batch, anchor, index],
                    "applied_exchange": exchange[batch, anchor, index],
                    "structural_self": self_access[batch, anchor, index],
                    "measure": measure[batch, index],
                }
                expected_rows[name].append(row)
                batch_offsets[name].append(batch_offsets[name][-1] + len(indices))
        for name in expected_offsets:
            previous_batches = sum(len(row["member_index"]) for row in expected_rows[name][:-ee.shape[1]])
            expected_offsets[name].append([value + previous_batches for value in batch_offsets[name]])

    actual_losses = []
    oracle_losses = []
    for name, packed in (("full", directory.full), ("restricted", directory.restricted)):
        torch.testing.assert_close(packed.offsets, torch.tensor(expected_offsets[name]), rtol=0, atol=0)
        for flat_row, expected_row in enumerate(expected_rows[name]):
            batch, anchor = divmod(flat_row, ee.shape[1])
            actual_row = packed.row(batch, anchor)
            for column, expected in expected_row.items():
                torch.testing.assert_close(getattr(actual_row, column), expected, rtol=0, atol=0, msg=column)
            coefficient = (1.0 + 0.17 * batch + 0.11 * anchor
                           + expected_row["member_index"].to(dtype) * 0.03
                           + expected_row["original_id"].to(dtype) * 0.001)
            actual_losses.append((actual_row.access_weight * coefficient).sum())
            oracle_losses.append((expected_row["access_weight"] * coefficient).sum())
    actual_gradient = torch.autograd.grad(sum(actual_losses), (em, ee), retain_graph=True)
    oracle_gradient = torch.autograd.grad(sum(oracle_losses), (em, ee))
    for actual, expected in zip(actual_gradient, oracle_gradient):
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-14)


def test_prepared_directory_uses_role_total_one_measure_with_unequal_counts() -> None:
    base = _base_config()
    model = AnchoredMemberRoutingCore(
        base, locator_mode="e_anchor", input_dataset="synthetic-r13-8", query_tile_size=8,
    ).eval()
    scene = _scene(module_count=2, environment_count=3)
    prepared = model.prepare(scene)
    directory = prepared.e_access_directory
    expected_mu = build_joint_member_measure(
        scene.module_mask, scene.environment_mask, scene.environment_measure,
        dtype=prepared.anchored_applied_EM.dtype,
    )
    torch.testing.assert_close(expected_mu[:, :2], torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(expected_mu[:, 2:], torch.tensor([[1 / 6, 2 / 6, 3 / 6]]))

    expected = torch.cat((prepared.anchored_applied_EM, prepared.anchored_applied_EE), dim=-1)
    # Build the exact E self ledger without conflating it with learned EE.
    self_ledger = torch.zeros_like(expected)
    self_ledger[0, torch.arange(3), 2 + torch.arange(3)] = 1.0
    weighted = (expected + self_ledger) * expected_mu[:, None, :]
    row_sum = weighted.sum(-1, keepdim=True)
    expected_access = torch.where(row_sum > 0, weighted / row_sum, torch.zeros_like(weighted))
    for anchor in range(3):
        row = directory.full.row(0, anchor)
        dense = expected_access[0, anchor]
        torch.testing.assert_close(row.access_weight, dense[row.member_index])
        torch.testing.assert_close(row.measure, expected_mu[0, row.member_index])
    assert prepared.work_counts["prepared_e_directory_sort_entries"] == 3 * (2 + 3)
    assert prepared.work_counts["prepared_e_directory_positive_exchange_entries"] == int(
        (directory.full.applied_exchange > 0).sum(),
    )

    assert prepared.anchored_route_provenance["em"]["stage_identity"].endswith("/em")
    assert prepared.anchored_route_provenance["ee"]["stage_identity"].endswith("/ee")
    assert not hasattr(prepared, "H")
    assert not any("proposal_M" in name for name, _ in model.named_parameters())

    invalid_scene = replace(scene, environment_mask=torch.tensor([[False, True, True]]))
    query = QueryBatch(
        coordinates=torch.tensor([[[0.2, 0.1]]]),
        query_mask=torch.ones((1, 1), dtype=torch.bool),
        anchor_indices=torch.tensor([[[0]]]),
        anchor_weights=torch.ones((1, 1, 1)),
        scene_key=scene.scene_key,
        coordinate_frame_key=scene.coordinate_frame_key,
    )
    with pytest.raises(ValueError, match="zero-measure E slot"):
        model._validate_anchor_carrier(invalid_scene, query)


def test_anchored_parameter_inventory_matches_u_except_removed_entrance() -> None:
    base = _base_config()
    u = SingleDistributionMemberRoutingCore(
        base, address_mode="ME", input_dataset="synthetic-r13-8", query_tile_size=8,
        retained_prior_law="none",
    )
    a = AnchoredMemberRoutingCore(
        base, locator_mode="e_anchor", input_dataset="synthetic-r13-8", query_tile_size=8,
    )
    u_state = u.state_dict()
    a_state = a.state_dict()
    assert "member_router.proposal_M.0.weight" in u_state
    assert not any(name.startswith("member_router.proposal_M.") for name in a_state)
    for name, value in a_state.items():
        torch.testing.assert_close(value, u_state[name], rtol=0.0, atol=0.0)
    config = a.get_config()
    assert AnchoredMemberRoutingCore.from_config(config).get_config() == config


def test_row_read_matches_historical_one_row_distribution_and_reuses_p_for_strength_and_values() -> None:
    torch.manual_seed(17)
    raw = torch.tensor([[2.0, -1.0, 0.25], [-0.5, 1.5, 0.1]], requires_grad=True)
    prior = torch.tensor([0.2, 0.3, 0.5], requires_grad=True)
    values = torch.tensor([[1.0, -2.0], [0.25, 1.5], [-1.0, 0.5]], requires_grad=True)
    query = torch.randn(2, 4, requires_grad=True)
    gate = nn.Linear(6, 2)
    query_mask = torch.tensor([True, True])
    candidate = torch.ones((3,), dtype=torch.bool)
    allowed = candidate[None, :].expand_as(raw)
    read = anchored_row_read(
        raw, prior, candidate, values, query, gate, 7, query_mask,
        score_calibration="inherited",
    )

    calibrated = prior_weighted_scores(raw, prior[None, :].expand_as(raw), allowed)
    expected_p = original_role_conditional_likelihood(
        prior[None, :].expand_as(raw), calibrated, allowed,
    )
    torch.testing.assert_close(read.probabilities, expected_p, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(read.expectation, (expected_p * calibrated).sum(-1))
    torch.testing.assert_close(read.context_before_strength, expected_p @ values)
    torch.testing.assert_close(read.context, (expected_p @ values) * read.strength[:, None])
    assert (read.probabilities > 0).all()
    torch.testing.assert_close(read.probabilities.sum(-1), torch.ones(2))
    loss = read.context.square().sum() + read.expectation.square().sum()
    gradients = torch.autograd.grad(loss, (raw, prior, values, query, *gate.parameters()))
    assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)

    # An empty typed role stays exactly empty and its strength is zero.
    empty = anchored_row_read(
        raw.new_empty((1, 0)), prior.new_empty((0,)), torch.empty((0,), dtype=torch.bool), None,
        query[:1], gate, 1, torch.tensor([True]), reduce_values=False,
    )
    assert empty.probabilities.shape == (1, 0)
    assert empty.strength.item() == 0.0


def test_completed_context_blending_does_not_renormalize_lambda_over_nonempty_rows() -> None:
    contexts = torch.tensor([[[[2.0], [0.0]]]], requires_grad=True)
    weights = torch.tensor([[[0.25, 0.75]]], requires_grad=True)
    blended = blend_anchored_contexts(contexts, weights)
    torch.testing.assert_close(blended, torch.tensor([[[0.5]]]))
    grad_context, grad_weights = torch.autograd.grad(blended.sum(), (contexts, weights))
    assert grad_context[0, 0, 0, 0] == 0.25
    assert grad_weights[0, 0, 0] == 2.0

    zeroed = blend_anchored_contexts(contexts, torch.tensor([[[0.0, 1.0]]]))
    torch.testing.assert_close(zeroed, torch.zeros_like(zeroed))
