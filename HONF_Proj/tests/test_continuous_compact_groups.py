"""Focused algebra and recovery tests for Level-R compact retrieval."""
from types import SimpleNamespace

import pytest
import torch

from honf_forward_core.continuous_joint.compact import CompactGroupRetrieval
from honf_forward_core.continuous_joint.primitives import entmax15, positive_quadratic_gate
from honf_forward_core.continuous_joint.unified import (
    CompactUnifiedJointCore,
    UnifiedJointCore,
)
from honf_forward_core.joint_native.contracts import TypedNativeRecords
from tests.test_thermal_content_hypergraph_v2 import _core_config


def _organization(*, module_order=None, modules=5, environments=4):
    torch.manual_seed(811)
    dtype = torch.float64
    module_coordinates = torch.tensor(
        [[0., 0.], [1., 0.], [1., 0.], [2., 0.], [3., 0.]], dtype=dtype
    )[:modules]
    environment_coordinates = torch.tensor(
        [[0., 1.], [1., 1.], [2., 1.], [3., 1.]], dtype=dtype
    )[:environments]
    module_ids = torch.tensor([99, 5, 4, 3, 2], dtype=torch.long)[:modules]
    if module_order is not None:
        module_order = torch.as_tensor(module_order, dtype=torch.long)
        module_coordinates = module_coordinates[module_order]
        module_ids = module_ids[module_order]
    module_coordinates = module_coordinates[None].clone().requires_grad_()
    environment_coordinates = environment_coordinates[None].clone().requires_grad_()
    module_mask = torch.ones((1, modules), dtype=torch.bool)
    environment_mask = torch.ones((1, environments), dtype=torch.bool)
    module_measure = torch.ones((1, modules), dtype=dtype)
    environment_measure = torch.ones((1, environments), dtype=dtype)
    if modules:
        module_measure[0, 0] = 0  # A physical M source remains eligible in C.
    if environments:
        environment_measure[0, 0] = 0  # E donor eligibility follows positive measure.
    module_features = torch.randn(1, modules, 2, dtype=dtype)
    environment_features = torch.randn(1, environments, 2, dtype=dtype)
    if module_order is not None:
        module_features = module_features[:, module_order]
    scene = SimpleNamespace(
        batch_size=1,
        num_modules=modules,
        num_environment_tokens=environments,
        module_features=module_features,
        environment_features=environment_features,
        module_coordinates=module_coordinates,
        environment_coordinates=environment_coordinates,
        module_characteristic_lengths=torch.ones_like(module_coordinates),
        environment_characteristic_lengths=torch.ones_like(environment_coordinates),
        module_mask=module_mask,
        environment_mask=environment_mask,
        module_measure=module_measure,
        environment_measure=environment_measure,
        module_ids=module_ids[None].clone(),
        environment_ids=(torch.arange(20, 20 + environments)[None]),
        operating_features=torch.tensor([[0.3]], dtype=dtype),
    )
    records = TypedNativeRecords.from_scene(scene)
    module_states = torch.randn(1, modules, 8, dtype=dtype, requires_grad=True)
    environment_states = torch.randn(1, environments, 8, dtype=dtype, requires_grad=True)
    organization = SimpleNamespace(
        scene=scene,
        joint_records=records,
        module_states=module_states,
        environment_states=environment_states,
        frame_origin=torch.zeros((1, 2), dtype=dtype),
        frame_scale=torch.ones((1, 2), dtype=dtype),
    )
    return organization


def _retrieval():
    model = CompactGroupRetrieval(
        width=8, heads=2, coordinate_dim=2, module_feature_dim=2,
        environment_feature_dim=2, operating_feature_dim=1,
    ).double()
    with torch.no_grad():
        for calibration in model.membership_calibration.values():
            calibration.bias.fill_(2.0)
    return model


def _candidate_key_pairs(topology):
    return {
        (kind, tuple(sorted((topology.group_keys[left][1], topology.group_keys[right][1]))))
        for left, right, kind in topology.candidate_pairs
    }


def _literal_membership(model, organization, compact):
    records = organization.joint_records
    donor_state = (organization.module_states, organization.environment_states)
    donor_coords = (records.module_coordinates, records.environment_coordinates)
    donor_lengths = (records.module_lengths, records.environment_lengths)
    donor_valid = (records.module_present, records.environment_present & (records.environment_measure > 0))
    rows = []
    for slot in range(compact.allocated_count):
        kind = int(compact.group_types[0, slot])
        type_rows = []
        for donor_type in (0, 1):
            route = {(0, 0): "mm", (0, 1): "me", (1, 0): "em", (1, 1): "ee"}[(kind, donor_type)]
            score = model.route_scorers[route](
                compact.controls[:, slot:slot + 1], compact.centroids[:, slot:slot + 1],
                compact.lengths[:, slot:slot + 1], donor_state[donor_type],
                donor_coords[donor_type], donor_lengths[donor_type], organization.frame_scale,
            )
            allowed = donor_valid[donor_type][:, None, :]
            calibrated = model.membership_calibration[route](score, allowed)
            type_rows.append(positive_quadratic_gate(calibrated) * allowed)
        rows.append(torch.cat(type_rows, -1))
    return torch.cat(rows, 1) if rows else compact.membership.new_zeros(compact.membership.shape)


def _literal_query_access(model, organization, compact, query, query_states, local, query_mask):
    """Reference the previous all-K typed scorer with a mask on each bank."""
    raw_m = model.query_scorers["M"](
        query, query_states, compact.controls, compact.centroids, compact.lengths,
        organization.frame_scale,
    )
    raw_e = model.query_scorers["E"](
        query, query_states, compact.controls, compact.centroids, compact.lengths,
        organization.frame_scale,
    )
    allowed_m = compact.group_mask & (compact.group_types == 0)
    allowed_e = compact.group_mask & (compact.group_types == 1)
    logits_m = model.query_calibration.module(
        raw_m, allowed_m[:, None, :].expand(-1, query_states.shape[1], -1),
    )
    logits_e = model.query_calibration.environment(
        raw_e, allowed_e[:, None, :].expand(-1, query_states.shape[1], -1),
    ) + model.query_calibration.environment_offset.to(raw_e)
    logits = torch.where((compact.group_types == 0)[:, None, :], logits_m, logits_e)
    allowed = compact.group_mask[:, None, :] & query_mask[:, :, None]
    weights = entmax15(logits / model.entmax_temperature.to(logits), allowed)
    access = torch.bmm(weights, compact.membership) * organization.joint_records.eligible[:, None]
    module_gate = local + (1 - local) * access[..., :organization.scene.num_modules]
    gates = torch.cat((module_gate, access[..., organization.scene.num_modules:]), -1)
    gates = gates * organization.joint_records.eligible[:, None] * query_mask[..., None]
    return gates, weights, logits


def test_sibling_tree_is_type_local_coordinate_ordered_and_tie_bundles_stay_singleton():
    model = _retrieval()
    organization = _organization()
    first = model.build(organization, alpha=0.0, cut_active=False)
    assert first.membership.shape == (1, 9, 9)
    assert first.pair_mask.sum().item() == 3
    pairs = _candidate_key_pairs(first.topology[0])
    assert pairs == {
        (0, ((2,), (3,))),
        (1, ((20,), (21,))),
        (1, ((22,), (23,))),
    }
    # The two records at (1, 0) are kept as separate leaves and neither is
    # selected by numeric ID as a representative.
    tied_ids = {4, 5}
    assert all(not (set(key_ids) & tied_ids) for _kind, key_ids in pairs)
    permuted = _organization(module_order=[4, 2, 0, 3, 1])
    second = model.build(permuted, alpha=0.0, cut_active=False)
    assert _candidate_key_pairs(second.topology[0]) == pairs


def test_direct_k_by_n_membership_matches_literal_reference_and_keeps_zero_measure_m():
    model = _retrieval()
    organization = _organization()
    with torch.no_grad():
        for policy in model.compatibility.values():
            policy.output.bias.fill_(2.0)
    compact = model.build(organization, alpha=0.0, cut_active=True)
    assert compact.membership.shape == (1, 6, 9)
    assert compact.live_counts.tolist() == [6]
    assert compact.topology[0].accepted_pairs
    assert len(compact.topology[0].archive_to_active) == 6
    assert len(compact.topology[0].retained_key_by_record) == 9
    torch.testing.assert_close(compact.membership, _literal_membership(model, organization, compact),
                               rtol=1e-10, atol=1e-12)
    # M measure zero is a physical source column, while zero-measure E is not
    # an information donor. No record axis was pooled into group controls.
    assert torch.any(compact.membership[..., 0] > 0)
    assert torch.count_nonzero(compact.membership[..., 5]) == 0

    weights = torch.linspace(-0.7, 0.9, compact.membership.numel(), dtype=torch.float64).reshape_as(
        compact.membership
    )
    variables = (organization.scene.module_coordinates, organization.module_states,
                 model.route_scorers["mm"].key.weight,
                 model.membership_calibration["mm"].bias)
    actual_grad = torch.autograd.grad((compact.membership * weights).sum(), variables, retain_graph=True)
    literal = _literal_membership(model, organization, compact)
    literal_grad = torch.autograd.grad((literal * weights).sum(), variables)
    for actual, expected in zip(actual_grad, literal_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
    assert torch.isfinite(actual_grad[0]).all() and actual_grad[0].count_nonzero() > 0
    assert torch.isfinite(actual_grad[2]).all() and actual_grad[2].count_nonzero() > 0


def test_query_access_uses_q_by_k_weights_and_structural_terms_differentiate():
    model = _retrieval()
    organization = _organization()
    soft = model.build(organization, alpha=0.5, cut_active=False)
    query = SimpleNamespace(coordinates=torch.tensor([[[0.2, 0.4], [2.2, 0.7]]], dtype=torch.float64))
    query_states = torch.randn(1, 2, 8, dtype=torch.float64, requires_grad=True)
    local = torch.zeros(1, 2, 5, dtype=torch.float64)
    local[0, 0, 1] = 0.3
    query_mask = torch.tensor([[True, True]])
    gates, weights, logits = model.query_access(soft, organization, query, query_states, local, query_mask)
    assert logits.shape == weights.shape == (1, 2, 9)
    assert gates.shape == (1, 2, 9)
    torch.testing.assert_close(weights.sum(-1), torch.ones((1, 2), dtype=torch.float64))
    access = torch.bmm(weights, soft.membership) * organization.joint_records.eligible[:, None]
    expected = torch.cat((local + (1 - local) * access[..., :5], access[..., 5:]), -1)
    expected = expected * organization.joint_records.eligible[:, None]
    torch.testing.assert_close(gates, expected, rtol=1e-10, atol=1e-12)

    components = model.structural_losses(soft, organization.joint_records)
    assert set(components) == {"tie", "capacity"}
    policy_bias = model.compatibility["M"].output.bias
    grad = torch.autograd.grad(components["capacity"].sum(), policy_bias)[0]
    assert torch.isfinite(grad) and grad.abs() > 0

    old = entmax15(torch.zeros((1, 1, 3), dtype=torch.float64))
    merged = entmax15(torch.zeros((1, 1, 2), dtype=torch.float64))
    # Identical sibling rows do not make entmax invariant to replacing two
    # logits by one merged logit: the old pair carries 2/3 mass, the new row 1/2.
    assert not torch.isclose(old[..., :2].sum(), merged[..., :1].sum())


def test_typed_q_to_k_scoring_matches_literal_all_k_reference_and_gradients():
    first, second = _organization(), _organization()
    scene = SimpleNamespace(
        batch_size=2,
        num_modules=5,
        num_environment_tokens=4,
        module_features=torch.cat((first.scene.module_features, second.scene.module_features)),
        environment_features=torch.cat((first.scene.environment_features, second.scene.environment_features)),
        module_coordinates=torch.cat((first.scene.module_coordinates, second.scene.module_coordinates)),
        environment_coordinates=torch.cat((first.scene.environment_coordinates, second.scene.environment_coordinates)),
        module_characteristic_lengths=torch.ones((2, 5, 2), dtype=torch.float64),
        environment_characteristic_lengths=torch.ones((2, 4, 2), dtype=torch.float64),
        module_mask=torch.ones((2, 5), dtype=torch.bool),
        environment_mask=torch.ones((2, 4), dtype=torch.bool),
        module_measure=torch.ones((2, 5), dtype=torch.float64),
        environment_measure=torch.ones((2, 4), dtype=torch.float64),
        module_ids=torch.cat((first.scene.module_ids, second.scene.module_ids)),
        environment_ids=torch.cat((first.scene.environment_ids, second.scene.environment_ids)),
        operating_features=torch.tensor([[0.3], [0.3]], dtype=torch.float64),
    )
    # The two scenes have different bank sizes: max K is 6, while separate
    # bank padding allocates 5 M slots plus 4 E slots for one Q-by-K pass.
    scene.module_mask[1, 1:] = False
    scene.environment_mask[0, 1:] = False
    scene.module_measure[:, 0] = 0
    scene.environment_measure[:, 0] = 0
    organization = SimpleNamespace(
        scene=scene,
        joint_records=TypedNativeRecords.from_scene(scene),
        module_states=torch.cat((first.module_states, second.module_states)),
        environment_states=torch.cat((first.environment_states, second.environment_states)),
        frame_origin=torch.zeros((2, 2), dtype=torch.float64),
        frame_scale=torch.ones((2, 2), dtype=torch.float64),
    )
    model = _retrieval()
    compact = model.build(organization, alpha=0.0, cut_active=False)
    assert compact.live_counts.tolist() == [6, 5]
    assert compact.score_slot_count == 9

    query_states = torch.randn((2, 3, 8), dtype=torch.float64, requires_grad=True)
    query_coordinates = torch.randn((2, 3, 2), dtype=torch.float64, requires_grad=True)
    query = SimpleNamespace(coordinates=query_coordinates)
    local = torch.rand((2, 3, 5), dtype=torch.float64)
    query_mask = torch.tensor([[True, False, True], [True, True, True]])
    actual = model.query_access(compact, organization, query, query_states, local, query_mask)
    expected = _literal_query_access(model, organization, compact, query, query_states, local, query_mask)
    for actual_value, expected_value in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-10, atol=1e-12)
    valid = compact.group_mask[:, None, :] & query_mask[:, :, None]
    torch.testing.assert_close(
        actual[2].masked_fill(~valid, 0), expected[2].masked_fill(~valid, 0), rtol=1e-10, atol=1e-12,
    )

    torch.manual_seed(902)
    gate_probe = torch.randn_like(actual[0])
    weight_probe = torch.randn_like(actual[1])
    logit_probe = torch.randn_like(actual[2])
    actual_loss = ((actual[0] * gate_probe).sum() + (actual[1] * weight_probe).sum()
                   + (actual[2].masked_fill(~valid, 0) * logit_probe).sum())
    expected_loss = ((expected[0] * gate_probe).sum() + (expected[1] * weight_probe).sum()
                     + (expected[2].masked_fill(~valid, 0) * logit_probe).sum())
    variables = [query_states, query_coordinates, compact.controls, compact.centroids]
    variables += list(model.query_scorers.parameters())
    variables += list(model.query_calibration.parameters())
    actual_grad = torch.autograd.grad(actual_loss, variables, retain_graph=True, allow_unused=True)
    expected_grad = torch.autograd.grad(expected_loss, variables, allow_unused=True)
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        assert (actual_value is None) == (expected_value is None)
        if actual_value is not None:
            torch.testing.assert_close(actual_value, expected_value, rtol=1e-9, atol=1e-11)


def test_identical_membership_rows_preserve_summed_mass_and_parameter_derivatives():
    base = torch.tensor([[[0.15, 0.25, 0.35, 0.25]]], dtype=torch.float64, requires_grad=True)
    membership = torch.cat((base, base), 1)
    weights = torch.tensor([[[0.3, 0.7], [0.8, 0.2]]], dtype=torch.float64, requires_grad=True)
    old_access = torch.bmm(weights, membership)
    merged_access = torch.bmm(weights.sum(-1, keepdim=True), base)
    torch.testing.assert_close(old_access, merged_access, rtol=0, atol=1e-15)
    probe = torch.tensor([[[0.4, -0.2, 0.8, -0.3], [-0.6, 0.5, 0.2, 0.7]]], dtype=torch.float64)
    variables = (base, weights)
    old_grad = torch.autograd.grad((old_access * probe).sum(), variables, retain_graph=True)
    merged_grad = torch.autograd.grad((merged_access * probe).sum(), variables)
    for actual, expected in zip(old_grad, merged_grad):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)


def test_batch_padding_tracks_live_k_and_masks_unused_allocated_group():
    first, second = _organization(), _organization()
    second.scene.module_coordinates = torch.tensor(
        [[[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]]],
        dtype=torch.float64,
    ).requires_grad_()
    scene = SimpleNamespace(
        batch_size=2,
        num_modules=5,
        num_environment_tokens=4,
        module_features=torch.cat((first.scene.module_features, second.scene.module_features)),
        environment_features=torch.cat((first.scene.environment_features, second.scene.environment_features)),
        module_coordinates=torch.cat((first.scene.module_coordinates, second.scene.module_coordinates)),
        environment_coordinates=torch.cat((first.scene.environment_coordinates, second.scene.environment_coordinates)),
        module_characteristic_lengths=torch.ones((2, 5, 2), dtype=torch.float64),
        environment_characteristic_lengths=torch.ones((2, 4, 2), dtype=torch.float64),
        module_mask=torch.ones((2, 5), dtype=torch.bool),
        environment_mask=torch.ones((2, 4), dtype=torch.bool),
        module_measure=torch.ones((2, 5), dtype=torch.float64),
        environment_measure=torch.ones((2, 4), dtype=torch.float64),
        module_ids=torch.cat((first.scene.module_ids, second.scene.module_ids)),
        environment_ids=torch.cat((first.scene.environment_ids, second.scene.environment_ids)),
        operating_features=torch.tensor([[0.3], [0.3]], dtype=torch.float64),
    )
    scene.module_measure[:, 0] = 0
    scene.environment_measure[:, 0] = 0
    organization = SimpleNamespace(
        scene=scene,
        joint_records=TypedNativeRecords.from_scene(scene),
        module_states=torch.cat((first.module_states, second.module_states)),
        environment_states=torch.cat((first.environment_states, second.environment_states)),
        frame_origin=torch.zeros((2, 2), dtype=torch.float64),
        frame_scale=torch.ones((2, 2), dtype=torch.float64),
    )
    model = _retrieval()
    with torch.no_grad():
        for policy in model.compatibility.values():
            policy.output.bias.fill_(2.0)
    compact = model.build(organization, alpha=0.0, cut_active=True)
    assert compact.membership.shape == (2, 6, 9)
    assert compact.live_counts.tolist() == [6, 5]
    assert compact.allocated_count == 6
    assert not compact.group_mask[1, 5]
    assert compact.group_types[1, 5] == -1
    assert compact.membership[1, 5].count_nonzero() == 0


@pytest.mark.parametrize("modules,environments,expected_groups", [(0, 3, 2), (3, 0, 3), (0, 0, 0)])
def test_empty_typed_banks_and_empty_scene_keep_valid_compact_axes(modules, environments, expected_groups):
    model = _retrieval()
    organization = _organization(modules=modules, environments=environments)
    with torch.no_grad():
        for policy in model.compatibility.values():
            policy.output.bias.fill_(2.0)
    compact = model.build(organization, alpha=0.0, cut_active=True)
    assert compact.membership.shape == (1, expected_groups, modules + environments)
    assert compact.group_mask.sum().item() == expected_groups
    losses = model.structural_losses(compact, organization.joint_records)
    assert all(torch.isfinite(value).all() for value in losses.values())


def test_hard_cut_state_roundtrips_optimizer_exclusions_and_parent_clocks():
    parent = UnifiedJointCore(
        _core_config("A2-D"), input_dataset="thermal", routing_mode="hypergraph",
        native_backend="packed", checkpoint_pipeline=False,
    )
    parent.set_training_clock(4000, 13000)
    parent.joint_begin_update(4000, 12)
    parent.joint_macro_clock.fill_(13000)
    parent._joint_macro_clock = 13000
    compact = CompactUnifiedJointCore.from_unified_parent(parent)
    assert int(compact.joint_epoch) == 4000
    assert int(compact.joint_update_index) == 12
    assert int(compact.joint_macro_clock) == 13000
    assert compact.compact_parameter_migration["copied_names"]
    assert compact.compact_parameter_migration["new_names"]
    assert compact.compact_parameter_migration["dropped_names"]
    assert compact.compact_frozen_parameter_names
    assert all(not parameter.requires_grad for name, parameter in compact.named_parameters()
               if name.startswith(("router.dispatch_parameters.", "router.direct_parameters.")))

    compact.set_compact_loss_coefficients(0.03, 0.02)
    compact.set_compact_phase(39, arm="merge")
    with pytest.raises(ValueError, match="after the merge arm's local epoch 40 update"):
        compact.activate_hard_cut()
    compact.set_compact_phase(40, arm="merge")
    assert compact.compact_alpha == 1.0
    compact.activate_hard_cut()
    assert compact.compact_alpha == 0.0
    with pytest.raises(ValueError, match="cannot move backward"):
        compact.set_compact_phase(39, arm="merge")
    state = compact.export_compact_state()
    restored = CompactUnifiedJointCore.from_config(compact.get_config())
    restored.load_state_dict(compact.state_dict(), strict=True)
    restored.load_compact_state(state)
    assert restored.export_compact_state() == state
    assert int(restored.joint_epoch) == 4000
    assert int(restored.joint_update_index) == 12
    assert int(restored.joint_macro_clock) == 13000
    assert all(not parameter.requires_grad for parameter in restored.compact_retrieval.compatibility.parameters())
    assert all(not parameter.requires_grad for name, parameter in restored.named_parameters()
               if name.startswith(("router.dispatch_parameters.", "router.direct_parameters.")))


def test_compact_recovery_rejects_inconsistent_cut_and_unprocessed_epochs():
    parent = UnifiedJointCore(
        _core_config("A2-D"), input_dataset="thermal", routing_mode="hypergraph",
        native_backend="packed", checkpoint_pipeline=False,
    )
    compact = CompactUnifiedJointCore.from_unified_parent(parent)
    compact.set_compact_phase(40, arm="merge")
    pre_cut = compact.export_compact_state()
    restored_pre_cut = CompactUnifiedJointCore.from_config(compact.get_config())
    restored_pre_cut.load_compact_state(pre_cut)
    assert restored_pre_cut.compact_local_epoch.item() == 40
    assert not restored_pre_cut.compact_cut_active

    mismatch = dict(pre_cut, cut_active=True, policy_frozen=False)
    with pytest.raises(ValueError, match="cut_active and policy_frozen fields must agree"):
        restored_pre_cut.load_compact_state(mismatch)

    missed_cut = dict(pre_cut, local_epoch=41)
    with pytest.raises(ValueError, match="uncut merge checkpoint cannot resume"):
        restored_pre_cut.load_compact_state(missed_cut)

    compact.activate_hard_cut()
    invalid_state_dict = compact.state_dict()
    invalid_state_dict["compact_policy_frozen"] = torch.tensor(False)
    restored = CompactUnifiedJointCore.from_config(compact.get_config())
    with pytest.raises(RuntimeError, match="compact_cut_active and compact_policy_frozen state must agree"):
        restored.load_state_dict(invalid_state_dict, strict=True)


@pytest.mark.parametrize(
    "metadata_name",
    ("group_mask", "group_types", "extents", "pair_slots", "bank_group_indices", "bank_group_masks"),
)
def test_prepared_guard_covers_compact_routing_metadata(metadata_name):
    from tests.test_thermal_compact_adapter import _compact_adapter

    model, structure, fluid, local = _compact_adapter(2)
    model.eval()
    prepared = model.prepare_native(
        structure, fluid, local_query_points=local, ntheta=16, scene_key=("compact-metadata-stamp",),
    )
    compact = prepared.organization.compact_groups
    if metadata_name == "group_mask":
        compact.group_mask[0, 0].logical_not_()
    elif metadata_name == "group_types":
        compact.group_types.add_(0)
    elif metadata_name == "extents":
        compact.extents.add_(0)
    elif metadata_name == "pair_slots":
        compact.pair_slots.add_(0)
    elif metadata_name == "bank_group_indices":
        compact.bank_group_indices[0].add_(0)
    else:
        compact.bank_group_masks[0][0, 0].logical_not_()
    with pytest.raises(ValueError, match="Prepared compact membership or controls changed"):
        model.core.validate_prepared_state(prepared.organization, prepared.query_batch)
