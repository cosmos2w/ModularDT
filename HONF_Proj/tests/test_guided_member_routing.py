"""Focused R13-4 checks for early relation-guided original-member access."""
from dataclasses import replace

import pytest
import torch

from honf_forward_core.interaction_reused_member_routing import (
    GuidedMemberRoutingCore,
    RelativeMemberRoutingCore,
)
from honf_forward_core.interaction_reused_member_routing.guided import prior_weighted_scores
from honf_forward_core.interaction_reused_member_routing.relative import mass_prefix_mask
from tests.test_interaction_reused_member_routing import _base_config, _query, _scene


def _core(mode="guide", *, tolerance=0.01, tile_size=2):
    return GuidedMemberRoutingCore(
        _base_config(), member_routing_mode=mode, input_dataset="synthetic-native-contract",
        prior_mass_tolerance=tolerance, query_tile_size=tile_size,
    )


def _uniform_fixed_prior(core, prepared, query):
    pi = torch.full((query.batch_size, query.num_queries, prepared.scene.num_modules),
                    1. / max(prepared.scene.num_modules, 1), dtype=prepared.B_E.dtype)
    B_E = torch.zeros_like(prepared.B_E)
    valid = prepared.nu_E > 0
    B_E[valid[:, None, :].expand_as(B_E)] = prepared.nu_E[:, None, :].expand_as(B_E)[
        valid[:, None, :].expand_as(B_E)].reciprocal()
    return {"fixed_pi": pi, "fixed_B_E": B_E}


def test_v3_config_and_parameter_inventory_match_relative_checkpoint():
    relative = RelativeMemberRoutingCore(_base_config(), input_dataset="synthetic-native-contract")
    full = _core("full")
    guide = _core("guide")
    assert full.architecture == "R13-4-Full"
    assert guide.architecture == "R13-4-Guide"
    assert full.get_config()["schema"] == "interaction_reused_member_routing_v3"
    assert GuidedMemberRoutingCore.from_config(guide.get_config()).get_config() == guide.get_config()
    assert set(full.state_dict()) == set(relative.state_dict())
    for name, value in relative.state_dict().items():
        torch.testing.assert_close(full.state_dict()[name], value, rtol=0, atol=0)
    assert all(parameter.requires_grad for parameter in full.member_router.proposal_M.parameters())
    assert all(parameter.requires_grad for parameter in guide.member_router.proposal_M.parameters())


def test_relation_prior_prefix_selects_before_member_mlp_and_uses_original_id_ties():
    scene = replace(_scene(module_count=3, environment_count=6),
                    environment_measure=torch.tensor([[0., 1., 1., 1., 1., 1.]]))
    query = _query(2)
    core = _core(tolerance=0.41).eval()
    prepared = core.prepare(scene)
    controls = _uniform_fixed_prior(core, prepared, query)
    scored_rows = []
    handle = core.member_router.member_E.register_forward_pre_hook(
        lambda _module, args: scored_rows.append(int(args[0].shape[0])))
    try:
        with core.read_execution("restricted"):
            result = core.decode(prepared, query, intervention=controls, return_trace=True)
    finally:
        handle.remove()

    trace = result.trace
    assert trace is not None
    assert torch.equal(trace["full_eligible_mask_E"], trace["eligible_candidate_mask_E"])
    assert (trace["chosen_candidate_count_E"] < trace["eligible_candidate_count_E"]).all()
    # E id 100 has zero physical measure; the equal-mass prefix therefore
    # chooses the first three eligible original IDs, regardless of storage order.
    selected_ids = scene.environment_ids[:, None, :].expand_as(trace["chosen_candidate_mask_E"])[
        trace["chosen_candidate_mask_E"]].reshape(query.num_queries, -1)
    assert torch.equal(selected_ids, torch.tensor([[101, 102, 103], [101, 102, 103]]))
    torch.testing.assert_close(trace["prior_P_E"][..., 1:], torch.full_like(trace["prior_P_E"][..., 1:], 0.2))
    assert scored_rows == [int(trace["chosen_candidate_mask_E"].sum())]
    assert result.work_counts["query_E_cheap_score_rows"] == int(trace["chosen_candidate_mask_E"].sum())
    assert result.work_counts["query_E_prior_prefix_sort_rows"] == query.num_queries * scene.num_environment_tokens
    assert result.work_counts["query_M_candidate_rows"] == query.num_queries * scene.num_modules

    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    permuted_scene = replace(
        scene,
        environment_features=scene.environment_features[:, permutation],
        environment_coordinates=scene.environment_coordinates[:, permutation],
        environment_mask=scene.environment_mask[:, permutation],
        environment_measure=scene.environment_measure[:, permutation],
        environment_ids=scene.environment_ids[:, permutation],
        environment_characteristic_lengths=scene.environment_characteristic_lengths[:, permutation],
    )
    permuted_core = _core(tolerance=0.41).eval()
    permuted_prepared = permuted_core.prepare(permuted_scene)
    permuted_controls = _uniform_fixed_prior(permuted_core, permuted_prepared, query)
    with permuted_core.read_execution("restricted"):
        permuted = permuted_core.decode(permuted_prepared, query, intervention=permuted_controls,
                                        return_trace=True)
    permuted_ids = permuted_scene.environment_ids[:, None, :].expand_as(
        permuted.trace["chosen_candidate_mask_E"])[permuted.trace["chosen_candidate_mask_E"]].reshape(
            query.num_queries, -1)
    assert torch.equal(permuted_ids.sort(-1).values, selected_ids.sort(-1).values)


def test_weighted_calibration_fades_vanishing_prior_and_prefix_boundary_is_exact():
    scores = torch.tensor([[[0.2, 1.1, -2.0]]], dtype=torch.double)
    allowed = torch.ones_like(scores, dtype=torch.bool)
    epsilon = torch.tensor(1.e-9, dtype=torch.double, requires_grad=True)
    prior = torch.stack((0.5 - 0.5 * epsilon, 0.5 - 0.5 * epsilon, epsilon)).reshape(1, 1, 3)
    calibrated = prior_weighted_scores(scores, prior, allowed)
    baseline_mask = torch.tensor([[[True, True, False]]])
    baseline = prior_weighted_scores(scores, torch.tensor([[[0.5, 0.5, 0.0]]], dtype=torch.double), baseline_mask)
    torch.testing.assert_close(calibrated[..., :2], baseline[..., :2], atol=2.e-8, rtol=2.e-8)
    derivative = torch.autograd.grad(calibrated[..., :2].sum(), epsilon)[0]
    assert torch.isfinite(derivative) and derivative.abs() > 0

    tied = torch.tensor([[[0.4, 0.4, 0.2]]])
    ids = torch.tensor([[30, 10, 20]])
    selected = mass_prefix_mask(tied, torch.ones_like(tied, dtype=torch.bool), ids, 0.6)
    assert torch.equal(selected, torch.tensor([[[False, True, False]]]))
    assert torch.all((tied * ~selected).sum(-1) <= 0.6)


@pytest.mark.parametrize("module_count,environment_count", [(0, 4), (1, 0), (0, 0)])
def test_empty_and_no_m_address_scenes_return_finite_zero_context(module_count, environment_count):
    scene = _scene(module_count=module_count, environment_count=environment_count)
    core = _core(tolerance=0.4, tile_size=8).eval()
    prepared = core.prepare(scene)
    with core.read_execution("restricted"):
        result = core.decode(prepared, _query(3), return_trace=True)
    assert torch.isfinite(result.joint_fields).all()
    assert not result.trace["chosen_candidate_mask_E"].any()
    if module_count == 0:
        assert not result.trace["chosen_candidate_mask_M"].any()
    result.joint_fields.square().mean().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in core.parameters()
               if parameter.grad is not None)


def test_live_b_gradients_and_explicit_full_restricted_modes_ignore_train_eval_state():
    torch.manual_seed(812)
    core = _core(tolerance=0.55, tile_size=8)
    scene = _scene()
    query = _query(4)
    prepared = core.prepare(scene)

    core.eval()
    with core.read_execution("restricted"):
        restricted_eval = core.decode(prepared, query, return_trace=True)
    full_eval = core.decode(prepared, query, return_trace=True)
    assert (restricted_eval.trace["chosen_candidate_count_E"] <=
            restricted_eval.trace["eligible_candidate_count_E"]).all()
    assert torch.equal(full_eval.trace["chosen_candidate_mask_E"], full_eval.trace["full_eligible_mask_E"])
    assert torch.equal(restricted_eval.trace["scene_eligible_count_E"], full_eval.trace["scene_eligible_count_E"])

    targets = (prepared.B_E, prepared.member_keys_E, prepared.member_values_E,
               core.member_router.proposal_M[-1].weight)
    gradients = torch.autograd.grad(full_eval.joint_fields.square().mean(), targets, allow_unused=True)
    assert all(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
               for gradient in gradients)

    # Checkpointed training tiles replay their captured mode after both
    # context managers exit, and full/restricted losses can be backpropagated
    # together through the unchanged shared parameters.
    controls = _uniform_fixed_prior(core, prepared, query)
    core.zero_grad(set_to_none=True)
    core.eval()
    with core.read_execution("full"):
        full_reference = core.decode(prepared, query, intervention=controls)
    with core.read_execution("restricted"):
        restricted_reference = core.decode(prepared, query, intervention=controls)
    reference_loss = full_reference.joint_fields.square().mean() + restricted_reference.joint_fields.square().mean()
    probe_parameter = core.member_router.member_E[0].weight
    reference_gradient = torch.autograd.grad(reference_loss, probe_parameter, retain_graph=True)[0]

    core.train()
    with core.read_execution("full"):
        full_train = core.decode(prepared, query, intervention=controls)
    with core.read_execution("restricted"):
        restricted_train = core.decode(prepared, query, intervention=controls)
    assert full_train.work_counts["query_E_candidate_rows"] > restricted_train.work_counts["query_E_candidate_rows"]
    train_loss = full_train.joint_fields.square().mean() + restricted_train.joint_fields.square().mean()
    train_gradient = torch.autograd.grad(train_loss, probe_parameter, retain_graph=True)[0]
    assert torch.isfinite(train_gradient).all() and train_gradient.abs().max() > 0
    torch.testing.assert_close(train_gradient, reference_gradient, atol=2e-6, rtol=2e-5)
