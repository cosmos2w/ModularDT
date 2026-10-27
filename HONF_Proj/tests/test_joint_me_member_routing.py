from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import honf_forward_core.interaction_reused_member_routing.joint as joint_module
from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2, QueryBatch, SceneBatch
from honf_forward_core.interaction_reused_member_routing import (
    GuidedMemberRoutingCore,
    JointMemberRoutingCore,
)
from honf_forward_core.interaction_reused_member_routing.joint import (
    assemble_joint_incidence,
    build_joint_member_measure,
    build_joint_retrieval,
    joint_member_prefix_mask,
)


def _base_config(*, output_dim: int = 4, context_depth: int = 2) -> dict:
    return ContentHypergraphCoreV2(
        module_input_dim=9,
        environment_input_dim=6,
        coordinate_dim=2,
        query_input_dim=0,
        operating_input_dim=12,
        output_dim=output_dim,
        architecture="A2-D",
        context_dim=16,
        num_heads=4,
        context_depth=context_depth,
        local_hidden_dim=16,
        dropout=0.0,
        init_seed=731,
    ).get_config()


def _scene(module_count: int = 3, environment_count: int = 6) -> SceneBatch:
    module_x = torch.linspace(0.15, 1.65, module_count)
    module_coordinates = torch.stack((module_x, torch.linspace(-0.1, 0.2, module_count)), dim=-1)
    environment_x = torch.linspace(0.0, 1.8, environment_count)
    environment_coordinates = torch.stack((environment_x, torch.zeros_like(environment_x)), dim=-1)
    return SceneBatch(
        module_features=torch.randn(1, module_count, 9),
        environment_features=torch.randn(1, environment_count, 6),
        module_coordinates=module_coordinates[None],
        environment_coordinates=environment_coordinates[None],
        module_mask=torch.ones((1, module_count), dtype=torch.bool),
        environment_mask=torch.ones((1, environment_count), dtype=torch.bool),
        module_measure=torch.linspace(0.0, 1.0, module_count)[None],
        environment_measure=torch.linspace(0.0, 1.0, environment_count)[None],
        module_ids=torch.arange(20, 20 + module_count)[None],
        environment_ids=torch.arange(100, 100 + environment_count)[None],
        local_radius=torch.full((1, module_count), 0.2),
        operating_features=torch.randn(1, 12),
        coordinate_origin=torch.tensor([[0.0, -0.5]]),
        coordinate_scale=torch.tensor([[2.0, 1.0]]),
        module_measure_reference=torch.ones(1),
        environment_measure_reference=torch.ones(1),
        scene_key="round13-5-test-scene",
        normalizer_key="round13-5-test-normalizer",
        coordinate_frame_key="round13-5-test-frame",
        module_characteristic_lengths=torch.full((1, module_count, 2), 0.12),
        environment_characteristic_lengths=torch.full((1, environment_count, 2), 0.1),
    )


def _query(count: int = 5) -> QueryBatch:
    x = torch.linspace(0.05, 1.75, count)
    coordinates = torch.stack((x, torch.linspace(-0.15, 0.25, count)), dim=-1)
    return QueryBatch(
        coordinates=coordinates[None],
        query_mask=torch.ones(1, count, dtype=torch.bool),
        query_ids=torch.arange(500, 500 + count)[None],
        scene_key="round13-5-test-scene",
        coordinate_frame_key="round13-5-test-frame",
    )


def _core(mode: str = "ME", *, tile_size: int = 4) -> JointMemberRoutingCore:
    return JointMemberRoutingCore(
        _base_config(),
        address_mode=mode,
        input_dataset="synthetic-native-contract",
        query_tile_size=tile_size,
    )


def test_joint_measure_and_whole_row_retrieval_apply_mu_once() -> None:
    module_mask = torch.tensor([[True, True, False]])
    environment_mask = torch.tensor([[True, True, False]])
    environment_measure = torch.tensor([[2.0, 6.0, 0.0]])
    mu = build_joint_member_measure(module_mask, environment_mask, environment_measure)
    torch.testing.assert_close(mu, torch.tensor([[0.5, 0.5, 0.0, 0.25, 0.75, 0.0]]))

    H = torch.tensor([[[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 2.0]]])
    R, row_mass, raw = build_joint_retrieval(H, mu)
    torch.testing.assert_close(raw, H * mu[:, None, :])
    torch.testing.assert_close(row_mass, torch.tensor([[0.75, 0.0]]))
    torch.testing.assert_close(R[0, 0], torch.tensor([2 / 3, 0.0, 0.0, 1 / 3, 0.0, 0.0]))
    assert R[0, 1].count_nonzero() == 0


def test_final_cycle_blocks_keep_direction_provenance_and_no_extra_e_self() -> None:
    torch.manual_seed(18)
    core = _core()
    core.eval()
    scene = _scene(3, 6)
    prepared = core.prepare(scene)
    m, e = scene.num_modules, scene.num_environment_tokens
    stages = prepared.final_applied_context_stages
    assert [stages[name]["snapshot"].stage_identity for name in ("mm", "em", "ee", "me")] == [
        "cycle1/mm", "cycle1/em", "cycle1/ee", "cycle1/me"
    ]
    torch.testing.assert_close(prepared.H_exchange[:, :m, :m], prepared.H_MM_applied)
    torch.testing.assert_close(prepared.H_exchange[:, :m, m:], prepared.H_ME_applied)
    torch.testing.assert_close(prepared.H_exchange[:, m:, :m], prepared.H_EM_applied)
    torch.testing.assert_close(prepared.H_exchange[:, m:, m:], prepared.H_EE_applied)
    torch.testing.assert_close(prepared.H_self[:, :m, :m], torch.eye(m)[None])
    assert prepared.H_self[:, m:, :].count_nonzero() == 0
    torch.testing.assert_close(prepared.H[:, m:, m:], prepared.H_EE_applied)
    torch.testing.assert_close(prepared.retrieval_raw_coefficients,
                               prepared.H * prepared.member_measure_joint[:, None, :])
    row_sums = prepared.retrieval_R.sum(-1)
    torch.testing.assert_close(row_sums[prepared.joint_address_eligible],
                               torch.ones_like(row_sums[prepared.joint_address_eligible]))
    assert set(prepared.H_route_provenance) == {"MM", "EM", "EE", "ME"}
    assert prepared.H_route_provenance["EM"]["receiver_role"] == "E"
    assert prepared.H_route_provenance["EM"]["donor_role"] == "M"
    assert prepared.H_route_provenance["ME"]["receiver_role"] == "M"
    assert prepared.H_route_provenance["ME"]["donor_role"] == "E"

    trace = core.capture_interaction_routing_trace(prepared, _query(3))
    for name in ("H_MM_applied", "H_EM_applied", "H_EE_applied", "H_ME_applied", "H_self_M"):
        assert name in trace
    assert trace["candidate_ids_M"].shape[-1] == trace["candidate_ids_E"].shape[-1] == 4
    assert trace["joint_candidate_mask"].shape == (1, 3, m + e)
    assert trace["conditional_preference_M"].shape == (1, 3, m)
    assert trace["conditional_preference_E"].shape == (1, 3, e)
    for block in ("MM", "EM", "EE", "ME", "self_M"):
        assert trace[f"address_block_mass_{block}"].shape == (1, 3)


def test_four_block_incidence_helper_preserves_ee_diagonal_and_m_only_identity() -> None:
    mask = torch.tensor([[True, False]])
    mm = torch.tensor([[[0.2, 0.0], [0.0, 0.0]]])
    em = torch.tensor([[[0.3, 0.0], [0.4, 0.0]]])
    ee = torch.tensor([[[0.7, 0.2], [0.1, 0.8]]])
    me = torch.tensor([[[0.6, 0.1], [0.0, 0.0]]])
    exchange, self_ledger, H, blocks = assemble_joint_incidence(mm, em, ee, me, mask)
    torch.testing.assert_close(H, exchange + self_ledger)
    torch.testing.assert_close(self_ledger[:, :2, :2], torch.diag_embed(mask.float()))
    assert self_ledger[:, 2:, :].count_nonzero() == 0
    torch.testing.assert_close(H[:, 2:, 2:].diagonal(dim1=-2, dim2=-1), ee.diagonal(dim1=-2, dim2=-1))
    assert blocks["EE"] is ee


def test_joint95_ties_use_role_then_original_id_and_survive_storage_permutation() -> None:
    weights = torch.full((1, 1, 4), 0.25)
    allowed = torch.ones_like(weights, dtype=torch.bool)
    module_ids = torch.tensor([[7, 2]])
    environment_ids = torch.tensor([[1, 0]])
    selected = joint_member_prefix_mask(weights, allowed, module_ids, environment_ids, 0.5)
    assert set(module_ids[0, selected[0, 0, :2]].tolist()) == {2, 7}
    assert not bool(selected[0, 0, 2:].any())

    permuted = joint_member_prefix_mask(
        weights,
        allowed[..., [1, 0, 3, 2]],
        module_ids[:, [1, 0]],
        environment_ids[:, [1, 0]],
        0.5,
    )
    selected_ids = {
        ("M", int(module_ids[0, slot]))
        for slot in range(2) if bool(permuted[0, 0, 1 - slot])
    } | {
        ("E", int(environment_ids[0, slot]))
        for slot in range(2) if bool(permuted[0, 0, 2 + 1 - slot])
    }
    assert selected_ids == {("M", 2), ("M", 7)}


def test_config_and_state_dict_transfer_match_guide_parameter_inventory() -> None:
    guide = GuidedMemberRoutingCore(
        _base_config(), member_routing_mode="guide", input_dataset="synthetic-native-contract",
        prior_mass_tolerance=0.05, query_tile_size=4,
    )
    joint = _core("ME")
    assert set(guide.state_dict()) == set(joint.state_dict())
    for name, value in guide.state_dict().items():
        assert value.shape == joint.state_dict()[name].shape, name
    result = joint.load_state_dict(guide.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    for name, value in guide.state_dict().items():
        torch.testing.assert_close(value, joint.state_dict()[name], rtol=0, atol=0)

    config = joint.get_config()
    assert config["schema"] == "interaction_reused_joint_member_routing_v1"
    assert config["architecture"] == "R13-5-ME-Joint95"
    assert JointMemberRoutingCore.from_config(config).get_config() == config
    alternate = dict(config, architecture="R13-5-M-Joint95", address_mode="M")
    assert JointMemberRoutingCore.from_config(alternate).architecture == "R13-5-M-Joint95"


@pytest.mark.parametrize("modules,environments,expected_role", [(0, 4, "E"), (3, 0, "M")])
def test_missing_opposite_role_executes_a_pure_typed_read(modules, environments, expected_role) -> None:
    torch.manual_seed(12)
    core = _core("ME")
    core.eval()
    scene = _scene(modules, environments)
    prepared = core.prepare(scene)
    output = core.decode(prepared, _query(2), return_trace=True)
    trace = output.trace
    assert trace is not None
    candidate_m = trace["candidate_mask_M"]
    candidate_e = trace["candidate_mask_E"]
    if expected_role == "E":
        assert not bool(candidate_m.any()) and bool(candidate_e.any())
        assert trace["context_M"].count_nonzero() == 0
    else:
        assert bool(candidate_m.any()) and not bool(candidate_e.any())
        assert trace["context_E"].count_nonzero() == 0
    assert bool(torch.isfinite(output.joint_fields).all())


@pytest.mark.parametrize(
    "favored_role,suppressed_cross_route",
    [("E", "em"), ("M", "me")],
)
def test_controlled_synthetic_joint95_can_be_pure_typed_with_both_roles_present(
    favored_role: str, suppressed_cross_route: str,
) -> None:
    """A capability control, not a claim about native learned scene locality."""
    torch.manual_seed(19)
    core = _core("ME")
    core.eval()
    scene = _scene(3, 6)
    # Remove only the directed cross-role path that would otherwise leak the
    # favored address role into the opposite member inventory.
    with torch.no_grad():
        core.applied_incidence.routes[suppressed_cross_route].bias.fill_(-100.0)
    prepared = core.prepare(scene)
    query = _query(2)
    modules, environments = scene.num_modules, scene.num_environment_tokens
    eligible_roles = torch.cat((scene.module_mask,
                                scene.environment_mask & (scene.environment_measure > 0)), dim=-1)
    eligible_addresses = prepared.joint_address_eligible & eligible_roles
    eligible_M = eligible_addresses[..., :modules]
    eligible_E = eligible_addresses[..., modules:]
    assert bool(eligible_M.any()) and bool(eligible_E.any())
    pi = torch.zeros(1, query.num_queries, modules + environments)
    if favored_role == "E":
        pi[..., :modules] = eligible_M.to(pi.dtype) * (1.0e-4 / int(eligible_M.sum()))
        pi[..., modules:] = eligible_E.to(pi.dtype) * ((1.0 - 1.0e-4) / int(eligible_E.sum()))
        opposite = "M"
    else:
        pi[..., :modules] = eligible_M.to(pi.dtype) * ((1.0 - 1.0e-4) / int(eligible_M.sum()))
        pi[..., modules:] = eligible_E.to(pi.dtype) * (1.0e-4 / int(eligible_E.sum()))
        opposite = "E"
    assert bool((pi[eligible_addresses[:, None].expand_as(pi)] > 0).all())
    assert not bool((pi[~eligible_addresses[:, None].expand_as(pi)] > 0).any())

    scored_rows: dict[str, int] = {"M": 0, "E": 0}
    handles = []
    for role in ("M", "E"):
        def observe(_module, inputs, *, _role=role):
            scored_rows[_role] += int(inputs[0].shape[0])
        handles.append(getattr(core.member_router, "member_" + role).register_forward_pre_hook(observe))
    try:
        with core.read_execution("restricted"):
            output = core.decode(
                prepared, query, intervention={"fixed_pi": pi}, return_trace=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    trace = output.trace
    assert trace is not None
    candidate_M = trace["candidate_mask_M"]
    candidate_E = trace["candidate_mask_E"]
    if favored_role == "E":
        assert not bool(candidate_M.any()) and bool(candidate_E.any())
        assert float(trace["prior_role_mass_E"].min()) > 0.95
        assert trace["context_M"].count_nonzero() == 0
    else:
        assert bool(candidate_M.any()) and not bool(candidate_E.any())
        assert float(trace["prior_role_mass_M"].min()) > 0.95
        assert trace["context_E"].count_nonzero() == 0
    assert scored_rows[opposite] == 0
    assert bool(torch.isfinite(output.joint_fields).all())


def test_equal_shared_address_score_balances_roles_and_whole_row_retrieval_preserves_them() -> None:
    """Catch M/E count imbalance and the per-address raw-identity distortion."""
    scene = replace(_scene(1, 192), environment_measure=torch.ones(1, 192))
    modules, environments = scene.num_modules, scene.num_environment_tokens
    environment_allowed = scene.environment_mask & (scene.environment_measure > 0)
    address_allowed = torch.cat((scene.module_mask, environment_allowed), dim=-1)[:, None, :]
    omega, allowed = JointMemberRoutingCore._address_base_measure(scene, address_allowed, "ME")
    equal_shared_scores = torch.zeros_like(omega)
    proposal = joint_module._masked_softmax(
        equal_shared_scores + torch.where(
            allowed,
            omega.clamp_min(torch.finfo(omega.dtype).tiny).log(),
            torch.zeros_like(omega),
        ),
        allowed,
    )
    torch.testing.assert_close(proposal[..., :modules].sum(-1), torch.full((1, 1), 0.5))
    torch.testing.assert_close(proposal[..., modules:].sum(-1), torch.full((1, 1), 0.5))

    measure = build_joint_member_measure(
        scene.module_mask, scene.environment_mask, scene.environment_measure,
    )
    identity_H = torch.eye(modules + environments)[None]
    dense_H = torch.ones_like(identity_H)
    for H in (identity_H, dense_H):
        retrieval, _row_mass, _raw = build_joint_retrieval(H, measure)
        prior = torch.bmm(proposal, retrieval)
        torch.testing.assert_close(prior[..., :modules].sum(-1), torch.full((1, 1), 0.5))
        torch.testing.assert_close(prior[..., modules:].sum(-1), torch.full((1, 1), 0.5))

    # Deliberately wrong comparator: globally normalizing raw Hμ per address
    # applies μ after proposal and makes an identity H depend on inventory size.
    wrong_raw = torch.bmm(proposal, identity_H * measure[:, None, :])
    wrong_global = wrong_raw / wrong_raw.sum(-1, keepdim=True)
    expected_raw_M = 192.0 / 193.0
    torch.testing.assert_close(
        wrong_global[..., :modules].sum(-1),
        torch.full((1, 1), expected_raw_M),
        atol=1.0e-6,
        rtol=0,
    )


def test_m1_uses_the_shared_proposal_scorer_for_query_dependent_e_addresses() -> None:
    torch.manual_seed(26)
    core = _core("ME")
    core.eval()
    scene = _scene(1, 8)
    prepared = core.prepare(scene)
    trace = core.capture_interaction_routing_trace(prepared, _query(4))
    e_scores = trace["proposal_scores_joint"][..., 1:]
    assert float((e_scores[:, 1:] - e_scores[:, :-1]).abs().max()) > 1.0e-6
    assert core.member_router.proposal_M is not None
    # One scorer owns both role calls; no copied E scorer is introduced.
    assert not hasattr(core.member_router, "proposal_E")


def test_joint95_member_scores_run_only_for_selected_original_rows() -> None:
    torch.manual_seed(31)
    core = _core("ME")
    core.eval()
    prepared = core.prepare(_scene(5, 10))
    query = _query(3)
    score_rows = []
    handles = [
        core.member_router.member_M.register_forward_pre_hook(
            lambda _module, inputs: score_rows.append(("M", int(inputs[0].shape[0])))
        ),
        core.member_router.member_E.register_forward_pre_hook(
            lambda _module, inputs: score_rows.append(("E", int(inputs[0].shape[0])))
        ),
    ]
    try:
        with core.read_execution("full"):
            full = core.decode(prepared, query, return_trace=True)
        full_rows = list(score_rows)
        score_rows.clear()
        with core.read_execution("restricted"):
            restricted = core.decode(prepared, query, return_trace=True)
        restricted_rows = list(score_rows)
    finally:
        for handle in handles:
            handle.remove()

    for observed, trace in ((full_rows, full.trace), (restricted_rows, restricted.trace)):
        assert trace is not None
        assert observed == [
            ("M", int(trace["candidate_mask_M"].sum())),
            ("E", int(trace["candidate_mask_E"].sum())),
        ]
    assert sum(count for _role, count in restricted_rows) <= sum(count for _role, count in full_rows)
    assert any(restricted_count < full_count
               for (_role, restricted_count), (_same_role, full_count)
               in zip(restricted_rows, full_rows))


def test_address95_is_post_score_and_recomposes_the_member_prior() -> None:
    torch.manual_seed(33)
    core = _core("ME")
    core.eval()
    prepared = core.prepare(_scene(4, 8))
    query = _query(3)
    with core.read_execution("full"):
        uncut = core.decode(prepared, query, return_trace=True)
        normalized_cut = core.decode(
            prepared, query, intervention={"address_coverage": 0.5}, return_trace=True,
        )
        raw_cut = core.decode(
            prepared, query,
            intervention={"address_coverage": 0.5, "retrieval_mode": "raw"},
            return_trace=True,
        )
    assert uncut.trace is not None and normalized_cut.trace is not None and raw_cut.trace is not None
    before = uncut.trace
    normalized = normalized_cut.trace
    raw = raw_cut.trace
    assert normalized_cut.work_counts["query_M_proposal_score_rows"] == int(before["address_eligible_mask"][..., :4].sum())
    assert normalized_cut.work_counts["query_E_proposal_score_rows"] == int(before["address_eligible_mask"][..., 4:].sum())
    assert torch.equal(normalized["proposal_weights_joint_uncut"], before["proposal_weights_joint"])
    torch.testing.assert_close(normalized["prior_P_uncut"], before["prior_P"])
    assert bool((normalized["address_omitted_transport_fraction"] <= 0.50001).all())
    assert bool((raw["address_omitted_transport_fraction"] <= 0.50001).all())
    for trace in (normalized, raw):
        mask = trace["address_candidate_mask"]
        weights = trace["proposal_weights_joint"]
        assert bool((weights[~mask] == 0).all())
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    expected_normalized = torch.bmm(
        normalized["proposal_weights_joint"], prepared.retrieval_R,
    )
    torch.testing.assert_close(normalized["prior_P"], expected_normalized)
    expected_raw_transport = (
        raw["proposal_weights_joint_uncut"] * prepared.retrieval_row_mass[:, None, :]
    )
    torch.testing.assert_close(raw["address_transport_mass"], expected_raw_transport)


def test_native_field_loss_has_live_gradients_to_all_captured_blocks_and_values() -> None:
    torch.manual_seed(39)
    core = _core("ME")
    core.train()
    prepared = core.prepare(_scene(5, 8))
    output = core.decode(prepared, _query(4), return_trace=True)
    objective = output.joint_fields.square().mean()
    targets = (
        prepared.H_MM_applied, prepared.H_EM_applied,
        prepared.H_EE_applied, prepared.H_ME_applied,
        prepared.member_values_M, prepared.member_values_E,
        output.trace["proposal_logits_M"], output.trace["proposal_logits_E"],
    )
    gradients = torch.autograd.grad(objective, targets, allow_unused=True)
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None)


def test_checkpoint_recomputation_keeps_restricted_mode_after_context_exits(monkeypatch) -> None:
    torch.manual_seed(44)
    core = _core("ME", tile_size=2)
    core.train()
    calls = []
    original = joint_module.joint_member_prefix_mask

    def capture(*args, **kwargs):
        calls.append(tuple(args[0].shape))
        return original(*args, **kwargs)

    monkeypatch.setattr(joint_module, "joint_member_prefix_mask", capture)
    prepared = core.prepare(_scene(5, 10))
    with core.read_execution("restricted"):
        output = core.decode(prepared, _query(4))
    output.joint_fields.square().mean().backward()
    # Two forward tiles plus their two checkpoint recomputations all remain restricted.
    assert len(calls) == 4
    assert all(shape == (1, 2, 15) for shape in calls)
