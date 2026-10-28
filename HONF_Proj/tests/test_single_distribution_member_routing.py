from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from honf_forward_core.interaction_reused_member_routing import faithful, consistent
from honf_forward_core.interaction_reused_member_routing.relative import read_probability
from honf_forward_core.interaction_reused_member_routing.single_distribution_core import (
    SingleDistributionMemberRoutingCore,
    _normalized_role_statistics,
)
from honf_forward_core.interaction_reused_member_routing.typed_organization import (
    organization_conditioned_role_contribution,
)
from honf_forward_core.interaction_reused_member_routing.typed_organization_core import (
    TypedOrganizationMemberRoutingCore,
)
from tests.test_faithful_member_routing import _base_config, _query, _scene


def _models():
    h = TypedOrganizationMemberRoutingCore(_base_config(), query_tile_size=4)
    u = SingleDistributionMemberRoutingCore(_base_config(), query_tile_size=4)
    u.load_state_dict(h.state_dict(), strict=True)
    return h, u


def test_statistics_use_the_exact_final_tensor_without_renormalizing():
    p = torch.tensor([[[0.2, 0.7999, 0.0]]], dtype=torch.double, requires_grad=True)
    ell = torch.tensor([[[0.7, -0.4, 0.0]]], dtype=torch.double)
    selected = torch.tensor([[[True, True, False]]])
    actual_p, mean, entropy = _normalized_role_statistics(p, ell, selected, torch.tensor([7]))
    assert actual_p is p
    torch.testing.assert_close(mean, (p * ell).sum(-1), rtol=0, atol=0)
    expected_entropy = -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1) / torch.log(p.new_tensor(7.))
    torch.testing.assert_close(entropy, expected_entropy, rtol=0, atol=0)
    # A second normalization would change both moments and their Jacobians.
    normalized_again = consistent.joint_role_statistics(p, ell, selected, torch.tensor([7]))
    assert not torch.equal(mean, normalized_again[1])
    assert not torch.equal(entropy, normalized_again[2])
    assert torch.autograd.grad((mean + entropy).sum(), p)[0].isfinite().all()


def test_empty_singleton_and_padded_statistics_preserve_exact_zeros():
    p = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]], dtype=torch.double)
    scores = torch.tensor([[[0., 0., 0.], [0.4, 0., 0.]]], dtype=torch.double)
    candidates = p > 0
    same_p, mean, entropy = _normalized_role_statistics(p, scores, candidates, torch.tensor([[0, 1]]))
    assert same_p is p
    assert torch.equal(mean, torch.tensor([[0., 0.4]], dtype=torch.double))
    assert torch.equal(entropy, torch.zeros_like(entropy))
    with pytest.raises(ValueError, match="scene_eligible_count"):
        _normalized_role_statistics(p, scores, candidates, torch.ones(1, 3))


def test_u_config_and_learned_inventory_are_versioned_without_new_parameters():
    h, u = _models()
    assert tuple(h.state_dict()) == tuple(u.state_dict())
    assert all(torch.equal(h.state_dict()[name], value) for name, value in u.state_dict().items())
    config = u.get_config()
    assert config["schema"] == "interaction_reused_single_distribution_member_routing_v1"
    assert config["architecture"] == "R13-7-U-ME-SingleDistributionJoint95"
    restored = SingleDistributionMemberRoutingCore.from_config(config)
    restored.load_state_dict(u.state_dict(), strict=True)
    assert restored.get_config() == config
    with pytest.raises(ValueError):
        SingleDistributionMemberRoutingCore(_base_config(), retained_prior_law="K")


@pytest.mark.parametrize("mode", ["full", "restricted"])
def test_same_state_upstream_plan_and_local_context_change_identity(mode):
    h, u = _models()
    h.double().eval()
    u.double().eval()
    scene, query = _scene(), _query()
    # Resolve the small context difference without FP32 subtraction noise.
    def double_batch(batch):
        return replace(batch, **{
            field.name: value.double() for field in fields(batch)
            if torch.is_tensor(value := getattr(batch, field.name)) and value.is_floating_point()
        })
    scene, query = double_batch(scene), double_batch(query)
    mask = query.query_mask.clone()
    mask[0, -1] = False
    query = replace(query, query_mask=mask)
    prepared_h, prepared_u = h.prepare(scene), u.prepare(scene)
    old = h.decode(prepared_h, query, intervention={"read_mode": mode}, return_trace=True)
    new = u.decode(prepared_u, query, intervention={"read_mode": mode}, return_trace=True)
    th, tu = old.trace, new.trace
    for key in (
        "proposal_weights_joint", "retrieval_R", "prior_P", "candidate_mask_joint",
        "candidate_ids_M", "candidate_ids_E", "member_logits_M", "member_logits_E",
        "member_ell_M", "member_ell_E", "H_MM_applied", "H_EM_applied",
        "H_EE_applied", "H_ME_applied", "H_self_M", "H_route_cycle_indices",
    ):
        torch.testing.assert_close(th[key], tu[key], rtol=0, atol=0, msg=key)
    for role, values in (("M", prepared_h.member_values_M), ("E", prepared_h.member_values_E)):
        p = tu[f"member_value_probabilities_{role}"]
        statistic_p = tu[f"member_strength_probabilities_{role}"]
        assert p.data_ptr() == statistic_p.data_ptr()
        torch.testing.assert_close(p, th[f"member_contribution_pre_strength_{role}"], rtol=0, atol=0)
        mu = torch.bmm(p, values)
        rho_h, rho_u = th[f"member_read_strength_{role}"], tu[f"member_read_strength_{role}"]
        expected_delta = (rho_u - rho_h)[..., None] * mu
        delta = tu[f"context_{role}"] - th[f"context_{role}"]
        torch.testing.assert_close(delta, expected_delta, rtol=1e-9, atol=2e-14)
        torch.testing.assert_close(tu[f"context_{role}"], rho_u[..., None] * mu, rtol=1e-12, atol=2e-14)
        assert torch.equal(tu[f"context_{role}"][0, -1], torch.zeros_like(mu[0, -1]))
        assert new.work_counts[f"query_{role}_member_softmax_rows"] == 0
    assert torch.equal(new.joint_fields[0, -1], torch.zeros_like(new.joint_fields[0, -1]))
    assert new.joint_fields.isfinite().all()


def test_u_ordinary_and_traced_reads_never_construct_parallel_o_statistics(monkeypatch):
    _, u = _models()
    u.eval()

    def forbidden(*args, **kwargs):
        raise AssertionError("U must not calculate O likelihoods or renormalize its statistics")

    monkeypatch.setattr(faithful, "original_role_conditional_likelihood", forbidden)
    monkeypatch.setattr(consistent, "joint_role_statistics", forbidden)
    prepared, query = u.prepare(_scene()), _query()
    ordinary = u.decode(prepared, query)
    traced = u.decode(prepared, query, return_trace=True)
    torch.testing.assert_close(ordinary.joint_fields, traced.joint_fields, rtol=0, atol=0)
    assert ordinary.work_counts["query_joint_likelihood_rows"] == 0
    assert traced.work_counts["query_joint_likelihood_rows"] > 0  # requested legacy A diagnostic
    assert traced.work_counts["query_M_member_softmax_rows"] == 0
    assert traced.work_counts["query_E_member_softmax_rows"] == 0


def test_one_permitted_row_and_singleton_role_keep_h_forward_behavior():
    h, u = _models()
    retrieval = torch.tensor([[[.4, .1, .2, .3], [.1, .3, .4, .2], [.2, .3, .1, .4]]], dtype=torch.double)
    proposal = torch.tensor([[[0., 1., 0.]]], dtype=torch.double)
    prior = torch.bmm(proposal, retrieval)
    ell = torch.tensor([[[-.7, .6, .8, -.1]]], dtype=torch.double)
    for candidate in (torch.ones_like(ell, dtype=torch.bool),
                      torch.tensor([[[True, False, False, False]]])):
        kwargs = dict(full_candidate=candidate, proposal=proposal, retrieval_R=retrieval,
                      row_mass=torch.ones(1, 3, dtype=torch.double), module_count=2)
        ph, vh, _ = h._role_read_contributions(None, prior, ell, candidate, **kwargs)
        pu, vu, _ = u._role_read_contributions(None, prior, ell, candidate, **kwargs)
        assert pu is vu
        torch.testing.assert_close(vh, vu, rtol=0, atol=0)
        torch.testing.assert_close(ph, pu, rtol=1e-12, atol=1e-12)
        for role_slice in (slice(0, 2), slice(2, 4)):
            old_stats = h._role_statistics(ph[..., role_slice], ell[..., role_slice], candidate[..., role_slice], torch.tensor([2]))
            new_stats = u._role_statistics(pu[..., role_slice], ell[..., role_slice], candidate[..., role_slice], torch.tensor([2]))
            for a, b in zip(old_stats, new_stats):
                torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_composed_h_distribution_statistics_strength_and_values_pass_gradcheck():
    torch.manual_seed(137)
    retrieval = (torch.rand(1, 3, 4, dtype=torch.double) + .2).requires_grad_()
    proposal = (torch.rand(1, 2, 3, dtype=torch.double) + .2).requires_grad_()
    ell = torch.randn(1, 2, 4, dtype=torch.double, requires_grad=True)
    values = torch.randn(1, 4, 2, dtype=torch.double, requires_grad=True)
    candidate = torch.tensor([[[True, True, False, True], [False, False, False, False]]])

    def read(r, w, scores, v):
        p = organization_conditioned_role_contribution(r, w, scores, candidate)
        _, expectation, entropy = _normalized_role_statistics(p, scores, candidate, torch.tensor([4]))
        rho = read_probability(.3 * expectation - .4 * entropy + .1) * candidate.any(-1)
        return rho[..., None] * torch.bmm(p, v)

    assert torch.autograd.gradcheck(read, (retrieval, proposal, ell, values), eps=1e-6, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("retained_role", ["M", "E", "empty"])
def test_pure_and_empty_role_decodes_do_not_force_a_read(retained_role):
    _, u = _models()
    u.eval()
    scene, query = _scene(), _query()

    def retrieval_view(prepared, controls):
        del controls
        r = torch.zeros_like(prepared.retrieval_R)
        if retained_role != "empty":
            member = 0 if retained_role == "M" else scene.num_modules
            r[..., member] = 1.
        return r, r, r.sum(-1), "synthetic_pure_role"

    u._retrieval_view = retrieval_view
    output = u.decode(u.prepare(scene), query, return_trace=True)
    for role in ("M", "E"):
        p = output.trace[f"member_value_probabilities_{role}"]
        if role == retained_role:
            torch.testing.assert_close(p.sum(-1), torch.ones_like(p.sum(-1)), rtol=0, atol=0)
        else:
            assert torch.equal(p, torch.zeros_like(p))
            for name in ("member_conditional_expectation", "member_conditional_entropy",
                         "member_read_strength", "context"):
                value = output.trace[f"{name}_{role}"]
                assert torch.equal(value, torch.zeros_like(value)), name
    assert output.joint_fields.isfinite().all()


def test_u_checkpointed_paired_decode_preserves_its_own_fields_and_gradients():
    _, separate = _models()
    _, paired = _models()
    paired.load_state_dict(separate.state_dict(), strict=True)
    separate.train()
    paired.train()
    scene, query = _scene(), _query()
    organization = separate.prepare(scene)
    full = separate.decode(organization, query, intervention={"read_mode": "full"})
    restricted = separate.decode(organization, query, intervention={"read_mode": "restricted"})
    full_pair, restricted_pair, shared = paired.decode_paired(paired.prepare(scene), query)
    for a, b in ((full, full_pair), (restricted, restricted_pair)):
        torch.testing.assert_close(a.joint_fields, b.joint_fields, rtol=0, atol=0)
        for name, count in a.work_counts.items():
            assert b.work_counts.get(name, 0) + shared.get(name, 0) == count, name
    probe = torch.linspace(-.5, .5, full.joint_fields.numel()).reshape_as(full.joint_fields)
    ((full.joint_fields * probe).sum() + .7 * (restricted.joint_fields * probe).sum()).backward()
    ((full_pair.joint_fields * probe).sum() + .7 * (restricted_pair.joint_fields * probe).sum()).backward()
    right = dict(paired.named_parameters())
    for name, parameter in separate.named_parameters():
        other = right[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            assert parameter.grad.isfinite().all()
            torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-4, atol=2e-6, msg=name)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in paired.typed_read_gates.parameters())
