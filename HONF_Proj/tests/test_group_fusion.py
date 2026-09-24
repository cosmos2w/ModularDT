from __future__ import annotations

import torch

from honf_forward_core.interface_fields.sparse_incidence_router import (
    SparseIncidenceGroupRouter,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase
from honf_forward_core.organization import group_fusion
from honf_forward_core.organization.group_fusion import (
    ADMM_STEPS,
    EPS_ABS,
    EPS_REL,
    MAX_ADMM_ITERATIONS,
    GroupFusionPlan,
    build_group_descriptors,
    effective_query_centers,
    source_footprint_weights,
    weighted_sparsemax,
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:2" if torch.cuda.device_count() > 2 else "cuda:0")
    return torch.device("cpu")


def _sparsemax(scores: torch.Tensor) -> torch.Tensor:
    sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)
    rank = torch.arange(
        1, scores.shape[-1] + 1, device=scores.device, dtype=scores.dtype
    )
    cumulative = sorted_scores.cumsum(dim=-1) - 1.0
    support = rank * sorted_scores > cumulative
    support_count = support.sum(dim=-1, keepdim=True).clamp_min(1)
    tau = torch.gather(cumulative, -1, support_count - 1) / support_count.to(scores.dtype)
    return torch.relu(scores - tau)


def test_weighted_sparsemax_matches_literal_unequal_multiplicity_and_gradient() -> None:
    device = _device()
    torch.manual_seed(12)
    scores = torch.tensor(
        [
            [[0.8, 0.6, 0.5], [0.1, 0.9, 0.2]],
            [[0.3, 0.7, 0.4], [0.9, 0.2, 0.6]],
        ],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    multiplicity = torch.tensor([[2, 1, 3], [1, 3, 2]], device=device)
    mass, density = weighted_sparsemax(scores, multiplicity)

    literal_mass_rows = []
    literal_density_rows = []
    for batch_index in range(scores.shape[0]):
        expanded_logits = torch.repeat_interleave(
            scores[batch_index],
            multiplicity[batch_index],
            dim=-1,
        )
        expanded_density = _sparsemax(expanded_logits)
        mass_rows = []
        density_rows = []
        start = 0
        for count in multiplicity[batch_index].tolist():
            values = expanded_density[..., start : start + count]
            mass_rows.append(values.sum(dim=-1))
            density_rows.append(values.mean(dim=-1))
            start += count
        literal_mass_rows.append(torch.stack(mass_rows, dim=-1))
        literal_density_rows.append(torch.stack(density_rows, dim=-1))
    literal_mass = torch.stack(literal_mass_rows)
    literal_density = torch.stack(literal_density_rows)
    torch.testing.assert_close(mass, literal_mass, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(density, literal_density, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(mass.sum(dim=-1), torch.ones_like(mass[..., 0]))

    mass_coeff = torch.randn_like(mass)
    density_coeff = torch.randn_like(density)
    compact_loss = (mass * mass_coeff + density * density_coeff).sum()
    compact_grad = torch.autograd.grad(compact_loss, scores, retain_graph=True)[0]
    literal_loss = (literal_mass * mass_coeff + literal_density * density_coeff).sum()
    literal_grad = torch.autograd.grad(literal_loss, scores)[0]
    torch.testing.assert_close(compact_grad, literal_grad, atol=1.0e-11, rtol=1.0e-11)


def test_descriptor_builder_uses_parent_fallback_and_exact_decode() -> None:
    device = _device()
    keys = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device=device)
    module_centers = torch.tensor([[[1.0], [0.0]]], device=device)
    environment_centers = torch.tensor([[[0.0], [2.0]]], device=device)
    module_mass = torch.tensor([[1.0, 0.0]], device=device)
    environment_mass = torch.tensor([[0.0, 1.0]], device=device)
    cm, ce = effective_query_centers(
        module_centers, environment_centers, module_mass, environment_mass
    )
    descriptor = build_group_descriptors(keys, cm, ce, torch.tensor([2.0], device=device))
    plan = GroupFusionPlan.build(
        descriptor,
        torch.zeros((1, 2, 2), device=device),
        torch.ones((1, 2), dtype=torch.bool, device=device),
        0.0,
        control_dim=2,
        geometry_scale=2.0,
        parent_query_keys=keys,
        parent_module_query_centers=cm,
        parent_environment_query_centers=ce,
    )
    torch.testing.assert_close(plan.query_keys, keys, rtol=0.0, atol=0.0)
    torch.testing.assert_close(plan.module_query_centers, cm, rtol=0.0, atol=0.0)
    torch.testing.assert_close(plan.environment_query_centers, ce, rtol=0.0, atol=0.0)


def test_zero_strength_is_exact_identity_and_preserves_original_slots() -> None:
    device = _device()
    descriptors = torch.randn((1, 4, 6), dtype=torch.float64, device=device)
    weights = torch.ones((1, 4, 4), dtype=torch.float64, device=device)
    active = torch.tensor([[True, False, True, True]], device=device)
    plan = GroupFusionPlan.build(
        descriptors,
        weights,
        active,
        0.0,
        control_dim=2,
    )
    assert plan.class_index.tolist() == [[0, -1, 2, 3]]
    assert plan.valid.tolist() == [[True, False, True, True]]
    torch.testing.assert_close(
        plan.fused_descriptors[:, active[0]],
        descriptors[:, active[0]],
        rtol=0.0,
        atol=0.0,
    )

    incidence = torch.rand((1, 3, 4), dtype=torch.float64, device=device)
    incidence[:, :, 1] = 0.0
    control = torch.randn((1, 4, 3), dtype=torch.float64, device=device)
    merged, moment = plan.reduce_source_banks(incidence, control)
    torch.testing.assert_close(merged, incidence, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        moment,
        incidence[..., None] * control[:, None, :, :],
        rtol=0.0,
        atol=0.0,
    )


def test_source_footprint_weights_are_balanced_symmetric_and_masked() -> None:
    device = _device()
    descriptors = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float64,
        device=device,
    )
    module_incidence = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64, device=device
    )
    environment_incidence = torch.tensor(
        [[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float64, device=device
    )
    module_measure = torch.tensor([[0.5, 0.5]], dtype=torch.float64, device=device)
    environment_measure = torch.tensor([[0.5, 0.5]], dtype=torch.float64, device=device)
    active_module = torch.tensor([[True, False]], device=device)
    active_groups = torch.tensor([[True, True, False]], device=device)
    weights = source_footprint_weights(
        module_incidence,
        environment_incidence,
        module_measure,
        environment_measure,
        active_module,
        descriptors,
        active_groups,
    )
    torch.testing.assert_close(weights, weights.transpose(-1, -2))
    torch.testing.assert_close(
        weights.diagonal(dim1=-2, dim2=-1),
        torch.zeros_like(weights[..., 0]),
    )
    assert torch.equal(weights[:, 2, :], torch.zeros_like(weights[:, 2, :]))
    torch.testing.assert_close(weights.sum(dim=-1).amax(dim=-1), torch.ones(1, device=device, dtype=weights.dtype))


def test_admm_merges_repeated_access_functions_reversibly_and_stably(
    monkeypatch: object,
) -> None:
    device = _device()
    descriptors = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
                [6.0, 0.0, 0.0, 0.0],
                [6.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    groups = descriptors.shape[1]
    weights = (
        torch.ones((1, groups, groups), dtype=torch.float32, device=device)
        - torch.eye(groups, dtype=torch.float32, device=device)[None, :, :]
    ) / float(groups - 1)
    active = torch.ones((1, groups), dtype=torch.bool, device=device)
    plan = GroupFusionPlan.build(
        descriptors, weights, active, 0.5, control_dim=2, merge_tolerance=1.0e-5
    )
    assert plan.class_index.tolist() == [[0, 0, 1, 2, 2]]
    assert plan.multiplicity.tolist() == [[2, 1, 2, 0, 0]]
    assert plan.valid.tolist() == [[True, True, True, False, False]]
    assert plan.case_group_count.tolist() == [3]
    assert ADMM_STEPS == 64
    assert torch.isfinite(plan.primal_residual).all()
    assert torch.isfinite(plan.dual_residual).all()
    assert float(plan.max_component_projection_displacement.max()) <= 1.0e-5

    half_tolerance = GroupFusionPlan.build(
        descriptors,
        weights,
        active,
        0.5,
        control_dim=2,
        merge_tolerance=5.0e-6,
    )
    assert torch.equal(plan.class_index, half_tolerance.class_index)
    monkeypatch.setattr(group_fusion, "ADMM_STEPS", 128)
    doubled_iterations = GroupFusionPlan.build(
        descriptors, weights, active, 0.5, control_dim=2, merge_tolerance=1.0e-5
    )
    assert torch.equal(plan.class_index, doubled_iterations.class_index)
    torch.testing.assert_close(
        plan.fused_descriptors,
        doubled_iterations.fused_descriptors,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    grad = torch.autograd.grad(plan.fused_descriptors.square().sum(), descriptors)[0]
    assert torch.isfinite(grad).all()


def test_source_resolved_moments_match_literal_virtual_fused_router() -> None:
    device = _device()
    descriptors = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float64,
        device=device,
    )
    weights = (
        torch.ones((1, 3, 3), dtype=torch.float64, device=device)
        - torch.eye(3, dtype=torch.float64, device=device)[None, :, :]
    ) / 2.0
    active = torch.ones((1, 3), dtype=torch.bool, device=device)
    plan = GroupFusionPlan.build(
        descriptors, weights, active, 0.5, control_dim=2
    )
    assert plan.class_index.tolist() == [[0, 0, 1]]
    incidence = torch.tensor(
        [[[0.7, 0.2, 0.1], [0.0, 0.4, 0.6]]], dtype=torch.float64, device=device
    )
    control = torch.tensor(
        [[[1.0, 2.0], [3.0, -1.0], [-2.0, 4.0]]], dtype=torch.float64, device=device
    )
    merged_incidence, moment_bank = plan.reduce_source_banks(incidence, control)
    fused_scores = torch.tensor(
        [[[0.8, 0.2, 0.0], [0.1, 0.7, 0.0]]], dtype=torch.float64, device=device
    )
    mass, density = weighted_sparsemax(fused_scores, plan.multiplicity, plan.valid)
    virtual_scores = fused_scores.gather(
        -1, plan.class_index.clamp_min(0)[:, None, :].expand(-1, fused_scores.shape[1], -1)
    )
    virtual_mass, _ = weighted_sparsemax(
        virtual_scores,
        torch.ones_like(plan.class_index),
        plan.active,
    )
    rho_compact = torch.einsum("bqr,bsr->bqs", density, merged_incidence)
    rho_virtual = torch.einsum("bqk,bsk->bqs", virtual_mass, incidence)
    moment_compact = torch.einsum("bqr,bsrd->bqsd", density, moment_bank)
    moment_virtual = torch.einsum(
        "bqk,bsk,bkd->bqsd", virtual_mass, incidence, control
    )
    torch.testing.assert_close(rho_compact, rho_virtual, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(
        moment_compact, moment_virtual, atol=1.0e-12, rtol=1.0e-12
    )
    torch.testing.assert_close(mass.sum(dim=-1), torch.ones_like(mass[..., 0]))


def test_fusion_gradients_reach_descriptors_and_pair_weights() -> None:
    device = _device()
    descriptors = torch.tensor(
        [[[0.1, 0.3, 0.0, 0.0], [1.0, -0.2, 0.0, 0.0], [3.0, 0.7, 0.0, 0.0]]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    base_weights = (
        torch.ones((1, 3, 3), dtype=torch.float64, device=device)
        - torch.eye(3, dtype=torch.float64, device=device)[None, :, :]
    ) / 2.0
    weights = base_weights.clone().requires_grad_(True)
    plan = GroupFusionPlan.build(
        descriptors,
        weights,
        torch.ones((1, 3), dtype=torch.bool, device=device),
        0.1,
        control_dim=2,
    )
    objective = (plan.fused_descriptors * torch.tensor(
        [[[0.2, -0.4, 1.0, 0.5], [0.6, 0.1, -0.2, 0.3], [0.7, 0.8, 0.2, -0.1]]],
        dtype=torch.float64,
        device=device,
    )).sum()
    descriptor_grad, weight_grad = torch.autograd.grad(
        objective, (descriptors, weights)
    )
    assert torch.isfinite(descriptor_grad).all()
    assert torch.isfinite(weight_grad).all()
    assert float(descriptor_grad.abs().sum()) > 0.0
    assert float(weight_grad.abs().sum()) > 0.0


def test_source_footprint_construction_retains_design_gradients() -> None:
    device = _device()
    descriptors = torch.tensor(
        [[[0.1, 0.2, 0.0, 0.0], [0.8, -0.1, 0.3, 0.0], [1.9, 0.4, 0.0, 0.2]]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    module_incidence = torch.tensor(
        [[[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    environment_incidence = torch.tensor(
        [[[0.5, 0.2, 0.3], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4], [0.2, 0.5, 0.3]]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    module_measure = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64, device=device)
    environment_measure = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64, device=device)
    active_module = torch.ones((1, 3), dtype=torch.bool, device=device)
    active_groups = torch.ones((1, 3), dtype=torch.bool, device=device)
    weights = source_footprint_weights(
        module_incidence,
        environment_incidence,
        module_measure,
        environment_measure,
        active_module,
        descriptors,
        active_groups,
    )
    probe = torch.tensor(
        [[[0.0, 0.3, -0.2], [0.3, 0.0, 0.7], [-0.2, 0.7, 0.0]]],
        dtype=torch.float64,
        device=device,
    )
    gradients = torch.autograd.grad(
        (weights * probe).sum(),
        (descriptors, module_incidence, environment_incidence),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0


def test_empty_footprint_columns_and_inactive_nan_descriptors_have_finite_gradients() -> None:
    device = _device()
    descriptors = torch.tensor(
        [[[0.1, 0.2, 0.0, 0.0], [0.8, -0.1, 0.3, 0.0], [float("nan")] * 4]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    module_incidence = torch.tensor(
        [[[0.7, 0.3, float("nan")], [0.2, 0.8, float("nan")]]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    environment_incidence = torch.tensor(
        [[[0.6, 0.4, float("nan")], [0.1, 0.9, float("nan")], [0.8, 0.2, float("nan")]]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    weights = source_footprint_weights(
        module_incidence,
        environment_incidence,
        torch.tensor([[0.5, 0.5]], dtype=torch.float64, device=device),
        torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64, device=device),
        torch.ones((1, 2), dtype=torch.bool, device=device),
        descriptors,
        torch.tensor([[True, True, False]], device=device),
    )
    assert torch.isfinite(weights).all()
    gradients = torch.autograd.grad(
        weights.square().sum(),
        (descriptors, module_incidence, environment_incidence),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
    assert torch.equal(gradients[0][:, 2], torch.zeros_like(gradients[0][:, 2]))
    assert torch.equal(
        gradients[1][:, :, 2], torch.zeros_like(gradients[1][:, :, 2])
    )
    assert torch.equal(
        gradients[2][:, :, 2], torch.zeros_like(gradients[2][:, :, 2])
    )


def test_zero_admm_edges_and_singleton_sigma_have_finite_gradients() -> None:
    device = _device()
    descriptors = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [2.0, -1.0, 0.2, 0.1],
                [float("nan"), float("nan"), float("nan"), float("nan")],
            ]
        ],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    active = torch.tensor([[True, True, True, False]], device=device)
    raw = torch.ones((1, 4, 4), dtype=torch.float64, device=device)
    raw = raw - torch.eye(4, dtype=torch.float64, device=device)[None, :, :]
    raw = raw * (active[:, :, None] & active[:, None, :]).to(raw.dtype) / 2.0
    pair_weights = raw.requires_grad_(True)
    plan = GroupFusionPlan.build(
        descriptors,
        pair_weights,
        active,
        0.5,
        control_dim=2,
    )
    assert torch.isfinite(plan.fused_descriptors).all()
    objective = plan.fused_descriptors.square().sum() + plan.lambda_b.sum()
    descriptor_gradient, weight_gradient = torch.autograd.grad(
        objective, (descriptors, pair_weights)
    )
    assert torch.isfinite(descriptor_gradient).all()
    assert torch.isfinite(weight_gradient).all()
    assert torch.equal(
        descriptor_gradient[:, 3], torch.zeros_like(descriptor_gradient[:, 3])
    )

    singleton_descriptors = torch.tensor(
        [[[0.25, -0.5, 0.0, 0.0], [float("nan")] * 4]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    singleton_weights = torch.zeros(
        (1, 2, 2), dtype=torch.float64, device=device, requires_grad=True
    )
    singleton = GroupFusionPlan.build(
        singleton_descriptors,
        singleton_weights,
        torch.tensor([[True, False]], device=device),
        0.5,
        control_dim=2,
    )
    singleton_objective = singleton.fused_descriptors.square().sum() + singleton.lambda_b.sum()
    singleton_gradients = torch.autograd.grad(
        singleton_objective, (singleton_descriptors, singleton_weights)
    )
    for gradient in singleton_gradients:
        assert torch.isfinite(gradient).all()
    assert singleton.sigma.item() == 0.0


def test_real_router_prepared_descriptors_are_stable_to_tolerance_and_iterations(
    monkeypatch: object,
) -> None:
    device = _device()
    torch.manual_seed(15103)
    batch, modules, environments, hidden = 2, 5, 9, 16
    module_states = torch.randn((batch, modules, hidden), device=device)
    environment_states = torch.randn((batch, environments, hidden), device=device)
    global_token = torch.randn((batch, hidden), device=device)
    module_centers = torch.rand((batch, modules, 2), device=device)
    environment_centers = torch.rand((batch, environments, 2), device=device)
    module_present = torch.ones((batch, modules), device=device)
    module_present[:, -1] = 0.0
    encoded = EncodedInterfaceCase(
        module_tokens=module_states,
        env_tokens=environment_states,
        global_token=global_token,
        module_centers=module_centers,
        env_coords=environment_centers,
        module_present=module_present,
        module_features=torch.zeros((batch, modules, 1), device=device),
        env_features=None,
        env_weights=torch.full((batch, environments), 1.0 / environments, device=device),
        coordinate_scale=torch.tensor([[1.5, 1.0], [1.0, 1.4]], device=device),
    )
    router = SparseIncidenceGroupRouter(
        hidden,
        fourier_frequencies=2,
        spatial_dim=2,
    ).to(device)
    prepared = router.prepare(encoded, module_states, environment_states)
    query_keys = router.normalized_query_key_bank(prepared)
    module_query_centers, environment_query_centers = effective_query_centers(
        prepared.module_centres,
        prepared.environment_centres,
        prepared.module_mass,
        prepared.environment_mass,
    )
    geometry_scale = router._geometry_scale(encoded)
    descriptors = build_group_descriptors(
        query_keys,
        module_query_centers,
        environment_query_centers,
        geometry_scale,
    )
    weights = source_footprint_weights(
        prepared.module_membership,
        prepared.environment_membership,
        prepared.module_measure,
        prepared.environment_measure,
        encoded.module_present > 0.5,
        descriptors,
        prepared.phase_occupied,
    )
    plan64 = GroupFusionPlan.build(
        descriptors,
        weights,
        prepared.phase_occupied,
        0.5,
        control_dim=16,
        geometry_scale=geometry_scale,
        parent_query_keys=query_keys,
        parent_module_query_centers=module_query_centers,
        parent_environment_query_centers=environment_query_centers,
    )
    plan_half_tolerance = GroupFusionPlan.build(
        descriptors,
        weights,
        prepared.phase_occupied,
        0.5,
        control_dim=16,
        geometry_scale=geometry_scale,
        merge_tolerance=5.0e-6,
    )
    assert torch.equal(plan64.class_index, plan_half_tolerance.class_index)
    monkeypatch.setattr(group_fusion, "ADMM_STEPS", 128)
    plan128 = GroupFusionPlan.build(
        descriptors,
        weights,
        prepared.phase_occupied,
        0.5,
        control_dim=16,
        geometry_scale=geometry_scale,
        merge_tolerance=1.0e-5,
    )
    assert torch.equal(plan64.class_index, plan128.class_index)
    torch.testing.assert_close(
        plan64.fused_descriptors,
        plan128.fused_descriptors,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.isfinite(plan64.primal_residual).all()
    assert torch.isfinite(plan64.dual_residual).all()
    assert (plan64.case_group_count <= plan64.active_proposal_count).all()


def test_solver_gradients_match_finite_differences_away_from_merge_boundary() -> None:
    device = _device()
    descriptors = torch.tensor(
        [[[0.2, 0.3, 0.0, 0.0], [1.0, 0.2, 0.0, 0.0], [2.5, -0.1, 0.0, 0.0]]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    weights = (
        torch.ones((1, 3, 3), dtype=torch.float64, device=device)
        - torch.eye(3, dtype=torch.float64, device=device)[None, :, :]
    ).requires_grad_(True)
    active = torch.ones((1, 3), dtype=torch.bool, device=device)

    def fused(z: torch.Tensor, pair_weights: torch.Tensor) -> torch.Tensor:
        return GroupFusionPlan.build(
            z,
            pair_weights,
            active,
            0.03,
            control_dim=2,
        ).fused_descriptors

    assert torch.autograd.gradcheck(
        fused,
        (descriptors, weights),
        eps=1.0e-6,
        atol=2.0e-5,
        rtol=2.0e-4,
        fast_mode=True,
    )


def test_converged_planner_is_detached_fp64_and_stable_to_class_tolerance() -> None:
    assert EPS_ABS == 1.0e-9
    assert EPS_REL == 1.0e-8
    assert MAX_ADMM_ITERATIONS == 512
    device = _device()
    descriptors = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0],
          [6.0, 0.0, 0.0, 0.0], [6.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    groups = descriptors.shape[1]
    weights = (
        torch.ones((1, groups, groups), dtype=descriptors.dtype, device=device)
        - torch.eye(groups, dtype=descriptors.dtype, device=device)[None]
    ) / float(groups - 1)
    active = torch.ones((1, groups), dtype=torch.bool, device=device)
    query_keys = torch.randn((1, groups, 2), dtype=descriptors.dtype, device=device)
    module_centers = torch.randn((1, groups, 1), dtype=descriptors.dtype, device=device)
    environment_centers = torch.randn_like(module_centers)

    plan = GroupFusionPlan.build_converged(
        descriptors,
        weights,
        active,
        0.5,
        control_dim=2,
        parent_query_keys=query_keys,
        parent_module_query_centers=module_centers,
        parent_environment_query_centers=environment_centers,
    )
    half_tolerance = GroupFusionPlan.build_converged(
        descriptors,
        weights,
        active,
        0.5,
        control_dim=2,
        merge_tolerance=5.0e-6,
    )

    assert plan.solver_converged is not None and plan.solver_converged.tolist() == [True]
    assert 1 <= int(plan.solver_iterations[0]) <= MAX_ADMM_ITERATIONS
    assert plan.primal_residual.dtype == torch.float64
    assert plan.dual_residual.dtype == torch.float64
    assert plan.primal_tolerance is not None and plan.primal_tolerance.dtype == torch.float64
    assert plan.dual_tolerance is not None and plan.dual_tolerance.dtype == torch.float64
    assert (plan.primal_residual <= plan.primal_tolerance).all()
    assert (plan.dual_residual <= plan.dual_tolerance).all()
    assert plan.class_index.tolist() == [[0, 0, 1, 2, 2]]
    assert torch.equal(plan.class_index, half_tolerance.class_index)
    assert not plan.fused_descriptors.requires_grad
    assert not plan.sigma.requires_grad
    assert not plan.lambda_b.requires_grad
    # The singleton compact class is the original parent query/access row.
    torch.testing.assert_close(plan.query_keys[:, 1], query_keys[:, 2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        plan.module_query_centers[:, 1], module_centers[:, 2], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        plan.environment_query_centers[:, 1],
        environment_centers[:, 2],
        rtol=0.0,
        atol=0.0,
    )


def test_converged_planner_preserves_parent_slots_without_merges_and_falls_back() -> None:
    device = _device()
    descriptors = torch.tensor(
        [
            [[0.1, 0.2, 0.0, 0.0], [float("nan")] * 4, [0.8, -0.1, 0.0, 0.0], [2.0, 0.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [6.0, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    active = torch.tensor(
        [[True, False, True, True], [True, True, True, True]], device=device
    )
    weights = torch.zeros((2, 4, 4), dtype=descriptors.dtype, device=device)
    weights[1] = torch.ones((4, 4), dtype=descriptors.dtype, device=device) - torch.eye(
        4, dtype=descriptors.dtype, device=device
    )
    parent_keys = torch.randn((2, 4, 2), device=device)
    plan = GroupFusionPlan.build_converged(
        descriptors,
        weights,
        active,
        0.5,
        control_dim=2,
        parent_query_keys=parent_keys,
        max_iterations=1,
    )

    assert plan.solver_converged is not None
    assert plan.solver_converged.tolist() == [True, False]
    assert plan.solver_iterations.tolist() == [0, 1]
    assert plan.class_index.tolist() == [[0, -1, 2, 3], [0, 1, 2, 3]]
    assert plan.multiplicity.tolist() == [[1, 0, 1, 1], [1, 1, 1, 1]]
    assert torch.equal(plan.valid, active)
    assert not plan.pair_fused.any()
    assert torch.equal(plan.fused_descriptors[0, active[0]], descriptors[0, active[0]])
    assert torch.equal(plan.fused_descriptors[1], torch.nan_to_num(descriptors[1]))
    torch.testing.assert_close(
        plan.query_keys[0, active[0]], parent_keys[0, active[0]], rtol=0.0, atol=0.0
    )
    assert (plan.primal_residual[1] > plan.primal_tolerance[1]) or (
        plan.dual_residual[1] > plan.dual_tolerance[1]
    )
