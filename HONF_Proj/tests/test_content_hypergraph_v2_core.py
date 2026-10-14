from __future__ import annotations

from dataclasses import replace
from types import MethodType

import pytest
import torch
from torch.nn import functional as F

from honf_forward_core.content_hypergraph import (
    ContentHypergraphCoreV2,
    QueryBatch,
    SceneBatch,
)
from honf_forward_core.content_hypergraph.content_bank_v2 import _measure_weighted_edge_read
from honf_forward_core.content_hypergraph.spatial_features_v2 import (
    coordinate_gamma,
    edge_pair_score_features,
    pair_score_features,
    query_edge_features,
    query_source_features,
    relative_gamma,
)
from honf_forward_core.content_hypergraph.staged_attention_v2 import (
    SpatialRouteAttentionV2,
    explicit_spatial_attention,
)


def _scene(batch: int = 2) -> SceneBatch:
    module_features = torch.tensor([[0.2, 0.4], [0.7, -0.3], [-0.1, 0.8]])
    environment_features = torch.tensor([[0.1, 0.2, 0.3], [0.4, -0.2, 0.5], [-0.5, 0.7, 0.1], [0.8, 0.1, -0.4]])
    module_coordinates = torch.tensor([[0.0, 0.0], [0.5, 0.2], [1.0, 0.4]])
    environment_coordinates = torch.tensor([[-0.1, 0.3], [0.35, 0.1], [0.7, 0.5], [1.1, 0.2]])
    values = dict(
        module_features=module_features[None].expand(batch, -1, -1).clone(),
        environment_features=environment_features[None].expand(batch, -1, -1).clone(),
        module_coordinates=module_coordinates[None].expand(batch, -1, -1).clone(),
        environment_coordinates=environment_coordinates[None].expand(batch, -1, -1).clone(),
        module_mask=torch.ones((batch, 3), dtype=torch.bool),
        environment_mask=torch.ones((batch, 4), dtype=torch.bool),
        module_measure=torch.tensor([[0.7, 1.1, 0.9]]).expand(batch, -1).clone(),
        environment_measure=torch.tensor([[0.2, 0.4, 0.3, 0.5]]).expand(batch, -1).clone(),
        module_ids=torch.tensor([[13, 29, 47]]).expand(batch, -1).clone(),
        environment_ids=torch.tensor([[101, 107, 131, 149]]).expand(batch, -1).clone(),
        local_radius=torch.full((batch, 3), 0.32),
        operating_features=torch.tensor([[0.3, -0.2]]).expand(batch, -1).clone(),
        coordinate_origin=torch.tensor([[-0.1, 0.0]]).expand(batch, -1).clone(),
        coordinate_scale=torch.tensor([[1.2, 0.6]]).expand(batch, -1).clone(),
        module_measure_reference=torch.ones(batch),
        environment_measure_reference=torch.ones(batch),
        module_characteristic_lengths=torch.tensor([[[0.2, 0.12], [0.18, 0.10], [0.25, 0.14]]]).expand(batch, -1, -1).clone(),
        environment_characteristic_lengths=torch.full((batch, 4, 2), 0.1),
        scene_key="scene-A",
        normalizer_key="train-transform-v2",
        coordinate_frame_key="frame-A",
    )
    return SceneBatch(**values)


def _query(batch: int = 2, count: int = 5) -> QueryBatch:
    coordinates = torch.tensor([[0.06, 0.03], [0.49, 0.19], [0.91, 0.37], [0.52, 0.22], [0.4, 0.3]])
    features = torch.tensor([[0.1, 0.5], [-0.3, 0.7], [0.4, -0.1], [0.2, 0.6], [0.3, -0.4]])
    return QueryBatch(
        coordinates=coordinates[None].expand(batch, -1, -1)[:, :count].clone(),
        features=features[None].expand(batch, -1, -1)[:, :count].clone(),
        query_ids=torch.arange(count)[None].expand(batch, -1).clone(),
        scene_key="scene-A",
        coordinate_frame_key="frame-A",
        receiver_key="native-panel",
    )


def _core(architecture: str = "A2-H6", **overrides) -> ContentHypergraphCoreV2:
    values = dict(
        module_input_dim=2,
        environment_input_dim=3,
        coordinate_dim=2,
        query_input_dim=2,
        operating_input_dim=2,
        output_dim=3,
        architecture=architecture,
        context_dim=32,
        num_heads=4,
        context_depth=2,
        num_hyperedges=6,
        thermal_response_rank=4,
        local_hidden_dim=32,
        dropout=0.0,
        init_seed=1947,
    )
    values.update(overrides)
    core = ContentHypergraphCoreV2(**values)
    core.eval()
    return core


def _explicit_typed_gather(
    edge_query,
    donor_key,
    donor_value,
    geometry_score,
    anchors,
    edge_scales,
    donor_coordinates,
    donor_lengths,
    donor_measure,
    measure_reference,
    donor_mask,
    frame_scale,
    temperature,
):
    """Independent reference for the full typed W score and measure law."""
    q = F.normalize(edge_query, dim=-1, eps=1.0e-6)
    k = F.normalize(donor_key, dim=-1, eps=1.0e-6)
    scores = torch.einsum("bhd,bnd->bhn", q, k) * temperature
    descriptor, _relative_gamma, pair_length = edge_pair_score_features(
        anchors, donor_coordinates, frame_scale, edge_scales, donor_lengths
    )
    assert torch.all(pair_length > 0)
    delta = anchors[:, :, None, :] - donor_coordinates[:, None, :, :]
    scores = scores + geometry_score(descriptor).squeeze(-1)
    scores = scores - 0.5 * (delta / edge_scales[:, :, None, :]).square().sum(-1)
    measure = donor_measure.to(scores.dtype)
    valid = donor_mask & (measure > 0)
    safe_measure = torch.where(valid, measure, torch.ones_like(measure))
    scores = scores + (
        torch.log(safe_measure.to(torch.float64))
        - torch.log(measure_reference.to(torch.float64)[:, None])
    )[:, None, :].to(scores.dtype)
    allowed = valid[:, None, :].expand_as(scores)
    scores64 = scores.to(torch.float64).masked_fill(~allowed, -torch.inf)
    has_donor = allowed.any(-1, keepdim=True)
    scores64 = torch.where(has_donor, scores64, torch.zeros_like(scores64))
    weights = torch.softmax(scores64, dim=-1) * allowed.to(torch.float64)
    weights = torch.where(has_donor, weights, torch.zeros_like(weights)).to(scores.dtype)
    return weights, torch.einsum("bhn,bnd->bhd", weights, donor_value)


def test_v2_schema_explicit_frame_positive_lengths_and_gamma_values():
    core = _core()
    config = core.get_config()
    assert config["schema"] == "content_hypergraph_core_v2"
    assert ContentHypergraphCoreV2.from_config(config).get_config() == config
    with pytest.raises(ValueError, match="schema"):
        ContentHypergraphCoreV2.from_config({**config, "schema": "content_hypergraph_core_v1"})

    scene = _scene()
    gamma = coordinate_gamma(scene.module_coordinates, scene.coordinate_origin, scene.coordinate_scale)
    assert gamma.shape == (2, 3, 14)
    torch.testing.assert_close(gamma[..., :2], (scene.module_coordinates - scene.coordinate_origin[:, None, :]) / scene.coordinate_scale[:, None, :])
    with pytest.raises(ValueError, match="explicit coordinate_origin"):
        core.prepare(replace(scene, coordinate_origin=None, coordinate_scale=None))
    with pytest.raises(ValueError, match="characteristic_lengths"):
        core.prepare(replace(scene, module_characteristic_lengths=None))
    bad_lengths = scene.module_characteristic_lengths.clone()
    bad_lengths[0, 0, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        core.prepare(replace(scene, module_characteristic_lengths=bad_lengths))


def test_pair_lengths_are_geometric_means_per_axis():
    dest = torch.tensor([[[0.0, 0.0]]])
    donor = torch.tensor([[[3.0, 4.0], [1.0, 0.0]]])
    dest_length = torch.tensor([[[4.0, 9.0]]])
    donor_length = torch.tensor([[[1.0, 4.0], [16.0, 1.0]]])
    features, pair_length = pair_score_features(
        dest, donor, torch.tensor([[10.0, 10.0]]), dest_length, donor_length
    )
    torch.testing.assert_close(pair_length, torch.tensor([[[[2.0, 6.0], [8.0, 3.0]]]]))
    torch.testing.assert_close(features[..., -2:], torch.asinh((dest[:, :, None] - donor[:, None]) / pair_length))


def test_normalized_sdpa_matches_complete_reference_and_empty_mm_is_exact_zero():
    torch.manual_seed(31)
    route = SpatialRouteAttentionV2(16, 4, 2).eval()
    destination = torch.randn(2, 3, 16, requires_grad=True)
    donor = torch.randn(2, 4, 16, requires_grad=True)
    xq = torch.randn(2, 3, 2)
    xk = torch.randn(2, 4, 2)
    lq = torch.rand(2, 3, 2) + 0.1
    lk = torch.rand(2, 4, 2) + 0.1
    measure = torch.tensor([[0.3, 0.0, 0.5, 0.2], [0.1, 0.2, 0.3, 0.4]])
    mask = torch.ones((2, 4), dtype=torch.bool)
    destination_mask = torch.tensor([[True, True, True], [True, False, True]])
    frame = torch.tensor([[2.0, 3.0], [1.5, 2.5]])
    reference = torch.tensor([1.2, 0.7])
    actual, _ = route(
        destination, donor, xq, xk, lq, lk, measure, reference,
        mask, destination_mask, frame,
    )

    batch, nq, _ = destination.shape
    nk = donor.shape[1]
    q = route.query(route.query_norm(destination)).view(batch, nq, 4, 4).transpose(1, 2)
    kn = route.key_norm(donor)
    k = route.key(kn).view(batch, nk, 4, 4).transpose(1, 2)
    v = route.value(kn).view(batch, nk, 4, 4).transpose(1, 2)
    features, _ = pair_score_features(xq, xk, frame, lq, lk)
    geometry = route.geometry_score(features).permute(0, 3, 1, 2)
    allowed = mask[:, None, :].expand(batch, nq, nk) & (measure > 0)[:, None, :]
    allowed = allowed & destination_mask[:, :, None]
    attended, _weights = explicit_spatial_attention(
        q, k, v, allowed, geometry, measure, reference, route.temperature
    )
    expected = route.output(attended.transpose(1, 2).reshape(batch, nq, 16))
    expected = expected * (allowed.any(-1) & destination_mask)[:, :, None]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    loss = actual.square().sum()
    grad = torch.autograd.grad(loss, (destination, donor, route.raw_temperature))
    assert all(torch.isfinite(item).all() for item in grad)
    assert route.temperature.min() >= 1 and route.temperature.max() <= 32

    one = torch.randn(2, 1, 16, requires_grad=True)
    identity = torch.tensor([[[0.0, 0.0]], [[1.0, 1.0]]])
    length = torch.ones(2, 1, 2)
    self_route, _ = route(
        one, one, identity, identity, length, length,
        torch.ones(2, 1), torch.ones(2), torch.ones(2, 1, dtype=torch.bool),
        torch.ones(2, 1, dtype=torch.bool), torch.ones(2, 2), remove_self=True,
    )
    assert torch.equal(self_route, torch.zeros_like(self_route))
    self_route.sum().backward()
    assert torch.equal(one.grad, torch.zeros_like(one.grad))


def test_common_named_weights_match_and_final_w_couples_c_once():
    h6 = _core("A2-H6")
    direct = _core("A2-D")
    params_h6 = dict(h6.named_parameters())
    params_direct = dict(direct.named_parameters())
    common_names = set(params_h6).intersection(params_direct)
    assert "module_encoder.0.weight" in common_names
    assert "query_encoder.net.0.weight" in common_names
    assert "local_reader.relative_projection.weight" in common_names
    for name in common_names:
        assert params_h6[name].shape == params_direct[name].shape, name
        torch.testing.assert_close(params_h6[name], params_direct[name], rtol=0, atol=0)

    scene = _scene()
    prepared = h6.prepare(scene)
    assert prepared.module_gather_weights.shape == (2, 6, 3)
    assert prepared.environment_gather_weights.shape == (2, 6, 4)
    torch.testing.assert_close(prepared.module_gather_weights.sum(-1), torch.ones(2, 6))
    torch.testing.assert_close(prepared.environment_gather_weights.sum(-1), torch.ones(2, 6))
    module_value = scene.module_measure.new_tensor([[0.0, 0.0, 0.0], [0.5, 0.8, 1.2]])
    masked_scene = replace(scene, module_mask=torch.tensor([[False, False, False], [True, True, True]]))
    masked_org = h6.prepare(masked_scene)
    assert torch.equal(masked_org.module_gather_weights[0], torch.zeros_like(masked_org.module_gather_weights[0]))

    kappa, expected_c = h6.coefficient_bank(
        scene,
        prepared.module_states,
        prepared.edges,
        prepared.module_gather_weights,
        prepared.scene_summary,
        prepared.operating_state,
        prepared.edge_anchors,
        prepared.frame_scale,
    )
    torch.testing.assert_close(prepared.source_coefficients, expected_c)
    torch.testing.assert_close(
        prepared.source_coefficients,
        (prepared.module_gather_weights[..., None] * kappa).permute(0, 1, 3, 2),
    )
    assert prepared.source_coefficients.shape == (2, 6, 4, 3)


def test_measure_summary_stays_finite_and_refinement_invariant_across_extreme_scales():
    states = torch.tensor([[[1.0, -2.0], [5.0, 4.0]]], requires_grad=True)
    valid = torch.ones((1, 2), dtype=torch.bool)
    baseline_measure = torch.tensor([[0.25, 0.75]], requires_grad=True)
    baseline_mean, _ = ContentHypergraphCoreV2._measure_summary(states, baseline_measure, valid)
    baseline_grad = torch.autograd.grad(baseline_mean.sum(), baseline_measure)[0]

    for scale in (1.0e-30, 1.0e30):
        measure = (baseline_measure.detach() * scale).requires_grad_()
        mean, total = ContentHypergraphCoreV2._measure_summary(states, measure, valid)
        gradient = torch.autograd.grad(mean.sum(), measure)[0]
        torch.testing.assert_close(mean, baseline_mean, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(total / scale, torch.ones_like(total), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(gradient * scale, baseline_grad, atol=2e-6, rtol=2e-6)
        assert torch.isfinite(mean).all() and torch.isfinite(gradient).all()

    refined_states = torch.tensor([[[1.0, -2.0], [1.0, -2.0], [5.0, 4.0]]])
    refined_measure = torch.tensor([[0.125, 0.125, 0.75]])
    refined_valid = torch.ones((1, 3), dtype=torch.bool)
    refined_mean, _ = ContentHypergraphCoreV2._measure_summary(
        refined_states, refined_measure, refined_valid
    )
    torch.testing.assert_close(refined_mean, baseline_mean.detach(), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("architecture", ["A2-H6", "A2-D"])
def test_affine_kernel_apply_increment_prepared_cache_and_adjoint(architecture):
    core = _core(architecture)
    scene = _scene()
    query = _query(batch=2, count=5)
    organization = core.prepare(scene)
    controls = torch.tensor([[0.0, 0.8, 1.2], [0.3, 0.0, 2.0]])
    output = core.decode(organization, query, controls=controls, return_basis=True)
    kernels = core.export_source_kernels(organization, query)
    expected = torch.einsum("bqm,bm->bq", kernels, controls)
    torch.testing.assert_close(output.thermal_temperature, expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(core.apply_increment(organization, query, controls), expected)
    zero = core.decode(organization, query, controls=torch.zeros_like(controls))
    assert torch.equal(zero.thermal_temperature, torch.zeros_like(zero.thermal_temperature))

    tangent = torch.randn_like(controls)
    cotangent = torch.randn(2, query.num_queries)
    jvp = core.heat_jvp(organization, query, tangent)
    vjp = core.heat_vjp(organization, query, cotangent)
    torch.testing.assert_close((jvp * cotangent).sum(), (tangent * vjp).sum(), atol=3e-6, rtol=3e-5)

    prepared = core.prepare_affine_query(organization, query)
    retained = core.apply_prepared_affine(prepared, controls)
    torch.testing.assert_close(retained.thermal_temperature, expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(core.heat_vjp_prepared_affine(prepared, cotangent), vjp, atol=3e-6, rtol=3e-5)
    if architecture == "A2-D":
        assert retained.work_counts["direct_dense_reference_entries"] == 2 * 5 * 3


def test_batch_two_edge_gates_head_axes_and_bounded_query_trace():
    core = _core("A2-H6")
    organization = core.prepare(_scene())
    query = _query(batch=2, count=5)
    controls = torch.tensor([[0.3, 0.8, 0.2], [0.1, 0.7, 0.9]])
    gates = torch.tensor([[1.0, 0.0, 0.5, 1.0, 0.5, 0.2], [0.0, 1.0, 0.5, 0.2, 1.0, 0.3]])
    result = core.decode(
        organization,
        query,
        controls=controls,
        edge_gates=gates,
        return_trace=True,
        trace_request={
            "include": ["edge_access", "edge_messages", "edge_temperature_contributions", "local_pair_module_indices"],
            "query_indices": [1, 3],
            "max_elements": 500_000,
        },
    )
    trace = result.trace
    assert trace["edge_access"].shape == (2, 2, 4, 6)
    assert trace["edge_messages"].shape == (2, 2, 6, 32)
    assert trace["edge_temperature_contributions"].shape == (2, 2, 6, 4)
    torch.testing.assert_close(trace["trace_query_indices"], torch.tensor([1, 3]))
    assert torch.equal(trace["edge_access"][0, :, :, 1], torch.zeros_like(trace["edge_access"][0, :, :, 1]))
    assert torch.equal(trace["edge_access"][1, :, :, 0], torch.zeros_like(trace["edge_access"][1, :, :, 0]))

    with pytest.raises(ValueError, match="max_elements"):
        core.decode(
            organization,
            query,
            return_trace=True,
            trace_request={"include": ["edge_messages"], "max_elements": 1},
        )


def test_preparation_exposes_both_typed_gather_passes_and_edge_interaction_trace():
    core = _core("A2-H6")
    trace_names = [
        "module_weights_first",
        "environment_weights_first",
        "module_content_first",
        "environment_content_first",
        "module_content_second",
        "environment_content_second",
        "edge_edge_attention",
    ]
    organization = core.prepare(_scene(), trace_request={"include": trace_names})
    assert set(organization.trace) == set(trace_names)
    assert organization.trace["module_weights_first"].shape == (2, 6, 3)
    assert organization.trace["environment_weights_first"].shape == (2, 6, 4)
    assert organization.trace["module_content_first"].shape == (2, 6, 32)
    assert organization.trace["environment_content_second"].shape == (2, 6, 32)
    assert organization.trace["edge_edge_attention"].shape == (2, 4, 6, 6)


def test_a2d_response_coefficients_are_not_source_softmax_weighted_and_local_gate_zeroes_local():
    core = _core("A2-D")
    scene = _scene()
    query = _query(batch=2, count=5)
    organization = core.prepare(scene)
    query_states = core._query_states(organization, query)
    ref = core._module_measure_reference(scene, query_states.dtype)
    first = core.direct_reader(
        scene, query, query_states, organization.frame_scale, ref,
        organization.direct_source_keys, organization.direct_source_values,
    )
    changed_measure = scene.module_measure.clone()
    changed_measure[:, 0] *= 7.0
    changed_scene = replace(scene, module_measure=changed_measure)
    changed = core.direct_reader(
        changed_scene, query, query_states, organization.frame_scale, ref,
        organization.direct_source_keys, organization.direct_source_values,
    )
    torch.testing.assert_close(first[2], changed[2], atol=0, rtol=0)
    assert not torch.allclose(first[1], changed[1])

    controls = torch.ones(2, 3)
    all_local = core.decode(organization, query, controls=controls)
    no_local = core.decode(organization, query, controls=controls, local_gates=torch.zeros(2, 3))
    torch.testing.assert_close(no_local.thermal_temperature, core.apply_increment(
        organization, query, controls, local_gates=torch.zeros(2, 3)
    ))
    assert all_local.thermal_temperature.shape == no_local.thermal_temperature.shape


def test_prepared_state_invalidates_on_geometry_or_parameter_change():
    core = _core("A2-H6")
    scene = _scene()
    query = _query()
    organization = core.prepare(scene)
    core.validate_prepared_state(organization, query)
    scene.module_characteristic_lengths[0, 0, 0] += 0.001
    with pytest.raises(ValueError, match="Scene tensors changed"):
        core.validate_prepared_state(organization, query)
    fresh = core.prepare(scene)
    with torch.no_grad():
        next(core.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="parameters changed"):
        core.validate_prepared_state(fresh, query)


def test_prepared_affine_detects_local_gate_mutation_and_rank_zero_has_no_dead_modes():
    core = _core("A2-H6")
    organization = core.prepare(_scene())
    query = _query()
    local_gates = torch.ones(2, 3)
    prepared = core.prepare_affine_query(organization, query, local_gates=local_gates)
    with torch.no_grad():
        local_gates[0, 0] = 0.5
    with pytest.raises(ValueError, match="factors changed"):
        core.apply_prepared_affine(prepared, torch.ones(2, 3))

    for architecture in ("A2-H6", "A2-D"):
        rank_zero = _core(architecture, thermal_response_rank=0)
        names = [name for name, _ in rank_zero.named_parameters()]
        assert not any("thermal" in name for name in names)
        reader = rank_zero.edge_reader if architecture == "A2-H6" else rank_zero.direct_reader
        torch.testing.assert_close(
            reader.temperature,
            torch.full_like(reader.temperature, 8.0**0.5),
            atol=1e-6,
            rtol=1e-6,
        )


def test_typed_w_matches_explicit_geometry_measure_law_and_signed_coupling():
    core = _core("A2-H6")
    scene = _scene()
    organization = core.prepare(
        scene,
        trace_request={
            "include": [
                "scene_summary_initial",
                "module_weights_first",
                "environment_weights_first",
            ]
        },
    )
    bank = core.content_bank
    anchors, scales, normalized_anchors = bank._geometry(
        scene, organization.frame_origin, organization.frame_scale
    )
    initial_geometry = torch.cat(
        (
            bank._gamma_from_normalized(normalized_anchors),
            torch.log((scales / organization.frame_scale[:, None, :]).clamp_min(torch.finfo(scales.dtype).tiny)),
            organization.scene_summary[:, None, :].expand(-1, bank.edge_count, -1),
        ),
        dim=-1,
    )
    initial = bank.initial_norm(
        bank.edge_seeds[None] + bank.initial_projection(initial_geometry)
    )
    module_key = bank.module_key(bank.module_key_norm(organization.module_states))
    environment_key = bank.environment_key(bank.environment_key_norm(organization.environment_states))
    module_value = bank.module_value(organization.module_states)
    environment_value = bank.environment_value(organization.environment_states)
    first_query = bank._edge_query(initial)
    second_query = bank._edge_query(organization.mixed_content)

    expected_m_first, _ = _explicit_typed_gather(
        first_query, module_key, module_value, bank.module_geometry_score,
        anchors, scales, scene.module_coordinates, scene.module_characteristic_lengths,
        scene.module_measure, scene.module_measure_reference, scene.module_mask,
        organization.frame_scale, bank.gather_temperature,
    )
    expected_e_first, _ = _explicit_typed_gather(
        first_query, environment_key, environment_value, bank.environment_geometry_score,
        anchors, scales, scene.environment_coordinates, scene.environment_characteristic_lengths,
        scene.environment_measure, scene.environment_measure_reference, scene.environment_mask,
        organization.frame_scale, bank.gather_temperature,
    )
    expected_m_final, _ = _explicit_typed_gather(
        second_query, module_key, module_value, bank.module_geometry_score,
        anchors, scales, scene.module_coordinates, scene.module_characteristic_lengths,
        scene.module_measure, scene.module_measure_reference, scene.module_mask,
        organization.frame_scale, bank.gather_temperature,
    )
    expected_e_final, _ = _explicit_typed_gather(
        second_query, environment_key, environment_value, bank.environment_geometry_score,
        anchors, scales, scene.environment_coordinates, scene.environment_characteristic_lengths,
        scene.environment_measure, scene.environment_measure_reference, scene.environment_mask,
        organization.frame_scale, bank.gather_temperature,
    )
    torch.testing.assert_close(organization.trace["module_weights_first"], expected_m_first, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(organization.trace["environment_weights_first"], expected_e_first, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(organization.module_gather_weights, expected_m_final, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(organization.environment_gather_weights, expected_e_final, atol=2e-6, rtol=2e-6)

    # Reconstruct signed kappa from its projections; W is coupled once, last.
    source_delta = scene.module_coordinates[:, None, :, :] - organization.edge_anchors[:, :, None, :]
    source_geometry = relative_gamma(source_delta, organization.frame_scale)
    coeff = core.coefficient_bank
    source_hidden = coeff.edge_projection(organization.edges)[:, :, None, :]
    source_hidden = source_hidden + coeff.module_projection(organization.module_states)[:, None, :, :]
    source_hidden = source_hidden + coeff.geometry_projection(source_geometry)
    source_hidden = source_hidden + coeff.scene_projection(organization.scene_summary)[:, None, None, :]
    source_hidden = source_hidden + coeff.operating_projection(organization.operating_state)[:, None, None, :]
    kappa = coeff.output(F.silu(source_hidden))
    expected_c = (
        organization.module_gather_weights[..., None] * kappa
    ).permute(0, 1, 3, 2) * scene.module_mask[:, None, None, :]
    torch.testing.assert_close(organization.source_coefficients, expected_c, atol=1e-7, rtol=1e-6)
    assert (kappa < 0).any() and (kappa > 0).any()


@pytest.mark.parametrize("architecture", ["A2-H6", "A2-D"])
def test_reader_alpha_matches_explicit_headwise_softmax_and_work_counts(architecture):
    core = _core(architecture, context_dim=128, local_hidden_dim=128)
    scene, query = _scene(), _query(batch=2, count=5)
    organization = core.prepare(scene)
    controls = torch.ones(scene.batch_size, scene.num_modules)
    if architecture == "A2-H6":
        message_shapes = []
        message_hook = core.edge_reader.message_output.register_forward_hook(
            lambda _module, _inputs, output: message_shapes.append(tuple(output.shape))
        )
        thermal_head_shapes = []
        thermal_hooks = [
            head.register_forward_hook(
                lambda _module, _inputs, output: thermal_head_shapes.append(tuple(output.shape))
            )
            for head in core.edge_reader.thermal_heads
        ]
        actual = core.decode(
            organization, query, controls=controls,
            trace_request={"include": ["edge_base_access"]},
        )
        for hook in [message_hook, *thermal_hooks]:
            hook.remove()
        assert message_shapes == [(2, 5, 6, 128)]
        assert thermal_head_shapes == [(2, 5, 6, 1)] * 4

        query_states = core._query_states(organization, query)
        geometry, _relative = query_edge_features(
            query.coordinates, organization.edge_anchors, organization.frame_scale, organization.edge_scales
        )
        q = core.edge_reader.query_key(core.edge_reader.query_norm(query_states)).view(2, 5, 4, 32)
        k = organization.edge_keys.view(2, 6, 4, 32)
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)
        logits = torch.einsum("bqrd,bhrd->bqhr", q, k)
        logits = logits * core.edge_reader.temperature[None, None, None, :]
        logits = logits + core.edge_reader.geometry_score(geometry)
        allowed = torch.ones_like(logits, dtype=torch.bool)
        expected = torch.softmax(logits.permute(0, 1, 3, 2), dim=-1)
        torch.testing.assert_close(actual.trace["edge_base_access"], expected, atol=2e-6, rtol=2e-6)
        assert actual.work_counts["edge_rows"] == 2 * 5 * 6
        assert actual.work_counts["edge_message_rows"] == 2 * 5 * 6
        assert actual.work_counts["edge_feature_message_entries"] == 2 * 5 * 6 * 128
        assert actual.work_counts["edge_access_rows"] == 2 * 5 * 6 * 4
    else:
        actual = core.decode(
            organization, query,
            trace_request={"include": ["direct_source_access"]},
        )
        query_states = core._query_states(organization, query)
        geometry, _relative = query_source_features(
            query.coordinates, scene.module_coordinates, organization.frame_scale,
            scene.module_characteristic_lengths,
        )
        reader = core.direct_reader
        q = reader.query_key(reader.query_norm(query_states)).view(2, 5, 4, 32)
        k = organization.direct_source_keys.view(2, 3, 4, 32)
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)
        logits = torch.einsum("bqrd,bmrd->bqmr", q, k)
        logits = logits * reader.temperature[None, None, None, :]
        logits = logits + reader.geometry_score(geometry)
        valid = scene.module_mask & (scene.module_measure > 0)
        logits = logits + (
            torch.log(torch.where(valid, scene.module_measure, torch.ones_like(scene.module_measure)))
            - torch.log(scene.module_measure_reference[:, None])
        )[:, None, :, None]
        logits = logits.permute(0, 1, 3, 2).masked_fill(~valid[:, None, None, :], -torch.inf)
        expected = torch.softmax(logits, dim=-1).permute(0, 1, 3, 2)
        torch.testing.assert_close(actual.trace["direct_source_access"], expected, atol=2e-6, rtol=2e-6)
        assert actual.work_counts["edge_rows"] == 0
        assert actual.work_counts["direct_source_message_entries"] == 2 * 5 * 3 * 128
        # The signed dense source coefficients are constructed during decode,
        # even without controls; this cost is not hidden in the later sum.
        assert actual.work_counts["direct_dense_reference_entries"] == 2 * 5 * 3


def test_staged_route_dependency_order_and_live_raw_summary_reinjection():
    core = _core("A2-H6")
    scene = _scene()
    block = core.context_blocks[0]
    states_m = torch.randn(2, 3, core.context_dim)
    states_e = torch.randn(2, 4, core.context_dim)
    inputs = {}
    handles = []
    for route_name in ("mm", "em", "ee", "me"):
        route = getattr(block, route_name)
        handles.append(
            route.register_forward_pre_hook(
                lambda _module, args, name=route_name: inputs.__setitem__(name, (args[0], args[1]))
            )
        )
    ff_calls = {"module": 0, "environment": 0}
    handles.append(block.module_ff.register_forward_hook(lambda *_args: ff_calls.__setitem__("module", ff_calls["module"] + 1)))
    handles.append(block.environment_ff.register_forward_hook(lambda *_args: ff_calls.__setitem__("environment", ff_calls["environment"] + 1)))
    module_next, environment_next, trace = block(states_m, states_e, scene, scene.coordinate_scale)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(inputs["mm"][0], states_m, rtol=0, atol=0)
    torch.testing.assert_close(inputs["mm"][1], states_m, rtol=0, atol=0)
    torch.testing.assert_close(inputs["em"][0], states_e, rtol=0, atol=0)
    torch.testing.assert_close(inputs["em"][1], trace["module_after_mm"], rtol=0, atol=0)
    torch.testing.assert_close(inputs["ee"][0], inputs["ee"][1], rtol=0, atol=0)
    torch.testing.assert_close(inputs["ee"][0], trace["environment_after_em"], rtol=0, atol=0)
    torch.testing.assert_close(inputs["me"][0], trace["module_after_mm"], rtol=0, atol=0)
    torch.testing.assert_close(inputs["me"][1], trace["environment_after_ee"], rtol=0, atol=0)
    torch.testing.assert_close(module_next, trace["module_after_me"], rtol=0, atol=0)
    torch.testing.assert_close(environment_next, trace["environment_after_ee"], rtol=0, atol=0)
    assert ff_calls == {"module": 1, "environment": 1}

    calls = []
    original_summary = core._scene_summary

    def capture_summary(self, *args, **kwargs):
        output = original_summary(*args, **kwargs)
        calls.append((args[1], args[2], output))
        return output

    core._scene_summary = MethodType(capture_summary, core)
    organization = core.prepare(
        scene,
        trace_request={
            "include": [
                "scene_summary_initial",
                "cycle0/module_boundary", "cycle0/environment_boundary",
                "cycle0/module_after_me", "cycle0/environment_after_ee",
                "cycle1/module_boundary", "cycle1/environment_boundary",
                "cycle1/module_after_me", "cycle1/environment_after_ee",
            ]
        },
    )
    assert len(calls) == core.context_depth + 1
    torch.testing.assert_close(calls[0][0], organization.raw_module_states, rtol=0, atol=0)
    torch.testing.assert_close(calls[0][1], organization.raw_environment_states, rtol=0, atol=0)
    torch.testing.assert_close(organization.trace["scene_summary_initial"], calls[0][2], rtol=0, atol=0)
    assert not torch.equal(calls[0][2], calls[1][2])

    assert len({id(core.module_raw_projection), id(core.environment_raw_projection),
                id(core.module_scene_projection), id(core.environment_scene_projection)}) == 4
    for gate in (core.module_raw_gate, core.environment_raw_gate, core.module_scene_gate, core.environment_scene_gate):
        torch.testing.assert_close(gate.detach(), torch.tensor(0.1))
    mask_m = scene.module_mask[:, :, None].to(organization.module_states.dtype)
    mask_e = scene.environment_mask[:, :, None].to(organization.environment_states.dtype)
    expected_m0 = (
        organization.raw_module_states
        + core.module_raw_gate * core.module_raw_projection(organization.raw_module_states)
        + core.module_scene_gate * core.module_scene_projection(calls[0][2])[:, None]
    ) * mask_m
    expected_e0 = (
        organization.raw_environment_states
        + core.environment_raw_gate * core.environment_raw_projection(organization.raw_environment_states)
        + core.environment_scene_gate * core.environment_scene_projection(calls[0][2])[:, None]
    ) * mask_e
    torch.testing.assert_close(organization.trace["cycle0/module_boundary"], expected_m0, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(organization.trace["cycle0/environment_boundary"], expected_e0, atol=1e-6, rtol=1e-6)
    expected_m1 = (
        organization.trace["cycle0/module_after_me"]
        + core.module_raw_gate * core.module_raw_projection(organization.raw_module_states)
        + core.module_scene_gate * core.module_scene_projection(calls[1][2])[:, None]
    ) * mask_m
    expected_e1 = (
        organization.trace["cycle0/environment_after_ee"]
        + core.environment_raw_gate * core.environment_raw_projection(organization.raw_environment_states)
        + core.environment_scene_gate * core.environment_scene_projection(calls[1][2])[:, None]
    ) * mask_e
    torch.testing.assert_close(organization.trace["cycle1/module_boundary"], expected_m1, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(organization.trace["cycle1/environment_boundary"], expected_e1, atol=1e-6, rtol=1e-6)


def test_typed_summary_ignores_environment_token_duplication_at_fixed_measure_and_size():
    core = _core("A2-H6")
    scene = _scene()
    module_states = torch.randn(2, 3, core.context_dim)
    environment_states = torch.randn(2, 4, core.context_dim)
    operation = core._operating_state(scene, torch.float32)
    frame_scale = scene.coordinate_scale
    module_reference, environment_reference = core._measure_references(scene, torch.float32)
    baseline = core._scene_summary(
        scene, module_states, environment_states, operation, frame_scale,
        module_reference, environment_reference,
    )
    env = scene.environment_features
    coords = scene.environment_coordinates
    lengths = scene.environment_characteristic_lengths
    ids = scene.environment_ids
    mask = scene.environment_mask
    measure = scene.environment_measure
    duplicate_scene = replace(
        scene,
        environment_features=torch.cat((env[:, :1], env[:, :1], env[:, 1:]), dim=1),
        environment_coordinates=torch.cat((coords[:, :1], coords[:, :1], coords[:, 1:]), dim=1),
        environment_characteristic_lengths=torch.cat((lengths[:, :1], lengths[:, :1], lengths[:, 1:]), dim=1),
        environment_ids=torch.cat((ids[:, :1], ids[:, :1] + 10_000, ids[:, 1:]), dim=1),
        environment_mask=torch.cat((mask[:, :1], mask[:, :1], mask[:, 1:]), dim=1),
        environment_measure=torch.cat((measure[:, :1] / 2, measure[:, :1] / 2, measure[:, 1:]), dim=1),
    )
    duplicate_states = torch.cat(
        (environment_states[:, :1], environment_states[:, :1], environment_states[:, 1:]), dim=1
    )
    refined = core._scene_summary(
        duplicate_scene, module_states, duplicate_states, operation, frame_scale,
        module_reference, environment_reference,
    )
    torch.testing.assert_close(refined, baseline, atol=2e-7, rtol=2e-7)


def test_empty_typed_donors_and_extreme_measure_gather_backward():
    core = _core("A2-H6")
    scene = _scene()
    no_module_scene = replace(
        scene,
        module_mask=torch.zeros_like(scene.module_mask),
        module_measure=torch.zeros_like(scene.module_measure),
    )
    no_module = core.prepare(
        no_module_scene,
        trace_request={"include": ["module_content_first", "module_content_second"]},
    )
    assert torch.equal(no_module.module_gather_weights, torch.zeros_like(no_module.module_gather_weights))
    assert torch.equal(no_module.trace["module_content_first"], torch.zeros_like(no_module.trace["module_content_first"]))
    assert torch.equal(no_module.trace["module_content_second"], torch.zeros_like(no_module.trace["module_content_second"]))
    assert torch.equal(no_module.source_coefficients, torch.zeros_like(no_module.source_coefficients))

    no_environment_scene = replace(
        scene,
        environment_mask=torch.zeros_like(scene.environment_mask),
        environment_measure=torch.zeros_like(scene.environment_measure),
    )
    no_environment = core.prepare(
        no_environment_scene,
        trace_request={"include": ["environment_content_first", "environment_content_second"]},
    )
    assert torch.equal(no_environment.environment_gather_weights, torch.zeros_like(no_environment.environment_gather_weights))
    assert torch.equal(no_environment.trace["environment_content_first"], torch.zeros_like(no_environment.trace["environment_content_first"]))
    assert torch.equal(no_environment.trace["environment_content_second"], torch.zeros_like(no_environment.trace["environment_content_second"]))

    empty_query = torch.randn(2, 6, core.context_dim)
    empty_key = torch.empty(2, 0, core.context_dim)
    empty_weights, empty_content = _measure_weighted_edge_read(
        empty_query, empty_key, empty_key, core.content_bank.module_geometry_score,
        torch.empty(2, 6, 2), torch.empty(2, 6, 2), torch.empty(2, 0, 2),
        torch.empty(2, 0, 2), torch.empty(2, 0), torch.ones(2),
        torch.empty(2, 0, dtype=torch.bool), torch.ones(2, 2),
        core.content_bank.gather_temperature,
    )
    assert empty_weights.shape == (2, 6, 0)
    assert empty_content.shape == (2, 6, core.context_dim)
    assert torch.equal(empty_content, torch.zeros_like(empty_content))

    baseline = core.prepare(scene)
    for scale in (1.0e-30, 1.0e30):
        module_measure = (scene.module_measure * scale).detach().requires_grad_(True)
        environment_measure = (scene.environment_measure * scale).detach().requires_grad_(True)
        scaled_scene = replace(
            scene,
            module_measure=module_measure,
            environment_measure=environment_measure,
            module_measure_reference=torch.full_like(scene.module_measure_reference, scale),
            environment_measure_reference=torch.full_like(scene.environment_measure_reference, scale),
        )
        scaled = core.prepare(scaled_scene)
        torch.testing.assert_close(scaled.module_gather_weights, baseline.module_gather_weights, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(scaled.environment_gather_weights, baseline.environment_gather_weights, atol=3e-6, rtol=3e-6)
        objective = scaled.module_gather_weights.square().sum() + scaled.environment_gather_weights.square().sum()
        module_grad, environment_grad = torch.autograd.grad(objective, (module_measure, environment_measure))
        assert torch.isfinite(module_grad).all() and torch.isfinite(environment_grad).all()


@pytest.mark.parametrize("architecture", ["A2-H6", "A2-D"])
def test_work_count_aliases_and_apply_increment_optional_counts_share_one_read(architecture):
    core = _core(architecture)
    organization, query = core.prepare(_scene()), _query(batch=2, count=5)
    controls = torch.randn(2, 3)
    plain = core.apply_increment(organization, query, controls)
    reader = core.edge_reader if architecture == "A2-H6" else core.direct_reader
    calls = []
    handle = reader.message_output.register_forward_hook(
        lambda _module, _inputs, output: calls.append(tuple(output.shape))
    )
    counted, work = core.apply_increment(
        organization, query, controls, return_work_counts=True
    )
    handle.remove()
    assert len(calls) == 1
    assert calls[0] == (2, 5, 6 if architecture == "A2-H6" else 3, core.local_hidden_dim)
    torch.testing.assert_close(counted, plain, rtol=0, atol=0)
    assert work["field_head_rows"] == 0
    assert work["receiver_rows"] == 2 * 5
    if architecture == "A2-H6":
        assert work["edge_rows"] == work["edge_message_rows"] == 2 * 5 * 6
        assert work["edge_feature_message_entries"] == 2 * 5 * 6 * core.local_hidden_dim
    else:
        assert work["edge_rows"] == 0
        assert work["direct_dense_reference_entries"] == 2 * 5 * 3
    prepared_counts = organization.work_counts
    expected_routes = {
        "context_mm_rows": 2 * 2 * 3 * 3,
        "context_em_rows": 2 * 2 * 4 * 3,
        "context_ee_rows": 2 * 2 * 4 * 4,
        "context_me_rows": 2 * 2 * 3 * 4,
    }
    for key, value in expected_routes.items():
        assert prepared_counts[key] == value
    if architecture == "A2-H6":
        assert prepared_counts["content_gather_first_pass_rows"] == 2 * 6 * (3 + 4)
        assert prepared_counts["content_gather_second_pass_rows"] == 2 * 6 * (3 + 4)
        assert prepared_counts["edge_interaction_head_pair_rows"] == 2 * 4 * 6 * 6
