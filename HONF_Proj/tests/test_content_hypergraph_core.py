from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from honf_forward_core.content_hypergraph import (
    ContentHypergraphCore,
    QueryBatch,
    SceneBatch,
    explicit_attention,
)
from honf_forward_core.content_hypergraph.decoding import pack_local_pairs
from honf_forward_core.content_hypergraph.organization import _measure_weighted_pooling


def _scene() -> SceneBatch:
    return SceneBatch(
        module_features=torch.tensor([[[0.2, 0.4], [0.7, -0.3], [-0.1, 0.8]]]),
        environment_features=torch.tensor([[[0.1, 0.2, 0.3], [0.4, -0.2, 0.5], [-0.5, 0.7, 0.1], [0.8, 0.1, -0.4]]]),
        module_coordinates=torch.tensor([[[0.0, 0.0], [0.5, 0.2], [1.0, 0.4]]]),
        environment_coordinates=torch.tensor([[[-0.1, 0.3], [0.35, 0.1], [0.7, 0.5], [1.1, 0.2]]]),
        module_mask=torch.ones((1, 3), dtype=torch.bool),
        environment_mask=torch.ones((1, 4), dtype=torch.bool),
        module_measure=torch.tensor([[0.7, 1.1, 0.9]]),
        environment_measure=torch.tensor([[0.2, 0.4, 0.3, 0.5]]),
        module_ids=torch.tensor([[13, 29, 47]]),
        environment_ids=torch.tensor([[101, 107, 131, 149]]),
        local_radius=torch.full((1, 3), 0.32),
        operating_features=torch.tensor([[0.3, -0.2]]),
        scene_key="scene-A",
        normalizer_key="train-transform-v1",
        coordinate_frame_key="frame-A",
    )


def _query() -> QueryBatch:
    return QueryBatch(
        coordinates=torch.tensor([[[0.06, 0.03], [0.49, 0.19], [0.91, 0.37], [0.52, 0.22]]]),
        features=torch.tensor([[[0.1, 0.5], [-0.3, 0.7], [0.4, -0.1], [0.2, 0.6]]]),
        query_ids=torch.tensor([[3, 5, 8, 13]]),
        scene_key="scene-A",
        coordinate_frame_key="frame-A",
    )


def _core(architecture: str = "A-H6", **overrides) -> ContentHypergraphCore:
    kwargs = {
        "module_input_dim": 2,
        "environment_input_dim": 3,
        "coordinate_dim": 2,
        "query_input_dim": 2,
        "operating_input_dim": 2,
        "context_dim": 16,
        "num_heads": 4,
        "context_depth": 2,
        "num_hyperedges": 6,
        "thermal_response_rank": 4,
        "local_hidden_dim": 8,
        "architecture": architecture,
        "init_seed": 1947,
    }
    kwargs.update(overrides)
    core = ContentHypergraphCore(**kwargs)
    core.eval()
    return core


@pytest.mark.parametrize("edge_score", [-55.0, -100.0])
@pytest.mark.parametrize("measure_scale", [1.0, 1.0e-25])
def test_measure_weighted_pooling_is_stable_for_tiny_mass_and_matches_fp64(edge_score, measure_scale):
    scores = torch.tensor(
        [[[0.0, 0.0, edge_score], [0.0, 0.0, edge_score], [0.0, 0.0, edge_score]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    measure = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32) * measure_scale
    mask = torch.ones((1, 3), dtype=torch.bool)
    reference = torch.tensor([measure_scale], dtype=torch.float32)

    incidence, pooling, edge_mass, log_mass, available = _measure_weighted_pooling(
        scores, measure, mask, reference
    )
    values = torch.tensor([[-0.7, 0.2, 1.1]], dtype=torch.float32)
    loss = (pooling[:, 2] * values).sum() + log_mass[:, 2].sum()
    gradient = torch.autograd.grad(loss, scores)[0]

    score64 = scores.detach().to(torch.float64).requires_grad_(True)
    measure64 = measure.to(torch.float64)
    incidence64 = torch.softmax(score64, dim=-1)
    weighted64 = incidence64 * measure64[:, :, None]
    expected_pooling64 = (weighted64 / weighted64.sum(dim=1, keepdim=True)).transpose(1, 2)
    expected_log_mass64 = torch.log1p(weighted64.sum(dim=1) / reference.to(torch.float64)[:, None])
    expected_loss64 = (expected_pooling64[:, 2] * values.to(torch.float64)).sum() + expected_log_mass64[:, 2].sum()
    expected_gradient64 = torch.autograd.grad(expected_loss64, score64)[0]

    assert torch.isfinite(pooling).all()
    assert torch.isfinite(log_mass).all()
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(pooling.double(), expected_pooling64, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(log_mass.double(), expected_log_mass64, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(gradient.double(), expected_gradient64, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(pooling[0, 2], torch.tensor([1.0, 2.0, 3.0]) / 6.0)
    if measure_scale == 1.0:
        assert (edge_mass[:, 2] > 0).all()
    else:
        assert torch.equal(edge_mass[:, 2], torch.zeros_like(edge_mass[:, 2]))
    assert available[:, 2].all()
    assert (incidence[:, :, 2] > 0).all()


def test_measure_weighted_pooling_empty_type_has_exact_zero_summary_and_finite_gradients():
    scores = torch.randn((2, 3, 4), dtype=torch.float32, requires_grad=True)
    measure = torch.zeros((2, 3), dtype=torch.float32)
    mask = torch.tensor([[False, False, False], [True, True, True]])
    reference = torch.ones((2,), dtype=torch.float32)

    incidence, pooling, edge_mass, log_mass, available = _measure_weighted_pooling(
        scores, measure, mask, reference
    )
    loss = pooling.sum() + edge_mass.sum() + log_mass.sum()
    gradient = torch.autograd.grad(loss, scores)[0]

    assert torch.equal(incidence[0], torch.zeros_like(incidence[0]))
    torch.testing.assert_close(incidence[1].sum(dim=-1), torch.ones_like(incidence[1, :, 0]))
    assert torch.equal(pooling, torch.zeros_like(pooling))
    assert torch.equal(edge_mass, torch.zeros_like(edge_mass))
    assert torch.equal(log_mass, torch.zeros_like(log_mass))
    assert not available.any()
    assert torch.isfinite(gradient).all()
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_measure_weighted_pooling_preserves_normal_mass_forward_rounding():
    torch.manual_seed(812)
    scores = torch.randn(2, 17, 6, requires_grad=True)
    measure = torch.rand(2, 17) + 0.2
    mask = torch.rand(2, 17) > 0.2
    reference = torch.tensor([0.7, 1.3])
    incidence, pooling, mass, feature, _ = _measure_weighted_pooling(scores, measure, mask, reference)
    weighted = incidence.transpose(1, 2) * torch.where(mask, measure, 0)[:, None, :]
    expected_pooling = weighted / weighted.sum(-1)[:, :, None]
    expected_feature = torch.log1p(mass / reference[:, None])
    assert torch.equal(pooling, expected_pooling)
    assert torch.equal(feature, expected_feature)
    gradient = torch.autograd.grad((pooling.square()).sum() + feature.sum(), scores, create_graph=True)[0]
    second = torch.autograd.grad(gradient.square().sum(), scores)[0]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(second).all()


def test_measure_weighted_pooling_jvp_matches_analytic_fp64_sum():
    torch.manual_seed(813)
    scores = torch.randn(1, 5, 3)
    direction = torch.randn_like(scores)
    measure = torch.tensor([[0.2, 0.7, 1.1, 0.4, 1.3]])
    mask = torch.ones((1, 5), dtype=torch.bool)
    reference = torch.tensor([0.8])

    def actual(value):
        _, pooling, _, feature, _ = _measure_weighted_pooling(value, measure, mask, reference)
        return pooling, feature

    def expected(value):
        weighted = torch.softmax(value, -1).transpose(1, 2) * measure.double()[:, None, :]
        mass = weighted.sum(-1)
        return weighted / mass[:, :, None], torch.log1p(mass / reference.double()[:, None])

    _, tangent = torch.func.jvp(actual, (scores,), (direction,))
    _, reference_tangent = torch.func.jvp(expected, (scores.double(),), (direction.double(),))
    for observed, wanted in zip(tangent, reference_tangent, strict=True):
        torch.testing.assert_close(observed.double(), wanted, atol=1e-5, rtol=1e-4)


def test_explicit_attention_matches_sdpa_and_zeros_empty_rows():
    torch.manual_seed(4)
    query = torch.randn(2, 3, 4, 8, dtype=torch.float64)
    key = torch.randn(2, 3, 5, 8, dtype=torch.float64)
    value = torch.randn(2, 3, 5, 8, dtype=torch.float64)
    allowed = torch.rand(2, 4, 5) > 0.35
    allowed[0, 0] = False
    bias = torch.randn(2, 3, 4, 5, dtype=torch.float64) * 0.1

    actual = explicit_attention(query, key, value, allowed, bias)
    additive = bias.masked_fill(~allowed[:, None], -torch.inf)
    expected = F.scaled_dot_product_attention(query, key, value, additive)
    expected[0, :, 0] = 0.0

    assert torch.isfinite(actual).all()
    assert torch.equal(actual[0, :, 0], torch.zeros_like(actual[0, :, 0]))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_prepare_supports_one_module_empty_environment_and_inside_frame_seeds():
    core = _core()
    scene = SceneBatch(
        module_features=torch.ones((1, 1, 2)),
        environment_features=torch.empty((1, 0, 3)),
        module_coordinates=torch.tensor([[[2.0, -1.0]]]),
        environment_coordinates=torch.empty((1, 0, 2)),
        module_mask=torch.ones((1, 1), dtype=torch.bool),
        environment_mask=torch.empty((1, 0), dtype=torch.bool),
        module_measure=torch.ones((1, 1)),
        environment_measure=torch.empty((1, 0)),
        module_ids=torch.tensor([[19]]),
        environment_ids=torch.empty((1, 0), dtype=torch.long),
        local_radius=torch.ones((1, 1)),
        operating_features=torch.zeros((1, 2)),
    )
    organization = core.prepare(scene)
    query = QueryBatch(coordinates=torch.tensor([[[2.0, -1.0]]]), features=torch.zeros((1, 1, 2)))
    output = core.decode(organization, query)

    assert organization.environment_states.shape == (1, 0, 16)
    assert organization.module_incidence.shape == (1, 1, 6)
    normalized_anchors = (organization.edge_anchors - organization.frame_origin[:, None, :]) / organization.frame_scale[
        :, None, :
    ]
    assert torch.all(normalized_anchors >= 0.0)
    assert torch.all(normalized_anchors <= 1.0)
    assert torch.isfinite(output.nonlinear_field).all()


def test_measure_partition_invariance_and_query_chunking():
    core = _core()
    scene = _scene()
    organization = core.prepare(scene)
    query = _query()
    expected = core.decode(organization, query).nonlinear_field

    chunks = []
    for start, stop in ((0, 1), (1, 3), (3, 4)):
        tile = QueryBatch(
            coordinates=query.coordinates[:, start:stop],
            features=query.features[:, start:stop],
            query_ids=query.query_ids[:, start:stop],
            scene_key=query.scene_key,
            coordinate_frame_key=query.coordinate_frame_key,
        )
        chunks.append(core.decode(organization, tile).nonlinear_field)
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, rtol=1e-5, atol=1e-6)

    # Splitting an environmental quadrature token into identical half-measure
    # copies preserves attention and typed measure-weighted edge content.
    env_features = torch.cat((scene.environment_features, scene.environment_features[:, 1:2]), dim=1)
    env_coordinates = torch.cat((scene.environment_coordinates, scene.environment_coordinates[:, 1:2]), dim=1)
    env_mask = torch.cat((scene.environment_mask, scene.environment_mask[:, 1:2]), dim=1)
    env_measure = torch.cat(
        (
            scene.environment_measure[:, :1],
            scene.environment_measure[:, 1:2] * 0.5,
            scene.environment_measure[:, 2:],
            scene.environment_measure[:, 1:2] * 0.5,
        ),
        dim=1,
    )
    env_ids = torch.cat((scene.environment_ids, torch.tensor([[199]])), dim=1)
    split_scene = replace(
        scene,
        environment_features=env_features,
        environment_coordinates=env_coordinates,
        environment_mask=env_mask,
        environment_measure=env_measure,
        environment_ids=env_ids,
    )
    split = core.decode(core.prepare(split_scene), query).nonlinear_field
    torch.testing.assert_close(split, expected, rtol=2e-5, atol=2e-6)


def test_h6_factorized_jvp_vjp_and_flow_control_separation():
    core = _core()
    organization = core.prepare(_scene())
    query = _query()
    controls = torch.tensor([[0.3, 0.8, 0.1]])
    tangent = torch.tensor([[-0.2, 0.1, 0.4]])
    cotangent = torch.tensor([[0.2, -0.1, 0.7, 0.3]])

    output = core.decode(organization, query, controls, return_basis=True)
    second = core.decode(organization, query, 2.0 * controls)
    kernels = core.export_source_kernels(organization, query, accumulation_dtype=torch.float64)
    expected_temperature = torch.einsum("bqm,bm->bq", kernels, controls.double())
    expected_jvp = torch.einsum("bqm,bm->bq", kernels, tangent.double())
    expected_vjp = torch.einsum("bq,bqm->bm", cotangent.double(), kernels)

    torch.testing.assert_close(output.thermal_temperature.double(), expected_temperature)
    torch.testing.assert_close(
        core.apply_increment(organization, query, tangent, accumulation_dtype=torch.float64),
        expected_jvp,
    )
    torch.testing.assert_close(
        core.heat_jvp(organization, query, tangent, accumulation_dtype=torch.float64),
        expected_jvp,
    )
    torch.testing.assert_close(
        core.heat_vjp(organization, query, cotangent, accumulation_dtype=torch.float64),
        expected_vjp,
    )
    torch.testing.assert_close(
        core.decode(organization, query, torch.zeros_like(controls)).thermal_temperature, torch.zeros((1, 4))
    )
    torch.testing.assert_close(second.nonlinear_field, output.nonlinear_field)
    torch.testing.assert_close(second.thermal_temperature, 2.0 * output.thermal_temperature)
    assert output.thermal_basis.shape == (1, 4, 6, 4)
    assert output.trace is None
    assert output.work_counts["edge_rows"] == 24
    assert output.work_counts["local_candidate_rows"] == 12

    traced = core.decode(organization, query, controls, return_trace=True)
    restored = core.decode(organization, query, controls, edge_gates=torch.ones((1, 6)))
    torch.testing.assert_close(restored.nonlinear_field, output.nonlinear_field)
    torch.testing.assert_close(restored.thermal_temperature, output.thermal_temperature)
    torch.testing.assert_close(traced.trace["edge_access"], traced.trace["edge_used_access"])
    assert traced.trace is not None
    assert traced.trace["edge_messages"].shape == (1, 4, 6, 8)

    gates = torch.ones((1, 6))
    gates[:, 0] = 0.0
    removed = core.decode(organization, query, controls, edge_gates=gates, return_trace=True)
    access = removed.trace["edge_used_access"]
    assert torch.equal(access[:, :, 0], torch.zeros_like(access[:, :, 0]))
    torch.testing.assert_close(access[:, :, 1:], traced.trace["edge_access"][:, :, 1:])
    removed_kernels = core.export_source_kernels(organization, query, edge_gates=gates)
    torch.testing.assert_close(
        removed.thermal_temperature,
        torch.einsum("bqm,bm->bq", removed_kernels, controls),
    )

    uniform = core.decode(organization, query, return_trace=True, uniform_access=True)
    torch.testing.assert_close(uniform.trace["edge_base_access"], torch.full((1, 4, 6), 1.0 / 6.0))
    assert not torch.allclose(uniform.trace["edge_access"], uniform.trace["edge_base_access"])


def test_a_d_direct_source_control_matches_its_opt_in_kernel_export():
    core = _core("A-D")
    organization = core.prepare(_scene())
    query = _query()
    controls = torch.tensor([[0.3, 0.8, 0.1]])
    output = core.decode(organization, query, controls)
    kernels = core.export_source_kernels(organization, query)

    torch.testing.assert_close(
        output.thermal_temperature,
        torch.einsum("bqm,bm->bq", kernels, controls),
    )
    assert organization.num_hyperedges == 0
    assert output.work_counts["edge_rows"] == 0
    assert output.work_counts["direct_source_rows"] == 12
    assert kernels.shape == (1, 4, 3)
    (output.nonlinear_field.square().sum() + output.thermal_temperature.square().sum()).backward()
    assert all(parameter.grad is not None for parameter in core.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in core.parameters())
    with pytest.raises(ValueError, match="only available for A-H6"):
        core.decode(organization, query, edge_gates=torch.ones((1, 6)))


def test_packed_local_rows_match_fallback_and_keep_weight_gradients():
    core = _core()
    scene = _scene()
    organization = core.prepare(scene)
    query = _query()
    fallback = core.decode(organization, query, torch.tensor([[0.3, 0.8, 0.1]]))
    selected = pack_local_pairs(scene, query, organization.frame_scale)
    live_weights = selected.support_weight.detach().clone().requires_grad_(True)
    packed_query = QueryBatch(
        coordinates=query.coordinates,
        features=query.features,
        query_ids=query.query_ids,
        local_batch_indices=selected.batch_index,
        local_query_indices=selected.query_index,
        local_module_indices=selected.module_index,
        local_weights=live_weights,
        scene_key=query.scene_key,
        coordinate_frame_key=query.coordinate_frame_key,
    )
    packed = core.decode(organization, packed_query, torch.tensor([[0.3, 0.8, 0.1]]))
    torch.testing.assert_close(packed.nonlinear_field, fallback.nonlinear_field)
    torch.testing.assert_close(packed.thermal_temperature, fallback.thermal_temperature)
    assert packed.work_counts["local_candidate_rows"] == 0
    packed.thermal_temperature.sum().backward()
    assert live_weights.grad is not None
    assert torch.isfinite(live_weights.grad).all()


def test_shared_named_initialization_and_ad_has_no_dormant_hypergraph_bank():
    direct = _core("A-D")
    organized = _core("A-H6")
    direct_state = direct.state_dict()
    organized_state = organized.state_dict()
    shared = set(direct_state) & set(organized_state)
    assert shared
    for name in shared:
        assert direct_state[name].shape == organized_state[name].shape
        assert torch.equal(direct_state[name], organized_state[name]), name
    assert not any(name.startswith(("organizer.", "edge_reader.", "coefficient_bank.")) for name in direct_state)
    assert direct.organizer is None
    assert direct.edge_reader is None
    assert direct.coefficient_bank is None
    assert direct.context_blocks[-1].em is None
    assert direct.context_blocks[-1].environment_ff is None
    direct_organization = direct.prepare(_scene())
    h6_organization = organized.prepare(_scene())
    assert torch.equal(direct_organization.module_states, h6_organization.module_states)
    assert direct.final_environment_update is False
    assert organized.final_environment_update is True

    wind = _core("A-H6", thermal_response_rank=0)
    assert wind.coefficient_bank is None
    assert wind.local_reader.thermal_output is None
    assert wind.edge_reader.thermal_basis is None


def test_default_architecture_config_is_standalone_and_json_safe():
    configured = ContentHypergraphCore(2, 3, 3)
    config = configured.get_config()
    assert config["context_dim"] == 128
    assert config["num_heads"] == 4
    assert config["context_depth"] == 2
    assert config["architecture"] == "A-H6"
    assert config["final_environment_update"] is True
    assert json.loads(json.dumps(config)) == config
    reconstructed = ContentHypergraphCore.from_config(config)
    assert reconstructed.get_config() == config
    for key, value in configured.state_dict().items():
        assert torch.equal(value, reconstructed.state_dict()[key]), key


def test_all_h6_parameters_receive_finite_gradients():
    core = _core()
    core.train()
    organization = core.prepare(_scene())
    output = core.decode(organization, _query(), torch.tensor([[0.3, 0.8, 0.1]]))
    loss = output.nonlinear_field.square().sum() + output.thermal_temperature.square().sum()
    loss.backward()

    missing = [name for name, parameter in core.named_parameters() if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in core.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert missing == []
    assert nonfinite == []


def test_invalid_measure_radius_and_stale_prepared_ownership_are_rejected():
    core = _core()
    scene = _scene()
    with pytest.raises(ValueError, match="nonnegative"):
        core.prepare(replace(scene, module_measure=torch.tensor([[0.7, -0.1, 0.9]])))
    with pytest.raises(ValueError, match="positive"):
        core.prepare(replace(scene, local_radius=torch.tensor([[0.32, 0.0, 0.32]])))
    with pytest.raises(ValueError, match="coordinate_scale"):
        core.prepare(
            replace(
                scene,
                coordinate_origin=torch.zeros((1, 2)),
                coordinate_scale=torch.tensor([[1.0, 0.0]]),
            )
        )

    organization = core.prepare(scene)
    with torch.no_grad():
        next(core.parameters()).add_(0.001)
    with pytest.raises(ValueError, match="parameters changed"):
        core.decode(organization, _query())


def test_model_buffer_and_scene_identity_staleness_are_rejected():
    core = _core()
    scene = _scene()
    organization = core.prepare(scene)
    with torch.no_grad():
        core.organizer.reference_anchor_seeds[0, 0].add_(0.001)
    with pytest.raises(ValueError, match="buffers changed"):
        core.decode(organization, _query())

    core = _core()
    scene = _scene()
    organization = core.prepare(scene)
    object.__setattr__(scene, "normalizer_key", "different-train-transform")
    with pytest.raises(ValueError, match="identity or preprocessing"):
        core.decode(organization, _query())
