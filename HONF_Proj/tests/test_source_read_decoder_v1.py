from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from honf_forward_core.content_hypergraph import (
    ContentHypergraphCoreV2,
    QueryBatch,
    SceneBatch,
)
from honf_forward_core.content_hypergraph.source_read_response import SourceReadAffineFactorsV1


def _scene() -> SceneBatch:
    return SceneBatch(
        module_features=torch.tensor([[[0.2, 0.4], [0.7, -0.3], [-0.1, 0.8]]]),
        environment_features=torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, -0.2, 0.5], [-0.5, 0.7, 0.1], [0.8, 0.1, -0.4]]]
        ),
        module_coordinates=torch.tensor([[[0.0, 0.0], [0.45, 0.2], [0.9, 0.4]]]),
        environment_coordinates=torch.tensor([[[-0.1, 0.3], [0.35, 0.1], [0.7, 0.5], [1.1, 0.2]]]),
        module_mask=torch.tensor([[True, False, True]]),
        environment_mask=torch.ones((1, 4), dtype=torch.bool),
        module_measure=torch.tensor([[0.7, 0.0, 1.1]]),
        environment_measure=torch.tensor([[0.2, 0.4, 0.3, 0.5]]),
        module_ids=torch.tensor([[13, 29, 47]]),
        environment_ids=torch.tensor([[101, 107, 131, 149]]),
        local_radius=torch.full((1, 3), 0.7),
        operating_features=torch.tensor([[0.3, -0.2]]),
        coordinate_origin=torch.tensor([[-0.1, 0.0]]),
        coordinate_scale=torch.tensor([[1.2, 0.6]]),
        module_measure_reference=torch.ones(1),
        environment_measure_reference=torch.ones(1),
        module_characteristic_lengths=torch.tensor([[[0.2, 0.12], [0.18, 0.10], [0.25, 0.14]]]),
        environment_characteristic_lengths=torch.full((1, 4, 2), 0.1),
        scene_key="source-read-scene",
        normalizer_key="source-read-transform",
        coordinate_frame_key="source-read-frame",
    )


def _query() -> QueryBatch:
    return QueryBatch(
        coordinates=torch.tensor([[[0.06, 0.03], [0.49, 0.19], [0.91, 0.37], [0.52, 0.22]]]),
        features=torch.tensor([[[0.1, 0.5], [-0.3, 0.7], [0.4, -0.1], [0.2, 0.6]]]),
        query_mask=torch.tensor([[True, True, True, False]]),
        query_ids=torch.arange(4)[None],
        scene_key="source-read-scene",
        coordinate_frame_key="source-read-frame",
        receiver_key="source-read-panel",
    )


def _core(**overrides) -> ContentHypergraphCoreV2:
    config = dict(
        module_input_dim=2,
        environment_input_dim=3,
        coordinate_dim=2,
        query_input_dim=2,
        operating_input_dim=2,
        output_dim=3,
        architecture="A2-D",
        context_dim=128,
        num_heads=4,
        context_depth=1,
        num_hyperedges=6,
        thermal_response_rank=4,
        local_hidden_dim=128,
        dropout=0.0,
        init_seed=1947,
    )
    config.update(overrides)
    return ContentHypergraphCoreV2(**config)


def _floating_copy(value: torch.Tensor, *, requires_grad: bool = False) -> torch.Tensor:
    result = value.detach().clone().to(torch.float64)
    if requires_grad:
        result.requires_grad_()
    return result


def _double_scene(*, present_zero_measure: bool = False, requires_grad: bool = False) -> SceneBatch:
    scene = _scene()
    updates = {}
    for item in fields(scene):
        value = getattr(scene, item.name)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            updates[item.name] = _floating_copy(
                value, requires_grad=requires_grad and item.name in {"module_features", "module_coordinates"}
            )
    if present_zero_measure:
        updates["module_mask"] = torch.ones_like(scene.module_mask)
    return replace(scene, **updates)


def _double_query(*, requires_grad: bool = False) -> QueryBatch:
    query = _query()
    updates = {"coordinates": _floating_copy(query.coordinates, requires_grad=requires_grad)}
    updates["features"] = _floating_copy(query.features, requires_grad=requires_grad)
    return replace(query, **updates)


def _five_source_scene() -> SceneBatch:
    scene = _double_scene(present_zero_measure=True)
    return replace(
        scene,
        module_features=torch.cat(
            (scene.module_features, torch.tensor([[[0.3, -0.2], [-0.4, 0.6]]], dtype=torch.float64)), dim=1
        ),
        module_coordinates=torch.cat(
            (scene.module_coordinates, torch.tensor([[[3.0, 3.0], [4.0, 4.0]]], dtype=torch.float64)), dim=1
        ),
        module_mask=torch.ones((1, 5), dtype=torch.bool),
        module_measure=torch.cat((scene.module_measure, torch.tensor([[0.8, 0.9]], dtype=torch.float64)), dim=1),
        module_ids=torch.tensor([[13, 29, 47, 61, 73]]),
        local_radius=torch.full((1, 5), 0.7, dtype=torch.float64),
        module_characteristic_lengths=torch.cat(
            (scene.module_characteristic_lengths,
             torch.tensor([[[0.2, 0.12], [0.25, 0.14]]], dtype=torch.float64)), dim=1
        ),
    )


def _core_with_controller(
    policy: str,
    *,
    executor: str = "packed",
    threshold_gain: float = 0.65,
    scene: SceneBatch | None = None,
) -> tuple[ContentHypergraphCoreV2, SceneBatch, QueryBatch]:
    scene = _double_scene(present_zero_measure=True) if scene is None else scene
    query = _double_query()
    model = _core(source_read_policy=policy, source_read_executor=executor).double().eval()
    model.set_source_read_training_epoch(600)
    organization = model.prepare(scene)
    query_states = model._query_states(organization, query)
    from honf_forward_core.content_hypergraph.source_read_sparse import original_source_scores_v1

    descriptor, _alpha = original_source_scores_v1(
        model.direct_reader,
        scene,
        query,
        query_states,
        organization.frame_scale,
        model._module_measure_reference(scene, query_states.dtype),
        organization.direct_source_keys,
    )
    controller = model.direct_reader.source_read
    assert controller is not None
    score = controller.utility_scores(
        organization.source_read_source_projection,
        controller.project_query_states(query_states),
        descriptor,
    ).sum(dim=-1)
    valid = query.query_mask[:, :, None] & scene.module_mask[:, None, :]
    maximum = float(score.detach()[valid.expand_as(score)].max())
    threshold = maximum / float(threshold_gain)
    controller.install_calibration_(
        thresholds=(threshold, threshold * 1.4),
        context_scales=(1.0, 1.0, 1.0, 1.0),
        heating_envelope=2.5,
    )
    controller.mark_trained_()
    controller.freeze_metadata_()
    return model, scene, query


def _decode(model, scene, query, *, executor: str, controls: torch.Tensor, mode: str = "fitted"):
    with model.source_read_override_v1(mode, executor=executor, operating_point=0):
        organization = model.prepare(scene)
        output = model.decode(
            organization,
            query,
            controls=controls,
            trace_request={"include": ("source_read_gates", "source_read_support"), "max_elements": 4096},
        )
        kernel = model.export_source_kernels(organization, query, accumulation_dtype=torch.float64)
    return organization, output, kernel


def _compare_parameter_gradients(left: ContentHypergraphCoreV2, right: ContentHypergraphCoreV2) -> None:
    left_parameters, right_parameters = dict(left.named_parameters()), dict(right.named_parameters())
    assert left_parameters.keys() == right_parameters.keys()
    for name in left_parameters:
        lgrad, rgrad = left_parameters[name].grad, right_parameters[name].grad
        assert (lgrad is None) == (rgrad is None), name
        if lgrad is not None:
            torch.testing.assert_close(lgrad, rgrad, atol=2e-10, rtol=2e-9, msg=name)


@pytest.mark.parametrize("policy", ("utility", "completion"))
def test_selected_executors_match_gates_fields_thermal_k_and_fp64_gradients(policy: str) -> None:
    controls_value = torch.tensor([[0.35, -0.4, 0.25]], dtype=torch.float64)
    models = []
    outputs = []
    kernels = []
    scenes = []
    queries = []
    for executor in ("packed", "dense_masked", "packet"):
        scene = _double_scene(present_zero_measure=True, requires_grad=True)
        query = _double_query(requires_grad=True)
        model, _, _ = _core_with_controller(policy, executor=executor, scene=scene)
        controls = controls_value.clone().requires_grad_()
        with model.source_read_override_v1("fitted", executor=executor, operating_point=0):
            organization = model.prepare(scene)
            output = model.decode(
                organization,
                query,
                controls=controls,
                trace_request={"include": ("source_read_gates", "source_read_support"), "max_elements": 4096},
            )
            kernel = model.export_source_kernels(organization, query, accumulation_dtype=torch.float64)
        loss = output.nonlinear_field.square().sum() + 0.31 * output.thermal_temperature.square().sum()
        loss.backward()
        models.append(model)
        outputs.append(output)
        kernels.append(kernel.detach())
        scenes.append(scene)
        queries.append(query)

    reference = outputs[0]
    assert reference.trace is not None
    gates = reference.trace["source_read_gates"]
    support = reference.trace["source_read_support"]
    assert torch.any((gates > 0) & (gates < 1))
    assert torch.any(~support & torch.tensor([[[True, True, True]] * 4]))
    assert float(gates[0, 1, 1]) == 1.0  # present zero-measure source is geometrically protected
    assert int(reference.work_counts["source_read_fine_selected_rows"]) > 0
    assert int(reference.work_counts["source_read_fine_selected_rows"]) < 4 * 3

    for output, kernel in zip(outputs[1:], kernels[1:], strict=True):
        assert output.trace is not None
        torch.testing.assert_close(output.trace["source_read_gates"], gates, atol=0.0, rtol=0.0)
        assert torch.equal(output.trace["source_read_support"], support)
        torch.testing.assert_close(output.nonlinear_field, reference.nonlinear_field, atol=1e-11, rtol=1e-10)
        torch.testing.assert_close(output.thermal_temperature, reference.thermal_temperature, atol=1e-11, rtol=1e-10)
        torch.testing.assert_close(kernel, kernels[0], atol=2e-11, rtol=2e-10)

    # A present zero-measure source remains a signed Thermal column although
    # it is absent from the original measure-weighted information attention.
    full_model = models[0]
    with full_model.source_read_override_v1("full", executor="packed"):
        full_organization = full_model.prepare(scenes[0])
        full_kernel = full_model.export_source_kernels(full_organization, queries[0], accumulation_dtype=torch.float64)
    torch.testing.assert_close(kernels[0][0, 1, 1], full_kernel[0, 1, 1], atol=2e-11, rtol=2e-10)
    assert kernels[0][0, 1, 1].abs() > 0

    for model, scene, query in zip(models[1:], scenes[1:], queries[1:], strict=True):
        _compare_parameter_gradients(models[0], model)
        torch.testing.assert_close(
            scenes[0].module_coordinates.grad, scene.module_coordinates.grad, atol=2e-10, rtol=2e-9
        )
        torch.testing.assert_close(queries[0].coordinates.grad, query.coordinates.grad, atol=2e-10, rtol=2e-9)
        torch.testing.assert_close(scenes[0].module_features.grad, scene.module_features.grad, atol=2e-10, rtol=2e-9)


@pytest.mark.parametrize("policy", ("utility", "completion"))
def test_all_one_gates_recover_the_legacy_direct_reader_exactly(policy: str) -> None:
    model, scene, query = _core_with_controller(policy, threshold_gain=1.0e6)
    controls = torch.tensor([[0.3, -0.2, 0.4]], dtype=torch.float64)
    with model.source_read_override_v1("fitted", executor="packed"):
        organization = model.prepare(scene)
        selected = model.decode(organization, query, controls=controls, trace_request={
            "include": ("source_read_gates",), "max_elements": 4096,
        })
        assert selected.trace is not None
        valid = query.query_mask[:, :, None] & scene.module_mask[:, None, :]
        assert torch.equal(selected.trace["source_read_gates"][valid], torch.ones_like(selected.trace["source_read_gates"][valid]))
        assert selected.work_counts["source_read_source_full_fastpath"] == 1
    with model.source_read_override_v1("full", executor="packed"):
        organization = model.prepare(scene)
        legacy = model.decode(organization, query, controls=controls)
    assert torch.equal(selected.nonlinear_field, legacy.nonlinear_field)
    assert torch.equal(selected.thermal_temperature, legacy.thermal_temperature)


@pytest.mark.parametrize("policy", ("utility", "completion"))
@pytest.mark.parametrize("mode", ("nearest_fitted", "directional_fitted", "shuffle_fitted"))
def test_geometry_controls_match_fitted_counts_and_preserve_completion(policy: str, mode: str) -> None:
    model, scene, query = _core_with_controller(policy, scene=_five_source_scene(), threshold_gain=1.1)
    controls = torch.full((1, 5), 0.2, dtype=torch.float64)
    with model.source_read_override_v1("fitted", executor="packed", operating_point=0):
        fitted = model.decode(model.prepare(scene), query, controls=controls, trace_request={
            "include": ("source_read_support", "direct_source_access"), "max_elements": 4096,
        })
    with model.source_read_override_v1(mode, executor="packed", operating_point=0):
        organization = model.prepare(scene)
        geometric = model.decode(organization, query, controls=controls, trace_request={
            "include": ("source_read_support", "source_read_gates", "direct_source_access"),
            "max_elements": 4096,
        })
        factors = model.prepare_affine_query(organization, query)
        assert isinstance(factors.direct_coefficients, SourceReadAffineFactorsV1)
        assert factors.direct_coefficients.has_completion == (policy == "completion")
        with pytest.raises(ValueError, match="envelope"):
            model.decode(organization, query, controls=controls * 100)
    torch.testing.assert_close(
        geometric.trace["source_read_support"].sum(-1), fitted.trace["source_read_support"].sum(-1),
        atol=0, rtol=0,
    )
    torch.testing.assert_close(geometric.trace["direct_source_access"], fitted.trace["direct_source_access"],
                               atol=0, rtol=0)
    assert torch.equal(geometric.trace["source_read_gates"], geometric.trace["source_read_support"].double())
    assert geometric.work_counts["source_read_source_utility_score_rows"] == 4 * 5


@pytest.mark.parametrize("policy", ("utility", "completion"))
def test_heat_independent_plan_and_prepared_affine_jvp_vjp(policy: str) -> None:
    model, scene, query = _core_with_controller(policy)
    with model.source_read_override_v1("fitted", executor="packed"):
        organization = model.prepare(scene)
        h0 = torch.tensor([[0.3, -0.4, 0.2]], dtype=torch.float64)
        h1 = torch.tensor([[-0.2, 0.25, 0.35]], dtype=torch.float64)
        out0 = model.decode(organization, query, controls=h0, trace_request={
            "include": ("source_read_gates", "source_read_support"), "max_elements": 4096,
        })
        out1 = model.decode(organization, query, controls=h1, trace_request={
            "include": ("source_read_gates", "source_read_support"), "max_elements": 4096,
        })
        outsum = model.decode(organization, query, controls=h0 + h1)
        torch.testing.assert_close(outsum.thermal_temperature, out0.thermal_temperature + out1.thermal_temperature,
                                   atol=1e-11, rtol=1e-10)
        torch.testing.assert_close(out0.trace["source_read_gates"], out1.trace["source_read_gates"], atol=0, rtol=0)
        assert torch.equal(out0.trace["source_read_support"], out1.trace["source_read_support"])

        retained = model.prepare_affine_query(organization, query)
        assert isinstance(retained.direct_coefficients, SourceReadAffineFactorsV1)
        applied = model.apply_prepared_affine(retained, h0)
        torch.testing.assert_close(applied.thermal_temperature, out0.thermal_temperature, atol=1e-11, rtol=1e-10)
        direction = torch.tensor([[0.11, -0.07, 0.16]], dtype=torch.float64)
        cotangent = torch.tensor([[0.3, -0.2, 0.5, 0.0]], dtype=torch.float64)
        _, jvp = torch.autograd.functional.jvp(
            lambda heat: model.apply_prepared_affine(retained, heat).thermal_temperature,
            (h0,),
            (direction,),
        )
        vjp = model.heat_vjp_prepared_affine(retained, cotangent, accumulation_dtype=torch.float64)
        torch.testing.assert_close((jvp * cotangent).sum(), (direction * vjp).sum(), atol=1e-11, rtol=1e-10)
        torch.testing.assert_close(
            model.apply_prepared_affine(retained, direction).thermal_temperature,
            jvp,
            atol=1e-11,
            rtol=1e-10,
        )


@pytest.mark.parametrize("policy", ("utility", "completion"))
def test_empty_selected_set_is_finite_and_packet_padding_is_accounted(policy: str) -> None:
    scene = _double_scene(present_zero_measure=True)
    model, scene, query = _core_with_controller(policy, threshold_gain=0.2, scene=scene)
    far_query = replace(
        query,
        coordinates=torch.full_like(query.coordinates, 10.0),
        receiver_key="far-query",
    )
    with model.source_read_override_v1("fitted", executor="packet"):
        organization = model.prepare(scene)
        empty = model.decode(organization, far_query, controls=torch.zeros((1, 3), dtype=torch.float64), trace_request={
            "include": ("source_read_support",), "max_elements": 4096,
        })
    assert empty.trace is not None
    assert not empty.trace["source_read_support"].any()
    assert empty.work_counts["source_read_fine_selected_rows"] == 0
    assert empty.work_counts["source_read_fine_executed_rows"] == 0
    assert empty.work_counts["source_read_fine_padding_rows"] == 0
    assert torch.isfinite(empty.nonlinear_field).all()
    assert torch.isfinite(empty.thermal_temperature).all()

    packet_model, packet_scene, packet_query = _core_with_controller(
        policy, executor="packet", scene=_five_source_scene()
    )
    with packet_model.source_read_override_v1("fitted", executor="packet"):
        organization = packet_model.prepare(packet_scene)
        packet = packet_model.decode(organization, packet_query, controls=torch.zeros((1, 5), dtype=torch.float64))
    selected = packet.work_counts["source_read_fine_selected_rows"]
    executed = packet.work_counts["source_read_fine_executed_rows"]
    padding = packet.work_counts["source_read_fine_padding_rows"]
    assert executed == selected + padding
    assert executed >= selected
    assert padding > 0
    assert torch.isfinite(packet.nonlinear_field).all()


@pytest.mark.parametrize("policy", ("utility", "completion"))
def test_prepared_ownership_rejects_model_router_scene_id_query_and_factor_mutations(policy: str) -> None:
    # Model parameter mutation invalidates a prepared organization.
    model, scene, query = _core_with_controller(policy)
    organization = model.prepare(scene)
    with torch.no_grad():
        model.field_head.net[0].weight[0, 0].add_(0.01)
    with pytest.raises(ValueError, match="parameters changed"):
        model.decode(organization, query)

    # Router generation and override identity are part of preparation ownership.
    model, scene, query = _core_with_controller(policy)
    with model.source_read_override_v1("fitted"):
        routed = model.prepare(scene)
        retained = model.prepare_affine_query(routed, query)
    with pytest.raises(ValueError, match="configuration is stale"):
        model.apply_prepared_affine(retained, torch.zeros((1, 3), dtype=torch.float64))

    # Physical source IDs are correspondence data but still belong to the
    # prepared scene snapshot and cannot be edited in place behind it.
    model, scene, query = _core_with_controller(policy)
    with model.source_read_override_v1("fitted"):
        organization = model.prepare(scene)
        with torch.no_grad():
            scene.module_ids[0, 0].add_(1000)
        with pytest.raises(ValueError, match="Scene tensors changed"):
            model.decode(organization, query)

    # Query reorder requires rebuilding a retained query plan; after rebuilding,
    # outputs follow the explicit requested order.
    model, scene, query = _core_with_controller(policy)
    permutation = torch.tensor([2, 0, 1, 3])
    with model.source_read_override_v1("fitted"):
        organization = model.prepare(scene)
        retained = model.prepare_affine_query(organization, query)
        original = model.apply_prepared_affine(retained, torch.zeros((1, 3), dtype=torch.float64))
        reordered_query = replace(
            query,
            coordinates=query.coordinates.index_select(1, permutation),
            features=query.features.index_select(1, permutation),
            query_mask=query.query_mask.index_select(1, permutation),
            query_ids=query.query_ids.index_select(1, permutation),
            receiver_key="explicit-reordered-panel",
        )
        rebuilt = model.prepare_affine_query(organization, reordered_query)
        reordered = model.apply_prepared_affine(rebuilt, torch.zeros((1, 3), dtype=torch.float64))
        torch.testing.assert_close(reordered.thermal_temperature, original.thermal_temperature.index_select(1, permutation),
                                   atol=1e-11, rtol=1e-10)
        torch.testing.assert_close(reordered.nonlinear_field, original.nonlinear_field.index_select(1, permutation),
                                   atol=1e-11, rtol=1e-10)
        # In-place query-ID changes stale the retained query factors.
        with torch.no_grad():
            reordered_query.query_ids[0, 0].add_(500)
        with pytest.raises(ValueError, match="Receiver/support tensors changed"):
            model.apply_prepared_affine(rebuilt, torch.zeros((1, 3), dtype=torch.float64))

    # A retained selected-pair response detects direct factor mutation.
    model, scene, query = _core_with_controller(policy)
    with model.source_read_override_v1("fitted"):
        organization = model.prepare(scene)
        retained = model.prepare_affine_query(organization, query)
        factors = retained.direct_coefficients
        assert isinstance(factors, SourceReadAffineFactorsV1)
        changed = factors.completion_query if factors.has_completion else factors.residual_coefficients
        assert changed is not None and changed.numel() > 0
        with torch.no_grad():
            changed.reshape(-1)[0].add_(0.01)
        with pytest.raises(ValueError, match="Prepared affine factors changed"):
            model.apply_prepared_affine(retained, torch.zeros((1, 3), dtype=torch.float64))


def test_source_axis_permutation_preserves_physical_outputs() -> None:
    model, scene, query = _core_with_controller("completion")
    controls = torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float64)
    permutation = torch.tensor([2, 0, 1])
    permuted_scene = replace(
        scene,
        module_features=scene.module_features.index_select(1, permutation),
        module_coordinates=scene.module_coordinates.index_select(1, permutation),
        module_mask=scene.module_mask.index_select(1, permutation),
        module_measure=scene.module_measure.index_select(1, permutation),
        module_ids=scene.module_ids.index_select(1, permutation),
        local_radius=scene.local_radius.index_select(1, permutation),
        module_characteristic_lengths=scene.module_characteristic_lengths.index_select(1, permutation),
    )
    with model.source_read_override_v1("fitted", executor="packed"):
        original_org = model.prepare(scene)
        original = model.decode(original_org, query, controls=controls)
    permuted_controls = controls.index_select(1, permutation)
    with model.source_read_override_v1("fitted", executor="packed"):
        permuted_org = model.prepare(permuted_scene)
        permuted = model.decode(permuted_org, query, controls=permuted_controls)
    torch.testing.assert_close(permuted.nonlinear_field, original.nonlinear_field, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(permuted.thermal_temperature, original.thermal_temperature, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("policy", ("utility", "completion"))
def test_empty_packet_preserves_gradient_presence_and_adamw_state(policy: str) -> None:
    models, optimizers = [], []
    for executor in ("packed", "dense_masked", "packet"):
        model, scene, query = _core_with_controller(policy, executor=executor, threshold_gain=1e-4)
        query = replace(query, coordinates=torch.full_like(query.coordinates, 100.0), receiver_key="empty-support")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        with model.source_read_override_v1("fitted", executor=executor):
            output = model.decode(model.prepare(scene), query, controls=torch.full((1, 3), 0.2, dtype=torch.float64))
        assert output.work_counts["source_read_fine_selected_rows"] == 0
        loss = output.nonlinear_field.square().sum() + output.thermal_temperature.square().sum()
        loss.backward()
        models.append(model)
        optimizers.append(optimizer)
    for candidate in models[1:]:
        _compare_parameter_gradients(models[0], candidate)
    for optimizer in optimizers:
        optimizer.step()
    for model in models[1:]:
        for (name, expected), (other_name, actual) in zip(models[0].named_parameters(), model.named_parameters(), strict=True):
            assert name == other_name
            torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9, msg=name)
    # The zero-valued fine gradients must still advance identical named moments.
    for model, optimizer in zip(models[1:], optimizers[1:], strict=True):
        for (name, parameter), (_other_name, candidate) in zip(models[0].named_parameters(), model.named_parameters(), strict=True):
            expected_state, actual_state = optimizers[0].state[parameter], optimizer.state[candidate]
            assert expected_state.keys() == actual_state.keys(), name
            for key in expected_state:
                torch.testing.assert_close(actual_state[key], expected_state[key], atol=2e-10, rtol=2e-9, msg=f"{name}/{key}")
