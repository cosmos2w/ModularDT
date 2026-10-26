from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2, QueryBatch, SceneBatch
from honf_forward_core.content_hypergraph.query_chunks_v1 import select_query_rows_v1
from honf_forward_core.interaction_reused_member_routing import InteractionReusedMemberRoutingCore


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
    module_mask = torch.ones((1, module_count), dtype=torch.bool)
    environment_mask = torch.ones((1, environment_count), dtype=torch.bool)
    module_measure = torch.linspace(0.0, 1.0, module_count)[None]
    environment_measure = torch.linspace(0.0, 1.0, environment_count)[None]
    return SceneBatch(
        module_features=torch.randn(1, module_count, 9),
        environment_features=torch.randn(1, environment_count, 6),
        module_coordinates=module_coordinates[None],
        environment_coordinates=environment_coordinates[None],
        module_mask=module_mask,
        environment_mask=environment_mask,
        module_measure=module_measure,
        environment_measure=environment_measure,
        module_ids=torch.arange(20, 20 + module_count)[None],
        environment_ids=torch.arange(100, 100 + environment_count)[None],
        local_radius=torch.full((1, module_count), 0.2),
        operating_features=torch.randn(1, 12),
        coordinate_origin=torch.tensor([[0.0, -0.5]]),
        coordinate_scale=torch.tensor([[2.0, 1.0]]),
        module_measure_reference=torch.ones(1),
        environment_measure_reference=torch.ones(1),
        scene_key="round13-3-test-scene",
        normalizer_key="round13-3-test-normalizer",
        coordinate_frame_key="round13-3-test-frame",
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
        scene_key="round13-3-test-scene",
        coordinate_frame_key="round13-3-test-frame",
    )


def _core(mode: str = "route", *, tile_size: int = 2, output_dim: int = 4,
          context_depth: int = 2) -> InteractionReusedMemberRoutingCore:
    model = InteractionReusedMemberRoutingCore(
        _base_config(output_dim=output_dim, context_depth=context_depth),
        member_routing_mode=mode,
        input_dataset="synthetic-native-contract",
        query_tile_size=tile_size,
    )
    return model


def test_config_common_weights_joint_output_and_removed_legacy_paths() -> None:
    route = _core("route")
    full = _core("full")
    assert route.architecture == "R13-3-Route"
    assert full.architecture == "R13-3-Full"
    assert route.get_config()["input_capability"] == "joint_prescribed_records_v1"
    assert InteractionReusedMemberRoutingCore.from_config(route.get_config()).get_config() == route.get_config()
    for name, value in route.member_router.state_dict().items():
        torch.testing.assert_close(value, full.member_router.state_dict()[name], rtol=0, atol=0)
    assert route.joint_begin_update is None
    assert not hasattr(route, "scene_summary_net")
    assert not hasattr(route, "local_reader")
    assert not hasattr(route, "field_head")
    assert not any(name.startswith("direct_reader.") for name, _ in route.named_parameters())

    route.eval()
    prepared = route.prepare(_scene())
    output = route.decode(prepared, _query(3))
    assert output.joint_fields.shape == (1, 3, 4)
    assert output.nonlinear_field.shape == (1, 3, 3)
    assert output.thermal_temperature.shape == (1, 3)
    assert output.thermal_basis is None

    nonthermal = _core("route", output_dim=2)
    nonthermal_output = nonthermal.decode(nonthermal.prepare(_scene()), _query(2))
    assert nonthermal_output.joint_fields.shape == (1, 2, 2)
    with pytest.raises(AttributeError, match="fourth"):
        _ = nonthermal_output.thermal_temperature
    with pytest.raises(ValueError, match="exactly two"):
        _core("route", context_depth=1)


def test_actual_two_cycle_gates_feed_original_id_routing_and_dimensionless_measure() -> None:
    torch.manual_seed(18)
    core = _core("route", tile_size=4)
    core.eval()
    scene = _scene(module_count=3, environment_count=6)
    prepared = core.prepare(scene, trace_request={"include": []})

    assert len(prepared.context_stage_traces) == 2
    assert [stage["me_native_route_snapshot"].stage_identity for stage in prepared.context_stage_traces] == [
        "cycle0/me", "cycle1/me"
    ]
    final_mm = prepared.final_applied_context_stages["mm"]["incidence"]
    final_me = prepared.final_applied_context_stages["me"]["incidence"]
    identity = torch.diag_embed(scene.module_mask.to(prepared.B_M.dtype))
    off_diagonal = ~torch.eye(scene.num_modules, dtype=torch.bool)[None]
    torch.testing.assert_close(prepared.B_M, final_mm * off_diagonal + identity)
    torch.testing.assert_close(prepared.B_E, final_me * scene.environment_mask[:, None])
    torch.testing.assert_close(prepared.B_M.diagonal(dim1=1, dim2=2), scene.module_mask.to(prepared.B_M.dtype))
    assert prepared.B_M_edge_index.shape[1] == 3
    assert prepared.B_E_edge_index.shape[1] == 3
    assert prepared.nu_M[0, 0] == 1  # present zero-measure M remains addressable by identity
    positive = scene.environment_mask & (scene.environment_measure > 0)
    mean_measure = scene.environment_measure[positive].mean()
    torch.testing.assert_close(prepared.nu_E[positive], scene.environment_measure[positive] / mean_measure)
    assert prepared.nu_E[0, 0] == 0
    assert prepared.work_counts["context_ee_score_rows"] == 2 * scene.num_environment_tokens**2
    assert prepared.work_counts["cached_scene_projection_rows_M"] == scene.num_modules
    assert prepared.work_counts["cached_scene_projection_rows_E"] == scene.num_environment_tokens

    query = _query(3)
    output = core.decode(prepared, query, return_trace=True)
    trace = output.trace
    assert trace is not None
    assert torch.equal(trace["original_module_ids"], scene.module_ids)
    assert torch.equal(trace["original_environment_ids"], scene.environment_ids)
    torch.testing.assert_close(trace["B_M"], prepared.B_M)
    torch.testing.assert_close(trace["B_E"], prepared.B_E)
    assert output.work_counts["query_M_cheap_score_rows"] == int(trace["candidate_mask_M"].sum())
    assert output.work_counts["query_E_cheap_score_rows"] == int(trace["candidate_mask_E"].sum())
    assert output.work_counts["query_address_coefficient_pair_rows"] == (
        query.num_queries * scene.num_modules * (scene.num_modules + scene.num_environment_tokens)
    )
    for indices_name, weights_name in (("selected_value_indices_M", "member_weights_M"),
                                       ("selected_value_indices_E", "member_weights_E")):
        rows = trace[indices_name]
        weights = trace[weights_name]
        assert rows.ndim == 2 and rows.shape[-1] == 3
        if rows.numel():
            b, q, source = rows.unbind(-1)
            assert bool((weights[b, q, source] > 0).all())


@pytest.mark.parametrize("module_count,environment_count", [(0, 4), (1, 0), (0, 0)])
def test_empty_typed_inventories_keep_empty_applied_gates_and_decode(module_count, environment_count) -> None:
    core = _core("route", tile_size=4)
    core.eval()
    scene = _scene(module_count, environment_count)
    prepared = core.prepare(scene)
    assert prepared.B_M.shape == (1, module_count, module_count)
    assert prepared.B_E.shape == (1, module_count, environment_count)
    output = core.decode(prepared, _query(3))
    assert output.joint_fields.shape == (1, 3, 4)
    assert bool(torch.isfinite(output.joint_fields).all())
    if module_count == 0:
        assert output.work_counts["query_M_proposal_score_rows"] == 0
        assert output.work_counts["query_E_cheap_score_rows"] == 0


@pytest.mark.parametrize("module_count", [1, 12])
def test_m1_and_m12_task_loss_has_live_proposal_member_value_and_applied_gate_gradients(module_count) -> None:
    # These deterministic fixture seeds keep the learned null-aware route in
    # an active region for both inventory sizes without changing production
    # initialization or adding an artificial gradient path.
    torch.manual_seed(1 if module_count == 1 else 39)
    core = _core("route", tile_size=2)
    core.train()
    scene = _scene(module_count, 8)
    prepared = core.prepare(scene)
    output = core.decode(prepared, _query(5))
    loss = output.joint_fields.square().mean()
    loss.backward()

    def grad_norm(parameter):
        return 0.0 if parameter.grad is None else float(parameter.grad.norm())

    required = {
        "M proposal scorer": core.member_router.proposal_M[-1].weight,
        "M member scorer": core.member_router.member_M[-1].weight,
        "E member scorer": core.member_router.member_E[-1].weight,
        "M original values": core.member_router.value_M.weight,
        "E original values": core.member_router.value_E.weight,
        "applied ME relation": core.applied_incidence.routes["me"].head_logits,
        "joint head": core.member_router.joint_head.weight,
    }
    norms = {name: grad_norm(parameter) for name, parameter in required.items()}
    assert all(value > 1.0e-9 for value in norms.values()), norms
    assert core.member_router.proposal_M[-1].bias.item() == pytest.approx(-1.5)
    assert core.applied_incidence.routes["me"].bias.item() == pytest.approx(1.5)


def test_full_query_intervention_tiling_dense_union_parity_and_probe_semantics() -> None:
    torch.manual_seed(0)
    core = _core("route", tile_size=4)
    core.eval()
    scene = _scene(3, 8)
    prepared = core.prepare(scene)
    query = _query(7)
    baseline = core.decode(prepared, query)
    dense = core.dense_candidate_union(prepared, query)
    torch.testing.assert_close(dense.joint_fields, baseline.joint_fields, rtol=0, atol=1.0e-6)
    assert dense.work_counts["query_E_value_execution_rows"] >= baseline.work_counts["query_E_value_execution_rows"]
    assert dense.trace is None

    sliced_query = select_query_rows_v1(query, torch.tensor([1, 4, 6]))
    baseline_trace = core.capture_interaction_routing_trace(prepared, sliced_query)
    query_indices = torch.tensor([1, 4, 6])
    b_control, b_prepare = core.member_routing_intervention(
        "b_fixed_candidates_scores", prepared, query, query_indices=query_indices,
    )
    assert b_prepare is None
    b_output = core.decode(prepared, query, intervention=b_control)
    assert b_output.joint_fields.shape == baseline.joint_fields.shape
    b_trace = core.capture_interaction_routing_trace(prepared, sliced_query, intervention=b_control)
    assert torch.equal(b_trace["candidate_mask_M"], baseline_trace["candidate_mask_M"])
    assert torch.equal(b_trace["candidate_mask_E"], baseline_trace["candidate_mask_E"])
    torch.testing.assert_close(b_trace["proposal_weights_M"], baseline_trace["proposal_weights_M"])
    torch.testing.assert_close(b_trace["member_logits_M"], baseline_trace["member_logits_M"])
    torch.testing.assert_close(b_trace["member_logits_E"], baseline_trace["member_logits_E"])
    assert not torch.equal(b_trace["B_E"], baseline_trace["B_E"])
    assert not torch.equal(b_trace["member_weights_E"], baseline_trace["member_weights_E"])

    for probe in ("geometry_b_comparable_counts", "geometry_member_scores_fixed_b_pi", "all_candidates_open"):
        control, prepare_options = core.member_routing_intervention(
            probe, prepared, query, query_indices=query_indices,
        )
        assert prepare_options is None
        result = core.decode(prepared, query, intervention=control)
        assert result.joint_fields.shape == baseline.joint_fields.shape
        probe_trace = core.capture_interaction_routing_trace(prepared, sliced_query, intervention=control)
        if probe == "geometry_b_comparable_counts":
            assert not torch.equal(probe_trace["B_E"], baseline_trace["B_E"])
        elif probe == "geometry_member_scores_fixed_b_pi":
            torch.testing.assert_close(probe_trace["proposal_weights_M"], baseline_trace["proposal_weights_M"])
            assert not torch.equal(probe_trace["member_logits_E"], baseline_trace["member_logits_E"])
        else:
            torch.testing.assert_close(result.joint_fields, baseline.joint_fields, rtol=0, atol=1.0e-6)
            assert result.work_counts["query_E_value_execution_rows"] >= baseline.work_counts[
                "query_E_value_execution_rows"]

    baseline_tiles = [
        core.capture_interaction_routing_trace(
            prepared, select_query_rows_v1(query, torch.arange(start, min(start + core.query_tile_size,
                                                                          query.num_queries))),
        )
        for start in range(0, query.num_queries, core.query_tile_size)
    ]
    full_pi = torch.cat([tile["proposal_weights_M"] for tile in baseline_tiles], dim=1)
    full_masks_M = torch.cat([tile["candidate_mask_M"] for tile in baseline_tiles], dim=1)
    full_masks_E = torch.cat([tile["candidate_mask_E"] for tile in baseline_tiles], dim=1)
    full_logits_M = torch.cat([tile["member_logits_M"] for tile in baseline_tiles], dim=1)
    full_logits_E = torch.cat([tile["member_logits_E"] for tile in baseline_tiles], dim=1)
    explicit = core.decode(prepared, query, intervention={
        "fixed_pi": full_pi,
        "fixed_candidate_mask_M": full_masks_M,
        "fixed_candidate_mask_E": full_masks_E,
        "fixed_member_logits_M": full_logits_M,
        "fixed_member_logits_E": full_logits_E,
    })
    torch.testing.assert_close(explicit.joint_fields, baseline.joint_fields, rtol=0, atol=1.0e-6)
    with pytest.raises(ValueError, match="cover all"):
        core.decode(prepared, query, intervention={"fixed_pi": full_pi[:, :2]})

    _unused, removal_options = core.member_routing_intervention(
        "remove_contextual_message_recompute", prepared, query, query_indices=query_indices,
    )
    assert removal_options is not None
    changed = core.prepare(scene, trace_request={"include": []}, **removal_options)
    stage0_me = next(stage for stage in changed.context_stage_traces
                     if stage["me_native_route_snapshot"].stage_identity == "cycle0/me")
    snapshot = stage0_me["me_native_route_snapshot"]
    gate = stage0_me["me_applied_incidence"]
    removal = removal_options["context_message_removal"]
    removed = ((snapshot.destination_ids == removal["destination_id"])[:, :, None]
               & (snapshot.donor_ids == removal["donor_id"])[:, None, :])
    assert bool(removed.any())
    assert bool((gate.masked_select(removed) == 0).all())


def test_prepared_route_state_mutation_is_rejected() -> None:
    core = _core("route", tile_size=4).eval()
    prepared = core.prepare(_scene())
    query = _query(3)
    with torch.no_grad():
        prepared.B_E[0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="routing state changed"):
        core.decode(prepared, query)


def test_typed_nulls_are_independent_and_all_null_reads_no_source_values():
    torch.manual_seed(39)
    core = _core("route", tile_size=4).eval()
    scene, query = _scene(3, 8), _query(3)
    prepared = core.prepare(scene)
    pi = torch.full((1, 3, 3), 1 / 3)
    for empty, active, M_logit, E_logit in (("M", "E", -100.0, 100.0),
                                          ("E", "M", 100.0, -100.0)):
        output = core.decode(prepared, query, return_trace=True, intervention={
            "fixed_pi": pi, "fixed_member_logits_M": torch.full((1, 3, 3), M_logit),
            "fixed_member_logits_E": torch.full((1, 3, 8), E_logit),
        })
        assert output.work_counts[f"query_{empty}_value_execution_rows"] == 0
        assert output.work_counts[f"query_{active}_value_execution_rows"] > 0
        assert torch.equal(output.trace[f"member_null_{empty}"], torch.ones(1, 3))
    control = {"fixed_pi": torch.zeros_like(pi)}
    empty = core.decode(prepared, query, intervention=control)
    changed_scene = replace(scene, module_features=scene.module_features + 100,
        environment_features=scene.environment_features - 100,
        operating_features=scene.operating_features + 100)
    changed = core.decode(core.prepare(changed_scene), query, intervention=control)
    torch.testing.assert_close(empty.joint_fields, changed.joint_fields, rtol=0, atol=0)
    assert empty.work_counts["query_M_value_execution_rows"] == 0
    assert empty.work_counts["query_E_value_execution_rows"] == 0


def test_original_id_paths_and_member_query_permutations_preserve_reads():
    torch.manual_seed(39)
    core = _core("route", tile_size=4).double().eval()
    scene = _scene(3, 8)
    scene = replace(scene, **{name: value.double() for name, value in vars(scene).items()
                              if torch.is_tensor(value) and value.is_floating_point()})
    query = _query(3)
    query = replace(query, coordinates=query.coordinates.double())
    prepared = core.prepare(scene)
    output = core.decode(prepared, query, return_trace=True)
    trace = output.trace
    for role in ("M", "E"):
        selected = trace[f"selected_value_indices_{role}"]
        matrix = trace[f"B_{role}"]
        identifiers = scene.module_ids if role == "M" else scene.environment_ids
        ids = []
        for batch, q, member in selected.tolist():
            assert bool(((trace["proposal_weights_M"][batch, q] > 0)
                         & (matrix[batch, :, member] > 0)).any())
            ids.append((int(query.query_ids[batch, q]), int(identifiers[batch, member])))
        assert len(ids) == len(set(ids))
        assert len(ids) == output.work_counts[f"query_{role}_value_execution_rows"]
    module_order, environment_order = torch.tensor([2, 0, 1]), torch.tensor([7, 2, 5, 0, 6, 1, 4, 3])
    values = {}
    for name, value in vars(scene).items():
        if torch.is_tensor(value) and name.startswith("module_") and value.ndim >= 2:
            values[name] = value.index_select(1, module_order)
        elif torch.is_tensor(value) and name.startswith("environment_") and value.ndim >= 2:
            values[name] = value.index_select(1, environment_order)
    values["local_radius"] = scene.local_radius.index_select(1, module_order)
    permuted_scene = replace(scene, **values)
    query_order = torch.tensor([2, 0, 1])
    permuted_query = select_query_rows_v1(query, query_order)
    reordered = core.decode(core.prepare(permuted_scene), permuted_query, return_trace=True)
    torch.testing.assert_close(reordered.joint_fields, output.joint_fields[:, query_order], rtol=1e-10, atol=1e-10)
    for role, id_name in (("M", "original_module_ids"), ("E", "original_environment_ids")):
        for new_q, old_q in enumerate(query_order.tolist()):
            def weights_by_id(t, q):
                return {int(identifier): float(weight) for identifier, weight in
                    zip(t[id_name][0], t[f"member_weights_{role}"][0, q]) if weight > 0}
            before, after = weights_by_id(trace, old_q), weights_by_id(reordered.trace, new_q)
            assert before.keys() == after.keys()
            for identifier in before:
                assert before[identifier] == pytest.approx(after[identifier], abs=1e-10)
    core.query_tile_size = 1
    tiled = core.decode(core.prepare(scene), query)
    torch.testing.assert_close(tiled.joint_fields, output.joint_fields, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint replay regression")
def test_m12_cuda_checkpoint_replay_with_duplicate_address_paths() -> None:
    torch.manual_seed(39)
    core = _core("route", tile_size=2).cuda().train()
    scene = _scene(12, 12)
    scene = SceneBatch(**{
        name: (value.cuda() if isinstance(value, torch.Tensor) else value)
        for name, value in scene.__dict__.items()
    })
    query = _query(9)
    query = QueryBatch(**{
        name: (value.cuda() if isinstance(value, torch.Tensor) else value)
        for name, value in query.__dict__.items()
    })
    output = core.decode(core.prepare(scene), query)
    output.joint_fields.square().mean().backward()
    assert output.joint_fields.shape == (1, 9, 4)
    assert core.member_router.proposal_M[-1].weight.grad is not None
