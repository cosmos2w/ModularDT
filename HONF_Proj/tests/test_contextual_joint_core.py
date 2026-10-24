from __future__ import annotations

from dataclasses import replace

import torch

from honf_forward_core.content_hypergraph import QueryBatch, SceneBatch
from honf_forward_core.contextual_organization import ContextualOrganizationJointCore


def _scene(*, far_environment: bool = False, heat: float = 0.0, requires_grad: bool = False) -> SceneBatch:
    module_coordinates = torch.tensor([[[0.0, 0.0], [0.5, 0.1], [1.0, 0.2], [1.5, 0.3]]])
    environment_coordinates = torch.tensor(
        [[[0.0, 0.0], [0.4, 0.1], [0.9, 0.15], [1.3, 0.2], [1.7, 0.3]]]
    )
    if far_environment:
        module_coordinates = torch.zeros((1, 4, 2))
        environment_coordinates = torch.tensor(
            [[[4.8, 0.0], [5.0, 0.0], [5.2, 0.0], [5.4, 0.0], [5.6, 0.0]]]
        )
    module_features = torch.randn((1, 4, 9))
    module_features[..., 8] = heat
    if requires_grad:
        module_features.requires_grad_()
    environment_features = torch.randn((1, 5, 6))
    operating = torch.linspace(-0.4, 0.5, 12).reshape(1, 12)
    return SceneBatch(
        module_features=module_features,
        environment_features=environment_features,
        module_coordinates=module_coordinates,
        environment_coordinates=environment_coordinates,
        module_mask=torch.ones((1, 4), dtype=torch.bool),
        environment_mask=torch.ones((1, 5), dtype=torch.bool),
        module_measure=torch.ones((1, 4)),
        environment_measure=torch.tensor([[0.2, 0.3, 0.4, 0.3, 0.2]]),
        module_ids=torch.tensor([[11, 17, 23, 29]]),
        environment_ids=torch.tensor([[101, 103, 107, 109, 113]]),
        local_radius=torch.full((1, 4), 0.06 if far_environment else 0.24),
        operating_features=operating,
        coordinate_origin=torch.tensor([[0.0, 0.0]]),
        coordinate_scale=torch.tensor([[6.0, 2.0]] if far_environment else [[2.0, 0.6]]),
        module_measure_reference=torch.ones((1,)),
        environment_measure_reference=torch.ones((1,)),
        scene_key="contextual-scene",
        normalizer_key="train-only-normalizer",
        coordinate_frame_key="physical-frame",
        module_characteristic_lengths=torch.full((1, 4, 2), 0.08 if far_environment else 0.12),
        environment_characteristic_lengths=torch.full((1, 5, 2), 0.2),
    )


def _query(coordinates: torch.Tensor, *, key="contextual-scene") -> QueryBatch:
    return QueryBatch(
        coordinates=coordinates.reshape(1, -1, 2),
        scene_key=key,
        coordinate_frame_key="physical-frame",
    )


def _core(architecture: str = "J-H", **kwargs) -> ContextualOrganizationJointCore:
    values = dict(
        module_input_dim=9,
        environment_input_dim=6,
        coordinate_dim=2,
        operating_input_dim=12,
        output_dim=4,
        architecture=architecture,
        context_dim=16,
        num_heads=4,
        context_depth=2,
        local_hidden_dim=16,
        init_seed=731,
        query_tile_size=2,
    )
    values.update(kwargs)
    core = ContextualOrganizationJointCore(**values)
    core.eval()
    return core


def test_prepare_decode_backward_and_strict_reload() -> None:
    torch.manual_seed(23)
    scene = _scene()
    query = _query(torch.tensor([[0.05, 0.01], [0.45, 0.10], [0.92, 0.16], [1.4, 0.25]]))
    core = _core()
    output = core.decode(core.prepare(scene), query, return_trace=True)
    assert output.joint_fields is not None and output.joint_fields.shape == (1, 4, 4)
    assert torch.equal(output.nonlinear_field, output.joint_fields[..., :3])
    assert torch.equal(output.thermal_temperature, output.joint_fields[..., 3:4])
    assert output.trace is not None
    assert output.trace["module_query_gates"][0]["query_indices"].numel() > 0
    assert output.work_counts["module_query_key_score_rows"] > 0
    assert output.work_counts["module_local_geometry_rows"] == query.num_queries * int(scene.module_mask.sum())
    assert output.work_counts["module_local_packed_pair_rows"] == 0
    assert "contextual_module_records" not in output.work_counts
    loss = output.joint_fields.square().mean()
    loss.backward()
    assert core.field_head.weight.grad is not None
    assert torch.isfinite(core.field_head.weight.grad).all()
    assert core.module_value.weight.grad is not None
    restored = ContextualOrganizationJointCore.from_config(core.get_config())
    restored.load_state_dict(core.state_dict(), strict=True)
    restored.eval()
    restored_output = restored.decode(restored.prepare(scene), query)
    assert torch.allclose(output.joint_fields.detach(), restored_output.joint_fields, atol=1e-6, rtol=1e-6)

    packed_query = QueryBatch(
        coordinates=query.coordinates[:, :1],
        local_batch_indices=torch.tensor([0]),
        local_query_indices=torch.tensor([0]),
        local_module_indices=torch.tensor([0]),
        local_weights=torch.tensor([0.75]),
        scene_key=query.scene_key,
        coordinate_frame_key=query.coordinate_frame_key,
    )
    packed_output = core.decode(core.prepare(scene), packed_query)
    assert packed_output.work_counts["module_local_geometry_rows"] == 0
    assert packed_output.work_counts["module_local_packed_pair_rows"] == 1


def test_query_tiles_match_unsplit_decode() -> None:
    torch.manual_seed(31)
    scene = _scene()
    coordinates = torch.tensor(
        [[0.02, 0.01], [0.4, 0.08], [0.8, 0.14], [1.2, 0.21], [1.65, 0.29], [0.7, 0.12]]
    )
    core = _core(query_tile_size=3)
    prepared = core.prepare(scene)
    full_query = _query(coordinates)
    full = core.decode(prepared, full_query).joint_fields
    reference = core.decode_reference(prepared, full_query).joint_fields
    assert torch.allclose(full, reference, atol=2e-6, rtol=2e-6)
    left = core.decode(prepared, _query(coordinates[:2])).joint_fields
    right = core.decode(prepared, _query(coordinates[2:])).joint_fields
    assert torch.allclose(full, torch.cat((left, right), dim=1), atol=2e-6, rtol=2e-6)


def test_environment_only_read_keeps_heat_in_contextual_prediction() -> None:
    torch.manual_seed(37)
    core = _core()
    query = _query(torch.tensor([[5.0, 0.0], [5.3, 0.0]]))
    cold_scene = _scene(far_environment=True, heat=0.0, requires_grad=True)
    hot_scene = replace(_scene(far_environment=True, heat=1.0), scene_key="contextual-scene")
    cold_output = core.decode(core.prepare(cold_scene), query)
    hot_output = core.decode(core.prepare(hot_scene), query)
    assert cold_output.work_counts["module_query_source_edges"] == 0
    assert cold_output.work_counts["environment_query_source_edges"] > 0
    assert torch.isfinite(cold_output.joint_fields).all()
    assert not torch.allclose(cold_output.joint_fields, hot_output.joint_fields, atol=1e-7, rtol=1e-7)
    cold_output.joint_fields[..., 0].square().sum().backward()
    assert cold_scene.module_features.grad is not None
    assert cold_scene.module_features.grad[..., 8].abs().sum() > 0


def test_leaf_architecture_and_cached_subtree_intervention() -> None:
    torch.manual_seed(41)
    scene = _scene()
    adaptive = _core("J-H")
    for policy in adaptive.frontier_policies.values():
        torch.nn.init.zeros_(policy.stop_head.weight)
        torch.nn.init.constant_(policy.stop_head.bias, 5.0)
    adaptive_prepared = adaptive.prepare(scene)
    module_bank = adaptive_prepared.module_banks[0]
    assert module_bank.frontier.K == 1
    assert module_bank.forest is not None and module_bank.forest.root is not None
    unchanged_environment = adaptive_prepared.environment_banks[0]
    changed = adaptive.intervene_frontier(
        adaptive_prepared, "module", 0, module_bank.forest.root, stop=False
    )
    assert changed.module_states is adaptive_prepared.module_states
    assert changed.environment_banks[0] is unchanged_environment
    assert changed.module_banks[0].frontier.K >= 2

    direct = _core("J-D")
    for policy in direct.frontier_policies.values():
        torch.nn.init.zeros_(policy.stop_head.weight)
        torch.nn.init.constant_(policy.stop_head.bias, 5.0)
    direct_bank = direct.prepare(scene).module_banks[0]
    assert direct_bank.frontier.K == direct_bank.present_count
    direct_prepared = direct.prepare(scene)
    direct_forest = direct_prepared.module_banks[0].forest
    assert direct_forest is not None and direct_forest.root is not None
    target = next(index for index in direct_forest.internal_indices if index != direct_forest.root)
    target_ancestry = set(direct_forest.nodes[target].ancestry_ids)
    outside_leaf_ids = {
        direct_forest.nodes[index].ancestry_ids[0]
        for index in range(direct_forest.record_count)
        if direct_forest.nodes[index].ancestry_ids[0] not in target_ancestry
    }
    probe = direct.intervene_frontier(direct_prepared, "module", 0, target, stop=True)
    probe_frontier = probe.module_banks[0].frontier
    assert probe_frontier is not None and target in probe_frontier.node_indices
    assert outside_leaf_ids <= set(value for group in probe_frontier.ancestry_ids for value in group)


def test_operating_and_record_context_change_prepared_states() -> None:
    torch.manual_seed(43)
    core = _core()
    scene = _scene()
    controlled = replace(scene, operating_features=scene.operating_features + 0.5)
    first = core.prepare(scene)
    second = core.prepare(controlled)
    assert not torch.allclose(first.environment_states, second.environment_states)
    assert not torch.allclose(first.module_states, second.module_states)
