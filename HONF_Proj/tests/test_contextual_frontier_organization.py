from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from honf_forward_core.content_hypergraph import QueryBatch, SceneBatch
from honf_forward_core.contextual_organization import ContextualOrganizationJointCore
from honf_runtime.contextual_frontier_organization import (
    decode_receiver_roles,
    compare_reprepared_source_change,
    evaluate_common_e_only_quality,
    fixed_train_panel,
    inspect_prepared,
    measure_native_cost,
)


def _scene() -> SceneBatch:
    return SceneBatch(
        module_features=torch.randn((1, 4, 9)),
        environment_features=torch.randn((1, 5, 6)),
        module_coordinates=torch.tensor([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]]]),
        environment_coordinates=torch.tensor([[[4.8, 0.0], [5.0, 0.0], [5.2, 0.0], [5.4, 0.0], [5.6, 0.0]]]),
        module_mask=torch.ones((1, 4), dtype=torch.bool),
        environment_mask=torch.ones((1, 5), dtype=torch.bool),
        module_measure=torch.ones((1, 4)),
        environment_measure=torch.ones((1, 5)),
        module_ids=torch.tensor([[11, 17, 23, 29]]),
        environment_ids=torch.tensor([[101, 103, 107, 109, 113]]),
        local_radius=torch.full((1, 4), 0.02),
        operating_features=torch.zeros((1, 12)),
        coordinate_origin=torch.zeros((1, 2)),
        coordinate_scale=torch.tensor([[6.0, 2.0]]),
        module_measure_reference=torch.ones((1,)),
        environment_measure_reference=torch.ones((1,)),
        scene_key="inspect-scene",
        normalizer_key="train-normalizer",
        coordinate_frame_key="physical-frame",
        module_characteristic_lengths=torch.full((1, 4, 2), 0.02),
        environment_characteristic_lengths=torch.full((1, 5, 2), 0.2),
    )


def _query(points: torch.Tensor) -> QueryBatch:
    return QueryBatch(
        coordinates=points.reshape(1, -1, 2),
        scene_key="inspect-scene",
        coordinate_frame_key="physical-frame",
    )


def _core() -> ContextualOrganizationJointCore:
    core = ContextualOrganizationJointCore(
        module_input_dim=9,
        environment_input_dim=6,
        coordinate_dim=2,
        operating_input_dim=12,
        output_dim=4,
        architecture="J-H",
        context_dim=16,
        num_heads=4,
        context_depth=2,
        local_hidden_dim=16,
        init_seed=819,
        query_tile_size=2,
    )
    core.eval()
    for policy in core.frontier_policies.values():
        nn.init.zeros_(policy.stop_head.weight)
        nn.init.constant_(policy.stop_head.bias, 5.0)
    return core


def test_inspection_uses_executed_rows_and_receiver_roles_reuse_preparation() -> None:
    torch.manual_seed(83)
    core = _core()
    scene = _scene()
    query = _query(torch.tensor([[5.0, 0.0], [5.25, 0.0], [5.5, 0.0]]))
    organization = core.prepare(scene)
    native = SimpleNamespace(
        organization=organization,
        query_batch=query,
        scene_key=scene.scene_key,
        union=SimpleNamespace(unique_counts=torch.tensor([query.num_queries])),
    )

    receipt, arrays = inspect_prepared(core, native)
    json.dumps(receipt, allow_nan=False)
    assert receipt["module_source_read_empty"]
    assert not receipt["environment_source_read_empty"]
    assert receipt["e_only_query_rows"] > 0
    assert receipt["banks"]["environment"]["actual_K"] == 1
    assert arrays["environment_query_source_local_index"].size > 0
    assert arrays["environment_selected_scene_rows"].size > 0
    assert receipt["cost_inventory"]["contextual_attention_pair_slots_per_cycle"]["total"] == 81
    assert receipt["cost_inventory"]["contextual_attention_head_pair_rows_all_cycles"] == 648
    assert receipt["cost_inventory"]["candidate_distance_accounting_by_bank"]["module"][
        "unique_unordered_geometry_distance_evaluations"
    ] == 6
    assert receipt["cost_inventory"]["candidate_distance_accounting_by_bank"]["module"][
        "geometry_distance_matrix_cells_allocated"
    ] == 16

    role_receipt, outputs = decode_receiver_roles(
        core,
        native,
        {"fluid": _query(torch.tensor([[5.0, 0.0], [5.25, 0.0]])),
         "surface": _query(torch.tensor([[5.5, 0.0]]))},
        return_trace=True,
    )
    assert role_receipt["same_prepared_organization_for_all_roles"]
    assert role_receipt["prepare_calls"] == 0
    assert outputs["fluid"].joint_fields.shape == (1, 2, 4)
    assert outputs["surface"].joint_fields.shape == (1, 1, 4)
    assert core.decode(organization, query).joint_fields.shape == (1, 3, 4)


def test_fixed_train_panel_covers_strata_and_records_review_boundaries() -> None:
    cases = []
    ids = []
    for count in (1, 1, 2, 2, 3, 3, 4, 4):
        case_id = f"case-{len(cases):02d}"
        ids.append(case_id)
        cases.append({"case_id": case_id, "structure": {"module_present": [1.0] * count}})
    provider = SimpleNamespace(
        training_case_ids=tuple(ids),
        training_cases=tuple(cases),
        manifest={"manifest_sha256": "fixed-input-manifest"},
    )
    panel = fixed_train_panel(provider, max_cases=6, max_per_module_count=2)
    assert set(panel["by_module_count"]) == {"1", "2", "3", "4"}
    assert len(panel["case_ids"]) == 6
    assert panel["review_epochs"] == [100, 500]
    assert panel["subset_manifest_sha256"] == "fixed-input-manifest"
    assert panel == fixed_train_panel(provider, max_cases=6, max_per_module_count=2)


def test_common_e_only_validation_uses_actual_reads_and_shared_physical_mask() -> None:
    torch.manual_seed(87)
    core = _core()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core
            self.architecture = "J-H"

    model = Model()
    points = torch.tensor([[5.0, 0.0], [5.25, 0.0], [5.5, 0.0]])
    scene = _scene()
    target = torch.randn((1, 3, 5))
    point_weights = torch.ones((1, 3))
    stencil = SimpleNamespace(
        indices=torch.arange(3).reshape(1, 3, 1).expand(-1, -1, 4),
        weights=torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 4).expand(1, 3, -1),
        valid=torch.ones((1, 3), dtype=torch.bool),
    )
    batch = SimpleNamespace(
        case_keys=("validation-1",),
        scene_inputs=scene,
        receivers=points,
        targets=SimpleNamespace(point_weights=point_weights, field_targets=target),
    )

    class Provider:
        validation_case_ids = ("validation-1",)

        def native_evaluation_batches(self, *, split, case_keys, microbatch_size):
            assert split == "validation" and tuple(case_keys) == ("validation-1",) and microbatch_size == 1
            yield batch

        def predict_native(self, current_model, scene_inputs, receivers, architecture, split, *, epoch):
            query = _query(receivers)
            organization = current_model.core.prepare(scene_inputs)
            output = current_model.core.decode(organization, query)
            fields = output.joint_fields
            native_field = torch.cat((fields[..., :3], fields[..., :1] * 0.0, fields[..., 3:4]), dim=-1)
            prepared = SimpleNamespace(
                organization=organization,
                query_batch=query,
                stencils={"fluid": stencil},
            )
            predictions = SimpleNamespace(native_prepared=prepared, native_main={"pred_field": native_field})
            return predictions, None

    original_epoch = core.organization_epoch
    result = evaluate_common_e_only_quality({"J-H": (model, Provider())}, ("validation-1",), epoch=100)
    row = result["per_case"][0]
    assert row["common_e_only_physical_queries"] == 3
    assert row["actual_read_edges_by_arm"]["J-H"]["module"] == 0
    assert row["actual_read_edges_by_arm"]["J-H"]["environment"] > 0
    assert result["equal_case_mean_relative_l2_on_common_mask"]["J-H"]["temperature"] is not None
    assert core.organization_epoch == original_epoch


def test_changed_prescribed_input_forces_fresh_context_prepare() -> None:
    torch.manual_seed(88)
    core = _core()
    base_scene = _scene()
    points = torch.tensor([[5.0, 0.0], [5.25, 0.0]])

    class Adapter:
        def __init__(self):
            self.core = core

        def scene_batch(self, structure, *, scene_key):
            features = base_scene.module_features.clone()
            features[..., 8] = float(structure["heat"])
            return replace(base_scene, module_features=features, scene_key=scene_key)

        def prepare_native(self, structure, fluid_xy, *, local_query_points, ntheta, scene_key):
            scene = self.scene_batch(structure, scene_key=scene_key)
            query = QueryBatch(
                coordinates=fluid_xy.reshape(1, -1, 2), scene_key=scene_key,
                coordinate_frame_key=scene.coordinate_frame_key,
            )
            return SimpleNamespace(
                organization=core.prepare(scene), query_batch=query, scene_key=scene_key,
                fluid_xy=fluid_xy, local_query_points=local_query_points, ntheta=ntheta,
            )

    adapter = Adapter()
    original = adapter.prepare_native(
        {"heat": 0.0}, points, local_query_points=None, ntheta=32, scene_key="source-probe"
    )
    receipt, arrays = compare_reprepared_source_change(adapter, original, {"heat": 1.0})
    assert receipt["reprepared_new_native_object"]
    assert receipt["reprepared_new_context_object"]
    assert receipt["same_physical_records_geometry_and_masks"]
    assert receipt["changed_input_tensors"]["module_features"]
    assert arrays["module_context_delta"].shape == original.organization.module_states.shape


def test_native_cost_separates_prepare_decode_backward_without_optimizer_steps() -> None:
    torch.manual_seed(89)
    module = nn.Linear(3, 2)
    source = torch.randn((4, 3), requires_grad=True)
    result = measure_native_cost(
        lambda: source * 0.5,
        module,
        reference=lambda prepared: module(prepared.detach()),
        loss=lambda output: output.square().mean(),
        module=module,
        repeats=2,
    )
    assert result["optimizer_steps"] == 0
    assert result["device"] == "cpu"
    assert result["prepare"]["mean_ms"] >= 0
    assert result["decode"]["mean_ms"] >= 0
    assert result["reference_decode"]["mean_ms"] >= 0
    assert result["backward"]["mean_ms"] >= 0
    assert result["cuda_memory"] is None
