"""Repeated-state query-union checks through the real seven-field 3-D core."""
from dataclasses import fields, replace

import numpy as np
import torch

from honf_forward_core.content_hypergraph import QueryBatch
from honf_forward_core.interaction_reused_member_routing import DetailAnchoredMemberRoutingCore
from honf_runtime.observation_blocks import (
    ObservationDraw, SamplingBatch, extract_observations, observation_response_losses,
    prepare_observation_sampling, union_observation_dependencies,
)
from tests.anchored_fixtures import _base_config
from tests.test_e_anchored_packed_execution import fixture
from tests.test_observation_blocks import make_block


def _three_dimensional_scenes():
    _, scene, _ = fixture("local_e_cover")
    result = []
    for batch in range(scene.batch_size):
        values = {f.name: value[batch:batch + 1].clone() for f in fields(scene)
                  if torch.is_tensor(value := getattr(scene, f.name))}
        for name in ("module_coordinates", "environment_coordinates"):
            value = values[name]
            values[name] = torch.cat((value, torch.full_like(value[..., :1], .25 + .1 * batch)), dim=-1)
        for name in ("module_characteristic_lengths", "environment_characteristic_lengths"):
            value = values[name]
            values[name] = torch.cat((value, value[..., :1]), dim=-1)
        values["coordinate_origin"] = torch.zeros(1, 3, dtype=torch.double)
        values["coordinate_scale"] = torch.ones(1, 3, dtype=torch.double)
        values["scene_key"] = f"scene_{batch}"
        result.append(replace(scene, **values))
    return result


def _physical_blocks():
    masks = np.ones((2, 3, 7), dtype=bool)
    masks[0, 1, 5] = False
    masks[1, 2, 1] = False
    blocks = []
    for block_name in ("primary_fields", "repeated_fields"):
        block = make_block(block_name, n=7, dimensions=3, rows=3, mask=masks,
                           references=np.arange(42, dtype=np.float64).reshape(2, 3, 7) / 25.)
        points = np.array([[.1, .1, .2], [.6, .2, .4], [1., .3, .7]])
        extractions = []
        for extraction in block.extractions:
            dependencies = tuple(replace(dep, coordinates=tuple(point))
                                 for dep, point in zip(extraction.dependencies, points))
            extraction = replace(extraction, dependencies=dependencies)
            if block_name == "primary_fields":
                extraction = replace(extraction,
                    dependency_indices=np.append(extraction.dependency_indices, 1),
                    observation_indices=np.append(extraction.observation_indices, 0),
                    component_indices=np.append(extraction.component_indices, 6),
                    output_indices=np.append(extraction.output_indices, 6),
                    coefficients=np.append(extraction.coefficients, -.25))
            extractions.append(extraction)
        blocks.append(replace(block, coordinates=points, extractions=tuple(extractions)))
    return tuple(blocks)


def test_union_matches_unmerged_joint_core_outputs_losses_and_gradients_on_replay():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        scenes = _three_dimensional_scenes()
        base = {**_base_config(), "coordinate_dim": 3, "output_dim": 7}
        model = DetailAnchoredMemberRoutingCore(base, query_tile_size=2, retained_mass=.73).double().train()
        blocks = _physical_blocks()
        plan = prepare_observation_sampling(blocks)
        draws = SamplingBatch(plan, (ObservationDraw(0, 0, 1), ObservationDraw(0, 0, 1),
                                     ObservationDraw(1, 1, 2)))
        union = union_observation_dependencies(blocks, draws)
        assert len(union.dependencies) == 6 and union.requested_dependencies == 8

        def replay(merged):
            prepared = [model.prepare(scene) for scene in scenes]
            work = {"calls": 0, "queries": 0, "score_pairs": 0, "value_pairs": 0}

            def predict(scene_index, dependencies):
                coordinates = torch.as_tensor([dep.coordinates for dep in dependencies], dtype=torch.double)[None]
                count = len(dependencies)
                weights = coordinates[..., 0].sigmoid()
                query = QueryBatch(coordinates=coordinates, query_mask=torch.ones(1, count, dtype=torch.bool),
                    query_ids=torch.tensor([[int(dep.native_id.rsplit("_", 1)[1]) for dep in dependencies]]),
                    anchor_indices=torch.tensor([[[0, 1]] * count]),
                    anchor_weights=torch.stack((weights, 1. - weights), -1), scene_key=f"scene_{scene_index}",
                    coordinate_frame_key=scenes[scene_index].coordinate_frame_key)
                full, restricted, shared = model.decode_paired(prepared[scene_index], query)
                work["calls"] += 1
                work["queries"] += count
                work["score_pairs"] += sum(shared[f"query_unique_union_raw_score_pairs_{role}"] for role in ("M", "E"))
                work["value_pairs"] += sum(output.work_counts[f"query_anchor_coalesced_value_pairs_{role}"]
                    for output in (full, restricted) for role in ("M", "E"))
                return torch.stack((full.joint_fields[0], restricted.joint_fields[0]))

            if merged:
                joint = torch.zeros(2, len(union.dependencies), 7, dtype=torch.double)
                for scene_index in range(2):
                    positions = [i for i, dep in enumerate(union.dependencies) if dep.scene_id == f"scene_{scene_index}"]
                    decoded = predict(scene_index, [union.dependencies[i] for i in positions])
                    joint = joint.index_copy(1, torch.tensor(positions), decoded)
                predictions = [torch.stack([extract_observations(block, scene, joint, union.mappings[bi][scene])
                                             for scene in range(2)], dim=1)
                               for bi, block in enumerate(blocks)]
            else:
                predictions = []
                for bi, block in enumerate(blocks):
                    scene_predictions = []
                    for scene_index, extraction in enumerate(block.extractions):
                        selected = np.flatnonzero(union.mappings[bi][scene_index] >= 0)
                        mapping = np.full(len(extraction.dependencies), -1, dtype=np.int64)
                        mapping[selected] = np.arange(len(selected))
                        decoded = predict(scene_index, [extraction.dependencies[i] for i in selected])
                        scene_predictions.append(extract_observations(block, scene_index, decoded, mapping))
                    predictions.append(torch.stack(scene_predictions, dim=1))
            losses = observation_response_losses(blocks, predictions, draws)
            total = sum(losses[name] for name in ("extra_absolute", "point_response", "spatial_response"))
            gradients = torch.autograd.grad(total, tuple(model.parameters()), allow_unused=True)
            return predictions, losses, gradients, work

        merged_first, merged_repeat, separate_first, separate_repeat = (
            replay(True), replay(True), replay(False), replay(False))
        assert merged_first[3]["calls"] == 2 and merged_first[3]["queries"] == 6
        assert separate_first[3]["calls"] == 4 and separate_first[3]["queries"] == 8
        assert merged_first[3]["score_pairs"] > 0 and merged_first[3]["value_pairs"] > 0
        assert any(prediction.abs().max() > 0 for prediction in merged_first[0])
        for replayed in (merged_repeat, separate_first, separate_repeat):
            for actual, expected in zip(merged_first[0], replayed[0]):
                torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
            for name in ("extra_absolute", "point_response", "spatial_response"):
                assert merged_first[1][name].isfinite() and merged_first[1][name] > 0
                torch.testing.assert_close(merged_first[1][name], replayed[1][name], rtol=2e-10, atol=2e-11)
            for actual, expected in zip(merged_first[2], replayed[2]):
                assert (actual is None) == (expected is None)
                if actual is not None:
                    assert torch.isfinite(actual).all()
                    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-10)
        assert any(grad is not None and grad.abs().max() > 0 for grad in merged_first[2])
    finally:
        torch.set_num_threads(previous_threads)
