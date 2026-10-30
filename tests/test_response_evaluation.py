"""The response seam reruns real preparation and preserves executed evidence."""
from dataclasses import replace

import pytest
import torch

from honf_runtime.response_evaluation import ResponseDesign, evaluate_response_designs
from tests.test_r14_query_relative_values import _model_scene_query


def test_response_panel_rebuilds_and_returns_actual_evidence():
    torch.set_num_threads(1)
    core, scene, query = _model_scene_query()
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core
    model = Adapter().eval()
    calls = []
    def build(offset):
        calls.append(offset)
        coordinates = scene.module_coordinates.clone()
        coordinates[:, 0, 0] += offset
        return replace(scene, module_coordinates=coordinates), query
    result = evaluate_response_designs(model, [ResponseDesign('base', 0.), ResponseDesign('move', .01)],
        build, lambda scene, query, output: {'fields': output.joint_fields, 'mask': query.query_mask})
    base, moved = result['designs']
    assert calls == [0., .01]
    assert not torch.equal(base['native']['fields'], moved['native']['fields'])
    assert torch.equal(base['module_ids'], scene.module_ids)
    assert base['query_trace']['anchor_read_original_id'].numel() > 0
    assert base['organization']['directory'].full.original_id.numel() > 0
    assert len(base['organization']['applied_stages']) > 0
    assert base['organization']['applied_stages'][0]['mm_native_route_snapshot'].stage_identity.endswith('/mm')
    assert base['organization']['directory_route_provenance']['em']['stage_identity'].endswith('/em')
    assert base['work']['prepare'] and base['work']['decode']
    assert base['seconds']['total'] >= 0
    with pytest.raises(ValueError):
        evaluate_response_designs(model, [ResponseDesign('x', 0), ResponseDesign('x', 1)], build, None)
