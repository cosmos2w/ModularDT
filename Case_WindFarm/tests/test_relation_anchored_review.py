from types import SimpleNamespace

import numpy as np
import torch

from test_relation_anchored_wind import _adapter, _case
from windfarm.training.relation_anchored_review import cover_boundary_response, layout_sensitivity, review_native_batches


def test_sampled_native_metrics_profiles_and_count_groups():
    class Provider:
        def make_scene(self, inputs):
            return inputs
        def predict_native(self, model, scene, receivers, **kwargs):
            return SimpleNamespace(main_mps=torch.ones((1,5,3))), {"work": 1}
        def work_counts(self, batch, pred, aux):
            return {"native_receivers": 5}
    batch = SimpleNamespace(scene_inputs=(), receivers=SimpleNamespace(
        role_ids=np.arange(5)[None], coordinates_D=np.array([[[0,0,.1],[0,0,.2],[0,0,.3],[0,0,1.1],[0,0,1.2]]])),
        targets=SimpleNamespace(velocity_mps=np.zeros((1,5,3)), case_metadata=({"layout_index": 7, "n_turbines": 2},)))
    result = review_native_batches(None, Provider(), [batch], height_edges_D=[0, 1, 2])
    assert result["physical_metrics"]["by_active_module_count_M"]["2"]["row_count"] == 1
    assert result["vertical_profile"]["bins"][0]["query_count"] == 3
    assert result["vertical_profile"]["bins"][1]["component_rmse_mps"] == [1.,1.,1.]
    assert result["work_counts"]["native_receivers"] == 5


def test_surrogate_sensitivity_and_boundary_use_complete_decode():
    model, case = _adapter(), _case()
    points = np.array([[.6,.7,.9], [1.4,1.2,1.1]])
    direction = np.array([[.3,-.2,0.],[-.1,.15,0.]])
    result = layout_sensitivity(model, case, points, direction, steps=(1e-3, 1e-4))
    small = result["finite_difference"][1]
    assert small["difference_rms_mps_per_D"] < 1e-4 * max(small["ad_rms_mps_per_D"], 1.)
    probe = cover_boundary_response(model, case, points, points + [1e-6,0,0])
    assert probe["query_count"] == 2 and probe["max_absolute_velocity_change_mps"] >= 0
