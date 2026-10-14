from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tools.thermal_native_reference_evaluate import common_native_mask, native_metric, panel_identity


def test_reference_metric_preserves_zero_reference_norm_and_common_mask():
    metric = native_metric(np.array([0.0, 999.0]), np.array([0.0, 10.0]), [True, False])
    assert metric["relative_l2"] is None
    assert metric["rmse"] == 0.0 and metric["valid_rows"] == 1
    with pytest.raises(ValueError, match="nonempty"):
        native_metric([1.0], [0.0], [False])
    with pytest.raises(ValueError, match="axes"):
        native_metric([1.0, 2.0], [1.0], [True])


def test_reference_mask_uses_original_cell_centers_and_excludes_solid_cells():
    x, y = np.meshgrid((np.arange(128) + 0.5) * 12 / 128, (np.arange(64) + 0.5) * 6 / 64)
    solid = np.zeros((64, 128), dtype=bool)
    solid[7, 9] = True
    mask, grid = common_native_mask({"x_grid": x, "y_grid": y, "module_mask": solid,
                                     "structure": {"domain_length_x": [12.0], "domain_length_y": [6.0]}})
    assert mask.shape == (64, 128) and grid.shape == (64, 128, 2)
    assert np.array_equal(mask, ~solid)


@pytest.mark.parametrize("protocol,train_count,validation_count", [
    ("development", 150, 22), ("full_train1000", 600, 89),
])
def test_reference_panel_keeps_development_and_full_memberships_distinct(protocol, train_count, validation_count):
    provider = SimpleNamespace(protocol=protocol, training_case_ids=tuple(f"train{i}" for i in range(train_count)),
                               validation_case_ids=tuple(f"val{i}" for i in range(validation_count)),
                               identity_payload=lambda: {"source_binding": {"protocol": protocol}})
    assert panel_identity(provider)["train_count"] == train_count
    provider.validation_case_ids = (*provider.validation_case_ids[:-1], provider.training_case_ids[0])
    with pytest.raises(ValueError, match="crosses"):
        panel_identity(provider)


def test_reference_full_panel_rejects_the_historical_duplicate():
    provider = SimpleNamespace(protocol="full_train1000", training_case_ids=tuple(f"train{i}" for i in range(600)),
                               validation_case_ids=("0273", *(f"val{i}" for i in range(88))))
    with pytest.raises(ValueError, match="duplicate0273"):
        panel_identity(provider)
