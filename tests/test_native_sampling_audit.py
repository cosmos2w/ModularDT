from types import SimpleNamespace

import numpy as np
import torch

from honf_runtime.unified_training import SamplingKey
from channelthermal.training.unified_task import _sample_primary
from windfarm.training.unified_task import WindRefinementTask


def test_thermal_sampling_audit_does_not_change_native_draw(monkeypatch):
    import channelthermal.data.collation as collation

    monkeypatch.setattr(collation, "ChannelThermalBatchCollator", lambda: lambda rows: rows)
    original = {"case_id": "native-001", "structure": {},
                "query_xy": np.arange(40).reshape(20, 2),
                "field_targets": np.arange(20), "point_weights": np.ones(20),
                "interface_target": np.ones((1, 16)), "interface_condition_valid_mask": np.ones((1, 16)),
                "module_internal_temperature_points": np.arange(12).reshape(2, 6),
                "module_internal_query_points": np.arange(12).reshape(6, 2)}
    key = SamplingKey(1, 1, 0, 0, "joint", "test")
    kwargs = dict(epoch=1, key=key, training=True,
                  budget={"fluid_queries": 4, "material_queries_per_module": 3}, device=torch.device("cpu"))
    audited, digest = _sample_primary([original], [0], sampling_audit=True, **kwargs)
    plain, absent = _sample_primary([original], [0], sampling_audit=False, **kwargs)
    assert digest and absent is None
    np.testing.assert_array_equal(audited[0]["query_xy"], plain[0]["query_xy"])
    np.testing.assert_array_equal(audited[0]["module_internal_query_points"], plain[0]["module_internal_query_points"])


def test_wind_batch_skips_digest_at_source_when_disabled(monkeypatch):
    import windfarm.training.unified_task as wind

    provider = object.__new__(WindRefinementTask)
    provider._train_row_set = {0}
    provider.sampling_audit = False
    scene = SimpleNamespace(case_name="wind-001", layout_index=0, wind_direction_deg=270.,
                            module_present=np.array([1.]))
    sample = SimpleNamespace(target_mps=np.zeros((2, 3), dtype=np.float32))
    provider._sample_row = lambda row, key: (scene, np.zeros((2, 3)), np.zeros(2), sample)
    monkeypatch.setattr(wind, "_query_sampling_sha256", lambda *args: (_ for _ in ()).throw(AssertionError("hashed")))
    batch = provider.make_batch((0,), SamplingKey(1, 1, 0, 0, "joint", "test"))
    assert "query_sampling_sha256" not in batch.auxiliary
