"""Native cached tangents must not be mistaken for bounded heat endpoints."""

from __future__ import annotations

import pytest
import torch

from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
from test_thermal_content_hypergraph_v2 import _core_config, _prepared, _stats


@pytest.mark.parametrize("policy", ("utility", "completion"))
def test_native_cached_tangent_outside_endpoint_envelope_preserves_all_observations(policy, monkeypatch):
    config = _core_config("A2-D")
    config.update(source_read_policy=policy, source_read_executor="packed")
    model = ThermalContentHypergraphAdapterV2(_stats(), core_config=config, nx=8, ny=6).eval()
    controller = model.core.direct_reader.source_read
    controller.install_calibration_(thresholds=(10.0, 20.0), context_scales=(1., 1., 1., 1.),
                                    heating_envelope=1.0)
    controller.mark_trained_()
    controller.freeze_metadata_()
    model.core.set_source_read_training_epoch(600)
    tangent = torch.tensor([[3.0, -4.0]])
    with torch.no_grad():
        prepared = _prepared(model)
        reference = model.heat_jvp(prepared, tangent, accumulation_dtype=torch.float64)
        retained = model.prepare_affine_application(prepared)
        with pytest.raises(ValueError, match="envelope"):
            model.apply_native(retained, tangent, accumulation_dtype=torch.float64)

        def forbid_fine_read(*args, **kwargs):
            raise AssertionError("A retained native response reread fine source values")

        monkeypatch.setattr(model.core, "_read_for_response", forbid_fine_read)
        monkeypatch.setattr(model.core, "_decode_features", forbid_fine_read)
        actual = model.heat_jvp(retained, tangent, accumulation_dtype=torch.float64)
        counted, work = model.apply_native_with_work_counts(
            retained, tangent, increment=True, accumulation_dtype=torch.float64)
    assert work["direct_source_rows"] == 0
    for name in ("pred_field", "pred_interface", "pred_internal_temperature"):
        torch.testing.assert_close(actual[name], reference[name], atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(counted[name], reference[name], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(actual["pred_field"][..., :4],
                               torch.zeros_like(actual["pred_field"][..., :4]), atol=0, rtol=0)
