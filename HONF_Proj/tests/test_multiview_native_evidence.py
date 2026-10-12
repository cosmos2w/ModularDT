"""Focused contracts for multiview evidence bindings and physical reducers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HONF_ROOT = Path(__file__).resolve().parents[1]
if str(HONF_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(HONF_ROOT / "src"))

from honf_forward_core.interface_fields.interaction_preserving_joint import (
    InteractionPreservingJointCore,
)
from tools.diagnostics.multiview_native_evidence import (
    CACHE_BINDING_FIELDS,
    apply_native_validation_policy,
    cache_binding_matches,
    cache_binding_mismatches,
    equal_case_summary,
    equal_cluster_summary,
    masked_field_metrics,
    signed_response_metrics,
    stratified_case_summary,
    validate_multiview_native_trace,
    wind_role_metrics,
)


def _cache_binding() -> dict:
    return {
        "dataset_id": "wind_farm_volume_v1",
        "partition": "fixed24_dev",
        "sample_ids": [69, 70, 71],
        "checkpoint_sha256": "a" * 64,
        "normalizer_sha256": "b" * 64,
        "query_sha256": "c" * 64,
        "mask_sha256": "d" * 64,
        "target_sha256": "e" * 64,
        "output_schema": "native_velocity_v1",
        "units": {"Ux": "m/s", "Uy": "m/s", "Uz": "m/s"},
    }


def test_cache_reuse_requires_every_semantic_binding() -> None:
    requested = _cache_binding()
    assert set(CACHE_BINDING_FIELDS) <= requested.keys()
    assert cache_binding_matches(requested, dict(requested))

    for field in ("sample_ids", "checkpoint_sha256", "query_sha256", "mask_sha256", "target_sha256", "units"):
        changed = dict(requested)
        changed[field] = [999] if field == "sample_ids" else "different"
        assert not cache_binding_matches(requested, changed)
        assert field in cache_binding_mismatches(requested, changed)

    incomplete = dict(requested)
    incomplete.pop("query_sha256")
    assert not cache_binding_matches(incomplete, requested)


def test_classic_validation_defaults_remain_untouched_and_fresh_policy_is_explicit() -> None:
    class Core:
        def __init__(self) -> None:
            self.values: list[bool] = []

        def set_consolidated_native_validation(self, enabled: bool) -> None:
            self.values.append(enabled)

    class Model:
        def __init__(self) -> None:
            self.core = Core()

    model = Model()
    assert apply_native_validation_policy(model, "classic_native_default") == "classic_native_default_unchanged"
    assert model.core.values == []
    assert apply_native_validation_policy(model, "consolidated_route_flags_v1") == "consolidated_route_flags_v1_applied"
    assert model.core.values == [True]
    with pytest.raises(ValueError, match="Unsupported"):
        apply_native_validation_policy(model, "silently_disable_checks")


def test_masked_metrics_use_only_fixed_valid_cells() -> None:
    prediction = np.array([[1.0, 3.0], [1000.0, np.nan]])
    reference = np.array([[0.0, 1.0], [-1000.0, np.nan]])
    mask = np.array([[True, True], [False, False]])
    result = masked_field_metrics(prediction, reference, mask)
    assert result["count"] == 2
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["relative_l2"] == pytest.approx(np.sqrt(5.0) / 1.0)
    assert result["signed_bias"] == pytest.approx(1.5)
    assert result["reference_l2_norm"] == pytest.approx(1.0)
    assert result["reference_rms"] == pytest.approx(1.0 / np.sqrt(2.0))


def test_equal_case_and_cluster_reducers_do_not_pool_cells_or_direction_rows() -> None:
    case_summary = equal_case_summary({"small_mask": 2.0, "large_mask": 4.0})
    assert case_summary["equal_case_mean"] == 3.0

    summary = equal_cluster_summary(
        {"69": 0.0, "70": 0.0, "71": 0.0, "171": 9.0},
        {"69": "layout23", "70": "layout23", "71": "layout23", "171": "layout57"},
    )
    assert summary["cluster_count"] == 2
    assert summary["row_count"] == 4
    assert summary["equal_cluster_mean"] == 4.5
    assert summary["per_cluster_row_count"] == {"layout23": 3, "layout57": 1}

    strata = stratified_case_summary(
        {"0277": 1.0, "0291": 2.0, "0294": 4.0},
        {"0277": 3, "0291": 5, "0294": 7},
        required_strata=[3, 5, 7],
    )
    assert strata["strata"]["5"]["equal_case_mean"] == 2.0
    with pytest.raises(ValueError, match="required strata"):
        stratified_case_summary({"0277": 1.0}, {"0277": 3}, required_strata=[3, 5, 7])


def test_wind_vector_rmse_sums_components_without_dividing_by_three() -> None:
    roles = np.repeat(
        ["volume", "hub_slab", "downstream_envelope", "near_turbine", "background"],
        [2, 2, 2, 2, 2],
    )
    reference = np.tile(np.array([2.0, 0.02, 0.04]), (10, 1))
    prediction = reference + 1.0
    result = wind_role_metrics(prediction, reference, roles, np.ones(10, dtype=bool), expected_total_queries=10)
    assert result["vector_rmse_m_s"] == pytest.approx(np.sqrt(3.0))
    assert result["component_rmse_m_s"] == pytest.approx([1.0, 1.0, 1.0])
    assert result["component_reference_rms_m_s"] == pytest.approx([2.0, 0.02, 0.04])
    assert result["total_query_count"] == 10


def test_multiview_trace_shapes_and_content_sum_match_executed_aggregate() -> None:
    content = np.zeros((1, 2, 3, 2, 64), dtype=np.float64)
    content[:, :, 0] = 0.25
    content[:, :, 1] = 0.75
    content[:, :, 2] = 1.0
    aggregate = content.sum(axis=2)
    trace = {
        "s": np.ones((1, 2, 3, 2)),
        "rho": np.ones((1, 2, 3, 2)),
        "w": np.ones((1, 2, 3, 2)),
        "view_ids": np.array(["view0", "view1"]),
        "weighted_message_content": content,
        "head_aggregate": aggregate,
        "aggregate": aggregate.reshape(1, 2, 128),
    }
    result = validate_multiview_native_trace(trace, expected_views=2)
    assert result["view_count"] == 2
    assert result["weighted_content_reconstruction_max_abs"] == 0.0

    broken = dict(trace)
    broken["head_aggregate"] = aggregate + 0.1
    broken["aggregate"] = broken["head_aggregate"].reshape(1, 2, 128)
    with pytest.raises(ValueError, match="does not exactly reproduce"):
        validate_multiview_native_trace(broken, expected_views=2)


def test_wind_metrics_promote_endpoints_before_subtraction() -> None:
    roles = np.repeat(
        ["volume", "hub_slab", "downstream_envelope", "near_turbine", "background"], 2,
    )
    prediction = np.full((10, 3), 3.0e38, dtype=np.float32)
    reference = -prediction
    result = wind_role_metrics(
        prediction, reference, roles, np.ones(10, dtype=bool), expected_total_queries=10,
    )
    assert np.isfinite(result["vector_rmse_m_s"])
    assert result["vector_rmse_m_s"] == pytest.approx(float(prediction[0, 0]) * 2.0 * np.sqrt(3.0))


def test_r1_trace_retains_implicit_single_128_wide_view() -> None:
    content = np.arange(1 * 2 * 3 * 128, dtype=np.float32).reshape(1, 2, 3, 128)
    head_aggregate = content.sum(axis=2, dtype=np.float32)[:, :, None, :]
    trace = {
        "s": np.ones((1, 2, 3), dtype=np.float32),
        "rho": np.ones((1, 2, 3), dtype=np.float32),
        "w": np.ones((1, 2, 3), dtype=np.float32),
        "view_ids": np.array([0]),
        "weighted_message_content": content,
        "head_aggregate": head_aggregate,
        "aggregate": head_aggregate.squeeze(-2),
    }
    result = validate_multiview_native_trace(trace, expected_views=1)
    assert result["message_width_per_view"] == 128
    assert result["weighted_message_shape"] == [1, 2, 3, 128]
    assert result["native_dtype_exact_head_reduction"] is True


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_core_r1_and_r2_traces_match_native_reduction_contract(
    device: str, dtype: torch.dtype,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("An authorized CUDA device is required for native GPU reduction qualification.")
    generator = torch.Generator().manual_seed(8301)
    rand = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    values = {
        "sources": rand(1, 4, 4),
        "context": rand(1, 3),
        "centers": rand(1, 4, 2),
        "present": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float64),
        "lengths": torch.tensor([[2.0, 1.5]], dtype=torch.float64),
        "source_lengths": torch.full((1, 4), 0.2, dtype=torch.float64),
        "source_measures": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float64),
        "environment_tokens": rand(1, 192, 5),
        "environment_coords": rand(1, 192, 2),
        "environment_present": torch.ones(1, 192, dtype=torch.float64),
        "environment_measures": torch.linspace(0.4, 1.6, 192, dtype=torch.float64)[None],
        "source_ids": torch.tensor([[31, 7, 19, -1]], dtype=torch.int64),
        "environment_ids": torch.arange(192, dtype=torch.int64)[None],
        "domain_origin": torch.tensor([[-1.0, -0.5]], dtype=torch.float64),
    }
    values = {name: tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)
              for name, tensor in values.items()}
    for mode, expected_views, expected_route_ndim, expected_width in (
        ("P-R", 1, 3, 128),
        ("P-R2", 2, 4, 64),
    ):
        model = InteractionPreservingJointCore(
            4, 3, 5, spatial_dim=2, hidden=12, message=128, mode=mode,
            collective_width=6, regional_anchors=0, field_outputs=2,
            affine_outputs=0, initialization_seed=73,
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            for scorer in model.native_routing.score_mlps.values():
                projection = scorer[-1].weight
                projection.copy_(torch.linspace(
                    -0.1, 0.1, projection.numel(), device=device, dtype=projection.dtype,
                ).reshape_as(projection))
            prepared = model.prepare_context(**values, routing_trace=True)
        assert len(prepared.native_routing_trace) == 6
        for native_trace in prepared.native_routing_trace:
            assert native_trace["s"].ndim == expected_route_ndim
            assert native_trace["weighted_message_content"].shape[-1] == expected_width
            checked = validate_multiview_native_trace(
                native_trace, expected_views=expected_views, reduction_device=device,
            )
            assert checked["native_dtype_exact_head_reduction"] is True
            assert checked["native_dtype_exact_flattened_aggregate"] is True
            assert checked["reduction_backend"] == "torch"
            assert checked["reduction_device"].startswith(device)


def test_trace_reconstruction_rejects_silent_dtype_or_device_substitution() -> None:
    content = torch.ones(1, 2, 3, 2, 64, dtype=torch.float32)
    head = content.sum(dim=2)
    trace = {
        "s": torch.ones(1, 2, 3, 2), "rho": torch.ones(1, 2, 3, 2),
        "w": torch.ones(1, 2, 3, 2), "view_ids": torch.arange(2),
        "weighted_message_content": content, "head_aggregate": head,
        "aggregate": head.reshape(1, 2, 128),
    }
    with pytest.raises(TypeError, match="same tensor dtype"):
        validate_multiview_native_trace({**trace, "head_aggregate": head.double()})
    arrays = {name: value.numpy() for name, value in trace.items()}
    with pytest.raises(TypeError, match="original tensor trace representation"):
        validate_multiview_native_trace(arrays, reduction_device="cpu")


def test_response_replay_keeps_endpoint_peak_change_distinct_from_max_delta() -> None:
    baseline = np.array([0.0, 10.0])
    perturbed = np.array([9.0, 10.5])
    result = signed_response_metrics(baseline, perturbed, baseline, perturbed, np.ones(2, dtype=bool))
    assert result["delta_rmse"] == 0.0
    assert result["measured_endpoint_peak_change"] == pytest.approx(0.5)
    assert result["measured_max_delta"] == pytest.approx(9.0)
