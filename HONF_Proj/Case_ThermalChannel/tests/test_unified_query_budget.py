from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
import torch

from channelthermal.training.unified_task import (
    ThermalRefinementTask,
    _resolve_query_budget,
    _sample_primary,
)
from honf_runtime.unified_training import SamplingKey, _sampling_dataset_id


BASE_BUDGET = {
    "fluid_queries": 1024,
    "material_queries_per_module": 32,
    "surface_stride": 4,
    "microbatch_cases": 8,
    "effective_cases": 48,
    "operator_rows_per_case": 128,
}


def _case(case_id: str = "case-1"):
    return {
        "case_id": case_id,
        "structure": {"module_present": np.ones((1,), dtype=np.float32)},
        "query_xy": np.stack((np.arange(4096), np.zeros(4096)), axis=-1).astype(np.float32),
        "field_targets": np.zeros((4096, 5), dtype=np.float32),
        "point_weights": np.ones((4096,), dtype=np.float32),
        "interface_target": np.zeros((1, 8, 1), dtype=np.float32),
        "interface_condition_valid_mask": np.ones((1, 8), dtype=np.float32),
        "module_internal_temperature_points": np.zeros((1, 64), dtype=np.float32),
        "module_internal_query_points": np.stack((np.arange(64), np.zeros(64)), axis=-1).astype(np.float32),
    }


def _response_family(family_id: str):
    fluid_count, local_count = 4096, 64
    return {
        "family_id": family_id,
        "structure": {"module_present": np.ones((1,), dtype=np.float32)},
        "fluid_xy": np.stack((np.arange(fluid_count), np.zeros(fluid_count)), axis=-1).astype(np.float32),
        "fluid_valid": np.ones((fluid_count,), dtype=bool),
        "material_local": np.stack((np.arange(local_count), np.zeros(local_count)), axis=-1).astype(np.float32),
        "heat_increment": np.ones((1,), dtype=np.float32),
        "deltas": {
            "fluid_fields": np.zeros((fluid_count, 5), dtype=np.float32),
            "interface": np.zeros((8, 1), dtype=np.float32),
            "solid_temperature": np.zeros((local_count, 1), dtype=np.float32),
        },
        "module_count": 1,
        "interface_valid": np.ones((8, 1), dtype=bool),
        "material_valid": np.ones((1, local_count), dtype=bool),
    }


def test_query_budget_override_changes_only_native_query_counts():
    budget, override = _resolve_query_budget(BASE_BUDGET, {"fluid_queries": 2048})
    assert override == {"fluid_queries": 2048}
    assert budget == {**BASE_BUDGET, "fluid_queries": 2048}
    assert _resolve_query_budget(BASE_BUDGET, None) == (BASE_BUDGET, None)


@pytest.mark.parametrize("override", [
    {"effective_cases": 24}, {"microbatch_cases": 16}, {"surface_stride": 8},
    {"operator_rows_per_case": 64}, {"fluid_queries": True}, {"material_queries_per_module": 0},
])
def test_query_budget_override_rejects_nonquery_or_invalid_values(override):
    with pytest.raises(ValueError):
        _resolve_query_budget(BASE_BUDGET, override)


def test_case_epoch_primary_queries_keep_prefix_across_budget_and_packing():
    first, second = _case(), _case("case-2")
    small_key = SamplingKey(0, 7, 0, 0, "warmup", "full_detail",
                            sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id="Thermal:fixed25")
    packed_key = SamplingKey(0, 7, 1, 2, "hard", "adaptive_detail",
                             sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id="Thermal:fixed25")
    small, _ = _sample_primary([first], [0], epoch=7, key=small_key, training=True,
                               budget=BASE_BUDGET, device=torch.device("cpu"))
    larger_budget = {**BASE_BUDGET, "fluid_queries": 2048}
    larger, _ = _sample_primary([first], [0], epoch=7, key=packed_key, training=True,
                                budget=larger_budget, device=torch.device("cpu"))
    packed, _ = _sample_primary([first, second], [0, 1], epoch=7, key=packed_key, training=True,
                                budget=BASE_BUDGET, device=torch.device("cpu"))
    assert np.array_equal(small["query_xy"][0, :, 0].numpy(), larger["query_xy"][0, :1024, 0].numpy())
    assert np.array_equal(small["query_xy"][0].numpy(), packed["query_xy"][0].numpy())


def test_validation_batches_use_the_separate_fixed_budget():
    provider = object.__new__(ThermalRefinementTask)
    provider.validation_cases = tuple({"case_id": str(index)} for index in range(22))
    provider.validation_budget = dict(BASE_BUDGET)
    seen = []

    def record_batch(self, case_indices, epoch, **kwargs):
        seen.append({"indices": tuple(case_indices), "epoch": epoch, **kwargs})
        return None

    provider._batch_from_indices = MethodType(record_batch, provider)
    list(provider.validation_batches())
    assert [len(row["indices"]) for row in seen] == [8, 8, 6]
    assert all(row["training"] is False and row["include_response"] is False for row in seen)
    assert all(row["budget"] == BASE_BUDGET for row in seen)


def test_response_addendum_prefix_and_validation_queries_use_fixed_budget():
    baseline = object.__new__(ThermalRefinementTask)
    candidate = object.__new__(ThermalRefinementTask)
    baseline.budget = dict(BASE_BUDGET)
    candidate.budget = {**BASE_BUDGET, "fluid_queries": 2048}
    baseline.validation_budget = dict(BASE_BUDGET)
    candidate.validation_budget = dict(BASE_BUDGET)
    key = SamplingKey(0, 11, 0, 0, "warmup", "adaptive_detail",
                      sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id="Thermal:fixed25")
    train_family = _response_family("0001")
    short = baseline._response_sample(train_family, training=True, key=key)
    long = candidate._response_sample(train_family, training=True, key=key)
    assert np.array_equal(short["fluid_ids"], long["fluid_ids"][:1024])
    assert np.array_equal(short["material_ids"], long["material_ids"])
    development_family = _response_family("0304")
    baseline_validation = baseline._response_sample(development_family, training=False, key=None)
    candidate_validation = candidate._response_sample(development_family, training=False, key=None)
    assert len(candidate_validation["fluid_ids"]) == 1024
    assert np.array_equal(baseline_validation["fluid_ids"], candidate_validation["fluid_ids"])


def test_case_epoch_formal_dataset_identity_ignores_query_budget():
    from channelthermal.training.unified_formal import FormalThermalRefinementTask

    common = {
        "dataset_binding": {"training_case_ids_sha256": "fixed-original-train-hash"},
        "validation_binding": {}, "normalization_binding": {}, "normalization_stats_sha256": "stats",
        "flow_checkpoint": "flow.pt", "flow_checkpoint_sha256": "flow-sha",
        "architecture_reference_checkpoint": "reference.pt", "architecture_reference_checkpoint_sha256": "reference-sha",
        "training_response_families": [], "training_response_sources": [],
        "response_reference_scales": {}, "operator_decision": {},
        "fresh_fine_state_sha256": "fine", "fresh_refined_state_sha256": "refined",
        "startup_validation_manifest_sha256": "dev-manifest", "startup_validation_source": {},
        "selection": {"response_reference_context": "formal3902 only"},
    }

    def identity(query_count):
        provider = object.__new__(FormalThermalRefinementTask)
        provider.formal_recipe = {**common, "sampling_version": "case_epoch_v1",
                                  "budget": {"fluid_queries": query_count}}
        provider.startup_benchmark = False
        return provider.identity_payload()

    low, high = identity(1024), identity(2048)
    assert low["manifest_fingerprint"] == high["manifest_fingerprint"] == "fixed-original-train-hash"
    assert _sampling_dataset_id(low) == _sampling_dataset_id(high)
