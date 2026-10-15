from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import source_read_wind_evaluate as evaluator
from windfarm.training.content_hypergraph_task import ROLE_NAMES, WindContentHypergraphTask, WindSourceReadTaskV1


def test_native_panel_builder_explicitly_bypasses_formal_two_region_override(monkeypatch) -> None:
    provider = object.__new__(WindSourceReadTaskV1)
    provider.sampling_seed = 42
    provider.sampling_version = "case_epoch_v1"
    provider.sampling_dataset_id = "sealed-wind-development"
    provider.train_rows = np.asarray([1, 2, 3], dtype=np.int64)
    calls = {}

    def fake_make_batch(self, rows, key, **kwargs):
        calls.update(rows=tuple(rows), key=key, kwargs=kwargs)
        return "five-role-batch"

    monkeypatch.setattr(WindContentHypergraphTask, "_make_batch", fake_make_batch)
    result = evaluator.make_native_role_panel_batch(provider, 17, split="validation")

    assert result == "five-role-batch"
    assert calls["rows"] == (17,)
    assert calls["kwargs"] == {
        "validation": True,
        "stream_validation": True,
        "query_count": 8192,
        "track_samples": False,
    }
    assert calls["key"].sampling_version == "case_epoch_v1"
    assert calls["key"].dataset_id == "sealed-wind-development"
    assert calls["key"].seed == 42
    assert calls["key"].epoch == 0
    assert calls["key"].stage == "native_endpoint"
    assert calls["key"].microbatch_index == 0

    evaluator.make_native_role_panel_batch(provider, 1, split="validation", batch_index=7)
    assert calls["key"].stage == "native_endpoint"
    assert calls["key"].microbatch_index == 7

    evaluator.make_native_role_panel_batch(provider, 1, split="train-fixture", batch_index=3)
    assert calls["kwargs"]["validation"] is False
    assert calls["kwargs"]["stream_validation"] is False
    assert calls["key"].stage == "source_read_train_fixture"
    assert calls["key"].microbatch_index == 3


def test_case_selection_is_split_bound_and_train_default_is_input_only() -> None:
    provider = SimpleNamespace(
        validation_rows=np.asarray([10, 20, 30], dtype=np.int64),
        train_rows=np.asarray([1, 2, 3, 4], dtype=np.int64),
        view=SimpleNamespace(metadata={"n_turbines": np.asarray([0, 7, 12, 2, 15] + [0] * 26)}),
    )
    assert evaluator.choose_case_keys(provider, "validation", None) == (10, 20, 30)
    assert evaluator.choose_case_keys(provider, "train-fixture", None) == (3, 4)
    assert evaluator.choose_case_keys(provider, "train-fixture", (2,)) == (2,)
    with pytest.raises(ValueError, match="sealed provider split"):
        evaluator.choose_case_keys(provider, "validation", (99,))
    too_many = SimpleNamespace(
        validation_rows=provider.validation_rows,
        train_rows=np.arange(1, 10, dtype=np.int64),
        view=provider.view,
    )
    with pytest.raises(ValueError, match="limited to 8"):
        evaluator.choose_case_keys(too_many, "train-fixture", tuple(range(1, 10)))


def test_plane_exports_can_be_disabled_for_bounded_train_fixture() -> None:
    provider = SimpleNamespace(
        view=SimpleNamespace(metadata={"n_turbines": np.asarray([0, 6, 14, 3, 12])})
    )
    assert evaluator._plane_export_rows(provider, (1, 2, 3, 4), enabled=False) == {}
    assert evaluator._plane_export_rows(provider, (1, 2, 3, 4), enabled=True) == {"low_M": 3, "high_M": 2}


def test_case_metadata_reads_native_wd_deg_key() -> None:
    provider = SimpleNamespace(
        view=SimpleNamespace(
            metadata={
                "case": np.asarray(["native-case"]),
                "layout_index": np.asarray([42]),
                "wd_deg": np.asarray([285.0]),
                "n_turbines": np.asarray([13]),
            }
        )
    )
    assert evaluator._case_metadata(provider, 0) == {
        "row_index": 0,
        "case": "native-case",
        "layout_index": 42,
        "wind_direction_deg": 285.0,
        "n_turbines": 13,
    }


def test_native_role_metrics_are_raw_physical_rmse_and_summary_has_m_layout_tails() -> None:
    roles = np.tile(np.arange(5, dtype=np.int8), 2)
    targets = np.zeros((10, 3), dtype=np.float64)
    prediction = np.tile(np.asarray([[1.0, 2.0, 2.0]]), (10, 1))
    metadata = {"row_index": 5, "layout_index": 3, "n_turbines": 7}
    row = evaluator.physical_role_metrics(metadata, prediction, targets, roles)

    for role in ROLE_NAMES:
        assert row["roles"][role]["component_rmse_mps"] == [1.0, 2.0, 2.0]
        assert row["roles"][role]["vector_rmse_mps"] == 3.0
        assert row["roles"][role]["query_count"] == 2

    second = evaluator.physical_role_metrics(
        {"row_index": 6, "layout_index": 3, "n_turbines": 7},
        prediction * 2,
        targets,
        roles,
    )
    other_layout = evaluator.physical_role_metrics(
        {"row_index": 9, "layout_index": 4, "n_turbines": 12},
        prediction * 3,
        targets,
        roles,
    )
    summary = evaluator.summarize_physical_role_metrics((row, second, other_layout))
    assert summary["physical_metric_scope"].startswith("raw m/s RMSE")
    assert summary["overall"]["roles"]["volume"]["vector_rmse_mps"]["mean"] == 6.0
    assert summary["overall"]["roles"]["volume"]["vector_rmse_mps"]["worst"] == 9.0
    assert set(summary["by_layout_index"]) == {"3", "4"}
    assert set(summary["by_active_module_count_M"]) == {"7", "12"}


def test_policy_validation_requires_fitted_checkpoint_metadata_only_for_fitted_mode() -> None:
    class Reader:
        source_read = None

    model = SimpleNamespace(direct_reader=Reader())
    evaluator._validate_policy_model(model, "full")
    evaluator._validate_policy_model(model, "attn95")
    with pytest.raises(ValueError, match="TRAIN-fitted controller"):
        evaluator._validate_policy_model(model, "fitted")

    controller = SimpleNamespace(trained=torch.tensor(True), calibrated=torch.tensor(True))
    model.direct_reader.source_read = controller
    evaluator._validate_policy_model(model, "fitted")


def test_source_provider_guard_rejects_writable_catalogue_and_role_scale_dependency() -> None:
    provider = object.__new__(WindSourceReadTaskV1)
    provider.role_component_scales = None
    provider.role_scale_calibration = None
    provider.validation_rows = np.arange(24)
    provider.train_rows = np.arange(72)
    provider.identity_payload = lambda: {
        "protocol": "source_read_development_v1",
        "wind_test_target_values_read": False,
        "source_reader_variant": "D0",
    }
    provider.catalogue_cache = SimpleNamespace(persistent_dir=Path("/unused"), read_only_persistent=False)
    with pytest.raises(ValueError, match="shared catalogue in read-only mode"):
        evaluator._validate_source_provider(provider)

    provider.catalogue_cache.read_only_persistent = True
    provider.role_component_scales = {"volume": np.ones(3)}
    with pytest.raises(ValueError, match="historical role-scale cache dependency"):
        evaluator._validate_source_provider(provider)


def test_native_panel_hash_comparison_checks_same_row_order_ids_targets_and_sampler() -> None:
    reference = {
        1: {
            "panel_identity_sha256": "ids-coordinates",
            "target_values_sha256": "targets",
            "native_endpoint_sha256": "all-values",
            "query_sampling_sha256": "receipt",
        }
    }
    assert evaluator._assert_exact_panel_match(reference, dict(reference), label="Dscale") is None
    candidate = dict(reference)
    candidate[1] = {**reference[1], "target_values_sha256": "different-targets"}
    with pytest.raises(ValueError, match=r"panel mismatch at row 1 \(target_values_sha256\)"):
        evaluator._assert_exact_panel_match(reference, candidate, label="Dscale")
    with pytest.raises(ValueError, match="same ordered native physical panel rows"):
        evaluator._assert_exact_panel_match(reference, {2: reference[1]}, label="Dscale")


def test_saved_source_panel_binding_requires_exact_endpoint_hashes(tmp_path: Path) -> None:
    path = tmp_path / "source-result.json"
    path.write_text(
        json.dumps(
            {
                "schema": "wind_source_read_native_evaluation_v1",
                "status": "completed",
                "row_order": [69, 426],
                "query_panel": {
                    "wind_test_target_values_read": False,
                    "row_native_endpoint_sha256": {"69": "h69", "426": "h426"},
                },
            }
        )
    )
    panels = {
        69: {"native_endpoint_sha256": "h69"},
        426: {"native_endpoint_sha256": "h426"},
    }
    binding = evaluator._assert_saved_native_panel_match(path, panels, (69, 426))
    assert binding["all_native_endpoint_hashes_match"] is True
    assert binding["row_order"] == [69, 426]
    assert binding["sha256"]

    panels[426]["native_endpoint_sha256"] = "mismatch"
    with pytest.raises(ValueError, match="native endpoint mismatch at row 426"):
        evaluator._assert_saved_native_panel_match(path, panels, (69, 426))


def test_native_panel_manifest_hash_matches_archived_round2_array_order(monkeypatch) -> None:
    provider = object.__new__(WindSourceReadTaskV1)
    provider.sampling_seed = 42
    provider.sampling_version = "case_epoch_v1"
    provider.sampling_dataset_id = "sealed-wind"
    coordinates = np.arange(8192 * 3, dtype=np.float32).reshape(8192, 3)
    flat = np.arange(8192, dtype=np.int64)
    roles = (np.arange(8192, dtype=np.int8) % len(ROLE_NAMES))[None]
    targets = np.zeros((1, 8192, 3), dtype=np.float32)
    fake_batch = SimpleNamespace(
        receivers=SimpleNamespace(
            coordinates_D=coordinates[None],
            flat_indices=flat[None],
            role_ids=roles,
        ),
        targets=SimpleNamespace(velocity_mps=targets),
        auxiliary={"query_sampling_sha256": "receipt"},
    )
    monkeypatch.setattr(evaluator, "make_native_role_panel_batch", lambda *args, **kwargs: fake_batch)

    panel = evaluator._prepare_native_role_panels(provider, (69,), "validation")[69]
    assert panel["native_endpoint_sha256"] == evaluator._array_digest(flat, roles[0], coordinates, targets[0])
    assert panel["sampling_key"]["stage"] == "native_endpoint"
    assert panel["sampling_key"]["batch_index"] == 0


def test_named_paths_and_explicit_case_ids_reject_ambiguous_inputs(tmp_path: Path) -> None:
    first = tmp_path / "one.pt"
    second = tmp_path / "two.pt"
    assert evaluator.parse_named_paths((f"D0={first}", f"SComplete={second}"), "--checkpoint") == {
        "D0": first.resolve(),
        "SComplete": second.resolve(),
    }
    with pytest.raises(ValueError, match="Duplicate"):
        evaluator.parse_named_paths((f"D0={first}", f"D0={second}"), "--checkpoint")
    assert evaluator.parse_case_keys("69, 426") == (69, 426)
    with pytest.raises(ValueError, match="unique"):
        evaluator.parse_case_keys("69,69")
