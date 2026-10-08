from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from honf_runtime.unified_training import _sampling_dataset_id as engine_sampling_dataset_id
from torch import nn

from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline
from windfarm.training import unified_formal
from windfarm.training.unified_task import (
    DEFAULT_ROLE_QUERY_COUNTS,
    WIND_GATE_COMPACT_C1_TRANSITION,
    WIND_GATE_COMPACT_C1_VERSION,
    WIND_W1_RECIPE_ID,
    WIND_W3_RECIPE_ID,
    resolve_wind_recipe,
)


def _metadata() -> dict[str, np.ndarray]:
    groups = np.repeat(np.arange(200, dtype=np.int64), 3)
    directions = np.tile(np.asarray([270.0, 285.0, 300.0]), 200)
    counts = np.tile(np.arange(6, 31, dtype=np.int16), 24)[:600]
    turbine_xy = np.full((600, 30, 2), np.nan, dtype=np.float32)
    for row, count in enumerate(counts):
        turbine_xy[row, : int(count)] = np.arange(int(count) * 2, dtype=np.float32).reshape(-1, 2)
    return {
        "case": np.asarray([f"gen_{group:04d}_wd{int(direction)}" for group, direction in zip(groups, directions)]),
        "layout": np.asarray([f"gen_{group:04d}" for group in groups]),
        "layout_index": groups,
        "wd_deg": directions,
        "n_turbines": counts,
        "turbine_xy_D": turbine_xy,
    }


class _FakeView:
    def __init__(self, root: Path):
        self.metadata = _metadata()
        self.volume = SimpleNamespace(volume_root=root / "family_volume")
        self.token_shape = (2, 2, 2)


def _fake_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_root = tmp_path / "wind"
    volume_root = data_root / "family_volume"
    volume_root.mkdir(parents=True)
    for name in (
        "U.npy", "case.npy", "layout_index.npy", "wd_deg.npy", "run_shape.npy",
        "run_cell_offsets.npy", "run_x_offsets.npy", "run_y_offsets.npy", "run_z_offsets.npy",
        "x_cell_m.npy", "y_cell_m.npy", "z_cell_m.npy",
    ):
        (volume_root / name).write_bytes(b"metadata-fixture")
    monkeypatch.setattr(unified_formal, "WindFarmNativeView", lambda *args, **kwargs: _FakeView(data_root))
    return data_root


def _config(tmp_path: Path, data_root: Path) -> dict[str, str]:
    return {
        "data_root": str(data_root),
        "derived_root": str(tmp_path / "derived"),
        "formal_root": str(tmp_path / "formal"),
    }


def _normalizer() -> VelocityNormalizer:
    return VelocityNormalizer(
        mean=np.asarray([1.0, 0.0, 0.0]),
        std=np.asarray([0.2, 0.1, 0.1]),
        safe_std=np.asarray([0.2, 0.1, 0.1]),
        sample_count_per_row=8192,
        source_rows=420,
        seed=42,
    )


def _profile() -> VerticalProfileBaseline:
    return VerticalProfileBaseline(
        bin_centers_D=np.asarray([0.1, 6.15]),
        values_mps=np.asarray([[4.0, 0.1, -0.2], [12.0, 0.5, 0.2]]),
        counts=np.asarray([100, 100], dtype=np.int64),
    )


def test_metadata_only_recipe_binds_full_seed42_split_without_training_ready_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root = _fake_dataset(tmp_path, monkeypatch)
    monkeypatch.setattr(
        unified_formal,
        "fit_velocity_statistics",
        lambda *args, **kwargs: pytest.fail("metadata-only recipe must not read target values"),
    )

    recipe = unified_formal.prepare_recipe(_config(tmp_path, data_root), metadata_only=True)
    assert recipe["status"] == "metadata_only_not_ready"
    assert recipe["ready_for_training"] is False
    assert recipe["target_values_read"] is False
    assert recipe["partition_identity"]["split"]["partitions"]["train"]["rows"] == 420
    assert recipe["partition_identity"]["split"]["partitions"]["validation"]["rows"] == 90
    assert recipe["partition_identity"]["split"]["partitions"]["test"]["rows"] == 90
    assert recipe["training_profile"]["microbatch_cases"] == 4
    assert recipe["training_profile"]["effective_cases"] == 24
    validation = unified_formal.validate_recipe(recipe)
    assert validation["ready_for_training"] is False
    assert validation["target_values_read"] is False
    with pytest.raises(ValueError, match="completed full TRAIN-only normalization"):
        unified_formal.create_task(recipe)


def test_manual_formal_metadata_binds_selected_w3_width_without_reading_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root = _fake_dataset(tmp_path, monkeypatch)
    monkeypatch.setattr(
        unified_formal,
        "fit_velocity_statistics",
        lambda *args, **kwargs: pytest.fail("metadata-only W3 preparation must not read target values"),
    )

    recipe = unified_formal.prepare_recipe(
        {**_config(tmp_path, data_root), "wind_recipe_id": WIND_W3_RECIPE_ID},
        metadata_only=True,
    )

    assert recipe["status"] == "metadata_only_not_ready"
    assert recipe["target_values_read"] is False
    assert recipe["wind_training_recipe"]["recipe_id"] == WIND_W3_RECIPE_ID
    assert recipe["model"]["hidden"] == 128
    assert recipe["model"]["message"] == 128
    assert recipe["model"]["environment_token_shape"] == [2, 2, 2]


def test_full_prepare_fits_only_420_original_train_rows_and_binds_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root = _fake_dataset(tmp_path, monkeypatch)
    observed: dict[str, object] = {}

    def fake_fit(dataset, rows, *, samples_per_row, seed, profile_bins):
        observed.update(rows=np.asarray(rows).copy(), samples=samples_per_row, seed=seed, bins=profile_bins)
        return _normalizer(), _profile()

    def fake_role_fit(view, rows):
        row_ids = np.asarray(rows, dtype=np.int64)
        return {
            "schema_version": 1,
            "recipe_id": unified_formal.FORMAL_RECIPE_ID,
            "method": "pooled componentwise standard deviation over deterministic Q=1024 five-role native samples",
            "training_scope": "all 420 original TRAIN rows only",
            "training_rows_sha256": unified_formal._indices_sha256(row_ids),
            "seed": 42,
            "role_query_counts_per_row": dict(DEFAULT_ROLE_QUERY_COUNTS),
            "role_sample_counts": {name: 420 * count for name, count in DEFAULT_ROLE_QUERY_COUNTS.items()},
            "scalar_scale_rule": "sqrt(mean of the three per-component TRAIN variances)",
            "minimum_scale_mps": 1.0e-3,
            "component_statistics": {},
            "role_scales_mps": {name: 1.0 for name in DEFAULT_ROLE_QUERY_COUNTS},
            "query_sampling_sha256": "query-sample-test",
            "target_sample_sha256": "target-sample-test",
            "target_values_read": "TRAIN only",
            "solver_attempts": 0,
        }

    monkeypatch.setattr(unified_formal, "fit_velocity_statistics", fake_fit)
    monkeypatch.setattr(unified_formal, "_fit_formal_role_scales", fake_role_fit)
    recipe = unified_formal.prepare_recipe(_config(tmp_path, data_root), metadata_only=False)
    assert recipe["status"] == "prepared_ready"
    assert recipe["ready_for_training"] is True
    assert recipe["normalization_fit"]["status"] == "fitted_from_all_420_original_TRAIN_rows"
    assert len(observed["rows"]) == 420
    assert np.array_equal(observed["rows"], recipe["partition_identity"]["train_rows"])
    assert observed["samples"] == 8192
    assert observed["seed"] == 42
    assert Path(recipe["normalization_path"]).is_file()
    assert Path(recipe["normalization_binding_path"]).is_file()
    assert Path(recipe["role_scale_path"]).is_file()
    assert recipe["role_scale_calibration"]["status"] == "fitted_from_all_420_original_TRAIN_rows"
    validation = unified_formal.validate_recipe(recipe)
    assert validation["ready_for_training"] is True
    assert validation["normalization_fit_performed"] is False
    assert validation["source_rows"] == {"train": 420, "validation": 90, "test_metadata_only": 90}


def test_manual_formal_recipe_preserves_explicit_compact_gate_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_root = _fake_dataset(tmp_path, monkeypatch)
    training_recipe = resolve_wind_recipe(WIND_W1_RECIPE_ID)
    assert training_recipe is not None
    training_recipe.pop("recipe_sha256")
    training_recipe["gate_version"] = WIND_GATE_COMPACT_C1_VERSION
    training_recipe["gate_transition"] = list(WIND_GATE_COMPACT_C1_TRANSITION)
    training_recipe["recipe_sha256"] = unified_formal._stable_digest(training_recipe)
    config = _config(tmp_path, data_root)
    config["wind_training_recipe"] = training_recipe

    formal_recipe, _, _ = unified_formal._metadata_recipe(config)
    bound = formal_recipe["wind_training_recipe"]
    assert bound["gate_version"] == WIND_GATE_COMPACT_C1_VERSION
    assert bound["gate_transition"] == [0.35, 0.65]


def test_versioned_formal_recipe_uses_shared_four_layout_train_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root = _fake_dataset(tmp_path, monkeypatch)
    observed: dict[str, object] = {}

    def fake_fit(dataset, rows, *, samples_per_row, seed, profile_bins):
        del dataset, samples_per_row, seed, profile_bins
        observed["normalization_rows"] = np.asarray(rows).copy()
        return _normalizer(), _profile()

    def fake_role_fit(view, rows, train_layout_indices, background_profile, normalizer, **kwargs):
        del background_profile, normalizer
        row_ids = np.asarray(rows, dtype=np.int64)
        layouts = tuple(int(value) for value in train_layout_indices[:4])
        layout_by_row = np.asarray(view.metadata["layout_index"], dtype=np.int64)
        calibration_rows = [int(row) for row in row_ids if int(layout_by_row[int(row)]) in set(layouts)]
        training_recipe = resolve_wind_recipe(WIND_W1_RECIPE_ID)
        assert training_recipe is not None
        counts = dict(training_recipe["calibration_role_query_counts"])
        observed["calibration_rows"] = calibration_rows
        observed["calibration_kwargs"] = kwargs
        return {
            "schema_version": 1,
            "method": "same panel W0 target variance and W1 profile residual RMS",
            "scope": "selected TRAIN rows only",
            "training_fingerprint": kwargs["training_fingerprint"],
            "training_rows_sha256": unified_formal._indices_sha256(row_ids),
            "calibration_layout_indices": list(layouts),
            "calibration_row_indices": calibration_rows,
            "calibration_query_count_per_row": 1024,
            "role_query_counts_per_row": counts,
            "role_sample_counts": {name: len(calibration_rows) * count for name, count in counts.items()},
            "scalar_scale_rule": "max(sqrt(mean of three TRAIN target component variances), 1e-3 m/s)",
            "component_scale_rule": "max(profile residual RMS, physical floor, 0.10 scalar scale)",
            "physical_component_floor_mps": 0.009,
            "scalar_role_scales_mps": {name: 1.0 for name in counts},
            "component_role_scales_mps": {name: [1.0, 0.2, 0.1] for name in counts},
            "component_statistics": {},
            "query_sampling_sha256": "query-sample-test",
            "target_sample_sha256": "target-sample-test",
            "background_profile_sha256": "profile-test",
            "sampling_version": "case_epoch_v1",
            "catalogue_build_or_lookup_seconds": 1.25,
            "native_target_sampling_seconds": 2.5,
            "catalogue_cache_after_calibration": {"hits": 0, "misses": 12},
            "target_values_read": "TRAIN calibration panel only",
            "solver_attempts": 0,
        }

    monkeypatch.setattr(unified_formal, "fit_velocity_statistics", fake_fit)
    monkeypatch.setattr(unified_formal, "_fit_train_role_scales", fake_role_fit)
    config = {**_config(tmp_path, data_root), "wind_recipe_id": WIND_W1_RECIPE_ID}
    recipe = unified_formal.prepare_recipe(config, metadata_only=False)

    assert recipe["wind_training_recipe"]["recipe_id"] == WIND_W1_RECIPE_ID
    assert recipe["role_query_counts"] == recipe["wind_training_recipe"]["role_query_counts"]
    assert len(observed["normalization_rows"]) == 420
    assert len(observed["calibration_rows"]) == 12
    assert recipe["role_scale_calibration"]["status"] == "fitted_from_shared_four_layout_original_TRAIN_panel"
    assert recipe["role_scale_profile_path"]
    assert Path(recipe["role_scale_profile_path"]).is_file()
    assert recipe["role_scales_mps"] == {name: 1.0 for name in DEFAULT_ROLE_QUERY_COUNTS}
    validation = unified_formal.validate_recipe(recipe)
    assert validation["ready_for_training"] is True


def test_formal_provider_uses_5000_epoch_schedule_and_full_scope_identity(tmp_path: Path):
    class View:
        def __init__(self):
            self.token_shape = (2, 2, 2)
            self.metadata = {
                "layout_index": np.repeat(np.arange(6, dtype=np.int64), 3),
                "wd_deg": np.tile(np.asarray([270.0, 285.0, 300.0]), 6),
            }

    rows = np.arange(18, dtype=np.int64)
    layouts = np.arange(6, dtype=np.int64)
    manifest = {
        "subset_id": unified_formal.FORMAL_RECIPE_ID,
        "manifest_sha256": "formal-manifest-test",
        "train_layout_indices": layouts.tolist(),
        "validation_layout_indices": [],
        "train_layout_selection": {"selected_turbine_counts": list(range(6, 31)), "uncovered_turbine_counts": []},
    }
    recipe = {
        "recipe_sha256": "formal-recipe-test",
        "source_metadata_sha256": "metadata-test",
        "source_file_inventory": [],
        "partition_identity": {"train_rows_sha256": "train-test"},
        "normalization_sha256": "normalization-test",
        "normalization_binding_sha256": "binding-test",
    }
    provider = unified_formal.WindFormalRefinementTask(
        View(),  # type: ignore[arg-type]
        train_rows=rows,
        validation_rows=np.asarray([], dtype=np.int64),
        normalizer=_normalizer(),
        background_profile=_profile(),
        manifest=manifest,
        manifest_path=tmp_path / "formal_recipe.json",
        normalization_path=tmp_path / "normalization.json",
        role_scales={name: 1.0 for name in DEFAULT_ROLE_QUERY_COUNTS},
        role_scale_sha256="role-scales-test",
        formal_recipe=recipe,
    )
    model = nn.Sequential(nn.Linear(2, 3))
    groups = provider.optimizer_groups(model, "warmup", "warmup")
    assert groups
    assert all(group.schedule.total_epochs == 5000 for group in groups)
    assert all(group.schedule.hold_through_epoch == 2000 for group in groups)
    assert all(group.schedule.peak_lr == pytest.approx(3.0e-4) for group in groups)
    assert all(group.schedule.final_lr == pytest.approx(3.0e-6) for group in groups)


def test_versioned_formal_sampler_key_uses_full_train_membership(tmp_path: Path):
    class View:
        def __init__(self):
            self.token_shape = (2, 2, 2)
            self.metadata = {
                "layout_index": np.repeat(np.arange(6, dtype=np.int64), 3),
                "wd_deg": np.tile(np.asarray([270.0, 285.0, 300.0]), 6),
            }

    training_recipe = resolve_wind_recipe(WIND_W1_RECIPE_ID)
    assert training_recipe is not None
    train_rows = np.arange(18, dtype=np.int64)
    train_fingerprint = "full-train-membership-sha"
    manifest = {
        "subset_id": unified_formal.FORMAL_RECIPE_ID,
        "manifest_sha256": "local-formal-manifest-sha",
        "train_layout_indices": list(range(6)),
        "validation_layout_indices": [],
        "train_layout_selection": {"selected_turbine_counts": list(range(6, 31)), "uncovered_turbine_counts": []},
    }
    formal_recipe = {
        "recipe_sha256": "prepared-full-dataset-recipe-sha",
        "source_metadata_sha256": "metadata-sha",
        "source_file_inventory": [],
        "partition_identity": {"train_rows_sha256": train_fingerprint},
        "normalization_sha256": "normalization-sha",
        "normalization_binding_sha256": "binding-sha",
        "role_scale_sha256": "role-scale-sha",
    }
    normalization_path = tmp_path / "normalization.json"
    normalization_path.write_text("fixture")
    provider = unified_formal.WindFormalRefinementTask(
        View(),  # type: ignore[arg-type]
        train_rows=train_rows,
        validation_rows=np.asarray([], dtype=np.int64),
        normalizer=_normalizer(),
        background_profile=_profile(),
        manifest=manifest,
        manifest_path=tmp_path / "formal_recipe.json",
        normalization_path=normalization_path,
        role_scales={name: 0.5 for name in DEFAULT_ROLE_QUERY_COUNTS},
        role_scale_sha256="role-scale-sha",
        role_query_counts=training_recipe["role_query_counts"],
        role_component_scales={name: (0.2, 0.1, 0.1) for name in DEFAULT_ROLE_QUERY_COUNTS},
        role_scale_calibration={"training_rows_sha256": train_fingerprint},
        recipe=training_recipe,
        sampling_dataset_fingerprint=train_fingerprint,
        formal_recipe=formal_recipe,
    )
    provider._message_scale = 1.0
    identity = provider.identity_payload()
    assert identity["subset_manifest_sha256"] == train_fingerprint
    assert identity["sampling_dataset_id"] == engine_sampling_dataset_id(identity)
