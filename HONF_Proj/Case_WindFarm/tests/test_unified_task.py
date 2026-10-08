from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from honf_runtime.unified_training import SamplingKey, TaskBatch

import windfarm.training.unified_task as unified_task_module
from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline
from windfarm.shared_interaction import WindFarmRefinedInteractionModel
from windfarm.splits import make_group_split
from windfarm.training.unified_formal import WindFormalRefinementTask
from windfarm.training.unified_task import (
    DEFAULT_ROLE_QUERY_COUNTS,
    WIND_GATE_COMPACT_C1_TRANSITION,
    WIND_GATE_COMPACT_C1_VERSION,
    WIND_W0_RECIPE_ID,
    WIND_W1_RECIPE_ID,
    WIND_W2_RECIPE_ID,
    WIND_W3_RECIPE_ID,
    WindReceiverInputs,
    WindRefinementTask,
    WindTargets,
    _as_scene_batch,
    _fit_train_role_scales,
    _query_sampling_sha256,
    _read_fixed_rows,
    _role_ids,
    _scene_inputs,
    _stable_json_sha256,
    resolve_wind_recipe,
)


@pytest.mark.parametrize(
    ("recipe_id", "query_count", "objective", "hidden"),
    [
        (WIND_W0_RECIPE_ID, 1024, "scalar_role", 64),
        (WIND_W1_RECIPE_ID, 1024, "component_role", 64),
        (WIND_W2_RECIPE_ID, 4096, "component_role", 64),
        (WIND_W3_RECIPE_ID, 1024, "component_role", 128),
    ],
)
def test_versioned_wind_recipes_seal_exact_role_mixture_and_sampler(recipe_id, query_count, objective, hidden):
    recipe = resolve_wind_recipe(recipe_id)
    assert recipe is not None
    assert recipe["query_count"] == query_count
    assert sum(recipe["role_query_counts"].values()) == query_count
    assert recipe["objective"] == objective
    assert recipe["dataset_profile"] == "wind_shared_fixed24_v1"
    assert recipe["sampling_version"] == SamplingKey.CASE_EPOCH_VERSION
    assert recipe["model"]["hidden"] == hidden
    assert recipe["model"]["message"] == hidden
    assert recipe["model"]["environment_token_shape"] == [2, 2, 2]
    if query_count == 4096:
        assert recipe["role_query_counts"] == {
            name: 4 * count for name, count in DEFAULT_ROLE_QUERY_COUNTS.items()
        }


def test_wind_recipe_rejects_unsealed_query_counts_and_legacy_profile_is_unchanged():
    recipe = resolve_wind_recipe(WIND_W2_RECIPE_ID)
    assert recipe is not None
    recipe["role_query_counts"]["background"] += 1
    recipe["recipe_sha256"] = _stable_json_sha256(
        {key: value for key, value in recipe.items() if key != "recipe_sha256"}
    )
    with pytest.raises(ValueError, match="canonical mixture"):
        resolve_wind_recipe(recipe=recipe)
    assert unified_task_module.resolve_wind_recipe() is None


def _compact_gate_recipe(recipe_id: str = WIND_W1_RECIPE_ID) -> dict:
    recipe = resolve_wind_recipe(recipe_id)
    assert recipe is not None
    recipe.pop("recipe_sha256")
    recipe["gate_version"] = WIND_GATE_COMPACT_C1_VERSION
    recipe["gate_transition"] = list(WIND_GATE_COMPACT_C1_TRANSITION)
    recipe["recipe_sha256"] = _stable_json_sha256(recipe)
    return recipe


def test_compact_c1_wind_recipe_is_opt_in_strict_and_hash_bound():
    legacy = resolve_wind_recipe(WIND_W1_RECIPE_ID)
    assert legacy is not None
    assert "gate_version" not in legacy and "gate_transition" not in legacy

    compact = resolve_wind_recipe(recipe=_compact_gate_recipe())
    assert compact is not None
    assert compact["gate_version"] == WIND_GATE_COMPACT_C1_VERSION
    assert compact["gate_transition"] == [0.35, 0.65]
    assert compact["recipe_sha256"] != legacy["recipe_sha256"]

    missing_transition = _compact_gate_recipe()
    missing_transition.pop("gate_transition")
    missing_transition.pop("recipe_sha256")
    with pytest.raises(ValueError, match="requires its sealed gate_transition"):
        resolve_wind_recipe(recipe=missing_transition)

    altered_transition = _compact_gate_recipe()
    altered_transition["gate_transition"] = [0.3, 0.7]
    altered_transition["recipe_sha256"] = _stable_json_sha256(
        {key: value for key, value in altered_transition.items() if key != "recipe_sha256"}
    )
    with pytest.raises(ValueError, match="sealed to the TRAIN-diagnosed"):
        resolve_wind_recipe(recipe=altered_transition)


def test_compact_c1_provider_forwards_gate_without_adding_legacy_identity_fields():
    provider = _provider(_compact_gate_recipe())
    provider._message_scale = 1.0
    case = _case()
    scene = _as_scene_batch((_scene_inputs(case),), torch.device("cpu"))
    receivers = WindReceiverInputs(
        coordinates_D=np.asarray([[[0.5, 0.2, 0.875], [4.0, 1.0, 1.1]]], dtype=np.float32),
        role_ids=np.asarray([[0, 1]], dtype=np.int8),
    )
    model = WindFarmRefinedInteractionModel(
        velocity_transform=_normalizer(), background_profile=_profile(), hidden=16, message=12
    )
    model.eval()
    provider.predict_native(model, scene, receivers, "adaptive_detail", "hard", epoch=900)
    assert model.core.refinement_policy.gate_version == WIND_GATE_COMPACT_C1_VERSION
    assert model.core.refinement_policy.gate_transition == WIND_GATE_COMPACT_C1_TRANSITION


def test_component_objective_changes_training_denominator_but_common_validation_stays_scalar():
    recipe = resolve_wind_recipe(WIND_W1_RECIPE_ID)
    assert recipe is not None
    provider = _provider(recipe)
    provider.role_scales = {name: 2.0 for name in DEFAULT_ROLE_QUERY_COUNTS}
    provider._message_scale = 1.0
    predictions = unified_task_module.WindPredictions(
        main_mps=torch.tensor([[[1.0, 2.0, 3.0]]]),
        full_mps=None,
        role_ids=torch.tensor([[0]], dtype=torch.int64),
        auxiliary={},
        execution_mode="full",
    )
    targets = WindTargets(
        velocity_mps=np.zeros((1, 1, 3), dtype=np.float32),
        row_indices=(0,),
        case_metadata=(),
    )
    terms = provider.loss_terms(
        predictions,
        targets,
        "warmup",
        {"base_numerator": torch.tensor(0.0), "base_denominator": 1.0},
    )
    assert terms["native_role/volume"].numerator.item() == pytest.approx(3.0)
    assert terms["native_role/volume"].denominator == 3.0
    assert provider.validation_role_query_counts == DEFAULT_ROLE_QUERY_COUNTS

    five_role_predictions = unified_task_module.WindPredictions(
        main_mps=torch.tensor([[[1.0, 2.0, 3.0]] * 5]),
        full_mps=None,
        role_ids=torch.arange(5, dtype=torch.int64)[None],
        auxiliary={"baseline_mps": torch.zeros((1, 5, 3))},
        execution_mode="full",
    )
    metrics = provider.validation_metrics(
        five_role_predictions,
        WindTargets(
            velocity_mps=np.zeros((1, 5, 3), dtype=np.float32),
            row_indices=(0,),
            case_metadata=({"row_index": 0},),
        ),
        {},
    )
    assert metrics["rows"][0]["field_score"] == pytest.approx(7.0 / 6.0)
    assert all(metrics["rows"][0]["roles"][role]["role_scale_mps"] == 2.0 for role in DEFAULT_ROLE_QUERY_COUNTS)


def test_wind_scale_calibration_keeps_w0_variance_and_w1_profile_residual_rules(monkeypatch):
    from windfarm.workflows.joint_forward import ROLE_NAMES

    view = _View()

    def deterministic_samples(case, rng, role_query_counts, *, catalogue_cache, rng_by_role):
        del rng, catalogue_cache, rng_by_role
        counts = dict(role_query_counts)
        values = []
        slices = {}
        start = 0
        for role in ROLE_NAMES:
            count = counts[role]
            # A constant target keeps W0's source-derived variance at its 1e-3 m/s floor.
            values.append(np.tile(np.asarray([5.0, 0.2, 0.0], dtype=np.float32), (count, 1)))
            slices[role] = slice(start, start + count)
            start += count
        target = np.concatenate(values, axis=0)
        return SimpleNamespace(
            coordinates_D=np.zeros((start, 3), dtype=np.float32),
            target_mps=target,
            flat_indices=np.arange(start, dtype=np.int64),
            role_slices=slices,
        )

    class Cache:
        def get(self, case):
            del case
            return None

        def summary(self):
            return {"hits": 0, "misses": 0, "cached_bytes": 0}

    monkeypatch.setattr(unified_task_module, "sample_native_role_queries", deterministic_samples)
    result = _fit_train_role_scales(
        view,  # type: ignore[arg-type]
        np.arange(18, dtype=np.int64),
        tuple(range(6)),
        _profile(),
        _normalizer(),
        training_fingerprint="fixed-train-membership",
        catalogue_cache=Cache(),  # type: ignore[arg-type]
    )
    assert result["calibration_row_indices"] == list(range(12))
    assert result["role_query_counts_per_row"] == DEFAULT_ROLE_QUERY_COUNTS
    for role in ROLE_NAMES:
        assert result["scalar_role_scales_mps"][role] == pytest.approx(1.0e-3)
        # The profile at z=0 is [4, 0.1, -0.2]; residual RMS includes the signed mean.
        assert result["component_role_scales_mps"][role] == pytest.approx([1.0, 0.1, 0.2])
    assert result["physical_component_floor_mps"] == pytest.approx(0.009)


def _normalizer() -> VelocityNormalizer:
    return VelocityNormalizer(
        mean=np.asarray([1.0, 0.0, 0.0]),
        std=np.asarray([0.2, 0.1, 0.1]),
        safe_std=np.asarray([0.2, 0.1, 0.1]),
        source_rows=72,
    )


def _profile() -> VerticalProfileBaseline:
    return VerticalProfileBaseline(
        bin_centers_D=np.asarray([0.0, 2.0], dtype=np.float64),
        values_mps=np.asarray([[4.0, 0.1, -0.2], [12.0, 0.5, 0.2]], dtype=np.float64),
        counts=np.asarray([100, 100], dtype=np.int64),
        z_min_D=0.0,
        z_max_D=2.0,
    )


def _case(row: int = 0, layout: int = 0, direction: float = 270.0):
    from windfarm.geometry import environment_representation, global_geometry_features, support_geometry

    support = support_geometry(
        np.asarray([-400.0, 0.0, 800.0], dtype=np.float32),
        np.asarray([-320.0, 0.0, 320.0], dtype=np.float32),
        np.asarray([20.0, 70.0, 130.0], dtype=np.float32),
    )
    environment = environment_representation(support, token_shape=(2, 2, 2))
    centers = np.asarray([[0.0, 0.0, 0.875], [3.0, 0.5, 0.875]], dtype=np.float32)
    return SimpleNamespace(
        index=row,
        case=f"gen_{row:04d}_wd{int(direction)}",
        layout=f"layout-{layout}",
        layout_index=layout,
        wind_direction_deg=direction,
        n_turbines=2,
        support=support,
        module_centers=centers,
        module_present=np.ones(2, dtype=np.float32),
        module_features=np.asarray([[0.5, 0.875], [0.5, 0.875]], dtype=np.float32),
        global_context=global_geometry_features(support, direction, 2),
        env_coords=environment.coords_D,
        env_features=environment.features,
        env_weights=environment.weights_D3,
    )


class _View:
    def __init__(self):
        self.metadata = {
            "layout_index": np.repeat(np.arange(14, dtype=np.int64), 3),
            "wd_deg": np.tile(np.asarray([270.0, 285.0, 300.0]), 14),
        }

    def run(self, row: int):
        return _case(row, int(self.metadata["layout_index"][row]), float(self.metadata["wd_deg"][row]))


def _provider(recipe=None) -> WindRefinementTask:
    view = _View()
    train_rows = np.arange(0, 18, dtype=np.int64)
    validation_rows = np.arange(18, 42, dtype=np.int64)
    manifest = {
        "subset_id": "wind_shared_fixed24_v1",
        "manifest_sha256": "test-manifest-sha",
        "train_layout_indices": list(range(6)),
        "validation_layout_indices": list(range(6, 14)),
        "train_layout_selection": {
            "selected_turbine_counts": [6, 7, 8, 9],
            "uncovered_turbine_counts": [18],
        },
    }
    kwargs = {}
    if recipe is not None:
        kwargs = {
            "role_query_counts": dict(recipe["role_query_counts"]),
            "role_component_scales": {name: (1.0, 2.0, 3.0) for name in DEFAULT_ROLE_QUERY_COUNTS},
            "role_scale_calibration": {"training_rows_sha256": "test-train"},
            "recipe": recipe,
            "sampling_dataset_fingerprint": manifest["manifest_sha256"],
        }
    return WindRefinementTask(
        view,  # type: ignore[arg-type]
        train_rows=train_rows,
        validation_rows=validation_rows,
        normalizer=_normalizer(),
        background_profile=_profile(),
        manifest=manifest,
        manifest_path=Path("/tmp/fixed_subset_manifest.json"),
        normalization_path=Path("/tmp/train_only_normalization.json"),
        role_scales={name: 1.0 for name in DEFAULT_ROLE_QUERY_COUNTS},
        role_scale_sha256="test-role-scale-sha",
        **kwargs,
    )


def _fixed_manifest(groups: np.ndarray, directions: np.ndarray) -> tuple[dict, object]:
    split = make_group_split(groups, seed=42)
    train_layouts = np.sort(np.unique(groups[split.train]))[:24]
    validation_layouts = np.sort(np.unique(groups[split.validation]))[:8]
    train_rows = np.flatnonzero(np.isin(groups, train_layouts)).astype(np.int64)
    validation_rows = np.flatnonzero(np.isin(groups, validation_layouts)).astype(np.int64)

    def selection(rows: np.ndarray, layouts: np.ndarray) -> dict:
        return {
            "used_target_values_for_selection": False,
            "row_indices": rows.tolist(),
            "layout_indices": layouts.tolist(),
        }

    payload = {
        "subset_id": "wind_shared_fixed24_v1",
        "used_targets_for_subset_selection": False,
        "source_split": {
            "seed": 42,
            "group_key": "layout_index",
            "test_target_values_read": False,
            "train_row_indices_sha256": split.metadata["partitions"]["train"]["row_indices_sha256"],
            "validation_row_indices_sha256": split.metadata["partitions"]["validation"]["row_indices_sha256"],
            "original_train_layout_count": split.metadata["partitions"]["train"]["groups"],
            "original_validation_layout_count": split.metadata["partitions"]["validation"]["groups"],
            "original_test_layout_count": split.metadata["partitions"]["test"]["groups"],
        },
        "train_row_indices": train_rows.tolist(),
        "validation_row_indices": validation_rows.tolist(),
        "train_row_count": int(train_rows.size),
        "validation_row_count": int(validation_rows.size),
        "direction_categories": [270.0, 285.0, 300.0],
        "train_layout_indices": train_layouts.tolist(),
        "validation_layout_indices": validation_layouts.tolist(),
        "train_layout_selection": selection(train_rows, train_layouts),
        "validation_layout_selection": selection(validation_rows, validation_layouts),
    }
    payload["manifest_sha256"] = _stable_json_sha256(payload)
    return payload, split


def test_fixed_wind_manifest_binds_source_splits_and_all_direction_rows(tmp_path: Path):
    groups = np.repeat(np.arange(60, dtype=np.int64), 3)
    directions = np.tile(np.asarray([270.0, 285.0, 300.0]), 60)
    manifest, split = _fixed_manifest(groups, directions)
    view = SimpleNamespace(metadata={"layout_index": groups, "wd_deg": directions})
    path = tmp_path / "fixed_subset_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    _, train_rows, validation_rows = _read_fixed_rows(view, split, path)  # type: ignore[arg-type]
    assert train_rows.shape == (72,)
    assert validation_rows.shape == (24,)

    corrupted = dict(manifest)
    corrupted["train_layout_selection"] = dict(manifest["train_layout_selection"])
    corrupted["train_layout_selection"]["row_indices"] = list(reversed(train_rows.tolist()))
    corrupted.pop("manifest_sha256")
    corrupted["manifest_sha256"] = _stable_json_sha256(corrupted)
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="nested layout/row inventory"):
        _read_fixed_rows(view, split, path)  # type: ignore[arg-type]


def test_query_receipt_binds_native_cells_but_not_matched_arm():
    sample = SimpleNamespace(
        coordinates_D=np.zeros((1024, 3), dtype=np.float32),
        flat_indices=np.arange(1024, dtype=np.int64),
        role_sample_counts=DEFAULT_ROLE_QUERY_COUNTS,
    )
    full_key = SamplingKey(42, 10, 2, 3, "warmup", "full_detail")
    adaptive_key = SamplingKey(42, 10, 2, 3, "warmup", "adaptive_detail")
    receipt = _query_sampling_sha256((6,), (sample,), full_key)
    assert len(receipt) == 64
    assert _query_sampling_sha256((6,), (sample,), adaptive_key) == receipt
    changed = SimpleNamespace(
        coordinates_D=np.ones((1024, 3), dtype=np.float32),
        flat_indices=sample.flat_indices,
        role_sample_counts=DEFAULT_ROLE_QUERY_COUNTS,
    )
    assert _query_sampling_sha256((6,), (changed,), full_key) != receipt


def test_wind_predictor_does_not_claim_an_unpaid_adaptive_full_replay():
    torch.manual_seed(8)
    provider = _provider()
    case = _case()
    scene = _as_scene_batch((_scene_inputs(case),), torch.device("cpu"))
    receivers = WindReceiverInputs(
        coordinates_D=np.asarray([[[0.5, 0.2, 0.875], [4.0, 1.0, 1.1]]], dtype=np.float32),
        role_ids=np.asarray([[0, 1]], dtype=np.int8),
    )
    model = WindFarmRefinedInteractionModel(
        velocity_transform=_normalizer(), background_profile=_profile(), hidden=16, message=12
    )
    model.eval()

    adaptive, adaptive_state = provider.predict_native(
        model, scene, receivers, "adaptive_detail", "soft", epoch=700, temperature=0.55
    )
    assert adaptive.full_mps is None
    assert adaptive_state["phase"] == "hard"
    assert adaptive_state["temperature"] == pytest.approx(0.55)

    full, _ = provider.predict_native(model, scene, receivers, "full_detail", "soft", epoch=700)
    assert full.full_mps is full.main_mps


def test_wind_stage_hook_accepts_and_checks_shared_engine_temperature():
    provider = _provider()
    receipt = provider.on_phase_start(
        object(), phase="warmup", epoch=1, arm="warmup", temperature=1.0
    )
    assert receipt["temperature"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="shared engine schedule"):
        provider.on_phase_start(
            object(), phase="soft", epoch=601, arm="adaptive_detail", temperature=0.5
        )


def test_wind_cost_calibration_is_train_only_bounded_and_checkpointable():
    torch.manual_seed(11)
    provider = _provider()
    case = _case()
    scene_inputs = _scene_inputs(case)
    coordinates = np.resize(
        np.asarray([[0.5, 0.2, 0.875], [4.0, 1.0, 1.1], [2.0, -1.0, 2.5]], dtype=np.float32),
        (1024, 3),
    )[None]
    roles = _role_ids(DEFAULT_ROLE_QUERY_COUNTS)[None]
    targets = WindTargets(
        velocity_mps=np.zeros((1, 1024, 3), dtype=np.float32),
        row_indices=(0,),
        case_metadata=({"row_index": 0, "layout_index": 0, "wind_direction_deg": 270.0},),
    )
    batch = TaskBatch(
        scene_inputs=(scene_inputs,),
        receivers=WindReceiverInputs(coordinates, roles),
        targets=targets,
        case_keys=(0,),
    )
    provider.message_calibration_rows = (0,)
    provider._message_scale = 0.2
    provider._expected_work_calibration_batches = lambda: (batch,)  # type: ignore[method-assign]
    model = WindFarmRefinedInteractionModel(
        velocity_transform=_normalizer(), background_profile=_profile(), hidden=16, message=12
    )
    model.core.refinement.residual_scale.fill_(0.2)

    receipt = provider.calibrate_expected_work_weight(model)
    assert receipt["row_indices"] == [0]
    assert receipt["validation_values_read"] is False
    assert 0.0 <= receipt["coefficient"] <= 0.1
    assert math_isfinite(receipt["router_native_gradient_norm"])
    state = provider.training_state_dict()
    restored = _provider()
    restored.load_training_state_dict(state)
    assert restored.training_state_dict() == state


def test_wind_checkpoint_restore_can_replace_only_device_local_prefit_rms():
    saved = _provider()
    saved._message_scale = 0.2
    saved._message_scale_calibration = {
        "method": "pre-fit fine source-message RMS",
        "layout_indices": [0, 1, 2, 3],
        "row_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "query_count_per_row": 1024,
        "query_coordinates_sha256": "a" * 64,
        "active_message_component_count": 24,
        "message_rms_latent": 0.2,
        "minimum_scale": 1.0e-6,
        "target_values_read": False,
    }
    checkpoint_state = saved.training_state_dict()

    cpu_provider = _provider()
    cpu_provider._message_scale = 0.20000000025
    cpu_provider._message_scale_calibration = {
        **saved._message_scale_calibration,
        "message_rms_latent": cpu_provider._message_scale,
    }
    with pytest.raises(ValueError, match="active fixed TRAIN calibration"):
        cpu_provider.load_training_state_dict(checkpoint_state)

    cpu_provider.load_training_state_dict(
        checkpoint_state, allow_prefit_calibration_replace=True
    )
    assert cpu_provider._message_scale == checkpoint_state["message_scale"]
    assert cpu_provider._message_scale_calibration == checkpoint_state["message_scale_calibration"]


def test_wind_prefit_calibration_replace_rejects_different_query_panel():
    saved = _provider()
    saved._message_scale = 0.2
    saved._message_scale_calibration = {
        "method": "pre-fit fine source-message RMS",
        "layout_indices": [0, 1, 2, 3],
        "row_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "query_count_per_row": 1024,
        "query_coordinates_sha256": "a" * 64,
        "active_message_component_count": 24,
        "message_rms_latent": 0.2,
        "minimum_scale": 1.0e-6,
        "target_values_read": False,
    }
    checkpoint_state = saved.training_state_dict()
    changed_provider = _provider()
    changed_provider._message_scale = 0.21
    changed_provider._message_scale_calibration = {
        **saved._message_scale_calibration,
        "query_coordinates_sha256": "b" * 64,
        "message_rms_latent": 0.21,
    }
    with pytest.raises(ValueError, match="same pre-fit TRAIN calibration"):
        changed_provider.load_training_state_dict(
            checkpoint_state, allow_prefit_calibration_replace=True
        )


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(float(value)))


@pytest.mark.parametrize(
    "device_name",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda",
            id="cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable."),
        ),
    ],
)
def test_open_router_importance_denominator_matches_live_read_on_provider_device(
    monkeypatch, device_name: str
):
    """The TRAIN denominator must count the exact FP32 pairs used by the readout."""
    from windfarm.geometry import global_geometry_features

    device = torch.device(device_name)
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    provider = _provider()
    provider.device = device

    original_run = provider.view.run
    boundary_center = np.asarray([6.1097913, 6.1475134, 0.875], dtype=np.float32)

    def mixed_source_count_case(row: int):
        case = original_run(row)
        case.module_centers[0] = boundary_center
        if row == 0:
            # A one-source row is padded to the two-source batch capacity.
            case.module_centers = case.module_centers[:1].copy()
            case.module_present = case.module_present[:1].copy()
            case.module_features = case.module_features[:1].copy()
            case.n_turbines = 1
            case.global_context = global_geometry_features(
                case.support, case.wind_direction_deg, case.n_turbines
            )
        return case

    monkeypatch.setattr(provider.view, "run", mixed_source_count_case)

    near_below = np.nextafter(np.float32(4.0), np.float32(0.0))
    near_above = np.nextafter(np.float32(4.0), np.float32(np.inf))
    coordinates = np.repeat(np.asarray([[18.0, 18.0, 0.875]], dtype=np.float32), 1024, axis=0)
    coordinates[:6] = np.asarray(
        [
            [near_below, 0.0, 0.875],
            [4.0, 0.0, 0.875],
            [near_above, 0.0, 0.875],
            [near_below, 0.0, 0.875],
            [4.0, 0.0, 0.875],
            [near_above, 0.0, 0.875],
        ],
        dtype=np.float32,
    ) + boundary_center - np.asarray([0.0, 0.0, 0.875], dtype=np.float32)
    # Native e519 boundary pair: CPU near_weight=0, GPU2=2**-24.
    # This made the old CPU prepass differ from the live CUDA denominator.
    coordinates[6] = np.asarray([6.0283365, 4.9475636, 4.689722], dtype=np.float32)
    coordinates[7] = boundary_center

    def fake_native_sample(case, rng, role_query_counts, *, catalogue_cache):
        del rng, catalogue_cache
        return SimpleNamespace(
            coordinates_D=coordinates.copy(),
            # Deliberately target-independent; denominator construction must
            # use only the sealed scene and the sampled physical queries.
            target_mps=np.zeros((1024, 3), dtype=np.float32),
            flat_indices=np.arange(1024, dtype=np.int64),
            role_sample_counts=dict(role_query_counts),
        )

    monkeypatch.setattr(unified_task_module, "sample_native_role_queries", fake_native_sample)
    batch = provider.make_batch(
        (0, 1), SamplingKey(42, 1, 0, 0, "open", "adaptive_detail")
    )
    assert batch.scene_inputs[0].module_centers.shape == (1, 3)
    assert batch.scene_inputs[1].module_centers.shape == (2, 3)

    near_calls: list[tuple[torch.device, int]] = []
    original_near_weight = unified_task_module.InteractionContextCore.near_weight

    def record_near_weight(receivers, centers, source_lengths, present):
        near_calls.append((receivers.device, int(receivers.shape[1])))
        return original_near_weight(receivers, centers, source_lengths, present)

    monkeypatch.setattr(
        unified_task_module.InteractionContextCore,
        "near_weight",
        staticmethod(record_near_weight),
    )
    denominators = provider.loss_denominators(
        (batch,), phase="open", arm="adaptive_detail"
    )
    # The predictor's protected-locality calculation uses the same device and
    # bounded receiver reads. Comparing only the final count would not catch a
    # regression that silently moves this prepass back to CPU or processes all
    # 1024 receivers in one allocation.
    assert near_calls == [(device, 512), (device, 512)]

    model = WindFarmRefinedInteractionModel(
        velocity_transform=_normalizer(),
        background_profile=_profile(),
        hidden=16,
        message=12,
    ).to(device)
    model.train()
    scene = provider.make_scene(batch.scene_inputs)
    assert torch.equal(
        scene.present.detach().cpu(), torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    )
    _, auxiliary = provider.predict_native(
        model,
        scene,
        batch.receivers,
        execution_mode="adaptive_detail",
        phase="open",
        epoch=501,
        temperature=1.0,
    )
    actual = int(auxiliary["router_importance_denominator"].detach().item())
    predicted = int(denominators["router_importance_loss"])
    assert 0 < predicted < 1024 * 3
    assert actual == predicted


def test_formal_wind_task_assembles_eight_case_batch_with_separate_limit(monkeypatch):
    provider = _provider()
    sampled_rows: list[int] = []

    def fake_native_sample(case, rng, role_query_counts, *, catalogue_cache):
        del rng, catalogue_cache
        sampled_rows.append(int(case.index))
        return SimpleNamespace(
            coordinates_D=np.full((1024, 3), float(case.index), dtype=np.float32),
            target_mps=np.full((1024, 3), float(case.index + 1), dtype=np.float32),
            flat_indices=np.arange(1024, dtype=np.int64),
            role_sample_counts=dict(role_query_counts),
        )

    monkeypatch.setattr(unified_task_module, "sample_native_role_queries", fake_native_sample)
    key = SamplingKey(42, 1, 0, 0, "warmup", "full_detail")
    with pytest.raises(ValueError, match="limit of 4"):
        provider.make_batch(tuple(range(5)), key)

    formal = WindFormalRefinementTask(
        provider.view,
        train_rows=provider.train_rows,
        validation_rows=provider.validation_rows,
        normalizer=provider.normalizer,
        background_profile=provider.background_profile,
        manifest=provider.manifest,
        manifest_path=provider.manifest_path,
        normalization_path=provider.normalization_path,
        role_scales=provider.role_scales,
        role_scale_sha256=provider.role_scale_sha256,
        formal_recipe={
            "recipe_sha256": "formal-recipe-sha",
            "source_metadata_sha256": "metadata-sha",
            "source_file_inventory": [],
            "partition_identity": {},
            "normalization_sha256": "normalization-sha",
            "normalization_binding_sha256": "normalization-binding-sha",
            "role_scale_sha256": "role-scale-sha",
        },
    )
    batch = formal.make_batch(tuple(range(8)), key)
    assert batch.case_keys == tuple(range(8))
    assert batch.receivers.coordinates_D.shape == (8, 1024, 3)
    assert batch.targets.velocity_mps.shape == (8, 1024, 3)
    assert batch.targets.row_indices == tuple(range(8))
    assert sampled_rows == list(range(8))

    with pytest.raises(ValueError, match="limit of 24"):
        formal.make_batch(tuple(range(25)), key)
    with pytest.raises(ValueError, match="outside the sealed"):
        formal.make_batch((18,), key)
    with pytest.raises(ValueError, match="cannot repeat"):
        formal.make_batch((0, 0), key)
    assert sampled_rows == list(range(8))

    monkeypatch.setattr(unified_task_module, "_sha256", lambda _path: "fixed-normalization-sha")
    provider.view.token_shape = (2, 2, 2)
    provider._message_scale = 1.0
    formal._message_scale = 1.0
    assert "max_microbatch_cases" not in provider.identity_payload()
    assert "max_microbatch_cases" not in formal.identity_payload()
    assert "gate_version" not in provider.identity_payload()
    compact_provider = _provider(_compact_gate_recipe())
    compact_provider.view.token_shape = (2, 2, 2)
    compact_provider._message_scale = 1.0
    compact_identity = compact_provider.identity_payload()
    assert compact_identity["resolved_recipe"]["gate_version"] == WIND_GATE_COMPACT_C1_VERSION
    assert compact_identity["resolved_recipe"]["gate_transition"] == [0.35, 0.65]
