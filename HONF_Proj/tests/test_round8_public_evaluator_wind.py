from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import content_hypergraph_evaluate as evaluator
import numpy as np
import pytest


class _WindProvider:
    VOLUME_ROLE_ID = 0
    HUB_BAND_ROLE_ID = 1
    VOLUME_QUERIES = 6144
    HUB_BAND_QUERIES = 2048
    VALIDATION_VOLUME_QUERIES = 8192
    VALIDATION_HUB_BAND_QUERIES = 2048
    VOLUME_WEIGHT = 0.75
    HUB_BAND_WEIGHT = 0.25
    VALIDATION_VOLUME_WEIGHT = 0.8
    VALIDATION_HUB_BAND_WEIGHT = 0.2

    def __init__(self) -> None:
        self.adapter = SimpleNamespace(
            normalizer=SimpleNamespace(
                safe_std=np.asarray([2.0, 3.0, 4.0]),
                u_ref_mps=2.0,
            )
        )

    def native_evaluation_rows(self, split: str, case_keys: list[int] | None) -> tuple[int, ...]:
        if case_keys is not None:
            return tuple(case_keys)
        if split == "train":
            return (7, 13)
        return (101, 109)

    def make_scene(self, scene_inputs: object) -> object:
        return scene_inputs

    def predict_native(self, model: object, scene: object, receivers: object, *args: object, **kwargs: object):
        del model, scene, args, kwargs
        prediction = SimpleNamespace(
            main_mps=np.zeros((1, receivers.role_ids.shape[1], 3), dtype=np.float64),
            role_ids=receivers.role_ids.copy(),
        )
        return prediction, {}

    def native_evaluation_batches(self, split: str, case_keys: list[int] | None,
                                  *, microbatch_size: int | None = None):
        del case_keys, microbatch_size
        return [self.train_batch if split == "train" else self.validation_batch]

    def reduce_native_evaluation_metrics(self, records: object, *, split: str,
                                         case_keys: list[int] | None):
        del records, split, case_keys
        return {
            "query_count_per_row": 8192,
            "endpoint_rows": [5],
            "endpoint_selection": "legacy reducer query-count default",
            "rows": [{"row_index": 5, "val_weighted_mse": 0.25}],
        }

    def work_counts(self, batch: object, prediction: object, auxiliary: object) -> dict[str, int]:
        del batch, prediction, auxiliary
        return {"test_counter": 1}

    def validation_metrics(self, predictions: object, targets: object, auxiliary: object):
        del predictions, targets, auxiliary
        return {"rows": [{"row_index": 5, "val_weighted_mse": 0.25}]}


def _round8_identity() -> tuple[dict[str, object], dict[str, object]]:
    recipe: dict[str, object] = {
        "schema": "native_budgeted_dispatch_recipe_v2",
        "family": "native_budgeted_dispatch_v2",
        "model_family": "native_budgeted_dispatch_v2",
        "workflow": "native_budgeted_dispatch_joint_v2",
        "dataset": "wind",
        "architecture": "A2-D",
    }
    payload: dict[str, object] = {
        "workflow": "native_budgeted_dispatch_joint_v2",
        "experiment_identity": {
            "family": "native_budgeted_dispatch_v2",
            "model_family": "native_budgeted_dispatch_v2",
            "workflow": "native_budgeted_dispatch_joint_v2",
            "provider_identity": {"test_fixture": "synthetic"},
        },
    }
    return payload, recipe


def _train_batch(role_ids: np.ndarray | None = None) -> tuple[_WindProvider, object, object]:
    provider = _WindProvider()
    expected_roles = np.concatenate((np.zeros(6144, dtype=np.int8), np.ones(2048, dtype=np.int8)))[None]
    if role_ids is not None:
        expected_roles = role_ids
    case_keys = (5,)
    targets = SimpleNamespace(
        row_indices=np.asarray(case_keys),
        case_metadata=({
            "row_index": 5,
            "volume_query_count": 6144,
            "hub_band_query_count": 2048,
        },),
        velocity_mps=np.zeros((1, 8192, 3), dtype=np.float64),
    )
    receivers = SimpleNamespace(role_ids=expected_roles)
    batch = SimpleNamespace(case_keys=case_keys, targets=targets, receivers=receivers)
    predictions = SimpleNamespace(
        main_mps=np.zeros((1, 8192, 3), dtype=np.float64),
        role_ids=expected_roles.copy(),
    )
    targets.velocity_mps[:, :6144] = -np.asarray([4.0, 6.0, 8.0])
    targets.velocity_mps[:, 6144:] = -np.asarray([8.0, 6.0, 4.0])
    return provider, batch, predictions


def _endpoint_batch(volume: int, hub_band: int) -> object:
    role_ids = np.concatenate((
        np.zeros(volume, dtype=np.int8), np.ones(hub_band, dtype=np.int8),
    ))[None]
    queries = volume + hub_band
    targets = SimpleNamespace(
        row_indices=np.asarray([5]),
        case_metadata=({
            "row_index": 5,
            "volume_query_count": volume,
            "hub_band_query_count": hub_band,
        },),
        velocity_mps=np.zeros((1, queries, 3), dtype=np.float64),
    )
    return SimpleNamespace(
        case_keys=(5,), targets=targets,
        receivers=SimpleNamespace(role_ids=role_ids), scene_inputs=object(), auxiliary={},
    )


def _round8_cli_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> tuple[_WindProvider, object]:
    provider = _WindProvider()
    provider.train_batch = _endpoint_batch(6144, 2048)
    provider.validation_batch = _endpoint_batch(8192, 2048)
    payload, recipe = _round8_identity()
    identity = payload["experiment_identity"]
    identity["recipe"] = recipe  # type: ignore[index]
    payload["epoch"] = 2500
    monkeypatch.setattr(evaluator, "load_evaluation_components", lambda checkpoint, device: (object(), provider, payload))
    checkpoint = tmp_path / "round8_wind.pt"  # type: ignore[operator]
    checkpoint.write_bytes(b"synthetic sealed checkpoint marker")
    return provider, checkpoint


def test_round8_wind_dispatch_uses_saved_workflow_and_family_only() -> None:
    payload, recipe = _round8_identity()
    assert evaluator._is_round8_wind_payload(payload, recipe)
    old_payload = {**payload, "workflow": "native_retrieval_joint_v1"}
    assert not evaluator._is_round8_wind_payload(old_payload, recipe)

    wrong_family = dict(payload)
    wrong_family["experiment_identity"] = {
        **payload["experiment_identity"],  # type: ignore[arg-type]
        "family": "native_retrieval_v1",
    }
    with pytest.raises(ValueError, match="exact saved family and workflow"):
        evaluator._is_round8_wind_payload(wrong_family, recipe)


def test_round8_wind_train_metrics_use_source_train_weights_and_namespace() -> None:
    provider, batch, predictions = _train_batch()
    role_counts = evaluator._round8_wind_role_counts(provider, batch, "train")
    assert role_counts == {"volume": 6144, "hub_band": 2048}
    metrics = evaluator._round8_wind_train_metrics(predictions, batch.targets, provider, batch)
    row = metrics["rows"][0]
    assert row["train_volume_mse"] == pytest.approx(1.0)
    assert row["train_hub_band_mse"] == pytest.approx(1.75)
    assert row["train_weighted_mse"] == pytest.approx(1.1875)
    assert not any(name.startswith("val_") for name in row)


def test_round8_wind_train_rejects_changed_role_membership_and_order() -> None:
    roles = np.concatenate((np.ones(2048, dtype=np.int8), np.zeros(6144, dtype=np.int8)))[None]
    provider, batch, _ = _train_batch(roles)
    with pytest.raises(ValueError, match="exact native role membership/order"):
        evaluator._round8_wind_role_counts(provider, batch, "train")


def test_round8_wind_train_rejects_jointly_swapped_declared_role_ids() -> None:
    roles = np.concatenate((np.ones(6144, dtype=np.int8), np.zeros(2048, dtype=np.int8)))[None]
    provider, batch, _ = _train_batch(roles)
    provider.VOLUME_ROLE_ID = 1
    provider.HUB_BAND_ROLE_ID = 0
    with pytest.raises(ValueError, match="volume=0 and hub-band=1"):
        evaluator._round8_wind_role_counts(provider, batch, "train")


def test_round8_wind_train_reducer_keeps_equal_case_scores_separate_from_dev() -> None:
    provider = _WindProvider()
    records = [
        {"rows": [{"row_index": 7, "train_volume_mse": 1.0, "train_hub_band_mse": 2.0,
                   "train_weighted_mse": 1.25}]},
        {"rows": [{"row_index": 13, "train_volume_mse": 3.0, "train_hub_band_mse": 4.0,
                   "train_weighted_mse": 3.25}]},
    ]
    metrics = evaluator._reduce_round8_wind_train_metrics(records, provider, (7, 13))
    assert metrics["train_weighted_mse"] == pytest.approx(2.25)
    assert metrics["train_endpoint_rows"] == [7, 13]
    assert metrics["role_query_counts"] == {"volume": 6144, "hub_band": 2048}
    assert "val_weighted_mse" not in metrics


def test_round8_wind_dev_metadata_reports_observed_q10240_correction() -> None:
    metrics = evaluator._round8_wind_dev_metadata(
        {"query_count_per_row": 8192, "val_weighted_mse": 0.3},
        {"volume": 8192, "hub_band": 2048},
    )
    assert metrics["query_count_per_row"] == 10240
    assert metrics["dev_objective_weights"] == {"volume": 0.8, "hub_band": 0.2}
    assert metrics["query_count_reducer_correction"]["inherited_reducer_value"] == 8192
    with pytest.raises(ValueError, match="Q10240"):
        evaluator._round8_wind_dev_metadata(
            {"query_count_per_row": 8192}, {"volume": 6144, "hub_band": 2048},
        )


def test_round8_wind_monitor_label_does_not_require_five_role_rows() -> None:
    metrics = evaluator._label_round8_wind_monitor(
        {"train_weighted_mse": 0.4, "train_endpoint_rows": [7, 13]},
        split="train",
        role_counts={"volume": 6144, "hub_band": 2048},
    )
    assert metrics["evaluation_panel"] == "fixed_checkpoint_monitor"
    assert metrics["monitor_rows"] == [7, 13]
    assert metrics["monitor_data_split"] == "train"
    assert metrics["monitor_score_semantics"] == "source TRAIN metric; not a DEV selector"
    assert metrics["query_count_per_row"] == 8192


def test_non_round8_monitor_labels_keep_the_existing_role_reduction() -> None:
    payload, recipe = _round8_identity()
    payload["workflow"] = "native_retrieval_joint_v1"
    assert not evaluator._is_round8_wind_payload(payload, recipe)
    original = {
        "query_count_per_row": 8192,
        "endpoint_rows": [101],
        "rows": [{"roles": {"volume": {"query_count": 8192}}}],
    }
    result = evaluator.label_monitor_metrics(original)
    assert result["query_count_per_row"] == 8192
    assert result["monitor_rows"] == [101]
    assert "endpoint_rows" not in result


def test_public_round8_wind_train_panel_uses_train_metric_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object,
) -> None:
    provider, checkpoint = _round8_cli_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(provider, "validation_metrics", lambda *args: pytest.fail("DEV metrics used for TRAIN"))
    output = tmp_path / "native_train"  # type: ignore[operator]
    assert evaluator.main([
        "--checkpoint", str(checkpoint), "--output", str(output), "--device", "cpu",
        "--split", "train", "--panel", "native", "--case-keys", "5",
    ]) == 0
    result = evaluator.json_safe(__import__("json").loads((output / "result.json").read_text()))
    metrics = result["metrics"]
    assert metrics["train_row_count"] == 1
    assert metrics["train_weighted_mse"] == pytest.approx(0.0)
    assert result["receiver_scope"].startswith("source TRAIN Q8192")


def test_public_round8_wind_dev_panel_corrects_legacy_q8192_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object,
) -> None:
    _, checkpoint = _round8_cli_fixture(monkeypatch, tmp_path)
    output = tmp_path / "dev"  # type: ignore[operator]
    assert evaluator.main([
        "--checkpoint", str(checkpoint), "--output", str(output), "--device", "cpu",
        "--split", "validation", "--panel", "native", "--case-keys", "5",
    ]) == 0
    result = __import__("json").loads((output / "result.json").read_text())
    assert result["metrics"]["query_count_per_row"] == 10240
    assert result["metrics"]["query_count_reducer_correction"]["inherited_reducer_value"] == 8192
    assert result["receiver_scope"].startswith("exposed DEV Q10240")


def test_public_round8_wind_monitor_labels_train_rows_without_role_metric_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object,
) -> None:
    provider, checkpoint = _round8_cli_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        evaluator, "fixed_monitor_batches", lambda *args: ([provider.train_batch], [5]),
    )
    monkeypatch.setattr(provider, "validation_metrics", lambda *args: pytest.fail("DEV metrics used for TRAIN monitor"))
    output = tmp_path / "monitor_train"  # type: ignore[operator]
    assert evaluator.main([
        "--checkpoint", str(checkpoint), "--output", str(output), "--device", "cpu",
        "--split", "train", "--panel", "monitor",
    ]) == 0
    result = __import__("json").loads((output / "result.json").read_text())
    metrics = result["metrics"]
    assert metrics["evaluation_panel"] == "fixed_checkpoint_monitor"
    assert metrics["monitor_rows"] == [5]
    assert metrics["monitor_score_semantics"] == "source TRAIN metric; not a DEV selector"


class _CapturedMonitorEngine:
    def __init__(self, captures: dict[str, object]) -> None:
        self.captures = captures

    def epoch_case_order(self, provider: object, epoch: int):
        self.captures["epoch_case_order"] = (provider, epoch)
        return [17, 23], "fixed-sampling-id"

    def epoch_update_slices(self, count: int):
        self.captures["slice_count"] = count
        return [slice(0, count)]

    def prepare_update_microbatches(self, provider: object, **kwargs: object):
        self.captures["prepare"] = (provider, kwargs)
        return ["prepared-monitor-batch"]


def test_fixed_monitor_batches_uses_converted_moment_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_recipe = {
        "family": "moment_hyperedges_round5",
        "architecture": "round5-envelope-architecture",
    }
    converted_recipe = {"family": "inherited_attention_v1", "architecture": "A2-D-native"}
    conversions: list[object] = []
    captures: dict[str, object] = {}
    engine = _CapturedMonitorEngine(captures)
    factory_calls: list[tuple[object, str]] = []

    def convert(recipe: object) -> dict[str, str]:
        conversions.append(recipe)
        return converted_recipe

    def engine_for(recipe: object, device: str) -> _CapturedMonitorEngine:
        factory_calls.append((recipe, device))
        return engine

    from honf_runtime import moment_hyperedge_protocol

    moment_module = ModuleType("inherited_attention_train")
    moment_module.engine_for = engine_for  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "inherited_attention_train", moment_module)
    monkeypatch.setattr(moment_hyperedge_protocol, "native_recipe_for_moment_hyperedges", convert)
    provider = SimpleNamespace(device="cpu")

    batches, ordered = evaluator.fixed_monitor_batches(provider, original_recipe, epoch=700, split="train")

    assert list(batches) == ["prepared-monitor-batch"]
    assert ordered == [17, 23]
    assert conversions == [original_recipe]
    assert factory_calls == [(converted_recipe, "cpu")]
    prepared_provider, kwargs = captures["prepare"]  # type: ignore[misc]
    assert prepared_provider is provider
    assert kwargs["arm"] == "A2-D-native"
    assert kwargs["sampling_dataset_id"] == "fixed-sampling-id"


def test_fixed_monitor_batches_keeps_round8_base_recipe_unwrapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_recipe = {"architecture": "A2-D", "recipe_owner": "D0-source"}
    round8_recipe = {
        "family": "native_budgeted_dispatch_v2",
        "architecture": "native-budgeted-wrapper",
        "base_recipe": base_recipe,
    }
    captures: dict[str, object] = {}
    engine = _CapturedMonitorEngine(captures)
    factory_calls: list[tuple[object, str]] = []

    def engine_for(recipe: object, device: str) -> _CapturedMonitorEngine:
        factory_calls.append((recipe, device))
        return engine

    content_module = ModuleType("content_hypergraph_train")
    content_module.engine_for = engine_for  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "content_hypergraph_train", content_module)
    provider = SimpleNamespace(device="cpu")

    batches, ordered = evaluator.fixed_monitor_batches(provider, round8_recipe, epoch=2500, split="train")

    assert list(batches) == ["prepared-monitor-batch"]
    assert ordered == [17, 23]
    assert factory_calls == [(base_recipe, "cpu")]
    prepared_provider, kwargs = captures["prepare"]  # type: ignore[misc]
    assert prepared_provider is provider
    assert kwargs["arm"] == "A2-D"
