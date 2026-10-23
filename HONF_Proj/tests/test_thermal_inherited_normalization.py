"""Focused contracts for reusing saved Thermal training normalization."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from channelthermal.data.datasets import H5Normalizer
from channelthermal.training import content_hypergraph_task_v2 as thermal_task_v2


def _install_fake_factory_dependencies(monkeypatch):
    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "src/config_core/forward/content_hypergraph/thermal_a2-h6_development.json"
    )
    recipe = json.loads(recipe_path.read_text())
    train_ids = tuple(f"train-{index:03d}" for index in range(150))
    validation_ids = tuple(f"validation-{index:03d}" for index in range(22))
    manifest = {
        "manifest_sha256": thermal_task_v2.FIXED25_FINGERPRINT,
    }
    catalog = [
        *({"case_id": case_id, "split": "train"} for case_id in train_ids),
        *({"case_id": case_id, "split": "test"} for case_id in validation_ids),
    ]
    data_path = Path(__file__).resolve()
    monkeypatch.setattr(
        thermal_task_v2, "_load_data_binding",
        lambda *_args, **_kwargs: (manifest, data_path, None),
    )
    monkeypatch.setattr(
        thermal_task_v2, "read_case_catalog",
        lambda _path: ({"metadata_sha256": "fixture-metadata"}, catalog),
    )
    monkeypatch.setattr(
        thermal_task_v2, "development_case_ids",
        lambda _manifest, split: train_ids if split == "train" else validation_ids,
    )
    loaded_normalizers = []

    def read_cases(_path, _ids, *, normalizer, **_kwargs):
        assert isinstance(normalizer, H5Normalizer)
        loaded_normalizers.append(normalizer)
        return []

    monkeypatch.setattr(thermal_task_v2, "_read_selected_cases", read_cases)
    monkeypatch.setattr(thermal_task_v2, "_read_response_families", lambda *_args: ({
        "family_id": "fixture", "source_sha256": "fixture-source",
    },))
    fake_response_module = types.ModuleType("thermal_source_response_fit")
    fake_response_module.response_scales = lambda _families: {"fluid": 1.0}
    monkeypatch.setitem(sys.modules, "thermal_source_response_fit", fake_response_module)

    class FakeAdapter:
        def __init__(self, stats, **_kwargs):
            self.normalization_stats = stats

        def to(self, _device):
            return self

    captured = {}

    class FakeProvider:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def preparation_summary(self):
            return {"fixture": True}

    monkeypatch.setattr(thermal_task_v2, "ThermalContentHypergraphAdapterV2", FakeAdapter)
    monkeypatch.setattr(thermal_task_v2, "ThermalContentHypergraphTaskV2", FakeProvider)
    return recipe, data_path, train_ids, loaded_normalizers, captured


def test_factory_reuses_supplied_parent_stats_without_refitting(monkeypatch):
    recipe, _path, _train_ids, loaded_normalizers, captured = _install_fake_factory_dependencies(monkeypatch)
    saved_stats = {
        "field_mean_by_channel": np.asarray([1.25, -2.5, 0.75], dtype=np.float64),
        "field_std_by_channel": np.asarray([3.0, 4.0, 5.0], dtype=np.float64),
        "internal_temperature_std": np.asarray([7.5], dtype=np.float64),
    }

    def forbidden_fit(*_args, **_kwargs):
        pytest.fail("Inherited parent normalization must not call fit_global_normalizer.")

    monkeypatch.setattr(thermal_task_v2, "fit_global_normalizer", forbidden_fit)
    model, _provider, _summary = thermal_task_v2.build_thermal_content_hypergraph_task_v2(
        recipe, normalization_stats=saved_stats,
    )

    for key, expected in saved_stats.items():
        np.testing.assert_array_equal(model.normalization_stats[key], expected)
        for normalizer in loaded_normalizers:
            np.testing.assert_array_equal(normalizer.stats[key], expected.astype(np.float32))
    assert len(loaded_normalizers) == 2
    assert captured["source_binding"]["normalization_source"] == (
        "supplied saved H5Normalizer statistics (no refit)"
    )


def test_factory_keeps_legacy_fit_when_stats_are_omitted(monkeypatch):
    recipe, data_path, train_ids, loaded_normalizers, captured = _install_fake_factory_dependencies(monkeypatch)
    fitted_stats = {"field_mean_by_channel": np.asarray([0.5], dtype=np.float32)}
    calls = []

    def fake_fit(path, case_ids):
        calls.append((path, tuple(case_ids)))
        return H5Normalizer(fitted_stats)

    monkeypatch.setattr(thermal_task_v2, "fit_global_normalizer", fake_fit)
    model, _provider, _summary = thermal_task_v2.build_thermal_content_hypergraph_task_v2(recipe)

    assert calls == [(data_path, train_ids)]
    assert all(normalizer.stats is fitted_stats for normalizer in loaded_normalizers)
    assert model.normalization_stats is fitted_stats
    assert captured["source_binding"]["normalization_source"] == (
        "TRAIN-only H5Normalizer fit_global_normalizer"
    )
