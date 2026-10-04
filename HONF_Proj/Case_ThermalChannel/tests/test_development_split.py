"""Selection provenance, leakage boundaries and selected-only native fitting."""

from __future__ import annotations

import copy
import json

import h5py
import numpy as np
import pytest
from channelthermal.data.datasets import (
    GlobalChannelThermalDataset,
    GlobalModuleAlignmentDataset,
    H5Normalizer,
    LocalModuleDataset,
    fit_global_normalizer,
)
from channelthermal.data.development_split import (
    build_development_manifest,
    development_case_ids,
    manifest_fingerprint,
    resolve_development_manifest,
    validate_development_manifest,
)


def _case(h5, case_id, split, *, m=1, re=50.0, value=2.0, targets=True):
    group = h5.require_group("cases").create_group(case_id)
    group.attrs.update(split=split, converged=True)
    group.create_dataset("module_present", data=[1] * m + [0])
    group.create_dataset("heat_powers", data=[value] * m + [1e8])
    group.create_dataset("module_centers", data=np.arange((m + 1) * 2).reshape(m + 1, 2))
    group.create_group("material_parameters").attrs.update(re=re, solid_k=1.0, solid_alpha=0.01)
    if targets:
        group.create_dataset("sampled_points", data=np.column_stack((np.zeros((4, 2)), np.full((4, 5), value))))
        condition = np.full((m + 1, 3, 8), value)
        condition[-1] = 1e8
        group.create_dataset("interface_condition", data=condition)
        interface = np.full((m + 1, 3, 2), value)
        interface[-1] = 1e8
        group.create_dataset("interface_target", data=interface)
        internal = np.full((m + 1, 2, 2), value)
        internal[-1] = 1e8
        group.create_dataset("module_internal_temperature", data=internal)
        group.create_dataset("module_internal_mask", data=np.ones((2, 2)))
        group.create_dataset("interface_condition_valid_mask", data=np.ones((m + 1, 3)))


def _root(h5, ids, splits):
    h5.create_dataset("case_ids", data=np.asarray(ids, dtype="S"))
    h5.create_dataset("splits", data=np.asarray(splits, dtype="S"))
    h5.attrs.update(field_dim=5, n_interface_points=3, max_modules=5, local_grid_size=2, dataset_id="tiny-native")
    h5.create_dataset("interface_condition_feature_names", data=np.asarray(
        ["theta", "normal_x", "normal_y", "T_outside", "u_normal", "u_tangent", "h_proxy", "h_effective"], dtype="S"))
    normalization = h5.create_group("normalization")
    normalization.create_dataset("field_mean_by_channel", data=np.full(5, 999.0))
    normalization.create_dataset("field_std_by_channel", data=np.ones(5))


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / "catalog.h5"
    with h5py.File(path, "w") as h5:
        ids, splits = [], []
        for split, count in [("train", 36), ("test", 24)]:
            for i in range(count):
                case_id = f"{split}_{i:03d}"
                ids.append(case_id)
                splits.append(split)
                _case(h5, case_id, split, m=1 + i % 3, re=[40, 90, 150][i // 3 % 3], value=1 + i / 10, targets=False)
        _root(h5, ids, splits)
    return path


def _manifest(path):
    return build_development_manifest(path, policy={"excluded_test_case_ids": []})


def test_fixed_stratified_membership_and_rng_independence(catalog):
    before = np.random.get_state()
    first, second = _manifest(catalog), _manifest(catalog)
    assert first == second and first["manifest_sha256"] == manifest_fingerprint(first)
    assert first["partitions"]["train"]["selected_count"] == 9
    assert first["partitions"]["test"]["selected_count"] == 6
    assert set(first["partitions"]["train"]["module_quotas"]) == {"1", "2", "3"}
    for split in ("train", "test"):
        ids = development_case_ids(first, split)
        assert len(ids) == len(set(ids)) and all(value.startswith(split) for value in ids)
    assert not set(development_case_ids(first, "train")) & set(development_case_ids(first, "test"))
    assert development_case_ids(first, "val") == development_case_ids(first, "test")
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_heat_diversity_where_quota_affords_three_strata(catalog):
    manifest = build_development_manifest(catalog, policy={"fraction": 0.75, "excluded_test_case_ids": []})
    for partition in manifest["partitions"].values():
        groups = {}
        for row in partition["strata"]:
            groups.setdefault((row["module_count"], row["re_band"]), []).append(row)
        for rows in groups.values():
            if sum(row["selected"] for row in rows) >= len(rows):
                assert all(row["selected"] > 0 for row in rows)


def test_source_change_and_cross_partition_tamper_are_rejected(catalog):
    manifest = _manifest(catalog)
    tampered = copy.deepcopy(manifest)
    tampered["partitions"]["test"]["case_ids"][0] = manifest["partitions"]["train"]["case_ids"][0]
    tampered["manifest_sha256"] = manifest_fingerprint(tampered)
    with pytest.raises(ValueError, match="deterministic regeneration"):
        validate_development_manifest(tampered, catalog)
    with h5py.File(catalog, "a") as h5:
        h5["cases/train_000/material_parameters"].attrs["re"] = 41.0
    with pytest.raises(ValueError, match="metadata/stat"):
        validate_development_manifest(manifest, catalog)


def test_runtime_missing_manifest_vs_explicit_checkpoint_reader(catalog, tmp_path):
    manifest = _manifest(catalog)
    cfg = {"development_manifest": str(tmp_path / "absent.json"), "development_manifest_sha256": manifest["manifest_sha256"], "development_subset": manifest}
    with pytest.raises(ValueError, match="original fixed"):
        resolve_development_manifest(cfg, catalog)
    assert resolve_development_manifest(cfg, catalog, allow_embedded=True) == manifest
    cfg["development_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="configured fingerprint"):
        resolve_development_manifest(cfg, catalog, allow_embedded=True)
    assert resolve_development_manifest({}, catalog) is None


@pytest.fixture
def native(tmp_path):
    path = tmp_path / "native.h5"
    with h5py.File(path, "w") as h5:
        _case(h5, "a", "train", value=2.0)
        # Neither the excluded train group nor held-out group has target arrays:
        # touching either during selected fitting or construction must fail.
        _case(h5, "b", "train", value=1e7, targets=False)
        _case(h5, "c", "test", value=1e8, targets=False)
        _root(h5, ["a", "b", "c"], ["train", "train", "test"])
    return path


def test_selected_normalizer_ignores_unselected_heldout_and_padding(native):
    fitted = fit_global_normalizer(native, ["a"])
    assert len(fitted.stats) == 12
    for key, value in fitted.stats.items():
        if "mean" in key:
            np.testing.assert_array_equal(value, np.full_like(value, 2.0))
    train = GlobalChannelThermalDataset(native, split="train", case_ids=["a"])
    assert len(train) == 1 and train.selected_case_ids == ["a"]
    np.testing.assert_array_equal(train.normalizer.stats["field_mean_by_channel"], np.full(5, 2.0))
    val = GlobalChannelThermalDataset(native, split="test", case_ids=["c"], normalizer=train.normalizer)
    assert val.normalizer is train.normalizer
    alignment = GlobalModuleAlignmentDataset(native, split="train", case_ids=["a"])
    assert alignment.records == [("a", 0)]
    np.testing.assert_array_equal(alignment.normalizer.stats["internal_temperature_mean"], [2.0])
    legacy = GlobalChannelThermalDataset(native, split="train")
    assert legacy.selected_case_ids == ["a", "b"]
    np.testing.assert_array_equal(legacy.normalizer.stats["field_mean_by_channel"], np.full(5, 999.0))
    for dataset in (train, val, alignment, legacy):
        dataset.close()


@pytest.mark.parametrize("ids", [[], ["a", "a"], ["c"], ["missing"], "a", [1]])
def test_native_selection_rejects_invalid_membership_before_target_access(native, ids):
    with pytest.raises(ValueError, match="case|partition"):
        GlobalChannelThermalDataset(native, split="train", case_ids=ids)
    with pytest.raises(ValueError, match="case|partition"):
        GlobalModuleAlignmentDataset(native, split="train", case_ids=ids, normalizer=H5Normalizer({}))


def test_selected_validation_never_fits_heldout_stats(native):
    with pytest.raises(ValueError, match="training-only normalizer"):
        GlobalChannelThermalDataset(native, split="test", case_ids=["c"])
    with pytest.raises(ValueError, match="training-only normalizer"):
        GlobalModuleAlignmentDataset(native, split="test", case_ids=["c"])


def test_selected_fit_legacy_h_effective_uses_proxy_stats(native):
    with h5py.File(native, "a") as h5:
        names = h5["interface_condition_feature_names"][...][:-1]
        del h5["interface_condition_feature_names"]
        h5.create_dataset("interface_condition_feature_names", data=names)
        group = h5["cases/a"]
        condition = group["interface_condition"][...][..., :-1]
        condition[..., 6] = 7.0
        del group["interface_condition"]
        group.create_dataset("interface_condition", data=condition)
    normalizer = fit_global_normalizer(native, ["a"])
    assert normalizer.stats["interface_condition_mean"].shape == (8,)
    np.testing.assert_array_equal(normalizer.stats["interface_condition_mean"][-2:], [7.0, 7.0])
    np.testing.assert_array_equal(normalizer.stats["interface_condition_std"][-2:], [1.0, 1.0])


def test_manifest_json_roundtrip(catalog, tmp_path):
    manifest = _manifest(catalog)
    path = tmp_path / "fixed.json"
    path.write_text(json.dumps(manifest))
    cfg = {"development_manifest": str(path), "development_manifest_sha256": manifest["manifest_sha256"]}
    assert resolve_development_manifest(cfg, catalog) == manifest


def test_standalone_stage_a_packed_normalizer_is_unchanged(tmp_path):
    path = tmp_path / "local.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("case_ids", data=np.asarray(["local"], dtype="S"))
        h5.create_dataset("splits", data=np.asarray(["train"], dtype="S"))
        for key, shape in {"module_params": (1, 2), "port_tokens": (1, 3, 2),
                           "interface_targets": (1, 3, 2), "internal_query_points": (1, 4, 2)}.items():
            h5.create_dataset(key, data=np.zeros(shape))
        h5.create_group("normalization").create_dataset("heat_power_mean", data=[123.0])
    dataset = LocalModuleDataset(path)
    assert len(dataset) == 1
    np.testing.assert_array_equal(dataset.normalizer.stats["heat_power_mean"], [123.0])
    dataset.close()
