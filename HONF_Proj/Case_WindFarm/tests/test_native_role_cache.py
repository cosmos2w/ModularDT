"""Cross-launch catalogue reuse must preserve geometry, targets and RNG streams."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from windfarm.workflows import joint_forward as joint
from windfarm.workflows.native_role_cache import training_catalogue_cache_directory


def make_case():
    return SimpleNamespace(
        index=7,
        layout_index=7,
        diameter_m=2.0,
        hub_height_m=2.0,
        module_centers=np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        module_present=np.ones(1, dtype=np.float32),
        run=SimpleNamespace(
            nx=4,
            ny=3,
            nz=3,
            cell_count=36,
            x_m=np.asarray([0.0, 2.0, 4.0, 6.0]),
            y_m=np.asarray([0.0, 2.0, 4.0]),
            z_m=np.asarray([1.0, 3.0, 9.0]),
            U=np.arange(108, dtype=np.float32).reshape(36, 3),
        ),
    )


def draw(case, cache):
    return joint.sample_native_role_queries(
        case,
        np.random.default_rng(932),
        {role: 13 for role in joint.ROLE_NAMES},
        catalogue_cache=cache,
    )


def test_warm_catalogue_is_exact_readonly_and_targets_remain_fresh(tmp_path, monkeypatch):
    case = make_case()
    reference = draw(case, joint.NativeRoleCatalogueCache())
    cold = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path)
    first = draw(case, cold)
    assert cold.build_count == cold.disk_write_count == 1
    joint._native_role_catalogue_recipe()  # Capture actual implementation before sentinel.
    monkeypatch.setattr(joint.NativeRoleCatalogueCache, "_build_catalogue", lambda *a: pytest.fail("rebuilt"))
    warm = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path)
    second = draw(case, warm)
    catalogue = warm.get(case)
    assert warm.disk_hit_count == 1 and warm.build_count == 0
    assert isinstance(catalogue.coordinates_D, np.memmap)
    for array in (catalogue.coordinates_D, *catalogue.role_cdf.values(), *catalogue.role_indices.values()):
        assert not array.flags.writeable
    for sample in (first, second):
        np.testing.assert_array_equal(sample.flat_indices, reference.flat_indices)
        np.testing.assert_array_equal(sample.coordinates_D, reference.coordinates_D)
        np.testing.assert_array_equal(sample.target_mps, reference.target_mps)
        assert sample.role_support_volume_m3 == reference.role_support_volume_m3
        assert sample.geometry_sha256 == reference.geometry_sha256
    case.run.U = case.run.U + 100
    refreshed = draw(case, warm)
    np.testing.assert_array_equal(refreshed.flat_indices, reference.flat_indices)
    np.testing.assert_array_equal(refreshed.target_mps, reference.target_mps + 100)


@pytest.mark.parametrize("change", ["axes", "diameter", "hub_height", "centers", "precision", "layout"])
def test_geometry_change_cannot_reuse_persisted_catalogue(tmp_path, change):
    original = make_case()
    cold = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path)
    first = cold.get(original)
    changed = make_case()
    if change == "axes":
        changed.run.x_m += 0.125
    elif change == "diameter":
        changed.diameter_m += 0.125
    elif change == "hub_height":
        changed.hub_height_m += 0.125
    elif change == "centers":
        changed.module_centers[0, 0] += 0.125
    elif change == "precision":
        changed.module_centers = changed.module_centers.astype(np.float64)
        changed.module_centers[0, 0] += 1e-10
    else:
        changed.layout_index += 1
    with pytest.raises(RuntimeError, match="changed geometry"):
        cold.get(changed)
    fresh = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path)
    assert fresh.get(changed).geometry_sha256 != first.geometry_sha256
    assert fresh.disk_hit_count == 0 and fresh.build_count == 1


@pytest.mark.parametrize("corruption", ["bytes", "missing", "manifest", "truncated"])
def test_corrupt_or_partial_entry_is_rebuilt_before_sampling(tmp_path, corruption):
    case = make_case()
    cold = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path)
    expected = draw(case, cold)
    entry = cold._store.entry(case.layout_index, expected.geometry_sha256)
    if corruption == "manifest":
        path = entry / "manifest.json"
        binding = json.loads(path.read_text())
        binding["geometry_sha256"] = "0" * 64
        path.write_text(json.dumps(binding))
    else:
        path = entry / "cdf_volume.npy"
        if corruption == "missing":
            path.unlink()
        elif corruption == "truncated":
            path.write_bytes(path.read_bytes()[:100])
        else:
            payload = bytearray(path.read_bytes())
            payload[-9] ^= 1
            path.write_bytes(payload)
    warm = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path)
    recovered = draw(case, warm)
    assert warm.build_count == warm.disk_write_count == 1 and warm.disk_hit_count == 0
    assert len(list(cold._store.directory.glob(".invalid-*"))) == 1
    np.testing.assert_array_equal(recovered.flat_indices, expected.flat_indices)


def test_evicted_and_oversized_catalogues_reuse_disk(tmp_path):
    case = make_case()
    # A one-byte LRU cannot retain even one catalogue; disk reuse still works.
    cache = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path, max_cached_bytes=1)
    first = draw(case, cache)
    second = draw(case, cache)
    assert cache.build_count == 1 and cache.disk_hit_count == 1
    assert cache.summary()["catalogue_count"] == 0
    np.testing.assert_array_equal(first.flat_indices, second.flat_indices)


def test_changed_recipe_and_abandoned_staging_cannot_supply_old_arrays(tmp_path, monkeypatch):
    case = make_case()
    cold = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path)
    cold.get(case)
    (cold._store.directory / ".building-interrupted").mkdir()
    recipe = {**joint._native_role_catalogue_recipe(), "schema_version": 2}
    monkeypatch.setattr(joint, "_native_role_catalogue_recipe", lambda: recipe)
    fresh = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path)
    fresh.get(case)
    assert fresh.build_count == 1 and fresh.disk_hit_count == 0
    assert fresh._store.directory != cold._store.directory


_PROCESS_CODE = """
import json, os, time
from test_native_role_cache import make_case, draw
from windfarm.workflows import joint_forward as joint
joint._native_role_catalogue_recipe()
original = joint.NativeRoleCatalogueCache._build_catalogue
def build(self, *args):
    if os.environ.get("FORBID_BUILD"):
        raise RuntimeError("warm launch rebuilt geometry")
    with open(os.environ["BUILD_RECEIPT"], "a") as stream:
        stream.write("built\\n")
    time.sleep(0.2)
    return original(self, *args)
joint.NativeRoleCatalogueCache._build_catalogue = build
cache = joint.NativeRoleCatalogueCache(persistent_dir=os.environ["CACHE_DIRECTORY"])
sample = draw(make_case(), cache)
print(json.dumps({"builds": cache.build_count, "hits": cache.disk_hit_count,
                  "indices": sample.flat_indices.tolist()}))
"""


def test_independent_concurrent_launches_build_once_then_warm_without_builder(tmp_path):
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "CACHE_DIRECTORY": str(tmp_path),
        "BUILD_RECEIPT": str(tmp_path / "build_receipt"),
    }
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _PROCESS_CODE],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=45)
        assert process.returncode == 0, stderr
        results.append(json.loads(stdout))
    assert sorted(result["builds"] for result in results) == [0, 1]
    assert results[0]["indices"] == results[1]["indices"]
    warm = subprocess.run(
        [sys.executable, "-c", _PROCESS_CODE],
        env={**environment, "FORBID_BUILD": "1"},
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert warm.returncode == 0, warm.stderr
    assert json.loads(warm.stdout)["hits"] == 1
    assert (tmp_path / "build_receipt").read_text().splitlines() == ["built"]


def test_full_and_segmented_prebuild_reuse_sealed_rows_without_test_or_targets(tmp_path):
    from test_unified_task import _fixed_manifest
    from wind_native_catalogue_cache import prepare_rows, select_rows

    groups = np.repeat(np.arange(200), 3)
    directions = np.tile([270.0, 285.0, 300.0], 200)
    manifest, split = _fixed_manifest(groups, directions)
    path = tmp_path / "fixed_subset_manifest.json"
    path.write_text(json.dumps(manifest))
    view = SimpleNamespace(metadata={"layout_index": groups, "wd_deg": directions})
    full, full_binding = select_rows(view, split, "full", path)
    segmented, segmented_binding = select_rows(view, split, "segmented", path)
    assert full.shape == (510,) and segmented.shape == (96,)
    assert set(segmented).issubset(set(full)) and not set(full) & set(split.test)
    assert full_binding["train_row_count"] == 420
    assert segmented_binding["train_row_count"] == 72
    assert segmented_binding["subset_manifest_sha256"] == manifest["manifest_sha256"]

    class GeometryOnlyRun:
        def __init__(self):
            self.__dict__.update({key: value for key, value in vars(make_case().run).items() if key != "U"})

        @property
        def U(self):
            pytest.fail("prebuild read target values")

    visited = []

    def run(row):
        assert row not in set(split.test)
        visited.append(row)
        case = make_case()
        case.index = row
        case.run = GeometryOnlyRun()
        return case

    view.run = run
    cache = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path / "cache", max_cached_bytes=1)
    result = prepare_rows(view, segmented[:2], cache)
    assert visited == segmented[:2].tolist()
    assert result["catalogue_build_count"] == 1 and result["persistent_disk_hit_count"] == 1
    # The same geometry remains reusable for a row from the full selection.
    cache = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path / "cache", max_cached_bytes=1)
    assert prepare_rows(view, full[:1], cache)["catalogue_build_count"] == 0


def test_training_directory_override_and_disable(monkeypatch, tmp_path):
    monkeypatch.delenv("HONF_WIND_CATALOGUE_CACHE_DIR", raising=False)
    assert training_catalogue_cache_directory().name == "native_role_catalogues_v1"
    monkeypatch.setenv("HONF_WIND_CATALOGUE_CACHE_DIR", str(tmp_path))
    assert training_catalogue_cache_directory() == tmp_path
    monkeypatch.setenv("HONF_WIND_CATALOGUE_CACHE_DIR", "off")
    assert training_catalogue_cache_directory() is None


def test_provider_enables_persistence_without_changing_scientific_identity(monkeypatch, tmp_path):
    from test_unified_task import _provider

    monkeypatch.setenv("HONF_WIND_CATALOGUE_CACHE_DIR", "off")
    original = _provider()
    monkeypatch.setenv("HONF_WIND_CATALOGUE_CACHE_DIR", str(tmp_path))
    persisted = _provider()
    assert persisted.catalogue_cache.persistent_dir == tmp_path
    normalization = tmp_path / "train_only_normalization.json"
    normalization.write_text("{}\n")
    for provider in (original, persisted):
        provider._message_scale = 0.2  # Identical synthetic prefit calibration.
        provider.view.token_shape = (4, 4, 4)
        provider.normalization_path = normalization
    assert persisted.identity_payload() == original.identity_payload()


def test_inconsistent_native_grid_is_rejected_before_reusing_arrays(tmp_path):
    case = make_case()
    cache = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path)
    cache.get(case)
    case.run.cell_count += 1
    with pytest.raises(ValueError, match="cell count disagrees"):
        cache.get(case)
