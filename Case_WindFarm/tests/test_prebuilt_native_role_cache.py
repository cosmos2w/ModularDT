"""Training consumes verified catalogues; only explicit preparation may build."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import textwrap
import errno

import numpy as np
import pytest

from windfarm.workflows import joint_forward as joint
from windfarm.workflows.native_role_cache import NativeRoleCatalogueStore, compatible_v1_recipe


def case():
    return SimpleNamespace(index=7, layout_index=7, diameter_m=2., hub_height_m=2.,
        module_centers=np.array([[1., 1., 1.]], dtype=np.float32), module_present=np.ones(1),
        run=SimpleNamespace(nx=4, ny=3, nz=3, cell_count=36,
            x_m=np.array([0., 2., 4., 6.]), y_m=np.array([0., 2., 4.]), z_m=np.array([1., 3., 9.]),
            U=np.arange(108, dtype=np.float32).reshape(36, 3)))


@pytest.fixture(autouse=True)
def clear_recipe():
    joint._native_role_catalogue_recipe.cache_clear()
    yield
    joint._native_role_catalogue_recipe.cache_clear()


def forbid_build(*args, **kwargs):
    pytest.fail('Training executed the native catalogue builder')


def test_plot_comments_and_docstrings_do_not_invalidate_numerical_identity(monkeypatch):
    from windfarm import study_spatial
    expected = joint._native_role_catalogue_recipe()
    original = joint.inspect.getsource
    monkeypatch.setattr(study_spatial, 'render_native_plane', lambda *args: 'new RMSE annotation')
    monkeypatch.setattr(joint.inspect, 'getsource', lambda function:
        textwrap.dedent(original(function)).replace('Generate exact C-order coordinates', 'Revised documentation')
        + '\n# Unrelated presentation change\n')
    joint._native_role_catalogue_recipe.cache_clear()
    assert joint._native_role_catalogue_recipe() == expected
    assert compatible_v1_recipe(expected) is not None


def altered_coordinates(run, flat, diameter):
    return np.column_stack((run.x_m[flat % run.nx], np.zeros(len(flat)), np.ones(len(flat)))) / diameter


def test_numerical_dependency_change_cannot_use_v1_payloads(monkeypatch):
    before = joint._native_role_catalogue_recipe()
    monkeypatch.setattr(joint, 'native_coordinates', altered_coordinates)
    joint._native_role_catalogue_recipe.cache_clear()
    changed = joint._native_role_catalogue_recipe()
    assert changed['implementation_sha256'] != before['implementation_sha256']
    assert compatible_v1_recipe(changed) is None


def test_training_miss_fails_with_preparation_command_without_builder(tmp_path, monkeypatch):
    from windfarm.training.content_hypergraph_task import _shared_read_only_catalogue_cache
    joint._native_role_catalogue_recipe()
    monkeypatch.setattr(joint.NativeRoleCatalogueCache, '_build_catalogue', forbid_build)
    monkeypatch.setenv('HONF_WIND_CATALOGUE_CACHE_DIR', str(tmp_path))
    cache = _shared_read_only_catalogue_cache()
    with pytest.raises(FileNotFoundError, match='layout 7.*geometry.*--scope full --prepare'):
        cache.get(case())
    assert cache.build_count == cache.disk_write_count == 0
    assert cache.persistent_miss_count == 1
    assert not list(tmp_path.iterdir())


def test_disabled_training_cache_cannot_fall_back_to_ram(monkeypatch):
    from windfarm.training.content_hypergraph_task import _shared_read_only_catalogue_cache
    monkeypatch.setenv('HONF_WIND_CATALOGUE_CACHE_DIR', 'off')
    with pytest.raises(FileNotFoundError, match='prebuilt shared catalogue.*--prepare'):
        _shared_read_only_catalogue_cache()


def prepare(directory):
    cache = joint.NativeRoleCatalogueCache(persistent_dir=directory, read_only_persistent=False)
    catalogue = cache.get(case())
    assert cache.build_count == cache.disk_write_count == 1
    return cache, catalogue


def test_explicit_preparation_then_training_uses_disk_and_fresh_targets(tmp_path, monkeypatch):
    _, expected = prepare(tmp_path)
    monkeypatch.setattr(joint.NativeRoleCatalogueCache, '_build_catalogue', forbid_build)
    cache = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path)
    catalogue = cache.get(case())
    assert cache.disk_hit_count == 1 and cache.build_count == cache.disk_write_count == 0
    np.testing.assert_array_equal(catalogue.coordinates_D, expected.coordinates_D)
    assert all(not a.flags.writeable for a in (catalogue.coordinates_D, *catalogue.role_cdf.values(), *catalogue.role_indices.values()))
    counts = {name: 13 for name in joint.ROLE_NAMES}
    first = joint.sample_native_role_queries(case(), np.random.default_rng(92), counts, catalogue_cache=cache)
    changed = case(); changed.run.U += 100
    second = joint.sample_native_role_queries(changed, np.random.default_rng(92), counts, catalogue_cache=cache)
    np.testing.assert_array_equal(first.flat_indices, second.flat_indices)
    np.testing.assert_array_equal(first.target_mps + 100, second.target_mps)


def test_verified_v1_payloads_are_reused_in_place(tmp_path, monkeypatch):
    current_function = joint._native_role_catalogue_recipe
    current = current_function()
    legacy = compatible_v1_recipe(current)
    monkeypatch.setattr(joint, '_native_role_catalogue_recipe', lambda: legacy)
    prepared, expected = prepare(tmp_path)
    entry = prepared._store.entry(7, expected.geometry_sha256)
    before = {p.name: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in entry.iterdir()}
    monkeypatch.setattr(joint, '_native_role_catalogue_recipe', current_function)
    monkeypatch.setattr(joint.NativeRoleCatalogueCache, '_build_catalogue', forbid_build)
    monkeypatch.setattr(NativeRoleCatalogueStore, 'write', forbid_build)
    cache = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path, max_cached_bytes=1)
    for _ in range(2):
        actual = cache.get(case())
        np.testing.assert_array_equal(actual.coordinates_D, expected.coordinates_D)
        assert cache._store.last_loaded_entry == entry
    assert cache.disk_hit_count == cache.legacy_disk_hit_count == 2
    assert cache.build_count == cache.disk_write_count == 0
    assert before == {p.name: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in entry.iterdir()}


@pytest.mark.parametrize('corruption', ['checksum', 'missing', 'dtype', 'geometry'])
def test_incompatible_prebuilt_entry_is_not_silently_rebuilt(tmp_path, monkeypatch, corruption):
    prepared, catalogue = prepare(tmp_path)
    entry = prepared._store.entry(7, catalogue.geometry_sha256)
    manifest = json.loads((entry / 'manifest.json').read_text())
    p = entry / 'coordinates.npy'
    if corruption == 'missing':
        p.unlink()
    elif corruption == 'checksum':
        data = bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(data)
    elif corruption == 'dtype':
        np.save(p, np.asarray(catalogue.coordinates_D, dtype=np.float64), allow_pickle=False)
        manifest['arrays']['coordinates'].update(file_bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    else:
        manifest['geometry_sha256'] = '0' * 64
    (entry / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(joint.NativeRoleCatalogueCache, '_build_catalogue', forbid_build)
    cache = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match='incompatible prebuilt.*--prepare'):
        cache.get(case())
    assert cache.build_count == cache.disk_write_count == 0
    assert not list(entry.parent.glob('.invalid-*'))


def test_higher_precision_geometry_cannot_reuse_float32_entry(tmp_path, monkeypatch):
    prepare(tmp_path)
    monkeypatch.setattr(joint.NativeRoleCatalogueCache, '_build_catalogue', forbid_build)
    changed = case(); changed.module_centers = changed.module_centers.astype(np.float64)
    changed.module_centers[0, 0] += 1e-10
    with pytest.raises(FileNotFoundError, match='prebuilt Wind catalogue'):
        joint.NativeRoleCatalogueCache(persistent_dir=tmp_path).get(changed)


@pytest.mark.parametrize('error_number', [errno.EMFILE, errno.ENFILE])
def test_descriptor_exhaustion_is_not_reported_as_a_catalogue_miss(tmp_path, monkeypatch, error_number):
    from windfarm.workflows import native_role_cache as store
    prepare(tmp_path)
    monkeypatch.setattr(joint.NativeRoleCatalogueCache, '_build_catalogue', forbid_build)
    def exhausted(path):
        raise OSError(error_number, 'Too many open files', str(path))
    monkeypatch.setattr(store, '_sha256', exhausted)
    cache = joint.NativeRoleCatalogueCache(persistent_dir=tmp_path)
    with pytest.raises(OSError) as caught:
        cache.get(case())
    assert caught.value.errno == error_number
    assert cache.persistent_miss_count == cache.build_count == cache.disk_write_count == 0


def test_real_shared_catalogue_load_forbids_builder_and_writes(monkeypatch):
    from windfarm.data import WindFarmNativeView
    from windfarm.training.unified_task import DEFAULT_DATA_ROOT, DEFAULT_DERIVED_ROOT, _load_original_split
    if not Path(DEFAULT_DATA_ROOT).exists():
        pytest.skip('Real Wind data unavailable')
    view = WindFarmNativeView(DEFAULT_DATA_ROOT, allow_npz_metadata_fallback=True)
    split = _load_original_split(view, DEFAULT_DERIVED_ROOT)
    actual_case = view.run(int(split.train[0]))
    joint._native_role_catalogue_recipe()
    monkeypatch.setattr(joint.NativeRoleCatalogueCache, '_build_catalogue', forbid_build)
    monkeypatch.setattr(NativeRoleCatalogueStore, 'write', forbid_build)
    cache = joint.NativeRoleCatalogueCache(max_cached_bytes=1)
    actual = cache.get(actual_case)
    assert isinstance(actual.coordinates_D, np.memmap)
    assert cache.disk_hit_count == 1 and cache.build_count == cache.disk_write_count == 0
