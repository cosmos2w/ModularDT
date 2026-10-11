"""Native input metadata joins for reference-anchored Wind panel freezing."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/reference_anchored_diagnose.py"
_SPEC = importlib.util.spec_from_file_location("reference_anchored_diagnose_panel", _MODULE_PATH)
_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_cli)


def _inputs():
    # Sorted category metadata intentionally differs from layout ID order.
    manifest = {"direction_categories": [270., 285., 300.], "train_layout_selection": {
        "layout_indices": [2, 7, 10, 12], "row_indices": list(range(12)),
        "selected_turbine_counts": [6, 8, 18, 30],
        "direction_categories": [270., 285., 300.], "direction_rows_per_layout": 3}}
    identity = {"train_layout_indices": [2, 7, 10, 12], "train_rows": list(range(12))}
    metadata = {"layout_index": np.repeat([2, 7, 10, 12], 3),
                "wd_deg": np.tile([270., 285., 300.], 4),
                "n_turbines": np.repeat([30, 6, 18, 8], 3)}
    return manifest, identity, metadata


def test_wind_panel_ranks_native_counts_instead_of_sorted_category_labels():
    manifest, identity, metadata = _inputs()
    panel = _cli._select_wind_metadata_panel(manifest, identity, identity, metadata, 3)
    assert panel["selected_layout_indices"] == [7, 10, 2]
    assert panel["selected_turbine_counts"] == [6, 18, 30]
    assert panel["selected_row_indices"] == [3, 4, 5, 6, 7, 8, 0, 1, 2]


@pytest.mark.parametrize("mutation", ["direction", "count", "membership"])
def test_wind_panel_rejects_native_metadata_or_train_exposure_drift(mutation):
    manifest, identity, metadata = _inputs()
    full = deepcopy(identity)
    if mutation == "direction":
        metadata["wd_deg"][0] = 200.
    elif mutation == "count":
        metadata["n_turbines"][0] = 29
    else:
        full["train_rows"].remove(0)
    with pytest.raises(ValueError):
        _cli._select_wind_metadata_panel(manifest, identity, full, metadata, 4)


def test_wind_panel_metadata_loader_never_opens_fields(tmp_path, monkeypatch):
    from windfarm.io import VOLUME_ARRAYS

    volume = tmp_path / "family_volume"
    volume.mkdir()
    # Discovery may inspect field filenames; any attempt to open them as arrays
    # must fail because they deliberately contain no NPY header.
    for name in VOLUME_ARRAYS:
        (volume / f"{name}.npy").write_bytes(b"field target access is forbidden")
    np.save(volume / "layout_index.npy", np.array([7, 7, 7]))
    np.save(volume / "wd_deg.npy", np.array([270., 285., 300.]))
    compact = tmp_path / "family_tensor"
    compact.mkdir()
    np.save(compact / "n_turbines.npy", np.array([6, 6, 6]))
    original_load = np.load
    opened = []

    def input_only_load(path, **kwargs):
        opened.append(Path(path).name)
        assert Path(path).name in {"layout_index.npy", "wd_deg.npy", "n_turbines.npy"}
        return original_load(path, **kwargs)

    monkeypatch.setattr(np, "load", input_only_load)
    metadata, receipt = _cli._load_wind_panel_metadata(tmp_path)
    assert opened == ["layout_index.npy", "wd_deg.npy", "n_turbines.npy"]
    assert metadata["n_turbines"].tolist() == [6, 6, 6]
    assert receipt["field_arrays_opened"] is False
    assert receipt["target_values_read"] is False
