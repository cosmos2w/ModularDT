import h5py
import numpy as np
import pytest

from hetmf.io import validate_raw_case, write_case_atomic
from hetmf.processing import pack_case


def _solution(n=3):
    zeros = np.zeros((n, n), dtype=np.float64)
    return {"v": zeros, "theta": zeros, "q_sheet": zeros, "q_contact": zeros,
            "j_face_x": np.zeros((n, n + 1)), "j_face_y": np.zeros((n + 1, n)),
            "heat_flux_face_x": np.zeros((n, n + 1)),
            "heat_flux_face_y": np.zeros((n + 1, n)),
            "terminal_currents": np.zeros(2)}


def test_atomic_case_hash_resume_and_raw_precision(tmp_path, monkeypatch):
    monkeypatch.setattr("hetmf.io.require_owned_root", lambda path: path)
    (tmp_path / "raw").mkdir()
    scene = {"state_id": "R00", "environment": {"s": [[1.0]]},
             "contacts": [{"u": 1.0}, {"u": -1.0}]}
    first = write_case_atomic(tmp_path, scene, _solution(), grid=3, solver_hash="solver")
    assert not first["reused"]
    with h5py.File(first["raw_path"]) as handle:
        assert handle["potential"].dtype == np.float64
    assert validate_raw_case(first["raw_path"], scene, "solver")["grid"] == 3
    second = write_case_atomic(tmp_path, scene, _solution(), grid=3, solver_hash="solver")
    assert second["reused"]
    with pytest.raises(RuntimeError):
        write_case_atomic(tmp_path, scene, _solution(), grid=3, solver_hash="changed")


def test_pack_recreates_removed_empty_staging_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("hetmf.io.require_owned_root", lambda path: path)
    monkeypatch.setattr("hetmf.processing.require_owned_root", lambda path: path)
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    scene = {"state_id": "R00", "contacts": [{"u": 1.0}, {"u": -1.0}]}
    write_case_atomic(tmp_path, scene, _solution(), grid=3, solver_hash="solver")
    (tmp_path / "temporary").rmdir()
    result = pack_case(tmp_path, "R00", 3)
    assert not result["reused"]
    with h5py.File(result["path"], "r") as handle:
        assert handle["fields"].shape == (3, 3, 5)
    assert not list((tmp_path / "temporary").iterdir())
