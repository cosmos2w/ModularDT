from pathlib import Path
import pytest
from hetmf.resources import require_data2, require_owned_root, link_dataset


def test_real_data2_mount_and_scope():
    assert require_data2()["mount"][0] == "/data2"
    with pytest.raises(ValueError):
        require_owned_root(Path("/tmp/not-hetmf"))


def test_link_helper_refuses_regular_paths(tmp_path, monkeypatch):
    root = tmp_path / "pilot_root"
    for name in ("raw", "processed", "figures", "inputs"):
        (root / name).mkdir(parents=True)
    monkeypatch.setattr("hetmf.resources.require_owned_root", lambda path: Path(path))
    case = tmp_path / "Case_ElectroThermal"
    (case / "Dataset").mkdir(parents=True)
    (case / "pyproject.toml").touch()
    (case / "Dataset" / "pilot").mkdir()
    with pytest.raises(FileExistsError):
        link_dataset(root, case)


def test_link_helper_idempotent_and_refuses_foreign_dangling_link(tmp_path, monkeypatch):
    root = tmp_path / "pilot_root"
    for name in ("raw", "processed", "figures", "inputs"):
        (root / name).mkdir(parents=True)
    monkeypatch.setattr("hetmf.resources.require_owned_root", lambda path: Path(path))
    case = tmp_path / "Case_ElectroThermal"
    (case / "Dataset").mkdir(parents=True)
    (case / "pyproject.toml").touch()
    first = link_dataset(root, case)
    assert link_dataset(root, case) == first
    (case / "Dataset" / "raw").unlink()
    (case / "Dataset" / "raw").symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        link_dataset(root, case, replace_owned_links=True)


def test_link_helper_preserves_multiple_checkout_ownership(tmp_path, monkeypatch):
    import json
    root = tmp_path / "pilot_root"
    for name in ("raw", "processed", "figures", "inputs"):
        (root / name).mkdir(parents=True)
    monkeypatch.setattr("hetmf.resources.require_owned_root", lambda path: Path(path))
    cases = []
    for checkout in ("first", "second"):
        case = tmp_path / checkout / "Case_ElectroThermal"
        (case / "Dataset").mkdir(parents=True)
        (case / "pyproject.toml").touch()
        cases.append(case)
        link_dataset(root, case)
    recorded = json.loads((root / "receipts" / "browsing_links.json").read_text())
    assert str(cases[0] / "Dataset" / "pilot") in recorded
    assert str(cases[1] / "Dataset" / "pilot") in recorded
