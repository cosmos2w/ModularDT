from pathlib import Path

import pytest

from hetmf.resources import link_training_artifacts


def test_training_browsing_links_preserve_regular_and_foreign_paths(tmp_path, monkeypatch):
    case = tmp_path / "Case_ElectroThermal"
    (case / "Dataset").mkdir(parents=True)
    (case / "pyproject.toml").touch()
    target = tmp_path / "standard.h5"
    target.write_bytes(b"fixture")
    monkeypatch.setattr("hetmf.resources.require_owned_root", lambda path: Path(path).resolve())
    (case / "Dataset/standard.h5").write_text("user file")
    with pytest.raises(FileExistsError, match="regular"):
        link_training_artifacts(case, {"standard.h5": target})
    (case / "Dataset/standard.h5").unlink()
    (case / "Dataset/standard.h5").symlink_to(tmp_path / "foreign")
    with pytest.raises(FileExistsError, match="foreign"):
        link_training_artifacts(case, {"standard.h5": target})
    (case / "Dataset/standard.h5").unlink()
    expected = {"standard.h5": str(target)}
    assert link_training_artifacts(case, {"standard.h5": target}) == expected
    assert link_training_artifacts(case, {"standard.h5": target}) == expected
    with pytest.raises(ValueError, match="unknown"):
        link_training_artifacts(case, {"../escape": target})


def test_link_targets_are_preflighted_before_creating_any_link(tmp_path, monkeypatch):
    case = tmp_path / "Case_ElectroThermal"
    (case / "Dataset").mkdir(parents=True)
    (case / "pyproject.toml").touch()
    target = tmp_path / "standard.h5"
    target.touch()
    monkeypatch.setattr("hetmf.resources.require_owned_root", lambda path: Path(path).resolve())
    with pytest.raises(FileNotFoundError):
        link_training_artifacts(case, {"standard.h5": target, "expanded.h5": tmp_path / "missing"})
    assert not (case / "Dataset/standard.h5").exists()
