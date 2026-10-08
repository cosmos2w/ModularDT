import json
import os

import pytest
from standardize_run_artifacts import standardize


def test_relocation_retains_checkpoint_bytes_and_legacy_reference(tmp_path):
    checkpoint = tmp_path / "latest_model.pt"
    checkpoint.write_bytes(b"immutable saved state")
    (tmp_path / "history.json").write_text('[{"epoch": 100}]')
    (tmp_path / "validation_epoch_0100.json").write_text('{"field_score": 0.2}')
    (tmp_path / "active_process.json").write_text('{"status": "completed", "pid": 2147483647}')
    before = standardize(tmp_path)
    assert before["status"] == "planned_only" and checkpoint.is_file() and not checkpoint.is_symlink()
    result = standardize(tmp_path, apply=True)
    assert result["status"] == "completed"
    assert checkpoint.is_symlink() and checkpoint.read_bytes() == b"immutable saved state"
    assert (tmp_path / "checkpoints/latest_model.pt").is_file()
    assert (tmp_path / "metrics/history.json").read_text() == '[{"epoch": 100}]'
    assert (tmp_path / "evaluations/validation/validation_epoch_0100.json").is_file()
    receipt = json.loads((tmp_path / "logs/artifact_relocation.json").read_text())
    assert receipt["status"] == "completed" and receipt["verified_artifacts"] == 4
    assert standardize(tmp_path, apply=True)["status"] == "already_standardized"


def test_relocation_refuses_live_owner_without_changing_artifacts(tmp_path):
    (tmp_path / "latest_model.pt").write_bytes(b"state")
    (tmp_path / "active_process.json").write_text(json.dumps({"pid": os.getpid(), "status": "running"}))
    with pytest.raises(RuntimeError, match="still alive"):
        standardize(tmp_path, apply=True)
    assert (tmp_path / "latest_model.pt").read_bytes() == b"state"
    assert not (tmp_path / "checkpoints").exists()


def test_relocation_refuses_conflict_before_any_move(tmp_path):
    (tmp_path / "history.json").write_text('[]')
    (tmp_path / "latest_model.pt").write_bytes(b"state")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/latest_model.pt").write_bytes(b"other")
    with pytest.raises(ValueError, match="Destination exists"):
        standardize(tmp_path, apply=True)
    assert (tmp_path / "history.json").is_file()
    assert (tmp_path / "latest_model.pt").read_bytes() == b"state"


def test_relocation_only_shares_byte_identical_rolling_states(tmp_path):
    (tmp_path / "latest_model.pt").write_bytes(b"same exact state")
    (tmp_path / "last.pt").write_bytes(b"same exact state")
    standardize(tmp_path, apply=True)
    assert (tmp_path / "checkpoints/last.pt").stat().st_ino == (tmp_path / "checkpoints/latest_model.pt").stat().st_ino
    receipt = json.loads((tmp_path / "logs/artifact_relocation.json").read_text())
    assert len(receipt["byte_identical_checkpoint_aliases"]) == 1


def test_relocation_refuses_alias_collision_before_any_move(tmp_path):
    (tmp_path / "history.json").write_text('[]')
    (tmp_path / "latest.pt").write_bytes(b"one saved state")
    (tmp_path / "latest_model.pt").write_bytes(b"another saved state")
    with pytest.raises(ValueError, match="Multiple source artifacts"):
        standardize(tmp_path, apply=True)
    assert (tmp_path / "history.json").is_file()
    assert (tmp_path / "latest.pt").read_bytes() == b"one saved state"
    assert (tmp_path / "latest_model.pt").read_bytes() == b"another saved state"


def test_relocation_refuses_dangling_destination_link(tmp_path):
    (tmp_path / "latest_model.pt").write_bytes(b"saved state")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/latest_model.pt").symlink_to("missing.pt")
    with pytest.raises(ValueError, match="Destination exists"):
        standardize(tmp_path, apply=True)
    assert (tmp_path / "latest_model.pt").read_bytes() == b"saved state"
    assert (tmp_path / "checkpoints/latest_model.pt").is_symlink()
