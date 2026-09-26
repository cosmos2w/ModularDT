from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from train import _resolve_resume_run_dir, _validate_managed_resume


def test_managed_checkpoint_resolves_run_root_and_rejects_identity_conflict(tmp_path) -> None:
    run_dir = tmp_path / "Run_2105_fixture"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "latest.pt"
    checkpoint.write_bytes(b"fixture")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "case_id": "ThermalChannel",
                "model_family": "honf_forward",
                "workflow": "forward",
            }
        ),
        encoding="utf-8",
    )

    resolved = _resolve_resume_run_dir(checkpoint)

    assert resolved == run_dir
    assert resolved != checkpoint_dir
    bundle = SimpleNamespace(
        effective={
            "case": {"id": "WindFarm"},
            "model_family": "honf_forward",
        }
    )
    with pytest.raises(ValueError, match="Resume run identity mismatch"):
        _validate_managed_resume(resolved, bundle, "forward")


def test_checkpoint_directory_without_nearby_manifest_is_rejected(tmp_path) -> None:
    checkpoint = tmp_path / "unmanaged" / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")

    with pytest.raises(FileNotFoundError, match="refusing to treat a managed checkpoints directory"):
        _resolve_resume_run_dir(checkpoint)


def test_standalone_checkpoint_keeps_its_parent_as_resume_directory(tmp_path) -> None:
    checkpoint = tmp_path / "standalone.pt"
    checkpoint.write_bytes(b"fixture")

    assert _resolve_resume_run_dir(checkpoint) == tmp_path
