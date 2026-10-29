from __future__ import annotations

import json
from pathlib import Path

from honf_runtime.run_layout import RunLayout
from windfarm.workflows import train_forward


def test_canonical_best_field_checkpoint_has_atomic_compatibility_alias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "canonical-run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"artifact_layout_version": 1}), encoding="utf-8"
    )
    layout = RunLayout(run_dir)
    layout.ensure()
    alias_path = layout.write_path("best_model.pt")
    alias_path.write_bytes(b"superseded checkpoint")
    saved_paths: list[Path] = []

    def fake_save(path: Path, **_kwargs: object) -> None:
        saved_paths.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"best field checkpoint")

    monkeypatch.setattr(train_forward, "_save_checkpoint", fake_save)
    train_forward._save_best_checkpoint(layout, test_payload=True)

    canonical_path = layout.path("best_by_field_mse_model.pt")
    assert saved_paths == [canonical_path]
    assert alias_path.is_symlink()
    assert alias_path.resolve() == canonical_path.resolve()
    assert alias_path.read_bytes() == b"best field checkpoint"
    physical_checkpoints = [
        path for path in canonical_path.parent.glob("*.pt") if not path.is_symlink()
    ]
    assert physical_checkpoints == [canonical_path]


def test_legacy_best_checkpoint_names_keep_separate_root_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "legacy-run"
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text("epoch\n", encoding="utf-8")
    layout = RunLayout(run_dir)
    assert not layout.canonical_writes
    saved_paths: list[Path] = []

    def fake_save(path: Path, **_kwargs: object) -> None:
        saved_paths.append(path)
        path.write_bytes(path.name.encode("utf-8"))

    monkeypatch.setattr(train_forward, "_save_checkpoint", fake_save)
    train_forward._save_best_checkpoint(layout, test_payload=True)

    assert saved_paths == [run_dir / "best_model.pt", run_dir / "best_by_field_mse_model.pt"]
    assert all(path.is_file() and not path.is_symlink() for path in saved_paths)
