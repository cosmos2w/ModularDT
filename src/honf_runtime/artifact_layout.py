"""Canonical, category-oriented paths for managed run artifacts.

Historical HONF workflows wrote plots and evaluations beside checkpoints.  New
managed runs keep those readers working, but producers use this module so one
artifact has one canonical location.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


EVALUATION_LAYOUT_VERSION = 2


def find_managed_run_dir(path: str | Path) -> Path | None:
    """Return the nearest ancestor carrying a managed ``run_manifest.json``."""

    resolved = Path(path).expanduser().resolve()
    candidate = resolved if resolved.is_dir() else resolved.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "run_manifest.json").is_file():
            return directory
    return None


def default_evaluation_root(
    checkpoint_path: str | Path,
    *,
    evaluation_kind: str = "single_case",
) -> Path:
    """Resolve the canonical root for an evaluation associated with a checkpoint."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    run_dir = find_managed_run_dir(checkpoint)
    owner = run_dir if run_dir is not None else checkpoint.parent
    return owner / "evaluations" / str(evaluation_kind)


@dataclass(frozen=True)
class EvaluationArtifactLayout:
    """Named subdirectories for one timestamped evaluation job."""

    root: Path

    @classmethod
    def at(cls, root: str | Path) -> "EvaluationArtifactLayout":
        path = Path(root).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return cls(root=path)

    @property
    def fields(self) -> Path:
        return self.root / "fields"

    @property
    def organization(self) -> Path:
        return self.root / "organization"

    @property
    def routing(self) -> Path:
        return self.root / "routing"

    @property
    def topology(self) -> Path:
        return self.root / "topology"

    @property
    def plans(self) -> Path:
        return self.root / "plans"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def arrays(self) -> Path:
        return self.root / "arrays"

    @property
    def diagnostics(self) -> Path:
        return self.root / "diagnostics"

    def ensure(self, *categories: str) -> None:
        """Create only the category directories required by this evaluation."""

        for category in categories:
            path = getattr(self, category)
            if not isinstance(path, Path):
                raise ValueError(f"Unknown evaluation artifact category: {category!r}")
            path.mkdir(parents=True, exist_ok=True)


def evaluation_artifact_inventory(
    root: str | Path,
    *,
    excluded_names: Iterable[str] = ("evaluation_manifest.json",),
) -> list[dict[str, object]]:
    """Return a stable relative-path inventory annotated by top-level category."""

    directory = Path(root).resolve()
    excluded = set(excluded_names)
    inventory: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        relative = path.relative_to(directory)
        inventory.append(
            {
                "path": str(relative),
                "category": relative.parts[0] if len(relative.parts) > 1 else "job",
                "size_bytes": path.stat().st_size,
            }
        )
    return inventory


def finalize_evaluation_job(
    root: str | Path,
    *,
    kind: str,
    checkpoint_path: str | Path,
    requested_checkpoint: str | None = None,
) -> Path:
    """Write the exact job inventory and attach it to its managed source run."""

    # Local import avoids making the run store depend on this layout module.
    from .run_store import RunStore, atomic_write_json

    directory = Path(root).resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    run_dir = find_managed_run_dir(checkpoint)
    manifest_path = directory / "evaluation_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 2,
            "artifact_layout_version": EVALUATION_LAYOUT_VERSION,
            "status": "completed",
            "kind": str(kind),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_run": None if run_dir is None else str(run_dir),
            "requested_checkpoint": requested_checkpoint,
            "resolved_checkpoint": str(checkpoint),
            "artifacts": evaluation_artifact_inventory(directory),
        },
    )
    if run_dir is not None:
        RunStore.record_evaluation(run_dir, directory)
    return manifest_path
