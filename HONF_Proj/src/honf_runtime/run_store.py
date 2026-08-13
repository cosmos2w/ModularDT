"""Standard run-directory naming, reservation, and provenance snapshots."""

from __future__ import annotations

import json
import hashlib
import csv
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config_loader import ConfigBundle
from .paths import PROJECT_ROOT, resolve_path
from .reproducibility import environment_snapshot, source_state_snapshot


def _safe_name(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip()).strip("_")
    return cleaned or "run"


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> None:
    """Materialize a canonical artifact without duplicating large files when possible."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


@dataclass(frozen=True)
class RunProposal:
    """A run path chosen before any directory is created."""

    run_uuid: str
    run_id: str
    display_name: str
    path: Path
    created_at: str


class RunStore:
    """Resolve and create standardized case/model-family result directories."""

    def __init__(self, output_root: str | Path, *, project_root: str | Path = PROJECT_ROOT):
        self.project_root = Path(project_root).resolve()
        self.output_root = resolve_path(output_root, project_root=self.project_root)

    def family_root(
        self,
        *,
        case_id: str,
        workflow: str,
        model_family: str,
        local_module_id: str | None = None,
    ) -> Path:
        case_root = self.output_root / _safe_name(case_id)
        if workflow == "local_module":
            return case_root / "Local_Module_Runs" / _safe_name(local_module_id or "default")
        if model_family == "honf_forward":
            return case_root / "HONF_Forward_Runs"
        if model_family == "honf_inverse":
            return case_root / "HONF_Inverse_Runs"
        return case_root / "Baselines" / _safe_name(model_family) / "Runs"

    def propose(
        self,
        *,
        case_id: str,
        workflow: str,
        model_family: str,
        run_id: str,
        run_name: str,
        local_module_id: str | None = None,
    ) -> RunProposal:
        now = datetime.now(timezone.utc)
        stamp = now.astimezone().strftime("%Y%m%d_%H%M%S")
        root = self.family_root(
            case_id=case_id,
            workflow=workflow,
            model_family=model_family,
            local_module_id=local_module_id,
        )
        existing = sorted(path for path in root.glob(f"Run_{run_id}_*") if path.is_dir()) if root.exists() else []
        if existing:
            formatted = "\n  ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Run_ID={run_id} is already used in {root}. Choose another Run ID or resume explicitly:\n  {formatted}"
            )
        path = root / f"Run_{run_id}_{stamp}_{_safe_name(run_name)}"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing run directory: {path}")
        return RunProposal(
            run_uuid=str(uuid.uuid4()),
            run_id=run_id,
            display_name=str(run_name),
            path=path,
            created_at=now.isoformat(),
        )

    def create(
        self,
        proposal: RunProposal,
        bundle: ConfigBundle,
        *,
        launch_facts: Mapping[str, Any] | None = None,
    ) -> Path:
        """Atomically reserve a proposed path and save configuration sources."""

        proposal.path.parent.mkdir(parents=True, exist_ok=True)
        proposal.path.mkdir(exist_ok=False)
        config_dir = proposal.path / "configs"
        config_dir.mkdir()
        for relative in (
            "checkpoints",
            "metrics",
            "plots/training",
            "plots/diagnostics",
            "evaluations",
            "comparisons",
            "logs",
            "environment",
        ):
            (proposal.path / relative).mkdir(parents=True, exist_ok=True)
        atomic_write_json(config_dir / "core_source.json", bundle.core_source_payload or bundle.core)
        atomic_write_json(config_dir / "case_source.json", bundle.case_source_payload or bundle.case)
        if bundle.experiment_source is not None:
            atomic_write_json(config_dir / "experiment_overlay.json", bundle.experiment)
        atomic_write_json(config_dir / "cli_overrides.json", bundle.overrides)
        atomic_write_json(config_dir / "resolved_config.json", bundle.effective)
        provenance = {
            "schema_version": 1,
            "core_source": str(bundle.core_source),
            "core_source_sha256": _file_sha256(bundle.core_source),
            "core_schema_version": bundle.core.get("schema_version"),
            "case_source": str(bundle.case_source),
            "case_source_sha256": _file_sha256(bundle.case_source),
            "case_schema_version": bundle.case.get("schema_version"),
            "resolved_config_sha256": bundle.config_hash,
            "cli_overrides": bundle.overrides,
            "experiment_overlay": None if bundle.experiment_source is None else str(bundle.experiment_source),
            "experiment_overlay_sha256": (
                None if bundle.experiment_source is None else _file_sha256(bundle.experiment_source)
            ),
        }
        atomic_write_json(config_dir / "config_provenance.json", provenance)
        software = environment_snapshot()
        source_state = source_state_snapshot(self.project_root)
        atomic_write_json(proposal.path / "environment" / "software.json", software)
        atomic_write_json(proposal.path / "environment" / "source_state.json", source_state)
        manifest = {
            "schema_version": 1,
            "run_uuid": proposal.run_uuid,
            "run_id": proposal.run_id,
            "display_name": proposal.display_name,
            "status": "created",
            "created_at": proposal.created_at,
            "started_at": None,
            "ended_at": None,
            "updated_at": proposal.created_at,
            "case_id": bundle.effective["case"]["id"],
            "case_schema_version": bundle.case.get("schema_version"),
            "case_plugin": bundle.case.get("plugin"),
            "model_family": bundle.effective["model_family"],
            "core_schema_version": bundle.core.get("schema_version"),
            "workflow": bundle.effective["workflow"],
            "config_sha256": bundle.config_hash,
            "core_config": str(bundle.core_source),
            "case_config": str(bundle.case_source),
            "launch_resources": dict(launch_facts or {}),
            "environment": software,
            "source_state": source_state,
            "checkpoints": {},
            "evaluations": [],
        }
        atomic_write_json(proposal.path / "run_manifest.json", manifest)
        return proposal.path

    @staticmethod
    def finalize_artifacts(run_dir: Path) -> dict[str, str]:
        """Populate the canonical artifact tree while retaining legacy workflow names."""

        run_dir = Path(run_dir).resolve()
        checkpoint_aliases = {
            "best_total": ("best_model.pt", "best_total.pt"),
            "best_field": ("best_by_field_mse_model.pt", "best_field.pt"),
            "best_temperature": ("best_by_temperature_mse_model.pt", "best_temperature.pt"),
            "best_autonomous": ("best_predicted_model.pt", "best_autonomous.pt"),
            "latest": ("latest_model.pt", "latest.pt"),
        }
        inventory: dict[str, str] = {}
        for selector, (legacy_name, canonical_name) in checkpoint_aliases.items():
            source = run_dir / legacy_name
            if source.is_file():
                destination = run_dir / "checkpoints" / canonical_name
                _link_or_copy(source, destination)
                inventory[selector] = str(destination)

        history = next((run_dir / name for name in ("metrics.csv", "loss_history.csv") if (run_dir / name).is_file()), None)
        if history is not None:
            _link_or_copy(history, run_dir / "metrics" / "metrics.csv")
        summary = run_dir / "summary.json"
        if summary.is_file():
            _link_or_copy(summary, run_dir / "metrics" / "summary.json")
        for plot in run_dir.glob("*.png"):
            _link_or_copy(plot, run_dir / "plots" / "training" / plot.name)
        for directory_name in ("diagnostics", "diagnostic_plots"):
            diagnostics = run_dir / directory_name
            if diagnostics.is_dir():
                for plot in diagnostics.rglob("*.png"):
                    relative = plot.relative_to(diagnostics)
                    _link_or_copy(plot, run_dir / "plots" / "diagnostics" / relative)
        return inventory

    @staticmethod
    def metric_summary(run_dir: Path) -> dict[str, Any]:
        """Extract last epoch and workflow-provided best metrics for the manifest."""

        run_dir = Path(run_dir)
        details: dict[str, Any] = {}
        history = next((run_dir / name for name in ("metrics.csv", "loss_history.csv") if (run_dir / name).is_file()), None)
        if history is not None:
            with history.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if rows:
                details["last_completed_epoch"] = int(float(rows[-1]["epoch"]))
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            details["best_metrics"] = {
                key: float(value)
                for key, value in summary.items()
                if key.startswith("best_") and isinstance(value, (int, float))
            }
        return details

    @staticmethod
    def update_status(run_dir: Path, status: str, **details: Any) -> None:
        manifest_path = run_dir / "run_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        normalized_status = str(status)
        now = datetime.now(timezone.utc).isoformat()
        manifest["status"] = normalized_status
        manifest["updated_at"] = now
        if normalized_status == "running":
            manifest["started_at"] = manifest.get("started_at") or now
            manifest["ended_at"] = None
        elif normalized_status != "created":
            # Every non-running state ends this invocation. An explicit resume
            # clears ``ended_at`` while preserving the original start time.
            manifest["ended_at"] = now
        if normalized_status in {"running", "completed"} or normalized_status.startswith(("stopped", "cancelled")):
            for stale_key in ("error_type", "error_message", "traceback"):
                manifest.pop(stale_key, None)
        manifest.update(details)
        atomic_write_json(manifest_path, manifest)

    @staticmethod
    def record_evaluation(run_dir: Path, evaluation_dir: Path) -> None:
        """Append one post-processing child to an existing run manifest."""

        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            return
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        entries = list(manifest.get("evaluations", []))
        value = str(evaluation_dir.resolve())
        if value not in entries:
            entries.append(value)
        manifest["evaluations"] = entries
        atomic_write_json(manifest_path, manifest)
