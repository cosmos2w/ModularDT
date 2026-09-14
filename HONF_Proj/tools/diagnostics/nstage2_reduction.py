#!/usr/bin/env python3
"""Reduce the bounded NStage2 endpoint evidence.

This module only reads existing evaluation tables and managed metric histories.
It writes computed comparison artifacts beneath the supplied NStage2 study root;
it never evaluates a model, selects a checkpoint by copying it, or invents a
score for an unavailable candidate.

``reduce_nstage2`` accepts either the repository root or the study root.  For a
repository root, the study root defaults to
``diagnostics/generated/interface_operator_study/nstage2`` for the historical
epoch-500 endpoint.  An explicit mature endpoint of 5000 uses the sibling
``nstage2/maturity5000`` namespace.  The endpoint table layout is deliberately
explicit::

    nstage2/<track>/endpoint<epoch>/tables/
    nstage2/<track>/best_field/tables/

The maintained parent exact and saved-best table sets are read from their
historical locations.  Their files are referenced in the manifest and are
never copied.
If a candidate has no separate saved-best table set, its exact-endpoint tables
may also be referenced for the saved-best phase only after the trusted
saved-best checkpoint proves to be the same endpoint model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["reduce_nstage2"]

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Reuse the maintained reader and reduction definitions.  They intentionally
# remain the source of truth for the existing five-model tables.
from analyze_honf_maturity import (  # type: ignore[import-not-found]
    CHANNELS,
    STRATA,
    number,
    pooled,
    read_csv,
    write_csv,
)

_write_csv = write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_RELATIVE = Path("diagnostics/generated/interface_operator_study")
PARENT_STUDY_NAME = "five_model_epoch5000"
RUN_ROOT_RELATIVE = Path("Trained_Results/ThermalChannel/HONF_Forward_Runs")
DEFAULT_ENDPOINT_EPOCH = 500
MATURE_ENDPOINT_EPOCH = 5000

PARENT_RUNS = ("1401", "1801", "1804", "1805", "1806")
CANDIDATE_RUNS = ("1807", "1808")
RUN_LABELS = {
    "1401": "Legacy 1401",
    "1801": "Latent 1801",
    "1804": "Dense 1804",
    "1805": "Reader 1805",
    "1806": "Regional 1806",
    "1807": "NStage2-A Hierarchical Regional 1807",
    "1808": "NStage2-B Group-Mediated Reader 1808",
}
RUN_TRACK = {"1807": "track_a", "1808": "track_b"}
ANCHORS = ("0273", "0653", "0283", "0298", "0302")
ANCHOR_REASONS = {
    "0273": "established interface anchor; fixed before NStage2 results",
    "0653": "established interface anchor; fixed before NStage2 results",
    "0298": "established difficult anchor; fixed before NStage2 results",
    "0302": "established interface anchor; fixed before NStage2 results",
    "0283": "mature Dense/Regional difficult case; fixed as the fifth anchor before NStage2 results",
}

# The historical report has only metric minima for these epochs.  There is no
# corresponding matched parent checkpoint table to select, so the reducer
# records the audit fact and never turns these minima into a parent comparison.
PARENT_HISTORY_MINIMA = {
    "1401": 466,
    "1801": 490,
    "1804": 493,
    "1805": 468,
    "1806": 491,
}

PAIR_REQUESTS = (
    ("1807", "1806", "A_vs_1806"),
    ("1807", "1804", "A_vs_1804"),
    ("1808", "1805", "B_vs_1805"),
)

# These bases cover the requested global, near/far, channel, port, internal
# temperature, and flux views.  The table reducer also records any absent
# metric explicitly through the source/schema manifest rather than filling it.
CORE_BASES = (
    "global_field_fluid_norm",
    "global_field_near_interface_norm",
    "global_field_far_fluid_norm",
    "global_field_local_radius_norm",
    "global_field_outside_local_radius_norm",
    "internal_temperature_physical",
    "interface_t_surface_physical",
    "interface_q_normal_physical",
    "port_t_env_final_physical",
    "port_h_effective_final_physical",
    "port_t_env_provisional_physical",
    "port_h_effective_provisional_physical",
)
CHANNEL_BASES = tuple(f"field_{channel}_fluid_norm" for channel in CHANNELS) + tuple(
    f"field_{channel}_fluid_physical" for channel in CHANNELS
)
# These are the maintained scalar engineering-error columns in the endpoint
# table schema.  They are deliberately enumerated from the existing evaluator
# output; no new KPI or derived target is introduced here.
ENGINEERING_KPI_BASES = (
    "pressure_drop_inlet_minus_outlet_physical_error",
    "pressure_drop_inlet_minus_outlet_physical_abs_error",
    "pressure_drop_inlet_minus_outlet_physical_relative_error",
    "mean_outlet_temperature_physical_error",
    "mean_outlet_temperature_physical_abs_error",
    "mean_outlet_temperature_physical_relative_error",
    "mean_active_module_temperature_physical_error",
    "mean_active_module_temperature_physical_abs_error",
    "mean_active_module_temperature_physical_relative_error",
)
POOL_BASES = tuple(dict.fromkeys(CORE_BASES + CHANNEL_BASES))
STRATA_BASES = (
    "global_field_fluid_norm",
    "global_field_near_interface_norm",
    "global_field_far_fluid_norm",
    "internal_temperature_physical",
    "interface_q_normal_physical",
    "port_t_env_final_physical",
    "port_h_effective_final_physical",
    "port_t_env_provisional_physical",
    "port_h_effective_provisional_physical",
)
PAIR_BASES = tuple(dict.fromkeys(CORE_BASES + CHANNEL_BASES + ENGINEERING_KPI_BASES))
HISTORY_WINDOW_SIZES = (50, 100, 250, 500, 1000)
HISTORY_MILESTONES = (500, 1000, 2500, 5000)

_EPOCH_TOKEN = re.compile(r"epoch[_-](\d+)", re.IGNORECASE)
_RUN_TOKEN = re.compile(r"Run[_-](\d+)(?:[_-]|$)", re.IGNORECASE)
_EXACT_PHASE_TOKEN = re.compile(r"exact(\d+)$", re.IGNORECASE)


def _validate_endpoint_epoch(endpoint_epoch: int) -> int:
    """Validate and normalize a requested endpoint epoch."""

    if isinstance(endpoint_epoch, bool):
        raise TypeError("endpoint_epoch must be a positive integer")
    try:
        value = int(endpoint_epoch)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"endpoint_epoch must be a positive integer: {endpoint_epoch!r}") from exc
    if value <= 0:
        raise ValueError(f"endpoint_epoch must be a positive integer: {endpoint_epoch!r}")
    return value


def _exact_phase(endpoint_epoch: int) -> str:
    """Return the phase label whose number matches the endpoint exactly."""

    return f"exact{_validate_endpoint_epoch(endpoint_epoch)}"


def _phase_endpoint_epoch(phase: str) -> int | None:
    """Parse an exact endpoint epoch from a phase label when one is present."""

    match = _EXACT_PHASE_TOKEN.fullmatch(str(phase))
    return None if match is None else int(match.group(1))


@dataclass
class TableSet:
    """One phase/run table source and its validated rows."""

    run: str
    model: str
    phase: str
    table_dir: Path
    status: str
    reason: str | None = None
    rows: list[dict[str, str]] | None = None
    summary: dict[str, str] | None = None
    checkpoint: str | None = None
    checkpoint_epoch: int | None = None

    @property
    def available(self) -> bool:
        return self.status == "available" and self.rows is not None and self.summary is not None

    def status_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "run": self.run,
            "model": self.model,
            "phase": self.phase,
            "status": self.status,
            "table_dir": str(self.table_dir),
        }
        if self.reason:
            row["reason"] = self.reason
        if self.checkpoint:
            row["checkpoint"] = self.checkpoint
        if self.checkpoint_epoch is not None:
            row["checkpoint_epoch"] = self.checkpoint_epoch
        return row


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _resolve_roots(
    root: str | Path | None,
    endpoint_epoch: int = DEFAULT_ENDPOINT_EPOCH,
) -> tuple[Path, Path]:
    """Resolve ``(project_root, nstage2_study_root)`` from either root form."""

    endpoint_epoch = _validate_endpoint_epoch(endpoint_epoch)
    supplied = PROJECT_ROOT if root is None else Path(root).expanduser()
    supplied = supplied.resolve()
    project: Path | None = None
    for candidate in (supplied, *supplied.parents):
        if (candidate / "src").is_dir() and (candidate / "diagnostics").is_dir():
            project = candidate
            break
    if project is None:
        project = PROJECT_ROOT

    if supplied.name == "nstage2" or (supplied / "track_a").exists() or (supplied / "track_b").exists():
        study = supplied
        # Keep the historical default rooted directly at nstage2, while the
        # mature evaluation has its own namespace beneath that study root.
        if endpoint_epoch != DEFAULT_ENDPOINT_EPOCH and supplied.name == "nstage2":
            study = supplied / f"maturity{endpoint_epoch}"
    elif supplied.name == PARENT_STUDY_NAME:
        study = supplied.parent / "nstage2"
    elif supplied == project or (supplied / "diagnostics").is_dir():
        study = project / STUDY_RELATIVE / "nstage2"
    else:
        # A non-existent path is most useful as an explicitly requested study
        # root; this also makes temporary fixture roots straightforward.
        study = supplied
    if endpoint_epoch != DEFAULT_ENDPOINT_EPOCH and study.name == "nstage2":
        # Apply the mature namespace after every accepted root form (project,
        # nstage2, parent maturity study, or the default root).  Without this
        # final normalization, ``--endpoint-epoch 5000`` with no --root would
        # write into the historical endpoint-500 comparison.
        study = study / f"maturity{endpoint_epoch}"
    return project, study


def _parent_table_dir(
    project: Path,
    run: str,
    endpoint_epoch: int = DEFAULT_ENDPOINT_EPOCH,
) -> Path:
    endpoint_epoch = _validate_endpoint_epoch(endpoint_epoch)
    study = project / STUDY_RELATIVE
    if endpoint_epoch == DEFAULT_ENDPOINT_EPOCH:
        if run == "1805":
            return study / "group_reader_recovery" / "endpoint" / "tables"
        if run == "1806":
            return study / "regional_response" / "endpoint500" / "tables"
        return (
            project
            / RUN_ROOT_RELATIVE
            / "CompareModels"
            / "Stage3_Run1401_1804_1801_1802_Epoch500_90Case"
            / "tables"
        )

    # The maintained five-model maturity reducer uses the same source split:
    # 1801/1806 were added to five_model_epoch5000/evaluation, while the
    # original 1401/1804/1805 endpoint tables remain in the dedicated
    # epoch5000 comparison.  Both locations are referenced in place.
    if endpoint_epoch == MATURE_ENDPOINT_EPOCH and run in ("1801", "1806"):
        return study / PARENT_STUDY_NAME / "evaluation" / "tables"
    if endpoint_epoch == MATURE_ENDPOINT_EPOCH:
        return study / "epoch5000_comparison" / "evaluation" / "tables"

    # Match analyze_honf_maturity.py for any future explicit endpoint while
    # retaining the ordinary, named source locations.
    if run in ("1801", "1806"):
        return study / PARENT_STUDY_NAME / "evaluation" / "tables"
    return study / f"epoch{endpoint_epoch}_comparison" / "evaluation" / "tables"


def _parent_best_table_dir(project: Path, endpoint_epoch: int) -> Path:
    """Return the maintained parent saved-best table directory."""

    endpoint_epoch = _validate_endpoint_epoch(endpoint_epoch)
    if endpoint_epoch != MATURE_ENDPOINT_EPOCH:
        raise ValueError(
            "parent saved-best table sources are maintained only for the mature epoch5000 study"
        )
    return project / STUDY_RELATIVE / PARENT_STUDY_NAME / "best_field_evaluation" / "tables"


def _candidate_table_dir(
    study: Path,
    run: str,
    phase: str,
    endpoint_epoch: int = DEFAULT_ENDPOINT_EPOCH,
) -> Path:
    endpoint_epoch = _validate_endpoint_epoch(endpoint_epoch)
    track = RUN_TRACK[run]
    if phase == _exact_phase(endpoint_epoch):
        return study / track / f"endpoint{endpoint_epoch}" / "tables"
    if phase == "best_field":
        return study / track / "best_field" / "tables"
    raise ValueError(f"Unknown NStage2 table phase {phase!r}.")


def _legacy_candidate_table_dir(study: Path, run: str, endpoint_epoch: int) -> Path:
    """Resolve a candidate trajectory table, including the prior NStage2 root."""

    endpoint_epoch = _validate_endpoint_epoch(endpoint_epoch)
    current = study / RUN_TRACK[run] / f"endpoint{endpoint_epoch}" / "tables"
    if current.is_dir() or study.name != f"maturity{MATURE_ENDPOINT_EPOCH}":
        return current
    # Mature evaluation intentionally keeps the historical candidate 500
    # tables under nstage2.  Reference them in place when available; never
    # materialize a second copy under maturity5000.
    return study.parent / RUN_TRACK[run] / f"endpoint{endpoint_epoch}" / "tables"


def _checkpoint_run(row: Mapping[str, Any], run: str) -> bool:
    checkpoint = str(row.get("checkpoint", ""))
    row_run = str(row.get("run", ""))
    return bool(
        f"Run_{run}_" in checkpoint
        or f"Run-{run}-" in checkpoint
        or row_run == run
        or _RUN_TOKEN.search(checkpoint or "") and _RUN_TOKEN.search(checkpoint).group(1) == run
    )


def _checkpoint_epoch_from_name(checkpoint: str | None) -> int | None:
    if not checkpoint:
        return None
    match = _EPOCH_TOKEN.search(str(checkpoint))
    return None if match is None else int(match.group(1))


def _filter_table_rows(
    rows: list[dict[str, str]],
    run: str,
    phase: str,
    endpoint_epoch: int | None = None,
) -> list[dict[str, str]]:
    selected = [row for row in rows if _checkpoint_run(row, run)]
    # A table directory is already run-owned for NStage2 candidates.  Permit a
    # missing Run_<id> token there, but only when the whole table is one exact
    # 90-case set; never mix multiple checkpoint sets silently.
    if not selected and len(rows) == 90:
        selected = list(rows)
    phase_epoch = _phase_endpoint_epoch(phase) if endpoint_epoch is None else _validate_endpoint_epoch(endpoint_epoch)
    if phase_epoch is not None:
        exact = [
            row
            for row in selected
            if _checkpoint_epoch_from_name(row.get("checkpoint")) == phase_epoch
            or str(row.get("epoch", "")) == str(phase_epoch)
        ]
        selected = exact
    return selected


def _summary_rows(
    rows: list[dict[str, str]],
    run: str,
    phase: str,
    endpoint_epoch: int | None = None,
) -> list[dict[str, str]]:
    selected = [row for row in rows if _checkpoint_run(row, run)]
    if not selected and len(rows) == 1:
        selected = list(rows)
    phase_epoch = _phase_endpoint_epoch(phase) if endpoint_epoch is None else _validate_endpoint_epoch(endpoint_epoch)
    if phase_epoch is not None:
        exact = [
            row
            for row in selected
            if _checkpoint_epoch_from_name(row.get("checkpoint")) == phase_epoch
            or str(row.get("epoch", "")) == str(phase_epoch)
        ]
        selected = exact
    return selected


def _checkpoint_path(project: Path, table_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((project / path, table_dir.parent / path, table_dir / path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return path.resolve() if path.is_absolute() else None


def _load_trusted_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load one local checkpoint on CPU through the maintained trusted loader."""

    from honf_runtime.compat import load_trusted_checkpoint

    return load_trusted_checkpoint(path, map_location="cpu")


def _load_checkpoint_epoch(project: Path, table_dir: Path, checkpoint: str | None) -> int | None:
    parsed = _checkpoint_epoch_from_name(checkpoint)
    if parsed is not None:
        return parsed
    path = _checkpoint_path(project, table_dir, checkpoint)
    if path is None:
        return None
    try:
        state = _load_trusted_checkpoint(path)
    except Exception as exc:  # pragma: no cover - depends on checkpoint format
        raise ValueError(f"Could not read selected checkpoint epoch from {path}: {exc}") from exc
    for key in ("epoch", "current_epoch"):
        if key in state:
            try:
                return int(state[key])
            except (TypeError, ValueError):
                pass
    return None


def _checkpoint_epoch_from_payload(payload: Mapping[str, Any]) -> int | None:
    for key in ("epoch", "current_epoch"):
        if key not in payload:
            continue
        try:
            return int(payload[key])
        except (TypeError, ValueError):
            pass
    return None


def _tensor_values_match(left: Any, right: Any) -> bool:
    """Compare two checkpoint tensor values exactly after moving them to CPU."""

    try:
        import torch

        if torch.is_tensor(left) and torch.is_tensor(right):
            left_cpu = left.detach().cpu()
            right_cpu = right.detach().cpu()
            return (
                tuple(left_cpu.shape) == tuple(right_cpu.shape)
                and left_cpu.dtype == right_cpu.dtype
                and bool(torch.equal(left_cpu, right_cpu))
            )
    except (ImportError, AttributeError, TypeError, RuntimeError):
        pass
    try:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
    except (TypeError, ValueError):
        return False
    return (
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and bool(np.array_equal(left_array, right_array))
    )


def _model_state_dicts_match(
    endpoint_payload: Mapping[str, Any], best_payload: Mapping[str, Any]
) -> tuple[bool, int, str | None]:
    """Return whether two checkpoint model states have the same tensor values."""

    endpoint_state = endpoint_payload.get("model_state_dict")
    best_state = best_payload.get("model_state_dict")
    if not isinstance(endpoint_state, Mapping) or not isinstance(best_state, Mapping):
        return False, 0, "checkpoint is missing a mapping-valued model_state_dict"
    endpoint_keys = set(endpoint_state)
    best_keys = set(best_state)
    if endpoint_keys != best_keys:
        missing = sorted(endpoint_keys - best_keys)
        extra = sorted(best_keys - endpoint_keys)
        return False, 0, f"model_state_dict keys differ (missing={missing}, extra={extra})"
    for key in sorted(endpoint_keys):
        if not _tensor_values_match(endpoint_state[key], best_state[key]):
            return False, len(endpoint_keys), f"model_state_dict tensor differs at {key!r}"
    return True, len(endpoint_keys), None


def _candidate_run_dir(project: Path, run: str, endpoint: TableSet) -> Path | None:
    """Resolve the managed candidate run directory owning the endpoint checkpoint."""

    endpoint_path = _checkpoint_path(project, endpoint.table_dir, endpoint.checkpoint)
    if endpoint_path is not None:
        for candidate in (endpoint_path.parent, endpoint_path.parent.parent):
            if candidate.name.startswith(f"Run_{run}_") and candidate.is_dir():
                return candidate.resolve()
    run_root = project / RUN_ROOT_RELATIVE
    candidates = sorted(path.resolve() for path in run_root.glob(f"Run_{run}_*") if path.is_dir())
    return candidates[0] if len(candidates) == 1 else None


def _candidate_best_checkpoint(project: Path, run: str, endpoint: TableSet) -> Path | None:
    """Find the saved field-best checkpoint for the candidate owning endpoint."""

    run_dir = _candidate_run_dir(project, run, endpoint)
    if run_dir is None:
        return None
    # The historical filename is the policy identity.  The canonical alias is
    # accepted only when a managed run has already materialized that alias.
    for candidate in (
        run_dir / "best_by_field_mse_model.pt",
        run_dir / "checkpoints" / "best_field.pt",
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _reuse_exact_endpoint_for_best(
    project: Path,
    run: str,
    endpoint: TableSet,
    best_table_dir: Path,
) -> TableSet:
    """Reuse endpoint tables for saved-best only after strict checkpoint proof."""

    declared_endpoint_epoch = endpoint.checkpoint_epoch
    target_endpoint_epoch = declared_endpoint_epoch or _phase_endpoint_epoch(endpoint.phase)
    if target_endpoint_epoch is None:
        target_endpoint_epoch = _checkpoint_epoch_from_name(endpoint.checkpoint)
    if target_endpoint_epoch is None:
        return _unavailable(
            run,
            "best_field",
            best_table_dir,
            "saved-best table reuse refused because the exact endpoint epoch is unavailable",
        )

    if not endpoint.available:
        return _unavailable(
            run,
            "best_field",
            best_table_dir,
            "saved-best table reuse refused because the candidate "
            f"exact{target_endpoint_epoch} endpoint tables are unavailable",
        )
    endpoint_path = _checkpoint_path(project, endpoint.table_dir, endpoint.checkpoint)
    best_path = _candidate_best_checkpoint(project, run, endpoint)
    if endpoint_path is None or not endpoint_path.is_file():
        return _unavailable(
            run,
            "best_field",
            best_table_dir,
            "saved-best table reuse refused because the "
            f"exact{target_endpoint_epoch} checkpoint is unavailable: {endpoint.checkpoint!r}",
        )
    if best_path is None:
        return _unavailable(
            run,
            "best_field",
            best_table_dir,
            "saved-best table reuse refused because best_by_field_mse_model.pt is unavailable",
        )
    try:
        endpoint_payload = _load_trusted_checkpoint(endpoint_path)
        best_payload = _load_trusted_checkpoint(best_path)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:  # pragma: no cover - local checkpoint files
        return _unavailable(
            run,
            "best_field",
            best_table_dir,
            f"saved-best table reuse refused because trusted checkpoint loading failed: {exc}",
        )
    endpoint_epoch = _checkpoint_epoch_from_payload(endpoint_payload)
    if endpoint_epoch is None:
        endpoint_epoch = _checkpoint_epoch_from_name(str(endpoint_path))
    if endpoint_epoch is None:
        endpoint_epoch = target_endpoint_epoch
    best_epoch = _checkpoint_epoch_from_payload(best_payload)
    if endpoint_epoch != target_endpoint_epoch:
        # Prefer the trusted payload epoch when it is available.  A table set
        # carrying an inconsistent checkpoint label must not silently become a
        # saved-best result for another endpoint.
        return _unavailable(
            run,
            "best_field",
            best_table_dir,
            "saved-best table reuse refused because the endpoint checkpoint epoch disagrees with its table phase "
            f"(table_epoch={target_endpoint_epoch!r}, checkpoint_epoch={endpoint_epoch!r})",
        )
    if best_epoch != endpoint_epoch:
        return _unavailable(
            run,
            "best_field",
            best_table_dir,
            "saved-best table reuse refused because the endpoint and saved-best checkpoints are not both "
            f"epoch{endpoint_epoch} (endpoint_epoch={endpoint_epoch!r}, saved_best_epoch={best_epoch!r})",
        )
    matches, tensor_count, mismatch = _model_state_dicts_match(endpoint_payload, best_payload)
    if not matches:
        return _unavailable(
            run,
            "best_field",
            best_table_dir,
            "saved-best table reuse refused because endpoint and saved-best model weights differ"
            + (f": {mismatch}" if mismatch else ""),
        )
    if endpoint.rows is None or endpoint.summary is None:
        return _unavailable(
            run,
            "best_field",
            best_table_dir,
            "saved-best table reuse refused because endpoint rows or summary are unavailable",
        )
    reason = (
        f"reused exact{endpoint_epoch} endpoint evaluation tables in place for saved-best: "
        f"trusted CPU comparison found all {tensor_count} model_state_dict tensors equal; "
        f"endpoint and best_by_field_mse_model.pt are both epoch{endpoint_epoch}"
    )
    return TableSet(
        run,
        RUN_LABELS[run],
        "best_field",
        endpoint.table_dir,
        "available",
        reason=reason,
        rows=endpoint.rows,
        # Keep the source summary's endpoint checkpoint for table provenance;
        # the selected saved-best checkpoint is carried by TableSet.checkpoint.
        summary=endpoint.summary,
        checkpoint=str(best_path),
        checkpoint_epoch=best_epoch,
    )


def _source_manifest(table_dir: Path) -> dict[str, Any]:
    expected_files = ("per_case_metrics.csv", "model_summary_metrics.csv")
    result: dict[str, Any] = {
        "table_dir": str(table_dir),
        "expected_files": list(expected_files),
        "files": {},
        "comparison_manifest": None,
    }
    for name in expected_files:
        path = table_dir / name
        if not path.is_file():
            continue
        rows = read_csv(path)
        result["files"][name] = {
            "path": str(path),
            "row_count": len(rows),
            "field_count": len(rows[0]) if rows else 0,
            "fields": list(rows[0]) if rows else [],
        }
    for candidate in (table_dir.parent / "comparison_manifest.json", table_dir.parent.parent / "comparison_manifest.json"):
        if candidate.is_file():
            try:
                manifest = json.loads(candidate.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed comparison manifest: {candidate}") from exc
            result["comparison_manifest"] = {
                "path": str(candidate),
                "schema_version": manifest.get("schema_version"),
                "status": manifest.get("status"),
                "kind": manifest.get("kind"),
                "arguments": manifest.get("arguments", []),
            }
            break
    return result


def _unavailable(run: str, phase: str, table_dir: Path, reason: str) -> TableSet:
    return TableSet(run, RUN_LABELS[run], phase, table_dir, "unavailable", reason=reason)


def _load_table_set(project: Path, run: str, phase: str, table_dir: Path, *, required: bool) -> TableSet:
    if not table_dir.is_dir():
        reason = f"table directory is unavailable: {table_dir}"
        if required:
            raise FileNotFoundError(reason)
        return _unavailable(run, phase, table_dir, reason)
    case_path = table_dir / "per_case_metrics.csv"
    summary_path = table_dir / "model_summary_metrics.csv"
    if not case_path.is_file() or not summary_path.is_file():
        reason = f"required table files are unavailable beneath {table_dir}"
        if required:
            raise FileNotFoundError(reason)
        return _unavailable(run, phase, table_dir, reason)
    case_rows = read_csv(case_path)
    summary_rows = read_csv(summary_path)
    phase_epoch = _phase_endpoint_epoch(phase)
    selected_cases = _filter_table_rows(case_rows, run, phase)
    selected_summary = _summary_rows(summary_rows, run, phase)
    if len(selected_cases) != 90 or len({row.get("case_id") for row in selected_cases}) != 90:
        raise ValueError(
            f"Expected exactly 90 unique cases for {run}/{phase} in {case_path}; "
            f"selected {len(selected_cases)} rows."
        )
    if len(selected_summary) != 1:
        raise ValueError(
            f"Expected exactly one summary row for {run}/{phase} in {summary_path}; "
            f"selected {len(selected_summary)} rows."
        )
    summary = selected_summary[0]
    checkpoint = summary.get("checkpoint") or selected_cases[0].get("checkpoint")
    checkpoint_epoch = phase_epoch if phase_epoch is not None else _load_checkpoint_epoch(project, table_dir, checkpoint)
    if phase == "best_field" and checkpoint_epoch is None:
        reason = f"selected checkpoint epoch is unavailable for {checkpoint!r}"
        if required:
            raise FileNotFoundError(reason)
        return _unavailable(run, phase, table_dir, reason)
    return TableSet(
        run,
        RUN_LABELS[run],
        phase,
        table_dir,
        "available",
        rows=selected_cases,
        summary=summary,
        checkpoint=checkpoint,
        checkpoint_epoch=checkpoint_epoch,
    )


def _validate_against_reference(item: TableSet, reference: TableSet) -> None:
    if not item.available or not reference.available:
        return
    assert item.rows is not None and reference.rows is not None
    reference_by_case = {row["case_id"]: row for row in reference.rows}
    item_by_case = {row["case_id"]: row for row in item.rows}
    if set(item_by_case) != set(reference_by_case):
        missing = sorted(set(reference_by_case) - set(item_by_case))
        extra = sorted(set(item_by_case) - set(reference_by_case))
        raise ValueError(f"Case set mismatch for {item.run}/{item.phase}: missing={missing}, extra={extra}")
    common = set(reference.rows[0]).intersection(item.rows[0])
    target_keys = sorted(key for key in common if key.endswith(("_target_sse", "_num_values")))
    for case_id, row in item_by_case.items():
        baseline = reference_by_case[case_id]
        for key in target_keys:
            actual, expected = number(row.get(key)), number(baseline.get(key))
            if actual is None and expected is None:
                continue
            if actual is None or expected is None or not np.isclose(actual, expected, rtol=1e-7, atol=1e-7):
                raise ValueError(f"Target/count mismatch for {item.run}/{item.phase}/{case_id}/{key}")
        for key in STRATA:
            if key in row and key in baseline and row[key] != baseline[key]:
                raise ValueError(f"Stratum mismatch for {item.run}/{item.phase}/{case_id}/{key}")


def _identity(item: TableSet) -> dict[str, Any]:
    row = item.status_row()
    row.update(
        {
            "run": item.run,
            "model": item.model,
            "checkpoint": item.checkpoint or "",
            "checkpoint_epoch": "" if item.checkpoint_epoch is None else item.checkpoint_epoch,
        }
    )
    return row


def _distribution(values: Iterable[float]) -> dict[str, Any] | None:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return None
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std()),
        "p05": float(np.quantile(finite, 0.05)),
        "p95": float(np.quantile(finite, 0.95)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _raw_metric_key(base: str) -> str:
    if base in ENGINEERING_KPI_BASES:
        return base
    if base.endswith("_norm"):
        return base + "_l2"
    return base + "_relative_l2"


def _phase_rows(item: TableSet, bases: Sequence[str]) -> list[dict[str, Any]]:
    if not item.available:
        return [_identity(item)]
    assert item.rows is not None
    output: list[dict[str, Any]] = []
    for base in bases:
        result = pooled(item.rows, base)
        if result is None:
            output.append({**_identity(item), "metric": base, "metric_status": "unavailable"})
            continue
        output.append({**_identity(item), "metric": base, "metric_status": "available", **result})
    return output


def _case_metric_rows(item: TableSet, metrics: Sequence[str]) -> list[dict[str, Any]]:
    if not item.available:
        return [_identity(item)]
    assert item.rows is not None
    output: list[dict[str, Any]] = []
    for source in item.rows:
        identity = _identity(item)
        for base in metrics:
            raw_key = _raw_metric_key(base)
            value = number(source.get(raw_key))
            if value is None:
                continue
            output.append(
                {
                    **identity,
                    "case_id": source.get("case_id", ""),
                    "metric": base,
                    "raw_key": raw_key,
                    "value": value,
                    **{axis: source.get(axis, "") for axis in STRATA},
                }
            )
    return output


def _equal_case_rows(item: TableSet, metrics: Sequence[str]) -> list[dict[str, Any]]:
    if not item.available:
        return [_identity(item)]
    assert item.rows is not None
    output: list[dict[str, Any]] = []
    for base in metrics:
        key = _raw_metric_key(base)
        result = _distribution(number(row.get(key)) for row in item.rows if number(row.get(key)) is not None)
        if result is None:
            output.append({**_identity(item), "metric": base, "metric_status": "unavailable"})
        else:
            output.append({**_identity(item), "metric": base, "metric_status": "available", **result})
    return output


def _strata_rows(item: TableSet) -> list[dict[str, Any]]:
    if not item.available:
        return [_identity(item)]
    assert item.rows is not None
    output: list[dict[str, Any]] = []
    for axis in STRATA:
        labels = sorted({row.get(axis, "") for row in item.rows})
        for label in labels:
            subset = [row for row in item.rows if row.get(axis, "") == label]
            for base in STRATA_BASES:
                result = pooled(subset, base)
                if result is None:
                    continue
                output.append(
                    {
                        **_identity(item),
                        "axis": axis,
                        "stratum": label,
                        "num_cases": len(subset),
                        "metric": base,
                        **result,
                    }
                )
    return output


def _difficult_rows(item: TableSet) -> list[dict[str, Any]]:
    if not item.available:
        return [_identity(item)]
    assert item.rows is not None
    by_case = {row.get("case_id"): row for row in item.rows}
    output: list[dict[str, Any]] = []
    for case_id in ANCHORS:
        source = by_case.get(case_id)
        if source is None:
            output.append({**_identity(item), "case_id": case_id, "status": "unavailable", "reason": "anchor case absent"})
            continue
        for base in CORE_BASES + CHANNEL_BASES + ENGINEERING_KPI_BASES:
            value = number(source.get(_raw_metric_key(base)))
            if value is None:
                continue
            output.append(
                {
                    **_identity(item),
                    "case_id": case_id,
                    "anchor_reason": ANCHOR_REASONS[case_id],
                    "metric": base,
                    "value": value,
                }
            )
    return output


def _write_phase_outputs(
    comparison: Path,
    slug: str,
    items: Sequence[TableSet],
) -> dict[str, str]:
    distributions = CORE_BASES + CHANNEL_BASES + ENGINEERING_KPI_BASES
    outputs = {
        "headline": comparison / f"{slug}_headline.csv",
        "pooled": comparison / f"{slug}_pooled_metrics.csv",
        "case": comparison / f"{slug}_case_metrics.csv",
        "equal": comparison / f"{slug}_equal_case_distributions.csv",
        "channels": comparison / f"{slug}_channels.csv",
        "physical": comparison / f"{slug}_physical.csv",
        "engineering_kpis": comparison / f"{slug}_engineering_kpis.csv",
        "strata": comparison / f"{slug}_strata.csv",
        "difficult": comparison / f"{slug}_difficult_cases.csv",
    }
    headline: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    equal_rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    difficult_rows: list[dict[str, Any]] = []
    for item in items:
        headline_row = _identity(item)
        if item.available:
            assert item.rows is not None
            global_pool = pooled(item.rows, "global_field_fluid_norm")
            distribution = _distribution(
                number(row.get("global_field_fluid_norm_l2"))
                for row in item.rows
                if number(row.get("global_field_fluid_norm_l2")) is not None
            )
            if global_pool:
                headline_row.update(
                    {
                        "global_field_fluid_norm_pooled_mse": global_pool["mse"],
                        "global_field_fluid_norm_pooled_relative_l2": global_pool["relative_l2"],
                        "global_field_fluid_norm_num_values": global_pool["num_values"],
                    }
                )
            if distribution:
                headline_row.update({f"global_field_fluid_norm_equal_case_{key}": value for key, value in distribution.items()})
        headline.append(headline_row)
        pooled_rows.extend(_phase_rows(item, POOL_BASES))
        case_rows.extend(_case_metric_rows(item, distributions))
        equal_rows.extend(_equal_case_rows(item, distributions))
        strata_rows.extend(_strata_rows(item))
        difficult_rows.extend(_difficult_rows(item))
    _write_csv(outputs["headline"], headline)
    _write_csv(outputs["pooled"], pooled_rows)
    _write_csv(outputs["case"], case_rows)
    _write_csv(outputs["equal"], equal_rows)
    channel_rows = [row for row in pooled_rows if row.get("metric") in CHANNEL_BASES]
    physical_rows = [row for row in pooled_rows if row.get("metric") in CORE_BASES]
    kpi_rows = [row for row in equal_rows if row.get("metric") in ENGINEERING_KPI_BASES]
    unavailable_rows = [_identity(item) for item in items if not item.available]
    _write_csv(outputs["channels"], channel_rows or unavailable_rows)
    _write_csv(outputs["physical"], physical_rows or unavailable_rows)
    _write_csv(outputs["engineering_kpis"], kpi_rows or unavailable_rows)
    _write_csv(outputs["strata"], strata_rows)
    _write_csv(outputs["difficult"], difficult_rows)
    return {name: str(path) for name, path in outputs.items()}


def _pair_requests(endpoint_epoch: int, *, all_baselines: bool | None = None) -> tuple[tuple[str, str, str], ...]:
    """Return the requested candidate/baseline pairs for an endpoint study."""

    endpoint_epoch = _validate_endpoint_epoch(endpoint_epoch)
    if all_baselines is None:
        # Keep the original three requested comparisons in the historical
        # endpoint-500 reducer.  Mature mode also exposes each candidate
        # against every maintained parent for a useful, explicit context.
        all_baselines = endpoint_epoch != DEFAULT_ENDPOINT_EPOCH
    if not all_baselines:
        return PAIR_REQUESTS
    requests: list[tuple[str, str, str]] = []
    for candidate in CANDIDATE_RUNS:
        prefix = "A" if candidate == "1807" else "B"
        for baseline in PARENT_RUNS:
            requests.append((candidate, baseline, f"{prefix}_vs_{baseline}"))
    return tuple(requests)


def _pair_outputs(
    comparison: Path,
    items: Mapping[str, TableSet],
    endpoint_epoch: int = DEFAULT_ENDPOINT_EPOCH,
    *,
    slug: str | None = None,
    all_baselines: bool | None = None,
) -> dict[str, str]:
    endpoint_epoch = _validate_endpoint_epoch(endpoint_epoch)
    slug = _exact_phase(endpoint_epoch) if slug is None else slug
    case_path = comparison / f"{slug}_pairs.csv"
    summary_path = comparison / f"{slug}_pair_summary.csv"
    contribution_path = comparison / f"{slug}_channel_mse_contributions.csv"
    case_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    for candidate, baseline, pair_name in _pair_requests(endpoint_epoch, all_baselines=all_baselines):
        candidate_item = items.get(candidate)
        baseline_item = items.get(baseline)
        identity = {
            "pair": pair_name,
            "candidate": candidate,
            "candidate_model": RUN_LABELS[candidate],
            "baseline": baseline,
            "baseline_model": RUN_LABELS[baseline],
            "candidate_epoch": "" if candidate_item is None else candidate_item.checkpoint_epoch or "",
            "baseline_epoch": "" if baseline_item is None else baseline_item.checkpoint_epoch or "",
        }
        if candidate_item is None or baseline_item is None or not candidate_item.available or not baseline_item.available:
            candidate_status = "missing" if candidate_item is None else f"{candidate_item.status}:{candidate_item.reason}"
            baseline_status = "missing" if baseline_item is None else f"{baseline_item.status}:{baseline_item.reason}"
            summary_rows.append(
                {
                    **identity,
                    "status": "unavailable",
                    "reason": f"candidate={candidate_status}; baseline={baseline_status}",
                }
            )
            continue
        assert candidate_item.rows is not None and baseline_item.rows is not None
        candidate_by_case = {row["case_id"]: row for row in candidate_item.rows}
        baseline_by_case = {row["case_id"]: row for row in baseline_item.rows}
        for base in PAIR_BASES:
            key = _raw_metric_key(base)
            values: list[tuple[str, float, float]] = []
            for case_id in sorted(candidate_by_case):
                candidate_value = number(candidate_by_case[case_id].get(key))
                baseline_value = number(baseline_by_case[case_id].get(key))
                if candidate_value is None or baseline_value is None:
                    continue
                values.append((case_id, candidate_value, baseline_value))
                case_rows.append(
                    {
                        **identity,
                        "status": "available",
                        "metric": base,
                        "case_id": case_id,
                        "candidate_value": candidate_value,
                        "baseline_value": baseline_value,
                        "delta": candidate_value - baseline_value,
                    }
                )
            if not values:
                summary_rows.append({**identity, "status": "metric_unavailable", "metric": base})
                continue
            delta = np.asarray([candidate_value - baseline_value for _, candidate_value, baseline_value in values])
            summary_rows.append(
                {
                    **identity,
                    "status": "available",
                    "metric": base,
                    "n": int(delta.size),
                    "wins": int((delta < 0).sum()),
                    "losses": int((delta > 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "p05_delta": float(np.quantile(delta, 0.05)),
                    "p95_delta": float(np.quantile(delta, 0.95)),
                }
            )
        # Decompose the global normalized-fluid MSE delta by physical channel
        # using the same common scalar-value denominator as the maintained
        # maturity reducer.  Keep this alongside paired rows so a renderer can
        # expose why an aggregate delta moved.
        candidate_pool = pooled(candidate_item.rows, "global_field_fluid_norm")
        baseline_pool = pooled(baseline_item.rows, "global_field_fluid_norm")
        if candidate_pool is not None and baseline_pool is not None:
            count = candidate_pool["num_values"]
            if count == baseline_pool["num_values"] and count > 0:
                for channel in CHANNELS:
                    key = f"field_{channel}_fluid_norm_sse"
                    candidate_values = [number(row.get(key)) for row in candidate_item.rows]
                    baseline_values = [number(row.get(key)) for row in baseline_item.rows]
                    if any(value is None for value in candidate_values + baseline_values):
                        continue
                    candidate_sse = sum(value for value in candidate_values if value is not None)
                    baseline_sse = sum(value for value in baseline_values if value is not None)
                    contribution_rows.append(
                        {
                            "pair": pair_name,
                            "candidate": candidate,
                            "baseline": baseline,
                            "candidate_epoch": candidate_item.checkpoint_epoch or "",
                            "baseline_epoch": baseline_item.checkpoint_epoch or "",
                            "channel": channel,
                            "global_mse_delta_contribution": (candidate_sse - baseline_sse) / count,
                        }
                    )
    _write_csv(case_path, case_rows or [{"status": "unavailable", "reason": "no requested pair has both endpoint table sets"}])
    _write_csv(summary_path, summary_rows)
    _write_csv(
        contribution_path,
        contribution_rows
        or [{"status": "unavailable", "reason": "no requested pair has matched channel SSE columns"}],
    )
    return {
        "pairs": str(case_path),
        "pair_summary": str(summary_path),
        "channel_mse_contributions": str(contribution_path),
    }


def _history_source(project: Path, study: Path, run: str) -> tuple[Path | None, str | None]:
    if run in PARENT_RUNS:
        path = project / STUDY_RELATIVE / PARENT_STUDY_NAME / "history" / "history_trajectory.csv"
        return (path if path.is_file() else None, "maintained five-model history trajectory")
    track = RUN_TRACK[run]
    for candidate in (
        study / track / "history" / "history_trajectory.csv",
        study / track / "history_trajectory.csv",
        study / track / "metrics.csv",
    ):
        if candidate.is_file():
            return candidate, "NStage2 study history"
    run_root = project / RUN_ROOT_RELATIVE
    run_dirs = sorted(path for path in run_root.glob(f"Run_{run}_*") if path.is_dir())
    metric_paths: list[Path] = []
    for directory in run_dirs:
        for candidate in (directory / "metrics.csv", directory / "metrics" / "metrics.csv"):
            if candidate.is_file():
                metric_paths.append(candidate)
                # Run closeout also exposes metrics/metrics.csv as an alias.
                # Select one history per run directory, preferring the writer's path.
                break
    if len(metric_paths) > 1:
        raise ValueError(f"Multiple managed history files found for candidate run {run}: {metric_paths}")
    return (metric_paths[0], "managed NStage2 run history") if metric_paths else (None, None)


def _history_rows(path: Path, run: str) -> list[dict[str, Any]]:
    rows = read_csv(path)
    output: list[dict[str, Any]] = []
    for row in rows:
        row_run = str(row.get("run", ""))
        if row_run and row_run != run:
            continue
        epoch = number(row.get("epoch"))
        if epoch is None:
            continue
        copy = dict(row)
        copy["run"] = run
        copy.setdefault("run_name", "")
        copy["epoch"] = int(epoch)
        output.append(copy)
    return output


def _history_outputs(
    project: Path,
    study: Path,
    comparison: Path,
    endpoint_epoch: int = DEFAULT_ENDPOINT_EPOCH,
) -> dict[str, Any]:
    endpoint_epoch = _validate_endpoint_epoch(endpoint_epoch)
    source_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for run in PARENT_RUNS + CANDIDATE_RUNS:
        path, description = _history_source(project, study, run)
        if path is None:
            source_rows.append(
                {
                    "run": run,
                    "model": RUN_LABELS[run],
                    "status": "unavailable",
                    "reason": "training metrics history is not present yet",
                }
            )
            continue
        rows = _history_rows(path, run)
        if not rows:
            source_rows.append(
                {
                    "run": run,
                    "model": RUN_LABELS[run],
                    "status": "unavailable",
                    "reason": f"history source has no parseable rows: {path}",
                    "source": str(path),
                }
            )
            continue
        bounded = [row for row in rows if 1 <= int(row["epoch"]) <= endpoint_epoch]
        for row in bounded:
            row["model"] = RUN_LABELS[run]
        all_rows.extend(bounded)
        source_rows.append(
            {
                "run": run,
                "model": RUN_LABELS[run],
                "status": "available",
                "source": str(path),
                "description": description,
                f"rows_through_{endpoint_epoch}": len(bounded),
                "last_epoch": max((int(row["epoch"]) for row in bounded), default=None),
            }
        )

    learning_path = comparison / "learning_curves.csv"
    _write_csv(learning_path, all_rows or [{"status": "unavailable", "reason": "no training histories available"}])

    update_prefixes = ("parameter_update_norm", "preclip_gradient_norm")
    update_rows: list[dict[str, Any]] = []
    for row in all_rows:
        for key, value in row.items():
            prefix = next((prefix for prefix in update_prefixes if key == prefix or key.startswith(prefix + "_")), None)
            if prefix is None:
                continue
            converted = number(value)
            if converted is None:
                continue
            component = key[len(prefix) :].lstrip("_") or "total"
            update_rows.append(
                {
                    "run": row["run"],
                    "model": row["model"],
                    "epoch": row["epoch"],
                    "kind": prefix,
                    "component": component,
                    "value": converted,
                }
            )
    updates_path = comparison / "gradient_updates.csv"
    _write_csv(updates_path, update_rows or [{"status": "unavailable", "reason": "gradient/update columns are not present"}])

    training_metrics = (
        "loss_total",
        "field_mse",
        "temperature_mse",
        "val_loss_total",
        "val_field_mse",
        "val_temperature_mse",
    )
    window_rows: list[dict[str, Any]] = []
    for run in PARENT_RUNS + CANDIDATE_RUNS:
        rows = sorted((row for row in all_rows if row["run"] == run), key=lambda row: int(row["epoch"]))
        if not rows:
            continue
        window = rows[-50:]
        for metric in training_metrics:
            values = [number(row.get(metric)) for row in window]
            values = [value for value in values if value is not None]
            if not values:
                continue
            summary = _distribution(values)
            assert summary is not None
            window_rows.append(
                {
                    "run": run,
                    "model": RUN_LABELS[run],
                    "window_first_epoch": int(window[0]["epoch"]),
                    "window_last_epoch": int(window[-1]["epoch"]),
                    "metric": metric,
                    **summary,
                }
            )
        # Include component-level gradients/updates in the same last-window
        # artifact, while retaining the long-form per-epoch table above.
        component_keys = sorted(
            {
                key
                for row in window
                for key in row
                if any(key == prefix or key.startswith(prefix + "_") for prefix in update_prefixes)
            }
        )
        for metric in component_keys:
            values = [number(row.get(metric)) for row in window]
            values = [value for value in values if value is not None]
            if not values:
                continue
            summary = _distribution(values)
            assert summary is not None
            window_rows.append(
                {
                    "run": run,
                    "model": RUN_LABELS[run],
                    "window_first_epoch": int(window[0]["epoch"]),
                    "window_last_epoch": int(window[-1]["epoch"]),
                    "metric": metric,
                    **summary,
                }
            )
    window_path = comparison / "last50_window.csv"
    _write_csv(window_path, window_rows or [{"status": "unavailable", "reason": "no last-50 training window available"}])

    # Keep the endpoint tail available for the existing renderer and add a
    # few named trailing windows for mature convergence inspection.  A window
    # is clipped to the rows that actually exist for a run; no missing epochs
    # are zero-filled or interpolated.
    larger_window_rows: list[dict[str, Any]] = []
    for run in PARENT_RUNS + CANDIDATE_RUNS:
        rows = sorted((row for row in all_rows if row["run"] == run), key=lambda row: int(row["epoch"]))
        if not rows:
            continue
        for requested_size in HISTORY_WINDOW_SIZES:
            window = rows[-requested_size:]
            for metric in training_metrics:
                values = [number(row.get(metric)) for row in window]
                values = [value for value in values if value is not None]
                if not values:
                    continue
                summary = _distribution(values)
                assert summary is not None
                larger_window_rows.append(
                    {
                        "run": run,
                        "model": RUN_LABELS[run],
                        "requested_window_epochs": requested_size,
                        "window_size": len(window),
                        "window_first_epoch": int(window[0]["epoch"]),
                        "window_last_epoch": int(window[-1]["epoch"]),
                        "metric": metric,
                        **summary,
                    }
                )
    larger_windows_path = comparison / "history_windows.csv"
    _write_csv(
        larger_windows_path,
        larger_window_rows
        or [{"status": "unavailable", "reason": "no trailing history windows available"}],
    )

    milestone_rows: list[dict[str, Any]] = []
    milestones = tuple(milestone for milestone in HISTORY_MILESTONES if milestone <= endpoint_epoch)
    for run in PARENT_RUNS + CANDIDATE_RUNS:
        rows_by_epoch = {
            int(row["epoch"]): row for row in all_rows if row["run"] == run
        }
        for milestone in milestones:
            row = rows_by_epoch.get(milestone)
            if row is None:
                milestone_rows.append(
                    {
                        "run": run,
                        "model": RUN_LABELS[run],
                        "milestone_epoch": milestone,
                        "status": "unavailable",
                        "reason": "history has no row at this milestone",
                    }
                )
                continue
            milestone_rows.append(
                {
                    "run": run,
                    "model": RUN_LABELS[run],
                    "milestone_epoch": milestone,
                    "status": "available",
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"run", "model"}
                    },
                }
            )
    milestones_path = comparison / "history_milestones.csv"
    _write_csv(
        milestones_path,
        milestone_rows
        or [{"status": "unavailable", "reason": "no history milestones requested"}],
    )

    history_manifest = comparison / "history_manifest.json"
    _write_json(
        history_manifest,
        {
            "budget_epoch": endpoint_epoch,
            "window": f"last 50 available epochs through exact endpoint {endpoint_epoch}",
            "window_sizes": list(HISTORY_WINDOW_SIZES),
            "milestones": list(milestones),
            "sources": source_rows,
            "outputs": {
                "learning_curves": str(learning_path),
                "last50_window": str(window_path),
                "history_windows": str(larger_windows_path),
                "history_milestones": str(milestones_path),
                "gradient_updates": str(updates_path),
            },
        },
    )
    return {
        "sources": source_rows,
        "outputs": {
            "learning_curves": str(learning_path),
            "last50_window": str(window_path),
            "history_windows": str(larger_windows_path),
            "history_milestones": str(milestones_path),
            "gradient_updates": str(updates_path),
            "history_manifest": str(history_manifest),
        },
    }


def _trajectory_table_sets(
    project: Path,
    study: Path,
    endpoint_epoch: int,
    parent_exact: Mapping[str, TableSet],
    exact: Mapping[str, TableSet],
    reference: TableSet,
) -> list[TableSet]:
    """Load the optional full-grid 500/2500/endpoint trajectory.

    Parent tables are maintained historical sources.  Candidate 500 tables
    may still live beneath the original nstage2 root when the mature study is
    namespaced under ``nstage2/maturity5000``.  Missing midpoint candidate
    tables remain explicitly unavailable so an endpoint result is still
    reducible while a later evaluation can fill the trajectory.
    """

    endpoint_epoch = _validate_endpoint_epoch(endpoint_epoch)
    if endpoint_epoch != MATURE_ENDPOINT_EPOCH:
        return []
    items: list[TableSet] = []
    for epoch in (DEFAULT_ENDPOINT_EPOCH, 2500, MATURE_ENDPOINT_EPOCH):
        phase = _exact_phase(epoch)
        for run in PARENT_RUNS:
            if epoch == endpoint_epoch:
                item = parent_exact[run]
            else:
                item = _load_table_set(
                    project,
                    run,
                    phase,
                    _parent_table_dir(project, run, epoch),
                    required=False,
                )
            _validate_against_reference(item, reference)
            items.append(item)
        for run in CANDIDATE_RUNS:
            if epoch == endpoint_epoch:
                item = exact[run]
            else:
                item = _load_table_set(
                    project,
                    run,
                    phase,
                    _legacy_candidate_table_dir(study, run, epoch),
                    required=False,
                )
            _validate_against_reference(item, reference)
            items.append(item)
    return items


def reduce_nstage2(
    root: str | Path | None = None,
    endpoint_epoch: int = DEFAULT_ENDPOINT_EPOCH,
) -> dict[str, Any]:
    """Write the NStage2 comparison artifacts and return their manifest.

    Parent exact-endpoint tables are required because they are the comparison
    reference.  Missing candidate endpoint tables are represented as an
    ordinary ``unavailable`` status with a reason; no candidate score is
    synthesized.  A missing candidate saved-best table set can reuse the
    exact-endpoint tables only after trusted checkpoint identity proof.

    The default endpoint remains epoch 500 for compatibility with the original
    NStage2 study.  ``endpoint_epoch=5000`` reads the mature candidate tables
    beneath ``maturity5000`` and adds the maintained five-model exact and
    saved-best sources, without changing those historical files.
    """

    endpoint_epoch = _validate_endpoint_epoch(endpoint_epoch)
    project, study = _resolve_roots(root, endpoint_epoch)
    comparison = study / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    exact_phase = _exact_phase(endpoint_epoch)

    parent_exact: dict[str, TableSet] = {
        run: _load_table_set(
            project,
            run,
            exact_phase,
            _parent_table_dir(project, run, endpoint_epoch),
            required=True,
        )
        for run in PARENT_RUNS
    }
    reference = parent_exact["1401"]
    for item in parent_exact.values():
        _validate_against_reference(item, reference)

    exact: dict[str, TableSet] = dict(parent_exact)
    exact.update(
        {
            run: _load_table_set(
                project,
                run,
                exact_phase,
                _candidate_table_dir(study, run, exact_phase, endpoint_epoch),
                required=False,
            )
            for run in CANDIDATE_RUNS
        }
    )
    for run in CANDIDATE_RUNS:
        _validate_against_reference(exact[run], reference)

    best: dict[str, TableSet] = {}
    if endpoint_epoch == MATURE_ENDPOINT_EPOCH:
        parent_best_dir = _parent_best_table_dir(project, endpoint_epoch)
        best.update(
            {
                run: _load_table_set(project, run, "best_field", parent_best_dir, required=True)
                for run in PARENT_RUNS
            }
        )
    for run in CANDIDATE_RUNS:
        best_table_dir = _candidate_table_dir(study, run, "best_field", endpoint_epoch)
        loaded = _load_table_set(project, run, "best_field", best_table_dir, required=False)
        if loaded.available or best_table_dir.is_dir():
            # A maintained saved-best evaluation, when present, is authoritative.
            best[run] = loaded
        else:
            # The endpoint evaluator is the only evaluation allowed to supply
            # these rows.  Reuse it only after proving the saved-best checkpoint
            # is the same endpoint model as the exact endpoint checkpoint.
            best[run] = _reuse_exact_endpoint_for_best(project, run, exact[run], best_table_dir)
    for item in best.values():
        _validate_against_reference(item, reference)

    outputs: dict[str, Any] = {}
    outputs[exact_phase] = _write_phase_outputs(
        comparison,
        exact_phase,
        [exact[run] for run in PARENT_RUNS + CANDIDATE_RUNS],
    )
    outputs["pairs"] = _pair_outputs(comparison, exact, endpoint_epoch)
    best_runs = PARENT_RUNS + CANDIDATE_RUNS if endpoint_epoch == MATURE_ENDPOINT_EPOCH else CANDIDATE_RUNS
    outputs["best_field"] = _write_phase_outputs(
        comparison,
        "best_field",
        [best[run] for run in best_runs],
    )
    if endpoint_epoch == MATURE_ENDPOINT_EPOCH:
        outputs["best_field_pairs"] = _pair_outputs(
            comparison,
            best,
            endpoint_epoch,
            slug="best_field",
            all_baselines=True,
        )
        trajectory = _trajectory_table_sets(project, study, endpoint_epoch, parent_exact, exact, reference)
        outputs["trajectory"] = _write_phase_outputs(comparison, "trajectory", trajectory)
    history = _history_outputs(project, study, comparison, endpoint_epoch)
    outputs["history"] = history["outputs"]

    source_manifest: dict[str, Any] = {}
    for item in tuple(parent_exact.values()) + tuple(exact[run] for run in CANDIDATE_RUNS) + tuple(best.values()):
        source_manifest[f"{item.phase}:{item.run}"] = {
            **item.status_row(),
            "source": _source_manifest(item.table_dir),
        }
    if endpoint_epoch == MATURE_ENDPOINT_EPOCH:
        source_manifest["trajectory"] = [
            {
                **item.status_row(),
                "source": _source_manifest(item.table_dir),
            }
            for item in trajectory
        ]
    _write_json(comparison / "source_manifest.json", source_manifest)

    command = (
        "PYTHONPATH=src:tools/diagnostics "
        "/home/wanglz/miniconda3/envs/ModularDT/bin/python "
        f"tools/diagnostics/nstage2_reduction.py --root {study}"
    )
    if endpoint_epoch != DEFAULT_ENDPOINT_EPOCH:
        command += f" --endpoint-epoch {endpoint_epoch}"
    entrypoint = {
        "module": "tools.diagnostics.nstage2_reduction",
        "callable": (
            "reduce_nstage2(root)"
            if endpoint_epoch == DEFAULT_ENDPOINT_EPOCH
            else f"reduce_nstage2(root, endpoint_epoch={endpoint_epoch})"
        ),
        "command": command,
        "source_of_truth": "tools/diagnostics/analyze_honf_maturity.py::read_csv,pooled,STRATA,CHANNELS",
    }

    candidate_statuses = [item.status for item in exact.values() if item.run in CANDIDATE_RUNS]
    best_statuses = [item.status for item in best.values()]
    unavailable = [item.status_row() for item in tuple(exact.values()) + tuple(best.values()) if not item.available]
    requested_pairs = _pair_requests(endpoint_epoch)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete" if not unavailable else "partial_unavailable",
        "project_root": str(project),
        "study_root": str(study),
        "comparison_root": str(comparison),
        exact_phase: {
            "runs": [item.status_row() for item in exact.values()],
            "candidate_statuses": candidate_statuses,
            "requested_pairs": [
                {"pair": name, "candidate": candidate, "baseline": baseline}
                for candidate, baseline, name in requested_pairs
            ],
        },
        "best_field": {
            "runs": [item.status_row() for item in best.values()],
            "candidate_statuses": best_statuses,
            "selection": (
                "candidate saved-best table with actual checkpoint epoch read from its checkpoint; "
                "when best_field tables are absent, exact500 endpoint tables may be reused only for an equal epoch500 model"
                if endpoint_epoch == DEFAULT_ENDPOINT_EPOCH
                else "saved-best table with actual checkpoint epoch read from its checkpoint; "
                f"when candidate best_field tables are absent, exact{endpoint_epoch} endpoint tables may be reused only "
                f"for an equal epoch{endpoint_epoch} model"
            ),
        },
        "entrypoint": entrypoint,
        "fixed_difficult_cases": [
            {"case_id": case_id, "reason": ANCHOR_REASONS[case_id]} for case_id in ANCHORS
        ],
        "source_manifest": str(comparison / "source_manifest.json"),
        "history": history,
        "outputs": outputs,
        "unavailable": unavailable,
        "definitions": {
            "pooled_relative_l2": "sqrt(sum case SSE / sum case target SSE)",
            "pooled_mse": "sum case SSE / sum scalar values",
            "equal_case_distribution": "unweighted distribution of per-case relative-L2 or relative-error rows",
            "paired_delta": "candidate minus baseline; negative means lower reported error",
            "exact_endpoint": f"epoch {endpoint_epoch} evaluation tables selected by the epoch_{endpoint_epoch:04d}_model.pt checkpoint",
            "best_field": f"saved-best table; when candidate tables are absent, exact{endpoint_epoch} endpoint tables are reused only after "
            f"trusted CPU proof of equal epoch{endpoint_epoch} model_state_dict tensors; actual checkpoint epoch is recorded",
            "engineering_kpi_columns": list(ENGINEERING_KPI_BASES),
            "engineering_kpi_source": "maintained per_case_metrics.csv columns; equal-case distributions preserve the evaluator's signed, absolute, and relative error columns",
            "population": "90-case development holdout; not independent physical-reference validation",
        },
    }
    if endpoint_epoch == MATURE_ENDPOINT_EPOCH:
        payload["parent_best_through_5000"] = {
            "status": "available",
            "table_dir": str(_parent_best_table_dir(project, endpoint_epoch)),
            "selection": "best_by_field_mse_model.pt selected through the mature 5000-epoch budget; actual checkpoint epochs are read by the trusted loader",
        }
        payload["trajectory"] = {
            "epochs": [DEFAULT_ENDPOINT_EPOCH, 2500, MATURE_ENDPOINT_EPOCH],
            "candidate_midpoint_policy": "candidate endpoint500 tables are reused from the original nstage2 root when present; endpoint2500 tables are read from maturity5000; missing midpoint tables remain unavailable",
        }
    else:
        payload["parent_best_through_500"] = {
            "status": "unavailable",
            "reason": "No matched parent best-through-500 checkpoint artifacts; history minima are metrics only and are not selected checkpoints.",
            "history_minimum_epochs": PARENT_HISTORY_MINIMA,
        }
    _write_json(comparison / "comparison.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repository root or NStage2 study root (default: repository study location)",
    )
    parser.add_argument(
        "--endpoint-epoch",
        type=int,
        default=DEFAULT_ENDPOINT_EPOCH,
        help="exact endpoint epoch to reduce (default: 500; mature comparison uses 5000)",
    )
    args = parser.parse_args(argv)
    payload = reduce_nstage2(args.root, endpoint_epoch=args.endpoint_epoch)
    print(json.dumps({"status": payload["status"], "comparison_root": payload["comparison_root"], "unavailable": payload["unavailable"]}, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
