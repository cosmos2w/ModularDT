"""Shared checkpoint selector names and explicit fallback policy."""

from __future__ import annotations

from typing import Any, Mapping


CHECKPOINT_FILES = {
    "best": "best_model.pt",
    "best_total": "best_model.pt",
    "best_field": "best_by_field_mse_model.pt",
    "best_by_field_mse": "best_by_field_mse_model.pt",
    "best_temperature": "best_by_temperature_mse_model.pt",
    "best_by_temperature_mse": "best_by_temperature_mse_model.pt",
    "best_autonomous": "best_predicted_model.pt",
    "best_predicted": "best_predicted_model.pt",
    "latest": "latest_model.pt",
}


def checkpoint_filename(selector: str) -> str:
    """Return a filename for a known selector; reject silent substitutions."""

    key = str(selector).strip().lower()
    try:
        return CHECKPOINT_FILES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown checkpoint selector {selector!r}; choose one of {sorted(CHECKPOINT_FILES)}.") from exc


def validate_checkpoint_identity(
    payload: Mapping[str, Any],
    *,
    case_id: str,
    model_family: str,
    workflow: str,
    local_module_id: str | None = None,
) -> None:
    """Validate versioned identity while retaining unversioned history support."""

    version = payload.get("checkpoint_schema_version")
    if version is not None and int(version) != 1:
        raise ValueError(f"Unsupported checkpoint_schema_version={version!r}; expected 1.")
    expected = {"case_id": case_id, "model_family": model_family, "workflow": workflow}
    if local_module_id is not None:
        expected["local_module_id"] = local_module_id
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) is not None and payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint identity mismatch: {mismatches}")
