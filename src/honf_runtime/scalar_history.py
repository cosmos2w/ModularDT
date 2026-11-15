"""Append-only scalar tables for new run producers.

Training uses a provider-declared wide table spanning its scheduled phases.
Values and identifiers are returned as strings; callers
choose numerical conversions without losing leading zeroes in identifiers.
Recovery is exclusively the responsibility of the locked training writer.
"""

from __future__ import annotations

import csv
import io
import os
import time
from pathlib import Path
from typing import Iterable, Mapping

VALIDATION_COLUMNS = ("epoch", "partition", "case_id", "member_id", "metric", "value", "count")


def _complete_bytes(path: Path) -> tuple[bytes, bytes]:
    if not path.exists():
        return b"", b""
    raw = path.read_bytes()
    end = raw.rfind(b"\n") + 1
    return raw[:end], raw[end:]


def read_scalar_rows(path: str | Path) -> list[dict[str, str]]:
    """Read complete rows without changing a possibly active writer's file."""
    complete, _ = _complete_bytes(Path(path))
    if not complete:
        return []
    return list(csv.DictReader(io.StringIO(complete.decode("utf-8"), newline="")))


_TRAIN_GROUPS = ("train_losses", "learning_rates", "work_counts", "train_diagnostics")
_INTEGER_FIELDS = {"epoch", "case_visits", "optimizer_updates", "microbatches", "partial_update_cases"}


def column_differences(saved: Iterable[str], incoming: Iterable[str]) -> str:
    """Describe membership and position changes without conflating them."""
    saved, incoming = list(saved), list(incoming)
    return (f"added={sorted(set(incoming) - set(saved))}; "
            f"missing={sorted(set(saved) - set(incoming))}; "
            f"reordered={[(name, saved.index(name), incoming.index(name)) for name in saved
                          if name in incoming and saved.index(name) != incoming.index(name)]}")


def training_recording_definitions(
    columns: Iterable[str], terms: Mapping[str, Mapping[str, object]],
    overrides: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, object]:
    """Describe a provider's explicit column sequence, never configuration key order."""
    columns = list(columns)
    if (not columns or columns[0] != "epoch" or len(columns) != len(set(columns))
            or any(not isinstance(name, str) or not name for name in columns)):
        raise ValueError("Training columns must be unique nonempty strings starting with epoch")
    base = {
        "epoch": ("int", "epochs", "Completed training epoch"),
        "phase": ("str", "1", "Training schedule phase"),
        "arm": ("str", "1", "Executed prediction path"),
        "temperature": ("float", "1", "Routing temperature at this epoch"),
        "case_visits": ("int", "cases", "Training case visits in this epoch"),
        "optimizer_updates": ("int", "updates", "Optimizer updates in this epoch"),
        "microbatches": ("int", "microbatches", "Training microbatches in this epoch"),
        "partial_update_cases": ("int", "cases", "Cases in the final incomplete optimizer update"),
        "train_seconds": ("float", "s", "Training duration for this epoch, excluding validation"),
        "train_loss_total": ("float", "weighted objective units", "Sum of recorded weighted TRAIN components in training_total_order"),
    }
    definitions = {}
    for column in columns:
        if column in base:
            dtype, unit, meaning = base[column]
        elif column.startswith("train_losses."):
            name = column.removeprefix("train_losses.")
            metadata = terms.get(name, {})
            dtype, unit, meaning = ("float", metadata.get("unit", "provider-defined"),
                metadata.get("meaning", metadata.get("formula", f"Provider-defined weighted loss {name}")))
        elif column.startswith("learning_rates."):
            dtype, unit, meaning = "float", "optimizer learning-rate units", f"Applied learning rate for {column.split('.', 1)[1]}"
        elif column == "work_counts.gradient_norm":
            dtype, unit, meaning = "float", "gradient units", "Sum of pre-clipping parameter gradient L2 norms over optimizer updates"
        else:
            dtype, unit, meaning = "float", "provider-defined", f"Provider scalar {column}; epoch reduction follows the provider and trainer"
        definitions[column] = {"dtype": dtype, "unit": unit, "meaning": meaning}
    for column, definition in (overrides or {}).items():
        if column not in definitions:
            raise ValueError(f"Column definition has no declared training column: {column!r}")
        if set(definition) != {"dtype", "unit", "meaning"} or any(
                not isinstance(value, str) or not value for value in definition.values()):
            raise ValueError(f"Column {column!r} requires explicit dtype, unit and meaning strings")
        definitions[column] = dict(definition)
    validation_specs = (
        ("int", "epochs", "Completed training epoch"),
        ("str", "1", "Dataset partition"),
        ("str", "1", "Original case identifier, blank for pooled measurements"),
        ("str", "1", "Original member identifier, blank for pooled measurements"),
        ("str", "1", "Provider metric or validation objective name"),
        ("float", "metric-dependent", "Measured value in the named metric's units"),
        ("float", "metric-dependent count", "Provider count, blank when not supplied"),
    )
    loss_order = [name.removeprefix("train_losses.") for name in columns if name.startswith("train_losses.")]
    return {
        "training_columns": columns,
        "training_column_definitions": definitions,
        "training_loss_order": loss_order,
        "training_total_order": loss_order,
        "validation_columns": list(VALIDATION_COLUMNS),
        "validation_column_definitions": {name: dict(zip(("dtype", "unit", "meaning"), spec))
                                          for name, spec in zip(VALIDATION_COLUMNS, validation_specs)},
        "training_reduction": "Mean of optimizer-update weighted contributions",
        "validation_reduction": "Provider-defined pooled validation measurements",
        "validation_total_order": "Lexicographic order of matched TRAIN and validation loss names",
        "terms": dict(terms),
    }


def resolve_training_recording(saved: Mapping[str, object], incoming: Mapping[str, object]) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """Validate content and reuse saved order; legacy definitions stay untouched."""
    columns, requested = tuple(saved["training_columns"]), tuple(incoming["training_columns"])
    differences = column_differences(columns, requested)
    changed = []
    if len(columns) != len(set(columns)) or not columns or columns[0] != "epoch":
        changed.append("saved training_columns are not unique or do not start with epoch")
    # Lists carry order; mappings carry meaning. Incoming display order is immaterial.
    for field in ("training_column_definitions", "validation_column_definitions",
                  "training_reduction", "validation_reduction", "validation_total_order", "validation_columns"):
        if field in saved and saved[field] != incoming.get(field):
            if isinstance(saved[field], Mapping) and isinstance(incoming.get(field), Mapping):
                before, after = saved[field], incoming[field]
                for name in sorted(set(before) | set(after)):
                    if before.get(name) != after.get(name):
                        changed.append(f"{field}.{name}: saved={before.get(name)!r}, incoming={after.get(name)!r}")
            else:
                changed.append(f"{field}: saved={saved[field]!r}, incoming={incoming.get(field)!r}")
    before_terms, after_terms = saved.get("terms", {}), incoming.get("terms", {})
    semantic_keys = ("unit", "meaning", "formula", "weight", "reduction")
    for name in sorted(set(before_terms) | set(after_terms)):
        before = {key: value for key, value in before_terms.get(name, {}).items() if key in semantic_keys}
        after = {key: value for key, value in after_terms.get(name, {}).items() if key in semantic_keys}
        if before != after:
            changed.append(f"terms.{name}: saved={before!r}, incoming={after!r}")
    loss_names = tuple(name.removeprefix("train_losses.") for name in columns if name.startswith("train_losses."))
    # Older providers were not required to declare columns in reduction order.
    # Preserve their computational path rather than inventing a historical order.
    order = tuple(saved["training_loss_order"]) if "training_loss_order" in saved else None
    if order is not None and (len(order) != len(set(order)) or set(order) != set(loss_names)):
        changed.append(f"saved training_loss_order={order!r} does not enumerate loss columns={loss_names!r}")
    total_order = tuple(saved.get("training_total_order", sorted(loss_names)))
    if len(total_order) != len(set(total_order)) or set(total_order) != set(loss_names):
        changed.append(f"saved training_total_order={total_order!r} does not enumerate loss columns={loss_names!r}")
    if set(columns) != set(requested) or changed:
        raise ValueError(f"Incompatible training recording: {differences}; "
                         f"numerical meaning changed={bool(changed)}; " + "; ".join(changed))
    return columns, order


def validate_scalar_header(path: str | Path, columns: Iterable[str]) -> None:
    """Check existing append positions before fitting, without changing the table."""
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return
    with path.open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream), [])
    if header != list(columns):
        raise ValueError("Scalar table header differs from saved recording: "
                         + column_differences(columns, header)
                         + "; saved definitions unchanged; positional append would mislabel values")


def training_scalar_row(row: Mapping[str, object], columns: tuple[str, ...]) -> dict[str, object]:
    """Extract declared per-epoch scalars, failing on undeclared loss terms."""
    result: dict[str, object] = {name: row.get(name) for name in columns if "." not in name}
    for group in _TRAIN_GROUPS:
        values = row.get(group, {})
        if not isinstance(values, Mapping):
            raise TypeError(f"{group} must be a scalar mapping")
        if group in {"train_losses", "learning_rates"}:
            missing = {f"{group}.{name}" for name in values} - set(columns)
            if missing:
                raise ValueError(f"Undeclared {group} columns: {sorted(missing)}")
        for name, value in values.items():
            column = f"{group}.{name}"
            if column in columns:
                result[column] = value
    return result


def validation_scalar_rows(row: Mapping[str, object]) -> list[dict[str, object]]:
    """Flatten measured validation scalars, retaining case/member identifiers."""
    result: list[dict[str, object]] = []
    epoch = int(row["epoch"])

    def emit(value: object, metric: str, case_id: str = "", member_id: str = "", count: object = "") -> None:
        if isinstance(value, Mapping):
            count = value.get("count", count)
            for key, item in value.items():
                if key not in {"case_id", "member_id", "count", "loss_objective"}:
                    emit(item, f"{metric}.{key}" if metric else str(key), case_id, member_id, count)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping):
                    emit(item, metric, str(item.get("case_id", case_id)), str(item.get("member_id", member_id)), count)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result.append({"epoch": epoch, "partition": "VALIDATION", "case_id": case_id,
                           "member_id": member_id, "metric": metric, "value": value, "count": count})

    emit(row.get("validation", {}), "validation")
    emit(row.get("validation_objective", {}), "validation_objective")
    return result


def read_csv_training_history(path: str | Path) -> list[dict[str, object]]:
    """Reconstruct the scalar training and validation view used by the trainer."""
    path = Path(path)
    result: list[dict[str, object]] = []
    by_epoch: dict[int, dict[str, object]] = {}
    for record in read_scalar_rows(path):
        row: dict[str, object] = {}
        for name, value in record.items():
            if value == "":
                continue
            if name in ("phase", "arm"):
                row[name] = value
            elif "." in name and name.split(".", 1)[0] in _TRAIN_GROUPS:
                group, key = name.split(".", 1)
                row.setdefault(group, {})[key] = float(value)
            else:
                row[name] = int(value) if name in _INTEGER_FIELDS else float(value)
        epoch = int(row["epoch"])
        if epoch in by_epoch:
            raise ValueError(f"Duplicate scalar history epoch {epoch}")
        by_epoch[epoch] = row
        result.append(row)
    for record in read_scalar_rows(path.with_name("validation.csv")):
        epoch = int(record["epoch"])
        if epoch not in by_epoch:
            raise ValueError(f"Validation epoch {epoch} has no training record")
        # Case/member measurements remain available in the CSV; plotted and
        # selection values use only the aggregate records.
        if record["case_id"] or record["member_id"]:
            continue
        parts = record["metric"].split(".")
        target = by_epoch[epoch]
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = float(record["value"])
    return result


def append_scalar_rows(path: str | Path, columns: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> None:
    """Append scalar records and flush them before checkpoint publication."""
    path = Path(path)
    records = list(rows)
    if not records:
        return
    if not columns or columns[0] != "epoch" or len(columns) != len(set(columns)):
        raise ValueError("Scalar table columns must start with epoch and be unique")
    for row in records:
        if set(row) - set(columns):
            raise ValueError("Scalar record contains undeclared columns")
        if any(isinstance(value, (dict, list, tuple, set)) for value in row.values()):
            raise TypeError("Scalar table cells cannot contain arrays or objects")
        if any("\n" in str(value) or "\r" in str(value) for value in row.values() if value is not None):
            raise ValueError("Scalar table cells cannot contain line breaks")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        complete, trailing = _complete_bytes(path)
        if trailing:
            raise ValueError("Torn scalar table requires locked recovery before append")
        validate_scalar_header(path, columns)
    else:
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(columns)
            stream.flush()
            os.fsync(stream.fileno())
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        for row in records:
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def recover_scalar_rows(path: str | Path, completed_epoch: int, *, writer_lock_held: bool) -> Path | None:
    """Reconcile one table to the checkpoint boundary, preserving discarded bytes.

    The caller must own the existing exclusive run lock. A distinct evidence
    file is created only when bytes were actually removed; repeating recovery
    after interruption is therefore harmless.
    """
    if not writer_lock_held:
        raise RuntimeError("Scalar recovery requires the exclusive training lock")
    path = Path(path)
    if not path.exists():
        return None
    raw = path.read_bytes()
    complete, trailing = _complete_bytes(path)
    lines = complete.splitlines(keepends=True)
    if not lines:
        kept, discarded = b"", raw
    else:
        header = next(csv.reader([lines[0].decode("utf-8")]))
        if not header or header[0] != "epoch" or len(header) != len(set(header)):
            raise ValueError("Unsupported scalar table header")
        kept_parts = [lines[0]]
        discarded_parts = []
        for line in lines[1:]:
            fields = next(csv.reader([line.decode("utf-8")]))
            if len(fields) != len(header):
                raise ValueError("Malformed complete scalar record")
            (kept_parts if int(fields[0]) <= completed_epoch else discarded_parts).append(line)
        kept, discarded = b"".join(kept_parts), b"".join(discarded_parts) + trailing
    if not discarded:
        return None
    # If interruption occurred after writing evidence but before replacing the
    # table, reuse that evidence rather than creating a second copy.
    evidence = next((candidate for candidate in path.parent.glob(f"{path.stem}.recovery_*{path.suffix}")
                     if candidate.is_file() and candidate.read_bytes() == discarded), None)
    if evidence is None:
        evidence = path.with_name(f"{path.stem}.recovery_{time.time_ns()}{path.suffix}")
        with evidence.open("xb") as stream:
            stream.write(discarded)
            stream.flush()
            os.fsync(stream.fileno())
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(kept)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return evidence
