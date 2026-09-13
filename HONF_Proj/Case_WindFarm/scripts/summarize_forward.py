"""Summarize two completed WindFarm forward runs without executing models.

The input paths are deliberately explicit.  Metric paths use the form
``MODEL:ENDPOINT=PATH`` and history paths use ``MODEL=PATH``::

    python summarize_forward.py \
      --metric classic:best_selection_val=/path/classic/validation/metrics.json \
      --metric classic:exact500_val=/path/classic/epoch500/metrics.json \
      --metric classic:reserved_test_best=/path/classic/test/metrics.json \
      --metric dense:best_selection_val=/path/dense/validation/metrics.json \
      --metric dense:exact500_val=/path/dense/epoch500/metrics.json \
      --metric dense:reserved_test_best=/path/dense/test/metrics.json \
      --history classic=/path/classic/metrics.csv \
      --history dense=/path/dense/metrics.csv \
      --output-dir /path/to/ignored/comparison

This script reads JSON and CSV only.  It never imports the WindFarm data
reader, constructs a model, or scans a run directory.  Missing endpoint
records remain explicit in the output; no result is inferred.  The report
does not compute confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ENDPOINTS = ("best_selection_val", "exact500_val", "reserved_test_best")
MODELS = ("classic", "dense")
CHANNELS = ("Ux", "Uy", "Uz")
SCOPES = ("volume", "hub_band", "downstream_envelope")
STATISTICS = ("count", "mean", "median", "p95", "worst")
_EXPECTED_SPLITS = {
    "best_selection_val": "validation",
    "exact500_val": "validation",
    "reserved_test_best": "test",
}

_ENDPOINT_ALIASES = {
    "bestselectionval": "best_selection_val",
    "best_selection_val": "best_selection_val",
    "bestselectionvalidation": "best_selection_val",
    "best_selection_validation": "best_selection_val",
    "exact500val": "exact500_val",
    "exact500_val": "exact500_val",
    "exact500_validation": "exact500_val",
    "exact_500_val": "exact500_val",
    "exact500": "exact500_val",
    "reservedtestbest": "reserved_test_best",
    "reserved_test_best": "reserved_test_best",
    "testbest": "reserved_test_best",
    "test_best": "reserved_test_best",
}
_MODEL_ALIASES = {
    "classic": "classic",
    "classic_k6": "classic",
    "dense": "dense",
    "dense_pairwise": "dense",
}


def _canonical_model(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _MODEL_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown model label {value!r}; expected classic or dense.") from exc


def _canonical_endpoint(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    key = re.sub(r"_+", "_", key)
    try:
        return _ENDPOINT_ALIASES[key]
    except KeyError as exc:
        expected = ", ".join(ENDPOINTS)
        raise ValueError(f"Unknown endpoint label {value!r}; expected one of {expected}.") from exc


def _split_assignment(spec: str, *, description: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"{description} must have the form LABEL=PATH: {spec!r}")
    label, path = spec.split("=", 1)
    if not label.strip() or not path.strip():
        raise ValueError(f"{description} must have non-empty label and path: {spec!r}")
    return label.strip(), path.strip()


def _parse_metric_specs(specs: Iterable[str]) -> dict[tuple[str, str], Path]:
    parsed: dict[tuple[str, str], Path] = {}
    for spec in specs:
        label, raw_path = _split_assignment(spec, description="--metric")
        if ":" not in label:
            raise ValueError(f"--metric label must have MODEL:ENDPOINT: {spec!r}")
        raw_model, raw_endpoint = label.split(":", 1)
        key = (_canonical_model(raw_model), _canonical_endpoint(raw_endpoint))
        if key in parsed:
            raise ValueError(f"Duplicate metric input for {key[0]} / {key[1]}.")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Metric JSON does not exist: {path}")
        parsed[key] = path
    return parsed


def _parse_history_specs(specs: Iterable[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        raw_model, raw_path = _split_assignment(spec, description="--history")
        model = _canonical_model(raw_model)
        if model in parsed:
            raise ValueError(f"Duplicate history input for {model}.")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Training history does not exist: {path}")
        parsed[model] = path
    if set(parsed) != set(MODELS):
        raise ValueError("Exactly two histories are required: one classic=PATH and one dense=PATH.")
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metric JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Metric JSON must contain an object at the top level: {path}")
    return payload


def _metric_body(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept the direct evaluate output and one explicit wrapper level."""

    for key in ("result", "metrics"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping) and "cases" in candidate:
            return candidate
    return payload


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numeric_vector(value: Any, *, length: int = 3) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        return None
    result = [_finite_float(item) for item in value]
    return None if any(item is None for item in result) else [float(item) for item in result]


def _summary(values: Iterable[float]) -> dict[str, Any] | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[middle]
    else:
        median = 0.5 * (ordered[middle - 1] + ordered[middle])
    index = 0.95 * (len(ordered) - 1)
    lower = math.floor(index)
    upper = min(lower + 1, len(ordered) - 1)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "p95": p95,
        "worst": ordered[-1],
    }


def _vector_summary(values: Iterable[Sequence[float]]) -> dict[str, Any] | None:
    rows = [list(map(float, value)) for value in values]
    if not rows:
        return None
    columns = list(zip(*rows, strict=True))
    return {
        "channels": list(CHANNELS),
        "statistics": {
            statistic: [(_summary(column) or {}).get(statistic) for column in columns]
            for statistic in STATISTICS
        },
    }


def _case_rows(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = body.get("cases", [])
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("Metric JSON cases must be a list of objects.")
    return [row for row in rows if isinstance(row, Mapping)]


def _case_identities(cases: Sequence[Mapping[str, Any]], path: Path) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for index, row in enumerate(cases):
        source_index = row.get("source_index")
        case_name = row.get("case")
        if source_index is None and case_name is None:
            raise ValueError(f"Metric case {index} has no source_index or case identifier: {path}")
        identity: dict[str, Any] = {}
        if source_index is not None:
            try:
                identity["source_index"] = int(source_index)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Metric case {index} has an invalid source_index: {path}") from exc
        if case_name is not None:
            identity["case"] = str(case_name)
        if row.get("layout_index") is not None:
            identity["layout_index"] = int(row["layout_index"])
        if row.get("wind_direction_deg") is not None:
            identity["wind_direction_deg"] = float(row["wind_direction_deg"])
        identities.append(identity)
    if not identities:
        raise ValueError(f"Metric JSON has no cases to pair: {path}")
    return identities


def _endpoint_metadata(
    body: Mapping[str, Any],
    *,
    endpoint: str,
    cases: Sequence[Mapping[str, Any]],
    path: Path,
) -> dict[str, Any]:
    required = ("split", "checkpoint_epoch", "q_volume", "q_band", "sample_seed")
    missing = [name for name in required if body.get(name) is None]
    if missing:
        raise ValueError(f"Metric JSON lacks endpoint metadata {missing}: {path}")
    split = str(body["split"])
    expected_split = _EXPECTED_SPLITS[endpoint]
    if split != expected_split:
        raise ValueError(
            f"Endpoint {endpoint} requires split={expected_split!r}, got {split!r}: {path}"
        )
    checkpoint_epoch = _finite_float(body["checkpoint_epoch"])
    q_volume = _finite_float(body["q_volume"])
    q_band = _finite_float(body["q_band"])
    sample_seed = _finite_float(body["sample_seed"])
    if checkpoint_epoch is None or q_volume is None or q_band is None or sample_seed is None:
        raise ValueError(f"Metric endpoint metadata must be finite numeric values: {path}")
    if q_volume <= 0 or q_band <= 0 or q_volume != int(q_volume) or q_band != int(q_band):
        raise ValueError(f"Metric endpoint query budgets must be positive integers: {path}")
    if endpoint == "exact500_val" and checkpoint_epoch != 500:
        raise ValueError(
            f"Endpoint exact500_val requires checkpoint_epoch=500, got {int(checkpoint_epoch)}: {path}"
        )
    provenance = body.get("selection_metric_provenance")
    if endpoint in {"best_selection_val", "reserved_test_best"}:
        if not isinstance(provenance, Mapping):
            raise ValueError(f"Endpoint {endpoint} requires selection_metric_provenance: {path}")
        if provenance.get("checkpoint_role") != "best_validation_checkpoint":
            raise ValueError(
                f"Endpoint {endpoint} requires checkpoint_role='best_validation_checkpoint': {path}"
            )
        if provenance.get("validation_only_selection") is not True:
            raise ValueError(f"Endpoint {endpoint} lacks validation-only checkpoint provenance: {path}")
        if provenance.get("reserved_test_used_for_selection") is not False:
            raise ValueError(f"Endpoint {endpoint} has invalid reserved-test selection provenance: {path}")
    return {
        "checkpoint_epoch": int(checkpoint_epoch),
        "q_volume": int(q_volume),
        "q_band": int(q_band),
        "sample_seed": int(sample_seed),
        "split": split,
        "checkpoint_selector": None if not isinstance(provenance, Mapping) else provenance.get("checkpoint_selector"),
        "checkpoint_role": None if not isinstance(provenance, Mapping) else provenance.get("checkpoint_role"),
        "selection_metric_provenance": provenance,
        "case_count": len(cases),
        "case_ids": _case_identities(cases, path),
    }


def _equal_case_scope(cases: Sequence[Mapping[str, Any]], scope: str) -> dict[str, Any]:
    scalar_fields = (
        "standardized_mse",
        "vector_relative_l2",
        "baseline_rmse_mps",
        "baseline_standardized_mse",
        "baseline_ux_residual_rmse_mps",
        "baseline_ux_residual_energy_mps2",
        "baseline_ux_residual_relative_l2",
        "rmse_mps",
        "mae_mps",
    )
    vectors = {
        "channel_rmse_mps": "channel_rmse_mps",
        "channel_mae_mps": "channel_mae_mps",
        "channel_rmse_over_u_ref": "channel_rmse_over_u_ref",
        "channel_mae_over_u_ref": "channel_mae_over_u_ref",
    }
    result: dict[str, Any] = {}
    for field in scalar_fields:
        values = [_finite_float(row.get(f"{scope}_{field}")) for row in cases]
        valid = [float(value) for value in values if value is not None]
        summary = _summary(valid)
        if summary is not None:
            result[field] = summary
    for output_name, field in vectors.items():
        values = [_numeric_vector(row.get(f"{scope}_{field}")) for row in cases]
        valid = [value for value in values if value is not None]
        summary = _vector_summary(valid)
        if summary is not None:
            result[output_name] = summary
    return result


def _pooled_scope(body: Mapping[str, Any], scope: str) -> dict[str, Any] | None:
    pooled = body.get("pooled")
    if not isinstance(pooled, Mapping):
        return None
    source = pooled.get(scope)
    if not isinstance(source, Mapping):
        return None
    result: dict[str, Any] = {}
    for field in (
        "available",
        "count",
        "weight_sum",
        "vector_relative_l2",
        "error_squared_integral",
        "target_energy_integral",
        "channel_rmse_mps",
        "channel_mae_mps",
        "channel_rmse_over_u_ref",
        "channel_mae_over_u_ref",
    ):
        value = source.get(field)
        if field in {"error_squared_integral", "target_energy_integral", "channel_rmse_mps", "channel_mae_mps",
                     "channel_rmse_over_u_ref", "channel_mae_over_u_ref"}:
            vector = _numeric_vector(value)
            if vector is not None:
                result[field] = vector
        elif field == "available":
            if isinstance(value, bool):
                result[field] = value
        elif field == "count":
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                result[field] = int(value)
        else:
            finite = _finite_float(value)
            if finite is not None:
                result[field] = finite
    return result or None


def _layout_counts(body: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    layout_level = body.get("layout_level")
    per_layout = layout_level.get("per_layout", {}) if isinstance(layout_level, Mapping) else {}
    if not isinstance(per_layout, Mapping):
        per_layout = {}
    directions = []
    for value in per_layout.values():
        if isinstance(value, Mapping):
            count = _finite_float(value.get("directions"))
            if count is not None:
                directions.append(int(count))
    layout_count = _finite_float(layout_level.get("layouts")) if isinstance(layout_level, Mapping) else None
    return {
        "case_count": int(_finite_float(body.get("rows")) or len(cases)),
        "layout_count": int(layout_count) if layout_count is not None else len(per_layout),
        "directions_total": sum(directions),
        "directions_per_layout": {
            "min": min(directions) if directions else None,
            "max": max(directions) if directions else None,
            "values": directions,
        },
    }


def _summarize_metric(model: str, endpoint: str, path: Path) -> dict[str, Any]:
    body = _metric_body(_load_json(path))
    cases = _case_rows(body)
    metadata = _endpoint_metadata(body, endpoint=endpoint, cases=cases, path=path)
    return {
        "model": model,
        "endpoint": endpoint,
        "source": str(path),
        "split": body.get("split"),
        "checkpoint": body.get("checkpoint"),
        "endpoint_metadata": metadata,
        "case_ids": metadata["case_ids"],
        "layout_counts": _layout_counts(body, cases),
        "equal_case": {scope: _equal_case_scope(cases, scope) for scope in SCOPES},
        "pooled_energy": {scope: _pooled_scope(body, scope) for scope in SCOPES},
        "downstream_envelope_source": body.get("downstream_envelope_source"),
    }


def _numeric_difference(left: Any, right: Any) -> Any:
    """Return left-minus-right for numeric report values, preserving shape."""

    if isinstance(left, bool) or isinstance(right, bool):
        return None
    left_number = _finite_float(left)
    right_number = _finite_float(right)
    if left_number is not None and right_number is not None:
        return left_number - right_number
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result = {}
        for key in left.keys() & right.keys():
            value = _numeric_difference(left[key], right[key])
            if value is not None:
                result[key] = value
        return result or None
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        if len(left) != len(right):
            return None
        values = [_numeric_difference(a, b) for a, b in zip(left, right, strict=True)]
        return values if all(value is not None for value in values) else None
    return None


def _paired_delta(classic: Mapping[str, Any] | None, dense: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if classic is None or dense is None:
        return None
    result: dict[str, Any] = {"direction": "classic_minus_dense"}
    for key in ("equal_case", "pooled_energy", "layout_counts"):
        value = _numeric_difference(classic.get(key), dense.get(key))
        if value is not None:
            result[key] = value
    return result


def _validate_metric_alignment(endpoint_records: Sequence[Mapping[str, Any]]) -> None:
    """Reject mismatched model inputs before reporting a paired difference."""

    by_endpoint = {str(record["endpoint"]): record for record in endpoint_records}
    for record in endpoint_records:
        classic = record.get("classic")
        dense = record.get("dense")
        if not isinstance(classic, Mapping) or not isinstance(dense, Mapping):
            continue
        classic_meta = classic["endpoint_metadata"]
        dense_meta = dense["endpoint_metadata"]
        for field in ("q_volume", "q_band", "sample_seed", "split"):
            if classic_meta[field] != dense_meta[field]:
                raise ValueError(
                    f"Paired {record['endpoint']} inputs disagree on {field}: "
                    f"classic={classic_meta[field]!r}, dense={dense_meta[field]!r}"
                )
        if classic["case_ids"] != dense["case_ids"]:
            raise ValueError(f"Paired {record['endpoint']} inputs have different ordered case IDs.")

    for model in MODELS:
        present = [record[model] for record in endpoint_records if isinstance(record.get(model), Mapping)]
        if not present:
            continue
        first = present[0]
        first_meta = first["endpoint_metadata"]
        for current in present[1:]:
            current_meta = current["endpoint_metadata"]
            for field in ("q_volume", "q_band", "sample_seed"):
                if current_meta[field] != first_meta[field]:
                    raise ValueError(
                        f"{model} endpoint inputs disagree on {field}: "
                        f"{first['endpoint']}={first_meta[field]!r}, "
                        f"{current['endpoint']}={current_meta[field]!r}"
                    )
        validation_records = [
            record for record in (by_endpoint.get("best_selection_val"), by_endpoint.get("exact500_val"))
            if isinstance(record, Mapping) and isinstance(record.get(model), Mapping)
        ]
        if len(validation_records) == 2:
            first_validation = validation_records[0][model]
            second_validation = validation_records[1][model]
            if first_validation["case_ids"] != second_validation["case_ids"]:
                raise ValueError(f"{model} validation endpoints have different ordered case IDs.")


def _read_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Training history is empty: {path}")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if None in row:
            raise ValueError(f"Training history has more values than headers: {path}")
        epoch = _finite_float(row.get("epoch"))
        if epoch is None:
            raise ValueError(f"Training history has a row without epoch: {path}")
        parsed_row: dict[str, Any] = {"epoch": int(epoch)}
        for key, value in row.items():
            if key == "epoch":
                continue
            parsed_row[key] = _finite_float(value)
        parsed.append(parsed_row)
    return parsed


def _history_summary(model: str, path: Path) -> dict[str, Any]:
    rows = _read_history(path)
    if not any(row.get("loss_total") is not None for row in rows):
        raise ValueError(f"Training history lacks loss_total: {path}")
    return {
        "model": model,
        "source": str(path),
        "rows": len(rows),
        "first_epoch": rows[0]["epoch"],
        "last_epoch": rows[-1]["epoch"],
        "exact500_present": any(row["epoch"] == 500 for row in rows),
        "fields": sorted({key for row in rows for key in row if key != "epoch"}),
        "training_objective": {
            "history_field": "loss_total",
            "definition": "0.75 * train_volume_mse + 0.25 * train_band_mse",
            "values": [
                {"epoch": row["epoch"], "value": row["loss_total"]}
                for row in rows
                if row.get("loss_total") is not None
            ],
            "validation_total_not_plotted": True,
            "validation_sampling_definition": "0.8 * val_volume_mse + 0.2 * val_band_mse",
        },
        "comparable_curve_fields": {
            "volume": ["train_volume_mse", "val_volume_mse"],
            "hub_band": ["train_band_mse", "val_band_mse"],
        },
    }


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = ("model", "endpoint", "scope", "aggregation", "metric", "component", "statistic", "value", "source")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _append_metric_rows(
    output: list[dict[str, Any]],
    *,
    model: str,
    endpoint: str,
    source: str,
    scope: str,
    aggregation: str,
    metric: str,
    value: Any,
) -> None:
    if isinstance(value, Mapping) and "statistics" in value:
        statistics = value.get("statistics")
        channels = value.get("channels", CHANNELS)
        if isinstance(statistics, Mapping) and isinstance(channels, Sequence):
            for statistic, values in statistics.items():
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    for component, scalar in zip(channels, values, strict=False):
                        if _finite_float(scalar) is not None:
                            output.append({
                                "model": model, "endpoint": endpoint, "scope": scope,
                                "aggregation": aggregation, "metric": metric,
                                "component": str(component), "statistic": str(statistic),
                                "value": float(scalar), "source": source,
                            })
        return
    if isinstance(value, Mapping):
        for statistic, scalar in value.items():
            if _finite_float(scalar) is not None:
                output.append({
                    "model": model, "endpoint": endpoint, "scope": scope,
                    "aggregation": aggregation, "metric": metric, "component": "scalar",
                    "statistic": str(statistic), "value": float(scalar), "source": source,
                })
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for component, scalar in zip(CHANNELS, value, strict=False):
            if _finite_float(scalar) is not None:
                output.append({
                    "model": model, "endpoint": endpoint, "scope": scope,
                    "aggregation": aggregation, "metric": metric, "component": component,
                    "statistic": "value", "value": float(scalar), "source": source,
                })
        return
    scalar = _finite_float(value)
    if scalar is not None:
        output.append({
            "model": model, "endpoint": endpoint, "scope": scope,
            "aggregation": aggregation, "metric": metric, "component": "scalar",
            "statistic": "value", "value": scalar, "source": source,
        })


def _comparison_csv_records(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for endpoint in comparison["endpoints"]:
        for model in MODELS:
            record = endpoint[model]
            if record is None:
                continue
            for scope, values in record["equal_case"].items():
                for metric, value in values.items():
                    _append_metric_rows(records, model=model, endpoint=record["endpoint"], source=record["source"],
                                        scope=scope, aggregation="equal_case", metric=metric, value=value)
            for scope, values in record["pooled_energy"].items():
                if values is None:
                    continue
                for metric, value in values.items():
                    _append_metric_rows(records, model=model, endpoint=record["endpoint"], source=record["source"],
                                        scope=scope, aggregation="pooled", metric=metric, value=value)
            for metric, value in record["layout_counts"].items():
                if isinstance(value, Mapping):
                    continue
                if _finite_float(value) is not None:
                    records.append({
                        "model": model, "endpoint": record["endpoint"], "scope": "all",
                        "aggregation": "layout_counts", "metric": metric, "component": "scalar",
                        "statistic": "value", "value": float(value), "source": record["source"],
                    })
        paired = endpoint.get("paired_delta")
        if isinstance(paired, Mapping):
            for scope, values in paired.get("equal_case", {}).items():
                if isinstance(values, Mapping):
                    for metric, value in values.items():
                        _append_metric_rows(
                            records,
                            model="classic_minus_dense",
                            endpoint=endpoint["endpoint"],
                            source="paired endpoint inputs",
                            scope=scope,
                            aggregation="paired_delta",
                            metric=metric,
                            value=value,
                        )
            for scope, values in paired.get("pooled_energy", {}).items():
                if isinstance(values, Mapping):
                    for metric, value in values.items():
                        _append_metric_rows(
                            records,
                            model="classic_minus_dense",
                            endpoint=endpoint["endpoint"],
                            source="paired endpoint inputs",
                            scope=scope,
                            aggregation="paired_delta",
                            metric=metric,
                            value=value,
                        )
    return records


def _plot_histories(histories: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"classic": "#1f77b4", "dense": "#d62728"}
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for model in MODELS:
        rows = histories[model]
        required_fields = ("train_volume_mse", "val_volume_mse", "train_band_mse", "val_band_mse")
        missing = [field for field in required_fields if not any(row.get(field) is not None for row in rows)]
        if missing:
            raise ValueError(f"{model} history lacks comparable learning-curve fields {missing}.")
        epochs = [row["epoch"] for row in rows]
        for axis, train_key, val_key, title in (
            (axes[0], "train_volume_mse", "val_volume_mse", "Volume MSE"),
            (axes[1], "train_band_mse", "val_band_mse", "Hub-band MSE"),
        ):
            train_values = [row.get(train_key) for row in rows]
            val_values = [row.get(val_key) for row in rows]
            train_points = [(epoch, value) for epoch, value in zip(epochs, train_values, strict=True) if value is not None]
            val_points = [(epoch, value) for epoch, value in zip(epochs, val_values, strict=True) if value is not None]
            if train_points:
                axis.plot(*zip(*train_points, strict=True), color=colors[model], label=f"{model} train")
            if val_points:
                axis.plot(*zip(*val_points, strict=True), color=colors[model], linestyle="--", label=f"{model} validation")
            axis.set_title(title)
            axis.set_ylabel("MSE")
            axis.grid(alpha=0.25)
    axes[1].set_xlabel("Epoch")
    axes[0].legend(loc="best", fontsize="small", ncol=2)
    for axis in axes:
        if any(row["epoch"] == 500 for rows in histories.values() for row in rows):
            axis.axvline(500, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def build_comparison(metric_paths: Mapping[tuple[str, str], Path], history_paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    endpoint_records: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        classic = (
            _summarize_metric("classic", endpoint, metric_paths[("classic", endpoint)])
            if ("classic", endpoint) in metric_paths else None
        )
        dense = (
            _summarize_metric("dense", endpoint, metric_paths[("dense", endpoint)])
            if ("dense", endpoint) in metric_paths else None
        )
        endpoint_records.append({
            "endpoint": endpoint,
            "classic": classic,
            "dense": dense,
            "paired_delta": _paired_delta(classic, dense),
        })
    _validate_metric_alignment(endpoint_records)
    history_rows = {model: _read_history(path) for model, path in history_paths.items()}
    comparison = {
        "schema_version": 1,
        "models": list(MODELS),
        "endpoints": endpoint_records,
        "histories": {model: _history_summary(model, path) for model, path in history_paths.items()},
        "claims": {
            "confidence_intervals": False,
            "statement": "This paired summary reports direct run metrics and learning curves; it makes no confidence-interval or population-generalization claim.",
        },
    }
    return comparison, history_rows


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", action="append", default=[], metavar="MODEL:ENDPOINT=PATH")
    parser.add_argument("--history", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--require-all-endpoints",
        action="store_true",
        help="Fail unless both models provide best-selection validation, exact-500 validation, and reserved-test-best JSON.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    metric_paths = _parse_metric_specs(args.metric)
    history_paths = _parse_history_specs(args.history)
    missing = [(model, endpoint) for model in MODELS for endpoint in ENDPOINTS if (model, endpoint) not in metric_paths]
    if args.require_all_endpoints and missing:
        missing_text = ", ".join(f"{model}:{endpoint}" for model, endpoint in missing)
        raise ValueError(f"Missing required endpoint metric inputs: {missing_text}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison, history_rows = build_comparison(metric_paths, history_paths)
    comparison["missing_endpoints"] = [f"{model}:{endpoint}" for model, endpoint in missing]
    (output_dir / "paired_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "paired_comparison.csv", _comparison_csv_records(comparison))
    _plot_histories(history_rows, output_dir / "learning_curves.png")
    print(f"[windfarm-summary] output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
