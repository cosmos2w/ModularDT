"""Build report-ready Run 2102/2103 WindFarm analysis figures.

The command consumes explicit endpoint metrics, history, and completed
comparison artifact paths.  It performs the case/sampling/checkpoint audit
before constructing any paired comparison, then writes compact CSV/JSON
evidence beside static Matplotlib figures.

All accuracy fields in this module are for the volume field.  Equal-case
physical RMSE is in m/s, equal-case standardized MSE is the normalized
training-space quantity, and pooled vector relative L2 is a separate pooled
field score.  They are deliberately kept as separate rows and panels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ORDER = ("classic", "dense")
ENDPOINT_ORDER = ("best_validation", "epoch2500_validation", "best_test")
ENDPOINT_SPLIT = {
    "best_validation": "validation",
    "epoch2500_validation": "validation",
    "best_test": "test",
}
EXPECTED_Q_VOLUME = 32768
EXPECTED_Q_BAND = 8192
EXPECTED_SAMPLE_SEED = 42
EXPECTED_EPOCH = 2500
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42

MODEL_COLORS = {"classic": "#0072B2", "dense": "#D55E00"}
ENDPOINT_COLORS = {
    "best_validation": "#009E73",
    "epoch2500_validation": "#CC79A7",
    "best_test": "#E69F00",
}
ENDPOINT_LABELS = {
    "best_validation": "Best validation",
    "epoch2500_validation": "Epoch 2,500 validation",
    "best_test": "Best checkpoint test",
}


@dataclass(frozen=True)
class MetricArtifact:
    """One model/checkpoint/split metrics artifact."""

    model: str
    endpoint: str
    path: Path
    payload: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def parse_assignment(raw: str, *, separator: str = "=") -> tuple[str, Path]:
    """Parse a ``label=path`` CLI value while preserving path ``=`` characters."""

    if separator not in raw:
        raise ValueError(f"expected LABEL{separator}PATH, got {raw!r}")
    label, path = raw.split(separator, 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"expected non-empty label and path, got {raw!r}")
    return label, Path(path)


def parse_metric_assignments(raw_paths: Iterable[str]) -> dict[tuple[str, str], Path]:
    """Parse repeated ``MODEL:ENDPOINT=PATH`` values."""

    parsed: dict[tuple[str, str], Path] = {}
    for raw in raw_paths:
        label, path = parse_assignment(raw)
        if ":" not in label:
            raise ValueError(f"metric label must be MODEL:ENDPOINT, got {label!r}")
        model, endpoint = (part.strip() for part in label.split(":", 1))
        key = (model, endpoint)
        if key in parsed:
            raise ValueError(f"duplicate metrics artifact for {model}:{endpoint}")
        parsed[key] = path
    expected = {(model, endpoint) for model in MODEL_ORDER for endpoint in ENDPOINT_ORDER}
    missing = sorted(expected - set(parsed))
    extra = sorted(set(parsed) - expected)
    if missing or extra:
        message = []
        if missing:
            message.append(f"missing {missing}")
        if extra:
            message.append(f"unexpected {extra}")
        raise ValueError("metrics artifacts do not cover the six expected endpoints: " + "; ".join(message))
    return parsed


def load_metric_artifacts(assignments: Mapping[tuple[str, str], Path]) -> dict[tuple[str, str], MetricArtifact]:
    """Load all six metrics files without changing their payloads."""

    return {
        key: MetricArtifact(key[0], key[1], Path(path), _read_json(Path(path)))
        for key, path in assignments.items()
    }


def _case_identity(row: Mapping[str, Any]) -> tuple[str, int, int, float]:
    """Return the identity used for cross-model and cross-endpoint alignment."""

    return (
        str(row.get("case", "")),
        int(row["source_index"]),
        int(row["layout_index"]),
        float(row["wind_direction_deg"]),
    )


def _metric_case_identity(row: Mapping[str, Any]) -> tuple[str, int, int, float]:
    required = {"case", "source_index", "layout_index", "wind_direction_deg"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"metrics case row is missing identity fields {missing}")
    return _case_identity(row)


def _assert_same_identity(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], label: str) -> None:
    left_ids = [_metric_case_identity(row) for row in left]
    right_ids = [_metric_case_identity(row) for row in right]
    if len(left_ids) != len(set(left_ids)) or len(right_ids) != len(set(right_ids)):
        raise ValueError(f"{label} contains duplicate case identities")
    if left_ids != right_ids:
        if set(left_ids) == set(right_ids):
            raise ValueError(f"{label} contains the same cases in a different order")
        missing = sorted(set(left_ids) - set(right_ids))[:3]
        extra = sorted(set(right_ids) - set(left_ids))[:3]
        raise ValueError(f"{label} case identities differ; missing={missing}, extra={extra}")


def _validate_checkpoint_role(endpoint: str, payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    provenance = payload.get("selection_metric_provenance")
    if not isinstance(provenance, dict):
        raise TypeError(f"{path} has no selection_metric_provenance")
    role = str(provenance.get("checkpoint_role", ""))
    validation_only = bool(provenance.get("validation_only_selection", False))
    reserved_test = bool(provenance.get("reserved_test_used_for_selection", True))
    selection_split = provenance.get("selection_split")
    checkpoint_epoch = int(payload.get("checkpoint_epoch", -1))
    if endpoint in {"best_validation", "best_test"}:
        expected = {
            "checkpoint_role": "best_validation_checkpoint",
            "validation_only_selection": True,
            "reserved_test_used_for_selection": False,
            "selection_split": "validation",
        }
        actual = {
            "checkpoint_role": role,
            "validation_only_selection": validation_only,
            "reserved_test_used_for_selection": reserved_test,
            "selection_split": selection_split,
        }
        if actual != expected:
            raise ValueError(f"{path} is not a validation-only selected checkpoint: {actual}")
    elif endpoint == "epoch2500_validation":
        if role != "explicit_checkpoint" or checkpoint_epoch != EXPECTED_EPOCH:
            raise ValueError(f"{path} is not the explicit epoch-2500 checkpoint: role={role!r}, epoch={checkpoint_epoch}")
        if validation_only or reserved_test or selection_split is not None:
            raise ValueError(f"{path} incorrectly records terminal epoch-2500 selection metadata: {provenance}")
    return {
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_role": role,
        "validation_only_selection": validation_only,
        "selection_split": selection_split,
        "reserved_test_used_for_selection": reserved_test,
    }


def _validate_alignment_artifact(
    path: Path,
    artifacts: Mapping[tuple[str, str], MetricArtifact],
    *,
    expected_q_volume: int,
    expected_q_band: int,
    expected_seed: int,
) -> dict[str, Any]:
    payload = _read_json(path)
    contracts = payload.get("endpoint_contracts")
    if not isinstance(contracts, dict):
        raise TypeError(f"{path} has no endpoint_contracts object")
    for model in MODEL_ORDER:
        for endpoint in ENDPOINT_ORDER:
            key = f"{model}:{endpoint}"
            contract = contracts.get(key)
            if not isinstance(contract, dict):
                raise TypeError(f"{path} is missing endpoint contract {key}")
            expected = {
                "q_band": expected_q_band,
                "q_volume": expected_q_volume,
                "sample_seed": expected_seed,
                "split": ENDPOINT_SPLIT[endpoint],
            }
            for field, value in expected.items():
                if contract.get(field) != value:
                    raise ValueError(f"{path} contract {key} has {field}={contract.get(field)!r}, expected {value!r}")
            if int(contract.get("rows", -1)) != len(artifacts[(model, endpoint)].payload.get("cases", [])):
                raise ValueError(f"{path} contract {key} row count disagrees with metrics artifact")
    expected = payload.get("expected")
    if isinstance(expected, dict):
        for field, value in (("q_volume", expected_q_volume), ("q_band", expected_q_band), ("sample_seed", expected_seed)):
            if expected.get(field) != value:
                raise ValueError(f"{path} expected contract has {field}={expected.get(field)!r}, expected {value!r}")
    identity = payload.get("case_identity_match")
    if isinstance(identity, dict):
        for endpoint in ENDPOINT_ORDER:
            row = identity.get(endpoint)
            if isinstance(row, dict) and not bool(row.get("classic_dense_equal")):
                raise ValueError(f"{path} reports non-matching classic/dense identities for {endpoint}")
    return payload


def _validate_endpoint_summary(path: Path, artifacts: Mapping[tuple[str, str], MetricArtifact]) -> dict[str, Any]:
    payload = _read_json(path)
    summaries = payload.get("endpoint_summaries")
    if not isinstance(summaries, list):
        raise TypeError(f"{path} has no endpoint_summaries list")
    found = {(str(row.get("model")), str(row.get("endpoint"))) for row in summaries if isinstance(row, dict)}
    expected = set(artifacts)
    if found != expected:
        raise ValueError(f"{path} endpoint summary keys differ from supplied metrics: missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    return payload


def validate_endpoint_artifacts(
    artifacts: Mapping[tuple[str, str], MetricArtifact],
    *,
    alignment_path: Path | None = None,
    endpoint_summary_path: Path | None = None,
    expected_q_volume: int = EXPECTED_Q_VOLUME,
    expected_q_band: int = EXPECTED_Q_BAND,
    expected_seed: int = EXPECTED_SAMPLE_SEED,
) -> dict[str, Any]:
    """Validate endpoint contracts before making paired claims.

    The returned audit is also written to evidence by :func:`generate_report`.
    """

    expected_keys = {(model, endpoint) for model in MODEL_ORDER for endpoint in ENDPOINT_ORDER}
    if set(artifacts) != expected_keys:
        raise ValueError(f"expected six endpoint artifacts, got {sorted(artifacts)}")
    metadata: dict[str, Any] = {}
    for key in sorted(artifacts):
        artifact = artifacts[key]
        payload = artifact.payload
        if payload.get("workflow") != "forward":
            raise ValueError(f"{artifact.path} does not describe a forward workflow")
        if payload.get("split") != ENDPOINT_SPLIT[artifact.endpoint]:
            raise ValueError(f"{artifact.path} split={payload.get('split')!r}, expected {ENDPOINT_SPLIT[artifact.endpoint]!r}")
        for field, expected in (("q_volume", expected_q_volume), ("q_band", expected_q_band), ("sample_seed", expected_seed)):
            if payload.get(field) != expected:
                raise ValueError(f"{artifact.path} has {field}={payload.get(field)!r}, expected {expected!r}")
        cases = payload.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"{artifact.path} has no cases list")
        if int(payload.get("rows", -1)) != len(cases):
            raise ValueError(f"{artifact.path} rows disagrees with cases length")
        metadata[f"{artifact.model}:{artifact.endpoint}"] = _validate_checkpoint_role(artifact.endpoint, payload, artifact.path)
        identities = [_metric_case_identity(row) for row in cases]
        if len(identities) != len(set(identities)):
            raise ValueError(f"{artifact.path} contains duplicate case identities")
        if len({identity[2] for identity in identities}) != 30:
            raise ValueError(f"{artifact.path} should contain 30 held-out layouts, found {len({identity[2] for identity in identities})}")
        layout_counts = Counter(identity[2] for identity in identities)
        if set(layout_counts.values()) != {3}:
            raise ValueError(f"{artifact.path} does not contain exactly three directions per layout: {layout_counts}")

    for endpoint in ENDPOINT_ORDER:
        _assert_same_identity(
            artifacts[("classic", endpoint)].payload["cases"],
            artifacts[("dense", endpoint)].payload["cases"],
            f"{endpoint} classic/dense pair",
        )
    for model in MODEL_ORDER:
        _assert_same_identity(
            artifacts[(model, "best_validation")].payload["cases"],
            artifacts[(model, "epoch2500_validation")].payload["cases"],
            f"{model} validation endpoints",
        )
    alignment_payload = None
    if alignment_path is not None:
        alignment_payload = _validate_alignment_artifact(
            alignment_path,
            artifacts,
            expected_q_volume=expected_q_volume,
            expected_q_band=expected_q_band,
            expected_seed=expected_seed,
        )
    summary_payload = _validate_endpoint_summary(endpoint_summary_path, artifacts) if endpoint_summary_path else None
    return {
        "paired_claims_allowed": True,
        "identity_basis": ["case", "source_index", "layout_index", "wind_direction_deg"],
        "expected_q_volume": expected_q_volume,
        "expected_q_band": expected_q_band,
        "expected_sample_seed": expected_seed,
        "endpoint_splits": dict(ENDPOINT_SPLIT),
        "endpoint_metadata": metadata,
        "rows_by_endpoint": {
            endpoint: len(artifacts[("classic", endpoint)].payload["cases"]) for endpoint in ENDPOINT_ORDER
        },
        "layouts_by_endpoint": {
            endpoint: len({int(row["layout_index"]) for row in artifacts[("classic", endpoint)].payload["cases"]})
            for endpoint in ENDPOINT_ORDER
        },
        "alignment_artifact_checked": alignment_path is not None,
        "endpoint_summary_checked": endpoint_summary_path is not None,
        "alignment_artifact": str(alignment_path) if alignment_path else None,
        "endpoint_summary_artifact": str(endpoint_summary_path) if endpoint_summary_path else None,
        "alignment_payload": alignment_payload,
        "endpoint_summary_payload": summary_payload,
    }


def _load_history(path: Path, *, max_epoch: int) -> list[dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            try:
                epoch = int(float(raw["epoch"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path} contains a row without an integer epoch") from exc
            if epoch > max_epoch:
                continue
            row: dict[str, float] = {"epoch": float(epoch)}
            for field in ("loss_total", "train_volume_mse", "val_volume_mse", "train_band_mse", "val_band_mse"):
                value = raw.get(field, "")
                if value in ("", None):
                    row[field] = float("nan")
                else:
                    row[field] = _finite(value, label=f"{path}:{field}")
            rows.append(row)
    rows.sort(key=lambda row: row["epoch"])
    epochs = [int(row["epoch"]) for row in rows]
    if not rows or epochs[-1] != max_epoch:
        raise ValueError(f"{path} does not reach epoch {max_epoch}")
    if len(epochs) != len(set(epochs)):
        raise ValueError(f"{path} contains duplicate epochs")
    return rows


def _history_by_epoch(rows: Sequence[Mapping[str, float]]) -> dict[int, Mapping[str, float]]:
    return {int(row["epoch"]): row for row in rows}


def _case_rows_from_metrics(artifacts: Mapping[tuple[str, str], MetricArtifact]) -> dict[str, list[dict[str, Any]]]:
    """Construct paired rows directly from metrics JSON as a cross-check source."""

    output: dict[str, list[dict[str, Any]]] = {}
    for endpoint in ENDPOINT_ORDER:
        classic = artifacts[("classic", endpoint)].payload["cases"]
        dense = artifacts[("dense", endpoint)].payload["cases"]
        rows: list[dict[str, Any]] = []
        for left, right in zip(classic, dense):
            left_id = _metric_case_identity(left)
            right_id = _metric_case_identity(right)
            if left_id != right_id:
                raise ValueError(f"{endpoint} metrics pair identity changed while building paired rows")
            classic_rmse = _finite(left.get("volume_rmse_mps"), label=f"{endpoint} classic volume_rmse_mps")
            dense_rmse = _finite(right.get("volume_rmse_mps"), label=f"{endpoint} dense volume_rmse_mps")
            rows.append(
                {
                    "endpoint": endpoint,
                    "source_index": int(left["source_index"]),
                    "case": str(left["case"]),
                    "layout_index": int(left["layout_index"]),
                    "wind_direction_deg": float(left["wind_direction_deg"]),
                    "classic_volume_rmse_mps": classic_rmse,
                    "dense_volume_rmse_mps": dense_rmse,
                    "dense_minus_classic_volume_rmse_mps": dense_rmse - classic_rmse,
                }
            )
        output[endpoint] = rows
    return output


def load_and_validate_per_case(
    path: Path,
    artifacts: Mapping[tuple[str, str], MetricArtifact],
) -> list[dict[str, Any]]:
    """Load the completed per-case artifact and reconcile it to metrics JSON."""

    required = {
        "endpoint",
        "source_index",
        "case",
        "layout_index",
        "wind_direction_deg",
        "classic_volume_rmse_mps",
        "dense_volume_rmse_mps",
        "dense_minus_classic_volume_rmse_mps",
    }
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} is empty")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    expected = _case_rows_from_metrics(artifacts)
    reconciled: list[dict[str, Any]] = []
    for endpoint in ENDPOINT_ORDER:
        supplied = [row for row in rows if row.get("endpoint") == endpoint]
        if len(supplied) != len(expected[endpoint]):
            raise ValueError(f"{path} has {len(supplied)} {endpoint} rows; expected {len(expected[endpoint])}")
        supplied_by_id = {(str(row["case"]), int(row["source_index"])): row for row in supplied}
        if len(supplied_by_id) != len(supplied):
            raise ValueError(f"{path} has duplicate case/source identities in {endpoint}")
        for wanted in expected[endpoint]:
            key = (wanted["case"], wanted["source_index"])
            row = supplied_by_id.get(key)
            if row is None:
                raise ValueError(f"{path} is missing {endpoint} case {key}")
            for field in ("layout_index",):
                if int(row[field]) != wanted[field]:
                    raise ValueError(f"{path} {endpoint} {key} has {field} mismatch")
            for field in ("wind_direction_deg", "classic_volume_rmse_mps", "dense_volume_rmse_mps", "dense_minus_classic_volume_rmse_mps"):
                supplied_value = _finite(row[field], label=f"{path}:{field}")
                if not math.isclose(supplied_value, float(wanted[field]), rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError(f"{path} {endpoint} {key} disagrees with metrics JSON for {field}")
            reconciled.append(
                {
                    **wanted,
                    "wind_direction_deg": float(row["wind_direction_deg"]),
                }
            )
    if len(reconciled) != len(rows):
        raise ValueError(f"{path} has unknown endpoint rows")
    return reconciled


def load_strata(path: Path) -> list[dict[str, Any]]:
    required = {
        "endpoint",
        "stratum",
        "classic_cases",
        "dense_cases",
        "classic_layouts",
        "dense_layouts",
        "classic_mean_volume_rmse_mps",
        "dense_mean_volume_rmse_mps",
        "dense_minus_classic_volume_rmse_mps",
        "dense_reduction_percent_volume_rmse",
    }
    with path.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    if not raw_rows:
        raise ValueError(f"{path} is empty")
    missing = sorted(required - set(raw_rows[0]))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        endpoint = str(row["endpoint"])
        if endpoint not in ENDPOINT_ORDER:
            raise ValueError(f"{path} has unknown endpoint {endpoint!r}")
        parsed = dict(row)
        for field in ("classic_cases", "dense_cases", "classic_layouts", "dense_layouts"):
            parsed[field] = int(row[field])
        for field in (
            "classic_mean_volume_rmse_mps",
            "dense_mean_volume_rmse_mps",
            "dense_minus_classic_volume_rmse_mps",
            "dense_reduction_percent_volume_rmse",
        ):
            parsed[field] = _finite(row[field], label=f"{path}:{field}")
        if parsed["classic_cases"] != parsed["dense_cases"] or parsed["classic_layouts"] != parsed["dense_layouts"]:
            raise ValueError(f"{path} has unpaired strata counts for {endpoint}:{row['stratum']}")
        if not math.isclose(
            parsed["dense_minus_classic_volume_rmse_mps"],
            parsed["dense_mean_volume_rmse_mps"] - parsed["classic_mean_volume_rmse_mps"],
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError(f"{path} has inconsistent dense-minus-classic stratum delta")
        rows.append(parsed)
    return rows


def build_accuracy_rows(artifacts: Mapping[tuple[str, str], MetricArtifact]) -> list[dict[str, Any]]:
    rows = []
    for endpoint in ENDPOINT_ORDER:
        for model in MODEL_ORDER:
            payload = artifacts[(model, endpoint)].payload
            equal = payload["equal_case"]
            pooled = payload["pooled"]["volume"]
            rows.append(
                {
                    "endpoint": endpoint,
                    "endpoint_label": ENDPOINT_LABELS[endpoint],
                    "model": model,
                    "split": payload["split"],
                    "checkpoint_epoch": int(payload["checkpoint_epoch"]),
                    "checkpoint_role": payload["selection_metric_provenance"]["checkpoint_role"],
                    "rows": int(payload["rows"]),
                    "layouts": len({int(case["layout_index"]) for case in payload["cases"]}),
                    "q_volume": int(payload["q_volume"]),
                    "q_band": int(payload["q_band"]),
                    "sample_seed": int(payload["sample_seed"]),
                    "equal_case_physical_rmse_mps": _finite(equal["mean_volume_rmse_mps"], label="equal-case RMSE"),
                    "equal_case_standardized_mse": _finite(equal["mean_volume_standardized_mse"], label="equal-case standardized MSE"),
                    "pooled_vector_relative_l2": _finite(pooled["vector_relative_l2"], label="pooled vector L2"),
                }
            )
    return rows


def bootstrap_layout_direction_means(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap 30 layout-level means of dense-minus-classic RMSE.

    The result is descriptive for the supplied held-out layouts.  It is not a
    population-generalization interval.
    """

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    groups: dict[int, list[float]] = defaultdict(list)
    for row in paired_rows:
        groups[int(row["layout_index"])].append(float(row["dense_minus_classic_volume_rmse_mps"]))
    if len(groups) != 30 or {len(values) for values in groups.values()} != {3}:
        raise ValueError(f"expected 30 layouts with three direction deltas each, got { {key: len(v) for key, v in groups.items()} }")
    layout_indices = sorted(groups)
    layout_means = np.asarray([np.mean(groups[index]) for index in layout_indices], dtype=float)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(layout_means), size=(replicates, len(layout_means)))
    sampled_means = layout_means[sample_indices].mean(axis=1)
    return {
        "n_layouts": len(layout_means),
        "directions_per_layout": 3,
        "replicates": int(replicates),
        "seed": int(seed),
        "estimate_mean_dense_minus_classic_volume_rmse_mps": float(np.mean(layout_means)),
        "bootstrap_lower_2_5_percent_mps": float(np.quantile(sampled_means, 0.025)),
        "bootstrap_upper_97_5_percent_mps": float(np.quantile(sampled_means, 0.975)),
        "descriptive_scope": "held-out layouts only; not population-generalization",
        "layout_means": [
            {"layout_index": index, "mean_dense_minus_classic_volume_rmse_mps": float(mean)}
            for index, mean in zip(layout_indices, layout_means)
        ],
    }


def build_paired_bootstrap_rows(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    rows = []
    for endpoint in ENDPOINT_ORDER:
        result = bootstrap_layout_direction_means(
            [row for row in paired_rows if row["endpoint"] == endpoint],
            replicates=replicates,
            seed=seed,
        )
        rows.append({"endpoint": endpoint, **{key: value for key, value in result.items() if key != "layout_means"}})
    return rows


def build_paired_bootstrap_detail_rows(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Return one auditable row for every endpoint/layout bootstrap unit."""

    rows = []
    for endpoint in ENDPOINT_ORDER:
        result = bootstrap_layout_direction_means(
            [row for row in paired_rows if row["endpoint"] == endpoint],
            replicates=replicates,
            seed=seed,
        )
        for layout in result["layout_means"]:
            rows.append(
                {
                    "endpoint": endpoint,
                    "layout_index": int(layout["layout_index"]),
                    "layout_mean_dense_minus_classic_volume_rmse_mps": float(
                        layout["mean_dense_minus_classic_volume_rmse_mps"]
                    ),
                    "estimate_mean_dense_minus_classic_volume_rmse_mps": result[
                        "estimate_mean_dense_minus_classic_volume_rmse_mps"
                    ],
                    "bootstrap_lower_2_5_percent_mps": result["bootstrap_lower_2_5_percent_mps"],
                    "bootstrap_upper_97_5_percent_mps": result["bootstrap_upper_97_5_percent_mps"],
                    "replicates": result["replicates"],
                    "seed": result["seed"],
                    "descriptive_scope": result["descriptive_scope"],
                }
            )
    return rows


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "font.size": 10,
            "savefig.dpi": 180,
        }
    )
    return plt


def _save_figure(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    fig.clf()
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def plot_learning_curves(
    histories: Mapping[str, Sequence[Mapping[str, float]]],
    selected_epochs: Sequence[int],
    best_epochs: Mapping[str, int],
    destination: Path,
) -> Path:
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True, constrained_layout=True)
    panels = (
        ("train_volume_mse", "val_volume_mse", "Volume standardized MSE"),
        ("train_band_mse", "val_band_mse", "Hub-band standardized MSE"),
    )
    for ax, (train_metric, validation_metric, title) in zip(axes, panels):
        for model in MODEL_ORDER:
            rows = histories[model]
            epoch = np.asarray([row["epoch"] for row in rows], dtype=float)
            train_value = np.asarray([row[train_metric] for row in rows], dtype=float)
            validation_value = np.asarray([row[validation_metric] for row in rows], dtype=float)
            color = MODEL_COLORS[model]
            ax.plot(epoch, train_value, color=color, lw=1.2, ls="--", alpha=.65, label=f"{model.capitalize()} train")
            ax.plot(epoch, validation_value, color=color, lw=1.8, label=f"{model.capitalize()} validation")
            lookup = _history_by_epoch(rows)
            for selected in selected_epochs:
                row = lookup.get(int(selected))
                if row is not None and math.isfinite(row[validation_metric]):
                    ax.plot(selected, row[validation_metric], marker="o", ms=3.3, color=color)
            best = int(best_epochs[model])
            best_row = lookup.get(best)
            if best_row is not None and math.isfinite(best_row[validation_metric]):
                ax.plot(best, best_row[validation_metric], marker="D", ms=4.2, color=color, markeredgecolor="white", mew=.5)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("training epoch")
        ax.set_ylabel("MSE")
        ax.set_xticks(sorted({int(epoch) for epoch in selected_epochs}))
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", color="#D9D9D9", lw=.6, alpha=.7)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Run 2102 / Run 2103 aligned learning curves through epoch 2,500", fontsize=12)
    return _save_figure(fig, destination)


def plot_accuracy_overview(rows: Sequence[Mapping[str, Any]], destination: Path) -> Path:
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    panels = (
        ("equal_case_physical_rmse_mps", "Equal-case physical volume RMSE [m/s]"),
        ("equal_case_standardized_mse", "Equal-case volume standardized MSE [normalized]"),
        ("pooled_vector_relative_l2", "Pooled volume vector relative L2 [dimensionless]"),
    )
    x = np.arange(len(ENDPOINT_ORDER), dtype=float)
    width = .36
    by_key = {(str(row["model"]), str(row["endpoint"])): row for row in rows}
    for ax, (field, ylabel) in zip(axes, panels):
        for offset, model in ((-width / 2, "classic"), (width / 2, "dense")):
            values = [float(by_key[(model, endpoint)][field]) for endpoint in ENDPOINT_ORDER]
            ax.bar(x + offset, values, width=width, color=MODEL_COLORS[model], label=model.capitalize())
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", lw=.6, alpha=.7)
    for ax in axes:
        ax.set_xticks(x, [ENDPOINT_LABELS[endpoint] for endpoint in ENDPOINT_ORDER], rotation=28, ha="right")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    fig.suptitle("WindFarm forward accuracy overview: metrics retain their distinct aggregation meanings", fontsize=12)
    return _save_figure(fig, destination)


def plot_paired_rmse(paired_rows: Sequence[Mapping[str, Any]], destination: Path) -> Path:
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, endpoint in zip(axes, ENDPOINT_ORDER):
        subset = [row for row in paired_rows if row["endpoint"] == endpoint]
        classic = np.asarray([row["classic_volume_rmse_mps"] for row in subset])
        dense = np.asarray([row["dense_volume_rmse_mps"] for row in subset])
        upper = max(float(np.max(classic)), float(np.max(dense))) * 1.03
        ax.scatter(classic, dense, s=19, alpha=.7, color=MODEL_COLORS["dense"], edgecolors="none")
        ax.plot([0, upper], [0, upper], color="#555555", lw=1, ls="--")
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(ENDPOINT_LABELS[endpoint])
        ax.set_xlabel("Classic volume RMSE [m/s]")
        ax.grid(color="#D9D9D9", lw=.6, alpha=.55)
    axes[0].set_ylabel("Dense volume RMSE [m/s]")
    fig.suptitle("Paired held-out case physical RMSE (90 aligned cases per endpoint)", fontsize=12)
    return _save_figure(fig, destination)


def plot_paired_deltas(
    paired_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    destination: Path,
) -> Path:
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    bootstrap_by_endpoint = {str(row["endpoint"]): row for row in bootstrap_rows}
    for ax, endpoint in zip(axes, ENDPOINT_ORDER):
        subset = [row for row in paired_rows if row["endpoint"] == endpoint]
        values = np.sort(np.asarray([row["dense_minus_classic_volume_rmse_mps"] for row in subset], dtype=float))
        x = np.arange(1, len(values) + 1)
        ax.scatter(x, values, s=11, alpha=.65, color=MODEL_COLORS["dense"], edgecolors="none")
        summary = bootstrap_by_endpoint[endpoint]
        estimate = float(summary["estimate_mean_dense_minus_classic_volume_rmse_mps"])
        lower = float(summary["bootstrap_lower_2_5_percent_mps"])
        upper = float(summary["bootstrap_upper_97_5_percent_mps"])
        ax.axhline(estimate, color="#000000", lw=1.5, label="layout-mean estimate")
        ax.axhspan(lower, upper, color="#999999", alpha=.18, label="10k bootstrap 95% interval")
        ax.axhline(0.0, color="#666666", lw=.8, ls="--")
        ax.set_title(ENDPOINT_LABELS[endpoint])
        ax.set_xlabel("paired cases sorted by delta")
        ax.set_ylabel("Dense − Classic RMSE [m/s]")
        ax.text(.02, .96, f"mean {estimate:+.4f} m/s", transform=ax.transAxes, va="top", fontsize=8.5)
        ax.grid(axis="y", color="#D9D9D9", lw=.6, alpha=.7)
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(.5, -.02, "Bootstrap resamples 30 layout-level direction means; descriptive for these held-out layouts, not population-generalization.", ha="center", fontsize=8.5)
    fig.suptitle("Paired physical RMSE deltas", fontsize=12)
    return _save_figure(fig, destination)


def _stratum_family(value: str) -> str:
    return value.split("=", 1)[0]


def plot_strata(rows: Sequence[Mapping[str, Any]], destination: Path) -> Path:
    plt = _mpl()
    families = [
        "M_bin",
        "aspect_xy_bin",
        "direction",
        "volume_quartile",
        "mean_nn_D_bin",
        "min_sep_D_bin",
        "nn_dispersion_bin",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.4), constrained_layout=True)
    axes_flat = axes.ravel()
    for axis_index, family in enumerate(families):
        ax = axes_flat[axis_index]
        subset = [row for row in rows if _stratum_family(str(row["stratum"])) == family]
        labels = sorted({str(row["stratum"]).split("=", 1)[1] for row in subset}, key=lambda value: (len(value), value))
        x = np.arange(len(labels), dtype=float)
        for endpoint in ENDPOINT_ORDER:
            values = []
            for label in labels:
                matches = [row for row in subset if row["endpoint"] == endpoint and str(row["stratum"]).split("=", 1)[1] == label]
                values.append(float(matches[0]["dense_reduction_percent_volume_rmse"]) if matches else float("nan"))
            ax.plot(x, values, marker="o", ms=3.5, lw=1.3, color=ENDPOINT_COLORS[endpoint], label=ENDPOINT_LABELS[endpoint])
        ax.axhline(0.0, color="#666666", lw=.7, ls="--")
        ax.set_title(family.replace("_", " "))
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylabel("Dense reduction [%]")
        ax.grid(axis="y", color="#D9D9D9", lw=.6, alpha=.7)
    axes_flat[-1].axis("off")
    axes_flat[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Geometry-stratum comparison (dense reduction in equal-case physical volume RMSE)", fontsize=12)
    return _save_figure(fig, destination)


def load_cost_records(assignments: Mapping[str, Path]) -> list[dict[str, Any]]:
    records = []
    for supplied_label, path in assignments.items():
        payload = _read_json(path)
        label = str(payload.get("label", supplied_label)).strip() or supplied_label
        if label != supplied_label:
            raise ValueError(f"{path} label={label!r} disagrees with supplied label {supplied_label!r}")
        full = payload.get("full_prepare_plus_decode")
        update = payload.get("disposable_update")
        prepared = payload.get("prepared_decode")
        if not isinstance(full, dict) or not isinstance(update, dict) or not isinstance(prepared, dict):
            raise TypeError(f"{path} lacks matched full_prepare_plus_decode/disposable_update/prepared_decode records")
        records.append(
            {
                "model": label,
                "protocol": payload.get("protocol"),
                "queries_per_case": payload.get("queries_per_case"),
                "receiver_chunk_size": payload.get("receiver_chunk_size"),
                "trainable_parameters": int(payload["trainable_parameters"]),
                "full_prepare_plus_decode_mean_ms": _finite(full["mean_ms"], label=f"{path}:full mean_ms"),
                "full_prepare_plus_decode_peak_allocated_mib": _finite(full["peak_allocated_mib"], label=f"{path}:full peak allocated"),
                "disposable_update_mean_ms": _finite(update["milliseconds"], label=f"{path}:update milliseconds"),
                "disposable_update_peak_allocated_mib": _finite(update["peak_allocated_mib"], label=f"{path}:update peak allocated"),
                "prepared_decode_mean_ms": _finite(prepared["mean_ms"], label=f"{path}:prepared mean_ms"),
                "source_path": str(path),
            }
        )
    if {row["model"] for row in records} != set(MODEL_ORDER):
        raise ValueError(f"cost JSONs must include exactly {MODEL_ORDER}")
    return sorted(records, key=lambda row: MODEL_ORDER.index(row["model"]))


def plot_cost(records: Sequence[Mapping[str, Any]], destination: Path) -> Path:
    plt = _mpl()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fields = (
        ("full_prepare_plus_decode_mean_ms", "Full prepare + decode mean [ms]"),
        ("disposable_update_mean_ms", "One disposable update [ms]"),
        ("trainable_parameters", "Trainable parameters [count]"),
        ("full_prepare_plus_decode_peak_allocated_mib", "Full prepare + decode peak allocated [MiB]"),
    )
    x = np.arange(len(records))
    labels = [str(row["model"]).capitalize() for row in records]
    for ax, (field, ylabel) in zip(axes.ravel(), fields):
        values = [float(row[field]) for row in records]
        if field == "trainable_parameters":
            values = [value / 1e6 for value in values]
            ylabel = "Trainable parameters [million]"
        ax.bar(x, values, color=[MODEL_COLORS[str(row["model"])] for row in records], width=.56)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", lw=.6, alpha=.7)
    fig.suptitle("Matched WindFarm model-cost measurements", fontsize=12)
    return _save_figure(fig, destination)


def _output_paths(output_dir: Path, include_cost: bool) -> dict[str, Path]:
    figures = output_dir / "figures"
    evidence = output_dir / "evidence"
    paths = {
        "learning_curve_figure": figures / "learning_curves.png",
        "accuracy_overview_figure": figures / "accuracy_overview.png",
        "paired_rmse_figure": figures / "paired_physical_rmse.png",
        "paired_delta_figure": figures / "paired_physical_rmse_deltas.png",
        "strata_figure": figures / "geometry_strata_comparison.png",
        "validation_audit": evidence / "validation_audit.json",
        "analysis_manifest": evidence / "analysis_manifest.json",
        "accuracy_table": evidence / "accuracy_overview.csv",
        "learning_table": evidence / "learning_curve_selected.csv",
        "paired_table": evidence / "paired_case_rmse.csv",
        "bootstrap_table": evidence / "paired_layout_bootstrap.csv",
        "bootstrap_summary_table": evidence / "paired_layout_bootstrap_summary.csv",
        "strata_table": evidence / "strata_comparison.csv",
    }
    if include_cost:
        paths.update({"cost_figure": figures / "model_cost.png", "cost_table": evidence / "model_cost.csv"})
    return paths


def generate_report(
    *,
    artifacts: Mapping[tuple[str, str], MetricArtifact],
    histories: Mapping[str, Path],
    alignment_path: Path,
    endpoint_summary_path: Path,
    per_case_path: Path,
    strata_path: Path,
    output_dir: Path,
    selected_epochs: Sequence[int] = (500, 1000, 1500, 2000, 2500),
    max_epoch: int = EXPECTED_EPOCH,
    cost_paths: Mapping[str, Path] | None = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Generate all requested figures/tables after validating inputs."""

    if set(histories) != set(MODEL_ORDER):
        raise ValueError(f"histories must contain exactly {MODEL_ORDER}")
    selected = sorted({int(epoch) for epoch in selected_epochs})
    if not selected or selected[0] <= 0 or selected[-1] > max_epoch:
        raise ValueError("selected epochs must be positive and no later than max_epoch")
    audit = validate_endpoint_artifacts(
        artifacts,
        alignment_path=alignment_path,
        endpoint_summary_path=endpoint_summary_path,
    )
    histories_loaded = {model: _load_history(path, max_epoch=max_epoch) for model, path in histories.items()}
    paired_rows = load_and_validate_per_case(per_case_path, artifacts)
    strata_rows = load_strata(strata_path)
    accuracy_rows = build_accuracy_rows(artifacts)
    best_epochs = {
        model: int(artifacts[(model, "best_validation")].payload["checkpoint_epoch"])
        for model in MODEL_ORDER
    }
    learning_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        lookup = _history_by_epoch(histories_loaded[model])
        epochs = sorted(set(selected) | {best_epochs[model]})
        for epoch in epochs:
            row = lookup.get(epoch)
            if row is None:
                raise ValueError(f"history for {model} lacks selected epoch {epoch}")
            learning_rows.append(
                {
                    "model": model,
                    "epoch": epoch,
                    "is_selected_common_epoch": epoch in selected,
                    "is_best_validation_epoch": epoch == best_epochs[model],
                    **{field: (None if not math.isfinite(float(row[field])) else float(row[field])) for field in ("loss_total", "train_volume_mse", "val_volume_mse", "train_band_mse", "val_band_mse")},
                }
            )
    bootstrap_rows = build_paired_bootstrap_rows(
        paired_rows,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    bootstrap_detail_rows = build_paired_bootstrap_detail_rows(
        paired_rows,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    paths = _output_paths(output_dir, bool(cost_paths))
    accuracy_fields = list(accuracy_rows[0])
    learning_fields = list(learning_rows[0])
    paired_fields = list(paired_rows[0])
    bootstrap_fields = list(bootstrap_rows[0])
    bootstrap_detail_fields = list(bootstrap_detail_rows[0])
    strata_fields = list(strata_rows[0])
    _write_csv(paths["accuracy_table"], accuracy_rows, accuracy_fields)
    _write_csv(paths["learning_table"], learning_rows, learning_fields)
    _write_csv(paths["paired_table"], paired_rows, paired_fields)
    _write_csv(paths["bootstrap_table"], bootstrap_detail_rows, bootstrap_detail_fields)
    _write_csv(paths["bootstrap_summary_table"], bootstrap_rows, bootstrap_fields)
    _write_csv(paths["strata_table"], strata_rows, strata_fields)
    plot_learning_curves(histories_loaded, selected, best_epochs, paths["learning_curve_figure"])
    plot_accuracy_overview(accuracy_rows, paths["accuracy_overview_figure"])
    plot_paired_rmse(paired_rows, paths["paired_rmse_figure"])
    plot_paired_deltas(paired_rows, bootstrap_rows, paths["paired_delta_figure"])
    plot_strata(strata_rows, paths["strata_figure"])
    cost_records: list[dict[str, Any]] = []
    if cost_paths:
        cost_records = load_cost_records(cost_paths)
        _write_csv(paths["cost_table"], cost_records, list(cost_records[0]))
        plot_cost(cost_records, paths["cost_figure"])
    _write_json(paths["validation_audit"], audit)
    manifest = {
        "schema_version": 1,
        "inputs": {
            "metrics": {f"{model}:{endpoint}": str(artifact.path) for (model, endpoint), artifact in artifacts.items()},
            "histories": {model: str(path) for model, path in histories.items()},
            "alignment": str(alignment_path),
            "endpoint_summary": str(endpoint_summary_path),
            "per_case": str(per_case_path),
            "strata": str(strata_path),
            "cost": {model: str(path) for model, path in (cost_paths or {}).items()},
        },
        "selected_epochs": selected,
        "max_epoch": max_epoch,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "unit": "layout-level mean of three wind-direction deltas",
            "scope": "descriptive for these held-out layouts; not population-generalization",
        },
        "generated": {name: str(path) for name, path in paths.items()},
    }
    _write_json(paths["analysis_manifest"], manifest)
    return {"audit": audit, "manifest": manifest, "paths": {name: str(path) for name, path in paths.items()}, "cost_records": cost_records}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", action="append", required=True, metavar="MODEL:ENDPOINT=PATH", help="Repeat six times.")
    parser.add_argument("--history", action="append", required=True, metavar="MODEL=PATH", help="Repeat once for classic and dense.")
    parser.add_argument("--alignment-json", type=Path, required=True)
    parser.add_argument("--endpoint-summary", type=Path, required=True)
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--strata", type=Path, required=True)
    parser.add_argument("--cost-json", action="append", default=[], metavar="MODEL=PATH", help="Optional matched cost JSON; repeat for both models.")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports/windfarm_forward_2500_comparison"))
    parser.add_argument("--selected-epochs", default="500,1000,1500,2000,2500")
    parser.add_argument("--max-epoch", type=int, default=EXPECTED_EPOCH)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metric_paths = parse_metric_assignments(args.metrics)
        histories = dict(parse_assignment(raw) for raw in args.history)
        cost_paths = dict(parse_assignment(raw) for raw in args.cost_json)
        selected_epochs = [int(value.strip()) for value in args.selected_epochs.split(",") if value.strip()]
        result = generate_report(
            artifacts=load_metric_artifacts(metric_paths),
            histories=histories,
            alignment_path=args.alignment_json,
            endpoint_summary_path=args.endpoint_summary,
            per_case_path=args.per_case,
            strata_path=args.strata,
            output_dir=args.output_dir,
            selected_epochs=selected_epochs,
            max_epoch=args.max_epoch,
            cost_paths=cost_paths or None,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"forward analysis figure generation failed: {exc}") from exc
    print(json.dumps(result["paths"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
