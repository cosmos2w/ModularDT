#!/usr/bin/env python3
"""Fit and audit a layout-grouped WindFarm packet-action risk selector.

The native action table is produced at one frozen forward checkpoint by
``evaluate_matured_action_table.py``. Only planner features enter either fit;
measured role errors are used for train labels and post-selection audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from honf_forward_core.interface_fields.action_aware_frontier import incumbent_role_log_limit
from honf_forward_core.interface_fields.action_risk_fit import (
    ActionEvidenceRow,
    crossfit_action_risk,
    evaluate_action_policy,
    fit_action_risk_head,
    fit_ridge_action_baseline,
)

ACTION_KEYS = ("root", "two_packet", "four_packet", "full_access")
RELATIVE_ALLOWANCE = 0.10
MAX_FIT_UPDATES = 200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _float_tensor(values: Any) -> torch.Tensor:
    result = torch.as_tensor(np.asarray(values).copy(), dtype=torch.float32)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Action table contains nonfinite features or role errors.")
    return result


def _checked_features(
    npz: Mapping[str, np.ndarray], row: Mapping[str, Any], digests: Mapping[str, str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = []
    for name in ("packet_rows", "budget_vector", "receiver_role_features"):
        key = str(row[f"{name}_npz_key"])
        if key not in npz:
            raise ValueError(f"Action feature {key!r} is absent from the bound NPZ.")
        value = np.asarray(npz[key])
        expected = str(digests[key])
        actual = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Action feature {key!r} fails its SHA-256 binding.")
        tensors.append(_float_tensor(value))
    packet, budget, roles = tensors
    if packet.ndim != 2 or budget.ndim != 1 or roles.ndim != 2:
        raise ValueError("Packet, budget, and receiver-role feature ranks changed.")
    return packet, budget, roles


def _role_tensor(values: Mapping[str, Any] | Sequence[float], roles: Sequence[str]) -> torch.Tensor:
    if isinstance(values, Mapping):
        if set(values) != set(roles):
            raise ValueError("Physical role error map differs from the bound Wind role order.")
        values = [values[role] for role in roles]
    return _float_tensor(values)


def _load_action_table(
    table_path: Path, feature_path: Path, mask_path: Path, checkpoint_path: Path,
) -> tuple[list[ActionEvidenceRow], dict[str, Any]]:
    payload = json.loads(table_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("Unknown Wind action table format.")
    expected_forward_sha = str(payload["forward_checkpoint_sha256"])
    if _sha256(checkpoint_path) != expected_forward_sha:
        raise ValueError("Selected G checkpoint differs from the measured action table.")
    if _sha256(feature_path) != payload["feature_npz_sha256"]:
        raise ValueError("Planner feature archive differs from the measured action table.")
    if _sha256(mask_path) != payload["mask_npz_sha256"]:
        raise ValueError("Realized permission-mask archive differs from the measured action table.")
    observed_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in payload["rows"]:
        observed_splits[str(row["split"])].add(str(row["family_key"]))
    split_families = {key: tuple(sorted(value)) for key, value in sorted(observed_splits.items())}
    if len(split_families.get("train_fit", ())) < 2:
        raise ValueError("Layout grouped risk fitting needs at least two train layouts.")
    split_sets = [set(values) for values in split_families.values()]
    if any(split_sets[i] & split_sets[j] for i in range(len(split_sets)) for j in range(i + 1, len(split_sets))):
        raise ValueError("Physical layout families overlap across splits.")
    roles = tuple(payload["roles"])
    if len(roles) != 5:
        raise ValueError("Wind role order must have five physical roles.")
    floor = _role_tensor(payload["numerical_floor_role_mps"], roles)
    allowance = _role_tensor(payload["absolute_allowance_role_mps"], roles)
    if floor.shape != (5,) or not bool((floor > 0).all()) or allowance.shape != (5,) or bool((allowance < 0).any()):
        raise ValueError("Role floor and physical allowance must be finite and aligned.")
    case_metadata: dict[str, dict[str, Any]] = {}
    rows: list[ActionEvidenceRow] = []
    with np.load(feature_path, allow_pickle=False) as npz:
        for action in payload["rows"]:
            case_key = str(action["case_key"])
            family_key = str(action["family_key"])
            split = str(action["split"])
            if split not in split_families or family_key not in split_families[split]:
                raise ValueError(f"Case {case_key!r} has an unbound split or layout family.")
            if action["forward_checkpoint_sha256"] != expected_forward_sha:
                raise ValueError(f"Case {case_key!r} uses stale G physical or scorer weights.")
            identity = {"family_key": family_key, "split": split,
                        "query_panel": str(action["query_panel"]),
                        "row_id": int(action["row_id"]), "layout_id": int(action["layout_id"]),
                        "module_count": int(action["module_count"])}
            if case_key in case_metadata and case_metadata[case_key] != identity:
                raise ValueError(f"Case {case_key!r} changes identity across actions.")
            case_metadata[case_key] = identity
            action_key = str(action["action_key"])
            if action_key not in ACTION_KEYS:
                raise ValueError(f"Unknown Wind action {action_key!r}.")
            packet, budget, descriptors = _checked_features(
                npz, action, payload["feature_sha256_by_npz_key"]
            )
            incumbent = _role_tensor(action["incumbent_role_error"], roles)
            candidate = _role_tensor(action["candidate_role_error"], roles)
            if any(value.shape != (5,) or bool((value < 0).any()) for value in (candidate, incumbent)):
                raise ValueError("Candidate or W-full reference physical role errors are invalid.")
            is_full = action_key == "full_access"
            if bool(action["full_access"]) != is_full:
                raise ValueError("Only the explicit full-access row may bypass packet selection.")
            count = action["nonredundant_k"]
            if is_full and count is not None:
                raise ValueError("Full access must have undefined external packet K.")
            if not is_full and (not isinstance(count, int) or count < 1):
                raise ValueError("Sparse packet action needs measured nonredundant K.")
            trained_sparse = bool(action["trained_sparse"])
            if is_full and trained_sparse:
                raise ValueError("Full fallback cannot count as sparse exposure.")
            exact_work = float(action["exact_work"])
            full_work = float(action["full_work"])
            if not math.isfinite(exact_work) or not math.isfinite(full_work) or full_work <= 0 or not 0 <= exact_work <= full_work * (1 + 1e-6):
                raise ValueError("Canonical action and full work must be raw finite work units.")
            rows.append(ActionEvidenceRow(
                family_key=family_key,
                case_key=case_key,
                action_key=action_key,
                forward_sha256=expected_forward_sha,
                packet_rows=packet,
                budget_vector=budget,
                receiver_role_features=descriptors,
                candidate_role_error=candidate,
                incumbent_role_error=incumbent,
                numerical_floor=floor,
                exact_work=exact_work,
                nonredundant_k=1 if is_full else count,
                trained_sparse=trained_sparse,
                full_access=is_full,
            ))
    actual_actions: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.action_key in actual_actions[row.case_key]:
            raise ValueError(f"Repeated action {row.action_key!r} in {row.case_key!r}.")
        actual_actions[row.case_key].add(row.action_key)
    if any(actions != set(ACTION_KEYS) for actions in actual_actions.values()):
        raise ValueError("Every Wind case must have one exact four-action frontier.")
    return rows, {
        "source": payload,
        "case_metadata": case_metadata,
        "split_families": split_families,
        "roles": roles,
        "allowance": allowance,
        "checkpoint_sha256": expected_forward_sha,
    }


def _train_fixed_limits(
    rows: Sequence[ActionEvidenceRow], train_families: Sequence[str],
    case_metadata: Mapping[str, Mapping[str, Any]], allowance: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    calibration = []
    for family in sorted(train_families):
        primary = [
            row for row in rows if row.family_key == family and row.full_access
            and case_metadata[row.case_key]["query_panel"] == "fixed"
        ]
        if not primary:
            raise ValueError(f"Train layout {family!r} needs a fixed full-access panel.")
        incumbent_median = torch.stack([row.incumbent_role_error.double() for row in primary]).median(dim=0).values
        floor = primary[0].numerical_floor.double()
        if any(not torch.equal(row.numerical_floor.double(), floor) for row in primary[1:]):
            raise ValueError(f"Train layout {family!r} changes the role numerical floor across directions.")
        limit = incumbent_role_log_limit(
            incumbent_median, floor,
            allowance.double(), relative_allowance=RELATIVE_ALLOWANCE,
        )
        calibration.append({"family_key": family, "case_keys": [row.case_key for row in primary],
                            "fixed_direction_count": len(primary),
                            "incumbent_role_rmse_mps_median": incumbent_median.tolist(),
                            "role_log_limit": limit.tolist()})
    limits = torch.tensor([record["role_log_limit"] for record in calibration]).amin(dim=0)
    return limits, calibration


def _predictions(
    rows: Sequence[ActionEvidenceRow], crossfit: Any, neural_fit: Any,
    ridge_fit: Any, train_families: set[str],
) -> tuple[dict[tuple[str, str], torch.Tensor], dict[tuple[str, str], torch.Tensor]]:
    train_neural = dict(zip(crossfit.row_keys, crossfit.neural, strict=True))
    train_ridge = dict(zip(crossfit.row_keys, crossfit.ridge, strict=True))
    neural: dict[tuple[str, str], torch.Tensor] = {}
    ridge: dict[tuple[str, str], torch.Tensor] = {}
    with torch.no_grad():
        for row in rows:
            if not row.trained_sparse and not row.full_access:
                continue
            key = (row.case_key, row.action_key)
            if row.family_key in train_families:
                neural[key] = train_neural[key]
                ridge[key] = train_ridge[key]
            else:
                neural[key] = neural_fit.model(
                    row.packet_rows, row.budget_vector, row.receiver_role_features,
                    nonredundant_k=row.nonredundant_k,
                ).detach().cpu()
                ridge[key] = ridge_fit.predict(row).detach().cpu()
    return neural, ridge


def _ranking(
    rows: Sequence[ActionEvidenceRow], prediction: Mapping[tuple[str, str], torch.Tensor],
) -> dict[str, Any]:
    grouped: dict[str, list[ActionEvidenceRow]] = defaultdict(list)
    for row in rows:
        if row.trained_sparse:
            grouped[row.case_key].append(row)
    concordant = discordant = tied = 0
    for case_rows in grouped.values():
        for left_index, left in enumerate(case_rows):
            for right in case_rows[left_index + 1:]:
                actual = float(left.target_log_risk.max() - right.target_log_risk.max())
                if abs(actual) <= 1e-3:
                    continue
                predicted = float(prediction[(left.case_key, left.action_key)].max()
                                  - prediction[(right.case_key, right.action_key)].max())
                if abs(predicted) <= 1e-3:
                    tied += 1
                elif predicted * actual > 0:
                    concordant += 1
                else:
                    discordant += 1
    return {"comparable_action_pairs": concordant + discordant + tied,
            "concordant": concordant, "discordant": discordant, "tied": tied,
            "accuracy_excluding_ties": concordant / (concordant + discordant)
            if concordant + discordant else None}


def _policy_summary(
    results: Sequence[Any], rows: Sequence[ActionEvidenceRow],
    case_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    full_work = {row.case_key: row.exact_work for row in rows if row.full_access}
    oracle = [row for row in results if row.measured_oracle_action_key is not None]
    by_m: defaultdict[int, Counter[int]] = defaultdict(Counter)
    for row in results:
        if row.selected_nonredundant_k is not None:
            by_m[int(case_metadata[row.case_key]["module_count"])][int(row.selected_nonredundant_k)] += 1
    return {
        "case_count": len(results),
        "measured_oracle_available_count": len(oracle),
        "supported_sparse_deployment_count": sum(row.selected_supported_sparse for row in results),
        "explicit_full_fallback_count": sum(row.unsupported_at_budget for row in results),
        "false_safe_choice_count": sum(len(row.false_safe_sparse) for row in results),
        "false_reject_choice_count": sum(len(row.false_reject_sparse) for row in results),
        "mean_selected_exact_work": float(np.mean([row.selected_work for row in results])) if results else None,
        "mean_selected_work_over_same_case_full": float(np.mean([
            row.selected_work / full_work[row.case_key] for row in results
        ])) if results else None,
        "mean_oracle_sparse_work_when_available": float(np.mean([
            row.measured_oracle_sparse_work for row in oracle
        ])) if oracle else None,
        "selected_nonredundant_k_counts": dict(sorted(Counter(
            row.selected_nonredundant_k for row in results if row.selected_nonredundant_k is not None
        ).items())),
        "selected_nonredundant_k_by_module_count": {
            str(module_count): dict(sorted(counts.items())) for module_count, counts in sorted(by_m.items())
        },
        "selected_action_counts": dict(sorted(Counter(row.selected_action_key for row in results).items())),
        "per_case": [{
            "case_key": row.case_key,
            "family_key": row.family_key,
            "selected_action_key": row.selected_action_key,
            "measured_oracle_action_key": row.measured_oracle_action_key,
            "unsupported_at_budget": row.unsupported_at_budget,
            "selected_supported_sparse": row.selected_supported_sparse,
            "measured_adequate_sparse": list(row.measured_adequate_sparse),
            "predicted_safe_sparse": list(row.predicted_safe_sparse),
            "false_safe_sparse": list(row.false_safe_sparse),
            "false_reject_sparse": list(row.false_reject_sparse),
            "selected_exact_work": row.selected_work,
            "selected_nonredundant_k": row.selected_nonredundant_k,
            "module_count": case_metadata[row.case_key]["module_count"],
        } for row in results],
    }


def fit_and_evaluate(
    rows: Sequence[ActionEvidenceRow], metadata: Mapping[str, Any], *,
    updates: int = 120, seed: int = 2112,
) -> tuple[dict[str, Any], Any | None]:
    if not 1 <= updates <= MAX_FIT_UPDATES:
        raise ValueError("Risk fit update count exceeds the bounded diagnostic allocation.")
    forward_sha = metadata["checkpoint_sha256"]
    splits = metadata["split_families"]
    train_families = tuple(splits["train_fit"])
    train_set = set(train_families)
    allowance = metadata["allowance"].double()
    limits, calibration = _train_fixed_limits(rows, train_families, metadata["case_metadata"], allowance)
    full_work = {row.case_key: row.exact_work for row in rows if row.full_access}
    result: dict[str, Any] = {
        "forward_checkpoint_sha256": forward_sha,
        "role_order": list(metadata["roles"]),
        "split_families": {key: list(value) for key, value in splits.items()},
        "absolute_allowance_role_mps": allowance.tolist(),
        "fixed_train_role_log_limits": limits.tolist(),
        "train_gate_calibration": calibration,
        "relative_physical_allowance": RELATIVE_ALLOWANCE,
        "selector_fit": "unavailable_no_exposed_train_sparse_action",
        "optimizer_calls": 0,
        "split_results": {},
        "action_prediction_rows": [{
            "case_key": row.case_key,
            "family_key": row.family_key,
            "split": metadata["case_metadata"][row.case_key]["split"],
            "query_panel": metadata["case_metadata"][row.case_key]["query_panel"],
            "module_count": metadata["case_metadata"][row.case_key]["module_count"],
            "action_key": row.action_key,
            "trained_sparse": row.trained_sparse,
            "full_access": row.full_access,
            "nonredundant_k": None if row.full_access else row.nonredundant_k,
            "exact_work": row.exact_work,
            "exact_work_over_full": row.exact_work / full_work[row.case_key],
            "candidate_role_rmse_mps": row.candidate_role_error.tolist(),
            "wfull_role_rmse_mps": row.incumbent_role_error.tolist(),
            "measured_signed_log_role_risk": row.target_log_risk.tolist(),
            "neural_predicted_log_role_risk": None,
            "ridge_predicted_log_role_risk": None,
        } for row in rows],
    }
    for split, families in splits.items():
        split_rows = [row for row in rows if row.family_key in set(families)]
        result["split_results"][split] = {
            "case_count": len({row.case_key for row in split_rows}),
            "fixed_action": {action: {
                "measured_rows": sum(row.action_key == action for row in split_rows),
                "exposed_sparse_rows": sum(row.action_key == action and row.trained_sparse for row in split_rows),
                "adequate_sparse_rows": sum(
                    row.action_key == action and row.trained_sparse and bool((
                        row.candidate_role_error.double() <=
                        (1 + RELATIVE_ALLOWANCE) * row.incumbent_role_error.double() + allowance
                    ).all()) for row in split_rows
                ),
                "nonredundant_k_counts": dict(sorted(Counter(
                    row.nonredundant_k for row in split_rows if row.action_key == action and row.trained_sparse
                ).items())),
                "nonredundant_k_by_module_count": {
                    str(module_count): dict(sorted(Counter(
                        row.nonredundant_k for row in split_rows
                        if row.action_key == action and row.trained_sparse
                        and int(metadata["case_metadata"][row.case_key]["module_count"]) == module_count
                    ).items()))
                    for module_count in sorted({
                        int(metadata["case_metadata"][row.case_key]["module_count"])
                        for row in split_rows if row.action_key == action and row.trained_sparse
                    })
                },
            } for action in ACTION_KEYS[:-1]}
        }
    if not any(row.trained_sparse for row in rows if row.family_key in train_set):
        return result, None
    folds = min(5, len(train_families))
    crossfit = crossfit_action_risk(
        rows, current_forward_sha256=forward_sha, train_families=train_families,
        folds=folds, updates=updates, seed=seed,
    )
    neural_fit = fit_action_risk_head(
        rows, current_forward_sha256=forward_sha, train_families=train_families,
        updates=updates, seed=seed + 1997,
    )
    ridge_fit = fit_ridge_action_baseline(
        rows, current_forward_sha256=forward_sha, train_families=train_families,
    )
    neural_map, ridge_map = _predictions(rows, crossfit, neural_fit, ridge_fit, train_set)
    for record in result["action_prediction_rows"]:
        key = (record["case_key"], record["action_key"])
        if key in neural_map:
            record["neural_predicted_log_role_risk"] = neural_map[key].tolist()
            record["ridge_predicted_log_role_risk"] = ridge_map[key].tolist()
    train_in_sample = {}
    with torch.no_grad():
        for row in rows:
            if row.family_key in train_set and row.trained_sparse:
                train_in_sample[(row.case_key, row.action_key)] = neural_fit.model(
                    row.packet_rows, row.budget_vector, row.receiver_role_features,
                    nonredundant_k=row.nonredundant_k,
                ).detach().cpu()
    result.update({
        "selector_fit": "fitted",
        "optimizer_calls": updates * (folds + 1),
        "fit_updates_per_fold": updates,
        "crossfit_folds": folds,
        "neural_train_first_loss": neural_fit.first_loss,
        "neural_train_last_loss": neural_fit.last_loss,
        "ridge_train_feature_mean": ridge_fit.mean.tolist(),
        "ridge_train_feature_scale": ridge_fit.scale.tolist(),
        "ridge_train_coefficients": ridge_fit.coefficients.tolist(),
        "neural_empirical_upper_margin_by_role": crossfit.neural_upper_margin.tolist(),
        "ridge_empirical_upper_margin_by_role": crossfit.ridge_upper_margin.tolist(),
        "train_in_sample_ranking": _ranking(
            [row for row in rows if row.family_key in train_set], train_in_sample
        ),
    })
    for split, families in splits.items():
        split_rows = [row for row in rows if row.family_key in set(families)]
        case_keys = {row.case_key for row in split_rows}
        allowances = {case_key: allowance for case_key in case_keys}
        for name, predictions, margin in (
            ("neural", neural_map, crossfit.neural_upper_margin),
            ("ridge", ridge_map, crossfit.ridge_upper_margin),
        ):
            measured = evaluate_action_policy(
                split_rows, predictions, current_forward_sha256=forward_sha,
                fixed_role_log_limits=limits,
                absolute_allowance_by_case=allowances,
                empirical_margin=margin,
                relative_allowance=RELATIVE_ALLOWANCE,
            )
            result["split_results"][split][name] = {
                **_policy_summary(measured, split_rows, metadata["case_metadata"]),
                "ranking": _ranking(split_rows, predictions),
            }
    return result, neural_fit.model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-json", type=Path, required=True)
    parser.add_argument("--feature-npz", type=Path, required=True)
    parser.add_argument("--mask-npz", type=Path, required=True)
    parser.add_argument("--selected-g-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-updates", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2112)
    args = parser.parse_args()
    rows, metadata = _load_action_table(
        args.table_json.resolve(), args.feature_npz.resolve(), args.mask_npz.resolve(),
        args.selected_g_checkpoint.resolve()
    )
    result, model = fit_and_evaluate(rows, metadata, updates=args.fit_updates, seed=args.seed)
    result["input_identity"] = {
        "table_json": str(args.table_json.resolve()),
        "table_sha256": _sha256(args.table_json.resolve()),
        "feature_npz": str(args.feature_npz.resolve()),
        "feature_npz_sha256": _sha256(args.feature_npz.resolve()),
        "mask_npz": str(args.mask_npz.resolve()),
        "mask_npz_sha256": _sha256(args.mask_npz.resolve()),
        "selected_g_checkpoint": str(args.selected_g_checkpoint.resolve()),
        "selected_g_checkpoint_sha256": _sha256(args.selected_g_checkpoint.resolve()),
        "driver_sha256": _sha256(Path(__file__).resolve()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if model is not None:
        model_path = args.output_dir / "wind_action_risk_head.pt"
        torch.save({"format_version": 1, "forward_checkpoint_sha256": metadata["checkpoint_sha256"],
                    "role_order": list(metadata["roles"]), "fit_updates": args.fit_updates,
                    "train_families": list(metadata["split_families"]["train_fit"]),
                    "state_dict": model.state_dict()}, model_path)
        result["risk_head_checkpoint"] = str(model_path.resolve())
        result["risk_head_checkpoint_sha256"] = _sha256(model_path)
    _write_json(args.output_dir / "wind_action_selector_results.json", result)
    print(json.dumps({"selector_fit": result["selector_fit"],
                      "optimizer_calls": result["optimizer_calls"],
                      "splits": {key: value["case_count"] for key, value in result["split_results"].items()},
                      "output": str(args.output_dir / "wind_action_selector_results.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
