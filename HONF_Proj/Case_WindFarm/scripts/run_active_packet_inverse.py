"""Run bounded frozen-interface WindFarm position completion for Run 2111.

The Stage-C forward checkpoint is selected before this program opens any
inverse target or development outcome. Generated layouts have no new CFD
reference; their candidate-field checks are explicitly surrogate predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_inverse_core.models.frozen_packet_diffusion import ConditionalPacketDenoiser
from honf_runtime.compat import load_trusted_checkpoint
from run_active_packet_reuse import (
    DEFAULT_CONFIG,
    EXPECTED_SOURCE_SHA256,
    NativeRoleCatalogueCache,
    _batch_from_sample,
    _build_frozen_utility_organizer,
    _load_config,
    _load_source,
    _native_inputs,
    _native_rng,
    _new_model_from_source,
    _rows_sha256,
    _write_json,
    sample_native_role_queries,
)

from windfarm.inverse.packet_completion import (
    WindCandidateInterfaceBuilder,
    WindCandidatePacketProvider,
    evaluate_matched_wind_completion,
    hidden_set_error_D,
    make_wind_completion_task,
    train_matched_wind_diffusion,
    wind_surrogate_predictor,
)


def _checkpoint_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_forward_selection(selection_path: Path, checkpoint_path: Path) -> tuple[dict[str, Any], str]:
    """Bind inverse fitting to a recorded forward-only version decision."""

    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selected, dict) or selected.get("inverse_outcomes_consulted") is not False:
        raise ValueError("Forward selection must explicitly precede inverse outcomes")
    if Path(selected["selected_checkpoint_path"]).resolve() != checkpoint_path:
        raise ValueError("Inverse checkpoint differs from the frozen forward selection")
    if selected["selected_checkpoint_sha256"] != _checkpoint_sha256(checkpoint_path):
        raise ValueError("Selected forward checkpoint changed after the freeze decision")
    evidence_path = Path(selected["forward_evidence_path"]).resolve()
    if selected["forward_evidence_sha256"] != _checkpoint_sha256(evidence_path):
        raise ValueError("Forward-only selection evidence changed after the freeze decision")
    if not str(selected.get("selection_rule", "")).strip():
        raise ValueError("Forward selection must record its measured decision rule")
    timestamp = float(selected["selection_time_unix"])
    if not np.isfinite(timestamp) or timestamp <= 0:
        raise ValueError("Forward selection needs a finite positive timestamp")
    if evidence_path.stat().st_mtime > timestamp + 1.0:
        raise ValueError("Forward evidence was modified after the recorded freeze decision")
    return selected, _checkpoint_sha256(selection_path)


def _development_rows(view: Any, validation_rows: np.ndarray, count: int) -> list[int]:
    """Select distinct layouts across turbine-count strata without fields."""

    cases = [view.run(int(row)) for row in validation_rows]
    cases.sort(key=lambda case: (case.n_turbines, case.layout_index, case.index))
    if count > len({int(case.layout_index) for case in cases}):
        raise ValueError("Insufficient distinct development layouts")
    targets = np.linspace(0, len(cases) - 1, count)
    chosen: list[int] = []
    used_layouts: set[int] = set()
    for target in targets:
        ranked = sorted(
            enumerate(cases),
            key=lambda pair: (abs(pair[0] - target), pair[1].layout_index, pair[1].index),
        )
        selected = next(case for _, case in ranked if int(case.layout_index) not in used_layouts)
        chosen.append(int(selected.index))
        used_layouts.add(int(selected.layout_index))
    return chosen


def _sample_summary(rows: list[dict[str, object]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    observed_by_draw: dict[tuple[str, int, int], dict[str, object]] = {}
    for row in rows:
        groups[(str(row["control"]), str(row["arm"]))].append(row)
        if row["control"] == "as_observed":
            observed_by_draw[(str(row["arm"]), int(row["row_index"]), int(row["sample_index"]))] = row
    summary: dict[str, Any] = {}
    for (control, arm), group in sorted(groups.items()):
        acceptable = [row for row in group if bool(row["acceptable"])]
        per_task: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in acceptable:
            per_task[int(row["row_index"])].append(row)
        diversity: list[float] = []
        for task_rows in per_task.values():
            for first, second in combinations(task_rows, 2):
                diversity.append(
                    hidden_set_error_D(
                        np.asarray(first["generated_centers_D"]),
                        np.asarray(second["generated_centers_D"]),
                        np.asarray(first["visible_mask"], dtype=bool),
                    )
                )
        def mean(name: str, records: list[dict[str, object]] = group) -> float:
            return float(np.mean([float(row[name]) for row in records]))

        paired_shifts = []
        if control != "as_observed":
            for row in group:
                baseline = observed_by_draw[(arm, int(row["row_index"]), int(row["sample_index"]))]
                paired_shifts.append(hidden_set_error_D(
                    np.asarray(row["generated_centers_D"]),
                    np.asarray(baseline["generated_centers_D"]),
                    np.asarray(row["visible_mask"], dtype=bool),
                ))

        summary[f"{control}/{arm}"] = {
            "attempts": len(group),
            "valid_geometry_fraction_before_repair": float(np.mean([
                bool(row["geometry"]["finite"])
                and bool(row["geometry"]["inside_design_box"])
                and bool(row["geometry"]["inside_native_domain"])
                and bool(row["geometry"]["rotors_nonoverlap"])
                for row in group
            ])),
            "valid_geometry_fraction_after_repair": None,
            "repair_or_rejection_used": False,
            "acceptable_fraction_surrogate_threshold": len(acceptable) / len(group),
            "hidden_set_error_D_mean": mean("hidden_set_error_D"),
            "surrogate_observed_rmse_mps_mean": mean("surrogate_observed_rmse_mps"),
            "surrogate_held_rmse_mps_mean": mean("surrogate_held_rmse_mps"),
            "paired_hidden_position_shift_vs_as_observed_D_mean": (
                None if not paired_shifts else float(np.mean(paired_shifts))
            ),
            "independent_surrogate_disagreement_rmse_mps_mean": (
                None
                if group[0]["independent_surrogate_disagreement_rmse_mps"] is None
                else mean("independent_surrogate_disagreement_rmse_mps")
            ),
            "acceptable_hidden_set_diversity_D_mean": (
                None if not diversity else float(np.mean(diversity))
            ),
            "acceptable_diversity_pairs": len(diversity),
            "fixed_weight_packet_rewire_swaps": sum(
                sum(int(value) for value in (row["fixed_weight_packet_rewire_swaps"] or {}).values())
                for row in group
            ),
            "samples_with_effective_packet_rewire": sum(
                bool(row["fixed_weight_packet_rewire_swaps"])
                and any(int(value) > 0 for value in row["fixed_weight_packet_rewire_swaps"].values())
                for row in group
            ),
            "organizer_calls": sum(int(row["organizer_calls"]) for row in group),
            "candidate_full_forward_calls": sum(int(row["full_forward_calls"]) for row in group),
            "complete_candidate_seconds": sum(float(row["complete_candidate_seconds"]) for row in group),
        }
    return summary


def run(args: argparse.Namespace) -> Path:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "0":
        raise RuntimeError("Wind inverse must run with physical GPU 0 only")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA RTX 6000 Ada Generation":
        raise RuntimeError("Authorized physical GPU 0 did not resolve to the expected device")
    device = torch.device("cuda:0")
    stage_c_path = args.stage_c_checkpoint.resolve()
    selected_forward, selection_sha = _verify_forward_selection(
        args.forward_selection.resolve(), stage_c_path
    )
    config = _load_config(args.config)
    view, split, train_rows, split_record = _native_inputs(config)
    _, source, normalizer, source_sha = _load_source(config)
    stage_c = load_trusted_checkpoint(stage_c_path, map_location="cpu")
    if not isinstance(stage_c, dict):
        raise TypeError("Frozen Stage-C checkpoint must be a mapping")
    if stage_c.get("source_sha256") != source_sha or source_sha != EXPECTED_SOURCE_SHA256:
        raise ValueError("Frozen Wind checkpoint source identity differs from the approved W-full u1500")
    if stage_c.get("train_rows_sha256") != _rows_sha256(train_rows):
        raise ValueError("Frozen Wind checkpoint training-layout exclusion record differs")
    if int(stage_c.get("frontier_role_count", -1)) != 5:
        raise ValueError("Wind Stage C must carry exactly five physical-role utility outputs")
    if not isinstance(stage_c.get("model_state_dict"), dict) or not isinstance(
        stage_c.get("organizer_state_dict"), dict
    ):
        raise TypeError("Frozen Stage C checkpoint lacks physical or organizer tensors")
    route = str(stage_c.get("secondary_route", "")).upper()
    budget = float(stage_c.get("budget_fraction", float("nan")))
    tolerance = tuple(float(value) for value in stage_c.get("role_tolerance", ()))
    if route not in {"MM", "EM"} or budget not in {0.9, 0.75}:
        raise ValueError("Frozen Wind checkpoint route/budget is outside the declared experiment")
    if len(tolerance) != 5 or not all(np.isfinite(tolerance)):
        raise ValueError("Frozen Wind checkpoint lacks five dimensionless role tolerances")

    preflight_case = view.run(int(train_rows[0]))
    counts = config["forward"]["stage_a"]["role_query_counts"]
    preflight_sample = sample_native_role_queries(
        preflight_case,
        _native_rng(int(config["seed"]), 0, int(train_rows[0]), 710),
        counts,
        catalogue_cache=NativeRoleCatalogueCache(),
    )
    preflight_batch = _batch_from_sample(preflight_case, preflight_sample, normalizer, device)
    model = _new_model_from_source(source, normalizer, preflight_batch, device)
    model.load_state_dict(stage_c["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    model.core.backend.set_cover_mode("external")
    model.core.backend.set_cover_executor("dense_masked")
    retained_w_full = _new_model_from_source(source, normalizer, preflight_batch, device)
    retained_w_full.eval().requires_grad_(False)
    with torch.no_grad():
        encoded = model.core.encode_case(preflight_batch)
        tree = model.core.backend.build_case_trees(encoded)[0]
    organizer = _build_frozen_utility_organizer(
        model.core,
        encoded,
        tree,
        stage_c["organizer_state_dict"],
        forward_config=config["forward"],
        device=device,
        frontier_role_count=5,
    )
    version = _checkpoint_sha256(stage_c_path)
    builder = WindCandidateInterfaceBuilder(
        model,
        organizer,
        budget_fractions={"QE": budget, route: budget},
        role_tolerance=tolerance,
        numerical_state_version=version,
    )
    provider_factory = lambda known: WindCandidatePacketProvider(known, builder)
    inverse_dir = stage_c_path.parent / "inverse_wind_position_completion"
    inverse_dir.mkdir(parents=True, exist_ok=True)
    development_rows = _development_rows(
        view,
        np.asarray(split.validation, dtype=np.int64),
        int(config["inverse"]["development_task_count"]),
    )
    train_tasks = [
        make_wind_completion_task(
            view.run(int(row)), partition="train", hidden_count=1, seed=211100 + int(row)
        )
        for row in train_rows
    ]
    development_tasks = [
        make_wind_completion_task(
            view.run(int(row)), partition="development", hidden_count=1,
            seed=211200 + int(row),
        )
        for row in development_rows
    ]
    _write_json(inverse_dir / "task_contract.json", {
        "forward_selection_record": str(args.forward_selection.resolve()),
        "forward_selection_record_sha256": selection_sha,
        "forward_selection_rule": selected_forward["selection_rule"],
        "forward_evidence_path": selected_forward["forward_evidence_path"],
        "forward_evidence_sha256": selected_forward["forward_evidence_sha256"],
        "frozen_stage_c_checkpoint": str(stage_c_path),
        "frozen_stage_c_sha256": version,
        "source_sha256": source_sha,
        "train_rows_sha256": split_record["student_train_rows_sha256"],
        "train_task_rows": list(map(int, train_rows)),
        "development_rows_selected_before_inverse_outcomes": development_rows,
        "development_layouts": [int(view.run(row).layout_index) for row in development_rows],
        "design_hidden_count": 1,
        "fixed_sensor_count": 24,
        "observed_sensor_count": 16,
        "held_sensor_count": 8,
        "rotor_radius_D": float(preflight_case.module_features[0, 0]),
        "max_native_sensor_snap_D": float(max(task.sensors.snap_distance_D.max() for task in development_tasks)),
        "physical_gpu": 0,
        "logical_cuda_device": 0,
        "reference_source": "stored OpenFOAM clean-layout velocities; no generated-layout CFD",
    })
    torch.manual_seed(int(args.seed))
    denoiser = ConditionalPacketDenoiser(
        design_dim=2,
        module_dim=16,
        sensor_dim=6,
        embedding_dim=int(organizer.node_encoder[0].out_features),
        hidden_dim=96,
        layers=3,
    )
    started = time.monotonic()
    attempt_log_path = inverse_dir / "inverse_optimizer_attempts.jsonl"
    if args.evaluate_only:
        completed = load_trusted_checkpoint(args.resume, map_location="cpu")
        if not isinstance(completed, dict) or int(completed.get("update_count", -1)) != int(args.updates):
            raise ValueError("Evaluation-only resume must use the completed requested update checkpoint")
    matched = train_matched_wind_diffusion(
        train_tasks,
        denoiser_template=denoiser,
        provider_factory=provider_factory,
        frozen_modules={"forward": model, "organizer": organizer},
        updates=int(args.updates),
        steps=int(config["inverse"]["reverse_steps"]),
        seed=int(args.seed),
        device=device,
        checkpoint_dir=inverse_dir / "checkpoints",
        checkpoint_every=50,
        resume_checkpoint=args.resume,
        attempt_log_path=attempt_log_path,
    )
    attempt_rows = [json.loads(line) for line in attempt_log_path.read_text(encoding="utf-8").splitlines()]
    attempt_counts = {
        arm: sum(row["arm"] == arm for row in attempt_rows)
        for arm in ("I-G", "I-dense")
    }
    training_summary = {
        "updates_per_arm": matched.updates_per_arm,
        "optimizer_attempts_by_arm_including_failed_invocations": attempt_counts,
        "optimizer_attempt_log": str(attempt_log_path),
        "organizer_calls_shared_by_matched_noisy_state": matched.organizer_calls,
        "initial_denoiser_sha256": matched.initial_denoiser_hash,
        "frozen_state_hashes_before": matched.frozen_state_hashes_before,
        "frozen_state_hashes_after": matched.frozen_state_hashes_after,
        "graph_loss_final": matched.graph_losses[-1],
        "dense_loss_final": matched.dense_losses[-1],
        "graph_loss_history": matched.graph_losses,
        "dense_loss_history": matched.dense_losses,
        "elapsed_seconds": time.monotonic() - started,
    }
    if not args.evaluate_only:
        _write_json(inverse_dir / "training_summary.json", training_summary)
    if args.training_only:
        return inverse_dir
    rows = evaluate_matched_wind_completion(
        development_tasks,
        matched=matched,
        provider_factory=provider_factory,
        surrogate_predictor=wind_surrogate_predictor(model),
        independent_surrogate_predictor=wind_surrogate_predictor(retained_w_full),
        seed=int(args.seed) + 1,
        include_rewired=True,
        device=device,
    )
    with (inverse_dir / "sample_attempts.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    _write_json(inverse_dir / "sample_summary.json", {
        "groups": _sample_summary(rows),
        "development_task_count": len(development_tasks),
        "samples_per_task_per_arm_per_control": 8,
        "observation_slack_mps": 0.5,
        "stored_reference_scope": "clean-layout observed/held sensors only",
        "generated_candidate_scope": "geometry and frozen same-student surrogate only",
        "test_split_opened": False,
    })
    return inverse_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--forward-selection", type=Path, required=True)
    parser.add_argument("--stage-c-checkpoint", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=21119)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true", help="Load a completed inverse checkpoint and run development sampling without optimizer updates")
    arguments = parser.parse_args()
    if not 1 <= arguments.updates <= 800:
        parser.error("--updates must be between 1 and 800 per arm")
    if arguments.evaluate_only and (arguments.resume is None or arguments.training_only):
        parser.error("--evaluate-only requires --resume and cannot be combined with --training-only")
    print(run(arguments), flush=True)


if __name__ == "__main__":
    main()
