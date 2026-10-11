"""Reusable fixed-panel diagnosis for existing HONF providers and TrainingEngine.

The context-factory contract is ``factory(config, checkpoint_payload)`` and
returns ``(model, provider, engine)``. It must construct an existing model and
provider without training or refitting calibration. The CLI restores the
checkpoint's model, provider-training, optimizer, and sampling metadata, then
asks the existing provider to build native batches from an explicit TRAIN-only
panel manifest. Counterfactual mode performs exactly one cloned TrainingEngine
update per declared intervention; it never mutates the source checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel" / "src",
              PROJECT_ROOT / "Case_WindFarm" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from honf_runtime.compat import load_trusted_checkpoint  # noqa: E402
from honf_runtime.run_layout import RunLayout  # noqa: E402
from honf_runtime.unified_training import (  # noqa: E402
    LossTerm,
    SamplingKey,
    _diagnostic_age_branch_amendment,
    _engine_config_payload,
    _optimizer_spec_payload,
    _sampling_dataset_id,
    capture_rng_state,
    named_optimizer_state,
    restore_rng_state,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _diagnostic_result_checkpoint(output: Path, completed_epoch: int) -> tuple[Path, str]:
    """Resolve and hash the exact review checkpoint written by a fit stage."""

    checkpoint = RunLayout(output).read_path(f"epoch_{int(completed_epoch)}_model.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"TrainingEngine completed epoch {completed_epoch}, but its review checkpoint is missing: {checkpoint}"
        )
    return checkpoint, _sha256(checkpoint)


def _validate_diagnostic_fit_output(output: Path, resume_checkpoint: Path | None) -> Path:
    """Require a fresh directory for a branch and the exact latest run for resume."""

    output = output.expanduser().resolve()
    if resume_checkpoint is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("A diagnostic fit requires a fresh child run directory.")
        return output
    resume_checkpoint = resume_checkpoint.expanduser().resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"Diagnostic resume run directory does not exist: {output}")
    latest = RunLayout(output).read_path("latest_model.pt")
    if not latest.is_file() or latest.resolve() != resume_checkpoint:
        raise ValueError("Diagnostic resume must use this child run's exact latest checkpoint.")
    return output


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


class _DiagnosticObjectiveProvider:
    """Apply a declared objective override through an existing task provider.

    The wrapper delegates native sampling, prediction, optimizer groups,
    validation metrics, and provider state to the original adapter. It changes
    only the declared LossTerm weights and the fixed TRAIN case inventory.
    """

    def __init__(self, base: Any, case_keys: list[Any], loss_weight_overrides: Mapping[str, Any]):
        self._base = base
        self._case_keys = list(case_keys)
        if not self._case_keys or len({_case_token(value) for value in self._case_keys}) != len(self._case_keys):
            raise ValueError("Diagnostic fitting case keys must be nonempty and unique.")
        self._allowed_case_tokens = {_case_token(value) for value in self._case_keys}
        self._loss_weight_overrides = dict(loss_weight_overrides)
        metadata_fn = getattr(base, "loss_metadata", None)
        metadata = metadata_fn() if callable(metadata_fn) else None
        if not isinstance(metadata, Mapping):
            raise TypeError("Diagnostic objectives require the provider's declared loss_metadata mapping.")
        if set(self._loss_weight_overrides) - set(metadata):
            raise ValueError("A diagnostic loss override names no declared provider loss term.")
        for name, value in self._loss_weight_overrides.items():
            if (not isinstance(name, str) or not name or type(value) not in (int, float)
                    or not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError("Diagnostic loss weights must be finite nonnegative scalars.")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    @property
    def model(self) -> Any:
        """Keep provider identity checks bound to the actual diagnostic model."""
        return self._base.model

    @model.setter
    def model(self, value: Any) -> None:
        # Disposable update/probe APIs temporarily bind a cloned model and
        # restore the original in finally. Writing through prevents a wrapper
        # attribute from shadowing the native provider's model reference.
        self._base.model = value

    def identity_payload(self) -> Mapping[str, Any]:
        identity = copy.deepcopy(dict(self._base.identity_payload()))
        if "diagnostic" in identity:
            raise ValueError("A provider cannot be wrapped with nested diagnostic identities.")
        identity["diagnostic"] = {
            "fit_case_keys": copy.deepcopy(self._case_keys),
            "loss_weight_overrides": copy.deepcopy(self._loss_weight_overrides),
        }
        return identity

    def epoch_cases(self, epoch: int, seed: int) -> list[Any]:
        original = list(self._base.epoch_cases(epoch, seed))
        filtered = [value for value in original if _case_token(value) in self._allowed_case_tokens]
        if ({_case_token(value) for value in filtered} != self._allowed_case_tokens
                or len(filtered) != len(self._case_keys)):
            raise ValueError("Diagnostic fitting inventory left the provider's current TRAIN exposure.")
        return filtered

    def _override_terms(self, terms: Mapping[str, LossTerm]) -> dict[str, LossTerm]:
        # Providers may emit response or PDE LossTerms only for microbatches
        # containing their eligible TRAIN rows. Constructor validation above
        # rejects undeclared override names; absence from an individual native
        # call is a valid zero contribution, not a missing objective binding.
        return {
            name: LossTerm(term.numerator, term.denominator,
                           float(self._loss_weight_overrides.get(name, term.weight)))
            for name, term in terms.items()
        }

    def loss_terms(self, predictions: Any, targets: Any, phase: str, auxiliary_state: Any):
        terms = self._base.loss_terms(predictions, targets, phase, auxiliary_state)
        return self._override_terms(terms)

    def validation_loss_terms(self, predictions: Any, targets: Any, auxiliary_state: Any,
                              *, batch: Any, arm: str):
        function = getattr(self._base, "validation_loss_terms", None)
        if not callable(function):
            return {}
        return self._override_terms(function(predictions, targets, auxiliary_state, batch=batch, arm=arm))

    def loss_metadata(self) -> Mapping[str, Mapping[str, str]]:
        metadata = copy.deepcopy(dict(self._base.loss_metadata()))
        for name, weight in self._loss_weight_overrides.items():
            item = dict(metadata[name])
            item["weight"] = f"diagnostic objective override: {float(weight):.9g}; term retained"
            metadata[name] = item
        return metadata


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"Refusing to replace an existing different receipt: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _factory(spec: str):
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("Context factory must use the form 'python.module:callable'.")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"Context factory {spec!r} is not callable.")
    return function


def _load_context(args: argparse.Namespace):
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha256 = _sha256(checkpoint)
    payload = load_trusted_checkpoint(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state_dict"), Mapping):
        raise TypeError("Checkpoint does not contain a model_state_dict mapping.")
    source_identity = payload.get("experiment_identity")
    if not isinstance(source_identity, Mapping):
        raise TypeError("Checkpoint lacks an experiment_identity mapping.")
    expected_provider_identity = source_identity.get("provider_identity")
    if not isinstance(expected_provider_identity, Mapping):
        raise TypeError("Checkpoint lacks its sealed provider identity.")
    diagnostic = expected_provider_identity.get("diagnostic")
    factory_payload = payload
    if diagnostic is not None:
        if (not isinstance(diagnostic, Mapping)
                or set(diagnostic) != {"fit_case_keys", "loss_weight_overrides"}
                or not isinstance(diagnostic["fit_case_keys"], list)
                or not isinstance(diagnostic["loss_weight_overrides"], Mapping)):
            raise ValueError("Checkpoint diagnostic provider identity is malformed or contains undeclared controls.")
        factory_identity = copy.deepcopy(dict(source_identity))
        factory_provider_identity = copy.deepcopy(dict(expected_provider_identity))
        factory_provider_identity.pop("diagnostic")
        factory_identity["provider_identity"] = factory_provider_identity
        factory_payload = dict(payload)
        factory_payload["experiment_identity"] = factory_identity
    factory_config = _read_json(args.factory_config.expanduser().resolve())
    factory_config = {
        **factory_config,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
    }
    context = _factory(args.context_factory)(factory_config, factory_payload)
    if not isinstance(context, tuple) or len(context) not in (3, 4):
        raise TypeError("Context factory must return (model, provider, engine) and may add data-access provenance.")
    model, provider, engine = context[:3]
    context_provenance = context[3] if len(context) == 4 else {
        "declaration": "context factory supplied no data-access provenance"
    }
    if not isinstance(context_provenance, Mapping):
        raise TypeError("Optional context factory provenance must be a mapping.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    state_loader = getattr(provider, "load_training_state_dict", None)
    if callable(state_loader):
        state_loader(payload.get("provider_training_state", {}))
    elif payload.get("provider_training_state"):
        raise TypeError("Checkpoint has provider training state but the provider cannot restore it.")
    if diagnostic is not None:
        provider = _DiagnosticObjectiveProvider(
            provider, list(diagnostic["fit_case_keys"]), diagnostic["loss_weight_overrides"])
    provider_identity = dict(provider.identity_payload())
    if provider_identity != dict(expected_provider_identity):
        raise ValueError("Factory provider identity differs from the checkpoint's sealed identity.")
    return checkpoint, payload, model, provider, engine, provider_identity, dict(context_provenance)


def _case_token(value: Any) -> tuple[str, str]:
    if isinstance(value, np.generic):
        value = value.item()
    return type(value).__name__, json.dumps(value, sort_keys=True, separators=(",", ":"))


def _make_batches(provider: Any, engine: Any, panel: Mapping[str, Any], *, section: str,
                  phase: str, arm: str, epoch: int, seed: int,
                  sampling_dataset_id: str) -> list[Any]:
    specs = panel.get(section)
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"Panel must declare a nonempty {section} list.")
    sampling_version = str(engine.config.sampling_version)
    allowed_cases = {_case_token(value) for value in provider.epoch_cases(epoch, seed)}
    batches = []
    for item in specs:
        if not isinstance(item, Mapping) or not isinstance(item.get("case_keys"), list) or not item["case_keys"]:
            raise ValueError(f"Every {section} entry must declare nonempty ordered case_keys.")
        case_keys = list(item["case_keys"])
        if any(_case_token(value) not in allowed_cases for value in case_keys):
            raise ValueError(f"{section} contains a case outside provider.epoch_cases() TRAIN membership.")
        key = SamplingKey(
            seed=seed,
            epoch=epoch,
            update_index=int(item.get("update_index", 0)),
            microbatch_index=int(item.get("microbatch_index", 0)),
            stage=str(item.get("stage", phase)),
            arm=str(item.get("arm", arm)),
            sampling_version=sampling_version,
            dataset_id=sampling_dataset_id,
        )
        batch = provider.make_batch(case_keys, key)
        if tuple(batch.case_keys) and list(batch.case_keys) != case_keys:
            raise ValueError("Provider changed the declared ordered native case keys.")
        if not tuple(batch.case_keys):
            from honf_runtime.unified_training import TaskBatch

            batch = TaskBatch(batch.scene_inputs, batch.receivers, batch.targets, batch.auxiliary, tuple(case_keys))
        batches.append(batch)
    return batches


def _measure_native_metrics(
    model: Any,
    provider: Any,
    engine: Any,
    batches: list[Any],
    *,
    arm: str,
    phase: str,
    epoch: int,
    maximum_queries_per_role: int,
) -> dict[str, Any]:
    if not batches:
        raise ValueError("A fixed native-metrics panel requires at least one batch.")
    metric_fn = getattr(provider, "validation_metrics", None)
    reducer = getattr(provider, "reduce_native_metrics", None)
    if not callable(metric_fn) or not callable(reducer):
        raise TypeError("Native physical diagnosis requires the existing provider validation_metrics/reducer methods.")
    case_keys = [key for batch in batches for key in batch.case_keys]
    if not case_keys or len({_case_token(key) for key in case_keys}) != len(case_keys):
        raise ValueError("Native metric batches must visit unique TRAIN case keys exactly once.")
    state_reader = getattr(provider, "training_state_dict", None)
    state_loader = getattr(provider, "load_training_state_dict", None)
    provider_state = copy.deepcopy(state_reader()) if callable(state_reader) else None
    rng_state = capture_rng_state()
    previous_mode = model.training
    records: list[Mapping[str, Any]] = []
    samples: list[Mapping[str, Any]] = []
    work_records: list[Mapping[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                scene = provider.make_scene(batch.scene_inputs)
                predictions, auxiliary_state = provider.predict_native(
                    model,
                    scene,
                    batch.receivers,
                    arm,
                    phase,
                    epoch=epoch,
                    temperature=engine.config.temperature_for_epoch(epoch),
                )
                records.append(metric_fn(predictions, batch.targets, auxiliary_state))
                sample_fn = getattr(provider, "diagnostic_native_sample", None)
                if callable(sample_fn):
                    samples.append(sample_fn(
                        predictions,
                        batch,
                        maximum_queries_per_role=maximum_queries_per_role,
                    ))
                work_records.append(provider.work_counts(batch, predictions, auxiliary_state))
                auxiliary = batch.auxiliary if isinstance(batch.auxiliary, Mapping) else {}
                batch_records.append({
                    "case_keys": list(batch.case_keys),
                    "query_sampling_sha256": auxiliary.get("query_sampling_sha256"),
                    "all_fine_main_full_same_tensor_values": bool(
                        getattr(predictions, "main_mps", None) is not None
                        and getattr(predictions, "full_mps", None) is not None
                        and torch.equal(predictions.main_mps, predictions.full_mps)
                    ),
                })
        parameters = inspect.signature(reducer).parameters
        reducer_kwargs = {}
        if "expected_row_count" in parameters:
            reducer_kwargs["expected_row_count"] = len(case_keys)
        if "expected_case_ids" in parameters:
            reducer_kwargs["expected_case_ids"] = [str(key) for key in case_keys]
        reduced = reducer(records, **reducer_kwargs)
    finally:
        model.train(previous_mode)
        try:
            if provider_state is not None and callable(state_loader):
                state_loader(provider_state)
        finally:
            restore_rng_state(rng_state)
    total_work: dict[str, float] = {}
    for row in work_records:
        for name, value in row.items():
            total_work[name] = total_work.get(name, 0.0) + float(value)
    return {
        "case_keys": case_keys,
        "metric_execution_mode": arm,
        "metric_route_phase": phase,
        "physical_metrics": reduced,
        "native_amplitude_and_residual_samples": samples,
        "work_counts": total_work,
        "batch_records": batch_records,
        "optimizer_updates": 0,
    }


def _run_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint, payload, model, provider, engine, provider_identity, context_provenance = _load_context(args)
    panel_path = args.panel.expanduser().resolve()
    panel = _read_json(panel_path)
    epoch = panel.get("epoch")
    seed = engine.config.seed
    phase = panel.get("phase")
    arm = panel.get("arm")
    if type(epoch) is not int or epoch < 1 or not isinstance(phase, str) or not isinstance(arm, str):
        raise ValueError("Panel requires positive integer epoch and named phase/arm values.")
    source_epoch = payload.get("epoch")
    if type(source_epoch) is not int or source_epoch < 1:
        raise ValueError("Checkpoint epoch is not a positive integer.")
    if panel.get("source_epoch", source_epoch) != source_epoch:
        raise ValueError("Panel source_epoch does not match the checkpoint age.")
    sampling_dataset_id = (
        _sampling_dataset_id(provider_identity)
        if engine.config.sampling_version == SamplingKey.CASE_EPOCH_VERSION else ""
    )
    batch_section = "metric_batches" if args.command == "native-metrics" else "update_batches"
    update_batches = _make_batches(
        provider, engine, panel, section=batch_section, phase=phase, arm=arm,
        epoch=epoch, seed=seed, sampling_dataset_id=sampling_dataset_id,
    )
    started = time.perf_counter()
    if args.command == "native-metrics":
        result = _measure_native_metrics(
            model,
            provider,
            engine,
            update_batches,
            arm=arm,
            phase=phase,
            epoch=epoch,
            maximum_queries_per_role=int(panel.get("maximum_queries_per_role", 8)),
        )
    elif args.command == "loss":
        result = engine.measure_macro_loss(
            model, provider, update_batches, phase=phase, arm=arm, epoch=epoch,
            training_mode=bool(panel.get("training_mode", True)),
        )
    elif args.command == "gradients":
        result = engine.diagnose_macro_update(
            model, provider, update_batches, phase=phase, arm=arm, epoch=epoch,
            term_groups=panel.get("gradient_groups"),
            training_mode=bool(panel.get("training_mode", True)),
            optimizer_state_by_name=payload.get("optimizer_state_by_name"),
        )
    elif args.command == "counterfactual-updates":
        probe_batches = _make_batches(
            provider, engine, panel, section="probe_batches", phase=phase, arm=arm,
            epoch=epoch, seed=seed, sampling_dataset_id=sampling_dataset_id,
        )
        interventions = panel.get("interventions")
        if not isinstance(interventions, Mapping):
            raise ValueError("Counterfactual panel must include an interventions mapping.")
        process_rng_state = capture_rng_state()
        saved_rng_state = payload.get("rng_state")
        try:
            if isinstance(saved_rng_state, Mapping):
                restore_rng_state(saved_rng_state)
            result = engine.compare_disposable_updates(
                model, provider, update_batches, probe_batches,
                phase=phase, arm=arm, epoch=epoch,
                optimizer_state_by_name=payload.get("optimizer_state_by_name"),
                interventions=interventions,
                declared_optimizer_updates=args.declared_optimizer_updates,
                dv_term=panel.get("dv_term"),
            )
        finally:
            restore_rng_state(process_rng_state)
    else:
        raise ValueError(f"Unsupported diagnosis command {args.command!r}.")
    explicit_batches = list(update_batches)
    if args.command == "counterfactual-updates":
        explicit_batches.extend(probe_batches)
    explicit_case_keys = [
        key
        for batch in explicit_batches
        for key in batch.case_keys
    ]
    return {
        "schema_version": 1,
        "diagnosis": args.command,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_epoch": source_epoch,
        "epoch_for_schedule_and_sampler": epoch,
        "phase": phase,
        "arm": arm,
        "provider_identity": provider_identity,
        "panel_path": str(panel_path),
        "panel_sha256": _sha256(panel_path),
        "context_factory_provenance": context_provenance,
        "optimizer_rng_state_source": (
            "checkpoint" if args.command == "counterfactual-updates" and isinstance(payload.get("rng_state"), Mapping)
            else "process_state_for_disposable_probe"
        ),
        "explicit_batch_case_keys": explicit_case_keys,
        "provider_membership_check": "provider.epoch_cases() membership check; no target access is implied by this check",
        "target_access_claim": (
            "explicit native batches listed above; context construction access is reported separately in context_factory_provenance"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_updates": int(result.get("optimizer_updates", 0)),
        "result": result,
    }


def _run_diagnostic_fit_branch(args: argparse.Namespace) -> dict[str, Any]:
    """Preflight or run a bounded diagnostic child through TrainingEngine.fit."""
    plan_path = args.fit_plan.expanduser().resolve()
    plan = _read_json(plan_path)
    if plan.get("schema_version") != 1 or plan.get("kind") != "reference_anchored_objective_diagnostic":
        raise ValueError("Fit plan must be a version-1 reference-anchored objective diagnostic.")
    checkpoint, payload, model, base_provider, engine, source_provider_identity, context_provenance = _load_context(args)
    source_sha256 = _sha256(checkpoint)
    source_identity = payload.get("experiment_identity")
    if not isinstance(source_identity, Mapping):
        raise TypeError("Diagnostic source checkpoint has no experiment identity.")
    if plan.get("source_checkpoint_sha256") != source_sha256:
        raise ValueError("Fit plan source checkpoint hash differs from the immutable checkpoint.")
    if (plan.get("source_epoch") != payload.get("epoch")
            or plan.get("source_arm") != payload.get("arm")):
        raise ValueError("Fit plan source age or arm differs from the checkpoint.")
    arm = plan.get("arm")
    source_epoch = int(plan["source_epoch"])
    fit_case_keys = plan.get("fit_case_keys")
    overrides = plan.get("loss_weight_overrides")
    reviews = plan.get("review_epochs")
    if (not isinstance(arm, str) or not isinstance(fit_case_keys, list) or not fit_case_keys
            or not isinstance(overrides, Mapping) or not isinstance(reviews, list) or not reviews):
        raise TypeError("Fit plan lacks arm, fixed case keys, overrides, or review epochs.")
    if source_provider_identity != dict(source_identity.get("provider_identity", {})):
        raise ValueError("Rebuilt source provider identity no longer equals the checkpoint binding.")
    case_tokens = [_case_token(value) for value in fit_case_keys]
    if len(set(case_tokens)) != len(case_tokens):
        raise ValueError("Diagnostic fit plan case keys must be unique.")
    provider = _DiagnosticObjectiveProvider(base_provider, fit_case_keys, overrides)
    source_sampling_id = _sampling_dataset_id(dict(source_identity["provider_identity"]))
    child_sampling_id = _sampling_dataset_id(dict(provider.identity_payload()))
    saved_sampling_id = source_identity.get("sampling_dataset_id")
    if source_sampling_id != child_sampling_id or source_sampling_id != saved_sampling_id:
        raise ValueError("Source, child, and saved checkpoint sampling_dataset_id values differ.")

    presentation = plan.get("engine_presentation", {})
    allowed_presentation = {
        "monitor_every", "monitor_epochs", "checkpoint_epochs", "latest_every",
        "curve_every", "write_initial_artifacts",
    }
    if not isinstance(presentation, Mapping) or set(presentation) - allowed_presentation:
        raise ValueError("Fit plan may change only declared monitoring/checkpoint presentation controls.")
    config_values = dict(presentation)
    for name in ("monitor_epochs", "checkpoint_epochs"):
        if isinstance(config_values.get(name), list):
            config_values[name] = tuple(config_values[name])
    engine.config = replace(engine.config, **config_values)

    run_label = plan.get("run_label")
    run_identity_label = plan.get("run_identity")
    child_identity = copy.deepcopy(dict(source_identity))
    child_identity["provider_identity"] = dict(provider.identity_payload())
    if run_label is not None:
        child_identity["run"] = str(run_label)
    if run_identity_label is not None:
        child_identity["run_identity"] = str(run_identity_label)

    start_epoch = source_epoch + 1
    phase = engine.config.stage_for_epoch(start_epoch)
    group_specs = tuple(provider.optimizer_groups(model, arm, phase))
    sealed_identity = {
        **child_identity,
        "provider_identity": dict(provider.identity_payload()),
        "engine_config": _engine_config_payload(engine.config),
        "selection_policy": asdict(engine.selection),
        "optimizer_schedule_contract": [_optimizer_spec_payload(spec) for spec in group_specs],
        "sampling_dataset_id": child_sampling_id,
    }
    declaration = {
        "kind": "reference_anchored_objective_diagnostic",
        "source_epoch": source_epoch,
        "source_checkpoint_sha256": source_sha256,
        "review_stop": int(plan.get("review_stop", reviews[-1])),
        "review_epochs": list(reviews),
        "fit_case_keys": copy.deepcopy(fit_case_keys),
        "scientific_update_limit": int(plan["scientific_update_limit"]),
        "evidence_path": str(Path(plan["evidence_path"]).expanduser().resolve()),
        "evidence_sha256": str(plan["evidence_sha256"]),
        "reason": str(plan["reason"]),
        "loss_weight_overrides": dict(overrides),
    }
    stop_after = int(args.stop_after if args.stop_after is not None else declaration["review_stop"])
    if stop_after not in reviews or stop_after <= source_epoch:
        raise ValueError("A diagnostic fit stage must stop at one of its predeclared review epochs.")
    branch_contract = _diagnostic_age_branch_amendment(
        source_identity,
        sealed_identity,
        declaration,
        source_sha256=source_sha256,
        case_keys=provider.epoch_cases(start_epoch, engine.config.seed),
        stop_after=stop_after,
    )

    # Create the real optimizer object with the checkpoint's named moments,
    # then compare the restored state byte-for-byte. This is a zero-step check.
    optimizer, optimizer_specs = engine._make_optimizer(
        model,
        provider,
        arm,
        start_epoch,
        payload.get("optimizer_state_by_name"),
        group_specs=group_specs,
    )
    saved_optimizer = payload.get("optimizer_state_by_name")
    if not isinstance(saved_optimizer, Mapping):
        raise TypeError("Source checkpoint has no named AdamW state.")
    actual_optimizer = named_optimizer_state(model, optimizer)
    if actual_optimizer.get("groups") != saved_optimizer.get("groups"):
        raise ValueError("Diagnostic child optimizer groups differ from the source named-state groups.")
    if actual_optimizer.get("state_by_name", {}).keys() != saved_optimizer.get("state_by_name", {}).keys():
        raise ValueError("Diagnostic child optimizer parameter-state membership differs from the source.")
    for name, saved_slots in saved_optimizer["state_by_name"].items():
        actual_slots = actual_optimizer["state_by_name"][name]
        if set(actual_slots) != set(saved_slots):
            raise ValueError(f"Diagnostic child optimizer state slots changed for parameter {name!r}.")
        for key, expected in saved_slots.items():
            actual = actual_slots[key]
            if torch.is_tensor(expected):
                if not torch.is_tensor(actual) or not torch.equal(actual.cpu(), expected.cpu()):
                    raise ValueError(f"Diagnostic child optimizer moment differs for {name!r}/{key}.")
            elif actual != expected:
                raise ValueError(f"Diagnostic child optimizer scalar state differs for {name!r}/{key}.")
    if [group["group_name"] for group in optimizer.param_groups] != list(payload.get("optimizer_group_names", [])):
        raise ValueError("Diagnostic child optimizer group names differ from the source checkpoint.")
    required_protocol = plan.get("required_engine_protocol", {})
    if not isinstance(required_protocol, Mapping):
        raise TypeError("required_engine_protocol must be a mapping when present.")
    allowed_protocol = {"effective_cases", "microbatch_cases", "sampling_version", "training_mode"}
    if set(required_protocol) - allowed_protocol:
        raise ValueError("required_engine_protocol contains an unsupported binding.")
    actual_protocol = {
        "effective_cases": engine.config.effective_cases,
        "microbatch_cases": engine.config.microbatch_cases,
        "sampling_version": engine.config.sampling_version,
        "training_mode": engine.config.training_mode,
    }
    for name, expected in required_protocol.items():
        if actual_protocol[name] != expected:
            raise ValueError(f"Diagnostic child protocol {name} differs from the explicitly required plan value.")
    if branch_contract["planned_optimizer_updates"] != int(plan["scientific_update_limit"]):
        raise ValueError("Review ages and frozen TRAIN cases do not use exactly the declared update budget.")

    resume_path = None
    resume_sha256 = None
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint.expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Diagnostic continuation checkpoint does not exist: {resume_path}")
        resume_payload = load_trusted_checkpoint(resume_path, map_location="cpu")
        resume_epoch = int(resume_payload.get("epoch", -1))
        if resume_epoch not in reviews or resume_epoch >= stop_after:
            raise ValueError("Diagnostic continuation must advance from a prior declared review epoch.")
        if resume_payload.get("arm") != arm or resume_payload.get("experiment_identity") != sealed_identity:
            raise ValueError("Diagnostic continuation checkpoint identity/arm differs from the sealed child.")
        branch_parent = resume_payload.get("branch_parent")
        branch_declaration = branch_parent.get("diagnostic_age_branch") if isinstance(branch_parent, Mapping) else None
        if not isinstance(branch_declaration, Mapping):
            raise ValueError("Diagnostic continuation checkpoint lacks its preserved parent branch declaration.")
        for name in ("kind", "source_epoch", "source_checkpoint_sha256", "review_stop", "review_epochs",
                     "fit_case_keys", "scientific_update_limit", "evidence_path", "evidence_sha256",
                     "reason", "loss_weight_overrides"):
            if branch_declaration.get(name) != declaration.get(name):
                raise ValueError(f"Diagnostic continuation declaration changed at {name!r}.")
        child_history = resume_payload.get("history", [])
        completed_child_updates = sum(int(row["optimizer_updates"]) for row in child_history)
        expected_child_updates = branch_contract["updates_per_epoch"] * (resume_epoch - source_epoch)
        if completed_child_updates != expected_child_updates:
            raise ValueError("Diagnostic continuation history does not match its absolute review age.")
        resume_sha256 = _sha256(resume_path)

    preflight = {
        "schema_version": 1,
        "diagnosis": "diagnostic_fit_branch_preflight",
        "mode": "execute" if args.execute else "dry_run",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": source_sha256,
        "source_checkpoint_path": str(checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "source_epoch": source_epoch,
        "source_arm": payload["arm"],
        "arm": arm,
        "provider_identity": dict(provider.identity_payload()),
        "sampling_dataset_id": {
            "source": source_sampling_id,
            "child": child_sampling_id,
            "saved_source": saved_sampling_id,
            "all_equal": True,
        },
        "review_epochs": list(reviews),
        "planned_optimizer_updates": branch_contract["planned_optimizer_updates"],
        "requested_stage_stop": stop_after,
        "planned_stage_optimizer_updates": branch_contract["updates_per_epoch"] * (stop_after - (
            int(resume_payload["epoch"]) if args.resume_checkpoint is not None else source_epoch)),
        "updates_per_epoch": branch_contract["updates_per_epoch"],
        "fit_case_count": len(fit_case_keys),
        "fit_case_keys_sha256": hashlib.sha256(json.dumps(
            fit_case_keys, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest(),
        "loss_weight_overrides": dict(overrides),
        "optimizer_group_names": [group["group_name"] for group in optimizer.param_groups],
        "optimizer_named_state_exact_source_match": True,
        "optimizer_updates": 0,
        "branch_guard": branch_contract,
        "context_factory_provenance": context_provenance,
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "elapsed_seconds": time.perf_counter() - args.started_at,
    }
    if resume_path is not None:
        preflight["resume_checkpoint_path"] = str(resume_path)
        preflight["resume_checkpoint_sha256"] = resume_sha256
        preflight["resume_epoch"] = int(resume_payload["epoch"])
        preflight["completed_child_optimizer_updates_before_resume"] = completed_child_updates
    if not args.execute:
        return preflight
    output = _validate_diagnostic_fit_output(args.run_output, resume_path)
    fit_kwargs = {
        "identity": child_identity,
        "arm": arm,
        "stop_after": stop_after,
    }
    if resume_path is None:
        fit_kwargs.update(branch_from_checkpoint=checkpoint, diagnostic_age_branch=declaration)
    else:
        fit_kwargs["resume_checkpoint"] = resume_path
    result = engine.fit(model, provider, output, **fit_kwargs)
    fit_summary_path = RunLayout(output).read_path("fit_summary.json")
    fit_summary = _read_json(fit_summary_path)
    stage_updates = int(fit_summary["optimizer_updates"])
    expected_stage_updates = int(preflight["planned_stage_optimizer_updates"])
    if stage_updates != expected_stage_updates:
        raise ValueError(
            f"TrainingEngine fit_summary reports {stage_updates} updates; expected {expected_stage_updates} for this review stage."
        )
    completed_epoch = int(result["completed_epoch"])
    result_checkpoint, result_checkpoint_sha256 = _diagnostic_result_checkpoint(output, completed_epoch)
    return {
        **preflight,
        "mode": "execute",
        "run_output": str(output),
        "fit_result": result,
        "fit_summary_path": str(fit_summary_path),
        "fit_summary_sha256": _sha256(fit_summary_path),
        "fit_summary_optimizer_updates": stage_updates,
        "fit_summary_cumulative_optimizer_updates": int(fit_summary["cumulative_optimizer_updates"]),
        "optimizer_updates": stage_updates,
        "planned_stage_optimizer_updates": preflight["planned_stage_optimizer_updates"],
        "completed_epoch": completed_epoch,
        # Keep checkpoint_path/checkpoint_sha256 bound to the immutable input
        # checkpoint; identify the trained endpoint separately.
        "result_checkpoint_path": str(result_checkpoint),
        "result_checkpoint_sha256": result_checkpoint_sha256,
    }


def _load_wind_panel_metadata(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read only input metadata, avoiding the native dataset's field mmaps."""
    from windfarm.io import discover_volume_root

    volume_root = discover_volume_root(root.expanduser().resolve())
    arrays = {}
    bindings = {}
    for name in ("layout_index", "wd_deg"):
        path = volume_root / f"{name}.npy"
        arrays[name] = np.asarray(np.load(path, allow_pickle=False))
        bindings[name] = {"path": str(path), "sha256": _sha256(path)}
    count_path = volume_root.parent / "family_tensor" / "n_turbines.npy"
    if count_path.is_file():
        arrays["n_turbines"] = np.asarray(np.load(count_path, allow_pickle=False))
        bindings["n_turbines"] = {"path": str(count_path), "sha256": _sha256(count_path)}
    else:
        archive_path = next((path for path in (volume_root.parent / "family_tensor.npz",
                                               volume_root / "family_tensor.npz") if path.is_file()), None)
        if archive_path is None:
            raise ValueError("Wind panel selection requires native n_turbines input metadata.")
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays["n_turbines"] = np.asarray(archive["n_turbines"]).copy()
        bindings["n_turbines"] = {"path": str(archive_path), "member": "n_turbines",
                                  "sha256": _sha256(archive_path)}
    semantic = {name: array.tolist() for name, array in arrays.items()}
    return arrays, {"files": bindings, "semantic_sha256": hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest(), "field_arrays_opened": False, "target_values_read": False}


def _select_wind_metadata_panel(manifest: Mapping[str, Any], subset: Mapping[str, Any],
                                full: Mapping[str, Any], metadata: Mapping[str, Any],
                                layout_count: int) -> dict[str, Any]:
    """Join actual per-row counts to IDs; manifest count categories are unordered."""
    if layout_count < 1:
        raise ValueError("Wind panel layout count must be positive.")
    selection = manifest["train_layout_selection"]
    layouts = list(map(int, selection["layout_indices"]))
    fixed_rows = list(map(int, selection["row_indices"]))
    groups = np.asarray(metadata["layout_index"])
    directions = np.asarray(metadata["wd_deg"])
    native_counts = np.asarray(metadata["n_turbines"])
    if (groups.ndim != 1 or directions.shape != groups.shape or native_counts.shape != groups.shape
            or len(set(layouts)) != len(layouts) or len(set(fixed_rows)) != len(fixed_rows)
            or not fixed_rows or min(fixed_rows) < 0 or max(fixed_rows) >= groups.size):
        raise ValueError("Wind input metadata or fixed TRAIN inventory has invalid shapes/IDs.")
    directions_expected = list(map(float, manifest["direction_categories"]))
    if (directions_expected != [270.0, 285.0, 300.0]
            or selection["direction_categories"] != directions_expected
            or selection["direction_rows_per_layout"] != 3):
        raise ValueError("Wind panel requires the frozen three-direction contract.")
    rows_by_layout, counts = {}, {}
    for layout in layouts:
        rows = [row for row in fixed_rows if int(groups[row]) == layout]
        values = native_counts[rows]
        if (len(rows) != 3 or set(map(float, directions[rows])) != set(directions_expected)
                or np.unique(values).size != 1 or not np.isfinite(values).all()
                or np.any(values != values.astype(int)) or not 1 <= int(values[0]) <= 30):
            raise ValueError("Wind layout native count/direction metadata does not verify.")
        rows_by_layout[layout] = sorted(rows, key=lambda row: float(directions[row]))
        counts[layout] = int(values[0])
    if set(map(int, groups[fixed_rows])) != set(layouts):
        raise ValueError("Wind fixed rows and layout inventory disagree.")
    if sorted(set(counts.values())) != sorted(set(map(int, selection["selected_turbine_counts"]))):
        raise ValueError("Wind native count categories differ from the frozen input-only manifest.")
    eligible = sorted(set(map(int, subset["train_layout_indices"]))
                      & set(map(int, full["train_layout_indices"])) & set(layouts),
                      key=lambda layout: (counts[layout], layout))
    if len(eligible) < layout_count:
        raise ValueError("Wind common TRAIN inventory is smaller than the requested panel.")
    positions = np.rint(np.linspace(0, len(eligible) - 1, layout_count)).astype(int)
    selected = [eligible[index] for index in positions]
    rows = [row for layout in selected for row in rows_by_layout[layout]]
    if (len(set(selected)) != layout_count or set(rows) - set(map(int, subset["train_rows"]))
            or set(rows) - set(map(int, full["train_rows"]))):
        raise ValueError("Wind selected direction rows are outside common TRAIN exposure.")
    return {
        "selection": "deterministic native turbine-count rank sample from common fixed/subset/full TRAIN layouts; retain all directions",
        "eligible_layout_count": len(eligible),
        "eligible_turbine_counts": [counts[layout] for layout in eligible],
        "eligible_layout_count_map": {str(layout): counts[layout] for layout in eligible},
        "selected_layout_indices": selected,
        "selected_turbine_counts": [counts[layout] for layout in selected],
        "direction_categories_deg": directions_expected,
        "selected_row_indices": rows,
        "selected_rows_by_layout": {str(layout): rows_by_layout[layout] for layout in selected},
        "data_access": "native layout/count/direction input metadata and sealed TRAIN memberships only; no field arrays or TEST labels opened",
    }


def _freeze_panels(args: argparse.Namespace) -> dict[str, Any]:
    from channelthermal.data.development_split import (
        DEFAULT_POLICY,
        _choose,
        _policy,
        read_case_catalog,
        validate_development_manifest,
    )

    thermal_manifest_path = args.thermal_manifest.expanduser().resolve()
    thermal_train_identity_path = args.thermal_train_identity.expanduser().resolve()
    thermal_full_identity_path = args.thermal_full_identity.expanduser().resolve()
    wind_manifest_path = args.wind_manifest.expanduser().resolve()
    wind_subset_identity_path = args.wind_subset_identity.expanduser().resolve()
    wind_full_identity_path = args.wind_full_identity.expanduser().resolve()

    thermal_manifest = _read_json(thermal_manifest_path)
    thermal_source_path = Path(thermal_manifest["source"]["dataset_path"])
    thermal_manifest = validate_development_manifest(thermal_manifest, thermal_source_path)
    source, records = read_case_catalog(thermal_source_path)
    by_id = {row["case_id"]: row for row in records}
    thermal_train_identity = _read_json(thermal_train_identity_path)
    thermal_full_identity = _read_json(thermal_full_identity_path)
    t_train_ids = set(thermal_train_identity["provider_identity"]["training_case_ids"])
    t_full_ids = set(thermal_full_identity["provider_identity"]["training_case_ids"])
    fixed_train_ids = set(thermal_manifest["partitions"]["train"]["case_ids"])
    eligible_ids = sorted(t_train_ids & t_full_ids & fixed_train_ids)
    eligible = [by_id[case_id] for case_id in eligible_ids]
    policy = _policy({**DEFAULT_POLICY, "seed": int(args.selection_seed), "fraction": 1.0})
    selected_ids, strata = _choose(
        eligible,
        int(args.thermal_case_count),
        policy,
        np.random.default_rng(np.random.SeedSequence([policy["seed"], 31])),
    )
    selected = [by_id[case_id] for case_id in selected_ids]
    modules = sorted({int(row["module_count"]) for row in eligible})
    if {int(row["module_count"]) for row in selected} != set(modules):
        raise ValueError("Thermal selector did not retain every eligible module-count category.")
    thermal_panel = {
        "source_metadata_sha256": source["metadata_sha256"],
        "fixed_manifest_sha256": thermal_manifest["manifest_sha256"],
        "selection": "deterministic stratified TRAIN sample selected by module count, Reynolds band and heat rank metadata",
        "selection_seed": policy["seed"],
        "eligible_count": len(eligible_ids),
        "eligible_module_counts": modules,
        "selected_case_ids": selected_ids,
        "selected_metadata": [
            {key: row[key] for key in ("case_id", "module_count", "re", "total_heat", "input_sha256")}
            for row in selected
        ],
        "strata": strata,
        "train_identity_sha256": _sha256(thermal_train_identity_path),
        "full_identity_sha256": _sha256(thermal_full_identity_path),
        "data_access": "case IDs, original splits, module presence, heat powers, module centers and material input attributes only; no targets loaded",
    }

    wind_manifest = _read_json(wind_manifest_path)
    wind_subset_identity = _read_json(wind_subset_identity_path)
    wind_full_identity = _read_json(wind_full_identity_path)
    subset = wind_subset_identity["provider_identity"]
    full = wind_full_identity["provider_identity"]
    if wind_manifest.get("manifest_sha256") != subset.get("subset_manifest_sha256"):
        raise ValueError("Wind fixed manifest does not match the subset checkpoint identity.")
    metadata, metadata_binding = _load_wind_panel_metadata(args.wind_data_root)
    wind_panel = _select_wind_metadata_panel(
        wind_manifest, subset, full, metadata, int(args.wind_layout_count)
    )
    wind_panel.update({
        "fixed_manifest_sha256": wind_manifest["manifest_sha256"],
        "subset_identity_sha256": _sha256(wind_subset_identity_path),
        "full_identity_sha256": _sha256(wind_full_identity_path),
        "native_metadata_binding": metadata_binding,
    })
    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "reference-anchored fixed-weight TRAIN fit diagnosis",
        "created_from_metadata_only": True,
        "panels": {"thermal": thermal_panel, "wind": wind_panel},
        "evaluation_protocol": {
            "fixed_receivers": "use each existing provider's native deterministic sampler and sealed case/query keys",
            "test_access": "forbidden",
        },
    }
    result["panel_manifest_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return result


def _add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--context-factory", required=True,
                        help="Python factory in module:callable form; returns model, provider, TrainingEngine.")
    parser.add_argument("--factory-config", type=Path, required=True,
                        help="JSON configuration passed to the context factory.")
    parser.add_argument("--panel", type=Path, required=True,
                        help="JSON panel with TRAIN case batches and fixed phase/arm/epoch.")
    parser.add_argument("--output", type=Path, required=True, help="Write-once JSON receipt path.")
    parser.add_argument("--pythonpath", action="append", default=[],
                        help="Optional source root added before importing the context factory; repeatable.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    panels = commands.add_parser("freeze-panels", help="Freeze metadata-only fixed TRAIN panels from supplied manifests.")
    panels.add_argument("--thermal-manifest", type=Path, required=True)
    panels.add_argument("--thermal-train-identity", type=Path, required=True)
    panels.add_argument("--thermal-full-identity", type=Path, required=True)
    panels.add_argument("--wind-data-root", type=Path, required=True,
                        help="Native Wind input metadata; field arrays are never opened for selection.")
    panels.add_argument("--wind-manifest", type=Path, required=True)
    panels.add_argument("--wind-subset-identity", type=Path, required=True)
    panels.add_argument("--wind-full-identity", type=Path, required=True)
    panels.add_argument("--thermal-case-count", type=int, default=24)
    panels.add_argument("--wind-layout-count", type=int, default=12)
    panels.add_argument("--selection-seed", type=int, required=True)
    panels.add_argument("--output", type=Path, required=True)

    loss = commands.add_parser("loss", help="Measure fixed-batch LossTerms using native provider sampling.")
    _add_context_args(loss)
    gradients = commands.add_parser("gradients", help="Diagnose exact LossTerm gradients without an optimizer step.")
    _add_context_args(gradients)
    metrics = commands.add_parser(
        "native-metrics",
        help="Measure provider-native physical metrics and compact prediction/residual samples on fixed TRAIN rows.",
    )
    _add_context_args(metrics)
    compare = commands.add_parser("counterfactual-updates", help="Compare declared one-step cloned AdamW interventions.")
    _add_context_args(compare)
    compare.add_argument("--declared-optimizer-updates", type=int, required=True,
                         help="Must equal the exact number of intervention arms in the fixed panel manifest.")

    fit_branch = commands.add_parser(
        "fit-branch",
        help="CPU-preflight or run a hash-bound bounded objective child through the existing TrainingEngine.fit.",
    )
    fit_branch.add_argument("--checkpoint", type=Path, required=True)
    fit_branch.add_argument("--context-factory", required=True)
    fit_branch.add_argument("--factory-config", type=Path, required=True)
    fit_branch.add_argument("--fit-plan", type=Path, required=True,
                            help="JSON plan with frozen TRAIN cases, selected loss weights, reviews and evidence hash.")
    fit_branch.add_argument("--run-output", "--output-dir", dest="run_output", type=Path, required=True,
                            help="Fresh child run directory; --output-dir is recognized by the resource governor.")
    fit_branch.add_argument("--output", type=Path, required=True,
                            help="Write-once JSON preflight or completed-fit receipt.")
    fit_branch.add_argument("--execute", action="store_true",
                            help="Run the declared bounded fit; without this flag only CPU preflight runs.")
    fit_branch.add_argument("--stop-after", type=int, default=None,
                            help="Stop at this predeclared review age; defaults to the final declared review.")
    fit_branch.add_argument("--resume-checkpoint", type=Path, default=None,
                            help="Resume this child run's exact latest checkpoint instead of creating a fresh branch.")
    fit_branch.add_argument("--pythonpath", action="append", default=[],
                            help="Optional source root added before importing the context factory; repeatable.")

    args = parser.parse_args(argv)
    if args.command == "freeze-panels":
        result = _freeze_panels(args)
        _write_once(args.output.expanduser().resolve(), result)
        print(json.dumps({
            "output": str(args.output.expanduser().resolve()),
            "panel_manifest_sha256": result["panel_manifest_sha256"],
            "thermal_cases": len(result["panels"]["thermal"]["selected_case_ids"]),
            "wind_layouts": len(result["panels"]["wind"]["selected_layout_indices"]),
            "wind_direction_rows": len(result["panels"]["wind"]["selected_row_indices"]),
            "optimizer_updates": 0,
            "target_values_read": False,
        }, indent=2, sort_keys=True))
        return 0
    for path in args.pythonpath:
        resolved = str(Path(path).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    if args.command == "counterfactual-updates" and args.declared_optimizer_updates < 1:
        parser.error("--declared-optimizer-updates must be positive.")
    args.started_at = time.perf_counter()
    receipt = (_run_diagnostic_fit_branch(args) if args.command == "fit-branch"
               else _run_diagnosis(args))
    _write_once(args.output.expanduser().resolve(), receipt)
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "source_checkpoint_sha256": receipt.get("source_checkpoint_sha256", receipt["checkpoint_sha256"]),
        "result_checkpoint_sha256": receipt.get("result_checkpoint_sha256"),
        "optimizer_updates": receipt["optimizer_updates"],
        "elapsed_seconds": receipt["elapsed_seconds"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
