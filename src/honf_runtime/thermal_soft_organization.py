"""Bounded TRAIN-only soft organization; optional maintained-engine capability.

No optimizer lives here. Calibration runs the maintained macro backward with a
throwaway optimizer and aborts before its step, then restores model/provider/RNG.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


TERMS = tuple(f"compact_{term}_{bank}" for bank in ("M", "E") for term in ("tie", "capacity"))


def validate_softonly_config(config):
    r, t = config["round12"], config["training"]
    expected = {
        "compact_arm": "soft_only", "compact_mode": "soft_only",
        "cut_after_local_epoch": None, "freeze_policy": False,
        "alpha_schedule": {"kind": "cosine", "first_epoch": 1, "last_ramp_epoch": 200,
                           "initial": 0.1, "terminal": 1.0},
        "calibration_epochs": [1, 100, 200], "calibration_max_cases": 96,
        "structural_loss_scale": 0.02, "structural_full_gradient_cap": 0.1,
        "geometry_gradient_scale": 0.1, "geometry_every_primary_updates": 4,
        "geometry_train_families": ["0001", "0318", "0333", "0348"],
        "geometry_development_families": ["0304", "0320", "0335", "0350"],
        "organizer_review_epochs": [100, 300, 600, 1000],
        "tie_loss_gradient": "stop_outer_probability_v1",
    }
    for key, value in expected.items():
        if r.get(key) != value:
            raise ValueError(f"Frozen soft_only contract differs: round12.{key}")
    if (t["checkpoint_epochs"] != list(range(100, 1001, 100))
            or config["initialization"]["clock_mapping"] != {
                "sampler_parent_epoch": 0, "controller_parent_coordinate": 3000,
                "parent_update_index": 0, "parent_macro_clock": 0}):
        raise ValueError("Soft-only scratch cadence/clocks differ from the declared contract.")
    if r.get("budget") != {"training_seconds": 129600, "qualification_seconds": 7200,
                            "validation_seconds": 14400, "epoch_reserve_seconds": 600}:
        raise ValueError("Soft-only requires explicit finite training/qualification/validation budgets.")


def select_calibration_panel(cases, limit=96):
    """Input-only round-robin, lexicographic IDs within every module-count stratum."""
    strata = {}
    for case in cases:
        present = np.asarray(case["structure"]["module_present"])
        if present.ndim != 1 or not np.isfinite(present).all():
            raise ValueError(f"Calibration case {case.get('case_id')} has invalid module_present metadata.")
        count = int((present > 0.5).sum())
        strata.setdefault(count, []).append(str(case["case_id"]))
    for ids in strata.values():
        ids.sort()
    if limit < len(strata):
        raise ValueError("Calibration cannot cover every module-count category within this limit.")
    panel = []
    for index in range(max(map(len, strata.values()))):
        for count in sorted(strata):
            if index < len(strata[count]):
                panel.append(strata[count][index])
                if len(panel) == limit:
                    return tuple(panel)
    return tuple(panel)


def gradient_norm(gradients):
    values = [g.detach().double().square().sum() for g in gradients.values() if g is not None]
    return math.sqrt(float(torch.stack(values).sum().cpu())) if values else 0.0


def gradient_balance(native, other, names):
    """Both terms use exactly these names; absent VJPs are mathematical zeros."""
    left = {n: native[n] for n in names if n in native}
    right = {n: other[n] for n in names if n in other}
    a, b = gradient_norm(left), gradient_norm(right)
    dot = sum(float((left[n].double() * right[n].double()).sum().cpu())
              for n in names if n in left and n in right)
    positive = negative = zeros = 0
    for name in names:
        if name in left and name in right:
            product = left[name] * right[name]
            positive += int((product > 0).sum())
            negative += int((product < 0).sum())
            zeros += int((product == 0).sum())
    return {"native_norm": a, "term_norm": b, "signed_dot": dot,
            "cosine": dot / (a * b) if a > 0 and b > 0 else None,
            "same_sign_elements": positive, "opposed_sign_elements": negative,
            "zero_product_elements_present": zeros,
            "native_missing_names_zero_filled": [n for n in names if n not in native],
            "term_missing_names_zero_filled": [n for n in names if n not in other],
            "parameter_names": list(names)}


class _CalibrationFinished(Exception):
    pass


class SoftOrganizationRuntime:
    def __init__(self, provider):
        from .thermal_geometry_supervision import GeometrySupervision
        self.provider = provider
        self.config = provider._research_config
        self.panel = select_calibration_panel(provider.training_cases)
        self.geometry = GeometrySupervision(provider.atlas_directory,
            fluid_queries=512, material_queries_per_module=64, ntheta=16, seed=0)
        self.state = {"schema": "thermal_soft_organization_state_v2",
                      "panel_case_ids": list(self.panel), "calibrated_epochs": [],
                      "coefficients": {name: 0.0 for name in TERMS},
                      "geometry_coefficient": 0.0, "primary_updates": 0,
                      "response_microbatches": 0, "response_scene_visits": 0,
                      "response_query_rows": 0, "calibration_scene_visits": 0,
                      "response_native_query_rows": 0,
                      "calibration_primary_query_rows": 0,
                      "calibration_response_query_rows": 0,
                      "calibration_response_native_query_rows": 0,
                      "training_elapsed_seconds": 0.0, "review_elapsed_seconds": 0.0,
                      "geometry_identity": self.geometry.metadata,
                      "organizer_reviews": []}
        self.calibrating = False
        self.last_native = {}
        self.last_raw = {}
        self.last_record = {}
        self.run = None
        self.started = time.monotonic()
        self.elapsed_base = 0.0
        provider.model.core.joint_apply_additive_gradients = self.apply_additives

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        if (state.get("schema") != self.state["schema"]
                or state.get("panel_case_ids") != list(self.panel)
                or state.get("geometry_identity") != self.geometry.metadata):
            raise ValueError("Soft-only resume changed calibration inputs or physical references.")
        # In-process OOM/calibration rollback must not erase failed-attempt
        # wall time. A newly constructed resume also honors saved elapsed time.
        live_elapsed = self.elapsed_base + time.monotonic() - self.started
        self.state = copy.deepcopy(state)
        self.elapsed_base = max(float(state["training_elapsed_seconds"]), live_elapsed)
        self.started = time.monotonic()

    def _named(self):
        return {n: p for n, p in self.provider.model.named_parameters() if p.requires_grad}

    def apply_additives(self, objective):
        named = self._named()
        reverse = {p: n for n, p in named.items()}
        native = {n: p.grad.detach().clone() for n, p in named.items() if p.grad is not None}
        raw = {name: {reverse[p]: g / objective.additive_denominators[name]
                       for p, g in gradients.items() if p in reverse}
               for name, gradients in objective.additive_gradients.items()}
        self.last_native, self.last_raw = native, raw
        combined = {}
        record = {"native_full_norm": gradient_norm(native), "terms": {}}
        record["actual_coverage"] = {k: v for k, v in objective.work.items()
                                    if "eligible_pairs" in k or "valid_scenes" in k}
        for name in TERMS:
            bank = name[-1]
            policy_names = [n for n in named if f"compatibility.{bank}." in n]
            if not policy_names:
                raise ValueError("Soft organizer has no trainable bank policy parameters.")
            gradients = raw.get(name, {})
            balance = gradient_balance(native, gradients, policy_names)
            coefficient = float(self.state["coefficients"][name])
            balance["coefficient"] = coefficient
            record["terms"][name] = balance
            for n, g in gradients.items():
                contribution = coefficient * g
                combined[n] = combined.get(n, 0) + contribution
        raw_norm = gradient_norm(combined)
        bound = 0.1 * record["native_full_norm"]
        scale = min(1.0, bound / raw_norm) if raw_norm > 0 else 1.0
        record.update(structural_full_norm_before_cap=raw_norm,
                      structural_cap_scale=scale, structural_full_norm=raw_norm * scale)
        self.last_record = record
        if self.calibrating:
            return
        for n, gradient in combined.items():
            contribution = gradient * scale
            if named[n].grad is None:
                named[n].grad = contribution
            else:
                named[n].grad.add_(contribution)
        for name, values in objective.additive_values.items():
            objective.losses[name] = (float(torch.stack(values).sum().cpu()) /
                objective.additive_denominators[name] * self.state["coefficients"][name] * scale)
            objective.work[name + "_case_requests"] = objective.additive_denominators[name]
        for key in ("native_full_norm", "structural_full_norm", "structural_full_norm_before_cap", "structural_cap_scale"):
            objective.work["soft_" + key] = record[key]
        for name, detail in record["terms"].items():
            for key in ("native_norm", "term_norm", "signed_dot", "same_sign_elements", "opposed_sign_elements"):
                objective.work[f"soft_{name}_{key}"] = detail[key]

    def _geometry_gradients(self, model, index, *, calibration=False):
        named = self._named()
        loss, detail = self.geometry.loss(model, index, calibration=calibration)
        names = [n for n in named if ".compatibility." not in n]
        all_names = list(named)
        gradients = torch.autograd.grad(loss, [named[n] for n in all_names], allow_unused=True)
        return loss, detail, {n: g.detach() for n, g in zip(all_names, gradients) if g is not None}, names

    def before_step(self, model, optimizer, losses, work, *, epoch, update_index):
        if self.calibrating:
            raise _CalibrationFinished()
        next_update = self.state["primary_updates"] + 1
        if next_update % 4 == 0:
            index = next_update // 4 - 1
            loss, detail, gradients, names = self._geometry_gradients(model, index)
            coefficient = self.state["geometry_coefficient"]
            native_norm = gradient_norm({n: self.last_native[n] for n in names if n in self.last_native})
            raw_norm = gradient_norm({n: gradients[n] for n in names if n in gradients})
            # Fixed calibrated coefficient, with a downward-only safety cap.
            scale = min(1.0, 0.1 * native_norm / (coefficient * raw_norm)) if coefficient * raw_norm > 0 else 1.0
            for n, gradient in gradients.items():
                p = dict(model.named_parameters())[n]
                contribution = gradient * (coefficient * scale)
                if p.grad is None:
                    p.grad = contribution
                else:
                    p.grad.add_(contribution)
            losses["geometry_response"] = float(loss.detach()) * coefficient * scale
            work.update(geometry_microbatches=1, geometry_scene_visits=detail["scene_visits"],
                        geometry_query_rows=detail["query_rows"], geometry_raw_gradient_norm=raw_norm,
                        geometry_native_query_rows=detail["native_query_rows"],
                        geometry_native_shared_norm=native_norm,
                        geometry_weighted_gradient_norm=coefficient * scale * raw_norm,
                        geometry_full_weighted_gradient_norm=coefficient * scale * gradient_norm(gradients),
                        geometry_gradient_cap_scale=scale)
            self.state["response_microbatches"] += 1
            self.state["response_scene_visits"] += detail["scene_visits"]
            self.state["response_query_rows"] += detail["query_rows"]
            self.state["response_native_query_rows"] += detail["native_query_rows"]
        self.state["primary_updates"] = next_update
        self.last_native = {}
        self.last_raw = {}

    def before_update(self, engine, model, provider, optimizer, epoch, update_index):
        self.run = engine._execution_runtime_path.parent if engine._execution_runtime_path else self.run
        model.core.set_compact_phase(epoch, arm="soft_only")
        if update_index == 0 and epoch in (1, 100, 200) and epoch not in self.state["calibrated_epochs"]:
            self.calibrate(engine, model, provider, epoch)
        if self.run is not None:
            elapsed = self.elapsed_base + time.monotonic() - self.started
            if elapsed >= self.config["round12"]["budget"]["training_seconds"] - 600:
                self._request_stop("training wall-time budget reserve")

    def calibrate(self, engine, model, provider, epoch):
        from . import research_training as research
        from .unified_training import TrainingEngine, capture_rng_state, restore_rng_state
        state = research._cpu_state_snapshot(model.state_dict())
        provider_state = research._cpu_state_snapshot(provider.training_state_dict())
        rng = research._cpu_state_snapshot(capture_rng_state())
        mode = model.training
        started = time.monotonic()
        elapsed_before = self.elapsed_base + started - self.started
        record = None
        try:
            self.calibrating = True
            disposable, _ = engine._make_optimizer(model, provider, model.architecture, epoch, None)
            batches = engine.prepare_update_microbatches(provider, arm=model.architecture, phase="joint",
                epoch=epoch, update_index=0, update_cases=self.panel,
                sampling_dataset_id=provider.sampling_dataset_id)
            try:
                TrainingEngine._run_update(engine, model, provider, disposable, batches,
                    phase="joint", arm=model.architecture,
                    epoch=3000 + epoch, update_index=0)
                raise RuntimeError("No-update calibration unexpectedly reached optimizer.step.")
            except _CalibrationFinished:
                pass
            record = copy.deepcopy(self.last_record)
            coefficients = {}
            for name, detail in record["terms"].items():
                a, b = detail["native_norm"], detail["term_norm"]
                coefficients[name] = 0.02 * a / b if a > 0 and b > 0 else 0.0
                if not math.isfinite(coefficients[name]):
                    raise FloatingPointError("Nonfinite calibrated coefficient; no amplification fallback allowed.")
                detail["calibrated_coefficient"] = coefficients[name]
            loss, detail, gradients, names = self._geometry_gradients(model, 0, calibration=True)
            geometry = gradient_balance(self.last_native, gradients, names)
            a, b = geometry["native_norm"], geometry["term_norm"]
            geometry_coefficient = 0.1 * a / b if a > 0 and b > 0 else 0.0
            if not math.isfinite(geometry_coefficient):
                raise FloatingPointError("Nonfinite geometry calibration.")
            geometry.update(coefficient=geometry_coefficient, physical_exposure=detail,
                            loss=float(loss.detach()))
            counts = {str(int((np.asarray(c["structure"]["module_present"]) > 0.5).sum())): 0
                      for c in provider.training_cases}
            for c in provider.training_cases:
                if str(c["case_id"]) in self.panel:
                    module_count = int((np.asarray(c["structure"]["module_present"]) > 0.5).sum())
                    counts[str(module_count)] += 1
            record.update(schema="thermal_softonly_no_update_calibration_v2", epoch=epoch,
                          alpha=model.core.compact_alpha, case_ids=list(self.panel),
                          module_count_coverage=counts, geometry=geometry,
                          optimizer_updates=0, elapsed_seconds=time.monotonic() - started,
                          primary_query_rows=len(self.panel) * 1024)
        finally:
            self.calibrating = False
            model.load_state_dict(state, strict=True)
            provider.load_training_state_dict(provider_state)
            self.elapsed_base = elapsed_before + time.monotonic() - started
            self.started = time.monotonic()
            restore_rng_state(rng)
            model.train(mode)
            model.zero_grad(set_to_none=True)
            model.core._joint_entries = []
            model.core._context_entries = []
            model.core._compact_loss_entries = []
            self.last_native = {}
            self.last_raw = {}
        self.state["coefficients"] = coefficients
        self.state["geometry_coefficient"] = geometry_coefficient
        self.state["calibrated_epochs"].append(epoch)
        self.state["calibration_scene_visits"] += len(self.panel) + detail["scene_visits"]
        self.state["calibration_primary_query_rows"] += len(self.panel) * 1024
        self.state["calibration_response_query_rows"] += detail["query_rows"]
        self.state["calibration_response_native_query_rows"] += detail["native_query_rows"]
        record["model_optimizer_rng_restored_no_step"] = True
        if self.run is not None:
            research._atomic_json(self.run / "metrics" / f"soft_calibration_epoch_{epoch:04d}.json", record)
            research._atomic_json(self.run / "configs" / "soft_inputs_and_references.json", {
                "panel_case_ids": list(self.panel), "geometry": self.geometry.metadata})
        print(json.dumps({"event": "softonly_calibrated", "epoch": epoch,
                          "coefficients": coefficients, "geometry_coefficient": geometry_coefficient}), flush=True)

    def _request_stop(self, reason):
        from .unified_training import _atomic_json
        if self.run is not None:
            path = self.run / "CLEAN_STOP_REQUEST.json"
            if not path.exists():
                _atomic_json(path, {"schema": "ordinary_clean_stop_request_v1",
                    "request_id": "softonly-budget", "requested_unix": time.time(),
                    "requested_by": "internal_softonly_budget", "reason": reason})

    def completed_epoch(self, model, engine, epoch, output, row):
        from . import research_training as research
        self.run = output
        self.state["training_elapsed_seconds"] = self.elapsed_base + time.monotonic() - self.started
        from .thermal_soft_reviews import run_internal_review
        if epoch in (100, 300, 600, 1000) and epoch not in self.state["organizer_reviews"]:
            start = time.monotonic()
            result = run_internal_review(model, self.provider, epoch, output)
            row["soft_organizer_review"] = result
            self.state["organizer_reviews"].append(epoch)
            self.state["review_elapsed_seconds"] += time.monotonic() - start
        # Reserve space for atomic latest/best/milestone replacement and history.
        storage = research.verify_storage(self.config, output=output,
            estimated_bytes=256 * 2**20, peak_temporary_bytes=256 * 2**20)
        if (self.state["training_elapsed_seconds"] >= 129000
                or storage["output_bytes_before"] + 512 * 2**20 >= 12 * 2**30):
            self._request_stop("finite training/storage budget boundary")
        row["soft_organization_exposure"] = {k: copy.deepcopy(v) for k, v in self.state.items()
            if k not in ("geometry_identity", "panel_case_ids")}


def qualify_softonly(config_path, device, output):
    """Four disposable macros and all calibration boundaries, never a run resume."""
    from . import research_training as research
    from .unified_training import capture_rng_state, restore_rng_state, named_optimizer_state
    config, _ = research.load_research_config(config_path)
    validate_softonly_config(config)
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError("Qualification evidence must have a new owned output directory.")
    if not str(root).startswith("/data2/wanglz/ModularDT/") or not Path("/data2").is_mount():
        raise ValueError("Qualification artifacts must use the mounted DATA2 workspace.")
    parent = root.parent
    while not parent.exists():
        parent = parent.parent
    if parent.stat().st_dev != Path("/data2").stat().st_dev:
        raise ValueError("Qualification output is not bound to the DATA2 filesystem.")
    if Path(config["storage"]["output_root"]).exists():
        raise FileExistsError("Qualification cannot share an existing scientific run.")
    research.verify_storage(config, estimated_bytes=512 * 2**20, peak_temporary_bytes=2**30)
    root.mkdir(parents=True)
    for name in ("metrics", "configs", "checkpoints"):
        (root / name).mkdir()
    selected, device_info = research.resolve_device(device, permitted_devices=[device])
    started = time.monotonic()
    model, source, source_path, source_sha = research._source_components(config, selected)
    contract = source["experiment_identity"]["optimizer_schedule_contract"]
    model, provider = research._build_research_provider(config, model, source, contract,
        case_ids=None, case_limit=None, local_horizon=1000, diagnostic=False, identity_microbatch=24)
    engine = research._engine_for(config, device=selected, parent_age=0)
    engine.bind_execution_runtime(root / "execution_runtime.json", nominal_microbatch_cases=24)
    optimizer, specs = engine._make_optimizer(model, provider, model.architecture, 1, None)
    torch.cuda.reset_peak_memory_stats(selected) if selected.type == "cuda" else None
    results = []
    for epoch in (1, 100, 200, 201):
        if time.monotonic() - started >= 6600:
            raise RuntimeError("Qualification reached its 2-hour budget reserve; scientific launch forbidden.")
        model.train(True)
        case_ids = provider.soft_organization.panel[:48]
        batches = engine.prepare_update_microbatches(provider, arm=model.architecture, phase="joint",
            epoch=epoch, update_index=0, update_cases=case_ids,
            sampling_dataset_id=provider.sampling_dataset_id)
        tick = time.monotonic()
        values = engine._run_update(model, provider, optimizer, batches, phase="joint",
            arm=model.architecture, epoch=epoch, update_index=0)
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)
        results.append({"local_epoch": epoch, "elapsed_seconds": time.monotonic() - tick,
                        "losses": values[0], "work": values[1]})
        if bool(model.core.compact_cut_active) or bool(model.core.compact_policy_frozen):
            raise RuntimeError("Soft-only qualification unexpectedly cut/froze the policy.")
    # Own-state strict restoration, including named Adam, provider exposure and
    # every visible CUDA RNG stream. No qualification weights enter the new fit.
    payload = engine._checkpoint_payload(model, optimizer,
        identity={"provider_identity": provider.identity_payload()}, arm=model.architecture,
        epoch=201, history=[], best_field=1.0, best_guarded=1.0,
        sampler_state={"completed_epoch": 201}, group_specs=specs,
        provider_training_state=provider.training_state_dict())
    checkpoint = root / "checkpoints" / "qualification_only_model.pt"
    from .unified_training import _atomic_torch_save
    _atomic_torch_save(checkpoint, payload)
    from .compat import load_trusted_checkpoint
    saved = load_trusted_checkpoint(checkpoint, map_location="cpu")
    restored, restored_provider = research._build_provider_for_resume(
        config, saved, contract, None, 1000, False, selected)
    restored_provider.load_training_state_dict(saved["provider_training_state"])
    for name, value in model.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    restore_rng_state(saved["rng_state"])
    # Named Adam loader is exercised on the same declared parameter contract.
    restored_optimizer, _ = engine._make_optimizer(restored, restored_provider, model.architecture, 201,
                                                  saved["optimizer_state_by_name"])
    before = named_optimizer_state(model, optimizer)
    after = named_optimizer_state(restored, restored_optimizer)
    _assert_state_equal(before, after)
    macro_times = [row["elapsed_seconds"] for row in results if row["local_epoch"] == 201]
    projected_seconds = max(macro_times) * 13 * 1000 * 1.75 + 1800
    checkpoint_size = checkpoint.stat().st_size
    estimated_retained = checkpoint_size * 16 + 1024**3
    if projected_seconds >= 129000 or estimated_retained >= 12 * 2**30:
        raise RuntimeError("Measured runtime/storage projection exceeds the frozen budget; launch forbidden.")
    receipt = {"schema": "thermal_softonly_qualification_v2", "status": "passed",
        "config_sha256": research._canonical_sha256(config), "device": device_info,
        "elapsed_seconds": time.monotonic() - started, "macros": results,
        "calibrations": provider.soft_organization.state["calibrated_epochs"],
        "response_microbatches": provider.soft_organization.state["response_microbatches"],
        "estimated_training_seconds_with_75pct_margin_and_1800s_overhead": projected_seconds,
        "checkpoint_bytes": checkpoint_size, "estimated_retained_output_bytes": estimated_retained,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(selected) if selected.type == "cuda" else None,
        "train_cases": list(provider.training_case_ids), "exposed_val_cases": list(provider.validation_case_ids),
        "normalizer_key": model.normalizer_key, "geometry_references": provider.soft_organization.geometry.metadata,
        "strict_model_named_adam_all_visible_rng_restore": True,
        "qualification_weights_discarded_for_scientific_fit": True}
    research._atomic_json(root / "qualification.json", receipt)
    return {"status": "softonly_qualification_passed", "evidence": str(root / "qualification.json"),
            "elapsed_seconds": receipt["elapsed_seconds"], "projected_training_seconds": projected_seconds}


def _assert_state_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_state_equal(a, b)
    else:
        assert left == right
