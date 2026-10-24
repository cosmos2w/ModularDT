"""Streaming organizer gradients for the Thermal contextual-frontier task.

Native field losses keep their ordinary conditional hard-frontier gradient.
This helper accumulates detached VJPs for expected record work and the sampled
frontier score function, then adds calibrated organizer gradients immediately
before the optimizer step. It intentionally does not own an optimizer.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from honf_runtime.thermal_soft_organization import (
    gradient_balance,
    gradient_norm,
    select_calibration_panel,
)


_BANKS = ("module", "environment")
_BASE_TERMS = tuple(f"{kind}_{bank}" for bank in _BANKS for kind in ("capacity", "reinforce"))
_CALIBRATION_EPOCHS = (1, 21, 100)


def _clone_gradients(gradients: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {str(name): value.detach().clone() for name, value in gradients.items() if value is not None}


def _add_gradients(target: dict[str, torch.Tensor], values: Mapping[str, torch.Tensor]) -> None:
    for name, gradient in values.items():
        if gradient is None:
            continue
        detached = gradient.detach()
        if name in target:
            target[name].add_(detached)
        else:
            target[name] = detached.clone()


def _accumulator_snapshot(accumulators: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "bank": row["bank"],
            "gradients": _clone_gradients(row["gradients"]),
            "samples": int(row["samples"]),
            "objective_value_sum": row.get("objective_value_sum", 0.0),
            "objective_value_samples": row.get("objective_value_samples", 0),
        }
        for name, row in accumulators.items()
    }


class ContextualJointObjectiveRuntime:
    """Accumulate per-bank VJPs and apply their calibrated, capped sum.

    ``collect`` is called while one native prediction graph is alive. Its
    ``per_case_native_losses`` argument is the weighted physical task loss for
    each case, before any organizer objective is added. ``raw_terms`` accepts
    additional named gradients (for example a separately replayed functional
    probe) as rows containing ``bank``, ``gradients`` and optional ``samples``.
    The caller owns those functional comparisons and their replay cadence.
    """

    SCHEMA = "thermal_contextual_joint_objective_v1"

    def __init__(
        self,
        model: torch.nn.Module | Any,
        provider: Any | None = None,
        *,
        target_fraction: float = 0.02,
        full_gradient_cap: float = 0.10,
        baseline_decay: float = 0.9,
        calibration_epochs: Sequence[int] = _CALIBRATION_EPOCHS,
        calibration_max_cases: int = 96,
    ) -> None:
        if not 0.0 < float(target_fraction) or not 0.0 < float(full_gradient_cap):
            raise ValueError("Organizer calibration and cap fractions must be positive.")
        if not 0.0 <= float(baseline_decay) < 1.0:
            raise ValueError("baseline_decay must be in [0, 1).")
        epochs = tuple(int(value) for value in calibration_epochs)
        if any(value < 1 for value in epochs) or tuple(sorted(set(epochs))) != epochs:
            raise ValueError("Calibration epochs must be positive, sorted, and unique.")
        if not 1 <= int(calibration_max_cases) <= 96:
            raise ValueError("Calibration panel must contain at most 96 cases.")

        if isinstance(model, torch.nn.Module):
            self.model = model
            self.provider = provider
        elif getattr(model, "model", None) is not None:
            self.provider = model if provider is None else provider
            self.model = self.provider.model
        else:
            raise TypeError("Pass a model or a provider with a bound .model.")
        self.core = getattr(self.model, "core", self.model)
        if not callable(getattr(self.core, "organizer_parameters", None)):
            raise TypeError("Contextual core must expose organizer_parameters(bank).")

        self.target_fraction = float(target_fraction)
        self.full_gradient_cap = float(full_gradient_cap)
        self.baseline_decay = float(baseline_decay)
        self.calibration_epochs = epochs
        self.calibration_max_cases = int(calibration_max_cases)
        self._baseline: dict[str, float] = {}
        self._pending_baseline: dict[str, list[float]] = {}
        self._accumulators: dict[str, dict[str, Any]] = {}
        self._native_calibration_gradients: dict[str, torch.Tensor] = {}
        self._native_calibration_cases = 0
        self._current_case_coverage: dict[str, int] = {bank: 0 for bank in _BANKS}
        self._current_vjp_coverage: dict[str, dict[str, int]] = {}
        self.last_collection: dict[str, Any] | None = None
        self.calibration_extra: list[Any] = []
        self._calibrating = False
        self.state: dict[str, Any] = {
            "schema": self.SCHEMA,
            "coefficients": {},
            "calibrated_epochs": [],
            "baselines": {},
            "calibrations": [],
            "last_update": {},
        }

    def _named_model_parameters(self) -> dict[str, torch.nn.Parameter]:
        return {name: value for name, value in self.model.named_parameters() if value.requires_grad}

    def _inventory(self, bank: str) -> dict[str, torch.nn.Parameter]:
        if bank not in _BANKS:
            raise ValueError(f"Unknown organizer bank {bank!r}.")
        model_named = self._named_model_parameters()
        model_names_by_identity = {id(parameter): name for name, parameter in model_named.items()}
        core_inventory = self.core.organizer_parameters(bank)
        if not isinstance(core_inventory, Mapping):
            raise TypeError("organizer_parameters(bank) must return a named parameter mapping.")
        inventory = {}
        for core_name, parameter in core_inventory.items():
            if not isinstance(parameter, torch.nn.Parameter) or not parameter.requires_grad:
                continue
            name = model_names_by_identity.get(id(parameter))
            if name is None:
                raise ValueError(f"Organizer parameter {core_name!r} is not owned by the supplied model.")
            inventory[name] = parameter
        if not inventory:
            raise ValueError(f"The {bank} organizer inventory contains no trainable model parameters.")
        return inventory

    def _structural_multiplier(self, epoch: int) -> float:
        """Read the shared frontier ramp, defaulting to unity for test doubles."""
        policies = getattr(self.core, "frontier_policies", None)
        if policies is None:
            return 1.0
        values = [float(policy.structural_multiplier_for_epoch(epoch)) for policy in policies.values()]
        if not values:
            return 1.0
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise FloatingPointError("The frontier structural multiplier is invalid.")
        if any(not math.isclose(value, values[0], rel_tol=0.0, abs_tol=1e-12) for value in values[1:]):
            raise ValueError("Typed frontier policies must share one structural multiplier schedule.")
        return values[0]

    def organizer_parameters(self, bank: str) -> dict[str, torch.nn.Parameter]:
        """Expose the complete bank inventory under model-qualified names."""
        return self._inventory(bank)

    def _normalize_injected_names(self, bank: str, gradients: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inventory = self._inventory(bank)
        aliases = {}
        model_names_by_identity = {id(parameter): name for name, parameter in inventory.items()}
        for core_name, parameter in self.core.organizer_parameters(bank).items():
            model_name = model_names_by_identity.get(id(parameter))
            if model_name is not None:
                aliases[str(core_name)] = model_name
        normalized = {}
        for raw_name, gradient in gradients.items():
            name = str(raw_name)
            target = name if name in inventory else aliases.get(name)
            if target is None:
                raise ValueError(f"Raw term parameter {name!r} is outside the {bank} organizer inventory.")
            if gradient is not None:
                if target in normalized:
                    raise ValueError(f"Raw term supplies parameter {target!r} more than once.")
                normalized[target] = gradient
        return normalized

    @staticmethod
    def _case_ids(predictions: Any, per_case_native_losses: Any, case_ids: Sequence[Any] | None) -> tuple[str, ...]:
        if case_ids is not None:
            result = tuple(str(value) for value in case_ids)
        elif isinstance(per_case_native_losses, Mapping):
            result = tuple(str(value) for value in per_case_native_losses)
        else:
            prepared = getattr(predictions, "native_prepared", None)
            scene = getattr(prepared, "scene", None)
            keys = getattr(scene, "scene_key", None)
            if isinstance(keys, (tuple, list)):
                result = tuple(str(value) for value in keys)
            elif keys is not None:
                result = (str(keys),)
            else:
                raise ValueError("case_ids are required when prediction scene keys are unavailable.")
        if not result or len(set(result)) != len(result):
            raise ValueError("A collect call requires unique, nonempty case IDs.")
        return result

    @staticmethod
    def _loss_vector(per_case_native_losses: Any, case_ids: tuple[str, ...]) -> torch.Tensor:
        if isinstance(per_case_native_losses, Mapping):
            value_by_id = {str(key): value for key, value in per_case_native_losses.items()}
            try:
                values = [value_by_id[case_id] for case_id in case_ids]
            except KeyError as error:
                raise ValueError(f"Missing native loss for case {error.args[0]!r}.") from error
            values = [torch.as_tensor(value) for value in values]
            device, dtype = values[0].device, values[0].dtype
            vector = torch.stack([value.reshape(()) for value in values]).to(device=device, dtype=dtype)
        else:
            vector = torch.as_tensor(per_case_native_losses)
            if vector.numel() != len(case_ids):
                raise ValueError("per_case_native_losses must have one scalar per case ID.")
            vector = vector.reshape(-1)
        if vector.numel() != len(case_ids) or not torch.isfinite(vector.detach()).all():
            raise ValueError("Native case losses must be finite scalars aligned with case IDs.")
        if not torch.is_floating_point(vector):
            raise TypeError("Native per-case losses must use a floating-point tensor dtype.")
        return vector

    @staticmethod
    def _prepared(predictions: Any) -> Any:
        prepared = getattr(predictions, "native_prepared", None)
        if prepared is None:
            prepared = getattr(predictions, "prepared", None)
        if prepared is None:
            raise TypeError("Predictions must expose native_prepared or prepared contextual state.")
        if not hasattr(prepared, "module_banks") and getattr(prepared, "organization", None) is not None:
            prepared = prepared.organization
        return prepared

    @staticmethod
    def _bank_rows(prepared: Any, bank: str) -> tuple[Any, ...]:
        records = getattr(prepared, f"{bank}_banks", None)
        if records is None:
            raise TypeError(f"Prepared context has no {bank}_banks inventory.")
        return tuple(records)

    def _vjp(self, scalar: torch.Tensor, inventory: Mapping[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
        if not torch.is_tensor(scalar) or scalar.numel() != 1:
            raise TypeError("A raw organizer objective must be a scalar tensor.")
        if not scalar.requires_grad:
            return {}
        names = tuple(inventory)
        values = tuple(inventory[name] for name in names)
        gradients = torch.autograd.grad(scalar, values, retain_graph=True, allow_unused=True)
        return {name: gradient.detach() for name, gradient in zip(names, gradients) if gradient is not None}

    def _accumulate(self, name: str, bank: str, gradients: Mapping[str, torch.Tensor], samples: int,
                    objective_value: float | None = None) -> None:
        if not name or bank not in _BANKS or type(samples) is not int or samples < 1:
            raise ValueError("Named raw gradients need a valid bank and positive sample denominator.")
        inventory = self._inventory(bank)
        unknown = set(gradients) - set(inventory)
        if unknown:
            raise ValueError(f"Raw term {name!r} contains parameters outside the {bank} inventory: {sorted(unknown)}")
        row = self._accumulators.setdefault(name, {
            "bank": bank, "gradients": {}, "samples": 0,
            "objective_value_sum": 0.0, "objective_value_samples": 0,
        })
        if row["bank"] != bank:
            raise ValueError(f"Raw term {name!r} changed organizer bank within an update.")
        _add_gradients(row["gradients"], gradients)
        row["samples"] += samples
        if objective_value is not None:
            if not math.isfinite(float(objective_value)):
                raise FloatingPointError(f"Raw objective value for {name!r} is nonfinite.")
            row["objective_value_sum"] += float(objective_value)
            row["objective_value_samples"] += samples
        coverage = self._current_vjp_coverage.setdefault(name, {"inventory_parameters": len(inventory), "touched_parameters": 0})
        coverage["touched_parameters"] = len(set(row["gradients"]))

    def add_raw_gradients(
        self,
        name: str,
        bank: str,
        gradients: Mapping[str, torch.Tensor],
        *,
        samples: int = 1,
        objective_value: float | None = None,
    ) -> None:
        """Inject a named sum-VJP; ``samples`` is its later averaging divisor."""
        if not isinstance(gradients, Mapping):
            raise TypeError("Raw named gradients must be a parameter-name mapping.")
        self._accumulate(str(name), bank, self._normalize_injected_names(bank, gradients), int(samples),
                         objective_value=objective_value)

    def collect(
        self,
        predictions: Any,
        per_case_native_losses: Any,
        epoch: int,
        *,
        case_ids: Sequence[Any] | None = None,
        raw_terms: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Collect case-mean expected-work, score-function, and optional VJPs."""
        if type(epoch) is not int or epoch < 1:
            raise ValueError("collect epoch must be a positive integer.")
        ids = self._case_ids(predictions, per_case_native_losses, case_ids)
        native_losses = self._loss_vector(per_case_native_losses, ids)
        prepared = self._prepared(predictions)

        # Calibration needs a replayed native reference VJP. Ordinary updates
        # already have the native conditional gradient in parameter .grad, so
        # do not duplicate a whole-model VJP on every microbatch.
        if self._calibrating:
            native_sum = native_losses.sum()
            native_gradients = self._vjp(native_sum, self._named_model_parameters())
            _add_gradients(self._native_calibration_gradients, native_gradients)
            self._native_calibration_cases += len(ids)

        summaries: dict[str, Any] = {"case_ids": list(ids), "epoch": epoch, "banks": {}}
        for bank in _BANKS:
            rows = self._bank_rows(prepared, bank)
            if len(rows) != len(ids):
                raise ValueError(f"Prepared {bank} scene count differs from the case-loss vector.")
            eligible = [(case_index, row, row.frontier) for case_index, row in enumerate(rows)
                        if getattr(row, "frontier", None) is not None]
            self._current_case_coverage[bank] += len(eligible)
            if eligible:
                inventory = self._inventory(bank)
                normalized_expectations = []
                score_terms = []
                for case_index, row, frontier in eligible:
                    count = int(getattr(row, "present_count", 0))
                    if count < 1:
                        raise ValueError("An active frontier must contain at least one original record.")
                    expected = getattr(frontier, "expected_K", None)
                    log_prob = getattr(frontier, "log_prob", None)
                    if not torch.is_tensor(expected) or not torch.is_tensor(log_prob):
                        raise TypeError("Each active frontier must expose expected_K and log_prob tensors.")
                    if not bool(torch.isfinite(expected.detach())) or not bool(torch.isfinite(log_prob.detach())):
                        raise FloatingPointError("Frontier expected_K and log_prob must be finite.")
                    if bool(getattr(frontier, "capacity_active", True)):
                        normalized_expectations.append(expected.reshape(()) / count)
                    if bool(getattr(frontier, "score_function_active", True)):
                        baseline = self._baseline.get(ids[case_index], 0.0)
                        advantage = native_losses[case_index].detach() - native_losses.new_tensor(baseline)
                        # The REINFORCE reward is the native physical loss alone;
                        # capacity and injected functional terms never enter it.
                        score_terms.append(advantage * log_prob.reshape(()))
                    self._pending_baseline.setdefault(ids[case_index], []).append(
                        float(native_losses[case_index].detach().cpu())
                    )
                capacity_cases = len(normalized_expectations)
                score_cases = len(score_terms)
                capacity_sum = torch.stack(normalized_expectations).sum() if capacity_cases else None
                score_sum = torch.stack(score_terms).sum() if score_cases else None
                if capacity_sum is not None:
                    capacity_gradients = self._vjp(capacity_sum, inventory)
                    self._accumulate(f"capacity_{bank}", bank, capacity_gradients, capacity_cases,
                                     objective_value=float(capacity_sum.detach().cpu()))
                if score_sum is not None:
                    score_gradients = self._vjp(score_sum, inventory)
                    self._accumulate(f"reinforce_{bank}", bank, score_gradients, score_cases,
                                     objective_value=float(score_sum.detach().cpu()))
                summaries["banks"][bank] = {
                    "eligible_cases": len(eligible),
                    "capacity_active_cases": capacity_cases,
                    "score_function_active_cases": score_cases,
                    "normalized_expected_k_sum": 0.0 if capacity_sum is None else float(capacity_sum.detach().cpu()),
                    "native_score_sum": 0.0 if score_sum is None else float(score_sum.detach().cpu()),
                    "frontiers_with_policy_decisions": sum(
                        int(bool(getattr(frontier, "reached_internal", ()))) for _, _, frontier in eligible
                    ),
                }
            else:
                summaries["banks"][bank] = {
                    "eligible_cases": 0,
                    "capacity_active_cases": 0,
                    "score_function_active_cases": 0,
                    "normalized_expected_k_sum": 0.0,
                    "native_score_sum": 0.0,
                    "frontiers_with_policy_decisions": 0,
                }

        if raw_terms:
            for name, row in raw_terms.items():
                if not isinstance(row, Mapping) or "bank" not in row or "gradients" not in row:
                    raise ValueError("Each raw_terms row needs bank and gradients fields.")
                self.add_raw_gradients(str(name), str(row["bank"]), row["gradients"],
                                       samples=int(row.get("samples", len(ids))),
                                       objective_value=row.get("objective_value"))
        self.last_collection = summaries
        return None

    def _averaged_terms(self) -> dict[str, dict[str, Any]]:
        result = {}
        for name, row in self._accumulators.items():
            denominator = int(row["samples"])
            result[name] = {
                "bank": row["bank"],
                "samples": denominator,
                "gradients": {key: value / denominator for key, value in row["gradients"].items()},
                "objective_value": (row["objective_value_sum"] / row["objective_value_samples"]
                                    if row.get("objective_value_samples", 0) else None),
            }
        return result

    def _clear_update(self) -> None:
        self._accumulators.clear()
        self._pending_baseline.clear()
        self._native_calibration_gradients.clear()
        self._native_calibration_cases = 0
        self._current_case_coverage = {bank: 0 for bank in _BANKS}
        self._current_vjp_coverage.clear()

    def _update_baselines(self) -> None:
        for case_id, observed in self._pending_baseline.items():
            if not observed:
                continue
            current = sum(observed) / len(observed)
            previous = self._baseline.get(case_id)
            self._baseline[case_id] = current if previous is None else (
                self.baseline_decay * previous + (1.0 - self.baseline_decay) * current
            )
        self.state["baselines"] = dict(self._baseline)

    def before_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        losses: dict[str, float],
        work: dict[str, Any],
        *,
        epoch: int,
        update_index: int,
    ) -> dict[str, Any]:
        """Add calibrated organizer gradients before engine clipping and step."""
        if self._calibrating:
            return {}
        if model is not self.model:
            raise ValueError("before_step received a model other than the objective owner.")
        if epoch in self.calibration_epochs and epoch not in self.state["calibrated_epochs"]:
            provider = self.provider
            if provider is not None and callable(getattr(provider, "contextual_calibration_panel", None)):
                self.calibrate(provider, epoch, optimizer=optimizer)

        named = self._named_model_parameters()
        native = {name: parameter.grad.detach().clone() for name, parameter in named.items()
                  if parameter.grad is not None}
        native_full_norm = gradient_norm(native)
        terms = self._averaged_terms()
        combined: dict[str, torch.Tensor] = {}
        term_records: dict[str, Any] = {}
        for name in sorted(set(terms) | set(self.state["coefficients"])):
            row = terms.get(name, {"bank": name.rsplit("_", 1)[-1], "samples": 0, "gradients": {}})
            bank = row["bank"]
            inventory = self._inventory(bank)
            names = tuple(inventory)
            gradients = row["gradients"]
            reference = {key: value for key, value in native.items() if key in inventory}
            balance = gradient_balance(reference, gradients, names)
            coefficient = float(self.state["coefficients"].get(name, 0.0))
            term_records[name] = {
                **balance,
                "bank": bank,
                "samples": int(row["samples"]),
                "coefficient": coefficient,
                "inventory_parameters": len(names),
                "raw_vjp_coverage": int(self._current_vjp_coverage.get(name, {}).get("touched_parameters", 0)),
            }
            if not math.isfinite(coefficient) or coefficient < 0:
                raise FloatingPointError(f"Invalid calibrated coefficient for {name!r}.")
            if coefficient == 0.0:
                continue
            for parameter_name, gradient in gradients.items():
                contribution = gradient * coefficient
                if parameter_name in combined:
                    combined[parameter_name].add_(contribution)
                else:
                    combined[parameter_name] = contribution.clone()

        raw_combined_norm = gradient_norm(combined)
        structural_multiplier = self._structural_multiplier(epoch)
        scheduled_combined_norm = raw_combined_norm * structural_multiplier
        cap = self.full_gradient_cap * native_full_norm
        scale = min(1.0, cap / scheduled_combined_norm) if scheduled_combined_norm > 0.0 else 1.0
        weighted = {
            name: gradient * structural_multiplier * scale
            for name, gradient in combined.items()
        }
        for name, gradient in weighted.items():
            if not bool(torch.count_nonzero(gradient.detach())):
                continue
            if name not in named:
                raise ValueError(f"Organizer gradient references non-model parameter {name!r}.")
            if named[name].grad is None:
                named[name].grad = gradient
            else:
                named[name].grad.add_(gradient)

        record = {
            "epoch": int(epoch),
            "update_index": int(update_index),
            "structural_multiplier": structural_multiplier,
            "native_full_gradient_norm": native_full_norm,
            "organizer_raw_combined_norm": raw_combined_norm,
            "organizer_scheduled_gradient_norm": scheduled_combined_norm,
            "organizer_weighted_gradient_norm": scheduled_combined_norm * scale,
            "organizer_cap": cap,
            "organizer_cap_scale": scale,
            "banks": dict(self._current_case_coverage),
            "terms": term_records,
        }
        self.state["last_update"] = copy.deepcopy(record)
        for name, detail in term_records.items():
            work[f"contextual_{name}_raw_norm"] = detail["term_norm"]
            work[f"contextual_{name}_coefficient"] = detail["coefficient"]
            work[f"contextual_{name}_coverage"] = detail["samples"]
        work["contextual_native_full_gradient_norm"] = native_full_norm
        work["contextual_organizer_gradient_norm"] = scheduled_combined_norm * scale
        work["contextual_structural_multiplier"] = structural_multiplier
        work["contextual_organizer_cap_scale"] = scale
        for name, value in terms.items():
            if value["objective_value"] is not None:
                losses[f"contextual_{name}"] = float(
                    value["objective_value"] * self.state["coefficients"].get(name, 0.0)
                    * structural_multiplier * scale
                )

        self._update_baselines()
        self._clear_update()
        return record

    def before_optimizer_step(self, model, optimizer, losses, work, *, epoch, update_index):
        """Maintained-engine hook alias."""
        return self.before_step(model, optimizer, losses, work, epoch=epoch, update_index=update_index)

    def _panel(self, provider: Any) -> tuple[str, ...]:
        cases = getattr(provider, "training_cases", None)
        if not cases:
            raise TypeError("Calibration provider must expose selected TRAIN cases.")
        return select_calibration_panel(cases, limit=min(self.calibration_max_cases, len(cases)))

    def calibrate(
        self,
        provider: Any,
        epoch: int,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        panel_runner: Any | None = None,
    ) -> dict[str, Any]:
        """Measure coefficients on a fixed TRAIN-only panel, restoring all state.

        The runner contract is ``runner(provider, case_ids, epoch, runtime)``;
        it returns an iterable of ``(predictions, per_case_native_losses,
        case_ids)`` rows or calls ``runtime.collect`` directly. No optimizer is
        stepped by this method. It snapshots model, provider, optimizer, RNG,
        gradients, and runtime state so the panel replay leaves training exact.
        """
        if type(epoch) is not int or epoch not in self.calibration_epochs:
            raise ValueError(f"epoch must be one of {self.calibration_epochs}.")
        if epoch in self.state["calibrated_epochs"]:
            return copy.deepcopy(self.state["calibrations"][-1])
        runner = panel_runner
        if runner is None:
            runner = getattr(provider, "contextual_calibration_panel", None)
        if not callable(runner):
            raise ValueError("Calibration requires a provider contextual_calibration_panel runner.")

        from honf_runtime.unified_training import capture_rng_state, restore_rng_state

        case_ids = self._panel(provider)
        model_state = copy.deepcopy(self.model.state_dict())
        training_state_reader = getattr(provider, "training_state_dict", None)
        training_state_loader = getattr(provider, "load_training_state_dict", None)
        if callable(training_state_reader) != callable(training_state_loader):
            raise TypeError("Calibration provider must expose both training state read and restore hooks.")
        provider_state = copy.deepcopy(training_state_reader()) if callable(training_state_reader) else None
        optimizer_state = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
        rng_state = capture_rng_state()
        module_modes = {name: module.training for name, module in self.model.named_modules()}
        gradients_enabled = torch.is_grad_enabled()
        provider_counters = {
            name: copy.deepcopy(getattr(provider, name))
            for name in ("training_samples", "validation_samples", "_pending", "_segment_started", "_budget_binding")
            if hasattr(provider, name)
        }
        organization_clock = (
            getattr(self.core, "_organization_epoch", None),
            getattr(self.core, "_organization_update", None),
        )
        model_grads = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                       for name, parameter in self.model.named_parameters()}
        runtime_state = self.state_dict()
        raw_accumulators = _accumulator_snapshot(self._accumulators)
        pending = copy.deepcopy(self._pending_baseline)
        native_grads = _clone_gradients(self._native_calibration_gradients)
        native_cases = self._native_calibration_cases
        coverage = copy.deepcopy(self._current_case_coverage)
        vjp_coverage = copy.deepcopy(self._current_vjp_coverage)
        last_collection = copy.deepcopy(self.last_collection)
        calibration_extra_before = copy.deepcopy(self.calibration_extra)
        self._accumulators = {}
        self._pending_baseline = {}
        self._native_calibration_gradients = {}
        self._native_calibration_cases = 0
        self._current_case_coverage = {bank: 0 for bank in _BANKS}
        self._current_vjp_coverage = {}
        self.calibration_extra = []
        self._calibrating = True
        try:
            result = runner(provider, case_ids, epoch, self)
            if result is not None:
                rows = (result,) if isinstance(result, tuple) and len(result) in (2, 3) else result
                for item in rows:
                    if not isinstance(item, Sequence) or len(item) not in (2, 3):
                        raise ValueError("Calibration panel rows must be (predictions, losses[, case_ids]).")
                    predictions, losses = item[0], item[1]
                    row_ids = item[2] if len(item) == 3 else None
                    self.collect(predictions, losses, epoch, case_ids=row_ids)

            native_average = {
                name: gradient / self._native_calibration_cases
                for name, gradient in self._native_calibration_gradients.items()
            } if self._native_calibration_cases else {}
            averaged_terms = self._averaged_terms()
            report_terms: dict[str, Any] = {}
            coefficients = {}
            all_term_names = sorted(set(_BASE_TERMS) | set(averaged_terms))
            for name in all_term_names:
                bank = averaged_terms.get(name, {}).get("bank", name.rsplit("_", 1)[-1])
                inventory = self._inventory(bank)
                names = tuple(inventory)
                native_bank = {key: value for key, value in native_average.items() if key in inventory}
                row = averaged_terms.get(name, {"gradients": {}, "samples": 0})
                balance = gradient_balance(native_bank, row["gradients"], names)
                native_norm, term_norm = balance["native_norm"], balance["term_norm"]
                coefficient = (self.target_fraction * native_norm / term_norm
                               if native_norm > 0.0 and term_norm > 0.0 else 0.0)
                if not math.isfinite(coefficient):
                    raise FloatingPointError(f"Nonfinite calibration coefficient for {name!r}.")
                coefficients[name] = coefficient
                report_terms[name] = {
                    **balance,
                    "bank": bank,
                    "samples": int(row["samples"]),
                    "coefficient": coefficient,
                    "target_fraction": self.target_fraction,
                    "zero_norm_rule": "coefficient_zero_if_native_or_term_norm_is_zero",
                }

            report = {
                "schema": self.SCHEMA,
                "epoch": epoch,
                "structural_multiplier": self._structural_multiplier(epoch),
                "calibration_coefficients_are_pre_ramp": True,
                "case_ids": list(case_ids),
                "case_count": len(case_ids),
                "native_cases_collected": self._native_calibration_cases,
                "optimizer_steps": 0,
                "terms": report_terms,
                "functional_probe_exposure": copy.deepcopy(self.calibration_extra),
                "model_provider_optimizer_rng_restored": True,
                "budget_wall_time_restored": False,
                "wall_time_policy": "actual cumulative budget time includes no-step replay and failed attempts",
            }
        finally:
            self._calibrating = False
            self.model.load_state_dict(model_state, strict=True)
            if provider_state is not None:
                training_state_loader(provider_state)
            if (organization_clock[0] is not None and organization_clock[1] is not None
                    and callable(getattr(self.core, "set_organization_clock", None))):
                self.core.set_organization_clock(*organization_clock)
            if optimizer is not None and optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            for name, parameter in self.model.named_parameters():
                saved_gradient = model_grads[name]
                parameter.grad = None if saved_gradient is None else saved_gradient.to(parameter)
            for name, module in self.model.named_modules():
                module.training = module_modes[name]
            torch.set_grad_enabled(gradients_enabled)
            for name, value in provider_counters.items():
                setattr(provider, name, value)
            restore_rng_state(rng_state)
            self.load_state_dict(runtime_state)
            self._accumulators = raw_accumulators
            self._pending_baseline = pending
            self._native_calibration_gradients = native_grads
            self._native_calibration_cases = native_cases
            self._current_case_coverage = coverage
            self._current_vjp_coverage = vjp_coverage
            self.last_collection = last_collection
            self.calibration_extra = calibration_extra_before

        self.state["coefficients"].update(coefficients)
        self.state["calibrated_epochs"].append(epoch)
        self.state["calibrated_epochs"].sort()
        self.state["calibrations"].append(copy.deepcopy(report))
        return report

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "coefficients": {str(key): float(value) for key, value in self.state["coefficients"].items()},
            "calibrated_epochs": [int(value) for value in self.state["calibrated_epochs"]],
            "baselines": {str(key): float(value) for key, value in self._baseline.items()},
            "calibrations": copy.deepcopy(self.state["calibrations"]),
            "last_update": copy.deepcopy(self.state["last_update"]),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != self.SCHEMA:
            raise ValueError("Unsupported contextual objective state schema.")
        coefficients = state.get("coefficients", {})
        baselines = state.get("baselines", {})
        if not isinstance(coefficients, Mapping) or not isinstance(baselines, Mapping):
            raise TypeError("Contextual objective coefficients/baselines must be mappings.")
        for name, value in coefficients.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"Invalid contextual coefficient for {name!r}.")
        for case_id, value in baselines.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Invalid native EMA baseline for {case_id!r}.")
        self._baseline = {str(key): float(value) for key, value in baselines.items()}
        self.state = {
            "schema": self.SCHEMA,
            "coefficients": {str(key): float(value) for key, value in coefficients.items()},
            "calibrated_epochs": [int(value) for value in state.get("calibrated_epochs", ())],
            "baselines": dict(self._baseline),
            "calibrations": copy.deepcopy(list(state.get("calibrations", ()))),
            "last_update": copy.deepcopy(dict(state.get("last_update", {}))),
        }
        # Checkpoint/provider restoration is also the TrainingEngine's OOM
        # rollback boundary. Discard any partial macro-update VJPs and reward
        # observations so a replay cannot double-count failed microbatches.
        self._clear_update()
        self.last_collection = None
