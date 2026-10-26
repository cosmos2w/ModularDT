"""Native-only Thermal provider for Round13-3 interaction-reused member routing."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from honf_runtime.unified_training import OptimizerGroupSpec, ScheduleSpec
from ..member_routing import ThermalInteractionReusedMemberRoutingAdapter
from .content_hypergraph_task import ThermalContentPredictions, THERMAL_CONTENT_LOSS_WEIGHTS
from .contextual_joint_task import ThermalContextualJointTask
from .joint_task import _load_data_binding, _read_selected_cases, validate_full_train_followup_membership
from .contextual_joint_task import _structure_only
from ..data.datasets import fit_global_normalizer
from ..data.development_split import development_case_ids, read_case_catalog


WORKFLOW = "interaction_reused_member_routing_joint_training_v1"
PROVIDER_SCHEMA = "thermal_interaction_reused_member_routing_provider_v1"
CONFIG_SECTION_SCHEMA = "thermal_member_routing_r13_3_v1"


class ThermalInteractionReusedMemberRoutingTask(ThermalContextualJointTask):
    """Use only the maintained native Thermal observation and objective path."""

    def __init__(self, *, model, training_cases, validation_cases, manifest, data_path,
                 recipe, device, manifest_path=None, catalog=(), source_binding=None):
        super().__init__(model=model, training_cases=training_cases, validation_cases=validation_cases,
                         manifest=manifest, data_path=data_path, recipe=recipe, device=device,
                         manifest_path=manifest_path, catalog=catalog, source_binding=source_binding,
                         native_only=True)
        self.native_weights = dict(recipe["objective"]["native_weights"])
        if set(self.native_weights) != set(THERMAL_CONTENT_LOSS_WEIGHTS) - {"response"}:
            raise ValueError("Round13-3 native objective must retain all maintained Thermal receiver roles.")
        if any(not math.isfinite(float(value)) or float(value) < 0
               for value in self.native_weights.values()):
            raise ValueError("Round13-3 native objective weights must be finite and nonnegative.")

    def identity_payload(self):
        payload = super().identity_payload()
        payload.update({
            "schema": PROVIDER_SCHEMA,
            "workflow": WORKFLOW,
            "architecture": self.model.architecture,
            "member_routing_schema": self.model.CORE_SCHEMA,
            "adapter_state": self.model.state_payload(),
            "native_only_training": True,
            "response_training_family_ids": [],
            "response_supervision": None,
            "objective": {
                "native_weights": copy.deepcopy(self.native_weights),
                "native_weights_fixed": True,
                "joint_fields": ["u", "v", "p", "temperature"],
                "prescribed_heat_in_context": True,
                "response_training_families": [],
                "response_fit": False,
                "organizer_objective": False,
            },
            "normalization_fit_scope": "selected TRAIN only",
            "test_read": False,
            "solver_attempts": 0,
            "recipe": copy.deepcopy(self.recipe),
        })
        return payload

    def _schedule_payload(self):
        lr = float(self.recipe["optimizer"]["lr"])
        schedule_horizon = int(self.recipe["training"]["optimizer_schedule_horizon"])
        return {
            "peak_lr": lr,
            "warmup_start_lr": lr,
            "warmup_epochs": 0,
            "hold_through_epoch": schedule_horizon - 1,
            "total_epochs": schedule_horizon,
            "final_lr": lr,
        }

    def optimizer_groups(self, model, arm, stage):
        del stage
        if model is not self.model or arm != model.architecture:
            raise ValueError("Wrong interaction-reused member-routing optimizer owner.")
        names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if not names:
            raise ValueError("Member-routing model has no trainable parameters.")
        optimizer = self.recipe["optimizer"]
        return (OptimizerGroupSpec(
            "joint", names,
            ScheduleSpec(**self._schedule_payload()),
            weight_decay=float(optimizer["weight_decay"]),
            betas=tuple(map(float, optimizer["betas"])),
            eps=float(optimizer["eps"]),
        ),)

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        del auxiliary_state
        if phase != "joint":
            raise ValueError("Member-routing native field loss is defined only for the joint phase.")
        from honf_runtime.unified_training import LossTerm
        roles = self._role_losses(predictions, targets)
        return {name: LossTerm(values.sum(), len(targets.case_ids), float(self.native_weights[name]))
                for name, values in roles.items()}

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch=None, arm=None):
        del auxiliary_state, batch, arm
        from honf_runtime.unified_training import LossTerm
        roles = self._role_losses(predictions, targets)
        return {name: LossTerm(values.sum(), len(targets.case_ids), float(self.native_weights[name]))
                for name, values in roles.items()}

    def before_optimizer_step(self, model, optimizer, losses, work, *, epoch, update_index):
        # The maintained native-only branch also enforces its ordinary wall
        # budget. It never enters response or organizer callbacks.
        super().before_optimizer_step(model, optimizer, losses, work,
                                      epoch=epoch, update_index=update_index)

    def training_state_dict(self):
        return {**super().training_state_dict(),
                "schema": "thermal_member_routing_native_training_state_v1"}

    def load_training_state_dict(self, state):
        if not isinstance(state, Mapping) or state.get("schema") != "thermal_member_routing_native_training_state_v1":
            raise ValueError("Saved member-routing provider state is malformed.")
        if min(int(state.get(name, 0)) for name in
               ("primary_updates", "training_samples", "validation_samples")) < 0:
            raise ValueError("Saved member-routing provider counters cannot be negative.")
        elapsed = float(state.get("training_elapsed_seconds", 0.0))
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Saved member-routing elapsed time must be finite and nonnegative.")
        restored = dict(state)
        restored.pop("schema")
        super().load_training_state_dict(restored)

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0,
                       temperature=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported member-routing native prediction options: {sorted(kwargs)}")
        if model is not self.model or execution_mode != model.architecture:
            raise ValueError("Wrong member-routing model/arm.")
        prepared = model.prepare_native(
            scene.structure,
            receivers.fluid_xy,
            physical_heat=receivers.heat,
            local_query_points=receivers.local_query_points,
            ntheta=receivers.ntheta,
            scene_key=scene.case_ids,
        )
        main, decode, _output = model.apply_native_with_work_counts(prepared, receivers.heat)
        work = {
            "native_fluid_queries": int(receivers.fluid_xy.shape[0] * receivers.fluid_xy.shape[1]),
            "native_union_rows": int(prepared.union.unique_counts.sum()),
            "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
        }
        work.update({f"core_prepare_primary_{name}": int(value)
                     for name, value in prepared.organization.work_counts.items()})
        work.update({f"core_decode_primary_{name}": int(value) for name, value in decode.items()})
        prediction = ThermalContentPredictions(prepared, main, work=work)
        return prediction, {"work": work}

    def _prepare_for_probe(self, model, batch, **prepare_options):
        return model.prepare_native(
            batch.scene_inputs.structure,
            batch.receivers.fluid_xy,
            physical_heat=batch.receivers.heat,
            local_query_points=batch.receivers.local_query_points,
            ntheta=batch.receivers.ntheta,
            scene_key=batch.scene_inputs.case_ids,
            **prepare_options,
        )

    @staticmethod
    def _query_subset(query, query_indices):
        from honf_forward_core.content_hypergraph.contracts import QueryBatch

        indices = torch.as_tensor(query_indices, device=query.coordinates.device, dtype=torch.long)
        if (indices.ndim != 1 or not indices.numel() or bool((indices < 0).any())
                or bool((indices >= query.num_queries).any()) or len(set(indices.tolist())) != indices.numel()):
            raise ValueError("Route-trace query indices must be unique, nonempty in-range rows.")
        return QueryBatch(
            coordinates=query.coordinates.index_select(1, indices),
            features=None if query.features is None else query.features.index_select(1, indices),
            query_mask=None if query.query_mask is None else query.query_mask.index_select(1, indices),
            query_ids=None if query.query_ids is None else query.query_ids.index_select(1, indices),
            scene_key=query.scene_key,
            coordinate_frame_key=query.coordinate_frame_key,
            receiver_key=query.receiver_key,
        )

    @staticmethod
    def _as_numpy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _trace_rows(cls, model, prepared, query_indices, intervention=None):
        queries = prepared.query_batch
        indices = tuple(int(value) for value in query_indices)
        subset = cls._query_subset(queries, indices)
        captured = model.core.capture_interaction_routing_trace(
            prepared.organization, subset, intervention=intervention,
        )
        if not isinstance(captured, Mapping):
            raise TypeError("Core interaction route trace must be a mapping of named tensors.")

        def row_values(name, batch_index, query_index=None):
            value = captured.get(name)
            if value is None:
                return None
            array = cls._as_numpy(value)
            if query_index is not None and array.ndim >= 2 and array.shape[1] == len(indices):
                array = array[batch_index, query_index]
            elif array.ndim and array.shape[0] == subset.batch_size:
                array = array[batch_index]
            return array

        def scalar(value):
            value = cls._as_numpy(value)
            return value.item() if value.ndim == 0 else value.tolist()

        def member_rows(ids, weights, eligible=None):
            if ids is None or weights is None:
                return []
            ids, weights = cls._as_numpy(ids).reshape(-1), cls._as_numpy(weights).reshape(-1)
            eligible = (np.ones_like(weights, dtype=bool) if eligible is None
                        else cls._as_numpy(eligible).reshape(-1).astype(bool))
            return [{"original_id": scalar(ids[index]), "weight": float(weights[index]),
                     "candidate": bool(eligible[index])}
                    for index in range(len(ids)) if eligible[index] or weights[index] > 0]

        org = prepared.organization
        metadata = []
        for item in getattr(org, "context_stage_metadata", ()):
            metadata.append({
                "stage": str(item.get("stage", item.get("stage_identity", ""))),
                "route": str(item.get("route", "")),
                "source_state_version": str(item.get("source_state_version", "")),
                "application": str(item.get("application", "contextual_value_reduction")),
                "eligible_pair_rows": int(item.get("eligible_edges", 0)),
                "executed_positive_value_edges": int(item.get("positive_value_edges", 0)),
            })
        rows = []
        for batch_index in range(subset.batch_size):
            module_ids = row_values("original_module_ids", batch_index)
            environment_ids = row_values("original_environment_ids", batch_index)
            B_M = row_values("B_M", batch_index)
            B_E = row_values("B_E", batch_index)
            query_ids = (None if subset.query_ids is None
                         else cls._as_numpy(subset.query_ids)[batch_index])
            coordinates = cls._as_numpy(subset.coordinates)[batch_index]
            for local_index, original_index in enumerate(indices):
                def select(name):
                    return row_values(name, batch_index, local_index)

                proposal = select("proposal_weights_M")
                candidate_M = select("candidate_mask_M")
                candidate_E = select("candidate_mask_E")
                beta_M, beta_E = select("member_weights_M"), select("member_weights_E")
                address_rows = []
                for address, address_id in enumerate(np.asarray(module_ids).reshape(-1)):
                    m_members = np.flatnonzero(cls._as_numpy(B_M)[address] > 0)
                    e_members = np.flatnonzero(cls._as_numpy(B_E)[address] > 0)
                    address_rows.append({
                        "address_id": scalar(address_id),
                        "M_members": [{"original_id": scalar(np.asarray(module_ids).reshape(-1)[index]),
                                       "coefficient": float(cls._as_numpy(B_M)[address, index])}
                                      for index in m_members],
                        "E_members": [{"original_id": scalar(np.asarray(environment_ids).reshape(-1)[index]),
                                       "coefficient": float(cls._as_numpy(B_E)[address, index])}
                                      for index in e_members],
                    })
                rows.append({
                    "query_index": int(original_index),
                    "query_id": None if query_ids is None else scalar(np.asarray(query_ids).reshape(-1)[local_index]),
                    "query_coordinates": cls._as_numpy(coordinates)[local_index].astype(float).tolist(),
                    "original_module_inventory": [
                        {"original_id": scalar(value), "coordinates": scalar(org.scene.module_coordinates[batch_index, index]),
                         "present": bool(org.scene.module_mask[batch_index, index])}
                        for index, value in enumerate(np.asarray(module_ids).reshape(-1))],
                    "original_environment_inventory": [
                        {"original_id": scalar(value), "coordinates": scalar(org.scene.environment_coordinates[batch_index, index]),
                         "present": bool(org.scene.environment_mask[batch_index, index])}
                        for index, value in enumerate(np.asarray(environment_ids).reshape(-1))],
                    "null_weights": {"address": float(select("proposal_null")),
                                     "M": float(select("member_null_M")), "E": float(select("member_null_E"))},
                    "proposal_members_M": member_rows(module_ids, proposal,
                                                       cls._as_numpy(proposal) > 0),
                    "candidate_members_M": member_rows(module_ids, np.zeros_like(cls._as_numpy(module_ids), dtype=float), candidate_M),
                    "candidate_members_E": member_rows(environment_ids, np.zeros_like(cls._as_numpy(environment_ids), dtype=float), candidate_E),
                    "selected_value_members_M": member_rows(module_ids, beta_M, cls._as_numpy(beta_M) > 0),
                    "selected_value_members_E": member_rows(environment_ids, beta_E, cls._as_numpy(beta_E) > 0),
                    "omitted_final_member_ids_M": [scalar(value) for value, weight in zip(np.asarray(module_ids).reshape(-1), cls._as_numpy(beta_M).reshape(-1)) if weight == 0],
                    "omitted_final_member_ids_E": [scalar(value) for value, weight in zip(np.asarray(environment_ids).reshape(-1), cls._as_numpy(beta_E).reshape(-1)) if weight == 0],
                    "applied_address_support": address_rows,
                    "context_stage_provenance": metadata,
                    "support_scope_note": (
                        "Decoder member-read support is not the complete input influence graph; upstream directed "
                        "contextual dependencies, including E-to-E exchange, remain separate."
                    ),
                })
        if len(rows) != len(indices) * subset.batch_size:
            raise ValueError("Core route trace did not return all selected original-query rows.")
        return captured, tuple(rows)

    @staticmethod
    def _trace_selected_counts(trace):
        counts = {}
        for role in ("M", "E"):
            rows = trace.get(f"selected_value_indices_{role}")
            counts[role] = int(rows.shape[0]) if torch.is_tensor(rows) else 0
        return {"scope": "three fixed query rows", "measure": "exact-positive beta rows", **counts}

    @classmethod
    def _route_weight_changes(cls, baseline, intervention):
        result = {"comparison": "baseline minus intervention trace over the three fixed query rows"}
        for label, key in (("proposal_pi", "proposal_weights_M"),
                           ("M_cheap_scores", "member_logits_M"), ("E_cheap_scores", "member_logits_E"),
                           ("M_beta", "member_weights_M"), ("E_beta", "member_weights_E"),
                           ("B_M", "B_M"), ("B_E", "B_E")):
            before = baseline.get(key)
            after = intervention.get(key)
            if before is None or after is None:
                continue
            left, right = cls._as_numpy(before).astype(np.float64), cls._as_numpy(after).astype(np.float64)
            difference = right - left
            result[label] = {"l1": float(np.abs(difference).sum()),
                             "max_abs": float(np.abs(difference).max()) if difference.size else 0.0}
        for role in ("M", "E"):
            source_ids = cls._as_numpy(
                baseline[f"original_{'module' if role == 'M' else 'environment'}_ids"]
            ).reshape(-1)
            before = cls._as_numpy(baseline[f"selected_value_indices_{role}"])
            after = cls._as_numpy(intervention[f"selected_value_indices_{role}"])
            def selected_set(rows):
                return {(int(query), int(source_ids[int(source)])) for _batch, query, source in rows}
            left, right = selected_set(before), selected_set(after)
            result[f"{role}_selected_ids"] = {
                "baseline_count": len(left), "intervention_count": len(right),
                "changed_query_member_pairs": len(left.symmetric_difference(right)),
            }
            result[f"{role}_candidate_ids_unchanged"] = bool(np.array_equal(
                cls._as_numpy(baseline[f"candidate_mask_{role}"]),
                cls._as_numpy(intervention[f"candidate_mask_{role}"]),
            ))
        return result

    def member_routing_trace(self, model, prediction, query_indices):
        _trace, rows = self._trace_rows(model, prediction.native_prepared, query_indices)
        if len(rows) != 3:
            raise ValueError("member_routing_trace must return exactly three original-query route rows.")
        return rows

    def member_routing_probe(self, model, batch, base_prediction, query_indices, probe_name):
        base_prepared = base_prediction.native_prepared
        core = model.core
        subset = self._query_subset(base_prepared.query_batch, query_indices)
        intervention, prepare_options = core.member_routing_intervention(
            probe_name, base_prepared.organization, subset,
        )
        baseline_trace, _baseline_rows = self._trace_rows(model, base_prepared, query_indices)
        geometry_calibration = None
        if probe_name == "geometry_b_comparable_counts":
            # A bounded no-fit count match on the declared three queries.
            # Geometry coefficients stay in [0,1] on the actual allowed IDs;
            # no field error or physical reference chooses the control.
            geometry_calibration = {}
            target_counts = self._trace_selected_counts(baseline_trace)
            for role in ("M", "E"):
                key = f"fixed_B_{role}"
                geometry = intervention[key]
                original = getattr(base_prepared.organization, f"B_{role}")
                trials = []
                best = None
                for floor in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
                    coefficients = torch.where(original > 0,
                        geometry + floor * (1 - geometry), torch.zeros_like(geometry))
                    trial = {**intervention, key: coefficients}
                    captured = core.capture_interaction_routing_trace(
                        base_prepared.organization, subset, intervention=trial,
                    )
                    selected = int(captured[f"selected_value_indices_{role}"].shape[0])
                    trials.append({"geometry_floor": floor, "selected_rows": selected})
                    distance = abs(selected - int(target_counts[role]))
                    if best is None or distance < best[0]:
                        best = (distance, floor, coefficients, selected)
                    if distance == 0:
                        break
                intervention[key] = best[2]
                geometry_calibration[role] = {
                    "target_selected_rows": int(target_counts[role]),
                    "control_selected_rows": best[3], "geometry_floor": best[1],
                    "exact_count_match": best[0] == 0, "trials": trials,
                    "selection_basis": "three fixed query counts only; no field/reference error",
                }
        prepared = (self._prepare_for_probe(model, batch, **prepare_options)
                    if prepare_options else base_prepared)
        # Query-dependent fixed pi/candidate/logit controls are measured on the
        # same three-query slice as the route trace. Full native fields use only
        # query-independent controls so they can be streamed through core tiles.
        native_intervention = {
            key: value for key, value in intervention.items()
            if key in {"fixed_B_M", "fixed_B_E", "member_score_mode", "value_execution"}
        }
        native, decode, output = model.apply_native_with_work_counts(
            prepared, batch.receivers.heat,
            intervention=native_intervention or None,
        )
        prediction = ThermalContentPredictions(prepared, native, work=decode)
        trace_intervention = {} if prepare_options else intervention
        probe_trace, trace_rows = self._trace_rows(
            model, prepared, query_indices, intervention=trace_intervention,
        )
        if len(trace_rows) != 3:
            raise ValueError("Fixed routing probes must preserve three intervention-aware query traces.")
        base_counts = self._trace_selected_counts(baseline_trace)
        selected_counts = self._trace_selected_counts(probe_trace)
        route_changes = self._route_weight_changes(baseline_trace, probe_trace)
        work = dict(decode)
        work["context_message_reprepared"] = int(bool(prepare_options))
        if output.work_counts:
            work.update({f"probe_output_{key}": int(value) for key, value in output.work_counts.items()})
        return {
            "prediction": prediction,
            "work_counts": work,
            "trace": trace_rows,
            "baseline_selected_member_counts": base_counts,
            "selected_member_counts": selected_counts,
            "route_weight_changes": route_changes,
            "geometry_count_calibration": geometry_calibration,
            "full_native_selected_member_counts": {
                role: int(decode.get(f"query_{role}_selected_value_rows", 0))
                for role in ("M", "E")},
        }

    def member_routing_benchmark(self, model, batch, warmups=1, repetitions=3):
        import time
        from honf_runtime.unified_training import capture_rng_state, restore_rng_state
        from honf_runtime.contextual_frontier_training import _engine_config
        from honf_runtime.research_training import ResearchTrainingEngine

        if int(warmups) != 1 or int(repetitions) != 3:
            raise ValueError("Round13-3 native benchmark requires one warmup and three measured repeats.")
        device = next(model.parameters()).device
        use_cuda = device.type == "cuda"

        def sync():
            if use_cuda:
                torch.cuda.synchronize(device)

        phases = {name: [] for name in ("prepare", "cached_decode", "complete_native", "training_macro")}
        phase_work: dict[str, dict[str, int]] = {}
        baseline_mode = model.training
        model_snapshot = copy.deepcopy(model.state_dict())
        provider_snapshot = copy.deepcopy(self.training_state_dict())
        clock_snapshot = (int(getattr(model.core, "_joint_epoch", 0)),
                          int(getattr(model.core, "_joint_update_index", 0)))
        benchmark_rng = capture_rng_state()
        budget_binding = self._budget_binding
        self._budget_binding = None
        batch_case_ids = tuple(map(str, batch.targets.case_ids))
        if len(batch_case_ids) != 1 or batch_case_ids[0] not in self._train_index_by_id:
            raise ValueError("Native macro timing must be anchored on one selected TRAIN case.")
        if self.effective_batch_size != 48 or self.microbatch_size != 24:
            raise ValueError("Round13-3 training-macro benchmark requires two 24-case microbatches in a 48-case update.")
        macro_ids = (batch_case_ids[0], *tuple(case_id for case_id in sorted(self.training_case_ids)
                                               if case_id != batch_case_ids[0])[:47])
        if len(macro_ids) != self.effective_batch_size or len(set(macro_ids)) != self.effective_batch_size:
            raise ValueError("Could not construct a distinct full effective-batch native TRAIN macro.")
        macro_epoch = max(1, int(self.total_epochs))
        macro_batches = []
        engine = ResearchTrainingEngine(_engine_config(self.recipe), device=device, parent_age=0)
        parameter_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if not parameter_names:
            raise ValueError("Native macro benchmark found no trainable model parameters.")
        peak_baseline = int(torch.cuda.memory_allocated(device)) if use_cuda else 0
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)

        def count_work(mapping):
            return {str(key): int(value) for key, value in mapping.items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                    and not isinstance(value, bool) and math.isfinite(float(value))
                    and float(value).is_integer()}

        def normalize_core_work(mapping):
            raw = count_work(mapping)
            normalized = {key: value for key, value in raw.items()
                          if not key.startswith(("core_prepare_primary_", "core_decode_primary_"))}
            for key, value in raw.items():
                if key.startswith("core_decode_primary_"):
                    normalized[key.removeprefix("core_decode_primary_")] = value
            for key, value in raw.items():
                if key.startswith("core_prepare_primary_"):
                    normalized.setdefault(key.removeprefix("core_prepare_primary_"), value)
            return normalized

        def timed(name, pass_index, callback):
            sync()
            start_time = time.perf_counter()
            value = callback()
            sync()
            elapsed = time.perf_counter() - start_time
            if pass_index:
                phases[name].append(float(elapsed))
            return value

        def run_macro():
            nonlocal macro_batches
            model.load_state_dict(model_snapshot, strict=True)
            model.train(True)
            optimizer, specs = engine._make_optimizer(
                model, self, model.architecture, macro_epoch, None,
            )
            engine._set_schedules(optimizer, specs, macro_epoch)
            macro_batches = engine.prepare_update_microbatches(
                self, arm=model.architecture, phase="joint", epoch=macro_epoch,
                update_index=0, update_cases=macro_ids,
                sampling_dataset_id=self.sampling_dataset_id,
            )
            with torch.enable_grad():
                _losses, work, _denominators, _sampling = engine._run_update(
                    model, self, optimizer, macro_batches, phase="joint",
                    arm=model.architecture, epoch=macro_epoch, update_index=0,
                )
            changed = sum(not torch.equal(model_snapshot[name], model.state_dict()[name])
                          for name in parameter_names)
            if not changed:
                raise RuntimeError("Native full-batch macro optimizer step changed no model parameter.")
            return normalize_core_work(work), int(changed)

        try:
            model.eval()
            for pass_index in range(int(warmups) + int(repetitions)):
                def prepare_inference():
                    with torch.no_grad():
                        return self._prepare_for_probe(model, batch)

                prepared = timed("prepare", pass_index, prepare_inference)

                def cached_decode():
                    with torch.no_grad():
                        return model.core.decode(prepared.organization, prepared.query_batch)

                decoded = timed("cached_decode", pass_index, cached_decode)
                phase_work["cached_decode"] = count_work(decoded.work_counts or {})

                def complete_native():
                    with torch.no_grad():
                        prediction, auxiliary = self.predict_native(
                            model, batch.scene_inputs, batch.receivers, model.architecture,
                            "validation", epoch=macro_epoch,
                        )
                        self.validation_metrics(prediction, batch.targets, auxiliary)
                        return prediction

                completed = timed("complete_native", pass_index, complete_native)
                phase_work["complete_native"] = normalize_core_work(completed.work or {})
                model.zero_grad(set_to_none=True)
                model.train(True)
                macro_work, changed = timed("training_macro", pass_index, run_macro)
                model.load_state_dict(model_snapshot, strict=True)
                model.zero_grad(set_to_none=True)
                phase_work["training_macro"] = macro_work
                phase_work["training_macro_parameter_updates"] = {"changed_parameter_count": changed}
                model.eval()
        finally:
            model.load_state_dict(model_snapshot, strict=True)
            model.train(baseline_mode)
            model.core.set_organization_clock(*clock_snapshot)
            self.load_training_state_dict(provider_snapshot)
            restore_rng_state(benchmark_rng)
            self._budget_binding = budget_binding
            model.zero_grad(set_to_none=True)
        if any(not torch.equal(before, after) for before, after in zip(model_snapshot.values(), model.state_dict().values())):
            raise RuntimeError("Member-routing benchmark changed model state after disposable macro restoration.")
        peak_memory = int(torch.cuda.max_memory_allocated(device)) if use_cuda else 0
        if any(len(samples) != 3 for samples in phases.values()):
            raise RuntimeError("Member-routing benchmark did not retain three samples per phase.")
        native_work = phase_work["complete_native"]
        macro_work = phase_work["training_macro"]

        def selection_rows(work):
            return {
                role: {
                    "cheap_score_rows": int(work.get(f"query_{role}_cheap_score_rows", 0)),
                    "selected_value_rows": int(work.get(f"query_{role}_selected_value_rows", 0)),
                } for role in ("M", "E")
            }

        def context_rows(work):
            result = {}
            for route in ("mm", "me", "ee", "em"):
                prefix = f"context_{route}_"
                positive = int(work.get(prefix + "positive_value_edges", 0))
                result[route.upper()] = {
                    "scored_pair_rows": int(work.get(prefix + "score_rows", 0)),
                    "candidate_pair_rows": int(work.get(prefix + "eligible_edges", 0)),
                    "executed_pair_rows": positive,
                    "positive_value_edges": positive,
                }
            return result

        selected = selection_rows(native_work)
        macro_selected = selection_rows(macro_work)
        for role in ("M", "E"):
            selected[role]["training_macro_cheap_score_rows"] = macro_selected[role]["cheap_score_rows"]
            selected[role]["training_macro_selected_value_rows"] = macro_selected[role]["selected_value_rows"]
        context = context_rows(native_work)
        macro_context = context_rows(macro_work)
        for route in context:
            context[route]["training_macro"] = macro_context[route]
        return {
            "warmup_count": 1,
            "repetitions": 3,
            "phases": {name: {"samples_seconds": samples} for name, samples in phases.items()},
            "work_counts": {"complete_native": native_work, "training_macro": macro_work,
                            "phase_counts": phase_work},
            "peak_memory_bytes": peak_memory,
            "peak_memory_increment_bytes": max(0, peak_memory - peak_baseline),
            "selection_value_rows": selected,
            "context_score_work": context,
            "training_macro_case_ids": list(macro_ids),
            "training_macro_microbatch_sizes": [len(macro_batches[0].case_keys), len(macro_batches[1].case_keys)],
            "training_macro_optimizer_steps": 1,
            "training_macro_path": "maintained ResearchTrainingEngine batching, recovery snapshots, native objective, clipping, and AdamW; fresh optimizer construction and sampling charged",
        }


def _training_membership(config, catalog, manifest):
    membership = str(config["training"]["membership"])
    if membership in ("full_train600", "scratch_train600_exposed_val89"):
        train_ids = tuple(str(row["case_id"]) for row in catalog if row["split"] == "train")
        val_ids = tuple(str(row["case_id"]) for row in catalog
                        if row["split"] == "test" and row["case_id"] != "0273")
        validate_full_train_followup_membership(train_ids, val_ids)
        return train_ids, val_ids
    if membership == "fixed25_train150_dev22":
        return tuple(development_case_ids(manifest, "train")), tuple(development_case_ids(manifest, "test"))
    raise ValueError(f"Unsupported Round13-3 membership {membership!r}.")


def build_thermal_member_routing_task(config, *, arm=None, device="cpu"):
    recipe = copy.deepcopy(dict(config))
    section = recipe.get("member_routing", {})
    model_config = section.get("model")
    selected_arm = str(arm or section.get("arm", ""))
    if selected_arm not in ThermalInteractionReusedMemberRoutingAdapter.ARCHITECTURES:
        raise ValueError("Round13-3 builder requires a supported Full or Route arm.")
    if not isinstance(model_config, Mapping) or model_config.get("architecture") != selected_arm:
        raise ValueError("Round13-3 model config must match the selected named arm.")
    manifest, data_path, manifest_path = _load_data_binding(
        recipe["dataset"].get("manifest_path"), recipe["dataset"].get("data_path"),
    )
    catalog_source, catalog = read_case_catalog(data_path)
    train_ids, validation_ids = _training_membership(recipe, catalog, manifest)
    normalizer = fit_global_normalizer(data_path, train_ids)
    training_cases = _read_selected_cases(data_path, train_ids, split="train",
                                          normalizer=normalizer, include_grid=False)
    validation_cases = _read_selected_cases(data_path, validation_ids, split="test",
                                            normalizer=normalizer, include_grid=False)
    heat_values = np.concatenate([
        np.asarray(case["structure"]["heat_powers"])[
            np.asarray(case["structure"]["module_present"]) > 0.5
        ] for case in training_cases
    ])
    heat_scale = max(float(np.sqrt(np.mean(heat_values.astype(np.float64) ** 2))), 1e-6)
    initialization_seed = int(recipe["initialization"].get("seed", 0))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        model = ThermalInteractionReusedMemberRoutingAdapter(
            normalizer.stats, core_config=model_config, heat_scale=heat_scale,
        ).to(device)
    source_stat = data_path.stat()
    provider = ThermalInteractionReusedMemberRoutingTask(
        model=model,
        training_cases=training_cases,
        validation_cases=validation_cases,
        manifest=manifest,
        data_path=data_path,
        recipe=recipe,
        device=device,
        manifest_path=manifest_path,
        catalog=catalog,
        source_binding={
            "dataset_path": str(data_path),
            "dataset_size_bytes": int(source_stat.st_size),
            "dataset_mtime_ns": int(source_stat.st_mtime_ns),
            "catalog_metadata_sha256": catalog_source["metadata_sha256"],
            "manifest_path": None if manifest_path is None else str(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "normalization_source": "TRAIN-only fit_global_normalizer",
            "dataset_copy_created": False,
            "persistent_cache_created": False,
        },
    )
    return model, provider


def load_member_routing_checkpoint(checkpoint, device):
    """Rebuild a trusted Round13-3 checkpoint's adapter/provider without fitting."""
    from honf_runtime.compat import load_trusted_checkpoint
    from honf_runtime.contextual_frontier_training import _build_task, _read_json

    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    identity = payload.get("experiment_identity")
    provider_identity = identity.get("provider_identity") if isinstance(identity, Mapping) else None
    if (not isinstance(identity, Mapping) or identity.get("workflow") != WORKFLOW
            or not isinstance(provider_identity, Mapping)
            or provider_identity.get("schema") != PROVIDER_SCHEMA):
        raise ValueError("Checkpoint is not a maintained Round13-3 member-routing training checkpoint.")
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    manifest = _read_json(run_dir / "run_manifest.json")
    config = manifest.get("config")
    arm = manifest.get("arm")
    if not isinstance(config, Mapping) or arm not in ThermalInteractionReusedMemberRoutingAdapter.ARCHITECTURES:
        raise ValueError("Member-routing checkpoint run manifest is malformed.")
    model, provider = _build_task(config, arm, torch.device(device))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    provider.load_training_state_dict(payload.get("provider_training_state", {}))
    if provider.identity_payload() != dict(provider_identity):
        raise ValueError("Rebuilt member-routing provider identity differs from the checkpoint.")
    model.eval()
    return model, provider, payload


__all__ = [
    "WORKFLOW", "PROVIDER_SCHEMA", "CONFIG_SECTION_SCHEMA",
    "ThermalInteractionReusedMemberRoutingTask",
    "build_thermal_member_routing_task", "load_member_routing_checkpoint",
]
