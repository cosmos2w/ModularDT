"""Maintained native Thermal provider for Round13-8 E-anchored organization."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from honf_runtime.unified_training import SamplingKey
from ..e_anchor_locator import attach_thermal_e_anchor_query
from ..member_routing import ThermalAnchoredMemberRoutingAdapter, ThermalDetailAnchoredMemberRoutingAdapter
from .content_hypergraph_task import ThermalContentPredictions
from .joint_member_routing_task import _finite_scalar
from .member_routing_task import ThermalInteractionReusedMemberRoutingTask
from .r13_6_member_routing_task import _AnchoredPairedMemberRoutingTask
from .r13_6_member_routing_task import _build_thermal_member_routing_task


CONFIG_SECTION_SCHEMA = "thermal_r13_8_e_anchored_member_routing_v1"
SOURCE_KIND = "scratch_r13_8_e_anchored_member_routing"
CORE_SCHEMA = "interaction_reused_e_anchored_member_routing_v1"
WORKFLOW = "interaction_reused_e_anchored_member_routing_training_v1"
PROVIDER_SCHEMA = "thermal_r13_8_e_anchored_member_routing_provider_v1"
TRAINING_STATE_SCHEMA = "thermal_r13_8_e_anchored_member_routing_training_state_v1"
ARM_A = "R13-8-A-EAnchor"
ARM_L = "R13-8-L-LocalECover"
ARM_DETAIL = "R14-Detail-LocalECover"
DETAIL_CORE_SCHEMA = "interaction_reused_detail_anchored_member_routing_v1"
DETAIL_CONFIG_SCHEMA = "thermal_r14_detail_anchored_member_routing_v1"
DETAIL_SOURCE_KIND = "scratch_r14_detail_anchored_member_routing"
DETAIL_WORKFLOW = "interaction_reused_detail_anchored_member_routing_training_v1"
DETAIL_PROVIDER_SCHEMA = "thermal_r14_detail_anchored_member_routing_provider_v1"
DETAIL_STATE_SCHEMA = "thermal_r14_detail_anchored_member_routing_training_state_v1"

_VARIANTS = {
    CORE_SCHEMA: {
        "adapter": ThermalAnchoredMemberRoutingAdapter,
        "arm": None,
        "candidate": None,
        "workflow": WORKFLOW,
        "provider_schema": PROVIDER_SCHEMA,
        "training_state_schema": TRAINING_STATE_SCHEMA,
    },
    DETAIL_CORE_SCHEMA: {
        "adapter": ThermalDetailAnchoredMemberRoutingAdapter,
        "arm": ARM_DETAIL,
        "workflow": DETAIL_WORKFLOW,
        "provider_schema": DETAIL_PROVIDER_SCHEMA,
        "training_state_schema": DETAIL_STATE_SCHEMA,
        "candidate": "Detail",
    },
}


class AnchoredMemberRoutingHealth:
    """Sparse native-gradient health observation on the fixed TRAIN panel."""

    STATE_SCHEMA = "thermal_r13_8_anchored_member_routing_health_state_v1"

    def __init__(self, provider, config: Mapping[str, Any]):
        if not isinstance(config, Mapping) or config.get("enabled") is not True:
            raise ValueError("R13-8 health requires enabled=true.")
        self.provider = provider
        self.observation_epochs = int(config.get("observation_epochs", 12))
        self.later_every = int(config.get("later_every", 10))
        self.pause_on_collapse = bool(config.get("pause_on_collapse", True))
        panel = tuple(int(value) for value in config.get("panel_module_counts", (1, 2, 5, 12)))
        if self.observation_epochs < 1 or self.later_every < 1 or not panel or len(set(panel)) != len(panel):
            raise ValueError("R13-8 health cadence or module-count panel is invalid.")
        from ..data.development_split import development_case_ids

        fixed_train = set(map(str, development_case_ids(provider.manifest, "train")))
        admitted = {str(case["case_id"]): case for case in provider.training_cases}
        by_count: dict[int, list[str]] = {count: [] for count in panel}
        for case_id in fixed_train & set(admitted):
            present = np.asarray(admitted[case_id]["structure"]["module_present"]) > 0.5
            count = int(present.sum())
            if count in by_count:
                by_count[count].append(case_id)
        missing = [count for count, case_ids in by_count.items() if not case_ids]
        if missing:
            raise ValueError(f"R13-8 health panel lacks fixed TRAIN module counts {missing}.")
        self.panel_case_ids = tuple(min(by_count[count]) for count in panel)
        self.state = {
            "schema": self.STATE_SCHEMA,
            "panel_case_ids": list(self.panel_case_ids),
            "initial_observation": None,
            "last_observation": None,
            "last_observed_epoch": 0,
        }

    @property
    def initial_observation(self):
        return self.state.get("initial_observation")

    def scheduled(self, epoch: int) -> bool:
        epoch = int(epoch)
        if 0 < epoch <= self.observation_epochs:
            return True
        return epoch > self.observation_epochs and (epoch - self.observation_epochs) % self.later_every == 0

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        if state is None:
            return
        if (not isinstance(state, Mapping) or state.get("schema") != self.STATE_SCHEMA
                or list(state.get("panel_case_ids", ())) != list(self.panel_case_ids)):
            raise ValueError("Saved R13-8 health state differs from the fixed TRAIN panel.")
        self.state = copy.deepcopy(dict(state))

    def record_initial(self, observation):
        self.state["initial_observation"] = copy.deepcopy(observation)
        self.state["last_observation"] = copy.deepcopy(observation)
        self.state["last_observed_epoch"] = int(observation["training_epoch"])

    def record_epoch(self, observation):
        self.state["last_observation"] = copy.deepcopy(observation)
        self.state["last_observed_epoch"] = int(observation["training_epoch"])
        from .joint_member_routing_task import JointMemberRoutingHealth
        return JointMemberRoutingHealth.pause_reason(self.initial_observation, observation)

    @staticmethod
    def _gradient_norm(gradient):
        return 0.0 if gradient is None else _finite_scalar(gradient.detach().double().norm())

    def _observe_case(self, model, case_id: str) -> dict[str, Any]:
        provider = self.provider
        key = SamplingKey(
            seed=int(provider.seed), epoch=1, update_index=0, microbatch_index=0,
            stage="joint", arm=str(provider.training_sampling_arm),
            sampling_version=provider.sampling_version,
            dataset_id=provider.sampling_dataset_id,
        )
        batch = provider.make_batch((case_id,), key)
        prepared = provider._prepare_for_probe(model, batch)
        expected_heat = torch.as_tensor(
            prepared.structure["heat_powers"], device=batch.receivers.heat.device,
            dtype=batch.receivers.heat.dtype,
        )
        if (expected_heat.shape != batch.receivers.heat.shape
                or not torch.equal(expected_heat, batch.receivers.heat)):
            raise RuntimeError("R13-8 health observed prescribed heat changed before native decoding.")
        paired = model.core.decode_paired(
            prepared.organization, prepared.query_batch,
        )
        if not isinstance(paired, tuple) or len(paired) != 3:
            raise TypeError("R13-8 health requires the actual paired full/restricted decoder outputs.")
        full_output, restricted_output, shared_counts = paired
        outputs = {"full": full_output, "restricted": restricted_output}
        native_by_mode, losses = {}, {}
        objective = None
        for mode, output in outputs.items():
            native = model._extract_native_roles(
                prepared, output.joint_fields[..., 3:4], output.joint_fields[..., :3],
            )
            native_by_mode[mode] = native
            prediction = ThermalContentPredictions(prepared, native, work=output.work_counts)
            role_losses = provider._role_losses(prediction, batch.targets)
            losses[mode] = {name: _finite_scalar(value.detach().mean())
                            for name, value in role_losses.items()}
            mode_objective = sum(float(provider.native_weights[name]) * values.mean()
                                 for name, values in role_losses.items())
            objective = mode_objective * 0.5 if objective is None else objective + mode_objective * 0.5
        if not torch.is_tensor(objective) or not objective.requires_grad:
            raise RuntimeError("R13-8 paired native objective is disconnected from autograd.")

        organization = prepared.organization
        directory = getattr(organization, "e_access_directory", None)
        if directory is None:
            raise RuntimeError("R13-8 native health requires the compiled E access directory.")
        target_specs = [
            ("applied_EM", getattr(organization, "anchored_applied_EM", None)),
            ("applied_EE", getattr(organization, "anchored_applied_EE", None)),
            ("member_values_M", getattr(organization, "member_values_M", None)),
            ("member_values_E", getattr(organization, "member_values_E", None)),
            ("full_access_weights", directory.full.access_weight),
            ("restricted_access_weights", directory.restricted.access_weight),
        ]
        # Score evidence is taken from the actual trainable scorer parameters,
        # not a downstream trace view that can be disconnected by indexing.
        scorer_modules = {
            "member_scores_M": model.core.member_router.member_M,
            "member_scores_E": model.core.member_router.member_E,
            "read_gate_M": model.core.typed_read_gates["M"],
            "read_gate_E": model.core.typed_read_gates["E"],
            "applied_relation_EM": model.core.applied_incidence.routes.em,
            "applied_relation_EE": model.core.applied_incidence.routes.ee,
        }
        if model.CORE_SCHEMA == DETAIL_CORE_SCHEMA:
            scorer_modules.update({
                "query_relative_values_M": model.core.relative_value["M"],
                "query_relative_values_E": model.core.relative_value["E"],
            })
        named_targets = [(name, value) for name, value in target_specs
                         if torch.is_tensor(value) and value.requires_grad]
        missing = [name for name, value in target_specs
                   if not torch.is_tensor(value) or not value.requires_grad]
        if missing:
            raise RuntimeError(f"R13-8 health cannot inspect live prepared tensors {missing}.")
        target_labels = [name for name, _value in named_targets]
        target_tensors = [value for _name, value in named_targets]
        grouped_labels: dict[str, list[int]] = {}
        for group_name, module in scorer_modules.items():
            indices = []
            for parameter in module.parameters():
                if parameter.requires_grad:
                    indices.append(len(target_tensors))
                    target_tensors.append(parameter)
            grouped_labels[group_name] = indices
            if not indices:
                raise RuntimeError(f"R13-8 health cannot inspect trainable {group_name} parameters.")
        gradients = torch.autograd.grad(objective, tuple(target_tensors), allow_unused=True)
        gradient_connected: dict[str, bool] = {}
        gradient_norms = {
            name: self._gradient_norm(gradient)
            for name, gradient in zip(target_labels, gradients[:len(target_labels)], strict=True)
        }
        gradient_connected.update({
            name: gradient is not None
            for name, gradient in zip(target_labels, gradients[:len(target_labels)], strict=True)
        })
        for group_name, indices in grouped_labels.items():
            norms = [self._gradient_norm(gradients[index]) for index in indices]
            gradient_norms[group_name] = math.sqrt(sum(value * value for value in norms))
            gradient_connected[group_name] = any(gradients[index] is not None for index in indices)
        gradient_paths = {
            name: {"available": gradient_connected[name], "nonzero": value > 0.0,
                  "l2": float(value)}
            for name, value in gradient_norms.items()
        }

        from honf_forward_core.content_hypergraph.query_chunks_v1 import select_query_rows_v1

        trace_count = min(8, int(prepared.query_batch.num_queries))
        trace_queries = select_query_rows_v1(
            prepared.query_batch, torch.arange(trace_count, device=prepared.query_batch.coordinates.device),
        )
        bounded_traces = {}
        with torch.no_grad():
            for mode in ("full", "restricted"):
                output = model.core.decode(
                    prepared.organization, trace_queries,
                    intervention={"read_mode": mode}, return_trace=True,
                )
                if not isinstance(output.trace, Mapping):
                    raise RuntimeError("R13-8 health could not read the bounded actual directory trace.")
                bounded_traces[mode] = output.trace

        # Verify the declared E-slot identity map and self entry, and measure
        # the actual compiled full/restricted rows that the query reuses.
        row_summaries = {"full": [], "restricted": []}
        query_carrier = prepared.query_batch
        active_carriers = torch.nonzero(
            (query_carrier.anchor_weights[0, :trace_count] > 0)
            & (query_carrier.anchor_indices[0, :trace_count] >= 0), as_tuple=False,
        )
        scene = prepared.scene_batch
        for query_index, cover_index in active_carriers.tolist():
            anchor_slot = int(query_carrier.anchor_indices[0, query_index, cover_index])
            anchor_id = int(scene.environment_ids[0, anchor_slot])
            own_global = int(directory.module_count + directory.own_e_member_index[0, anchor_slot])
            for mode in ("full", "restricted"):
                table = getattr(directory, mode)
                row = table.row(0, anchor_slot)
                self_entries = [index for index, member in enumerate(row.member_index.tolist())
                                if int(member) == own_global and not bool(row.is_module[index])
                                and int(row.original_id[index]) == anchor_id
                                and float(row.structural_self[index]) > 0]
                if len(self_entries) != 1:
                    raise RuntimeError(f"R13-8 {mode} row lost its mapped structural E self entry.")
                row_summaries[mode].append({
                    "query_index": int(query_index),
                    "cover_index": int(cover_index),
                    "anchor_slot": anchor_slot,
                    "anchor_original_id": anchor_id,
                    "self_entry_count": len(self_entries),
                    "entry_count": int(row.member_index.numel()),
                    "module_entry_count": int(row.is_module.sum()),
                    "environment_entry_count": int((~row.is_module).sum()),
                    "nonself_entry_count": int((row.member_index != own_global).sum()),
                    "normalized_access_mass": _finite_scalar(row.access_weight.sum()),
                    "positive_applied_exchange_entries": int((row.applied_exchange > 0).sum()),
                })
        if not row_summaries["full"] or not row_summaries["restricted"]:
            raise RuntimeError("R13-8 health fixed TRAIN queries did not reach an E access row.")
        restricted_mass = [row["normalized_access_mass"] for row in row_summaries["restricted"]]
        if any(value < float(directory.retained_mass) - 1e-6 for value in restricted_mass):
            raise RuntimeError("R13-8 restricted directory row fell below its declared joint retained mass.")

        row_local_strength = {}
        for mode, trace in bounded_traces.items():
            row_local_strength[mode] = {}
            for role in ("M", "E"):
                rho = trace.get(f"anchor_read_strength_{role}")
                if not torch.is_tensor(rho):
                    raise RuntimeError(f"R13-8 {mode} trace omitted row-local {role} read strength.")
                active = query_carrier.anchor_weights[:, :trace_count] > 0
                selected = rho[:, :trace_count][active]
                row_local_strength[mode][role] = {
                    "rows": int(selected.numel()),
                    "mean": _finite_scalar(selected.mean()) if selected.numel() else 0.0,
                    "min": _finite_scalar(selected.min()) if selected.numel() else 0.0,
                    "max": _finite_scalar(selected.max()) if selected.numel() else 0.0,
                }
        carrier_weights = query_carrier.anchor_weights[:, :trace_count]
        locator_weights = {
            "rows": int(carrier_weights.shape[0] * carrier_weights.shape[1]),
            "active_anchor_entries": int((carrier_weights > 0).sum()),
            "mean_active_anchors_per_query": _finite_scalar((carrier_weights > 0).sum(-1).float().mean()),
            "maximum_row_sum_error": _finite_scalar((carrier_weights.sum(-1) - 1).abs().max()),
            "minimum_positive_weight": _finite_scalar(carrier_weights[carrier_weights > 0].min()),
            "maximum_weight": _finite_scalar(carrier_weights.max()),
            "weighted_centroid_coordinate_gradient_l2": None,
        }
        active_anchor_ids = query_carrier.anchor_indices[:, :trace_count][
            query_carrier.anchor_weights[:, :trace_count] > 0
        ]
        anchor_usage = torch.bincount(active_anchor_ids, minlength=scene.environment_coordinates.shape[1])
        locator_weights["unique_anchor_slots_used"] = int((anchor_usage > 0).sum())
        locator_weights["anchor_slots_reused_by_multiple_queries"] = int((anchor_usage > 1).sum())
        if model.core.locator_mode == "local_e_cover":
            live_coordinates = query_carrier.coordinates.detach().clone().requires_grad_(True)
            live_query = replace(query_carrier, coordinates=live_coordinates,
                                 anchor_indices=None, anchor_weights=None)
            live_query = attach_thermal_e_anchor_query(
                live_query, scene, model.core.locator_mode,
                environment_nx=model.environment_nx, environment_ny=model.environment_ny,
            )
            safe_indices = live_query.anchor_indices.clamp_min(0)
            environment_x = scene.environment_coordinates[..., 0][:, None, :].expand(
                -1, live_query.coordinates.shape[1], -1,
            ).gather(-1, safe_indices)
            centroid = (environment_x * live_query.anchor_weights).sum()
            coordinate_gradient = torch.autograd.grad(centroid, live_coordinates)[0]
            locator_weights["weighted_centroid_coordinate_gradient_l2"] = _finite_scalar(
                coordinate_gradient.detach().double().norm(),
            )

        # A bounded locator-only check records the actual owner/cover weights
        # on both axes around a deterministic internal coverage boundary. The
        # physical controls and scene are fixed; no material-interface claim
        # is made from this artificial E-grid probe.
        origin = scene.coordinate_origin[0]
        extent = scene.coordinate_scale[0]
        boundary_locator: dict[str, Any] = {}
        for axis_index, axis_name, orth_index in ((0, "x", 1), (1, "y", 0)):
            axis_count = model.environment_nx if axis_index == 0 else model.environment_ny
            orth_count = model.environment_ny if orth_index == 1 else model.environment_nx
            axis_step = extent[axis_index] / axis_count
            orth_slot = orth_count // 2
            orth_coordinate = origin[orth_index] + (orth_slot + 0.5) * extent[orth_index] / orth_count
            if model.core.locator_mode == "e_anchor":
                face = origin[axis_index] + (axis_count // 2) * axis_step
                kind = "hard_cell_ownership_face"
            else:
                center_slot = axis_count // 2
                face = origin[axis_index] + (center_slot + 0.5) * axis_step
                kind = "smooth_cover_center_knot"
            offsets = torch.tensor([-0.05, 0.0, 0.05], device=origin.device, dtype=origin.dtype) * axis_step
            coordinates = origin.new_zeros((1, 3, 2))
            coordinates[..., axis_index] = face + offsets
            coordinates[..., orth_index] = orth_coordinate
            boundary_query = type(query_carrier)(
                coordinates=coordinates,
                query_mask=torch.ones((1, 3), dtype=torch.bool, device=coordinates.device),
                scene_key=query_carrier.scene_key,
                coordinate_frame_key=query_carrier.coordinate_frame_key,
                receiver_key=query_carrier.receiver_key,
            )
            boundary_query = attach_thermal_e_anchor_query(
                boundary_query, scene, model.core.locator_mode,
                environment_nx=model.environment_nx, environment_ny=model.environment_ny,
            )
            samples = []
            for row in range(3):
                active = boundary_query.anchor_weights[0, row] > 0
                slots = boundary_query.anchor_indices[0, row, active]
                weights = boundary_query.anchor_weights[0, row, active]
                samples.append({
                    "coordinate": _finite_scalar(coordinates[0, row, axis_index]),
                    "anchor_slots": slots.detach().cpu().tolist(),
                    "anchor_original_ids": scene.environment_ids[0, slots].detach().cpu().tolist(),
                    "weights": weights.detach().double().cpu().tolist(),
                })
            boundary_locator[axis_name] = {
                "kind": kind,
                "face_coordinate": _finite_scalar(face),
                "offsets_in_spacing": [-0.05, 0.0, 0.05],
                "samples": samples,
                "physical_controls_held_fixed": True,
                "physical_interface_classification": "not_assessed; artificial E coverage boundary",
            }

        trace_shapes = {
            mode: {
                "anchor_queries": int(trace["anchor_indices"].shape[1]),
                "packed_read_members": int(trace["anchor_read_member_index"].numel()),
                "read_probability_entries": int(trace["anchor_read_probabilities"].numel()),
                "score_union_members_M": int(trace["member_score_union_values_M"].numel()),
                "score_union_members_E": int(trace["member_score_union_values_E"].numel()),
                "mean_reused_anchor_mass": _finite_scalar(trace["anchor_weights"].sum(-1).mean()),
            }
            for mode, trace in bounded_traces.items()
        }
        proposal_names = [name for name, _parameter in model.named_parameters()
                          if "proposal_M" in name or "proposal_scores" in name]
        repeatability, heat_sensitivity = {}, {}
        with torch.no_grad():
            repeat_prepared = provider._prepare_for_probe(model, batch)
            heat_batch = replace(batch, receivers=replace(
                batch.receivers, heat=batch.receivers.heat * 1.1,
            ))
            heat_prepared = provider._prepare_for_probe(model, heat_batch)
            from .member_routing_health import _native_field_delta
            for mode in ("full", "restricted"):
                repeat_native, _repeat_counts, _repeat_output = model.apply_native_with_work_counts(
                    repeat_prepared, batch.receivers.heat,
                    intervention={"read_mode": mode},
                )
                heat_native, _heat_counts, _heat_output = model.apply_native_with_work_counts(
                    heat_prepared, heat_batch.receivers.heat,
                    intervention={"read_mode": mode},
                )
                repeatability[mode] = _native_field_delta(
                    native_by_mode[mode], repeat_native, model, prepared,
                )
                heat_sensitivity[mode] = _native_field_delta(
                    native_by_mode[mode], heat_native, model, prepared,
                )
        if not isinstance(shared_counts, Mapping):
            raise TypeError("R13-8 paired shared work counters must be a mapping.")
        return {
            "case_id": str(case_id),
            "module_count": int(prepared.scene_batch.module_mask.sum().item()),
            "native_fieldwise_normalized_mse": losses,
            "paired_shared_work_counts": {str(name): int(value) for name, value in shared_counts.items()},
            "native_objective_gradient_l2": gradient_norms,
            "native_objective_gradient_paths": gradient_paths,
            "proposal_entrance_parameters_present": proposal_names,
            "prescribed_controls": {
                "heat": batch.receivers.heat.detach().cpu().tolist(),
                "heat_plus10_native_fieldwise_delta": heat_sensitivity,
                "same_input_repeatability_native_fieldwise_delta": repeatability,
                "operating_context": prepared.scene_batch.operating_features.detach().cpu().tolist(),
            },
            "modes": {
                mode: {
                    "heat_plus10_fieldwise_native_delta": heat_sensitivity[mode],
                    "repeatability_fieldwise_native_delta": repeatability[mode],
                }
                for mode in ("full", "restricted")
            },
            "compiled_directory_rows": row_summaries,
            "locator_weights": locator_weights,
            "locator_boundary_observation": boundary_locator,
            "row_local_read_strength": row_local_strength,
            "trace": trace_shapes,
            "bounded_trace_query_count": trace_count,
            "native_query_count": int(prepared.query_batch.num_queries),
        }

    def observe(self, model, *, training_epoch: int) -> dict[str, Any]:
        from honf_runtime.unified_training import capture_rng_state, restore_rng_state

        rng = capture_rng_state()
        provider_state = copy.deepcopy(self.provider.training_state_dict())
        training_mode = bool(model.training)
        core = model.core
        core_clock = (getattr(core, "_joint_epoch", None), getattr(core, "_joint_update_index", None))
        model.eval()
        try:
            with torch.enable_grad():
                cases = [self._observe_case(model, case_id) for case_id in self.panel_case_ids]
        finally:
            model.train(training_mode)
            if core_clock[0] is not None:
                core._joint_epoch = core_clock[0]
            if core_clock[1] is not None:
                core._joint_update_index = core_clock[1]
            self.provider.load_training_state_dict(provider_state)
            restore_rng_state(rng)
        return {
            "schema": "thermal_r13_8_anchored_member_routing_health_v1",
            "training_epoch": int(training_epoch),
            "scope": "fixed TRAIN cases; complete Q=1024 native full/restricted objective and actual decoder traces",
            "case_ids": list(self.panel_case_ids),
            "cases": cases,
            "all_required_live_gradients_observed": all(
                all(case["native_objective_gradient_paths"].get(name, {}).get("available") is True
                    for name in ("applied_EM", "applied_EE", "member_values_M", "member_values_E",
                                 "full_access_weights", "restricted_access_weights", "member_scores_M",
                                 "member_scores_E", "read_gate_M", "read_gate_E",
                                 "applied_relation_EM", "applied_relation_EE"))
                for case in cases
            ),
            "entrance_scorer_absent": all(not case["proposal_entrance_parameters_present"] for case in cases),
            "mode_restored": bool(model.training) == training_mode,
            "device": str(self.provider.device),
            "interpretation": "Measured native-loss adjoints cover final applied EM/EE, compiled full/restricted access weights, original member values, actual member scorers, row-local gates, and applied EM/EE route parameters. Availability, nonzero magnitude, and legal empty-role support are reported separately; empty M rows or broad/similar rows are descriptive and do not request a stop.",
        }


class ThermalR138AnchoredMemberRoutingTask(_AnchoredPairedMemberRoutingTask):
    """R13-8 paired native objective with versioned A/L identity and live traces."""

    VARIANTS = _VARIANTS
    EXPECTED_SOURCE_KIND = SOURCE_KIND
    LINEAGE = "R13-8"
    training_state_schema = TRAINING_STATE_SCHEMA

    def __init__(self, **kwargs):
        # Reuse R13-5's paired full/restricted objective and native tasks while
        # avoiding its R13-6 health checker, whose applied-block assumptions do
        # not describe the compiled E directory.
        ThermalInteractionReusedMemberRoutingTask.__init__(self, **kwargs)
        variant = _VARIANTS.get(self.model.CORE_SCHEMA)
        if variant is None or self.model.architecture not in variant["adapter"].ARCHITECTURES:
            raise ValueError("R13-8 task requires an A/L anchored adapter.")
        self._member_routing_variant = variant
        self.arm = str(self.model.architecture)
        self.candidate = "A" if self.arm == ARM_A else "L" if self.arm == ARM_L else variant["candidate"]
        self.workflow = variant["workflow"]
        self.provider_schema = variant["provider_schema"]
        self.training_state_schema = variant["training_state_schema"]
        initialization = self.recipe["initialization"]
        if getattr(self.model.core, "mediated_mode", "none") in ("P", "G"):
            self.seed = int(initialization["seed"])
        self.source_kind = str(initialization["source_kind"])
        self.source_parent_age = int(initialization.get("source_parent_age", 0))
        expected_source = DETAIL_SOURCE_KIND if self.arm == ARM_DETAIL else SOURCE_KIND
        if (self.source_kind != expected_source or self.source_parent_age != 0
                or initialization.get("source_checkpoint") is not None
                or initialization.get("pretrained_weights") is not False
                or initialization.get("source_arm") != f"scratch_common_seed{self.seed}"):
            raise ValueError("Anchored training requires a declared fresh scratch lineage with no source checkpoint.")
        self.training_sampling_arm = str(self.recipe["training"].get("sampling_arm", "R13-3-Relative"))
        health_config = self.recipe.get("member_routing", {}).get("health")
        self._member_routing_health = None
        if isinstance(health_config, Mapping) and health_config.get("enabled") is True:
            if getattr(self.model.core, "mediated_mode", "none") in ("P", "G"):
                from .r13_12_mediated_health import MediatedMemberRoutingHealth
                self._member_routing_health = MediatedMemberRoutingHealth(self, health_config)
            else:
                self._member_routing_health = AnchoredMemberRoutingHealth(self, health_config)

    def anchored_organization_trace(self, model, prediction, query_indices, *, read_mode="restricted"):
        if model is not self.model or model.core.schema != self.model.CORE_SCHEMA:
            raise ValueError("Anchored trace must use the provider's R13-8 model.")
        if read_mode not in ("full", "restricted"):
            raise ValueError("R13-8 trace read_mode must be 'full' or 'restricted'.")
        from honf_forward_core.content_hypergraph.query_chunks_v1 import select_query_rows_v1

        prepared = prediction.native_prepared
        query = select_query_rows_v1(prepared.query_batch, query_indices)
        output = model.core.decode(
            prepared.organization, query,
            intervention={"read_mode": read_mode}, return_trace=True,
        )
        if not isinstance(output.trace, Mapping):
            raise RuntimeError("Anchored decoder omitted the actual organization trace.")
        return {
            "schema": "thermal_r13_8_anchored_organization_trace_v1",
            "query_indices": [int(value) for value in torch.as_tensor(query_indices).reshape(-1).tolist()],
            "read_mode": read_mode,
            "trace": output.trace,
            "work_counts": {str(name): int(value) for name, value in output.work_counts.items()},
            "same_prepared_context": True,
        }

    @staticmethod
    def _directory_intervention_spec(prepared, query_indices, request):
        if isinstance(request, str):
            request = {"kind": request}
        if not isinstance(request, Mapping):
            raise TypeError("directory_intervention must be a mode or mapping.")
        spec = dict(request)
        kind = spec.pop("kind", spec.pop("mode", spec.pop("directory_intervention", None)))
        if kind not in ("self_only", "remove_member", "reassign_member"):
            raise ValueError("Unsupported R13-8 directory intervention.")
        spec["access_intervention"] = kind
        if kind == "self_only":
            return spec, {"eligible": True, "kind": kind}
        required = ({"member_role", "member_id"} if kind == "remove_member" else
                    {"source_role", "source_id", "target_role", "target_id"})
        if required <= set(spec):
            selected = {"eligible": True, "kind": kind, "selected_ids": {
                **{name: spec[name] for name in sorted(required)},
                **({"anchor_id": int(spec["anchor_id"])} if spec.get("anchor_id") is not None else {}),
            }}
            if spec.get("anchor_id") is not None:
                selected["anchor_id"] = int(spec["anchor_id"])
            return spec, selected

        # Pick the first selected query's first positive anchor. Source order is
        # the actual packed full-row order; target order is stable role then ID.
        indices = torch.as_tensor(query_indices, dtype=torch.long,
                                  device=prepared.query_batch.coordinates.device).reshape(-1)
        if not indices.numel():
            return spec, {"eligible": False, "kind": kind, "reason": "empty_query_selection"}
        batch_index = 0
        query_index = int(indices[0])
        carrier_indices = prepared.query_batch.anchor_indices[batch_index, query_index]
        carrier_weights = prepared.query_batch.anchor_weights[batch_index, query_index]
        active = torch.nonzero(carrier_weights > 0, as_tuple=False).reshape(-1)
        if not active.numel():
            return spec, {"eligible": False, "kind": kind, "reason": "selected_query_has_no_anchor"}
        anchor_slot = int(carrier_indices[int(active[0])])
        anchor_id = int(prepared.scene_batch.environment_ids[batch_index, anchor_slot])
        spec["anchor_id"] = anchor_id
        directory = prepared.organization.e_access_directory
        row = directory.full.row(batch_index, anchor_slot)
        own_global = (int(directory.module_count)
                      + int(directory.own_e_member_index[batch_index, anchor_slot]))
        member_position = next((position for position, global_index in enumerate(row.member_index.tolist())
                                if int(global_index) != own_global), None)
        if member_position is None:
            return spec, {"eligible": False, "kind": kind, "reason": "no_nonself_member_in_full_row",
                          "anchor_id": anchor_id, "anchor_original_id": anchor_id}
        source_role = "M" if bool(row.is_module[member_position]) else "E"
        source_id = int(row.original_id[member_position])
        if kind == "remove_member":
            spec.update(member_role=source_role, member_id=source_id)
            return spec, {"eligible": True, "kind": kind, "anchor_id": anchor_id,
                          "selected_ids": {
                "member_role": source_role, "member_id": source_id, "anchor_id": anchor_id}}

        scene = prepared.scene_batch
        present: list[tuple[int, int, str]] = []
        for role_index, (role, ids, mask) in enumerate((
                ("M", scene.module_ids[batch_index], scene.module_mask[batch_index]),
                ("E", scene.environment_ids[batch_index], scene.environment_mask[batch_index]))):
            present.extend((role_index, int(ids[index]), role)
                           for index in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist())
        in_row = {("M" if bool(row.is_module[index]) else "E", int(row.original_id[index]))
                  for index in range(row.original_id.numel())}
        targets = sorted((item for item in present if (item[2], item[1]) not in in_row),
                         key=lambda item: (item[0], item[1]))
        if not targets:
            return spec, {"eligible": False, "kind": kind, "reason": "no_present_target_outside_full_row",
                          "anchor_id": anchor_id, "source_role": source_role, "source_id": source_id}
        _role_order, target_id, target_role = targets[0]
        spec.update(source_role=source_role, source_id=source_id,
                    target_role=target_role, target_id=target_id)
        return spec, {"eligible": True, "kind": kind, "anchor_id": anchor_id, "anchor_original_id": int(
            scene.environment_ids[batch_index, anchor_slot]), "selected_ids": {
                "source_role": source_role, "source_id": source_id,
                "target_role": target_role, "target_id": target_id,
                "anchor_id": anchor_id}}

    def anchored_intervention(self, model, batch, base_prediction, query_indices, *,
                              directory_intervention, read_mode="restricted"):
        if model is not self.model or read_mode not in ("full", "restricted"):
            raise ValueError("R13-8 intervention must use the provider model and a maintained read mode.")
        prepared = base_prediction.native_prepared
        spec, selected = self._directory_intervention_spec(prepared, query_indices, directory_intervention)
        if not selected.get("eligible"):
            return {
                "schema": "thermal_r13_8_anchored_intervention_v1",
                "directory_intervention": selected,
                "prediction": base_prediction,
                "baseline_prediction": base_prediction,
                "baseline_trace": self.anchored_organization_trace(
                    model, base_prediction, query_indices, read_mode=read_mode,
                ),
                "intervention_trace": None,
                "work_counts": {"intervention_not_run": 1},
                "trace": None,
                "reassignment": selected if selected.get("kind") == "reassign_member" else None,
                "same_prepared_context": True,
            }
        core_spec = {key: value for key, value in spec.items()
                     if key in {"access_intervention", "source_role", "source_id",
                                "target_role", "target_id", "anchor_id"}}
        if spec.get("access_intervention") == "remove_member":
            core_spec["source_role"] = spec["member_role"]
            core_spec["source_id"] = spec["member_id"]
        core_spec["read_mode"] = read_mode
        intervention_output = model.core.decode(
            prepared.organization, prepared.query_batch,
            intervention=core_spec, return_trace=False,
        )
        native = model._extract_native_roles(
            prepared, intervention_output.joint_fields[..., 3:4],
            intervention_output.joint_fields[..., :3],
        )
        prediction = ThermalContentPredictions(prepared, native, work=intervention_output.work_counts)
        trace_record = self.anchored_organization_trace(
            model, base_prediction, query_indices, read_mode=read_mode,
        )
        from honf_forward_core.content_hypergraph.query_chunks_v1 import select_query_rows_v1

        selected_query = select_query_rows_v1(prepared.query_batch, query_indices)
        trace_output = model.core.decode(
            prepared.organization, selected_query,
            intervention=core_spec, return_trace=True,
        )
        if not isinstance(trace_output.trace, Mapping):
            raise RuntimeError("Anchored intervention omitted its actual E-directory trace.")
        return {
            "schema": "thermal_r13_8_anchored_intervention_v1",
            "directory_intervention": selected,
            "prediction": prediction,
            "baseline_prediction": base_prediction,
            "baseline_trace": trace_record["trace"],
            "intervention_trace": trace_output.trace,
            "trace": {
                "baseline": trace_record["trace"],
                "intervention": trace_output.trace,
            },
            "baseline_work_counts": dict(getattr(base_prediction, "work", {}) or {}),
            "intervention_work_counts": {str(name): int(value)
                                         for name, value in intervention_output.work_counts.items()},
            "same_prepared_context": prepared is base_prediction.native_prepared,
            "prepared_context_identity": id(prepared),
            "work_counts": {
                "baseline": dict(getattr(base_prediction, "work", {}) or {}),
                "intervention": {str(name): int(value)
                                 for name, value in intervention_output.work_counts.items()},
            },
            "reassignment": selected if selected.get("kind") == "reassign_member" else None,
        }

    def anchored_boundary_sweep(self, model, batch, query_index, probe_coordinates, *,
                                offsets_in_spacing=None, read_mode="restricted"):
        """Rebuild native union and live A/L anchors for each coordinate probe."""
        if model is not self.model or read_mode not in ("full", "restricted"):
            raise ValueError("R13-8 boundary sweep requires the provider model and a maintained read mode.")
        if batch.receivers.fluid_xy.shape[0] != 1:
            raise ValueError("R13-8 boundary sweep accepts one selected native scene at a time.")
        query_index = int(query_index)
        original = batch.receivers.fluid_xy
        if not 0 <= query_index < original.shape[1]:
            raise IndexError("Boundary fluid query index is outside the selected native batch.")
        points = torch.as_tensor(probe_coordinates, device=original.device, dtype=original.dtype)
        if points.ndim != 2 or points.shape[-1] != 2 or points.shape[0] < 1:
            raise ValueError("Boundary probe coordinates must have shape [S,2].")
        prepared = self._prepare_for_probe(model, batch)
        native, native_counts, _native_output = model.apply_native_with_work_counts(
            prepared, batch.receivers.heat,
            intervention={"read_mode": read_mode},
        )
        from ..native_observations import interpolate_native, native_cell_center_stencil

        nx, ny = int(model.nx), int(model.ny)
        lengths = prepared.scene_batch.coordinate_scale
        origin = (prepared.scene_batch.coordinate_origin
                  if prepared.scene_batch.coordinate_origin is not None
                  else lengths.new_zeros((lengths.shape[0], 2)))
        native_receivers = original
        cell_grid = (native_receivers - origin[:, None]) / (
            lengths[:, None] / lengths.new_tensor([nx, ny])
        ) - 0.5
        rounded_grid = cell_grid.round()
        lattice_tolerance = 8 * torch.finfo(cell_grid.dtype).eps * max(nx, ny)
        if not bool(((cell_grid - rounded_grid).abs() <= lattice_tolerance).all()):
            raise RuntimeError("Native boundary interpolation requires the unchanged cell-centred receiver grid.")
        ix, iy = rounded_grid[..., 0].long(), rounded_grid[..., 1].long()
        if (bool(((ix < 0) | (ix >= nx) | (iy < 0) | (iy >= ny)).any())
                or native_receivers.shape[1] != nx * ny):
            raise RuntimeError("Native boundary interpolation requires the complete fixed Thermal grid.")
        grid_index = (iy * nx + ix).reshape(-1)
        if torch.unique(grid_index).numel() != nx * ny:
            raise RuntimeError("Native boundary receiver rows do not cover the grid exactly once.")
        grid_to_receiver = torch.empty((nx * ny,), device=original.device, dtype=torch.long)
        grid_to_receiver.scatter_(0, grid_index, torch.arange(nx * ny, device=original.device))
        native_field_grid = native["pred_field"].index_select(1, grid_to_receiver)
        native_stencil = native_cell_center_stencil(
            (points[None] - origin[:, None]), lengths, nx=nx, ny=ny,
        )
        native_probe_values = interpolate_native(native_field_grid, native_stencil)[0]

        decoder_rows, native_rows, anchor_indices, anchor_weights, work_rows = [], [], [], [], []
        score_trace_rows = []
        decoder_scale = model.field_std[[0, 1, 2, 4]].detach().to(original)
        decoder_mean = torch.cat((
            model.field_mean[[0, 1, 2]].detach().to(original),
            original.new_zeros((1,)),
        ))
        native_scale = model.field_std[[0, 1, 2, 3, 4]].detach().to(original)
        for probe_index, point in enumerate(points):
            query_mask = torch.ones((1, 1), device=point.device, dtype=torch.bool)
            direct = type(prepared.query_batch)(
                coordinates=point.reshape(1, 1, 2), query_mask=query_mask,
                scene_key=prepared.query_batch.scene_key,
                coordinate_frame_key=prepared.query_batch.coordinate_frame_key,
                receiver_key=prepared.query_batch.receiver_key,
            )
            packed = model._packed_local_pairs(prepared.scene_batch, direct.coordinates, query_mask)
            direct = replace(
                direct,
                local_batch_indices=packed[0], local_query_indices=packed[1],
                local_module_indices=packed[2], local_weights=packed[3],
            )
            direct = attach_thermal_e_anchor_query(
                direct, prepared.scene_batch, model.core.locator_mode,
                environment_nx=model.environment_nx, environment_ny=model.environment_ny,
            )
            direct_output = model.core.decode(
                prepared.organization, direct,
                intervention={"read_mode": read_mode}, return_trace=True,
            )
            if not isinstance(direct_output.trace, Mapping):
                raise RuntimeError("R13-8 boundary probe omitted its actual anchored read trace.")
            normalized_fields = direct_output.joint_fields[0, 0]
            decoder_rows.append(normalized_fields * decoder_scale + decoder_mean)
            native_rows.append(native_probe_values[probe_index])
            anchor_indices.append(direct.anchor_indices[0, 0])
            anchor_weights.append(direct.anchor_weights[0, 0])
            trace = direct_output.trace
            score_trace_rows.append({
                "member_score_union": {
                    role: {
                        "indices": trace[f"member_score_union_indices_{role}"].detach().cpu().tolist(),
                        "raw_scores": trace[f"member_score_union_values_{role}"].detach().cpu().tolist(),
                    }
                    for role in ("M", "E")
                },
                "anchor_read": {
                    key: trace[key].detach().cpu().tolist()
                    for key in (
                        "anchor_read_E_slots", "anchor_read_indices",
                        "anchor_read_member_read_index", "anchor_read_member_index",
                        "anchor_read_original_id", "anchor_read_is_module",
                        "anchor_read_access_prior", "anchor_read_probabilities",
                        "anchor_read_calibrated_scores", "anchor_read_strength",
                        "anchor_read_coalesced_union_keys",
                        "anchor_read_coalesced_member_coefficients",
                    )
                },
            })
            work_rows.append({
                **{f"decoder_{name}": int(value) for name, value in direct_output.work_counts.items()},
                "query_anchor_locator_rows": 1,
                "query_anchor_index_rows": int((direct.anchor_weights > 0).sum()),
            })
        return {
            "schema": "thermal_r13_8_anchored_boundary_sweep_v1",
            "locator_mode": model.core.locator_mode,
            "read_mode": read_mode,
            "query_index": query_index,
            "probe_coordinates": points.detach(),
            "offsets_in_spacing": None if offsets_in_spacing is None else list(map(float, offsets_in_spacing)),
            "decoder": {
                "values": torch.stack(decoder_rows),
                "output_scale": decoder_scale,
                "field_names": ["u", "v", "p", "temperature"],
                "units": "native physical units",
            },
            "native_extraction": {
                "values": torch.stack(native_rows),
                "output_scale": native_scale,
                "field_names": ["u", "v", "p", "omega", "temperature"],
                "units": "native physical units",
                "native_grid_nx_ny": [nx, ny],
                "native_grid_cell_count": nx * ny,
                "interpolation_valid_mask": native_stencil.valid[0].detach(),
                "sampling_method": (
                    "existing cell-centre piecewise-bilinear interpolation of the fixed-grid native "
                    "pred_field; omega is curl-extracted once on the unchanged native grid"
                ),
                "off_grid_curl_evaluated": False,
            },
            "anchor_indices": torch.stack(anchor_indices),
            "anchor_weights": torch.stack(anchor_weights),
            "score_trace_by_probe": score_trace_rows,
            "work_counts_by_probe": work_rows,
            "shared_fixed_work_counts": {
                **{f"prepare_{name}": int(value)
                   for name, value in prepared.organization.work_counts.items()},
                **{f"native_{name}": int(value) for name, value in native_counts.items()},
            },
            "prepared_context_reused_for_all_probes": True,
            "locator_rebuilt_for_each_probe": True,
            "physical_interface_control": {
                "kind": "artificial_coverage_face",
                "prescribed_controls_held_fixed": True,
                "native_interpolation_uses_fixed_cell_center_grid": True,
                "native_curl_evaluated_only_on_fixed_grid": True,
                "native_off_grid_validity_mask_required": True,
            },
        }

    # Compatibility for evaluator code written against the former public name.
    def member_routing_trace(self, model, prediction, query_indices):
        return self.anchored_organization_trace(model, prediction, query_indices)


def build_thermal_r13_8_anchored_member_routing_task(config, *, arm=None, device="cpu"):
    """Build the strict A/L adapter and maintained native TRAIN/VAL provider."""
    detail = arm == ARM_DETAIL
    schema = DETAIL_CORE_SCHEMA if detail else CORE_SCHEMA
    return _build_thermal_member_routing_task(
        config, arm=arm, device=device,
        task_type=ThermalR138AnchoredMemberRoutingTask,
        variants={schema: {**_VARIANTS[schema], "arm": str(arm)}},
        section_schema=DETAIL_CONFIG_SCHEMA if detail else CONFIG_SECTION_SCHEMA,
        source_kind=DETAIL_SOURCE_KIND if detail else SOURCE_KIND,
        lineage="R14 detail" if detail else "R13-8 A/L",
        required_address_mode=None,
    )


__all__ = [
    "ARM_A", "ARM_L", "CONFIG_SECTION_SCHEMA", "CORE_SCHEMA", "PROVIDER_SCHEMA",
    "SOURCE_KIND", "TRAINING_STATE_SCHEMA", "WORKFLOW",
    "ThermalR138AnchoredMemberRoutingTask",
    "build_thermal_r13_8_anchored_member_routing_task",
]
