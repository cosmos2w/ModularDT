"""Thermal native and response task for Round13 spatial group functions."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import time
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import torch

from ..spatial_interaction_groups import ThermalSpatialInteractionGroupsAdapter
from .content_hypergraph_task import ThermalContentPredictions
from .contextual_joint_task import ThermalContextualJointTask
from .spatial_interaction_group_objective import (
    conditional_utility_huber,
    merge_named_gradients,
)


class SpatialInteractionGroupObjectiveRuntime:
    """Keep the maintained task hook while replacing legacy policy VJPs."""

    def __init__(self):
        self.state = {"schema": "spatial_group_paired_utility_v1"}

    def collect(self, *_args, **_kwargs):
        return None

    def before_optimizer_step(self, *_args, **_kwargs):
        return None

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        self.state = copy.deepcopy(state)


def _checkpoint_metadata(payload):
    identity = payload.get("experiment_identity")
    if not isinstance(identity, dict):
        raise ValueError("Initialization checkpoint has no maintained experiment identity.")
    provider = identity.get("provider_identity")
    if not isinstance(provider, dict):
        raise ValueError("Initialization checkpoint has no Thermal provider identity.")
    adapter = provider.get("adapter_state")
    if not isinstance(adapter, dict):
        raise ValueError("Initialization checkpoint has no self-contained Thermal adapter state.")
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("Initialization checkpoint has no model state dictionary.")
    provider_state = payload.get("provider_training_state")
    if not isinstance(provider_state, dict):
        raise ValueError("Initialization checkpoint has no provider training state.")
    return identity, provider, adapter, model_state, provider_state


def _load_initialization_checkpoint(config, arm):
    from honf_runtime.compat import load_trusted_checkpoint

    initialization = config["initialization"]
    source = Path(initialization["source_checkpoint"]).expanduser().resolve(strict=True)
    payload = load_trusted_checkpoint(source, map_location="cpu")
    if int(payload.get("epoch", -1)) != int(initialization.get("source_parent_age", -2)):
        raise ValueError("Initialization checkpoint epoch differs from its declared source age.")
    identity, provider, adapter, model_state, provider_state = _checkpoint_metadata(payload)
    kind = initialization.get("source_kind")
    if kind == "balanced_singleton_recovery_epoch100":
        if (int(payload["epoch"]) != 100 or payload.get("arm") != "G-EE"
                or identity.get("workflow") != "interaction_groups_joint_training_v1"
                or identity.get("training_membership") != "fixed25_train150_dev22"
                or provider.get("protocol") != "development"
                or adapter.get("schema") != "thermal_interaction_groups_adapter_v1"
                or adapter.get("core", {}).get("architecture") != "G-EE"
                or adapter.get("core", {}).get("schema") != "interaction_groups_v1"
                or adapter.get("core", {}).get("adaptive_groups") not in (False, None)):
            raise ValueError("Development initialization must be the literal balanced singleton epoch100 checkpoint.")
    elif kind == "same_model_development_epoch100":
        if (int(payload["epoch"]) != 100 or payload.get("arm") != arm
                or identity.get("workflow") != "spatial_interaction_groups_joint_training_v1"
                or identity.get("training_membership") != "fixed25_train150_dev22"
                or provider.get("schema") != "thermal_spatial_interaction_groups_provider_v1"
                or adapter.get("schema") != ThermalSpatialInteractionGroupsAdapter.SCHEMA
                or adapter.get("core", {}).get("architecture") != arm
                or adapter.get("core", {}).get("schema") != ThermalSpatialInteractionGroupsAdapter.CORE_SCHEMA):
            raise ValueError("Full-TRAIN continuation must inherit its matching F13 development epoch100 endpoint.")
    elif kind == "same_model_development_epoch500":
        if (int(payload["epoch"]) != 500 or payload.get("arm") != arm
                or identity.get("workflow") != "spatial_interaction_groups_joint_training_v1"
                or identity.get("training_membership") != "fixed25_train150_dev22"
                or provider.get("schema") != "thermal_spatial_interaction_groups_provider_v1"
                or adapter.get("schema") != ThermalSpatialInteractionGroupsAdapter.SCHEMA
                or adapter.get("core", {}).get("architecture") != arm
                or adapter.get("core", {}).get("schema") != ThermalSpatialInteractionGroupsAdapter.CORE_SCHEMA):
            raise ValueError("Full-TRAIN training must inherit its matching F13 development epoch500 endpoint.")
    else:
        raise ValueError(f"Unsupported spatial-group initialization source kind {kind!r}.")
    response_coefficient = provider_state.get("response_coefficient")
    if (type(response_coefficient) not in (int, float)
            or not math.isfinite(float(response_coefficient)) or float(response_coefficient) <= 0):
        raise ValueError("Initialization checkpoint has no finite positive saved response coefficient.")
    if not isinstance(adapter.get("normalization_stats"), dict) or float(adapter.get("heat_scale", 0.0)) <= 0:
        raise ValueError("Initialization checkpoint lacks its fitted normalizer or heat scale.")
    return source, payload, identity, provider, adapter, model_state, provider_state


def _stratified_case_ids(training_cases):
    by_count = defaultdict(list)
    for case in training_cases:
        count = int((np.asarray(case["structure"]["module_present"]) > 0.5).sum())
        by_count[count].append(str(case["case_id"]))
    for case_ids in by_count.values():
        case_ids.sort()
    return {count: tuple(case_ids) for count, case_ids in sorted(by_count.items())}


def _normalizer_lineage_is_valid(source_kind, provider_identity):
    if source_kind == "balanced_singleton_recovery_epoch100":
        return provider_identity.get("normalization_fit_scope") == "selected TRAIN only"
    if source_kind in (
            "same_model_development_epoch100",
            "same_model_development_epoch500"):
        lineage = provider_identity.get("initialization")
        return (
            provider_identity.get("normalization_fit_scope") ==
            "inherited from the literal epoch100 initialization checkpoint; no refit"
            and isinstance(lineage, dict)
            and lineage.get("source_kind") == "balanced_singleton_recovery_epoch100"
            and lineage.get("normalization_mode") == "inherit_balanced_singleton_train150_stats"
            and lineage.get("normalization_scope") == "selected TRAIN only"
        )
    return False


class ThermalSpatialInteractionGroupTask(ThermalContextualJointTask):
    """Use native Thermal losses for both paired alternatives and response fitting."""

    def __init__(self, *, initialization, initialization_source, initialization_identity,
                 initialization_provider, initialization_report, **kwargs):
        self.initialization = copy.deepcopy(dict(initialization))
        self.initialization_source = str(initialization_source)
        self.initialization_identity = copy.deepcopy(initialization_identity)
        self.initialization_provider = copy.deepcopy(initialization_provider)
        self.initialization_report = copy.deepcopy(initialization_report)
        source_state = kwargs.pop("source_provider_state")
        kwargs["response_families"] = ("0348",)
        super().__init__(**kwargs)
        # The bounded response objective uses TRAIN0348 for both alternating
        # heat and single-geometry exposures. Other atlas cohorts are not loaded.
        self.response_coefficient = float(source_state["response_coefficient"])
        self.state.update({
            "response_coefficient": self.response_coefficient,
            "response_coefficient_source": self.initialization_source,
            "response_calibration_epoch": "inherited",
            "response_microbatches": 0,
            "response_scene_visits": 0,
            "paired_utility_probes": 0,
            "paired_utility_case_forward_presentations": 0,
            "paired_utility_records": [],
            "primary_updates": 0,
            "frontier_exposure_by_epoch": {},
        })
        self._utility_case_strata = _stratified_case_ids(self.training_cases)
        self._utility_stratum_counts = tuple(self._utility_case_strata)

    def _make_organizer(self):
        return SpatialInteractionGroupObjectiveRuntime()

    def _schedule_payload(self):
        schedule_horizon = int(self.recipe["training"].get("optimizer_schedule_horizon", self.total_epochs))
        return {
            "peak_lr": 3e-4,
            "warmup_start_lr": 3e-4,
            "warmup_epochs": 0,
            "hold_through_epoch": schedule_horizon - 1,
            "total_epochs": schedule_horizon,
            "final_lr": 3e-4,
        }

    @property
    def core_adaptive_training(self):
        return self.model.training

    def identity_payload(self):
        identity = super().identity_payload()
        identity.update({
            "schema": "thermal_spatial_interaction_groups_provider_v1",
            "workflow": "spatial_interaction_groups_joint_training_v1",
            "architecture": self.model.architecture,
            "adapter_state": self.model.state_payload(),
            "spatial_interaction_group_schema": ThermalSpatialInteractionGroupsAdapter.CORE_SCHEMA,
            "initialization": {
                "source_kind": self.initialization.get("source_kind"),
                "source_checkpoint": self.initialization_source,
                "source_arm": self.initialization.get("source_arm"),
                "source_parent_age": int(self.initialization.get("source_parent_age", -1)),
                "normalization_mode": self.initialization.get("normalization_mode"),
                "normalization_scope": self.initialization_provider.get("normalization_fit_scope"),
                "parent_training_case_ids": list(self.initialization_provider.get("training_case_ids", ())),
                "tensor_mapping": self.initialization_report,
                "optimizer_state": "fresh_adamw",
            },
            "objective": {
                "native_weights": copy.deepcopy(self.native_weights),
                "joint_fields": ["u", "v", "p", "temperature"],
                "response_training_families": ["0348"],
                "response_every_primary_updates": 4,
                "response_coefficient": self.response_coefficient,
                "response_coefficient_inherited": True,
                "response_shared_gradient_cap_fraction": 0.1,
                "response_cap_shared_parameter_rule": (
                    "module/environment/operating encoders, MM/ME stages excluding EE, "
                    "group reader, fusion trunk, and field head"
                ),
                "response_scale_exploration": False,
                "paired_utility_every_primary_updates": 4,
                "paired_utility_label": "(parent_loss-descendant_loss)/(stop_gradient((parent_loss+descendant_loss)/2)+1e-8)",
                "paired_utility_huber_delta": 1.0,
                "paired_utility_huber_coefficient": 1.0,
                "paired_utility_physical_weight": "one_half_pair_average_per_actual_primary_macro_case",
                "scale_exploration_fraction": 0.25,
                "scale_exploration_depth_distribution": "uniform_available_depth",
                "scale_exploration_shared_keys_across_arms": True,
                "additional_work_pressure": 0.0,
            },
            "normalization_fit_scope": "inherited from the literal epoch100 initialization checkpoint; no refit",
            "prescribed_heat_in_context": True,
            "response_training_family_ids": ["0348"],
            "solver_attempts": 0,
            "test_read": False,
        })
        return identity

    def make_batch(self, case_keys, key):
        batch = super().make_batch(case_keys, key)
        self.model.core.set_frontier_context(
            epoch=key.epoch,
            update=key.update_index,
            seed=key.seed,
            exploration=True,
        )
        return batch

    def predict_native(self, model, scene, receivers, execution_mode, phase,
                       epoch=0, temperature=None, forced_frontier=None):
        if model is not self.model or execution_mode != model.architecture:
            raise ValueError("Wrong spatial interaction-group model/arm.")
        prepared = model.prepare_native(
            scene.structure,
            receivers.fluid_xy,
            physical_heat=receivers.heat,
            local_query_points=receivers.local_query_points,
            ntheta=receivers.ntheta,
            scene_key=scene.case_ids,
            forced_frontier=forced_frontier,
        )
        main, decode = model.apply_native_with_work_counts(prepared, receivers.heat)
        work = {
            "native_fluid_queries": int(receivers.fluid_xy.shape[0] * receivers.fluid_xy.shape[1]),
            "native_union_rows": int(prepared.union.unique_counts.sum()),
            "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
        }
        work.update({f"core_prepare_primary_{name}": int(value)
                     for name, value in prepared.organization.work_counts.items()})
        work.update({f"core_decode_primary_{name}": int(value) for name, value in decode.items()})
        predictions = ThermalContentPredictions(prepared, main, work=work)
        if (phase == "joint" and self.model.training and torch.is_grad_enabled()
                and forced_frontier is None):
            self._record_primary_frontier_exposure(prepared, int(epoch))
        return predictions, {"work": work}

    @staticmethod
    def _empty_frontier_tally():
        return {
            "case_presentations": 0,
            "group_count_histogram": {},
            "group_depth_histogram": {},
            "exploration_depth_histogram": {},
        }

    def _record_primary_frontier_exposure(self, prepared, epoch):
        epoch_key = str(int(epoch))
        epoch_record = self.state["frontier_exposure_by_epoch"].setdefault(
            epoch_key, {"case_presentations": 0, "by_mode": {}, "by_update": {}}
        )
        update_index = int(self.model.core.organization_update.detach().cpu())
        update_record = epoch_record["by_update"].setdefault(
            str(update_index), {"case_presentations": 0, "by_mode": {}}
        )
        frontiers = prepared.organization.layers[0]
        for frontier in frontiers:
            mode = getattr(frontier.frontier, "selection_mode", None)
            if mode not in ("learned", "scale_exploration"):
                raise RuntimeError(f"Primary TRAIN frontier has unexpected selection mode {mode!r}.")
            depth = getattr(frontier.frontier, "exploration_depth", None)
            if mode == "scale_exploration" and type(depth) is not int:
                raise RuntimeError("Scale-exploration frontier omitted its selected hierarchy depth.")
            epoch_record["case_presentations"] += 1
            update_record["case_presentations"] += 1
            for record in (epoch_record, update_record):
                tally = record["by_mode"].setdefault(mode, self._empty_frontier_tally())
                tally["case_presentations"] += 1
                group_count = str(int(frontier.frontier.K))
                tally["group_count_histogram"][group_count] = (
                    tally["group_count_histogram"].get(group_count, 0) + 1
                )
                for node in frontier.frontier.nodes:
                    group_depth = str(int(frontier.frontier.depths[node]))
                    tally["group_depth_histogram"][group_depth] = (
                        tally["group_depth_histogram"].get(group_depth, 0) + 1
                    )
                if depth is not None:
                    depth_key = str(int(depth))
                    tally["exploration_depth_histogram"][depth_key] = (
                        tally["exploration_depth_histogram"].get(depth_key, 0) + 1
                    )

    def _probe_case_id(self, probe_index):
        if not self._utility_stratum_counts:
            raise ValueError("Spatial paired utility requires selected TRAIN cases.")
        stratum_slot = probe_index % len(self._utility_stratum_counts)
        stratum = self._utility_stratum_counts[stratum_slot]
        case_ids = self._utility_case_strata[stratum]
        within_stratum = probe_index // len(self._utility_stratum_counts)
        return stratum, case_ids[within_stratum % len(case_ids)]

    @staticmethod
    def _candidate_id(candidate):
        if isinstance(candidate, Mapping):
            for key in ("node_id", "node", "id"):
                if key in candidate:
                    return int(candidate[key])
        if isinstance(candidate, (tuple, list)) and candidate:
            return int(candidate[0])
        return int(candidate)

    def _utility_node(self, metadata, probe_index):
        metadata = list(metadata)
        if not metadata:
            raise ValueError("Balanced spatial hierarchy exposed no internal utility candidates.")
        # Cycle over input-derived depth/extent bands, then over candidates in
        # each band. This includes candidates below a learned stop and never
        # consults targets or errors when selecting the paired intervention.
        bands = defaultdict(list)
        for row in metadata:
            depth, size = int(row["depth"]), int(row["size"])
            depth_band = 0 if depth <= 2 else 1 if depth <= 5 else 2
            size_band = 0 if size <= 16 else 1 if size <= 64 else 2
            bands[(depth_band, size_band)].append(dict(row))
        ordered_bands = sorted(bands)
        stratum_count = len(self._utility_stratum_counts)
        band_slot = int((probe_index // stratum_count) % len(ordered_bands))
        band = ordered_bands[band_slot]
        candidates = sorted(bands[band], key=lambda row: int(row["node_id"]))
        candidate_slot = int(
            (probe_index // (stratum_count * len(ordered_bands))) % len(candidates)
        )
        detail = dict(candidates[candidate_slot])
        detail.update({
            "candidate_band": list(band),
            "candidate_band_index": band_slot,
            "candidate_index_within_band": candidate_slot,
            "candidate_count_in_band": len(candidates),
            "candidate_count_total": len(metadata),
        })
        return int(detail["node_id"]), detail

    def _pair_forced_frontiers(self, node, probe_index):
        extra_depth = 2 if probe_index % 2 else 1
        descendant_mode = "children" if extra_depth == 1 else "grandchildren"
        return {0: (node, "parent")}, {0: (node, descendant_mode)}, extra_depth

    def _paired_utility_gradients(self, *, epoch, update_index, probe_index,
                                  macro_case_count, named_parameters):
        from honf_runtime.unified_training import SamplingKey

        stratum, case_id = self._probe_case_id(probe_index)
        probe_seed = int(self.seed + probe_index)
        key = SamplingKey(
            probe_seed,
            int(epoch),
            int(update_index),
            int(probe_index),
            "spatial_utility_probe",
            self.model.architecture,
            sampling_version=self.sampling_version,
            dataset_id=self.sampling_dataset_id,
        )
        samples_before = self.training_samples
        try:
            batch = self.make_batch((case_id,), key)
        finally:
            self.training_samples = samples_before
        self.model.core.set_frontier_context(
            epoch=epoch, update=update_index, seed=probe_seed, exploration=False,
        )
        target_free_scene = self.model.scene_batch(
            {
                **batch.scene_inputs.structure,
                "heat_powers": batch.receivers.heat,
            },
            scene_key=batch.scene_inputs.case_ids,
        )
        metadata = self.model.core.utility_candidate_metadata_for_scene(
            target_free_scene, batch_index=0,
        )
        node, node_detail = self._utility_node(metadata, probe_index)

        forced_parent, forced_descendants, extra_depth = self._pair_forced_frontiers(node, probe_index)
        scene = self.make_scene(batch.scene_inputs)
        parent_prediction, parent_auxiliary = self.predict_native(
            self.model, scene, batch.receivers, self.model.architecture, "train",
            epoch=epoch, forced_frontier=forced_parent,
        )
        parent_roles = self._role_losses(parent_prediction, batch.targets)
        parent_loss = sum(self.native_weights[name] * value for name, value in parent_roles.items()).sum()
        parent_gradients = torch.autograd.grad(
            parent_loss, tuple(named_parameters.values()), allow_unused=True,
        )
        features = self.model.core.utility_features(
            parent_prediction.native_prepared.organization,
            node,
            batch_index=0,
            detach=True,
        )
        del parent_prediction

        descendant_prediction, descendant_auxiliary = self.predict_native(
            self.model, scene, batch.receivers, self.model.architecture, "train",
            epoch=epoch, forced_frontier=forced_descendants,
        )
        descendant_roles = self._role_losses(descendant_prediction, batch.targets)
        descendant_loss = sum(
            self.native_weights[name] * value for name, value in descendant_roles.items()
        ).sum()
        descendant_gradients = torch.autograd.grad(
            descendant_loss, tuple(named_parameters.values()), allow_unused=True,
        )
        del descendant_prediction

        physical_gradients = {}
        physical_factor = 0.5 / float(macro_case_count)
        for (name, parameter), parent_gradient, descendant_gradient in zip(
                named_parameters.items(), parent_gradients, descendant_gradients, strict=True):
            if parent_gradient is None and descendant_gradient is None:
                continue
            total = torch.zeros_like(parameter)
            if parent_gradient is not None:
                total.add_(parent_gradient.detach())
            if descendant_gradient is not None:
                total.add_(descendant_gradient.detach())
            physical_gradients[name] = total.mul_(physical_factor)

        raw_advantage = parent_loss.detach() - descendant_loss.detach()
        normalized_label = raw_advantage / ((0.5 * (parent_loss.detach() + descendant_loss.detach())).detach() + 1e-8)
        utility_prediction, utility_loss, utility_gradients = conditional_utility_huber(
            self.model.core, features, normalized_label,
        )
        model_name_by_parameter_id = {
            id(parameter): name for name, parameter in named_parameters.items()
        }
        utility_named_gradients = {
            model_name_by_parameter_id[id(parameter)]: gradient
            for parameter, gradient in utility_gradients
            if id(parameter) in model_name_by_parameter_id
        }
        if len(utility_named_gradients) != len(utility_gradients):
            raise RuntimeError("Could not map conditional utility gradients to their named core parameters.")

        def role_values(roles):
            return {name: float(value.detach().mean()) for name, value in roles.items()}

        probe_record = {
            "probe_index": int(probe_index),
            "epoch": int(epoch),
            "update_index": int(update_index),
            "case_id": case_id,
            "module_count_stratum": int(stratum),
            "node_id": int(node),
            "node_candidate": node_detail,
            "descendant_depth": int(extra_depth),
            "parent_native_loss": float(parent_loss.detach()),
            "descendant_native_loss": float(descendant_loss.detach()),
            "raw_signed_advantage": float(raw_advantage),
            "normalized_signed_advantage": float(normalized_label.detach()),
            "predicted_normalized_advantage": float(utility_prediction),
            "utility_huber_loss": float(utility_loss),
            "parent_native_components": role_values(parent_roles),
            "descendant_native_components": role_values(descendant_roles),
            "physical_gradient_weight": "0.5 per alternative / actual primary macro case count",
            "macro_case_count": int(macro_case_count),
            "parent_work": dict(parent_auxiliary["work"]),
            "descendant_work": dict(descendant_auxiliary["work"]),
        }
        return physical_gradients, utility_named_gradients, probe_record

    def _response_gradient(self, native, *, index, epoch, update_index):
        from honf_runtime.thermal_soft_organization import gradient_balance, gradient_norm

        self.model.core.set_frontier_context(
            epoch=epoch, update=update_index, seed=self.seed + int(index), exploration=False,
        )
        named = {name: parameter for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        loss, detail = self.responses.loss(self.model, index)
        gradients = torch.autograd.grad(loss, tuple(named.values()), allow_unused=True)
        raw = {
            name: gradient.detach()
            for (name, _parameter), gradient in zip(named.items(), gradients, strict=True)
            if gradient is not None
        }
        shared_modules = (
            "module_encoder.", "environment_encoder.", "operating_encoder.",
            "layers.", "group_reader.", "fusion_trunk.", "field_head.",
        )
        shared = tuple(
            name for name in native
            if name in raw
            and any(token in name for token in shared_modules)
            and ".ee." not in name
            and not any(token in name for token in ("utility_head", "frontier", "policy_head", "cut_head"))
        )
        balance = gradient_balance(native, raw, shared)
        native_norm = balance["native_norm"]
        response_norm = balance["term_norm"]
        cap = min(1.0, 0.1 * native_norm / (self.response_coefficient * response_norm)) \
            if self.response_coefficient * response_norm > 0 else 1.0
        scale = self.response_coefficient * cap
        return loss, detail, raw, scale, {
            "native_shared_norm": native_norm,
            "raw_shared_norm": response_norm,
            "raw_full_norm": gradient_norm(raw),
            "coefficient": self.response_coefficient,
            "cap_scale": cap,
        }

    def _macro_case_count(self, update_index):
        start = int(update_index) * self.effective_batch_size
        return max(0, min(self.effective_batch_size, len(self.training_cases) - start))

    def before_optimizer_step(self, model, optimizer, losses, work, *, epoch, update_index):
        if model is not self.model or model.architecture not in ThermalSpatialInteractionGroupsAdapter.ARCHITECTURES:
            raise ValueError("Wrong spatial interaction-group optimizer owner or architecture.")
        if self.native_weights != self.recipe["objective"]["native_weights"]:
            raise RuntimeError("Spatial-group training must preserve the declared native loss weights.")
        named = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        native = {
            name: parameter.grad.detach().clone()
            for name, parameter in named.items() if parameter.grad is not None
        }
        updates_per_epoch = math.ceil(len(self.training_cases) / self.effective_batch_size)
        primary_update = (epoch - 1) * updates_per_epoch + update_index + 1
        probe_result = None
        response_result = None
        if primary_update % 4 == 0:
            macro_case_count = self._macro_case_count(update_index)
            if macro_case_count <= 0:
                raise ValueError("Spatial paired utility observed an empty primary optimizer macro.")
            probe_index = int(self.state["paired_utility_probes"])
            probe_result = self._paired_utility_gradients(
                epoch=epoch,
                update_index=update_index,
                probe_index=probe_index,
                macro_case_count=macro_case_count,
                named_parameters=named,
            )
            response_index = int(self.state["response_microbatches"])
            response_result = self._response_gradient(
                native, index=response_index, epoch=epoch, update_index=update_index,
            )

        if probe_result is not None:
            physical_gradients, utility_gradients, probe_record = probe_result
            merge_named_gradients(named, physical_gradients)
            merge_named_gradients(named, utility_gradients)
            losses["spatial_paired_native"] = 0.5 * (
                probe_record["parent_native_loss"] + probe_record["descendant_native_loss"]
            ) / probe_record["macro_case_count"]
            losses["spatial_utility_huber"] = probe_record["utility_huber_loss"]
            work["spatial_paired_utility_probes"] = 1
            work["spatial_paired_case_forward_presentations"] = 2
            work["spatial_paired_macro_case_count"] = probe_record["macro_case_count"]
            for prefix, key in (("spatial_paired_parent", "parent_work"),
                                ("spatial_paired_descendant", "descendant_work")):
                for name, value in probe_record[key].items():
                    work[f"{prefix}_{name}"] = float(value)
            self.state["paired_utility_records"].append(probe_record)
            self.state["paired_utility_probes"] += 1
            self.state["paired_utility_case_forward_presentations"] += 2

        if response_result is not None:
            loss, detail, gradients, scale, balance = response_result
            merge_named_gradients(named, {name: gradient * scale for name, gradient in gradients.items()})
            self.state["response_microbatches"] += 1
            self.state["response_scene_visits"] += 2
            self.state.setdefault("response_records", []).append({
                "update": primary_update,
                "kind": detail.get("kind"),
                "loss": float(loss.detach()),
                "exposure": detail.get("exposure", {}),
                "gradient": balance,
            })
            losses["response"] = float(loss.detach()) * scale
            work["response_microbatches"] = 1
            work["response_scene_visits"] = 2

        self.state["primary_updates"] = primary_update
        if self._budget_binding is not None:
            from honf_runtime.unified_training import _atomic_json

            output, budget, reserve = self._budget_binding
            elapsed = self.state.get("training_elapsed_seconds", 0.0) + time.monotonic() - self._segment_started
            if elapsed >= budget - reserve:
                path = output / "CLEAN_STOP_REQUEST.json"
                if not path.exists():
                    _atomic_json(path, {
                        "schema": "ordinary_clean_stop_request_v1",
                        "request_id": "spatial-interaction-groups-budget",
                        "requested_unix": time.time(),
                        "requested_by": "internal_spatial_interaction_groups_budget",
                        "reason": "finite training wall budget reached",
                    })


def build_thermal_spatial_interaction_group_task(config, *, arm=None, device="cpu"):
    """Build a TRAIN150 or explicitly inherited-normalization full-TRAIN task."""
    from ..data.development_split import development_case_ids, read_case_catalog
    from .contextual_joint_task import _load_data_binding, _read_selected_cases
    from .joint_task import validate_full_train_followup_membership

    recipe = copy.deepcopy(dict(config))
    initialization = recipe["initialization"]
    selected_arm = arm or recipe["interaction_groups"]["arm"]
    (source_path, parent_payload, parent_identity, parent_provider,
     parent_adapter, parent_model_state, parent_training_state) = _load_initialization_checkpoint(
        recipe, selected_arm,
    )
    manifest, data_path, manifest_path = _load_data_binding(
        recipe["dataset"]["manifest_path"], recipe["dataset"]["data_path"],
    )
    source_catalog, catalog = read_case_catalog(data_path)
    parent_train_ids = tuple(development_case_ids(manifest, "train"))
    parent_validation_ids = tuple(development_case_ids(manifest, "test"))
    if (tuple(map(str, parent_provider.get("training_case_ids", ()))) != parent_train_ids
            or tuple(map(str, parent_provider.get("validation_case_ids", ()))) != parent_validation_ids
            or parent_provider.get("subset_manifest_sha256") != manifest["manifest_sha256"]
            or not _normalizer_lineage_is_valid(initialization["source_kind"], parent_provider)):
        raise ValueError("Initialization endpoint normalizer is not bound to this exact fixed25 TRAIN150/DEV22 manifest.")
    membership = recipe["training"]["membership"]
    if membership == "fixed25_train150_dev22":
        training_ids = parent_train_ids
        validation_ids = parent_validation_ids
        if initialization.get("normalization_mode") != "inherit_balanced_singleton_train150_stats":
            raise ValueError("Development F13 arms must inherit the balanced-singleton TRAIN150 normalizer.")
    elif membership == "full_train600":
        training_ids = tuple(str(row["case_id"]) for row in catalog if row["split"] == "train")
        validation_ids = tuple(
            str(row["case_id"]) for row in catalog
            if row["split"] == "test" and str(row["case_id"]) != "0273"
        )
        validate_full_train_followup_membership(training_ids, validation_ids)
        if initialization.get("normalization_mode") != "inherit_matching_f13_train150_stats":
            raise ValueError("Full-TRAIN F13 configs must state their inherited TRAIN150 normalization explicitly.")
    else:
        raise ValueError("Spatial-group task supports fixed25 TRAIN150 or full TRAIN600 membership only.")

    normalization_stats = parent_adapter["normalization_stats"]
    from ..data.datasets import H5Normalizer

    normalizer = H5Normalizer(normalization_stats)
    training = _read_selected_cases(
        data_path, training_ids, split="train", normalizer=normalizer, include_grid=False,
    )
    validation = _read_selected_cases(
        data_path, validation_ids, split="test", normalizer=normalizer, include_grid=False,
    )

    model_config = copy.deepcopy(dict(recipe["interaction_groups"]["model"]))
    core_config = {
        "module_input_dim": 9,
        "environment_input_dim": 6,
        "operating_input_dim": 12,
        "coordinate_dim": 2,
        "query_input_dim": 0,
        "output_dim": 4,
        "context_dim": 128,
        "num_heads": 4,
        "context_depth": 2,
        "init_seed": int(initialization.get("seed", 0)),
        **model_config,
        "architecture": selected_arm,
        "schema": ThermalSpatialInteractionGroupsAdapter.CORE_SCHEMA,
    }
    model = ThermalSpatialInteractionGroupsAdapter(
        normalization_stats,
        core_config=core_config,
        heat_scale=float(parent_adapter["heat_scale"]),
    ).to(device)
    if initialization["source_kind"] == "balanced_singleton_recovery_epoch100":
        from honf_forward_core.spatial_interaction_groups import map_compatible_parent_tensors

        parent_core_state = {
            name.removeprefix("core."): value
            for name, value in parent_model_state.items()
            if name.startswith("core.")
        }
        if not parent_core_state:
            raise ValueError("Balanced-singleton initialization has no core tensors to map.")
        mapped_state, mapping_report = map_compatible_parent_tensors(parent_core_state, model.core)
        model.core.load_state_dict(mapped_state, strict=True)
    else:
        if parent_adapter.get("core") != model.get_config().get("core"):
            raise ValueError("Full-TRAIN endpoint core config differs from its matched development arm.")
        model.load_state_dict(parent_model_state, strict=True)
        mapping_report = {
            "copied": sorted(parent_model_state),
            "new": [],
            "dropped": [],
        }
    model.core.set_frontier_context(
        epoch=1,
        update=0,
        seed=int(recipe["training"].get("seed", initialization.get("seed", 0))),
        exploration=None,
    )

    provider = ThermalSpatialInteractionGroupTask(
        model=model,
        training_cases=training,
        validation_cases=validation,
        manifest=manifest,
        data_path=data_path,
        recipe=recipe,
        device=device,
        manifest_path=manifest_path,
        catalog=catalog,
        source_binding={"dataset_path": str(data_path), "existing_catalog": source_catalog},
        initialization=initialization,
        initialization_source=str(source_path),
        initialization_identity=parent_identity,
        initialization_provider=parent_provider,
        initialization_report=mapping_report,
        source_provider_state=parent_training_state,
    )
    return model, provider


__all__ = [
    "ThermalSpatialInteractionGroupTask",
    "build_thermal_spatial_interaction_group_task",
]
