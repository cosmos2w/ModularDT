"""Versioned Thermal v2 provider for the shared content-hypergraph engine."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from honf_runtime.content_hypergraph_formal import FORMAL_PROTOCOL, validate_formal_d_recipe
from honf_runtime.source_read_protocol import (
    SOURCE_READ_DEVELOPMENT,
    SOURCE_READ_FULL,
    SOURCE_READ_VARIANTS,
    validate_source_read_recipe,
)
from honf_runtime.unified_training import OptimizerGroupSpec, SamplingKey, ScheduleSpec

from ..content_hypergraph import (
    THERMAL_CONTENT_COORDINATE_FRAME_V2,
    THERMAL_CONTENT_GEOMETRY_CONTRACT_V2,
    THERMAL_CONTENT_NATIVE_LAW_CONTRACT_V2,
    ThermalContentHypergraphAdapterV2,
)
from ..data.datasets import fit_global_normalizer
from ..data.development_split import development_case_ids, read_case_catalog
from .content_hypergraph_task import (
    THERMAL_CONTENT_FULL_TRAIN_DATASET,
    THERMAL_CONTENT_FULL_TRAIN_PROTOCOL,
    THERMAL_CONTENT_PROVIDER_V2_SCHEMA,
    THERMAL_CONTENT_RECIPE_V2_SCHEMA,
    ThermalContentHypergraphTask,
    ThermalContentPredictions,
    ThermalContentReceivers,
    ThermalContentSceneInputs,
    _ids_sha256,
    _stable_sha256,
)
from .joint_task import _load_data_binding, _read_selected_cases, validate_full_train_followup_membership
from .unified_task import (
    DEFAULT_ATLAS_DIRECTORY,
    FIXED25_FINGERPRINT,
    TRAIN_RESPONSE_IDS,
    _read_response_families,
)

_MODEL_FIELDS = {
    "module_input_dim": 8,
    "environment_input_dim": 8,
    "coordinate_dim": 2,
    "query_input_dim": 0,
    "operating_input_dim": 14,
    "output_dim": 3,
    "context_dim": 128,
    "num_heads": 4,
    "context_depth": 2,
    "num_hyperedges": 6,
    "thermal_response_rank": 4,
    "local_hidden_dim": 128,
    "dropout": 0.0,
    "init_seed": 0,
}
_TYPED_CENTROID_MODE = "typed_w_centroids_v1"
_OPTIMIZER_FIELDS = {"weight_decay": 1.0e-4, "bias_and_norm_weight_decay": 0.0}


class ThermalContentHypergraphTaskV2(ThermalContentHypergraphTask):
    """Reuse the sealed Thermal targets, sampling, and objective under v2 identity."""

    def identity_payload(self) -> Mapping[str, Any]:
        payload = dict(super().identity_payload())
        payload["schema"] = THERMAL_CONTENT_PROVIDER_V2_SCHEMA
        payload["coordinate_frame"] = {
            **dict(payload["coordinate_frame"]),
            "name": THERMAL_CONTENT_COORDINATE_FRAME_V2,
            "origin": "[0,0]",
            "extent": "[Lx,Ly]",
        }
        payload["thermal_v2_geometry"] = copy.deepcopy(THERMAL_CONTENT_GEOMETRY_CONTRACT_V2)
        payload["thermal_v2_native_laws"] = copy.deepcopy(THERMAL_CONTENT_NATIVE_LAW_CONTRACT_V2)
        payload["provider_inheritance"] = {
            "dataset_membership": "unchanged_from_thermal_content_hypergraph_provider_v1",
            "query_sampling": "case_epoch_v1_unchanged",
            "response_addendum": "original_train_families_and_scales_unchanged",
            "objective": "native_flow_temperature_flux_and_response_weights_unchanged",
        }
        return payload

    def preparation_summary(self) -> Mapping[str, Any]:
        return {
            **dict(super().preparation_summary()),
            "provider_schema": THERMAL_CONTENT_PROVIDER_V2_SCHEMA,
            "adapter_schema": "thermal_content_hypergraph_adapter_v2",
            "geometry_contract": copy.deepcopy(THERMAL_CONTENT_GEOMETRY_CONTRACT_V2),
            "optimizer_bias_and_norm_weight_decay": 0.0,
        }

    def predict_native(
        self,
        model: nn.Module,
        scene: ThermalContentSceneInputs,
        receivers: ThermalContentReceivers,
        execution_mode: str,
        phase: str,
        epoch: int = 0,
        temperature: float | None = None,
    ) -> tuple[ThermalContentPredictions, Mapping[str, Any]]:
        del epoch, temperature
        if model is not self.model or execution_mode not in ("A2-D", "A2-H6"):
            raise ValueError("Thermal v2 prediction requires the bound A2-D/A2-H6 model.")
        if model.architecture != execution_mode or phase not in ("joint", "validation", "development"):
            raise ValueError("Thermal v2 prediction phase or architecture differs from its recipe.")
        prepared = model.prepare_native(
            scene.structure,
            receivers.fluid_xy,
            local_query_points=receivers.local_query_points,
            ntheta=receivers.ntheta,
            scene_key=scene.case_ids,
        )
        main, primary_decode_work = model.apply_native_with_work_counts(prepared, receivers.heat)
        response_prepared = response_main = None
        response_decode_work: Mapping[str, int] = {}
        if receivers.response_fluid_xy is not None:
            if (
                scene.response_structure is None
                or receivers.response_local_query_points is None
                or receivers.response_heat is None
            ):
                raise ValueError("Thermal v2 TRAIN response addendum is missing geometry or heat controls.")
            response_prepared = model.prepare_native(
                scene.response_structure,
                receivers.response_fluid_xy,
                local_query_points=receivers.response_local_query_points,
                ntheta=receivers.response_ntheta,
                scene_key=("original_train_response",),
            )
            response_main, response_decode_work = model.apply_native_with_work_counts(
                response_prepared, receivers.response_heat, increment=True,
            )
        work = {
            "native_fluid_queries": int(receivers.fluid_xy.shape[0] * receivers.fluid_xy.shape[1]),
            "native_surface_receiver_slots": int(prepared.stencils["surface"].valid.numel()),
            "native_outside_receiver_slots": int(prepared.stencils["outside"].valid.numel()),
            "native_material_receiver_slots": int(prepared.stencils["material"].valid.numel()),
            "native_union_rows": int(prepared.union.unique_counts.sum().item()),
            "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
            "response_fluid_queries": 0 if receivers.response_fluid_xy is None else int(
                receivers.response_fluid_xy.shape[0] * receivers.response_fluid_xy.shape[1]
            ),
            "response_surface_receiver_slots": 0 if response_prepared is None else int(
                response_prepared.stencils["surface"].valid.numel()
            ),
            "response_outside_receiver_slots": 0 if response_prepared is None else int(
                response_prepared.stencils["outside"].valid.numel()
            ),
            "response_material_receiver_slots": 0 if response_prepared is None else int(
                response_prepared.stencils["material"].valid.numel()
            ),
            "response_native_union_rows": 0 if response_prepared is None else int(
                response_prepared.union.unique_counts.sum().item()
            ),
            "response_native_local_pairs": 0 if response_prepared is None else int(
                response_prepared.query_batch.local_weights.numel()
            ),
        }

        def add_core_counts(prefix: str, values: Mapping[str, int]) -> None:
            for name, value in values.items():
                if not isinstance(name, str) or type(value) is not int or value < 0:
                    raise RuntimeError("Thermal v2 core returned an invalid work-count mapping.")
                work[f"{prefix}_{name}"] = value

        primary_prepare_work = prepared.organization.work_counts
        if not isinstance(primary_prepare_work, Mapping):
            raise RuntimeError("Thermal v2 primary preparation omitted core work counters.")
        add_core_counts("core_prepare_primary", primary_prepare_work)
        add_core_counts("core_decode_primary", primary_decode_work)
        if response_prepared is not None:
            response_prepare_work = response_prepared.organization.work_counts
            if not isinstance(response_prepare_work, Mapping):
                raise RuntimeError("Thermal v2 response preparation omitted core work counters.")
            add_core_counts("core_prepare_response", response_prepare_work)
            add_core_counts("core_decode_response_increment", response_decode_work)
        predictions = ThermalContentPredictions(prepared, main, response_prepared, response_main, work)
        return predictions, {"work": work}

    def optimizer_groups(self, model: nn.Module, arm: str, stage: str):
        groups = tuple(super().optimizer_groups(model, arm, stage))
        named = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        no_decay_names = {name for group in groups if group.weight_decay == 0.0 for name in group.parameter_names}
        if no_decay_names != {
            name for name in named if name.endswith(".bias") or "norm" in name.lower()
        }:
            raise ValueError("Thermal v2 optimizer must explicitly exclude every bias and normalization parameter from decay.")
        if any(group.weight_decay not in (0.0, 1.0e-4) for group in groups):
            raise ValueError("Thermal v2 optimizer has an undeclared weight-decay value.")
        return groups


class ThermalFormalContentHypergraphTaskV1(ThermalContentHypergraphTaskV2):
    """Full-TRAIN A2-D provider with the historical constant AdamW schedule."""

    def optimizer_groups(self, model: nn.Module, arm: str, stage: str):
        del stage
        if model is not self.model or arm != "A2-D" or model.architecture != "A2-D":
            raise ValueError("Thermal formal optimizer requires its bound A2-D model.")
        names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if not names:
            raise ValueError("Thermal formal A2-D has no trainable parameters.")
        schedule = ScheduleSpec(
            peak_lr=3.0e-4,
            warmup_start_lr=3.0e-4,
            warmup_epochs=0,
            hold_through_epoch=4999,
            total_epochs=5000,
            final_lr=3.0e-4,
        )
        return (OptimizerGroupSpec("formal_joint", names, schedule, weight_decay=1.0e-5),)

    def identity_payload(self) -> Mapping[str, Any]:
        payload = dict(super().identity_payload())
        training = dict(payload["training"])
        training.update({
            "optimizer_schedule": {
                "kind": "constant",
                "learning_rate": 3.0e-4,
                "total_epochs": 5000,
                "historical_anchor": "Run_1804_and_Run_1404_saved_epoch_0500_and_epoch_5000_AdamW_state",
            },
            "weight_decay": 1.0e-5,
            "bias_and_norm_weight_decay": 1.0e-5,
            "bias_and_norm_decay_exclusion": False,
        })
        payload["training"] = training
        objective = dict(payload["objective"])
        objective["historical_objective_compatibility"] = {
            "compatible": False,
            "reason": (
                "A2-D exposes native flow, affine temperature, heat-null flow, and q-proxy/response laws; "
                "the historical models' port/interface heads have no compatible A2-D output heads."
            ),
            "formal_objective": "current equal-case native flow-temperature-flux and TRAIN response objective",
        }
        payload["objective"] = objective
        payload["formal_contract"] = {
            "protocol": FORMAL_PROTOCOL,
            "horizon_epochs": 5000,
            "approved_stop_after": 5000,
            "normalization_scope": "all_original_TRAIN600_only",
            "validation_scope": "canonical89_exposed_original_test_excluding_0273",
        }
        return payload

    def preparation_summary(self) -> Mapping[str, Any]:
        return {
            **dict(super().preparation_summary()),
            "protocol": FORMAL_PROTOCOL,
            "optimizer": "AdamW",
            "learning_rate_schedule": "constant_3e-4_through_epoch_5000",
            "weight_decay": 1.0e-5,
            "bias_and_norm_weight_decay": 1.0e-5,
            "optimizer_bias_norm_exclusion": False,
            "historical_port_interface_objective_compatible": False,
        }


class ThermalSourceReadTaskV1(ThermalContentHypergraphTaskV2):
    """Thermal native task with Round3 cohorts and the successful formal optimizer."""

    def __init__(
        self,
        *,
        source_read_protocol: str,
        source_reader_variant: str,
        reader_execution_backend: str,
        source_value_conditioning: str,
        **kwargs: Any,
    ) -> None:
        if source_read_protocol not in (SOURCE_READ_DEVELOPMENT, SOURCE_READ_FULL):
            raise ValueError("Thermal source-read task requires a declared Round3 protocol.")
        if source_reader_variant not in SOURCE_READ_VARIANTS:
            raise ValueError("Thermal source-read task has an unknown reader variant.")
        host_variant = source_reader_variant
        if source_reader_variant in ("S-Utility", "S-Complete"):
            recipe = kwargs.get("recipe")
            host_variant = recipe.get("dense_host_variant") if isinstance(recipe, Mapping) else None
            if host_variant not in ("D0", "D-scale"):
                raise ValueError("Thermal sparse source-read task requires its declared dense host variant.")
        expected_conditioning = "source_scale" if host_variant == "D-scale" else "none"
        if source_value_conditioning != expected_conditioning:
            raise ValueError("Thermal source-read variant and value conditioning disagree.")
        if reader_execution_backend not in ("legacy", "reduced_projection"):
            raise ValueError("Thermal source-read task has an unknown execution backend.")
        self.source_read_protocol = source_read_protocol
        self.source_reader_variant = source_reader_variant
        self.reader_execution_backend = reader_execution_backend
        self.source_value_conditioning = source_value_conditioning
        super().__init__(**kwargs)
        expected_parent_protocol = "development" if source_read_protocol == SOURCE_READ_DEVELOPMENT else FORMAL_PROTOCOL
        if self.protocol != expected_parent_protocol:
            raise ValueError("Thermal source-read task was bound through the wrong native cohort path.")

    def _set_source_read_training_epoch(self, model: nn.Module, epoch: int) -> None:
        if model is not self.model:
            raise ValueError("Thermal source-read epoch dispatch requires its bound model.")
        core = getattr(model, "core", None)
        setter = getattr(core, "set_source_read_training_epoch", None)
        if not callable(setter):
            raise RuntimeError("Thermal source-read core is missing its epoch-bound selection transition.")
        setter(epoch)

    def predict_native(
        self,
        model: nn.Module,
        scene: ThermalContentSceneInputs,
        receivers: ThermalContentReceivers,
        execution_mode: str,
        phase: str,
        epoch: int = 0,
        temperature: float | None = None,
    ) -> tuple[ThermalContentPredictions, Mapping[str, Any]]:
        # Source-read scopes alone carry the persisted 500..600 selection
        # transition. Epoch 0 remains a core-defined no-op on inherited state.
        self._set_source_read_training_epoch(model, epoch)
        return super().predict_native(
            model, scene, receivers, execution_mode, phase,
            epoch=epoch, temperature=temperature,
        )

    def _source_read_counter_snapshot(self) -> dict[str, Any]:
        return {
            name: copy.deepcopy(getattr(self, name))
            for name in ("physical_training_samples", "physical_training_queries", "training_samples", "training_queries")
            if hasattr(self, name)
        }

    def _restore_source_read_counters(self, snapshot: Mapping[str, Any]) -> None:
        for name, value in snapshot.items():
            setattr(self, name, copy.deepcopy(value))

    def source_read_positive_train_envelope(self, train_case_keys: Sequence[Any]) -> float:
        """Return the preregistered positive envelope from valid physical TRAIN heating."""
        keys = tuple(str(value) for value in train_case_keys)
        if keys != self.training_case_ids:
            raise ValueError("Thermal source-read envelope requires the exact ordered sealed TRAIN cohort.")
        maximum = 0.0
        for case in self.training_cases:
            structure = case.get("structure")
            if not isinstance(structure, Mapping):
                raise ValueError("Thermal TRAIN case is missing its native source structure.")
            heat = torch.as_tensor(structure.get("heat_powers"), dtype=torch.float64)
            present = torch.as_tensor(structure.get("module_present"), dtype=torch.bool)
            if heat.ndim != 1 or present.shape != heat.shape:
                raise ValueError("Thermal TRAIN heat and source-presence arrays must be aligned vectors.")
            valid = heat[present]
            if valid.numel() == 0 or not torch.isfinite(valid).all():
                raise ValueError("Thermal TRAIN cohort has missing or nonfinite valid physical heating.")
            maximum = max(maximum, float(valid.abs().max().item()))
        envelope = 2.0 * maximum
        if not math.isfinite(envelope) or envelope <= 0:
            raise ValueError("Thermal TRAIN physical heating does not define a positive finite envelope.")
        return envelope

    def source_read_teacher_trace(
        self,
        model: nn.Module,
        case_key: Any,
        key: SamplingKey,
        max_query_rows: int,
    ) -> Mapping[str, torch.Tensor | None]:
        """Build one bounded, target-free trace over a native Thermal query union."""
        if model is not self.model or str(case_key) not in self._train_index_by_id:
            raise ValueError("Thermal source-read teacher probes require this model and a sealed TRAIN case.")
        initial_key = key.epoch == 500 and key.stage == "source_read_auxiliary_initial"
        refresh_key = 501 <= key.epoch <= 2500 and key.stage == "source_read_auxiliary_refresh"
        if (not (initial_key or refresh_key) or key.dataset_id != self.sampling_dataset_id):
            raise ValueError(
                "Thermal source-read teacher keys must be an e500 initial fit or a bounded "
                "501..2500 current-model TRAIN refresh."
            )
        from dataclasses import replace
        from honf_runtime.source_read_auxiliary import select_evenly_spaced_teacher_rows

        counters = self._source_read_counter_snapshot()
        try:
            self._set_source_read_training_epoch(model, key.epoch)
            # A nonzero microbatch index prevents make_batch from attaching the
            # epoch's optional response-family addendum to this single-case probe.
            probe_key = replace(key, microbatch_index=1)
            batch = self.make_batch((str(case_key),), probe_key)
            scene = self.make_scene(batch.scene_inputs)
            prepared = model.prepare_native(
                scene.structure,
                batch.receivers.fluid_xy,
                local_query_points=batch.receivers.local_query_points,
                ntheta=batch.receivers.ntheta,
                scene_key=scene.case_ids,
            )
            query = select_evenly_spaced_teacher_rows(prepared.query_batch, max_query_rows)
            trace_method = getattr(model.core, "source_read_teacher_trace_v1", None)
            if not callable(trace_method):
                raise RuntimeError("Thermal A2-D core is missing the bounded Round3 teacher trace API.")
            trace = trace_method(prepared.organization, query)
            if not isinstance(trace, Mapping) or not torch.is_tensor(trace.get("thermal_k")):
                raise ValueError("Thermal teacher trace must include the native signed thermal coefficients.")
            return trace
        finally:
            self._restore_source_read_counters(counters)

    def source_read_physical_calibration_loss(
        self,
        model: nn.Module,
        case_key: Any,
        key: SamplingKey,
    ) -> torch.Tensor:
        """Evaluate the unchanged all-full native TRAIN objective for one case."""
        if model is not self.model or str(case_key) not in self._train_index_by_id:
            raise ValueError("Thermal physical calibration requires this model and a sealed TRAIN case.")
        if key.epoch != 500 or key.dataset_id != self.sampling_dataset_id:
            raise ValueError("Thermal physical calibration keys must bind dense e500 TRAIN.")
        from dataclasses import replace

        counters = self._source_read_counter_snapshot()
        try:
            probe_key = replace(key, update_index=0, microbatch_index=1)
            batch = self.make_batch((str(case_key),), probe_key)
            scene = self.make_scene(batch.scene_inputs)
            predictions, auxiliary = self.predict_native(
                model, scene, batch.receivers, "A2-D", "joint", epoch=500,
            )
            terms = self.loss_terms(predictions, batch.targets, "joint", auxiliary)
            objective: torch.Tensor | None = None
            for name, term in terms.items():
                denominator = float(term.denominator.detach().item()) if torch.is_tensor(term.denominator) else float(term.denominator)
                weight = float(term.weight)
                if denominator < 0 or not math.isfinite(denominator) or not math.isfinite(weight):
                    raise ValueError(f"Thermal physical calibration term {name!r} has invalid scaling.")
                if denominator == 0:
                    if float(term.numerator.detach()) != 0.0:
                        raise ValueError(f"Thermal physical calibration term {name!r} has no valid elements.")
                    continue
                piece = term.numerator.reshape(()) * (weight / denominator)
                objective = piece if objective is None else objective + piece
            if objective is None or not objective.requires_grad or not torch.isfinite(objective):
                raise ValueError("Thermal physical calibration did not produce a differentiable native loss.")
            return objective
        finally:
            self._restore_source_read_counters(counters)

    def optimizer_groups(self, model: nn.Module, arm: str, stage: str):
        del stage
        if model is not self.model or arm != "A2-D" or model.architecture != "A2-D":
            raise ValueError("Thermal source-read optimizer requires its bound A2-D model.")
        names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if not names:
            raise ValueError("Thermal source-read A2-D has no trainable parameters.")
        schedule = ScheduleSpec(
            peak_lr=3.0e-4,
            warmup_start_lr=3.0e-4,
            warmup_epochs=0,
            hold_through_epoch=self.total_epochs - 1,
            total_epochs=self.total_epochs,
            final_lr=3.0e-4,
        )
        return (OptimizerGroupSpec("source_read_joint", names, schedule, weight_decay=1.0e-5),)

    def _source_read_profile(self) -> str:
        return (
            "fixed25_v1_TRAIN150_DEV22"
            if self.source_read_protocol == SOURCE_READ_DEVELOPMENT
            else "original_TRAIN600_canonical89"
        )

    def _source_read_scope_payload(self) -> dict[str, Any]:
        full = self.source_read_protocol == SOURCE_READ_FULL
        return {
            "protocol": self.source_read_protocol,
            "dataset_profile": self._source_read_profile(),
            "source_reader_variant": self.source_reader_variant,
            "reader_execution_backend": self.reader_execution_backend,
            "source_value_conditioning": self.source_value_conditioning,
            "normalization_scope": (
                "all original TRAIN600 only" if full else "fixed25 selected TRAIN150 only"
            ),
            "future_horizon_epochs": self.total_epochs,
            "approved_stop_after": 1000 if full else 2500,
            "execution_cadence": copy.deepcopy(dict(self.recipe["execution"])),
        }

    def identity_payload(self) -> Mapping[str, Any]:
        payload = dict(super().identity_payload())
        payload["protocol"] = self.source_read_protocol
        payload["dataset_profile"] = self._source_read_profile()
        payload["source_reader_variant"] = self.source_reader_variant
        payload["source_read_contract"] = self._source_read_scope_payload()
        payload["source_binding"] = {
            **dict(payload["source_binding"]),
            "protocol": self.source_read_protocol,
            "dataset_profile": self._source_read_profile(),
        }
        training = dict(payload["training"])
        training.update({
            "sampling_seed": 0,
            "total_epochs": self.total_epochs,
            "optimizer": "AdamW",
            "optimizer_schedule": {
                "kind": "constant",
                "learning_rate": 3.0e-4,
                "warmup_epochs": 0,
                "hold_through_epoch": self.total_epochs - 1,
                "total_epochs": self.total_epochs,
                "final_learning_rate": 3.0e-4,
            },
            "weight_decay": 1.0e-5,
            "bias_and_norm_weight_decay": 1.0e-5,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "gradient_clip": 1.0,
            "all_parameters_train_from_start": True,
            "source_reader_variant": self.source_reader_variant,
            "reader_execution_backend": self.reader_execution_backend,
            "source_value_conditioning": self.source_value_conditioning,
        })
        payload["training"] = training
        payload["recipe"] = copy.deepcopy(self.recipe)
        return payload

    def preparation_summary(self) -> Mapping[str, Any]:
        full = self.source_read_protocol == SOURCE_READ_FULL
        summary = dict(super().preparation_summary())
        summary.update({
            "protocol": self.source_read_protocol,
            "dataset_profile": self._source_read_profile(),
            "source_reader_variant": self.source_reader_variant,
            "reader_execution_backend": self.reader_execution_backend,
            "source_value_conditioning": self.source_value_conditioning,
            "future_horizon_epochs": self.total_epochs,
            "approved_stop_after": 1000 if full else 2500,
            "normalization_scope": (
                "all original TRAIN600 only" if full else "fixed25 selected TRAIN150 only"
            ),
            "optimizer": "AdamW all parameters, constant 3e-4, weight decay 1e-5",
            "gradient_clip": 1.0,
        })
        return summary


def build_thermal_content_hypergraph_task_v2(
    recipe: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> tuple[ThermalContentHypergraphAdapterV2, ThermalContentHypergraphTaskV2, Mapping[str, Any]]:
    """Build Thermal A2-D/A2-H6 for fixed25, the H6 follow-up, or formal A2-D."""
    if not isinstance(recipe, Mapping) or recipe.get("schema") != THERMAL_CONTENT_RECIPE_V2_SCHEMA:
        raise ValueError("Thermal v2 builder requires schema=content_hypergraph_recipe_v2.")
    formal = recipe.get("protocol") == FORMAL_PROTOCOL
    if formal:
        recipe = validate_formal_d_recipe(recipe)
        if recipe.get("dataset") != "thermal":
            raise ValueError("Thermal formal builder requires dataset=thermal.")
    protocol = str(recipe.get("protocol", ""))
    if protocol not in ("development", "full_train1000", FORMAL_PROTOCOL):
        raise ValueError("Thermal v2 protocol must be development, full_train1000, or formal_d_full_train_v1.")
    architecture = str(recipe.get("architecture", "")).upper()
    if architecture not in ("A2-D", "A2-H6"):
        raise ValueError("Thermal v2 architecture must be A2-D or A2-H6.")
    if recipe.get("dataset") not in ("thermal", "ThermalChannel"):
        raise ValueError("Thermal v2 builder requires dataset=thermal.")
    if recipe.get("model_family") != "honf_content_hypergraph":
        raise ValueError("Thermal v2 recipe must retain model_family=honf_content_hypergraph.")
    if recipe.get("adapter_schema") != "thermal_content_hypergraph_adapter_v2":
        raise ValueError("Thermal v2 recipe must select the versioned Thermal adapter explicitly.")
    if formal:
        if recipe.get("optimizer_bias_norm_exclusion") is not False:
            raise ValueError("Thermal formal recipe must decay bias and normalization parameters.")
    elif recipe.get("optimizer_bias_norm_exclusion") is not True:
        raise ValueError("Thermal v2 recipe must explicitly exclude bias and normalization weights from decay.")
    expected_epochs = 2500 if protocol == "development" else 5000
    if int(recipe.get("total_epochs", -1)) != expected_epochs:
        raise ValueError(f"Thermal v2 {protocol} horizon is fixed at {expected_epochs} epochs.")
    if (
        int(recipe.get("seed", -1)) != 0
        or int(recipe.get("primary_queries", -1)) != 1024
        or int(recipe.get("effective_batch_size", -1)) != 48
    ):
        raise ValueError("Thermal v2 is sealed to seed 0, Q=1024, and effective batch 48.")
    microbatch_size = recipe.get("microbatch_size")
    if type(microbatch_size) is not int or not 1 <= microbatch_size <= 48:
        raise ValueError("Thermal v2 microbatch packing must be an integer in [1, 48].")
    if recipe.get("precision") != "fp32" or recipe.get("edge_access") != "full":
        raise ValueError("Thermal v2 requires FP32 and full edge access.")
    optimizer = recipe.get("optimizer")
    if not formal and (not isinstance(optimizer, Mapping) or dict(optimizer) != {
        "name": "AdamW",
        **_OPTIMIZER_FIELDS,
    }):
        raise ValueError("Thermal v2 requires explicit AdamW decay=1e-4 and bias/norm decay=0.")
    if protocol == "full_train1000" and (
        architecture != "A2-H6"
        or int(recipe.get("approved_stop_after", 0)) != 1000
    ):
        raise ValueError("Thermal v2 full_train1000 is A2-H6 with stop-after 1000.")
    if formal and (architecture != "A2-D" or int(recipe.get("approved_stop_after", 0)) != 5000):
        raise ValueError("Thermal formal protocol requires A2-D with approved_stop_after=5000.")

    model_fields = recipe.get("model")
    if not isinstance(model_fields, Mapping):
        raise ValueError("Thermal v2 model fields must match the sealed explicit constructor defaults.")
    model_fields = dict(model_fields)
    typed_centroids = "query_edge_geometry_mode" in model_fields
    if typed_centroids and (
        model_fields["query_edge_geometry_mode"] != _TYPED_CENTROID_MODE or architecture != "A2-H6"
    ):
        raise ValueError("Thermal typed centroid geometry is supported only for A2-H6 with the declared mode.")
    expected_model_fields = {
        **_MODEL_FIELDS,
        **({"query_edge_geometry_mode": _TYPED_CENTROID_MODE} if typed_centroids else {}),
    }
    if model_fields != expected_model_fields:
        raise ValueError("Thermal v2 model fields must match the sealed explicit constructor defaults.")
    core_config = {
        "schema": "content_hypergraph_core_v2",
        "architecture": architecture,
        **model_fields,
    }

    resolved_manifest, data_path, manifest_path = _load_data_binding(
        recipe.get("manifest"), recipe.get("data_path"),
    )
    base_manifest_sha256 = str(resolved_manifest["manifest_sha256"])
    catalog_source, catalog = read_case_catalog(data_path)
    case_split_by_id = {str(record["case_id"]): str(record["split"]).lower() for record in catalog}
    catalog_ids = tuple(case_split_by_id)
    if protocol == "development":
        train_ids = tuple(development_case_ids(resolved_manifest, "train"))
        validation_ids = tuple(development_case_ids(resolved_manifest, "test"))
        if len(train_ids) != 150 or len(validation_ids) != 22 or resolved_manifest["manifest_sha256"] != FIXED25_FINGERPRINT:
            raise ValueError("Thermal v2 development requires exact fixed25_v1 TRAIN150/DEV22 membership.")
        effective_manifest = dict(resolved_manifest)
        followup_identity = None
    else:
        train_ids = tuple(record["case_id"] for record in catalog if record["split"] == "train")
        original_test_ids = tuple(record["case_id"] for record in catalog if record["split"] == "test")
        if len(original_test_ids) != 90 or original_test_ids.count("0273") != 1:
            raise ValueError("Thermal v2 full follow-up requires original test90 with historical duplicate 0273.")
        validation_ids = tuple(case_id for case_id in original_test_ids if case_id != "0273")
        followup_identity = validate_full_train_followup_membership(train_ids, validation_ids)
        followup_manifest = {
            "execution_protocol": FORMAL_PROTOCOL if formal else THERMAL_CONTENT_FULL_TRAIN_PROTOCOL,
            "dataset_protocol": THERMAL_CONTENT_FULL_TRAIN_DATASET,
            "base_manifest_sha256": base_manifest_sha256,
            "training_case_ids": list(train_ids),
            "training_membership_sha256": followup_identity["training_membership_sha256"],
            "validation_case_ids": list(validation_ids),
            "validation_membership_sha256": followup_identity["validation_membership_sha256"],
            "excluded_validation_id": "0273",
            "excluded_validation_reason": "historical duplicate in original90 validation; preserve canonical89",
        }
        if formal:
            effective_manifest = {
                **resolved_manifest,
                "formal_full_train_v1": followup_manifest,
                "manifest_sha256": _stable_sha256(followup_manifest),
            }
        else:
            effective_manifest = {
                **resolved_manifest,
                "full_train_followup1000": followup_manifest,
                "manifest_sha256": _stable_sha256(followup_manifest),
            }

    normalizer = fit_global_normalizer(data_path, train_ids)
    training_cases = _read_selected_cases(
        data_path, train_ids, split="train", normalizer=normalizer, include_grid=False,
    )
    validation_cases = _read_selected_cases(
        data_path, validation_ids, split="test", normalizer=normalizer, include_grid=False,
    )
    atlas_directory = Path(recipe.get("atlas_directory", DEFAULT_ATLAS_DIRECTORY)).expanduser().resolve()
    train_families = _read_response_families(atlas_directory, TRAIN_RESPONSE_IDS)
    tools_directory = Path(__file__).resolve().parents[4] / "tools"
    import sys

    if str(tools_directory) not in sys.path:
        sys.path.insert(0, str(tools_directory))
    from thermal_source_response_fit import response_scales

    saved_response_scales = response_scales(train_families)
    model = ThermalContentHypergraphAdapterV2(
        normalizer.stats,
        core_config=core_config,
        nx=128,
        ny=64,
        environment_nx=24,
        environment_ny=8,
    ).to(device)
    source_stat = data_path.stat()
    source_binding = {
        "dataset_path": str(data_path),
        "dataset_id": "thermal_channel_global_v1",
        "dataset_size_bytes": int(source_stat.st_size),
        "dataset_mtime_ns": int(source_stat.st_mtime_ns),
        "catalog_metadata_sha256": catalog_source["metadata_sha256"],
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "base_manifest_sha256": base_manifest_sha256,
        "protocol": protocol,
        "normalization_source": "TRAIN-only H5Normalizer fit_global_normalizer",
        "response_family_sources": [
            {"family_id": str(item["family_id"]), "sha256": str(item["source_sha256"])}
            for item in train_families
        ],
        "full_followup_membership": followup_identity,
        "original_test_targets_outside_declared_panel_read": False,
    }
    provider_type = ThermalFormalContentHypergraphTaskV1 if formal else ThermalContentHypergraphTaskV2
    provider = provider_type(
        model=model,
        training_cases=training_cases,
        validation_cases=validation_cases,
        manifest=effective_manifest,
        base_manifest_sha256=base_manifest_sha256,
        data_path=data_path,
        source_binding=source_binding,
        train_families=train_families,
        response_scales=saved_response_scales,
        recipe=recipe,
        seed=0,
        primary_queries=1024,
        microbatch_size=microbatch_size,
        effective_batch_size=48,
        total_epochs=expected_epochs,
        device=device,
        protocol=protocol,
        case_split_by_id=case_split_by_id,
        catalog_case_ids=catalog_ids,
        manifest_path=manifest_path,
        atlas_directory=atlas_directory,
    )
    preparation = provider.preparation_summary()
    return model, provider, preparation


def build_thermal_formal_content_task_v1(
    recipe: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> tuple[ThermalContentHypergraphAdapterV2, ThermalFormalContentHypergraphTaskV1, Mapping[str, Any]]:
    """Build the sealed fresh A2-D full-TRAIN formal profile."""
    if not isinstance(recipe, Mapping) or recipe.get("protocol") != FORMAL_PROTOCOL:
        raise ValueError(f"Thermal formal builder requires protocol={FORMAL_PROTOCOL}.")
    return build_thermal_content_hypergraph_task_v2(recipe, device=device)


def build_thermal_source_read_task_v1(
    recipe: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> tuple[ThermalContentHypergraphAdapterV2, ThermalSourceReadTaskV1, Mapping[str, Any]]:
    """Build one explicit Round3 Thermal scope without changing historical builders."""
    recipe = validate_source_read_recipe(recipe)
    if recipe.get("dataset") != "thermal":
        raise ValueError("Thermal source-read builder requires dataset=thermal.")
    if recipe.get("model_family") != "honf_content_hypergraph":
        raise ValueError("Thermal source-read recipe must retain model_family=honf_content_hypergraph.")
    if recipe.get("adapter_schema") != "thermal_content_hypergraph_adapter_v2":
        raise ValueError("Thermal source-read recipe must select the versioned Thermal adapter explicitly.")

    source_read_protocol = str(recipe["protocol"])
    full = source_read_protocol == SOURCE_READ_FULL
    parent_protocol = FORMAL_PROTOCOL if full else "development"
    total_epochs = int(recipe["total_epochs"])
    model_fields = dict(recipe["model"])
    reader_execution_backend = str(model_fields.get("reader_execution_backend", "legacy"))
    source_value_conditioning = str(model_fields.get("source_value_conditioning", "none"))
    core_config = {
        "schema": "content_hypergraph_core_v2",
        "architecture": "A2-D",
        **model_fields,
    }

    resolved_manifest, data_path, manifest_path = _load_data_binding(
        recipe.get("manifest"), recipe.get("data_path"),
    )
    base_manifest_sha256 = str(resolved_manifest["manifest_sha256"])
    catalog_source, catalog = read_case_catalog(data_path)
    case_split_by_id = {str(record["case_id"]): str(record["split"]).lower() for record in catalog}
    catalog_ids = tuple(case_split_by_id)
    if full:
        train_ids = tuple(record["case_id"] for record in catalog if record["split"] == "train")
        original_test_ids = tuple(record["case_id"] for record in catalog if record["split"] == "test")
        if len(original_test_ids) != 90 or original_test_ids.count("0273") != 1:
            raise ValueError("Thermal source_read_full1000 requires original test90 with duplicate 0273.")
        validation_ids = tuple(case_id for case_id in original_test_ids if case_id != "0273")
        cohort_identity = validate_full_train_followup_membership(train_ids, validation_ids)
        cohort_scope = {
            "protocol": SOURCE_READ_FULL,
            "dataset_protocol": THERMAL_CONTENT_FULL_TRAIN_DATASET,
            "base_manifest_sha256": base_manifest_sha256,
            "training_case_ids": list(train_ids),
            "training_membership_sha256": cohort_identity["training_membership_sha256"],
            "validation_case_ids": list(validation_ids),
            "validation_membership_sha256": cohort_identity["validation_membership_sha256"],
            "excluded_validation_id": "0273",
            "excluded_validation_reason": "historical duplicate; preserve canonical89",
            "normalization_scope": "all original TRAIN600 only",
            "future_horizon_epochs": total_epochs,
            "approved_stop_after": int(recipe["approved_stop_after"]),
        }
    else:
        train_ids = tuple(development_case_ids(resolved_manifest, "train"))
        validation_ids = tuple(development_case_ids(resolved_manifest, "test"))
        if (
            len(train_ids) != 150
            or len(validation_ids) != 22
            or base_manifest_sha256 != FIXED25_FINGERPRINT
        ):
            raise ValueError("Thermal source_read_development_v1 requires exact fixed25_v1 TRAIN150/DEV22.")
        cohort_identity = None
        cohort_scope = {
            "protocol": SOURCE_READ_DEVELOPMENT,
            "dataset_protocol": "fixed25_v1",
            "base_manifest_sha256": base_manifest_sha256,
            "training_case_ids": list(train_ids),
            "training_membership_sha256": _ids_sha256(train_ids),
            "validation_case_ids": list(validation_ids),
            "validation_membership_sha256": _ids_sha256(validation_ids),
            "normalization_scope": "fixed25 selected TRAIN150 only",
            "future_horizon_epochs": total_epochs,
            "approved_stop_after": int(recipe["approved_stop_after"]),
        }
    effective_manifest = {
        **resolved_manifest,
        "source_read_scope": cohort_scope,
        "manifest_sha256": _stable_sha256(cohort_scope),
    }

    normalizer = fit_global_normalizer(data_path, train_ids)
    training_cases = _read_selected_cases(
        data_path, train_ids, split="train", normalizer=normalizer, include_grid=False,
    )
    validation_cases = _read_selected_cases(
        data_path, validation_ids, split="test", normalizer=normalizer, include_grid=False,
    )
    atlas_directory = Path(recipe.get("atlas_directory", DEFAULT_ATLAS_DIRECTORY)).expanduser().resolve()
    train_families = _read_response_families(atlas_directory, TRAIN_RESPONSE_IDS)
    tools_directory = Path(__file__).resolve().parents[4] / "tools"
    import sys

    if str(tools_directory) not in sys.path:
        sys.path.insert(0, str(tools_directory))
    from thermal_source_response_fit import response_scales

    saved_response_scales = response_scales(train_families)
    model = ThermalContentHypergraphAdapterV2(
        normalizer.stats,
        core_config=core_config,
        nx=128,
        ny=64,
        environment_nx=24,
        environment_ny=8,
    ).to(device)
    source_stat = data_path.stat()
    source_binding = {
        "dataset_path": str(data_path),
        "dataset_id": "thermal_channel_global_v1",
        "dataset_size_bytes": int(source_stat.st_size),
        "dataset_mtime_ns": int(source_stat.st_mtime_ns),
        "catalog_metadata_sha256": catalog_source["metadata_sha256"],
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "base_manifest_sha256": base_manifest_sha256,
        "protocol": source_read_protocol,
        "dataset_profile": (
            "original_TRAIN600_canonical89" if full else "fixed25_v1_TRAIN150_DEV22"
        ),
        "normalization_source": (
            "TRAIN-only H5Normalizer fit_global_normalizer over original TRAIN600"
            if full else "TRAIN-only H5Normalizer fit_global_normalizer over fixed25 TRAIN150"
        ),
        "source_reader_variant": str(recipe["source_reader_variant"]),
        "reader_execution_backend": reader_execution_backend,
        "source_value_conditioning": source_value_conditioning,
        "response_family_sources": [
            {"family_id": str(item["family_id"]), "sha256": str(item["source_sha256"])}
            for item in train_families
        ],
        "full_followup_membership": cohort_identity,
        "original_test_targets_outside_declared_panel_read": False,
    }
    provider = ThermalSourceReadTaskV1(
        model=model,
        training_cases=training_cases,
        validation_cases=validation_cases,
        manifest=effective_manifest,
        base_manifest_sha256=base_manifest_sha256,
        data_path=data_path,
        source_binding=source_binding,
        train_families=train_families,
        response_scales=saved_response_scales,
        recipe=recipe,
        seed=int(recipe["seed"]),
        primary_queries=int(recipe["primary_queries"]),
        microbatch_size=int(recipe["microbatch_size"]),
        effective_batch_size=int(recipe["effective_batch_size"]),
        total_epochs=total_epochs,
        device=device,
        protocol=parent_protocol,
        case_split_by_id=case_split_by_id,
        catalog_case_ids=catalog_ids,
        manifest_path=manifest_path,
        atlas_directory=atlas_directory,
        source_read_protocol=source_read_protocol,
        source_reader_variant=str(recipe["source_reader_variant"]),
        reader_execution_backend=reader_execution_backend,
        source_value_conditioning=source_value_conditioning,
    )
    return model, provider, provider.preparation_summary()


__all__ = [
    "ThermalContentHypergraphTaskV2",
    "ThermalFormalContentHypergraphTaskV1",
    "ThermalSourceReadTaskV1",
    "build_thermal_content_hypergraph_task_v2",
    "build_thermal_formal_content_task_v1",
    "build_thermal_source_read_task_v1",
]
