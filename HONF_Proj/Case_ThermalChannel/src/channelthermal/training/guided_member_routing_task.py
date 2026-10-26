"""Maintained native Thermal task for the R13-4 Full/Guide comparison."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from honf_runtime.unified_training import LossTerm
from ..guided_member_routing import ThermalGuidedMemberRoutingAdapter
from .content_hypergraph_task import (
    THERMAL_CONTENT_LOSS_WEIGHTS,
    ThermalContentPredictions,
)
from .member_routing_health import MemberRoutingHealth
from .member_routing_task import (
    RELATIVE_TRAINING_STATE_SCHEMA,
    ThermalInteractionReusedMemberRoutingTask,
    _training_membership,
)
from .joint_task import _load_data_binding, _read_selected_cases
from .contextual_joint_task import _structure_only
from ..data.datasets import fit_global_normalizer
from ..data.development_split import read_case_catalog


WORKFLOW = "interaction_reused_member_routing_joint_training_v3"
PROVIDER_SCHEMA = "thermal_interaction_reused_member_routing_provider_v3"
CONFIG_SECTION_SCHEMA = "thermal_member_routing_r13_4_v1"
TRAINING_STATE_SCHEMA = "thermal_member_routing_native_training_state_v3"
FULL_ARM = "R13-4-Full"
GUIDE_ARM = "R13-4-Guide"


@dataclass
class ThermalGuidedPairPredictions:
    """The two same-parameter training paths and their measured work."""

    full: ThermalContentPredictions
    restricted: ThermalContentPredictions
    work: Mapping[str, int | float]


class GuidedMemberRoutingHealth(MemberRoutingHealth):
    """Observe the deployed arm and a compact paired TRAIN query probe."""

    def scheduled(self, epoch: int) -> bool:
        initial = self.initial_observation
        if not isinstance(initial, Mapping):
            return super().scheduled(epoch)
        origin = int(initial["training_epoch"])
        relative_epoch = int(epoch) - origin
        if relative_epoch <= 0:
            return False
        if relative_epoch <= self.observation_epochs:
            return True
        return (relative_epoch - self.observation_epochs) % self.later_every == 0

    def observe(self, model, *, training_epoch: int) -> dict[str, Any]:
        mode = "restricted" if model.architecture == GUIDE_ARM else "full"
        with model.core.read_execution(mode):
            observation = super().observe(model, training_epoch=training_epoch)
        observation["execution_scope"]["guided_pair_probe"] = (
            "same prepared fixed TRAIN scene; eight fixed sampled fluid queries; Full and restricted Guide reads"
        )
        return observation

    def _run_case(self, model, case_id: str) -> dict[str, Any]:
        # The parent observe() owns RNG, provider counters, mode, clocks and
        # gradient restoration around this callback, including the paired probe.
        record = super()._run_case(model, case_id)
        record["guided_pair_probe"] = self._paired_probe(model, case_id)
        return record

    def _paired_probe(self, model, case_id: str) -> dict[str, Any]:
        from honf_runtime.unified_training import SamplingKey
        from .member_routing_health import _native_field_delta

        provider = self.provider
        key = SamplingKey(
            seed=int(provider.seed), epoch=1, update_index=0, microbatch_index=0,
            stage="joint", arm=str(model.architecture),
            sampling_version=provider.sampling_version,
            dataset_id=provider.sampling_dataset_id,
        )
        batch = provider.make_batch((case_id,), key)
        query_count = min(8, int(batch.receivers.fluid_xy.shape[1]))
        if query_count < 1:
            raise ValueError("R13-4 health requires at least one fixed TRAIN query.")
        indices = torch.arange(query_count, device=batch.receivers.fluid_xy.device)
        probe = replace(
            batch,
            receivers=replace(
                batch.receivers,
                fluid_xy=batch.receivers.fluid_xy.index_select(1, indices),
            ),
            targets=replace(
                batch.targets,
                field_targets=batch.targets.field_targets.index_select(1, indices),
                point_weights=batch.targets.point_weights.index_select(1, indices),
            ),
        )
        prepared = provider._prepare_for_probe(model, probe)
        heat_probe = replace(
            probe,
            receivers=replace(probe.receivers, heat=probe.receivers.heat * 1.1),
        )
        heat_prepared = provider._prepare_for_probe(model, heat_probe)
        rng_cpu = torch.random.get_rng_state()
        device = torch.device(provider.device)
        rng_cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

        def run_read(prepared_scene, physical_heat, mode, *, trace):
            torch.random.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state(rng_cuda, device)
            with model.core.read_execution(mode):
                return model.apply_native_with_work_counts(
                    prepared_scene, physical_heat, return_trace=trace,
                )

        with torch.no_grad():
            full_native, full_counts, full_output = run_read(
                prepared, probe.receivers.heat, "full", trace=True,
            )
            restricted_native, restricted_counts, restricted_output = run_read(
                prepared, probe.receivers.heat, "restricted", trace=True,
            )
            full_heat_native, _full_heat_counts, _full_heat_output = run_read(
                heat_prepared, heat_probe.receivers.heat, "full", trace=False,
            )
            restricted_heat_native, _restricted_heat_counts, _restricted_heat_output = run_read(
                heat_prepared, heat_probe.receivers.heat, "restricted", trace=False,
            )

        full_prediction = ThermalContentPredictions(prepared, full_native, work=full_counts)
        restricted_prediction = ThermalContentPredictions(prepared, restricted_native,
                                                          work=restricted_counts)
        full_losses = provider._role_losses(full_prediction, probe.targets)
        restricted_losses = provider._role_losses(restricted_prediction, probe.targets)
        full_trace = full_output.trace
        restricted_trace = restricted_output.trace
        required_trace = (
            isinstance(full_trace, Mapping)
            and isinstance(restricted_trace, Mapping)
            and torch.is_tensor(full_trace.get("conditional_preference_E"))
            and torch.is_tensor(restricted_trace.get("chosen_candidate_mask_E"))
        )
        if not required_trace:
            raise RuntimeError("R13-4 health is missing its conditional preference/candidate trace.")

        full_preference = full_trace["conditional_preference_E"]
        restricted_candidates = restricted_trace["chosen_candidate_mask_E"]
        if full_preference.shape != restricted_candidates.shape:
            raise ValueError("R13-4 full preference and restricted candidate axes differ.")
        captured_preference = (full_preference * restricted_candidates.to(full_preference)).sum(-1)
        relation_prior = restricted_trace["prior_P_E"]
        prior_mass_captured = (relation_prior * restricted_candidates.to(relation_prior)).sum(-1)
        eligible = restricted_trace["full_eligible_mask_E"]

        def scalar_mean(value: torch.Tensor) -> float:
            return float(value.detach().double().mean().cpu())

        native_losses = {
            name: {
                "full": scalar_mean(full_losses[name]),
                "restricted": scalar_mean(restricted_losses[name]),
                "restricted_minus_full": scalar_mean(restricted_losses[name] - full_losses[name]),
            }
            for name in full_losses
        }
        candidate_rows = int(restricted_counts.get("query_E_cheap_score_rows", 0))
        full_score_rows = int(full_counts.get("query_E_cheap_score_rows", 0))
        return {
            "case_id": str(case_id),
            "query_count": query_count,
            "native_role_loss_normalized_mse": native_losses,
            "prescribed_heat_plus10_fieldwise_native_delta": {
                "full": _native_field_delta(full_native, full_heat_native, model, prepared),
                "restricted": _native_field_delta(
                    restricted_native, restricted_heat_native, model, prepared,
                ),
                "interpretation": "fixed eight-query TRAIN scene sensitivity; paired path values are not causal claims",
            },
            "conditional_preference_mass_captured_mean": scalar_mean(captured_preference),
            "relation_prior_mass_captured_mean": scalar_mean(prior_mass_captured),
            "E_candidate_count_mean": scalar_mean(restricted_candidates.sum(-1).to(torch.float32)),
            "E_eligible_candidate_count_mean": scalar_mean(eligible.sum(-1).to(torch.float32)),
            "M_candidate_count_mean": scalar_mean(
                restricted_trace["chosen_candidate_mask_M"].sum(-1).to(torch.float32)
            ),
            "E_member_score_rows": {
                "full": full_score_rows,
                "restricted": candidate_rows,
                "avoided_before_member_mlp": max(0, full_score_rows - candidate_rows),
            },
            "physical_field_usefulness_is_measured_by_native_role_losses": True,
            "preference_mass_is_not_a_field_error_bound": True,
        }


class ThermalGuidedMemberRoutingTask(ThermalInteractionReusedMemberRoutingTask):
    """Keep the maintained native losses while training Guide on two readers."""

    training_state_schema = TRAINING_STATE_SCHEMA

    def __init__(self, *, model, training_cases, validation_cases, manifest, data_path,
                 recipe, device, manifest_path=None, catalog=(), source_binding=None):
        super().__init__(
            model=model, training_cases=training_cases, validation_cases=validation_cases,
            manifest=manifest, data_path=data_path, recipe=recipe, device=device,
            manifest_path=manifest_path, catalog=catalog, source_binding=source_binding,
        )
        if model.CORE_SCHEMA != ThermalGuidedMemberRoutingAdapter.CORE_SCHEMA:
            raise ValueError("R13-4 Thermal task requires the guided member-routing core.")
        self.workflow = WORKFLOW
        self.provider_schema = PROVIDER_SCHEMA
        self.training_state_schema = TRAINING_STATE_SCHEMA
        self.training_sampling_arm = str(
            recipe.get("training", {}).get("sampling_arm", "R13-3-Relative")
        )
        initialization = recipe.get("initialization", {})
        self.source_kind = str(initialization.get("source_kind", ""))
        self.source_parent_age = int(initialization.get("source_parent_age", 0))
        if self.source_kind not in (
            "relation_guided_e500_transfer", "scratch_relation_guided_member_routing",
        ):
            raise ValueError("R13-4 requires an explicit transferred-parent or scratch source kind.")
        if self.source_parent_age not in (0, 500):
            raise ValueError("R13-4 task source age must be scratch epoch 0 or Relative parent epoch 500.")
        if ((self.source_kind == "relation_guided_e500_transfer") != (self.source_parent_age == 500)):
            raise ValueError("R13-4 source kind and parent age disagree.")
        health_config = recipe.get("member_routing", {}).get("health")
        self._member_routing_health = None
        if isinstance(health_config, Mapping) and health_config.get("enabled") is True:
            self._member_routing_health = GuidedMemberRoutingHealth(self, health_config)

    def identity_payload(self):
        payload = super().identity_payload()
        payload.update({
            "schema": self.provider_schema,
            "workflow": self.workflow,
            "architecture": self.model.architecture,
            "member_routing_schema": self.model.CORE_SCHEMA,
            "adapter_state": self.model.state_payload(),
            "training_sampling_arm": self.training_sampling_arm,
            "loss_lineage": {
                "source_kind": self.source_kind,
                "parent_completed_epoch": self.source_parent_age,
                "new_reader_epochs_start_at": self.source_parent_age + 1,
                "native_loss_roles": list(self.native_weights),
                "guide_path_metrics": ["full_reference", "restricted_guide"],
            },
            "training_reader_objective": (
                "native physical loss for Full; Guide averages the same per-case native physical losses "
                "from sequential Full-reference and restricted reads"
            ),
            "guide_evaluation_execution": "restricted",
            "guide_auxiliary_objective_weight": 0.0,
        })
        return payload

    def source_training_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Convert the R13-3 provider state while resetting health comparison."""
        if not isinstance(state, Mapping):
            raise TypeError("Transferred member-routing provider state must be a mapping.")
        source_schema = state.get("schema")
        if source_schema == TRAINING_STATE_SCHEMA:
            return copy.deepcopy(dict(state))
        if source_schema != RELATIVE_TRAINING_STATE_SCHEMA:
            raise ValueError("R13-4 transfer requires an R13-3 Relative provider training state.")
        converted = copy.deepcopy(dict(state))
        converted["schema"] = TRAINING_STATE_SCHEMA
        converted.pop("member_routing_health", None)
        return converted

    def prepare_transfer_health(self, model, source_epoch: int | None = None) -> dict[str, Any] | None:
        """Record a new health baseline after transferred state and RNG restore."""
        health = self._member_routing_health
        if health is None:
            return None
        if model is not self.model:
            raise ValueError("R13-4 health baseline must observe its owned transferred model.")
        origin = self.source_parent_age if source_epoch is None else int(source_epoch)
        if origin != self.source_parent_age:
            raise ValueError("R13-4 health origin must match the declared provider source age.")
        if health.initial_observation is not None:
            return None
        observation = health.observe(model, training_epoch=origin)
        health.record_initial(observation)
        return observation

    def paired_native_predictions(self, model, prepared, physical_heat=None) -> ThermalGuidedPairPredictions:
        """Run both same-parameter executions over one shared prepared scene."""
        if model is not self.model or model.architecture not in (FULL_ARM, GUIDE_ARM):
            raise ValueError("Paired R13-4 reads require the owned Full/Guide model.")
        with model.core.read_execution("full"):
            full_native, full_counts, _full_output = model.apply_native_with_work_counts(
                prepared, physical_heat,
            )
        with model.core.read_execution("restricted"):
            restricted_native, restricted_counts, _restricted_output = model.apply_native_with_work_counts(
                prepared, physical_heat,
            )
        work = {
            "native_fluid_queries": int(prepared.query_batch.num_queries * prepared.query_batch.batch_size),
            "native_union_rows": int(prepared.union.unique_counts.sum()),
            "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
        }
        work.update({f"core_prepare_primary_{name}": int(value)
                     for name, value in prepared.organization.work_counts.items()})
        work.update({f"core_decode_guide_full_{name}": int(value)
                     for name, value in full_counts.items()})
        work.update({f"core_decode_guide_restricted_{name}": int(value)
                     for name, value in restricted_counts.items()})
        return ThermalGuidedPairPredictions(
            ThermalContentPredictions(prepared, full_native, work=full_counts),
            ThermalContentPredictions(prepared, restricted_native, work=restricted_counts),
            work,
        )

    def completed_training_epoch(self, model, engine, epoch, output, row):
        delegate = getattr(super(), "completed_training_epoch", None)
        if callable(delegate):
            delegate(model, engine, epoch, output, row)
        row["reader_epoch_context"] = {
            "schema": "thermal_guided_reader_epoch_context_v1",
            "source_kind": self.source_kind,
            "source_parent_completed_epoch": self.source_parent_age,
            "absolute_training_epoch": int(epoch),
            "additional_epochs_after_parent": max(0, int(epoch) - self.source_parent_age),
            "new_member_reader_training_begins_after_epoch": self.source_parent_age,
            "loss_curve_epoch_axis": "absolute completed epoch; additional epoch count is explicit",
        }
        work = row.get("work_counts", {})
        prefix = "guide_reader_native_loss_metric__"
        case_count = float(work.get(f"{prefix}case_count", 0.0)) if isinstance(work, Mapping) else 0.0
        if case_count <= 0:
            return
        paths: dict[str, dict[str, float]] = {"full_reference": {}, "restricted_guide": {}}
        for path, metric_path in (("full", "full_reference"), ("restricted", "restricted_guide")):
            path_prefix = f"{prefix}{path}__"
            for key, numerator in work.items():
                if not key.startswith(path_prefix) or not key.endswith("__sum"):
                    continue
                role = key[len(path_prefix):-len("__sum")]
                paths[metric_path][role] = float(numerator) / case_count
        if all(paths.values()):
            row["paired_reader_native_losses"] = {
                "schema": "thermal_guided_paired_native_loss_summary_v1",
                "epoch_context": copy.deepcopy(row["reader_epoch_context"]),
                "per_case_mean_normalized_mse": paths,
                "guide_objective": "arithmetic mean of the matching Full and restricted native role losses",
                "auxiliary_objective_weight": 0.0,
            }

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0,
                       temperature=None, **kwargs):
        del phase, epoch, temperature
        if kwargs:
            raise TypeError(f"Unsupported R13-4 native prediction options: {sorted(kwargs)}")
        if model is not self.model or execution_mode != model.architecture:
            raise ValueError("Wrong R13-4 member-routing model/arm.")
        prepared = model.prepare_native(
            scene.structure,
            receivers.fluid_xy,
            physical_heat=receivers.heat,
            local_query_points=receivers.local_query_points,
            ntheta=receivers.ntheta,
            scene_key=scene.case_ids,
        )
        if model.architecture == GUIDE_ARM and model.training:
            paired = self.paired_native_predictions(model, prepared, receivers.heat)
            work = dict(paired.work)
            work["native_fluid_queries"] = int(receivers.fluid_xy.shape[0] * receivers.fluid_xy.shape[1])
            paired.work = work
            return paired, {"work": dict(paired.work)}

        read_mode = getattr(model.core, "_read_execution_override", None)
        if read_mode is None:
            read_mode = "full" if model.architecture == FULL_ARM else "restricted"
        if read_mode not in ("full", "restricted"):
            raise ValueError(f"Unsupported R13-4 native evaluator read mode {read_mode!r}.")
        with model.core.read_execution(read_mode):
            native, decode, _output = model.apply_native_with_work_counts(prepared, receivers.heat)
        work = {
            "native_fluid_queries": int(receivers.fluid_xy.shape[0] * receivers.fluid_xy.shape[1]),
            "native_union_rows": int(prepared.union.unique_counts.sum()),
            "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
        }
        work.update({f"core_prepare_primary_{name}": int(value)
                     for name, value in prepared.organization.work_counts.items()})
        work.update({f"core_decode_primary_{name}": int(value) for name, value in decode.items()})
        work["member_routing_primary_read_mode_full"] = int(read_mode == "full")
        work["member_routing_primary_read_mode_restricted"] = int(read_mode == "restricted")
        predictions = ThermalContentPredictions(prepared, native, work=work)
        return predictions, {"work": work}

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        if not isinstance(predictions, ThermalGuidedPairPredictions):
            return super().loss_terms(predictions, targets, phase, auxiliary_state)
        if phase != "joint":
            raise ValueError("R13-4 paired native losses are defined only for the joint phase.")
        full_roles = self._role_losses(predictions.full, targets)
        restricted_roles = self._role_losses(predictions.restricted, targets)
        if set(full_roles) != set(restricted_roles) or set(full_roles) != set(self.native_weights):
            raise ValueError("R13-4 Full and restricted native loss roles differ.")
        if isinstance(auxiliary_state, dict):
            auxiliary_state["guide_native_path_losses"] = {
                path: {name: float(values.detach().mean()) for name, values in roles.items()}
                for path, roles in (("full", full_roles), ("restricted", restricted_roles))
            }
        if isinstance(predictions.work, Mapping):
            metric_work = dict(predictions.work)
        else:
            metric_work = {}
        metric_prefix = "guide_reader_native_loss_metric__"
        metric_work[f"{metric_prefix}case_count"] = float(len(targets.case_ids))
        for path, roles in (("full", full_roles), ("restricted", restricted_roles)):
            for name, values in roles.items():
                metric_work[f"{metric_prefix}{path}__{name}__sum"] = float(
                    values.detach().double().sum().cpu()
                )
        predictions.work = metric_work
        return {
            name: LossTerm(
                (full_roles[name].sum() + restricted_roles[name].sum()) * 0.5,
                len(targets.case_ids), float(self.native_weights[name]),
            )
            for name in full_roles
        }

    def work_counts(self, batch, predictions, auxiliary_state):
        del batch, auxiliary_state
        if isinstance(predictions, ThermalGuidedPairPredictions):
            return {str(name): value for name, value in predictions.work.items()}
        if predictions.work is not None:
            return {str(name): value for name, value in predictions.work.items()}
        return {}


def build_thermal_guided_member_routing_task(config, *, arm=None, device="cpu"):
    """Build the R13-4 adapter/provider on the maintained Thermal data path."""
    recipe = copy.deepcopy(dict(config))
    section = recipe.get("member_routing", {})
    model_config = section.get("model")
    selected_arm = str(arm or section.get("arm", ""))
    if section.get("schema") != CONFIG_SECTION_SCHEMA:
        raise ValueError("R13-4 member-routing recipe requires its v1 config-section schema.")
    if not isinstance(model_config, Mapping) or model_config.get("architecture") != selected_arm:
        raise ValueError("R13-4 model config must match the selected named arm.")
    if (model_config.get("schema") != ThermalGuidedMemberRoutingAdapter.CORE_SCHEMA
            or selected_arm not in ThermalGuidedMemberRoutingAdapter.ARCHITECTURES):
        raise ValueError("R13-4 builder requires a supported versioned Full/Guide arm.")

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
    heat_parts = [
        np.asarray(case["structure"]["heat_powers"])[
            np.asarray(case["structure"]["module_present"]) > 0.5
        ] for case in training_cases
    ]
    heat_values = np.concatenate(heat_parts) if heat_parts else np.asarray([0.0])
    heat_scale = max(float(np.sqrt(np.mean(heat_values.astype(np.float64) ** 2))), 1e-6)
    initialization_seed = int(recipe.get("initialization", {}).get("seed", 0))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        model = ThermalGuidedMemberRoutingAdapter(
            normalizer.stats, core_config=model_config, heat_scale=heat_scale,
        ).to(device)
    source_stat = data_path.stat()
    provider = ThermalGuidedMemberRoutingTask(
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


__all__ = [
    "WORKFLOW", "PROVIDER_SCHEMA", "CONFIG_SECTION_SCHEMA", "TRAINING_STATE_SCHEMA",
    "ThermalGuidedPairPredictions", "GuidedMemberRoutingHealth",
    "ThermalGuidedMemberRoutingTask", "build_thermal_guided_member_routing_task",
]
