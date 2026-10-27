"""Maintained native Thermal task for the R13-5A/B member-routing supplement."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from ..data.datasets import fit_global_normalizer
from ..data.development_split import read_case_catalog
from ..supplement_member_routing import (
    ThermalOrganizationConditionedMemberRoutingAdapter,
    ThermalSupplementMemberRoutingAdapter,
)
from .joint_member_routing_task import (
    JointMemberRoutingHealth,
    ThermalJointPairPredictions,
    ThermalJointMemberRoutingTask,
)
from .content_hypergraph_task import ThermalContentPredictions
from .member_routing_task import (
    ThermalInteractionReusedMemberRoutingTask,
    _training_membership,
)
from .joint_task import _load_data_binding, _read_selected_cases


WORKFLOW = "interaction_reused_consistent_joint_member_routing_training_v1"
PROVIDER_SCHEMA = "thermal_r13_5_supplement_member_routing_provider_v1"
CONFIG_SECTION_SCHEMA = "thermal_r13_5_supplement_member_routing_v1"
TRAINING_STATE_SCHEMA = "thermal_r13_5_supplement_member_routing_training_state_v1"
ARM = "R13-5A-ME-ConsistentJoint95"
B_ARM = "R13-5B-ME-OrganizationConditionedJoint95"
B_WORKFLOW = "interaction_reused_organization_conditioned_member_routing_training_v1"
JOINT_BLOCKS = ("H_MM_applied", "H_EM_applied", "H_EE_applied", "H_ME_applied")
CONTRIBUTION_RULE = "joint95 prior_times_likelihood normalized across selected M and E members before typed strengths"
B_CONTRIBUTION_RULE = (
    "organization-conditioned retained-view competition coalesced onto selected original M/E members; "
    "joint contribution normalized before typed strengths"
)

_SUPPLEMENT_VARIANTS = {
    ThermalSupplementMemberRoutingAdapter.CORE_SCHEMA: {
        "adapter": ThermalSupplementMemberRoutingAdapter,
        "arm": ARM,
        "workflow": WORKFLOW,
        "candidate": "A",
        "contribution_rule": CONTRIBUTION_RULE,
    },
    ThermalOrganizationConditionedMemberRoutingAdapter.CORE_SCHEMA: {
        "adapter": ThermalOrganizationConditionedMemberRoutingAdapter,
        "arm": B_ARM,
        "workflow": B_WORKFLOW,
        "candidate": "B",
        "contribution_rule": B_CONTRIBUTION_RULE,
    },
}


class SupplementMemberRoutingHealth(JointMemberRoutingHealth):
    """Retain the native R13-5 health panel and compact A/B member-read masses."""

    STATE_SCHEMA = "thermal_r13_5_supplement_member_routing_health_state_v1"

    def __init__(self, provider, config: Mapping[str, Any]):
        super().__init__(provider, config)
        self._captured_traces: dict[str, Mapping[str, Any]] = {}

    def _native_with_first_eight_trace(self, model, prepared):
        native, work, trace = JointMemberRoutingHealth._native_with_first_eight_trace(
            model, prepared,
        )
        mode = getattr(model.core, "_read_execution_override", None) or "restricted"
        self._captured_traces[str(mode)] = trace
        return native, work, trace

    @staticmethod
    def _mass_summary(trace: Mapping[str, Any]) -> dict[str, Any]:
        before = {}
        after = {}
        for role in ("M", "E"):
            value = trace.get(f"member_contribution_pre_strength_{role}")
            if not torch.is_tensor(value) or value.ndim < 2:
                raise RuntimeError(f"R13-5 supplement trace lacks the pre-strength joint contribution for {role}.")
            if not torch.isfinite(value).all() or (value < 0).any():
                raise FloatingPointError(f"R13-5 supplement pre-strength {role} contribution is not finite and nonnegative.")
            before[role] = value
            scaled = trace.get(f"member_contribution_{role}")
            if not torch.is_tensor(scaled) or scaled.shape != value.shape:
                raise RuntimeError(f"R13-5 supplement trace lacks the typed-strength contribution for {role}.")
            if not torch.isfinite(scaled).all() or (scaled < 0).any():
                raise FloatingPointError(f"R13-5 supplement typed-strength {role} contribution is not finite and nonnegative.")
            after[role] = scaled

        total = before["M"].sum(dim=-1) + before["E"].sum(dim=-1)
        active = total > 0
        if active.any() and not torch.allclose(
                total[active], torch.ones_like(total[active]), rtol=2e-5, atol=2e-6):
            raise RuntimeError("R13-5 supplement pre-strength member contributions do not sum to one jointly across M and E.")
        if (~active).any() and not torch.equal(total[~active], torch.zeros_like(total[~active])):
            raise RuntimeError("R13-5 supplement empty joint member reads must have exactly zero contribution mass.")

        role_masses = {role: value.sum(dim=-1) for role, value in before.items()}
        conditional_mass_error = {}
        for role in ("M", "E"):
            mass = role_masses[role]
            positive = mass > 0
            conditional = torch.where(
                positive.unsqueeze(-1), before[role] / mass.clamp_min(torch.finfo(mass.dtype).tiny).unsqueeze(-1),
                torch.zeros_like(before[role]),
            )
            conditional_sum = conditional.sum(dim=-1)
            if positive.any() and not torch.allclose(
                    conditional_sum[positive], torch.ones_like(conditional_sum[positive]),
                    rtol=2e-5, atol=2e-6):
                raise RuntimeError(f"R13-5 supplement normalized conditional {role} mass is not one when that role is present.")
            if (~positive).any() and torch.count_nonzero(conditional_sum[~positive]).item():
                raise RuntimeError(f"R13-5 supplement empty {role} roles must have zero conditional mass.")
            conditional_mass_error[role] = float(
                (conditional_sum[positive] - 1).abs().max().detach().cpu()
            ) if positive.any() else 0.0
            if (~positive).any() and torch.count_nonzero(after[role][~positive]).item():
                raise RuntimeError(f"R13-5 supplement empty {role} roles must have zero post-strength contribution.")

        result = {
            "joint_pre_strength_mass_mean": float(total.detach().double().mean().cpu()),
            "joint_pre_strength_mass_max_abs_error": float((total[active] - 1).abs().max().detach().cpu())
            if active.any() else 0.0,
            "empty_joint_query_count": int((~active).sum().item()),
            "role_pre_strength_mass_mean": {
                role: float(value.detach().double().mean().cpu()) for role, value in role_masses.items()
            },
            "role_conditional_mass_max_abs_error": conditional_mass_error,
            "role_post_strength_mass_mean": {
                role: float(value.sum(dim=-1).detach().double().mean().cpu())
                for role, value in after.items()
            },
            "trace_query_count": int(total.numel()),
            "interpretation": "pre-strength member weights are normalized jointly across selected M/E records; bounded typed strengths are applied afterward",
        }
        b_trace_keys = (
            "organization_conditioned_retained_view_mass",
            "organization_conditioned_scaled_view_likelihood_mass",
            "organization_conditioned_retained_prior_mass",
            "organization_conditioned_effective_view_weights",
            "organization_conditioned_view_mixture_mass",
            "organization_conditioned_pre_normalization_mass",
            "organization_conditioned_pre_normalization_error",
        )
        present_b_keys = [key for key in b_trace_keys if key in trace]
        if present_b_keys:
            if len(present_b_keys) != len(b_trace_keys):
                raise RuntimeError("R13-5B trace is missing organization-conditioned d/Z/D/effective-view state.")

            def compact_stats(value: torch.Tensor, *, nonnegative: bool = True) -> dict[str, Any]:
                if not torch.is_tensor(value):
                    raise RuntimeError("R13-5B health received a non-tensor organization-conditioned statistic.")
                values = value.detach().to(torch.float64).reshape(-1)
                if not bool(torch.isfinite(values).all()):
                    raise FloatingPointError("R13-5B organization-conditioned health statistic is nonfinite.")
                if nonnegative and bool((values < 0).any()):
                    raise FloatingPointError("R13-5B organization-conditioned mass statistic is negative.")
                if values.numel() == 0:
                    return {"count": 0, "mean": None, "min": None, "max": None,
                            "sum": 0.0, "zero_count": 0}
                return {
                    "count": int(values.numel()),
                    "mean": float(values.mean().cpu()),
                    "min": float(values.min().cpu()),
                    "max": float(values.max().cpu()),
                    "sum": float(values.sum().cpu()),
                    "zero_count": int((values == 0).sum().item()),
                }

            retained = trace[b_trace_keys[0]]
            scaled = trace[b_trace_keys[1]]
            effective = trace[b_trace_keys[3]]
            mixture = trace[b_trace_keys[4]]
            proposal_m, proposal_e = trace.get("proposal_weights_M"), trace.get("proposal_weights_E")
            if not all(torch.is_tensor(value) and value.ndim == 3 for value in
                       (retained, scaled, effective, mixture, proposal_m, proposal_e)):
                raise RuntimeError("R13-5B d/Z/effective-view traces must retain [B,Q,address] axes.")
            if not (retained.shape == scaled.shape == effective.shape == mixture.shape
                    and retained.shape[-1] == proposal_m.shape[-1] + proposal_e.shape[-1]):
                raise RuntimeError("R13-5B view statistics do not share the M/E address inventory.")
            modules = int(proposal_m.shape[-1])

            def by_role(value: torch.Tensor) -> dict[str, dict[str, Any]]:
                return {
                    "all": compact_stats(value),
                    "M": compact_stats(value[..., :modules]),
                    "E": compact_stats(value[..., modules:]),
                }

            result["organization_conditioned_view_summary"] = {
                "retained_view_mass_d": by_role(retained),
                "scaled_view_likelihood_mass_Z": by_role(scaled),
                "retained_prior_mass_D": compact_stats(trace[b_trace_keys[2]]),
                "effective_view_weights": by_role(effective),
                "view_mixture_mass": by_role(mixture),
                "pre_normalization_mass": compact_stats(trace[b_trace_keys[5]]),
                "pre_normalization_mass_minus_one": compact_stats(trace[b_trace_keys[6]], nonnegative=False),
                "trace_query_count": int(trace[b_trace_keys[2]].numel()),
                "interpretation": "B mixes organization rows using retained proposal mass and per-view likelihood normalization; d/Z/D and effective view weights are compact read diagnostics",
            }
        return result

    def _observe_case(self, model, case_id: str) -> dict[str, Any]:
        self._captured_traces = {}
        try:
            case = super()._observe_case(model, case_id)
            for mode in ("full", "restricted"):
                trace = self._captured_traces.get(mode)
                if not isinstance(trace, Mapping):
                    raise RuntimeError(f"R13-5 supplement health did not retain the actual {mode} decode trace.")
                case["modes"][mode]["joint_contribution_mass"] = self._mass_summary(trace)
            return case
        finally:
            self._captured_traces.clear()


class ThermalSupplementMemberRoutingTask(ThermalJointMemberRoutingTask):
    """Reuse the maintained TRAIN data, native objective, paired read and runner hooks."""

    training_state_schema = TRAINING_STATE_SCHEMA

    def __init__(self, **kwargs):
        # Bypass only the older core-schema assertion; all maintained provider
        # initialization and paired physical-loss behavior remain inherited.
        ThermalInteractionReusedMemberRoutingTask.__init__(self, **kwargs)
        variant = _SUPPLEMENT_VARIANTS.get(self.model.CORE_SCHEMA)
        if variant is None or self.model.architecture != variant["arm"]:
            raise ValueError("R13-5 supplement task requires a matching A or B adapter/core/architecture.")
        self._supplement_variant = variant
        self.candidate = str(variant["candidate"])
        self.arm = str(variant["arm"])
        self.workflow = str(variant["workflow"])
        self.provider_schema = PROVIDER_SCHEMA
        self.training_state_schema = TRAINING_STATE_SCHEMA
        initialization = self.recipe["initialization"]
        self.source_kind = str(initialization["source_kind"])
        self.source_parent_age = int(initialization.get("source_parent_age", 0))
        if (self.source_kind != "scratch_r13_5_supplement"
                or self.source_parent_age != 0
                or initialization.get("source_checkpoint") is not None
                or initialization.get("pretrained_weights") is not False):
            raise ValueError(f"R13-5{self.candidate} requires a fresh scratch lineage with no source checkpoint.")
        self.training_sampling_arm = str(
            self.recipe["training"].get("sampling_arm", "R13-3-Relative")
        )
        health_config = self.recipe.get("member_routing", {}).get("health")
        self._member_routing_health = None
        if isinstance(health_config, Mapping) and health_config.get("enabled") is True:
            self._member_routing_health = SupplementMemberRoutingHealth(self, health_config)

    def identity_payload(self):
        payload = super().identity_payload()
        payload.update({
            "schema": self.provider_schema,
            "workflow": self.workflow,
            "architecture": self.model.architecture,
            "member_routing_schema": self.model.CORE_SCHEMA,
            "adapter_state": self.model.state_payload(),
            "candidate": self.candidate,
            "retrieval_calibration": self.model.core.retrieval_calibration,
            "score_calibration": self.model.core.score_calibration,
            "contribution_rule": self._supplement_variant["contribution_rule"],
            "joint_member_coverage": 0.95,
            "native_objective": {
                "paired_full_restricted": True,
                "full_path": ("all eligible original M/E members under the live organization"
                              if self.candidate == "A" else
                              "all eligible original M/E members compete through their live organization views"),
                "restricted_path": ("joint95 prefix selected before query-conditioned member scoring"
                                    if self.candidate == "A" else
                                    "joint95 prefix selected before organization-conditioned member competition"),
                "guide_kl_weight": 0.0,
                "organizer_objective": False,
            },
        })
        return payload

    def source_training_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        del state
        raise ValueError(f"R13-5{self.candidate} is scratch-only; source weights and training state are not promoted.")

    def _paired_predictions(self, prepared, physical_heat):
        """Share the live tile plan; retain separate native reads and losses."""
        model = self.model
        model.validate_prepared(prepared)
        if physical_heat is not None:
            expected = prepared.structure["heat_powers"]
            given = torch.as_tensor(physical_heat, device=expected.device)
            if given.shape != expected.shape or not torch.equal(given.to(expected), expected):
                raise ValueError("Heat changed: rebuild the complete joint prepared context.")
        full, restricted, shared_counts = model.core.decode_paired(
            prepared.organization, prepared.query_batch,
        )
        work = {
            "native_fluid_queries": int(prepared.fluid_xy.shape[0] * prepared.fluid_xy.shape[1]),
            "native_union_rows": int(prepared.union.unique_counts.sum()),
            "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
        }
        work.update({f"core_prepare_primary_{name}": int(value)
                     for name, value in prepared.organization.work_counts.items()})
        work.update({f"core_decode_shared_{name}": int(value)
                     for name, value in shared_counts.items()})
        predictions = []
        for mode, output in (("full", full), ("restricted", restricted)):
            fields = output.joint_fields
            native = model._extract_native_roles(prepared, fields[..., 3:4], fields[..., :3])
            work.update({f"core_decode_joint_{mode}_{name}": int(value)
                         for name, value in output.work_counts.items()})
            predictions.append(ThermalContentPredictions(prepared, native, work=output.work_counts))
        return ThermalJointPairPredictions(predictions[0], predictions[1], work)


def build_thermal_supplement_member_routing_task(config, *, arm=None, device="cpu"):
    """Build A or B with the same Thermal data, normalization and native targets."""
    recipe = copy.deepcopy(dict(config))
    section = recipe.get("member_routing", {})
    model_config = section.get("model")
    selected_arm = str(arm or section.get("arm", ""))
    if section.get("schema") != CONFIG_SECTION_SCHEMA:
        raise ValueError(f"R13-5 supplement requires {CONFIG_SECTION_SCHEMA}.")
    if not isinstance(model_config, Mapping):
        raise ValueError("R13-5 supplement requires a saved A or B model configuration.")
    variant = _SUPPLEMENT_VARIANTS.get(str(model_config.get("schema")))
    if (variant is None or selected_arm != variant["arm"]
            or model_config.get("architecture") != selected_arm):
        raise ValueError("R13-5 supplement arm, core schema and architecture must select the same A or B variant.")
    adapter_type = variant["adapter"]
    if (model_config.get("address_mode") != "ME"
            or model_config.get("joint_member_coverage") != 0.95):
        raise ValueError("R13-5 supplement requires the four-block ME view and joint95 member policy.")
    if model_config.get("retrieval_calibration") not in ("normalized", "row_strength"):
        raise ValueError("R13-5 retrieval calibration must be selected from the maintained equations.")
    if model_config.get("score_calibration") not in ("inherited", "bounded_tanh"):
        raise ValueError("R13-5 score calibration must be selected from the maintained equations.")
    if variant["candidate"] == "B" and (
            model_config.get("retrieval_calibration") != "normalized"
            or model_config.get("score_calibration") != "inherited"):
        raise ValueError("R13-5B must use the selected A normalized/inherited conventions.")
    manifest, data_path, manifest_path = _load_data_binding(
        recipe["dataset"].get("manifest_path"), recipe["dataset"].get("data_path"),
    )
    catalog_source, catalog = read_case_catalog(data_path)
    train_ids, validation_ids = _training_membership(recipe, catalog, manifest)
    normalizer = fit_global_normalizer(data_path, train_ids)
    training_cases = _read_selected_cases(
        data_path, train_ids, split="train", normalizer=normalizer, include_grid=False,
    )
    validation_cases = _read_selected_cases(
        data_path, validation_ids, split="test", normalizer=normalizer, include_grid=False,
    )
    heat_values = np.concatenate([
        np.asarray(case["structure"]["heat_powers"])[
            np.asarray(case["structure"]["module_present"]) > 0.5
        ] for case in training_cases
    ])
    heat_scale = max(float(np.sqrt(np.mean(heat_values.astype(np.float64) ** 2))), 1e-6)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(recipe["initialization"].get("seed", 0)))
        model = adapter_type(
            normalizer.stats, core_config=model_config, heat_scale=heat_scale,
        ).to(device)
    source_stat = data_path.stat()
    provider = ThermalSupplementMemberRoutingTask(
        model=model, training_cases=training_cases, validation_cases=validation_cases,
        manifest=manifest, data_path=data_path, recipe=recipe, device=device,
        manifest_path=manifest_path, catalog=catalog,
        source_binding={
            "dataset_path": str(data_path), "dataset_size_bytes": int(source_stat.st_size),
            "dataset_mtime_ns": int(source_stat.st_mtime_ns),
            "catalog_metadata_sha256": catalog_source["metadata_sha256"],
            "manifest_path": None if manifest_path is None else str(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "normalization_source": "TRAIN-only fit_global_normalizer",
            "dataset_copy_created": False, "persistent_cache_created": False,
        },
    )
    return model, provider


__all__ = [
    "ARM", "B_ARM", "WORKFLOW", "B_WORKFLOW", "PROVIDER_SCHEMA", "CONFIG_SECTION_SCHEMA",
    "TRAINING_STATE_SCHEMA", "CONTRIBUTION_RULE", "B_CONTRIBUTION_RULE",
    "SupplementMemberRoutingHealth", "ThermalSupplementMemberRoutingTask",
    "build_thermal_supplement_member_routing_task",
]
