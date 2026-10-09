"""Fixed-cohort Thermal provider for fresh joint regional field learning."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from honf_runtime.unified_training import LossTerm, OptimizerGroupSpec, SamplingKey, ScheduleSpec, TaskBatch

from ..data.datasets import GlobalChannelThermalDataset, H5Normalizer, fit_global_normalizer
from ..data.development_split import (
    development_case_ids,
    read_case_catalog,
    validate_development_manifest,
)
from ..joint_regional import (
    JOINT_THERMAL_CHANNEL_ORDER,
    JOINT_THERMAL_ID,
    JOINT_THERMAL_MODES,
    JointThermalRegionalAdapter,
)
from .unified_task import (
    DEFAULT_ATLAS_DIRECTORY,
    FIXED25_FINGERPRINT,
    OPERATOR_ROWS_PER_CASE,
    Q_PROXY_COEFFICIENT,
    RESPONSE_SURFACE_STRIDE,
    TRAIN_RESPONSE_IDS,
    ThermalPredictions,
    ThermalRefinementTask,
    _operator_row_indices,
    _operator_stencil_receivers,
    _read_response_families,
    _residual_from_stencil_kernel,
    _tensor_digest,
)

JOINT_TRAINING_MODE = "joint"
JOINT_MICROBATCH_DEFAULT = 4
JOINT_EFFECTIVE_BATCH_DEFAULT = 48
JOINT_PRIMARY_QUERY_DEFAULT = 1024
JOINT_MANIFEST_DEFAULT = Path("/data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json")
JOINT_SOURCE_ROOT = Path(__file__).resolve().parents[4]
JOINT_TOOLS = JOINT_SOURCE_ROOT / "tools"
JOINT_NEAR_RADIUS_MULTIPLE = 2.0


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _case_ids_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8") + b"\0")
    return digest.hexdigest()


def _load_data_binding(manifest: Mapping[str, Any] | str | Path | None, data_path: str | Path | None):
    if manifest is None:
        manifest_path = JOINT_MANIFEST_DEFAULT
        if not manifest_path.is_file():
            raise FileNotFoundError(f"The sealed fixed25_v1 manifest is unavailable: {manifest_path}")
        loaded = json.loads(manifest_path.read_text())
    elif isinstance(manifest, Mapping):
        manifest_path = None
        loaded = copy.deepcopy(dict(manifest))
    else:
        manifest_path = Path(manifest).expanduser().resolve()
        loaded = json.loads(manifest_path.read_text())

    bound_data_path = Path(data_path).expanduser().resolve() if data_path is not None else Path(
        loaded.get("source", {}).get("dataset_path", "")
    ).expanduser().resolve()
    if not str(bound_data_path) or not bound_data_path.is_file():
        raise FileNotFoundError(f"The fixed Thermal packed dataset is unavailable: {bound_data_path}")
    verified = validate_development_manifest(
        loaded,
        bound_data_path,
        expected_fingerprint=FIXED25_FINGERPRINT,
    )
    return verified, bound_data_path, manifest_path


def _read_selected_cases(dataset_path: Path, case_ids: Sequence[str], *, split: str, normalizer: H5Normalizer,
                         include_grid: bool) -> list[dict[str, Any]]:
    dataset = GlobalChannelThermalDataset(
        dataset_path,
        split=split,
        points_per_case=None,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        seed=0,
        include_grid=include_grid,
        normalizer=normalizer,
        case_ids=case_ids,
    )
    if dataset.selected_case_ids != list(case_ids):
        raise ValueError("Joint Thermal loader changed the fixed source-case ordering.")
    cases = [dataset[index] for index in range(len(dataset))]
    for case in cases:
        present = np.asarray(case["structure"]["module_present"]) > 0.5
        if not np.array_equal(np.flatnonzero(present), np.arange(int(present.sum()))):
            raise ValueError("Joint Thermal source columns require contiguous original module slots.")
        case["structure"]["module_source_ids"] = np.arange(len(present), dtype=np.int64)
    return cases


def _load_recipe_helpers():
    if str(JOINT_TOOLS) not in sys.path:
        sys.path.insert(0, str(JOINT_TOOLS))
    from thermal_source_response_fit import build_balances, response_scales

    return build_balances, response_scales


def _gradient_norm(loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter], *, retain_graph=False) -> float:
    gradients = torch.autograd.grad(loss, tuple(parameters), allow_unused=True, retain_graph=retain_graph)
    squared = sum(float(gradient.detach().double().square().sum().cpu()) for gradient in gradients if gradient is not None)
    value = math.sqrt(squared)
    if not math.isfinite(value):
        raise FloatingPointError("Joint Thermal auxiliary calibration produced a nonfinite gradient norm.")
    return value


class JointThermalTask(ThermalRefinementTask):
    """Thermal native training records with a genuinely joint flow/T objective.

    The base class is used only for its fixed25 primary/response sampling and
    native batch records. Every refinement execution hook and loss path is
    replaced here; no parent state or frozen flow partner is accepted.
    """

    def __init__(
        self,
        *,
        model: JointThermalRegionalAdapter,
        training_cases: Sequence[Mapping[str, Any]],
        validation_cases: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
        data_path: Path,
        source_binding: Mapping[str, Any],
        train_families: Sequence[Mapping[str, Any]],
        balances: Sequence[Any],
        normalization_stats: Mapping[str, Any],
        mode: str,
        device: torch.device | str,
        budget: Mapping[str, Any],
        effective_batch_size: int,
        total_epochs: int,
        formal_full: bool,
        response_coefficient: float,
        operator_coefficient: float,
        operator_rows_per_case: int = OPERATOR_ROWS_PER_CASE,
        atlas_directory: Path = DEFAULT_ATLAS_DIRECTORY,
        auxiliary_calibration: Mapping[str, Any] | None = None,
        validation_scope: str | None = None,
        optimizer_schedule: ScheduleSpec | None = None,
        weight_decay: float | None = None,
        native_sampling_protocol: str | None = None,
    ) -> None:
        if mode not in JOINT_THERMAL_MODES or model.mode != mode:
            raise ValueError("Joint Thermal task mode must match its freshly initialized adapter.")
        if operator_rows_per_case != OPERATOR_ROWS_PER_CASE:
            raise ValueError("The sealed Thermal joint operator uses exactly128 rows per TRAIN case.")
        for name, value in (("response", response_coefficient), ("operator", operator_coefficient)):
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"Joint Thermal {name} coefficient must be finite and lie in [0,1].")
        if total_epochs < 2:
            raise ValueError("Joint Thermal schedule horizon must be at least two epochs.")

        self.joint_mode = mode
        self.data_path = Path(data_path).resolve()
        self.source_binding = copy.deepcopy(dict(source_binding))
        self.formal_full = bool(formal_full)
        self.formal_validation_scope = validation_scope
        self.formal_optimizer_schedule = optimizer_schedule
        self.formal_weight_decay = weight_decay
        self.native_sampling_protocol = native_sampling_protocol
        self.effective_batch_size = int(effective_batch_size)
        self.total_epochs = int(total_epochs)
        self.operator_rows_per_case = int(operator_rows_per_case)
        self.auxiliary_calibration = None if auxiliary_calibration is None else copy.deepcopy(dict(auxiliary_calibration))
        self.flow_role_weights = {name: 0.5 / 4 for name in ("u", "v", "p", "omega")}
        self.thermal_role_weights = {name: 0.5 / 3 for name in ("fluid", "surface", "material")}
        self.response_weight = float(response_coefficient)
        self.operator_weight = float(operator_coefficient)
        self.q_proxy_weight = float(Q_PROXY_COEFFICIENT)
        self.use_operator = True

        stats = copy.deepcopy(dict(normalization_stats))
        if "interface_targets_std" not in stats and "interface_target_std" in stats:
            stats["interface_targets_std"] = np.asarray(stats["interface_target_std"], dtype=np.float32).copy()
        if "interface_targets_mean" not in stats and "interface_target_mean" in stats:
            stats["interface_targets_mean"] = np.asarray(stats["interface_target_mean"], dtype=np.float32).copy()
        # The legacy task initializer remains a narrow reusable owner of the
        # native role samplers/record types; all of its learned-parent and
        # refinement behavior is bypassed by overrides below.
        recipe = {
            "budget": dict(budget),
            "calibration": {
                "response_scales": _load_recipe_helpers()[1](train_families),
                "response_coefficient": self.response_weight,
                "operator_coefficient": self.operator_weight,
            },
            "operator_decision": {
                "operator_constraint": "qualified",
                "reason": "new joint Thermal family retains the established TRAIN-only native discrete operator",
            },
            "weight_decay": 1.0e-4,
            "manifest_sha256": manifest["manifest_sha256"],
        }
        super().__init__(
            model=model,
            parent_model=None,
            parent=None,
            parent_path=None,
            flow_path=None,
            atlas_directory=atlas_directory,
            training_cases=training_cases,
            validation_cases=validation_cases,
            manifest=manifest,
            train_families=train_families,
            development_families=(),
            balances=balances,
            optimizer_seed=None,
            device=device,
            stats=stats,
            recipe=recipe,
        )
        self.response_scales = recipe["calibration"]["response_scales"]
        self.response_coefficient = self.response_weight
        self.operator_coefficient = self.operator_weight
        self._geometry_helper = model
        self._joint_loss_metadata = self._build_loss_metadata()

    def identity_payload(self) -> Mapping[str, Any]:
        train_ids = list(self._train_ids)
        validation_ids = list(self._validation_ids)
        training_hash = _case_ids_hash(train_ids)
        fixed25_train_hash = _case_ids_hash(development_case_ids(self.manifest, "train"))
        if self.formal_full and training_hash == fixed25_train_hash:
            raise ValueError("Manual fullTRAIN identity cannot reuse fixed25_v1 quarter TRAIN membership.")
        normalization_stats = {
            str(name): np.asarray(value).tolist() for name, value in self.stats.items()
        }
        payload = {
            "task": "ThermalChannel",
            "task_provider": "thermal_joint_regional_v1",
            "training_mode": JOINT_TRAINING_MODE,
            "arm": self.joint_mode,
            "dataset_split": "original_train_full_manual_only" if self.formal_full else "fixed25_v1",
            "manifest_fingerprint": self.manifest["manifest_sha256"],
            "manifest_binding_scope": "fixed25_v1_validation_sanity_panel_only" if self.formal_full else "fixed25_v1_training_and_validation",
            "training_membership_scope": "all_original_train_600_manual_only" if self.formal_full else "fixed25_v1_selected_train_150",
            "training_membership_sha256": training_hash,
            "fixed25_v1_train_membership_sha256": fixed25_train_hash,
            "training_case_ids": train_ids,
            "training_case_ids_sha256": training_hash,
            "training_case_count": len(train_ids),
            "validation_case_ids": validation_ids,
            "validation_case_ids_sha256": _case_ids_hash(validation_ids),
            "validation_case_count": len(validation_ids),
            "validation_scope": "fixed25_v1_exposed_DEV22_sanity_only" if self.formal_full else "fixed25_v1_exposed_DEV22",
            "source_binding": copy.deepcopy(self.source_binding),
            "model_family": JOINT_THERMAL_ID,
            "model_config": self.model.model_config(),
            "adapter_config": self.model.adapter_config(),
            "normalization_scope": "all_original_train_only" if self.formal_full else "selected_fixed25_train_only",
            "normalization_stats_sha256": _tensor_digest(self.stats),
            "normalization_stats": normalization_stats,
            "native_query_budget": dict(self.budget),
            "native_metric_supports": {
                "fluid_temperature_near": f"valid fixed primary receivers within {JOINT_NEAR_RADIUS_MULTIPLE:g} local physical source radii of the nearest active module",
                "fluid_temperature_far": f"valid fixed primary receivers beyond {JOINT_NEAR_RADIUS_MULTIPLE:g} local physical source radii of every active module",
                "material_peak": "maximum over finite, in-domain sampled native material receivers for each active module",
            },
            "effective_batch_size": self.effective_batch_size,
            "operator_rows_per_case": self.operator_rows_per_case,
            "response_family_ids": [str(item["family_id"]) for item in self.train_families],
            "response_scales": dict(self.response_scales),
            "loss_weights": {
                "flow_group": {"group_weight": 0.5, "per_role": dict(self.flow_role_weights)},
                "thermal_group": {"group_weight": 0.5, "per_role": dict(self.thermal_role_weights)},
                "q_proxy": {"coefficient": self.q_proxy_weight, "inside_balanced_thermal_group": False},
                "response": {"coefficient": self.response_weight},
                "operator_residual": {"coefficient": self.operator_weight},
            },
            "auxiliary_calibration": copy.deepcopy(self.auxiliary_calibration),
            "schedule": {
                "peak_lr": 3.0e-4,
                "warmup_start_lr": 3.0e-5,
                "warmup_epochs": 20,
                "hold_through_epoch": min(1000, self.total_epochs - 1),
                "total_epochs": self.total_epochs,
                "final_lr": 3.0e-6,
                "optimizer": "AdamW",
                "gradient_clip_norm": 1.0,
            },
            "external_learned_model_dependencies": [],
            "formal_full_run_policy": "manual_only_not_launched" if self.formal_full else None,
            "solver_attempts": 0,
        }
        if getattr(self, "formal_validation_scope", None) == "canonical89":
            payload["manifest_binding_scope"] = "fixed25_v1_source_metadata_only"
            payload["validation_scope"] = "formal_canonical89_exposed_validation"
        schedule = getattr(self, "formal_optimizer_schedule", None)
        weight_decay = getattr(self, "formal_weight_decay", None)
        if schedule is not None:
            payload["schedule"].update(asdict(schedule))
        if schedule is not None or weight_decay is not None:
            payload["schedule"].update({"weight_decay": 1.0e-4 if weight_decay is None else weight_decay,
                                        "betas": [0.9, 0.999], "eps": 1.0e-8})
        if getattr(self, "native_sampling_protocol", None) == "baseline_formal_v1":
            payload["native_sampling_identity"] = {
                "protocol": "baseline_formal_v1", "dataset": "ThermalChannel",
                "train_membership_fingerprint": _json_hash(train_ids),
                "primary_stream": "primary_native_queries",
                "response_stream": "response_addendum",
                "validation_sampling_indices": dict(self.validation_sampling_indices),
                "validation_seed_rule": "1000 + original90_index * 104729",
                "operator_seed_rule": "originalTRAIN_index * 104729 + epoch * 1000003 + 17",
            }
        return payload

    def _build_loss_metadata(self) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for role in ("u", "v", "p", "omega"):
            result[f"flow/{role}"] = {
                "panel": "native_prediction",
                "label": f"Flow {role}",
                "unit": "dimensionless scaled squared error",
                "formula": f"Per-case weighted native query MSE of {role}, divided by its selected-TRAIN standard deviation squared.",
                "weight": "0.5/4 equal flow-family share",
            }
        for role in ("fluid", "surface", "material"):
            result[f"temperature/{role}"] = {
                "panel": "native_prediction",
                "label": f"Temperature {role}",
                "unit": "dimensionless scaled squared error",
                "formula": f"Per-case native {role} MSE, divided by its selected-TRAIN standard deviation squared and its native valid-role denominator.",
                "weight": "0.5/3 equal thermal-family share",
            }
        result["q_proxy"] = {
            "panel": "native_prediction",
            "label": "Native q proxy",
            "unit": "dimensionless scaled squared error",
            "formula": "Per-case native surface-flux proxy MSE divided by its selected-TRAIN standard deviation squared.",
            "weight": "0.05 outside the equal flow/thermal group balance",
        }
        result["response"] = {
            "panel": "physics",
            "label": "TRAIN heat response",
            "unit": "dimensionless scaled squared error",
            "formula": "Equal mean of fluid, surface and material temperature-increment MSE on the existing original-TRAIN response addendum.",
            "weight": f"TRAIN-only calibrated coefficient {self.response_weight:.9g}",
        }
        result["operator_residual"] = {
            "panel": "physics",
            "label": "Native discrete thermal operator",
            "unit": "dimensionless row-normalized squared residual",
            "formula": "Per-case mean A K - B residual over 128 fixed-stratum native rows using stored TRAIN velocity only in the supervision-side operator coefficient.",
            "weight": f"TRAIN-only calibrated coefficient {self.operator_weight:.9g}",
        }
        return result

    def loss_metadata(self) -> Mapping[str, Mapping[str, str]]:
        return copy.deepcopy(self._joint_loss_metadata)

    def _prepare_response_sample(
        self, family: Mapping[str, Any], *, training: bool, key: SamplingKey | None,
        budget: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Keep physical source IDs integral through the legacy native sampler."""
        sample = super()._prepare_response_sample(family, training=training, key=key, budget=budget)
        source_ids = family["structure"].get("module_source_ids")
        if source_ids is not None:
            source_ids = np.asarray(source_ids)
            if source_ids.ndim != 1 or not np.issubdtype(source_ids.dtype, np.integer):
                raise ValueError("TRAIN response addenda require ordered integer physical source identities.")
            sample["structure"]["module_source_ids"] = torch.as_tensor(
                source_ids.copy(), dtype=torch.long
            )[None]
        return sample

    def phase_reporting(self, *, arm: str, epoch: int, phase: str) -> Mapping[str, Any]:
        if arm != self.joint_mode:
            raise ValueError("Joint Thermal provider was invoked under another experiment arm.")
        return {
            "training_mode": JOINT_TRAINING_MODE,
            "phase": str(phase),
            "epoch": int(epoch),
            "all_joint_parameters_active": True,
            "flow_and_temperature_supervised_together": True,
        }

    def loss_denominators(self, batches: Sequence[TaskBatch], phase: str, arm: str) -> Mapping[str, float]:
        del phase
        if arm != self.joint_mode:
            raise ValueError("Joint Thermal loss denominator request used the wrong arm label.")
        totals: dict[str, float] = {}
        for batch in batches:
            count = float(len(batch.targets.case_ids))
            if count <= 0:
                raise ValueError("Joint Thermal macrobatch cannot contain an empty microbatch.")
            for role in ("u", "v", "p", "omega"):
                name = f"flow/{role}"
                totals[name] = totals.get(name, 0.0) + count
            for role in ("fluid", "surface", "material"):
                name = f"temperature/{role}"
                totals[name] = totals.get(name, 0.0) + count
            totals["q_proxy"] = totals.get("q_proxy", 0.0) + count
            if self.use_operator and batch.targets.case_indices:
                totals["operator_residual"] = totals.get("operator_residual", 0.0) + count
            if batch.targets.response is not None:
                totals["response"] = totals.get("response", 0.0) + 1.0
        return totals

    def predict_native(
        self,
        model,
        scene,
        receivers,
        execution_mode,
        phase,
        epoch=0,
        temperature=1.0,
        *,
        threshold=0.5,
    ):
        del phase, epoch, temperature, threshold
        if model is not self.model or execution_mode != self.joint_mode:
            raise ValueError("Joint Thermal prediction requires its own candidate and J-mode arm label.")
        from honf_runtime.compat import recursive_to_device

        rx = recursive_to_device(receivers, self.device)
        main_prepared = model.prepare_native(
            scene.structure,
            rx.fluid_xy,
            local_query_points=rx.local_query_points,
            ntheta=16,
            chunk_size=model.receiver_tile,
        )
        main = model.apply_native(main_prepared, rx.heat)
        response_prepared = response_main = None
        if rx.response_fluid_xy is not None:
            if scene.response_structure is None or rx.response_local_query_points is None or rx.response_heat is None:
                raise ValueError("Joint Thermal response addendum is missing native input receivers or heat increment.")
            response_prepared = model.prepare_native(
                scene.response_structure,
                rx.response_fluid_xy,
                local_query_points=rx.response_local_query_points,
                ntheta=16,
                chunk_size=model.receiver_tile,
            )
            response_main = model.apply_native(response_prepared, rx.response_heat, increment=True)
        work = {
            "native_fluid_queries": int(rx.fluid_xy.shape[0] * rx.fluid_xy.shape[1]),
            "native_operator_rows": 0,
        }
        work["native_valid_fluid_queries"] = int(main_prepared.stencils["fluid"].valid.sum().item())
        work["native_surface_queries"] = int(main_prepared.stencils["surface"].valid.numel())
        work["native_material_queries"] = int(main_prepared.stencils["material"].valid.numel())
        predictions = ThermalPredictions(
            model=model,
            execution_mode=execution_mode,
            phase="joint",
            native_prepared=main_prepared,
            native_main=main,
            native_full=None,
            response_prepared=response_prepared,
            response_main=response_main,
            response_full=None,
            auxiliary={},
            work=work,
        )
        return predictions, {"work": work}

    def _native_role_losses(self, output: Mapping[str, torch.Tensor], prepared, targets):
        stats = self.stats
        field_std = output["pred_field"].new_tensor(np.asarray(stats["field_std_by_channel"], dtype=np.float32))
        query_weight = targets.point_weights.to(output["pred_field"])
        query_weight = query_weight * prepared.stencils["fluid"].valid.to(query_weight)
        predicted = output["pred_field"]
        target_fields = targets.field_targets.to(predicted)
        flow_finite = torch.isfinite(target_fields[..., :4])
        flow_errors = (predicted[..., :4] - torch.nan_to_num(target_fields[..., :4])) / field_std[:4]
        flow_weights = query_weight[..., None] * flow_finite
        flow_per_channel = (flow_errors.square() * flow_weights).sum(1) / flow_weights.sum(1).clamp_min(1.0e-12)
        result: dict[str, torch.Tensor] = {
            f"flow/{role}": flow_per_channel[:, index]
            for index, role in enumerate(("u", "v", "p", "omega"))
        }

        fluid_weight = query_weight * torch.isfinite(target_fields[..., 4])
        fluid_error = (predicted[..., 4] - torch.nan_to_num(target_fields[..., 4])) / field_std[4]
        result["temperature/fluid"] = (
            (fluid_error.square() * fluid_weight).sum(1) / fluid_weight.sum(1).clamp_min(1.0e-12)
        )

        present = prepared.source_present.to(predicted)
        surface_shape = output["pred_interface"].shape[:-1]
        surface_valid = prepared.stencils["surface"].valid.reshape(surface_shape)
        outside_valid = prepared.stencils["outside"].valid.reshape(surface_shape)
        surface_target = targets.interface_target[..., 0].to(predicted)
        surface_mask = present[..., None] * surface_valid * torch.isfinite(surface_target)
        surface_scale = max(float(np.asarray(stats["interface_targets_std"]).reshape(-1)[0]), 1.0e-6)
        surface_error = (output["pred_interface"][..., 0] - torch.nan_to_num(surface_target)) / surface_scale
        surface_per_module = (surface_error.square() * surface_mask).sum(-1) / surface_mask.sum(-1).clamp_min(1)
        active_surface_module = present * (surface_mask.sum(-1) > 0)
        result["temperature/surface"] = (
            surface_per_module * active_surface_module
        ).sum(-1) / active_surface_module.sum(-1).clamp_min(1)

        material_target = targets.material_targets.to(predicted)
        material_error = (
            output["pred_internal_temperature"][..., 0] - torch.nan_to_num(material_target)
        ) / max(float(np.asarray(stats["internal_temperature_std"]).reshape(-1)[0]), 1.0e-6)
        material_valid = (
            present[..., None]
            * prepared.stencils["material"].valid.reshape(material_target.shape).to(present.dtype)
            * torch.isfinite(material_target)
        )
        result["temperature/material"] = (
            material_error.square() * material_valid
        ).sum((1, 2)) / material_valid.sum((1, 2)).clamp_min(1)

        q_target = targets.interface_target[..., 1].to(predicted)
        q_scale = max(float(np.asarray(stats["interface_targets_std"]).reshape(-1)[1]), 1.0e-6)
        q_mask = present[..., None] * surface_valid * outside_valid * torch.isfinite(q_target)
        q_error = (output["pred_interface"][..., 1] - torch.nan_to_num(q_target)) / q_scale
        q_per_module = (q_error.square() * q_mask).sum(-1) / q_mask.sum(-1).clamp_min(1)
        q_modules = present * (q_mask.sum(-1) > 0)
        result["q_proxy"] = (q_per_module * q_modules).sum(-1) / q_modules.sum(-1).clamp_min(1)
        result["flow_group"] = torch.stack([result[f"flow/{role}"] for role in ("u", "v", "p", "omega")]).mean(0)
        result["thermal_group"] = torch.stack(
            [result[f"temperature/{role}"] for role in ("fluid", "surface", "material")]
        ).mean(0)
        return result

    def _operator_loss(self, predictions: ThermalPredictions, targets) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if not targets.case_indices:
            return None, {}
        from channelthermal.source_response_residual import DiscreteThermalBalance

        prepared = predictions.native_prepared
        device = prepared.context.centers.device
        coefficients = torch.cat([self.balances[index].coefficients for index in targets.case_indices]).to(device)
        slots = torch.cat([self.balances[index].source_slots for index in targets.case_indices]).to(device)
        balance = DiscreteThermalBalance(coefficients, slots, prepared.source_present)
        rows = _operator_row_indices(self.balances, targets.case_indices, targets.epoch, device)
        query, receiver_ids, gather, stencil, ix, iy = _operator_stencil_receivers(
            balance, rows, prepared.context.lengths, self.model.nx, self.model.ny
        )
        prepared_read = self.model.core.prepare_receivers(
            prepared.context,
            query,
            receiver_ids=receiver_ids,
            chunk_size=self.model.receiver_tile,
        )
        kernel = prepared_read.dense_kernel()[..., 0]
        residual = _residual_from_stencil_kernel(kernel, balance, rows, gather, stencil, ix, iy)
        temperature_scale = max(float(np.asarray(self.stats["field_std_by_channel"]).reshape(-1)[4]), 1.0e-6)
        residual = residual / temperature_scale
        denominator = (prepared.source_present.sum(-1) * self.operator_rows_per_case).clamp_min(1)
        per_case = (residual.square() * prepared.source_present[:, None]).sum((1, 2)) / denominator
        receipt = {
            "operator_rows": int(rows.numel()),
            "rows_per_case": self.operator_rows_per_case,
            "kernel_source_columns": int(balance.present.sum().item()),
            "unique_stencil_receiver_rows": int(sum(torch.unique(stencil[index]).numel() for index in range(stencil.shape[0]))),
            "physical_solves": 0,
            "uses_stored_velocity_only_for_supervision_coefficients": True,
        }
        return per_case, receipt

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        del phase
        if predictions.model is not self.model or predictions.execution_mode != self.joint_mode:
            raise ValueError("Joint Thermal objective received predictions from another mode/model.")
        values = self._native_role_losses(predictions.native_main, predictions.native_prepared, targets)
        terms = {
            f"flow/{role}": LossTerm(values[f"flow/{role}"].sum(), len(targets.case_ids), self.flow_role_weights[role])
            for role in ("u", "v", "p", "omega")
        }
        terms.update({
            f"temperature/{role}": LossTerm(
                values[f"temperature/{role}"].sum(), len(targets.case_ids), self.thermal_role_weights[role]
            )
            for role in ("fluid", "surface", "material")
        })
        terms["q_proxy"] = LossTerm(
            values["q_proxy"].sum(), len(targets.case_ids), self.q_proxy_weight
        )
        operator_values, operator_receipt = self._operator_loss(predictions, targets)
        if operator_values is not None:
            terms["operator_residual"] = LossTerm(
                operator_values.sum(), len(targets.case_ids), self.operator_weight
            )
            auxiliary_state["operator"] = operator_receipt
            predictions.work.update({
                "operator_rows": operator_receipt["operator_rows"],
                "operator_unique_stencil_receiver_rows": operator_receipt["unique_stencil_receiver_rows"],
            })
        if targets.response is not None:
            if predictions.response_main is None or predictions.response_prepared is None:
                raise RuntimeError("Joint Thermal TRAIN response addendum lost its native affine output.")
            response = self._response_loss(predictions.response_main, targets.response)
            terms["response"] = LossTerm(response, 1.0, self.response_weight)
        auxiliary_state["work"] = dict(predictions.work)
        return terms

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch, arm):
        del auxiliary_state, batch
        if arm != self.joint_mode:
            raise ValueError("Joint Thermal validation objective received a mismatched arm.")
        values = self._native_role_losses(predictions.native_main, predictions.native_prepared, targets)
        terms = {
            f"flow/{role}": LossTerm(values[f"flow/{role}"].sum(), len(targets.case_ids), self.flow_role_weights[role])
            for role in ("u", "v", "p", "omega")
        }
        terms.update({
            f"temperature/{role}": LossTerm(
                values[f"temperature/{role}"].sum(), len(targets.case_ids), self.thermal_role_weights[role]
            )
            for role in ("fluid", "surface", "material")
        })
        terms["q_proxy"] = LossTerm(values["q_proxy"].sum(), len(targets.case_ids), self.q_proxy_weight)
        return terms

    def validation_metrics(self, predictions, targets, auxiliary_state):
        del auxiliary_state
        values = self._native_role_losses(predictions.native_main, predictions.native_prepared, targets)
        pred = predictions.native_main["pred_field"]
        target = targets.field_targets.to(pred)
        weights = targets.point_weights.to(pred) * predictions.native_prepared.stencils["fluid"].valid.to(pred)
        rows = []
        for index, case_id in enumerate(targets.case_ids):
            prepared = predictions.native_prepared
            present = prepared.source_present[index] > 0.5
            flow_rmse = {}
            for channel, role in enumerate(("u", "v", "p", "omega")):
                role_weight = weights[index] * torch.isfinite(target[index, :, channel])
                error = pred[index, :, channel] - torch.nan_to_num(target[index, :, channel])
                mse = (error.square() * role_weight).sum() / role_weight.sum().clamp_min(1.0e-12)
                flow_rmse[role] = float(torch.sqrt(mse.clamp_min(0)).detach())
            fluid_weight = weights[index] * torch.isfinite(target[index, :, 4])
            fluid_error = pred[index, :, 4] - torch.nan_to_num(target[index, :, 4])
            fluid_mse = (fluid_error.square() * fluid_weight).sum() / fluid_weight.sum().clamp_min(1.0e-12)
            fluid_rmse = torch.sqrt(fluid_mse.clamp_min(0))
            fluid_receiver_valid = prepared.stencils["fluid"].valid[index]
            center_delta = prepared.flow_receivers[index, :, None, :] - prepared.context.centers[index, None, :, :]
            normalized_distance = torch.linalg.vector_norm(center_delta, dim=-1) / prepared.context.source_lengths[
                index, None, :
            ].clamp_min(1.0e-12)
            normalized_distance = normalized_distance.masked_fill(~present[None], float("inf"))
            nearest_source_distance = normalized_distance.amin(-1)
            finite_fluid = torch.isfinite(target[index, :, 4]) & torch.isfinite(pred[index, :, 4])
            support_valid = (weights[index] > 0) & fluid_receiver_valid & finite_fluid
            near_mask = support_valid & (nearest_source_distance <= JOINT_NEAR_RADIUS_MULTIPLE)
            far_mask = support_valid & (nearest_source_distance > JOINT_NEAR_RADIUS_MULTIPLE)

            def support_rmse(mask, error):
                count = int(mask.sum().item())
                if count == 0:
                    return None, 0, 0.0
                squared_sum = float(error[mask].double().square().sum().item())
                return math.sqrt(squared_sum / count), count, squared_sum

            near_rmse, near_count, near_squared_sum = support_rmse(near_mask, fluid_error)
            far_rmse, far_count, far_squared_sum = support_rmse(far_mask, fluid_error)
            interface_target = targets.interface_target[index]
            surface_valid = predictions.native_prepared.stencils["surface"].valid[index].reshape(
                predictions.native_main["pred_interface"].shape[1:-1]
            )
            outside_valid = predictions.native_prepared.stencils["outside"].valid[index].reshape_as(surface_valid)
            surface_mask = present[:, None] & surface_valid & torch.isfinite(interface_target[..., 0])
            surface_error = predictions.native_main["pred_interface"][index, ..., 0] - torch.nan_to_num(interface_target[..., 0])
            surface_mse = (surface_error.square() * surface_mask).sum() / surface_mask.sum().clamp_min(1)
            material_target = targets.material_targets[index]
            material_valid = prepared.stencils["material"].valid.reshape(
                predictions.native_main["pred_internal_temperature"].shape[:-1]
            )[index]
            material_mask = present[:, None] & material_valid & torch.isfinite(material_target)
            material_error = predictions.native_main["pred_internal_temperature"][index, ..., 0] - torch.nan_to_num(material_target)
            material_mse = (material_error.square() * material_mask).sum() / material_mask.sum().clamp_min(1)
            peak_valid = material_mask & torch.isfinite(
                predictions.native_main["pred_internal_temperature"][index, ..., 0]
            )
            has_peak = peak_valid.any(-1) & present
            predicted_peak = predictions.native_main["pred_internal_temperature"][index, ..., 0].masked_fill(
                ~peak_valid, -torch.inf
            ).amax(-1)
            target_peak = material_target.masked_fill(~peak_valid, -torch.inf).amax(-1)
            peak_error = predicted_peak[has_peak] - target_peak[has_peak]
            peak_rmse = torch.sqrt(peak_error.square().mean().clamp_min(0)) if peak_error.numel() else None
            source_ids = prepared.context.source_ids[index]
            peak_rows = [
                {
                    "source_id": int(source_ids[module_index].item()),
                    "absolute_error": float(value.abs().detach()),
                }
                for module_index, value in zip(torch.nonzero(has_peak, as_tuple=True)[0].tolist(), peak_error)
            ]
            q_target = interface_target[..., 1]
            q_mask = present[:, None] & surface_valid & outside_valid & torch.isfinite(q_target)
            q_error = predictions.native_main["pred_interface"][index, ..., 1] - torch.nan_to_num(q_target)
            q_mse = (q_error.square() * q_mask).sum() / q_mask.sum().clamp_min(1)
            flow_group = values["flow_group"][index]
            thermal_group = values["thermal_group"][index]
            row = {
                "case_id": str(case_id),
                "module_count": int(present.sum().item()),
                "flow_group_standardized_mse": float(flow_group.detach()),
                "thermal_group_standardized_mse": float(thermal_group.detach()),
                "field_score": float((0.5 * flow_group + 0.5 * thermal_group).detach()),
                "fluid_temperature_rmse": float(fluid_rmse.detach()),
                "near_fluid_temperature_rmse": near_rmse,
                "near_fluid_temperature_query_count": near_count,
                "near_fluid_temperature_squared_error_sum": near_squared_sum,
                "far_fluid_temperature_rmse": far_rmse,
                "far_fluid_temperature_query_count": far_count,
                "far_fluid_temperature_squared_error_sum": far_squared_sum,
                "surface_temperature_rmse": float(torch.sqrt(surface_mse.clamp_min(0)).detach()),
                "material_temperature_rmse": float(torch.sqrt(material_mse.clamp_min(0)).detach()),
                "sampled_material_peak_rmse": None if peak_rmse is None else float(peak_rmse.detach()),
                "material_peak_abs_error_by_source": peak_rows,
                "q_proxy_rmse": float(torch.sqrt(q_mse.clamp_min(0)).detach()),
                "flow_rmse": flow_rmse,
                "flow_rmse_unit_by_channel": ["native dataset unit"] * 4,
                "temperature_unit": "packed_dataset_native_temperature",
                "channel_order": list(JOINT_THERMAL_CHANNEL_ORDER),
            }
            rows.append(row)
        return {"case_rows": rows}

    def reduce_native_metrics(self, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        rows = [row for record in records for row in record["case_rows"]]
        ids = [str(row["case_id"]) for row in rows]
        if len(rows) != len(self._validation_ids) or len(set(ids)) != len(ids) or set(ids) != set(self._validation_ids):
            raise ValueError("Joint Thermal validation must visit the exact exposed fixed cohort once per case.")
        result: dict[str, Any] = {
            "scope": (
                "formal canonical89 primary; exposed validation; equal-case native measurements"
                if getattr(self, "formal_validation_scope", None) == "canonical89" else
                "manual originalTRAIN600 preparation; fixed25_v1 exposed DEV22 sanity panel only"
                if self.formal_full else "fixed25_v1 exposed DEV22; equal-case native measurements"
            ),
            "case_count": len(rows),
            "case_ids": ids,
            "channel_order": list(JOINT_THERMAL_CHANNEL_ORDER),
            "field_score": float(np.mean([row["field_score"] for row in rows])),
            "flow_group_standardized_mse": float(np.mean([row["flow_group_standardized_mse"] for row in rows])),
            "thermal_group_standardized_mse": float(np.mean([row["thermal_group_standardized_mse"] for row in rows])),
            "per_case": rows,
        }
        for role in ("u", "v", "p", "omega"):
            values = np.asarray([row["flow_rmse"][role] for row in rows], dtype=np.float64)
            result[f"flow_{role}_rmse_mean"] = float(values.mean())
            result[f"flow_{role}_rmse_p90"] = float(np.quantile(values, 0.9))
        for metric in ("fluid_temperature_rmse", "surface_temperature_rmse", "material_temperature_rmse", "q_proxy_rmse"):
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_p90"] = float(np.quantile(values, 0.9))
        peak_values = np.asarray([
            row["sampled_material_peak_rmse"] for row in rows
            if row["sampled_material_peak_rmse"] is not None
        ], dtype=np.float64)
        result["sampled_material_peak_rmse_mean"] = float(peak_values.mean()) if peak_values.size else None
        result["sampled_material_peak_rmse_p90"] = float(np.quantile(peak_values, 0.9)) if peak_values.size else None
        peak_errors = [item["absolute_error"] for row in rows for item in row["material_peak_abs_error_by_source"]]
        result["material_peak_absolute_error_mean"] = float(np.mean(peak_errors)) if peak_errors else None
        for support in ("near", "far"):
            count_key = f"{support}_fluid_temperature_query_count"
            sum_key = f"{support}_fluid_temperature_squared_error_sum"
            count = int(sum(row[count_key] for row in rows))
            squared_error_sum = float(sum(row[sum_key] for row in rows))
            result[f"{support}_fluid_temperature_query_count"] = count
            result[f"{support}_fluid_temperature_rmse"] = (
                math.sqrt(squared_error_sum / count) if count else None
            )
            supported_case_rmse = [
                row[f"{support}_fluid_temperature_rmse"] for row in rows
                if row[count_key] > 0 and row[f"{support}_fluid_temperature_rmse"] is not None
            ]
            result[f"{support}_fluid_temperature_case_rmse_mean"] = (
                float(np.mean(supported_case_rmse)) if supported_case_rmse else None
            )
        return result

    def optimizer_groups(self, model, arm: str, stage: str) -> Sequence[OptimizerGroupSpec]:
        del stage
        if model is not self.model or arm != self.joint_mode:
            raise ValueError("Joint Thermal optimizer request has a mismatched model or arm.")
        names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if not names or any(not name.startswith("core.") for name in names):
            raise ValueError("Every active Thermal joint parameter must belong to the one fresh core.")
        schedule = getattr(self, "formal_optimizer_schedule", None) or ScheduleSpec(
            peak_lr=3.0e-4,
            warmup_start_lr=3.0e-5,
            warmup_epochs=min(20, self.total_epochs - 1),
            hold_through_epoch=min(1000, self.total_epochs - 1),
            total_epochs=self.total_epochs,
            final_lr=3.0e-6,
        )
        weight_decay = getattr(self, "formal_weight_decay", None)
        return (OptimizerGroupSpec("joint", names, schedule,
                                   weight_decay=1.0e-4 if weight_decay is None else weight_decay),)

    def work_counts(self, batch: TaskBatch, predictions, auxiliary_state) -> Mapping[str, int | float]:
        del batch
        return {name: float(value) for name, value in auxiliary_state.get("work", predictions.work).items()}

    def training_state_dict(self) -> Mapping[str, Any]:
        return {
            "training_mode": JOINT_TRAINING_MODE,
            "arm": self.joint_mode,
            "response_coefficient": self.response_weight,
            "operator_coefficient": self.operator_weight,
            "auxiliary_calibration": copy.deepcopy(self.auxiliary_calibration),
        }

    def load_training_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("training_mode") != JOINT_TRAINING_MODE or state.get("arm") != self.joint_mode:
            raise ValueError("Saved Thermal task state belongs to another joint mode.")
        for name, current in (("response_coefficient", self.response_weight), ("operator_coefficient", self.operator_weight)):
            if float(state.get(name, float("nan"))) != current:
                raise ValueError(f"Saved joint Thermal {name} differs from the active recipe.")
        saved = state.get("auxiliary_calibration")
        self.auxiliary_calibration = None if saved is None else copy.deepcopy(dict(saved))

    def calibrate_auxiliary_coefficients(self) -> Mapping[str, Any]:
        """Measure one fresh TRAIN-only response/operator gradient ratio.

        The root campaign calls this on its designated calibration arm/device,
        then passes the sealed coefficients unchanged to sibling modes.
        """
        if self.auxiliary_calibration is not None:
            raise RuntimeError("Joint Thermal auxiliary coefficients are already calibrated and sealed.")
        module_counts = [int(np.asarray(case["structure"]["module_present"]).sum()) for case in self.training_cases]
        selected_indices = [next((index for index, count in enumerate(module_counts) if count == desired), None)
                            for desired in (1, 3, 10, 12)]
        if any(index is None for index in selected_indices):
            raise ValueError("Joint Thermal calibration requires TRAIN examples at1,3,10,12 modules.")
        parameters = tuple(parameter for parameter in self.model.parameters() if parameter.requires_grad)
        field_gradient_norms: list[float] = []
        flow_gradient_norms: list[float] = []
        thermal_gradient_norms: list[float] = []
        operator_gradient_norms: list[float] = []
        for update_index, index in enumerate(selected_indices):
            key = SamplingKey(
                0,
                1,
                update_index,
                0,
                "joint_auxiliary_calibration",
                self.joint_mode,
                sampling_version=SamplingKey.CASE_EPOCH_VERSION,
                dataset_id="thermal_fixed25_v1",
            )
            batch = self._batch_from_indices([int(index)], 1, key=key, training=True,
                                             include_response=False, budget=self.budget)
            scene = self.make_scene(batch.scene_inputs)
            predictions, _ = self.predict_native(self.model, scene, batch.receivers, self.joint_mode, "joint")
            values = self._native_role_losses(predictions.native_main, predictions.native_prepared, batch.targets)
            flow_obj = 0.5 * values["flow_group"].mean()
            thermal_obj = 0.5 * values["thermal_group"].mean()
            base_obj = flow_obj + thermal_obj
            flow_gradient_norms.append(_gradient_norm(flow_obj, parameters, retain_graph=True))
            thermal_gradient_norms.append(_gradient_norm(thermal_obj, parameters, retain_graph=True))
            # The qualified operator penalty reads this same prepared affine
            # graph, so keep it alive until its gradient norm is measured.
            field_gradient_norms.append(_gradient_norm(base_obj, parameters, retain_graph=True))
            operator_values, _receipt = self._operator_loss(predictions, batch.targets)
            if operator_values is not None:
                operator_gradient_norms.append(_gradient_norm(operator_values.mean(), parameters))

        response_gradient_norms = []
        for update_index, family in enumerate(self.train_families):
            key = SamplingKey(
                0,
                1,
                update_index,
                0,
                "joint_response_calibration",
                self.joint_mode,
                sampling_version=SamplingKey.CASE_EPOCH_VERSION,
                dataset_id="thermal_fixed25_v1",
            )
            response_target = self._prepare_response_sample(family, training=True, key=key, budget=self.budget)
            structure = {key: value.to(self.device) for key, value in response_target["structure"].items()}
            prepared = self.model.prepare_native(
                structure,
                response_target["fluid_xy"].to(self.device),
                local_query_points=response_target["local"].to(self.device),
                ntheta=16,
                chunk_size=self.model.receiver_tile,
            )
            output = self.model.apply_native(
                prepared,
                response_target["heat"].to(self.device),
                increment=True,
            )
            # _response_sample materializes exactly the already declared TRAIN
            # addendum and preserves its original native IDs and masks.
            targets = self._response_targets_from_sample(response_target, self.device)
            response_loss = self._response_loss(output, targets)
            response_gradient_norms.append(_gradient_norm(response_loss, parameters))

        field_norm = float(np.mean(field_gradient_norms))
        response_norm = float(np.mean(response_gradient_norms))
        operator_norm = float(np.mean(operator_gradient_norms)) if operator_gradient_norms else 0.0
        target_added_gradient_ratio = 0.5
        response_coefficient = min(1.0, target_added_gradient_ratio * field_norm / max(response_norm, 1.0e-12))
        operator_coefficient = min(1.0, target_added_gradient_ratio * field_norm / max(operator_norm, 1.0e-12))
        receipt = {
            "method": "fresh_joint_init_fixed_train_gradient_ratio_v1",
            "calibration_mode": self.joint_mode,
            "calibration_case_ids": [self._train_ids[int(index)] for index in selected_indices],
            "calibration_module_counts": [1, 3, 10, 12],
            "response_family_ids": [str(item["family_id"]) for item in self.train_families],
            "flow_group_gradient_norms": flow_gradient_norms,
            "thermal_group_gradient_norms": thermal_gradient_norms,
            "balanced_field_gradient_norms": field_gradient_norms,
            "response_gradient_norms": response_gradient_norms,
            "operator_gradient_norms": operator_gradient_norms,
            "target_added_gradient_ratio": target_added_gradient_ratio,
            "response_coefficient": float(response_coefficient),
            "operator_coefficient": float(operator_coefficient),
            "q_proxy_coefficient": self.q_proxy_weight,
            "validation_values_read": False,
            "optimizer_steps": 0,
            "stored_velocity_used_only_for_operator_supervision": True,
        }
        self.response_weight = float(response_coefficient)
        self.operator_weight = float(operator_coefficient)
        self.response_coefficient = self.response_weight
        self.operator_coefficient = self.operator_weight
        self.auxiliary_calibration = receipt
        self._joint_loss_metadata = self._build_loss_metadata()
        return copy.deepcopy(receipt)

    @staticmethod
    def _response_targets_from_sample(sample: Mapping[str, Any], device: torch.device):
        from .unified_task import ThermalResponseTargets

        return ThermalResponseTargets(
            family_id=str(sample["family_id"]),
            fluid=sample["fluid_target"].to(device=device, dtype=torch.float32),
            surface=sample["surface_target"].to(device=device, dtype=torch.float32),
            surface_mask=sample["surface_mask"].to(device=device, dtype=torch.float32),
            material=sample["material_target"].to(device=device, dtype=torch.float32),
            material_mask=sample["material_mask"].to(device=device, dtype=torch.float32),
        )


def build_thermal_joint_task(
    mode: str = "J-H",
    device: torch.device | str = "cpu",
    hidden: int = 128,
    message: int = 128,
    microbatch_size: int = JOINT_MICROBATCH_DEFAULT,
    effective_batch_size: int = JOINT_EFFECTIVE_BATCH_DEFAULT,
    primary_queries: int = JOINT_PRIMARY_QUERY_DEFAULT,
    manifest: Mapping[str, Any] | str | Path | None = None,
    formal_full: bool = False,
    regional_anchors: int = 16,
    locality_prior_strength: float = 0.0,
    receiver_tile: int = 512,
    seed: int = 0,
    total_epochs: int = 2500,
    operator_rows_per_case: int = OPERATOR_ROWS_PER_CASE,
    response_coefficient: float = 0.1,
    operator_coefficient: float = 0.1,
    data_path: str | Path | None = None,
    atlas_directory: str | Path = DEFAULT_ATLAS_DIRECTORY,
    depth: int = 2,
    validation_queries: int | None = None,
    auxiliary_calibration: Mapping[str, Any] | None = None,
    validation_scope: str | None = None,
    optimizer_schedule: ScheduleSpec | None = None,
    weight_decay: float | None = None,
    native_sampling_protocol: str | None = None,
) -> tuple[JointThermalRegionalAdapter, JointThermalTask]:
    """Construct a fresh joint candidate and its fixed25_v1 provider.

    ``formal_full=True`` prepares the original600 TRAIN cases and fits all
    normalizers on those cases, but only returns a manual-only recipe/provider;
    this factory never starts or advances training.
    """
    if mode not in JOINT_THERMAL_MODES:
        raise ValueError(f"Joint Thermal mode must be one of {JOINT_THERMAL_MODES}.")
    if isinstance(locality_prior_strength, (bool, np.bool_)):
        raise TypeError("Joint Thermal locality prior strength must be numeric, not boolean.")
    try:
        locality_prior_strength = float(locality_prior_strength)
    except (TypeError, ValueError) as error:
        raise ValueError("Joint Thermal locality prior strength must be finite and nonnegative.") from error
    if not math.isfinite(locality_prior_strength) or locality_prior_strength < 0:
        raise ValueError("Joint Thermal locality prior strength must be finite and nonnegative.")
    if locality_prior_strength and mode != "J-H":
        raise ValueError("Joint Thermal locality prior is only supported by J-H.")
    if microbatch_size < 1 or effective_batch_size < 1 or microbatch_size > effective_batch_size:
        raise ValueError("Joint Thermal microbatch must be positive and no larger than its effective batch.")
    if primary_queries < 1 or receiver_tile < 1:
        raise ValueError("Joint Thermal primary query count and receiver tile must be positive.")
    if operator_rows_per_case != OPERATOR_ROWS_PER_CASE:
        raise ValueError("The declared joint Thermal objective requires exactly128 operator rows per case.")
    if formal_full and total_epochs != 5000:
        raise ValueError("The manual fullTRAIN joint recipe is bound to a5000-epoch horizon.")
    if any(option is not None for option in (validation_scope, optimizer_schedule, weight_decay,
                                            native_sampling_protocol)) and not formal_full:
        raise ValueError("Formal comparison controls require the separate fullTRAIN identity.")
    if validation_scope not in (None, "canonical89"):
        raise ValueError("Joint Thermal formal validation scope must be canonical89.")
    if optimizer_schedule is not None and (not isinstance(optimizer_schedule, ScheduleSpec)
                                          or optimizer_schedule.total_epochs != total_epochs):
        raise ValueError("Formal optimizer schedule must match the declared horizon.")
    if weight_decay is not None and (type(weight_decay) not in (int, float)
                                    or not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError("Formal weight decay must be finite and nonnegative.")
    if native_sampling_protocol not in (None, "baseline_formal_v1"):
        raise ValueError("Unsupported formal native sampling protocol.")
    if native_sampling_protocol is not None and validation_scope != "canonical89":
        raise ValueError("Baseline formal sampling requires the canonical89 primary panel.")

    resolved_manifest, resolved_data_path, manifest_path = _load_data_binding(manifest, data_path)
    if formal_full:
        _source, catalog = read_case_catalog(resolved_data_path)
        train_ids = tuple(record["case_id"] for record in catalog if record["split"] == "train")
        validation_ids = development_case_ids(resolved_manifest, "test")
        if len(train_ids) != 600:
            raise ValueError("Manual joint Thermal fullTRAIN preparation requires the original600 TRAIN cases.")
        if validation_scope == "canonical89":
            original90 = tuple(record["case_id"] for record in catalog if record["split"] == "test")
            validation_ids = tuple(case_id for case_id in original90 if case_id != "0273")
            if (len(original90) != 90 or original90.count("0273") != 1
                    or len(validation_ids) != 89 or len(set(validation_ids)) != 89
                    or set(train_ids).intersection(validation_ids)):
                raise ValueError("Formal validation requires original90 with duplicate0273 excluded from canonical89.")
    else:
        train_ids = development_case_ids(resolved_manifest, "train")
        validation_ids = development_case_ids(resolved_manifest, "test")
        if len(train_ids) != 150 or len(validation_ids) != 22:
            raise ValueError("Joint Thermal development requires literal fixed25_v1 150/22 membership.")

    normalizer = fit_global_normalizer(resolved_data_path, train_ids)
    training_cases = _read_selected_cases(
        resolved_data_path,
        train_ids,
        split="train",
        normalizer=normalizer,
        include_grid=True,
    )
    validation_cases = _read_selected_cases(
        resolved_data_path,
        validation_ids,
        split="test",
        normalizer=normalizer,
        include_grid=False,
    )
    build_balances, _response_scales = _load_recipe_helpers()
    balances = build_balances(training_cases)
    families = _read_response_families(Path(atlas_directory).expanduser().resolve(), TRAIN_RESPONSE_IDS)

    from honf_runtime.compat import set_seed

    set_seed(int(seed))
    model = JointThermalRegionalAdapter(
        mode=mode,
        normalization_stats=normalizer.stats,
        hidden=hidden,
        message=message,
        regional_anchors=regional_anchors,
        depth=depth,
        receiver_tile=receiver_tile,
        locality_prior_strength=locality_prior_strength,
        seed=seed,
    ).to(device)
    budget = {
        "fluid_queries": int(primary_queries),
        "material_queries_per_module": 32,
        "surface_stride": RESPONSE_SURFACE_STRIDE,
        "microbatch_cases": int(microbatch_size),
        "effective_cases": int(effective_batch_size),
        "operator_rows_per_case": int(operator_rows_per_case),
        "receiver_tile": int(receiver_tile),
    }
    if validation_queries is not None:
        if validation_queries < 1:
            raise ValueError("Joint Thermal validation query count must be positive.")
        validation_budget = {**budget, "fluid_queries": int(validation_queries)}
    else:
        validation_budget = None
    stat = resolved_data_path.stat()
    source_binding = {
        "dataset_path": str(resolved_data_path),
        "dataset_id": "thermal_channel_global_v1",
        "dataset_file_size_bytes": int(stat.st_size),
        "dataset_file_mtime_ns": int(stat.st_mtime_ns),
        "dataset_catalog_metadata_sha256": resolved_manifest["source"]["metadata_sha256"],
        "development_manifest_path": None if manifest_path is None else str(manifest_path),
        "development_manifest_sha256": resolved_manifest["manifest_sha256"],
        "formal_scope": "original_train_600" if formal_full else "fixed25_v1_train_150",
    }
    provider = JointThermalTask(
        model=model,
        training_cases=training_cases,
        validation_cases=validation_cases,
        manifest=resolved_manifest,
        data_path=resolved_data_path,
        source_binding=source_binding,
        train_families=families,
        balances=balances,
        normalization_stats=normalizer.stats,
        mode=mode,
        device=device,
        budget=budget,
        effective_batch_size=effective_batch_size,
        total_epochs=total_epochs,
        formal_full=formal_full,
        response_coefficient=response_coefficient,
        operator_coefficient=operator_coefficient,
        operator_rows_per_case=operator_rows_per_case,
        atlas_directory=Path(atlas_directory).expanduser().resolve(),
        auxiliary_calibration=auxiliary_calibration,
        validation_scope=validation_scope,
        optimizer_schedule=optimizer_schedule,
        weight_decay=weight_decay,
        native_sampling_protocol=native_sampling_protocol,
    )
    if validation_budget is not None:
        provider.validation_budget = validation_budget
    if native_sampling_protocol == "baseline_formal_v1":
        provider.validation_sampling_indices = {case_id: index for index, case_id in enumerate(original90)
                                                if case_id in set(validation_ids)}
    return model, provider
