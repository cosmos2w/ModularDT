"""Thermal native and response task for InteractionGroupCore development arms."""

from __future__ import annotations

import copy
import math
import time

import torch

from ..interaction_groups import ThermalInteractionGroupsAdapter
from .contextual_joint_task import ThermalContextualJointTask


class ThermalInteractionGroupTask(ThermalContextualJointTask):
    """Maintain the joint native objective and response supervision without legacy probes."""

    def _make_organizer(self):
        from .interaction_group_objective import InteractionGroupObjectiveRuntime

        self._native_weights_contract = copy.deepcopy(self.native_weights)
        return InteractionGroupObjectiveRuntime(self.model, self)

    def identity_payload(self):
        identity = super().identity_payload()
        identity.update(
            {
                "schema": "thermal_interaction_groups_provider_v1",
                "architecture": self.model.architecture,
                "adapter_state": self.model.state_payload(),
                "interaction_group_schema": "interaction_groups_v1",
                "objective": {
                    "native_weights": copy.deepcopy(self.native_weights),
                    "joint_fields": ["u", "v", "p", "temperature"],
                    "response_every_primary_updates": 4,
                    "response_shared_gradient_cap_fraction": 0.1,
                    "native_weights_fixed": True,
                    "native_weights_frozen_through_epoch": 100,
                    "response_calibration_epoch": 1,
                },
            }
        )
        return identity

    def _response_gradient(self, native, *, index, calibrate=False):
        from honf_runtime.thermal_soft_organization import gradient_balance, gradient_norm

        named = {name: parameter for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        loss, detail = self.responses.loss(self.model, index)
        gradients = torch.autograd.grad(loss, tuple(named.values()), allow_unused=True)
        raw = {
            name: gradient.detach()
            for (name, _parameter), gradient in zip(named.items(), gradients)
            if gradient is not None
        }
        # Cap response gradients on the shared encoder and field path. Group
        # frontier/proposal projections remain owned by the group objective.
        shared = tuple(
            name
            for name in native
            if name in raw
            and (
                name.startswith(
                    (
                        "core.module_encoder.",
                        "core.environment_encoder.",
                        "core.operating_encoder.",
                        "core.fusion_trunk.",
                        "core.field_head.",
                    )
                )
                or (name.startswith("core.layers.") and ".frontier." not in name and ".ee." not in name)
            )
        )
        balance = gradient_balance(native, raw, shared)
        native_norm, response_norm = balance["native_norm"], balance["term_norm"]
        if calibrate:
            self.state["response_coefficient"] = (
                0.1 * native_norm / response_norm if native_norm > 0 and response_norm > 0 else 0.0
            )
            self.state.setdefault("response_calibrations", []).append(
                {
                    "epoch": int(self.model.core.organization_epoch),
                    "balance": balance,
                    "coefficient": self.state["response_coefficient"],
                    "kind": detail.get("kind"),
                }
            )
        coefficient = float(self.state.get("response_coefficient", 0.0))
        cap = min(1.0, 0.1 * native_norm / (coefficient * response_norm)) if coefficient * response_norm > 0 else 1.0
        return (
            loss,
            detail,
            raw,
            coefficient * cap,
            {
                "native_shared_norm": native_norm,
                "raw_shared_norm": response_norm,
                "raw_full_norm": gradient_norm(raw),
                "coefficient": coefficient,
                "cap_scale": cap,
            },
        )

    def before_optimizer_step(self, model, optimizer, losses, work, *, epoch, update_index):
        if model is not self.model or model.architecture not in {"G-EE", "G-F", "G-H"}:
            raise ValueError("Wrong InteractionGroupCore optimizer owner or architecture.")
        if self.native_weights != self._native_weights_contract:
            raise RuntimeError("Interaction-group training must preserve the declared native loss weights.")
        named = dict(model.named_parameters())
        native = {
            name: parameter.grad.detach().clone() for name, parameter in named.items() if parameter.grad is not None
        }

        # The group objective owns its architecture-specific evidence and policy
        # updates. It intentionally replaces the old contextual calibration and
        # functional-probe callbacks.
        self.organizer.before_optimizer_step(
            model,
            optimizer,
            losses,
            work,
            epoch=epoch,
            update_index=update_index,
        )

        if update_index == 0 and epoch == 1:
            _, detail, _, _, _ = self._response_gradient(native, index=0, calibrate=True)
            self.state["calibration_response_scene_visits"] = self.state.get("calibration_response_scene_visits", 0) + 2
            self.state["response_calibration_epoch"] = 1

        primary_update = (
            (epoch - 1) * math.ceil(len(self.training_cases) / self.effective_batch_size) + update_index + 1
        )
        if primary_update % 4 == 0:
            index = primary_update // 4 - 1
            loss, detail, gradients, scale, balance = self._response_gradient(native, index=index)
            for name, gradient in gradients.items():
                contribution = gradient * scale
                if named[name].grad is None:
                    named[name].grad = contribution.clone()
                else:
                    named[name].grad.add_(contribution)
            self.state["response_microbatches"] = self.state.get("response_microbatches", 0) + 1
            self.state["response_scene_visits"] = self.state.get("response_scene_visits", 0) + 2
            self.state.setdefault("response_records", []).append(
                {
                    "update": primary_update,
                    "kind": detail.get("kind"),
                    "loss": float(loss.detach()),
                    "exposure": detail.get("exposure", {}),
                    "gradient": balance,
                }
            )
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
                    _atomic_json(
                        path,
                        {
                            "schema": "ordinary_clean_stop_request_v1",
                            "request_id": "interaction-groups-budget",
                            "requested_unix": time.time(),
                            "requested_by": "internal_interaction_groups_budget",
                            "reason": "finite training wall budget reached",
                        },
                    )


def build_thermal_interaction_group_task(config, *, arm=None, device="cpu"):
    """Build the selected-TRAIN data provider with a versioned group-core adapter."""
    import copy as _copy

    import numpy as np

    from ..data.datasets import fit_global_normalizer
    from ..data.development_split import development_case_ids, read_case_catalog
    from .contextual_joint_task import (
        _load_data_binding,
        _read_selected_cases,
    )

    recipe = _copy.deepcopy(dict(config))
    dataset = recipe["dataset"]
    manifest, data_path, manifest_path = _load_data_binding(dataset["manifest_path"], dataset["data_path"])
    source, catalog = read_case_catalog(data_path)
    if recipe.get("training", {}).get("membership") != "fixed25_train150_dev22":
        raise ValueError(
            "Interaction-group development requires fixed25 TRAIN150/DEV22; full-data work is a separate protocol."
        )
    train_ids = tuple(development_case_ids(manifest, "train"))
    val_ids = tuple(development_case_ids(manifest, "test"))
    normalizer = fit_global_normalizer(data_path, train_ids)
    training = _read_selected_cases(data_path, train_ids, split="train", normalizer=normalizer, include_grid=False)
    validation = _read_selected_cases(data_path, val_ids, split="test", normalizer=normalizer, include_grid=False)
    heat_values = np.concatenate(
        [
            np.asarray(case["structure"]["heat_powers"])[np.asarray(case["structure"]["module_present"]) > 0.5]
            for case in training
        ]
    )
    heat_scale = max(float(np.sqrt(np.mean(heat_values.astype(np.float64) ** 2))), 1e-6)
    group_config = recipe["interaction_groups"]["model"]
    group_config = {key: value for key, value in group_config.items() if key != "forest_neighbors"}
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
        "init_seed": 0,
        **group_config,
        "architecture": arm or recipe["interaction_groups"]["arm"],
        "schema": "interaction_groups_v1",
    }
    model = ThermalInteractionGroupsAdapter(normalizer.stats, core_config=core_config, heat_scale=heat_scale).to(device)
    provider = ThermalInteractionGroupTask(
        model=model,
        training_cases=training,
        validation_cases=validation,
        manifest=manifest,
        data_path=data_path,
        recipe=recipe,
        device=device,
        manifest_path=manifest_path,
        catalog=catalog,
        source_binding={"dataset_path": str(data_path), "existing_catalog": source},
    )
    return model, provider
