"""Sparse differentiable frozen-HONF bridge for stage-four consistency.

Normalized generated layout ``D`` and physical context ``c`` run through the
current autonomous predicted-port forward model. Request ``R`` receives smooth
functional residuals, planned compact ``G`` is compared with differentiable
realized ``G_hat``, and an optional corrector is evaluated exactly once. The
forward parameters stay frozen; gradients exist only with respect to inverse
outputs. Exact inference/evaluation still uses the non-smooth verifier.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from honf_inverse_core.models.hierarchical_inverse import HierarchicalInverseDesigner
from honf_inverse_core.models.matching import match_tokens
from honf_inverse_core.training.losses import joint_consistency_losses, layout_geometry_loss

from .differentiable_functionals import functional_token_values, normalized_request_residuals
from .verifier import FrozenThermalChannelVerifier


def bounded_layout_correction_target(
    target_layout: torch.Tensor,
    sampled_layout: torch.Tensor,
    sampled_present: torch.Tensor,
    target_present: torch.Tensor,
    *,
    max_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a bounded target only where sampled and feasible slots overlap.

    Both layouts use the case contract's active-first, lexicographic module
    order.  Presence counts can nevertheless differ.  Since the one-pass
    corrector intentionally preserves topology, a sampled module without a
    corresponding feasible target module receives no supervised displacement
    instead of being attracted to an inactive zero-padding slot.
    """

    if target_layout.shape != sampled_layout.shape or target_layout.ndim != 3:
        raise ValueError("Correction target layouts must share shape [B,M,3].")
    if sampled_present.shape != target_present.shape or sampled_present.shape != sampled_layout.shape[:2]:
        raise ValueError("Correction target presence masks must share shape [B,M].")
    overlap = sampled_present.float() * target_present.float()
    target = (target_layout - sampled_layout.detach()).clamp(
        -float(max_delta), float(max_delta)
    )
    return target * overlap.unsqueeze(-1), overlap


class DifferentiableThermalChannelVerifier:
    """Own fixed probe grids and expose a trainer-compatible joint-loss hook."""

    def __init__(
        self,
        verifier: FrozenThermalChannelVerifier,
        *,
        inverse_heat_mean: float,
        inverse_heat_std: float,
        functional_mean: np.ndarray,
        functional_std: np.ndarray,
        query_grid: tuple[int, int] = (32, 16),
        local_grid_size: int = 18,
        matching: str = "canonical",
    ) -> None:
        self.verifier = verifier
        self.model = verifier.model
        self.device = verifier.device
        self.inverse_heat_mean = float(inverse_heat_mean)
        self.inverse_heat_std = float(inverse_heat_std)
        self.functional_mean = torch.as_tensor(functional_mean, dtype=torch.float32, device=self.device)
        self.functional_std = torch.as_tensor(functional_std, dtype=torch.float32, device=self.device)
        self.matching = str(matching)
        if self.matching not in {"canonical", "hungarian", "sinkhorn"}:
            raise ValueError(f"Unsupported compact-plan matching mode: {self.matching!r}")
        nx, ny = map(int, query_grid)
        lx = float(self.model.config.core_honf.domain_length_x)
        ly = float(self.model.config.core_honf.domain_length_y)
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, ly, ny, device=self.device),
            torch.linspace(0.0, lx, nx, device=self.device),
            indexing="ij",
        )
        self.query_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        axis = torch.linspace(-1.0, 1.0, int(local_grid_size), device=self.device)
        local_y, local_x = torch.meshgrid(axis, axis, indexing="ij")
        mask = local_x.square() + local_y.square() <= 1.0
        self.local_query = torch.stack([local_x[mask], local_y[mask]], dim=-1)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def _denormalize_layout(
        self,
        layout: torch.Tensor,
        present: torch.Tensor,
        context_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        domain = context_raw[:, 8:10]
        centers = layout[..., :2] * domain[:, None, :]
        heat = (layout[..., 2] * self.inverse_heat_std + self.inverse_heat_mean) * present
        return centers, heat

    def _forward_heat(self, physical_heat: torch.Tensor) -> torch.Tensor:
        if not self.verifier.normalize_inputs:
            return physical_heat
        stats = self.verifier.checkpoint.get("global_normalization_stats", {})
        mean = torch.as_tensor(stats.get("heat_power_mean", 0.0), device=physical_heat.device, dtype=physical_heat.dtype)
        std = torch.as_tensor(stats.get("heat_power_std", 1.0), device=physical_heat.device, dtype=physical_heat.dtype).clamp_min(1.0e-8)
        return (physical_heat - mean) / std

    def _denormalize_outputs(
        self,
        field: torch.Tensor,
        internal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.verifier.normalize_targets:
            return field, internal
        stats = self.verifier.checkpoint.get("global_normalization_stats", {})
        field_mean = torch.as_tensor(stats["field_mean_by_channel"], device=field.device, dtype=field.dtype)
        field_std = torch.as_tensor(stats["field_std_by_channel"], device=field.device, dtype=field.dtype).clamp_min(1.0e-8)
        internal_mean = torch.as_tensor(stats["internal_temperature_mean"], device=field.device, dtype=field.dtype)
        internal_std = torch.as_tensor(stats["internal_temperature_std"], device=field.device, dtype=field.dtype).clamp_min(1.0e-8)
        return field * field_std + field_mean, internal * internal_std + internal_mean

    @staticmethod
    def _canonical_compact(
        organizer: Mapping[str, torch.Tensor],
        heat: torch.Tensor,
        present: torch.Tensor,
        domain: torch.Tensor,
    ) -> torch.Tensor:
        A_mh = organizer["A_mh"] * present.unsqueeze(-1)
        A_eh = organizer["A_eh"]
        source = organizer["hyper_source_coords"]
        region = (
            organizer["hyper_region_coords"]
            if "hyper_region_coords" in organizer
            else organizer["hyper_thermal_region_coords"]
        )
        module_mass = organizer["hyper_module_mass"]
        env_mass = organizer["hyper_env_mass"]
        env_coords = organizer["env_coords"]
        strength = torch.sqrt(module_mass * env_mass + 1.0e-6)
        active = (strength > 0.05).to(strength.dtype)
        column_weights = A_eh / A_eh.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        centered = env_coords[:, :, None, :] - region[:, None, :, :]
        scale = torch.sqrt((column_weights.unsqueeze(-1) * centered.square()).sum(dim=1).clamp_min(0.0))
        heat_mass = (A_mh * (heat.abs() * present).unsqueeze(-1)).sum(dim=1)
        heat_fraction = heat_mass / heat_mass.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        owners = A_mh.argmax(dim=-1)
        hard = torch.zeros_like(module_mass)
        hard.scatter_add_(1, owners, present)
        hard = hard / present.sum(dim=1, keepdim=True).clamp_min(1.0)
        source_normalized = source / domain[:, None, :]
        region_normalized = region / domain[:, None, :]
        scale_normalized = scale / domain[:, None, :]
        plan = torch.stack(
            [
                active,
                source_normalized[..., 0], source_normalized[..., 1],
                region_normalized[..., 0], region_normalized[..., 1],
                module_mass, env_mass, strength,
                scale_normalized[..., 0], scale_normalized[..., 1],
                heat_fraction, hard,
            ],
            dim=-1,
        )
        sorted_rows = []
        for row in plan:
            keys = [
                (
                    0 if float(row[index, 0].detach()) > 0.5 else 1,
                    float(row[index, 1].detach()), float(row[index, 2].detach()),
                    float(row[index, 3].detach()), float(row[index, 4].detach()),
                    -float(row[index, 7].detach()), index,
                )
                for index in range(row.shape[0])
            ]
            order = torch.as_tensor(sorted(range(row.shape[0]), key=keys.__getitem__), device=row.device)
            sorted_rows.append(row[order])
        return torch.stack(sorted_rows)

    def forward_candidate(
        self,
        layout: torch.Tensor,
        present: torch.Tensor,
        context_raw: torch.Tensor,
        request: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centers, physical_heat = self._denormalize_layout(layout, present, context_raw)
        batch = layout.shape[0]
        domain = context_raw[:, 8:10]
        expected = layout.new_tensor(
            [self.model.config.core_honf.domain_length_x, self.model.config.core_honf.domain_length_y]
        )
        if not torch.allclose(domain, expected.expand_as(domain), atol=1.0e-5, rtol=1.0e-5):
            raise ValueError("Generated context domain must match the frozen checkpoint domain.")
        structure = {
            "re": context_raw[:, 0:1],
            "u_in": context_raw[:, 1:2],
            "module_centers": centers,
            "heat_powers": self._forward_heat(physical_heat),
            "module_present": present,
            "material_params": context_raw[:, 2:8],
            "domain_length_x": context_raw[:, 8:9],
            "domain_length_y": context_raw[:, 9:10],
        }
        query = self.query_xy.unsqueeze(0).expand(batch, -1, -1)
        local = self.local_query.unsqueeze(0).expand(batch, -1, -1)
        local_module_params = layout.new_zeros((batch, layout.shape[1], 7))
        local_module_params[..., 0] = physical_heat
        local_module_params[..., 1] = context_raw[:, None, 5]
        local_module_params[..., 2] = context_raw[:, None, 3]
        local_module_params = local_module_params * present.unsqueeze(-1)
        outputs = self.model(
            structure,
            query,
            interface_condition=None,
            local_module_params=local_module_params,
            teacher_port_tokens=None,
            local_query_points=local,
            local_port_condition_mode="predicted",
            mixed_teacher_ratio=0.0,
        )
        field, internal = self._denormalize_outputs(
            outputs["pred_field"], outputs["pred_internal_temperature"]
        )
        realized = self._canonical_compact(outputs["organizer_aux"], physical_heat, present, domain)
        functional = functional_token_values(
            pred_field=field,
            query_xy=query,
            pred_internal_temperature=internal,
            module_centers=centers,
            module_present=present,
            module_radius=context_raw[:, 7],
            domain_length_x=context_raw[:, 8],
            domain_length_y=context_raw[:, 9],
            request=request,
        )
        residual, request_loss = normalized_request_residuals(
            functional, request, self.functional_mean, self.functional_std
        )
        return realized, residual, request_loss

    def joint_loss_hook(
        self,
        designer: HierarchicalInverseDesigner,
        batch: Mapping[str, Any],
        encoding: Any,
        planned_plan: torch.Tensor,
        layout_endpoint: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        present = batch["module_present"].float()
        realized, residual, request_loss = self.forward_candidate(
            layout_endpoint, present, batch["context_raw"].float(), batch["request"]
        )
        raw_request_loss = request_loss
        realized_aligned = match_tokens(planned_plan, realized, method=self.matching)
        plan_distance = (planned_plan - realized_aligned).square().mean()
        geometry = layout_geometry_loss(layout_endpoint, present, batch["geometry_constraints"])
        correction_magnitude = None
        correction_target_loss = None
        if designer.corrector is not None:
            correction = designer.correct_once(
                planned_plan=planned_plan,
                layout=layout_endpoint,
                module_present=present,
                realized_plan=realized,
                request_residuals=residual,
                encoding=encoding,
                enabled=True,
            )
            assert correction is not None
            corrected_realized, _, corrected_request = self.forward_candidate(
                correction.corrected_layout,
                present,
                batch["context_raw"].float(),
                batch["request"],
            )
            # Train the optional pass to improve over its own immutable raw
            # candidate, rather than merely reaching a low absolute loss by
            # moving the shared layout flow. Detaching the raw baseline avoids
            # the degenerate solution of making the uncorrected path worse.
            paired_improvement = torch.relu(
                corrected_request - raw_request_loss.detach() + 0.01
            )
            request_loss = corrected_request + paired_improvement
            corrected_aligned = match_tokens(
                correction.corrected_plan, corrected_realized, method=self.matching
            )
            plan_distance = (correction.corrected_plan - corrected_aligned).square().mean()
            correction_magnitude = correction.magnitude
            max_delta = float(designer.corrector.max_layout_delta)
            target_delta, target_mask = bounded_layout_correction_target(
                batch["layout"].float(),
                layout_endpoint,
                present,
                batch.get("target_module_present", present),
                max_delta=max_delta,
            )
            per_slot = (correction.delta_layout - target_delta).square().mean(dim=-1)
            correction_target_loss = (
                (per_slot * target_mask).sum() / target_mask.sum().clamp_min(1.0)
            )
        return joint_consistency_losses(
            request_loss=request_loss,
            plan_distance=plan_distance,
            geometry_loss=geometry,
            correction_magnitude=correction_magnitude,
            correction_target_loss=correction_target_loss,
        )


__all__ = ["DifferentiableThermalChannelVerifier", "bounded_layout_correction_target"]
