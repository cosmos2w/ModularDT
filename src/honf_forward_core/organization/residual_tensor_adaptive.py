"""Case-adaptive tensor residual organizer.

The Phase-2 organizer is intentionally isolated from the scalar Phase-1
implementation.  It constructs one centered vector interaction tensor and
then extracts non-negative rank-one tensor mechanisms with reductions and
elementwise products only.  Its parameter shapes depend on the established
hidden width and the configured interaction width, never on a runtime module
count or inferred mechanism count.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import UnifiedForwardConfig
from .helpers import (
    EPS,
    _as_batched_coords,
    _assignment_purity,
    _descriptor_first_features,
    _mechanism_descriptors,
    _relative_delta,
    _weighted_coords,
    _weighted_scale,
)


class CaseAdaptiveTensorResidualOrganizer(nn.Module):
    """Extract case-specific tensor mechanisms from encoded HONF tokens.

    The implementation follows the Phase-2 formulation:

    * environment tokens are centered before content projection;
    * the global token only supplies multiplicative FiLM modulation;
    * a centered signed vector score is squared, giving a zero-baseline
      non-negative interaction tensor;
    * residual extraction is algebraic and strength is kept separate from
      incidence membership;
    * hard support is used for forward values in both train and eval, with a
      straight-through soft-support surrogate for gradients.
    """

    def __init__(self, config: UnifiedForwardConfig):
        """Initialize shared tensor projections and batched state mixer."""

        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        interaction_dim = int(config.residual_interaction_dim)
        geometry_frequencies = int(config.residual_coupling_fourier_frequencies)
        geometry_dim = 2 + 4 * geometry_frequencies
        geometry_hidden_dim = max(hidden_dim // 2, 8)

        self.module_content_proj = nn.Linear(hidden_dim, interaction_dim)
        self.environment_content_proj = nn.Linear(hidden_dim, interaction_dim)
        self.global_module_film = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, interaction_dim),
        )
        self.global_environment_film = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, interaction_dim),
        )
        # A small geometry network is evaluated once before residual
        # extraction.  No hidden-width transform is called in the loop.
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_dim, geometry_hidden_dim),
            nn.GELU(),
            nn.Linear(geometry_hidden_dim, interaction_dim),
        )

        self.me_context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.module_value = nn.Linear(hidden_dim, hidden_dim)
        self.environment_value = nn.Linear(hidden_dim, hidden_dim)
        self.content_to_state = nn.Linear(interaction_dim, hidden_dim)

        # mbar, ebar, cbar, g (4H), source/region coordinates (4), and
        # strength/rho/marginal (3).
        mechanism_input_dim = 4 * hidden_dim + 7
        mechanism_hidden_dim = max(hidden_dim, 32)
        self.mechanism_mixer = nn.Sequential(
            nn.LayerNorm(mechanism_input_dim) if config.use_layer_norm else nn.Identity(),
            nn.Linear(mechanism_input_dim, mechanism_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(mechanism_hidden_dim, hidden_dim),
        )

        # The final FiLM layers are zero initialized so that a fresh model
        # starts with multiplicative scale exactly one.
        nn.init.zeros_(self.global_module_film[-1].weight)
        nn.init.zeros_(self.global_module_film[-1].bias)
        nn.init.zeros_(self.global_environment_film[-1].weight)
        nn.init.zeros_(self.global_environment_film[-1].bias)

    @staticmethod
    def _validate_inputs(
        module_tokens: torch.Tensor,
        env_tokens: torch.Tensor,
        module_centers: torch.Tensor,
        env_coords: torch.Tensor,
        module_present: torch.Tensor,
        global_token: Optional[torch.Tensor],
    ) -> None:
        """Validate the generic organizer tensor contract."""

        if module_tokens.ndim != 3:
            raise ValueError("module_tokens must have shape [B,M,H].")
        if env_tokens.ndim not in {2, 3}:
            raise ValueError("env_tokens must have shape [B,E,H] or [E,H].")
        if module_centers.ndim not in {2, 3} or module_centers.shape[-1] != 2:
            raise ValueError("module_centers must have shape [B,M,2] or [M,2].")
        if env_coords.ndim not in {2, 3} or env_coords.shape[-1] != 2:
            raise ValueError("env_coords must have shape [B,E,2] or [E,2].")
        if module_present.ndim != 2:
            raise ValueError("module_present must have shape [B,M].")
        if module_tokens.shape[:2] != module_present.shape:
            raise ValueError("module_tokens and module_present must agree on [B,M].")
        if global_token is not None:
            if global_token.ndim != 2 or global_token.shape[0] != module_tokens.shape[0]:
                raise ValueError("global_token must have shape [B,H].")

    @staticmethod
    def _geometry_features(delta: torch.Tensor, cfg: UnifiedForwardConfig) -> torch.Tensor:
        """Encode normalized relative offsets using the configured Fourier map."""

        scale_x, scale_y = cfg.spatial_scale()
        scales = delta.new_tensor([max(float(scale_x), EPS), max(float(scale_y), EPS)])
        normalized = delta / scales
        pieces = [normalized]
        for frequency in range(1, int(cfg.residual_coupling_fourier_frequencies) + 1):
            angle = 2.0 * math.pi * float(frequency) * normalized
            pieces.extend((torch.sin(angle), torch.cos(angle)))
        return torch.cat(pieces, dim=-1)

    @staticmethod
    def _active_sum(values: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """Sum a tensor over active module rows without packed-padding drift."""

        if values.ndim < 2 or active_mask.ndim != 2 or values.shape[:2] != active_mask.shape:
            raise ValueError("active_sum expects values [B,M,...] and active_mask [B,M].")
        batch_indices = (
            torch.arange(values.shape[0], device=values.device)[:, None]
            .expand_as(active_mask)[active_mask]
        )
        active_values = values[active_mask]
        result = values.new_zeros((values.shape[0], *values.shape[2:]))
        return result.index_add(0, batch_indices, active_values)

    @classmethod
    def _normalize_module_factor(
        cls,
        values: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize non-negative module factors over active rows."""

        values = values.clamp_min(0.0) * active_mask.to(dtype=values.dtype)
        denominator = cls._active_sum(values.unsqueeze(-1), active_mask).squeeze(-1)
        count = active_mask.sum(dim=-1).clamp_min(1).to(dtype=values.dtype)
        fallback = active_mask.to(dtype=values.dtype) / count.unsqueeze(-1)
        normalized = values / denominator.unsqueeze(-1).clamp_min(torch.finfo(values.dtype).tiny)
        return torch.where(
            denominator.unsqueeze(-1) > torch.finfo(values.dtype).tiny,
            normalized,
            fallback,
        )

    @staticmethod
    def _normalize_dense_factor(values: torch.Tensor) -> torch.Tensor:
        """Normalize a non-negative environment/content factor with fallback."""

        values = values.clamp_min(0.0)
        denominator = values.sum(dim=-1, keepdim=True)
        fallback = values.new_full(values.shape, 1.0 / float(max(values.shape[-1], 1)))
        normalized = values / denominator.clamp_min(torch.finfo(values.dtype).tiny)
        return torch.where(
            denominator > torch.finfo(values.dtype).tiny,
            normalized,
            fallback,
        )

    @classmethod
    def _supported_factor_normalize(
        cls,
        factor: torch.Tensor,
        support: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize factors over the mechanism axis under hard/soft support.

        The denominator is a true factor mass (not mechanism amplitude).  A
        first-supported mechanism is used only for a genuine zero denominator;
        this keeps active token rows normalized and all inactive rows exactly
        zero, including zero-interaction baselines.
        """

        support = support.to(device=factor.device, dtype=factor.dtype)
        weighted = factor * support[:, None, :]
        denominator = weighted.sum(dim=-1, keepdim=True)
        if token_mask is None:
            active = torch.ones_like(denominator, dtype=torch.bool)
        else:
            active = token_mask.to(device=factor.device, dtype=torch.bool).unsqueeze(-1)
        supported = support > 0
        first_index = supported.to(dtype=factor.dtype).argmax(dim=-1)
        first_hot = F.one_hot(first_index, num_classes=factor.shape[-1]).to(factor.dtype)
        fallback = first_hot[:, None, :].expand_as(weighted)
        tiny = torch.finfo(factor.dtype).tiny
        normalized = weighted / denominator.clamp_min(tiny)
        normalized = torch.where(denominator > tiny, normalized, fallback)
        normalized = normalized * active.to(dtype=factor.dtype)
        return normalized, denominator.squeeze(-1)

    @classmethod
    def _active_weighted_coords(
        cls,
        coords: torch.Tensor,
        weights: torch.Tensor,
        active_mask: torch.Tensor,
        cfg: UnifiedForwardConfig,
    ) -> torch.Tensor:
        """Reduce module coordinates while excluding packed inactive rows."""

        denominator = cls._active_sum(weights, active_mask).clamp_min(EPS)
        weighted_coords = weights.unsqueeze(-1) * coords.unsqueeze(-2)
        periodic_axes = cfg.periodic_dimensions()
        scale_x, scale_y = cfg.spatial_scale()
        lengths = coords.new_tensor([max(float(scale_x), EPS), max(float(scale_y), EPS)])
        if not periodic_axes:
            numerator = cls._active_sum(weighted_coords, active_mask)
            return numerator / denominator.unsqueeze(-1)

        normalized_weights = weights / denominator.unsqueeze(1)
        outputs = []
        for axis in range(2):
            coordinate = coords[..., axis]
            if axis in periodic_axes:
                angles = 2.0 * math.pi * coordinate / lengths[axis]
                sin_sum = cls._active_sum(
                    normalized_weights * torch.sin(angles).unsqueeze(-1),
                    active_mask,
                )
                cos_sum = cls._active_sum(
                    normalized_weights * torch.cos(angles).unsqueeze(-1),
                    active_mask,
                )
                mean = torch.remainder(
                    torch.atan2(sin_sum, cos_sum) / (2.0 * math.pi) * lengths[axis],
                    lengths[axis],
                )
            else:
                mean = cls._active_sum(weighted_coords[..., axis], active_mask) / denominator
            outputs.append(mean)
        return torch.stack(outputs, dim=-1)

    @classmethod
    def _active_weighted_scale(
        cls,
        coords: torch.Tensor,
        weights: torch.Tensor,
        centroids: torch.Tensor,
        active_mask: torch.Tensor,
        cfg: UnifiedForwardConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return module weighted variance and scale without padded rows."""

        delta = _relative_delta(centroids[:, None, :, :], coords[:, :, None, :], cfg)
        variance = cls._active_sum(weights.unsqueeze(-1) * delta.square(), active_mask)
        return variance, torch.sqrt(variance.clamp_min(EPS))

    @classmethod
    def _active_assignment_purity(
        cls,
        assignment: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Measure winner-owned assignment mass over active module rows."""

        winners = assignment.argmax(dim=-1)
        winner_mask = F.one_hot(winners, num_classes=assignment.shape[-1]).to(assignment.dtype)
        numerator = cls._active_sum(assignment * winner_mask, active_mask)
        denominator = cls._active_sum(assignment, active_mask)
        return numerator / denominator.clamp_min(EPS)

    @staticmethod
    def _mass_statistics(
        mass: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return minimum, p05, and mean mass with optional token masking."""

        active = (
            torch.ones_like(mass, dtype=torch.bool)
            if token_mask is None
            else token_mask.to(device=mass.device, dtype=torch.bool)
        )
        count = active.sum(dim=-1).clamp_min(1)
        masked = mass.masked_fill(~active, torch.finfo(mass.dtype).max)
        sorted_mass = masked.sort(dim=-1).values
        p05_index = torch.floor(0.05 * (count - 1).to(dtype=mass.dtype)).to(torch.long)
        p05 = sorted_mass.gather(-1, p05_index.unsqueeze(-1)).squeeze(-1)
        mean = (mass * active.to(dtype=mass.dtype)).sum(dim=-1) / count.to(dtype=mass.dtype)
        return masked.amin(dim=-1), p05, mean

    def _factor_step(
        self,
        residual: torch.Tensor,
        module_mask: torch.Tensor,
        valid_case: torch.Tensor,
        refinement_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform the algebraic non-negative factor refinement for one step."""

        dtype = residual.dtype
        valid = valid_case.to(dtype=dtype)
        module_mass = residual.sum(dim=(2, 3))
        a = self._normalize_module_factor(module_mass, module_mask) * valid[:, None]

        # The configured count is retained for compatibility with the Phase-1
        # field.  Every iteration remains reductions/products only.
        for _ in range(max(int(refinement_steps), 1)):
            b_mass = torch.einsum("bm,bmed->bed", a, residual).sum(dim=-1)
            b = self._normalize_dense_factor(b_mass) * valid[:, None]
            w_mass = torch.einsum("bm,be,bmed->bd", a, b, residual)
            w = self._normalize_dense_factor(w_mass) * valid[:, None]
            a_mass = torch.einsum("be,bd,bmed->bm", b, w, residual)
            a = self._normalize_module_factor(a_mass, module_mask) * valid[:, None]

        # Recompute b and w once from the refined module factor as required by
        # the tensor-residual formulation.
        b_mass = torch.einsum("bm,bmed->bed", a, residual).sum(dim=-1)
        b = self._normalize_dense_factor(b_mass) * valid[:, None]
        w_mass = torch.einsum("bm,be,bmed->bd", a, b, residual)
        w = self._normalize_dense_factor(w_mass) * valid[:, None]
        return a, b, w

    def forward(
        self,
        module_tokens: torch.Tensor,
        env_tokens: torch.Tensor,
        module_centers: torch.Tensor,
        env_coords: torch.Tensor,
        module_present: torch.Tensor,
        *,
        cfg: Optional[UnifiedForwardConfig] = None,
        global_token: Optional[torch.Tensor] = None,
        return_residual_interaction_tensor: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Build tensor mechanisms and the established organizer contract."""

        cfg = self.config if cfg is None else cfg
        self._validate_inputs(
            module_tokens,
            env_tokens,
            module_centers,
            env_coords,
            module_present,
            global_token,
        )

        batch_size, module_width, hidden_dim = module_tokens.shape
        if module_width <= 0:
            raise ValueError("case_adaptive_tensor_residual requires at least one module slot.")
        if env_tokens.ndim == 2:
            env_tokens = env_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        if env_tokens.shape[0] != batch_size:
            raise ValueError("env_tokens batch dimension must match module_tokens.")
        if env_tokens.shape[-1] != hidden_dim:
            raise ValueError("module_tokens and env_tokens must have the same hidden width.")
        if env_tokens.shape[1] <= 0:
            raise ValueError("case_adaptive_tensor_residual requires environment tokens.")

        device = module_tokens.device
        dtype = module_tokens.dtype
        module_centers = _as_batched_coords(
            module_centers.to(device=device, dtype=dtype),
            batch_size,
        )
        env_coords_b = _as_batched_coords(
            env_coords.to(device=device, dtype=dtype),
            batch_size,
        )
        if module_centers.shape[:2] != (batch_size, module_width):
            raise ValueError("module_centers must align with module_tokens as [B,M,2].")
        if env_coords_b.shape[:2] != env_tokens.shape[:2]:
            raise ValueError("env_coords must align with env_tokens as [B,E,2].")
        module_present = module_present.to(device=device, dtype=dtype)
        active_module_mask = module_present > 0
        active_module_count = active_module_mask.sum(dim=-1)
        if bool((active_module_count <= 0).any()):
            raise ValueError("case_adaptive_tensor_residual requires one active module per case.")

        if global_token is None:
            global_token = torch.zeros(batch_size, hidden_dim, device=device, dtype=dtype)
        else:
            global_token = global_token.to(device=device, dtype=dtype)
            if global_token.shape[-1] != hidden_dim:
                raise ValueError("global_token hidden width must match module_tokens.")

        # Center environment content before projection.  In particular, this
        # removes the common global vector that the model encoder may already
        # have added to every environment token.
        module_centered = F.layer_norm(module_tokens, (hidden_dim,))
        environment_centered = F.layer_norm(
            env_tokens - env_tokens.mean(dim=1, keepdim=True),
            (hidden_dim,),
        )
        module_content = self.module_content_proj(module_centered)
        environment_content = self.environment_content_proj(environment_centered)
        module_scale = 1.0 + torch.tanh(self.global_module_film(global_token))
        environment_scale = 1.0 + torch.tanh(self.global_environment_film(global_token))
        u = module_content * module_scale[:, None, :]
        v = environment_content * environment_scale[:, None, :]

        delta_me = _relative_delta(
            module_centers[:, :, None, :],
            env_coords_b[:, None, :, :],
            cfg,
        )
        geometry = self.geometry_encoder(self._geometry_features(delta_me, cfg))
        signed_interaction = u[:, :, None, :] * v[:, None, :, :] + geometry
        signed_interaction = signed_interaction - signed_interaction.mean(dim=2, keepdim=True)
        interaction_tensor = signed_interaction.square() * module_present[:, :, None, None]

        # Normalize by total active L1 interaction mass.  _active_sum keeps
        # appended inactive module storage from changing the reduction path.
        interaction_mass = self._active_sum(interaction_tensor, active_module_mask).sum(dim=(1, 2))
        # The denominator is reduced over all environment/content entries and
        # then broadcast over the complete tensor.
        residual = interaction_tensor / interaction_mass[:, None, None, None].clamp_min(EPS)

        interaction_environment_mass = interaction_tensor.sum(dim=-1)
        interaction_row_mass = interaction_environment_mass.sum(dim=-1)
        normalized_interaction_row_mass = interaction_row_mass / interaction_mass[:, None].clamp_min(EPS)
        A_me = interaction_environment_mass / interaction_environment_mass.sum(dim=-1, keepdim=True).clamp_min(EPS)
        A_me = A_me * active_module_mask[:, :, None].to(dtype=dtype)
        module_env_context = torch.einsum("bme,beh->bmh", A_me, env_tokens)
        module_tokens_for_hyper = (
            module_tokens + 0.25 * self.me_context_proj(module_env_context)
        ) * module_present[:, :, None]

        # A case-dependent cap is packed only to the largest cap in this
        # batch.  The candidate axis therefore stays independent of raw
        # trailing padding width.
        cap_multiplier = float(cfg.residual_mechanism_cap_multiplier)
        minimum_active_edges = max(int(cfg.minimum_active_edges), 1)
        k_min = torch.minimum(
            active_module_count,
            active_module_count.new_full(
                active_module_count.shape,
                minimum_active_edges,
            ),
        )
        case_cap = torch.ceil(active_module_count.to(dtype=torch.float32) * cap_multiplier).to(torch.long)
        case_cap = torch.maximum(case_cap, k_min.to(torch.long))
        max_possible_cap = max(int(math.ceil(cap_multiplier * module_width)), 1)
        all_slots = torch.arange(max_possible_cap, device=device, dtype=torch.long)
        packed_slots = torch.nonzero(all_slots < case_cap.max(), as_tuple=False).squeeze(-1)
        packed_count = int(packed_slots.shape[0])
        active_step_mask = packed_slots[None, :] < case_cap[:, None]

        module_value = self.module_value(module_tokens_for_hyper)
        environment_value = self.environment_value(env_tokens)
        # rho is a fraction of the *normalized* residual mass.  Using the raw
        # interaction mass here would make the stopping threshold depend on
        # arbitrary projection amplitude rather than on unexplained content.
        residual_l1_initial = self._active_sum(residual, active_module_mask).sum(
            dim=(1, 2)
        ).clamp_min(EPS)
        rho_previous = torch.ones(batch_size, device=device, dtype=dtype)
        rho_values = []
        marginal_values = []
        strength_values = []
        module_factor_values = []
        environment_factor_values = []
        content_factor_values = []
        extraction_mask_values = []
        case_finished = torch.zeros(batch_size, device=device, dtype=torch.bool)

        for step in range(packed_count):
            valid_case = active_step_mask[:, step] & ~case_finished
            extraction_mask_values.append(valid_case)
            a, b, w = self._factor_step(
                residual,
                active_module_mask,
                valid_case,
                int(cfg.residual_factor_refinement_steps),
            )
            pattern = a[:, :, None, None] * b[:, None, :, None] * w[:, None, None, :]
            numerator_by_module = (residual * pattern).sum(dim=(2, 3))
            denominator_by_module = pattern.square().sum(dim=(2, 3))
            numerator = self._active_sum(numerator_by_module.unsqueeze(-1), active_module_mask).squeeze(-1)
            denominator = self._active_sum(denominator_by_module.unsqueeze(-1), active_module_mask).squeeze(-1)
            strength = (numerator / denominator.clamp_min(EPS)).clamp_min(0.0)
            strength = strength * valid_case.to(dtype=dtype)

            updated_residual = F.relu(residual - strength[:, None, None, None] * pattern)
            residual = torch.where(valid_case[:, None, None, None], updated_residual, residual)
            residual_mass = self._active_sum(residual, active_module_mask).sum(dim=(1, 2))
            rho_next = residual_mass / residual_l1_initial
            rho_next = torch.where(valid_case, rho_next, rho_previous)
            marginal = (rho_previous - rho_next).clamp_min(0.0)

            rho_values.append(rho_next)
            marginal_values.append(marginal)
            strength_values.append(strength)
            module_factor_values.append(a)
            environment_factor_values.append(b)
            content_factor_values.append(w)
            rho_previous = rho_next

            if not self.training:
                # Evaluation does not need to extract candidates after a case
                # has stopped or exhausted its safety cap.  Keep the packed
                # output shape stable by padding those later candidates below.
                stop_now = (
                    valid_case
                    & ((step + 1) >= k_min)
                    & (rho_next <= float(cfg.residual_stop_fraction))
                )
                cap_now = valid_case & ((step + 1) >= case_cap)
                case_finished = case_finished | stop_now | cap_now | ~active_step_mask[:, step]
                if bool(case_finished.all()):
                    break

        computed_steps = len(rho_values)
        if computed_steps < packed_count:
            padding_steps = packed_count - computed_steps
            zero_strength = torch.zeros(batch_size, device=device, dtype=dtype)
            zero_module = torch.zeros(batch_size, module_width, device=device, dtype=dtype)
            zero_environment = torch.zeros(batch_size, env_tokens.shape[1], device=device, dtype=dtype)
            zero_content = torch.zeros(
                batch_size,
                int(cfg.residual_interaction_dim),
                device=device,
                dtype=dtype,
            )
            for _ in range(padding_steps):
                rho_values.append(rho_previous)
                marginal_values.append(zero_strength)
                strength_values.append(zero_strength)
                module_factor_values.append(zero_module)
                environment_factor_values.append(zero_environment)
                content_factor_values.append(zero_content)
                extraction_mask_values.append(torch.zeros(batch_size, device=device, dtype=torch.bool))

        residual_fraction_trace = torch.stack(
            [torch.ones(batch_size, device=device, dtype=dtype), *rho_values],
            dim=-1,
        )
        residual_marginal = torch.stack(marginal_values, dim=-1)
        residual_strength = torch.stack(strength_values, dim=-1)
        residual_module_factor = torch.stack(module_factor_values, dim=-1)
        residual_environment_factor = torch.stack(environment_factor_values, dim=-1)
        residual_content_factor = torch.stack(content_factor_values, dim=-1)
        extraction_mask = torch.stack(extraction_mask_values, dim=-1)
        residual_module_factor = residual_module_factor * extraction_mask[:, None, :].to(dtype=dtype)
        residual_environment_factor = residual_environment_factor * extraction_mask[:, None, :].to(dtype=dtype)
        residual_content_factor = residual_content_factor * extraction_mask[:, None, :].to(dtype=dtype)

        step_numbers = (packed_slots + 1).to(dtype=dtype).unsqueeze(0)
        previous_rho = residual_fraction_trace[:, :-1]
        hard_case_edge_mask = active_step_mask & (
            (step_numbers <= k_min[:, None])
            | (previous_rho > float(cfg.residual_stop_fraction))
        )
        hard_case_edge_mask = hard_case_edge_mask.to(torch.bool)
        stop_reached = (
            active_step_mask
            & (step_numbers >= k_min[:, None])
            & (residual_fraction_trace[:, 1:] <= float(cfg.residual_stop_fraction))
        ).any(dim=-1)
        cap_hit = ~stop_reached
        hard_case_edge_count = hard_case_edge_mask.sum(dim=-1).to(dtype=dtype)

        temperature = max(float(cfg.residual_soft_stop_temperature), EPS)
        edge_survival_soft = torch.where(
            step_numbers <= k_min[:, None],
            torch.ones_like(previous_rho),
            torch.sigmoid(
                (previous_rho - float(cfg.residual_stop_fraction)) / temperature
            ),
        )
        edge_survival_soft = edge_survival_soft * active_step_mask.to(dtype=dtype)
        edge_survival_soft = edge_survival_soft * extraction_mask.to(dtype=dtype)

        # Incidence is topology only: lambda/strength is deliberately absent
        # from these support weights.  The straight-through blend has exact
        # hard forward values in both modes while preserving a soft gradient.
        hard_A_mh, hard_module_mass = self._supported_factor_normalize(
            residual_module_factor,
            hard_case_edge_mask.to(dtype=dtype),
            active_module_mask,
        )
        soft_A_mh, soft_module_mass = self._supported_factor_normalize(
            residual_module_factor,
            edge_survival_soft,
            active_module_mask,
        )
        hard_A_eh, hard_environment_mass = self._supported_factor_normalize(
            residual_environment_factor,
            hard_case_edge_mask.to(dtype=dtype),
            None,
        )
        soft_A_eh, soft_environment_mass = self._supported_factor_normalize(
            residual_environment_factor,
            edge_survival_soft,
            None,
        )
        A_mh = soft_A_mh + (hard_A_mh - soft_A_mh).detach()
        A_eh = soft_A_eh + (hard_A_eh - soft_A_eh).detach()

        module_mass_raw = self._active_sum(A_mh, active_module_mask)
        environment_mass_raw = A_eh.sum(dim=1)
        hyper_module_mass = module_mass_raw / module_mass_raw.sum(dim=-1, keepdim=True).clamp_min(EPS)
        hyper_env_mass = environment_mass_raw / environment_mass_raw.sum(dim=-1, keepdim=True).clamp_min(EPS)
        source_weights = A_mh / module_mass_raw.unsqueeze(1).clamp_min(EPS)
        region_weights = A_eh / environment_mass_raw.unsqueeze(1).clamp_min(EPS)
        hyper_source_coords = self._active_weighted_coords(
            module_centers,
            source_weights,
            active_module_mask,
            cfg,
        )
        hyper_region_coords = _weighted_coords(env_coords_b, region_weights, cfg)
        hyper_source_variance, hyper_source_scale = self._active_weighted_scale(
            module_centers,
            source_weights,
            hyper_source_coords,
            active_module_mask,
            cfg,
        )
        hyper_region_variance, hyper_region_scale = _weighted_scale(
            env_coords_b,
            region_weights,
            hyper_region_coords,
            cfg,
        )

        candidate_module_mass_raw = self._active_sum(residual_module_factor, active_module_mask)
        candidate_environment_mass_raw = residual_environment_factor.sum(dim=1)
        candidate_module_mass_fraction = candidate_module_mass_raw / candidate_module_mass_raw.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(EPS)
        candidate_environment_mass_fraction = candidate_environment_mass_raw / candidate_environment_mass_raw.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(EPS)
        candidate_module_weights = residual_module_factor / candidate_module_mass_raw.unsqueeze(1).clamp_min(EPS)
        candidate_environment_weights = residual_environment_factor / candidate_environment_mass_raw.unsqueeze(1).clamp_min(EPS)
        candidate_source_coords = self._active_weighted_coords(
            module_centers,
            candidate_module_weights,
            active_module_mask,
            cfg,
        )
        _, candidate_source_scale = self._active_weighted_scale(
            module_centers,
            candidate_module_weights,
            candidate_source_coords,
            active_module_mask,
            cfg,
        )
        candidate_region_coords = _weighted_coords(env_coords_b, candidate_environment_weights, cfg)
        _, candidate_region_scale = _weighted_scale(
            env_coords_b,
            candidate_environment_weights,
            candidate_region_coords,
            cfg,
        )

        source_scales = module_centers.new_tensor(
            [max(float(cfg.spatial_scale()[0]), EPS), max(float(cfg.spatial_scale()[1]), EPS)]
        )
        module_summary = torch.einsum("bmk,bmh->bkh", residual_module_factor, module_value)
        environment_summary = torch.einsum("bek,beh->bkh", residual_environment_factor, environment_value)
        content_summary = self.content_to_state(residual_content_factor.permute(0, 2, 1))
        candidate_state_input = torch.cat(
            [
                module_summary,
                environment_summary,
                content_summary,
                candidate_source_coords / source_scales,
                candidate_region_coords / source_scales,
                residual_strength.unsqueeze(-1),
                previous_rho.unsqueeze(-1),
                residual_marginal.unsqueeze(-1),
                global_token[:, None, :].expand(-1, packed_count, -1),
            ],
            dim=-1,
        )
        candidate_hyper_state = self.mechanism_mixer(candidate_state_input)
        candidate_hyper_state = candidate_hyper_state * active_step_mask[:, :, None].to(dtype=dtype)
        hyper_state = candidate_hyper_state

        hyper_strength = torch.sqrt(hyper_module_mass * hyper_env_mass + EPS)
        hyper_strength = hyper_strength * hard_case_edge_mask.to(dtype=dtype)
        hyper_module_purity = self._active_assignment_purity(A_mh, active_module_mask)
        hyper_env_purity = _assignment_purity(A_eh)
        edge_quality = torch.sqrt(hyper_module_purity * hyper_env_purity)
        edge_quality = edge_quality * hard_case_edge_mask.to(dtype=dtype)
        mechanism_descriptor_features = _descriptor_first_features(
            hyper_source_coords,
            hyper_source_scale,
            hyper_region_coords,
            hyper_region_scale,
            hyper_module_mass,
            hyper_env_mass,
            hyper_module_purity,
            hyper_env_purity,
            hard_case_edge_mask.to(dtype=dtype),
            cfg,
        )
        (
            mechanism_geometry_features,
            mechanism_mass_features,
            mechanism_raw_features,
            hyper_source_region_distance,
            hyper_source_region_downstream,
            hyper_source_region_lateral,
        ) = _mechanism_descriptors(
            hyper_source_coords,
            hyper_region_coords,
            hyper_module_mass,
            hyper_env_mass,
            hyper_strength,
            module_mass_raw,
            environment_mass_raw,
            module_present,
            env_tokens.shape[1],
            cfg,
        )

        selected_module_probability_mass = (
            residual_module_factor * hard_case_edge_mask[:, None, :].to(dtype=dtype)
        ).sum(dim=-1)
        selected_environment_probability_mass = (
            residual_environment_factor * hard_case_edge_mask[:, None, :].to(dtype=dtype)
        ).sum(dim=-1)
        active_modules = active_module_mask
        module_coverage = (
            ((selected_module_probability_mass >= float(cfg.selection_token_threshold)) & active_modules)
            .sum(dim=-1)
            .to(dtype=dtype)
            / active_modules.sum(dim=-1).clamp_min(1).to(dtype=dtype)
        )
        environment_coverage = (
            selected_environment_probability_mass >= float(cfg.selection_token_threshold)
        ).to(dtype=dtype).mean(dim=-1)
        module_entry_count = active_modules.sum(dim=-1).clamp_min(1).to(dtype=dtype) * active_module_count.to(dtype)
        candidate_module_nonzero_fraction = (
            self._active_sum((residual_module_factor > 0).to(dtype=dtype), active_modules).sum(dim=-1)
            / module_entry_count
        )
        selected_module_nonzero_fraction = (
            self._active_sum((A_mh > 0).to(dtype=dtype), active_modules).sum(dim=-1)
            / module_entry_count
        )
        candidate_environment_nonzero_fraction = (residual_environment_factor > 0).to(dtype=dtype).mean(dim=(1, 2))
        selected_environment_nonzero_fraction = (A_eh > 0).to(dtype=dtype).mean(dim=(1, 2))
        module_mass_min, module_mass_p05, module_mass_mean = self._mass_statistics(
            selected_module_probability_mass,
            active_modules,
        )
        environment_mass_min, environment_mass_p05, environment_mass_mean = self._mass_statistics(
            selected_environment_probability_mass,
            None,
        )
        empty_selected = hard_case_edge_mask & (
            (module_mass_raw <= EPS) | (environment_mass_raw <= EPS)
        )
        monotonic_violation = (
            residual_fraction_trace[:, 1:] - residual_fraction_trace[:, :-1]
        ).clamp_min(0.0).amax(dim=-1)
        edge_support_gap = (hard_case_edge_mask.to(dtype=dtype) - edge_survival_soft).abs().mean(dim=-1)

        output: Dict[str, torch.Tensor] = {
            "A_mh": A_mh,
            "A_eh": A_eh,
            "A_mh_hard": hard_A_mh,
            "A_mh_soft": soft_A_mh,
            "A_eh_hard": hard_A_eh,
            "A_eh_soft": soft_A_eh,
            "candidate_A_mh": residual_module_factor,
            "candidate_A_eh": residual_environment_factor,
            "candidate_hyper_state": candidate_hyper_state,
            "hyper_state": hyper_state,
            "hyper_source_coords": hyper_source_coords,
            "hyper_region_coords": hyper_region_coords,
            "hyper_source_variance": hyper_source_variance,
            "hyper_source_scale": hyper_source_scale,
            "hyper_region_variance": hyper_region_variance,
            "hyper_region_scale": hyper_region_scale,
            "hyper_module_mass_raw": module_mass_raw,
            "hyper_env_mass_raw": environment_mass_raw,
            "hyper_module_mass": hyper_module_mass,
            "hyper_env_mass": hyper_env_mass,
            "hyper_module_purity": hyper_module_purity,
            "hyper_env_purity": hyper_env_purity,
            "candidate_module_mass_fraction": candidate_module_mass_fraction,
            "candidate_environment_mass_fraction": candidate_environment_mass_fraction,
            "candidate_module_purity": self._active_assignment_purity(
                residual_module_factor,
                active_module_mask,
            ),
            "candidate_environment_purity": _assignment_purity(residual_environment_factor),
            "candidate_source_coords": candidate_source_coords,
            "candidate_source_scale": candidate_source_scale,
            "candidate_region_coords": candidate_region_coords,
            "candidate_region_scale": candidate_region_scale,
            "hyper_strength": hyper_strength,
            "edge_quality": edge_quality,
            "edge_active_mask": hard_case_edge_mask.to(dtype=dtype).detach(),
            "hard_selected_edge_mask": hard_case_edge_mask.to(dtype=dtype).detach(),
            "edge_transition_gate": hard_case_edge_mask.to(dtype=dtype),
            "edge_viable_mask": active_step_mask.to(dtype=dtype),
            "effective_edge_mask": hard_case_edge_mask.to(dtype=dtype).detach(),
            "candidate_edge_viable_mask": active_step_mask.to(dtype=dtype),
            "candidate_extraction_mask": extraction_mask.to(dtype=dtype),
            "candidate_edge_count": module_tokens.new_full((batch_size,), float(packed_count)),
            "selected_edge_count": hard_case_edge_count,
            "viable_selected_edge_count": hard_case_edge_count,
            "hard_selected_edge_count": hard_case_edge_count,
            "active_edge_count": hard_case_edge_count,
            "edge_transition_gate_sum": hard_case_edge_mask.to(dtype=dtype).sum(dim=-1),
            "empty_selected_edge_count": empty_selected.sum(dim=-1).to(dtype=dtype),
            "selection_module_coverage": module_coverage,
            "selection_environment_coverage": environment_coverage,
            "candidate_module_nonzero_fraction": candidate_module_nonzero_fraction,
            "candidate_environment_nonzero_fraction": candidate_environment_nonzero_fraction,
            "selected_module_nonzero_fraction": selected_module_nonzero_fraction,
            "selected_environment_nonzero_fraction": selected_environment_nonzero_fraction,
            "selected_module_probability_mass_min": module_mass_min,
            "selected_module_probability_mass_p05": module_mass_p05,
            "selected_module_probability_mass_mean": module_mass_mean,
            "selected_environment_probability_mass_min": environment_mass_min,
            "selected_environment_probability_mass_p05": environment_mass_p05,
            "selected_environment_probability_mass_mean": environment_mass_mean,
            "pre_fallback_zero_support_module_rows": (
                active_modules & (hard_module_mass <= EPS)
            ).sum(dim=-1).to(dtype=dtype),
            "post_fallback_zero_support_module_rows": (
                active_modules & (A_mh.sum(dim=-1) <= EPS)
            ).sum(dim=-1).to(dtype=dtype),
            "pre_fallback_zero_support_environment_rows": (
                hard_environment_mass <= EPS
            ).sum(dim=-1).to(dtype=dtype),
            "post_fallback_zero_support_environment_rows": (
                A_eh.sum(dim=-1) <= EPS
            ).sum(dim=-1).to(dtype=dtype),
            "mechanism_geometry_features": mechanism_geometry_features,
            "mechanism_mass_features": mechanism_mass_features,
            "mechanism_raw_features": mechanism_raw_features,
            "mechanism_descriptor_features": mechanism_descriptor_features,
            "hyper_source_region_distance": hyper_source_region_distance,
            "hyper_source_region_downstream": hyper_source_region_downstream,
            "hyper_source_region_lateral": hyper_source_region_lateral,
            "module_tokens": module_tokens,
            "module_tokens_for_hyper": module_tokens_for_hyper,
            "env_tokens": env_tokens,
            "env_coords": env_coords_b,
            "module_centers": module_centers,
            "module_present": module_present,
            "A_me": A_me,
            "module_env_context": module_env_context,
            "selection_transition_fraction": module_tokens.new_zeros(()),
            "module_sparsity_fraction": module_tokens.new_zeros(()),
            "environment_sparsity_fraction": module_tokens.new_zeros(()),
            "query_sparsity_fraction": module_tokens.new_zeros(()),
            "training_progress_epoch": module_tokens.new_zeros(()),
            "routing_execution_gathered": module_tokens.new_tensor(float(cfg.routing_execution == "gathered")),
            "hyper_module_assignment_uniform": module_tokens.new_zeros(()),
            "residual_stop_fraction": module_tokens.new_tensor(float(cfg.residual_stop_fraction)),
            "residual_soft_stop_temperature": module_tokens.new_tensor(float(cfg.residual_soft_stop_temperature)),
            "residual_fraction_trace": residual_fraction_trace,
            "residual_marginal_explained_fraction": residual_marginal,
            "residual_mechanism_strength": residual_strength,
            "residual_module_factor": residual_module_factor,
            "residual_environment_factor": residual_environment_factor,
            "residual_content_factor": residual_content_factor,
            "residual_coupling_row_mass": normalized_interaction_row_mass,
            "hard_case_edge_mask": hard_case_edge_mask.to(dtype=dtype).detach(),
            "edge_survival_soft": edge_survival_soft,
            "case_adaptive_soft_edge_count": edge_survival_soft.sum(dim=-1),
            "case_adaptive_edge_count": hard_case_edge_count,
            "case_adaptive_edge_cap": case_cap.to(dtype=dtype),
            "case_adaptive_stop_reached": stop_reached.to(dtype=dtype),
            "case_adaptive_cap_hit": cap_hit.to(dtype=dtype),
            "case_adaptive_stop_margin": (
                float(cfg.residual_stop_fraction)
                - residual_fraction_trace.gather(
                    -1,
                    hard_case_edge_count.to(torch.long)
                    .clamp(min=1, max=max(packed_count, 1))
                    .unsqueeze(-1),
                ).squeeze(-1)
            ),
            "residual_monotonic_violation_max": monotonic_violation,
            "case_adaptive_support_gap": edge_support_gap,
        }
        if return_residual_interaction_tensor:
            output["residual_interaction_tensor"] = interaction_tensor
        return output


__all__ = ["CaseAdaptiveTensorResidualOrganizer"]
