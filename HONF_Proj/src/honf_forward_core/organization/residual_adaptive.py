"""Case-adaptive residual mechanism organizer.

This module contains the only organizer implementation whose runtime mechanism
axis is derived from the input case.  All extraction operations use shared
parameters; the module axis is padded to the largest active module count in a
batch solely so that the result can be represented by dense tensors.
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


class CaseAdaptiveResidualOrganizer(nn.Module):
    """Extract shared nonnegative rank-one mechanisms from a case residual.

    The coupling and factor networks have no parameter whose shape depends on
    the number of modules or on an inferred mechanism count.  A short Python
    loop over the packed module axis is intentional: residual subtraction is
    sequential, while all batch cases are processed by the same tensorized
    operations at each step.
    """

    def __init__(self, config: UnifiedForwardConfig):
        """Initialize shared residual-organizer projections and mixers."""

        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        geometry_frequencies = int(config.residual_coupling_fourier_frequencies)
        geometry_dim = 2 + 4 * geometry_frequencies

        self.global_to_module = nn.Linear(hidden_dim, hidden_dim)
        self.global_to_environment = nn.Linear(hidden_dim, hidden_dim)
        self.coupling_module_query = nn.Linear(hidden_dim, hidden_dim)
        self.coupling_environment_key = nn.Linear(hidden_dim, hidden_dim)
        self.coupling_geometry = nn.Sequential(
            nn.Linear(geometry_dim, max(hidden_dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 8), 1),
        )

        self.me_context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.module_value = nn.Linear(hidden_dim, hidden_dim)
        self.environment_value = nn.Linear(hidden_dim, hidden_dim)
        self.module_anchor = nn.Linear(2 * hidden_dim, 1)
        self.environment_query = nn.Linear(hidden_dim, hidden_dim)
        self.environment_key = nn.Linear(hidden_dim, hidden_dim)
        self.module_query = nn.Linear(hidden_dim, hidden_dim)
        self.module_key = nn.Linear(hidden_dim, hidden_dim)

        mechanism_input_dim = 3 * hidden_dim + 7
        mechanism_hidden_dim = max(hidden_dim, 32)
        self.mechanism_mixer = nn.Sequential(
            nn.LayerNorm(mechanism_input_dim) if config.use_layer_norm else nn.Identity(),
            nn.Linear(mechanism_input_dim, mechanism_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(mechanism_hidden_dim, hidden_dim),
        )

    @staticmethod
    def _validate_inputs(
        module_tokens: torch.Tensor,
        env_tokens: torch.Tensor,
        module_centers: torch.Tensor,
        env_coords: torch.Tensor,
        module_present: torch.Tensor,
        global_token: Optional[torch.Tensor],
    ) -> None:
        """Validate the dense generic organizer contract before arithmetic."""

        if module_tokens.ndim != 3:
            raise ValueError("module_tokens must have shape [B,M,H].")
        if env_tokens.ndim == 2:
            # A shared environment token table is expanded by ``forward``.
            pass
        elif env_tokens.ndim != 3:
            raise ValueError("env_tokens must have shape [B,E,H] or [E,H].")
        if module_centers.ndim not in {2, 3} or module_centers.shape[-1] != 2:
            raise ValueError("module_centers must have shape [B,M,2] or [M,2].")
        if env_coords.ndim not in {2, 3} or env_coords.shape[-1] != 2:
            raise ValueError("env_coords must have shape [B,E,2] or [E,2].")
        if module_present.ndim != 2:
            raise ValueError("module_present must have shape [B,M].")
        if module_tokens.shape[0] != module_present.shape[0] or module_tokens.shape[1] != module_present.shape[1]:
            raise ValueError("module_tokens and module_present must agree on [B,M].")
        if global_token is not None:
            if global_token.ndim != 2 or global_token.shape[0] != module_tokens.shape[0]:
                raise ValueError("global_token must have shape [B,H].")

    @staticmethod
    def _geometry_features(delta: torch.Tensor, cfg: UnifiedForwardConfig) -> torch.Tensor:
        """Encode normalized relative coordinates with shared Fourier features."""

        scale_x, scale_y = cfg.spatial_scale()
        scales = delta.new_tensor([max(float(scale_x), EPS), max(float(scale_y), EPS)])
        normalized = delta / scales
        pieces = [normalized]
        frequencies = int(cfg.residual_coupling_fourier_frequencies)
        for frequency in range(1, frequencies + 1):
            angle = 2.0 * math.pi * float(frequency) * normalized
            pieces.extend((torch.sin(angle), torch.cos(angle)))
        return torch.cat(pieces, dim=-1)

    @staticmethod
    def _supported_factor_normalize(
        base_factor: torch.Tensor,
        support: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply support before row normalization with a numerical fallback.

        The fallback is used only when a genuine zero denominator occurs.  It
        chooses the first active extracted mechanism, preserving exact zeros
        for padded tokens and hard-inactive mechanism columns.
        """

        support = support.to(device=base_factor.device, dtype=base_factor.dtype)
        weighted = base_factor * support[:, None, :]
        retained_mass = weighted.sum(dim=-1)
        active_tokens = (
            torch.ones_like(retained_mass, dtype=torch.bool)
            if token_mask is None
            else token_mask.to(device=base_factor.device, dtype=torch.bool)
        )
        # A merely small survival weight is still meaningful support and must
        # be renormalized proportionally.  Trigger the deterministic fallback
        # only when the weighted denominator is an actual floating-point zero
        # (including underflow), rather than at the model-scale EPS used by
        # ordinary geometry/statistical divisions.
        zero_rows = active_tokens & (retained_mass <= torch.finfo(base_factor.dtype).tiny)
        active_mechanisms = support > 0
        first_index = active_mechanisms.to(dtype=base_factor.dtype).argmax(dim=-1)
        first_hot = F.one_hot(first_index, num_classes=base_factor.shape[-1]).to(base_factor.dtype)
        fallback = base_factor * first_hot[:, None, :]
        weighted = weighted + fallback * zero_rows.unsqueeze(-1).to(base_factor.dtype)
        normalized = weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(base_factor.dtype).tiny
        )
        normalized = normalized * active_tokens.unsqueeze(-1).to(base_factor.dtype)
        return normalized, retained_mass

    @staticmethod
    def _active_sum(values: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """Sum a module-axis tensor without including packed padding rows.

        A reduction over ``M_pack`` can take a different floating-point path
        when an otherwise identical batch is padded to a larger module width.
        Gathering active rows before the case reduction keeps the residual
        extraction numerically invariant to that storage-only padding while
        remaining fully tensorized over cases.
        """

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
    def _active_softmax(
        cls,
        logits: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Softmax over active modules only, excluding packed padding rows."""

        if logits.ndim != 2:
            raise ValueError("active_softmax expects logits with shape [B,M].")
        active_mask = active_mask.to(device=logits.device, dtype=torch.bool)
        masked_logits = logits.masked_fill(~active_mask, torch.finfo(logits.dtype).min)
        maximum = masked_logits.amax(dim=-1, keepdim=True)
        unnormalized = torch.exp(masked_logits - maximum) * active_mask.to(logits.dtype)
        denominator = cls._active_sum(unnormalized, active_mask).unsqueeze(-1)
        return unnormalized / denominator.clamp_min(EPS)

    @classmethod
    def _active_weighted_coords(
        cls,
        coords: torch.Tensor,
        weights: torch.Tensor,
        active_mask: torch.Tensor,
        cfg: UnifiedForwardConfig,
    ) -> torch.Tensor:
        """Reduce module coordinates while excluding packed inactive rows."""

        if coords.ndim != 3 or coords.shape[-1] != 2 or weights.ndim not in {2, 3}:
            raise ValueError("active_weighted_coords expects coords [B,M,2] and weights [B,M,K?].")
        denominator = cls._active_sum(weights, active_mask).clamp_min(EPS)
        periodic_axes = cfg.periodic_dimensions()
        scale_x, scale_y = cfg.spatial_scale()
        lengths = coords.new_tensor([max(float(scale_x), EPS), max(float(scale_y), EPS)])
        if weights.ndim == 2:
            denominator_for_weights = denominator.unsqueeze(-1)
            weighted_coords = weights.unsqueeze(-1) * coords
        else:
            denominator_for_weights = denominator.unsqueeze(1)
            weighted_coords = weights.unsqueeze(-1) * coords.unsqueeze(-2)
        if not periodic_axes:
            numerator = cls._active_sum(weighted_coords, active_mask)
            return numerator / denominator.unsqueeze(-1)

        normalized_weights = weights / denominator_for_weights
        outputs = []
        for axis in range(2):
            coordinate = coords[..., axis]
            if weights.ndim == 2:
                coordinate_sin = torch.sin(2.0 * math.pi * coordinate / lengths[axis])
                coordinate_cos = torch.cos(2.0 * math.pi * coordinate / lengths[axis])
                sin_sum = cls._active_sum(normalized_weights * coordinate_sin, active_mask)
                cos_sum = cls._active_sum(normalized_weights * coordinate_cos, active_mask)
                periodic_mean = torch.remainder(
                    torch.atan2(sin_sum, cos_sum) / (2.0 * math.pi) * lengths[axis],
                    lengths[axis],
                )
            else:
                coordinate_sin = torch.sin(2.0 * math.pi * coordinate / lengths[axis]).unsqueeze(-1)
                coordinate_cos = torch.cos(2.0 * math.pi * coordinate / lengths[axis]).unsqueeze(-1)
                sin_sum = cls._active_sum(normalized_weights * coordinate_sin, active_mask)
                cos_sum = cls._active_sum(normalized_weights * coordinate_cos, active_mask)
                periodic_mean = torch.remainder(
                    torch.atan2(sin_sum, cos_sum) / (2.0 * math.pi) * lengths[axis],
                    lengths[axis],
                )
            if axis in periodic_axes:
                outputs.append(periodic_mean)
            else:
                outputs.append(
                    cls._active_sum(weighted_coords[..., axis], active_mask)
                    / denominator
                )
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
        """Return module weighted variance/scale without packed-row drift."""

        delta = _relative_delta(centroids[:, None, :, :], coords[:, :, None, :], cfg)
        variance = cls._active_sum(weights[..., None] * delta.square(), active_mask)
        return variance, torch.sqrt(variance.clamp_min(EPS))

    @classmethod
    def _active_assignment_purity(
        cls,
        assignment: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Measure assignment purity while excluding packed module rows."""

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
        """Return min, p05, and mean retained mass without looping over cases."""

        active = (
            torch.ones_like(mass, dtype=torch.bool)
            if token_mask is None
            else token_mask.to(device=mass.device, dtype=torch.bool)
        )
        count = active.sum(dim=-1).clamp_min(1)
        masked = mass.masked_fill(~active, torch.finfo(mass.dtype).max)
        sorted_mass = masked.sort(dim=-1).values
        p05_index = torch.floor(0.05 * (count - 1).to(dtype=mass.dtype)).to(torch.long)
        p05 = sorted_mass.gather(dim=-1, index=p05_index.unsqueeze(-1)).squeeze(-1)
        mean = (mass * active.to(dtype=mass.dtype)).sum(dim=-1) / count.to(dtype=mass.dtype)
        return masked.amin(dim=-1), p05, mean

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
    ) -> Dict[str, torch.Tensor]:
        """Build case-specific mechanisms and the standard organizer contract."""

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
            raise ValueError("case_adaptive_residual requires at least one module slot.")
        if env_tokens.ndim == 2:
            env_tokens = env_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        if env_tokens.shape[0] != batch_size:
            raise ValueError("env_tokens batch dimension must match module_tokens.")
        if env_tokens.shape[-1] != hidden_dim:
            raise ValueError("module_tokens and env_tokens must have the same hidden width.")
        if env_tokens.shape[1] <= 0:
            raise ValueError("case_adaptive_residual requires at least one environment token.")

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
        if env_coords_b.shape[0] != batch_size or env_coords_b.shape[1] != env_tokens.shape[1]:
            raise ValueError("env_coords must align with env_tokens as [B,E,2].")
        module_present = module_present.to(device=device, dtype=dtype)
        active_module_mask = module_present > 0
        active_module_count = active_module_mask.sum(dim=-1)
        if bool((active_module_count <= 0).any()):
            raise ValueError("case_adaptive_residual requires at least one active module per case.")

        if global_token is None:
            global_token = torch.zeros(batch_size, hidden_dim, device=device, dtype=dtype)
        else:
            global_token = global_token.to(device=device, dtype=dtype)
            if global_token.shape[0] != batch_size:
                raise ValueError("global_token batch dimension must match module_tokens.")
            if global_token.shape[-1] != hidden_dim:
                raise ValueError("global_token hidden width must match module_tokens.")

        conditioned_module = module_tokens + self.global_to_module(global_token).unsqueeze(1)
        conditioned_environment = env_tokens + self.global_to_environment(global_token).unsqueeze(1)
        conditioned_module = conditioned_module * module_present.unsqueeze(-1)

        delta_me = _relative_delta(
            module_centers[:, :, None, :],
            env_coords_b[:, None, :, :],
            cfg,
        )
        coupling_logits = torch.einsum(
            "bmh,beh->bme",
            self.coupling_module_query(conditioned_module),
            self.coupling_environment_key(conditioned_environment),
        ) / math.sqrt(float(hidden_dim))
        coupling_logits = coupling_logits + self.coupling_geometry(
            self._geometry_features(delta_me, cfg)
        ).squeeze(-1)
        coupling = F.softplus(coupling_logits) * module_present.unsqueeze(-1)
        coupling_norm_sq = self._active_sum(
            coupling.square().sum(dim=-1),
            active_module_mask,
        )
        coupling_norm = torch.sqrt(coupling_norm_sq).unsqueeze(-1).unsqueeze(-1)
        normalized_coupling = coupling / coupling_norm.clamp_min(EPS)
        coupling_row_mass = normalized_coupling.sum(dim=-1)

        A_me = normalized_coupling / coupling_row_mass.unsqueeze(-1).clamp_min(EPS)
        module_env_context = torch.einsum("bme,beh->bmh", A_me, env_tokens)
        # Global conditioning is used by the coupling/factor scores above;
        # retain the established module-organizer token contract here so that
        # the auxiliary A_me path remains numerically comparable to legacy
        # organizers.
        module_tokens_for_hyper = module_tokens + 0.25 * self.me_context_proj(module_env_context)
        module_tokens_for_hyper = module_tokens_for_hyper * module_present.unsqueeze(-1)

        # K_pack is the largest active module count in this batch.  The
        # collator normally provides exactly this width, but deriving it from
        # the mask keeps trailing inactive padding storage-only and prevents
        # the mechanism axis from changing when callers over-pad a batch.
        active_column_count = active_module_count.max()
        pack_indices = torch.arange(module_width, device=device)
        pack_indices = torch.nonzero(
            pack_indices < active_column_count,
            as_tuple=False,
        ).squeeze(-1)
        packed_modules = int(pack_indices.shape[0])
        active_step_mask = pack_indices[None, :] < active_module_count[:, None]
        k_min = torch.minimum(
            active_module_count,
            active_module_count.new_full(active_module_count.shape, float(max(int(cfg.minimum_active_edges), 1))),
        )

        residual = normalized_coupling
        initial_residual_norm_sq = self._active_sum(
            residual.square().sum(dim=-1),
            active_module_mask,
        ).clamp_min(EPS)
        rho_previous = torch.ones(batch_size, device=device, dtype=dtype)
        rho_values = []
        marginal_values = []
        strength_values = []
        module_factor_values = []
        environment_factor_values = []
        mechanism_state_values = []

        module_anchor_input = torch.cat(
            [module_tokens_for_hyper, global_token.unsqueeze(1).expand(-1, module_width, -1)],
            dim=-1,
        )
        anchor_score = self.module_anchor(module_anchor_input).squeeze(-1)
        module_value = self.module_value(module_tokens_for_hyper)
        environment_value = self.environment_value(env_tokens)
        source_scales = module_centers.new_tensor(
            [max(float(cfg.spatial_scale()[0]), EPS), max(float(cfg.spatial_scale()[1]), EPS)]
        )

        for step in range(packed_modules):
            valid_case = active_step_mask[:, step]
            valid_case_f = valid_case.to(dtype=dtype)
            row_mass = residual.sum(dim=-1)
            module_logits = anchor_score + torch.log(row_mass.clamp_min(EPS))
            a = self._active_softmax(module_logits, active_module_mask)
            a = a * valid_case_f.unsqueeze(-1)

            # Shared alternating refinement rounds.  The valid-case mask is
            # applied after each normalization so padded mechanisms are exact
            # zeros without introducing a detached count decision.
            for _ in range(int(cfg.residual_factor_refinement_steps)):
                source = self._active_weighted_coords(
                    module_centers,
                    a.unsqueeze(-1),
                    active_module_mask,
                    cfg,
                )[:, 0, :]
                env_support = torch.einsum("bme,bm->be", residual, a)
                module_summary = self._active_sum(
                    a.unsqueeze(-1) * module_value,
                    active_module_mask,
                )
                environment_logits = torch.einsum(
                    "bh,beh->be",
                    self.environment_query(module_summary),
                    self.environment_key(conditioned_environment),
                ) / math.sqrt(float(hidden_dim))
                environment_logits = environment_logits + torch.log(env_support.clamp_min(EPS))
                environment_logits = environment_logits + self.coupling_geometry(
                    self._geometry_features(
                        _relative_delta(source[:, None, :], env_coords_b, cfg),
                        cfg,
                    )
                ).squeeze(-1)
                b = F.softmax(environment_logits, dim=-1)
                b = b * valid_case_f.unsqueeze(-1)

                environment_summary = torch.einsum("be,beh->bh", b, environment_value)
                module_support = torch.einsum("bme,be->bm", residual, b)
                refined_module_logits = torch.einsum(
                    "bh,bmh->bm",
                    self.module_query(environment_summary),
                    self.module_key(module_tokens_for_hyper),
                ) / math.sqrt(float(hidden_dim))
                refined_module_logits = refined_module_logits + torch.log(module_support.clamp_min(EPS))
                refined_module_logits = refined_module_logits.masked_fill(
                    ~active_module_mask,
                    torch.finfo(dtype).min,
                )
                a = self._active_softmax(refined_module_logits, active_module_mask)
                a = a * valid_case_f.unsqueeze(-1)

            source = self._active_weighted_coords(
                module_centers,
                a.unsqueeze(-1),
                active_module_mask,
                cfg,
            )[:, 0, :]
            region = _weighted_coords(env_coords_b, b.unsqueeze(-1), cfg)[:, 0, :]
            module_summary = self._active_sum(
                a.unsqueeze(-1) * module_value,
                active_module_mask,
            )
            environment_summary = torch.einsum("be,beh->bh", b, environment_value)
            pattern = a.unsqueeze(-1) * b.unsqueeze(1)
            pattern_norm_sq = self._active_sum(
                pattern.square().sum(dim=-1),
                active_module_mask,
            ).clamp_min(EPS)
            explained = self._active_sum(
                (residual * pattern).sum(dim=-1),
                active_module_mask,
            )
            strength = explained / pattern_norm_sq
            strength = strength.clamp_min(0.0) * valid_case_f
            updated_residual = F.relu(residual - strength[:, None, None] * pattern)
            residual = torch.where(valid_case[:, None, None], updated_residual, residual)
            rho_next = self._active_sum(
                residual.square().sum(dim=-1),
                active_module_mask,
            ) / initial_residual_norm_sq
            rho_next = torch.where(valid_case, rho_next, rho_previous)
            marginal = (rho_previous - rho_next).clamp_min(0.0)

            state_input = torch.cat(
                [
                    module_summary,
                    environment_summary,
                    source / source_scales,
                    region / source_scales,
                    strength.unsqueeze(-1),
                    rho_previous.unsqueeze(-1),
                    marginal.unsqueeze(-1),
                    global_token,
                ],
                dim=-1,
            )
            mechanism_state = self.mechanism_mixer(state_input)

            rho_values.append(rho_next)
            marginal_values.append(marginal)
            strength_values.append(strength)
            module_factor_values.append(a)
            environment_factor_values.append(b)
            mechanism_state_values.append(mechanism_state)
            rho_previous = rho_next

        residual_fraction_trace = torch.stack(
            [torch.ones(batch_size, device=device, dtype=dtype), *rho_values],
            dim=-1,
        )
        residual_marginal = torch.stack(marginal_values, dim=-1)
        residual_strength = torch.stack(strength_values, dim=-1)
        residual_module_factor = torch.stack(module_factor_values, dim=-1)
        residual_environment_factor = torch.stack(environment_factor_values, dim=-1)
        candidate_state = torch.stack(mechanism_state_values, dim=1)
        candidate_state = candidate_state * active_step_mask.unsqueeze(-1).to(dtype)

        step_numbers = (pack_indices + 1).to(dtype=dtype).unsqueeze(0)
        hard_case_edge_mask = active_step_mask & (
            (step_numbers <= k_min[:, None])
            | (residual_fraction_trace[:, :-1] > float(cfg.residual_stop_fraction))
        )
        stop_reached = (
            active_step_mask
            & (step_numbers >= k_min[:, None])
            & (residual_fraction_trace[:, 1:] <= float(cfg.residual_stop_fraction))
        ).any(dim=-1)
        cap_hit = ~stop_reached
        case_edge_count = hard_case_edge_mask.sum(dim=-1).to(dtype=dtype)

        temperature = max(float(cfg.residual_soft_stop_temperature), EPS)
        soft_survival = torch.where(
            step_numbers <= k_min[:, None],
            torch.ones_like(residual_fraction_trace[:, :-1]),
            torch.sigmoid(
                (residual_fraction_trace[:, :-1] - float(cfg.residual_stop_fraction)) / temperature
            ),
        )
        soft_survival = soft_survival * active_step_mask.to(dtype)
        effective_edge_mask = soft_survival if self.training else hard_case_edge_mask.to(dtype)
        effective_edge_mask = effective_edge_mask * active_step_mask.to(dtype)

        effective_factor_support = effective_edge_mask * residual_strength
        A_mh, selected_module_probability_mass = self._supported_factor_normalize(
            residual_module_factor,
            effective_factor_support,
            active_module_mask,
        )
        A_eh, selected_environment_probability_mass = self._supported_factor_normalize(
            residual_environment_factor,
            effective_factor_support,
            None,
        )

        module_mass_raw = self._active_sum(A_mh, active_module_mask)
        env_mass_raw = A_eh.sum(dim=1)
        hyper_module_mass = module_mass_raw / module_mass_raw.sum(dim=-1, keepdim=True).clamp_min(EPS)
        hyper_env_mass = env_mass_raw / env_mass_raw.sum(dim=-1, keepdim=True).clamp_min(EPS)
        source_weights = A_mh / module_mass_raw.unsqueeze(1).clamp_min(EPS)
        region_weights = A_eh / env_mass_raw.unsqueeze(1).clamp_min(EPS)
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

        candidate_module_mass_raw = self._active_sum(
            residual_module_factor,
            active_module_mask,
        )
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

        hyper_strength = torch.sqrt(hyper_module_mass * hyper_env_mass + EPS) * effective_edge_mask
        hyper_module_purity = self._active_assignment_purity(A_mh, active_module_mask)
        hyper_env_purity = _assignment_purity(A_eh)
        edge_quality = torch.sqrt(hyper_module_purity * hyper_env_purity) * effective_edge_mask
        mechanism_descriptor_features = _descriptor_first_features(
            hyper_source_coords,
            hyper_source_scale,
            hyper_region_coords,
            hyper_region_scale,
            hyper_module_mass,
            hyper_env_mass,
            hyper_module_purity,
            hyper_env_purity,
            effective_edge_mask,
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
            env_mass_raw,
            module_present,
            env_tokens.shape[1],
            cfg,
        )

        hard_mask = hard_case_edge_mask.to(dtype=dtype)
        selected_module_probability_mass = (
            residual_module_factor * hard_mask[:, None, :]
        ).sum(dim=-1)
        selected_environment_probability_mass = (
            residual_environment_factor * hard_mask[:, None, :]
        ).sum(dim=-1)
        active_modules = active_module_mask
        module_coverage = (
            ((selected_module_probability_mass >= float(cfg.selection_token_threshold)) & active_modules).sum(dim=-1).to(dtype)
            / active_modules.sum(dim=-1).clamp_min(1).to(dtype)
        )
        env_coverage = (selected_environment_probability_mass >= float(cfg.selection_token_threshold)).to(dtype).mean(dim=-1)
        # Fractions are defined over each case's viable module-by-candidate
        # rectangle, not over the batch storage width.  This keeps the scalar
        # diagnostics invariant when trailing inactive packing is added.
        module_entry_count = active_modules.sum(dim=-1).clamp_min(1) * active_module_count.to(dtype)
        candidate_module_nonzero_fraction = (
            self._active_sum(
                (residual_module_factor > 0).to(dtype),
                active_modules,
            ).sum(dim=-1)
            / module_entry_count
        )
        selected_module_nonzero_fraction = (
            self._active_sum(
                (A_mh > 0).to(dtype),
                active_modules,
            ).sum(dim=-1)
            / module_entry_count
        )
        candidate_environment_nonzero_fraction = (residual_environment_factor > 0).to(dtype).mean(dim=(1, 2))
        selected_environment_nonzero_fraction = (A_eh > 0).to(dtype).mean(dim=(1, 2))
        module_mass_min, module_mass_p05, module_mass_mean = self._mass_statistics(
            selected_module_probability_mass,
            active_modules,
        )
        env_mass_min, env_mass_p05, env_mass_mean = self._mass_statistics(
            selected_environment_probability_mass,
            None,
        )
        empty_selected = hard_case_edge_mask & (
            (module_mass_raw <= EPS) | (env_mass_raw <= EPS)
        )
        monotonic_violation = (
            residual_fraction_trace[:, 1:] - residual_fraction_trace[:, :-1]
        ).clamp_min(0.0).amax(dim=-1)

        # ``hyper_state`` is support-independent content.  Only packed steps
        # beyond a case's module cap are zeroed; support is applied once in
        # incidence/query routing, as required by the decoder contract.
        hyper_state = candidate_state * active_step_mask.unsqueeze(-1).to(dtype)
        edge_active_mask = hard_case_edge_mask.to(dtype).detach()
        output: Dict[str, torch.Tensor] = {
            "A_mh": A_mh,
            "A_eh": A_eh,
            "candidate_A_mh": residual_module_factor,
            "candidate_A_eh": residual_environment_factor,
            "hyper_state": hyper_state,
            "candidate_hyper_state": candidate_state,
            "hyper_source_coords": hyper_source_coords,
            "hyper_region_coords": hyper_region_coords,
            "hyper_source_variance": hyper_source_variance,
            "hyper_source_scale": hyper_source_scale,
            "hyper_region_variance": hyper_region_variance,
            "hyper_region_scale": hyper_region_scale,
            "hyper_module_mass_raw": module_mass_raw,
            "hyper_env_mass_raw": env_mass_raw,
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
            "edge_active_mask": edge_active_mask,
            "hard_selected_edge_mask": hard_case_edge_mask.to(dtype).detach(),
            "edge_transition_gate": effective_edge_mask,
            "edge_viable_mask": active_step_mask.to(dtype),
            "effective_edge_mask": effective_edge_mask,
            "candidate_edge_viable_mask": active_step_mask.to(dtype),
            # Compatibility count describes the dense packed candidate axis;
            # the scientifically meaningful per-case count is exported as
            # ``case_adaptive_edge_count`` below.
            "candidate_edge_count": module_tokens.new_full((batch_size,), float(packed_modules)),
            "selected_edge_count": case_edge_count,
            "viable_selected_edge_count": case_edge_count,
            "hard_selected_edge_count": case_edge_count,
            "edge_transition_gate_sum": effective_edge_mask.sum(dim=-1),
            "empty_selected_edge_count": empty_selected.sum(dim=-1).to(dtype),
            "active_edge_count": case_edge_count,
            "selection_module_coverage": module_coverage,
            "selection_environment_coverage": env_coverage,
            "candidate_module_nonzero_fraction": candidate_module_nonzero_fraction,
            "candidate_environment_nonzero_fraction": candidate_environment_nonzero_fraction,
            "selected_module_nonzero_fraction": selected_module_nonzero_fraction,
            "selected_environment_nonzero_fraction": selected_environment_nonzero_fraction,
            "selected_module_probability_mass_min": module_mass_min,
            "selected_module_probability_mass_p05": module_mass_p05,
            "selected_module_probability_mass_mean": module_mass_mean,
            "selected_environment_probability_mass_min": env_mass_min,
            "selected_environment_probability_mass_p05": env_mass_p05,
            "selected_environment_probability_mass_mean": env_mass_mean,
            "pre_fallback_zero_support_module_rows": (active_modules & (selected_module_probability_mass <= EPS)).sum(dim=-1).to(dtype),
            "post_fallback_zero_support_module_rows": (active_modules & (A_mh.sum(dim=-1) <= EPS)).sum(dim=-1).to(dtype),
            "pre_fallback_zero_support_environment_rows": (selected_environment_probability_mass <= EPS).sum(dim=-1).to(dtype),
            "post_fallback_zero_support_environment_rows": (A_eh.sum(dim=-1) <= EPS).sum(dim=-1).to(dtype),
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
            "residual_coupling_row_mass": coupling_row_mass,
            "edge_survival_weight": effective_edge_mask,
            "case_adaptive_soft_edge_count": soft_survival.sum(dim=-1),
            "hard_case_edge_mask": hard_case_edge_mask.to(dtype).detach(),
            "case_adaptive_edge_count": case_edge_count,
            "case_adaptive_edge_cap": active_module_count.to(dtype),
            "case_adaptive_stop_reached": stop_reached.to(dtype),
            "case_adaptive_cap_hit": cap_hit.to(dtype),
            "case_adaptive_stop_margin": (
                float(cfg.residual_stop_fraction) - residual_fraction_trace.gather(
                    -1,
                    case_edge_count.to(torch.long).clamp(min=1, max=packed_modules).unsqueeze(-1),
                ).squeeze(-1)
            ),
            "residual_monotonic_violation_max": monotonic_violation,
        }
        return output
