"""Exact source-conditioned specialization of the Run-1501 fine reader.

The source organizer and physical MM/ME/EM preparation are inherited from
Run 1501.  This backend specializes the fine QM/QE reads to the all-closed
operator: it prepares source-local controls once, removes query-to-group
routing, and retains the original physical query/source functions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .group_control_router import PreparedGroupControl
from .sparse_incidence_group_control import (
    SparseIncidenceGroupControlPairwiseField,
)
from .sparse_incidence_router import SparseIncidenceGroupRouter
from .types import EncodedInterfaceCase


@dataclass(frozen=True)
class SourceConditionedPreparedControl(PreparedGroupControl):
    """Source-side Run-1501 state without query routing or final centres."""

    phase_occupied: torch.Tensor
    k_active: torch.Tensor
    pi: torch.Tensor
    kappa: torch.Tensor
    proposal_module_mass: torch.Tensor
    proposal_environment_mass: torch.Tensor
    proposal_occupied: torch.Tensor


class SourceConditionedRouter(SparseIncidenceGroupRouter):
    """Run-1501 source preparation with query-only modules omitted."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # These projections exist only to form query-to-prototype logits. The
        # static source-conditioned operator has no such path or parameters.
        del self.query_fourier
        del self.query_projection
        del self.query_group_projection

    def _source_controls_and_measures(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # The established router requires positive environment quadrature
        # weights. Preserve its exact path for ordinary cases, and define the
        # zero-source extension explicitly so an empty type has zero measure.
        if int(environment_states.shape[1]) > 0:
            return super()._source_controls_and_measures(
                encoded, module_states, environment_states
            )

        if module_states.ndim != 3 or environment_states.ndim != 3:
            raise ValueError("module_states and environment_states must have shape [B,S,H].")
        batch = int(module_states.shape[0])
        global_control = self.global_projection(self.global_norm(encoded.global_token))
        module_control = self._source_control(
            module_states,
            encoded.module_centers,
            global_control,
            self.module_projection,
            self.module_source_norm,
            self.module_global_projection,
            self.module_position_projection,
            encoded,
        )
        active_modules = encoded.module_present > 0.5
        module_count = active_modules.sum(dim=1, keepdim=True)
        module_measure = active_modules.to(module_states.dtype) / module_count.clamp_min(1).to(
            module_states.dtype
        )
        return (
            global_control,
            module_control,
            module_states.new_zeros(batch, 0, self.control_dim),
            active_modules,
            module_measure,
            module_states.new_zeros(batch, 0),
        )

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
    ) -> SourceConditionedPreparedControl:
        """Prepare the same assignments and controls without final centres."""

        (
            global_control,
            module_control,
            environment_control,
            active_modules,
            module_measure,
            environment_measure,
        ) = self._source_controls_and_measures(
            encoded, module_states, environment_states
        )
        batch = int(module_states.shape[0])
        codes = self.group_codes.to(device=module_states.device, dtype=module_states.dtype)
        valid = torch.ones(
            (batch, self.group_count),
            device=module_states.device,
            dtype=torch.bool,
        )
        (
            module_membership,
            environment_membership,
            proposal_module_mass,
            proposal_environment_mass,
            _proposal_joint_centres,
            proposal_occupied,
        ) = self._build_assignments(
            encoded,
            module_control,
            environment_control,
            active_modules,
            module_measure,
            environment_measure,
            codes,
            valid,
        )

        module_mass = torch.einsum(
            "bs,bsk->bk", module_measure, module_membership
        )
        environment_mass = torch.einsum(
            "bs,bsk->bk", environment_measure, environment_membership
        )
        phase_occupied = (module_mass + environment_mass) > 0.0
        if bool((phase_occupied.sum(dim=-1) < 1).any()):
            raise RuntimeError("source-conditioned preparation produced an empty phase.")
        k_active = phase_occupied.sum(dim=-1).to(dtype=module_states.dtype)
        # Keep the inherited proposal calibration: it is not recomputed from
        # the static reader's logical rank-one query access.
        pi = 0.5 * (module_mass + environment_mass)
        kappa = pi.square().sum(dim=-1).clamp_min(torch.finfo(pi.dtype).eps).reciprocal()
        module_moment = torch.einsum(
            "bs,bsk,bsd->bkd",
            module_measure,
            module_membership,
            module_control,
        ) * kappa[:, None, None]
        environment_moment = torch.einsum(
            "bs,bsk,bsd->bkd",
            environment_measure,
            environment_membership,
            environment_control,
        ) * kappa[:, None, None]
        group_input = torch.cat(
            [
                module_moment,
                environment_moment,
                kappa[:, None, None] * module_mass[..., None],
                kappa[:, None, None] * environment_mass[..., None],
                global_control[:, None, :].expand(-1, self.group_count, -1),
                codes[None, :, :].expand(batch, -1, -1),
            ],
            dim=-1,
        )
        group_control = torch.tanh(self.group_control(group_input))
        group_control = group_control * phase_occupied[..., None].to(group_control.dtype)
        return SourceConditionedPreparedControl(
            module_membership=module_membership,
            environment_membership=environment_membership,
            module_measure=module_measure,
            environment_measure=environment_measure,
            module_mass=module_mass,
            environment_mass=environment_mass,
            module_control=module_control,
            environment_control=environment_control,
            group_control=group_control,
            global_control=global_control,
            phase_occupied=phase_occupied,
            k_active=k_active,
            pi=pi,
            kappa=kappa,
            proposal_module_mass=proposal_module_mass,
            proposal_environment_mass=proposal_environment_mass,
            proposal_occupied=proposal_occupied,
        )

    def route_queries(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("source_conditioned_pairwise_honf has no query-to-group route.")


class SourceConditionedPairwiseField(SparseIncidenceGroupControlPairwiseField):
    """Run-1501 source preparation with an exact all-closed fine reader."""

    router_class = SourceConditionedRouter
    phase_interaction_diagnostics = False
    phase_diagnostic_prefix = "source_conditioned_"
    executor_policy = "rectangular_reference"
    diagnostic_executor_independent = True

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> dict[str, Any]:
        """Prepare source controls and physical banks for one live phase."""

        if not isinstance(self.router, SourceConditionedRouter):
            raise TypeError("source-conditioned backend requires its source-only router.")
        dense_encoded = replace(
            encoded,
            coordinate_scale=self._dense_coordinate_scale(
                encoded, int(module_states.shape[0])
            ),
        )
        fine = self.prepare_fine_messages(dense_encoded, module_states)
        global_env = encoded.global_token[:, None, :].expand(
            -1, fine["env_tokens"].shape[1], -1
        )
        contextual_env = fine["env_tokens"] + self.env_update(
            torch.cat(
                [fine["env_tokens"], fine["environment_messages"], global_env],
                dim=-1,
            )
        )
        controls = self.router.prepare(encoded, fine["module_tokens"], contextual_env)

        k_active = controls.k_active.clamp_min(1.0)[:, None, None]
        module_u = torch.einsum(
            "bmk,bkd->bmd", controls.module_membership, controls.group_control
        )
        environment_u = torch.einsum(
            "bek,bkd->bed", controls.environment_membership, controls.group_control
        )
        module_rho = controls.module_membership.sum(dim=-1) / k_active[..., 0]
        environment_rho = controls.environment_membership.sum(dim=-1) / k_active[..., 0]
        module_n = module_u / k_active
        environment_n = environment_u / k_active

        module_first_affine = self._prepare_module_affine(encoded, fine["module_tokens"])
        module_gain = 1.0 + torch.tanh(self.module_control_gain(module_n))

        environment_key, raw_environment_value = self.env_attention.project_source(
            contextual_env
        )
        environment_value_gain = 1.0 + torch.tanh(
            self.environment_value_control(environment_u)
        )
        value_gain = environment_value_gain.reshape(
            contextual_env.shape[0],
            contextual_env.shape[1],
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        environment_value = raw_environment_value * value_gain
        environment_score_gain = 1.0 + torch.tanh(
            self.environment_score_control(environment_n)
        )
        folded_environment_key = environment_key * environment_score_gain.permute(
            0, 2, 1
        )[..., None]

        module_sigma = (controls.module_measure * module_rho).sum(dim=-1)
        environment_sigma = (controls.environment_measure * environment_rho).sum(dim=-1)
        state: dict[str, Any] = {
            "module_tokens": fine["module_tokens"],
            "env_tokens": contextual_env,
            "source_conditioned_controls": controls,
            "module_first_affine": module_first_affine,
            "module_source_u": module_u,
            "module_source_rho": module_rho,
            "module_source_n": module_n,
            "module_source_gain": module_gain,
            "environment_source_u": environment_u,
            "environment_source_rho": environment_rho,
            "environment_source_n": environment_n,
            "environment_source_score_gain": environment_score_gain,
            "environment_source_value_gain": environment_value_gain,
            "environment_keys_unfolded": environment_key,
            "environment_keys": folded_environment_key,
            "environment_values": environment_value,
            "source_conditioned_module_sigma": module_sigma,
            "source_conditioned_environment_sigma": environment_sigma,
        }
        # ``return_routing_maps`` is accepted for the common core contract;
        # source maps are attached by preparation_aux, not rebuilt here.
        del return_routing_maps
        return state

    def preparation_aux(
        self,
        state: dict[str, Any],
        *,
        include_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        controls = state["source_conditioned_controls"]
        module_source_valid = controls.module_measure > 0.0
        environment_source_valid = controls.environment_measure > 0.0
        summary = {
            "source_conditioned_registered_capacity": controls.module_membership.new_full(
                (int(controls.module_membership.shape[0]),), float(self.group_count)
            ).detach(),
            "source_conditioned_active_proposal_count": controls.k_active.detach(),
            "source_conditioned_kappa": controls.kappa.detach(),
            "source_conditioned_pi": controls.pi.detach(),
            "source_conditioned_module_mass": controls.module_mass.detach(),
            "source_conditioned_environment_mass": controls.environment_mass.detach(),
            "source_conditioned_module_sigma": state[
                "source_conditioned_module_sigma"
            ].detach(),
            "source_conditioned_environment_sigma": state[
                "source_conditioned_environment_sigma"
            ].detach(),
            "source_conditioned_module_source_count": module_source_valid.sum(
                dim=-1
            ).to(controls.module_measure.dtype).detach(),
            "source_conditioned_environment_source_count": environment_source_valid.sum(
                dim=-1
            ).to(controls.environment_measure.dtype).detach(),
        }
        if include_diagnostics:
            summary.update(
                {
                    "source_conditioned_module_incidence": controls.module_membership.detach(),
                    "source_conditioned_environment_incidence": controls.environment_membership.detach(),
                    "source_conditioned_group_control": controls.group_control.detach(),
                    "source_conditioned_module_u": state["module_source_u"].detach(),
                    "source_conditioned_module_r": state["module_source_rho"].detach(),
                    "source_conditioned_module_n": state["module_source_n"].detach(),
                    "source_conditioned_environment_u": state[
                        "environment_source_u"
                    ].detach(),
                    "source_conditioned_environment_r": state[
                        "environment_source_rho"
                    ].detach(),
                    "source_conditioned_environment_n": state[
                        "environment_source_n"
                    ].detach(),
                    "source_conditioned_environment_score_gain": state[
                        "environment_source_score_gain"
                    ].detach(),
                    "source_conditioned_environment_value_gain": state[
                        "environment_source_value_gain"
                    ].detach(),
                }
            )
        return summary

    def _module_source_psi(
        self,
        source_affine: torch.Tensor,
        relative_features: torch.Tensor,
        source_gain: torch.Tensor,
    ) -> torch.Tensor:
        relative_weight, _ = self._module_first_slices()
        hidden = F.gelu(source_affine + F.linear(relative_features, relative_weight, None))
        hidden = hidden * source_gain
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self._module_tail, hidden, use_reentrant=False)
        return self._module_tail(hidden)

    def _read_module_source_conditioned(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_count, _ = receivers.shape
        module_count = int(encoded.module_centers.shape[1])
        if module_count == 0:
            return receivers.new_zeros(batch, query_count, self.hidden_dim)
        scales = self._dense_coordinate_scale(encoded, batch)
        relative = (
            receivers[:, :, None, :] - encoded.module_centers[:, None, :, :]
        ) / scales
        psi = self._module_source_psi(
            state["module_first_affine"][:, None, :, :],
            self.relative_fourier(relative),
            state["module_source_gain"][:, None, :, :],
        )
        controls: SourceConditionedPreparedControl = state[
            "source_conditioned_controls"
        ]
        source_weights = controls.module_measure * state["module_source_rho"]
        weighted_sum = (
            psi * source_weights[:, None, :, None]
        ).sum(dim=2)
        output = self.query_module_output
        linear = F.linear(weighted_sum, output.weight, bias=None)
        active_count = (encoded.module_present > 0.5).sum(dim=-1).to(receivers.dtype)
        active_scale = active_count / (1.0 + active_count)
        sigma = state["source_conditioned_module_sigma"]
        kappa = controls.kappa.to(device=receivers.device, dtype=receivers.dtype)
        return kappa[:, None, None] * (
            active_scale[:, None, None] * linear
            + sigma[:, None, None] * output.bias[None, None, :]
        )

    def _environment_geometry(
        self,
        receivers: torch.Tensor,
        encoded: EncodedInterfaceCase,
    ) -> torch.Tensor:
        scale = encoded.coordinate_scale
        if scale.ndim == 1:
            scale = scale[None, None, None, :]
        elif scale.ndim == 2:
            scale = scale[:, None, None, :]
        elif scale.ndim == 3:
            scale = scale[:, :, None, :]
        relative = (receivers[:, :, None, :] - encoded.env_coords[:, None, :, :]) / scale
        return self._mlp(self.env_geometry_bias, self.relative_fourier(relative)).permute(
            0, 3, 1, 2
        )

    def _read_environment_source_conditioned(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_count, _ = receivers.shape
        environment_count = int(encoded.env_coords.shape[1])
        controls: SourceConditionedPreparedControl = state[
            "source_conditioned_controls"
        ]
        sigma = state["source_conditioned_environment_sigma"]
        kappa = controls.kappa.to(device=receivers.device, dtype=receivers.dtype)
        if environment_count == 0:
            return receivers.new_zeros(batch, query_count, self.hidden_dim)

        query = self.env_attention.project_query(self.env_query(receiver_features))
        keys = state["environment_keys"]
        values = state["environment_values"]
        scores = torch.matmul(query, keys.transpose(-1, -2)) / (float(self.head_dim) ** 0.5)
        scores = scores + self._environment_geometry(receivers, encoded)
        source_prior = controls.environment_measure * state["environment_source_rho"]
        tiny = torch.finfo(scores.dtype).tiny
        valid = source_prior > 0.0
        scores = scores + torch.log(source_prior.clamp_min(tiny))[:, None, None, :]
        valid_heads = valid[:, None, None, :]
        masked_scores = torch.where(
            valid_heads,
            scores,
            torch.full_like(scores, -torch.inf),
        )
        row_has_support = valid_heads.any(dim=-1, keepdim=True)
        safe_scores = torch.where(row_has_support, masked_scores, torch.zeros_like(masked_scores))
        weights = torch.softmax(safe_scores, dim=-1)
        weights = weights * valid_heads.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(tiny)
        response = torch.matmul(weights, values).transpose(1, 2).reshape(
            batch, query_count, self.hidden_dim
        )
        response = self.env_attention.output(response)
        return (kappa * sigma)[:, None, None] * response

    def _route(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("source_conditioned_pairwise_honf has no query-to-group route.")

    def read(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return exact static ``C_M + C_E`` without creating query routes."""

        del return_routing_maps
        module_context = self._read_module_source_conditioned(state, encoded, receivers)
        environment_context = self._read_environment_source_conditioned(
            state, encoded, receivers, receiver_features
        )
        return module_context + environment_context, {}


__all__ = ["SourceConditionedPairwiseField"]
