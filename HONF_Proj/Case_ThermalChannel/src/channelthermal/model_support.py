"""Non-registering physical and compatibility helpers for the model facade."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class ChannelThermalModelSupportMixin:
    """Preserve ThermalChannel coupling helpers without owning module state."""

    def _should_use_local_outputs(self, mode: str) -> bool:
        """Resolve auto/local/global internal prediction mode against attachment state."""

        if mode == "global_head":
            return False
        if mode == "local_surrogate":
            if not self.local_coupling.has_local_surrogate:
                raise RuntimeError("internal_prediction_mode='local_surrogate' requires an attached local surrogate.")
            return True
        return bool(self.config.channelthermal.use_local_surrogate and self.local_coupling.has_local_surrogate)

    def _legacy_organizer_aux(
        self,
        core_output: Dict[str, Any],
        adapter: Any,
        env_coords: torch.Tensor,
    ) -> Dict[str, Any]:
        """Expose the compact organizer keys expected by plotting and plan export."""

        org_keys = {
            "A_me",
            "A_mh",
            "A_eh",
            "A_mh_hard",
            "A_mh_soft",
            "A_eh_hard",
            "A_eh_soft",
            "candidate_A_mh",
            "candidate_A_eh",
            "hyper_state",
            "candidate_hyper_state",
            "hyper_source_coords",
            "hyper_source_variance",
            "hyper_region_coords",
            "hyper_region_variance",
            "hyper_source_scale",
            "hyper_region_scale",
            "hyper_module_mass_raw",
            "hyper_env_mass_raw",
            "hyper_module_mass",
            "hyper_env_mass",
            "hyper_module_purity",
            "hyper_env_purity",
            "hyper_strength",
            "edge_quality",
            "edge_active_mask",
            "hard_selected_edge_mask",
            "edge_transition_gate",
            "candidate_edge_viable_mask",
            "edge_viable_mask",
            "effective_edge_mask",
            "candidate_edge_count",
            "selected_edge_count",
            "viable_selected_edge_count",
            "hard_selected_edge_count",
            "edge_transition_gate_sum",
            "empty_selected_edge_count",
            "active_edge_count",
            "selection_module_coverage",
            "selection_environment_coverage",
            "candidate_module_mass_fraction",
            "candidate_environment_mass_fraction",
            "candidate_module_purity",
            "candidate_environment_purity",
            "candidate_source_coords",
            "candidate_source_scale",
            "candidate_region_coords",
            "candidate_region_scale",
            "candidate_module_nonzero_fraction",
            "candidate_environment_nonzero_fraction",
            "selected_module_nonzero_fraction",
            "selected_environment_nonzero_fraction",
            "selected_module_probability_mass_min",
            "selected_module_probability_mass_p05",
            "selected_module_probability_mass_mean",
            "selected_environment_probability_mass_min",
            "selected_environment_probability_mass_p05",
            "selected_environment_probability_mass_mean",
            "pre_fallback_zero_support_module_rows",
            "post_fallback_zero_support_module_rows",
            "pre_fallback_zero_support_environment_rows",
            "post_fallback_zero_support_environment_rows",
            "empty_support_count",
            "fallback_support_count",
            "selection_transition_fraction",
            "module_sparsity_fraction",
            "environment_sparsity_fraction",
            "query_sparsity_fraction",
            "training_progress_epoch",
            "routing_execution_gathered",
            "module_env_context",
            "module_centers",
            "module_present",
            "env_coords",
            "module_tokens",
            "env_tokens",
            "module_features_raw",
            # Case-adaptive residual organizer diagnostics. Keep these in the
            # compatibility view so downstream training/reporting code can
            # consume the organizer's direct values without reconstructing
            # them from legacy strength thresholds.
            "residual_stop_fraction",
            "residual_soft_stop_temperature",
            "residual_fraction_trace",
            "residual_marginal_explained_fraction",
            # Phase-2 tensor residual diagnostics.  The full interaction
            # tensor is opt-in, but when the organizer supplies it the
            # compatibility view must preserve it for explicit evaluation
            # requests rather than silently dropping it.
            "residual_interaction_tensor",
            "residual_content_factor",
            "residual_mechanism_strength",
            "residual_module_factor",
            "residual_environment_factor",
            "residual_coupling_row_mass",
            "edge_survival_weight",
            "edge_survival_soft",
            "hard_case_edge_mask",
            "case_adaptive_edge_count",
            "case_adaptive_edge_cap",
            "case_adaptive_soft_edge_count",
            "case_adaptive_stop_reached",
            "case_adaptive_cap_hit",
            "case_adaptive_stop_margin",
            "residual_monotonic_violation_max",
            "case_adaptive_support_gap",
        }
        org = {key: core_output[key] for key in org_keys if key in core_output}
        org["hyper_thermal_region_coords"] = core_output.get("hyper_region_coords")
        if self.config.core_honf.organizer_mode in {
            "exchangeable_slots",
            "case_adaptive_residual",
            "case_adaptive_tensor_residual",
        }:
            # Visualize only the effective field generators. For the residual
            # organizers this is the hard/effective support used by the
            # decoder.  ``edge_survival_soft`` is a gradient surrogate and
            # must not change the compatibility mask used for reports.
            if self.config.core_honf.organizer_mode == "case_adaptive_tensor_residual":
                # Phase 2 guarantees hard forward support in train and eval;
                # prefer its explicit hard mask even if a caller also retains
                # a soft/effective surrogate for debugging.
                effective_mask = core_output.get("hard_case_edge_mask")
                if effective_mask is None:
                    effective_mask = core_output.get("effective_edge_mask")
            else:
                effective_mask = core_output.get("effective_edge_mask")
            if effective_mask is None:
                effective_mask = core_output.get("hard_case_edge_mask")
            if effective_mask is None:
                effective_mask = core_output.get("edge_active_mask")
            if effective_mask is None:
                effective_mask = torch.ones_like(core_output["hyper_strength"])
            org["active_hyperedge_mask"] = effective_mask
        else:
            # Preserve the fixed-projection compatibility visualization.
            org["active_hyperedge_mask"] = (core_output["hyper_strength"] > 0.05).to(dtype=core_output["hyper_strength"].dtype)
        org["module_centers"] = adapter.module_centers
        org["module_present"] = adapter.module_present
        org["heat_powers"] = adapter.heat_powers
        org["env_coords"] = env_coords
        return org

    def _temperature_from_field_output(self, field_values: torch.Tensor) -> torch.Tensor:
        """Extract and, when needed, denormalize the configured temperature channel."""

        names = list(self.config.channelthermal.field_names)
        if "temperature" not in names:
            raise ValueError("field_names must contain 'temperature' for port/global coupling.")
        temperature_index = names.index("temperature")
        temperature = field_values[..., temperature_index]
        if not self.global_normalize_targets:
            return temperature
        mean = self.global_normalization_stats.get("field_mean_by_channel")
        std = self.global_normalization_stats.get("field_std_by_channel")
        if mean is None or std is None:
            return temperature
        mean_t = torch.as_tensor(mean, device=field_values.device, dtype=field_values.dtype)
        std_t = torch.as_tensor(std, device=field_values.device, dtype=field_values.dtype)
        if mean_t.numel() <= temperature_index or std_t.numel() <= temperature_index:
            return temperature
        return temperature * std_t[temperature_index].clamp_min(1.0e-6) + mean_t[temperature_index]

    def _port_subset_indices(self, ntheta: int, device: torch.device) -> torch.Tensor:
        # ChannelThermal-specific: compare T_env only on a controlled subset of
        # angular boundary points to keep this auxiliary physical loss cheap.
        """Choose evenly spaced angular ports for the global consistency loss."""

        count = int(self.config.channelthermal.port_global_consistency_num_points)
        count = max(1, min(count, int(ntheta)))
        if count >= int(ntheta):
            return torch.arange(int(ntheta), device=device)
        return torch.linspace(0, int(ntheta) - 1, count, device=device).round().long()

    def _decode_global_temperature_at_ports(
        self,
        port_tokens: torch.Tensor,
        module_state: torch.Tensor,
        org: Dict[str, torch.Tensor],
        global_token: torch.Tensor,
        module_centers: torch.Tensor,
        module_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Decode global temperature just outside a configured subset of module ports."""

        del module_state
        batch, num_modules, ntheta, _ = port_tokens.shape
        indices = self._port_subset_indices(ntheta, port_tokens.device)
        selected_ports = port_tokens.index_select(dim=-2, index=indices)
        normals = selected_ports[..., 1:3]
        radius = float(self.config.core_honf.module_radius) + float(self.config.channelthermal.port_global_consistency_radius_offset)
        outside_xy = module_centers[:, :, None, :] + radius * normals
        outside_xy = torch.stack(
            [
                outside_xy[..., 0].clamp(0.0, float(self.config.core_honf.domain_length_x)),
                outside_xy[..., 1].clamp(0.0, float(self.config.core_honf.domain_length_y)),
            ],
            dim=-1,
        )
        flat_xy = outside_xy.reshape(batch, num_modules * int(indices.numel()), 2)
        # Keep the full organizer output here. The reduced legacy organizer aux
        # is only for plotting and lacks mechanism features used by the decoder.
        port_output = self.core.decode_queries(
            flat_xy,
            None,
            org,
            global_token,
            query_features=self._query_features(flat_xy),
        )
        temperature = self._temperature_from_field_output(port_output["pred_field"]).reshape(batch, num_modules, int(indices.numel()))
        target_t_env = selected_ports[..., 3]
        valid_mask = module_present[:, :, None].expand_as(temperature)
        return temperature * valid_mask, target_t_env * valid_mask, valid_mask, port_output

    def _global_temperature_for_all_ports(
        self,
        port_tokens: torch.Tensor,
        module_state: torch.Tensor,
        org: Dict[str, torch.Tensor],
        global_token: torch.Tensor,
        module_centers: torch.Tensor,
        module_present: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Decode outside temperature at every port for one interaction refinement."""

        del module_state
        batch, num_modules, ntheta, _ = port_tokens.shape
        normals = port_tokens[..., 1:3]
        radius = float(self.config.core_honf.module_radius) + float(self.config.channelthermal.port_global_consistency_radius_offset)
        outside_xy = module_centers[:, :, None, :] + radius * normals
        outside_xy = torch.stack(
            [
                outside_xy[..., 0].clamp(0.0, float(self.config.core_honf.domain_length_x)),
                outside_xy[..., 1].clamp(0.0, float(self.config.core_honf.domain_length_y)),
            ],
            dim=-1,
        )
        flat_xy = outside_xy.reshape(batch, num_modules * ntheta, 2)
        refinement_output = self.core.decode_queries(
            flat_xy,
            None,
            org,
            global_token,
            query_features=self._query_features(flat_xy),
        )
        temperature = self._temperature_from_field_output(refinement_output["pred_field"]).reshape(batch, num_modules, ntheta)
        return temperature * module_present[:, :, None], refinement_output

    def _infer_ntheta(
        self,
        interface_condition: Optional[torch.Tensor],
        teacher_port_tokens: Optional[torch.Tensor],
    ) -> int:
        """Infer the angular port count from supplied tensors or configuration."""

        if interface_condition is not None and interface_condition.ndim >= 4:
            return int(interface_condition.shape[-2])
        if teacher_port_tokens is not None and teacher_port_tokens.ndim >= 4:
            return int(teacher_port_tokens.shape[-2])
        return int(self.config.channelthermal.default_num_interface_points)
