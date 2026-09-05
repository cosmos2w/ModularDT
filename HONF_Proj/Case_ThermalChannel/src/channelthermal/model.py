"""CHANNELTHERMAL-SPECIFIC HONF full-model wrapper.

Inputs use the legacy ChannelThermal global forward signature: a `structure`
dictionary or equivalent keyword tensors plus query coordinates, teacher port
conditions, local module parameters, and local query points. Outputs are a
legacy-compatible dictionary with global field, internal/interface predictions,
selected port tokens, base/final organizer diagnostics, and local-response
latents. The wrapper is ChannelThermal-specific; the underlying HONF organizer
and decoder remain reusable across domains.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from honf_forward_core.model import HONFNeuralField
from honf_forward_core.interface_fields import InterfaceFieldCore
from honf_forward_core.config import BatchData
from honf_forward_core.selection.predictive_rank import (
    build_deterministic_case_probes,
    select_probe_fidelity_support,
)
from .config import ChannelThermalHONFConfig
from .environment import ChannelThermalEnvironmentBuilder
from .input_adapter import ChannelThermalInputAdapter
from .fallback_heads import FallbackHeadConfig, GlobalFallbackHeads
from .local_coupling import (
    LocalSurrogateCoupling,
    build_local_module_params_from_global,
    teacher_port_tokens_from_interface_condition,
)
from .model_support import ChannelThermalModelSupportMixin
from .interface_field_coupling import (
    PreparedInterfaceChannelThermalCase,
    forward_interface_field,
)


@dataclass(frozen=True)
class PreparedChannelThermalCase:
    """Case-static HONF state reused to decode arbitrary global query chunks.

    ``organizer`` contains final module/environment/hyperedge tensors, while
    ``global_token [B,D]`` represents case-level flow, material, heat, and
    geometry context. Neither field depends on the requested global queries.
    """

    organizer: Dict[str, torch.Tensor]
    global_token: torch.Tensor


class ChannelThermalHONFModel(ChannelThermalModelSupportMixin, nn.Module):
    """Legacy-compatible ChannelThermal wrapper around the CORE HONF model."""

    def __init__(self, config: ChannelThermalHONFConfig, *, attach_local_from_checkpoint: bool = True):
        """Initialize ChannelThermalHONFModel and its required state."""

        super().__init__()
        self.config = config
        hidden = int(config.core_honf.hidden_dim)
        if config.core_honf.forward_architecture == "legacy_honf":
            self.core = HONFNeuralField(config.core_honf)
        else:
            self.core = InterfaceFieldCore(config.core_honf)
        self.input_adapter = ChannelThermalInputAdapter(
            global_feature_schema=str(config.channelthermal.global_feature_schema),
            legacy_active_fraction_reference_slots=config.channelthermal.legacy_active_fraction_reference_slots,
        )
        self.environment_builder = ChannelThermalEnvironmentBuilder()
        self.global_normalize_targets = False
        self.global_normalization_stats: Dict[str, Any] = {}
        self.local_coupling = LocalSurrogateCoupling(
            hidden_dim=hidden,
            local_surrogate_latent_dim=int(config.channelthermal.local_surrogate_latent_dim),
            dropout=float(config.core_honf.dropout),
            use_layer_norm=bool(config.core_honf.use_layer_norm),
            local_module_params_from_used_ports=bool(config.channelthermal.local_module_params_from_used_ports),
            local_surrogate_flux_mode=str(config.channelthermal.local_surrogate_flux_mode),
            local_surrogate_flux_blend_alpha=float(config.channelthermal.local_surrogate_flux_blend_alpha),
        )
        self.fallback_heads = GlobalFallbackHeads(
            hidden,
            FallbackHeadConfig(
                hidden_dim=int(config.channelthermal.fallback_hidden_dim),
                internal_query_dim=int(config.channelthermal.fallback_internal_query_dim),
                interface_dim=int(config.channelthermal.fallback_interface_dim),
                fourier_frequencies=int(config.channelthermal.fallback_fourier_frequencies),
                dropout=float(config.core_honf.dropout),
            ),
        )
        path = config.channelthermal.local_surrogate_checkpoint_path
        if bool(attach_local_from_checkpoint) and bool(config.channelthermal.use_local_surrogate) and path:
            self.local_coupling.attach_from_checkpoint(path, freeze=bool(config.channelthermal.freeze_local_surrogate), map_location="cpu")

    def set_global_target_normalization(self, stats: Optional[Dict[str, Any]], *, normalize_targets: bool) -> None:
        """Store global target statistics used by local/global consistency paths."""

        self.global_normalization_stats = dict(stats or {})
        self.global_normalize_targets = bool(normalize_targets)
        self.local_coupling.set_global_target_normalization(stats, normalize_targets=normalize_targets)

    @property
    def local_surrogate_attached(self) -> bool:
        """Return whether a Stage-A local surrogate is currently attached."""

        return self.local_coupling.has_local_surrogate

    def set_edge_capacity(self, capacity: int) -> None:
        """Set the core runtime candidate-edge budget."""

        if self.config.core_honf.forward_architecture != "legacy_honf":
            raise ValueError("Edge capacity is owned by legacy_honf and is not applicable here.")
        self.core.set_edge_capacity(capacity)

    def set_training_progress(self, *, epoch: int, total_epochs: Optional[int] = None) -> None:
        """Set core organizer warmup progress once per epoch."""

        self.core.set_training_progress(epoch=epoch, total_epochs=total_epochs)

    def selection_state(self) -> Dict[str, Optional[int]]:
        """Return explicit selection progress for checkpoint metadata."""

        return self.core.selection_state()

    def extract_hypergraph_plan(
        self,
        organizer_aux: Dict[str, Any],
        module_present: torch.Tensor,
        *,
        detach: bool = True,
    ) -> Dict[str, Any]:
        """Return the canonical static organizer plan used by inverse tooling.

        This ChannelThermal wrapper provides a model-side API so inverse code
        can consume the same compact static plan as evaluation without depending
        on evaluator control flow. Query-dependent routing maps and raw tokens
        are intentionally excluded; those are recomputed by the decoder for a
        generated physical design.
        """

        if self.config.core_honf.forward_architecture != "legacy_honf":
            raise ValueError("Hypergraph plans are not defined for interface-field baselines.")
        from honf_forward_core.evaluation.hypergraph_plan import extract_hypergraph_plan

        return extract_hypergraph_plan(
            organizer_aux,
            module_present,
            detach=detach,
            domain_length_x=float(self.config.core_honf.domain_length_x),
            domain_length_y=float(self.config.core_honf.domain_length_y),
        )

    def forward(
        self,
        structure: Optional[Dict[str, torch.Tensor]] = None,
        query_xy: Optional[torch.Tensor] = None,
        *,
        re: Optional[torch.Tensor] = None,
        u_in: Optional[torch.Tensor] = None,
        module_centers: Optional[torch.Tensor] = None,
        heat_powers: Optional[torch.Tensor] = None,
        module_present: Optional[torch.Tensor] = None,
        material_params: Optional[torch.Tensor] = None,
        interface_condition: Optional[torch.Tensor] = None,
        local_module_params: Optional[torch.Tensor] = None,
        teacher_port_tokens: Optional[torch.Tensor] = None,
        local_query_points: Optional[torch.Tensor] = None,
        local_port_condition_mode: str = "predicted",
        mixed_teacher_ratio: float = 0.5,
        return_predicted_port_outputs: bool = False,
        return_routing_maps: bool = False,
        return_edge_fields: bool = False,
        return_port_global_consistency: bool = False,
        return_prepared_state: bool = False,
        return_organizer_passes: bool = False,
        return_organizer_diagnostics: bool = False,
        case_edge_selection_mode: Optional[str] = None,
        case_edge_probe_relative_rms_tolerance: Optional[float] = None,
        case_edge_probe_channel_tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Predict the global field and per-module thermal responses.

        Tensor dimensions are ``B`` cases, ``M`` padded module slots, ``Q``
        global queries, ``Ql`` local disk queries, and ``P`` angular ports.
        Required inputs are module centers ``[B,M,2]``, heat ``[B,M]``, module
        mask ``[B,M]``, and ``query_xy [B,Q,2]``. Optional teacher interface
        conditions are ``[B,M,P,C]`` and local parameters are ``[B,M,7]``.

        Physical inputs first become generic HONF tokens. The base organizer
        predicts port conditions ``[B,M,P,5]``. If Stage A is active, its
        internal ``[B,M,Ql,1]`` and interface ``[B,M,P,2]`` responses refine
        module tokens before a final organizer/decoder pass. The primary result
        is ``pred_field [B,Q,F]`` plus the local outputs and organizer/routing
        diagnostics. ``local_port_condition_mode`` selects teacher, predicted,
        or mixed boundary conditions during training and evaluation.
        """

        if query_xy is None:
            raise ValueError("query_xy is required.")
        if self.config.core_honf.forward_architecture != "legacy_honf":
            return forward_interface_field(
                self,
                structure=structure,
                query_xy=query_xy,
                re=re,
                u_in=u_in,
                module_centers=module_centers,
                heat_powers=heat_powers,
                module_present=module_present,
                material_params=material_params,
                interface_condition=interface_condition,
                local_module_params=local_module_params,
                teacher_port_tokens=teacher_port_tokens,
                local_query_points=local_query_points,
                local_port_condition_mode=local_port_condition_mode,
                mixed_teacher_ratio=mixed_teacher_ratio,
                return_predicted_port_outputs=return_predicted_port_outputs,
                return_routing_maps=return_routing_maps,
                return_edge_fields=return_edge_fields,
                return_port_global_consistency=return_port_global_consistency,
                return_prepared_state=return_prepared_state,
                return_organizer_passes=return_organizer_passes,
                return_organizer_diagnostics=return_organizer_diagnostics,
                case_edge_selection_mode=case_edge_selection_mode,
                case_edge_probe_relative_rms_tolerance=case_edge_probe_relative_rms_tolerance,
                case_edge_probe_channel_tolerance=case_edge_probe_channel_tolerance,
            )

        if structure is not None:
            re = structure.get("re", re)
            u_in = structure.get("u_in", u_in)
            module_centers = structure.get("module_centers", module_centers)
            heat_powers = structure.get("heat_powers", heat_powers)
            module_present = structure.get("module_present", module_present)
            material_params = structure.get("material_params", material_params)
            domain_length_x = structure.get("domain_length_x")
            domain_length_y = structure.get("domain_length_y")
        else:
            domain_length_x = None
            domain_length_y = None
        if query_xy is None:
            raise ValueError("query_xy is required.")
        if module_centers is None or heat_powers is None or module_present is None:
            raise ValueError("module_centers, heat_powers, and module_present are required.")
        device = query_xy.device
        dtype = query_xy.dtype
        batch = int(query_xy.shape[0])
        if re is None:
            re = query_xy.new_zeros(batch, 1)
        if u_in is None:
            u_in = query_xy.new_zeros(batch, 1)
        if material_params is None:
            material_params = query_xy.new_zeros(batch, int(self.config.channelthermal.material_param_dim))

        adapter = self.input_adapter(
            re=re.to(device=device, dtype=dtype),
            u_in=u_in.to(device=device, dtype=dtype),
            module_centers=module_centers.to(device=device, dtype=dtype),
            heat_powers=heat_powers.to(device=device, dtype=dtype) * float(self.config.channelthermal.heat_scale),
            module_present=module_present.to(device=device, dtype=dtype),
            material_params=material_params.to(device=device, dtype=dtype),
            domain_length_x=None if domain_length_x is None else domain_length_x.to(device=device, dtype=dtype),
            domain_length_y=None if domain_length_y is None else domain_length_y.to(device=device, dtype=dtype),
        )
        env = self.environment_builder(
            batch_size=batch,
            num_env_tokens_x=int(self.config.core_honf.num_env_tokens_x),
            num_env_tokens_y=int(self.config.core_honf.num_env_tokens_y),
            domain_length_x=float(self.config.core_honf.domain_length_x),
            domain_length_y=float(self.config.core_honf.domain_length_y),
            device=device,
            dtype=dtype,
        )
        honf_batch = BatchData(
            module_centers=adapter.module_centers,
            module_present=adapter.module_present,
            module_features=adapter.module_features,
            global_context=adapter.global_context,
            query_xy=query_xy.float(),
            query_time=None,
            target_field=None,
            case_name="channelthermal",
            metadata={},
            env_coords=env.env_coords,
            env_features=env.env_features,
        )
        # Encode/organize only. The ChannelThermal local response changes
        # module tokens, so decoding a field here would be discarded work.
        final_only_selection = (
            self.config.core_honf.organizer_mode == "exchangeable_slots"
            and self.config.core_honf.edge_selection_mode == "quality_coverage"
            and self.config.core_honf.selection_warmup_mode == "all_viable"
            and int(self.config.core_honf.selection_start_epoch) >= 0
        )
        request_organizer_diagnostics = bool(return_organizer_diagnostics or return_routing_maps)
        base_output = self.core.encode_and_organize(
            honf_batch,
            organizer_selection_override="all" if final_only_selection else None,
            return_residual_interaction_tensor=request_organizer_diagnostics,
        )
        base_org = self._legacy_organizer_aux(base_output, adapter, env.env_coords)
        base_module_state = base_output["module_tokens"]
        env_state = base_output["env_tokens"]
        global_token = base_output["global_token"]

        if teacher_port_tokens is None and interface_condition is not None:
            teacher_port_tokens = teacher_port_tokens_from_interface_condition(interface_condition.float())
        ntheta = self._infer_ntheta(interface_condition, teacher_port_tokens)
        pred_port_tokens = self.local_coupling.port_head(
            base_module_state,
            base_org["module_env_context"],
            adapter.heat_powers,
            global_token,
            ntheta=ntheta,
            module_present=adapter.module_present,
        )

        mode = str(self.config.channelthermal.internal_prediction_mode)
        use_local_outputs = self._should_use_local_outputs(mode)
        module_state = base_module_state
        local_ports_used = pred_port_tokens
        local_outputs: Optional[Dict[str, torch.Tensor]] = None
        local_response_summary: Optional[torch.Tensor] = None
        interface_diagnostics: Dict[str, torch.Tensor] = {}
        predicted_port_diagnostics: Dict[str, torch.Tensor] = {}
        final_pred_port_tokens = pred_port_tokens
        provisional_org_raw: Optional[Dict[str, torch.Tensor]] = None
        if use_local_outputs:
            if local_module_params is None:
                local_module_params = build_local_module_params_from_global(
                    adapter.heat_powers,
                    interface_condition.float() if interface_condition is not None else None,
                    material_params.to(device=device, dtype=dtype) if material_params is not None else None,
                    adapter.module_present,
                )
            local_ports_used = self.local_coupling.choose_local_ports(
                pred_port_tokens=pred_port_tokens,
                teacher_port_tokens=teacher_port_tokens,
                mode=local_port_condition_mode,
                mixed_teacher_ratio=float(mixed_teacher_ratio),
            )
            local_module_params_used = self.local_coupling.local_module_params_for_ports(
                local_module_params.to(device=device, dtype=dtype),
                local_ports_used,
                adapter.module_present,
            )
            local_outputs = self.local_coupling.call_local_surrogate(
                local_module_params_used,
                local_ports_used,
                local_query_points.to(device=device, dtype=dtype) if torch.is_tensor(local_query_points) else None,
                adapter.module_present,
            )
            pred_interface, interface_diagnostics = self.local_coupling.assemble_interface(
                local_outputs=local_outputs,
                local_ports=local_ports_used,
                module_state=base_module_state,
                module_present=adapter.module_present,
            )
            local_response_summary = self.local_coupling.local_response_summary(
                local_outputs=local_outputs,
                module_present=adapter.module_present,
                interface_override=pred_interface,
            )
            module_state = self.local_coupling.fuse_module_state(
                base_module_state,
                local_outputs,
                local_response_summary,
                adapter.module_present,
            )
            refinement_steps = int(self.config.channelthermal.interaction_refinement_steps)
            if refinement_steps == 1 and (str(local_port_condition_mode).lower() != "teacher" or teacher_port_tokens is None):
                # ChannelThermal-specific one-way interaction refinement:
                # use provisional local response to decode outside temperatures,
                # refine T_env/h, rerun the local surrogate, then fuse final state.
                provisional_org_raw = self.core.organizer(
                    module_tokens=module_state,
                    env_tokens=env_state,
                    module_centers=adapter.module_centers,
                    env_coords=env.env_coords,
                    module_present=adapter.module_present,
                    global_token=global_token,
                    geometry_mode=self.config.core_honf.geometry_mode,
                    selection_override="all" if final_only_selection else None,
                    return_residual_interaction_tensor=request_organizer_diagnostics,
                )
                provisional_org_raw["module_features_raw"] = adapter.module_features
                outside_temperature, refinement_diag = self._global_temperature_for_all_ports(
                    local_ports_used,
                    module_state,
                    provisional_org_raw,
                    global_token,
                    adapter.module_centers,
                    adapter.module_present,
                )
                interface_diagnostics["refinement_use_hyper_mechanism_encoder"] = refinement_diag.get(
                    "use_hyper_mechanism_encoder",
                    outside_temperature.new_zeros(()),
                )
                refined_ports = self.local_coupling.port_refinement_head(
                    module_state,
                    local_ports_used,
                    outside_temperature,
                    local_response_summary,
                    adapter.module_present,
                )
                local_ports_used = refined_ports
                if str(local_port_condition_mode).lower() == "predicted" or teacher_port_tokens is None:
                    final_pred_port_tokens = refined_ports
                local_module_params_used = self.local_coupling.local_module_params_for_ports(
                    local_module_params.to(device=device, dtype=dtype),
                    local_ports_used,
                    adapter.module_present,
                )
                local_outputs = self.local_coupling.call_local_surrogate(
                    local_module_params_used,
                    local_ports_used,
                    local_query_points.to(device=device, dtype=dtype) if torch.is_tensor(local_query_points) else None,
                    adapter.module_present,
                )
                pred_interface, refreshed_interface_diagnostics = self.local_coupling.assemble_interface(
                    local_outputs=local_outputs,
                    local_ports=local_ports_used,
                    module_state=module_state,
                    module_present=adapter.module_present,
                )
                interface_diagnostics.update(refreshed_interface_diagnostics)
                local_response_summary = self.local_coupling.local_response_summary(
                    local_outputs=local_outputs,
                    module_present=adapter.module_present,
                    interface_override=pred_interface,
                )
                module_state = self.local_coupling.fuse_module_state(
                    base_module_state,
                    local_outputs,
                    local_response_summary,
                    adapter.module_present,
                )

            if bool(return_predicted_port_outputs) and str(local_port_condition_mode).lower() != "predicted":
                predicted_module_params = self.local_coupling.local_module_params_for_ports(
                    local_module_params.to(device=device, dtype=dtype),
                    final_pred_port_tokens,
                    adapter.module_present,
                )
                predicted_local_outputs = self.local_coupling.call_local_surrogate(
                    predicted_module_params,
                    final_pred_port_tokens,
                    local_query_points.to(device=device, dtype=dtype) if torch.is_tensor(local_query_points) else None,
                    adapter.module_present,
                )
                predicted_interface, _ = self.local_coupling.assemble_interface(
                    local_outputs=predicted_local_outputs,
                    local_ports=final_pred_port_tokens,
                    module_state=base_module_state,
                    module_present=adapter.module_present,
                )
                predicted_port_diagnostics = {
                    "predicted_port_internal_temperature": predicted_local_outputs["internal_temperature"],
                    "predicted_port_interface": predicted_interface,
                }

        if local_outputs is None and not final_only_selection:
            # No local response changed the module tokens; the base organizer
            # is already the exact final organizer and must not be recomputed.
            final_org_raw = base_output
        else:
            final_org_raw = self.core.organizer(
                module_tokens=module_state,
                env_tokens=env_state,
                module_centers=adapter.module_centers,
                env_coords=env.env_coords,
                module_present=adapter.module_present,
                global_token=global_token,
                geometry_mode=self.config.core_honf.geometry_mode,
                selection_override=None,
                return_residual_interaction_tensor=request_organizer_diagnostics,
            )
            final_org_raw["module_features_raw"] = adapter.module_features
        resolved_case_selection = str(
            self.config.core_honf.case_edge_selection_mode
            if case_edge_selection_mode is None
            else case_edge_selection_mode
        )
        if resolved_case_selection not in {"none", "probe_fidelity"}:
            raise ValueError(
                "case_edge_selection_mode must be 'none' or 'probe_fidelity'."
            )
        if resolved_case_selection == "probe_fidelity":
            if self.training:
                raise RuntimeError(
                    "probe_fidelity case-edge selection is evaluation-only; call model.eval()."
                )
            if self.config.core_honf.organizer_mode != "fixed_projection":
                raise ValueError(
                    "probe_fidelity case-edge selection requires fixed_projection organization."
                )
            if query_xy.device.type == "cuda":
                torch.cuda.synchronize(query_xy.device)
            selection_started = time.perf_counter()
            probe_xy, probe_valid = build_deterministic_case_probes(
                env.env_coords,
                adapter.module_centers,
                adapter.module_present,
                module_radius=float(self.config.core_honf.module_radius),
                limit=int(self.config.core_honf.case_edge_probe_limit),
                source=str(self.config.core_honf.case_edge_probe_source),
                domain_length_x=float(self.config.core_honf.domain_length_x),
                domain_length_y=float(self.config.core_honf.domain_length_y),
            )

            def decode_probe_fields(
                probe_queries: torch.Tensor,
                probe_organizer: Dict[str, Any],
                probe_global_token: torch.Tensor,
            ) -> torch.Tensor:
                return self.core.decode_queries(
                    query_xy=probe_queries.float(),
                    query_time=None,
                    organizer_output=probe_organizer,
                    global_token=probe_global_token,
                    query_features=self._query_features(probe_queries.float()),
                )["pred_field"]

            final_org_raw = select_probe_fidelity_support(
                final_org_raw,
                global_token,
                probe_xy,
                probe_valid,
                decode_probe_fields,
                relative_rms_tolerance=float(
                    self.config.core_honf.case_edge_probe_relative_rms_tolerance
                    if case_edge_probe_relative_rms_tolerance is None
                    else case_edge_probe_relative_rms_tolerance
                ),
                channel_tolerance=float(
                    self.config.core_honf.case_edge_probe_channel_tolerance
                    if case_edge_probe_channel_tolerance is None
                    else case_edge_probe_channel_tolerance
                ),
                search=str(self.config.core_honf.case_edge_probe_search),
            )
            if query_xy.device.type == "cuda":
                torch.cuda.synchronize(query_xy.device)
            final_org_raw["predictive_selection_seconds"] = final_org_raw[
                "hyper_state"
            ].new_full(
                (batch,),
                float(time.perf_counter() - selection_started),
            )
        decoder_output = self.core.decode_queries(
            query_xy=query_xy.float(),
            query_time=None,
            organizer_output=final_org_raw,
            global_token=global_token,
            query_features=self._query_features(query_xy.float()),
            return_routing_maps=bool(return_routing_maps),
            return_edge_fields=bool(return_edge_fields),
        )
        final_output: Dict[str, Any] = {}
        final_output.update(final_org_raw)
        final_output.update(decoder_output)
        org = self._legacy_organizer_aux(final_output, adapter, env.env_coords)

        if local_outputs is not None:
            pred_internal = local_outputs["internal_temperature"]
            module_response_latent = local_outputs["module_response_latent"]
            interface_source = "local_surrogate"
        else:
            pred_internal = self.fallback_heads.predict_internal(module_state, local_query_points, adapter.module_present)
            pred_interface = self.fallback_heads.predict_interface(module_state, ntheta=ntheta, module_present=adapter.module_present)
            module_response_latent = module_state
            interface_source = "global_head"

        if bool(return_port_global_consistency):
            port_global_temperature, port_global_t_env, port_global_mask, port_global_diag = self._decode_global_temperature_at_ports(
                final_pred_port_tokens,
                module_state,
                final_org_raw,
                global_token,
                adapter.module_centers,
                adapter.module_present,
            )
        else:
            # Dense port-global probes are loss/diagnostic-only. Keep the legacy
            # keys but avoid the extra decoder pass for standard forwards.
            port_global_temperature = query_xy.new_empty(batch, adapter.module_centers.shape[1], 0)
            port_global_t_env = query_xy.new_empty(batch, adapter.module_centers.shape[1], 0)
            port_global_mask = query_xy.new_empty(batch, adapter.module_centers.shape[1], 0)
            port_global_diag = {"use_hyper_mechanism_encoder": decoder_output["pred_field"].new_zeros(())}
        result = {
            "pred_field": decoder_output["pred_field"],
            "pred_internal_temperature": pred_internal,
            "pred_interface": pred_interface,
            "pred_port_condition": final_pred_port_tokens,
            "pred_port_condition_raw": pred_port_tokens,
            "local_port_condition_used": local_ports_used,
            "pred_port_global_temperature": port_global_temperature,
            "pred_port_global_temperature_target": port_global_t_env,
            "pred_port_global_consistency_mask": port_global_mask,
            "interface_source": interface_source,
            "pred_interface_source": interface_source,
            "module_response_latent": module_response_latent,
            "organizer_aux": org,
            "base_organizer_aux": base_org,
            "routing_aux": {
                key: value
                for key, value in decoder_output.items()
                if key
                not in {
                    "pred_field",
                    "pred_field_background",
                    "pred_field_by_edge",
                    "edge_contribution_abs_mean",
                    "edge_contribution_rms",
                    "edge_contribution_energy_fraction",
                }
            },
        }
        if return_organizer_passes:
            result["provisional_organizer_aux"] = (
                {}
                if provisional_org_raw is None
                else self._legacy_organizer_aux(provisional_org_raw, adapter, env.env_coords)
            )
        for key in (
            "pred_field_background",
            "pred_field_by_edge",
            "edge_contribution_abs_mean",
            "edge_contribution_rms",
            "edge_contribution_energy_fraction",
        ):
            if key in decoder_output:
                result[key] = decoder_output[key]
        result["routing_aux"]["port_global_use_hyper_mechanism_encoder"] = port_global_diag.get(
            "use_hyper_mechanism_encoder",
            decoder_output["pred_field"].new_zeros(()),
        )
        result.update(interface_diagnostics)
        result.update(predicted_port_diagnostics)
        if return_prepared_state:
            result["prepared_state"] = PreparedChannelThermalCase(
                organizer=final_org_raw,
                global_token=global_token,
            )
        return result

    def decode_prepared(
        self,
        prepared: PreparedChannelThermalCase | PreparedInterfaceChannelThermalCase,
        query_xy: torch.Tensor,
        *,
        return_routing_maps: bool = False,
        return_edge_fields: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Decode ``query_xy [B,Q,2]`` without recomputing case/local physics."""

        if isinstance(prepared, PreparedInterfaceChannelThermalCase):
            if prepared.architecture != self.config.core_honf.forward_architecture:
                raise ValueError("Prepared interface-field architecture does not match the model.")
            return self.core.decode_queries(
                prepared.prepared,
                query_xy.float(),
                query_features=self._query_features(query_xy.float()),
                return_routing_maps=return_routing_maps,
                return_edge_fields=return_edge_fields,
            )

        return self.core.decode_queries(
            query_xy=query_xy.float(),
            query_time=None,
            organizer_output=prepared.organizer,
            global_token=prepared.global_token,
            query_features=self._query_features(query_xy.float()),
            return_routing_maps=return_routing_maps,
            return_edge_fields=return_edge_fields,
        )

    def _query_features(self, query_xy: torch.Tensor) -> torch.Tensor | None:
        """Inject case geometry for maintained configs; avoid doubling legacy features."""

        if self.config.core_honf.boundary_feature_mode != "none":
            return None
        return self.environment_builder.query_features(
            query_xy,
            domain_length_x=float(self.config.core_honf.domain_length_x),
            domain_length_y=float(self.config.core_honf.domain_length_y),
        )
