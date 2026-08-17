"""CHANNELTHERMAL-SPECIFIC HONF configuration.

Inputs are nested dictionaries from the bundled configs or checkpoint metadata.
Outputs are dataclasses that combine reusable CORE HONF settings with
ChannelThermal adapter and compatibility settings. The nested wrapper is
specific to ChannelThermal, while `core_honf` remains reusable across domains.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from honf_forward_core.config import UnifiedForwardConfig


@dataclass
class ChannelThermalSpecificConfig:
    """ChannelThermal adapter and compatibility settings."""

    field_names: list[str] = field(default_factory=lambda: ["u", "v", "p", "omega", "temperature"])
    material_param_dim: int = 6
    heat_scale: float = 1.0
    global_feature_schema: str = "padding_invariant_v2"
    legacy_active_fraction_reference_slots: int | None = None
    use_local_surrogate: bool = False
    local_surrogate_checkpoint_path: str | None = None
    freeze_local_surrogate: bool = True
    local_surrogate_latent_dim: int = 128
    local_module_params_from_used_ports: bool = True
    local_surrogate_flux_mode: str = "surrogate"
    local_surrogate_flux_blend_alpha: float = 0.5
    interaction_refinement_steps: int = 0
    port_global_consistency_radius_offset: float = 0.05
    port_global_consistency_num_points: int = 32
    internal_prediction_mode: str = "auto"
    default_num_interface_points: int = 64
    fallback_internal_query_dim: int = 2
    fallback_interface_dim: int = 2
    fallback_hidden_dim: int = 128
    fallback_fourier_frequencies: int = 4

    def __post_init__(self) -> None:
        """Validate ChannelThermal schema and physical coupling modes."""

        if len(self.field_names) == 0 or len(set(self.field_names)) != len(self.field_names):
            raise ValueError("field_names must be a non-empty list of unique channel names.")
        if int(self.material_param_dim) != 6:
            raise ValueError("The current ChannelThermal material schema has exactly 6 values.")
        if float(self.heat_scale) <= 0.0:
            raise ValueError("heat_scale must be positive.")
        if self.global_feature_schema not in {"legacy_v1", "padding_invariant_v2"}:
            raise ValueError("global_feature_schema must be 'legacy_v1' or 'padding_invariant_v2'.")
        if self.global_feature_schema == "legacy_v1":
            if (
                self.legacy_active_fraction_reference_slots is None
                or int(self.legacy_active_fraction_reference_slots) <= 0
            ):
                raise ValueError("legacy_v1 requires a positive legacy_active_fraction_reference_slots value.")
        if self.internal_prediction_mode not in {"auto", "local_surrogate", "global_head"}:
            raise ValueError("internal_prediction_mode must be 'auto', 'local_surrogate', or 'global_head'.")
        if self.local_surrogate_flux_mode not in {"surrogate", "physics_from_port", "corrected_physics", "blend"}:
            raise ValueError("local_surrogate_flux_mode must be 'surrogate', 'physics_from_port', 'corrected_physics', or 'blend'.")
        if int(self.interaction_refinement_steps) not in {0, 1}:
            raise ValueError("interaction_refinement_steps supports only 0 or 1.")
        if not 0.0 <= float(self.local_surrogate_flux_blend_alpha) <= 1.0:
            raise ValueError("local_surrogate_flux_blend_alpha must be in [0, 1].")
        if int(self.port_global_consistency_num_points) <= 0:
            raise ValueError("port_global_consistency_num_points must be positive.")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "ChannelThermalSpecificConfig":
        """Build strict ChannelThermal settings from a config/checkpoint mapping."""

        payload = dict(payload or {})
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(key for key in payload if key not in allowed and not str(key).startswith("_"))
        if unknown:
            raise ValueError(f"Unknown ChannelThermal settings: {unknown}")
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    def to_dict(self) -> Dict[str, Any]:
        """Serialize domain-specific settings to a dictionary."""

        return asdict(self)


@dataclass
class ChannelThermalHONFConfig:
    """Combined config for the ChannelThermal HONF wrapper."""

    core_honf: UnifiedForwardConfig = field(default_factory=UnifiedForwardConfig)
    channelthermal: ChannelThermalSpecificConfig = field(default_factory=ChannelThermalSpecificConfig)

    def __post_init__(self) -> None:
        """Check that named physical fields match the core output width."""

        if len(self.channelthermal.field_names) != int(self.core_honf.field_dim):
            raise ValueError("field_names length must equal core_honf.field_dim.")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "ChannelThermalHONFConfig":
        """Build the nested core/domain configuration with strict key checks."""

        payload = dict(payload or {})
        core_payload = payload.get("core_honf", payload.get("core", {}))
        channel_payload = payload.get("channelthermal", {})
        if "core_honf" in payload or "core" in payload or "channelthermal" in payload:
            allowed_top = {"core_honf", "core", "channelthermal"}
            unknown = sorted(key for key in payload if key not in allowed_top and not str(key).startswith("_"))
            if unknown:
                raise ValueError(f"Unknown ChannelThermalHONFConfig sections: {unknown}")
        if not core_payload and not channel_payload:
            core_keys = set(UnifiedForwardConfig.__dataclass_fields__)  # type: ignore[attr-defined]
            legacy_keys = {
                "max_num_modules",
                "use_hyper_context",
                "use_hypergraph_gated_pairwise_kernel",
                "use_direct_module_env_decoder",
                "use_near_module_context",
                "use_global_context",
                "use_dynamic_tokens",
                "use_local_surrogate_patch",
            }
            unknown = sorted(
                key for key in payload
                if key not in core_keys and key not in legacy_keys and not str(key).startswith("_")
            )
            if unknown:
                raise ValueError(f"Unknown flat core settings: {unknown}")
            core_payload = {key: value for key, value in payload.items() if key in core_keys}
        core_payload = dict(core_payload or {})
        channel_payload = dict(channel_payload or {})
        legacy_max_modules = core_payload.get("max_num_modules", payload.get("max_num_modules"))
        if legacy_max_modules is not None and "global_feature_schema" not in channel_payload:
            channel_payload["global_feature_schema"] = "legacy_v1"
            channel_payload["legacy_active_fraction_reference_slots"] = int(legacy_max_modules)
        return cls(
            core_honf=UnifiedForwardConfig.from_dict(core_payload),
            channelthermal=ChannelThermalSpecificConfig.from_dict(channel_payload),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize both reusable and domain-specific configuration sections."""

        return {
            "core_honf": self.core_honf.to_dict(),
            "channelthermal": self.channelthermal.to_dict(),
        }

    @property
    def field_dim(self) -> int:
        """Return the number of predicted global field channels."""

        return int(self.core_honf.field_dim)

    @property
    def module_radius(self) -> float:
        """Return the physical circular-module radius."""

        return float(self.core_honf.module_radius)

    @property
    def use_local_surrogate(self) -> bool:
        """Return whether Stage-A coupling is configured."""

        return bool(self.channelthermal.use_local_surrogate)
