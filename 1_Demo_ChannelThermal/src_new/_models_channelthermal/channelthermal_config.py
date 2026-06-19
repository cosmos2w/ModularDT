"""CHANNELTHERMAL-SPECIFIC HONF configuration.

Inputs are nested dictionaries from `Configs_new` or checkpoint metadata.
Outputs are dataclasses that combine reusable CORE HONF settings with
ChannelThermal adapter and compatibility settings. The nested wrapper is
specific to ChannelThermal, while `core_honf` remains reusable across domains.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from _models_core.honf_types import UnifiedForwardConfig


@dataclass
class ChannelThermalSpecificConfig:
    """ChannelThermal adapter and compatibility settings."""

    field_names: list[str] = field(default_factory=lambda: ["u", "v", "p", "omega", "temperature"])
    material_param_dim: int = 6
    heat_scale: float = 1.0
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
    enable_fallback_heads: bool = False
    default_num_interface_points: int = 64
    fallback_internal_query_dim: int = 2
    fallback_interface_dim: int = 2
    fallback_hidden_dim: int = 128
    fallback_fourier_frequencies: int = 4

    def __post_init__(self) -> None:
        if self.internal_prediction_mode not in {"auto", "local_surrogate", "global_head"}:
            raise ValueError("internal_prediction_mode must be 'auto', 'local_surrogate', or 'global_head'.")
        if self.local_surrogate_flux_mode not in {"surrogate", "physics_from_port", "corrected_physics", "blend"}:
            raise ValueError("local_surrogate_flux_mode must be 'surrogate', 'physics_from_port', 'corrected_physics', or 'blend'.")
        if int(self.interaction_refinement_steps) not in {0, 1}:
            raise ValueError("interaction_refinement_steps supports only 0 or 1.")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "ChannelThermalSpecificConfig":
        payload = dict(payload or {})
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChannelThermalHONFConfig:
    """Combined config for the ChannelThermal HONF wrapper."""

    core_honf: UnifiedForwardConfig = field(default_factory=UnifiedForwardConfig)
    channelthermal: ChannelThermalSpecificConfig = field(default_factory=ChannelThermalSpecificConfig)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "ChannelThermalHONFConfig":
        payload = dict(payload or {})
        core_payload = payload.get("core_honf", payload.get("core", {}))
        channel_payload = payload.get("channelthermal", {})
        if not core_payload:
            core_keys = set(UnifiedForwardConfig.__dataclass_fields__)  # type: ignore[attr-defined]
            core_payload = {key: value for key, value in payload.items() if key in core_keys}
        return cls(
            core_honf=UnifiedForwardConfig.from_dict(dict(core_payload or {})),
            channelthermal=ChannelThermalSpecificConfig.from_dict(dict(channel_payload or {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_honf": self.core_honf.to_dict(),
            "channelthermal": self.channelthermal.to_dict(),
        }

    @property
    def field_dim(self) -> int:
        return int(self.core_honf.field_dim)

    @property
    def module_radius(self) -> float:
        return float(self.core_honf.module_radius)

    @property
    def use_local_surrogate(self) -> bool:
        return bool(self.channelthermal.use_local_surrogate)
