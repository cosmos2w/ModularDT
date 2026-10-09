"""Wind physical adapter for the shared joint regional-field core.

Wind owns native rotor-diameter coordinates, target-free scene construction,
the TRAIN-fitted empirical height profile, and conversion of the learned
nonlinear residual back to m/s. The shared core owns the learned interaction
and receiver-addressable regional representation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from honf_forward_core.interface_fields.interaction_core import DependencySpec, InteractionScene
from honf_forward_core.interface_fields.joint_regional import JointRegionalFieldCore
from torch import nn

from .geometry import POSITIONAL_SCALE_D
from .normalization import VelocityNormalizer, VerticalProfileBaseline


def _value_signature(value: Any) -> tuple[Any, ...]:
    """Capture tensor ownership/version or the small bytes of a native transform."""

    if value is None or isinstance(value, (bool, int, str)):
        return (type(value).__qualname__, value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, DependencySpec):
        return (
            "dependency",
            id(value),
            value.dataset,
            value.output_law,
            tuple(value.configuration_inputs),
            tuple(value.applicable_controls),
            tuple(value.output_roles),
            tuple(value.units),
            tuple(value.edges),
            tuple(value.prepared_nodes),
        )
    if torch.is_tensor(value):
        try:
            version = int(value._version)
        except RuntimeError as error:
            if "Inference tensors do not track version counter" not in str(error):
                raise
            version = None
        return (
            "tensor", id(value), int(value.data_ptr()), version, tuple(value.shape),
            tuple(value.stride()), str(value.dtype), str(value.device),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_value_signature(item) for item in value))
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        return (type(value).__module__, type(value).__qualname__, id(value))
    return (
        "array", id(value), array.dtype.str, tuple(array.shape),
        hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
    )


def _scene_signature(scene: InteractionScene) -> tuple[Any, ...]:
    return tuple((name, _value_signature(getattr(scene, name))) for name in scene.__dataclass_fields__)


def _parameter_signature(module: nn.Module) -> tuple[Any, ...]:
    result = []
    for name, parameter in module.named_parameters():
        try:
            version = int(parameter._version)
        except RuntimeError as error:
            if "Inference tensors do not track version counter" not in str(error):
                raise
            version = None
        result.append((name, id(parameter), int(parameter.data_ptr()), version, tuple(parameter.shape)))
    return tuple(result)


def _profile_signature(profile: VerticalProfileBaseline) -> tuple[Any, ...]:
    return (
        _value_signature(profile.bin_centers_D),
        _value_signature(profile.values_mps),
        _value_signature(profile.counts),
        float(profile.z_min_D).hex(),
        float(profile.z_max_D).hex(),
    )


def _normalizer_signature(normalizer: VelocityNormalizer) -> tuple[Any, ...]:
    return (
        _value_signature(normalizer.mean),
        _value_signature(normalizer.std),
        _value_signature(normalizer.safe_std),
        float(normalizer.u_ref_mps).hex(),
        float(normalizer.std_floor).hex(),
    )


def wind_receiver_features(scene: InteractionScene, receivers_D: torch.Tensor) -> torch.Tensor:
    """Return the native Wind support-relative features used by the old adapter."""

    if receivers_D.ndim != 3 or receivers_D.shape[0] != scene.centers.shape[0] or receivers_D.shape[-1] != 3:
        raise ValueError("Wind receivers must have shape [B,Q,3] in rotor-diameter coordinates.")
    scale = receivers_D.new_tensor(POSITIONAL_SCALE_D)
    lower = scene.context[:, 5:8] * scale
    extent = scene.context[:, 8:11] * scale
    upper = lower + extent
    lower_distance = (receivers_D - lower[:, None]) / scale
    upper_distance = (upper[:, None] - receivers_D) / scale
    absolute_height = receivers_D[..., 2:3] / scale[2]
    return torch.cat((lower_distance, upper_distance, absolute_height), dim=-1)


@dataclass(frozen=True)
class PreparedWindJointRegional:
    """Owned prepared Wind context for repeated receiver reads."""

    owner_id: int
    core_context: Any
    scene: InteractionScene
    scene_signature: tuple[Any, ...]
    parameter_signature: tuple[Any, ...]
    normalizer_signature: tuple[Any, ...]
    profile_signature: tuple[Any, ...]


class WindFarmJointRegionalModel(nn.Module):
    """Fresh Wind residual model executing the common joint regional core."""

    FAMILY = "joint_regional_hypergraph"

    def __init__(
        self,
        *,
        velocity_transform: VelocityNormalizer,
        background_profile: VerticalProfileBaseline,
        mode: str = "J-H",
        hidden: int = 128,
        message: int = 128,
        regional_anchors: int = 32,
        depth: int = 2,
        receiver_tile: int = 512,
        seed: int = 42,
        max_sources: int = 30,
        locality_prior_strength: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in {"J-H", "J-geometry", "J-direct"}:
            raise ValueError("Wind joint regional mode must be J-H, J-geometry, or J-direct.")
        if int(receiver_tile) <= 0:
            raise ValueError("Wind receiver tile must be positive.")
        if background_profile is None:
            raise ValueError("Wind joint regional prediction requires its TRAIN-fitted height profile.")
        locality_prior_strength = float(locality_prior_strength)
        if not np.isfinite(locality_prior_strength) or locality_prior_strength < 0.0:
            raise ValueError("Wind locality_prior_strength must be finite and nonnegative.")
        if locality_prior_strength > 0.0 and mode != "J-H":
            raise ValueError("Wind locality prior is supported only for J-H.")
        self.velocity_transform = velocity_transform
        self.background_profile = background_profile
        self.mode = str(mode)
        self.seed = int(seed)
        self.receiver_tile = int(receiver_tile)
        self.max_sources = int(max_sources)
        self.locality_prior_strength = locality_prior_strength
        self.core_config = {
            "source_width": 2,
            "context_width": 11,
            "environment_width": 7,
            "spatial_dim": 3,
            "hidden": int(hidden),
            "message": int(message),
            "mode": self.mode,
            "regional_anchors": int(regional_anchors),
            "depth": int(depth),
            "field_outputs": 3,
            "affine_outputs": 0,
            "query_width": 7,
            "max_sources": self.max_sources,
        }
        if locality_prior_strength:
            self.core_config["locality_prior_strength"] = locality_prior_strength
        self.core = JointRegionalFieldCore(**self.core_config)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> WindFarmJointRegionalModel:
        payload = dict(config)
        if payload.get("family") != cls.FAMILY or payload.get("dataset") != "WindFarm":
            raise ValueError("Saved model configuration is not a standalone Wind joint-regional model.")
        normalizer = VelocityNormalizer.from_dict(dict(payload["velocity_transform"]))
        profile = VerticalProfileBaseline.from_dict(dict(payload["background_profile"]))
        core = dict(payload["core"])
        if (
            int(core.get("source_width", -1)) != 2
            or int(core.get("context_width", -1)) != 11
            or int(core.get("environment_width", -1)) != 7
            or int(core.get("spatial_dim", -1)) != 3
            or int(core.get("field_outputs", -1)) != 3
            or int(core.get("affine_outputs", -1)) != 0
            or int(core.get("query_width", -1)) != 7
        ):
            raise ValueError("Saved Wind joint core dimensions differ from the native nonlinear three-vector contract.")
        return cls(
            velocity_transform=normalizer,
            background_profile=profile,
            mode=str(core["mode"]),
            hidden=int(core["hidden"]),
            message=int(core["message"]),
            regional_anchors=int(core["regional_anchors"]),
            depth=int(core["depth"]),
            receiver_tile=int(payload.get("receiver_tile", 512)),
            seed=int(payload.get("seed", 42)),
            max_sources=int(core["max_sources"]),
            locality_prior_strength=float(core.get("locality_prior_strength", 0.0)),
        )

    def export_config(self) -> dict[str, Any]:
        return {
            "family": self.FAMILY,
            "dataset": "WindFarm",
            "seed": self.seed,
            "receiver_tile": self.receiver_tile,
            "core": dict(self.core_config),
            "velocity_transform": self.velocity_transform.to_dict(),
            "background_profile": self.background_profile.to_dict(),
            "native_coordinate_frame": "rotor_diameters",
            "output_units": "m/s",
            "output_law": "nonlinear_configuration_residual_on_train_height_profile",
            "background_interpretation": "TRAIN-fitted empirical height profile, including mean wake structure",
            "exact_affine_increments": False,
        }

    def prepare_scene(self, scene: InteractionScene) -> PreparedWindJointRegional:
        if not isinstance(scene, InteractionScene):
            raise TypeError("Wind joint regional preparation accepts only target-free InteractionScene inputs.")
        if scene.sources.ndim != 3 or scene.sources.shape[-1] != 2:
            raise ValueError("Wind source nodes must have shape [B,M,2].")
        if scene.context.ndim != 2 or scene.context.shape[-1] != 11:
            raise ValueError("Wind scene context must have shape [B,11].")
        if scene.centers.shape != (*scene.sources.shape[:2], 3):
            raise ValueError("Wind module centers must align with source nodes in 3-D rotor-diameter coordinates.")
        if scene.environment_tokens is None or scene.environment_tokens.shape[-1] != 7:
            raise ValueError("Wind joint regional context requires seven geometry-only environment features.")
        if scene.environment_coords is None or scene.environment_coords.shape[-1] != 3:
            raise ValueError("Wind environment coordinates must use the native 3-D rotor-diameter frame.")
        if scene.sources.shape[1] > self.max_sources:
            raise ValueError("Wind source count exceeds the model's sealed physical source capacity.")
        positional_scale = scene.centers.new_tensor(POSITIONAL_SCALE_D)
        domain_origin = scene.context[:, 5:8] * positional_scale
        core_context = self.core.prepare_context(
            sources=scene.sources,
            context=scene.context,
            centers=scene.centers,
            present=scene.present,
            lengths=scene.lengths,
            source_lengths=scene.source_lengths,
            source_measures=scene.source_measures,
            environment_tokens=scene.environment_tokens,
            environment_coords=scene.environment_coords,
            environment_present=scene.environment_present,
            environment_measures=scene.environment_measures,
            source_ids=scene.source_ids,
            domain_origin=domain_origin,
        )
        return PreparedWindJointRegional(
            owner_id=id(self),
            core_context=core_context,
            scene=scene,
            scene_signature=_scene_signature(scene),
            parameter_signature=_parameter_signature(self),
            normalizer_signature=_normalizer_signature(self.velocity_transform),
            profile_signature=_profile_signature(self.background_profile),
        )

    def _assert_prepared(self, prepared: PreparedWindJointRegional) -> None:
        if not isinstance(prepared, PreparedWindJointRegional) or prepared.owner_id != id(self):
            raise ValueError("Prepared Wind context belongs to another joint regional model; rebuild it.")
        if _scene_signature(prepared.scene) != prepared.scene_signature:
            raise ValueError("Wind scene tensors changed after preparation; rebuild prepared context.")
        if _parameter_signature(self) != prepared.parameter_signature:
            raise ValueError("Wind model parameters changed after preparation; rebuild prepared context.")
        if _normalizer_signature(self.velocity_transform) != prepared.normalizer_signature:
            raise ValueError("Wind velocity transform changed after preparation; rebuild prepared context.")
        if _profile_signature(self.background_profile) != prepared.profile_signature:
            raise ValueError("Wind empirical profile changed after preparation; rebuild prepared context.")
        assert_owned = getattr(self.core, "assert_owned", None)
        if callable(assert_owned):
            assert_owned(prepared.core_context)

    def predict_standardized(
        self,
        prepared: PreparedWindJointRegional,
        receivers_D: torch.Tensor,
        *,
        receiver_features: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        self._assert_prepared(prepared)
        features = (
            wind_receiver_features(prepared.scene, receivers_D)
            if receiver_features is None
            else receiver_features
        )
        if features.shape != (*receivers_D.shape[:2], 7):
            raise ValueError("Wind receiver features must align with [B,Q] and have width seven.")
        output = self.core.predict_fields(
            prepared.core_context,
            receivers_D,
            receiver_features=features,
            chunk_size=self.receiver_tile if chunk_size is None else int(chunk_size),
        )
        if output.shape != (*receivers_D.shape[:2], 3):
            raise RuntimeError("Wind joint regional core returned an invalid three-component field shape.")
        return output

    def profile_at_receivers(self, receivers_D: torch.Tensor) -> torch.Tensor:
        if receivers_D.ndim != 3 or receivers_D.shape[-1] != 3:
            raise ValueError("Wind receivers must have shape [B,Q,3] in rotor-diameter coordinates.")
        centers = receivers_D.new_tensor(self.background_profile.bin_centers_D)
        values = receivers_D.new_tensor(self.background_profile.values_mps)
        z = receivers_D[..., 2].contiguous()
        lower_index = torch.searchsorted(centers, z).sub(1).clamp(0, centers.numel() - 2)
        lower_z = centers[lower_index]
        upper_z = centers[lower_index + 1]
        fraction = ((z - lower_z) / (upper_z - lower_z)).clamp(0.0, 1.0)
        lower = values[lower_index]
        upper = values[lower_index + 1]
        return lower + fraction[..., None] * (upper - lower)

    def predict_physical(
        self,
        prepared: PreparedWindJointRegional,
        receivers_D: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        residual = self.predict_standardized(prepared, receivers_D, chunk_size=chunk_size)
        scale = residual.new_tensor(self.velocity_transform.safe_std)
        return self.profile_at_receivers(receivers_D) + residual * scale * float(self.velocity_transform.u_ref_mps)

    def forward(self, scene: InteractionScene, receivers_D: torch.Tensor) -> torch.Tensor:
        return self.predict_physical(self.prepare_scene(scene), receivers_D)

    def normalize_residual(self, physical_mps: torch.Tensor, receivers_D: torch.Tensor) -> torch.Tensor:
        if physical_mps.shape != receivers_D.shape:
            raise ValueError("Wind physical velocity must align with its receiver coordinates.")
        scale = physical_mps.new_tensor(self.velocity_transform.safe_std)
        return (physical_mps - self.profile_at_receivers(receivers_D)) / (
            scale * float(self.velocity_transform.u_ref_mps)
        )

    def apply_increment(self, *_args: Any, **_kwargs: Any) -> None:
        raise ValueError("Wind joint regional output is nonlinear and does not support exact affine increments.")


__all__ = [
    "PreparedWindJointRegional",
    "WindFarmJointRegionalModel",
    "wind_receiver_features",
]
