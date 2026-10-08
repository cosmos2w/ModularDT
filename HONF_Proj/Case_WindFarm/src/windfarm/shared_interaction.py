"""Opt-in WindFarm bridge to the dataset-independent interaction core.

Wind owns the physical coordinates, turbine descriptors, environmental
quadrature, three velocity units, and target normalization. The readout itself
is the same ``NonlinearFieldReadout`` used by other datasets; this adapter
does not route through the legacy ``InterfaceFieldCore`` family.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from honf_forward_core.interface_fields.interaction_core import (
    DependencySpec,
    InteractionScene,
    NonlinearFieldReadout,
    PreparedResponseContext,
)
from honf_forward_core.interface_fields.interaction_refinement import RefinedNonlinearFieldReadout
from torch import nn

from .geometry import POSITIONAL_SCALE_D
from .normalization import VelocityNormalizer, VerticalProfileBaseline

WIND_INTERACTION_DEPENDENCY = DependencySpec(
    dataset="WindFarm",
    output_law="nonlinear",
    configuration_inputs=(
        "turbine_geometry",
        "wind_direction",
        "native_domain_support",
        "environment_geometry_and_measure",
        "reference_inflow",
    ),
    applicable_controls=(),
    output_roles=("Ux", "Uy", "Uz"),
    units=("m/s", "m/s", "m/s"),
    edges=(
        ("turbine_geometry", "source_context"),
        ("environment_geometry_and_measure", "source_context"),
        ("wind_direction", "global_context"),
        ("global_context", "velocity_field"),
    ),
    prepared_nodes=("source_context", "global_context"),
)


@dataclass
class PreparedWindInteraction:
    """Prepared source/environment context with its explicit Wind scene."""

    scene: InteractionScene
    context: PreparedResponseContext
    case_name: str
    layout_index: int
    wind_direction_deg: float
    input_sha256: str
    velocity_transform_signature: tuple[Any, ...]
    background_profile_signature: tuple[Any, ...] | None
    ownership_signature: tuple[Any, ...]


def _case_input_sha256(case: Any) -> str:
    digest = hashlib.sha256()
    metadata = {
        "case": str(getattr(case, "case", "")),
        "layout_index": int(getattr(case, "layout_index", -1)),
        "wind_direction_deg": float(getattr(case, "wind_direction_deg", float("nan"))),
        "diameter_m": float(getattr(case, "diameter_m", 80.0)),
        "hub_height_m": float(getattr(case, "hub_height_m", 70.0)),
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    for name, value in (
        ("module_centers", case.module_centers),
        ("module_present", case.module_present),
        ("module_features", case.module_features),
        ("global_context", case.global_context),
        ("support_lower_D", case.support.lower_D),
        ("support_upper_D", case.support.upper_D),
        ("support_extent_D", case.support.extent_D),
        ("environment_coords", case.env_coords),
        ("environment_features", case.env_features),
        ("environment_weights", case.env_weights),
    ):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _ownership_metadata(value: Any) -> Any:
    """Describe mutable inputs by identity and version, without reading values."""

    if torch.is_tensor(value):
        try:
            version = int(value._version)
        except RuntimeError as error:
            if "Inference tensors do not track version counter" not in str(error):
                raise
            version = None
        return (
            "tensor",
            id(value),
            value.data_ptr(),
            version,
            tuple(value.shape),
            tuple(value.stride()),
            int(value.storage_offset()),
            str(value.dtype),
            str(value.device),
            bool(value.requires_grad),
        )
    if value is None or isinstance(value, (bool, int, str)):
        return (type(value).__qualname__, value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, tuple):
        return ("tuple", tuple(_ownership_metadata(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_ownership_metadata(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (str(key), _ownership_metadata(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ),
        )
    if isinstance(value, DependencySpec):
        return (
            "dependency",
            id(value),
            tuple(
                (name, _ownership_metadata(getattr(value, name)))
                for name in value.__dataclass_fields__
            ),
        )
    return (type(value).__module__, type(value).__qualname__, id(value))


def _scene_ownership_signature(scene: InteractionScene) -> tuple[Any, ...]:
    return (
        id(scene),
        tuple(
            (name, _ownership_metadata(getattr(scene, name)))
            for name in scene.__dataclass_fields__
        ),
    )


def _context_ownership_signature(context: PreparedResponseContext) -> tuple[Any, ...]:
    fields = tuple(
        (name, _ownership_metadata(getattr(context, name)))
        for name in context.__dataclass_fields__
    )
    dynamic = tuple(
        (name, _ownership_metadata(getattr(context, name, None)))
        for name in ("owner", "dependency", "model_reference", "parameter_signature")
    )
    return (id(context), fields, dynamic)


def _velocity_transform_signature(transform: VelocityNormalizer) -> tuple[Any, ...]:
    """Bind prepared outputs and derivatives to the exact physical transform.

    ``VelocityNormalizer`` is a frozen dataclass, but its NumPy arrays remain
    mutable. Capture both object/array identity and their bytes so that an
    in-place edit, array replacement, or numerically equivalent replacement
    cannot silently change physical endpoints or local derivatives.
    """

    arrays = []
    for name in ("mean", "std", "safe_std"):
        value = np.asarray(getattr(transform, name))
        contiguous = np.ascontiguousarray(value)
        arrays.append(
            (
                name,
                id(value),
                contiguous.dtype.str,
                tuple(contiguous.shape),
                hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest(),
            )
        )
    scalars = (
        float(transform.u_ref_mps).hex(),
        int(transform.sample_count_per_row),
        int(transform.source_rows),
        int(transform.seed),
        float(transform.std_floor).hex(),
    )
    return (id(transform), tuple(arrays), scalars)


def _background_profile_signature(profile: VerticalProfileBaseline | None) -> tuple[Any, ...] | None:
    if profile is None:
        return None
    arrays = []
    for name in ("bin_centers_D", "values_mps", "counts"):
        value = np.asarray(getattr(profile, name))
        contiguous = np.ascontiguousarray(value)
        arrays.append(
            (
                name,
                id(value),
                contiguous.dtype.str,
                tuple(contiguous.shape),
                hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest(),
            )
        )
    return (
        id(profile),
        tuple(arrays),
        float(profile.z_min_D).hex(),
        float(profile.z_max_D).hex(),
    )


def _prepared_ownership_signature(
    model: WindFarmSharedInteractionModel,
    scene: InteractionScene,
    context: PreparedResponseContext,
    case_name: str,
    layout_index: int,
    wind_direction_deg: float,
    input_sha256: str,
    velocity_transform_signature: tuple[Any, ...],
    background_profile_signature: tuple[Any, ...] | None,
) -> tuple[Any, ...]:
    core_contract = (
        id(model.core),
        _ownership_metadata(getattr(model.core, "output_law", None)),
        _ownership_metadata(model.core.config),
        _ownership_metadata(getattr(model.core, "refinement_config", None)),
    )
    return (
        _scene_ownership_signature(scene),
        _context_ownership_signature(context),
        core_contract,
        _ownership_metadata(case_name),
        _ownership_metadata(layout_index),
        _ownership_metadata(wind_direction_deg),
        _ownership_metadata(input_sha256),
        velocity_transform_signature,
        background_profile_signature,
    )


def _tensor(value: Any, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)


def wind_scene_from_case(
    case: Any,
    *,
    device: torch.device | str = "cpu",
    centers: torch.Tensor | None = None,
) -> InteractionScene:
    """Build a whitelisted, target-free scene from one native row.

    Coordinates and characteristic lengths are in rotor diameters. Source
    records contain the WindFarm rotor radius and hub height; environmental
    donors retain their geometry features and centre-box measures. No native
    velocity target is read here.
    """

    target_device = torch.device(device)
    centers_t = (
        _tensor(case.module_centers[None], device=target_device)
        if centers is None
        else centers.to(device=target_device, dtype=torch.float32)
    )
    if centers_t.ndim == 2:
        centers_t = centers_t.unsqueeze(0)
    if centers_t.shape[0] != 1 or centers_t.shape[-1] != 3:
        raise ValueError("WindFarm scene centers must have shape [1,M,3] or [M,3].")

    sources = _tensor(case.module_features[None], device=target_device)
    present = _tensor(case.module_present[None], device=target_device)
    global_context = _tensor(case.global_context[None], device=target_device)
    extent = np.asarray(case.support.extent_D, dtype=np.float32)
    lengths = _tensor(extent[None], device=target_device)
    if lengths.shape != (1, 3) or np.any(extent <= 0.0):
        raise ValueError("WindFarm support must provide three positive rotor-diameter lengths.")

    # module_features[0] is rotor radius. The near/far length is its physical
    # diameter, so the core's dimensionless near blend is scaled in Wind units.
    source_lengths = 2.0 * sources[..., 0]
    if bool((source_lengths <= 0.0).any()):
        raise ValueError("WindFarm turbine diameters must be positive.")
    source_measures = torch.ones_like(present)

    environment_tokens = _tensor(case.env_features[None], device=target_device)
    environment_coords = _tensor(case.env_coords[None], device=target_device)
    environment_measures = _tensor(case.env_weights[None], device=target_device)
    environment_present = torch.ones_like(environment_measures)
    source_ids = torch.arange(centers_t.shape[1], device=target_device, dtype=torch.long)[None]

    return InteractionScene(
        sources=sources,
        context=global_context,
        centers=centers_t,
        present=present,
        lengths=lengths,
        source_lengths=source_lengths,
        dependency=WIND_INTERACTION_DEPENDENCY,
        source_measures=source_measures,
        environment_tokens=environment_tokens,
        environment_coords=environment_coords,
        environment_present=environment_present,
        environment_measures=environment_measures,
        source_ids=source_ids,
    )


def wind_receiver_features(case: Any, receivers_D: torch.Tensor) -> torch.Tensor:
    """Compute the Wind-owned seven support features with live tensor inputs."""

    if receivers_D.ndim != 3 or receivers_D.shape[0] != 1 or receivers_D.shape[-1] != 3:
        raise ValueError("WindFarm receivers must have shape [1,Q,3] in rotor diameters.")
    scale = receivers_D.new_tensor(POSITIONAL_SCALE_D)
    lower = receivers_D.new_tensor(np.asarray(case.support.lower_D, dtype=np.float32))
    upper = receivers_D.new_tensor(np.asarray(case.support.upper_D, dtype=np.float32))
    lower_distance = (receivers_D - lower) / scale
    upper_distance = (upper - receivers_D) / scale
    absolute_height = receivers_D[..., 2:3] / scale[2]
    return torch.cat((lower_distance, upper_distance, absolute_height), dim=-1)


def _wind_receiver_features_from_scene(scene: InteractionScene, receivers_D: torch.Tensor) -> torch.Tensor:
    """Recover the adapter's support box from its declared global inputs."""

    if receivers_D.ndim != 3 or receivers_D.shape[-1] != 3:
        raise ValueError("WindFarm receivers must have shape [B,Q,3] in rotor diameters.")
    if receivers_D.shape[0] != scene.context.shape[0]:
        raise ValueError("WindFarm receiver and scene batch dimensions differ.")
    scale = receivers_D.new_tensor(POSITIONAL_SCALE_D)
    lower = scene.context[:, 5:8] * scale
    extent = scene.context[:, 8:11] * scale
    upper = lower + extent
    lower_distance = (receivers_D - lower[:, None]) / scale
    upper_distance = (upper[:, None] - receivers_D) / scale
    absolute_height = receivers_D[..., 2:3] / scale[2]
    return torch.cat((lower_distance, upper_distance, absolute_height), dim=-1)


class WindFarmSharedInteractionModel(nn.Module):
    """Dataset adapter/model bridge using the shared nonlinear readout."""

    def __init__(
        self,
        *,
        velocity_transform: VelocityNormalizer,
        hidden: int = 64,
        message: int = 64,
        max_sources: int = 30,
    ) -> None:
        super().__init__()
        self.velocity_transform = velocity_transform
        self.background_profile: VerticalProfileBaseline | None = None
        self.core = NonlinearFieldReadout(
            source_width=2,
            context_width=11,
            environment_width=7,
            spatial_dim=3,
            hidden=int(hidden),
            message=int(message),
            output_width=3,
            query_width=7,
            max_sources=int(max_sources),
        )

    def prepare_case(
        self,
        case: Any,
        *,
        device: torch.device | str | None = None,
        centers: torch.Tensor | None = None,
    ) -> PreparedWindInteraction:
        target_device = next(self.parameters()).device if device is None else torch.device(device)
        scene = wind_scene_from_case(case, device=target_device, centers=centers)
        context = self.core.prepare(scene)
        case_name = str(getattr(case, "case", ""))
        layout_index = int(getattr(case, "layout_index", -1))
        wind_direction_deg = float(getattr(case, "wind_direction_deg", float("nan")))
        input_sha256 = _case_input_sha256(case)
        velocity_transform_signature = _velocity_transform_signature(self.velocity_transform)
        background_profile_signature = _background_profile_signature(self.background_profile)
        ownership_signature = _prepared_ownership_signature(
            self,
            scene,
            context,
            case_name,
            layout_index,
            wind_direction_deg,
            input_sha256,
            velocity_transform_signature,
            background_profile_signature,
        )
        return PreparedWindInteraction(
            scene=scene,
            context=context,
            case_name=case_name,
            layout_index=layout_index,
            wind_direction_deg=wind_direction_deg,
            input_sha256=input_sha256,
            velocity_transform_signature=velocity_transform_signature,
            background_profile_signature=background_profile_signature,
            ownership_signature=ownership_signature,
        )

    def _assert_prepared_owned(self, prepared: PreparedWindInteraction) -> None:
        velocity_transform_signature = _velocity_transform_signature(self.velocity_transform)
        if velocity_transform_signature != prepared.velocity_transform_signature:
            raise ValueError("WindFarm velocity transform changed; rebuild prepared state.")
        background_profile_signature = _background_profile_signature(self.background_profile)
        if background_profile_signature != prepared.background_profile_signature:
            raise ValueError("WindFarm background profile changed; rebuild prepared state.")
        current = _prepared_ownership_signature(
            self,
            prepared.scene,
            prepared.context,
            prepared.case_name,
            prepared.layout_index,
            prepared.wind_direction_deg,
            prepared.input_sha256,
            velocity_transform_signature,
            background_profile_signature,
        )
        if current != prepared.ownership_signature:
            raise ValueError("WindFarm prepared scene/context metadata changed; rebuild prepared state.")
        self.core.assert_owned(prepared.context)

    def predict_standardized(
        self,
        prepared: PreparedWindInteraction,
        receivers_D: torch.Tensor,
        *,
        receiver_features: torch.Tensor | None = None,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        self._assert_prepared_owned(prepared)
        features = (
            _wind_receiver_features_from_scene(prepared.scene, receivers_D)
            if receiver_features is None
            else receiver_features
        )
        return self.core.predict(
            prepared.context,
            receivers_D,
            receiver_features=features,
            chunk_size=int(chunk_size),
        )

    def predict_case(
        self,
        case: Any,
        prepared: PreparedWindInteraction,
        receivers_D: torch.Tensor,
        *,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        if int(getattr(case, "layout_index", -1)) != prepared.layout_index:
            raise ValueError("Prepared WindFarm state belongs to a different layout.")
        if str(getattr(case, "case", "")) != prepared.case_name:
            raise ValueError("Prepared WindFarm state belongs to a different native row.")
        if _case_input_sha256(case) != prepared.input_sha256:
            raise ValueError("WindFarm configuration/context inputs changed; rebuild prepared state.")
        return self.predict_standardized(prepared, receivers_D, chunk_size=chunk_size)

    def denormalize_tensor(self, standardized: torch.Tensor) -> torch.Tensor:
        mean = standardized.new_tensor(self.velocity_transform.mean)
        std = standardized.new_tensor(self.velocity_transform.safe_std)
        return (standardized * std + mean) * float(self.velocity_transform.u_ref_mps)

    def normalize_tensor(self, physical_mps: torch.Tensor) -> torch.Tensor:
        mean = physical_mps.new_tensor(self.velocity_transform.mean)
        std = physical_mps.new_tensor(self.velocity_transform.safe_std)
        return (physical_mps / float(self.velocity_transform.u_ref_mps) - mean) / std

    def predict_physical_case(
        self,
        case: Any,
        prepared: PreparedWindInteraction,
        receivers_D: torch.Tensor,
        *,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        return self._physical_from_standardized(
            self.predict_case(case, prepared, receivers_D, chunk_size=chunk_size),
            receivers_D,
        )

    def _physical_from_standardized(
        self, standardized: torch.Tensor, receivers_D: torch.Tensor | None = None
    ) -> torch.Tensor:
        del receivers_D
        return self.denormalize_tensor(standardized)

    def apply_increment(self, *_args: Any, **_kwargs: Any) -> None:
        self.core.apply_increment()

    def linearize_case(
        self,
        prepared: PreparedWindInteraction,
        receivers_D: torch.Tensor,
        tangent: torch.Tensor,
        *,
        wrt: str = "centers",
        receiver_features: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        self._assert_prepared_owned(prepared)
        if wrt not in ("centers", "sources", "context", "receivers"):
            raise ValueError("Local linearization supports centers, sources, context or receivers.")
        value = receivers_D if wrt == "receivers" else getattr(prepared.scene, wrt)
        if tangent.shape != value.shape:
            raise ValueError("Local tangent must match its declared configuration variable.")

        def evaluate(changed: torch.Tensor) -> torch.Tensor:
            scene = prepared.scene if wrt == "receivers" else replace(prepared.scene, **{wrt: changed})
            query = changed if wrt == "receivers" else receivers_D
            context = self.core.prepare(scene)
            features = (
                _wind_receiver_features_from_scene(scene, query)
                if receiver_features is None
                else receiver_features
            )
            return self._physical_from_standardized(
                self.core.predict(context, query, receiver_features=features), query
            )

        values, jvp = torch.autograd.functional.jvp(
            evaluate, value, tangent, create_graph=torch.is_grad_enabled()
        )
        return {
            "values": values,
            "jvp": jvp,
            "wrt": wrt,
            "units": "m/s",
            "semantics": "local AD linearization at the supplied scene; no exact finite-response guarantee",
        }


class WindFarmRefinedInteractionModel(WindFarmSharedInteractionModel):
    """Fresh nonlinear refinement family with a common TRAIN height profile."""

    def __init__(
        self,
        *,
        velocity_transform: VelocityNormalizer,
        background_profile: VerticalProfileBaseline,
        hidden: int = 64,
        message: int = 64,
        max_sources: int = 30,
        base_width: int = 16,
        router_hidden: int = 32,
        residual_scale: float = 1.0,
    ) -> None:
        nn.Module.__init__(self)
        self.velocity_transform = velocity_transform
        self.background_profile = background_profile
        self.core = RefinedNonlinearFieldReadout(
            source_width=2,
            context_width=11,
            environment_width=7,
            spatial_dim=3,
            hidden=int(hidden),
            message=int(message),
            output_width=3,
            query_width=7,
            max_sources=int(max_sources),
            base_width=int(base_width),
            router_hidden=int(router_hidden),
            residual_scale=float(residual_scale),
        )

    def profile_at_receivers(self, receivers_D: torch.Tensor) -> torch.Tensor:
        if self.background_profile is None:
            raise RuntimeError("Wind refinement requires the sealed TRAIN height-profile baseline.")
        if receivers_D.ndim != 3 or receivers_D.shape[-1] != 3:
            raise ValueError("WindFarm receivers must have shape [B,Q,3] in rotor diameters.")
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

    def _physical_from_standardized(
        self, standardized: torch.Tensor, receivers_D: torch.Tensor | None = None
    ) -> torch.Tensor:
        if receivers_D is None:
            raise ValueError("Wind residual denormalization requires its physical receiver coordinates.")
        scale = standardized.new_tensor(self.velocity_transform.safe_std)
        background = self.profile_at_receivers(receivers_D)
        return background + standardized * scale * float(self.velocity_transform.u_ref_mps)

    def normalized_field_from_physical(
        self, physical_mps: torch.Tensor, receivers_D: torch.Tensor
    ) -> torch.Tensor:
        mean = physical_mps.new_tensor(self.velocity_transform.mean)
        scale = physical_mps.new_tensor(self.velocity_transform.safe_std)
        return (physical_mps / float(self.velocity_transform.u_ref_mps) - mean) / scale

    def residual_from_physical(
        self, physical_mps: torch.Tensor, receivers_D: torch.Tensor
    ) -> torch.Tensor:
        scale = physical_mps.new_tensor(self.velocity_transform.safe_std)
        return (physical_mps - self.profile_at_receivers(receivers_D)) / (
            scale * float(self.velocity_transform.u_ref_mps)
        )

    def predict_refined_batch(
        self,
        scene: InteractionScene,
        receivers_D: torch.Tensor,
        *,
        execution_mode: str,
        phase: str,
        training_signal: bool,
        temperature: float = 1.0,
        execution_backend: str = "selected",
        chunk_size: int = 512,
        route_chunk_size: int = 512,
        gate_version: str = "hard_v1",
        gate_transition: tuple[float, float] = (0.35, 0.65),
    ) -> tuple[Any, dict[str, Any]]:
        if not isinstance(self.core, RefinedNonlinearFieldReadout):
            raise TypeError("Wind refinement adapter lost its shared nonlinear readout.")
        stage = "open" if phase == "warmup" else phase
        self.core.reset_auxiliary()
        self.core.set_execution(
            mode=execution_mode,
            phase=stage,
            temperature=float(temperature),
            training_signal=bool(training_signal),
            execution_backend=str(execution_backend),
            gate_version=str(gate_version),
            gate_transition=tuple(float(value) for value in gate_transition),
        )
        context = self.core.prepare(scene)
        features = _wind_receiver_features_from_scene(scene, receivers_D)
        prediction = self.core.read_refinement(
            context,
            receivers_D,
            receiver_features=features,
            training_signal=bool(training_signal),
            chunk_size=int(chunk_size),
            route_chunk_size=int(route_chunk_size),
        )
        auxiliary = dict(prediction.auxiliary)
        auxiliary.update(self.core.auxiliary_terms())
        return prediction, auxiliary

    def linearize_case(
        self,
        prepared: PreparedWindInteraction,
        receivers_D: torch.Tensor,
        tangent: torch.Tensor,
        *,
        wrt: str = "centers",
        receiver_features: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if not isinstance(self.core, RefinedNonlinearFieldReadout):
            raise TypeError("Wind refinement adapter lost its shared nonlinear readout.")
        policy = self.core.refinement_policy
        try:
            self.core.set_execution(
                mode=policy.mode,
                phase=policy.phase,
                threshold=policy.threshold,
                temperature=policy.temperature,
                training_signal=False,
                execution_backend=policy.execution_backend,
            )
            return super().linearize_case(
                prepared,
                receivers_D,
                tangent,
                wrt=wrt,
                receiver_features=receiver_features,
            )
        finally:
            self.core.refinement_policy = policy


__all__ = [
    "WIND_INTERACTION_DEPENDENCY",
    "PreparedWindInteraction",
    "WindFarmRefinedInteractionModel",
    "WindFarmSharedInteractionModel",
    "wind_receiver_features",
    "wind_scene_from_case",
]
