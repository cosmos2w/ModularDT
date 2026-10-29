"""Scene-prepared typed moment statistics for inherited attention.

This module contains tensor-only primitives.  It does not know about native
dataset loaders, query batches, or model construction; callers supply the
already-normalized geometry and the exact typed association axes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, field, replace
from typing import Any, Literal

import torch
from torch import Tensor, nn


MomentKind = Literal["module", "environment"]


def _stamp(tensor: Tensor) -> tuple[int, int, int]:
    try:
        version = int(tensor._version)
    except RuntimeError:
        version = -1
    return id(tensor), int(tensor.data_ptr()), version


def _tensor_tree(value: Any) -> tuple[Tensor, ...]:
    if isinstance(value, Tensor):
        return (value,)
    if hasattr(value, "__dataclass_fields__"):
        return tuple(tensor for item in fields(value) for tensor in _tensor_tree(getattr(value, item.name)))
    if isinstance(value, (tuple, list)):
        return tuple(tensor for item in value for tensor in _tensor_tree(item))
    return ()


def _unique_tensors(values: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    observed: set[int] = set()
    result = []
    for tensor in values:
        if id(tensor) not in observed:
            observed.add(id(tensor))
            result.append(tensor)
    return tuple(result)


class MomentGraphLifetime:
    """Small hook target matching the inherited core's consumed-graph guard."""

    def __init__(self) -> None:
        self.consumed = False

    def mark_consumed(self, gradient: Tensor) -> Tensor:
        self.consumed = True
        return gradient


class HeadwiseMomentLift(nn.Module):
    """Bias-free independent eight-wide lifts for each attention head."""

    def __init__(
        self,
        num_heads: int = 4,
        moment_width: int = 8,
        value_width: int = 32,
        *,
        init_seed: int = 0,
    ) -> None:
        super().__init__()
        if min(num_heads, moment_width, value_width) < 1:
            raise ValueError("Moment lift dimensions must be positive.")
        self.num_heads = int(num_heads)
        self.moment_width = int(moment_width)
        self.value_width = int(value_width)
        self.weight = nn.Parameter(torch.empty(num_heads, moment_width, value_width))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(init_seed))
        with torch.no_grad():
            for head in range(num_heads):
                nn.init.xavier_uniform_(self.weight[head], generator=generator)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 4 or values.shape[1] != self.num_heads or values.shape[-1] != self.value_width:
            raise ValueError("Moment lift values must have shape [B,H,N,32].")
        return torch.einsum("bhnd,hkd->bhnk", values, self.weight)


@dataclass(frozen=True)
class MomentRoundoffAudit:
    """Predeclared input-dtype allowance and explicit negative-variance flags."""

    positive_donor_count: Tensor  # integer [B,H,E]
    gamma_order: Tensor  # integer [B,H,E], 4*N+16
    epsilon: float
    gamma: Tensor  # [B,H,E]
    latent_raw_variance: Tensor  # [B,H,E,8]
    latent_raw_scale: Tensor  # [B,H,E,8]
    latent_allowance: Tensor  # [B,H,E,8]
    latent_clamped: Tensor  # bool [B,H,E,8]
    latent_large_negative: Tensor  # bool [B,H,E,8]
    spatial_raw_diagonal: Tensor  # [B,H,E,D]
    spatial_raw_scale: Tensor  # [B,H,E,D]
    spatial_allowance: Tensor  # [B,H,E,D], diagonal only
    spatial_clamped: Tensor  # bool [B,H,E,D]
    spatial_large_negative: Tensor  # bool [B,H,E,D]


@dataclass(frozen=True)
class TypedMomentStatistics:
    """One type's association-contracted statistics and their exact source bindings."""

    kind: MomentKind
    full_value_mean: Tensor  # [B,H,E,32], existing projected value aggregate
    lifted_value_mean: Tensor  # [B,H,E,8]
    relative_centroid: Tensor  # [B,H,E,D]
    position_value_cross: Tensor  # raw relative C, [B,H,E,D,8]
    spatial_covariance_upper: Tensor  # torch.triu_indices order, [B,H,E,D(D+1)/2]
    latent_variance: Tensor  # diagonal, [B,H,E,8]
    normalized_donor_length: Tensor  # P-weighted, [B,H,E,D]
    typed_valid: Tensor  # bool [B,H,E]
    row_mass: Tensor  # actual sum(P), [B,H,E]
    audit: MomentRoundoffAudit
    probabilities: Tensor  # original P reference [B,H,E,N]
    values: Tensor  # final typed token states [B,H,N,32]
    lift_weight: Tensor  # [H,8,32]
    normalized_donor_coordinates: Tensor  # [B,N,D]
    normalized_anchor_coordinates: Tensor  # [B,E,D]
    normalized_donor_characteristic_lengths: Tensor  # [B,N,D]
    donor_valid: Tensor  # bool [B,N]
    anchor_valid: Tensor  # bool [B,E]
    donor_ids: Tensor  # [B,N]
    anchor_ids: Tensor  # [B,E]

    def __post_init__(self) -> None:
        if self.kind not in ("module", "environment"):
            raise ValueError("Typed moment statistics must describe module or environment associations.")
        if self.probabilities.ndim != 4:
            raise ValueError("Typed moment probabilities must have shape [B,H,E,N].")
        batch, heads, anchors, donors = self.probabilities.shape
        if self.values.shape != (batch, heads, donors, 32):
            raise ValueError("Typed moment source values must have shape [B,H,N,32].")
        if self.lift_weight.shape != (heads, 8, 32):
            raise ValueError("Typed moment lifts must have shape [H,8,32].")
        coordinate_dim = self.normalized_donor_coordinates.shape[-1]
        if coordinate_dim not in (2, 3):
            raise ValueError("Typed moment geometry supports 2-D or 3-D coordinates.")
        if self.normalized_donor_coordinates.shape != (batch, donors, coordinate_dim):
            raise ValueError("Typed donor coordinates must have shape [B,N,D].")
        if self.normalized_anchor_coordinates.shape != (batch, anchors, coordinate_dim):
            raise ValueError("Typed anchor coordinates must have shape [B,E,D].")
        if self.normalized_donor_characteristic_lengths.shape != (batch, donors, coordinate_dim):
            raise ValueError("Typed donor lengths must have shape [B,N,D].")
        if self.donor_valid.shape != (batch, donors) or self.donor_valid.dtype != torch.bool:
            raise ValueError("Typed donor validity must be boolean [B,N].")
        if self.anchor_valid.shape != (batch, anchors) or self.anchor_valid.dtype != torch.bool:
            raise ValueError("Typed anchor validity must be boolean [B,E].")
        if self.donor_ids.shape != (batch, donors) or self.anchor_ids.shape != (batch, anchors):
            raise ValueError("Typed donor and anchor IDs must match their tensor axes.")
        if self.donor_ids.dtype not in (torch.int32, torch.int64) or self.anchor_ids.dtype not in (
            torch.int32, torch.int64
        ):
            raise TypeError("Typed moment IDs must use integer tensor dtypes.")
        if self.full_value_mean.shape != (batch, heads, anchors, 32):
            raise ValueError("Typed full value means must have shape [B,H,E,32].")
        if self.lifted_value_mean.shape != (batch, heads, anchors, 8):
            raise ValueError("Typed lifted means must have shape [B,H,E,8].")
        if self.relative_centroid.shape != (batch, heads, anchors, coordinate_dim):
            raise ValueError("Typed relative centroids must have shape [B,H,E,D].")
        if self.position_value_cross.shape != (batch, heads, anchors, coordinate_dim, 8):
            raise ValueError("Typed raw cross moments must have shape [B,H,E,D,8].")
        upper_width = coordinate_dim * (coordinate_dim + 1) // 2
        if self.spatial_covariance_upper.shape != (batch, heads, anchors, upper_width):
            raise ValueError("Typed spatial covariance upper triangles have an invalid width.")
        if self.latent_variance.shape != (batch, heads, anchors, 8):
            raise ValueError("Typed latent variances must have shape [B,H,E,8].")
        if self.normalized_donor_length.shape != (batch, heads, anchors, coordinate_dim):
            raise ValueError("Typed weighted donor lengths must have shape [B,H,E,D].")
        if self.typed_valid.shape != (batch, heads, anchors) or self.typed_valid.dtype != torch.bool:
            raise ValueError("Typed row validity must be boolean [B,H,E].")
        if self.row_mass.shape != (batch, heads, anchors):
            raise ValueError("Typed association mass must have shape [B,H,E].")

    def tensors(self) -> tuple[Tensor, ...]:
        return _unique_tensors(_tensor_tree(self))


@dataclass(frozen=True)
class PreparedMomentHyperedges:
    """The actual scene cache shared by M/H processors and Thermal transport.

    Tensor fields are references to the live prepared factors.  The cache seal
    records tensor identity, storage and in-place versions; no payload copy is
    made for export.  ``graph_lifetime`` can be the model's existing lifetime
    guard or a ``MomentGraphLifetime`` for standalone use.
    """

    module: TypedMomentStatistics
    environment: TypedMomentStatistics
    descriptor: Tensor  # [B,E,708] or [B,E,814]
    base_projected_state: Tensor  # bias-free Z0 [B,E,128]
    module_measure: Tensor  # raw normalized native measure [B,M]
    source_carriers: Tensor | None  # [B,H,M,8], heat independent
    module_coordinates: Tensor  # normalized xi_i [B,M,D]
    environment_coordinates: Tensor  # normalized xi_a [B,E,D]
    module_ids: Tensor
    environment_ids: Tensor
    module_mask: Tensor
    environment_mask: Tensor
    source_module_coordinates: Tensor
    source_environment_coordinates: Tensor
    frame_key: Any
    scene_version: Any
    owner_token: Any
    normalizer_key: Any = None
    scene_key: Any = None
    processor_preactivation: Tensor | None = None
    processor_hidden: Tensor | None = None
    context_residual: Tensor | None = None
    processed_state: Tensor | None = None
    heat_mix_constant: Tensor | None = None  # [B,H,E,8] or None
    heat_mix_spatial: Tensor | None = None  # [B,H,E,8,D] or None
    graph_lifetime: Any = None
    _tensor_versions: tuple[tuple[int, int, int], ...] = field(
        init=False, repr=False, compare=False
    )
    _cache_seal: tuple[Any, ...] = field(init=False, repr=False, compare=False)
    _hooks_registered: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        b, h, e, _ = self.module.probabilities.shape
        if self.environment.probabilities.shape[:3] != (b, h, e):
            raise ValueError("Typed moment tensors must share batch, head and anchor axes.")
        if self.descriptor.shape[:2] != (b, e) or self.descriptor.shape[-1] not in (708, 814):
            raise ValueError("The prepared descriptor must be [B,E,708] or [B,E,814].")
        if self.base_projected_state.shape != (b, e, 128):
            raise ValueError("The bias-free base projected state must be [B,E,128].")
        if self.environment_coordinates.shape[:2] != (b, e):
            raise ValueError("Normalized environment coordinates must match the anchor axes.")
        modules = self.module.probabilities.shape[-1]
        coordinate_dim = self.module_coordinates.shape[-1]
        if self.module_coordinates.shape != (b, modules, coordinate_dim):
            raise ValueError("Normalized module coordinates must match the source M axis.")
        if self.environment_coordinates.shape[-1] != coordinate_dim or coordinate_dim not in (2, 3):
            raise ValueError("Normalized module and environment coordinates must share D=2 or D=3.")
        if self.module_ids.shape != self.module_mask.shape or self.environment_ids.shape != self.environment_mask.shape:
            raise ValueError("Prepared IDs and masks must share their physical source/anchor axes.")
        if self.module_ids.shape != (b, modules) or self.environment_ids.shape != (b, e):
            raise ValueError("Prepared IDs/masks do not match the typed source and anchor axes.")
        if self.module_mask.dtype != torch.bool or self.environment_mask.dtype != torch.bool:
            raise TypeError("Prepared source/anchor masks must be boolean.")
        if self.source_module_coordinates.shape != (b, modules, coordinate_dim):
            raise ValueError("Source module coordinate binding must have shape [B,M,D].")
        if self.source_environment_coordinates.shape != (b, e, coordinate_dim):
            raise ValueError("Source environment coordinate binding must have shape [B,E,D].")
        if self.module_measure.shape != (b, modules):
            raise ValueError("Prepared module measures must have shape [B,M].")
        if self.source_carriers is not None and self.source_carriers.shape != (b, h, modules, 8):
            raise ValueError("Prepared Thermal source carriers must have shape [B,H,M,8].")
        for name, tensor, shape in (
            ("processor preactivation", self.processor_preactivation, (b, e, 128)),
            ("processor hidden", self.processor_hidden, (b, e, 128)),
            ("context residual", self.context_residual, (b, e, 128)),
            ("processed state", self.processed_state, (b, e, 128)),
        ):
            if tensor is not None and tensor.shape != shape:
                raise ValueError(f"Prepared {name} must have shape {shape}.")
        if self.heat_mix_constant is not None and self.heat_mix_constant.shape != (b, h, e, 8):
            raise ValueError("Prepared constant heat mixes must have shape [B,H,E,8].")
        if self.heat_mix_spatial is not None and self.heat_mix_spatial.shape != (b, h, e, 8, coordinate_dim):
            raise ValueError("Prepared spatial heat mixes must have shape [B,H,E,8,D].")
        if (self.heat_mix_constant is None) != (self.heat_mix_spatial is None):
            raise ValueError("Thermal mix outputs must either both be present or both be absent.")
        if self.graph_lifetime is None:
            object.__setattr__(self, "graph_lifetime", MomentGraphLifetime())
        tensors = self.tensors()
        object.__setattr__(self, "_tensor_versions", tuple(_stamp(tensor) for tensor in tensors))
        object.__setattr__(self, "_cache_seal", (
            self.frame_key, self.scene_version, self.owner_token, self.normalizer_key,
            self.scene_key, tuple(_stamp(tensor) for tensor in tensors),
        ))

    def tensors(self) -> tuple[Tensor, ...]:
        return _unique_tensors(_tensor_tree((
            self.module, self.environment, self.descriptor, self.base_projected_state,
            self.module_measure, self.source_carriers, self.module_coordinates,
            self.environment_coordinates, self.module_ids, self.environment_ids,
            self.module_mask, self.environment_mask, self.source_module_coordinates,
            self.source_environment_coordinates, self.processor_preactivation,
            self.processor_hidden, self.context_residual, self.processed_state,
            self.heat_mix_constant, self.heat_mix_spatial,
        )))

    @property
    def cache_seal(self) -> tuple[Any, ...]:
        """Opaque construction seal for owners that bind the scene cache."""
        return self._cache_seal

    def validate_unchanged(self) -> None:
        current = tuple(_stamp(tensor) for tensor in self.tensors())
        if current != self._tensor_versions:
            raise ValueError("Prepared moment/hyperedge factors changed after construction.")
        if getattr(self.graph_lifetime, "consumed", False):
            raise ValueError("Prepared moment/hyperedge autograd graph was consumed; rebuild the scene cache.")

    def validate_binding(
        self,
        *,
        frame_key: Any = None,
        scene_version: Any = None,
        owner_token: Any = None,
        normalizer_key: Any = None,
        scene_key: Any = None,
        module_ids: Tensor | None = None,
        environment_ids: Tensor | None = None,
        module_coordinates: Tensor | None = None,
        environment_coordinates: Tensor | None = None,
    ) -> None:
        self.validate_unchanged()
        for name, expected, actual in (
            ("frame", frame_key, self.frame_key),
            ("scene version", scene_version, self.scene_version),
            ("normalizer", normalizer_key, self.normalizer_key),
            ("scene", scene_key, self.scene_key),
        ):
            if expected is not None and expected != actual:
                raise ValueError(f"Prepared moments use a different {name} binding.")
        if owner_token is not None and owner_token is not self.owner_token:
            raise ValueError("Prepared moments belong to a different scene-cache owner.")
        for name, expected, actual in (
            ("module IDs", module_ids, self.module_ids),
            ("environment IDs", environment_ids, self.environment_ids),
            ("module coordinates", module_coordinates, self.source_module_coordinates),
            ("environment coordinates", environment_coordinates, self.source_environment_coordinates),
        ):
            if expected is not None and expected is not actual:
                raise ValueError(f"Prepared moments use different {name} tensor objects.")

    def register_lifetime_hooks(self, lifetime: Any | None = None) -> None:
        """Make backward through any newly prepared differentiable output consume the owner graph."""
        if self._hooks_registered:
            return
        target = self.graph_lifetime if lifetime is None else lifetime
        hook = getattr(target, "mark_consumed", None)
        if not callable(hook):
            raise TypeError("The prepared graph lifetime must expose mark_consumed(gradient).")
        new_outputs = (
            self.module.full_value_mean, self.module.lifted_value_mean,
            self.module.relative_centroid, self.module.position_value_cross,
            self.module.spatial_covariance_upper, self.module.latent_variance,
            self.module.normalized_donor_length, self.module.row_mass,
            self.module.audit.latent_raw_variance, self.module.audit.latent_raw_scale,
            self.module.audit.gamma, self.module.audit.latent_allowance, self.module.audit.spatial_raw_diagonal,
            self.module.audit.spatial_raw_scale, self.module.audit.spatial_allowance,
            self.environment.full_value_mean, self.environment.lifted_value_mean,
            self.environment.relative_centroid, self.environment.position_value_cross,
            self.environment.spatial_covariance_upper, self.environment.latent_variance,
            self.environment.normalized_donor_length, self.environment.row_mass,
            self.environment.audit.latent_raw_variance, self.environment.audit.latent_raw_scale,
            self.environment.audit.gamma, self.environment.audit.latent_allowance, self.environment.audit.spatial_raw_diagonal,
            self.environment.audit.spatial_raw_scale, self.environment.audit.spatial_allowance,
            self.descriptor, self.base_projected_state, self.processor_preactivation,
            self.processor_hidden, self.context_residual, self.processed_state,
            self.heat_mix_constant, self.heat_mix_spatial,
        )
        for tensor in _unique_tensors(_tensor_tree(new_outputs)):
            if tensor.requires_grad:
                tensor.register_hook(hook)
        object.__setattr__(self, "graph_lifetime", target)
        object.__setattr__(self, "_hooks_registered", True)

    def with_processor_outputs(
        self,
        *,
        preactivation: Tensor,
        hidden: Tensor,
        context_residual: Tensor,
        processed_state: Tensor,
        heat_mix_constant: Tensor | None,
        heat_mix_spatial: Tensor | None,
        graph_lifetime: Any | None = None,
    ) -> PreparedMomentHyperedges:
        """Return a fresh sealed package; never mutate an existing cache in place."""
        result = replace(
            self,
            processor_preactivation=preactivation,
            processor_hidden=hidden,
            context_residual=context_residual,
            processed_state=processed_state,
            heat_mix_constant=heat_mix_constant,
            heat_mix_spatial=heat_mix_spatial,
            graph_lifetime=self.graph_lifetime if graph_lifetime is None else graph_lifetime,
        )
        result.register_lifetime_hooks()
        return result


@dataclass(frozen=True)
class PreparedMomentHyperedgeView:
    """Read-only metadata view over the actual prepared arrays; contains no copies."""

    prepared: PreparedMomentHyperedges
    schema: str = "prepared_moment_hyperedge_view_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedMomentHyperedges):
            raise TypeError("Moment export views must reference the prepared cache itself.")
        self.prepared.validate_unchanged()

    @property
    def descriptor(self) -> Tensor:
        return self.prepared.descriptor

    @property
    def module(self) -> TypedMomentStatistics:
        return self.prepared.module

    @property
    def environment(self) -> TypedMomentStatistics:
        return self.prepared.environment

    @property
    def module_ids(self) -> Tensor:
        return self.prepared.module_ids

    @property
    def environment_ids(self) -> Tensor:
        return self.prepared.environment_ids

    @property
    def score_stage(self) -> str:
        return "last_em_ee_scores"

    @property
    def payload_stage(self) -> str:
        return "final_contextualized_token_values"

    def validate_unchanged(self) -> None:
        self.prepared.validate_unchanged()


def export_prepared_moment_hyperedges(
    prepared: PreparedMomentHyperedges,
) -> PreparedMomentHyperedgeView:
    """Expose the canonical cache through a frozen view without recomputation or copying."""
    return PreparedMomentHyperedgeView(prepared)


def _expect(shape: tuple[int, ...], observed: Tensor, name: str) -> None:
    if tuple(observed.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(observed.shape)}; expected {shape}.")


def compute_typed_moments(
    *,
    kind: MomentKind,
    values: Tensor,
    full_value_mean: Tensor,
    probabilities: Tensor,
    lift_weight: Tensor,
    normalized_donor_coordinates: Tensor,
    normalized_anchor_coordinates: Tensor,
    normalized_donor_characteristic_lengths: Tensor,
    donor_valid: Tensor,
    anchor_valid: Tensor,
    donor_ids: Tensor,
    anchor_ids: Tensor,
) -> TypedMomentStatistics:
    """Compute one type's typed moments from donor-only lifts and exact P rows.

    Input tensor dtype controls arithmetic (FP32 production; explicit FP64
    calls provide the independent reference path).  Probabilities are masked
    by their declared validity but never renormalized.
    """
    if kind not in ("module", "environment"):
        raise ValueError("Typed moment kind must be module or environment.")
    if probabilities.dtype not in (torch.float32, torch.float64):
        raise TypeError("Moment contractions require FP32 production or FP64 reference inputs.")
    if not probabilities.is_floating_point():
        raise TypeError("Association probabilities must be floating point.")
    if probabilities.device.type not in ("cpu", "cuda"):
        raise ValueError("Moment contractions currently support CPU and CUDA tensors.")
    if values.ndim != 4 or probabilities.ndim != 4:
        raise ValueError("Values and typed associations must have shapes [B,H,N,32] and [B,H,E,N].")
    batch, heads, donors, value_width = values.shape
    if value_width != 32:
        raise ValueError("Round5 retains the full width-32 typed value mean.")
    p_batch, p_heads, anchors, p_donors = probabilities.shape
    if (batch, heads, donors) != (p_batch, p_heads, p_donors):
        raise ValueError("Typed values and P must share batch, head and donor axes.")
    coordinate_dim = normalized_donor_coordinates.shape[-1]
    if coordinate_dim not in (2, 3):
        raise ValueError("Moment geometry supports the native 2-D or 3-D coordinate frames.")
    _expect((batch, heads, anchors, 32), full_value_mean, "full_value_mean")
    _expect((heads, 8, 32), lift_weight, "lift_weight")
    _expect((batch, donors, coordinate_dim), normalized_donor_coordinates, "normalized_donor_coordinates")
    _expect((batch, anchors, coordinate_dim), normalized_anchor_coordinates, "normalized_anchor_coordinates")
    _expect((batch, donors, coordinate_dim), normalized_donor_characteristic_lengths,
            "normalized_donor_characteristic_lengths")
    _expect((batch, donors), donor_valid, "donor_valid")
    _expect((batch, anchors), anchor_valid, "anchor_valid")
    _expect((batch, donors), donor_ids, "donor_ids")
    _expect((batch, anchors), anchor_ids, "anchor_ids")
    if donor_valid.dtype != torch.bool or anchor_valid.dtype != torch.bool:
        raise TypeError("Moment source and anchor masks must be boolean.")
    if donor_ids.dtype not in (torch.int32, torch.int64) or anchor_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("Moment source and anchor IDs must be integer tensors.")
    tensors = (values, full_value_mean, lift_weight, normalized_donor_coordinates,
               normalized_anchor_coordinates, normalized_donor_characteristic_lengths,
               donor_valid, anchor_valid, donor_ids, anchor_ids)
    if any(tensor.device != probabilities.device for tensor in tensors):
        raise ValueError("All typed moment inputs must share the P device.")

    dtype = probabilities.dtype
    device_type = probabilities.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        p = probabilities
        allowed = anchor_valid[:, None, :, None] & donor_valid[:, None, None, :]
        p = p * allowed.to(dtype)
        source_values = values.to(dtype)
        weight = lift_weight.to(dtype)
        xi = normalized_donor_coordinates.to(dtype)
        xi_anchor = normalized_anchor_coordinates.to(dtype)
        length = normalized_donor_characteristic_lengths.to(dtype)
        u = torch.einsum("bhnd,hkd->bhnk", source_values, weight)

        # A deterministic type-local shift, shared across its anchor rows.
        # The integer choice is discrete, while the gathered coordinate/value
        # remains live in autograd.
        has_donor = donor_valid.any(dim=-1)
        first_index = donor_valid.to(torch.int64).argmax(dim=-1)
        batch_index = torch.arange(batch, device=p.device)
        c_x = xi[batch_index, first_index]
        c_x = torch.where(has_donor[:, None], c_x, torch.zeros_like(c_x))
        c_u = u.gather(2, first_index[:, None, None, None].expand(batch, heads, 1, 8)).squeeze(2)
        c_u = torch.where(has_donor[:, None, None], c_u, torch.zeros_like(c_u))

        # Donor-only lifts.  The largest value lift is [B,H,N,D*8]; no
        # [B,H,E,N,D,8] anchor/donor/value tensor is formed.
        z_x = xi - c_x[:, None, :]
        z_u = u - c_u[:, :, None, :]
        position_value = (xi[:, None, :, :, None] * u[:, :, :, None, :]).reshape(
            batch, heads, donors, coordinate_dim * 8
        )
        z_x_outer = z_x[..., :, None] * z_x[..., None, :]
        tri = torch.triu_indices(coordinate_dim, coordinate_dim, device=p.device)
        donor_lift = torch.cat((
            u,
            position_value,
            u.square(),
            z_u,
            z_u.square(),
            z_x[:, None].expand(batch, heads, donors, coordinate_dim),
            z_x_outer[:, None].expand(batch, heads, donors, coordinate_dim, coordinate_dim).flatten(-2),
            length[:, None].expand(batch, heads, donors, coordinate_dim),
        ), dim=-1)
        aggregate = torch.einsum("bhen,bhnf->bhef", p, donor_lift)

        cursor = 0
        weighted_u = aggregate[..., cursor : cursor + 8]
        cursor += 8
        weighted_xu = aggregate[..., cursor : cursor + coordinate_dim * 8].reshape(
            batch, heads, anchors, coordinate_dim, 8
        )
        cursor += coordinate_dim * 8
        weighted_u2 = aggregate[..., cursor : cursor + 8]
        cursor += 8
        weighted_zu = aggregate[..., cursor : cursor + 8]
        cursor += 8
        weighted_zu2 = aggregate[..., cursor : cursor + 8]
        cursor += 8
        weighted_zx = aggregate[..., cursor : cursor + coordinate_dim]
        cursor += coordinate_dim
        weighted_zxx = aggregate[..., cursor : cursor + coordinate_dim * coordinate_dim].reshape(
            batch, heads, anchors, coordinate_dim, coordinate_dim
        )
        cursor += coordinate_dim * coordinate_dim
        weighted_length = aggregate[..., cursor : cursor + coordinate_dim]
        if cursor + coordinate_dim != donor_lift.shape[-1]:
            raise RuntimeError("Internal donor moment-lift offsets are inconsistent.")

        row_mass = p.sum(dim=-1)
        typed_valid = (row_mass > 0) & anchor_valid[:, None, :]
        mass = row_mass
        c_rel = c_x[:, None, :] - xi_anchor
        c_rel = c_rel[:, None].expand(batch, heads, anchors, coordinate_dim)
        c_u_row = c_u[:, :, None, :]

        # Stable means preserve the actual row mass instead of assuming one.
        lifted_mean = weighted_zu + mass[..., None] * c_u_row
        relative_centroid = weighted_zx + mass[..., None] * c_rel
        cross = weighted_xu - xi_anchor[:, None, :, :, None] * lifted_mean[..., None, :]

        # Reconstruct the literal second moment about zero using the live
        # donor shift and all row-mass drift terms.
        latent_raw_second = (
            weighted_zu2 + 2.0 * c_u_row * weighted_zu + mass[..., None] * c_u_row.square()
        )
        latent_variance_raw = (
            weighted_zu2 - weighted_zu.square()
            + 2.0 * c_u_row * weighted_zu * (1.0 - mass[..., None])
            + mass[..., None] * (1.0 - mass[..., None]) * c_u_row.square()
        )

        # The donor lift represents E[(xi-c)(xi-c)^T].  Expand it around the
        # anchor-relative shift c_x-xi_a, retaining the actual row mass s.
        weighted_zx_matrix = weighted_zx[..., :, None]
        c_matrix = c_rel[..., :, None]
        spatial_covariance_raw = (
            weighted_zxx - weighted_zx_matrix * weighted_zx[..., None, :]
            + (1.0 - mass)[..., None, None] * (
                weighted_zx_matrix * c_rel[..., None, :]
                + c_matrix * weighted_zx[..., None, :]
            )
            + (mass * (1.0 - mass))[..., None, None] * (c_matrix * c_rel[..., None, :])
        )
        spatial_diagonal = spatial_covariance_raw.diagonal(dim1=-2, dim2=-1)
        spatial_diagonal_raw = spatial_diagonal
        spatial_raw_second = (
            weighted_zxx.diagonal(dim1=-2, dim2=-1)
            + 2.0 * weighted_zx * c_rel
            + mass[..., None] * c_rel.square()
        )

        epsilon = torch.finfo(dtype).eps
        # Use the declared full donor count as an input-independent upper bound
        # on positive donors.  This keeps the allowance conservative without
        # introducing a data-dependent device-to-host synchronization.
        max_gamma_order = 4.0 * donors + 16.0
        if 1.0 - max_gamma_order * epsilon <= 0:
            raise ValueError("The declared moment roundoff order is too large for the input dtype.")
        positive_count = (p > 0).sum(dim=-1)
        gamma_order = 4 * positive_count + 16
        gamma_order_float = gamma_order.to(dtype)
        gamma = (gamma_order_float * epsilon) / (1.0 - gamma_order_float * epsilon)
        tiny = torch.finfo(dtype).tiny
        latent_scale = torch.maximum(weighted_u2.abs(), lifted_mean.square()).clamp_min(tiny)
        spatial_scale = torch.maximum(spatial_raw_second.abs(), relative_centroid.square()).clamp_min(tiny)
        latent_allowance = gamma[..., None] * latent_scale
        spatial_allowance = gamma[..., None] * spatial_scale
        latent_large = latent_variance_raw < -latent_allowance
        spatial_large = spatial_diagonal < -spatial_allowance
        latent_clamped = (latent_variance_raw < 0) & ~latent_large
        spatial_clamped = (spatial_diagonal < 0) & ~spatial_large
        latent_variance = torch.where(latent_clamped, torch.zeros_like(latent_variance_raw), latent_variance_raw)
        spatial_diagonal = torch.where(spatial_clamped, torch.zeros_like(spatial_diagonal), spatial_diagonal)
        raw_upper = spatial_covariance_raw[..., tri[0], tri[1]]
        diag_mask = tri[0] == tri[1]
        upper_diagonal = spatial_diagonal.index_select(-1, tri[0])
        spatial_upper = torch.where(diag_mask, upper_diagonal, raw_upper)

        full_mean = torch.where(typed_valid[..., None], full_value_mean.to(dtype),
                                torch.zeros_like(full_value_mean, dtype=dtype))
        lifted_mean = torch.where(typed_valid[..., None], lifted_mean, torch.zeros_like(lifted_mean))
        relative_centroid = torch.where(typed_valid[..., None], relative_centroid, torch.zeros_like(relative_centroid))
        cross = torch.where(typed_valid[..., None, None], cross, torch.zeros_like(cross))
        spatial_upper = torch.where(typed_valid[..., None], spatial_upper, torch.zeros_like(spatial_upper))
        latent_variance = torch.where(typed_valid[..., None], latent_variance, torch.zeros_like(latent_variance))
        weighted_length = torch.where(typed_valid[..., None], weighted_length, torch.zeros_like(weighted_length))
        latent_allowance = torch.where(typed_valid[..., None], latent_allowance, torch.zeros_like(latent_allowance))
        spatial_allowance = torch.where(typed_valid[..., None], spatial_allowance, torch.zeros_like(spatial_allowance))
        latent_raw_scale = torch.where(typed_valid[..., None], latent_scale, torch.zeros_like(latent_scale))
        spatial_raw_scale = torch.where(typed_valid[..., None], spatial_scale, torch.zeros_like(spatial_scale))
        latent_variance_raw = torch.where(typed_valid[..., None], latent_variance_raw,
                                         torch.zeros_like(latent_variance_raw))
        spatial_diagonal = torch.where(typed_valid[..., None], spatial_diagonal,
                                       torch.zeros_like(spatial_diagonal))
        latent_clamped = latent_clamped & typed_valid[..., None]
        spatial_clamped = spatial_clamped & typed_valid[..., None]
        latent_large = latent_large & typed_valid[..., None]
        spatial_large = spatial_large & typed_valid[..., None]

    return TypedMomentStatistics(
        kind=kind,
        full_value_mean=full_mean,
        lifted_value_mean=lifted_mean,
        relative_centroid=relative_centroid,
        position_value_cross=cross,
        spatial_covariance_upper=spatial_upper,
        latent_variance=latent_variance,
        normalized_donor_length=weighted_length,
        typed_valid=typed_valid,
        row_mass=row_mass,
        audit=MomentRoundoffAudit(
            positive_donor_count=positive_count,
            gamma_order=gamma_order,
            epsilon=epsilon,
            gamma=gamma,
            latent_raw_variance=latent_variance_raw,
            latent_raw_scale=latent_raw_scale,
            latent_allowance=latent_allowance,
            latent_clamped=latent_clamped,
            latent_large_negative=latent_large,
            spatial_raw_diagonal=torch.where(typed_valid[..., None], spatial_diagonal_raw,
                                             torch.zeros_like(spatial_diagonal_raw)),
            spatial_raw_scale=spatial_raw_scale,
            spatial_allowance=spatial_allowance,
            spatial_clamped=spatial_clamped,
            spatial_large_negative=spatial_large,
        ),
        probabilities=probabilities,
        values=values,
        lift_weight=lift_weight,
        normalized_donor_coordinates=normalized_donor_coordinates,
        normalized_anchor_coordinates=normalized_anchor_coordinates,
        normalized_donor_characteristic_lengths=normalized_donor_characteristic_lengths,
        donor_valid=donor_valid,
        anchor_valid=anchor_valid,
        donor_ids=donor_ids,
        anchor_ids=anchor_ids,
    )


def build_moment_descriptor(
    module: TypedMomentStatistics,
    environment: TypedMomentStatistics,
    *,
    normalized_anchor_coordinates: Tensor,
    normalized_anchor_characteristic_lengths: Tensor,
    operating_state: Tensor,
) -> Tensor:
    """Concatenate the fixed 708/814 descriptor in the guide's order."""
    if module.kind != "module" or environment.kind != "environment":
        raise ValueError("Moment descriptor blocks must be ordered module then environment.")
    batch, heads, anchors, _ = module.full_value_mean.shape
    if environment.full_value_mean.shape[:3] != (batch, heads, anchors):
        raise ValueError("Module and environment descriptors must share [B,H,E].")
    coordinate_dim = normalized_anchor_coordinates.shape[-1]
    if coordinate_dim not in (2, 3):
        raise ValueError("Moment descriptors support only 2-D or 3-D geometry.")
    if normalized_anchor_characteristic_lengths.shape != (batch, anchors, coordinate_dim):
        raise ValueError("Anchor characteristic lengths must be normalized [B,E,D].")
    if operating_state.shape != (batch, 128):
        raise ValueError("The existing operating state must be [B,128].")

    def type_block(stats: TypedMomentStatistics) -> Tensor:
        means = stats.full_value_mean.permute(0, 2, 1, 3).reshape(batch, anchors, heads * 32)
        per_head = torch.cat((
            stats.lifted_value_mean,
            stats.relative_centroid,
            stats.position_value_cross.flatten(start_dim=-2),
            stats.spatial_covariance_upper,
            stats.latent_variance,
            stats.typed_valid[..., None].to(stats.full_value_mean.dtype),
            stats.normalized_donor_length,
        ), dim=-1)
        per_head = per_head.permute(0, 2, 1, 3).reshape(batch, anchors, -1)
        return torch.cat((means, per_head), dim=-1)

    descriptor = torch.cat((
        normalized_anchor_coordinates,
        normalized_anchor_characteristic_lengths,
        operating_state[:, None, :].expand(batch, anchors, 128),
        type_block(module),
        type_block(environment),
    ), dim=-1)
    expected_width = 708 if coordinate_dim == 2 else 814
    if descriptor.shape != (batch, anchors, expected_width):
        raise RuntimeError(f"Fixed moment descriptor width is {descriptor.shape[-1]}, expected {expected_width}.")
    return descriptor


__all__ = [
    "HeadwiseMomentLift",
    "MomentGraphLifetime",
    "MomentRoundoffAudit",
    "PreparedMomentHyperedgeView",
    "PreparedMomentHyperedges",
    "TypedMomentStatistics",
    "build_moment_descriptor",
    "compute_typed_moments",
    "export_prepared_moment_hyperedges",
]
