"""Thermal native adapter for the shared content-hypergraph core.

This adapter owns the prescribed Thermal feature transform, physical units,
native-grid union, curl stencil, and affine heat interface. It contains no
legacy predictor and never reads target tensors while preparing a scene.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import torch
from torch import nn

from honf_forward_core.content_hypergraph.contracts import (
    CoreOutput,
    PreparedOrganization,
    QueryBatch,
    SceneBatch,
)
from honf_forward_core.content_hypergraph.model import ContentHypergraphCore
from honf_forward_core.content_hypergraph.response import PreparedAffineQuery

from .flow_curl import (
    NativeCurlReadStencil,
    build_native_curl_read_stencil,
    physical_uvp_native_curl,
)
from .native_observations import (
    NativeGridUnion,
    NativeStencil,
    build_native_solid_mask,
    interpolate_native,
    merge_native_grid_receiver_union,
    native_cell_center_stencil,
    thermal_context_tensors,
)

THERMAL_CONTENT_ADAPTER_SCHEMA = "thermal_content_hypergraph_adapter_v1"
THERMAL_CONTENT_COORDINATE_FRAME = "thermal_physical_rectangular_xy_v1"
THERMAL_CONTENT_FEATURE_SCHEMA = "thermal_geometry_context_e192_v1"
THERMAL_CONTENT_MODULE_FEATURES = 8
THERMAL_CONTENT_ENVIRONMENT_FEATURES = 8
THERMAL_CONTENT_OPERATING_FEATURES = 14
THERMAL_CONTENT_ENVIRONMENT_SHAPE = (8, 24)
THERMAL_CONTENT_SURFACE_POINTS = 16
THERMAL_CONTENT_NATIVE_THETA_COUNT = 64
THERMAL_CONTENT_FIELD_ORDER = ("u", "v", "p", "omega", "temperature")
ROUND5_MOMENT_CORE_SCHEMA = "scene_prepared_moment_hyperedge_core_v1"
ROUND6_READOUT_CORE_SCHEMA = "hyperedge_readout_core_v1"


def _iag_core_from_config(config: Mapping[str, Any]) -> nn.Module:
    if config.get("schema") == ROUND6_READOUT_CORE_SCHEMA:
        from honf_forward_core.inherited_attention.readout_model import HyperedgeReadoutCoreV1

        return HyperedgeReadoutCoreV1.from_config(dict(config))
    if config.get("schema") == ROUND5_MOMENT_CORE_SCHEMA:
        from honf_forward_core.inherited_attention.moment_model import MomentHyperedgeCoreV1

        return MomentHyperedgeCoreV1.from_config(dict(config))
    from honf_forward_core.inherited_attention.model import InheritedAttentionCoreV1

    return InheritedAttentionCoreV1.from_config(dict(config))


def _stats_json(stats: Mapping[str, Any]) -> dict[str, list[float] | float | int | str | bool | None]:
    result = {}
    for name, value in stats.items():
        array = np.asarray(value)
        result[str(name)] = array.tolist() if array.ndim else array.item()
    return result


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _same_value(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_same_value(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, (tuple, list))
            and isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(_same_value(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


@dataclass
class PreparedThermalContentHypergraph:
    """One live scene organization and shared native receiver union."""

    owner: int
    structure: Mapping[str, Any]
    structure_snapshot: Mapping[str, Any]
    normalizer_key: str
    normalization_stats_snapshot: Mapping[str, Any]
    adapter_transform_snapshot: Mapping[str, torch.Tensor]
    fluid_xy: torch.Tensor
    fluid_xy_snapshot: torch.Tensor
    local_query_points: torch.Tensor | None
    local_query_points_snapshot: torch.Tensor | None
    scene_batch: SceneBatch
    organization: PreparedOrganization
    query_batch: QueryBatch
    union: NativeGridUnion
    stencils: Mapping[str, NativeStencil]
    native_solid_mask: torch.Tensor
    native_solid_mask_snapshot: torch.Tensor
    curl_stencil: NativeCurlReadStencil
    source_present: torch.Tensor
    lengths: torch.Tensor
    interface_delta: torch.Tensor
    interface_conductivity: torch.Tensor
    role_shapes: Mapping[str, tuple[int, ...]]
    ntheta: int
    scene_key: Any
    affine_query: PreparedAffineQuery | None = None


class ThermalContentHypergraphAdapter(nn.Module):
    """Self-contained A-D/A-H6 Thermal model with physical native extraction."""

    def __init__(
        self,
        normalization_stats: Mapping[str, Any],
        *,
        core_config: Mapping[str, Any] | None = None,
        core: ContentHypergraphCore | None = None,
        nx: int = 128,
        ny: int = 64,
        environment_nx: int = 24,
        environment_ny: int = 8,
    ) -> None:
        super().__init__()
        if nx < 2 or ny < 2 or (environment_nx, environment_ny) != (24, 8):
            raise ValueError("Thermal uses a native grid and the fixed 24x8 geometry-only environment catalogue.")
        stats = _stats_json(normalization_stats)
        # The shared HDF5 TRAIN fitter retains the legacy singular interface
        # statistic names; the existing joint adapter canonicalizes these at
        # its boundary too. Keep both forms readable in checkpoint metadata.
        if "interface_targets_mean" not in stats and "interface_target_mean" in stats:
            stats["interface_targets_mean"] = copy.deepcopy(stats["interface_target_mean"])
        if "interface_targets_std" not in stats and "interface_target_std" in stats:
            stats["interface_targets_std"] = copy.deepcopy(stats["interface_target_std"])
        for key, width in (("field_mean_by_channel", 5), ("field_std_by_channel", 5),
                           ("interface_targets_std", 2), ("internal_temperature_std", 1)):
            if key not in stats:
                raise ValueError(f"Thermal adapter needs TRAIN-fitted {key}.")
            value = np.asarray(stats[key], dtype=np.float64).reshape(-1)
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError(f"Thermal TRAIN statistic {key} must contain {width} finite values.")
            if key.endswith(("_std", "_std_by_channel")) and np.any(value <= 0):
                raise ValueError(f"Thermal TRAIN scale {key} must be positive.")
        if core is None:
            config = dict(core_config or {})
            required = {
                "module_input_dim": THERMAL_CONTENT_MODULE_FEATURES,
                "environment_input_dim": THERMAL_CONTENT_ENVIRONMENT_FEATURES,
                "coordinate_dim": 2,
                "query_input_dim": 0,
                "operating_input_dim": THERMAL_CONTENT_OPERATING_FEATURES,
                "output_dim": 3,
            }
            for name, expected in required.items():
                if name in config and config[name] != expected:
                    raise ValueError(f"Thermal content core requires {name}={expected}.")
                config[name] = expected
            core = ContentHypergraphCore(**config)
        core_config_value = core.get_config()
        native_retrieval_schema = core_config_value.get("schema") == "native_retrieval_core_v1"
        budgeted_native_schema = core_config_value.get("schema") == "native_budgeted_dispatch_v2"
        if native_retrieval_schema:
            from honf_forward_core.native_retrieval.core import NativeRetrievalCore

            expected_native_fields = {
                "schema", "base_model_config", "native_tau_mm", "native_tau_me",
                "native_tau_q", "native_backend", "environment_value_reader",
            }
            base_core_config = core_config_value.get("base_model_config")
            if (not isinstance(core, NativeRetrievalCore)
                    or set(core_config_value) != expected_native_fields
                    or core_config_value.get("environment_value_reader") is not False
                    or not isinstance(base_core_config, Mapping)
                    or base_core_config.get("schema") != "content_hypergraph_core_v2"
                    or base_core_config.get("architecture") != "A2-D"):
                raise ValueError("Thermal base adapter accepts only the explicit no-E native A2-D capability.")
        elif budgeted_native_schema:
            from honf_forward_core.native_retrieval.budgeted_core import BudgetedNativeCore

            expected_budgeted_fields = {
                "schema", "base_model_config", "native_backend", "routing_seed",
                "membership_bias", "lambda0", "environment_value_reader",
            }
            base_core_config = core_config_value.get("base_model_config")
            if (not isinstance(core, BudgetedNativeCore)
                    or set(core_config_value) != expected_budgeted_fields
                    or core_config_value.get("environment_value_reader") is not False
                    or not isinstance(base_core_config, Mapping)
                    or base_core_config.get("schema") != "content_hypergraph_core_v2"
                    or base_core_config.get("architecture") != "A2-D"):
                raise ValueError("Thermal base adapter accepts only the explicit no-E budgeted A2-D capability.")
        elif core_config_value.get("schema") in ("continuous_joint_native_v1", "unified_continuous_joint_native_v1", "compact_unified_joint_v1"):
            from honf_forward_core.continuous_joint.model import ContinuousJointCore
            from honf_forward_core.continuous_joint.unified import UnifiedJointCore
            if core_config_value.get('schema')=='unified_continuous_joint_native_v1':
                from honf_forward_core.continuous_joint.unified import UnifiedJointCore as ContinuousJointCore
            elif core_config_value.get('schema') == 'compact_unified_joint_v1':
                from honf_forward_core.continuous_joint.unified import CompactUnifiedJointCore as ContinuousJointCore
            if not isinstance(core,ContinuousJointCore):raise TypeError('Continuous capability requires ContinuousJointCore.')
            base_core_config=ContinuousJointCore.validate_config(core_config_value)
        elif core_config_value.get("schema") == "joint_native_hypergraph_v1":
            from honf_forward_core.joint_native.model import JointNativeCore

            if not isinstance(core, JointNativeCore):
                raise TypeError("Thermal joint-native capability requires JointNativeCore.")
            base_core_config = JointNativeCore.validate_config(core_config_value)
        else:
            base_core_config = core_config_value
        iag_affine_capability = (
            base_core_config.get("schema") in (
                "inherited_attention_core_v1", ROUND5_MOMENT_CORE_SCHEMA,
                ROUND6_READOUT_CORE_SCHEMA,
            )
            and base_core_config.get("architecture") == "IAG-C"
            and base_core_config.get("thermal_response") is True
            and base_core_config.get("signed_carrier_width") == 8
        )
        legacy_affine_capability = (
            type(base_core_config.get("thermal_response_rank")) is int
            and base_core_config["thermal_response_rank"] >= 1
        )
        if (
            base_core_config["module_input_dim"] != THERMAL_CONTENT_MODULE_FEATURES
            or base_core_config["environment_input_dim"] != THERMAL_CONTENT_ENVIRONMENT_FEATURES
            or base_core_config["coordinate_dim"] != 2
            or base_core_config["query_input_dim"] != 0
            or base_core_config["operating_input_dim"] != THERMAL_CONTENT_OPERATING_FEATURES
            or base_core_config["output_dim"] != 3
            or not (legacy_affine_capability or iag_affine_capability)
        ):
            raise ValueError("Thermal content core dimensions or affine response capability are incompatible.")
        self.core = core
        self.nx, self.ny = int(nx), int(ny)
        self.environment_nx, self.environment_ny = int(environment_nx), int(environment_ny)
        self.normalization_stats = stats
        self.register_buffer(
            "field_mean",
            torch.as_tensor(stats["field_mean_by_channel"], dtype=torch.float32).reshape(5),
        )
        self.register_buffer(
            "field_std",
            torch.as_tensor(stats["field_std_by_channel"], dtype=torch.float32).reshape(5),
        )
        self.register_buffer(
            "interface_target_std",
            torch.as_tensor(stats["interface_targets_std"], dtype=torch.float32).reshape(2),
        )
        self.register_buffer(
            "internal_temperature_std",
            torch.as_tensor(stats["internal_temperature_std"], dtype=torch.float32).reshape(1),
        )
        self.normalizer_key = _canonical_hash({
            "schema": THERMAL_CONTENT_FEATURE_SCHEMA,
            "coordinate_frame": THERMAL_CONTENT_COORDINATE_FRAME,
            "normalization_stats": stats,
            "native_grid_nx_ny": [self.nx, self.ny],
            "environment_grid_nx_ny": [self.environment_nx, self.environment_ny],
        })

    @property
    def architecture(self) -> str:
        return self.core.architecture

    def get_config(self) -> dict[str, Any]:
        """Return the standalone adapter constructor config and shared-core spec."""
        return {
            "schema": THERMAL_CONTENT_ADAPTER_SCHEMA,
            "coordinate_frame": THERMAL_CONTENT_COORDINATE_FRAME,
            "feature_schema": THERMAL_CONTENT_FEATURE_SCHEMA,
            "native_grid_nx_ny": [self.nx, self.ny],
            "environment_grid_nx_ny": [self.environment_nx, self.environment_ny],
            "core": self.core.get_config(),
        }

    def state_payload(self) -> dict[str, Any]:
        """Return every adapter transform needed for dataset-free inference."""
        return {
            **self.get_config(),
            "normalization_stats": _stats_json(self.normalization_stats),
            "normalizer_key": self.normalizer_key,
        }

    @classmethod
    def from_state_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
        state_dict: Mapping[str, Any] | None = None,
    ) -> ThermalContentHypergraphAdapter:
        """Reconstruct a model from checkpoint metadata without loading data."""
        if payload.get("schema") != THERMAL_CONTENT_ADAPTER_SCHEMA:
            raise ValueError("Unsupported Thermal content-hypergraph adapter payload.")
        if payload.get("coordinate_frame") != THERMAL_CONTENT_COORDINATE_FRAME:
            raise ValueError("Thermal checkpoint uses a different physical coordinate frame.")
        if payload.get("feature_schema") != THERMAL_CONTENT_FEATURE_SCHEMA:
            raise ValueError("Thermal checkpoint uses a different prescribed feature inventory.")
        grid = payload.get("native_grid_nx_ny")
        environment = payload.get("environment_grid_nx_ny")
        if not isinstance(grid, (list, tuple)) or len(grid) != 2:
            raise ValueError("Thermal checkpoint is missing its native grid dimensions.")
        if environment != [24, 8] and tuple(environment or ()) != (24, 8):
            raise ValueError("Thermal checkpoint is missing the fixed E192 environment catalogue.")
        adapter = cls(
            payload["normalization_stats"],
            core_config=dict(payload["core"]),
            nx=int(grid[0]),
            ny=int(grid[1]),
            environment_nx=int(environment[0]),
            environment_ny=int(environment[1]),
        )
        if adapter.get_config() != {
            key: payload[key] for key in adapter.get_config()
        } or adapter.normalizer_key != payload.get("normalizer_key"):
            raise ValueError("Thermal adapter payload differs from its canonical reconstructed config.")
        if state_dict is not None:
            adapter.load_state_dict(state_dict, strict=True)
        return adapter.to(device)

    def scene_batch(self, structure: Mapping[str, Any], *, scene_key: Any = None) -> SceneBatch:
        """Convert only prescribed geometry/context into the shared E192 scene."""
        inventory = thermal_context_tensors(
            structure,
            environment_nx=self.environment_nx,
            environment_ny=self.environment_ny,
            preserve_dtype=True,
        )
        centers = inventory["centers"]
        batch, _, _ = centers.shape
        env_count = self.environment_nx * self.environment_ny
        source_ids = inventory["source_ids"]
        env_ids = torch.arange(env_count, device=centers.device, dtype=torch.long)[None].expand(batch, -1)
        env_area = inventory["lengths"].prod(-1)
        measure_reference = env_area.clamp_min(torch.finfo(centers.dtype).tiny)
        radius = inventory["source_lengths"]
        local_radius = 3.0 * radius
        return SceneBatch(
            module_features=inventory["sources"],
            environment_features=inventory["environment_tokens"],
            module_coordinates=centers,
            environment_coordinates=inventory["environment_coords"],
            module_mask=inventory["present"] > 0.5,
            environment_mask=inventory["environment_present"] > 0.5,
            module_measure=inventory["source_measures"],
            environment_measure=inventory["environment_measures"],
            module_ids=source_ids,
            environment_ids=env_ids,
            local_radius=local_radius,
            operating_features=inventory["context"],
            coordinate_origin=centers.new_zeros(batch, 2),
            coordinate_scale=inventory["lengths"],
            module_measure_reference=measure_reference,
            environment_measure_reference=measure_reference,
            scene_key=scene_key,
            normalizer_key=self.normalizer_key,
            coordinate_frame_key=THERMAL_CONTENT_COORDINATE_FRAME,
        )

    def _query_union(
        self,
        scene: SceneBatch,
        fluid_xy: torch.Tensor,
        local_query_points: torch.Tensor | None,
        ntheta: int,
    ) -> tuple[QueryBatch, NativeGridUnion, dict[str, NativeStencil], torch.Tensor, NativeCurlReadStencil,
               torch.Tensor, torch.Tensor, torch.Tensor, dict[str, tuple[int, ...]]]:
        if type(ntheta) is not int or ntheta not in (THERMAL_CONTENT_SURFACE_POINTS, THERMAL_CONTENT_NATIVE_THETA_COUNT):
            raise ValueError("Thermal surface extraction requires the explicit 16-point monitor or native64 catalogue.")
        centers = scene.module_coordinates
        present = scene.module_mask.to(centers.dtype)
        lengths = scene.coordinate_scale
        if lengths is None:
            raise ValueError("Thermal native extraction requires the prescribed rectangular coordinate scale.")
        fluid_xy = torch.as_tensor(fluid_xy, device=centers.device, dtype=centers.dtype)
        if fluid_xy.ndim != 3 or fluid_xy.shape[0] != centers.shape[0] or fluid_xy.shape[-1] != 2:
            raise ValueError("Thermal fluid receivers must be [B,Q,2].")
        material = torch.as_tensor(
            # Use the core's target-free scene inputs, never a heat/field target.
            self._prepared_material(scene), device=centers.device, dtype=centers.dtype,
        )
        radius = material[:, 5:6]
        delta = torch.minimum(
            (lengths / lengths.new_tensor([self.nx, self.ny])).amin(-1, keepdim=True),
            0.15 * radius,
        )
        theta = torch.arange(
            ntheta, device=centers.device, dtype=centers.dtype
        ) * (2 * torch.pi / ntheta)
        normals = torch.stack((theta.cos(), theta.sin()), -1)
        roles = {
            "fluid": fluid_xy,
            "surface": centers[:, :, None] + radius[:, None, :, None] * normals[None, None],
            "outside": centers[:, :, None] + (radius + delta)[:, None, :, None] * normals[None, None],
        }
        if local_query_points is not None:
            local = torch.as_tensor(local_query_points, device=centers.device, dtype=centers.dtype)
            if local.ndim == 3:
                local = local[:, None].expand(-1, centers.shape[1], -1, -1)
            if local.ndim != 4 or local.shape[:2] != centers.shape[:2] or local.shape[-1] != 2:
                raise ValueError("Thermal material queries must be [B,M,Q,2] or shared [B,Q,2].")
            roles["material"] = centers[:, :, None] + radius[:, None, :, None] * local
        stencils = {
            name: native_cell_center_stencil(xy, lengths, nx=self.nx, ny=self.ny)
            for name, xy in roles.items()
        }
        solid_mask = build_native_solid_mask(
            centers, present, radius, lengths, nx=self.nx, ny=self.ny
        )
        curl = build_native_curl_read_stencil(
            fluid_xy, lengths, solid_mask, nx=self.nx, ny=self.ny
        )
        union = merge_native_grid_receiver_union(
            stencils,
            roles,
            present,
            lengths,
            nx=self.nx,
            ny=self.ny,
            mandatory_grid_indices=curl.node_indices,
            mandatory_counts=curl.unique_counts,
        )
        if union.mandatory_to_union is None:
            raise RuntimeError("Thermal native union omitted its mandatory curl node mapping.")
        mapped_mandatory = union.mandatory_to_union
        batch_index = torch.arange(centers.shape[0], device=centers.device)[:, None, None]
        curl_gather = mapped_mandatory[batch_index, curl.gather_indices]
        padding = union.grid_indices.shape[1] - mapped_mandatory.shape[1]
        if padding < 0:
            raise RuntimeError("Thermal native union is shorter than its mandatory curl node inventory.")
        if padding:
            mapped_mandatory = torch.cat(
                (mapped_mandatory, mapped_mandatory[:, :1].expand(-1, padding)), dim=1
            )
        curl_union = NativeCurlReadStencil(
            query_xy=union.grid_coordinates,
            node_indices=mapped_mandatory,
            gather_indices=curl_gather,
            solid_nodes=curl.solid_nodes,
            x_span=curl.x_span,
            y_span=curl.y_span,
            unique_counts=curl.unique_counts,
        )
        query_mask = torch.arange(union.grid_indices.shape[1], device=centers.device)[None] < union.unique_counts[:, None]
        packed = self._packed_local_pairs(scene, union.grid_coordinates, query_mask)
        query = QueryBatch(
            coordinates=union.grid_coordinates,
            query_mask=query_mask,
            query_ids=union.grid_indices,
            local_batch_indices=packed[0],
            local_query_indices=packed[1],
            local_module_indices=packed[2],
            local_weights=packed[3],
            scene_key=scene.scene_key,
            coordinate_frame_key=THERMAL_CONTENT_COORDINATE_FRAME,
            receiver_key=("thermal_native_union_v1", self.nx, self.ny, ntheta, tuple(union.grid_indices.shape)),
        )
        material_values = self._prepared_material(scene)
        material_tensor = torch.as_tensor(material_values, device=centers.device, dtype=centers.dtype)
        conductivity = (
            2.0 * material_tensor[:, 3:4] * material_tensor[:, 4:5]
            / (material_tensor[:, 3:4] + material_tensor[:, 4:5]).clamp_min(1.0e-12)
        )
        # Keep the receiver catalogue's logical shapes. Native stencils flatten
        # receiver dimensions to a single list for the shared union.
        role_shapes = {name: tuple(values.shape[1:-1]) for name, values in roles.items()}
        return query, union, union.stencils, solid_mask, curl_union, lengths, delta, conductivity, role_shapes

    @staticmethod
    def _prepared_material(scene: SceneBatch) -> torch.Tensor:
        """Recover the six prescribed coefficients from retained source features."""
        # The module descriptors contain [x/Lx,y/Ly,r/Lx,r/Ly,100*alpha_s,
        # 100*alpha_f,k_s,k_f]. Global nu and radius are in operating context.
        # Every padded slot shares the same case material inventory.
        # module features alone have the radius in physical units via their
        # normalized radii and coordinate scale.
        coords = scene.coordinate_scale
        if coords is None:
            raise ValueError("Thermal scene lost its physical rectangular coordinate scale.")
        features = scene.module_features
        radius = features[:, :1, 2] * coords[:, 0:1]
        context = scene.operating_features
        # Operating order is [Re/100,u_in,100*nu,100*alpha_s,100*alpha_f,
        # k_s,k_f,r,Lx/12,Ly/6,M/12,M/area,0,0].
        if context is None or context.shape[-1] != THERMAL_CONTENT_OPERATING_FEATURES:
            raise ValueError("Thermal scene lost the prescribed 14-wide operating context.")
        return torch.cat((
            context[:, 2:7] / context.new_tensor([100.0, 100.0, 100.0, 1.0, 1.0]),
            radius,
        ), dim=-1)

    @staticmethod
    def _packed_local_pairs(
        scene: SceneBatch,
        coordinates: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack smooth-support local rows without a BxQxM distance tensor."""
        batch_indices: list[torch.Tensor] = []
        query_indices: list[torch.Tensor] = []
        module_indices: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        centers = scene.module_coordinates
        radii = scene.local_radius
        for batch_index in range(coordinates.shape[0]):
            valid_queries = query_mask[batch_index]
            active_modules = torch.nonzero(scene.module_mask[batch_index], as_tuple=True)[0].tolist()
            for module_index in active_modules:
                delta = coordinates[batch_index] - centers[batch_index, module_index]
                ratio = torch.linalg.vector_norm(delta, dim=-1) / radii[batch_index, module_index].clamp_min(
                    torch.finfo(coordinates.dtype).tiny
                )
                selected = torch.nonzero((ratio < 1.0) & valid_queries, as_tuple=True)[0]
                if selected.numel() == 0:
                    continue
                selected_ratio = ratio[selected]
                window = (1.0 - selected_ratio).clamp_min(0.0)
                batch_indices.append(torch.full_like(selected, batch_index))
                query_indices.append(selected)
                module_indices.append(torch.full_like(selected, module_index))
                weights.append(window.pow(4) * (4.0 * selected_ratio + 1.0))
        device = coordinates.device
        if not batch_indices:
            empty = torch.empty((0,), dtype=torch.long, device=device)
            return empty, empty, empty, coordinates.new_empty((0,))
        return (
            torch.cat(batch_indices),
            torch.cat(query_indices),
            torch.cat(module_indices),
            torch.cat(weights),
        )

    def prepare_native(
        self,
        structure: Mapping[str, Any],
        fluid_xy: torch.Tensor,
        *,
        local_query_points: torch.Tensor | None = None,
        ntheta: int = THERMAL_CONTENT_NATIVE_THETA_COUNT,
        scene_key: Any = None,
        forced_frontier: Mapping[int, Any] | None = None,
        context_message_removal: Mapping[str, Any] | None = None,
    ) -> PreparedThermalContentHypergraph:
        """Prepare one shared edge bank and union for all native output roles."""
        scene_batch = self.scene_batch(structure, scene_key=scene_key)
        if context_message_removal is not None:
            if (getattr(self.core, "schema", None) not in {
                    "interaction_reused_member_routing_v1", "interaction_reused_member_routing_v2",
                    "interaction_reused_member_routing_v3"}
                    or forced_frontier is not None):
                raise ValueError("Context-message removal is supported only by the Round13-3 member-routing core.")
            organization = self.core.prepare(
                scene_batch, context_message_removal=dict(context_message_removal),
            )
        else:
            organization = (self.core.prepare(scene_batch) if forced_frontier is None else
                            self.core.prepare(scene_batch, forced_frontier=forced_frontier))
        (query, union, stencils, solid_mask, curl_stencil, lengths, delta,
         conductivity, role_shapes) = self._query_union(scene_batch, fluid_xy, local_query_points, ntheta)
        if query.scene_key != organization.scene_key:
            raise ValueError("Thermal query union and shared organization belong to different scenes.")
        adapter_transform_snapshot = {
            name: getattr(self, name).detach().clone()
            for name in ("field_mean", "field_std", "interface_target_std", "internal_temperature_std")
        }
        return PreparedThermalContentHypergraph(
            owner=id(self),
            structure=structure,
            structure_snapshot=_snapshot(structure),
            normalizer_key=self.normalizer_key,
            normalization_stats_snapshot=_snapshot(self.normalization_stats),
            adapter_transform_snapshot=adapter_transform_snapshot,
            fluid_xy=fluid_xy,
            fluid_xy_snapshot=fluid_xy.detach().clone(),
            local_query_points=local_query_points,
            local_query_points_snapshot=None if local_query_points is None else local_query_points.detach().clone(),
            scene_batch=scene_batch,
            organization=organization,
            query_batch=query,
            union=union,
            stencils=stencils,
            native_solid_mask=solid_mask,
            native_solid_mask_snapshot=solid_mask.detach().clone(),
            curl_stencil=curl_stencil,
            source_present=scene_batch.module_mask,
            lengths=lengths,
            interface_delta=delta,
            interface_conductivity=conductivity,
            role_shapes=role_shapes,
            ntheta=ntheta,
            scene_key=scene_key,
        )

    def validate_prepared(
        self,
        prepared: PreparedThermalContentHypergraph,
        *,
        structure: Mapping[str, Any] | None = None,
        fluid_xy: torch.Tensor | None = None,
        local_query_points: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(prepared, PreparedThermalContentHypergraph) or prepared.owner != id(self):
            raise ValueError("Thermal prepared native state belongs to another adapter.")
        if prepared.normalizer_key != self.normalizer_key:
            raise ValueError("Thermal normalization identity changed; rebuild the native preparation.")
        if not _same_value(self.normalization_stats, prepared.normalization_stats_snapshot):
            raise ValueError("Thermal normalization metadata changed; rebuild the native preparation.")
        if any(
            not torch.equal(getattr(self, name), snapshot)
            for name, snapshot in prepared.adapter_transform_snapshot.items()
        ):
            raise ValueError("Thermal physical transforms changed; rebuild the native preparation.")
        current_structure = prepared.structure if structure is None else structure
        if not _same_value(current_structure, prepared.structure_snapshot):
            raise ValueError("Thermal geometry or prescribed context moved; rebuild the native preparation.")
        current_fluid = prepared.fluid_xy if fluid_xy is None else fluid_xy
        if not torch.equal(current_fluid, prepared.fluid_xy_snapshot):
            raise ValueError("Thermal primary receivers changed; rebuild the native preparation.")
        current_local = prepared.local_query_points if local_query_points is None else local_query_points
        if prepared.local_query_points_snapshot is None:
            if current_local is not None:
                raise ValueError("Thermal material receiver catalogue changed; rebuild the native preparation.")
        elif current_local is None or not torch.equal(current_local, prepared.local_query_points_snapshot):
            raise ValueError("Thermal material receiver catalogue changed; rebuild the native preparation.")
        if not torch.equal(prepared.native_solid_mask, prepared.native_solid_mask_snapshot):
            raise ValueError("Thermal geometry mask changed; rebuild the native preparation.")

    def _physical_temperature(self, temperature: torch.Tensor) -> torch.Tensor:
        scale = self.field_std[4].to(device=temperature.device, dtype=temperature.dtype)
        return temperature * scale

    def _extract_native_roles(
        self,
        prepared: PreparedThermalContentHypergraph,
        temperature: torch.Tensor,
        flow_normalized: torch.Tensor | None,
        *,
        accumulation_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        if temperature.ndim == 3 and temperature.shape[-1] == 1:
            temperature = temperature[..., 0]
        if temperature.shape != prepared.union.grid_indices.shape:
            raise ValueError("Thermal response values do not align with the prepared shared receiver union.")
        physical_temperature = self._physical_temperature(temperature)
        if flow_normalized is None:
            fluid_flow = physical_temperature.new_zeros((*prepared.stencils["fluid"].valid.shape, 4))
        else:
            if accumulation_dtype is not None:
                flow_normalized = flow_normalized.to(accumulation_dtype)
            mean = self.field_mean.to(device=flow_normalized.device, dtype=flow_normalized.dtype)
            std = self.field_std.to(device=flow_normalized.device, dtype=flow_normalized.dtype)
            fluid_flow = physical_uvp_native_curl(flow_normalized, prepared.curl_stencil, mean, std)
        fluid_temperature = interpolate_native(physical_temperature[..., None], prepared.stencils["fluid"])
        pred_field = torch.cat((fluid_flow, fluid_temperature), dim=-1)

        surface_shape = prepared.role_shapes["surface"]
        surface_temperature = interpolate_native(
            physical_temperature[..., None], prepared.stencils["surface"]
        ).reshape(prepared.source_present.shape[0], *surface_shape, 1)
        outside_temperature = interpolate_native(
            physical_temperature[..., None], prepared.stencils["outside"]
        ).reshape(prepared.source_present.shape[0], *prepared.role_shapes["outside"], 1)
        delta = prepared.interface_delta[:, None, :, None].to(surface_temperature)
        conductivity = prepared.interface_conductivity[:, None, :, None].to(surface_temperature)
        q_normal = -conductivity / delta.clamp_min(torch.finfo(delta.dtype).tiny) * (
            outside_temperature - surface_temperature
        )
        present = prepared.source_present[:, :, None, None].to(surface_temperature.dtype)
        interface = torch.cat((surface_temperature, q_normal), dim=-1) * present

        if "material" in prepared.stencils:
            material_shape = prepared.role_shapes["material"]
            internal = interpolate_native(
                physical_temperature[..., None], prepared.stencils["material"]
            ).reshape(prepared.source_present.shape[0], *material_shape, 1)
            internal = internal * prepared.source_present[:, :, None, None].to(internal.dtype)
        else:
            internal = physical_temperature.new_zeros(
                prepared.source_present.shape[0], prepared.source_present.shape[1], 0, 1
            )
        return {
            "pred_field": pred_field,
            "fluid_temperature": fluid_temperature,
            "pred_interface": interface,
            "pred_internal_temperature": internal,
        }

    def apply_native(
        self,
        prepared: PreparedThermalContentHypergraph,
        physical_heat: torch.Tensor,
        *,
        increment: bool = False,
        accumulation_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode nonlinear u/v/p and apply affine heat without dense kernels."""
        self.validate_prepared(prepared)
        controls = torch.as_tensor(physical_heat, device=prepared.scene_batch.module_features.device)
        if not controls.is_floating_point():
            controls = controls.to(self.field_std.dtype)
        if controls.shape != prepared.scene_batch.module_mask.shape:
            raise ValueError("Thermal physical heat controls must match the scene's padded module slots.")
        if prepared.affine_query is not None:
            response_options = ({"check_envelope": not increment}
                if getattr(self.core, "source_read_policy", "full") != "full" else {})
            output = self.core.apply_prepared_affine(
                prepared.affine_query, controls, accumulation_dtype=accumulation_dtype,
                **response_options)
            temperature = output.thermal_temperature
            flow = None if increment else output.nonlinear_field
        elif increment:
            temperature = self.core.apply_increment(
                prepared.organization,
                prepared.query_batch,
                controls,
                accumulation_dtype=accumulation_dtype,
            )
            flow = None
        else:
            output: CoreOutput = self.core.decode(
                prepared.organization,
                prepared.query_batch,
                controls=controls,
                accumulation_dtype=accumulation_dtype,
            )
            if output.thermal_temperature is None:
                raise RuntimeError("Thermal content core omitted its affine temperature output.")
            temperature = output.thermal_temperature
            flow = output.nonlinear_field
        result = self._extract_native_roles(
            prepared, temperature, flow, accumulation_dtype=accumulation_dtype
        )
        if increment:
            result["pred_field"] = torch.cat(
                (result["pred_field"][..., :4].zero_(), result["fluid_temperature"]), dim=-1
            )
        return result

    def prepare_affine_application(
        self, prepared: PreparedThermalContentHypergraph,
    ) -> PreparedThermalContentHypergraph:
        """Retain query factors explicitly for repeated heating or increments.

        The ordinary training path does not allocate this additional retained
        object. New controls then execute contractions and native extraction,
        while geometry/model/normalizer/query changes still invalidate reuse.
        """
        self.validate_prepared(prepared)
        factors = self.core.prepare_affine_query(prepared.organization, prepared.query_batch)
        return replace(prepared, affine_query=factors)

    def apply_increment(
        self,
        prepared: PreparedThermalContentHypergraph,
        delta_heat: torch.Tensor,
        *,
        accumulation_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.apply_native(
            prepared, delta_heat, increment=True, accumulation_dtype=accumulation_dtype
        )

    def heat_jvp(
        self,
        prepared: PreparedThermalContentHypergraph,
        tangent: torch.Tensor,
        *,
        accumulation_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.apply_increment(prepared, tangent, accumulation_dtype=accumulation_dtype)

    def heat_vjp(
        self,
        prepared: PreparedThermalContentHypergraph,
        cotangent: torch.Tensor,
        *,
        accumulation_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Control transpose product for a cotangent on the neural grid union.

        Use heat_vjp_native for cotangents on extracted physical observations.
        """
        self.validate_prepared(prepared)
        cotangent = torch.as_tensor(cotangent, device=prepared.query_batch.coordinates.device)
        scale = self.field_std[4].to(device=cotangent.device, dtype=cotangent.dtype)
        if prepared.affine_query is not None:
            return self.core.heat_vjp_prepared_affine(
                prepared.affine_query, cotangent * scale, accumulation_dtype=accumulation_dtype)
        return self.core.heat_vjp(
            prepared.organization,
            prepared.query_batch,
            cotangent * scale,
            accumulation_dtype=accumulation_dtype,
        )

    def heat_vjp_native(
        self,
        prepared: PreparedThermalContentHypergraph,
        cotangents: Mapping[str, torch.Tensor],
        *,
        accumulation_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Pull native observation cotangents back through extraction and factors.

        This applies the transpose of native interpolation/difference operators
        to one small grid-union variable, then contracts the response factors.
        It performs no dense K export and no per-source neural predictions.
        """
        self.validate_prepared(prepared)
        allowed = {"pred_field", "fluid_temperature", "pred_interface", "pred_internal_temperature"}
        if not cotangents or set(cotangents) - allowed:
            raise ValueError("Native heat cotangents must name actual extracted observation tensors.")
        dtype = accumulation_dtype or prepared.query_batch.coordinates.dtype
        keep_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            union = torch.zeros(prepared.union.grid_indices.shape,
                                device=prepared.query_batch.coordinates.device, dtype=dtype, requires_grad=True)
            observations = self._extract_native_roles(prepared, union, None, accumulation_dtype=dtype)
            objective = union.sum() * 0
            for name, cotangent in cotangents.items():
                cotangent = torch.as_tensor(cotangent, device=union.device, dtype=dtype)
                if cotangent.shape != observations[name].shape:
                    raise ValueError(f"Native {name} cotangent has the wrong observation axes.")
                objective = objective + (observations[name] * cotangent).sum()
            pullback, = torch.autograd.grad(objective, union, create_graph=keep_graph)
        # The native pullback already contains the physical temperature scale.
        if prepared.affine_query is not None:
            return self.core.heat_vjp_prepared_affine(prepared.affine_query, pullback,
                                                     accumulation_dtype=accumulation_dtype)
        return self.core.heat_vjp(prepared.organization, prepared.query_batch, pullback,
                                  accumulation_dtype=accumulation_dtype)

    def export_source_kernels(
        self,
        prepared: PreparedThermalContentHypergraph,
        *,
        accumulation_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Explicitly opt in to dense physical source-kernel materialization."""
        self.validate_prepared(prepared)
        kernels = self.core.export_source_kernels(
            prepared.organization,
            prepared.query_batch,
            accumulation_dtype=accumulation_dtype,
        )
        return kernels * self.field_std[4].to(device=kernels.device, dtype=kernels.dtype)


THERMAL_CONTENT_ADAPTER_V2_SCHEMA = "thermal_content_hypergraph_adapter_v2"
THERMAL_CONTENT_COORDINATE_FRAME_V2 = "thermal_physical_rectangular_xy_v2"
THERMAL_CONTENT_FEATURE_SCHEMA_V2 = "thermal_geometry_context_e192_v2"
THERMAL_CONTENT_GEOMETRY_CONTRACT_V2 = {
    "frame_origin": "[0,0]",
    "frame_extent": "[Lx,Ly]",
    "module_characteristic_length": "native_radius_r_broadcast_xy",
    "environment_characteristic_length": "native_quadrature_cell_[Lx/24,Ly/8]",
    "module_size_features": "[r/Lx,r/Ly] in module_features[2:4]",
    "local_reader_support_radius": "3*r",
    "length_pairing": "geometric_mean_per_axis",
}
THERMAL_CONTENT_NATIVE_LAW_CONTRACT_V2 = {
    "flow": "fresh_nonlinear_u_v_p_head_independent_of_heat",
    "curl": "physical_uvp_native_curl_cell_centred_v1",
    "temperature": "linear_physical_heat_response_with_exact_T_of_zero_equal_zero",
    "thermal_response": "factorized_native_affine_source_response",
    "interpolation": "native_cell_center_stencil_v1",
    "q_proxy": "minus_harmonic_conductivity_over_native_delta_times_outside_minus_surface_temperature",
    "response_addendum": "same_original_train_thermal_families_and_native_objective_v1",
}


def _content_hypergraph_core_v2_type() -> type[nn.Module]:
    """Load the additive v2 core without changing v1 import/replay behavior."""
    from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2

    return ContentHypergraphCoreV2


class ThermalContentHypergraphAdapterV2(ThermalContentHypergraphAdapter):
    """Thermal v2 adapter with explicit physical object and cell lengths.

    Native flow conversion, interpolation, curl, flux proxy, and affine heat
    operations are inherited verbatim from the v1 adapter. Only the versioned
    core, frame identity, and geometry inputs differ.
    """

    def __init__(
        self,
        normalization_stats: Mapping[str, Any],
        *,
        core_config: Mapping[str, Any] | None = None,
        core: nn.Module | None = None,
        nx: int = 128,
        ny: int = 64,
        environment_nx: int = 24,
        environment_ny: int = 8,
    ) -> None:
        if (environment_nx, environment_ny) != (24, 8):
            raise ValueError("Thermal v2 requires the fixed 24x8 geometry-only environment catalogue.")
        core_type = _content_hypergraph_core_v2_type()
        if core is None:
            config = dict(core_config or {})
            native_schema = config.get("schema") == "native_retrieval_core_v1"
            budgeted_schema = config.get("schema") == "native_budgeted_dispatch_v2"
            joint_schema = config.get("schema") == "joint_native_hypergraph_v1"
            continuous_schema = config.get("schema") in ("continuous_joint_native_v1", "unified_continuous_joint_native_v1", "compact_unified_joint_v1")
            if continuous_schema:
                from honf_forward_core.continuous_joint.model import ContinuousJointCore
                if config.get('schema')=='unified_continuous_joint_native_v1':
                    from honf_forward_core.continuous_joint.unified import UnifiedJointCore as ContinuousJointCore
                elif config.get('schema') == 'compact_unified_joint_v1':
                    from honf_forward_core.continuous_joint.unified import CompactUnifiedJointCore as ContinuousJointCore
                base_config = ContinuousJointCore.validate_config(config)
                config_type = ContinuousJointCore
            elif joint_schema:
                from honf_forward_core.joint_native.model import JointNativeCore

                base_config = JointNativeCore.validate_config(config)
                config_type = JointNativeCore
            elif native_schema or budgeted_schema:
                if budgeted_schema:
                    from honf_forward_core.native_retrieval.budgeted_core import BudgetedNativeCore

                    config_type = BudgetedNativeCore
                    native_fields = {
                        "schema", "base_model_config", "native_backend", "routing_seed",
                        "membership_bias", "lambda0", "environment_value_reader",
                    }
                else:
                    from honf_forward_core.native_retrieval.core import NativeRetrievalCore

                    config_type = NativeRetrievalCore
                    native_fields = {
                        "schema", "base_model_config", "native_tau_mm", "native_tau_me",
                        "native_tau_q", "native_backend", "environment_value_reader",
                    }
                if set(config) != native_fields or config.get("environment_value_reader") is not False:
                    raise ValueError("Thermal native core config must use an exact explicit no-E schema.")
                base_config = config.get("base_model_config")
                if not isinstance(base_config, Mapping) or base_config.get("schema") != "content_hypergraph_core_v2":
                    raise ValueError("Thermal native retrieval requires its exact v2 A2-D base config.")
            else:
                if config.get("schema") != "content_hypergraph_core_v2":
                    raise ValueError("Thermal v2 requires a content_hypergraph_core_v2 config.")
                base_config = config
                config_type = core_type
            required = {
                "module_input_dim": THERMAL_CONTENT_MODULE_FEATURES,
                "environment_input_dim": THERMAL_CONTENT_ENVIRONMENT_FEATURES,
                "coordinate_dim": 2,
                "query_input_dim": 0,
                "operating_input_dim": THERMAL_CONTENT_OPERATING_FEATURES,
                "output_dim": 3,
            }
            for name, expected in required.items():
                if base_config.get(name) != expected:
                    raise ValueError(f"Thermal v2 core requires explicit {name}={expected}.")
            if (native_schema or budgeted_schema or joint_schema or continuous_schema) and base_config.get("architecture") != "A2-D":
                raise ValueError("Thermal native retrieval requires the unchanged A2-D architecture.")
            core = config_type.from_config(config)
        if not isinstance(core, core_type):
            raise TypeError("Thermal v2 adapter requires ContentHypergraphCoreV2.")
        core_config_value = core.get_config()
        native_schema = core_config_value.get("schema") == "native_retrieval_core_v1"
        budgeted_schema = core_config_value.get("schema") == "native_budgeted_dispatch_v2"
        joint_schema = core_config_value.get("schema") == "joint_native_hypergraph_v1"
        continuous_schema = core_config_value.get("schema") in ("continuous_joint_native_v1", "unified_continuous_joint_native_v1", "compact_unified_joint_v1")
        if continuous_schema:
            from honf_forward_core.continuous_joint.model import ContinuousJointCore
            if not isinstance(core,ContinuousJointCore):raise TypeError('Continuous capability requires ContinuousJointCore.')
            if core_config_value.get('schema')=='unified_continuous_joint_native_v1':
                from honf_forward_core.continuous_joint.unified import UnifiedJointCore as ContinuousJointCore
            elif core_config_value.get('schema') == 'compact_unified_joint_v1':
                from honf_forward_core.continuous_joint.unified import CompactUnifiedJointCore as ContinuousJointCore
            dimensions_config=ContinuousJointCore.validate_config(core_config_value)
        elif joint_schema:
            from honf_forward_core.joint_native.model import JointNativeCore

            if not isinstance(core, JointNativeCore):
                raise TypeError("Thermal joint-native capability requires JointNativeCore.")
            dimensions_config = JointNativeCore.validate_config(core_config_value)
        elif native_schema or budgeted_schema:
            if budgeted_schema:
                from honf_forward_core.native_retrieval.budgeted_core import BudgetedNativeCore

                config_type = BudgetedNativeCore
                native_fields = {
                    "schema", "base_model_config", "native_backend", "routing_seed",
                    "membership_bias", "lambda0", "environment_value_reader",
                }
            else:
                from honf_forward_core.native_retrieval.core import NativeRetrievalCore

                config_type = NativeRetrievalCore
                native_fields = {
                    "schema", "base_model_config", "native_tau_mm", "native_tau_me",
                    "native_tau_q", "native_backend", "environment_value_reader",
                }
            if (not isinstance(core, config_type) or set(core_config_value) != native_fields
                    or core_config_value.get("environment_value_reader") is not False):
                raise ValueError("Thermal v2 accepts only explicit no-E native-retrieval capabilities.")
            dimensions_config = core_config_value["base_model_config"]
        else:
            if core_config_value.get("schema") != "content_hypergraph_core_v2":
                raise ValueError("Thermal v2 core config has the wrong schema.")
            dimensions_config = core_config_value
        expected_dimensions = {
            "module_input_dim": THERMAL_CONTENT_MODULE_FEATURES,
            "environment_input_dim": THERMAL_CONTENT_ENVIRONMENT_FEATURES,
            "coordinate_dim": 2,
            "query_input_dim": 0,
            "operating_input_dim": THERMAL_CONTENT_OPERATING_FEATURES,
            "output_dim": 3,
        }
        if any(dimensions_config.get(name) != expected for name, expected in expected_dimensions.items()):
            raise ValueError("Thermal v2 core dimensions differ from the native Thermal feature contract.")
        if dimensions_config.get("architecture") not in (("A2-D",) if native_schema or budgeted_schema or joint_schema or continuous_schema else ("A2-D", "A2-H6")):
            raise ValueError("Thermal v2 architecture must be A2-D or A2-H6.")
        super().__init__(
            normalization_stats,
            core=core,
            nx=nx,
            ny=ny,
            environment_nx=environment_nx,
            environment_ny=environment_ny,
        )
        self.normalizer_key = _canonical_hash({
            "schema": THERMAL_CONTENT_FEATURE_SCHEMA_V2,
            "coordinate_frame": THERMAL_CONTENT_COORDINATE_FRAME_V2,
            "geometry_contract": THERMAL_CONTENT_GEOMETRY_CONTRACT_V2,
            "normalization_stats": self.normalization_stats,
            "native_grid_nx_ny": [self.nx, self.ny],
            "environment_grid_nx_ny": [self.environment_nx, self.environment_ny],
        })

    def get_config(self) -> dict[str, Any]:
        """Return the standalone v2 adapter and native-geometry contract."""
        return {
            "schema": THERMAL_CONTENT_ADAPTER_V2_SCHEMA,
            "coordinate_frame": THERMAL_CONTENT_COORDINATE_FRAME_V2,
            "feature_schema": THERMAL_CONTENT_FEATURE_SCHEMA_V2,
            "geometry_contract": copy.deepcopy(THERMAL_CONTENT_GEOMETRY_CONTRACT_V2),
            "native_law_contract": copy.deepcopy(THERMAL_CONTENT_NATIVE_LAW_CONTRACT_V2),
            "native_grid_nx_ny": [self.nx, self.ny],
            "environment_grid_nx_ny": [self.environment_nx, self.environment_ny],
            "core": self.core.get_config(),
        }

    def state_payload(self) -> dict[str, Any]:
        """Serialize all versioned transforms needed for dataset-free replay."""
        return {
            **self.get_config(),
            "normalization_stats": _stats_json(self.normalization_stats),
            "normalizer_key": self.normalizer_key,
        }

    @classmethod
    def from_state_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
        state_dict: Mapping[str, Any] | None = None,
    ) -> ThermalContentHypergraphAdapterV2:
        """Reconstruct the v2 adapter from a self-contained checkpoint payload."""
        if payload.get("schema") != THERMAL_CONTENT_ADAPTER_V2_SCHEMA:
            raise ValueError("Unsupported Thermal v2 content-hypergraph adapter payload.")
        if payload.get("coordinate_frame") != THERMAL_CONTENT_COORDINATE_FRAME_V2:
            raise ValueError("Thermal v2 checkpoint uses a different physical coordinate frame.")
        if payload.get("feature_schema") != THERMAL_CONTENT_FEATURE_SCHEMA_V2:
            raise ValueError("Thermal v2 checkpoint uses a different prescribed feature inventory.")
        if payload.get("geometry_contract") != THERMAL_CONTENT_GEOMETRY_CONTRACT_V2:
            raise ValueError("Thermal v2 checkpoint has a different object/cell length convention.")
        if payload.get("native_law_contract") != THERMAL_CONTENT_NATIVE_LAW_CONTRACT_V2:
            raise ValueError("Thermal v2 checkpoint has a different native output-law contract.")
        grid = payload.get("native_grid_nx_ny")
        environment = payload.get("environment_grid_nx_ny")
        if not isinstance(grid, (list, tuple)) or len(grid) != 2:
            raise ValueError("Thermal v2 checkpoint is missing its native grid dimensions.")
        if environment != [24, 8] and tuple(environment or ()) != (24, 8):
            raise ValueError("Thermal v2 checkpoint is missing the fixed E192 environment catalogue.")
        core_type = _content_hypergraph_core_v2_type()
        core_config = dict(payload.get("core", {}))
        if core_config.get("schema") == "native_retrieval_core_v1":
            from honf_forward_core.native_retrieval.core import NativeRetrievalCore

            core = NativeRetrievalCore.from_config(core_config)
        elif core_config.get("schema") == "native_budgeted_dispatch_v2":
            from honf_forward_core.native_retrieval.budgeted_core import BudgetedNativeCore

            core = BudgetedNativeCore.from_config(core_config)
        elif core_config.get("schema") in ("continuous_joint_native_v1", "unified_continuous_joint_native_v1", "compact_unified_joint_v1"):
            from honf_forward_core.continuous_joint.model import ContinuousJointCore
            if core_config.get('schema')=='unified_continuous_joint_native_v1':
                from honf_forward_core.continuous_joint.unified import UnifiedJointCore as ContinuousJointCore
            elif core_config.get('schema') == 'compact_unified_joint_v1':
                from honf_forward_core.continuous_joint.unified import CompactUnifiedJointCore as ContinuousJointCore
            core = ContinuousJointCore.from_config(core_config)
        elif core_config.get("schema") == "joint_native_hypergraph_v1":
            from honf_forward_core.joint_native.model import JointNativeCore

            core = JointNativeCore.from_config(core_config)
        else:
            core = core_type.from_config(core_config)
        adapter = cls(
            payload["normalization_stats"],
            core=core,
            nx=int(grid[0]),
            ny=int(grid[1]),
            environment_nx=int(environment[0]),
            environment_ny=int(environment[1]),
        )
        expected_config = adapter.get_config()
        if expected_config != {key: payload[key] for key in expected_config}:
            raise ValueError("Thermal v2 adapter payload differs from its canonical reconstructed config.")
        if adapter.normalizer_key != payload.get("normalizer_key"):
            raise ValueError("Thermal v2 adapter normalizer identity differs from its checkpoint payload.")
        if state_dict is not None:
            adapter.load_state_dict(state_dict, strict=True)
        return adapter.to(device)

    def scene_batch(self, structure: Mapping[str, Any], *, scene_key: Any = None) -> SceneBatch:
        """Bind explicit native frame, module radius, and E192 cell lengths."""
        scene = super().scene_batch(structure, scene_key=scene_key)
        # The native radius is source_lengths from the prescribed Thermal
        # inventory. Reuse the normalized radii and physical frame to recover
        # it without introducing another source of geometry or targets.
        radius = scene.module_features[..., 2:3] * scene.coordinate_scale[:, None, 0:1]
        module_lengths = radius.expand(-1, -1, 2)
        cell = scene.coordinate_scale / scene.coordinate_scale.new_tensor(
            [self.environment_nx, self.environment_ny]
        )
        environment_lengths = cell[:, None, :].expand(-1, scene.environment_features.shape[1], -1)
        return replace(
            scene,
            module_characteristic_lengths=module_lengths,
            environment_characteristic_lengths=environment_lengths,
            coordinate_frame_key=THERMAL_CONTENT_COORDINATE_FRAME_V2,
        )

    def apply_native_with_work_counts(
        self,
        prepared: PreparedThermalContentHypergraph,
        physical_heat: torch.Tensor,
        *,
        increment: bool = False,
        accumulation_dtype: torch.dtype | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        """Apply one native read and return counters from that same core call.

        The v1 public return shape stays unchanged. V2 training uses this
        additive method so the task ledger can include the core's real decode
        or affine-reader counts without replaying an expensive forward pass.
        """
        self.validate_prepared(prepared)
        controls = torch.as_tensor(physical_heat, device=prepared.scene_batch.module_features.device)
        if not controls.is_floating_point():
            controls = controls.to(self.field_std.dtype)
        if controls.shape != prepared.scene_batch.module_mask.shape:
            raise ValueError("Thermal physical heat controls must match the scene's padded module slots.")

        if prepared.affine_query is not None:
            response_options = ({"check_envelope": not increment}
                if getattr(self.core, "source_read_policy", "full") != "full" else {})
            output = self.core.apply_prepared_affine(
                prepared.affine_query, controls, accumulation_dtype=accumulation_dtype,
                **response_options,
            )
            temperature = output.thermal_temperature
            flow = None if increment else output.nonlinear_field
            core_work = output.work_counts
        elif increment:
            output = self.core.apply_increment(
                prepared.organization,
                prepared.query_batch,
                controls,
                accumulation_dtype=accumulation_dtype,
                return_work_counts=True,
            )
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("Thermal v2 core did not return affine-increment work counters.")
            temperature, core_work = output
            flow = None
        else:
            output = self.core.decode(
                prepared.organization,
                prepared.query_batch,
                controls=controls,
                accumulation_dtype=accumulation_dtype,
            )
            if output.thermal_temperature is None:
                raise RuntimeError("Thermal v2 core omitted its affine temperature output.")
            temperature = output.thermal_temperature
            flow = output.nonlinear_field
            core_work = output.work_counts

        if temperature is None or not isinstance(core_work, Mapping):
            raise RuntimeError("Thermal v2 core omitted native temperature or decode-work counters.")
        result = self._extract_native_roles(
            prepared, temperature, flow, accumulation_dtype=accumulation_dtype,
        )
        if increment:
            result["pred_field"] = torch.cat(
                (result["pred_field"][..., :4].zero_(), result["fluid_temperature"]), dim=-1,
            )
        counts: dict[str, int] = {}
        for name, value in core_work.items():
            if not isinstance(name, str) or type(value) is not int or value < 0:
                raise RuntimeError("Thermal v2 core returned an invalid decode-work counter.")
            counts[name] = value
        return result, counts

    def _query_union(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Keep native interpolation unchanged while versioning the query key."""
        result = super()._query_union(*args, **kwargs)
        query = result[0]
        union = result[1]
        ntheta = args[3] if len(args) > 3 else kwargs.get("ntheta")
        query = replace(
            query,
            coordinate_frame_key=THERMAL_CONTENT_COORDINATE_FRAME_V2,
            receiver_key=("thermal_native_union_v2", self.nx, self.ny, ntheta, tuple(union.grid_indices.shape)),
        )
        return (query, *result[1:])


class ThermalInheritedAttentionAdapterV1(ThermalContentHypergraphAdapterV2):
    """Round4 native Thermal transforms with an explicit IAG-C receiver cover.

    The established scene, native-grid union, curl, interpolation, and affine
    heat machinery remains the implementation source. This adapter owns only
    the IAG-C binding and the geometry-derived cover attached to each complete
    native query union.
    """

    SCHEMA = "thermal_inherited_attention_adapter_v1"
    CORE_SCHEMA = "inherited_attention_core_v1"
    COVER_SCHEMA = "smoothstep_grid_receiver_cover_from_native_environment_v1"

    def __init__(
        self,
        normalization_stats: Mapping[str, Any],
        *,
        core_config: Mapping[str, Any] | None = None,
        core: nn.Module | None = None,
        nx: int = 128,
        ny: int = 64,
        environment_nx: int = 24,
        environment_ny: int = 8,
    ) -> None:
        if (environment_nx, environment_ny) != (24, 8):
            raise ValueError("Thermal IAG-C retains the native 24x8 geometry-only environment catalogue.")
        if core is None:
            config = dict(core_config or {})
            if (config.get("schema") not in (self.CORE_SCHEMA, ROUND5_MOMENT_CORE_SCHEMA,
                                               ROUND6_READOUT_CORE_SCHEMA)
                    or config.get("architecture") != "IAG-C"):
                raise ValueError("Thermal IAG-C requires a registered IAG-C core schema.")
            core = _iag_core_from_config(config)
        config = core.get_config()
        required = {
            "architecture": "IAG-C",
            "module_input_dim": THERMAL_CONTENT_MODULE_FEATURES,
            "environment_input_dim": THERMAL_CONTENT_ENVIRONMENT_FEATURES,
            "coordinate_dim": 2,
            "query_input_dim": 0,
            "operating_input_dim": THERMAL_CONTENT_OPERATING_FEATURES,
            "output_dim": 3,
            "thermal_response": True,
            "signed_carrier_width": 8,
        }
        if (config.get("schema") not in (self.CORE_SCHEMA, ROUND5_MOMENT_CORE_SCHEMA,
                                           ROUND6_READOUT_CORE_SCHEMA)
                or any(config.get(name) != expected for name, expected in required.items())):
            raise ValueError("Thermal IAG-C core differs from the native dimensions or signed affine response law.")
        # Bypass the v2 architecture validator while retaining its unchanged
        # physical extraction and measured-work implementation.
        ThermalContentHypergraphAdapter.__init__(
            self,
            normalization_stats,
            core=core,
            nx=nx,
            ny=ny,
            environment_nx=environment_nx,
            environment_ny=environment_ny,
        )
        self.normalizer_key = _canonical_hash({
            "schema": self.SCHEMA,
            "coordinate_frame": THERMAL_CONTENT_COORDINATE_FRAME_V2,
            "geometry_contract": THERMAL_CONTENT_GEOMETRY_CONTRACT_V2,
            "normalization_stats": self.normalization_stats,
            "native_grid_nx_ny": [self.nx, self.ny],
            "environment_grid_nx_ny": [self.environment_nx, self.environment_ny],
        })

    def get_config(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "coordinate_frame": THERMAL_CONTENT_COORDINATE_FRAME_V2,
            "feature_schema": THERMAL_CONTENT_FEATURE_SCHEMA_V2,
            "geometry_contract": copy.deepcopy(THERMAL_CONTENT_GEOMETRY_CONTRACT_V2),
            "native_law_contract": copy.deepcopy(THERMAL_CONTENT_NATIVE_LAW_CONTRACT_V2),
            "receiver_cover": self.COVER_SCHEMA,
            "native_grid_nx_ny": [self.nx, self.ny],
            "environment_grid_nx_ny": [self.environment_nx, self.environment_ny],
            "core": self.core.get_config(),
        }

    def state_payload(self) -> dict[str, Any]:
        return {
            **self.get_config(),
            "normalization_stats": _stats_json(self.normalization_stats),
            "normalizer_key": self.normalizer_key,
        }

    def scene_batch(self, structure: Mapping[str, Any], *, scene_key: Any = None) -> SceneBatch:
        """Attach one live, scene-owned receiver topology to the prepared scene."""
        from honf_forward_core.inherited_attention import InheritedSceneBatch, prepare_grid_topology

        scene = super().scene_batch(structure, scene_key=scene_key)
        if scene.coordinate_origin is None or scene.coordinate_scale is None:
            raise ValueError("Thermal IAG-C topology requires explicit native origin and scale.")
        legal_bounds = torch.stack(
            (scene.coordinate_origin, scene.coordinate_origin + scene.coordinate_scale), dim=1,
        )
        values = {item.name: getattr(scene, item.name) for item in fields(SceneBatch)}
        inherited = InheritedSceneBatch(**values, receiver_legal_bounds=legal_bounds)
        topology = prepare_grid_topology(
            inherited.environment_coordinates,
            inherited.environment_ids,
            inherited.environment_mask,
            frame_key=inherited.coordinate_frame_key,
            scene_version=inherited.tensor_version_stamp(),
            owner_token=object(),
            normalizer_key=inherited.normalizer_key,
            scene_key=inherited.scene_key,
            legal_bounds=inherited.receiver_legal_bounds,
        )
        return replace(inherited, receiver_grid_topology=topology)

    @classmethod
    def from_state_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
        state_dict: Mapping[str, Any] | None = None,
    ) -> ThermalInheritedAttentionAdapterV1:
        if payload.get("schema") != cls.SCHEMA or payload.get("receiver_cover") != cls.COVER_SCHEMA:
            raise ValueError("Unsupported Thermal IAG-C adapter or receiver-cover schema.")
        adapter = cls(
            payload["normalization_stats"],
            core=_iag_core_from_config(dict(payload["core"])),
            nx=int(payload["native_grid_nx_ny"][0]),
            ny=int(payload["native_grid_nx_ny"][1]),
            environment_nx=int(payload["environment_grid_nx_ny"][0]),
            environment_ny=int(payload["environment_grid_nx_ny"][1]),
        )
        config = adapter.get_config()
        if config != {key: payload[key] for key in config} or adapter.normalizer_key != payload.get("normalizer_key"):
            raise ValueError("Thermal IAG-C adapter payload differs from its canonical reconstructed transforms.")
        if state_dict is not None:
            adapter.load_state_dict(state_dict, strict=True)
        return adapter.to(device)

    def _query_union(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        from honf_forward_core.inherited_attention import (
            smoothstep_grid_receiver_cover,
            with_receiver_cover,
        )

        result = super()._query_union(*args, **kwargs)
        query = result[0]
        scene = args[0] if args else kwargs["scene"]
        topology = getattr(scene, "receiver_grid_topology", None)
        if topology is None:
            raise ValueError("Thermal IAG-C receiver cover requires the scene's prepared grid topology.")
        cover = smoothstep_grid_receiver_cover(
            query.coordinates,
            query_ids=query.query_ids,
            frame_key=scene.coordinate_frame_key,
            query_mask=query.query_mask,
            topology=topology,
            scene_version=scene.tensor_version_stamp(),
            owner_token=topology.owner_token,
            normalizer_key=scene.normalizer_key,
            scene_key=scene.scene_key,
        )
        return (with_receiver_cover(query, cover), *result[1:])


__all__ = [
    "THERMAL_CONTENT_ADAPTER_SCHEMA",
    "THERMAL_CONTENT_ADAPTER_V2_SCHEMA",
    "THERMAL_CONTENT_COORDINATE_FRAME",
    "THERMAL_CONTENT_COORDINATE_FRAME_V2",
    "THERMAL_CONTENT_ENVIRONMENT_SHAPE",
    "PreparedThermalContentHypergraph",
    "ThermalContentHypergraphAdapter",
    "ThermalContentHypergraphAdapterV2",
    "ThermalInheritedAttentionAdapterV1",
]
