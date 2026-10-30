"""Fresh 3-D Wind adapter for the public relation-anchored joint field model.

Support bounds are prescribed and fixed when a layout changes.  The input
centres and receiver coordinates remain live torch tensors through decoding.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product
from collections.abc import Mapping

import torch
from torch import nn
from honf_forward_core.content_hypergraph import QueryBatch, SceneBatch
from honf_forward_core.models import build_model, relation_anchored_config

from .content_hypergraph import WIND_CONTENT_COORDINATE_FRAME, WindContentHypergraphAdapter
from .geometry import POSITIONAL_SCALE_D


def _features(coordinates, lower, upper):
    scale = coordinates.new_tensor(POSITIONAL_SCALE_D)
    return torch.cat(((coordinates - lower) / scale, (upper - coordinates) / scale,
                      coordinates[..., 2:3] / scale[2]), dim=-1)


def attach_wind_e_cover(query: QueryBatch, scene: SceneBatch, shape: tuple[int, int, int]) -> QueryBatch:
    """Trilinear centre cover, clamped in outer half-cells, with actual slot lookup.

    Queries outside the declared centre-box support are rejected. Masked queries
    carry zero weights. Singleton axes are supported and have only one active
    choice; consequently each positive slot occurs once.
    """
    shape = tuple(int(n) for n in shape)
    if len(shape) != 3 or any(n < 1 for n in shape) or scene.num_environment_tokens != shape[0]*shape[1]*shape[2]:
        raise ValueError("Wind E shape must match the complete positive lattice.")
    if scene.coordinate_origin is None or scene.coordinate_scale is None:
        raise ValueError("Wind coverage requires fixed support bounds.")
    masks = query.query_mask if query.query_mask is not None else torch.ones(query.coordinates.shape[:2], dtype=torch.bool, device=query.coordinates.device)
    all_indices, all_weights = [], []
    for b in range(scene.batch_size):
        origin, extent = scene.coordinate_origin[b], scene.coordinate_scale[b]
        coords = scene.environment_coordinates[b]
        axes = [origin[d] + (torch.arange(n, device=coords.device, dtype=coords.dtype) + .5) * extent[d] / n for d, n in enumerate(shape)]
        grid_indices = [torch.argmin((coords[:, d, None] - axes[d]).abs(), dim=-1) for d in range(3)]
        linear = (grid_indices[0]*shape[1] + grid_indices[1])*shape[2] + grid_indices[2]
        expected = torch.stack([axes[d][grid_indices[d]] for d in range(3)], dim=-1)
        tol = extent.abs().amax() * (16 * torch.finfo(coords.dtype).eps)
        counts = torch.bincount(linear, minlength=scene.num_environment_tokens)
        if not bool(scene.environment_mask[b].all()) or not bool((counts == 1).all()) or not bool(((coords - expected).abs() <= tol).all()):
            raise ValueError("Wind E coordinates must map one-to-one to the declared lattice.")
        slots = torch.empty_like(linear).scatter(0, linear, torch.arange(linear.numel(), device=linear.device))
        q = query.coordinates[b]
        valid = masks[b]
        if not bool((~valid | ((q >= origin - tol) & (q <= origin + extent + tol)).all(-1)).all()):
            raise ValueError("A valid Wind query lies outside the fixed support bounds.")
        choices = []
        for d, n in enumerate(shape):
            if n == 1:
                zero = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
                choices.append(((zero, torch.ones_like(q[:, d])), (zero, torch.zeros_like(q[:, d]))))
            else:
                right = torch.searchsorted(axes[d].contiguous(), q[:, d].detach().contiguous(), right=True).clamp(1, n-1)
                left = right - 1
                t = ((q[:, d] - axes[d][left]) / (axes[d][right] - axes[d][left])).clamp(0, 1)
                choices.append(((left, 1-t), (right, t)))
        indices, weights = [], []
        for corner in product((0, 1), repeat=3):
            (ix, wx), (iy, wy), (iz, wz) = [choices[d][corner[d]] for d in range(3)]
            w = wx*wy*wz
            slot = slots[(ix*shape[1]+iy)*shape[2]+iz]
            active = valid & (w > 0)
            indices.append(torch.where(active, slot, -torch.ones_like(slot)))
            weights.append(torch.where(active, w, torch.zeros_like(w)))
        all_indices.append(torch.stack(indices, -1))
        all_weights.append(torch.stack(weights, -1))
    return replace(query, anchor_indices=torch.stack(all_indices), anchor_weights=torch.stack(all_weights))


class WindRelationAnchoredAdapter(nn.Module):
    """Case transforms, configurable E lattice and a joint three-velocity core."""
    SCHEMA = "wind_relation_anchored_adapter_v1"

    def __init__(self, normalizer, background_profile, *, base_model_config=None, core_config=None, environment_shape=(4, 4, 4)):
        super().__init__()
        self.environment_shape = tuple(int(n) for n in environment_shape)
        if len(self.environment_shape) != 3 or any(n < 1 for n in self.environment_shape):
            raise ValueError("environment_shape must have three positive entries.")
        if core_config is None:
            core_config = relation_anchored_config("relation_anchored", base_model_config)
        self.core = build_model("relation_anchored", core_config)
        base = self.core.get_config()["base_model_config"]
        if (base["coordinate_dim"], base["module_input_dim"], base["environment_input_dim"], base["query_input_dim"], base["operating_input_dim"], base["output_dim"]) != (3, 2, 7, 7, 11, 3):
            raise ValueError("Wind core requires 3-D coordinates, native feature widths and three joint velocity outputs.")
        self.transforms = WindContentHypergraphAdapter(normalizer, background_profile)

    def state_payload(self):
        return {"schema": self.SCHEMA, "environment_shape": list(self.environment_shape),
                "core": self.core.get_config(), "transforms": self.transforms.state_payload()}

    @classmethod
    def from_state_payload(cls, payload, *, state_dict=None, device="cpu"):
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA:
            raise ValueError("Unsupported Wind relation-anchored adapter state.")
        transforms = WindContentHypergraphAdapter.from_state_payload(payload["transforms"])
        model = cls(transforms.normalizer, transforms.background_profile, core_config=payload["core"], environment_shape=payload["environment_shape"])
        if model.state_payload() != dict(payload):
            raise ValueError("Wind adapter payload does not match its reconstructed configuration.")
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)
        return model.to(device)

    def make_scene(self, case, module_centers=None, *, module_ids=None):
        """Build one native target-free scene; module_centers has shape [M,3]."""
        parameter = next(self.core.parameters())
        def tensor(value, dtype=None):
            return torch.as_tensor(value, dtype=dtype or parameter.dtype, device=parameter.device)
        centers = tensor(case.module_centers) if module_centers is None else module_centers
        if centers.device != parameter.device or centers.dtype != parameter.dtype or centers.ndim != 2 or centers.shape[-1] != 3:
            raise ValueError("Layout tensor must be [M,3] on the core's device and dtype.")
        if not torch.equal(centers[:, 2].detach(), tensor(case.module_centers)[:, 2]):
            raise ValueError("Wind design changes are horizontal; turbine hub heights are prescribed.")
        m = centers.shape[0]
        features, mask = tensor(case.module_features), tensor(case.module_present, torch.bool)
        if features.shape != (m, 2) or mask.shape != (m,) or m < 1 or m > 30:
            raise ValueError("Wind module features and mask must match 1 to 30 original turbines.")
        origin, extent = tensor(case.support.lower_D), tensor(case.support.extent_D)
        if not bool((extent > 0).all()):
            raise ValueError("Wind support extent must be positive.")
        axes = [(torch.arange(n, device=parameter.device, dtype=parameter.dtype)+.5)/n for n in self.environment_shape]
        fraction = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        ecoords = origin + fraction * extent
        count = ecoords.shape[0]
        eweight = extent.prod() / count
        ids = (torch.arange(m, device=parameter.device, dtype=torch.long) if module_ids is None
               else torch.as_tensor(module_ids, device=parameter.device, dtype=torch.long))
        if ids.shape != (m,) or ids.unique().numel() != m:
            raise ValueError("Wind original module IDs must be distinct and aligned with the layout.")
        return SceneBatch(module_features=features[None], environment_features=_features(ecoords, origin, origin+extent)[None],
            module_coordinates=centers[None], environment_coordinates=ecoords[None], module_mask=mask[None],
            environment_mask=torch.ones((1,count), dtype=torch.bool, device=parameter.device),
            module_measure=mask.to(parameter.dtype)[None], environment_measure=eweight.expand(1,count),
            module_ids=ids[None], environment_ids=torch.arange(count, device=parameter.device)[None],
            local_radius=centers.new_full((1,m), 1.5), operating_features=tensor(case.global_context)[None],
            coordinate_origin=origin[None], coordinate_scale=extent[None], module_measure_reference=centers.new_ones(1),
            environment_measure_reference=eweight.reshape(1), scene_key=(str(case.case), int(case.index), int(case.layout_index), float(case.wind_direction_deg)),
            normalizer_key=self.transforms.normalizer_key, coordinate_frame_key=WIND_CONTENT_COORDINATE_FRAME,
            module_characteristic_lengths=(2*features[:, :1]).expand(m,3)[None],
            environment_characteristic_lengths=(extent/tensor(self.environment_shape)).expand(1,count,3))

    def make_query(self, scene, coordinates, *, query_ids=None, query_mask=None):
        if coordinates.ndim == 2:
            coordinates = coordinates[None]
        if coordinates.ndim != 3 or coordinates.shape[0] != scene.batch_size or coordinates.shape[-1] != 3:
            raise ValueError("Wind query coordinates must have shape [B,Q,3].")
        if query_mask is None:
            query_mask = torch.ones(coordinates.shape[:2], dtype=torch.bool, device=coordinates.device)
        if query_ids is None:
            query_ids = torch.arange(coordinates.shape[1], device=coordinates.device).expand(coordinates.shape[0], -1)
        origin, extent = scene.coordinate_origin[:,None,:], scene.coordinate_scale[:,None,:]
        query = QueryBatch(coordinates=coordinates, features=_features(coordinates, origin, origin+extent),
            query_mask=query_mask, query_ids=query_ids, scene_key=scene.scene_key,
            coordinate_frame_key=scene.coordinate_frame_key, receiver_key=(scene.scene_key, coordinates.shape[1]))
        return attach_wind_e_cover(query, scene, self.environment_shape)

    def forward(self, case, coordinates, module_centers=None, *, module_ids=None, query_ids=None, query_mask=None, return_trace=False):
        scene = self.make_scene(case, module_centers, module_ids=module_ids)
        query = self.make_query(scene, coordinates, query_ids=query_ids, query_mask=query_mask)
        organization = self.core.prepare(scene)
        output = self.core.decode(organization, query, return_trace=return_trace)
        velocity, _, _ = self.transforms.physical_velocity(output.joint_fields, query.coordinates)
        return velocity, output


__all__ = ["WindRelationAnchoredAdapter", "attach_wind_e_cover"]
