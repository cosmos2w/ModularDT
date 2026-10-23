"""Level-R compact retrieval groups for the unified continuous encoder.

The contextual encoder remains native and applied.  This module creates the
final K-by-N retrieval directory directly from live group controls and the
original prepared record states; physical record/value axes are never pooled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..content_hypergraph.spatial_features_v2 import (
    pair_score_features,
    query_source_score_features,
)
from ..joint_native.routing import JointDispatchParameters, ScoreCalibration
from ..content_hypergraph.staged_attention_v2 import bounded_temperature
from .primitives import entmax15, positive_quadratic_gate


@dataclass(frozen=True)
class SceneGroupTopology:
    """Compact scene-local lineage; indices always refer to original records."""

    group_members: tuple[tuple[int, ...], ...]
    group_types: tuple[int, ...]
    group_keys: tuple[tuple[int, tuple[int, ...]], ...]
    candidate_pairs: tuple[tuple[int, int, int], ...]
    proposed_pairs: tuple[tuple[int, int, int], ...]
    accepted_pairs: tuple[tuple[int, int, int], ...]
    archive_to_active: tuple[tuple[tuple[int, tuple[int, ...]], tuple[int, tuple[int, ...]]], ...]
    retained_key_by_record: tuple[tuple[int, tuple[int, ...]], ...]

    @staticmethod
    def _key_payload(key):
        return [int(key[0]), [int(value) for value in key[1]]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_members": [list(map(int, members)) for members in self.group_members],
            "group_types": list(map(int, self.group_types)),
            "active_group_keys": [self._key_payload(key) for key in self.group_keys],
            "candidate_pairs": [[int(u), int(v), int(kind)] for u, v, kind in self.candidate_pairs],
            "proposed_merge_pairs": [[int(u), int(v), int(kind)] for u, v, kind in self.proposed_pairs],
            "accepted_pairs": [[int(u), int(v), int(kind)] for u, v, kind in self.accepted_pairs],
            "archive_to_active": [
                [self._key_payload(old), self._key_payload(new)]
                for old, new in self.archive_to_active
            ],
            "retained_key_by_record": [
                [int(record), self._key_payload(key)] for record, key in self.retained_key_by_record
            ],
        }


@dataclass(frozen=True)
class CompactOrganization:
    """Prepared live groups and direct original-record membership."""

    membership: Tensor  # [B,K,N], not an N-by-N native directory
    group_mask: Tensor  # [B,K]
    group_types: Tensor  # [B,K], 0=M and 1=E
    controls: Tensor  # [B,K,width], routing only
    centroids: Tensor  # [B,K,D], routing only
    lengths: Tensor  # [B,K,D], positive route scales
    extents: Tensor  # [B,K,D], normalized per-axis extent
    pair_logits: Tensor  # [B,P], shared policy outputs per scene
    pair_mask: Tensor  # [B,P]
    pair_slots: Tensor  # [B,P,2], active sibling slots during soft tying
    topology: tuple[SceneGroupTopology, ...]
    bank_group_indices: tuple[Tensor, Tensor] | None = None
    bank_group_masks: tuple[Tensor, Tensor] | None = None

    @property
    def live_counts(self) -> Tensor:
        return self.group_mask.sum(-1)

    @property
    def allocated_count(self) -> int:
        return int(self.group_mask.shape[1])

    @property
    def score_slot_count(self) -> int:
        if self.bank_group_indices is None:
            raise ValueError("Prepared compact groups lack their typed score indices.")
        return sum(int(indices.shape[1]) for indices in self.bank_group_indices)


class SiblingCompatibility(nn.Module):
    """One shared symmetric policy per destination bank."""

    def __init__(self, input_width: int):
        super().__init__()
        self.hidden = nn.Linear(input_width, 32)
        self.output = nn.Linear(32, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, -2.0)

    def forward(self, pair_features: Tensor) -> Tensor:
        return self.output(F.silu(self.hidden(pair_features))).squeeze(-1)


class GroupControlProjector(nn.Module):
    """Identity-initialized latent map plus a zero geometry/type correction."""

    def __init__(self, width: int, coordinate_dim: int):
        super().__init__()
        self.latent = nn.Linear(width, width, bias=False)
        self.geometry_type = nn.Linear(3 * coordinate_dim + 2, width, bias=False)
        nn.init.eye_(self.latent.weight)
        nn.init.zeros_(self.geometry_type.weight)

    def forward(self, state_mean: Tensor, centroid: Tensor, lengths: Tensor, extent: Tensor,
                origin: Tensor, frame_scale: Tensor, bank_type: int) -> Tensor:
        normalized_centroid = (centroid - origin) / frame_scale
        log_lengths = torch.log((lengths / frame_scale).clamp_min(torch.finfo(lengths.dtype).tiny))
        type_code = F.one_hot(torch.full((), bank_type, device=state_mean.device, dtype=torch.long), 2)
        type_code = type_code.to(state_mean).expand(*state_mean.shape[:-1], 2)
        geometry = torch.cat((normalized_centroid, extent, log_lengths, type_code), -1)
        return self.latent(state_mean) + self.geometry_type(geometry)


class CompactRouteScorer(nn.Module):
    """Copied score-only part of one final applied contextual route."""

    def __init__(self, width: int, heads: int, coordinate_dim: int):
        super().__init__()
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = width // heads
        self.coordinate_dim = int(coordinate_dim)
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.geometry_score = nn.Sequential(
            nn.Linear(2 * coordinate_dim + 1, 32), nn.SiLU(), nn.Linear(32, heads, bias=False)
        )
        self.raw_temperature = nn.Parameter(torch.zeros(heads))

    @torch.no_grad()
    def copy_from_attention(self, attention) -> None:
        for name in ("query_norm", "key_norm", "query", "key", "geometry_score"):
            getattr(self, name).load_state_dict(getattr(attention, name).state_dict())
        self.raw_temperature.copy_(attention.raw_temperature)

    def forward(self, destination_states: Tensor, destination_coordinates: Tensor,
                destination_lengths: Tensor, donor_states: Tensor, donor_coordinates: Tensor,
                donor_lengths: Tensor, frame_scale: Tensor) -> Tensor:
        batch, groups, _ = destination_states.shape
        donors = donor_states.shape[1]
        if groups == 0 or donors == 0:
            return destination_states.new_zeros(batch, self.heads, groups, donors)
        q = self.query(self.query_norm(destination_states)).reshape(
            batch, groups, self.heads, self.head_width
        ).transpose(1, 2)
        k = self.key(self.key_norm(donor_states)).reshape(
            batch, donors, self.heads, self.head_width
        ).transpose(1, 2)
        q = F.normalize(q, dim=-1, eps=1.0e-6) * bounded_temperature(self.raw_temperature)[None, :, None, None]
        k = F.normalize(k, dim=-1, eps=1.0e-6)
        raw = torch.matmul(q, k.transpose(-2, -1))
        geometry, _ = pair_score_features(
            destination_coordinates, donor_coordinates, frame_scale,
            destination_lengths, donor_lengths,
        )
        return raw + self.geometry_score(geometry).permute(0, 3, 1, 2)


class CompactQueryScorer(nn.Module):
    """Score Q against only live group controls, with an independent copy."""

    def __init__(self, width: int, heads: int, coordinate_dim: int):
        super().__init__()
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = width // heads
        self.coordinate_dim = int(coordinate_dim)
        self.query_norm = nn.LayerNorm(width)
        self.source_norm = nn.LayerNorm(width)
        self.query_key = nn.Linear(width, width, bias=False)
        self.source_key = nn.Linear(width, width, bias=False)
        self.geometry_score = nn.Sequential(
            nn.Linear(2 * coordinate_dim + 1, 32), nn.SiLU(), nn.Linear(32, heads, bias=False)
        )
        self.raw_temperature = nn.Parameter(torch.zeros(heads))

    @torch.no_grad()
    def copy_from_reader(self, reader) -> None:
        for name in ("query_norm", "source_norm", "query_key", "source_key", "geometry_score"):
            getattr(self, name).load_state_dict(getattr(reader, name).state_dict())
        self.raw_temperature.copy_(reader.raw_temperature)

    def prepare_keys(self, group_controls: Tensor) -> Tensor:
        batch, count, _ = group_controls.shape
        if not count:
            return group_controls.new_zeros(batch, 0, self.width)
        return self.source_key(self.source_norm(group_controls))

    def forward(self, query, query_states: Tensor, group_controls: Tensor, group_centroids: Tensor,
                group_lengths: Tensor, frame_scale: Tensor) -> Tensor:
        batch, queries, _ = query_states.shape
        groups = group_controls.shape[1]
        if groups == 0 or queries == 0:
            return query_states.new_zeros(batch, self.heads, queries, groups)
        q = self.query_key(self.query_norm(query_states)).reshape(
            batch, queries, self.heads, self.head_width
        ).transpose(1, 2)
        keys = self.prepare_keys(group_controls).reshape(
            batch, groups, self.heads, self.head_width
        ).transpose(1, 2)
        q = F.normalize(q, dim=-1, eps=1.0e-6) * bounded_temperature(self.raw_temperature)[None, :, None, None]
        keys = F.normalize(keys, dim=-1, eps=1.0e-6)
        raw = torch.matmul(q, keys.transpose(-2, -1))
        geometry = query_source_score_features(
            query.coordinates.to(query_states), group_centroids, frame_scale, group_lengths,
        )
        return raw + self.geometry_score(geometry).permute(0, 3, 1, 2)


@dataclass
class _Tree:
    groups: list[tuple[int, ...]]
    kinds: list[int]
    keys: list[tuple[int, tuple[int, ...]]]
    pairs: list[tuple[int, int, int]]


class CompactGroupRetrieval(nn.Module):
    """Build and score typed live groups without an N-by-N retrieval bank."""

    TOPOLOGY_RULE = "same_type_adjacent_unique_coordinate_siblings_v1"
    CUT_RULE = "shared_compatibility_logit_strictly_positive_v1"

    def __init__(self, *, width: int, heads: int, coordinate_dim: int,
                 module_feature_dim: int, environment_feature_dim: int, operating_feature_dim: int):
        super().__init__()
        pair_geometry_width = 4 * coordinate_dim + operating_feature_dim
        self.compatibility = nn.ModuleDict({
            "M": SiblingCompatibility(2 * module_feature_dim + pair_geometry_width),
            "E": SiblingCompatibility(2 * environment_feature_dim + pair_geometry_width),
        })
        self.controls = nn.ModuleDict({
            "M": GroupControlProjector(width, coordinate_dim),
            "E": GroupControlProjector(width, coordinate_dim),
        })
        self.route_scorers = nn.ModuleDict({
            name: CompactRouteScorer(width, heads, coordinate_dim)
            for name in ("mm", "me", "em", "ee")
        })
        self.membership_calibration = nn.ModuleDict({
            name: ScoreCalibration(heads) for name in ("mm", "me", "em", "ee")
        })
        self.query_scorers = nn.ModuleDict({
            "M": CompactQueryScorer(width, heads, coordinate_dim),
            "E": CompactQueryScorer(width, heads, coordinate_dim),
        })
        self.query_calibration = JointDispatchParameters(heads)
        self.register_buffer("entmax_temperature", torch.ones(()))

    @torch.no_grad()
    def initialize_from_parent(self, parent) -> dict[str, list[str]]:
        """Copy compatible final-cycle route/read controls without aliasing them."""
        if not getattr(parent, "context_blocks", None):
            raise ValueError("Compact retrieval requires the parent's applied contextual encoder.")
        final = parent.context_blocks[-1]
        for route in ("mm", "me", "em", "ee"):
            self.route_scorers[route].copy_from_attention(getattr(final, route))
            self.membership_calibration[route].load_state_dict(
                parent.router.incidence.routes[route].state_dict()
            )
        self.query_scorers["M"].copy_from_reader(parent.direct_reader)
        self.query_scorers["E"].copy_from_reader(parent.environment_reader)
        self.query_calibration.load_state_dict(parent.router.dispatch_parameters.state_dict())
        self.entmax_temperature.copy_(parent.router.entmax_temperature)
        copied = [f"route_scorers.{name}" for name in ("mm", "me", "em", "ee")]
        copied += [f"membership_calibration.{name}" for name in ("mm", "me", "em", "ee")]
        copied += ["query_scorers.M", "query_scorers.E", "query_calibration", "entmax_temperature"]
        fresh = [name for name, _ in self.named_parameters() if name.startswith(("compatibility.", "controls."))]
        return {"copied_from_parent": copied, "new_parameters": fresh}

    @staticmethod
    def _record_key(bank: int, record_ids: list[int]) -> tuple[int, tuple[int, ...]]:
        return int(bank), tuple(sorted(int(value) for value in record_ids))

    @staticmethod
    def _coordinate_bundles(coordinates: Tensor, indices: list[int], origin: Tensor,
                            scale: Tensor) -> list[tuple[tuple[float, ...], list[int]]]:
        if not indices:
            return []
        normalized = ((coordinates.detach().to(torch.float64) - origin.detach().to(torch.float64))
                      / scale.detach().to(torch.float64)).cpu().tolist()
        buckets: dict[tuple[float, ...], list[int]] = {}
        for index in indices:
            key = tuple(float(component) for component in normalized[index])
            buckets.setdefault(key, []).append(index)
        return [(key, buckets[key]) for key in sorted(buckets)]

    def _tree(self, organization, batch_index: int) -> _Tree:
        """Pair adjacent unique leaves; coordinate-tie bundles separate runs."""
        scene = organization.scene
        records = organization.joint_records
        origin = organization.frame_origin[batch_index]
        scale = organization.frame_scale[batch_index]
        groups: list[tuple[int, ...]] = []
        kinds: list[int] = []
        keys: list[tuple[int, tuple[int, ...]]] = []
        pairs: list[tuple[int, int, int]] = []
        for kind, bank, offset, present, coordinates, ids in (
            ("M", 0, 0, scene.module_mask[batch_index], records.module_coordinates[batch_index], records.module_ids[batch_index]),
            ("E", 1, records.module_count, scene.environment_mask[batch_index], records.environment_coordinates[batch_index], records.environment_ids[batch_index]),
        ):
            ids_cpu = ids.detach().cpu().tolist()
            indices = [int(index) + offset for index in present.nonzero(as_tuple=False).flatten().detach().cpu().tolist()]
            local_indices = [index - offset for index in indices]
            bundles = self._coordinate_bundles(coordinates, local_indices, origin, scale)
            bucket_slots: list[list[int]] = []
            for _coordinate, bundle in bundles:
                slots: list[int] = []
                for local_index in bundle:
                    record_index = local_index + offset
                    slot = len(groups)
                    groups.append((record_index,))
                    kinds.append(bank)
                    record_id = int(ids_cpu[local_index])
                    keys.append(self._record_key(bank, [record_id]))
                    slots.append(slot)
                bucket_slots.append(slots)
            cursor = 0
            while cursor < len(bucket_slots):
                if (len(bucket_slots[cursor]) == 1 and cursor + 1 < len(bucket_slots)
                        and len(bucket_slots[cursor + 1]) == 1):
                    pairs.append((bucket_slots[cursor][0], bucket_slots[cursor + 1][0], bank))
                    cursor += 2
                else:
                    cursor += 1
        return _Tree(groups, kinds, keys, pairs)

    @staticmethod
    def _gather_children(values: Tensor, indices: Tensor) -> Tensor:
        """Gather at most two ancestors per group without a dense assignment."""
        batch, groups, children = indices.shape
        width = values.shape[-1]
        if values.shape[1] == 0:
            return values.new_zeros(batch, groups, children, width)
        joined = indices.reshape(batch, groups * children, 1).expand(-1, -1, width)
        return values.gather(1, joined).reshape(batch, groups, children, width)

    def _pair_logits(self, organization, pair_indices: Tensor, pair_types: Tensor,
                     pair_mask: Tensor) -> Tensor:
        records, scene = organization.joint_records, organization.scene
        coords = self._gather_children(records.coordinates, pair_indices)
        lengths = self._gather_children(records.lengths, pair_indices)
        scale = organization.frame_scale[:, None]
        separation = (coords[:, :, 0] - coords[:, :, 1]).abs() / scale
        # Sums and absolute differences make every child feature symmetric.
        length_sum = (lengths[:, :, 0] + lengths[:, :, 1]) / scale
        length_difference = (lengths[:, :, 0] - lengths[:, :, 1]).abs() / scale
        geometry = torch.cat((separation, separation, length_sum, length_difference), -1)
        operating = (scene.operating_features if scene.operating_features is not None
                     else coords.new_zeros(scene.batch_size, 0))
        operating = operating[:, None].expand(-1, pair_indices.shape[1], -1)
        result = coords.new_zeros(pair_mask.shape)
        for bank, name, features, offset in (
            (0, "M", scene.module_features, 0),
            (1, "E", scene.environment_features, records.module_count),
        ):
            local_indices = (pair_indices - offset).clamp(0, max(features.shape[1] - 1, 0))
            children = self._gather_children(features, local_indices)
            inputs = torch.cat((children[:, :, 0] + children[:, :, 1],
                                (children[:, :, 0] - children[:, :, 1]).abs(),
                                geometry, operating.to(features)), -1)
            values = self.compatibility[name](inputs)
            result = result + torch.where(pair_mask & (pair_types == bank), values, 0)
        return result

    def _descriptors(self, organization, native_states: Tensor, ancestry: Tensor,
                     ancestry_mask: Tensor, group_types: Tensor, group_mask: Tensor):
        records = organization.joint_records
        mask = ancestry_mask[..., None].to(native_states)
        count = mask.sum(2).clamp_min(1)
        states = self._gather_children(native_states, ancestry)
        coords = self._gather_children(records.coordinates, ancestry)
        lengths = self._gather_children(records.lengths, ancestry)
        mean_state = (states * mask).sum(2) / count
        centroid = (coords * mask).sum(2) / count
        variance = ((coords - centroid[:, :, None]).square() * mask).sum(2) / count
        squared_length = (lengths.square() * mask).sum(2) / count + variance
        # Padded slots use a positive scale to keep sqrt/log derivatives finite.
        group_lengths = torch.sqrt(squared_length + (~group_mask)[..., None].to(squared_length))
        high = coords.masked_fill(~ancestry_mask[..., None], -torch.inf).amax(2)
        low = coords.masked_fill(~ancestry_mask[..., None], torch.inf).amin(2)
        extent = torch.where(group_mask[..., None], high - low, 0) / organization.frame_scale[:, None]
        controls = []
        for bank, name in ((0, "M"), (1, "E")):
            controls.append(self.controls[name](mean_state, centroid, group_lengths, extent,
                organization.frame_origin[:, None], organization.frame_scale[:, None], bank))
        control = torch.where((group_types == 0)[..., None], controls[0], controls[1])
        return control * group_mask[..., None], centroid, group_lengths, extent

    @staticmethod
    def _gather_slots(values: Tensor, indices: Tensor) -> Tensor:
        return values.gather(1, indices[..., None].expand(-1, -1, values.shape[-1]))

    def _build_direct_membership(self, organization, controls, centroids, lengths,
                                 bank_group_indices, bank_group_masks) -> Tensor:
        records = organization.joint_records
        donor_states = (organization.module_states, organization.environment_states)
        donor_coords = (records.module_coordinates, records.environment_coordinates)
        donor_lengths = (records.module_lengths, records.environment_lengths)
        donor_valid = (records.module_present,
                       records.environment_present & (records.environment_measure > 0))
        membership = controls.new_zeros(controls.shape[0], controls.shape[1], records.ids.shape[1])
        for bank in (0, 1):
            indices, mask = bank_group_indices[bank], bank_group_masks[bank]
            group_controls = self._gather_slots(controls, indices)
            group_centroids = self._gather_slots(centroids, indices)
            group_lengths = self._gather_slots(lengths, indices)
            blocks = []
            for donor_type, route in enumerate(("mm", "me") if bank == 0 else ("em", "ee")):
                allowed = mask[:, :, None] & donor_valid[donor_type][:, None]
                score = self.route_scorers[route](group_controls, group_centroids, group_lengths,
                    donor_states[donor_type], donor_coords[donor_type], donor_lengths[donor_type],
                    organization.frame_scale)
                calibrated = self.membership_calibration[route](score, allowed)
                blocks.append(positive_quadratic_gate(calibrated) * allowed)
            values = torch.cat(blocks, -1)
            membership = membership.scatter_add(1, indices[..., None].expand_as(values), values)
        return membership

    def build(self, organization, *, alpha: float, cut_active: bool) -> CompactOrganization:
        """Build metadata once, then batch all live control and membership math."""
        scene, records = organization.scene, organization.joint_records
        batch = scene.batch_size
        reference = organization.module_states
        device = reference.device
        trees = [self._tree(organization, index) for index in range(batch)]
        max_pairs = max((len(tree.pairs) for tree in trees), default=0)
        pair_indices_rows, pair_type_rows, pair_slot_rows, pair_mask_rows = [], [], [], []
        for tree in trees:
            count = len(tree.pairs)
            pair_indices_rows.append([[tree.groups[u][0], tree.groups[v][0]] for u, v, _ in tree.pairs]
                                     + [[0, 0]] * (max_pairs - count))
            pair_type_rows.append([kind for _, _, kind in tree.pairs] + [-1] * (max_pairs - count))
            pair_slot_rows.append([[u, v] for u, v, _ in tree.pairs] + [[0, 0]] * (max_pairs - count))
            pair_mask_rows.append([True] * count + [False] * (max_pairs - count))
        pair_indices = torch.tensor(pair_indices_rows, device=device, dtype=torch.long).reshape(batch, max_pairs, 2)
        pair_types = torch.tensor(pair_type_rows, device=device, dtype=torch.long).reshape(batch, max_pairs)
        pair_slots = torch.tensor(pair_slot_rows, device=device, dtype=torch.long).reshape(batch, max_pairs, 2)
        eligible_pairs = torch.tensor(pair_mask_rows, device=device, dtype=torch.bool).reshape(batch, max_pairs)
        pair_logits = self._pair_logits(organization, pair_indices, pair_types, eligible_pairs)
        accepted_rows = (pair_logits.detach() > 0).cpu().tolist()

        specs_by_scene, topology_rows = [], []
        for batch_index, tree in enumerate(trees):
            pair_by_left = {u: (pair_index, v, kind) for pair_index, (u, v, kind) in enumerate(tree.pairs)}
            accepted = accepted_rows[batch_index]
            specs, active_keys, accepted_meta, archive_map = [], [], [], []
            slot = 0
            while slot < len(tree.groups):
                pair = pair_by_left.get(slot)
                if cut_active and pair is not None and accepted[pair[0]]:
                    pair_index, right, kind = pair
                    members = tree.groups[slot] + tree.groups[right]
                    key = self._record_key(kind, list(tree.keys[slot][1] + tree.keys[right][1]))
                    specs.append((members, kind))
                    active_keys.append(key)
                    archive_map.extend(((tree.keys[slot], key), (tree.keys[right], key)))
                    accepted_meta.append(tree.pairs[pair_index])
                    slot = right + 1
                else:
                    specs.append((tree.groups[slot], tree.kinds[slot]))
                    active_keys.append(tree.keys[slot])
                    slot += 1
            specs_by_scene.append(specs)
            topology_rows.append(SceneGroupTopology(
                tuple(members for members, _ in specs), tuple(kind for _, kind in specs), tuple(active_keys),
                tuple(tree.pairs), tuple(tree.pairs[i] for i in range(len(tree.pairs)) if accepted[i]),
                tuple(accepted_meta), tuple(archive_map),
                tuple((record_id, key) for key in active_keys for record_id in key[1])))
        max_groups = max((len(specs) for specs in specs_by_scene), default=0)
        ancestry_rows, ancestry_mask_rows, type_rows, mask_rows = [], [], [], []
        for specs in specs_by_scene:
            count = len(specs)
            ancestry_rows.append([list(members) + [0] * (2 - len(members)) for members, _ in specs]
                                  + [[0, 0]] * (max_groups - count))
            ancestry_mask_rows.append([[True] * len(members) + [False] * (2 - len(members)) for members, _ in specs]
                                       + [[False, False]] * (max_groups - count))
            type_rows.append([kind for _, kind in specs] + [-1] * (max_groups - count))
            mask_rows.append([True] * count + [False] * (max_groups - count))
        ancestry = torch.tensor(ancestry_rows, device=device, dtype=torch.long).reshape(batch, max_groups, 2)
        ancestry_mask = torch.tensor(ancestry_mask_rows, device=device, dtype=torch.bool).reshape(batch, max_groups, 2)
        group_types = torch.tensor(type_rows, device=device, dtype=torch.long).reshape(batch, max_groups)
        group_mask = torch.tensor(mask_rows, device=device, dtype=torch.bool).reshape(batch, max_groups)
        native_states = torch.cat((organization.module_states, organization.environment_states), 1)
        descriptors = self._descriptors(organization, native_states, ancestry, ancestry_mask, group_types, group_mask)
        if not cut_active and alpha > 0 and max_pairs:
            pair_ancestry_mask = eligible_pairs[..., None].expand(-1, -1, 2)
            pair_descriptors = self._descriptors(organization, native_states, pair_indices,
                pair_ancestry_mask, pair_types, eligible_pairs)
            weight = (float(alpha) * pair_logits.sigmoid() * eligible_pairs)[..., None, None]
            blended = []
            for values, combined in zip(descriptors, pair_descriptors):
                width = values.shape[-1]
                leaf = self._gather_children(values, pair_slots)
                delta = weight * (combined[:, :, None] - leaf)
                scatter_indices = pair_slots.reshape(batch, max_pairs * 2, 1).expand(-1, -1, width)
                correction = torch.zeros_like(values).scatter_add(1, scatter_indices,
                    delta.reshape(batch, max_pairs * 2, width))
                blended.append(values + correction)
            descriptors = tuple(blended)
        controls, centroids, lengths, extents = descriptors
        bank_indices, bank_masks = [], []
        for bank in (0, 1):
            positions = [[index for index, kind in enumerate(row) if kind == bank] for row in type_rows]
            allocated = max((len(row) for row in positions), default=0)
            bank_indices.append(torch.tensor([row + [0] * (allocated - len(row)) for row in positions],
                device=device, dtype=torch.long).reshape(batch, allocated))
            bank_masks.append(torch.tensor([[True] * len(row) + [False] * (allocated - len(row))
                for row in positions], device=device, dtype=torch.bool).reshape(batch, allocated))
        membership = self._build_direct_membership(organization, controls, centroids, lengths,
            bank_indices, bank_masks)
        return CompactOrganization(membership, group_mask, group_types, controls, centroids, lengths, extents,
            pair_logits, eligible_pairs if not cut_active else torch.zeros_like(eligible_pairs),
            pair_slots, tuple(topology_rows), tuple(bank_indices), tuple(bank_masks))

    def query_access(self, compact: CompactOrganization, organization, query, query_states: Tensor,
                     local_envelope: Tensor, query_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if compact.bank_group_indices is None or compact.bank_group_masks is None:
            raise ValueError("Compact query scoring requires prepared typed group indices.")
        batch, queries = query_states.shape[:2]
        group_logits = query_states.new_zeros(batch, queries, compact.allocated_count)
        for bank, name in ((0, "M"), (1, "E")):
            indices, mask = compact.bank_group_indices[bank], compact.bank_group_masks[bank]
            controls = self._gather_slots(compact.controls, indices)
            centroids = self._gather_slots(compact.centroids, indices)
            lengths = self._gather_slots(compact.lengths, indices)
            raw = self.query_scorers[name](query, query_states, controls, centroids, lengths,
                organization.frame_scale)
            allowed = mask[:, None] & query_mask[:, :, None]
            calibration = self.query_calibration.module if bank == 0 else self.query_calibration.environment
            logits = calibration(raw, allowed)
            if bank == 1:
                logits = logits + self.query_calibration.environment_offset.to(logits)
            logits = logits * allowed
            group_logits = group_logits.scatter_add(2, indices[:, None].expand(-1, queries, -1), logits)
        allowed = compact.group_mask[:, None, :] & query_mask[:, :, None]
        weights = entmax15(group_logits / self.entmax_temperature.to(group_logits), allowed)
        access = torch.bmm(weights, compact.membership)
        access = access * organization.joint_records.eligible[:, None, :]
        module_count = organization.scene.num_modules
        module_gate = local_envelope + (1 - local_envelope) * access[..., :module_count]
        gates = torch.cat((module_gate, access[..., module_count:]), -1)
        gates = gates * organization.joint_records.eligible[:, None, :] * query_mask[..., None]
        return gates, weights, group_logits

    @staticmethod
    def structural_losses(compact: CompactOrganization, records) -> dict[str, Tensor]:
        """Equal-scene means with equal nonempty donor-type weighting per pair."""
        batch, pairs = compact.pair_mask.shape
        rows = compact.membership
        if not pairs:
            zero = rows.sum((1, 2)) * 0
            return {"tie": zero, "capacity": zero + 1}
        children = CompactGroupRetrieval._gather_children(rows, compact.pair_slots)
        delta = (children[:, :, 0] - children[:, :, 1]).abs()
        eligible = records.eligible
        module_count = records.module_count
        distance_sum = rows.new_zeros(batch, pairs)
        type_count = rows.new_zeros(batch, 1)
        for donor_slice in (slice(0, module_count), slice(module_count, rows.shape[-1])):
            valid = eligible[:, donor_slice]
            count = valid.sum(-1, keepdim=True)
            distance_sum = distance_sum + (delta[:, :, donor_slice] * valid[:, None]).sum(-1) / count.clamp_min(1)
            type_count = type_count + (count > 0).to(rows)
        pair_discrepancy = distance_sum / type_count.clamp_min(1)
        probability = compact.pair_logits.sigmoid() * compact.pair_mask
        tie = (probability * pair_discrepancy).sum(-1) / compact.pair_mask.sum(-1).clamp_min(1)
        present = records.present.sum(-1).to(rows)
        capacity = (present - probability.sum(-1)) / present.clamp_min(1)
        return {"tie": tie, "capacity": capacity}


__all__ = [
    "CompactGroupRetrieval", "CompactOrganization", "SceneGroupTopology",
    "SiblingCompatibility", "GroupControlProjector",
]
