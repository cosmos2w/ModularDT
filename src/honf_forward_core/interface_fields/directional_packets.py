"""Opt-in directional packet scorer for typed receiver-local permissions.

The historical :class:`InputOnlyCoverOrganizer` remains untouched. This
module builds a separate MM receiver index from active module coordinates,
scores that index, then compiles only its MM result to exact native pair
access on the original case plan. This preserves the environmental anchor
identity guard and leaves every other typed route at explicit full access.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from .adaptive_interaction_cover import CaseLocalReceiverTree, MechanismPlan, ReceiverAnchorUniverse
from .input_cover_organizer import InputOnlyCoverOrganizer, OrganizerScores
from .types import EncodedInterfaceCase

FEATURE_WIDTH = 13
GEOMETRY_FLOOR = 1.0e-3
GEOMETRY_LENGTH_SCALE = 1.0  # WindFarm coordinates are already in rotor diameters.


@dataclass(frozen=True)
class DirectionalPacketScores(OrganizerScores):
    """Organizer logits plus the scoring-local typed MM index."""

    receiver_tree: CaseLocalReceiverTree | None = None
    active_module_ids: torch.Tensor | None = None
    geometry_prior: torch.Tensor | None = None
    raw_pair_features: torch.Tensor | None = None


def _transform_coordinates(coordinates: torch.Tensor, frame: torch.Tensor | None) -> torch.Tensor:
    dimension = int(coordinates.shape[-1])
    if frame is None:
        return coordinates
    if frame.shape != (dimension, dimension):
        raise ValueError("physical frame must be square and match the coordinate dimension")
    matrix = frame.to(device=coordinates.device, dtype=coordinates.dtype)
    identity = torch.eye(dimension, device=matrix.device, dtype=matrix.dtype)
    if not torch.allclose(matrix.transpose(0, 1) @ matrix, identity, atol=1.0e-5, rtol=1.0e-5):
        raise ValueError("physical frame must be orthonormal")
    return coordinates @ matrix


def _bounded_depth_two_tree(
    encoded: EncodedInterfaceCase,
    case: int,
    *,
    physical_frame: torch.Tensor | None = None,
) -> tuple[CaseLocalReceiverTree, torch.Tensor]:
    present = encoded.module_present[case] > 0.5
    active_ids = torch.nonzero(present, as_tuple=False).flatten()
    if active_ids.numel() < 2:
        raise ValueError("directed MM packets require at least two active module receivers")
    centers = _transform_coordinates(encoded.module_centers[case].index_select(0, active_ids), physical_frame)
    dtype, device = centers.dtype, centers.device
    count = int(active_ids.numel())
    universe = ReceiverAnchorUniverse(
        coordinates=centers,
        weights=torch.ones((count,), device=device, dtype=dtype),
        roles=torch.zeros((count,), device=device, dtype=torch.long),
        coordinate_scale=torch.ones((centers.shape[-1],), device=device, dtype=dtype),
    )
    tree = CaseLocalReceiverTree.build(
        universe,
        max_nodes=7,
        min_leaf_anchors=max(1, (count + 3) // 4),
        overlap_fraction=0.06,
        max_depth=2,
    )
    return tree, active_ids


class DirectionalPacketOrganizer(InputOnlyCoverOrganizer):
    """Small learned correction to the exact MM packet geometry prior.

    ``feature_mode='directional'`` retains signed physical axes. The matched
    ``'summary'`` control retains the same thirteen-wide scorer and packet
    geometry, but replaces the seven relative-coordinate features with the
    five axis-symmetric legacy summaries plus two fixed zeros.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        module_feature_dim: int,
        environment_feature_dim: int,
        hidden_dim: int = 96,
        feature_mode: str = "directional",
        geometry_length_scale: float = GEOMETRY_LENGTH_SCALE,
        scorer_lr: float = 1.0e-3,
        physical_frame: torch.Tensor | None = None,
    ) -> None:
        if feature_mode not in {"directional", "summary"}:
            raise ValueError("feature_mode must be 'directional' or 'summary'")
        if not torch.isfinite(torch.tensor(float(geometry_length_scale))) or geometry_length_scale <= 0:
            raise ValueError("geometry_length_scale must be finite and positive")
        super().__init__(
            state_dim=state_dim,
            module_feature_dim=module_feature_dim,
            environment_feature_dim=environment_feature_dim,
            hidden_dim=hidden_dim,
            role_count=8,
            source_chunk_size=128,
        )
        self.feature_mode = feature_mode
        self.geometry_length_scale = float(geometry_length_scale)
        self.scorer_lr = float(scorer_lr)
        if physical_frame is not None:
            frame = torch.as_tensor(physical_frame, dtype=torch.float32)
            if frame.ndim != 2 or frame.shape[0] != frame.shape[1]:
                raise ValueError("physical_frame must be a square matrix")
            if not torch.allclose(
                frame.T @ frame,
                torch.eye(frame.shape[0], dtype=frame.dtype),
                atol=1.0e-5,
                rtol=1.0e-5,
            ):
                raise ValueError("physical_frame must be orthonormal")
        else:
            frame = torch.empty((0, 0), dtype=torch.float32)
        self.register_buffer("_physical_frame", frame, persistent=False)
        # The legacy class dimensions and state-dict keys are not modified.
        # This opt-in model adds an actual-receiver state summary to each node.
        node_input_dim = 10 + self.role_count + 1 + 2 * self.state_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.directional_pair_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim + FEATURE_WIDTH, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        final = self.directional_pair_scorer[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.register_buffer("_progress", torch.zeros((), dtype=torch.float32), persistent=False)
        # Inherited legacy heads are present only for API compatibility; the
        # opt-in workflow trains its own typed scorer and embeddings below.
        for module in (self.split_head, self.pair_scorers):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def set_update(self, update: int) -> None:
        """Set residual ramp from the one-based optimizer update."""

        u = max(0, int(update))
        phase = min(float(u) / 100.0, 1.0)
        self._progress.copy_(0.5 - 0.5 * torch.cos(self._progress.new_tensor(torch.pi * phase)))

    @property
    def residual_ramp(self) -> float:
        return float(self._progress.detach().cpu())

    @staticmethod
    def _summary3(relative: torch.Tensor) -> torch.Tensor:
        return InputOnlyCoverOrganizer._spatial_summary(relative)

    def _receiver_node_embeddings(
        self,
        tree: CaseLocalReceiverTree,
        module_states: torch.Tensor,
        global_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        universe = tree.universe
        coordinates = universe.coordinates.to(device=module_states.device, dtype=module_states.dtype)
        weights = universe.weights.to(device=module_states.device, dtype=module_states.dtype)
        roles = universe.roles.to(device=module_states.device)
        scale = universe.coordinate_scale.to(device=coordinates.device, dtype=coordinates.dtype)
        centers: list[torch.Tensor] = []
        extents: list[torch.Tensor] = []
        role_rows: list[torch.Tensor] = []
        masses: list[torch.Tensor] = []
        state_summaries: list[torch.Tensor] = []
        for node in tree.nodes:
            ids = torch.as_tensor(node.anchor_indices, device=coordinates.device, dtype=torch.long)
            subset = coordinates.index_select(0, ids)
            subset_weights = weights.index_select(0, ids)
            subset_roles = roles.index_select(0, ids).to(torch.long)
            normalized = subset_weights / subset_weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
            centers.append((subset * normalized[:, None]).sum(dim=0) / scale)
            extents.append((subset.max(dim=0).values - subset.min(dim=0).values) / scale)
            role_rows.append(
                torch.nn.functional.one_hot(subset_roles, num_classes=self.role_count).to(coordinates.dtype).mean(dim=0)
            )
            masses.append(torch.log1p(subset_weights.sum() / weights.mean().clamp_min(1e-12)).reshape(1))
            receiver_states = module_states.index_select(0, ids)
            state_summaries.append((receiver_states * normalized[:, None]).sum(dim=0))
        center_tensor = torch.stack(centers)
        extent_tensor = torch.stack(extents)
        role_tensor = torch.stack(role_rows)
        mass_tensor = torch.cat(masses)
        receiver_state_tensor = torch.stack(state_summaries)
        global_rows = global_state.reshape(1, -1).expand(len(tree.nodes), -1)
        features = torch.cat(
            (
                self._spatial_summary(center_tensor),
                self._spatial_summary(extent_tensor),
                role_tensor,
                mass_tensor[:, None],
                receiver_state_tensor,
                global_rows,
            ),
            dim=-1,
        )
        return self.node_encoder(features), center_tensor, extent_tensor

    def packet_features(
        self,
        tree: CaseLocalReceiverTree,
        active_module_ids: torch.Tensor,
        module_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Return [packet, padded module source, 13] pair geometry features."""

        source_count = int(module_centers.shape[0])
        dimension = int(module_centers.shape[-1])
        if dimension not in {2, 3}:
            raise ValueError("directional packets support explicit 2-D or 3-D geometry")
        frame = None if self._physical_frame.numel() == 0 else self._physical_frame
        centers = _transform_coordinates(module_centers, frame)
        length = self.geometry_length_scale
        feature_rows: list[torch.Tensor] = []
        for node in tree.nodes:
            receiver_local = torch.as_tensor(node.anchor_indices, device=centers.device, dtype=torch.long)
            receiver_ids = active_module_ids.index_select(0, receiver_local)
            receiver_coords = centers.index_select(0, receiver_ids)
            packet_center = receiver_coords.mean(dim=0)
            packet_extent = (receiver_coords.max(dim=0).values - receiver_coords.min(dim=0).values) / length
            # Wind's axes are all observed and already wind aligned. In 2-D,
            # the third padded component is explicitly absent.
            axis_valid = torch.ones((dimension,), device=centers.device, dtype=centers.dtype)
            if dimension == 2:
                axis_valid = torch.cat((axis_valid, axis_valid.new_zeros(1)))
                packet_extent = torch.cat((packet_extent, packet_extent.new_zeros(1)))
            relative = (centers - packet_center[None, :]) / length
            if dimension == 2:
                relative = torch.cat((relative, relative.new_zeros((source_count, 1))), dim=-1)
            relative_square = relative.square()
            if self.feature_mode == "directional":
                relative_features = torch.cat(
                    (relative, relative.abs(), relative_square.sum(dim=-1, keepdim=True)), dim=-1
                )
            else:
                symmetric = self._summary3(relative)
                relative_features = torch.cat((symmetric, symmetric.new_zeros((source_count, 2))), dim=-1)
            extent_rows = packet_extent[None, :].expand(source_count, -1)
            axis_rows = axis_valid[None, :].expand(source_count, -1)
            packet_row = torch.cat((relative_features, extent_rows, axis_rows), dim=-1)
            if packet_row.shape[-1] != FEATURE_WIDTH:
                raise RuntimeError("directional packet feature width changed unexpectedly")
            feature_rows.append(packet_row)
        return torch.stack(feature_rows)

    def _geometry_prior(
        self,
        tree: CaseLocalReceiverTree,
        active_module_ids: torch.Tensor,
        module_centers: torch.Tensor,
        module_present: torch.Tensor,
    ) -> torch.Tensor:
        frame = None if self._physical_frame.numel() == 0 else self._physical_frame
        coordinates = _transform_coordinates(module_centers, frame)
        valid_ids = torch.nonzero(module_present > 0.5, as_tuple=False).flatten()
        eligible = coordinates.index_select(0, valid_ids)
        result: list[torch.Tensor] = []
        for node in tree.nodes:
            local = torch.as_tensor(node.anchor_indices, device=module_centers.device, dtype=torch.long)
            actual_ids = active_module_ids.index_select(0, local)
            receiver_subset = coordinates.index_select(0, actual_ids)
            delta = (eligible[None, :, :] - receiver_subset[:, None, :]) / self.geometry_length_scale
            raw = -delta.square().sum(dim=-1).mean(dim=0)
            mean = raw.detach().mean()
            std = raw.detach().std(unbiased=False).clamp_min(GEOMETRY_FLOOR)
            normalized = (raw - mean) / std
            row = module_centers.new_zeros((module_centers.shape[0],))
            row = row.index_copy(0, valid_ids, normalized)
            result.append(row)
        return torch.stack(result)

    def score_cases(
        self,
        encoded: EncodedInterfaceCase,
        prepared_state: Any,
        trees: Sequence[CaseLocalReceiverTree],
        *,
        budgets: Any = None,
    ) -> tuple[DirectionalPacketScores, ...]:
        del budgets
        if len(trees) != int(encoded.module_present.shape[0]):
            raise ValueError("one original base tree is required per encoded case")
        module_states = self._state(prepared_state, "module_states", encoded.module_tokens)
        global_states = self._state(prepared_state, "global_state", encoded.global_token)
        results: list[DirectionalPacketScores] = []
        for case, base_tree in enumerate(trees):
            del base_tree  # Scoring uses MM-local receivers; compilation keeps the original tree.
            frame = None if self._physical_frame.numel() == 0 else self._physical_frame
            local_tree, active_ids = _bounded_depth_two_tree(encoded, case, physical_frame=frame)
            module_state = module_states[case].to(device=encoded.module_centers.device)
            global_state = global_states[case].to(device=encoded.module_centers.device)
            node_embedding, _node_center, _extent = self._receiver_node_embeddings(
                local_tree, module_state.index_select(0, active_ids), global_state
            )
            module_feature = self._pad_features(
                encoded.module_features[case],
                int(encoded.module_present.shape[1]),
                self.source_feature_dim,
                module_state,
            )
            source_state, _unused = self._source_embeddings(
                module_states=module_state,
                environment_states=encoded.env_tokens[case].to(device=module_state.device),
                module_features=module_feature,
                environment_features=None,
                environment_coordinates=encoded.env_coords[case].to(device=module_state.device),
                environment_weights=encoded.env_weights[case].to(device=module_state.device),
                module_present=encoded.module_present[case].to(device=module_state.device),
            )
            pair_features = self.packet_features(local_tree, active_ids, encoded.module_centers[case])
            pair = torch.cat(
                (
                    node_embedding[:, None, :].expand(-1, pair_features.shape[1], -1),
                    source_state[None, :, :].expand(pair_features.shape[0], -1, -1),
                    pair_features,
                ),
                dim=-1,
            )
            residual = self.directional_pair_scorer(pair).squeeze(-1)
            valid = encoded.module_present[case].to(device=residual.device) > 0.5
            prior = self._geometry_prior(
                local_tree,
                active_ids,
                encoded.module_centers[case],
                encoded.module_present[case],
            )
            logits = prior + self._progress.to(residual.dtype) * residual
            logits = logits.masked_fill(~valid[None, :], -30.0)
            zero_modules = logits.new_zeros(logits.shape)
            zero_env = logits.new_zeros((len(local_tree.nodes), int(encoded.env_coords.shape[1])))
            mechanism_logits = {
                "MM": logits,
                "ME": zero_env,
                "EM": zero_modules,
                "QM": zero_modules,
                "QE": zero_env,
            }
            results.append(
                DirectionalPacketScores(
                    split_logits=logits.new_zeros((len(local_tree.nodes),)),
                    module_logits=logits,
                    environment_logits=zero_env,
                    mechanism_logits=mechanism_logits,
                    node_embeddings=node_embedding,
                    module_embeddings=source_state,
                    environment_embeddings=None,
                    budget_vector=logits.new_tensor((0.9, 1.0, 1.0, 1.0, 1.0)),
                    receiver_tree=local_tree,
                    active_module_ids=active_ids,
                    geometry_prior=prior,
                    raw_pair_features=pair_features,
                )
            )
        return tuple(results)

    def plans_from_scores(
        self,
        scores: Sequence[OrganizerScores],
        encoded: EncodedInterfaceCase,
        trees: Sequence[CaseLocalReceiverTree],
        *,
        hard: bool,
        straight_through_hard: bool = False,
        frontier_cuts: Sequence[Sequence[int]] | None = None,
        budget_fractions: dict[str, float] | None = None,
    ) -> tuple[MechanismPlan, ...]:
        if len(scores) != len(trees):
            raise ValueError("scores and original base plans must have the same batch size")
        local_trees = tuple(item.receiver_tree for item in scores)
        if any(tree is None for tree in local_trees):
            raise ValueError("directional MM scores are missing their local receiver index")
        normalized_cuts = frontier_cuts
        if normalized_cuts is None:
            # Default evaluation is the single unsplit packet. Training and
            # action sweeps pass their selected complete cut explicitly.
            normalized_cuts = tuple((0,) for _tree in local_trees)
        elif len(normalized_cuts) != len(scores):
            raise ValueError("one MM cut must be supplied for each case")
        local = self._local_permission_plans(
            scores,
            encoded,
            hard=hard,
            straight_through_hard=straight_through_hard,
            frontier_cuts=normalized_cuts,
            budget_fractions=budget_fractions,
        )
        result: list[MechanismPlan] = []
        for case, (score, base_tree, local_plan) in enumerate(zip(scores, trees, local, strict=True)):
            assert isinstance(score, DirectionalPacketScores)
            result.append(self._compile_local_mm_plan(score, encoded, base_tree, case, local_plan))
        return tuple(result)

    def _local_permission_plans(
        self,
        scores: Sequence[OrganizerScores],
        encoded: EncodedInterfaceCase,
        *,
        hard: bool,
        straight_through_hard: bool = False,
        frontier_cuts: Sequence[Sequence[int]] | None = None,
        budget_fractions: dict[str, float] | None = None,
    ) -> tuple[MechanismPlan, ...]:
        """Build permissions on the MM-local index before native compilation."""

        local_trees = tuple(item.receiver_tree for item in scores)
        if any(tree is None for tree in local_trees):
            raise ValueError("directional MM scores are missing their local receiver index")
        normalized_cuts = frontier_cuts
        if normalized_cuts is None:
            # Default evaluation is the single unsplit packet. Training and
            # action sweeps pass their selected complete cut explicitly.
            normalized_cuts = tuple((0,) for _tree in local_trees)
        elif len(normalized_cuts) != len(scores):
            raise ValueError("one MM cut must be supplied for each case")
        frame = None if self._physical_frame.numel() == 0 else self._physical_frame
        local_module_centers = torch.stack(
            [
                _transform_coordinates(encoded.module_centers[case], frame)
                for case in range(int(encoded.module_centers.shape[0]))
            ]
        )
        # The geometry tree lives in the declared physical frame. The local
        # projection must use receiver coordinates in that same frame; the
        # resulting exact pair weights are compiled back onto the original
        # native module coordinate order below.
        local_encoded = replace(encoded, module_centers=local_module_centers)
        return InputOnlyCoverOrganizer.plans_from_scores(
            self,
            scores,
            local_encoded,
            local_trees,  # type: ignore[arg-type]
            hard=hard,
            straight_through_hard=straight_through_hard,
            frontier_cuts=normalized_cuts,
            budget_fractions=budget_fractions or {"MM": 0.9},
        )

    def _compile_local_mm_plan(
        self,
        score: DirectionalPacketScores,
        encoded: EncodedInterfaceCase,
        base_tree: CaseLocalReceiverTree,
        case: int,
        local_plan: MechanismPlan,
    ) -> MechanismPlan:
        if score.receiver_tree is None:
            raise ValueError("directional MM score is missing its receiver index")
        active = encoded.module_present[case] > 0.5
        mm_receivers = encoded.module_centers[case]
        frame = None if self._physical_frame.numel() == 0 else self._physical_frame
        local_receivers = _transform_coordinates(mm_receivers, frame)
        access = score.receiver_tree.access(local_receivers, local_plan.split_gates)
        membership = local_plan.permission_matrix("MM")
        direct_weights = (access @ membership).clamp(0.0, 1.0)
        # The native direct MM route applies source validity, receiver
        # validity, and exact self-exclusion at its established boundary.
        full = MechanismPlan.full_access(
            base_tree,
            encoded.module_present[case],
            int(encoded.env_coords.shape[1]),
        )
        return full.with_direct_pair_access(
            "MM",
            mm_receivers,
            direct_weights,
            receiver_validity=active,
        )


__all__ = [
    "FEATURE_WIDTH",
    "DirectionalPacketOrganizer",
    "DirectionalPacketScores",
    "_bounded_depth_two_tree",
]
