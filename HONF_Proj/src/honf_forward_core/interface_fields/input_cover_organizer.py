"""Input-only amortization of case-local adaptive cover decisions.

The organizer consumes encoded physical inputs and current model-side states.
It has no interface for reference targets, case identifiers, oracle reports,
or validation arrays. During fitting it emits soft covers; in evaluation it
converts pre-activation scores to deterministic binary decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .adaptive_interaction_cover import AdaptiveCoverPlan, CaseLocalReceiverTree
from .types import EncodedInterfaceCase


@dataclass(frozen=True)
class OrganizerScores:
    """Shared pre-activation scores for one physical case."""

    split_logits: torch.Tensor  # [N], meaningful only at internal tree nodes
    module_logits: torch.Tensor  # [N,M]
    environment_logits: torch.Tensor  # [N,E]


def masked_binary_logit_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    observed: torch.Tensor,
    *,
    balance_classes: bool = False,
) -> torch.Tensor:
    """Binary cross entropy over oracle-observed decisions only.

    With ``balance_classes=True``, average the positive and negative class
    losses separately before combining them. This prevents the many retained
    sources in a sparse cover from swamping the observed prune decisions. If
    the observed decisions contain only one class, its ordinary mean is used.
    """

    if logits.shape != targets.shape or logits.shape != observed.shape:
        raise ValueError("organizer labels and observation mask must match the score shape")
    if observed.dtype != torch.bool:
        raise TypeError("organizer observation masks must be boolean")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("organizer logits must be finite")
    if bool((observed & ~torch.isfinite(targets)).any()):
        raise ValueError("observed organizer labels must be finite")
    selected_targets = targets[observed]
    if bool(((selected_targets < 0) | (selected_targets > 1)).any()):
        raise ValueError("observed organizer labels must lie in [0,1]")
    if selected_targets.numel() == 0:
        return logits.sum() * 0.0
    selected_logits = logits[observed]
    selected_targets = selected_targets.to(logits.dtype)
    losses = F.binary_cross_entropy_with_logits(selected_logits, selected_targets, reduction="none")
    if not balance_classes:
        return losses.mean()
    positive = selected_targets >= 0.5
    negative = ~positive
    class_losses = []
    if bool(positive.any()):
        class_losses.append(losses[positive].mean())
    if bool(negative.any()):
        class_losses.append(losses[negative].mean())
    return torch.stack(class_losses).mean()


class InputOnlyCoverOrganizer(nn.Module):
    """Shared node/source scorer that returns exact deterministic cover plans.

    ``plan_cases`` accepts only an ``EncodedInterfaceCase``, a mapping of
    current model-side tensors, and geometry-built trees. The core adapter
    must pass no target fields, case names/IDs, oracle labels, or split data.
    Calling ``train()`` selects differentiable soft plans for predictive-loss
    fitting; calling ``eval()`` selects hard plans for actual K/support
    measurement and inference.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        module_feature_dim: int,
        environment_feature_dim: int,
        hidden_dim: int = 96,
        role_count: int = 8,
        source_chunk_size: int = 128,
        split_threshold: float = 0.5,
        source_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if min(state_dim, hidden_dim, role_count, source_chunk_size) < 1:
            raise ValueError("organizer dimensions and source chunk size must be positive")
        if min(module_feature_dim, environment_feature_dim) < 0:
            raise ValueError("organizer feature dimensions cannot be negative")
        if not 0.0 < split_threshold < 1.0 or not 0.0 < source_threshold < 1.0:
            raise ValueError("organizer hard-decision thresholds must be between zero and one")
        self.state_dim = int(state_dim)
        self.module_feature_dim = int(module_feature_dim)
        self.environment_feature_dim = int(environment_feature_dim)
        self.source_feature_dim = max(self.module_feature_dim, self.environment_feature_dim)
        self.role_count = int(role_count)
        self.source_chunk_size = int(source_chunk_size)
        self.split_threshold = float(split_threshold)
        self.source_threshold = float(source_threshold)

        node_input_dim = 6 + self.role_count + 1 + self.state_dim
        source_input_dim = self.source_feature_dim + self.state_dim + 1 + 2
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.source_encoder = nn.Sequential(
            nn.Linear(source_input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.split_head = nn.Linear(hidden_dim, 1)
        self.pair_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _state(prepared_state: Any, name: str, fallback: torch.Tensor) -> torch.Tensor:
        if isinstance(prepared_state, Mapping):
            value = prepared_state.get(name)
        else:
            value = getattr(prepared_state, name, None)
        return fallback if value is None else value

    @staticmethod
    def _case_scale(scale: torch.Tensor, case: int) -> torch.Tensor:
        if scale.ndim == 1:
            return scale
        if scale.ndim == 2:
            return scale[case]
        if scale.ndim == 3 and int(scale.shape[1]) == 1:
            return scale[case, 0]
        raise ValueError("coordinate scale must provide one physical vector per case")

    @staticmethod
    def _pad_features(features: torch.Tensor | None, count: int, width: int, like: torch.Tensor) -> torch.Tensor:
        if width == 0:
            return like.new_zeros((count, 0))
        if features is None:
            return like.new_zeros((count, width))
        if features.ndim != 2 or int(features.shape[0]) != count or int(features.shape[1]) > width:
            raise ValueError("source features do not align with the configured organizer width")
        features = features.to(device=like.device, dtype=like.dtype)
        if int(features.shape[1]) == width:
            return features
        return F.pad(features, (0, width - int(features.shape[1])))

    def _node_embeddings(
        self,
        tree: CaseLocalReceiverTree,
        global_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        universe = tree.universe
        coordinates = universe.coordinates.to(device=global_state.device, dtype=global_state.dtype)
        weights = universe.weights.to(device=global_state.device, dtype=global_state.dtype)
        roles = universe.roles.to(device=global_state.device)
        if bool(((roles < 0) | (roles >= self.role_count)).any()):
            raise ValueError("receiver-anchor role code exceeds the configured organizer vocabulary")
        scale = universe.coordinate_scale.to(device=coordinates.device, dtype=coordinates.dtype)
        centers: list[torch.Tensor] = []
        extents: list[torch.Tensor] = []
        role_features: list[torch.Tensor] = []
        masses: list[torch.Tensor] = []
        for node in tree.nodes:
            indices = torch.as_tensor(node.anchor_indices, device=coordinates.device, dtype=torch.long)
            subset = coordinates.index_select(0, indices)
            subset_weights = weights.index_select(0, indices)
            subset_roles = roles.index_select(0, indices).to(torch.long)
            centers.append(subset.mean(dim=0) / scale)
            extents.append((subset.max(dim=0).values - subset.min(dim=0).values) / scale)
            role_features.append(F.one_hot(subset_roles, num_classes=self.role_count).to(coordinates.dtype).mean(dim=0))
            mean_weight = weights.mean().clamp_min(torch.finfo(weights.dtype).tiny)
            masses.append(torch.log1p(subset_weights.sum() / mean_weight).reshape(1))
        center_tensor = torch.stack(centers)
        extent_tensor = torch.stack(extents)
        role_tensor = torch.stack(role_features)
        mass_tensor = torch.cat(masses)
        global_rows = global_state.reshape(1, -1).expand(len(tree.nodes), -1)
        features = torch.cat(
            (center_tensor, extent_tensor, role_tensor, mass_tensor[:, None], global_rows), dim=-1
        )
        return self.node_encoder(features), center_tensor, extent_tensor

    def _source_embeddings(
        self,
        *,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
        module_features: torch.Tensor,
        environment_features: torch.Tensor | None,
        environment_weights: torch.Tensor,
        module_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype, device = module_states.dtype, module_states.device
        modules = module_states.shape[0]
        environments = environment_states.shape[0]
        if module_states.ndim != 2 or int(module_states.shape[1]) != self.state_dim:
            raise ValueError("module states do not match the organizer state dimension")
        if environment_states.shape != (environments, self.state_dim):
            raise ValueError("environment states do not match the organizer state dimension")
        if module_present.shape != (modules,):
            raise ValueError("module presence must align with input module states")
        if environment_weights.shape != (environments,) or bool((environment_weights <= 0).any()):
            raise ValueError("environment weights must be positive and align with environment states")
        module_feature_values = self._pad_features(module_features, modules, self.source_feature_dim, module_states)
        environment_feature_values = self._pad_features(
            environment_features, environments, self.source_feature_dim, environment_states
        )
        module_mass = torch.log1p(module_present.to(dtype).clamp_min(0.0))[:, None]
        normalized_environment_weights = environment_weights.to(device=device, dtype=dtype)
        normalized_environment_weights = normalized_environment_weights / normalized_environment_weights.mean()
        environment_mass = torch.log(normalized_environment_weights.clamp_min(torch.finfo(dtype).tiny))[:, None]
        module_type = module_states.new_tensor((1.0, 0.0)).expand(modules, -1)
        environment_type = environment_states.new_tensor((0.0, 1.0)).expand(environments, -1)
        module_input = torch.cat((module_feature_values, module_states, module_mass, module_type), dim=-1)
        environment_input = torch.cat(
            (environment_feature_values, environment_states, environment_mass, environment_type), dim=-1
        )
        return self.source_encoder(module_input), self.source_encoder(environment_input)

    def score_cases(
        self,
        encoded: EncodedInterfaceCase,
        prepared_state: Mapping[str, Any] | Any,
        trees: Sequence[CaseLocalReceiverTree],
    ) -> tuple[OrganizerScores, ...]:
        """Return shared pre-activation scores for an input-only case batch."""

        batch_size = int(encoded.module_centers.shape[0])
        if len(trees) != batch_size:
            raise ValueError("one geometry tree is required per encoded case")
        module_states = self._state(prepared_state, "module_states", encoded.module_tokens)
        environment_states = self._state(prepared_state, "environment_states", encoded.env_tokens)
        global_states = self._state(prepared_state, "global_state", encoded.global_token)
        if module_states.shape[:2] != encoded.module_present.shape:
            raise ValueError("prepared module states must align with encoded module inputs")
        if environment_states.shape[:2] != encoded.env_coords.shape[:2]:
            raise ValueError("environment states must align with encoded environment inputs")
        if global_states.shape[0] != batch_size:
            raise ValueError("one current global model state is required per encoded case")
        results: list[OrganizerScores] = []
        model_parameter = next(self.parameters())
        model_dtype, model_device = model_parameter.dtype, model_parameter.device
        module_states = module_states.to(device=model_device, dtype=model_dtype)
        environment_states = environment_states.to(device=model_device, dtype=model_dtype)
        global_states = global_states.to(device=model_device, dtype=model_dtype)
        for case, tree in enumerate(trees):
            node_embedding, node_center, _extent = self._node_embeddings(
                tree, global_states[case]
            )
            module_state = module_states[case]
            environment_state = environment_states[case]
            module_feature = encoded.module_features[case].to(device=model_device, dtype=model_dtype)
            environment_feature = (
                None
                if encoded.env_features is None
                else encoded.env_features[case].to(device=model_device, dtype=model_dtype)
            )
            module_present = encoded.module_present[case].to(device=model_device, dtype=model_dtype)
            module_embedding, environment_embedding = self._source_embeddings(
                module_states=module_state,
                environment_states=environment_state,
                module_features=module_feature,
                environment_features=environment_feature,
                environment_weights=encoded.env_weights[case].to(device=model_device, dtype=model_dtype),
                module_present=module_present,
            )
            scale = self._case_scale(encoded.coordinate_scale, case).to(
                device=node_center.device, dtype=node_center.dtype
            )
            module_relative = (
                encoded.module_centers[case].to(device=model_device, dtype=model_dtype)[None, :, :]
                - (node_center[:, None, :] * scale)
            ) / scale
            environment_relative = (
                encoded.env_coords[case].to(device=model_device, dtype=model_dtype)[None, :, :]
                - (node_center[:, None, :] * scale)
            ) / scale
            split_logits = self.split_head(node_embedding).squeeze(-1)
            module_logits = self._score_sources(node_embedding, module_embedding, module_relative)
            environment_logits = self._score_sources(node_embedding, environment_embedding, environment_relative)
            module_logits = module_logits.masked_fill(~(module_present > 0.5)[None, :], -30.0)
            results.append(OrganizerScores(split_logits, module_logits, environment_logits))
        return tuple(results)

    def _score_sources(
        self,
        nodes: torch.Tensor,
        sources: torch.Tensor,
        relative: torch.Tensor,
    ) -> torch.Tensor:
        node_count, source_count = int(nodes.shape[0]), int(sources.shape[0])
        chunks: list[torch.Tensor] = []
        for start in range(0, source_count, self.source_chunk_size):
            stop = min(source_count, start + self.source_chunk_size)
            node_block = nodes[:, None, :].expand(-1, stop - start, -1)
            source_block = sources[None, start:stop, :].expand(node_count, -1, -1)
            relative_block = relative[:, start:stop, :]
            pair_features = torch.cat((node_block, source_block, relative_block), dim=-1)
            chunks.append(self.pair_scorer(pair_features).squeeze(-1))
        return torch.cat(chunks, dim=1)

    def plans_from_scores(
        self,
        scores: Sequence[OrganizerScores],
        encoded: EncodedInterfaceCase,
        trees: Sequence[CaseLocalReceiverTree],
        *,
        hard: bool,
    ) -> tuple[AdaptiveCoverPlan, ...]:
        if len(scores) != len(trees) or len(trees) != int(encoded.module_present.shape[0]):
            raise ValueError("scores, geometry trees, and encoded cases must align")
        plans: list[AdaptiveCoverPlan] = []
        for case, (item, tree) in enumerate(zip(scores, trees, strict=True)):
            internal = torch.tensor(
                [not node.is_leaf for node in tree.nodes], device=item.split_logits.device, dtype=torch.bool
            )
            if hard:
                split = (item.split_logits >= torch.logit(item.split_logits.new_tensor(self.split_threshold))).to(
                    item.split_logits.dtype
                )
                module = (item.module_logits >= torch.logit(item.module_logits.new_tensor(self.source_threshold))).to(
                    item.module_logits.dtype
                )
                environment = (
                    item.environment_logits >= torch.logit(
                        item.environment_logits.new_tensor(self.source_threshold)
                    )
                ).to(item.environment_logits.dtype)
            else:
                split = torch.sigmoid(item.split_logits)
                module = torch.sigmoid(item.module_logits)
                environment = torch.sigmoid(item.environment_logits)
            split = split * internal.to(split.dtype)
            present = encoded.module_present[case].to(device=module.device) > 0.5
            module = module * present.to(module.dtype)[None, :]
            plans.append(AdaptiveCoverPlan(tree, split, module, environment))
        return tuple(plans)

    def plan_cases(
        self,
        encoded: EncodedInterfaceCase,
        prepared_state: Mapping[str, Any] | Any,
        trees: Sequence[CaseLocalReceiverTree],
    ) -> tuple[AdaptiveCoverPlan, ...]:
        """Implement the native policy API; train mode is soft, eval is hard."""

        scores = self.score_cases(encoded, prepared_state, trees)
        return self.plans_from_scores(scores, encoded, trees, hard=not self.training)

    def hard_plan_cases(
        self,
        encoded: EncodedInterfaceCase,
        prepared_state: Mapping[str, Any] | Any,
        trees: Sequence[CaseLocalReceiverTree],
    ) -> tuple[AdaptiveCoverPlan, ...]:
        scores = self.score_cases(encoded, prepared_state, trees)
        return self.plans_from_scores(scores, encoded, trees, hard=True)


__all__ = ["InputOnlyCoverOrganizer", "OrganizerScores", "masked_binary_logit_loss"]
