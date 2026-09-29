"""Input-only amortization of case-local adaptive cover decisions.

The organizer consumes encoded physical inputs and current model-side states.
It has no interface for reference targets, case identifiers, oracle reports,
or validation arrays. During fitting it emits soft covers; in evaluation it
converts pre-activation scores to deterministic binary decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .adaptive_interaction_cover import (
    INTERACTION_MECHANISMS,
    CaseLocalReceiverTree,
    MechanismPlan,
)
from .budgeted_frontier import (
    FrontierUtilityHead,
    FrontierUtilityPrediction,
    canonical_pair_catalog,
    enumerate_frontier_cuts,
    project_unique_pair_budget,
    split_gates_for_frontier,
)
from .types import EncodedInterfaceCase


@dataclass(frozen=True)
class OrganizerScores:
    """Shared pre-activation scores for one physical case."""

    split_logits: torch.Tensor  # [N], meaningful only at internal tree nodes
    module_logits: torch.Tensor  # [N,M]
    environment_logits: torch.Tensor  # [N,E]
    mechanism_logits: Mapping[str, torch.Tensor] = field(default_factory=dict)
    node_embeddings: torch.Tensor | None = None  # [N,H]
    module_embeddings: torch.Tensor | None = None  # [M,H]
    environment_embeddings: torch.Tensor | None = None  # [E,H]
    budget_vector: torch.Tensor | None = None  # [MM,ME,EM,QM,QE]


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
        quadrature_invariant_source_measure: bool = False,
        budget_dim: int = len(INTERACTION_MECHANISMS),
        frontier_utility_enabled: bool = False,
        frontier_role_count: int | None = None,
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
        self.quadrature_invariant_source_measure = bool(quadrature_invariant_source_measure)
        if budget_dim != len(INTERACTION_MECHANISMS):
            raise ValueError("budget_dim must provide one typed fraction for each mechanism")
        self.budget_dim = int(budget_dim)
        self.frontier_utility_enabled = bool(frontier_utility_enabled)
        if self.frontier_utility_enabled and frontier_role_count is None:
            raise ValueError(
                "frontier_role_count must be set from the case's physical error-role table"
            )
        if frontier_role_count is not None and frontier_role_count < 1:
            raise ValueError("frontier_role_count must be positive")
        self.frontier_role_count = (
            int(frontier_role_count) if frontier_role_count is not None else 0
        )

        # Geometry is summarized across its actual spatial axes, so one
        # organizer can consume both 2-D and 3-D input cases.
        node_input_dim = 10 + self.role_count + 1 + self.state_dim
        source_input_dim = self.source_feature_dim + self.state_dim + 1 + 2
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.source_encoder = nn.Sequential(
            nn.Linear(source_input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.split_head = nn.Linear(hidden_dim, 1)
        self.pair_scorers = nn.ModuleDict({
            mechanism: nn.Sequential(
                nn.Linear(2 * hidden_dim + 5, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for mechanism in INTERACTION_MECHANISMS
        })
        # The scalar budget is part of the encoded input through a parameter-
        # free multiplicative conditioning, preserving historical organizer
        # state-dict keys and exact all-access logits at budget 1. Frontier
        # utility is opt-in so legacy organizer checkpoints still load under
        # strict trust checks without missing-key exceptions.
        self.frontier_utility_head = (
            FrontierUtilityHead(
                hidden_dim=hidden_dim,
                budget_dim=self.budget_dim,
                role_count=self.frontier_role_count,
            )
            if self.frontier_utility_enabled
            else None
        )

    def _budget_matrix(
        self,
        budgets: Mapping[str, float | torch.Tensor] | torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalize explicit mechanism budgets to ``[B,5]`` fractions."""

        if budgets is None:
            return torch.ones((batch_size, self.budget_dim), device=device, dtype=dtype)
        if torch.is_tensor(budgets):
            values = budgets.to(device=device, dtype=dtype)
            if values.ndim == 1 and values.shape == (self.budget_dim,):
                values = values[None, :].expand(batch_size, -1)
            elif values.shape != (batch_size, self.budget_dim):
                raise ValueError("budget tensor must have shape [5] or [batch,5]")
        else:
            values = torch.ones((batch_size, self.budget_dim), device=device, dtype=dtype)
            for mechanism, raw in budgets.items():
                key = str(mechanism).upper()
                if key not in INTERACTION_MECHANISMS:
                    raise ValueError(f"unsupported mechanism budget {mechanism!r}")
                item = torch.as_tensor(raw, device=device, dtype=dtype)
                if item.numel() == 1:
                    values[:, INTERACTION_MECHANISMS.index(key)] = item.reshape(())
                elif item.shape == (batch_size,):
                    values[:, INTERACTION_MECHANISMS.index(key)] = item
                else:
                    raise ValueError("each mechanism budget must be scalar or have one value per case")
        if not bool(torch.isfinite(values).all()) or bool(((values < 0.0) | (values > 1.0)).any()):
            raise ValueError("all mechanism budgets must be finite fractions in [0,1]")
        return values

    @property
    def pair_scorer(self) -> nn.Module:
        """Compatibility alias for older diagnostics that inspect the QM head."""

        return self.pair_scorers["QM"]

    @staticmethod
    def _spatial_summary(values: torch.Tensor) -> torch.Tensor:
        """Map any nonempty coordinate axis to stable signed/scale summaries."""

        if values.ndim < 1 or int(values.shape[-1]) < 1:
            raise ValueError("spatial coordinates need at least one dimension")
        return torch.stack(
            (
                values.mean(dim=-1),
                values.abs().mean(dim=-1),
                values.square().mean(dim=-1).sqrt(),
                values.amax(dim=-1),
                values.amin(dim=-1),
            ),
            dim=-1,
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
            return scale[0 if int(scale.shape[0]) == 1 else case]
        if scale.ndim == 3 and int(scale.shape[1]) == 1:
            return scale[0 if int(scale.shape[0]) == 1 else case, 0]
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
        features = torch.cat((
            self._spatial_summary(center_tensor),
            self._spatial_summary(extent_tensor),
            role_tensor,
            mass_tensor[:, None],
            global_rows,
        ), dim=-1)
        return self.node_encoder(features), center_tensor, extent_tensor

    def _source_embeddings(
        self,
        *,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
        module_features: torch.Tensor,
        environment_features: torch.Tensor | None,
        environment_coordinates: torch.Tensor,
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
        if self.quadrature_invariant_source_measure:
            # A physical support decision must not depend on how one exact
            # quadrature atom is represented.  Group exact co-located atoms
            # with equal adapter features, then assign their summed measure
            # back to every equivalent source slot.  Normalize over unique
            # supports so splitting a slot changes neither its feature nor
            # the features of the other supports.  Native execution still
            # uses each original slot and its original quadrature weight.
            identity = torch.cat((
                environment_coordinates.to(device=device, dtype=dtype),
                environment_feature_values,
            ), dim=-1).detach()
            _unique_identity, inverse = torch.unique(
                identity, dim=0, sorted=True, return_inverse=True
            )
            group_weights = normalized_environment_weights.new_zeros(
                (int(_unique_identity.shape[0]),)
            )
            group_weights.index_add_(0, inverse, normalized_environment_weights)
            group_weights = group_weights / group_weights.mean().clamp_min(
                torch.finfo(dtype).tiny
            )
            normalized_environment_weights = group_weights.index_select(0, inverse)
        else:
            normalized_environment_weights = (
                normalized_environment_weights / normalized_environment_weights.mean()
            )
        environment_mass = torch.log(
            normalized_environment_weights.clamp_min(torch.finfo(dtype).tiny)
        )[:, None]
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
        *,
        budgets: Mapping[str, float | torch.Tensor] | torch.Tensor | None = None,
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
        budget_matrix = self._budget_matrix(
            budgets,
            batch_size=batch_size,
            device=model_device,
            dtype=model_dtype,
        )
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
                environment_coordinates=encoded.env_coords[case].to(
                    device=model_device, dtype=model_dtype
                ),
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
            mechanism_logits: dict[str, torch.Tensor] = {}
            for mechanism in INTERACTION_MECHANISMS:
                route_budget = budget_matrix[case, INTERACTION_MECHANISMS.index(mechanism)]
                if mechanism in {"MM", "EM", "QM"}:
                    logits = self._score_sources(
                        node_embedding, module_embedding, module_relative,
                        scorer=self.pair_scorers[mechanism],
                        mechanism=mechanism,
                        budget_fraction=route_budget,
                    )
                    logits = logits.masked_fill(~(module_present > 0.5)[None, :], -30.0)
                else:
                    logits = self._score_sources(
                        node_embedding, environment_embedding, environment_relative,
                        scorer=self.pair_scorers[mechanism],
                        mechanism=mechanism,
                        budget_fraction=route_budget,
                    )
                mechanism_logits[mechanism] = logits
            results.append(OrganizerScores(
                split_logits,
                mechanism_logits["QM"],
                mechanism_logits["QE"],
                mechanism_logits,
                node_embedding,
                module_embedding,
                environment_embedding,
                budget_matrix[case],
            ))
        return tuple(results)

    def _score_sources(
        self,
        nodes: torch.Tensor,
        sources: torch.Tensor,
        relative: torch.Tensor,
        *,
        scorer: nn.Module | None = None,
        mechanism: str = "QM",
        budget_fraction: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        node_count, source_count = int(nodes.shape[0]), int(sources.shape[0])
        budget = torch.as_tensor(budget_fraction, device=nodes.device, dtype=nodes.dtype)
        if budget.numel() != 1:
            raise ValueError("one scalar mechanism budget is required for each typed source score")
        conditioned_nodes = nodes * budget.reshape(1, 1)
        chunks: list[torch.Tensor] = []
        for start in range(0, source_count, self.source_chunk_size):
            stop = min(source_count, start + self.source_chunk_size)
            node_block = conditioned_nodes[:, None, :].expand(-1, stop - start, -1)
            source_block = sources[None, start:stop, :].expand(node_count, -1, -1)
            relative_block = relative[:, start:stop, :]
            pair_features = torch.cat((
                node_block,
                source_block,
                self._spatial_summary(relative_block),
            ), dim=-1)
            chunks.append((self.pair_scorer if scorer is None else scorer)(pair_features).squeeze(-1))
        return torch.cat(chunks, dim=1)

    def score_frontiers(
        self,
        scores: OrganizerScores,
        tree: CaseLocalReceiverTree,
        *,
        budget_vector: torch.Tensor | None = None,
        cuts: Sequence[Sequence[int]] | None = None,
        max_depth: int = 3,
    ) -> FrontierUtilityPrediction:
        """Predict Stage-B per-role distortion and utility for each cut."""

        if scores.node_embeddings is None:
            raise ValueError("frontier utility requires node embeddings from score_cases")
        if self.frontier_utility_head is None:
            raise RuntimeError(
                "frontier utility is disabled; construct the organizer with frontier_utility_enabled=True"
            )
        selected_cuts = tuple(cuts) if cuts is not None else enumerate_frontier_cuts(
            tree, max_depth=max_depth
        )
        budget = scores.budget_vector if budget_vector is None else budget_vector
        if budget is None:
            budget = scores.node_embeddings.new_ones((self.budget_dim,))
        return self.frontier_utility_head(scores.node_embeddings, selected_cuts, budget)

    def plans_from_scores(
        self,
        scores: Sequence[OrganizerScores],
        encoded: EncodedInterfaceCase,
        trees: Sequence[CaseLocalReceiverTree],
        *,
        hard: bool,
        straight_through_hard: bool = False,
        frontier_cuts: Sequence[Sequence[int]] | None = None,
        budget_fractions: Mapping[str, float] | None = None,
    ) -> tuple[MechanismPlan, ...]:
        """Build typed plans from scores.

        ``straight_through_hard`` keeps the hard thresholded topology in the
        forward pass while giving each binary gate the sigmoid surrogate
        gradient ``d sigmoid(logit) / d logit``. It requires ``hard=True``;
        the default soft training path and ordinary deterministic hard path
        are unchanged.
        """

        if straight_through_hard and not hard:
            raise ValueError("straight-through hard plans require hard=True")
        if len(scores) != len(trees) or len(trees) != int(encoded.module_present.shape[0]):
            raise ValueError("scores, geometry trees, and encoded cases must align")
        if frontier_cuts is not None and len(frontier_cuts) != len(trees):
            raise ValueError("one explicit frontier cut is required per case")
        normalized_budgets: dict[str, float] = {}
        for mechanism, fraction in (budget_fractions or {}).items():
            key = str(mechanism).upper()
            if key not in INTERACTION_MECHANISMS:
                raise ValueError(f"unsupported mechanism budget {mechanism!r}")
            value = float(fraction)
            if not 0.0 <= value <= 1.0:
                raise ValueError("budget fractions must lie in [0,1]")
            normalized_budgets[key] = value
        budgeted_trial = budget_fractions is not None
        plans: list[MechanismPlan] = []
        for case, (item, tree) in enumerate(zip(scores, trees, strict=True)):
            internal = torch.tensor(
                [not node.is_leaf for node in tree.nodes], device=item.split_logits.device, dtype=torch.bool
            )
            split_probabilities = torch.sigmoid(item.split_logits)
            if frontier_cuts is not None:
                split = split_gates_for_frontier(tree, frontier_cuts[case])
            elif hard:
                split_decisions = (
                    item.split_logits >= torch.logit(item.split_logits.new_tensor(self.split_threshold))
                ).to(item.split_logits.dtype)
                split = (
                    split_decisions + (split_probabilities - split_probabilities.detach())
                    if straight_through_hard else split_decisions
                )
            else:
                split = split_probabilities
            split = split * internal.to(split.dtype)
            present = encoded.module_present[case].to(device=split.device, dtype=split.dtype)
            typed_scores = item.mechanism_logits or {
                "MM": item.module_logits,
                "EM": item.module_logits,
                "QM": item.module_logits,
                "ME": item.environment_logits,
                "QE": item.environment_logits,
            }
            permissions: dict[str, torch.Tensor] = {}
            for mechanism in INTERACTION_MECHANISMS:
                logits = typed_scores[mechanism]
                fraction = normalized_budgets.get(mechanism)
                if budgeted_trial and fraction is None:
                    # Missing MechanismPlan keys have an explicit native
                    # full-access meaning. Keep unbudgeted routes as bypasses
                    # instead of accidentally thresholding their random init.
                    continue
                projection = None
                if fraction is not None:
                    catalog = canonical_pair_catalog(
                        encoded, tree, mechanism, case_index=case
                    )
                    projection = project_unique_pair_budget(
                        logits,
                        tree,
                        split.detach(),
                        budget_fraction=fraction,
                        source_validity=catalog.source_validity,
                        receiver_validity=catalog.receiver_validity,
                        pair_validity=catalog.pair_validity,
                        receiver_weights=catalog.receiver_weights,
                        receiver_coordinates=catalog.receiver_coordinates,
                    )
                if projection is not None:
                    threshold = projection.threshold.to(device=logits.device, dtype=logits.dtype)
                    if bool(torch.isinf(threshold)):
                        active_nodes = (tree.access(
                            canonical_pair_catalog(encoded, tree, mechanism, case_index=case).receiver_coordinates,
                            split.detach(),
                        ) > 0.0).any(dim=0)
                        valid_scores = logits.detach()[
                            active_nodes[:, None]
                            & catalog.source_validity.to(device=logits.device)[None, :]
                        ]
                        if valid_scores.numel():
                            # Empty hard projections still receive a finite,
                            # nonzero restoration signal in the soft shadow.
                            threshold = valid_scores.max() + 1.0
                    probabilities = torch.sigmoid(logits - threshold)
                    decisions = projection.membership.to(logits.dtype)
                    membership = (
                        decisions + (probabilities - probabilities.detach())
                        if straight_through_hard else decisions
                    ) if hard else probabilities
                elif hard:
                    probabilities = torch.sigmoid(logits)
                    decisions = (
                        logits >= torch.logit(logits.new_tensor(self.source_threshold))
                    ).to(logits.dtype)
                    membership = (
                        decisions + (probabilities - probabilities.detach())
                        if straight_through_hard else decisions
                    )
                else:
                    membership = torch.sigmoid(logits)
                if mechanism in {"MM", "EM", "QM"}:
                    membership = membership * (present > 0.5).to(membership.dtype)[None, :]
                permissions[mechanism] = membership
            plans.append(MechanismPlan(
                tree,
                split,
                present,
                int(encoded.env_coords.shape[1]),
                permissions,
            ))
        return tuple(plans)

    def plan_cases(
        self,
        encoded: EncodedInterfaceCase,
        prepared_state: Mapping[str, Any] | Any,
        trees: Sequence[CaseLocalReceiverTree],
        *,
        straight_through_hard: bool = False,
        budgets: Mapping[str, float | torch.Tensor] | torch.Tensor | None = None,
        frontier_cuts: Sequence[Sequence[int]] | None = None,
        budget_fractions: Mapping[str, float] | None = None,
    ) -> tuple[MechanismPlan, ...]:
        """Implement the native policy API; train mode is soft, eval is hard.

        Opt in with ``straight_through_hard=True`` during fitting to use hard
        forward masks and sigmoid surrogate gradients through each decision.
        """

        scores = self.score_cases(encoded, prepared_state, trees, budgets=budgets)
        hard = bool(straight_through_hard or not self.training)
        return self.plans_from_scores(
            scores,
            encoded,
            trees,
            hard=hard,
            straight_through_hard=bool(straight_through_hard),
            frontier_cuts=frontier_cuts,
            budget_fractions=budget_fractions,
        )

    def hard_plan_cases(
        self,
        encoded: EncodedInterfaceCase,
        prepared_state: Mapping[str, Any] | Any,
        trees: Sequence[CaseLocalReceiverTree],
        *,
        budgets: Mapping[str, float | torch.Tensor] | torch.Tensor | None = None,
        frontier_cuts: Sequence[Sequence[int]] | None = None,
        budget_fractions: Mapping[str, float] | None = None,
    ) -> tuple[MechanismPlan, ...]:
        scores = self.score_cases(encoded, prepared_state, trees, budgets=budgets)
        return self.plans_from_scores(
            scores,
            encoded,
            trees,
            hard=True,
            frontier_cuts=frontier_cuts,
            budget_fractions=budget_fractions,
        )


__all__ = ["InputOnlyCoverOrganizer", "OrganizerScores", "masked_binary_logit_loss"]
