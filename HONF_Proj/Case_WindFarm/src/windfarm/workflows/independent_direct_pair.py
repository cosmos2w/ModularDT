"""Small train-only learned direct-pair control for fixed WindFarm checkpoints.

The selector is a separate input-only MLP. Its labels are observed typed G2
actions from frozen training rows; unknown G2 cells remain masked. Selection
uses an exact native pair quota independently for each of MM/ME/EM/QM/QE. The
control never changes query/source coordinates or quadrature measures, and
label imitation alone is not evidence of physical competence.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    INTERACTION_MECHANISMS,
    MechanismPlan,
)
from honf_forward_core.interface_fields.packet_controls import project_direct_pair_budget
from torch import nn
from torch.nn import functional as F

from . import native_cover_organizer as base
from . import native_cover_organizer_fit as typed_fit

MAX_DIRECT_SELECTOR_OPTIMIZER_CALLS = 200
DIRECT_SELECTOR_LABEL_SOURCE = "verified frozen-training G2 typed-plan supervision"


@dataclass(frozen=True)
class NativePairInput:
    """One native typed receiver/source catalogue and its input-only features."""

    mechanism: str
    receiver_features: torch.Tensor
    source_features: torch.Tensor
    global_features: torch.Tensor
    receiver_coordinates: torch.Tensor  # raw native coordinates; retained for direct-plan receiver binding
    source_coordinates: torch.Tensor  # raw native coordinates
    coordinate_scale: torch.Tensor
    receiver_measure: torch.Tensor
    source_measure: torch.Tensor
    eligible_pairs: torch.Tensor
    target_pairs: torch.Tensor | None = None
    observed_pairs: torch.Tensor | None = None
    receiver_uses_query_features: bool = False

    def __post_init__(self) -> None:
        if self.mechanism not in INTERACTION_MECHANISMS:
            raise ValueError(f"unknown typed direct-pair mechanism {self.mechanism!r}")
        receiver_count = int(self.receiver_coordinates.shape[0])
        source_count = int(self.source_coordinates.shape[0])
        if self.receiver_coordinates.shape != (receiver_count, 3):
            raise ValueError("direct-pair receiver coordinates must have shape [R,3]")
        if self.source_coordinates.shape != (source_count, 3):
            raise ValueError("direct-pair source coordinates must have shape [S,3]")
        if self.receiver_features.ndim != 2 or self.receiver_features.shape[0] != receiver_count:
            raise ValueError("receiver input features must have shape [R,F]")
        if self.source_features.ndim != 2 or self.source_features.shape[0] != source_count:
            raise ValueError("source input features must have shape [S,D]")
        if self.global_features.ndim != 1 or self.coordinate_scale.shape != (3,):
            raise ValueError("global input and coordinate scale must have shapes [D] and [3]")
        if self.receiver_measure.shape != (receiver_count,) or self.source_measure.shape != (source_count,):
            raise ValueError("native measure vectors must align with their receiver/source axes")
        expected = (receiver_count, source_count)
        if self.eligible_pairs.shape != expected or self.eligible_pairs.dtype != torch.bool:
            raise ValueError("native eligibility must be a boolean [R,S] matrix")
        if (self.target_pairs is None) != (self.observed_pairs is None):
            raise ValueError("G2 targets and observation masks must be supplied together")
        if self.target_pairs is not None:
            if self.target_pairs.shape != expected or self.observed_pairs.shape != expected:
                raise ValueError("G2 targets and observation masks must match the native [R,S] axes")
            if self.target_pairs.dtype != torch.bool or self.observed_pairs.dtype != torch.bool:
                raise TypeError("G2 direct-pair targets and observation masks must be boolean")
            if bool((self.target_pairs & ~self.observed_pairs).any()):
                raise ValueError("unobserved G2 pairs cannot be marked positive")
            if bool((self.observed_pairs & ~self.eligible_pairs).any()):
                raise ValueError("G2 observations cannot include physically ineligible native pairs")
        tensors = (
            self.receiver_features,
            self.source_features,
            self.global_features,
            self.receiver_coordinates,
            self.source_coordinates,
            self.coordinate_scale,
            self.receiver_measure,
            self.source_measure,
            self.eligible_pairs,
        )
        if any(value.device != self.eligible_pairs.device for value in tensors):
            raise ValueError("all direct-pair input tensors must be on one device")
        if self.target_pairs is not None and (
            self.target_pairs.device != self.eligible_pairs.device
            or self.observed_pairs.device != self.eligible_pairs.device
        ):
            raise ValueError("G2 targets, observation masks, and native eligibility must share one device")
        if not bool(torch.isfinite(self.receiver_features).all()) or not bool(
            torch.isfinite(self.source_features).all()
        ):
            raise ValueError("direct-pair input features must be finite")
        if not bool(torch.isfinite(self.global_features).all()):
            raise ValueError("direct-pair global input features must be finite")
        if not bool(torch.isfinite(self.receiver_coordinates).all()) or not bool(
            torch.isfinite(self.source_coordinates).all()
        ):
            raise ValueError("direct-pair coordinates must be finite")
        if not bool(torch.isfinite(self.coordinate_scale).all()) or bool((self.coordinate_scale <= 0).any()):
            raise ValueError("direct-pair coordinate scale must be finite and positive")
        if not bool(torch.isfinite(self.receiver_measure).all()) or bool((self.receiver_measure < 0).any()):
            raise ValueError("receiver measures must be finite and nonnegative")
        if not bool(torch.isfinite(self.source_measure).all()) or bool((self.source_measure < 0).any()):
            raise ValueError("source measures must be finite and nonnegative")
        if bool((self.eligible_pairs & (self.receiver_measure[:, None] <= 0)).any()) or bool(
            (self.eligible_pairs & (self.source_measure[None, :] <= 0)).any()
        ):
            raise ValueError("eligible pairs must have positive native input measures")

    @property
    def known_pairs(self) -> torch.Tensor:
        if self.observed_pairs is None:
            return torch.zeros_like(self.eligible_pairs)
        return self.eligible_pairs & self.observed_pairs


@dataclass(frozen=True)
class NativeDirectPairCase:
    row_index: int
    layout_index: int
    g2_plan_sha256: str
    pair_inputs: Mapping[str, NativePairInput]

    def __post_init__(self) -> None:
        if set(self.pair_inputs) != set(INTERACTION_MECHANISMS):
            raise ValueError("a direct-pair case must carry all five typed mechanisms")
        if any(self.pair_inputs[name].mechanism != name for name in INTERACTION_MECHANISMS):
            raise ValueError("direct-pair case keys must match each native mechanism")
        object.__setattr__(self, "pair_inputs", MappingProxyType(dict(self.pair_inputs)))


@dataclass(frozen=True)
class DirectPairSplitProvenance:
    frozen_g2_split_sha256: str
    direct_pair_split_sha256: str
    g2_search_report_sha256: str
    g2_label_checkpoint_sha256: str
    student_checkpoint_sha256: str
    training_layout_order: tuple[int, ...]
    fit_layout_indices: tuple[int, ...]
    heldout_training_layout_index: int
    comparison_row_index: int | None
    heldout_layout_selection_rule: str
    development_layout_indices: tuple[int, ...]
    training_rows_by_layout: Mapping[int, tuple[int, ...]]
    development_rows_excluded: tuple[int, ...]

    @classmethod
    def from_frozen_g2_split(
        cls,
        split: Mapping[str, Any],
        *,
        g2_search_report_sha256: str,
        g2_label_checkpoint_sha256: str,
        student_checkpoint_sha256: str,
        comparison_row_index: int | None = None,
    ) -> DirectPairSplitProvenance:
        layouts = tuple(map(int, split["training_layout_indices"]))
        development_layouts = tuple(map(int, split["development_layout_indices"]))
        rows = {
            int(layout["layout_index"]): tuple(map(int, layout["rows_direction_order"]))
            for layout in split["training_layouts"]
        }
        development_rows = tuple(
            int(row) for layout in split["development_layouts"] for row in layout["rows_direction_order"]
        )
        if len(layouts) < 2 or set(rows) != set(layouts):
            raise ValueError("the frozen G2 split must provide at least two complete training layouts")
        if set(layouts) & set(development_layouts) or len(set(development_rows)) != len(development_rows):
            raise ValueError("frozen G2 training and development layout identities must be disjoint")
        row_to_layout = {
            row: layout_index
            for layout_index, layout_rows in rows.items()
            for row in layout_rows
        }
        if comparison_row_index is None:
            heldout = layouts[-1]  # deterministic geometry-frozen order; chosen before reading labels
            selection_rule = "last frozen training layout; selected before reading G2 labels"
        else:
            comparison_row_index = int(comparison_row_index)
            if comparison_row_index not in row_to_layout:
                raise ValueError("comparison row is not part of the frozen G2 training layouts")
            heldout = row_to_layout[comparison_row_index]
            selection_rule = (
                "entire training layout containing the selected comparison row; excludes that row's Q panel "
                "from direct-selector optimizer labels"
            )
        fit_layouts = tuple(layout for layout in layouts if layout != heldout)
        payload = {
            "frozen_g2_split_sha256": str(split["split_sha256"]),
            "training_layout_order": layouts,
            "fit_layout_indices": fit_layouts,
            "heldout_training_layout_index": heldout,
            "comparison_row_index": comparison_row_index,
            "heldout_layout_selection_rule": selection_rule,
            "development_layout_indices": development_layouts,
            "training_rows_by_layout": rows,
            "development_rows_excluded": development_rows,
        }
        split_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            frozen_g2_split_sha256=str(split["split_sha256"]),
            direct_pair_split_sha256=split_hash,
            g2_search_report_sha256=str(g2_search_report_sha256),
            g2_label_checkpoint_sha256=str(g2_label_checkpoint_sha256),
            student_checkpoint_sha256=str(student_checkpoint_sha256),
            training_layout_order=layouts,
            fit_layout_indices=fit_layouts,
            heldout_training_layout_index=heldout,
            comparison_row_index=comparison_row_index,
            heldout_layout_selection_rule=selection_rule,
            development_layout_indices=development_layouts,
            training_rows_by_layout=MappingProxyType(rows),
            development_rows_excluded=development_rows,
        )

    @property
    def fit_row_indices(self) -> tuple[int, ...]:
        return tuple(row for layout in self.fit_layout_indices for row in self.training_rows_by_layout[layout])

    @property
    def heldout_row_indices(self) -> tuple[int, ...]:
        return self.training_rows_by_layout[self.heldout_training_layout_index]

    @property
    def training_row_indices(self) -> tuple[int, ...]:
        return tuple(row for layout in self.training_layout_order for row in self.training_rows_by_layout[layout])

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": DIRECT_SELECTOR_LABEL_SOURCE,
            "frozen_g2_split_sha256": self.frozen_g2_split_sha256,
            "direct_pair_split_sha256": self.direct_pair_split_sha256,
            "g2_search_report_sha256": self.g2_search_report_sha256,
            "g2_label_checkpoint_sha256": self.g2_label_checkpoint_sha256,
            "student_checkpoint_sha256": self.student_checkpoint_sha256,
            "training_layout_order": list(self.training_layout_order),
            "fit_layout_indices": list(self.fit_layout_indices),
            "heldout_training_layout_index": self.heldout_training_layout_index,
            "comparison_row_index": self.comparison_row_index,
            "heldout_layout_selection_rule": self.heldout_layout_selection_rule,
            "development_layout_indices_excluded": list(self.development_layout_indices),
            "fit_row_indices": list(self.fit_row_indices),
            "heldout_row_indices": list(self.heldout_row_indices),
            "development_rows_excluded": list(self.development_rows_excluded),
            "development_g2_labels_loaded": False,
            "heldout_layout_used_for_optimizer_updates": False,
            "heldout_labels_are_for_generalization_diagnostics_only": True,
            "label_scope": (
                "binary typed permissions propagated through the saved hard plan, masked by the exact G2 "
                "mechanism_observed rows and native eligibility"
            ),
            "physical_reference_evaluation_required": True,
            "label_imitation_is_not_physical_competence": True,
        }


def load_verified_training_g2_documents(
    typed_search_dir: str | Path,
    layouts: tuple[Any, ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    """Load verified G2 labels for frozen training rows only; never open dev plans."""

    directory = Path(typed_search_dir).expanduser().resolve()
    report_path = directory / "typed_search_report.json"
    if not report_path.is_file():
        raise FileNotFoundError("direct-pair fit requires a completed typed G2 search report")
    report_stub = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint_hash = str(report_stub.get("checkpoint_sha256", ""))
    if not checkpoint_hash:
        raise ValueError("typed G2 report has no label-checkpoint identity")
    report, manifest, documents = typed_fit._typed_fit_artifacts(
        directory,
        checkpoint_sha256=checkpoint_hash,
        layouts=layouts,
    )
    expected_rows = set(map(int, report.get("training_rows", ())))
    if set(documents) != expected_rows:
        raise ValueError("verified G2 document loader did not return exactly the frozen training labels")
    if report.get("development_rows_used_for_search_or_fit") != []:
        raise ValueError("development G2 labels cannot be loaded for the direct-pair control")
    return report, manifest, documents


def aggregate_observed_g2_actions(
    reached_nodes: torch.Tensor,
    target_actions: torch.Tensor,
    observed_actions: torch.Tensor,
    eligible_pairs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve OR-style typed actions without turning unknowns into negatives.

    An observed positive at any reached node determines a positive pair. A
    negative pair is known only when at least one node is reached and every
    reached node's action for that source is observed and negative.
    """

    if reached_nodes.ndim != 2 or reached_nodes.dtype != torch.bool:
        raise ValueError("reached_nodes must be a boolean [Q,P] matrix")
    if target_actions.ndim != 2 or observed_actions.shape != target_actions.shape:
        raise ValueError("typed target and observation tables must share [P,S] axes")
    if observed_actions.dtype != torch.bool:
        raise TypeError("typed observation actions must be boolean")
    if target_actions.shape[0] != reached_nodes.shape[1]:
        raise ValueError("reached node count differs from typed action rows")
    if eligible_pairs.shape != (reached_nodes.shape[0], target_actions.shape[1]):
        raise ValueError("native eligibility must have [Q,S] axes matching reached receivers and sources")
    if eligible_pairs.dtype != torch.bool:
        raise TypeError("native pair eligibility must be boolean")
    if any(
        value.device != reached_nodes.device
        for value in (target_actions, observed_actions, eligible_pairs)
    ):
        raise ValueError("typed actions, reachability, and native eligibility must share one device")
    if bool(((target_actions != 0) & (target_actions != 1)).any()):
        raise ValueError("G2 typed pair actions must be hard binary permissions")

    reached = reached_nodes[:, :, None]
    observed_reached = reached & observed_actions[None, :, :]
    positive_known = (observed_reached & (target_actions[None, :, :] > 0.5)).any(dim=1)
    unknown_reached = (reached & ~observed_actions[None, :, :]).any(dim=1)
    any_reached = reached_nodes.any(dim=1, keepdim=True)
    negative_known = any_reached & ~unknown_reached & ~positive_known
    observed_pairs = (positive_known | negative_known) & eligible_pairs
    target_pairs = positive_known & observed_pairs
    return target_pairs, observed_pairs


class InputOnlyDirectPairSelector(nn.Module):
    """Independent route-specific MLP using only encoded inputs and geometry."""

    def __init__(self, *, state_dim: int, query_feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if min(state_dim, query_feature_dim, hidden_dim) < 1:
            raise ValueError("direct selector feature dimensions must be positive")
        self.state_dim = int(state_dim)
        self.query_feature_dim = int(query_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.query_encoder = nn.Sequential(
            nn.Linear(self.query_feature_dim, self.state_dim), nn.GELU(), nn.Linear(self.state_dim, self.state_dim)
        )
        pair_dim = 3 * self.state_dim + 12
        self.mechanism_heads = nn.ModuleDict({
            mechanism: nn.Sequential(
                nn.Linear(pair_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1),
            )
            for mechanism in INTERACTION_MECHANISMS
        })

    def pair_logits(
        self,
        pair_input: NativePairInput,
        receiver_indices: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> torch.Tensor:
        if receiver_indices.ndim != 1 or source_indices.shape != receiver_indices.shape:
            raise ValueError("pair indices must be aligned vectors")
        if receiver_indices.device != pair_input.eligible_pairs.device or source_indices.device != receiver_indices.device:
            raise ValueError("pair indices and their native catalogue must share one device")
        receiver_values = pair_input.receiver_features.index_select(0, receiver_indices)
        if pair_input.receiver_uses_query_features:
            if receiver_values.shape[1] != self.query_feature_dim:
                raise ValueError("query feature width differs from the fitted direct selector")
            receiver_state = self.query_encoder(receiver_values)
        else:
            if receiver_values.shape[1] != self.state_dim:
                raise ValueError("encoded receiver state width differs from the fitted direct selector")
            receiver_state = receiver_values
        source_state = pair_input.source_features.index_select(0, source_indices)
        if source_state.shape[1] != self.state_dim:
            raise ValueError("encoded source state width differs from the fitted direct selector")
        global_state = pair_input.global_features.expand(receiver_indices.shape[0], -1)
        receiver_xyz = pair_input.receiver_coordinates.index_select(0, receiver_indices)
        source_xyz = pair_input.source_coordinates.index_select(0, source_indices)
        receiver_xyz = receiver_xyz / pair_input.coordinate_scale
        source_xyz = source_xyz / pair_input.coordinate_scale
        delta = receiver_xyz - source_xyz
        positive_receiver_measures = pair_input.receiver_measure[pair_input.receiver_measure > 0]
        positive_source_measures = pair_input.source_measure[pair_input.source_measure > 0]
        receiver_measure = pair_input.receiver_measure.index_select(0, receiver_indices)
        source_measure = pair_input.source_measure.index_select(0, source_indices)
        receiver_log_measure = torch.log(
            receiver_measure / positive_receiver_measures.mean().clamp_min(torch.finfo(receiver_measure.dtype).tiny)
        )
        source_log_measure = torch.log(
            source_measure / positive_source_measures.mean().clamp_min(torch.finfo(source_measure.dtype).tiny)
        )
        geometry = torch.cat(
            (
                receiver_xyz,
                source_xyz,
                delta,
                delta.square().sum(dim=-1, keepdim=True),
                receiver_log_measure[:, None],
                source_log_measure[:, None],
            ),
            dim=-1,
        )
        features = torch.cat((receiver_state, source_state, global_state, geometry), dim=-1)
        return self.mechanism_heads[pair_input.mechanism](features).squeeze(-1)

    @torch.no_grad()
    def score_matrix(self, pair_input: NativePairInput, *, chunk_size: int = 65_536) -> torch.Tensor:
        """Score every exact eligible native pair in bounded feature chunks."""

        if chunk_size < 1:
            raise ValueError("pair scoring chunk size must be positive")
        scores = pair_input.receiver_coordinates.new_zeros(pair_input.eligible_pairs.shape)
        flat_candidates = torch.nonzero(pair_input.eligible_pairs.reshape(-1), as_tuple=False).flatten()
        source_count = int(pair_input.eligible_pairs.shape[1])
        for start in range(0, int(flat_candidates.numel()), chunk_size):
            flat = flat_candidates[start : start + chunk_size]
            receiver_ids = torch.div(flat, source_count, rounding_mode="floor")
            source_ids = torch.remainder(flat, source_count)
            scores.reshape(-1)[flat] = self.pair_logits(pair_input, receiver_ids, source_ids)
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("direct selector produced a nonfinite eligible pair score")
        return scores


@dataclass(frozen=True)
class DirectPairFitResult:
    optimizer_calls: int
    training_loss_last: float
    heldout_label_metrics: Mapping[str, Any]
    provenance: Mapping[str, Any]


def build_native_pair_inputs(
    record: Any,
    query_receivers: torch.Tensor,
    query_features: torch.Tensor,
    *,
    g2_plan: MechanismPlan | None = None,
    g2_supervision: Mapping[str, Any] | None = None,
) -> Mapping[str, NativePairInput]:
    """Build native axes and optional masked G2 support labels from inputs only."""

    if (g2_plan is None) != (g2_supervision is None):
        raise ValueError("a G2 plan and its observed supervision must be supplied together")
    encoded = record.encoded
    if encoded.module_present.shape[0] != 1 or encoded.env_coords.shape[0] != 1:
        raise ValueError("direct-pair case construction requires one native encoded case")
    if encoded.env_weights is None:
        raise ValueError("WindFarm direct-pair controls require the original environment quadrature weights")
    module_coords = encoded.module_centers[0].detach()
    environment_coords = encoded.env_coords[0].detach()
    modules = encoded.module_tokens[0].detach()
    environments = encoded.env_tokens[0].detach()
    global_state = encoded.global_token[0].detach()
    module_valid = encoded.module_present[0].detach() > 0.5
    environment_measure = encoded.env_weights[0].detach()
    environment_valid = environment_measure > 0.0
    if query_receivers.ndim != 2 or query_receivers.shape[-1] != 3:
        raise ValueError("WindFarm query receivers must have shape [Q,3]")
    if query_features.ndim != 2 or query_features.shape[0] != query_receivers.shape[0]:
        raise ValueError("input-only query features must align with the query receiver axis")
    if query_receivers.device != encoded.module_present.device or query_features.device != query_receivers.device:
        raise ValueError("query inputs and encoded case must use one device")
    scale = encoded.coordinate_scale
    if scale.ndim == 3:
        scale = scale[0, 0]
    elif scale.ndim == 2:
        scale = scale[0]
    if scale.shape != (3,):
        raise ValueError("encoded WindFarm coordinates must expose a three-axis scale")
    receiver_coords = {
        "MM": module_coords,
        "ME": module_coords,
        "EM": environment_coords,
        "QM": query_receivers.detach(),
        "QE": query_receivers.detach(),
    }
    source_coords = {
        "MM": module_coords,
        "ME": environment_coords,
        "EM": module_coords,
        "QM": module_coords,
        "QE": environment_coords,
    }
    receiver_features = {
        "MM": modules,
        "ME": modules,
        "EM": environments,
        "QM": query_features.detach(),
        "QE": query_features.detach(),
    }
    source_features = {
        "MM": modules,
        "ME": environments,
        "EM": modules,
        "QM": modules,
        "QE": environments,
    }
    receiver_measure = {
        "MM": torch.ones(len(module_coords), device=scale.device, dtype=scale.dtype),
        "ME": torch.ones(len(module_coords), device=scale.device, dtype=scale.dtype),
        "EM": torch.ones(len(environment_coords), device=scale.device, dtype=scale.dtype),
        "QM": torch.ones(len(query_receivers), device=scale.device, dtype=scale.dtype),
        "QE": torch.ones(len(query_receivers), device=scale.device, dtype=scale.dtype),
    }
    source_measure = {
        "MM": torch.ones(len(module_coords), device=scale.device, dtype=scale.dtype),
        "ME": environment_measure,
        "EM": torch.ones(len(module_coords), device=scale.device, dtype=scale.dtype),
        "QM": torch.ones(len(module_coords), device=scale.device, dtype=scale.dtype),
        "QE": environment_measure,
    }
    query_valid = torch.ones(len(query_receivers), device=scale.device, dtype=torch.bool)
    eligibility = {
        "MM": module_valid[:, None] & module_valid[None, :]
        & ~torch.eye(len(module_coords), device=scale.device, dtype=torch.bool),
        "ME": module_valid[:, None] & environment_valid[None, :],
        "EM": torch.ones(len(environment_coords), device=scale.device, dtype=torch.bool)[:, None]
        & module_valid[None, :],
        "QM": query_valid[:, None] & module_valid[None, :],
        "QE": query_valid[:, None] & environment_valid[None, :],
    }
    pair_inputs: dict[str, NativePairInput] = {}
    for mechanism in INTERACTION_MECHANISMS:
        targets = observed = None
        if g2_plan is not None and g2_supervision is not None:
            target_table = torch.as_tensor(
                g2_supervision["mechanism_targets"][mechanism],
                device=scale.device,
                dtype=scale.dtype,
            )
            observed_table = torch.as_tensor(
                g2_supervision["mechanism_observed"][mechanism],
                device=scale.device,
                dtype=torch.bool,
            )
            expected_table = g2_plan.permission_matrix(
                mechanism,
                source_count=int(target_table.shape[1]),
                module_present=encoded.module_present[0]
                if mechanism in {"MM", "EM", "QM"}
                else None,
            )
            if target_table.shape != expected_table.shape or not torch.equal(
                target_table, expected_table.detach().to(target_table.dtype)
            ):
                raise ValueError(f"{mechanism} G2 targets do not match the saved typed plan action table")
            if bool(((target_table != 0) & (target_table != 1)).any()):
                raise ValueError(f"{mechanism} G2 actions must be hard binary permissions")
            if bool(((g2_plan.split_gates != 0) & (g2_plan.split_gates != 1)).any()):
                raise ValueError("G2 plan reachability requires hard binary split gates")
            receiver_access = g2_plan.tree.access(receiver_coords[mechanism], g2_plan.split_gates.detach())
            reached = receiver_access > 0.0
            targets, observed = aggregate_observed_g2_actions(
                reached,
                target_table,
                observed_table,
                eligibility[mechanism],
            )
            if bool((observed & ~eligibility[mechanism]).any()):
                raise RuntimeError("G2 observation propagation escaped exact native pair eligibility")
        pair_inputs[mechanism] = NativePairInput(
            mechanism=mechanism,
            receiver_features=receiver_features[mechanism],
            source_features=source_features[mechanism],
            global_features=global_state,
            receiver_coordinates=receiver_coords[mechanism],
            source_coordinates=source_coords[mechanism],
            coordinate_scale=scale.detach(),
            receiver_measure=receiver_measure[mechanism],
            source_measure=source_measure[mechanism],
            eligible_pairs=eligibility[mechanism],
            target_pairs=targets,
            observed_pairs=observed,
            receiver_uses_query_features=mechanism in {"QM", "QE"},
        )
    return MappingProxyType(pair_inputs)


def build_verified_training_direct_pair_case(
    record: Any,
    document: Mapping[str, Any],
    provenance: DirectPairSplitProvenance,
    query_receivers: torch.Tensor,
    query_features: torch.Tensor,
) -> NativeDirectPairCase:
    """Bind one already-verified frozen train G2 label document to input-only features."""

    row_index = int(document.get("row_index", -1))
    if row_index not in provenance.training_row_indices:
        raise ValueError("direct-pair label document is outside the frozen G2 training rows")
    if row_index in provenance.development_rows_excluded:
        raise ValueError("development G2 labels are excluded from direct-pair fitting")
    layout_index = next(
        layout for layout, rows in provenance.training_rows_by_layout.items() if row_index in rows
    )
    if int(getattr(getattr(record, "case", None), "layout_index", -1)) != layout_index:
        raise ValueError("direct-pair input case layout differs from the frozen G2 row provenance")
    plan_hash = str(document.get("plan_hash", ""))
    if not plan_hash:
        raise ValueError("verified G2 training document is missing its canonical plan hash")
    device = query_receivers.device
    if query_features.device != device or record.encoded.module_present.device != device:
        raise ValueError("direct-pair inputs, encoded case, and G2 labels must use one device")
    g2_plan = typed_fit._typed_plan_from_document(record, dict(document), device)
    g2_supervision = typed_fit._typed_label_tensors(dict(document), device)
    pair_inputs = build_native_pair_inputs(
        record,
        query_receivers,
        query_features,
        g2_plan=g2_plan,
        g2_supervision=g2_supervision,
    )
    return NativeDirectPairCase(
        row_index=row_index,
        layout_index=layout_index,
        g2_plan_sha256=plan_hash,
        pair_inputs=pair_inputs,
    )


def _balanced_binary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets = targets.to(dtype=logits.dtype)
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = targets >= 0.5
    negative = ~positive
    classes = []
    if bool(positive.any()):
        classes.append(losses[positive].mean())
    if bool(negative.any()):
        classes.append(losses[negative].mean())
    if not classes:
        raise ValueError("direct-pair optimizer batch contains no observed labels")
    return torch.stack(classes).mean()


def _sample_label_indices(
    pair_input: NativePairInput,
    count: int,
    generator: torch.Generator,
    *,
    positive_pool: torch.Tensor | None = None,
    negative_pool: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pair_input.target_pairs is None or pair_input.observed_pairs is None:
        raise ValueError("direct-pair fitting requires observed G2 targets")
    known = pair_input.known_pairs
    positive = positive_pool
    negative = negative_pool
    if positive is None and negative is None:
        positive = torch.nonzero((known & pair_input.target_pairs).reshape(-1), as_tuple=False).flatten()
        negative = torch.nonzero((known & ~pair_input.target_pairs).reshape(-1), as_tuple=False).flatten()
    elif positive is None or negative is None:
        raise ValueError("positive and negative pair-sampling pools must be supplied together")
    if positive.device != known.device or negative.device != known.device:
        raise ValueError("cached direct-label sampling pools must share the native case device")
    if positive.numel() + negative.numel() == 0:
        raise ValueError(f"{pair_input.mechanism} has no observed native G2 pairs")
    source_count = int(known.shape[1])
    if positive.numel() and negative.numel():
        n_positive = count // 2
        n_negative = count - n_positive
    elif positive.numel():
        n_positive, n_negative = count, 0
    else:
        n_positive, n_negative = 0, count
    selected_parts = []
    if positive.numel():
        picks = torch.randint(positive.numel(), (n_positive,), device=positive.device, generator=generator)
        selected_parts.append(positive.index_select(0, picks))
    if negative.numel():
        picks = torch.randint(negative.numel(), (n_negative,), device=negative.device, generator=generator)
        selected_parts.append(negative.index_select(0, picks))
    selected = torch.cat(selected_parts)
    receiver_ids = torch.div(selected, source_count, rounding_mode="floor")
    source_ids = torch.remainder(selected, source_count)
    targets = pair_input.target_pairs[receiver_ids, source_ids]
    return receiver_ids, source_ids, targets


def _label_sampling_pool(
    pair_input: NativePairInput,
    *,
    max_pairs_per_class: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache a bounded with-replacement sampling bank, never unknown labels."""

    if pair_input.target_pairs is None or pair_input.observed_pairs is None:
        raise ValueError("direct-pair fitting requires observed G2 targets")
    if max_pairs_per_class < 1:
        raise ValueError("label sampling bank capacity must be positive")
    known = pair_input.known_pairs
    positive = torch.nonzero((known & pair_input.target_pairs).reshape(-1), as_tuple=False).flatten()
    negative = torch.nonzero((known & ~pair_input.target_pairs).reshape(-1), as_tuple=False).flatten()
    if positive.numel() + negative.numel() == 0:
        raise ValueError(f"{pair_input.mechanism} has no observed native G2 pairs")

    def bounded(values: torch.Tensor) -> torch.Tensor:
        if values.numel() <= max_pairs_per_class:
            return values
        draws = torch.randint(
            int(values.numel()),
            (max_pairs_per_class,),
            device=values.device,
            generator=generator,
        )
        return values.index_select(0, draws)

    return bounded(positive), bounded(negative)


def project_per_mechanism_pair_budgets(
    scores_by_mechanism: Mapping[str, torch.Tensor],
    eligible_by_mechanism: Mapping[str, torch.Tensor],
    requested_by_mechanism: Mapping[str, int],
) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    """Project each route independently and report exact plus closest work."""

    expected = set(INTERACTION_MECHANISMS)
    if set(scores_by_mechanism) != expected or set(eligible_by_mechanism) != expected:
        raise ValueError("direct-pair scores and eligibility must cover all five mechanisms")
    if set(requested_by_mechanism) != expected:
        raise ValueError("direct-pair work quotas must cover all five mechanisms")
    selected: dict[str, torch.Tensor] = {}
    requested: dict[str, int] = {}
    achieved: dict[str, int] = {}
    capacity: dict[str, int] = {}
    exact: dict[str, bool] = {}
    closest: dict[str, int] = {}
    for mechanism in INTERACTION_MECHANISMS:
        budget = requested_by_mechanism[mechanism]
        projection = project_direct_pair_budget(
            scores_by_mechanism[mechanism],
            budget,
            eligible_pairs=eligible_by_mechanism[mechanism],
        )
        selected[mechanism] = projection.selected_pairs
        requested[mechanism] = projection.requested_work
        achieved[mechanism] = projection.achieved_work
        capacity[mechanism] = projection.feasible_capacity
        exact[mechanism] = projection.exact_match
        closest[mechanism] = projection.closest_feasible_work
    requested_total = sum(requested.values())
    achieved_total = sum(achieved.values())
    return MappingProxyType(selected), {
        "per_mechanism_requested_exact_native_pairs": requested,
        "per_mechanism_selected_pairs": achieved,
        "per_mechanism_feasible_capacity": capacity,
        "per_mechanism_exact_match": exact,
        "closest_feasible_pair_count_by_mechanism": closest,
        "total_requested_exact_native_pairs": requested_total,
        "total_selected_pairs": achieved_total,
        "total_exact_match": achieved_total == requested_total and all(exact.values()),
    }


def _label_metrics(
    selector: InputOnlyDirectPairSelector,
    cases: Sequence[NativeDirectPairCase],
    *,
    chunk_size: int = 65_536,
) -> dict[str, Any]:
    counts = {mechanism: {"observed": 0, "positive": 0, "correct": 0, "predicted_positive": 0} for mechanism in INTERACTION_MECHANISMS}
    for case in cases:
        for mechanism in INTERACTION_MECHANISMS:
            pair_input = case.pair_inputs[mechanism]
            if pair_input.target_pairs is None or pair_input.observed_pairs is None:
                raise ValueError("held-out label diagnostics require G2 targets and observation masks")
            known = pair_input.known_pairs
            scores = selector.score_matrix(pair_input, chunk_size=chunk_size)
            predicted = scores >= 0.0
            truth = pair_input.target_pairs
            item = counts[mechanism]
            item["observed"] += int(known.sum().detach().cpu())
            item["positive"] += int((known & truth).sum().detach().cpu())
            item["correct"] += int((known & (predicted == truth)).sum().detach().cpu())
            item["predicted_positive"] += int((known & predicted).sum().detach().cpu())
    result = {}
    for mechanism, item in counts.items():
        observed = item["observed"]
        result[mechanism] = {
            **item,
            "masked_accuracy": item["correct"] / observed if observed else None,
            "masked_positive_fraction": item["positive"] / observed if observed else None,
            "masked_predicted_positive_fraction": item["predicted_positive"] / observed if observed else None,
        }
    return {
        "by_mechanism": result,
        "semantics": "threshold-zero label-imitation diagnostic over observed, eligible held-out G2 pairs only",
        "interpretation_limit": "label generalization is not physical competence; evaluate the control against native references",
    }


def fit_input_only_direct_pair_selector(
    selector: InputOnlyDirectPairSelector,
    cases: Sequence[NativeDirectPairCase],
    provenance: DirectPairSplitProvenance,
    *,
    optimizer_calls: int = MAX_DIRECT_SELECTOR_OPTIMIZER_CALLS,
    batch_pairs_per_mechanism: int = 128,
    seed: int = 73021,
    learning_rate: float = 1.0e-3,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> DirectPairFitResult:
    """Fit on observed G2 labels from seven train layouts; hold one out whole."""

    if isinstance(optimizer_calls, bool) or not isinstance(optimizer_calls, int):
        raise TypeError("direct-selector optimizer calls must be an integer")
    if not 1 <= optimizer_calls <= MAX_DIRECT_SELECTOR_OPTIMIZER_CALLS:
        raise ValueError(f"direct-selector fit is capped at {MAX_DIRECT_SELECTOR_OPTIMIZER_CALLS} optimizer calls")
    if batch_pairs_per_mechanism < 2 or learning_rate <= 0.0:
        raise ValueError("direct-selector batch size and learning rate must be positive")
    by_row = {int(case.row_index): case for case in cases}
    if len(by_row) != len(cases) or set(by_row) != set(provenance.training_row_indices):
        raise ValueError("direct-pair cases must be exactly the frozen G2 training rows")
    for case in cases:
        expected_layout = next(
            layout for layout, rows in provenance.training_rows_by_layout.items() if case.row_index in rows
        )
        if int(case.layout_index) != expected_layout or not case.g2_plan_sha256:
            raise ValueError("direct-pair case identity differs from its frozen training-layout G2 label")
        if any(case.pair_inputs[m].target_pairs is None for m in INTERACTION_MECHANISMS):
            raise ValueError("every train-only case must carry its observed G2 labels")
    fit_cases = [by_row[row] for row in provenance.fit_row_indices]
    heldout_cases = [by_row[row] for row in provenance.heldout_row_indices]
    if not fit_cases or not heldout_cases:
        raise ValueError("direct-pair split requires nonempty fit rows and one held-out training layout")
    if {case.layout_index for case in fit_cases} != set(provenance.fit_layout_indices):
        raise ValueError("optimizer rows do not cover the declared fit layouts")
    if any(case.layout_index != provenance.heldout_training_layout_index for case in heldout_cases):
        raise ValueError("held-out diagnostics must use exactly one full training layout")

    device = next(selector.parameters()).device
    if any(
        pair_input.eligible_pairs.device != device
        for case in cases
        for pair_input in case.pair_inputs.values()
    ):
        raise ValueError("selector and direct-pair examples must use one device")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    py_rng = random.Random(int(seed))
    candidates_by_mechanism: dict[str, list[tuple[NativePairInput, torch.Tensor, torch.Tensor]]] = {}
    for mechanism in INTERACTION_MECHANISMS:
        candidates = []
        for case in fit_cases:
            item = case.pair_inputs[mechanism]
            if item.target_pairs is None or not bool(item.known_pairs.any()):
                continue
            positive_pool, negative_pool = _label_sampling_pool(
                item,
                max_pairs_per_class=max(2048, batch_pairs_per_mechanism * 8),
                generator=generator,
            )
            candidates.append((item, positive_pool, negative_pool))
        if not candidates:
            raise ValueError(f"fit split has no observed G2 labels for {mechanism}")
        candidates_by_mechanism[mechanism] = candidates
    optimizer = torch.optim.AdamW(selector.parameters(), lr=learning_rate)
    selector.train()
    last_loss = float("nan")
    for optimizer_call in range(1, optimizer_calls + 1):
        if progress_callback is not None:
            progress_callback({"event": "optimizer_attempted", "optimizer_call": optimizer_call})
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for mechanism in INTERACTION_MECHANISMS:
            pair_input, positive_pool, negative_pool = py_rng.choice(candidates_by_mechanism[mechanism])
            receiver_ids, source_ids, targets = _sample_label_indices(
                pair_input,
                batch_pairs_per_mechanism,
                generator,
                positive_pool=positive_pool,
                negative_pool=negative_pool,
            )
            logits = selector.pair_logits(pair_input, receiver_ids, source_ids)
            losses.append(_balanced_binary_loss(logits, targets))
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("independent direct-pair training loss is nonfinite")
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if progress_callback is not None:
            progress_callback({
                "event": "optimizer_completed",
                "optimizer_call": optimizer_call,
                "training_loss": last_loss,
            })
    selector.eval()
    heldout_metrics = _label_metrics(selector, heldout_cases)
    return DirectPairFitResult(
        optimizer_calls=optimizer_calls,
        training_loss_last=last_loss,
        heldout_label_metrics=MappingProxyType(heldout_metrics),
        provenance=MappingProxyType(provenance.as_dict()),
    )


@dataclass(frozen=True)
class MatchedDirectPairControl:
    plan: MechanismPlan | None
    metadata: Mapping[str, Any]


def build_matched_direct_pair_control(
    record: Any,
    selector: InputOnlyDirectPairSelector,
    learned_plan: MechanismPlan,
    query_receivers: torch.Tensor,
    query_features: torch.Tensor,
    *,
    score_chunk_size: int = 65_536,
) -> MatchedDirectPairControl:
    """Project independent scores to learned-plan work, matching each route."""

    pair_inputs = build_native_pair_inputs(record, query_receivers, query_features)
    repeated_receivers = {
        mechanism: int(pair_input.receiver_coordinates.shape[0])
        - int(torch.unique(pair_input.receiver_coordinates, dim=0).shape[0])
        for mechanism, pair_input in pair_inputs.items()
    }
    if any(repeated_receivers.values()):
        raise ValueError(
            "direct native access requires unique exact receiver coordinates; choose one deterministic "
            "query panel without replacement and use it for every compared condition"
        )
    learned_work = base._typed_work_summary(
        learned_plan, record.encoded, query_receivers, include_root_child_support=False
    )
    requested = {
        mechanism: int(
            learned_work["mechanisms"][mechanism][
                f"cover_{mechanism.lower()}_unique_source_receiver_pairs"
            ]
        )
        for mechanism in INTERACTION_MECHANISMS
    }
    scores_by_mechanism = {}
    eligible_by_mechanism = {}
    for mechanism in INTERACTION_MECHANISMS:
        pair_input = pair_inputs[mechanism]
        scores_by_mechanism[mechanism] = selector.score_matrix(pair_input, chunk_size=score_chunk_size)
        eligible_by_mechanism[mechanism] = pair_input.eligible_pairs
    matrices, projection_report = project_per_mechanism_pair_budgets(
        scores_by_mechanism,
        eligible_by_mechanism,
        requested,
    )
    achieved = dict(projection_report["per_mechanism_selected_pairs"])
    requested_total = int(projection_report["total_requested_exact_native_pairs"])
    metadata: dict[str, Any] = {
        "control": "separately trained input-only direct-pair selector",
        "selection_inputs": "current checkpoint module/environment input tokens, query support features, geometry, and source measures",
        "g2_targets_used_for_selection": False,
        "learned_plan_used_for_selection": "per-mechanism exact native pair quotas only",
        "source_coordinates_and_native_measures_mutated": False,
        "receiver_index_binding": "exact unique native coordinate rows; no query/source rows are deduplicated",
        "duplicate_receiver_rows_by_mechanism": repeated_receivers,
        **projection_report,
        "physical_reference_evaluation_required": True,
        "label_imitation_is_not_physical_competence": True,
        "executor_requirement": "dense_masked reference; rectangular subset dispatch does not accept direct receiver/source masks",
        "measured_speed_or_deployment_claim": False,
    }
    if not metadata["total_exact_match"]:
        return MatchedDirectPairControl(plan=None, metadata=MappingProxyType(metadata))

    supports = {
        mechanism: matrices[mechanism].any(dim=0).to(dtype=record.encoded.module_present.dtype)
        for mechanism in INTERACTION_MECHANISMS
    }
    root = typed_fit._typed_root_permissions_plan(record, supports)
    direct_plan = typed_fit._TypedDirectPairPlan(
        root.tree,
        root.split_gates,
        root.module_present,
        root.environment_count,
        root.permissions,
        matrices,
        {mechanism: pair_inputs[mechanism].receiver_coordinates for mechanism in INTERACTION_MECHANISMS},
    )
    exact_work = base._typed_work_summary(
        direct_plan, record.encoded, query_receivers, include_root_child_support=False
    )
    actual = {
        mechanism: int(
            exact_work["mechanisms"][mechanism][
                f"cover_{mechanism.lower()}_unique_source_receiver_pairs"
            ]
        )
        for mechanism in INTERACTION_MECHANISMS
    }
    actual_total = int(exact_work["total_unique_source_receiver_pairs_across_mechanisms"])
    if actual != achieved or actual_total != requested_total:
        raise RuntimeError(
            "direct-pair masks did not preserve their per-mechanism quotas through the exact native compilers"
        )
    metadata["native_compiler_pairs_by_mechanism"] = actual
    metadata["native_compiler_total_pairs"] = actual_total
    metadata["plan_hash"] = direct_plan.canonical_hash()
    metadata["native_work_summary"] = exact_work
    return MatchedDirectPairControl(plan=direct_plan, metadata=MappingProxyType(metadata))


__all__ = [
    "DIRECT_SELECTOR_LABEL_SOURCE",
    "MAX_DIRECT_SELECTOR_OPTIMIZER_CALLS",
    "DirectPairFitResult",
    "DirectPairSplitProvenance",
    "InputOnlyDirectPairSelector",
    "MatchedDirectPairControl",
    "NativeDirectPairCase",
    "NativePairInput",
    "aggregate_observed_g2_actions",
    "build_matched_direct_pair_control",
    "build_native_pair_inputs",
    "build_verified_training_direct_pair_case",
    "fit_input_only_direct_pair_selector",
    "load_verified_training_g2_documents",
    "project_per_mechanism_pair_budgets",
]
