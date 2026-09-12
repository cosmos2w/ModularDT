#!/usr/bin/env python3
"""Bounded, evaluation-only diagnostics for the HONF NStage2 candidates.

The NStage2 study has two candidate families.  NStage2-A is the hierarchical
regional reader and NStage2-B is the group-mediated sparse reader.  This file
is deliberately a thin entry point around the maintained Stage-3/Stage-1
helpers.  It does not train, allocate a managed run, copy a checkpoint, or
make a claim about a missing CFD reference.

Run from ``HONF_Proj/``.  Checkpoints use explicit ``LABEL=PATH`` syntax::

    python tools/diagnostics/run_honf_nstage2.py interventions \
        --checkpoint nstage2-a=Trained_Results/.../epoch_0500_model.pt \
        --output diagnostics/generated/interface_operator_study/nstage2/a.json

The default intervention cases are the four established interface anchors
plus case 0283, the difficult case highlighted by the mature Dense/Regional
comparison.  They are fixed before a result is inspected.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    PROJECT_ROOT / "tools",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "Case_ThermalChannel" / "src",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Stage-3 owns trusted checkpoint loading, dataset/sample adaptation, the
# canonical physical error reducer, and output serialization.  Regional
# Stage-1 owns the small evaluation wrapper used by all established studies.
from run_regional_response_study import (  # type: ignore[import-not-found]
    _run_variant,
    backend_read_zero,
)
from run_stage3_interface_study import (  # type: ignore[import-not-found]
    _active_centers,
    _align_one_port_to_support_transition,
    _autograd_directional,
    _canonical_ground_truth_errors,
    _checkpoint_record,
    _direction_for_sample,
    _domain_lengths,
    _error_deltas,
    _field_kpis,
    _finite_difference_directional,
    _finite_stats,
    _forward_tensor_batch,
    _load_dataset,
    _load_model_spec,
    _load_raw_sample,
    _query_points,
    _regional_jvp_summary,
    _relative_difference,
    _state_keys,
    _support_key_set_for_centers,
    _valid_geometry,
    parse_checkpoint_specs,
    select_device,
    select_sample,
    write_json,
)

STUDY_SCHEMA_VERSION = 1
DEFAULT_ANCHOR_CASE_IDS = ("0273", "0653", "0283", "0298", "0302")
DEFAULT_PROFILE = "project://src/config_core/forward/nstage2_hierarchical_regional_context.json"
HIERARCHICAL_ARCHITECTURE = "hierarchical_regional_honf"
GROUP_MEDIATED_ARCHITECTURE = "sparse_interface_honf"
DEFAULT_NSTAGE2_QUERY_COUNT = 8192
DEFAULT_PROBE_CASE_IDS = ("0273", "0298")
DEFAULT_PROBE_QUERY_COUNT = 32

ANCHOR_SELECTION_REASONS = {
    "0273": "established interface anchor; selected before NStage2 results",
    "0653": "established interface anchor; selected before NStage2 results",
    "0298": "established difficult anchor; selected before NStage2 results",
    "0302": "established interface anchor; selected before NStage2 results",
    "0283": "mature Dense/Regional difficult case; fixed as the fifth anchor before NStage2 results",
}


def _architecture(model: Any) -> str:
    """Read the architecture field from the maintained model config."""

    return str(model.config.core_honf.forward_architecture)


def _role(model: Any) -> str | None:
    value = getattr(model.core, "_interface_read_role", None)
    return None if value is None else str(value)


def _phase_roles(mode: str) -> frozenset[str]:
    mode = str(mode)
    aliases = {
        "p0_only": "p0",
        "p1": "p1_only",
        "p2_only": "p2",
    }
    mode = aliases.get(mode, mode)
    values = {
        "p0": ("p0_port",),
        "p1_only": ("p1_refinement",),
        "p2": ("p2_field",),
    }
    try:
        return frozenset(values[mode])
    except KeyError as exc:
        raise ValueError(f"unknown NStage2 phase={mode!r}") from exc


def _zero_result(value: Any) -> Any:
    """Zero a context-like result while retaining auxiliary diagnostics."""

    if torch.is_tensor(value):
        return torch.zeros_like(value)
    if isinstance(value, tuple):
        if not value:
            return value
        return (_zero_result(value[0]), *value[1:])
    if isinstance(value, list):
        if not value:
            return value
        return [_zero_result(value[0]), *value[1:]]
    if isinstance(value, Mapping):
        # A few backend probes return a named context mapping.  Preserve
        # metadata and zero only the conventional context/value field.
        result = dict(value)
        for key in ("context", "value", "response", "environment_context", "hierarchical_context"):
            if key in result and torch.is_tensor(result[key]):
                result[key] = torch.zeros_like(result[key])
                return result
        return result
    return value


@contextlib.contextmanager
def hierarchical_environment_zero(model: Any, phase: str) -> Iterator[dict[str, Any]]:
    """Remove only NStage2-A's hierarchical environment branch for one phase.

    NStage2-A gathers hierarchical environmental rows in ``_segmented_read``.
    This concrete backend boundary is the only supported intervention point:
    the direct QM read remains in the backend's ordinary ``read`` method.
    """

    if _architecture(model) != HIERARCHICAL_ARCHITECTURE:
        raise ValueError(
            "hierarchical_environment_zero requires "
            f"{HIERARCHICAL_ARCHITECTURE!r}, got {_architecture(model)!r}"
        )
    roles = _phase_roles(phase)
    backend = model.core.backend
    original = getattr(backend, "_segmented_read", None)
    if not callable(original):
        raise TypeError(
            "NStage2-A requires HierarchicalRegionalField._segmented_read; "
            "refusing a broad read hook that would also remove the direct QM branch"
        )
    state = {
        "phase": str(phase),
        "roles": sorted(roles),
        "branch_method": "_segmented_read",
    }

    def zero_segment(*args: Any, **kwargs: Any) -> Any:
        value = original(*args, **kwargs)
        if _role(model) in roles:
            return _zero_result(value)
        return value

    backend._segmented_read = zero_segment

    try:
        yield state
    finally:
        backend._segmented_read = original


@contextlib.contextmanager
def hierarchical_fixed_level_one(model: Any) -> Iterator[dict[str, Any]]:
    """Use the fixed level-1 environmental read only for the final P2 read."""

    if _architecture(model) != HIERARCHICAL_ARCHITECTURE:
        raise ValueError(
            "hierarchical_fixed_level_one requires "
            f"{HIERARCHICAL_ARCHITECTURE!r}, got {_architecture(model)!r}"
        )
    backend = model.core.backend
    original = backend.read

    def read(*args: Any, **kwargs: Any) -> Any:
        if _role(model) == "p2_field":
            kwargs["fixed_level"] = 1
        return original(*args, **kwargs)

    backend.read = read
    try:
        yield {
            "phase": "p2",
            "roles": ["p2_field"],
            "implementation": "fixed_level=1",
        }
    finally:
        backend.read = original


@contextlib.contextmanager
def group_read_zero(model: Any, phase: str = "p2") -> Iterator[dict[str, Any]]:
    """Remove NStage2-B's local group read for one role, preserving env input."""

    if _architecture(model) != GROUP_MEDIATED_ARCHITECTURE:
        raise ValueError(
            "group_read_zero requires "
            f"{GROUP_MEDIATED_ARCHITECTURE!r}, got {_architecture(model)!r}"
        )
    roles = _phase_roles(phase)
    with backend_read_zero(model, roles=tuple(roles), component="main"):
        yield {"phase": str(phase), "roles": sorted(roles), "implementation": "backend_read_zero"}


@contextlib.contextmanager
def group_source_zero(model: Any, phase: str = "p2") -> Iterator[dict[str, Any]]:
    """Remove only group-state input to the coarse processor at one phase.

    NStage2-B's common preparation retains the environmental coarse input and
    accepts ``group_source_enabled=False`` before the unchanged coarse blocks.
    The role is read at preparation time, so this remains scoped to P2 even
    though the group states are constructed before the backend read.
    """

    if _architecture(model) != GROUP_MEDIATED_ARCHITECTURE:
        raise ValueError(
            "group_source_zero requires "
            f"{GROUP_MEDIATED_ARCHITECTURE!r}, got {_architecture(model)!r}"
        )
    roles = _phase_roles(phase)
    common = model.core.common
    original = common.prepare_coarse

    def prepare_coarse(*args: Any, **kwargs: Any) -> Any:
        if _role(model) in roles:
            kwargs["group_source_enabled"] = False
        return original(*args, **kwargs)

    common.prepare_coarse = prepare_coarse
    try:
        yield {"phase": str(phase), "roles": sorted(roles), "implementation": "group_source_enabled"}
    finally:
        common.prepare_coarse = original


@contextlib.contextmanager
def group_mediated_zero(model: Any, phase: str) -> Iterator[dict[str, Any]]:
    """Remove local group read and group-source coarse input for one phase."""

    with contextlib.ExitStack() as stack:
        local_state = stack.enter_context(group_read_zero(model, phase))
        coarse_state = stack.enter_context(group_source_zero(model, phase))
        yield {
            "phase": str(phase),
            "roles": sorted(_phase_roles(phase)),
            "implementation": "group_read_and_group_source",
            "components": {
                "local_group_read": local_state["implementation"],
                "coarse_group_source": coarse_state["implementation"],
            },
        }


# Public aliases make the intervention vocabulary easy to discover from a
# notebook while keeping the CLI names concise.
nstage2_a_environment_zero = hierarchical_environment_zero
nstage2_a_fixed_level_one = hierarchical_fixed_level_one
nstage2_b_group_read_zero = group_read_zero
nstage2_b_group_source_zero = group_source_zero
nstage2_b_group_mediated_zero = group_mediated_zero


def phase_variant_context(model: Any, mode: str) -> contextlib.AbstractContextManager[Any]:
    """Map a study intervention label to one role-scoped context manager."""

    architecture = _architecture(model)
    mode = str(mode)
    if architecture == HIERARCHICAL_ARCHITECTURE:
        aliases = {"A_p0": "p0", "A_p1_only": "p1_only", "A_p2": "p2"}
        mode = aliases.get(mode, mode)
        if mode in {"fixed_level1", "fixed_level_1", "A_fixed_level1", "p2_fixed_level1"}:
            return hierarchical_fixed_level_one(model)
        if mode not in {"p0", "p1_only", "p2"}:
            raise ValueError(f"unknown NStage2-A intervention={mode!r}")
        return hierarchical_environment_zero(model, mode)
    if architecture == GROUP_MEDIATED_ARCHITECTURE:
        aliases = {
            "B_p2_local_only": "p2_local_only",
            "B_p2_coarse_group_only": "p2_coarse_group_only",
            "B_p0": "p0",
            "B_p1_only": "p1_only",
            "B_p2": "p2",
            "p2_local_group": "p2_local_only",
            "p2_group_source": "p2_coarse_group_only",
        }
        mode = aliases.get(mode, mode)
        if mode in {"p0", "p1_only", "p2"}:
            return group_mediated_zero(model, mode)
        if mode == "p2_local_only":
            return group_read_zero(model, "p2")
        if mode == "p2_coarse_group_only":
            return group_source_zero(model, "p2")
        raise ValueError(f"unknown NStage2-B intervention={mode!r}")
    raise ValueError(f"NStage2 diagnostics do not support architecture={architecture!r}")


_phase_variant_context = phase_variant_context


def _prediction_differences(base: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "pred_field",
        "pred_interface",
        "pred_internal_temperature",
        "pred_port_condition",
    ):
        if key in base and key in variant:
            result[key] = _relative_difference(variant[key], base[key])
    return result


def inventory_existing_artifacts(
    root: Path | None = None,
) -> dict[str, Any]:
    """Return availability metadata without selecting unequal checkpoints."""

    project = PROJECT_ROOT if root is None else Path(root).expanduser().resolve()
    run_root = project / "Trained_Results" / "ThermalChannel" / "HONF_Forward_Runs"
    candidates: dict[str, Any] = {}
    for run_id in ("1807", "1808"):
        paths = sorted(run_root.glob(f"Run_{run_id}_*"))
        candidates[run_id] = {
            "directories": [str(path) for path in paths if path.is_dir()],
            "checkpoint_files": [
                str(path)
                for path in run_root.rglob("*")
                if path.is_file() and f"Run_{run_id}" in str(path)
            ],
        }
    parent_exact: dict[str, Any] = {}
    parent_globs = {
        "1401": "Run_1401_*",
        "1801": "Run_1801_*",
        "1802": "Run_1802_*",
        "1804": "Run_1804_*",
        "1805": "Run_1805_*",
        "1806": "Run_1806_*",
    }
    for run_id, pattern in parent_globs.items():
        paths = sorted(path for path in run_root.glob(pattern) if path.is_dir())
        parent_exact[run_id] = [
            str(path / "epoch_0500_model.pt") for path in paths if (path / "epoch_0500_model.pt").is_file()
        ]
    comparison_root = run_root / "CompareModels"
    exact_tables = {
        str(path / "tables")
        for path in comparison_root.glob("*Epoch500*")
        if path.is_dir() and (path / "tables").is_dir()
    }
    study_root = project / "diagnostics" / "generated" / "interface_operator_study"
    for relative in (
        Path("group_reader_recovery") / "endpoint" / "tables",
        Path("regional_response") / "endpoint500" / "tables",
    ):
        path = study_root / relative
        if path.is_dir():
            exact_tables.add(str(path))
    return {
        "run_root": str(run_root),
        "nstage2_candidates": candidates,
        "parent_exact_epoch_500_checkpoints": parent_exact,
        "parent_exact_epoch_500_table_sets": sorted(exact_tables),
        "parent_best_through_epoch_500": {
            "status": "not_located",
            "policy": "Do not infer a through-500 best from full-run best_* files.",
        },
    }


def _module_state_direction(module_states: torch.Tensor, module_index: int) -> torch.Tensor:
    """Return a unit direction in one encoded module-state slot."""

    if module_states.ndim != 3:
        raise ValueError("encoded module states must have shape [B,M,H]")
    if module_index < 0 or module_index >= int(module_states.shape[1]):
        raise IndexError(f"module_index={module_index} is outside module width {module_states.shape[1]}")
    direction = torch.zeros_like(module_states)
    direction[:, module_index] = 1.0
    direction[:, module_index] /= math.sqrt(float(module_states.shape[-1]))
    return direction


def _central_tensor_difference(
    function: Any,
    value: torch.Tensor,
    direction: torch.Tensor,
    step: float,
) -> torch.Tensor:
    with torch.no_grad():
        plus = function(value + float(step) * direction)
        minus = function(value - float(step) * direction)
    return (plus - minus) / (2.0 * float(step))


def _a_routing_details(
    model: Any,
    prepared: Any,
    receiver_sets: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Export detailed hierarchical rows for fixed physical receiver sets."""

    backend = model.core.backend
    state = prepared.backend_state
    encoded = prepared.encoded
    required = {
        "tree_states",
        "tree_coords",
        "tree_mass",
        "tree_valid",
        "tree_levels",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"hierarchical prepared state lacks routing fields: {missing}")
    result: dict[str, Any] = {}
    receiver_node_sets: dict[str, set[int]] = {}
    all_node_indices: list[torch.Tensor] = []
    all_eta: list[torch.Tensor] = []
    node_probe_candidates: list[dict[str, Any]] = []
    for receiver_name, receivers in receiver_sets.items():
        features = model.core._receiver_features(prepared, receivers)
        with torch.no_grad():
            context, aux = backend.read_environment(
                state,
                encoded,
                receivers,
                features,
                return_routing_maps=True,
            )
        batch_index = aux.get("hierarchical_incidence_batch")
        query_index = aux.get("hierarchical_incidence_query")
        node_index = aux.get("hierarchical_incidence_node")
        eta = aux.get("hierarchical_incidence_eta")
        attention = aux.get("hierarchical_incidence_attention")
        if any(value is None for value in (batch_index, query_index, node_index, eta)):
            raise ValueError("hierarchical routing maps were not returned by the concrete backend")
        rows: list[dict[str, Any]] = []
        node_set: set[int] = set()
        for query_number in range(int(receivers.shape[1])):
            selected = (batch_index == 0) & (query_index == query_number)
            nodes = node_index[selected]
            all_node_indices.append(nodes)
            all_eta.append(eta[selected])
            node_set.update(int(value) for value in nodes.detach().cpu().tolist())
            masses = state["tree_mass"][0].index_select(0, nodes)
            levels = state["tree_levels"].index_select(0, nodes)
            node_states = state["tree_states"][0].index_select(0, nodes)
            learned_attention = (
                attention[selected].mean(dim=-1)
                if attention is not None and bool(selected.any())
                else eta.new_empty((0,))
            )
            rows.append(
                {
                    "receiver_index": query_number,
                    "coordinate": receivers[0, query_number],
                    "node_ids": nodes,
                    "levels": levels,
                    "mass": masses,
                    "eta": eta[selected],
                    "learned_attention": learned_attention,
                    "state_norm": torch.linalg.vector_norm(node_states, dim=-1),
                }
            )
            if nodes.numel():
                internal = (state["tree_levels"] > 0) & state["tree_valid"][0].bool()
                internal_ids = torch.nonzero(internal, as_tuple=False).flatten()
                if internal_ids.numel():
                    receiver = receivers[:, query_number : query_number + 1]
                    node_bounds_min = state["tree_bounds_min"][0].index_select(0, internal_ids)
                    node_bounds_max = state["tree_bounds_max"][0].index_select(0, internal_ids)
                    opening = backend._opening_blend(
                        receiver[0].expand(internal_ids.shape[0], -1),
                        node_bounds_min,
                        node_bounds_max,
                        float(backend.response_tree_opening_interval[0]),
                        float(backend.response_tree_opening_interval[1]),
                    )
                    zero_candidates = internal_ids[
                        (opening >= 1.0 - 1.0e-6) & ~torch.isin(internal_ids, nodes)
                    ]
                    if zero_candidates.numel():
                        node_probe_candidates.append(
                            {
                                "receiver_name": receiver_name,
                                "receiver_index": query_number,
                                "receiver": receiver,
                                "selected_node": nodes[:1],
                                "zero_eta_node": zero_candidates[:1],
                            }
                        )
        receiver_node_sets[receiver_name] = node_set
        result[receiver_name] = {
            "receiver_count": int(receivers.shape[1]),
            "context_norm": torch.linalg.vector_norm(context[0], dim=-1),
            "rows": rows,
            "incidence_row_count": int(node_index.numel()),
            "traversal_row_count": int(aux["hierarchical_traversal_rows"].item()),
        }

    aggregate_nodes = (
        torch.cat(all_node_indices, dim=0)
        if all_node_indices
        else state["tree_levels"].new_empty((0,))
    )
    aggregate_eta = (
        torch.cat(all_eta, dim=0)
        if all_eta
        else state["tree_mass"].new_empty((0,))
    )
    zero_mass = state["tree_mass"][0] <= 0.0
    valid_zero_mass = zero_mass & state["tree_valid"][0].bool()
    selected_zero_mass_rows = (
        int(zero_mass.index_select(0, aggregate_nodes).sum().item())
        if aggregate_nodes.numel()
        else 0
    )
    selected_zero_eta_rows = int((aggregate_eta <= 0.0).sum().item())
    shared = set.intersection(*receiver_node_sets.values()) if receiver_node_sets else set()
    result["shared_node_ids"] = sorted(set.union(*receiver_node_sets.values())) if receiver_node_sets else []
    result["cross_receiver_shared_node_ids"] = sorted(shared)
    result["zero_weight_node_ids"] = torch.nonzero(zero_mass, as_tuple=False).flatten()
    result["valid_zero_weight_node_ids"] = torch.nonzero(valid_zero_mass, as_tuple=False).flatten()
    result["zero_weight_direct_influence"] = {
        "status": (
            "verified_no_selected_zero_weight_row"
            if selected_zero_mass_rows == 0 and selected_zero_eta_rows == 0
            else "violation"
        ),
        "selected_zero_mass_row_count": selected_zero_mass_rows,
        "selected_zero_eta_row_count": selected_zero_eta_rows,
        "explanation": (
            "The traversal retains only positive-eta rows, and no selected row has zero mass."
            if selected_zero_mass_rows == 0 and selected_zero_eta_rows == 0
            else "A selected routing row has zero mass and requires investigation."
        ),
    }
    result["zero_weight_node_probe"] = (
        _a_single_node_state_influence(model, prepared, node_probe_candidates[0])
        if node_probe_candidates
        else {
            "status": "unavailable",
            "reason": (
                "no actual receiver had an omitted internal node with opening blend exactly one "
                "(which would produce retained eta exactly zero)"
            ),
        }
    )
    return result


def _a_single_node_state_influence(
    model: Any,
    prepared: Any,
    candidate: Mapping[str, Any],
    *,
    steps: Sequence[float] = (1.0e-3, 5.0e-4),
) -> dict[str, Any]:
    """Perturb one frozen tree state while retaining receiver geometry and Q/K inputs."""

    backend = model.core.backend
    state = prepared.backend_state
    encoded = prepared.encoded
    receiver = candidate["receiver"]
    selected_node = int(candidate["selected_node"][0].item())
    zero_eta_node = int(candidate["zero_eta_node"][0].item())
    base_tree = state["tree_states"].detach()
    hidden = int(base_tree.shape[-1])
    if hidden < 2:
        raise ValueError("single-node control requires at least two tree-state channels")
    row_direction = base_tree.new_zeros((hidden,))
    row_direction[0] = 1.0 / math.sqrt(2.0)
    row_direction[1] = -1.0 / math.sqrt(2.0)

    def read_from_tree(tree_states: torch.Tensor) -> torch.Tensor:
        projected_key, projected_value = backend.project_environment_sources(tree_states)
        frozen_state = dict(state)
        frozen_state["tree_keys"] = projected_key
        frozen_state["tree_values"] = projected_value
        features = model.core._receiver_features(prepared, receiver)
        return backend.read_environment(
            frozen_state,
            encoded,
            receiver,
            features,
            return_routing_maps=False,
        )[0]

    output: dict[str, Any] = {
        "status": "ok",
        "receiver_name": candidate["receiver_name"],
        "receiver_index": int(candidate["receiver_index"]),
        "selected_node_id": selected_node,
        "omitted_zero_eta_node_id": zero_eta_node,
        "receiver_geometry_fixed": True,
        "query_projection_and_geometry_fixed": True,
        "state_direction": "deterministic mean-zero [1,-1,0,...]/sqrt(2) perturbation of one frozen tree-state row",
        "state_direction_vector": row_direction.detach().cpu().tolist(),
        "nodes": {},
    }
    for label, node in (
        ("selected_positive_eta", selected_node),
        ("omitted_zero_eta", zero_eta_node),
    ):
        direction = torch.zeros_like(base_tree)
        direction[:, node] = row_direction
        jvp = torch.autograd.functional.jvp(
            read_from_tree,
            (base_tree,),
            (direction,),
            create_graph=False,
            strict=False,
        )[1]
        finite = {
            f"h={step:g}": _finite_stats(
                _central_tensor_difference(read_from_tree, base_tree, direction, step)
            )
            for step in steps
        }
        output["nodes"][label] = {
            "jvp": _finite_stats(jvp),
            "jvp_norm": float(torch.linalg.vector_norm(jvp).detach().cpu()),
            "finite_difference": finite,
        }
    output["interpretation"] = (
        "The omitted-node probe changes only the frozen projected tree key/value row; the receiver query, geometry bias, eta, and routing incidence remain fixed."
    )
    return output


def _a_conditional_route_probe(
    model: Any,
    prepared: Any,
    receiver_sets: Mapping[str, torch.Tensor],
    module_index: int,
    *,
    steps: Sequence[float] = (1.0e-3, 5.0e-4),
) -> dict[str, Any]:
    """Probe A's fresh tree, fine-EM, and direct QM routes under one perturbation."""

    backend = model.core.backend
    encoded = prepared.encoded
    base_states = prepared.module_states.detach()
    direction = _module_state_direction(base_states, module_index)
    step_values = tuple(float(step) for step in steps)

    def fresh_state(states: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return backend.prepare(encoded, states, return_routing_maps=False)

    def tree_states(states: torch.Tensor) -> torch.Tensor:
        return fresh_state(states)["tree_states"]

    def fine_em_messages(states: torch.Tensor) -> torch.Tensor:
        return fresh_state(states)["environment_messages"]

    valid = fresh_state(base_states)["tree_valid"].bool()
    tree_jvp = torch.autograd.functional.jvp(
        tree_states,
        (base_states,),
        (direction,),
        create_graph=False,
        strict=False,
    )[1]
    tree_steps = {
        f"h={step:g}": _central_tensor_difference(tree_states, base_states, direction, step)
        for step in step_values
    }
    fine_em_jvp = torch.autograd.functional.jvp(
        fine_em_messages,
        (base_states,),
        (direction,),
        create_graph=False,
        strict=False,
    )[1]
    fine_em_steps = {
        f"h={step:g}": _central_tensor_difference(fine_em_messages, base_states, direction, step)
        for step in step_values
    }

    def route_read(states: torch.Tensor, receivers: torch.Tensor, *, qm: bool) -> torch.Tensor:
        state = fresh_state(states)
        features = model.core._receiver_features(prepared, receivers)
        if qm:
            return backend.read_module(state, encoded, receivers, features)
        return backend.read_environment(
            state,
            encoded,
            receivers,
            features,
            return_routing_maps=False,
        )[0]

    output: dict[str, Any] = {
        "module_index": int(module_index),
        "state_direction": "unit normalized all-ones direction in the first active encoded module state",
        "geometry_fixed": True,
        "global_context_fixed": True,
        "raw_environment_encodings_fixed": True,
        "fresh_tree_preparation": True,
        "conditional_ad_fd_validation": {
            "agreement": "reported_without_threshold",
            "steps": step_values,
            "reason": (
                "The prescribed 1e-3 and 5e-4 state steps are retained for the diagnostic; "
                "no step sweep or higher-precision full-model check is applied."
            ),
        },
        "hierarchical_state": {
            "jvp": _finite_stats(tree_jvp, valid),
            "steps": {
                key: _finite_stats(value, valid) for key, value in tree_steps.items()
            },
        },
        "fineEM_environment_messages": {
            "jvp": _finite_stats(fine_em_jvp),
            "steps": {
                key: _finite_stats(value) for key, value in fine_em_steps.items()
            },
            "nonzero_environment_source_count": int(
                (torch.linalg.vector_norm(fine_em_jvp, dim=-1) > 1.0e-8).sum().item()
            ),
        },
        "receivers": {},
    }
    for receiver_name, receivers in receiver_sets.items():
        mask = torch.ones(receivers.shape[:2], device=receivers.device, dtype=torch.bool)
        route_row: dict[str, Any] = {"receiver_count": int(receivers.shape[1])}
        for route_name, qm in (("hierarchical_environment", False), ("QM_direct_module", True)):
            route = lambda states, receivers=receivers, qm=qm: route_read(
                states, receivers, qm=qm
            )
            route_jvp = torch.autograd.functional.jvp(
                route,
                (base_states,),
                (direction,),
                create_graph=False,
                strict=False,
            )[1]
            route_steps = {}
            for step in step_values:
                finite = _central_tensor_difference(route, base_states, direction, step)
                summary = _regional_jvp_summary(route_jvp, finite, mask)
                route_steps[f"h={step:g}"] = {
                    "step": step,
                    "summary": summary,
                    "agreement": "reported_without_threshold",
                }
            route_row[route_name] = {
                "jvp": _finite_stats(route_jvp),
                "steps": route_steps,
                "nonzero_receiver_count": int(
                    (torch.linalg.vector_norm(route_jvp, dim=-1) > 1.0e-8).sum().item()
                ),
            }
        output["receivers"][receiver_name] = route_row
    output["interpretation"] = {
        "fineEM": "Module-conditioned environmental messages and freshly prepared tree states measure broad environmental influence across retained sources.",
        "QM_direct_module": "The direct module route is reported separately and must not be conflated with broad fine-EM environmental influence.",
        "hierarchical_environment": "The hierarchy route reads freshly prepared environmental tree states at fixed receiver geometry.",
        "jvp_fd": "Assess numerical agreement from the reported discrepancies at both prescribed state steps. Small float32 responses can have poorly resolved finite differences; a nonzero JVP alone does not establish a reliable sensitivity magnitude. No step sweep or higher-precision full-model check is applied.",
    }
    return output


def _opening_blend_float64_reference(
    receivers: torch.Tensor,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    near_radius: float,
    far_radius: float,
) -> torch.Tensor:
    """Evaluate the opening smoothstep in float64 for numerical comparison only."""

    receivers64 = receivers.detach().to(dtype=torch.float64)
    bounds_min64 = bounds_min.detach().to(dtype=torch.float64)
    bounds_max64 = bounds_max.detach().to(dtype=torch.float64)
    center = 0.5 * (bounds_min64 + bounds_max64)
    radius = 0.5 * torch.linalg.vector_norm(bounds_max64 - bounds_min64, dim=-1)
    radius = radius.clamp_min(torch.finfo(torch.float64).eps)
    relative = receivers64 - center
    normalized_squared_distance = relative.square().sum(dim=-1) / radius.square()
    near_squared = float(near_radius) ** 2
    far_squared = float(far_radius) ** 2
    transition = (
        (normalized_squared_distance - near_squared) / (far_squared - near_squared)
    ).clamp(0.0, 1.0)
    one_minus = 1.0 - transition
    return one_minus.square() * (1.0 + 2.0 * transition)


def _a_resolution_shell_probe(
    model: Any,
    prepared: Any,
) -> dict[str, Any]:
    """Construct one hierarchy shell with endpoint and interior coordinate checks."""

    backend = model.core.backend
    state = prepared.backend_state
    levels = state["tree_levels"]
    valid = state["tree_valid"][0].bool()
    candidates = torch.nonzero(valid & (levels > 0), as_tuple=False).flatten()
    if not candidates.numel():
        return {"status": "unavailable", "reason": "prepared tree has no valid internal node"}
    node = int(candidates[0].item())
    bounds_min = state["tree_bounds_min"][0, node]
    bounds_max = state["tree_bounds_max"][0, node]
    center = 0.5 * (bounds_min + bounds_max)
    radius = 0.5 * torch.linalg.vector_norm(bounds_max - bounds_min)
    near_radius, far_radius = (
        float(value) for value in backend.response_tree_opening_interval
    )
    if not (0.0 < near_radius < far_radius):
        raise ValueError("hierarchical response opening interval must satisfy 0 < near < far")
    epsilon = max(float(radius.detach().cpu()) * 1.0e-4, 1.0e-6)
    distances = (
        ("interior", 0.5 * (near_radius + far_radius) * float(radius)),
        ("near_minus", near_radius * float(radius) - epsilon),
        ("near_plus", near_radius * float(radius) + epsilon),
        ("far_minus", far_radius * float(radius) - epsilon),
        ("far_plus", far_radius * float(radius) + epsilon),
    )
    direction = center.new_tensor([1.0, 0.0])
    points = torch.stack([center + direction * distance for _, distance in distances], dim=0)
    features = model.core._receiver_features(prepared, points.unsqueeze(0))
    with torch.no_grad():
        _, aux = backend.read_environment(
            state,
            prepared.encoded,
            points.unsqueeze(0),
            features,
            return_routing_maps=True,
        )
    incidence_node = aux.get("hierarchical_incidence_node", points.new_empty((0,), dtype=torch.long))
    incidence_query = aux.get("hierarchical_incidence_query", points.new_empty((0,), dtype=torch.long))
    incidence_eta = aux.get("hierarchical_incidence_eta", points.new_empty((0,)))
    blend = backend._opening_blend(
        points,
        bounds_min.expand(points.shape[0], -1),
        bounds_max.expand(points.shape[0], -1),
        near_radius,
        far_radius,
    )
    reference_blend = _opening_blend_float64_reference(
        points,
        bounds_min.expand(points.shape[0], -1),
        bounds_max.expand(points.shape[0], -1),
        near_radius,
        far_radius,
    )
    endpoint_rows = []
    for index, (label, distance) in enumerate(distances):
        selected = incidence_query == index
        endpoint_rows.append(
            {
                "label": label,
                "distance_from_node_center_over_radius": float(distance / float(radius)),
                "coordinate": points[index],
                "opening_blend": blend[index],
                "opening_blend_actual_model_dtype": blend[index],
                "retained_eta_complement_actual_model_dtype": 1.0 - blend[index],
                "opening_blend_float64_scalar_reference": reference_blend[index],
                "retained_eta_complement_float64_scalar_reference": 1.0 - reference_blend[index],
                "target_node_eta": incidence_eta[(selected) & (incidence_node == node)],
                "incidence_node_ids": incidence_node[selected],
                "incidence_eta": incidence_eta[selected],
            }
        )

    def environment_scalar(receiver: torch.Tensor) -> torch.Tensor:
        # Recompute receiver Fourier features for each perturbed point while
        # retaining the prepared tree state, masses, and geometry tensors.
        receiver_features = model.core._receiver_features(prepared, receiver)
        context, _ = backend.read_environment(
            state,
            prepared.encoded,
            receiver,
            receiver_features,
            return_routing_maps=False,
        )
        return context[..., 0].sum()

    coordinate_steps = (1.0e-2, 5.0e-3)
    for index, row in enumerate(endpoint_rows):
        # The maintained reader consumes batched receiver coordinates with
        # shape (batch, query, xy), including for a single shell point.
        point = (
            points[index : index + 1]
            .unsqueeze(0)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        scalar = environment_scalar(point)
        gradient = (
            torch.autograd.grad(scalar, point, allow_unused=True)[0]
            if scalar.requires_grad
            else None
        )
        autograd_value = (
            0.0
            if gradient is None
            else float(gradient[..., 0].sum().detach().cpu())
        )
        derivatives: dict[str, Any] = {}
        for step in coordinate_steps:
            with torch.no_grad():
                plus = point.detach().clone()
                minus = point.detach().clone()
                plus[..., 0] += float(step)
                minus[..., 0] -= float(step)
                finite_value = (
                    environment_scalar(plus) - environment_scalar(minus)
                ) / (2.0 * float(step))
            finite = float(finite_value.detach().cpu())
            derivatives[f"h={step:g}"] = {
                "step": step,
                "autograd": autograd_value,
                "central_fd": finite,
                "absolute_difference": abs(autograd_value - finite),
                "relative_difference": abs(autograd_value - finite)
                / max(abs(finite), 1.0e-12),
                "signed": True,
            }
        row["coordinate_read_derivative"] = {
            "functional": "sum of first hierarchical environment-context channel",
            "direction": [1.0, 0.0],
            "steps": derivatives,
            "geometry_fixed": True,
            "tree_state_fixed": True,
            "receiver_features_refreshed": True,
        }
    return {
        "status": "ok",
        "node_id": node,
        "node_level": levels[node],
        "node_mass": state["tree_mass"][0, node],
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "near_radius": near_radius,
        "far_radius": far_radius,
        "epsilon": epsilon,
        "endpoints": endpoint_rows,
        "coordinate_steps": coordinate_steps,
        "opening_blend_reference": {
            "actual_model_dtype": str(blend.dtype),
            "float64_reference_only": True,
            "same_points_bounds_and_interval": True,
            "description": (
                "The float64 scalar reference evaluates the opening smoothstep and its "
                "retained-eta complement at the same FP32-valued points, bounds, and "
                "configured interval. It does not replace the actual model read or routing."
            ),
            "numerical_limitation": (
                "At the prescribed epsilon=1e-4*radius shell offset, the actual FP32 "
                "smoothstep can round the near-entry opening blend to one and the retained "
                "eta complement to zero. Actual incidences and AD/FD rows remain unchanged."
            ),
        },
        "interpretation": "The interior and four compact-support side points are a constructed shell around one internal node. Signed AD/central-FD values vary only receiver coordinates with prepared tree state and geometry fixed; no agreement threshold is applied.",
    }


def _b_receiver_connectivity(
    layout: Any,
    receivers: torch.Tensor,
    module_index: int,
) -> dict[str, Any]:
    from honf_forward_core.interface_fields.supports import lookup_receivers

    incidence = lookup_receivers(layout, receivers)
    source, module_group = layout.module_group_indices
    local_source = torch.remainder(source, int(layout.module_count))
    selected_module_groups = torch.unique(module_group[local_source == int(module_index)])
    receiver_groups: list[set[int]] = []
    for receiver_index in range(int(receivers.shape[1])):
        selected = incidence.receiver_group_indices[0] == receiver_index
        groups = {
            int(value)
            for value in torch.unique(incidence.receiver_group_indices[1, selected]).detach().cpu().tolist()
        }
        receiver_groups.append(groups)
    module_groups = {int(value) for value in selected_module_groups.detach().cpu().tolist()}
    rows = []
    labels = ("near_port", "far_field")
    for label, groups in zip(labels, receiver_groups):
        rows.append(
            {
                "receiver": label,
                "group_ids": sorted(groups),
                "shared_with_selected_module": sorted(groups.intersection(module_groups)),
                "connected_to_selected_module": bool(groups.intersection(module_groups)),
                "support_degree": int(incidence.degree[0, len(rows)].item()),
            }
        )
    shared = receiver_groups[0].intersection(receiver_groups[1]) if len(receiver_groups) == 2 else set()
    return {
        "selected_module_index": int(module_index),
        "selected_module_group_ids": sorted(module_groups),
        "receivers": rows,
        "near_far_shared_group_ids": sorted(shared),
        "near_far_share_connectivity": bool(shared),
    }


def _b_conditional_route_probe(
    model: Any,
    prepared: Any,
    receiver_sets: Mapping[str, torch.Tensor],
    module_index: int,
    *,
    steps: Sequence[float] = (1.0e-3, 5.0e-4),
) -> dict[str, Any]:
    """Compare B local and group-mediated coarse module dependence."""

    backend = model.core.backend
    common = model.core.common
    encoded = prepared.encoded
    base_states = prepared.module_states.detach()
    layout_cache = prepared.backend_state.cache
    direction = _module_state_direction(base_states, module_index)
    step_values = tuple(float(step) for step in steps)

    def backend_state(states: torch.Tensor) -> Any:
        return backend.prepare(encoded, states, layout_cache)

    def local_read(states: torch.Tensor, receivers: torch.Tensor) -> torch.Tensor:
        state = backend_state(states)
        features = model.core._receiver_features(prepared, receivers)
        return backend.read(state, encoded, receivers, features, return_routing_maps=False)[0]

    def coarse_read(
        states: torch.Tensor,
        receivers: torch.Tensor,
        *,
        detached_groups: bool,
    ) -> torch.Tensor:
        state = backend_state(states)
        packed = backend.packed_coarse_group_sources(state)
        group_states = packed.group_states.detach() if detached_groups else packed.group_states
        occupancy = packed.occupancy.detach() if detached_groups else packed.occupancy
        valid = packed.valid.detach() if detached_groups else packed.valid
        coarse_state = common.prepare_coarse(
            states,
            encoded.env_tokens,
            encoded.module_present,
            encoded.env_weights,
            packed_group_states=group_states,
            packed_group_occupancy=occupancy,
            packed_group_valid=valid,
        )
        features = model.core._receiver_features(prepared, receivers)
        return common.read_coarse(features, encoded.global_token, coarse_state)

    def route_summary(route: Any, receivers: torch.Tensor) -> dict[str, Any]:
        mask = torch.ones(receivers.shape[:2], device=receivers.device, dtype=torch.bool)
        jvp = torch.autograd.functional.jvp(
            route,
            (base_states,),
            (direction,),
            create_graph=False,
            strict=False,
        )[1]
        steps_summary = {}
        for step in step_values:
            finite = _central_tensor_difference(route, base_states, direction, step)
            summary = _regional_jvp_summary(jvp, finite, mask)
            steps_summary[f"h={step:g}"] = {
                "step": step,
                "summary": summary,
                "agreement": "reported_without_threshold",
            }
        return {"jvp": _finite_stats(jvp), "steps": steps_summary}

    def detached_coarse_jvp(receivers: torch.Tensor) -> torch.Tensor:
        return torch.autograd.functional.jvp(
            lambda states, receivers=receivers: coarse_read(
                states, receivers, detached_groups=True
            ),
            (base_states,),
            (direction,),
            create_graph=False,
            strict=False,
        )[1]

    layout = layout_cache.layout
    near = receiver_sets["physical_ports"][:, :1]
    field_receivers = receiver_sets["field_probes"]
    center = encoded.module_centers[0, module_index]
    distances = torch.linalg.vector_norm(field_receivers[0] - center, dim=-1)
    far_index = int(distances.argmax().item())
    far = field_receivers[:, far_index : far_index + 1]
    connectivity = _b_receiver_connectivity(
        layout,
        torch.cat([near, far], dim=1),
        module_index,
    )
    result: dict[str, Any] = {
        "module_index": int(module_index),
        "state_direction": "unit normalized all-ones direction in the first active encoded module state",
        "geometry_fixed": True,
        "global_context_fixed": True,
        "environmental_background_fixed": True,
        "local_correction_fixed": True,
        "coarse_processor_retained": True,
        "conditional_ad_fd_validation": {
            "agreement": "reported_without_threshold",
            "steps": step_values,
            "reason": (
                "The prescribed 1e-3 and 5e-4 state steps are retained for the diagnostic; "
                "no step sweep or higher-precision full-model check is applied."
            ),
        },
        "receiver_connectivity": connectivity,
        "selected_receiver_indices": {
            "near_port": 0,
            "far_field": far_index,
        },
        "selected_receiver_influence": {},
        "receivers": {},
    }
    for receiver_name, receivers in receiver_sets.items():
        route_row: dict[str, Any] = {"receiver_count": int(receivers.shape[1])}
        for route_name, route in (
            (
                "local_group_read",
                lambda states, receivers=receivers: local_read(states, receivers),
            ),
            (
                "group_mediated_coarse",
                lambda states, receivers=receivers: coarse_read(
                    states, receivers, detached_groups=False
                ),
            ),
        ):
            route_row[route_name] = route_summary(route, receivers)
        detached_jvps = []
        for _ in range(2):
            detached_jvps.append(detached_coarse_jvp(receivers))
        detached_norms = [
            float(torch.linalg.vector_norm(value).detach().cpu()) for value in detached_jvps
        ]
        repeat_delta = torch.linalg.vector_norm(detached_jvps[0] - detached_jvps[1])
        route_row["detached_group_mediated_coarse"] = {
            "repeat_count": len(detached_jvps),
            "jvp_norms": detached_norms,
            "repeat_jvp_difference_norm": float(repeat_delta.detach().cpu()),
            "direct_module_to_coarse_jvp_removed": all(value <= 1.0e-10 for value in detached_norms),
            "finite_difference_note": "Detached group states remove the direct autograd path; finite differences would still compare different detached values and are therefore not used as a zero-path test.",
        }
        result["receivers"][receiver_name] = route_row
    for selected_name, selected_receiver in (("near_port", near), ("far_field", far)):
        local_route = lambda states, receivers=selected_receiver: local_read(states, receivers)
        coarse_route = lambda states, receivers=selected_receiver: coarse_read(
            states, receivers, detached_groups=False
        )
        detached_jvps = [detached_coarse_jvp(selected_receiver) for _ in range(2)]
        detached_norms = [
            float(torch.linalg.vector_norm(value).detach().cpu()) for value in detached_jvps
        ]
        result["selected_receiver_influence"][selected_name] = {
            "local_group_read": route_summary(local_route, selected_receiver),
            "group_mediated_coarse": route_summary(coarse_route, selected_receiver),
            "detached_group_mediated_coarse": {
                "jvp_norms": detached_norms,
                "repeat_jvp_difference_norm": float(
                    torch.linalg.vector_norm(detached_jvps[0] - detached_jvps[1]).detach().cpu()
                ),
                "direct_module_to_coarse_jvp_removed": all(
                    value <= 1.0e-10 for value in detached_norms
                ),
            },
        }
    result["interpretation"] = {
        "local_group_read": "Compact conditional group connectivity remains available at its receiver supports.",
        "group_mediated_coarse": "The environmental coarse background and unchanged coarse processor remain active while live group states carry module dependence.",
        "conditional_ad_fd": "Assess numerical agreement from the reported discrepancies at both prescribed state steps. Small float32 responses can have poorly resolved finite differences; a nonzero JVP alone does not establish a reliable sensitivity magnitude. No step sweep or higher-precision full-model check is applied.",
        "detached_groups": "Repeated detached-group JVPs test the structural zero direct module-to-coarse path; this exact structural result is kept separate from the unconfirmed conditional JVP/FD magnitudes.",
    }
    return result


def _physical_kpi_for_centers(
    model: Any,
    sample: Mapping[str, Any],
    query: torch.Tensor,
    centers: torch.Tensor,
    device: torch.device,
    metric: str,
) -> torch.Tensor:
    """Compute a fixed-query KPI from the model's raw normalized field output.

    The maintained forward helper returns ``pred_field`` without applying the
    dataset target denormalization transform.  The coordinate is physical
    geometry space, but the resulting scalar and its derivative remain in
    normalized model-output units.
    """

    outputs = _forward_tensor_batch(model, sample, query, device, centers=centers)
    values = _field_kpis(
        outputs["pred_field"],
        query,
        list(model.config.channelthermal.field_names),
    )
    if metric not in values:
        raise KeyError(f"field KPI {metric!r} is unavailable")
    return values[metric]


def _physical_coordinate_probe(
    model: Any,
    sample: Mapping[str, Any],
    query: torch.Tensor,
    device: torch.device,
    module_index: int,
    direction: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run signed full-path AD/FD checks in normalized model-output units."""

    radius = float(model.config.core_honf.module_radius)
    steps = (0.01 * radius, 0.005 * radius)
    if direction is None:
        direction = _direction_for_sample(sample, model, module_index, max(steps))
    else:
        direction = np.asarray(direction, dtype=np.float64).reshape(-1)
        if direction.shape != (2,) or not np.isfinite(direction).all():
            raise ValueError("direction must be a finite 2-vector")
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError("direction must be nonzero")
        direction = direction / norm
    result: dict[str, Any] = {
        "module_index": int(module_index),
        "direction": direction,
        "coordinate_units": "model geometry coordinate units",
        "functional_units": "normalized model-output units per coordinate unit",
        "denormalization_applied": False,
        "steps": {},
        "signed_functionals": {
            "mean_temperature": "fixed-query mean temperature in normalized model-output units",
            "pressure_drop": "fixed-query inlet-minus-outlet pressure in normalized model-output units using x quartiles",
        },
    }
    for metric in ("mean_temperature", "pressure_drop"):
        metric_rows: dict[str, Any] = {}
        for step in steps:
            try:
                if metric == "mean_temperature":
                    autograd_value = _autograd_directional(
                        model,
                        sample,
                        query,
                        module_index,
                        direction,
                        device,
                        fine=False,
                    )
                    finite_value = _finite_difference_directional(
                        model,
                        sample,
                        query,
                        module_index,
                        direction,
                        step,
                        device,
                        fine=False,
                    )
                else:
                    centers_np = np.asarray(sample["structure"]["module_centers"], dtype=np.float32).copy()
                    centers = torch.from_numpy(centers_np).unsqueeze(0).to(device).requires_grad_(True)
                    scalar = _physical_kpi_for_centers(
                        model, sample, query, centers, device, metric
                    )
                    gradient = torch.autograd.grad(scalar, centers, allow_unused=True)[0]
                    if gradient is None:
                        autograd_value = 0.0
                    else:
                        vector = torch.as_tensor(direction, device=device, dtype=gradient.dtype)
                        autograd_value = float(
                            torch.dot(gradient[0, int(module_index)], vector).detach().cpu()
                        )
                    plus = centers_np.copy()
                    minus = centers_np.copy()
                    plus[module_index] += direction * float(step)
                    minus[module_index] -= direction * float(step)
                    present = _active_centers(sample)[1]
                    lx, ly = _domain_lengths(sample, model)
                    if not _valid_geometry(
                        plus,
                        present,
                        radius,
                        lx,
                        ly,
                    ) or not _valid_geometry(
                        minus,
                        present,
                        radius,
                        lx,
                        ly,
                    ):
                        raise ValueError("central-difference step leaves the valid geometry domain")
                    plus_value = _physical_kpi_for_centers(
                        model,
                        sample,
                        query,
                        torch.from_numpy(plus).unsqueeze(0).to(device),
                        device,
                        metric,
                    )
                    minus_value = _physical_kpi_for_centers(
                        model,
                        sample,
                        query,
                        torch.from_numpy(minus).unsqueeze(0).to(device),
                        device,
                        metric,
                    )
                    finite_value = float(
                        ((plus_value - minus_value) / (2.0 * float(step))).detach().cpu()
                    )
                metric_rows[f"h={step:g}"] = {
                    "step": step,
                    "autograd": autograd_value,
                    "finite_difference": finite_value,
                    "relative_difference": abs(autograd_value - finite_value)
                    / max(abs(finite_value), 1.0e-12),
                    "signed": True,
                }
            except (KeyError, ValueError) as exc:
                metric_rows[f"h={step:g}"] = {
                    "status": "unavailable",
                    "reason": f"{type(exc).__name__}: {exc!s}",
                }
        result["steps"][metric] = metric_rows
    return result


def _b_support_transition_probe(
    model: Any,
    sample: Mapping[str, Any],
    query: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    radius = float(model.config.core_honf.module_radius)
    max_step = 0.01 * radius
    try:
        aligned_sample, module_index, direction, metadata = _align_one_port_to_support_transition(
            sample,
            model,
            max_step,
        )
        centers = np.asarray(
            aligned_sample["structure"]["module_centers"], dtype=np.float64
        )
        plus = centers.copy()
        minus = centers.copy()
        plus[module_index] += direction * max_step
        minus[module_index] -= direction * max_step
        plus_keys = _support_key_set_for_centers(
            model, aligned_sample, query, plus, module_index, device
        )
        minus_keys = _support_key_set_for_centers(
            model, aligned_sample, query, minus, module_index, device
        )
        return {
            "status": "ok",
            "module_index": int(module_index),
            "direction": direction,
            "step": max_step,
            "aligned_metadata": metadata,
            "minus_group_count": len(minus_keys),
            "plus_group_count": len(plus_keys),
            "symmetric_key_difference_count": len(plus_keys.symmetric_difference(minus_keys)),
            "minus_keys": sorted(minus_keys),
            "plus_keys": sorted(plus_keys),
            "topology_changed": bool(plus_keys != minus_keys),
            "coordinate_sensitivity": _physical_coordinate_probe(
                model,
                aligned_sample,
                query,
                device,
                module_index,
                direction=direction,
            ),
            "interpretation": "This is the maintained aligned compact-support transition construction; no finite-difference step was tuned after construction.",
        }
    except ValueError as exc:
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc!s}",
            "interpretation": "No transition result is substituted when the maintained construction cannot find a valid geometry.",
        }


def run_probes(args: argparse.Namespace) -> dict[str, Any]:
    """Run plan-10.4 conditional influence and coordinate probes."""

    specs = parse_checkpoint_specs(args.checkpoint)
    if len(specs) != 1:
        raise ValueError("NStage2 probes requires exactly one labelled checkpoint")
    device = select_device(args.device)
    spec = specs[0]
    model, checkpoint = _load_model_spec(spec, device)
    architecture = _architecture(model)
    if architecture not in {HIERARCHICAL_ARCHITECTURE, GROUP_MEDIATED_ARCHITECTURE}:
        raise ValueError(f"NStage2 probes do not support architecture={architecture!r}")
    dataset, dataset_path = _load_dataset(checkpoint, args)
    case_ids = [str(value) for value in (args.case_id or DEFAULT_PROBE_CASE_IDS)]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("NStage2 probe case IDs must be unique")
    state_before = _state_keys(model)
    rows: list[dict[str, Any]] = []
    try:
        from channelthermal.interface_field_coupling import _port_coordinates

        for case_id in case_ids:
            sample = select_sample(dataset, case_id, 0)
            query = torch.from_numpy(
                _query_points(sample, int(args.query_count))
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                base = _forward_tensor_batch(
                    model,
                    sample,
                    query,
                    device,
                    return_prepared_state=True,
                    return_routing_maps=True,
                )
            prepared = base["prepared_state"].prepared
            encoded = prepared.encoded
            active_modules = torch.nonzero(encoded.module_present[0] > 0.5).flatten()
            if not active_modules.numel():
                raise ValueError(f"case {case_id} contains no active module")
            source_module = int(active_modules[0].item())
            ntheta = int(base["pred_port_condition"].shape[-2])
            all_ports = _port_coordinates(model, encoded.module_centers, ntheta)
            selected_modules = active_modules[:2]
            angle_indices = torch.arange(
                0,
                ntheta,
                max(1, ntheta // 4),
                device=device,
            )[:4]
            ports = all_ports[:, selected_modules][:, :, angle_indices].reshape(1, -1, 2)
            receiver_sets = {"physical_ports": ports, "field_probes": query}
            row: dict[str, Any] = {
                "case_id": case_id,
                "architecture": architecture,
                "query_count": int(query.shape[1]),
                "source_module_index": source_module,
                "active_module_indices": active_modules,
                "port_module_indices": selected_modules,
                "port_angle_indices": angle_indices,
                "receiver_sets": receiver_sets,
                "conditional_encoded_module": {
                    "geometry_fixed": True,
                    "global_context_fixed": True,
                    "raw_environment_encodings_fixed": True,
                },
                "physical_coordinate_sensitivity": _physical_coordinate_probe(
                    model,
                    sample,
                    query,
                    device,
                    source_module,
                ),
            }
            if architecture == HIERARCHICAL_ARCHITECTURE:
                row["ordinary_hierarchical_routing"] = _a_routing_details(
                    model,
                    prepared,
                    receiver_sets,
                )
                row["conditional_encoded_module"] = _a_conditional_route_probe(
                    model,
                    prepared,
                    receiver_sets,
                    source_module,
                )
                row["resolution_transition_shell"] = _a_resolution_shell_probe(model, prepared)
            else:
                row["conditional_encoded_module"] = _b_conditional_route_probe(
                    model,
                    prepared,
                    receiver_sets,
                    source_module,
                )
            rows.append(row)
        if architecture == GROUP_MEDIATED_ARCHITECTURE:
            transition_case = case_ids[1] if len(case_ids) > 1 else case_ids[0]
            transition_sample = select_sample(dataset, transition_case, 0)
            transition_query = torch.from_numpy(
                _query_points(transition_sample, int(args.query_count))
            ).unsqueeze(0).to(device)
            rows[case_ids.index(transition_case)]["support_transition"] = _b_support_transition_probe(
                model,
                transition_sample,
                transition_query,
                device,
            )
    finally:
        dataset.close()
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "task": "probes",
        "study": "honf_nstage2",
        "plan_section": "10.4",
        "candidate_family": "A_hierarchical_regional"
        if architecture == HIERARCHICAL_ARCHITECTURE
        else "B_group_mediated_reader",
        "device": str(device),
        "checkpoint": _checkpoint_record(spec, checkpoint, model),
        "dataset": str(dataset_path),
        "case_ids": case_ids,
        "results": rows,
        "artifact_inventory": inventory_existing_artifacts(),
        "state_dict_structure_unchanged": state_before == _state_keys(model),
        "interpretation": {
            "A": "A conditionally perturbs one encoded module state while retaining geometry, global context, and raw environmental encodings; module-conditioned fine-EM environmental messages/tree states and direct QM are reported as separate routes at the same actual ports and fixed field probes.",
            "B": "B compares local group and group-mediated coarse routes with environmental coarse background, global input, local correction, and coarse processor retained; repeated detached groups test the direct structural path.",
            "coordinate_derivatives": "Signed fixed-query mean-temperature and pressure-drop AD/FD values are reported in normalized model-output units; they are frozen-model self-consistency checks, not physical-reference sensitivity evidence.",
        },
        "limitations": [
            "No training, optimizer update, managed run allocation, or extra optimizer step is performed.",
            "The probes are fixed case diagnostics for 0273 and 0298, not a population estimate.",
            "A zero-weight node exclusion and B detached-group JVP establish graph-level statements only; they do not replace physical-reference validation.",
            "Missing or unconstructible transition examples are reported as unavailable rather than filled with simulated results.",
            "Conditional route AD/FD discrepancies are reported at the prescribed 1e-3 and 5e-4 state steps without an automatic pass/fail threshold. Assess each stored checkpoint from its measured values; poor agreement limits quantitative sensitivity claims.",
            "The A shell reports a float64 opening-blend calculation only as a same-point numerical reference; actual model FP32 incidences and AD/FD values are retained unchanged.",
        ],
    }


def _candidate_modes(architecture: str) -> tuple[str, ...]:
    if architecture == HIERARCHICAL_ARCHITECTURE:
        return ("p0", "p1_only", "p2", "fixed_level1")
    if architecture == GROUP_MEDIATED_ARCHITECTURE:
        return ("p0", "p1_only", "p2", "p2_local_only", "p2_coarse_group_only")
    raise ValueError(f"NStage2 diagnostics do not support architecture={architecture!r}")


def run_interventions(args: argparse.Namespace) -> dict[str, Any]:
    """Run bounded phase/component removals against one stored candidate."""

    specs = parse_checkpoint_specs(args.checkpoint)
    if len(specs) != 1:
        raise ValueError("NStage2 interventions requires exactly one labelled checkpoint")
    device = select_device(args.device)
    spec = specs[0]
    model, checkpoint = _load_model_spec(spec, device)
    architecture = _architecture(model)
    modes = _candidate_modes(architecture)
    dataset, dataset_path = _load_dataset(checkpoint, args)
    try:
        case_ids = [str(value) for value in (args.case_id or DEFAULT_ANCHOR_CASE_IDS)]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("NStage2 anchor case IDs must be unique")
        raw_samples = {
            case_id: _load_raw_sample(dataset_path, args.split, case_id)
            for case_id in case_ids
        }
        state_before = _state_keys(model)
        rows: list[dict[str, Any]] = []
        for case_id in case_ids:
            sample = __import__(
                "channelthermal.evaluation.prepared", fromlist=["select_sample"]
            ).select_sample(dataset, case_id, 0)
            query_np = _query_points(sample, int(args.query_count))
            base = _run_variant(model, sample, query_np, device)
            base_errors = _canonical_ground_truth_errors(
                base, sample, raw_samples[case_id], dataset, model, checkpoint
            )
            row: dict[str, Any] = {
                "case_id": case_id,
                "selection_reason": ANCHOR_SELECTION_REASONS.get(
                    case_id, "caller-specified case; selection predates this run"
                ),
                "architecture": architecture,
                "query_count": len(query_np),
                "ground_truth_errors": {"normal": base_errors},
                "interventions": {},
            }
            for mode in modes:
                variant = _run_variant(
                    model,
                    sample,
                    query_np,
                    device,
                    phase_variant_context(model, mode),
                )
                errors = _canonical_ground_truth_errors(
                    variant, sample, raw_samples[case_id], dataset, model, checkpoint
                )
                row["ground_truth_errors"][mode] = errors
                differences = _prediction_differences(base, variant)
                row["interventions"][mode] = {
                    "prediction_difference": differences.get("pred_field", {}),
                    "prediction_differences": differences,
                    "error_deltas": _error_deltas(base_errors, errors),
                    "role_scope": (
                        sorted(_phase_roles(mode))
                        if mode in {"p0", "p1_only", "p2"}
                        else ["p2_field"]
                    ),
                }
            rows.append(row)
        return {
            "schema_version": STUDY_SCHEMA_VERSION,
            "task": "interventions",
            "study": "honf_nstage2",
            "candidate_family": "A_hierarchical_regional"
            if architecture == HIERARCHICAL_ARCHITECTURE
            else "B_group_mediated_reader",
            "checkpoint": _checkpoint_record(spec, checkpoint, model),
            "dataset": str(dataset_path),
            "case_ids": case_ids,
            "case_selection_reasons": {
                case_id: ANCHOR_SELECTION_REASONS.get(
                    case_id, "caller-specified case; selection predates this run"
                )
                for case_id in case_ids
            },
            "results": rows,
            "artifact_inventory": inventory_existing_artifacts(),
            "state_dict_structure_unchanged": state_before == _state_keys(model),
            "interpretation": {
                "prediction_difference": "frozen-model reliance only",
                "error_delta": "intervened-minus-normal canonical ground-truth error; positive worsens error",
                "A": "hierarchical environment branch only; common coarse/local routes are preserved",
                "A_fixed_level1": "P2 environmental read uses fixed hierarchy level 1 on the same prepared weights",
                "B_phase": "local group read and group-source coarse input are both removed at the tagged phase; environmental coarse input remains",
                "B_local": "group read only at P2; environmental preparation and coarse source are preserved",
                "B_coarse": "group-state coarse source only at P2; environmental coarse input and group read are preserved",
            },
            "limitations": [
                "No training, optimizer update, or managed run allocation is performed.",
                "These are frozen-checkpoint intervention measurements, not new physical-reference validation.",
                "Parent full-run best checkpoints are not compared with NStage2 candidates unless an explicit matched budget is supplied.",
            ],
        }
    finally:
        dataset.close()


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Delegate the physical two-step smoke exactly to the maintained helper."""

    from run_regional_response_study import run_smoke as _run_existing_smoke  # type: ignore[import-not-found]

    return _run_existing_smoke(args)


def run_timing(args: argparse.Namespace) -> dict[str, Any]:
    """Delegate timing to the maintained regional protocol without copying it."""

    from run_regional_response_study import run_timing_protocol  # type: ignore[import-not-found]

    return run_timing_protocol(args)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    parser.add_argument("--device", default="cpu", help="evaluation device, e.g. cpu or cuda:0")
    parser.add_argument("--dataset", default=None, help="override the packed HDF5 dataset path")
    parser.add_argument("--split", default="test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)

    smoke = subparsers.add_parser(
        "smoke", help="delegate the existing two-real-batch physical smoke"
    )
    smoke.add_argument("--output", type=Path, required=True, help="JSON output path")
    smoke.add_argument("--profile", default=DEFAULT_PROFILE)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--points-per-case", type=int, default=128)
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--case-count", type=int, default=1)
    smoke.add_argument("--small-case-id", action="append", default=None)
    smoke.add_argument("--large-case-id", action="append", default=None)

    interventions = subparsers.add_parser(
        "interventions",
        aliases=("phase_interventions", "phase-interventions"),
        help="bounded NStage2 phase/component interventions",
    )
    _add_common_arguments(interventions)
    interventions.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    interventions.add_argument("--case-id", action="append", default=None)
    interventions.add_argument("--query-count", type=int, default=DEFAULT_NSTAGE2_QUERY_COUNT)
    interventions.set_defaults(handler=run_interventions)

    probes = subparsers.add_parser(
        "probes",
        aliases=("conditional_probes", "conditional-probes"),
        help="plan-10.4 conditional influence, transition, and coordinate probes",
    )
    _add_common_arguments(probes)
    probes.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    probes.add_argument("--case-id", action="append", default=None)
    probes.add_argument("--query-count", type=int, default=DEFAULT_PROBE_QUERY_COUNT)
    probes.set_defaults(handler=run_probes)

    timing = subparsers.add_parser(
        "timing",
        help="delegate the maintained phase-separated timing protocol",
    )
    _add_common_arguments(timing)
    timing.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    timing.add_argument("--case-id", action="append", default=None)
    timing.add_argument("--query-count", type=int, default=8192)
    timing.add_argument("--shape", action="append", default=[], metavar="M,E,Q")
    timing.add_argument("--query-batch-size", type=int, default=32768)
    timing.add_argument("--receiver-chunk-size", type=int, default=2048)
    timing.add_argument("--warmup", type=int, default=2)
    timing.add_argument("--repetitions", type=int, default=5)
    timing.add_argument("--synthetic-warmup", type=int, default=1)
    timing.add_argument("--synthetic-repetitions", type=int, default=3)
    timing.add_argument("--execution-test", action="store_true")
    timing.set_defaults(handler=run_timing)
    smoke.set_defaults(handler=run_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = args.handler(args)
    payload["output"] = str(Path(args.output).expanduser().resolve())
    write_json(Path(args.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
