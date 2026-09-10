"""Prepared-case collection and chunked field decoding."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.evaluation.loading import make_batch
from channelthermal.model import ChannelThermalHONFModel


# Sparse-interface layouts are flattened across a case batch rather than
# padded with a leading batch dimension.  ``predict_case`` always evaluates
# one physical case, but blindly removing axis 0 from these arrays would still
# discard all but the first group (or the first row of a COO incidence list).
_SPARSE_FLATTENED_INTERACTION_KEYS = {
    "support_centres",
    "support_lattice_keys",
    "support_group_batch",
    "support_case_group_offsets",
    "module_group_indices",
    "module_geometric_membership",
    "module_learned_membership",
    "module_weighted_membership",
    "environment_group_indices",
    "environment_geometric_membership",
    "environment_learned_membership",
    "environment_weighted_membership",
    "group_occupancy",
    "group_occupancy_envelope",
    "group_covered_volume_ratio",
    "group_module_degree",
    "group_environment_degree",
    "group_state_norm",
    "group_key_norm",
    "group_value_norm",
}


def serialize_interaction_aux(aux: Dict[str, Any]) -> Dict[str, Any]:
    """Detach one-case interaction diagnostics without truncating sparse COO data."""

    architecture = str(aux.get("forward_architecture", ""))
    result: Dict[str, Any] = {}
    for key, value in aux.items():
        if not torch.is_tensor(value):
            result[key] = value
            continue
        array = value.detach().cpu().numpy()
        if (
            architecture == "sparse_interface_honf"
            and key in _SPARSE_FLATTENED_INTERACTION_KEYS
        ):
            result[key] = array
        else:
            result[key] = array[0] if array.ndim > 0 else array
    return result


def _interaction_tensor_request_kwargs(model: Any, requested: bool) -> dict[str, bool]:
    """Return an opt-in wrapper flag for Phase-2 interaction tensors."""

    if not requested:
        return {}
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - unusual proxy models
        return {}
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    for name in (
        "return_interaction_tensor",
        "return_residual_interaction_tensor",
        "return_tensor_diagnostics",
        # ChannelThermal's public facade names the opt-in organizer bundle
        # ``return_organizer_diagnostics`` while the generic core uses the
        # more specific residual-tensor spelling.
        "return_organizer_diagnostics",
    ):
        if name in parameters or accepts_kwargs:
            return {name: True}
    return {}


def select_sample(dataset: GlobalChannelThermalDataset, case_id: Optional[str], case_index: int) -> Dict[str, Any]:
    """Select sample."""

    if len(dataset) == 0:
        raise RuntimeError("No global channel thermal cases are available for evaluation.")
    if case_id is not None:
        for idx, candidate in enumerate(dataset.selected_case_ids):
            if str(candidate) == str(case_id):
                return dataset[idx]
        raise KeyError(f"case_id={case_id!r} not found in split {dataset.split!r}.")
    return dataset[min(max(int(case_index), 0), len(dataset) - 1)]


def aggregate_routed_module_retention(chunks: Sequence[np.ndarray]) -> Dict[str, float]:
    """Aggregate routed-pair retention values exactly across query chunks."""

    values = (
        np.concatenate([np.asarray(chunk).reshape(-1) for chunk in chunks]).astype(
            np.float64, copy=False
        )
        if chunks
        else np.zeros((0,), dtype=np.float64)
    )
    return {
        "routed_module_retained_mass_mean": float(np.mean(values)) if values.size else 0.0,
        "routed_module_retained_mass_p05": float(np.quantile(values, 0.05)) if values.size else 0.0,
        "routed_module_retained_mass_min": float(np.min(values)) if values.size else 0.0,
        "routed_query_edge_pair_count": float(values.size),
    }


def predict_case(
    model: ChannelThermalHONFModel,
    sample: Dict[str, Any],
    device: torch.device,
    *,
    query_batch_size: int,
    local_port_condition_mode: str,
    mixed_teacher_ratio: float,
    return_routing_maps: bool = False,
    return_topology_signature: bool = False,
    return_prepared_state: bool = False,
    return_interaction_tensor: bool = False,
    case_edge_selection_mode: str | None = None,
    case_edge_probe_relative_rms_tolerance: float | None = None,
    case_edge_probe_channel_tolerance: float | None = None,
) -> Dict[str, Any]:
    """Prepare one physical case once, then decode its query grid in chunks.

    Evaluation-only diagnostics may request the retained prepared state.  The
    default remains the historical compact result and does not expose the
    wrapper-internal state to callers.  The interaction tensor is similarly
    opt-in because it is only needed for Phase-2 rank/visualization reports.
    """

    x_grid = sample["x_grid"]
    y_grid = sample["y_grid"]
    query_xy = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1).astype(np.float32)
    pred_chunks = []
    routing_chunks: Dict[str, list[np.ndarray]] = {}
    routed_module_retention_chunks: list[np.ndarray] = []
    first_outputs = None
    prepared_state = None
    need_routing = bool(return_routing_maps or return_topology_signature)
    with torch.no_grad():
        for start in range(0, query_xy.shape[0], int(query_batch_size)):
            chunk = query_xy[start : start + int(query_batch_size)]
            if prepared_state is None:
                batch = make_batch(sample, chunk, device)
                outputs = model(
                    batch["structure"],
                    batch["query_xy"],
                    interface_condition=batch.get("interface_condition"),
                    local_module_params=batch.get("local_module_params"),
                    teacher_port_tokens=batch.get("teacher_port_tokens"),
                    local_query_points=batch.get("module_internal_query_points"),
                    local_port_condition_mode=local_port_condition_mode,
                    mixed_teacher_ratio=float(mixed_teacher_ratio),
                    return_routing_maps=need_routing,
                    return_edge_fields=bool(return_topology_signature),
                    return_prepared_state=True,
                    case_edge_selection_mode=case_edge_selection_mode,
                    case_edge_probe_relative_rms_tolerance=case_edge_probe_relative_rms_tolerance,
                    case_edge_probe_channel_tolerance=case_edge_probe_channel_tolerance,
                    **_interaction_tensor_request_kwargs(model, return_interaction_tensor),
                )
                prepared_state = outputs.pop("prepared_state")
                first_outputs = outputs
            else:
                chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).to(device=device)
                decoder_output = model.decode_prepared(
                    prepared_state,
                    chunk_tensor,
                    return_routing_maps=need_routing,
                    return_edge_fields=bool(return_topology_signature),
                )
                outputs = {
                    "pred_field": decoder_output["pred_field"],
                    "routing_aux": {key: value for key, value in decoder_output.items() if key != "pred_field"},
                }
            pred_chunks.append(outputs["pred_field"].detach().cpu().numpy()[0])
            routing_aux = outputs.get("routing_aux", {})
            retained_module_mass = routing_aux.get("retained_module_incidence_mass")
            routed_pair_mask = routing_aux.get("routed_query_edge_pair_mask")
            if torch.is_tensor(retained_module_mass) and torch.is_tensor(routed_pair_mask):
                retained_np = retained_module_mass.detach().cpu().numpy()[0]
                routed_np = routed_pair_mask.detach().cpu().numpy()[0].astype(bool, copy=False)
                routed_module_retention_chunks.append(retained_np[routed_np])
            if need_routing:
                key_map = {
                    "query_hyper_attention": "query_hyper_attention",
                    "pairwise_edge_contribution": "pairwise_edge_contribution",
                    "c_H_norm": "c_H_norm",
                    "c_pair_norm": "c_pair_norm",
                    "dominant_hyperedge": "dominant_hyperedge",
                    "hyper_attention_entropy_map": "hyper_attention_entropy",
                    "dense_module_context_norm": "dense_module_context_norm",
                    "main_context_norm": "main_context_norm",
                    "coarse_context_norm": "coarse_context_norm",
                    "local_context_norm": "local_context_norm",
                    "main_context_fraction": "main_context_fraction",
                    "coarse_context_fraction": "coarse_context_fraction",
                    "local_context_fraction": "local_context_fraction",
                    "local_neighbor_count": "local_neighbor_count",
                    "group_read_degree": "group_read_degree",
                    "group_read_weight_mass": "group_read_weight_mass",
                    "group_read_geometric_availability": "group_read_geometric_availability",
                    "group_read_conditional_weight": "group_read_conditional_weight",
                    "group_read_conditional_value_norm": "group_read_conditional_value_norm",
                    "group_read_logit_mean": "group_read_logit_mean",
                    "group_read_logit_std": "group_read_logit_std",
                    "group_read_max_weight": "group_read_max_weight",
                    "group_read_group_index": "group_read_group_index",
                    "group_read_geometric_weight": "group_read_geometric_weight",
                    "group_read_normalized_weight": "group_read_normalized_weight",
                }
                for source_key, target_key in key_map.items():
                    value = routing_aux.get(source_key)
                    if torch.is_tensor(value):
                        routing_chunks.setdefault(target_key, []).append(value.detach().cpu().numpy()[0])
                for source_key in (
                    "dense_environment_attention",
                    "regional_environment_attention",
                    "latent_query_attention",
                ):
                    value = routing_aux.get(source_key)
                    if torch.is_tensor(value):
                        # The backend reports [batch, heads, query, source].  A
                        # head mean leaves a compact, receiver-indexed influence
                        # map that concatenates correctly across query chunks.
                        compact = value.detach().float().mean(dim=1).cpu().numpy()[0]
                        routing_chunks.setdefault(source_key, []).append(compact)
            if return_topology_signature:
                edge_fields = outputs.get("pred_field_by_edge")
                if edge_fields is None:
                    edge_fields = outputs.get("routing_aux", {}).get("pred_field_by_edge")
                if torch.is_tensor(edge_fields):
                    routing_chunks.setdefault("pred_field_by_edge", []).append(
                        edge_fields.detach().cpu().numpy()[0]
                    )
    if first_outputs is None:
        raise RuntimeError("No prediction chunks were produced.")
    pred_field = np.concatenate(pred_chunks, axis=0).reshape(*x_grid.shape, model.config.field_dim)
    result = {
        "pred_field_grid": pred_field.astype(np.float32),
        "pred_internal_temperature": first_outputs["pred_internal_temperature"].detach().cpu().numpy()[0],
        "pred_interface": first_outputs["pred_interface"].detach().cpu().numpy()[0],
        "pred_port_condition": first_outputs["pred_port_condition"].detach().cpu().numpy()[0],
        "pred_port_condition_raw": first_outputs.get(
            "pred_port_condition_raw", first_outputs["pred_port_condition"]
        ).detach().cpu().numpy()[0],
        "interface_flux_mode": first_outputs.get("interface_source", "unknown"),
        "organizer_aux": {
            key: value.detach().cpu().numpy()[0] if torch.is_tensor(value) and value.ndim > 0 else value
            for key, value in first_outputs["organizer_aux"].items()
        },
        "base_organizer_aux": {
            key: value.detach().cpu().numpy()[0] if torch.is_tensor(value) and value.ndim > 0 else value
            for key, value in first_outputs.get("base_organizer_aux", {}).items()
        },
        "interaction_aux": serialize_interaction_aux(
            {
                key: value
                for key, value in first_outputs.get("interaction_aux", {}).items()
                if key not in {
                    "dense_environment_attention",
                    "regional_environment_attention",
                    "latent_query_attention",
                }
            }
        ),
    }
    result["routing_aux"] = aggregate_routed_module_retention(
        routed_module_retention_chunks
    )
    if need_routing:
        result["routing_maps"] = {
            key: np.concatenate(chunks, axis=0)
            for key, chunks in routing_chunks.items()
            if chunks
        }
    if return_topology_signature and prepared_state is not None:
        structure_targets = sample.get("structure_targets", {})
        target_coords = structure_targets.get("env_token_coords") if isinstance(structure_targets, dict) else None
        if target_coords is not None:
            with torch.no_grad():
                target_query = torch.as_tensor(target_coords, dtype=torch.float32, device=device).unsqueeze(0)
                target_output = model.decode_prepared(
                    prepared_state,
                    target_query,
                    return_routing_maps=True,
                    return_edge_fields=False,
                )
            target_attention = target_output.get("query_hyper_attention")
            if torch.is_tensor(target_attention):
                result["structure_query_hyper_attention"] = target_attention.detach().cpu().numpy()[0]
    if return_prepared_state and prepared_state is not None:
        result["_prepared_state"] = prepared_state
    return result
