from __future__ import annotations

import numpy as np
import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.evaluation.topology_signature import (
    canonicalize_topology_signature,
    compare_topology_signatures,
    evaluate_structure_relations,
    extract_topology_signature,
    load_topology_signature,
    save_topology_signature,
    validate_topology_signature,
)
from honf_forward_core.model import HONFNeuralField


def _config() -> UnifiedForwardConfig:
    return UnifiedForwardConfig(
        field_dim=3,
        domain_length_x=6.0,
        domain_length_y=3.0,
        coordinate_scale=[6.0, 3.0],
        periodic_axes=[],
        num_env_tokens_x=5,
        num_env_tokens_y=3,
        num_hyperedges=4,
        organizer_mode="exchangeable_slots",
        edge_capacity=6,
        initial_active_edges=4,
        minimum_active_edges=2,
        edge_selection_mode="quality_coverage",
        selection_coverage_rate=0.8,
        module_assignment_normalizer="entmax15",
        environment_assignment_normalizer="entmax15",
        query_assignment_normalizer="entmax15",
        environment_locality_mode="compact_kernel",
        environment_locality_strength=1.0,
        minimum_region_scale=0.08,
        hidden_dim=24,
        dropout=0.0,
        decoder_mode="enhanced_honf_pairwise",
        pairwise_kernel_hidden_dim=24,
        pairwise_kernel_num_layers=2,
        mechanism_state_mode="descriptor_first",
        field_assembly_mode="edge_additive",
        routing_execution="gathered",
        query_module_limit=3,
        query_edge_limit=3,
    )


def _batch() -> BatchData:
    generator = torch.Generator().manual_seed(227)
    return BatchData(
        module_centers=torch.rand(1, 7, 2, generator=generator) * torch.tensor([6.0, 3.0]),
        module_present=torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        module_features=torch.randn(1, 7, 5, generator=generator),
        global_context=torch.randn(1, 4, generator=generator),
        query_xy=torch.rand(1, 31, 2, generator=generator) * torch.tensor([6.0, 3.0]),
        query_time=None,
        target_field=None,
        case_name="topology-signature-test",
        metadata={},
    )


def _model_and_outputs() -> tuple[HONFNeuralField, BatchData, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(229)
    model = HONFNeuralField(_config()).eval()
    batch = _batch()
    with torch.no_grad():
        organized = model.encode_and_organize(batch)
        decoded = model.decode_queries(
            batch.query_xy,
            None,
            organized,
            organized["global_token"],
            return_routing_maps=True,
            return_edge_fields=True,
        )
    return model, batch, organized, decoded


def _extract(
    organized: dict[str, torch.Tensor],
    batch: BatchData,
    decoded: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
    *,
    canonicalize: bool = True,
) -> dict[str, np.ndarray]:
    return extract_topology_signature(
        organized,
        batch.module_present,
        decoder_outputs=decoded,
        reference_query_xy=batch.query_xy,
        reference_measure="fixed_test_queries",
        field_names=("u", "v", "temperature"),
        domain_length_x=6.0,
        domain_length_y=3.0,
        case_id=batch.case_name,
        canonicalize=canonicalize,
    )


def _permute_edges(signature: dict[str, np.ndarray], permutation: np.ndarray) -> dict[str, np.ndarray]:
    result = {key: np.asarray(value).copy() for key, value in signature.items()}
    for key in ("edge_mask", "edge_features", "query_route_summary", "field_contribution_summary"):
        result[key] = result[key][permutation]
    for key in (
        "module_incidence",
        "environment_incidence",
        "candidate_module_incidence",
        "candidate_environment_incidence",
    ):
        if key in result:
            result[key] = result[key][:, permutation]
    result["edge_relations"] = result["edge_relations"][permutation][:, permutation]
    result["serialization_permutation"] = np.arange(len(permutation), dtype=np.int64)
    return result


def test_schema_round_trip_and_distinct_module_counts(tmp_path) -> None:
    _, batch, organized, decoded = _model_and_outputs()
    signature = _extract(organized, batch, decoded)
    path = tmp_path / "topology_signature.npz"
    save_topology_signature(path, signature)
    restored = load_topology_signature(path)

    assert int(restored["schema_version"]) == 3
    assert int(restored["num_module_slots"]) == 7
    assert int(restored["active_module_count"]) == 3
    assert int(restored["candidate_edge_count"]) == 6
    assert int(restored["active_edge_count"]) == int(restored["edge_mask"].sum())
    assert int(restored["field_contribution_available"]) == 1
    for key in signature:
        np.testing.assert_array_equal(restored[key], canonicalize_topology_signature(signature)[key])


def test_matching_and_canonical_content_are_edge_permutation_invariant() -> None:
    _, batch, organized, decoded = _model_and_outputs()
    signature = _extract(organized, batch, decoded, canonicalize=False)
    permutation = np.asarray([3, 0, 5, 2, 1, 4])
    permuted = _permute_edges(signature, permutation)
    validate_topology_signature(permuted)

    comparison = compare_topology_signatures(signature, permuted)
    assert comparison["topology_distance"] == 0.0
    assert comparison["unmatched_edge_count"] == 0

    first = canonicalize_topology_signature(signature)
    second = canonicalize_topology_signature(permuted)
    for key in first:
        if key != "serialization_permutation":
            np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(
        canonicalize_topology_signature(first)["serialization_permutation"],
        first["serialization_permutation"],
    )


def test_reference_query_summaries_are_chunk_independent() -> None:
    model, batch, organized, full = _model_and_outputs()
    chunks = []
    with torch.no_grad():
        for query in torch.tensor_split(batch.query_xy, 5, dim=1):
            if query.shape[1]:
                chunks.append(
                    model.decode_queries(
                        query,
                        None,
                        organized,
                        organized["global_token"],
                        return_routing_maps=True,
                        return_edge_fields=True,
                    )
                )
    full_signature = _extract(organized, batch, full, canonicalize=False)
    chunk_signature = _extract(organized, batch, chunks, canonicalize=False)

    np.testing.assert_allclose(
        full_signature["query_route_summary"], chunk_signature["query_route_summary"], rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        full_signature["field_contribution_summary"],
        chunk_signature["field_contribution_summary"],
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        full_signature["edge_relations"], chunk_signature["edge_relations"], rtol=2e-6, atol=2e-6
    )


def test_optional_relation_metrics_are_finite_and_labeled() -> None:
    _, batch, organized, decoded = _model_and_outputs()
    signature = _extract(organized, batch, decoded, canonicalize=False)
    module_target = np.eye(7, dtype=np.float32)
    route = decoded["query_hyper_attention"].detach().cpu().numpy()[0]
    environment_target = np.zeros((route.shape[0], 7), dtype=np.float32)
    metrics = evaluate_structure_relations(
        signature,
        module_affinity_target=module_target,
        query_hyper_attention=route,
        environment_module_target=environment_target,
        active_edge_count_target=np.asarray([2.0]),
        has_solved_targets=False,
    )

    assert metrics["target_source"] == "fallback"
    assert np.isfinite(metrics["module_affinity_mse"])
    assert np.isfinite(metrics["environment_module_influence_mse"])
    assert np.isfinite(metrics["active_edge_count_absolute_error"])
