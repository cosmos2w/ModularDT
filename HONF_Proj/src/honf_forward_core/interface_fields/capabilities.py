"""Explicit spatial capabilities for the shared field architecture registry."""

CAMPAIGN_ARCHITECTURES = frozenset({
    "adaptive_receiver_hypergraph_honf",
    "faithful_receiver_hypergraph_honf",
    "native_context_tree_honf",
    "native_context_global_control_honf",
    "overlap_control_hypergraph_honf",
    "local_overlap_hypergraph_honf",
})

THREE_DIMENSIONAL_ARCHITECTURES = frozenset({
    "legacy_honf", "dense_pairwise_field", "three_term_full_access_honf",
    "direct_pairwise_control_honf",
    "adaptive_interaction_cover_honf", "routed_pairwise_honf", "fixed_group_pairwise_honf",
    *CAMPAIGN_ARCHITECTURES,
})


def supported_spatial_dimensions(architecture: str) -> tuple[int, ...]:
    return (2, 3) if architecture in THREE_DIMENSIONAL_ARCHITECTURES else (2,)


def has_typed_hypergraph_export(architecture: str) -> bool:
    return architecture in CAMPAIGN_ARCHITECTURES
