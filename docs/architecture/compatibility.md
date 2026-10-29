# Forward compatibility contract

The public catalogue exposes hypergraph-routed, dense pairwise, content-attention, unified hypergraph, relation-anchored and active-domain relation-anchored families. Existing six-hyperedge and dense checkpoints retain their serialized architecture identifiers, tensor names, normalization, physical transforms and case adapters. Public family names do not replace those saved interfaces.

Use the matching case loader and original model configuration. Evaluation loads state strictly rather than accepting missing or unexpected weights. Managed profiles are indexed in `src/config_core/forward/profile_registry.json`; a profile describes a compatible workflow, not a scientific baseline or a guarantee for every case.

Training resume preserves parameter order, optimizer groups and state, RNG, sampler, data membership, normalization, schedule and complete history. Contract tests cover strict checkpoint loading and optimizer/RNG next-update behavior. A successful load alone does not establish a completed continuation or field accuracy on new data.

Canonical run and evaluation layouts have one producer location. Historical selectors remain readable, and existing run artifacts are not silently rewritten. See [checkpoint handling](../checkpoints.md) and [run artifacts](../guides/Run_Artifacts_and_Monitoring.md).
