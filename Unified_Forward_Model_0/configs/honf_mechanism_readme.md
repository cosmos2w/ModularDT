# Mechanism-Aware HONF Config Notes

The mechanism encoder treats each learned hyperedge as a generic field mechanism: a source coordinate, a field-region coordinate, source-to-region displacement, module/environment mass, strength, and learned token state. These descriptors are generic source-region features, not thermal-specific rules.

## New Fields

- `use_hyper_mechanism_encoder`: enables the decoder-side mechanism encoder.
- `mechanism_include_geometry`: includes normalized source/region coordinates, displacement, distance, downstream/upstream, and lateral descriptors.
- `mechanism_include_masses`: includes normalized/log-scaled module and environment mass plus hyperedge strength.
- `mechanism_hidden_dim`: hidden width for the mechanism MLP; `null` uses `hidden_dim`.
- `hyper_attention_topk`: `0` keeps dense query-to-H softmax; `1` or `2` makes each query use only top-k hyperedges.
- `hyper_attention_temperature`: softmax temperature for query-to-H attention.
- `sparse_hyper_attention_detach_mask`: detaches the top-k mask while keeping gradients through selected logits.

## Recommended Values

- `hyper_attention_topk: 0` for dense compatibility checks.
- `hyper_attention_topk: 2` for the sparse interpretable mechanism-HONF model.
- `hyper_attention_temperature: 1.0` initially.
- `pairwise_kernel_hidden_dim: 64` for memory-controlled enhanced HONF.
- `num_hyperedges: 6` or `8` for ChannelThermal comparisons.

## Diagnostics To Inspect

- `hyper_attention_effective_edges`
- `active_edge_count`
- `env_mass_entropy_norm` and `module_mass_entropy_norm`
- `pairwise_context_norm` versus `hyper_context_norm`
- `pairwise_kernel_gate`
