# HONF forward-core contract

This package implements the application-neutral Hypergraph Operator Neural
Field. It accepts `BatchData` tensors, creates module and environment tokens,
organizes them into learned hyperedges, and decodes a field at arbitrary query
coordinates. It does not own physical field names, HDF5 schemas, material
features, boundary conditions, local solvers, or plotting.

Supported decoder modes are preserved from the reference implementation,
including global HONF decoding, local near-field augmentation, and the enhanced
pairwise kernel. Learned/uniform organizer assignment, learned/geometric query
routing, sparse top-k routing, Fourier geometry features, mechanism features,
geometry bias, auxiliary incidence output, and prepared-state chunked decoding
remain configurable.

Reusable neural blocks live in `nn.py`. One configurable MLP implementation
preserves both historical dropout-index layouts, and one Fourier implementation
explicitly supports the core's grouped $\pi 2^k$ convention and the local
module's interleaved $2\pi 2^k$ convention. These options prevent silent
feature-order or state-key changes when loading established checkpoints.

The stable batch fields are `module_centers`, `module_present`,
`module_features`, `global_context`, `env_coords`, `env_features`,
`query_xy`, optional `query_features`/`query_time`, optional `target_field`, and
JSON-safe metadata. `query_xy` is retained as a compatibility name; its
semantics are generic query coordinates.

Maintained cases supply domain boundary descriptors through `query_features`;
the core's `rectangular` mode and its historical `channel` alias remain only to
reconstruct established checkpoints without changing their query-encoder input.
Neutral `coordinate_scale=[s_x,s_y]` and `periodic_axes=[...]` settings govern
normalization and minimum-image geometry. Historical configs fall back to
`domain_length_x/y` and interpret `geometry_mode="periodic"` as both axes.

Core code may depend on PyTorch and NumPy only. A physical case must adapt its
samples to this contract and interpret output channels itself. See the root
`Model_Explain.md` for equations and `docs/case_plugin.md` for the extension
boundary.

## Curated reusable model bundle

This source bundle exposes the historical `HONFNeuralField`, the `InterfaceFieldCore` dense pairwise path, and `ContentHypergraphCoreV2` with its A2-D architecture. `HONFNeuralField` and the mature Dense interface modules use the reusable source cut at `97916241e6f11326cc8babce5d412fefc827a053` (the same core family as merge `f4f393413b838749952e75be783e4fbd1570b54b`); the A2-D content modules and their small route-contract support come from readiness commit `8036452dd9c5e6f69e5a3592bdb6d7251bbe4702`.

The `content_hypergraph` package keeps the V2 API and its imported support files, while package exports load on demand. This avoids importing the unrelated native-retrieval implementation when constructing `ContentHypergraphCoreV2`. The `native_retrieval` and `inherited_attention` package initializers expose only the contract files required by the V2 staged-attention code in this bundle.

`BatchData` retains the historical module and environment inputs, optional `env_weights`, query features, and case metadata. Adapter-specific record IDs, characteristic lengths, and physical controls remain case-owned inputs rather than new shared-core fields.

`InterfaceFieldCore` is configured with `forward_architecture="dense_pairwise_field"` and `interface_model` settings. Its reusable call boundary is `encode_case(batch)`, `prepare(encoded, encoded.module_tokens)`, and `decode_queries(prepared, query_xy, query_features)`; `HONFNeuralField` retains its `forward(BatchData)` interface. `ContentHypergraphCoreV2` accepts its own `SceneBatch` and `QueryBatch` contracts and is not routed through `BatchData`.
