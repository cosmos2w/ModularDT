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
