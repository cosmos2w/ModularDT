# HONF topology signature

Schema v3 exports a design's learned organization as an unordered active-edge
set plus edge-to-edge relations. It is separate from the schema-v2 hypergraph
plan, which remains available for its existing checkpoint and inverse-design
consumers.

The compressed NPZ contract contains:

- `edge_mask [K_cap]`, `edge_features [K_cap,F_t]`, and
  `edge_relations [K_cap,K_cap,F_r]`;
- selected module/environment incidence and optional pre-selection candidate
  incidence;
- reference-query routing summaries and per-channel signed-mean, RMS, and
  energy-fraction contribution summaries;
- the padded module-slot width and active module count as distinct scalars;
- candidate and active edge counts;
- reference-measure identity, query count, and a SHA-256 digest of the exact
  float32 probe coordinates;
- case and forward-checkpoint provenance; and
- `serialization_permutation`, mapping serialized edges back to runtime slots.

`canonicalize_topology_signature` exists for deterministic files and display.
It is not a training target and does not assign persistent edge identity.
`compare_topology_signatures` selects active edges, matches token features with
Hungarian assignment (including a configurable unmatched-edge cost), then
compares matched relation tensors. A pure edge permutation therefore has zero
distance.

For ChannelThermal evaluation, opt in with:

```bash
evaluate.py --workflow forward ... --export-topology-signature
```

This uses the deterministic evaluation grid as the reference query measure,
records checkpoint SHA-256 provenance, renders active source/region ellipses,
membership matrices, an overlap graph, and—when edge-additive outputs are
available—per-field contribution maps. If case-owned solved or fallback
structure targets are available, relation-reconstruction metrics are written
with their source explicitly labeled. They are diagnostics only; no topology
loss or edge-count penalty is enabled.

Two exported signatures can be compared without assuming serialized edge
identity:

```bash
python tools/compare_honf_topologies.py first.npz second.npz
```
