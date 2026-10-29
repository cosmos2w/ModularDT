# Model families and workflow identifiers

The public model catalogue in [`honf_forward_core.models`](../src/honf_forward_core/models.py) exposes six named families: `hypergraph_routed`, `dense_pairwise`, `content_attention`, `unified_hypergraph`, `relation_anchored` and `relation_anchored_active`. A family selects an original strict core configuration. It does not reconstruct a physical case or replace the case checkpoint loader and saved normalization.

The older plugin registry in [`honf_runtime.registry`](../src/honf_runtime/registry.py) uses coarser serialized workflow identifiers. `honf_forward` retains six-hyperedge and dense managed workflows; `honf_content_hypergraph` retains content-attention workflows. `honf_inverse` remains reserved in that registry; standalone case inverse contracts are separate. Unknown or retired workflow identifiers raise an error. Preserve old checkpoint identifiers rather than renaming them to the new public display names.

Cases own physical records, channel semantics, targets, masks, normalization, losses and verification. Core modules remain case-neutral, and new case integrations must not weaken strict existing checkpoint or resume validation.
