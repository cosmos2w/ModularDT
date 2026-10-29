# HONF hierarchical inverse core

The inverse core is case-neutral. It models the one-to-many hierarchy

```text
structured request R + context c -> compact mechanism plan G
compact G + R + c             -> padded physical design D
frozen case verifier(D,c)      -> realized mechanism G_hat
```

`RequestSetEncoder` is permutation invariant. `ConditionalPlanFlow` and
`ConditionalLayoutFlow` use conditional rectified flow for continuous state
and supervised binary/count heads. `JointConsistencyCorrector` is optional and
can be applied once only. Case physics, exact functionals, HDF5, and forward
model calls do not enter this package.

Two explicit topology paths are available. Missing mode fields select the
checkpoint-compatible `indexed` plan flow and `ordered_flat` layout
conditioning. `exchangeable_set` removes the edge embedding, uses the noisy
flow state plus shared self-attention for token differentiation, supports
runtime edge capacity, and trains against Sinkhorn-matched set targets.
`set_cross_attention` pools active topology tokens and cross-attends layout
slots to them without flattening an ordered edge axis. Its dataset provenance
must name `honf_topology_signature` schema version 3 and the exact forward
checkpoint SHA-256; it cannot consume the compact-plan dataset implicitly.

The public inference shape is:

```python
designer = HierarchicalInverseDesigner.load("best_corrected_model.pt", device="cuda:0")
designer.attach_verifier(case_runtime)
result = designer.sample_candidates(
    request=request,
    context=context,
    num_plans=8,
    layouts_per_plan=4,
    correct_once=True,
)
```

The serializable result separates four meanings:

- `raw_unguided`: generated and exactly verified raw candidates;
- `corrected`: exactly verified one-pass proposals, including worse proposals;
- `accepted_one_pass`: one representative per lineage, using the proposal only
  when exact request violation improves and geometry remains valid;
- `final_ranked`: diversity-aware top candidates selected from accepted lineage
  representatives.

The trust-region decision adds no correction iteration: one raw HONF call and,
when enabled, one corrected HONF call are the maximum per lineage. Proposal-only,
accepted-population, and raw success metrics are all retained before final
ranking, so reranking is never presented as generator success.
