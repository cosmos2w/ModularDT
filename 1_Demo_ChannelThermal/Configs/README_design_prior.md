# Mechanism Design Prior Configuration Notes

The design-prior path is now a hypergraph-centric mechanism prior plus a
conditional layout realizer:

```text
mechanism = realized hypergraph organization + field-behavior descriptor
p(D | mechanism, context)
```

The design library is a mechanism source:

```text
layout -> frozen-forward HONF -> realized hypergraph -> field behavior
```

It is not used to learn a generic valid-layout prior.

## Mechanism Parameters

- `num_clusters`: number of mechanism clusters. Start with `24`; useful sweeps
  are `12`, `24`, and `48`.
- `hypergraph_weight`: weight for normalized realized-hypergraph features.
  Useful range: `0.5` to `2.0`.
- `behavior_weight`: weight for normalized behavior descriptors. Useful range:
  `0.2` to `1.0`.
- `include_count_descriptor`: include module count as part of the mechanism
  feature. Keep enabled for count-constrained design tasks.
- `count_weight`: normalized count-descriptor weight. Start with `0.25`.
- `kmeans_iterations`: NumPy k-means iterations for unsupervised mechanism
  discovery. `100` is usually enough for the starter library size.

## Layout Realizer Parameters

- `hidden_dim`: MLP width for the conditional rectified-flow realizer.
- `condition_dim`: embedding size for mechanism plus context.
- `layout_flow_weight`: main velocity matching loss weight.
- `mask_component_weight`: extra weight on active-mask slots.
- `active_center_weight`: center-coordinate weight for true active modules.
- `inactive_center_weight`: center-coordinate weight for inactive padded slots.
- `geometry_weight`: differentiable overlap/boundary validity pressure.
- `count_weight`: penalty for matching generated active count to the mechanism
  count/target count.

## Search Parameters

- `mechanism_cem_iterations`: number of posterior CEM updates in mechanism
  feature space.
- `mechanism_cem_population`: mechanisms evaluated per CEM iteration.
- `mechanism_cem_elite_frac`: fraction used to update the Gaussian proposal.
- `mechanism_prior_weight`: penalty for moving far from the learned mechanism
  clusters. Useful range: `0.01` to `0.2`.
- `hypergraph_realization_weight`: desired-vs-realized hypergraph consistency
  weight. Useful range: `0.2` to `2.0`.
- `layouts_per_mechanism`: number of layouts sampled for each mechanism before
  forward verification. Useful range: `1` to `8`.
- `filter_mechanisms_by_count`: prefer mechanism clusters whose decoded count
  descriptor matches objective hard count constraints.
- `ranking_score_key`: choose `"internal_total_score"` to use mechanism
  regularizers during posterior search, or `"fair_objective_score"` to rank
  only by the field-functional objective.

For the first fair A-vs-E comparison, use
`inverse_targets_v2/field_functional_chip_plume_demo_fair.json` as the
objective and treat `best_objective_score` as the headline metric.
`mechanism_prior_weight` and `hypergraph_realization_weight` are search
regularizers/diagnostics; the template keeps them modest at `0.02` and `0.5`.
Raw layout CEM has no desired mechanism and is compared only by the same
objective score.

## Method Roles

Use `mechanism_prior` as a diagnostic: it samples mechanisms from the atlas and
realizes layouts without posterior guidance.

Use `mechanism_guided` as the actual hypergraph-centric generative inverse
method. It searches mechanism-feature space, realizes layouts through
`p(D | mechanism, context)`, forward-verifies every layout, and scores the
field-functional objective plus optional mechanism-realization terms.

Raw layout CEM remains the direct optimizer baseline. It has no desired
mechanism, so it should not receive a hypergraph-realization penalty.
