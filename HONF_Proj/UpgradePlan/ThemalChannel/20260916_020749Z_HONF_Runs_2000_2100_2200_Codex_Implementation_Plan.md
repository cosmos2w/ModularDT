# HONF Runs 2000 / 2100 / 2200 — Codex Implementation and Evaluation Plan
## Three Goal-mode tasks for routing-index neural fields

**Read with:** `HONF_Runs_2000_2100_2200_Mathematical_Design.md`. Equation numbers refer to that document.  
**Conceptual source:** the user's `dynamic_sparse_hypergraph_routing.md`.  
**Repository:** `cosmos2w/ModularDT`, branch `agent/honf-core-next`.  
**Planning-time head:** `9ae17f1eed259976336e4ad2359d95014440783f`. A newer legitimate head is not a reason to reset or discard work. Inspect relevant differences and retain them.

---

## 0. Objective, scope, and execution authority

Implement three separately selectable routing strategies over one shared fine-response backend:

- Run **2000**: module-induced hubs;
- Run **2100**: three-step fixed-data mean-shift hubs;
- Run **2200**: a finite 64-entry routing dictionary.

The common scientific objective is to replace repeated QM and QE evaluation with a lightweight two-hop index followed by **delayed evaluation of unique selected fine pairs**. Hubs are routing descriptors, not a second field-value representation. Dense's fine contextual source preparation and existing physical loop remain.

Each Goal-mode task may train **one fresh managed run through at most epoch 500**. It may execute ordinary focused tests, bounded disposable optimizer checks, and the evaluation specified here. It may not launch width/temperature/seed sweeps, parent continuation, another architecture, or a continuation beyond 500. A's or B's early accuracy is not a software prerequisite for developing the other strategies.

Run 1401 is the classic HONF reference. Run 1804 is the strong dense reference. Existing 1801/1805/1806/1807/1808 results remain useful context, not extra training tasks.

### 0.1 Research criteria are not approval machinery

“Evaluation gates” in this document mean recorded scientific questions, numerical checks, and continuation recommendations. Do not build a gate service, status machine, policy engine, experiment approval framework, contract freeze, cryptographic digest scheme, new baseline snapshot, or extra provenance system.

Use ordinary Git history, the existing run allocator, named configuration fields, trusted loading, types, and standard tests. No concrete scenario here requires those primitives to be replaced. Preserve all existing security and checkpoint/data integrity behavior; do not remove old safeguards because this plan does not add new ones.

Blocking authorization checks belong at real boundaries: overwriting or deleting data, changing external systems, permission/security changes, production releases, or exceeding the user's training authorization. Ordinary heuristic observations such as a dense support histogram are not blocking gates. Run actual code and measure it. Invalid arithmetic or an actual OOM is a real execution failure to diagnose, not a reason to invent a synthetic pass.

### 0.2 Historical claims and new design decisions

Read the mathematical document's Section 0 before coding. In particular:

- do not label Run 1401 as lacking fine pair interactions;
- do not assert that early pooling is the proven universal cause of previous errors;
- do not treat a sparse matrix as a measured speed gain;
- do not call the dictionary infinite or free when inactive;
- do not use hard Top-C or discrete mean-shift merging in the formal profiles;
- do not replace the derived occupancy-measure query sparsemax by ordinary unweighted sparsemax;
- do not claim physical barriers are validated on the current ThermalChannel data;
- do not use straight-through gradients or soft/hard training differences.

---

## 1. Repository read map and separation of responsibilities

### 1.1 Maintained generic source to read

Read the relevant definitions and actual call sites, not just README descriptions:

| Current path under `HONF_Proj/` | Why it matters |
|---|---|
| `src/honf_forward_core/config.py` | `UnifiedForwardConfig`, `InterfaceFieldConfig`, `BatchData`, architecture parsing and legacy compatibility |
| `src/honf_forward_core/interface_fields/core.py` | shared encoder/factory, prepared state, receiver chunking, common coarse/local combination |
| `src/honf_forward_core/interface_fields/types.py` | encoded/prepared/read tensor containers |
| `src/honf_forward_core/interface_fields/dense_pairwise.py` | simultaneous MM/ME/EM, fine environmental update, direct QM, QE geometry/attention |
| `src/honf_forward_core/interface_fields/common.py` | prepared attention projections, coarse/local paths and field head |
| `src/honf_forward_core/decoding/pairwise.py` | historical fused/gathered semantics; useful reference but not a new backend to modify |
| `src/honf_forward_core/organizer.py`, `decoder.py` | preserve 1401 and all legacy modes |
| `src/config_core/forward/dense_pairwise_interface_context.json` | common scientific/training settings |
| `src/config_core/forward/schema.json`, `profile_registry.json` | configuration discovery and ordinary validation |
| `src/honf_runtime/` | use existing config/run/checkpoint/training/diagnostic utilities; locate owners via imports |

Also inspect current Stage-7 decoder-context ablations at head 9ae17f1, and the merged WindFarm paths. Do not overwrite these unrelated developments.

### 1.2 ThermalChannel-specific source to read

| Current path | Responsibility to preserve |
|---|---|
| `Case_ThermalChannel/src/channelthermal/input_adapter.py` | raw physical attributes, module/global feature schemas |
| `Case_ThermalChannel/src/channelthermal/environment.py` | known rectangular boundary descriptors and environment grids |
| `Case_ThermalChannel/src/channelthermal/interface_field_coupling.py` | P0/P1/P2 role scopes, local-operator response refresh, physical query placement |
| `Case_ThermalChannel/src/channelthermal/local_coupling.py` | physical port and flux semantics |
| `Case_ThermalChannel/src/channelthermal/model.py` and plugin/factory call sites | selection of generic core and case adapter |
| `Case_ThermalChannel/src/channelthermal/evaluation/` and `evaluation_tools/` | canonical physical metrics, masks and plots |
| `tools/diagnostics/run_stage3_interface_study.py` | standard loading, execution profiling and physical-reference request helpers |
| `tools/diagnostics/run_regional_response_study.py`, `run_honf_nstage2.py` | role-scoped interventions and prepared-query diagnostics |

If a path has moved, follow its imports. Do not create a duplicate implementation because a guessed filename is absent.

### 1.3 Reports to read

Use `docs/reports/HONF_NStage2_Comparison_Report.md`, the five-model maturity report, regional-response report, and group-reader recovery report. Generated tables/checkpoints referenced by those reports may be local-only. Reuse them in place; unavailable files remain explicitly unavailable.

The mature NStage2 evidence is a warning about overhead: exact A is worse and slower than Regional; the selected A score is competitive but not a measured speed win. This does not prescribe another tree optimization in the current tasks.

---

## 2. New generic architecture: one backend, three router plug-ins

### 2.1 Architecture/config identity

Add exactly one new public architecture:

```text
forward_architecture = "routed_pairwise_honf"
```

Its router selector is

```text
interface_model.routing.strategy = "module_hubs" | "mean_shift" | "dictionary"
```

Do not add these strategies as new historical organizer modes. They replace continuous reads in the nonlegacy interface family and do not implement legacy `hyper_state` output contracts.

Recommended source layout:

```text
src/honf_forward_core/interface_fields/
    routed_pairwise.py
    routing_index/
        __init__.py
        types.py
        geometry.py
        sparse_projection.py
        router.py
        candidates.py
        pair_join.py
        fine_read.py
```

This is a suggested division, not a file-count target. Use fewer files if it improves readability. Separate candidate generators, shared sparse algebra, and fine execution clearly. Do not create three copies of the same decoder or trainer.

### 2.2 Parameter ownership

Use one `RoutedPairwiseField` inheriting or composing Dense's public preparation/read primitives. Prefer retaining inherited learned module names for MM/ME/EM, `module_update`, `env_update`, `query_module_message`, `query_module_output`, and environmental attention. Put all new trainable routing modules under `core.backend.router.*`.

- 2000/2100 use shared source/receiver projections and the shared propensity head.
- 2100 adds numerical iterations, not edge-specific parameters.
- 2200 additionally owns `router.candidates.dictionary_features` and `dictionary_anchor_logits`.

Old modes must not instantiate these modules. Historical state-dict keys, parameter registration order, optional defaults, and forward calculations remain as before. Loading a new model with an old checkpoint is not automatically supported; historical checkpoints still load into their own original architecture.

### 2.3 Prepared data structures

Plain dataclasses or small typed mappings are sufficient:

```text
RoutingCandidates
    coords[B,K,d]
    descriptors[B,K,D]
    propensity[B,K]
    valid[B,K]
    candidate_origin (module index or dictionary index; metadata only)

TypedSourceIncidence
    source_weights[B,N]        # omega
    membership[B,N,K]          # A, scalar only; acceptable preparation storage
    hub_measure[B,K]           # mu
    hub_to_source_indices      # inverted positive incidence lists
    membership_values         # live, not detached

PreparedRoutingIndex
    candidates
    module_incidence
    environment_incidence
    optional small strategy diagnostics

PackedPairs
    batch_index[I]
    receiver_index[I]
    source_index[I]
    prior[I]                  # coalesced Pi, live gradient
    raw_path_count
    unique_pair_count
```

The prepared field continues to retain separate fine contextual module and environment states. It may cache fine environment projections from the current physical graph. Never cache a trainable tensor globally across iterations. Candidate geometry/encodings that depend on live design variables must also refresh when those variables change.

Detailed Pi/alpha exports are optional. Never return all `[B,Q,N]` matrices from ordinary full-field inference.

---

## 3. Common implementation tasks, owned by Goal 1

### 3.1 Sparse projections

Implement ordinary masked source sparsemax and the source-measure query projection from mathematical Equations (7)–(11).

Query API should return density d, probability alpha=mu*d, and small support summaries. The production two-hop join consumes d and A directly; it must not reconstruct `A/mu` and divide tiny columns.

Sort occupied logits and calculate weighted prefix thresholds on-device. Handle invalid/zero-mass padded coordinates without `0*(-inf)`, NaN prefix sums, or indexing an empty active set. Each nonempty typed source measure sums to one. The M=0 case has no module projection; the environment route uses its defined background candidate for module-seeded strategies.

A tiny scalar float64 reference should be available in tests. Production may use FP32 threshold arithmetic and stable scalar higher precision where demonstrated necessary. Do not make all H-wide neural tensors FP64 and do not silently clamp positive memberships or source measures to an epsilon.

Do not implement a hand-written detached threshold unless its analytical backward includes derivatives through both logits and mu and is verified. Ordinary autograd through sorted valid prefixes is preferable initially. At exact support boundaries, document the selected subgradient and compare one-sided derivatives rather than requiring a nonexistent unique derivative.

The source sparsemax singleton Jacobian is zero. This is not a broken backward; it must be visible in the diagnostics. No rescue curriculum or anti-collapse loss is part of these goals.

### 3.2 Descriptor and candidate calculations

Use mathematical Equation (1), D=32, the stable norm map, and a bounded propensity head. Reuse existing receiver Fourier features. No H-wide neural network is evaluated separately for every query–hub pair: query/source descriptors are projected once and scalar bilinear/geometric affinities are broadcast or matrix-multiplied.

The case geometry provider supplies low-dimensional point descriptors. Its local feature sampling must not turn into dense query–environment attention. No field target or solver answer is accessible to the router.

All candidate membership is recomputed each preparation. Exact zero usage may be compacted for the current query execution, but no parameter or future candidate is permanently deleted.

### 3.3 Pair join before fine instantiation

Build inverted hub-to-source lists during preparation, then join positive query densities to positive source memberships on GPU. Use live path weights `omega * A * d`.

Coalesce duplicate `(batch, receiver, source)` keys before fine evaluation. Use ordinary integer keys and tensor indexing; do not introduce a new cryptographic identifier system. Standard sort/unique/scatter or a clear CSR-style implementation is appropriate.

The production algorithm must have:

```text
route -> enumerate paths -> coalesce pairs -> fine network -> weighted reduction
```

not

```text
fine network over every pair -> multiply by sparse mask
```

The dense test oracle may materialize scalar Pi for small shapes. Production must not construct a query-by-hub-by-source tensor. No cap may drop valid positive pairs. Report raw path multiplicity, unique pair counts, source occupancy, and dispatch time. A small hub count is not itself an efficiency result.

A loop over receiver or fixed-size neural tiles is acceptable. Python loops over individual receivers, modules, or every active hub, especially with `.item()`/`bool(tensor)` in each iteration, are not an adequate GPU implementation. No custom Triton/CUDA kernel is mandated in the first goal; profile first and make one bounded execution correction only when the trace identifies a real bottleneck.

### 3.4 Fine QM executor

Gather contextual module states, relative coordinates, and global tokens for unique positive pairs only. Call the inherited `query_module_message` with exactly its current input ordering and existing Fourier scaling. Reduce with `M/(1+M) * Pi`, then apply the inherited output projection.

Respect the output bias in the empty-module and dense-limit comparisons. Padded modules are never selected. Distinguish saved work from removing padding versus removing active physical pairs; both must be reported.

### 3.5 Fine QE executor

Retain every contextual fine environmental source. Project K/V once per prepared graph. The routed environmental prior Pi already contains quadrature; do not apply an extra `log(nu)` or multiply by nu a second time.

For selected QE pairs only:

1. calculate the original relative Fourier features and geometry-bias MLP;
2. gather the appropriate per-head query/key and calculate content scores;
3. apply stable positive-prior attention normalization per receiver/head;
4. gather and reduce original fine values;
5. apply the inherited attention output projection.

Do not pass hub descriptors as values. Do not first pool projected K/V into hubs. Pair priors are shared across heads; fine content/geometry attention remains head-specific.

Use tiles for H-wide gathered values and micro-MLP activations. A suggested neural-row tile is 16,384, a memory execution parameter only, with no truncation. Activation checkpointing follows the existing flag. Scalar route/prior graphs must remain live when neural tiles are recomputed in backward.

### 3.6 Factory and physical integration

Add one conditional factory branch in `InterfaceFieldCore`. The existing common context and final head are unchanged. Preserve old `forward_architecture` behavior, including the newly added Stage-7 decoder ablations and the merged WindFarm model paths.

The new backend's `prepare` returns fine states plus the routing index. Its `read` supplies the sum of fine routed QM and QE contexts and cheap summaries. Existing P0/P1/P2 role handling stays in the case wrapper. A fresh routing index is required for each refreshed physical source state.

Thread geometry information through optional fields or an explicit runtime provider, using the simplest extension consistent with existing dataclasses. Defaults must be None for old paths. Do not serialize executable callbacks or case objects in model weights; reconstruct them through the maintained case adapter at load/runtime.

---

## 4. Case-specific work: ThermalChannel only where physically necessary

### 4.1 Geometry provider

Add a small `ChannelThermalRoutingGeometry` helper under the case package, or extend the existing environment helper without changing historical calls. It provides:

- point descriptors using the same known channel-wall/inlet/outlet/centreline conventions;
- the actual case bounds;
- routing scale `4 * module_radius`;
- a neutral resistance for the current formal dataset;
- optional support for a supplied resistance field in explicit numerical fixtures.

The generic core may know d, coordinates, measures, and relation names. It may not assume x is streamwise, use a 2-D cylinder formula, interpret a material column as adiabatic, or hardcode radius 0.45.

The adapter's coordinate normalization scale and the physical domain extent are different concepts. Preserve the established reference quadrature in formal comparisons. Do not silently “fix” its total volume for only the new candidates. Record the convention and allow future adapters to supply physical measures explicitly.

### 4.2 Barrier handling

The current channel samples do not justify inventing internal thin-wall labels. Do not learn an unsupervised `rho_env` and describe it as validated resistance. Do not penalize every source-module segment because its endpoint starts inside that module's own solid body. If future data provides barriers, the case provider must declare relation type, endpoint treatment, and whether the quantity is a finite resistance or certified disconnection.

One artificial barrier fixture can check the Equation (2) sampler, coordinate gradients, and reversal/type semantics. It is **not** a physical reconstruction benchmark and should not enter the 90-case accuracy tables.

### 4.3 Physical outputs and data policy

No changes to port variables, frozen Stage A, number of refinements, interface/flux heads, dataset targets, losses, normalization, or the 600/90 split. The current `test` split is the repeatedly used **development holdout**. Do not call it an untouched test set.

P0 is the initial port read. P1-only preserves normal P0 and modifies feedback. P2 is final field reading and cannot retroactively change completed internal/port/interface states. Extend existing role-scoped tools rather than introducing broad decoder hooks with ambiguous scope.

Do not modify inverse generation, module-count decisions, or WindFarm training in this round. Generic small 3-D tensor tests establish dimension-neutral implementation, not 3-D physical accuracy.

---

## 5. Configuration templates

Create three complete candidate profiles alongside the existing Dense template:

```text
src/config_core/forward/routing_module_hubs_context.json
src/config_core/forward/routing_mean_shift_context.json
src/config_core/forward/routing_dictionary_context.json
```

Copy the **scientific values** from the maintained Dense profile and actual resolved Run-1804 settings. Its source profile still carries historical run ID 1800; the successful baseline is 1804. Do not propagate 1800 into the new candidates.

Add a nested optional `RoutingIndexConfig` to the generic configuration and ordinary JSON schema. No new fields need to appear as active scientific settings in historical profiles. Avoid expanding old serialized defaults in a way that breaks existing comparisons/checkpoint reconstruction.

Suggested explicit new block:

```json
"routing": {
  "strategy": "module_hubs",
  "descriptor_dim": 32,
  "router_hidden_dim": 64,
  "source_normalizer": "sparsemax",
  "query_normalizer": "source_measure_sparsemax",
  "temperature": 1.0,
  "content_scale": 2.0,
  "geometry_scale": 0.25,
  "propensity_scale": 0.25,
  "resistance_mode": "adapter",
  "execution": "gathered",
  "fine_pair_chunk_size": 16384
}
```

The geometry provider supplies the actual length scale and returns zero additional resistance for the initial ThermalChannel dataset. Ordinary dataclass validation may reject invalid dimensions/temperatures. These are conventional invalid-input checks, not a research gate layer.

Run 2100 adds:

```json
"strategy": "mean_shift",
"mean_shift_steps": 3,
"mean_shift_feature_bandwidth": 1.0
```

Run 2200 adds:

```json
"strategy": "dictionary",
"dictionary_size": 64
```

There is no Top-C, learned count, residual-stop threshold, gate warmup, balancing penalty, or entropy target. Avoid filling a profile with dozens of unused future options.

Shared scientific settings:

| Setting | Value |
|---|---|
| H / message width / fine attention heads | 256 / 128 / 4 |
| environment grid | inherited 24 × 8 |
| common coarse tokens / blocks | 8 / 1 |
| common coarse source | unchanged `module_states` |
| local radius factor | 2.5 |
| receiver chunk during training/accuracy | 128 |
| activation checkpointing | true |
| seed | 0 |
| AdamW learning rate / weight decay | 3e-4 / 1e-5 |
| clipping | inherited norm 1 |
| AMP / dropout | false / 0 |
| ports / refinement | predicted; no curriculum; one refinement |
| train batch / sampled queries | inherited 48 / 1024 |
| authorized profile default epochs | 500 |
| milestones | 10, 50, 100, 250, 500, 1000, 2500, 5000 |
| initialization / resume | neither for initial launches |

Milestones beyond 500 support only a future user-authorized continuation. Register all three as research candidates, not the recommended forward profile.

---

## 6. Shared mathematical tests: small, purposeful, executable

Use one compact numerical test module and normal integration tests. Reuse the tests across strategies rather than repeating a large suite for every task.

### 6.1 Projection and routing tests

1. Ordinary source sparsemax: simplex sum, exact zeros, shift invariance, padding, support-dependent Jacobian, singleton zero derivative, and tied logits away from a forced hard Top-C operation.
2. Measure-aware query projection: weighted KKT residual, dense-limit `d=1`, gradients through both z and mu, zero hub measure, all remaining mass in one hub, and finite tiny positive masses.
3. Effective Pi row mass is one for nonempty sources; compare packed pair weights with `omega * (d @ A.T)` on small tensors.
4. Duplicate hub paths coalesce correctly and the fine network is called once per unique pair.
5. Split-source quadrature duplication is invariant. Hub splitting with identical logits and split A preserves Pi **conditional on those supplied incidences**; ordinary source sparsemax after duplicating raw candidates is not asserted invariant.
6. Empty last-hub contribution vanishes continuously in the mathematical operator. Evaluate one-sided differences at support events, not an arbitrary single central gradcheck across nondifferentiability.
7. Gradient-check the full source-incidence -> mu -> query-density -> Pi composition in float64 away from ties. Include a test that would fail if mu or selected weights were detached.

### 6.2 Fine reader tests

8. With fixed uniform typed priors, compare the new QM/QE reads with Dense at identical weights; compare source, receiver, input and parameter gradients as well as values. Do not call arbitrary full-positive learned priors “Dense parity.”
9. With nonuniform sparse priors, compare gathered micro-execution to a **dense arithmetic oracle using those same priors**. Test one-shot/chunk/tile versions, including repeated pair paths.
10. Verify counters show no micro-network evaluation for omitted pairs; sparse geometry scoring must also occur after selection in QE.
11. Verify no field-value dependency on hub descriptors when effective Pi and fine source states are held fixed.
12. Retained environmental priors include nu exactly once; split quadrature duplication and unequal masses exercise this.

### 6.3 Integration tests

13. Module permutation, batch composition, trailing-padding invariance, and query-chunk invariance. Query counts may change K-query-union diagnostics, not predictions at an unchanged query.
14. P0/P1/P2 refreshes retain gradients and do not reuse stale fine projections/routing.
15. Construct small 2-D and 3-D generic cases; preserve optional adapter fields through device movement and input conversion.
16. Old forward families and actual available 1401/1804 checkpoints load through their original paths. Reuse existing golden tests; do not create new snapshots or hashes.

Use ordinary numerical tolerances appropriate to float64 references and FP32 repeated execution. Keep differences and same-chunk controls when elementwise tolerances fail; do not loosen historical tests to turn a new implementation into a pass.

### 6.4 Actual GPU optimizer check

For each candidate, execute one disposable canonical physical optimizer step in a small-M batch and one in the largest ordinary M=12 bucket on its assigned GPU. Use the real predicted-port loop, loss builder, AdamW, clipping, and frozen Stage A. Do not save or reuse these weights in the formal run.

Record preclip total and component norms with FP64 norm reduction, clipping scale, finite parameters, optimizer update norms, timing, and allocated/reserved peaks. This protects against the previous situation where finite scalar losses concealed enormous gradients and effectively stalled training. Nonzero router gradients are not required for an artificially singleton route in every row; interpret the Jacobian and report actual source/query support fractions.

A real OOM may require smaller **execution tiles/checkpointed intermediates**, not a silent change in training batch, physical losses, sparse support, or dataset. Repeat only the failed batch on a new disposable instance and retain the failure description.

---

## 7. Goal 1 — Shared infrastructure and Run 2000

### Concrete task order

1. Read the source proposal, both new plan documents, and the current repository implementation.
2. Implement the common routed backend, candidate/type containers, sparse projections, pair join, fine readers, configuration path, and case geometry provider.
3. Implement module-induced candidates (mathematical Section 7). No explicit ReLU hub prefilter and no exact topological K target.
4. Add ordinary targeted tests and role-aware diagnostics. Execute them and the two real optimizer checks.
5. Run a bounded unprofiled inference measurement plus one short CPU/CUDA trace on a real case and the larger synthetic shape. The trace is diagnostic, not a pass substitute or a demand for a custom kernel before training.
6. Commit coherent source and configuration through normal Git. Launch Run 2000 once from scratch to 500 epochs.
7. Observe early supports and finite preclip/update behavior through ordinary run logs. No background agent infrastructure is required. Use an existing low-cost monitor only if already available, for passive summaries; it must not tune or restart experiments.
8. Complete the 500-epoch evaluation in Section 10 and write the report. End this goal; do not launch 2100/2200 or continuation automatically.

A sparse-matrix histogram that is still dense is a scientific observation, not permission to force a lower budget halfway through the run. A failed scientific result can still leave usable shared software for the next two strategies.

### Launch surface

From the actual `HONF_Proj` root, after using existing configuration resolution to confirm the intended values:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py \
  --config project://src/config_core/forward/routing_module_hubs_context.json \
  --workflow forward --device cuda:0 --epochs 500 \
  --run-id 2000 --run-name routed_module_hubs --yes
```

The interpreter location is the established workstation example. Use the actual maintained interpreter on another host and report the executed command. An optional normal `--dry-run` may inspect resolution but does not replace execution. Do not prepend a required third-party command wrapper.

---

## 8. Goal 2 — Run 2100, mean-shift candidates

Use the shared executor delivered by Goal 1. Do not wait for a favorable Run-2000 score as a mathematical prerequisite; the question is whether candidate adaptation helps.

1. Implement exactly three fixed-data mean-shift iterations. Keep the per-pass source coordinates/descriptors live but fixed across the iterations. Recompute candidate-to-source resistance only where supplied.
2. Return all M candidate trajectories/final centres as routing candidates. Do not merge, round, cluster, or deduplicate their identities in training or inference.
3. Add tests for T=0 equivalence to module-seeded positions/descriptors, source permutation equivariance, finite multi-step gradients, and synthetic separated-mode versus broad-bandwidth consensus behaviour. The consensus case should be reported as possible, not excluded by an assertion that the method must always find multiple modes.
4. Reuse shared projection/executor/physical tests and run two disposable real optimizer checks.
5. Launch Run 2100 once from scratch through 500. No initialization from Run 2000.
6. Evaluate through the same schema. Add drift distance, candidate separation, iteration movement, descriptor concentration, and candidate-generation time. A diagnostic approximate mode count has an explicit tolerance and does not control execution.
7. Write the comparison report and stop. No automatic bandwidth/T sweep, alternative merging rule, or continuation.

Example launch:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:Case_ThermalChannel/src \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py \
  --config project://src/config_core/forward/routing_mean_shift_context.json \
  --workflow forward --device cuda:0 --epochs 500 \
  --run-id 2100 --run-name routed_mean_shift --yes
```

Physical GPU 2 is logical `cuda:0` after masking. Do not use logical `cuda:2` in that process.

---

## 9. Goal 3 — Run 2200, finite routing dictionary

Use the same prepared source states, pair compiler, readers, and losses. Do not turn this task into an expert-network capacity experiment.

1. Add 64 trainable D=32 dictionary descriptors and trainable normalized-domain anchors. Initialize finite interior positions and distinct descriptors with seed 0. No massive candidate allocation is required to test the strategy.
2. Use the same propensity, source sparsemax, and source-measure query projection. No hard Top-C operation is in the canonical path. Do not instantiate per-dictionary-entry fine MLPs.
3. Recompute all source-to-candidate logits per preparation; compact only exact unused current sources for read-time execution. Empty entries remain parameter candidates in later passes.
4. Test module order invariance, dictionary permutation equivariance when parameters and metadata are permuted together, empty entries, finite anchor-coordinate gradients, and common shared executor behaviour.
5. Execute two disposable real optimizer checks, then launch Run 2200 through 500 on available physical GPU 0 by default. Do not oversubscribe GPU 0 with a still-running 2000 job; ordinary scheduling is sufficient.
6. Report Kmax, Ksource, query fan-out, true per-hub module/environment occupancy, duplicate-path expansion, dictionary parameter/optimizer memory, task gradients, and route turnover. Do not call dormant entries free or count them as deleted.
7. Complete the three-candidate 500 comparison using existing artifacts; no new 2000/2100 training or rerun is needed. End the goal.

Example launch:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py \
  --config project://src/config_core/forward/routing_dictionary_context.json \
  --workflow forward --device cuda:0 --epochs 500 \
  --run-id 2200 --run-name routed_dictionary --yes
```

Run IDs are explicit requests, not permission to overwrite occupied runs. The existing allocator handles collisions. Report an occupied ID rather than silently deleting or repurposing its directory.

---

## 10. One common evaluation protocol at each 500 endpoint

### 10.1 Accuracy and convergence

Run the standard complete-grid evaluation on all 90 development cases for the exact epoch-500 checkpoint. Use predicted ports, checkpoint-owned normalization, training/accuracy receiver chunk 128, and the existing masks/KPIs. Save debug arrays only for 0273, 0653, 0283, 0298, and 0302.

Reuse existing exact-500 1401/1804 scalar tables. Contextual parent results may include 1801/1805/1806 when already available. Do not compare a new 500 checkpoint against a baseline trained through 5000 as though budgets matched.

Evaluate the candidate's saved-best-by-validation-field through 500 separately if it differs from the exact endpoint. Record its actual epoch. If matched parent best-through-500 weights are unavailable, do not substitute their best-through-5000 files. Sampled validation curves and full-grid fluid L2 have different denominators/definitions.

Report:

- pooled fluid SSE/count/target energy, MSE and normalized relative L2;
- equal-case mean/median/p95/worst and paired wins/deltas;
- each field channel in normalized and physical units;
- near-interface/far-fluid scores;
- internal/surface temperature, flux and provisional/final port errors;
- pressure-drop, outlet-temperature, and active-module-temperature KPIs;
- established module-count/spacing/wall/heating strata;
- exact milestone and trailing-50/100 learning summaries;
- preclip/update support metrics with missing observations distinguished from zeros.

Numerical anchors from the existing exact-500 tables: Legacy L2 about 0.117148; Dense 0.098741; Regional 0.096652. These are reference observations, **not blocking thresholds**. No 500-epoch declaration of final convergence follows from a small difference.

### 10.2 Routing usefulness, not just activity

Use the five fixed anchors for a bounded set of frozen interventions:

1. **Uniform routing at P2:** replace both effective priors with their source measures and execute all valid fine pairs at the same trained weights. This is a routing removal/dense-support intervention, not the trained Dense baseline.
2. **Routed QM zero at P2**, and **routed QE zero at P2**, separately. Keep common coarse/local contexts.
3. **Uniform routing at P0** and **P1-only**, recomputing downstream physical effects. Normal P0 remains for P1-only.
4. One **common coarse removal at P2** to measure how much the retained bypass contributes. It is not a sole-bus test.

Retain prediction discrepancies and intervened-minus-normal **ground-truth error** separately. Include exceptions and tail cases. P2 cannot change completed module physics except ordinary independent-repeat numerical differences.

Where both typed priors are uniform, the arithmetic Dense reference is useful. A large discrepancy from uniform routing indicates routing dependence; it does not prove correct physical sparsity. A uniform intervention improving error suggests harmful learned routing at that checkpoint.

### 10.3 Sparse execution and missed-source audit

Report counts by P0/P1/P2 and source type:

- active physical source count versus padded width;
- candidate K, source-active K, query-active K;
- source incidence nnz and true Dk occupancy distribution;
- unique fine pair count and dense active-pair reference;
- raw path count before coalescing and duplicate expansion ratio;
- singleton/zero membership probabilities and route turnover;
- hub-generation, route-score/projection, join/dedup, fine-kernel and reduction costs.

On two anchors with 32 fixed queries, additionally compute the omitted fine sources in an **evaluation-only audit**, without changing normal prediction. Compare normal sparse output with the same-weight uniform/full-support result. For the environmental route also report the full Dense-attention probability mass on sources omitted by the router, separately from field error. For the module route report omitted message norms with explicit acknowledgement that vector cancellation and nonlinear heads prevent treating norm mass as physical importance.

This small audit distinguishes “sparse and useful” from “sparse by missing relevant information.” It is not used as a training target or a fourth model.

### 10.4 Gradients and support changes

Use ordinary anchor 0273 and difficult anchor 0298:

- fixed-source routing gradcheck/JVP to isolate z/A/mu/Pi;
- direct fine-source perturbation with routing and other contextual states fixed: omitted fine pairs have zero direct effect;
- full model perturbation: allow Dense preparation, global/local paths, and physical coupling to change, so total module influence need not vanish;
- position and heat-attribute AD/FD at the same prescribed physical step sizes, showing signed values and error in the proper normalized/physical units;
- construct one sparsemax support-entry/exit example and record actual support identities, one-sided derivatives, and continuity;
- for 2100, include derivatives through all mean-shift iterations;
- for 2200, include finite gradients of active anchors and explicitly report zero-task-gradient inactive entries.

When finite differences are cancellation-limited, use a small float64 numerical subgraph reference to understand the issue; do not secretly tune a full-model step until it appears to pass. Model-versus-itself AD/FD is not a physical gradient certificate.

No inverse generative run is launched. Discrete module insertion/deletion remains outside this authorization.

### 10.5 Physics-aware evidence

Run one artificial barrier sampler fixture and one neutral-resistance identity test as unit/numerical checks. Formal ThermalChannel results must state `barrier benefit: Evidence Missing` unless actual appropriate solver-labelled layouts are available.

The 16 established independent pair/triple reference requests remain in place. Ingest trustworthy responses if already supplied, but do not launch a new solver-development project or use surrogate predictions as reference truth. Do not reinterpret correlations in learned route maps as higher-order physical validation.

---

## 11. Measured compute/memory ledger

### 11.1 Required workloads

Use the established real anchors 0273 and 0653 with 8,192 queries, plus:

```text
(M,E,Q) = (32,768,65536)
(M,E,Q) = (128,3072,262144)
```

Generate actual layouts, descriptors, memberships, and source occupancy. Never synthesize conveniently peaked attention weights for a “model scaling” claim. These large shapes are execution-only until reference fields exist.

Use one common physical GPU for model comparisons, process-isolated model measurements when needed, receiver chunk 2048 for the main inference ledger, outer query batch 32768, detailed maps off. Keep the configured chunk 128 for scientific accuracy/training and show its cost separately on one anchor. Do not mix chunk gains with architecture gains.

Measure 2 warmups/5 repetitions on real anchors and 1 warmup/3 repetitions on large shapes initially. Report median and repeat range, not a confidence interval inferred from a few repetitions.

### 11.2 Timing scopes and allocations

Measure separately:

- encode plus layout/geometry;
- complete physical preparation plus one query;
- prepared field decoding;
- full physical forward;
- a disposable canonical forward/backward/update step;
- live, incremental and total peak allocated memory and reserved memory.

Separately measured phase medians overlap and need not sum to total forward. Release unrelated retained preparations/models between measurements. Parameter count, checkpoint size, optimizer storage, active fine-pair count, and peak activation allocation answer different questions and remain separate.

Record actual M versus Mpack. Delayed execution removes padding automatically; include an active-only dense fine-kernel timing reference on small shapes so that all gain is not attributed to learned physical sparsity.

### 11.3 Profile actual execution

Capture one short CPU/CUDA profiler trace for a real anchor and one representative large forward, including routing scores, sparse projections, two-hop joins, deduplication, pair networks, weighted attention, and reductions. Use an existing maintained profiler if suitable. Profiled measurements are for attribution; unprofiled synchronized repeats supply latency conclusions.

Look for host/device synchronization, per-query Python work, expanded H-wide gathers, duplicate evaluation, sort/unique costs, and expensive optional diagnostics. This directly addresses the Run-1807 failure mode. Do not promise speed based solely on I_M/I_E counts.

One bounded implementation correction that preserves the mathematical operator is allowed when the trace identifies concrete waste. Remeasure and retain the before/after scope. Do not add a new kernel language/runtime, compressed field branch, hard truncation rule, or hidden dense fallback as an unreported “optimization.”

A source-measure sparsemax can legitimately select all sources. In that case a mathematically identical dense execution fast path may be documented later, but it is not a license to omit routing overhead from the measured end-to-end result. Do not implement shape-dependent scientific models under the same config.

---

## 12. Visualization and diagnostic organization

Use existing field/physical plotting utilities. Add one shared routing visualization, not one bespoke package per run.

For each of the five anchors, show:

- physical module positions, environmental sample geometry, and candidate hub positions;
- source incidences for M and E separately;
- a selected port and far-field receiver, its active hubs, and actual unique fine source pairs;
- both query–hub probabilities and final query–source priors;
- actual Dk occupancy and path-duplication counts;
- normal prediction/reference/error with common masks/scales;
- support turnover or a support-transition trace;
- for 2100, candidate trajectories;
- for 2200, available versus active dictionary entries.

Label route weights as learned routing, not physical influence. Use separately measured JVP/intervention maps for influence. A sorted block-looking incidence matrix is not a topology-quality target.

Store only selected detailed arrays. Do not dump every epoch's Q-by-source matrices or duplicate baseline field arrays.

---

## 13. Continuation assessment at epoch 500

Write an evidence-based recommendation for each candidate. These are research judgements, not an automated blocking threshold framework.

**Promising:** the real physical model trains with finite useful updates, accuracy is competitive or improving plausibly, selected fine work is genuinely reduced, and measured resource savings are useful without severe thermal/tail regression.

**Unresolved:** learning remains healthy but slower; accuracy and sparsity trade off; route construction dominates on small cases; or a promising result lacks enough later convergence. Recommend whether a user-authorized same-run extension would resolve a specific uncertainty.

**Unpromising:** persistent catastrophic error or stalled updates, dense effective pairs with added overhead, excessive source omission with harmful physical errors, or hard support artifacts that invalidate useful inverse gradients. Diagnose rather than automatically changing the model mid-run.

No fixed “within 15% at 500” rule determines continuation. The previous Dense/Regional maturity reversals require both exact endpoint and trajectory context. Conversely, finite scalar losses do not excuse a recorded gradient explosion or effectively frozen optimizer.

Specific interpretation traps:

- K=1 with uniform Pi may be accurate Dense-like computation and an efficiency failure, not physical-rank discovery.
- Larger K or prettier blocks are not improvements.
- Fewer high-width pairs with a slower total forward is an unsuccessful acceleration claim.
- Large branch-removal effects do not show better representation than Dense.
- Exact zeros and self-consistent AD do not prove correct omitted physical influence.
- A finite dictionary using few entries is not literal unbounded topology birth/death.

The goal ends after reporting. Supply an actual, unexecuted resume command for a proposed continuation; the user decides whether to extend to 2500 or 5000.

---

## 14. Outputs, reports, and hygiene

Keep three generated study subdirectories under one existing-style root, for example:

```text
diagnostics/generated/interface_operator_study/dynamic_sparse_routing/
    run2000/
    run2100/
    run2200/
    comparison/
```

Use the existing allocator/manifests for formal runs. Each run subdirectory can contain `numerics`, `physical_smoke`, `endpoint500`, `interventions`, `routing`, `timing`, and `figures` as needed. Do not create duplicate `final/final2/recheck/new` trees or copy checkpoints into reports. Keep failed execution logs distinct from final measurements.

Maintain only reusable code and concise reports in Git. Respect existing local-only/ignored output rules. Preserve existing security, old reports, and old model arithmetic.

Reports:

```text
docs/reports/HONF_Routing_Run2000_Epoch500_Report.md
docs/reports/HONF_Routing_Run2100_Epoch500_Report.md
docs/reports/HONF_Routing_Run2200_Epoch500_Report.md
docs/reports/HONF_Dynamic_Sparse_Routing_Comparison_Report.md
```

After Goal 1/2, the comparison may explicitly contain pending rows. Goal 3 completes it from existing results; it must not silently rerun old model training.

Required report sections:

1. actual code/config differences and any deviation from these documents;
2. executed commands, run paths, actual GPU mapping and endpoint;
3. uniform-Dense and sparse dense-oracle numerical results;
4. source/candidate/query cardinalities and actual fine-pair counts;
5. gradient/update health, sparsemax singleton and support-turnover behavior;
6. complete accuracy and thermal/tail tradeoffs, exact versus separately selected;
7. phase interventions and properly scoped derivatives;
8. actual latency/memory/profile attribution, including route-discovery cost;
9. physical-reference and barrier-evidence limits;
10. continuation recommendation and unexecuted resume command.

Use existing named-column history handling. If diagnostic fields vary by strategy, a small separate named routing-metrics CSV with epoch/phase keys is preferable to appending incompatible row layouts to an existing file. This is normal data handling, not a new snapshot/versioning framework.

---

## 15. Definition of the three completed goals

**Goal 1 is complete** when common sparse routing/dispatch is executable, the module-hub strategy is integrated without historical regressions, Run 2000 completes its authorized endpoint or documents an actual execution failure, and its evidence/report/handoff exist.

**Goal 2 is complete** when the fixed-data mean-shift strategy uses that common executor, Run 2100 is assessed through at most 500, and candidate-motion benefits/costs are separated from fine-kernel changes.

**Goal 3 is complete** when the finite dictionary strategy is assessed through at most 500 and the common comparison distinguishes the three candidate generators without claiming infinite capacity, free dormant experts, or proven physical topology.

No goal auto-launches another goal or longer training. No model is promoted solely from active routing or lower average error. The research deliverable is a reproducible answer to whether routing-before-fine-instantiation preserves useful physical computation at a measured lower cost.

---

## 16. Source references

Use the bibliography [R1]–[R13] in the mathematical document. In particular, sparsemax [R1] supports source projection, mean shift [R4–R5] supports finite differentiable candidate motion, and MegaBlocks/AdaSplash/NSA/MoBA [R7–R10] motivate the execution and routing-vs-value distinctions. None proves these HONF candidates will be sparse, accurate, or faster.

Primary repository reference at planning:

`https://github.com/cosmos2w/ModularDT/tree/9ae17f1eed259976336e4ad2359d95014440783f/HONF_Proj`

Primary existing reports are the mature NStage2 and five-model studies. The attached routing document is a research proposal, not an implementation or benchmark. This coding plan intentionally preserves the gap between a mathematically well-defined candidate and empirically demonstrated success.
