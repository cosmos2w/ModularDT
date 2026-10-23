# Stage 7 Model Enhancement: Sparse Context Fusion and Factorized Pairwise Execution

## Purpose

This document defines the next Stage-7 forward-model work after acceptance of Run 1401. The scope is intentionally narrow. The goal is to remove the main query-scale computational bottleneck while preserving the structured organization recovered by Stage 7, then test one targeted replacement of the expensive pairwise kernel. Only after selecting the best K=6 model should the project perform a small K-scaling audit.

This is a Goal-mode implementation plan. It defines the scientific questions, code boundaries, compatibility requirements, evaluation gates, run policy, and artifact-output rules. Codex should inspect the current implementation and make the smallest coherent set of changes that satisfies these goals rather than mechanically following pseudocode.

## 1. Frozen starting point and decision discipline

The formal Stage-7 numerical reference remains:

- Run 1401, `stage7_structured_context`;
- best-by-field checkpoint, epoch 4585;
- fixed `K=6` softmax organizer;
- raw organizer hyperedge state;
- dense context fusion;
- legacy four-layer query-module pair MLP;
- shared AdamW learning rate `3e-4`;
- current Stage-A best checkpoint and current ThermalChannel dataset/normalization.

Run 1500 is a long-run continuation experiment intended mainly to observe the 5K-to-10K training regime. It is not the baseline for this structural round, is not a warm-start source for the new model, and should not block the work below. When Run 1500 finishes it may be used as a long-budget trajectory reference, but structural decisions in this round should remain anchored to Run 1401 and matched-budget comparisons.

The central discipline for this round is:

1. Test execution sparsity without changing the learned Run-1401 representation.
2. Test one factorized pair-kernel architecture from scratch while keeping the organizer, field assembly, data, loss, and optimization policy otherwise unchanged.
3. Do not introduce organizer regularization, hard module-edge specialization, staged curricula, adaptive edge birth/death, additive field assembly, new physical priors, distillation, or warm-starting between model families.
4. Do not launch many formal runs. Most sparse-execution work is frozen-checkpoint evaluation. The factorized model gets one initial K=6 training run and earns continuation only by passing explicit gates.
5. Preserve all historical checkpoint and configuration behavior as loadable compatibility paths. The accepted Run-1401 checkpoint must remain a strict-load numerical reference.

## 2. Current model and the actual bottleneck

The accepted context-fusion decoder constructs query-to-edge attention

\[
\alpha_{qk}=\operatorname{softmax}_k(\ell_{qk}),
\]

and a hyperedge value context

\[
c_q^H=\sum_k \alpha_{qk}V_h h_k.
\]

The expensive branch is the hypergraph-gated query-module pair context. With edge-mass-normalized module incidence

\[
\bar A_{mk}=\frac{A^{MH}_{mk}}{\sum_{m'}A^{MH}_{m'k}+\epsilon},
\]

the current implementation evaluates a shared pair MLP for every query-module pair,

\[
\psi(q,m),
\]

then constructs edge-local pair contexts

\[
c^{pair}_{qk}=\sum_m \bar A_{mk}\psi(q,m),
\]

and finally reduces them with query-to-edge attention,

\[
c_q^{pair}=g_{pair}\sum_k\alpha_{qk}c^{pair}_{qk}.
\]

In the dense code path this materializes query-module pair embeddings and then a `[B,Q,K,H]` edge-pair context. The dominant cost therefore scales with `Q*M` through a four-layer width-256 MLP. The `Q*K` attention is comparatively cheap because the accepted model has only six edges.

The existing gathered research path already computes

\[
\beta_{qm}=\sum_k\alpha_{qk}\bar A_{mk},
\]

but currently uses it mainly to rank modules before reconstructing edge-specific contexts. That machinery proves useful correctness concepts, but it is not yet the right large-Q context-fusion executor.

The near-module Gaussian branch is also `O(QM)`, but it is much cheaper than the pair MLP because it does not run a deep hidden-width network on every pair. Keep it unchanged in the first sparse-execution implementation so the scientific comparison remains clean.

The most relevant maintained files are:

- `src/honf_forward_core/decoder.py`: query routing, context fusion, final field head;
- `src/honf_forward_core/decoding/pairwise.py`: pair kernel and current dense/gathered pair execution;
- `src/honf_forward_core/decoding/research.py`: additive/gathered compatibility behavior;
- `src/honf_forward_core/config.py`: strict core configuration;
- `src/honf_forward_core/model.py`: encode/organize/decode orchestration;
- `src/honf_forward_core/organizer.py`: accepted fixed organizer and checkpoint-visible state;
- `src/config_core/forward/stage7_structured_context.json`: Run-1401 model class;
- `tools/diagnostics/evaluate_retained_mass_pruning.py`: existing evaluation-only retained-mass machinery;
- `tools/diagnostics/benchmark_stage5_checkpoints.py`: current synchronized prepared/full-forward benchmark;
- `tools/diagnostics/evaluate_stage5_accuracy.py` and `evaluate_topology_quality.py`: maintained complete-split accuracy and representation diagnostics.

Do not rewrite the organizer during this round.

## 3. Model Test A — fused sparse context execution

### 3.1 Exact algebraic simplification

For context fusion, the current two-stage pair aggregation can be reordered exactly:

\[
\begin{aligned}
c_q^{pair}
&=g_{pair}\sum_k\alpha_{qk}\sum_m\bar A_{mk}\psi(q,m)\\
&=g_{pair}\sum_m\left(\sum_k\alpha_{qk}\bar A_{mk}\right)\psi(q,m)\\
&=g_{pair}\sum_m\beta_{qm}\psi(q,m).
\end{aligned}
\]

This is the central Stage-7 enhancement. It does not change the organizer, hyperedge state, query attention, pair-kernel weights, pair gate, global context, near-module context, field head, or loss. It removes the need to construct the edge-specific `[B,Q,K,H]` pair context during ordinary context-fusion prediction.

`beta` is also a scientifically useful quantity. It is the learned query-conditioned routing distribution over physical modules induced by the hypergraph. Because `alpha` is normalized over edges and each column of `bar A` is normalized over modules,

\[
\sum_m\beta_{qm}=1.
\]

This gives the sparse executor a natural retained-mass criterion and gives later diagnostics a more meaningful module-influence map than trying to make `A_mh` artificially categorical.

### 3.2 Minimal configuration extension

Keep configuration changes small and explicit. Prefer three clearly owned fields rather than adding a family of routing heuristics:

- `pairwise_aggregation_mode`: historical default `edge_explicit`; new value `fused_query_module`;
- `query_module_retained_mass_floor`: default `1.0`, constrained to `[0,1]`;
- `pairwise_kernel_mode`: historical default `legacy_mlp`; Test B adds `factorized_gated`.

Reuse existing `routing_execution` and `query_module_limit` rather than inventing more switches.

Required semantics:

- `edge_explicit + dense`: exact historical Run-1401 path;
- `edge_explicit + gathered`: preserve Stage-1--6 compatibility behavior;
- `fused_query_module + dense`: exact full-support fused context, evaluating all active modules;
- `fused_query_module + gathered`: retained-mass sparse fused context.

Historical configs that lack the new fields must reconstruct the old path. Do not change parameter names, parameter ownership, or state-dict structure for the legacy model.

For the new fused gathered path, `query_module_limit=0` means no hard count cap. If it is positive, treat it as an optional maximum execution budget after ranking. The retained-mass floor is the scientific selection criterion; a hard cap is only a deployment budget. If the cap prevents the requested mass from being retained, do not hide that condition: report the actual retained mass and a floor-violation diagnostic.

`module_incidence_retained_mass_floor` remains the existing edge-specific compatibility setting. Do not silently reinterpret it as the new `beta`-mass threshold.

### 3.3 Dense fused reference path

Implement the full-support fused executor first. It must:

- calculate the identical dense `alpha_qk` used by Run 1401;
- keep the full hyperedge value context `c_H` unchanged;
- form `beta_qm = alpha_qk @ bar_A_mk`;
- evaluate the existing pair MLP for all active modules;
- aggregate directly with `beta` into `[B,Q,H]`;
- avoid materializing `[B,Q,K,H]` unless an explicitly requested diagnostic requires edge-local reconstruction;
- leave global, near-module, LayerNorm, pair gate, and field head untouched;
- work with prepared/chunked decoding without query-chunk dependence.

This path uses the frozen Run-1401 weights. It requires no training run.

Because a changed contraction order may produce floating-point reduction differences, exact bitwise equality is not required. The target parity is the same tight scale already used by gathered-routing tests, approximately `rtol=2e-6`, `atol=2e-6`, with stricter zero-difference checks wherever operation order is unchanged.

### 3.4 Sparse fused execution

After dense fused parity is proven, add a vectorized sparse executor based on `beta`.

For each query, sort or top-k the active module weights and select the smallest prefix whose cumulative `beta` mass reaches

\[
\tau_M=\texttt{query\_module\_retained\_mass\_floor}.
\]

Then evaluate the pair kernel only for selected query-module pairs:

\[
\widehat c_q^{pair}=g_{pair}\sum_{m\in S_q}\beta_{qm}\psi(q,m).
\]

Do **not** renormalize `beta` after truncation for this frozen-reference execution path. The sparse result should be a true truncated approximation of the accepted dense formula. Renormalization would change the learned context magnitude and pair-gate calibration.

Implementation requirements:

- gathering must happen before the expensive pair kernel;
- do not use Python loops over queries;
- use regular padded gathered tensors or equivalent vectorized indexing suitable for GPU execution;
- avoid a full per-query sort when a bounded `topk` can satisfy the same execution policy;
- scalar diagnostics are the default; dense routing maps are materialized only when explicitly requested;
- the new selection/gathering code must scale with selected pairs rather than first constructing the full pair embedding and pruning afterward.

For the current ThermalChannel case, active module count is small, so speedup can be modest. The executor should be judged mainly by exactness and by scaling tests with larger synthetic `M` and `Q`, not by forcing a large speedup on five-module examples.

### 3.5 New routing diagnostics

The fused executor should expose enough information to prove what it is doing without bloating normal outputs.

Always-available scalar summaries should include:

- available active modules;
- mean selected modules per query;
- selected-pair / dense-pair ratio;
- retained `beta` mass mean, p05, and minimum;
- retained-mass floor violation fraction;
- evaluated pair count;
- whether aggregation is edge-explicit or fused;
- whether execution is dense or gathered.

Only when routing maps are explicitly requested may it return dense arrays such as `query_module_routing_beta [B,Q,M]`, selected-module masks, or module-influence maps.

The existing query-to-edge and topology diagnostics must remain unchanged.

### 3.6 Test-A correctness gates

Before any sparse approximation is evaluated scientifically:

- Run-1401 best-by-field must strict-load with the same historical state keys;
- historical `edge_explicit + dense` replay must remain unchanged;
- `fused_query_module + dense` must match historical dense prediction to approximately `2e-6` relative/absolute tolerance;
- full-support fused results must match one-shot versus chunked prepared decoding;
- gradients must remain finite in training mode;
- fused dense execution must not create `[B,Q,K,H]` pair context in normal context-fusion prediction;
- explicit routing-map requests may allocate diagnostics but must not alter predictions;
- no new trainable parameters are allowed for Test A.

### 3.7 Test-A sparse promotion gates

Use a small screening sweep first, then run the complete 90-case split only for the selected sparse setting.

Initial retained-mass candidates:

\[
\tau_M\in\{0.999,0.995,0.99,0.98\}.
\]

Screen them on one representative case plus synthetic large-`M` benchmarks. Select at most one primary sparse setting and, only if the tradeoff is genuinely ambiguous, one secondary setting for full-split evaluation.

For promotion of the sparse executor on Run-1401 weights:

- pooled complete-split field MSE degradation versus fused dense `<= 0.5%`;
- each output-channel MSE degradation `<= 1%`;
- near-interface temperature and vorticity MSE degradation `<= 1%`;
- case-p95 MSE degradation `<= 2%`;
- retained `beta` mass diagnostics must agree with the configured threshold, except explicitly reported hard-cap violations;
- topology and dense `alpha_qk` diagnostics must be numerically unchanged;
- on large-`M`, large-`Q` scaling tests, selected-pair execution should show meaningful memory/runtime reduction. A practical target is at least ~30% reduction in prepared-decoder incremental memory or time at `M>=32`, with larger gains expected as `M` increases.

If ThermalChannel's small active-module count does not permit 20% route removal, do not fail the executor solely for that reason. Record the result and rely on scaling benchmarks for the efficiency claim.

## 4. Model Test B — factorized gated pair kernel

### 4.1 Scientific question

The legacy pair kernel concatenates relative geometry, module presence, the 256-wide module token, and raw module features, then applies a four-layer hidden-width MLP to every query-module pair. This is expressive but expensive because module-dependent computation is repeated for every query.

Test one factorized kernel that separates module encoding from query-relative geometry so module-side work can be prepared once per case and only a low-rank interaction is evaluated per selected pair.

Do not test multiple fashionable architectures in this round.

### 4.2 Proposed factorized kernel

Let `R = pairwise_kernel_hidden_dim` be the factor rank. Reuse that existing configuration field rather than adding a separate rank hyperparameter.

A suitable implementation is:

\[
u_m=f_M([m_m,S_m,P_m])\in\mathbb{R}^{R},
\]

\[
r_{qm}=f_R(\Phi(\Delta_{qm}))\in\mathbb{R}^{R},
\]

\[
g_{qm}=\sigma(W_g(u_m+r_{qm})),
\]

\[
z_{qm}=\operatorname{LN}\left(u_m+r_{qm}+g_{qm}\odot u_m\odot r_{qm}\right),
\]

\[
\psi(q,m)=W_o z_{qm}\in\mathbb{R}^{H}.
\]

This preserves an explicit multiplicative module/geometry interaction while moving expensive module encoding out of the per-query loop. Codex may choose an equivalent small residual implementation if it is cleaner, but the factorization principle must remain clear and the new kernel must be materially smaller than the legacy MLP.

Recommended first and only planned K=6 candidate:

- `pairwise_kernel_mode = factorized_gated`;
- `pairwise_kernel_hidden_dim = 96`;
- `pairwise_kernel_num_layers = 2` for the module/relative encoders;
- same Fourier frequency setting as Run 1401 unless the implementation shows a compatibility reason otherwise;
- include the encoded module token and raw module features as in Run 1401;
- keep the existing scalar pairwise gate concept;
- use `fused_query_module + dense` during training so the architecture test is not confounded by discrete sparse selection.

Do not perform a rank sweep in the initial round. If the R=96 candidate fails the 500-epoch gate, stop and diagnose before launching another formal model.

### 4.3 Prepared-state caching

For the factorized mode, calculate `u_m` once per prepared case and reuse it across query chunks. This cache is runtime state, not checkpoint state. It must not duplicate model parameters inside the prepared object.

The ordinary non-prepared forward path may calculate `u_m` once per forward call. Both paths must produce the same prediction.

Do not generalize the entire HONF core to 3-D in this round. However, do not introduce new 2-D-only assumptions into the aggregation or factorization logic. Existing relative-geometry construction can remain the source of pair features for the current ThermalChannel test.

### 4.4 Training policy

The factorized model is trained end-to-end from scratch. Preserve all other Stage-7 choices:

- fixed `K=6` organizer;
- softmax module/environment/query assignments;
- raw organizer hyperedge state;
- context fusion;
- same environment grid;
- same input features, global context, Stage-A model, local coupling, and one refinement;
- same field/local/interface/port losses;
- same shared AdamW `3e-4`, weight decay, gradient clipping, data sampling, normalization, batch policy, and seed unless a strict technical reason requires otherwise.

Do not initialize the factorized model from Run 1401 or Run 1500. Do not introduce curriculum training or sparse gathered training in this round.

After training, evaluate the same factorized checkpoint under both fused dense and fused sparse execution. That gives the combined best-efficiency model without creating another training run.

## 5. Minimal run and continuation policy

The round should use as few formal training runs as possible.

### Phase A — implementation and frozen-checkpoint validation

No training run ID is needed.

1. Unit/regression tests for historical dense behavior.
2. Run-1401 best checkpoint: historical dense versus fused dense parity.
3. One representative physical case for sparse threshold screening and synchronized runtime measurement.
4. Synthetic scaling benchmark over increasing query/module counts.
5. Complete 90-case evaluation only for the selected sparse threshold, plus fused dense reference.

Do not create a separate result directory for every mass floor. One evaluation job should store the sweep table and selected-setting evidence.

### Phase B — one factorized K=6 training run

Allocate one unused managed run ID at launch time. Do not hard-code a run number into reusable templates before checking the run store.

Train initially to epoch 500 and compare to the matched Run-1401 trajectory. Run 1401's maintained matched-budget references are approximately:

| Epoch | validation field MSE, trailing-50 | validation temperature MSE, trailing-50 |
|---:|---:|---:|
| 500 | `2.2987e-2` | `1.5089e-2` |
| 1000 | `1.1217e-2` | `8.3802e-3` |
| 2500 | `3.4105e-3` | `3.7403e-3` |
| 5000 | `2.1983e-3` | `2.2895e-3` |

The first decision point is epoch 500.

Continue the same run beyond 500 only if all of the following are broadly satisfied:

- trailing field MSE is within about `1.15x` Run-1401's matched epoch-500 value;
- temperature MSE is within about `1.15x`;
- no evidence of unstable training or pathological channel imbalance;
- environment and query organization remain differentiated and pass the established Stage-7 structural bands (`environment cosine <0.20`, `environment rank >3.5`, `query cosine <0.55`, `query rank >3.0`) or are clearly trending into them;
- the factorized kernel materially reduces pair-kernel parameters and shows a plausible large-`M` runtime/memory advantage.

If it passes, resume the same run to epoch 2500. Do not launch a second copy.

At epoch 2500, prefer continuation to 5000 only if matched field/temperature MSE are within about `1.10x` Run 1401 and topology remains healthy, or if the convergence slope gives strong evidence that a small remaining gap is still closing while efficiency gains are substantial.

A 10K continuation is exceptional. Only extend beyond 5K if the 5K model is already scientifically competitive, the late trajectory is still clearly improving, and the additional budget would answer a specific unresolved question. Do not train to 10K merely for symmetry with Run 1500 or historical Run 1000.

### Phase C — final K=6 round comparison

The comparison set should contain only four logical entries, not four training runs:

1. Run 1401 best, historical dense reference;
2. the same Run-1401 checkpoint under the selected sparse fused executor;
3. factorized K=6 best checkpoint under fused dense execution;
4. the same factorized checkpoint under the selected sparse fused executor.

This cleanly separates execution change from representation change.

For final promotion of the factorized model at 5K or its run-owned best checkpoint, target:

- complete-split pooled field MSE `<=1.10x` Run-1401 best (`9.5379e-4` reference);
- no individual field channel worse than about `1.15x` Run 1401;
- Stage-7 topology gates still pass;
- no degradation of Stage-A/local-interface contracts;
- a material large-scale efficiency gain, with ~25% or greater improvement in prepared-decoder time or incremental memory at large `M,Q` as a useful minimum target;
- lower pair-kernel parameter count than the legacy model.

If the factorized model misses these gates, keep the sparse fused executor as the Stage-7 enhancement and do not force another architectural run.

## 6. Scaling benchmark requirements

Current ThermalChannel has too few active modules to represent future 3-D cost. Add a benchmark mode that can exercise the core decoder with controlled synthetic shapes while preserving tensor contracts.

At minimum cover:

\[
Q\in\{8192,65536,262144,10^6\},
\]

\[
M\in\{5,12,32,64,128\},
\]

with `K=6` first. Use query chunking whenever full one-shot execution would create an unrealistic OOM condition; record the chunk size.

Report:

- prepared-decoder median and p95 time;
- incremental peak allocated and reserved GPU memory;
- evaluated query-module pair count;
- selected modules per query;
- retained `beta` mass;
- pairwise kernel parameter count;
- end-to-end full-forward time for at least the real 8192-query ThermalChannel case.

Do not interpret output query count as model capacity. The organizer should be prepared once per case and reused across query chunks.

## 7. K-scaling audit after model selection

Do not mix K-scaling into the two main tests above. First select the best K=6 architecture/execution combination.

Then perform a small adaptive audit rather than automatically training four long runs.

Initial audit:

- selected K=6 model is the reference and requires no new run;
- train K=4 and K=8 variants only to epoch 500 from scratch under otherwise identical settings;
- compare matched accuracy, topology rank/cosine, regional separation, `beta` routing concentration, parameters, and runtime.

Decision:

- if neither K=4 nor K=8 provides a meaningful Pareto improvement, keep K=6 and stop;
- if one alternative is clearly promising, extend only that one to 2500 and, if justified, 5000;
- test K=12 only if K=8 shows clear evidence that representation capacity is still unsaturated;
- do not introduce automatic edge split/merge during this audit.

The audit should answer whether `K` behaves as a useful latent-rank/capacity knob for the selected model, not attempt to discover a unique number of physical mechanisms.

## 8. Code-structure requirements

Keep stable ownership boundaries.

`HypergraphOrganizerCore` should not gain sparse-execution logic. The organizer exports the same incidences, geometry, state, and diagnostics.

`HypergraphFieldDecoder` remains the owner of query routing and final context fusion. It decides whether pair aggregation is edge-explicit or fused.

`HypergraphGatedPairwiseKernel` or a clean replacement facade remains the owner of pair-feature construction, pair-kernel mode, and dense/gathered query-module evaluation. If implementation becomes clearer by splitting the legacy MLP and factorized kernel into focused internal modules, keep checkpoint-visible ownership deterministic and do not move historical parameters in a way that breaks old state keys.

`ResearchDecoderExecutionMixin` should continue to own additive/legacy research execution. Do not route the accepted context-fusion model through additive code merely to reuse sparsity machinery.

Prepared ThermalChannel evaluation should cache only reusable runtime tensors. It must not duplicate modules or parameter state.

Prefer extending existing tests and maintained diagnostics over creating parallel `*_v2.py`, `*_new.py`, or stage-specific copies.

## 9. Tests that Codex must add or extend

At minimum cover:

- historical Run-1000/Run-1401 golden replay remains valid;
- historical context-fusion dense state-dict key inventory is unchanged;
- fused dense versus historical dense parity with identical weights;
- fused dense one-shot versus query-chunk parity;
- fused gathered full-retention versus fused dense parity;
- sparse truncation uses original, non-renormalized `beta` weights;
- selected pair count is genuinely smaller before the pair kernel when sparsity is active;
- no inactive padded module is evaluated;
- retained-mass diagnostics are correct over routed query-module pairs;
- dense routing maps are absent from ordinary execution and present only when requested;
- factorized module codes are prepared once and reused by chunked decode;
- factorized dense one-shot/chunk parity;
- finite backward pass for both new aggregation mode and factorized kernel;
- strict configuration validation and historical defaults.

Do not weaken existing tolerances merely to make a new path pass.

## 10. Diagnostic and artifact-output rules

The repository already defines a canonical category-oriented evaluation layout and explicitly separates maintained diagnostic source from generated evidence. Follow it.

### Maintained source

- reusable diagnostic/evaluation Python belongs in `tools/diagnostics/`;
- do not add new generated evidence beside those scripts;
- do not create new top-level `diagnostics/*.py` wrappers unless compatibility genuinely requires one;
- prefer extending `evaluate_retained_mass_pruning.py`, the existing checkpoint benchmark, accuracy evaluator, and topology evaluator instead of cloning them;
- if one new synthesis script is needed, use a single clearly named maintained tool such as `tools/diagnostics/analyze_stage7_model_enhance.py` rather than several one-off files;
- do not create a new training entry point; use the existing `train.py` workflow with a small experiment overlay.

### Managed training results

Each formal training run must stay inside its normal managed `Run_####_*` directory with the existing checkpoint, metric, plot, config, manifest, and provenance conventions. Do not write ad-hoc comparison CSV/PNG files at the run root.

### Per-checkpoint evaluations

Use the canonical `evaluations/<kind>/<timestamped-job>/` layout with `EvaluationArtifactLayout` categories. Create only needed categories. Typical enhancement jobs should use:

```text
evaluations/stage7_model_enhance/<job>/
├── metrics/
├── arrays/                 only when raw arrays are genuinely needed
├── organization/           only for selected checkpoints/settings
├── routing/                only for selected checkpoints/settings
├── diagnostics/
├── summary.json
└── evaluation_manifest.json
```

Do not duplicate full field arrays for every retained-mass threshold. A sparse sweep should produce one metrics table and only materialize detailed maps for the selected setting.

### Cross-model synthesis

Generated scratch/comparison output belongs under the ignored

`diagnostics/generated/stage7_model_enhance/<timestamp>/`

root or an equivalent canonical managed evaluation job. A synthesis directory should contain reduced tables/plots and references to source evaluation manifests, not copied checkpoints, repeated NPZ field arrays, or duplicated evaluation folders.

If the round is accepted, write one concise tracked scientific report, for example `diagnostics/Stage7_Model_Enhance_Evaluation.md`, that references the generated evidence. Do not commit generated PNG/CSV/NPZ artifacts merely to make the report self-contained.

The existing `.gitignore` already ignores managed runs, evaluations, generated diagnostics, checkpoints, arrays, plots, and CSVs. Preserve that cleanliness.

## 11. Configuration and experiment-file policy

Keep `stage7_structured_context.json` as the accepted Run-1401 profile until a new model is formally promoted.

Create at most two reusable Stage-7 experiment overlays:

1. a fused-context execution overlay that selects `fused_query_module` while leaving model weights/architecture otherwise identical;
2. a factorized-pair-kernel overlay based on Stage 7 with `factorized_gated`, `R=96`, K=6, and the same training policy.

Retained-mass thresholds should be evaluation arguments or recorded overrides, not separate JSON files per threshold.

Do not change `recommended_forward_profile` in the registry until the comparison is complete. Do not mark an unfinished experiment as the new current scientific model.

## 12. Required final comparison report

At the end of this round, produce one comparison summary that answers four questions directly:

1. Does fused dense context execution preserve Run-1401 numerics while reducing unnecessary intermediate tensors?
2. How much sparse `beta`-mass pruning can Run 1401 tolerate before measurable field degradation, and how does the executor scale with `Q` and `M`?
3. Does the factorized gated pair kernel recover Stage-7 accuracy/topology by 500/2500/5000 epochs while reducing pairwise model/execution cost?
4. Which K=6 combination should become the platform for the later K-scaling audit and inverse-model work?

The final decision must be one of:

- retain Run-1401 architecture and promote only the fused sparse executor;
- promote factorized K=6 with dense fused execution;
- promote factorized K=6 with sparse fused execution;
- no promotion if neither enhancement meets the stated accuracy/correctness gates.

Avoid inventing a new model family name unless the structure is actually promoted.

## 13. Definition of done

This Stage-7 enhancement round is complete when:

- Run-1401 historical behavior is still replayable;
- fused dense context execution has passed parity tests;
- one sparse retained-mass setting has been evaluated on the complete split and scaling benchmark;
- exactly one planned factorized K=6 model has reached its justified stopping point (500, 2500, or 5000; 10K only by explicit evidence);
- the factorized checkpoint has been evaluated under both dense and selected sparse fused execution if it survived the 500-epoch gate;
- all generated outputs follow the canonical structured layout and no results directory is polluted by one-off copies;
- one final comparison report identifies the K=6 winner and states whether the later K-scaling audit should proceed.
