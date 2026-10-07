# HONF Forward-Model Upgrade: Codex Goal-Mode Implementation Plan

## 0. Mission

Upgrade the forward Hypergraph-Organized Neural Field (HONF) into a **topology-adaptive, physically sparse, edge-additive neural field** while preserving the existing forward and inverse workflows as selectable checkpoint-compatible modes.

Starting repository state:

- Repository: `cosmos2w/ModularDT`
- Project root: `HONF_Proj`
- Existing variable-module-count work must be preserved.

The upgrade should produce five capabilities:

1. **Anonymous, exchangeable hyperedges**
   - Hyperedge parameters must not be tied to edge index.
   - Runtime edge capacity must not determine learned parameter shapes.
   - A design-dependent active subset must be selected from candidate edges.

2. **Adaptive active edge count**
   - Begin from a practical default count such as `K_def=6`.
   - Allow simple designs to use fewer active edges.
   - Allow complex or overloaded interaction structures to use additional candidate edges.
   - Achieve this through competition, quality, coverage, and redundancy—not through a large collection of count and anti-collapse penalties.

3. **Physical sparsity with fuzzy overlap**
   - Module, environment, and query routing should have exact zero entries where appropriate.
   - Hyperedge regions should be spatially compact but may overlap.
   - Sparse routing must eventually reduce actual expensive computation, not only make dense matrices look sparse.

4. **Mechanism-first, edge-additive field generation**
   - Explicit topology descriptors must participate in the main prediction path.
   - The predicted field must be assembled as an exact background-plus-edge sum.
   - Edge-wise field contributions must be exportable and testable.

5. **A reusable topology signature**
   - The learned hypergraph must be represented as an unordered active edge set plus edge-to-edge relations.
   - This signature becomes the main design characteristic exported by the forward model.
   - A later inverse model must generate this set without fixed edge identities or ordered-plan flattening.

This is a controlled architectural upgrade, not a full rewrite.

---

## 1. Non-negotiable design rules

### 1.1 Keep physical-case ownership intact

- `src/honf_forward_core` remains independent of ThermalChannel, HDF5, plotting, and physical field-channel semantics.
- `Case_ThermalChannel` continues to own physical features, Stage-A coupling, losses, plots, and optional structure targets.
- Do not move ThermalChannel-specific assumptions into the reusable core.

### 1.2 Preserve old checkpoints naturally

Old checkpoints must reconstruct their original computation path.

Use descriptive configuration modes:

| Concern | Existing computation | New computation |
|---|---|---|
| Organizer | `fixed_projection` | `exchangeable_slots` |
| Field assembly | `context_fusion` | `edge_additive` |
| Mechanism state | `residual_concat` | `descriptor_first` |
| Assignment transform | `softmax` | `entmax15` |
| Sparse execution | `dense` | `gathered` |
| Inverse plan tokens | `indexed` | `exchangeable_set` |
| Plan conditioning | `ordered_flat` | `set_cross_attention` |

Rules:

- A saved config that does not contain the new mode fields must be resolved to the existing computation.
- New training profiles must specify the new modes explicitly.
- Instantiate only the modules needed by the selected mode so strict state-dict loading remains meaningful.
- Do not force old weights into the new organizer or additive decoder.
- A separate warm-start utility may reuse compatible encoders and case-coupling weights later, but it is not required for the first upgrade.

### 1.3 Naming

Do not introduce public names containing words such as `legacy`, `maintained`, `old`, or `new`.

Use:

- descriptive behavior names;
- schema versions where needed;
- compatibility aliases only inside parsing code.

Do not rename existing state-dict keys merely for style.

### 1.4 Minimal-change discipline

- Keep the current fixed-projection organizer code path intact.
- Add the exchangeable organizer beside it.
- Keep the current context-fusion decoder intact.
- Add the edge-additive path in the same decoder or a focused helper.
- Do not rewrite Stage A, port prediction, local coupling, dataset readers, or runtime dispatch.
- Do not upgrade the inverse model until the forward topology schema and edge-additive field are accepted.

---

## 2. Target model: Topology-Adaptive HONF

For each case \(b\), let the module set be

\[
\mathcal M_b=\{(\mathbf x_{bm},\mathbf s_{bm})\}_{m=1}^{N_b},
\]

the environment-token set be

\[
\mathcal E_b=\{(\mathbf y_{be},\mathbf r_{be})\}_{e=1}^{E_b},
\]

the case context be \(\mathbf c_b\), and the query coordinate be \(\mathbf q_{bq}\).

The forward operator remains

\[
\widehat{\mathbf u}_b(\mathbf q)
=
\mathcal G_\Theta(\mathcal M_b,\mathcal E_b,\mathbf c_b;\mathbf q).
\]

The upgraded model introduces a runtime candidate-edge budget \(K_{\mathrm{cap}}\) and a design-dependent active mask

\[
z_{bk}\in\{0,1\},\qquad
K_{\mathrm{active},b}=\sum_{k=1}^{K_{\mathrm{cap}}}z_{bk}.
\]

`K_cap` is a computation budget, not an edge identity and not a learned-parameter dimension.

Recommended first ThermalChannel profile:

```text
K_def = 6
K_cap = 8
K_min = 1
slot_refinement_steps = 2
```

The shared model should also be able to instantiate another `K_cap` without changing learned parameter shapes.

---

## 3. Exchangeable candidate-edge organizer

### 3.1 Anonymous slot initialization

Do not use:

```python
nn.Linear(hidden_dim, K)
nn.Embedding(K, hidden_dim)
```

in the exchangeable organizer.

Construct candidate slots as

\[
\mathbf h^{(0)}_{bk}
=
f_{\mathrm{base}}(\mathbf g_b)
+
f_{\mathrm{scale}}(\mathbf g_b)
\odot \boldsymbol\xi_k,
\]

where:

- \(\mathbf g_b\) is a pooled case state;
- every candidate uses the same learned functions;
- \(\{\boldsymbol\xi_k\}\) is a deterministic, non-trainable, zero-mean symmetry-breaking code set generated for the requested `K_cap`;
- permuting the code set must only permute the edge axis of the result.

A practical code generator may use normalized sinusoidal or low-discrepancy codes. It must not create learned per-index semantics.

Add a unit test that permutes the candidate codes and verifies that:

- active-edge features permute consistently;
- total predicted field remains unchanged;
- the exported unordered topology is unchanged after matching.

### 3.2 Iterative competitive assignment

Use two shared refinement iterations by default.

For module tokens \(\mathbf z_{bm}\) and candidate edge states \(\mathbf h_{bk}^{(t)}\),

\[
\ell^{M,(t)}_{bmk}
=
\frac{
(W_Q^M\mathbf z_{bm})^\top
(W_K^M\mathbf h_{bk}^{(t)})
}{\sqrt H}
+
b^M_{bmk}.
\]

For environment tokens,

\[
\ell^{E,(t)}_{bek}
=
\frac{
(W_Q^E\mathbf e_{be})^\top
(W_K^E\mathbf h_{bk}^{(t)})
}{\sqrt H}
+
b^E_{bek}.
\]

Normalize over candidate edges for every input token:

\[
P^{M,(t)}_{bmk}
=
\operatorname{Norm}_k(\ell^{M,(t)}_{bmk}),
\]

\[
P^{E,(t)}_{bek}
=
\operatorname{Norm}_k(\ell^{E,(t)}_{bek}).
\]

The new default `Norm` is `entmax15`; `softmax` remains selectable.

For aggregation into one edge, normalize each edge column over the corresponding input tokens:

\[
\bar P^M_{bmk}
=
\frac{P^M_{bmk}}
{\sum_j P^M_{bjk}+\varepsilon},
\qquad
\bar P^E_{bek}
=
\frac{P^E_{bek}}
{\sum_j P^E_{bjk}+\varepsilon}.
\]

Then form shared updates:

\[
\mathbf u^M_{bk}
=
\sum_m \bar P^M_{bmk}W_V^M\mathbf z_{bm},
\]

\[
\mathbf u^E_{bk}
=
\sum_e \bar P^E_{bek}W_V^E\mathbf e_{be},
\]

\[
\mathbf h_{bk}^{(t+1)}
=
\operatorname{SharedUpdate}
\left(
\mathbf h_{bk}^{(t)},
\mathbf u^M_{bk},
\mathbf u^E_{bk}
\right).
\]

A GRU-style update or a residual MLP is acceptable. All parameters must be shared across \(k\).

### 3.3 Geometry-aware but not geometry-determined

After module assignment, compute the source centroid

\[
\boldsymbol\mu^S_{bk}
=
\sum_m \bar P^M_{bmk}\mathbf x_{bm}.
\]

After environment assignment, compute the region centroid and diagonal variance

\[
\boldsymbol\mu^R_{bk}
=
\sum_e \bar P^E_{bek}\mathbf y_{be},
\]

\[
(\boldsymbol\sigma^R_{bk})^2
=
\sum_e
\bar P^E_{bek}
(\mathbf y_{be}-\boldsymbol\mu^R_{bk})^{\odot2}.
\]

Also compute a module-side dispersion

\[
(\boldsymbol\sigma^S_{bk})^2
=
\sum_m
\bar P^M_{bmk}
(\mathbf x_{bm}-\boldsymbol\mu^S_{bk})^{\odot2}.
\]

Use a broad geometry bias in the first iteration and the previous iteration's region state in later iterations. Geometry must encourage compactness without preventing a downstream or nonlocal region from being learned.

A suitable smooth compactness factor is

\[
\kappa(r)=\left[\max(1-r^2,0)\right]^2,
\]

where \(r\) is an anisotropically normalized distance to the previous region. Add

\[
\lambda_{\mathrm{loc}}\log(\kappa(r)+\varepsilon)
\]

to semantic assignment logits. Keep `locality_strength` configurable.

Do not hard-code a ThermalChannel downstream direction in the reusable core.

---

## 4. Adaptive active-edge selection without a count penalty

### 4.1 Why not literal split/merge mutations

Do not begin with discrete operations that clone, delete, or edit edge tensors during the forward pass. They introduce:

- unstable gradients;
- state-dict and batching complications;
- brittle threshold schedules;
- ambiguous optimizer state;
- edge-index bookkeeping that contradicts exchangeability.

Instead, maintain candidate slots and interpret selection as:

- **splitting:** several candidate slots discover complementary support and are all required for coverage;
- **merging/absorption:** overlapping or low-novelty candidates are not selected, and their mass is renormalized into selected candidates.

This provides the intended behavior without a hand-coded topology mutation process.

### 4.2 Raw quality score

From raw candidate incidences \(P^M\) and \(P^E\), define winner indices

\[
w^M_{bm}=\arg\max_k P^M_{bmk},
\qquad
w^E_{be}=\arg\max_k P^E_{bek}.
\]

Define assignment-purity scores

\[
Q^M_{bk}
=
\frac{
\sum_{m:w^M_{bm}=k}P^M_{bmk}
}{
\sum_m P^M_{bmk}+\varepsilon
},
\]

\[
Q^E_{bk}
=
\frac{
\sum_{e:w^E_{be}=k}P^E_{bek}
}{
\sum_e P^E_{bek}+\varepsilon
}.
\]

Use a combined quality such as

\[
Q_{bk}
=
\sqrt{Q^M_{bk}Q^E_{bk}}.
\]

The implementation may add a bounded compactness factor, but do not create a long weighted score with many arbitrary terms.

### 4.3 Coverage and novelty selection

Sort candidates by detached quality.

For a selected set \(S\), define token coverage

\[
C_M(S)
=
\frac{1}{N_b}
\sum_m
\mathbf 1
\left[
\sum_{k\in S}P^M_{bmk}\ge\tau_M
\right],
\]

\[
C_E(S)
=
\frac{1}{E_b}
\sum_e
\mathbf 1
\left[
\sum_{k\in S}P^E_{bek}\ge\tau_E
\right].
\]

Define candidate redundancy using the maximum cosine overlap of its module and environment membership columns with already selected edges. A candidate is novel when this overlap is below a configured maximum.

Select candidates in descending quality until:

```text
module coverage >= coverage_rate
and
environment coverage >= coverage_rate
and
at least K_min edges are active
```

During the initial warmup, select the best `K_def` candidates. After warmup, select the smallest quality/coverage subset up to `K_cap`.

Recommended initial values:

```text
selection_mode = quality_coverage
selection_warmup_epochs = 200
initial_active_edges = 6
minimum_active_edges = 1
coverage_rate = 0.95
token_coverage_threshold = 0.50
maximum_redundancy = 0.85
```

These are starting values, not scientific constants.

### 4.4 Mask and renormalize

Let \(z_{bk}\) be the detached selected mask. Apply it to raw incidence and renormalize each input row:

\[
A^M_{bmk}
=
\frac{
P^M_{bmk}z_{bk}
}{
\sum_jP^M_{bmj}z_{bj}+\varepsilon
},
\]

\[
A^E_{bek}
=
\frac{
P^E_{bek}z_{bk}
}{
\sum_jP^E_{bej}z_{bj}+\varepsilon
}.
\]

This is the absorption step: deselected candidates contribute zero, and selected edges recover complete token mass.

Keep both raw and active tensors in organizer output:

```text
candidate_A_mh
candidate_A_eh
A_mh
A_eh
edge_quality
edge_active_mask
active_edge_count
```

### 4.5 Warmup control

Add a small model API such as:

```python
model.set_training_progress(epoch=..., total_epochs=...)
```

or a narrower organizer method. The training workflow sets it once per epoch.

- Evaluation always uses fully enabled selection.
- Existing organizer mode ignores this API.
- Do not pass epoch tensors through every model call.

### 4.6 Default regularization policy

Do not add a direct `target_active_edges` penalty to the new default.

The default structural controls should be architectural:

- competitive normalization;
- quality;
- coverage;
- novelty;
- geometry bias;
- active masking.

Keep hard-concrete gates, differentiable top-k, and explicit count heads as optional experiments only.

---

## 5. Physical sparsity with fuzzy overlap

### 5.1 Exact sparse probability maps

Implement and test a reusable `entmax15` helper in:

```text
src/honf_forward_core/routing.py
```

Use it for:

- module-to-edge assignment;
- environment-to-edge assignment;
- query-to-edge routing.

Requirements:

- exact zeros for sufficiently low logits;
- nonnegative outputs;
- row sums equal one after masking;
- finite gradients;
- a deterministic CPU/CUDA implementation;
- no new dependency unless unavoidable.

Keep `softmax` selectable.

Fuzzy overlap is not the same as dense routing. Multiple edges may have nonzero support on the same module, environment region, or query, while unrelated edges receive exact zero.

### 5.2 Sparse query routing

For query state \(\mathbf d_{bq}\) and descriptor-first edge state \(\widetilde{\mathbf h}_{bk}\),

\[
\ell^{QH}_{bqk}
=
\frac{
(W_Q\mathbf d_{bq})^\top
(W_K\widetilde{\mathbf h}_{bk})
}{\sqrt H}
+
b^{geo}_{bqk}
+
\log(z_{bk}+\varepsilon).
\]

Then

\[
\alpha_{bqk}
=
\operatorname{entmax}_{1.5,k}
(\ell^{QH}_{bqk}).
\]

Inactive edges must receive exactly zero.

Retain optional hard top-\(r\) execution after the sparse probability is computed.

### 5.3 Do not confuse sparse weights with sparse computation

Replacing softmax with entmax does **not** by itself reduce the cost of:

- computing all routing logits;
- creating all query-module pair embeddings;
- executing an MLP for every dense pair;
- executing an edge head for every candidate edge.

The implementation must report sparsity and computational selection separately.

---

## 6. Descriptor-first mechanism state

The present residual-concatenation mechanism encoder can ignore explicit descriptors. Add a new mode in which descriptors form the primary state.

Let the explicit descriptor contain at least:

\[
\mathbf g_{bk}
=
[
\boldsymbol\mu^S,
\boldsymbol\sigma^S,
\boldsymbol\mu^R,
\boldsymbol\sigma^R,
\boldsymbol\mu^R-\boldsymbol\mu^S,
d_{SR},
m^M,m^E,
Q^M,Q^E,
z
].
\]

Include periodic-coordinate handling already present in the core.

Define:

\[
\mathbf h^{mech}_{bk}
=
f_{mech}(\mathbf g_{bk}),
\]

\[
\mathbf h^{content}_{bk}
=
f_{content}(\mathbf h^{raw}_{bk}),
\]

\[
\widetilde{\mathbf h}_{bk}
=
\operatorname{LayerNorm}
\left(
\mathbf h^{mech}_{bk}
+
\rho\,\mathbf h^{content}_{bk}
\right),
\]

where \(\rho\) is fixed or bounded.

Recommended first value:

```text
mechanism_latent_residual_scale = 0.35
```

The topology descriptors therefore cannot be bypassed, while the bounded content residual retains information not expressible by geometry and masses alone.

Use \(\widetilde{\mathbf h}\) for:

- query keys;
- query values;
- edge-additive field prediction;
- topology export summaries.

The existing mechanism encoder remains available under `residual_concat`.

---

## 7. Exact additive field assembly

### 7.1 Required equation

The new field path must satisfy

\[
\boxed{
\widehat{\mathbf u}_{bq}
=
\widehat{\mathbf u}^{bg}_{bq}
+
\sum_{k=1}^{K_{\mathrm{cap}}}
\widehat{\mathbf u}^{edge}_{bqk}
}
\]

with

\[
\widehat{\mathbf u}^{edge}_{bqk}
=
z_{bk}\alpha_{bqk}
f_{edge}
\left(
\mathbf d_{bq},
\widetilde{\mathbf h}_{bk},
\boldsymbol\psi_{bqk},
\mathbf r_{bqk}
\right).
\]

Here:

- \(\boldsymbol\psi_{bqk}\) is query-to-source/region geometry;
- \(\mathbf r_{bqk}\) is the edge-local query-module interaction context;
- \(f_{edge}\) is shared over all edges.

The background field should be

\[
\widehat{\mathbf u}^{bg}_{bq}
=
f_{bg}
\left(
\mathbf d_{bq},
\mathbf c^G_{bq},
\mathbf c^{env}_{bq}
\right).
\]

Do not give the background branch a full direct module-memory path in the default additive profile. Module-conditioned detail should pass through edge contributions.

### 7.2 Reuse the present pairwise structure

The current pairwise kernel already forms edge-local pair context before summing over edges. Refactor it so the additive path can consume the non-detached tensor

```text
edge_pair_context [B,Q,K,H]
```

or its gathered equivalent.

The context-fusion path must keep its current behavior and return signature.

### 7.3 Output contract

Standard training does not need to retain a full `[B,Q,K,F]` tensor. Support:

```text
pred_field
pred_field_background                 optional or small
pred_field_by_edge                    only when explicitly requested
edge_contribution_abs_mean            [B,K,F]
edge_contribution_rms                 [B,K,F]
edge_contribution_energy_fraction     [B,K,F]
```

Add a flag such as:

```text
return_edge_fields=False
```

The exact closure test must verify

```python
pred_field == pred_field_background + pred_field_by_edge.sum(dim=2)
```

within numerical tolerance when edge fields are requested.

### 7.4 Prevent silent background takeover

Track, but do not immediately penalize:

- background field norm;
- summed edge field norm;
- per-channel edge contribution fraction;
- fraction of module perturbation response explained by edge contributions.

If the background branch consistently explains nearly all design-dependent variation, treat that as an evaluation failure before adding another regularizer.

---

## 8. Sparse execution that produces real speedup

### 8.1 Cheap hypergraph-based module routing

Define the query-to-module relevance induced by the hypergraph:

\[
\beta_{bqm}
=
\sum_k
\alpha_{bqk}
\bar A^M_{bmk}.
\]

Computing \(\beta\) is cheap compared with evaluating the pair MLP.

Select at most `R_m` active modules per query:

\[
S_{bq}
=
\operatorname{TopR_m}_m(\beta_{bqm}).
\]

Gather only their:

- centers;
- encoded tokens;
- raw module features;
- presence values;
- module-to-edge memberships.

Evaluate the expensive pair MLP on `[B,Q,R_m,*]`, not `[B,Q,M,*]`.

Recommended first value:

```text
query_module_limit = 8
```

### 8.2 Edge execution limit

Optionally select at most `R_e` active edges per query after sparse routing:

```text
query_edge_limit = 3
```

The shared edge field head is evaluated only for those edge indices in `gathered` mode.

### 8.3 Two execution modes

Implement:

```text
routing_execution = dense
routing_execution = gathered
```

Requirements:

- `dense` remains the debugging/reference path.
- `gathered` becomes the intended new deployment path only after validation.
- When `query_module_limit >= M` and `query_edge_limit >= K_active`, dense and gathered outputs must match within tolerance using the same weights.
- The code must avoid first building the dense `[B,Q,M,H]` pair tensor and then slicing it.

### 8.4 Runtime diagnostics

Record:

```text
candidate_edge_count
active_edge_count
mean_query_nonzero_edges
pairwise_available_modules
pairwise_selected_modules
pairwise_selection_ratio
edge_head_available_routes
edge_head_selected_routes
edge_head_selection_ratio
```

### 8.5 Benchmark tool

Add:

```text
tools/benchmark_honf_sparse_routing.py
```

Measure:

- median and p95 forward latency;
- peak CUDA memory;
- query throughput;
- pair-MLP evaluated pair count;
- edge-head evaluated route count.

Use synthetic cases covering:

```text
M = 12, 32, 64, 128
Q = 1024, 8192, 32768
K_cap = 6, 8, 12, 16
```

Use CUDA events, warmup iterations, synchronization, and repeated measurements.

Do not claim an order-of-magnitude acceleration unless the measured workload demonstrates it.

---

## 9. The topology signature

### 9.1 Mathematical object

The topology of one design is not an ordered vector. Define it as

\[
\mathcal T(D,c)
=
\left(
\{\mathbf t_k:z_k=1\},
\mathbf R
\right),
\]

where \(\mathbf t_k\) is an active edge token and \(\mathbf R\) is an edge-to-edge relation tensor.

Recommended edge token contents:

```text
active probability / hard mask
quality
module-side centroid
module-side scale
environment-region centroid
environment-region scale
source-to-region displacement and distance
module mass
environment mass
module assignment purity
environment assignment purity
effective module count
mean query routing on a reference probe measure
per-output-channel contribution mean/RMS/fraction
```

Recommended relations:

\[
O^M_{k\ell}
=
\frac{
(A^M_{\cdot k})^\top A^M_{\cdot\ell}
}{
\|A^M_{\cdot k}\|
\|A^M_{\cdot\ell}\|+\varepsilon
},
\]

\[
O^E_{k\ell}
=
\frac{
(A^E_{\cdot k})^\top A^E_{\cdot\ell}
}{
\|A^E_{\cdot k}\|
\|A^E_{\cdot\ell}\|+\varepsilon
},
\]

\[
O^Q_{k\ell}
=
\mathbb E_{\mathbf q\sim\nu_Q}
[\alpha_k(\mathbf q)\alpha_\ell(\mathbf q)].
\]

Also store relative source and region geometry between edge pairs.

### 9.2 Reference query measure

Query-dependent topology summaries must use a reproducible reference set, not a random training query sample.

For ThermalChannel, use the evaluation grid or a fixed probe grid. For another case, the case plugin supplies the reference query measure.

### 9.3 Export schema

Add a new exporter:

```text
src/honf_forward_core/evaluation/topology_signature.py
```

Suggested schema:

```text
schema_name = honf_topology_signature
schema_version = 3
edge_mask                    [K_cap]
edge_features                [K_cap,F_t]
edge_relations               [K_cap,K_cap,F_r]
module_incidence             [M,K_cap]
environment_incidence        [E,K_cap]
candidate_module_incidence   optional
candidate_environment_incidence optional
query_route_summary          [K_cap,F_q]
field_contribution_summary   [K_cap,F]
num_module_slots             scalar
active_module_count          scalar
candidate_edge_count         scalar
active_edge_count            scalar
```

Fix the existing ambiguity where a padded slot count can be labeled as module count.

Keep the current plan exporter intact for checkpoints and inverse artifacts that use its existing schema.

### 9.4 Canonical sorting

Canonical sorting is allowed only for:

- display;
- JSON/NPZ serialization;
- human-readable comparison.

It must not define edge identity in training.

The exporter must retain the permutation used for serialization.

### 9.5 Topology distance

For two signatures, define set matching over active edge tokens:

\[
d_T
=
\min_{P\in\Pi}
\sum_{k\ell}P_{k\ell}c(\mathbf t_k,\widehat{\mathbf t}_\ell)
+
\lambda_R
\|\mathbf R-P\widehat{\mathbf R}P^\top\|_F^2.
\]

For unequal active counts, use padded null edges or an unbalanced assignment cost. The implementation may begin with masked Sinkhorn or Hungarian matching.

---

## 10. Optional topology supervision already supported by ThermalChannel data

The packed ThermalChannel data may expose:

```text
env_module_influence_target
module_affinity_target
active_edge_count_target
has_solved_structure_targets
```

Do not supervise edge indices.

Construct permutation-invariant relations from learned incidence.

Module affinity:

\[
\widehat C^{MM}_{mn}
=
\sum_k
\frac{
A^M_{mk}A^M_{nk}
}{
\sum_jA^M_{jk}+\varepsilon
}.
\]

Environment-to-module influence can be reconstructed from query routing on the structure target coordinates:

\[
\widehat C^{EM}_{em}
=
\sum_k
\alpha_k(\mathbf y_e)
\bar A^M_{mk}.
\]

First implement these as evaluation metrics.

Only after confirming that they correlate with useful topology should a small optional loss be enabled:

\[
\mathcal L_{\mathrm{topology}}
=
\lambda_{MM}D(\widehat C^{MM},C^{MM})
+
\lambda_{EM}D(\widehat C^{EM},C^{EM}).
\]

Rules:

- use only rows with `has_solved_structure_targets=1`;
- keep the loss off by default in the first architecture run;
- do not use direct edge labels;
- do not let an active-edge-count target override quality/coverage selection;
- record solved-target and fallback-target metrics separately.

---

## 11. Required configuration additions

Add to `UnifiedForwardConfig` or the appropriate strict schemas:

```text
organizer_mode
edge_capacity
initial_active_edges
minimum_active_edges
slot_refinement_steps
slot_code_mode

edge_selection_mode
selection_warmup_epochs
selection_coverage_rate
selection_token_threshold
selection_maximum_redundancy

module_assignment_normalizer
environment_assignment_normalizer
query_assignment_normalizer
entmax_alpha

environment_locality_mode
environment_locality_strength
minimum_region_scale

mechanism_state_mode
mechanism_latent_residual_scale

field_assembly_mode
routing_execution
query_edge_limit
query_module_limit

topology_signature_enabled
```

Validation rules:

- fixed-projection mode requires `num_hyperedges > 0`;
- exchangeable-slot mode requires `edge_capacity > 0`;
- `initial_active_edges <= edge_capacity`;
- `minimum_active_edges <= initial_active_edges`;
- gathered limits must be nonnegative;
- mode names must be strict;
- old saved dictionaries missing these keys are filled with the existing computation modes before dataclass construction.

Suggested new forward profile:

```text
src/config_core/forward/adaptive_sparse_additive.json
```

Do not mutate `enhanced_honf_pairwise.json` into the new architecture.

---

## 12. Code map

### 12.1 Core files to modify

```text
src/honf_forward_core/config.py
src/honf_forward_core/model.py
src/honf_forward_core/organizer.py
src/honf_forward_core/decoder.py
src/honf_forward_core/training/diagnostics.py
src/honf_forward_core/evaluation/hypergraph_plan.py   # compatibility only
```

### 12.2 Focused new files

```text
src/honf_forward_core/routing.py
src/honf_forward_core/topology.py
src/honf_forward_core/evaluation/topology_signature.py
tools/benchmark_honf_sparse_routing.py
```

### 12.3 ThermalChannel changes should be narrow

```text
Case_ThermalChannel/src/channelthermal/model.py
Case_ThermalChannel/src/channelthermal/workflows/train_forward.py
Case_ThermalChannel/src/channelthermal/evaluation_tools/
Case_ThermalChannel/configs/case_default.json
```

Expected changes:

- pass training progress to the organizer;
- expose optional edge-field and topology outputs;
- add optional topology metrics/loss;
- add topology plots;
- preserve all Stage-A/local-coupling behavior.

### 12.4 Later inverse files

Do not edit these until the forward topology schema is stable:

```text
src/honf_inverse_core/models/plan_flow.py
src/honf_inverse_core/models/layout_flow.py
src/honf_inverse_core/models/hierarchical_inverse.py
src/honf_inverse_core/models/matching.py
Case_ThermalChannel/src/channelthermal/inverse/compact_plan.py
Case_ThermalChannel/src/channelthermal/inverse/dataset_io.py
```

---

## 13. Goal-mode phases

## Goal 0 — Freeze the reference behavior

### Objective

Create a reproducible acceptance baseline before modifying the architecture.

### Tasks

1. Record the starting commit and working-tree state.
2. Run:
   ```bash
   pytest -q tests Case_ThermalChannel/tests
   ```
3. Run compile and JSON validation already used by the project.
4. If the reference checkpoint and dataset are available:
   - evaluate one fixed case;
   - save field arrays, metrics, topology export, and artifact hashes.
5. Record current parameter count and a dense-routing benchmark.
6. Create:
   ```text
   docs/forward_upgrade_reference.md
   ```

### Acceptance

- Existing tests pass or pre-existing failures are explicitly recorded.
- No source change occurs in this goal except the reference document.

### Completion summary — 2026-08-17

- Status: accepted.
- Frozen commit: `2afa84759858931a236321e0086750734466dcec` on `main`;
  the pre-existing `.gitignore` modification was preserved.
- Full suite: `116 passed, 1 skipped in 14.32s`; the skip requires optional
  local inverse integration artifacts.
- Compile validation passed and all 24 tracked JSON documents parsed.
- Autonomous Run 0002 checkpoint evaluation completed on CUDA device 0 for
  test case `0273`; arrays, metrics, existing schema-v2 plan, diagnostics, and
  hashes were saved in the run's ignored `evaluations/forward_upgrade_goal0`
  directory.
- Reference model: 3,507,625 total parameters and 2,472,486 trainable
  parameters.
- Prepared dense decoder at `Q=8192`: 22.932545 ms median, 30.695423 ms p95,
  451,502,592 peak allocated CUDA bytes, 98,304 dense module routes, and 49,152
  dense edge routes.
- No training was launched and no generated artifact was added to version
  control. Full commands, metrics, hashes, limitations, and compatibility
  behavior are recorded in `docs/forward_upgrade_reference.md`.

---

## Goal 1 — Add compatibility scaffolding and strict modes

### Objective

Introduce descriptive configuration switches without changing numerical behavior.
Take results in HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs as old models.

### Tasks

1. Add mode fields and strict validation.
2. Add config-resolution logic:
   - saved payload missing mode fields -> fixed projection, context fusion, residual concat, softmax, dense execution;
   - new profile explicitly selects upgraded modes.
3. Instantiate mode-specific modules only.
4. Add tests for:
   - old config dictionary resolution;
   - unknown mode rejection;
   - strict state-dict reconstruction of an old-mode model;
   - unchanged old-mode output with a fixed random seed.

### Acceptance

- The existing profile produces the same tensors as before.
- Old checkpoint loading does not report new missing parameters.
- No upgraded computation is enabled yet.

### Completion summary — 2026-08-17

- Status: accepted.
- Added strict behavior modes, cross-field validation, and compatibility
  resolution in `UnifiedForwardConfig`; missing mode fields select fixed
  projection, residual concatenation, context fusion, softmax assignment, and
  dense execution.
- Added the fully explicit `adaptive_sparse_additive.json` profile without
  changing `enhanced_honf_pairwise.json` or enabling the existing optional
  organizer regularizer.
- Added schema entries, migration documentation, and focused config/checkpoint
  tests. No mode-specific neural modules were added in this scaffolding stage,
  so the compatible model state remains unchanged.
- Focused command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q tests/test_forward_upgrade_config.py tests/test_core_contract.py tests/test_config_and_registry.py`
  -> `37 passed in 7.29s`.
- Full command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q tests Case_ThermalChannel/tests`
  -> `128 passed, 1 skipped in 9.83s` (the same optional integration-artifact
  skip as Goal 0).
- Compile validation, the upgrade profile/core schema JSON parse, and
  `git diff --check` passed. Ruff was not installed in the maintained conda
  environment and was therefore not part of the feasible gate.
- The actual Run 0002 checkpoint loaded strictly with no missing or unexpected
  keys. Its field, internal-temperature, interface, and port-condition arrays
  were bit-for-bit equal to the Goal 0 artifacts (`max_abs_diff=0.0`).
- No training or generated checkpoint run was launched.

---

## Goal 2 — Implement edge-additive output and descriptor-first state on the fixed organizer

### Objective

Gain explicit field decomposition before changing topology discovery.

### Tasks

1. Add `mechanism_state_mode="descriptor_first"`.
2. Add module and environment dispersions to organizer descriptors.
3. Refactor pairwise code to expose edge-local pair context.
4. Add `field_assembly_mode="edge_additive"`.
5. Add optional edge-field output and contribution summaries.
6. Keep all routing dense and keep the current fixed-projection organizer.

### Tests

- exact additive closure;
- inactive edge contributes zero;
- permuting the fixed edge axis consistently does not change total field;
- gradients reach the mechanism encoder and edge head;
- old context-fusion mode remains unchanged;
- prepared chunk decoding matches one-shot decoding.

### Acceptance

- Structural tests pass.
- One-batch CPU and CUDA smokes complete.
- No long training is launched by Codex.

### Completion summary — 2026-08-17

- Status: accepted.
- The fixed organizer now exports source/environment diagonal variance and
  scale, assignment purities, edge quality, an all-active mask, and a
  normalized 16-value descriptor-first feature vector. Existing mechanism
  descriptor tensors remain unchanged for checkpoint compatibility.
- `descriptor_first` uses a shared descriptor encoder plus a bounded shared
  content residual (`mechanism_latent_residual_scale`); no edge-index
  parameter is present.
- `edge_additive` instantiates only its environment/global background modules
  and shared edge head. It computes
  `pred_field = pred_field_background + pred_field_by_edge.sum(dim=2)`, returns
  edge fields only when requested, and returns detached per-edge
  contribution summaries during standard execution. The background has no
  module-memory input.
- The pairwise kernel exposes its non-detached edge-local context internally;
  `context_fusion` keeps its checkpoint modules, keys, numerical behavior, and
  normal return contract.
- Focused command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q tests/test_forward_additive.py tests/test_forward_upgrade_config.py tests/test_core_contract.py tests/test_forward_losses.py Case_ThermalChannel/tests/test_dynamic_collation.py`
  -> `45 passed in 4.92s`.
- Existing coupled CUDA diagnostics:
  `check_honf_hardening.py --device cuda:0 --points 32` and
  `check_global_modes.py --device cuda:0 --points 32` both passed, including
  fallback, teacher, predicted, mixed, Stage-A, and local-coupling paths.
- Additive CUDA core smoke: field shape `(2,13,3)`, finite backward gradients,
  closure maximum `1.4901161193847656e-08`. Additive ThermalChannel/Stage-A
  smoke: field `(1,32,5)`, edge fields `(1,32,6,5)`, exact reported closure
  `0.0`, and `interface_source=local_surrogate`.
- Full command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q tests Case_ThermalChannel/tests`
  -> `135 passed, 1 skipped in 11.74s` (the same optional artifact skip).
- Compile validation, 25 JSON parses (tracked files plus the untracked upgrade
  profile), and `git diff --check` passed.
- Run 0002 strict state loading still reported no missing/unexpected keys; all
  four frozen evaluation arrays remained bit-for-bit equal with maximum
  absolute difference `0.0`.
- No training or generated checkpoint run was launched.

---

## Goal 3 — Implement exchangeable candidate slots and adaptive selection

### Objective

Remove learned edge identity and make active count design-dependent.

### Tasks

1. Implement anonymous slot-code generation.
2. Implement shared iterative module/environment slot refinement.
3. Implement raw quality, coverage, novelty, active mask, and renormalization.
4. Add training-progress warmup.
5. Add raw and selected organizer diagnostics.
6. Confirm model parameter shapes are independent of `edge_capacity`.

### Tests

- candidate-code permutation equivariance;
- module permutation invariance;
- padding-width invariance;
- runtime `K_cap=6` and `K_cap=10` use the same learned parameter shapes;
- active count stays in valid bounds;
- every active module/environment row retains unit assignment mass;
- inactive edges have zero selected mass;
- no NaN when one edge is selected;
- no NaN with zero inactive padded modules.

### Acceptance

- The exchangeable organizer runs under dense softmax first.
- A smoke batch shows non-identical candidate supports and finite edge quality.
- Do not add sparsity transforms until this goal passes.

### Completion summary — 2026-08-17

- Status: accepted.
- Added an `exchangeable_slots` organizer beside the unchanged fixed
  projection branch. Candidate states use case-conditioned shared base/scale
  maps plus deterministic zero-mean sinusoidal or low-discrepancy codes; there
  are no embeddings or learned parameters indexed by candidate edge.
- Shared module/environment query, key, value, GRU update, and normalization
  parameters perform two refinement iterations by default. Runtime edge
  capacity is mutable without changing any parameter/state shape.
- Added detached assignment-purity quality, code-equivariant tie breaking,
  novelty filtering, module/environment threshold coverage, warmup top-count
  selection, active masking, and row renormalization. No edge-count objective
  or count head was added.
- Core and ThermalChannel APIs expose runtime capacity and once-per-epoch
  training progress. The forward workflow sets progress before each epoch;
  fixed projection ignores both APIs.
- Raw/selected incidences, candidate/selected states, quality, active mask and
  count, coverage, scales, and purities are available in organizer diagnostics.
- Focused command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q tests/test_exchangeable_organizer.py tests/test_forward_additive.py tests/test_forward_upgrade_config.py tests/test_core_contract.py Case_ThermalChannel/tests/test_forward_variable_modules.py Case_ThermalChannel/tests/test_dynamic_collation.py tests/test_runtime_services.py`
  -> `54 passed in 4.78s`.
- Exchangeable-only focused command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q tests/test_exchangeable_organizer.py tests/test_forward_additive.py tests/test_core_contract.py`
  -> `33 passed in 5.56s`.
- CUDA synthetic smoke selected `[3,3]` active edges from six, reached module
  and environment coverage `[1.0,1.0]`, produced candidate-support standard
  deviation `0.0143903559`, finite quality, and closure `1.4901161193847656e-08`.
  A ThermalChannel/Stage-A CUDA smoke also completed at candidate capacity 8
  with field `(1,32,5)`, edge width 8, finite output, and exact reported
  closure `0.0`.
- Full command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q tests Case_ThermalChannel/tests`
  -> `142 passed, 1 skipped in 14.39s` (the same optional artifact skip).
- Compile validation and `git diff --check` passed. Run 0002 strict loading
  remained clean and all four frozen arrays remained bit-for-bit equal with
  maximum absolute difference `0.0`.
- Known boundary: exchangeable assignments and query routing are intentionally
  dense softmax in this accepted stage; entmax/locality begins only in Goal 4.
- No training or generated checkpoint run was launched.

---

## Goal 4 — Add sparse localized incidence and query routing

### Objective

Produce exact-zero, spatially coherent, overlapping interaction supports.

### Tasks

1. Add tested `entmax15`.
2. Enable it independently for module, environment, and query routing.
3. Add source/region dispersion and compact geometry bias.
4. Ensure active masks are applied before final normalization.
5. Add sparsity/locality diagnostics.

### Tests

- exact zeros occur on a controlled logits example;
- row normalization and masking are correct;
- gradients are finite;
- two nearby edges may overlap;
- distant irrelevant edge routes become zero in a synthetic geometry test;
- periodic coordinate tests still pass.

### Acceptance

- Dense execution still works.
- Sparsity is observable in diagnostics.
- No efficiency claim is made yet.

### Completion summary — 2026-08-17

- Status: accepted.
- Added dependency-free `entmax15` and strict softmax/entmax assignment
  dispatch with exact mask zeros, unit valid-row mass, zero all-masked rows,
  finite gradients, half-precision float32 working arithmetic, and matching
  CPU/CUDA results.
- Module, environment, and query normalizers are independently configurable.
  Existing context-fusion configs still take their original softmax branch.
- Exchangeable environment refinement now uses a broad source-centered first
  region followed by previous-region anisotropic compact-kernel bias. Query
  routing uses the selected region centroids/scales with periodic minimum-image
  offsets and applies the active mask before normalization.
- Added candidate/selected module and environment nonzero fractions plus query
  nonzero fraction and mean nonzero edges. Probability sparsity remains
  explicitly separate from `routing_execution=dense`.
- Focused command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q tests/test_sparse_routing.py tests/test_exchangeable_organizer.py tests/test_forward_additive.py tests/test_core_contract.py`
  -> `42 passed in 3.77s`.
- Focused cases cover controlled exact zeros, masked normalization, finite
  gradients, CPU/CUDA parity, overlapping nearby edges, distant-edge zero
  routing, periodic wraparound, and finite model backward.
- Adaptive ThermalChannel/Stage-A CUDA smoke at `Q=64`: field `(1,64,5)`,
  eight candidates, seven active edges, environment nonzero fraction
  `0.6028646231`, query nonzero fraction `0.48828125`, mean nonzero query edges
  `3.90625`, `execution=dense`, exact reported closure `0.0`, and finite
  output. The random untrained module assignments were dense; a controlled
  scaled-logit model test verifies exact-zero module entmax routing.
- Full command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q tests Case_ThermalChannel/tests`
  -> `151 passed, 1 skipped in 12.70s` (the same optional artifact skip).
- Compile validation, 25 JSON parses, and `git diff --check` passed. Run 0002
  strict loading remained clean and all four frozen arrays remained
  bit-for-bit equal with maximum absolute difference `0.0`.
- The adaptive profile deliberately remains `routing_execution=dense` at this
  boundary. No latency, memory, or computational-sparsity claim is made.
- No training or generated checkpoint run was launched.

### Post-acceptance locality correction — 2026-08-18

- Status: implemented for validation after static diagnosis of the early adaptive runs.
- The adaptive profile no longer compounds compact-support log-biases during environment construction and query routing. Both sites now use the same bounded Gaussian log-bias, `b = -lambda * min(radius_squared, radius_cap_squared) / 2`, with `environment_locality_strength=1.0` and `locality_radius_cap=3.0`; entmax remains the sole learned probability transform that creates exact-zero support.
- `compact_kernel` remains a strict accepted mode with its previous formula for configurations that explicitly selected it. Missing locality fields still resolve to `none`, so fixed-projection/context-fusion configurations and checkpoint module paths are unchanged.
- The forward upgrade ladder and sparse-routing benchmark now exercise the bounded Gaussian setting. Unit coverage verifies exact bounded values, finite positive pre-entmax kernel weights, interior gradients, and unchanged compact-kernel arithmetic.
- Focused command: `CUDA_VISIBLE_DEVICES=0 pytest -q tests/test_sparse_routing.py tests/test_forward_upgrade_config.py tests/test_gathered_routing.py tests/test_exchangeable_organizer.py tests/test_forward_additive.py` in the `ModularDT` environment -> `45 passed in 2.50s`.
- Full command: `CUDA_VISIBLE_DEVICES=0 pytest -q` in the `ModularDT` environment -> `178 passed, 1 skipped in 8.23s`; the skip is the unavailable optional local inverse integration artifact.
- Adaptive dry-run with the trusted local Stage-A checkpoint validated the strict config, dataset, checkpoint, GPU 0, and output resolution without starting a run. The five-stage CUDA correctness ladder completed with finite outputs, zero/roundoff additive closure, bounded-Gaussian modes in stages D/E, no training steps, and no written output artifact.
- No training was launched and no checkpoint or generated run artifact was created.

---

## Goal 5 — Add gathered sparse execution

### Objective

Turn learned topology into actual pairwise and edge-head compute savings.

### Tasks

1. Compute hypergraph-induced query-module relevance \(\beta\).
2. Gather top routed modules before the pair MLP.
3. Gather top routed edges before the edge field head.
4. Add dense/gathered modes.
5. Add runtime counters and benchmark tool.

### Tests

- gathered equals dense when limits include all modules/edges;
- gathered does not allocate a dense pair embedding tensor;
- backward pass is finite;
- masked/inactive modules can never be selected;
- query chunks match one-shot output.

### Acceptance

- On synthetic `M>=64`, gathered mode reduces measured pair-MLP work and peak memory.
- Report latency honestly; do not make gathered mode the profile default if it is slower at the intended workloads.

### Completion summary — 2026-08-17

- Status: accepted.
- Added a true gathered pairwise path: hypergraph-induced query-module
  relevance is computed first, inactive/padded modules are excluded, and only
  the selected ragged module set is materialized before `pair_mlp`.
- Added a true gathered additive-edge path: active nonzero query-edge routes
  are selected before the shared edge head, flattened for execution, and
  scattered directly into the summed field. The dense per-edge field tensor
  is only constructed when explicitly requested.
- Added honest execution diagnostics for available, selected, and evaluated
  pair/edge routes. Probability sparsity and gathered execution remain
  distinct concepts; dense mode retains its original computation.
- Added `tools/benchmark_honf_sparse_routing.py` with bounded defaults and an
  opt-in full workload matrix over module counts `(12,32,64,128)`, queries
  `(1024,8192,32768)`, and capacities `(6,8,12,16)`.
- Focused command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q tests/test_forward_upgrade_config.py tests/test_gathered_routing.py tests/test_sparse_routing.py tests/test_exchangeable_organizer.py tests/test_forward_additive.py`
  -> `40 passed in 2.25s`.
- At `M=64,Q=8192,K=8`, hidden width 64, limits 8 modules/3 edges, and ten
  prepared-decode repetitions, dense versus gathered median latency was
  `12.972944 ms` versus `5.451776 ms`; incremental peak CUDA memory was
  `667,827,712` versus `116,146,176` bytes; evaluated pair contexts were
  `524,288` versus `65,536`; evaluated edge routes were `65,536` versus
  `22,728`.
- The small `M=64,Q=1024,K=8` workload was reported honestly: gathered was
  slower (`5.122592 ms` versus `4.823040 ms`) despite reducing incremental
  peak memory (`16,220,160` versus `85,196,288` bytes). At the intended larger
  query workload it is both faster and smaller, so only the explicit adaptive
  profile now selects `routing_execution=gathered`; missing fields and the
  existing profile still resolve to `dense`.
- Full-limit (`M=12,Q=1024,K=8`, limits 12/8) gathered and dense fields agreed
  to `5.9604645e-08` maximum absolute and `6.5757490e-08` relative L2 error.
- Full command: `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q`
  -> `156 passed, 1 skipped in 10.90s` (the unchanged optional local inverse
  integration-artifact skip).
- Compile validation, 25 JSON parses, and `git diff --check` passed. Run 0002
  loaded strictly with no missing/unexpected keys, and all five frozen NPZ
  arrays were bit-for-bit equal with maximum absolute difference `0.0`.
- No training was launched. Compatibility evaluation artifacts were written
  outside the repository under `/tmp`; no generated artifact was added to
  version control.

---

## Goal 6 — Export and evaluate the topology signature

### Objective

Make topology a stable, testable design characteristic.

### Tasks

1. Add schema and exporter.
2. Add static edge tokens and edge relations.
3. Add reference-query routing/contribution summaries.
4. Add matching-based topology distance.
5. Add visualization:
   - active source/region ellipses;
   - module/environment membership;
   - edge overlap graph;
   - per-field contribution maps.
6. Add optional relation-reconstruction metrics from ThermalChannel structure targets.

### Tests

- schema round trip;
- permutation-invariant comparison;
- canonical serialization is deterministic;
- `active_module_count` is distinct from padded slot count;
- routed summaries are independent of query chunk size;
- topology distance is zero for an edge permutation.

### Acceptance

- One evaluation case produces the complete topology artifact.
- Existing plan export remains available for its original consumers.

### Completion summary — 2026-08-17

- Status: accepted; the forward topology schema gate is closed before any
  inverse-model change.
- Added the generic unordered `honf_topology_signature` schema version 3 with
  masked static edge tokens, module/environment incidence, optional candidate
  incidence, reference-query routing summaries, per-output-channel
  contribution summaries, nine edge-relation channels, explicit padded and
  active module counts, candidate/active edge counts, query-measure digest,
  case/checkpoint provenance, and the retained serialization permutation.
- Canonical ordering is used only for deterministic serialization/display.
  Active topology comparison uses Hungarian token matching, an explicit
  unmatched-edge cost, and matched relation error; a pure edge permutation has
  exactly zero distance in tests. No canonical edge position is exposed as a
  training target.
- Added generic module-affinity and query-to-module relation reconstruction
  plus masked MSE/relative-L2 diagnostics. ThermalChannel passes its own solved
  or fallback targets and records that source; no topology loss or edge-count
  penalty was enabled.
- Added opt-in `--export-topology-signature` evaluation support and active
  source/region ellipses, module/environment membership views, an edge-overlap
  graph, and per-field edge contribution maps when edge-additive outputs are
  available. Large route and edge-field maps remain opt-in. Added
  `tools/compare_honf_topologies.py` for matching-based artifact comparison.
- The existing schema-v2 `hypergraph_plan` implementation and CLI option were
  not changed. Focused compatibility tests for its inverse-dataset consumers
  passed alongside the new schema.
- Focused command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q tests/test_topology_signature.py tests/test_gathered_routing.py tests/test_sparse_routing.py tests/test_forward_additive.py tests/test_forward_upgrade_config.py Case_ThermalChannel/tests/test_topology_signature_visualization.py Case_ThermalChannel/tests/test_inverse_compact_plan.py Case_ThermalChannel/tests/test_inverse_dataset_builder.py Case_ThermalChannel/tests/test_resources_and_checkpoint.py`
  -> `55 passed in 12.64s`.
- Run 0002 case `0273` was evaluated on CUDA device 0 with both export flags.
  The schema-v3 artifact records 12 padded slots versus 3 active modules, six
  active edges, 8,192 deterministic evaluation-grid probes and their SHA-256
  digest, solved-target relation metrics, checkpoint SHA-256 provenance, and
  three static topology figures. Its context-fusion checkpoint correctly marks
  per-field edge contributions unavailable; complete edge-additive
  contribution summaries and the fourth map are covered by focused tests.
- The comparison command on the exported artifact against itself returned
  topology, feature, and relation distances `0.0` with no unmatched edges.
- Full command: `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q`
  -> `161 passed, 1 skipped in 15.17s` (the unchanged optional local inverse
  integration-artifact skip).
- Compile validation, 25 JSON parses, and `git diff --check` passed. All five
  Run 0002 frozen arrays remained bit-for-bit equal with maximum absolute
  difference `0.0`.
- Evaluation artifacts were written under `/tmp`; no generated artifact or
  training run was added to the repository. The schema contract and CLI are
  documented in `docs/topology_signature.md`.

---

## Goal 7 — Run the controlled forward-model evaluation ladder

### Models

Use a narrow ladder, not a Cartesian product:

| ID | Organizer | Mechanism | Field | Routing | Execution |
|---|---|---|---|---|---|
| A | fixed projection | residual concat | context fusion | softmax | dense |
| B | fixed projection | descriptor first | edge additive | softmax | dense |
| C | exchangeable slots | descriptor first | edge additive | softmax | dense |
| D | exchangeable slots | descriptor first | edge additive | entmax/localized | dense |
| E | exchangeable slots | descriptor first | edge additive | entmax/localized | gathered |

### Evaluation order

1. Unit and contract tests.
2. One-batch CPU/CUDA smoke.
3. Bounded equal-budget diagnostic run.
4. Full equal-budget training only for A and the best upgraded candidate.
5. Module-count and geometry extrapolation.
6. Stability/intervention audit.
7. Efficiency benchmark.

### Required metrics

#### Forward accuracy

- total validation objective;
- field MSE;
- physical RMSE and relative \(L_2\) per channel;
- temperature MSE;
- internal-temperature error;
- interface error;
- autonomous port metrics.

#### Additive structure

\[
\epsilon_{\mathrm{closure}}
=
\frac{
\|\widehat U-U^{bg}-\sum_kU^k\|_2
}{
\|\widehat U\|_2+\varepsilon
}.
\]

Report:

- background/edge norm ratio;
- edge contribution by field channel;
- active edge contribution concentration.

#### Topology sparsity and complexity

- `K_active` distribution versus module count;
- nonzero density of \(A^{MH}\), \(A^{EH}\), and \(\alpha\);
- source and region dispersion;
- module/environment overlap;
- dead-candidate fraction;
- quality and coverage.

#### Stability

After optimal matching, compare topology under:

- module-slot permutation;
- extra inactive padding;
- small center/heat perturbation;
- different query samples;
- neighboring checkpoints;
- independent training seeds.

#### Faithfulness

1. Decoder faithfulness is exact by additive construction.
2. Module intervention:
   - remove or perturb modules strongly assigned to one edge;
   - rerun the full model;
   - compare the realized field change with that edge's contribution map.
3. Descriptor intervention:
   - shuffle source/region descriptors between edges;
   - verify prediction and routing degrade.
4. Incidence intervention:
   - shuffle or uniformize \(A^{MH}\) and \(A^{EH}\);
   - measure field degradation.

#### Efficiency

- latency;
- throughput;
- peak memory;
- evaluated pair count;
- evaluated edge-route count;
- approximation error versus dense routing.

### Acceptance guidance

Hard software gates:

- old mode reconstructs exactly;
- additive closure is within floating-point tolerance;
- invariance/equivariance tests pass;
- topology schema is deterministic;
- full-limit gathered routing matches dense routing.

Scientific gates:

- upgraded equal-budget field error should not regress materially;
- `K_active` must not collapse to one value without evidence that the dataset requires it;
- active edges must carry nontrivial field contribution;
- intervention tests must show that topology affects the output;
- speedup claims require measured latency and memory improvements.

Do not hide a small accuracy tradeoff. Report it against gains in interpretability, topology stability, and compute.

### Completion summary — 2026-08-17

- Status: accepted for the user-authorized correctness scope. Scientific model
  selection, equal-budget optimization, extrapolation accuracy, and trained
  intervention claims remain explicitly unmeasured because this task forbids
  training runs.
- Added `tools/check_forward_upgrade_ladder.py`, which instantiates the narrow
  A–E architecture ladder on one deterministic batch, performs no optimizer or
  backward step, and reports modes, finiteness, parameter counts, active edges,
  routing density, actual evaluated-route counters, additive closure, and the
  gathered approximation difference from the identical-weight dense sparse
  variant.
- GPU 0 command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT python tools/check_forward_upgrade_ladder.py --device cuda:0 --modules 12 --active-modules 7 --queries 256 --edge-capacity 8 --hidden-dim 32`
  -> exit 0, `training_steps=0`, and finite `(1,256,5)` fields for A–E.
- B, C, and D had exact reported additive closure `0.0`; E closure was
  `2.9802322e-08`. D's entmax/localized query density was `0.4951172` with
  `3.9609375` mean nonzero edges. With the same D weights, E evaluated 1,024
  pair contexts versus 3,072 and 667 edge routes versus 2,048. The deliberately
  limited gathered result differed from dense by `0.1266563` maximum absolute
  and `0.1750441` relative L2, so no accuracy-equivalence claim is made for
  limited routing.
- Focused command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q tests/test_tools.py tests/test_forward_upgrade_config.py tests/test_forward_additive.py tests/test_exchangeable_organizer.py tests/test_sparse_routing.py tests/test_gathered_routing.py tests/test_topology_signature.py`
  -> `49 passed in 4.86s`.
- Full command: `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q`
  -> `162 passed, 1 skipped in 22.05s` (the unchanged optional local inverse
  integration-artifact skip).
- Compile validation and `git diff --check` passed. This phase adds only the
  bounded diagnostic tool and its test; it changes no checkpoint/module path,
  forward configuration default, or saved-model behavior.
- No training, generated artifact, or scientific quality claim was produced.

---

## Goal 8 — Remove fixed edge identity from the inverse model

Start only after Goal 6 is accepted.

### 8.1 Plan flow

Add:

```text
plan_token_mode = indexed
plan_token_mode = exchangeable_set
```

For `exchangeable_set`:

- remove `edge_embedding`;
- use the rectified-flow noise state itself as token differentiation;
- use shared token blocks;
- add a small permutation-equivariant interaction block:
  - pooled global set context, or
  - self-attention without positional edge embeddings;
- support runtime `K_cap`;
- produce an activity mask compatible with the forward topology signature.

### 8.2 Layout conditioning

Add:

```text
plan_conditioning_mode = ordered_flat
plan_conditioning_mode = set_cross_attention
```

For `set_cross_attention`:

- remove the flattened `Linear(K*F,H)` path;
- encode each active topology token with a shared encoder;
- pool mean/max for global conditioning;
- let layout tokens cross-attend to active topology tokens;
- mask inactive topology tokens;
- make behavior invariant to edge permutation.

### 8.3 Training target

The inverse target is the active topology set, not canonical edge positions.

Use matching:

- Hungarian for evaluation;
- Sinkhorn or another differentiable matching for training;
- null/dummy edges for unequal active counts;
- optional relation consistency after token matching.

Do not use canonical order as supervision.

### 8.4 Dataset and checkpoint compatibility

- Keep the current compact-plan schema and inverse mode for its checkpoints.
- Add a topology-set dataset schema for the upgraded forward checkpoint.
- Tie the inverse dataset to the exact topology schema and forward checkpoint hash.
- Do not silently mix old compact plans with the new topology tokens.

### Tests

- plan velocity equivariant to edge permutation;
- layout distribution invariant to plan-edge permutation;
- runtime edge capacity does not change parameter shapes;
- matching loss is zero for a pure permutation;
- old inverse checkpoint reconstructs its indexed/ordered modes.

### Completion summary — 2026-08-17

- Status: accepted. Work began only after Goal 6 accepted the forward topology
  schema.
- Added strict `plan_token_mode=indexed|exchangeable_set` and
  `plan_conditioning_mode=ordered_flat|set_cross_attention`. Missing fields
  resolve to indexed/canonical/ordered behavior and instantiate the original
  `edge_embedding`, `ordered_plan_projection`, and state-dict paths exactly.
- The exchangeable plan flow has no edge-index embedding. Noisy flow state is
  token differentiation; shared MLP and permutation-equivariant self-attention
  blocks predict velocity/activity. Runtime edge capacity can change without
  changing any parameter shape.
- Set layout conditioning has no flattened edge projection. It uses shared
  topology-token encoding, active-mask mean/max pooling, and masked
  cross-attention from layout slots to topology tokens. Plan-edge permutation
  leaves layout velocity, presence, and count outputs invariant within floating
  tolerance.
- Added Sinkhorn assignment to the set-mode flow-matching target and Hungarian
  set evaluation/loss support, including implicit null tokens for unequal
  capacities. Set mode rejects canonical or Hungarian training supervision;
  canonical ordering remains only in endpoint serialization.
- Added a ThermalChannel topology-set contract that converts accepted
  signature-v3 exports to unordered 12-value inverse tokens and carries the
  relation tensor. Added a distinct HDF5 schema name and validation path bound
  to `honf_topology_signature` v3 and an exact forward checkpoint SHA-256.
  Training rejects compact-plan/set-mode mismatches and validates schema/hash
  equality rather than silently mixing datasets.
- The fixed-width joint corrector is explicitly unavailable for exchangeable
  mode because it contains edge-indexed flattened parameters. No replacement
  edge-indexed parameter, edge-count penalty, optimizer step, or training run
  was added.
- Added a mode-complete
  `src/config_core/inverse/train_inverse_topology_set_template.json`. Its
  all-zero checkpoint digest is a deliberate non-runnable placeholder and must
  be replaced by the dataset/checkpoint digest; runtime validation rejects it.
- Focused command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q tests/test_inverse*.py tests/test_plan_flow.py tests/test_layout_flow.py tests/test_joint_corrector.py Case_ThermalChannel/tests/test_inverse*.py`
  -> `62 passed, 1 skipped in 10.66s` (the optional local inverse integration
  artifact is unavailable).
- CUDA correctness smoke: plan permutation equivariance maximum difference
  `3.5762787e-07`, layout plan-permutation invariance maximum difference
  `1.4901161e-07`, finite gradients, runtime K=7 output `(1,7,10)`, unchanged
  K=4→7 parameter shapes, and no edge embedding/ordered-plan projection.
- Full command: `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q`
  -> `173 passed, 1 skipped in 13.11s`.
- Compile validation, 26 JSON parses, and `git diff --check` passed. Synthetic
  checkpoint tests establish bitwise-equivalent missing/explicit indexed modes,
  strict existing state-key loading, and schema/hash-complete exchangeable
  checkpoint round trips. No prior inverse checkpoint artifact was present
  under `Trained_Results/ThermalChannel/HONF_Inverse_Runs` for an additional
  real-checkpoint test.
- Forward modules/configuration were not changed in this phase; the frozen
  Run 0002 path therefore remains the same fixed-projection/context-fusion
  path established by the prior bitwise gates.

---

## Final verification sweep — 2026-08-17

- Status: accepted; all requested implementation phases and feasible gates are
  complete.
- Final full command:
  `CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q`
  -> `173 passed, 1 skipped in 14.61s`. The only skip remains the optional
  local inverse integration test whose external artifacts are unavailable.
- Final static commands:
  `rtk conda run -n ModularDT python -m compileall -q src Case_ThermalChannel/src tests tools train.py evaluate.py`,
  the 26-document JSON parser, and `rtk git diff --check` all exited 0.
  The optional `jsonschema` package is not installed; strict runtime config
  constructors successfully resolved both inverse templates after substituting
  a non-placeholder test digest.
- Final Run 0002 CUDA evaluation on case `0273` loaded the checkpoint strictly
  with `missing=[]` and `unexpected=[]`. Field, ground-truth, internal
  temperature, interface, and port-condition arrays were all bit-for-bit equal
  to Goal 0 with maximum absolute difference `0.0`.
- No long training, optimizer step, committed generated artifact, or default
  edge-count penalty was introduced. Evaluation outputs remain under `/tmp`;
  ignored Python bytecode is not part of the change set. The pre-existing root
  `.gitignore` modification was preserved without editing it.
- Compatibility behavior: saved forward configs missing upgrade modes select
  fixed projection, residual concatenation, context fusion, softmax, and dense
  execution; saved inverse configs missing set modes select indexed tokens,
  canonical matching, and ordered-flat conditioning. The explicit adaptive
  profiles select gathered sparse additive forward execution and
  exchangeable-set/set-cross-attention inverse execution respectively.

Changed implementation/configuration files:

```text
Case_ThermalChannel/src/channelthermal/model.py
Case_ThermalChannel/src/channelthermal/workflows/train_forward.py
Case_ThermalChannel/src/channelthermal/workflows/evaluate_forward.py
Case_ThermalChannel/src/channelthermal/evaluation_tools/topology_signature_visualization.py
Case_ThermalChannel/src/channelthermal/inverse/__init__.py
Case_ThermalChannel/src/channelthermal/inverse/dataset_io.py
Case_ThermalChannel/src/channelthermal/inverse/topology_set.py
Case_ThermalChannel/src/channelthermal/workflows/train_inverse_hierarchical.py
src/config_core/forward/adaptive_sparse_additive.json
src/config_core/inverse/train_inverse_topology_set_template.json
src/config_core/schemas/core_config.schema.json
src/config_core/schemas/inverse_hierarchical_config.schema.json
src/honf_forward_core/config.py
src/honf_forward_core/routing.py
src/honf_forward_core/organizer.py
src/honf_forward_core/decoder.py
src/honf_forward_core/model.py
src/honf_forward_core/evaluation/__init__.py
src/honf_forward_core/evaluation/topology_signature.py
src/honf_inverse_core/README.md
src/honf_inverse_core/models/hierarchical_inverse.py
src/honf_inverse_core/models/plan_flow.py
src/honf_inverse_core/models/layout_flow.py
src/honf_inverse_core/models/matching.py
src/honf_inverse_core/training/trainer.py
src/honf_inverse_core/training/checkpointing.py
tools/benchmark_honf_sparse_routing.py
tools/check_forward_upgrade_ladder.py
tools/compare_honf_topologies.py
```

Changed documentation/tests:

```text
docs/forward_upgrade_reference.md
docs/migration.md
docs/topology_signature.md
tests/test_forward_upgrade_config.py
tests/test_forward_additive.py
tests/test_exchangeable_organizer.py
tests/test_sparse_routing.py
tests/test_gathered_routing.py
tests/test_topology_signature.py
tests/test_plan_flow.py
tests/test_layout_flow.py
tests/test_inverse_config.py
tests/test_inverse_matching.py
tests/test_inverse_checkpointing.py
tests/test_tools.py
Case_ThermalChannel/tests/test_topology_signature_visualization.py
Case_ThermalChannel/tests/test_inverse_topology_set.py
Case_ThermalChannel/tests/test_inverse_dataset_builder.py
```

This goal-plan file is ignored by Git but was updated after every phase as
requested.

---

## 14. Additional code-review hardening

These are small and should not distract from the main upgrade.

1. In dynamic collation, explicitly zero every inactive module slot after compaction rather than relying entirely on source padding.
2. In topology exports, distinguish:
   - padded/runtime slot width;
   - active module count.
3. Keep structure-target semantics case-owned.
4. Do not put channel-temperature assumptions back into core losses.
5. Replace new public compatibility labels with behavior-based names; accept any already-saved strings as parser aliases.
6. Add no new dense diagnostic tensor to standard training output.
7. Make all large routing maps opt-in.

---

## 15. Literature-derived design rationale

The implementation should cite and acknowledge these ideas in documentation, without claiming that HONF is equivalent to any one method.

1. **Slot Attention**, NeurIPS 2020, arXiv:2006.15055  
   Exchangeable latent slots specialize through iterative competitive attention.

2. **Adaptive Slot Attention**, CVPR 2024, arXiv:2406.09196  
   Uses candidate slots, design-dependent slot count, and masked decoding.

3. **QASA**, arXiv:2601.12936, 2026 preprint  
   Warns that a count penalty can conflict with reconstruction and motivates quality-guided selection decoupled from the reconstruction loss.

4. **When Slots Compete: Slot Merging**, arXiv:2603.11246, 2026 preprint  
   Uses overlap-based merging as a lightweight response to redundant slots.

5. **Dynamic Hypergraph Neural Networks**, IJCAI 2019  
   Reconstructs hypergraph structure dynamically from learned representations.

6. **Totally Dynamic Hypergraph Neural Networks**, IJCAI 2023  
   Explicitly addresses adjustable hyperedge number.

7. **You Are AllSet**, ICLR 2022, arXiv:2106.13264  
   Frames hypergraph propagation as compositions of learnable multiset functions.

8. **Deep Sets**, NeurIPS 2017, and **Set Transformer**, ICML 2019  
   Provide the basic permutation-invariant/equivariant design principles needed for unordered topology tokens.

9. **Sparsemax**, ICML 2016, and **Adaptively Sparse Transformers**, EMNLP 2019  
   Show differentiable attention transformations that produce exact zeros.

10. **Learning Sparse Neural Networks through \(L_0\) Regularization**, ICLR 2018  
    Provides hard-concrete gates as an optional structural-learning tool, not the recommended first default here.

11. **DSelect-k**, NeurIPS 2021, and **Differentiable Top-k with Optimal Transport**, NeurIPS 2020  
    Provide optional differentiable subset-selection mechanisms.

12. **Spatial Mixture-of-Experts**, NeurIPS 2022  
    Demonstrates learned spatial routing and the importance of routing-specific training.

13. **Multipole Graph Neural Operator**, NeurIPS 2020  
    Motivates separating local and long-range interactions while obtaining real computational scaling.

14. **Learning Latent Permutations with Gumbel-Sinkhorn Networks**, ICLR 2018  
    Supports permutation-aware matching for the later inverse topology set.

---

## 16. Deliverables required from Codex

At the end of each goal, report:

```text
Goal completed
Files changed
Public configuration changes
Checkpoint-compatibility behavior
Tests run and exact result
Smoke commands run and exact result
Known limitations
Next goal
```

Final deliverables:

1. upgraded source;
2. strict config schemas;
3. new `adaptive_sparse_additive.json` profile;
4. unit and integration tests;
5. topology signature exporter;
6. sparse-routing benchmark tool;
7. documentation updates;
8. a migration note explaining mode inference;
9. no generated training runs or checkpoint binaries committed.
