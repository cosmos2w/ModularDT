# HONF Round5 — scene-prepared moment hyperedges

**Purpose:** implement and test a richer, reusable hyperedge representation without restoring the nonlocal query–source value MLP. **Mode:** bounded Codex Goal-mode development, followed only by qualifying, separately identified full-TRAIN e1000 probes. **Reviewed repository:** `cosmos2w/ModularDT`, branch `agent/honf-core-next`, commit `8426e12f07b07acbe744d13a80ef2abb4edd5073`. **Evidence:** the complete `HONF_Attention_Inherited_Gathering_Round4_Report.md`, the preserved Round3 report and Round2 formal results, and the current inherited-attention implementation. This document proposes work; it does not report new native training or GPU results.

## 0. The decision in plain language

Keep the successful idea of preparing information once per scene. Improve what that prepared information contains, and remove avoidable query-side work. Do not launch another sparse-normalization campaign, train another utility selector, or restore an all-source cheap path plus selected expensive corrections.

The primary experiment has three versions of one architecture: **C0**, the same-function optimized IAG-C soft control; **M**, a moment-enriched representation with an affine hyperedge processor; and **H**, the same moments and dimensions with a nonlinear hyperedge processor. Thermal and Wind use the same computational implementation and separate learned weights/native adapters. An optional first-order spatial reconstruction is the only conditional architectural revision. Member feedback, extra contextualization layers, adaptive anchor counts and loss-balancing campaigns are deferred.

The main question is: **Can spatially richer, collectively processed hyperedges recover the information lost by simple averaging, while retaining the memory advantage and making complete calls economical?** A second question is whether nonlinear collective processing improves on equally equipped linear processing. Nonzero attention, nonlinear responses and attractive graphs alone do not answer either question.

### Non-negotiable boundaries

- Preserve A2-D/D0, Round2–4, all historical scientific checkpoints, original source data, native transforms, sampler identities and solver ledger `326/326`. No new solver calls, inverse generator or Wind TEST access.
- Keep two staged `MM → EM → EE → ME` contextualization cycles, width128/four heads, existing reinjection and compact local reader. No frozen D-sep, Stage-A, H6 bank, nonlocal pair-value bypass or second complete predictor.
- Keep Thermal heat outside the scene encoder, memberships, moments and nonlinear hyperedge processor. Retain the zero-intercept law `T(X,h)=K(X)h`; Wind remains nonlinear with its native profile/residual transform.
- Separate a mathematical same-function optimization, a different trained architecture, and a physically validated predictor. Never use one kind of evidence as another.
- Preserve current dataset memberships and native objectives. Do not combine these architectural changes with a wider network, a new environment grid, a different loss, a new split or a new query budget.

## 1. What Round4 actually established

### 1.1 Read the dataset-specific results correctly

The mature Thermal models are **IAG-C**; the mature Wind models are **IAG-Q**. Wind-C stopped at e500 after a fixed-TRAIN capacity diagnosis; Q was a fresh replacement, not a continuation of C. There is no mature Wind-C/e2500 result to reuse as though it already exists.

| Exposed native endpoint, literal e2500 | Round3 D0 | Round4 soft | Meaning |
|---|---:|---:|---|
| Thermal fluid T mean RMSE | 0.87718 | 0.86976 | Approximately preserved, not the principal soft-C failure |
| Thermal transverse v | 0.0030744 | 0.0043575 | About 41.7% worse |
| Thermal pressure p | 0.011296 | 0.016620 | About 47.1% worse |
| Thermal q proxy | 3.2970 | 2.5978 | About 21.2% better |
| Wind near-turbine vector RMSE, m/s | 0.234814 | 0.743510 | Large Q-model failure |
| Wind downstream vector RMSE, m/s | 0.171749 | 0.558821 | Large Q-model failure |
| Thermal fixed-B8/Q8192 allocated peak, GiB | 3.011 | 1.161 | About 61.5% lower |
| Wind fixed-B8/Q8192 allocated peak, GiB | 5.994 | 2.315 | About 61.4% lower, measured for Q |

Use full tables and raw records rather than these rounded values for new comparisons. Thermal values are packed-native units, not established Kelvin/SI pressure/heat flux. Wind velocity is m/s. Fixed-B8 stress memory is not a full optimizer/auxiliary training peak. Complete high-M calls were 16.8% slower for Thermal soft and 36.3% slower for Wind soft; sparse variants did not repair this. Both prepared-new-query and prepared-heating scopes were also slower in the report. Historical training-time sums are not newly controlled same-GPU speed comparisons.

Soft Thermal preserved temperature and improved saved material-response error, but its flow fit deteriorated. Wind-Q had both a fixed-TRAIN fitting deficit and worsening exposed-DEV near/downstream errors after early checkpoints. More epochs alone are not the proposed remedy. Association interventions demonstrate computational use, not sufficiency of the representation, uniquely useful group assignments or physical causality.

### 1.2 Important code corrections

The exact projection opportunity is in `InheritedAttentionCoreV1._decode_features`: C gathers `mc` and `ec` separately, then calls `gather_output(torch.cat((mc, ec), dim=-1))`. The attribute is `gather_output`, not a hypothetical replacement reader. This is a 256→128 affine map under the current width.

**Thermal already connects gathered context to temperature.** `response_phi` receives `cat(qs, global_context, local)`, as does the nonlinear field head. Do not diagnose a missing context-to-temperature connection. The existing source carrier depends on contextualized modules; enriching collective context and adding source-resolved spatial heat moments are proposed capacity changes, not bug repairs.

The C projection can move before cover gathering. The same step does **not** apply unchanged to Q, whose final weights depend on the query and differ between module/environment donor types. Preserve Q as a historical comparison; the new family deliberately studies a richer C-style cached representation for both datasets.

## 2. Source audit and implementation boundaries

All paths below are relative to `HONF_Proj/` unless stated otherwise. They identify inspected files/functions at the reviewed commit. Resolve additional native-adapter and TrainingEngine call sites by searching their actual imports; do not invent another training stack.

| Source | Observed behavior | Required action |
|---|---|---|
| `src/honf_forward_core/inherited_attention/model.py`, `InheritedAttentionCoreV1.prepare` | Ordered context, last EM/EE score export, final typed values, separate C aggregates; Q instead prepares keys | Reuse context unchanged. Introduce small payload-preparation hooks and a versioned moment-hyperedge extension, not a copied 55-kB core |
| Same file, `_decode_features` | Two indexed C gathers followed by `gather_output`; field and `response_phi` both consume the context | Add opt-in anchor-side projection; new ordinary prediction reads one processed context cache |
| Same file, `_anchor_response`, `_apply_controls`, affine/JVP/VJP methods | Source-indexed linear heating via `P/m`, carriers, cover and query phi | Preserve the ordinary law; extend transport with linear heat-weighted moments and heat-independent mixing |
| Same file, `_query_work` and preparation counters | Some GPU reductions are converted to Python integers during ordinary calls | Keep exact counters on-device or derive shape counts; materialize once at the reporting boundary. Profile before assigning a speed contribution |
| `inherited_attention/readers.py`, `_gather_cover_values`, `gather_cover_indexed` | Materializes compact selected values and their weighted intermediate | Keep a transparent Torch reference; test a bounded compiled/fused weighted-gather implementation with a correct backward |
| `inherited_attention/atlas.py`, `smoothstep_grid_receiver_cover` | Reconstructs grid axes/mapping, including a lattice-to-anchor match, and expands axes for searchsorted | Separate scene-owned grid topology from query cover construction. Use batched searchsorted without Q-replicating the axis where supported |
| `inherited_attention/contracts.py` | Typed associations, token/anchor values and versioned ReceiverCover | Extend the actual typed contracts; add explicit moment/hyperedge/cache metadata without duplicating numerical payloads |
| `inherited_attention/query_reader.py` | Q has cell grouping and query-conditioned donor attention | Preserve. Do not claim Q-to-C conversion is function-preserving or rely on its slow cell-selected policy as a favorable new baseline |
| `content_hypergraph/staged_attention_v2.py` | SDPA contextualization, last-stage score export | No contextualizer redesign. Do not turn on full tracing to obtain the needed score snapshots |
| Native Thermal/Wind model factories and shared TrainingEngine | Dataset-owned transforms, physical extraction, fixed memberships and guarded continuation | Register the new family with the existing engine. Add a real native attachment/resume check before a long fit |
| Repository-root `AGENTS.md` | Immutable histories, complete outgoing-history audit, storage cleanup, readable reports | Follow current instructions. Keep generated arrays, figures and one-time analysis code local/ignored |

Refactor only the narrow preparation/read/response hooks needed for reuse. Preserve old constructor defaults, checkpoint schemas and runtime behavior. An opt-in new schema can add a nested `PreparedMomentHyperedges` object; include every new live tensor in factor-version/lifetime checks. An explicit hypergraph export should be a view of actual prepared tensors, not a second re-computed organizing model.

## 3. Stage E: same-function execution improvements

### 3.1 Move the affine type fusion to anchors

Suppress batch/head flattening in the notation. Let the current map be `W=[W_M,W_E]` with bias `b`:

\[
 c_q=W_M\sum_a\Lambda_{qa}G^M_a+W_E\sum_a\Lambda_{qa}G^E_a+b.
\]

Prepare the unbiased anchor feature

\[
 Z_a^0=W_MG^M_a+W_EG^E_a,
 \qquad c_q=\sum_a\Lambda_{qa}Z_a^0+b.
\]

This identity follows from linearity. Keep `b` outside the gather, applied once per query, so an invalid row or a declared subprobability diagnostic does not accidentally change bias counting. Keep existing final query masks. Flatten heads in the same order as the current `mc/ec.reshape`; a transpose mistake can preserve shapes and silently change the function.

For Q8192/E192, this particular 256→128 projection executes at approximately 42.7 times fewer locations. That is a layer-operation ratio, not a complete latency prediction. At Q1024 the ratio is smaller; at very small Q the anchor preparation can cost more. Benchmark complete calls and prepared-scene reuse separately.

Projection before *cover gathering* is valid even when the M/E association distributions differ. Moving the projection before each head's donor pooling requires additional care because head-specific probabilities and a full cross-head projection do not commute in general. Do not make that additional move implicitly.

Existing diagnostic exports may still request `mc/ec`. Produce those two contexts only for explicit tracing/intervention diagnostics; keep that extra work outside normal inference and normal training.

### 3.2 Cache topology, not detached geometry derivatives

Introduce a scene-owned `PreparedGridTopology` containing sorted axis identities, lattice-to-environment indices, legal bounds and their version/ownership binding. Its discrete mapping can be reused while the scene is unchanged. Query cover construction retains live query coordinates and, where relevant, live selected anchor coordinates and lengths. A geometry update invalidates the topology/cache. Do not detach geometry to gain speed.

Remove the repeated lattice-match construction from every query chunk. Avoid materializing a `[B,Q,axis_size]` copy solely for searchsorted: evaluate the supported batched `[B,axis_size]`/`[B,Q]` route, with old indices/weights as the reference. Preserve exact native IDs, anchor permutations, singleton-axis behavior, padded queries and legal-boundary checks.

Move data-dependent validation out of inner compiled arithmetic only through a validated public boundary plus version checks. Never disable stale-state rejection, consumed-autograd-graph guards or native-ID checking. Diagnostic counters must remain exact, but need not synchronize the host per tensor per chunk.

### 3.3 Optimize the compact gather, after measuring it

The ordinary context should gather one projected width128 tensor, rather than two width128 tensors. Use the existing indexed operation as the mathematical reference. First test a small tensor-only `torch.compile` region on the installed environment; include compilation/cold costs separately. Do not upgrade PyTorch or introduce a new dependency merely to enable compilation.

A fused implementation must compute both gradients: scatter-add into anchor values and derivatives of the weighted sum with respect to cover weights, allowing query/geometry gradients through those weights. Do not silently make backward nondeterministic or sever its higher-level response graph. Test duplicate anchor indices, singleton axes, zero weights, padding and chunk/order invariance.

A custom kernel is optional only if profiling shows this gather is consequential and the compiler does not produce a useful implementation. Cap that engineering attempt at one hour. An unsuccessful custom kernel must not consume the model-learning round. Retain the qualified Torch reference.

### 3.4 Qualification and arithmetic scope

At the saved Thermal C-soft e2500 checkpoint, test original versus optimized forward, parameter/source/query gradients, precise heat JVP/VJP, local/native extraction and one disposable actual optimizer boundary. Test Wind-C at its saved e500 state, not Wind-Q under a falsely equivalent formula. Test zero/invalid/mixed-size cases in synthetic fixtures and native low/high-M cases on Q1024/Q8192.

Use the existing independently specified TRAIN-scale and propagated-native-operator bounds. Do not relax historical tolerances after seeing an error. Report unchanged-weight native differences separately from physical-reference error. An old strict generic Thermal interface-screen failure remains an old failure; neither erase it nor require unrelated historical numerical problems to be solved before any new architecture can be trained.

If one optimization cannot qualify, retain it as experimental and use the last qualified executor for all matched learning arms. Identical mathematics is not bitwise arithmetic; optimizer continuation needs its own qualification. A failure to qualify an optional faster kernel is not a scientific reason to cancel all moment/hyperedge learning.

## 4. The new representation: typed moment hyperedges

### 4.1 Definition and interpretation

Let \(\mathcal X\) contain target-free geometry, module/environment attributes, positive measures, masks, operating conditions and physical identities. For head \(h\) and receiving environment anchor \(a\), keep the inherited soft associations

\[
 P^M_{ahi}=\frac{\mu_i^M\exp(s^M_{ahi})}{\sum_j\mu_j^M\exp(s^M_{ahj})},
 \qquad
 P^E_{ahj}=\frac{\mu_j^E\exp(s^E_{ahj})}{\sum_k\mu_k^E\exp(s^E_{ahk})}.
\]

Measures are normalized/handled exactly once by the current typed implementation. Retain last-EM/EE score-stage metadata separately from final-token payload-stage metadata. Each `(anchor ID, head ID)` is a typed, directed information hyperedge, with M and E donor memberships and geometry-owned receiver access. This is an explicit representation of the existing computation, not proof of physical higher-order causality or a newly learned anchor count.

Soft associations remain fully positive. The first version therefore studies *useful collective computation and cached-query reuse*, not source-pruning acceleration. Do not add entropy, degree, diversity, sparsity, orthogonality or occupancy penalties.

### 4.2 Why moments must precede the bottleneck

A nonlinear function applied only to an already-collapsed mean cannot distinguish two donor configurations with the same mean. A simple example has sources at normalized x=−1,+1, equal weights, and values `(1,−1)` versus `(−1,1)`. Both means are zero; their first position–value moments are −1 and +1. The example motivates retaining cross moments; it does not prove which moments the physical datasets require.

Use coordinates normalized by the existing scene origin/scale:

\[
 \xi_i=(x_i-o)\oslash L,\qquad \xi_a=(x_a-o)\oslash L,
 \qquad r_{ai}=\xi_i-\xi_a.
\]

Do not use raw world coordinates, case indices or physical-ID embeddings. Use the same native frame as the adapter; Wind data are already aligned. This proposal supplies translation-consistent relative statistics, not automatic rotational equivariance.

For each type, preserve the full existing value mean \(G^t_{ah}\) of head width32. Project donor values to a **moment width of eight per head**, \(u^t_{hi}=L_h^tV^t_{hi}\). Then cache:

\[
 \bar u^t_{ah}=\sum_iP^t_{ahi}u^t_{hi},
 \quad d^t_{ah}=\sum_iP^t_{ahi}r_{ai},
\]

\[
 C^t_{ah\ell r}=\sum_iP^t_{ahi}(r_{ai})_\ell u^t_{hir},
\]

\[
 \Sigma^{x,t}_{ah}=\sum_iP^t_{ahi}r_{ai}r_{ai}^{\mathsf T}-d^t_{ah}(d^t_{ah})^{\mathsf T},
 \qquad
 v^{u,t}_{ah}=\sum_iP^t_{ahi}(u^t_{hi})^{\odot2}-(\bar u^t_{ah})^{\odot2}.
\]

Store only the symmetric upper triangle of spatial covariance and the diagonal latent variance. Keep M/E statistics separate. Include typed validity flags, declared characteristic lengths and the existing operating state. Reuse already available typed mass/count metadata where needed; never add module area to environment volume or invent one shared mixed-unit mass.

The descriptor \(\eta_a\) concatenates the existing typed means, these compact statistics and geometry/operating features. It exists at **E anchors, not Q queries**. Use a fixed, documented block ordering. Start with exactly eight moment coordinates per head, not a rank sweep. The covariance/variance terms are descriptive statistics, not physical energy or calibrated uncertainty.

### 4.3 Compute moments from donor-only lifts

Do not construct a wide `[B,H,E,N,r]` position–value array merely to calculate moments. Prepare donor features once:

\[
 [u_i,\ \xi_i\otimes u_i,\ u_i^{\odot2},\ \xi_i,\ \operatorname{vech}(\xi_i\xi_i^{\mathsf T})],
\]

then apply the same P contraction to that lifted feature tensor. For example,

\[
 C_{ah\ell r}=\sum_iP_{ahi}\xi_{i\ell}u_{hir}-\xi_{a\ell}\bar u_{ahr}.
\]

Center the spatial moments after aggregation. This removes the anchor–donor–value expansion while retaining the intended statistics. Intermediate sizes and contraction paths must be inspected, not inferred from an einsum string. No wide Q×M/Q×E value tensor is allowed on the nonlocal route.

Variance can suffer cancellation. Use a stable small-aggregate implementation and an FP64 reference. Clamp only negative roundoff within a predeclared scale-aware allowance; flag larger negatives. Do not take square roots of variance in the first version. Handle empty typed rows with explicit zeros and validity flags rather than an arbitrary large epsilon or learned missing-type token.

### 4.4 Three processors with one query representation

The common base is

\[
 Z_a^0=W_MG_a^M+W_EG_a^E.
\]

**C0:** \(Z_a=Z_a^0\).

**M:**

\[
 Z_a=Z_a^0+W_2(W_1\eta_a+b_1)+b_2.
\]

**H:**

\[
 Z_a=Z_a^0+W_2\operatorname{SiLU}(W_1\eta_a+b_1)+b_2.
\]

Use hidden width128 and identical descriptor/output dimensions in M and H. The two-layer affine control intentionally retains the same trainable parameter count and optimization depth; replacing SiLU with identity isolates the nonlinear map at this site. It does **not** make M an independent-source model: its contextualized values, attention, statistical variances, thermal contractions and downstream head remain nonlinear or interactive.

Initialize `W2` and `b2` to zero, with ordinarily initialized nonzero `W1`; do not also initialize a multiplicative residual gate to zero. With both output and gate zero the new block can become gradient-dead. If no gate is needed, omit it. The initial complete function must recover C0. Confirm output-layer gradients on the first real update and upstream moment/processor gradients on subsequent updates.

The ordinary query reads

\[
 c_q=\sum_{a\in\mathcal A(q)}\Lambda_{qa}Z_a+b.
\]

The field head and the existing Thermal `response_phi(qs,c_q,local)` keep their original input widths. Do not concatenate all raw moments at every query. This is the main constraint that keeps additional computation concentrated in scene preparation.

A nonlinear hyperedge processor cannot recover distinctions missing from all its inputs. Even mean-plus-first-moment/covariance statistics can collide. Make no universal injectivity, universality or sufficient-statistic claim for this bounded representation.

## 5. Thermal: richer transport without losing heat linearity

### 5.1 Retain the existing connection; enrich its source-resolved payload

The current phi already sees global/local context. H therefore affects temperature through phi as soon as its context residual is nonzero. To test the user's additional heat-moment idea, give M and H the same compact source-resolved transport extension. Keep all of its coefficients heat independent.

Let \(m_i\) be the current positive normalized module measure and \(t_{ihr}\) the existing eight signed carrier channels. Define

\[
 R^0_{ahr}(h)=\sum_i\frac{P^M_{ahi}}{m_i}t_{ihr}h_i,
 \qquad
 R^\ell_{ahr}(h)=\sum_i\frac{P^M_{ahi}}{m_i}(r_{ai})_\ell t_{ihr}h_i.
\]

The ratio `P/m` preserves the current discrete-control convention. No extra physical measure is multiplied into heating. In the uniform-measure-density limit, the transport is a source sum, not an unintended average over source count.

Have the same per-anchor processor produce small heat-independent **diagonal** mixing coefficients \(a^0_{ahr},a^\ell_{ahr}\), via separate zero-output heads on the same processor hidden state:

\[
 \widetilde R_{ahr}(h)=(1+a^0_{ahr})R^0_{ahr}(h)+\sum_{\ell=1}^{d}a^\ell_{ahr}R^\ell_{ahr}(h).
\]

Start with diagonal carrier mixing, not an R×R matrix, extra carrier heads, extra context cycles or a second response network. All `a` outputs start at zero. Both M and H have this same extension, so their controlled difference remains the processor nonlinearity. C0 retains original transport.

The output is

\[
 \widehat T(q;\mathcal X,h)=
 \sum_{a,h',r}\Lambda_{qa}\phi_{qh'r}(\mathcal X)\widetilde R_{ah'r}(h)
 +T_{\mathrm{local}}(q;\mathcal X,h).
\]

Here \(h'\) denotes attention head, distinct from the heating vector. Heating never enters a nonlinear activation, a membership, a query cover, moment statistics or a normalizer in this path. The nonlinear processing is in the geometry/configuration-dependent coefficient, not in the applied control.

### 5.2 Reuse and exact source identities

One equivalent source-indexed scene coefficient is

\[
 C_{ahri}(\mathcal X)=\frac{P^M_{ahi}}{m_i}t_{ihr}
 \left[1+a^0_{ahr}+\sum_\ell a^\ell_{ahr}(r_{ai})_\ell\right].
\]

This permits preparing a modest `[B,H,E,R,M]` transport or retaining its equivalent factors, then applying many heating vectors without rerunning contextualization, moments or the nonlinear processor. Measure which representation is cheaper at the actual M/E/R. It is acceptable to retain an E×M source-indexed scene factor; the forbidden return is a wide Q×M nonlocal value MLP. Do not claim that a larger factor cache is necessarily smaller merely because no dense K is exported.

Preserve

\[
 T(\mathcal X,0)=0,\quad T(h+\delta h)-T(h)=T(\delta h),\quad
 \partial_h^2T=0,\quad\partial_h(u,v,p)=0.
\]

The same prepared factors supply heat JVP and VJP, with actual source-ID correspondence and transpose native extraction. The ordinary path need not materialize K[Q,M]; explicitly requested K exports must agree with JVP/VJP. Maxima are recomputed at endpoints: `max(T(h+dh)) - max(T(h))`, never `max(T(dh))`.

No environment token becomes a fictitious independent heater. No sign constraint is imposed on K solely from an attention picture. Saved single-direction heat families do not independently identify every K column.

## 6. Receiver resolution: diagnose first; one coherent revision

The current smoothstep cover is C1 but flat at anchor centers and not linearly reproducing. This is a property of the gathered-context path, not of the entire predictor: the query encoder, local branch and phi still vary. Do not diagnose a field plateau solely from the cover formula.

At fixed TRAIN4 cases, examine residuals and derivative discrepancies against the query's phase within its environment cell. Use native points/masks, and show the same statistic for D0 and C0. Inspect whether strong phase-aligned residual structure persists after moment enrichment; do not mistake ordinary spatially varying target difficulty for a cover artifact.

If the diagnosed limitation remains, the single allowed architecture revision is a **first-order local reconstruction of cached hyperedge states**:

\[
 c_q=\sum_a\Lambda_{qa}
 \left[Z_a+\sum_\ell(\xi_q-\xi_a)_\ell D_{a\ell}\right]+b.
\]

`D` is a learned heat-independent per-anchor slope tensor produced during scene preparation. It starts at zero, so the attachment preserves its parent. Read the same four/eight anchors; do not change membership, E, contextualizer depth or local support. Implement the weighted affine reconstruction with compact indexed reads and one output tensor, rather than materializing a wide donor/query network.

This uses the existing partition of unity and does not require a new learned cover router. It is not merely changing smoothstep to bilinear interpolation. If \(Z_a=A\xi_a+c\) and \(D_a=A\), then the read reproduces \(A\xi_q+c\) for every valid query, including declared legal boundary extensions. At an interior anchor center, its derivative can equal D instead of being forced to zero. With a C1 cover and fixed smooth anchor data the reconstructed context remains C1. These statements concern representational capability; trained slopes need not equal true physical derivatives.

Start by changing only the heat-independent context reconstruction. Thermal's phi then sees this improved context while the response remains linear in heat. Do not simultaneously introduce separate heat-transport slope tensors. Use the same jet option in both dataset implementations; one dataset may decline it on evidence.

This revision may branch only from an H/e1000 checkpoint after the TRAIN diagnosis. The unmodified H must still finish e2500 as its matched continuation control. A jet child has 1000 inherited and 1500 new epochs, not 2500 new-architecture epochs. At most one such child per dataset is allowed. Do not use this allowance for member feedback, increased widths, new losses or anchor-grid expansion.

## 7. Unified implementation and clean interfaces

### 7.1 Shared model contract

Add a small versioned `MomentHyperedgeCore` extension with explicit settings:

```text
base_reader: IAG-C
association_normalization: soft
context_dim: 128
num_heads: 4
context_depth: 2
moment_width_per_head: 8
processor: none | affine | silu
processor_hidden: 128
local_affine_reconstruction: false | true   # conditional child only
scene_executor: dense
query_executor: indexed_projected
```

The source processor, typed moments, state projection and indexed reconstruction are genuinely shared code in 2-D/3-D. Dataset adapters own descriptors/frames, native receiver covers, measures, legality, outputs, statistics and optional affine-control capability. Thermal/Wind have separate weights, not a claim of zero-shot transfer or one universal normalization.

Use small modules such as `moments.py`, `hyperedge_processor.py` and `cover_execution.py` inside the existing inherited-attention package, with names consistent with repository style. Extend actual factories and config validation through explicit new schemas. Preserve `InheritedAttentionCoreV1` loading unchanged. Avoid a new parallel data loader, optimizer, callback system, graph controller, trainer or experiment ledger.

### 7.2 Explicit hypergraph and cache object

The new prepared view must expose, without changing prediction:

- Native module/environment/anchor identities, separate types/measures/masks, head IDs, coordinate frame and score/payload stage.
- Actual P, ordinary means, compact moments/statistics, processed Z, optional D, and source-indexed affine factors where present.
- Actual receiver-cover indices/weights and local physical pairs; clear differentiation between headwise positive support and unions.
- Weight/buffer/geometry/normalizer/receiver/lifetime binding, work counters and which arrays were actually retained versus computed for a trace.

A graph export maps real tensor paths. Do not invent hyperedge labels such as “constriction” or “wake-overlap group” before testing them. Soft groups are overlapping weighted distributions, not discovered binary physical components. An E192/E64 grid is an adapter discretization, not adaptive selection of hyperedge count.

### 7.3 Keep physical adapters intact

Thermal: one self-contained jointly trainable core; heat-independent flow; source-affine T; no temperature mean addition; native grid union, curl, surface/outside/material interpolation and conductivity/spacing q proxy unchanged. Respect zero-boundary capability. All material and interface outputs remain derived from the same T field.

Wind: unchanged already-aligned coordinates, geometry-only E64 records, operating inputs and query features; preserve `profile(z) + u_ref * safe_std * residual`. No second coordinate rotation or absolute-velocity normalizer mean. Retain native cells/crosswind cuts and component-vector definitions. The existing five-role endpoint board remains an evaluation board, not a replacement training objective.

## 8. A bounded learning experiment, not a new model portfolio

### 8.1 Shared starting point and matched arms

Use each dataset's **own Round4 IAG-C soft e500** checkpoint as the common starting function for C0/M/H. Thermal C/e500 and the stopped original Wind C/e500 are distinct native models. Do not initialize Wind from its mature Q model or mix the two representations' optimizer state.

| Arm | Common inherited state | New representation | Planned endpoint |
|---|---|---|---|
| C0 | Same-dataset C-soft e500 | Same-function optimized C, no moments | Absolute e2500 |
| M | Same C-soft e500 | Compact moments + affine processor and moment heat transport where applicable | Absolute e2500 |
| H | Same C-soft e500 | Identical moments/dimensions + SiLU processor | Absolute e2500 |

Preserve all common weights and named optimizer moments, RNG/sampler state and 500-epoch history. New M/H tensors are initialized identically by name; their new optimizer states start at actual birth. The zero-output additions recover C0 at attachment. All physical/contextual and new parameters train thereafter; this is not a frozen-backbone adapter fit.

Run a new C0 continuation under the same qualified execution path as M/H when operation ordering changes. Reuse the old Thermal C2500 trajectory only if the actual continuation contract and optimizer-boundary arithmetic support that reuse; otherwise do not call it a bitwise-matched control. The old report remains contextual evidence even when a new control is needed. Wind C has no saved e2500 trajectory, so its C0 continuation is real new work.

These are warm, controlled development continuations: 500 inherited plus 2000 new epochs per arm. They test adding scene-level capacity to the same foundation. They do not prove fresh-start superiority. A qualifying fresh full-TRAIN probe later checks that separate question.

### 8.2 Preserve the established training semantics

Thermal: fixed TRAIN150/DEV22; named initialization0 and sampling0; Q1024 primary fluid plus unchanged native auxiliaries/response addendum; effective48/micro24; 150 case visits and four actual updates per epoch. Selected-TRAIN normalization only.

Wind: fixed TRAIN72/DEV24; original layout grouping; initialization0/sampling42; Q8192 comprising 6144 volume and 2048 hub-band; effective16/micro8; 72 visits and five actual updates per epoch. The training objective remains `.75 volume + .25 hub`, equal standardized channel loss. Preserve the TRAIN-fitted profile/residual normalizer. The omitted TRAIN M18 stratum remains disclosed, not silently replaced.

Continue the Round4 all-parameter AdamW constant3e-4, weight_decay1e-5, betas(.9,.999), eps1e-8 and clip1. Do not change the schedule to improve a particular arm. Preserve final partial macro-batches. Use identical input/query fingerprints across new C0/M/H continuations.

Each 2000-epoch continuation adds Thermal300,000 visits/8,000 ordinary updates or Wind144,000 visits/10,000 updates. Actual native receiver unions and response work are separately counted; do not infer them from primary Q. Count the common 500 prefix once in resource use, while reporting inherited age in each trajectory.

### 8.3 Reviews and completion

Review after 100 new epochs, then at absolute e1000, e1500 and e2500. Keep the ordinary saved100 checkpoints, latest/best aliases and curve refresh every10. Healthy candidates receive the full planned horizon, even when a short review is mixed. Stop only for nonfinite/invalid operation, resource limits, or a clear persistent fitting breakdown documented against TRAIN and the controlled baseline—not because an early single DEV point fails the final target.

At e1000, decide whether a conditional jet child is warranted from the fixed TRAIN diagnosis and remaining budget. Do not alter the already-running primary arms. All final primary comparisons use literal e2500; the common saved-cadence field selector is a separately labelled secondary result. Do not choose a checkpoint using saved response outcomes or graph attractiveness.

### 8.4 Gradient competition is a diagnostic, not an extra campaign

On a small fixed TRAIN panel at attachment and e1000, measure gradients of flow/curl versus temperature/q/response groups at the inherited raw score tensors and at shared contextual parameters. Record cosine, norms and ratios; negative cosine at one tensor is not proof that loss conflict caused the field failure.

Respect consumed-graph guards. Use separate deterministic identical-weight forward preparations for gradient groups where necessary, with no optimizer updates or stale-graph reuse. Save the actual loss scales and callback cadence. Do not automatically add PCGrad, GradNorm, FAMO, adaptive weighting or a new task-specific organizer in this round. A persistent conflict after the representation experiment can motivate the next narrowly controlled learning revision.

## 9. Tests that are sufficient to trust the experiment

The point is to find correctness errors before long fitting, not to maximize a test count. Reuse existing tests and add focused checks for changed behavior.

### 9.1 Pure algebra and shape contracts

Check projection-before-gather equality and gradients; bias exactly once; local masks; moment literal-versus-lifted formulas; finite empty types; donor and anchor permutation with IDs; relative-coordinate transformations; environmental measure splitting in a deliberately identical-token fixture; source-control identity; processed-state serialization and stale invalidation.

Do not impose whole-model invariance to duplicating a physical module: module count and separately applied heating are real inputs. Environmental duplicate-measure tests concern the stated representation contract, and must not quietly change contextual inputs/count semantics.

Check zero-initialized M/H complete recovery; active gradients after two updates; identical M/H shapes and parameter inventories; and real native TrainingEngine attach/save/resume on a small disposable state. New checkpoint keys, optimizer moments and sampler identities must survive a true engine round-trip, not only `state_dict` loading.

### 9.2 Native tests

Use low/high-M native TRAIN fixtures at Q1024/Q8192, plus one mixed-size batch per dataset. Test full output transforms, native extraction, finite source/query gradients and the unchanged local support. Thermal additionally tests T(0), heat independence of flow, linear increments, VJP/JVP adjoints, source permutations with controls, and explicit K consistency on a small requested export. Compare precise accumulation separately from ordinary FP32 endpoint subtraction.

The new projected/fused executor must also pass a real backward and one native optimizer boundary. Training and inference qualification are distinct. On optional jet code, test affine reproduction, anchor-center slopes, continuity from both sides of cell boundaries, legal outer extension and fixed IDs. These are software fixtures, not physical validation.

### 9.3 Bounded synthetic checks performed while drafting this plan

Independent constructed CPU FP64 checks—not repository/native tests—gave projection difference 2.22e-15, projection-gradient difference 2.84e-14, literal/lifted first-moment difference 8.88e-16, affine-jet reproduction difference 1.78e-15, exact zero-heating output, affine-increment difference 4.26e-14 and heat-adjoint difference 2.84e-14. They support the written algebra only. Codex must run the native and GPU checks above; these small examples qualify no implementation, checkpoint, performance gain or physical result.

## 10. How to test organizing value without overclaiming

### 10.1 Controlled field comparison

C0→M tests the richer statistics/transport package. M→H tests the nonlinear anchor processor with the same descriptor, payload dimensions, initialization and parameter count. H→H-jet, if executed, tests local reconstruction against its unchanged continuation parent. D0 remains the stronger practical development reference with its own original reader. Mature classics remain unequal-training reference distance on the same physical panel.

Do not turn one aggregate normalized loss into the conclusion. Thermal must show T, u/v/p, curl, surface/material T, q and endpoint peaks. Wind must show all five role-vector errors and all three components, with near/downstream transverse fidelity visible. Use equal cases/rows as before and group Wind directions by layout for interpretation. Report means/p90 and worst/strata where already available, without treating thousands of queries as independent experiments.

### 10.2 Actual hyperedge interventions

At the input-fixed scenes Thermal0291/0687 and Wind69/426, export three receiver neighborhoods with actual source/anchor IDs. Show ordinary means, spatial centroids/extent, selected value–position cross moments and the processed Z alongside P and Lambda. Latent norms and covariance have latent/nondimensional labels, not energy or physical response units.

Interventions keep weights fixed and distinguish three scopes: (a) zero only new moment/processor output, retaining the original branch; (b) change memberships while holding contextual token states fixed and **recompute all dependent means, moments, Z and thermal factors**; (c) a real input-geometry perturbation that rebuilds contextualization as well. An unchanged Z cache after changing P is a stale intervention, not a valid counterfactual.

Use typed admissible mass-preserving perturbations. A small 2×2 mass exchange can preserve row and column totals and positive measure support; choose its size before errors and forbid negative probabilities. A donor permutation is a control only if its measure semantics are valid, or if the entire donor tuple is relabelled consistently for an invariance test. Merely swapping soft columns does not automatically preserve both degree and measure. Keep semantic perturbations distinct from pure ID permutations. Restore outputs exactly when restoring the original cache/model state.

### 10.3 Higher-order evidence: what is and is not isolated

A fixed normalized weighted sum is linear in its supplied values. A nonlinear function of group statistics can have nonzero mixed derivatives in those statistics. Demonstrate this directly at the **processor output**, e.g.

\[
 \Delta_{ij}^{\mathrm{proc}}=F(\eta+\delta_i+\delta_j)-F(\eta+\delta_i)-F(\eta+\delta_j)+F(\eta).
\]

With fixed externally supplied descriptor increments, this is zero for the affine processor up to arithmetic, and may be nonzero for H. This checks that the new processor actually represents collective non-additivity. It does not identify physical source triples.

Do not expect the full M predictor to have zero source-geometry interaction: statistics such as variance, the contextualizer, normalization, source transport and downstream head are already nonlinear. Likewise, a nonlinear maximum objective can show mixed effects even when individual fields add linearly. Report processor curvature, whole-model geometry contrast and physical-reference contrast as different quantities.

Use existing compatible saved geometry-response records only when their exact endpoints, input rounding, source IDs, masks and numerical scope are verified. A two-body geometry contrast needs all four aligned states; a third-order contrast needs all eight. Missing states remain unavailable. Do not generate new solver labels or infer absent combinations. Thermal heat–heat mixed contrasts should remain zero by construction; use geometry changes, not heating nonlinearity, to investigate multi-body physical dependence.

Member-specific hyperedge→source feedback inspired by ED-HNN is deferred. The existing contextualizer already exchanges node information. First establish whether richer cached edge computation is useful before adding another feedback cycle.

## 11. Complete cost, memory and the next-run decision

### 11.1 Measure the right operations

Compare, on the same physical GPU and exact input/output fixtures: original C versus optimized C at the same weights; C0/M/H at trained matched ages; candidate versus D0; and optional fused versus reference gathering at identical weights. Use profiler-off synchronized interleaved repetitions, two warmups and seven trials, reporting medians plus observed spread. Repeat an ambiguous small gain in a second independent block. A profiled trace diagnoses kernels/host stalls; it is not the headline timer.

Report four distinct scopes: complete host-input→native-CPU-output call; prepared-scene/new-query; prepared fixed-query/new-heating where applicable; and complete TrainingEngine epoch. Include masks, cover construction, identities, local work, native extraction and return copies in their correct scopes. Do not add overlapping medians. New larger scene preparations must be charged to each new geometry.

At fixed B8, measure Q1024/4096/8192 with the same native auxiliaries and actual unions. Show absolute allocated and reserved peaks, not baseline-subtracted memory alone. Measure actual optimizer-boundary memory separately. Do not shrink B to obtain a memory win. Only after selecting a model may a bounded equivalent-effective-batch packing test choose production microbatching; this does not retroactively alter the architectural comparison.

Counters must expose contextual attention, lifted donor rows/channels, scene contractions, processed anchors, projected anchor rows, query gather bytes/reads, local pairs, field/phi rows, thermal source-response factors and actual retained storage. `far_pair_value_rows=0` is necessary but insufficient. Inspect for hidden `[B,Q,N,128]` or `[B,H,Q,E,D]` expansions, Q-replicated validation work, excessive scalar synchronizations and duplicate native field decoding.

### 11.2 Success has three levels

**Engineering result:** projection/topology/gather changes preserve the checked function and reduce a measured cost. This can be delivered even if the new architecture does not qualify.

**Research direction:** a matched moment/nonlinear model materially improves the deficient fields, preserves useful response behavior and remains memory efficient. About 10% reduction of the named Thermal flow or Wind wake error versus C0 is a useful signal; it is not automatically a formal-run qualification. H need not win: an M win would favor richer statistics over extra nonlinearity, an informative result rather than a reason to hide M.

**Conditional fresh full-TRAIN e1000 probe:** retain the prior Round4 research gates as the main anchor, evaluated against matched segmented D0: priority means within10%, priority p90 within20%, other means within15%; and either at least10% lower complete high-M latency, or at least20% lower absolute B8/Q8192 allocated memory with no more than15% complete slowdown. The existing accuracy-led alternative remains available: at least5% improvement in the named priority, other guards preserved, and cost/memory no more than10% higher. Re-evaluate on the actual new weights/fixtures; do not substitute old IAG cost ratios.

For Thermal, require that a qualifying candidate not give away the retained soft-C q/material-response progress through an unexplained large regression; show all saved-response errors, but do not invent a physical-inverse certification gate. For Wind, passing versus a poor C0 is insufficient: a model still several times worse than D0 does not qualify simply because it is memory efficient. These are pre-experiment criteria; do not change them after outcomes.

No mature-classic replacement, universal hypergraph superiority or sparse-executor speedup is required for the first useful cached representation. A full soft model can qualify. A positive nonlinear-processor claim needs its M comparison, not only an ablation that changes outputs.

### 11.3 Exactly bounded full-data follow-up

Subject to continuing user authorization and remaining resources, launch at most one **fresh** full-TRAIN e1000 probe per dataset for the qualified family. Preserve the existing conditional allowance; no auto-e2500/e5000 full-data extension. New full-TRAIN membership and TRAIN normalization must be fitted/bound explicitly, with fresh weights/optimizer. Do not resume segmented weights and rename their exposure formal.

Use the corresponding prospective e5000 schedule only where the maintained full-run recipe requires it, with an enforced literal1000 stop. Compare to saved A2-D/classic e1000 checkpoints when available on identical native receivers; mature reference checkpoints are a separate contextual column. Missing same-age checkpoints remain missing. Report actual visits/updates/queries and selected versus terminal ages. These probes test fresh training/scaling potential, not final generalization or inverse readiness.

## 12. Execution sequence and bounded autonomy

1. **Read/prepare:** reconcile branch/head with8426e12, current instructions, protected jobs, native manifests, exact C/e500 parents and old reports. Freeze existing source/input bindings and resource budgets. Do not reset user work.
2. **Execution baseline:** implement projection/topology improvements, test native same-function behavior, measure. A failed optional fused kernel leaves the qualified reference available.
3. **Model implementation:** add typed moments, M/H processors, diagonal moment heat transport, explicit hypergraph views and real TrainingEngine/factory integration. Confirm zero-residual recovery and actual multi-update gradients.
4. **Matched learning:** run C0/M/H continuations on both segmented datasets, review real fits and complete the healthy e2500 horizons. No sparsity penalty, sparse child, new teacher or new loss.
5. **Conditional revision:** only the documented first-order cached reconstruction may be tried, with an unchanged parent control and enough remaining budget. Source fixes are allowed, but any equation change needs a distinct run identity; never hot-edit a frozen runtime.
6. **Evaluate/decide:** native fields, stored responses, compact organizing interventions, actual costs and memory. Launch only qualifying fresh1000 probes under the stated boundary.
7. **Close:** produce the report and short evidence index, preserve numerical failures and limits, clean owned disposable artifacts, audit the entire outgoing commit range, commit/push the non-default branch and verify remote/local tips.

Use a proposed ceiling of **20 aggregate GPU-associated process-hours and14 elapsed hours**, including setup, failed calls, pilots, learning, evaluation and optional full-data probes; reserve the final two elapsed hours for reporting/cleanup. Tighter current user instructions dominate. Forecast all healthy primary endpoints before starting; reserve their maturation before optional children or extensive diagnostics. If forecasts exceed the ceiling, ask for a scope decision before launch rather than silently delivering truncated “completed” fits.

**Device authority:** Round4 explicitly superseded the old0/2 note with physical **GPU1 and GPU2**, while protecting GPU0's unrelated PhyCoFlow job. Inspect current authorization, UUIDs and occupancy before any allocation. Do not use GPU0 merely because an older plan allowed it. Never kill unrelated jobs. Use the ModularDT interpreter and installed versions, FP32/TF32-off/no-autocast for the matched experiment unless a separately qualified, explicitly approved precision experiment is undertaken.

Keep new persistent storage within12GiB `/data` and1GiB `/home`, with20GiB available-space floor, reusing exact shared caches and no dataset copy. Forecast temporary/cache/export space too. Retain scientific checkpoints at declared cadence; hardlinks/copies must preserve ownership safeguards. Cap disposable engineering at400 actual optimizer updates, separate from every scientific continuation. Do not spend scientific epochs on discarded initialization or auxiliary teacher fitting.

## 13. Required deliverables and readable figures

Deliver maintained shared modules/configuration/tests, inert standalone construction/loading examples for both datasets, the actual trained checkpoints/history identities in the local artifact store, a small machine-readable endpoint table, and a new `HONF_Scene_Prepared_Moment_Hyperedges_Round5_Report.md`. Keep prior reports unchanged.

The report should start with one page answering: what changed; which native fields improved or regressed; how much complete time/memory changed; whether H beat M; and whether a fresh1000 probe ran. Separate Predictor, Organizer, Execution and Inverse-interface judgments.

Retain at most five main figure families:

1. Matched physical learning curves at saved ages, including fixed-weight TRAIN/DEV and actual elapsed/update exposure.
2. Native reference/prediction/signed-error maps on the fixed two scenes per dataset; visible relative L2, Thermal flow and T, Wind Ux/Uy/Uz and wake cuts. No 20-panel illegible montage.
3. A physical-layout hyperedge view with three receiver neighborhoods: actual memberships, centroid/extent/cross-moment summaries, processed-state difference and an actual intervention result. Do not substitute latent-norm color for physical contribution.
4. Fidelity versus complete call cost and fixed-B8 memory versus Q, separating mathematical optimization and new-model effects.
5. Saved Thermal responses and the conditional cover/processor diagnostics that actually determined a decision; no fictional inverse-design panel.

Every numeric figure must come from saved arrays with source IDs, native masks/units, checkpoint/exposure and comparator scope. Inspect each retained PDF/raster and all links. Prose/captions remain single unwrapped source lines as required by AGENTS. Generated figures/data/one-off renderers stay ignored. At closeout report exact owned bytes/files removed and current filesystem capacity; do not delete scientific evidence or reinterpret old missing bindings as new loss.

## 14. Research grounding and boundaries

**AllSet — Chien et al., ICLR2022.** The paper organizes hypergraph learning as compositions of learnable multiset functions, with DeepSets/attention implementations. It supports making hyperedges compute on aggregated member information instead of treating an incidence diagram as the entire model. The weighted spatial moment construction, cached native field decoder and heat-affine transport here are proposed adaptations, not results proved by AllSet.

- Paper: https://arxiv.org/abs/2106.13264
- Authors' code: https://github.com/jianhao2016/AllSet

**ED-HNN — Wang et al., ICLR2023.** It provides an equivariant hypergraph diffusion architecture with receiver-dependent node updates, relevant to a later member-specific feedback extension. It does not establish Thermal/Wind field fidelity or physical causality of learned groups. Do not import its complete graph-training stack or add a feedback round now.

- Paper: https://arxiv.org/abs/2207.06680
- Authors' code: https://github.com/Graph-COM/ED-HNN

**PNA — Corso et al., NeurIPS2020.** Multiple aggregation functions can improve discrimination of continuous-feature neighborhoods. This motivates looking beyond a single mean. It does not make the small moment set proposed here injective, physically sufficient or universally superior.

- Paper: https://papers.neurips.cc/paper_files/paper/2020/hash/99cad265a1768cc2dd013f0e740300ae-Abstract.html

**PyTorch compilation guidance.** Data-dependent Python control flow and scalar extraction can break compiled regions; profile the installed version and isolate tensor-only kernels. A compiler is not a substitute for correct masks, reuse lifetime or a complete-call measurement.

- Official guidance: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html

### Reviewed repository references

- `https://github.com/cosmos2w/ModularDT/tree/8426e12f07b07acbe744d13a80ef2abb4edd5073/HONF_Proj/src/honf_forward_core/inherited_attention`
- `model.py`: actual prepare, decode, `gather_output`, `response_phi`, source-affine application and ownership checks.
- `readers.py`: typed `P @ V`, indexed cover gather and explicit dense references.
- `atlas.py`: last-route normalization and native regular-grid smoothstep cover.
- `contracts.py`/`query_reader.py`: typed identities/cache boundaries and the distinct Q implementation; inspect their complete current contents before modification.
- Round4 report: mathematical law, native tables, training identities, complete-cost/memory scopes, latest GPU1/2 authorization and preserved failures.

**Completion criterion:** leave the repository with one understandable shared direction and a decisive matched result: whether richer moment hyperedges recover useful field information at a worthwhile complete cost. A well-supported M-over-H result or a failed nonlinear hypothesis is acceptable; another unmeasured architectural expansion, misleading sparse claim, or collection of startup checks is not.
