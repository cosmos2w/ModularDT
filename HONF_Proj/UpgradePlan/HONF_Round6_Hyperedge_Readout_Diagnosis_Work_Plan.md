# HONF Round6 — diagnose the hyperedge-to-query readout before changing the organization

**Mode:** bounded Codex Goal-mode implementation and measured development. **Reviewed repository:** `cosmos2w/ModularDT`, commit `aec07be947bb90f967c4471e5ed88c270f29ab3b`. **Primary source:** `HONF_Proj/docs/reports/HONF_Scene_Prepared_Moment_Hyperedges_Round5_Report.md`, read in full together with the relevant current core, reader, native-adapter and training-entrypoint code. **Status:** a proposed execution contract; no Round6 experiment or performance improvement is claimed here.

## 0. Decision in plain language

Keep the attention-organized, scene-prepared hypergraph. Test whether the decoder discards useful distinctions when it averages several hyperedge states before applying its query-conditioned nonlinearity. Do not add another hyperedge processor, more moments, another organizer, a larger grid, a wider local support, or a new physical loss in the primary comparison.

The first learning comparison must hold the **numerical prepared scene representation fixed**, not merely use the same architecture and initialization. Freeze the Round5 H scene-preparation parameters and compare two parameter-matched receiver adapters: nonlinear processing **after** pooling versus nonlinear processing **before** pooling. They consume the same processed anchor states, the same query features, the same signed query–anchor geometry, the same four/eight cover slots and the same cover weights. Train the same downstream heads in both. This directly tests the position of query-conditioned nonlinear computation without giving one arm a richer scene representation.

If the local comparison remains weak, permit one bounded **full-bank access diagnostic** using the same frozen anchor payload, with no physical-source reads. A positive result would identify a useful direction in receiver access; a negative result would still not prove that all useful source information has been destroyed. If the local comparison gives meaningful TRAIN evidence, permit one separate matched, jointly trained confirmation from the retained H e1000 checkpoint to absolute e2500. Frozen-feature fits and jointly trained models answer different questions and must never be merged into one evidence claim.

The primary deliverable is an interpretable diagnosis with executable code and real fitting, not a requirement to obtain a pass. Do not manufacture a positive conclusion by changing objectives, choosing a favorable native endpoint, widening tolerances, or doing a sequence of undeclared architecture repairs.

### Required outcomes

| Question | Required evidence |
| --- | --- |
| Does pooling before receiver-conditioned nonlinearity restrict this decoder? | Matched frozen-representation `R-P` versus `R-E` fits, plus a controlled pooling-nullspace test |
| Are useful distinctions present elsewhere in the same prepared bank? | Only if needed: the explicitly labelled full-bank `R-G` diagnostic, with unchanged frozen payload |
| Does the promising readout work when representation and decoder learn together? | Only after the TRAIN signal gate: fresh readout attachment to the common H e1000 parent and matched e2500 endpoints |
| Is the tradeoff practically useful? | Matched native fields, complete calls, prepared scopes, forward/extraction memory, loss-backward memory, and actual optimizer memory reported separately |
| Is the original organization preserved? | Exact frozen P/means/moments/Z checks in the frozen study; unchanged construction law but potentially different learned P in joint confirmation |

**No fresh full-TRAIN run, new PDE/CFD/reference solve, inverse search, renewed sparsification or loss-balancing campaign is authorized by this plan.** Jointly trained candidates can be assessed for eligibility under the standing bands; a later fresh full-TRAIN experiment remains a separately authorized task. This prevents an expensive diagnostic from silently becoming another formal campaign.

## 1. What Round5 establishes, and what remains a hypothesis

Use the report's literal-e2500 native boards, not the selected-age board, as the principal historical reference. All figures and numerical links in Round5 remain historical and protected. Its supplied Markdown was reviewed; its locally linked checkpoints, arrays and figure payloads must be opened and verified by Codex in the research workspace before being reused. The planning review did not independently rerun those checkpoints.

| Native quantity | D0 | Round5 C0 | Round5 M | Round5 H |
| --- | ---: | ---: | ---: | ---: |
| Thermal fluid-T mean RMSE | 0.877178 | 0.923182 | 0.879167 | 0.789884 |
| Thermal v mean RMSE | 0.00307437 | 0.00446099 | 0.00411811 | 0.00406619 |
| Thermal p mean RMSE | 0.0112959 | 0.0160367 | 0.0154728 | 0.0149746 |
| Wind near-vector mean RMSE, m/s | 0.234814 | 0.627808 | 0.624931 | 0.629116 |
| Wind downstream-vector mean RMSE, m/s | 0.171749 | 0.505084 | 0.503453 | 0.502916 |
| Wind H near Ux / Uy / Uz mean RMSE, m/s | 0.227819 / 0.0477233 / 0.0302226 | — | — | 0.622367 / 0.0811185 / 0.042298 |
| Wind background-vector mean RMSE, m/s | 0.0389066 | Read original board | Read original board | 0.144862 |

Thermal H improves fluid-T mean by about 9.95% against D0, as well as by 14.44% against C0 and 10.16% against M. Its v/p means are still about 32% above D0. The temperature gain must not be treated as uniform thermal-role recovery: H's surface/material means and some small-module strata remain weak. Wind H is close to M/C0 and far above D0; its fixed TRAIN4 near-vector mean is 0.443487 m/s. This is not solely an unseen-layout failure.

Round5 measured B8/Q8192 native forward/extraction allocated memory of 0.4144 GiB for Thermal H and 0.6238 GiB for Wind H, compared with D0's 1.7721 and 4.1882 GiB. These are **not** the Round4 target-free backward experiment. Complete high-M H/D0 latency ratios were 1.3636 and 1.3577. The small prepared-query gains for Wind cannot be substituted for complete-call gains or extrapolated to a new reader.

The previous TRAIN diagnosis found broad within-cell errors rather than a consistent boundary-localized cause; both optional jets were deferred. Do not reinstate jets merely by repeating the already-known smoothstep argument. The Thermal gradient diagnostic found small positive cosines and a small flow-to-thermal norm ratio, not demonstrated destructive gradient conflict. Do not prescribe PCGrad, FAMO or reweighting as an established remedy.

The processor-curvature result is especially important: H's FP64 descriptor mixed effects were below the matched affine FP32 cancellation control at the tested tiny increments. Computational use is established; a resolved ordinary-FP32 higher-order physical effect is not. Round6 should prioritize field recovery over another campaign to produce nonzero interaction pictures.

### 1.1 There are two different aggregation boundaries

For one fixed typed head, the ordinary preparation includes

\[
G=PV.
\]

It is not generally valid to assert that this operation necessarily destroys all source distinctions. For example, a matrix P with E rows and M columns can have full column rank when E is at least M. Wind has E64 and at most 30 physical modules. Several distinct mixtures can collectively retain information that a single mean loses. This is a conditional linear-algebra observation, not a finding that the actual trained P or the fused Z is invertible.

The later receiver boundary is

\[
c_q^0=\sum_{a\in\mathcal A(q)}\Lambda_{qa}Z_a+b_g.
\]

Here four/eight vectors are collapsed into one vector before the field head acts. At a fixed query and fixed other inputs, different anchor packets with the same weighted sum are indistinguishable to this context path. A nonlinear message formed **before** summation can distinguish some such packets. This is the restriction Round6 tests first.

Neither statement is a theorem that the complete predictor cannot learn localized fields: query features, local source reads, globally contextualized tokens and nonlinear downstream heads remain present. Distinguish a restriction of an intermediate map from a proved lower bound on the whole learned model.

### 1.2 Preserve the actual hypergraph semantics

The inherited P tensors have typed, head-specific donor-to-anchor views. The current M/H processor consumes the concatenated descriptor and produces **one fused width-128 state per environment anchor**. Do not draw four independent processed Z states per anchor merely because P has four heads. The new receiver adapter operates on that fused state; its new bottleneck channels are not additional physical heads.

An environment anchor is a location at which information is stored. It is not necessarily a turbine, a wake origin, a material parcel or the centroid of its donors. Query–anchor offsets provide signed spatial addressing; they do not by themselves restore a lost query–turbine displacement. This distinction must remain visible in the Wind interpretation.

## 2. Source-level review and edit boundaries

Paths below are relative to the repository root. Symbols are authoritative if line numbers move. Preserve historical classes, schema dispatch, recipes and checkpoint loading.

| Inspected source | Finding | Round6 action |
| --- | --- | --- |
| `HONF_Proj/src/honf_forward_core/inherited_attention/model.py`, `_prepare_scene_readout`, `_gather_scene_readout`, `_decode_features` (reviewed approximately lines 664–789) | A bias-free anchor projection is gathered; one global context reaches both `field_head` and `response_phi`; query encoding already occurs before the gather | Add a narrow opt-in receiver-read hook that receives the already-computed query state. Do not copy the complete decoder or run the query encoder twice |
| `.../inherited_attention/moment_model.py`, `prepare`, `_attach_moment_hyperedges`, `_prepare_scene_readout` | H produces the fused processed anchor state after the base C preparation | Build readout projections only after the complete H state exists. Use the same H payload in all primary arms |
| `.../inherited_attention/moment_model.py`, `_record_moment_work_counts` (approximately lines 297–390) | Preparation still traverses cache storage and copies audit counts to CPU; this path is called for M/H even though the base exact-counter option can be false | Profile this actual path. Make optional statistics truly optional for new measured execution, retaining mandatory numerical checks and explicit unavailable counters |
| `.../inherited_attention/moment_model.py`, `_source_transport_coefficients`, `_anchor_response`, `heat_vjp_prepared_affine`, `export_source_kernels` | Source-linear heat factors are built from raw factors; ordinary application materializes `[B,H,E,R,M]` coefficients | Preserve this law. One optional same-function moment-contraction executor may avoid materializing that coefficient tensor for ordinary heat application; validate separately |
| `.../inherited_attention/cover_execution.py`, `_weighted_projected_cover_sum`, `gather_projected_cover_indexed` | Indexed selected values are multiplied by live cover weights then summed | Reuse validation and selected indexing; add a compact, tiled pre-pooling nonlinear reader rather than a dense source-value path |
| `.../inherited_attention/model.py`, prepared-state dataclasses and `_factors` | Ownership, weight/buffer versions, factor versions and graph consumption are checked | Stamp the new readout projections. Do not permit stale projections after reader/head updates or share owner-bound caches across model instances |
| `.../content_hypergraph/readers_v2.py`, `DirectSourceReaderV2` (approximately lines 296–467) | D0 makes a nonlinear query/source/relative-geometry message before source reduction | Use this as the mechanistic contrast, not as a silent bypass in a hypergraph candidate |
| `.../content_hypergraph/readers_v2.py`, `QueryEncoderV2`, `LocalDetailReaderV2`, `FieldHeadV2` | Query geometry/operation and the existing compact source path are separate from global pooling | Freeze query/local parameters in the primary isolation; keep local support unchanged throughout this round |
| `HONF_Proj/Case_WindFarm/src/windfarm/content_hypergraph.py` | Native wind-aligned rotor-diameter frame, E64 and `WIND_CONTENT_LOCAL_RADIUS_D = 1.5` are explicit | Preserve rotation, profile, normalization, features and local support. Measure coverage rather than assume every near-role receiver has a local read |
| `HONF_Proj/tools/moment_hyperedge_train.py` | Round5 CLI/recipe guards intentionally admit own C-soft e500 attachments and stops through e2500 | Add a separately versioned Round6 entrypoint/family; do not loosen the Round5 protocol to admit unrelated freezes or ages |
| `HONF_Proj/src/honf_runtime/moment_hyperedge_factory.py`, `moment_hyperedge_child.py`, `moment_hyperedge_protocol.py`, `content_hypergraph.py`, `unified_training.py` | Existing native factories, child handling, source bindings and TrainingEngine supply the integration points | Inspect these helpers fully during implementation; reuse their engine/provider machinery with explicit Round6 trainable-scope and age rules |
| `AGENTS.md` | Storage, native evidence, checkpoint cadence, source formatting and outgoing-range audit are mandatory | Follow the current applicable instructions; do not copy old GPU permissions or historical free-space values |

The review identifies opportunities, not measured operator-level explanations for the entire 36% latency penalty. Profile before assigning a percentage to a kernel or to validation overhead.

## 3. Primary receiver model: same information, different reduction order

Use working names `R-P` (pool first) and `R-E` (edge processing first). These are not P attention, the historical P family, or new memberships. Retain Round5 H's full-soft EM/EE construction and its existing projected context as the common base.

### 3.1 Inputs and dimensions

For each scene let `Z[B,E,128]` be the exact completed H processed state; let `q_state[B,Q,128]` come from the unchanged query encoder. The cover supplies `anchor_indices[B,Q,C]`, weights and validity, with C=4 in Thermal and C=8 in Wind. Both arms use the same source/native identities, original query features and query masks.

Add only four trainable linear maps and one hidden bias:

\[
A_Z:\mathbb R^{128}\to\mathbb R^{64},\quad
A_q:\mathbb R^{128}\to\mathbb R^{64},\quad
A_g:\mathbb R^{7d}\to\mathbb R^{64},\quad
W_o:\mathbb R^{64}\to\mathbb R^{128},\quad b_r\in\mathbb R^{64}.
\]

All maps have no bias except the explicit b_r. There is no learned receiver score, top-k rule, extra state processor, layer normalization of individual Z vectors, new type mixing, or additional source information in the primary pair. The exact new parameter inventory must match between R-P and R-E. Width 64 is fixed before fitting; do not search widths.

Prepare `A_Z Z_a` once in each valid prepared scene forward. Compute `A_q q_state` once per receiver, not once per cover slot. Apply W_o after the hidden-state reduction, not at every slot. Preparation projections depend on the new trainable weights and must be rebuilt after they change, even while the original H bank is frozen.

### 3.2 Signed relative geometry

Use a common smooth seven-channel-per-coordinate feature map

\[
\psi(r)=\big[\operatorname{asinh}(r),\sin(\pi r),\cos(\pi r),
\sin(\pi r/2),\cos(\pi r/2),\sin(\pi r/4),\cos(\pi r/4)\big].
\]

For Wind, r is the existing signed native `(query_D - anchor_D)` vector, in downstream/crosswind/vertical order; one coordinate unit is already one rotor diameter. Do not rotate it again, divide by D twice, take absolute streamwise distance, or replace the native coordinates with a whole-farm normalized distance that removes the chosen D-scale addressing. This is a receiver feature convention, not a new output transform.

For Thermal, r is `(query - anchor) / frame_scale`, using the existing explicit scene frame. No unverified SI length or anchor-as-heater radius is introduced. The two datasets share the reader implementation but own their coordinate-scale metadata. Freeze these conventions in the Round6 schema and test them. Both R-P and R-E receive exactly the same geometry features; an observed difference is not the result of giving geometry to only one arm.

Do not add donor centroids, nearest-source IDs, individual source values, target-dependent wake widths, or per-case embeddings to this pair. The existing legal native query features remain unchanged. Query–source geometry can be used to stratify errors offline, not smuggled into one candidate's inputs.

### 3.3 The matched maps

Let

\[
w_{qa}=\Lambda_{qa}\,\mathbf 1_{qa}^{\mathrm{valid}},\qquad
s_q=\sum_a w_{qa},\qquad
u_{qa}=A_ZZ_a+A_g\psi(r_{qa}),\qquad v_q=A_q q_{\mathrm{state}}+b_r.
\]

The common existing context is

\[
c_q^0=\sum_a w_{qa}Z_a+b_g.
\]

The pool-first control is

\[
\delta c_q^{P}=W_o\operatorname{SiLU}\left(s_qv_q+\sum_a w_{qa}u_{qa}\right),
\qquad c_q^P=c_q^0+\delta c_q^P.
\]

The edge-first candidate is

\[
\delta c_q^{E}=W_o\sum_a w_{qa}\operatorname{SiLU}(v_q+u_{qa}),
\qquad c_q^E=c_q^0+\delta c_q^E.
\]

Use the actual floating-point s_q, not an assumed exact one, for the control's query/bias term. Normally valid cover rows have unit mass within the existing contract, while padded queries have zero mass. With identity activation instead of SiLU these maps agree algebraically, including subprobability primitive fixtures. This supplies a clean control for reduction order. Do not silently renormalize the original cover.

Initialize A_Z, A_q and A_g identically by name with nonzero weights; set W_o exactly to zero and b_r to zero. Do not add a second zero gate. Both arms initially recover the parent function, while W_o has a usable first-update gradient and the input projections obtain gradients after W_o moves. Do not bypass the residual computation by detecting that W_o is zero: that would remove its learning signal.

The existing field head and Thermal response_phi consume the resulting context. This is a receiver-context residual inside one predictor, not a second physical network. R-P is deliberately stronger than unchanged H because it receives the same added capacity and geometry as R-E. H e2500 remains the fixed historical reference; it is not an equal-additional-training control.

### 3.4 Why this is not the failed Round4 IAG-Q

IAG-Q formed query-specific probabilities over cached **donors** and read their cached values. R-E operates on the already prepared **hyperedge states**, retains the same Lambda and applies a signed-geometry-conditioned nonlinear map to each selected state before reduction. It neither reconstructs `beta = Lambda P` for prediction nor revisits original source values. Explain this distinction in the report rather than presenting query conditioning as a previously untried general idea.

R-E also differs from adding another scene processor: the new nonlinearity depends on the receiver. A scene-only F cannot implement arbitrary query-dependent processing merely by being deeper.

### 3.5 Memory and work contract

The ordinary new nonlinear tensor may have shape `[B,Q_tile,C,64]`, never `[B,Q,M,128]` or `[B,Q,E,128]`. Use existing native query tiling and a fixed documented tile limit; do not shrink B or Q to produce an apparent saving. A small literal reference implementation may allocate a complete tensor in tests only.

Report the actual selected slot count, positive-weight count, 64-channel nonlinear entries, projection rows, geometry feature entries, retained readout bytes, maximum live tile bytes and execution time. At high M, this evaluates fewer width-scaled nonlinear entries than D0's full source reader, but it evaluates more than Round5's simple gather. These counts are not a predicted speedup. Low-M cases may have no arithmetic advantage at all.

## 4. Thermal law: improve receiver discrimination without changing heat transport

In the primary pair, keep the complete Round5 H source-carrier and heat-mixing preparation frozen. In particular, keep

\[
C_{ahri}(\mathcal X)=\frac{P^M_{ahi}}{m_i}t_{ihr}
\left[1+a^0_{ahr}+\sum_\ell a^\ell_{ahr}(\xi_i-\xi_a)_\ell\right]
\]

and the existing Lambda transport unchanged. New contexts can change the learned heat-independent receiver factor phi through the existing response_phi head, but no actual h enters Z, the reader, query features or phi:

\[
T(q,h)=\sum_{a,h_d,r,i}\Lambda_{qa}\phi_{q h_d r}(\mathcal X)
C_{a h_d r i}(\mathcal X)h_i+T_{\mathrm{local}}(q,h).
\]

Here h_d denotes an attention-head index, not heating. Prefer distinct names in code and equations. All physical source columns remain explicit. Flow remains heat-independent, and zero heating, increments, JVP/VJP and source-axis permutation laws are preserved.

The new readout changes only the heat-independent query-side factor, not the source transport itself. For R-G, global access affects the context used by phi but does not secretly replace the original Lambda in the heat contraction. Do not interpret that diagnostic as relaxing every possible thermal-transport restriction.

Precise FP64 checks must cast the existing raw factors before their contractions. Keep ordinary FP32 subtraction failures visible under the original thresholds. Recompute material maxima at each endpoint. Do not impose positivity on learned K, relabel negative coefficients as physical cooling, or add a conservation penalty unsupported by this benchmark.

## 5. Stage A — protected baseline, correctness and zero-update diagnosis

Read current AGENTS instructions and inspect the working branch, uncommitted user changes, remote state, source/report hashes and any commits after the reviewed aec07be. Do not reset user work. Record the current applicable GPU authorization; the latest report uses physical GPUs 1/2 and protects GPU0. Verify UUIDs and occupancy rather than reuse process IDs from Round5.

Verify the real `/data` mount and expected filesystem UUID before resolving caches or creating output directories. The previous external unmount must not result in writing a new apparent dataset into an empty mount point. Inventory available space, shared-cache bindings, protected roots and the round-owned output budget first.

Load the trusted Thermal H and Wind H **literal e2500** checkpoints through maintained native factories, without constructing a training provider merely to perform a small inference load. Resolve paths and SHA256 from the retained Round5 receipt; do not guess checkpoint filenames from report labels. Also locate the H e1000 checkpoints for the conditional joint stage, without modifying them.

### A1. Baseline reproduction

Reproduce the saved fixed TRAIN4 native outputs/metrics for each H parent and bind the original transforms, native IDs, masks and query roles. Reuse exact original queries and retained numerical arrays. Load D0 and Round5 C0/M only for the small declared comparisons that need them; do not rerun their entire historical evaluation campaigns. Record any unavailable legacy per-case hashes as unavailable.

Before training, prepare matching R-P/R-E scenes and verify P, full typed means, moments, descriptor, processed Z, source carriers, heat mixes, query states and local outputs match the parent and each other. Check both model values and required native derivatives at zero residual. Model-instance ownership tokens differ legitimately; numerical factor equality does not permit passing one instance's bound cache to another.

### A2. Distinguish compression, access and optimization hypotheses

On the existing fixed TRAIN4 only, record singular spectra and numerical-rank sensitivity of the module P per head and of the stacked cover-accessible P rows for the input-fixed probes. Use more than one stated numerical-rank threshold, record row/column scaling, and label the measurement as a linear mixing diagnostic. E64>=M does not guarantee well-conditioned recovery; poor conditioning does not prove failure of the nonlinear network. Do not insert a pseudoinverse, source reconstruction or SVD-selected mode bank into inference.

Record variation across selected Z packets and the share of variation lost by their Lambda-weighted mean. Compare spatially nearby queries without treating them as independent scenes. Latent norms are not physical influence. Keep complete bank and local packet conclusions separate.

For Wind, audit native Ux error by role, signed residual mean, residual variance and downstream distance. At the per-scene/per-role level use `MSE = bias^2 + centered_error_variance`, then retain the declared case/layout reducers; do not decompose the arithmetic mean RMSE as if it were a pooled MSE. Use the unchanged profile and report `U_profile_x - U_x` for deficit depth/recovery. A background-role label need not mean the point has absolutely no upstream influence.

Measure the proportion of native near/downstream/background receivers with zero, one or overlapping active 1.5D local reads, as well as support-weight mass. Stratify by signed source-relative downstream/lateral/vertical offsets using inputs only. Report within-local-support and outside-local-support errors. Do not infer that changing 1.5D is the remedy merely because many downstream points have no local read.

For Thermal, retain native u/v/p/curl and all temperature/q/peak roles. Reuse the existing phase diagnosis rather than commissioning another broad cover-boundary campaign. No primary readout decision is based on DEV images.

### A3. Pooling-nullspace primitive test

At a fixed valid query with two nonzero cover weights, construct small controlled changes to two Z vectors such that `w1*delta_Z1 + w2*delta_Z2 = 0`, keeping geometry and every other input fixed. Use a moderate declared norm, not a near-cancellation-sized perturbation. The original pooled context and R-P must remain unchanged within arithmetic tolerance. With nonzero, nondegenerate test reader weights, R-E should be able to respond. Repeat with identity activation, where both readers must agree.

This establishes an implemented representational distinction. Such independently changed latent vectors need not correspond to a realizable physical scene, and their effects are not physical multi-body truth. The trained R-E response can be small or zero; do not tune examples on DEV until a large effect appears.

## 6. Stage B — the primary frozen-representation learning comparison

### B1. What “fixed” means

Freeze the contextualizer, raw/scene injections, all encoders including the query encoder and operating encoder, typed value projections/norms, moment lifts, hyperedge processor, gather_output, source carriers, local reader and all preparation/output-normalization buffers. Do not advance an otherwise inert normalization buffer or update any running statistic. Freeze weights using `requires_grad_(False)` and an exact optimizer allowlist, not gradient hooks that alter values or block required input derivatives.

Train **only** the new receiver adapter plus the existing `field_head` and, in Thermal, `response_phi`. The two arms have identical trainable tensor names, shapes, starting values, optimizer specifications and objective reductions. Local source parameters and support stay fixed. This isolates the additional receiver map while allowing both competitors to adapt their common output decoding.

This is a deliberately bounded frozen-representation diagnosis, not a revival of a frozen D-sep flow partner or a production two-stage predictor. No historical classic supplies field outputs. Inference remains one native model with one preparation and one decoder.

Use a new fit identity and fresh AdamW state for **all trainable fit parameters**, including the adapted inherited heads. Do not describe this as exact continuation of the original all-parameter optimizer. Store the parent identity and age separately. On exact resume of the new fit, restore its own complete optimizer/RNG/history state.

Do not reuse a whole PreparedInheritedOrganization across an optimizer update and disable its version validator. The safest initial implementation reprepares the frozen scene each forward, which still gives the same numerical bank. Reusing immutable raw frozen features is optional and must have a separate, scoped contract; trainable A_Z projections must always be refreshed. Preserve geometry/query differentiation by rebuilding live graphs whenever derivatives with respect to inputs are requested.

### B2. Data, native losses and exposure

| Setting | Thermal | Wind |
| --- | --- | --- |
| Parent | Round5 H literal e2500 | Round5 H literal e2500 |
| Selected TRAIN / exposed DEV | Existing 150 / 22 | Existing 72 rows / 24 rows, grouped into 24 / 8 layouts |
| Effective / microbatch | 48 / 24 | 16 / 8 |
| Training queries | Existing Q1024 fluid + unchanged auxiliaries and scheduled TRAIN response | Existing Q8192: 6144 volume + 2048 hub |
| Updates per full fit epoch | 4, including final six-case macro | 5, including final eight-row macro |
| Objective | Original complete Round5 Thermal objective | Original 0.75 volume / 0.25 hub objective and component scaling |
| Optimizer | AdamW 3e-4, weight decay 1e-5, betas .9/.999, eps 1e-8, clip 1 | Same |
| Primary fit horizon | 500 new full selected-data epochs | 500 new full selected-data epochs |
| Conditional common extension | Up to 1000 fit epochs, not automatic | Up to 1000 fit epochs, not automatic |

Use the original sampling namespace and deterministic seeds, with `sampler_absolute_epoch = 2500 + fit_epoch` for both arms; encode this mapping explicitly. Retain the mature original native loss/callback rules. The learning history's main age is **fit age**, not a fabricated all-parameter epoch 3000/3500. Each fit epoch visits every selected case/row. No repeated tiny batch counts as a development epoch.

Both arms receive a real 100-epoch health review, then normally complete 500 if finite and within resource limits. Failure to meet a final field gate at 100 is not a reason to stop a healthy comparison. A one-/two-epoch smoke only establishes startup and belongs to the disposable engineering ledger.

Save milestones/latest/best every 100 fit epochs and the final/clean stop; refresh curves every 10 epochs with tqdm. Retain every validation result in history, without separate files for every ordinary review. The secondary checkpoint selectors remain the original Thermal field_score and Wind volume monitor; literal common fit age is primary. Do not select a new wake-optimal checkpoint retrospectively.

### B3. Evaluation panels and the extension rule

Reuse the input-fixed TRAIN4 for detailed native fields. In addition, define one small, deterministic, fixed-weight fitting audit covering all selected TRAIN cases/rows: Thermal Q1024 with the existing sampled auxiliaries; Wind Q1024 total across the five roles, with fixed role counts 205/205/205/205/204. This is **evaluation only** and does not change training query probabilities or loss weights. Declare its native IDs before training; preserve role overlaps. Stream it rather than create a duplicate dataset. It is TRAIN fitting evidence, not a new validation split.

The fixed low-budget all-TRAIN audit complements the detailed TRAIN4; its numbers are not interchangeable with full-native DEV or the old TRAIN4 board. Evaluate it at fit 0/100/300/500 and, if extended, 1000. Evaluate exposed DEV using the unchanged full native board at fit 0/500/final and the unchanged monitor at its ordinary cadence. Reuse outputs whenever selected and endpoint weights coincide.

A common 500-to-1000 extension is allowed only if either arm's all-TRAIN priority score improves by at least 3% from fit300 to fit500, or its actual native training objective's final 100-epoch median improves by at least 3% over fit301–400, with finite gradients and enough reserved time to complete **both** arms and their final evidence. Record this decision from TRAIN evidence before using new DEV outcomes to choose an action. No LR, batch, loss, width or schedule search is allowed. If the condition is not met, retain 500 as the completed diagnostic horizon.

### B4. Predeclared readout signal

For Wind define the main fitting contrast as the geometric mean of the R-E/R-P **near Ux** and **downstream Ux** native mean-RMSE ratios. For Thermal use the geometric mean of the **v** and **p** ratios. Calculate ratios between the same audit panels, using the existing physical/native units for each field before forming dimensionless ratios. Report individual means, tails and absolute errors as well.

A useful local-readout signal requires at least a 10% reduction in that priority contrast on the all-TRAIN audit, no named priority component worsening by more than 5%, and the same improvement direction on at least three of the four detailed TRAIN fixtures. Wind additionally requires near/downstream vector means not to worsen by more than 5% and volume/background Ux and vector means not to worsen by more than 10%. Thermal additionally protects fluid/surface/material T and q means within 5% of R-P and u/curl means within 10%. If a protected baseline error is numerically zero, report absolute error and unavailable ratio rather than invent a stabilizing denominator.

R-E must also improve the same all-TRAIN priority contrast by at least 5% against the zero-fit H parent and must not worsen the original fixed-weight native training objective against that parent. A favorable R-E/R-P ratio produced only by deterioration of R-P is not a recovery signal. These parent comparisons use the same audit support and reducers.

This gate identifies a mechanism worth jointly training; it is not statistical significance or deployment qualification. Report paired per-case/layout distributions and every tradeoff. Do not require the frozen representation to beat D0 to count a useful diagnostic, but do not call a small change from a poor Wind baseline “wake recovery.”

## 7. Stage C — one conditional full-bank access diagnostic

Run this only in a dataset where the local R-E/R-P study does not produce a useful local-readout signal, or where it does but the all-TRAIN deficient-field error remains more than twice the matched saved D0 fitting error and the diagnosis is still about access. Establish the D0 fitting comparison on the same query support; otherwise the ratio condition is unavailable. The decision uses TRAIN, not favorable DEV field pictures.

`R-G` starts independently from the same H e2500 parent and the same initial new reader/head tensors as R-E, with fresh fit optimizer state and the identical fit-age query streams. Complete the same terminal fit age as the local pair, 500 or 1000. Do not warm-start R-G from a trained R-E and compare it against a shorter-trained local arm.

Keep the original local base context c0, but replace only the correction's four/eight-slot reduction with a full-bank reduction:

\[
\delta c_q^{G}=W_o\sum_{a\in\mathcal E_{\mathrm{valid}}}\nu_a
\operatorname{SiLU}\big(A_q q_{\mathrm{state}}+b_r+A_Z Z_a+A_g\psi(r_{qa})\big).
\]

Use the existing positive environment-anchor measure normalized over valid anchors as nu; for a uniform lattice this is uniform. This is a **receiver-side integration weight over cached anchor states**, not an extra multiplication of donor measure inside P. State that the correction has global support with fixed weights; do not call it learned sparse access. Keep the same new parameter count and signed geometry features as the local R-E arm. No new key network, dot-product router, physical donor, extra moment, or bank processor is added.

Tile queries and, where necessary, anchor reductions without retaining a full `[B,Q,E,64]` training tensor. Use a literal dense small-tensor oracle for value/gradient checks. Preserve required higher derivatives. Price every E64/E192 read; global access is not expected to have the local reader's cost.

Call R-G a useful access signal only when its all-TRAIN priority contrast improves by at least 10% against both the equal-fit-age local R-E and the zero-fit H parent, with the Section 6.4 nonpriority guards applied against those comparators and the original native training objective not worsening. Smaller or mixed changes remain measured diagnostic results.

A better R-G would show that this broader readout of the **same** bank is useful; because the integration weights also change, it does not uniquely identify support radius as the only cause. It is not an oracle upper bound on all decoders and it does not justify deploying the expensive global path. A failed R-G fit does not prove that no decoder could use the bank successfully.

Retain the result as an access-relaxation diagnosis. Do not add an automatic global-to-sparse distillation, anchor selection, source bypass or new end-to-end R-G campaign in this round. A global-only positive result points to a later query-access design task.

## 8. Stage D — conditional matched joint confirmation, not frozen-bank evidence

This stage is allowed independently for each dataset only after the local R-E signal in Section 6.4, clean native-law checks and a prospective resource forecast. It is optional; it is not needed to complete a negative but informative Round6.

Use each dataset's retained Round5 **H e1000** as the common parent, not e2500 and not a classic/D0 checkpoint. Attach new R-P and R-E modules at zero output using identical named initial tensors. Do not import parameters, learned geometry scales, fitted heads or optimizer moments from the frozen e2500 diagnostic. That would contaminate the equal-age confirmation with extra trained exposure.

Continue all original H parameters and both common output heads jointly, retaining inherited named optimizer moments for existing parameters; initialize moments only when new reader parameters first receive an update. Use the unchanged original data, objectives, effective batches, sampling streams, constant LR and absolute schedule. Continue 1500 new full selected-data epochs to literal **e2500**. Every healthy admitted pair receives this complete horizon unless a correctness failure or hard resource interruption prevents it. Save/review at the original 100-epoch cadence, with health reviews at e1100/e1500/e2000 and the final e2500 board.

This comparison holds the **organization rule and payload architecture** fixed, not the numerical P/eta/Z. Those tensors can differ as the two models coadapt. Say this explicitly. It is the practical test of the readout modification, while Stage B supplies the strict fixed-representation isolation.

At equal absolute e2500, compare new R-P/R-E with the preserved R5 H and D0 native boards, verifying target/query bindings. The new pair is the controlled comparison; historical H/D0 have their own numerical runtime/lineage caveats. Reuse historical predictions where valid, but measure all cost ratios anew on the same device and scopes.

Use the standing research qualification bands without relaxation: priority means <=1.10 times D0, priority p90 <=1.20, all other means <=1.15. Thermal priorities include fluid T/v/p; Wind includes near/downstream vectors and all six components. The efficiency route requires either complete high-M latency <=0.90 times D0 or B8/Q8192 forward/extraction allocated memory <=0.80 with complete latency <=1.15. The accuracy-led route requires named fluid-T or near-vector mean <=0.95 times D0, other guards preserved, and latency/memory <=1.10. Also disclose retention or loss of R5 H temperature and saved-response gains.

A joint candidate that meets these conditions can be labelled **eligible for a separately authorized fresh full-TRAIN probe**. Do not launch it here. A frozen e2500-plus-fit model cannot satisfy this equal-age qualification by being compared to an uncontinued D0 e2500.

## 9. Execution diagnosis and bounded same-function work

### 9.1 Profile the actual complete path

Before optimizing, collect one short profiler trace for each native dataset on the fixed high-M/Q8192 H fixture. Separate host packing/transfers, scene preparation, base contextualization, moment preparation, query/local planning, new reader, native field/derivative extraction, synchronization and return transfer. Use nested scopes honestly: do not add inclusive times or subtract medians from independently timed calls.

Keep profiler-off latency measurements authoritative. CUDA event time, host wall time, CPU wait and complete application latency are different quantities. Do not allocate GPU0 or benchmark over another user's active workload. If no uncontended authorized window exists, label contention and do not decide a tight latency gate from it.

### 9.2 Optional exact-counter correction

In the inspected M/H preparation, `_record_moment_work_counts` gathers audit counts via `.detach().to(...).cpu().tolist()` and traverses storage metadata. A base `collect_exact_work_counts=False` does not automatically remove this work. Make the new execution mode omit optional counts/storage traversals on the timed fast path while preserving every mandatory finite/variance/identity check and meaningful errors. Use an untimed diagnostic pass to obtain exact work records. Unmeasured counts must be marked unavailable or omitted, never recorded as zero work.

Keep an explicit historical/diagnostic counter-on mode and test identical predictions, loss terms, input/parameter derivatives and optimizer steps. Do not use the performance task as permission to remove cache ownership, masks, version checks, native extraction or mandatory numerical checks. Code-level removal of a synchronization is an opportunity, not proof that it caused the observed slowdown.

### 9.3 Optional Thermal contraction reassociation

One bounded alternative may compute heat-weighted moments directly:

\[
R^{(0)}_{ahr}=\sum_i(P^M_{ahi}/m_i)t_{ihr}h_i,
\qquad
R^{(\ell)}_{ahr}=\sum_i(P^M_{ahi}/m_i)t_{ihr}h_i\xi_{i\ell}
-\xi_{a\ell}R^{(0)}_{ahr},
\]

\[
R_{ahr}=(1+a^0_{ahr})R^{(0)}_{ahr}+\sum_\ell a^\ell_{ahr}R^{(\ell)}_{ahr}.
\]

This is the existing law, not an added spatial-moment model. It can avoid the ordinary `[B,H,E,R,M]` coefficient materialization. Maintain the literal coefficient implementation as the oracle and explicit-K export path; derive a matching transpose for the new contraction rather than return an unrelated autograd approximation. Casting must precede contraction in the precise path. Qualify outputs, arbitrary test cotangents, heat gradients, geometry gradients and double backward. If the alternate executor misses a fixed native criterion, retain the original executor; do not delay the readout study for an open-ended arithmetic repair.

### 9.4 Do not repeat the failed compiler assumption

Round5's installed AOTAutograd route did not support the native double backward. Eager Torch is the default. No framework upgrade, custom CUDA/Triton project or whole-model compilation is required. Any optional backend must pass actual native higher-derivative tests before training uses it; an inference or first-gradient pass alone is insufficient.

Limit all optional same-function tuning to two engineering process-hours within the total budget. Seal the chosen execution before scientific fits and use it consistently across compared arms. Do not hot-edit an active source snapshot.

## 10. Native diagnosis and scientific controls after fitting

### Wind

Report all five roles and all three components, with near/downstream Ux primary. Keep vector RMSE as `sqrt(mean(sum(component_error^2)))`, without dividing by three. Use m/s. Retain row summaries and group correlated directions by layout for distributional interpretation. No cell-level bootstrap creates independent layout evidence.

Use original stored native cut locations and planes. Show deficit depth, crosswind width, displacement, downstream recovery and signed background error without interpolating targets into a more favorable plane. Source-relative bins may distinguish upstream, 0–1.5D, 1.5–4D, 4–8D and farther-downstream distances where data exist; state overlap and counts. They are descriptive diagnostics, not new training masks.

Keep the background profile and normalizer identical. Do not subtract an observed per-scene error bias at inference. Bias/variance decomposition can diagnose a uniform residual offset; it cannot itself authorize correction using targets.

### Thermal

Report fluid T/u/v/p/curl, surface/material T, q proxy and material endpoint peaks. Use original masks and native extraction. Compare shared-parameter gradients only as diagnostics; no additional adaptive loss method is introduced. Temperature accuracy and the source-linear heat law remain separate checks.

Reuse the saved 0291/0294/0687 signed heat-transfer records at terminal jointly trained candidates, if that stage ran. Frozen fits need only the bounded native heat-law checks and one aligned saved-response sanity replay; do not repeat a large response campaign for every monitor. The missing 0277 baseline remains unavailable. No new target, solver call or inverse result is synthesized.

### Organization and readout use

On the fixed original probes, retain P, Lambda, Z and the actual new per-edge hidden messages, their weighted reduction and the final context. For R-P, plot its actual pooled preactivation and resulting correction, not invented per-edge nonlinear messages. For R-G, show all accessed anchors and their fixed nu; do not top-k the executor and label it full support.

Repeat one actual association intervention and exact restoration per donor type on each fixed scene, recomputing every dependent H state and new readout projection. Preserve a no-intervention original organization. Same-weight intervention effects measure use, not net training benefit. A new residual removal uses that arm's coadapted base and is not equivalent to the independently trained comparator.

Use the pooling-nullspace test and trained message variation as mechanism evidence. No new four-/eight-state geometry experiment is required. Do not keep searching descriptor directions until processor curvature becomes visually large; no compatible new physical interaction reference is available.

## 11. Required tests before scientific training

### Mathematical reader tests

- Tiny explicit reference versus indexed/tiled implementation in FP64 and FP32; identity-activation R-P/R-E equivalence with the actual row-mass convention; exact initial zero-residual parent recovery without suppressing new gradients.
- Full parameter-count and initial-tensor equality between the primary pair; first-update nonzero W_o gradients and subsequent nonzero input-projection gradients using actual native objectives.
- Masked queries, empty/zero-weight slots, duplicate anchor indices, valid singleton axes, padding and source/anchor permutations; no fabricated response from a masked row.
- Independent `gradcheck`/`gradgradcheck` on small smooth fixtures, then real native higher-derivative checks. Compare query-coordinate, selected-anchor-coordinate, Z and parameter gradients; test away from discrete cell changes as well as one-sided behavior at boundaries.
- Primitive pooling-nullspace controls; both readers agree when the activation is affine, while nondegenerate edge-first test weights can distinguish equal-mean packets.

### Prepared-state and integration tests

- New projections invalidated by a reader optimizer step, a parent-weight update, buffer mutation, geometry/frame/normalizer changes, source ID/order changes or atlas intervention. Consumed autograd graphs cannot be reused.
- The frozen study's parent parameters/buffers are bitwise unchanged after full native epochs; optimizer parameter IDs exactly match the allowlist; P/moments/Z equality persists at matched inputs through the fit. Hash outside the timer, not once per GPU query.
- Freezing parameters does not incorrectly detach input gradients. Inference and input-VJP paths must reprepare live geometry as needed. A stale frozen-feature cache cannot masquerade as the ordinary differentiable model.
- Query encoder called once per decode; new readout projection prepared from final H Z, not preliminary C state; local support/pairs and head order unchanged; trace-on/off outputs identical.
- Native Thermal heat-null, flow independence, affine increments, random-sign JVP/VJP, source-axis transpose/export agreement and endpoint maxima; native Wind zero residual maps to the original profile with no normalizer mean or second rotation.
- Chunk sizes 1/7/512 and a normal native chunk, mixed-M batches and partial macro-batches; original native output and gradient tolerances are recorded before comparisons. Retain the inherited generic FP32 miss as a historical limitation, not a newly passed universal guarantee.
- Actual TrainingEngine attach/save/resume for each dataset and both reduction orders; fresh frozen-fit optimizer state versus inherited joint-stage optimizer state handled distinctly; case/query/RNG histories and scope restore exactly.
- Historical C0/M/H, C/Q and D0 loading remains strict and unchanged. Round5 guards continue rejecting Round6 identities. No unrestricted `strict=False` loading or catch-all fallback.
- A small target-poison test: inference scene preparation/readout must not access native fields, residual errors, per-role loss labels, case embeddings or validation-derived weights.

Run focused tests first, then the affected existing native/core/runtime suites. Report exact commands, overlapping test groups and skipped cases honestly. Do not turn a static import or synthetic smoke into a native-training pass.

## 12. Coding deliverables and proposed interfaces

Use opt-in, versioned names; the following are proposed additions, not claims that these files or CLI options already exist.

1. `HONF_Proj/src/honf_forward_core/inherited_attention/hyperedge_readout.py`: tensor-only matched R-P/R-E primitives, readout parameter module, prepared readout projection contract and bounded full-bank diagnostic. Reuse existing indexing/cover validation.
2. A narrowly scoped `readout_model.py` or equivalent extension of the maintained moment core: the receiver hook, configuration/ownership stamps, trace/work output and strict reconstruction. Do not duplicate `_decode_features`, heat application, local reader or native extraction wholesale.
3. A small Round6 runtime protocol/attachment module and `HONF_Proj/tools/hyperedge_readout_train.py` facade over the existing TrainingEngine/provider factories. Declare `phase=frozen_probe|joint_confirmation`, `reduction=pool_first|edge_first`, `access=local_cover|full_bank_diagnostic`, parent SHA/age, fit-age mapping and exact trainable scope.
4. Focused tests, recipe examples, `docs/guides/Hyperedge_Readout_Diagnosis.md`, and `docs/reports/HONF_Hyperedge_Readout_Diagnosis_Round6_Report.md` with compact machine-readable result/decision files.

A suggested module boundary is `read_prepared_context(organization, query, query_states, ...)`. The base implementation must retain the old context arithmetic when no Round6 reader is present. Pass query_states already produced by `_decode_features`. New readout projections should be attached after `MomentHyperedgeCoreV1.prepare` finishes and registered in factor-version/lifetime checks. Association interventions must explicitly clear/rebuild dependent projections; do not rely on accidental retention or loss of dynamic dataclass attributes.

Return the actual executed context, optional readout trace and shape-derived work separately from trace-only typed means. The old module/environment mixtures no longer sum to the full new nonlinear correction and must not be presented as its exact effective source weights. Likewise, the original four P heads and a width-64 reader bottleneck are different axes.

The CLI must support CPU-only recipe validation/dry run, distinct output roots, explicit stop ages, clean-stop/latest resume and standalone inference loading without constructing an optimizer or dataset. Document only commands actually exercised. Keep resource launch wrappers and one-off analyses local/ignored, rather than publishing another ad hoc trainer or a general approval framework.

## 13. Cost and memory measurement contract

Compare on the existing input-fixed low/high-M TRAIN fixtures, with identical native requested outputs and receiver unions. Use physical GPU1 or another currently authorized uncontended lane consistently, record the actual GPU/software/precision state, and exclude model/catalogue loading from application timings while reporting setup separately.

| Scope | What must be included |
| --- | --- |
| Complete native call | Prescribed host inputs, H2D, complete scene preparation, query/local planning, readout, native extraction/derivatives/masks, synchronization and CPU return |
| Prepared-scene, new-query call | Retained valid scene representation; fresh native receiver request/plan, readout, extraction and transfer |
| Fixed-query, new-heat call | Thermal only; correct prepared query factors, fresh controls, same source-linear contraction and native outputs |
| Forward/extraction memory | Reproduce the Round5 B8/Q1024/4096/8192 native scope, including native derivative work but no extra loss-backward |
| Loss-backward memory | A separate fixed-B8 matrix using the identical target-free/native backward scope in both arms; do not compare directly to the forward-only percentages |
| Actual optimizer memory | All native macros in one real selected-data epoch, final partial macro and scheduled Thermal response included; separate from disposable gradient stress |

Use two warmups and at least seven alternating profiler-off paired trials, retaining raw samples, median and p90. A second paired block is allowed when the declared uncertainty interval straddles a decision band; retain both blocks rather than the faster one. Label intervals as repeat/measurement uncertainty, not seed or population confidence.

Match the trainable/frozen parameter scope, requested input gradients and autograd mode within each cost comparison. Freezing preparation can itself reduce backward memory; never attribute a frozen-probe-versus-joint difference to the reader architecture. The frozen parent reference must use the corresponding frozen scope, and joint D0 comparisons must use the agreed full native gradient scope.

The primary frozen study compares R-P/R-E costs and reports the common parent. Final joint qualification requires a newly measured paired D0 comparator, not an old machine-time number. Do not insist on a deployment speed pass before completing the bounded representation diagnosis, but never turn a memory-only win into qualification.

Report the new preparation projection cost, hidden-element count, geometry work, cover/local counts, raw byte allocations and counters-on/off overhead. If an optional implementation fails equivalence, show its timing as an unqualified executor experiment and retain the correct reference for all scientific fits.

## 14. Resource envelope and stopping rules

This plan sets ceilings, not a target to consume them. Codex must prospectively forecast each admitted stage and reserve its matched counterpart and final evidence before launching it. Use the existing authorized workstation/environment; do not perform account, package-stack, GPU-permission or system-service changes.

| Resource | Hard ceiling / rule |
| --- | --- |
| Aggregate GPU-associated process time | 20 hours, including failures, pilots, profiling and all optional stages |
| Elapsed round time | 14 hours from first substantive execution; final two hours reserved for report/cleanup |
| Scientific optimizer updates | At most 54,000 across all phases, including the optional extensions/diagnostics/confirmations; discarded/replayed actual work also consumes headroom |
| Disposable engineering optimizer attempts | At most 400, separately counted; smokes/pilots never become scientific initialization |
| Optional same-function tuning | At most two process-hours inside the above envelope |
| New persistent /data payload | 12 GiB, including checkpoints, any allowed frozen-feature cache and numerical exports |
| New workspace payload | 1 GiB |
| Temporary /data and /tmp | 1 GiB and 256 MiB respectively; no undeclared compiler/browser/dataset caches |
| Minimum free filesystem headroom | 20 GiB on each affected filesystem, measured before allocation |
| New duplicate raw datasets/shared caches | Zero; reuse exact verified shared catalogues |
| Optional frozen-feature RAM cache | 8 GiB maximum, explicit owner and invalidation; no detached-cache shortcut for requested input derivatives |
| Selected figures | 128 MiB total; PDF masters with only necessary small Markdown raster companions |
| New reference solves / inverse designs | Zero; inherited solver usage stays 326/326 and target locks remain intact |

At the maximal permitted ages, the primary frozen pair costs 18,000 updates (two arms ×1000×[4+5]); two full-bank diagnostics cost at most 9,000; joint confirmations cost 27,000 (two arms ×1500×[4+5]). Their sum is 54,000 before any replay overhead. Therefore, after discarded/replayed work, not all optional maxima may still fit. Do not declare replay work free. The ordinary four 500-epoch frozen fits require only 9,000 new updates; a clear negative result can close the round there.

Do not start a stage that cannot finish its paired horizon and evidence inside the remaining budget. If interrupted externally, restore checkpoint-contained weights/optimizer/RNG/history, preserve uncheckpointed tails as lossless failure evidence, and count known/unknown discarded work separately. Do not attach a later CSV history to older weights. Recheck mount UUID and source immutability after reboot.

Protect all prior reports, dataset/source files, shared caches, scientific checkpoints, selected/latest aliases and unrelated jobs. Clean only round-owned disposable payloads under an exact allowlist with sizes, ownership, link/inode information, relevant hashes, active-reader checks and dependency checks. Preserve compact provenance and failure receipts. Actual free-space deltas are not identical to the sum of deleted file bytes on a shared host.

## 15. Decision tree and what each possible result means

| Result | Supported interpretation | Next action within this plan |
| --- | --- | --- |
| R-E clearly beats R-P with frozen P/eta/Z | Query-conditioned processing before the local sum uses distinctions the matched pooled adapter does not use as effectively | Admit the bounded joint pair if native laws/resources pass |
| R-P and R-E both improve similarly over fixed H | Added receiver/head fitting helps; the pre-pooling ordering has not earned a unique advantage | Retain the result; no automatic larger network. Run R-G only under its diagnostic trigger |
| R-E fits TRAIN substantially better but DEV remains weak | Capacity/readout restriction is partly relieved; transfer remains unresolved | Joint confirmation can test coadaptation, but no fresh full-TRAIN pass follows from frozen fit alone |
| R-G improves strongly while local R-E does not | Broader access to the same bank is useful under this full-bank correction; all information is not obviously absent from the bank | Preserve global-cost evidence; no automatic new selector, source bypass or global deployment |
| Neither local nor global readouts fit adequately | This bounded decoder family fails on the frozen representation; bottleneck/optimization/feature limitations remain unresolved | Stop, list measured limitations and a narrowly justified future payload test; do not claim an impossibility theorem |
| Joint R-E beats R-P but P/eta/Z diverge | A practical coadapted readout package improves; fixed-representation causal evidence still comes from Stage B | Apply standing native/cost bands and disclose changes in the learned organization |
| Native fields improve but latency fails | Useful accuracy/capacity direction, unqualified execution tradeoff | Report it; no qualification by memory alone |
| A baseline or law check fails | Software/evidence problem, not an architecture verdict | Repair only the specific correctness issue within budget or close with blocked scientific scope |

A negative frozen probe does not rule out successful fresh joint learning, but it also does not authorize an unbounded fresh-start search. A positive joint result from this one exposed selection does not establish independent-test, seed-stable or physically causal organizing value.

## 16. Report, figures and closeout

Begin the Round6 report with the actual decision: what was held fixed, what changed, whether local pre-pooling processing helped, whether global access was needed, and whether joint confirmation ran. Separate predictor, organizer/readout, execution and inverse-interface evidence. Distinguish measured results, mathematical controls, hypotheses and blocked stages.

Retain four compact figure families from saved numerical evidence: (1) matched fit curves and native role/component/error-bias summaries, with frozen and joint ages separated; (2) original-coordinate Thermal field residuals and Wind Ux/wake cuts, with other components and background visible; (3) actual P→Z→receiver reads, matched pooled versus edge-first hidden messages and pooling-nullspace/intervention controls; (4) complete/prepared costs and separately scoped memory. Do not regenerate every historical figure or retain many redundant exports. Show a failed readout result plainly rather than enlarge the graph artwork.

Use the existing fixed scenes (Thermal 0291/0687, Wind rows69/426) for DEV illustration and the original TRAIN4 for diagnosis. All scientific interpretation comes from their correct supports/units and the full declared boards, not the most favorable panel. Visually inspect every selected figure. Embed only retained companions, verify every link after cleanup, and keep each prose paragraph/caption on one source line.

The final report and machine-readable decision should contain: exact parents/SHAs and runtime source, frozen/trainable parameter inventories, phase and sampler-age definitions, literal and selected checkpoints, per-case/layout metrics and all guards, actual optimizer/case/query counts, fixed-bank equality receipts, law/gradient limits, raw timing scopes, optional-stage decisions, protected-data checks, cleanup receipt and remaining capacity. Report no new inverse result; source-linear software validity is not inverse readiness.

Commit durable source/tests/configs/guides/report text only. Enable the repository pre-push artifact hook, audit the **entire outgoing range**, including artifacts added and later deleted in unpublished commits, inspect/fetch remote state, push the current non-default working branch, and verify local/remote tips match. Generated figures/data, source snapshots, one-time scripts and cleanup helpers remain local/ignored. Never rewrite protected published history to hide a failed experiment.

### Definition of done

The round is complete when the native integration and relevant tests pass or have explicit blockers; both healthy primary pairs complete their declared full fitting horizons; optional stages run only under their stated gates; the exact frozen-representation claim is verified; fields/readout/cost evidence is saved and inspected; all misses are reported; disposable artifacts are cleaned; and durable changes are committed, pushed and read back. A meaningful negative diagnosis is complete work. A startup message, nonzero intervention, attractive graph or memory percentage alone is not.

## 17. Outside methodological context, not evidence about HONF

Chen, Liu and Wang, *Learning Continuous Image Representation with Local Implicit Image Function*, CVPR 2021, motivates decoding from local latent features together with continuous coordinates. It supports the usefulness of considering coordinate-conditioned latent readout; it does not establish that HONF's frozen hyperedge bank contains sufficient wake information or that the proposed reduction order will improve these datasets. Official paper page: https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.html .

Li et al., *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*, NeurIPS 2023, uses graph-neural-operator transformations into and out of a latent grid. It provides related methodological context for treating latent processing and geometry-aware output reconstruction as distinct design choices. It does not validate a hypergraph interpretation, source causality, an E64 limitation or our latency expectations. Official proceedings page: https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html .

The paired formulas, literal freeze experiment, full-bank diagnostic and resource gates above are this plan's proposed tests, not claims imported from those papers.
