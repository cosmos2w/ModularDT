# HONF Round7 — native-token hypergraph retrieval, without a pooled decoder bank

**Status:** proposed implementation and experimental plan, not completed evidence. **Reviewed repository:** `cosmos2w/ModularDT`, branch `agent/honf-core-next`, remote tip `b48ecfec46b1c601c0111e360f5b9bd6e29ab2df` on 17 October 2026. **Primary source:** the complete uploaded `HONF_Hyperedge_Readout_Diagnosis_Round6_Report.md`, read together with the retained Round2–Round5 reports and the current source files listed below. The review did not execute the user's checkpoints or inspect locally ignored figure masters. Codex must reproduce the relevant native boundaries in the actual ModularDT environment.

This plan supersedes the *future-development direction* of Round5/6. It does not rewrite their results, weaken their historical numerical contracts, delete their implementations, or authorize modification of unrelated work. New development returns to the uncompressed A2-D computation. H, C, moment banks, R-P, R-E and R-G are historical controls, not parents for the new candidate.

## 0. The architectural decision

**Hyperedges are a retrieval structure over original token records, not replacement value vectors.** A hyperedge stores native member IDs, typed/head-specific association scores, permissions and execution metadata. It does not store a pooled mean, moment descriptor, prototype, learned edge state or low-rank substitute as the mandatory decoder value channel.

The new path is:

\[
\text{native M/E tokens}
\xrightarrow{\text{unchanged A2-D contextualization}}
\begin{cases}
\text{individual contextualized token records},\\
\text{attention-derived overlapping member lists}
\end{cases}
\xrightarrow{\text{query selects lists; union and deduplicate IDs}}
\text{nonlinear query–original-token interactions}
\xrightarrow{\text{receiver-dependent reduction}}
\text{native fields}.
\]

The prohibition is on **mandatory pre-query compression of decoder values**, not on every sum anywhere in a transformer. Existing contextualizer attention, residual updates, scene reinjection and the final query-conditioned attention reduction remain. Here “original token” means an individually identified input token with its own contextualized state, original coordinates, features, mask, measure and ID. It does not mean discarding contextualization and reading only unencoded features.

Four invariants govern the new implementation:

1. Every native M/E record remains available in the prepared token store; retrieval changes which records a query evaluates, not which records exist.
2. An expensive nonlinear value is constructed only **after** its native token ID has been selected. Selection may use inexpensive scalar scores, but never requires the expensive value it proposes to avoid.
3. With all permissions open and no new environment consumer enabled, the candidate recovers the maintained A2-D function and its native derivative contracts. This is a same-weight implementation condition, not a claim that an adapted full-access mode will always be accurate.
4. Membership, information-attention weighting and physical response coefficients are separate quantities with explicit normalizations. Thermal support and coefficients are independent of the applied heating vector.

There is no requirement to learn a small universal K, activate six groups, occupy every group, or produce a visually attractive graph. There is also no promise of a speedup: M is small in these datasets, so retrieving fewer nonlinear rows can still lose to regular dense arithmetic. Measure that early.

## 1. What Round6 established—and what it did not test

Round6 fulfilled the earlier frozen-bank experiment. Thermal R-P/R-E completed 500 new fit epochs; Wind completed 1,000. R-E/R-P TRAIN priority ratios were 0.999945 and 1.001697. Full-bank R-G/R-E ratios were 1.023791 and 1.072472. These are negative results for changing the readout of that frozen H representation, not tests of native-token retrieval.

The distinction is explicit in `inherited_attention/hyperedge_readout.py`: its reader does not read physical source values and projects the completed H state. Reading each Z before SiLU does not undo the earlier formation of Z. The full-bank diagnostic also consumed Z, not the original donor inventory.

Some extra fitting helped Thermal. Full-native DEV fluid-T means were H 0.789884, R-P 0.712796 and R-E 0.707876. Wind near-vector means were H 0.629116, R-P 0.665549 and R-E 0.660857 m/s. Neither a useful Thermal fit nor a negative Wind fit establishes a universal theorem about all pooled models. The reason to stop this line is that it does not implement the user's intended organizing mechanism, and its measured tradeoff is uncompetitive.

R-G additionally raised complete high-M latency by 6.276 times relative to R-E in Thermal and 3.026 times in Wind. Do not use its low frozen-backward memory as an argument for more bank experiments. Keep its numerical evidence intact.

Round3 supplies a different warning that remains directly relevant: post-fit retention of 95% of decoder attention mass raised Thermal fluid-T error 45.16% and material-T error 106.48%. At 99%, 90.74%/96.44% of Thermal/Wind fine rows remained. Thus neither attention mass nor a new interpretation of it is a physical-error certificate. The new retrieval policy must participate in training and must be evaluated against physical targets, not just attention coverage.

### A reporting issue to resolve, not silently edit

Round6 repeatedly calls the Wind reference “analytic-wake/shared-grid.” The current `Case_WindFarm/Dataset/dataset_manifest.json` describes uploaded CFD data and ragged OpenFOAM fields; earlier Wind reports use that description. Before new plots, join the actual provider file path, dataset key, manifest hash and native target IDs. State which source supplied the new measurements. This review identifies inconsistent descriptions, not proof that Round6 loaded the wrong data. Preserve the old report and issue a concise erratum only after establishing the actual binding. Do not substitute a new dataset or claim this explains the model error.

## 2. Code findings and implementation boundaries

All paths below are relative to `HONF_Proj/`, except `AGENTS.md`. Function names are preferable to mutable line numbers when implementing against a later checkout.

| Reviewed source | Finding | Round7 action |
|---|---|---|
| `src/honf_forward_core/content_hypergraph/model_v2.py` | A2-D retains individual final module/environment states, original scene data and per-module projected keys/values. Its D branch has an empty edge bank. | Use this family as the parent; add narrow preparation/read hooks with unchanged defaults. Do not inherit from the moment/readout family. |
| `src/honf_forward_core/content_hypergraph/staged_attention_v2.py` | The two cycles execute MM → EM → EE → ME; route Q/K and geometry are already computed. The current production score-snapshot API accepts only EM/EE. | Add a separate native-route capture contract for the last-cycle MM/ME, with optional EM provenance. Do not transpose an EM probability matrix and call it ME. Keep SDPA for ordinary context updates. |
| `src/honf_forward_core/content_hypergraph/readers_v2.py`, `DirectSourceReaderV2` | Nonlinear query–module values are evaluated before attention reduction. The scalar Thermal coefficient deliberately does not multiply the information-attention probability. | Separate cheap scoring from selected-pair value evaluation. Reuse the original parameters and nonlinear formula. Preserve the signed scalar head. |
| Same file, `LocalDetailReaderV2` | A separate compact local source path already exists, including native support and Thermal coefficients. | Preserve it initially. Protect its source IDs in candidate retrieval where appropriate, but do not count its existing rows as new sparse savings. |
| `src/honf_forward_core/content_hypergraph/source_read_plans.py` | Useful shape/validity, smooth-gate and packet utilities coexist with the failed Round3 utility scorer/completion model. | Reuse tested low-level utilities only. Do not instantiate `SourceReadControllerV1`, a completion factor or a separate importance network. |
| `src/honf_forward_core/content_hypergraph/source_read_execution.py` | Post-reduction linear projection and bias-mass helpers already exist. | Reuse only under new same-function numerical qualification. A failed historical FP32 factoring mode must not silently become the default. |
| `src/honf_forward_core/inherited_attention/hyperedge_readout.py` | R-P/R-E/R-G require prepared H Z and a ReceiverCover. | No scientific dependency from the new decoder. Preserve old loaders/tests. |
| `src/honf_runtime/content_hypergraph.py` and existing TrainingEngine/provider code | Strict loading, native transforms, sampling, lineage and optimizer handling are available. | Add one opt-in native-retrieval family and branch declaration; do not create another trainer or relax old family schemas. |
| `Case_ThermalChannel/src/channelthermal/content_hypergraph.py` | Dataset-owned geometry, source heating, native extraction and response laws. | Reuse normalization, field extraction, source identity and allowed input separation. No new thermal surrogate or frozen D-sep. |
| `Case_WindFarm/src/windfarm/content_hypergraph.py` | Dataset-owned aligned coordinates, E64 geometry, fitted profile and native residual transform. | Preserve the frame and transform; do not add another rotation or absolute normalizer mean. |
| Root `AGENTS.md` | New outputs belong under verified `/data2/wanglz/ModularDT/`; shared caches and historical artifacts are protected. | Follow current policy, including outgoing-history audit and cleanup. |

### Important scope distinction: A2-D is not already a query–M-plus-query–E reader

The inspected D branch projects and directly reads **module states**. Environment records influence them through contextualization; they are preserved in the prepared organization but do not already have a separate direct query–environment value branch. Therefore “reuse the existing M/E decoder” would be an inaccurate implementation instruction.

The primary experiment below changes only retrieval of A2-D's existing module value path. The shared hypergraph and retrieval API carry M and E records from the start, but primary E records are **available/retrieved metadata and upstream context**, not a claimed new direct field-value consumer. A small, explicitly controlled direct-E extension is specified in Section 10. It must not be added invisibly or credited with sparsifying a branch absent from D0.

This staging is deliberate: first test the proposed retrieval mechanism without changing the proven value function; then test the direct environment consumer with its own uncompressed control. The final report must say whether the M-only stage or the actual M+E stage was executed. A metadata-only E demonstration does not complete the latter.

## 3. The three shared runtime objects

### 3.1 NativeTokenStore

Retain separate M/E arrays containing native IDs, coordinates, characteristic lengths, masks, measures, raw feature records, final contextualized states and individually projected decoder keys/values. A cached projection is tokenwise: its leading record axis still corresponds one-to-one to native IDs. No projection may contract the record axis.

Bind the store to model parameters/buffers, scene tensors, native frame/normalizer, token IDs and numerical policy. Preparation runs once per scene forward, not once per query tile. During training or geometry differentiation its tensors remain connected to the current graph. Long-lived reuse across optimizer steps is rejected. Geometry-only cached catalogues and model-dependent token stores have different lifetimes.

### 3.2 NativeRetrievalHypergraph

A structural hyperedge is seeded by an actual material/module token s. One valid module produces one seed family; head and route views remain separate metadata. A practical identity is `(scene, seed_native_id, preparation_cycle, route, preparation_head)`.

For each seed retain module and environment member IDs, inherited probabilities/densities, final membership permissions, validity/self rules and typed CSR or bitset views. **There is no `edge_value`, `edge_mean`, `moment`, `prototype`, `Z` or learned edge embedding consumed by the decoder.** A descriptive centroid is unnecessary; should a plot compute one, it has no inference use.

The number of seed families is M, not a chosen latent H. This is input-adaptive cardinality, not a claim to have learned an independent optimal number of groups. Membership sizes and query-selected families vary with the scene and query. Preserve overlap; do not turn them into disjoint connected components, merge source identities, or compute transitive closure.

### 3.3 NativeQueryRetrievalPlan

For each receiver retain selected seed IDs, unique member IDs by type, per-decoder-head permissions, unique executed-pair IDs, padding/bucket layout and the reason for any dense-masked dispatch. Retain an explanation path from query → seed family → original record. The plan stores no pooled value.

Separate four notions in its counters: retrieved IDs, IDs consumed by a field-value reader, eligible value rows and physically executed value rows. Do not count an E member list as a field read, four head memberships as four separate native sources, or repeated appearances through overlapping groups as repeated physical values.

## 4. Scene-owned neighborhoods from the actual preparation attention

Use final-cycle **MM** rows for module members and final-cycle **ME** rows for environmental members. The convention is destination first: MM(s,i) means module s reads module donor i; ME(s,j) means module s reads environment donor j. EM(j,s) is a different directed relation. It can be exported for provenance and a future controlled extension, but it is not silently transposed, averaged into ME, or composed into an unrestricted two-hop closure in this first version.

Let t identify the donor type and r a preparation attention head. Over a row's actual eligible donors, define:

\[
\nu^t_{si}=\frac{\mu^t_i}{\sum_{j\in\mathcal V^t_s}\mu^t_j},\qquad
A^t_{sri}=\frac{\nu^t_{si}\exp L^t_{sri}}{\sum_{j\in\mathcal V^t_s}\nu^t_{sj}\exp L^t_{srj}}.
\]

Here L includes the actual bounded head temperature and geometry bias, but excludes log donor measure. Preserve MM self-exclusion in the contextualizer. Add the seed itself to its **retrieval** module list with permission one; this does not change the contextualizer's self rule.

A raw probability cutoff is sampling-density dependent. Use a dimensionless density-relative score:

\[
\rho^t_{sri}=\frac{A^t_{sri}}{\nu^t_{si}},\qquad
r^t_{sri}=\frac{\rho^t_{sri}}{\max_{j\in\mathcal V^t_s}\rho^t_{srj}}
=\exp\left(L^t_{sri}-\max_{j\in\mathcal V^t_s}L^t_{srj}\right).
\]

This ratio has a defined purpose: membership screening. It is neither an attention mass nor a physical influence. The final expression avoids unnecessary probability division and is stable for small positive measures. It also makes exact donor-measure splitting transparent: identical donor records with split measure have the same relative-density score, provided their states and eligible set satisfy the algebraic fixture. Real mesh refinement may change states and geometry and is not covered by that identity.

For threshold tau in (0, 0.5), use the existing endpoint-flat quintic permission:

\[
S_5(x)=\begin{cases}0,&x\le0,\\6x^5-15x^4+10x^3,&0<x<1,\\1,&x\ge1,\end{cases}
\qquad b^t_{sri}=S_5\!\left(\frac{r^t_{sri}-\tau_t}{\tau_t}\right).
\]

At tau=0, dispatch explicitly to all-open eligible permissions; do not divide by zero. Positive support begins before value evaluation. Empty donor rows are exactly empty. Equal maximum scores are all retained; never break scientific ties using token IDs. Native-ID sorting is allowed only for reproducible storage order.

Initially use separate scalar tau_MM and tau_ME values shared across heads, with row/head normalization performed separately. Different attention heads are not interchangeable or averaged into one “physical importance” map. The number of heads is a channel allocation, not the number of physical hyperedges. Do not add entropy/diversity losses or trainable thresholds in this round.

### Score capture without defeating the contextualizer

Keep the existing SDPA update unchanged. Add opt-in last-cycle capture of the already computed normalized Q, K, geometry bias and allowed mask, including MM diagonal exclusion. If explicit logits require one additional QK multiplication, charge it. Do not enable all-layer `return_attention=True` merely to obtain two routing tensors, and do not replace fused updates with a Python attention implementation.

Capture the actual stage state, not a reconstructed attention map over final states labelled as historical attention. Decoder payloads are final contextualized native token values; a preparation-stage association over them is a reused learned routing relation, not a literal decomposition of their complete residual ancestry.

Do not change the old EM/EE-only `RouteScoreSnapshot` contract in a way that invalidates historical loaders. Add a general native-route snapshot type or a separately versioned extension. Required fields include exact destination/donor IDs, route, head count, stage identity, allowed mask, raw scores, measure convention and owner/version binding.

## 5. Query selection, union and deduplication

Compute A2-D's existing cheap query–module content/geometry scores **once**, before nonlinear values. Those same scores remain available for final module attention. No additional query importance MLP, cluster-key network, value-error predictor or learned prototype is introduced.

Let ell(q,h,s) be the existing decoder's raw content/geometry score, excluding log measure. Define a query-seed relative-density score and permission using the same construction:

\[
r^Q_{qhs}=\exp\bigl(\ell_{qhs}-\max_j\ell_{qhj}\bigr),
\qquad a_{qhs}=S_5\!\left(\frac{r^Q_{qhs}-\tau_Q}{\tau_Q}\right).
\]

Each source with positive a selects its native-seeded neighborhoods. Selection is not a hard nearest-environment assignment, a regular-grid ReceiverCover, or interpolation through fixed anchors. Queries anywhere in the legal domain use their own geometry and token scores. The environment catalogue remains native input data, not a fixed latent decoder bank.

A transparent, duplicate-invariant permission union is:

\[
p^t_{qhi}=\max_{s,r}\{a_{qhs}b^t_{sri}\},\qquad
\bar p^M_{qhi}=\max\{p^M_{qhi},\chi^{\mathrm{protect}}_{qi}\}.
\]

For modules include b(s,s)=1. Decoder head h and preparation head r are separate axes: numerical head labels across independently parameterized routes do not imply matching physical semantics. Initially take the conservative union over preparation heads, while retaining the original views in exports. Do not average them and hide lost support. Measure whether this union becomes too dense.

The max-product union is a proposed routing law, not an inherited attention probability. It keeps permissions in [0,1], does not multiply contributions because a token belongs to several groups, and is invariant to an exactly duplicated group record. It is continuous but piecewise differentiable; competing-max ties are not claimed differentiable. A tilewise implementation must avoid a full `[B,Q,decoder_head,seed,preparation_head,donor]` allocation. For ordinary values, preparation-head permissions can be reduced by an exact max once per scene and indexed by selected seeds, while the unreduced views remain available for provenance. This reduction combines routing metadata only, never token values. Count the remaining query–seed–member union work; it can be quadratic in M in the worst case and is not free merely because the output plan is sparse.

Existing geometry-local pairs retain their original value function. Define $\chi^{\mathrm{protect}}_{qi}$ from the existing nonnegative compact geometry window before its physical-measure multiplier; do not add a hard radius switch or a new radius. Every original pair with positive local window therefore has positive direct retrieval permission. Query protection guarantees that all native locally required module IDs are included in the direct candidate union, where this is the selected policy; it does not delete the separate local branch or combine its coefficients with direct coefficients prematurely. Count direct/local rows separately. This is a conservative first version, not a new wake cone or radius enlargement.

For each query define actual executed module support as the union over decoder heads:

\[
\mathcal S_M(q)=\{i:\max_h\bar p^M_{qhi}>0\}.
\]

Construct unique `(batch, query, type, native_id)` rows. A value is computed once and shared across heads that consume it. No source is evaluated multiple times merely because several seed groups retrieve it. Query plans retain E support analogously, but the primary M-only reader must mark E records as not directly value-consumed.

### Why support may remain large

Every selected seed includes itself, neighborhoods overlap, head unions can be broad, and contextual attention need not agree with decoder importance. These are genuine risks, not bugs to hide with a top-K cap. Report the final union, not merely per-head sparsity. An all-full result is allowed and is an honest negative result for this particular retrieval rule. Do not insert learned merging or tune thresholds after seeing DEV errors to manufacture a small K.

## 6. Native-token value computation and physical output laws

### 6.1 Preserve the A2-D nonlinear reader

For each retrieved native module i use the original per-token cached projection and the original query-relative value features:

\[
v_{qi}=W_o\,\operatorname{SiLU}\!\left(W_q q_s+W_m z_i+W_\gamma\gamma(q-x_i)\right)+b_o.
\]

The actual original parameterization and bias placement are authoritative. Do not replace this with a cached v_i followed by a dot product, and do not evaluate it at a group centroid. Query and source linear projections can be prepared on their own axes; selected relative features and the nonlinearity remain query–token dependent.

For information reduction use the existing decoder logits and physical measure once:

\[
w_{qhi}=\frac{\bar p^M_{qhi}\mu_i\exp\ell_{qhi}}{\sum_{j\in\mathcal S_M(q)}\bar p^M_{qhj}\mu_j\exp\ell_{qhj}},
\qquad c_q=\operatorname{concat}_h\sum_iw_{qhi}v^{(h)}_{qi}.
\]

All-empty or masked heads produce zero without NaNs. The seed-max rule should normally ensure a valid nonempty module head when a scene has modules; test it rather than hiding an accidental empty set by an unreported full fallback. Retain original query features, local context and field head.

During the permission transition, positive permissions can be fractional. They are routing gates, not estimated physics. The deployed reference is the **same compact continuous-permission function** evaluated on its exact positive support. Do not binarize positive permissions to one at evaluation and call that the trained function. A binary-support experiment would be separately named and is not a default in this plan.

### 6.2 Thermal: information weights are not heating coefficients

Let k_raw(q,i) be A2-D's signed scalar output on the native query–module value, before applying current h. The information attention w must not be inserted into the Thermal source sum. If fractional routing permissions are used, define their separate scalar union from the **protected module permissions**, $g_{qi}=\max_h\bar p^M_{qhi}$, and make the actual coefficient explicit:

\[
K_i^{\mathrm{route}}(X,q)=g_{qi}(X)\,k_{qi}^{\mathrm{raw}}(X)+K_i^{\mathrm{local}}(X,q),
\qquad T(X,q,h)=\sum_iK_i^{\mathrm{route}}(X,q)h_i.
\]

The routing gate determines a permitted/attenuated value path; it is not normalized over heaters and does not become a nonnegative physical response coefficient. k_raw and K remain signed. With all gates one this recovers the original direct-plus-local law. No softmax of heaters, pooled heat mode, forcing-dependent cutoff, current-h importance score or mass multiplier is added to this physical sum.

All contextualization, query features, graph construction, query permissions and coefficients are heat-independent. Geometry changes rebuild the relevant state; heat-only changes can reuse fixed-geometry prepared source-resolved factors. Zero heating gives exact zero temperature response, and flow remains independent of heating. Surface, material, native curl and q-proxy extraction remain dataset-owned and unchanged.

A post-fit attention cutoff cannot certify this sum. Offline, with all survivor coefficients and query context held fixed, omitted source terms obey the elementary bound

\[
\left\|\sum_{i\notin\mathcal S}K_i h_i\right\|\le\sum_{i\notin\mathcal S}\bar h_i\|K_i\|.
\]

This is only a diagnostic using explicitly available coefficients and a declared control envelope; computing all omitted expensive coefficients at inference would defeat retrieval. Do not turn it into an implicit all-source guard. If a direct-E extension changes query conditioning, even survivor K values can change; evaluate that separately rather than applying a missing-column identity across different contexts.

### 6.3 Wind and the reusable core

Wind retains its nonlinear velocity head and its exact native profile/residual transform. It does not inherit Thermal's affine forcing law. Both datasets use the same token-store, memberships, permission union and selected reader implementation; their feature builders, units, source masks, measures, local geometry and physical heads stay in their adapters.

There is no special turbine-ID embedding, output lookup, second wind rotation, target-velocity context, inference wake formula or error-conditioned support. The existing wind-aligned frame already makes downstream/crosswind/height information available in the ordinary query–source geometry. Do not enlarge the 1.5D local branch in the primary experiment: full nonlocal native source values are again available through retrieval, which is the change being tested.

## 7. Execution design: prove a plausible payoff before long fitting

### 7.1 Refactor the reader into cheap score and expensive value phases

Add a new opt-in reader interface with methods equivalent to `prepare_tokens`, `score_queries`, `build_retrieval_plan`, `evaluate_selected_values` and `reduce_native_outputs`. Keep the historical `forward` route intact. Separate score-only geometry from Fourier/value geometry: the current helper computes both together, so merely skipping the final MLP after constructing all wide pair features is not sufficient.

A cheap `[B,Q,M,H]` score or Boolean mask is permitted and fully counted. A full nonlocal `[B,Q,M,128]` value allocation is not permitted in selected mode. Existing scene `[B,M,E]` routing metadata is also permitted. Do not claim sublinear source search—the first version still scores all M seeds. Its hypothesis is cheaper **nonlinear value execution**, not an asymptotic nearest-neighbor search result.

### 7.2 Two same-function backends

Maintain one dense-masked reference and one selected execution backend. They must receive the exact same native IDs and permissions and produce the same function to the declared numerical contracts. Dense-masked evaluation is a correctness and cost control, never a claimed sparse saving.

Start with compact source-count buckets and tensorized gathering; use a native-ID bitset or small Boolean union where appropriate. M<=30 makes a compact module bitset possible in current fixtures, but the public API must not hardcode a 30-source maximum or silently overflow. Environmental lists require a scalable representation. A vectorized Boolean/CSR reference is sufficient before any compiled specialization.

Avoid per-query Python loops, per-query `unique`, host `tolist`/`item`, CUDA synchronizations in the reader, full-data mask hashing and graph rebuilds per query chunk. Group queries for **execution** by exact support signature only when the grouping cost pays. Such buckets are not learned hyperedges or spatial latent anchors. Padding within a bucket is charged. A long head union can eliminate the advantage of nominally sparse headwise masks.

Dense dispatch for high retained fractions is allowed only when it computes the **same selected-permission function**. It must not silently replace it with a different all-open function. Report how often the executor uses selected versus dense-masked mode and the rows actually executed in each.

### 7.3 Bounded engineering comparison

Before scientific fits, qualify the value backend on actual low/high-M TRAIN fixtures, with injected full/approximately 75%/50%/25% native support patterns and representative overlaps. These are synthetic execution controls, not accuracy evidence. Test B1/B8 and Q1024/8192; add the Q4096 memory point after the basic path works. Reuse the same weights, masks, query features, precision and native output request. Compare against both the original A2-D execution and dense-masked execution of the selected function.

The necessary payoff condition is

\[
t_{\mathrm{capture}}+t_{\mathrm{membership}}+t_{\mathrm{seed/union}}+t_{\mathrm{packing}}+t_{\mathrm{selected\ values}}+t_{\mathrm{reduction}}
< t_{\mathrm{original\ decoder}}.
\]

Measured complete calls, not this symbolic inequality or removed rows, determine success. Preserve once-per-scene preparation, prepared-scene/new-query and fixed-query/new-heat scopes separately. Do not add overlapping timing medians.

If even an ideal 50% native support cannot outperform the original decoder on high-M Q8192, allow **one bounded execution repair**, such as bucket layout or a tensor-only compiled gather/value/reduction kernel. Do not introduce a new model. If it still fails, retain the correct implementation and stop before a long sparsity campaign; explain that the execution opportunity is missing at these shapes. A metadata-only graph is not a substitute for a value executor.

No framework upgrade is required. Test available compiler/backend capabilities in the pinned environment. FlexAttention is not a drop-in replacement for an arbitrary nonlinear query–token value MLP. M<=30 can also occupy less than one typical attention block. Keep SDPA for contextualization and optimize the actual selected value computation rather than assuming a sparse-attention library removes it.

## 8. Training the retrieval function without an auxiliary importance network

Hard selection alone has no ordinary derivative through excluded IDs. Do not claim otherwise, and do not assume that a straight-through label automatically trains omitted values. The proposed compact permission has exact zeros and differentiable transition regions. Gradients flow through selected values, their live permissions, existing query scores and the captured preparation-score tensors. The native contextualizer also remains trainable through its original attention updates.

Use a gradual all-open-to-compact transition, not a mature dense checkpoint pruned only at evaluation. No frozen H bank, head-only correction fit, source reconstruction, pseudoinverse, detached-physics shadow or new scorer is part of the primary candidate.

### Primary paired trajectories

Per dataset, reconstruct its exact retained **D0 segmented e500** source, including recipe, normalization, named AdamW state and deterministic query namespace. Create two independent continuations:

| Arm | Value function | Retrieval | Trainable scope |
|---|---|---|---|
| D-control | Original A2-D direct module plus local reader | Full; no graph overhead in ordinary calls | All original parameters |
| G-native-M | Identical native module/local value functions | Native-seeded typed hypergraph; selected module rows | All original parameters, with task gradients through live permissions; no new learned organizer parameters |

Both continue absolute e501→e2500: **500 inherited plus 2,000 new epochs**, not 2,500 new epochs. No R5/R6 weights enter. This is a matched continuation experiment conditional on a competent dense parent, not a from-scratch architecture comparison. Reuse the exact D0 numerical policy and native objective, not an approximately similar Round5/6 recipe.

If an exact e500 source is genuinely unavailable, first document the missing binding. A common fresh dense e1–500 prefix can be trained once per dataset within the total budget and then branched. Do not replace it with an H parent, a mature full-data checkpoint, a different normalization, or an invented old e500 state.

### Threshold declaration and limited TRAIN-only calibration

The default candidate uses tau_MM=tau_ME=0.20 and tau_Q=0.20 at terminal routing strength. Before scientific training, evaluate at most three **input-only** query thresholds {0.10,0.20,0.35}, with the member thresholds fixed. Use the fixed four TRAIN fixtures and the qualified executor to inspect actual unions and cost. No field error, DEV outcome, current heating amplitude or inverse objective chooses a threshold.

Select a threshold from these three only when it yields a useful high-M executed-module reduction with manageable complete overhead; prefer the least restrictive threshold among comparable cost points. Seal one value per dataset. If all three leave nearly full unions, report that rather than increasing the threshold until a preferred histogram appears. If none creates a computationally plausible point, the full long fit is not automatically required. This is an execution-opportunity decision, not physical qualification of a cutoff.

The three values are proposed engineering settings, not literature-derived guarantees. Keep them, the selected policy and its reason in the recipe. Do not add another search over per-head temperatures, source-count caps or member thresholds in this round.

### Absolute-age routing curriculum

- At the e500 branch boundary, all permissions are one and the native function is exactly D0 through the original full reader.
- Over e501–1000, increase the declared thresholds continuously from zero to the sealed terminal values. This is gradual training under the new value-access law; all positive permissions count as support. New score capture/union work is charged.
- From e1001–2000, keep terminal thresholds. Every eighth absolute epoch uses one declared all-open **ordinary native epoch**, replacing that epoch's sparse mode rather than adding a second prediction. This limited exploration keeps individual values exposed; it is not a guarantee of useful gradient on every excluded membership. It is also not a teacher or cheap-completion path.
- From e2001–2500, use the terminal compact-permission function on every ordinary epoch. The literal endpoint is evaluated at that exact function.

The D-control uses the same case/query/response streams at each absolute epoch. Its lack of access-mode variation is the intended treatment difference. Record every full-exploration epoch and its real work. Do not call the all-open function at adapted weights a validated accuracy fallback unless its native fields are measured separately.

Maintain the original D0 optimizer and LR schedule, clipping, decay groups, role losses, query counts, source measures and scheduled Thermal response addendum. Do not import Round6's fresh-Adam head-only settings. If the retained D0 protocol differs from a table in an older report, the bound recipe and actual engine calls take precedence, and the discrepancy is disclosed.

### Reviews and sufficient maturation

Review at absolute e600, e1000, e1500 and e2500. Early reviews check numerical health, gradient connectivity, actual support/packing and resource forecasts. A healthy candidate is allowed to reach the declared e2500 endpoint even when early field differences are mixed. Do not substitute 100 new epochs for the mature comparison. Stop earlier only for a real correctness failure, persistent large training divergence, loss of the value-preservation contract, an absent execution opportunity that was overlooked, resource limits or a user stop.

No threshold, objective or representation revision is selected from new DEV outcomes mid-trajectory. A code bug repair is versioned, its affected work disclosed and exact replay/resume verified. A scientific change requires a new child identity and is not hidden inside a resume.

## 9. Primary decision criteria: preserve the useful predictor, then judge retrieval

These are proposed Round7 development decision bands, not inherited physical certification or confidence limits. Seal them before new DEV results. They distinguish an implemented mechanism, a promising research direction and a deployable speedup.

### 9.1 Field-preservation board

Use the complete fixed DEV22/DEV24 native boards on common coordinates, targets, masks and native output definitions. Primary guards versus the equally continued D-control are:

- Thermal fluid/surface/material T and v/p: each mean RMSE ratio <=1.05; each p90 <=1.10. Report u, native curl, q proxy and recomputed module peaks separately; no protected secondary mean above1.10 or p90 above1.15 for a strong preservation verdict.
- Wind near/downstream Ux and vector errors: each mean ratio <=1.05 and p90 <=1.10. Keep all other components and roles; no protected component/role mean above1.10 or p90 above1.15 for strong preservation.

A narrower **research-direction** result may be reported when priority means stay within1.10 and tails within1.15, alongside a clear reduction in actual value work and no severe scene/stratum failure. It is not promoted by averaging away a failed channel. No improvement over D-control is mandatory: the objective is similar accuracy with useful organizing/execution potential, not another large accuracy-gain gate.

Also compare each candidate's actual terminal result with the retained D0 e2500 on the same current support, documenting any exact protocol differences. This prevents a deteriorated control from manufacturing a favorable ratio. Warm-start history remains explicit.

### 9.2 Organizer-use board

Required implementation evidence: actual variable membership lists; query-selected families; unique consumed native IDs; exact correspondence to evaluated value rows; removal/restoration; and absence of a mandatory pooled value path. Demonstrate changes within the same module-count stratum, not merely M-dependent group counts.

A useful retrieval direction additionally needs meaningful partial use on a nontrivial fraction of high-M queries. Target at least25% fewer **executed nonlocal module value rows** at the high-M fixtures, counting padding and dense dispatch. Report all-full fraction, per-head and final-union degree distributions, overlaps and support stability. This is a target, not permission to force a cardinality.

Use equal-final-degree **direct decoder-score selection**, nearest/directional native-source selection, and a preparation-membership rewire on the fixed detail panel. The decoder-score control reuses the same cheap query scores without the prepared neighborhoods; it is the important test of whether inherited group retrieval helps beyond ordinary direct selection. Also show a seed-only neighborhood control with the selected seeds held fixed: a donor admitted through another seed despite its own zero query-seed permission is a concrete retrieval expansion. A seed-only control normally changes the final degree, so it is not an equal-work claim. Where fractional gates are compared, preserve the multiset of admitted gate values as well as the final degree where feasible, and explicitly disclose constraints that cannot be matched. Controls share the same trained weights and preserve valid/protected sources. Match measures as well as degree only where that is actually achievable; otherwise label degree-only matching. Recompute attention normalization under the control rather than copying an incompatible denominator. Large output changes establish use; superior physical error at matched work is the stronger usefulness result.

### 9.3 Cost board

The denominator is **D-control/A2-D**, not R-G or a deliberately slow dense-masked reader. Show both native-default and a fairly tuned same-function dense A2-D execution where qualified. Use the same device, precision, B, Q and complete output scope.

A measured end-to-end sparse acceleration requires complete high-M Q8192 latency at most0.90 of qualified A2-D, with no hidden transfer, preparation, union or packing exclusions. A research candidate with meaningful partial work, preserved fields and complete latency within1.15 of D-control can be retained as a direction, but must be labelled **not yet faster**. If it is slower by more than that, do not recommend replacing D0 merely because rows or memory shrink.

Fixed-B8 Q1024/4096/8192 forward/backward memory must include the real trainable contextualizer, not a frozen representation. Actual complete optimizer epochs are separate measurements. Retain baseline allocated, peak allocated and peak reserved values, activation-checkpointing settings and every OOM. Do not hide memory through a smaller candidate minibatch or use optimizer-free peaks as training memory.

### 9.4 Native interface and response

Require all-open recovery, valid masks/IDs, no target leakage, numerical native read consistency, heat independence and the applicable affine/adjoint laws. Reuse the existing saved Thermal response families at the final checkpoint; do not launch new solver calls or inverse search. Keep finite physical-response error distinct from algebraic precision. Current geometry-support transitions have only piecewise derivative meaning; no blanket smooth inverse-readiness claim follows.

The current solver ledger stays326/326. Wind TEST and unexposed Thermal targets stay locked. Full source training is not automatically authorized by this plan. Deliver a recommendation and inert recipe only if the new evidence supports one; any full-TRAIN launch is the user's next decision.

## 10. Direct retrieval of environmental values: a separate, explicit extension

The primary exact-A2-D experiment is intentionally M-value retrieval. It must not be advertised as a trained direct M+E decoder. The following small extension is designed now so the reusable core can support the complete proposal without silently adding an untested branch.

### 10.1 Actual E consumer—not a graph decoration

An optional `NativeEnvironmentReader` consumes the **retrieved original environment records individually**. It forms query–environment nonlinear messages using original e_j, q and signed q−x_j geometry, then performs a receiver-dependent attention reduction. No environment group mean is formed before this interaction.

Use one narrow32-channel nonlinear message and apply its32→128 projection **after** the receiver reduction. Cached environment projections are per native record. The final projection begins at zero. Its output can add to the existing direct module reader's query-value preactivation, while the original seed-selection scores remain based on the original query state. This permits environmental information to affect the signed Thermal K through the existing nonlinear module message without giving E tokens independent heating amplitudes. Keep the original local branch unchanged.

This is an architectural extension, not free reuse: A2-D has no trained query–E decoder. It needs explicit parameters, shape/measure conventions, complete costs and a matched full-access control. Reapplying a preparation route's projections to query states is also a new read relation; it must not be labelled an exact captured attention map. A compact independently initialized typed score projection may be needed for this **value reader**, but not a separate importance/utility network for selecting its support. Retrieval support still comes from the already declared native hypergraph.

### 10.2 Conditional matched pilot

Only after primary M retrieval preserves native fields and has a plausible execution result, admit a separate pair per qualifying dataset:

- `ME-full`: the same new narrow E consumer, but direct access to all original E records; module support and original G-native-M weights are identical to its sibling.
- `ME-retrieved`: the same consumer and parameter initialization, consuming the E union supplied by the existing native hypergraph.

Both start from the same chosen primary checkpoint with identical new zero-output projections and fresh optimizer state under a prospectively sealed pilot recipe. Neither receives a head fitted in the other arm. At pilot step0 both recover their common M-only parent. Give each500 complete segmented epochs; the existing native objectives, target streams and current parameter allowlist match. This is a500-new-epoch extension at a different architecture age, not a same-age result against the original D0e2500. Do not expand to another2500-epoch portfolio in this round.

Admission also requires a cost-only native E consumer preflight; if its matched full-access control is unaffordable within the round envelope, leave the pilot unlaunched and state why. The pilot compares native E access at equal new capacity; it does not prove that both are better than the original A2-D or that module selection is responsible for an E gain.

Report E identities actually consumed, nonlinear QE rows, typed physical measures, original frame and measured output effects. If no E pilot is admitted, the final result is explicitly **A2-D module-reader retrieval with a reusable typed M/E interface**, not completion of direct M+E field decoding. Do not substitute metadata export for this learning test.

## 11. Concrete code work packages

Implement a small new package, for example `src/honf_forward_core/native_retrieval/`, with clear boundaries. Names below are proposed new files, not assertions that they already exist.

### W0 — baseline and protection, then stop the old scientific direction

Read the current Round6 report and branch diff; verify b48ecfe and any later user commits without resetting the checkout. Freeze references to D0/e500/e2500, formal A2-D, native classics and R4–R6 results. Check current processes and permitted device UUIDs. Do not alter active jobs. Reproduce A2-D low/high-M native inference and its native loading/normalization contract. Preserve every old experiment, but instantiate none of the C/H/readout models in the new scientific candidate.

Resolve the Wind reference-source description and the current D0 sampling/optimizer identity. Do not refit normalization. Confirm the original engine's final partial batch and response-callback cadence. This is a short baseline task, not another prolonged audit campaign.

### W1 — `contracts.py` and `route_capture.py`

Add `NativeTokenStore`, `NativeRetrievalHypergraph`, `NativeQueryRetrievalPlan` and a general typed route snapshot. Expose native IDs and permitted views; bind source tensor versions, scene, weights, frame, normalizer, query IDs and thresholds. Keep score-stage metadata distinct from final-value-stage metadata. Add final-MM/ME opt-in capture to the contextualizer with unchanged output defaults and a bounded optional trace.

Avoid a universal reflective traversal or hashing of all numerical tensors at every query. Reuse current ownership/version mechanisms. Validate public boundaries once and keep internal tiled tensor loops free of redundant host checks. Invalidation is required after a weight update, geometry/mask/measure change, token-order change, configuration/threshold change or backward consumption of a live prepared graph.

### W2 — `membership.py` and `retrieval.py`

Implement the declared relative-density thresholds, seed self-membership, head/type metadata, query-seed score reuse, max-product permission union, protection and native-ID deduplication. Keep the scalar source/reference measure distinctions explicit. Provide a small clear tensor reference before optimizing storage. Include M=0/1, empty E, padding, unequal measures and tied scores. Do not introduce ReceiverCover, virtual anchors, top-K latent slots or automatic group merging.

### W3 — `reader.py` and `execution.py`

Factor the A2-D direct reader into reusable scoring and native selected nonlinear values without changing its historical constructor/state keys. Implement dense-masked and selected execution of the exact same permission law. Gather original token states and geometry before values; reduce heads only after nonlinear interaction. Retain signed Thermal source coefficients and an opt-in prepared sparse coefficient object for repeated h/JVP/VJP. Never materialize an ordinary dense source coefficient bank merely to select support.

Use bounded tiles or source-count buckets. Preserve gradient accumulation through repeated sources and deterministic native ordering. Optional compiled code stays tensor-only and has a faithful eager reference. Record actual executed/padded rows and cache bytes. Do not count masks as value-work savings.

### W4 — `core.py`, adapters, runtime and recipes

Create an opt-in A2-D-derived core with no content bank and no frozen partner. Hook into the existing TrainingEngine, strict checkpoint registry and data-free load path. Common parameters must be name-preserving; no arbitrary state-dict positional copy. Child metadata declares the D0 parent and the new permission curriculum. Add exact resume tests including current threshold phase, optimizer moments and sampling epoch.

Use the existing Thermal/Wind adapters with small optional capability declarations, not copied adapters. Environmental direct-read capability is off in primary recipes and explicit in optional Section10 recipes. A round-specific CLI may create/run the sealed recipes but delegates all optimization to TrainingEngine. Do not build a new training orchestrator or loosen old guards to accept new checkpoints.

### W5 — learning, evidence and closeout

Run the primary matched trajectories after correctness and cost feasibility, with one dataset per authorized lane as practical. Create an understandable summary board first, then a compact evidence index. Run optional E pilots only under Section10 and remaining resource reservation. Review actual code, output routes and saved native fields—not just test counts. Commit durable code/tests/docs on the non-default branch, audit the complete outgoing range, push and verify matching tips. Generated arrays, figures, one-time scripts and checkpoints stay local and ignored.

## 12. Focused tests that prevent returning to the wrong architecture

These tests are mandatory and should be compact; do not inflate the study with hundreds of redundant model probes.

**Value-path tests.** In all-open mode, recover A2-D native outputs, applicable input/parameter gradients, loss and a real optimizer step. Trace the nonlinear value input and verify that each row is a single original token, not a group state. Use a constructed equal-mean two-token fixture to show that selected query–token nonlinear processing retains distinctions a pre-query mean would erase. A source ID is never a learned numeric feature.

**Membership tests.** Preserve allowed masks, empty rows, exact self-exclusion in MM, retrieval seed inclusion and typed head axes. Test relative-density membership under an exact donor-measure split, reporting only that algebraic scope. Verify source/seed permutation equivariance with pulled-back IDs. Exactly duplicating a group record or reordering groups cannot change the resulting permissions/outputs. Overlap must not duplicate a value contribution.

**Execution tests.** Packed versus dense-masked outputs, input/parameter gradients and one optimizer boundary use the same selected permissions. Test full, partial, one-source, all-local, empty query and heterogeneous-M cases. Test arbitrary query chunking/permutation and repeated query IDs without changing support or normalization. A batch companion must not change a query's graph policy. Verify actual hook-counted nonlinear rows against the plan, including bucket padding and dense dispatch.

**Learning-path tests.** During transition, demonstrate a nonzero task gradient through an existing preparation-score parameter via a live permission on a constructed/native nondegenerate fixture. Excluded zero-permission rows have no direct value gradient through that row; record this rather than fabricating one. Full exploration epochs are observable in the history. Verify no unreported extra full prediction or teacher callback occurs.

**Thermal tests.** Heat-null flow, T(0)=0, finite increments, signed source-ID columns, JVP/VJP transpose consistency and exact source identity are preserved. Use precise factor accumulation and separately preserve native FP32 limits. Membership cannot change when only h changes. Derived q/curl and endpoint maxima are recomputed through the original extraction, not copied from the baseline. Distinguish fixed-support geometry derivatives from rebuilt-support finite changes and report gate/tie crossings.

**Ownership and target-isolation tests.** Reject changed weights, geometry, measures, masks, normalization, token order and stale query/graph lifetimes. Poison target fields in unused training containers; predictions and support must not depend on them. Cache acceleration is not permission to detach needed geometry gradients.

Use existing domain-owned native tolerances where they apply. New isolated FP64 algebra tests can be strict, but do not apply a source-equation tolerance to a different physical derivative. Do not widen an old gate after seeing a miss; retain failed arithmetic results separately from physical error and identify which executor is qualified.

## 13. Fair comparison and small, intelligible evidence

Three boards answer different questions:

- **Matched mechanism:** D-control versus G-native-M at identical additional age, common parent, native protocol, queries, objective and support. This is the primary retrieval experiment.
- **Practical reference distance:** retained formal A2-D and native Thermal1404/1804 and Wind2102/2103 on the same exposed panel, each with its own checkpoint-native adapters and transforms. Label actual selected/literal ages: for example the retained Wind classics often use e2440/e2475 from2500 histories, not assumed5000 checkpoints. They are not matched new training controls.
- **Conditional direct-E pilot:** ME-full versus ME-retrieved with identical new consumer capacity, parent and500 new epochs, plus their common M-only parent. Do not merge this board into the primary age comparison.

Reuse trustworthy saved reference arrays when coordinate/target/mask/checkpoint bindings match; do not rerun every historical family. No conclusion should use percentage improvements against unlike normalization scores or unlike query supports. Show equal-case Thermal summaries and the recorded Wind row/layout reducers, with correlated directions grouped for uncertainty descriptions. Preserve M strata and all roles; do not substitute vector RMSE divided by sqrt3.

### Four figure families maximum for the main report

1. **What the field looks like.** Two fixed input-selected scenes per dataset: reference, D-control, G-native-M and signed residuals with native RMSE/relativeL2. Include Wind Ux wakes and at least one transverse cut; Thermal T/v and native material peaks. Use the same field/error scales across compared models.
2. **What this query actually read.** Native modules and environmental records in physical coordinates, selected seed families, their overlapping member lists, final unique consumed IDs and executed query–token links. Show retrieved-but-unconsumed E metadata differently in the primary experiment. No fictitious latent nodes or group-value heatmaps.
3. **Accuracy versus actual cost.** Actual native error against measured complete latency and fixed-B memory, plus eligible/selected/padded/executed value counts. Separate same-function backend comparison from different trained architectures.
4. **Does the graph operation matter?** Count-matched native geometry controls, a typed membership rewire, duplicate-path invariance, remove/restore effects and a bounded saved Thermal heat-response replay. Use software and physical-reference labels accurately.

A short front-page table must answer: Did accuracy stay near A2-D? Were original tokens really read? How many expensive rows actually disappeared? Was the complete call faster? Did direct E values participate? What remains untested? Keep detailed numerical inventories in one appendix/index instead of letting the principal answer become a compliance log.

## 14. Resource and storage envelope

This is a proposed bounded Goal, not work already authorized on a newly discovered device. Retain the latest permitted physical GPU1/GPU2 lanes after checking current user instructions and UUIDs; GPU0 and unrelated jobs remain untouched. A newer explicit device instruction overrides this proposal. Do not evict another process or assume a logical cuda index identifies a particular physical GPU.

Proposed ceiling: **12 aggregate GPU-associated hours, 10 elapsed hours including at least1 hour reserved for closeout**, with a maximum of **45,000 new scientific optimizer updates** for the two primary continuations per dataset plus all admitted500-epoch E pilots under the recorded4/5-update protocols. A fresh common e1–500 prefix, if needed because no exact source exists, must fit within these ceilings by reducing optional work, not by silently exceeding them. Up to200 disposable engineering updates are separate and charged. No full-TRAIN, inverse or solver campaign is included.

Before admitting a long stage, measure complete native epochs and reserve enough time for the entire paired horizon and final native board. A ceiling is not a requirement to spend unused time. Conversely, a healthy admitted2500 endpoint must not be replaced by a one-epoch smoke merely because startup passed. If resource limits make completion impossible, stop at a common saved age, record the missing endpoint and do not claim maturation.

Use the existing ModularDT environment and numerical policy. Record current available memory and leave at least20% or8 GiB device headroom, whichever is larger. Increase microbatch packing only through a qualified same-effective-batch check; do not change query counts or scientific optimizer batch size to improve a timing chart.

Store new generated outputs below verified `/data2/wanglz/ModularDT/diagnostics/native_token_retrieval_round7_<timestamp>/`, symlinked from the ignored workspace. Proposed new persistent budget20 GiB, temporary budget10 GiB and minimum filesystem headroom20 GiB. Reuse verified datasets/catalogues at their current paths. Native input counts are small enough that graph metadata can usually be prepared in memory; do not duplicate full datasets to save a few kernel calls.

Before cleanup, freeze protected roots and build the exact recursive allowlist required by AGENTS.md. Remove only round-owned disposable caches, superseded renderings and temporary outputs after checking active readers, mappings and links. Retain scientific checkpoints/milestones, complete histories, numerical evidence, native references and compact manifests. Report actual files/bytes removed and remaining capacity. Do not create more cleanup/audit data than the numerical study needs.

## 15. Deliverables and stop rules

Deliver maintained code for native seeded memberships, native-token retrieval and the selected nonlinear reader; strict recipes/load/resume integration; focused tests; all completed primary histories; a clear report and one machine-readable evidence index. Preserve the original reports. The graph interface must be useful without a particular visualization renderer.

The report must distinguish:

- **Implemented and numerically exercised** from **trained and physically evaluated**.
- M-only A2-D retrieval from actual direct M+E decoding.
- Per-head membership sparsity from final-union and executed-row sparsity.
- Source-identity preservation from physical source causality.
- TRAIN adaptation, exposed DEV transfer and independent TEST evidence.
- Same-function faster execution from architecture changes or a different numerical backend.

No automatic rescue by adding pooled edge content, moment banks, fixed latent slots, cheap all-source completion, another utility network, a new loss portfolio, k-means prototypes, geometry jets, or new physical solves. A negative native-token retrieval result must identify whether the observed limitation was membership coverage, response importance, head unions, selector learning or executor cost. It must not trigger another bank-compression experiment under a new name.

The primary success condition is modest but substantive: **an intact A2-D-like predictor uses attention-derived, layout-dependent native member lists to reduce actual nonlinear decoding work at near-baseline accuracy, with the execution cost reported honestly.** A faster complete call is stronger evidence; a small graph alone is not.

## 16. Research context and source ledger

The following external works are methodological context, not evidence that this particular HONF model will work.

**BiFormer — Zhu et al., CVPR2023, “Vision Transformer with Bi-Level Routing Attention.”** Its useful precedent is coarse routing followed by fine token-to-token attention over retrieved regions. It does not justify replacing native tokens with pooled values. Its region-level routing summaries and image regions are not copied into this design: HONF uses native module seeds and already learned preparation relations. Paper DOI `10.1109/CVPR52729.2023.00995`; author code `https://github.com/rayleizhu/BiFormer`.

**Routing Transformer — Roy et al., TACL2021, “Efficient Content-Based Sparse Attention with Routing Transformers.”** It supports content-dependent retrieval and sparse attention as a viable general family. Its online k-means centroids and sequence-length regime are not the proposed HONF mechanism. Paper `https://aclanthology.org/2021.tacl-1.4/`; official implementation `https://github.com/google-research/google-research/tree/master/routing_transformer`.

**PyTorch2.6 SDPA documentation.** SDPA returns an attended output, not a free attention-probability buffer, and backend arithmetic/eligibility differs. Preserve the current fast contextualization and explicitly charge score capture. Documentation `https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html`.

**PyTorch FlexAttention documentation.** Block masks can express structured sparse attention, but do not implement an arbitrary nonlinear pair-value function or ensure a win at a thirty-source axis. Treat the actual installed2.6 API and a measured native prototype as authoritative rather than copying current-main signatures. Current reference `https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html`.

**Repository source ledger:** the branch tip was resolved live through GitHub to `b48ecfec46b1c601c0111e360f5b9bd6e29ab2df`. Reviewed: root `AGENTS.md`; `content_hypergraph/model_v2.py` preparation, direct branch and policy hooks; `staged_attention_v2.py` complete ordered routes and score export; `readers_v2.py` direct/local readers; `source_read_execution.py`; `source_read_plans.py`; `inherited_attention/hyperedge_readout.py` prepared-Z contract and receiver construction; runtime loading dispatch; and Wind's dataset manifest. Original local paths/checkpoints/figures are not presumed accessible in a new clone.

**Report ledger:** Round6 complete uploaded text, its continuation/field/cost/precision tables, and retained Round2/3/4/5 evidence. Any source description inconsistency remains explicit until the actual provider binding is verified. No new accuracy, latency, sparsity, derivative or inverse result is claimed by this plan.
