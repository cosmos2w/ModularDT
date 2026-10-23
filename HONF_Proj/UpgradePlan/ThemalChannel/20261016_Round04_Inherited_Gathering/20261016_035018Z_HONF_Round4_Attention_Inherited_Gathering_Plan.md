# HONF Round4: attention-inherited gathering and reusable field decoding

**Status:** proposed implementation and experiment plan, not a completed model result. **Mode:** Codex Goal mode with bounded development and an explicitly conditional full-TRAIN follow-up. **Reviewed repository:** `cosmos2w/ModularDT`, non-default branch `agent/honf-core-next`, commit `36d1ca0459497f4c1e9fa5c99ba0845f80d9e671`. **Primary evidence:** the complete `HONF_A2D_Sparse_Readout_Round3_Report.md`, its Round2 predecessor, and the current contextualizer, readers, source-read controller, scene preparation and native adapters. References and source locations are indexed at the end.

## 0. Decision and scope

Build a new, opt-in **attention-inherited gathering (IAG)** decoder on the successful A2-D contextualizer. The organizing object must participate in the actual information aggregation, not merely select corrections to a second all-source approximation. Preserve the completed A2-D predictors and all historical evidence. Do not instantiate Round3's utility scorer, rank-16 completion, calibration manager or periodic teacher refresh in the new path.

The principal change is **replacement of repeated nonlocal query–source value construction by scene-level gathering and reusable values**. Existing environment tokens provide spatially registered receiving anchors. Their last-cycle module/environment attention logits define overlapping typed source associations. A continuous, input-only receiver cover transfers those associations to arbitrary queries. The same associations weight the values and, after a separately trained sparse normalization, define exact support.

Start by learning this representation with soft gathering. Only then test a sparse-normalized child. This is not post-training attention-mass pruning of A2-D. A2-D's old signed thermal coefficient was independent of its information-attention weights; the new thermal response must be trained to use the new gathering law itself.

The preferred first realization, **IAG-C**, caches weighted token aggregates at environment anchors and reassociates the query read exactly. It removes the nonlocal query–source MLP altogether. One bounded alternative, **IAG-Q**, is allowed only when an early, measured capacity diagnosis shows that cached aggregate interpolation discards necessary query-specific discrimination. IAG-Q reuses the same inherited associations and projected attention keys/values for ordinary query attention inside their support; it does not add a utility network or a cheap-plus-fine field branch. Do not implement a larger architecture portfolio.

### Deliverables

1. A shared, dataset-independent IAG implementation with production attention export, typed scene associations, reusable values, continuous receiver covers, nonlinear readout and optional source-indexed linear-control transport.
2. A matched soft-versus-trained-sparse experiment on both existing segmented datasets, preserving their native objectives, observation operators and unified TrainingEngine.
3. A working indexed-query executor and a measured selected-edge preparation executor, each compared against its mathematically identical dense reference; preparation, inference and training costs must be separated.
4. An intuitive report showing physical fields, actual inherited routes, reconstruction versus complete cost, and bounded response checks. At most one fresh full-TRAIN e1000 candidate per dataset may follow the decision rules below; no automatic longer run is authorized.

**Success is not “all heads look sparse.”** Success is an affordable, accurate representation whose scene-level associations demonstrably organize what a receiver reads. Exact-zero weights, reuse savings and sparse-kernel savings are three different findings.

## 1. What Round3 actually established

Round3 D0 remains the preferred dense development model. On the declared high-M TRAIN fixtures at Q8192, S-Complete reduced fine rows by 41.05% in Thermal and 59.86% in Wind, but complete cold calls rose from 46.269 to 76.666 ms and from 28.467 to 52.340 ms. Fixed-B8/Q8192 allocated memory increased from 2.966 to 3.790 GiB in Thermal and fell only from 5.946 to 5.852 GiB in Wind. These are negative execution results, not just a missed accuracy threshold. [R3, “Mature complete cost and fixed-B8 scaling”]

Native accuracy also regressed. Thermal fitted S-Complete increased fluid-T mean error by 8.18%, pressure by 21.43%, material temperature by 24.29% and surface temperature by 25.78%. Wind near/downstream vector errors increased by 4.44%/9.09%, with a larger background regression. Useful q-proxy and some saved heat-response gains remain real but do not override those endpoint failures. [R3, “Mature sparse fidelity”]

Two qualifications matter. First, S-Utility did not obtain a physical-learning result: its initial auxiliary fits encountered a CPU/CUDA accumulator failure after using their allotted steps. Do not present that implementation failure as evidence that every utility method is scientifically incapable. Second, the co-adapted S-Complete all-full function was not an accurate fallback: Thermal fluid-T error at those weights was 7.887 times D0. Re-enabling omitted computations can change a trained model substantially. [R3, “Organizer”]

The attention-mass control is more directly relevant to the proposed redesign. At 95% retained attention mass, Thermal kept 72.60% of physical fine pairs but increased fluid-T/material-T errors by 45.16%/106.48%. At 99%, it kept 90.74% of Thermal and 96.44% of Wind pairs. The old weights therefore did not identify a useful, accurate sparse operating point. They were not trained as the thermal transport law. [R3, “Organizer”]

The earlier post-reduction projection was mathematically valid in FP64 but failed some fixed FP32 native/GPU checks. Round3's dense scientific fits retained legacy arithmetic. Do not call that optional backend qualified, and do not make its repair the main activity of this round. Preserve old failure evidence. [R3, “Execution”]

The correction to our development strategy is substantive: remove redundant query–source value construction and the separate selection-learning problem, rather than add another mechanism around them. The Round2 full-TRAIN A2-D success remains protected; its source identity, staged contextualization and native output laws are the assets to retain.

## 2. Source audit and the usable boundary

All paths below are relative to `HONF_Proj/` unless noted otherwise. They were inspected at the reviewed commit; inspect subsequent changes before editing, without resetting user work.

| Source | Observed implementation | Round4 action |
|---|---|---|
| `src/honf_forward_core/content_hypergraph/staged_attention_v2.py` | Ordered MM → EM → EE → ME; SDPA; normalized Q/K, bounded temperatures, geometric score biases and typed donor measures. `return_attention=True` recomputes explicit weights after SDPA. | Reuse the contextualizer. Add an opt-in, minimal production export of only the last EM and EE score ingredients/logits. Do not enable every debug trace. |
| `src/honf_forward_core/content_hypergraph/model_v2.py` | Geometry descriptors built once per preparation; raw/scene reinjection; final M/E states and strict prepared-state ownership/version checks. | Reuse preparation primitives and identity rules in a new family. Avoid duplicating the large old model or allocating unused H6/direct/controller parameters. |
| `src/honf_forward_core/content_hypergraph/readers_v2.py` | The direct reader caches source projections, but still constructs `[B,Q,M,128]` nonlinear pair values; thermal K is deliberately not multiplied by information attention. | Leave historical behavior unchanged. Replace this nonlocal reader in IAG; do not merely insert a mask around it. Retain the compact local reader and native field head where dimensionally compatible. |
| `src/honf_forward_core/content_hypergraph/source_read_plans.py` | Separate utility MLP, threshold metadata and rank-16 completion. | Historical reference only; no instance, optimizer group, calibration or refresh manager in IAG. |
| `src/honf_forward_core/content_hypergraph/contracts.py` | Scene/query masks, measures, IDs, frames and prepared ownership. | Reuse the boundary concepts; add small versioned atlas/cover/response contracts, not another provenance subsystem. |
| `Case_ThermalChannel/src/channelthermal/content_hypergraph.py` | E192 on a 24×8 catalogue; native 128×64 grid unions, surface/outside/material extraction, derived curl and q proxy; local radius 3r. | Preserve native operators. Supply a target-free environment-grid cover and bind a fresh, jointly trained IAG core. No frozen D-sep/Stage-A. |
| `Case_WindFarm/src/windfarm/content_hypergraph.py` | E64 on a 4×4×4 catalogue, seven query features, native aligned frame, profile-plus-residual physical transform, local radius 1.5D. | Preserve transforms/geometry/roles. Supply the corresponding 3-D cover; no second rotation or normalizer-mean addition. |
| `src/honf_runtime/unified_training.py`, maintained content-family runtime/tools/providers | Existing joint engine, deterministic case/query streams, strict loading and separately guarded protocols. | Register a new opt-in workflow/recipe version using the same engine. Do not weaken old full-run guards or reinterpret old checkpoints. |
| Repository-root `AGENTS.md` | Retention, storage, outgoing-history upload audit, report formatting and cadence rules. | Follow throughout; keep prose paragraphs/captions on single source lines. |

### 2.1 Route direction and state age are not negotiable details

In this implementation, **EM means E receives M**: its rows are environment destinations and its columns are module donors. ME means M receives E. The last cycle evaluates EM with `environment_states` and `module_after_mm`; EE then uses `environment_after_em`; the final ME and token FFNs subsequently change states again.

Consequently, last EM/EE scores are associations learned at their declared stages, not an exact decomposition of final token ancestry. The new decoder deliberately reuses those associations to gather final contextualized payloads. Record both the score-origin stage and payload-state stage. Do not relabel old scores as “attention between final tokens,” multiply attention matrices together and call the product a physical influence operator, or quietly recompute a different final-state attention law.

This is a trainable reuse hypothesis: downstream field gradients will reach the reused score computation. It is not a claim that the already trained D0 attention is a ready-made physical graph.

### 2.2 Export narrowly without replacing fused contextualization

For the last cycle's EM and EE calls, expose the already computed normalized/scaled Q, normalized K, additive geometric score, validity and donor measures through an internal optional return object. Construct their raw score matrices once per scene from those live tensors. The original contextualizer output must still use its unchanged SDPA call. Return neither all cycles' weights nor intermediate messages by default.

SDPA does not expose its internal attention matrix through the current public return value. Therefore this is a small, explicitly charged score recomputation, not free access to a hidden fused buffer. Cache only what the new decoder consumes. With B1/H4/E192/M12, the two exported matrices contain 156,672 scalar entries; with E64/M30, 24,064. Batch size, autograd and temporary storage multiply these costs and must be measured.

Avoid another K/V projection or geometric-feature calculation merely to construct the export. A new decoder may project final token payloads once; that has a different purpose and state age. Never detach the atlas in training. Legacy `return_trace=False` behavior must remain unchanged when IAG is not requested.

## 3. Relevant research and what we will actually borrow

**BiFormer, CVPR 2023** filters candidate regions before token attention and uses gathered, GPU-friendly dense operations on the remaining regions. Its official `bra_nchw.py` projects Q/K/V once, pools detached regional Q/K and uses top-k region routing. Borrow the separation of reusable regional decisions from fine reads and its concern for block execution. Do not copy detached routing, query-batch-dependent pooling, fixed top-k or vision-only indexing into a geometry-differentiable field interface. [P1, C-BiFormer]

**Agent Attention, ECCV 2024** factors query reading through intermediate agents: agents gather token information and queries read agents. This motivates reassociation and reusable payloads. Its positional/local additions are also a warning against treating plain pooled features as sufficient. Here the receiving anchors already exist as physical-coordinate environment tokens; we do not introduce another six-slot global agent bank. [P2]

**Adaptively Sparse Transformers and entmax** show that a normalization itself can produce exact-zero, input-dependent support rather than rely on a second threshold-prediction network. We use this principle with a fixed 1.5 normalization after soft training, not a learned-alpha search. The donor-measure extension below is a mathematical specification for this project; it is not claimed to be supplied by the standard package. [P3, C-entmax]

**RIGNO, NeurIPS 2025** provides a relevant operator-learning precedent for separating regional representation from physical input/output discretizations. Borrow the interface principle, not its full graph processor or JAX stack. Its results are not evidence that our chosen E192/E64 cover has adequate capacity. [P4]

**Attention-flow work** cautions that residual connections and layer mixing make raw attention an incomplete explanation of model dependence. Our exported routes are tested computational associations, not physical causality. [P5]

These papers motivate components, not a guarantee of HONF accuracy or acceleration. No framework/package upgrade, external pretrained model, new dataset or new solver label is required.

## 4. General mathematical contract

### 4.1 Inputs and contextualization

Let the structural scene be

\[
\mathcal X=(\{(i,x_i,m_i,\mu_i)\}_{i=1}^{M},\{(a,e_a,u_a,\nu_a)\}_{a=1}^{E},c,\mathcal D),
\]

where identities, coordinates, features, positive measures and presence masks are explicit. The adapter owns units, the coordinate frame, domain geometry and the dependency declaration \(\mathcal D\). Targets are not reachable from this inference object. Configuration controls change \(\mathcal X\); separately applicable controls, such as Thermal heating, do not enter the heat-independent encoder.

Retain the two ordered typed cycles and raw/scene reinjection:

\[
(Z_M,Z_E,g)=\mathcal C_\theta(\mathcal X).
\]

Width 128, four heads and two cycles remain the initial capacity. Separate datasets train separate weights through the same core implementation. Do not add a third cycle, wider model, enlarged environment catalogue, extra loss or new normalization experiment in this round.

### 4.2 The inherited association atlas

Let \(s^M_{a i h}\) be the last EM content-plus-geometry score, **excluding** its additive log donor measure. Let \(s^E_{a b h}\) be the corresponding last EE score. Indices \(a\) are receiving environment anchors, \(i\) are physical modules, \(b\) are environment donors, and \(h\) is an attention head.

For each type, normalize only its own positive, present donor measure:

\[
\bar\mu_i=\frac{\mu_i}{\sum_{j\in M_{\rm valid}}\mu_j},\qquad
\bar\nu_b=\frac{\nu_b}{\sum_{j\in E_{\rm valid}}\nu_j}.
\]

The soft atlas is

\[
P^M_{a i h}=\frac{\bar\mu_i\exp(s^M_{a i h})}{\sum_j\bar\mu_j\exp(s^M_{a j h})},\qquad
P^E_{a b h}=\frac{\bar\nu_b\exp(s^E_{a b h})}{\sum_j\bar\nu_j\exp(s^E_{a j h})}.
\]

This reproduces the respective original route probabilities in the soft case, up to the declared arithmetic implementation, because its per-scene measure reference cancels in softmax. Invalid donors and destinations remain zero. An absent donor type returns a zero aggregate; an empty type must not create a fabricated probability distribution or NaNs.

Use EM and EE **separately** as the two typed associations. The first version does not add a new EM×EE rollout, reverse ME as if it were an inverse, or score a new independent organizer. Other interactions remain represented in contextual states.

### 4.3 A continuous query-to-environment cover

The adapter supplies a sparse receiver cover

\[
\Lambda_{qa}=\lambda_a(q)\ge0,\qquad \sum_{a\in\mathcal A(q)}\lambda_a(q)=1.
\]

For the current regular catalogues, \(|\mathcal A(q)|\le4\) in 2-D and \(\le8\) in 3-D. Determine neighbors from the declared environment coordinate axes and native IDs, not array order assumptions. Do not assign a discontinuous single nearest owner to each query.

The initial cover is a tensor-product, C1 smoothstep partition. On each interval between adjacent environment coordinates, let \(t\in[0,1]\), \(S(t)=3t^2-2t^3\), and use weights \(1-S(t),S(t)\) on the two endpoints. Products give the 2-D/3-D cover. At interval transitions the vanishing endpoint derivatives prevent a spurious first-derivative jump in the cover. Coordinate encoding and the local reader remain continuous query-dependent paths; this is not a piecewise-constant physical field.

This is a modeling choice, not an interpolation-exactness claim: smoothstep does not reproduce an arbitrary linear field between anchors and has flat interpolation slopes at anchor coordinates. Measure its effect on spatial derivatives and residual patterns. Keep its definition fixed during the primary experiment. Use constant edge extension inside the legal physical domain where queries extend beyond the environment-center box; reject queries outside the adapter's legal domain rather than silently extrapolate an unspecified geometry.

The current Thermal inventory keeps every E192 anchor present, including anchors geometrically inside a module; those are legal prescribed-context tokens, not stored fluid measurements. Preserve that inventory and keep its validity separate from the native fluid/solid observation masks. Do not delete an anchor using target fields or assume that its spatial territory is a material-free fluid cell.

The generic core consumes `ReceiverCover(anchor_ids, weights, validity, query_ids, frame_key)`. It does not know Thermal grid dimensions or Wind rotor units. An unstructured, input-declared cover fixture should exercise the generic interface without creating a third training dataset. Future providers may supply other continuous covers, periodic handling or different anchor counts under explicit contracts.

The effective associations are

\[
\beta^M_{q i h}=\sum_{a\in\mathcal A(q)}\lambda_a(q)P^M_{a i h},\qquad
\beta^E_{q b h}=\sum_{a\in\mathcal A(q)}\lambda_a(q)P^E_{a b h}.
\]

These are the same inherited probabilities used for weighting and support. No learned utility score is computed at the receiver. In the sparse model the query's direct source support is the union of positive associations of its nonzero cover anchors, across the actually used heads. That union can be much denser than an individual anchor row; measure it honestly.

## 5. IAG-C: reusable values are the primary decoder

### 5.1 Typed token values and exact reassociation

Project each final contextual token once to a reusable payload, with per-head width 32:

\[
v^M_{i h}=V_M(Z_{M,i})_h,\qquad v^E_{b h}=V_E(Z_{E,b})_h.
\]

Use ordinary tokenwise projections with normalization consistent with the retained architecture. Do not construct a nonlinear `[B,Q,M,128]` or `[B,Q,E,128]` value tensor. The module/environment values are typed; combining them must not mix their physical measures or treat them as independent physical heat inputs.

Prepare anchor aggregates once per scene:

\[
G^M_{a h}=\sum_iP^M_{a i h}v^M_{i h},\qquad
G^E_{a h}=\sum_bP^E_{a b h}v^E_{b h}.
\]

Then each query reads only its cover anchors:

\[
c^M_{q h}=\sum_a\lambda_a(q)G^M_{a h},\qquad
c^E_{q h}=\sum_a\lambda_a(q)G^E_{a h}.
\]

The key identity is

\[
\boxed{(\Lambda P)V=\Lambda(PV).}
\]

This is an exact identity **within the new model**, not an equivalence to the old A2-D nonlinear pair reader. It permits one scene-level donor aggregation followed by small indexed query gathers. Do not construct \(\beta[Q,N]\) merely to implement a computation that can be reassociated.

Concatenate the two typed contexts and combine them through one bias-aware linear projection into the retained 128-wide nonlocal context. The original pointwise field head then reads the query encoding, combined context and existing compact local context. Any output projection shared by all queries of an anchor may be placed in preparation when algebra permits; preserve every bias exactly once. Do not add a second full field predictor.

### 5.2 Native nonlinear fields

For a generic nonlinear output block,

\[
\widehat f(q)=D_\theta\big(z_q,c^M_q,c^E_q,c^{\rm local}_q\big).
\]

Thermal's \(u,v,p\) are produced by fresh jointly trained heads, followed by unchanged physical inverse transforms and native curl extraction. Wind produces its normalized nonlinear residual, followed by the exact maintained profile/scale conversion. Geometry and environmental context remain live in both datasets.

### 5.3 Thermal: the signed, source-resolved control transport

Nonnegative attention weights are not signed thermal response coefficients. Add one small **linear-control capability**, not a second temperature network. A heat-independent tokenwise head produces a signed carrier \(t_{i h r}\), with initially \(r=1,\ldots,8\) per head, giving 32 signed coordinates per source. A heat-independent query head produces \(\phi_{q h r}\) from query and gathered/local contexts. Its output is dimensionless; the carrier and adapter jointly implement the declared normalized-temperature-per-native-heating units.

Use the dimensionless association density

\[
D^M_{a i h}=P^M_{a i h}/\bar\mu_i
\]

for valid module donors, and zero otherwise. This is **not a probability**. It makes flat association scores reproduce an unnormalized signed source sum, rather than inadvertently averaging heating across the number of modules. This amplitude convention is an explicit new parameterization; it is not a physical conservation theorem. Keep native heating units unchanged. Extremely small donor measures need a declared numerical conditioning check, not a silent clamp that alters the law.

At fixed geometry prepare/apply

\[
L_{a h r}(h)=\sum_iD^M_{a i h}\,t_{i h r}\,h_i,
\]

\[
\boxed{\widehat T_{\rm norm}(q,h)
=\sum_{h,r}\phi_{q h r}\sum_{a\in\mathcal A(q)}\lambda_a(q)L_{a h r}(h)
+\widehat T^{\rm local}_{\rm norm}(q,h).}
\]

The existing compact local affine branch remains, with its native smooth support. This is a spatial near-field branch already in A2-D, not a newly trained cheap all-source baseline plus selected expensive corrections. The far response is entirely replaced by inherited, reusable transport. Do not keep the old dense thermal far reader behind this equation.

The explicit response coefficient, exported only for bounded diagnostics, is

\[
K^{\rm norm}_{qi}
=\sum_{a,h,r}\lambda_a(q)\phi_{qhr}D^M_{aih}t_{ihr}
+K^{\rm local,norm}_{qi}.
\]

Apply the maintained native temperature scale once, with no absolute temperature mean. Thus \(T(q,0)=0\), \(\partial T/\partial h_i=K_{qi}\), and \(\Delta T(q)=K_q\Delta h\) remain exact model laws at fixed structure. Heating must not affect the atlas, nonlinear flow, query cover, \(\phi\), or support. Environment values are heat-independent context, not separately actuated heaters.

For repeated heating, cache \(P,t,\Lambda,\phi\) and local factors, then apply the small linear contractions. Avoid retaining `[Q,M]` coefficients in ordinary prediction or adjoints. Implement the transpose by reversing these contractions. Do not put a nonlinear activation, LayerNorm, adaptive normalization, control-dependent mask or field-dependent denominator after the heat-dependent aggregation.

This is not the old fixed-H6 bottleneck. There are E spatial anchors with overlapping coverage, source-indexed carriers and a local path, and the model has a continuous query head. Nonetheless it is a factorized representation and may lose capacity. “Numerical rank is full” is not a substitute for good field reconstruction. Do not force positive kernel signs or uniform source occupancy from this specification.

### 5.4 Keep the local/native boundary honest

Retain Thermal 3r and Wind 1.5D local support, actual overlapping sources, masks and native extraction. Local detail remains computed on those supported pairs only. An environmental attention zero does not delete a compulsory local pair or change a material/flux stencil. Report nonlocal atlas support and local pairs separately.

The first version adds no distant query-dependent correction. If a cached decoder cannot fit distant or near-wake structure adequately, use the bounded IAG-Q alternative below or report that capacity limitation. Do not silently reconstruct Round3 under another name.

## 6. Trained sparse normalization without a separate selector

### 6.1 Why a measure-aware normalization

After the soft representation is established, use a fixed \(\alpha=1.5\) entropy normalization on the **same raw inherited logits** for decoder gathering. Keep contextualizer SDPA itself unchanged. Decoder task gradients still update the underlying route parameters and all shared states.

For one nonempty typed donor row with normalized positive measure \(m_i\), define

\[
p_\alpha(s;m)=\underset{p_i\ge0,\,\sum_i p_i=1}{\arg\max}
\left[\sum_i p_i s_i+
\frac{1-\sum_i m_i(p_i/m_i)^\alpha}{\alpha(\alpha-1)}\right].
\]

As \(\alpha\to1\), the entropy becomes the negative relative entropy to \(m\), giving \(p_i\propto m_i e^{s_i}\). For \(\alpha=1.5\), the unique normalized solution is

\[
\boxed{p_i=m_i\left[\frac{s_i-\tau}{2}\right]_+^2,
\qquad\sum_i m_i\left[\frac{s_i-\tau}{2}\right]_+^2=1.}
\]

The threshold \(\tau\) is determined by normalization, per anchor/head/type. It is not fitted by a separate network, chosen from DEV error, a fixed “top six,” or a claimed physical-error tolerance. Exact support and weights emerge from the same law. Invalid donors are excluded before normalization. The empty-row result is defined separately as zero.

This measure extension is project-specific mathematics built on entmax, not a claimed verbatim third-party algorithm. Blindly applying ordinary entmax to `score + log(measure)` does not generally preserve the same splitting/refinement behavior as measure-weighted softmax. Do not silently drop this issue when making the core reusable.

### 6.2 A bounded implementation, not an iterative GPU planner

Provide a simple FP64 reference root solver for tests. For production, use a vectorized sorted-prefix solution once per scene, with an explicitly stable implementation. With descending shifted scores and prefix sums

\[
A_k=\sum_{i\le k}m_i,\quad B_k=\sum_{i\le k}m_i s_i,\quad C_k=\sum_{i\le k}m_i s_i^2,
\]

candidate roots satisfy

\[
\tau_k=\frac{B_k-\sqrt{B_k^2-A_k(C_k-4)}}{A_k},
\qquad s_k>\tau_k\ge s_{k+1}.
\]

Shift each score row by a common maximum before solving; check discriminants and normalization against the FP64 reference. Do not hide invalid numerical roots behind arbitrary source-ID tie-breaking, recurrent retries or posthoc renormalization that changes the specified derivative. Ordinary tied logits are handled symmetrically. No Python loop over queries or iterative per-query root finder is allowed.

For fixed positive measures, the score Jacobian on its active support uses \(g_i=\sqrt{m_i p_i}\):

\[
J=\operatorname{diag}(g)-\frac{gg^\top}{\sum_i g_i}.
\]

Retain derivatives through actual positive measures where they depend on geometry; a hand-written backward that ignores measure derivatives is insufficient. Test the composed, normalized-measure function against finite differences. Exact support changes are C1 in this regular 1.5 law, but that does not make the entire native geometry pipeline globally smooth.

The uniform-measure law is intentionally a measure-relative entropy parameterization; at the same numerical logits its scale is not identical to a library's counting-measure entmax. Record that convention and retain the inherited learned score temperatures. Do not conduct a separate temperature/alpha/sparsity-coefficient sweep in the primary study.

### 6.3 Training schedule

The sparse child branches only from its own selected new soft architecture's literal e500 checkpoint. It inherits physical weights, optimizer moments, transform, RNG and sampling age. Over epochs 501–800 blend the two decoder distributions using a fixed deterministic schedule,

\[
P_e=(1-\xi_e)P_{\rm soft}+\xi_e P_{1.5},\qquad
\xi_e=\operatorname{clip}((e-500)/300,0,1).
\]

This blend normalizes already defined distributions and preserves live gradients; it has no teacher labels, utility regressor or separate optimizer. Every positive entry during blending still executes and is counted. From epoch 801 through 2500 use only the exact 1.5 distribution; no dense shadow or periodic dense field loss is attached.

A hard-zero logit can still change through contextualizer gradients and shared score parameters; the soft start avoids immediate untrained exclusion. Do not claim every omitted edge has a direct nonzero decoder gradient. If supports collapse harmfully, retain the measured failure rather than force a uniform occupancy target. One bounded correctness repair is different from retuning thresholds to rescue DEV metrics.

## 7. Bounded alternative: IAG-Q when cached aggregates are insufficient

IAG-C is the simplest complete realization of the requested reuse. Its possible failure is clear: several receivers may need different mixtures of tokens even inside the same environment cell. A spatial cover and pointwise query head may not retain enough discrimination. Diagnose this with fixed-weight TRAIN residuals and the initial soft fit, rather than assuming every loss is a sparsity failure.

One replacement reader, IAG-Q, is allowed per dataset before launching its sparse child. It uses the same atlas and cached individual final-token values/keys. Its query-conditioned weights are

\[
w^t_{qi h}=
\frac{\beta^t_{qi h}\exp(\widehat q^t_{qh}\cdot\widehat k^t_{ih}\,T_h)}
{\sum_j\beta^t_{qj h}\exp(\widehat q^t_{qh}\cdot\widehat k^t_{jh}\,T_h)},
\qquad t\in\{M,E\}.
\]

Reuse the last EM/EE route's corresponding Q/K projection, normalization and temperature parameters, applied to the query encoding and final payload states. Parameter sharing means the same live tensors, not copied initial values; register and optimize each parameter once. This is an explicit repurposing of those parameters, not a second utility network. The inherited \(\beta\) remains both the support and a multiplicative weight. A zero association remains zero; never add an epsilon floor that resurrects every edge. Use numerically stable products or masked log weights with explicit empty rows.

Gather cached token values only on the union of positive inherited associations. Do not construct nonlinear 128-wide pair messages. Geometry remains in tokens, the inherited atlas and query encoding; no new pairwise geometry MLP is added in this alternative. Nonlinear field readout uses the resulting typed contexts. Thermal uses \(w^M/\bar\mu\), the same signed source carriers and a heat-independent query response head, preserving the affine law. Its ordinary implementation may stream the selected scalar coefficients, but it must not allocate a dense Q×M×128 payload.

IAG-Q loses the exact \(\Lambda(PV)\) shortcut because its weights depend on the query. Benchmark that cost explicitly. Use structural cell candidate lists and batched gathered attention rather than sorting every query or reusing Round3's degree-bucket controller. Multiple cover anchors and heads can make the union large; record actual executed widths, padding and repeated reads. No claim of a sparse win follows automatically.

This alternative is **a replacement of the inadequate cached reader**, not an extra full-output branch. It must receive a fresh, honestly labelled soft training trajectory and its own e500 sparse child if selected. Do not combine it with source-scale conditioning, low-rank completion, larger E, deeper context or loss rebalancing. If its cost is uncompetitive, preserve the result and stop that alternative.

## 8. Execution design and costs to measure

### 8.1 Main execution order

```text
prepare_scene(structure):
    encode prescribed M/E/operation once
    run the two retained typed context cycles with minimal last EM/EE export
    construct typed soft or 1.5 atlas from live inherited logits
    project final token payloads once
    prepare P_M @ V_M and P_E @ V_E (IAG-C)
    retain P_M, source carriers, identity/version stamps and selected-edge lists

prepare_query(request):
    form the adapter's unchanged native observation/curl union
    construct a sparse, geometry-owned receiver cover over E anchors
    encode query coordinates/features once; prepare native local pairs once
    gather cached anchor payloads (IAG-C), or token values within atlas support (IAG-Q)
    evaluate one pointwise nonlinear field head and the heat-independent response head
    retain factors only when a reusable affine query is explicitly requested

apply_heat(prepared_query, h):
    apply source-indexed linear carriers with exactly the prepared association
    add the retained compact local response; scale and extract native observations
    do not rerun contextualization, selection or nonlinear readers
```

No query-batch-dependent regional pooling is permitted. Changing query order, tile size, batch companions or a subset request must not change a physical query's cover or association. Local native masks and observation unions remain complete when comparing derived curl/q.

### 8.2 Three distinct efficiencies

**Representation efficiency:** replacing the remote Q×M pair-value MLP removes its quadratic-in-width operation. This is the main opportunity, even for a fully soft atlas.

**Reuse efficiency:** IAG-C computes \(PV\) once per scene and performs at most 4/8 anchor reads per query. These are real indexed reads, not a dense Q×E bank multiplied by zeros. In IAG-Q this shortcut is absent; price its token reads separately.

**Sparse preparation efficiency:** a 1.5 atlas may let the scene-level \(PV\) and thermal transport skip zero donor edges. Implement an exact selected-edge alternative and measure it against a dense contraction of the same atlas. A dense GEMM can still be faster at E192/M12 or E64/M30. In that case use the faster qualified backend and report “sparse association with dense scene preparation,” not “zero edges were skipped.” Query-local indexed reuse may remain valuable independently.

All four typed contextualizer routes remain full access. Their work, raw/global reinjection, and final token ancestry are never credited as sparse savings. Local pairs remain a separate cost. Fewer negative or small coefficients, smaller entropy and fewer printed links are not operation counts.

### 8.3 Avoid avoidable host and autograd overhead

Validate ownership, schema, geometry and query identity once at the public request boundary, then use already validated internal tiles. Do not hash or copy full states per receiver tile. Reuse the existing guards, and never weaken stale-state detection to obtain a timing improvement.

No per-query Python loops, `.tolist()` support construction, CPU sorting, per-query dynamic lists or per-query calibration. For regular covers, construct anchor indices/weights by vectorized coordinate arithmetic; group by geometry-defined cell only when a backend needs it. Index grouping is independent of requested batch order and field errors. Price any sorting, transfers and padding.

Retain differentiability through values, inherited scores, valid measures and cover weights. Cache tensors with their autograd graph only inside the current valid forward/gradient lifetime. After an optimizer update or consumed graph, rebuild. Inference reuse at unchanged weights is allowed; changed geometry, source order, frames, masks, normalization, policy, buffers or model ownership invalidates it. Heating reuse is allowed only under the declared affine capability.

### 8.4 Complexity accounting

For IAG-C the added scene association arithmetic is on H(E×M+E×E), and scene payload aggregation is O((EM+E²)d). Query gathering is O(Qcd), c≤4/8, plus query/local/field-head work. Thermal adds O(EMHr) control transport and O(QcHr) application, with H=4 and r=8 initially. Count repeated control application separately from preparation.

For IAG-Q, add actual selected query-token dot products and value reads; do not report IAG-C's complexity for it. Neither expression includes the unchanged contextualizer, native union construction or local branch, which must be included in end-to-end measurement. Low M, large native observation unions and host dispatch can limit the observed benefit.

### 8.5 Required native benchmarks

Use the same retained high/low-M TRAIN fixtures and native requests as the new D0 comparison whenever exact identities are available. Compare D0 and new candidates on the same authorized GPU, matched numerical policy, fixed batch, output scope and sane tile choices. Round3's deterministic-child setting differed from D0; do not ascribe every old timing difference to architectural operations.

Measure cold scene→physical CPU output, prepared scene→new queries, and Thermal repeated heating as separate alternative scopes. Include cover/atlas formation, packing, extraction, synchronization and CPU return where they occur. Keep checkpoint/data loading outside those calls and account for them separately. Use two warmups and seven rotated/interleaved profiler-off repetitions, with all observed ranges retained.

At fixed B8, measure native forward/backward at requested Q1024/4096/8192, including real Thermal unions and derived outputs. Record absolute allocated and reserved peaks, resident baselines and actual local/atlas/query rows. Also measure one real optimizer-boundary peak per dataset, including its scheduled native auxiliary response; prediction-only backward is not optimizer memory. A denser Wind receiver timing may be added without extra training or physical claims if the native catalogue supports it.

One small profiler capture per necessary case is sufficient. Explain exclusive or nested scopes correctly. Do not spend the round on dozens of noisy timing variants or custom fused kernels before demonstrating a worthwhile computation pattern.

## 9. Implementation work packages

### W0 — preserve the reference and resolve the experiment contract

Read root `AGENTS.md`, the two relevant reports and the current runtime/recipe dispatch. Verify the remote tip and inspect any later user changes. Preserve all completed A2-D formal/development states, failed Round3 variants, data locks, classic loaders and solver ledger 326/326. The new family must coexist with them.

Recover the exact D0 segmented recipes, normalizers, native validation panels, source/runtime identities and saved e500/e2500 states. Check that the intended new recipe uses the successful formal sampling/objective semantics, not the obsolete five-role Wind development training loss. Reuse D0 results only after their native IDs, masks, targets and relevant protocol match. Reusing a control does not require rerunning its whole trajectory; retain its actual source and execution policy.

Seal a compact `experiment_contract.json` with families, modes, epochs, partition hashes, loss/sampler settings, allowed devices, resource limits, primary fields and the decision rules below. Use the existing manifest/identity machinery. No new approval service or elaborate receipt hierarchy is needed.

### W1 — minimal production atlas and cover contracts

Add a small optional last-cycle export to the shared attention implementation, with legacy default parity. Suggested contracts are `RouteScoreSnapshot`, `PreparedAttentionAtlas`, `ReceiverCover` and `PreparedAffineGather`. Each holds explicit axes, valid masks, physical/native IDs and ownership/version metadata. Source IDs are correspondence, never learned embedding indices.

Keep the new implementation in a clearly versioned package such as `src/honf_forward_core/inherited_attention/`, or an equivalently compact, reviewed submodule layout. Suggested files: `contracts.py`, `normalization.py`, `atlas.py`, `readers.py`, `model.py`. Prefer composition of the existing encoders/context/local primitives over copying the old 100-kB model or inheriting unused branches.

The source export must identify last EM/EE stages, their raw logits without log measure, and separately cached final values. Debug traces are opt-in, capped and sampled. Ordinary training must not export every Q×source association for plotting.

### W2 — IAG-C and native adapters

Implement both typed gathered contexts and the generic optional linear-control transport. Preserve the adapter-level Thermal temperature/curl/q extraction, material/source correspondence and Wind profile-residual conversion exactly. Add only the environment cover provider and new-core binding. Both datasets use the same preparation/decoder primitives with separate transforms, physical heads and weights.

Implement native public `predict`, scene reuse, affine apply/increment/adjoint, and stale-state rejection through the existing public workflows. The first model is self-contained and trains all intended context/readout parameters jointly. No reference checkpoint is a hidden runtime dependency. No existing direct far reader, utility scorer, H6 bank or completion module may remain instantiated unused.

### W3 — normalization and selected execution

Implement the measure-aware 1.5 reference/production pair and integrate its sparse child schedule. For IAG-C, implement an actual selected scene-edge contraction plus the exact dense contraction of the same probabilities. Implement indexed query covers without Q×E dense payload construction. For IAG-Q, if authorized by the capacity diagnosis, use structural candidate lists and cached token attention, without creating a second selection subsystem.

Report the fastest numerically qualified backend for each scope. A failed exact-backend comparison must not be disguised as a physical modeling loss; a slow selected kernel must not be declared a speedup from its operation count. Preserve both types of evidence.

### W4 — targeted correctness before meaningful fitting

Test the following substantive behaviors, with small parametrized fixtures rather than a large matrix of nearly identical launches:

- Legacy attention/default A2-D behavior remains unchanged when export/new family is disabled; exported soft route probabilities agree with an explicit masked reference at the correct stage.
- Positive measures, zero/absent donors, empty types, M=1, padding and permutations; typed masses are not mixed or multiplied twice. Splitting identical computational donor atoms preserves the normalization/gather primitive under the declared measure convention; this is not a claim that duplicating physical modules leaves the whole contextualizer invariant, because it contains actual module-count features. Physical actuator splitting additionally requires the adapter's explicit control convention.
- Weighted 1.5 normalization, shift invariance, nonnegativity, normalization, exact zeros, gradients through logits and positive measures, and reference/production agreement near support transitions.
- Literal `(ΛP)V` versus `Λ(PV)` values and gradients; indexed versus dense-reference cover gathering, source ordering and duplicate/native-query handling.
- Query permutation/subsets/tiling/batch-composition invariance; cover continuity across representative environment-cell boundaries, with full native stencil support for q/curl comparisons.
- Native Thermal T(0)=0, heat-null flow, affine increments and adjoints; Wind zero-residual→profile, correct units and no extra rotation/normalizer mean.
- Native low/high-M finite gradients and parameter updates, exact same-backend save/resume, stale model/scene/query/policy/frame rejection and training-graph lifetime handling.

Reuse existing numerical tolerances where the contracts are the same. Before new outputs are opened, specify FP64 mathematical-reference tolerances and FP32 physical/derived-output tolerances for the new family. A new representation is not expected to match old A2-D outputs. Identical-function backend parity is a separate requirement. Do not require bitwise equality of entire independent training trajectories across different GPU reduction backends; do require reproducible same-backend resume and preserve every failed stronger check.

A CPU review-only algebra probe accompanying preparation of this plan tested constructed FP64 tensors, not native models: normalization/splitting errors were ≤6.7e-16, reassociation error ≤4.5e-16, and the signed affine/adjoint examples agreed within 3.1e-14; finite-difference checks of logits and positive measures passed. That probe used a separate CPU Torch environment and does not qualify ModularDT, native geometry, GPU performance, masks or this repository implementation. Codex must reproduce the needed tests in the actual maintained environment.

## 10. Bounded training protocol

### 10.1 Preserve the successful data semantics

| Setting | Thermal | Wind |
|---|---|---|
| Segmented membership | Existing fixed25 TRAIN150 / exposed DEV22 | Existing fixed24 TRAIN72 direction rows / exposed DEV24, grouped as 24/8 layouts |
| Initialization / sampling | Seed 0 / maintained case-epoch stream | Initialization 0, sampling 42 |
| Effective / starting microbatch | 48 / 24 | 16 / 8 |
| Primary TRAIN queries | Q1024 fluid plus unchanged native auxiliaries | Q8192 total: 6144 volume + 2048 hub band |
| TRAIN objective | Maintained equal-case flow/curl, three-temperature, q and original-TRAIN response terms | Maintained 0.75 volume / 0.25 hub-band equal-channel standardized velocity objective |
| Optimizer | Constant all-parameter AdamW 3e-4, weight decay 1e-5, clip 1 | Same |
| Context | W128, heads4, depth2, E192 | W128, heads4, depth2, E64 |
| Normalization | Fit/reuse exact selected-TRAIN-only transform | Fit/reuse exact selected-TRAIN-only normalizer/profile |
| Monitoring | Existing field selector at saved 100 ages; curves every 10 | Existing volume-MSE selector at saved 100 ages; separate native five-role board |

Retain explicit source-ID masks and native observation work. Do not multiply Wind Q8192 by five, divide vector RMSE by three, reclassify the Thermal exposed DEV as a blind test, or treat three Wind directions from one layout as independent replications. The selected Wind TRAIN omission of M18 remains disclosed.

Common named contextualizer/local tensors initialize identically under the declared seed. New payload/query/response-head tensors receive deterministic named initialization. Do not load a mature full-TRAIN checkpoint into a segmented candidate. Case/query streams and denominator accumulation remain shared between new soft and sparse trajectories. Packing changes may be made before the scientific seal only, after matched-denominator checks; filling VRAM is not a scientific objective.

### 10.2 Experiment sequence and maximum portfolio

**Stage A: architecture and throughput preflight.** Complete W0–W4, then a bounded four-case/low-high-M TRAIN learnability check and real optimizer-boundary memory probe. Dispose of those pilot weights; they do not initialize science. Do not evaluate a portfolio of hidden widths, environment grids, ranks or losses. Default signed carrier width is 8 per head.

**Stage B: one fresh soft IAG-C trajectory per dataset.** Review e100 for correctness and learning, e500 for fixed-weight TRAIN/native DEV fidelity, and e1000 for fit/generalization direction. Healthy learning may mature to e2500 without having to beat a mature classic early. Fixed-weight TRAIN reviews use actual saved weights and the same queries, not rotating pre-update losses.

**Only allowed representation revision:** when the early soft fit shows a clear loss of intra-region discrimination or persistent fixed-TRAIN underfit, run the IAG-Q replacement under the same segmented contract. Record the evidence and decision before starting it. It is a fresh trajectory with its own age; no reset is hidden. At most one replacement per dataset. Do not use this allowance to change the contextualizer, losses, capacity or environment catalogue. Stop the superseded soft candidate at its declared review and retain its states.

**Stage C: one sparse child per dataset.** Choose the viable new soft family first, then branch its own exact e500 state. Continue physical training through absolute e2500 using the e501–800 normalization transition and exact 1.5 gathering thereafter. No learned utility initialization, dense-label calibration, scorer-only fit or auxiliary teacher refresh is present. All original physical objectives remain active. Its corresponding soft trajectory is the matched control.

If soft representation is clearly uncompetitive after the allowed revision, do not spend a sparse training campaign attempting to disguise that loss. Finish the negative representation result and preserve D0. Conversely, an efficient soft model with meaningful inherited gathering is a useful result even if its sparse normalization remains dense or slow.

Maximum primary trajectories are two new soft models plus two sparse children across both datasets, with at most two conditional replacement soft trajectories. Do not revive D-scale, S-Utility, S-Complete, H6 or another decoder family in this round.

### 10.3 Exposure and age accounting

A fresh Thermal soft e2500 trajectory has 375,000 primary case visits and 10,000 ordinary updates; Wind has 180,000 direction-row visits and 12,500 updates under this batch16 protocol. Thermal primary queries total 384,000,000 before native unions/addenda; Wind totals 1,474,560,000. Preserve actual observed counters rather than only those nominal products.

The sparse Thermal child adds 300,000 visits/8,000 updates after its inherited e500 prefix; Wind adds 144,000 visits/10,000 updates. Its normalization-transition and exact-sparse ages must be visible. Pilot, failed, discarded and extra benchmark optimizer updates are separately charged, not hidden in “zero-cost development.” Reused parent epochs are not newly executed training.

## 11. Evaluation and decision rules

These are prospective research criteria for this new representation, not a relaxation or reclassification of Round3's failed criteria. Seal them before inspecting new native outcomes. Use one compact decision sheet, not dozens of unrelated promotion gates.

### 11.1 Three comparison boards

**Board A — controlled mechanism:** compare the new soft and sparse models at the same literal e2500, same inputs/native outputs, and the same saved-checkpoint selection rule. Also show their common e500 starting point. This is the main test of changing the gathering normalization. A same-weight dense implementation of sparse weights is an arithmetic/execution control, not the independently trained soft predictor.

**Board B — useful model:** compare each candidate against unchanged Round3 D0 on the exact segmented panels and current same-device cost fixtures. This asks whether changing the decoder preserved the established field capability and purchased efficiency. Existing D0 is the control, not a new “best” mode selected from the candidate's adapted weights.

**Board C — practical reference distance:** retain mature full-TRAIN Thermal1404/1804 and Wind2102/2103 on the same native evaluation panel, with their own transforms, inputs and actual checkpoint ages. Do not turn their unequal data/exposure advantage into a same-age architecture failure. Never confuse the segmented D0 numbers with the much better completed formal A2-D numbers.

### 11.2 Priority fields and interpretable evidence

Thermal priorities are fluid-T mean/p90, transverse v and pressure; surface/material temperature, q and endpoint module peaks remain explicit guard fields. Preserve u and native curl. Report actual temperature/flow units as packed-native, not unverified Kelvin or SI flux.

Wind priorities are near/downstream vector errors and all six Ux/Uy/Uz component means, plus priority row-p90. Retain volume, hub and background fields. Group direction rows by layout in summaries; no receiver-count confidence intervals.

Detailed examples remain input-fixed Thermal0291/0687 and Wind69/426, using actual native fields, surfaces and hub-height planes/cuts, not smoothed point-cloud substitutes. Use the existing full DEV22/24 native board at endpoints, with all available module-count groups. Population scores are not replaced by two attractive maps.

### 11.3 Qualification of a representation direction

A new soft or sparse model is eligible for a research full-TRAIN e1000 probe when all of the following hold: native/software contracts pass in its declared backend; priority mean errors are within 10% of D0 and priority p90 within 20%; other field/component means remain within 15%, with any larger tail miss explicitly reviewed rather than hidden; and complete high-M inference is at least 10% faster than D0 **or** fixed-B8/Q8192 allocated memory is at least 20% lower with complete latency no more than 15% slower. A meaningful priority accuracy gain of at least 5% with all other mean guards and no more than 10% cost/memory increase is an alternative accuracy-led qualification.

These are operating bands, not safety certificates or multi-seed significance thresholds. Do not manufacture a gain by changing query tiles only for the candidate, subtracting residency from only its memory, excluding difficult cases or switching checkpoint-selection policies. Low-M cost is also reported; a high-M winner is not a universal winner.

Sparse normalization additionally needs a native-field comparison with its own soft model: priority means within 5% and primary tails within 10% are the target. A larger measured tradeoff may remain research evidence but does not silently pass as accuracy-preserving sparsity. To describe genuinely selective organization, require at least 20% absent eligible nonlocal associations on the high-M scene-level atlas and report the query/head-union support; do not force this fraction with a new penalty. To describe **sparse executor acceleration**, require the selected-edge backend to outperform the same sparse weights executed densely, with actual zero-edge work removal. Model-level speed versus D0 alone does not establish that narrower claim.

An efficient IAG-C soft model may qualify even when selected scene-edge kernels are slower than small dense GEMMs. The correct conclusion would be “attention-inherited, reusable decoding works; incremental sparse-preparation acceleration is unproved,” rather than another false all-or-nothing report.

### 11.4 Organizer interventions that answer the actual question

On the four fixed scenes, export last EM/EE associations, receiving anchor coordinates/IDs, three declared query covers, typed donor IDs, effective query support and the values actually gathered. Show probability mass separately from signed thermal response coefficients and latent-value norms.

Use a small, fixed set of unchanged-weight interventions: replace the inherited association by typed measure-only gathering; permute which anchor receives each association row while leaving geometry and token values fixed; remove one input-selected nonzero association and restore it; and compare indexed/selected versus literal dense execution of the same complete probability array. Mark when intervention renormalization changes the law. Removal without renormalization tests a contribution; it is not a new valid normalized policy.

Uniform/shuffled-atlas effects alone show reliance, not superior learned organization. Compare against a matched geometry-derived association control in read-only evaluation, preserving actual donor measures and per-row support degree where applicable. The primary empirical added-value comparison remains trained soft versus sparse and both versus D0. Do not launch a third training arm merely to obtain an attractive graph.

Use source perturbations that rebuild the scene to inspect changed geometry/operating-context support on TRAIN fixtures without new targets. A shifted graph is an input response, not proof that the predicted physical response is correct.

### 11.5 Native interfaces and response evidence

Retain exact heat-null and affine software checks, source-column/adjoint checks, query consistency and smooth-region geometry VJPs. Replay only the already saved Thermal response families at the selected new checkpoints, with existing baseline validity, native masks and quadrature. Missing0277 baseline remains missing. No new solver attempt, inverse optimizer, generated design, Wind physical perturbation label or unlocked TEST outcome is allowed.

FP64 factor contraction and FP32 endpoint subtraction have different cancellation behavior; report them separately. Entmax/cover differentiability does not certify the discrete native solid mask, observation topology or unobserved physical geometry response. Source identity and reusable derivatives are interface capabilities, not inverse readiness.

## 12. Conditional fresh full-TRAIN e1000 follow-up

The user's preceding development instructions permitted bounded fresh e1000 full-TRAIN probes. Apply that scope only after the new representation qualifies and current device/resource authority is confirmed. At most one selected model per dataset may launch; it can be the useful soft model when sparse normalization does not qualify. Do not launch both soft and sparse formal runs just because both exist.

Use fresh initialization, empty optimizer state, full-TRAIN-only transforms and the successful formal sampling/objective semantics. Preserve current full memberships Thermal600/Wind420 and exposed canonical89/VALID90 monitoring; Wind TEST remains locked. The stop is **literal e1000**, not a 5000 run merely reviewed at1000. A prospective longer horizon is metadata, not authorization to continue.

A sparse formal model repeats its specified soft e1–500 start and e501–800 transition, leaving only 200 exact-sparse epochs before the e1000 stop. State this maturity limit plainly. Do not replace that schedule with an untested stronger early cut to make the short formal look sparse longer. A later longer run remains a separate user decision.

At e1000, compare native fields with saved formal A2-D e1000 and actual classic e1000 checkpoints on the same panels, not a mixture of mature selected states pretending to be age1000. Verify actual update counts, sample exposure, normalizers and output-law differences; a matched epoch is not necessarily matched work. Mature reference columns remain contextual. Reuse already qualified classic arrays only when panel/target/native-input identity matches.

If either dataset fails qualification or the resource reserve is insufficient, retain its tested recipe and a precise no-launch reason. Do not mark an unexecuted formal workflow as trained, or extend a segmented child by relabelling it full-TRAIN.

## 13. Resource, storage and reporting boundaries

The latest Round3 report authorized **physical GPUs 0 and 2**, with GPU1's unrelated job protected. This supersedes the older 0/1 assumption for planning. Reconfirm actual device UUIDs, occupancy and current authorization before execution. Do not infer that a device is free now from the previous closeout. Expose one authorized physical UUID per worker as logical cuda:0. Use the existing ModularDT interpreter and pinned packages; no package upgrade or unrelated process termination.

Default round ceiling: 20 aggregate GPU-associated process-hours and 14 elapsed hours after the first scientific GPU command, reserving the last two elapsed hours for evaluation/report/cleanup. These include failures, pilots, native benchmarks, optional replacement fits and any qualified formal follow-up. Before starting a stage, forecast its remaining fit plus native review against the reserve. The ceiling is not an instruction to consume it. Run at most one scientific training writer per authorized device. No background automation or hidden follow-on campaign.

Limit disposable engineering optimizer steps to 400 in total; they cannot initialize science. There is no auxiliary scorer-fit budget because that subsystem is absent. Keep the existing scientific 100-epoch milestone cadence, latest/best aliases and full histories, with curves every10 and tqdm progress. Avoid generating a new complete checkpoint merely for every figure or read-only replay.

Record current filesystem capacities before materialization. New disk budgets are at most12 GiB under `/data` and1 GiB under `/home`, with20 GiB free headroom on each. Prior89.85/42.65 GiB figures are historical snapshots, not a current promise. Reuse exact-key shared catalogues; no dataset copies, per-arm geometry cache stores or permanent new cache without explicit budget. A memory cache has a separate declared ceiling.

Do not turn provenance into the dominant project. Reuse existing manifests, checkpoint loaders, source hashes, counters and protection checks; add only the small atlas/cover/policy identities needed by the new computation. Preserve scientific and failure evidence, but retain only selected figures and native arrays needed to reproduce claims.

### Final report and artifacts

Create a new report, preserving Round2 and Round3. Its first screen should answer: **Did field accuracy survive? What did a query read? What computation was actually removed? Was the full call faster?** Follow with separate Predictor/Organizer/Inverse gains, misses, measured evidence and next step.

Retain no more than five main figure families: (1) native fields/residuals and surfaces/cuts, (2) learning and fixed-weight TRAIN/DEV behavior, (3) an environment-anchor → original-source → three-query graph with actual weights and values, (4) native accuracy versus complete latency plus fixed-B8 memory/query scaling, and (5) saved response/interface checks and boundary examples. Put detailed tensors and all per-case tables in a compact evidence index. Every plotted comparison identifies dataset, split, checkpoint, native support, units and model mode.

One architecture diagram may explain the implementation, but it must not be presented as a measured physical graph. Do not hide a dense preparation behind a sparse edge drawing. Report c≤4/8 receiver anchor reads, typed atlas nonzeros, query-union source support, local pairs, actual executed rows/padding and decoder parameters separately.

At closeout remove only owned disposable outputs after verifying retained artifacts and dependencies; report exact removed bytes/files and measured remaining capacity. Preserve historical source/checkpoints, declared milestones, native data/caches, required response records and unrelated jobs. Audit the entire outgoing commit range, including files added then deleted in unpublished commits; keep generated data, figures and one-time diagnostic/rendering scripts local/ignored. Commit durable implementation/tests/docs on the current non-default branch, push without disturbing user work, and verify remote/local tips.

## 14. Stop conditions and how to interpret the outcome

Stop a stage for nonfinite training, broken native contracts, target leakage, exhausted device/storage/time allowance or a reproducible catastrophic representation failure. A normal immature error above a mature full-TRAIN classic is not itself a stop condition. Finish healthy authorized maturation before drawing a final sparse-versus-soft conclusion.

Do not respond to an IAG-C miss by increasing every width, adding a learned risk model or restoring an all-source correction pipeline. Use the one diagnosed IAG-Q alternative or preserve the negative result. Do not force sparse supports for visual appeal. Do not call a fully dense atlas with fast query caching a sparse-preparation success; do not discard that useful caching result merely because it is not sparse.

The strongest positive result would be: **the same task-trained EM/EE score fields organize contextual token values, a receiver reads those values through a small continuous spatial cover, native fidelity is retained, and the new computation is cheaper without a second selector.** The sparse child would additionally show that the learned association can remove scene-level or selected-query reads without materially degrading fields. Whether those findings hold is the purpose of this experiment, not a conclusion supplied by the design.

## References and inspected source map

### Repository evidence

- **[R3]** `HONF_Proj/docs/reports/HONF_A2D_Sparse_Readout_Round3_Report.md`, uploaded complete report and current branch. Main sections: Predictor; Mature sparse fidelity; Organizer; Execution; Mature complete cost; Protocol; Inverse and response limits.
- **[R2]** `HONF_Proj/docs/reports/HONF_Attention_Round2_Report.md`, especially its completed formal-D extension and comparison contract. Protected predecessor; no retrospective rewriting.
- **[C1]** [Staged attention at reviewed commit](https://github.com/cosmos2w/ModularDT/blob/36d1ca0459497f4c1e9fa5c99ba0845f80d9e671/HONF_Proj/src/honf_forward_core/content_hypergraph/staged_attention_v2.py), `SpatialRouteAttentionV2`, `StagedContextCycleV2`.
- **[C2]** [Direct/local readers](https://github.com/cosmos2w/ModularDT/blob/36d1ca0459497f4c1e9fa5c99ba0845f80d9e671/HONF_Proj/src/honf_forward_core/content_hypergraph/readers_v2.py), `DirectSourceReaderV2`, `FieldHeadV2`.
- **[C3]** [Preparation and ownership](https://github.com/cosmos2w/ModularDT/blob/36d1ca0459497f4c1e9fa5c99ba0845f80d9e671/HONF_Proj/src/honf_forward_core/content_hypergraph/model_v2.py).
- **[C4]** [Historical Round3 controller](https://github.com/cosmos2w/ModularDT/blob/36d1ca0459497f4c1e9fa5c99ba0845f80d9e671/HONF_Proj/src/honf_forward_core/content_hypergraph/source_read_plans.py), excluded from the new path.
- **[C5]** [Thermal adapter](https://github.com/cosmos2w/ModularDT/blob/36d1ca0459497f4c1e9fa5c99ba0845f80d9e671/HONF_Proj/Case_ThermalChannel/src/channelthermal/content_hypergraph.py).
- **[C6]** [Wind adapter](https://github.com/cosmos2w/ModularDT/blob/36d1ca0459497f4c1e9fa5c99ba0845f80d9e671/HONF_Proj/Case_WindFarm/src/windfarm/content_hypergraph.py).
- **[C-native]** [Thermal prescribed token inventory and observation helpers](https://github.com/cosmos2w/ModularDT/blob/36d1ca0459497f4c1e9fa5c99ba0845f80d9e671/HONF_Proj/Case_ThermalChannel/src/channelthermal/native_observations.py), `thermal_context_tensors`: regular center coordinates, all-present environment anchors, typed area measures and native interpolation.
- **[C7]** [Root development/upload/storage rules](https://github.com/cosmos2w/ModularDT/blob/36d1ca0459497f4c1e9fa5c99ba0845f80d9e671/AGENTS.md).

### External primary research and implementations

- **[P1]** Zhu et al., *BiFormer: Vision Transformer with Bi-Level Routing Attention*, CVPR2023. [Publisher paper page](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_BiFormer_Vision_Transformer_With_Bi-Level_Routing_Attention_CVPR_2023_paper.html). **[C-BiFormer]** [Official code](https://github.com/rayleizhu/BiFormer), inspected `public_release/ops/bra_nchw.py` and `ops/torch/rrsda.py`. Its region pooling/top-k/detach choices are not adopted blindly.
- **[P2]** Han et al., *Agent Attention: On the Integration of Softmax and Linear Attention*, ECCV2024. [Publisher page](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6652_ECCV_2024_paper.php), [open paper](https://arxiv.org/html/2312.08874v3), [official implementation](https://github.com/LeapLabTHU/Agent-Attention). Its agents are a related factorization, not an exact description of the proposed inherited atlas.
- **[P3]** Correia, Niculae and Martins, *Adaptively Sparse Transformers*, EMNLP-IJCNLP2019. [Primary paper](https://aclanthology.org/D19-1223/). Peters, Niculae and Martins, *Sparse Sequence-to-Sequence Models*, ACL2019. [Primary paper](https://aclanthology.org/P19-1146/). **[C-entmax]** [Official entmax implementation](https://github.com/deep-spin/entmax), inspected `entmax/activations.py`. The measure-relative entropy/root equations in this plan require their own implementation checks.
- **[P4]** *RIGNO: A Graph-based Framework for Robust and Accurate Operator Learning for PDEs on Arbitrary Domains*, NeurIPS2025. [Publisher abstract](https://proceedings.nips.cc/paper_files/paper/2025/hash/dcb91f43033bb1d367d1848806dee98d-Abstract-Conference.html), [open paper](https://arxiv.org/html/2501.19205v2), [official repository](https://github.com/camlab-ethz/rigno). This is regional-operator context, not a runtime dependency or a transferred empirical claim.
- **[P5]** Abnar and Zuidema, *Quantifying Attention Flow in Transformers*, ACL2020. [Primary paper](https://aclanthology.org/2020.acl-main.385/). Raw stage attention is not the complete information flow of a residual multilayer model.
- **[P6]** [PyTorch2.6 SDPA documentation](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html). Respect its return type, masks, backend limitations and numerical policy; calling SDPA is not proof of a particular fused-kernel dispatch.

External source links identify research and inspected public code. Before copying any implementation, inspect its license, pin the source revision and limit reuse to the needed primitive. Do not vendor a vision model, add a heavy framework dependency or assert external performance numbers as HONF results.
