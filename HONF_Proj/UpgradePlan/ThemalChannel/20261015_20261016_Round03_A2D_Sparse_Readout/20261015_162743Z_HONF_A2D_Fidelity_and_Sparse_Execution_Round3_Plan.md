# HONF Round 3 — Preserve A2-D, improve source reading, and execute useful sparse plans

**Mode:** bounded implementation and research in Codex Goal mode.  
**Reviewed repository:** `cosmos2w/ModularDT`, `agent/honf-core-next`, commit `7274b7dc494243a524c732276aa944f6642801cc`.  
**Primary evidence:** `HONF_Proj/docs/reports/HONF_Attention_Round2_Report.md`, including its October 15 formal-D extension; the companion evidence index; the current v2 source and formal training guide.  
**Status:** this is a proposed execution contract, not a record of completed new experiments. It supersedes the old H6-only follow-up preference for this new phase. It does not alter historical run permissions or evidence.

## Executive decision

Keep A2-D as the reference architecture. Do not replace its contextualized physical sources with another mandatory fixed-size content bank. The next organizer should decide **which individual source reads need expensive evaluation at a receiver**, while preserving the current contextualization and physical output laws.

The phase has three complementary objectives:

1. Improve the remaining native-field errors through one small, physically motivated change to the existing source reader, rather than another backbone redesign or a loss-weight search.
2. First make the existing dense reader a stronger execution baseline through algebraic reuse; then compare several source-preserving sparsity strategies against that optimized baseline.
3. Turn successful source selection into an actual reusable execution object, with native identities, query-set-consistent semantics, measured physical fidelity, and honest complete cost.

The default deliverable is still a good dense model when a sparse policy is unhelpful. Dense fallback is a valid application choice, but is not counted as sparse success. Full-TRAIN follow-up is allowed for at most one qualified candidate per dataset, stopping at **1,000 epochs**. No 2,500/5,000-epoch full-data extension is authorized here.

---

## 1. What the completed results establish — and what they do not

### 1.1 The successful reference is now genuinely useful

The report's literal-age Thermal comparison is:

| Native mean RMSE | A2-D e5000 | Dense1804 e5000 | Interpretation |
|---|---:|---:|---|
| u | 0.00466647 | 0.00650424 | A2-D improves |
| v | 0.000681343 | 0.000331887 | Important remaining miss |
| p | 0.00274512 | 0.00224869 | Remaining miss |
| Native curl | 0.0127165 | 0.0244341 | A2-D improves |
| Fluid temperature | 0.244292 | 0.237375 | Close at literal age, not an accuracy win |
| Surface temperature | 0.187833 | 0.429475 | A2-D improves |
| Material temperature | 0.197525 | 0.329348 | A2-D improves |
| q proxy | 0.857823 | 1.53376 | A2-D improves |
| Module endpoint peak | 0.230427 | 0.317573 | A2-D improves in mean |

Material/peak tails remain worse. At selected ages, fluid-T is `0.245423` for D4760 versus `0.218656` for Dense4738. Do not replace that selected-state comparison with the more favorable literal-age comparison, or vice versa. Thermal quantities retain packed-dataset native units. [R1, formal native tables]

Wind A2-D e2500 has near/downstream vector RMSE `0.0991353 / 0.0603161 m/s`, versus Dense2103 e2475 `0.0739270 / 0.0389866`. It improves four vector-role means over 2102, but not the near-turbine mean. The remaining transverse/vertical errors must be visible separately from the much larger Ux component. [R1, formal Wind tables]

Fixed-B8/Q8192 native gradient-memory peaks are `2.77185 GB` Thermal and `5.47623 GB` Wind, substantially below the corresponding classic measurements. These are native forward/backward high-water marks, not complete optimizer-state or pure activation-memory figures. [R1, formal memory table]

### 1.2 The current acceleration is not sparse decoding

Both context cycles, all direct source reads, and the compact local path still execute. The formal model has no H6 content bank. Its advantage currently comes from its computational formulation, capacity and execution policy, not learned source pruning. [R1; C1–C4]

The report also shows why the next baseline must be optimized honestly: Wind2102's large-tile diagnostic takes roughly `9.1 / 16.4 ms`, versus its native-default roughly `277 ms`. A2-D's own tile policy must be assessed before calling a new mask an acceleration. Keep declared defaults and separately qualified best policies as separate columns. [R1, formal cost section]

### 1.3 Formal training changed consequential settings

Do not silently reuse the previous segmented protocol as though it were the successful formal protocol.

- Previous Wind development: initialization seed42, Q4096, effective batch24, five-role/component-balanced objective, warmup/hold/cosine schedule.
- Successful formal Wind: initialization seed0, sampling seed42, Q8192=`6144 volume + 2048 hub band`, effective batch16, equal-channel standardized objective with region weights `0.75/0.25`, constant AdamW LR `3e-4`.
- Formal Thermal also uses constant LR `3e-4` and all-parameter weight decay `1e-5`, unlike the prior development decay/exclusion policy.

The new segmented comparison will inherit the **successful formal training semantics**, with TRAIN-only transforms fitted on the smaller TRAIN cohort. Old segmented numbers remain historical context, not the matched control. This does not assert that the formal improvement was caused uniquely by the objective; population, work and schedule changed too. [R1; R2; C7–C8]

### 1.4 What remains unmeasured

The formal update did not replay saved heat-response families on the newly mature formal checkpoints. Its exact-zero heat and path-decomposition checks verify software laws, not new physical-response accuracy. Individual learned K columns are not independently validated Green functions. No new solver, Wind displacement truth or inverse campaign is authorized in this phase. [R1, formal inverse boundary]

---

## 2. Source review: preserve these parts; change these parts

The reviewed implementation already includes geometry-bearing values, multiscale coordinates, two ordered `MM -> EM -> EE -> ME` cycles, raw/scene reinjection, and fresh jointly trained native heads. Do not propose those existing features as new work. [C3–C5]

| Source / symbol | Actual finding | Action in this phase |
|---|---|---|
| `content_hypergraph/staged_attention_v2.py` | Four typed routes, live normalized Q/K, measure-aware SDPA, refreshed states | Preserve the equations, depth, width and route inventory |
| `model_v2.py::prepare` | Contextualization and source projections are prepared once; static route geometry is reused | Preserve this separation and ownership; do not rerun preparation per query tile |
| `readers_v2.py::DirectSourceReaderV2.forward` | One nonlinear 128-wide hidden vector per query/source; `message_output` is applied to every pair before a headwise weighted sum | Implement and qualify post-reduction output projection |
| The same direct reader | Thermal coefficient is `thermal_output(messages)` and is intentionally NOT multiplied by information-attention weights | Thermal selection must consider thermal coefficient importance, not attention alone |
| `spatial_features_v2.py::query_source_features` | Score features include `asinh(delta/source_length)`; value features use domain-scaled Fourier offsets | Test adding the existing source-scale descriptor to the value preactivation |
| `LocalDetailReaderV2.forward` | Query/source projections are applied after gathering repeated pair rows; shared `flow_output` precedes a weighted `index_add` | Preproject unique query/source states and move the linear output after the local reduction, with its mass-dependent bias |
| `model_v2.py::_decode_features` | The nonlinear field head follows global/local aggregation; dense scalar K is constructed per A2-D tile | Preserve the nonlinear head; add a sparse affine-factor representation only for selected-reader modes |
| `model_v2.py::_reader_work_counts` | The D scalar coefficient-entry counter multiplies by `thermal_response_rank`, although the returned D tensor is `[B,Q,M]` | Validate counters against actual shapes; fix new telemetry without rewriting old receipts |
| Native Thermal/Wind adapters | Correct native transforms, geometry, shared thermal observations and laws | Reuse, with only versioned decoder/response interfaces |
| `content_hypergraph_train.py` and formal providers | Old development, H6 full1000, and formal-D protocols are separate and guarded | Add a new bounded source-read study protocol; do not bypass or weaken historical guards |

The v2 files and old loaders remain available for exact historical replay. Prefer small new reader/planner/executor modules with explicit versioned dispatch over copying the approximately 90-kB `model_v2.py` into another large model implementation.

---

## 3. Immutable scientific and application boundaries

**One shared model implementation, separate dataset weights.** Thermal and Wind retain dataset-owned features, coordinate frames, transforms, native observations and loss definitions. The shared core must not contain special case-ID rules, an analytic wake predictor, a CFD lookup, frozen D-sep, Stage-A or a second full field model.

**Keep all context interactions.** Sparsification in this round applies to the direct receiver/source read, not MM/EM/EE/ME. Every source can still influence contextualized states upstream. Selecting a source for detailed reading is therefore not the same as excluding its physical influence from the model.

**Preserve the physical capabilities.** Thermal flow is heat-independent and temperature remains linear in separately supplied heating with zero temperature at zero heating. All native temperature observations derive from one field. Wind remains nonlinear and uses its established profile-residual conversion and aligned frame.

**Separate three errors.** Track (a) physical error versus stored references, (b) distortion versus the same dense learned function, and (c) arithmetic differences between two executors for the same mathematical sparse function. A pass on one is not a pass on the others.

**Preserve the incumbent.** Completed A2-D formal and development checkpoints, histories, selectors, raw evidence, classic native loaders and all earlier protected artifacts remain immutable. No historical file cleanup, retraining or automatic replacement is authorized.

**Do not force morphology.** No target number of groups, equal occupancy, obligatory sparse fraction in every scene, or mandatory disconnected regions. The objective is useful information selection and execution, not a desired graph image.

---

## 4. Establish the best inexpensive dense execution first

### 4.1 The current direct reader in equations

For a prepared scene, let `d=128`, with four output feature heads. Write

\[
x_{qi}=A z_q+v_i+G\gamma(q-x_i),\qquad
\nu_{qi}=\operatorname{SiLU}(x_{qi}),\qquad
m_{qi}=W_o\nu_{qi}+b_o.
\]

Let `W_o^(a)` and `b_o^(a)` denote the rows belonging to output head a. The current flow/Wind context is

\[
c_q^{(a)}=\sum_i\alpha_{qi}^{(a)}m_{qi}^{(a)}.
\]

The Thermal direct coefficient is

\[
k_{qi}=w_T^\top m_{qi}+b_T,
\]

with presence/query masking, independently of alpha. Local temperature coefficients add through their native compact weights. These equations summarize the actual direct reader, not the H6 reader. [C1]

### 4.2 Exact change of contraction order

For arbitrary effective weights \(\beta_{qi}^{(a)}\), including masks or attenuation,

\[
\boxed{
c_q^{(a)}
=W_o^{(a)}\left(\sum_i\beta_{qi}^{(a)}\nu_{qi}\right)
+b_o^{(a)}\left(\sum_i\beta_{qi}^{(a)}\right).
}
\]

Do not assume the second sum is one: empty rows, non-renormalized masking and floating-point reductions all matter. Because `message_output` mixes input feature coordinates before the output is split, each head's accumulated vector has width **128**, not32. Accumulate `[B,Q,4,128]`, then apply the corresponding `[32,128]` output slice. Simply averaging four 32-wide preactivation blocks is incorrect.

The thermal projection can be composed directly:

\[
w_K^\top=w_T^\top W_o,\qquad b_K=w_T^\top b_o+b_T,
\qquad
\boxed{k_{qi}=w_K^\top\nu_{qi}+b_K.}
\]

Keep these products differentiable during training. Do not detach a composed projection or cache it across an optimizer update. In inference it belongs to the prepared model-version state.

This removes the repeated `128 x128` **output** projection from every direct pair. It does not remove the pair-specific nonlinearity, geometry, score, or source inventory. It changes neither the model's expressive class nor its trained parameters. Actual speed and memory still require measurement.

A constructed CPU FP64 check performed during this plan review compared both expressions with nontrivial fractional/zero masks and VJPs through hidden inputs, scores and all output weights/biases. Maximum context/K/VJP differences were approximately `8.9e-16 / 3.6e-15 / 1.4e-14`. This is algebra-only evidence, not repository-checkpoint or GPU qualification.

### 4.3 Apply the same principle to local reads

Preproject each unique query and module state once, then gather projected rows. For local hidden vectors \(u_{qi}\) and actual support/measure/gate weight \(w_{qi}\), use

\[
\sum_i w_{qi}(W_Lu_{qi}+b_L)
=W_L\sum_iw_{qi}u_{qi}+b_L\sum_iw_{qi}.
\]

Keep the thermal local coefficient, compact window, measure reference and all overlapping pairs unchanged. Do not apply a donor measure twice. Native local-pair discovery, support and gradients remain authoritative.

### 4.4 Qualification and execution choice

Implement `legacy` and `reduced_projection` as two backends for the same learned state. Qualify at current segmented and mature low/high-M checkpoints in both datasets using:

- primary model-space/native outputs, native curl and q with their propagated observation budgets;
- source/query-coordinate VJPs, heat JVP/VJP, and parameter gradients;
- one cloned real AdamW step, including named moments and post-step outputs;
- empty donors, zero/partial gates, padded sources, permutations and query chunks;
- explicit trace reconstruction only for requested trace rows.

Use the existing v2 arithmetic tolerances before viewing failures: primary FP32 `atol=3e-6, rtol=3e-5` in TRAIN-scaled spaces, gradient `atol=1e-5, rtol=1e-4`, FP64 reference `atol=1e-9, rtol=1e-7`; propagate native linear observation coefficients for derived outputs. Reuse stricter applicable historical checks without rewriting them. Do not substitute the looser classic cross-device parity envelope for executor equivalence.

If a backend fails, preserve the result, keep legacy available, and distinguish inference-only from training qualification. Do not spend the whole phase chasing last-bit parity. Never silently alter a running lineage to use the new backend.

Benchmark a small fixed tile set `{128,512,2048,8192}` on TRAIN fixtures. Choose the fastest qualified configuration under the same memory headroom; record forward and training choices separately. All learned sparse candidates must subsequently be compared to this qualified, optimized dense baseline, not only to native tile128.

---

## 5. One focused forward-accuracy candidate: source-scale information in values

### 5.1 Hypothesis

The current direct score already knows the smooth physical-scale displacement

\[
\eta_{qi}=\operatorname{asinh}\left((q-x_i)/\ell_i\right).
\]

The direct value path uses domain-scaled Fourier offsets instead. In Thermal, the score cannot directly carry this feature into K because K is deliberately not attention-weighted. Module states already contain size information, so this is **not** a claim that radius or geometry is entirely missing. The hypothesis is that a direct physical-scale path into the nonlinear value helps resolve body-scale and downstream structure with negligible extra capacity. [C1–C2]

### 5.2 Candidate `D-scale`

Keep every existing A2-D component and add one bias-free projection:

\[
\boxed{
x_{qi}^{\rm scale}
=A z_q+v_i+G\gamma(q-x_i)+G_\ell\eta_{qi}.
}
\]

`G_ell` has only `coordinate_dim x128` weights:256 for Thermal,384 for Wind. Initialize it to zero after the existing name-based initializer, so the initial function and all unchanged named tensors match D0. No new field branch, wider contextualizer, extra layer, source gate or extra loss is introduced.

Reuse the already calculated physical-scale descriptor. Do not recompute its delta/asinh independently for scoring and values. Preserve generic per-axis characteristic lengths supplied by adapters; do not hard-code a Thermal plume formula, a Wind wake cutoff or a universal flow direction.

Compare **D0 versus D-scale** under the protocol in Section10, to literal2500. Keep the old surface/material/flux wins and inspect v/p and thermal tails. Wind near/downstream Ux/Uy/Uz are all reported. If the change does not help, retain D0. Sparsity development does not depend on this hypothesis winning.

No new Fourier frequency sweep is authorized: multiscale coordinates already exist. No new positivity, divergence-free or Navier–Stokes penalty is added to an analytic/shared-grid Thermal benchmark without a separate generator-consistency study.

---

## 6. Sparse decoding: three strategies, one source-preserving interface

All strategies act on individual contextualized sources. They use a shared plan/executor and retain the complete context network and explicit local branch. The three strategies are deliberately different; only the two learned strategies are eligible for physical co-adaptation.

| Strategy | Purpose | New learned capacity | Principal limitation |
|---|---|---|---|
| S-Attn | Cheap, no-fit headwise attention-mass selection | None | Attention alone does not control Thermal K or nonlinear output error |
| S-Utility | Predict importance of actual source messages and thermal coefficients before expensive values | One small shared scorer | Omitted-tail error and score miscalibration |
| S-Complete | Retain a cheap source-preserving completion and sparsely evaluate only its nonlinear residual | Small rank16 completion plus the same scorer interface | Completion can be too weak or its overhead can outweigh savings |

Do not add entmax, a new tree, a second routing network for every physical channel and a learned packet hierarchy as further independent candidate families.

### 6.1 Shared semantics

Compute legal lightweight all-source scores/projections first; select support **before** computing expensive pair-value geometry/features/nonlinearities for selected modes. A small `[B,Q,M,heads]` score tensor is allowed. Dense `[B,Q,M,128]` fine activations are not allowed on the deployed sparse path unless a declared same-function dense fallback is selected.

Use one physical-pair support shared by the four feature heads and Thermal's coefficient read. A head-only support reduction is not an executed-pair reduction if their union is dense. Preserve per-head weights inside the shared support.

For direct omission, define `beta = alpha * g`; do not renormalize over the retained sources. This approximates the existing reader rather than turning it into a new conditional softmax without disclosure. An explicit renormalized variant can be a bounded read-only diagnostic, not another fitted family.

Thermal's ordinary route must not depend on the current heating vector. Selection may use structural inputs, encoded states, query coordinates and a **declared control envelope** independent of the particular heating call. Keep the plan fixed across nominal heat, increments and transpose products. Flow and Thermal eligibility can differ: positive attention measure must not silently erase an otherwise present and valid heating-control column.

### 6.2 S-Attn: a necessary cheap control, not the preferred Thermal method

For each query, take the union of the smallest source sets retaining95% or99% of mass in every head. Include the declared protected near sources. Retain original head weights, and evaluate selected fine rows only.

Measure native physical error and same-dense distortion. Record physical-pair union size, not just each head's effective count. For Thermal evaluate the joint flow-and-temperature result: do not assert a heat-preservation guarantee from alpha. This control answers whether a sophisticated scorer is necessary.

Include equal-executed-count nearest, adapter-declared directional geometry where meaningful, and source-score-shuffle controls. Their counts include protected support and actual head unions. No reference outcome may determine the geometric rule or the displayed scenes.

### 6.3 S-Utility: the primary learned selector

Use a small width32 predictor operating on cached low-dimensional source/query projections and the existing small relative-geometry descriptor. Its outputs estimate nonnegative magnitudes of:

- each head's weighted source contribution to the direct latent context, normalized by fixed TRAIN context scales;
- for Thermal, the source's signed K magnitude multiplied by a declared positive heating envelope and divided by the native-temperature scale.

Magnitude is a selection signal, not a claim that the output contribution is always positive. Keep signed coefficients and messages for reconstruction and interpretation. Headwise magnitude sums or a smooth positive combination are preferable to a head average that hides a large contribution in one head.

Training targets come from dense **model computations on TRAIN only**, not independent physical source labels. For example,

\[
\rho^{F,a}_{qi}=\|\alpha^{(a)}_{qi}m^{(a)}_{qi}\|/s_{c,a},
\qquad
\rho^T_{qi}=H_i|k_{qi}|/s_T.
\]

Here `H_i` is a declared bound on absolute source heating for the operating contract, not the source's current heat. The selector must not remove a source solely because current heating is zero. Obtain `H_i` from the dataset-owned admissible-control contract, or declare a finite positive TRAIN-only operating envelope before calibration; do not describe a sampled TRAIN maximum as a certified physical limit. A requested heat vector outside the declared envelope must be rejected by that sparse-response contract or evaluated by an explicitly reported full-access policy. The geometry/flow path itself must not be selected from current heat, and both endpoints of an increment comparison must use the same declared policy. Exact signed K magnitudes are available only during bounded dense teacher queries, not through an unpriced full read at deployment.

A useful model-only inequality is

\[
|T_D(q;h)-T_S(q;h)|
\leq\sum_i(1-g_{qi})H_i|k_{qi}|,
\quad |h_i|\leq H_i,
\]

when the same local branch is preserved. The inequality uses the **actual** K, not a learned estimate. Therefore it motivates labels and diagnostics but does not certify an arbitrary deployed learned plan. Propagate this error through native observation stencils when analyzing q; reference fidelity remains a separate empirical test.

#### Smooth support rather than a hard top-k discontinuity as the default

A fixed scalar threshold per operating point permits adaptive counts without a fixed H. For a smooth nonnegative combined score s and threshold tau>0, use an endpoint-flat gate:

\[
t=\operatorname{clip}((s-\tau)/\tau,0,1),\qquad
\psi(t)=6t^5-15t^4+10t^3,
\qquad g=\psi(t).
\]

Only pairs with `g>0` need fine values. The transition region is a real attenuated model contribution and must enter distortion/work accounting. It is not an invisible soft shadow. Attach near protection with a smooth geometry-only envelope that is one in the protected interior and zero outside the existing local-support radius; `g_total=1-(1-g)(1-g_near)` avoids a discontinuous Boolean override. The existing local branch and all of its pairs remain unchanged.

Use fixed TRAIN-calibrated thresholds rather than choosing tau from the current batch/query mean. This preserves single-query/subset/chunk semantics. A dense execution fallback can evaluate the same g-weighted function; it may not silently change g to1 because a batch happens to have many active pairs.

Calibrate at most two nominal operating points targeting roughly75% and50% of expensive active source-pair rows on a fixed TRAIN probe. These are operating-point targets, not forced per-scene counts or guaranteed DEV budgets. Select the least aggressive useful point under TRAIN fidelity and measured cost. Do not repeatedly tighten thresholds when the model resists sparsity.

### 6.4 S-Complete: cheap all-source information plus selected corrections

This candidate addresses scenes where dropping even many individually small terms creates unacceptable error. It must not reinstate an expensive full base MLP alongside the original full fine MLP.

Work in the pre-output-projection hidden space. Let r=16 and define a small source-preserving completion

\[
a_q=A_b z_q,\qquad b_i=B_b z_i,\qquad
\nu^0_{qi}=U(a_q\odot b_i).
\]

Biases can be represented by declared constant features. Source states remain individually indexed and contextualized; r is a cheap algebraic feature rank, not a count of physical sources or hyperedge slots. Fine decoding uses

\[
\boxed{
\widetilde\nu_{qi}=\nu^0_{qi}
+g_{qi}\big(\nu^{\rm fine}_{qi}-\nu^0_{qi}\big).
}
\]

At g=1 this is the original full reader. Implement an actual all-full fast path that directly executes the fine reader and bypasses cancelling completion work. A weak completion cannot irreversibly bottleneck the model because selected corrections retain the original individual source nonlinearity.

Do not materialize `nu0[B,Q,M,128]`. For each head,

\[
\sum_i\alpha^{(a)}_{qi}\nu^0_{qi}
=U\left(a_q\odot\sum_i\alpha^{(a)}_{qi}b_i\right).
\]

Add only selected residual vectors before the post-reduction output projection. For Thermal, the completion contribution contracts source heating before query expansion; selected residual K entries are sparse. In particular, include the scalar bias contribution as `b_K * sum_i h_i` over present heating sources, together with the completion and selected residual terms; do not add that bias a second time for selected corrections. Attention weights never multiply the thermal coefficient.

The utility targets now measure **fine-minus-completion** error, not full-message magnitude. Fit the completion using TRAIN dense-read targets and output-relevant projections/K, not only an arbitrary hidden-state norm. End-to-end native physical supervision remains primary during co-adaptation.

This is an explicit approximation. There is no claim that rank16 is universally sufficient. If the completion cannot support a useful correction budget, retain S-Utility or dense. Do not grow rank, add a separate environmental decoder and expand local support in the same repair.

---

## 7. What the organizer means, and how packets execute

### 7.1 The organization is a source-preserving execution hypergraph

For query q, let

\[
\mathcal S(q)=\{\mathrm{ID}_i:g_{qi}>0\}.
\]

Queries sharing an exact support set can be represented by a packet

\[
\mathcal E_b=\mathcal R_b\cup\mathcal S_b,
\qquad \mathcal S(q)=\mathcal S_b\ \text{for }q\in\mathcal R_b.
\]

This is a directed, typed execution relation between receiver records and physical-source records. Sources may belong to many packets. There is no fixed six-slot bank and no pooling of source identities into a centroid. The number of distinct packets is an outcome of the query support and model, not a learned capacity to be forced downward.

Actual head weights, gates, signed Thermal coefficients and source states remain point-specific within a packet. Sharing an index list does **not** permit sharing a query-dependent nonlinear message between different receivers.

This is not evidence of physical causal groups. The shared context has global ancestry, and S-Complete also retains cheap information from every source. The value of the organization is tested by physical error, exact execution, reuse and equal-work controls.

### 7.2 One mathematical policy, bounded executor choices

Provide two sparse executors behind the same plan:

1. **Packed-pair reference:** gather admitted `(batch,query,source)` rows, evaluate their fine hidden vectors, and scatter/reduce headwise contributions and sparse K.
2. **Support-packet/bucket executor:** group identical source-support signatures or pad only to the selected degree within small degree buckets, evaluate contiguous batches, and restore original query order. Actual support and weights are unchanged.

At the current M<=30, a compact support signature is convenient; an integer-bitmask fast path may be used when safe. Do not make the reusable model assume M<=30 or reinterpret IDs as learned ordinal features. A general index-list representation remains the contract.

No recursive spatial tree, per-query Python loop, CPU round-trip for each mask, or global clustering solve. Count sorting, bucketing, metadata, gather/scatter and extra padding. If signatures are mostly unique, use packed pairs or dense-masked execution rather than forcing merges.

A coarser shared **superset** over a spatial tile changes support/work and is not an exact packet optimization. It is not part of the primary fitted candidates. Any bounded optional superset diagnostic must retain its different executed-pair count and native-field comparison.

### 7.3 Why generic sparse attention is not a drop-in replacement

The expensive A2-D value is query-dependent: `SiLU(query_projection + source_projection + relative_projection)`. Ordinary SDPA/FlexAttention uses donor values shared across queries. A score/mask API does not automatically accelerate this custom pair-value computation. Keep SDPA for context; use explicit selected pair-value execution for the decoder. [C1; X2]

Native Sparse Attention is relevant as an example of co-designing selection and hardware execution, not as a transferable speedup guarantee: its long-context setting is unlike a decoder with at most30 native sources. Entmax is relevant to sparse distributions, but zero weights alone do not remove a value evaluation, and Thermal K remains independent of attention. [X1–X3]

Do not upgrade the whole environment or introduce a new attention dependency merely to try another mask. Optional compilation is limited to pure kernels supported by the installed environment, with cold compilation and recompilation measured. Newer online documentation must not be assumed to describe installed Torch2.6 APIs.

---

## 8. Sparse interfaces and numerical behavior

A prepared source plan must bind at least:

- model/core/adapter/router identities and parameter/buffer versions;
- scene identity, coordinate frame, units, normalization and source IDs/masks;
- query IDs, coordinates and query tensor versions;
- branch-specific eligibility, protected support, selected source indices, gates, original head weights and chosen operating point;
- the Thermal control-envelope contract, independent of current heat;
- executor policy and measured logical/allocated/executed row counts.

Reuse is valid only while these bindings remain valid. Moving modules, changing receivers, changing source inventory, changing thresholds, changing learned weights or changing transforms invalidates the relevant plan. A query reorder may reorder an existing bound plan only through an explicit verified ID mapping.

Extend the affine response interface with sparse coefficients plus optional completion factors. Ordinary apply/JVP/VJP must not reconstruct a dense K or rerun the fine reader. Explicit K export remains a separately priced operation, preserving every physical control column and native extraction order.

Full-access recovery, zero heat, flow heat-null, affine addition/scaling, source permutation, padded sources, empty sets, query subset/chunk consistency, stale-state rejection and exact restoration remain required. Compare sparse executor versus dense-masked reference at **identical g, alpha and coefficients**; different masks are a model comparison, not an arithmetic check.

For geometry derivatives, recompute live geometry and smooth gate values. Do not detach a live continuous gate or support weight just to pass a finite-difference test. The endpoint-flat gate helps avoid a jump when a term appears, but integer support discovery and existing native observation stencils still have scoped derivative semantics. Test both fixed-support derivatives and a small support-transition sweep; do not claim global differentiability of the full dataset catalogue. Native curl and q must be tested on ordinary sparse fields because neighborwise errors can be amplified by differentiation.

Do not clip signed K to positive, overwrite null channels after inference, or insert a response teacher at deployment. A negative estimated K is a warning to compare with available physical evidence, not by itself permission to impose a different law.

---

## 9. Bounded diagnostics before additional fitting

Use a small input-fixed TRAIN diagnostic panel spanning low/high module counts, plus existing native DEV visual scenes only after rules are fixed. These diagnostics are for selecting a method, not building a new target dataset.

### 9.1 Cost decomposition

On one low-M and one high-M TRAIN fixture per dataset, measure preparation, score/geometry generation, fine hidden evaluation, projection/reduction, local branch, native extraction and complete calls. Do not add overlapping medians. Use a short profiler capture only to identify actual kernel/launch costs; headline timing is profiler-off.

The simple cost relation is

\[
T_{\rm sparse}=T_{\rm fixed}+T_{\rm score}+T_{\rm plan}
+T_{\rm gather}+T_{\rm selected\ value}+T_{\rm reduce}+T_{\rm extract}.
\]

After Section4, the dense value path may be much cheaper. Reductions in the old `QM d^2` count are not credited to sparsity. Judge sparse saving against the **post-optimization** cost and measured Amdahl limit.

### 9.2 Model-derived sparsifiability screen

At existing D development checkpoints, and later the chosen aligned-protocol D host, construct bounded dense teacher traces on TRAIN receivers. Compare retention75%/50% and full access using:

- original attention-mass ranking;
- exact model-value/K importance ranking as a diagnostic, never a deployable cheap oracle;
- learned scorer proposals;
- nearest/directional/shuffled equal-count controls.

The exact-value ranking is not guaranteed to be the globally optimal subset; cancellation and the nonlinear field head can defeat a sum-of-magnitudes ranking. Report actual same-dense distortion, stored-reference error and work. Use at most256 input-fixed receiver probes per TRAIN scene for label construction, with bounded storage. Never produce a full dense K archive for every native query/case merely to train the router.

If even a model-derived useful subset is unavailable at75% retention, prefer testing S-Complete over pressuring a deletion scorer. If the optimized dense read leaves no plausible complete-call saving, keep the experiment as a bounded memory/reuse study and say so.

---

## 10. Unified, protocol-aligned segmented training

### 10.1 Fixed populations and successful-formal semantics

Use existing input-fixed memberships: Thermal TRAIN150/exposed DEV22; Wind TRAIN72/exposed DEV24. Wind retains24 TRAIN layouts and8 DEV layouts with three directions each; the omitted TRAIN M18 stratum remains disclosed. No new target access beyond established development panels.

| Setting | Thermal | Wind |
|---|---|---|
| Initialization seed | 0 | 0 |
| Sampling seed / semantics | 0; existing native case/epoch draws | 42; formal native two-region streams |
| Effective batch / initial microbatch | 48 /24 | 16 /8 |
| Primary queries per case | 1024 fluid, plus native auxiliaries | 8192 total:6144 volume +2048 hub band |
| Optimizer | Constant AdamW3e-4, all-parameter decay1e-5, clip1 | Same |
| Physical objective | Current formal joint flow/thermal/q/TRAIN-response objective | Formal equal-channel standardized0.75 volume +0.25 hub-band objective |
| Context/local/read width |128 /128 /128 |128 /128 /128 |
| Context cycles / heads / E |2 /4 /192 |2 /4 /64 |
| Checkpoint/DEV review |100-epoch saved cadence; curves10 | Same development cadence |
| Development stop |2500 |2500 |

Reuse successful formal provider logic through a new explicit segmented scope; do not fake a full-TRAIN identity or refit scales using DEV. Keep Wind's five-role/component evaluation board as a diagnostic even though the training objective has two regions. Freeze the segmented selector before new outcomes: Thermal formal broad-field monitor, Wind formal volume-MSE monitor, both on the declared100 cadence. Report all priority fields separately from the selector.

Expected primary exposure per2500 trajectory:

- Thermal:375,000 case visits,10,000 ordinary updates,384,000,000 primary fluid queries; native union/auxiliary/response rows separately.
- Wind:180,000 row visits,12,500 ordinary updates,1,474,560,000 primary queries; five updates per72-row epoch including its8-row final update.

These are not the old Wind7,500-update/Q4096 protocol. Microbatch packing may change only after a same-function qualification and before a lineage starts; effective batch, sample streams and denominators remain fixed.

### 10.2 Stage A: accuracy pair

Train D0 and D-scale freshly to2500 in each dataset, with common named initialization, native query/case streams and physical losses. They use the same qualified execution backend; when reduced projection is not training-qualified, use legacy for both.

At100 verify learning and native correctness. At500/1000 assess physical priority means, fixed-weight TRAIN versus DEV, actual resource forecasts and continued healthy fitting. Complete healthy runs to2500. Retain literal100/500/1000/2500, declared100 milestones, latest/best aliases and histories. Do not stop a healthy run solely because it has not beaten a mature classic.

Choose the dense host by the predeclared physical/cost rules in Section12. D-scale is optional: no win means D0 remains the host. The architecture choice is recorded as development selection, not an independent confirmatory result.

### 10.3 Stage B: compare learned sparse children from the same saved dense prefix

Use the chosen host's exact **e500 checkpoint** for at most two sparse children per dataset: S-Utility and S-Complete. The already completed dense host supplies the equal-age continuation control. Each child has500 inherited physical epochs and2000 new physical epochs, reaching absolute2500. Preserve all inherited physical Adam moments and sampling/schedule age. New scorer/completion moments start at their actual birth.

This is a conditional learning study, not a claim that the sparse children were fresh independent2500-epoch runs. The full-data phase below starts genuinely fresh.

Before opening a sparse child's new DEV results:

1. Fit its scorer and, when applicable, rank16 completion on the frozen e500 host's TRAIN model targets for at most256 auxiliary optimizer updates. Charge these updates and teacher reads separately. No new physical labels or mature formal teacher are used.
2. Fix the two TRAIN operating-point thresholds and select one primary training point using TRAIN fidelity/cost. Primary default is the less aggressive useful point; retain the other only as a bounded inference frontier.
3. Use a100-epoch transition at absolute501–600: `g_effective = (1-lambda)+lambda*g`, lambda rising0→1. This interval executes dense fine rows whenever the coefficient is nonzero and is reported accordingly.
4. From601, execute only actual selected fine rows during ordinary physical updates. Original physical objectives stay unchanged. Preserve a small supervised scorer/completion signal so omitted pairs are not permanently deprived of learning signal.

During co-adaptation, refresh current-model dense utility labels on at most one TRAIN case and128 probes every8 ordinary updates. Keep total extra dense teacher rows below5% of primary fine-row exposure, measured with native unions. These are bounded training-only auxiliary reads; do not run a complete dense shadow at every sparse step. Teacher outputs are detached labels, while the main selected model retains ordinary gradients through context, values and continuous gates.

Record native physical loss, utility/completion losses, their gradient share, density and full/sparse read counts separately. Use one fixed TRAIN-calibrated auxiliary coefficient per candidate, capped to a modest initial gradient share (5% of the native objective on fixed calibration batches); record the cap/realized share. Do not sweep that coefficient on DEV. Main model fidelity, not a large routing-label loss, determines the outcome.

Stop clearly broken children for correctness, nonfinite state, resource limits or persistent severe native regression; preserve their evidence. Otherwise allow the declared2500 endpoint. At most one substantive repair across the sparse study per dataset is allowed, chosen before its next outcome review and recorded as a new child/source lineage. No unbounded sequence of gates, ranks and loss recipes.

### 10.4 Fit-count ceiling

At most four principal physical trajectories per dataset: D0, D-scale, S-Utility, S-Complete. The two sparse trajectories reuse a dense500 prefix rather than reexecuting it. S-Attn and geometric controls are read-only policies, not additional full training runs. There is no H6 rerun, third sparse family, second-seed portfolio or new dataset campaign in this phase.

---

## 11. Measurements that make the result understandable

Use a compact front page with **accuracy, complete cost, actual fine work, and the operating decision**. Put long inventories and numerical qualification detail in an appendix.

### 11.1 Physical fields

Thermal: native mean, median, p90 and worst for u/v/p/curl/fluid T/surface T/material T/q/endpoint peak; M-stratified summaries. Endpoint peaks come from maxima on the common material support, not maxima of increments. Explicitly watch v/p, fluid-T tails and material/peak tails while retaining the present surface/q advantage.

Wind: native Ux/Uy/Uz and vector RMSE for all five roles; near/downstream six-component summary; p90/worst, module-count/direction/layout summaries. Report layout clustering and avoid treating millions of receivers as independent cases.

Use full-native Thermal endpoints and the exact retained Wind Q8192 five-role panel. Raw reference values, masks, coordinate frames and classic native transforms are immutable. TRAIN monitor, DEV monitor and native endpoint are different scopes.

### 11.2 Sparse fidelity and training behavior

For each learned model, evaluate sparse and all-full modes at the same weights, plus its dense continuation reference. This distinguishes weight adaptation from restriction error. Report:

- same-function dense-versus-sparse distortion in model and native units;
- stored-reference errors, both absolute and ratios to the dense host;
- each head's support, the physical-pair union and protected/fallback fractions;
- exact nonlinear rows, extra bucket padding, score rows, base rows and teacher-refresh rows;
- one input-selected source removal/restoration with actual signed/output changes;
- query-order/subset/chunk checks and a small geometry/support transition sweep.

No full-mask fallback is counted as a successful partial plan. No per-head sparse count is substituted for the executed physical-pair union. No reduction obtained only by removing already padded sources is credited to learned physical sparsity.

### 11.3 Cost

On the same device, compare legacy dense, optimized dense, dense-masked sparse reference, packed sparse, and the one qualified packet backend. Keep output scopes, precision, target-free/inference modes and CPU return identical. Use two warmups and seven rotated profiler-off repeats. Report observed ranges, not unsupported confidence claims.

Show complete cold-scene calls, prepared new-query calls, same-query repeated heat controls, and complete training epochs separately. A cached query K used for repeated heating is useful response reuse, not new spatial sparse decoding.

Maintain B8 while increasing Q1024/4096/8192 for native forward/backward memory. Include allocated/reserved baselines and one complete optimizer-boundary peak. A single optional Q32768 target-free stress is allowed, with unique versus repeated queries disclosed. Do not shrink B to manufacture a memory advantage.

### 11.4 Six figure families, not a diagnostic wall

1. **Learning:** physical priority error against saved age, actual updates and measured complete training work; dense versus selected sparse comparisons.
2. **Thermal:** reference/dense/sparse/signed error on two input-fixed native scenes, plus one surface/q curve; include per-panel RMSE and relativeL2.
3. **Wind:** the existing low/high-M native planes and2D/4D/8D cuts for all three components; no point-cloud projection presented as a CFD plane.
4. **Organizer:** three fixed receivers, physical source IDs, actual retained/cheap/omitted paths and a matched removal/restoration. Show support maps and packet membership without relabelling weights as physical heat fractions.
5. **Fidelity–cost frontier:** native error versus complete milliseconds and B8 memory; raw fine-row reduction in a separate panel. Include optimized dense and equal-work geometry controls.
6. **Response guard:** limited saved Thermal transfers, endpoint-peak changes, heat-null and numerical-law summary. No inverse generator panel.

Cap overview density at about6–8 panels per page. Full arrays/metric tables belong in machine-readable evidence. Readable source IDs, native units and clear legends matter more than the number of exported figures.

---

## 12. Predeclared research decisions

These are research operating criteria, not physical safety certificates or proof of superiority. Freeze them before new candidate outcomes are opened.

### 12.1 Choosing D-scale versus D0

Prefer D-scale only when a named primary physical target improves at least5% at the matched literal2500 endpoint:

- Thermal: fluid-T mean or the geometric mean of v/p mean-error ratios.
- Wind: the geometric mean of the six near/downstream component-error ratios.

Other priority means should be within5%, remaining native-role means within10%, and priority p90 within10% of D0. Complete high-M latency and actual training-epoch cost should remain within15% of D0. Report all exceptions; a marginal mixed result retains D0 rather than becoming a new default. If the native-tail change is visibly inconsistent, do not select a model solely because a pooled scalar passes.

### 12.2 Qualifying a useful sparse research operating point

Compare a sparse child both to its same-weight all-full mode and to the corresponding dense host at the same absolute age. For automatic full-data follow-up require:

- all primary means within5% of the dense host, other role/component means within10%, primary p90 within10%; Thermal q/native curl and material/peak tails remain explicitly checked;
- at least30% fewer actual fine nonlinear source-pair rows on the declared high-M workload after unions/padding, not merely a lower soft-mask mean;
- either at least10% lower complete high-M inference time, **or** at least20% lower fixed-B8/Q8192 memory with no more than15% complete-call slowdown;
- correctness, native laws, genuine source-selection use and query/plan ownership tests pass;
- learned selection is not materially worse than simple equal-work geometry controls on the population comparison; document ties as no demonstrated learned-selector advantage;
- bounded saved Thermal response replay shows no new gross direction/endpoint failure attributable to masking. A small noisy response tradeoff is reported separately, not converted into an inverse-readiness pass.

A memory-only qualifier is labelled **memory/reuse direction**, not sparse speedup. Small-M full execution is allowed, with its fraction reported. The criterion does not require sparse wins on every receiver or every dataset.

### 12.3 Outcomes that are still useful

- Exact output-projection reuse wins, but sparsity does not: keep the improved dense executor and explain the remaining bottleneck.
- D-scale improves accuracy, but no sparse point qualifies: keep the accuracy candidate and its optional sparse interface as research only.
- Sparse support preserves fidelity but is slower: retain one bounded implementation/reference, do not launch further variants solely to make its graph prettier.
- Thermal and Wind choose different operating policies: acceptable with one shared implementation and dataset-owned calibration. Do not force the same threshold or policy when the physical readout laws differ.

A full-data follow-up can select a qualified accuracy-only D-scale instead of a failed sparse model. This consumes that dataset's one allowed follow-up, not an additional run. Unchanged D0 alone is not a reason to repeat an already available full1000 reference.

---

## 13. Conditional fresh full-TRAIN1,000 follow-up

At most two fresh full-data runs: one chosen candidate per dataset. Qualify after completed segmented evidence, source/recipe freeze and a remaining-budget forecast. Do not launch one full run for every sparse candidate.

- Thermal: all600 TRAIN cases, full-TRAIN transforms, effective48, Q1024 plus unchanged auxiliaries, constant LR3e-4, declared future5000 horizon, **hard stop1000**.
- Wind: all420 TRAIN rows, full-TRAIN transform/profile, effective16, Q8192 with the successful0.75/0.25 region objective, constant LR3e-4, declared future2500 horizon, **hard stop1000**.
- Both: fresh learned weights and Adam moments, no development checkpoint or mature teacher initialization. One dataset-specific native adapter plus the shared core. No frozen learned partner.

For a sparse candidate, reproduce its declared training algorithm: dense physical warmup through500, TRAIN-only utility/completion initialization,501–600 transition and actual sparse training afterward. The e1000 result has only400 fully sparse physical epochs; report that scope. It is potential for later extension, not sparse-model maturity. Fit/calibrate utility scales and thresholds on full TRAIN only using the same rule fixed in development.

Expected ordinary work at1000: Thermal600,000 visits/13,000 updates/614,400,000 primary queries; Wind420,000 visits/27,000 updates/3,440,640,000 primary queries. Auxiliary router/completion steps, dense refreshes, native unions and failures are additional measured work. Retain the successful formal monitoring/selection definitions: Thermal every10 epochs on its broad-field monitor, Wind every5 epochs on its fixed volume-MSE selector; retain curves every10, bounded100-epoch milestones, and exact latest/best aliases. The common native diagnostic board remains separate from those selectors.

Use a new `source_read_full1000` authorization-aware recipe path. Do not reuse `formal_d_full_train_v1` to obtain automatic5000 permission or masquerade as a qualified A2-H6 run. CLI stop1001/2500/5000 must reject before allocation for this phase. No queued continuation job.

Compare literal new1000 to the **saved formal A2-D1000**, plus actual classic1000 checkpoints on the same native endpoints. Add mature reference columns only as practical distance, preserving exact ages. Old selected4760/2465 states cannot stand in for1000. If the new candidate changes value geometry or training algorithm, label the comparison accordingly; it is not an isolated sparse-executor ablation.

Retain physical forward accuracy, tails, complete cost and actual route maps for the new full state. A small arithmetic test or good segmented result must not substitute for that real native1000 board.

---

## 14. Concrete implementation sequence

### P0 — Reconcile current state and seal the study

Read repository `AGENTS.md`, the current report/evidence index, native protocol guides and this plan. Confirm the live tip and local working state; inspect any changes after the reviewed commit. Do not reset the user's work. Inventory the exact current D checkpoints, source runtime, native transforms, saved1000 anchors, data locks and owned artifact roots. Reuse existing preservation machinery; do not create another general audit framework.

Write one concise experiment manifest containing the fixed cohorts, model/recipe choices, branch lineage, thresholds/calibration rules, primary metrics, budget, GPU mapping, retention and stop rules. Confirm report-metric reproduction on one native fixture per dataset before a population rerun.

### P1 — Qualify a better dense executor

Implement Section4 behind opt-in execution dispatch. Test algebra, real native gradients/updates, local projection reuse and B8 memory. Record profiler-off complete timings, actual source projections and head-reduction tensors. Choose a qualified backend/tile policy for both sides of each learning comparison.

### P2 — Add D-scale and run the controlled accuracy pair

Implement the zero-initialized source-scale value input, explicit config and strict roundtrip. Keep all physical objectives and successful-formal sampling semantics. Run D0/D-scale on both datasets with the bounded stages in Section10; preserve the selected host's e500 state for sparse descendants.

### P3 — Build a real source-plan API and three policy screens

Add score/value separation, exact all-source recovery, branch-aware support, smooth gates and selected value evaluation. Implement S-Attn and equal-work geometric controls. Build bounded TRAIN labels and S-Utility; add rank16 S-Complete only as the second learned candidate. Test packed and dense-masked parity before large native calls. Support-packet grouping is an execution choice, not a fourth learned family.

### P4 — Mature the bounded sparse children

Use identical chosen dense500 physical parent/moments, named new parameters, query streams and original objectives. Complete eligible children to absolute2500 within the budget, with explicit auxiliary-work and actual sparse training counts. Apply one substantive repair allowance per dataset at most. Do not change a checkpoint's scientific identity in place.

### P5 — Decide and execute the fresh full1000 follow-ups

Freeze the winning method/operating point per dataset, or select the qualified accuracy-only model. Verify CUDA-hidden dry-run, fresh normalization, intended macro/micro batches, one real maximum-size optimizer update on a disposable model, and hard1000 stops. Launch only those one-per-dataset runs. Evaluate saved-age full references without training classics again.

### P6 — Deliver and close

Provide `HONF_A2D_Sparse_Readout_Round3_Report.md`, a short evidence index, actual tested commands, explicit selected recipe(s), one-page result summary, six readable figure families and a next-step recommendation. Preserve all successful and failed scientific states. Clean only exact round-owned disposable files. Audit outgoing Git content, commit/push on the non-default branch and verify remote/local tips. End with a factual completed/blocked/failed table, not “Goal complete” as a substitute for the scientific decisions.

---

## 15. Suggested code boundaries

Names below are proposed; adapt them to the repository's maintained conventions without duplicating the whole architecture.

| Area | Proposed addition / minimal change |
|---|---|
| Pure arithmetic | `content_hypergraph/source_read_execution.py`: headwise pre-output reductions, exact K projection composition, local reduction, legacy/reduced backends |
| Reader change | `content_hypergraph/source_reader_variants.py`: D-scale and reader hooks sharing v2 projections/states; no new contextualizer |
| Planning | `content_hypergraph/source_read_plans.py`: small utility scorer, gates, branch eligibility, source/query IDs and plan validation |
| Sparse execution | `content_hypergraph/source_read_sparse.py`: packed reference, bounded degree/signature buckets, exact dense-masked fallback |
| Response | Versioned sparse affine-query object and contractions; explicit dense-K export only |
| Model integration | Small versioned decoder factory/hooks in the current core; serialize policy/executor/schema and reject stale prepared state |
| Dataset adapters | Reuse `channelthermal/content_hypergraph.py` and `windfarm/content_hypergraph.py`; add decoder/response capability dispatch without changing native transformations |
| Providers | Reuse formal sampler/objective implementations in `training/content_hypergraph_task_v2.py` and `windfarm/training/content_hypergraph_task.py`; explicit segmented/full1000 scopes |
| Engine / CLI | Reuse TrainingEngine, optimizer groups, source bindings and clean-stop; add new protocol/authorization validation rather than weakening old validators |
| Evaluation | Extend current native evaluator, benchmark and response tools to expose dense/sparse same-weight policies and actual row counters |
| Tests | Focused source-reader arithmetic, plans, native laws, selected/dense-masked gradients, real optimizer boundary, provenance and stop tests |

Do not infer sparse counts from intended policy strings. Add counters where the fine function is actually called. The D thermal output is a scalar per source/query; count its actual elements rather than multiplying by the H6 rank flag.

---

## 16. Resource, safety and storage envelope

Use only physical GPU0 and GPU1 after confirming their current UUIDs and availability. The report's last snapshot contained an unrelated GPU1 process; do not kill, pause or modify it. If a requested lane is unavailable, serialize work on an authorized available lane or report a resource block. Do not switch to GPU2 automatically.

Scientific Python remains `/home/wanglz/miniconda3/envs/ModularDT/bin/python`; use the installed FP32/TF32-off numerical baseline and bounded CPU threads. Check actual installed package versions, GPU ownership, RAM and disk at startup; historical capacities are not current facts.

Proposed ceilings for this phase:

- 20 aggregate GPU-associated process-hours, including starts, failures, teachers, preparation and evaluation;14 elapsed hours from first scientific GPU work, with the final2 hours reserved for reporting/cleanup. They are ceilings, not a reason to spend all of them.
- At most4 principal physical trajectories per dataset as specified, plus one fresh full1000 follow-up per dataset. At most256 initial auxiliary fit updates per learned sparse child, and400 total disposable engineering optimizer updates across the phase. Periodic teacher reads have their separate5% row cap.
- Reforecast at100/500/1000 and before each optional child/full launch. Reserve evaluation/native response work before using the remaining budget for training. When budget is insufficient, preserve a useful completed comparison rather than start an unfinishable portfolio.
- Peak new persistent storage at most12GiB on `/data`,1GiB on `/home`; retain at least20GiB free on each after predicted peaks. No new persistent dataset/geometry cache or source-data copy. Bounded RAM reuse of existing verified catalogues is allowed with explicit actual-host headroom.
- Keep at least the larger of20% of device capacity or8GiB free during training probes. More GPU utilization is not itself a scientific target; use larger microbatch/tiles only with unchanged effective updates and native numerics.

Store declared100-epoch milestones, latest/best aliases, source/config/normalizer bindings, all scalar histories, selected native arrays and the necessary figure masters. Prefer bounded model-target label buffers to all-query dense dumps. No per-step figure rendering, duplicate HTML/PDF bundles, massive per-query trace exports or repeated source-tree copies. Failure logs and compact mismatch arrays remain; exact cleanup may remove only owned transient copies after required scientific evidence is retained.

WindTEST and previously unexposed Thermal targets remain locked. Repeatedly exposed canonical89/fullVALID90 results must not be described as independent tests. Solver usage remains326/326; no new reference attempts, inverse optimizer, design search or generator training.

---

## 17. Completion checklist

The report must answer, in ordinary language:

**Forward:** Does the small value-conditioning change improve a named remaining physical error without erasing the current advantages? What is still worse than Dense1804/2103?

**Execution:** Did moving shared projections after aggregation save complete work at unchanged weights? After that optimization, does source selection still save complete time or meaningful fixed-B memory?

**Organizer:** Which sources are read finely at three named receivers, which are omitted or retained cheaply, what happens when a selected read is removed, and do simpler equal-work rules perform as well?

**General workflow:** Do both datasets use the same implementation and training phases with their own legal transforms/output laws, and can a fresh full1000 model be reproduced without a mature learned partner?

**Response boundary:** Are heat laws, native observations, numerical derivatives and plan ownership correct? Which saved physical responses remain wrong or unmeasured? Do not report inverse readiness from these software tests.

A convincing outcome is a better dense baseline plus at least one sparse operating point that earns its cost on real native fields. A negative sparsity result is still useful if it identifies a clear value-retention or hardware limit without discarding the successful A2-D model.

---

## Sources and review scope

### Repository/report basis (all code reviewed at7274b7dc494243a524c732276aa944f6642801cc)

- **[R1]** `HONF_Proj/docs/reports/HONF_Attention_Round2_Report.md`, complete original study and October15 formal-D extension; attached copy read in full. Its reported GPU timings and native accuracy are source-derived, not independently rerun during this review.
- **[R2]** `HONF_Proj/docs/guides/Content_Hypergraph_Formal_D.md`, including successful-formal sampler/objective distinctions and output-only continuation.
- **[R3]** `HONF_Proj/docs/reports/HONF_Attention_Round2_Evidence_Index.md`, inspected scientific contract/reference mappings and endpoint identities; local ignored arrays were not accessed by this review.
- **[C1]** `HONF_Proj/src/honf_forward_core/content_hypergraph/readers_v2.py`, especially `DirectSourceReaderV2`, `LocalDetailReaderV2`, `QueryEncoderV2`, `FieldHeadV2`.
- **[C2]** `.../spatial_features_v2.py`, `query_source_features`, `relative_gamma`, `pair_score_features`.
- **[C3]** `.../staged_attention_v2.py`, masked reference/SDPA and staged state updates.
- **[C4]** `.../model_v2.py`, constructor, initializers, preparation, prepared-state validation, `_decode_features`, direct/affine readout and work counters.
- **[C5]** `.../contracts.py` as referenced by the current prepared/reader implementation; Codex must inspect the full schema before extending it. This review did not independently exhaust every contract definition.
- **[C6]** Current Thermal/Wind native adapter and response laws as documented inR1/R2 and their reviewed reader/provider call sites. Revalidate the actual modified integration boundaries.
- **[C7]** `HONF_Proj/tools/content_hypergraph_train.py`, version/protocol dispatch, engine settings and stop validation.
- **[C8]** `HONF_Proj/Case_ThermalChannel/src/channelthermal/training/content_hypergraph_task_v2.py`, native prediction/response work and formal optimizer identity.

The physical-size value hypothesis, projection reassociation, sparse policies, decision criteria and new experiment design above are this review's proposals. They are not findings already established by the report.

### External primary sources consulted for design context

- **[X1]** Yuan et al., *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention* (2025), arXiv2502.11089. `https://arxiv.org/abs/2502.11089`. Relevant lesson: selection and hardware execution must be co-designed. Its long-context language-model results are not HONF speed predictions.
- **[X2]** PyTorch, *FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention*. `https://pytorch.org/blog/flexattention/`. Relevant distinction: score masking, block-skipping and mask-construction overhead are different; standard values are donor-only. Check installed-version support rather than importing current documentation assumptions into Torch2.6.
- **[X3]** Correia, Niculae and Martins, *Adaptively Sparse Transformers*, EMNLP-IJCNLP2019,2174–2184. `https://aclanthology.org/D19-1223/`. Relevant distinction: learning sparse probability support is not by itself a selected execution implementation or a thermal-response support guarantee.

