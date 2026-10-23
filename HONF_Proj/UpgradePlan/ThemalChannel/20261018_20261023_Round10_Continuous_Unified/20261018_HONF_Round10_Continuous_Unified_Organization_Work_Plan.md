# HONF Round10 — mature controls, continuous sparse retrieval, and unified interaction organization

**Status:** prospective development and experiment plan, not completed evidence.  
**Prepared:** 18 October 2026.  
**Reviewed repository:** `cosmos2w/ModularDT`, branch `agent/honf-core-next`, remote tip `9e5fd74a02a80e9cd1bb21d74d57dde10bbd26b5`.  
**Primary report:** `HONF_Joint_Native_Hypergraph_Round9_e3500_Report.md`, read in full, including its negative results, amended horizon, physical-reference limits and historical failures.  
**Uploaded report SHA256:** `857d69da09cb45310b4327956d54c6057dcf1c65570ed923460c2e08af6b35a7`.  
**Review boundary:** static inspection of committed implementation through the GitHub connector. No local checkpoint, ignored figure master, GPU experiment, uncommitted workstation change, or HET-MF dataset was executed or independently validated by this review.

This is one integrated campaign. Do not stop after reproducing the e3500 boards and request another plan. Finish the prescribed mature controls, implement the specified continuity/memory revision, and execute the automatic unified-preparation branch when its declared conditions are met. Conditions below determine the next action within this plan. Resource exhaustion or a genuine correctness defect still warrants a resumable stop; never relabel incomplete work as completion.

The general research objective is unchanged:

> Learn from multi-field reconstruction → discover a reusable organization of original module/environment interactions → reuse that organization for modular-design evaluation and eventually generation.

Forward accuracy, physical response, field continuity, actual information paths, sparse execution and inverse utility are separate evidence axes. Neither maximum forward accuracy nor speed is the project's defining objective. Conversely, a useful E-conditioned predictor is not, by itself, evidence that hypergraph grouping is useful.

---

## 0. Executive decision: do both, without mixing their evidence

**A. Complete the original matched e5000 horizons.** Continue all four Thermal arms and all four Wind arms from their literal e3500 scientific checkpoints to e5000, preserving the scientific function, original schedule, optimizer state, sampling, losses and native physics. Thermal runs have scheduling priority. This closes the maturity question with 2,000 router epochs, including the original deterministic phase, rather than another prematurely stopped router experiment. It does not assume additional training will remove hard-route seams.

**B. Build and train one continuous sparse-retrieval family.** Retain the E-first, original-record decoder. Replace discontinuous membership, one-winner dispatch and hard nearest/best-donor protections with a continuous, exact-zero retrieval function. Train its hypergraph and direct-sparse versions from the same shared dense M+E e3000 parents. Use a memory-bounded score→routing→E→M→local computation, not checkpointing of wide values alone. The all-open member must recover the dense M+E mathematical function.

**C. Implement unified preparation and run its controlled comparison automatically.** The same typed incidence mechanism must govern actual contextual value exchange, not only describe a dense contextualizer after the fact. Remove hidden learned global-summary broadcasts and layout-derived input aggregates in a separately versioned local-input capability. Run a matched dense/hypergraph/direct triplet first on Thermal; run the Wind triplet if the predeclared safety/resource conditions are satisfied. This is not a license to delete useful dense controls or claim that dense scoring has no influence.

**D. Make the design-use diagnostic scientifically relevant.** Replay compatible saved candidate rankings and physical perturbations under explicit provenance. Memory success is no longer a prerequisite for an offline ranking diagnostic. No new solver, locked TEST, full-population training, HET-MF bulk generation, or unbounded inverse-search campaign is authorized by this plan.

### Campaign sizes

| Track | Thermal main updates | Wind main updates | Required/conditional |
|---|---:|---:|---|
| A: four old arms, e3501–5000 | 24,000 | 30,000 | Required |
| B: continuous H/S, e3001–5000 | 16,000 | 20,000 | Required after numerical qualification |
| C: unified D/H/S, e3001–5000 | 24,000 | 30,000 | Thermal automatic gate; Wind second automatic gate |
| Maximum main scientific updates | 64,000 | 80,000 | **144,000 total** |

The Track-A total is 54,000; adding the existing 27,000 Round9 updates would complete the originally planned 81,000-update Round9 campaign. Count the old 27,000 as inherited history, not newly executed work. Tracks B/C are new scientific functions, not extensions of the hard-router result. Maximum terminal trajectories: eight mature old arms, four continuous-read arms, and six unified arms. This is a bounded comparison, not a hyperparameter sweep.

---

## 1. Evidence interpretation that must survive the next round

### 1.1 Report-derived findings

The report establishes executed original M/E reads and useful E conditioning. Thermal H's DEV fluid/surface/material temperature means improve by 9.03%/7.28%/5.54% against dense M. S is at least as strong on several important quantities; H TRAIN surface/material means are 46.70%/49.76% above S. D9-ME itself substantially degrades some Thermal fields. Keep **both** dense controls and trained S.

The e3500 hypergraph is not yet reliably sparse at its final reader: Thermal H reads 97.65% of eligible DEV E pairs and 98.10% on selected TRAIN. H/S have different per-type work; neither a lower loss nor a visually structured N×N directory establishes a grouping advantage at equal work.

H/S received only 500 router epochs. Their original deterministic training phase starts at router age 1001. Insufficient sparsity is therefore inconclusive. Hard routing nevertheless already produces a separate architectural defect: at the smallest reported Thermal half-step, temperature jumps remain 0.268725 for H and 0.163653 for S while dense controls at the same coordinates vary by roughly 10^-6. Do not explain these finite jumps away as incomplete maturation.

The selected training-memory target failed. Relative to the actual all-open M+E control at aligned steps, H allocated peaks were 17.54–28.19% higher on Thermal and 81.43–84.70% higher on Wind. Complete-cycle peaks did not improve. The report did not establish a phase-separated S selected-memory saving.

Physical response is not resolved: mixed geometry-response fluid-temperature error averages 0.029384 for H versus 0.019085 for dense M, a 53.97% increase. The saved catalogue does not supply independent environmental forcing or a complete geometry-plus-heat factorial. All eight invariance audits passing their envelopes is **not** a smoothness or physical-response certificate.

The report's 156 completed audit scopes mean completed measurements, not 156 scientific successes. Its generator-revision uncertainty, failed support-shift diagnostic, strict Thermal one-step parity miss, and limited independent reference catalogue remain explicit historical facts.

### 1.2 Newly inspected implementation findings

These are source findings, not measured profiler attribution:

| Source / symbol | Observed implementation | Consequence to investigate |
|---|---|---|
| `joint_native/routing.py::binary_sample` | Hard `logits >= 0` forward with a training relaxation | A continuous coordinate change can abruptly change read support |
| `routing.py::dispatch_joint_native_seed` | One canonical winning seed | Entire member lists can switch at a seed boundary |
| `routing.py::nearest_environment_protection`, `direct_typed_gates` | Hard nearest E and best-donor floors | Replacing dispatch alone cannot remove every seam |
| `joint_native/graph.py::build_joint_native_graph` | Best cross-type donor is a mandatory hard winner | Scene/geometry changes can switch a forced cross-type member |
| `graph.py::dense_st_gate` and `JointNativeRouter.route` | Dense B×Q×N gate algebra even around selected execution | A sparse value list does not imply sparse routing activations |
| `joint_native/readers.py::read_typed` | Full-request scores, alpha, gate and support are constructed before value tiling/checkpointing | Narrow-score MLP activations and routing state can dominate memory |
| `readers.py::_flat_reduce` | Packed values still index a full alpha tensor | Value sparsity alone leaves a full B×Q×N×H buffer |
| `joint_native/model.py::_prepare_native_decode` | Pre-E M/E scores are built for routing; E read computes the same pre-E E scores again | Reuse E scores within a live tile; post-E M scores are different and must still be recomputed |
| `joint_native/budget.py::collect_work_observation` | Expected/controller/observed work objects include full support-shaped tensors; retained in `_joint_entries` | Reduce and checkpoint work paths properly; merely storing a scalar does not free its autograd ancestry |
| `content_hypergraph/model_v2.py::prepare` | Dense typed cycles plus learned scene-summary broadcasts; directory capture only at last cycle | The present directory is a decoder organizer, not the controller of all upstream information |
| `content_hypergraph/staged_attention_v2.py::SpatialRouteAttentionV2.forward` | Native capture can accompany the dense attention update | Captured scores are not proof of skipped contextual value exchange |
| Thermal `native_observations.py::thermal_context_tensors` | E nearest-module/occupancy features are derived from all M; operating context includes count/density | Input-side aggregation must also appear in a shortcut/provenance audit |
| Wind `content_hypergraph.py` feature contract | Operating feature 3 is turbine-count fraction; support bounds are prescribed frame/domain metadata | Remove count only in the new local-input capability; retain and declare the actual frame/domain |
| `tools/joint_native_train.py` | Original time/device/root/horizon contracts are embedded in launch validation | Do not reopen an expired Round9 receipt or disable guards; create a new execution authorization and explicit continuation mapping |

For scale, a hypothetical FP32 score-MLP hidden tensor of shape B8×Q8192×E192×32 contains 1.5 GiB; a B8×Q8192×E192×H4 score tensor contains 192 MiB. These are shape arithmetic, **not measured live allocations**. Measure which tensors are actually saved, duplicated and alive at the peak.

### 1.3 What this plan does not infer

It does not attribute all errors to gating, prove that a particular allocation caused the measured peak, identify attention with physical causality, or assume that more epochs solve seams. It also does not infer fresh physical truth from a model-side input shift. New formulas below are design proposals and must be tested as such.

---

## 2. Non-negotiable scientific and engineering contracts

Keep Thermal's source-qualified heat-affine capability and Wind's native velocity transform unchanged in the existing datasets. Source heating remains an original per-solid-cell amplitude, not integrated watts; native T and q-proxy are not silently converted to SI. E records never become artificial heat sources. Flow remains heat-independent only for the existing Thermal capability.

Preserve original IDs, positive-measure versus presence semantics, native query/role identity, all native auxiliary losses and response cadence, equal-case aggregation, TRAIN-only normalization and locked TEST boundaries. Preserve sampler seeds (Thermal 0, Wind 42), with initialization seed 0 where inherited. Do not replace native targets with parent predictions.

Record separately:

1. value-plane paths: which original records contribute values at each stage;
2. control-plane paths: which inputs determine scores, gates, normalizers and routing;
3. externally supplied scene data: geometry frame, boundaries, materials and operating conditions;
4. derived quantities and physical readout laws.

A sparse value graph does not imply conditional independence from every omitted input because the control plane may still inspect it. The new architecture must remove **unrecorded value broadcasts**, not conceal the existence of cheap dense routing scores.

No learned fixed-width edge bank, pooled edge prototype, mandatory six-slot bottleneck, nearest-neighbor replacement of learned groups, posthoc output smoothing, or hidden dense fallback. Final reduction of already query-conditioned individual messages remains allowed; any finite contextual vector is an explicit representation choice, not an asserted sufficient statistic.

---

## 3. Track A: finish the old experiment without rewriting it

### 3.1 Exact source inventory

Use the e3500 evidence index to resolve physical paths below the existing Round9 run root. Verify literal file hashes before loading. Do not train an inference-only export or a best-field alias.

| Arm | e3500 checkpoint SHA256 |
|---|---|
| Thermal D9-M | `9fde741d332b6fbba7081023d5f75330e3a0e552462a4ea06751a4f4d288e208` |
| Thermal D9-ME | `3ef9f8f8cdda780fa56c83ccbc2000de929986b53616e597093926a12f359d46` |
| Thermal H9-ME | `843e17df16d56ed52335122b39dac7f8d8652f770d2b3b2e71b25cc11db8afab` |
| Thermal S9-ME | `b5a266412d9df7dd9f08dc047f3cc705423d40d6119cfa9d8e712eedf2ec886d` |
| Wind D9-M | `4abd8977bdc97085c49be4119b19189238d1e1627bb066f55af3ee2653d1255c` |
| Wind D9-ME | `f499b8df220c1a773548556be3ab8fb6d62824ec6eb7838ba937aaab94c064c9` |
| Wind H9-ME | `8bda53d5a5f7f6ab20db395b7cff4b6b9509640280c82250ac08d4d9feac0f96` |
| Wind S9-ME | `d0381d8c36b67133b40dc5cda3772e93491ea27e17d360b99e001e2c7bbd7520` |

The common dense M+E e3000 sources for Tracks B/C are:

- Thermal: `4afbefa80d8a9aaa1ed3deba5cc7032b914e85f9fc20484e81b39a622e2d904c`.
- Wind: `f367756a2292f04342991d2b9db788ae3bbbb048a9bacf67840f21910e914fbf`.

These are checkpoint file SHA256 values, not Git commits or Git blob hashes. Source-location hashes and readable exports are different objects.

### 3.2 Continuation identity

Create new Track-A output directories with an immutable parent-reference record. Restore model parameters/buffers, named Adam states and their steps, RNG, sampler namespace, absolute age, macro/cadence clocks and dual windows. Keep old directories and original receipts read-only. Continuing the mathematical experiment does not require appending to the closed directory.

The report identifies separate frozen scientific source snapshots for dense M and M+E fits. Reconstruct the relevant source closure rather than assuming the latest evaluator is the original training code. A public evaluator-only repair is not automatically a training-source amendment. Retain exact source hashes for each subprocess.

Write a new Round10 execution reservation around the inherited scientific identity. Validate both: unchanged scientific recipe and new bounded execution authorization. Never replace a historical resource hash with a newly computed one while claiming the old recipe is unchanged.

Use Thermal effective/micro batches **48/8**, not the earlier Round8 48/24. Wind remains 16/8. Each full Thermal development epoch visits 150 cases in four macro-updates; Wind visits 72 rows in five. Keep partial macros and response calls.

### 3.3 Horizon and review policy

Continue e3501–5000, with reviews at e4000, e4500 and e5000. The original sparse schedule remains anchored at e3001; do not restart its age. At e4001 it enters the original deterministic phase. Retain checkpoints every 100 epochs and full histories. Literal e5000 is primary; selected aliases are secondary.

Scheduling priority: finish the Thermal four-arm stages together, then Wind; independent runs may use separately reserved devices. Do not give H more epochs than S or drop the stronger S control.

Stop only for a user stop, unsafe resources, nonfinite/semantic failure, or persistent catastrophic divergence under the report's existing declared rule. Do not introduce ordinary fidelity/memory/speed stopping gates into this maturity study. More subtle scientific misses are results.

An engineering optimization is not applied to Track-A scientific fitting unless a separately documented same-function amendment is genuinely required for safe execution. Prefer using the frozen fitting implementation for Track A and testing optimized replay in new namespaces. Changing floating reduction order can preserve toleranced semantics without preserving the exact optimization trajectory; never call that bitwise continuation.

---

## 4. Memory work package: repair allocation lifetimes, not just row counts

This package is required before substantial new-family training. It is also a same-weight diagnostic on e3500 H/S and dense M+E. A profiler observation, a predicted footprint and a measured improvement must have separate labels.

### 4.1 Measure the real peak

Use actual mixed-scene native batches, including Thermal auxiliaries/response and Wind's real provider. Compare one loaded model per fresh process. Profile selected and dense-router steps, forward/backward, and four-epoch native cycles for **both H and S**. Distinguish live allocations, reserved allocator memory, immutable model/optimizer memory, and host transfers.

Instrument saved tensors by shape/storage identity without retaining their values. Avoid a profiler hook that itself holds tensors and creates the leak under investigation. Attribute bytes to preparation, pre-E scoring, directory/dispatch, work constraints, E normalization/value, M normalization/value, local branch, physical extraction and optimizer. Save a small digest/table; do not dump every activation.

Include full-Q versus Qtile scaling, E64/E192, low/high M, mixed batching and repeated calls. The primary live-tensor lifetime test spans at least **40 complete real-objective epochs**, discarding updates afterward; short native boundary tests are engineering evidence, not scientific training.

### 4.2 Tile the entire dependency chain

The new executor's recomputation unit is:

```
query tile → narrow M/E scores → retrieval gates and work numerators
           → selected E nonlinear messages and reduction
           → E-conditioned query
           → post-E M scores and selected M messages
           → local response / field features / source coefficients
```

Do not build full-request alpha/score-MLP activation tensors outside this unit and then call the value stage “checkpointed.” Keep only compact outputs, necessary base inputs and explicitly bounded metadata across tiles. Existing scene states can be reused, with their autograd graph intact.

Use non-reentrant checkpointing for the complete pure tile where appropriate, or a verified custom recomputation function. No RNG draw, resource counter mutation, directory reconstruction with a different sample, validation side effect, or closure holding wide tensors inside recomputation. Recompute the identical mathematical gate function; continuous models have no stochastic routing in their primary path.

Use Qtile=256 as the initial implementation setting; qualify 128/256/512 and select using real memory/cost measurements without changing receiver counts. No per-query Python loops and no B×Q×N×N or B×Q×N×N×d tensor. Internal donor blocks are permitted for narrow scoring and normalization.

### 4.3 Required implementation changes

- Reuse pre-E E scores within the tile. Post-E M scores are not interchangeable with pre-E dispatch scores.
- Return compact contexts/source coefficients in ordinary calls. Full alpha, membership and per-token messages are opt-in bounded traces, not unconditional outputs.
- Selected value execution gathers native rows before Fourier features and nonlinear values. A dense reference evaluates the same function independently.
- Reduce work observations inside the checkpointed tile to per-case numerators/denominators. Simply appending a differentiable `.sum()` outside checkpointing still retains its upstream graph.
- Do not detach the work loss or E-conditioned query to manufacture memory savings.
- Preserve macro-level loss/constraint arithmetic. A mean of independently hinged microbatch penalties is not the declared macro objective.
- Avoid repeated Boolean scans of the entire selected-index array for every query tile; pack/query-sort or construct tile-local indices once.
- Keep the NN value block and all physical source coefficients. Do not substitute low-rank/pooled decoder values.
- Check same-weight dense/packed predictions, E-conditioning values, source K, native losses, parameter gradients and one-step Adam boundaries within declared tolerances. A long replay diverging after many floating-point updates is not a same-weight memory comparison.

### 4.4 Fair comparison

Apply the same tiling/recomputation options to dense M+E. Measure peak with the actual all-open function, not only a dense same-mask reference that retains H's router. Report the irreducible difference between dense-score routing overhead and saved wide values. Synthetic forced masks may measure the executor's attainable memory benefit but are not learned sparsity evidence.

Do not claim a full-cycle capacity saving from a selected step if another phase dominates. For new continuous models, compare their actual training cycle (no periodic hard-estimator steps) and separately charge any optional engineering dense reference. Timing is descriptive, not an admission criterion.

---

## 5. Track B mathematical proposal: continuous exact-zero native retrieval

### 5.1 Why a different function is necessary

At a hard boundary, changing a selected source can change a nonzero message or signed source coefficient by a finite amount. More epochs may make neighboring decisions agree accidentally, but do not make hard switching continuous by construction.

Replacing only one-seed dispatch with softmax is insufficient: hard memberships, hard cross-type partners, nearest-E protections and direct best-donor floors remain. A dense positive softmax over all sources would also remove actual sparsity.

The proposed replacement has continuous coefficients that reach **exact zero**, with overlapping positive seed access. It retains native records and creates no learned pooled edge value. This is a new scientific model; do not retrofit its claims or outputs into Round9.

### 5.2 Typed native incidence

Let N=M+E and keep typed original records and native seed IDs. For each contextual route t in {MM, ME, EM, EE}, use the actual captured raw scores, original head mixture and bounded scale to form a finite calibrated preactivation a^t_si. Reuse shared scalar/head calibration rather than introducing a separate importance network.

Use the proposed continuous gate

\[
 b^t_{si}=\psi(a^t_{si}),\qquad
 \psi(a)=\frac{[a]_+^2}{1+[a]_+^2}.
\]

It is exactly zero for a≤0, positive for a>0, bounded below one, and C1 at zero. Implement a numerically stable equivalent for large finite inputs. Mask invalid donors structurally. A present seed's own eligible native record has b_ss=1 as an identity correspondence, not a nearest-neighbor decision. Keep contextual MM self-exclusion separate from retrieval self-membership.

Remove forced **winning cross-type partners** in this family. Every optional cross-type value path uses the same continuous incidence. Empty-type behavior is handled explicitly by the normalizer below, not by a hard fallback.

The birth calibrator uses at most eight fixed TRAIN input batches and targets approximately 0.98 mean final per-type access, with near-open actual support. It must inspect the composed query access, not only sigmoid membership probabilities. All-open tests force b=1 on eligible pairs. Ordinary birth immediately after the all-open boundary can differ slightly from the dense parent; record that difference rather than claim exact recovery at a near-open nonunit gate.

### 5.3 Continuous sparse seed access

Use 1.5-entmax on the existing calibrated **pre-E** typed query–seed scores:

\[
 \pi(q)=\operatorname{entmax}_{1.5}(s(q)),\qquad
 \pi_s\ge0,\quad\sum_s\pi_s=1
\]

for nonempty scenes. Empty scenes use zero access. Use a verified implementation and an independent small CPU reference, with typed masks and exact permutation behavior. This replaces categorical one-winner selection with continuously weighted, potentially sparse overlap. Do not impose top-1, top-2, maximum active-seed count, ID tie-breaking, or a forced geometric partition.

Keep native type identities and type-specific score calibration. Measure selected seed types and E-catalogue cardinality bias. A new type prior or normalization must not be silently added to cure an unfavorable result.

The primitive is motivated by sparse continuous probability mappings [E1]; the HONF composition, loss and usefulness remain new and unvalidated. Supporting multiple active seeds does not by itself guarantee useful final-source sparsity.

### 5.4 Native final access, without a Boolean union

Define

\[
 a_i(q)=\sum_s\pi_s(q)b_{si},\qquad 0\le a_i(q)\le1.
\]

It is a convex combination of native member permissions. A source reached through several seed paths is evaluated **once**, using its final a_i. Do not evaluate a separate native value per path or sum duplicate full-amplitude source responses.

For M only, preserve the existing smooth local geometry envelope ℓ_i(q)∈[0,1] through

\[
 g_i^M(q)=\ell_i(q)+(1-\ell_i(q))a_i(q).
\]

For E use g_i^E=a_i. Remove hard nearest-E, best-M and best-E protections. Do not turn a positive local window into a Boolean full-strength exception: that recreates a jump at the compact support boundary. The original local reader remains a separately declared branch, with its existing compact window and no enlarged radius.

The active native value set is exactly {i:g_i>0}. **No arbitrary posthoc cutoff** is allowed. A tiny positive gate still costs a full selected value row. Report its magnitude and count it.

This changes the Round9 full-amplitude-binary-read convention. It does not change the physical law: the new continuous gate is part of the learned source response K_i, not an interpretation as the physical importance or sign of that source. Track ungated k_i and gated K_i=g_i k_i separately; watch for large compensating k_i and narrow transition layers.

### 5.5 Normalization that stays defined as a type's support disappears

For one type/head, first form the original all-eligible measure-weighted score distribution

\[
 w_i=\frac{\mu_i\exp(L_i)}{\sum_{j\in\mathrm{eligible}}\mu_j\exp(L_j)}.
\]

Use standard stable log-sum-exp arithmetic and the inherited explicit measure convention. For no eligible information donors, w=0. Define

\[
 u_i=g_iw_i,\quad Z=\sum_i u_i,\quad
 D_\varepsilon(Z)=Z+\varepsilon(1-Z)^2,\quad
 c=\frac{\sum_i u_i m_i}{D_\varepsilon(Z)},\quad\varepsilon=10^{-3}.
\]

For 0≤Z≤1, D is positive; the total message mass Z/D≤1. The missing mass has a **fixed zero message**, not an unreported learned/global value. At Z=1 this is the original information reduction; at Z=0 it returns zero continuously. This is a proposed normalization regularizer, not a known physical law. Fix ε in the new schema, measure Z and attenuation distributions, and apply the same formula in C-H, C-S and their all-open reference. Do not use an arbitrary `if no sources: nearest()` repair.

Retain proper bias placement. For a contextual output projection with bias, weigh its bias by the actual reduced message mass so a zero donor path does not inject a purported donor message. Local per-record FFN/bias terms are separate local operations. The decoder's original per-record message bias remains inside each message as before.

The denominator depends on cheap all-eligible scores. That dependence belongs to the control plane. It is not proof of zero influence from records omitted by the value plane.

### 5.6 E-conditioned original M responses remain intact

\[
 m^E_{qj}=\Psi_E(q_0,z_j^E,\Delta x_{qj}),\quad
 q_E=q_0+A_Ec_E(q),\quad
 m^M_{qi}=\Psi_M(q_E,z_i^M,\Delta x_{qi}).
\]

Local M response and field heads receive q_E. Readout M scores use q_E; retrieval/seed scores use q_0. This preserves the non-circular ordering established in Round9. Compute selected E nonlinear values before reducing; no scene-prepared E average substitutes for them.

For current Thermal:

\[
 T(q;h)=\sum_i\left[g_i^M(q)k_i(q_E,z_i^M)+K_i^{\mathrm{local}}(q_E)\right]h_i.
\]

Gates and contextual states are heat-independent in this capability. Neither information softmax weights nor quadrature measures are inserted a second time into the signed source sum. Source i remains source i. Zero heat gives zero T; flow is heat-null. A nonzero thermal offset is not authorized for the current data. Wind retains the exact saved physical velocity conversion.

For future HET-MF, nonlinear voltage-to-temperature dependence must be an explicit case capability; never impose the Thermal heat whitelist or K h law on the generic core. This round only adds suitable interface boundaries/tests, not HET-MF training.

### 5.7 Conditional continuity statement and its limits

For fixed valid record inventory, if the input encoders, scores and value functions are continuous in q/X, the composed b, π, g, guarded reductions and native field are continuous. A value removed at g=0 contributes zero at that boundary. Several overlapping seeds may enter/leave entmax support without replacing an entire nonzero field branch discontinuously.

C1 behavior is expected where the constituent features are C1 and the selected sparse primitive satisfies its differentiability assumptions. Do **not** promise blanket C2 or arbitrary topology/count derivatives. Native cell sampling, physical interfaces, min-distance feature ties, geometry rasterization and module insertion remain separate boundaries. Thermal continuous-core seams and its cell-center/curl API must still be evaluated separately.

There is no straight-through estimator in the primary continuous model. Its training and inference use the same continuous function and exact autograd derivative. At a closed C1 gate, the true derivative is zero; dense evaluation does not magically restore a nonzero omitted-value gradient. Use near-open initialization, gradual work pressure and shared trainable thresholds, with an explicit recovery test showing that a temporarily closed route can reopen under shared parameter changes. Never advertise an all-closed independent gate as recoverable by a zero exact derivative.

---

## 6. Continuous direct selector and fair comparisons

**C10-H:** continuous typed directory b, sparse continuous seed weights π, final native access g, legacy dense contextualization.

**C10-S:** the same reader, same normalizer, same continuous gate ψ, same smooth local M envelope and same pre-E raw score inputs, but direct query-to-M/E calibrated gates without a seed directory. No best-donor or nearest-E hard floor.

**Dense mathematical control:** all gates open, original dense M+E value function. Qualify it against D9-ME at identical common weights and native inputs. When all-open equivalence is established, Track A's D9-ME continuation is the matched trained dense baseline; do not train an unnecessary duplicate. If equivalence fails because of an intended value-function change, stop that change and repair the design rather than silently using an unmatched dense baseline.

Common all-open parity is toleranced numerical equivalence, not bitwise equality of every alternative reduction. The prior strict Thermal one-step miss stays in history.

Hypergraph-versus-direct conclusions require reporting actual M and E work separately, native fields and learned parameter counts. Equal target fractions do not guarantee equal realized work. Use trained curves and same-weight exact-count controls; only make work-matched grouping claims where both typed retention rates overlap within five percentage points, or on a prospectively specified matched-work interval. A parameter-count difference of tens of scalars is still disclosed.

Do not force top-K evaluation or count-match by masking unadapted weights and call it an independently trained comparison. A post-training intervention has a different evidential meaning.

---

## 7. Track-B learning protocol and actual-work control

### 7.1 Shared parents and native exposure

Branch C10-H/S separately from the exact shared dense M+E e3000 checkpoints listed in §3. Copy common physical weights and named Adam states. New route/calibration parameters and new work-dual states start fresh and are declared. Do not use H9/S9 e3500 learned routing weights as a hidden privileged initialization.

Train each through e5000: **2,000 complete new epochs**. Native provider membership, normalization, losses, queries, LR=3e-4, weight decay=1e-5, effective/microbatch 48/8 and 16/8, and response calls remain matched to Track A. New route parameters have an explicitly declared optimizer group; use the inherited learning rate and no weight decay on scalar/head calibration. Do not alter the original common-weight decay.

### 7.2 Schedule: continuous throughout, not a deterministic phase omitted by a stop

Let r=e−3000. In every phase, forward uses the deployed continuous function with no logistic/Gumbel sampling and no hard-forward estimator.

| Sparse age | Purpose | M removable target | E removable target |
|---|---|---:|---:|
| r1–200 | Near-open adaptation; ramp work pressure from zero | 0.98→0.95 | 0.98→0.90 |
| r201–1000 | Gradual learning of exact-zero access | 0.95→0.85 | 0.90→0.60 |
| r1001–2000 | Mature fixed policy/targets | 0.85 | 0.60 |

Use fixed gate shape ψ. The entmax temperature is input-calibrated once and fixed throughout the scientific trajectory; do not anneal it toward argmax. On the fixed TRAIN input panel, evaluate the four narrow-score-only candidates {0.5, 1, 2, 4}; select the one whose median positive seed count is closest to four, breaking ties by proximity to temperature 1. This is an initialization heuristic, not a fixed seed count or an inference cap: later learned scores can change participation freely. Record complete counts, seed types and entropy, including cases where the heuristic cannot achieve its nominal target. One calibration value per dataset is sealed before target-based evaluation; no field error selects it.

No extra output seam-loss or teacher-distillation objective is included in the primary comparison. Architectural continuity should be established independently of a seam penalty. Native field/reference losses remain the learning task.

### 7.3 Counts and surrogate work are different

For each type, the actual nonlinear work is

\[
 C_t=\sum_{q,i\in t}\mathbf1[g_i(q)>0].
\]

For differentiable pressure use the labelled support surrogate

\[
 \widetilde C_t=\sum_{q,i\in t}\eta_\delta(g_i(q)),\quad
 \eta_\delta(g)=\frac{(1+\delta)g}{g+\delta}.
\]

δ decays from 10^-2 to 10^-4 over r1–1000 and stays fixed thereafter. This is **not an exact expected count**. Log C and C-tilde together, including how many positive gates are below 10^-4, 10^-3, 10^-2 and 0.1. A low gate-mass surrogate with dense actual support is a failed sparsity result.

M's compulsory count is the positive original smooth-local envelope support; E has no artificial nearest floor. The source inventory and type-specific degenerate cases determine the denominator. Do not reuse Round9's mandatory best-partner floor formula. Whenever the compulsory local/eligibility floor equals the possible count, including locally covered singleton receivers, assign zero pruning residual rather than an impossible target. A distant singleton query without that local floor is not automatically protected by this rule; its necessary thermal response must be learned and explicitly checked.

Aggregate native request/auxiliary rows within case, then across cases with the declared equal-case macro normalization. Use observed actual counts for controller feedback, and the smooth proxy only for gradient pressure. Reuse typed dual arithmetic but create a new schema for its changed residual semantics. Calibrate positive initial weights on fixed TRAIN task/proxy gradients; explicitly handle unsupported zero gradient norms instead of silently using zero lambda.

Use one declared typed work controller per family. Do not open an unrestricted threshold search. The reporting target is a useful fidelity–actual-work tradeoff, not merely matching λ or a surrogate count. Training survives an ordinary missed work target and reports the miss.

### 7.4 Reviews and maturity

Native reviews at e3200/e3500/e4000/e4500/e5000; checkpoints every 100 epochs. The e3500 review is diagnostic, not another default stop. Run all 2,000 epochs unless an explicit catastrophic/nonfinite/resource rule fires. A few optimizer updates qualify software only; they do not establish scientific learning.

Record source/checkpoint, role, M stratum, native scales, positive support, attention/gate mass, E-conditioning norms, pre/post query differences, raw/gated K, realized work and saved-tensor memory. Track dense/read/source-prep counters independently.

---

## 8. Unified organization from local records through preparation and decoding

### 8.1 What “unified” means here

Use one typed incidence **policy and interface** throughout staged contextualization and decoder retrieval. This does not require a numerically identical matrix at every depth: its inputs and actual physical record states evolve. Every exported incidence carries its real stage, source state/version, original IDs and application role.

For a contextual stage with destination s and donor i, construct continuous b^l_si from that stage's actual raw typed scores **before** computing its value update. Apply b to the measure-aware donor reduction. Retain original record axes, projected per-record values and residual local state. No learned hyperedge latent replaces the value channel.

The original stage order remains MM→EM(new M)→EE(new E)→ME(new E), twice. The same route-specific gate policy parameters are shared across depths initially; the attention projections remain the original stage-specific ones. The final directory reuses the last cycle's actual applied incidences, with retrieval self-membership distinguished from contextual self-exclusion. Do not claim those four sequential stages are one simultaneous physical adjacency.

An implementation that still runs all four dense updates, then exports thresholded scores afterward, has **not** implemented unified preparation.

### 8.2 Local-input capability: make hidden aggregations explicit

Create a new input capability, `local_records_context_v1`, for the unified triplet. Existing adapters and checkpoints keep their old capability.

- Each M encoder receives its own features, position/length, validity and externally supplied operating context.
- Each E encoder receives its own prescribed physical/geometric features and the same external context.
- No learned global pooling/broadcast of current M/E latent values is injected between cycles.
- In Thermal, remove the layout-derived E nearest-module and occupancy channels from value inputs (zero their reserved slots only in this explicitly new transform). Remove count/density operating channels from the neural value stream. Keep frame, prescribed inlet/Re and legitimate case-global material constants.
- In Wind, remove `turbine_count_fraction` from the value input in this new capability. Retain prescribed wind direction, reference speed and the actual domain/support geometry. Do not change coordinate frames or secretly rederive a different target domain.
- Physical source masks, frame metadata, quadrature and row counts still exist for execution and normalization; they are not learned hidden embeddings.
- The existing Thermal radius/material inventory is case-global. Future per-module properties must remain local rather than being pooled into operating context.

Preserve physical reconstruction of input geometry: module occupancy still enters the true physical problem and native target extraction. Only its unrecorded aggregation into E latent **value inputs** is removed in this new model comparison. This is not removal of solids from the dataset.

A changed local-input transform is a scientific change. Give it its own schema/hash and field description. Preserve TRAIN output scales and physical units; do not refit them to compensate. Export the dropped/retained feature names, not just positional indices.

### 8.3 Global summaries and skip paths

Disable construction and application of `scene_summary_net` outputs, `module_scene_projection`, `environment_scene_projection` and their broadcast gates in the unified capability. Do not calculate a full summary and multiply by zero while claiming its memory is gone.

Own-record raw residual skips and FFNs remain valid local paths. Operating context is a declared exogenous path. No global neural residual, parallel dense predictor, dense all-source correction, or target-conditioned port is added to repair a poor fit.

Initialization copies every compatible common tensor from dense M+E e3000. Dropped summary parameters/optimizer entries are explicitly listed; they are not called “missing keys” to be ignored generally. U-D/H/S share the same new input transform and same common initialization. Parent-function equality is **not expected** after deliberately removing nonzero aggregate paths; qualify the new all-open U reference on its own equations.

### 8.4 Unified controls

| Arm | Local-input capability / summary broadcasts | Contextual exchanges | Query retrieval |
|---|---|---|---|
| U10-D | New local inputs; no learned global summary | All-open native typed exchange | All-open E/M |
| U10-H | Same | Continuous typed incidence controls values | Reuse native incidence with continuous seed access |
| U10-S | Same | Same contextual incidence policy/parameterization as U-H | Independent continuous direct query gates |

U-H and U-S share the upstream architecture and initialization, not their independently evolved trained weights. This controls the query organization mechanism without giving S an unrelated dense preparation path. U-D measures the effect of the local-input/no-summary architecture without forced sparsity. It is a necessary control: comparing U-H only to the legacy dense decoder cannot isolate the consequences of removing aggregate paths.

Keep two contextual cycles; do not add a capacity expansion campaign. Measure whether the E-conditioned query vector becomes a bottleneck (conditioning norm, small-panel Jacobian rank/sensitivity and native errors), but do not simultaneously add multiple edge latents or triple-product values.

### 8.5 Training and automatic admission

Implement and test the unified capability regardless of whether it ultimately trains. Thermal U-D/H/S train from the shared e3000 boundary through e5000 if:

1. source/target isolation, continuous-value/gradient, native-law and fresh-state tests pass;
2. Track-B Thermal C-H completes its horizon without catastrophic collapse (no priority selected-TRAIN mean persistently above 2× same-age dense M over two 100-epoch windows; ordinary +10% misses are not this rule);
3. the new U all-open reference and a complete native epoch fit within the reserved hardware and remaining matched-triplet budget.

These are safety/feasibility conditions, not a requirement to beat S, attain memory savings or win DEV accuracy. On satisfaction, proceed automatically to the whole Thermal triplet. On failure, retain the implementation/diagnosis and continue the remaining already-admitted tracks; do not demand another conversation to interpret the gate.

For Wind, after its Track-B endpoint, use the same tests and catastrophic threshold plus a reserved complete-triplet forecast. The Wind U triplet has lower scheduling priority than completing Thermal U and all Track-A/B runs. No individual promising U arm receives a full trajectory while its controls receive only smokes.

U training uses the same 2,000-epoch query schedule and native objective. Begin its contextual gates near open; ramp **context-edge** retained-support target from .98 to .90 over the first 1,000 ages, then hold .90. Decoder targets remain .85 M/.60 E above its actual local floors. Context-edge work is reported separately from decoder value rows; a dropped attention edge is not a dropped nonlinear query-token MLP row. Share the contextual gate parameters between depths initially. No unannounced per-layer threshold search.

### 8.6 Shortcut tests

On a small synthetic native scene, construct two value-plane components and freeze the entire execution plan including normalizers/continuous coefficients. Perturb a raw record in component A and verify component B's values do not change unless a declared value path connects them. Rebuild the same test with ordinary live routing to show separately that score/control dependencies may propagate a change. Do not freeze only final binary masks and then assume live normalizers are frozen too.

Run this at encoder outputs, each MM/EM/EE/ME stage, prepared records, E context, M messages and final field. Poison target containers and unavailable reference metadata. Test that preparation is independent of the set/order/chunking of requested decoder receivers.

For actual legal layout changes, rebuild all legitimate geometry-derived metadata and contextual states. Export both the last-layer read set and the unrolled ancestor paths. These paths describe the learned computation; independent physical causality still needs compatible physical interventions.

---

## 9. Continuous sparse execution and gradient correctness

A dense mathematical reference computes every eligible original value, applies continuous g and the declared normalizer, and reduces. The packed executor computes only g>0 values, deduplicated by (batch, native query ID, type, native record ID). Cheap dense scores/control-plane work remain charged.

At g=0, the proposed C1 gate has zero first derivative. This permits the selected value path to match the dense first derivative at a support birth/death, provided the gate/entmax/normalizer derivatives themselves are retained. Verify this; do not infer it from equal forwards.

Do not cache just integer supports for geometry autodiff and omit the derivative through live gate coefficients. Conversely, Track-A hard-router derivatives must remain labelled fixed-route; they do not become full derivatives by being evaluated through the new API.

Required numerical fixtures:

- independent FP64 literal native loops for E then M; values, information weights, continuous coefficients and native K;
- dense/packed/checkpointed forward and parameter/input VJP/JVP agreement;
- a gate crossing zero, several entmax seeds tying/entering support, a local envelope boundary, and an empty type;
- duplicates, source permutations, typed-ID collisions, query-alone/chunked/mixed-batch/repeated calls;
- a physical M source with zero information measure; an E record with zero measure; absent padding with poisoned values;
- simultaneous E and M support changes; full source-role fields, not just latent messages;
- all-open recovery, zero-E diagnostic and exact restoration under declared numerical envelopes;
- AMP/TF32 changes are not used to repair memory; retain FP32/TF32-off scientific fitting and FP64 reference fixtures;
- non-reentrant checkpoint recomputation follows the identical gate coefficients/indices and charges actual recompute work without duplicate logical progress updates.

Pin the installed framework version. Official current checkpointing documentation is implementation guidance, not permission to upgrade the user's environment mid-campaign [E2,E3].

---

## 10. Seam and design-sensitivity evaluation: a primary result

### 10.1 Reuse every old seam

Read the saved e3500 H/S seam catalogue and test **all** new candidates and all four mature old arms at those same physical coordinates. Separately search a fixed number of new candidate support transitions using inputs/scores only. Include quiet ordinary regions and material/domain boundaries as labelled controls.

Measure maximum/RMS per-field increments J(ε)=|f(q+εd)−f(q−εd)| for a geometric ε sequence spanning at least six half-steps until native resolution/FP precision becomes limiting. Plot increments and J(ε)/(2ε); include output-normalized and absolute units. A discontinuity has a finite plateau as ε shrinks. A continuous but excessively steep transition can have a large derivative and is not a complete inverse-readiness success.

The continuity acceptance statement is: no unexplained finite jump plateau above the numerical floor at tested ordinary-coordinate support transitions, with shrinking increments agreeing with the dense reference of the **same continuous model**. Do not replace this with a single arbitrary maximum-jump threshold. Use analytic primitive tests to establish the function's conditional continuity; use the physical cases to assess its practical gradients and conditioning.

Thermal's arbitrary continuous-core evaluator, literal cell-center interpolation/curl, solid boundaries and grid snapping are different scopes. Do not compare a snap-to-same-cell pair and call it proof of continuous native geometry. Preserve the current Wind bounds-only query-feature repair and test it with actual native scene types.

### 10.2 Geometry sensitivity

Test legal module translations with fixed domain/frame and unchanged module IDs. Recompute local environment features according to the active capability; do not use stale distances, support indices or K caches. Compare live continuous autodiff with finite differences away from physical raster/mask boundaries, and label crossing those physical boundaries separately.

Report the magnitude and pattern of derivatives, not merely a finite gradient. A near-zero gradient can mean a dead model. An extreme finite gradient can still make inverse optimization unusable.

Preserve field/response outputs per source and distinguish changing source amplitude, changing geometry and changing independent E inputs. Existing Thermal heat amplitudes have a benchmark-specific affine dependence. Heat–heat mixed derivatives must remain zero at fixed geometry; geometry–geometry or geometry–environment effects need not.

---

## 11. Predictor and organizer evidence boards

### 11.1 Native fields

At literal e4000/e4500/e5000 (and the new families' e3500 checkpoint for learning comparison), evaluate the same native panels used at the e3500 stop. Primary final boards cover selected TRAIN150/72 and exposed DEV22/24. Keep all physical roles, masks, common receiver hashes, absolute errors, mean/p90/worst IDs and populated M/layout strata.

Use the existing reporting bands: priority mean within +10%, priority p90 within +15%, secondary mean within +10%, and investigate populated priority strata above +25%. Report against **both same-age legacy dense M and dense M+E**, and U-H/U-S additionally against U-D. These bands label results; ordinary misses do not erase a completed trajectory.

Keep the Thermal small-M TRAIN distinction; there is no independent M1/M2 DEV claim. Wind direction rows sharing one layout remain correlated. No formal TEST metrics or population-confidence claims from these exposed development panels.

### 11.2 E use versus hypergraph use

Repeat zero-E-context and zero-E-value interventions with fixed same-model state/coefficient plans. Check actual E nonlinear message hooks, E-conditioned M/local responses, per-field output changes, reference errors and restoration. Ensure an empty E branch is not advertised as a sparse E success.

Compare H against trained S, original-score and geometric final-count controls, type-isolated rewiring and fixed-routing/native-value perturbations. Report final per-type work after every intervention; degree-preserving directory edits can change the query union and are not automatically work-matched.

The continuous case requires matching coefficient **and** support semantics. Replacing a live continuous coefficient with 1 is a different intervention from removing a source. Separate membership changes, seed weights, source-value changes and physical input changes. Never conflate gate mass with executed reads.

### 11.3 Adaptation and high-order evidence

Use the existing legal layout-shift panel, same IDs and fresh preparation. Measure incidence coefficients, positive sets, entmax seed mass, final native reads and output change. Count-only invariance does not mean coefficient/path invariance.

For a saved physical quartet:

\[
 I_{AB}=F(X+A+B)-F(X+A)-F(X+B)+F(X).
\]

Compare each model's quartet residual with the compatible saved physical residual using common-valid receivers and original units. Also show endpoint errors and single-intervention deltas; mixed residuals can amplify endpoint errors. Treat correlations of learned co-retrieval with physical interaction magnitude as descriptive, not unique graph recovery.

Reuse only reference states that actually exist and whose units/layout/heat/frame/source binding match. Do not fill missing factorial corners with neural predictions, rerun an unresolved historical generator silently, or mix HET-MF records into Thermal's reference ledger.

Independent environmental-forcing claims remain unavailable on these legacy datasets. A validated, separately approved HET-MF pilot may support future input/schema tests, but this plan does not authorize bulk HET-MF data or a training campaign on an unfinished pilot.

---

## 12. Design-use diagnostic without an arbitrary memory prerequisite

Remove the old requirement that an offline saved-candidate ranking can run only after a 20% memory saving. That condition confounds a scientific decision-use question with an engineering result. Preserve the old unqualified status; create a separately authorized Round10 diagnostic.

Inventory the existing six-candidate/single-M3-layout collection. Verify complete compatible physical candidate references and the objective/constraint definition before evaluating anything. If compatibility or reference coverage is insufficient, report precisely which candidate/quantity is missing and do not invent a ranking ground truth.

When the catalogue is usable, evaluate all matched models at literal endpoints. Report objective-value error, rank agreement, reference regret of the selected candidate, feasibility/constraint classification, inference work and whether the objective was exposed during training. Report individual candidates; a six-item single-layout result is not generative design success.

The diagnostic can run on a model with poor memory or a hard seam because it does not use that model's geometry gradient. Label such limitations. Continuous models also get bounded gradient/finite-difference design-sensitivity tests, but no new geometry optimization with unvalidated physical reference is launched. No claim of valid generated designs, diversity, or physically verified improvement follows from surrogate-only optimization.

---

## 13. Software boundaries and implementation work packages

### W0 — short source/reservation audit

Verify current branch ancestry, user edits, checkpoint hashes, source snapshots, feature/numerical identities, disk mount, GPU UUIDs and actual process occupancy. Record that Round9 is closed. Resolve source snapshots and reference artifacts through their existing index rather than rebuilding giant repository inventories repeatedly.

Deliver `docs/reports/HONF_Round10_Initial_Audit.md` with report-derived facts, static source findings, new hypotheses and exact execution plan. Keep this a focused source map, not another sprawling report before learning begins.

### W1 — mature continuation adapter

Add `tools/honf_round10_continue.py` and a small versioned continuation/execution contract under `honf_runtime`. Restore the inherited scientific state into new output directories. Integrate with the existing TrainingEngine; do not write a second training loop. Regression-test e3500→next-update resume against the frozen native engine, including partial macros and dual/cadence windows.

Do not edit original e3500 receipts, regenerate their hashes, disable GPU/mount checks, or treat inference exports as resumable state. Launch Track A once qualified; development of new models can continue in a separate worktree while those sources remain frozen.

### W2 — tiled typed executor and memory diagnostics

Refactor reusable primitives into pure score/normalization/value/reduction functions with compact trace contracts. Implement full-pipeline checkpointing/streaming and reduced work-observation lifetimes. Qualify same-weight old-function replay without mutating the old scientific implementation. Implement S phase-separated memory measurement alongside H.

### W3 — continuous routing primitives

Suggested package `src/honf_forward_core/continuous_joint/`:

```
contracts.py       # typed records, stage-bound coefficients, compact traces
primitives.py      # positive-quadratic gate, guarded reduction, verified entmax
routing.py         # seed mass, continuous direct gates, local envelope blend
incidence.py       # typed per-stage b coefficients and native correspondence
reader.py          # E-first native values; tile/dense/packed references
model.py           # C-H/C-S; all-open dense-equivalence capability
budget.py          # actual support counts vs explicit smooth proxy
```

These names are proposed new files, not claims that they exist. Reuse proven native geometry/physics primitives instead of copying thousands of lines. Do not inherit Round8/9's epoch assumptions or no-E schema by stacking another accidental subclass workaround. Declare a standalone schema such as `continuous_joint_native_v1` with explicit capability/config objects.

### W4 — runtime and coherent training

Add `continuous_joint_{protocol,factory,runtime}.py` or similarly focused modules under `honf_runtime`, plus `tools/continuous_joint_train.py`. Use the existing providers/TrainingEngine. Keep dataset sampling identity independent of new model-family identity so the matched query stream remains unchanged.

Provide plan/prepare/fit/resume/status/export commands; CPU preparation must not allocate GPU or perform optimization. An admitted fit performs complete declared epochs. State includes route parameters, all clocks/controller state, common optimizer, transforms and feature capability. Data-free inference export must not require private ancestor checkpoints or datasets.

### W5 — unified preparation capability

Implement `continuous_joint/preparation.py` and case-local feature-capability adapters. Introduce a small tested hook before typed contextual value reduction, with the old all-open path retaining legacy defaults. Remove latent summary paths and masked feature channels only when the unified capability is selected.

Build U-D/H/S from one common initialization inventory. Export a stage-by-stage value/control provenance map. Sparse contextual execution must perform actual selected reductions, but does not need to outperform SDPA; retain SDPA as the fair all-open implementation. Do not materialize huge pair-value tensors merely to make the method look more hypergraphic.

### W6 — native evaluation, seams, references and automatic gates

Reuse existing physical board and report consumers with explicit new model-family dispatch. Add continuous-coefficient traces, saved-tensor memory scopes and continuous versus physical-grid boundary tests. Read and fix the existing dataset APIs rather than inventing a scene `.support` member. Preserve the previous Wind seam repair.

A campaign driver executes the DAG in §14 and writes a decision receipt at each transition. It does not ask the user whether to proceed when the declared gate is already met. No successful completed audit needs to be rerun after a cosmetic report fix.

### W7 — final report, export, preservation and push

Deliver one integrated report with separate Track-A/B/C conclusions and one compact evidence index. Include positive and negative results, actual update/epoch counts, source/endpoint hashes, resource accounting, missing gates and the final recommended representation. Push only this task's durable files; verify remote readback. Never force-push or reset user changes.

---

## 14. Integrated execution DAG and decision rules

1. **W0/W1:** verify/pin and start mature Thermal controls, then mature Wind controls under new resource reservations. Both cohorts are required; lack of speed or present sparsity is irrelevant to launch.
2. **W2/W3:** profile e3500, implement/qualify compact pipeline and continuous primitives. A failed primitive/numerical test blocks the new family, not the already-qualified old maturation.
3. **W4:** run C-H/C-S on Thermal and Wind, each through its full 2,000 epochs. A correct but not-faster executor is admissible. No r500 decision stop is substituted.
4. **W5:** implement unified primitives and input/provenance audit in parallel with these runs, in an isolated source closure. Run the Thermal U triplet when §8.5 passes; run the Wind U triplet when its gate and complete-triplet reservation pass.
5. **W6:** final same-age physical/continuity/work/memory/control boards; saved-reference and compatible ranking diagnostics. Do not postpone them pending a memory win.
6. **W7:** finish the integrated report and explicit decision, whether the organizer wins, loses or remains work-confounded. Never replace a negative result with another unplanned architecture campaign.

### Outcome branches

| Observed outcome | Automatic action within this campaign | Honest conclusion |
|---|---|---|
| Mature hard H becomes sparse but remains discontinuous | Complete its boards; proceed with continuous revision | Maturity and continuity are different problems |
| Continuous H/S remove seams but remain dense | Finish learning and work/memory boards; run admitted U comparison | Continuity improved; sparse usefulness not established |
| H and S both improve at similar work | Compare U, physical responses and ranking; retain S | E-conditioned sparse access may be useful without unique grouping gain |
| H is worse than S outside work overlap | Do not force a hypergraph win; report the Pareto mismatch | Grouping attribution unresolved/negative |
| U loses important fields after removing aggregates | Retain U-D comparison and stage/provenance evidence | Removing a useful context path can cost fidelity; no universal shortcut-free claim |
| U improves or preserves fields and closes value bypasses | Complete native response/memory controls and export it | Unified executed organization is supported within tested capability |
| Resources cannot reserve a complete conditional U triplet | Do not launch only H; complete required tracks and report unexecuted branch | Conditional scope incomplete, not a model failure |
| Numerical correctness cannot be qualified | Preserve failure, continue unrelated admitted tracks | No scientific result for that function |

No full-data training or new solver follows automatically from any row. Those would be separate scientific scope, unlike the steps already included here.

---

## 15. Acceptance tests, including cases likely to hide defects

### Physical/native contracts

Test every native role at all-open and nontrivial positive/zero support; Thermal zero/scaled/summed heat, heat-null flow, signed K and exported contraction; Wind original profile/residual units; no extra source measure or output mean; precise versus ordinary FP32 algebra distinguished. Include M1/M2 and high-M, zero-measure M and E, empty E, invalid padding and source permutation.

### Continuous mathematics

Independent entmax reference; tie symmetry; π normalization; ψ value/derivative at zero; g in [0,1]; protected smooth-envelope formula; positive guarded denominator; all-open equality and zero-support zero-message limit; native source deduplication across multiple seed paths; finite-difference/analytic parameter and coordinate gradients across support entry/exit. Run separate tests for forward compactness and backward correctness.

### No hidden fallback or value shortcut

Counter hooks demonstrate E and M original values before reduction; sparse support executes no excluded wide values in the primary forward. Dense reference/recomputation is separately counted. A profile/trace must reveal any fallback. U contains no live scene-summary broadcast; own-record skips remain. Frozen full-plan component tests and live-control-plane tests are distinct.

### Memory and persistence

Saved-tensor storage identity/shape inventory, repeated native batches with weak-reference release, at least 40 complete engineering epochs, donor/query tile invariance, checkpointed/uncheckpointed gradients, no metric-graph retention, no full trace buffer in ordinary inference, graph/store rejection after backward consumption, owner/version/frame/normalizer checks, atomic resumed checkpoints.

### Training identity

Resume before/after a partial macro and typed dual update; recursive tensor/Python clocks agree; native cadence and case/query hashes match controls; same shared parent, optimizer group mappings and dropped U parameters fully enumerated; numeric order differences not renamed bitwise equivalence. Count failed/replayed steps separately from successful epoch progress.

### Evaluation semantics

Selected TRAIN versus exposed DEV versus locked TEST; actual provider receiver-role quotas; physical units and whole valid panels; no double interpolation/rotation; saved quartets only when all corners match; no parent-teacher predictions as truth; conditional ranking without memory gate; exact final typed counts for controls.

A single scalar “all tests passed” is not the final report. Identify the named categories and every unresolved strict/numerical/scientific miss. Source-appropriate declared tolerances are fixed before measurement; do not relax a failed tolerance retrospectively.

---

## 16. Resources, coexistence and data2 layout

### 16.1 Proposed new envelope

This plan proposes, for its enumerated campaign, **72 GPU-associated process-hours**, **96 wall-clock hours**, with **8 GPU-associated hours reserved for final evaluation/closeout**. This is a new prospective envelope, not a claim that the user already reserved specific devices or that old unused Round9 time can be reused. When executing this plan, seal actual start/devices/caps and forecast complete tracks before launch. User-specified resource settings take precedence.

Cap new main scientific updates at 144,000, discarded engineering updates at 2,048, retained payload at 64 GiB, temporary at 16 GiB, peak new payload at 80 GiB, and free `/data2` headroom at 100 GiB. These bounds are not a target to exhaust. Store only required checkpoints, compact physical arrays, diagnostics and figures, not per-epoch full activation dumps.

GPU1 is the preferred HONF lane, following the reported device priority. Resolve its current UUID and identity. GPU2 is usable only if the user's active reservation allows it **and** it is idle/unreserved by HET-MF or other work. Never infer availability from the old “GPU1 released” report, move another task, send it a signal, or start a competing job on a reserved device. Do not use GPU0/other hardware implicitly.

Use one job per reserved GPU. Parallel jobs share an atomic **short-held** ledger transaction for budget reservation; do not hold a campaign-global lock for the lifetime of a process and unintentionally serialize different GPUs. Each device has an exclusive task lock. Charge summed process time across devices as device-associated time and wall time separately. No claim of kernel-active time without measurement.

Maintain at least max(8 GiB, 20% device capacity) free, plus the measured preventive allocation margin (initially 6 GiB for these native jobs). Derive per-process allocator limits from the actual device and live occupancy, not a blind constant. CPU planning/rendering must keep CUDA hidden. Record monitor gaps honestly.

### 16.2 Forecast before, not instead of, learning

Measure complete native epochs for the proposed execution modes and batch settings. Forecast all required Track-A/B runs plus the admitted complete U triplets, with monitoring, checkpoint I/O and conservative overhead. The report's old 6.766 associated hours for 27,000 updates is context only; different new executors/preparation costs require measured forecasts.

Prioritize complete matched sets. Do not shrink the data, query budget, physical objective or horizon to make a forecast fit. Use permitted tiling/recomputation, wait for a reserved lane, or record a resumable resource stop. “More resources” is not unlimited permission; reaching the cap remains real.

### 16.3 Directories and source isolation

Proposed payload root:

```
/data2/wanglz/ModularDT/diagnostics/continuous_joint_round10_<UTC>/
  receipts/       # exact sources, reservations, transition decisions, ledgers
  recipes/        # immutable scientific/execution declarations
  runs/
    mature_round9/<dataset>/<arm>/
    continuous_read/<dataset>/<H|S>/
    unified/<dataset>/<D|H|S>/
  arrays/         # unique native/reference diagnostics, not dataset copies
  figures/        # selected PDF/PNG, caption/array bindings
  exports/        # inference-only standalone packages
  source_closures/
  tmp/
```

Expose it with a verified ignored symlink beneath `HONF_Proj/diagnostics/generated/`. All large/generated data, compilation/test caches and temporary figure payloads stay on `/data2`; code, small configs, reports and schemas are durable Git files. Set cache/temp variables before imports where necessary.

Use separate worktrees or immutable source closures for frozen mature fitting, new continuous development and unrelated HET-MF work. Do not change the shared active branch out from under a running process. Verify ancestry at start and reconcile only own changes at final push; preserve all later user edits.

Do not copy legacy datasets. Reuse exact logical dataset and manifest identities. Use safe symlink replacement only for owned links; a real directory at the intended link path is not disposable. Cleanup deletes only exact own-run allowlists after dependency/open-handle checks and protects every checkpoint, original failure and user file. Avoid spending substantial research time auditing unrelated entire disks repeatedly.

---

## 17. Report and visualization contract

Deliver:

- `HONF_Round10_Continuous_Unified_Organization_Report.md`;
- one evidence index with source/checkpoint/manifest hashes and exact completed horizons;
- tested CLI examples and a short reusable-core guide;
- a model-card style summary distinguishing legacy physics capabilities, continuous-read mode and unified local-input mode;
- stand-alone exports for qualified literal endpoints, including all controls needed to reproduce conclusions.

### Essential figure groups

1. **Maturity:** all eight old arms e2500–5000, shared ancestry, native losses, typed realized work, stochastic/relaxed versus deterministic support and duals. Do not attach e5000 cost to another checkpoint's best error.
2. **Continuity:** old and new fields on the same saved seam brackets; log-scale jump-versus-step and derivatives, numerical floors and physical-grid limits labelled.
3. **Unified information flow:** compact stage-by-stage diagram separating local/raw/exogenous paths, routing scores, applied native incidences and E-conditioned M values; show what the old global broadcast did and where it was removed.
4. **Actual organization:** typed b coefficients and exact positive sets, multiple seed weights for named receivers, final deduplicated M/E values, local envelopes and true executed-row hooks. Show dense/unhelpful cases too.
5. **Fidelity and physical response:** all role summaries/tails/strata, representative spatial fields, mixed saved physical quartets and endpoint-vs-response errors.
6. **Memory:** measured live-byte attribution and saved-tensor shapes; fair optimized dense versus selected step and full actual cycle; S included. Show number of positive reads beside memory, not as a substitute.
7. **Decision use:** small compatible saved-candidate table/plot, selected candidate regret and constraint status, explicitly not generative-design success.

Keep figures readable: a few named original records and queries, shared field/error scales, clear units, captions that say whether arrays are reference, model or control. No generic graph decoration, causal-edge labels, selected cherry-picked success only, or more pages than necessary. A target of 12–18 useful figure pages is preferable to reproducing every receipt as an image.

Use saved numeric arrays for rerendering. Review every retained page for clipping, mismatched IDs, omitted units and inconsistent color scales. Report original failed numerical/owner attempts separately from later successful repairs. A formatting fix never authorizes another GPU experiment.

### Final decision must answer

- Did the old hard organizer change after full maturity, and at what actual M/E work?
- Did the continuous family remove routing jump plateaus without simply becoming dense or hiding full values?
- What allocations actually caused the memory overhead, and which were measurably removed?
- Does H add value beyond trained direct selection at overlapping work?
- Does unified preparation preserve useful fields while replacing unrecorded value aggregation with applied typed paths?
- Which physical responses and saved design choices improve or fail?
- Which evidence still cannot establish independent M–E physics on the legacy datasets?

A scientifically useful outcome can retain S as the strongest predictor and H as an unproven representation. Do not redefine success after seeing the result.

---

## 18. Explicit stop-list

Do not:

- call e3500 or 500 router epochs the completed original experiment;
- merge new continuous outputs into the old H9/S9 evidence;
- substitute one/two epochs, dry-runs or engineering probes for the prescribed learning horizons;
- apply softmax everywhere, compute every value, then call it sparse;
- use top-K/argmax/nearest or hard mandatory winner switches inside the purported continuous path;
- omit positive tiny gates from work accounting or treat gate mass as row count;
- detach routing/E context/common preparation to create a memory improvement;
- claim a saved-tensor cause before profiling or label shape arithmetic as a measured allocation;
- reuse the pre-E M score as the post-E M score;
- reinstate a global pooled neural shortcut to repair U without its own scientific amendment;
- infer original-input independence from sparse final values;
- infer physical causality from E ablation or graph rewiring;
- treat the current Thermal heat-affine rule as a universal HONF assumption;
- fabricate missing counterfactual corners, rebuild an unresolved historical solver silently, or unlock TEST;
- require a speed/memory win before a bounded offline saved-candidate ranking;
- intrude on the HET-MF GPU/data/branch reservation;
- disable resource validation or overwrite closed receipts/checkpoints;
- open an unbounded architecture portfolio when a comparison is negative.

---

## 19. Source map and external precedents

### Repository/report basis

All repository paths below were reviewed at `9e5fd74a02a80e9cd1bb21d74d57dde10bbd26b5`; use the current start audit to resolve later changes without resetting them.

- Primary uploaded report: *Joint native hypergraph decoder: matched e3500 decision stop*. Relevant sections: Predictor; Organizer; Physical responses; Exact training identity; Decision. Its own physical-reference limits are retained in this plan.
- `HONF_Proj/docs/reports/HONF_Joint_Native_Hypergraph_Round9_e3500_Evidence_Index.md`: checkpoint/parent hashes, architecture counts, source closures and historical repairs.
- `HONF_Proj/src/honf_forward_core/joint_native/{model,readers,routing,graph,budget}.py`: actual E-conditioning, hard gates/floors, score/value lifetimes, staged directory and work semantics.
- `HONF_Proj/src/honf_forward_core/content_hypergraph/{model_v2,staged_attention_v2}.py`: learned summary broadcasts, stage order, dense contextual update and native capture.
- `HONF_Proj/Case_WindFarm/src/windfarm/content_hypergraph.py`: actual prescribed feature and velocity-transform contract.
- `HONF_Proj/tools/joint_native_train.py`: original managed launch/resume/resource implementation. Other runtime modules must be read at implementation time before edits; their entire contents were not independently executed during this review.
- Thermal `native_observations.py::thermal_context_tensors`, re-read at the pinned commit: the count/density context and nearest-module/occupancy E feature construction. Verify current named channels again before implementing the new capability; do not rely on numeric offsets alone.
- `content_hypergraph/readers_v2.py::LocalDetailReaderV2.forward`: the existing compact window weights the local information sum, and its signed source support is applied during contraction; keep that distinction when integrating the continuous global reader.

For durable source links use the full pinned commit:
`https://github.com/cosmos2w/ModularDT/blob/9e5fd74a02a80e9cd1bb21d74d57dde10bbd26b5/<repository-relative-path>`.

### External primary sources: design guidance, not HONF evidence

**[E1]** Ben Peters, Vlad Niculae and André F. T. Martins, *Sparse Sequence-to-Sequence Models*, ACL 2019, pp. 1504–1519, DOI 10.18653/v1/P19-1146. `https://aclanthology.org/P19-1146/`. The α-entmax family supplies sparse probability mappings and gradient algorithms. It does not validate the proposed HONF incidence composition, physical interpretation, exact-zero work controller or field accuracy. Check the primitive's mathematical assumptions and exact masked implementation.

**[E2]** PyTorch, *torch.utils.checkpoint*, official documentation. `https://docs.pytorch.org/docs/stable/checkpoint.html`. Consult the documentation matching the **installed pinned version**. Checkpointed functions must reproduce compatible computation on recomputation; memory savings depend on region boundaries and retained inputs.

**[E3]** PyTorch Foundation, *Current and New Activation Checkpointing Techniques in PyTorch*, originally 5 March 2025. `https://pytorch.org/blog/activation-checkpointing-techniques/`. Relevant principle: saved inputs/outside-region intermediates and placement determine what is retained; this is not a claim that enabling a flag fixes the observed HONF memory peak.

**[E4]** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra and Christopher Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022. `https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html`. Relevant principle: exact computation can be tiled to reduce materialized intermediates; fewer nominal operations alone do not guarantee speed. Its specialized attention kernels do not directly implement HONF's nonlinear query–record values.

The continuous membership gate, guarded normalization, stage-unified capability and experiment decisions in this plan are **proposals derived for this project**. They are not presented as established results of these papers or as completed HONF evidence.
