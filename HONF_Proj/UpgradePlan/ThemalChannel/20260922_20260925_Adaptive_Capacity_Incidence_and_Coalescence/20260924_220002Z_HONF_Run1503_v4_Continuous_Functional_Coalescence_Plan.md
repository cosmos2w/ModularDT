# HONF 1503-v4: continuous functional coalescence and exact compact execution

## Decision and scientific objective

Retain Run 1505 as the first successful **case-dependent coalescence and reconstruction** experiment in this line. Preserve its exact e500 and distinct e473 temperature-selected checkpoints and all existing reports. Do not extend the unchanged candidate to 5,000 epochs.

Develop one new candidate around this rule:

> Remove a redundant interaction function only after its distinguishing contribution has continuously vanished. Packing equal functions must change storage and computation, not the predictor's value.

Keep the successful source-control quotient. Replace the online convex-clustering planner and its hard singleton-to-barycenter switch with a small, continuous, source-action-aware contraction of query-access functions. The new contraction is organized on one fixed proposal-function tree; its contraction amounts and compact class counts remain case- and phase-dependent.

Suggested architecture identifier: `continuous_functional_coalescence_honf`.
Suggested experiment name: `1503_v4_continuous_functional_coalescence`.
Let the existing allocator choose the next unused numeric run ID. Do not overwrite Run 1505 or assume an ID is available.

This is one architectural experiment, not a sweep. Keep the historical v2/v3 models replayable. Do not add an auxiliary field branch, a cardinality penalty, top-k source selection, a second learned router, or a new attention kernel in this round.

---

## 0. Evidence boundary and source-reading task

The assessment accompanying this plan read both uploaded Run-1505 reports in full. The available GitHub connection still returned branch state `aaee3fa9bcc2baf4760d34aebdd57c4344e14bc7`; the v3 launch commit `cb6ba95` and the v3 profile were not resolvable there. Therefore **the latest v3 source was not independently audited by the author of this plan**. Parent routing and ThermalChannel coupling source at `aaee3fa` were inspected. The v3-specific descriptions below are grounded in the reports, not an asserted line-by-line audit of inaccessible source.

Codex must read the actual local branch implementation before editing. Locate the existing classes and tests with ordinary repository search, rather than guessing their filenames:

```bash
rg -n "converged_identity_preserving_coalescence_honf|weighted_sparsemax|lambda_b|eps_abs" HONF_Proj/src HONF_Proj/tests
rg -n "port_refinement_head|_outside_coordinates|p1_refinement|p2_port_global_consistency" HONF_Proj/Case_ThermalChannel/src
```

Read in particular:

- `docs/reports/HONF_Run1503_v3_Converged_Identity_Preserving_Coalescence.md`;
- `docs/reports/HONF_Run1505_E500_Organizer_Diagnostics.md`;
- actual v3 planner, class extraction, access-function reconstruction, quotient routing, and source-moment preparation;
- parent `sparse_incidence_router.py`, `sparse_incidence_group_control.py`, and `group_control_pairwise.py`;
- `Case_ThermalChannel/src/channelthermal/interface_field_coupling.py` and the current loss assembly;
- existing quotient, mixed-batch, solver, boundary, and execution tests.

Report material mismatches between code and the reports. Do not assume a new implementation bug merely because a previous experiment underperformed.

### Evidence to carry forward

All figures below refer to the reports' existing development population, not an independent test set:

| Quantity | Run 1502 exact e500 | Run 1505 exact e500 |
|---|---:|---:|
| Pooled normalized fluid relative L2 | 0.10867 | 0.09789 |
| Pooled near-interface relative L2 | 0.10539 | 0.10192 |
| Pooled far-fluid relative L2 | 0.11829 | 0.09907 |
| Physical internal-temperature relative L2 | 0.04464 | 0.04185 |
| Physical surface-temperature relative L2 | 0.06018 | 0.06016 |
| Physical surface heat-flux relative L2 | 0.22239 | 0.22540 |
| Final environmental port-temperature relative L2 | 0.07657 | 0.09278 |

The e500 candidate improves fluid error in 72/90 cases but worsens final port temperature in 86/90 and flux in 60/90. P2 R=9/10/11/12 occurs in 4/53/28/5 cases, with all twelve provisional functions occupied. Mean R is 10.3778: a 13.52% reduction in access classes, not a fivefold reduction in physical work.

The exact e500/best-field/best-total weights coincide. Its e473 temperature-selected checkpoint is distinct: field/port relative L2 is 0.10226/0.06994. Comparing that with parent e500 is selection sensitivity, not a matched-epoch win. Do not mix e500 field accuracy with e473 port accuracy into a fictional single checkpoint.

At e500, 254 accepted merged phase partitions survived the specified strict numerical replays. Four case-phase plans fell back to singleton; two changed with a larger solver budget. A recorded trained boundary has a one-sided field jump of 0.03047 relative L2. The five-case matched executor panel gives a median complete-application ratio of 4.97 and preparation ratio of 6.70 versus the parent; this timing panel is not all 90 cases.

---

## 1. What is retained, and what is replaced

### Retain

1. The 1502 fine MM/ME/EM preparation, physical source coordinates/measures, and contextualized fine states.
2. K=12 provisional capacity, D=16 controls, environmental sparsemax and module entmax-1.5, the original query projection, and phase-local preparation.
3. Exactly three final context terms: `C_g + C_M + C_E`.
4. Multiplicity-correct query mass/density, compact source incidence `Abar`, and source-resolved moment `B`.
5. Original source-local environmental value modulation, provisional `kappa_0`, and the inherited module/environment output normalization.
6. One expensive interaction per physical query-source pair and one environmental softmax over the unique supported source union.
7. Historical rectangular execution as the reference and initial production policy. Retain the existing optional fused CSR implementation unchanged.

### Replace in the new architecture only

- FP64 ADMM in every prepared phase;
- tolerance-detected class identity as an approximation to equality;
- sudden replacement of singleton access functions by renormalized keys and recomputed-centroid functions;
- the rule that *every* unmerged function must be unchanged, including immediately before a finite-distance merger.

The replacement rule is more precise:

> Clearly distinct functions remain unchanged. Potentially redundant functions can approach each other through an explicit continuous transition. Functions are compacted only at exact equality of the implemented access functions.

This relaxes singleton identity only inside the transition region. It does not reintroduce global proximal drift of every group, as happened in v2.

---

## 2. Why another tolerance adjustment cannot fix the design boundary

Let `f_sep(x)` denote the read before merging and `f_merge(x)` the read after packing and replacing its access functions. A hard switch at a nonzero redundancy distance is continuous only if

\[
\lim_{x\to x_*^-} f_{sep}(x)=\lim_{x\to x_*^+}f_{merge}(x).
\]

Convergence of a clustering objective does not enforce that identity for the subsequent neural reader.

In the reported v3 construction, two unmatched operations can change the function at the switch:

\[
\operatorname{RMS}\!\left((t_i+t_j)/2\right)
\ne (t_i+t_j)/2,
\]

and generally

\[
\left\|q-(c_i+c_j)/2\right\|
\ne (\|q-c_i\|+\|q-c_j\|)/2.
\]

Preserving `Abar` and `B` proves equivalence to the virtual **already-merged** model. It does not establish equivalence to the unmerged model at the switching surface.

The new architecture must make the access functions themselves equal before changing R. Numerical compaction is then an exact quotient, not a model intervention.

---

## 3. First bounded diagnosis of the retained Run 1505

Execute this on static e500 and e473 checkpoints before modifying their implementations. No new training is required for this diagnosis.

### 3.1 Port-path audit

Trace the exact final-port-temperature output used by the metric through the current adapter, field read, refinement head, local surrogate, and loss. Record which tensor is normalized, which is in physical units, what masks/weights apply, and whether/how that output is supervised.

The accessible parent code reads P1 outside temperature before `port_refinement_head` constructs refined ports; P2 field decoding occurs afterwards. Verify this on the current branch. Improved P2 field error is not proof of improved P1 port feedback.

On a bounded representative panel, then on the existing full-grid evaluator where needed, execute these **read-only counterfactuals at the same weights**:

- normal candidate;
- parent/uncoalesced access in P0 only, with all dependent later states recomputed;
- parent access in P1 only, with normal P0;
- parent access in P2 only, preserving already-computed ports;
- parent access in all phases.

For the P2-only case, final port outputs should not change unless the current code has a documented feedback dependency. This is a useful scope test. Do not add a production P0/P1 parent bypass to rescue a metric.

Measure full field, outside temperature before refinement, final port temperature, internal/surface temperatures, and flux. Run both e500 and e473. A beneficial intervention shows same-weight reliance/sensitivity; it does not by itself explain differences between independently trained models.

### 3.2 Boundary decomposition

Replay the recorded e500 case-0641 boundary, module slot 0, near x=2.56413269–2.56422424. Preserve all physical-feasibility checks already available.

At identical coordinates compare:

- normal left/right partition selection;
- an explicitly fixed left partition;
- an explicitly fixed right partition;
- uncoalesced access.

Separate changes due to partition selection, access-key/geometry reconstruction, and downstream P0/P1 feedback. Inspect original and replacement logits, route masses/densities, rho, n, QM/QE context, and physical outputs. Do not infer the responsible phase solely from the plotted P2 plan.

### 3.3 Cost diagnosis

Use synchronized end-to-end timing, then one focused profile. Attribute preparation cost to solver launches/synchronization, actual solver arithmetic, descriptor/weight construction, packing, and ordinary physical preparation. A small problem dimension does not imply low GPU latency.

A same-input, fixed-plan replay may be used as a **diagnostic lower bound** on the cost of the old model without solving. It is not a deployable cross-geometry cache and must not be reported as a production speedup.

Do not spend this round building a faster ADMM kernel. The next production candidate removes this iterative solve.

---

## 4. New mathematical object: a continuous transform of access functions

Let the ordinary parent access functions at one prepared physical phase be

\[
\ell^0_k(q)
=\frac{v(q)^Tt_k}{\sqrt D}+g_k(q),\quad k=1,\ldots,K,
\]

where `g_k` is exactly the parent geometry function, including its source-type-center handling. Let `ell0(q)` be the K-vector.

The new router first forms

\[
\boxed{\widetilde\ell(q)=T(x)\ell^0(q)},
\]

where `x` denotes the case and prepared phase, and `T` is a small K-by-K continuous matrix. This is a transform of **query-access functions**, not physical field values or source states.

Require:

- `T=I` when no contraction is requested;
- entries nonnegative and each row sums to one;
- T varies continuously with the case inside the inherited parent's continuity region;
- closed subtrees have exactly identical rows of T;
- packing uses those constructed identities, not an arbitrary closeness tolerance.

Keep the original parent query feature/projection and logits. Do not learn a second query network.

### 4.1 A two-function illustration

For functions i and j, write

\[
\bar\ell=(\ell^0_i+\ell^0_j)/2,\qquad
\Delta\ell=(\ell^0_i-\ell^0_j)/2.
\]

Define a residual-detail coefficient `s` in [0,1]:

\[
\widetilde\ell_i=\bar\ell+s\Delta\ell,\quad
\widetilde\ell_j=\bar\ell-s\Delta\ell.
\]

At s=1, both parent functions are retained. As s decreases they approach each other continuously. At s=0, they are exactly the same function for **every query**, and the existing quotient can pack them without changing the predictor.

There is no new coarse field response and no second fine response in this construction.

---

## 5. One fixed function tree, not an online graph search

Use a binary tree over the twelve provisional function IDs. A full tree has eleven internal nodes. Each internal node `C` names a subset of leaves; node subsets are nested or disjoint.

The tree defines possible combinations. It does **not** prescribe which combinations close or a constant R. Per-case/phase continuous coefficients determine the active quotient.

This is an explicit modelling restriction: only subtree classes are available in this experiment. It trades unrestricted online clustering for a bounded, continuous, inexpensive structure. Report this restriction and its measured coverage of useful mergers; do not describe it as equivalent to the old convex-clustering problem.

### 5.1 Build the tree once from training inputs

Use the static **Run-1505 exact e500** checkpoint only to estimate a useful proposal-function scaffold. Read its **600 training cases**, not the 90-case development results, and collect raw, pre-coalescence access functions and source controls by phase.

Estimate pair dissimilarity with the source-action metric in Section 6 for two-leaf candidate sets, averaged over the training cases/phases where the pair is eligible. Build one deterministic complete-linkage agglomerative tree on the resulting twelve-by-twelve dissimilarity table. Complete linkage here is only a fixed small offline tree-construction rule; no clustering library is required.

Do not hard-code the visible `[0,10]`, `[1,8]`, `[6,7]` pairs from the illustrative development cases. Their recurrence motivates an audit, not a license to choose the tree on test/development images.

Store the eleven leaf subsets in ordinary model/config buffers. The tree is checkpoint-owned architecture metadata, not a new provenance system. It is not rebuilt during inference or changed halfway through the new run. Permutation tests must permute both proposal parameters and the tree metadata.

This uses a prior trained model for structural initialization only. The formal new model's neural weights and optimizer still start from the same fresh seed as Run 1502; there is no checkpoint-weight warm start or teacher-output distillation. Disclose the structural reuse in the report.

If the current local data cannot support the prescribed train-only extraction, report that missing input instead of silently using the 90 development cases or inventing a tree from their plots.

### 5.2 Continuous subtree contraction

For node C, let `P_C` average entries inside C and leave other entries unchanged. With residual-detail coefficient `s_C`:

\[
M_C=s_C I+(1-s_C)P_C.
\]

Apply the eleven node transforms in fixed bottom-up order:

\[
T=\prod_C M_C.
\]

Because subsets are nested or disjoint, the corresponding averaging/detail projections commute mathematically. A closed descendant remains tied after an ancestor transformation. Closing an ancestor makes every descendant function identical.

Equivalent implementation: start from an identity matrix, gather the node's rows, compute their mean, and replace those rows by

\[
\text{row mean}+s_C(\text{rows}-\text{row mean}).
\]

Use explicit tensor branches at s=0 and s=1: copy the common mean for s=0 and the original rows for s=1. This avoids floating-point cancellation compromising exact row identity or the parent-exact endpoint. The branch is compatible with the zero endpoint derivative of the taper below.

A fixed tree avoids a new discontinuity from data-dependent nearest-pair matching or online hierarchical reordering.

---

## 6. Determine redundancy from what the reader consumes

A small descriptor distance alone does not establish small physical readout error. The new redundancy score measures the effect of a hypothetical function contraction on the reader's existing source-overlap/control inputs.

This score is **not a guarantee of physical output accuracy**. It is more directly related to the inherited operator than centroid/key distance alone and gives interface queries explicit representation without adding a separate field branch.

### 6.1 Query-independent formation catalogue

Each case/phase supplies a small deterministic catalogue of probe coordinates and masks through the adapter:

- the existing environment-source coordinates for broad spatial coverage;
- actual physical port coordinates;
- outside-temperature/refinement coordinates available at that phase.

Use current case geometry and already-available phase state only. No future P1/P2 values may be requested while preparing P0. For P0, use its physical port coordinates and the existing geometry-defined outside sampling rule where available. P1/P2 can use their already-known outside coordinates.

Do not use ground-truth fields to form a runtime plan. Do not use the caller's requested output-query set. The plan must be unchanged by query chunk size, query order, or which output locations a caller requests.

Keep environment and interface catalogues as separate roles; 192 background samples must not swamp a small but important port set. Use existing valid-module masks and quadrature weights. Reuse query projection/Fourier results within preparation.

### 6.2 Hypothetical node tie

For node C and a probe p, calculate

\[
\alpha^0_p=\operatorname{sparsemax}(\ell^0(p)),\quad
\alpha^{(C)}_p=\operatorname{sparsemax}(P_C\ell^0(p)),\quad
\Delta\alpha^{(C)}_p=\alpha^{(C)}_p-\alpha^0_p.
\]

These are twelve-wide routing operations, not extra physical forwards. They use the same parent validity mask. Ineligible nodes do not affect T.

### 6.3 Exact small Gram representation of read-input changes

All original source/control tensors remain live. Prepare four K-by-K positive-semidefinite Gram matrices. Let

\[
G^M_A=(A^M)^T\operatorname{diag}(\nu^M)A^M,\qquad
G^E_A=(A^E)^T\operatorname{diag}(\nu^E)A^E.
\]

Let V_module have rows `module_control_gain(h_k)` and V_score have rows `environment_score_control(h_k)`, using the existing bias-free linear maps. Then

\[
G^M_U=G^M_A\odot(V_{module}V_{module}^T),\qquad
G^E_Z=G^E_A\odot(V_{score}V_{score}^T).
\]

The four blocks represent changes to module rho, the module gain preactivation `W_gain n`, environment rho, and environment per-head score controls. The physical reader still conserves the full raw B/n moment. Using projected controls in the discrepancy metric accounts for the existing control maps, rather than treating every raw D direction as equally important. In particular,

\[
\|\Delta\alpha F\|_W^2
=\Delta\alpha\,G_F\,(\Delta\alpha)^T.
\]

This equality computes a source-resolved control error without constructing an extra probe-by-source-by-D tensor for each node. The Gram matrices summarize the error calculation only; they do not replace the fine sources in the physical reader.

For role t and block G, define

\[
D^2_{C,t,G}
=\frac{\max_{p\in\mathcal P_t}
\Delta\alpha^{(C)}_p G(\Delta\alpha^{(C)}_p)^T}
{\sum_{p\in\mathcal P_t}w_{t,p}\,\alpha^0_pG(\alpha^0_p)^T+\epsilon_G},
\qquad \sum_p w_{t,p}=1.
\]

Use the maximum rather than an average in the numerator so a localized port read is not hidden by many benign locations. Normalize each block separately. Choose an ordinary small numerical floor in the dimensionless control units (`1e-8` initially); zero-energy blocks contribute zero. Symmetrize floating-point Gram matrices and clamp only tiny negative roundoff in their quadratic forms.

Set

\[
D_C^2=\max_{t,G}D^2_{C,t,G}.
\]

This is continuous with standard piecewise derivatives, including max and sparsemax. It need not be differentiable at every tie. Do not call it a CFD error bound or a physical conservation law.

Retain the raw numerator and denominator summaries as well, so a small ratio cannot be misread without its scale. Record both node scores and the *combined* post-transform read-input discrepancy on the same catalogues. Several acceptable local contractions can accumulate; individual-node scores do not bound the final nonlinear prediction.

### 6.4 Fixed smooth firm taper

Initial, declared modelling values for this one candidate:

```text
read_input_close_rms = 0.02
read_input_keep_rms  = 0.06
```

These are relative control-input discrepancy scales, not allowed percentages of field/temperature error and not values inferred from the reports. They are a conservative initial hypothesis, not an optimum. Do not sweep or silently increase them to obtain a desired R histogram.

With current scheduled scales `a < b`, define

\[
u_C=\operatorname{clip}\left(\frac{D_C^2-a^2}{b^2-a^2},0,1\right),\qquad
s_C=6u_C^5-15u_C^4+10u_C^3.
\]

Thus:

- below a, detail vanishes exactly and the node can close;
- between a and b, access functions move continuously;
- above b, the parent functions are unchanged by this node.

The taper has zero first and second derivative at its endpoints. This is a smooth firm-shrinkage-inspired modelling choice, **not** the ADMM solution, and not a claimed statistical optimum for HONF.

If conservative thresholds produce too few mergers, report the result. Do not replace the physical-preservation objective with a prescribed K reduction.

### 6.5 Inactive/vanishing proposal handling

Retain the parent's active-group mask. Do not contract a node containing an inactive proposal. To avoid an additional hard transition as a proposal mass approaches zero, fade its contraction eligibility to zero *before* the parent's zero-mass mask changes.

One explicit rule, using the existing normalized provisional joint mass pi_k, is an endpoint-flat smoothstep eligibility e_k that is 0 at pi_k <= 1e-6 and 1 at pi_k >= 2e-6. Set node contraction amount to

\[
\gamma_C=(1-s_C)\prod_{k\in C}e_k,
\]

and use residual coefficient `1-gamma_C` in M_C. This changes no physical source mass and deletes no proposal. Fully eligible nodes still reach exact closure; low-mass nodes revert continuously to the parent before occupancy changes.

Evaluate hypothetical routes using finite raw logits plus the ordinary mask, not arithmetic involving zero times negative infinity. Exercise inactive groups in real numerical tests.

---

## 7. Exact quotient and the continuity boundary

### 7.1 Compact only constructed equal functions

A compact class is a maximal closed subtree, with unresolved leaves retained as singleton classes. A closed subtree is recognized from an exact plateau (`gamma_C == 1`), not from `distance < numerical_class_tolerance` on approximately equal functions.

Let C_r be the resulting classes, `m_r=|C_r|`, and k_r a representative. The compact access function is

\[
\boxed{L_r(q)=T_{k_r,:}\ell^0(q)}.
\]

Do not replace it by a new RMS-renormalized query key or a single barycentric geometric distance.

For optimization, the content dot product may use the combined key `sum_k T_rk t_k` **without another RMS normalization**. Its geometry must still equal `sum_k T_rk g_k(q)`. Computing `ell0 @ T_compact.T` first is a simple exact reference. The remaining K-wide cheap geometry does not invalidate compact source/control contractions, but must be counted honestly.

### 7.2 Preserve the existing mass/density distinction

\[
a_{qr}=m_r[L_r(q)-\tau_q]_+,\quad \sum_ra_{qr}=1,\quad
b_{qr}=a_{qr}/m_r.
\]

Then keep

\[
\bar A^S_{sr}=\sum_{k\in C_r}A^{S,0}_{sk},\qquad
B^S_{sr}=\sum_{k\in C_r}A^{S,0}_{sk}h^0_k,
\]

\[
\rho^S_{qs}=\sum_r b_{qr}\bar A^S_{sr},\qquad
n^S_{qs}=\sum_r b_{qr}B^S_{sr}.
\]

All environmental K/V rows, the original value-control sum `sum_r B_jr`, physical coordinates, source measures, and provisional kappa_0 remain. Do not recompute normalization from R.

### 7.3 Why changing R need not change the prediction

Before packing, consider the virtual K-slot model with logits `T ell0`. Rows belonging to a closed class are exactly equal as functions of q. Sparsemax assigns equal constituent densities inside that class. The compact equations above therefore reproduce its rho and n, including heterogeneous source controls.

As a node approaches closure, its difference coefficients and their first derivatives approach zero. Packing at closure changes only representation. On a fixed physically valid region where the parent source/access operations are continuous, the new virtual predictor is continuous; its exact compact representation has the same value.

Ordinary sparsemax support changes are continuous, though not globally differentiable. For an environmental row approaching zero support, preserve the current overlap-mass envelope and zero-support implementation. With bounded values, its context magnitude tends to zero with overlap mass, rather than jumping from a normalized response to zero.

This is a conditional mathematical argument, not proof that every inherited source-occupancy/geometry boundary is smooth. Test the complete model, including such boundaries. Do not claim global C1/C2 design smoothness: max, sparsemax, norms, and the parent implementation have their own nonsmooth points.

### 7.4 Gradients

Backpropagate through the continuous taper, read-input metric, access transform, source incidence, and control moments. Do not detach the score merely to simplify training, and do not use a straight-through estimator.

Only integer packing/class metadata is discrete. At a closed plateau the virtual functions and relevant derivatives agree, so compact gradients must match the virtual model's gradients. All constituent fine controls/parameters remain registered and live; no optimizer state is deleted on merging.

A closed node can reopen when its original functions/source organization change with input or training. This removes permanent parameter death. It does not prove that every collapsed configuration will spontaneously escape; report diversity, control updates, and actual reopenings.

---

## 8. Implementation and GPU route

### 8.1 Reuse rather than build a parallel stack

Add one opt-in architecture and reuse the current v3 quotient/moment code after inspecting it. Proposed new helper filenames, not claims about existing filenames:

- `functional_fusion_tree.py`: fixed metadata, action Gram matrices, probe discrepancy, taper, T, class packing;
- `continuous_functional_coalescence.py`: thin parent-compatible router/backend integration;
- one experiment profile and focused tests;
- one diagnostic runner reusing current physical evaluators and boundary tools.

Do not duplicate the fine physical preparation or QM/QE implementations.

### 8.2 Bounded computation

The per-phase new work is:

- a fixed routing-probe catalogue;
- four 12-by-12 Gram matrices;
- at most eleven twelve-way hypothetical route transforms;
- eleven scalar contraction coefficients;
- a small matrix transform and one compact plan.

There is no iterative convergence loop, high-precision solve, tolerance-based union-find, or CPU optimizer in inference. Use the current neural FP32 policy. Do not globally enable reduced precision or change CUDA visibility.

Vectorize the candidate-node/probe dimension. Compute source-action discrepancies with Gram contractions. Never materialize a new `[B, nodes, probes, E, D]` tensor or `[B,Q,E,K]` physical path tensor.

Prepare once per physical phase and reuse across its receiver chunks. Do not share detached prepared values across optimizer steps or changed geometries.

### 8.3 Mixed batches and actual compaction

First implement the virtual K-wide formulation for readable correctness, then the compact Abar/B path. Mixed cases may use different R; pack to the actual batch maximum R and mask padding. Report both individual R and executed padded width.

Avoid a per-case/per-class Python CUDA loop. Tree subsets are fixed metadata, so loops over eleven nodes can be static and tensorized. Use at most the minimal host interaction needed for the existing packed-buffer allocation, not `.item()` inside every node/head/query loop.

An optional `torch.compile` path is appropriate only if the installed PyTorch 2.6 environment actually supports the implemented operations and it wins synchronized timing. Keep eager CPU/GPU fallback. Do not require an environment upgrade, a new solver package, or compiler success before executing the readable model.

### 8.4 Expectation, not a speed claim

The objective is to eliminate Run 1505's large planning penalty and bring preparation close to the parent. A 13.5% reduction in a small routing axis cannot by itself remove a dense QE physical rectangle.

Do not promise latency below Run 1502, Dense, or 1404 from this formulation alone. Measure it. No new fused attention kernel belongs to this candidate; retain rectangular by default and use the existing CSR implementation as a separately timed option.

---

## 9. Required execution tests

Tests are part of implementation, not substitutes for real training or physics evaluation.

### 9.1 Algebra and function tests

- T identity, nonnegativity, row sums, and mean preservation in exact/small numerical fixtures.
- Nested and disjoint contractions; closed descendant remains closed after ancestor contraction.
- A closed parent overrides open/partial descendants without generating new function differences.
- Compact versus virtual logits, weighted sparsemax mass/density, rho, n, QM, QE, and field.
- Gradients for query coordinates, module coordinates/properties, all constituent codes, source encoders, value/score/module control maps, and continuous taper inputs.
- Source permutation and joint prototype/tree permutation behavior.
- Unequal multiplicity, inactive proposals, missing source types, zero support, and mixed compact widths.
- Source-action Gram discrepancy versus explicit source-resolved tensors in small fixtures.
- Raw mixture-of-logits implementation versus any optimized key/geometry implementation. Include a counterexample that would fail if geometry were silently replaced by a single centroid or combined keys renormalized.

### 9.2 Boundary tests before physical training

Construct an actual closure boundary of the taper, not a disconnected mock of class labels. Compare virtual and packed model outputs on both sides for progressively shrinking brackets. The left/right difference must shrink toward the model's floating-point repeatability floor, not approach a finite plateau.

Check one-sided limits and fixed-side AD/finite differences. Also exercise the outer identity boundary, nested closures, and near-zero proposal eligibility. Check gradients at a closed node only through the actual taper; an independently variable hard class flag is not a physical derivative.

### 9.3 Parent-exact initial phase

At scheduled zero contraction, use the parent path directly. Verify logits, field, ports, and first gradients. New fixed metadata must not consume neural initialization RNG or add trainable parameters unexpectedly.

### 9.4 Real predicted-port updates

Execute the existing ordinary optimizer builder and full loss on one mixed low-M batch and one M=10 batch. Demonstrate finite, nonzero updates through both the original physical network and the active transition coefficients. An all-identity smoke batch is insufficient to validate the new transition path; include a controlled numerical transition fixture and a real case that actually enters the transition when available.

Record whole-step wall time, peak allocation, and new preparation work. Do not extrapolate a GPU speedup from a tiny synthetic algebra test.

### 9.5 Independent prototype supplied with this plan

`cfq_algebra_check.py` is an independent CPU float64 toy calculation, not repository code. It checks:

- nested continuous transforms;
- compact/virtual nonlinear response and gradient parity;
- the source-action Gram identity;
- a 5-to-6-class transition whose output discrepancy vanishes with bracket width.

The local result has compact/virtual output and gradient differences below 6e-17 and Gram error below 2e-17. In the constructed boundary test, relative differences fall from 4.10e-5 to 4.25e-14 as half-brackets fall from 1e-2 to 1e-5. These numbers validate the illustrative algebra only. They establish no HONF accuracy, port fidelity, GPU speed, or CFD sensitivity. The prototype used CPU PyTorch 2.10; production compatibility must be checked in the repository's existing 2.6 environment.

---

## 10. Formal experiment: one fresh candidate, bounded at 500

Start neural weights and optimizer from the same fresh seed/settings as Run 1502. The structural tree is the declared train-input-only prior from Section 5. Do not resume or overwrite Run 1505. Do not add a second seed, hyperparameter grid, teacher field loss, or loss-weight redesign in this experiment.

Retain K=12, D=16, source/query normalizers, dataset split, optimizer, precision, clipping, loss terms, predicted ports, frozen local surrogate, and physical refinement count. Keep training query chunking unchanged.

### Schedule

- Epochs 1–50: contraction disabled, exact parent path.
- Epochs 51–150: smoothly increase both discrepancy scales from zero to their declared final values, using the same endpoint-flat polynomial on `(epoch-50)/100`.
- Epochs 151–500: fixed scales.

At zero scale, call the parent explicitly instead of dividing by a zero transition width. At inference, use the selected checkpoint's recorded schedule state. Do not apply a final-strength planner retroactively to an early checkpoint and label it the native model.

### Epoch 50 review

Verify real parent tracking, full-model loss/update health, and absence of hidden preparation overhead when contraction is disabled. Compare against stored same-stage parent outputs/logs without demanding bitwise agreement where environment/source changes make it unwarranted.

The structural prior exists, but R is expected to remain at occupied capacity. This is not a formation failure. Inspect raw train-input node scores and predicted transition usage, not a manually imposed K target.

### Epoch 150 review

Execute all 90 cases for formation at deterministic Q1024 and physical evaluation at full Q8192. Use exact endpoint and distinct validation-selected checkpoints separately.

Report:

- R and true merged classes by P0/P1/P2;
- number of transition nodes versus exactly closed nodes;
- compact Kq, virtual support, and both participation measures;
- provisional/compact source degree and unique physical support;
- node and combined read-input discrepancies, separately at ports and environment probes;
- fluid/near/far/channel errors, final port temperature, flux, internal/surface temperatures;
- actual trained design transitions;
- preparation, prepared P2, application, and optimizer cost.

A small early scalar deficit alone should not mechanically stop a healthy candidate. Conversely, a real finite jump, persistent broad port degradation, or another large preparation penalty warrants a research stop rather than running toward 500 by default. Do not change thresholds, tree, or losses midrun to conceal a failed hypothesis.

### Epoch 500 review and stop

Stop no later than 500. Evaluate exact e500 and distinct best-field/total/temperature checkpoints, each with its own formation and physical evidence. Use paired cases and module-count strata. Do not combine complementary outputs from different checkpoints.

A recommendation for a separate 5,000-epoch maturation should address four independent results:

1. meaningful case-dependent functional reduction, not merely a low effective-K statistic;
2. competitive field reconstruction and useful port/flux behavior at the same checkpoint;
3. no new finite merge-boundary jump in the tested full physical model;
4. removal of the order-of-magnitude planning overhead, with actual matched application cost.

These are empirical research judgments, not a new gate service or arbitrary scalar release checklist. A candidate can succeed on formation without proving sub-parent latency; say so explicitly. A field-only improvement with a repeatable design jump is not ready as a modular-design operator.

No automatic continuation beyond 500 is authorized by this plan.

---

## 11. Boundary and physical-preservation evaluation

Revisit the recorded case-0641 geometry and search bounded feasible coordinate sweeps on several module-count strata. The new model's transitions may move; testing only the old coordinate is insufficient.

For observed transitions, use decreasing brackets, e.g. physical widths around 1e-2, 1e-3, 1e-4, with smaller widths only where FP32 coordinate resolution permits. Use one-sided extrapolation and unchanged-forward repeatability. Keep module feasibility checks.

Record continuous taper values, exact closure flags, T rows, compact IDs, rho/n, P0/P1 outside temperatures, final port outputs, fields, and representative scalar design objectives. Separate ordinary steep but continuous variation from a nonzero limiting jump.

Check relevant derivatives away from boundaries and both one-sided limits near closure. Include module heat powers/properties where supported, not just query coordinates. Test new eigenvalues/control structure only as diagnostics; do not substitute them for physical output measurements.

Neither autograd agreement nor neural continuity verifies CFD sensitivities. If solver-reference directional derivatives are unavailable, mark that evidence missing.

---

## 12. Correct reporting of K and utilization

Keep distinct:

- registered proposal capacity;
- occupied proposals;
- compact R;
- transition-node count;
- exact compact Kq;
- virtual constituent support;
- unique physical pair support;
- executed padded/valid rows.

For compact masses a_r and multiplicities m_r:

\[
K_{eff,compact}=\frac1{\sum_r a_r^2},\qquad
K_{eff,virtual}=\frac1{\sum_r a_r^2/m_r}.
\]

Both are useful, but compact participation automatically changes on grouping and is not evidence of additional physical pruning. A class can carry useful mass without ever being argmax. Do not remove C3/C4-like classes on the basis of missing dominant-colored regions.

Report recurring proposal-pair/subtree frequencies across cases. Distinguish globally recurring redundancy from case-specific changes; report within-module-count variation and solver-free handling of cases that were old fallbacks. R need not increase monotonically with module count and is not the physical module count.

---

## 13. Timing protocol and interpretation

Use the current physical GPU and software environment with ordinary device visibility. Record hardware, PyTorch/CUDA version, precision, maps flag, native and overridden inner chunks, outer query batching, and checkpoint identity using existing machinery.

Benchmark all compared models at matched Q8192 and matched inner chunks, including 2048 where supported. Keep the native-128 table separately for historical context. Use maps off, warmups, CUDA synchronization, and repeated measurements in reversed order when noise is material.

Start with the existing five-case M=3/5/7/10 panel. At e500, extend the main rectangular application comparison over all 90 cases. Report the scope of each table accurately.

Count formation-probe routing and Gram arithmetic separately from physical QM/QE work. A routing-only hypothetical collapse is not a physical geometry-MLP row. Include the new preparer's cost in full application and training-step measurements.

Do not claim the fixed tree, T, or R reduction has reduced the fine graph unless measured support/row counts show it. Optional existing CSR results stay separate; a memory gain is not a latency gain. Do not enlarge the scope to a new kernel project this round.

---

## 14. Main uncertainties and falsification outcomes

This is a new, testable formulation, not a guaranteed repair.

- A fixed tree may exclude important non-subtree mergers. Measure coverage rather than asserting equivalence to convex clustering.
- A small control-input discrepancy on finite probes is not a uniform field-error bound. Test omitted locations, full grids, and physical interface outputs.
- Conservative discrepancy thresholds may yield too few mergers. Do not increase them to manufacture a desired R distribution.
- Local node discrepancies can accumulate. Measure the combined transform and its physical effect.
- Initializing a scaffold from a previous trained model is architectural reuse. Disclose it; it does not establish an untouched test result.
- Exact packing removes the *new* hard switch but does not repair every inherited nonsmooth/source-occupancy boundary.
- Reduced planner cost should help relative to 1505, but may still leave the model slower than the optimized parent because fine physical work remains.
- Port error may involve training/selection or the refinement head, rather than only the organizer. The phase audit and e473/e500 comparison must keep these possibilities separate.

If the new core is fast and continuous but fails to coalesce, that is a negative result on its redundancy criterion or tree family, not permission to hide a learned deletion gate inside it. If it coalesces and preserves fields but still harms ports, use the recorded interventions to design the next targeted physical-loss/feedback experiment rather than adding an unexamined bypass.

---

## 15. Research-development constraints

Use ordinary Git, configs, checkpoint/run management, typing, focused tests, and actual execution. Preserve existing strict loading and security behavior. Keep historical checkpoints and architectures intact.

Do not introduce cryptographic hashes, contract freezes, baseline snapshots, approval databases, gating services, monitoring daemons, or unrelated cleanup. Blocking infrastructure belongs only at genuine irreversible/security/release boundaries; scientific review points here are recorded judgments about actual measured runs.

Preflight and heuristic checks do not replace real numerical readers, gradients, optimizer updates, formal training, or ground-truth evaluations. Do not wait for unrelated runs, launch extra seeds, or perform unrequested long continuations.

---

## 16. Deliverables and definition of done

Commit the new opt-in implementation, tree metadata and its train-only construction procedure, focused tests, one bounded run's evaluation tools/data summaries, and a concise report.

The report must distinguish:

- source-verified facts, diagnosed mechanisms, and untested hypotheses;
- the old v3 operator and the new continuous transform;
- function contraction versus exact compaction;
- latent source-moment conservation versus actual physical accuracy/conservation;
- case-dependent R versus physical source support;
- field reconstruction versus port/flux behavior;
- prototype CPU algebra versus repository/GPU/CFD execution;
- preparation speed recovery versus a genuine sub-parent acceleration.

The task ends after the single run stops at or before epoch 500, with a reasoned recommendation. It does not end merely because a unit test reports fewer classes, and it does not automatically launch 5,000 epochs.

Final design statement:

\[
\boxed{\text{Learn which functional differences can vanish; compact only the resulting equal functions.}}
\]

---

## References and attribution

### Project evidence

[S1] `HONF_Run1503_v3_Converged_Identity_Preserving_Coalescence.md`, especially the e50 parent tracking, e500 accuracy, e473 sensitivity, trained boundary, and matched timing sections.

[S2] `HONF_Run1505_E500_Organizer_Diagnostics.md`, especially R/Kq definitions, all-proposal occupancy, full-grid distributions, non-dominant class utilization, and representative/fallback cases.

[S3] Accessible parent source at `aaee3fa9bcc2baf4760d34aebdd57c4344e14bc7`: `sparse_incidence_router.py` and `Case_ThermalChannel/src/channelthermal/interface_field_coupling.py`. Codex must verify the latest local equivalents.

### External methodological context

[R1] Gao, H.-Y. and Bruce, A. G. (1997), *WaveShrink with firm shrinkage*, Statistica Sinica 7, 855–874. Inspiration for retaining well-separated details, exactly removing small details, and interpolating continuously between them. The proposed quintic function-transform construction is not the paper's estimator and inherits no risk-optimality claim.

[R2] Martins, A. F. T. and Astudillo, R. F. (2016), *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification*, ICML/PMLR 48, 1614–1623. Sparse simplex projection background; the existing multiplicity quotient and this tree transform are project-specific constructions.

[R3] PyTorch 2.6 documentation, *CUDA semantics*: asynchronous execution, synchronized timing, graph-capture constraints, and memory accounting. Documentation is not evidence that compiling or graphing this particular code is faster.

[R4] Chi, E. C. and Lange, K., *Splitting Methods for Convex Clustering*, arXiv:1304.0499. Historical solver context. The new candidate deliberately does not solve the convex-clustering objective online.

Public source locations:

```text
https://www3.stat.sinica.edu.tw/statistica/j7n4/j7n43/j7n43.htm
https://proceedings.mlr.press/v48/martins16.html
https://docs.pytorch.org/docs/2.6/notes/cuda.html
https://arxiv.org/abs/1304.0499
```
