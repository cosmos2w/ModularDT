# HONF Run 1501 maturation and the next controlled refinement

## Codex work plan: preserve the successful model, establish executable sparsity, test one scientific change

**Repository:** `cosmos2w/ModularDT`  
**Branch:** `agent/honf-core-next`  
**Inspected reference:** `fe8c0f2a61aa3ead841944ce9b5fc37119d589a2` (`fe8c0f2`)  
**Parent:** `sparse_incidence_group_control_honf`, Run 1501  
**One optional new scientific run:** Run 1502, final-environment-source sparsemax refinement  
**Scope:** one existing-run continuation, exact execution/evidence corrections, and at most one new candidate. No sweep.

> Central objective: a query should access a small, learned subgraph of shared multi-entity groups, retain individually addressable physical source information, and execute each required fine query–source interaction at most once. Group control must influence the nonlinear response; it must not become a substitute pooled field or decorative visualization.

---

## 0. Read this first: decisions and boundaries

**Continue Run 1501 from its exact epoch-500 checkpoint to total epoch 5000.** Restore its optimizer, scheduler if present, RNG, normalization, physical losses, and dataset split through the existing resume path. Do not restart from the visually or numerically attractive best-total checkpoint at epoch 444. It is an evaluation checkpoint, not the continuation point. If resoration has already launched, do not do it repeatedly.

First inspect existing jobs and the run manifest. If the user has already resumed 1501, leave that process alone and do not launch a duplicate. Use another free physical GPU and an ordinary Git worktree for independent code/evaluation work. Do not edit the source checkout used by a running job.

Do not change the mathematical model, optimizer recipe, source/query normalizers, or learning-rate policy of the resumed 1501. In particular, do not reinterpret a scheduler's original total duration silently. Use the repository's existing continuation semantics and report them.

The remaining work has two distinct meanings:

- **Exact implementation work:** a new execution order for the same 1501 predictor. It must load the same checkpoint and preserve outputs and tested first derivatives. It does not need a new training run or model number.
- **Scientific refinement:** optional Run 1502 changes only the final environmental assignment normalizer from entmax-1.5 to sparsemax. This is a new learned operator and must be evaluated as such, not described as executor parity.

Do not add case-level dynamic-K gates, mass competition, occupancy deletion plans, a sparsity curriculum, entropy/load-balance penalties, hard top-k, new geometric barriers, sampled environmental interpolation, coarse/local bypasses, or another wide group-value bank.

### Completion checklist

- [ ] Preserve or correctly resume the existing 1501 trajectory.
- [ ] Repair evidence scope/geometry/work accounting; do not alter predictions to make figures look plausible.
- [ ] Establish clean, same-device, maps-off execution references.
- [ ] Implement and measure one support-union block executor, with an exact rectangular fallback.
- [ ] Audit scientific role, boundary behavior, and checkpoint selection on existing 1501 weights.
- [ ] Prepare one optional Run-1502 profile with a single mathematical change.
- [ ] Train 1502 only if the bounded source-selectivity experiment is scientifically informative; initial review at 50, then 150/500 as justified.
- [ ] Report 1501 maturation and 1502 separately. No automatic 1502 extension beyond 500.

---

## 1. Evidence established so far—and what is not established

Source: `HONF_Run1501_SparseIncidence_Evaluation.md` at the inspected commit.

| Evidence | Source-supported conclusion |
|---|---|
| Exact-500 pooled fluid relative L2: 0.11149 | Healthy, competitive interim model, but worse than Dense exact-500 at 0.09874 |
| Saved-best-total e444: pooled 0.09891 | Strong representational potential; not a policy-matched victory over Dense's exact endpoint |
| e444 wins 48/90 cases against Dense e500 | Promising paired performance; one seed and a repeatedly used development holdout |
| Kq mean 2.9254 → 3.1182 → 3.1673 at e50/e150/e500 | Nontrivial query-local routing; not universal prototype collapse |
| RE 0.7148 → 0.6733 → 0.6277 | Environmental unique support becomes narrower as training proceeds |
| RM 0.6997 → 0.7200 → 0.7497 | Module support becomes broader, not sparser |
| Maintained fine readers remain rectangular | Mathematical support reduction is not current forward-call elimination |
| Selected diagnostic reader is slower and higher-memory | The existing pair-materializing implementation is not useful on the tested shapes |

Do not combine e444 accuracy with e500 routing into a claim about one checkpoint. Export e444 formation using the same query cohort before making a joint accuracy–support claim.

The report does not establish mature accuracy, out-of-distribution compositionality, calibrated uncertainty, causal group identities, or superiority over a matched trained no-group-control model. Preserve those limits.

The 90-case split is a **development holdout**, not an untouched final benchmark. Use it for continuity with prior work but do not call it independent final validation. Do not change the existing split to manufacture a new comparison in this task.

---

## 2. Mathematical specification of the parent to preserve

Suppress batch and physical-phase indices where unambiguous.

### 2.1 Symbols and dimensions

- B: number of cases in the batch.
- M_a: active module count; M_pad: padded module storage width. **M_pad is not K.**
- E: environmental source count (192 in the present ThermalChannel profile).
- Q: receiver count; distinguish requested Q, outer evaluator chunk Q_eval, and inner core chunk Q_core.
- K=12: registered group capacity.
- D=16: group-control width.
- H=256: fine physical hidden width.
- J=128: Dense QM message width.
- N_h=4: environmental attention heads; d_h=H/N_h.
- x_i, y_j, q: physical module, environment, and receiver coordinates.
- omega_i^M, omega_j^E: nonnegative normalized physical source measures, with zero mass on padded modules.
- z_i^(p), e_j^(p): individually retained, phase-dependent fine states following Dense MM/ME/EM preparation.
- u_i^M,u_j^E,g_c: low-dimensional source/global controls.
- c_k: learned group prototypes.
- A^M in R^(B×M_pad×K), A^E in R^(B×E×K): final source incidence.
- alpha in R^(B×Q×K): query incidence.
- h_k in R^D: bounded many-body group control, not a direct field-value token.

### 2.2 Phase-local organization

Retain all existing MM/ME/EM fine messages, module/environment updates, and physical P0/P1/P2 calls. Every phase recomputes its source organization; only model parameters are shared across phases.

Content logits produce proposal entmax assignments:

\[
L^S_{sk}=\langle u_s^S,c_k\rangle/\sqrt D,
\qquad
P^S_{s:}=\operatorname{entmax}_{1.5}(L^S_{s:}),
\quad S\in\{M,E\}.
\]

The implemented masks, source measures, and unit temperatures remain authoritative. Proposal masses generate joint physical centres r_k^0. One geometry correction gives

\[
B^S_{sk}=-\|x_s-r_k^0\|/s_{\rm geo},
\qquad
s_{\rm geo}=0.25\sqrt{\sum_d L_d^2},
\]

\[
A^S_{s:}=\operatorname{entmax}_{1.5}(L^S_{s:}+B^S_{s:}).
\]

Phase-empty proposal columns are masked within that phase. They are not permanently deleted from the registered prototype bank. Preserve the current empty-source handling and avoid differentiating diagnostic integer IDs.

Final source masses and continuous occupancy scale are

\[
\mu_k^S=\sum_s\omega_s^S A^S_{sk},
\qquad
\pi_k=\tfrac12(\mu_k^M+\mu_k^E),
\qquad
\kappa=(\sum_k\pi_k^2)^{-1}.
\]

The present group control is a tanh-bounded learned function of the scaled source moments, source masses, global control, and prototype. Keep its exact implementation.

**Interpretation:** kappa is a concentration statistic used by this model's scaling convention. It is not a physical conservation law, a discovered number of physical mechanisms, or proof of an optimal rank.

### 2.3 Query incidence

Run 1501 uses

\[
\bar v_q=\operatorname{RMS}(v_q),
\qquad
\bar k_k=\operatorname{RMS}(c_k+W_hh_k),
\]

\[
\ell_{qk}=\langle\bar v_q,\bar k_k\rangle/\sqrt D+b^Q_{\rm geom}(q,k),
\qquad
\alpha_{q:}=\operatorname{masked\ sparsemax}(\ell_{q:}).
\]

RMS uses the current epsilon, and query geometry uses the current module/environment centre distances. Preserve both in 1501 and the initial 1502 candidate.

### 2.4 Group-controlled physical pairs

\[
\rho^S_{qs}=\sum_k\alpha_{qk}A^S_{sk},
\qquad
n^S_{qs}=\sum_k\alpha_{qk}A^S_{sk}h_k.
\]

For bounded h,

\[
\|n^S_{qs}\|_\infty\le\rho^S_{qs}.
\]

The expensive QM function is evaluated once per physical pair, using the existing split first affine, GELU, multiplicative group modulation, and remaining Dense-width MLP. Preserve the unnormalized moment; do not add n/rho.

The module read's exact current scaling is

\[
W_M(q)=\sum_i\omega_i^M\rho^M_{qi}\psi_M(q,i,n^M_{qi}),
\qquad
G_M(q)=\sum_i\omega_i^M\rho^M_{qi},
\]

\[
C_M(q)=\kappa\frac{M_a}{1+M_a}\,W_o^M W_M(q)
+\kappa G_M(q)b_o^M.
\]

This equation records the placement of the output bias, not just the weighted sum. Do not silently change that placement in an executor.

For environmental head a, source-side modulation prepares one K/V bank per source. Query-dependent control is

\[
\zeta^a_{qj}=\sum_k\alpha_{qk}A^E_{jk}(B_hh_k)_a.
\]

The score is

\[
s^a_{qj}=\frac{Q_q^a\cdot K_j^a}{\sqrt{d_h}}
\bigl[1+\tanh(\zeta^a_{qj})\bigr]
+b^a_{\rm geom}(q,y_j)+\log\omega_j^E+\log\rho^E_{qj}.
\]

Normalize over supported sources, preserve the learned per-source value gain, concatenate heads, and apply the output projection:

\[
C_E(q)=\kappa G_E(q)\operatorname{Out}
\left[\operatorname{concat}_a\sum_{j\in\mathcal N_E(q)}p^a_{qj}V_j^a\right],
\qquad
G_E(q)=\sum_j\omega_j^E\rho^E_{qj}.
\]

An unsupported source has exactly zero participation. A receiver with no supported source of a type returns zero for that complete type contribution, including its output bias. No hidden epsilon route is added.

The field remains

\[
C(q)=C_g(q)+C_M(q)+C_E(q),
\qquad
\widehat U(q)=D_\theta(\operatorname{LN}C(q)).
\]

### 2.5 What makes this a useful hypergraph hypothesis

The three incidences define shared multi-entity organization. Through h_k and n_qs, changing other members can alter a fine source's response at q. This is richer than multiplying a group-independent pair function by a routing coefficient.

However, many-body dependence can also arise through Dense MM/ME/EM preparation. A nonzero mixed derivative alone does not prove that the group branch is responsible. Establish group-specific reliance by interventions that preserve incidence while removing only collective modulation, and label them as surrogate-model evidence.

---

## 3. Stage A — preserve maturation and repair evidence

### 3.1 Continue Run 1501, do not replace it

Known run directory from the report:

```text
HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/
  Run_1501_20260922_211056_sparse_incidence_adaptive_honf/
```

Resolve exact checkpoint filenames from the manifest and actual files; do not guess a resume path. Resume the exact epoch-500 trajectory, unless a later authorized continuation is already running. Record the real command and checkpoint/epoch selection.

Use existing checkpoint retention and metric reporting. Retain ordinary latest/best checkpoints and add only missing milestones at 1000, 2500, and 5000 through the maintained configuration mechanism. No fresh 1501 seed, restart, optimizer reset, or switch to e444.

At 1000 and 2500 use compact formation checks on the same eight diagnostic cases plus ordinary validation logs; do not repeat every expensive 90-case counterfactual. At 5000 compare full 90-case exact endpoints and explicitly selected checkpoints. Preserve original reports and label later evidence separately.

Persistent nonfinite computation or a material correctness failure merits stopping to investigate. A missed speed target, a single bad validation epoch, or wider Kq does not automatically terminate a healthy maturation run.

### 3.2 Checkpoint policies

Produce two separate tables:

1. exact epoch N against exact epoch N;
2. saved-best validation-total through the same budget against the same selection policy, when comparator checkpoints exist.

Optionally show best-field as its own labeled policy, never substitute it silently. Do not choose checkpoints using the 90-case full-grid test result. If a comparator's corresponding saved checkpoint is unavailable, say so.

Run the lightweight formation audit on both e444 and e500 now. Do not attach e500 support numbers to e444 predictions. At later milestones repeat this rule.

### 3.3 Fix work accounting before interpreting speed

The existing summary mixes a correct full-Q rectangular count with partially propagated diagnostics. Audit the full path from backend → core → physical wrapper → `predict_case` → population reducer.

For a phase with Q_p receivers:

\[
P_{M,\rm rect}=\sum_b Q_{p,b}M_{{\rm pad},b},
\quad
P_{M,\rm valid}=\sum_b Q_{p,b}M_{a,b},
\quad
P_{M,\rm padding}=P_{M,\rm rect}-P_{M,\rm valid},
\]

\[
P_{E,\rm rect}=\sum_b Q_{p,b}E_b.
\]

The module denominator is **not Q×K**. In this dataset M_pad=12 and K=12 happen to coincide. Remove that ambiguity from code, captions, and legends.

The report's e50 values imply M_a≈4198.1/(1024×0.6997)≈5.86 only as a rough ratio-of-means check. The reported mean padded rows 789 cannot simultaneously be the full-Q padding for a 12-slot rectangle with roughly six active modules. A 128-query subtotal is a plausible explanation, not a proven correction. Recompute from actual per-case masks and summed chunk counts.

Specifically inspect `group_control_environment_geometry_rows_forward`: the current rectangular reader computes geometry for all rectangular pairs, but its diagnostic value is derived from positive support. Count the executed geometry rows, not the mask count.

Aggregate numerators/denominators before division; distinguish averages of per-case ratios from pooled ratios. Add one synthetic test with **M_pad != K** and at least two unequal receiver chunks, plus one real Q1024/Q8192 consistency check. Verify both outer evaluator chunks and inner core chunks.

### 3.4 Fix figure provenance without changing the physical data

The supplied representative board shows a module marker at the origin and a striped/stepped query sample pattern. Audit both:

- `render_case_board` currently plots every stored module row. Filter by the actual module-present mask, label active module IDs, and confirm physical coordinate units match the environment. Do not assume the marker at the origin is a real module.
- `_query_sample` chooses a rounded linspace through the flattened full grid. This produces a legitimate deterministic sample, but its spatial scatter can look striped. Do not interpret the sampling pattern as a physical routing pattern.
- For the two representative full-field boards use all Q8192 original grid coordinates and the original reshape/index mapping. For the 90-case Q1024 population keep the fixed subset but label it clearly.
- Use equal physical aspect ratio, domain bounds, active module outlines, a visible M/E matrix separator, and query support degree separate from dominant group/confidence.
- Sorted assignment heatmaps are descriptive displays. They do not prove block-diagonal physical topology.

One maintained renderer should emit clean vector PDF and a PNG preview into the existing evaluation folder. No new plotting framework or duplicated per-case artifact hierarchy.

### 3.5 Interpret “internal error” and thermal accuracy literally

The current table uses an `internal mean relative-L2` / internal-module-cell reconstruction metric. Trace the exact target, mask, channels, normalization, and denominator before equating it with the separate local-surrogate internal-temperature metric.

Report fluid, near/far fluid, internal cells, internal temperature, interface temperature, normal heat flux, port temperature/effective h, and p95/worst cases under their actual definitions. Add missing metrics from existing model outputs/targets; do not infer physical benefits from a different aggregate.

---

## 4. Stage B — exact support-union execution for the existing model

### 4.1 First establish three timing scopes

Current selected-executor `_forward()` always sets `return_routing_maps=True`. It benchmarks the diagnostic selected path, not a maps-off deployment path. `predict_case` also converts prediction chunks and requested routing arrays to CPU inside its call.

Preserve those reported results as **evaluator-with-diagnostics measurements**. Add, do not retroactively substitute:

1. **GPU-resident full physical forward:** actual physical solve, final GPU outputs, maps off, no metric/plot/CPU-array export.
2. **Prepared P2 decode:** prepared state made outside timing, current physical values, maps off, output kept on GPU.
3. **Existing application evaluator:** `predict_case`, explicitly report chunking, return flags, CPU transfer scope, and full wall time.

Same GPU, dtype, precision policy, receiver coordinates, port settings, and matched outer/core chunk sizes for each pairwise comparison. Measure the native/default profile separately from one aligned large-query chunk setting. Do not compare 1501's 128-query core to a comparator's 2048-query core without labeling it.

Use 0273 and 0653; include 0644 and 0686 only as low/high-support anchors after the initial measurement. Three warmups and ten synchronized repeats are sufficient. Use synchronized wall time for end-to-end latency and CUDA events as a separate device-stream metric. Benchmark one model/mode at a time; do not retain the previous mode's GPU outputs or routing maps during the next memory measurement. Store parity outputs on CPU outside timed intervals.

For training, one disposable B48/Q1024 real M1 and M12 batch per model with matched optimizer/loss and a short warmed step benchmark suffices. Do not derive new training-speed claims from old runs measured under different protocols.

If attribution is still unclear, take one bounded CPU/CUDA trace of full forward and P2 for 1501 and Dense. Profile and headline timing are separate: shape/memory tracing itself can retain tensors and alter overhead.

### 4.2 Why the previous selected implementation is expensive

`_read_environment_partial` gathers query, key, and value vectors per supported physical pair, builds pair-level scores, then performs segmented maxima, normalizers, and index-add value reductions. At broad support this discards the source-bank reuse of regular matrix multiplication. Reducing scalar row counts need not reduce latency or activation memory.

Do not repeat that implementation under another filename. Do not normalize once per hyperedge and mix the resulting responses: that is a different model and recreates the per-group physical duplication problem of 1405.

### 4.3 Exact support equivalence classes

Let

\[
S^M_{ik}=\mathbf1[A^M_{ik}>0],
\quad
S^E_{jk}=\mathbf1[A^E_{jk}>0],
\quad
U_{qk}=\mathbf1[\alpha_{qk}>0].
\]

For either source type:

\[
\mathcal N_S(q)=\{s:\exists k,\ U_{qk}S^S_{sk}=1\}.
\]

Encode the twelve positive group bits in an integer word:

\[
b_q=\sum_{k=0}^{11}2^kU_{qk},
\quad b_s=\sum_{k=0}^{11}2^kS_{sk}.
\]

Then

\[
s\in\mathcal N_S(q)\iff(b_q\mathbin{\&}b_s)\ne0.
\]

For a case and receiver chunk, queries with the same support signature have the same source union. Group them into disjoint query buckets. Within one bucket:

- gather query rows once;
- gather the **unique** supported source bank once;
- compute all live rho/control weights for those rows;
- run ordinary rectangular QM/QE over that smaller query×source block;
- scatter the output rows back once.

Two signatures with identical source union may share an execution bucket, but each query retains its own alpha and numeric weights. This is an optional simple deduplication, not a requirement to build a new general compiler.

Every query belongs to exactly one execution bucket for a source type. Every source occurs once within its union. Therefore no group-overlap multiplier exists:

\[
P_{S,\rm fine}=\sum_{\text{buckets }t} |Q_t|\,|\mathcal N_S(t)|
\]

when buckets have identical exact support. If buckets are combined using a support superset, count all actually evaluated rectangle entries and retain an exact per-query mask.

### 4.4 Preserve the parent operator exactly

Environmental normalization is over the union \(\mathcal N_E(q)\), not separately over group members. Preserve source measures, rho, head-wise multiplicative score control, geometric bias, value gains, kappa, and output biases exactly.

For a bucket t:

```text
Q_t [heads, queries_t, d_h]
K_t,V_t [heads, sources_t, d_h]
alpha_t [queries_t, K]
A_t [sources_t, K]
       ↓
rho_t and scalar/head control
       ↓
Q_t K_t^T + existing geometry/control/measure terms
       ↓
one softmax across supported source union
       ↓
weights @ V_t
```

Do not materialize Q/K/V with one H-wide copy per pair. The Q×S scalar score/control matrices and the module D-wide moments are acceptable bounded intermediates; their costs must still be recorded.

Cache phase-static source masks, source index lists that have actually been used, and prepared K/V only inside the current physical phase. Refresh all state-dependent tensors at P0/P1/P2. No cross-step autograd cache or detached learned values.

Support metadata is discrete and may be detached. Numeric selected memberships, alpha, group controls, values, and geometry remain live. First-derivative equivalence is expected away from support-change boundaries; do not claim global smoothness across a discrete dispatch boundary.

### 4.5 Bounded implementation and exact fallback

Prefer one new helper such as `support_block_reader.py` and an explicit runtime executor option for the sparse-incidence backend. Do not make `return_routing_maps` choose the predictor or executor.

Keep `rectangular` as the historical default. Add `support_blocks` and, only if useful, one simple measured hybrid decision. If support is broad or buckets are too small, run the exact rectangular reference. Dense fallback is honest execution of the same operator; it is not learned sparsity.

Do not allocate a 2^12 table. Process only observed query signatures. Prototype K/V remains once per source, not per group. Reuse existing masked reductions and source projections.

Start with B1 prepared inference, then verify mixed B and a real backward pass. Do not launch a fresh 1501 training run to validate an executor. The continuing 1501 training keeps its reference execution. Once parity and speed are established, new checkpoints can be evaluated by either executor.

No custom Triton/CUDA kernel is required in this goal. If support blocks still lose, report the actual crossover and stop executor work rather than adding a second kernel project.

### 4.6 Exactness and performance evidence

One small algebra test should compare dense and support-bucket results for multiple signatures, shared sources, and a receiver with no support for one source type. Then test real e444 and e500 checkpoints on the two anchors, including full physical outputs and all connected first gradients on a disposable real predicted-port batch.

Use established absolute/relative tolerances with near-zero handling, not bitwise equality. Existing 5e-5 maximum field-difference evidence is only a starting convention; also report normalized relative error and losses. Do not call a tested output-only path training-equivalent until backward is tested.

Log logical group paths, unique valid support, executed fine rows, padded/superset rows, signature count, source-bank gather size, and preparation/index cost. Support construction and row permutations belong inside the timed execution.

A useful implementation can win on larger Q/E while not winning on Q8192/E192. Label each workload honestly. Small synthetic scaling tests assess execution only, never physical generalization.

---

## 5. Scientific evaluation that keeps the hypergraph central

Use a fixed eight-case panel: 0273, 0653, 0644, 0686 and four cases selected from existing development strata before new results are inspected. Keep the panel stable across counterfactuals. Full 90-case evaluation is reserved for the principal milestones.

### 5.1 Three distinct claims

1. **Adaptive organization:** measured A/alpha supports vary with case, query, and phase.
2. **Useful collective conditioning:** group-dependent modulation contributes beyond topology or plain pairwise values.
3. **Adaptive computation:** the normal selected executor actually omits fine rows and lowers measured cost.

A model can support any subset of these. Do not infer one from another.

### 5.2 Minimal frozen interventions

On existing e444/e500 and later mature weights:

- **Group-modulation neutralization:** retain numeric A, alpha, rho, kappa, and geometry; set only the group-control modulation of QM hidden features and QE values/scores to identity. This changes the predictor and measures reliance, not a retrained baseline.
- **Query-route neutralization:** uniformize alpha over currently valid groups while retaining sources. Report error and direct output change separately.
- **Geometry/content separation:** suppress only query/source learned content or only the finite geometry bias in a documented recomputation. A geometric compactness improvement over shuffled assignments alone cannot show dependence on physical source attributes.
- **Module parameter perturbation:** keep positions fixed and vary an existing continuous material/heating/port input within its trained range. Determine whether A, h, alpha, and predicted field respond. No claim of physical correctness without corresponding simulation/measurement targets.

Do not zero h in the query router when the intended intervention is to preserve routing. Recompute all dependent cached gain banks consistently with the intervention; stale caches invalidate the test.

### 5.3 Collective interaction and inverse-design relevance

For two module attributes x_i,x_j, define a finite mixed response

\[
\Delta_{ij}\widehat U
=\widehat U(x+\delta_i+\delta_j)-\widehat U(x+\delta_i)
-\widehat U(x+\delta_j)+\widehat U(x).
\]

Compare this response with group modulation enabled and neutralized, keeping the intervention protocol explicit. It detects model nonadditivity and dependence on the group path. It is not new CFD validation and does not establish a unique hypergraph function class.

Check autograd versus finite differences for a few module coordinates/attributes and receiver coordinates. Distinguish points away from support boundaries from crossings. Sparsemax has a piecewise-linear Jacobian, with singleton support having zero local route-logit derivative; a sparse support need not imply an unusable full-field gradient because fine kernels provide other paths. Measure the complete derivative.

For a small coordinate sweep across a support transition, plot field value, support, and finite-difference slope. Continuous sparsemax weights alone do not prove globally smooth output: source-centre empty-group rules and conditional normalization also matter. Mark any jump rather than smoothing it away in a figure.

### 5.4 Normalization audit, not a new normalization module

Export the current amplitude factors

\[
a_M(q)=\kappa G_M(q),\qquad a_E(q)=\kappa G_E(q).
\]

Check their distributions by near/far/interface region and their relationship with errors and support changes. A single source-mass concentration scale cannot generally make both type-specific overlaps unit mass for every query. Record this as a property of the model, not a discovered numerical bug or conservation law.

Do not change kappa, source measures, or amplitude placement in Run 1501 or initial Run 1502. A future normalization experiment needs its own evidence and should not be bundled here.

---

## 6. Stage C — one optional Run 1502, not another redesign

### 6.1 Hypothesis

Run 1501 already made queries selective. The remaining environmental source union may be broad because each environmental token belongs to several groups. Test whether sharpening **only the final environmental incidence** improves the fidelity–support tradeoff.

Do not sharpen all three matrices together. Do not change K, query scoring, geometry, phase sharing, fine kernels, or losses.

### 6.2 Exact mathematical change

Keep content proposals P^M,P^E on entmax-1.5. Their centres and the geometric logits are therefore generated by the same formula as 1501.

Keep final module incidence:

\[
A^M_{i:}=\operatorname{entmax}_{1.5}(L^M_{i:}+B^M_{i:}).
\]

Change only final environmental incidence:

\[
\boxed{
A^E_{j:}=\operatorname{masked\ sparsemax}(L^E_{j:}+B^E_{j:}).
}
\]

Keep query sparsemax, prototype-anchored RMS keys, phase-local refresh, group-control width, kappa formula, and fine responses unchanged. Final masses/centres/h and subsequent physical phases naturally change as consequences of this single intervention; do not claim that they are frozen.

Sparsemax is not guaranteed to produce a desired support count or preserve accuracy. It can create singleton environmental memberships with zero local routing gradient and can remove useful cross-group sharing. Those are the risks being tested. Do not add a compensating entropy loss or schedule.

### 6.3 Code approach

Prefer one optional config field within the existing sparse-incidence family:

```text
interface_model.environment_refinement_normalizer:
    "entmax15"  # historical/default Run 1501
    "sparsemax" # proposed Run 1502
```

Introduce a narrow final-environment-normalization hook in the reusable assignment builder. Historical routers call the existing entmax implementation exactly. Only the Run-1502 profile opts into sparsemax. Ensure the field is not accidentally applied to the proposal stage or module assignments.

Keep architecture family `sparse_incidence_group_control_honf`, or use a thin explicit alias only if the existing configuration system genuinely requires it. Do not duplicate the backend or create another inheritance stack merely to change one normalizer.

Proposed profile:

```text
src/config_core/forward/experiments/
  run1502_environment_refinement_sparsemax.json
```

Use the established overlay format, not a new config loader. Before assigning run ID 1502, inspect the managed allocator; do not overwrite any existing run with that ID.

### 6.4 Bounded pre-training counterfactual

On the fixed eight-case panel and the existing e444/e500 states:

1. Use the parent normalizer.
2. Replace only the final E normalizer and correctly refresh its downstream state.
3. Compare A_E, h, alpha, source-union support, physical outputs, thermal/interface errors, and connected gradients.

This is not a trained accuracy forecast. It determines whether the one proposed change actually reduces source overlap and whether it creates an immediate structural pathology.

Measure both support reduction and source/query semantic effects. If support remains essentially unchanged, or the new rule only produces a universal dead/single global group, document that this candidate lacks a useful premise and do not manufacture a different variant in the same goal.

A notable frozen prediction change alone is not a reason to forbid end-to-end training: the new operator must relearn. A clear loss of all gradients or unsupported query/type behavior is a correctness problem to fix before a managed run.

### 6.5 Managed Run 1502

At most one fresh candidate, same seed and training/data recipe as 1501, from scratch. No parent warm start or distillation; the comparison is a matched learning experiment. Use a free physical GPU without competing with 1501 maturation.

Identity:

```text
run-id: 1502  # confirm unused
run-name: sparse_incidence_environment_refinement
```

Initially use the exact rectangular reader in both parent/candidate scientific comparisons. The exact block executor is evaluated on frozen checkpoints separately, so hardware dispatch does not become a second model variable.

Reviews:

- **Epoch 10:** numerical health, source/query supports, and actual updates. Finite but broad support is not an automatic failure.
- **Epoch 50:** compare training trajectories and the fixed diagnostic panel against 1501 e50. Review source degree, query degree, RE/RM, singleton fractions, near/interface thermal error, time, and memory. A 50-epoch accuracy deficit alone is insufficient to reject a steadily learning model.
- **Epoch 150:** if the earlier evidence supports the hypothesis, continue the same run/optimizer. Perform the 90-case matched physical and formation comparison. Distinguish genuine support gains from merely collapsing h or globalizing the field.
- **Epoch 500:** if still scientifically useful, complete the same run and report exact endpoints plus matched best-validation policies. Stop and ask before extending 1502 beyond 500.

No fixed requirement to beat Dense latency at 50. No automatic stop simply because query degree grows modestly during improving reconstruction. Persistent numerical failure, universal unusable collapse, or sustained major physical regression with no improvement in the targeted source support can justify early closure, with the evidence recorded.

Do not introduce a second normalizer candidate, temperature sweep, extra loss, or phase-sharing change if this run fails.

---

## 7. Acceptance of claims and final reporting

### 7.1 Parent maturation

At total epoch 5000, report at least exact-5000 and best-validation-total-through-5000, using equivalent available policies for Dense/1406/1404. Keep e444 and e500 results labeled as early evidence rather than retroactively choosing the most favorable comparison.

Full 90-case results: pooled and equal-case fluid errors, median/p95/worst, near/far, each field channel, internal cells and thermal metrics, interface/port quantities, paired case differences, and existing module-count/heating/spacing strata. One seed remains one seed; casewise uncertainty intervals do not replace seed uncertainty.

Use fixed sampling for longitudinal supports, plus full-grid supports on representative cases. Add a small check with a second spatially stratified query subset to determine whether flattened-grid sampling materially biases support statistics. Do not rerun full population audits many times for prettier plots.

### 7.2 Executor

Keep evaluator cost, GPU-only cost, prepared-read cost, and training-step cost separate. For every actual executor win report:

- checkpoint, source revision, GPU, dtype, exact query counts and two chunk levels;
- maps/debug flags and CPU-transfer scope;
- logical support and executed rectangle/selected rows;
- index/permutation overhead;
- time distribution, allocated and reserved memory;
- output and connected-gradient equivalence.

A faster rectangular implementation is a useful exact optimization, but is not learned fine-work skipping. A selected path with lower support but higher cost is not accelerated.

### 7.3 Candidate 1502

State the hypothesis and sole normalizer difference clearly. Compare at equal epochs and equivalent selection policies. Keep scientific performance of rectangular reference distinct from the optional exact block executor.

If 1502 is worse but 1501 matures well, retain 1501. Do not treat one negative source-sharpening test as failure of the central HONF concept.

### 7.4 Deliverables and code ownership

Use existing folders and one current report, with per-run numerical evidence stored under their managed `evaluations/` roots. Suggested report:

```text
docs/reports/HONF_Run1501_Maturation_and_Run1502_Study.md
```

Suggested shared frozen-execution study folder, if no existing evaluator manifest covers it:

```text
diagnostics/generated/run1501_support_block_study/
```

Prefer extending these maintained modules:

```text
src/honf_forward_core/interface_fields/sparse_incidence_router.py
src/honf_forward_core/interface_fields/sparse_incidence_group_control.py
src/honf_forward_core/interface_fields/group_control_pairwise.py
src/honf_forward_core/interface_fields/occupancy_group_router.py
src/honf_forward_core/interface_fields/core.py
Case_ThermalChannel/src/channelthermal/evaluation/prepared.py
tools/diagnostics/run_run1501_selected_executor_benchmark.py
tools/diagnostics/run_run1501_population.py
tools/diagnostics/run_run1409_occupancy_population.py
tools/diagnostics/sparse_incidence_evidence.py
tools/diagnostics/render_run1501_report_summary.py
```

General HONF owns routing, normalizers, exact readers, and metadata contracts. ThermalChannel owns coordinate/mask extraction, physical port setup, targets, and engineering metrics. Do not hard-code case IDs, channel boundaries, or output meaning in the general router.

### 7.5 Research-code discipline

Use ordinary Git/worktrees, existing managed runs, current trusted checkpoint loading, focused tests, and empirical execution. Keep historical modes and checkpoint behavior intact.

Do not add new hashes, cryptographic provenance, frozen contracts, snapshot frameworks, approval services, monitoring daemons, or a broad release process. Existing safeguards remain. Scientific budget reviews are written evidence-based decisions, not new CI gates.

Required tests should fit the actual changes: old default normalizer behavior; new E-final hook scope; M_pad≠K accounting; multi-chunk map/ledger invariance; active-source geometry alignment; dense/block forward and gradient parity; empty-type support; one real physical update. Run the maintained relevant regression suite once after integration rather than repeatedly redoing unrelated full suites.

---

## 8. References and provenance

### Repository evidence inspected

All following paths were inspected at `fe8c0f2a61aa3ead841944ce9b5fc37119d589a2`:

- `docs/reports/HONF_Run1501_SparseIncidence_Evaluation.md`: checkpoint policy, population supports, evaluator and selected-path results.
- `interface_fields/sparse_incidence_router.py`: phase-local entmax source organization, kappa, RMS/prototype query keys, geometry, query sparsemax.
- `interface_fields/sparse_incidence_group_control.py`: inherited fine readers, kappa/K amplitude rescaling, diagnostic rerouting.
- `interface_fields/group_control_pairwise.py`: exact QM/QE formulas, split affine, fine-pair materialization in the partial reader, complete-QE checkpoint boundary, and work counters.
- `channelthermal/evaluation/prepared.py`: outer query chunks and CPU transfers inside `predict_case`.
- `tools/diagnostics/run_run1501_selected_executor_benchmark.py`: maps-enabled selected benchmarking and retained mode outputs.
- `tools/diagnostics/run_run1409_occupancy_population.py`: deterministic flattened-grid sampling and geometry payload.
- `tools/diagnostics/sparse_incidence_evidence.py`: population interpretation and plotting.

### Methodological sources

- Martins & Astudillo (2016), *From Softmax to Sparsemax*. Primary source for sparsemax and its active-set Jacobian: https://proceedings.mlr.press/v48/martins16.html
- Peters, Niculae & Martins (2019), *Sparse Sequence-to-Sequence Models*. Primary source for entmax and sparse probability transformations: https://aclanthology.org/P19-1146/
- Official PyTorch profiler documentation. Shape/stack profiling incurs overhead and can retain tensor references; use profiler traces for attribution, not headline timing: https://docs.pytorch.org/docs/main/profiler

These sources support the normalizers and measurement caveats. They do not establish HONF physical generalization, causal interpretation, or GPU acceleration.

### Local algebra check accompanying this plan

A standalone CPU float64 check grouped queries by support signature and evaluated the same overlap-controlled environmental operator over source unions. It included one receiver with no environmental support. Dense/block maximum output and input-gradient differences were both approximately 5.6e-17 for the synthetic example (33 supported pairs versus 63 rectangular pairs). This verifies the algebraic regrouping only. It is not an HONF checkpoint test, an implemented repository executor, or a GPU speed result.
