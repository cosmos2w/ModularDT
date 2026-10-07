# HONF next-round research plan: compile the source-conditioned core before relearning conditional detail

**Starting evidence:** Run 1507 exact e500 and its selected e474/e460 checkpoints; repository `cosmos2w/ModularDT`, branch `agent/honf-core-next`, inspected head `151b12be01d837b5c1731eed944bcc211ea37bc2` (report continuation after `f482539`).

**Central goal:** Remove redundant interaction functions while preserving the physical information they organize.

**Primary decision:** Do not launch another expected-R penalty or anti-collapse experiment first. Derive and implement the exact all-closed source-conditioned reader, measure it on existing weights, identify which remaining source-conditioned mechanisms matter, and train one bounded clean-core candidate. Conditional query organization must subsequently demonstrate incremental physical value over this simpler reference. A nonconstant R histogram is not itself a scientific objective.

This plan supersedes the previous v5 plan for *new* work. Historical models, checkpoints, and the completed Run 1507 remain intact. The allocator determines the next free numeric run ID; do not overwrite or rename an existing run.

---

## 1. What the experiment series established

| Stage | Question actually tested | Observed lesson to carry forward |
|---|---|---|
| 1500 | Can source-mass competition reduce case-level capacity? | The handoff records the same two surviving prototypes on all 90 cases, nearly complete source support, and inferior accuracy/cost. Fewer groups can imply denser physical access. |
| 1501 | Can source organization and query-local sparsemax retain fine physical information? | Meaningful query sparsity and strong mature fidelity emerged, but all 12 source slots remained occupied and the fine reader remained rectangular. |
| 1502 | Can environmental incidence become sharper? | Environmental sparsemax reduced overlap with useful fidelity tradeoffs; it did not remove rectangular physical work. This remains the mature comparison baseline. |
| Original 1503 | Can coarse group values plus opened fine reads improve the tradeoff? | Extra coarse work, per-group normalization, and packing overhead did not deliver the intended replacement of fine work. Do not revive this additive branch. |
| 1504 / v2 | Can convex access-function fusion yield a source-preserving quotient? | The quotient/moment algebra worked. Fixed-iteration convergence and pre-merge descriptor displacement undermined the experiment. |
| 1505 / v3 | Does convergence-corrected, singleton-preserving fusion help? | Real case-dependent R=9–12 and improved e500 field error, but a trained ~3.05% design jump, port regression, and ~5x application cost. Preserve it as evidence, not as the deployment base. |
| 1506 / v4 | Can a smooth function transform remove the hard replacement? | Continuous-transform algebra was useful, but the passive score never entered its contraction region. It paid overhead without removing functions. |
| 1507 / v5 | Can physical loss plus expected-class pressure train coalescence? | All saved late models have R=1. Fidelity recovered by e500, but conditional organization did not. This reveals a useful static source-conditioned model hidden inside a costly conditional wrapper. |

These are distinct scientific outcomes, not an all-or-nothing chain of failed models. Exact compaction, fine-source preservation, sparse incidence, and the reusable physical loop are durable results.

The latest report changes the e150 fidelity judgment: at exact e500, equal-case five-field fluid L2 is 0.09336 versus parent 0.10568; pooled L2 is 0.09667 versus 0.10867. However, near-interface mean L2 is 0.10532 versus 0.09875, final T-port mean is 0.09762 versus 0.09257, and T-port maximum is 0.29128 versus 0.20088. Most of the port deficit is in M=3. Do not replace these separate quantities with one score.

All 90 `test` IDs have also been used for development validation. None of the above is an untouched generalization result. The 1507 e500 checkpoint is also its saved best-field checkpoint; e474/e460 selections are not matched to unavailable parent-best-within-500 weights.

## 2. The research questions for this round

1. Can the *observed all-closed operator* be specialized exactly so that absent query distinctions stop consuming work?
2. Which source-conditioned controls remain useful after query-to-group routing disappears?
3. Can a clean source-conditioned core recover the favorable field result while avoiding the near-interface and low-M T-port weaknesses?
4. Only after those questions: does a small amount of query-dependent control improve the physical Pareto tradeoff enough to justify its cost?

Do not infer that R=1 means one physical source, one uniform field, or no many-body information. Also do not infer that twelve source-control generators are redundant merely because their query-access functions have become equal.

---

## 3. Algebra of the all-closed model

### 3.1 Notation and assumptions

Suppress batch/phase indices in equations; code must retain them. Let K=12 be the registered proposal width, A^S_{sk} the provisional row-normalized source incidence for source type S in {M,E}, h_k the D=16 control, and nu^S_s the existing normalized physical source measure. In the surveyed Run 1507 checkpoints all K proposals are occupied.

The source encoder still computes A^M, A^E, h, and the provisional calibration

\[
\pi_k=\tfrac12(\mu_k^M+\mu_k^E),\qquad
\kappa=\left(\sum_k\pi_k^2\right)^{-1}.
\]

Do not recompute kappa using logical R=1.

When the tree root is closed, its transform is

\[
T=\frac1K\mathbf1\mathbf1^\top.
\]

All virtual access logits are the same scalar function of q. By translation invariance of sparsemax,

\[
\alpha_{qk}=1/K.
\]

The common logit is irrelevant to the physical prediction. The *low-dimensional query-to-group router* can be removed from this specialized operator.

### 3.2 Preserve the source-side sufficient controls

Define once per prepared phase

\[
u_s^S=\sum_k A^S_{sk}h_k,
\qquad
r_s^S=\frac1K\sum_k A^S_{sk},
\qquad
\bar n_s^S=\frac{u_s^S}{K}.
\]

Here `u` is a D-dimensional **control vector for each original physical source**, not a pooled physical field value. Retain every fine source state and coordinate.

The original reader contractions reduce to

\[
\rho^S_{qs}=r_s^S,\qquad n^S_{qs}=\bar n_s^S.
\]

For a valid unit-normalized incidence row, r_s=1/K. Keep the directly computed row-sum version initially to preserve padding and small floating-point normalization differences. No Q x S x K contraction is needed.

Every valid source has positive support in this limit. Thus R=1 removes query-group distinctions but **does not create physical pair sparsity**; it generally restores dense valid query-source access.

### 3.3 Exact module specialization

The existing fine QM message has the form

\[
\psi_{qi}=F_{\rm tail}\!\left(
\operatorname{GELU}(a_i+W_\delta\phi(q-x_i))
\odot [1+\tanh(W_M\bar n_i^M)]
\right),
\]

where a_i is the already-prepared first affine from module/global state. The gain

\[
g_i^M=1+\tanh(W_M\bar n_i^M)
\]

is now source-local and must be prepared once, not recomputed for every query.

Retain the exact output-bias and count calibration. With a_M=M_active/(1+M_active),

\[
C_M(q)=\kappa\left[
 a_M W_{M,\rm out}\sum_i\nu_i^M r_i^M\psi_{qi}
 +\sigma_M b_{M,\rm out}
\right],\qquad
\sigma_M=\sum_i\nu_i^M r_i^M.
\]

This formula follows from the inherited K-scaled finalizer followed by the sparse-incidence subclass's kappa/K scaling. Do not lose the output bias or multiply kappa/K twice.

For nonempty normalized module measures and exact unit incidence rows, sigma_M=1/K. For an empty module type, sigma_M=0; the module term must remain zero, including its output bias.

### 3.4 Exact environmental specialization

Let k_{jh}, v_{jh} be the original per-head projected fine K/V, d_h the head width, and q_h(q) the **physical environmental attention query**. This query remains indispensable and must not be confused with the removed query-to-group projection.

The source-local value gain uses the unscaled control:

\[
\hat v_j=v_j\odot[1+\tanh(W_Vu_j^E)].
\]

The multiplicative content-score gain uses the scaled control:

\[
g_{jh}^E=1+\tanh([W_E\bar n_j^E]_h).
\]

Since this gain is independent of q, fold it into the already projected key:

\[
\boxed{\hat k_{jh}=g_{jh}^E k_{jh}.}
\]

Perform this multiplication **after** source LayerNorm/projection. Moving it before nonlinear normalization changes the model.

The attention score becomes

\[
s_{qjh}=\frac{q_h(q)^\top\hat k_{jh}}{\sqrt{d_h}}
 +g_{\rm geom,h}(q,x_j)+\log(\nu_j^E r_j^E).
\]

Use the same valid-source mask and one normalized attention over the full source union. With sigma_E=sum_j nu_j^E r_j^E,

\[
C_E(q)=\kappa\sigma_E\,O_E\!\left(
\operatorname{concat}_h\sum_j\operatorname{softmax}_j(s_{q\cdot h})\hat v_{jh}
\right).
\]

The outside factor multiplies the **complete output projection, including its bias**. For a nonempty type with unit rows sigma_E=1/K; log(r_j) is then a common shift and may be removed after reference parity is established. Retain the general formula first.

### 3.5 What remains query dependent

The specialized model still has:

- fine nonlinear QM relative-coordinate interactions;
- physical QE query/key attention and learned geometry;
- source-conditioned gains and all fine physical source values;
- the global background C_g(q);
- the existing physical port/local-surrogate/refinement loop.

The final field remains `D(LayerNorm(C_g + C_M + C_E))` under the existing head. Rank one in the auxiliary query-group access distribution is not rank one in the physical solution operator.

### 3.6 Scope of exactness

There are three distinct comparisons:

1. Specialized source-local algebra vs the old reader **forced to the all-closed root**, with identical physical weights: an algebraic equivalence subject to floating-point ordering.
2. Specialized algebra vs the original Run 1507 at inputs where its deterministic controller actually closes the root: the same equivalence applies there.
3. Specialized algebra vs the original conditional model at unseen inputs that might reopen: **not guaranteed equivalent**. The new static profile deliberately has no reopening controller.

Report these separately. An all-90 R=1 survey is not a proof that the old controller is globally constant. The explicit new profile should be named `source_conditioned_pairwise_honf` or similarly; do not silently alter historical `task_trained_functional_coalescence_honf` semantics.

For cases with inactive proposal slots, first compare against an explicit uniform-over-active-proposals reference. In that reference use K_active in alpha and r/n, while preserving the inherited physical calibration. Record that as a specified extension of the static model, not proof of equivalence to an old nonclosed controller. Never renormalize by padded physical-module width.

---

## 4. Durable implementation

### 4.1 Preferred code structure

Add one thin source-conditioned backend, reusing the parent fine preparation and source organizer. Extract reusable helpers only where both parent and new backend need them; do not add another deep chain of v2/v3/v4/v5 subclasses.

Suggested responsibilities (names may follow existing conventions):

- `source_conditioned_pairwise.py`: source-local preparation and specialized QM/QE reads;
- existing `sparse_incidence_router.py`: retain the source preparation, separate source-only preparation from optional query-access construction as narrowly as practical;
- `core.py`, existing configuration/registry: one opt-in architecture identifier;
- existing physical wrapper: unchanged P0/P1/P2 and predicted-port behavior;
- focused reusable tests for exact specialization and physical wrapper integration.

Historical checkpoints continue to load through their historical classes. A local conversion helper may load the old model normally and copy the explicitly required source/fine/head parameters to the new profile. Enumerate the removed controller/query-only keys. Do not use a blanket `strict=False` to hide missing physical weights. Do not copy checkpoints into Git or create a new snapshot framework.

### 4.2 Computations absent from the new inference path

Remove from this profile:

- learned detail controller, stochastic samples, node descriptors, Haar transform;
- actual/expected-R traversal and complexity loss;
- repeated CPU extraction of tree child references;
- query-to-group logits, group-centre query-distance bank, and sparsemax;
- virtual/compact membership packing and per-query group-control contraction;
- diagnostic source-action probe evaluation.

Do not retain a dummy width-12 alpha tensor just to call the old read unmodified. Do not confuse a generic width-one wrapper with specialization: the wrapper still computes the obsolete functions before discarding them.

### 4.3 Computations retained initially

Retain A/h/kappa source preparation. Source query-key projections used only by the removed query router can be absent, but prototype group codes are still used in source preparation and cannot simply be deleted. Core coordinate/Fourier features may also serve the physical QE query or C_g and must remain where used.

Keep the contextual fine MM/ME/EM operations, every original physical state, source coordinates/measures, Stage-A local surrogate, heads, and physical feedback. Reuse projections across chunks only within the same live prepared phase. No detached cross-design cache.

### 4.4 Numerical and environment policy

Use the repository's actual PyTorch 2.6 / CUDA / FP32 environment and existing numerical settings. Do not update the environment or enable reduced precision to obtain a speedup.

The key-gain fold changes floating-point reduction order. Keep an unfused source-local multiplication reference and compare first gradients, then use the folded implementation if its measured discrepancies are acceptable at the already established numerical scale. A same-mode repeat gives a useful floating-point control. Do not claim bitwise identity where only tolerance agreement holds.

An optional SDPA implementation may use the learned geometry plus log source prior as an additive floating mask, `dropout_p=0.0`, and `is_causal=False`. Observe the actually selected backend; FP32 and arbitrary bias do not guarantee FlashAttention eligibility. Compare against the explicit matmul/softmax path. Keep whichever wins complete application time within measured parity. No new custom kernel is part of the default scope.

Module padding may be compacted once per case/bucket if profiling identifies it as material. Preserve canonical physical-module IDs and scatter outputs back for ports. Do not modify the physical geometry or drop any active source. Treat this as a separate exact implementation change and report it separately.

---

## 5. Stage A: compile existing Run 1507 before new training

Use exact e500 and, secondarily, e474/e460 with each checkpoint labelled separately.

### A1. Execute the specialization

First run CPU synthetic and real GPU physical tests. Then evaluate all 90 development cases at Q8192, predicted ports, using the same masks/normalization as the current report.

Compare:

- original automatic virtual model;
- original explicit compact model;
- forced-all-closed original model;
- specialized source-conditioned model.

Full all-90 parity is needed for the original vs specialized e500 comparison, not only three low-M examples. Add a bounded training-input panel spanning occupied proposal counts and M. Compare field, P0 raw ports, P1 outside temperature, final h/T ports, interface flux, internal/surface outputs, and selected parameter/input gradients.

The independent `check_all_closed_specialization.py` supplied with this plan checks the algebra on a synthetic CPU model. It is not a substitute for these repository/checkpoint/GPU tests.

### A2. Time what remains

At Q8192 and native 128 / matched 2048 inner chunks, measure preparation, full forward, prepared P2, and complete application separately. Use maps off; measure operation rows in separate untimed reads. Start with M=3 and M=10 cases, then the established all-90 application protocol when the implementation is final.

Use interleaved/reversed model order, at least five paired repetitions on the small panel, and the existing synchronized timing tools. Record total/incremental allocation and actual source-control/QM/QE calls. Do not add overlapping scope times or label CUDA-event time as complete CPU application cost.

For geometry-heavy execution, one small and one larger established synthetic shape can clarify scaling; those are execution tests, not extrapolated physical accuracy.

### A3. Observable completion condition

The specialized graph must actually omit the controller, group routing, and per-query control contractions, even if the end-to-end speedup is modest. The algebra promises removal of those calculations; it does not promise removal of the fine QE rectangle.

If this is still slower than the 1502 reader, use the measured profile to explain the remaining cost. Do not automatically start a new sparse-kernel project.

---

## 6. Stage B: identify the useful source-conditioned mechanism

Run bounded same-weight interventions on the compiled exact-e500 model. Start on a fixed, stratified training-input panel; include case 0647 and a small fixed development panel only for diagnosis, not parameter or threshold fitting.

The following are local diagnostics, not new production branches:

1. Set the QM source modulation to identity, leaving all physical module messages intact.
2. Set the QE content-score gain to identity, leaving keys/values/geometry intact.
3. Set the QE value gain to identity.
4. Replace source-local u_s by a case/source-type mean, retaining original fine source states and measures.
5. Replace kappa by a declared constant only as a diagnostic; do not silently make this the new model.

Report intervention-minus-normal **ground-truth error**, not just output change. A large output change does not prove a component improves accuracy; a small change alone does not prove that its training role was unnecessary.

Where phase separation is useful, perform P0-only, P1-only, and P2-only interventions with all dependent downstream physical states recomputed. In particular, a P2-only field change must not be described as improving an already-computed P1 final port.

Use these results to decide what to retain. Default is to retain all source controls and kappa. Remove a source mechanism from a fresh model only after actual evidence and a bounded optimization test show that the simpler alternative is credible.

### Physical interpretation

The remaining source encoder may still use twelve latent control generators. Do not call it a physically meaningful twelve-edge partition solely because it uses A and h. Check source-local controls under feasible changes of geometry, operating input, and heating/material parameters, and compare predicted physical responses against available targets or independently available simulation output. New solver data are optional if already available and authorized; never imply that surrogate finite differences are CFD truth.

---

## 7. Stage C: resolve the port/near-field tradeoff before adding another router

The e500 weakness is specific: near-interface field error and final T-port tail, especially M=3. Preserve that focus while retaining fluid-field and far-field gains.

### C1. Audit the existing supervision, without assuming a bug

Inspect the resolved loss configuration and actual tensor semantics of:

- `port_condition_loss` and nonzero effective port-supervised weight;
- final vs raw port targets and temperature scaling;
- phase-global consistency and its detach direction;
- physical interface/local-temperature losses;
- per-module/point normalization versus equal-case evaluation;
- batch/bucket weighting across M and error tails.

The inspected source already contains a direct scaled T-port term when configured. Do not assert that port supervision is missing without checking the effective configuration. Per-module averaging is not automatically wrong; measure its contribution to the reported equal-case objective.

Check target consistency on training inputs: the port temperature target, outside-temperature field target, local-surrogate interfaces, and existing normalization may encode different sampled quantities. Do not force equality where the data definitions differ.

### C2. Bounded optimization pilots

Codex may use **at most three pilots of 200 real optimizer updates each, 600 total, with a three-GPU-hour ceiling** excluding the one full formal run. Use the same initialization, optimizer state, ordered training examples, and evaluation panel for recipes intended to be compared directly.

Recommended pilot reference: copy the same static Run 1502 e50 state into the candidate variants in memory. These are learning diagnostics; their weights must not initialize the fresh formal run. Preserve the copied optimizer moments for shared parameters where the existing machinery supports a correct explicit mapping, otherwise reset all compared pilot optimizers consistently and label it.

Permitted hypotheses, in order:

- **Pilot A:** the source-conditioned static reader, inherited loss and optimizer.
- **Pilot B if immediate removal causes a large transient:** same weights/loss, but a deterministic continuous transition from parent logits to uniform access over the first half of the pilot, then fully static. No controller or expected-R penalty.
- **Pilot C only for a diagnosed fidelity issue:** change one physically justified training choice, such as an explicitly per-case-normalized T-port term if the audit shows a mismatch with the desired risk. A bounded head/last-layer refinement under existing losses is also permitted. Do not add multiple corrective field branches or jointly sweep several loss weights.

Pilot outcomes are physical reconstruction, final-port and near-field means/tails, gradient health, and actual time. A short pilot is not an estimate of mature performance. It is valid to omit unused pilots. Do not rank recipes with different initial checkpoints as if they were controlled alternatives.

If inherited static training is credible, prefer it. A deterministic warmup is a training recipe, not evidence of case-adaptive organization. Any changed temperature weighting must be reported as a separate intended change, not as an exact implementation fix.

---

## 8. One fresh bounded formal core experiment

Allocate one new run with a descriptive name such as `source_conditioned_pairwise_core`. Keep Run 1507 untouched at e500.

Default recipe:

- source-conditioned static reader from initialization;
- original 600/90 split, seed 0, fine widths, K=12 **source-control capacity**, D=16;
- inherited fine physical preparation, predicted ports, losses, optimizer, precision, and training chunk;
- no query-to-group controller, no expected-R term, no online tree/ADMM/Gram probes;
- no warm start from mature 1507 or 1502 neural weights.

Reuse ordinary initialization helpers; preserve seeded values of shared components where practical without creating a snapshot/freezing infrastructure. A parent construction followed by explicit removal of query-only modules is acceptable if needed to retain initialization order. Materialize lazy tensors through actual execution, then count parameters accurately.

If a pilot supports deterministic warmup or a single port-risk adjustment, document and select that recipe before the formal run. Warmup may keep temporary parent routing through e50, blend to uniform by e150, then use the specialized static graph. Do not describe a warmed-up static model as dynamically coalescing at e500.

### Epoch 50

Review actual training and physical outputs. Report source-control use, source/pair work, parameter/update health, and complete cost. If a deterministic warmup was selected, separate its cost from the final static operator. Do not reject a healthy learning model solely because it trails an e50 scalar: the 1507 continuation demonstrated substantial later recovery.

### Epoch 150

Use the full-grid same-stage parent checkpoint and retain the complete metric vector. Inspect source modulation and M-stratified T-port/near-field behavior. For this *static* experiment, R=1 is the specified architecture, not a collapse diagnostic. Do not waste time searching for a topology transition that the profile cannot have.

Stop early only for a concrete uncorrected implementation/numerical failure or compelling persistent physical regression after bounded remedies. Otherwise continue the one candidate to the e500 research ceiling, because the key question includes learning without the old controller curriculum.

### Epoch 500

Stop and evaluate exact e500 and each distinct saved selection separately. Pair organization/cost with the same weights; identify cross-epoch or unmatched selection-policy comparisons. Compare at least:

- Run 1502 exact e500;
- original Run 1507 exact e500;
- its exact specialized static replay;
- the new freshly trained static candidate.

Keep Dense/1404 as separate cost/fidelity context only when matching measurements are available. Do not mix mature 5000-epoch errors with fresh e500 errors into a single model ranking.

Report equal-case and pooled field errors, near/far, individual fields, P0/P1/final ports, flux, module/surface temperature, M strata, p90 and maximum tails, cost, and retained trainable/total parameter counts.

A 5000-epoch recommendation is a separate scientific judgment. It may be justified for a clean, efficient static model with credible physical quality even though R is constant. It is not an adaptive-hypergraph claim. Do not launch it automatically.

---

## 9. Conditional detail may be added back only after it earns a role

This is a bounded diagnostic option, not a second default formal run.

If the clean static model retains a specific port/near-field deficit after the preceding work, test whether *query-dependent control* can fix it while holding the source-conditioned physical core fixed. Do not reopen old gates and interpret output movement as a trained benefit; the unused routing functions are no longer coadapted to the physical weights.

One controlled alternative is a single learned contrast from the existing proposal basis:

\[
\alpha(q)=\operatorname{sparsemax}(\zeta_\theta(q,c,p)\,h_n),
\]

where h_n is a fixed zero-sum proposal contrast and zeta is a small continuous scalar access function. At zeta=0 the route is uniform. It still modulates the same source overlaps/moments and uses the same single physical fine read. There is no additional field-value path.

Choose any contrast type using training evidence, not the worst development-case label. Refit its query function with physical losses; do not use an arbitrary forced-opening coefficient as the conclusion. One such diagnostic can consume the third pilot slot, not expand the total pilot budget. Compare against a same-compute static/head-refinement control and the best static checkpoint.

Only if a trained detail improves the relevant field/port risk without unacceptable regressions should the *next round* study conditional selection. That next study should avoid the root-suppresses-all path: use independently trained physical corrections or policy exploration that does not starve every descendant when one ancestor closes. It must compare against a best static policy, charge measured executed work rather than just R, and preserve continuous representation changes.

If no detail yields material benefit, accept the static simplification. Do not impose minimum R, entropy, or R variance simply to draw a diverse graph.

---

## 10. Tests and empirical evidence

### Reusable tests

- All-closed virtual vs compact vs source-local specialized rho/n, QM/QE, field, and first gradients.
- Bias and kappa placement; correct u versus u/K distinction for value and score gains.
- Post-projection key folding, compared with score multiplication in the reference order.
- Mixed active/padded modules, zero source-type measure, nonuniform source weights, and unequal query chunks.
- Physical module permutation with canonical output restoration.
- No omitted source/head parameter during explicit old-to-new conversion.
- No query-to-group gradient in the forced all-closed operator, while physical query-coordinate and source gradients remain live.
- CPU import and ordinary PyTorch fallback remain usable.

### Real execution

- Actual low-M mixed and high-M predicted-port optimizer batches, not mocks.
- Q8192 physical parity on all90 frozen e500 cases.
- Small feasible coordinate and operating sweeps with fixed fluid queries, and AD/FD checks away from inherited singular/mask boundaries.
- Target-based evaluations and repeated synchronized timing.

A static reader has no learned merge/split event. Its ordinary differentiability still depends on inherited source masks, normalization and geometry; do not claim a global continuity theorem. Likewise, successful synthetic tests do not replace physical wrapper execution.

### Required causal distinctions in the final report

- Frozen specialization vs retraining: only the former isolates algebraic simplification.
- Input dependence of source controls vs conditional graph topology: the static model can have the former without the latter.
- Removal of query-access work vs skipping physical pairs: these are different operations.
- Output perturbation vs useful contribution: target error is needed for the latter.
- Training-time role vs inference-time dependence: an inactive endpoint component may have influenced training.

---

## 11. Bounded problem-solving autonomy for Codex

Work actively rather than escalating ordinary implementation issues. A concrete failure should lead to a reproducer, a causal explanation, and an empirical alternative.

Within the pilot budget, reasonable remedies include:

- keeping the source-score multiplication reference if key folding exceeds numerical tolerances;
- changing operation ordering or batching, rather than changing model semantics;
- explicit PyTorch attention instead of an unsupported/slow fused backend;
- source-local gains and fixed shapes before trying compiled execution;
- narrowing an incompatible checkpoint conversion by enumerating actual required keys;
- a single physically justified loss/reduction correction after measuring the mismatch;
- deterministic curriculum if abrupt all-closed access gives a large optimization transient.

Use at most two materially different remedies per ordinary blocker before documenting it; target at most 45 minutes on one noncritical implementation blockage. These are time budgets, not new runtime safeguards. Finish independent useful work rather than looping on the blocked path. Do not exceed 600 pilot optimizer updates, three GPU-hours of pilots, or one new 500-epoch formal run without a new request.

Do not run hundreds of epochs with an undiagnosed inactive mechanism. Do not stop a physically learning static candidate merely because it lacks the dynamic organization it no longer claims to learn.

User-facing closeout should emphasize completed changes, measured results, and genuinely unresolved research/technical limitations. Keep ordinary resolved debugging details in local logs or a compact note; do not hide adverse experiments or relabel failed accuracy/cost evidence as success.

---

## 12. Preserve the repository's existing rules; add no defensive infrastructure

Read and follow current `AGENTS.md` and `.githooks/pre-push` before writes/pushes. Use the existing run/checkpoint machinery and security behavior.

Commit durable source, reusable tests, the selected config, and the written research report on the working branch. Keep one-time diagnostic scripts, figures, measurements, and checkpoints in ignored local paths. Audit the entire outgoing commit range and verify local/remote tips after an ordinary push as the standing rule requires.

Do not add new cryptographic hashes, contract freezes, baseline snapshots, approval workflows, monitoring daemons, or arbitrary blocking gates. Do not weaken the existing hook to upload a report package. No force-push or destructive historical-run cleanup is authorized.

These instructions do not remove standard tests or numerical error handling. Scientific reviews decide whether additional experimentation is worthwhile; they are not a new production/security framework.

---

## 13. Deliverables and claim boundaries

Deliver:

1. A readable source-conditioned backend and explicit conversion/replay path.
2. Focused reusable tests and actual GPU/physical parity results.
3. Frozen all-90 exact-e500 fidelity and execution comparisons.
4. Source-control and port/near-field diagnoses with physical target metrics.
5. Up to three bounded pilot records and one selected fresh-core recipe.
6. One completed or scientifically stopped fresh run, reviewed at 50/150 and at most 500.
7. A report that answers what source organization remains necessary and whether a learned query distinction improves on the best static reference.

The near-term success is **not** a prescribed R distribution. It is a smaller executed model, with its physical information preserved and its remaining accuracy/cost tradeoffs measured.

The long-term objective can be expressed conceptually as minimizing executed cost subject to declared field/interface/design-fidelity requirements. Do not equate `minimize expected R` with that problem. Set any physical tolerances from the task and available empirical uncertainty; do not invent a tolerance solely to accept a candidate. This round needs no new constrained optimizer to express that scientific preference.

### References and inspected implementation basis

- `HONF_Run1507_Task_Trained_Functional_Coalescence_Diagnostics.md`, especially “Epoch-500 organization and physical fidelity” and “Exactness, native boundaries, and complete cost at e500”.
- Prior source reports for Runs 1500–1506; distinguish their epochs and metric definitions.
- `interface_fields/task_trained_functional_coalescence.py`: `prepare`, `_route`, compact moment preparation, automatic virtual policy.
- `organization/learned_functional_detail.py`: ancestor products, expected-class recursion, deterministic/stochastic retention, inference-time complexity calculation.
- `interface_fields/group_control_pairwise.py`: `_module_psi`, `_module_finalize`, `_read_module_complete`, `_read_environment_complete`.
- `interface_fields/sparse_incidence_group_control.py`: final kappa/K factors.
- `Case_ThermalChannel/src/channelthermal/training/epoch.py`: configured physical loss assembly and port normalization.
- Louizos, Welling, Kingma, *Learning Sparse Neural Networks through L0 Regularization*, arXiv:1712.01312. This motivates the historical gate distribution, not a guarantee of conditionality or HONF speed.
- Official PyTorch 2.6 `scaled_dot_product_attention` and CUDA timing documentation. Backend availability and numerical ordering must be checked in the actual environment.
