# HONF: fidelity-budgeted joint forward maturation and an adaptive interaction interface

## 0. Purpose, starting point, and scope

**Working branch:** `agent/honf-core-next` in `cosmos2w/ModularDT`.

**Reviewed implementation:** `25196daef0f9e9fe0d59342c5577db3a927c6410`. The inspected branch tip was `5b7167d5d7715546842194e8a4f8fddee9df15bf`; its only change from `25196da` is the illustrated study report. Use current Git history to account for any subsequent changes; these identities document the review, not a new contract-freeze mechanism.

**Primary evidence:** `HONF_Proj/docs/reports/_bk/20260927_101609Z_HONF_Receiver_Local_Interface_and_Decision_Recovery_Study.md`, especially Sections 2, 4, 6, and 7. Historical reports remain evidence for their own experiments. Do not relabel an old checkpoint or rewrite its outcome.

### Central goal

Learn a reusable, physically indexed organization of module–environment interactions through multi-physics prediction, and determine whether this organization preserves the responses needed for inverse design.

This round must move from **training a pruning controller on a frozen predictor** to **maturing a jointly adapted forward operator and its interaction interface under explicit fidelity budgets**.

A small K, fewer logical pairs, a teacher-gate pass, and a lower weighted training loss are not substitutes for that result. Conversely, a prototype need not immediately beat Dense in latency to justify a structural learning experiment.

### Main decisions

1. Preserve intact Thermal Run1804 selected e4738 and WindFarm Run2103 selected e2475 as frozen references and available execution fallbacks. Preserve Run1502 selected e4794 as the Thermal sparse-incidence comparator.
2. Do not promote or simply continue the current G5 u650 organizer or Thermal R1 u200 response arm.
3. Make WindFarm the primary joint-forward experiment: its stored native fields support broad value training and whole-grid evaluation now. The uploaded `wind_farm_volume_v1` resource contains cell-centred OpenFOAM CFD fields; distinguish this existing reference dataset from new CFD solves and from the ThermalChannel local analytic-wake-plus-thermal generator.
4. Reuse the native encoders, fine messages, typed permissions, physical measures, and existing output heads. Do not introduce another replacement field head, coarse/fine value branch, or early pooled-source representation.
5. Replace the fixed weighted pressure to prune with fidelity-first training and an explicit role-wise error-budget formulation.
6. Give omitted QE sources a usable training signal. A straight-through binary logit followed by a hard log-prior mask does not supply that signal by itself.
7. Compile an execution view of a prepared plan once. Do not re-walk the anchor tree and rebuild static diagnostics in every receiver chunk.
8. Continue Thermal response development as a bounded secondary task with broader historical replay and a tested nonlinear interaction adaptation, not another repetition of the identical eight-family head-only recipe.
9. Prepare inverse use around a fixed, evaluated forward version, named decision quantities, live trial states, baseline correction, and independent acceptance. Do not train a separate inverse network this round.

---

## 1. What the completed study established

### 1.1 Retain these achievements

- Typed MM/ME/EM/QM/QE permissions are now integrated into native computation rather than being a visualization-only layer.
- All-access native identity, phase matching, current trial-state recomputation, source-permutation behavior, and important measure checks exist.
- A genuinely trained input-only organizer was executed. Its u650 hard plans passed the stated Q1024 teacher criterion on all 12 organizer-development directions.
- The learned support reduced exact counted native pairs by 53.5% on those probes. This is a real logical-work result, not a measured acceleration.
- Native physical-reference evaluation, whole-grid evaluation, and teacher distortion are reported separately.
- Per-module baseline correction and the distinction between a historical candidate union and a new matched-policy experiment are implemented.

### 1.2 Do not overstate these achievements

- The final QE packet count was K=2 for every reviewed development row. Input-dependent source support is demonstrated; useful deterministic adaptive K is not.
- Other mechanisms retained full valid support. The current result is predominantly environmental read selection, not a learned module-interaction topology.
- The two QE source sets have Jaccard overlap 0.859–0.981. They are not two independently isolated physical subsystems.
- The learned model already had reference-volume RMSE 0.025108 versus 0.012010 m/s at Q1024. Whole-grid failure was not the first indication of a fidelity deficit.
- Complete-grid checks failed all five reference roles on both selected layouts.
- The 15.04-second packed timing is for the saved K=1/E480 control, not the learned K=2 organizer. Neither timing includes evidence of deployed learned-plan speed.
- ThermalChannel local reference means the analytic-wake/shared-grid thermal generator; WindFarm native-grid reference means the existing uploaded OpenFOAM CFD fields. No new CFD or Thermal physical-reference solve occurred in the completed round.

### 1.3 Why the existing weighted loss is the wrong continuation target

The inspected typed predictive loss is

\[
L = L_{teacher}+0.25L_{reference}+0.25L_{label}+0.01C_{work}.
\]

All native predictor parameters were frozen; only organizer parameters were optimized. The same per-row Q1024 search probes were reused during predictive training.

The report shows total loss decreasing while teacher and reference losses increased and work decreased. Thus the selected weighted objective accepted a worse physical predictor in exchange for support reduction. More iterations of that objective are not equivalent to forward maturation.

The current 0.10 teacher criterion is in checkpoint-standardized channel units. It is not “at most 10% additional physical-reference error.” Retain it for reproducing historical results, not as the primary new accuracy criterion.

---

## 2. Research questions for this round

**Q1: Co-adaptation.** Can the native fine operator recover accurate field and receiver quantities after useful interaction omissions, when its own parameters are allowed to adapt?

**Q2: Reversibility of learning.** Can training restore an omitted source when restoring it reduces the actual prediction error? Test individual permissions, not only a nonzero aggregate pair-head gradient.

**Q3: Adaptive organization.** At the same declared fidelity tolerance, do different designs or operating contexts require different nonredundant interaction packets? Does this survive comparisons with a shared-root source subset and an equally trained direct-pair control?

**Q4: Execution.** Can an already prepared full-access or partial plan run without repeated tree traversal, host synchronization, and static accounting inside the query loop?

**Q5: Decision readiness.** Does a selected forward/interface version predict useful finite changes and constraint margins, rather than only absolute values or self-consistent AD derivatives?

The questions are related, but an unresolved Thermal mesh-sensitive label must not automatically prevent WindFarm value/organization training. Likewise, a slow exact executor must not prevent measuring structural fidelity.

---

## 3. Minimal architecture: a jointly trained native packet operator

Write the forward model as

\[
\widehat y(q;d,c)=F_{\theta}(q,d,c;H_\phi(d,c)).
\]

- `theta` contains the native physical encoders/messages/updates/readers/head parameters selected for co-adaptation.
- `phi` contains the input-only organizer.
- `H` contains typed receiver-local source permissions and a receiver partition/cover.
- The frozen incumbent `B` is a separate reference; its predictions are never inputs to `H` at deployment.

A packet is a directed object

\[
e=(\tau,p,\mathcal R_e,\mathcal S_e,a_e),
\]

where `tau` is mechanism, `p` is physical phase when relevant, `R_e` is receiver support, `S_e` is a source set, and `a_e` is receiver access. Source identity means the actual module identity or the adapter's coordinate/feature/measure convention, not an arbitrary tensor column regarded as globally physical.

Retain

\[
w_{rs}^{\tau,p}=\sum_e a_{re}^{\tau,p}b_{es}^{\tau,p},
\quad a_{re}\ge0,
\quad\sum_e a_{re}=1.
\]

Each fine source contribution is evaluated once per supported receiver-source pair. Source quadrature and access remain separate until the native normalization requires their product. Environmental attention has **one normalization over the source union**, not one softmax per packet.

For the attention path,

\[
A_r=\frac{\sum_s \mu_s w_{rs}e^{\ell_{rs}}v_s}
{\sum_s \mu_s w_{rs}e^{\ell_{rs}}}.
\]

For additive native messages, keep their current full-model denominator unless a separately identified architectural experiment changes it. Do not renormalize by retained source count merely to hide missing-message amplitude.

### What remains unchanged initially

The Dense common/coarse/local paths and field head are retained at initialization. The model must reproduce the teacher with full access before optimization. This is a function-preserving insertion, not proof that the graph covers every information route.

The first co-adaptation scope is:

- train fine MM/ME/EM message/update modules;
- train fine QM/QE read modules and existing field-head layers;
- train the organizer separately;
- initially freeze global/source encoders and the existing coarse/local context builders;
- keep normalizers, units, dataset adapters and reference definitions unchanged.

A bounded pilot may broaden the encoder scope if the first scope cannot recover reference fidelity. It must broaden the full-access control identically. Do not introduce new trainable coarse/local bypass capacity to compensate for omitted fine routes.

### Many-body interpretation

The nonlinear receiver update following selected messages remains capable of collective source dependence. Packet arity and a nonlinear update are not proof of irreducible physical many-body interactions. Preserve source-resolved information and test collective response separately where reference evidence exists.

Do not add an extra pooled `h_e -> field value` branch in this round. Existing source-control quotient machinery is retained for historical architectures; it is not required to be reinserted into a Dense model that did not use it.

---

## 4. Repair the missing-source learning signal

### 4.1 Inspect and reproduce the issue

In `AdaptiveCoverPairwiseField._read_cover_dense_masked`, an environmental prior equal to zero is clamped before a logarithm and masked to `-inf`. For a fixed support pattern, an omitted source therefore has no direct local task derivative through its own QE log-prior. A straight-through sigmoid used before this operation does not undo that downstream derivative cut.

Shared scorer parameters and supervised labels can still reopen a source indirectly. The claim is not that all learning globally stops. The problem is that the prediction error does not locally tell this omitted QE permission to restore itself.

Add a focused test on a small actual attention read and a real native prepared case:

1. Omit one source whose restoration measurably reduces target/teacher error.
2. Measure the permission-logit gradient under the current implementation.
3. Measure a finite restore intervention and its actual loss change.
4. Record which mechanisms retain a local permission gradient and which do not.

Do not equate “pair-head gradient norm is nonzero” with “all omitted sources are recoverable.”

### 4.2 Primary remedy: hard physical values with an explicitly soft organizer gradient

Use an exact hard forward for training values and for physical-parameter gradients. Supply a separate continuous shadow only for organizer gradients.

Let `h` be the detached deterministic hard plan, and `p_phi` the soft plan. Define

\[
y_h=F_\theta(d,c;h),
\]

\[
y_s=F_{\operatorname{stopgrad}(\theta)}(d,c;p_\phi),
\]

\[
\boxed{y_{train}=y_h+[y_s-\operatorname{stopgrad}(y_s)].}
\]

Use the parentheses exactly: the bracket is identically zero in the forward value for finite tensors.

This gives:

- exact hard-plan forward values;
- physical-parameter gradients only from the hard forward;
- organizer gradients from a soft path with positive prior support;
- no soft/shadow path at inference or inverse evaluation.

All model parameters, buffers, and encoded features used in the shadow must be detached from `theta`, while organizer scores remain live. Detach design coordinates, operating inputs, receiver coordinates and derived query features in the shadow as well, so it cannot add an unintended second design/input gradient. Do not wrap the shadow in `torch.no_grad()`: its operations must still track the organizer tensors. Construct organizer features from current pre-interaction inputs, but detach their path back into `theta` for the organizer update. The hard plan must not carry an additional straight-through path into `phi`, otherwise gradients are double-counted.

A functional parameter call, a narrow explicit functional reader, or a tested custom VJP is acceptable. Do not mutate a live module's `requires_grad` flags between unfinished forwards. Preserve evaluation-mode dropout/buffer behavior and checkpointing RNG consistency.

This is a **surrogate topology gradient**, not an exact derivative of a discrete plan. Report it as such. Prove in tests that physical parameter gradients equal the hard-only reference, and show actual native reopening interventions during training.

### 4.3 Bounded alternative

If the shadow bridge is too expensive or unstable, alternate:

- hard-plan physical-parameter updates; and
- organizer-only continuous-plan updates with the physical parameters frozen.

Evaluate the deterministic hard model every review and quantify its soft/hard discrepancy. A small number of source-restoration counterfactuals can check whether the surrogate direction is useful. Do not silently deploy the soft model, invent an unbiased estimator claim, or replace the physical output by a second learned predictor.

At most one of these gradient recipes becomes the formal candidate.

---

## 5. Change the data coverage and the meaning of fidelity

### 5.1 Use native training support, not one immutable Q1024 stencil per row

Extend the existing `windfarm/workflows/train_forward.py` and native dataset/sampler rather than building another large single-run workflow around archived diagnostics.

Use the teacher's original training layouts, excluding the four previously reserved organizer-development layouts from **new** student/organizer updates. Their prior exposure to the teacher is disclosed. Do not train on those four layouts and continue calling them held organizer evidence.

The original native validation rows remain development data; they have been reviewed repeatedly and are not a new untouched final test. Previously opened final/reference trial pools remain retrospective. Do not inspect a reserved final split merely to select this candidate.

Sample new native cells at each update. A reasonable initial per-case allocation is:

- 4,096 volume queries;
- 1,024 hub-slab queries;
- 1,024 downstream-envelope queries;
- 1,024 near-turbine queries;
- 1,024 background queries.

Reduce batch size before reducing protected-role diversity. The exact counts may change for available memory, but both matched arms must use the same draws. Role overlap is intentional in the role-balanced objective; keep a separate volume estimator using the appropriate volume sample/weights. Do not describe the role mixture as an unbiased volume integral.

Respect native quadrature and sampling probabilities. Query roles/targets guide training and evaluation, not the input-only organizer. Evaluate on deterministic audit points distinct from the training draws, and stream complete native grids at the scheduled reviews.

Cache only static coordinates/indices/targets or frozen-teacher outputs for an explicitly fixed query set. Do not cache a trainable student's encoded states or physical preparation across optimizer steps.

### 5.2 Calibrate in physical units and reference-error space

For role `r`, use a consistently defined physical-unit RMSE `E_r`. For the first controlled comparison retain the report's transparent reference budget:

\[
B_r=(1+\delta_r)E_r^B+\tau_r,
\qquad \delta_r=0.10,
\qquad \tau_r=10^{-5}\ \mathrm{m/s},
\]

where `E_r^B` is the intact incumbent error evaluated on the applicable training calibration measure. These are research-comparison tolerances, not engineering safety limits.

The first budget deliberately retains the previous whole-grid allowance. Do not silently widen it after seeing a failed candidate. A separately labelled Pareto sensitivity may show a looser tradeoff, but must not replace the primary comparison.

For optimization define

\[
g_r=\frac{L_r^{reference}}{B_r^2}-1,
\quad L_r^{reference}=E_r^2.
\]

Use training-only calibration batches for risk updates. Compute the budgets on the same role measure as the loss. Report aggregate role risk and the distribution of per-layout budget exceedance; a good population mean must not hide systematically bad high-M layouts.

The 0.10 standardized teacher metric remains a secondary diagnostic. Teacher distillation can protect a useful incumbent, but physical-reference loss is primary where stored native labels exist.

### 5.3 Why this differs from the old criterion

A bound on `||student-teacher|| / channel_std` is not a bound on `(student reference error)/(teacher reference error)`. A highly accurate teacher can tolerate a small absolute prediction perturbation that nevertheless doubles its reference error.

The triangle inequality is a conservative interpretation tool, not the new training objective. Evaluate the actual reference loss directly; do not infer it from teacher closeness.

---

## 6. Fidelity-budgeted training, with physical co-adaptation

### 6.1 Matched arms

Use one pair from the same native Run2103 initialization:

**W-full:** complete native access, trainable physical scope and fresh native sampling.

**W-packet:** the same physical scope plus the typed organizer and hard permissions.

Same target data, sampling order, physical-parameter optimizer, update count, normalization and validation policy. Count W-packet's extra teacher/shadow/counterfactual calls separately rather than pretending its compute equals W-full.

Historical G5 u650 remains a frozen-organizer control. It does not initialize the primary student: its aggressive omissions already violate the reference budget.

### 6.2 Stages

**Updates 1–150: conservative initialization.**

Keep hard access full while checking native training and warming the organizer with coherent observed labels. Unobserved candidate decisions remain unknown, not negative. Do not use class-balanced BCE plus threshold 0.5 as a claim of calibrated source-retention probability. Report false omissions and label provenance. Prediction parity is required only before optimization; subsequent matched arms may diverge by design.

**Updates 151–500: fidelity recovery under a mild demonstrated cover.**

Apply the coherent G2 cover on the labelled training subset, or a conservative organizer prediction where applicable. Keep unlabelled cases at full access until the input organizer has predictive evidence. Train the physical student to recover the reference targets. Set the work incentive to zero. This is the crucial experiment missing from G5: can the fine operator adapt to actual omissions?

**Updates 501 onward: constrained joint refinement.**

Release organizer decisions, with the missing-source gradient remedy. Optimize work only while explicitly protecting reference error. Keep a nonzero proportion of full-access replay batches so source paths remain trainable and rare interactions are not erased.

A practical primal-dual proposal is

\[
\mathcal L_{\theta,\phi}
=
L_{reference}
+\beta_TL_{teacher}
+\beta_HL_{native\ context}
+\eta C_{proxy}
+\sum_r\lambda_r g_r,
\]

\[
\lambda_r\leftarrow[\lambda_r+\eta_\lambda\widehat g_r]_+.
\]

- `L_native context` is optional weak matching of the existing fine-route contexts, initially useful to detect/limit a student abandoning the organized route in favor of an unmasked common branch. It is not a new field-value bypass.
- `C_proxy` supplies a differentiable direction; actual hard pairs, packet count, padding, and wall time are measured separately.
- Risk multipliers use measured hard-forward losses, even when organizer gradients use the soft shadow.
- Normalize multiplier updates, smooth only with an explicit short running estimate, and retain the raw losses.
- If budget violation persists over repeated training audits, set the work incentive to zero and recover fidelity; do not keep sacrificing fidelity until a scalar loss looks good.
- Initialize coefficients from a bounded training calibration and record them as hyperparameters, not as guarantees.

Naive primal-dual training of a nonconvex network has no automatic feasibility or convergence guarantee. Treat this as a testable recipe. If it oscillates, the permitted remedy is simpler alternating fidelity recovery and a small support-reduction proposal with refit, not another optimizer framework.

### 6.3 Work objective and the QE dominance problem

The old sum of all physical pairs is dominated by QE at large Q. It can reward only environmental source subsampling while leaving module communication unchanged.

For structural learning, report both:

\[
C_{all}=\frac{\sum_\tau W_\tau}{\sum_\tau W^{full}_\tau},
\qquad
C_{balanced}=\frac1{5}\sum_\tau\frac{W_\tau}{W^{full}_\tau}.
\]

Use a mechanism-balanced proxy during the structural phase, with absent mechanisms omitted. This is an explicit structural objective, **not measured FLOPs or latency**. Do not force a minimum omission in any mechanism.

A module-bearing route must be meaningfully examined. A final result that only reduces QE remains an environmental-read organization result, even if aggregate pair reduction is large.

### 6.4 Give maturation a meaningful opportunity

Use reviews at updates 150, 500, 1,500, 3,000, and the authorized endpoint. Report complete training-layout passes, fresh query coverage, and actual optimizer calls. Do not equate updates with epochs.

Allow up to 6,000 optimizer updates per formal Wind arm, or 500 actual native training epochs, or the round wall budget, whichever occurs first. No automatic 5,000-epoch job is authorized.

Do not stop merely because an early held metric is slightly worse. Continue while train and development fidelity are recovering under the unchanged scientific recipe. Stop or switch to the bounded remedy if losses are persistently deteriorating, the mechanism is disconnected, or the experiment only buys work reduction by consuming fidelity.

A budget stop is an incomplete maturation result, not proof of an architectural capacity limit.

---

## 7. Adaptive K: select resolution, do not prescribe a histogram

### 7.1 Preserve the definition

Report separately:

- candidate tree capacity;
- active frontier nodes;
- nonredundant typed packets after identical actions are combined;
- positive packet support per query;
- union of physical sources and exact pairs;
- executed width/rectangles and unmasked routes.

K is a case-local count at a fixed declared tolerance, not the physical module count and not an inverse-participation statistic.

### 7.2 Explore splits without initially perturbing physics

For a node with membership `b_n`, split it and copy `b_n` to both children. Since child access sums to parent access, the resulting source prior is unchanged. This is a function-preserving way to make finer candidate structure available.

Then learn different child supports from prediction evidence. Do not initialize children by deleting all sources across a geometric boundary. Keep source coordinates/states fine.

Perform bounded train-only split trials beyond the root on regions where current receiver error or source-importance heterogeneity suggests a distinction. At least one deeper refinement and one root-union/merge control must actually execute. This is exploration, not a requirement to retain those extra nodes.

Initially keep the existing geometry tree as a candidate index. If the current shared tree demonstrably prevents different mechanisms from using appropriate receiver resolutions, permit mechanism-specific frontiers over the same geometry index as one bounded remedy. Do not replace the entire tree algorithm and training objective simultaneously.

### 7.3 Canonical equivalence

For current permission-only packets with identical `b_e` and no distinct packet value/state, combine their access functions:

\[
\bar a_{qC}=\sum_{e\in C}a_{qe}.
\]

This leaves `w_qs` exactly unchanged. Once a future architecture assigns different packet-control states, equality of source masks alone is no longer sufficient for this quotient.

Do not use rounded latent vectors or approximate mask equality to claim exact compaction. Similarity-based structural changes are new predictive approximations and require an actual error check.

### 7.4 Required organization comparisons

At selected checkpoints evaluate:

1. Learned packets.
2. The same per-case global source union applied as one root packet.
3. Full access with the **same student weights**.
4. Intact incumbent and the matched W-full refit.
5. A learned direct-pair scorer or an accurately trained simpler source selector at matched logical work, on a bounded panel.

The existing geometry-ranked and slot-indexed controls remain historical controls. They are not sufficient to establish that grouping outperforms competent independent source selection.

Support variation is useful evidence even when K is constant. But do not call it adaptive K. If root-equivalent support is sufficient after co-adaptation, report that finding and do not add an R-variance penalty to manufacture diversity.

---

## 8. Compile the interface once; keep continuous trial computation live

### 8.1 Specific hot-path work to remove

The inspected code calls `is_full_access_plan` inside `AdaptiveCoverPairwiseField.read`. `MechanismPlan.is_full_access` computes `active_node_mask`, which evaluates the receiver tree on its anchor universe. This repeats for receiver chunks. Other paths repeatedly construct packet summaries, scalar GPU tensors, source unions, and compatibility accesses.

This is not an inherent cost of hypergraphs. The explicit all-access path being about nine times slower than policy-free Dense at Q40960 is direct evidence of avoidable wrapper work.

### 8.2 Minimal prepared execution view

Construct one private immutable execution view per case/phase and current plan, reusing `PreparedInterfaceField` where practical. It should contain:

- active frontier and parent/child relations;
- typed full-access flags;
- compact identical-action classes;
- selected source index sets per typed receiver block;
- static count metadata;
- coordinate/measure conventions needed to evaluate access on live queries.

Validate phase, source ordering, shape, and frame when binding a plan to preparation. Preserve existing checks and historical loading. Do not weaken the pre-push or security rules. Do not add new hash schemes; existing archival identifiers are not hot-loop routing operations.

### 8.3 All-access execution

For a fixed hard all-access plan without organizer-gradient requirements, dispatch directly to the native method using precomputed flags. Aggregate its counts analytically outside the chunk loop. It should not require anchor traversal or transfer of permission tensors to the host on every read.

The training shadow is a different execution mode: do not apply a fast path that disconnects its continuous organizer gradients.

### 8.4 Partial execution

Start with contiguous source-subset rectangles and shared source projections prepared once per physical phase. Use the union of a receiver block's sources, retaining the exact live per-query prior within the rectangle. This may compute some extra pairs to obtain regular GPU work; report that padding honestly.

Source omission from one mechanism is not deletion of that source from another mechanism's state or receiver catalogue.

The current masked QE path upcasts log-prior normalization to FP64. Do not silently change its precision while claiming same-executor parity. First remove structural overhead at the existing precision; then test a stable FP32 log-sum-exp normalization separately against the reference implementation if profiling justifies it. Never change TF32/AMP settings only for the favored model.

Keep the packed path as a correctness/memory control. Do not begin with another custom Triton/CUDA kernel.

### 8.5 Live geometry and gradients

Fixed connectivity inside a design trust region does not mean fixed numerical preparation. Recompute encoded states, relative geometry, physical source projections, ports, and output quantities at each trial and after each optimizer update.

Cache geometric index metadata only when valid. For differentiable trial coordinates, do not reuse detached geometry as though it were a live source tensor. No reuse of stale student states across weights or designs.

---

## 9. Thermal response development: broaden evidence before repeating the same fit

The R1 result is a real response-training result, unlike the earlier terminal-affine-only experiment. Do not repeat the outdated claim that only six tensors were trained.

However, it still leaves the native interaction kernels and local surrogate frozen, trains on eight response families, and lacks response-training Re90 coverage. Its broad historical replay must be checked: the report describes one historical case per family, not demonstrated coverage of all 600 historical training designs.

### 9.1 Retain incumbents and identify the adaptation scope

Use intact Run1804 e4738 for the primary paired experiment. Run1502 e4794 is a read-only comparator and one optional pilot, not a parallel sweep.

The first new scope adds the native module/environment update and fine message/read blocks that feed the existing P0/P1 interface to the previously trainable nonlinear heads. Preserve original heads and initialize exactly from the checkpoint. Keep local-surrogate weights frozen initially.

Before broad fitting, execute a tiny overfit diagnostic on train-only resolved receiver changes. If the newly trainable interaction path cannot affect the target response, trace that dependency. If measured-port substitution or an isolated local-surrogate test identifies a local representation problem, report it and select one narrow response adapter or local recalibration as the remedy. Do not unfreeze every component without a controlled comparison.

### 9.2 Broader matched sampling

Cycle broad historical training designs with equal-case value replay rather than only one fixed historical case per response family. Use identical draws in the value-only and response arms. Keep full pressure bands and a measured near-interface/solid-receiver sampling policy.

Use the eight existing training response families and any newly authorized training-only extensions. Existing Re90 calibration families remain excluded from fitting unless the experiment is explicitly redefined as development and a separate validation population is supplied. Do not quietly relabel them.

### 9.3 Optional response data use within the known physical allowance

The reconciled standing ledger reports 300/320 used, leaving 20. The separate M0 ledger is not a free interchangeable allowance.

No new solve is necessary for the primary Wind study. For Thermal, at most 16 remaining calls may be used for a deliberately chosen **training-coverage experiment**, not another underpowered inverse winner comparison:

- four training layouts, one per M=3/5/7/10, not the calibration/final response families;
- one geometry move fixed per layout before reading its new response;
- baseline and the same move at two recorded operating contexts, preferably a covered context and Re90;
- 4 layouts x 2 contexts x 2 states = 16 attempts.

Use exact embedded physical configurations, not a same-named archived case. Change coupled context inputs through the native adapter/generator convention, rather than changing a Reynolds label while accidentally leaving incompatible derived inputs. New outcomes are generator-specific training/coverage evidence at the recorded grid. They do not provide a continuum derivative or a universal Reynolds invariance law. Charge failed attempts. If the live ledger leaves fewer calls or scope is unclear, stop new solves and continue stored-data work.

Reserve the remaining calls; no new physical inverse pilot or external Wind CFD is authorized by this plan.

### 9.4 Response objective

Keep one absolute operator. Obtain all finite outputs by evaluating the complete baseline and trial designs, using common-fluid and material-coordinate conventions.

Include direct finite per-module peak and maintained pressure increments, alongside field/interface responses and historical absolute values. Compare with a matched value-only arm. Use physical scales and numerical floors only where measured; unresolved mixed responses are not zero labels.

Report train, same-layout context transfer, fresh-layout transfer where available, and broad values separately. Fixed-heat nulls are controls of the local generator, not universal fluid laws.

Allow up to 2,000 updates per arm within the overall round budget, with reviews after 200, 500, 1,000 and 2,000. Continue only while the selected recipe is recovering train/development risk; do not call a small 200-update local adaptation a fully matured reusable response model.

---

## 10. What “fully trained and reliable” means before inverse use

Do not define it as a checkpoint reaching a particular epoch number. A candidate is ready for an explicitly bounded inverse task only when the following have been measured at the same selected weights and precision:

1. Adequate whole-domain and protected-receiver values on the intended data distribution.
2. Useful finite changes at the intended move scale, including per-module quantities before objective reduction.
3. Pressure/constraint increments and uncertainty that are meaningful relative to the available margin.
4. A stable deterministic interface inside the local optimization region, with declared behavior on rebuilding.
5. Hard-plan, compiled-path, and complete-wrapper consistency.
6. A training curve that is no longer improving only by sacrificing a protected quantity for cost.

A mature Dense value predictor is a strong starting point, not already a response-certified inverse model. A mature sparse predictor still needs decision tests. A reliable graph at one tolerance is not a guarantee at larger moves or unseen boundary conditions.

Do not start a learned inverse network simply because a forward run finishes. First make a fixed-version forward-driven optimizer work under reference checking. This supplies meaningful future inverse-training targets and failures.

---

## 11. Reusable inverse interface and evaluation

Retain separate views:

- **transport:** typed direct source-to-receiver permissions;
- **response:** design-variable effects on named receiver quantities, including indirect paths;
- **decision:** the objective/constraint-specific use of those responses.

The physical graph `H(d,c;epsilon)` must not change the underlying field merely because the user asks a different objective. The downstream decision selector may prioritize different receivers or packets.

A forward interface should return the selected model identity, current prepared physical state, typed packet structure, declared unmasked global/local routes, named receiver functionals, and supported context/move scale. Do not build a new certification database around it.

Use existing per-module correction:

\[
\widetilde T_i(d;d_0)=T_i^{ref}(d_0)+\widehat T_i(d)-\widehat T_i(d_0),
\quad
\widetilde J(d;d_0)=\max_i\widetilde T_i(d;d_0).
\]

Pressure receives the analogous same-model increment correction. This cancels value offsets, not slope error.

Freeze only combinatorial connectivity during a local inner search. Recompute all continuous states from caller-owned live design tensors. Rebuild at accepted anchors and measure the discrepancy at the same design. Do not label fixed-topology AD/FD agreement as physical derivative validation.

### No new learned-graph decision claim from P2-only pressure perturbations

The existing P2 intervention changed pressure slightly and left all module peaks unchanged. If the objective is a peak material temperature, test packets affecting the actual P0/P1/local-response paths. A disconnected packet cannot guide that objective just because it is visually sparse.

### Stored inverse evidence this round

Use previously opened candidate pools only for retrospective calibration, sign/ranking error, feasibility diagnostics, and empirical regret on a fully defined observed union. Do not treat c00/c01/c02 in different policy pools as the same design, infer missing physical outcomes, or call those families untouched tests.

Use the simpler full-access or sensitivity-based proposal method as a control. A joint move can be useful because one coordinate helps pressure and another helps temperature, even with zero mixed physical response. Do not label objective nonlinearity or constraint coordination as an identified many-body law.

For WindFarm, evaluate only quantities derivable from available stored velocity fields under a documented definition. No power/AEP or continuous-layout design gain is claimed without the necessary independent reference contract.

---

## 12. Evaluation schedule and decision table

### Before formal training

Execute native all-access parity, the closed-QE gradient diagnostic, a physical-parameter gradient-isolation test for the shadow method, and a compiled full-access timing replay. Then execute real optimizer steps with the chosen trainable scope.

These do not replace training or reference evaluation. A failed ordinary implementation test should trigger a bounded repair, not a new research conclusion.

### At each substantive review

Record:

- hard reference error in physical units for all five Wind roles;
- teacher discrepancy as a separate number;
- role budget violations and per-layout tails;
- same-M / different-layout and same-layout / different-direction behavior;
- candidate capacity, raw frontier, canonical K, query packet degree, source unions and actual pairs;
- whether omitted permissions receive useful restoration signals;
- physical and organizer parameter updates separately;
- complete and per-stage timing, with diagnostics scope stated;
- how much of the predictor remains outside the organized routes.

At updates 500 and 1,500, stream at least two representative training native grids and two development native grids, chosen by input geometry before inspecting their errors. At the selected endpoint evaluate the established native validation panel, and stream a bounded multi-M/layout set of complete grids. Label this as repeated development evidence.

Store a bounded set of actual residual slices and native cell coordinates so the next report can display real spatial errors rather than infer them from role aggregates. Raw arrays remain local under the repository artifact rule.

### How to interpret outcomes

| Outcome | Interpretation and action |
|---|---|
| Native physical risk recovers; K remains constant | Useful sparse forward candidate, adaptive-K claim not established. Test shared-root equivalence and deeper supported refinements; do not force diversity. |
| K varies; physical risk fails | Structure-learning result only. Recover fidelity or reduce omissions before inverse promotion. |
| Physical risk and K are credible; executor is slow | Retain scientific result and complete the narrowly scoped execution repair. Do not erase successful labels because the current executor is poor. |
| Work falls while reference risk grows | Wrong optimization tradeoff, not forward maturation. Stop work pressure, inspect gradient signals and constraints. |
| W-full also degrades | First diagnose data measure, optimization rate, parameter scope, or domain adaptation. Do not blame the graph alone. |
| W-full succeeds and W-packet repeatedly fails | A meaningful sparse-representation limit at this support/budget; restore sources or change the factorization. |
| Thermal response gain is train-only | No inverse promotion. Distinguish coverage failure from inability to fit and from numerical reference limits. |

No automatic 5,000-epoch continuation follows from any single row. Provide one final maturity recommendation based on the selected fixed model and its stated task domain.

---

## 13. Focused tests and actual execution

Use existing tests and add only tests attached to the proposed changes:

- all-access function identity through native wrappers;
- active versus omitted QE permission gradients and real restoration-loss changes;
- hard-forward/shadow-value identity and no double physical-parameter gradient;
- fixed-plan parameter and design-coordinate gradient parity against the hard reference;
- function-preserving split and identical-action packet compaction;
- per-mechanism normalization and quadrature preservation;
- all-empty typed behavior exactly matching the declared contract;
- source permutation, query permutation/chunking, and duplicate quadrature measure handling;
- fresh sampling and layout-split exclusion;
- constrained-risk units, budget arithmetic, and work-only stopping behavior;
- compiled full-access path without repeated tree/host operations per chunk;
- outputs and gradients for subset versus masked execution;
- live recomputation after changing a design or optimizer state;
- physical per-module correction before reducing the peak.

For the executor compare Q=64/256 inverse-like reads, Q1024, Q8192, and Q40960; at least two source/layout sizes; native chunk and one common larger chunk where memory permits. Measure one objective-plus-backward call as well as forward. Include actual organizer formation in a complete learned-plan measurement.

Use interleaved synchronized measurements, common precision settings, and recorded external GPU occupancy. Do not promise a speedup from pair counts. Do not use CUDA_LAUNCH_BLOCKING or global deterministic settings as production optimizations.

---

## 14. Code route: small reusable changes, not another diagnostic framework

Expected touchpoints:

- `src/honf_forward_core/interface_fields/adaptive_cover_field.py`: hard/soft organizer-gradient separation, prepared fast dispatch, reuse of source projections, exact subset execution.
- `.../adaptive_interaction_cover.py`: compile static frontier/type metadata once; preserve serialization, validity and phase checks.
- `.../core.py`: attach compiled execution view per preparation; keep live physical states; avoid per-chunk static metadata reconstruction.
- `.../input_cover_organizer.py`: retain typed source scoring; allow explicit hard and soft plans; preserve dimension-aware and measure-aware features; do not accept labels/objectives/case IDs at deployment.
- `Case_WindFarm/src/windfarm/workflows/train_forward.py`: reuse the actual native training loop and fresh data sampling.
- `Case_WindFarm/src/windfarm/workflows/native_cover_organizer_fit.py`: leave the historical fixed-panel reproduction path available; factor out reusable objective/plan helpers rather than expanding it with another several-thousand-line monolith.
- A compact new training configuration/workflow for the matched fidelity-budgeted student, using the existing run store and checkpoint machinery.
- `Case_ThermalChannel/.../response_control/`: controlled expansion of trainable interaction scope, broad historical replay, paired finite functional objectives.
- Existing inverse contracts: fixed model/version, fixed combinatorial plan, live trial tensors and response summaries.

The new formal training path must not require an unrelated old failed G0-job file as a prerequisite. Preserve historical reproductions and actual checkpoint/security validation; remove only that accidental experimental dependency from the new path.

No new artifact hashes, source-freeze system, approval database, background monitor, or generic policy engine. Existing security and pre-push measures remain intact. Use ordinary Git history, resolved configuration, standard tests, checkpoint loading, and simple counters.

---

## 15. Bounded autonomy and resource envelope

### Active problem solving

Codex is authorized to try up to **three short native optimization remedies**, at most 200 optimizer calls each, before choosing the formal recipe. Examples:

- shadow gate gradient versus alternating hard-physical/soft-organizer updates;
- a smaller physical fine-tuning rate or broader fine-message scope;
- fidelity-first alternating optimization if multipliers oscillate;
- vectorized compiled rectangles if packed execution is dominated by Python/host work.

Use a small real training panel, preserve failures, and keep the initial weights/data comparable. Do not choose a recipe by development K appearance. Synthetic tests alone cannot select a native learning recipe.

When blocked: isolate the first failing condition, make a minimal reproducer, try one or two justified local alternatives, then report the unresolved limit. Do not repeatedly restart a many-hour job on the same unresolved issue. Do not claim a framework limitation because one optional compiler/backend fails; retain the explicit PyTorch reference.

### New-round ceilings

- Physical GPU 2 only, after checking available memory and existing jobs. Never terminate unrelated work.
- Total GPU-associated experiment wall ceiling: **24 hours**, including training, evaluation and performance work; no double-counting nested timings.
- Total attempted optimizer calls across all pilots and arms: **18,000**.
- Wind formal pair: at most 6,000 optimizer calls per arm and at most 500 native epochs each.
- Thermal formal pair: at most 2,000 optimizer calls per arm.
- Short remedies: at most 600 calls total; include them in the total above.
- New native oracle/restoration/proposal forwards: at most 4,096, separate from ordinary accounted training calls. Reuse valid local evidence where its teacher/plan/measure remains applicable.
- New local physical solves: zero by default; optional training-coverage panel at most 16, bounded by the live remaining standing allowance. No new physical inverse trials or external CFD solves.
- No second full seed or broad hyperparameter sweep this round.

Stop gracefully at the applicable bound and save the latest valid state through existing mechanisms. A new scientific round has its own simple resource totals; do not create a new system of cryptographic budgets. Keep old ledgers intact.

---

## 16. Definition of done and final report

The task is complete when it delivers:

1. A code-supported diagnosis of why the old fixed weighted pruning objective passed the teacher criterion while losing reference fidelity.
2. A tested missing-QE-source learning remedy and an explicitly stated surrogate-gradient policy.
3. A prepared, lightweight all-access/partial execution path with measured output/gradient parity and real timings.
4. One matched native full-access versus joint packet-forward training experiment with fresh field samples and meaningful maturation coverage, or a concretely documented failure after the bounded remedies.
5. Case-dependent hard source organization and K statistics evaluated at a fixed fidelity budget, without forced diversity.
6. A matched, expanded-scope Thermal response experiment if its bounded fit-capability diagnosis supports proceeding; otherwise a localized explanation of the unresolved response problem.
7. A fixed-version inverse-readiness assessment, preserving baseline correction, pressure limits and the scope of stored evidence.
8. A concise recommendation: which forward model should be retained for prediction, which for organization research, what can be used for bounded inverse proposals, and what remains unverified.

The report must show a matrix separating **value fidelity, response fidelity, adaptive K, direct transport sparsity, whole-information-path coverage, runtime, and inverse utility**. No single green pass should stand in for all seven.

Commit durable code, focused tests, configuration and the English report. Raw diagnostics, figures, native fields, checkpoints and one-time scripts remain local. Follow `AGENTS.md` and the existing pre-push hook.

### Central success criterion

\[
\boxed{
\text{A trained physical operator and adaptive packet interface that retain
useful field and decision information while removing unnecessary interaction work.}
}
\]

A small graph attached to a frozen model, an untrained gradient path, or a low composite loss purchased with worse physical error is not that result.

---

## 17. Evidence and methodological references

### Repository evidence reviewed

- `docs/reports/HONF_Receiver_Local_Interface_and_Decision_Recovery_Study.md`: Sections 2 (R1), 4 (G5/G6), 6 (execution), 7 (inverse), 8 (resources).
- `Case_WindFarm/.../native_cover_organizer_fit.py`, reviewed at `25196da`: typed constants near the file start; role losses and work helper around lines 800–940; predictor freezing and fixed records around 2350–2450; weighted predictive objective around 2620–2710.
- `src/.../adaptive_cover_field.py`, reviewed at `25196da`: hard log-prior masked read around 1570–1650; per-read full-access checks and bridges around 1720–1830.
- `src/.../adaptive_interaction_cover.py`: `MechanismPlan.is_full_access`, `active_node_mask`, `frontier_summary`, `access_for`.
- `Case_ThermalChannel/configs/response_control_native_nonlinear_interface.json`: actual nonlinear scope, objective activation and frozen modules.
- `Case_WindFarm/.../train_forward.py`: native sampled dataset, physical weighting, grouped splits, and reusable training machinery.

These source observations are not claims that the reviewed implementation was rerun in the planning session. The independent attention-gate calculation below is deliberately small and does not establish a trained-model result.

### External methodological context

- Cotter, Jiang and Sridharan, **Two-Player Games for Efficient Non-Convex Constrained Optimization**, ALT/PMLR 98 (2019), https://proceedings.mlr.press/v98/cotter19a.html. Motivates treating fidelity constraints separately from a scalar reward and warns against assuming ordinary nonconvex Lagrangian training is guaranteed. This plan does not inherit their stochastic-classifier guarantees for a deterministic neural operator.
- Luo, O'Leary-Roseberry, Chen and Ghattas, **Efficient PDE-Constrained Optimization under High-Dimensional Uncertainty Using Derivative-Informed Neural Operators**, arXiv:2305.20053, https://arxiv.org/abs/2305.20053. Supports joint attention to state and design response; it does not validate the present generator's mesh-sensitive derivatives.
- PyTorch 2.6 **CUDA semantics**, https://docs.pytorch.org/docs/2.6/notes/cuda.html. Relevant to synchronization-aware timing, device transfers and precision. Use the actual installed version; no dependency upgrade is required by this plan.

### Small independent planning check

A two-source float64 CPU attention calculation was executed during preparation of this plan. Under a straight-through binary prior followed by the present hard support/log mask, an omitted source had zero gate gradient. The proposed zero-valued soft shadow gave it a nonzero local gate gradient, retained exact hard forward value, and preserved the hard-only physical-parameter gradient. This checks the stated autograd mechanism only. Real native permissions, complete-wrapper gradients, reopening behavior, GPU execution and trained fidelity remain required Codex experiments.
