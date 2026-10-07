# HONF next phase: establish the value of an interaction distinction

**Mode:** bounded Goal-mode research round  
**Repository / branch:** `cosmos2w/ModularDT`, `agent/honf-core-next`  
**Evidence baseline:** commit `b55dd280`; inspect later changes before editing  
**Execution envelope:** six hours elapsed, at most ten aggregate GPU-associated hours, two authorized GPUs  
**Primary approach:** reuse saved evidence and fixed forward weights; run only a diagnosis-selected small training pilot  
**Not authorized:** another maturation campaign, a large inverse sample matrix, new reference/CFD solves, or automatic continuation

## 0. Read this first

The central goal remains:

> Learn a hypergraph-inspired organizer through multi-field reconstruction that represents module–environment interactions and can be reused in flexible generative inverse design, with demonstrable generalization and interpretability.

This round is not an attempt to finish that entire pipeline. It must answer a smaller question that determines the next useful investment:

> **Are useful interaction-group distinctions already present but poorly selected, or do the existing trained computations fail to make grouping useful? For ThermalChannel, how much of the deficit exists before sparse access is applied?**

Do not restart the old sequence of four forward-training arms, a large cut search, new selector families, hundreds of inverse samples, and a broad timing matrix. A clear negative diagnosis is a completed research result; a passing software test alone is not.

The main deliverable is a short decision supported by two parallel investigations:

- **WindFarm:** separate action availability, ranking, risk calibration, and the actual value of receiver grouping at the saved u4910 weights.
- **ThermalChannel:** separate adaptation of the physical weights from access-induced error at saved checkpoints. Try one small correction only if that diagnosis identifies an actionable cause.

A tiny frozen-inverse check is optional and comes last. It is not a prerequisite for finishing the report.

---

## 1. What the evidence supports, and what is newly proposed

### 1.1 Source-derived starting facts

The principal source is [the controlled-maturation report](../../docs/reports/_bk/20261002_132648Z_HONF_Controlled_Maturation_and_Action_Aware_Organization_Report.md), especially “Closed resource accounting,” “Wind all90 predictor review,” “Wind selected frontier and execution,” and “Selected-endpoint inverse outcomes and limits.”

| Established observation | Consequence for this round |
|---|---|
| Wind G/P were selected at u4910. G-root improves the five-role average relative to P, but individual protected-role tails remain. | Preserve this result. Do not restart the physical comparison merely to increase an update count. |
| Later Wind training used MM=0.90 and QE=1.00. Every audited hard mode executed the full QE rectangle. | Describe the current candidate as MM-restricted / QE-full. Do not imply that its maturation validated environmental sparsification. |
| The Wind action table contains adequate sparse options in 8/9 train, 3/6 development, and 2/3 held-audit cases; the fitted selector rejects them all. | A selection diagnosis can proceed without new forward training. Three rows in the held audit represent only one layout. |
| Requested Wind two/four cuts both realize K=2 on the action-table cases. The root union and count-matched geometry controls can outperform the four-packet action. | Determine whether distinct receiver assignments contribute. Do not count requested nodes as distinct effective computations. |
| Thermal u1300 has no adequate G sparse action on its 12-case primary table; G full access also misses the retained-reference rule. | Do not train a new Thermal selector on an empty adequate-action set. Diagnose physical adaptation first. |
| The round charged about 36.10 GPU-associated hours, including about 27.52 hours of training-step time. Thermal P was the expensive training arm. | Separate research diagnosis from long training. Elapsed delivery time and aggregate compute are both bounded below. |
| Wind inverse G/Dense terminals were identical on 60/64 ordinary paired draws; Thermal’s selected inverse produced no completed selected samples. | Do not scale inverse evaluation before a small real trajectory and effective intervention are demonstrated. |

References and units must remain faithful to the source. Wind reference fields are stored OpenFOAM CFD outputs. Thermal targets are the local analytic-wake/shared-grid benchmark outputs, with `q_normal` a flux proxy. Learned retained models are controls, not physical truth. No generated Wind layout has new CFD truth in this round.

### 1.2 Decisions proposed by this plan

The six-hour limit, small panels, attribution tests, and optional pilot below are **new experimental choices**, not conclusions already demonstrated by the report.

Maintain the historical physical-risk thresholds for comparison. Analyze why a rule fails without weakening it until it produces desired sparse selections. A research diagnostic may deliberately evaluate an inadequate action, provided it is labelled as such and is not promoted into normal deployment.

The long-term adaptive-K objective remains. This round is permitted to conclude that the existing K distinctions are not useful. It must not manufacture K variation or force an edge to survive to satisfy the objective.

---

## 2. Time, scope, and completion: one short decision round

### 2.1 Elapsed time is the primary delivery boundary

Start a single elapsed-time clock at the first substantive local inspection. The clock includes coding, tests, loading, CPU work, GPU jobs, failed attempts, waits, rendering, and closeout. Resuming a process does not restart the clock.

- **By 30 minutes:** identify the available evidence and report the actual two-device assignment.
- **By two hours:** write a brief progress note with preliminary Wind selection and Thermal attribution findings. Do not wait for a full figure suite.
- **By four hours:** decide whether the one optional correction pilot is informative enough to finish. Do not start a new scientific branch after this point.
- **By five hours:** stop new optimization and new sampling. Use only bounded extraction from completed results and essential closeout tests.
- **By six hours:** finish the report, ordinary focused tests, outgoing artifact audit, commit, and push. If an external failure prevents push, report it plainly; do not conceal it or extend research work.

The cap is an execution authorization, not a statement that every scientific question must be answered within six hours. An unfinished measurement is labelled incomplete; it does not become a negative physical result.

### 2.2 Two GPUs, unequal scientific tasks

Use the two user-authorized devices in parallel after checking existing jobs. Historically these were physical GPU0 for Wind and GPU2 for Thermal; resolve the current assignment rather than assuming it. Never interrupt an unrelated job.

Default allocation: at most four GPU-associated hours per lane, with at most two shared hours for necessary tests and final measured checks. The **ten-hour aggregate ceiling and six-hour elapsed boundary both apply**. CPU-only selector work should run on CPU.

These are maximum allowances, not spending goals. The mandatory investigations should primarily reuse arrays and need much less GPU time if the saved evidence is complete.

### 2.3 Small operational units

Before a job, measure one representative real unit: one wrapper call, one optimizer step with its actual stencil, or one complete draw. Use that observation to choose a block that fits the remaining interval. A forecast is not a substitute for executing the first unit.

Persist each completed case, checkpoint, and sample immediately using the existing mechanisms. Do not reserve an entire multi-hour G/P block before allowing one useful short block to run. Do not wait for 256 samples before saving the first one.

Use ordinary process timeouts and cooperative stopping at a safe checkpoint boundary. Only terminate processes launched for this round. Keep a 60-minute closeout reserve rather than spending the entire allowance on experiments.

### 2.4 Explicit workload limits

- No default forward continuation and no new full 90-row review.
- At most **128 new inference-only complete-wrapper calls per lane**, including retries, controls, and final sample scoring. Internal decode chunks are reported separately, not falsely counted as independent cases.
- At most **600 CPU selector optimizer steps in total** across all new fits/folds; reuse existing fitted outputs first.
- At most **one Thermal paired correction pilot**, default 40 updates per arm and absolute maximum 60 per arm, with a 75-minute combined fit ceiling.
- At most **eight new complete inverse trails across the entire round**, not per lane or per condition.
- Zero new local-reference or CFD solves.
- No epoch quota, historical-pass completion quota, extra seed, parameter sweep, or automatic follow-on job.

A valid resource boundary should end a job with a saved partial result, not trigger the creation of another retry campaign.

---

## 3. Preserve the current models and reuse their evidence

### Wind fixed research models

- Run2112 G u4910: main grouped-model subject.
- Run2112 P u4910: fixed independently trained direct-pair comparator.
- Retained Run2110 W-full u1500: reference predictor.

### Thermal fixed research models

- G/P u1300: subjects of error attribution, not accepted deployment models.
- Run1804 e4738: retained native reference.
- At most two earlier saved **G** states from the same lineage, preferably u200 and one pre-loss-amendment state already in the report’s evidence. Choose availability/recipe boundaries, not whichever produces the best development score.

### Existing artifacts to locate

Start with the report’s local links to the selected Wind action table and selector outputs, selected support/control arrays, Thermal u1300 action table, fixed/refreshed review arrays, training/exposure curves, and exact inverse checkpoints/attempt records.

The Wind table is reported as 21 cases / 84 actions: 18 primary cases plus three disjoint-query repeats. Repeats are checks, not additional independent layouts. The primary fit set contains nine direction rows; determine and print its actual number of distinct layouts before fitting anything.

Reuse the archived numerical payloads. Do not regenerate every figure, rematerialize the full Wind volume, or rehash large datasets merely to begin analysis. Resolve a missing artifact with one targeted search or native recomputation within the call cap. If still missing, mark that comparison unavailable and continue the independent lane.

Preserve checkpoint-native normalizers, physical wrappers, source measures, local surrogates, and units. Compare errors in physical output units on identical receivers; do not compare normalized losses from different checkpoint statistics as if they were the same quantity.

---

## 4. Common mathematical language

Let `x` contain a physical layout and operating context, and let `a` denote the realized typed interaction action. Define

\[
F_\theta(x;a), \qquad a=\mathrm{full}
\]

for the complete physical wrapper with a fixed set of physical weights. A requested tree cut is not yet an action: its source permissions, receiver access, eligible sources, phase, and normalization determine the action.

For a named role `r`, use the report’s weighted physical RMSE:

\[
E_r(x,a)=\left[\sum_q \omega_{rq}
\|F_{\theta,r}(x;a;q)-y_r(x;q)\|^2\right]^{1/2},
\qquad \sum_q\omega_{rq}=1.
\]

Preserve each role’s component convention and mask. Unknown, empty, or unresolved roles are reported as such; no replacement by zero.

Report four different structural quantities:

1. realized nonredundant packet count;
2. effective receiver–source weights/support on the same physical axes;
3. logical/weighted work;
4. actual executor work and measured elapsed call time.

A packet-node/source incidence count and a native query/source pair count have different receiver axes. Do not divide one by the other to claim compression.

The research goal is a **reconstruction-useful and reusable interaction distinction**. Neither a packet label nor a favorable error correlation establishes physical causality or an irreducible many-body law.

---

## 5. Wind investigation A: where does the selector reject useful options?

### A1. Replay decisions from the existing table, without native calls

Use the current `action_aware_frontier.py` and `action_risk_fit.py` implementation. First reproduce the archived decision using its saved features, weights, thresholds, and empirical margins. Do not fit another head until this replay is understood.

For every eligible action and role, export:

\[
\ell_{ar}=\log\frac{E_{ar}+\epsilon_r}{E_{Br}+\epsilon_r},
\quad \widehat\ell_{ar},\quad m_r,\quad L_r,
\quad \widehat\ell_{ar}+m_r-L_r.
\]

Here `epsilon` is a numerical stabilizer, `m` the existing train-crossfit residual margin, and `L` the actual deployment log threshold. They are not the same thing as the physical allowance.

Record three decisions separately:

- **Historical deployed decision:** prediction plus saved margin and saved threshold.
- **Unmargined diagnostic:** same prediction/threshold with the margin removed, clearly not a new deployment policy.
- **Measured oracle:** uses reference errors only for retrospective analysis.

Break each false rejection into a trace: missing/ineligible action, raw risk overprediction, one particular role dominating the maximum, margin rejection, or a discrepancy between the fixed deployment threshold and the per-case retrospective physical rule. These effects can overlap; do not force them into a misleading additive count.

The current code maps per-case physical rules into log units but deploys train-fixed role limits. Compute the consequence of that distinction. Do not assert that the two rules are identical, or fix it by reading test-case reference errors at inference.

### A2. Measure ranking separately from calibrated acceptance

For each case, rank the trained actions by worst-role error relative to the historical allowance and compute:

- whether the lowest-predicted-risk action is among the measured best actions;
- action ordering accuracy for differences above the observed query-repeat discrepancy;
- excess measured worst-role risk of the chosen action;
- the fraction of oracle opportunities rejected;
- false accepted actions and their actual risk magnitude;
- the always-full and train-best-fixed-action controls.

Do not fit on the old held audit. It is already exposed evidence, not an untouched final test; label any new replay accordingly. Split by complete layouts, never randomly by direction or action row. Use leave-one-layout-out evaluation when the fit set is too small for five folds.

Use both absolute physical errors and differences between actions on the same case. The latter help distinguish case difficulty from the relative choice. Do not turn a small number of layouts into hundreds of independent observations by expanding roles or queries.

### A3. One inexpensive selector remedy, only after the trace

Choose **one** of the following, from training-family evidence only:

- a per-role bias correction when the head orders actions competently but systematically miscalibrates their risk; or
- a low-dimensional ridge model of action differences when shared case difficulty dominates and the ranking itself is poor.

For the latter, an available label is

\[
\Delta\ell_{ar}
=\log\frac{E_{ar}+\epsilon_r}{E_{\mathrm{full},r}+\epsilon_r}.
\]

Use existing input-only action descriptors; differences to the full-action descriptor are allowed. This target cancels the shared incumbent denominator. If absolute acceptance is attempted, separately predict the same-student-full risk relative to the retained predictor and combine the two predictions. Calibrate the **combined out-of-family residual**; do not add unrelated confidence bounds and call the result a guarantee.

A relative-ranking improvement alone is a ranking result. It must not be promoted as calibrated sparse deployment.

No new large neural head, conformal-coverage claim, threshold sweep, or use of development outcomes to choose a correction is authorized. Existing empirical margins are not formal coverage certificates. If the evidence is too small to fit a trustworthy acceptance rule, retain full-access deployment and report that limitation.

**Deliverable:** one page answering whether the present failure is primarily action ranking, absolute-risk prediction, conservative uncertainty, action eligibility, or insufficient distinct training layouts.

---

## 6. Wind investigation B: does the group distinction itself help?

A selector repair is worthwhile only if some grouped computation offers a useful tradeoff. Test this at **fixed u4910 physical weights**.

### B1. Start from saved controls

The report already finds that root union and geometry masks can beat the requested four-packet action. Reuse those records, then add a small matched replay only where needed.

Use no more than six cases: four training directions/layouts chosen by input metadata to include a same-layout direction change and, if available, a same-M different-layout pair; two existing development cases chosen by the same rule. Do not select only cases known to favor grouping. Use one fixed role-stratified native query panel and one disjoint panel where saved repeats are absent.

Compare up to these six actions:

1. trained root;
2. trained realized two-packet action;
3. requested four-packet action, reporting if it is identical to action 2;
4. one root carrying the union of the grouped source sets;
5. existing geometry or direct-score control at matched eligible work;
6. one **effective** receiver-assignment intervention, if possible.

This is at most 72 case/action/panel calls before any reused results, leaving headroom under the lane limit. Avoid complete-volume exports; one native plane can be rendered from already saved data.

### B2. Intervene on the computation, not just the label

Write the actual effective access as

\[
w_{rs}=\sum_e A_{re}B_{es}
\]

only if this is exactly the current backend’s mixing rule; otherwise export and use that rule. Keep source quadrature, eligibility, exclusions of self-interaction, and physical phase intact.

For two meaningful packets, exchange their **source permissions between receiver supports**, leaving receiver supports fixed. Renaming both packet axes together is a permutation identity and is not an intervention.

Before a forward call, count changed effective receiver–source entries. If none change, record an inactive intervention and do not cite identical predictions as robustness or interpretability evidence. If eligible counts or weighted work change, disclose them. Claim a strictly work-matched intervention only after checking both quantities; a degree-preserving swap need not preserve quadrature-weighted work.

No repeated randomized search for a favorable rewire is allowed. One deterministic, valid intervention per selected case is enough for diagnosis.

### B3. Do not misinterpret the root-union comparison

The root union may expose more sources to more receivers than the grouped action. A lower error under that union does not alone refute grouping: it may purchase accuracy with more information.

A grouped action earns a useful distinction by either:

- improving reference prediction at matched effective access cost; or
- retaining the declared physical fidelity with meaningfully less effective work than a simpler control.

For this round, measured logical work and executor rows remain separate. A logical tradeoff can be scientifically useful without a speedup, but should not be promoted as deployment acceleration.

If grouped and union computations have identical effective access, extra nodes are representational duplication. Do not count that as adaptive K.

### B4. Connect to transfer without inventing causal truth

On the input-selected paired cases, report changes in physical source membership, receiver access, reference role error, and any existing source perturbation response. No cross-direction source matching is allowed solely by tensor-column number. Use the dataset’s valid coordinate/frame correspondence; mark unavailable correspondences honestly.

A finite difference on a perturbed design without a corresponding solved field is a **surrogate sensitivity**, not a physical-reference result. Existing Wind direction pairs have reference values, but need not be interpreted as continuous design derivatives.

**Deliverable:** one concrete diagram showing whether a receiver grouping helps, is redundant, or is worse than its simpler alternative. State exactly what has been established beyond “the organizer sees the case.”

---

## 7. Thermal investigation: attribute the deficit before changing the loss again

### T1. Exact error decomposition on aligned output vectors

For the same role, case, weights, and query measure define

\[
e_B=F_B(x;\mathrm{full})-y,
\]
\[
d=F_G(x;\mathrm{full})-F_B(x;\mathrm{full}),
\qquad
h=F_G(x;a)-F_G(x;\mathrm{full}).
\]

Then

\[
F_G(x;a)-y=e_B+d+h.
\]

Use weighted squared errors to attribute the changes exactly:

\[
\Delta_{\mathrm{adapt}}
=\|e_B+d\|_W^2-\|e_B\|_W^2
=2\langle e_B,d\rangle_W+\|d\|_W^2,
\]
\[
\Delta_{\mathrm{access}}
=\|e_B+d+h\|_W^2-\|e_B+d\|_W^2
=2\langle e_B+d,h\rangle_W+\|h\|_W^2.
\]

These are exact output-space accounting identities. They do **not** identify a unique causal training component, and RMSE differences are not additively decomposable in this way.

Apply the same identity to finite responses, using the common-fluid mask or module-attached receiver coordinates already established by the atlas. Never subtract unrelated query orderings. Keep pressure and per-module peaks as separately defined scalar functionals.

### T2. Small checkpoint panel

Reuse available prediction arrays first. Otherwise evaluate:

- two training families spanning low/high M;
- two existing Re90 development families spanning low/high M;
- baseline plus one previously resolved position or heating response per family;
- a small predeclared historical-value panel, no larger than eight cases.

Use retained Run1804, current G/P, and at most two earlier G checkpoints. Run full-access historical comparisons first. Add current hard-action comparisons only where they answer the attribution question. Respect the 128-wrapper-call lane allowance; do not expand to the full atlas by default.

Report separately:

- historical absolute-field drift;
- interface/material drift;
- finite-response error;
- additional access distortion;
- loss-schedule and trainable-scope changes preceding the observed drift.

Do not assert that completing the 1,200th historical visit would repair the problem. Conversely, do not call the architecture converged merely because its last run ended.

### T3. One small correction pilot, conditional on an actionable diagnosis

Default is **no new physical training**. If the decomposition shows that a specific adaptation issue dominates, choose one matched test using training evidence only.

Preferred option when full-access physical drift dominates:

- branch two temporary copies from the same selected G checkpoint;
- keep full access in both, freeze organizer weights, and disable organizer/shadow optimization;
- keep the same historical/stencil stream, parameter scope, seed, and optimizer policy;
- arm 0 uses the actual current recorded loss;
- arm 1 changes one documented value/response weighting or replay-mixture choice, selected using training-role errors/gradients only;
- default 40, maximum 60 updates per arm, and at most 75 minutes combined.

Use the actual checkpoint recipe rather than guessing what weights were active at u1300. Do not change weight coefficients, sampling, trainable scope, and learning rate simultaneously. A reset optimizer is allowed only if both arms use the same disclosed reset.

If a normalization, coordinate, or target-binding bug is found instead, fix that narrow bug and repeat the affected replay. Do not spend the pilot allocation on a training workaround for an implementation error.

If access distortion dominates while full access remains credible, do not run the full-access correction above. One bounded permission-only adjustment on already trained actions can replace it, with fixed physical weights, training references only, and identical time/update ceilings. A small correction is a fit probe, not proof of maturity.

Use development cases for the fixed after-pilot assessment, not to choose the pilot objective. Preserve the original u1300 result. If the pilot remains actively learning at its stop, report that fact and recommend a separate continuation rather than launching it.

**Deliverable:** a signed-error attribution figure and one specific next training hypothesis, supported or rejected by the optional pilot. Do not rebuild the Thermal selector this round when no adequate actions exist.

---

## 8. Tiny frozen-inverse readiness check: optional, with complete outputs

### 8.1 What this can establish

This is an interface and conditioning check, not a new inverse benchmark and not independent physical validation. Do not train another generator by default. Reuse the selected inverse checkpoints and repaired spatial conditioner.

Begin only if the primary lane diagnoses have been saved, before the five-hour stopping boundary, and a real timing check shows completion fits the remaining allowance.

Choose **one dataset**, not both, using operational readiness before opening new sample outcomes. Prefer a task with a known valid supplied design domain and a nontrivial actual link intervention. If neither lane has one, record the reason and defer sampling.

### 8.2 First complete unit

Run one condition, one existing model, one seed, twenty existing reverse steps, and batch size one. Save each intermediate state. Save the terminal before any additional scoring or condition begins.

If a single draw is not expected to finish within 120 seconds after measuring real step cost, stop and profile one provider call plus one denoiser step. Do not launch another whole sampling matrix. The complete optional inverse allocation is at most 20 minutes including profiling and up to eight trails.

Do not silently change the trained diffusion schedule to fewer steps. That would be a different sampler requiring its own interpretation.

### 8.3 Minimal paired comparison

At most two tasks, one shared seed each, two observation conditions, and two link settings gives eight trails total:

- original observed values;
- one declared within-training-range value change at fixed sensor locations;
- actual packet links versus full links, using the **same denoiser weights** for the first mechanistic test.

The same-denoiser comparison identifies sensitivity to links; it is not a fair comparison of separately trained inverse models. Existing independently trained G/Dense results can be shown separately without launching another matrix.

If packet and full links are equal, or the policy falls back to full access, say that no active graph-reuse comparison is available. An explicitly forced research action may be tested as an intervention, with its forward inadequacy disclosed; do not label it deployed or physically valid.

The provider sees only visible inputs, fixed sensor positions/values, and the current candidate. Never form links, embeddings, bounds, caches, or denoising context from the hidden clean target. Recompute candidate-dependent states at every step. Cache only genuinely invariant quantities.

### 8.4 Geometry and physical claims

For Wind, retain the distinction between the public design box, native data support, and rotor clearance. A common training-derived domain changes task coverage; report which tasks it excludes. Do not infer allowable bounds from a hidden target or claim that public-box validity means row-native validity.

For Thermal, check nonnegative allocation, exact total heat, observation residual, and maximum heat fraction. Valid total heat is not satisfactory conditioning. With zero new solves, candidate field errors are frozen-surrogate checks only.

A failed or incomplete sample should still leave useful output: last state, time per step, provider/denoiser cost, attempted steps, and the reason for stopping. Do not record unobserved terminal outcomes as failures of generated designs.

---

## 9. Coding scope: extend existing tools, do not build another experiment framework

Inspect and reuse these paths at the local branch revision:

| Area | Existing code to start from |
|---|---|
| Input-only action descriptors and selection | `HONF_Proj/src/honf_forward_core/interface_fields/action_aware_frontier.py` |
| Risk fitting, cross-fitting, policy audit | `HONF_Proj/src/honf_forward_core/interface_fields/action_risk_fit.py` |
| Wind saved action-table loading/evaluation | `HONF_Proj/Case_WindFarm/scripts/evaluate_matured_action_table.py`, `fit_matured_action_selector.py` |
| Same-weight controls and Thermal role evaluation | `HONF_Proj/Case_ThermalChannel/scripts/run_selected_weight_controls.py`, `evaluate_matured_action_selector.py` |
| Thermal actual training recipe | `HONF_Proj/Case_ThermalChannel/scripts/run_controlled_maturation.py`, `src/channelthermal/response_control/training.py` |
| Inverse sampler and observation path | `HONF_Proj/src/honf_inverse_core/models/frozen_packet_diffusion.py` |
| Candidate providers | `Case_WindFarm/src/windfarm/inverse/packet_completion.py`, `Case_ThermalChannel/src/channelthermal/inverse/packet_reuse.py` |

High-value changes, only as required by the selected investigation:

- a selection trace exposing prediction, threshold, uncertainty margin, eligibility, and rejection reason;
- one reusable vector-error attribution helper;
- immediate per-sample persistence and a narrowly scoped progress callback;
- the single justified calibration/correction option and focused tests.

Do not duplicate multi-thousand-line launchers, add a plugin system, or refactor unrelated historical architectures. A one-time extraction or rendering script stays ignored; generic utilities and tests can be committed. A short configuration-driven research entry point is preferable to another frozen orchestration stack.

### Focused tests

Use only tests applicable to code changed in this round:

- the squared-error decomposition closes numerically, including negative cross terms;
- selection replay reproduces the historical choice without changing its thresholds;
- diagnostic removal of a margin cannot silently change default deployment;
- layout-grouped folds contain no direction leakage and no repeated-query pseudo-cases;
- packet relabelling is invariant but the proposed receiver/source intervention changes actual access;
- role masks, units, and normalization match between compared tensors;
- saved completed sample survives interruption before the next sample;
- current-candidate providers exclude clean hidden targets.

These tests support execution correctness. They do not replace native measurements or establish graph physics.

---

## 10. Evaluation discipline and the reporting contract

### 10.1 Small tables that answer the actual questions

Wind must separate:

- adequate action existence;
- action ordering;
- risk calibration before the margin;
- margin-induced rejection;
- selected physical error;
- added value of receiver grouping at matched information/work;
- execution overhead.

Thermal must separate:

- retained-model error;
- adapted full-access error;
- additional hard-access error;
- finite-response error;
- optional correction outcome.

Do not combine differently normalized roles into an unexplained scalar. Use the source report’s primary definitions, with any new normalized view labelled as diagnostic. Preserve maximum/p90 tails where meaningful, but do not estimate population tail probabilities from two development families.

All previously inspected development and held-audit outputs remain exposed research evidence. No new untouched-test or cross-dataset-generalization claim is authorized.

### 10.2 Required opening summary

The final report must start with no more than one page that a reader can understand without the implementation history:

| Goal | What we gained | What remains missing | Strength of evidence | One next action |
|---|---|---|---|---|
| Predictor | Actual observation | Actual limit | Fixed weights / pilot / reference source | Concrete decision |
| Organizer | Actual observation | Actual limit | Input-only / oracle / intervention | Concrete decision |
| Inverse system | Actual observation or explicitly not run | Actual limit | Existing surrogate / complete samples / no new truth | Concrete decision |

Then answer, in ordinary language:

1. Was Wind blocked by the chooser, by the grouped computations, or both?
2. Did Thermal lose accuracy before masking sources?
3. Was any graph distinction both effective and useful?
4. What finished within the clock, and what was intentionally not attempted?

“No inverse campaign was run because the graph was inactive” is an acceptable answer. “All goals complete” is not a substitute for these distinctions.

### 10.3 Minimal necessary figures: four, with an optional fifth

1. **Wind selection trace:** measured versus predicted action risk before/after the existing margin, accompanied by the physical receiver/source picture for one preselected case.
2. **Grouping-value picture:** field or role-residual comparison for grouped, root-union, and one effective matched intervention, with actual access/work alongside it. Use native coordinates, not tensor index as a spatial axis.
3. **Thermal attribution:** retained/full-adapted/hard-adapted physical fields or material-receiver profiles, plus signed squared-error change from adaptation and access. Include a representative and an unfavorable case selected by a disclosed rule.
4. **Research timeline:** elapsed time and GPU-associated work by task, marking real training, inference, setup, and closeout without adding overlapping scopes twice. If a pilot runs, include its loss/role trend.
5. **Optional inverse:** one completed paired design trail or heat-allocation sequence, with observation condition, validity, and surrogate residual. If no trail completes, replace it with a small attempt/progress panel—not an invented image.

Use existing renderers where practical. Retain PDF masters and small raster companions according to `AGENTS.md`. Do not generate an additional figure for every metric. Captions state reference source, units, checkpoint, case/role, sampling/mask, and limitations. Clip colors only for display and report unclipped errors.

### 10.4 Evidence and software policy

Retain detailed numerical appendices and local arrays, but keep the report’s front page readable. Reuse ordinary Git history, checkpoint metadata, tests, and existing artifact checks.

Do not introduce new cryptographic hashes, contract freezes, approval layers, immutable snapshots, or global blocking “gates” for this short study. Existing trusted-checkpoint loading, security checks, archival bindings, and the pre-push artifact hook must not be deleted or bypassed. If an archived runner cannot accommodate a new scientific recipe, invoke the underlying maintained APIs in a separate clearly named experiment instead of weakening historical guarantees.

Use research criteria to allocate the remaining effort, not to stop an independent lane because another metric is unknown. Irreversible writes, cross-system changes, security boundaries, and formal releases retain their required checks. Resource limits remain real limits.

Commit reusable source/tests/configuration and the written report, then push the existing non-default branch under the standing repository rule. Keep one-time runners, generated plots, data, predictions, and new pilot checkpoints in ignored local paths. Do not spend closeout regenerating every historical artifact.

---

## 11. Bounded autonomy and remedies

Codex should fix ordinary issues without escalating each one, but not drift into a new project.

Allowed examples:

- correct a role-axis or feature-scale inconsistency;
- reproduce a selector from its actual saved checkpoint;
- reduce a diagnostic query panel while retaining the defined protected roles;
- fix a non-effective rewire and show the actual changed entries;
- persist completed samples before evaluation;
- use an existing rectangular reference path when a packed path is unsuitable.

For any substantive failure: preserve the result, state a short hypothesis, execute the smallest decisive check, and choose at most one algorithmic remedy per lane. Limit ordinary implementation retry loops to two attempts on the same blocked task. If a dependency cannot be resolved in roughly 20 minutes, record it and use existing evidence rather than reconstructing the whole prior campaign.

Do not use a shorter run’s failure to assert a universal architecture limit. Do not use a resource-censored run as a reason to authorize continuation automatically. At the six-hour closeout, recommend the next experiment; do not launch it.

---

## 12. What each possible outcome means

| Result this round | Appropriate next phase; not authorized automatically |
|---|---|
| Wind has useful action distinctions, accurate ordering, but overconservative/poorly estimated risk | Expand independent training-layout evidence for a small selector; retain protected deployment criteria. |
| Wind has adequate sparse actions, but root/geometry controls match or beat grouped actions at matched work | Reconsider the representation that defines packets; do not invest in a larger K-selector. |
| Wind grouping preserves reconstruction with less logical work, but executor overhead dominates | A dedicated narrow execution study may be justified; the present result is not a speedup. |
| Thermal error is dominated by physical-weight drift and one correction improves a fixed development panel | Request a separate maturation job for that single recipe, with a clear time/exposure forecast. |
| Thermal full-access fit is credible but sparse access harms the relevant roles | Change permission learning or the physical route being organized, not the risk threshold. |
| Thermal train responses improve while transfer remains weak | Specify missing coverage/representation evidence; do not repeatedly train the same eight families indefinitely. |
| A tiny inverse test shows meaningful observation dependence and a consequential graph intervention | Plan a proper matched inverse experiment after the forward version is selected. |
| Packet/full inverse computation is equal or no complete draw is feasible | Record the inactive/unfinished interface; do not generate hundreds more trails. |

The minimum successful closeout is not a new model that wins everywhere. It is:

> **A supported decision about where the next unit of research effort should go: physical representation, action selection, execution, or inverse conditioning.**

The central HONF milestone remains more ambitious: a learned interaction organization that is active, preserves the relevant physics, and offers measurable value when reused beyond reconstruction. This focused round determines which obstacle to that milestone is real, without another day-long pipeline expansion.

---

## 13. Source and inspection notes

Evidence basis:

- `docs/reports/HONF_Controlled_Maturation_and_Action_Aware_Organization_Report.md`, selected u4910/u1300 results and local evidence links.
- `docs/reports/HONF_Active_Organization_and_Frozen_Inverse_Reuse_Study.md`, historical experiments only; do not mix their checkpoints or criteria into selected-endpoint tables.
- `src/honf_forward_core/interface_fields/action_aware_frontier.py` and `action_risk_fit.py` at `b55dd280`, inspected for risk labels, ranking, cross-fit residual margins, and fixed deployment limits.
- Root `AGENTS.md`, inspected for completion/upload and scientific-report requirements.

The plan’s author reviewed these reports and relevant pushed code, not the ignored local checkpoint payloads or every process log. Codex must inspect actual local availability at startup. No public-source research or new physical result is implied by this work plan.
