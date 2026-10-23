# HONF next phase: controlled maturation and action-aware organization

## Goal-mode brief

**Central research goal:** learn a reusable organization of module–environment interactions through physical-field reconstruction, then test whether that frozen organization improves flexible conditional inverse generation. Predictor fidelity, organizer usefulness, and inverse quality are three distinct outcomes.

**This round's question:**

> After removing identifiable training confounds and giving the sparse computations a meaningful, measured training opportunity, does a grouped interaction model preserve useful physics, and can an input-only selector choose among its genuinely trained computations?

This is a maturation-and-identification round, not another wholesale architecture replacement. Keep the native physical model, fine sources, typed interaction interface, and grouped/direct controls. Change the training protocol and the information supplied to the frontier selector. Repair one concrete full-access shadow inconsistency before allocating a longer run.

Run WindFarm and ThermalChannel concurrently on the two user-authorized GPUs. Use one device per dataset lane; run matched arms in comparable blocks on that lane's device. Inspect actual device identities and existing jobs before use. Do not interrupt unrelated work.

### Decisions to execute

1. **Preserve all historical results.** Review starts from implementation commit `2e792a9`; branch `312921f` adds reporting guidance and curated figure references. Inspect local HEAD and any newer changes before editing.
2. **Do not resume Wind G-u500 under its old mixed-stress curriculum.** Start the controlled maturation branch from the matched Wind G/P u100 checkpoints, before the 0.75 stress phase. These are warm starts for an amended experiment, not an unchanged continuation. If either u100 checkpoint is unavailable or incompatible, initialize both arms from the same retained Run2110 W-full u1500 and repeat the short common warm-up.
3. **Use Thermal G/P u200 as the matched continuation base.** Do not use the R4/R5 selector as a training oracle. Keep Run1804 e4738 as the fixed native reference and Run1502 e4794 as a historical comparator.
4. **Give the primary 0.90 capacity a stable training interval.** Do not interleave 0.75 stress updates into the primary maturation run. Evaluate 0.75 only at frozen reviews initially.
5. **Train a small, explicit cut family well before expanding it.** Start with a two-packet scaffold, then exercise root/two/four-packet cuts with documented exposure. The scaffold is not an adaptive-K result.
6. **Protect the retained reference, not only the student's own full-access mode.** A deteriorating full-access student must not make its own sparse/full ratio look like success.
7. **After forward stabilization, rebuild cut evidence at the selected weights and fit an action-aware selector.** Do not reuse the old u100/u200 tables as labels for different physical weights or permission scorers.
8. **Repair inverse-conditioning information loss now, but defer another expensive inverse superiority experiment.** Existing frozen-inverse results remain valid negative pilot results. This phase does not automatically consume new reference solves.
9. **Classify stopped models explicitly:** implementation failure, unstable optimization, overfit, empirically plateaued within budget, or resource-censored. An endpoint alone is not convergence.

### What this round should NOT do

No new ADMM/merger method, graph-count penalty, desired K histogram, new field-value pooling branch, replacement local surrogate, unbounded density sweep, automatic 5,000-epoch job, or elaborate new approval/provenance system. No declaration that a slow scientific executor disproves the representation. No inverse claim based on a selected organizer that is known to be unsupported.

---

## 1. Reading the previous result correctly

### 1.1 Predictor

The prior report contains promising local comparisons: Wind G-u500 beats P near turbines on 64/90 rows; Thermal G-u200 wins 23/32 role entries on the small inspected development pair. Neither establishes mature architectural superiority. Wind P is better on four other median roles, and the retained full-access Wind reference is stronger overall.

The primary Wind physical training reached 500 updates, not 500 dataset epochs. Native rows were drawn with replacement. Covering 132 distinct layouts does not establish repeated coverage of each layout, direction, capacity, and cut. The Thermal comparison reached 200 physical updates per arm. These are short adaptation results on mature initial weights, not fresh convergence studies of the new sparse operators.

Wind's deterioration after introduction of 0.75 stress is strong evidence against blindly continuing that recipe. It is not proof that the intended primary 0.90 operator cannot recover with a stable training distribution. Use fixed-case replays to separate case mix, physical-weight drift, and masking distortion.

### 1.2 Organizer

Thermal Stage B found physically measured adequate cuts in 6/10 fit families, with minimum passing K between 1 and 4. This is a real computational opportunity within those training families. R4's failure to select those opportunities is a selector failure, not proof that no adequate cut exists.

Wind Stage B had no adequate cut in 12/12 measured cases at the old weights. A utility network cannot manufacture an adequate physical computation by training longer on that all-negative table.

The current Wind G-u500 root result and the Stage-C research model are distinct versions. Stage C originated at G-u100 and received 100 more physical updates; the inverse experiment did not reuse the G-u500 physical endpoint. State this clearly in all comparisons.

### 1.3 Inverse system

The first actual frozen-organizer generative pilots are useful infrastructure and experimental controls. They do not establish an inverse advantage: Wind geometry validity and observation-value conditioning are weak; Thermal respects total heat but often concentrates it on one module, and Dense has lower sensor error on the small local-reference panel.

An unsupported forward organizer, a shallow/short-trained inverse model, sensor-conditioning information loss, and out-of-domain proposals are separate limitations. Do not attribute their combination solely to the quality of the learned hypergraph.

### 1.4 Visual interpretation

The existing field figures show competent wake or thermal reconstruction plus localized failures. Source-support figures show learned computation, not identified causal edges. Thermal frontier plots are particularly valuable because they separate available measured cuts from incorrect learned choices. Sampling trails show the sampler trajectory, not monotonic optimization progress.

Keep these distinctions in the next report, without repeating every audit statistic in its opening summary.

---

## 2. Mandatory source diagnosis and narrow repairs

### 2.1 Full-access grouped shadow handles the wrong infinity

Inspect:

- `src/honf_forward_core/interface_fields/input_cover_organizer.py::plans_from_scores`
- `src/honf_forward_core/interface_fields/budgeted_frontier.py::project_unique_pair_budget`
- `src/honf_forward_core/interface_fields/native_joint_shadow.py`
- `Case_WindFarm/scripts/run_active_packet_reuse.py::_direct_projected_access`

At `2e792a9`, grouped projection returns a **negative-infinite threshold for budget 1**, as expected for full access. `plans_from_scores` then uses `torch.isinf(threshold)` for its empty-projection remedy and replaces either sign of infinity with `max(valid_scores)+1`. Its soft permissions at a full budget therefore become

\[
 p_{es}=\sigma(z_{es}-\max z-1)\leq\sigma(-1)\approx0.269,
\]

rather than one. The direct path separately handles negative infinity with full hard AND soft weights.

The grouped hard forward remains full access; this finding does not invalidate its reported hard predictions. It makes the organizer's auxiliary gradient at scheduled full-budget replays inconsistent with the full-budget action and with the direct control. Its historical training impact is not yet quantified.

**Repair semantics:**

- Full budget: exact full hard/soft access on valid pairs. No pruning-score update is needed for that mechanism.
- Empty projection at a positive partial budget: keep the declared empty hard result and the finite soft restoration route where that is the existing intended training rule; diagnose why the positive-budget plan emptied.
- Intentional zero budget: define and test its semantics explicitly; it is not a normal operating point in this experiment.
- Mixed typed budgets: only the genuinely partial mechanisms receive topology gradients.
- Do not use zero gradients followed by AdamW momentum/decay and call that a frozen router. Skip an inapplicable route optimizer step or preserve the established equivalent `grad=None` behavior.

Add focused tests for both infinities, ties, padding, mixed mechanisms, native full/hard value parity, and physical versus organizer gradients. Execute an actual paired native replay. If historical full-budget steps did not enter this branch in a given lane, record that narrower exposure rather than claiming a measured effect there.

### 2.2 Audit which cuts actually received training

Wind `_seeded_cut` selects the minimum, median, or maximum cut size. For a full depth-three tree, the 26-cut size counts are:

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Number of cuts | 1 | 1 | 2 | 5 | 6 | 6 | 4 | 1 |

The present size rule reaches K=1,5,8: only eight of those 26 cuts. It does not train all cuts later evaluated by Stage B. Real early-leaf trees can differ; compute their actual coverage rather than applying this table blindly.

Export **data-row × cut-path × capacity exposure**, not just total updates or distinct layouts. Thermal's randomized 26-cut schedule needs the same accounting even though its sampling rule differs.

A cut without adequate training can be evaluated as an extrapolation probe, but cannot be silently treated as a matured candidate action.

### 2.3 The current utility representation omits the realized action

`FrontierUtilityHead` currently receives a mean of node embeddings plus a budget vector. It does not directly inspect the retained source set, discarded set, score margins, or selected source features. It also regresses work that can be computed from the candidate plan.

Do not merely increase utility updates on the same representation. Section 7 specifies the replacement after forward maturation.

### 2.4 Sensor conditioning loses location–value pairing under full links

Inspect `ConditionalPacketDenoiser` in `src/honf_inverse_core/models/frozen_packet_diffusion.py`.

The initial sensor encoder is a single linear map. Under full sensor access, the returned observation message averages these embeddings. For features formed by concatenating coordinate and value,

\[
\frac1S\sum_s W[x_s,y_s]
= W_x\bar x+W_y\bar y.
\]

Swapping standardized values among fixed sensor locations while preserving their mean leaves this message unchanged. With full links, the subsequent updates retain this limitation: they do not reconstruct the discarded location–value association. A K=1 graph with shared sensor access can have the same issue. Nonuniform graph access or sensor-specific normalization changes the exact invariance scope, so test both the constructed case and native controls.

This is a representation limitation, not a claim that it explains every inverse failure. More training alone cannot restore information an invariant aggregation discards.

Repair the shared inverse conditioner with a nonlinear per-sensor token AND receiver-specific relative-coordinate attention/message passing. Graph and Dense inverse arms receive the same upgraded conditioner. The graph supplies a prior/support restriction, not a different observation encoder. Tests must verify simultaneous sensor-token permutation invariance AND sensitivity to reassigned values at fixed coordinates. See Section 9.

### 2.5 Keep the work comparisons honest

Canonical weighted budgets and live-query pair counts differ. Do not project direct P differently on each requested chunk in production to manufacture a fair count: that would make access depend on the output query batch. Keep a query-independent canonical definition, and perform count-matched replays as explicitly separate controls.

At minimum, report canonical work, live per-role pair density, actual executed rows, and all unchanged paths. Low-M integer MM caps are especially coarse: retaining 90% of six eligible off-diagonal pairs can require retaining five, a 16.7% omission. This is not a 10% realized reduction.

---

## 3. One forward architecture; a better controlled training experiment

### 3.1 Preserve the physical operator

For design/context \(d,c\), mechanism \(m\), and receiver \(q\), use the existing typed packet weights

\[
w^{m}_{qs}=\sum_{e\in\mathcal C}a_e(q;d,c)b^m_{es}(d,c),
\qquad \sum_e a_e(q)=1.
\]

Here \(\mathcal C\) is a complete receiver-tree cut and \(b\) is its learned source permission. A shared packet is a reusable source-access action across receivers; it is not a pooled physical field value.

Keep the existing fine nonlinear messages, source quadrature, one environmental normalization over the unique supported source union, native denominators, output biases, and physical coupling. MM/ME/EM/QM/QE mechanisms remain separately reported. Coarse/local paths remain explicit, not hidden evidence of graph sufficiency.

Dense-masked execution is acceptable for scientific training; the experiment must not imply row savings from masked rows. Do not start a custom kernel project in this round.

### 3.2 Work at one primary capacity

The formal maturation target is the existing 0.90 QE and selected-MM capacity vector, retaining its documented canonical measure. Remove 0.75 from optimizer sampling. The 0.75 case is a frozen-checkpoint robustness probe only.

First execute a small same-input route-local diagnosis: full, QE-only partial, MM-only partial, and combined partial. Check per-receiver degree, empty rows, attention normalization, source-score range, and loss/gradient changes. If one mechanism causes a repeatable severe instability, one training-only remedy may relax that mechanism to 0.95 or leave it full while the other matures. Apply the same amendment to G and P and label it a narrower graph-scope experiment. Do not silently claim success on both mechanisms afterward.

This is not a sweep. Retain the primary setting unless actual native execution identifies a reason to change it.

### 3.3 Cut curriculum: first give specialization a fair opportunity

Use three default actions specified by abstract tree paths:

- root: `['']`;
- one root split: `['L','R']`;
- balanced depth-two cut: `['LL','LR','RL','RR']`.

If a path is unavailable because of an early leaf, record the available cut; do not invent nodes or silently increase capacity. The design task's geometry generates the tree as before.

**Phase A — stable two-packet scaffold.** Exercise the root-split action at primary capacity for two complete eligible primary-data passes, with one full-access replay after every four sparse updates. Here one Wind primary pass covers the 408 allowed native direction rows; one Thermal primary pass covers every eligible response family once. Thermal historical replay has its own independent without-replacement 600-case clock; a response-family pass is not a historical-data epoch. Learn the actual source permissions and fine physical weights. K=2 in this phase is a deliberate control and is not adaptive-K evidence.

**Phase B — train the supported action family.** Rotate root/two/four-packet actions in blocks covering a complete shuffled primary-data pass, rather than changing among many untrained cuts every update. Retain the full-access replay frequency. Sample new physical queries at every update. Train the same G weights across actions; do not create separate field experts or one checkpoint per cut.

P follows the same physical case/query/capacity stream and update budget. It does not need a packet cut, but sees exactly the same sparse/full schedule. Keep its independently trained scores and correct memory checkpointing.

Finish at least two complete primary sparse-data passes for EACH of the three actions before calling the action-family experiment sufficiently exposed for frontier comparison, unless an actual irreparable numerical failure occurs. For Thermal additionally complete at least two aggregate historical replay passes (1,200 distinct-case visits with wrapping counted), and at least 100 visits per response family over the whole maturation branch; balance action exposure within families. The historical requirement is aggregate, not 1,200 visits for every cut. For Wind the required two scaffold passes plus six action-family passes correspond to about 3,264 primary native-row visits, plus scheduled full-access replay, within the 6,000-update ceiling. If resources expire first, report `resource_censored`; do not substitute an arbitrary 200/500 update endpoint for convergence.

This training menu is intentionally small. Do not evaluate all 26 cuts as trained choices. Additional cuts may be exercised later only after a stable smaller family demonstrates useful behavior.

### 3.4 Fair initialization and run identity

**Wind:** warm start both G and P from their matched u100 model/scorer checkpoints. Their first 50 full-access steps were matched; u100 precedes the destructive stress mix. Keep optimizer state where compatible, record the amended schedule and changed full-shadow semantics, and use a new experiment identity. Do not describe it as an exact continuation of the historical recipe. If both u100 states are not recoverable, use identical intact Run2110 W-full u1500 physical states and repeat the shared short warm-up for both arms.

**Thermal:** warm start G and P from their matched u200 physical/scorer states. The amended curriculum is a new branch of the experiment. Discard neither old checkpoints nor the frozen R4/R5 results. Utility weights do not initialize physical supervision. If the fuller train-only replay shows irreversible physical drift, one matched reset to Run1804 is allowed as a documented remedy—not one reset for only the unfavorable arm.

No warm start from a development-selected inverse outcome. No mix of G's favorable checkpoint with P's endpoint while calling it a matched-duration comparison.

### 3.5 Loss and protection against full-model drift

For G and P, retain the existing reference roles and physical output conventions. Use

\[
\mathcal L_{\rm sparse}
=\sum_r\omega_r\,\frac{\|F_{\theta,H}(d,c)-y\|^2_{W_r}}{s_r^2}
+\mathcal L_{\rm finite/decision}\quad\text{(Thermal, measured labels only)}.
\]

The fixed scales \(s_r\) come from the existing training-only normalization, or one documented pre-fit recalibration shared by both arms. Do not redefine them to make a failed score pass.

Full-access replay has both reference value supervision and a small retained-incumbent anchor:

\[
\mathcal L_{\rm full}
=\sum_r\omega_r\frac{\|F_{\theta,1}-y\|^2_{W_r}}{s_r^2}
+\lambda_a\sum_r\omega_r\frac{\|F_{\theta,1}-F_B\|^2_{W_r}}{s_r^2}.
\]

The teacher is detached and is a regularizer, not physical truth. Start \(\lambda_a\) with one train-only gradient-scale calibration so its nonzero gradient is approximately 10% of the reference-value gradient; clip the scalar to a documented sensible range rather than allowing division by a near-zero gradient. Because the anchor gradient is zero at exact identity, calibrate on the warm-start state or after the first small native adaptation block. If the anchor remains exactly zero, use a small fixed initial value and report it. No online reweighting to chase the development set.

Use separate clipping/scaling for physical and route parameters so an oversized route gradient does not silently rescale the physical update. Report preclip norms, actual update norms, and fixed-panel loss trends. Preserve same-input physical-gradient checks for the hard/soft training bridge.

Full access is a scheduled training control, not a fallback substituted into a failed sparse sample. Evaluate hard sparse prediction as it actually runs.

---

## 4. Fair maturation is measured in exposure and learning, not run labels

### 4.1 Required clocks

Track four simple clocks in the normal training log. Use the separate primary/historical definitions above rather than one ambiguous epoch counter:

1. Physical optimizer updates and route updates separately.
2. Effective data passes: processed native cases divided by eligible training cases.
3. Sparse exposure by layout/direction or physical family, cut, capacity, and phase.
4. GPU-associated active job time and measured update-time distribution.

Thermal also records historical-case coverage and response-family coverage separately. Eleven correlated stencil states are not eleven independent layouts. Wind's three directions are distinct input rows but share one layout family.

Use shuffled without-replacement passes for the main data schedule. Record fresh-query sampling independently. Cycling only eight response neighborhoods does not give broad historical-value coverage.

### 4.2 Reviews are measurements, not automatic failure gates

Reviews should occur at the current checkpoint, after 0.5, 1, 2, 4, 8 and (where reached) 12 effective passes, plus whenever the capacity/cut stage changes. A review may be lighter than a full final report.

At each review, run the same fixed audit panel and a refreshed query panel. Report:

- reference hard/full role errors against the retained incumbent;
- train and development errors separately;
- per-layout tail, not only an aggregate;
- source-mask turnover and saturation;
- per-cut exposure and adequacy;
- full-access student drift;
- actual gradient/update behavior;
- timing and resource projection.

Do not stop a healthy, improving run just because it has not yet beaten the mature incumbent. Do not continue blindly because losses are finite.

### 4.3 Practical convergence assessment

After the minimum cut/data exposure, assess fixed-panel trends over at least three reviews spanning two full data passes. A working definition of a plateau is less than roughly 1% change in the balanced validation risk and protected tail, with similarly stagnant training risk, after one learning-rate reduction and an additional complete pass.

This is a diagnostic heuristic, not a convergence proof. Report the measurements. Distinguish:

- **Still learning:** train and development improve; continue within budget.
- **Underfit within current scope:** both are poor; test one train-only overfit/scoping remedy.
- **Overfit:** train improves while family-level validation persistently worsens; more identical updates are not the preferred remedy.
- **Unstable:** repeated fixed-case spikes or mask churn; repair rate/curriculum/gradient behavior before longer training.
- **Empirically plateaued:** no useful improvement after sufficient exposure and the one rate reduction.
- **Resource-censored:** time or update ceiling reached before the above assessment is possible.

Do not use learning-curve extrapolation as proof of eventual superiority. Show the observed curve and honest remaining uncertainty.

### 4.4 Boundaries on computation

**Total authorized envelope for this proposed round: 48 GPU-associated active job-hours summed across both authorized devices.** This is a ceiling, not a target, and not 48 hours per GPU.

Suggested allocation:

| Work | Ceiling |
|---|---:|
| Native diagnosis, regression tests and bounded remedies, both lanes | 3 GPU-hours |
| Matched Wind forward maturation, both arms together | 14 GPU-hours |
| Matched Thermal forward maturation, both arms together | 24 GPU-hours |
| Frontier evidence, selector fitting, final fields/controls/cost | 5 GPU-hours |
| Inverse conditioner repair and small readiness check | 2 GPU-hours |
| **Total** | **48 GPU-hours** |

Per-arm physical-update ceilings: 6,000 additional Wind updates and 2,400 additional Thermal updates. All optimizer calls, including route and small utility updates, also count toward a 24,000-call round ceiling. A single optimizer updating both physical and route groups is one call with two component counters; do not double-charge it as two calls and then also charge its physical step again.

Measure 50–100 representative steady-state updates before forecasting the time needed for minimum exposure. If the forecast does not fit, prioritize matched primary-capacity forward exposure and final evaluation; reduce optional oracle breadth or inverse work first. A maximum 20% reallocation between rows is allowed while respecting the total. If the required minimum still cannot be reached, stop at a normal checkpoint with a costed continuation recommendation and the `resource_censored` label.

No automatic 5,000-epoch training. The proposed longer budget is meant to answer the user's concern about premature rejection without creating an unbounded job.

No new Thermal reference solve or Wind CFD solve is authorized by default. The last report records only six unreserved Thermal attempts remaining under its live allowance; verify, preserve, and do not consume them for routine training or figure generation.

### 4.5 Bounded remedies, not broad sweeps

Allow at most two material recipe remedies per dataset before selecting the maturation recipe. Each gets at most 150 real updates, charged inside the diagnosis budget. Examples: reduce route LR; correct the full-budget shadow; soften one excessively discrete mechanism's cap; expand a demonstrably blocked physical parameter scope; fix an observed normalization error.

Reuse already validated code. One minimal reproducer and at most two implementation alternatives are enough before reporting an unresolved engineering blocker. Do not create another suite of numerous one-off approval gates to manage these pilots.

---

## 5. Reference risk must not improve merely because its denominator worsened

Retain the legacy sparse/same-student-full ratio for continuity, but never use it alone for a quality claim.

For case i and role r, report three values at every selected checkpoint:

\[
E^B_{ir},\qquad E^{\rm full\ student}_{ir},\qquad E^{\rm hard\ student}_{ir}.
\]

A sparse model can pass `hard <= 1.10 * full_student` after both have deteriorated. This is not retained-incumbent fidelity. The previous Wind result is an example, not an arithmetic error in its report.

For the primary organized-forward promotion, use the fixed retained reference allowance

\[
E^{\rm hard\ student}_{ir}\leq 1.10E^B_{ir}+\epsilon_r,
\]

alongside balanced risk, tails, and the matched direct arm. Here \(\epsilon_r\) is an explicitly measured software-numerical allowance or a separately justified reference-discrepancy quantity; those two meanings must not be conflated. This is a research comparison, not a universal engineering tolerance.

Do not require every noisy individual sample to pass before allowing research training to continue. Promotion and inverse use demand stronger evidence than continuation of an improving experiment. A small declared fidelity compromise may later be justified by inverse benefit, but that tradeoff must be stated explicitly rather than introduced by a drifting reference.

---

## 6. Evaluation that separates maturity, architecture, and execution

### 6.1 Matched forward tables

For each lane include:

- retained native source B;
- both new students in their own full-access modes;
- grouped and independently trained direct hard primary modes;
- the grouped model at each genuinely trained cut;
- the frozen 0.75 stress check, separately labelled;
- the selected adaptive policy only after Section 7.

Use identical physical queries for paired evaluations, physical units, equal-family/layout summaries, and component/role tails. Show exact endpoint and selected checkpoint separately, and distinguish equal-update from per-arm validation-selected comparisons. Selecting one checkpoint should consider the protected physical roles, not merely a favorable aggregate.

Wind: use fixed volume, downstream, hub, near-turbine and background panels, then the established 90-row sampled evaluation and at least one typical plus one difficult complete native plane. Bootstrap uncertainty at layout level rather than pretending three directions or thousands of cells are independent cases.

Thermal: evaluate broad stored absolute fields, material/interface roles, measured finite responses, per-module peak changes, and pressure increments. Keep existing inspected families as development. Report the local analytic-wake/shared-grid reference accurately; no CFD claim.

### 6.2 Count-matched and same-weight controls

At selected weights compare:

- grouped cuts versus the same-case root source union;
- grouped versus direct P at matched live pair counts, as a diagnostic-only reprojection;
- geometry-selected support;
- a feasible degree/size-preserving rewire with the actual number of changed links;
- case-varying versus fixed training-population organizer features.

If rewiring makes zero effective switches, label the control inactive and do not infer robustness or irrelevance. Do not force an invalid rewire simply to create a nonzero statistic.

G/P differences mix adapted physical weights and access design. Same-weight interventions explain current computation; separately trained matched arms estimate system-level performance. Neither alone proves physical causality.

### 6.3 Training-order sensitivity

Retain a common data/query/capacity sequence for matched G/P. Report optimizer and exposure clocks as well as wall time. Do not change only P's test density to make a desired comparison; count-matching is a separately named condition.

If a promising result survives maturation and budget remains, reserve one short independent-seed verification at the selected recipe rather than spending the same budget on many new hyperparameters. If no replication is run, state that plainly.

---

## 7. Learn adaptive K from matured actions, not stale and undertrained cuts

### 7.1 Rebuild the measured action table

Only after selecting a forward/scorer version, construct fresh training-only candidate measurements for the three trained cuts plus full access. Use varied module counts, multiple layouts at the same M, and native operating directions/contexts.

Use up to 24 Wind training layouts (all available directions where the evaluation budget permits) and the existing Thermal fit families. Begin with baseline plus one resolved perturbation on Thermal; expand only a small number of promising cut comparisons to the full stencil. Use a disjoint query repeat on a subset. Do not spend the entire phase enumerating 26 cuts on ten families again.

Every row contains the exact action, role errors against reference, same-student full distortion, fixed-incumbent risk, canonical and live work, nonredundant K, and whether that cut received the required training exposure. The table belongs to its fixed physical/scorer weights. Refresh affected rows after any further joint change; never feed stale labels to the final selector and claim consistency.

### 7.2 Action-aware feature representation

Replace mean-node-only utility with a small permutation-invariant action encoder. For each packet e and typed mechanism m, compute input-only summaries such as

\[
 z_{e,m} = \left[
 z_e^{\rm receiver},\ b_m,\
 \sum_s\mu_s b^m_{es},\
 \frac{\sum_s\mu_s b^m_{es}\psi(z_s,x_s-c_e)}{\sum_s\mu_s b^m_{es}+\varepsilon},\
 \frac{\sum_s\mu_s(1-b^m_{es})\psi(z_s,x_s-c_e)}{\sum_s\mu_s(1-b^m_{es})+\varepsilon},\
 \text{source count, spread, degree quantiles, score-margin summary}
 \right].
\]

These are planner summaries, not pooled physical field values. Use current pre-interaction source inputs, receiver geometry, typed permissions and known context. Do not pass reference error, clean inverse targets, case ID, partition ID, or post-reference features.

Aggregate packet summaries with at least mean AND sum/count plus max, rather than only a normalized mean that loses cardinality. Include per-role receiver descriptors and explicit K. Keep the head small; twelve/ten-family labels cannot justify an enormous network.

Compute candidate work exactly from the small canonical action; do not learn a work head when the quantity is already available cheaply. Account for that plan-scoring cost in inference.

### 7.3 Risk targets and learning

Use signed log role-risk targets rather than clipping every improvement to zero:

\[
 y_{i\mathcal C r}=
 \log\frac{E_{i\mathcal C r}+\epsilon_r}{E^B_{ir}+\epsilon_r}.
\]

The denominator is used only to form the training label. The inference head predicts this risk from inputs and the candidate action. When numerical floors dominate a role, retain its absolute error and mark the comparative signal unresolved.

Use a robust regression loss plus a small within-case cut-ranking loss. Fit and compare one simple kernel/ridge or nearest-neighbor action baseline on the SAME family splits; this can reveal whether neural optimization or missing action information is the immediate problem. Do not call hundreds of cut rows independent training families.

First test whether the representation can fit the small training table and rank its known good versus bad cuts. If it cannot, investigate features/scales before spending more native model calls. At most one feature or optimizer remedy follows this diagnostic.

Use family/layout-grouped cross-fitting to estimate optimistic prediction errors. A held-fold residual margin or small ensemble can supply an empirical upper-risk estimate; do not claim formal coverage with this small, selected, nonexchangeable population. Final deployment thresholds are calibrated on training folds only.

### 7.4 Selection and meaningful K

Choose the lowest exact work action among predicted adequate cuts, using nonredundant packet count only as a tie-break. Otherwise return explicit full access and `unsupported_at_budget`.

Report:

- measured oracle availability;
- training-table and held-family ranking;
- false-safe and false-reject cut choices;
- selected versus best measured acceptable cost;
- supported sparse deployment rate;
- K distribution at fixed capacity and within M;
- whether a simpler fixed cut attains the same tradeoff.

If matured actions do not contain adequate sparse choices, classify the limitation as **representation/capacity at the tested setting**. If good cuts exist but selection misses them, classify it as **amortization/calibration**. This distinction determines the next research step.

Do not demand that K vary when the measured best adequate action is constant. The adaptive-K claim remains unestablished in that outcome, but the predictor can still be useful. No entropy, variance, or minimum-K penalty is added to create apparent diversity.

---

## 8. Execution: remove obvious repetition, but do not let a kernel project consume maturation

Keep the existing compiled-plan and rectangular-subset reference paths. Before optimizing a long run:

- avoid recomputing the same organizer scores and budget projection solely for per-update logging;
- return the actual hard-plan support/work from the existing forward call;
- collect detailed path diagnostics on scheduled reviews, not every small operation;
- cache only appropriate fixed input geometry, never hidden trial designs, target fields or stale model states;
- keep normal plan-validity/security checks intact at their existing boundaries.

Native attention source normalization and denominators remain unchanged. Full access should use the native path and no unnecessary soft organizer execution. Use activation checkpointing or source chunks for P as already verified; compare gradients after a memory amendment.

Measure complete encoding + organizer + preparation + decode at Q64/Q1024/Q8192 and one large shape, with matched precision/chunking and interleaved repeats. One sparse scientific model may still run slower because it scores and masks dense pairs. Report that rather than withholding all learning analysis.

No new Triton/CUDA kernel in the primary task. Any small execution remedy must fit within the diagnosis allocation and preserve the operator on real inputs.

---

## 9. Inverse preparation: repair conditioning, then postpone the superiority claim

### 9.1 Do the inexpensive representation repair

Upgrade the shared denoiser's sensor pathway to retain location–value binding. A suitable small formulation is

\[
 t_s=\mathrm{MLP}([\phi(x_s),\bar y_s,\text{validity/channel flags}]),
\]

\[
 o_i=\sum_s\mathrm{softmax}_s\left(
 q_i^\top k_s/\sqrt h+eta(x_i-x_s)+\log a_{si}
 \right)v(t_s),
\]

with explicit masking/empty-support handling. Dense uses full admissible a; graph uses the frozen forward-derived access prior. Do not collapse sensor tokens through a single linear mean first. Both arms share model size, normalization, data and sampling schedule.

At current candidate designs, recompute candidate embeddings and links without clean hidden variables. Retain the complete mask and known-observation protocol. Any per-sensor train normalization must support alternate sensor sets through an explicit rule; do not index train statistics with future task IDs.

Test: simultaneous sensor permutation invariance; fixed-location value-swap sensitivity; dropped-channel/observation handling; no clean-design leakage; no gradient into the frozen forward model; and identical full-link graph/Dense behavior where expected.

### 9.2 Bound designs in the known admissible domain during generation

For Wind, the current sigmoid enforces a global design box, but that box is not the per-case native domain. Use the intersection of the task-supplied public design region and native support, reduced by the appropriate rotor clearance where required. Do not derive a hidden-position label from clean hidden geometry. If native support was itself selected using the hidden layout, disclose the information available to the inverse task and test a common public-domain control.

Visible turbines remain fixed. Rejection or differentiable/iterative separation handling applies identically to graph and Dense; report pre- and post-repair quality separately. A repaired valid layout is not a calibrated posterior sample.

For heat, keep nonnegative supplied-total parameterization. Fit any latent scaling on training data only, and report extreme-allocation frequency relative to the training distribution. One-hot allocation is not mathematically invalid; its physical observation mismatch and distributional support are the relevant questions.

### 9.3 A readiness check, not another full inverse study

Use at most two GPU-hours for the repaired conditioner and a small matched train-only fit/sampling check. The purpose is to show that the architecture can use observation values at their locations and maintain known geometry constraints. It is not a new generalization result or a reason to select forward weights.

Do not retrain a large inverse experiment on the old unsupported Stage-C cut simply because a frozen-interface pipeline exists. A later matched inverse reuse study must bind to the newly selected forward/organizer version. If only an unsupported graph remains, retain Dense conditioning as the control and explicitly defer the graph superiority test.

No additional local reference or CFD solve follows automatically. The eight prior local checks remain historical evidence.

---

## 10. Code organization

Prefer extending maintained modules over appending another thousand-line experimental script.

| Area | Existing location | Intended change |
|---|---|---|
| Grouped budget semantics | `interface_fields/input_cover_organizer.py` | Correct full/empty thresholds; expose actual realized actions |
| Cut/budget math | `interface_fields/budgeted_frontier.py` | Keep valid union accounting; add trained-cut catalogue and richer utility adapter |
| Shadow training | `interface_fields/native_joint_shadow.py`, `native_direct_pair.py` | Full-access bypass and separated gradient tests |
| Wind train entrypoint | `Case_WindFarm/scripts/run_active_packet_reuse.py` | Amended maturation configuration, full data passes, meaningful reviews |
| Wind native workflow | `Case_WindFarm/src/windfarm/workflows/` | Reusable bounded maturation/evaluation functions instead of duplicated script logic |
| Thermal forward | `Case_ThermalChannel/scripts/run_active_packet_forward.py` and `response_control/active_packet.py` | Primary-only schedule, trained cut exposure, broad replay, matched continuation |
| Thermal selector | `Case_ThermalChannel/scripts/run_active_packet_stage_c.py` | New-weight action table and family-wise selection tests |
| Shared inverse | `honf_inverse_core/models/frozen_packet_diffusion.py` | Nonlinear spatially resolved sensor conditioning, historical class retained |
| Wind inverse geometry | `Case_WindFarm/src/windfarm/inverse/packet_completion.py` | Public-domain parameterization and validity tests |
| Reporting | `AGENTS.md` and final study report | Preserve and follow the three-goal/visual reporting policy |

Do not weaken old checkpoint/security validation to accommodate new run identities. Add an explicit new configuration/workflow while retaining historical load semantics. Use normal Git history, configuration, checkpoints, and tests. Do not add cryptographic manifests, contract freezes, approval tables, or new blocking infrastructure.

---

## 11. Tests and actual execution

A focused suite should cover:

1. Budget-one hard/soft identity and route-gradient behavior for G/P.
2. Positive-infinity empty handling distinct from negative-infinity full handling.
3. Exact hard value and physical/input gradient isolation under the shadow.
4. Nonzero restoration signal for an omitted native QE source where the measured loss favors restoration.
5. Cut-path availability and counted training exposure.
6. Same G/P input, query, primary-capacity and exposure streams.
7. Unequal measures, MM self/padding, ineligible routes and actual per-receiver support.
8. Candidate utility sensitivity to different source actions on the same receiver cut.
9. No held-family labels in risk fitting and no stale action table after weight updates.
10. Sensor permutation versus value-location reassignment.
11. Public-domain geometry handling and visible-design immutability.
12. Historical checkpoint loading and existing security/artifact behavior.

Then execute real optimizer steps at low/high module count, actual periodic evaluations, and synchronized final timing. Passing tests cannot substitute for any of those measurements.

---

## 12. Required report and figures

### 12.1 Opening explanation: one page before the technical detail

Begin with a table for **predictor, organizer, inverse system** giving:

- what we gained;
- what did not work;
- whether the candidate was actually mature enough to judge;
- the single most relevant measurement;
- the next unresolved question.

Add a short paragraph explaining the decision: continue, retain, redesign, or resource-censored. Do not use “Goal complete” as a synonym for scientific success. Explain G, P, retained B, oracle action, learned research action, and deployment fallback once in plain language.

### 12.2 Maturity figure is mandatory

Plot fixed-panel reference errors against BOTH effective training passes and active GPU-hours. Mark warm-up, scaffold, action-family introduction, LR change, and resource stop. Include train/development, hard/full, retained-reference lines, and tail behavior. Add an exposure heatmap for layout/cut/capacity so “fully trained” is an evidence-based statement.

### 12.3 Necessary measured visualizations

Retain a compact main set, normally 6–8 selected figures:

- Wind reference/prediction/residual maps on a representative and difficult native plane;
- Thermal fluid/interface/material fields and finite responses, including the failing material role;
- source access drawn on physical geometry, K/support statistics, and bypass labels;
- same-M and changed-context examples with valid source/receiver correspondence;
- measured oracle frontier versus learned selection, including false-safe and false-reject choices;
- fidelity versus realized work and complete latency, with fixed/full/direct controls;
- inverse-conditioning swap/validity readiness panels, or an explicit “not retrained” summary using labelled historical evidence;
- design/sample trails only for actual executed samples, with invalid outcomes retained in the summary.

Use common color scales for comparisons, state clipped fractions/maxima, display empty native bins honestly, and never invent streamlines or CFD fields from metric aggregates. Each figure needs a quantitative paragraph followed by one plain-language conclusion.

Follow current `AGENTS.md`: a short figure index; PDF masters; minimal PNG companions for Markdown embedding; tested relative links; generated arrays/figures and one-time renderers local and ignored. Remove redundant figure exports only under the existing reporting rule; preserve numerical evidence and historical results.

### 12.4 Detailed appendix

Keep the full per-role, per-layout, per-family, selected/endpoint, counts, configuration, runtime, failures and uncertainty tables. The user's request for a clearer main narrative is not permission to discard quantitative detail.

---

## 13. Definition of done

This round is complete when:

- the full-access shadow inconsistency is resolved and tested natively;
- old training exposure and the actual cut menu are reconstructed;
- one controlled G/P primary-capacity maturation pair has run in each dataset within the stated resource ceiling;
- maturity or resource censoring is explicitly diagnosed, with curves rather than a step-count assertion;
- frozen-forward risks are assessed against both the retained incumbent and the direct control;
- available matured cut opportunities are separated from learned selector mistakes;
- an action-aware selector is actually fitted where evidence permits, or its precise data/fit limitation is demonstrated;
- the inverse sensor-binding/geometry issues receive a bounded repair check without contaminating forward selection;
- the final report explains predictor, organizer and inverse results plainly and includes validated measured figures;
- durable changes and report are tested, committed and pushed under the current repository rule; local/remote status is verified and raw evidence remains local.

The desired result is not necessarily the longest run or the lowest K. It is a defensible answer to whether the organized physical computation can learn adequately and be selected reliably after a fair training opportunity.

---

## Source notes for Codex

Read the local complete files rather than relying only on this summary:

- `docs/reports/HONF_Active_Organization_and_Frozen_Inverse_Reuse_Study.md` — latest G/P training, frontier and inverse evidence; especially the u100/u500 paths, Thermal Stage B/C, and resource section.
- `Case_WindFarm/configs/active_packet_organizer_reuse.json` — old 50/150 warm/primary schedule and full/0.90/0.75 mix.
- `Case_WindFarm/scripts/run_active_packet_reuse.py::_seeded_cut`, grouped training loop, `_direct_projected_access` — actual exposure, clipping, and threshold semantics.
- `src/honf_forward_core/interface_fields/input_cover_organizer.py::plans_from_scores`, `score_frontiers` — full-access soft behavior and utility inputs.
- `src/honf_forward_core/interface_fields/budgeted_frontier.py::FrontierUtilityHead` — mean-node pooling and current target normalization.
- `src/honf_inverse_core/models/frozen_packet_diffusion.py::ConditionalPacketDenoiser` — sensor encoding and aggregation.
- `Case_WindFarm/src/windfarm/inverse/packet_completion.py` — condition features and global versus native design bounds.
- `AGENTS.md` at the current branch tip — durable code and figure/report policy.

The proposed cut-coverage and sensor-pooling arguments were independently checked with small deterministic algebra examples. They are not native checkpoint results. Execute the native reproductions and report their actual scope.
