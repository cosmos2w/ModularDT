# HONF: response-competent forward refinement

## Goal-mode brief

**Research starting point:** `cosmos2w/ModularDT`, branch `agent/honf-core-next`, reviewed at `197a68238c0c617a61225665cc18c2311548e975`.

**Central goal:** learn from multi-field reconstruction, discover a reusable organization of module–environment interactions, and reuse that organization to generate and evaluate modular designs. Reconstruction is the learning vehicle; the organizer is the representation; inverse generation is the eventual application.

**This phase's question:** can the existing inexpensive joint interface become part of a forward model that preserves field accuracy and predicts the direction and magnitude of physical changes better than an equally trained separable interface?

Retain the once-wrapper tensor organizer, source-resolved fine readers, native coarse/local context, and P0/P1/P2 coupling. Do not invent another organizer, recursive Tree, pruning schedule, inverse generator, or model portfolio. Change the learning conditions: permit controlled physical co-adaptation, train on several existing response families, and actually penalize the benchmark's known heat-to-flow null.

Run **one matched two-arm refinement** of the existing H-add and H-joint architectures. Both receive the same reconstruction, response, and null objectives, the same trainable physical scope, and the same data/exposure. This is a practical model-improvement experiment, not an isolation of whether data breadth, loss scaling, or co-adaptation alone causes improvement.

**Expected deliverables:** two real refined forward checkpoints; an understandable field/cost/response comparison; a measured test of whether joint organization helps physical responses; and a small replay of choices among already solved designs. No generative inverse training or newly solved design is authorized.

**Hard scope:** ThermalChannel only; fixed25_v1 primary data; GPUs 1/2 only; target 500 new refinement epochs per arm after an e100 review; conditional matched extension to 1,000 within 12 aggregate GPU-associated hours and 8 elapsed hours. **Zero new solver attempts. The cumulative physical-reference ledger remains 326/326.**

---

## 1. What the last experiment established—and what it did not

### 1.1 A short baseline dashboard

Numbers below are equal-case native-unit RMSEs on the 22 exposed validation cases. The old-model columns are retained scientific references, not identically trained arms of the next experiment.

| Quantity | G-fast e1000 | Tensor-H e1000 | H-add selected fit300 | H-joint selected fit100 |
|---|---:|---:|---:|---:|
| Fluid T | 1.17451 | 0.940162 | 1.05314 | 1.05924 |
| Surface T | 1.23966 | 1.09030 | 1.09523 | 1.09986 |
| Material T | 1.04896 | 0.930890 | 0.885417 | 0.900310 |
| Fluid u | 0.0306896 | 0.0317718 | 0.0320716 | 0.0302616 |
| Fluid p | 0.0143447 | 0.0173950 | 0.0149430 | 0.0150415 |

H-joint improves the three thermal means over G-fast, but not over selected H-add. At matched fit500 it improves several flow means while its fluid/material T means are effectively tied with, and slightly worse than, H-add. Neither comparison establishes a uniformly superior forward model. Keep the stronger retained Dense-D25 checkpoint in the final reference dashboard when its exact comparable evaluation can be verified; do not make a gain over G-fast stand in for beating the strongest available predictor. Run1804 remains an unequal-data, mature historical reference, not a matched development control.

The selected H-joint is **soft at fit100**. Exact fit500 K variation is a different endpoint. Do not combine fit100 accuracy or interventions with fit500 sparse-group claims.

### 1.2 Computation is now affordable, but not sparse execution

The completed interface fits averaged about 10.25 seconds per 150-case train-plus-validation epoch, about 85 minutes of measured epoch work per arm for 500 epochs. Peak training allocation was about 5.19 GiB. Whole-round GPU-associated process accounting was 3.034311 hours, not CUDA busy time.

On the recorded B8/Q1024 high-M panel, complete inference medians were 0.210556 s for G-fast and 0.253645 s for H-joint. Input forward-plus-VJP was 0.455120 s and 0.552279 s. These are approximately 20% and 21% overhead, respectively; they are not a speedup. The low-M results and small repetition counts must remain disclosed.

All four retained models execute the same 43,972,896 padded fine-MLP rows over the all 22 normal panel. Frozen parameters reduce optimizer work, not the need to propagate activations through the physical model to train an upstream interface.

### 1.3 The joint term is active and locally useful

At selected H-joint fit100, removing only I increased fluid-T error by 8.694%, 9.458%, and 3.762% on 0277/0294/0687; it reduced error by 0.310% on 0291. Surface/material errors show the same favorable/unfavorable case split. This is a useful end-to-end model intervention, not proof of universally useful grouping, unique physical decomposition, or physical causality.

The current formula and once-wrapper mechanism are therefore worth retaining. Repeating a large collection of S/R/I visual diagnostics is not the next research objective.

### 1.4 Actual physical changes remain the decisive weakness

The six primary finite responses come from **three layouts, each with two opposite heat transfers**, not six independent layouts or six independent local heat directions.

| Mean finite-response RMSE | G-fast | Tensor-H | H-add300 | H-joint100 |
|---|---:|---:|---:|---:|
| Fluid T | 0.0431030 | 0.0278715 | 0.0467881 | 0.0437068 |
| Surface T | 0.0543307 | 0.0332353 | 0.0684079 | 0.0582474 |
| Material T | 0.0536341 | 0.0343091 | 0.0677365 | 0.0575351 |

H-joint improves over H-add but still loses to G-fast and Tensor-H on all three means. On 0291 plus, the reference mean fluid-T change is +0.0289138, but H-joint predicts -0.0354399. Its minus direction is also wrong. Such a model can point an inverse optimizer in the wrong direction despite a plausible absolute field and numerically consistent AD/FD.

The local reference has exactly zero u/v/p/omega changes under heat-only perturbations. All current models leak heat changes into flow. The null loss in the completed experiment was explicitly zero; no exact physical dependency restriction was built into its architecture.

Eleven states exist from twelve charged attempts. The missing 0277 baseline remains missing. Its minus-to-plus span is a secondary two-endpoint comparison, not a recovered baseline or a seventh/eighth primary response.

### 1.5 Why 500 was a sufficient interface-fit review

The last validation windows plateaued or worsened while training error continued to improve. Another 500 identical frozen-host epochs is not justified by those curves. This does not prove that the architecture has reached its capacity with physical co-adaptation or a different response-learning objective.

The next experiment is a **new, explicitly labelled refinement**, not a continuation claimed to repair an accidentally undertrained physical backbone.

---

## 2. Code-informed diagnosis and implementation boundaries

Read the pinned report and these live code paths before editing. Reconcile a later branch tip rather than silently assuming the review commit is still current.

| Existing component | What the inspected code does | Next-phase treatment |
|---|---|---|
| `src/honf_forward_core/interface_fields/tensor_query_interaction.py` | Constructs S/R/I; only QM/QE receive new corrections; uses stable nested-tanh gain arithmetic; defaults to a frozen host | Reuse numerical operator; add an explicit opt-in refinement training policy and an independently declared soft admission mode |
| `src/honf_forward_core/interface_fields/tensor_source_group_residual.py` | Builds one input-only tensor source plan; shared plan across physical phases; refreshes phase content | Retain; no recursive construction, extra physical shadow, or source pooling before the fine nonlinear reader |
| `src/honf_forward_core/interface_fields/dense_pairwise.py` and `typed_hypergraph_field.py` | Evaluate source-resolved MM/ME/EM/QM/QE and ordinary dense physical rectangles | Retain architecture and execution; unfreeze only the declared physical parameters |
| `Case_ThermalChannel/src/channelthermal/training/interface_fit.py` | Enforces that only the new interface parameters train; requires null coefficient zero; preserves the previous experiment's lineage | Keep historical validation intact. Introduce a separate refinement declaration, not a bypass of the frozen-fit checks |
| `.../training/campaign_response.py` | Existing family callback; absolute-field scales; inherited response coefficient calibrated from an initial nominal 5% gradient ratio and capped at 0.1; selected-data exposure restricted to at most four selected anchors | Reuse record loading/receiver matching, but implement an explicit response-refinement objective and a declared auxiliary-data contract |
| `.../workflows/train_forward.py` and `.../training/epoch.py` | Native four-update development epoch and last-batch response callback; separate fit and objective ages | Preserve accumulation correctness, case/query sampling, loss denominators, and actual call accounting |
| `.../training/stop_request.py` | Supported epoch-boundary stop through the existing atomic checkpoint writer | Reuse unchanged |
| `.../interaction_evidence/response_atlas.py`, `response_dataset.py`, `types.py`; `.../response_control/native.py` | Native reference records and differentiable complete-wrapper evaluation | Reuse. Never substitute model predictions for missing reference outputs |

The inspected code supports three concerns, not a proven single cause of every error:

1. The previous optimization could change controls but not the physical kernels or prediction heads that use them.
2. The recent campaign used TRAIN0348 response supervision. Repeated visits to one neighborhood are not broad response coverage.
3. Normalizing small finite changes by absolute-field standard deviations, then applying an inherited small coefficient, is not the same objective as learning accurate relative responses. The actual gradient influence in the new trainable scope must be measured rather than inferred from the old coefficient.

Shared hidden controls can also affect flow and thermal outputs together, and query corrections influence later states through ports. Frozen weights therefore never guaranteed fixed flow predictions. Nonzero known-null supervision is necessary in this proposed empirical refinement, but it is **not an architectural guarantee** of exact heat-independent flow.

The nested tanh may suppress some control sensitivities. Measure that once on training inputs if useful; do not declare saturation the cause from one plot or redesign the gate merely to increase an intervention norm.

---

## 3. Non-negotiable scientific and resource scope

### 3.1 Primary development population

Keep `/data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json` unchanged: 150 primary training cases and 22 exposed validation cases. The reported semantic fingerprint is `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`; distinguish this from a literal-file SHA.

Reuse identical primary case membership, ordering/sampling seeds, Q1024 queries, microbatch8/effective48, FP32 policy, native denominators and selected-training normalization in both arms. An epoch means visiting all 150 selected cases. Do not inflate epoch counts by changing the case definition.

Report all 22 statistics and all available module-count strata. Detailed full fields/graphs default to 0277/0291/0294/0687. Do not choose display cases from favorable errors. The repeated validation panel is development evidence, not an untouched test set.

### 3.2 Explicit small response-data addendum—not a hidden change to fixed25

**This plan proposes and, when adopted for execution, authorizes a separately identified auxiliary response addendum using at most four existing original-TRAIN anchor families.** It does not authorize a full-data training sweep, a primary-manifest change, or ingestion of the newly audited validation responses.

Default fit families are `0001`, `0318`, `0333`, and `0348`, covering M3/5/7/10. Use only their existing baseline and two opposite heat-transfer states for the primary response task. Resolve exact metadata and local files; numeric IDs alone do not prove eligibility.

Default response-development families are `0304`, `0320`, `0335`, and `0350`. Hold all of their response labels out of refinement gradients and coefficient calibration. Their historical use and possible primary absolute-value exposure must be recorded. They are **response-withheld in this refinement**, not necessarily unseen layouts or previously unopened research data.

Before fitting, verify:

- Original source partition and physical-family identity, including the `0001/0273` duplicate relationship.
- No fit-family or duplicate-family overlap with response-development or current fixed4 audit families.
- Exact presence, validity, input normalization, module IDs, query coordinates, and source provenance of every baseline/variant.
- Which fit anchors already belong to the primary150 and which are extra auxiliary anchors. Report `150 primary cases + N extra auxiliary anchor families`, not simply `150 total training cases` if N is nonzero.
- Existing baseline-to-trial labels use the same rounded inputs and receiver joins. Do not mix a stored H5 baseline with a newly rounded trial.

The current native development guard intentionally permits only selected anchors. Do not remove that guard. Add a separate explicit `response_addendum` declaration for this new refinement profile, restricted to the named original-TRAIN families, and leave old profiles' behavior unchanged. Reuse existing provenance and resume machinery; do not build a new generic permission framework.

If a named family is genuinely unavailable, do not silently replace it with a validation family or run a solver. Identify the gap and execute the remaining matched, valid scope only if it still tests cross-family learning; describe the reduced coverage. A single available fit family cannot support the planned cross-family conclusion.

The primary150 normalization stays fixed. Response-only scales use only the declared response-fit families. The expanded training identity is explicit and prevents claims of equal-data superiority over the previous frozen study.

### 3.3 Solver and inverse limits

No new local or CFD solve, no rerun of failed0277, no solve disguised as data augmentation, and no increased cumulative allowance: **326/326 remains unchanged**.

The fixed4 audit labels never enter training, loss-scale selection, or an after-the-fact coefficient search. In this new round they are reused exposed evaluation evidence, not newly held truth. Baseline-to-trial evaluation remains six primary comparisons; 0277 minus-to-plus remains secondary.

No inverse denoiser, diffusion model, generative sampler, or continuous design search is trained or run. The allowed decision-use check is the finite stored-pool replay in Section9.

---

## 4. The two actual forward-model runs

Use the existing H-add and H-joint architectures; avoid introducing new scientific acronyms in the report. Call them **separable-interface refinement** and **joint-interface refinement**, with exact run IDs and configurations in the appendix.

### 4.1 Matched warm starts

Start both from their exact **fit100** checkpoints from the completed matched experiment, not selected H-add300 versus H-joint100. Both checkpoints exist under the100-epoch cadence. Their inherited G-fast physical tensors must still match; their interface parameters differ because each underwent its own matched100-epoch fit.

Record the prior histories honestly: G-fast physical backbone e1000; interface prefit100; new refinement age0. These are not fresh random models and the interfaces do not have identical trained values at this new starting point. They have matched prior exposure and a common physical backbone. Load no pilot or newly audited target-conditioned state.

Create fresh optimizer state for both refinements. Retain all model values and fixed normalizers at initialization. Do not reuse head-only optimizer metadata while claiming a physical-refinement resume. Check each arm against its own parent function at refinement0; cross-arm outputs are not expected to be equal.

### 4.2 Retain soft organization during this phase

Pin both new runs to the same soft admission/donor formulation used at fit100. Keep memberships input-dependent and learned; only the numerical admission formulation is fixed. Rebuild the plan for each changed layout/context/heat input and share it across that wrapper's phases.

Introduce an explicit admission-mode option independent of optimizer age. Do not fake the model's epoch to freeze a curriculum or repeatedly reset training progress. Old soft/blend/sparse schedules remain replayable for old checkpoints.

This is **not** a claim that soft groups solve adaptive K. The selected best H-joint was soft, and neither current cost nor physical evidence justifies another forced transition. Exact K and future physical pruning are postponed. Optional effective participation statistics belong in the appendix, not the headline success criterion.

### 4.3 Physical co-adaptation scope

Train the new query organizer and the existing fine interaction/decoding path that can change the response. The initial allowlist should include:

- `core.backend.tensor_query_interaction.*`;
- `core.backend.mm_message`, `me_message`, `em_message`, `module_update`, `env_update`;
- `core.backend.query_module_message`, `query_module_output`, `env_query`, `env_attention`, `env_geometry_bias`;
- the existing native field head and initial/refinement port prediction heads, resolved to actual named parameters before launch.

Freeze input/output normalization, input encoders, native coarse/local context builders, the inherited global organizer/calibration projections, and the Stage-A local surrogate. Do not accidentally unfreeze them through an optimizer group, weight decay, or a general model `.train()` helper.

Use the same allowlist by semantic component in both arms, and report exact names/scalar counts. If source ownership differs from these paths, document the equivalent current paths rather than guessing a broad `backend.*` prefix. Do not introduce early pooling or replace predicted ports with target ports.

Suggested initial AdamW learning rates: `1e-4` for the existing query interface and `1e-5` for the newly trainable physical path, with weight decay `1e-5`. These are proposal settings, not measured optima. Use the same settings in both arms, fixed through the primary500. No learning-rate portfolio.

The old frozen-fit constructor and campaign checks intentionally prohibit this operation. Implement an opt-in refinement policy and a new refinement identity; preserve the original frozen mode and its tests. Code must reject a silent change of trainable scope on ordinary resume.

---

## 5. Learn fields and physical changes together

Let x contain geometry, prescribed operating context, and non-design attributes; let h be the vector of physical heat-source amplitudes; let r identify a physical receiver and output role. Use the complete native operator:

\[
\hat y_\theta=F_\theta(x,h),\qquad
\Delta\hat y_\theta=F_\theta(x,h+\delta h)-F_\theta(x,h).
\]

The heat entries are dataset source amplitudes, not certified watts. A fixed sum is an input constraint, not a demonstrated physical energy balance.

Both arms optimize the same objective:

\[
\mathcal L=
\mathcal L_{\rm primary\ reconstruction}
+\beta\mathcal L_{\rm auxiliary\ absolute}
+\lambda_\Delta\mathcal L_{\rm thermal\ change}
+\lambda_0\mathcal L_{\rm heat\!\to\!flow\ null}.
\]

### 5.1 Primary and auxiliary absolute fields

Retain the existing primary native multi-field/port/material objective and its denominators. This is the main protection against degrading a good predictor to fit a few responses.

On each selected response pair, also supervise both absolute states using their actual stored fields. A suitable fixed starting choice is `beta=0.25`, after making this a case-normalized auxiliary term. Do not allow the number of material receivers or active modules to silently give a family extra weight. This addendum supplies real labels; no teacher pseudo-labels are introduced.

### 5.2 Positive thermal changes with response-scale normalization

For each thermal role a in {fluid T, surface T, material T}, define a training-only response scale

\[
s_a^\Delta=\max\left[
\sqrt{\operatorname{mean}_{f,t\in\mathcal D_{\rm response-fit}}
 \|\Delta y_{f,t,a}\|_{w}^2},\ \epsilon_a
\right].
\]

Use normalized positive quadrature weights within each role and equal family/direction weights. A practical declared floor is the larger of a training-only storage/roundoff indicator and `1e-3` times that role's primary-training standard deviation. Record both values; this floor prevents unstable division and is not a CFD or grid-error certificate.

Then

\[
\mathcal L_{\rm thermal\ change}
=\frac13\sum_a
\left\|\frac{\Delta\hat y_a-\Delta y_a}{s_a^\Delta}\right\|_w^2.
\]

The q-normal proxy may retain a small auxiliary change term, with fixed weight0.25 relative to the mean thermal-role term. Keep it separately reported; do not optimize a claimed conservation law from this proxy. Do not add a new mixed-heat interaction target or infer irreducible higher-order physics from a nonzero control I(q,s).

The default task is heat-response refinement. Geometry-response reference data remain available for later work but are not silently mixed into this new primary loss. This makes the response task interpretable and prevents a cycle dominated by unrelated geometric variants.

### 5.3 Known heat-to-flow null on diverse primary training layouts

In this benchmark's inspected default generator, fixed-geometry heat changes leave u/v/p/omega unchanged. For a valid heat-only perturbation,

\[
\mathcal L_0=\frac14\sum_{c\in\{u,v,p,\omega\}}
\left\|\frac{F_c(x,h+\delta h)-F_c(x,h)}
 {\sigma_c^{\rm primary}}\right\|_w^2.
\]

Use nonzero, declared perturbation sizes. Do not make this loss appear small by shrinking all steps. Log both raw increments and increment-per-unit-heat magnitudes. If different step sizes are used, normalize by the declared dimensionless step consistently for training; do not compare unnormalized losses across changed amplitudes.

Cycle one primary training case per epoch. For M>=2 use a deterministic feasible balanced transfer within bounds derived once from the primary training heat data. Alternate fixed fractions of the feasible bound, for example0.05 and0.10, with the same sequence in both arms. For M1, a valid single-source heat perturbation is a null-dependency check, not a fixed-sum inverse move. Record negligible-step skips and their replacement rule before training.

No reference solve is needed for this analytic null. **No thermal target is supplied for these new null-only trials.** Never teach T to remain unchanged, and never impose this null on position changes, on WindFarm, or on a more general heat-coupled flow solver. Place the capability declaration in the Thermal adapter/configuration, not as a universal shared-core assumption.

Reuse the existing native null machinery where it fits; keep both complete calls differentiable. A learned null penalty is not guaranteed exact for arbitrary future inputs. Evaluate that limitation explicitly.

### 5.4 Calibrate once for the new scope

Do not inherit `response_scale` from the frozen-head campaign. The trainable parameters, response labels, and response normalizers have changed.

On a small deterministic training-only calibration panel spanning the four fit families, measure primary-value, response, and null gradient norms in the **declared trainable scope**. Use one shared coefficient pair for both arms, based on pooled initial calibration measurements. Suggested initial gradient budgets are25% of the primary-value norm for positive changes and10% for the null term. Freeze the resulting coefficients before any response-development or fixed4 evaluation is used.

These ratios are an explicit proposal, not guaranteed optimal. Record resulting coefficients and component gradient norms. If a term has zero/invalid gradients, fix the disconnected implementation rather than silently setting its coefficient to zero. Permit one bounded train-only calibration correction, not a post-validation coefficient sweep. Retain clipping, and disclose when it regularly changes the intended effective gradient balance.

No extra optimizer boundaries are introduced merely for bookkeeping. Preserve the native four updates per complete primary epoch and correctly scale auxiliary gradients under accumulation. Explain whether callbacks share the last boundary; verify one accumulated update against a literal matched reference on a disposable small case.

---

## 6. Training schedule and budget

### Stage 0 — bounded preparation and executable check

Target <=45 minutes elapsed for code-informed inventory, data validation, parameter policy, and native smoke tests. Reuse existing tests and numerical artifacts instead of rerunning the historical campaign.

Verify update0 replay against each arm's own parent, one effective48 update per arm, expected changed/frozen parameters, nonzero organizer/physical gradients, correct callback families, resume/stop behavior, and absence of target values in planner inputs. Resolve the already known three-value0687 numerical qualification without widening tolerances or making that tiny discrepancy the central research task. Budget at most one short targeted numerical investigation; retain any remaining miss transparently.

### Stage 1 — 100 new refinement epochs per arm

Visit every primary case each epoch. Cycle fit families and transfer signs equally and apply the declared diverse null schedule. Save ordinary milestones/latest/best-field/plots every100 epochs.

At100, review all 22 sampled field statistics, the actual full-role checkpoint evaluation, fit and response-development changes, null leakage, native cost, and real learning exposure. The fixed4 audit labels stay out of intervention or hyperparameter selection; open their final replay after the primary training decision.

Continue both unchanged to500 if finite training and the cost forecast support the task. A failure to establish inverse readiness at100 is not by itself an early-stop reason. An implementation error, nonfinite states, repeated severe broad-field deterioration, or an exceeded resource forecast is.

### Stage 2 — primary matched endpoint at500

The deliverable is two completed500 refinement endpoints, unless an explicit failure or resource constraint prevented this. Report any partial result as partial; do not mark all research goals met merely because the code exists.

Use the same existing best-field selector over saved100 checkpoints. Report the matched500 comparison as the primary architecture comparison, and selected checkpoints separately. Do not substitute one arm's best response checkpoint or an unsaved favorable age.

Extension of both arms to1,000 is allowed only when saved late training/development curves support a specific remaining learning question and the measured remaining cost fits the ceiling. Do not extend because budget remains. Decide before opening the final fixed4 replay; do not stop a healthy500 run at200 solely because a difficult response still fails.

### Resource management

Hard ceiling:12 aggregate GPU-associated process-hours and8 elapsed hours, including failed starts, tests, evaluations, and closeout. Reserve at least45 minutes for final evaluation, figures, and delivery. The measured previous10.25-second frozen epoch is context only; co-adaptation and new callbacks have a different cost. Forecast from at least10 completed current epochs, with callback scopes included and no nested double counting.

Use physical GPUs1/2 only and verify logical mappings. Do not touch unrelated processes. If another user occupies a GPU, use the other authorized lane, run bounded work at a safe batch configuration with disclosed contention, or do CPU preparation; do not poll indefinitely. Match effective batches, case/query streams, and training budgets even if microbatching must change for both arms. Report associated wall time separately from isolated performance measurements and CUDA allocator measurements.

If the500 pair will not fit, reduce optional diagnostics and dense export duplication first. Do not silently reduce primary case coverage, skip response families, erase failures, or launch detached work beyond the task budget. Save a supported epoch-boundary checkpoint and explain the incomplete scope when necessary.

---

## 7. Evaluation: a small set of decision-relevant questions

### 7.1 Field inference accuracy

Evaluate all 22 at the common500 endpoint and each selected checkpoint. Reuse an evaluation when these are identical. Preserve all 24 native role statistics, per-M means, p90, maximum and paired-case errors in the appendix.

The first-page accuracy table should show only fluid/surface/material T, u, and p, with the two new arms and a compact retained-reference comparison. Show the retained Dense-D25 values only after checking exact cohort/masks/units; mark architecture/input/history differences. Do not replay all 690 cases or call the22 exposed rows a generalization test.

Use complete native grids and material coordinates for the final comparisons. Render the same four representatives from saved arrays. Keep q-proxy, initial/final ports, and peak errors available but do not substitute them for these primary fields.

### 7.2 Actual computation efficiency

Benchmark the two new arms and G-fast on the same device, native precision, receiver chunk, B8/Q1024 low/high-M panels, with alternating order and five warmed inference repetitions. Include a B1/Q8192 complete-field measurement on one low/high-M case to avoid equating the small-query panel with full-field inference. Measure two input forward-plus-VJP repetitions per representative condition as a feasibility/cost check, not a latency-distribution estimate.

Report complete inference, complete input-gradient cost, peak allocated memory, and complete epoch cost. If extra scope must be cut, retain complete-call measures before prepared-decode microbenchmarks. No dense-row counts or nominal K may substitute for seconds.

Target unchanged inference architecture and <=1.5x G-fast complete-call cost on the matched primary panels. This is a proposed engineering budget, not a previously demonstrated universal bound. Training may cost more because actual physical gradients and callbacks are now used; disclose the reason and measured amount.

### 7.3 Predicting physical changes

Use the four response-development families for the declared progress review, then run the frozen final checkpoints on the six current fixed4 primary transfers and secondary0277 span. Keep their exposed status visible. Record fit-family versus response-development versus fixed4 results in separate small tables.

Report only a few primary response quantities in the main text:

- Thermal response RMSE for fluid/surface/material T, against the zero-change control, each arm's parent, and retained Tensor-H.
- Correct or wrong direction for materially nonzero mean fluid-T changes; show0291 plus and minus explicitly.
- Changes at modules whose own heat is unchanged; report physical module identity rather than only a pooled statistic.
- Known-null u/v/p/omega and8%-band pressure leakage in native units and common training-standard-deviation units. Relative error against zero is undefined.

Count layouts and independent design directions separately from signs, channels, receiver points, and module peaks. One plus/minus heat direction does not identify the full M-1 fixed-sum heat Jacobian. Small-scale numerical warnings are not certified grid-error bounds. Do not create resolved-sign labels by choosing a threshold after observing model errors.

### 7.4 Does the organizer help these changes?

Perform one meaningful intervention on the final **joint** model: remove I and replay both endpoints of the same finite heat comparisons. Keep native fine sources, weights, source plans and ordinary phase updates correctly specified; plans remain fresh functions of each input. Compare error in the **predicted physical change**, not just the size of the output difference after removing I.

This supplements the independently trained separable-refinement comparison. It does not replace it. Co-adaptation can distribute knowledge into physical weights, so a large/small same-weight ablation is not a unique accounting of training benefit.

Use only one representative physical graph page: input-selected module/group, its actual M/E donor measures and receiver access, normal response, response with I removed, and measured reference response. Export signed control contrasts separately from nonnegative membership and all global/coarse/local/upstream paths. No all-case wall of latent-norm panels.

---

## 8. Predeclared success targets and honest negative outcomes

These are proposed research decision rules, not engineering certification thresholds.

**Accuracy target:** preserve each arm's main-field quality while improving response. At the final review, flag >5% degradation of a core mean or >10% degradation of its case-error p90 versus that arm's matched starting point. Such a miss cannot be hidden by better aggregate MSE. Also report the gap to stronger retained Dense/Tensor references rather than promoting on parent improvement alone.

**Response target:** improve thermal response error across response-development families, not just the four fit neighborhoods. A useful next result would match or beat retained Tensor-H's three macro thermal-response errors on the reused fixed4 primary panel, correct both substantial0291 mean-sign failures, and show better response on untouched-own-heat receivers. Do not call the target met from an improvement only over the weaker H-add parent.

**Null target:** seek at least90% reduction of the matched-parent heat-only flow leakage on the response-development and broad all 22 null panels at the same step sizes, with all four channels reported. A numerical-floor-limited result should show raw values and the declared floor. Reduction is not exact invariance, and a remaining consequential false pressure change still blocks trustworthy constraint use.

**Organizer target:** the refined joint model should earn value over the equally trained separable model in fields or physical responses, and its I-removal should expose meaningful response utility on more than a handpicked positive example. The result may legitimately be that both predictors improve but the joint organization offers no added benefit. Report that, do not force more groups or enlarge I.

**Efficiency target:** remain within the declared complete-call overhead and the total training budget, without new recursive planning or extra evaluation-only code in the hot path.

A failure after proper500 refinement is actionable: it says that this empirical co-adaptation/data scope is insufficient. Preserve it and identify whether the unresolved limit is positive-response coverage, response-family transfer, physical dependency leakage, or added organizer value. Do not automatically launch a new architecture, add more penalties, train inverse models, or restart formal jobs.

If heat-null leakage remains stubborn, the next separately reviewed architectural question is explicit case-owned heat/flow dependency separation. Do not hot-add that redesign in the present comparison, and do not present a hard-coded zero as learned physical knowledge. Similarly, a heat-affine response parameterization would require an explicit generator/assumption audit rather than assuming general multiphysics is linear in heat.

---

## 9. Limited inverse-use check: choices among already solved designs

This section is a **stored-candidate decision replay**, not an inverse model, search, generator, or new design validation campaign.

For0291/0294/0687, use exactly the stored baseline/minus/plus states. Define before prediction comparison a simple thermal decision: choose the candidate with the lowest maximum material temperature, with all input constraints respected. Use true per-module maxima on native material grids, not a smooth-max objective relabelled as a physical maximum.

Each model ranks the identical candidate pool from design/context inputs only. Reveal reference outcomes only for scoring. No model may choose a different pool, move the baseline, tune a bias correction on these outcomes, or obtain extra samples.

Report predicted best state, measured best state or unresolved tie, realized benchmark regret, and the reference ranking gap. If a gap is too small for the available numerical evidence, keep it unresolved rather than counting a forced correct/wrong choice. Choosing among three stored points does not establish continuous inverse optimization or generative generalization.

Physical flow/pressure is unchanged for these heat-only candidates. Pressure therefore supplies a null-consistency check, not a meaningful new feasibility boundary. Report any surrogate-created false pressure difference separately; do not claim a calibrated false-feasible rate from a pool containing no genuine pressure-changing design.

The0277 minus-to-plus pair can be shown separately, with no invented third state. No denoising or optimization trail exists in this round: plot actual candidate layouts/heat assignments and arrows between evaluated states, clearly labelled as a finite replay.

---

## 10. Software, provenance, and numerical checks

Implement small reusable additions, not one large workflow framework:

1. An opt-in refinement profile/attachment helper with explicit parent100 histories, trainable allowlist, fresh optimizer, new refinement age, fixed soft admission formulation, and distinct dataset-plus-response-addendum identity.
2. A response-refinement objective/callback using existing typed records, native wrappers, receiver samplers, correct accumulated gradient scaling, and train-only response scales.
3. An extended compact evaluation summary for field accuracy, measured runtime, response direction/error, and stored-pool decisions.

Preserve the historical `interface_fit` and `NativeCampaignResponse` guards. Do not weaken checkpoint trust, source/normalizer checks, or training/test provenance to make a new configuration load. The new experiment must have a new declared identity. Avoid constructing the current native wrapper or collecting full diagnostic tensors repeatedly when existing preparation can be reused safely within the same exact input/phase; never reuse state across different trial designs.

Tests must cover: old frozen and new refinement policy separation; exact allowed trainable/frozen inventories; role and family filtering; exclusion of fixed4 and response-development labels from gradients/scales; correct baseline/variant joins; normalization; accumulation; one real native forward/backward/update; parent replay; resumed step reproducibility; existing epoch-boundary stop; one-plan-per-wrapper phase sharing; and the absence of solver calls in evaluation/null augmentation.

Use current tolerances; report the inherited generic-Global/G-fast strict-VJP misses and high-M zero-correction interface miss separately from new results. Do not reinterpret all passing unit tests as numerical or physical certification. A modest FP32 discrepancy is not a reason to replace a scientific response experiment with an unbounded identity-repair project.

Codex may revise tactical implementation based on training-only evidence and measured cost. Document the reason, expected effect, exact affected components, and whether comparability changes. After outcomes have been used to change a scientific choice, create a new declared comparison rather than silently continuing under the old identity. No additional model portfolio is authorized.

---

## 11. Required report: readable front page, detailed appendix

The first page must answer, in plain language:

**Predictor:** Are the physical fields better, worse, or essentially unchanged, compared with our previous best useful models? Which important field still fails?

**Organizer:** Does the joint grouping improve an independently trained comparison and a real physical-response test, or is it only active inside the network?

**Inverse system:** Can the model select the better already-solved heat allocation, and what still prevents actual inverse generation or continuous optimization?

Use one compact dashboard. Do not alternate selected/endpoint, pooled/equal-case, normalized/native, soft/sparse or field/response definitions without a visible label. Put full 24-role, per-M, tail, lineage and numerical details in an appendix and machine-readable source tables. Their detail is valuable; it should not obscure the decision.

Include at most five necessary, visually inspected figure groups made from saved numerical evidence:

- Learning and measured complete cost, including parent and refinement ages.
- Reference/predicted fields and residuals on the fixed representative panel.
- Measured versus predicted finite thermal response, including the wrong-direction case and unchanged-own-heat receivers.
- One actual organizer/response-utility page with nonnegative memberships separated from signed controls and dense/global paths.
- Stored candidate heat layouts, rankings and regret, or an explicit unexecuted result if that replay failed.

Every figure needs a short quantitative caption and one sentence stating what it establishes. Native coordinates and units, checkpoint/stage, partition, physical-source limitations, clipping policy and missing results must be clear. Never fabricate a generative trail or a new solver result to fill the reporting template.

End with A/B/C explicitly:

A. Added value over the matched separable refinement—not just over an older frozen checkpoint.
B. Interpretation matches executed information paths, with no physical-sparsity or causal claim unless separately supported.
C. Response behavior on families withheld from the refinement gradients, with historical exposure and direction-rank limits disclosed.

Finish with one recommended next investment. Do not append an automatic 5,000-epoch launch recommendation merely because recipes can be parsed. Formal full-data training remains a separately requested, manually launched stage. Preserve formal3501/3502 histories and all scientific checkpoints.

---

## 12. Delivery and sources

Create a durable English report, the two refinement configurations and runnable preparation/resume commands, maintained source/tests, and a compact measured results file. Keep checkpoints, raw arrays, profiler captures, generated figures and one-time renderers in ignored local paths. Keep selected PDF figures with small raster companions for Markdown embedding. Do not delete scientific history or overwrite the previous report to make new findings look retroactive.

Audit the complete outgoing commit range, including files added and later deleted. Use the repository artifact hook; commit and push the non-default branch and verify the remote/local tips. A clean code push is a delivery check, not evidence that scientific goals succeeded.

### Source registry for this proposed plan

All measured baseline facts come from the following project sources; all new objectives, thresholds, data-addendum choices and training settings above are proposals.

- **R1:** `HONF_Proj/docs/reports/_bk/20261005_231703Z_HONF_Receiver_Interaction_Identification_and_Physical_Audit_Report.md`, especially Final learning/cost/selection; Predictor all 22; Organizer utility/cost; Authorized physical execution; Matched finite-response evidence. Report delivered at197a682.
- **R2:** `HONF_Proj/docs/reports/_bk/20261005_185023Z_HONF_Lean_Interaction_Reset_Development_Report.md`, matched G-fast/Tensor results, Tree cost diagnosis, and reference histories.
- **R3:** `HONF_Proj/docs/reports/_bk/20260926_064922Z_HONF_Interaction_Response_and_Inverse_Design_Study.md` and `HONF_Native_Recovery_and_Adaptive_Interaction_Study.md`, historical train/response families and reference-source limits. Availability and exact family metadata must be rechecked locally; these reports do not authorize new solves.
- **C1:** `HONF_Proj/src/honf_forward_core/interface_fields/tensor_query_interaction.py`, read at197a682: components, physical gain, query-only correction, freeze policy, exports.
- **C2:** `HONF_Proj/Case_ThermalChannel/src/channelthermal/training/interface_fit.py`, read at197a682: frozen-only campaign validation, parent attachment, optimizer and resume constraints.
- **C3:** `HONF_Proj/Case_ThermalChannel/src/channelthermal/training/campaign_response.py`, read at197a682: selected-anchor guard, family/variant callback, field-scale response loss and inherited coefficient calibration.
- **C4:** `HONF_Proj/Case_ThermalChannel/src/channelthermal/workflows/train_forward.py`, read at197a682: physical/fit ages, native callback and telemetry.
- **C5:** `HONF_Proj/src/honf_forward_core/interface_fields/dense_pairwise.py` and `typed_hypergraph_field.py`, read at197a682: named fine physical components, dense executor and ordinary task gradients.
- **G1:** `HONF_Proj/docs/guides/Thermal_Receiver_Interface_Fitting.md`, current frozen-fit workflow, not a refinement authorization.
- **G2:** `HONF_Proj/docs/guides/Thermal_Model_Development_Protocol.md` and `AGENTS.md`: development/formal separation and durable-delivery rules. Re-read current versions before implementation.

**Completion means:** actual bounded learning and a clear scientific verdict, with any unmet targets stated. It does not mean that HONF, inverse readiness, or formal promotion is automatically solved.
