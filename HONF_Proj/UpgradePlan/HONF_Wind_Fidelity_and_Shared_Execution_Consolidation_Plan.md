# HONF next phase: Wind fidelity recovery and shared execution consolidation

**Mode:** Codex Goal mode; bounded development. **Reviewed revision:** `agent/honf-core-next` at `f792dd6ed340dba6954af7ac58df5be8cbb8a729` (8 October 2026). **Primary source:** `HONF_Proj/docs/reports/HONF_Unified_Training_and_Adaptive_Interaction_Refinement_Report.md`, the updated completed-formal comparison, plus its archived development report. **Document status:** proposed work, not completed experiments. Repository paths below are relative to `HONF_Proj/`, except root `AGENTS.md`.

## 0. The decision: consolidate this family, do not reset HONF again

The goal remains **multi-field reconstruction → reusable organization of module–environment interactions → subsequent modular inverse design**. The shared source-preserving refinement family is now worth retaining. Thermal3903 is a useful forward research model with a real accuracy–cost tradeoff. Wind2201 is not yet an adequate replacement for its native classics. The shared optimizer engine and actual base/fine organizer already exist; do not spend this round inventing their replacements or claiming them as new deliverables.

The next question is: **can we recover Wind's missing component-wise and spatial fidelity, then retain useful adaptive detail at a reasonable complete cost, without breaking Thermal's successful affine interface or the common training workflow?** This is primarily a Wind learning experiment plus a common execution repair. Thermal is the preservation and transfer lane, not another unrestricted model search.

Keep the executed interaction law `B(q,i) + g(q,i) [F(q,i) − B(q,i)]`. Every physical source retains its base contribution and whole-scene context. The organizer decides where fine corrections are worthwhile; it does not have to delete sources, produce a prescribed K, or resemble a recursive tree. A successful research direction need not beat Dense1804 everywhere or prove a mature causal hypergraph. It must combine credible native fields, faithful adaptive information paths, and a measured accuracy–cost tradeoff that is worth further study.

Do not automatically launch or extend full-data runs in this goal. Existing formal3901/3902/3903 and Wind2201, all classics, all developmental parents, and the 100 historical protected bindings remain unchanged. No new solver attempts, inverse generator, design search, or new physical-response labels are authorized. The standing solver ledger remains **326/326**.

### Required outcome

Deliver an evidence-based Wind recipe decision supported by substantial matched development training, an improved or clearly diagnosed shared executor, and a concise physical report. Do not finish after startup tests and a list of possible fixes when the approved learning stages remain healthy and affordable. Conversely, do not hide an unsuccessful hypothesis behind a larger portfolio or call a resource-censored arm mature.

## 1. What the completed report actually establishes

These are source-derived observations, not expectations for the new runs. Use the updated model identities: Thermal1404/1501/1804/3902/3903 and Wind2102/2103/2201. Do not silently substitute older1401/1502 measurements.

| Item | Measured result | Development consequence |
|---|---|---|
| Thermal fields | 3903 fluid/surface/material RMSE is 0.321108/0.306690/0.301657. Dense1804 gives 0.237375/0.429475/0.329348. Sampled module-peak RMSE is 0.318111 versus Dense0.317573. | Preserve 3903's successful roles. Fluid-temperature fidelity remains a limitation, not a reason to discard the family. |
| Thermal heating responses | 3903 mean fluid/surface/material response RMSE is 0.007644/0.012036/0.014022 on three exposed layouts and six signed states, better than the compared endpoints. | Preserve precise heating semantics and this response evidence; these are not independently validated source columns. |
| Thermal inverse objective | On0291, true baseline/plus peaks are18.520241/18.129892, but3903 predicts19.058064/19.127707 and selects baseline. Realized regret is0.390348 native T. | Audit per-module peak values and responses. Average field/response RMSE cannot substitute for the maximum-temperature decision. Do not train on this validation ranking. |
| Wind fields | Vector RMSE volume/near-turbine is0.058991/0.158383m/s for2201, versus0.033638/0.095497 for2102 and0.019100/0.074455 for2103. Both classics win all15 component–role cells. | Recover the underlying Wind predictor before asking routing to solve this entire gap. |
| Training exposure | Wind2201 uses5000 epochs, Q1024, effective24 and8 environment records. Classics use2500 epochs, Q8192, effective16 and512 records. | Equal or greater epoch count does not establish equal spatial supervision, capacity, or training conditions. |
| Current execution | Wind adaptive/all-fine/all-base complete calls are23.83/12.15/24.02ms on the fixed timing panel. Thermal3903 is39.68ms versus3902 25.79 andDense1804 158.34ms. | Source-preserving models are relatively inexpensive; current adaptive execution is not faster than its own all-fine path. Separate these two facts. |
| Actual organizer | Learned equal-degree routing beats several geometry controls, and remove/restore operations change the real outputs and restore them exactly. | Keep the organizer direction. Extend utility measurements beyond attractive support maps. |
| Memory | Wind adaptive additional inference allocation is about9.11MiB versus7.91MiB all-fine. Thermal's measured allocation tradeoff is different. | Low absolute allocation is not proof of an adaptive memory saving, high GPU utilization, or an accurate model. Measure training peaks separately. |

The nominal completed Wind training horizons imply approximately2.1504billion primary queries for2201 and8.6016billion for a2500-epoch classic, before auxiliary work and replay. These are calculated queried rows, not unique cells or independent physical observations. The classics received roughly four times as many primary query evaluations despite half as many epochs. The report also records5700-plus seconds concentrated in three Wind epochs; catalogue preparation is a plausible cause, not a measured attribution.

Thermal's canonical89 is repeatedly exposed validation, not an independent test. Wind VALID90 is30 correlated three-direction layouts, and TEST90 remains locked. The current phase uses the smaller development populations below. Reuse the formal report for mature context instead of recomputing the entire formal comparison at every review.

## 2. Code findings and hypotheses to distinguish

| Source and function | Confirmed implementation fact | What must not be assumed | Next action |
|---|---|---|---|
| `Case_WindFarm/src/windfarm/training/unified_task.py`, `predict_native`, `loss_terms` | Predictions are transformed to physical velocity; one scalar scale per role divides squared errors pooled over all three components. Output normalization is component-wise, but the native loss is not component-balanced. | This alone has not been experimentally proved to cause the transverse error. | Measure component losses/gradients, then perform one controlled component-balanced objective comparison. |
| Same provider and `training/unified_formal.py` | Q1024, H64/message64 and E8 are sealed recipe choices; formal role scales are freshly TRAIN-fitted while the previous development role scales were inherited constants. | These settings are universal architectural requirements, or eight environment records alone explain the error. | Introduce versioned new recipes with common scale-fitting rules, configurable Q/capacity/context, while preserving old profiles. |
| `src/honf_runtime/unified_training.py`, `SamplingKey.seed_for` | Query seeds include update and microbatch indices. | Microbatch changes preserve identical future samples. | Add a versioned case/epoch/role/draw sampling stream independent of packing. |
| `src/honf_forward_core/interface_fields/interaction_refinement.py`, both refined readouts | Adaptive inference repeatedly calls `nonzero`, gathers selected features, evaluates fine MLPs and scatters into a base tensor inside receiver chunks. All-fine takes a separate vectorized fast path. | Fewer fine rows guarantee less complete time. | Profile and repair dispatch/packing; compare execution backends for the same mathematical adaptive output. |
| `SourceRefinement.read` and `auxiliary` | Base and gate computations touch all pairs; base loss sees a detached fine target but live base/context inputs. | Base fitting cannot affect the shared representation merely because its target is detached. | Measure auxiliary/native gradient shares; retain the objective unless a separately documented causal probe justifies a change. |
| `_combine` and `SourceRefinement.route` | Hard gates switch between F and B; protected near support is a Boolean test of a smooth near weight. | Fixed-route AD checks establish continuity at route/protection boundaries. | Compare native artifacts across execution modes and inspect bounded route transitions. A continuity remedy is conditional, not presumed necessary. |
| `interaction_core.py`, `_read_features`, `_geometry`, `NonlinearFieldReadout` | Local/domain-relative Fourier geometry already exists; source-wise nonlinear reads precede a normalized source reduction and a small nonlinear head. | Local geometry is absent, or Fourier features/late reduction are proved faulty by a striped picture. | Preserve those features initially. Use small-fit and artifact tests to locate a representation limitation before changing it. |
| `Case_WindFarm/src/windfarm/shared_interaction.py` | The current adapter already binds velocity normalization and background-profile ownership. The public physical derivative path is implemented. | These previously repaired issues remain open. | Reuse their regressions; do not rebuild this interface again. |

The most actionable working hypotheses are unequal component learning pressure, insufficient or poorly distributed spatial supervision per update, insufficient representation for the Wind residual, and inefficient adaptive execution. They are not interchangeable explanations. Test them in the order below rather than changing all of them simultaneously.

## 3. Scope, data, and immutable boundaries

Use **Thermal fixed25_v1:150 TRAIN cases/22 exposed DEV cases**, including its existing auxiliary response-family declarations and selected-TRAIN normalization. Preserve all available Thermal training module-count categories. Detailed Thermal exports remain0277/0291/0294/0687, with a separately identified formal0291 read-only diagnostic permitted below.

Use **Wind wind_shared_fixed24_v1:24 TRAIN layouts/72 direction rows and8 DEV layouts/24 direction rows**. Preserve all three directions per selected layout. The existing24-layout selection omits TRAIN M18; retain and disclose this limitation. Do not silently add another layout or claim that the Wind subset covers every category. TEST90 targets remain unread. Detailed Wind planes remain the input-selected low/high-M DEV rows69/426, plus a bounded TRAIN panel for diagnostics.

Freeze IDs, normalizer scope, baseline-profile fitting, query policies, initialization seeds and review rules before each comparison. The same dataset adapter and target source must serve every arm in that comparison. New loss scales are fitted only on the frozen TRAIN population. Different recipes may intentionally change Q or capacity, but must report those changes and their real work.

Formal3903/2201 may be read to diagnose their existing failures and benchmark semantics-preserving execution. They must not initialize a fresh quarter-data experiment or supply hidden teacher targets for it. Use fresh Wind fine weights for the new controlled screen. For any conditional Thermal child fit, use the exact development lineage and its development flow checkpoint, not formal3901/3903.

No case-ID lookup predictor, native-target velocity in environment features, analytic wake replacement at inference, output smoothing that disguises reconstruction errors, or supervised gate selected from validation error is allowed. Existing TRAIN-fitted empirical background profiles remain explicitly empirical, not physical inlet boundary conditions.

## 4. Stage0 — bounded diagnosis before training

Spend at most about60minutes on the initial diagnosis, excluding unavoidable documented native data initialization. Its purpose is to choose and enable training, not generate a second encyclopedic audit.

### 4.1 Reproduce the relevant failure, not every historical metric

Read-only Wind2201 replay should compare normal adaptive, same-weight all-fine, and all-base on identical native receivers for the fixed low/high-M scenes. Include Ux, Uy and Uz separately. Add one soft-route or fixed-route read only where it helps locate visible seams. Use the same physical transforms, source order, masks and native coordinates. Confirm scalar metrics are recomputed from the exact arrays used to render the maps.

Verify frame/axis conventions, component scaling, background addition, source/query ordering, inference chunk joins, and target-free inputs. The report already reproduced the native Q1024 selector exactly; do not assume a visualization bug merely because the fields look unusual. Compare raw queried values with rendering inputs before attributing a visual pattern to learning.

Record whether strange bands, rings or spots persist under all-fine. If they do, routing is not their sole cause. If they disappear or move under route interventions, quantify that contribution. A difference in one scene is not an explanation of the entire population error gap.

### 4.2 Component-learning diagnosis

On no more than three fixed TRAIN minibatch panels spanning low/high M, record physical residual RMS and native-loss contribution separately for each role and component. Record gradient norms for Ux/Uy/Uz task terms on the last output layer and one shared readout block, together with base-loss and router-loss gradient shares. Use a few aggregated norms/cosines rather than exporting per-parameter gradients.

Check for near-zero or disproportionately small transverse update signals and for shared-capacity competition. Test with all-fine and no routing cost to distinguish a host-field problem from selection. This measurement informs the loss experiment; do not rescale losses using DEV outcomes.

### 4.3 Training-data and spatial coverage diagnosis

Measure native-cell uniqueness/repetition and role/source-neighborhood coverage on a short real TRAIN stream. Q1024 already includes all five roles; describe whether the issue is sampling density or variance, not an invented absence of wake samples. Retain duplicate cross-role samples under their declared weights, and do not count repeated cells as new physics.

One small TRAIN-only fitting test, capped at400 disposable updates on a fixed small layout panel, is permitted when needed to distinguish a failing fitting path from generalization. Record it separately and do not use its weights to initialize the scientific arms. Do not tune a model on the two displayed DEV planes.

### 4.4 Thermal peak-objective localization

Reuse the already saved0291 three-state inputs/references. At unchanged3903 weights compare normal adaptive and all-fine predictions of every physical module's sampled maximum, the module/cell setting the global maximum, and baseline-to-plus changes. Separate baseline bias, incorrect per-module slope, a change of the maximizing module/cell, and route-induced error. Include3902 as a retained reference.

This is diagnosis only. No extra physical states, no fit to0291 ranking, no clipping, and no new peak penalty selected from this validation case. If all-fine repairs the ranking, report a route-specific miss; if both fail, do not claim a routing repair will fix it. The failure remains in the report even if mean response accuracy improves elsewhere.

### 4.5 Existing test failures

Identify the37 broader-suite failures mentioned in the development closeout and compare with the reviewed parent revision/environment. Classify newly introduced failures, resource-dependent failures, legacy numerical limitations and unrelated pre-existing failures. Fix failures introduced in the touched training/reader/ownership paths. Do not require an unrelated repository-wide numerical cleanup before the bounded scientific work, and do not describe149 focused passes as a completely green repository.

## 5. Stage1 — make extra GPU capacity scientifically useful

### 5.1 Preserve the real shared engine

Extend `TrainingEngine` and the two providers through versioned configuration; do not create new dataset-specific optimizer loops. The engine must still own visitation, accumulation, clipping, optimizer/scheduler clocks, monitoring, checkpoint/resume and accounting. Adapters own data meaning, target transforms, native losses and affine/nonlinear capabilities.

Keep existing sealed recipes loadable and their original seeds unchanged. A new profile version may expose query count, query tile size, model width, environment token shape, loss-scale rule and execution backend. Store resolved values in checkpoints and reject a silent resume under another recipe. Do not edit a loaded running trainer in place.

### 5.2 Sampling independent of microbatch packing

Introduce a new stream keyed by `(dataset, seed, epoch, physical case/direction ID, role, stream, draw index)`. Arm identity and microbatch/update position must not determine a case's query coordinates. Query sequences should be prefix-consistent when a role requests more draws, allowing a Q1024 panel to be a declared subset of a larger panel where feasible.

Tests must show identical sampled native IDs for the same cases/epochs under different microbatch packing and consistent exact loss denominators. Compare gradients and one optimizer update on those identical examples at the existing FP32 tolerance; do not demand universal bitwise equality across different reduction orders. Preserve the old stream as an explicit legacy version.

Changing the effective optimizer batch remains a scientific change. Initially keep Wind effective24 and Thermal effective48. Increase physical microbatch size and/or query tiles for throughput without silently changing those effective batches or reducing optimizer updates per epoch.

### 5.3 Measure throughput and memory, not memory occupancy alone

Perform a small restored-state TRAIN benchmark with native high-M and mixed-M batches, full scheduled losses and backward. Include the all-fine replay/base/router work used during training. Prices of inference alone cannot size the training job.

Use a staged, non-Cartesian benchmark: first compare feasible microbatches8/16/24 for Wind at the chosen Q and fixed effective24; then compare receiver tiles512/2048/8192 for the selected microbatch. Include the incumbent microbatch when different. A maximum of eight configuration points is sufficient; stop configurations that run out of memory without repeated retries. Use actual tail-batch handling.

Record case visits/s, native query rows/s, optimizer updates/s, complete epoch time, peak allocated/reserved memory and available-device headroom. Report asynchronous GPU work and host stalls honestly. The objective is improved throughput and useful learning, not reaching a chosen VRAM percentage. Keep a safety reserve of at least4GiB and roughly20% of the memory currently available to the authorized process; larger reserves may be needed under contention. Never evict or terminate another user's process.

Use spare memory first for larger per-case query panels and efficient execution tiles, then for justified representation capacity. A larger microbatch alone does not increase the network's representational capacity. Increasing Q changes gradient sampling/work; increasing hidden width changes the hypothesis class; increasing effective batch changes optimization. Keep these comparisons distinct.

FP32 remains the scientific reference. Mixed precision, compilation or CUDA graph capture may be tested only as bounded engineering options supported by the installed stack. No environment upgrade is implied. Preserve FP64 precise Thermal increment accumulation and gate/support/numerical checks; do not silently convert a failed precision test into a pass.

### 5.4 Avoid repeated cold catalogue construction

Profile target-free geometry/role catalogue formation separately from native target gathers. Reuse already bound catalogues across training stages and process restarts where the existing storage rules allow; use a bounded versioned disk/memory cache rather than repeatedly reconstructing the entire panel. Do not place all native target volumes in GPU memory merely to fill it. Cache ownership must include geometry/frame/role definitions, not validation outcomes.

Charge cold formation, preparation, replay after failures, and warm training separately. A catalogue-loaded epoch must not be called a cold epoch, and a multi-minute initialization spike must not be multiplied by5000 to forecast normal training.

## 6. Stage2 — one controlled Wind learning ladder

Use fresh, matched initialization within each applicable architecture. All screen arms are all-fine through epoch500, with base approximation trained under the declared common objective and no cost pressure on the field. This isolates host fidelity before introducing adaptive deployment. Use identical case visitation, effective batch, corresponding LR schedule and new sampling-stream version. Changes in query work and capacity are explicit interventions, not equal-compute claims.

### 6.1 The new component-balanced objective

The current objective is, schematically, an average of `||u_pred − u_target||² / s_role²` over three physical components. Introduce one alternative with one TRAIN-derived scale for each role/component:

\[
L_{\mathrm{component}}=\frac{1}{5}\sum_{r=1}^{5}\frac{1}{3}\sum_{c\in\{x,y,z\}}\frac{1}{N_r}\sum_{q\in r}\left[\frac{\widehat u_c(q)-u_c(q)}{s_{rc}}\right]^2.
\]

Fit `s_rc` from RMS residuals around the same fixed TRAIN empirical background on a deterministic TRAIN-only native-role calibration panel. Include means rather than silently discarding a systematic component bias. Set a predeclared floor `s_rc = max(raw_residual_RMS_rc, existing_physical_component_floor_c, 0.10 * scalar_role_scale_r)`. This bounds the maximum relative emphasis on a nearly null channel; the0.10 floor is a prospective engineering choice, not a measured physical noise floor. Store scales in physicalm/s and preserve exact formulas in the recipe.

Use this rule in both new development and future formal providers, fitting separate values only from their respective TRAIN memberships. For W0, retain the current scalar-per-role functional form but derive its scalar role scales from the same frozen TRAIN calibration panel and the formal recipe's documented scalar rule; W1 changes only component weighting relative to that shared calibration. W0 is a fresh current-family subset control, not an exact reproduction of full-data2201 or of the older development recipe with inherited role constants. Do not retain an accidental development/formal difference in how weights are defined. If source inspection establishes a different mandatory native floor, document the one amendment before scientific updates. Do not sweep the floor or component weights on DEV.

Keep reported physical component/role RMSE, residual maps and the old common evaluation score separately. A new training loss need not decrease numerically relative to the old loss to be better. Do not call a lower newly rescaled scalar objective an accuracy gain.

The background remains the same empirical TRAIN profile across matched arms. This experiment does not simultaneously change the profile, impose a new wake formula, train an inverse head or inject measured flow.

### 6.2 Screen matrix and ceilings

| Arm | Objective | Native queries per case | Representation | Purpose / maximum screen age |
|---|---|---:|---|---|
| W0 | Current scalar-per-role objective |1024|H64/message64, E8|Fresh current-family control;100 review, then500 |
| W1 | Component-balanced objective above |1024|Same asW0|Isolate component weighting;100 review, then500 |
| W2 | Same asW1 |4096|Same asW1|Isolate denser per-case supervision;100 review, then500 |
| W3, conditional only | Selected W1/W2 objective and Q |Unchanged from its parent recipe|One representation change only|At most one500-epoch follow-up, selected by the rule below |

Role counts forQ4096 must be deterministic, sum exactly4096 and preserve the five-role mixture, with receiver weights/denominators explicit. Query IDs use the prefix-consistent new sampler. Keep originalQ1024 as the low-work control. Q8192 is available for measurement and an explicit later manual recipe; do not automatically add a fifth Q8192 scientific arm.

At100 inspect native component errors, finite gradients and representative wake structure. Extend healthy fitting to500 even if it has not approached the mature classics. Do not make an underfit100-epoch candidate pass a formal fidelity gate before it can learn.

W3 is permitted only if the first ladder leaves a concrete representation question. Choose **one**, not both: (a) H128/message128 with E8 unchanged when TRAIN fit/capacity or shared-readout evidence motivates it; or (b) E64, shape8×4×2, with H64/message64 unchanged when inadequate spatial context is the better-supported hypothesis. Choice, parameter count, projected cost and reason must be recorded before its first update. Use prescribed geometry-only environment records. A different quadrature grid is a changed representation, not an exact environment-atom splitting invariance test.

Do not assume E512 is needed because the classics used it. Do not introduce a dense receiver×environment fine stack or a new global latent bank in W3. Likewise, do not increase width merely to consume spare memory. If the diagnosis identifies a real correctness defect rather than capacity, repair it first and use the W3 allowance only for the single justified repaired recipe.

The ladder identifies recipe components locally. Comparisons at equal epoch have unequal query work where Q changes; show accuracy versus optimizer updates, queried rows and measured time. Q4096 uses four times the per-case primary query work; do not label that arm equally cheap or infer that it has four times as many independent data cases.

## 7. Stage3 — mature the useful recipe and its organizer

After the500 review choose one host recipe using predeclared component/role evidence: prioritize clear recovery of Uy/Uz and wake structure while protecting Ux/background behavior and the measured cost envelope. Avoid choosing from the aggregate score alone. Preserve every screen checkpoint and outcome. Do not pick an unmeasured mixture of favorable settings from different arms.

Continue the W0 control and the selected full-detail host to **2500 total development epochs**, with100-epoch reviews. Branch one Adaptive-detail child from the selected host's exact500 checkpoint, including fine/base weights, moments, data identity and schedule. If W0 itself is selected, do not duplicate its full-detail continuation. The main final comparison is therefore at most **three mature Wind lineages**: current-recipe control, selected Full-detail, and its matched Adaptive-detail child.

The Full/Adaptive pair must use identical corresponding initialization, sampling, case/update budgets, native losses and fine/base LR schedules. Existing500 warmup,600 open,800 soft and801+ hard stages can be retained under a new explicit recipe. Both receive the same all-fine supervision/replay accounting. Recalibrate the declared TRAIN-only work coefficient using the new native objective at the intended stage; record its actual gradient share and cap. Do not tune it for a prescribed selected-source count.

Use the same absolute2500-epoch schedule rather than restarting the LR schedule at each review. Every100 milestone remains a review, not a termination command. Continue healthy non-divergent candidates through meaningful maturity. At minimum reach a matched1000 review;2500 is the intended endpoint, subject to the explicit resource ceiling and sustained adverse evidence. If resources bind, stop the final comparison at a common completed milestone rather than fully mature only the favorable arm.

An example sustained stop reason is worsening protected component means and tails across three saved reviews, accompanied by no relevant field improvement and no plausible unfinished stage. A small oscillation, one noisy metric or absence of a universal Dense win is not a stop reason. Do not add a new architecture after the selected mature branch disappoints. Finish the diagnosis and report what the bounded experiment learned.

A longer development fit is not a full-data result. Conversely, a model failing a few inverse checks does not automatically invalidate an otherwise useful forward research recipe. Keep those decisions distinct.

## 8. Shared execution repair: preserve the adaptive function

### 8.1 Distinguish execution backend from mathematical route

Implement a backend comparison for the **same selected adaptive function**, separate from `all_fine`/`all_base` mathematical interventions:

- Dense-masked backend: compute the required base/fine values in vectorized blocks and combine them with the same learned gate/support.
- Selected backend: evaluate only the selected fine pairs, retaining all required base contributions and protected near reads, with exactly the same mixed output.
- All-fine mathematical control: bypass the base/router and evaluateF everywhere. This can be faster, but generally predicts a different function and is not a semantics-preserving execution fallback.

For small M or high retained fractions, the dense-masked backend may be preferable. That is a legitimate implementation choice with dense arithmetic honestly counted; it does not erase the organizer's mathematical effect. A dispatcher may use TRAIN-benchmarked tensor shapes, support sizes and available memory, never target errors, to choose a backend. Always retain explicit backend overrides for reproducibility.

### 8.2 Remove avoidable work in the current hot path

Profile a bounded low/high-M native call first. Then prioritize larger batched MLP calls, fewer query-chunk boundaries, one request/tile-level support compaction where feasible, reusable source/global projections and geometry features, and reduced gather/scatter/temporary-tensor overhead. Avoid repeated device scalar extraction for diagnostics. Public ordinary calls should not retain full pair arrays unless requested.

Investigate the repeated CUDA `nonzero` calls and index construction as synchronization candidates. PyTorch documents that CUDA `torch.nonzero` synchronizes host and device. This source fact motivates measurement; it does not prove that it accounts for all reported latency. Use profiler traces to distinguish kernel time, host dispatch, synchronization, base/router computation and staging.

The execution tile need not equal the number of statistically sampled receivers. Wire configurable tiles through the public Wind batch adapter, not only through an internal evaluation call. Larger tiles are allowed if actual training/inference memory and parity permit. Optional `torch.compile` must use the installed version and keep an eager reference; count cold compilation and graph breaks separately. Do not mask an unsupported compiler or precision failure.

Validate unchanged adaptive outputs, actual masks, source IDs, cold/prepared reads and input derivatives at existing tolerances before citing a speed gain. Source and query permutation tests, partial-batch/padding tests, environment-measure splitting and stale-state rejection remain. Do not weaken checks to obtain a benchmark result.

### 8.3 Benchmark useful complete scopes

Time native complete calls, prepared repeated reads, and input VJPs separately. Include flow composition and native surface/material/port extraction for Thermal where applicable; include Wind scene/context, physical output conversion and CPU transfer. Checkpoint/HDF5 opening, geometry catalogue creation and optional compilation are separate cold costs, not silently omitted from end-to-end totals.

For repeat measurements use at least one warmup and eight alternating repetitions on the fixed low/high-M panel when affordable. Save individual repetitions and report medians plus spread. Different datasets/devices are not one hardware ranking. Include all-fine, dense-masked adaptive and selected adaptive at identical weights/queries, and record all active, padded, base, gate, protected and actually evaluated fine rows.

Use Q1024, Q8192 and one existing full native Wind plane for the selected pair; avoid a large M×Q×chunk×precision Cartesian grid. Training benchmarks must include the scheduled objectives and an actual optimizer boundary. Record peak allocated and reserved memory, process residency and CPU cache separately. Lower inference allocation must not stand in for training memory, and neither is GPU compute utilization.

## 9. Conditional continuity repair, not an automatic new router campaign

A hard gate can change the interaction byF−B when its threshold flips. A Boolean protection boundary can do the same even when its underlying geometric near weight is smooth. Fixed-route derivatives do not test either transition. Use input-only bounded line sweeps on at most two TRAIN scenes to determine whether these transitions produce material jumps in the current fields; report model-only continuity evidence separately from stored native physical errors.

Only if this diagnosis identifies a consequential seam, one **compact-support continuous refinement weight** is permitted in the selected Adaptive branch. A candidate parameterization is

\[
t=\operatorname{clip}\!\left(\frac{\ell-\ell_0}{\ell_1-\ell_0},0,1\right),\qquad g_{\mathrm{optional}}=t^2(3-2t),\qquad g=1-(1-p)(1-g_{\mathrm{optional}}),
\]

where `ell` is a cheap learned score and `p` is a dataset-scaled continuous near-protection weight. At deployment, fine corrections are evaluated only whereg>0; the correction vanishes with its derivative at an optional-support boundary. Keep the inner near region fully fine and evaluate fine values throughout any geometric transition support. Thermal's original near-head contribution and affine heating independence must remain intact.

This is a new mathematical route, not a parity optimization of old hard checkpoints. It changes the transition annulus semantics and must have its own version, tests and trained identity. Seal one transition choice from TRAIN diagnostics before the final Adaptive branch; do not sweep thresholds on DEV. Continuous gate width and selected-pair counts are not physical interaction probabilities.

Training must retain a route-recovery signal for currently closed corrections. The already paid full fine replay and an explicitly isolated organizer surrogate can supply it. Physical input derivatives in the public inference API must differentiate the actual deployed function, including a live continuous gate where used; they must not silently return the training surrogate. For legacy hard gates, retain the declared fixed-route derivative semantics.

If no consequential route seam is established, keep the existing gate mathematics and defer this option. Either outcome is acceptable. No second large router portfolio is allowed.

## 10. Thermal preservation and optional transfer test

Thermal3903 is not to be replaced just to make every new Wind option run identically. Shared software and training protocols permit dataset-specific feature widths, context resolution and output laws. The generic core must not acquire hardcoded rotor units, Wind components, native grid sizes or Thermal heat assumptions.

For pure execution/configuration changes, verify unchanged3903/3902 native fields, source-resolved responses, precise heating increments, heat-null flow, units and ownership using retained weights. Complete one disposable real Thermal effective48 optimizer boundary and exact checkpoint/resume check under the new packing-independent sampler. These tests validate the common workflow without claiming a new Thermal accuracy result.

If the conditional gate or another shared mathematical change is selected, perform one bounded matched Thermal development child comparison: legacy mathematics versus the new mathematics from the same preserved development parent, fixed25_v1, identical objectives and new sampling stream. Review100 and continue to500 when healthy. This is the only extra Thermal scientific pair allowed, and is not initialized from formal3903. The objective is transfer compatibility and preservation, not a new Thermal5000 campaign.

Measure fluid/surface/material fields, module-peak tails and the already defined response-development families. Keep counted0291/0294/0687 and saved decision pools as disclosed final diagnostics, not tuning targets. Do not introduce a heat-dependent gate, reintroduce heat-to-flow leakage, or switch on the previously ineffective predicted-flow injection. Frozen D-sep vorticity weakness remains a separate unchanged result.

## 11. Organizer usefulness: A, B, and C with fewer but clearer tests

### A — Does learning choose useful fine corrections?

On the full fixed development panel, compare normal adaptive and same-weight all-fine native component/role errors. On a small fixed physical panel, compare learned routing with nearest/upstream/shuffle at equal fine degree and identical near protection. Report physical error as well as distortion relative to the host model.

For continuous gates, equal-work controls must match evaluated positive-support counts; also preserve the multiset of nonzero gate weights when testing assignment alone. Otherwise identify the comparison as a change of both assignment and weighting. No extra latent-group capacity is introduced.

Show whether decisions vary with receiver and with a real stored change of operating direction, using matched source identities and correct coordinate/vector transforms. Grouping receiver points with similar chosen source sets is a valid packet representation, but group counts, physical source counts and algebraic response rank are different quantities. Do not reward varying K merely to make a figure interesting.

### B — Does the explanation match executed information flow?

Export physical sources, environmental context ancestry, cheap base paths and actual fine upgrades. Preserve all base paths even where no fine upgrade is selected. On one input-selected example per dataset, remove one optional correction, recompute the real output, and restore it. Thermal may show the exact signed coefficient correction times heating; Wind must show the nonlinear output change, not relabel a latent-message norm as m/s.

Report whether the removed correction helps or harms reference fidelity where a stored reference exists. An active harmful correction remains evidence of execution, not evidence of good organizing. Repeatability and exact restoration supportB, not physical causality.

### C — Does response knowledge transfer?

Retain the existing saved Thermal finite-response/geometry limits and their exposure history. Wind's nominal three-direction field comparisons are not independently solved geometry/yaw/thrust perturbation tests. No newC truth is created by more queries, derivatives or epochs. Local JVP/VJP and numerical transition checks qualify the interface only.

Do not block all forward progress until broad inverse validity is established. Conversely, do not advertise improved forward RMSE as validated inverse design. The current stage selects a direction and a forward recipe.

## 12. Review decisions and research targets

The following numbers are **prospective research priorities**, not already attained results or physical certification thresholds. They guide choices without requiring a universal classic-model win.

| Review axis | Desired evidence | Failure interpretation |
|---|---|---|
| Wind recovery | At mature matched ages, aim for at least25% lower near/downstream Uy andUz error than the new current-recipe control, with Ux/background within about5% in mean and10% in tail, or a clearly justified superior combined tradeoff. | If one channel stays essentially unlearned, do not hide it in vector-relativeL2 or a favorable aggregate. |
| Host sufficiency | Better native wake shapes and cross-sections, not only smaller training loss; a credible cost relative to existing inexpensive and classic models. | A fine host that still cannot reconstruct the required field needs a different diagnosed representation investment, not more routing pressure. |
| Adaptive fidelity | Aim for protected-role means within5% and tails within10% of its matched Full-detail host while delivering useful selection. Exact equality or occasional adaptive gains are permissible. | Attribute host and route losses separately. More aggressive sparsity is not a rescue for an inaccurate host. |
| Efficiency | Improve current adaptive overhead for the same function; show a useful measured error–time/memory frontier. Prefer similar or lower complete cost than native K6 when fidelity becomes comparable. | A small model can legitimately use more memory/time to gain fidelity. No mandatory speedup versus all-fine is required to retain a promising organizer, but the extra cost must be explicit. |
| Interpretation | Actual receiver/scene adaptation, correct source identity, and measured output effects; no forced K or invented environmental sparsity. | Spatially variable masks alone are insufficient. |
| Thermal protection | Preserve successful native fields, affine response law and heat independence; retain peak-objective failure unless measured repair occurs without fitting that validation outcome. | Do not trade away Thermal's successful path for superficial code uniformity. |

Use three separate closing decisions: **continue this research direction**, **prepare a manual formal forward recipe**, and **qualify inverse application**. These need not have the same answer. A promising forward recipe with unresolved inverse physics may deserve a manual formal comparison. A passed source-identity test alone does not.

At every100 review show a small stable scorecard: Ux/Uy/Uz physical errors for Wind volume/near/downstream, three Thermal temperatures if newly trained, selected-detail fraction, complete runtime and current training age/work. Detailed native exports occur only at100,500,1000 and final/selected endpoints where they answer a question; reuse arrays when states coincide. Do not add dozens of nearly duplicate qualification panels.

## 13. Time, GPU, and failure budget

Proposed ceiling: **16 aggregate GPU-associated hours and10 elapsed hours**, including both authorized devices, all failed starts, disposable probes, native evaluations and training. Reserve the final hour for interpretation, report/figure validation and Git closeout. The ceiling is not a target to exhaust. Reforecast from actual complete epochs at100/500 and before final maturation.

Use only the GPUs currently authorized by the user/repository session. The historical default is GPUs1/2, but a later explicit allocation may supersede it. Verify UUID/logical mapping; do not infer authorization forGPU0 from old report figures. Leave unrelated processes untouched. If contention persists, proceed on a usable authorized device or continue CPU/cache/test work, document contention and measured throughput, and reforecast. Do not wait indefinitely for a perfectly empty GPU or overcommit memory to prove occupancy.

Prioritize, in order: Wind correctness/objective experiment; substantial selected-host and Full/Adaptive maturation; semantics-preserving executor repair; necessary Thermal preservation/conditional transfer. If needed, omit conditionalW3 or optional continuity work before censoring all main fits. Do not claim completed2500 epochs when the actual budget only reached a shorter common endpoint.

Record complete successful epochs, optimizer updates, case/direction visits, sampled and unique native query counts where measured, fine/base/gate work, cold setup, warm loop time, validation, saves, failed/replayed work and outer process envelopes. Avoid multiplying estimated time by5000 and presenting it as a measured total. De-duplicate shared warmup accounting and do not count shared lineages twice in campaign totals.

Keep one best alias, latest state and declared milestones at the100-epoch cadence. Best selection is limited to saved checkpoints and its rule is identical within matched arms. Preserve terminal and selected ages separately. Use the existing clean-stop/strict-resume machinery. Every change of objective, Q, architecture, normalization or sampler version creates a declared new recipe/branch, not a silent resume amendment.

## 14. Deliverables: make the science easy to see

### 14.1 Durable implementation

Update the existing common engine and providers rather than introduce a replacement framework. Deliver versioned recipe configuration, component-balanced Wind loss/scales, packing-independent sampling, supported execution backends/configurable tiles, required cache fixes and focused regressions. Preserve legacy loading and actual mathematical laws. Any conditional representation/gate change must have its own explicit configuration and model identity.

Tests must cover native scalar/units, target poisoning, source and receiver identity/permutation, partial/padded minibatches, effective-batch denominator equivalence, stream-prefix and packing invariance, precise Thermal increments/nulls, nonlinear Wind output transforms, input derivatives, gate transitions where changed, backend parity and stale-state rejection. Compare the affected broader failures with the parent; never overwrite their historical evidence.

### 14.2 Main report and figures

Open with approximately one page answering: **Is Wind visibly more accurate? What did the additional GPU resources purchase? Does adaptive organization still help? What costs remain? What can be reused in later inverse work?** Separate new measured outcomes from hypotheses that were not exercised.

Limit the main presentation to about six to eight figure families, with a detailed numerical appendix:

1. A compact physical scorecard plus learning curves at matched mature ages, including real work/time rather than onlyepochs.
2. Low/high-M Wind Ux/Uy/Uz native fields: reference,2201, selected new Full and Adaptive, with common per-component scales and separately signed residuals. Use readable pages, not dozens of tiny panels on one sheet.
3. Native wake cross-sections at input-defined downstream locations and heights, showing deficit amplitude, width and transverse structure. Identify exact native cells or explicitly labelled interpolation; do not manufacture solved continuous profiles.
4. An artifact-localization board: all-fine versus adaptive and any measured route boundary. State whether the strange structures belong to the host, routing or both; do not infer causation solely from visual similarity.
5. A receiver-centered interaction board with all base paths, selected fine upgrades and actual output effects, plus one stored direction-change illustration.
6. Fidelity versus complete runtime/memory, including dense-masked and packed execution of the same adaptive function, and a small training-resource panel distinguishing Q, microbatch, capacity and cold setup.
7. A compact Thermal preservation/peak-diagnostic board; retain0291's failed decision when not repaired.

All population statistics use the fixed development panel. A larger final validation is a separate authorized stage, not repeated model development on the full source TEST/VALID pool. Report physical units, masks, layout correlations, selected/terminal checkpoints and clipping. Do not use a small relative error dominated by meanUx to hide poorUy/Uz wakes. Keep numeric arrays and failed outcomes in the appendix.

### 14.3 Formal handoff

Prepare a tested **manual-only**5000-epoch full-data recipe for at most one selected Wind Full/Adaptive pair, with clear alternative deployment backends. The user chooses whether to run one or both. Fit its normalizers on fullTRAIN only, with the same new loss-scale procedure and dataset-owned semantics; do not import subset scales. Provide prepare/dry-run/start/status/clean-stop/resume commands, measured development/native startup forecasts, GPU mapping and checkpoint selection rules. No actual formal optimizer or full-data training starts in this goal.

A pure runtime repair of Thermal3903 may be deployable without retraining if unchanged-function qualification passes. Do not manufacture a new Thermal5000 run merely to attach a new name. A changed mathematical gate requires a separate manual recipe and evidence; report the distinction.

### 14.4 Repository closeout

Follow root `AGENTS.md`. Keep report paragraphs and captions on one continuous source line, with normal blank lines/Markdown structure. Generated arrays, figures, checkpoints, one-time renderers and diagnostic drivers remain in ignored local paths. Retain inspected PDF masters and only required small Markdown companions; remove only superseded presentation duplicates after checking the retained figures, never scientific histories.

Commit durable code/tests/guides/report on the non-default working branch, audit the entire outgoing commit range including added-then-deleted artifacts, run the repository artifact hook, push, and verify remote/local tips. Preserve all protected checkpoint/record bindings and the326/326 ledger. A local commit or startup message alone is not completion.

## 15. Sources and basis of this proposed plan

**S1 — Primary measured results:** `docs/reports/HONF_Unified_Training_and_Adaptive_Interaction_Refinement_Report.md` atf792dd6, especially comparison identity, completed training, native field tables, complete cost, organizer interventions and saved-pool failure. The attached report was read in full. Its formal results replace earlier development headlines but do not erase them.

**S2 — Previous development evidence:** `docs/reports/_bk/20261008_113547Z_HONF_Unified_Training_and_Adaptive_Interaction_Refinement_Development_Report.md`. Important distinctions include the actual trained H64/message64/E8 family, successful shared engine, limited transverse learning, repaired packet/normalizer guards, and inherited versus newly fitted objective scales. Its correction that an earlier H32 pilot actually used the native512-record default must not be replaced by the earlier erroneousE8 description.

**S3 — Reviewed implementation:** `src/honf_forward_core/interface_fields/interaction_core.py`; `interaction_refinement.py`; `src/honf_runtime/unified_training.py`; `Case_WindFarm/src/windfarm/shared_interaction.py`; `normalization.py`; `training/unified_task.py`; `training/unified_formal.py`; `tools/unified_interaction_train.py`; `tools/unified_interaction_formal.py`; and root `AGENTS.md`. Source findings inSection2 refer to the fetched functions atf792dd6. This planning review did not rerun training or establish a measured profiler attribution.

**S4 — External engineering reference only:** official PyTorch `torch.nonzero` documentation states the CUDA host/device synchronization behavior: https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html . The official performance tuning guide informs measurement of batching, synchronization and compilation rather than a promise of speed: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html . Check the installed version before applying an API suggestion. No external paper is treated as evidence that this HONF recipe will succeed.

**The central commitment:** preserve the efficient source-resolved family and the organizer that now genuinely executes, spend additional resources on identified Wind learning needs, and choose the next investment from controlled mature comparisons—not from another collection of unrelated gains and misses.
