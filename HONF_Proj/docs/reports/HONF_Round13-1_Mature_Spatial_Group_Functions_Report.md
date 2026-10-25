# Round13-1: mature Thermal spatial group functions

The two original spatial-group models received400 more epochs on the fixed TRAIN150 development membership, preserving their literal epoch100 optimizer, provider, RNG, sampling and100-row history. Each resulting development500 model then initialized a separate500-epoch TRAIN600 adaptation with fresh AdamW and provider/sampler clocks. Both retained the original TRAIN150 normalization and heat scale. The representation, joint prescribed-control physics, E→E exchange, early group→M relay, utility objective, exploration and response-fitting cohort remained unchanged. This campaign adds training age and native training data; it does not introduce a new model portfolio.

**Status:** closed at the user’s stop request. Both mandatory development500 and full-TRAIN500 experiments and their native VAL89 evaluations completed. The single selected Route T reserve clean-stopped with exit0 at full-data779 for F13-0 and755 for F13-1. Epoch1000 remains unfinished; no further training or endpoint evaluation was launched after the stop request.

**Archive location:** at the user’s subsequent request, all four Round13-1 run directories were moved to `_bk/14_20261025_Round13_1_Maturation` beneath the Thermal forward-run root. Checkpoints, histories and identities remain intact; recorded original paths and launch/resume commands are historical provenance. See the [current location map](../guides/Thermal_Forward_Run_Archive.md). Any future exact replay would require restoring original physical run/parent paths before execution; no continuation is scheduled.

At full-data500, F13-1 has lower canonical VAL89 mean and p90 errors than F13-0 for all nine native quantities. Its fluid-temperature mean RMSE is0.6978 versus1.0028 and surface-temperature mean RMSE is0.6363 versus0.9589. The organizer has moved beyond the predominantly root cuts of the original100-epoch screen, and the first-order modes and early relay have measurable downstream effects. Physical response errors improve unevenly and homogeneous zero-heat temperatures remain nonzero. These measured gains do not establish physical validity, convergence, independent TEST performance or inverse-design success.

## Identity, state and data clocks

The inherited representation starts at the literal balanced-singleton recovery checkpoint `thermal_interaction_groups_balanced_dev_v2/checkpoints/epoch_0100_model.pt`. Original F13 development1–100 was a model adaptation with fresh AdamW, not an exact resume of that parent. Round13-1 development101–500 is a full-state continuation of those literal F13 epoch100 states in fresh namespaces. Full-TRAIN1–500 is a new same-model adaptation from each arm's own development500 checkpoint, with fresh AdamW/provider/RNG sampling clocks as declared by the full-data recipe. Route T resumes the same full-data state at500; it is not another initialization.

The maintained runner now admits the declared future horizon and source age through a narrow F13 continuation seam. Wrong arm, membership, normalizer ancestry, model identity, named optimizer and CUDA RNG compatibility checks remain active. Historical100-epoch recipes remain valid. The original constant AdamW schedule remains LR3e-4, WD1e-5, clip1. A native disposable continuation compared100→102 uninterrupted with100→101→reload→102: tensors, named Adam state, RNG, sampler and scientific provider state matched exactly. An intermediate clean-stop validation accounts for the extra validation-sample counter in the segmented engineering replay. This55.30-second qualification is charged separately and is not a third scientific arm.

TRAIN150 uses the existing fixed-quarter manifest, preserving all available module-count categories. TRAIN600 visits all600 source training cases each epoch. Canonical VAL89 excludes0273 and is an exposed source validation partition; DEV22 and the additional67 are reported separately. No TEST membership, new solver or external physical-reference family was opened. The comparison uses the same TRAIN150-fitted normalizer and heat scale1.3164895030821728 in every stage, and the same fixed TRAIN150 mean baseline for all four endpoint evaluations. No statistics are fitted on validation.

| Exposure stage | Per-arm epochs | New primary case visits | Main updates | Primary fluid queries | Utility pairs | Response pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Inherited original F13 development |1–100|15000|400|15360000|100|100|
| A: development continuation |101–500|60000|1600|61440000|400|400|
| B: fresh full-TRAIN adaptation |1–500|300000|6500|307200000|1625|1625|
| T: F13-0 only, actual partial continuation |501–779|167400|3627|171417600|906|906|
| T: F13-1 only, actual partial continuation |501–755|153000|3315|156672000|828|828|

The last two rows are actual unequal-age partial reserve results, not a matched1000 comparison. Each utility pair means two additional physical forward/backward presentations and one detached head regression. Each response pair means two perturbed endpoint-scene presentations. These are separate from primary case visits and optimizer updates. Development has effective48/micro24 with a six-case final macro: seven primary microbatches and four main updates per epoch. Full data has25 microbatches,13 main updates and a24-case final macro per epoch. Native surface, material, vorticity and union/stencil receiver counts are retained separately in the numerical work ledger; they are not additional primary fluid-Q draws. Historical parent age, development age and full-data age must not be added and called a single comparable epoch count.

Every100-epoch scientific milestone, latest/best alias, complete history and10-epoch loss-curve refresh is retained. Development500 and full500 best-field aliases both select500 and have exactly the terminal model weights, so no duplicate best-checkpoint evaluation was needed. Both full-data arms have identical primary case/query identities across all500 epochs. The full-data epoch100 checkpoint verifies fresh Adam steps1300 for151 physical parameters and325 for four utility-head parameters, rather than inherited development optimizer ages.

## Predictor: native fields and spatial detail

All values below are equal-case native RMSE in the packed benchmark units. Temperature and q-proxy retain their dataset-native units; no unverified SI conversion is applied. Module peak is recomputed from active-module endpoint maxima and reduced as per-case endpoint-peak RMSE. Masks, p90, worst IDs, per-module details and module-count strata are retained in each `result.json`.

| Quantity | Development500 F13-0 mean | Development500 F13-1 mean | Full500 F13-0 mean / p90 | Full500 F13-1 mean / p90 |
| --- | ---: | ---: | ---: | ---: |
| u |0.152890|0.053292|0.040939 /0.050862|0.028310 /0.033247|
| v |0.025475|0.009242|0.006853 /0.008668|0.004487 /0.005810|
| p |0.099105|0.035546|0.024383 /0.031326|0.015367 /0.019448|
| Vorticity |0.328626|0.141179|0.110583 /0.137367|0.082017 /0.101017|
| Fluid temperature |3.270061|1.457613|1.002763 /1.285311|0.697816 /0.882160|
| Surface temperature |3.814133|1.892803|0.958891 /1.351028|0.636262 /0.887428|
| Material temperature |4.338853|1.797323|0.997614 /1.308580|0.692231 /0.903090|
| q-proxy |9.015229|4.265322|2.770607 /3.442495|2.230039 /2.682317|
| Module peak endpoint |4.430697|1.798689|0.897456 /1.362654|0.582690 /0.905674|

The matched architecture comparison is within each age and data regime. Moving from development500 to full500 changes both the native training population and the number of updates; it cannot identify a pure data-only effect. Internal monitoring uses sampledQ1024/surface16/material32 receivers and differs from the full-native endpoint reduction. On the internal400→500 interval, F13-0 improves all nine mean errors and F13-1 improves eight; F13-1 material-temperature mean rises0.77%. Fluid-temperature means fall11.16% and10.95%, respectively. These late improvements motivated the single unchanged Route T extension rather than treating500 as convergence.

On the fixed DEV4 surface panels at full500, mean signed-bias RMS, demeaned angular RMS and total surface RMSE are0.9667/0.7246/1.2081 for F13-0 and0.5642/0.5069/0.7585 for F13-1. On TRAIN4 they are0.6050/0.6576/0.8936 and0.3977/0.5119/0.6482. Better surface averages therefore include angular detail, while remaining residuals and worst modules must still be inspected. Detailed fields are retained only for the declared small panels, not all89 cases.

![Learning trajectories and canonical native errors](../../diagnostics/round13_1_mature_20261025/R/figures/01_learning_and_val89.png)

Figure1. Literal development500 and full500 histories and native VAL89 endpoint reductions, with the same TRAIN150 mean baseline. The table above supplies all-nine means/p90; the underlying results also separate DEV22 from the additional67. These are exposed validation measurements, not independent TEST scores.

![Full500 physical fields and signed residuals](../../diagnostics/round13_1_mature_20261025/R/figures/02_full500_physical_fields.png)

Figure2. Fixed DEV case0287 at each arm's full500 terminal checkpoint, showing physical reference, prediction and signed residual for fluid temperature, u, vorticity and material temperature at their actual saved receivers. Reference arrays are native benchmark fields; geometry gaps and masked receivers remain visible. Error maps support local field assessment, not causal interpretation of learned support.

![Surface bias, angular detail and spatial-mode interventions](../../diagnostics/round13_1_mature_20261025/R/figures/03_surface_and_slope_interventions.png)

Figure3. Fixed-panel surface bias and demeaned angular residuals, together with same-weight relay/slope interventions. At full500, removing final-read slopes raises F13-1's mean weighted native loss about93%; removing slopes throughout including the relay raises it about72%. F13-0 has no slope modes and its zero-slope intervention is unchanged. Interventions execute dependent recomputation and do not train replacement models.

Predictor gains are substantial under the measured native protocol. Remaining questions are localized thermal/peak errors, homogeneous-boundary failures and the unfinished matched trajectory through1000. A future zero-heat exposure would be a separate protocol change; it was not silently introduced here.

## Organizer: utility, coverage, common covers and executed dependence

The shared deterministic balanced hierarchy has192 environment leaves,383 total nodes and191 internal candidates. The learned cut is scene/context conditioned and reused across both interaction stages. F13-0 has one value mode per selected group; F13-1 adds two first-order spatial modes. Keys remain mean based, groups are exclusive/connected and every receiver reads all selected groups. Thus K controls representation/read work but this implementation still retains dense E→E and all-node policy statistics. It does not establish sparse end-to-end execution.

At full500, the four fixed TRAIN cover cases0006/0051/0348/0631 select K64/71/77/81 for F13-0 and107/121/89/73 for F13-1. Their learned-cover mean weighted native losses are0.022097 and0.013343, versus0.023393 and0.013537 for the diagnostic K32 cover. The learned covers are5.54% and1.44% better on this small panel. A root-to-K32 gap by itself therefore cannot be relabelled a mature selector failure. Seven common covers remain same-checkpoint diagnostic interventions, not extra fitted arms or permission to deploy a validation-selected K. No root/all-leaf/K ratio was a launch gate.

Fresh TRAIN24×3 probes at full500 give72 comparisons per arm, including three input-selected internal-node bands. Relative utility-head MAE is0.1588 for F13-0 and0.1395 for F13-1. Conditional sign agreement for normalized absolute gains above0.05 is24/24 and25/27; the eligible denominators are part of the claim. Mean selected-action regret in raw weighted native-loss units is1.67e-5 and6.57e-5. The head predicts48 descendant and24 parent actions in both arms; actual native preferences are50/22 and53/19. Finite-panel signs and low regret do not prove a globally optimal cut.

Saved full500 training records contain1625 utility pairs,813 child and812 grandchild actions across the ten available module-count strata1/2/3/4/5/6/7/9/10/12. The same index still chooses `stratum %10` and `action %2`, so each stratum is coupled to one action parity. More total probes do not imply all stratum/action combinations were observed. The baseline procedure was preserved during maturation and Route T; this coupling remains a concrete future sampler-amendment hypothesis, not a change included in these results.

Effective auxiliary weighting also differs by data regime. Every development utility pair falls on the six-case final macro and contributes pair-average native loss with total weight1/6. In full500,1500 pairs fall on48-case macros and125 on24-case macros, with total pair-native weights1/48 and1/24. Each of its two alternatives receives half that weight. The detached utility-head regression and response-gradient cap are distinct terms. Equal pair counts or coefficients do not make the development and full-data scalar objectives identical.

At full500, bypassing the early group→M relay raises fixed-panel mean native loss about393% in F13-0 and247% in F13-1. Removing organizer-only relation features changes the mean about+0.26% and0%, respectively; ME relations remain active in this intervention. The full500 F13-1 input-sensitivity check uses three preparations of saved case0348. A heat-transfer change preserves both-stage K89 cuts while changing the post-relay M latent by1.25%, relay message by0.79% and stage latents/relations by nonzero amounts. A geometry change alters both cuts and changes post-relay M by3.40% and the relay message by4.44%. Executed dependence is present even where a hard cut is unchanged. Neither learned relations, support maps nor sensitivity norms establish physical causality.

![Actual spatial support and reusable group functions](../../diagnostics/round13_1_mature_20261025/R/figures/04_actual_spatial_support.png)

Figure4. Saved selected groups, actual relation/support information and final-stage reusable reader coefficients at full500. F13-1 carries three modes per group and F13-0 one. These are executed model objects; latent coefficients are not physical temperature fields. Stage0 reader coefficients are released during ordinary evaluation, so this figure uses the retained final-stage coefficients and does not imply a missing stage0 dump was measured.

![Utility coverage, conditional errors and common-cover gaps](../../diagnostics/round13_1_mature_20261025/R/figures/05_utility_coverage_and_covers.png)

Figure5. Saved provider exposure counts, fresh TRAIN72 utility comparisons, seven same-weight covers and bounded off-path contexts. Utility labels come from paired native losses with dependent relay/stage-two recomputation; validation labels never train the head. Conditional sign denominators, raw regret and observed action coverage distinguish head fitting from a global partition guarantee.

The organizer now supplies useful nontrivial partitions and spatial functions on measured panels. The action-coverage alias and dense execution remain unresolved. Future work should isolate coverage repair or a representation/key restriction in one controlled experiment rather than infer a remedy from a desired K.

## Prescribed-control and geometry responses; inverse limits

The eight saved reference families are0001,0304,0318,0320,0333,0335,0348 and0350. Only0348 supplies response-loss gradients throughout this campaign. Other families are split into TRAIN-reference-only and response-validation-only cohorts. After TRAIN600 fitting, their baseline native scenes may be training examples; held-out response supervision does not mean unseen native layouts. No new physical solver or inverse optimizer was run, and optional ranking was not expanded.

At full500, across component/family summaries the median relative non-null heat-response error falls0.3517→0.2021 from F13-0 to F13-1 and single-geometry error0.9351→0.5833. Mixed central derivatives remain poor at8.064→4.880; raw four-corner contrasts remain poor at3.528→2.915. These medians are descriptive combinations of components with different units and physical magnitudes; the retained tables preserve each role's absolute error, reference RMS, cohort, valid count and near-null exclusions. The corresponding aggregate raw four-corner absolute-error median worsens0.04723→0.05996, despite the lower relative summary. There is no uniform response improvement claim.

Response-validation-only relative medians are heat0.3292→0.1798, single geometry0.9811→0.9285, mixed central47.83→19.41 and raw four-corner5.080→2.345. The final raw figure has only one valid relative component and35 near-null exclusions; it cannot support a broad held-out interaction claim. Large central derivative errors remain a clear miss. Raw four-corner contrasts and central mixed derivatives are reported separately and are not interchangeable physical quantities.

At homogeneous zero heat, full500 fluid-temperature RMS is2.6935 for F13-0 and3.1580 for F13-1, surface-temperature RMS5.0615/4.6790 and material-temperature RMS5.9464/5.3010. q-proxy RMS is7.3794/5.0180. Longer ordinary fitting has not enforced the homogeneous temperature solution. The relative native validation gains do not remove this physical failure or certify inverse designs. No heat-to-flow independence constraint or decoder correction was added.

![Saved physical-response families](../../diagnostics/round13_1_mature_20261025/R/figures/06_physical_response_families.png)

Figure6. Saved heat, single-geometry, mixed-central and raw four-corner response evidence at full500, with relative component medians and excluded near-null counts shown by cohort; physical-reference RMS and absolute errors remain in the saved analysis tables. Selected response maps show signed differences in their native units. These arrays are reused physical evidence, not newly solved responses or successful inverse-design trials.

Response gains are useful but incomplete. Inverse performance remains unmeasured: neither sampler diversity, learned organization nor saved forward response panels establishes valid optimized designs. A future inverse study needs explicit physical evaluation and is outside this campaign.

## Executor and complete cost

Timing uses one warmup followed by three synchronized uninstrumented repetitions for native prepare, cached decode and complete native calls on M1 case0006 and M10 case0348, each atQ32/Q1024/full native. All selected groups are read, retained E→E remains dense, and original M/E raw value-query bypass counters are zero. Prepared tensor-storage counts exclude Python objects and are not total process memory; CUDA allocated/reserved peaks are reported separately. Small Q still includes the maintained union of native surface/material/stencil receivers and is not a standalone32-query decoder benchmark.

At full500, F13-0 complete native median times for M1 Q32/Q1024/full are23.73/25.53/37.48ms and M10 are28.14/29.76/38.72ms. F13-1 is about24.02/26.2/38.7ms and28.8/30.8/41.3ms, respectively. The complete24-case final optimizer-macro replay is1.964s versus2.088s, with input preparation measured separately. The replay restores endpoint weights/moments/RNG/provider state and uses the saved final request counters; it is not a historical replay with unavailable pre-final weights. Main loss/backward, paired physical alternatives, detached head loss, response exposure and Adam step are all included. F13-1's field gains at500 do not provide a complete execution speedup on these measurements.

One warm F13-1 development500 M10 profile measured complete core preparation29.02ms, native union/stencil construction15.07ms, stage-one/all-node moments4.16ms and stage-two moments4.46ms; selected coefficient formation was about0.50/0.52ms. Complete instrumented native time was84.09ms, including16 final query tiles and retained E→E. These nested stage timings are not additive independent costs, and diagnostic instrumentation differs from uninstrumented timing. Host geometry/cut work, both-stage all-node moments and native receiver construction remain concrete overheads. No unqualified function-changing performance patch was inserted into active maturation.

![Complete work, timing, memory and homogeneous-boundary errors](../../diagnostics/round13_1_mature_20261025/R/figures/07_work_cost_memory_zero_heat.png)

Figure7. Native complete-call and optimizer-macro cost, CUDA allocated/reserved peaks, fidelity against primary work and zero-heat residuals from literal500 endpoints. Actual process-associated training charges are reported separately below. A smaller cut does not imply fewer dense E² interactions or lower complete latency, and concurrent process-wall hours are not hardware utilization.

| Completed process | F13-0 elapsed | F13-1 elapsed | Summed process-associated hours |
| --- | ---: | ---: | ---: |
| A: development101–500 |2398.68s|2832.94s|1.4532|
| B: full-TRAIN1–500 |19577.00s|21576.36s|11.4315|
| Native endpoint evaluation, development500 |49.94s|51.38s|0.0281|
| Native endpoint evaluation, full500 |31.35s|32.02s|0.0176|
| Cost checks, development500 |23.47s|23.67s|0.0131|
| Cost checks, full500 |25.76s|26.00s|0.0144|

Qualification55.30s and the F13-1 input-sensitivity check9.47s add0.0180 process-associated hours. The base measurements listed above total approximately12.976 hours, well below the32-hour base cap; CPU-only unit tests and rendering are separate from GPU-process charge. Provider saved elapsed at full500 is19553.2731/21551.9680 seconds and excludes construction/pre-bind time, so it is not substituted for GNU-time whole-process charge. The user-stopped reserve adds11104.44s and11137.66s (6.1784 process-associated hours). The entire measured campaign charge is68955.44s, or19.1543 process-associated GPU-hours, below the32-hour base and48-hour total ceilings; reserve use is below16 hours. Concurrent process-wall charge is not hardware utilization.

## The one selected reserve: unchanged Route T

The ordinary [reserve recipe](../../configs/research/thermal_spatial_groups_r13_1_reserve_t_v1.json) records the decision after both full500 evaluations: both arms still improve materially, including about11% lower fluid-temperature means over the final monitored interval. Both therefore resumed their original full-data500 states toward1000 with named Adam, provider, RNG, absolute schedule and primary/auxiliary sampling preserved, until the user requested an immediate wrap-up. No Route U amendment, additional seed or K controller is run. The observed coverage alias and zero-heat failure remain present rather than being silently repaired during this continuation.

The runtime allocation is kept outside scientific config identity. Absolute cumulative provider-time caps are48354s and50352s, each about eight provider-hours beyond its own saved500 elapsed, with the existing600-second clean-save reserve. Omitted resume flags reuse the saved cap rather than granting a fresh allowance. GNU-time process charge is tracked separately under the16-hour reserve and48-hour campaign ceiling. The existing signal handler clean-stopped F13-0 at779 and F13-1 at755, with exit0 and both owned GPU processes released. Latest/best-field checkpoints select779/755, all declared milestones through700 remain, complete histories have779/755 rows, and final PDF/PNG curves and sampled native VAL89 clean-stop reviews are saved. The original500-row history prefixes are unchanged. Final cumulative full-data visits/updates/utility pairs/response pairs are467400/10127/2531/2531 and453000/9815/2453/2453. Final sampled validation field scores are0.0219509 and0.0109990; these unequal-age internal scores are descriptive and are not substituted for matched full-native evaluation. The four completed500 native evaluations remain the scientific comparison. No full-native evaluation or cost replay was run at the partial stops. Finishing1000 would require221/245 further epochs (132600/147000 visits and2873/3185 main updates), and requires a new user instruction to resume; no waiter or automatic continuation remains.

## Reproduction, evidence and closeout

The [maintained recipe](../guides/Thermal_Spatial_Group_Functions_Recipe.md) contains the actual development continuation, full-data adaptation and Route T resume commands. Run in `ModularDT` with one physical device visible per process and logical `--device cuda:0`; GPU visibility may change between processes while the number of saved CUDA RNG streams remains compatible. The supported native evaluator is `tools/honf_research.py evaluate-contextual --kind spatial-groups --checkpoint <literal checkpoint> --device cuda:0 --output <new DATA2 directory>`. Reusing an existing completed namespace through a fresh-train command is rejected; use the maintained same-run resume path.

Durable changes are in the maintained contextual runner, narrow F13 continuation helper, Thermal spatial task, spatial evaluator, CLI, four maturation configs, reserve recipe and focused tests. The initial integration suite passed66 tests; the runtime-budget revision also passed66 focused tests in8.77s. Native qualification and actual A/B fitting supply execution evidence beyond config parsing. Existing plotting/checkpoint cadence, safe checkpoint loading, job ownership and artifact hooks remain active. No new hash/freeze/snapshot/admission service was introduced.

All generated artifacts remain on mounted DATA2 through the existing workspace symlinks. Numerical evidence is under `diagnostics/round13_1_mature_20261025`: four literal500 evaluation directories with JSON/NPZ, complete endpoint cost records, `analysis_base500.json`, response/input-sensitivity evidence and compact qualification/state receipts. Run roots are under `Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_1_{dev,full}_f13_{0,1}`. The selected figure index is pages1–7 above, with one [seven-page PDF master](../../diagnostics/round13_1_mature_20261025/R/figures/round13_1_mature_evidence.pdf); no eighth reserve page was generated because the user ended the unequal-age partial continuation before the planned endpoint. Generated arrays/figures and one-time analysis/rendering helpers are local ignored artifacts and are not uploaded to Git.

Closeout removed180 round-owned disposable files/symlinks (110568716 logical bytes;110931968 allocated bytes) and213 empty child directories after exact inventory, ownership/inode/link and process-reference checks. This covers test fixtures, qualification checkpoint/plot payloads and unlaunched1000-review wrappers. Compact qualification histories/manifests/logs and passing receipts remain. Earlier cleanup removed27 superseded figure exports (23831723 bytes) and388038 bytes of initial fixtures; total removed payload is134788477 logical bytes. All scientific milestones/latest/best states, complete histories, selected numerical response/field arrays, seven-page PDF/PNG evidence, compact provenance and cleanup receipts are retained. The original six R13 checkpoint/alias metadata records remain unchanged; unrelated jobs were not signalled. The final retained campaign payload is1439923877 logical bytes (about1.34GiB), below12GiB. DATA2 has3214827577344 bytes free and home has82346819584 bytes free; all three declared output/cache symlink bindings still resolve to mounted DATA2. Scientific outputs on home remain0 bytes. The compact local stop/storage/cleanup receipts record the observations and exact removal inventory. Durable changes are audited and pushed on the existing non-default branch; generated artifacts remain ignored and local.
