# HONF focused organizer utility and predictor diagnosis

This round separates useful source filtering from useful receiver grouping, and physical-weight adaptation from source-access error. We reused Wind u4910 and Thermal u1300 evidence, added a six-case Wind replay, tested one matched 40-update Thermal correction, and completed four frozen Wind inverse trails. The historical maturation runs and deployment thresholds are preserved. No new reference or CFD solves were performed.

| Goal | What we gained | What remains missing | Strength of evidence | One next action |
|---|---|---|---|---|
| Predictor | Thermal value weight 8 improves 14/16 fixed development absolute channel/case comparisons against the weight-4 pilot; median RMSE ratio 0.9821. The inherited Wind G-versus-P average gain is preserved. | Thermal still loses to Run1804 in all 16 development absolute comparisons; response and peak-temperature misses remain. | Two exposed Re90 families, paired 40-update fits, stored benchmark targets; no independent new physics. | Evaluate this single value/response-balance hypothesis on additional independent training families before requesting maturation. |
| Organizer | Wind availability, ranking and margin rejection are separated. Active MM interventions and exact work counts are now visible. | Added receiver grouping has no demonstrated advantage over simpler controls at matched work in this panel; safe sparse selection and executor savings are missing. | Three training layouts; six input-selected native cases with disjoint query checks and fixed weights. | Reconsider packet construction against the geometry control before enlarging the K-selector. |
| Inverse | Four complete, immediately persisted trails demonstrate active candidate links and a functioning frozen interface. | All decoded designs saturate identically, miss the changed condition, and leave native support. | One task/seed, same u64 denoiser, forced links, surrogate scoring only. | Diagnose denoiser scale and sigmoid saturation before another sampling or fitting round. |

**Was Wind blocked by selection or by grouping? Both.** Adequate saved sparse actions exist in 13/18 primary cases, but the archived chooser rejects all of them. Removing its margin selects inadequate actions in 9/18 cases. Separately, the native replay finds source-filtering gains that a root action can match, and geometry controls that match or reduce work while improving reconstruction.

**Did Thermal lose accuracy before masking? Yes.** G full access is worse than retained Run1804 in all 16 development absolute channel/case comparisons. Its median signed adaptation change is +1.099 times the same channel's baseline squared error; the additional access change is +0.132. This is exact output-space accounting, not a causal assignment of training mistakes.

**Was a graph distinction effective and useful?** The interventions change actual access, but added grouping has not earned a consistent reconstruction/work advantage. In inverse sampling, links and raw logits change while decoded coordinates saturate. Neither result establishes physical causality or a reusable, physically valid design advantage.

**What finished?** Both diagnoses, the paired pilot, four inverse trails, five inspected figures and focused software checks. All GPU work ended by 18:00:54 UTC. There was no maturation continuation, new Thermal selector, new reference solve, larger inverse matrix, extra seed, or automatic follow-on job.

## Evidence scope and identities

The first substantive inspection was **2026-10-02 16:50:11 UTC**. The working branch is `agent/honf-core-next`, with evidence baseline `b55dd28`. Wind ran on physical **GPU1**, UUID `GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e`; Thermal ran on physical **GPU2**, UUID `GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`, using the **ModularDT** environment and explicit `cuda:1`/`cuda:2`. GPU0 was not used by this round. Native work overlapped across the two authorized lanes.

Wind subjects are Run2112 G/P **u4910**, with retained Run2110 W-full **u1500** as a learned control. Wind targets are stored OpenFOAM CFD velocities in m/s; coordinates are in the native downstream/crosswind frame in rotor diameters, **D = 80 m**. Thermal subjects are G/P **u1300**, retained Run1804 **e4738**, and one earlier same-lineage G state **u900**, before the recorded value-weight transition at u901. The historical comparison used saved u900/u1300 arrays; no fresh u200 replay was run. Thermal targets are the stored analytic-wake/shared-grid benchmark, in dataset units. `q_normal` is a heat-flux proxy; no SI conversion is established. Learned retained predictors are controls, not physical truth.

Previously inspected development and held-audit outcomes remain exposed research evidence. Direction rows, roles, query repeats and module peaks are not independent test families. The historical [controlled-maturation report](20261002_132648Z_HONF_Controlled_Maturation_and_Action_Aware_Organization_Report.md) retains the prior all90 reviews and repeated timing benchmarks; they were not rerun or counted as new results here.

## Wind: availability, ordering and calibration

The saved table has **21 cases / 84 actions**: 18 primary cases plus three disjoint-query repeats from one layout. The nine training direction rows contain **three layouts** (24, 120, 173); development has six rows/two layouts (91, 198); exposed audit has three rows/one layout (190). We reproduced all 18 historical full-access choices. The saved final head reproduces all 36 development/audit action rows; training replay uses archived out-of-layout predictions because fold weights were not saved.

| Quantity on primary cases | Train | Development | Exposed audit |
|---|---:|---:|---:|
| Adequate sparse option exists | 8/9 | 3/6 | 2/3 |
| Historical chooser takes a sparse action | 0/9 | 0/6 | 0/3 |
| Diagnostic removal of margin selects a false-safe action | 4/9 | 4/6 | 1/3 |
| Full fallback meets the retained-W-full physical allowance | 4/9 | 5/6 | 3/3 |
| Train-selected fixed `two_packet` meets that allowance | 8/9 | 2/6 | 2/3 |

All **63/63** saved sparse action rows, including repeats, are eligible. All **33 adequate primary sparse action rows** pass raw predicted fixed limits; the saved empirical margin causes every rejection. Downstream envelope is the largest margined violation on all 33. Raw prediction overstates at least one role on 32/33 adequate actions, yet removing the margin admits **21 false-safe candidate rows** and selects a false-safe action in **9/18 cases**, reaching **1.3604 times** its physical allowance. A full fallback with no selected sparse false-safe is therefore not evidence that the full predictor is adequate.

In role order volume, hub slab, downstream envelope, near turbine, background, saved log margins are **0.13081, 0.15074, 0.20084, 0.08985, 0.12961**; the log limits are approximately **log(1.1) = 0.09531**. Replacing fixed limits by retrospective per-case limits changes them by at most **1.40e-7 log units** and changes zero decisions on this table. Those per-case limits require reference outcomes and remain diagnostic. The empirical margins are not formal coverage guarantees.

![Wind measured/predicted risk and native source picture](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure01_wind_selection.png)

**Figure 1.** The same 54 primary sparse actions appear before and after the saved margin. Row594/layout198/270°/M27 was chosen by input metadata for the source picture and role trace. Its saved root packet selects 24/27 module sources and all 512 environment sources. Packet centers are frontier summaries, not native MM receiver axes; incidence counts do not imply native-pair compression. The figure supports margin-induced rejection and poor raw acceptance, while preserving the physical-reference and layout-count limits. [PDF master](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure01_wind_selection.pdf).

One bounded ranking remedy used eight-component PCA/ridge on **input-only sparse-minus-same-G-full descriptors**, predicting each role's `log((E_sparse + epsilon)/(E_G-full + epsilon))`. Scaling, PCA and regression were fitted within each leave-layout-out fold. Three folds plus one final training fit give **four analytic solutions and zero new optimizer steps**. No development/audit outcomes selected this recipe.

On the **34 rolewise training contrasts** larger than the archived query-repeat discrepancy, concordant ordering is **6/34 (17.6%)** for archived neural OOF, **10/34 (29.4%)** for archived absolute ridge, and **11/34 (32.4%)** for the new relative ridge. These contrasts are clustered within three layouts. Relative minimax matches the repeat-tolerant measured-best set on 6/9 training, 5/6 development and 3/3 audit rows; the audit still contains one layout. The remedy misses most resolved training contrasts and does not calibrate absolute risk. Deployment margins, limits and full fallback remain unchanged.

## Wind: does receiver grouping add value?

The six cases were chosen before outcomes from input metadata: training rows72/73 (layout24, M6, 270/285°), row360 (layout120, M7, 270°), row519 (layout173, M21, 270°), and development rows594/595 (layout198, M27, 270/285°). A same-M/different-layout pair was unavailable within those train/development domains. Each has the archived **2,048-query role panel** and one new panel with **zero overlap of native query-index sets**. Cross-role duplicate indices in the fixed panel remain valid samples, rather than new independent receivers.

All six fixed G-two results reproduce archived role RMSE within **5.5e-8 m/s**; forwarded G-four results within **5.6e-8 m/s**. Both requested cuts have nonredundant **K = 2**, but K equality is not access equality. Exact live MM/QE equality holds on 4/12 panels. Rows72/73/360 have meaningful MM differences between cuts (2/2/8 entries, maximum weight change 1); the other exactness failures have QE roundoff around 5.96e-8 without support changes. A CPU replay can differ at that epsilon scale. No distinct K=4 computation or adaptive-K success is claimed. The archived table itself has different two/four errors on rows72/73/360.

| Fixed-weight comparison, confirmed on both query panels | Measured result | What it supports |
|---|---|---|
| M6: two packets versus root union | 25 versus 30 selected MM pairs, out of the same 30 eligible pairs; lower error in every role | Source filtering can help, but the root action matches the two-packet prediction and work. |
| M7: geometry versus two packets | Exactly 36 MM pairs for each; geometry lowers all five role errors | Added receiver grouping does not help this matched-cost layout. |
| M27: geometry versus two packets, both directions | 624 versus 626 MM pairs; geometry lowers all five role errors on all four case/panels | A simpler control wins with slightly less effective work in this development panel. |
| Whole source-permission row swap | Changes 5–156 MM weights; QE changes zero entries; adds 5–17 MM pairs | The intervention is active, but never strictly work matched. Its error effects are mixed. |

The swap exchanges complete MM/QE permission rows on fixed receiver supports, preserving source measure, validity, self-exclusion and phase. A simultaneous relabelling of both packet axes leaves the effective matrix unchanged; that invariance is not the intervention. Row594's swap changes **156 MM weights / 139 binary support entries**, increasing MM pairs from **626 to 639**. Root union exposes all **702** eligible MM pairs; its lower error can purchase more information.

![Wind role residuals, effective access and work](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure02_wind_grouping.png)

**Figure 2.** Row594 shows the 256 sampled hub-slab receivers projected onto native x/y, spanning z/D = **0.416–1.370**, rather than a solved plane. Numerical vector-role RMSE is unclipped; streamwise residual colors clip at ±**0.1438 m/s** for display. Grouped/root-union/geometry/swap hub-slab RMSEs are **0.020825/0.015549/0.015406/0.014900 m/s**, at **626/702/624/639** MM pairs. The displayed matrix uses physical source/receiver identities, labelled by native coordinates. Geometry wins here at slightly lower work; the swap and union are not work-matched controls. [PDF master](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure02_wind_grouping.pdf).

On this 2,048-query M27 panel, every shown action executes **1,048,576 QE rows** and **55,296 QM rows**. QM is not MM. Canonical QE support uses a different receiver panel and must not replace these live counts. Canonical total-work fractions are **99.961/100.000/99.960/99.967%**; environmental/full-access work dominates. MM execution-row savings are unavailable from these counters. The four synchronized single-call wrapper times are **94.6/47.4/97.8/93.2 ms**; they are operational units, not repeated speed benchmarks. No overall executor saving or speedup is established.

Direction transfer uses each row's own native frame. For layout198, slotwise pair-distance matrices agree within **1.33e-6 D**, with proper-rotation fit RMS **2.83e-7 D**. There is no explicit external turbine-ID field; we do not equate cross-direction sources solely by tensor column. No new solved source-perturbation field was available, so no physical finite-difference or causal-interaction claim is made.

## Thermal: adaptation, access and one correction

For aligned vectors, the reusable helper accounts for `e_B = F_B(full)-y`, `d = F_G(full)-F_B(full)` and `h = F_G(hard)-F_G(full)`. With each channel's common positive-weight mask, adaptation is `2<e_B,d> + ||d||²`, and access is `2<e_B+d,h> + ||h||²`. Signed cross terms are retained; maximum absolute numerical closure residual across all three identities and all G rows is **1.33e-15**, at dev0310 absolute interface flux-proxy adaptation. These deltas are not additive RMSE differences or unique causal training components. Different physical channels are never summed as a mixed-unit physical error.

The fixed families are training **0304/M3 and 0348/M10**, and exposed Re90 development **0310/M3 and 0355/M10**. Baseline and previously solved `i_plus` responses are compared on identical Eulerian receivers or module-attached IDs. The response is raw `i_plus - baseline`, without division by displacement: x moves are approximately **0.15/0.075/0.15/0.15** dataset-length units for moved modules **0/4/1/2**, respectively.

| G-u1300 comparison against Run1804 | Train, 16 channel/cases | Development, 16 channel/cases |
|---|---:|---:|
| Absolute full-access RMSE improves | 1/16 | 0/16 |
| Raw finite-response full-access RMSE improves | 12/16 | 5/16 |
| Absolute access increases squared error | 6/16 | 12/16 |
| Median absolute adaptation delta / channel baseline squared error | +0.919 | +1.099 |
| Median absolute access delta / channel baseline squared error | −0.073 | +0.132 |

P-u1300 also worsens all 16 development absolute channel/cases before masking; hard access worsens 13/16 further. Thus a new Thermal selector has no credible adequate-action set. Negative access terms on some channels partly compensate full drift and remain visible.

![Thermal fields, signed attribution and finite responses](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure03_thermal_attribution.png)

**Figure 3.** Input-selected train0304 has **1,687 valid sampled fluid receivers**; dev0310 has **7,976/8,192** valid native-grid receivers. White unqueried regions have no field estimate. Full temperature RMSE is **0.3179/0.2191** dataset temperature units; RMS of hard-minus-full temperature is **0.072/0.165**. Colors clip only for display. Each bar normalizes by its own channel's baseline squared error. Dev0310 was selected by the declared largest hard-G signal-relative error rule over the two predeclared development families: interface flux-proxy RMSE is **0.90579/1.75007/1.80148** for Run1804/G-full/G-hard. The module0 raw response follows material receiver angle; dev0310's moved module is module1. This supports pre-access deficit and additional channel-dependent distortion, without an SI flux or causal claim. [PDF master](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure03_thermal_attribution.pdf).

The earlier u900 comparison covers eight saved training families / 64 physical channel comparisons. u1300 improves **55/64** relative to u900; median RMSE ratios by role are fluid **0.9525**, interface **0.8699**, solid temperature **1.1097**. The later value-weight transition therefore does not show uniform deterioration and does not uniquely explain the already-poor retained-model comparison. The historical u1300 run was resource-censored, not established converged.

The actionable training hypothesis was stronger value fit while holding response learning fixed. Two temporary copies from G-u1300 used full access, **80 physical tensors trainable**, **40 route tensors frozen**, shadow disabled, identical 40-update family/history/query stream and checkpoint RNG, and retained AdamW state (**lr 1e-5, weight decay 1e-5**). The only loss-weight change was **value 4 → 8**; finite, finite-peak, pressure-response and pressure-value coefficients remain 1. Historical-value and retained-anchor mechanisms remain in the actual maintained objective; anchor coefficient **0.114926** is unchanged. Every route tensor and its optimizer state stays exactly equal to u1300; all 80 physical tensors change. Development outcomes did not choose this recipe.

Both arms end at **u1340 after exactly 40 updates**. The scheduled u1320 training review stopped the control segment; its saved RNG/optimizer/cursor checkpoint was resumed only to the already-authorized u1340 endpoint. Combined fit wall time is **435.58 s**, below 75 minutes. The first actual step took **5.692 s** and peaked at **14.75/14.84 GB allocated/reserved**. Total-objective initial physical gradient norms are **2.5220/2.9627** at the same u1301 stencil; route gradients are absent. These are not component-gradient or causal dominance measurements.

| Fixed after-pilot measurement | Value4 | Value8 | Incremental value8 versus value4 |
|---|---:|---:|---:|
| Train absolute channel/cases improved versus u1300 | 11/16 | 14/16 | 14/16 wins; median ratio 0.9772 |
| Development absolute channel/cases improved versus u1300 | 9/16 | 9/16 | 14/16 wins; median ratio 0.9821 |
| Development absolute median/max RMSE ratio versus u1300 | 0.9783 / 1.0968 | 0.9707 / 1.1108 | Small median gain, persistent regressions |
| Development finite-response channel/cases improved versus u1300 | 9/16 | 10/16 | 9/16 wins; median ratio 0.9990 |
| Development absolute channels better than Run1804 | 0/16 | 0/16 | No retained-control recovery |

Pressure and module peaks are separate functionals. Across four families, baseline pressure-drop MAE is **7.12e-4/7.87e-4** for value4/value8, versus **8.76e-4 u1300 / 1.031e-3 Run1804** in dataset pressure units; both pilots reproduce all **8/8** stored pressure-feasibility labels. Value8 worsens this absolute pressure MAE against value4. Raw pressure-response MAE is **8.245e-5/8.213e-5**, with value8 winning only 2/4 families.

All **26 family-qualified module IDs** are retained for baseline and i_plus (52 module-state references per model). Baseline peak-temperature MAE over these modules is **0.426/0.408** for value4/value8, versus **0.302 Run1804 / 0.413 u1300**; i_plus is **0.486/0.470**, versus **0.425/0.468**, in dataset temperature units. Value8's maximum baseline peak error is **1.3055** at `0355:module:7` versus **0.8051** for Run1804 at that ID. Its i_plus maximum is **1.6139** at `0348:module:9`, versus Run1804's panel maximum **1.1712** at a different ID (`0348:module:6`). No matched-slot or global worst-case equivalence is implied. The pilot supports a small absolute-value balance gain, while missing retained field and peak fidelity and giving little response improvement. It remains a fit probe; no continuation or deployment promotion follows.

## Frozen inverse: active links, saturated output

After the primary diagnoses were saved, we reused the selected Wind **u64 inverse checkpoint** on development row50/layout16/300°/M10, with one hidden turbine. Exactly **four trails** use the same denoiser, same initial projected state and reverse-noise seed, **20 existing reverse steps**, and batch one. Original observations and one changed input each run forced `two_packet` versus `full_access`. Observed x at fixed sensor14, coordinate **(12.2, 5.0, 0.875)D**, changes **8.299531 → 5.537949 m/s**, within marginal training range **[5.537949, 8.989769] m/s**. Joint physical feasibility of that observation change is unvalidated.

The provider uses visible positions, fixed sensor coordinates, public context and the current candidate; it rebuilds candidate encoder/tree/scores every step. Hidden clean positions enter only privileged retrospective scoring after terminal persistence, never links, bounds or caches. Normal deployment/full fallback is unchanged. The inverse forced action uses QE-active candidate access, which differs from the forward replay's QE-full states and is not validated environmental sparsification.

The archived preflight was bound to a different Git revision (`a1c07d2`); its rejection and safeguards were preserved. A separate maintained-API study restored the existing trusted checkpoint/task bindings. The first complete trail persisted before a score-adapter shape failure. Removing accidental `[0]` indexing restored the expected `[M,3]` centers; the saved terminal was scored without redrawing. A recovered metadata list initially reversed state filenames; CPU extraction was repaired by actual timestep names, leaving saved sample files intact. Direct array equality confirms all four initial states match and each callback t00 equals its terminal.

![Wind inverse complete paths, links and observation residuals](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure05_inverse_saturation.png)

**Figure 5.** All four decoded paths and terminal designs coincide after float32 sigmoid saturation; the terminal hidden turbine is **(−15, 15)D**. Raw logits still differ by up to **0.243866**. At the terminal, actual packet/full differences are **9/90 MM** and **688/8,192 QE** entries; ME/EM/QM are unchanged. Original-condition trajectory totals across 21 provider states are 192 MM and 14,448 QE changed entries. Public center-box/support and rotor-clearance checks pass (**2.7421D** clearance), but native-row support fails. Surrogate observed RMSE is **0.062202 m/s** against original inputs and **0.399408 m/s** against changed inputs; held original-sensor RMSE is **0.067159 m/s**. Hidden-layout distance **19.4075D** is post-sample geometric error. These are model-only checks, not generated-layout CFD validation, valid designs or sampler diversity. [PDF master](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure05_inverse_saturation.pdf).

The first sample/persistence interval is **4.410 s** from atomic-file timestamps; its initial CUDA peak was not captured before scoring failed. The recovery session's original pair decision took **7.276 s** after model load; later trail durations including terminal link rebuild are **2.725/4.546/2.730 s**, with peak allocation **112 MiB** in that resumed scope. The entire optional allocation, including CPU readiness, failed scoring and recovery, is conservatively charged **16.895 minutes** from 17:44:00 through 18:00:53.676 UTC, below 20 minutes. No new inverse fitting or further trails were started.

## Resource closeout, figures and reusable software

![One elapsed clock, charged GPU work and pilot trends](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure04_timeline_and_pilot.png)

**Figure 4.** The one elapsed clock includes diagnosis, coding, startup failures, native work and closeout. Wind attempt envelopes total **160.943 s**; Thermal owned processes including two startup failures and the fixed assessment total **476.736 s**. Conservatively charging the whole inverse allocation gives aggregate GPU-associated charge **1,651.355 s = 0.459 h**, below 10 hours. This is an upper-bound charge, not a CUDA-kernel-time measurement; the inverse interval includes CPU work because its initial failed process lacked a complete timing/peak receipt. Fit, inference and setup subscopes are already inside job durations and are not added again. Loss curves show unweighted checkpoint-normalized training terms; fixed physical role ratios show the small correction gain. Final elapsed delivery time and remote synchronization are recorded in the [local closeout receipt](../../../diagnostics/generated/focused_diagnosis_20261002/closeout_receipt.json), after this report's measurement snapshot. [PDF master](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure04_timeline_and_pilot.pdf).

| New work scope | Completed / authorized limit |
|---|---|
| Wind inference-only complete-wrapper attempts | **73/128**: 68 native grouping calls + 4 successful terminal scores + 1 failed score |
| Thermal inference-only complete-wrapper attempts | **23/128**: 16 fixed assessment calls + 7 setup captures, including failed-start captures |
| Paired Thermal optimizer updates | **40/40 per arm**, 80 total; no extension to 60 |
| Training-scoped complete wrappers | **960 student + 176 no-grad teacher anchor-loss calls**, required by the maintained objective; reported separately from standalone inference |
| CPU selector optimization | **0/600 optimizer steps**; four analytic ridge solutions |
| Inverse trails / fit | **4/8 complete trails**, zero new inverse optimizer steps |
| New reference/CFD solves | **0** |

The six-hour deadline is 22:50:11 UTC, with new optimization/sampling prohibited after 21:50:11. All native work ended over four hours before that deadline, preserving the 60-minute closeout reserve. The [preliminary progress note](../../../diagnostics/generated/focused_diagnosis_20261002/progress_note.md) was saved at 17:52 UTC, before the two-hour checkpoint. No unrelated process was interrupted.

Reusable changes are limited to the aligned weighted-error attribution helper, train-layout-only relative PCA/ridge fitting, and a detached-state inverse progress callback. Default deployment selection is unchanged. The callback reports initial timestep and each completed reverse update; exceptions stop sampling while caller-persisted snapshots survive. Focused tests verify negative cross terms and closure, grouped folds/held-label isolation, default-margin preservation, terminal survival after interruption, and current-candidate provider exclusion of hidden targets. **39 focused tests passed** in ModularDT; Ruff and `git diff --check` passed. The combined provider suite initially needed the Wind source directory on `PYTHONPATH`; rerunning with the correct source paths passed. No tests are presented as physical evidence.

The single figure index below retains PDF masters and only the small PNG companions needed for direct Markdown display. All five were visually inspected against saved arrays, with native coordinate/mask checks and display-only clipping. Superseded exports were overwritten; no extra generated visual versions are retained. Plots, numerical appendices, one-time runners and pilot checkpoints remain in ignored local paths and are excluded from outgoing Git history. Reusable source/tests and this report are committed and pushed to the existing non-default branch under the full outgoing-history audit and `.githooks` artifact gate; final tip equality is in the closeout receipt.

| Selected figure | PDF master | Principal numerical source |
|---|---|---|
| 1. Wind selection | [PDF](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure01_wind_selection.pdf) | [Selector trace and result](../../../diagnostics/generated/focused_diagnosis_20261002/wind_selector/wind_selector_result.md) |
| 2. Grouping value | [PDF](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure02_wind_grouping.pdf) | [Native replay result](../../../diagnostics/generated/focused_diagnosis_20261002/wind_groups/wind_group_result.md) |
| 3. Thermal attribution | [PDF](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure03_thermal_attribution.pdf) | [Aligned attribution](../../../diagnostics/generated/focused_diagnosis_20261002/thermal/t1_t2_attribution.json) |
| 4. Timeline/pilot | [PDF](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure04_timeline_and_pilot.pdf) | [Paired audit](../../../diagnostics/generated/focused_diagnosis_20261002/thermal/t3_paired_value_weight_pilot/paired_training_audit.json), [resource accounting](../../../diagnostics/generated/focused_diagnosis_20261002/resource_accounting.json) |
| 5. Inverse saturation | [PDF](../../../diagnostics/generated/focused_diagnosis_20261002/figures/figure05_inverse_saturation.pdf) | [Four-trail result](../../../diagnostics/generated/focused_diagnosis_20261002/inverse/attempt01_row50_fourtrail/result_note.md) |

Detailed local appendices are the selector role/case/ranking CSVs; Wind `compactnative_group_utility_summary.json` and `attempt03/cpu_post_audit.json`; Thermal `fixed_assessment/summary.json`, `t3_paired_metrics_summary.json` and `t3_per_module_peak_errors.json`; and inverse `result.json`, typed-link traces and per-timestep states under the indexed paths. They preserve exact query seeds, receiver IDs, per-channel units, controls and failures without uploading generated data.
