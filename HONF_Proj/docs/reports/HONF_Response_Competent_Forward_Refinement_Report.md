# HONF response-competent forward refinement: measured development report

Both separable and joint interfaces now adapt the same fine physical paths and field/port heads, learn from four existing heat-response families, and receive a positive heat-to-flow null penalty. Native architecture and once-wrapper organization remain. This combined refinement improves fields and responses over its own parents, but leaves major response/dependency misses. It does not isolate each new training ingredient.

**Predictor:** both refined models improve all five core fields over their parents and the retained reference means, but initial-port temperature deteriorates. **Organizer:** joint control has small local thermal-response utility and little advantage over the equally refined separable arm. **Inverse:** stored-pool rankings are correct, but wrong thermal-response directions and heat-to-flow leakage still block trustworthy continuous inverse use.

Both arms completed 500 new epochs and independently selected epoch 500 using the saved-cadence best-field criterion. Native fields, fit/development/null responses, final fixed-four replay, I-removal, stored-pool choices and complete-call costs are measured. Identical selected/endpoint field and response measurements were reused. The STOP500 decision and checkpoint selections were sealed before the final fixed-four replay; the conditional extension to 1,000 was not taken. Solver usage remains **326/326**, with **zero new attempts**. Formal3501/3502 and earlier G-fast/Tensor-H/receiver histories are preserved and were not restarted.

## Predictor: gains, misses, measured evidence and next step

All five main means and p90s improve over each own fit-100 parent. Add fluid/surface/material temperature means improve 29.43%/30.10%/27.37%; Joint improves 26.45%/27.78%/25.64%. Formal core degradation gates pass. Matched differences remain small and mixed: Add thermal and pressure field means are slightly better, while Joint u mean is slightly better. These are the same repeatedly exposed 22 validation cases, not independent test evidence.

| Equal-case native RMSE, matched 500 / selected 500 | H-add | H-joint | Dense-D25 retained | G-fast retained | Tensor-H retained |
| --- | --- | --- | --- | --- | --- |
| fluid/temperature | 0.770312 | 0.779114 | 0.949971 | 1.17451 | 0.940162 |
| surface_temperature | 0.788326 | 0.794336 | 0.974319 | 1.23966 | 1.0903 |
| material_temperature | 0.66308 | 0.669472 | 0.865606 | 1.04896 | 0.93089 |
| fluid/u | 0.0220367 | 0.0217359 | 0.0284574 | 0.0306896 | 0.0317718 |
| fluid/p | 0.0122219 | 0.0122465 | 0.0139807 | 0.0143447 | 0.017395 |

Dense-D25 uses padding_invariant_v2 inputs; native references use source_local_v3. Cohort/masks/denominators/normalization are audited, but architecture, input representation and training history differ. Retained references establish context rather than an identically trained causal comparison.

The full 24-role appendix retains an initial-port miss. Outside-temperature mean/p90 worsen 21.75%/12.96% for Add and 20.91%/15.51% for Joint. Every M stratum has a worse mean; M3 mean/p90/max reach +43.26%/+58.02%/+74.71% Add and +37.30%/+48.34%/+61.34% Joint. M10 has only four cases. These noncore warnings do not create retrospective formal gates.

Initial h-effective remains about 11.87 RMSE, slightly worse than parents. Final outside-temperature improves 25–27% but means 1.1274/1.1382 remain above retained Tensor 0.9532. Final h-effective means 0.8287/0.8276 remain above Tensor 0.7939 and Dense 0.7252. q-normal proxy means improve 10.7%/10.4% to 3.3520/3.3537; absolute proxy errors remain substantial even as their means beat retained references. Edge pressure-difference means/p90 improve, but Add maximum remains 4.28% worse than its parent. This edge diagnostic differs from the 8%-band pressure-response functional.

Predictor next step: retain both selected500 development references and carry the initial-port and per-M tail misses into the next comparison as explicit monitoring requirements.

Interim 100 review/continuation receipts remain preserved separately under report_inputs/refinement100; final all-24/per-M/p90/max/paired evidence is under refinement500.

## Organizer: small measured utility, limited added value

All 38 organizer and 80 authorized physical tensors changed; all 264 frozen tensors and global/local normalizers remain exact. DEV thermal responses improve over parents, while Joint's matched advantage is only 1.40% fluid and 1.30% material, with surface effectively tied. Final fixed-four Joint thermal errors are 0.45%/0.59%/0.63% below Add for fluid/surface/material. The current plan gives no numeric organizer-improvement threshold.

Removing I from the same Joint500 state increases DEV fluid/surface/material response error by 0.76%/0.22%/0.30%, and fixed-four error by 0.72%/0.21%/0.13%. The q-normal proxy improves on removal by 0.29% DEV and 0.054% fixed four. Small mixed effects support limited local utility rather than broad meaningful organizer benefit. Same-state intervention is distinct from independently co-adapted arm comparison.

The retained interface separates source-only S, receiver-only R and source-receiver joint I controls; P0/P1/P2 organization is shared once within each native wrapper, with fresh input plans at changed endpoints. There is no recursive Tree rebuild, shadow pass or forced K. Fine MM/ME/EM/QM/QE readers remain dense; global calibration, receiver centering, coarse/local context and upstream port coupling remain information paths. Learned memberships are latent controls, not physical causality or sparse executor proof. Full-call costs show overhead, not savings.

Organizer next step: retain the equally trained separable model as the control for any separately reviewed dependency intervention; current small joint effects do not justify an expanded organizer portfolio.

## Inverse use: response gains do not establish trustworthy constraints

Fixed-four thermal means improve over parents by 15.82%/13.70%/14.13% Add and 15.57%/14.47%/14.52% Joint. New fluid/surface/material errors are 0.037067/0.050119/0.049496 Add and 0.036901/0.049822/0.049184 Joint. They remain 32–51% above retained Tensor-H's 0.027871/0.033235/0.034309. Both 0291 fluid mean signs remain wrong: truth minus/plus is -0.028919/+0.028914; Add predicts +0.010420/-0.015508 and Joint +0.009844/-0.015100. These substantial sign misses do not have a numerical-floor explanation.

Broad all-22 null u/v/p leakage decreases only 12.80%/4.32%/8.34% Add and 11.33%/4.02%/8.10% Joint; omega worsens 10.19%/11.82%. All four channels fail the 90% target. No certified evaluation floor is declared. Relative accuracy to zero is undefined, so floor-limited success cannot be claimed. Mean 8%-band false pressure-change error improves to 7.01e-5/7.34e-5 from parent 1.92e-4/2.01e-4, but nonzero heat-to-flow dependence still prevents trustworthy constraint use.

Both new arms correctly select plus on 0291/0294/0687 and minus on secondary 0277 from the identical already solved finite pools, with zero realized native material-maximum regret. Primary best-versus-next reference gaps are 0.390348/0.235369/0.028889; secondary 0277 gap is 0.092199. Its fresh baseline is missing. Retained G-fast/Tensor already make the same zero-regret choices, so this does not establish an improved decision policy or continuous inverse readiness. No new candidate, inverse search, generator or solver call occurred.

Inverse next step: retain the finite replay as the measured decision-use result. Correct0291thermal directions and substantial heat-only flow leakage remain prerequisites for a future continuous inverse claim.

The sealed STOP500 decision used the completed field/DEV/null review before opening final fixed-four results. Budget could have supported another500, but Add DEV material-response error worsened100→500 by0.79%, Joint improved by3.31%, both arms retained the broadnull/omega misses and about21% initial-port temperature regression, and their independently trained response differences remained small. Falling TRAIN thermal loss alone did not establish a specific further transfer/dependency learning question. This is a scope-limited negative result, not evidence of exhausted capacity or a plateau.

## Response, null and complete-cost tables

These are equal-layout/equal-signed-comparison macro native-unit response RMSEs, separated by supervision status. Temperature units are the saved generator's dataset units, not certified SI temperatures. The fit and DEV cohorts each contain four independent layout directions and eight signed comparisons.

| Response cohort | Arm | Fluid T | Surface T | Material T |
| --- | --- | ---: | ---: | ---: |
| Supervised fit families | Add500 | 0.067994 | 0.073279 | 0.065464 |
| Supervised fit families | Joint500 | 0.067324 | 0.072759 | 0.065351 |
| Response-withheld DEV | Add500 | 0.096167 | 0.103982 | 0.104342 |
| Response-withheld DEV | Joint500 | 0.094818 | 0.103988 | 0.102984 |

DEV gains over own parents are 17.37%/10.18%/4.88% Add and 17.70%/10.45%/7.44% Joint. They measure transfer outside refinement response gradients, with historical exposure explicitly retained.

The reused final audit contains three primary layouts, three independent heat-transfer directions and six signed comparisons. These exposed cases were opened after the training-horizon/selection seal, not treated as an independent test.

| Fixed-four primary control | Fluid T response RMSE | Surface T | Material T |
| --- | ---: | ---: | ---: |
| Zero predicted change | 0.051185 | 0.097755 | 0.117894 |
| G-fast1000 retained | 0.043103 | 0.054331 | 0.053634 |
| Tensor-H1000 retained | 0.027871 | 0.033235 | 0.034309 |
| Add parent fit100 | 0.044036 | 0.058076 | 0.057642 |
| Joint parent fit100 | 0.043707 | 0.058247 | 0.057535 |
| Add refinement500 | 0.037067 | 0.050119 | 0.049496 |
| Joint refinement500 | 0.036901 | 0.049822 | 0.049184 |

The unchanged-own-heat receiver audit retains 32 primary module-direction rows across 16 distinct physical receiver IDs. Mean absolute module-peak response error falls from 0.039124 to 0.034477 Add and 0.039532 to 0.034378 Joint; retained Tensor-H is 0.023241. Both new arms beat zero predicted change on only 16/32 rows. The module IDs and each error remain in the [paired finite-response record](../../diagnostics/generated/response_refinement_20261005/report_inputs/final_counted_comparison.json).

The all-22 null panel uses identical parent/candidate 0.10/0.20 steps and Q256 sampled receivers. Training uses 0.05/0.10 steps instead. All22 actual increments/fractions were verified; only the detailed fixed-four files independently preserve query identities, while the other eighteen use the common deterministic sampler declaration. This sampled panel does not certify full-grid invariance. The reference changes are exactly zero for u/v/p/omega under this case-owned heat-only generator; relative error to zero is undefined.

| Broad22 null channel | Add500 native RMS | Joint500 native RMS | Add / Joint reduction from parent |
| --- | ---: | ---: | ---: |
| u | 0.00116055 | 0.00114919 | 12.80% / 11.33% |
| v | 0.000109962 | 0.000110043 | 4.32% / 4.02% |
| p | 0.000475683 | 0.000471977 | 8.34% / 8.10% |
| omega | 0.00403251 | 0.00403729 | -10.19% / -11.82% |

No channel reaches 90% reduction. DEV null reductions are u/v/p/omega 14.26%/10.00%/15.78%/-0.71% Add and 9.95%/8.63%/14.55%/-0.23% Joint, also misses. The [raw review](../../diagnostics/generated/response_refinement_20261005/report_inputs/paired_refinement500_review_primary.json) retains native values and matched steps.

The six final primary transfers additionally retain all-channel native and common TRAIN-standard-deviation scaled null errors:

| Final primary null | Add native / TRAIN-scaled RMSE | Joint native / TRAIN-scaled RMSE |
| --- | ---: | ---: |
| u | 0.00100597 / 0.00249014 | 0.0010132 / 0.00250803 |
| v | 0.000108649 / 0.00236682 | 0.000109916 / 0.00239441 |
| p | 0.000420484 / 0.0027623 | 0.000422518 / 0.00277567 |
| omega | 0.00442905 / 0.00434686 | 0.00445438 / 0.00437172 |

The shared selected-TRAIN standard deviations are u=0.403982699, v=0.0459052473, p=0.152221978, omega=1.01890945; scaling is not relative accuracy to the zero reference. The 8%-band functional is mean pressure over valid native fluid rows with x<=0.08Lx minus mean pressure over x>=0.92Lx; its response is trial minus baseline. It differs from the port edge-pressure diagnostic. For this functional, mean absolute false response is about 7.01e-5 Add / 7.34e-5 Joint, or 4.61e-4 / 4.82e-4 of the common pressure standard deviation. Maximum absolute false responses are 1.67661e-4 / 1.73050e-4 native, about 0.00110 / 0.00114 TRAIN-scaled. True pressure response is zero. No false-feasible-rate calibration is possible in a pool with no genuinely pressure-changing design.

Complete inference / whole physical heat-and-query input forward+VJP medians, seconds, measured on the same GPU1:

| Actual native panel | G-fast | Add500 | Joint500 |
| --- | ---: | ---: | ---: |
| B8 Q1024 actual M1 | 0.111585 / 0.275633 | 0.136412 / 0.306486 | 0.134994 / 0.303119 |
| B8 Q1024 actual M12 | 0.202115 / 0.419485 | 0.247189 / 0.527874 | 0.246580 / 0.523938 |
| B1 Q8192 actual M1 | 0.183535 / 0.404913 | 0.309201 / 0.760985 | 0.307964 / 0.746206 |
| B1 Q8192 actual M12 | 0.230673 / 0.523653 | 0.375661 / 0.927249 | 0.375005 / 0.918953 |

The B8 primary overhead target <=1.5x passes. Full-grid B1 costs exceed 1.5x; there is no universal overhead pass or speedup. These TRAIN cost panels have actual M1/M12, not padded M12 standing for validation M10. Five warmed alternating inference and two input-VJP repetitions are feasibility measurements, not a latency distribution or new AD/FD certification. The heat leaf feeds both normalized global and physical local heat; queries are physical coordinates. Module-position VJP and backward physical-row counts were not measured.

Peak allocated inference memory at B8/M12 is 612.29 MiB G / 613.81 MiB Add / 613.81 MiB Joint. Whole input-VJP peaks are 5,081.24 / 5,988.60 / 5,991.96 MiB, with 4,997.07 / 5,902.93 / 5,906.29 MiB extra above the common resident model/input baseline. All three models and inputs share that baseline. These values are allocator measurements, not standalone-process memory or driver busy time. Complete physical-phase timing/work and every low/high-M condition remain in the [cost record](../../diagnostics/generated/response_refinement_20261005/evaluation/selected_complete_native_cost/cost.json).

## Inspected figure index

Five selected PDF masters retain twelve page companions for direct Markdown display. All twelve PNG pages were visually inspected by root, including the revised fidelity-versus-work axis; saved numerical sources, native masks, clipping (zero), hashes and figure limits are in [the figure index](../../diagnostics/generated/response_refinement_20261005/figures/FIGURE_INDEX.md) and [root inspection receipt](../../diagnostics/generated/response_refinement_20261005/engineering/root_final_figure_visual_inspection.json). Superseded local preflight visual exports are removed after selected-master validation; raw science, training-cadence plots, histories and render-source receipts are preserved.

1. [Learning and complete cost, one page](../../diagnostics/generated/response_refinement_20261005/figures/01_learning_complete_cost.pdf).
2. [Physical fields and residuals, four pages](../../diagnostics/generated/response_refinement_20261005/figures/02_native_fields.pdf).
3. [Finite responses and controls, five pages](../../diagnostics/generated/response_refinement_20261005/figures/03_finite_responses.pdf).
4. [Actual organizer and physical utility, one page](../../diagnostics/generated/response_refinement_20261005/figures/04_organizer_response_utility.pdf).
5. [Stored heat choices and regret, one page](../../diagnostics/generated/response_refinement_20261005/figures/05_stored_heat_choices.pdf).

![Learning, normalized validation fidelity versus measured epoch work and complete native cost](../../diagnostics/generated/response_refinement_20261005/figures/01_learning_complete_cost.png)

Figure 1. G-fast backbone1000 → interface prefit100 → refinement0–500. Both exposed22 normalized Q1024 monitoring curves improve; cumulative train+validation work is 1.552963/1.552367 hours Add/Joint and averages 11.1813/11.1770 seconds per complete epoch. This work axis excludes startup, standalone evaluation and inherited prefit. The lower panels retain B8 actual M1/M12 costs and B1 full-grid warnings (inference 1.63–1.68x, input VJP 1.75–1.88x G-fast). This supports affordable refinement and measured overhead, not sparse execution or independent-test fidelity.

![Native fluid temperature references, selected500 predictions and residuals for the fixed four representatives](../../diagnostics/generated/response_refinement_20261005/figures/02_native_fields.png)

Figure 2a. Complete native fluid-temperature fields for exposed validation0277/0291/0294/0687, selected Add/Joint500; equal-case all22 RMSE is 0.770312/0.779114 in dataset temperature units. Native coordinates and common per-case field/residual scales are preserved with no clipping. The nominal packed-H5 baseline includes0277; it is not a substitute for its missing fresh counted-response baseline. The fields support improved absolute predictions, not correct finite heat directions.

![Native material temperature and residuals on physical module disks](../../diagnostics/generated/response_refinement_20261005/figures/02_native_fields_material_temperature.png)

Figure 2b. The same fixed representatives use actual material-grid coordinates and active-module IDs; all22 material-temperature RMSE is 0.663080/0.669472 Add/Joint. Native disk geometry and per-case scales are preserved. Better mean fields do not certify maximum-temperature constraints.

![Native surface temperature traces and residuals for every active representative module](../../diagnostics/generated/response_refinement_20261005/figures/02_native_fields_surface_temperature.png)

Figure 2c. All active modules use the saved64 angular surface rows with native angle/normal target joins, no synthetic endpoints; display offsets are presentation only. All22 surface-temperature RMSE is 0.788326/0.794336 Add/Joint. The trace detail exposes local residuals that a pooled scalar could conceal; h-effective masks are not substituted for the temperature mask.

![Native u and pressure reference, selected500 predictions and residuals](../../diagnostics/generated/response_refinement_20261005/figures/02_native_fields_velocity_pressure.png)

Figure 2d. Complete native u/p maps on the same representatives; all22 means are u=0.022037/0.021736 and p=0.012222/0.012246 Add/Joint in dataset velocity/pressure units. These lower absolute errors coexist with false heat-only flow changes; field accuracy does not prove dependency invariance.

![Reference and predicted spatial finite thermal responses on the primary audit and secondary0277 span](../../diagnostics/generated/response_refinement_20261005/figures/03_finite_responses.png)

Figure 3a. Saved analytic-wake/shared-grid thermal responses on three primary layouts and the separately labelled0277 minus-to-plus span. Primary fluid response RMSE is 0.037067/0.036901 Add/Joint; complete native coordinates, design increments, residuals and dataset units are shown. The secondary span cannot recover the failed0277 fresh baseline. These are reused exposed benchmark responses, not CFD or an independent test.

![Both0291 heat directions, spatial response residuals and wrong signed fluid means](../../diagnostics/generated/response_refinement_20261005/figures/03_finite_responses_0291_both_signs.png)

Figure 3b. Reference mean fluid-temperature changes are -0.028919/+0.028914 for minus/plus. Add predicts +0.010420/-0.015508; Joint +0.009844/-0.015100. Both directions remain wrong despite smaller response RMSE. No post-hoc sign threshold or certified grid-error floor is used to turn these misses into a pass.

![Physical module identities and response errors where each receiver own heat is unchanged](../../diagnostics/generated/response_refinement_20261005/figures/03_finite_responses_unchanged_heat_receivers.png)

Figure 3c. Thirty-two primary module-direction rows across sixteen distinct physical receiver IDs. Mean absolute module-peak response error is 0.034477 Add / 0.034378 Joint versus 0.023241 retained Tensor-H; both new arms beat zero change on16/32 rows. Responses at unchanged-own-heat modules demonstrate cross-source dependence, with substantial prediction misses retained.

![Response-withheld development families with native thermal responses and residuals](../../diagnostics/generated/response_refinement_20261005/figures/03_finite_responses_development_spatial.png)

Figure 3d. DEV0304/0320/0335/0350 contributes four independent directions/eight signed comparisons outside refinement response gradients, scales and calibration. Mean fluid/surface/material response errors are 0.096167/0.103982/0.104342 Add and 0.094818/0.103988/0.102984 Joint. These improvements demonstrate limited transfer in this refinement, with historical research exposure disclosed.

![Fixed audit thermal response errors against zero-change, matched parents and retained controls](../../diagnostics/generated/response_refinement_20261005/figures/03_finite_responses_thermal_controls.png)

Figure 3e. The six primary signed comparisons use identical saved references/queries and selected500 states. Joint improves its parents by15.57%/14.47%/14.52% fluid/surface/material, but stays32.40%/49.91%/43.36% worse than Tensor-H. The control comparison prevents parent gains from standing in for the stronger response target.

![Actual counted0291 source memberships, limited receiver access, signed joint control and endpoint response utility](../../diagnostics/generated/response_refinement_20261005/figures/04_organizer_response_utility.png)

Figure 4. Actual selected Joint500 counted0291 baseline, input-selected P0 group2, five physical modules/192 environment donors. Nonnegative module/environment donor integrals are1.000000/0.99999994; signed latent I is shown separately from membership and access. The first128 P2 QM receivers are a bottom-boundary subset; complete P0/P1/P2 plans are retained but later receiver chunks and MM/ME/EM accesses are not drawn. On0291 plus, normal fluid response RMSE0.078612 becomes0.078849 without I (0.30% worse), while both signed means remain negative against truth+0.028914. This establishes small local response utility with dense/global/coarse/local/upstream paths retained, not physical causality.

![Actual stored heat allocations, finite candidate rankings and native material maximum regret](../../diagnostics/generated/response_refinement_20261005/figures/05_stored_heat_choices.png)

Figure 5. Physical candidate layouts/heat assignments and evaluated-state arrows for the three primary pools0291/0294/0687. The secondary two-point0277 choice is measured separately and is not drawn on this page. Both arms choose plus in all three primary pools and minus for0277, with zero realized regret; primary reference best-to-next gaps are0.390348/0.235369/0.028889 dataset temperature units. Numerical warnings are about1.3e-4, not certified grid bounds. Retained controls already make the same choices, and biased predicted absolute maxima remain visible. There is no generative design trail, new solve, continuous inverse or calibrated feasibility claim.

## Methods and scientific identity

H-add Run 3801 and H-joint Run 3802 each inherit their own exact interface-fit 100 checkpoint on the shared physical 1000 G-fast history. Their trained interface values differ at the new starting point; inherited physical values and prior exposure match. Both begin with fresh AdamW moments at refinement 0. The inherited backbone history is 1,000 epochs plus 100 interface prefit epochs; the physical objective age used by the trainer is 1,100 plus genuine refinement age, while sampling and selection use the new age. Explicit soft admission is independent of age and preserves learned input-dependent memberships, fresh plans per input, and once-wrapper phase sharing. Historical curriculum and frozen-interface-fit guards remain intact.

The primary fixed25_v1 manifest is unchanged: 150 primary TRAIN cases and 22 exposed validation cases, semantic fingerprint 933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044. Both arms retain Q=1024, microbatch 8 / effective batch 48, FP32, identical seeds/case-query streams, native loss denominators, and selected-TRAIN normalization. All available validation strata are reported: six cases each at M3/M5/M7 and four at M10.

The response addendum uses original TRAIN 0001/0318/0333/0348, their saved converged baseline and opposite heat_transfer_minus/plus states. Only 0348 is already primary TRAIN; this is 150 primary cases plus three extra auxiliary families. TRAIN 0001 and TEST 0273 have duplicate physical inputs, and 0273 remains excluded from primary validation. Response-development 0304/0320/0335/0350 is outside the current 150/22 and withheld from refinement gradients, response scales, and coefficient calibration, but its historical exposure is disclosed. These are response-withheld families for this refinement, not untouched research data. Fixed 0277/0291/0294/0687 labels are evaluation only.

Each fit and response-development cohort contains four layouts, four independent heat-transfer directions, and eight signed baseline-relative comparisons. The fixed-four primary panel contains three layouts, three independent directions, and six signed comparisons; the 0277 minus-to-plus span is secondary. Final review metadata makes that distinction explicit without changing numerical measurements. The original 100-epoch receipt is preserved rather than rewritten.

The policy trains 38 organizer tensors/120557 scalars at 1e-4 and 80 fine-physical/field/initial-port/refinement-port tensors/2143117 scalars at 1e-5, with weight decay 1e-5. The 264 frozen tensors/3563043 scalars include input encoders, native coarse/local context, inherited global organizer/calibration, and frozen Stage-A. Normalization is restored before attachment and stays byte-identical. Stage-A parameters stay frozen while its differentiable computation remains in the native path. Exact semantic names are listed in the attachment/optimizer receipts; no broad backend unfreeze or target-port substitution is used.

Both arms use the inherited primary reconstruction objective, auxiliary absolute supervision of both response states with fixed weight 0.25, equal-role/family/direction positive thermal-change supervision, and a q-normal proxy change term at 0.25. Response RMS scales 0.2871017747 / 0.4400275852 / 0.5540046111 for fluid/surface/material T and 0.9177381694 for q-proxy use only fit labels. Floors combine a saved-FP32-spacing indicator and 1e-3 of primary TRAIN standard deviation; neither certifies physical discretization error. Saved FP32 outputs are widened before subtraction, not re-solved in FP64.

Shared response/null coefficients 0.01756907360776242 / 0.2556857054188304 were fixed from pooled initial training-only gradient measurements spanning four families in both arms, targeting initial25%/10% of the primary gradient norm. These are initial pooled budgets, not ongoing component-ratio guarantees. Gradient clipping stays at norm1.0. The15scheduled first-update diagnostics per arm all record clip scale1.0 (largest sampled preclip norm0.38097 Add /0.36815 Joint), but auxiliary callbacks share the fourth update boundary; those samples do not measure callback-boundary clipping or certify all2,000updates. No ongoing component-gradient balance was logged. See [saved gradient diagnostics](../../diagnostics/generated/response_refinement_20261005/engineering/final_saved_gradient_diagnostics.json). The successful native calibration is attempt 2; the failed first attempt is preserved. No response-development / fixed-four coefficient search occurred. Both differentiable null calls use one deterministic primary TRAIN input per epoch and alternate feasible 0.05 / 0.10 fractions. M >= 2 uses balanced transfer; M1 checks genuine heat-only null dependence without a fixed-sum claim. No trial thermal label is supplied. The capability is confined to the inspected analytic-wake ThermalChannel generator.

The graph provenance audit separates the nominal all-22 H5 baseline call from the counted finite-response baseline. At 0291, the inspected FP32 heat/context inputs, active module coordinates, complete native fluid coordinates, and displayed first 128 P2 QM receiver rows match exactly. That does not establish complete P0 plan identity: the nominal call uses a separate adapter receiver catalogue of 1,036 anchors, while the counted call has a different catalogue. The nominal graph cannot be substituted as the exact counted-response plan. See engineering/graph_nominal_counted_input_provenance_e100.json.

One bounded graph capture was approved within the already planned selected Joint 0291 counted baseline wrapper. This is an evaluation-only metadata/export amendment: it adds no model forward, optimizer update, solver attempt, training-data exposure, or new scientific comparison. It leaves parameters, objectives, coefficients, data memberships, input plans, and native numerical computation unchanged. The graph was captured on the actual selected Joint 0291 counted baseline. Its checkpoint SHA and native input/query identity are recorded in evaluation/h-joint_selected_counted/0291/phase_graphs.provenance.json. Final graph rendering is complete and its current-byte root visual inspection passes; nominal-versus-counted provenance limits are retained.

## Exposure, measured cost and budget

Each arm completes 500 genuine epochs: 75,000 primary case visits, 2,000 optimizer updates, 76,800,000 primary fluid queries, 1,000 response wrappers and 1,000 null wrappers. Every epoch visits all 150 cases and has four boundaries. Callback streams are identical. Null inputs cover three 150-case cycles plus 50 cases, with 250 uses of each 0.05/0.10 fraction, 80 M1 visits and zero negligible-step skips. Each fit family receives 63 minus and 62 plus visits. Source: engineering/completed500_exposure_cost_late_windows.json.

Train-plus-validation totals are 5,590.668 s Add and 5,588.520 s Joint, averaging 11.1813/11.1770 s per epoch. Auxiliary timings are nested, not additive. The response/null callbacks account for228.33/144.33s Add and231.55/147.63s Joint within training. Actual physical gradients and extra differentiable endpoint calls explain why this scope differs from the earlier frozen-interface objective. Against the prior10.25s mean as historical context, the current11.18s is about9% higher; this is not a matched attribution of individual overheads. Frozen parameters reduce optimizer work but still propagate activations/adjoints for the upstream interface. Two initial failed trainers each used 150 cases/four updates before CSV persistence failed; they restarted from refinement 0 after the CSV-only repair. Lost work is charged separately and is not a recovered epoch.

The final same-device FP32 cost benchmark completes 108 wrappers and 24 input VJPs with model state unchanged, five alternating warmed inference repetitions and two VJP repetitions per condition. B8/Q1024 inference overhead relative to G-fast is 20.98–22.30%; forward-plus-input-VJP overhead is 9.97–25.84%. B1/Q8192 inference overhead is 62.57–68.47% and VJP overhead 75.49–87.94%. Dense physical input-row work remains. CUDA extra allocation subtracts the common resident model/input baseline; it is not isolated process memory or hardware kernel counts. Source: evaluation/selected_complete_native_cost/{cost,receipt}.json.

The hard envelope is 12 aggregate GPU-associated process-hours and 8 elapsed hours, including failed starts and final work. Isolated synchronized latency differs from process-associated wall. Current accounting closes all 34 unique GPU process envelopes (31 completed, three failed) at 3.391849310 associated hours, with observed elapsed 2.407414841 hours. These measured values lie below 12/8 ceilings. Delivery elapsed continues beyond that observation; stale forecasts are not final totals. The physical-reference ledger remains 326/326 with zero new solver attempts.

## Numerical and software qualifications

Own-parent initialization replay passes the original atol=rtol=2e-5 on sampled M3/M10 native outputs, and all initial inherited state tensors/normalizers are byte exact. Sampled function compatibility is not a universal bitwise-forward claim. The native resumed next update is numerically within those unchanged tolerances but not bitwise identical: maximum parameter differences are 4.470348358154297e-8 for Add and 3.725290298461914e-8 for Joint. The authoritative native_update_and_resume.json explicitly sets resume_all_state_bitwise=false.

The preserved calibrate_native_attempt2.py harness (SHA256 b5a67058a83275f4f678dc62e366290fdd4af6cd57962a216a6fcc2fa3ec0afd) had an incorrect literal resume_all_state_bitwise=True in its output construction. native_resume_metadata_correction.json records that defect and supersedes it with measured deltas and the corrected receipt hash 649ed8806614e18e5f1b893af09c7dc9a14159dfa998553bb75f43db796c8e07. The original harness was neither edited nor rerun to conceal the metadata error; this correction made no new model/update/solver call. The separate CPU AdamW/dropout/accumulation stop-save-resume regression is bitwise exact and cannot turn the native GPU result into a bitwise pass.

The independent 100-epoch CPU checkpoint audit passes all paired dataset/loss/case/model-except-arm/initial-RNG/calibration/addendum/inventory/query-update settings. Both optimizer groups retain the declared rates/weight decay, all 118 authorized tensors changed, and no forbidden frozen state or normalization changed. No selection runtime buffers are present in these actual state dictionaries. The independent 500-epoch audit also passes: both actual payload ages are 500; all 118 authorized tensors changed; all 264 frozen tensors, global/local normalizers, and Stage-A provenance remain exact; each optimizer has 118 states in the declared 38/80 groups and all 11 paired identity checks pass. This is state/lineage evidence, not a field, response, organizer, or inverse verdict. The strict selected-state audit also passes. Both aliases select genuine age 500 by minimum saved-cadence field criterion and match their retained milestones scientifically, including optimizer/RNG/normalizers. Serialized container hashes differ; equivalence is exact scientific-state equivalence rather than file-byte equality. Root sealed horizon and selections before candidate fixed-four reads. See checkpoint_pair_identity_selected_after500.json and report_inputs/final_horizon_and_selection_seal.json. Ordinary resume preserves exact declaration/content hashes and strict checkpoint trust; passing tests/state audits certify those software contracts rather than physical-response truth.

The maintained-source CPU regression receipt records 94 passing tests in 7.10 s at source commit 6bac45fb7b9bf4286524a18b3ecdb5a7dc7a8529. Coverage includes policy separation, frozen/interface/query contracts, stop, preparation, evaluation, cost, campaign response, and the native response objective. This is software verification, not final 500-epoch measurement or physical certification. See engineering/final_cpu_regression_receipt.json.

Retain the inherited five strict high-M generic-Global/G-fast input-VJP misses and older 0687 three-value q-proxy zero-correction interface miss at their original tolerances. New own-parent checks or final VJP cost do not repair those historical comparisons. Native units are generator scales rather than certified SI/CFD references; q-normal remains a proxy, heat sums are input constraints rather than demonstrated energy balances, and pressure-flow independence is generator-specific.


Final fixed-four comparison joins seven summaries, with all 12 global normalizers byte exact and eight paired query/reference/delta identity checks. Retained scalar solid reductions differed at up to 8.33e-17 despite identical arrays. All seven summaries were recomputed in memory from exact FP64 endpoint deltas with one C-order finite_metrics recipe. Original summaries/arrays remain untouched; scalar reconciliation is logged and no model tolerance changed. DEV I-removal uses the same guarded recipe. See final_counted_comparison.json and final_development_I_utility.json.
## Detailed appendix and reproducibility

Scientific measurements and unmet targets are recorded separately from software and upload checks. Local artifacts resolve through an ignored symlink to `/data/wanglz/ModularDT/thermal_development/response_refinement_20261005`; the Git report intentionally depends on those preserved local scientific artifacts. Generated figures, arrays, checkpoints, profiler captures, renderer scripts and the symlink are not uploaded. A fresh clone requires restoring the same local artifact root to display figures.

| Appendix | Authoritative local evidence |
| --- | --- |
| A: lineage, exact trainable/frozen policy | [Preparation pair audit](../../diagnostics/generated/response_refinement_20261005/engineering/preparation_pair_audit.json), [strict500 state audit](../../diagnostics/generated/response_refinement_20261005/engineering/checkpoint_pair_identity_refinement500.json), [selected-state audit](../../diagnostics/generated/response_refinement_20261005/engineering/checkpoint_pair_identity_selected_after500.json) |
| A: exposure and selection | [All500 exposure/cost/windows](../../diagnostics/generated/response_refinement_20261005/engineering/completed500_exposure_cost_late_windows.json), [STOP500 decision](../../diagnostics/generated/response_refinement_20261005/report_inputs/refinement500_horizon_decision.json), [pre-final selection seal](../../diagnostics/generated/response_refinement_20261005/report_inputs/final_horizon_and_selection_seal.json) |
| B: all24 roles and tails | [Role mean/p90/max CSV](../../diagnostics/generated/response_refinement_20261005/report_inputs/refinement500/all24_role_statistics.csv), [per-M CSV](../../diagnostics/generated/response_refinement_20261005/report_inputs/refinement500/all24_by_module_count.csv), [paired-case CSV](../../diagnostics/generated/response_refinement_20261005/report_inputs/refinement500/paired_case_all24_rmse.csv), [findings](../../diagnostics/generated/response_refinement_20261005/report_inputs/refinement500/compact_all24_findings.md) |
| C: fit/DEV responses and all22 null | [Paired500 review](../../diagnostics/generated/response_refinement_20261005/report_inputs/paired_refinement500_review_primary.json), [detailed verified review](../../diagnostics/generated/response_refinement_20261005/report_inputs/paired_refinement500_development_details_verified_primary.json) |
| D: final response, all-channel/scaled null, pressure band, own-heat-unchanged receiver IDs | [Guarded finite comparison](../../diagnostics/generated/response_refinement_20261005/report_inputs/final_counted_comparison.json); original per-arm arrays and summaries under evaluation/ remain unchanged |
| E: independent arm and I-removal utility | [DEV I utility](../../diagnostics/generated/response_refinement_20261005/report_inputs/final_development_I_utility.json), [counted graph provenance](../../diagnostics/generated/response_refinement_20261005/evaluation/h-joint_selected_counted/0291/phase_graphs.provenance.json), [graph NPZ](../../diagnostics/generated/response_refinement_20261005/evaluation/h-joint_selected_counted/0291/phase_graphs.npz) |
| F: latency, allocation and execution work | [Complete cost JSON](../../diagnostics/generated/response_refinement_20261005/evaluation/selected_complete_native_cost/cost.json), [cost receipt](../../diagnostics/generated/response_refinement_20261005/evaluation/selected_complete_native_cost/receipt.json), [actual M1/M12 identity](../../diagnostics/generated/response_refinement_20261005/engineering/root_cost_actual_module_counts.json), [whole-round accounting](../../diagnostics/generated/response_refinement_20261005/accounting/final_closeout.json) |
| G: finite choice gap/regret | [Stored-pool decisions and native maxima](../../diagnostics/generated/response_refinement_20261005/report_inputs/final_counted_comparison.json);0277 missing baseline remains explicit |
| H: software and numerical qualifications | [94-test receipt](../../diagnostics/generated/response_refinement_20261005/engineering/final_cpu_regression_receipt.json), [native update/resume](../../diagnostics/generated/response_refinement_20261005/engineering/native_update_and_resume.json), [metadata correction](../../diagnostics/generated/response_refinement_20261005/engineering/native_resume_metadata_correction.json) |
| H: figure, completion and upload proof | [Root visual inspection](../../diagnostics/generated/response_refinement_20261005/engineering/root_final_figure_visual_inspection.json), [compact measured results](../../diagnostics/generated/response_refinement_20261005/report_inputs/final_measured_results.json), [completion checklist](../../diagnostics/generated/response_refinement_20261005/report_inputs/plan_completion_checklist.json), [delivery audit](../../diagnostics/generated/response_refinement_20261005/engineering/final_delivery_audit.json) |

Exact run directories and canonical checkpoint hashes are in [prepared_response_refinements.json](../../diagnostics/generated/response_refinement_20261005/prepared_response_refinements.json) and the strict state receipts. Add500 SHA256 is `13e2c8bd7c32a8a939b4d18b065300c04c73854412c203efabd728b5cd93006f`; Joint500 is `88e1204e80b4fa939bb6f7f3957ade6f52f3efe1f01d0da762a8257fa34a1c39`. Best-field aliases preserve exact scientific state but have different serialized-container hashes. Each run retains age0,100,200,300,400,500 plus latest/best-field aliases, with no intermediate25-epoch snapshots or extra selector copies.

The maintained [workflow guide](../guides/Thermal_Response_Refinement.md) supplies the actual preparation and100→500 resume commands; [separable](../../src/config_core/forward/thermal_response_refinement/h-add.json) and [joint](../../src/config_core/forward/thermal_response_refinement/h-joint.json) templates are unbound declarations, not direct training configurations. The commands document completed histories and do not authorize duplicate or formal launches. Historical receiver and Lean reports remain separate.

Implementation amendments are limited and declared: explicit opt-in co-adaptation policy/age and shared train-only calibration, native response/null callbacks, fixed CSV scalar persistence after the preserved failed start, stricter atlas/query/step guards, saved-response comparison, and evaluation-only counted graph export. The later evaluation/export changes were not imported or hot-edited into the running trainer. Numerical scalar reconciliation affects only ignored CPU summaries computed in memory, not training, original arrays or tolerances. No outcome-driven objective/coefficient/backbone change occurred inside the comparison.

## A/B/C verdict and one next investment

**A — Added value over matched separable refinement.** Both predictors improve; joint advantages are small, mixed and insufficient to establish broad meaningful benefit. I-removal confirms small thermal-response effects rather than a decisive advantage.

**B — Interpretation matches executed paths.** Physical readers remain dense with global/coarse/local/upstream paths. Full-call overhead is measured. Actual counted graph metadata does not establish causal physics or executor savings.

**C — Behavior on response-withheld families.** DEV thermal responses improve with historical exposure disclosed, but Tensor-H remains stronger on fixed-four thermal errors; both 0291 fluid signs remain wrong and the 90% null target fails. Opposite signs describe one direction per layout rather than a complete fixed-sum Jacobian. Correct finite-pool choices do not establish continuous inverse readiness.

**One recommended next investment:** separately review case-owned heat/flow dependency separation, beginning with an audit of the ThermalChannel generator assumptions and exact heat/flow information paths. It does not establish exhausted capacity or a learning plateau: training thermal loss still falls. No architecture hot edit, stronger penalty, formal full-data or 5,000-epoch launch is recommended automatically.
