# HONF Tree-Lite execution and maturation

Tree now projects its collective controls before expanding them over receivers, skips unrequested inference diagnostics, and trains its organizer with local soft reductions around one hard physical pass. The completed fixed-quarter campaign trained Tree and Pair through 1,000 epochs and a fresh full-access Fine control through 500. The unchanged field-error selector chooses epoch 900 for both Tree and Pair. All final organizer, response and inverse measurements use those selected weights.

Tree acts as a directory for receivers: candidate regions specify source weights and collective controls, while the fine kernels still evaluate individual physical sources. The cheaper implementation preserves the tested initial hard operator and physical derivatives within numerical tolerances. Its organizer training derivative deliberately changes. Selected-weight atom invariance has eight physical-output failures, so this campaign does not establish complete numerical faithfulness.

| Result | Gains | Misses and limits | Evidence | Next step |
|---|---|---|---|---|
| Predictor | Tree900 lowers mean fluid/material/interface temperature RMSE versus Pair900 by 17.62%/6.41%/5.95%. | Velocity and vorticity are worse; material median and high-M material/interface comparisons miss. Dense remains stronger on all eight core means. | All 22 cases, all 24 roles, paired/stratified statistics and Figures 1–2. | Study the flow/thermal tradeoff and high-M misses before choosing another experiment. |
| Organizer | The measured high-M wrapper is 54.20% faster; a training boundary uses 40.63% less allocated CUDA memory. Actual conditional control exclusions and control reliance are measured. | Tree is still 2.25 times slower than Pair in that wrapper benchmark. Fine rows are not pruned. Eight whole-rebuild physical failures and one hook-restoration guard failure remain. A geometry-matched action reference matches utility closely on the fixed four. | Same-weight native proof, all 22 support/work statistics, fixed-four interventions and Figures 3–4. | Localize the interface numerical failures and failed guard; test unique grouping value separately from control reliance. |
| Inverse | All 32 bounded trails are measured. Pair improves observed error in 8/8; Tree has a lower absolute mean final held error in these tasks. | Initial errors differ; Pair fits observations and hidden heats better. Graph blocks show no clear advantage. Tree has 320 charged topology-invalid forwards. No held nonzero physical response truth or independently validated designs exist. | Null22, isolated force probes, one stored training atlas and Figures 5–6. | Address conditioning and response training, then request independent physical validation if warranted. |

## Comparison identity and checkpoint selection

Every scientific arm uses fixed25_v1: exactly 150 training and 22 exposed validation cases, train-only global normalization, original source train/test separation, initialization/query seed 0, FP32 physical kernels, Q1024 primary training, effective batch 48 and the absolute 1,000-epoch schedule. Validation strata are M3/M5/M7/M10 with 6/6/6/4 cases. Detailed exports use only 0277/0291/0294/0687. The frozen Stage-A model retains its own pretrained normalizer and checkpoint.

The manifest's semantic identity is `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`; its literal JSON-file SHA256 is `3a8adc105637dc20ea4d1eaf0e5640e45fc7db8dffd6d5f15a8a5d5f0af16e32`. These identify different objects. One accepted epoch visits 150 cases, uses 19 microbatches and four ordinary optimizer boundaries, and supplies 153,600 primary fluid queries. Atlas, null, shadow, diagnostics and discarded attempts are additional work.

| Model | Run and lineage | Trainable parameters | Important difference |
|---|---|---:|---|
| Tree-Lite | Run3302, exact Tree3204 e200 state; local_context_shadow_v1 begins e201 | 3,336,120 | Learned receiver rules and collective controls; structural objective retained |
| Pair-F | Run3202, exact e200 continuation | 3,207,565 | Independently trained direct pair modulation; same source-local three-term backbone |
| Fine-F | Fresh Run3301, common 249 materialized B-fine initialization tensors | 2,928,273 | Same source-local three-term physical backbone with full access and no modulation head |
| Dense-D25 | Read-only Run3101, already trained to 1,000 | 4,395,409 | Native initializer, coarse/local context, global-heat features, case-relative heat scale, no null penalty |

All models contain 1,035,139 common frozen scalars. Fine is a backbone-sufficiency control rather than an exact capacity match. Dense500 is an equal-age strong reference with the disclosed architecture, input and objective differences. Dense1000 is an existing reference at the completed 1,000-epoch horizon; Dense500 supplies the equal-age 500 comparison. Mature full-data Run1804 remains historical and is never relabelled as quarter-data evidence.

The selector minimizes the common sampled all-22 native field MSE over saved 100-epoch checkpoints. These are train-scaled multichannel MSEs, distinct from the dataset-unit physical RMSEs below. No unsaved epoch or temperature-specific checkpoint replaces this rule. Tree1000 and Pair1000 completed with exit 0; the retained selected ages are 900/900. Fine selects 500. Latest, best-field and declared milestones are preserved; mature Run1804 is intact.

| Candidate | Selected age / sampled field MSE | Endpoint age / sampled field MSE | Literal selected file |
|---|---:|---:|---|
| Tree3302 | 900 / 0.03972594 | 1000 / 0.04013288 | `epoch_0900_model.pt`, SHA256 `6e68919f4390cb87da844be20b19fdef75ff8c5fbd003375521cface18725692` |
| Pair3202 | 900 / 0.03698975 | 1000 / 0.04893762 | `epoch_0900_model.pt`, SHA256 `4c022fa760908cb5bb886c839fd17e56ab44fb4ac5db64b669e228162b842c74` |
| Fine3301 | 500 / 0.07364724 | 500 / 0.07364724 | `epoch_0500_model.pt`, SHA256 `e54dff1fb472fa5e32e09d30509daf1836abc4cf9e53fbacb88d7179b9fb0214` |

Tree's 318 and Pair's 269 saved model-state entries match their best-field aliases bitwise. Alias serialization hashes differ. Root independently checked Tree's ten eligible milestones, dataset fingerprint, both normalizers and model configuration. An endpoint graph or null panel cannot stand for a selected checkpoint of another age. The actual selected900 normal22 panels are reused; matching detail4 and deep tests are separate measurements. [Tree binding audit](/data/wanglz/ModularDT/thermal_development/tree_lite_20261004/root_final_tree_selected_checkpoint_binding.json), [Pair binding audit](/data/wanglz/ModularDT/thermal_development/tree_lite_20261004/pair_final_selected_checkpoint_binding.json).

![Completed learning curves, retained physical checkpoints, equal-age tradeoffs and charged allocation](../../diagnostics/generated/tree_lite_20261004/figures/01_learning.png)

**Figure 1.** Curves show full inherited Tree/Pair lineages through 1000, Fine through 500 and the existing Dense1000 reference; dots are retained 100-epoch checkpoints, with selected stars at 900/900/500/1000. Thirty-five saved all-22 physical panels supply the temperature and velocity curves. Equal-age500 ratios expose Tree's material/interface gains and flow misses. The allocation panel covers this round from 13:05:36 UTC and reaches 13.58 of 18 associated process-hours; historical parent costs are separate. MSE is train normalized; physical RMSE retains benchmark units. Allocation measures associated process lifetimes, not active GPU compute.

## Predictor: matched 500 gains and misses

The following means give each exposed validation case equal weight after evaluating the native role support. References use analytic-wake flow plus a numerically advanced shared-grid thermal benchmark. All 22 ordinary case statistics use one recorded thermal-convergence frame (`converged_final`). RMSEs retain dataset units, without independent CFD or SI calibration. Lower is better.

| Native role | Tree500 | Pair500 | Fine500 | Dense500 |
|---|---:|---:|---:|---:|
| far/omega | 0.122500 | 0.135781 | 0.140525 | 0.125008 |
| far/p | 0.019355 | 0.017622 | 0.015697 | 0.018017 |
| far/temperature | 1.708791 | 1.576796 | 1.622774 | 1.079433 |
| far/u | 0.074390 | 0.044170 | 0.049681 | 0.037970 |
| far/v | 0.005297 | 0.005337 | 0.004852 | 0.003981 |
| final_port/h_effective | 0.808456 | 0.713833 | 1.008641 | 0.956135 |
| final_port/outside_temperature | 1.684234 | 2.197660 | 2.656561 | 1.221041 |
| fluid/omega | 0.228151 | 0.254696 | 0.244876 | 0.182963 |
| fluid/p | 0.021913 | 0.020935 | 0.019760 | 0.018629 |
| fluid/temperature | 1.758742 | 1.706414 | 1.790461 | 1.100098 |
| fluid/u | 0.078214 | 0.048286 | 0.054539 | 0.039925 |
| fluid/v | 0.006068 | 0.006343 | 0.006264 | 0.004954 |
| initial_port/h_effective | 9.849548 | 9.848572 | 9.885578 | 11.428815 |
| initial_port/outside_temperature | 2.417994 | 2.120647 | 2.039635 | 2.591399 |
| inlet_outlet_pressure_difference | 0.016961 | 0.010630 | 0.008112 | 0.008364 |
| material_temperature | 1.252626 | 1.462747 | 1.836435 | 1.040032 |
| module_material_peak | 1.289538 | 1.721443 | 1.898214 | 1.062606 |
| near/omega | 0.508433 | 0.572189 | 0.535721 | 0.367312 |
| near/p | 0.031444 | 0.034632 | 0.035047 | 0.021946 |
| near/temperature | 1.996903 | 2.227704 | 2.387626 | 1.117293 |
| near/u | 0.094287 | 0.067193 | 0.075068 | 0.050631 |
| near/v | 0.009104 | 0.009957 | 0.011197 | 0.008125 |
| q_normal_proxy | 4.631226 | 4.388414 | 4.345345 | 4.076007 |
| surface_temperature | 1.389610 | 1.749967 | 1.966665 | 1.206524 |

Tree500 improves all eight main physical errors versus Tree200. Against Pair500 it improves v, omega, material temperature and surface temperature, but loses u, p, fluid temperature and the q-normal proxy. Material-temperature RMSE is 1.252626 versus 1.462747 (14.36% lower), and surface-temperature RMSE 1.389610 versus 1.749967 (20.59% lower). Fluid-temperature RMSE 1.758742 versus 1.706414 is 3.07% worse, and u 0.078214 versus 0.048286 is 61.98% worse. Against Fine500, material and surface errors improve 31.79%/29.34%; fluid temperature improves only 1.77%, with 11/22 case wins and a slightly negative paired median. This is a cross-field tradeoff, not uniform organizer superiority.

Paired differences use the identical 22 IDs, with delta=control RMSE minus Tree RMSE. Numerical ties satisfy abs(delta)<=1e-8+1e-6*max(abs(Tree),abs(control)); these are descriptive exposed-development counts, not significance or equivalence tests.

| Tree500 comparison | Fluid-T mean delta; wins/ties/losses | Material-T mean delta; wins/ties/losses | u mean delta; wins/ties/losses |
|---|---:|---:|---:|
| Pair500 | -0.052328; 11/0/11 | +0.210121; 15/0/7 | -0.029929; 0/0/22 |
| Fine500 | +0.031720; 11/0/11 | +0.583809; 19/0/3 | -0.023675; 0/0/22 |
| Dense500 | -0.658644; 2/0/20 | -0.212594; 4/0/18 | -0.038289; 0/0/22 |

Against Pair500, fluid-T deltas by M3/M5/M7/M10 are +0.164432/+0.282218/-0.056809/-0.872564; material-T deltas are +0.456155/+0.347359/+0.393642/-0.640068. The aggregate material advantage misses all four M10 cases. Tree loses u in every stratum. Near/far fields, ports, peak material temperature and pressure difference are retained in the complete 24-role table above and the local paired artifact; a favorable aggregate does not replace those checks.

![Matched500 physical temperature fields, signed residuals and per-module material errors on the four fixed representatives](../../diagnostics/generated/tree_lite_20261004/figures/figure2_physical_fields.png)

**Figure 2.** All four models use exact 500 weights, identical saved reference grids and masks. Rows 0277/0291/0294/0687 show reference and Tree temperature, four signed temperature residuals, Tree velocity residual, interface residual by source slot, and per-module material-temperature RMSE. Tree's hardest fluid-temperature case within this fixed panel is 0687 (M10); this does not claim it is the hardest of all 22. Material coordinates were not archived, so module RMSE replaces an invented material contour. Shared scales contain every retained residual without clipping. The measured fields support the 500 material/interface tradeoff and retain the high-M misses.

**Reference provenance.** Reference temperatures are numerical shared-grid solutions. Recorded convergence is a saved-frame temporal stopping condition, not mesh convergence or CFD certification.

## Final selected predictions and endpoint check

The declared sampled field-MSE rule selects literal Tree900 and Pair900 after each completed 1000 horizon. Fine remains at 500; Dense1000 is an existing read-only reference. The selected Tree/Pair ages match. Fine has a shorter horizon; Dense differs in backbone, capacity, heat/context features and objective. All values below are equal-case means of native RMSE on the fixed 22 exposed validation cases.

| Native role | Tree900 | Pair900 | Fine500 | Dense1000 |
|---|---:|---:|---:|---:|
| far/omega | 0.089587 | 0.085138 | 0.140525 | 0.060510 |
| far/p | 0.013178 | 0.015869 | 0.015697 | 0.013030 |
| far/temperature | 1.175860 | 1.394143 | 1.622774 | 0.913979 |
| far/u | 0.039958 | 0.029970 | 0.049681 | 0.026990 |
| far/v | 0.003227 | 0.003480 | 0.004852 | 0.003240 |
| final_port/h_effective | 0.743892 | 0.836486 | 1.008641 | 0.725213 |
| final_port/outside_temperature | 1.358518 | 1.966037 | 2.656561 | 1.524593 |
| fluid/omega | 0.195425 | 0.162608 | 0.244876 | 0.120396 |
| fluid/p | 0.015568 | 0.016984 | 0.019760 | 0.013981 |
| fluid/temperature | 1.215061 | 1.474960 | 1.790461 | 0.949971 |
| fluid/u | 0.041526 | 0.032992 | 0.054539 | 0.028457 |
| fluid/v | 0.004029 | 0.004273 | 0.006264 | 0.003605 |
| initial_port/h_effective | 8.838263 | 8.459603 | 9.885578 | 10.932989 |
| initial_port/outside_temperature | 2.364824 | 1.392564 | 2.039635 | 1.286997 |
| inlet_outlet_pressure_difference | 0.010667 | 0.008428 | 0.008112 | 0.008415 |
| material_temperature | 1.226645 | 1.310602 | 1.836435 | 0.865606 |
| module_material_peak | 1.134705 | 1.336865 | 1.898214 | 0.841038 |
| near/omega | 0.451309 | 0.363496 | 0.535721 | 0.275829 |
| near/p | 0.024028 | 0.020674 | 0.035047 | 0.018426 |
| near/temperature | 1.280978 | 1.759162 | 2.387626 | 1.114823 |
| near/u | 0.048509 | 0.045894 | 0.075068 | 0.035122 |
| near/v | 0.006732 | 0.006916 | 0.011197 | 0.005055 |
| q_normal_proxy | 3.968524 | 4.113171 | 4.345345 | 3.587552 |
| surface_temperature | 1.427279 | 1.517544 | 1.966665 | 0.974319 |

The exact 1000 endpoint comparison keeps both models at the same final age; it does not replace their selected 900 controls.

| Native role | Tree1000 | Pair1000 | Tree change against Pair |
|---|---:|---:|---:|
| fluid/u | 0.044411 | 0.042176 | -5.30% |
| fluid/v | 0.004433 | 0.004140 | -7.07% |
| fluid/p | 0.018514 | 0.016551 | -11.86% |
| fluid/omega | 0.197303 | 0.200147 | +1.42% |
| fluid/temperature | 1.229348 | 1.329717 | +7.55% |
| material_temperature | 1.148491 | 1.128135 | -1.80% |
| surface_temperature | 1.243998 | 1.281637 | +2.94% |
| q_normal_proxy | 4.055170 | 4.144896 | +2.16% |

Positive percentages indicate lower Tree RMSE. Native 1000 sampled field MSE is 0.04013288 for Tree and 0.04893762 for Pair; selected 900 is 0.03972594 and 0.03698975. Tree1000 improves material/interface over its 900 state while several flow roles regress. This is a selector and cross-field tradeoff, rather than evidence that every role has converged.

For paired statistics below, a positive difference is control RMSE minus Tree RMSE. Numerical ties use 1e-8 +1e-6×maximum absolute paired RMSE; this is a numerical convention, not statistical equivalence.

| Selected control | Native role | Mean difference | Median difference | Tree wins/ties/losses |
|---|---|---:|---:|---:|
| Pair900 | fluid/temperature | +0.259899 | +0.255086 | 17/0/5 |
| Pair900 | material_temperature | +0.083956 | -0.037886 | 10/0/12 |
| Pair900 | surface_temperature | +0.090265 | +0.006702 | 11/0/11 |
| Pair900 | fluid/u | -0.008534 | -0.007683 | 1/0/21 |
| Pair900 | fluid/omega | -0.032818 | -0.029593 | 0/0/22 |
| Fine500 | fluid/temperature | +0.575400 | +0.639410 | 20/0/2 |
| Fine500 | material_temperature | +0.609789 | +0.584575 | 19/0/3 |
| Fine500 | surface_temperature | +0.539386 | +0.441441 | 18/0/4 |
| Fine500 | fluid/u | +0.013013 | +0.010547 | 20/0/2 |
| Fine500 | fluid/omega | +0.049451 | +0.050152 | 22/0/0 |
| Dense1000 | fluid/temperature | -0.265090 | -0.158289 | 3/0/19 |
| Dense1000 | material_temperature | -0.361039 | -0.397277 | 4/0/18 |
| Dense1000 | surface_temperature | -0.452959 | -0.494040 | 3/0/19 |
| Dense1000 | fluid/u | -0.013069 | -0.012918 | 1/0/21 |
| Dense1000 | fluid/omega | -0.075030 | -0.070050 | 0/0/22 |

Against Pair900, Tree improves 14 of 24 mean roles and 6 of 8 core roles: fluid T 17.62%, material 6.41%, interface 5.95%, v 5.72%, p 8.34% and q-proxy 3.52%. Streamwise velocity is 25.87% worse (1/0/21), and vorticity 20.18% worse (0/0/22). The positive material mean accompanies a negative median and 10/0/12 case wins; pressure and q-proxy also have negative medians despite positive aggregate means.

| M stratum (cases) | Thermal role against Pair900 | Mean difference | Median difference | Tree wins/ties/losses |
|---|---|---:|---:|---:|
| 3 (6) | fluid/temperature | +0.151236 | +0.110868 | 3/0/3 |
| 3 (6) | material_temperature | +0.072059 | +0.123249 | 3/0/3 |
| 3 (6) | surface_temperature | +0.032318 | +0.073681 | 3/0/3 |
| 5 (6) | fluid/temperature | +0.313474 | +0.137691 | 4/0/2 |
| 5 (6) | material_temperature | +0.121071 | -0.092085 | 2/0/4 |
| 5 (6) | surface_temperature | +0.081850 | -0.087897 | 2/0/4 |
| 7 (6) | fluid/temperature | +0.376851 | +0.354980 | 6/0/0 |
| 7 (6) | material_temperature | +0.320439 | +0.237756 | 4/0/2 |
| 7 (6) | surface_temperature | +0.365793 | +0.268527 | 5/0/1 |
| 10 (4) | fluid/temperature | +0.167103 | +0.167441 | 4/0/0 |
| 10 (4) | material_temperature | -0.308595 | -0.340468 | 1/0/3 |
| 10 (4) | surface_temperature | -0.223485 | -0.228455 | 1/0/3 |

M10 material/interface have negative mean differences−0.308595/−0.223485 and 1/0/3 wins each. Fluid-T means improve in every M stratum, with 17/0/5 overall wins. Tree beats shorter-horizon Fine500 on 22/24 means; that comparison combines architecture and exposure. Dense1000 is stronger on all 8 core means and 21/24 total roles, with its architectural/objective differences preserved. No full-population or independent-test claim follows from this repeatedly exposed panel.

[Saved selected/endpoint/paired statistics](/data/wanglz/ModularDT/thermal_development/tree_lite_20261004/report_numerics_final.json). Context/role support counts are reconciled across all 22; summary-only checks do not independently prove all 22 grid/mask array hashes.

## Reviewed extension, contention and revisions

At the completed 500 review (17:29 UTC), the predeclared field selector chose Pair as the stronger nongroup control: 0.07221468 versus Fine's 0.07364724. Root approved only Tree and Pair for exact 500→1000 continuation, asking whether extra exposure would strengthen the material/interface gains while recovering flow and fluid-temperature misses. Model, optimizer, RNG, normalization, membership and absolute-schedule guards passed. Fine stayed at 500; no new arm or selector was introduced.

Tree's thermal regression at 600 recovered at 700; subsequent checkpoints changed the balance between flow and thermal roles. Pair also recovered after a regressing window. These measured revisions supported continuing healthy reconstruction rather than treating one role/window or weak auxiliary force as persistent divergence. Final900 selection improves Tree's fluid temperature versus Pair, while the material median and M10 misses remain. Detailed milestone reviews and literal hashes remain in the [development guide](../guides/Thermal_Tree_Lite_Development.md) and saved review receipts.

An unrelated GPU1 job appeared at 17:47:29 UTC and was absent by approximately 18:16. It was left untouched and excluded from our ledger; our concurrent slower Pair process remained charged. Pair's continuation process lasted 4,628 seconds, including 4,552.980 seconds of epoch timings and 75.020 seconds of loading/checkpoint/export/shutdown residual. Observed slowdown does not identify its sole cause. GPUs1/2 carried the authorized work; GPU0 was untouched.

The final package planning reserve changed from 95 to 105 aggregate minutes after actual diagnostic costs and selected-versus-endpoint cache needs were reviewed. This estimate did not change the 18-hour/12-hour ceilings and is not added to measured allocation. Final normal1000 and null1000 ran sequentially on GPU1. A CPU watcher failed before its native child because the system Python lacked `hashlib.file_digest`; rerunning the unchanged watcher with ModularDT incurred a preserved 127.029-second queue delay. No earlier native calls were erased or invented. Root's first saved-only checkpoint checker also encountered an array-comparison error; recursive exact comparison corrected the checker before binding acceptance, with no native/model call.

The implementation revisions were limited to projected controls/omitted inference analysis/local shadow, strict original-action replay, and explicit derivative participation. The local shadow is an intentional organizer-gradient policy change, not an equivalent gradient. Failed engineering probes and benchmark attempts remain charged. No learning-rate rescue, null-coefficient retune, early physical pooling, forced K or second seed was introduced.

## Organizer: execution and disclosed information paths

For positive pair density rho, the reader's affine projection satisfies W*(sum(A*B*h)/rho)+b=sum(A*B*(W*h))/rho+b. Projecting group controls before receiver expansion therefore removes 16-channel pair tensors from the numerical path. Empty density retains the original zero-control affine bias, and tanh remains after mixing. MM/ME/EM/QM use one projected channel; QE uses eight score/gain channels. Public diagnostic exports retain complete controls. Unrequested structural/repeated-path analysis is omitted and marked unmeasured rather than reported as zero work.

Native low/high-M e200 proof gives maximum relative physical-gradient differences 1.768e-6/3.885e-7 for original versus projected execution. Heat, center and query first derivatives agree within the recorded tolerances. Local shadow versus hard-only physical gradients differ 7.058e-10/1.397e-9; measured input differences are zero and hard values/RNG/counters agree. Whole/local organizer-gradient cosines 0.93625/0.99784 compare different surrogate derivatives, not equivalent training. The disposable two-update finite check is engineering evidence excluded from field selection. Local shadow performs one complete physical wrapper and two Stage-A calls instead of two/four, with separately recorded checkpoint replay of local reductions.

The prescribed global token comes only from global_encoder(batch.global_context). Four aggregate-heat slots are zero under source_local_v3; public flow/material/domain context and active module inventory remain. There is no learned M/E pooling inside this token. It enters fine updates, QM messages, shared background and port heads. Separately, the organizer planner broadcasts weighted pools of all phase-current M and raw E states into node descriptors and source logits. The M pool contains heat information even at P0: zero aggregate-heat context slots do not eliminate global learned planning.

The inherited `interaction_coarse_context_*` names measure the query/global background C_g; the three-term reader has no coarse latent bank. Their norm fraction is a share of latent vector norms, not a physical, energy or organizer-utility share. `loss_organizer=0` records only the disabled legacy anti-collapse regularizer: the campaign structural cost with weight 0.001 and local organizer task-gradient bridges remain live. `campaign_shadow_calls=0` counts extra complete soft-wrapper calls; local soft reductions, activation-checkpoint replay and null-response work have separate scopes. These meanings are traced in `legacy_metric_semantics_source_audit.md`.

Faithful h heads use membership-weighted M/E summaries, selected donor mass/presence, geometry/depth/role, prescribed context and phase. Their geometry projection excludes rich node embeddings, all-source latent pools and global total-mass statistics. recompute_controls rebuilds only these local summaries. The conditional content Jacobian treats phase-current latent M/E as independent leaves and holds realized memberships, measures, geometry, prescribed context and phase fixed. It excludes planner changes and upstream physical-input ancestry; an excluded latent donor is not an isolated physical subsystem.

Raw organizer E encoding is geometry-only and unchanged across P0/P1/P2. Physical QE uses phase-prepared environmental states after EM transport, so its values can depend on module heat. M states carry the first and second frozen Stage-A responses, port/outside-temperature reads and refinement. These value, control-content, planning and phase-ancestry paths must remain distinct.

Native QE support is full in the retained engineering proof and selected900 panel. Eligible omitted-donor soft-path restoration was verified with a constructed reduction; it was not a naturally excluded native donor test. The failed 192-call engineering benchmark attempt remains charged and is excluded from accepted latency statistics.

## Selected-weight organizer

Evidence binds selected Run3302 Tree900, checkpoint SHA256 `6e68919f4390cb87da844be20b19fdef75ff8c5fbd003375521cface18725692`, and the fixed 150/22 development manifest. Normal22 was reused; only four missing detailed fields/graphs were filled. The all 22 light organization pass used 256 geometry-selected fluid queries per case plus actual wrapper port/local streams, covering all 15 phase/routes. Its sampled receiver query panel differs from the full-grid detailed evaluation; the physical case geometry is unchanged.

Positive support below pools eligible pair counts across the22 cases and recorded streams, excluding padding; it is distinct from equal-case physical RMSE. Frontier and participating-group ranges span 22 cases; participating A>0 groups may lack eligible values. All routes allocate 15 slots. Complete group/receiver signatures were quantized at 1e-6 with capacity 4096; no recorder capacity was saturated; these counts concern the quantized signatures. Group distinctness spans 3–8 for MM/ME and 8–8 for EM/QM/QE; complete per-route receiver counts and real-node ranges remain in the numerical receipt.

| Route | Frontier / participating groups | Positive support | Physical MLP rows / calls |
|---|---:|---:|---:|
| P0/MM | 3–8 / 2–8 | 100% | 3,168 / 22 |
| P0/ME | 3–8 / 2–8 | 14.5833% | 50,688 / 22 |
| P0/EM | 8–8 / 8–8 | 100% | 50,688 / 22 |
| P0/QM | 8–8 / 5–8 | 97.7584% | 202,752 / 132 |
| P0/QE | 8–8 / 5–8 | 100% | 3,244,032 / 132 |
| P1/MM | 3–8 / 2–8 | 100% | 3,168 / 22 |
| P1/ME | 3–8 / 2–8 | 14.5833% | 50,688 / 22 |
| P1/EM | 8–8 / 8–8 | 100% | 50,688 / 22 |
| P1/QM | 8–8 / 5–8 | 98.8001% | 202,752 / 132 |
| P1/QE | 8–8 / 5–8 | 100% | 3,244,032 / 132 |
| P2/MM | 3–8 / 2–8 | 100% | 3,168 / 22 |
| P2/ME | 3–8 / 2–8 | 14.5833% | 50,688 / 22 |
| P2/EM | 8–8 / 8–8 | 100% | 50,688 / 22 |
| P2/QM | 8–8 / 8–8 | 99.5343% | 67,584 / 44 |
| P2/QE | 8–8 / 8–8 | 100% | 1,081,344 / 44 |

Density/weight deviations use eligible receiver×source physical measure; affine/gain RMS uses supported physical measure. QE affine RMS combines score/gain channels; effective gain is 1+tanh(gain). Source participation is reciprocal pooled receiver-weighted source HHI; group participation is inverse group HHI (case range). M/E donor HHI averages active groups using phase-owned source measure×control density. These concentration measures differ from exact support.

| Route | Density−1 / weight−1 RMS | Affine / effective-gain RMS | Source / group participation | M / E donor HHI |
|---|---:|---:|---:|---:|
| P0/MM | 0.0215829 / 0.0211646 | 0.0381177 / 1.01274 | 4.65026 / 1.8–7.25692 | 0.173852 / 0.00521314 |
| P0/ME | 3.111 / 3.111 | 0.79698 / 0.366009 | 17.9804 / 1.8–7.25692 | 0.778297 / 0.0555908 |
| P0/EM | 0.0421661 / 0.0421661 | 0.0214377 / 1.02143 | 4.92916 / 8–8 | 0.202875 / 0.00520986 |
| P0/QM | 0.496627 / 0.496627 | 4.03132 / 1.99931 | 3.98259 / 1.42947–7.00007 | 0.235947 / 0.00541355 |
| P0/QE | 0.165362 / 0.165362 | 1.87435 / 0.394316 | 186.89 / 1.42947–7.00007 | 0.271656 / 0.00533872 |
| P1/MM | 0.0213699 / 0.0209412 | 0.0245956 / 1.00498 | 4.65028 / 1.8–7.25692 | 0.173851 / 0.00521314 |
| P1/ME | 3.12647 / 3.12647 | 0.890784 / 0.320729 | 17.8193 / 1.8–7.25692 | 0.894612 / 0.0561103 |
| P1/EM | 0.0410973 / 0.0410973 | 0.021438 / 1.02143 | 4.92935 / 8–8 | 0.202867 / 0.00520988 |
| P1/QM | 0.468003 / 0.468003 | 3.97441 / 1.99921 | 4.13124 / 1.44251–7.04588 | 0.232453 / 0.00541696 |
| P1/QE | 0.170923 / 0.170923 | 2.14193 / 0.341975 | 186.55 / 1.44251–7.04588 | 0.27341 / 0.00534845 |
| P2/MM | 0.0213963 / 0.0209677 | 0.0240724 / 1.00423 | 4.65027 / 1.8–7.25692 | 0.173851 / 0.00521314 |
| P2/ME | 3.12394 / 3.12394 | 0.895229 / 0.320515 | 17.8456 / 1.8–7.25692 | 0.893608 / 0.0560291 |
| P2/EM | 0.0413702 / 0.0413702 | 0.0214381 / 1.02143 | 4.9292 / 8–8 | 0.202873 / 0.00520989 |
| P2/QM | 0.413508 / 0.413508 | 3.99057 / 1.99923 | 4.34365 / 7.05727–7.8536 | 0.232231 / 0.00541654 |
| P2/QE | 0.163381 / 0.163381 | 2.17917 / 0.334004 | 187.008 / 7.05727–7.8536 | 0.272139 / 0.00534808 |

The 22 wrappers executed 8,356,128 padded physical MLP rows in 814 fine calls, including repeated port/local streams; coarse/local modules, backward and hardware kernels are excluded. QE support remains 100%. Logical support/concentration produced no physical MLP omission. Same-M ranges exist in all 60 M×phase×route buckets (≥4 cases each); they reflect geometry/input variability, not controlled context attribution.

Whole rebuild used 98 wrappers on the all-22 input-only Q14 sensor panel, with deeper equal/multiple refinements and derivative pullbacks on the fixed four. Representation passed 21,372/21,372; physical outputs 408/416, stricter fine-core 397/416. All eight ordinary failures were aggregate `pred_interface` tensors: 0663 unequal; 0674 permutation/unequal; 0677 unequal; 0684 permutation/unequal; 0686 permutation/unequal. Maxabs=5.8174e-5–1.5259e-4, RMSE=8.7517e-6–1.4037e-5, pointwise tolerance ratio=1.03428–4.31159 at rtol/atol=2e-5/2e-5; all eight overlap strict atol=2e-6 failures. Per-channel/max-relative errors were not retained. Derivative comparisons passed 64/64: 48 both-used coordinate/feature/mass paths and 16 both-unused length paths. No tolerance changed or retry occurred.

Native locality used four wrappers and existing participating P0/ME groups 1/3/3/5. Frozen-membership M exclusions 2/4/6/8 and raw-E exclusions 164 each had exactly-zero content derivatives; admitted nonzero paths were M 1/1/1/2, E 28 each. All 16 channels used both inputs in autograd: 64 vector VJPs, eight recomputes. Separate live planner replays used eight raw-logit VJPs, four organizer prepares, with requires_grad=true/None=false. Potential eligible value pairs 384/192/384/384 and positive A·B far paths 56/28/56/56 do not establish physical effect. Maintained P0/QM native E exclusions remain separately labelled; its legacy Jacobian lacks per-channel None metadata.

Utility reused four normal references and ran 20 interventions, zero new normal wrappers. Fixed replay guarded P0/P1/P2 source/receiver identities/coordinates, ordered streams, measures, density/weight/support/edge/near and reconstructed pair controls; 161 accesses per representative, density reconstruction error 0. Physical values/ports/local physics remained live. Each normal/intervention executed 1,998,768 physical MLP rows. Equal-case fluid-temperature RMSE is in dataset units:

| Intervention | Mean T RMSE | Native changed-support pairs, four cases |
|---|---:|---|
| Cached normal | 1.03023 | — |
| control_identity_fixed_access | 1.93675 | 0 / 0 / 0 / 0 |
| full_access_fixed_controls | 1.07816 | 1476 / 2908 / 3444 / 5088 |
| geometry_reference_actions | 1.02986 | 0 / 0 / 0 / 0 |
| effective_rewire | 1.42652 | 144 / 1136 / 336 / 816 |
| root_union | 1.08502 | 0 / 448 / 0 / 168 |

Identity disables projections/biases at fixed access; full access retains original controls/biases with uniform permissions. Geometry reference reassigns learned joint permission/control tuples at fixed binary degrees/row multisets: weights changed despite zero support changes. It is a frozen action reassignment, not an independently trained geometry-only architecture. Rewire changes support/weights with unique logical pair work retained. Root union changes support in two cases, weights in all four. Identity/rewire worsen error; geometry reference improves it slightly (−0.000373096), so normal learned actions are consequential without uniformly best frozen utility.

Single-QE ablation selects existing groups 8/7/14/5, sets h→0 preserving bias and fixed upstream state/values/key-values/access/measures/queries. Four wrappers plus 28 normal / 32 ablated prepared reads yielded fluid ΔT RMS 0.753367/0.511308/0.650381/0.730813; T RMSE changed 1.061057→1.088332,0.962184→1.136530,0.919360→1.027387,1.178329→1.402895. This is a conditional P2 read effect.

Context replay used nearest compatible distinct selected-TRAIN prescribed tuples, unchanged geometry/heat/queries: eight wrappers compared current actions with original actions at the same trial context. Four measured ΔT RMS values were 0.0171717/0.00668269/0.00453851/0.0181356. Process exit 1 retained a false final hook-restoration guard; model/modes/selection/campaign/dataset/RNG/methods/backend/no-grad checks passed. Differing hooks/modules were not retained; mechanism is unproven. No retry/reference solve occurred.

![Selected900 actual organizer maps, conditional controls, residuals and one-group QE effects](../../diagnostics/generated/tree_lite_20261004/figures/figure3_organizer.png)

**Figure 3.** Tree900 fixed four (M3/5/7/10), 28 panels: actual P2 access, M/raw-E control density, source-measure/head-mean QE gain, normal T, signed benchmark residual, conditional h→0 ΔT. Action graphs use 8192 full-grid points; effects use 7976/7837/7691/7463 fluid points with nonfluid cells masked. QE support 100%; affine-probe maxerror≤1.90735e-6 and baseline-fluid maxerror≤2.86103e-6 (2e-5 tolerance). Conditional computation and shared-grid residuals support neither sparse execution nor causality.

## Measured runtime and memory

The profiler-off interleaved benchmark uses the same physical GPU1, resident workers, training cases 0006 (M1)/0231 (M12), FP32, Q1024/Q8192 and chunk 128. It retains 432 total calls and 292 measured calls. Medians/p90 describe four measured inference repeats or three training repeats after warmup; no confidence interval is inferred. Original/lowered Tree share exact e200 weights. Inference and profiler comparisons use hard wrappers. The training-time and training-memory rows compare original `whole_wrapper_shadow_v1` with lowered `local_context_shadow_v1`; they combine execution savings with the intentional organizer-derivative change. They do not compare local shadow against original hard-only training. Preparation excludes only final P2 fluid decoding and retains upstream/ancillary work; scopes overlap and cannot be added.

| Scope | M1 original→lowered/local | M12 original→lowered/local | Limit |
|---|---:|---:|---|
| Q8192 complete wrapper | 0.828412→0.368783s | 0.892988→0.408967s | Pair medians 0.178169/0.182065s leave Tree 2.07/2.25 times slower |
| Reused P2 decode | 0.620664→0.219283s | 0.601003→0.218324s | Prepared physical input; inclusive with wrapper |
| Frozen-parameter heat input VJP | 1.191698→0.615422s | 1.217555→0.655789s | Fixed nonuniform field adjoint, input derivative only |
| B8/Q1024 whole-wrapper→local shadow forward/backward/optimizer boundary | 1.454957→1.226953s | 2.589457→2.103051s | Reset outside timing; native calibration restored; not effective B48 or a whole epoch |
| Same B8 whole-wrapper→local shadow CUDA allocated peak | 2924.87→2128.37 MiB | 12388.53→7355.68 MiB | Active allocations; reserved memory and post-call residency are separate |

![Interleaved measured execution latency and training allocation, with separate explanatory profiler activities](../../diagnostics/generated/tree_lite_20261004/figures/figure4_runtime.png)

**Figure 4.** The matched high-M wrapper falls 54.20%; training boundary falls 18.78% and allocated peak 40.63% (12.10→7.18 GiB). Actual profiled CUDA kernel activities fall 79780→28744 for high M; profiling perturbs runtime and is used only to explain removed work. High-M summed kernel durations 133.28→57.54 ms are distinct from overlap-aware union and host latency. Inference extra allocation remains approximately 81.7→81.5 MiB, so there is no universal memory win. These measured savings come from execution and gradient-policy work; they do not establish learned physical-source pruning. Full-access fallback and final actual rows remain explicit in Figure 3.

## Selected-weight responses

Tree900 null22 used 256 fluid queries, 194 wrappers and 172 variants. Fractions 0.1/0.2 are feasible donor-transfer bounds, not fractions of total heat. Analytic-wake flow is heat-independent at fixed geometry/context; nonzero u/v/p/omega changes are null misses. Temperature changes lack held positive truth. Raw units are dataset-native; scaled values divide by training SD. Means weight cases equally after within-case averaging:

| Channel | f=.1 raw / training-SD RMS | f=.2 raw / training-SD RMS |
|---|---:|---:|
| u | 0.000722237 / 0.00178779 | 0.00144627 / 0.00358003 |
| v | 5.38708e-05 / 0.00117352 | 0.000107854 / 0.0023495 |
| p | 0.000227164 / 0.00149232 | 0.000454679 / 0.00298695 |
| omega | 0.00218904 / 0.00214841 | 0.00436514 / 0.00428413 |
| temperature | 0.0607861 / 0.00834887 | 0.121413 / 0.0166759 |

Maximum-case native RMS was 0.00241852(u),0.000126538(v),0.000640846(p),0.00857875(omega). No invented threshold declares a pass. Tree exceeds Pair900 in u/p/omega, with slightly smaller v; Fine500 has larger u/p and smaller omega than Tree. These selected checkpoint ages remain explicit.

Force probes fix 900 weights/objective age and vary training input seeds 101/300/500, not historical weights/gates. The same 207 named trainable tensors (3,336,120 parameters) receive native-boundary and saved coefficient 0.1 weighted-null VJPs:

| Input seed | Native norm | Weighted-null norm | Null/native ratio |
|---|---:|---:|---:|
| 101 | 0.400688 | 0.000252497 | 0.000630159 (0.0630159%) |
| 300 | 0.280712 | 0.000419293 | 0.00149368 (0.149368%) |
| 500 | 0.314811 | 0.000550405 | 0.00174837 (0.174837%) |

All three passed full model/optimizer/campaign/selection/dataset/modes/RNG/no-grad restoration with zero `Tensor.backward` and `optimizer.step` calls; the measured `autograd.grad` VJPs remain charged. Each charged one boundary plus four null wrappers (15 total). Small 0.0630–0.1748% force ratios describe isolated boundaries, not accumulated training impulse or guaranteed correction.

TRAIN0348 supplies one exposed family: 11 historical states (baseline/eight position/two heat variants), 10 responses and 11 wrappers; no new solve. Held positive families=0. Heat variants transfer ±0.125 between slots 0/1 at fixed geometry. Normalized saved quadrature/common valid support gives:

| Role | + / − ΔT RMSE | + / − zero-change RMS | Valid queries |
|---|---:|---:|---:|
| Fluid T | 0.0797933 / 0.0822247 | 0.156527 / 0.156527 | 7,435 |
| Interface T | 0.109689 / 0.101112 | 0.256417 / 0.256417 | 640 |
| Material T | 0.0637729 / 0.0691511 | 0.334492 / 0.334492 | 30,960 |

All six beat zero-change on exposed TRAIN0348. Against Pair900, Tree has larger fluid-response errors for both signs (0.079793/0.082225 versus 0.072608/0.068227), while interface and material errors improve for both signs. All six improve over shorter-horizon Fine500, with exposure confounded. Acceptance tolerances remain undeclared. Original sidecars record temporal convergence; the exact historical generator revision is unavailable. Ordinary thermal references are shared-grid numerical benchmarks, and flow is analytic-wake. No held fidelity, mesh/CFD certification or SI calibration follows. Fixed sum(q_i) constrains per-solid-cell source amplitude, not certified watts/energy conservation.

![Selected null leakage, isolated auxiliary force and stored training-family heat responses](../../diagnostics/generated/tree_lite_20261004/figures/figure5_response_null_force.png)

**Figure 5.** Sixteen panels at Tree900/Pair900/Fine500 show all 22 raw/training-SD null responses at 0.1/0.2, misses by M, model-only thermal sensitivity, three fixed-weight force ratios (Tree 0.063016/0.149368/0.174837%), exposed TRAIN0348 ±0.125 response errors versus zero-change, and four Tree ΔT point clouds. The latter are model changes, not perturbed truth. TRAIN atlas errors beat zero-change; held positive truth remains absent.

B misses remain whole-output failures, slightly better geometry-reference utility and failed context hook restoration; no sparse physical saving or causality is established. C misses remain nonzero known-null flow response, small final null force and zero held positive families. Stronger null correction and held positive response evidence remain future work; none was launched.

Sources: selected `evaluation/tree_final_selected_*` summaries; `final_organizer_all15_report_statistics.json`; `tree900_whole_physical8_readout.json`; `tree900_selected_lane_independent_saved_audit.json`; `tree900_selected_lane_actual_work_readout.json`; Figure 3/5 receipts. Ten unique child lifetimes total 254.807865 s; nested same-PID receipts deduplicated; nine exits 0 / context 1.

## Bounded frozen inverse reuse

Four fixed tasks (0277/0291/0294/0687), two public starts (uniform and capped random, seed 20261002 plus case ID) and four proposal modes give 32 trails: Tree joint, Tree graph block, size-matched random block, and Pair joint. Each allows at most ten attempted updates and allocation-fraction trust radii 0.05/0.025. Observed sensors alone accept proposals; held sensors and hidden heats are evaluation only. Rank probes are shared and separately charged. Each candidate uses an ordinary rebuild or a strictly valid fixed-action continuation, and accepted proposals are confirmed by an ordinary rebuild. No model is trained, no cap is widened and no reference is solved.

The capped simplex uses global selected-training active-source bounds 0.5036706328392029–1.999153733253479, fitted from 851 active training inputs. The public `sum_i q_i` is the sum of prescribed per-solid-cell source amplitudes, without an integrated-watt or energy-conservation calibration. Unsupported totals remain unavailable. Retained states meet caps, padding and native sum tolerance: global maximum sum residual 5.364418e-7 and zero cap excess. This is benchmark-input feasibility, not certification of a physical design. Rejected full heat vectors were not archived; retained vectors are independently audited and rejected-proposal cap enforcement is source backed.

| Mode (8 trails each) | Mean observed RMSE, start→final | Mean held RMSE, start→final | Mean hidden-heat RMSE, start→final | Observed / held / heat wins-ties-losses | Accepted / rejected updates |
|---|---:|---:|---:|---|---:|
| Tree joint | 2.147187→1.692055 | 2.290480→2.165055 | 0.438160→0.434446 | 4-4-0 / 3-4-1 / 2-4-2 | 18 / 62 |
| Tree graph | 2.147187→1.592828 | 2.290480→2.123544 | 0.438160→0.444647 | 5-3-0 / 4-3-1 / 2-3-3 | 10 / 70 |
| Tree random | 2.147187→1.535322 | 2.290480→2.127277 | 0.438160→0.446553 | 7-1-0 / 5-1-2 / 3-1-4 | 24 / 56 |
| Pair joint | 2.269596→0.607918 | 2.684910→2.368823 | 0.438160→0.368018 | 8-0-0 / 6-0-2 / 5-0-3 | 27 / 53 |

Wins/ties/losses here compare native initial and final errors, with exact ties. Tree has lower absolute final held means, but starts with lower model error; Pair has stronger observed and hidden-heat fitting. Tree graph's observed mean is worse than random's, and mean hidden-heat error worsens for both block methods. There is no clear graph-reuse advantage. Both Pair0291 starts worsen held error despite lower observed error (2.4405→3.6183 and 3.3347→3.5866). Fitting observed sensors does not establish unique heat recovery or generalization.

Graph proposals use proper module support unions from P0/P1 typed groups with 2≤k<M. They do not require sensor receiver participation, include P2, or deduplicate identical candidate subsets. Sampling uniformly over entries can weight repeated subsets. When no proper subset is available, the method falls back to all M modules; random proposals copy the graph trail's recorded sizes. The final graph panel has 70/80 proper steps and 10/80 full-M fallbacks. Graph/random subsets coincide on 19/80 steps: ten full-M and nine proper. All three Tree modes have bitwise identical retained physical, prediction, error and call arrays for 0277-start0 and 0291-start0; joint/graph also match those arrays for 0687-start1. Elapsed timing and selection metadata are not asserted equal. Only 0277-start0 uses full-M at every step, so equal trails elsewhere are measured behavior with no inferred mechanism. The stored first-matching support witness is not a unique selected phase/group. Nominal block dimension k−1 versus M−1 does not measure cap-face Jacobian rank or executor savings.

Initial uniform-allocation heat-tangent ranks are Tree and Pair 2/2, 4/4, 6/6 and 6/9 at M3/5/7/10. Tree's finite condition numbers are 1.2237/33.4137/694.0294; Pair's are 1.2688/40.4899/1165.6990. M10 has six observations for nine free heat coordinates and no full-rank condition number. These probes concern that initial point, not every cap face. M5 held failures occur despite full numerical rank.

Actual calls are 945 attempted forwards and 368 VJPs: Tree 711/264 (including shared rank 4/24), Pair 234/104 (rank 4/24). Tree's 320 topology-invalid forward exceptions split 104/127/89 across joint/graph/random. There are zero unknown/coding/runtime failures and zero failed VJPs. All failed and rejected calls remain charged. Tree records 6,977 organizer active-set projections and 8,784,432 attempted density entries. Heat-simplex call counts are source-reconstructed mathematical calls rather than instrumented kernels; coordinate blocks retain the full fine executor. The Tree process lasts 176.798 seconds and Pair's 48.256 seconds; nested cumulative native timestamps are not independent durations to sum. The sealed Pair8 role was reused without recomputing its ranks.

![All 32 inverse trails against attempted native work, caps, sum residuals and initial identifiability](../../diagnostics/generated/tree_lite_20261004/figures/inverse_final_panel.png)

**Figure 6.** Panels A/B show median and IQR of observed/held RMSE relative to each trail's start against charged forward+VJP counts, with individual trails and no endpoint extrapolation; shared rank costs are reported separately above. Panel C retains accepted/rejected updates and 104/127/89/0 invalid trial counts; trials are not updates. Panels D/E show retained source amplitudes, fixed caps and signed public-sum residuals. Panel F shows initial rank 2/2, 4/4, 6/6, 6/9. The table reports absolute equal-trail means, a different aggregation from the plotted relative medians. Native temperature/source units and the numerical thermal reference apply; neither sampler diversity nor cap feasibility proves valid designs.

[Saved32 audit](/data/wanglz/ModularDT/thermal_development/tree_lite_20261004/final_inverse/saved_audit_final32.json) and [independent raw-array audit](/data/wanglz/ModularDT/thermal_development/tree_lite_20261004/final_inverse32_independent_saved_audit.json) reconcile bindings, public starts, acceptance, caps, errors, work and graph policy without model replay.

## Actual exposure and cost scopes

The completed current round, starting 13:05:36 UTC, comprises Tree201–1000, Pair201–1000 and Fine1–500: **315,000 training case visits, 8,400 ordinary optimizer updates, 322,560,000 recorded primary training fluid queries and 47,308,800 primary validation fluid queries**. Each accepted epoch visits 150 training cases and 22 validation cases; their query counts are 153,600 and 22,528 respectively. The validation total is 46,200 case visits, with zero optimizer updates. These are repeated visits to the fixed 150/22 memberships. The historical Tree/Pair1–200 parents add 60,000 training visits, 1,600 updates, 61,440,000 training queries and 9,011,200 validation queries to their full lineage; they are outside this current-round total. Dense1000 and Tree-L100 remain separately labelled historical controls.

| Current-round arm/range | Stored-response examples / wrapper calls / logical role queries | Heat-null wrapper calls | Heat-null primary fluid queries | Heat-null logical role queries | CSV callback coverage |
|---|---:|---:|---:|---:|---|
| Tree201–1000 | 1,600 /1,600 /7,111,520 | 3,200 | 819,200 | 2,492,960 | 800/800 epochs |
| Pair201–1000 | 1,600 /1,600 /7,111,520 | 3,200 | 819,200 | 2,492,960 | 800/800 epochs |
| Fine1–500 | 800 /800 /3,555,760 | 1,600 | 409,600 | 1,247,840 | 400/500; epochs1–100 unscheduled |

These response/null CSV counters cover scheduled callbacks, invoked once on the final microbatch for epochs >100. Their recorded current-round sums are 4,000 response examples/wrapper calls and 17,778,800 logical response-role queries, plus 8,000 null wrappers, 2,048,000 null primary fluid queries and 6,233,760 null logical role queries. Fine's first 100 epochs have no scheduled callback entries; this is disclosed coverage, not a fabricated zero measurement. Logical query counts do not count all physical source reads, fine executor rows, autograd replay/VJPs or kernel work. Calibration, retained-checkpoint evaluations, engineering probes, execution benchmarks and final response/organizer/inverse work have separate receipts. No comprehensive query or physical-read total is obtained by adding these unlike scopes.

Accepted exposure excludes discarded/replayed work. Current-round assigned-process allocation retains failed probes, loading, evaluations and other charged lifetimes; final allocation is reported from the frozen current ledger. The older failed Pair101 and interrupted Tree101–103/partial104 attempts occurred before 13:05 and retain **0.298260406 associated process-hours** in their historical ledger; no old cost is transferred into this round. That figure includes measured intervals and a declared conservative console-bound start proxy. Associated process-hours, per-GPU interval unions, elapsed wall time and active-GPU compute are distinct; active-GPU compute is unmeasured. Full-process timings include cold loading and preparation, while nested diagnostic/attempt timestamps are not added as independent process durations.

[Actual native exposure reconciliation](/data/wanglz/ModularDT/thermal_development/tree_lite_20261004/final_native_exposure_counter_reconciliation.json).

The frozen final science ledger records **13.57683269984 aggregate GPU-associated process-hours of 18**. Device interval unions are GPU1 2.81984573530 hours and GPU2 9.54321913083 hours; their sum is 12.36306486613, and the union across both devices is 9.90420530339 hours. At the root science audit, elapsed time was 10.37818778647 of 12 hours; CPU report closeout continued afterward. These values are not active GPU compute.

The 76 ledger intervals comprise 72 uniquely identified PID allocations and four conservative early windows without exact PID lifetimes. Two Fine schema attempts share a 157-second window, and two Pair telemetry/preflight attempts share a 160-second window; each original attempt remains separately charged. Those bounds are not deduplicated into fictitious exact durations. All eleven final Tree native PIDs occur once and cover full spawn-to-exit lifetimes; nested receipt aliases are deduplicated. Final science finished before the 00:05:36 UTC cutoff, with final CPU closeout due before 01:05:36 UTC. [Independent science budget audit](/data/wanglz/ModularDT/thermal_development/tree_lite_20261004/root_final_science_budget_audit.json).

## A/B/C decisions and evidence index

**A — predictor.** Extra maturation produces a selected900 fluid-temperature advantage over trained Pair and improves 14/24 mean roles. Material/interface gains are modest and case dependent, with negative material median and M10 misses. Streamwise velocity and vorticity remain worse. Dense wins all eight core means; its context/backbone/objective differences prevent attribution to grouping alone. The exposed 22 cases support these bounded descriptive results, not full-population or independent-test claims.

**B — organizer.** Same-weight projected execution, one-pass local shadow and measured time/allocation savings are established in the engineering scopes. Membership-local content exclusions are measured, while all-source planning and phase ancestry remain. Fine physical rows are not pruned. Trained control reliance is stronger evidence than grouping uniqueness; geometry-matched action-reference utility is essentially tied. Eight physical invariance failures and the hook-restoration guard failure prevent a complete faithfulness/restoration claim.

**C — responses and inverse.** Known-null leakage remains nonzero and isolated weighted auxiliary forces are weak. Selected models outperform zero-change response on one stored training family, with no held nonzero physical truth. All 32 bounded inverse trails are complete, with feasible retained inputs, observed/held misses, rank limits and charged invalid trials. Neither clear graph-block advantage, unique source recovery nor independently valid designs is established.

| Figure | Retained PDF master | Numerical evidence |
|---|---|---|
| 1: learning and allocation | [PDF](../../diagnostics/generated/tree_lite_20261004/figures/01_learning.pdf) | Saved native metrics,35 all-22 panels and frozen ledger |
| 2: matched500 fields/residuals | [PDF](../../diagnostics/generated/tree_lite_20261004/figures/figure2_physical_fields.pdf) | Exact500 fixed four arrays and masks |
| 3: selected organizer | [PDF](../../diagnostics/generated/tree_lite_20261004/figures/figure3_organizer.pdf) | Actual selected900 graphs/support/work and conditional QE effect |
| 4: runtime and memory | [PDF](../../diagnostics/generated/tree_lite_20261004/figures/figure4_runtime.pdf) | Profiler-off interleaved samples and separate kernel traces |
| 5: null/force/response | [PDF](../../diagnostics/generated/tree_lite_20261004/figures/figure5_response_null_force.pdf) | Selected-null22, three force points and TRAIN0348 atlas |
| 6: bounded inverse | [PDF](../../diagnostics/generated/tree_lite_20261004/figures/inverse_final_panel.pdf) | All 32 sealed trails, retained states and attempted-call ledger |

Campaign artifacts are local at `/data/wanglz/ModularDT/thermal_development/tree_lite_20261004`. Detailed validation field/graph exports use only the fixed four representatives; all 22 validation statistics and rebuild checks use the fixed panel. Engineering low/high-M proofs and the stored TRAIN0348 atlas retain their separately labelled cases. The [49-item completion matrix](/data/wanglz/ModularDT/thermal_development/tree_lite_20261004/final_completion_matrix.md) maps the plan to measured receipts and misses. Six PDF masters and six small PNG companions remain in ignored presentation paths for direct Markdown display; scientific arrays, checkpoints and compressed profiler traces are preserved. One-time drivers, generated figures/data and run outputs are excluded from Git. Figure links are local artifacts, not remotely published images.

Run roots under `/data/wanglz/ModularDT/thermal_development/HONF_Forward_Runs/ThermalChannel/HONF_Forward_Runs` are `Run_3302_20261004_091947_thermal_tree_lite25_local_context_shadow_v1`, `Run_3202_20261004_004520_thermal_faithfulness25_pair-f_v1`, `Run_3301_20261004_094441_thermal_tree_lite25_fine-f_v1` and the read-only `Run_3101_20261004_003323_thermal_development25_b-native_v1`. Tree's exact200 parent is `Run_3204_20261004_005022_thermal_faithfulness25_tree-f_v2_tie_repair`. Dataset fingerprint is `4224093c22a67af4adfecc8b21d53548e4263ec2254c230dc83c89526b36da05`; source_local_v3 uses train-fixed heat scale 1.7748545408248901 and zeroes four aggregate-heat context slots. No membership, normalizer or schedule crossing is accepted.

Durable native source is `5f4ab424`; earlier lean/local revisions are `1c152b8`/`0c71fd7`, strict normal-action replay `b302cef`, and explicit derivative participation `5f4ab42`. A root suite passed 206 with two optional resource skips. Topology 51/3, replay 80/1 and usage 41/1 are affected checks with overlap, not independent totals. Native measured proofs and failures are reported separately. Final Markdown image/table rendering, source/evidence links and full outgoing-history/artifact-hook audit are retained in local closeout receipts. Durable code, tests and documentation are committed and pushed on `agent/honf-core-next`; generated evidence remains local.

The authorized training and measurement horizons are complete; no continuation command is needed. No Wind training, full-data expansion, new solve, forced-K diversity, inverse-head training or formal run was launched. Future experiments require a separately stated question and budget. This report recommends targeted numerical/dependency and conditioning studies before independent physical response validation.
