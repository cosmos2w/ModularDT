# Thermal shared-core campaign: conclusions at 500 epochs

The campaign put three learned interaction organizers into the native Thermal model and compared them with two fresh controls. All five completed genuine 500-epoch training. The organizers produced different, measurable interactions, but they did not improve the main physical errors over the controls or make subset execution faster. Inverse experiments established execution and conditioning readiness, with mixed one-step outcomes; they did not establish inverse-design quality or generative diversity.

This report concludes the requested five-arm comparison at 500 epochs. Native, Fine and Local later completed 1,000 epochs; those results are supplementary. Tree was intentionally stopped at 666 following the user's decision to end further training. Its later recovery history does not change the preserved 500-stage comparison. No paired-head training or 5,000-epoch training was launched.

## Comparison and evidence

Every retained epoch covers 600 distinct training cases, 75 microbatches, 13 optimizer updates and 614,400 primary sampled queries. Each arm therefore has 300,000 training case visits and 6,500 updates at epoch 500. Validation covers 90 cases per epoch. All arms use the same packed dataset, frozen Stage-A local surrogate, predicted ports and P0/P1/P2 coupling. A common physical-loss denominator and response amendment starts at epoch 101; its optimizer, RNG, calibration and absolute 5,000-epoch schedule continue across stage boundaries. Capacity differs: Native retains coarse/local contexts; Fine and the H arms use the shared fine reader.

| Arm / run | Exact fairness parent | Predeclared field selection | Separate temperature selection |
| --- | ---: | ---: | ---: |
| B-native / 2201 | 500 | 456 | 492 |
| B-fine / 2202 | 500 | 493 | 493 |
| H-tree / 2203 | 500 | 467 | 467 |
| H-overlap / 2204 | 500 | 292 | 269 |
| H-local / 2205 | 500 | 388 | 388 |

Field selection minimizes native sampled validation field MSE within the first 500 epochs. It is not a selection by the physical quantities below, and T-selected parents are not substituted. Exact and selected evaluations each contain 90 finite normal predictions. The primary comparison is canonical 89-case, excluding case 0273; compatibility 90-case is saved separately. Both are previously exposed development data. Mature Run1804 selected epoch 4738 is an unequal-age contextual reference, not a fresh matched-age control.

Units remain native benchmark scales; dimensional SI conversions are unverified. References come from the stored analytic/shared-grid benchmark, with no new CFD solves. The analytic q-normal quantity is a proxy. Full-field section pressure differences, response pressure increments and inverse named-endpoint pressure have different reductions.

![Training maturity and retained full-epoch coverage](../../diagnostics/generated/shared_core_campaign_20261002/figures/final_stage500_conclusions/01_training_maturity.png)

**Figure 1.** All five main arms reach 500 genuine epochs and 6,500 optimizer updates. Retained train plus validation elapsed hours are Native 4.344943, Fine 1.915834, Tree 16.762610, Overlap 9.608218 and Local 10.614756; setup, checkpoint/preview IO, interruptions and discarded attempts are excluded. Field-selected ages are 456/493/467/292/388, respectively. Sampled learning curves establish age and training behavior, not full-grid fidelity. Contention, engineering changes and discarded attempts remain separate from successful measured elapsed time; no exclusive-GPU speed ranking follows.

## Predictor: useful learning, persistent physical misses

The native Dense control is the strongest fresh accuracy baseline; the three-term Fine control uses less measured training time with weaker fidelity. The controls have the lowest principal physical errors among the fresh arms. Tree is the strongest new H comparison on the listed thermal roles, but still trails Native and the mature reference. Selection improves several quantities while worsening others. The table gives canonical 89-case equal-case RMSE means; pressure is section-difference absolute error.

| Arm / parent | Fluid T | Fluid u | Surface T | q-normal proxy | Material peak | Pressure AE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native exact epoch 500 | 0.745890 | 0.021589 | 0.724999 | 3.442214 | 0.517528 | 0.004399 |
| Native field-selected epoch 456 | 0.560723 | 0.022163 | 0.672854 | 3.491448 | 0.569771 | 0.003687 |
| Fine exact epoch 500 | 0.849954 | 0.026557 | 0.739772 | 3.750804 | 0.637695 | 0.006770 |
| Fine field-selected epoch 493 | 0.714392 | 0.025453 | 0.804033 | 3.694463 | 0.649535 | 0.007407 |
| Tree exact epoch 500 | 0.881500 | 0.030349 | 0.889281 | 3.615430 | 0.857808 | 0.007706 |
| Tree field-selected epoch 467 | 0.892862 | 0.030216 | 0.730441 | 3.618392 | 0.658864 | 0.006965 |
| Overlap exact epoch 500 | 3.390662 | 0.163987 | 3.122379 | 4.545853 | 2.717840 | 0.014745 |
| Overlap field-selected epoch 292 | 3.147519 | 0.082642 | 2.864735 | 4.371272 | 2.508833 | 0.013362 |
| Local exact epoch 500 | 2.064795 | 0.073531 | 2.014171 | 3.978034 | 1.845799 | 0.020545 |
| Local field-selected epoch 388 | 1.370173 | 0.057135 | 1.121311 | 3.934525 | 1.008235 | 0.015787 |
| Mature field-selected epoch 4738 | 0.218656 | 0.005721 | 0.448389 | 1.582913 | 0.346587 | 0.001014 |

The misses include tails. Native selection raises peak-error p90 from 0.8027 to 0.9405. Tree selection raises fluid-T mean/p90 from 0.881500/1.187148 to 0.892862/1.198006, despite reducing material-peak mean from 0.857808 to 0.658864. Overlap T p90 increases 4.280530→4.416176. Local selection improves T p90 2.845462→1.976316, while worsening velocity and pressure maxima. All 24 roles, pooled errors, M/Re strata and maxima are retained in the [saved numerical extraction](../../diagnostics/generated/shared_core_campaign_20261002/tmp/five_arm_stage500_final_scope_extraction_v1/cohort500_saved_extraction.json).

![High-module-count physical fields and residuals](../../diagnostics/generated/shared_core_campaign_20261002/figures/final_stage500_conclusions/02_physical_case0692.png)

**Figure 2.** Case 0692, M10: selected Native/Fine/Tree/Overlap/Local fluid-T RMSE is 0.64588/0.70688/0.90267/2.0917/1.5266; mature epoch 4738 is 0.17473. T, u and p predictions and residuals use identical grids, masks and references, with common unclipped scales across both displayed cases.

![Low-module-count physical fields and residuals](../../diagnostics/generated/shared_core_campaign_20261002/figures/final_stage500_conclusions/03_physical_case0274.png)

**Figure 3.** Case 0274, M3: selected fluid-T RMSE is 0.42269/0.43130/0.51746/2.1710/1.3192; mature epoch 4738 is 0.12925. All 36 displayed model/channel RMSE checks match saved metrics.

Responses add a distinct test of change, rather than absolute-field accuracy. Every exact and field-selected parent completes eight existing families, 88 absolute states and 80 correlated finite perturbations. These previously exposed families include calibration and final-review records; they are not 80 independent experiments.

| Arm | T response RMSE, exact→field | Pressure increment AE, exact→field |
| --- | ---: | ---: |
| Native | 0.135131→0.137524 | 0.000371→0.000381 |
| Fine | 0.144669→0.146574 | 0.000260→0.000267 |
| Tree | 0.167999→0.185696 | 0.000611→0.000569 |
| Overlap | 0.309989→0.260419 | 0.001925→0.000704 |
| Local | 0.273792→0.215254 | 0.001146→0.000630 |

![Matched finite-response evidence](../../diagnostics/generated/shared_core_campaign_20261002/figures/final_stage500_conclusions/05_responses.png)

**Figure 5.** Mature T response RMSE is 0.116739; zero-change is 0.274135. Selection improves Overlap/Local response T but worsens Native/Fine/Tree. All arms retain spurious flow or pressure changes on heat-transfer variants whose stored flow/pressure increments are exactly zero. Successful heat-only T/q-proxy responses therefore do not validate physical coupling generally. The next predictor work should improve full-field and heat-null fidelity under matched controls, rather than infer success from sampled validation loss alone.

## Organizer: conditional utility without measured speed gain

The H organizers produce actual typed MM/ME/EM/QM/QE plans throughout P0/P1/P2. Population exports contain 1,350 phase-by-route records per selected parent. Memberships are densities with native source measures, not binary incidence. Repeated binary supports are common; exact membership-plus-control duplicates are zero in the reported populations, so support-only deduplication is not justified.

The exported organizers also have different admission behavior. Overlap allocates eight proposals; selected nonempty groups vary in P0 and remain eight in P1/P2. Local nonempty K spans one to eight, with phase means 7.255556/7.011111/6.633333. Its six admission-rescue preparations occur across five M10 cases; they are separate from inverse or full-access fallback. Tree admission-rescue counts are unavailable in these exports and cannot be recorded as zero. Near-access and far-control roles remain explicit in Local. These recorded populations describe learned actions under the existing inputs, while their physical utility is assessed separately below.

A four-case strict panel holds reference actions authoritative. Normal Tree actions beat fixed-access zero controls and full access retaining learned controls on all 24 equal-case role means. Overlap wins 23/24 against both; Local wins 20/24 and 21/24. Bounded geometry changes preserve audited support degrees, exact-measure row permission multisets and protected near/invalid entries, but have mixed physical effects. These are same-weight conditional utility findings on four cases, not global assignment optimality or physical causality.

![Actual organizer actions, work and latency](../../diagnostics/generated/shared_core_campaign_20261002/figures/final_stage500_conclusions/04_graph_utility_work.png)

**Figure 4.** Case0692 actual Tree467 has eight active QE groups at each phase; the displayed group contains 39 environmental sources in each phase. Local388 has one active group with 93/93/94 environmental sources. These are measured memberships, separate from protected near permissions. Dense-reference H field evaluations execute 179,889,120 padded fine-MLP rows in 14,490 calls over 90 normal cases. Rectangular subset execution reduces actual rows, yet every measured H500 complete-wrapper timing contrast is slower. Selected full-Q complete medians, dense→subset seconds, are Tree M3 0.474446→0.516765 and M10 0.478300→0.520595; Local 0.523554→0.571946 and 0.522002→0.575792; Overlap 0.954593→1.142509 and 1.041834→1.285353.

Two small-query prepared-only contrasts are exceptions: M10/Q14 decode improves 0.010322→0.009727 s for selected Overlap epoch 292 and 0.007433→0.007372 s for exact Local epoch 500. These narrow results do not overturn the slower complete-wrapper or full-Q measurements. Native output and first-gradient parity gates pass before timing at declared tolerances. Complete-wrapper and prepared-P2 scopes are separate; they cannot be added. Actual fine rows exclude attention, organizer/coarse/local work, backward and export. Padding, reserved cache and driver memory are distinct. Fixed executor order, two exposed layouts and different device/host contention prevent an isolated cross-arm latency ranking. Logical support reduction is consequently not reported as a speedup. The next organizer work should address representation sensitivity and executor overhead before committing to longer training.

Tree also has a representation miss: fixed-prepared environmental atom splitting passes, whereas organizer-rebuild splitting fails all four case/parent checks. Maximum hidden-context differences are 0.0217912/0.0101552 for exact epoch 500 and 0.0110745/0.00674516 for selected epoch 467. These are fine hidden-context units, not temperature errors. The source-chain explanation is a hypothesis; the experiment does not establish continuity under representation changes.

## Inverse: completed readiness, missing quality protocol

At the 500 boundary, each evaluated parent completes 12 public tasks × one start × one update × three modes: 36 readiness trials. The controls have selected-parent panels; H exact and selected panels are available. All preserve nonnegative supplied-total heat. Nine fixed-total observation Jacobians are full-rank; three M10 tasks have rank 6 in nine free directions. Feasibility and rank do not establish identifiability, useful optimization or valid designs.

| Selected parent, graph mode | Observed RMSE, initial→final | Held RMSE, initial→final |
| --- | ---: | ---: |
| Native epoch 456 | 1.607598→2.196528 | 1.767392→2.054503 |
| Fine epoch 493 | 1.394177→2.098712 | 1.659176→2.263830 |
| Tree epoch 467 | 1.649598→1.486706 | 1.831229→1.966403 |
| Overlap epoch 292 | 3.269133→3.319606 | 3.566268→3.742736 |
| Local epoch 388 | 1.953392→1.794419 | 2.025590→1.807006 |

![Inverse readiness and material-error limits](../../diagnostics/generated/shared_core_campaign_20261002/figures/final_stage500_conclusions/06_inverse_material.png)

**Figure 6.** Tree epoch 467 improves observed error but worsens held error; its size-matched ungrouped held error, 1.945865, is also below graph's 1.966403. Local epoch 388 improves both with graph; ungrouped has lower observed error, 1.751009, but higher held error, 1.878345. Native/Fine graph and ungrouped paths are full-joint fallbacks, not learned graph gains. The displayed M10 allocation remains almost uniform after one step despite uneven stored heat, with observation rank 6 in nine free directions. Material/interface curves use the original forward inputs, not fields generated from the inverse candidates; their errors remain separate from observation matching.

The H panels record 12 proper graph updates per parent with zero full-joint fallback. This nonvacuous operation is useful readiness evidence, but no arm has the 108-trial, three-start, 30-update quality protocol at its 500-stage selection. Mature epoch 4738 and later Native epoch 962/Fine epoch 972/Local epoch 995 have completed 108-trial panels; they cannot fill the missing matched 500-stage comparison. A later inverse study would need matched multi-start optimization, held-sensor and hidden-heat checks, followed by physical-reference validation before any design claim. No denoising head was trained, and public graph-action probes with zero head updates are not Gaussian draws or evidence of valid sampler diversity.

## Delivery, lineage and next steps

The shared interaction implementation and domain-owned physical assembly are documented in the [Thermal/Wind architecture map](../guides/Shared_Core_Thermal_Wind_Architecture_Map.md) and [alignment evidence](HONF_Shared_Core_Alignment.md). Retained native Thermal/Wind replay tests passed without new Wind scientific training. The later soft-permission repair passed 105 affected tests and a captured native eight-case check: ten hard outputs and 132 physical gradients remained byte-identical, while all 53 soft gradients became finite. This later numerical repair is supplementary engineering evidence, not a retroactive replacement of the 500 measurements.

The [chronological status report](HONF_Shared_Core_Thermal_Strategy_Campaign_Status.md) preserves source, optimizer/RNG, calibration, normalization, immutable selection and failure/recovery receipts. Later Native/Fine/Local epoch 1000 results remain labelled there; Tree's intentional stop at epoch 666 is distinct from its earlier epoch 607 numerical failure and recovery. The final 500-epoch conclusion is that Tree offers an informative organizer hypothesis, while the fresh Native control remains stronger on principal physical errors and no H arm demonstrates complete-wrapper subset speed gains or inverse quality.

One Tree 500-stage manual continuation recipe is retained for exploratory use, alongside explicit fresh matched-seed 0 semantics in the [manual guide](../guides/Thermal_Manual_5000_Launch.md). Fifty focused software tests and actual workspace preparation/native parser checks support the handoff. They do not constitute 5,000-epoch training or future accuracy. The [actual preparation/parser receipt](../../diagnostics/generated/shared_core_campaign_20261002/tmp/manual_h_tree500_preparation_20261004/actual_preparation_execution_receipt.json) and [Q14 replay receipt](../../diagnostics/generated/shared_core_campaign_20261002/tmp/manual_h_tree500_Q14_native_replay/bounded_replay_receipt.json) complete the bounded startup check. The CPU1 case 0274 replay exits with code 0, with seven nonempty finite outputs, three disabled zero-width outputs, all 316 saved-state tensors/4,287,933 scalars unchanged and 109 frozen Stage-A tensors. Exact epoch 500 remains the parent, next epoch 501 and horizon 5000; no optimizer, gradient or RNG restoration occurs. This proves native replay, not future optimization, fidelity or convergence. No recipe auto-launches training. Fresh seed 0 is a separate initialization, not an independent-seed replication claim.

Further scientific work would require explicit authorization: matched 500-stage inverse quality, stronger response/physical validation, representation robustness and a parity-preserving executor that actually improves latency. The stopped campaign makes none of those unmeasured claims.

PDF figure index: [1 training](../../diagnostics/generated/shared_core_campaign_20261002/figures/final_stage500_conclusions/01_training_maturity.pdf), [2 M10 fields](../../diagnostics/generated/shared_core_campaign_20261002/figures/final_stage500_conclusions/02_physical_case0692.pdf), [3 M3 fields](../../diagnostics/generated/shared_core_campaign_20261002/figures/final_stage500_conclusions/03_physical_case0274.pdf), [4 organizer/work](../../diagnostics/generated/shared_core_campaign_20261002/figures/final_stage500_conclusions/04_graph_utility_work.pdf), [5 responses](../../diagnostics/generated/shared_core_campaign_20261002/figures/final_stage500_conclusions/05_responses.pdf), [6 inverse/material](../../diagnostics/generated/shared_core_campaign_20261002/figures/final_stage500_conclusions/06_inverse_material.pdf).
