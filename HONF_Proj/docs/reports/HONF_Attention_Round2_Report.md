# Round2 attention: geometry, content and native fields

This round makes geometry part of the information carried by tokens, completes ordered module/environment exchanges inside each attention cycle, and lets six learned contents gather typed donors directly. The new A2-D control and A2-H6 organized model share this preparation, native Thermal/Wind adapters and the existing TrainingEngine. One fresh revision per dataset additionally gives receiver queries the live module/environment centroids of each executed content gather. The complete mathematical and execution contract is the [Round2 plan](../../UpgradePlan/HONF_Attention_Round2_Geometry_and_Content_Development_Plan.md), read with the two Oct13 specifications.

**Predictor.** The direct controls improve strongly over the coherent first-round A-D on the same exposed development panels. Organized H6 remains substantially worse than equally trained A2-D. The centroid revision improves some aggregate/role errors but does not close that gap: literal epoch 2500 Thermal fluid-T RMSE is 1.85330 versus 0.949959 for A2-D; Wind near/downstream vector means are 0.686129/0.528404 m/s versus 0.273712/0.185800 m/s. All six fits reached 2500 with healthy finite optimization, but neither organized candidate qualifies for a fresh full-TRAIN 1000 run. The next predictor question is how the compressed far path can retain the source-specific information already useful to the direct decoder.

**Organizer.** Geometry, typed W, content reads, four receiver feature heads and the bounded local path actually execute. Same-weight access/content/local interventions measurably change native outputs. Full six-edge access is retained; fixed-B8 backward memory is lower than the direct control at the measured densities. However, active or differentiated gathers do not establish useful physical grouping: content similarity remains high, field accuracy is worse, and some removals improve an error. The next organizer question is which content/readout restriction loses predictive information, rather than whether all six slots are occupied.

**Inverse.** The field-selected Thermal models complete the existing three-family/six-signed-change response audit without new solves. Precise FP64 increments agree with their own affine endpoints, and response errors improve over retained first-round A-D on these saved references. H6 R1 remains worse than A2-D in all three displayed families. The reference is saved analytic-wake/shared-grid data, not CFD; there is no inverse generator, optimization campaign, new Wind geometry label or physical causality claim. Solver usage stays 326/326. A later inverse study needs separate physical validation and authorization.

## Decision and literal epoch 2500 endpoints

| Dataset | Model, literal epoch 2500 | Native broad score | Priority mean RMSE |
| --- | --- | --- | --- |
| Thermal | First-round A-D | 0.06239995 | fluid T 2.47444 |
| Thermal | A2-D | 0.01356915 | fluid T 0.949959 |
| Thermal | A2-H6 primary | 0.05497920 | fluid T 1.84244 |
| Thermal | A2-H6 R1 | 0.04943476 | fluid T 1.8533 |
| Wind | First-round A-D | 0.19483390 | near 0.64306; down 0.455483 m/s |
| Wind | A2-D | 0.04511322 | near 0.273712; down 0.1858 m/s |
| Wind | A2-H6 primary | 0.23022964 | near 0.671186; down 0.550186 m/s |
| Wind | A2-H6 R1 | 0.22082518 | near 0.686129; down 0.528404 m/s |

The frozen Section10 criteria are research improvement criteria, not classic-replacement or inverse-readiness gates. Thermal R1 improves fluid T by 25.10% and broad score by 20.78% over first-round A-D, and passes its old-baseline nonpriority/p90 guards. It fails the matched-control requirements: fluid/broad ratios are 1.95092/3.64317, above 1.15. Wind R1 has a six-component near/downstream ratio geometric mean of 0.99510 rather than at most 0.90, broad ratio 1.13340 rather than at most 1, and matched near/downstream vector ratios 2.50675/2.84394 rather than at most 1.15. Its downstream row-RMSE p90 ratio to old A-D is 1.26477, above 1.25. Those are measured misses; no difficult scene is removed and no bound is relaxed.

| Candidate | Full-TRAIN1000 status | 5000 status |
| --- | --- | --- |
| Thermal A2-H6 primary / R1 | Not launched: both fail matched native fields | Not launched; executable stop guard rejects 5000 |
| Wind A2-H6 primary / R1 | Not launched: broad/priority and tail criteria fail | Not launched; executable stop guard rejects 5000 |

Both revised candidates pass the measured high-M complete-call engineering ceiling: Thermal 1.52952× and Wind 1.78032× the retained first-round H6 in current paired replays. All six required density points per dataset complete. Correctness, content use, cost and resource availability do not override the failed field decision. Full recipes remain inert, with fresh full-TRAIN normalization/weights/moments and a prospective 5000 schedule but an enforced1000 execution stop before allocation. No full-data potential claim or matched new full 1000 pair is invented.

## Scientific identity, shared math and exposure

Thermal uses fixed25 TRAIN150/exposed DEV22 with seed 0, Q1024 primary fluid queries, effective batch 48 and microbatch 24: each epoch has four optimizer updates, seven microbatches and a six-case final update. Wind uses fixed24 TRAIN72/exposed DEV24, seed 42, Q4096 total across five native roles, effective batch 24 and microbatch 12: three updates and six microbatches. DEV24 is eight layouts with three native directions each. Thermal DEV22 is the protocol-exposed portion of the original source TEST catalogue, not an independent hidden test. Other Thermal TEST targets and Wind TEST targets remain locked.

Every pair reuses the same input-only fixed memberships, TRAIN-fitted transforms, name-based common-parameter initialization, case order, deterministic queries, objectives and schedule. All 2500 epochwise case/query hashes match A2-D for both primary and revised H6. AdamW uses 20-epoch warmup, peak 3e-4, hold through 1000 and cosine decay to 3e-6 at 2500; weight decay is 1e-4 Thermal/1e-5 Wind with explicit bias/norm exclusions and clipping 1. Thermal keeps four flow weights 1/8, three temperature weights 1/6, q-proxy 0.05 and the original TRAIN response addendum 0.1 once in the final macro-update. Wind retains its balanced five-role/component objective. An epoch visits every selected case, not the full source dataset.

The shared context is width 128, four heads and two ordered MM → EM → EE → ME cycles, with one token FFN per type/cycle. Smooth raw-plus-three-frequency coordinates reach values and score geometry. Q/K are head-normalized, the live learned scale is initialized at sqrt32 and bounded to[1,32], SDPA uses scale 1, and empty/masked rows are exact zero. Separate measure-normalized M/E scene pools include operation, actual module count and typed physical mass/size without environment sample count or mixed-unit mass sums. Raw/scene reinjection gates start at 0.1. There is no dense width 128 pair-value contextualization, frozen D-sep or Stage-A dependency.

Six spatial contents perform two shared-parameter, directly donor-normalized typed W reads with an intervening H×H residual and first-read residual. The receiver evaluates one 128-wide query–content message, split into four 32-wide heads, full-six access and one signed Thermal mode per head. Thermal uses C=W_M*kappa with physical donor measure entering W once, then contracts C*h before query expansion; ordinary prediction/increments/adjoints do not materialize K[Q,M]. A2-D keeps the uncompressed signed per-source coefficient without a heater softmax. Local support remains Thermal 3r/Wind 1.5D, with all overlaps and native compact windows. Learned scene summaries reach H6 receivers only through contents or explicit local module states; there is no hidden dense decoder bypass.

The adapters preserve native physical extraction. Thermal has fresh heat-independent u/v/p heads, physical native curl, exact affine heating and T(0)=0; surface/outside/material and conductivity/spacing-dependent q derive from the same temperature field without adding a temperature mean. Wind keeps height-profile plus normalized residual times safe_std*u_ref, without adding an absolute normalizer mean or rotating its already aligned frame again. Classic references retain their original input factory, E catalogue and transforms.

The sole substantive revision, `typed_w_centroids_v1`, adds live final-W module/environment coordinate first moments to query-relative geometry alongside the original anchor. It applies no extra measure or renormalization, keeps typed validity masks and exact empty zeros, and adds neither contents nor a second large reader. It is a fresh 2500-epoch fit with new weights/moments; no parent epochs are added to its age. The motivation is measured anchor/gather displacement—up to 1.438 edge-scale units in the Wind M30 representative and 1.263 in Thermal M5—not a demonstrated unique cause of the field error. A weighted centroid does not identify ownership of diffuse or multimodal content.

| Dataset | A2-D parameters | Primary H6 parameters | H6 R1 parameters |
| --- | --- | --- | --- |
| Thermal | 1,520,493 | 1,807,821 | 1,811,727 |
| Wind | 1,525,675 | 1,781,264 | 1,787,090 |

## Learning, maturation and selected checkpoints

| Completed2500 fit | Case visits | Updates | History TRAIN h | Training-process h | Field-selected age |
| --- | --- | --- | --- | --- | --- |
| Thermal A2-D | 375,000 | 10,000 | 1.017 | 1.065 | 1700 |
| Thermal A2-H6 primary | 375,000 | 10,000 | 1.026 | 1.070 | 1500 |
| Thermal A2-H6 R1 | 375,000 | 10,000 | 0.929 | 0.966 | 1200 |
| Wind A2-D | 180,000 | 7,500 | 0.973 | 1.021 | 1600 |
| Wind A2-H6 primary | 180,000 | 7,500 | 0.869 | 0.912 | 700 |
| Wind A2-H6 R1 | 180,000 | 7,500 | 0.638 | 0.672 | 800 |

Training-process time charges exact-run completed and failed GPU-associated training segments, including setup, monitoring and save overhead. History TRAIN-loop time is a narrower overlapping scope; never add them. Thermal primary-query exposure is 384,000,000 per fit plus native auxiliary/response work; Wind is 737,280,000 receiver queries per fit. Thermal has 17,500 microbatches and 15,000 partial-update case visits; Wind 15,000 microbatches and no tails. No optional fusion screen, second seed, sparse training campaign or second substantive revision is run.

![Thermal development learning, fixed-weight TRAIN replays and separately measured work/time](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/learning_final_v2/honf_round2_learning_work_thermal.png)

Figure 1a. All Thermal fits reach 2500. Literal fixed-TRAIN broad scores are 0.00228088/0.00432518/0.00417222 for D/primary H6/R1, while R1 native DEV broad remains0.04943476. Shared saved-weight ages 100/500/1000/2500 distinguish fitting from monitor changes; the late TRAIN/DEV gap is measured, not proof of one architectural cause.

![Wind development learning, native role monitors and measured training work/time](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/learning_final_v2/honf_round2_learning_work_wind.png)

Figure 1b. Wind fixed-TRAIN broad scores fall to 0.00640355/0.03592058/0.03237188 for D/primary H6/R1, while organized DEV errors rise after their early minima. Old v1 fixed-TRAIN evidence exists only at 500 for Wind; the missing ages are left missing rather than replayed or interpolated.

The common selector uses the lowest saved broad-field monitor at 100-epoch checkpoints, never the subsequent saved-response audit. Actual native selected broad scores are Thermal D e1700:0.01189792, primary H6 e1500:0.03912227, R1 e1200:0.04387094; Wind D e1600:0.04287159, primary H6 e700:0.16052818, R1 e800:0.16704920. These are separate from literal epoch 2500 qualification. The R1 Thermal selected fluid-T mean is 1.90449. There is one seed and exposed DEV selection; these are neither confidence bounds nor independent test results.

## Native fields, all roles and difficult groups

All tables use arithmetic means of casewise Thermal or rowwise Wind RMSEs. Reported p90 is a percentile of case/row RMSE, not a pointwise-error percentile. Thermal flow/temperature values remain packed-dataset native units; Kelvin or SI flux units are not established. q is the maintained native normal-flux proxy. Wind is in m/s. Broad scores are dimensionless native standardized objectives; do not compare their magnitudes across datasets. Complete per-case/row medians, p90, worst errors and group IDs remain in the linked evidence index.

### Thermal literal epoch 2500 role RMSE

| Role | First-round A-D | A2-D | A2-H6 primary | A2-H6 R1 |
| --- | --- | --- | --- | --- |
| u | 0.0617148 | 0.0247372 | 0.0549807 | 0.0510697 |
| v | 0.0103755 | 0.00334295 | 0.00703636 | 0.00665861 |
| p | 0.029654 | 0.0129638 | 0.0290937 | 0.0285667 |
| omega | 0.167174 | 0.0574021 | 0.132672 | 0.123775 |
| temperature_fluid | 2.47444 | 0.949959 | 1.84244 | 1.8533 |
| temperature_surface | 2.03836 | 1.03814 | 2.07676 | 2.01184 |
| temperature_material | 1.9225 | 1.18359 | 2.18455 | 2.01209 |
| q_proxy | 4.29254 | 3.81651 | 4.21692 | 3.43183 |

| Statistic | First-round A-D | A2-D | A2-H6 primary | A2-H6 R1 |
| --- | --- | --- | --- | --- |
| Native broad field score | 0.06239995 | 0.01356915 | 0.05497920 | 0.04943476 |
| temperature_fluid case/row RMSE p90 | 3.52741 | 1.52634 | 3.06868 | 2.71881 |

| Module count | Cases/rows | First-round A-D | A2-D | A2-H6 primary | A2-H6 R1 |
| --- | --- | --- | --- | --- | --- |
| 3 | 6 | 2.24831 | 0.679336 | 1.33842 | 1.39088 |
| 5 | 6 | 2.69108 | 0.943071 | 2.20487 | 2.3151 |
| 7 | 6 | 2.55182 | 1.10604 | 2.08747 | 2.01151 |
| 10 | 4 | 2.37261 | 1.1321 | 1.68729 | 1.61688 |

Thermal module-count rows above give fluid-T mean RMSE on all 6/6/6/4 cases for M=3/5/7/10. R1 remains worse than A2-D in every group; its modest advantage in the displayed M10 case does not generalize to that whole group. Surface/material/q improvements on a particular module are not a pass for fluid fields. The first-round comparison retains its exact historical recipe, including the original bias/norm decay declaration discrepancy; it is not silently relabelled as the new matched protocol.

![Thermal case0291 native temperature and transverse velocity with signed D and H6 R1 errors](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/thermal_native_final/thermal_family2_a2-h6-r1-centroids_case0291_preview.png)

Figure 2a. Exposed DEV0291, M5, literal epoch 2500, identical128×64 grid and 7,837 valid fluid cells. D/R1 fluid-T RMSE is 1.1793/1.4560 and relativeL2 is 0.2205/0.2722; v RMSE is 0.003315/0.006572. Shared physical/error scales retain the visible wake-shape and transverse-velocity miss.

![Thermal case0687 native fields and residuals on identical saved masks](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/thermal_native_final/thermal_family2_a2-h6-r1-centroids_case0687_preview.png)

Figure 2b. Exposed DEV0687, M10, literal epoch 2500,7,463 valid fluid cells. R1 improves this scene’s fluid-T RMSE to 0.64528 from D 0.76954, while v worsens to 0.004679 from 0.002769. This input-selected example illustrates a mixed local result, not the all 22-case decision.

![Native surface temperature and normal-flux proxy versus saved angle for module0 in two Thermal scenes](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/thermal_native_final/thermal_family2_a2-h6-r1-centroids_surface_detail.png)

Figure 2c. Source-selected module0 preserves64/64 native angle rows without interpolation. D/R1 surface-T RMSE is 1.6738/1.0318 on0291 and 0.27321/0.19165 on0687; q-proxy RMSE is 5.3385/1.5194 and 1.3101/0.91158. These are selected-surface gains; all-role means and fluid misses remain above.

### Wind literal epoch 2500 role RMSE

| Role | First-round A-D | A2-D | A2-H6 primary | A2-H6 R1 |
| --- | --- | --- | --- | --- |
| volume | 0.187687 | 0.0753883 | 0.229468 | 0.219507 |
| hub_slab | 0.39786 | 0.15141 | 0.489982 | 0.468637 |
| downstream_envelope | 0.455483 | 0.1858 | 0.550186 | 0.528404 |
| near_turbine | 0.64306 | 0.273712 | 0.671186 | 0.686129 |
| background | 0.0672803 | 0.0331695 | 0.0860462 | 0.0818913 |

| Statistic | First-round A-D | A2-D | A2-H6 primary | A2-H6 R1 |
| --- | --- | --- | --- | --- |
| Native broad field score | 0.19483390 | 0.04511322 | 0.23022964 | 0.22082518 |
| near_turbine case/row RMSE p90 | 0.671212 | 0.346214 | 0.740527 | 0.805453 |
| downstream_envelope case/row RMSE p90 | 0.485805 | 0.223617 | 0.617209 | 0.614433 |

| Module count | Cases/rows | First-round A-D | A2-D | A2-H6 primary | A2-H6 R1 |
| --- | --- | --- | --- | --- | --- |
| 8 | 3 | 0.59439 / 0.44352 | 0.328742 / 0.201023 | 0.662673 / 0.556436 | 0.589335 / 0.482387 |
| 11 | 3 | 0.664662 / 0.466368 | 0.247261 / 0.164537 | 0.664497 / 0.532552 | 0.650195 / 0.492432 |
| 13 | 3 | 0.626879 / 0.450375 | 0.22733 / 0.1526 | 0.683361 / 0.598033 | 0.630217 / 0.516848 |
| 18 | 3 | 0.659278 / 0.47135 | 0.232869 / 0.155842 | 0.608024 / 0.526514 | 0.663634 / 0.531362 |
| 21 | 3 | 0.657294 / 0.425393 | 0.322951 / 0.220228 | 0.751548 / 0.561315 | 0.802271 / 0.524855 |
| 24 | 3 | 0.655845 / 0.450299 | 0.292996 / 0.21942 | 0.692171 / 0.499345 | 0.701991 / 0.50751 |
| 27 | 3 | 0.643427 / 0.454043 | 0.27728 / 0.191232 | 0.689186 / 0.588808 | 0.762275 / 0.611919 |
| 30 | 3 | 0.642708 / 0.482513 | 0.260271 / 0.181515 | 0.618029 / 0.538482 | 0.689116 / 0.559916 |

### Wind priority components (m/s)

| Role / component | First-round A-D | A2-D | A2-H6 primary | A2-H6 R1 |
| --- | --- | --- | --- | --- |
| near_turbine/Ux | 0.630274 | 0.265325 | 0.661686 | 0.676789 |
| near_turbine/Uy | 0.109927 | 0.0561777 | 0.0991199 | 0.0996276 |
| near_turbine/Uz | 0.0643574 | 0.0357168 | 0.0511161 | 0.0504502 |
| downstream_envelope/Ux | 0.449614 | 0.182209 | 0.544389 | 0.522355 |
| downstream_envelope/Uy | 0.0614557 | 0.0298999 | 0.0695924 | 0.0698337 |
| downstream_envelope/Uz | 0.0380487 | 0.0199558 | 0.0368364 | 0.0366818 |

Each Wind module-count row is one layout observed at its three native directions; near/downstream means are printed in that order. These eight small layout groups are descriptive, not eight independent large populations. R1 worsens near errors particularly at M21/M27/M30 and does not approach the direct control. The shared DEV panel contains 8192 receivers per row, four roles of 1640 and background 1632, with SHA594681…028ed. The old copied 820-role metadata is quarantined as a summary defect: actual old arrays/role metrics use 1640 and are never rewritten.

![Native Wind M8 hub-height Ux Uy Uz planes and signed D versus H6 R1 residuals](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/wind_native_final/wind_r1_round2_low_M_native_plane.png)

Figure 3a. Exposed DEV row 69, M8, stored z70.869949m/index16,52,548 native cells, literal epoch 2500. R1/D relativeL2 is 0.0525/0.0190 for Ux,0.7708/0.3217 for Uy and 1.0890/0.6102 for Uz. This is an actual stored plane in the aligned frame, not a projected point cloud or a new CFD solve.

![Native Wind M30 velocity components and matched residual scales](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/wind_native_final/wind_r1_round2_high_M_native_plane.png)

Figure 3b. Exposed DEV row 426, M30,99,750 native cells on the same stored hub plane. R1/D relativeL2 is 0.0651/0.0200,0.6375/0.3050 and 0.8574/0.4217 for Ux/Uy/Uz. Shared per-component physical and error scales expose the larger organized-model residuals.

![Native downstream crosswind cuts at geometry-selected approximately2D4D8D for all Wind velocity components](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/wind_native_final/wind_r1_round2_native_wake_cuts.png)

Figure 3c. Geometry-selected downstream cuts preserve saved native indices near2D/4D/8D, with actual offsets printed. M8 R1 Ux cut RMSE is 0.568/0.509/0.652m/s versus D 0.060/0.079/0.197; transverse components are shown separately. The M30 curves likewise preserve actual positions and do not substitute an interpolated CFD slice.

## Executed contents, heads and response factors

| Thermal all-DEV summary | Cases | Effective M donors | Effective E donors | Off-diagonal content cosine |
| --- | --- | --- | --- | --- |
| thermal primary | 22 | 4.208 | 123.633 | 0.9675 |
| thermal R1 | 22 | 3.444 | 104.661 | 0.9560 |

On all 24 Wind DEV rows, primary/R1 mean effective donor counts are 8.53/12.02 for module W and 28.66/44.93 for environment W; off-diagonal content cosine is 0.545/0.782. These are means over actual six-slot summaries. Receiver-head means use128 evenly spaced input-defined native queries per scene; detailed head accesses and local work are retained. R1 spreads its reads more broadly and makes contents more similar on average. Neither concentration nor entropy is a success gate or causal source truth.

![Executed Thermal module and environment gathers through six contents to three receivers with four head accesses and interventions](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/content_final/thermal_content.png)

Figure 4a. Thermal R1 e2500 scene 0687 shows actual W, final-content cosine 0.945, all four head accesses and three input-selected receivers; top3 donor links per type/edge are drawn, while both dense typed context cycles and H×H mixing remain executed upstream. The same-weight uniform/edge0/local interventions change native fluid-T errors on0291/0687. They establish reliance, not useful grouping or sparse savings.

![Actual signed Thermal W-coupled C factors and explicit source-indexed K at declared receivers](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/content_final/thermal_signed_factors.png)

Figure 4b. Actual C=W_M*kappa is displayed for all 6×4 factors and source IDs; explicit K includes the local term at three declared receivers. C is converted once by the TRAIN native-temperature scale7.2807598; K is already physical. C magnitude depends on factorization coordinates and is not physical strength independently of Phi.

![Executed Wind typed gathers content similarity head access and native intervention effects](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/content_final/wind_content.png)

Figure 4c. Wind R1 e2500 scene 426 has actual content cosine 0.800. Removing fixed edge0 changes physical-output RMS by 0.096/0.087m/s on M8/M30; local zero changes it by 0.184/0.176m/s. Edge0 removal slightly improves the M30 broad score while local zero worsens both. All49,152 edge messages and packed local rows still execute under these controls.

Input-only low/high-M TRAIN diagnostics are retained at initialization, 100, 1000 and 2500. Thermal high-M far K has numerical rank 12/12 on the fixed 128-row software panel at rtol 1e-7 even when most energy occupies a few modes; the single-module fixture has rank 1 and necessarily zero between-module variance. This is model response rank, not independently validated physical excitation rank. Nonzero gradients, live centroid gates, nonzero C and full numerical rank rule out several dead-path explanations but do not prove the compressed representation is adequate. No optional sparse executor test is claimed; full-six executed work and latency are reported directly.

## Complete costs and fixed-batch memory

Every GPU command uses ModularDT FP32/TF32-off/four CPU threads and exposes only physical GPU0 for Thermal or1 for Wind as logical cuda:0. Complete native timing starts at prescribed host scene/query inputs and ends at physical native outputs returned to host, including H2D/forward/D2H. Provider/catalogue loading, fixture extraction, fresh-scene GPU forwarding, prepared decoding and repeated controls have separate scopes. Do not add medians from these overlapping measurements. The recorded same-device H6 comparison uses alternating order,3 warmups and 7 measured samples per arm.

| Paired replay | TRAIN fixture | R1 complete ms | Retained v1 H6 ms | R1/v1 |
| --- | --- | --- | --- | --- |
| Thermal | low_m | 60.529 | 34.854 | 1.7366 |
| Thermal | high_m | 53.328 | 34.866 | 1.5295 |
| Wind | low_m | 36.345 | 19.330 | 1.8803 |
| Wind | high_m | 41.730 | 23.440 | 1.7803 |

Absolute invocation timings vary: unchanged Wind D high-M is 20.623ms in the primary invocation and 29.817ms in the R1 invocation; old H6 is 11.199ms in the primary paired replay and 23.440ms in the R1 paired replay. These are retained observations under a shared host, not evidence that R1 changed D or a controlled quiet-machine comparison. The cost gate uses each current paired ratio. Captured dispatch is CUDA CUTLASS memory-efficient attention, not a Flash-attention claim inferred from the SDPA API.

![Complete native inference medians with observed ranges and B8 query-density allocated reserved baselines](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/cost_final/cost_scaling_family5.png)

Figure 5a. All B8 target-free native forward/backward points complete at Q1024/4096/8192 without optimizer steps. At Q8192 R1 versus D peak allocated memory is 2.618/3.188 GB Thermal and 2.041/5.510 GB Wind, with reserved memory plotted separately. Thermal native stencils/auxiliary observations change the deduplicated union; Wind retains role-prefix packing. These are discrete native workloads, not a pure-Q asymptotic fit or complete training-step memory.

![Native priority tail and component accuracy versus full complete-call cost](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/cost_final/accuracy_complete_cost_family5.png)

Figure 5b. Thermal uses casewise fluid-T RMSE p90; Wind uses the geometric mean of six near/downstream component-RMSE means. Old A-D is a horizontal reference without a fabricated cost point. R1/D errors and actual complete latency share explicit literal epoch 2500 identities; separate paired-v1 and maintained-benchmark medians are never summed.

![Observed complete native executed rows and whole-campaign history and process costs](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/cost_final/executed_work_campaign_cost_family5.png)

Figure 5c. At Q8192/high-M, H6 executes49,152 edge messages and 196,608 feature-head accesses; direct source rows are 98,304 Thermal/M12 and 245,760 Wind/M30. Shared local rows remain7,812/2,390 in those TRAIN fixtures. R1 also executes98,304 typed-centroid descriptor rows. Lower logical source work is not sparse executor savings; unknown old counters are n/r rather than fabricated zeros.

## Saved response and practical reference distance

The selected A2-D e1700 and H6 R1 e1200 states are frozen by field monitors before response opening. Saved 0291/0294/0687 provide three independent layout/direction families and six signed changes. Their existing two signed 0277 records are read only for missing-baseline bookkeeping; no baseline or predictions are synthesized. No new physical-reference call, solver attempt or inverse optimizer is used. Module-peak change is maxT(endpoint) minus maxT(baseline), not max of the temperature increment.

| Selected model | Role/direction groups | Elements | FP64 max absolute discrepancy | FP64 failures | Preserved FP32 failures |
| --- | --- | --- | --- | --- | --- |
| A2-D | 18 | 387,616 | 2.09e-13 | 0 | 8,954 |
| A2-H6 R1 | 18 | 387,616 | 3.27e-13 | 0 | 10,904 |

![Saved signed Thermal reference increments selected D and H6 R1 predictions native peak changes and numerical arithmetic checks](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/responses_final_v3/figures/saved_responses_family6.png)

Figure 6. Same saved analytic-wake/shared-grid reference, not CFD. Mean signed-replay fluid-temperature RMSE for 0291/0294/0687 is 0.030617/0.012187/0.001639 for D and 0.063176/0.028750/0.002994 for R1, versus retained v1 A-D 0.120853/0.033032/0.007796. Both improve this limited reference, but H6 remains worse than D. The arithmetic panel covers 245,760 fluid-channel elements only; the table above includes all 387,616 fluid/interface/material elements. FP64 agreement is a software consistency result, not response-fidelity certification.

The complete qualified Thermal classic CPU board is reused on DEV22 at actual mature e5000: classic1404 fluid-T mean0.300044 and 1804 mean0.223490, versus new D 0.949959 and R1 1.85330. The retained public-tool qualification also checks actual classic e1000/e5000 on0291/0687; there is no full-DEV22 classic1000 board, so no global1000 result is guessed. Wrong-input older arrays and the failed GPU parity attempt remain quarantined and are not retroactively validated.

| Wind classic, exact sealed DEV24 | Actual age | Near vector RMSE m/s | Downstream vector RMSE m/s |
| --- | --- | --- | --- |
| 2102 | 1000 | 0.121093 | 0.0768383 |
| 2102 | 2440, mature selected | 0.0982757 | 0.0670205 |
| 2103 | 1000 | 0.0890118 | 0.0541174 |
| 2103 | 2475, mature selected | 0.0747785 | 0.0396296 |

These Wind classics are replayed with their native E512 inputs, own full-TRAIN420 normalizers and public physical predictor on the exact DEV24 receiver panel; previous full-VALID90 receiver draws differ and are not substituted. At actual e1000 they have27,000 updates, batch 16, Q8192 split 6144 volume+ 2048 band, volume/band losses 0.75/0.25 and constant LR3e-4 over the declared 2500 horizon. Mature selected ages2440/2475 have 65,880/66,825 actual updates. Initialization seed 0 and sampler seed 42 are declared; no pretraining or frozen partner is declared. Case/query exposure derived from recipes is nominal, not an instrumented historical counter. These differ from new TRAIN72 / batch 24/Q4096/role-balanced/cosine fits. The large practical gap is real on this panel, but it does not isolate architecture from exposure and objective differences or establish a new full 1000 result.

## Correctness, limitations and closeout

The absolute ModularDT Python3.12.13/Torch2.6.0+cu124 environment passes 321 combined targeted CPU tests in 21.12 s plus 8 nonoverlapping Wind R1 tests in 1.62 s: 329 unique tests across those commands. This covers explicit attention references, masking/empty rows, staged state freshness, measure refinement, native transforms, factorized response/adjoint and strict identity/resume boundaries. Actual R1 native GPU macro-objective checks give finite nonzero gradients to all 237 Thermal / 220 Wind trainable parameters without updates, and every parameter changes by 2500. Trusted checkpoint identities compare canonically across tuple/list serialization; all Adam moments, sampler states, paired hashes and imported source bindings are checked.

Default behavior is verified independently against the original26d6363 modules: all four native forwards and 813 parameter gradients are bitwise equal under controlled single-threaded autograd. Earlier four-thread backward hashes differed and remain preserved; that setting is not called a bitwise pass. Smooth FP64 geometry fixtures compare automatic and centered differences at h=1e-5 with stable local membership. They do not differentiate through NumPy catalogue construction or the native Thermal observation stencil and are not new physical geometry labels. Prepared-state stale/ownership rejection is tested separately.

There are 73 disposable optimizer updates in total, below 800; none initializes scientific fitting. The bounded four-case learning diagnostic occurred after the first 100-epoch review, a recorded ordering deviation. Thermal D lacks its final fixed-checkpoint tiny-fit replay, so its rotating pre-step losses are not reported as a fixed before/after result. Preliminary diagnostics under a different base interpreter and all failed numerical/plotting attempts remain identified in local evidence; only correct-runtime measurements support the conclusions here. No second substantive revision is used to turn a failed field criterion into a pass.

The round uses 5.8622 aggregate GPU-associated process hours of 16, including qualification/evaluation/cost attempts and failures; no owned GPU process remains active. All six scientific fits stop at 2500. At the storage closeout snapshot, the local round retains 4.569 GiB of distinct-inode files against the 12 GiB /data ceiling; /home has a separate 2 GiB new-byte ceiling and both filesystems require 20 GiB free headroom. Verified manual cleanup removes 538 files / 264,308,485 bytes of owned disposable test payloads, duplicate panels, temporary PDF pages and superseded visual/HTML exports. One earlier failed-macro temporary unlink lacks a pre-deletion size/hash and is explicitly excluded from those exact totals. Only the six selected PDF masters and 15 inline PNGs remain as figure exports; all scientific checkpoints/aliases/100-epoch milestones, native arrays, metrics, provenance and source scripts remain retained. Snapshot free capacity is 102,828,670,976 bytes (95.77 GiB) on /data and 43,211,214,848 bytes (40.24 GiB) on /home; exact pre/post receipts are local.

Final protected verification passes 993 prior-round files, 264 primary-before-R1 files, four sealed inputs and all 1919 inherited observations, including 74 preserved absences and 18 resolved reference symlinks. Cleanup independently hashes 396 scientific run files and 27 selected figure files before/after with no change. Wind’s shared catalogue matches its 6337-file / 72,616,475,668-byte metadata fingerprint and all runtime write counters remain zero; the fingerprint has no historical per-file content SHA baseline, so that stronger claim is not made. Thermal H5 saved size/mtime/inode bindings match; its original protection record also lacks a baseline SHA. Solver metadata remains 326/326 with zero new attempts, and the unrelated PhyCoFlow PID 248340 remains alive under its original command. The pre-existing user report deletion is preserved and unstaged.

The preferred predictor in this matched study is A2-D on both datasets. For organized research, R1 offers modest Thermal broad/q/tail and Wind broad/downstream gains over primary H6, with a small additional memory cost, but no consistent priority improvement or full-run qualification. Thermal’s remaining limitation is an organized far response that generalizes substantially worse than its uncompressed signed-source control despite trained factors and full numerical rank. Wind’s is a compressed content/readout path whose fixed-TRAIN fit improves while native near/downstream DEV fidelity remains poor and content similarity increases. A later bounded study should localize those losses before spending on full data; no extension or additional campaign is active.

## Selected figure masters and evidence

| Family | Retained PDF master |
| --- | --- |
| 1 Learning/work | [PDF](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/learning_final_v2/honf_round2_learning_work_family1.pdf) |
| 2 Thermal native fields | [PDF](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/thermal_native_final/thermal_A2D_a2-h6-r1-centroids_signed_native_T_v_family2.pdf) |
| 3 Wind native planes/cuts | [PDF](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/wind_native_final/wind_r1_round2_family3_native_planes_and_cuts.pdf) |
| 4 Contents and signed factors | [PDF](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/content_final/content_and_native_interventions.pdf) |
| 5 Complete costs/memory/work | [PDF](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/cost_final/cost_scaling_family5.pdf) |
| 6 Saved responses | [PDF](../../diagnostics/generated/attention_round2_20261014T135200Z/figures/responses_final_v3/figures/saved_responses_family6.pdf) |

The 15 small inline PNGs above accompany these six PDF masters. Exact arrays, receiver/mask bindings, source/checkpoint hashes, all native rows and tested commands are in the [evidence index](HONF_Attention_Round2_Evidence_Index.md). Generated evidence and one-time renderers remain local and ignored; durable core/adapters/configs/tests/docs are committed on the non-default working branch.
