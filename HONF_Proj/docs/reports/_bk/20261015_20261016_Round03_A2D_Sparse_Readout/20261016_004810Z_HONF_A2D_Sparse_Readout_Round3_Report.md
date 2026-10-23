# HONF A2-D Round3: accuracy and source-preserving sparse execution

This study preserves A2-D's shared contextualizer, physical source inventory and native Thermal/Wind output laws. It tests a small source-scale value projection and source selection before expensive pair values. The [Round2 report](../20261014_20261015_Round02_Attention_Geometry/20261014_193935Z_HONF_Attention_Round2_Report.md) retains the completed formal comparisons and historical H6 findings; those results are not replaced by this development study.

**Evidence status.** D0, D-scale and the two S-Complete children finish literal epoch 2500. D0 remains both dense hosts. Neither trained sparse operating point meets the predeclared native-field criteria, so no Round3 fresh full-TRAIN follow-up is launched. S-Utility has no physical-learning result after its bounded auxiliary calibration failures. The single deterministic-CUDA repair permits packed continuation; actual saved epoch 2500 primary 0 inference passes native laws, fields/VJPs and identical-function packed/dense-masked/packet checks. Those correctness passes do not turn the negative accuracy or cost result into a qualification. The measured final cost, organizer and response boards below determine the research decision; historical failures and Round2 formal evidence are retained.

The result separates three questions. The tiny source-scale projection gives no qualifying mature accuracy gain. Learned S-Complete reduces actual fine rows and substantially outperforms its count-matched nearest control, but its co-adapted fields regress against D0. All-full evaluation at those adapted weights is a different function and can be much worse; it is not an accuracy-preserving fallback. No classic-replacement or inverse-readiness gate is applied to reach this decision.

| Literal 2500 operating comparison | Native DEV mean RMSE | High-M complete cold call, ms | High-M fine rows retained | Absolute B8/Q8192 peak allocated, GiB | Decision |
|---|---|---:|---:|---:|---|
| Thermal D0 full |Fluid T 0.877178 |46.269 |100.00% |2.966 |Retain dense host |
| Thermal S-Complete fitted 0 |Fluid T 0.948895 |76.666 |58.95% |3.790 |Native accuracy, time and memory miss |
| Wind D0 full |Near/down vector 0.234814/0.171749 m/s |28.467 |100.00% |5.946 |Retain dense host |
| Wind S-Complete fitted 0 |Near/down vector 0.245244/0.187368 m/s |52.340 |40.14% |5.852 |Native accuracy, time and memory miss |

Accuracy uses exposed DEV22/24. Cost and fine rows use separate input-fixed TRAIN high-M fixtures, Thermal 0631/M12 and Wind 533/M30 at Q8192. Memory repeats those high-M fixtures at fixed B8; it measures native prediction-only forward/backward, rather than an optimizer step. Thermal field units are packed-native; Wind velocity is m/s. The table uses the new dedicated sparse-comparison D0 invocation, while the earlier D0/D-scale pair retains its own timed baseline and raw ranges. All are one-fit development evidence.

## Predictor: measured progress and current misses

The source-scale value projection does not qualify at the declared 2500 endpoint. Thermal fluid/material/surface/peak means worsen 10.59/50.91/39.48/88.59%; the geometric v/p ratio is 0.99973 and v p90 worsens 10.15%. Wind's six near/downstream component geometric ratio is 0.968112, a 3.19% improvement, below the required 5%; background vector/Ux means worsen 10.77/11.24%. D0 remains the dense host for each dataset. This is a negative result for this single value-conditioning hypothesis, not evidence that source size is physically irrelevant.

| Literal 2500 native endpoint | D0 mean RMSE | D-scale mean RMSE | D-scale / D0 mean | D-scale / D0 p90 |
|---|---:|---:|---:|---:|
| Thermal fluid T |0.877178 |0.970033 |1.1059 |1.0176 |
| Thermal v |0.00307437 |0.00313768 |1.0206 |1.1015 |
| Thermal p |0.0112959 |0.0110619 |0.9793 |1.0310 |
| Thermal native curl |0.0517713 |0.0505635 |0.9767 |1.0242 |
| Thermal material T |0.919647 |1.38780 |1.5091 |1.4266 |
| Thermal surface T |0.902680 |1.25903 |1.3948 |1.4317 |
| Thermal q proxy |3.29703 |3.32944 |1.0098 |0.9602 |
| Thermal module peak endpoint |0.952481 |1.79629 |1.8859 |1.3470 |
| Wind volume vector |0.0732274 |0.0756548 |1.0331 |1.0030 |
| Wind hub-slab vector |0.135345 |0.139187 |1.0284 |0.9681 |
| Wind downstream vector |0.171749 |0.170149 |0.9907 |0.9355 |
| Wind near-turbine vector |0.234814 |0.230513 |0.9817 |0.8811 |
| Wind background vector |0.0389066 |0.0430955 |1.1077 |1.1522 |

The intermediate 1000 review below is retained because it shows why the literal endpoint matters. Wind D-scale then improved all five vector-role means and its six-component geometric ratio was 0.94678. Those gains did not persist strongly enough to qualify at 2500. Thermal's two fixed TRAIN fixtures also showed larger D-scale temperature errors at 1000; the miss was not confined to exposed validation cases. The native 2500 quantitative review and separate host-selection receipts preserve all mean/median/p90/worst values and module-count/layout summaries.

| Dataset / native endpoint | D0 e1000 mean RMSE | D-scale e1000 mean RMSE | D-scale / D0 mean | D-scale / D0 p90 |
|---|---:|---:|---:|---:|
| Thermal fluid T |1.1391 |1.35457 |1.1892 |1.1621 |
| Thermal v |0.00479166 |0.00456669 |0.9530 |0.9567 |
| Thermal p |0.0159881 |0.0152285 |0.9525 |0.9216 |
| Thermal material T |1.18819 |1.43502 |1.2077 |1.1036 |
| Thermal surface T |1.08878 |1.43708 |1.3199 |1.3237 |
| Thermal module peak endpoint |1.10655 |1.65525 |1.4959 |1.3039 |
| Wind volume vector |0.115851 |0.105579 |0.9113 |0.8822 |
| Wind hub-slab vector |0.174112 |0.162969 |0.9360 |0.8954 |
| Wind downstream vector |0.234522 |0.216745 |0.9242 |0.9243 |
| Wind near-turbine vector |0.297852 |0.273401 |0.9179 |0.9012 |
| Wind background vector |0.0810953 |0.0733885 |0.9050 |0.9285 |

Thermal quantities retain packed-dataset native units, including the maintained q proxy; Kelvin, SI pressure and SI flux units are not established here. Wind quantities are in m/s. Means are arithmetic means of per-case or per-row RMSE; p90 is a percentile of those errors, not a pointwise-error percentile. Wind directions within a layout are correlated. This is one fit per trajectory on exposed development panels, without independent-test or statistical-significance claims.

The full native boards retain all Thermal roles and Wind components, medians, tails, module-count groups and per-case/row errors. The 1000 review, identities and forecasts are in `attention_round3_20261015T152913Z/receipts/dense_matched_e1000_review_and_forecast_v1.json`. Thermal support masks match exactly across candidates; Wind native cell IDs, role IDs, coordinates and target hashes match across candidates and the retained classic companion board.

The mature classics below are practical reference distance on exactly the new exposed DEV22/DEV24 panels. They use their own full-TRAIN transforms and selected ages: Thermal 1804/1404 at 4738/4890, Wind 2103/2102 at 2475/2440. D0 uses TRAIN150/72 and literal 2500. The different population, update exposure, schedule history and native output laws prevent a controlled architecture or equal-compute interpretation; these values also differ from whole-canonical 89/VALID90 summaries. The native/mask/target bindings are rechecked before reusing the saved classic predictions.

| Common exposed panel / native mean RMSE | D0 e2500 | Dense classic | Routing classic |
|---|---:|---:|---:|
| Thermal fluid T |0.8771784 |0.1979409 |0.2958702 |
| Thermal u |0.02020497 |0.005714887 |0.01067478 |
| Thermal v |0.00307437 |0.0003190356 |0.0004868096 |
| Thermal p |0.0112959 |0.002099993 |0.003131268 |
| Thermal native curl |0.05177126 |0.02346104 |0.02429508 |
| Thermal surface T |0.90268 |0.4031624 |0.4502785 |
| Thermal material T |0.9196474 |0.3004292 |0.3598858 |
| Thermal q proxy |3.297028 |1.55607 |2.086491 |
| Thermal module peak |0.9524806 |0.2852894 |0.3658318 |
| Wind volume vector, m/s |0.07322737 |0.01767422 |0.0362412 |
| Wind hub slab vector, m/s |0.1353453 |0.03401523 |0.05463217 |
| Wind downstream vector, m/s |0.1717489 |0.03962964 |0.06702045 |
| Wind near turbine vector, m/s |0.2348142 |0.07477853 |0.09827571 |
| Wind background vector, m/s |0.03890657 |0.01126189 |0.02682422 |

Every D0 development endpoint remains worse than Dense1804/2103 on these panels. This is consistent with substantial remaining sample/work distance; it does not undo the full-TRAIN A2-D gains and misses already measured in the Round2 report. Classic replacement is not a qualification requirement for this bounded development study.

![Thermal D0, D-scale and fitted S-Complete learning, native accuracy and cumulative measured work through 2500](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/round3_figure_families_primary_policy_e2500_final_v4/family1_thermal_history.png)

Figure 1a ([history PDF](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/round3_figure_families_primary_policy_e2500_final_v4/family1_histories.pdf)): D0 and D-scale are fresh 2500 fits; S-Complete reuses D0 epochs 1–500 and adds 2000 physical epochs with separate 256-step auxiliary fitting. Each displayed absolute history reaches 375000 visits and 10000 ordinary updates with identical case/query fingerprints. Native fitted fluid-T benefit at 1000 reverses by 2500, while q improves to 3.00840 versus D0 3.29703. The cumulative epoch-time curve includes inherited TRAIN/DEV history; child process bars count only its own new segments, including setup/auxiliary overhead. Decoded receiver rows include native unions/auxiliaries and are not the 384 million primary-query budget.

![Wind D0, D-scale and fitted S-Complete learning, six-component native accuracy and matched work through 2500](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/round3_figure_families_primary_policy_e2500_final_v4/family1_wind_history.png)

Figure 1b: each absolute Wind history reaches 180000 visits and 12500 ordinary updates; the sparse child reuses the first 500 epochs and adds 10000 ordinary updates. Its fitted six-component native RMSE remains 1.020885 times D0 at 2500, despite continuing absolute improvement. D-scale’s transient 1000 benefit narrows below the qualifying 5% by 2500. The volume-MSE selector does not stand in for all five native roles or component tails. DEV24 contains only eight correlated layouts, each with three directions.

![Thermal DEV0687 native temperature and v fields, D0 prediction and matched D0/D-scale signed residuals at 2500](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/dense_e2500_v1/family2_thermal_fields_DEV0687_D0_e2500_vs_D-scale_e2500.png)

Figure 2a ([native-field PDF](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/dense_e2500_v1/family2_thermal_fields_preview_and_children.pdf)): on the fixed M10 scene 0687, fluid-T RMSE is 0.861/1.15 and v RMSE 0.00245/0.00267 for D0/D-scale. Shared within-row color limits distinguish physical fields from signed errors. The companion 0291 page in the PDF shows T RMSE 0.827/0.961 and v0.00326/0.00338. These are illustrative saved native scenes; population selection uses all 22 cases.

![Thermal surface temperature and q-proxy native detail on fixed DEV0291 and 0687, both literal 2500 dense models](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/dense_e2500_v1/family2_thermal_surface_q_D0_e2500_vs_D-scale_e2500.png)

Figure 2b: the M10 scene 0687 surface RMSE rises 0.478→1.27 and q RMSE 1.31→1.73 for D-scale. Scene 0291 has similar surface RMSE 1.41/1.39 but q4.22/4.79. All observations derive from the same native temperature field. Ribbons show the 10–90% module spread, without an uncertainty or SI-flux claim. All five dense figure pages were rendered from saved arrays and visually inspected.

![Wind M30 DEV426 native Ux, Uy and Uz signed errors for both literal 2500 dense models](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/wind_dense_e2500_v1/family3_wind_row_426_error_comparison.png)

Figure 3a ([six-page Wind native PDF](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/wind_dense_e2500_v1/family3_wind_native_hub_planes_comparison.pdf)): on the M30 row 426 hub plane, D0/D-scale Ux RMSE is 0.1455/0.1711 m/s, Uy0.02533/0.02444 and Uz0.005018/0.005095. Corresponding Uy relative-L2 is 0.2679/0.2585 and Uz0.3129/0.3177. Separate component/error scales make the smaller transverse and vertical fields visible. This fixed illustrative plane is not the five-role population metric.

![Wind M8 DEV69 native crosswind cuts at three downstream offsets, with separate Ux, Uy and Uz profiles](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/wind_dense_e2500_v1/family3_wind_row_069_crosswind_cuts_comparison.png)

Figure 3b: row 69 uses stored native cuts at actual 1.945/4.008/7.996D offsets for requested 2/4/8D. D0/D-scale hub-plane Uz relative-L2 is 0.5059/0.5137; cut-relative errors remain large despite small absolute velocity error. Solid predictions and dashed stored targets expose amplitude and shape misses separately from Ux. The PDF includes both prediction/error planes and three-offset cuts for rows 69/426; all six pages were visually inspected on unchanged native arrays.

## Mature sparse fidelity: actual saved epoch 2500 children

S-Complete improves Thermal q-proxy mean error 8.75%, but fluid T/v/p means worsen 8.18/9.58/21.43%, curl 10.82%, material/surface 24.29/25.78%, and peak 32.79%. Pressure p90 worsens 20.59%, material/surface p90 22.24/24.30%, while fluid-T p90 improves 1.26% and peak p90 changes only 0.68%. Thus a better q proxy and one acceptable tail do not conceal the field and material-tail regressions. Thermal point 1 additionally raises fluid-T error 2.704 times D0; it remains only a bounded inference frontier.

| Literal 2500 native field | D0 mean RMSE | S-Complete fitted 0 mean | Fitted 0 / D0 mean | Fitted 0 / D0 p90 | Same-weight full / D0 mean | Fitted 1 / D0 mean |
|---|---:|---:|---:|---:|---:|---:|
| Thermal fluid T |0.8771784 |0.9488946 |1.0818 |0.9874 |7.8873 |2.7037 |
| Thermal u |0.02020497 |0.0223847 |1.1079 |1.2010 |2.2960 |1.9527 |
| Thermal v |0.00307437 |0.003368924 |1.0958 |1.0823 |3.5584 |1.0408 |
| Thermal p |0.0112959 |0.01371635 |1.2143 |1.2059 |2.8072 |1.7128 |
| Thermal native curl |0.05177126 |0.0573727 |1.1082 |1.1162 |1.7724 |1.4332 |
| Thermal surface T |0.90268 |1.135347 |1.2578 |1.2430 |5.6153 |2.2270 |
| Thermal material T |0.9196474 |1.143053 |1.2429 |1.2224 |5.3948 |2.2889 |
| Thermal q proxy |3.297028 |3.008396 |0.9125 |0.8891 |1.3687 |0.9738 |
| Thermal module peak endpoint |0.9524806 |1.264778 |1.3279 |1.0068 |5.2640 |2.3005 |
| Wind volume vector, m/s |0.07322737 |0.08212194 |1.1215 |1.1008 |1.6407 |1.1838 |
| Wind hub slab vector, m/s |0.1353453 |0.1481367 |1.0945 |1.0738 |1.3161 |1.1324 |
| Wind downstream envelope vector, m/s |0.1717489 |0.1873683 |1.0909 |1.1055 |1.2199 |1.1341 |
| Wind near turbine vector, m/s |0.2348142 |0.2452436 |1.0444 |0.9878 |1.0628 |1.0698 |
| Wind background vector, m/s |0.03890657 |0.04908161 |1.2615 |1.2253 |2.4547 |1.4321 |

The explicit Thermal tail review is mixed rather than a uniform collapse. Fitted 0 q p90/worst improve 11.09/16.11%; curl p90 worsens 11.62% while worst improves 12.95%. Material p90 worsens 22.24% while worst improves 13.38%; peak p90/worst worsen 0.68/4.41%, far less than its 32.79% mean regression. Fluid-T worst improves 13.98%, whereas v worst worsens 24.02%. The existing means/primary-p90 criteria already fail. No new universal 10% nonprimary-tail gate is imposed to make this outcome negative.

Wind fitted 0 near/downstream vector means worsen 4.44/9.09%, and volume/hub/background 12.15/9.45/26.15%. The six near/downstream component geometric ratio is 1.020885; that pooled figure does not override the failed downstream Ux/vector mean and p90 checks. Fitted 1 has geometric ratio 1.044078 and broader failures. All 20 component/role means, medians, p90 and worst errors are retained in `receipts/scomplete_literal2500_native_quantitative_review_v2.json`, with exact 24-row target/query endpoint hashes. Both points fail before any cost criterion is considered.

The regressions vary by geometry rather than disappearing within every module-count group. Thermal fitted fluid-T ratios are 1.267/1.056/1.030/1.023 for M3/5/7/10; material ratios are 1.371/1.184/1.184/1.304, with pressure worse in every group. Wind near/downstream vector ratios are 0.955/0.886 at M21 and 0.974/0.952 at M24, but 1.104/1.205 at M30 and 1.130/1.213 at M8. Layout and direction breakdowns remain grouped evidence from only eight exposed layouts, without receiver-level significance claims. `receipts/scomplete_e2500_strata_history_main_review_v1.json` binds these strata to saved per-case/row metrics.

At the same adapted weights, all-full Thermal fluid-T error is 7.887 times D0 and its v/p errors 3.558/2.807 times D0; fitted 0 is therefore much better than that all-full control while still worse than D0. Wind all-full volume/background errors are 1.641/2.455 times D0. Equal-count nearest 0 is also worse than fitted 0 in every Thermal field and all five Wind vector-role means. Its hard-unit versus fitted-smooth gate difference remains a consequential control limitation. These comparisons show useful adaptation to the selected function; they do not demonstrate an accurate dense fallback, physical source causality or a qualifying sparse model.

The first Thermal primary-board command accidentally omitted 0648 and produced 21 cases. Its JSON, command/log and failure receipt remain preserved and excluded from scientific summaries. A fresh exact 22-case board uses the original authorized argv with only its output path changed; no model setting or checkpoint changes. The legitimate full 22 board is `thermal_scomplete_e2500_native_primary0_cpu_v2`, and the untouched secondary board retains the same 22 cases. This evaluation-harness correction is distinct from the one permitted substantive model/numerical repair.

![Thermal 0687 native reference, D0, fitted S-Complete and signed child residual at 2500, with shared physical scales and visible relative-L2](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/round3_figure_families_primary_policy_e2500_final_v4/family2_thermal_fields_DEV0687_S-Complete_e2500_vs_D0_e2500.png)

Figure 2c ([three-page sparse field PDF](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/round3_figure_families_primary_policy_e2500_final_v4/family2_thermal_child_fields_e2500.pdf)): on fixed 0687, fluid-T RMSE improves 0.861→0.759 (relative-L2 0.0731→0.0644), while v worsens 0.00245→0.00338 (0.0748→0.103). The companion 0291 page instead shows T worsening 0.827→1.01 and v0.00326→0.00358. Both cases were input-selected before outcomes; neither local gain nor miss substitutes for DEV22. Physical scales are shared across REF and both predictions; the separate signed-error scale is centered at zero.

![Thermal surface temperature and extraction-consistent q detail for D0 and fitted S-Complete at 2500, both fixed scenes](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/round3_figure_families_primary_policy_e2500_final_v4/family2_thermal_surface_q_S-Complete_e2500_vs_D0_e2500.png)

Figure 2d: surface RMSE rises 1.41→1.81 on 0291 and 0.478→0.781 on 0687. Their q errors also rise 4.22→4.32 and 1.31→1.59, despite the pooled DEV22 q mean improving 8.75%. Faint per-module curves and 10–90% spread reveal variation without uncertainty or SI-flux claims. Arrays, native ordering and physical extraction are unchanged.

![Wind 426 native component signed errors, D0 full versus fitted S-Complete at 2500](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/wind_d0_scomplete_e2500_primary0_fitted_v3/family3_wind_row_426_error_comparison.png)

Figure 3c ([six-page dense/sparse Wind PDF](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/wind_d0_scomplete_e2500_primary0_fitted_v3/family3_wind_native_hub_planes_comparison.pdf)): on M30 row 426, Ux RMSE worsens 0.1455→0.1906 m/s while Uy0.02533→0.02525 and Uz0.005018→0.004643 improve slightly. Uy/Uz relative-L2 changes 0.2679→0.2671 and 0.3129→0.2895. Shared component-specific residual scales preserve the smaller transverse/vertical structures. This exact native hub plane is illustrative rather than the five-role population board.

![Wind 69 native crosswind2D/4D/8D cuts, all three components and saved targets versus both 2500 model policies](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/wind_d0_scomplete_e2500_primary0_fitted_v3/family3_wind_row_069_crosswind_cuts_comparison.png)

Figure 3d: on M8 row 69, the actual offsets are 1.945/4.008/7.996D. Fitted Uz cut relative-L2 improves from 0.7235/0.7682/0.6432 to 0.4899/0.6284/0.4771, but Ux errors at 4D/8D rise and significant crosswind shape/amplitude misses remain. Dashed saved targets and solid predictions use unchanged native-y samples; no projected point cloud or new CFD field is substituted. All six prediction/error/cut pages have been rendered and independently inspected.

## Organizer: what changes and what is measured

D-scale adds a bias-free 128-wide projection of the existing per-axis source-length displacement,256 Thermal or 384 Wind parameters. It is zeroed after named initialization so the initial function and common tensors match D0. The contextualizer, source reinjection, local branch, original direct-attention normalization and physical output laws remain unchanged.

S-Attn is a read-only control at 95%/99% per-head mass union with protected near sources. S-Utility learns a small heat-independent selector; S-Complete retains a bias-free rank 16 all-source base and evaluates selected fine-minus-base corrections. Selection precedes fine Fourier/value construction. Flow uses original attention coefficients without renormalization. Thermal retains signed source-indexed K independently of attention; its scalar bias contributes once per physically present heating source. The all-one endpoint calls the legacy reader directly.

For S-Complete, the hidden contribution is \(\widetilde\nu_{qi}=\nu^0_{qi}+g_{qi}(\nu^{\rm fine}_{qi}-\nu^0_{qi})\). Rank 16 describes the cheap algebraic factors, rather than a bank of 16 physical or latent source groups. Every physical source remains individually indexed in the completion and in the upstream contextualizer. A zero fine gate therefore removes one expensive correction while retaining that source's cheap and contextual influence. The Thermal operating envelope is twice the maximum absolute present-source heating in the sealed TRAIN cohort,3.99830747 in this development fit; it is a declared sampled operating contract, not a certified physical bound.

The exact constructor counts are 1520493/1529554 parameters for Thermal D0/S-Complete and 1525675/1534767 for Wind, adding 9061/9092 parameters, about 0.6%. Actual source selection, sorting, padding and rank-factor contractions remain additional executed work despite this small capacity change. The packet executor groups receivers into degree buckets while retaining each receiver's own source-index list; a shared bucket does not imply a common physical source group or a query-independent message.

The `nearest_fitted` geometry control first discovers each receiver's positive fitted support count, then selects the same number of nearby sources with hard unit gates and the same protected near sources. It retains the S-Complete all-source base, original attention and contextualizer. Its count match is useful for comparing operating choices, but its attenuation also differs from smooth fitted gates; it is not an isolated scorer-only or causal ablation. The fitted scorer used to discover the comparison count remains inside measured control cost.

The bounded engineering auxiliary checks preserve inherited dense tensors, named Adam state and RNG while fitting only new parameters on TRAIN-derived model targets. They are implementation checks, not trained sparse field-accuracy results. Their disposable updates are accounted separately from scientific physical training and from each child's later bounded initial auxiliary fit.

Selected-D0 maximum-query v13 tests pass fixed-weight native outputs and source/query/parameter/heat derivatives for both sparse families. Exact checkpoint serialization, sampling, provider and RNG restoration also pass. Numeric parameter and Adam-moment continuation pass, but the next native output exceeds the unchanged allowance by factors up to 1.173 Thermal and 2.034 Wind. These failures prevent physical sparse training at that stage. The single declared repair per dataset adds recipe-bound deterministic CUDA execution, retaining every model equation, loss, sampling rule and tolerance; its qualification is separate from the failed v13 record. The code passed 95 focused CPU checks and was committed/pushed as `8b98f68430621fec81d21946b7b74a63d220ad0d` before creating a new immutable runtime. Earlier runtimes remain intact.

The repaired v14 test passes fixed-weight inference and packed continuation on both variants/datasets, including now-exact next model/moment/native-output states. Dense-masked/packet post-step alternatives still have strict native failures on some variants; the raw overall receipts therefore remain failed. Physical training may use only the qualified packed reference, after production-memory and child-attachment checks. Alternative executors remain fixed-weight inference options requiring later qualification at the actual trained state. This separates a successful numerical repair from an unsupported claim that all training backends are interchangeable.

The first S-Utility starts reveal a separate implementation failure: CUDA FP64 head reductions were added to CPU FP64 threshold-screen accumulators. Both completed 256 finite auxiliary optimizer calls before calibration raised; no physical epoch 501 update, controller checkpoint or auxiliary receipt was written. Those calls consume the per-child allowance, so neither child is retried. Logs, raw ledger rows and a compact failure receipt preserve the distinction. The correction copies only the tiny reduced totals to the existing CPU accumulators and changes no learned function or threshold equation; all four family/thermal-mode CUDA screen checks pass with zero optimizer calls, together with 50 CPU checks. Commit `a29c589e0f9c05cadf96b5ac410990132fb2ebc9` is pushed and frozen as a separate runtime. The already-used single substantive numerical revision remains unchanged. S-Utility has no measured physical-learning result; each S-Complete child subsequently uses its own 256-step initial allowance.

Both S-Complete children complete the 100-epoch 501–600 transition with 400/500 new physical updates and 15000/7200 visits for Thermal/Wind. Every new epoch matches D0 case-order/query fingerprints, and the inherited 500-epoch histories are exact. Initial teacher traces use 256 rows per TRAIN case, within the plan maximum but twice the earlier 128-row forecast; refreshes remain 128 rows on one TRAIN case every 8 ordinary updates. Initial realized auxiliary/native gradient shares are 5.00% Thermal and 0.865% Wind, with fixed coefficients 0.0262211/1.0. Later shares exceed 5% on 156/400 and 281/500 updates, which is reported without retuning. Actual initial and refresh work, epochs, checkpoint hashes and peak allocated/reserved bytes are retained separately.

At literal 600, fitted Thermal mean ratios to same-age D0 are 0.9259 fluid T,0.8703 v,0.9731 p,0.9827 q and 0.7810 peak; material T rises 5.88% and u5.45%. Wind vector-role means rise 2.1–4.3%, with priority component means rising 2.1–5.2%. These are healthy early fits rather than mature qualification. Native fine execution retains 100.0000% of Thermal and 99.9995% of Wind matched full-reader rows: attenuation and completion affect outputs, but positive gates still request almost every fine value. No source-pruning saving is inferred from transition padding removal. The same-age native health receipt retains every mean/median/p90/worst ratio and exact endpoint-identity checks; later fitting keeps thresholds and coefficients fixed.

The actual saved 600 children pass CPU16-query guards and GPU 8192-query fixed-weight fields, VJPs, native heat laws and same-function packed/dense-masked/packet checks. Thermal retained-factor application also passes the propagated native bounds and stale/owner checks. Separately,72 query-order/subset/public-chunk comparison flags pass on the input-fixed four-scene panel; q/curl are checked only where complete native unions exist. A five-position input-only geometry sweep scopes gate behavior without claiming global continuity. These checks perform no fitting or optimizer updates and do not establish physical accuracy, mature sparse work or cost savings. The first Thermal GPU command omitted its required CPU-guard argument and exited before model/provider inference; its failure JSON and 5.36 seconds remain charged and preserved beside the corrected guarded calls.

At 1000, Thermal fitted/D0 mean ratios are 0.9202 fluid T,0.8618 v,0.9525 p,0.9508 curl,1.0303 q,0.9263 material T and 0.9780 peak. Wind fitted vector ratios are 1.0974 near,1.1651 downstream,1.1152 hub,1.2300 volume and 1.3101 background; downstream Ux rises 16.78%. These Wind misses prevent qualification at this review. The physical TRAIN objective and DEV selector nevertheless improve from 600, without a nonfinite state or field collapse, so both unchanged fits continue to the declared 2500 endpoint. The fitted physical fine-row fractions fall to 93.99% Thermal and 91.88% Wind on the native DEV panels; packed execution adds no padding there. These panel counts are separate from the high-M cost workload and the required 30% work saving.

All-full access at adapted weights is a separate function, rather than an automatically accurate fallback. At 1000 its Thermal fluid-T/v/p mean errors are 2.031/1.861/1.574 times D0, compared with 0.920/0.862/0.952 for fitted execution. This substantial co-adaptation effect is retained beside physical reference error and arithmetic executor checks. Wind's all-full and fitted errors are closer, but both regress versus D0. No threshold, rank or auxiliary coefficient is changed after this exposed review.

The actual trained epoch 2500 organizer is measured on the same input-fixed 128 geometry receivers and three named positions in each scene. Its Thermal 990/1354 native unions include observation neighbors; Wind uses 128 queries per scene. The selected fine support, original attention, smooth gates, native source IDs and actual degree buckets are exported directly from the saved controller. Rank completion and global contextual information remain available from every present source even when that source has zero fine gate.

| S-Complete2500 fitted 0 organizer probe | Physical full rows | Full executed rows | Retained physical fine rows | Packet executed rows | Packet padding |
|---|---:|---:|---:|---:|---:|
| Thermal 0291, M5, Qunion990 |4950 |11880 |3648 |4949 |1301 |
| Thermal 0687, M10, Qunion1354 |13540 |16248 |8496 |10371 |1875 |
| Wind 69, M8, Q128 |1024 |1024 |457 |567 |110 |
| Wind 426, M30, Q128 |3840 |3840 |1334 |1777 |443 |

At the three named positions, packet widths are 8/4/8 for Thermal 0291,8/4/12 for 0687,8/2/8 for Wind 69 and 16/8/30 for 426. Width includes padding; for example 426's downstream receiver selects 17 physical sources but executes a 30-wide bucket. These are actual per-query source lists rather than shared source centroids. Input-only probe counts differ from the complete native DEV panel and the Q8192 TRAIN cost fixture, so they do not replace the qualification work denominator.

Removing one selected fine-minus-base correction at each of the 12 named receivers preserves every other support entry, original attention and the cheap base, then restores exact model/router state and bitwise output. Thermal temperature changes include−4.490/+0.0175/−7.939 on 0687; Wind 426 Ux changes−0.0962/+0.0558/+1.2248 m/s. Large or signed effects establish use of those computational reads. They do not identify a physical source-response kernel, a heat fraction or a causal intervention on the physical system.

![Trained Thermal source attention, fine gates, source-preserving packets and exact fine-read intervention on fixed scene0687](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/figures/source_organization_scomplete_e2500_v1/family4_thermal_0687_S-Complete_e2500.png)

Figure 4a: trained S-Complete at literal 2500, primary 0, M10. Physical source IDs remain 0–9. Retained fine rows 8496 become 10371 executed rows after 1875 padding rows; the full path executes 16248 including 2708 nonpresent/padded rows. At the upstream named receiver, a small positive temperature change after removing one residual is possible because fine-minus-base coefficients are signed. This is model information flow on an input-only scene, without target-field or physical-causality evidence.

![Trained Wind M30 source attention, smooth fine gates, actual packet source IDs and restored read-removal velocity effects](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/figures/source_organization_scomplete_e2500_v1/family4_wind_W426_S-Complete_e2500.png)

Figure 4b: row 426 retains 1334/3840 physical pairs, executes 1777 with padding, and keeps all-source cheap/context paths. The downstream named receiver uses a 30-wide packet for 17 retained IDs. Removing its selected source 3 correction changes Ux by+1.2248 m/s, with exact restoration. The [four-page organizer PDF](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/figures/source_organization_scomplete_e2500_v1/family4_source_organization.pdf) also includes 0291 and Wind 69; no latency saving is inferred from these illustrations.

Actual 2500 fitted-primary query consistency passes 56 Thermal and 16 Wind comparison flags after saved-state preregistration. Inverse-restored order, direct half-subsets and 2/3 public query chunks agree; native q/curl use only complete observation unions. Five fixed geometry positions show the first Thermal gate moving from 1 into approximately 0.997 on 0291 or 0.985 on 0687; the inspected Wind source remains 1. This small sweep is descriptive and does not prove global continuity through catalogue/stencil changes. Both checkpoints/source files are rehashed unchanged, with zero optimizer, cache-write or new solver calls. The runtime environment shorthand typo in the prelaunch receipt is separately corrected: helpers bind or overwrite it with the full immutable revision, and actual source hashes remain exact.

The input-only organizer prototype uses Thermal 0291/0687 and Wind 69/426,128 geometry-selected receivers per scene and three named receiver positions. Thermal native extraction expands these requests to 990/1354 valid union queries; packet counts and source-attention exposure cover that complete union. Removing one selected fine read at the named receiver preserves original attention and other support, produces a finite native-output change, and restores model/router/output exactly in all 12 tested S-Attn95 interventions. This establishes reliance on a computational read. It does not establish physical causality or a useful trained sparse policy. Full-access masked source slots, selected packet padding and uninstrumented legacy counters are reported separately.

| D0 e1000 S-Attn95 input-only probe | Present source-pair rows | Retained physical fine rows | Executed packet fine rows | Full executed fine rows |
|---|---:|---:|---:|---:|
| Thermal 0291, M5 |4950 |3754 |4963 |11880 |
| Thermal 0687, M10 |13540 |7929 |9874 |16248 |
| Wind 69, M8 |1024 |989 |1024 |1024 |
| Wind 426, M30 |3840 |3426 |3826 |3840 |

Packet padding removes almost all Wind savings in this probe: the M30 scene drops 10.78% of physical pairs but only 0.36% of executed fine rows, and M8 saves no executed rows. Thermal 0291's58.22% executed-row reduction includes removal of 6930 padded/nonpresent full-reader slots; its physical pair reduction is only 24.16% and must not receive credit as 58% physical sparsity. Thermal 0687 removes 41.44% physical pairs and 39.23% executed rows. These input-only counts establish neither native accuracy nor latency. The matched nearest controls retain the same counts; different model-output effects and final native boards remain separate measurements.

The literal 2500 D0 read-only population controls now measure stored-reference error on the complete fixed native panels. Their all-full mode exactly reproduces the original 2500 board's native metrics and target/support identities. Equal-count nearest and shuffled policies preserve the same per-receiver physical support count; no controller is fitted. These packed population calls differ from the padded input-only prototype above, so their work denominators are reported separately.

| D0 e2500 read-only policy | Thermal retained fine fraction | Fluid-T mean RMSE | Material-T mean RMSE | q mean RMSE | Wind retained fine fraction | Near vector mean RMSE, m/s | Downstream vector mean RMSE, m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full |1.0000 |0.877178 |0.919647 |3.29703 |1.0000 |0.234814 |0.171749 |
| S-Attn95 |0.7260 |1.27335 |1.89895 |5.30295 |0.8326 |0.248431 |0.179085 |
| Equal-count nearest 95 |0.7260 |2.10241 |3.62181 |8.57076 |0.8326 |0.267232 |0.192875 |
| Equal-count shuffled95 |0.7260 |2.46000 |3.67645 |7.97632 |0.8326 |0.423342 |0.345653 |
| S-Attn99 |0.9074 |0.901111 |1.00715 |3.45786 |0.9644 |0.235080 |0.172106 |

Attention-based read selection is better than these equal-count controls, but 95% attention mass is not sufficient thermal-response importance: Thermal fluid-T error rises 45.16% and material-T error 106.48%. The 99% control limits distortion while retaining 90.74% of Thermal and 96.44% of Wind physical fine rows. Neither establishes a useful sparse operating point under the declared work criterion. All component/tail ratios, both 99% geometry controls and exact work counts are retained in `receipts/d0_e2500_readonly_controls_main_review_v1.json`; latency is not inferred from these counts.

## Execution: qualified baseline and cost scope

The post-reduction projection and local reuse have exact FP64 algebraic evidence. Their strict checkpoint/GPU qualification is mixed: reduced projection misses the predeclared FP32 checks in Thermal heat JVP and several Wind native/derivative paths. Those failures are retained. Every dense scientific trajectory therefore uses legacy arithmetic; no running lineage changes backend or tolerance.

The frozen dense runs use an 8192-query internal tile for Thermal and whole-request Wind decoding. Wind recipes contained an inert 8192 field that the frozen builder did not apply; the saved constructor and subsequent native loading preserve whole-request execution. TRAIN Q8192 is unchanged, while the Q10240 monitor remained whole-request. This discrepancy is disclosed rather than repaired within a trajectory. The 128-wide feature dimension is not a 128-query tile.

Complete-call measurements start with host input snapshots and include device transfer, scene preparation where applicable, query/native-union preparation, decoding, native extraction and CPU return. Checkpoint loading, catalogue/fixture setup and retained-query preparation are separately reported. Cold-scene, prepared-scene/new-query and repeated-heat scopes are nested alternatives and must not be added. Headline calls use eval mode, `torch.no_grad()`, two warmups and seven profiler-off repetitions; retained factors use versioned tensors rather than inference-mode tensors.

The shared scientific fixture loader stages TRAIN target arrays during excluded setup. The timed prediction calls and prediction-only native backward do not consult those targets. Earlier helper wording that targets were not read is inaccurate for TRAIN loading; the target-free claim applies to the measured computation. Wind TEST targets remain locked.

At 1000, high-M D-scale/D0 cold/prepared median ratios are 0.9621/1.0597 Thermal and 1.0751/1.1108 Wind. Thermal high-M repeated heat is 1.1237 times D0. These are separate model invocations with retained raw samples and ranges, not interleaved same-process model timing. They do not qualify a sparse speedup. The cost index is `attention_round3_20261015T152913Z/evaluation/source_read_complete_cost_gpu_final_index_v1.json`.

At 2500, high-M cold-call medians are 47.237/47.805 ms for Thermal D0/D-scale and 27.667/29.017 ms for Wind. D-scale/D0 ratios are 1.0120/1.0488. Prepared new-query ratios are 1.0159/1.1107; Thermal same-query heating is 1.0420. Last 100-epoch median training-cost ratios are 1.0096/1.0404. Thus cost passes the 15% allowance on both datasets; physical accuracy determines the negative D-scale decision. Each scope has two warmups and seven profiler-off repetitions with ranges/raw samples retained. Thermal high-M output hashes differ across repetitions despite stable request, geometry, query and work identities; do not imply bitwise CUDA repeatability from stable timing inputs.

Literal 2500 D0 fixed-batch native forward/backward now holds B8 at requested Q1024/4096/8192. Peak allocated memory is 1.206/1.938/2.966 GiB for Thermal and 0.818/3.017/5.946 GiB for Wind; peak reserved stays 4.404/7.971 GiB respectively after warming all shapes. Thermal actual native union sizes are 2810/5020/8192 per case. Wind's helper does not expose a native union count, so it remains unavailable; output shapes are[8, Q,3]. These prediction-only backward calls have finite gradients and zero optimizer updates. Loaded target residency contributes 564192B to Thermal's CUDA baseline, while Wind's196608B targets remain CPU NumPy. They are not complete optimizer/auxiliary-manager training peaks. The principal future memory statistic is absolute B8/Q8192 peak allocated bytes, fixed before learned-child outcomes, without subtracting baselines to inflate a saving; reserved memory and all baselines remain separately reported.

## Mature complete cost and fixed-B8 scaling

At high M, Thermal fitted 0 evaluates 57949/98304 fine rows, a 41.05% reduction, but cold complete-call time rises 46.269→76.666 ms,1.657 times D0. Wind evaluates 98654/245760 rows, a 59.86% reduction, but time rises 28.467→52.340 ms,1.839 times D0. The two warmups and seven profiler-off samples span 44.715–48.686/74.889–81.058 ms Thermal and 27.438–29.734/48.024–55.351 ms Wind for D0/fitted. These observed ranges do not support a sparse speedup.

| High-M inference executor, fitted 0 unless noted | Thermal cold median, ms | Thermal executed fine rows | Wind cold median, ms | Wind executed fine rows |
|---|---:|---:|---:|---:|
| D0 full, packed label uses original full path |46.269 |98304 |28.467 |245760 |
| S-Complete same-weight full |65.221 |98304 |46.167 |245760 |
| S-Complete dense-masked |76.951 |98304 |51.853 |245760 |
| S-Complete packed |76.666 |57949 |52.340 |98654 |
| S-Complete packet |84.223 |66483 |62.203 |135596 |
| Equal-count nearest 0, packed |75.977 |57949 |53.636 |98654 |
| S-Complete fitted 1, packed |76.255 |43665 |49.810 |72138 |

Packed and dense-masked compute the same primary 0 function, yet their complete times are nearly tied; gathering/planning/reduction overhead offsets fewer fine evaluations in this workload. Packet padding adds 8534 Thermal and 36942 Wind rows and is slower than packed; after padding its executed-row reductions are 32.37/44.83%, with each receiver's own source IDs preserved. Point 1 cuts more rows but fails native accuracy and remains slow. D0 S-Attn95/99 also slows the complete call:50.974/49.748 ms Thermal and 31.736/32.191 ms Wind, despite different retention. Logical or physical sparsity is therefore not executor efficiency.

Prepared-scene/new-query medians are 22.930→38.047 ms Thermal and 7.617→15.163 ms Wind for D0/fitted 0. Thermal same-query repeated heating is 4.598→7.035 ms and executes no new fine values in either retained-factor call. Those scopes are alternatives to cold prediction, not disjoint components to add. On low-M fixtures, fitted 0 still costs 69.889 versus 40.196 ms Thermal and 50.014 versus 26.396 ms Wind. There is no measured small-scene saving. Full versus fitted at S-Complete weights is a different function; child versus D0 also differs in weights and recipe-bound deterministic CUDA settings, so the ratio cannot isolate the causal cost of rank 16 or selection alone.

| Native forward/backward requested Q, fixed B8 | Thermal D0 / S-Complete full / fitted 0 peak allocated, GiB | Wind D0 / S-Complete full / fitted 0 peak allocated, GiB |
|---|---:|---:|
|1024 |1.206 /1.254 /1.458 |0.818 /0.865 /0.851 |
|4096 |1.938 /1.985 /2.531 |3.017 /3.064 /3.004 |
|8192 |2.966 /3.015 /3.790 |5.946 /5.992 /5.852 |

At the preregistered absolute B8/Q8192 statistic, Thermal fitted memory increases 27.80%; Wind saves only 1.58%. Neither meets 20%, and both exceed the 15% complete-call slowdown allowance. After warming all three shapes, peak reserved is 4.404/4.426/5.090 GiB Thermal and 7.971/8.002/7.619 GiB Wind for D0/full/fitted. Raw results retain allocated/reserved baselines, actual unions, finite gradients, native outputs and target-residency inventories. Fitted Q8192 fine execution drops to 463592 Thermal and 789232 Wind rows across B8, but peak memory is a measured allocator result rather than a rescaled fine-row count. The q/curl/native extraction graph and all-source planning/completion remain inside backward. These calls perform zero optimizer updates; actual optimizer-boundary peaks appear separately in the protocol accounting.

![Literal 2500 native accuracy versus complete cost, fixedB8 query-density allocated memory and actual high-M fine-row fractions for both datasets](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/figures/accuracy_cost_e2500_final_v2/family5_accuracy_cost_overview.png)

Figure 5a ([PDF](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/figures/accuracy_cost_e2500_final_v2/family5_accuracy_cost_overview.pdf)): the native error axis uses exposed DEV22 fluid-temperature RMSE for Thermal and the absolute geometric mean of six exposed DEV24 near/downstream component RMSEs for Wind; all individual-field/tail checks remain separate. Thermal error uses the explicitly labeled logarithmic axis. Timing ranges are seven observed profiler-off samples on the fixed high-M TRAIN workloads, not confidence intervals or costs of those DEV examples. Packed fitted 0 cuts fine rows but lies to the right of D0; neither B8 memory curve meets the qualifying saving. The same-weight full policy is a different, co-adapted function. Point 1 is a fixed inference diagnostic, and nearest 0 also changes smooth attenuation to hard gates. Packet padding and complete costs are retained in the preceding table and organizer figures. The overview omits D-scale because its older Thermal export lacks the strict host metadata required by this figure's normalizer; its measured matched comparison remains above.

A separate profiler diagnostic explains where additional work appears without supplying the headline latency. The corrected typed-event reader retains CPU user annotations and CUDA annotations separately; an earlier v3 summary accidentally overwrote CPU events with same-name CUDA annotations. Its four raw files and the correction receipt remain preserved. Version 4 uses two warmups and one capture per fixture, with zero optimizer updates. CPU scope time and attributed device time below are nested and overlap; neither column is an exclusive partition, and CUDA annotation durations are not kernel busy time. Profiler overhead is substantial, so the seven profiler-off complete-call samples above remain authoritative.

| High-M profiler scope | Thermal D0 CPU / attributed device, ms | Thermal fitted 0 CPU / attributed device, ms | Wind D0 CPU / attributed device, ms | Wind fitted 0 CPU / attributed device, ms |
|---|---:|---:|---:|---:|
| Complete annotated call |87.624 /4.668 |137.466 /6.698 |55.535 /4.865 |89.382 /6.253 |
| Organizer |35.810 /1.570 |54.532 /1.920 |37.730 /1.708 |56.181 /2.084 |
| Original full reader / selected decoder |3.115 /0.983 |18.882 /2.098 |2.850 /2.671 |16.527 /3.603 |
| Selected source plan |N/A |1.425 /0.094 |N/A |1.523 /0.103 |
| Selected fine values |N/A |0.775 /0.221 |N/A |0.882 /0.558 |
| Selected output projection |N/A |7.014 /0.964 |N/A |7.090 /1.620 |

All fitted profiled calls use the partial selected path in one query tile. Thermal query planning includes native union preparation and nested scene/organizer work; its 92.435 ms fitted CPU scope cannot be added to the organizer or complete scope. Wind uses physical-velocity extraction rather than a Thermal local-response event. Missing full-reader sort/scorer instrumentation is unavailable, rather than measured zero. The complete-call regressions are consistent with additional organizer, planning, contraction and dispatch work offsetting fewer fine values, but this one-capture diagnostic does not isolate the causal cost of each mechanism.

## Protocol and preservation

Thermal uses fixed TRAIN150/DEV22, effective 48/micro 24, seed 0; 1024 primary queries plus unchanged native auxiliaries. Wind uses fixed TRAIN72/DEV24, effective 16/micro 8, initialization 0, sampling 42; 6144 volume plus 2048 hub-band queries and the successful formal 0.75/0.25 objective. The older five-role development objective is historical context, not this training recipe. Wind's omitted TRAIN M18 stratum remains a limitation. Both candidates use TRAIN-only transforms, constant all-parameter AdamW 3e-4, weight decay 1e-5 and clip 1.

Thermal preserves the physical output law \(\widehat T(x, h)=\sum_{i\in\mathrm{present}}K_i(x)h_i\), with signed source coefficients and exact zero at zero heating. Its nonlinear u/v/p head is independent of heating; curl and the conductivity/distance q proxy are extracted from the native fields. Wind preserves \(\widehat U(x)=U_{\mathrm{profile, TRAIN}}(z)+u_{\mathrm{ref}}\, s_{\mathrm{safe}}\odot\widehat r(x)\) in the native unrotated frame. The residual uses the stored scale; the absolute-velocity normalizer mean is not added a second time. These [Thermal](../../../../Case_ThermalChannel/src/channelthermal/content_hypergraph.py) and [Wind](../../../../Case_WindFarm/src/windfarm/content_hypergraph.py) adapter laws differ consequentially from some historical direct-output objectives, as documented in Round2. Reusing the successful formal sampling/loss semantics does not make all classic native output laws or training histories identical.

Each 2500-epoch Thermal trajectory has 375000 case visits and 10000 ordinary optimizer updates; each Wind trajectory has 180000 visits and 12500 updates. Histories are contiguous from 1 through 2500, and case-order/query-sampling fingerprints and per-epoch visit/update budgets match exactly across D0/D-scale. A resumed segment summary counts only that segment and must not be mistaken for cumulative work. Validation remains every 100 development epochs, declared checkpoints every 100, curves every 10, full histories and latest/best aliases retained. The frozen implementation also overwrites the single latest alias on curve epochs; it does not create extra checkpoints or ordinary validation files.

Sparse children may branch only from the chosen dense host's exact e500 checkpoint, inheriting physical weights/moments/RNG and absolute age. Their 500-epoch prefix is reused evidence; each adds 2000 physical epochs to absolute 2500. Initial auxiliary fitting and periodic current-model teacher reads have separate counters and caps. The less aggressive TRAIN-calibrated point is the training default; another point remains an inference diagnostic. No mature reference initializes these fits.

Both children retain the exact 500-epoch D0 prefix and add 2000 physical epochs:8000/10000 new ordinary updates and 300000/144000 visits for Thermal/Wind, respectively. All 2500 case/query fingerprints match the dense reference. Each has 256 separate initial auxiliary updates, then 1000/1250 one-case refresh events on the fixed 8-update cadence. Initial plus refresh dense teacher rows are 924160/3215616, only 0.01884/0.02162% of new actual primary fine rows, well below 5%; those counts do not include a fictitious dense shadow on every step.

The initially calibrated auxiliary gradient share is not a permanent 5% cap. With coefficients frozen,5447/8000 Thermal and 7782/10000 Wind later updates exceed 5%; median shares are 7.99/8.01%, p90 36.92/16.79%, and maxima 407.75/140.64%. This is a material optimization limitation rather than a hidden retuning. Last 100 median complete TRAIN epochs cost 2.227/2.121 seconds versus D0 1.449/1.469, ratios 1.536/1.443. The new physical TRAIN portions total 4458.64/4535.41 seconds, with auxiliary setup, DEV and process overhead accounted separately. No claim of faster training follows from fewer fine rows.

Whole-epoch optimizer-boundary allocated/reserved peaks are 6.517/25.907 GiB Thermal and 9.940/46.479 GiB Wind. These include the actual auxiliary manager and persistent allocator cache across a long run, unlike prediction-only B8 backward. Wind reserved memory approaches device capacity without an OOM; the production-probe headroom check is not evidence that the same free-space headroom persisted throughout training. Final checkpoints, all declared 100 milestones, latest/best aliases and full histories remain retained.

Physical GPUs 0 and 2 are authorized for this Round3 study, using the ModularDT environment. GPU 1's unrelated job is protected. The phase ceilings are 20 aggregate GPU-associated process-hours and 14 elapsed hours, with the final 2 reserved for reporting/cleanup; new disk ceilings are 12 GiB on data and 1 GiB on home, with 20 GiB filesystem headroom. Exact shared caches are reused without dataset copies. The Round3 run root and local evidence remain ignored by Git.

The completed GPU ledger records 8.503 aggregate process-associated hours across 86 calls, including preparation, failures and final diagnostics; this is not GPU kernel busy time. All use physical 0/2, and none remains running. The post-cleanup audit is 8.47 elapsed hours after first scientific GPU work, inside the phase ceiling and reporting reserve. Disposable engineering work uses 117 actual updates and 178 conservatively charged updates of the 400 allowance; each of the four scientific child initializations consumes its separate 256-step auxiliary allowance, including failed S-Utility starts. No additional full-TRAIN run, automatic continuation or monitoring campaign is installed.

Final cleanup removes 654 owned regular payloads totaling 44,935,188 bytes, 47 temporary symlinks and 298 empty directories, with zero skips. This includes 71 superseded or redundant visual exports totaling 30,530,951 bytes; retained PDF masters cover the removed presentation copies. Histories, every declared checkpoint, latest/best aliases, native numerical arrays, failures/provenance, shared caches and source scripts remain. Original per-family figure indexes retain their historical bytes; the [current retained-figure index](../../../../diagnostics/generated/attention_round3_20261015T152913Z/RETAINED_FIGURE_INDEX.md) and cleanup receipt map their 15 obsolete PNG links to retained masters. These totals describe the final closeout batch, separately from earlier documented cleanup.

The ignored round root occupies 4.312 GiB of unique allocated regular-file blocks, deduplicating shared immutable runtime hardlinks, against its 12 GiB ceiling. The post-cleanup audit records 89.85 GiB free on data and 42.65 GiB on home/tmp, above the 20 GiB floor; no dataset copy or new persistent dataset cache is created. Whole-filesystem free-space changes include concurrent application activity and are not attributed wholesale to experiment outputs. All 3831 sealed historical/input files retain SHA256, inode, size and mtime; all 1677 frozen runtime files remain read-only and hash-identical. Six final epoch 2500 checkpoints and both D0 epoch 500 parents are rehashed unchanged. A consistent device-number namespace remapping is disclosed in the protection receipt rather than claimed identical. The Round2 report hash is unchanged, Wind TEST remains locked and solver usage stays 326/326.

The final tables receive independent saved-result/source checks, and every selected raster is visually inspected. Markdown source, local image/PDF links and decoded artifacts are checked using a local Markdown renderer, Pillow and pdfinfo; no browser rendering check is available in this environment. The [evidence index](20261016_004810Z_HONF_Attention_Round3_Evidence_Index.md) retains tested command recovery, qualification/source audits, exact cleanup receipts and final Git-upload verification. Generated artifacts and one-time helpers stay local; only durable implementation, tests, policies and reports belong to the working branch.

## Inverse and response limits

No inverse campaign or new solver is run. Solver usage remains 326/326. Wind TEST and unexposed targets remain locked. Saved Thermal response checks assess field and increment fidelity against their declared stored references; exact affine software arithmetic alone is not physical-response accuracy or inverse readiness. Selected application/JVP/VJP must retain signed coefficients/rank factors without ordinary dense K export, reject stale ownership and out-of-envelope endpoints, and allow larger tangent directions. An explicit full-access fallback is a separate measured function.

The new retained-formal reference replay uses A2-D TRAIN600 e5000 on the same three saved response families as D0 TRAIN150 e2500. Mean signed fluid-increment RMSE is 0.0234169/0.00522934/0.00103404 for 0291/0294/0687, compared with 0.0278781/0.0104657/0.00141708 for the development D0. Both have exact-zero direct heat-to-flow increments. This is practical reference distance across different populations and ages, without a matched-training or causal attribution claim. It also does not independently validate individual K columns. The original missing 0277 baseline and the 326/326 solver ledger are preserved.

D-scale has a modest saved-response benefit that the endpoint field table alone would miss: mean signed fluid-increment RMSE improves 3.30/2.82/8.49% on 0291/0294/0687, respectively. However,0294 interface/material increments worsen 24.36/5.18%, and all three retained formal-reference fluid increments remain better. The common saved families, valid receivers, endpoint-peak quantities and separate FP32/FP64 arithmetic checks are retained in `receipts/saved_dense_response_quantitative_review_v1.json`. This bounded response evidence does not override the predeclared 2500 physical-field host decision.

The mature S-Complete fitted policy has a positive but limited saved-response result. Mean signed fluid-increment RMSE in packed-native weighted temperature units is 0.0265542/0.00630892/0.00132940 on 0291/0294/0687, improving 4.75/39.72/6.19% over same-age D0. Interface/material increments improve on 0291/0294, but worsen 4.86/20.04% on 0687. Peak-change mean absolute errors improve from 0.0863245/0.00749929/0.00115693 to 0.0706154/0.00402983/0.000945758. The same-weight full policy is much worse on fluid increments,0.100948/0.0394441/0.00654417. These three saved geometry families show a response benefit alongside worse population endpoint fields; they are not a substitute for the failed native field criteria.

Every model has exact-zero direct and endpoint heat-to-flow increments. Fitted FP64 endpoint-minus-baseline versus precise factorized-increment checks have 0 failed elements among 245760 fluid-channel,5632 interface and 136224 material comparisons at 1e-9 absolute/1e-7 relative tolerance. Its retained FP32 checks still fail 609/2118/6877 elements under the historical 2e-6/2e-5 test. Those cancellation failures remain visible rather than being replaced by the FP64 result. On0294, fitted peak-change signs disagree on 2/14 source-endpoint quantities versus D0 4/14; zero/tiny reference changes and the nonlinear endpoint maximum prevent interpreting this as a robust direction certificate. Fitted 0687 has no new peak-sign mismatch; all-full has 4/20. No fresh physical reference or inverse optimization is performed.

The fitted response path retains selected signed coefficients plus rank factors and does not materialize dense scalar K. Avoiding dense K does not guarantee smaller retained storage: its direct representations contain 27843/28299/47759 selected coefficients plus 131168/131200/131248 completion factor entries on 0291/0294/0687, versus 40960/57344/81920 dense K entries for full access. These are scalar-entry counts, not comparable byte or latency measurements. The rank 16 query factors are substantial when the scene contains only 5/7/10 sources. The complete repeated-heat benchmark and B8 memory are reported separately. All response requests, native quadrature/valid supports, signed deltas and saved references match exactly across D0/full/fitted, as checked in `receipts/scomplete_e2500_saved_response_main_review_v1.json`.

![Saved Thermal response reference, D0 TRAIN150 e2500 and retained formal D TRAIN600 e5000; native increment maps, signed error summaries, endpoint peaks and separate arithmetic checks](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/response_retained_formal_context_v3/family6_saved_thermal_response.png)

Figure 6a ([PDF](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/response_retained_formal_context_v3/family6_saved_thermal_response.pdf)): the 0291 positive transfer has weighted relative-L2 error 0.2952 for development D0 and 0.2480 for the retained formal reference on 7837 common valid receivers. Endpoint peaks are recomputed on common material support. FP32 endpoint subtraction retains 721/245760 and 715/245760 failures against this replay's historical 2e-6/2e-5 arithmetic allowance; precise FP64 increments have 0/245760 for each model against 1e-9/1e-7. These arithmetic checks are separate from Round3 executor qualification. The three stored analytic-wake/shared-grid families are not CFD or inverse evidence; whiskers show the two signed directions, without a confidence claim. PDF and raster were visually inspected after fixing annotation overlap.

![Saved Thermal increments, nonlinear endpoint peaks and independent arithmetic checks for D0 and the two same-weight S-Complete policies at epoch 2500](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/figures/response_scomplete_e2500_v3/family6_saved_thermal_response.png)

Figure 6b ([PDF](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/figures/response_scomplete_e2500_v3/family6_saved_thermal_response.pdf)): for the 0291 positive transfer, fitted response weighted RMSE is 0.02655 and relative-L2 is 0.2811, versus D0 0.02788 and 0.2952. The lower-left summary retains both signs for all three saved families, including the much worse all-full adapted function. The endpoint-peak panel recomputes maxima at both endpoints. The right panel keeps the 609/245760 fitted FP32 cancellation failures alongside zero FP64 failures; arithmetic agreement does not imply physical reference agreement. All labels, scales, source ages and reference limits are visible in the inspected PDF/raster.

## Final decision and figure index

All declared healthy development trajectories reach literal 2500. D0 remains both dense hosts; neither sparse point passes the native-field criteria, so no conditional fresh full 1000 run is launched. S-Utility remains unavailable after its capped initialization failures. S-Complete provides real selected execution, actual read dependence and a bounded saved-response benefit, with native endpoint regressions and no qualifying complete-cost/memory gain. The final evidence distinguishes those measured outcomes from arithmetic correctness, and preserves classic reference distance without imposing a replacement gate. No inverse result is claimed.

For prediction, the useful result is a fully measured negative source-scale/sparse comparison, with a real q-proxy and saved-response benefit that does not transfer to the required endpoint fields. For organization, individual source identities and selected-read effects are verified, while all-source cheap/context influence and gate attenuation limit any interpretation as physical structure. For inverse use, only saved response arithmetic and reference-distance checks exist; no inverse designs, new solver data or readiness claim is supported.

The main research limitations are one initialization, exposed small development panels, correlated Wind directions, missing TRAIN M18 layouts, limited saved Thermal response families, later auxiliary-gradient shares above the initial calibration target, and a co-adapted full function that can be inaccurate. This implementation also retains rank 16 factors, sorting/gathering and all-source planning costs that are substantial at these small source counts. The next scientific revision would need a new explicitly budgeted experiment that controls these costs and preserves native fields; this round schedules no follow-up training or evaluation campaign. The unused fresh full-TRAIN workflow is not validated by these negative development results and is not the reason qualification fails.

| Figure family | Selected PDF master | What to inspect |
|---|---|---|
|1. Learning, native accuracy and work |[Two history pages](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/round3_figure_families_primary_policy_e2500_final_v4/family1_histories.pdf) |Literal ages, inherited versus new work, transient versus mature native effects |
|2. Thermal native fields and residuals |[Dense context](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/dense_e2500_v1/family2_thermal_fields_preview_and_children.pdf), [trained sparse comparison](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/round3_figure_families_primary_policy_e2500_final_v4/family2_thermal_child_fields_e2500.pdf) |Fixed0291/0687 fields, shared native scales, surface/q extraction and mixed local effects |
|3. Wind native planes and cuts |[Dense context](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/wind_dense_e2500_v1/family3_wind_native_hub_planes_comparison.pdf), [trained sparse comparison](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/wind_d0_scomplete_e2500_primary0_fitted_v3/family3_wind_native_hub_planes_comparison.pdf) |All three component predictions/errors and native downstream crosswind profiles |
|4. Actual organizer and packets |[Four trained-scene pages](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/figures/source_organization_scomplete_e2500_v1/family4_source_organization.pdf) |Source IDs, original attention, fine gates, padding and restored read interventions |
|5. Accuracy, complete cost and memory |[Six-panel overview](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/figures/accuracy_cost_e2500_final_v2/family5_accuracy_cost_overview.pdf) |Native error versus full-call timing, absolute fixedB8 three-Q memory, actual fine execution |
|6. Saved Thermal responses and limits |[Retained formal context](../../../../diagnostics/generated/attention_round3_20261015T152913Z/figures/response_retained_formal_context_v3/family6_saved_thermal_response.pdf), [trained sparse comparison](../../../../diagnostics/generated/attention_round3_20261015T152913Z/evaluation/figures/response_scomplete_e2500_v3/family6_saved_thermal_response.pdf) |Signed saved increments, nonlinear endpoint peaks and separate cancellation/physical errors |

All figures use retained numerical evidence and are linked locally through the ignored round root; generated figures, data and one-time renderers are excluded from Git. The [evidence index](20261016_004810Z_HONF_Attention_Round3_Evidence_Index.md) records source/checkpoint hashes, tested command recovery, failure scope, budgets and cleanup receipts. A fresh checkout needs the retained local artifact store to display these images. The Round2 report and historical references remain unchanged.
