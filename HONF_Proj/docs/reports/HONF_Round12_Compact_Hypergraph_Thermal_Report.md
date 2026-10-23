# HONF Round 12: Compact Hypergraph Retrieval for Thermal U-H

Round12 implements a separate retrieval-group axis over the inherited Thermal U-H encoder and completes the full common20 → matched R0/R1 local100 experiment. The software constructs group-to-original-record memberships and query-to-group scores directly, preserves the original physical values and Thermal source law, and supports ordinary checkpoint adaptation, forks, separate clocks, and recovery. The learned candidate is a negative scientific result: R1 accepts no merges on the 113 surveyed scenes and has higher VAL89 mean and p90 RMSE than R0 on every measured field. Retain R0 as the matched research control and keep the original U-H/U-S references; do not promote R1 as successful compression or a reliable inverse model.

The completed campaign contains 2,340 new scientific optimizer updates, 108,000 TRAIN case-visits, and 110,592,000 primary query presentations. It includes the single local40 cut activation and ordinary recovery through local100. No U-S amendment, Level U factorization, Wind work, locked TEST access, new solver, or long formal extension was launched.

## Lineage, protocol and executed budget

The literal source is `HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_u_h_full5000_v1/checkpoints/epoch_1000_model.pt` (44,140,718 bytes; existing SHA-256 `2efd008f4135c7e5c7d909e0174b31e6ad5b05a058c0e68914ffc727d29134d7`). The frozen U-S reference is its sibling `thermal_u_s_full5000_v1/checkpoints/epoch_1000_model.pt`. The source U-H epoch is 1000, with internal joint/controller epoch 4000, update index 12 and macro clock 13000. The A2-D adapter label is the expected implementation identity of this U-H research lineage.

Both children inherit the full TRAIN600 membership and saved TRAIN normalizer without refitting (`normalization_key=7ab3441ba2616c68224f9d829a92e2db4c59658b45d3f1acf4c535333896939b`). Sampling uses the inherited dataset namespace, seed 0, 1,024 primary queries per case, effective batch 48, microbatch 24, the same native losses, response schedule and M/E/context controller state. Native evaluation uses the canonical exposed VAL89 panel, excluding 0273, with M3/M5/M7/M10 counts 24/25/25/15. VAL89 is exposed development evidence, not independent TEST. TRAIN24 is the pre-existing input-stratified training panel; its fixed representatives are 0006/M1 and 0254/M12.

Common adaptation starts fresh AdamW moments using the actual parent formal_joint groups: LR 3e-4, WD 1e-5, betas (0.9,0.999), epsilon 1e-8 and gradient clipping 1.0. The source provider-summary WD says 1e-4; the serialized optimizer groups define the executed setting. This is an adaptation from parent weights, not an exact continuation of the parent optimizer trajectory. The local learning rate remains constant over the declared 100-epoch horizon.

| State | Sampler epoch | Controller epoch | Macro clock | Scientific work since source |
| --- | --- | --- | --- | --- |
| Source U-H e1000 | 1000 | 4000 | 13000 | Inherited source |
| Shared common20 | 1020 | 4020 | 13260 | 260 updates; 12,000 case-visits |
| Each local40 stop | 1040 | 4040 | 13520 | 520 cumulative updates per endpoint |
| Each local100 endpoint | 1100 | 4100 | 14300 | 1,300 cumulative updates; 60,000 case-visits per endpoint |
| Unique executed campaign | — | — | — | 260 + 1,040 + 1,040 = 2,340 updates; 108,000 visits |

All 100 R0/R1 epoch records match on case order, query sampling, sampler epoch, case-visits and update counts. The common prefix was executed once. At the fork, only the compact arm code differs; model parameters, Adam state, RNG and history otherwise match. Two TRAIN microbatches of 24 scenes calibrate the structural terms without an optimizer update. A separate disposable one-update software qualification is excluded from the retained scientific history and budget.

## Implemented representation and learning

Original records remain on N=M+E. Typed destination groups use a separate live K axis, padded only to the batch maximum. A direct C[B,K,N] maps each live group to eligible original donors; independent typed query scorers form Q×K scores, followed by one entmax over all live groups. The final original-record access is pi×C, combined with the existing local M envelope. The retired native N-seed retrieval directory and its pre-E M seed scores are absent from this path. Original donor masks, coordinates, lengths, quadrature measures, individual query-conditioned values and every physically present M source column are retained, including zero-heat sources. Positive-measure eligibility for E information is retained.

Two original MM→EM→EE→ME applied contextual cycles remain in the encoder and their weights continue to learn. The new retrieval C is computed from final prepared record states; it is not the encoder’s applied contextual incidence. E information is read first; the E-conditioned M information read and original signed affine Thermal source-response law remain. Group descriptors and routing geometry are recomputed from current states and original geometry. Descriptors summarize retrieval controls; physical values are read on the original record axis.

Normalized-coordinate lexicographic order forms adjacent same-type singleton sibling candidates, with odd carry and exact-coordinate tie bundles left unmerged. One shared small compatibility MLP is learned for each type using symmetric input-only features, geometry and operating conditions, without source IDs, heat, targets or per-case parameters. Local21–40 blends sibling descriptors and routing geometry by alpha×sigmoid(b), with alpha ramping to one. At local40, R1 activates the strict b>0 cut exactly once and freezes all eight compatibility parameters. No threshold search, forced merge or post-VAL retuning was performed. R0 stays singleton throughout.

The tie penalty averages eligible sibling-row differences with equal weight for nonempty donor types; capacity uses (N−sum(sigmoid(b)))/N. Their VJPs are accumulated once per primary scene alongside native macro gradients before clipping; response duplicates are excluded. Both terms vanish after the cut. At common20 and alpha=1, the tie native/raw gradient norms were 0.0300645/0.00336339, giving coefficient 0.178774665; capacity norms were 0.000489123/0.0979320, giving 9.9890278e-5. These are the planned 2% weak-gradient calibrations, not optimized hyperparameters.

The existing research runner now reads research checkpoints through the maintained safe run reader and adapter reconstruction, records compatible/new/excluded parameters, inherits saved normalization, forks the common state and continues separate local, sampler, controller and macro clocks. Unused native-seed selector parameters are excluded from the active optimizer. It uses the existing training engine, per-run writer lock and ordinary interruption recovery; no additional engine, hash-admission service, source-freezing framework or GPU reservation system was added.

## Predictor: gains, misses, evidence and next step

Gains: R0 improves the inherited U-H mean errors for all four flow fields, fluid T and q proxy. Misses: R0 does not improve U-H surface T, material T or module-peak mean errors, and U-S remains the stronger surface/material reference. R1 is worse than R0 on mean and p90 for all nine quantities. This single matched experiment supports retaining R0 as a research control, not a claim of statistical superiority over the parent lineages; parent-versus-child comparisons also include extra training and the new retrieval scorer.

Every entry below is an equal-case RMSE statistic in its own packed native output units; no cross-field aggregate is formed. Fluid fields use the full native 128×64 grid with physical fluid masks, surface and outside roles use the inherited 64 receivers per present module, and material T uses every stored material point. The peak endpoint is recomputed per physically present module from material predictions. q proxy is the inherited native proxy, not independently validated physical heat flux. Four exported representatives have elementwise matching receiver coordinates, targets, valid masks, physical IDs and peak truth across all four models; the full89 case IDs, module counts and terminal valid-entry counts align.

| Field | U-H e1000 mean / p90 | U-S e1000 mean / p90 | R0 local100 mean / p90 | R1 local100 mean / p90 | R1 mean vs R0 |
| --- | --- | --- | --- | --- | --- |
| u | 0.0119199 / 0.0171878 | 0.010685 / 0.0161179 | 0.0105283 / 0.0157443 | 0.0137592 / 0.0166657 | +30.7% |
| v | 0.00164392 / 0.002188 | 0.00162947 / 0.00206717 | 0.00153183 / 0.00205454 | 0.0018379 / 0.00236064 | +20.0% |
| p | 0.00640242 / 0.00982884 | 0.00598773 / 0.00926582 | 0.00575354 / 0.00895573 | 0.00700722 / 0.00952976 | +21.8% |
| omega | 0.02954 / 0.0423427 | 0.0303119 / 0.0461279 | 0.0275909 / 0.0402663 | 0.0310708 / 0.0424439 | +12.6% |
| Fluid T | 0.39411 / 0.543919 | 0.421801 / 0.605792 | 0.365555 / 0.529687 | 0.564668 / 0.816736 | +54.5% |
| Surface T | 0.308197 / 0.596402 | 0.294918 / 0.572164 | 0.316832 / 0.621592 | 0.626164 / 1.04987 | +97.6% |
| Material T | 0.324324 / 0.637418 | 0.301664 / 0.544238 | 0.343554 / 0.657756 | 0.704775 / 1.09838 | +105.1% |
| q proxy | 1.24966 / 2.24418 | 1.1694 / 2.16314 | 1.21355 / 2.17976 | 1.47613 / 2.34926 | +21.6% |
| Module-peak endpoint | 0.313373 / 0.76219 | 0.312638 / 0.654966 | 0.31949 / 0.727995 | 0.680675 / 1.12233 | +113.1% |

| Field | U-H worst (case) | U-S worst (case) | R0 worst (case) | R1 worst (case) |
| --- | --- | --- | --- | --- |
| u | 0.0240632 (0293) | 0.0285953 (0293) | 0.026752 (0293) | 0.0300068 (0293) |
| v | 0.0028546 (0295) | 0.00360367 (0293) | 0.00269257 (0302) | 0.00301217 (0293) |
| p | 0.0178847 (0285) | 0.0133298 (0293) | 0.0152856 (0285) | 0.0179223 (0285) |
| omega | 0.0506718 (0297) | 0.0593591 (0297) | 0.0498433 (0284) | 0.0546099 (0284) |
| Fluid T | 0.921186 (0297) | 0.883843 (0297) | 0.813639 (0297) | 1.11433 (0298) |
| Surface T | 1.60054 (0297) | 1.39289 (0297) | 1.482 (0297) | 1.53832 (0298) |
| Material T | 1.8876 (0297) | 1.66222 (0297) | 1.80136 (0297) | 1.66925 (0297) |
| q proxy | 4.5507 (0297) | 3.78986 (0297) | 4.93162 (0297) | 4.56497 (0297) |
| Module-peak endpoint | 2.16024 (0297) | 1.77944 (0297) | 2.0944 (0297) | 1.98692 (0297) |

R1 misses both descriptive bands (+10% mean and +15% p90 versus R0) for each priority temperature and peak quantity. Its worst material, q-proxy and peak cases are numerically lower than R0’s worst cases, but that does not offset the higher means/tails. Every M-stratum mean is higher for R1 on all nine fields. The following priority table gives R0→R1 mean RMSE by module count; full per-field mean/p90/worst and M strata remain in the saved boards and comparison JSON.

| VAL stratum | Fluid T | Surface T | Material T | Module peak |
| --- | --- | --- | --- | --- |
| M3 (24 cases) | 0.297285 → 0.408923 | 0.17534 → 0.356691 | 0.217944 → 0.440117 | 0.155056 → 0.397916 |
| M5 (25 cases) | 0.362135 → 0.484564 | 0.320589 → 0.537037 | 0.337867 → 0.601887 | 0.324273 → 0.555058 |
| M7 (25 cases) | 0.434747 → 0.684331 | 0.464202 → 0.843102 | 0.49974 → 0.931693 | 0.511402 → 0.933377 |
| M10 (15 cases) | 0.365167 → 0.747925 | 0.291344 → 0.844302 | 0.293698 → 0.921514 | 0.25476 → 0.92128 |

TRAIN24 also has higher R1 mean, p90 and worst-case errors on all nine fields; its mean ratios for fluid/surface/material T and peak are 1.651/2.515/2.522/3.144. This is evidence on already exposed training scenes, not an independent generalization test. Next step: use the retained R0 and U-H/U-S checkpoints for any separately chosen predictor study; this round does not support continuing R1 as a successful compact model.

![Nine native field RMSE panels for parent and terminal checkpoints](../../diagnostics/thermal_compact_round12_20261023/report_figures/thermal_native_field_mean_p90.png)

Figure 1. Canonical exposed VAL89, U-H/U-S e1000 and matched R0/R1 local100: circles show equal-case means and diamonds p90. R1/R0 mean ratios are 1.545 for fluid T, 1.976 for surface T, 2.051 for material T and 2.131 for module peaks. Each panel retains its own native units and scale; the figure supports a negative matched predictor result for R1.

![Native reference, R0 and R1 fluid velocity and temperature maps with signed residuals](../../diagnostics/thermal_compact_round12_20261023/report_figures/thermal_fluid_maps_case_0692.png)

Figure 2. Fixed VAL representative 0692/M10 at local100, with identical within-field prediction/reference color limits and common symmetric error limits. Physical fluid-T RMSE is 0.30718 for R0 and 0.62306 for R1; u RMSE is 0.010178 and 0.012194. White module interiors are excluded physical-fluid support. This saved representative illustrates the regression and does not replace the full89 field statistics.

## Organizer: gains, misses, evidence and next step

Gains: the direct K×N and Q×K implementation is exercised by actual native learning, compact states reload, and receiver roles reuse the same prepared membership bank. Misses: the learned policy accepts zero sibling pairs on VAL89 plus TRAIN24, 113 distinct scenes per arm. All live K equal original present N (193–204); both compact allocation and scorer slots therefore retain singleton size. The one cut and 60-epoch recovery are complete, but recovery of a genuinely reduced learned bank is unestablished. The synthetic K<N software tests demonstrate capability only.

On TRAIN0254/M12, each model has K_M=12, K_E=192, N=204 and C shape 204×204. R0 has 6,226 exact-positive C entries and R1 6,019; most M-group rows reach almost every original donor, while E-group rows reach fewer donors. This overlap is learned retrieval organization, not identity or physical group causality. Across VAL89 each arm allocates 3,484,910 C score-array cells and 17,610 group slots before query expansion; these shape-derived counts are not FLOPs or latency.

| VAL89 quantity | R0 | R1 |
| --- | --- | --- |
| M positive-access fraction, equal-case | 75.6502% | 75.9406% |
| E positive-access fraction, equal-case | 36.0486% | 36.3517% |
| M actually executed / valid candidate rows | 4,276,224 / 4,276,224 | 4,276,224 / 4,276,224 |
| E actually executed / valid candidate rows | 139,984,896 / 139,984,896 | 139,984,896 / 139,984,896 |
| Encoder contextual pair rows | 6,969,820 | 6,969,820 |
| Retired native gather / pre-E M seed score rows | 0 / 0 | 0 / 0 |
| Direct query-to-group score rows | 144,261,120 | 144,261,120 |
| Retained M / E reader score rows | 4,276,224 / 139,984,896 | 4,276,224 / 139,984,896 |

The configured dense_weighted executor evaluates every valid original M/E nonlinear value row in these one-case native boards. Positive support is not executed-row savings; this is also not the different mathematical all-fine control. Original encoder quadratic work and retained E/post-E M reader scores remain. There was no OOM, execution fallback or repacking in the scientific runs. No steady native-call latency or memory superiority was established, and optional segment/cost probes were not pursued.

![Actual K by N memberships, unchanged singleton ancestry and donor support](../../diagnostics/thermal_compact_round12_20261023/report_figures/03_compact_organization.png)

Figure 3. TRAIN0254/M12, terminal local100. Top panels show the actual C matrices and exact-positive entry counts; bottom left shows original M/E geometry with zero merged ancestry groups, and bottom right shows donor support per typed destination group. Both matrices remain 204×204. Their visible structure does not establish compaction or physical causal edges.

![Saved query group mass, original record access, dense execution and role reuse](../../diagnostics/thermal_compact_round12_20261023/report_figures/04_query_access_and_work_v2.png)

Figure 4. The same 40 input-selected native-grid query samples show pi and final access for R0/R1 on TRAIN0254. Surface/outside/material rows are neural stencil neighbors, not physical interpolated response coefficients; rows marked neural can lie inside solids. Role mass uses the full prepared receiver union, including the domain grid, rather than the physical-fluid error mask. VAL89 filled markers show 100% executed nonlinear value rows despite lower positive support; no runtime reduction is claimed.

The terminal copy-only intervention disables the largest role-balanced query-mass group on TRAIN0254: M group index10, ancestry original M record5. Live K falls 204→203 while allocated K stays204, then calibration and entmax are recomputed. Disabled mass is zero and maximum remaining mass-normalization error is 2.80e-6. Physical fluid-T prediction changes by RMSE 0.250343 (max 2.10136), while original model/preparation state remains unchanged. This verifies executed dependence on a learned retrieval group, not physical causality or a learned compaction event; actual dense execution is unchanged.

## Cut boundary, live geometry and recovery

Before/after local40 checkpoints differ only in cut-active and policy-frozen buffers; weights, optimizer, RNG, history and clocks are unchanged. The two TRAIN and four VAL boundary scenes all retain K=N and accept zero pairs. Nevertheless, ending soft tying changes the represented function: maximum masked prediction-difference RMSE across the six scenes is 0.0026446 for fluid T, 0.0029854 for surface T, 0.0026100 for material T and 0.014565 for q proxy. Corresponding maximum original source-response coefficient difference RMSE is 0.000391 for fluid T, 0.000808 for surface T, 0.000626 for material T and 0.00353 for q proxy. Per-field signed RMSE changes and all aligned M columns remain in the component-wise boundary JSON; no mixed-unit metric is used.

Both branches clean-stopped at40 and resumed with ordinary checkpoint loading to100. R1’s eight compatibility parameters remain exactly equal to their post-cut values. Parameter/Adam/provider/RNG/history equality was also checked using actual terminal100→horizon200 disposable forks, with zero new updates. CPU fixtures cover state restoration and the cut transition; this campaign does not claim bitwise GPU equivalence for a crash inside an optimizer macro.

| TRAIN case | AD d(mean fluid T)/dx | Represented central FD | Relative difference | C change RMSE | pi change RMSE |
| --- | --- | --- | --- | --- | --- |
| 0006 | -0.0365042 | -0.0365016 | 7.11295e-05 | 0.00126671 | 7.69234e-05 |
| 0254 | -0.165492 | -0.165403 | 0.000538269 | 0.00105395 | 0.000165602 |

Legal horizontal module translations rebuild original scenes and use a fixed exterior-fluid receiver set across the entire probe interval. Both C and matched-ancestry pi/final gates change with geometry while accepted ancestry remains unchanged. Four decreasing finite-difference steps are retained, with the largest fixed-topology step specified as primary rather than choosing the best numerical agreement. The bounded legal scan finds no ordering or accepted-ancestry crossing on these two scenes, so no measured hard-cut jump curve is available. Row-max calibration, sorting and future accepted merges can still introduce seams; local AD/FD agreement is not global C1 regularity or physical geometry-gradient fidelity.

![Measured hard-cut changes and bounded live geometry derivative probes](../../diagnostics/thermal_compact_round12_20261023/report_figures/05_cut_and_live_geometry_v2.png)

Figure 5. Four VAL and two TRAIN scenes at local40 retain the same K before/after the single cut, while physical fluid T and individual source-response coefficients change finitely. Lower panels show the full decreasing-h derivative sweep on TRAIN0006/0254. Primary absolute AD−FD differences are 2.60e-6 and 8.91e-5; the smaller-step behavior exposes FP32 cancellation. No topology crossing was found within the legal scanned intervals, and no global smoothness claim follows.

## Response and inverse decisions: gains, misses, evidence and next step

Gains: all four current checkpoint families now have native saved-response and original-pool ranking evidence; no historical U10 response result is substituted. R1 has a somewhat lower median relative-L2 fluid-T geometry first-difference error than R0 and correctly classifies all six saved candidates’ feasibility. Misses: geometry responses remain uneven, mixed geometry responses remain weak, and none of the four models orders the original six objectives well. R1’s ranking MAE and selected-candidate reference regret are worse than R0. These measurements do not establish reliable inverse design.

| Response | U-H median / p90 / mixed median | U-S median / p90 / mixed median | R0 median / p90 / mixed median | R1 median / p90 / mixed median |
| --- | --- | --- | --- | --- |
| Fluid T | 0.440 / 1.093 / 1.551 | 0.493 / 1.080 / 2.069 | 0.433 / 1.083 / 1.689 | 0.411 / 1.094 / 1.589 |
| Surface T | 0.802 / 1.114 / 1.414 | 0.740 / 1.174 / 1.530 | 0.789 / 1.132 / 1.437 | 0.841 / 1.176 / 1.322 |
| q proxy | 0.837 / 1.002 / 2.938 | 0.823 / 1.165 / 3.428 | 0.811 / 0.962 / 3.723 | 0.831 / 0.979 / 3.758 |
| Material T | 0.824 / 1.449 / 1.438 | 0.720 / 1.297 / 1.540 | 0.814 / 1.377 / 1.492 | 0.786 / 1.374 / 1.363 |

Response references are the existing TRAIN families 0001/0318/0333/0348: eight saved geometry perturbations and two geometry-geometry quartets per family, the baseline and existing heat-transfer plus/minus controls, and the independent-heater null. Values above are dimensionless relative L2 for nonzero geometry differences, computed per perturbation before aggregation. Very small independent-heat physical residuals are reported in absolute units in the retained result rather than inflated relative errors. The affine source law enforces model heat superposition; it does not prove correct nonlinear geometry responses or physical pair interactions.

The response panel includes 16 single-axis and 16 diagonal differences per model. The lower R1 fluid-T relative-L2 median does not mean lower absolute response error: across all32, R0 absolute RMSE median/p90 is 0.14722/0.26057 and R1 is 0.15100/0.26398 in native T units; R1 relative-L2 p90 also rises slightly. The independent-heater null has physical maximum absolute residual 1.91e-6 in fluid T, 9.54e-7 in surface T, 1.91e-6 in q proxy and 1.91e-6 in material T, with exact-zero reference flow response. Model null residuals are at most 1.1e-13 in each component’s own units. This near-null check supports the imposed affine law only.

| Model | Objective MAE | Spearman | TP/FP/TN/FN | Selected | Reference feasible | Feasible regret |
| --- | --- | --- | --- | --- | --- | --- |
| U-H | 0.207589 | -0.257 | 1/0/2/3 | c00 | True | 0.524815 |
| U-S | 0.728951 | -0.029 | 4/0/2/0 | c05 | True | 0.54532 |
| R0 | 0.317005 | -0.257 | 1/0/2/3 | c00 | True | 0.524815 |
| R1 | 0.62299 | -0.371 | 4/0/2/0 | c03 | True | 0.588267 |

The original six M3 candidates retain their IDs, heat total 3.479647219181061, tau=2 smooth native module-peak objective and pressure limit 0.13649363704785286 in dataset units. Regret is the selected saved physical-reference objective minus the best reference-feasible objective, and would be undefined for a reference-infeasible selection. No-feasible-prediction would mean no recommendation. U-S and R1 classify the six feasibility flags correctly, but all four objective correlations are negative; even each selected reference-feasible candidate has substantial regret. No new candidate, heat-allocation optimum or physically verified design was generated. Pressure-infeasible geometry is not repaired by changing heat.

## Actual fixed-layout response-cache reuse

For each of the four models, each family baseline is prepared once and explicitly promoted through the maintained prepare_affine_application API to PreparedAffineQueryV2. Twelve fresh native absolute reference calls, four promotions and twenty retained-factor applications per model cover baseline plus the two existing admissible heat vectors and their increments. Cached apply counters show zero neural-reader recomputation, while retained coefficient-entry consumption and native extraction are reported separately. The software fixture additionally raises if decoding or the direct reader is re-entered after promotion. This capability is shared by the parent references and both children, not a unique compact-hypergraph gain.

| Endpoint (own native units) | U-H cache parity / physical RMSE | U-S cache parity / physical RMSE | R0 cache parity / physical RMSE | R1 cache parity / physical RMSE |
| --- | --- | --- | --- | --- |
| Fluid T | 7.47e-16 / 0.45 | 4.29e-16 / 0.493 | 7.76e-16 / 0.461 | 7.44e-16 / 0.857 |
| Surface T | 1.04e-15 / 0.361 | 5.96e-16 / 0.317 | 1.28e-15 / 0.385 | 1.1e-15 / 1.1 |
| Material T | 1.17e-15 / 0.428 | 6.39e-16 / 0.35 | 1.11e-15 / 0.392 | 1.13e-15 / 1.21 |
| q proxy | 1.94e-14 / 1.27 | 9.72e-15 / 1.24 | 2.17e-14 / 1.22 | 2.04e-14 / 1.78 |

The table reports the worst per-endpoint RMSE over eight saved heat endpoints in each component, separating cached-versus-fresh FP64-accumulation agreement from fresh-versus-physical accuracy. Saved default-precision endpoint-difference comparisons are retained separately and include subtraction/FP32 effects; they are not exact bitwise identities. The older precision-promoted increment summaries did not populate affine_query and therefore did not demonstrate cache reuse; ACTUAL_CACHE_INTERPRETATION.md records that correction and only *_actual_cache artifacts support this claim. No complete-latency speedup or new physical validity follows from successful contraction.

For cached increments versus the saved default-precision fresh differences, worst maximum absolute discrepancies over the four models are 1.08e-4 in fluid T, 7.59e-5 in surface T, 8.81e-5 in material T, 9.34e-4 in q proxy and 3.87e-6 in omega, each in its own native units. These larger cancellation/precision differences remain visible alongside the direct fresh FP64 absolute-endpoint comparisons; they are not silently passed as bitwise parity.

![Current checkpoint geometry responses, original six candidate ranking and actual cached source response reuse](../../diagnostics/thermal_compact_round12_20261023/report_figures/06_saved_responses_and_ranking_v2.png)

Figure 6. The top panel contains 32 geometry first differences per model and role; lower left contains eight mixed geometry differences, lower center retains the original six objective/pressure controls, and lower right contrasts true cached absolute-endpoint parity with physical temperature errors. R1’s fluid response relative-L2 median improves slightly but its objective correlation is −0.371 and selected feasible regret is 0.58827. Reuse parity is much smaller than physical error and is available to all four models. Saved arrays provide reference truth; there is no new solver or verified design.

Next step: retain the negative R1 result, R0 control and original H/S checkpoints. A separately authorized future policy investigation would need a useful learned K<N bank before claiming compression recovery; continuing the frozen input-only policy on these unchanged scenes does not create accepted pairs. Retain U-S as a surface/material and feasibility reference, while recognizing that its saved-pool objective ranking is also poor. This round supplies user-controlled continuation entry points and starts no additional campaign.

## Validation, cost scope and continuation commands

The focused compact-core/native-adapter/additive-loss/inherited-normalizer tests passed 26 tests, the runner/fork/resume tests passed 28, and the final evaluator file passed 22. The core checks include literal FP64 membership and typed all-K score parity and gradients, uneven padding and eligibility, stale metadata, exact cut-state restoration and native backward. Existing continuous/unified compatibility checks also passed in the implementation round. Real GPU learning, saved-weight native fields, the cut boundary, actual terminal forks and actual affine-cache applications supply execution evidence beyond fixtures.

Scientific train-loop time summed over unique physical epoch executions is 7826.942 seconds (2.174 GPU-worker hours); this excludes evaluation, calibration, model loading, serialization and disposable qualification. GPU1 and GPU2 are RTX6000 Ada devices; GPU0 and unrelated runs were untouched. The disposable low/high-M macro qualification had finite nine-role objectives and peak allocated memory 4,250,391,552 bytes, but it included initialization/loading and is not a steady-call cost benchmark. No formal runtime or executor efficiency win is claimed.

These are the exercised ordinary interfaces. Both literal final checkpoints were actually forked to a disposable horizon200 identity on CPU and compared exactly on model state, named Adam state, provider clocks, history and RNG before those duplicate probe files were cleaned. The resume interface completed the actual40→100 recovery. The optional command below starts work only when the user runs it; choose a fresh output directory and an explicit horizon. R0 is the recommended research-control continuation state, not a certified inverse design model.

```bash
cd /home/wanglz/Desktop/src/ModularDT
conda activate ModularDT
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROUND_DIR=/data2/wanglz/ModularDT/diagnostics/thermal_compact_round12_20261023
NEXT_RUN="$ROUND_DIR/runs/R0_user_continue200"
export MPLCONFIGDIR="$ROUND_DIR/continuation_mpl"
python HONF_Proj/tools/honf_research.py fork-terminal   --run "$ROUND_DIR/runs/R0" --output "$NEXT_RUN" --local-horizon 200 --device cpu
set -o pipefail
python HONF_Proj/tools/honf_research.py resume   --run "$NEXT_RUN" --device cuda:2 --devices cuda:2 --stop-after-epoch 200   2>&1 | tee "$NEXT_RUN/user_continuation.log"
```

For an ordinary interruption of that explicitly created continuation, rerun the resume command with the same run directory, requested stop and permitted device. A separate R1 continuation can use the same fork-terminal interface with runs/R1 and a distinct output, preserving the negative result and frozen cut rather than retuning it. No such continuation was trained here.

```bash
# Reproduce a final saved-weight field board into a fresh ignored DATA2 directory.
python HONF_Proj/tools/compact_retrieval_evaluate.py native-board   --run-dir "$ROUND_DIR/runs/R0" --checkpoint epoch_0100_model.pt   --expected-epoch 100 --split validation --device cuda:2   --output "$ROUND_DIR/boards/R0_user_eval100"
```

The final native-board, responses, rankings, geometry-probes and actual-cache-k commands were executed on the declared source/final checkpoints. Milestone20/40/100, R1 before/after40, latest/best aliases, complete histories, curves and compact provenance remain. The terminal results use the explicit epoch100 checkpoint, not a selected best alias. Generated figures, arrays, checkpoints and one-time report helpers remain local ignored artifacts; durable source, tests, configs and this report are the upload boundary.

## Figure index and retained numerical evidence

Selected PDF masters: [1](../../diagnostics/thermal_compact_round12_20261023/report_figures/thermal_native_field_mean_p90.pdf); [2](../../diagnostics/thermal_compact_round12_20261023/report_figures/thermal_fluid_maps_case_0692.pdf); [3](../../diagnostics/thermal_compact_round12_20261023/report_figures/03_compact_organization.pdf); [4](../../diagnostics/thermal_compact_round12_20261023/report_figures/04_query_access_and_work_v2.pdf); [5](../../diagnostics/thermal_compact_round12_20261023/report_figures/05_cut_and_live_geometry_v2.pdf); [6](../../diagnostics/thermal_compact_round12_20261023/report_figures/06_saved_responses_and_ranking_v2.pdf). Their six small PNG companions are embedded above; all six pages were visually inspected and superseded visual exports were removed after validation.

Numerical entry points: [matched history/fields/organization](../../diagnostics/thermal_compact_round12_20261023/assessment/final_comparison.json), [R0 VAL89](../../diagnostics/thermal_compact_round12_20261023/boards/R0_local100_VAL89/result.json), [R1 VAL89](../../diagnostics/thermal_compact_round12_20261023/boards/R1_local100_VAL89/result.json), [R0 TRAIN24](../../diagnostics/thermal_compact_round12_20261023/boards/R0_local100_TRAIN24/result.json), [R1 TRAIN24 and intervention](../../diagnostics/thermal_compact_round12_20261023/boards/R1_local100_TRAIN24/result.json), [boundary VAL4](../../diagnostics/thermal_compact_round12_20261023/boundary/R1_cut_comparison_VAL4_components_v2.json), [boundary TRAIN2](../../diagnostics/thermal_compact_round12_20261023/boundary/R1_cut_comparison_TRAIN2_components_v2.json), [live geometry](../../diagnostics/thermal_compact_round12_20261023/geometry/R1_local100.json), [responses/decisions/cache synthesis](../../diagnostics/thermal_compact_round12_20261023/assessment/response_decision_cache_summary.json), and [terminal handoff qualification](../../diagnostics/thermal_compact_round12_20261023/qualification/actual_terminal_continuation_state.json). Each response_replay, rankings and heat_reuse/*_actual_cache directory retains its result/provenance and selected source-aligned arrays.

## Storage and repository closeout

Before output-producing work, /data2 was verified as the intended ext4 mount and the required Trained_Results, diagnostics and shared Wind catalogue symlinks resolved to their declared relocated-workspace targets. The round hard budget was 10 GiB new disk bytes, peak temporary budget 2 GiB and minimum /data2 headroom 200 GiB; no persistent new dataset-cache budget was allocated. Existing packed data and geometry/response caches were reused without copying datasets or changing precision, physical measures or query draws.

Closeout removed 2,342 allowlisted disposable files (202,349,001 logical bytes), plus the earlier verified duplicate result (718,110 bytes), for 203,067,111 bytes removed in this round. Removed payloads were completed test/cache data, two terminal-fork probe copies, one superseded failed cache partial, seven identified source bytecodes and six superseded visual exports. The round retains 566,560,559 bytes of scientific checkpoints, numerical evidence, selected six PDF/PNG figure pairs, compact manifests/logs and report helpers at the receipt snapshot. Actual free space after cleanup is 3,222,282,825,728 bytes on /data2 and 97,392,148,480 bytes on the home filesystem; the minimum /data2 headroom remains satisfied. Checkpoint digests, protected file metadata, source U-H e1000 bytes and all required symlink bindings remained unchanged across removal; unrelated running PhyCoFlow work used a separate tree.

The exact cleanup inventory and receipt remain in [cleanup](../../diagnostics/thermal_compact_round12_20261023/cleanup/round12_cleanup_receipt.json). Checkpoints, numerical reference evidence, histories, report helpers and compact failure logs are retained; migration backups and unrelated active-job roots are protected. Source U-H e1000 and workspace bindings are rechecked at closeout. The full outgoing commit range is audited through the repository artifact gate; [delivery verification](../../diagnostics/thermal_compact_round12_20261023/assessment/delivery_verification.json) records the final normal branch push and matching local/remote tips. Pre-existing user report/plan changes remain outside these commits.
