# HONF Interaction-Preserving Recovery: Recovery Failed at Strong-Reference Qualification

The campaign restored nonlinear source-conditioned MM/ME/EM message paths before an optional shallow collective residual, kept Thermal flow and temperature in one jointly trained model, and left Wind’s shared nonlinear core unchanged. The revised segmented T4111/P, T4112/P-G, and T4113/P-H arms were selected at e2500, and the user-authorized fresh full-TRAIN T4203/P-H and W2501/P follow-up reached normally stopped e1000 checkpoints. The new full-data comparisons show useful partial direction but do not clear the strong Thermal reference floor or establish Wind organizer benefit; the terminal decision is Recovery failed at strong-reference qualification.

## Current answers

**Reliability:** The terminal receipt says “Recovery failed” at strong-reference qualification. Segmented T411x P/G/H improve several exposed DEV22 fields, but their v means remain 1.94×/1.90×/1.70× the contextual D-sep mean with 0/22 lower-v cases, and fresh full-TRAIN T4203/P-H still has 7.69× Run1804’s e5000 v mean. Useful partial gains remain, but the declared recovery floor was not met; the strict B48-versus-six-B8 check still fails for derived omega/interface outputs.

**Organizer:** Segmented Thermal P-H is a provisional useful dense collective direction: it passes all 12 declared priority guards and improves most fields versus P-G, while P-G misses the pressure guard. The full-TRAIN follow-up has no matched P/H comparison, so it does not test full-data organizer benefit. Wind P-H remains mixed, with near-wake deterioration outweighing a small downstream gain; neither dataset earns organizer promotion.

**Cost:** On the same two predeclared Thermal detail scenes, complete native calls cost 0.080703/0.108318/0.133347 s for segmented P/P-G/P-H (1.00×/1.34×/1.65×); H misses the 1.5× target. Wind calls cost 1.00×/1.13594×/1.35449× across the same two detail rows. No sparse-executor savings were measured, and the strict B48 output mismatch remains. The closed campaign charged 11.91367 GPU-hours including its 1-second conservative allowance; no active sessions remain.

**Full-data follow-up status:** T4203/P-H and W2501/P both completed their fresh full-TRAIN e1000 horizons normally and stopped; field-only selection froze e1000 before physical comparison. Saved full-physics comparisons now cover the exposed Thermal canonical89 full-native grid and Wind fullVALID90 Q8192 rows across 30 layouts; these are validation evidence, not independent TEST results, and WindTEST remained locked. The current terminal decision and complete selection, comparison, infrastructure, and protection receipts are in the [appendix](20261010_214021Z_HONF_Interaction_Preserving_Recovery_20261010_Appendix.md#final-full-train-e1000-results-and-terminal-decision).

**Selected figure index:** [1. DEV trajectories and saved full-data references](#family-1-dev-trajectories-and-full-data-references) · [2. Native fields and residuals](#family-2-native-fields-and-residuals) · [3. Executed graph chains](#family-3-executed-graph-chains) · [4. Same-weight interventions](#family-4-same-weight-interventions) · [5. Fidelity and measured cost](#family-5-fidelity-and-measured-cost) · [6. Saved response replay](#family-6-saved-response-replay). [PDF masters and companion panels](20261010_214021Z_HONF_Interaction_Preserving_Recovery_20261010_Appendix.md#selected-age-figure-package-and-review-receipts).

## Predictor

**Selected revised Thermal P/G/H:** the field-only selectors chose e2500 from 25 scheduled monitors for all three arms before full-native physical metrics were read. The table reports equal-case mean / p90 of the 22 per-case RMSEs over the complete native fluid grid, totaling 170,825 valid points per arm; every valid prediction and reference is finite. These are exposed fixed25_v1 DEV22 results in benchmark-native channel units, with temperature units not asserted SI. The evaluations read the exposed DEV22 target subset from source H5 split `test`; unexposed original TEST targets were not read.

| Channel | T4111/P mean / p90 | T4112/P-G mean / p90 | T4113/P-H mean / p90 |
|---|---:|---:|---:|
| `u` | 0.0229956 / 0.0275702 | 0.0212918 / 0.0250455 | 0.0175695 / 0.0208356 |
| `v` | 0.00331994 / 0.00420435 | 0.00326183 / 0.00407420 | 0.00291914 / 0.00382848 |
| `p` | 0.00984368 / 0.0122014 | 0.0109920 / 0.0127113 | 0.0102173 / 0.0122321 |
| `omega` | 0.0658715 / 0.0868889 | 0.0594748 / 0.0752194 | 0.0513713 / 0.0655569 |
| Fluid temperature | 0.608297 / 0.848060 | 0.518543 / 0.777915 | 0.529143 / 0.771922 |

### Family 1: DEV trajectories and full-data references

![Thermal fixed-DEV22 Q1024 validation histories for historical T410x and revised T4111/P, T4112/P-G, and T4113/P-H, with selected-age checkpoints marked](../../../../diagnostics/generated/interaction_recovery_20261010/final_native_curl_selected_summary_v11_20261011T0234Z/family1_thermal_dev.png)

Caption: The Q1024 monitor curves use the exposed fixed25_v1 DEV22 panel; the revised selectors chose e2500 from 25 scheduled monitors. The separate full-native e2500 v means are 0.003320/0.003262/0.002919 for P/P-G/P-H, still 1.94×/1.90×/1.70× the contextual D-sep value; units are benchmark-native and unexposed original TEST targets were not read.

![Paired saved-reference error reductions for selected fresh full-TRAIN T4203/P-H and W2501/P](../../../../diagnostics/generated/interaction_recovery_20261010/optional_full_p_comparison_candidate/final_full1000_figures_v4_20261011T0520Z/family1_full_p_saved_reference_comparison.png)

Caption: The plot shows mean paired case-wise Thermal RMSE reduction over canonical89 and mean paired row-wise Wind vector-RMSE reduction over fullVALID90; positive favors the new checkpoint. T4203/P-H remains 7.69× Run1804 e5000 on v mean, while W2501/P is 2.13×/2.74× Wind2103 Dense e2475 on near/downstream means; these exposed panels use different training recipes and ages, so the plot is saved-reference context rather than a causal or matched-training effect.

### Family 2: Native fields and residuals

![Selected e2500 Thermal native predictions and full-grid residuals for DEV case 0684 across independent P, P-G, and P-H](../../../../diagnostics/generated/interaction_recovery_20261010/final_native_curl_selected_fields_graphs_v4_20261011T0234Z/family2_thermal_native_highM_0684.png)

Caption: DEV0684 contributes 7,474 valid fluid nodes on the native 64×128 grid; the panels show u, v, p, omega, and jointly predicted temperature for selected e2500 T411x arms with shared channel scales and unclipped full-grid relative-L2 residuals. Temperature uses the dataset-native scale, not asserted SI, and this is one fixed scene rather than the DEV22 summary.

![Selected e2500 Thermal native predictions and full-grid residuals for DEV case 0277 across independent P, P-G, and P-H](../../../../diagnostics/generated/interaction_recovery_20261010/final_native_curl_selected_fields_graphs_v4_20261011T0234Z/family2_thermal_native_lowM_0277.png)

Caption: DEV0277 contributes 7,976 valid fluid nodes on the same native 64×128 grid; the selected e2500 maps show all five channels and full-grid residuals on shared arm scales. The two fixed maps total 15,450 valid nodes, while the reported equal-case board covers all 22 exposed DEV cases.

![Selected full-TRAIN T4203/P-H predictions and residuals at fixed canonical89 case 0291](../../../../diagnostics/generated/interaction_recovery_20261010/optional_full_p_comparison_candidate/final_full1000_figures_v4_20261011T0520Z/family1_full_p_fixed_thermal_fields_T0291.png)

Caption: The selected e1000 field map compares T4203/P-H with the saved physical reference at fixed case 0291; masked relative-L2 errors are 0.0185, 0.098, 0.123, 0.0618, and 0.0659 for u, v, p, omega, and fluid temperature. The three-page PDF master covers cases 0291/0653/0687; Thermal colors use benchmark-native scales without an SI temperature assertion, and these exposed canonical89 cases are not independent TEST evidence.

The [appendix scorecard](20261010_214021Z_HONF_Interaction_Preserving_Recovery_20261010_Appendix.md#revised-selected-t411x-e2500-full-native-evidence-scorecards-probes-and-costs) adds worst-case CPID/module count, fixed-Q1024 temperature and q-proxy tails, all five Wind roles and component tails, module-count/direction strata, and the saved fixed TRAIN-probe versus DEV comparison.


**Measured Wind context:** on the shared fixed24 DEV direction rows, fresh source-resolved W2401/P has lower mean vector RMSE than the historical source-resolved all-fine Full reference across all five physical roles. Both report m/s and the same 24 exposed rows, but the reference uses H64/Q1024 while W2401 uses H128/Q4096; the rolewise differences are positive recovery context, not matched-query gains or causal effects.

| Physical role | W2401/P H128, Q4096 | Historical Full H64, Q1024 |
|---|---:|---:|
| Background | 0.03796 m/s | 0.05183 m/s |
| Volume | 0.08696 m/s | 0.11744 m/s |
| Hub slab | 0.17111 m/s | 0.23983 m/s |
| Near turbine | 0.28007 m/s | 0.44985 m/s |
| Downstream envelope | 0.20232 m/s | 0.25787 m/s |

### Wind DEV24 trajectories

![Wind DEV24 selected-checkpoint histories for unchanged W2401/P, W2402/P-G, and W2403/P-H, with row-mean and row-p95 curves](../../../../diagnostics/generated/interaction_recovery_20261010/final_native_curl_selected_summary_v11_20261011T0234Z/family1_wind_dev.png)

Caption: The unchanged W240x field-only selectors chose e2500 on the fixed DEV24 panel spanning eight layouts; solid curves show row-mean RMSE and dotted curves show p95 across rows, not spatial points or layouts. At e2500, W2401/P near/downstream vector means are 0.28007/0.20232 m/s; P-H improves downstream by 3.0% but worsens near-turbine error by 5.0%, and WindTEST remains locked.

### Wind native fields and residuals

![Selected e2500 Wind native velocity fields and residuals for DEV row 426 across W2401/P, W2402/P-G, and W2403/P-H](../../../../diagnostics/generated/interaction_recovery_20261010/final_native_curl_selected_fields_graphs_v4_20261011T0234Z/family2_wind_native_highM_426.png)

Caption: DEV row 426 spans 6,384,000 native 3D cells on a 350×285×64 grid; the selected e2500 W240x velocity maps and residuals use common component scales in m/s. This fixed-row detail illustrates spatial error structure and does not replace the 24-row, eight-layout role scorecard.

![Selected e2500 Wind native velocity fields and residuals for DEV row 69 across W2401/P, W2402/P-G, and W2403/P-H](../../../../diagnostics/generated/interaction_recovery_20261010/final_native_curl_selected_fields_graphs_v4_20261011T0234Z/family2_wind_native_lowM_69.png)

Caption: DEV row 69 spans 3,363,072 native 3D cells on a 302×174×64 grid; the selected e2500 W240x velocity maps and residuals use the same component scales in m/s as row 426. This single layout is descriptive detail, not an independent or TEST estimate.

All 15 Wind component means and all 15 component row-p95 values are also lower for W2401/P than the lower-query historical Full reference; the row identities, manifest, normalization, reducers, and comparison limits are hash-bound in the [saved source-resolved context receipt](../../../../diagnostics/generated/interaction_recovery_20261010/wind_P_source_resolved_reference_context_20261011T011838Z.json), SHA256 `cc575ad89a726e4fa75d4aad7e48084f141f6b4d831bcdec2879945aa5efeb55`. The reference and P share 72 TRAIN/24 DEV direction-row membership and the TRAIN-fitted normalization, while width, query count, loss objective, and schedule differ; this is evidence of useful recovery direction on exposed DEV, not a recipe-matched effect. W2302 J-geometry remains the matched Q4096 segmented comparator: P lowers near/downstream vector means from 0.32086/0.21662 to 0.28007/0.20232 m/s, while full-data Run2202 is unequal-exposure context and is not used as a matched recovery threshold.

**Predictor gains and tradeoffs:** versus original selected T4101/P e2400 on the same exposed DEV22 full-native reducer, revised T4111/P e2500 lowers mean u/v/p/omega by 20.3%/25.8%/20.2%/63.7%, while temperature mean rises 3.25%. Relative to independent P, P-G lowers u/v/omega/temperature means by 7.4%/1.8%/9.7%/14.8% but raises pressure by 11.7%; per-case wins are 15/22, 16/22, 2/22, 20/22, and 17/22 for u/v/p/omega/temperature. P-H lowers u/v/omega/temperature means by 23.60%/12.07%/22.01%/13.01%, while pressure rises 3.80%; its per-case wins are 22/22, 21/22, 7/22, 22/22, and 17/22. These are same-cohort, same-mask comparisons of the trained arms, not causal decompositions of the training effects.

**Miss:** P/G/H v means remain 1.94×/1.90×/1.70× the stored D-sep full-native mean and each arm has 0/22 cases below D-sep; H’s u mean is 1.03× D-sep with 10/22 lower-u cases. P’s selected-profile TRAIN gradient probe with input-only case selection retains a flow-v/flow-omega shared-core cosine of −0.521, so the opposing direction is not shown resolved. D-sep shares fixed25_v1 TRAIN150/DEV22 membership and the nominal 2,500-epoch lineage, but has a distinct flow-only objective, message width, and branched schedule; historical actual visits and updates were not verified, so it remains a contextual physical floor rather than a matched joint-model baseline.

**Predictor gains and misses in the full-TRAIN follow-up:** T4203/P-H improves on some saved Thermal references but misses the strong floor: its e1000 means are 0.013782/0.002551/0.009826/0.038194/0.447273 for u/v/p/omega/fluid T, while Run1804 e5000 is 0.006504/0.000332/0.002249/0.024434/0.237375; against Run3906 endpoint e5000 the new model is better on u and omega but worse on v, p, and fluid T. W2501/P has useful exposed fullVALID90 measurements but remains 2.06×/2.45×/2.52×/2.13×/2.74× the selected Wind2103 Dense e2475 vector means in background/volume/hub/near/downstream roles. Training age, objective, and environment inventory differ, so these saved comparisons do not isolate architecture effects.

**Next evidence:** the selected e2500 segmented boards and the full-TRAIN e1000 comparison are complete; the campaign is closed with no active sessions. The current runner stops at e1000, and the terminal receipt makes no 5,000-epoch recommendation or organizer promotion; the user decides whether any future extension of the saved full-TRAIN checkpoints is warranted.

### Conditional full-TRAIN e1000 selected physical results

T4203/P-H was selected at e1000 from ten saved monitors before full-physics evaluation; its canonical89 exposed validation comparison evaluates the full native grid over 89 cases (Q1024 is the training and monitoring query budget), with the same physical reducers as the pinned saved Thermal references, with query/training protocols and ages still unequal. Wind W2501/P was selected at e1000 before its fullVALID90 Q8192 comparison; the 90 rows cover 30 correlated layouts and vector row-p95 is across row RMSEs, not layout-level RMSE. Thermal values are benchmark-native channel scales (temperature is not asserted SI); Wind values are m/s.

| Thermal channel | Mean RMSE | Case p90 | Worst RMSE (CPID/M) |
|---|---:|---:|---|
| `u` | 0.013782 | 0.016548 | 0.018662 (0678/M10) |
| `v` | 0.002551 | 0.003557 | 0.004027 (0682/M10) |
| `p` | 0.009826 | 0.012241 | 0.013330 (0690/M10) |
| `omega` | 0.038194 | 0.046700 | 0.054585 (0688/M10) |
| Fluid temperature | 0.447273 | 0.644738 | 0.916027 (0300/M7) |
| Surface temperature | 0.479652 | 0.745519 | 1.076759 (0300/M7) |
| Material temperature | 0.438694 | 0.694123 | 1.093940 (0300/M7) |
| Sampled module-material peak | 0.419778 | 0.726288 | 1.263292 (0300/M7) |
| q-normal proxy | 1.725776 | 2.158733 | 3.000028 (0297/M7) |

| Wind role | Mean row vector RMSE (m/s) | Row-p95 RMSE (m/s) | Worst row RMSE (row ID) |
|---|---:|---:|---|
| Background | 0.022614 | 0.028226 | 0.034398 (row488) |
| Volume | 0.046730 | 0.062840 | 0.067124 (row173) |
| Hub slab | 0.083033 | 0.110569 | 0.128062 (row427) |
| Near turbine | 0.158236 | 0.187344 | 0.207730 (row488) |
| Downstream envelope | 0.107060 | 0.131007 | 0.140250 (row426) |

T4203/P-H is 7.69× Run1804 e5000 on fluid-v mean and 1.88× on fluid-temperature mean; relative to Run3906 J-H endpoint e5000 it lowers u and omega means but raises v, p, and fluid-temperature means. W2501/P is 2.06×/2.45×/2.52×/2.13×/2.74× the Wind2103 Dense selected-e2475 vector means over background/volume/hub/near/downstream; all five role comparisons favor Dense and the rowwise comparison has zero W2501 wins on 90 rows. These saved comparisons use different training recipes and ages, so they document the strong-reference miss and partial gains, not a causal or independent-TEST result. Full per-reference tails, module-count/direction strata, components, recipes, and receipt hashes are in the [appendix](20261010_214021Z_HONF_Interaction_Preserving_Recovery_20261010_Appendix.md#final-full-train-e1000-results-and-terminal-decision).

## Organizer

**Measured Wind tradeoff:** on the fixed DEV24 Q4096 role panel, P-H lowers downstream vector mean RMSE by 3.0% versus independent P but raises near-turbine vector mean by 5.0%; its near-turbine error is also higher than P on both saved full-native detail rows. P-H passes the declared twelve component mean/row-p95 eligibility guards, but that guard is not a promotion rule. W2402/P-G misses three named component guards, and its fixed-geometry substitution is an exact output no-op at the displayed receiver.

The selected T411x boards now support a provisional Thermal organizer assessment: P-H passes all 12 declared priority guards and shows broad flow gains versus P-G, while P-G misses the pressure guard and H costs 1.65× independent P. This is useful dense-model evidence, not proof of sparse-work savings or physical causality. Wind P-H remains mixed, with its near-wake worsening and higher cost despite a modest downstream gain, so shared cross-dataset organizer benefit remains unproved.

### Family 3: Executed graph chains

![Selected e2500 Thermal graph chain and same-query flow receiver at DEV case 0277](../../../../diagnostics/generated/interaction_recovery_20261010/final_native_curl_selected_fields_graphs_v4_20261011T0234Z/family3_thermal_graphchain_lowM_0277.png)

Caption: The exact DEV0277 q0 receiver join shows 26 access slots, 18 nonzero edges, three highlighted edges, and all three physical-source reads for P-G/P-H; direct P has no collective graph. P-G geometry substitution is an exact no-op, P-H geometry and donor reassignment change the receiver output, and restored defaults are bit-exact, describing functional dependence rather than physical causality.

![Selected e2500 Wind graph chain and same-query volume-role receiver at DEV row 69](../../../../diagnostics/generated/interaction_recovery_20261010/final_native_curl_selected_fields_graphs_v4_20261011T0234Z/family3_wind_graphchain_lowM_69.png)

Caption: The exact DEV row 69 volume-role q0 join shows 62 access slots, 38 nonzero edges, three highlighted edges, and all eight physical-source reads for P-G/P-H. P-G geometry substitution is an exact no-op; P-H geometry changes Ux/Uy/Uz by −0.01251/−0.00008/+0.00167 m/s, while restored defaults are bit-exact; the graph is an executed receiver trace, not a physical-causality claim.

### Family 4: Same-weight interventions

![Selected e2500 Thermal same-weight intervention comparison across DEV cases 0277 and 0684](../../../../diagnostics/generated/interaction_recovery_20261010/final_native_curl_selected_summary_v11_20261011T0234Z/family4_thermal_interventions.png)

Caption: This equal-case Q1024 comparison covers two fixed DEV identities in native dataset units and reports direct P plus saved same-weight residual, geometry/incidence/receiver, and matched-measure donor interventions for P-G/P-H. Restored-default predictions match each native baseline exactly; intervention responses measure dependence on those model components, not physical causality.

![Selected e2500 Wind same-weight intervention comparison for near-turbine and downstream roles](../../../../diagnostics/generated/interaction_recovery_20261010/final_native_curl_selected_summary_v11_20261011T0234Z/family4_wind_interventions.png)

Caption: The two-row Q4096 intervention panel reports near-turbine and downstream-envelope vector RMSE in m/s; selected P-H near-turbine vector mean is 0.341 m/s, distinct from its 0.056 m/s DEV24 Uy component mean because panel and reducer differ. Restored-default outputs match baseline exactly, and these same-weight changes do not establish physical causality.

**Miss and next evidence:** no organizer is promoted. Thermal P-H is a provisional useful dense collective direction, while P-G misses the pressure guard; Wind P-H remains mixed. The selected graph and same-weight intervention figures below show actual model responses, but they do not establish physical causality or sparse-executor savings.

## Measured inference work and cost

Wind complete native inference costs on the same two fixed full-native rows were 1.00×/1.13594×/1.35449× for W2401/P, W2402/P-G, and W2403/P-H. A same-weight direct extraction from W2403/P-H took 42.11760 s across those two rows, versus 42.89910 s for independent P and 58.10640 s for organized P-H; it retained all 218,424,576 source-read pairs and measured zero sparse-executor savings. This extraction is cost-only and is not an independently trained direct-P accuracy result.

**Thermal cost:** on the same two predeclared full-native DEV detail scenes, synchronized end-to-end native inference took 0.080703 s for P, 0.108318 s for P-G, and 0.133347 s for P-H (1.00×/1.34×/1.65×); H exceeds the 1.5× target but remains below the 2× red flag. The same-weight H direct extraction measured 0.078328 s with exact parity on two native checks and one GPU check, while retaining all source-read pairs; it is a cost-only extraction and measured zero sparse-executor savings. These are two-scene timings, not 22-case latency or a direct-P accuracy result. No sparse execution savings were measured, and logical support concentration or graph-disabled timing does not establish eliminated work. The full-access fallback, complete-call latency, peak memory, and charged training lifetime remain separate measures.

### Family 5: Fidelity and measured cost

![Thermal full-native fidelity versus complete-call cost for selected e2500 P, P-G, and P-H](../../../../diagnostics/generated/interaction_recovery_20261010/final_native_curl_selected_summary_v11_20261011T0234Z/family5_thermal_fidelity_vs_native_cost.png)

Caption: On the same two full-native DEV scenes, summed per-scene medians are 0.080703/0.108318/0.133347 s for P/P-G/P-H (1.00×/1.34×/1.65×); the H arm exceeds the 1.5× target. Points show equal-scene RMSE means and the two-scene range across u, v, p, omega, and temperature, with native dataset scales and no sparse-work inference.

![Wind full-native fidelity versus complete-call cost for selected e2500 P, P-G, and P-H](../../../../diagnostics/generated/interaction_recovery_20261010/final_native_curl_selected_summary_v11_20261011T0234Z/family5_wind_fidelity_vs_native_cost.png)

Caption: On the same two full-native DEV rows (9,747,072 cells total), W2401/P, W2402/P-G, and W2403/P-H take 42.899/48.731/58.106 s summed across per-row timing medians; fidelity remains separated by near-turbine/downstream Ux/Uy/Uz in m/s. Wind and Thermal workloads have different grids and output scopes, so their seconds are not compared across tasks.

## Inverse

No inverse training, inverse design campaign, or new solver run was performed. Historical solver usage remains 326/326, WindTEST remains locked, and the inherited protection bindings and user-owned plans remain intact.

**Gain:** no new inverse result is claimed. **Miss:** no inverse design quality, sample validity, or solver efficiency was tested. **Evidence:** the existing allowance is exhausted and no new solve occurred. **Next step:** none is part of this recovery comparison; any inverse campaign requires its own authorized scope and evidence.

### Family 6: Saved response replay

The plot below replays selected e2500 predictions against already-saved solver-frame references on three predeclared DEVELOPMENT pools; it is not inverse training, a new solve, or independent ground truth.

![Selected e2500 Thermal transfer-minus and transfer-plus response replay against saved frames for pools 0291, 0294, and 0687](../../../../diagnostics/generated/interaction_recovery_20261010/final_native_curl_selected_summary_v11_20261011T0234Z/family6_native_response_replay.png)

Caption: The three-pool replay uses dataset-native material-temperature units and shows finite transfer-minus/plus responses for all selected arms. For T4113/P-H at pool 0291, transfer-plus is −0.0795 versus the saved reference delta −0.3903, a visible response mismatch; no solver call, inverse search, new label, or TEST read occurred.

## Decision and final figure set

The terminal decision is **Recovery failed** at the declared strong-reference qualification, with useful partial directional gains retained. Segmented T411x P/G/H reached selected e2500 and passed the focused native-curl qualification, but the v floor and strict B48-versus-six-B8 derived-omega/interface mismatch remain. Full-TRAIN T4203/P-H and W2501/P both completed and were field-only selected at e1000; T4203/P-H remains materially above the stronger saved Thermal v/temperature references, and W2501/P remains behind the saved Dense2103 fullVALID90 vector means. The bounded follow-up does not prove a full-data organizer effect because there is no matched full-data Thermal P control; Wind organizer benefit is also unproved. No organizer is promoted, no full-5,000 development/formal recommendation or 5,000 command was prepared, and the current runner stop cap is 1,000; the user decides whether a future extension of the saved e1000 checkpoints is warranted. The root decision receipt is [final_scientific_recovery_organizer_decision_v1_20261011T0520Z.json](../../../../diagnostics/generated/interaction_recovery_20261010/final_scientific_recovery_organizer_decision_v1_20261011T0520Z.json), SHA256 `f3ba1e69cb7ab24ca8c91bc1d8f2f57deb4fec606208c73862d628eee921eae0`; full selection, paired comparison, visual review, resource closure, and protection receipts are indexed in the [appendix](20261010_214021Z_HONF_Interaction_Preserving_Recovery_20261010_Appendix.md#final-full-train-e1000-results-and-terminal-decision).

The selected e2500 checkpoints and summaries are bound by the [all-T field-only selector](../../../../diagnostics/generated/interaction_recovery_20261010/field_only_selection_native_curl_all_T_e2500_20261011T0206Z.json), SHA256 `537f7eeac6401a02a6a6cf358166daed7c51498bbd5f10a82a12ad49758030d2`, and the [revised-T/original-W combined measurement binding](../../../../diagnostics/generated/interaction_recovery_20261010/measurement_binding_revised_T_original_W_e2500_20261011T0211Z.json), SHA256 `14a05871734ea8ebff6591d902f80180e45faabe4c37327d74a70e25dd2f6087`. T4111/P [checkpoint](../../../../diagnostics/generated/interaction_recovery_20261010/runs/T4111/checkpoints/epoch_2500_model.pt), SHA256 `e52fc353a44f8f4d4290f92b1d93b56111f4a394885205ae8cc1189cfe02ae79`, and [full-native summary](../../../../diagnostics/generated/interaction_recovery_20261010/thermal_native_curl_fullnative_T4111_selected2500_attempt01/summary.json), SHA256 `74790e7cccc83747ee0690d67b6e3d10b130b9e2d89ea38120ba950dc8f871aa`; T4112/P-G [checkpoint](../../../../diagnostics/generated/interaction_recovery_20261010/runs/T4112/checkpoints/epoch_2500_model.pt), SHA256 `4185dbbbd6b3ab489c4998103c8c956aa91aae873e531042fe22db9b50f14488`, and [full-native summary](../../../../diagnostics/generated/interaction_recovery_20261010/thermal_native_curl_fullnative_T4112_selected2500_attempt01/summary.json), SHA256 `4c916fdbe960ff3d77ee894c514b48ca64e1f7b7c8db99b88c2f1dcd155aa520`; T4113/P-H [checkpoint](../../../../diagnostics/generated/interaction_recovery_20261010/runs/T4113/checkpoints/epoch_2500_model.pt), SHA256 `3966c509c376c9f116a2d6bf8f2579c8287d04c4decbc52d6f2d944ea31c3984`, and [full-native summary](../../../../diagnostics/generated/interaction_recovery_20261010/thermal_native_curl_fullnative_T4113_selected2500_attempt01/summary.json), SHA256 `b5daea318c16afdf22d6b7d48f54c5af0e638c678265c1107066bdc03c88838f`.

The report embeds 15 principal PNG figures across six families: 13 selected segmented evidence panels and two full-TRAIN follow-up panels. Root and independent visual review passed the 13 segmented panels; root and independent full-data numeric/visual review passed the two V4 panels, with the independent receipt noting that all nine displayed Thermal metrics are paired and three secondary keys are not. Appendix links provide PDF masters, companion panels, and hash-bound review receipts.
