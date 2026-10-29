# HONF Round13-5 Supplement: Consistent Integration, Measured Latency, and Organization-Conditioned Reading

**Report status:** Both authorized epoch-100 fits, complete endpoint profiles, A/B canonical VAL89 evaluations, and the trained-B100 matched-marginal control are complete. **Prepared:** 27 October 2026. **Scope:** the separate A/B supplement campaign on the current `agent/honf-core-next` branch.

This supplement keeps the working four-block M/E interaction model and changes how its learned organization governs member access. Candidate A carries a single joint contribution mass through prediction; Candidate B keeps each view's membership active while member likelihoods compete. The implementation also removes repeated shared planning where measured and factors equivalent scorer work. These are new, independent TRAIN600 lineages, separate from the original R13-5 campaign and its continuation.

**Current endpoint state:** A and B each completed the authorized 100-epoch formal fit with clean exits, and their complete VAL89 and native cost evaluations are included below. No extension beyond 100 epochs per candidate is authorized. VAL89 is exposed validation and is not pristine TEST evidence.

## Predictor

Candidate A uses one joint95 prefix over original M and E members, then forms `b_i^A = P_i e_i / Σ_{j∈C} P_j e_j`, where `P_i` is the joint retrieval prior and `e_i` is the bounded query-conditioned member likelihood. The existing bounded role strength `ρ` is applied after this shared normalization, so each role retains its share of the joint mass instead of being independently renormalized to full strength. The predictor still reduces original member values and sends the resulting M/E contexts to the general joint physical-field head.

For either read law, members outside the selected prefix receive zero contribution: `b_i^A = 0` for `i ∉ C`, and `b_i^B = 0` for `i ∉ C`.

The selected read calibration uses normalized views, `R_ai = H_ai μ_i / r_a`, `w_a = π_a`, and `P_i = Σ_a w_a R_ai`, together with the inherited prior-weighted mean/variance bounded score calibration. The two interpretations of view strength gave mixed fixed-panel results: retaining raw row strength won two of four cases and changed the mean paired native objective only from 0.10444 to 0.10417. The candidate-independent `4 tanh(s/4)` score alternative removed survivor-score shifts to float32 tolerance but worsened the paired native objective in all four fixed TRAIN cases, from 0.10444 to 0.69594. The chosen normalized/inherited pairing therefore follows the measured panel and the plan's default under mixed effects; trained endpoints now provide a separate preliminary check. The inherited calibration still changes the average survivor score by 0.1106 for E when the candidate set changes, which remains a limitation to report.

The fixed-state stencil replay shows why joint contribution matters: the old reader normalized likelihood within E before applying role strength, while A retains each E member's share of the common M+E distribution. On the separate old development-epoch-600 state and exposed VAL0286, changing the read law lowered restricted q-proxy RMSE from 11.296 to 4.305 native units, while the restricted composite field score worsened from 0.05772 to 0.06820, surface-temperature RMSE from 2.353 to 2.697, and material-temperature RMSE from 1.948 to 2.502. The full-reader composite score also rose from 0.05445 to 0.07034. These are fixed-weight mechanism measurements, not trained Candidate A results or a uniform predictor gain.

Candidate A's completed epoch-100 evaluation covers all 89 exposed VAL cases with no fitting, optimizer update, solver attempt, TEST access, or inverse evaluation during the review. Candidate B was evaluated on the same exposed VAL89 panel with the same zero-fit/zero-update/zero-solver/no-TEST scope. The four nonempty module-count populations for each candidate are M3/24, M5/25, M7/25, and M10/15; the per-M and per-case results are retained in the linked canonical evaluation artifacts.

**Candidate A, epoch 100:**

| Observable | Full mean / p90 / worst | Full worst ID | Restricted mean / p90 / worst | Restricted worst ID |
| --- | ---: | --- | ---: | --- |
| u | 0.136684 / 0.158214 / 0.195353 | 0301 | 0.139886 / 0.162265 / 0.197355 | 0301 |
| v | 0.022206 / 0.025603 / 0.034664 | 0301 | 0.022745 / 0.026186 / 0.035521 | 0301 |
| p | 0.083248 / 0.096540 / 0.124053 | 0301 | 0.084748 / 0.098335 / 0.124301 | 0301 |
| Vorticity | 0.308194 / 0.367324 / 0.430517 | 0301 | 0.314486 / 0.370774 / 0.445781 | 0301 |
| Fluid temperature | 2.70973 / 3.36022 / 4.38773 | 0292 | 2.73509 / 3.32543 / 4.22333 | 0295 |
| Surface temperature | 2.13587 / 3.46125 / 5.35869 | 0300 | 2.13493 / 3.48706 / 5.27862 | 0300 |
| Material temperature | 2.07486 / 3.80013 / 5.66207 | 0298 | 2.10482 / 3.74683 / 5.91800 | 0298 |
| q-proxy | 5.23097 / 7.53769 / 10.6662 | 0298 | 5.27652 / 7.66522 / 10.4905 | 0298 |
| Module peak endpoint RMSE | 2.07770 / 4.17070 / 5.40111 | 0297 | 2.09658 / 4.07898 / 5.44166 | 0298 |

**Candidate B, epoch 100:**

| Observable | Full mean / p90 / worst | Full worst ID | Restricted mean / p90 / worst | Restricted worst ID |
| --- | ---: | --- | ---: | --- |
| u | 0.136063 / 0.166772 / 0.180456 | 0299 | 0.139628 / 0.169685 / 0.206450 | 0300 |
| v | 0.022207 / 0.026336 / 0.029893 | 0285 | 0.022717 / 0.026546 / 0.030005 | 0302 |
| p | 0.081616 / 0.093075 / 0.108443 | 0302 | 0.083436 / 0.093944 / 0.110468 | 0302 |
| Vorticity | 0.306337 / 0.357049 / 0.442157 | 0299 | 0.312133 / 0.371095 / 0.462244 | 0299 |
| Fluid temperature | 2.66792 / 3.29598 / 4.41091 | 0300 | 2.68891 / 3.29897 / 4.13484 | 0300 |
| Surface temperature | 2.19095 / 3.47719 / 5.02825 | 0297 | 2.17785 / 3.46837 / 4.77654 | 0297 |
| Material temperature | 2.14927 / 3.35693 / 6.07154 | 0297 | 2.15870 / 3.49028 / 5.80219 | 0297 |
| q-proxy | 5.37717 / 7.88137 / 11.3681 | 0295 | 5.40127 / 7.79417 / 11.2808 | 0295 |
| Module peak endpoint RMSE | 2.06083 / 3.21109 / 6.28337 | 0297 | 2.07171 / 3.32760 / 5.90730 | 0297 |

Against the immutable original R13-5 epoch-100 readers, both candidates' case-mean RMSE is higher on all nine observables in both full and restricted execution. A's increases are +13.21% to +32.36% full and +9.27% to +31.00% restricted; B's are +12.59% to +29.77% full and +7.97% to +28.97% restricted. Against the original Guide100 restricted reference, both candidates are lower on eight of nine means, with fluid-temperature error +19.63% for A and +17.62% for B. Within each candidate, restriction raises eight of nine means: A by 0.87–2.43% and B by 0.44–2.62%; surface-temperature mean falls by 0.04% for A and 0.60% for B. The gap to original ME100 is already present in both full readers, so it cannot be assigned to joint95 restriction alone. These first full-data endpoints are preliminary measurements, not convergence or architecture verdicts.

| Observable | A full vs ME100 full | A restricted vs ME100 restricted | B full vs ME100 full | B restricted vs ME100 restricted | A restricted vs Guide100 restricted | B restricted vs Guide100 restricted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| u | +21.46% | +21.69% | +20.91% | +21.47% | -3.32% | -3.50% |
| v | +22.48% | +22.22% | +22.48% | +22.07% | -13.92% | -14.03% |
| p | +32.36% | +31.00% | +29.77% | +28.97% | -2.33% | -3.84% |
| Vorticity | +17.20% | +16.11% | +16.49% | +15.24% | -2.17% | -2.90% |
| Fluid temperature | +26.27% | +23.08% | +24.32% | +21.00% | +19.63% | +17.62% |
| Surface temperature | +13.21% | +10.51% | +16.13% | +12.73% | -7.61% | -5.75% |
| Material temperature | +16.86% | +17.18% | +21.05% | +20.18% | -3.85% | -1.39% |
| q-proxy | +21.01% | +17.15% | +24.39% | +19.92% | -5.70% | -3.47% |
| Module peak endpoint RMSE | +13.52% | +9.27% | +12.59% | +7.97% | -5.23% | -6.35% |

The module-count strata contain 24 M3 scenes/72 modules, 25 M5/125, 25 M7/175, and 15 M10/150. B's restricted mean is lower than A's on six observables and higher on surface temperature, material temperature, and q-proxy; those B-versus-A changes range from −1.69% to +2.56%. Local module-peak errors remain distinct from the 89-scene endpoint RMSE: across 522 modules, A full/restricted absolute endpoint error averages 1.834/1.846 native temperature units (p95 5.067/5.155; worst 12.579/13.058), while B averages 1.877/1.890 (p95 4.985/4.938; worst 11.733/12.617). M3 alone has smaller local means (0.94–1.05) but B's M3 local worst is about 4.0; it does not erase the larger M7 local extremes. The spatial and local panels therefore retain a separate fidelity caveat even where aggregate errors improve.

Both six-epoch development qualifications completed epochs 1–6, with a clean stop at epoch 3 and an exact resume through epoch 6, for 12 segmented epochs total. The formal A/B comparison starts independently from scratch with all 194 initial tensors identical, the same input-only TRAIN600 order, seed 0, common TRAIN-only normalization, and the declared VAL89 panel; no qualification weights transfer into either formal lineage. Each formal fit completed epochs 1–100 and stopped cleanly at the authorized endpoint. The common physical objective and two ordered MM→EM→EE→ME exchange cycles remain in both candidates.

The fixed Q=1024 native health probe found connected full-reader and active selected-member paths, but gradient connectivity was not uniformly positive for every restricted role: E-value gradients were zero at A/M2 and B/M2 and B/M5 because those selected E roles were empty. B's M12 restricted E proposal-score gradient was only about `9.7e-11`. Empty selected roles are a valid outcome of the reader; neither positive connectivity nor these small or zero gradients alone establish strong learning or causal effect.

A bounded repeatability check on the completed B100 checkpoint at TRAIN0006/M1 used Q1024 and 2,370 native union rows without fitting. A +10% heat perturbation reproduced the restricted-omega response (`0.057505` normalized-RMS change versus `0.006773` full), matching the stored `h` within `1e-7` with identical candidate masks and counts. The stored restricted-omega `r=0.005506` was not reproduced: grad-enabled first-8-then-512 tiling versus ordinary no-grad512 differed by `5e-7–1.1e-6`, independent ordinary no-grad512 repeats by `0.9e-6–2.4e-6`, and the matched-baseline repeat delta was `8.2e-7`. The layout/mode variation is real but too small in this replay to explain the stored `r`; that local anomaly remains unexplained. The heat response supports this connected local path only, not a general sensitivity or causal claim.

| Predictor evidence | What it supports | Limit |
| --- | --- | --- |
| Frozen calibration replay on four fixed TRAIN cases | A's selected normalized/inherited law and the joint-mass repair were exercised without fitting, solver calls, optimizer updates, or TEST access. | Fixed-state comparisons do not predict either 100-epoch endpoint. |
| Candidate A formal fit | The clean 100-epoch TRAIN600 trajectory is complete; sampled losses for all eight physical terms fell by about 80–97% from epoch 1 to epoch 100. | A canonical endpoint shows a preliminary miss against original ME100, but a first 100-epoch fit is not a convergence or architecture verdict. |
| Candidate B formal fit and VAL89 | The matched clean 100-epoch TRAIN600 trajectory and canonical full/restricted VAL89 evaluation are complete; sampled losses for all eight terms fell by about 80–97%. | Preliminary endpoint differences versus A are small and mixed; no maturity or architecture verdict follows from the first full-data endpoint. |
| Historic R13-4 epoch-500 context | Its own Guide restriction changed mean errors by at most 0.248%, while its full reader still differed from the earlier reference across seven of nine means. | This is historical context, not an A/B result; full-reader quality and read restriction must remain separate. |

## Organizer

Both candidates preserve the live directed four-block relation matrix and its stage provenance over the two ordered cycles; M self-access remains explicit and no E identity edge is invented. On the input-selected VAL0678/M10 extraction, A has 10 separate M identity entries and 7 positive diagonal entries inside its applied EE block, alongside 778 off-diagonal EE, 1,672 EM, 14 ME, and 35 MM positive relations; B has the same 10 M identity entries, 26 EE diagonal, 4,879 off-diagonal EE, 1,030 EM, 24 ME, and 39 MM positive relations. The captured `cycle1` stages are zero-based index 1, the final second exchange cycle; retained EE diagonal entries remain in EE and are not added to M self-access. These learned relations are input-conditioned organization, not physical causality, and the predictor has no pure-E target. Records retain `(role, original_id)`, coordinates, physical measures, and controls, and all contributions coalesce back onto original members before value reduction. The views remain overlapping memberships that govern legal access to individual records, not pooled replacements.

Candidate B inherits A's calibration, score law, prefix, strengths, head, and paired objective. Its only planned read-law difference is organization-conditioned competition: for each selected view it measures retained mass `d_a = Σ_{i∈C} R_ai`, likelihood mass `Z_a = Σ_{i∈C} R_ai e_i`, and total retained prior `D = Σ_a w_a d_a`, then uses `b_i^B = e_i Σ_a [(w_a d_a)/(D Z_a)] R_ai` before the same typed strength. Empty views contribute zero; repeated memberships share one member likelihood and coalesce onto the original record. B adds three measured contractions, which remain part of its execution cost.

A fixed-context matched-marginal control over the original R13-5 epoch-100 state altered derived view rows while preserving support and the joint member prior to at most `6.6e-7` on TRAIN0006 and `3.8e-8` on TRAIN0231. On its 13 fixed native queries per case, A and B read traces changed by at most `7.2e-7` in the joint field. This perturbation produced no resolved field response and did not test a trained B checkpoint.

The separate B100 same-bounded intervention reused one loaded B100 core and context for both read laws. It made 52 query-mode-law comparisons per scene (13 queries × 2 modes × 2 laws), 104 across M1 and M12, while preserving support, candidates, and work. Maximum joint-prior residuals were `5.96e-8` at M1 and `2.24e-8` at M12, and maximum `R` row changes were `4.36e-6` and `9.34e-5`; native field changes remained at FP32 scale, with a largest sampled B temperature difference of `2.86e-6`. No resolved membership-dependent native effect was established. Positivity-limited row perturbations are a limitation of this control, not proof that B reduces to a marginal law. A separate final-ME-edge removal and re-preparation changed B100 temperature by `0.03030/0.03784` at M1 and `0.46779/0.47501` at M12 for full/restricted reads. This distinct input-conditioned structural perturbation establishes neither physical causality nor solver accuracy.

A separate exchange intervention removed one positive final-cycle ME edge and re-prepared context. On TRAIN0006 it removed M0←E185 (applied gate 0.6244) and changed the sampled native temperature by at most 0.0118; on TRAIN0231 it removed M10←E184 (gate 0.4669) and changed it by at most 0.0523. These are fixed-state input-conditioned structural interventions, distinct from the matched-marginal read control; neither establishes physical causality or a recovered physical graph.

## Execution and cost

On the immutable original ME epoch-100 state, the fixed native profile found that the joint95 prefix reduced selected value rows by 75.3% for TRAIN0006/M1 and 83.2% for TRAIN0231/M12. It still scored the same 1,572,864 E proposals and retained the same address support-pair work in each full and restricted call. Complete full-versus-restricted native medians were 148.41 versus 152.35 ms for M1 and 155.18 versus 156.35 ms for M12; both increases fell within measured call-to-call spread. The prefix reduced selected value work, while baseline address proposals, composition, and support-pair costs remained charged; this fixed profile established no inference speedup.

The equivalent common-plan rewrite was measured on a fixed 48-case TRAIN macro split into two 24-case microbatches with Q1024. Across one warmup and three alternating measurements, complete prepared paired forward, native loss, and backward time fell from a 3.742 s median for separate full/restricted plans to 2.547 s with one shared live plan (31.9% lower). No optimizer update was performed, and the timing excludes fresh data sampling. In a separate three-repeat scorer-factor trial with paired planning, the same complete prepared macro fell from 2.549 to 2.069 s (18.8% lower); peak allocated memory fell from 13,055,651,840 to 9,262,375,424 bytes (12.16 to 8.63 GiB), while peak reserved memory remained 16,900,947,968 bytes (15.74 GiB). Candidate masks matched exactly, native output differences stayed within a few `1e-5`, the whole-gradient-vector relative-L2 difference was at most `2.93e-6`, and individual group differences were at most `4.97e-6`. These separate controlled same-state comparisons exclude data and optimizer work; they support the measured planning and factoring scopes, not a formal-run throughput claim.

Complete inference did not improve in the scorer-factor trial: five-call medians ranged from 145.6 to 147.2 ms for TRAIN0006 full reads and from 148.9 to 150.8 ms for TRAIN0231 full reads; the corresponding restricted medians moved from 150.9 to 151.4 ms and 156.4 to 156.7 ms. The formal endpoint profiles then measured five alternating full/restricted pairs on fixed TRAIN0006/M1 and TRAIN0231/M12 checkpoints. All restricted medians were slower than their full-reader counterparts. The associated 48-case ordinary macro timings included fresh query sampling, collation, and device transfer but no optimizer update: A measured 2.3961 s and B 2.3865 s. These independently trained states and candidate prefixes differ, so the near-equal numbers do not establish that B's contractions are free or that B is faster. B added three contractions of 17,374,846,464 pairs each per macro; separate checkpoint instrumentation observed nine backward-replay closure entries in each 24-case microbatch. Across the ordinary macro timing repeats, peak allocated/reserved memory was 10.80/17.96 GiB for A and 10.44/15.59 GiB for B. No sparse inference speedup was measured.

| Measured scope | Result | Interpretation |
| --- | --- | --- |
| Fixed native inference, immutable original ME100 M1 and M12 | Selected-value rows fell 75.3% and 83.2%; complete restricted calls were 1.2–4.0 ms slower, within sample spread. | Sparse work fell; end-to-end inference savings were not measured. |
| Shared paired planning, fixed 48-case macro | 3.742 s → 2.547 s median, with paired preparation, both reads, loss construction, and backward; data and optimizer excluded. | A measured improvement for this isolated planning rewrite. |
| Scorer factoring with paired planning, fixed 48-case macro | 2.549 s → 2.069 s median; allocated peak 12.16 → 8.63 GiB; reserved peak 15.74 GiB in both arms. | A separate measured factoring improvement; no optimizer update. |
| Formal A/B ordinary endpoint macro, 48 TRAIN cases | A 2.3961 s; B 2.3865 s, including fresh sampling and transfer but excluding optimizer; peak allocated/reserved 10.80/17.96 and 10.44/15.59 GiB. | Distinct trained states and prefixes prevent a causal timing comparison; B's three contractions and full baseline costs remain charged. |
| Formal A/B complete native inference, fixed TRAIN M1 and M12 | A full/restricted: 147.68/152.13 and 152.59/157.84 ms; B: 155.17/161.42 and 164.29/166.47 ms. | Restricted calls were slower in every measured pair; no sparse inference speedup. |
| B additional contraction and checkpoint replay | Each of three contractions executed 17,374,846,464 pair operations per 48-case macro; backward replay observed 9 closure entries per 24-case microbatch. | Additional work is measured and remains part of B's cost. |

## Figures

| Family | Embedded evidence |
| --- | --- |
| Flow reconstruction | [native_u](../../../../diagnostics/round13_5_supplement_20261027/figures/native_u.png) |
| Thermal and local interfaces | [native_temperature](../../../../diagnostics/round13_5_supplement_20261027/figures/native_temperature.png); [native_local_interfaces_and_material](../../../../diagnostics/round13_5_supplement_20261027/figures/native_local_interfaces_and_material.png) |
| VAL89 accuracy | [01_val89_fidelity_overview](../../../../diagnostics/round13_5_supplement_20261027/final_quantitative_figures/01_val89_fidelity_overview.png) |
| Training and learning-path health | [02_sampled_train_native_losses](../../../../diagnostics/round13_5_supplement_20261027/final_quantitative_figures/02_sampled_train_native_losses.png); [03_native_learning_path_health](../../../../diagnostics/round13_5_supplement_20261027/final_quantitative_figures/03_native_learning_path_health.png) |
| Applied organization and member reading | [applied_H_stages_VAL0678](../../../../diagnostics/round13_5_supplement_20261027/organization/figures/applied_H_stages_VAL0678.png); [spatial_R_views_fixed_scenes](../../../../diagnostics/round13_5_supplement_20261027/organization/figures/spatial_R_views_fixed_scenes.png); [query_member_read_progression_VAL0678](../../../../diagnostics/round13_5_supplement_20261027/organization/figures/query_member_read_progression_VAL0678.png) |
| Mechanism controls and execution | [fixed_state_mechanisms](../../../../diagnostics/round13_5_supplement_20261027/figures/fixed_state_mechanisms.png); [organization_sensitive_read](../../../../diagnostics/round13_5_supplement_20261027/figures/organization_sensitive_read.png); [execution_cost](../../../../diagnostics/round13_5_supplement_20261027/figures/execution_cost.png) |

The formal per-run loss curves are retained as PDF masters: [Candidate A TRAIN600 loss curves](../../../../diagnostics/round13_5_supplement_20261027/thermal_r13_5a_consistent_joint95_train600_formal_v1/plots/training/loss_curves.pdf) and [Candidate B TRAIN600 loss curves](../../../../diagnostics/round13_5_supplement_20261027/thermal_r13_5b_organization_conditioned_joint95_train600_formal_v1/plots/training/loss_curves.pdf).

![Stored velocity targets, Candidate A and B joint95 predictions, and signed residuals for all original modules in the fixed TRAIN/VAL panel.](../../../../diagnostics/round13_5_supplement_20261027/figures/native_u.png)

Figure 1. The fixed four-case panel contains TRAIN0003 and exposed VAL0274, VAL0293, and VAL0678. It shows stored targets, A/B joint95 predictions, and signed residuals across all original modules; these panels do not show full-access fields or use a single-module selection. The fields support local comparison on this exposed panel, not TEST or solver fidelity. [PDF master](../../../../diagnostics/round13_5_supplement_20261027/figures/native_u.pdf).

![Stored fluid and material temperature targets, Candidate A and B joint95 predictions, and signed residuals across all original modules in the fixed TRAIN/VAL panel.](../../../../diagnostics/round13_5_supplement_20261027/figures/native_temperature.png)

Figure 2. For the same four cases, this view overlays fluid and all material temperatures for every original module, with stored targets, A/B joint95 predictions, and signed residuals in native units. It shows modulewise temperature agreement and disagreement beside the VAL89 summaries; it does not establish material generalization beyond the exposed panel. [PDF master](../../../../diagnostics/round13_5_supplement_20261027/figures/native_temperature.pdf).

![Full and restricted interface curves and material-temperature detail for the first-present original module selected by input order.](../../../../diagnostics/round13_5_supplement_20261027/figures/native_local_interfaces_and_material.png)

Figure 3. This is the only single-module view: the first-present original module was selected by input order, and the full/restricted interface curves retain its original ID and geometry alignment. On VAL0293, the plotted module peak is 17.17 in the stored target, 13.62 for A, and 18.21 for B; local material and surface differences remain visible despite close case-mean metrics. [PDF master](../../../../diagnostics/round13_5_supplement_20261027/figures/native_local_interfaces_and_material.pdf).

![Nine-observable full and restricted VAL89 fidelity summaries for Candidates A and B.](../../../../diagnostics/round13_5_supplement_20261027/final_quantitative_figures/01_val89_fidelity_overview.png)

Figure 4. This overview summarizes all 89 exposed VAL cases across nine native observables and full/restricted execution. Relative to A, B's restricted mean improves on six observables and worsens on surface temperature, material temperature, and q-proxy; both candidates remain above original ME100 on all nine means. [PDF master](../../../../diagnostics/round13_5_supplement_20261027/final_quantitative_figures/01_val89_fidelity_overview.pdf).

![Sampled TRAIN native-loss histories for Candidates A and B from epoch 1 through 100.](../../../../diagnostics/round13_5_supplement_20261027/final_quantitative_figures/02_sampled_train_native_losses.png)

Figure 5. All eight sampled TRAIN loss terms decline by approximately 80–97% between the independent epoch-1 and epoch-100 histories. These are optimization measurements on changing model weights, not held-out validation metrics or a scalar ConFIG direction. [PDF master](../../../../diagnostics/round13_5_supplement_20261027/final_quantitative_figures/02_sampled_train_native_losses.pdf).

![Fixed-panel native learning-path health observations for A and B at epoch 100.](../../../../diagnostics/round13_5_supplement_20261027/final_quantitative_figures/03_native_learning_path_health.png)

Figure 6. The native health probe uses four fixed TRAIN cases and a complete Q=1024 objective; some restricted E-value gradients are zero where the selected E role is empty (A M2; B M2/M5), while active selected paths remain connected. B's M12 restricted E proposal-score gradient is only about `9.7e-11`. Here `r` labels same-input execution variation from grad-enabled first-8-then-512 tiling versus ordinary no-grad512 execution with fresh preparation, so it includes layout/mode variation rather than pure repeat noise. [PDF master](../../../../diagnostics/round13_5_supplement_20261027/final_quantitative_figures/03_native_learning_path_health.pdf).

![Applied directed M/E contextual relation blocks for epoch-100 A and B on exposed VAL0678.](../../../../diagnostics/round13_5_supplement_20261027/organization/figures/applied_H_stages_VAL0678.png)

Figure 7. VAL0678 has M10 and 192 E records. In the captured `cycle1` final stage (zero-based index 1, the second exchange cycle), A has 10 separate M identity entries and 7 positive diagonal EE relations, while B has 10 M identity entries and 26 EE diagonal relations; those EE diagonals remain within EE. The plot shows applied learned input organization, not causal physical links. [PDF master](../../../../diagnostics/round13_5_supplement_20261027/organization/figures/applied_H_stages_VAL0678.pdf).

![Spatial normalized retrieval views over original M and E records with three input-selected query addresses per fixed TRAIN0003 and VAL0678 scene.](../../../../diagnostics/round13_5_supplement_20261027/organization/figures/spatial_R_views_fixed_scenes.png)

Figure 8. The fixed TRAIN0003/M1 and exposed VAL0678/M10 scenes preserve typed original-member identities, coordinates, and physical measures while showing learned view membership and access for three input-selected query addresses per scene. Marker area is scaled within each role, so its visual magnitude is not a cross-role comparison. Each view governs access to individual records rather than replacing them with a pooled feature, and does not imply physical causality. [PDF master](../../../../diagnostics/round13_5_supplement_20261027/organization/figures/spatial_R_views_fixed_scenes.pdf).

![Post-organization query reading for VAL0678 after the final-cycle H blocks have been prepared.](../../../../diagnostics/round13_5_supplement_20261027/organization/figures/query_member_read_progression_VAL0678.png)

Figure 9. The VAL0678 detail replay follows member proposals, the joint95 candidate prefix, and final original-member contributions after the final-cycle H blocks have been applied. It shows query reading conditioned on prepared organization, not the encoder exchange trace itself; M/E roles and original IDs remain explicit. [PDF master](../../../../diagnostics/round13_5_supplement_20261027/organization/figures/query_member_read_progression_VAL0678.pdf).

![Fixed-state calibration and stencil mechanisms from the separately identified original checkpoints.](../../../../diagnostics/round13_5_supplement_20261027/figures/fixed_state_mechanisms.png)

Figure 10. The calibration panels use the allowed original formal epoch-100 state, while the VAL0286 stencil replay uses a separate transferred development-epoch-600 state. Their source checkpoints and scopes differ from A/B training; the restricted q-proxy improvement coexists with worse aggregate, surface, and material metrics and is a mechanism result rather than an A/B predictor result. [PDF master](../../../../diagnostics/round13_5_supplement_20261027/figures/fixed_state_mechanisms.pdf).

![Trained B100 matched-marginal read-law control and separate final ME exchange-removal intervention.](../../../../diagnostics/round13_5_supplement_20261027/figures/organization_sensitive_read.png)

Figure 11. On B100, 52 matched query-mode-law comparisons per scene (104 total across M1 and M12) preserved support, candidates, and work; maximum prior residuals were `5.96e-8` at M1 and `2.24e-8` at M12, and the largest sampled B temperature change was `2.86e-6`. Separate removal/re-preparation changed full/restricted temperature by `0.03030/0.03784` at M1 and `0.46779/0.47501` at M12. The bounded read perturbation had no resolved native effect; edge removal is a distinct structural intervention, not a causality claim. [PDF master](../../../../diagnostics/round13_5_supplement_20261027/figures/organization_sensitive_read.pdf).

![Measured native inference and paired training-macro work for the fixed and formal A/B scopes.](../../../../diagnostics/round13_5_supplement_20261027/figures/execution_cost.png)

Figure 12. The endpoint profiles show slower restricted full-call medians in every measured M1/M12 pair, while isolated shared-plan and scorer-factor studies reduced paired training-macro time by 31.9% and 18.8%, respectively. B's three extra contractions and baseline address/composition/support costs remain charged; these measurements establish no sparse inference speedup. [PDF master](../../../../diagnostics/round13_5_supplement_20261027/figures/execution_cost.pdf).

## Tested commands and scope

These commands are recorded completed invocations, not rerun instructions; their output directories already contain retained results. Both formal fits and endpoint evaluations used source commit `fb9a542`: A used physical GPU 2 via `CUDA_VISIBLE_DEVICES=GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`, and B used physical GPU 1 via `CUDA_VISIBLE_DEVICES=GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e`; both appeared to the process as `cuda:0`. The six-epoch qualifications resumed from commit `8c19d13` on physical GPU 2. The recorded wrappers set `TMPDIR` to `ROOT/tmp`, `XDG_CACHE_HOME` to `ROOT/tmp/cache`, and the run-specific CUDA/PyTorch cache paths under `ROOT/tmp`.

Focused implementation validation passed 105 checks in 14.37 seconds; the separate output-only unified-loss-curve correction passed 11 tests, for 116 focused checks total. The suite covers routing arithmetic and loop gradients, identity/permutation/padding/empty-role behavior, typed coalescing, shared-plan and affine equivalence, and candidate IDs. The qualifications demonstrate native learning and exact epoch-3-to-6 resume. Durable implementation commits `8c19d13` and `fb9a542` are pushed; no production changes followed these checks.

```bash
cd /data2/wanglz/ModularDT/round13_5_supplement_20261027/source

# Completed formal A fit; physical GPU 2 was exposed as cuda:0.
CUDA_VISIBLE_DEVICES=GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39 TMPDIR=/data2/wanglz/ModularDT/round13_5_supplement_20261027/tmp XDG_CACHE_HOME=/data2/wanglz/ModularDT/round13_5_supplement_20261027/tmp/cache PYTORCH_KERNEL_CACHE_PATH=/data2/wanglz/ModularDT/round13_5_supplement_20261027/tmp/cache/torch/kernels PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wanglz/miniconda3/envs/ModularDT/bin/python -B HONF_Proj/tools/honf_research.py train-member-routing --config HONF_Proj/configs/research/thermal_r13_5a_consistent_joint95_train600_formal_v1.json --output /data2/wanglz/ModularDT/round13_5_supplement_20261027/thermal_r13_5a_consistent_joint95_train600_formal_v1 --epochs 100 --device cuda:0

# Completed formal B fit; physical GPU 1 was exposed as cuda:0.
CUDA_VISIBLE_DEVICES=GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e TMPDIR=/data2/wanglz/ModularDT/round13_5_supplement_20261027/tmp XDG_CACHE_HOME=/data2/wanglz/ModularDT/round13_5_supplement_20261027/tmp/cache PYTORCH_KERNEL_CACHE_PATH=/data2/wanglz/ModularDT/round13_5_supplement_20261027/tmp/cache/torch/kernels PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wanglz/miniconda3/envs/ModularDT/bin/python -B HONF_Proj/tools/honf_research.py train-member-routing --config HONF_Proj/configs/research/thermal_r13_5b_organization_conditioned_joint95_train600_formal_v1.json --output /data2/wanglz/ModularDT/round13_5_supplement_20261027/thermal_r13_5b_organization_conditioned_joint95_train600_formal_v1 --epochs 100 --device cuda:0

# Completed exact qualification resumes from epoch 3 through epoch 6, both on physical GPU 2.
CUDA_VISIBLE_DEVICES=GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39 TMPDIR=/data2/wanglz/ModularDT/round13_5_supplement_20261027/tmp XDG_CACHE_HOME=/data2/wanglz/ModularDT/round13_5_supplement_20261027/tmp/cache /home/wanglz/miniconda3/envs/ModularDT/bin/python -B HONF_Proj/tools/honf_research.py resume-member-routing --run /data2/wanglz/ModularDT/round13_5_supplement_20261027/qualification/A_v1 --device cuda:0 --stop-after-epoch 6
CUDA_VISIBLE_DEVICES=GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39 TMPDIR=/data2/wanglz/ModularDT/round13_5_supplement_20261027/tmp XDG_CACHE_HOME=/data2/wanglz/ModularDT/round13_5_supplement_20261027/tmp/cache /home/wanglz/miniconda3/envs/ModularDT/bin/python -B HONF_Proj/tools/honf_research.py resume-member-routing --run /data2/wanglz/ModularDT/round13_5_supplement_20261027/qualification/B_v1 --device cuda:0 --stop-after-epoch 6

cd /data2/wanglz/ModularDT/round13_5_supplement_20261027/source/HONF_Proj

# Completed A evaluation, using physical GPU 2 as cuda:0.
CUDA_VISIBLE_DEVICES=GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39 TMPDIR=/data2/wanglz/ModularDT/round13_5_supplement_20261027/tmp XDG_CACHE_HOME=/data2/wanglz/ModularDT/round13_5_supplement_20261027/tmp/cache /home/wanglz/miniconda3/envs/ModularDT/bin/python -B tools/honf_research.py evaluate-supplement-member-routing --checkpoint /data2/wanglz/ModularDT/round13_5_supplement_20261027/thermal_r13_5a_consistent_joint95_train600_formal_v1/checkpoints/epoch_0100_model.pt --device cuda:0 --output /data2/wanglz/ModularDT/round13_5_supplement_20261027/evaluations/A_e100

# Completed B evaluation, using physical GPU 1 as cuda:0.
CUDA_VISIBLE_DEVICES=GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e TMPDIR=/data2/wanglz/ModularDT/round13_5_supplement_20261027/tmp XDG_CACHE_HOME=/data2/wanglz/ModularDT/round13_5_supplement_20261027/tmp/cache /home/wanglz/miniconda3/envs/ModularDT/bin/python -B tools/honf_research.py evaluate-supplement-member-routing --checkpoint /data2/wanglz/ModularDT/round13_5_supplement_20261027/thermal_r13_5b_organization_conditioned_joint95_train600_formal_v1/checkpoints/epoch_0100_model.pt --device cuda:0 --output /data2/wanglz/ModularDT/round13_5_supplement_20261027/evaluations/B_e100
```

## Inverse

No inverse campaign was requested or run, so there is no inverse result. The report does not infer design validity from sampler diversity or from the interaction graphs.

## Next steps

For the predictor, the main measured result is mixed: B's restricted mean improves on six of nine observables versus A, but its surface-temperature, material-temperature, and q-proxy means worsen, and both e100 endpoints remain behind original ME100. Any maturity claim needs a separately authorized full-data trajectory and fieldwise review with particular attention to pressure, q-proxy, fluid temperature, local material/surface interfaces, and M7 local peaks; this supplement stops at the authorized 100-epoch endpoints.

For the organizer, the applied graph and query reader demonstrably preserve typed original-member access, while the bounded B100 row perturbation did not produce a resolved native response. A future causal-sensitivity question would require a separately authorized positivity-preserving perturbation that changes row identity enough to exceed FP32 and measurement limits, with the same scene and candidate work held fixed; the present exchange removal remains an input-conditioned structural intervention, not physical causality.

## Evidence index

The numerical sources below are read-only supplement artifacts under the DATA2 round root; the fixed calibration and structural controls use the specifically authorized original fixed checkpoints without fitting or TEST access. The formal fits, canonical evaluations, and fixed-state controls are separate evidence scopes.

| Evidence | Artifact |
| --- | --- |
| Fixed-state calibration and four-case native objective | [Calibration evidence](../../../../diagnostics/round13_5_supplement_20261027/diagnostics/fixed_state_calibration_evidence_v1.json) |
| Exposed VAL0286 stencil replay and relation snapshot | [Stencil evidence](../../../../diagnostics/round13_5_supplement_20261027/diagnostics/fixed_state_calibration_stencil_v1.json) |
| Baseline bottleneck and restricted work accounting | [Profile readout](../../../../diagnostics/round13_5_supplement_20261027/profile_bottleneck_readout.md) |
| Shared paired-plan complete timing | [Paired-plan timing](../../../../diagnostics/round13_5_supplement_20261027/paired_plan_timing_v2.json) |
| Scorer factoring, output/gradient parity, inference and training macro | [Affine trial](../../../../diagnostics/round13_5_supplement_20261027/diagnostics/affine_trial_shared_pair_normalized_inherited.json) |
| Original-state fixed-context matched marginal and exchange removal | [Original-state native control](../../../../diagnostics/round13_5_supplement_20261027/matched_marginal_native_control_v5.json) |
| Trained B100 matched-marginal control and separate ME-edge removal | [B100 native control](../../../../diagnostics/round13_5_supplement_20261027/matched_marginal_native_control_B100.json) |
| Independent formal initialization and membership | [Initialization comparison](../../../../diagnostics/round13_5_supplement_20261027/formal_initialization_comparison.json) |
| Formal wrappers, launch/evaluation receipts, and endpoint profiles | [A wrapper](../../../../diagnostics/round13_5_supplement_20261027/launch_A_formal100.sh), [B wrapper](../../../../diagnostics/round13_5_supplement_20261027/launch_B_formal100.sh), [A launch](../../../../diagnostics/round13_5_supplement_20261027/formal_A_e100_launch.json), [B launch](../../../../diagnostics/round13_5_supplement_20261027/formal_B_e100_launch.json), [A evaluation invocation](../../../../diagnostics/round13_5_supplement_20261027/evaluations/A_e100.invocation.json), [B evaluation invocation](../../../../diagnostics/round13_5_supplement_20261027/formal_B_e100_eval_invocation.json), [A cost](../../../../diagnostics/round13_5_supplement_20261027/profile_supplement_endpoints_A100.json), [B cost](../../../../diagnostics/round13_5_supplement_20261027/profile_supplement_endpoints_B100.json) |
| Qualification stop/resume receipts and focused implementation checks | [A epoch-6 resume](../../../../diagnostics/round13_5_supplement_20261027/qualification_A_e6_launch.json), [B epoch-6 resume](../../../../diagnostics/round13_5_supplement_20261027/qualification_B_e6_launch.json), [105 focused checks](../../../../diagnostics/round13_5_supplement_20261027/implementation_checks.txt), [curve-output correction receipt](../../../../diagnostics/round13_5_supplement_20261027/qualification_curve_output_correction_receipt.json) |
| B100 matched-layout heat response and repeatability check | [TRAIN0006/M1 replay](../../../../diagnostics/round13_5_supplement_20261027/B100_M1_heat_repeatability.json), [replay helper](../../../../diagnostics/round13_5_supplement_20261027/B100_M1_heat_repeatability.py) |
| Candidate A epoch-100 canonical VAL89 evaluation | [A endpoint result](../../../../diagnostics/round13_5_supplement_20261027/evaluations/A_e100/result.json) |
| Candidate B epoch-100 canonical VAL89 evaluation | [B endpoint result](../../../../diagnostics/round13_5_supplement_20261027/evaluations/B_e100/result.json) |
| Combined endpoint comparisons against A/B and historical references | [Endpoint comparison](../../../../diagnostics/round13_5_supplement_20261027/endpoint_comparison.json) |
| Applied A/B relation and member-read extractions | [A extraction](../../../../diagnostics/round13_5_supplement_20261027/organization/A_e100.npz), [B extraction](../../../../diagnostics/round13_5_supplement_20261027/organization/B_e100.npz) |
| Loss histories and clean-stop/resume evidence | [A formal history](../../../../diagnostics/round13_5_supplement_20261027/thermal_r13_5a_consistent_joint95_train600_formal_v1/metrics/history.json), [B formal history](../../../../diagnostics/round13_5_supplement_20261027/thermal_r13_5b_organization_conditioned_joint95_train600_formal_v1/metrics/history.json), [A qualification history](../../../../diagnostics/round13_5_supplement_20261027/qualification/A_v1/metrics/history.json), [B qualification history](../../../../diagnostics/round13_5_supplement_20261027/qualification/B_v1/metrics/history.json) |
| Historical full-reader gap versus restriction at epoch 500 | [R13-4 epoch-500 comparison](20261026_185102Z_HONF_Round13-4_Relation_Guided_Member_Access_Report.md) |

## Storage closeout

The cleanup receipts document removal of 63 disposable file versions totalling 6,866,225 bytes, including round caches, synthetic test payloads, superseded visual exports, and three generations of a single summary fixture. The retained round inventory is 477,210,137 bytes (455.1 MiB; 1,211 regular files, without following symlinks), including both formal and qualification checkpoints, complete histories, curves, native numerical exports, selected PDF/PNG figures, the source worktree, helpers, and provenance. At the closeout measurement, `/data2` had 3,209,786,163,200 bytes free and `/home` had 77,402,271,744 bytes free, above the required 100/20 GiB headroom; the round remained below its 8 GiB payload ceiling. The exact allowlist had no open handles or mappings, all allowlisted payloads are absent, and 138 retained run/evaluation/figure files were unchanged across cleanup. All three mandatory DATA2 bindings remain valid; the unrelated PhyCoFlow process was still present and received no actions. The remaining 69,788 bytes beneath the temporary root are compact provenance metadata and a preserved source helper. [Storage inventory and cleanup receipts](../../../../diagnostics/round13_5_supplement_20261027/cleanup/round_closeout.json).
