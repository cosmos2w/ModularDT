# HONF Round 13-5: joint M/E organization and original-member access

**Completed endpoint:** 27 October 2026, independent seed-0 TRAIN600 epoch 100. **Selected run:** `thermal_r13_5_me_joint95_train600_formal_v1`. **Decision:** continue this same state to the authorized absolute epoch 500 after report validation, cleanup and push; this report contains completed epoch 100 evidence, not an epoch 500 result.

R13-5 gives both original M and E records their own organizational views, composes those views through four captured contextual exchange blocks, and chooses one joint 95% prefix of original members before member scoring. The independent model is learning useful query-dependent access, including in M1 scenes. At epoch 100, its restricted reader improves all nine global VAL89 mean, p90 and worst errors over the saved independent R13-4 epoch 100 reference. Local reconstruction remains mixed, and reduced member work has not produced a complete inference speedup.

| Result axis | Completed gain | Remaining issue | Next measurement |
| --- | --- | --- | --- |
| Organizer | Native task feedback reaches E views and all applied nonempty blocks; M1 now has query-dependent priors and candidate sets | Learned retrieval rows strongly favor M members; no pure-E prefix is observed, and E views overlap substantially | Same-lineage learning through 500, then a separately requested endpoint comparison |
| Predictor | Restricted global means improve 2.80–29.57%; p90 and worst summaries also improve for all nine observables | M3 module-peak mean worsens 13.59%; fixed local modules can regress; joint 95 adds 1.16–4.83% to own-full means | Preserve all field and tail histories through the authorized continuation |
| Execution | Member scores and selected-value reductions are actually skipped after early admission | Complete native calls remain slower; full address scoring and dense composition remain charged | Later optimization must measure complete runtime, preparation and backward cost |
| Inverse | No inverse result was requested or measured | Input sensitivity and candidate diversity do not establish valid designs | No solver or inverse campaign is started |

## Scope, identities and completed fitting

The fixed input-only membership, native targets, TRAIN-only normalization and exposed canonical VAL89 are reused, excluding duplicate 0273. VAL89 contains M3/24, M5/25, M7/25 and M10/15 cases; it is historically exposed validation, including files with older `test` naming, and is not a pristine TEST claim. Detailed fields remain the fixed TRAIN0003 and VAL0274/0293/0678 panel. Health uses fixed TRAIN0006/0051/0322/0231 (M1/M2/M5/M12). No new dataset copy, membership freeze, solver response, inverse fit or error-selected training membership was created.

| Lineage | Initialization | New completed fitting | Meaning |
| --- | --- | ---: | --- |
| M qualification | Common seed 0, fixed TRAIN150/DEV22 | 6 epochs | Execution qualification |
| ME qualification | Common seed 0, fixed TRAIN150/DEV22; exact same-state 6→12 recovery | 12 epochs | Execution and resume qualification |
| M development | Literal R13-4 independent Guidepoch 500 checkpoint | 100 TRAIN600 epochs | Parent 500 plus 100; absolute 600 |
| ME development | Same literal Guidepoch 500 checkpoint | 100 TRAIN600 epochs | Parent 500 plus 100; absolute 600 |
| ME formal | Independent seed 0, no source weights or optimizer | 100 TRAIN600 epochs | Preliminary independent endpoint |

The round has completed 18 segmented epochs and 300 new full-data epochs. The final authorized same-state 100→500 segment adds at most 400, giving at most 700 new full-data epochs including development. The segmented ceiling is 500, not a target. The formal prospective horizon 500 and constant learning rate 0.0003 were declared before its first 100 epochs; continuation changes neither the optimizer schedule nor scientific recipe.

The formal and R13-4 first 100 histories match TRAIN600 order, VAL89 order, existing case/query sampling records for every epoch, seed 0 initialization family, train normalization, effective batch 48, microbatch 24, primary query count 1024, optimizer settings and constant learning rate. Their reader equations and paired objectives differ; this is a matched-age architecture comparison rather than a causal isolation of E addressing alone. R13-3/R13-4 epoch 500 artifacts are mature context, not an equal-age acceptance threshold.

## Figure index

| Evidence | Selected figure |
| --- | --- |
| All fields, admission fidelity and measured member work | Figure 1 |
| Native u and temperature residual maps | Figures 2–3 |
| Original-module material and surface fidelity | Figure 4 |
| M1 proposal→prior→candidate flow and typed reads | Figures 5–6 |
| Four applied stages, self ledger and distant/shared tails | Figures 7–8 |
| Matched learning and prescribed-input sensitivity | Figures 9–10 |

## Predictor: completed independent epoch 100

All errors below use the saved native extraction and masks in packed dataset units. Each triple is **case-mean / case-p90 / worst case RMSE**; module peak is the declared endpoint temperature observable. Cases carry equal weight in these reductions; a global improvement does not imply every case or member improves.

| Observable | Guide 100 legacy restricted | ME100 joint 95 | ME100 worst ID | Change mean / p90 / worst |
| --- | ---: | ---: | --- | ---: |
| u | 0.14469 / 0.16873 / 0.22298 | 0.11495 / 0.13633 / 0.17061 | VAL0293 | -20.55% / -19.20% / -23.49% |
| v | 0.026423 / 0.030343 / 0.03607 | 0.01861 / 0.020928 / 0.023457 | VAL0289 | -29.57% / -31.03% / -34.97% |
| p | 0.086769 / 0.10075 / 0.11975 | 0.064694 / 0.073481 / 0.086558 | VAL0660 | -25.44% / -27.07% / -27.72% |
| Vorticity | 0.32146 / 0.36942 / 0.42188 | 0.27085 / 0.29698 / 0.34246 | VAL0293 | -15.74% / -19.61% / -18.82% |
| Fluid T | 2.2862 / 2.981 / 4.3869 | 2.2222 / 2.6231 / 3.6107 | VAL0295 | -2.80% / -12.01% / -17.69% |
| Surface T | 2.3108 / 3.4019 / 5.9007 | 1.9319 / 2.7206 / 3.9658 | VAL0300 | -16.40% / -20.03% / -32.79% |
| Material T | 2.1891 / 3.1738 / 8.0442 | 1.7962 / 2.5736 / 4.4154 | VAL0295 | -17.95% / -18.91% / -45.11% |
| q-proxy | 5.5953 / 7.4291 / 13.176 | 4.5039 / 5.6109 / 6.8517 | VAL0302 | -19.50% / -24.47% / -48.00% |
| Module peak | 2.2123 / 3.5625 / 9.2737 | 1.9187 / 2.8037 / 4.9422 | VAL0295 | -13.27% / -21.30% / -46.71% |

The full reader also improves all nine global mean/p90/worst summaries against Guide 100. Full access means all reachable eligible original members under this checkpoint’s learned directory, not an additional route that ignores organization. The following comparison separates its field reconstruction from joint 95 approximation.

| Observable | ME100 own full mean / p90 / worst | Joint 95 / own full change mean / p90 / worst |
| --- | ---: | ---: |
| u | 0.11254 / 0.1326 / 0.16443 | +2.14% / +2.81% / +3.75% |
| v | 0.018131 / 0.020635 / 0.022671 | +2.64% / +1.42% / +3.47% |
| p | 0.062893 / 0.072222 / 0.081095 | +2.86% / +1.74% / +6.74% |
| Vorticity | 0.26297 / 0.29136 / 0.33295 | +3.00% / +1.93% / +2.86% |
| Fluid T | 2.146 / 2.5714 / 3.5532 | +3.55% / +2.01% / +1.62% |
| Surface T | 1.8866 / 2.6141 / 4.0037 | +2.40% / +4.08% / -0.95% |
| Material T | 1.7755 / 2.4679 / 4.609 | +1.16% / +4.28% / -4.20% |
| q-proxy | 4.3228 / 5.3404 / 6.5085 | +4.19% / +5.07% / +5.27% |
| Module peak | 1.8303 / 2.7667 / 4.9445 | +4.83% / +1.34% / -0.05% |

The M-stratified mean changes below compare ME100 joint 95 with Guide 100 restricted. All absolute means, p90, worst IDs and per-case values for both readers and all references remain in [complete summary](../../diagnostics/round13_5_20261027/reviews/formal100_complete/summary.json) and its [CSV tables](../../diagnostics/round13_5_20261027/reviews/formal100_complete/field_summary.csv). Empty M strata are missing populations, not zero error.

| Observable | M3 (24) | M5 (25) | M7 (25) | M10 (15) |
| --- | ---: | ---: | ---: | ---: |
| u | -12.97% | -13.91% | -26.34% | -32.99% |
| v | -19.84% | -26.61% | -33.83% | -39.65% |
| p | -18.24% | -18.76% | -29.64% | -37.39% |
| Vorticity | -2.86% | -12.43% | -21.37% | -28.32% |
| Fluid T | +2.97% | +0.68% | -7.47% | -6.97% |
| Surface T | -12.80% | -9.60% | -20.69% | -21.80% |
| Material T | -3.55% | -18.86% | -25.55% | -14.64% |
| q-proxy | -3.01% | -18.83% | -25.72% | -24.78% |
| Module peak | +13.59% | -17.94% | -19.10% | -14.88% |

The main stratified miss is M3 module peak: restricted mean/p90/worst change +13.59/+8.59/+3.65%; its full-reader changes are +7.13/+8.54/+14.57%. Thus this miss is already present under full access. M7 flow means improve 21.37–33.83% at matched age. VAL0295 remains the new material and module-peak worst case, but its material RMSE falls 8.0442→4.4154 and peak 9.2737→4.9422. These are preliminary endpoint improvements rather than evidence of convergence or superiority to the mature epoch 500 references.

![Nine native errors, own-full admission effects, matched M versus ME development and fidelity versus member scoring work](../../diagnostics/round13_5_20261027/figures/native_accuracy.png)

Figure 1. Canonical VAL89, saved independent epoch 100 checkpoints: joint 95 mean changes range −2.80% (fluid T) to −29.57% (v), and all nine p90 changes are negative. Joint 95 adds 1.16–4.83% to this checkpoint’s full means. The lower-left development panel uses parent 500 plus 100 in both arms; the work panel uses actual VAL member scores (about 80.180→7.224 per native query) and also charges 197.9 address scores per query. Counts are not elapsed-time savings. [PDF master](../../diagnostics/round13_5_20261027/figures/native_accuracy.pdf).

![Fixed TRAIN0003 and three validation cases with native u targets, two independent epoch100 predictions and signed residuals](../../diagnostics/round13_5_20261027/figures/native_u.png)

Figure 2. Native u on TRAIN0003 and VAL0274/0293/0678, joint 95 for R13-5 and the saved restricted R13-4 reader, with identical geometry, prescribed controls, native query IDs and target masks. Columns share a target scale and the residual uses a signed native-unit scale; module circles retain original IDs and radius 0.45. The VAL89 u mean improves 0.14469→0.11495, while these fixed maps retain visible local flow residuals. [PDF master](../../diagnostics/round13_5_20261027/figures/native_u.pdf).

![Fixed native fluid and material temperature maps with aligned targets, predictions and signed residuals](../../diagnostics/round13_5_20261027/figures/native_temperature.png)

Figure 3. Same fixed panel and independent epoch 100 checkpoints, native fluid and original-module interior temperature. Material coordinates reuse the preserved native local receiver recipe, reconciled by original ID and target mask; nearest native grid cells add no interpolation or new physical reference. Global surface/material means improve 2.3108→1.9319 and 2.1891→1.7962; the maps still show spatially structured residuals. [PDF master](../../diagnostics/round13_5_20261027/figures/native_temperature.pdf).

![Original-module material RMSE and signed native surface residuals for the fixed detailed panel](../../diagnostics/round13_5_20261027/figures/original_module_detail.png)

Figure 4. All original modules in the fixed panel, packed native temperature units; solid R13-5 versus dashed R13-4 at epoch 100. TRAIN0003 material RMSE worsens 0.6450→1.0248, and VAL0293 M4 worsens 2.8234→6.7995 even as M0 improves 8.0629→4.1194. VAL0678 has broad gains with local misses (M6:1.2159→2.4403; M8:2.5750→3.1713). The complete per-module values are retained; global peak/material reductions do not replace this local fidelity check. [PDF master](../../diagnostics/round13_5_20261027/figures/original_module_detail.pdf).

## Organizer: actual relations and usable E views

Preparation captures final-cycle applied gates `H_exchange=[[G_MM,G_ME],[G_EM,G_EE]]`, then adds the declared M-only identity ledger `diag(I_M,0_E)`. MM uses cycle-boundary M states; EM uses boundary E and after-MM M; EE uses after-EM E; ME uses after-MM M and after-EE/FFN E. The captured EE diagonal is preserved without a second E identity. These are directed stage-specific applied gates, not simultaneous head-specific transport coefficients or a physical influence matrix. Original typed identity `(role,original_id)`, coordinates, measures and owning-object controls remain intact.

One shared scorer supplies independent M/E address scores with role-balanced base measures. One whole-row retrieval normalization gives `R_ai=H_ai μ_i/sum_j(H_aj μ_j)` and `P_i=sum_a π_a R_ai`; μ is applied once. Repeated paths coalesce into the original member column before a single typed 95 prefix and before member MLP/value reduction. Conditional typed preferences and learned strengths feed the existing single general joint head. There are no extra trainable parameters, pooled member replacements, forced local edges, K/type quotas, or additional decoder branches. The [implementation guide](../guides/R13_5_Joint_Member_Routing.md) describes exact schemas and equations.

The two 512-query fixed scenes reconcile all four captured blocks, exchange/self ledgers, R and P, with M padding width 12 explicitly separated from present module count. The table distinguishes E-address use from E-member mass; an E address can organize useful M access.

| Fixed scene | E-address proposal range | Current P(E) range | Maximum eligible R row mass M / E | Restricted M-only / mixed / E-only |
| --- | ---: | ---: | ---: | ---: |
| TRAIN0003 / M1 | 0.0121–0.9496 | 0.0204–0.0870 | 0.9802 / 0.1145 | 313 / 199 / 0 |
| VAL0678 / M10 | 0.0022–0.7551 | 0.0123–0.0954 | 0.9949 / 0.1865 | 275 / 237 / 0 |

At this learned endpoint, the maximum E row masses are below 0.95, so no address proposal can yield an E-only 95 prefix in these two row families. This is a measured property of their current H/R, which continue learning; it is not an immutable architecture restriction. Controlled tests with both roles present establish representable pure-M, pure-E and mixed execution. No quota is imposed. M1 has 313 M-only and 199 mixed reads, and multi-M has 275 M-only and 237 mixed reads; neither means scene controls or upstream E exchange were skipped.

![M1 legacy versus joint address proposals, distinct prior normalizations and actual pre-score candidate counts](../../diagnostics/round13_5_20261027/figures/organization/M1_role_flow.png)

Figure 5. Three fixed input-only queries from TRAIN0003’s 512-query panel. The Guidepoch 500 column is unequal-age mechanism context: its M and E priors are separately normalized, and mandatory M plus legacy E99 is different from the new joint 95 policy. The formal 100 column uses one normalized joint P; its three reads are M-only, M-only and M+E. Across all 512 queries, E-address proposal mass ranges 0.0121–0.9496 while P(E) ranges 0.0204–0.0870, demonstrating that address type and retrieved member type are different measurements. [PDF master](../../diagnostics/round13_5_20261027/figures/organization/M1_role_flow.pdf).

![Same M1 queries with original typed candidate IDs, conditional preferences and learned read strengths](../../diagnostics/round13_5_20261027/figures/organization/M1_typed_preferences.png)

Figure 6. Same queries and checkpoints as Figure 5. Each nonempty role’s conditional preference sums to one independently; rho supplies its learned read strength. At q3, E48 alone survives joint 95 and therefore has conditional preference 1, alongside M0. At q1/q2 no E member survives and the E context is zero. A small joint role prior is not itself a small conditional final read, which motivates the explicit admission-tail diagnosis below. [PDF master](../../diagnostics/round13_5_20261027/figures/organization/M1_typed_preferences.pdf).

![Four applied directed stage rows, separate M identity and captured EE diagonal in the formal M1 scene](../../diagnostics/round13_5_20261027/figures/organization/R13-5_stage_relation_rows.png)

Figure 7. Formal 100 TRAIN0003, fixed native query 6465 at(6.14,4.73). MM has zero allowed positive off-diagonal entries for M1; ME at M0 has 9 positive E entries, and selected E156 has one positive EM entry and 47 positive EE entries. M identity remains a separate coefficient 1; the selected captured EE diagonal is 0 and remains 0. Raw gate colors and typed original IDs describe learned execution, not physical causality. [PDF master](../../diagnostics/round13_5_20261027/figures/organization/R13-5_stage_relation_rows.pdf).

![Actual local and distant module-address relation rows and their shared original-member tails](../../diagnostics/round13_5_20261027/figures/organization/R13-5_multiM_local_distant_shared_tails.png)

Figure 8. Formal 100 VAL0678, fixed native query 4289 at(6.14,3.14). M6 is local (distance 0.87, proposal 0.669), M5 is distant (distance 4.87, proposal 0.002); both have positive applied MM/ME relations, with 4 shared M and 9 shared E tail members. No locality or winner rule created these displayed rows. Stars are organizational addresses, circles use native radius 0.45, and final reading remains over original records. [PDF master](../../diagnostics/round13_5_20261027/figures/organization/R13-5_multiM_local_distant_shared_tails.pdf).

## Productive native learning and paired losses

Every one of 100 saved epochs contains finite full and restricted losses for all eight physical training roles, and every role’s mean loss falls from epochs 1–10 to 91–100. The paired objective averages the two physical losses at 0.5 each; actual executed work is their sum, including recomputation, rather than half the work. These eight training roles differ from the nine reported observables because module peak is an additional evaluation reduction. Full histories, current curves, latest/best and the epoch 100 milestone are retained.

![Eight weighted training histories against Guide100 and separate full and joint95 native loss trajectories](../../diagnostics/round13_5_20261027/figures/matched_learning.png)

Figure 9. Complete first 100 TRAIN600 epochs, common seed 0 and sampling membership. Top rows show weighted physical training contributions, and lower rows show full versus restricted native losses independently; no scalar organizer score substitutes for fieldwise learning. The native full/restricted q-proxy epoch 100 losses are 0.25533/0.27626 and material 0.03210/0.03352. The curves are preliminary trajectories, not a convergence claim. [PDF master](../../diagnostics/round13_5_20261027/figures/matched_learning.pdf).

Twenty-one health observations cover four fixed TRAIN cases in both modes (168 case-mode records). Native objectives and gradients use 1024 primary samples and the complete prepared union of 2370–4355 queries, while retained routing-score traces cover only the first 8 union rows. EE/EM/ME actual positive-edge adjoints and route-parameter gradients are nonzero in 168/168 records; MM is nonzero in 126/168, with the 42 M1 records correctly having no allowed/positive MM edge or applied MM gradient. Nonzero dense forbidden-direction adjoints are explicitly excluded from path evidence.

M/E original-value gradients are nonzero in 168/168 and 167/168 records. E-address score gradients on the first 8 rows are numerically nonzero throughout (median 6.11e −8, minimum 1.13e −18); this shows connection, not uniformly strong use or all-query coverage. All 42 M1 mode observations retain E value feedback, including information that passed through contextual exchange. No recorded collapse alert was triggered. The [complete native learning review](../../diagnostics/round13_5_20261027/reviews/formal100_native_learning_review.json) retains all eight paired losses, exact provenance and the bounds of each gradient measurement.

![Fixed-input heat perturbation sensitivity compared with numerical repeatability in M1 and M12 scenes](../../diagnostics/round13_5_20261027/figures/native_control_sensitivity.png)

Figure 10. Same fixed q and geometry, prescribed module heat multiplied by 1.1, through the owning M input before contextual exchange. At epoch 100 all eight native outputs change above same-input numerical repeatability; across the complete health panel all 1344 field/mode/case/epoch response-to-repeatability comparisons exceed 1 (minimum 27.1, median 41958). This is learned prescribed-input dependence; no solver response accuracy or physical causal attribution is established. [PDF master](../../diagnostics/round13_5_20261027/figures/native_control_sensitivity.pdf).

## Development, calibration and diagnosed abnormalities

The revised independent R13-4 epoch 500 comparison guided this round: its restricted/full approximation was nearly exact, while seven mean regressions already existed under full access. R13-5 therefore retained original-member access and investigated both the full predictor and the new admission rule. The matched transfer pair ran concurrently on physical GPU1/M and GPU2/ME from the same original Guidepoch 500 state, preserving all 100 case/query records. ME improved six of nine restricted means versus M: u −2.39%, v +0.07%, p −3.27%, omega −1.75%, fluid T−2.89%, surface T+1.52%, material T−15.01%, q-proxy +2.24%, peak −11.63%. This justified testing ME independently, without treating transferred absolute 600 as 600 epochs of new-architecture scratch fitting.

Before fitting, four fixed TRAIN source replays compared count-based with prior-weighted member calibration. Prior-weighted/count native objectives and gradient cosines were M1:0.01449/0.01460,0.9995; M2:0.01958/0.01921,0.4919; M5:0.02470/0.03011,0.7394; M12:0.03885/0.09888,0.2112. Separate raw/whole-row-normalized and full/restricted replays reconciled H and exposed positively aligned full/restricted gradients (ME cosines 0.806–0.968; normalized/raw full 0.935–1.000). These are instantaneous sensitivities, not isolated trained benefits. No material equation defect was found; prior-weighted calibration, whole-row R and shared temperature 1 were retained for both arms.

Qualification setup initially failed before fitting because a native heat ndarray was not JSON serializable; conversion to a JSON list repaired it. The first curve rendering omitted new paired-native terms; an output-only correction rerendered the saved evidence without changing state. The first full-data transfer attempts stopped before their first added epoch because two saved source CUDA RNG streams were restored into a one-visible-device process. The transfer-only mapping used the historically declared active source cuda:1 stream on target cuda:0, preserving host RNG, compatible weights/Adam/provider/sampler state and documenting selected-stream scope; ordinary exact resume stayed unchanged. M6 and ME12 qualification, including same-state 6→12 recovery, and both 100-epoch development retries then completed.

The bounded actual-message-removal diagnostic initially exposed an adapter allowlist omission for the new schema. The maintained adapter and focused tests now permit the supported removal hook; no-intervention training arithmetic is unchanged. Native forward replay also corrected one-time diagnostic handling of query-dependent eligibility and padded M axes. Failure logs, repairs and successful receipts are retained; these setup/diagnostic failures added no fitting epochs. Focused implementation/runtime suites passed (124 tests with one CUDA-only skip, a later 94-test runtime review, seven curve tests, and 34 adapter/task tests; suites overlap and are not summed as unique tests).

### Admission tails are a separate mechanism

The ME transfer endpoint’s exposed VAL0286 q-proxy RMSE was 4.302 full versus 11.296 joint 95. A bounded post-hoc trace replay reproduced those native metrics without fitting or changing the fixed figure panel. At original M4 angle 53, q moved 5.266→−42.219 against target 16.549. The unchanged native stencil reconstructs the −47.485 difference from the changed surface/outside temperatures to 5e −7. Across 47 traced stencil queries, all five M candidates remained; E candidates changed from 168 to zero in 30 queries and one in 17. Sole E0 conditional preference changed about 0.157→1 with rho about 0.077, while M contexts stayed unchanged on that stencil. This locates an admission/conditional-renormalization discontinuity, not a new physics branch or query-coordinate change.

At independent formal 100, the same already-exposed case has q RMSE4.252 full and 5.068 joint 95 (+19.18%), below Guide 100’s 8.109/8.036 and below the formal all-VAL worst (full VAL0298:6.509; joint 95 VAL0302:6.852). The admission effect remains measurable, but the extreme transferred-case tail does not recur at this endpoint. We retain the prescribed joint 95 law and its explicit limits; this one-case audit authorizes no cutoff sweep, extra field branch, new training membership or second formal model. [Trace and exact stencil readout](../../diagnostics/round13_5_20261027/reviews/q_proxy_tail_diagnosis/val0286_exposed_audit_summary.json).

### Address cuts and shared organization

An eight-case no-fit formal 100 diagnostic compares no cuts, member 95, address 95 and both cuts independently. Across 24 bounded query traces their captured original uncut joint-prior ranges are respectively 1,0.9501–0.9835,0.9887–1 and 0.9381–0.9835. Full-reader E conditional preference captured by member 95 ranges 0–0.5499;95% joint-prior retention is not 95% conditional preference or a field-error bound. Two 95 stages are not a single 5% omission guarantee. Address 95 still performs the full M/E address scoring; it is not enabled in the selected trained model.

Formal 100 typed R support Jaccard/weight-cosine averages are M–M0.584/0.310, E–E0.835/0.963 and M–E0.111/0.450. E rows have 2105 repeated support-layout pairs but no exact coefficient-row duplicate in the fixed low/high-M reconstruction scenes. Existing composition already coalesces original-member visits. Exact identical-row grouping is algebraically checked, and repeated posting layouts can share indices while retaining live per-row weights; the current data offer no measured exact coefficient-row reduction or runtime benefit.

A three-neighbor relation-space E reconstruction preserves each destination’s positive support, original members, raw captured stage blocks, contextual states and member values. In fixed TRAIN M1/M12 its maximum training-scale-standardized masked-RMS full/restricted output changes are 0.0000625/0.000725 and 0.000267/0.004739. This is a derived decoder approximation, not a recaptured learned exchange. Self-only/exchange-only and hypothetical E-row reassignment controls remain distinct from actual message removal and re-preparation. Removing actual final-cycle ME M0←E185 in M1 or M10←E184 in M12 changes full/restricted fields by up to 0.00239/0.02648 and 0.00371/0.25121 respectively. These learned-execution interventions establish conditional use, not physical causality. Future sharing should start from exact index reuse and measured complete cost; no merge hierarchy, frozen organizer or pooled-value model was added. [Complete bounded diagnostics](../../diagnostics/round13_5_20261027/reviews/formal100_organization_diagnostics_summary.json).

## Work, memory and complete runtime

The endpoint profile uses fixed TRAIN0006/M1 and TRAIN0231/M12, one warmup and three measured complete native calls per mode on physical GPU1, pinned by UUID. Native preparation, all 8192 native queries, extraction, address/member scores, dense composition, sorting and reductions are charged. The paired native forward/backward profile is separate, also uses 8192 native queries, and performs no optimizer change; it is not the 1024-primary-query training epoch cost. Profiler spans overlap and are not additive totals.

| Scene / reader | Prepare median ms | Decode + extraction median ms | Complete median ms | M / E member-value rows | Peak allocated / reserved MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0006 / full | 34.50 | 102.84 | 137.34 | 8,192 / 1,572,864 | 232.22 / 337.64 |
| 0006 / restricted | 35.26 | 109.26 | 144.52 | 8,192 / 381,791 | 229.01 / 337.64 |
| 0231 / full | 44.18 | 124.21 | 168.36 | 98,304 / 434,176 | 247.11 / 914.36 |
| 0231 / restricted | 45.00 | 131.25 | 176.25 | 78,033 / 11,549 | 247.11 / 914.36 |

Restricted complete calls are about 5.23% slower in M1 and 4.68% slower in M12. Peak allocated memory falls only 232.22→229.01 MB in M1 and remains 247.11 MB in M12; allocator-reserved memory reflects process cache history and is reported separately. Both paths still execute 1,581,056/1,671,168 address-score rows and 305,143,808/340,918,272 dense coefficient products for M1/M12. Preparatory original-key/value projections remain charged once per scene. Early member admission avoids expensive member MLP rows, but that saving does not remove address, composition, indexing or sorting cost.

The separate paired native forward/backward on TRAIN0006 takes 2.071 s and peaks at 814.19 MB allocated/914.36 MB reserved; actual module hooks retain forward and checkpoint-recomputation calls.

The separate paired native forward/backward on TRAIN0231 takes 1.113 s and peaks at 541.05 MB allocated/922.75 MB reserved; actual module hooks retain forward and checkpoint-recomputation calls.

The development M/ME profiles on the same physical GPU2 also showed slower restricted complete calls in all four scene/arm combinations (M:119.94→122.46 ms and 152.59→166.20 ms; ME:136.67→139.66 ms and 170.84→178.59 ms). Concurrent training wall time on separate GPUs is not an isolated model speed comparison. Full access remains a measured reference, not an unreported deployment fallback. [Formal cost evidence](../../diagnostics/round13_5_20261027/profile_formal_e100.json); [development cost evidence](../../diagnostics/round13_5_20261027/profile_development_pair_summary.json).

## Epoch 100 decision and exact continuation boundary

The saved evidence supports the authorized continuation: the four-block relation path is correct, E views receive native feedback, fixed-input dependence remains far above numerical repeatability, all eight physical losses keep falling, and the matched-age field means and tails show a viable trajectory. The remaining M3 peak miss, individual module errors, conditional-admission sensitivity, broad E overlap and slower execution are measured developing limitations. There is no unresolved equation or disconnected learning-path defect that requires a restart or speculative head expansion. We select the tested continuous-address ME joint 95 design and preserve this one independent state.

The epoch 100 latest, best-field and milestone checkpoints all retain epoch 100. Latest includes model, Adam moments, Python/NumPy/CPU/CUDA RNG, provider and sampler state for the next epoch 101. The existing resume command uses this same run and its already-declared horizon 500; no optimizer reset, new normalization, second scratch model or additional development arm is authorized. The immediate startup receipt is a separate local operation record; successful startup is not completed 500 evidence.

```bash
CUDA_VISIBLE_DEVICES=GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python -B HONF_Proj/tools/honf_research.py resume-member-routing \
  --run /data2/wanglz/ModularDT/round13_5_20261027/thermal_r13_5_me_joint95_train600_formal_v1 \
  --device cuda:0 --devices cuda:0 --stop-after-epoch 500
```

The prepared [exact background command and handoff](../../diagnostics/round13_5_20261027/continuation_handoff_prepared.json) retains logging, health callbacks,10-epoch curves, latest/best and declared 100-epoch milestones. After completed report validation, cleanup and durable push, launch it, confirm correct source 100 and target 500 once, then end this task. Assign no observer, completion check or scheduled evaluation to this final continuation. [Later-on-request epoch500 comparison commands](../../diagnostics/round13_5_20261027/later_e500_comparison_commands.md) are prepared against the saved R13-3/R13-4 epoch 500 fields and organization; they are not executed or scheduled.

## Evidence preservation and round closeout

All generated outputs, one-time helpers, numerical arrays, checks, failure logs, curves and checkpoints remain ignored under `/data2/wanglz/ModularDT/round13_5_20261027`, reached through the declared workspace symlinks. DATA2 was verified mounted before payload work. The round retained-output budget is 4 GiB, temporary budget 512MiB, estimated peak 3.5 GiB, minimum headroom 100 GiB on DATA2 and 20 GiB on home. The shared dataset/catalogues are reused in place; protected migration originals and unrelated GPU0 work remain untouched.

Round cleanup removed 1,136 allowlisted files/symlinks totaling 35,865,680 logical bytes, including disposable tests/caches, abandoned setup locks, browser previews and superseded exports. The final removal pass rechecked ownership, device, inode, links, active handles/mappings and report dependencies; all allowlisted payloads are absent. It verified 324 retained files totaling 652,208,583 bytes before compact cleanup receipts: all scientific checkpoints, histories, metrics, selected native arrays, ten report PDF/PNG pairs, source helpers, failure evidence and initial user-edit backups remain. The 33 protected historical files retain their original metadata, and the unrelated GPU0 process retains its PID identity, owner, command and cwd. Original unstaged user deltas and the untracked Round12 draft were compared with their initial copies and preserved. [Exact cleanup receipts](../../diagnostics/round13_5_20261027/cleanup/round_cleanup_summary.json); [user-edit preservation](../../diagnostics/round13_5_20261027/user_edits_preservation_closeout.json).

Immediately before/after the final cleanup pass, DATA2 free capacity was 3,210,432,925,696 / 3,210,472,300,544 bytes (3.21047 TB after), and home was 78,320,254,976 / 78,320,254,976 bytes (78.32 GB after). Initial round free space was 3,211,126,390,784 bytes on DATA2 and 78,452,563,968 bytes on home. Filesystem movement includes unrelated jobs and allocation overhead; it is not the logical deletion total. All declared headroom and output budgets remain satisfied. No cache/data migration or historical-checkpoint retention change was performed.

The implementation/runtime fixes are already pushed as `9e1ce8e7` and `dae6138f`; the final durable report is committed and pushed after the completed endpoint checks and full outgoing-range audit. Generated figures, numerical outputs and one-time helpers remain local and ignored. [Figure and rendered-report review](../../diagnostics/round13_5_20261027/figure_visual_review.json).

### Evidence index

| Evidence | Local artifact |
| --- | --- |
| Fixed epoch 100 all-native evaluation | [VAL89 and fixed detailed panel](../../diagnostics/round13_5_20261027/evaluations/formal_e100/result.json) |
| All references, field tails, M strata and cases | [summary](../../diagnostics/round13_5_20261027/reviews/formal100_complete/summary.json) |
| Full paired losses and native health | [native learning review](../../diagnostics/round13_5_20261027/reviews/formal100_native_learning_review.json) |
| Source calibration and paired-loss replay | [calibration](../../diagnostics/round13_5_20261027/calibration_r13_4_e500.json), [raw/normalized and full/restricted](../../diagnostics/round13_5_20261027/joint_r13_4_e500_native_diagnostic_v2.json) |
| Matched development comparison | [pair review](../../diagnostics/round13_5_20261027/reviews/development_pair/development_pair_analysis.json) |
| Typed graph/prefix identity and stage readout | [formal organization readout](../../diagnostics/round13_5_20261027/organization_formal_e100_readout.json) |
| Fixed local module values | [local fidelity](../../diagnostics/round13_5_20261027/reviews/fixed_panel_local_fidelity.json) |
| Fit budget and preserved state | [epoch budget](../../diagnostics/round13_5_20261027/epoch_budget.json), [formal100 monitor receipt](../../diagnostics/round13_5_20261027/formal_first100_monitoring.json) |
