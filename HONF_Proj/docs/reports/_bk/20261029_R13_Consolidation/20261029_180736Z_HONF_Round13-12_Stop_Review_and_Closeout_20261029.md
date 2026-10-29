# HONF Round13-12: stopped experiment review and closeout

Round13-12 changed how an E-anchored query accesses original environmental records: P composes live module-to-environment destination distributions, while G conditions those distributions with the inherited final-EE geometry term. Both retain the general joint physical-field predictor, original M/E identities, prescribed controls, inherited score calibration and two contextual cycles. The measured recipes did not preserve matched-age field accuracy or deliver complete execution savings. At the user's instruction, all experiment expansion is cancelled and this round is closed using existing evidence only. This finding concerns the tested recipes and trajectories; it does not reject all module-mediated architectures.

## Completed work and stop status

| Lineage or stage | Actual completed training | Evidence status and disposition |
| --- | ---: | --- |
| Reused S TRAIN150, seed0 | Existing 1000 epochs; no new S fitting | Literal S200/S500 evidence reused for matched-age comparison; S1000 retained separately. |
| P TRAIN150 discovery, seed0 | 522 epochs | Native/organization/intervention/cost review completed at 500 epochs. The already-started continuation was clean-stopped at522; epochs501–522 have saved history/state but no completed comparable scientific assessment. |
| Inherited G TRAIN150 discovery, seed0 | 500 epochs | Native/organization/intervention/cost review completed at 500 epochs; stopped. |
| Named G-R radial revision, seed0 | 2 epochs | Initial execution segment completed before cancellation, with history, checkpoints and curves. Earlier zero-fit correctness/geometry qualification is not trained field-quality or organizer confirmation. No substantive native validation/cost assessment of the fitted G-R checkpoint was performed. |
| Independent scratch TRAIN150 confirmation | 0 epochs | Cancelled; no final recipe independently confirmed or selected. |
| Fresh TRAIN600 assessment and continuation | 0 epochs | All plans cancelled; no new full-population result or formal continuation. |

The total new TRAIN150 fitting was **1024 epochs across three lineages**: P 522 + G 500 + G-R 2. P's existing engine consumed an atomic clean-stop request at the nearest safe completed-epoch boundary, wrote its current validation/history/curves and latest engine state, acknowledged epoch 522, and exited successfully. Its history is continuous 1–522 with 150 declared TRAIN cases per epoch. G's history is continuous 1–500; G-R's is 1–2. All owned training/evaluation processes and owned launch sessions are closed. Both Luna monitors are closed, with no pending endpoint waits, automatic resumes or scheduled follow-ups. No other process was signalled. See [stop and retained-state receipt](../../../../diagnostics/round13_12_20261029/receipts/user_stop_closeout.json) and [owned-work closure receipt](../../../../diagnostics/round13_12_20261029/receipts/owned_work_closed.json).

The latest P recovery artifact is [latest engine checkpoint at the clean stop](../../../../diagnostics/round13_12_20261029/train150/thermal_r13_12p_e_anchor_mediated_train150_discovery_v1/checkpoints/latest_model.pt); its literal epoch500 milestone remains unchanged. G retains its epoch500/latest/best checkpoints, and G-R retains its epoch2/latest/best checkpoints. Complete histories, initial and current curves, declared 100-epoch milestones, normalization/provenance, negative results and numerical arrays are preserved. Historical command files are retained as an execution archive, not instructions to continue this cancelled round.

## Predictor: broad matched-age degradation

The principal scientific comparison is **S/P/G at literal epoch500**, using the same seed0 initialization, fixed TRAIN150/VAL22 membership, train-only normalization, sampling recipe, native objectives, paired full/restricted training and constant learning rate. These are segmented-development results, not full-population maturity or convergence evidence. VAL22 is an exposed development panel. Detail scenes0274/0293/0678 belong to canonical VAL89, originate in the packed source's `test` partition and lie outside VAL22; they are exposed validation illustrations, not an independent TEST assessment. TRAIN0003/M1 is an original-TRAIN routing illustration, with no held-out M1 fidelity claim.

All nine native means and all nine p90 values are worse for both P and G at 500 epochs. Restricted P mean deficits range12.2–41.0%; G deficits range17.6–34.9%. P fluid-temperature mean/p90 worsen41.0%/51.3%, and G material-temperature p90 worsens65.2%. P is worse on19/22 fluid-temperature cases and21/22 u/p/omega cases; G is worse on22/22 fluid-temperature cases. These broad, persistent differences cannot be dismissed as modest isolated fluctuation.

| Native observable | S mean / p90 | P mean / p90 | P change | G mean / p90 | G change |
| --- | ---: | ---: | ---: | ---: | ---: |
| u | 0.1686 / 0.21099 | 0.21653 / 0.2949 | +28.4% / +39.8% | 0.20849 / 0.2606 | +23.7% / +23.5% |
| v | 0.027654 / 0.033111 | 0.03444 / 0.041765 | +24.5% / +26.1% | 0.033551 / 0.042768 | +21.3% / +29.2% |
| p | 0.091757 / 0.11432 | 0.11681 / 0.14915 | +27.3% / +30.5% | 0.12379 / 0.15973 | +34.9% / +39.7% |
| omega | 0.3846 / 0.47562 | 0.47074 / 0.59605 | +22.4% / +25.3% | 0.45236 / 0.54003 | +17.6% / +13.5% |
| temperature fluid | 3.3344 / 4.367 | 4.7004 / 6.6072 | +41.0% / +51.3% | 4.3355 / 6.1989 | +30.0% / +41.9% |
| temperature surface | 3.8002 / 5.3625 | 4.5204 / 7.3918 | +19.0% / +37.8% | 4.8388 / 6.7985 | +27.3% / +26.8% |
| temperature material | 4.1721 / 5.6567 | 4.9042 / 7.3011 | +17.5% / +29.1% | 5.5546 / 9.3471 | +33.1% / +65.2% |
| q proxy | 8.9201 / 12.528 | 12.008 / 17.701 | +34.6% / +41.3% | 11.998 / 18.123 | +34.5% / +44.7% |
| module peak endpoint rmse | 4.5038 / 7.0186 | 5.0536 / 8.8413 | +12.2% / +26.0% | 5.772 / 9.8434 | +28.2% / +40.2% |

The deficit persists in **full-access mode**: all nine means and p90 values are worse there too. Full-mode P mean deficits are10.9–39.2% and p90 deficits19.6–50.3%; G mean deficits are16.2–36.3% and p90 deficits11.8–67.6%. Within each candidate, all four execution modes differ from full-mode means by only approximately±2%. Source/E coverage restriction therefore does not explain the main regression. Full per-case reductions, module-count-category coverage and all four modes are retained in [the500 numerical review](../../../../diagnostics/round13_12_20261029/reviews/discovery_e0500/review.json).

![All nine native equal-case means and p90 for matched S P G500 in full and restricted access](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500/native_fidelity_comparison.png)

Figure 1. Exposed VAL22, literal S/P/G500, native reference arrays and the same observable definitions. Bars are equal-case means and diamonds are p90; full and restricted access are distinct. Every physical observable is worse for both tested candidates, including full access. Native dataset units are preserved; no aggregate is substituted for fieldwise fidelity. [PDF master](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500/native_fidelity_comparison.pdf)

![Separate sampled weighted TRAIN terms and exposed development selector through epoch500](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500_learning/learning_review_through_epoch_0500.png)

Figure 2. Saved S/P/G histories through 500 epochs; all eight weighted sampled TRAIN contributions are shown separately from the exposed DEV field-selection metric. Training terms decrease while P/G validation selection stagnates relative to S. The reductions differ, and the figure does not invent a common validation objective. The later P522 tail is excluded from this matched-age figure. [PDF master](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500_learning/learning_review_through_epoch_0500.pdf)

![Saved P500 fluid velocity pressure temperature references predictions and signed residuals for scene0293](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500/P500_0293_fields.png)

Figure 3. Exposed canonical-validation scene0293, original source test partition, P500 restricted mode. Saved native fluid references, predictions and signed residuals retain their actual masks and source geometry coordinates. Recognizable field structure coexists with substantial temperature and flow errors; visual structure alone does not establish preserved accuracy. Matched S/G field panels and all-module solid/surface comparisons remain in the linked500 gallery. [PDF master](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500/P500_0293_fields.pdf)

## Organizer: active paths with unstable benefit

The contextual stages actually execute MM→EM→EE→ME in each of two cycles. Source selection and E reading then compose those stage-labelled relations with query information; this is not a new simultaneous physical-influence matrix. P normalizes each nonempty actual ME row using the original E measure, retains actual zeros, and combines those distributions with a live direct-E source. G changes the conditional geometry term. Original E values remain individually readable. Source alpha, direct distribution D, conditional distribution B, admitted candidates and final value beta are distinct quantities; low M value strength does not erase an M-mediated E route.

Fixed-context interventions demonstrate real model dependence. At P500, removing all M-mediated E routes worsens restricted fluid RMSE by 64.4% in 0274 and126.7% in 0678; suppressing M values separately worsens it 71.8%/146.5%. However, the same route can be harmful locally. In 0293 original M3, route removal reduces material RMSE 17.961→4.928 and surface RMSE 14.948→5.671, while M-value suppression alone leaves material RMSE 17.048. Whole-scene material error improves 37.3% under route removal while fluid error worsens 47.1%. Thus participation is established, but its benefit is inconsistent; removing all routes does not isolate one module's physical cause.

![Original0293 M3 saved reference and baseline route-removal and value-removal material temperature](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500/P500_0293_M3_route_diagnosis.png)

Figure 4. P500, exposed canonical-validation0293, restricted execution, the same original M3 and exact saved native receivers. The intervention holds contextual states fixed and recomputes legal remaining reads. Strong cold bias improves when mediated E routes are removed, whereas suppressing M values has much less effect. This is a model-path intervention, not a physical intervention or causal attribution. [PDF master](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500/P500_0293_M3_route_diagnosis.pdf)

G frequently uses the legitimate organized direct-E alternative. On 0678, its mean source alpha0 is 0.979, 91.4% of the fixed512 queries select no M, and mean admitted M/E counts are 0.086/90.02, compared with P 3.95/67.50 and S 6.36/9.75. G's own-E beta is only 0.00151 against total E beta 0.0521, so direct-source dominance is not own-E-value dominance. On M1, the inherited geometry remains connected and nonconstant but has weak address discrimination: conditioned-versus-unconditioned B has mean TV 0.0456, separated-anchor B95 sets have Jaccard 0.987, and 78.5% of row-centered geometry variance is a common destination profile. Across 192 anchors the geometry maximum favors only E0/E23. These measurements support weak useful address conditioning, not a disconnected path or a required geometric locality quota.

![G500 M1 stored direct geometry and conditional environmental distributions at matched anchor addresses](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500/G500_M1_prepared_conditionals.png)

Figure 5. TRAIN0003/M1 routing illustration, G500, original E addresses and source geometry coordinates. The single physical module has original ID M0. Stars mark E102/E108/E113 anchors; the red square marks M0, and gray lower-row points have exactly zero B. All displayed arrays were saved before cancellation; the inherited geometry bias and normalized conditional B are shown separately from actual applied exchange gates and final query values, on matched scales. Similar source distributions across separated anchors expose weak address discrimination. This is training-scene mechanism evidence, not held-out M1 field accuracy. [PDF master](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500/G500_M1_prepared_conditionals.pdf)

![P500 original-node overlapping source organizations and actual query read in scene0678](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500/P500_0678_hypergraph.png)

Figure 6. Exposed canonical-validation0678, P500 restricted mode, fixed saved query Q4289 at E108. The two leading M-source envelopes are a declared explanatory selection; every original E identity appears once, with remaining M records in a neutral strip. The actual read uses 38 E and 6 M values. Ellipse positions and areas are illustrative; they do not encode physical regions, discovered group count or causality. This query view is supplementary and does not stand in for a complete prepared global source library. [PDF master](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500/P500_0678_hypergraph.pdf)

**Missing organization evidence remains explicit.** The new global-organization guide calls for whole-layout access, original-member service maps and native overlapping hyperedges of the prepared interface. Existing P/G saved query views and actual stage snapshots are retained, but a complete prepared-source global gallery for D, EM source priors and M/(E,M)-addressed B was not completed. The historical C-only renderer cannot faithfully treat this interface as one fixed E-address directory. No synthetic directory, fabricated M receiver rows, reconstructed reverse exchange, model rerun or new export was added to fill that gap after cancellation. The missing full global view is unverified.

The existing regional collective probe retains set-versus-single-member prediction changes and nonadditivity, but native error effects are mixed and below1% in the measured scene/arm probes. No matched physical-response atlas exists for these fixed scenes, and the saved baseline zero-heat reference is analytic rather than a new solver result. Mature P/G zero-control response was not completed. None of these probes establishes physical causality or valid inverse designs.

## Row-mass alternative and gradient limits

The frozen P500 row-mass counterfactual reduces the amplification of weak normalized ME rows by retaining their original measured row mass. It improves0293 M3 material RMSE 17.961→5.219, but worsens **8/9 VAL22 mean errors in both full and restricted modes**. Restricted fluid-temperature mean rises 4.700→6.473 and material-temperature mean 4.904→6.404; only q proxy improves slightly. It is rejected as a broad repair. This was a same-weight zero-fit intervention, not a successfully retrained recipe. The full negative result is retained in [the row-mass counterfactual](../../../../diagnostics/round13_12_20261029/evaluation/P_e0500_rowmass_diagnostic/result.json).

Completed native B24/Q1024 FP32 gradient replay compared the previous and optimized preparation/cache paths at literal P200/G200. Gradient availability matched, global gradient relative L2 differences were 6.19e-7/9.98e-7, paired losses agreed, and native output relative differences remained at FP32 scale. This found no obvious cache-gradient error on that workload; it does not prove every training batch is unaffected. The separate fixed-batch unclipped norms were P500 1.010, G500 0.668 and S500 0.789, without ME dominating that sample. **A single-batch inspection cannot exclude clipping problems over the complete training trajectory.** No clipping-frequency experiment was performed or added after cancellation. See [the saved replay and gradient evidence](../../../../diagnostics/round13_12_20261029/receipts/b24_e200_cache_gradient_diagnosis.json).

A named radial G revision was implemented and correctness-qualified before cancellation, replacing rather than adding to the inherited geometry term with distance in native E-cell coordinates and no new parameters. Its actual fitted extent is only 2 epochs. Same-weight conditional-set differentiation and pretraining correctness checks do not establish fitted field preservation, stable organization or a final repair. It is not selected, confirmed or promoted.

## Complete latency, training computation and memory

The completed500 benchmark used physical GPU0, FP32, two warmups and five timed repeats, with no concurrent fitting. Complete native latency includes host/device movement, preparation, source/relation composition, candidate selection, query scoring, original-value reads, native extraction and CPU output. The paired training workload is the fixed TRAIN24/Q1024 native full/restricted forward and backward; it excludes an optimizer update and is not represented as an entire training-epoch speed measurement. Prepared decoding and complete execution remain separate in the saved evidence.

| Matched500 workload | S median ms | P median ms | G median ms |
| --- | ---: | ---: | ---: |
| Complete native M1, full | 121.94 | 160.01 | 164.58 |
| Complete native M1, restricted | 116.32 | 162.47 | 173.10 |
| Complete native M12, full | 117.28 | 165.31 | 164.27 |
| Complete native M12, restricted | 123.34 | 165.09 | 175.19 |
| TRAIN24 paired forward/backward | 504.23 | 759.26 | 1025.26 |

| Peak allocated memory at 500 epochs | S | P | G |
| --- | ---: | ---: | ---: |
| Complete native M1, restricted, MiB | 58.16 | 78.17 | 151.58 |
| Complete native M12, restricted, MiB | 58.50 | 60.83 | 113.48 |
| TRAIN24 paired forward/backward, GiB | 5.114 | 7.696 | 10.979 |

Restricted native execution is 33.9–39.7% slower for P and 42.0–48.8% slower for G than S. Paired forward/backward is 50.6%/103.3% slower, with higher allocated and reserved training memory. Repeat min/max, baseline/peak/increment allocated and reserved bytes, exact shapes and executed score/value/preparation counts are retained in [the complete cost comparison](../../../../diagnostics/round13_12_20261029/receipts/discovery500_complete_cost_comparison.json). A reduced E-score count or narrower learned support does not establish end-to-end savings.

![Measured complete restricted native and paired training latency and peak allocated memory for matched S P G500](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500/complete_cost_comparison.png)

Figure 7. Restricted native and paired training, same-GPU500 cost evidence; error bars retain timed-repeat min/max around medians, and memory is peak allocated over repeats. Native cases use8192 fluid receivers,3096 material receivers per M and64 surface angles. TRAIN24 uses Q1024,32 material receivers per M and16 surface angles. Workload boundaries differ and are not added into a fabricated total. Neither tested mediated recipe has an end-to-end benefit. [PDF master](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500/complete_cost_comparison.pdf)

Earlier execution work removed demonstrated redundant dispatch, repeated preparation/indexing, P normalization and score-union sorting; measured P200 high-M latency improved 236.1→204.5ms while still exceeding S 134.6ms. The later matched500 benchmark still shows the residual cost above. Its existence is measured; not every remaining component is quantitatively attributed. No further optimization or diagnostic experiment is undertaken to complete this report.

## Decision, scope and retained evidence

**Stop rationale:** the current P/G recipes combine broad matched-age field degradation, inconsistent route benefit, weak inherited-G address conditioning and higher complete costs. The row-mass alternative is not a repair, and the two-epoch radial revision provides no substantive basis for promotion. The available evidence does not justify further investment in this round. This is an evidence-based disposition of the tested recipes, not a verdict on all module-mediated architecture choices. No final recipe, independent confirmation or fresh TRAIN600 result is claimed.

For the predictor, some field structure and decreasing training losses remain, but all-nine exposed validation means/tails and cost fail the intended preservation goal. For the organizer, live original-member routing and overlapping service are established, but stable useful organization, strong address differentiation and the complete prepared-source global gallery are not established. For inverse work, no fitting, design trail or validity result was produced; it remains outside this round. All three dispositions are closed, with no further experiment proposed or scheduled.

The compact figure index is: Figure 1 all-nine means/tails; Figure 2 learning terms and selector; Figure 3 native fluid fields/residuals; Figure 4 original-M3 path diagnosis; Figure 5 inherited-G conditioning; Figure 6 supplementary native source hypergraph/read; Figure 7 complete costs. The [500 figure gallery](../../../../diagnostics/round13_12_20261029/figures/discovery_e0500/README.md) retains matched S/P/G field, material, surface, all-module peak, incidence and query panels with separate PDF masters. The [detailed500 review](../../../../diagnostics/round13_12_20261029/reviews/discovery_e0500/README.md) preserves the numerical reasoning and independent review recommendation. All included PNG panels use relative links; generated media and numerical evidence stay in the ignored DATA2 payload.

Durable implementation and recipe integration were tested before the stop instruction: the last focused/historical suite completed 103 passing tests, including dense reference gradients, A/L behavior, empty M/ME, high-contrast own-E access, configuration identity and checkpoint reload. The radial integration was audited and pushed as `fa001984bfb1c183edd2a2ec3a7fa0f978241cc0`. No new training, test suite, separate model-evaluation run or diagnostic experiment was launched after cancellation; the existing P engine completed its ordinary clean-stop boundary bookkeeping. Closeout checks are limited to operational stopping, existing-file/report links, image decoding/rendering, Git audit and minimal cleanup. Previously executed command paths and qualifications remain in [the historical command archive](../../../../diagnostics/round13_12_20261029/receipts/tested_manual_commands.md) and [the pre-stop integration receipt](../../../../diagnostics/round13_12_20261029/receipts/radial_integration_push.json); unused continuation commands are cancelled.

## Minimal cleanup and upload closeout

Final cleanup removed exactly four superseded temporary PDF-page PNG previews, 606359 logical bytes. The recursive allowlist records ownership, inode/device/link count and checksum, rechecks metadata before unlink, scans current report dependencies and accessible process handles/maps, and confirms protected train/evaluation/script/cache metadata unchanged. Inaccessible process sections are explicitly reported, rather than claiming privileged global visibility. No checkpoint, history, selected figure, source array, negative result, shared dataset/cache or migration backup was removed. See [the final cleanup receipt](../../../../diagnostics/round13_12_20261029/receipts/cleanup_preparation/final_minimal_cleanup_receipt.json).

Including the earlier documented visual/bytecode and renderer cleanup, the round removed 32 files and 12344680 logical bytes. Earlier receipts retain their actual, narrower consumer-check limits. The retained round occupies approximately 1.23 GiB before final report QA, far below the 24 GiB retained/4 GiB temporary limits. At the final minimal-cleanup measurement, DATA2 had 3197664870400 bytes free and home had 38542917632 bytes free; final capacity and retained inventory are recorded in the [closeout receipt](../../../../diagnostics/round13_12_20261029/receipts/final_closeout.json). All three declared workspace output symlinks remain bound to DATA2. Unrelated jobs and pre-existing user edits are preserved.

Only this durable report is added in the final documentation commit. The entire outgoing commit range and object paths are audited, the repository pre-push artifact gate is enabled, and the final local/remote branch-tip match is recorded in the local closeout receipt. Generated figures, data, caches, source snapshots and one-time render/diagnostic/cleanup helpers are excluded from Git. This report closes the round; no experimental continuation remains authorized by this execution.
