# Round12-3: Reusable singleton groups execute, but removing E→E fails surface fidelity

The new path removes ordinary E→E communication from G-F, independent retrieval membership, M compaction, direct M-value decoding and group-to-original-E expansion. Each of two interaction layers now contextualizes E from M, pools E into an explicit group representation, transforms that representation once, and uses its cached keys/values to update M. Queries read those same final group keys/values through bounded module postings and a geometric entrance. G-EE adds only an explicit E→E exchange as the matched communication control. The new core and native adapter execute; both scratch controls completed 100 development epochs. The adaptive G-H core completed native forward/backward/save/reload qualification, but G-H100 was not launched because G-F failed the prespecified surface-temperature preservation bands.

**Decision:** stop after the matched G-EE/G-F100 screen. G-F surface-temperature mean RMSE is 3.273 versus 2.788 for G-EE (ratio 1.174 > 1.10); its p90 ratio is 1.180 > 1.15. Heat-response preservation also misses several bands, single-geometry responses remain inaccurate against physical references, and zero-heat temperature rise remains nonzero. Deleting E→E saves training memory, but complete native inference improves by only 0.34–2.51% and the complete optimizer macro by 3.92%, below the 10% useful-gain criteria. No recovery reserve, continuation, additional architecture, inverse campaign, solver run or full-data training was used.

## Evidence and short figure index

The experiment uses fixed25 TRAIN150 and repeatedly exposed DEV22, normalization fitted only on selected TRAIN, seed 0, FP32, width 128/four heads/two layers, primary Q1024, nominal microbatch 24/effective 48 and AdamW 3e-4 / WD 1e-5. DEV22 is the selected part of the original source test partition, not an independent final test. The four detailed DEV cases are 0277/0287/0294/0684. All native quantities retain the packed dataset’s benchmark units; no new Kelvin, SI flux or physical-coordinate conversion is asserted. M1/M2 have six selected TRAIN examples each and no physical geometry-response atlas. They remain fitted TRAIN evidence.

| Figure | Question | PDF master |
| --- | --- | --- |
| 1 | All nine native boards and failed surface strata | [Native comparison](../../diagnostics/round12_3_execution_20261024/figures/01_native_comparison.pdf) |
| 2 | Native fields and signed residuals | [Fields](../../diagnostics/round12_3_execution_20261024/figures/02_native_fields.pdf) |
| 3 | Saved physical responses and zero-response usefulness | [Responses](../../diagnostics/round12_3_execution_20261024/figures/03_physical_responses.pdf) |
| 4 | Applied relations, reused values, interventions and complete native cost | [Reuse and cost](../../diagnostics/round12_3_execution_20261024/figures/04_group_reuse_and_cost.pdf) |

The retained [local evidence directory](../../diagnostics/round12_3_execution_20261024/) contains the input-only panel selection, original full histories/checkpoints through their run links, all-nine native metrics, physical response summaries, regional/intervention results, selected field/reference arrays, qualification evidence and cleanup receipt. Figures were generated with Python from these saved numerical files, visually inspected, and retained as PDF masters plus one PNG each for Markdown embedding. Generated payloads and one-time helpers are local on DATA2 and excluded from Git.

## Completed execution and fair exposure

| Arm | Physical GPU | Epochs | Primary visits / updates / microbatches | Engine seconds |
| --- | --- | --- | --- | --- |
| G-F | 0 | 100 | 15,000 / 400 / 700 | 865.04 |
| G-EE | 1 | 100 | 15,000 / 400 / 700 | 845.68 |
| G-H | 1 qualification only | no training | native M24 forward/backward ×3; no scientific Adam steps | 20.9 s for both qualification arms combined |

Each completed arm presents 15.36 million primary fluid queries, plus native surface/material/curl receiver work. Each has 100 response microbatches (50 heat and 50 single geometry), 200 response endpoint-scene visits and one separate two-scene no-step response calibration. Response fitting uses only admitted TRAIN0348 and the maintained 10%-of-shared-native-gradient target/downward cap before the same Adam step. Fixed coefficients are 1172.72 for G-F and 2027.75 for G-EE; these are gradient-balancing coefficients, not comparable physical loss units. Native objective weights remain fixed, including the explicitly defined epoch101 rule. No response fitting uses the other seven atlas families.

Common named initial tensors match exactly; only G-EE owns the extra E→E weights. Existing recorded case-order/query-sampling identities are compared in the retained verification receipt. Each arm saves epoch 100, latest and best-field; their model tensors are exactly equal within that arm because epoch 100 is its only declared monitoring checkpoint. Best-field is not an earlier, independently selected success. Native 100-epoch endpoints, saved response endpoints and the later intervention/cost passes are reported separately. A wrapper formatting error occurred after each completed native export; the exports were retained and reused unchanged while the missing response stage was subsequently run.

The initial qualification used real high-M24 batches with native forward/backward and strict checkpoint reconstruction, rather than a metadata dry run. Warm G-F forward/backward was approximately0.85/0.37s, peak about1.76GiB; G-H approximately2.25–2.32/0.54–0.57s, peak about1.62GiB. G-F native reload difference was0; the largest G-H native extracted difference was2.83e-5. This is an untrained sizing/reload observation, not a trained adaptive field or speed result and not bitwise GPU parity. The untrained qualification states and numerical receipt are retained. No work price or adaptive work-learning qualification was completed after the Stage-B stop.

## Predictor: useful native temperature information, failed preservation

| Native quantity | G-F mean RMSE | G-EE mean RMSE | Mean ratio | p90 ratio | TRAIN-mean predictor | G-F worst ID |
| --- | --- | --- | --- | --- | --- | --- |
| u | 0.0880 | 0.0985 | 0.894 | 0.907 | 0.4544 | 0677 |
| v | 0.0208 | 0.0233 | 0.894 | 0.915 | 0.0259 | 0684 |
| p | 0.0744 | 0.0916 | 0.812 | 0.782 | 0.0995 | 0688 |
| omega | 0.2574 | 0.2828 | 0.910 | 0.896 | 0.6913 | 0686 |
| temperature_fluid | 2.6952 | 2.6001 | 1.037 | 1.121 | 6.4489 | 0677 |
| temperature_surface | 3.2731 | 2.7882 | 1.174 | 1.180 | 7.1019 | 0663 |
| temperature_material | 2.9202 | 2.8158 | 1.037 | 1.100 | 7.0225 | 0663 |
| q_proxy | 7.9908 | 8.8378 | 0.904 | 0.954 | 7.5305 | 0645 |
| module_peak_endpoint_rmse | 2.6520 | 2.6893 | 0.986 | 0.902 | 6.5792 | 0655 |

Means are equal-case errors on native masks; p90 is over cases and is not a confidence interval. The module-peak board is the per-case RMSE of errors in separately recomputed original-module maxima over the identical stored material receiver sets. Both models beat the constant selected-TRAIN mean on all four priority temperature/peak boards; G-F’s q-proxy mean is worse than its constant baseline (7.991 versus7.530). G-F improves all four flow means relative to G-EE, but these improvements do not cancel the surface-temperature preservation failure.

![Figure 1. native_comparison](../../diagnostics/round12_3_execution_20261024/figures/01_native_comparison.png)

Figure1. Epoch100, exposed DEV22. The left panel divides equal-case native RMSE by the selected-TRAIN mean predictor error on the same masks; lower than1 beats that arithmetic baseline. The right panel shows surface-temperature means and p90 in native temperature units, with case counts. G-F/G-EE surface mean ratios are1.025/1.383/1.169/1.107 for M3/M5/M7/M10; the M7 p90 ratio is1.357. Thus the global failure is not a uniform temperature collapse, and the M5/M7 tails cannot be hidden by the mean fluid-temperature or peak boards.

![Figure 2. native_fields](../../diagnostics/round12_3_execution_20261024/figures/02_native_fields.png)

Figure2. Input-selected DEV0287 (M5), all stored native fluid receivers, with geometry and observation masks preserved. Rows are u/v/p/omega/T in native benchmark units; columns are saved physical reference, G-EE100, G-F100 and signed G-F minus reference. Reference/prediction scales match within each quantity; residual scales are separate. The figure exposes remaining local wake, pressure and thermal errors rather than certifying a physical surrogate from a scalar objective. Numerical fields for all four detailed DEV representatives remain retained, including surface/q-proxy/material targets and masks.

### M1/M2 TRAIN risk and boundary regions

| Quantity, TRAIN12 | G-F mean | G-EE mean | Mean ratio | p90 ratio |
| --- | --- | --- | --- | --- |
| u | 0.0797 | 0.0928 | 0.859 | 0.859 |
| v | 0.0141 | 0.0163 | 0.862 | 0.910 |
| p | 0.0552 | 0.0626 | 0.883 | 0.937 |
| omega | 0.1881 | 0.2055 | 0.915 | 0.959 |
| temperature_fluid | 1.9697 | 1.8622 | 1.058 | 1.223 |
| temperature_surface | 2.0331 | 2.2612 | 0.899 | 1.020 |
| temperature_material | 1.8958 | 1.9339 | 0.980 | 0.937 |
| q_proxy | 6.2719 | 7.3631 | 0.852 | 0.856 |
| module_peak_endpoint_rmse | 1.4462 | 1.2507 | 1.156 | 1.012 |

The fixed M1 cases are 0006, 0368, 0382, 0383, 0394, 0044; M2 cases are 0051, 0054, 0067, 0435, 0087, 0452. Selection uses module count and sorted Re/total-heat inputs only. On this fitted panel, G-F fluid-temperature p90 is1.223×G-EE and mean module-peak error1.156×G-EE, although mean surface/material errors do not regress. Boundary strips are the outer10% of the normalized domain; far-from-M means distance greater than0.15 times the shorter domain length. Their full u/v/p/omega/T results, counts and IDs are retained in each execution result. These measurements cannot establish M1/M2 physical geometry-response fidelity because the required reference does not exist in the admitted atlas.

## Organizer: values are used and reused; adaptive aggregation remains untested

Both completed controls have K_E¹=K_E²=192 on all 24 input-selected organizer scenes. Original module count remains M, and there is no learned M-value bank. Their singleton partitions bypass nomination/tree/policy work. Consequently these endpoints demonstrate a shared information/value route and bounded access, not learned grouping, compression or scene-adaptive subdivision. The connected exclusive G-H frontier is implemented and reversible, uses 352 catalogue-local edges per layer and linear children/statistics/ancestry-label storage, and has no target K or20-epoch leaf warmup. Its temperature/work schedules are continuous over absolute epochs1–500, with learning allowed at the first update. G-H100, parent/child physical sharing audits, TRAIN stage-price calibration and work-gradient qualification were not performed because the preceding communication screen failed. No result from an untrained small K is promoted into hypergraph success.

| G-F TRAIN case | M | Clamp early relay / ordinary T RMSE | Clamp final address / ordinary | Zero group values / ordinary | Exhaustive group search / ordinary |
| --- | --- | --- | --- | --- | --- |
| 0006 | 1 | 0.9956 | 1.0000 | 3.317 | 0.990 |
| 0051 | 2 | 0.9581 | 1.0004 | 3.287 | 0.955 |
| 0348 | 10 | 1.1051 | 1.0009 | 2.863 | 1.010 |
| 0631 | 12 | 1.1899 | 1.0005 | 2.903 | 0.900 |

The early intervention removes the actual group→M message in layer1, then recomputes every dependent later layer and query. It worsens high-M fluid-T errors by10.5% and19.0%, but improves the two small-M examples, so usefulness is scene-dependent. The final intervention removes the final group→M message while preserving the original group query candidates; it changes fluid-T RMSE by less than0.09% in G-F. Selected proposal logits and final M states have nonzero native gradients, including M1, but the measured final exchange utility is weak and not established. This is a reason to simplify or reject this version in subsequent design work, not to force a new loss or bypass that makes the path indispensable.

Zeroing the cached final group values under fixed query candidates multiplies G-F fluid-T RMSE by2.86–3.32. This establishes executed value dependence on the bounded four-scene panel. The same prepared group key/value tensors consumed by group→M are the source tensors gathered by the query reader; they are not independently reconstructed after M changes. The ordinary decoder reads zero original-E values and zero M values, with at most two module proposals, eight postings, nine spatial cell visits, four geometric entrances and eight retained group candidates. Empty/overflow/truncation counts are retained. These are computational relationships, not physical causal relations.

![Figure 4. group_reuse_and_cost](../../diagnostics/round12_3_execution_20261024/figures/04_group_reuse_and_cost.png)

Figure4. Top: actual applied R and B for original input column0 on TRAIN0348 (M10), followed by read frequency of the same final singleton group IDs for the first 1024 native-union queries. Each group is one E record here; no adaptive group diagram is implied. Bottom: real early/final-message, zero-value and exhaustive-group-search interventions on TRAIN0006/0051/0348/0631 (M1/M2/M10/M12), and complete native inference on the same idle physical GPU0. Group values carry useful information, but final address-update utility is weak; inference gains remain0.34–2.51%.

Exhaustive search scores the same192 prepared groups, not original E. Restricted search scores eight allocated group slots per query, a95.83% reduction versus that diagnostic, before any group neural score is computed. The exhaustive reader uses a deliberately different address function, so this is not a parity test. On TRAIN0631 its fluid-T error is0.900×the restricted error, exposing a possible coverage cost. Restricted-versus-exhaustive all-nine physical results remain retained; reduced scored groups alone do not qualify search quality. Same-function query chunking, replay and native/source-shape behavior are covered by focused tests.

## Responses: heat information is useful; geometry and zero heating remain failures

All eight physical atlas anchors remain original source TRAIN families. Only 0348 is response-fit;0001/0318/0333 are train-reference-only;0304/0320/0335/0350 are response-validation-only. None is relabeled as independent source DEV/TEST. Evaluation uses existing common-valid saved physical receivers:256 input-selected fluid points,16 surface angles per original module and all saved material points, with maxima recomputed separately for each endpoint. There are16 heat differences,32 single-geometry differences and8 centered mixed derivatives. Raw mixed four-corner contrasts are kept separately and never pooled into single geometry.

| Response kind | Priority component | Median absolute-error ratio GF/GEE | p90 ratio | GF median error / physical RMS | GEE median error / physical RMS | Near-null excluded |
| --- | --- | --- | --- | --- | --- | --- |
| heat | fluid_fields/temperature | 1.149 | 1.048 | 0.667 | 0.576 | 0 |
| heat | interface/T_surface | 1.030 | 1.028 | 0.532 | 0.526 | 0 |
| heat | solid_temperature/temperature | 1.065 | 1.176 | 0.402 | 0.390 | 0 |
| heat | module_peak_endpoint/temperature | 1.227 | 1.115 | 0.373 | 0.297 | 0 |
| single_geometry | fluid_fields/temperature | 0.905 | 0.958 | 1.191 | 1.204 | 0 |
| single_geometry | interface/T_surface | 0.957 | 0.755 | 1.659 | 2.353 | 0 |
| single_geometry | solid_temperature/temperature | 0.901 | 0.930 | 1.819 | 2.409 | 0 |
| single_geometry | module_peak_endpoint/temperature | 1.151 | 1.124 | 1.381 | 1.316 | 1 |
| mixed_central_derivative | fluid_fields/temperature | 0.870 | 1.199 | 1.617 | 2.058 | 3 |
| mixed_central_derivative | interface/T_surface | 0.718 | 0.952 | 4.926 | 5.230 | 1 |
| mixed_central_derivative | solid_temperature/temperature | 0.722 | 0.947 | 6.255 | 6.862 | 1 |
| mixed_central_derivative | module_peak_endpoint/temperature | 0.795 | 1.121 | 3.638 | 4.253 | 2 |

A zero-response predictor has relative error1 for non-null references. Heat-response medians beat it in all four priority components for both models, but G-F heat-fluid median preservation1.149 and heat-peak1.227 fail1.10; material p90 preservation1.176 also fails1.15. Single-geometry relative median errors exceed1 for all four G-F priority components, despite several favorable GF/GEE ratios, and the peak median ratio1.151 also fails preservation. Mixed derivatives remain poor in absolute relative terms; near-null entries are excluded only from relative metrics and retain absolute errors and reference magnitudes. Full results, tails and original identities for every component, including flow and q proxy, remain in the physical response summaries.

| Cohort | Kind | Records | Fluid-T GF/GEE median error ratio | GF median relative error | GEE median relative error |
| --- | --- | --- | --- | --- | --- |
| fit_train_0348 | heat | 2 | 1.176 | 0.778 | 0.661 |
| fit_train_0348 | single_geometry | 4 | 1.131 | 1.017 | 0.924 |
| fit_train_0348 | mixed_central_derivative | 1 | 1.167 | 1.178 | 1.010 |
| train_reference_only | heat | 6 | 1.205 | 0.687 | 0.570 |
| train_reference_only | single_geometry | 12 | 0.759 | 1.136 | 1.413 |
| train_reference_only | mixed_central_derivative | 3 | 1.772 | 1.617 | 2.058 |
| response_validation_only | heat | 8 | 1.107 | 0.626 | 0.561 |
| response_validation_only | single_geometry | 16 | 0.996 | 1.581 | 1.451 |
| response_validation_only | mixed_central_derivative | 4 | 0.719 | 3.668 | 3.487 |

![Figure 3. physical_responses](../../diagnostics/round12_3_execution_20261024/figures/03_physical_responses.png)

Figure3. Upper panels show pooled-record relative RMSE medians and p90 for the four priority quantities; these spans are descriptive tails, not confidence intervals. Dashed1 is the zero-response predictor. Lower panels compare actual saved reference fluid-temperature changes against both epoch 100 models for response-validation-only0304, with the same units on both axes; the mixed panel is a second position derivative in native temperature/position² units. Heat changes are partly useful, whereas single/mixed geometry behavior remains inadequate. Cohort-specific metrics and near-null absolute errors are retained separately.

| Model | Control family | Zero-heat fluid-T RMS | Surface-T RMS | Material-T RMS | Max absolute surface T |
| --- | --- | --- | --- | --- | --- |
| G-F | 0348 | 1.389 | 6.442 | 5.387 | 12.843 |
| G-F | 0304 | 1.806 | 2.054 | 0.676 | 4.628 |
| G-EE | 0348 | 1.536 | 4.782 | 3.904 | 11.592 |
| G-EE | 0304 | 1.953 | 2.285 | 1.235 | 5.495 |

The analytical homogeneous-boundary zero-heating temperature-rise target is0. Both models fail the bounded check on fit0348 and response-validation-only0304; G-F0348 surface RMS 6.442 and maximum 12.843 native T units are substantial. The objective did not add an analytical loss, heat superposition constraint or heat-independent-flow test. Temperature-only JVP/VJP probes retain every original heat column (10 on0348,3 on0304) and show nonzero sensitivities through complete re-preparation. They are model sensitivity evidence, not physical derivative validation or a replacement for the failed finite geometry references. No full Jacobian, flow-invariance probe or new physical solver was used.

## Executor: preparation and memory savings do not become a native speed claim

| Query rows | GF prepare ms | GEE prepare ms | GF cached ms | GEE cached ms | GF complete core ms | GEE complete core ms |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 8.120 | 9.395 | 3.124 | 3.113 | 11.221 | 12.482 |
| 1024 | 8.120 | 9.364 | 5.668 | 5.596 | 13.746 | 14.961 |
| 8192 | 8.081 | 9.367 | 40.519 | 40.133 | 48.586 | 49.201 |

These are averages of per-scene medians over the four fixed TRAIN scenes, each with a warmup and three repeated measures on the same idle RTX6000 Ada GPU0. Complete native inference additionally includes target-free input binding, native query/stencil union construction, interpolation, physical extraction and role joins: G-F/G-EE times are56.840/58.304ms (M1),58.097/58.317ms (M2),60.577/60.800ms (M10),61.054/61.263ms (M12). Small-Q32 complete-core speed improves about10.1%, but Q8192 complete-core improves only1.25% and the complete native calls only0.34–2.51%. The native ≥10% speed criterion is not met. Both trained functions differ, and this single-seed development comparison does not establish statistical significance.

| Arm | Complete macro s | Native forward s | Native backward s | Response/policy s | Clip/AdamW s | Peak allocated GiB |
| --- | --- | --- | --- | --- | --- | --- |
| G-F | 2.0572 | 1.2281 | 0.6589 | 0.1392 | 0.0021 | 1.574 |
| G-EE | 2.1411 | 1.2532 | 0.6827 | 0.1587 | 0.0030 | 2.110 |

The disposable timing macro uses the exact checkpoint’s named AdamW state,48 primary cases in two 24-case microbatches, native backward, the every-fourth-update response gradient, clipping and AdamW. One warmup and three recorded engineering steps per arm are restored afterward; they are not scientific training epochs or retained trained checkpoints. G-F’s median2.0572s is3.92% lower than G-EE2.1411s, so no ≥10% training-speed gain is claimed. Its measured peak1.574 GiB versus2.110GiB is25.40% lower: a real memory benefit from deleting E→E in this implementation, accompanied by the failed fidelity bands. Concurrent first 100 engine times (865.0/845.7s on separate GPUs) are exposure/runtime records and are not isolated causal speed measurements.

Actual counters distinguish allocated/padded operations from live positive-support relationships. On M12, two-layer encoder score pairs are38,016 in G-F versus332,928 in G-EE (88.58% fewer); the ordinary query reader still allocates eight group slots per query in both. At 8192 query rows it computes262,144 head/group scores and gathers65,536 group-value slots, with zero original-E and zero M-value query expansion. These trained controls therefore show no learned member-read reduction relative to G-F. Persistent prepared tensor storage is829,141–1,016,768bytes across the 24 organizer scenes for both arms. Both layers’ prepared group vectors are charged. Earlier training-history ME/MG/query counters recorded live masked support and omitted padding; those histories are preserved, while the corrected endpoint counters explicitly retain allocated and live quantities. No prediction tensor or scientific metric was changed by this accounting correction.

Substage timings are available for complete preparation, cached decode, complete core/native calls, native forward/backward, response/policy callback and clipping/AdamW. Individual nomination/pooling/index/lookup neural substages were not separately profiled; singleton controls have no nomination or frontier VJP work. No calibrated adaptive arithmetic/runtime work price or executor gain from G-H is asserted. Restricted search and zero ordinary member expansion are measured computational properties, not proof that a learned compact organization retained physical utility.

## Inverse, decision and next work

No inverse head, sample trail, ranking campaign or physical design was run. Group diagrams, smooth original-source derivatives and bounded candidate diversity do not establish valid designs, physical feasibility or regret. Predictor gains are limited to selected native quantities and useful heat-response medians; predictor misses include surface preservation, q-proxy usefulness, geometry responses and zero heating. Organizer gains are executed/reused singleton values and useful high-M early relay; organizer misses are untested adaptive aggregation and weak final-address utility. Executor gain is measured training memory; complete native/training speed targets are missed. These axes remain separate.

A subsequent explicitly authorized revision should first resolve the observed local/surface communication and source-law failures and decide whether the weak final exchange should be simplified. Preserve the matched controls and physical references when testing one change. The present data do not justify training G-H merely to obtain a smaller K, forcing leaves/roots, introducing a target K, adding a decoder rescue branch, extending beyond100, spending the recovery reserve or starting full-data training. This connected exclusive tree remains a restricted organizational hypothesis; neither this stop nor an untrained root choice settles unrestricted hypergraph organization.

## Delivery, verification and storage

The final focused suite passed 36 tests in 16.14s, including core aggregation/reuse, padded/empty donor handling, M1/M2, legal keyed frontiers, tiny-tree task/work enumeration, policy baseline/retry state, query bounds/chunking, adapter and maintained runner/native/response regressions; new files also pass Ruff. Actual high-M native qualification, both400-update histories, strict checkpoint reconstruction, saved-checkpoint native/reference evaluation and complete execution timings provide the runtime evidence. Existing TEST separation, DATA2 bindings, checkpoint/schema/RNG checks, named optimizer ownership, OOM retry machinery and the pre-push artifact hook are retained. No new cryptographic hashes, contract-freeze service, baseline snapshots or readiness gates were introduced; existing dataset/sampling identities are reused. The transient wrapper-print failure and the G-H GPU native reload discrepancy remain recorded rather than erased.

Conservatively charged GPU use is 0.606 GPU-hours against 24 initial/36 overall; measured training is 0.475 GPU-hours. Native wrapper time is charged at three minutes per arm because its separate elapsed print failed; response and execution passes are measured. The 12-hour reserve is unused. Retained round/run payloads are 130,861,847 bytes (<4 GiB); the declared estimated peak temporary budget was256 MiB under the512 MiB limit. Cleanup removed 187 files/links and 84 directories, totaling 13,796,499 bytes, including 18 superseded response PNGs,4 PDF QA rasters and exact round-owned test/bytecode scratch. Early tests used standard /tmp; their exact round-created trees were removed and the final suite used explicit DATA2 temporary storage. No scientific checkpoints, numerical/reference arrays, historical runs or protected migration backups were removed. The original cleanup inventory and completion receipt retain the temporary helper failure/recovery and confirm all allowlisted entries absent and protected retained metadata unchanged. Remaining observed capacity after cleanup is /data2 3,218,377,691,136 bytes (2.927TiB) and /home 84,089,159,680 bytes (78.314 GiB); DATA2 is mounted and all three declared bindings remain intact. Other jobs may change filesystem free space independently.

The durable implementation is [interaction_groups](../../src/honf_forward_core/interaction_groups/), the distinct [Thermal adapter](../../Case_ThermalChannel/src/channelthermal/interaction_groups.py), [task](../../Case_ThermalChannel/src/channelthermal/training/interaction_group_task.py), [execution evaluator](../../src/honf_runtime/interaction_group_evaluation.py) and [recipe guide](../guides/Thermal_Interaction_Group_Reuse_Recipe.md). The three ordinary versioned research configurations are linked from the guide; G-H remains a provisional untrained recipe requiring the unperformed adaptive calibration/qualification before any separately selected training stage. Durable source/tests/config/docs are committed and pushed on the current non-default branch after whole-outgoing-history audit. Generated data, figures, temporary helpers, manifests and receipts remain local on DATA2.
