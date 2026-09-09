# HONF group-reader recovery

Completed continuation (2026-09-09): see the [exact epoch-5,000 comparison](HONF_Epoch5000_Comparison_Report.md). Dense leads overall reconstruction; Reader retains numerical group activity but its relative field deficit grows. The original 500-epoch recovery findings remain unchanged.

Follow-up (2026-09-09): the [exact epoch-2,500 comparison](HONF_Epoch2500_Comparison_Report.md) updates the later-training ranking and thermal conclusions. The measurements and causal findings below remain the epoch-500 closeout evidence.

## Evidence status

This study tests whether separating geometric group availability from learned relative attention recovers useful module coupling and field reconstruction. One fresh 500-epoch run on physical GPU 0 completed. On the existing 90-case development holdout, pooled normalized fluid MSE falls from 0.028678 to 0.018054 (37.0%) and relative L2 from 0.176003 to 0.139648 (20.7%) versus Run 1802; 85 of 90 paired cases improve. The candidate remains behind Dense 1804 (0.098741 L2) and Legacy 1401 (0.117148 L2). Activation alone is not treated as success; phase-specific ground-truth tests below assess usefulness.

The baseline source was branch `agent/honf-core-next` at `116cb22`. Existing legitimate work and run directories are preserved. The scientific change retains `forward_architecture="sparse_interface_honf"` and adds `interface_model.group_read_mode="geometry_envelope_attention"`; absent settings retain historical `null_softmax`. All learned parameter names, supports, membership and message networks, coarse/local paths, physical coupling, losses, optimizer settings, and training chunk size remain unchanged.

For geometric weight `g(q,e)=a(e)B(e,q)`, the new context is `G(q) Σ π(q,e)v(e)`, with `G(q)=Σg(q,e)` and `π(q,e)∝g(q,e)exp(logit(q,e))`. Unsupported receivers return zero. The relative attention is invariant to a common finite logit shift; its effective mass equals geometric availability, which is a construction identity rather than evidence of physical usefulness.

## Historical evidence corrections

The Stage-3 `field_main_zero` hook suppressed every decoder call, including P1 outside-temperature feedback. The original results remain in [the completed study](HONF_Interface_Study_Report.md); they cannot be attributed solely to final P2 field reads. This study separates P0-only, P0-plus-P1 interface feedback, and P2-only interventions, with intervened-minus-normal ground-truth error changes as well as prediction discrepancies.

The earlier 47.2% latent-versus-dense excess concerns pooled relative L2, not pooled MSE. The measured historical tables are unchanged. Existing aggregate `backend` gradient norms include the coarse/local paths; the new observations retain that total and separately report group preparation, group reception, coarse, and local norms.

The previous synthetic `query_group_read_value_count` counted exported local slots, including `-1` padding. For the largest shape the stored mean degree 15.89599609375 implies 4,167,040 retained incidences, versus 4,194,304 available slots. The updated timing tool reports both quantities; historical timing/arrays remain intact. The bounded reader audit also explicitly masks padded physical receivers, unlike the older bulk port-magnitude proxies. Ground-truth physical error definitions are unchanged.

A diagnostic-only correction in `2102d8e` computes historical conditional attention and log-normalizers in positive-weight FP64 log space when detailed maps are requested. The earlier float32 tiny-clamped diagnostic ratio inflated some epoch-500 case-0632 conditional norms (maximum 61.44 despite value norms near 16.27). Historical context, mass, logits, and training arithmetic remain unchanged. The original audit is retained; the bounded 3-checkpoint × 8-case replay has a separate artifact and supplies the corrected conditional norms below.

## Stored Run-1802 diagnosis

Epochs 10, 50, and 500 were replayed on anchors 0273, 0653, 0298, and 0302 and deterministic training cases 0003, 0095, 0340, and 0632. The same four training cases supplied one canonical 1,024-query backward batch per checkpoint with zero parameter update. P0/P1 statistics exclude inactive module padding; per-incidence statistics exclude empty routing slots. Within-receiver logit spread is computed after centering. Every audited physical receiver had positive support, so supported-only and all-active-receiver means coincide for this sample. Full distributions, query/key norms, dot/bias terms, null mass, and branch norms are in `diagnosis/run1802_checkpoint_audit_stable_diagnostic_replay.json` under the study root.

These are equal-case means for four cases in each split. `G` is geometric availability; value and conditional-mixture columns are vector norms before whole-branch attenuation. P1 closely follows P0 and remains separately recorded in the artifact.

| Epoch | Split | Phase | G | Non-null mass | Mean logit | Within-receiver logit SD | Value norm | Conditional mixture norm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 10 | Holdout | P0 | 0.9437 | 3.31e-7 | -27.83 | 0.101 | 12.96 | 12.96 |
| 10 | Holdout | P2 | 0.6586 | 1.58e-7 | -27.81 | 0.096 | 12.96 | 12.96 |
| 10 | Training | P0 | 0.9239 | 4.13e-4 | -21.27 | 0.094 | 13.05 | 13.05 |
| 10 | Training | P2 | 0.5996 | 1.81e-4 | -21.25 | 0.097 | 13.05 | 13.05 |
| 50 | Holdout | P0 | 0.9437 | 2.45e-6 | -16.03 | 0.097 | 13.25 | 13.25 |
| 50 | Holdout | P2 | 0.6586 | 1.44e-6 | -15.96 | 0.097 | 13.25 | 13.25 |
| 50 | Training | P0 | 0.9239 | 2.52e-5 | -14.85 | 0.087 | 13.24 | 13.24 |
| 50 | Training | P2 | 0.5996 | 9.89e-6 | -14.71 | 0.094 | 13.24 | 13.24 |
| 500 | Holdout | P0 | 0.9437 | 1.07e-29 | -70.39 | 0.144 | 16.38 | 16.34 |
| 500 | Holdout | P2 | 0.6586 | 4.30e-28 | -68.88 | 0.141 | 16.38 | 16.35 |
| 500 | Training | P0 | 0.9239 | 8.76e-32 | -76.59 | 0.172 | 16.31 | 16.29 |
| 500 | Training | P2 | 0.5996 | 8.65e-31 | -75.32 | 0.178 | 16.31 | 16.29 |

Historical null mass is `1 − non-null mass`: it is already 0.999587 on average for training P0 at epoch 10 and rounds to 1 at epoch 500, while geometric availability remains high. Full null-mass distributions are retained in the audit. Geometric availability is unchanged across checkpoints and group values do not vanish. At epoch 500 the holdout P0 logit averages -70.39, mostly from the query/key dot product (-69.84), with a much smaller learned relative bias (-0.55). Mean P0 query/key norms are 25.23/46.98. The P2 dot/bias terms average -68.34/-0.54. This is strong evidence of whole-read suppression with retained geometry and finite values; it is not proof that those values encode useful interactions or that normalization is the uniquely causal optimization failure.

| Epoch | Group preparation gradient | Group receiver gradient | Coarse gradient | Local gradient |
|---|---:|---:|---:|---:|
| 10 | 3.332e-3 | 4.066e-3 | 4.327 | 4.533 |
| 50 | 1.240e-4 | 2.856e-4 | 1.263 | 0.843 |
| 500 | 2.307e-30 | 3.321e-30 | 0.767 | 0.365 |

These are pre-clip FP64 norm reductions from one fixed canonical batch per checkpoint. All recorded parameter updates are zero. The historical aggregate backend norm remains available but is not used as evidence of group learning.

The prepared-state replay on 0273 at epoch 10 uses the same group states under both readers and a common -60 logit shift. The corrected context changes by at most `1.19e-6` under that shift in float32 execution, whereas the old read is suppressed. Reader replacement itself changes the main context by up to 2.63. This is a mechanistic replay, not trained corrected-model accuracy evidence.

## Implementation and physical smoke

Commit `35abb68` adds the reader and bounded execution controls; `be8a7a0` completes opt-in exports and endpoint output controls; `8b6a351` adds phase-scored diagnostics and compact plots; `2102d8e` corrects only the historical detailed conditional statistics under underflow. Relative normalization uses supported-only `log(g)+logit`, with segmented maxima and scalar FP64 accumulation for FP32 inputs; vector messages remain in the model dtype. No epsilon floor detaches or replaces the geometric weights. Exact-zero unsupported receivers have zero context. The historical branch retains its original arithmetic and state-dict parameters; replay against that arithmetic had maximum absolute difference zero.

The scalar precision choice protects coordinate gradients near vanishing support. In an executed two-incidence check with logits `[-60,-59]`, weights `[s,2s]`, and values `[1,3]`, at `s=1e-44` the FP32-only derivative was approximately `[1.83214,3.11785]`, versus `[1.90193,3.08295]` with scalar FP64 normalization. Focused tests cover common shifts, unsupported receivers, tiny positive occupancy, coordinate/support derivatives, historical compatibility, and full physical-wrapper integration. This is a numerical implementation detail, not an added learned mechanism.

Two disposable, real physical batches completed forward, canonical loss, backward, clipping, and AdamW update on GPU 0 before run allocation. Both used 48 cases and 1,024 field queries per case, with frozen Stage A. These smoke observations establish executable gradients and updates; they do not establish generalization.

| Batch | Modules/case | Total loss | Field MSE | Wall time (s) | Peak allocated MiB | Group-prepare gradient | Group-receiver gradient | Group-prepare update | Group-receiver update |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Small | 1 | 10.8373 | 1.84884 | 1.342 | 6,728 | 44.3301 | 0.012683 | 0.238039 | 0.077448 |
| Large | 12 | 15.8026 | 3.11087 | 1.920 | 28,925 | 28.9056 | 0.004878 | 0.152262 | 0.040922 |

Losses, parameters, and relevant branch gradients were finite. Unused legacy diagnostic columns retain their established NaN sentinels. The smoke saved no model or optimizer and did not reserve a managed run.

## The one managed training run

The existing allocator created `Run_1805_20260908_002326_sparse_interface_geometry_read` from scratch with candidate profile `src/config_core/forward/sparse_interface_geometry_read_context.json`. Run 1802 was not resumed. The materialized trainable parameter count is 3,778,064. The run keeps seed 0, width 256/message width 128, support spacing 4, coarse 8/1, local radius 2.5, environment grid 24×8, receiver chunks 128, AdamW learning rate 3e-4 and weight decay 1e-5, clipping 1, no AMP, dropout 0, and all inherited losses and schedules.

A direct recursive comparison of the two existing `config_resolved.json` files found only the added reader setting (including its effective-model mirror) and the expected run ID/name changes. No training, physical-coupling, data, support, or optimizer setting differed.

The process started at 2026-09-08 04:23:26 UTC on physical GPU 0. Existing `metrics.csv`, checkpoints, optimizer inventory, and the redirected launch log provide the execution record. The exact epoch-500 checkpoint was written at 06:59:14 UTC and the process exited successfully. There are exactly 500 metric rows. Recorded training and validation wall times are 8,335.43 s and 935.10 s, totaling 9,270.53 s (2.575 h); launch-to-checkpoint elapsed time includes setup and checkpoint overhead. Peak allocated CUDA memory was 29,278.47 MiB (28.59 GiB). No failure, restart, additional managed trial, or continuation occurred.

The training CSV's read summaries are means over sampled P2 field receivers, including any unsupported receivers. Its sampled gradient norms come from the evolving training batches. They are not a matched repetition of the four-case stored-checkpoint audit. At epoch 500, training field MSE was 0.026227, validation field MSE 0.027529, and total loss 0.043510. Group preparation/receiver gradient norms were 0.40182/0.01053 and corresponding update norms 0.04306/0.01925; coarse/local gradient norms were 0.27734/0.32792. Mean sampled group-read mass and G were both 0.77834, with group-context magnitude fraction 0.38628. The endpoint evaluation uses epoch 500, not the best-validation checkpoint (best validation MSE 0.025244).

![Stored collapse and candidate training recovery](../../diagnostics/generated/interface_operator_study/group_reader_recovery/figures/reader_recovery.png)

The historical panels use the fixed four-case audit; candidate history panels use evolving sampled field receivers. Their absolute averages are not a matched batch comparison. Nonzero mass in the candidate is an identity of the formula; sustained group-specific gradients and updates establish learning activity, while the tests below address utility.

## Exact epoch-500 development-holdout evaluation

The candidate was evaluated once on all 90 established development cases with checkpoint-owned normalization and predicted physical port conditions. Existing comparator tables were reused in place. Reduction reconciled identical case IDs, per-case fluid-value counts and target sums of squares, and pooled raw sums for every model: 3,496,800 fluid channel values and normalized target squared sum 3,237,304.06052. Equal-case statistics and pooled statistics are labeled separately. Existing near-interface and far-fluid masks use nearest module surface distances of 0–0.25 and at least 1.0 in dataset coordinate units; these are unchanged absolute-distance definitions, separate from the radius-normalized layout strata. This is one seed on a repeatedly used development holdout, not a new untouched test set.

### Matched field reconstruction

| Metric | Legacy 1401 | Dense 1804 | Latent 1801 | Sparse 1802 | Reader 1805 |
|---|---:|---:|---:|---:|---:|
| Pooled normalized fluid MSE | 0.012705 | 0.009026 | 0.019554 | 0.028678 | 0.018054 |
| Pooled normalized fluid relative L2 | 0.117148 | 0.098741 | 0.145333 | 0.176003 | 0.139648 |
| Equal-case fluid relative L2 median | 0.110861 | 0.096741 | 0.135462 | 0.164947 | 0.138949 |
| Equal-case fluid relative L2 p95 | 0.140883 | 0.112534 | 0.203446 | 0.238872 | 0.161313 |
| Near-interface pooled normalized relative L2 | 0.117314 | 0.097592 | 0.102868 | 0.120664 | 0.107961 |
| Far-fluid pooled normalized relative L2 | 0.123058 | 0.099403 | 0.156809 | 0.191144 | 0.146243 |

### Physical per-channel fluid relative L2

| Metric | Legacy 1401 | Dense 1804 | Latent 1801 | Sparse 1802 | Reader 1805 |
|---|---:|---:|---:|---:|---:|
| u | 0.045986 | 0.029362 | 0.047478 | 0.057864 | 0.041773 |
| v | 0.077416 | 0.103761 | 0.139297 | 0.136233 | 0.124751 |
| p | 0.096772 | 0.088976 | 0.171671 | 0.208227 | 0.162806 |
| omega | 0.155603 | 0.113666 | 0.159485 | 0.208558 | 0.162415 |
| temperature | 0.115753 | 0.090229 | 0.120301 | 0.156272 | 0.118980 |

### Physical port, interface, and internal relative L2

| Metric | Legacy 1401 | Dense 1804 | Latent 1801 | Sparse 1802 | Reader 1805 |
|---|---:|---:|---:|---:|---:|
| Final outside temperature | 0.088037 | 0.068922 | 0.107010 | 0.132273 | 0.093577 |
| Final effective heat-transfer coefficient | 0.042249 | 0.041631 | 0.049036 | 0.050035 | 0.050515 |
| Internal temperature | 0.071717 | 0.069485 | 0.058140 | 0.071134 | 0.046182 |
| Surface temperature | 0.091510 | 0.087244 | 0.077663 | 0.093459 | 0.067883 |
| Normal heat flux | 0.216907 | 0.220073 | 0.203949 | 0.218099 | 0.217576 |

### Existing KPIs: mean absolute error in dataset-native physical units

| Metric | Legacy 1401 | Dense 1804 | Latent 1801 | Sparse 1802 | Reader 1805 |
|---|---:|---:|---:|---:|---:|
| Inlet-minus-outlet pressure drop | 0.009129 | 0.007186 | 0.013202 | 0.011123 | 0.009787 |
| Mean outlet temperature | 0.915529 | 1.019576 | 0.524360 | 0.543761 | 0.619708 |
| Mean active-module temperature | 0.459401 | 0.605719 | 0.195139 | 0.319927 | 0.189009 |

### Paired case differences

Differences below are candidate minus comparator in equal-case normalized fluid relative L2; negative is an improvement.

| Comparator | Improved / 90 | Mean delta | Median delta | p05 delta | p95 delta |
|---|---:|---:|---:|---:|---:|
| Legacy 1401 | 2 | 0.023435 | 0.022938 | 0.003050 | 0.044565 |
| Dense 1804 | 0 | 0.040778 | 0.038395 | 0.024191 | 0.063319 |
| Latent 1801 | 41 | -0.003369 | 0.002482 | -0.047043 | 0.019850 |
| Sparse 1802 | 85 | -0.033603 | -0.029736 | -0.085696 | 0.000307 |

### Predefined strata

Mean case-level normalized fluid relative L2. Module-count strata are the existing exact held-out counts. Spacing and wall thresholds use module-radius units; heat strata use the existing active-power coefficient of variation.

| Stratum | Cases | Legacy 1401 | Dense 1804 | Latent 1801 | Sparse 1802 | Reader 1805 |
|---|---:|---:|---:|---:|---:|---:|
| module_count_stratum: 10 | 15 | 0.136756 | 0.107615 | 0.141589 | 0.175437 | 0.145816 |
| module_count_stratum: 3 | 25 | 0.096632 | 0.082137 | 0.115667 | 0.141813 | 0.120367 |
| module_count_stratum: 5 | 25 | 0.110052 | 0.095094 | 0.141554 | 0.171530 | 0.135025 |
| module_count_stratum: 7 | 25 | 0.119678 | 0.104184 | 0.162737 | 0.195148 | 0.149902 |
| spacing_stratum: crowded_<1r | 45 | 0.123651 | 0.103674 | 0.149085 | 0.180216 | 0.145661 |
| spacing_stratum: intermediate_1r_to_2p5r | 30 | 0.106232 | 0.091877 | 0.137345 | 0.167354 | 0.132250 |
| spacing_stratum: separated_>=2p5r | 15 | 0.097277 | 0.081863 | 0.119572 | 0.147566 | 0.119821 |
| wall_proximity_stratum: interior_>=2p5r | 12 | 0.098533 | 0.083769 | 0.121094 | 0.143995 | 0.120668 |
| wall_proximity_stratum: middle_1p5r_to_2p5r | 50 | 0.115970 | 0.098139 | 0.142727 | 0.172968 | 0.140059 |
| wall_proximity_stratum: near_<1p5r | 28 | 0.115340 | 0.097765 | 0.144047 | 0.177410 | 0.138164 |
| heating_heterogeneity_stratum: high_cv_>=0p35 | 20 | 0.118133 | 0.100555 | 0.159085 | 0.193464 | 0.146259 |
| heating_heterogeneity_stratum: low_cv_<0p25 | 28 | 0.105161 | 0.091130 | 0.129501 | 0.157204 | 0.130213 |
| heating_heterogeneity_stratum: medium_cv_0p25_to_0p35 | 42 | 0.116743 | 0.097306 | 0.138454 | 0.168401 | 0.136868 |

All five physical field-channel errors improve over Run 1802. Internal and surface temperature relative L2 are the lowest among these five matched endpoints. The recovery is incomplete: relative field L2 remains 41.4% above Dense and 19.2% above Legacy. The candidate beats Latent in pooled L2 but only 41/90 individual cases; its better high-error tail explains that distinction. All 13 predefined strata improve in mean field L2 over the old sparse reader. That does not imply every physical metric improves in every stratum: for three-module cases, mean internal-temperature MAE increases from 0.321655 to 0.339997.

Against Run 1802, final heat-transfer-coefficient L2 increases slightly (0.05004 to 0.05051), normal heat-flux L2 barely changes, and outlet-temperature MAE increases from 0.54376 to 0.61971 (14.0%). These tradeoffs accompany stronger field, surface/internal-temperature, pressure-drop, and mean-module-temperature results. No incompatible physical units are combined into one score.

| Separate maturity reference | Epoch | Pooled normalized fluid MSE | Pooled normalized fluid relative L2 |
|---|---:|---:|---:|
| Legacy Run 1401 mature best-field | 4585 | 0.000954 | 0.03210 |

This maturity result is not an equal-budget competitor and supplies no evidence about a hypothetical continued Run 1805.

![Candidate anchor geometry, read, and physical error](../../diagnostics/generated/interface_operator_study/group_reader_recovery/figures/anchor_reader_maps.png)

G and effective mass coincide by construction. Conditional attention and context magnitude describe the learned computation; temperature error is measured against existing case reference data on fluid cells. These maps alone do not establish useful coupling.

## Execution measurements, separate from reader usefulness

Detailed routing exports are opt-in; inexpensive training summaries remain available. Inference can use one tested larger receiver chunk, 2,048, without changing checkpoint configuration or training's 128. A layout-scoped prepared lookup reuses integer support search metadata; continuous weights are recomputed so coordinate gradients remain live. The detailed mode includes richer audit statistics than the historical implementation, so its timing is labeled explicitly.

The completed two-anchor timings use GPU 0, 8,192 queries, five warmups, and ten repetitions. Each entry is full-forward median milliseconds, with model preparation included. These are execution comparisons of existing trained models, independent of the new reader's predictive result.

| Model | Anchor | 128 summary | 2,048 summary | 128 detailed | 2,048 detailed |
|---|---|---:|---:|---:|---:|
| Dense 1804 | 0273 | 157.19 | 33.70 | 154.86 | 33.61 |
| Dense 1804 | 0653 | 146.17 | 33.79 | 149.50 | 34.38 |
| Latent 1801 | 0273 | 126.12 | 29.20 | 125.86 | 28.44 |
| Latent 1801 | 0653 | 125.37 | 29.03 | 126.42 | 28.48 |
| Sparse 1802 | 0273 | 230.72 | 38.46 | 329.75 | 48.12 |
| Sparse 1802 | 0653 | 225.09 | 37.65 | 326.52 | 49.30 |
| Reader 1805 | 0273 | 240.73 | 39.31 | 312.31 | 45.18 |
| Reader 1805 | 0653 | 242.78 | 39.37 | 312.71 | 46.33 |

Common 2,048-summary phase timings (milliseconds) and full-forward peak memory (MiB). Physical preparation includes the established one-query output; prepared decoding excludes it. Separately timed phases need not sum exactly to the full-call median.

| Model | Anchor | Preparation + 1 query | Prepared 8,192-query decode | Full forward | Full peak allocated | Full peak reserved |
|---|---|---:|---:|---:|---:|---:|
| Dense 1804 | 0273 | 23.18 | 13.09 | 33.70 | 505.33 | 638.00 |
| Dense 1804 | 0653 | 22.51 | 13.51 | 33.79 | 487.14 | 620.00 |
| Latent 1801 | 0273 | 24.71 | 6.63 | 29.20 | 112.09 | 138.00 |
| Latent 1801 | 0653 | 23.29 | 6.82 | 29.03 | 112.54 | 138.00 |
| Sparse 1802 | 0273 | 29.78 | 12.04 | 38.46 | 137.29 | 240.00 |
| Sparse 1802 | 0653 | 28.51 | 12.01 | 37.65 | 138.83 | 256.00 |
| Reader 1805 | 0273 | 29.08 | 13.05 | 39.31 | 134.66 | 236.00 |
| Reader 1805 | 0653 | 29.53 | 13.16 | 39.37 | 138.82 | 260.00 |

The candidate's summary-mode chunk increase gives 6.12× and 6.17× full-forward speedups on 0273/0653. Its observed peak allocation rises from 98.52/80.26 MiB at chunk 128 to 134.66/138.82 MiB at 2,048. The unchanged old sparse reader also improves by 6.00×/5.98×, showing that most of this speed gain is an execution improvement independent of recovery. At 2,048, omitting detailed maps saves about 13%/15% of candidate full-forward time. These measurements evaluate the combined execution changes; they do not isolate a speed contribution from prepared-search reuse alone.

The original completed study's sparse 0273 full-forward measurement was 234.47 ms with its historical diagnostics. The updated detailed implementation exports additional audit statistics and takes 329.75 ms at chunk 128 for Run 1802; comparing that to 38.46 ms in summary/2,048 mode conflates richer diagnostics with chunking. The within-protocol table above is the appropriate execution comparison. Dense anchor rows reuse the completed bounded benchmark. Latent rows were refreshed after preparation-map propagation was corrected, and old sparse rows were refreshed after the diagnostic-only conditional precision correction. No historical timing table was rewritten. The new endpoint anchor profiler ran once, then repeated the same protocol after its untimed agreement probe was corrected; the second execution replaced its provisional JSON/CSV and supplies all anchor values reported here. The largest-shape endpoint benchmark ran once. These were measurement corrections, not additional training or a chunk sweep.

The largest existing synthetic shape is `(M,E,Q)=(128,3072,262144)`, with 480 retained groups, 2,775 module-group incidences, 48,872 environment-group incidences, and 4,167,040 actual query-group incidences. With configured 128 chunks and detailed routing, the pre-endpoint replay gave medians 9.048 s sparse, 7.097 s dense, and 3.757 s latent (one warmup, three repetitions). The completed common 2,048-summary comparison follows.

| Model | Median (s) | p95 (s) | Peak allocated MiB | Observed peak reserved MiB |
|---|---:|---:|---:|---:|
| dense1804 | 5.3331 | 5.3484 | 6995.51 | 9038.00 |
| latent1801 | 0.2740 | 0.2859 | 250.22 | 9046.00 |
| sparse1802 | 0.5011 | 0.5092 | 255.38 | 9046.00 |
| reader1805 | 0.5435 | 0.5473 | 255.63 | 9046.00 |

This test uses synthetic geometry with the real support builder and physical model wrapper, preparing once and decoding 262,144 queries in outer batches of 32,768 with receiver chunks 2,048. Each timed call rebuilds its preparation; it is not a PDE solve. All rows use summary mode, one warmup and three repetitions. Latent is fastest on this shape; the new sparse reader is about 8.5% slower than the unchanged old reader under this common execution mode and uses far less peak allocated memory than dense.

The reported reserved-memory high-water marks include allocator reuse across sequential model measurements: roughly 9,046 MiB reserved after dense persists for later models, despite their much lower live allocations. It is not a claim that each sparse model requires 9 GiB. Summary mode omits query routing arrays, so its zero-filled query-incidence export fields mean not exported; the 4,167,040 incidence count comes from the detailed run of the same synthetic geometry. The expected 4,194,304 slot capacity includes padding. No physical accuracy or superior mathematical complexity follows from these execution observations.


### Accuracy and measured cost

![Endpoint accuracy versus measured inference and training cost](../../diagnostics/generated/interface_operator_study/group_reader_recovery/figures/accuracy_cost.png)

The inference axis uses the mean of the full-forward medians for anchors 0273 and 0653, each with 8,192 queries, chunk 2,048, summary mode. It does not mix in bulk-evaluation timings from different protocols. Accuracy is the exact-500 pooled normalized fluid relative L2 on all 90 development cases. The training axis sums existing train and validation wall-time columns through epoch 500; it is recorded operational cost, not a controlled hardware-normalized complexity claim.

| Model | Train wall (s) | Validation wall (s) | Total through epoch 500 (s) |
|---|---:|---:|---:|
| Dense 1804 | 8,912.34 | 976.09 | 9,888.42 |
| Latent 1801 | 8,057.86 | 891.80 | 8,949.67 |
| Sparse 1802 | 8,399.80 | 932.90 | 9,332.69 |
| Reader 1805 | 8,335.43 | 935.10 | 9,270.53 |

### Numerical agreement of runtime modes

Focused chunk/map integration tests pass, but actual trained checkpoints are not elementwise identical under the inherited tolerances. A bounded check used three independent forwards per mode on both anchors for Latent 1801, Sparse 1802, and Reader 1805, comparing summary/detailed chunk 2,048 with a separate summary-128 reference. It retained the existing chunk tolerances (`rtol=2e-5`, `atol=2e-6`) and recorded every failure. All 36 changed-mode comparisons failed at least the interface tensor's strict elementwise check; some field tensors also failed. Same-128 repeated forwards can also fail. The profiler was corrected to compare independent outputs rather than an object with itself and to retain only public reference tensors on CPU. The final timing artifact comes from the corrected protocol; the separate repeat artifact supplies a three-repeat numerical comparison.

| Model | Changed-mode maximum field difference | Maximum relative field difference norm | Changed-mode maximum interface difference | Maximum relative interface difference norm | Same-128 repeat maximum interface difference |
|---|---:|---:|---:|---:|---:|
| Latent 1801 | 9.78e-6 | 3.06e-7 | 8.82e-6 | 1.82e-6 | 4.89e-6 |
| Sparse 1802 | 8.58e-6 | 3.33e-7 | 9.19e-6 | 1.65e-6 | 7.81e-6 |
| Reader 1805 | 9.54e-6 | 2.91e-7 | 1.18e-5 | 1.92e-6 | 1.11e-5 |

The norm ratios use the whole difference tensor divided by the reference tensor norm; their maxima and the elementwise maxima may come from different repeats. Values are in each tensor's model coordinates, not ground-truth error changes. These small changes are consistent with float32 accumulation/repeat variation, but the elementwise failures are not hidden or reclassified as passes. The larger chunk remains an opt-in runtime option; training and the reported 90-case accuracy use the unchanged configured chunk 128. No new acceptance threshold was introduced.

## Phase-specific usefulness and derivatives at epoch 500

The new phase scopes are P0 only, nested P0+P1, and final P2 only. The difference between nested P0+P1 and P0 is an incremental feedback effect **conditional on P0 suppression**, not a P1-only intervention with normal P0. Each scope recomputes the downstream physical sequence and records canonical ground-truth errors. Historical Run-1802 epoch-500 effects were zero on 0273, 0298, and 0302; on 0653, mean absolute normalized field prediction changes were approximately 2.91e-12 (P0), 1.50e-7 (P0+P1), and 1.65e-7 (P2), with P2 maximum 5.72e-6.

### Four-anchor ground-truth reliance tests

All deltas below are intervened minus normal relative L2. Positive means removal worsens the error. Fluid/near/far are normalized; internal and outside temperatures use denormalized physical values. Normal rows give absolute errors. These are reliance tests of the trained checkpoint, not retraining ablations.

| Anchor | Scope | Fluid | Near | Far | Internal T | Final outside T |
|---|---|---:|---:|---:|---:|---:|
| 0273 | Normal error | 0.106039 | 0.065092 | 0.115005 | 0.034145 | 0.119008 |
| 0273 | P0 delta | +0.000314 | +0.001212 | +0.000215 | +0.015884 | +0.021605 |
| 0273 | P0+P1 delta | +0.019459 | +0.077585 | -0.000277 | +0.263837 | +0.274138 |
| 0273 | P2 delta | +0.283212 | +0.477684 | +0.119575 | -0.000000 | +0.000000 |
| 0653 | Normal error | 0.122537 | 0.097981 | 0.122903 | 0.032215 | 0.081409 |
| 0653 | P0 delta | +0.001762 | +0.006281 | -0.000091 | +0.028492 | +0.013452 |
| 0653 | P0+P1 delta | +0.017343 | +0.050439 | -0.002529 | +0.318024 | +0.330574 |
| 0653 | P2 delta | +0.344036 | +0.476897 | +0.180535 | -0.000000 | +0.000000 |
| 0298 | Normal error | 0.182174 | 0.139100 | 0.220371 | 0.112314 | 0.153154 |
| 0298 | P0 delta | +0.000152 | +0.002384 | -0.001875 | +0.010786 | +0.014201 |
| 0298 | P0+P1 delta | +0.007889 | +0.037543 | -0.012610 | +0.227329 | +0.239474 |
| 0298 | P2 delta | +0.287147 | +0.407597 | +0.151212 | +0.000000 | -0.000000 |
| 0302 | Normal error | 0.157733 | 0.113669 | 0.190075 | 0.064919 | 0.138218 |
| 0302 | P0 delta | +0.000897 | +0.001703 | +0.000416 | -0.000823 | +0.004968 |
| 0302 | P0+P1 delta | +0.006407 | +0.028225 | -0.007566 | +0.153193 | +0.133247 |
| 0302 | P2 delta | +0.326544 | +0.449473 | +0.191855 | +0.000000 | +0.000000 |

### Equal-anchor phase deltas by physical output

| Scope | u | v | p | omega | Temperature | Surface T | Heat flux | Final h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | +0.000094 | +0.000090 | +0.000433 | +0.002633 | -0.000086 | +0.018918 | +0.008853 | +0.009863 |
| P0+P1 | +0.002982 | +0.001541 | -0.005521 | +0.042672 | +0.004137 | +0.304202 | +0.041465 | +0.004634 |
| P2 | +0.090287 | +0.149175 | +0.366056 | +0.476329 | +0.237482 | +0.000000 | +0.000000 | +0.000000 |

### Prediction changes in model output coordinates

Mean of per-anchor mean absolute differences in checkpoint-normalized field/interface/internal outputs; these are prediction discrepancies, not physical error scores. Raw multi-column port-token differences are retained in the JSON as debug proxies and are not combined into a physical metric here. The physical tables above exclude inactive receivers; these historical-style tensor discrepancy summaries include padding.

| Scope | Field | Interface | Internal temperature |
|---|---:|---:|---:|
| P0 | 0.00327616 | 0.0287666 | 0.0193649 |
| P0+P1 | 0.0257982 | 0.1432 | 0.170263 |
| P2 | 0.255672 | 3.54159e-07 | 5.70476e-08 |

### Incremental feedback effect conditional on P0 suppression

| Anchor | Fluid L2 delta | Internal T L2 delta | Final outside T L2 delta | Field prediction mean absolute difference |
|---|---:|---:|---:|---:|
| 0273 | +0.019145 | +0.247953 | +0.252533 | 0.0247552 |
| 0653 | +0.015581 | +0.289532 | +0.317122 | 0.0293584 |
| 0298 | +0.007738 | +0.216543 | +0.225273 | 0.0260623 |
| 0302 | +0.005510 | +0.154015 | +0.128278 | 0.0263103 |

P2 removal raises fluid L2 by 0.2832–0.3440 on every anchor (mean +0.3102) and worsens every physical field channel on average. Prepared interface changes have maximum 8.58e-6 and equal-anchor mean absolute change 3.54e-7 in model output coordinates, consistent with numerical repeat variation; those outputs have no P2-field dependency. Internal changes are smaller. The P0+P1 route contributes strongly to thermal coupling: removing it raises internal-temperature L2 by 0.1532–0.3180, with mean surface-temperature increase +0.3042. Its incremental effect beyond P0 suppression is clear on all four anchors. P0 alone is weaker, including a small internal-temperature improvement on 0302 when suppressed.

The usefulness is not uniform: suppressing P0+P1 improves far-fluid L2 on every anchor and pressure L2 by 0.00552 on average, even though near-interface, overall fluid, and thermal-interface errors worsen. This reveals a remaining conflict between helpful interface feedback and some global field targets. It does not by itself prove independent-coarse dominance.

Conditional encoded-state influence uses a unit all-ones direction for one module state, fixed geometry and global context, and direct sparse-backend reads that omit coarse/local outputs. Topological connectivity means sharing an actual lookup incidence with that module's groups; it is not a learned-weight threshold. Connected/disconnected vector-response norm distributions supplement the existing signed one-probe derivatives, avoiding a claim of useful influence from counts alone.


| Conditional vector derivative | Connected receivers | Mean norm | p95 norm | Maximum norm | Disconnected receivers | Disconnected mean / p95 / max |
|---|---:|---:|---:|---:|---:|---|
| 0298, direct autograd JVP | 6,976 | 0.009203 | 0.036350 | 0.053505 | 1,216 | 0 / 0 / 0 |
| 0302 aligned, direct autograd JVP | 7,360 | 0.017900 | 0.097498 | 0.183760 | 832 | 0 / 0 / 0 |

The state direction is unit-normalized all-ones in the selected module's encoded coordinates; these are context-vector derivative norms, not physical output derivatives. The original central differences at h=0.001/0.0005 gave connected means 0.00943/0.00964 for 0298 and 0.01806/0.01823 for 0302. Small disconnected FD means (0.000246/0.000612 and 0.000186/0.000377) prompted a bounded repeat check on these same two cases. Five identical-state repeats show roughly 10⁻⁶ output-vector variation. Repeated disconnected FD means roughly double when h halves, while direct autograd JVP is exactly zero there. This identifies a float32 execution/cancellation background, not a disconnected physical interaction; the uncorrected FD observations are retained rather than rounded to zero.

The historical one-probe sum-of-context derivative is retained but is weak evidence: the selected connected probes have signed derivatives around 10⁻¹⁵–10⁻¹⁰ and central differences quantize to zero. The full vector distributions above avoid equating a weak or cancelling scalar probe with branch non-use. No disconnected receiver exists for module 0 in the sampled 0273/0653 grids, so those outside-probe entries remain unavailable. The original study used 256 queries for its derivative connectivity counts; this endpoint uses 8,192 and its counts are not directly interchangeable.

![0302 phase errors and conditional influence](../../diagnostics/generated/interface_operator_study/group_reader_recovery/figures/0302_phase_intervention_influence.png)

Full-model coordinate differences at 0.01r and 0.005r rebuild each perturbed physical case. The support-transition check constructs an aligned variant of anchor 0302; it is distinct from the unmodified 0302 used for prediction/intervention scoring. A change of support keys across the finite-difference sides is reported explicitly. Self finite differences do not supply new ground-truth responses for that displaced geometry.


| Anchor | AD at h=0.01r | FD at h=0.01r | Relative difference | AD at h=0.005r | FD at h=0.005r | Relative difference |
|---|---:|---:|---:|---:|---:|---:|
| 0273 | +0.00559176 | +0.00552668 | 0.0117769 | +0.00559178 | +0.00556972 | 0.00395966 |
| 0653 | -0.02587603 | -0.02589987 | 0.000920611 | -0.02587602 | -0.02587504 | 3.78647e-05 |
| 0298 | +0.02233658 | +0.02233850 | 8.59676e-05 | +0.02233657 | +0.02232525 | 0.000507184 |
| 0302 | +0.03243674 | +0.03241830 | 0.000568705 | +0.03243678 | +0.03243155 | 0.000161157 |

The differentiated scalar is the existing model-space mean-temperature output on the fixed query grid, with module 0 moved in +x for 0273/0653/0298 and module 6 moved in +x for the aligned 0302 variant. These full-path derivatives include coarse/local paths and rebuild physical preparation. The maximum relative discrepancy falls from 1.18% at 0.01r to 0.396% at 0.005r, although individual cases need not improve monotonically in float32.

For the aligned 0302 variant, the first port is shifted by 0.0450292 in x to the support transition. The finite-difference sides have 16 versus 20 module support groups and four symmetric key differences. Autograd remains close to the finite difference despite this enumerated support change. This verifies this specific learned-computation transition, not arbitrary topology changes or physical shape derivatives.

## Research conclusion and physical-reference limits

The reader-only experiment recovers useful group-mediated information for the evaluated interfaces and final fields. It is more than an activation recovery: phase removal worsens ground-truth errors, and the corrected endpoint improves most development cases. It is also incomplete: Dense and Legacy remain stronger field predictors, feedback introduces far-field/pressure tradeoffs, and final heat transfer/outlet metrics do not improve uniformly.

The next research priority is a separately authorized, controlled study of how group-mediated thermal feedback and module-dependent coarse communication share information, including a support-matched sparse pairwise control. Group-first coarse communication is a plausible candidate, not a conclusion already established by this experiment. First resolve the pending solver comparisons to judge collective response and physical derivative quality. A longer reader run may test maturity, but this one-seed result does not justify an automatic continuation or a branch-usage penalty.

The existing 16 pair/triple perturbation requests remain pending at `diagnostics/generated/interface_operator_study/stage3/reference_requests/pending_reference_requests.json`. No trustworthy new solver outputs have been supplied. Learned-model autograd versus its own finite differences checks implementation consistency, not physical derivative accuracy or collective-response validity.

The established 90 cases are a repeatedly used development holdout and the experiment has one seed. The mature Run-1401 epoch-4585 result is a separate maturity reference. Run-1804 continuation is deferred. Group-first coarse communication and a support-matched sparse pairwise control remain later research decisions; neither is part of this run.

For a future portability task, physical volume/area quadrature weights should come from the case adapter; `coordinate_scale` is not a general substitute for those weights. That convention is unchanged in this reader-only experiment.

## Artifacts and execution

Primary generated study root: `diagnostics/generated/interface_operator_study/group_reader_recovery/`. Reusable source and this report are tracked in Git. Checkpoints, tables, selected arrays, and figures remain in the existing local run/diagnostic stores under the repository’s established ignore rules; no checkpoint copies or new baseline snapshots were created.

The disposable smoke reuses the maintained configuration/resource loaders, frozen Stage-A attachment, canonical `run_epoch`, and AdamW builder. Its script is `smoke/physical_smoke.py` under that study root. It reserves no run and saves no model or optimizer.

The launch dry-run completed successfully with the new profile, the existing 600/90 split, GPU 0, Run 1805, and 500 epochs. It did not reserve a run directory. The subsequent real launch used the ordinary allocator. The final CPU source suite passed **382 tests**, with three CUDA-specific checks skipped during that invocation; actual sparse physical backward/update execution is recorded above. No managed training failure or restart occurred. Final plot renders, command parsing/compilation, targeted Ruff checks, and `git diff --check` also completed successfully.

All commands below use working directory `/home/wanglz/Desktop/src/ModularDT/HONF_Proj` and the existing `ModularDT` conda interpreter directly.

Executed training launch:

```bash
rtk proxy bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py --config project://src/config_core/forward/sparse_interface_geometry_read_context.json --workflow forward --device cuda:0 --epochs 500 --run-id 1805 --run-name sparse_interface_geometry_read --yes > diagnostics/generated/interface_operator_study/group_reader_recovery/training_launch.log 2>&1'
```

Executed exact endpoint evaluation (full command also saved in `endpoint_commands.txt`):

```bash
rtk proxy bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python -u evaluate.py --config project://src/config_core/forward/sparse_interface_geometry_read_context.json --workflow compare --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1805_20260908_002326_sparse_interface_geometry_read/epoch_0500_model.pt --label "Geometry-envelope sparse HONF @500" --split test --case-ratio 1.0 --device cuda:0 --query-batch-size 32768 --local-port-condition-mode predicted --return-routing-maps --save-debug-npz --debug-case-id 0273 --debug-case-id 0653 --debug-case-id 0298 --debug-case-id 0302 --skip-figures --output-dir diagnostics/generated/interface_operator_study/group_reader_recovery/endpoint > diagnostics/generated/interface_operator_study/group_reader_recovery/endpoint_evaluation.log 2>&1'
```

Executed CPU source suite:

```bash
rtk proxy bash -c 'CUDA_VISIBLE_DEVICES= PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python -m pytest -q > diagnostics/generated/interface_operator_study/group_reader_recovery/source_tests_closeout.log 2>&1'
```

| Evidence | Path under the primary study root |
|---|---|
| Corrected stored checkpoint audit, including split/phase distributions | `diagnosis/run1802_checkpoint_audit_stable_diagnostic_replay.json` (pre-correction artifact retained alongside) |
| Executed diagnosis and endpoint intervention/gradient commands | `diagnosis/commands.txt` |
| Historical phase-explicit interventions | `interventions/run1802_epoch500_phase_explicit.json` |
| Candidate phase interventions and ground-truth errors | `interventions/run1805_epoch500_phase_explicit.json`, `interventions/phase_tables.md` |
| Candidate conditional influence and full coordinate derivatives | `gradients/run1805_epoch500_gradients.json` |
| Bounded disconnected FD background and direct JVP | `diagnosis/disconnected_vector_background.json`, `diagnosis/check_disconnected_background.py` |
| Compact figures and render commands | `figures/`, `figures/render_commands.txt` |
| Real physical smoke script, output, and log | `smoke/physical_smoke.py`, `smoke/physical_smoke.json`, `smoke/physical_smoke.log` |
| Training launch output | `training_launch.log` |
| Source regression output | `source_tests.log`, `source_tests_closeout.log` |
| Existing-anchor execution measurements | `timing/stage2_anchor_timing_chunk128_vs2048/gpu0_phase_timing.json` |
| Existing largest-shape execution measurements | `timing/stage2_synthetic_largest_old/gpu0_scaling.json` |
| Endpoint/common anchor execution measurements | `timing/stage2_candidate_latent_anchor_timing_chunk128_vs2048/gpu0_phase_timing.json` |
| Endpoint/common largest-shape execution measurements | `timing/stage2_synthetic_largest_endpoint_2048_summary/gpu0_scaling.json` |
| Timing command record | `timing/commands.txt` |
| Actual checkpoint repeat and changed-mode output comparisons | `timing/repeat_output_agreement.json`, `timing/repeat_output_agreement.py` |
| Executed full endpoint evaluation command | `endpoint_commands.txt` |
| Paired-table reduction using the existing comparators | `summarize_endpoint.py`, `endpoint/reduced_comparison.json`, `endpoint/paired_case_deltas.csv`, `endpoint/accuracy_tables.md` |
| Exact endpoint results and selected four-anchor arrays | `endpoint/tables/`, `endpoint/debug_npz/`, `endpoint_evaluation.log` |

The prior matched tables remain at `Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage3_Run1401_1804_1801_1802_Epoch500_90Case/tables/`; checkpoints and historical full-grid arrays are referenced in place.

Documentation-only later sparse continuation, **not executed in this task**:

```bash
rtk env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python train.py --config project://src/config_core/forward/sparse_interface_geometry_read_context.json --workflow forward --device cuda:0 --epochs 2500 --resume-checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1805_20260908_002326_sparse_interface_geometry_read/latest_model.pt --yes
```

This preserves the existing resume procedure and optimizer state if a later research decision authorizes continuation. The current goal stops at the new run's epoch-500 closeout; the command is not a scheduled continuation.
