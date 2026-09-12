# HONF NStage2 comparison: hierarchical regional reading and group-mediated coarse communication

**Status:** complete at the authorized epoch-500 endpoints, 2026-09-12. Both runs exited successfully. Exact endpoints, separately saved-best checkpoints, formal phase/influence probes, matched GPU0 timing, and offline figure validation are complete. No continuation was launched.

**Main findings:** A reduces the largest-shape environmental receiver interactions by 87.36% relative to Regional, but its current implementation remains 28.1% slower at inference chunk 2,048. It improves several thermal/physical quantities while its exact-500 aggregate field L2 is 5.10% worse; its separately selected epoch-491 checkpoint is substantially better than its own endpoint. B establishes useful local and distant group-mediated paths at roughly Reader's execution cost, but has 2.92% worse aggregate L2 and a larger temperature deficit. These are two early research results with different merits, not evidence for a combined model.

## Scientific questions and comparison policy

This study implements the two independent experiments in [the NStage2 plan](../../UpgradePlan/HONF_NStage2_Model_Development_and_Validation_Plan.md), following the [five-model maturity comparison](HONF_Five_Model_Epoch5000_Comparison_Report.md). Track A asks whether spatially varying response resolution can preserve collective environmental responses with fewer executed environmental receiver interactions. Track B asks whether the existing useful sparse groups improve reconstruction when they also provide the coarse path's module-dependent input.

| Track | Parent | New computation | Actual run | Physical GPU | Authorized endpoint |
|---|---|---|---|---|---|
| A | Regional 1806 | Shared environmental response hierarchy with smooth near/far reading | `1807` | 0 | 500 |
| B | Reader 1805 | Existing group states replace raw modules as the coarse module source | `1808` | 2 | 500 |

The experiments remain separate. Dense 1804, Regional 1806, Reader 1805, Latent 1801, and Legacy 1401 retain their historical functions and checkpoints. The primary reconstruction comparison reuses their existing exact-500, 90-case tables. Candidate saved-best-by-validation-field results are presented separately, with actual selected epochs. The older runs' current best files were selected through 5,000, and their best-through-500 weights were not found; those files cannot supply a matched early selection comparison.

The repeatedly inspected development holdout and one training seed do not establish independent test performance or variation across seeds. Conditional influence, interventions, and AD/FD checks assess model dependence and numerical self-consistency. The [16 independent physical-reference requests](../../diagnostics/generated/interface_operator_study/stage3/reference_requests/pending_reference_requests.json) remain pending: the retained request directory contains the request file and no solver-response artifacts.

## Implemented operators

### A: hierarchical regional response

`forward_architecture="hierarchical_regional_honf"` preserves Dense/Regional's simultaneous fine MM, ME, and EM message preparation. For each tree node, fine environmental encodings and module-conditioned EM responses are pooled by quadrature mass **before** the existing shared environmental-update MLP. A node's residual environmental encoding, pooled response, and unchanged global context produce its response state. The ordinary 24×8 environment has levels containing 192, 48, 12, 3, 2, and 1 nodes: 258 response states in total. Fine leaves remain present, and the common coarse path continues to receive the original fine environment.

Starting with root carrier one, the traversal retains an internal node with carrier times one minus its opening blend and sends carrier times the opening blend to each valid child. The blend is one within one node radius, zero beyond two radii, and a C1 smoothstep of squared normalized distance between them. Child carriers are not divided by child count: node quadrature masses supply the partition measure. Thus the hierarchical measure covers the complete positive-mass environment while moving response resolution with receiver position.

Only retained receiver–node rows enter the relative-coordinate Fourier features, geometry MLP, query–key products, and weighted value aggregation. The scalar weighted softmax uses stable float64 log normalization; neural values remain in the model dtype. Zero-weight nodes do not contribute. Detailed ragged routing exports are opt-in. The direct query–module read, common coarse/local paths, frozen Stage A, predicted-port policy, and physical refinement remain active.

Geometry metadata and weighted centroids are reused within a physical case. Learned states and projections refresh with module states at P0, P1, and P2. Fixed leaf reading recovers Dense and fixed first-parent reading recovers Regional in the focused numerical tests; mixed-resolution reading is its own operator.

### B: group-mediated coarse source

`forward_architecture="sparse_interface_honf"` and `group_read_mode="geometry_envelope_attention"` are retained. The new `coarse_module_source="group_states"` option replaces the coarse raw-module attention module with one attention module of the same width and head count. Its sources are the already-computed raw group states, weighted by geometric occupancy before learned activation. Valid groups are packed by case without introducing a new group neural encoder.

The eight coarse seeds, one processor block, fine environmental background attention, corrected local group read, local correction, losses, and physical loop are unchanged. Empty group sets supply zero group context while the environmental coarse input remains active. Detaching the packed group states therefore removes the direct encoded-module-to-coarse gradient path under fixed geometry and global/environmental inputs. It does not imply zero total physical module influence.

## Configuration and physical execution

The complete candidate profiles are `src/config_core/forward/nstage2_hierarchical_regional_context.json` and `src/config_core/forward/nstage2_group_mediated_reader_context.json`. Both inherit seed 0, hidden/message widths 256/128, four attention heads, learning rate 3e-4, weight decay 1e-5, clip norm 1, no AMP or dropout, 48 training cases per batch, 1,024 sampled field queries per case, and receiver chunks of 128. Frozen Stage A, predicted ports without a curriculum, one physical refinement, normalization, data split, and losses are retained. Checkpoint milestones remain 10, 50, 100, 250, 500, 1,000, 2,500, and 5,000; only training through 500 is executed in this study.

Integrated model source was committed and pushed as `048b5d3` before training. The ordinary CPU test suite then contained **458 passing tests and four CUDA skips**; the focused CUDA hierarchy backward test passed separately. Tests include fixed-resolution value and gradient limits, a dense attention oracle, mass consistency, permutation/padding cases, transition derivatives, group-source packing and zero-source behavior, historical configuration defaults, and physical adapter integration.

The subsequent diagnostic and comparison work is committed as `4968609`, `386adea`, `bc9623d`, `6776aa5`, and `cd2290c`; it does not change the trained models. After the first two additions, the complete ordinary suite passed **473 tests with four CUDA skips**. The initial timing-figure changes passed **10 focused reducer/renderer tests**; saved-best reuse, completed-run history-alias handling, and the final renderer changes passed **14 focused tests** and Ruff. These are separate test scopes. An intermediate single-threaded invocation failed an existing bitwise reference test; the isolated test and complete suite passed under the ordinary default CPU threading. No reference value was changed.

The final offline page passed **125/125 available model/case/channel selections** in Firefox, all five A hierarchy cases, all 35 A case/level maps, and eight deterministic-weight/learned-attention traces. Earlier checks explicitly exercised missing-to-available selections while training was unfinished; those historical records remain separate. B0273/B0298 rendered temperature arrays, masks, and scales matched their source data exactly. Crowded schematic labels and the hierarchy axis were made readable. Final targeted browser verification passed all **20 workload/phase/chunk timing selections** and checked the seven-model accuracy-cost plot against the largest-shape chunk-2,048 timing rows. Screenshots were inspected and the browser processes stopped.

Trusted CPU loading of the actual epoch-10 checkpoints confirms **4,395,409 trainable parameters for A**, matching Regional 1806, and **3,778,064 for B**, matching Reader 1805. All trainable parameter shapes match their parent after B's documented `coarse_module_attention` to `coarse_group_attention` rename. The setup log counts precede lazy materialization and are smaller. The inspection is retained as `comparison/materialized_parameters.json`; it performs no model evaluation or optimizer update.

The actual allocator created:

- `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1807_20260912_091246_nstage2_hierarchical_regional`
- `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1808_20260912_091247_nstage2_group_mediated_reader`

### Disposable optimizer checks and execution repairs

Each track executes at most one successful disposable optimizer step in the small-module bucket and one in the largest ordinary training bucket. These weights are discarded. They are execution checks, not reconstruction assessments.

Track A's first attempt stopped before an update because CUDA does not implement boolean scatter reduction. Occupancy reduction was changed to an integer operation followed by bool conversion. The next attempt completed the 48-case M=1 optimizer step, then exhausted GPU memory during backward on the 48-case M=12 batch. Because the existing helper wrote its JSON only after both batches, the completed small step has a progress-log record but no retained detailed gradient or memory summary. Its displayed field MSE was approximately 1.851 and its displayed loss approximately 15.05; these rounded log values are not substituted for missing full-precision metrics.

The existing activation-checkpointing option was extended to the complete gathered hierarchical read. Traversal and its live carrier weights remain outside the checkpoint, while gathered Q/K/V and attention intermediates recompute during backward. Focused tests verified output, input-gradient, and parameter-gradient agreement. The large batch was then run alone on a fresh disposable model, preserving the one-update-per-bucket limit. The training batch size and receiver chunk remained unchanged.

| Track / completed batch | Cases × modules | Time (s) | Peak allocated (MiB) | Peak reserved (MiB) | Parameter update norm | Maximum parameter change |
|---|---:|---:|---:|---:|---:|---:|
| A small | 48 × 1 | detailed result unavailable | unavailable | unavailable | unavailable | unavailable |
| A large, after execution repair | 48 × 12 | 3.988 | 23802.77 | 27846 | 0.564226 | 0.00030005 |
| B small | 48 × 1 | 1.761 | 6760.31 | 7212 | 0.510438 | 0.00030005 |
| B large | 48 × 12 | 2.294 | 28988.89 | 31532 | 0.327892 | 0.00030041 |

The retained A-large and both B records report finite parameters and applied optimizer updates. A-large regional preparation/receiver update norms are 0.190210/0.165407. B-large group preparation/receiver update norms are 0.157486/0.040549, and its coarse group-source/environment-source update norms are 0.077566/0.107879. The inherited clipping policy remains active. These cold disposable steps ran on their assigned GPUs and, for A's retry, a different disposable initialization sequence; their times are not a controlled architectural speed ratio.

Both failed A logs are preserved beside the successful records under `diagnostics/generated/interface_operator_study/nstage2/track_a/`. B's successful record is under `track_b/`. No checkpoint was copied or saved by these checks.

## Evaluation definitions and attribution boundaries

The primary field score is pooled normalized fluid relative L2: the square root of total squared prediction error divided by total squared target magnitude across the 90 cases and five field channels. Pooled MSE divides the same error sum by the number of evaluated scalar values. Equal-case mean, median, p95, and worst L2 answer a different question and are retained separately. Channel scores in normalized and physical coordinates, the existing near/far masks, physical port/internal/interface quantities, and the established geometry/heat strata use the maintained evaluator definitions.

Training-history validation MSE is the existing sampled-query training diagnostic. It is not substituted for the complete-grid fluid score. Histories are reduced by column name through epoch 500; the last 50 epochs are reported separately. Component gradients and parameter updates are sampled only at the existing recording epochs. Missing `nan` entries at other epochs mean unrecorded, not zero updates. Component norms can overlap their aggregate category and must not be added as a partition of total gradient energy.

P0 removal recomputes initial ports and the downstream physical loop. P1-only removal starts from normal P0 and changes interface feedback and its downstream results. P2 removal changes final field reading after the physical states have been computed. For A, the removed term is only the hierarchical environmental response. For B, joint removal omits local group reading and the group-source coarse contribution before the retained coarse processor; environmental coarse input remains. The separate B P2 removals distinguish these two group routes. Frozen A level-1 reading tests the already-trained weights under a different read rule, not the performance of a separately trained Regional model.

Conditional encoded-module probes freeze geometry, global context, raw environmental encodings, and the local correction. A's retained fine EM permits a module to influence all environmental response nodes, even when an individual node is not read by a receiver. The omitted-node control tests only the latter direct read edge. B's local route has compact group connectivity, while its coarse group route can reach distant receivers. Shared group identities are reported explicitly; a far receiver need not be disconnected.

Coordinate AD/FD runs the complete physical model at fixed field probes. Its signed functionals are mean temperature and the mean pressure difference between the probes' upstream/downstream x quartiles. Derivatives are in normalized model-output units per geometry-coordinate unit; they are distinct from the complete-grid, physical-unit engineering KPIs. The two coordinate steps are 0.01 and 0.005 module radius.

Execution measurements use sequential physical-GPU0 inference (NVIDIA RTX 6000 Ada Generation) on anchors 0273 and 0653 and synthetic shapes `(M,E,Q)=(32,768,65536)` and `(128,3072,262144)`. The comparison holds outer query batches at 32,768 and reports receiver chunks 128 and 2,048 separately, with detailed routing exports disabled. Real cases use two warmups and five repetitions; synthetic cases use one warmup and three repetitions. Encoding/layout construction, physical preparation, prepared decode, and full forward are distinct timings. Peak allocated and reserved memory are reported separately. Synthetic inputs establish execution cost only, not physical reconstruction at those scales.

Existing parent timing artifacts cover the 2,048 protocol, with exact-500 Dense/Regional measurements preferred where available and mature-checkpoint timing explicitly labelled otherwise. A complete matched parent set at chunk 128 was not found, so only the three relevant parents were measured anew at chunk 128, in addition to both candidate chunks. Accuracy-versus-cost figures hold workload, phase, and chunk fixed; the gain from a larger inference chunk is reported separately from the model's changed interaction count.

Early CPU diagnostic execution at epoch 10 exposed two numerical limitations. At the prescribed encoded-state steps 0.001 and 0.0005, weak float32 route responses yielded large AD/FD discrepancies, so a nonzero JVP alone does not establish an accurate sensitivity magnitude. At the prescribed near-side hierarchy shell offset, the float32 opening blend rounded to one and the retained parent weight to zero. Formal probes retain the fixed locations and steps and include a float64 scalar reference of the same geometric blend, without changing the trained model. The epoch-10 checks remain separately named local artifacts and do not enter endpoint plots or accuracy tables.

## Matched endpoint and checkpoint selection results

All primary rows below use the same 90 development cases at exact epoch 500. Parent tables are reused in place.

| Run / model | Pooled normalized fluid L2 | Pooled normalized fluid MSE | Equal-case mean L2 |
|---|---:|---:|---:|
| 1401 / Legacy | 0.117148 | 0.0127052 | 0.113449 |
| 1801 / Latent | 0.145333 | 0.0195542 | 0.140253 |
| 1804 / Dense | 0.098741 | 0.0090263 | 0.096107 |
| 1805 / Reader | 0.139648 | 0.0180545 | 0.136884 |
| 1806 / Regional | 0.096652 | 0.0086484 | 0.092714 |
| 1807 / A, hierarchical regional | 0.101586 | 0.0095539 | 0.098222 |
| 1808 / B, group-mediated reader | 0.143726 | 0.0191242 | 0.140766 |

A's endpoint L2 is 5.10% above Regional and 2.88% above Dense. B's is 2.92% above Reader. These are early, single-seed results. The existing five-model report documents substantially lower mature errors and checkpoint-sensitive Dense/Regional ordering; the new 500-epoch results do not replace those maturity conclusions.

| Candidate saved-best policy through 500 | Actual selected epoch | Pooled fluid L2 | Pooled fluid MSE | Equal-case mean L2 |
|---|---:|---:|---:|---:|
| A 1807 | 491 | 0.086652 | 0.0069514 | 0.084423 |
| B 1808 | 500 | 0.143726 | 0.0191242 | 0.140766 |

A's saved-best reconstruction L2 is 14.70% below its own endpoint. This substantial variation makes the final-epoch deficit an incomplete description of its current quality. However, the missing matched parent best-through-500 weights prevent a selected-versus-selected early ranking. In particular, A@491 being lower than Regional@500 is not evidence that A wins under the same checkpoint-selection policy. The study performs 270 new primary case-model evaluations: A exact 90, A selected 90, and B exact/same-selected 90.

## Track A: completed epoch-500 assessment

Run 1807 exited zero at epoch 500 on 2026-09-12 at 16:48:49 UTC. Its history contains epochs 1–500 exactly once, with finite validation field MSE. Trusted CPU loading confirms the exact endpoint stores epoch 500 and the best-by-field checkpoint stores epoch 491; both retain optimizer state. Its full-grid evaluations and formal probes ran sequentially on physical GPU0 after training exited.

### Reconstruction and thermal tradeoffs

A loses global L2 on 68/90 cases against Regional and 51/90 against Dense. Its median/p95/worst case L2 is 0.098656/0.122223/0.129343, versus Regional's 0.093194/0.119525/0.128841 and Dense's 0.096741/0.112534/0.124763. Its five diagnostic anchors are unusually favorable: A wins four against Regional and all five against Dense, so those figures cannot represent the population ordering by themselves.

| Pooled normalized L2 | Dense 1804 | Regional 1806 | A 1807 |
|---|---:|---:|---:|
| u | 0.051778 | 0.065453 | 0.042703 |
| v | 0.103761 | 0.068299 | 0.092697 |
| Pressure | 0.095269 | 0.101666 | 0.100055 |
| Vorticity | 0.113666 | 0.118440 | 0.132130 |
| Temperature | 0.113344 | 0.120350 | 0.114597 |
| Near-interface aggregate | 0.097592 | 0.094784 | 0.113974 |
| Far-fluid aggregate | 0.099403 | 0.094301 | 0.091585 |

A improves u against Regional on every case and worsens v on every case. It wins 53 pressure, 15 vorticity, and 49 temperature cases against Regional. The hierarchy's fine near states therefore have not produced a uniformly better near-interface field at this endpoint: near-interface L2 is about 20.2% higher than Regional, while far-fluid L2 is about 2.9% lower. Temperature is modestly better than Regional, rather than the cause of A's aggregate deficit.

| Pooled physical relative L2 | Dense 1804 | Regional 1806 | A 1807 |
|---|---:|---:|---:|
| Internal temperature | 0.069485 | 0.078462 | 0.038714 |
| Interface surface temperature | 0.087244 | 0.100236 | 0.054760 |
| Interface normal heat flux | 0.220073 | 0.219096 | 0.215358 |
| Final port environmental temperature | 0.068922 | 0.059373 | 0.097895 |
| Final effective heat-transfer coefficient | 0.041631 | 0.042513 | 0.047564 |
| Provisional port environmental temperature | 0.109385 | 0.098934 | 0.085578 |
| Provisional effective heat-transfer coefficient | 0.643571 | 0.685387 | 0.672483 |

Internal and surface temperatures improve substantially, as does heat flux slightly, while both final port quantities worsen. Equal-case mean outlet-temperature absolute error is 0.364534 for A versus 1.129186 Regional and 1.019576 Dense; active-module-temperature absolute error is 0.198576 versus 0.847600 and 0.605719. Pressure-drop absolute error is 0.005945, between Regional's 0.002772 and Dense's 0.007186. These are complementary engineering outcomes, not quantities to combine into one invented success score.

A's equal-case mean global-L2 difference is positive against Regional in every predefined module-count, spacing, wall-proximity, and heating-heterogeneity group, ranging from +0.00358 to +0.00869. Against Dense, the three-module group improves slightly (−0.000539), while the other module-count groups worsen by +0.00116 to +0.00908; every spacing, wall and heating group has a higher mean. All strata, including unfavorable cases, remain in `comparison/exact500_strata.csv` and the paired tables.

| Anchor | Dense fluid L2 | Regional fluid L2 | A fluid L2 |
|---|---:|---:|---:|
| 0273 | 0.076542 | 0.066278 | 0.075848 |
| 0653 | 0.092394 | 0.089841 | 0.088965 |
| 0283 | 0.100372 | 0.099379 | 0.093537 |
| 0298 | 0.124763 | 0.118867 | 0.109254 |
| 0302 | 0.109814 | 0.126673 | 0.109220 |

The inspected 0273 temperature map preserves the two principal warm bands and lacks B's pronounced upstream warm patch, although diffuse downstream error remains. On 0283, A recovers the main heated wake but underestimates parts of the upper downstream heated region and spreads temperature into cooler areas below it. The error maps retain these residuals even where the anchor's aggregate score improves.

### Learning, clipping and updates

| Metric | A endpoint | A mean over 451–500 | Regional mean over 451–500 | Dense mean over 451–500 |
|---|---:|---:|---:|---:|
| Train field MSE | 0.023680 | 0.017818 | 0.017383 | 0.015850 |
| Validation field MSE | 0.020270 | 0.019007 | 0.018685 | 0.017337 |
| Train temperature MSE | 0.009829 | 0.012719 | 0.011297 | 0.012073 |
| Validation temperature MSE | 0.011281 | 0.012885 | 0.011627 | 0.012089 |

A's full history shows substantial learning, but the final 50-epoch OLS slopes are upward: +1.87e−4/+1.50e−4 MSE per epoch for train/validation field and +6.47e−5/+6.52e−5 for train/validation temperature. Regional's matched validation slopes are slightly downward. The saved-best at 491 and the better late-window means than A's final field value show an active but variable early trajectory; it is not appropriate to describe the final window as steadily improving or as proof of permanent failure.

At A's recorded epoch-500 batch, regional preparation has pre-clip gradient/update norms 0.010134/0.037769, regional reading 0.257091/0.026940, direct QM 0.404077/0.024026, coarse 0.308448/0.069836, and local 0.328133/0.020155. The global clipping scale is 0.570838. Regional's matching source preparation/read update norms are 0.032600/0.026392 with clipping scale one. Only the epoch-500 sample falls inside the last-50 gradient-record window; these values are not continuous-window averages. All named components remain trainable and receive updates.

A's logged training/validation execution totals are 11,645.78/1,233.95 seconds, totaling 12,879.73 active epoch seconds, and its recorded training peak allocation is 24,292.55 MiB. These assigned-run logs are reported separately from controlled inference measurements.

### Phase usefulness is output dependent

The following deltas are intervened-minus-normal ground-truth fluid L2 on the five complete-grid anchors:

| A intervention | 0273 | 0653 | 0283 | 0298 | 0302 |
|---|---:|---:|---:|---:|---:|
| Remove hierarchy at P0 | −0.001568 | −0.000949 | −0.000058 | −0.001427 | −0.002121 |
| Remove hierarchy at P1 only | −0.006805 | −0.002290 | +0.003081 | −0.001921 | −0.012223 |
| Remove hierarchy at P2 | +0.699209 | +0.627224 | +0.604371 | +0.569625 | +0.599132 |
| Replace P2 read with fixed level 1 | +0.011856 | +0.007545 | +0.008572 | +0.003919 | +0.010424 |

The final hierarchical read is useful for aggregate field reconstruction on every anchor, and its trained mixed-resolution rule outperforms the frozen level-1 replacement on that metric. Full normalized field-tensor mean absolute prediction differences are 0.390–0.474 for P2 removal and 0.0230–0.0270 for fixed level 1. However, fixed level 1 improves temperature L2 on four anchors; 0283 is the exception. The different resolution contributions cannot be ranked solely by the aggregate field score.

P0 removal improves final-field global L2 on all five anchors, and P1 removal improves it on four. P1 removal also improves final-field temperature L2 on all five. Yet P1 removal **worsens internal-temperature physical RMSE on every anchor**, by 0.1367, 0.2195, 0.2839, 0.0223, and 0.4699 respectively; surface-temperature RMSE also worsens on all five. The existing physical feedback thus supplies useful internal/interface information even where its downstream field effect is unfavorable. P0 removal worsens final-port temperature RMSE on all five, while its internal-temperature RMSE improves on 0283 and 0298. With P1 removed, final-port temperature RMSE improves only on 0273, and heat-flux RMSE improves only on 0298. This mixed pattern matters more than the size of the final-field ablation alone.

P2 and fixed-level-1 interventions leave earlier physical quantities causally unchanged. Recomputed earlier tensors differ by approximately 1e−6–1e−5 in float32; these small residuals are reported as numerical repeatability effects, not a physical P2 pathway. The separately retained B unchanged-forward control measures differences of the same order; it is not substituted for an A-specific control.

### Shared multiresolution states and derivative limits

The actual 192-leaf hierarchy contains 258 nodes. Across the same 32 fixed field probes on each diagnostic case, it executes 1,548 positive receiver–node incidences and 1,767 traversal visits: 48.375 selected nodes per receiver. The selected incidences comprise 708 leaves, 505 level-1 nodes, 265 level-2 nodes, 43 level-3 nodes, and 27 level-4 nodes; the root is fully opened for these probes. The eight actual port probes execute 512 incidences on 0273 and 497 on 0298, or 64.0 and 62.125 per receiver. In this small environment, the selected count is therefore comparable to, or higher than, Regional's fixed 48 states. Large-environment work must be measured separately.

For these port/field receivers, the measured sum of retained carrier times node mass lies between 71.9999983 and 72.0000008, consistent with total supplied mass 72 within float32 reduction error. The vertical tree, mass/state views and per-receiver plots show deterministic resolution weights separately from learned attention. The same prepared source states are read at ports and field queries; their routing changes with receiver position.

The conditional encoded-module perturbation affects all 192 fine EM sources on both cases. Hierarchical-environment JVPs are nonzero at all eight ports and 32 field probes. Mean port/field JVP norms are 0.003126/0.005102 on 0273 and 0.001967/0.002459 on 0298. Direct QM remains a separate live route, with mean port/field JVP norms 0.029264/0.009515 and 0.015460/0.005983. An omitted zero-carrier node has exactly zero direct JVP, while a selected-node perturbation gives norms 0.026371 and 0.001389. Small finite-difference residuals in the omitted-node control are retained; GPU outputs are not claimed bitwise invariant.

At state steps 0.001/0.0005, mean relative hierarchical-route AD/FD discrepancies range from 9.93% to 38.71%, while direct-QM discrepancies range from 1.17% to 5.76%. The weaker hierarchical conditional responses have limited finite-difference precision. This establishes live paths and direct-read exclusion more clearly than precise response magnitudes.

Full physical-coordinate mean-temperature derivatives are approximately −0.0001309 on 0273 and −0.0135264 on 0298. At steps 0.0045/0.00225, their relative AD/FD discrepancies are 4.77%/1.20% and 0.0042%/0.0693%. Pressure derivatives are +0.0223869 and approximately −0.0002227, with discrepancies 0.0502%/0.0090% and 3.44%/6.60%. The small signed pressure/temperature responses are correspondingly more sensitive to float32 differencing error.

The constructed resolution-shell probe retains the prescribed interior point and offsets just below/above normalized distances 1 and 2. Across both cases, the selected context-coordinate derivative discrepancies range up to 1.204% at the unchanged steps 0.01/0.005. At the near-plus offset, float32 opening rounds to one and parent retention to zero, while the float64 scalar reference at the same coordinates gives parent retention 1.33268e−8. At far-minus, float32 opening remains approximately 5.33304e−8 and becomes zero at far-plus. These are reported arithmetic limits around the smooth mathematical transition; neither the sampling offsets nor the trained model were adjusted to force agreement. They establish numerical self-consistency at the measured locations, not independent physical sensitivity accuracy.

## Track B: completed epoch-500 assessment

Run 1808 exited successfully at epoch 500 on 2026-09-12. Its history contains each epoch 1–500 exactly once, with finite validation field MSE. The exact endpoint and `best_by_field_mse_model.pt` both store epoch 500, retain optimizer state, and contain exactly equal values for all 296 model-state tensors under ordinary trusted CPU comparison. The saved-best result therefore references the endpoint's existing 90-case tables in place. No additional checkpoint evaluation or copied table set is needed for that selection policy.

B's non-timing evaluation and probes ran on its now-free physical GPU2 while A continued training on GPU0. All three formal commands—endpoint evaluation, phase interventions, and conditional/coordinate probes—exited zero. The sequential matched timing protocol then completed on GPU0 after A's training and formal evaluation closed out.

### Reconstruction and physical quantities

Both the B and Reader-1805 endpoint tables contain the same 90 cases, strata, and target/count denominators. The global fluid score contains 3,496,800 scalar values and target squared norm 3,237,304.060521763.

| Exact-500 quantity | Reader 1805 | B 1808 | Interpretation |
|---|---:|---:|---|
| Pooled fluid normalized L2 | 0.139648 | 0.143726 | B +2.920% |
| Pooled fluid normalized MSE | 0.0180545 | 0.0191242 | B +0.0010698 |
| Equal-case mean fluid L2 | 0.136884 | 0.140766 | B loses 58/90 cases, wins 32 |
| Equal-case median fluid L2 | 0.138949 | 0.140751 | Higher central error |
| Equal-case p95 fluid L2 | 0.161313 | 0.171370 | Higher upper-tail error |
| Worst-case fluid L2 | 0.184907 | 0.177194 | Lower maximum despite worse p95 |
| Pooled near-interface L2 | 0.107961 | 0.103998 | B wins 62/90 cases |
| Pooled far-fluid L2 | 0.146243 | 0.154496 | B wins 36/90 cases |

Channel results isolate the main early deficit:

| Pooled normalized channel L2 | Reader 1805 | B 1808 |
|---|---:|---:|
| u | 0.073665 | 0.078580 |
| v | 0.124752 | 0.120411 |
| Pressure | 0.174322 | 0.163547 |
| Vorticity | 0.162415 | 0.167583 |
| Temperature | 0.149460 | 0.174970 |

Temperature contributes +0.0013082 to the overall normalized MSE difference of +0.0010698. Its contribution exceeds the net gap because pressure and v improve. Only 14/90 temperature cases improve. Physical field-temperature MSE rises from 0.945829 to 1.296246. Thus improved near-interface aggregate reconstruction does not mean improved thermal reconstruction throughout the field.

| Pooled physical relative L2 | Reader 1805 | B 1808 |
|---|---:|---:|
| Internal temperature | 0.046182 | 0.054420 |
| Interface surface temperature | 0.067883 | 0.076727 |
| Interface normal heat flux | 0.217576 | 0.213358 |
| Final port environmental temperature | 0.093577 | 0.105221 |
| Final effective heat-transfer coefficient | 0.050515 | 0.046663 |
| Provisional port environmental temperature | 0.145718 | 0.117275 |
| Provisional effective heat-transfer coefficient | 0.678393 | 0.592686 |

Internal-temperature MSE rises from 0.495148 to 0.687555 and final-port temperature MSE from 1.289744 to 1.630678; heat-flux MSE decreases from 12.562492 to 12.080171. The provisional port improvements therefore do not propagate into uniformly better final temperatures. Existing complete-grid engineering KPIs are also mixed: equal-case mean absolute pressure-drop error improves from 0.009787 to 0.008430, with 54/90 paired wins, while outlet-temperature absolute error worsens from 0.619708 to 0.670483. These quantities use the evaluator's physical units; the conditional derivative functionals below use normalized outputs.

The pooled global-L2 changes within module-count strata are −0.000948 for three modules, +0.010337 for five, +0.002668 for seven, and +0.003261 for ten. Five-module cases are the clearest unfavorable stratum, with only 3/25 wins. All spacing groups have higher pooled errors: crowded +0.003857, intermediate +0.005362, separated +0.001949. All wall-proximity groups also worsen in pooled L2. High heating heterogeneity is approximately tied (−0.000222), whereas low and medium heterogeneity worsen by +0.006448 and +0.004915. These are existing diagnostic strata, not an independent discovery/confirmation split.

| Anchor | B minus Reader fluid L2 | B minus Reader temperature L2 |
|---|---:|---:|
| 0273 | +0.007717 | +0.07043 |
| 0653 | +0.023032 | +0.09405 |
| 0283 | +0.002908 | +0.05446 |
| 0298 | −0.035586 | −0.06839 |
| 0302 | −0.002205 | −0.02504 |

The two improved anchors coexist with the broad thermal deficit. The HTML field panels use common ground-truth/prediction scales and common error scales across models so these examples can be inspected without concealing the unfavorable cases.

Visual inspection of B's physical-temperature maps shows a spurious warm patch near the upstream boundary on 0273, where the stored target remains cool, as well as misplaced downstream thermal bands. On 0298, the principal warm wakes and module locations are recognizable, but the prediction underestimates part of the upper downstream warm region and spreads heat into some cooler areas below it. These visible residuals are consistent with useful yet incomplete response reconstruction. B retains Reader's spatial support construction; the experiment changes how group states supply coarse communication, so these results do not demonstrate a better learned geometric organizer.

### Learning and recorded updates

B's train/validation field MSE decreases from 2.08303/1.95009 at epoch 1 to 0.023234/0.028115 at epoch 500. Train/validation temperature MSE decreases from 1.72352/1.23956 to 0.019038/0.024769. Over epochs 451–500, mean train/validation field MSE is 0.027085/0.033873, compared with Reader's 0.024426/0.030436; mean train/validation temperature MSE is 0.020653/0.031292, compared with Reader's 0.016817/0.024329. Ordinary least-squares slopes over these 50 rows, in MSE per epoch, are +1.49e−7/−1.67e−5 for B train/validation field and +2.97e−5/+5.21e−5 for train/validation temperature. Field validation is still improving locally, while both temperature curves slope upward. One noisy late window is not evidence of an irreversible plateau.

The endpoint main/coarse/local context fractions are 0.3631/0.4594/0.1775, versus Reader's 0.3863/0.4355/0.1782. Mean group count is about 41.1 and group-read mass about 0.778. These describe use of the representation, not its accuracy. At the recorded epoch-500 batch, coarse-group and coarse-environment pre-clip gradient norms are 0.03003 and 0.00773, and their parameter-update norms are 0.02525 and 0.02012. The corresponding source-split fields in the parent history are unrecorded, not zero. Thirteen of B's 15 recorded gradient samples through epoch 500 were clipped; the final recorded scale is one. This sampling does not establish an all-batch clipping rate.

Logged cumulative training and validation execution totals are 9,345.44 and 1,022.34 seconds. Their sum, 10,367.79 seconds, is recorded active epoch time, not the process's elapsed wall time and not a matched architecture benchmark. Its logged training peak allocated memory is 29,319.90 MiB.

### Useful group routes and phase attribution

The following values are **intervened minus normal ground-truth fluid L2**, each from the complete 8,192-query anchor grid:

| Removal phase / route | 0273 | 0653 | 0283 | 0298 | 0302 |
|---|---:|---:|---:|---:|---:|
| P0, joint groups | +0.000481 | +0.000548 | +0.000083 | +0.001769 | −0.000432 |
| P1 only, joint groups | +0.018447 | +0.008756 | +0.020179 | +0.017248 | +0.010902 |
| P2, joint groups | +0.313424 | +0.355581 | +0.361508 | +0.362214 | +0.299717 |
| P2, coarse group source only | +0.128555 | +0.092187 | +0.128900 | +0.143516 | +0.074411 |
| P2, local group read only | +0.243354 | +0.293576 | +0.239523 | +0.297701 | +0.282841 |

P1 removal raises internal-temperature physical RMSE by 1.740, 2.145, 1.903, 4.150, and 1.526 respectively. P2 removal raises field-temperature L2 for every anchor under all three route removals. Mean absolute prediction changes over the full normalized field tensor range from 0.198–0.286 for joint P2 removal, 0.106–0.123 for coarse-source removal, and 0.141–0.272 for local removal. These prediction differences include all 8,192 query positions, whereas the fluid L2 error deltas apply the fluid mask. The two route effects are not additive because they enter the retained nonlinear predictor.

Usefulness is not uniform across quantities. P0 removal improves field-temperature L2 on 0273, 0653, and 0302; P1 removal improves it on 0302 by 0.003042 despite worsening that anchor's global fluid L2. P0 also improves internal-temperature RMSE on 0302 and final-port temperature RMSE on 0653 and 0302. Therefore the useful P1/final-field dependence does not justify claiming that every physical feedback quantity benefits from every group input.

P2 cannot causally change already-computed port, interface, or internal states. The diagnostic helper recomputes a complete forward for each variant, so those earlier tensors show small float32 accumulation differences. A bounded control repeated two entirely unchanged forwards on 0273: maximum absolute differences were 7.63e−6 in fields, 7.47e−6 in interface outputs, 8.34e−7 in internal outputs, and 8.58e−6 in ports. This is the same order as the earlier-output P2 residuals. They are recorded as numerical repeatability limits, not retrospective P2 physical effects. The control is retained in `track_b/p2_roundoff_control.json`; the original intervention results remain unchanged.

### Connectivity and coordinate derivatives

Conditional probes use eight actual physical ports and 32 fixed field receivers from the same final prepared representation. Module 0's encoded-state direction is fixed; geometry, global context, environmental encodings, and local correction are held constant. The connected-near local JVP norms are 0.078399 on 0273 and 0.092519 on 0298. Their relative AD/FD discrepancies are 0.68%/1.34% and 0.53%/1.42% at the prescribed state steps 0.001/0.0005.

The far receiver on 0273 still shares groups 11, 12, and 13 with the perturbed module. Its local JVP is only 3.62e−10, so it is weakly connected rather than structurally disconnected. The far receiver on 0298 shares no groups and has an exactly zero local JVP. Its small finite-difference residues, with norms 2.91e−8/7.51e−6, are consistent with float32 difference noise; they are reported without imposing a threshold. In contrast, coarse-route JVP norms at these far receivers are 0.023156 and 0.009645. Detaching the packed group states gives exactly zero conditional coarse JVP at every selected port/field receiver in two repeats. This confirms the intended group-mediated encoded-state path without asserting zero total physical influence under geometry changes.

Selected coarse-route AD/FD relative discrepancies range from approximately 3.7% to 19.1%; they worsen at the smaller prescribed step. Weak local field probes likewise have poor relative agreement despite correct structural zero/nonzero relationships. The evidence supports connectivity and useful frozen route removal more strongly than high-precision conditional sensitivity magnitudes in float32.

Full physical-coordinate probes perturb module 0 in x, using steps 0.0045 and 0.00225. On 0273, signed mean-temperature AD is approximately −0.022717 and pressure-functional AD +0.037447; relative AD/FD discrepancies are below 0.060% for both quantities and steps. On 0298, mean-temperature AD is +0.034025 with discrepancies below 0.011%, while the smaller pressure derivative, −0.001244, has 2.19% and 6.10% discrepancies. Signs and discrepancies are retained in `track_b/probes.json`.

The prescribed support-transition probe on 0298 aligns module 6 with physical port 31 using an x shift of 0.0194564. The support count changes from 20 to 16 across the finite-difference endpoints. Mean-temperature AD is +0.124213 and pressure AD +0.171831; relative discrepancies are 0.080%/0.013% and 0.053%/0.010% at the two unchanged steps. This is evidence of self-consistent derivatives across this actual support change, not a solver-reference sensitivity validation.

## Actual work, latency and memory

All new timing commands completed sequentially on physical GPU0 after the formal endpoint work. The following are medians in milliseconds, with full-forward peak memory in MiB. Preparation includes the physical loop plus one query; encoding, preparation, prepared decode and full forward are independently timed scopes with overlapping work and must not be added. Per-repetition samples, p05/p95 values and allocator baselines remain in the source JSON. The small repetition counts describe these executions, not a confidence interval across machines or training seeds.

### Real physical cases

| Candidate | Receiver chunk | Anchor | Encoding / layout ms | Physical preparation + one query ms | Prepared decode ms | Full forward ms | Peak allocated / reserved MiB |
|---|---:|---|---:|---:|---:|---:|---:|
| A | 128 | 0273 | 9.911 | 127.869 | 510.566 | 672.922 | 84.22 / 118 |
| A | 128 | 0653 | 10.599 | 129.500 | 496.558 | 624.130 | 84.22 / 118 |
| A | 2048 | 0273 | 9.882 | 54.132 | 38.210 | 88.594 | 530.15 / 1058 |
| A | 2048 | 0653 | 10.356 | 55.435 | 38.128 | 84.313 | 529.61 / 1074 |
| B | 128 | 0273 | 3.862 | 55.657 | 178.890 | 234.695 | 79.53 / 110 |
| B | 128 | 0653 | 4.110 | 56.476 | 179.048 | 230.352 | 79.74 / 110 |
| B | 2048 | 0273 | 3.926 | 29.605 | 12.500 | 39.781 | 115.68 / 218 |
| B | 2048 | 0653 | 4.117 | 30.157 | 12.996 | 39.563 | 138.29 / 254 |

| Parent full forward ms | Checkpoint used for timing | Chunk 128: 0273 / 0653 | Chunk 2048: 0273 / 0653 |
|---|---|---:|---:|
| Dense 1804 | Exact 500 at both chunks | 145.481 / 144.360 | 35.007 / 33.662 |
| Regional 1806 | Exact 500 at both chunks | 153.210 / 151.610 | 32.230 / 31.774 |
| Reader 1805 | Exact 500 at 128; mature 5000 at 2048 | 230.695 / 230.132 | 38.029 / 38.633 |

At chunk 2,048, A's average of the two anchor medians is 86.454 ms, versus 32.002 Regional and 34.334 Dense: **2.70× and 2.52× longer**, respectively. Its encoding/layout alone takes about 10 ms, and both preparation and decode remain more expensive than Regional's 25–27 ms preparation and 8.3–8.7 ms decode. A's approximately 530 MiB real-case peak allocation also exceeds Regional's 155.09 MiB and Dense's 461.01 MiB. At chunk 128, A takes 4.26× Regional's average time. The implementation is especially inefficient for many small receiver chunks.

B is close to Reader: at matched exact-500/chunk-128 it is 1.73% slower on 0273 and 0.10% slower on 0653. At chunk 2,048 it is 4.61%/2.41% slower than the reused mature Reader measurements. The latter comparison describes execution at the labelled checkpoints; it is not a same-checkpoint chunk experiment for Reader.

### Established synthetic execution shapes

The small shape is `(M,E,Q)=(32,768,65536)` and the largest is `(128,3072,262144)`. Both use actual layout builders and the complete physical wrapper. Their errors are not physical-reference accuracy measurements.

| Model | Chunk | Small full forward ms | Largest full forward ms | Largest peak allocated / reserved MiB |
|---|---:|---:|---:|---:|
| Dense, exact 500 | 128 | 1041.399 | 6920.905 | 2269.41 / 2922 |
| Regional, exact 500 | 128 | 1135.117 | 4282.624 | 2268.10 / 2926 |
| Reader, exact 500 | 128 | 1678.278 | 6346.249 | 242.83 / 402 |
| A, exact 500 | 128 | 5149.912 | 22470.713 | 2297.02 / 2926 |
| B, exact 500 | 128 | 1610.097 | 6400.182 | 242.83 / 402 |
| Dense, exact 500 | 2048 | 350.738 | 5358.403 | 6707.28 / 9038 |
| Regional, exact 500 | 2048 | 162.077 | 2090.317 | 2274.19 / 2904 |
| Reader, mature 5000 | 2048 | 132.584 | 456.162 | 242.83 / 374 |
| A, exact 500 | 2048 | 463.631 | 2678.491 | 2297.01 / 2904 |
| B, exact 500 | 2048 | 132.763 | 436.082 | 242.83 / 374 |

On the largest shape at chunk 2,048, A takes **28.1% longer than Regional**, while using approximately the same peak allocated memory. It takes about half Dense's time and one third of Dense's peak allocated memory, but Regional already supplies a better measured time/memory tradeoff. On the smaller synthetic shape, A is 2.86× slower than Regional and 1.32× slower than Dense. At chunk 128, its largest-shape time is 5.25× Regional's.

B's matched exact-500/chunk-128 time is 4.06% lower than Reader on the small shape and 0.85% higher on the largest. At chunk 2,048, B is 0.13% slower on the small shape and 4.40% faster on the largest than the mature Reader timing. Their synthetic peak allocations are identical at these reported precisions. This is roughly comparable cost, not a large efficiency gain from group-mediated coarse communication.

The seven-model accuracy-cost figure also reuses Legacy and Latent's mature checkpoint timings for context. On the largest shape/chunk-2,048, their full-forward times are 1701.731 and 224.421 ms, with peak allocations 14428.29 and 241.10 MiB. The figure's y-axis is the matched exact-500 development L2 for every model; its x-axis is measured execution cost with each timing checkpoint identified in the hover/source notes. It is not a plot of synthetic-shape accuracy or an all-mature accuracy comparison.

### A: fewer gathered interactions, more prepared states

Counters below cover the complete physical forward, including port/interface reads and the final field queries. They count actually executed rows, not theoretical FLOPs or an attention mask applied after dense neural work. Environmental geometry rows equal selected receiver–node incidences. Four heads produce four query–key dot products per selected row; the geometry MLP row count is not multiplied again.

| A executed quantity | Anchor 0273 | Anchor 0653 | Small shape | Largest shape |
|---|---:|---:|---:|---:|
| Selected environment geometry rows | 456841 | 465136 | 5069789 | 27029597 |
| Per-head dot products | 1827364 | 1860544 | 20279156 | 108118388 |
| Traversal visits | 518582 | 527810 | 5652688 | 29826107 |
| Rows with fractional carrier, `0 < eta < 1` | 364156 | 371409 | 4224323 | 23060800 |
| Environmental-update rows | 774 | 774 | 3078 | 12294 |
| Projected source K rows, and separately V rows | 774 | 774 | 3078 | 12294 |
| Direct QM message rows | 116736 | 116736 | 2228256 | 35651712 |
| Fine MM message rows | 432 | 432 | 3072 | 49152 |
| Fine ME rows, and separately EM rows | 6912 | 6912 | 73728 | 1179648 |

The trees contain 258, 1,026 and 4,098 nodes for the ordinary, small and largest environments; three physical preparations account for the environmental-update/projected-source totals. Fractional-carrier rows include descendants affected by ancestor opening blends and are a subset of selected rows, not an additional neural-work total. Packed ordinary batches contain 12 module slots even when only three or five are active, explaining their equal executed fine message counts.

For the largest shape, Regional executes 213,910,272 environmental geometry rows and a fine Dense read executes 855,641,088. A's 27,029,597 rows are **87.36% fewer than Regional** and 96.84% fewer than the fine read. The corresponding small-shape Regional/fine counts are 13,369,536/53,478,144. On the ordinary anchors, Regional executes 466,944 rows, so A's reduction is only 2.16% on 0273 and 0.39% on 0653. Small-case geometry does not promise a large interaction saving.

A prepares more states: on the largest shape its 12,294 environmental-update rows compare with Regional's 2,304 and Dense's 9,216. Source K/V projection rows are 12,294 each for A and 2,304 each for Regional. The maintained Dense default also repeats source projection during reading; its measured projected-row count depends on receiver chunk, unlike its environmental-update count. That execution detail is not attributed to topology alone.

The unchanged coarse path executes 4,608 environmental and 288 packed module-attention pairs on either real anchor; valid module pairs are 72/120. These rise to 18,432/768 and 73,728/3,072 on the two synthetic shapes. Fine response preparation, direct QM and the coarse route remain real costs. The measured interaction reduction has not overcome the current traversal, gather/reduction, source-preparation and launch costs. The phase data establish where latency remains high; no kernel profile was collected to assign a precise fraction to any one operation.

### B: changed communication source at similar local work

| B executed quantity | Anchor 0273 | Anchor 0653 | Small shape | Largest shape |
|---|---:|---:|---:|---:|
| Coarse group-attention pairs | 816 | 1176 | 3840 | 11520 |
| Coarse group per-head dot products | 3264 | 4704 | 15360 | 46080 |
| Retained coarse environment pairs | 4608 | 4608 | 18432 | 73728 |
| Local group-read incidences / receiver-bias rows | 109366 | 138011 | 1107604 | 4430632 |
| Environment-to-group message rows | 2245 | 2838 | 12224 | 48768 |
| Module-to-group message rows | 207 | 315 | 2109 | 8325 |

The synthetic group sets contain 160 and 480 groups. B replaces Reader's 768/3,072 coarse raw-module pairs with 3,840/11,520 coarse group pairs on those shapes, while retaining the same environmental background pairs and local group-read work. The new source can therefore cost more coarse attention rows even though no new group encoder is added. Environmental head-dot counts are four times the environmental pair counts. This is a source/communication experiment; it does not reduce local support counts or promise a coarse-attention speedup.

### Separate inference execution gains and numerical limits

For each candidate's own exact checkpoint, changing only inference receiver chunk from 128 to 2,048 gives A 7.60×/7.40× lower full-forward time on the real anchors and 11.11×/8.39× on the small/largest synthetic shapes. B gains 5.90×/5.82× on the anchors and 12.13×/14.68× on the synthetic shapes. Real prepared-decode gains are approximately 13.0–13.4× for A and 13.8–14.3× for B. Encoding/layout cost is essentially unchanged. These execution gains are separate from A's changed interaction counts and B's changed coarse source.

The larger chunk raises A's ordinary peak allocation from about 84 to 530 MiB and its small-shape peak from 236.54 to 706.49 MiB; its largest peak remains about 2,297 MiB. B's ordinary peaks rise to 115.68/138.29 MiB, while its synthetic peaks remain 231.65/242.83 MiB. Reserved memory depends on allocator history and is retained separately in the tables. Training receiver chunks remain 128 throughout both runs.

All eight candidate synthetic measurements report execution status `ok`, and all four candidate timing artifacts report unchanged state-dict structure. **This does not mean every pointwise agreement check passed.** Across four candidate/chunk artifacts, two anchors, and the two normal-output comparisons per anchor, the maintained tolerances are `atol=2e-6, rtol=2e-5`:

| Output | Comparisons | `allclose=false` | Maximum absolute difference | Maximum relative L2 difference |
|---|---:|---:|---:|---:|
| Field | 16 | 3 | 1.57356e−5 | 3.09974e−7 |
| Interface | 16 | 16 | 9.05991e−6 | 1.76936e−6 |
| Internal | 16 | 0 | 1.04308e−6 | 2.45869e−7 |
| Port | 16 | 0 | 9.53674e−6 | 1.37426e−7 |

One field mismatch and eight interface mismatches occur even in the same-chunk comparison. These observations and the separate unchanged-forward control establish finite float32 repeatability limits; they cannot all be attributed to changing receiver chunk. The raw false flags and fixed tolerances are preserved. Small aggregate differences do not justify a claim of bitwise or universal pointwise equivalence, and no new threshold was introduced to relabel these checks as passes.

## Independent research recommendations

**A / Run 1807:** recommend continuing this same run to **2,500** for a maturity assessment, with higher priority than B. Its selected-checkpoint variation, much better internal/surface temperatures, useful P1 physical feedback, and useful multiresolution final read justify assessing convergence beyond 500. Its current implementation is not an efficiency replacement for Regional: even with 87.36% fewer large-shape environmental interactions it is slower, and its near-interface field error is worse. A continuation would assess whether the thermal/physical advantages coexist with mature field accuracy; it would not resolve the measured execution overhead. A later bounded execution study could address that overhead without changing this experiment's scientific model, but it is not an automatic task here.

**B / Run 1808:** give a same-run **2,500** maturity assessment lower priority; if training resources are limited, defer it behind A. B proves the intended local/coarse dependence and gives useful physical feedback, but does not yet improve Reader's principal reconstruction objective at similar cost. The temperature/far-field deficit and mixed late-window temperature trend are the main concerns, rather than inactive groups. Its near-interface, pressure and heat-flux gains remain useful evidence, and a 2.92% early aggregate deficit is insufficient to declare mature failure. There is no basis here to select B as a replacement for Reader or to combine it with A.

Neither track has catastrophic or persistently stalled numerical behavior in the completed training. Neither has established an across-seed accuracy advantage, independently validated physical sensitivities, or a superior mature accuracy/cost tradeoff. The mature Dense/Regional conclusions in the five-model report remain intact. No additional architecture, sweep, solver integration or automatic continuation is part of this closeout.

The following are **unexecuted same-run continuation commands**, not new launches. `--epochs 2500` sets the total endpoint, so each would add epochs 501–2500 from its exact-500 model and optimizer state. The ordinary runner reuses the checkpoint's run directory and managed ID. Existing 1,000/2,500 milestones and normal latest/best checkpoint handling remain in force. No command below has been run.

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj

# A: physical GPU 0, logical cuda:0; unexecuted
rtk proxy env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py \
  --config project://src/config_core/forward/nstage2_hierarchical_regional_context.json \
  --workflow forward --device cuda:0 --epochs 2500 \
  --resume-checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1807_20260912_091246_nstage2_hierarchical_regional/epoch_0500_model.pt \
  --yes

# B: physical GPU 2, logical cuda:0; unexecuted, lower priority
rtk proxy env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:Case_ThermalChannel/src \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py \
  --config project://src/config_core/forward/nstage2_group_mediated_reader_context.json \
  --workflow forward --device cuda:0 --epochs 2500 \
  --resume-checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1808_20260912_091247_nstage2_group_mediated_reader/epoch_0500_model.pt \
  --yes
```

## Local evidence and reproduction

The complete [executed-command record](../../diagnostics/generated/interface_operator_study/nstage2/commands.md) records the actual interpreter, GPU masks, profiles, run directories, evaluation arguments, and output paths. Generated data remain local under `diagnostics/generated/interface_operator_study/nstage2/`; source, tests, profiles, and this report use ordinary Git.

| Evidence | Location beneath the NStage2 output root |
|---|---|
| Disposable physical optimizer checks and retained failure logs | `track_a/physical_optimizer_steps*`, `track_b/physical_optimizer_steps*` |
| Actual parameter counts | `comparison/materialized_parameters.json` |
| Formal A 90-case endpoint and five anchor arrays | `track_a/endpoint500/tables/`, `track_a/endpoint500/debug_npz/` |
| A separately selected epoch-491 90-case evaluation | `track_a/best_field/tables/` |
| A full-grid phase interventions | `track_a/interventions.json` |
| A conditional influence, physical coordinate and resolution-shell probes | `track_a/probes.json` |
| Formal B 90-case endpoint and five anchor arrays | `track_b/endpoint500/tables/`, `track_b/endpoint500/debug_npz/` |
| B full-grid phase interventions | `track_b/interventions.json` |
| B conditional influence, physical coordinate and support-transition probes | `track_b/probes.json` |
| Unchanged-forward numerical control | `track_b/p2_roundoff_control.json` |
| Exact-500 pooled, equal-case, channel, physical, stratum and paired tables | `comparison/exact500_*.csv` |
| Separately labelled saved-best selection | `comparison/best_field_*.csv` |
| Named-column histories and recorded updates | `comparison/learning_curves.csv`, `comparison/last50_window.csv`, `comparison/gradient_updates.csv` |
| Candidate exact-500 timings at both receiver chunks | `track_a/timing_chunk128.json`, `track_a/timing_chunk2048.json`, `track_b/timing_chunk128.json`, `track_b/timing_chunk2048.json` |
| Newly measured three-parent exact-500 timings | `comparison/parent_timing_chunk128.json` |
| Reused exact-500 Dense/Regional chunk-2048 timings | `../regional_response/timing/regional_vs_dense.json` |
| Reused mature Legacy/Latent/Reader chunk-2048 timings | `../five_model_epoch5000/timing/five_model_timing.json` |
| Final field/mechanism and timing browser checks | `figures/endpoint_preview_check.json`, `figures/final_timing_preview_check.json` |
| Inspected field, hierarchy, mechanism and cost screenshots | `figures/a_endpoint_*.png`, `figures/b_endpoint_*.png`, `figures/final_*.png` |
| Offline interactive figures | [figures/index.html](../../diagnostics/generated/interface_operator_study/nstage2/figures/index.html) |

The reducers and figures can be regenerated from the existing artifacts without running a model:

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj
rtk proxy env CUDA_VISIBLE_DEVICES= PYTHONPATH=src:tools/diagnostics \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python \
  tools/diagnostics/nstage2_reduction.py \
  --root diagnostics/generated/interface_operator_study/nstage2
rtk proxy env CUDA_VISIBLE_DEVICES= PYTHONPATH=src:tools/diagnostics \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python \
  tools/diagnostics/render_nstage2_html.py \
  --root diagnostics/generated/interface_operator_study/nstage2
```

The page embeds its plotting library and data for offline preview. Both candidates' endpoint scores and maps are available. It retains explicit missing-evidence handling without substituting earlier checkpoints or parent arrays for candidate predictions. Exact-500 accuracy and selected-checkpoint accuracy remain distinct from the checkpoint provenance of reused timing measurements.
