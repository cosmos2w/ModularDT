# HONF Stage-7 Decoder-Context Ablation Report

**Runs:** 1401, 1402, 1403, and 1804  
**Primary checkpoint policy:** exact epoch 5000  
**Evaluation population:** the same 90-case development holdout, all 8,192 grid queries per case  
**Report date:** 2026-09-16

## Decision summary

The global decoder context is necessary for the Stage-7 model represented by Run 1401. Removing it in Run 1402 raises pooled fluid relative L2 from **0.03740 to 0.06644 (+77.7%)**, and Run 1402 loses all 90 paired cases. Removing near-module context as well (Run 1403) recovers much of that loss relative to Run 1402, but still ends **+40.4% worse than 1401** in pooled L2.

Run 1403 has a slightly better case median than 1401 and wins 49/90 pairs, but its p95 and maximum errors are much worse. Its favorable central cases therefore do not support removing both contexts. Run 1804 remains the accuracy leader at **0.02966**, while the Stage-7 family is much cheaper at inference on the measured shapes.

The routing diagnostics do not rescue either ablation. Runs 1402 and 1403 achieve lower module-affinity target error and more diffuse environment mass than 1401, yet reconstruct the field less accurately. All three models retain six fixed organizer edges and route queries over roughly 3.7–3.9 effective edges. These geometry summaries show organized, non-collapsed routing; they do not show useful physical coupling by themselves.

## Models and controlled comparison

- **1401 / Legacy:** original Stage-7 structured-context reference.
- **1402 / No global:** Run 1401 configuration with global decoder context disabled.
- **1403 / No global/near:** global and near-module decoder contexts disabled.
- **1804 / Dense:** dense pairwise field-adaptation reference. It has no learned hyperedge partition, so hyperedge clustering statistics are not applicable.

The exact checkpoints were evaluated with predicted port conditions and routing-map export. Current replay reproduces the historical Run 1401 result exactly; Run 1804 differs from the prior aggregate only at approximately 6e-10 in pooled relative L2. Every model uses the same case IDs, targets, masks, and query count. This is a one-seed development-holdout comparison, not an independent physical-reference validation.

## Reconstruction accuracy

| Run | Model | Pooled L2 | Equal-case mean | Median | p95 | Maximum | Worst case |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1401 | Legacy | 0.03740 | 0.03470 | 0.03162 | 0.06024 | 0.08062 | 0297 |
| 1402 | No global | 0.06644 | 0.06114 | 0.05306 | 0.11315 | 0.17800 | 0286 |
| 1403 | No global/near | 0.05252 | 0.04367 | 0.02928 | 0.10829 | 0.15626 | 0284 |
| 1804 | Dense | 0.02966 | 0.02643 | 0.02265 | 0.05216 | 0.06472 | 0298 |

Run 1403 beats Run 1402 on 80/90 exact-endpoint cases and reduces pooled L2 by **20.95%**. That conditional recovery means the near-module branch is counterproductive or redundant after global context is removed. It does not establish that near context is harmful when global context is present; a global-retained, near-disabled run would be required for that claim.

Run 1403's tail is the main concern: its median is 0.02928 versus 0.03162 for 1401, while its p95 is 0.10829 versus 0.06024. Difficult cases include 0284, 0286, 0283, 0295, 0297, and 0285. Dense 1804 beats 1403 in 82/90 cases and 1402 in all 90.

### Component and region errors

| Metric | 1401 | 1402 | 1403 | 1804 |
|---|---:|---:|---:|---:|
| Near-interface field | 0.03369 | 0.05078 | 0.03818 | 0.03509 |
| Far-fluid field | 0.04178 | 0.08152 | 0.06664 | 0.02649 |
| pressure | 0.04915 | 0.10836 | 0.09986 | 0.02557 |
| vorticity | 0.04120 | 0.06580 | 0.04279 | 0.04189 |
| internal temperature | 0.03474 | 0.04268 | 0.02861 | 0.02574 |
| surface temperature | 0.04600 | 0.05469 | 0.04018 | 0.03751 |
| normal heat flux | 0.13179 | 0.15315 | 0.12967 | 0.10349 |
| final outside temperature | 0.06359 | 0.11274 | 0.07243 | 0.07320 |

Run 1402 degrades every reported fluid channel; pressure more than doubles relative to 1401. Run 1403 nearly restores vorticity and is competitive on internal and interface temperature, but pressure and far-fluid reconstruction remain poor. Dense 1804 is strongest on velocity, pressure, internal module temperature, surface temperature, and normal heat flux. These results point to a field-communication failure rather than merely a port-read failure.

### Four established anchors

| Case | 1401 | 1402 | 1403 | 1804 |
|---|---:|---:|---:|---:|
| 0273 | 0.02292 | 0.04094 | 0.02041 | 0.01565 |
| 0653 | 0.02230 | 0.04235 | 0.02101 | 0.02116 |
| 0298 | 0.06362 | 0.07225 | 0.06169 | 0.06472 |
| 0302 | 0.04067 | 0.06886 | 0.09574 | 0.05546 |

The anchors expose the heterogeneity hidden by averages. Run 1403 is best among the Stage-7 models on 0273 and 0653 and is close to 1401 on 0298, but degrades sharply on 0302. The corresponding HTML includes a common-scale spatial error map for case 0298; it is demonstrative rather than a substitute for the 90-case statistics.

## Convergence and checkpoint sensitivity

| Epoch | 1401 | 1402 | 1403 | 1804 |
|---|---:|---:|---:|---:|
| 500 | 0.11715 | 0.12117 | 0.12426 | 0.09874 |
| 2500 | 0.04528 | 0.06989 | 0.05757 | 0.04884 |
| 5000 | 0.03740 | 0.06644 | 0.05252 | 0.02966 |

| Run | Parameters | Final val field MSE | Best val field MSE | First val MSE ≤ 0.01 | Recorded wall h | Peak CUDA MiB |
|---|---:|---:|---:|---:|---:|---:|
| 1401 | 2,473,510 | 0.001933 | 0.001510 | 831 | not logged | not logged |
| 1402 | 2,473,510 | 0.005665 | 0.002955 | 898 | 6.03 | 24620 |
| 1403 | 2,473,510 | 0.003069 | 0.002795 | 805 | 5.98 | 24524 |
| 1804 | 4,395,409 | 0.001714 | 0.001521 | 651 | 28.30 | 27210 |

| First validation field-MSE crossing | 1401 | 1402 | 1403 | 1804 |
|---|---:|---:|---:|---:|
| ≤ 0.02 | 461 | 510 | 409 | 357 |
| ≤ 0.01 | 831 | 898 | 805 | 651 |
| ≤ 0.005 | 1,431 | 1,732 | 1,597 | 1,109 |
| ≤ 0.003 | 2,013 | 4,486 | 3,542 | 1,764 |

Run 1403 crosses the first 0.01 validation-field threshold earlier than 1401 and 1402, but its exact endpoint and last-100-epoch validation behavior do not reach 1401. Run 1402 improves little after epoch 2500. Dense is already best at epoch 500 and becomes clearly best by epoch 5000.

Validation-selected best-field checkpoints give pooled L2 values of **0.03210 (1401, epoch 4585), 0.05099 (1402, epoch 4936), 0.05066 (1403, epoch 4588), and 0.02896 (1804, epoch 4738)**. These values use the same development holdout for selection and measurement, so they are sensitivity evidence. Exact epoch 5000 remains the primary policy.

The historical training wall records are not controlled across runs. Run 1804 includes continuation accounting and different execution conditions; the old 1401 history does not log the same decomposed timing and memory columns. They describe operational history, while the synchronized timing below supports architecture cost comparisons.

## Controlled computation cost

| Run | Anchor full ms | Anchor prepared decode ms | M32/E768/Q65k ms | M128/E3072/Q262k ms | Largest allocated MiB |
|---|---:|---:|---:|---:|---:|
| 1401 | 25.47 | 6.72 | 119.0 | 1740.7 | 14405 |
| 1402 | 27.04 | 6.99 | 125.8 | 1790.0 | 14405 |
| 1403 | 26.14 | 6.89 | 125.5 | 1817.2 | 14405 |
| 1804 | 35.31 | 13.04 | 376.2 | 5395.6 | 6677 |

The two context switches leave parameter count and measured allocation unchanged within the Stage-7 family. They also provide no repeatable speed benefit: small latency differences are comparable to run-to-run timing variation because the disabled contributions do not remove the surrounding model structure.

Dense 1804 is about **3.10×** slower than 1401 on the largest synthetic shape and about **1.39×** slower on the two-anchor mean. Its largest-shape incremental allocated memory is lower in this chunked benchmark, despite higher latency and parameter count. Allocation is architecture- and chunk-schedule-dependent and is not total process residency.

Only Dense 1804 currently exports operation rows in this benchmark. At the largest synthetic shape it reports 855,641,088 environment-geometry-bias operations, 35,651,712 query-module messages, 1,179,648 messages for each EM and ME direction, and 49,152 MM messages. The Stage-7 operation dictionaries are empty because that legacy wrapper is not instrumented; they must not be interpreted as zero work.

## Hyperedge routing and clustering quality

| Metric | 1401 | 1402 | 1403 |
|---|---:|---:|---:|
| Affinity target relative L2 ↓ | 0.57657 | 0.48458 | 0.50458 |
| Environment mass entropy / log K | 0.93401 | 0.97604 | 0.99254 |
| Largest environment mass | 0.33340 | 0.24337 | 0.18971 |
| Module mass shift | 0.02103 | 0.04954 | 0.02157 |
| Environment mass shift | 0.00141 | 0.00250 | 0.00088 |
| Effective query hyperedges | 3.83880 | 3.72309 | 3.86873 |
| Largest query weight | 0.46525 | 0.47596 | 0.44443 |
| Pairwise edge contribution norm | 2.23656 | 1.43354 | 0.43664 |

All values are equal-case means over 90 cases. Lower affinity-target relative L2 means closer agreement with the organizer target; entropy near one means mass is distributed relatively evenly across the fixed six edges. The active-edge-count target relative L2 is identical at 0.44405 for all three runs because this experiment retains the same fixed six edges. It does not test edge-count selection.

### Full-holdout assignment geometry

| Metric | 1401 | 1402 | 1403 |
|---|---:|---:|---:|
| Module row entropy / log K | 0.97659 | 0.86671 | 0.90192 |
| Module effective edges | 5.75412 | 4.77697 | 5.04808 |
| Largest module-row weight | 0.24032 | 0.35900 | 0.29797 |
| Largest module dominant occupancy | 0.47492 | 0.60820 | 0.42831 |
| Environment row entropy / log K | 0.36708 | 0.54476 | 0.57452 |
| Environment effective edges | 2.11997 | 2.94139 | 3.15239 |
| Largest environment-row weight | 0.75613 | 0.63932 | 0.61829 |
| Largest environment dominant occupancy | 0.35677 | 0.28328 | 0.22899 |
| Environment neighbor agreement | 0.67229 | 0.66566 | 0.65120 |
| Environment neighbor L1 | 0.55394 | 0.48012 | 0.46543 |
| Environment spatial smoothness | 0.72303 | 0.75994 | 0.76729 |
| Source separation | 0.03877 | 0.08224 | 0.07384 |
| Region separation | 0.26160 | 0.20358 | 0.14619 |
| Base→final alignment cost | 0.00149 | 0.00533 | 0.00366 |
| Base→final module assignment L1 | 0.03416 | 0.06045 | 0.04280 |
| Base→final environment assignment L1 | 0.00684 | 0.01775 | 0.01337 |

Run 1402 sharpens module assignments and concentrates dominant modules: module effective edges fall from 5.75 to 4.78 and the largest dominant occupancy rises from 0.475 to 0.608. Run 1403 remains sharper than 1401 but distributes dominant modules more evenly. Both ablations make environmental assignments more diffuse and spatially smoother, while reducing region separation. Their base-to-final assignment changes are also larger, so decoder context affects the organization recomputed through the physical loop.

Run 1402 moves aggregate module mass more than twice as much as 1401, while Run 1403 returns near the 1401 shift. Query routing remains multi-edge rather than collapsing to one edge. The pairwise edge-contribution norm falls from 2.2366 to 1.4335 and then 0.4366, yet 1403 reconstructs better than 1402. Contribution magnitude, entropy, clustering sharpness, and target affinity therefore cannot be read as direct evidence of useful field mediation.

Dense 1804 receives “not applicable” for these columns because it uses dense environment attention and direct query-module communication instead of a learned hyperedge partition. Missing values are not zeros and do not imply inferior organization.

## Conclusions and next research decision

1. **Retain global decoder context in the Stage-7 baseline.** Run 1402 is uniformly worse than 1401 and provides no measured cost reduction.
2. **Do not adopt Run 1403 as a replacement.** It fixes much of Run 1402's damage and improves many central cases, but pooled error and tail robustness remain substantially worse than 1401.
3. **Use Run 1804 as the accuracy reference and Run 1401 as the lower-cost structured-routing reference.** Dense gives the best reconstruction at higher latency; Stage-7 remains materially cheaper on the measured workloads.
4. **Treat organization metrics as diagnostics, not success criteria.** Better target affinity and diffuse hyperedge mass did not imply better reconstruction. Future changes should be judged through paired ground-truth field error and intervention-based influence.
5. If isolating the near-module branch remains scientifically important, the minimal next ablation is global context retained with only near context disabled. It should be run only to answer that specific interaction question; Runs 1402/1403 already rule out the proposed decoder simplifications as replacements.

## Reproducibility and artifacts

Primary reduced tables and machine-readable summary:

`diagnostics/generated/interface_operator_study/stage7_decoder_ablation/reduction/`

Exact candidate evaluation:

`diagnostics/generated/interface_operator_study/stage7_decoder_ablation/evaluation/`

Matched historical replay:

`diagnostics/generated/interface_operator_study/stage7_decoder_ablation/historical_evaluation/`

Trajectory, selected-checkpoint, and timing evidence:

`diagnostics/generated/interface_operator_study/stage7_decoder_ablation/trajectory_evaluation/`  
`diagnostics/generated/interface_operator_study/stage7_decoder_ablation/best_field_evaluation/`  
`diagnostics/generated/interface_operator_study/stage7_decoder_ablation/timing/four_model_timing.json`

Reducer and renderer:

`tools/diagnostics/analyze_stage7_decoder_ablation.py`  
`tools/diagnostics/render_stage7_decoder_ablation_report.py`

The exact evaluator command and checkpoint paths are recorded in each evaluation directory's `comparison_manifest.json`; execution logs are retained beside the output directories. The report does not claim CFD or new physical-reference validation.
