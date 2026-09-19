# HONF Stage-7 Decoder-Context Ablation Report

**Runs:** 1401, 1402, 1403, 1404, and 1804

**Primary checkpoint policy:** exact epoch 5000

**Evaluation population:** the same 90-case development holdout, all 8,192 grid queries per case

**Report date:** 2026-09-16

## Decision summary

The mature results strengthen the case for useful routed computation without overturning the decoder-context ablation. Removing global decoder context remains harmful: Run 1402 is **+77.7%** worse than Run 1401 in pooled fluid relative L2 and loses all 90 paired cases. Removing near context as well recovers part of that damage, but Run 1403 remains **+40.4%** worse than 1401 with a much heavier error tail.

Run 1404 changes a different mechanism. It retains global and near context, removes direct hyperedge-value context, and routes contextualized module-conditioned pair responses. At exact epoch 5000 it reaches **0.03549 pooled L2**, **5.1% better than 1401**, winning 58/90 paired cases. Its p95 is essentially unchanged and its maximum error is lower. This is a meaningful mature recovery from its poor epoch-500 result, not evidence that global context can be removed.

Dense Run 1804 remains the accuracy leader at **0.02966** and beats Run 1404 in 84/90 cases. Run 1404 is **19.7%** worse in pooled L2 but retains Stage-7-class inference cost. The practical frontier is **1804 for maximum accuracy, 1404 for the best mature routed accuracy, and 1401 as the simpler historical routed baseline.**

## Models and comparison boundary

- **1401 / Legacy:** original Stage-7 structured-context reference.
- **1402 / No global:** Run 1401 with global decoder context disabled.
- **1403 / No global/near:** global and near-module decoder contexts disabled.
- **1404 / Routing-only:** global and near context retained; direct hyperedge-value context disabled; contextualized module tokens feed the fused query-module pair path.
- **1804 / Dense:** dense pairwise field-adaptation accuracy reference, without a learned hyperedge partition.

Runs 1402 and 1403 are controlled decoder-context ablations. Run 1404 is an adjacent routing-path redesign, so its gains cannot be attributed to one context switch. Every exact checkpoint uses predicted port conditions and identical cases, masks, targets, and query counts. This is a one-seed development-holdout comparison, not independent physical-reference validation.

## Reconstruction accuracy at exact epoch 5000

| Run | Model | Pooled L2 | Equal-case mean | Median | p95 | Maximum | Worst case |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1401 | Legacy | 0.03740 | 0.03470 | 0.03162 | 0.06024 | 0.08062 | 0297 |
| 1402 | No global | 0.06644 | 0.06114 | 0.05306 | 0.11315 | 0.17800 | 0286 |
| 1403 | No global/near | 0.05252 | 0.04367 | 0.02928 | 0.10829 | 0.15626 | 0284 |
| 1404 | Routing-only | 0.03549 | 0.03335 | 0.02919 | 0.06053 | 0.06801 | 0284 |
| 1804 | Dense | 0.02966 | 0.02643 | 0.02265 | 0.05216 | 0.06472 | 0298 |

Run 1404 improves over 1401 in pooled error, equal-case mean, median, and maximum. Its main gains are pressure (**0.03723 vs 0.04915**), far-fluid error (**0.03928 vs 0.04178**), and field temperature (**0.04913 vs 0.05229**). It regresses on both velocity components, normal heat flux, and final outside temperature. The aggregate gain is real but component-specific.

Dense remains stronger on pooled error and most major components. Run 1404 is slightly better on near-interface field error and vorticity, but these exceptions do not offset Dense's global advantage.

### Component and region errors

| Metric | 1401 | 1402 | 1403 | 1404 | 1804 |
|---|---:|---:|---:|---:|---:|
| Near-interface field | 0.03369 | 0.05078 | 0.03818 | 0.03388 | 0.03509 |
| Far-fluid field | 0.04178 | 0.08152 | 0.06664 | 0.03928 | 0.02649 |
| u | 0.02127 | 0.03497 | 0.02363 | 0.02507 | 0.01436 |
| v | 0.01730 | 0.02786 | 0.02043 | 0.02075 | 0.01312 |
| pressure | 0.04915 | 0.10836 | 0.09986 | 0.03723 | 0.02557 |
| vorticity | 0.04120 | 0.06580 | 0.04279 | 0.04149 | 0.04189 |
| field temperature | 0.05229 | 0.08276 | 0.05892 | 0.04913 | 0.04070 |
| internal temperature | 0.03474 | 0.04268 | 0.02861 | 0.03484 | 0.02574 |
| surface temperature | 0.04600 | 0.05469 | 0.04018 | 0.04638 | 0.03751 |
| normal heat flux | 0.13179 | 0.15315 | 0.12967 | 0.13878 | 0.10349 |
| final outside temperature | 0.06359 | 0.11274 | 0.07243 | 0.06891 | 0.07320 |

### Paired outcomes against Run 1401

| Candidate | Wins | Losses | Mean case-L2 delta |
|---|---:|---:|---:|
| 1402 | 0 | 90 | 0.02643 |
| 1403 | 49 | 41 | 0.00897 |
| 1404 | 58 | 32 | -0.00135 |
| 1804 | 80 | 10 | -0.00827 |

Run 1404 wins 58/90 cases against 1401; Dense wins 80/90. Run 1403's 49/90 wins coexist with worse pooled error because its losses concentrate in difficult cases.

### Four established anchors

| Case | 1401 | 1402 | 1403 | 1404 | 1804 |
|---|---:|---:|---:|---:|---:|
| 0273 | 0.02292 | 0.04094 | 0.02041 | 0.02651 | 0.01565 |
| 0653 | 0.02230 | 0.04235 | 0.02101 | 0.02586 | 0.02116 |
| 0298 | 0.06362 | 0.07225 | 0.06169 | 0.06546 | 0.06472 |
| 0302 | 0.04067 | 0.06886 | 0.09574 | 0.04236 | 0.05546 |

Run 1404 is worse than Run 1401 on each of these four established anchors even though it wins 58/90 cases across the full holdout. The anchors expose difficult behavior but are not representative of the population ranking. The common-scale case-0298 maps in the HTML are demonstrative rather than a substitute for the 90-case statistics.

## Convergence and checkpoint sensitivity

| Epoch | 1401 | 1402 | 1403 | 1404 | 1804 |
|---|---:|---:|---:|---:|---:|
| 500 | 0.11715 | 0.12117 | 0.12426 | 0.15073 | 0.09874 |
| 2500 | 0.04528 | 0.06989 | 0.05757 | 0.05198 | 0.04884 |
| 5000 | 0.03740 | 0.06644 | 0.05252 | 0.03549 | 0.02966 |

Run 1404 is the slowest starter: its epoch-500 pooled L2 is **0.15073**, versus 0.11715 for 1401 and 0.09874 for Dense. By epoch 2500 it reaches 0.05198, still 14.8% behind 1401. Between epochs 2500 and 5000 it improves by 31.7%, overtakes 1401, and approaches the Dense frontier. Judging it at epoch 500 would therefore have produced the wrong mature ranking.

| Run | Parameters | Final val field MSE | Best val field MSE | First val MSE ≤ 0.01 | Recorded wall h | Peak CUDA MiB |
|---|---:|---:|---:|---:|---:|---:|
| 1401 | 2,473,510 | 0.001933 | 0.001510 | 831 | not logged | not logged |
| 1402 | 2,473,510 | 0.005665 | 0.002955 | 898 | 6.03 | 24620 |
| 1403 | 2,473,510 | 0.003069 | 0.002795 | 805 | 5.98 | 24524 |
| 1404 | 2,473,510 | 0.002003 | 0.001871 | 914 | 6.16 | 24841 |
| 1804 | 4,395,409 | 0.001714 | 0.001521 | 651 | 28.30 | 27210 |

| First validation field-MSE crossing | 1401 | 1402 | 1403 | 1404 | 1804 |
|---|---:|---:|---:|---:|---:|
| ≤ 0.02 | 461 | 510 | 409 | 550 | 357 |
| ≤ 0.01 | 831 | 898 | 805 | 914 | 651 |
| ≤ 0.005 | 1,431 | 1,732 | 1,597 | 1,663 | 1,109 |
| ≤ 0.003 | 2,013 | 4,486 | 3,542 | 2,516 | 1,764 |

Run 1404 reaches early validation thresholds later than 1401 and reaches 0.003 at epoch 2516, but its mature full-grid reconstruction is better. Its 5,004 raw history rows contain four duplicate epochs (851–854) from an interrupted resume; the reducer retains the final row for each epoch, yielding a consecutive 5,000-epoch series without modifying the source file.

### Validation-selected checkpoint sensitivity

| Run | Selected epoch | Pooled L2 | Equal-case mean | p95 | Maximum |
|---|---:|---:|---:|---:|---:|
| 1401 | 4,585 | 0.03210 | 0.02982 | 0.05056 | 0.07415 |
| 1402 | 4,936 | 0.05099 | 0.04350 | 0.09484 | 0.16231 |
| 1403 | 4,588 | 0.05066 | 0.04233 | 0.10644 | 0.13672 |
| 1404 | 4,890 | 0.03460 | 0.03244 | 0.06028 | 0.06690 |
| 1804 | 4,738 | 0.02896 | 0.02562 | 0.05281 | 0.06398 |

Run 1404's best-field checkpoint is epoch 4890 and improves pooled L2 by only **2.5%** relative to its exact endpoint. It remains worse than the selected Run 1401 and Dense checkpoints. These selected results reuse the development holdout and are sensitivity evidence; exact epoch 5000 remains primary.

Recorded training wall times were collected under different continuation and machine conditions. Run 1404 records 6.16 h and 24,841 MiB peak allocation, but these values are operational history rather than controlled architecture timing.

## Controlled computation cost

| Run | Anchor full ms | Anchor prepared decode ms | M32/E768/Q65k ms | M128/E3072/Q262k ms | Largest allocated MiB |
|---|---:|---:|---:|---:|---:|
| 1401 | 25.47 | 6.72 | 119.0 | 1740.7 | 14405 |
| 1402 | 27.04 | 6.99 | 125.8 | 1790.0 | 14405 |
| 1403 | 26.14 | 6.89 | 125.5 | 1817.2 | 14405 |
| 1404 | 27.09 | 6.67 | 124.1 | 1840.2 | 14516 |
| 1804 | 35.31 | 13.04 | 376.2 | 5395.6 | 6677 |

Run 1404 stays near the Stage-7 cost envelope: its two-anchor full-forward mean is 27.09 ms and its largest-shape median is 1840.2 ms. The largest result is 5.7% slower than the separately measured Run 1401 result, while Dense is 2.93× slower than Run 1404. Run 1404's parameter count remains 2,473,510.

Fine differences among Runs 1401–1404 are comparable to invocation variability. Incremental allocation depends on chunk scheduling and is not total process residency. Missing Stage-7 operation counters mean uninstrumented work, not zero work.

## Hyperedge organization

| Metric | 1401 | 1402 | 1403 | 1404 |
|---|---:|---:|---:|---:|
| Affinity target relative L2 ↓ | 0.57657 | 0.48458 | 0.50458 | 0.57462 |
| Environment mass entropy / log K | 0.93401 | 0.97604 | 0.99254 | 0.92212 |
| Largest environment mass | 0.33340 | 0.24337 | 0.18971 | 0.34600 |
| Module mass shift | 0.02103 | 0.04954 | 0.02157 | 0.01582 |
| Environment mass shift | 0.00141 | 0.00250 | 0.00088 | 0.00139 |
| Effective query hyperedges | 3.83880 | 3.72309 | 3.86873 | 4.61387 |
| Largest query weight | 0.46525 | 0.47596 | 0.44443 | 0.39065 |
| Pairwise edge contribution norm | 2.23656 | 1.43354 | 0.43664 | 2.55961 |

### Full-holdout assignment geometry

| Metric | 1401 | 1402 | 1403 | 1404 |
|---|---:|---:|---:|---:|
| Module row entropy / log K | 0.97659 | 0.86671 | 0.90192 | 0.96715 |
| Module effective edges | 5.75412 | 4.77697 | 5.04808 | 5.65951 |
| Largest module-row weight | 0.24032 | 0.35900 | 0.29797 | 0.25878 |
| Largest module dominant occupancy | 0.47492 | 0.60820 | 0.42831 | 0.50772 |
| Environment row entropy / log K | 0.36708 | 0.54476 | 0.57452 | 0.72791 |
| Environment effective edges | 2.11997 | 2.94139 | 3.15239 | 3.83562 |
| Largest environment-row weight | 0.75613 | 0.63932 | 0.61829 | 0.51822 |
| Largest environment dominant occupancy | 0.35677 | 0.28328 | 0.22899 | 0.48698 |
| Environment neighbor agreement | 0.67229 | 0.66566 | 0.65120 | 0.71493 |
| Environment neighbor L1 | 0.55394 | 0.48012 | 0.46543 | 0.35483 |
| Environment spatial smoothness | 0.72303 | 0.75994 | 0.76729 | 0.82258 |
| Source separation | 0.03877 | 0.08224 | 0.07384 | 0.04092 |
| Region separation | 0.26160 | 0.20358 | 0.14619 | 0.13603 |
| Base→final alignment cost | 0.00149 | 0.00533 | 0.00366 | 0.00366 |
| Base→final module assignment L1 | 0.03416 | 0.06045 | 0.04280 | 0.02940 |
| Base→final environment assignment L1 | 0.00684 | 0.01775 | 0.01337 | 0.01235 |

Run 1404 retains six fixed edges and multi-edge query routing. Its organization is neither collapsed nor sparse. Relative to 1401, its environment assignments are more diffuse and smoother, with lower region separation. Runs 1402 and 1403 show why such organization statistics cannot establish field usefulness by themselves.

## Run 1404 pathway usefulness at maturity

| Intervention | Fluid MSE | Δ fluid MSE | Prediction RMS difference |
|---|---:|---:|---:|
| Normal | 0.001885 | 0.000000 | 0.00000 |
| Uniform query→edge | 0.003457 | 0.001572 | 0.03592 |
| Uniform module→edge | 0.003514 | 0.001629 | 0.03643 |
| Base module token | 0.023640 | 0.021755 | 0.11584 |
| Suppress pair context | 1.382676 | 1.380791 | 1.23140 |

The mature pair pathway is causally active. Replacing contextualized module tokens with base tokens raises four-anchor mean fluid MSE from **0.001885 to 0.023640**. Suppressing pair context raises it to **1.382676**. Uniform query-to-edge and module-to-edge assignments produce smaller but adverse changes, showing that learned routing contributes beyond a nonzero pair path.

| Gradient group | Gradient tensors | Gradient norm | Finite |
|---|---:|---:|---:|
| organizer_module_score | 2 | 0.004558 | True |
| organizer_env_score | 2 | 0.000335 | True |
| environment_encoder | 4 | 0.002957 | True |
| module_environment_context | 6 | 0.017068 | True |
| query_to_hyper_routing | 4 | 0.000838 | True |
| pair_mlp | 8 | 0.169236 | True |
| hyper_value | 0 | 0.000000 | True |

The endpoint backward pass gives finite nonzero gradients to organizer scores, environment encoding, module-environment context, query-to-hyper routing, and the pair MLP. The disabled hyper-value branch has exactly zero gradient, as designed. Usefulness is supported by ground-truth error interventions and live gradients rather than activation magnitude alone.

### Retained-mass execution

| Beta floor | Selected modules mean | Range | Retained beta mass | Pair-MLP rows/case |
|---|---:|---:|---:|---:|
| 0.980 | 5.833 | 3–10 | 0.99999999 | 47,786 |
| 0.990 | 5.833 | 3–10 | 0.99999999 | 47,786 |
| 0.995 | 5.833 | 3–10 | 0.99999999 | 47,786 |
| 0.999 | 5.833 | 3–10 | 0.99999999 | 47,786 |
| 1.000 | 5.833 | 3–10 | 0.99999999 | 47,786 |

Every retained-mass floor from 0.98 through 1.00 selects all physically active modules. Gathered execution reduces padded pair rows from 98,304 to 47,787 per case, but this is padding removal rather than learned physical sparsity.

## Conclusions and next research decision

1. **Retain global decoder context.** Run 1402 remains a decisive negative result and offers no controlled cost benefit.
2. **Do not adopt Run 1403.** Its median and paired wins hide a severe tail and worse pooled error.
3. **Promote Run 1404 to the mature routed reference.** It beats Run 1401 on pooled error and 58/90 cases while preserving the lower-cost Stage-7 envelope.
4. **Keep Dense Run 1804 as the accuracy reference.** It remains materially more accurate, especially in the far field, at substantially higher latency.
5. **Do not claim learned execution sparsity for Run 1404.** Its useful routing affects predictions and ground-truth errors, but retained-mass thresholds keep every active module.
6. **Next experiment:** isolate whether Run 1404's remaining gap to Dense comes from fused pair-kernel field capacity or upstream context quality. Use one controlled mature change and paired full-field error; do not return to K optimization or routing-activity penalties.

## Reproducibility and artifacts

- Existing four-model evidence: `diagnostics/generated/interface_operator_study/stage7_decoder_ablation/`
- Run 1404 exact and best evaluation: `stage7_decoder_ablation/run1404_evaluation/`
- Run 1404 epoch-2500 trajectory: `stage7_decoder_ablation/run1404_trajectory_evaluation/`
- Run 1404 topology: `stage7_decoder_ablation/run1404_topology_quality_full90/`
- Run 1404 interventions and gradients: `stage7_decoder_ablation/run1404_interventions/`, `run1404_endpoint_gradient.json`
- Run 1404 retained-mass study: `stage7_decoder_ablation/run1404_retained_mass/`
- Controlled timing: `stage7_decoder_ablation/timing/four_model_timing.json`, `timing/run1404_timing.json`
- Reduced tables: `stage7_decoder_ablation/reduction/`
- Reducer and renderer: `tools/diagnostics/analyze_stage7_decoder_ablation.py`, `tools/diagnostics/render_stage7_decoder_ablation_report.py`

Evaluator manifests retain exact checkpoint paths and commands. Run 1404 checkpoint tensors and optimizer state are finite. No new CFD or physical-reference validation is claimed.
