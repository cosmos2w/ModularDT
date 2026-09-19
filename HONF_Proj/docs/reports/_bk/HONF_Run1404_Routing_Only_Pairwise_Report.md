# HONF Run 1404 Routing-Only Pairwise Report

**Candidate:** fixed-K Stage-7 routing with contextualized fine pair responses
**Run:** 1404, `routing_only_pairwise`
**Primary checkpoint:** exact epoch 500
**Evaluation:** all 90 development-holdout cases, 8,192 field queries per case
**Date:** 2026-09-16

## Decision

Run 1404 establishes that the fixed six-edge organizer can mediate a useful field computation without the direct hyperedge-value branch. At epoch 500, `c_H` is exactly zero and `hyper_value` receives no gradient, while organizer assignment, module-environment preparation, query-to-edge routing, and the legacy pair MLP all receive finite gradients. Uniform-routing and pair-suppression interventions substantially worsen ground-truth error, so the pair route is used rather than merely active.

The candidate is not yet an accuracy replacement. Exact-500 pooled fluid relative L2 is **0.15073**, versus **0.11715** for Run 1401 and **0.09874** for Dense 1804. The best checkpoint through epoch 500 is epoch 472 at **0.13050**, still 11.4% above Run 1401. The validation trajectory is still improving and the best checkpoint materially outperforms the endpoint, so a single continuation to epoch 2,500 is scientifically warranted. That continuation is provided as a handoff and was not executed.

Gathered execution removes padded-module work on the physical cases, but every active module remains selected at every tested retained-beta floor. It therefore demonstrates an implementation speed/memory opportunity, not learned physical sparsity. On the largest 128-module synthetic shape, beta 0.98 removes only 3.8% of pair-MLP rows and is slower and more memory-intensive than dense execution.

## Minimum-change implementation

The generic field `pairwise_module_token_source = "base" | "organizer_contextualized"` was added with historical default `base`. `HypergraphGatedPairwiseKernel` owns token selection for edge-explicit, fused, gathered, factorized/prepared, and diagnostic paths. No pair-MLP parameter shape changed. Historical checkpoints still load strictly with 237 model-state keys.

The Run 1404 overlay changes only the planned decoder and execution settings plus checkpoint milestones:

- `decoder_mode = enhanced_honf_pairwise_only`
- `use_hyper_value_context = false`
- `pairwise_aggregation_mode = fused_query_module`
- `pairwise_module_token_source = organizer_contextualized`
- `routing_execution = dense`
- `query_module_retained_mass_floor = 1.0`
- milestones at 10, 50, 100, 250, 500, 1,000, 2,500, and 5,000

The organizer remains fixed K=6; losses, widths, Stage A, physical refinement, port curriculum, seed, optimizer, normalization, and data are unchanged. Focused configuration, strict-load, arithmetic, token-source, and gathered-before-MLP tests passed: **95 passed**. The gathered full-support test also verifies selection occurs before the pair MLP.

## Pretraining frozen diagnostic

The mature Run 1401 epoch-5000 checkpoint was replayed on the four anchors plus six difficult cases. This is a frozen arithmetic diagnostic, not an estimate of trained Run 1404 performance.

| Frozen Run 1401 variant | Mean fluid MSE | Prediction RMS change from normal |
|---|---:|---:|
| Normal Run 1401 | 0.002997 | 0 |
| Remove `c_H`, base pair token | 0.557605 | 0.74549 |
| Remove `c_H`, contextualized pair token | 0.555630 | 0.74435 |

Mature Run 1401 depends strongly on its direct value branch. Substituting the contextualized token after training does not rescue it; retraining was necessary.

## Formal run and convergence

Run 1404 trained from scratch on physical GPU 0 and stopped exactly at epoch 500. Its manifest reports `completed`, exit code 0, and checkpoints at 10, 50, 100, 250, and 500. Training used 2,473,510 trainable parameters and took 2,177.1 seconds of logged train-plus-validation time, with peak allocated CUDA memory of 24,820.6 MiB.

| Epoch | Run 1401 validation field MSE | Dense 1804 | Run 1404 |
|---:|---:|---:|---:|
| 10 | 1.35767 | 0.58998 | 1.40361 |
| 50 | 0.25968 | 0.21934 | 0.34060 |
| 100 | 0.10671 | 0.08978 | 0.17310 |
| 250 | 0.05031 | 0.03901 | 0.05824 |
| 500 | 0.02360 | 0.01662 | 0.03645 |
| Best through 500 | 0.01841 at 466 | 0.01339 at 493 | 0.02482 at 472 |

Run 1404 is slower to converge than both references at this early checkpoint. The exact endpoint is also noisier than its selected epoch 472 checkpoint, which supports a maturity continuation while preventing an early accuracy claim.

## Matched 90-case reconstruction

Exact endpoints are primary; best-through-500 is a separate checkpoint policy selected on the same development holdout.

| Model | Pooled L2 | Equal-case mean | Median | p95 | Maximum | Worst case |
|---|---:|---:|---:|---:|---:|---:|
| Run 1401 exact 500 | 0.11715 | 0.11345 | 0.11086 | 0.14088 | 0.15024 | 0689 |
| Dense 1804 exact 500 | **0.09874** | **0.09611** | **0.09674** | **0.11253** | **0.12476** | 0298 |
| Run 1404 exact 500 | 0.15073 | 0.14543 | 0.14618 | 0.17986 | 0.20095 | 0690 |
| Run 1404 best through 500, epoch 472 | 0.13050 | 0.13015 | 0.12831 | 0.14948 | 0.16029 | 0278 |

Run 1401 beats exact Run 1404 on all 90 cases, and Dense 1804 beats it on all 90. Epoch 472 beats exact Run 1404 on 69/90 cases but still loses to Run 1401 on 72/90 and to Dense on all 90.

### Channel, region, and physical components

Values below are equal-case means of normalized relative L2 unless noted.

| Metric | Run 1401 | Dense 1804 | Run 1404 exact | Run 1404 best |
|---|---:|---:|---:|---:|
| Near-interface field | 0.11226 | **0.09126** | 0.13033 | 0.11306 |
| Far-fluid field | 0.12355 | **0.09915** | 0.16486 | 0.15473 |
| `u` | 0.08064 | **0.05208** | 0.11234 | 0.12547 |
| `v` | **0.07386** | 0.10278 | 0.08681 | 0.07489 |
| pressure | 0.10310 | **0.09500** | 0.12760 | 0.12250 |
| vorticity | 0.15112 | **0.10944** | 0.18875 | 0.17216 |
| fluid temperature | 0.13771 | **0.11275** | 0.18308 | 0.13372 |
| internal module cells | 0.11889 | 0.12389 | 0.18551 | **0.11052** |
| internal temperature, physical relative L2 | 0.05861 | 0.06051 | 0.09085 | **0.05609** |
| surface temperature, physical relative L2 | 0.08166 | 0.08311 | 0.12303 | **0.08132** |
| normal heat flux, physical relative L2 | **0.21378** | 0.21865 | 0.22748 | 0.22208 |
| final environmental port temperature | 0.09861 | **0.08294** | 0.10094 | 0.12549 |
| final effective port coefficient | 0.03739 | **0.03599** | 0.06100 | 0.03936 |

The selected Run 1404 checkpoint is already competitive on internal/interface temperature, while far-field and most fluid channels remain weak. This pattern is compatible with slower learning of global field communication; it does not prove eventual parity.

### Established anchors

| Case | Run 1401 | Dense 1804 | Run 1404 exact | Run 1404 best |
|---|---:|---:|---:|---:|
| 0273 | 0.09294 | **0.07654** | 0.10985 | 0.12773 |
| 0653 | 0.10658 | **0.09239** | 0.12931 | 0.12361 |
| 0298 | 0.13759 | **0.12476** | 0.19318 | 0.13613 |
| 0302 | 0.13378 | **0.10981** | 0.15968 | 0.13071 |

Checkpoint selection is heterogeneous: epoch 472 helps difficult anchors 0298 and 0302 but worsens 0273.

## Routing usefulness and gradients

The exact endpoint has mean query entropy 1.0738, 3.0526 effective edges, and maximum query weight 0.5789. Run 1401 at the matched epoch has 3.1863 effective edges and maximum weight 0.5491. Run 1404 therefore remains multi-edge but is modestly sharper. Its mean pair context norm is 7.6372, while `c_H` is exactly zero.

On train case 0001 with all 8,192 queries and the real composite loss:

| Parameter group | Gradient norm |
|---|---:|
| Organizer module assignment | 0.11566 |
| Organizer environment assignment | 0.008891 |
| Environment encoder | 0.009251 |
| Module-environment contextualization | 0.03139 |
| Query-to-hyper routing | 0.01553 |
| Pair MLP | 0.43407 |
| `hyper_value` | **0; no gradient tensors** |

The field loss was 0.11408, the pair-context norm was 59.96, and the direct hyperedge-value norm was zero.

Four-anchor interventions provide the stronger usefulness evidence:

| Intervention | Mean prediction RMS change | Mean fluid MSE | Fluid-MSE change | Near-interface MSE change | Far-fluid MSE change |
|---|---:|---:|---:|---:|---:|
| Normal | 0 | 0.02230 | 0 | 0 | 0 |
| Uniform query-to-edge `alpha` | 0.13744 | 0.05200 | +0.02969 | +0.14692 | +0.00931 |
| Uniform module-to-edge incidence | 0.13484 | 0.05358 | +0.03128 | +0.14952 | +0.00970 |
| Replace contextualized token with base token | 0.07437 | 0.02852 | +0.00621 | +0.00594 | +0.00620 |
| Suppress pair context | 1.09910 | 1.12153 | +1.09923 | +4.80896 | +0.69001 |

Both routing factors and the contextualized module token improve ground-truth error. The pair route is indispensable at this checkpoint. These are frozen interventions, so their magnitudes measure reliance rather than standalone component quality.

## Organization quality

The matched exact-500 topology audit shows a non-collapsed but more concentrated Run 1404 organization.

| Final selected representation, 90-case mean | Run 1401 | Run 1404 |
|---|---:|---:|
| Module row entropy / log K | 0.9110 | 0.8572 |
| Module effective edges | 5.1249 | 4.6741 |
| Largest module-row weight | 0.3224 | 0.3814 |
| Largest module dominant occupancy | 0.4356 | 0.6043 |
| Environment row entropy / log K | 0.3405 | 0.5033 |
| Environment effective edges | 2.0311 | 2.6024 |
| Largest environment-row weight | 0.7662 | 0.6628 |
| Environment neighbor agreement | 0.7263 | 0.7048 |
| Environment spatial smoothness | 0.7673 | 0.7747 |
| Source separation | 0.0824 | 0.0793 |
| Region separation | 0.2697 | 0.2267 |
| Base-to-final module assignment L1 | 0.1168 | 0.1450 |
| Base-to-final environment assignment L1 | 0.0179 | 0.0216 |

Run 1404 sharpens and concentrates module assignments while spreading environment assignments over more edges. Geometry remains organized and spatially smooth, but source/region separation is not improved. These statistics describe organization and cannot substitute for the intervention and field-error results.

## Dense and gathered execution

Across the 90 physical cases, all five floors select every active module: selected module count averages 5.83 (range 3–10) and retained beta mass is numerically one. Dense evaluation computes 98,304 padded query-module rows per case; gathered execution averages 47,787 rows because it excludes inactive padding. All floors produce the same selection. The maximum dense-versus-gathered field difference is approximately `3.34e-6`, and pooled ground-truth MSE changes are at numerical-noise scale.

On anchors 0273 and 0653, gathered execution reduces pair-MLP rows from 107,520 to 26,880 and 44,800 respectively. Prepared-decode medians fall from 6.31/6.37 ms to approximately 3.75/4.3 ms, and incremental allocated memory falls from about 363 MiB to 137 MiB. Full-forward medians fall from 26.16/26.50 ms to approximately 22–25 ms because physical preparation dominates.

| Largest synthetic M128/E3072/Q262144 | Pair-MLP rows | Median time | Incremental allocated memory |
|---|---:|---:|---:|
| Dense | 34,603,136 | 1,903.6 ms | 15.22 GB |
| Gathered beta 0.98 | 33,294,096 | 1,939.4 ms | 18.97 GB |
| Gathered beta 0.99 | 33,994,295 | 1,951.4 ms | 19.36 GB |
| Gathered beta 0.995 | 34,332,799 | 1,949.0 ms | 19.55 GB |
| Gathered beta 0.999/1.0 | 34,603,136 | 1,913.7/1,878.8 ms | 19.71 GB |

The physical-case benefit comes from eliminating padded modules. The 128-module case shows no useful large-shape sparsity regime: beta 0.98 removes only 3.8% of rows and gathered indexing costs more time and memory.

## Recommendation

Continue this same Run 1404 once to epoch 2,500 and repeat the exact/selected 90-case comparison plus the bounded four-anchor interventions. The reason is maturity, not early success: epoch 472 materially improves the endpoint, internal/interface temperatures are already credible, and causal routing use is established, while field accuracy still trails both references.

Do not change K, add a value branch, tune retained-mass floors, or launch another architecture before that assessment. The gathered path should remain an evaluation option for padded physical batches; it is not yet a learned sparsity result and should not drive the research decision.

## Artifacts and commands

- Run directory: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1404_20260916_092508_routing_only_pairwise/`
- Matched evaluation: `diagnostics/generated/interface_operator_study/run1404_routing_only/compare_exact_best_90/`
- Frozen diagnostic: `diagnostics/generated/interface_operator_study/run1404_routing_only/frozen_run1401/`
- Endpoint gradients: `diagnostics/generated/interface_operator_study/run1404_routing_only/endpoint_gradient.json`
- Interventions: `diagnostics/generated/interface_operator_study/run1404_routing_only/interventions/`
- Retained-mass study: `diagnostics/generated/interface_operator_study/run1404_routing_only/gathered_retained_mass/`
- Organization audit: `diagnostics/generated/interface_operator_study/run1404_routing_only/topology_quality_full90/`
- Timing: `diagnostics/generated/interface_operator_study/run1404_routing_only/timing.json`
- Training-history figure/table: `diagnostics/generated/interface_operator_study/run1404_routing_only/training_history/`
- Executed commands and unexecuted continuation handoff: `diagnostics/generated/interface_operator_study/run1404_routing_only/commands.txt`

The exact continuation command is the final entry in `commands.txt` and is explicitly marked **NOT EXECUTED**.
