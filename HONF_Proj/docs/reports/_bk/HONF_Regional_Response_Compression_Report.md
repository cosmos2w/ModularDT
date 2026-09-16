# HONF regional response compression

Run 1806 completed its single from-scratch, GPU-0 training through **exactly epoch 500**. The native regional model is a promising early partial compression: its 90-case pooled normalized fluid MSE is **0.00864838**, 4.19% below matched Dense 1804 at 500, and removing its regional P1/P2 reads worsens ground-truth field error on every anchor. Measured environmental-update and geometry-read rows fall fourfold, with lower inference memory and useful large-shape speed gains.

The accuracy gain is narrow: transverse velocity supplies the entire net MSE improvement over Dense, while the other four channels, internal/surface temperatures, and the worst-case field tail are worse. P0's thermal effect is mixed. Frozen 192-to-48 projected-state coarsening worsened fluid error in all eight cases; native training is a different experiment. Recommend a later user-authorized continuation of this same run to 2,500 to resolve these tradeoffs. **No continuation was executed.**

The working branch started clean at `6cf15f4` on `agent/honf-core-next`, matching the planning reference. Existing history, checkpoints, configuration profiles, trusted loading, and the 16 pending physical-reference requests are preserved.

## Scientific change and its limits

The native `regional_response_honf` family retains Dense's simultaneous fine module–module, module–environment, and environment–module messages. The environmental messages use the input module states, not the already-updated module states from that same preparation pass. For fine encoded environment `e_j`, fine module-conditioned response `a_EM_j`, and positive quadrature weight `nu_j`, the response in region `r` is

```text
mu_r = sum_{j in r} nu_j
xi_r = sum_{j in r} nu_j y_j / mu_r
e_bar_r = sum_{j in r} nu_j e_j / mu_r
a_bar_r = sum_{j in r} nu_j a_EM_j / mu_r
h_r = e_bar_r + env_update([e_bar_r, a_bar_r, global_context])
```

The current case adapter groups the 24×8 fine grid into 12×4 regions using 2×2 cells. Region IDs follow physical coordinates and metadata; they are deterministic memberships, not learned probabilities or a selected physical rank. The generic reduction consumes explicit IDs and weights. The inherited numerical quadrature convention is unchanged: uniform weights sum to the product of `coordinate_scale`, which is not a general physical-volume guarantee.

Receivers read the regional states with Dense's existing normalized multihead key/value projections, centroid geometry bias, and one `log(mu_r)` term. The direct nonlinear query–module read remains. The regional route replaces the contextual fine environmental route. The unchanged common coarse path still sees the original 192 fine environmental encodings; local correction, physical heads, frozen Stage A, and the single refinement remain in place.

P0 ports, P1 outside-temperature feedback, and P2 field queries use the same preparation/reader. Module-conditioned responses and projected sources refresh after module states change. Source projections may be reused across receiver chunks of the same prepared computation graph, with gradients retained.

This reduces environmental update/read rows from E to R, while retaining fine MM/ME/EM and query–module work. It is partial compression: its regional attention is O(QR), and the M², ME, and QM terms remain. No fourfold total-speed claim follows from reducing environmental sources fourfold.

## Evidence separation

The [epoch-5,000 comparison](HONF_Epoch5000_Comparison_Report.md) is prior maturity evidence: Dense, Legacy, and Reader have pooled normalized fluid relative L2 0.029661, 0.037401, and 0.065865. The [reader-recovery study](HONF_Group_Reader_Recovery_Report.md) supplies historical epoch-500 phase evidence and separately measured execution improvements. Neither is a new result of this candidate.

Frozen coarsening pools each head's projected keys and values **after** source LayerNorm and projection, then reads at weighted region centroids. The native model instead pools encoded environment and fine joint EM responses **before** the existing environmental-update MLP. These operations generally differ because nonlinear update, LayerNorm, and attention do not commute with averaging. Singleton grouping supplies a mathematical identity check, not permission to initialize the formal candidate from Dense.

Stage I uses the four established anchors 0273, 0653, 0298, and 0302. Four further cases were selected from the existing 20-case subset before inspecting coarsening results:

| Case | Active modules | Spacing stratum | Wall stratum | Heating CV stratum |
|---|---:|---|---|---|
| 0277 | 3 | Intermediate | Middle | Low |
| 0281 | 3 | Separated | Near | Low |
| 0291 | 5 | Crowded | Interior | Medium |
| 0680 | 10 | Crowded | Near | Medium |

Frozen P2-only approximation preserves normal physical preparation; frozen full-loop approximation recomputes P0/P1/P2 and all downstream local responses. Component removals and the native regional removals retain every unselected path. P1-only has normal P0, unlike the historical nested P0+P1 removal. Prediction discrepancy establishes reliance; intervened-minus-normal ground-truth error assesses whether that reliance is helpful at the measured checkpoint.

## Mature Dense component measurements

Exact Run 1804 epoch 5,000 has now been replayed on the eight fixed cases, with all 8,192 field queries per case. The following table uses **equal-case means over the four established anchors**. Normal values are absolute relative L2; removal rows are intervened minus normal. Fluid/near/far fields use normalized targets; thermal quantities use denormalized physical targets.

| Intervention | Fluid | Near | Far | Final outside T | Internal T | Surface T | Flux |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normal | 0.039248 | 0.046040 | 0.032112 | 0.089459 | 0.039681 | 0.057950 | 0.135585 |
| Remove P2 environmental read | +0.293881 | +0.215075 | +0.422722 | ≈0 | ≈0 | ≈0 | ≈0 |
| Remove P2 direct-module read | +0.451462 | +0.277831 | +0.482343 | ≈0 | ≈0 | ≈0 | ≈0 |
| Remove P2 coarse read | +0.671207 | +0.643551 | +0.640648 | ≈0 | ≈0 | ≈0 | ≈0 |
| Remove P2 local correction | +0.444353 | +0.708329 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |
| Remove environmental read through P0/P1/P2 | +0.290121 | +0.209174 | +0.420417 | −0.001974 | +0.012499 | +0.025298 | +0.048910 |

The environmental response carries useful final-field information in this mature checkpoint. The direct-module and common routes also matter; these frozen removals do not allocate independent shares of predictive value. Full-loop environmental removal worsens internal/surface temperature and flux on average, but slightly improves final outside-temperature error. P2-only removals leave the already-computed physical outputs unchanged apart from ordinary repeat variation. The eight-case fluid deltas are +0.302061 for P2 environmental removal and +0.298015 for full-loop removal; the four additional cases do not reverse the field conclusion.

Source: `diagnostics/generated/interface_operator_study/regional_response/diagnosis/run1804_epoch5000_interventions.json`. These are new mature reliance measurements, distinct from the earlier epoch-500 Reader study and from the native candidate endpoint below.

## Mature Reader phases and frozen coarsening

The new exact-5,000 Reader intervention uses a normal P0 for P1-only removal. Equal-case four-anchor error deltas are:

| Reader removal | Fluid | Near | Far | Final outside T | Internal T | Surface T | Flux |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normal absolute error | 0.091872 | 0.059571 | 0.103228 | 0.132365 | 0.073814 | 0.098570 | 0.137156 |
| P0 only | +0.007376 | +0.026128 | +0.000436 | +0.023016 | +0.025991 | +0.027808 | +0.199549 |
| P1 only, normal P0 | +0.013330 | +0.018229 | +0.007002 | +0.239758 | +0.155557 | +0.178149 | +0.093305 |
| P2 only | +0.338342 | +0.458689 | +0.185280 | ≈0 | ≈0 | ≈0 | ≈0 |

Reader therefore remains useful in all three roles on these mature anchors despite its worse overall reconstruction score. This supplies new frozen-checkpoint reliance evidence at 5,000; it does not change the scope of the historical epoch-500 nested P0+P1 intervention.

The corresponding prediction discrepancies are below: equal-case means of per-anchor mean absolute differences in checkpoint-normalized tensors. These summaries include padding and all field grid locations, unlike the active-physical-receiver and fluid-only ground-truth errors above. They are reported separately rather than combined across quantities.

| Mature model / removal | Field discrepancy | Interface discrepancy | Internal-T discrepancy |
|---|---:|---:|---:|
| Dense P2 environment | 0.155076 | ≈0 | ≈0 |
| Dense P2 direct module | 0.282103 | ≈0 | ≈0 |
| Dense P2 coarse | 0.473843 | ≈0 | ≈0 |
| Dense P2 local | 0.192612 | ≈0 | ≈0 |
| Dense environment through P0/P1/P2 | 0.150209 | 0.054728 | 0.018432 |
| Reader P0 | 0.007871 | 0.123517 | 0.026020 |
| Reader P1 only | 0.023248 | 0.143707 | 0.144165 |
| Reader P2 | 0.234898 | ≈0 | ≈0 |

The largest displayed-as-zero P2 interface discrepancy is 3.4e−7; the corresponding internal-temperature discrepancy is 5.7e−8. Raw multi-channel port-token discrepancies are retained in the source JSON but are not presented as a single physical error metric.

The frozen Dense experiment uses only the prescribed 2×2 layout. Projected per-head K/V are pooled once per refreshed prepared state: one compressed preparation in P2-only mode and three in full-loop mode. Normal and variant context captures align by physical role and receiver chunk; active-port masks exclude padded modules. Equal-case four-anchor results are:

| Frozen approximation | Fluid Δ | Near Δ | Far Δ | Final outside T Δ | Internal T Δ | Surface T Δ | Flux Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| P2 only | +0.006396 | −0.000870 | +0.015295 | ≈0 | ≈0 | ≈0 | ≈0 |
| Full P0/P1/P2 loop | +0.006152 | −0.001486 | +0.015473 | +0.001499 | −0.000178 | −0.000188 | +0.008514 |

The mean receiver-wise environmental-context discrepancy, divided by the corresponding normal context norm, is 0.34146 at P2 for P2-only coarsening. Full-loop values are 0.41068 at P0, 0.40639 at P1, and 0.34144 at P2. These context ratios are distinct from field errors. P2-only leaves P0/P1 contexts unchanged to numerical precision; the largest additional-case P1 residual is 3.43e−7 relative in case 0680.

All eight cases have worse fluid errors under both approximations. Across eight cases, mean fluid error rises from 0.033894 by 0.009645 for P2-only and 0.009338 for the full loop; full-loop flux error rises by 0.015569. The four additional cases also expose near-field loss: their inclusion changes the eight-case mean near-field delta to +0.008526/+0.007594. Coarsening loss is not confined to difficult anchors, and physical feedback does not restore the lost field information. It also does not produce a large additional mean field deterioration over P2-only in this bounded sample.

| Frozen case | Normal fluid L2 | P2-only fluid L2 | Full-loop fluid L2 |
|---|---:|---:|---:|
| 0273 | 0.015654 | 0.027483 | 0.027754 |
| 0653 | 0.021157 | 0.028443 | 0.028423 |
| 0298 | 0.064715 | 0.068084 | 0.067540 |
| 0302 | 0.055464 | 0.058564 | 0.057883 |
| 0277 | 0.021394 | 0.040103 | 0.039762 |
| 0291 | 0.030873 | 0.044430 | 0.043841 |
| 0680 | 0.038870 | 0.044490 | 0.045626 |
| 0281 | 0.023023 | 0.036715 | 0.035023 |

The diagnostic records instrumented read/forward wall times and memory. These single forwards include synchronization and context capture, with a cold first reference; they are not the controlled timing comparison. The separate repeated timing section supplies efficiency conclusions. Frozen coarsening and native pre-update grouping remain distinct experiments; no alternate grouping was searched after these negative results.

| Frozen diagnostic case/mode | P2 backend minus direct-module time (ms) | Full physical forward (ms) | Peak allocated MiB |
|---|---:|---:|---:|
| 0273 normal | 67.65 | 687.95 | 81.77 |
| 0273 P2-only | 138.17 | 380.50 | 82.53 |
| 0273 full-loop | 62.67 | 278.94 | 83.40 |
| 0653 normal | 62.74 | 269.07 | 83.75 |
| 0653 P2-only | 67.00 | 318.45 | 83.21 |
| 0653 full-loop | 67.57 | 290.72 | 84.08 |

The first timing column includes environmental geometry, source projection/pooling when first needed, attention, and wrapper overhead; it is a subtraction of two synchronized instrumented boundaries, not a standalone kernel time. These observations do not establish a frozen coarsening speed gain. They retain the actual costs alongside approximation error rather than substituting token-count ratios for execution.

Sources: `diagnosis/run1805_epoch5000_interventions.json` and `frozen_coarsening/run1804_epoch5000_frozen_192_to_48.json` under the regional study root.

![Mature Dense component and Reader phase effects on four anchors](../../diagnostics/generated/interface_operator_study/regional_response/figures/intervention_anchor_effects.png)

The top panels show normal errors; the bottom panels show removal-minus-normal error. A zero P2 internal-temperature change follows the intervention scope: the physical outputs were already computed before the final field read.

![Frozen projected-state coarsening on four anchors](../../diagnostics/generated/interface_operator_study/regional_response/figures/frozen_coarsening_anchor_effects.png)

The left panel compares field-error changes for the two approximation scopes. The right panel shows the **absolute mean receiver-vector discrepancy** for full-loop coarsening at P0/P1/P2, not the relative context ratios quoted above.

## Implementation and completed execution

The single generated study root is `diagnostics/generated/interface_operator_study/regional_response/`, with `diagnosis/`, `frozen_coarsening/`, `training/`, `endpoint500/`, `timing/`, and `figures/`. Formal checkpoints remain in the ordinary managed run directory. Existing comparator tables and arrays are referenced in place.

The literal executed study commands are recorded in [commands.txt](../../diagnostics/generated/interface_operator_study/regional_response/commands.txt), including all three mature-checkpoint calls. Commands below use the project directory as their working directory.

The native implementation was committed and pushed as `878f484`. Resource resolution and a recursive comparison against Run 1804's actual resolved configuration found no unexpected scientific/training differences. Expected differences are the architecture and region setting, run identity/name, and 500 versus the continued reference's 5,000 epochs. Run 1804's maintained profile ID was 1800 while its allocated run ID was 1804; that historical distinction is preserved in `training/config_profile_comparison.json`.

The maintained settings are seed 0, hidden/message widths 256/128, four attention heads, AdamW at learning rate 3e−4 and weight decay 1e−5, clipping 1, AMP off, dropout 0, activation checkpointing enabled, and training receiver chunk 128. The same 600/90 data split, normalization policy, predicted ports, frozen Stage A, one refinement, eight coarse tokens, local radius factor 2.5, and inherited losses were used. Checkpoints at 10/50/100/250/500 were saved by the ordinary milestone policy.

The complete CPU test suite passed **403 tests**, with three CUDA-specific tests skipped in that invocation (`training/source_tests_closeout.log`). Focused coverage includes weighted mass/centroid preservation, permutation/padding and split-weight duplicates, unequal-mass constant-within-region attention, singleton Dense/native forward and parameter/input gradients, live source caching, component/role isolation, and physical-wrapper integration. An earlier test invocation caught the study capture wrapper using the wrong nesting; both execution and its regression test were corrected before closeout. Actual frozen outputs include all intended role contexts and the correct active-port counts.

After completing the endpoint tools, the full CPU suite passed **407 tests**, with the same three CUDA skips, in 17.64 s (`training/source_tests_endpoint_tools_default_threads.log`). An optional `OMP_NUM_THREADS=1` invocation failed the pre-existing bitwise historical-background regression while 406 tests passed. The ordinary thread configuration passed without changing production code, test references, or tolerances; the failed invocation is retained in `training/source_tests_endpoint_tools.log`.

Before formal training, a fresh disposable candidate executed two canonical optimizer steps with predicted ports, frozen Stage A, physical refinement, all inherited losses, backward, clipping, and the established optimizer builder. Neither model nor optimizer state was saved or reused for the managed run.

| Real batch | Cases × queries | Modules per case | Wall time (s) | Peak allocated MiB | Peak reserved MiB | Regional preparation / reader gradient norms | Regional preparation / reader update norms |
|---|---|---:|---:|---:|---:|---|---|
| Small | 48 × 1,024 | 1 | 2.186 | 3,701.15 | 4,416 | 46.827 / 33.726 | 0.18928 / 0.16900 |
| Large | 48 × 1,024 | 12 | 3.009 | 23,058.20 | 26,764 | 19.758 / 18.556 | 0.12079 / 0.10581 |

Both steps produced finite parameters and applied updates. Gradient norms are pre-clipping observations; the clip scales were 0.002060 and 0.001812 with the unchanged norm limit 1. These initial losses and gradients establish real physical execution, not trained accuracy. The first step includes lazy materialization and cold execution; the two step times are operational observations rather than a controlled architecture speed comparison. Canonical diagnostics retain regional preparation (`em_message` and `env_update`), regional reader (query, attention, and geometry blocks), direct-module, coarse, and local groups separately.

The existing allocator created `Run_1806_20260910_093649_regional_response_interface` on physical GPU 0. Its single from-scratch launch completed exactly 500 epochs, with process exit code 0 and the ordinary manifest marked completed. The history contains 500 unique consecutive epochs, without a restart or continuation. Trusted CPU loading of its epoch-10 checkpoint confirms **4,395,409 trainable parameters**, matching Dense, with no uninitialized parameters left. The setup log's smaller 3,356,945 count precedes lazy materialization and is not the complete model count.

The ordinary run manifest records **2026-09-10 13:36:49–16:30:16 UTC** (09:36:49–12:30:16 local time). Summed training and validation times are 9,338.40 s and 986.58 s, respectively: 10,324.97 s (2.868 h) of epoch work, versus 10,406.92 s (2.891 h) manifest elapsed time. Peak allocated memory is 23,496.33 MiB at epoch 450. Although the historical CSV column says MB, the implementation divides bytes by 1024². The endpoint is `epoch_0500_model.pt` in the managed directory; no checkpoint was copied.

For separate historical context, Dense's first 500 epochs recorded 9,888.42 s of training plus validation and 27,208.66 MiB peak allocation. These runs were not a controlled training-speed comparison; the new run does not establish a training speedup. The complete new history reduction is `training/training_closeout.json`.

The physical device is GPU 0, an NVIDIA RTX 6000 Ada Generation with 49,140 MiB reported memory and driver 570.207. The ordinary run environment records Python 3.12.13 and PyTorch 2.6.0+cu124. `CUDA_VISIBLE_DEVICES=0` exposes that physical device as `cuda:0`.

Sampled milestone observations from the completed named-column CSV are:

| Epoch | Validation field MSE | Validation total loss | Regional preparation / reader gradient norms | Regional preparation / reader update norms |
|---|---:|---:|---|---|
| 10 | 0.518891 | 1.582103 | 1.11048 / 1.78146 | 0.02638 / 0.03507 |
| 50 | 0.203376 | 0.392818 | 0.14661 / 0.59829 | 0.03992 / 0.03915 |
| 100 | 0.093828 | 0.186004 | 0.08950 / 1.00067 | 0.02309 / 0.01469 |
| 250 | 0.073293 | 0.165052 | 0.02477 / 0.43372 | 0.02846 / 0.02320 |
| 500 | 0.015880 | 0.048132 | 0.00531 / 0.07179 | 0.03260 / 0.02639 |

These sampled validation metrics and evolving-batch gradients are distinct from exact-endpoint full-grid error and fixed-checkpoint intervention results.

The trajectory is not monotonic. Median validation field MSE falls from 0.090747 over epochs 101–150 to 0.064197 over 151–200, 0.045080 over 201–250, and **0.015470 over 451–500**. The last 50 values range from 0.011665 to 0.065576. The best logged field and temperature scores are 0.0116646 and 0.00659704 at epoch 491; neither best-selected value replaces the exact-500 evaluation. At 500, training field MSE is 0.0152245 and validation temperature MSE is 0.0143941. These observations show continuing, noisy learning rather than a persistently stalled trajectory.

At epoch 500, coarse gradient/update norms are 0.08359/0.05909 and local norms are 0.07707/0.01731, separately from the regional observations above. Only epoch 500 has finite recorded gradient/update samples in the last 50-epoch window, so those columns do not support an intra-window gradient trend. No catastrophic numerical or resource failure occurred in the formal run.

Commands executed from `HONF_Proj/`:

```bash
rtk proxy bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python -u tools/diagnostics/run_regional_response_study.py smoke --device cuda:0 --output diagnostics/generated/interface_operator_study/regional_response/training/physical_smoke.json > diagnostics/generated/interface_operator_study/regional_response/training/physical_smoke.log 2>&1'
rtk proxy bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py --config project://src/config_core/forward/regional_response_interface_context.json --workflow forward --device cuda:0 --epochs 500 --run-id 1806 --run-name regional_response_interface --yes > diagnostics/generated/interface_operator_study/regional_response/training/training_launch.log 2>&1'
```

## Exact-500 development-holdout comparison

The new exact-500 assessment completed all **90/90 cases** once using full grids, checkpoint-owned normalization, predicted ports, the configured 128 receiver chunk, and canonical physical metrics. Detailed routing/debug arrays were retained only for four anchors. Near/far field masks retain surface distances 0–0.25 and at least 1.0 in dataset coordinate units, distinct from radius-normalized layout strata. This is the repeatedly used development holdout, not an untouched test set.

The maintained reducer matched all 90 IDs across Regional, Dense, Legacy, and Reader; all 20,160 target/count checks across 56 columns and all 80 pooled SSE/target/count reconciliations passed. Global normalized metrics use 3,496,800 fluid channel values with common target squared norm 3,237,304.06052. Pooled MSE is total squared error divided by value count, and pooled relative L2 is the square root of total error energy divided by total target energy. Equal-case summaries reduce the individual case relative L2 values instead.

Dense's existing 90-case scalar tables are reused. Its saved exact-500 anchor arrays for 0273/0653/0298 are reused from the earlier `Stage2_Run1401_1804_1801_1802_Epoch500_20Case/debug_npz/` directory. The missing 0302 array was reconstructed with the standard evaluator on CPU, for that one case only, in `endpoint500/dense_anchor_0302/`. Its fluid relative L2 differs from the existing GPU scalar table by −1.22e−8; internal and final outside-temperature L2 differences are −1.86e−8 and −4.30e−8. This fills a missing figure input without replacing the matched baseline tables or copying existing arrays.



| Exact-500 model | Pooled MSE | Pooled L2 | Equal-case mean L2 | Median | P95 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Regional 1806 | **0.00864838** | **0.096652** | **0.092714** | **0.093194** | 0.119525 | 0.128841 |
| Dense 1804 | 0.00902626 | 0.098741 | 0.096107 | 0.096741 | **0.112534** | **0.124763** |
| Legacy 1401 | 0.01270523 | 0.117148 | 0.113449 | 0.110861 | 0.140883 | 0.150238 |
| Reader 1805 | 0.01805447 | 0.139648 | 0.136884 | 0.138949 | 0.161313 | 0.184907 |

Regional lowers pooled MSE/L2 relative to Dense by 4.186%/2.116%, Legacy by 31.93%/17.50%, and Reader by 52.10%/30.79%. Against Dense it wins 63/90 individual global cases, 57/90 near-field cases, and 65/90 far-field cases. The corresponding counts against Legacy are 87/85/85, and against Reader 90/82/88. The higher p95 and maximum than Dense are material: a small pooled improvement does not establish more reliable difficult-case reconstruction.

| Pooled relative L2 | Regional | Dense | Legacy | Reader |
|---|---:|---:|---:|---:|
| Near-interface normalized field | 0.09478 | 0.09759 | 0.11731 | 0.10796 |
| Far-fluid normalized field | 0.09430 | 0.09940 | 0.12306 | 0.14624 |
| Normalized u | 0.06545 | 0.05178 | 0.08109 | 0.07366 |
| Normalized v | 0.06830 | 0.10376 | 0.07742 | 0.12475 |
| Normalized p | 0.10167 | 0.09527 | 0.10362 | 0.17432 |
| Normalized vorticity | 0.11844 | 0.11367 | 0.15560 | 0.16241 |
| Normalized temperature | 0.12035 | 0.11334 | 0.14541 | 0.14946 |
| Physical u | 0.03712 | 0.02936 | 0.04599 | 0.04177 |
| Physical v | 0.06830 | 0.10376 | 0.07742 | 0.12475 |
| Physical p | 0.09495 | 0.08898 | 0.09677 | 0.16281 |
| Physical vorticity | 0.11844 | 0.11367 | 0.15560 | 0.16241 |
| Physical field temperature | 0.09581 | 0.09023 | 0.11575 | 0.11898 |

The entire net normalized MSE gain over Dense comes from v. Its contribution to global MSE decreases by 0.001354791, while u/p/vorticity/temperature add 0.000297850/0.000177913/0.000242321/0.000258827. The five contributions sum to −0.000377879. These use each channel's SSE divided by the common global value count, not an average of relative errors. Regional beats Dense in 90/90 v cases but only 10/90 u, 33/90 p, 31/90 vorticity, and 41/90 temperature cases. The early gain should therefore be described as channel-specific.

| Physical pooled relative L2 | Regional | Dense | Legacy | Reader |
|---|---:|---:|---:|---:|
| Provisional outside T | 0.09893 | 0.10939 | 0.34533 | 0.14572 |
| Final outside T | 0.05937 | 0.06892 | 0.08804 | 0.09358 |
| Provisional heat-transfer coefficient | 0.68539 | 0.64357 | 0.61031 | 0.67839 |
| Final heat-transfer coefficient | 0.04251 | 0.04163 | 0.04225 | 0.05051 |
| Internal T | 0.07846 | 0.06949 | 0.07172 | 0.04618 |
| Surface T | 0.10024 | 0.08724 | 0.09151 | 0.06788 |
| Interface heat flux | 0.21910 | 0.22007 | 0.21691 | 0.21758 |

Final outside temperature improves, but internal/surface temperatures worsen against all three baselines. Flux is essentially similar at this precision. The physical loop does not inherit a uniform improvement from the global field score.

| Existing KPI, mean absolute error in dataset physical units | Regional | Dense | Legacy | Reader |
|---|---:|---:|---:|---:|
| Pressure drop | 0.002772 | 0.007186 | 0.009129 | 0.009787 |
| Outlet temperature | 1.1292 | 1.0196 | 0.9155 | 0.6197 |
| Active-module mean temperature | 0.8476 | 0.6057 | 0.4594 | 0.1890 |

The 13 predefined strata overlap across axes; their counts must not be added as independent cases. Values below are pooled global normalized fluid relative L2.

| Stratum | n | Regional | Dense | Legacy | Reader |
|---|---:|---:|---:|---:|---:|
| High heating CV | 20 | 0.100185 | 0.102325 | 0.120399 | 0.148008 |
| Low heating CV | 28 | 0.089464 | 0.094532 | 0.107452 | 0.132982 |
| Medium heating CV | 42 | 0.098895 | 0.099368 | 0.120869 | 0.139280 |
| M10 | 15 | 0.113474 | 0.107752 | 0.136714 | 0.145796 |
| M3 | 25 | 0.076203 | 0.082597 | 0.096816 | 0.120886 |
| M5 | 25 | 0.089570 | 0.095552 | 0.110365 | 0.135237 |
| M7 | 25 | 0.101535 | 0.104630 | 0.119958 | 0.150513 |
| Crowded | 45 | 0.104572 | 0.104859 | 0.125880 | 0.146941 |
| Intermediate spacing | 30 | 0.089968 | 0.093724 | 0.108174 | 0.133823 |
| Separated | 15 | 0.074711 | 0.082398 | 0.097738 | 0.120300 |
| Interior wall distance | 12 | 0.081553 | 0.085032 | 0.099522 | 0.120883 |
| Middle wall distance | 50 | 0.098807 | 0.100207 | 0.119157 | 0.142037 |
| Near wall | 28 | 0.097535 | 0.100516 | 0.119156 | 0.141375 |

Regional wins 12/13 strata against Dense, losing M10; it wins all 13 against Legacy and Reader. Its five largest global errors are:

| Case | Regional L2 | Dense L2 | Regional minus Dense |
|---|---:|---:|---:|
| 0691 | 0.128841 | 0.111278 | +0.017563 |
| 0302 | 0.126673 | 0.109813 | +0.016859 |
| 0690 | 0.126399 | 0.111745 | +0.014654 |
| 0688 | 0.124367 | 0.101181 | +0.023187 |
| 0686 | 0.120063 | 0.112607 | +0.007457 |

These are predominantly M10/M7 crowded or middle-wall layouts; 0302 has intermediate spacing and high heating heterogeneity. The largest near-field error is 0.13281 at 0279, and the largest far-field error is 0.15809 at 0302. The same difficult-case and thermal metrics should be followed at later maturity.

Sources: `endpoint500/tables/` and `endpoint500/reduction/`. The reduction references Dense/Legacy tables in `CompareModels/Stage3_Run1401_1804_1801_1802_Epoch500_90Case/tables/` and Reader in `diagnostics/generated/interface_operator_study/group_reader_recovery/endpoint/tables/`; it does not recompute or replace them.

## Regional usefulness at exact 500

These frozen-checkpoint removals use all 8,192 queries on each established anchor. Values are equal-case four-anchor ground-truth relative-L2 means, with the same normalized-field/physical-thermal conventions as Stage I. P0 removal recomputes downstream physical outputs; P1-only preserves normal P0. Direct-module, coarse, and local paths stay present except for their explicitly named P2 removals.

| Regional intervention | Fluid | Near | Far | Final outside T | Internal T | Surface T | Flux |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normal absolute error | 0.100415 | 0.084747 | 0.110454 | 0.077640 | 0.081413 | 0.117415 | 0.227634 |
| Remove P0 regional | +0.001785 | +0.001134 | +0.002230 | +0.000075 | −0.006920 | −0.008655 | −0.003960 |
| Remove P1 regional only | +0.016582 | +0.045649 | +0.003116 | +0.017721 | +0.006557 | +0.016669 | +0.020261 |
| Remove P2 regional | +0.528281 | +0.314041 | +0.746283 | ≈0 | ≈0 | ≈0 | ≈0 |
| Remove P2 direct module | +0.389562 | +0.212507 | +0.411500 | ≈0 | ≈0 | ≈0 | ≈0 |
| Remove P2 coarse | +0.433034 | +0.499965 | +0.348363 | ≈0 | ≈0 | ≈0 | ≈0 |

| Case | P0 fluid Δ | P1-only fluid Δ | P2 fluid Δ |
|---|---:|---:|---:|
| 0273 | +0.000178 | +0.014845 | +0.623181 |
| 0653 | +0.001051 | +0.010681 | +0.540201 |
| 0298 | +0.001924 | +0.019441 | +0.464598 |
| 0302 | +0.003985 | +0.021360 | +0.485144 |

The regional route contributes useful final-field information and useful P1 physical feedback. P0 improves field error slightly on all four anchors but **worsens average internal/surface-temperature and flux error relative to its removal**. This mixed role should remain visible, rather than describing all regional effects as beneficial. The retained QM and coarse paths also carry useful information. Removal effects are not additive attribution or retraining ablations.

Prediction discrepancies below are equal-case means of mean absolute changes in normalized tensors, including padding/all field locations as in the mature table. They establish actual changed predictions separately from the ground-truth deltas.

| Removal | Field discrepancy | Interface discrepancy | Internal-T discrepancy |
|---|---:|---:|---:|
| P0 regional | 0.002390 | 0.017825 | 0.008952 |
| P1 regional only | 0.010349 | 0.041753 | 0.020471 |
| P2 regional | 0.364869 | ≈0 | ≈0 |
| P2 direct module | 0.260991 | ≈0 | ≈0 |
| P2 coarse | 0.336559 | ≈0 | ≈0 |

Displayed-as-zero P2 interface/internal discrepancies are at most 3.44e−7/5.29e−8. P2 happens after these physical outputs have been computed. Source: `endpoint500/regional_interventions.json`.

## Shared-state dependence and coordinate derivatives

Two bounded probes use ordinary anchor 0273 and difficult anchor 0298. For the encoded-module probe, perturb the first active encoded module in a unit all-ones direction with geometry and global input fixed, freshly prepare the regional response, then read those same P2-prepared regional states at eight actual module-port locations and 32 fixed field probes. The port locations come from four angles on each of the first two active modules. This isolates the regional subcomputation; direct QM, coarse, and local contributions are excluded from these JVPs. Sampling ports from a P2-prepared state demonstrates a shared receiver interface and does not relabel that preparation as P0.

| Case | Mean regional-state vector JVP norm | Mean port-read vector JVP norm | Mean field-read vector JVP norm |
|---|---:|---:|---:|
| 0273 | 0.017577 | 0.010693 | 0.011687 |
| 0298 | 0.004225 | 0.002377 | 0.003254 |

All active modules are globally available to all regions, and every receiver can read all 48 regional IDs. There is no structurally disconnected subset on which an exact zero derivative is expected. Deterministic cell membership, learned attention, and module-dependent response derivatives are different quantities.

The two preselected centered-FD steps, 0.001 and 0.0005, do **not** tightly match the encoded-state JVP in FP32. Mean receiver-wise relative JVP/FD discrepancies are:

| Case / receiver | Step 0.001 | Step 0.0005 |
|---|---:|---:|
| 0273 regional state | 0.0168 | 0.0345 |
| 0273 ports | 0.0433 | 0.0829 |
| 0273 fields | 0.0515 | 0.0979 |
| 0298 regional state | 0.0615 | 0.1227 |
| 0298 ports | 0.1771 | 0.3436 |
| 0298 fields | 0.1709 | 0.3088 |

The smaller step is worse, consistent with a finite-precision resolution limitation; this observation does not prove its sole cause. The raw AD and FD responses show dependence, but these figures should not be presented as high-accuracy JVP validation. No step search or tolerance change was used to turn them into passes.

A separate **full physical-loop coordinate** check differentiates the mean normalized predicted field temperature with respect to the first module's x coordinate, including changed physical preparation and downstream responses:

| Case | AD | Centered FD, h=0.01 | Centered FD, h=0.005 | Relative discrepancy at h=0.005 |
|---|---:|---:|---:|---:|
| 0273 | −0.001328769 | −0.001338869 | −0.001330674 | 0.001431 |
| 0298 | −0.017982604 | −0.017981231 | −0.017981231 | 0.0000763 |

This is close model self-consistency on the two sampled coordinate directions, not validation of the physical sensitivity against an external solver. Source: `endpoint500/regional_probes.json`.

## Measured inference cost and retained work

Timing uses physical GPU 0 and the same summary mode, outer query batch 32,768, receiver chunk 2,048, and complete physical wrapper for Dense and Regional at exact 500. The accuracy evaluation and training retain their configured receiver chunk 128. The previous 128-to-2,048 inference improvement is not claimed as a new architectural gain. Detailed routing exports are opt-in and disabled here.

Dense is measured both with its historical default projection behavior and with evaluation-only reuse of projected environmental K/V within each refreshed prepared state. Native regional preparation caches its regional projections with autograd retained. The Dense cache changes execution only and supplies a useful comparison against compression. Two warmups/five timed repetitions are used on anchors; one warmup/three repetitions on synthetic shapes. CUDA synchronization brackets wall timing. Hooks count operation rows in separate untimed passes.

Each phase has its own allocation lifetime: encoding, preparation, and full-forward timing do not retain an unrelated prepared case; prepared decode retains its required preparation. Models/cases are released between independent shape measurements. Peak allocated and peak reserved memory remain separate; reserved memory can retain allocator history. The timing artifact includes baseline and incremental peaks as well as total peaks. The lifetime correction was verified by 12 focused tests (1.49 s) and CPU execution of all timing variants before the actual GPU measurements; `c958b59` contains this diagnostic-only correction.

| Anchor / variant | Encode + layout (ms) | Physical prepare + one query (ms) | Prepared 8,192-query decode (ms) | Full forward median (ms) | Full p05–p95 (ms) | Peak allocated / reserved MiB |
|---|---:|---:|---:|---:|---|---|
| 0273 Dense default | 1.319 | 22.525 | 13.268 | 35.007 | 34.451–37.898 | 461.01 / 618 |
| 0273 Dense cached | 1.266 | 21.789 | 12.986 | 33.460 | 32.944–34.576 | 462.14 / 618 |
| 0273 Regional | 1.523 | 26.685 | 8.730 | 32.230 | 30.995–33.916 | 155.09 / 220 |
| 0653 Dense default | 1.200 | 23.172 | 13.265 | 33.662 | 32.783–35.388 | 461.01 / 618 |
| 0653 Dense cached | 1.737 | 22.747 | 14.420 | 39.931 | 35.944–56.049 | 462.14 / 618 |
| 0653 Regional | 1.553 | 25.333 | 8.326 | 31.774 | 30.601–34.506 | 155.09 / 220 |

Native preparation is slower on both anchors. Its prepared decode is faster and complete-forward medians are 7.9%/5.6% lower than default Dense, with overlapping timing ranges on 0653. These are modest anchor-level gains. Cached Dense has a visibly variable 0653 measurement and no consistent speed advantage over default; do not attribute that outlier as a reliable extra compression gain. Independently measured phase medians need not sum to the full-forward median. Full-forward baseline allocations are 29.16 MiB for Dense and 35.25 MiB for Regional; the total peak allocation reduction is about 66.4% on both anchors.

The synthetic tests use realized even rectangular grids: 48×16 fine cells → 24×8 regions (192) and 96×32 → 48×16 regions (768). They execute the physical wrapper at the requested sizes; all six model/variant measurements completed without OOM. They do not establish physical accuracy at those extrapolated shapes.

| Synthetic M/E/Q | Variant | Median full forward (ms) | Peak allocated MiB | Peak reserved MiB |
|---|---|---:|---:|---:|
| 32 / 768 / 65,536 | Dense default | 350.738 | 1727.50 | 2042 |
| 32 / 768 / 65,536 | Dense cached | 352.191 | 1729.00 | 2046 |
| 32 / 768 / 65,536 | Regional | 162.077 | 492.04 | 788 |
| 128 / 3,072 / 262,144 | Dense default | 5358.403 | 6707.28 | 9038 |
| 128 / 3,072 / 262,144 | Dense cached | 5405.080 | 6716.15 | 9038 |
| 128 / 3,072 / 262,144 | Regional | 2090.317 | 2274.19 | 2904 |

Relative to default Dense, regional inference is 2.16×/2.56× faster on these shapes, with about 71.5%/66.1% less total peak allocated memory. These are measured partial-compression benefits; neither a fourfold overall acceleration nor a training-speed result follows.

Actual untimed MLP output-row counts include all three refreshed preparations and physical receivers. Real anchors have active M=3/5 but padded execution width M_pack=12, explaining identical counts. Masking invalid sources preserves results but does not remove those counted padded evaluations.

| Executed row family | Anchor, Dense → Regional | M32/E768/Q65536 | M128/E3072/Q262144 |
|---|---|---|---|
| MM message | 432 → 432 | 3,072 → 3,072 | 49,152 → 49,152 |
| ME message | 6,912 → 6,912 | 73,728 → 73,728 | 1,179,648 → 1,179,648 |
| EM message | 6,912 → 6,912 | 73,728 → 73,728 | 1,179,648 → 1,179,648 |
| Environmental update | 576 → 144 | 2,304 → 576 | 9,216 → 2,304 |
| Environmental geometry bias | 1,867,776 → 466,944 | 53,478,144 → 13,369,536 | 855,641,088 → 213,910,272 |
| Direct query–module message | 116,736 → 116,736 | 2,228,256 → 2,228,256 | 35,651,712 → 35,651,712 |

The fourfold reduction occurs in the intended environmental update/read work; fine joint messages and QM remain. The common coarse path also remains supplied by the fine environment.

The existing `rtol=2e−5, atol=2e−6` output checks do not all pass. For Regional, 2,048-versus-128 field/internal/port tensors pass on both anchors, but interface tensors fail: maximum absolute changes are 6.79e−6/7.33e−6 and relative L2 changes 1.46e−6/1.75e−6. Independent same-2,048 repetitions also fail the interface check. Field maximum changes are 5.72e−6/8.58e−6, with relative L2 about 2.42e−7/2.53e−7.

Dense likewise has interface failures for both chunk comparisons; cached Dense 0653 also fails the field check (maximum 1.62e−5, relative L2 2.78e−7). Default Dense 0653 has field/interface failures even between independent same-chunk repetitions. The artifact retains every boolean and discrepancy; no tolerance or reference was changed. These small numerical differences and the noisy cached-Dense anchor measurement limit exact execution-equivalence claims, while the large synthetic cost differences are much larger than observed timing spread. Source: `timing/regional_vs_dense.json`.

## Anchor figures

The five new figures use saved exact-500 arrays and the measured endpoint/probe/timing artifacts. Physical field panels use temperature; the full five-channel quantitative comparison remains in the tables above. Shared-state norms and attention are descriptive, whereas the interventions supply the usefulness evidence.

![Fine environment and deterministic 48-region cover](../../diagnostics/generated/interface_operator_study/regional_response/figures/regional_cover_24x8_48.png)

Fine cells carry deterministic region IDs; stars mark weighted centroids and marker area represents regional quadrature mass. This regular grid has equal masses, 1.5 in the inherited numerical convention, for total mass 72.

![Shared regional states and receiver reads, ordinary anchor 0273](../../diagnostics/generated/interface_operator_study/regional_response/figures/regional_state_read_summary__0273.png)

![Shared regional states and receiver reads, difficult anchor 0298](../../diagnostics/generated/interface_operator_study/regional_response/figures/regional_state_read_summary__0298.png)

Ports and field probes read the same 48 P2-prepared state IDs. Heatmaps show head-averaged learned attention; the separate curves show response-state L2 norms, and bars show deterministic mass. None is labelled a physical influence map. Signed-vector dependence is assessed by the JVP and ground-truth interventions above.

![Exact-500 anchor fluid temperatures and signed errors](../../diagnostics/generated/interface_operator_study/regional_response/figures/regional_anchor_predictions_errors.png)

The four rows are 0273, 0653, 0298, and 0302. Reference, Regional, and Dense use dataset physical temperature units; error panels are prediction minus reference. Solid interiors are masked consistently using the saved fluid mask. Existing Dense arrays are reused, with the single missing 0302 array recovered as documented above.

![Sampled convergence, exact-500 accuracy and actual inference cost](../../diagnostics/generated/interface_operator_study/regional_response/figures/regional_convergence_accuracy_cost.png)

History lines use sampled logged validation field MSE through 500 with the maintained named-column reader. The accuracy/cost panel uses equal-case full-grid 90-case L2 against the median of measured full-forward anchor medians; the memory panel uses peak allocated memory. Primary Dense timing is its default projection variant. The raw table separately retains the cached variant and its variability. These logged and endpoint metrics have different definitions and are not merged into one curve.

Figure sources and exact inputs are recorded in `figures/endpoint500_figures.json` with four compact source CSVs. Generated arrays, figures, and tables remain local in the existing structured tree.

## Research recommendation and handoff

This is a **promising early result with unresolved thermal and difficult-case tradeoffs**. The model learned useful regional field reconstruction and P1 feedback, while the intended repeated environmental operations and inference memory decreased in actual execution. These observations support continuing the same scientific candidate later. They do not establish preserved mature Dense accuracy, a uniformly better physical loop, or a unique advantage over other equivalent pooling/attention formulations.

The earlier quantitative conclusions need only a scoped addition: mature Dense 1804 remains the measured 5,000-epoch reference, and the negative frozen projected-state approximation remains negative. At matched 500, native regional preparation achieves a small pooled advantage over Dense driven by v, together with a worse tail and thermal tradeoffs. This does not overturn the mature comparison. The first 500 epochs show improving learning rather than a final convergence result.

The next research action is a **user-authorized continuation of Run 1806 to 2,500**, comparing exact endpoints with the existing matched tables and tracking all five channels, M10/tail cases, P0 thermal effects, and P1/P2 usefulness. Best-by-field checkpoints should remain a separate comparison. No learning-curve extrapolation or early-score threshold determines that decision. Changes to fine source communication, QM, coarse sourcing, widths, losses, or region resolution are later structural questions.

Documented continuation command, **not executed**, from `/home/wanglz/Desktop/src/ModularDT/HONF_Proj`:

```bash
rtk env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python train.py \
  --config project://src/config_core/forward/regional_response_interface_context.json \
  --workflow forward --device cuda:0 --epochs 2500 \
  --resume-checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1806_20260910_093649_regional_response_interface/epoch_0500_model.pt \
  --yes
```

This would continue the existing managed run and optimizer rather than create a copied-checkpoint run. Milestones after 500 are configured for a later authorized continuation; none were executed here. Runs 1804 and 1805 were not resumed.

The **16 external physical-reference requests remain pending**. The development targets provide measured reconstruction errors, but model self-FD and frozen removals do not validate independent pair/triple collective physical responses. One seed, one regular 2-D region layout, globally available source/receiver relations, retained ME/M²/QM costs, and the finite-precision probe limits bound this study's conclusions.

There was no scientific departure from the one-candidate/one-run plan. Operational adjustments were confined to diagnostic role capture, prepared-allocation lifetimes, standard endpoint exports/figures, and recovery of one missing baseline anchor array. Existing training chunking, optimizer/data settings, trusted loading, historical evidence, and security behavior were preserved. No additional predictor, architecture sweep, penalty, rank/count selection, or external solver was introduced.

## Source and artifact index

Source was implemented in `878f484`; the study tooling and initial report were recorded in `0429b53`, with the diagnostic timing lifetime fix in `c958b59`. Production model arithmetic and the candidate configuration were unchanged during the formal training run. The final report/figure-helper closeout follows these commits normally on `agent/honf-core-next`.

| Item | Actual path, relative to `HONF_Proj/` |
|---|---|
| Backend | `src/honf_forward_core/interface_fields/regional_response.py` |
| Candidate profile | `src/config_core/forward/regional_response_interface_context.json` |
| Single study entry point | `tools/diagnostics/run_regional_response_study.py` |
| Run directory | `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1806_20260910_093649_regional_response_interface/` |
| Exact endpoint | Run directory + `epoch_0500_model.pt` |
| Existing run metadata / history | Run directory + `run_manifest.json` / `metrics.csv` |
| All executed study commands | `diagnostics/generated/interface_operator_study/regional_response/commands.txt` |
| Mature interventions | Study root + `diagnosis/run1804_epoch5000_interventions.json`, `run1805_epoch5000_interventions.json` |
| One frozen approximation | Study root + `frozen_coarsening/run1804_epoch5000_frozen_192_to_48.json` |
| Real optimizer steps / completion summary | Study root + `training/physical_smoke.json`, `training/training_closeout.json` |
| Full-90 tables / matched reduction | Study root + `endpoint500/tables/`, `endpoint500/reduction/` |
| Native usefulness / derivatives | Study root + `endpoint500/regional_interventions.json`, `endpoint500/regional_probes.json` |
| Actual timing / rows / numerical comparisons | Study root + `timing/regional_vs_dense.json` |
| Figures / source tables | Study root + `figures/endpoint500_figures.json` and referenced PNG/CSV files |

“Study root” is `diagnostics/generated/interface_operator_study/regional_response/`. Generated artifacts are deliberately local; source, configuration, tests, and this report follow ordinary Git history.
