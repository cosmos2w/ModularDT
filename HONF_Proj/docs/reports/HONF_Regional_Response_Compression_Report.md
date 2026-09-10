# HONF regional response compression

This study is in progress. It measures mature component/phase reliance, tests one frozen projected-state coarsening, and trains one native regional-response model through epoch 500. Results below will distinguish those experiments rather than treating frozen coarsening as equivalent to the trainable model.

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

Source: `diagnostics/generated/interface_operator_study/regional_response/diagnosis/run1804_epoch5000_interventions.json`. These are new mature reliance measurements, distinct from the earlier epoch-500 Reader study and from the future native candidate endpoint.

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

The diagnostic records instrumented read/forward wall times and memory. These single forwards include synchronization and context capture, with a cold first reference; they are not the controlled timing comparison. The separate repeated timing section will supply efficiency conclusions. Frozen coarsening and native pre-update grouping remain distinct experiments; no alternate grouping was searched after these negative results.

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

## Execution record and pending measurements

The single generated study root is `diagnostics/generated/interface_operator_study/regional_response/`, with `diagnosis/`, `frozen_coarsening/`, `training/`, `endpoint500/`, `timing/`, and `figures/`. Formal checkpoints remain in the ordinary managed run directory. Existing comparator tables and arrays are referenced in place.

The literal executed study commands are recorded in [commands.txt](../../diagnostics/generated/interface_operator_study/regional_response/commands.txt), including all three mature-checkpoint calls. Commands below use the project directory as their working directory.

The native implementation was committed and pushed as `878f484`. Resource resolution and a recursive comparison against Run 1804's actual resolved configuration found no unexpected scientific/training differences. Expected differences are the architecture and region setting, run identity/name, and 500 versus the continued reference's 5,000 epochs. Run 1804's maintained profile ID was 1800 while its allocated run ID was 1804; that historical distinction is preserved in `training/config_profile_comparison.json`.

The complete CPU test suite passed **403 tests**, with three CUDA-specific tests skipped in that invocation (`training/source_tests_closeout.log`). Focused coverage includes weighted mass/centroid preservation, permutation/padding and split-weight duplicates, unequal-mass constant-within-region attention, singleton Dense/native forward and parameter/input gradients, live source caching, component/role isolation, and physical-wrapper integration. An earlier test invocation caught the study capture wrapper using the wrong nesting; both execution and its regression test were corrected before closeout. Actual frozen outputs include all intended role contexts and the correct active-port counts.

After completing the endpoint tools, the full CPU suite passed **407 tests**, with the same three CUDA skips, in 17.64 s (`training/source_tests_endpoint_tools_default_threads.log`). An optional `OMP_NUM_THREADS=1` invocation failed the pre-existing bitwise historical-background regression while 406 tests passed. The ordinary thread configuration passed without changing production code, test references, or tolerances; the failed invocation is retained in `training/source_tests_endpoint_tools.log`.

Before formal training, a fresh disposable candidate executed two canonical optimizer steps with predicted ports, frozen Stage A, physical refinement, all inherited losses, backward, clipping, and the established optimizer builder. Neither model nor optimizer state was saved or reused for the managed run.

| Real batch | Cases × queries | Modules per case | Wall time (s) | Peak allocated MiB | Peak reserved MiB | Regional preparation / reader gradient norms | Regional preparation / reader update norms |
|---|---|---:|---:|---:|---:|---|---|
| Small | 48 × 1,024 | 1 | 2.186 | 3,701.15 | 4,416 | 46.827 / 33.726 | 0.18928 / 0.16900 |
| Large | 48 × 1,024 | 12 | 3.009 | 23,058.20 | 26,764 | 19.758 / 18.556 | 0.12079 / 0.10581 |

Both steps produced finite parameters and applied updates. Gradient norms are pre-clipping observations; the clip scales were 0.002060 and 0.001812 with the unchanged norm limit 1. These initial losses and gradients establish real physical execution, not trained accuracy. The first step includes lazy materialization and cold execution; the two step times are operational observations rather than a controlled architecture speed comparison. Canonical diagnostics retain regional preparation (`em_message` and `env_update`), regional reader (query, attention, and geometry blocks), direct-module, coarse, and local groups separately.

The existing allocator created `Run_1806_20260910_093649_regional_response_interface` on physical GPU 0. Its single from-scratch launch requests exactly 500 epochs. The run remains in progress at this writing. Trusted CPU loading of its epoch-10 checkpoint confirms **4,395,409 trainable parameters**, matching Dense, with no uninitialized parameters left. The setup log's smaller 3,356,945 count precedes lazy materialization and is not the complete model count.

The ordinary run manifest records its start at **2026-09-10 13:36:49 UTC** (09:36:49 local time). Sampled milestone observations from the live named-column CSV are:

| Epoch | Validation field MSE | Validation total loss | Regional preparation / reader gradient norms | Regional preparation / reader update norms |
|---|---:|---:|---|---|
| 10 | 0.518891 | 1.582103 | 1.11048 / 1.78146 | 0.02638 / 0.03507 |
| 50 | 0.203376 | 0.392818 | 0.14661 / 0.59829 | 0.03992 / 0.03915 |
| 100 | 0.093828 | 0.186004 | 0.08950 / 1.00067 | 0.02309 / 0.01469 |

These sampled validation metrics and evolving-batch gradients are distinct from exact-endpoint full-grid error and fixed-checkpoint intervention results.

Commands executed from `HONF_Proj/`:

```bash
rtk proxy bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python -u tools/diagnostics/run_regional_response_study.py smoke --device cuda:0 --output diagnostics/generated/interface_operator_study/regional_response/training/physical_smoke.json > diagnostics/generated/interface_operator_study/regional_response/training/physical_smoke.log 2>&1'
rtk proxy bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py --config project://src/config_core/forward/regional_response_interface_context.json --workflow forward --device cuda:0 --epochs 500 --run-id 1806 --run-name regional_response_interface --yes > diagnostics/generated/interface_operator_study/regional_response/training/training_launch.log 2>&1'
```

The exact-500 assessment will use the same 90-case development holdout, full grids, checkpoint-owned normalization, predicted ports, and canonical physical metrics. Near/far field masks retain surface distances 0–0.25 and at least 1.0 in dataset coordinate units, distinct from radius-normalized layout strata. Pooled and equal-case metrics will remain separate.

Dense's existing 90-case scalar tables are reused. Its saved exact-500 anchor arrays for 0273/0653/0298 are reused from the earlier `Stage2_Run1401_1804_1801_1802_Epoch500_20Case/debug_npz/` directory. The missing 0302 array was reconstructed with the standard evaluator on CPU, for that one case only, in `endpoint500/dense_anchor_0302/`. Its fluid relative L2 differs from the existing GPU scalar table by −1.22e−8; internal and final outside-temperature L2 differences are −1.86e−8 and −4.30e−8. This fills a missing figure input without replacing the matched baseline tables or copying existing arrays.

New timing uses identical runtime settings for Dense and Regional. Source-projection caching is labelled separately from model compression, and the previously measured 128→2,048 inference chunk improvement is not claimed as a new gain. Synthetic tests use complete even rectangular grids for both models at the requested M/E/Q sizes; their measurements concern execution, not physical accuracy.

The existing 16 external physical-reference requests remain pending. No model self-derivative check substitutes for independent physical-response validation. The formal candidate is an early, one-seed development assessment; its epoch-500 score alone will not be treated as a final convergence verdict.
