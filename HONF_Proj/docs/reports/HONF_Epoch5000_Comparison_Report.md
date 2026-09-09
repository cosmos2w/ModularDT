# HONF comparison at epoch 5,000

Date: 2026-09-09. The exact stored epoch-5,000 checkpoints of Legacy 1401, Dense 1804, and Geometry Reader 1805 were evaluated on the established 90-case development holdout. **Dense regains the overall field lead: its pooled normalized fluid relative L2 is 0.029661, versus Legacy 0.037401 and Reader 0.065865.** Dense is 20.70% lower than Legacy and now leads most thermal quantities as well. Reader remains substantially behind both on overall reconstruction.

This completes the requested comparison after the two authorized continuations. It follows the [epoch-2,500 comparison](HONF_Epoch2500_Comparison_Report.md), [500-epoch recovery study](HONF_Group_Reader_Recovery_Report.md), and [original interface study](HONF_Interface_Study_Report.md). Historical measurements are preserved. No new training, continuation, sweep, or monitoring task was launched.

## Changes to earlier conclusions

| Earlier conclusion | Exact epoch-5,000 result | Revision |
|---|---|---|
| At 2,500, Legacy overtakes Dense in combined field error because of temperature. | Dense now leads combined field error and temperature, with relative L2 0.029661 and 0.032404 respectively. | The 2,500 ranking was checkpoint-dependent. Dense is the strongest overall field candidate at the completed 5,000-epoch budget. |
| At 2,500, Dense leads all four hydrodynamic field channels. | Dense leads u, v, and pressure; Legacy has slightly lower vorticity error, 0.041203 versus 0.041888. | Retain Dense's broad hydrodynamic advantage, but remove the claim that it wins every hydrodynamic channel at this endpoint. |
| Legacy leads internal/surface temperature and outlet/module-temperature KPIs at 2,500. | Dense now leads all four. Reader is worst in internal and surface temperature. | Dense's later thermal improvement changes the reference model for these quantities. |
| Reader has the lowest normal heat-flux relative L2 at 2,500. | Dense 0.103487, Reader 0.105010, Legacy 0.131790. | Reader remains competitive on flux, but no longer has the lowest pooled error. |
| Reader's relative field deficit is growing despite active group reads. | Its excess relative L2 reaches 122.06% above Dense and 76.10% above Legacy; group context remains active. | This conclusion strengthens. Numerical activity does not establish competitive field reconstruction or beneficial causal group use. |

The earlier 500-epoch improvement over collapsed Run 1802, including 85/90 improved cases, remains a 500-only comparison. No later Run-1802 checkpoint was evaluated. Likewise, the phase-specific P0/P1/P2 interventions and support-derivative results in the recovery report remain evidence at their original checkpoints.

## Field reconstruction over the three evaluated epochs

All errors are lower-is-better. Epoch matching does not imply equal wall-clock or parameter budgets.

| Model | Fluid MSE @500 | Fluid MSE @2,500 | Fluid MSE @5,000 | Fluid relative L2 @500 | @2,500 | @5,000 | L2 reduction, 2,500→5,000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy 1401 | 0.01270523 | 0.00189854 | 0.00129505 | 0.117148 | 0.045285 | 0.037401 | 17.41% |
| Dense 1804 | 0.00902626 | 0.00220791 | **0.00081448** | 0.098741 | 0.048835 | **0.029661** | 39.26% |
| Reader 1805 | 0.01805446 | 0.00515218 | 0.00401630 | 0.139648 | 0.074600 | 0.065865 | 11.71% |

![Measured epoch-500, epoch-2500, and epoch-5000 field errors with paired endpoint cases](../../diagnostics/generated/interface_operator_study/epoch5000_comparison/figures/reader_epoch500_epoch2500_epoch5000_comparison.png)

The figure connects only the three measured checkpoints. At 5,000, Dense beats Legacy on 80/90 cases, compared with 30/90 at 2,500. Reader beats Dense on 0/90, compared with 12/90 at 2,500. The new overall ranking is therefore also supported by the paired case results.

Reader's relative L2 excess over Dense progresses from 41.43% at 500 to 52.76% at 2,500 and 122.06% at 5,000. Its absolute gap to Dense falls from 0.04091 to 0.02576, then grows to 0.03620. Against Legacy, the relative excess is 19.21%, 64.73%, and 76.10%, while the absolute gap is 0.02250, 0.02932, and 0.02846. Thus all three improve their pooled score, but Reader's improvement after 2,500 is much slower than Dense's.

| Epoch-5,000 field relative L2 | Legacy | Dense | Reader |
|---|---:|---:|---:|
| Equal-case mean | 0.034705 | **0.026433** | 0.054565 |
| Equal-case median | 0.031616 | **0.022646** | 0.039926 |
| Equal-case p95 | 0.060236 | **0.052161** | 0.130381 |
| Worst case | 0.080615 | **0.064715** | 0.177697 |
| Near-interface pooled | **0.033693** | 0.035086 | 0.047437 |
| Far-fluid pooled | 0.041778 | **0.026495** | 0.063757 |

The near-interface exception matters: Dense's overall win does not extend to the pooled near-interface score. Dense does have lower equal-case near-interface mean/median errors (0.027639/0.021420 versus Legacy 0.028994/0.023431), so the aggregation distinction matters here. Reader's global median improves materially from 0.057759 at 2,500, but its p95 barely changes from 0.133194 to 0.130381 and its worst-case error rises. A persistent difficult-case tail remains.

| Paired candidate minus comparator at 5,000 | Candidate wins / 90 | Mean field L2 delta | Median delta | Delta p05 | Delta p95 |
|---|---:|---:|---:|---:|---:|
| Dense − Legacy | 80 | −0.008271 | −0.008460 | −0.017428 | +0.001313 |
| Reader − Dense | 0 | +0.028132 | +0.018718 | +0.012033 | +0.086046 |
| Reader − Legacy | 2 | +0.019860 | +0.010374 | +0.001869 | +0.079261 |

Relative to each model's own 2,500 checkpoint, Legacy improves on 82/90 cases, Dense on 89/90, and Reader on 83/90. Relative to epoch 500, Legacy and Dense improve on all 90, while Reader improves on 89/90. These are per-case field-L2 comparisons; they do not imply every physical quantity improves.

### Predefined physical strata

Dense leads the equal-case mean global field relative L2 in all 13 predefined strata, recovering the breadth of its epoch-500 lead after leading only the 10-module stratum at 2,500. Reader trails both comparators in every stratum. The table retains the established thresholds; categories across different dimensions overlap.

| Stratum | Cases | Legacy @5,000 | Dense @5,000 | Reader @5,000 |
|---|---:|---:|---:|---:|
| 3 modules | 25 | 0.027164 | 0.017419 | 0.034396 |
| 5 modules | 25 | 0.034522 | 0.027906 | 0.057934 |
| 7 modules | 25 | 0.043460 | 0.034891 | 0.078798 |
| 10 modules | 15 | 0.032986 | 0.024906 | 0.042178 |
| Spacing <1 radius | 45 | 0.037423 | 0.030063 | 0.059688 |
| Spacing 1–2.5 radii | 30 | 0.034278 | 0.025155 | 0.055605 |
| Spacing ≥2.5 radii | 15 | 0.027404 | 0.018102 | 0.037119 |
| Wall distance <1.5 radii | 28 | 0.037230 | 0.028777 | 0.057203 |
| Wall distance 1.5–2.5 radii | 50 | 0.035837 | 0.026968 | 0.057373 |
| Wall distance ≥2.5 radii | 12 | 0.024094 | 0.018737 | 0.036711 |
| Heating CV <0.25 | 28 | 0.032656 | 0.023527 | 0.047254 |
| Heating CV 0.25–0.35 | 42 | 0.032540 | 0.024770 | 0.048505 |
| Heating CV ≥0.35 | 20 | 0.042119 | 0.033996 | 0.077527 |

Reader's particularly high errors in the 7-module and high-heating-heterogeneity strata persist. Its better 10-module result means these data still do not support a simple monotonic failure with increasing module count.

## Physical field channels and coupled quantities

| Physical fluid channel, pooled relative L2 @5,000 | Legacy | Dense | Reader |
|---|---:|---:|---:|
| u | 0.012064 | **0.008143** | 0.016446 |
| v | 0.017297 | **0.013121** | 0.059069 |
| Pressure | 0.045904 | **0.023883** | 0.077057 |
| Vorticity | **0.041203** | 0.041888 | 0.066364 |
| Temperature | 0.041622 | **0.032404** | 0.068220 |

Dense's fluid-temperature relative L2 falls from 0.067630 at 2,500 to 0.032404 at 5,000. The poor 2,500 temperature endpoint is therefore not a persistent late-training outcome. Legacy's pressure error slightly increases from 0.045065 to 0.045904; improvement is not uniform across channels.

The Dense-minus-Legacy normalized MSE difference changes from +0.000309366 at 2,500 to −0.000480571 at 5,000. At 5,000, channel contributions are −0.000045772 (u), −0.000028208 (v), −0.000248819 (pressure), +0.000012445 (vorticity), and −0.000170218 (temperature). Pressure and temperature supply the largest arithmetic advantages. The contributions sum to the global MSE difference; this is a raw-error decomposition, not causal attribution to an architecture component.

| Physical pooled relative L2 @5,000 | Legacy | Dense | Reader |
|---|---:|---:|---:|
| Provisional outside temperature | 0.226292 | **0.066390** | 0.076243 |
| Final outside temperature | **0.063595** | 0.073197 | 0.081575 |
| Provisional effective h | 0.513136 | **0.178643** | 0.234237 |
| Final effective h | 0.050224 | **0.046379** | 0.048912 |
| Internal temperature | 0.034742 | **0.025744** | 0.044122 |
| Interface surface temperature | 0.045998 | **0.037510** | 0.059751 |
| Interface normal heat flux | 0.131790 | **0.103487** | 0.105010 |

| Physical KPI, equal-case mean absolute error @5,000 | Legacy | Dense | Reader |
|---|---:|---:|---:|
| Inlet-minus-outlet pressure drop | 0.003042 | **0.001031** | 0.002140 |
| Mean outlet temperature | 0.261771 | **0.155314** | 0.275935 |
| Mean active-module temperature | 0.237370 | **0.113180** | 0.216052 |

Reader's heat-flux relative L2 is only 1.47% above Dense and 20.32% below Legacy. This is a specific strength, not a general thermal or field-reconstruction win. Its internal-temperature relative L2 changes only from 0.045564 to 0.044122 after 2,500, while mean-module-temperature MAE worsens from 0.205688 to 0.216052. Dense's mean-module-temperature MAE falls from 0.612714 to 0.113180. Legacy's internal/surface temperature errors and outlet/module-temperature MAEs worsen between the two endpoints despite its improved global field score.

Port accuracy also does not track field accuracy uniformly. Dense's final outside-temperature relative L2 worsens from 0.047728 at 2,500 to 0.073197 at 5,000, losing the lead to Legacy. Its provisional outside-temperature error also rises, from 0.048614 to 0.066390, while effective-h prediction improves. These observations warrant separate physical quantities rather than a single overall success label.

The earlier Reader advantage in equal-case final-h MAE also disappears: Dense now has 0.422273, Reader 0.459893, and Legacy 0.474802. Some metric-dependent exceptions remain in outside-temperature MAE: Reader is slightly below Dense for provisional/final mean MAE (0.675335/0.775083 versus 0.682447/0.784722), even though its pooled relative L2 is higher for both. The complete distributions are retained in the reduction tables.

## Established anchors

| Anchor, field relative L2 @5,000 | Legacy | Dense | Reader |
|---|---:|---:|---:|
| 0273 | 0.022920 | **0.015654** | 0.029223 |
| 0653 | 0.022296 | **0.021157** | 0.038247 |
| 0298 | **0.063619** | 0.064715 | 0.177697 |
| 0302 | **0.040667** | 0.055464 | 0.122322 |

The four anchors no longer have one shared Legacy/Dense ordering. Reader remains worst on all four. Its errors on 0298 and 0302 increase from 0.150329 and 0.118661 at 2,500, even as its overall score improves. Twelve debug NPZs preserve ordinary predictions and references for these cases; no new phase intervention is implied by those exports.

## Group activity and the remaining scientific question

Across the 90 complete field-query grids, Reader's equal-case mean P2 main-context norm grows from 6.7245 at 500 to 11.0303 at 2,500 and 14.7888 at 5,000. At 5,000, the main/coarse/local mean branch-norm fractions are 0.4114/0.4926/0.0960. Mean non-null read mass remains 0.658334, as expected from the geometry-envelope construction on unchanged case geometry. These are query-context summaries, not fluid-only accuracy scores or padding-contaminated port proxies.

| Reader epoch-5,000 logged norm | Group preparation | Group reception | Coarse | Local |
|---|---:|---:|---:|---:|
| Pre-clip gradient | 0.0106324 | 0.00198478 | 0.00824312 | 0.00616422 |
| Parameter update | 0.0149906 | 0.00865985 | 0.0163458 | 0.00482191 |

The reader has not reverted to a vanishing numerical context. The unresolved question is whether its learned group information improves the field and physical coupling at this trained endpoint, especially on the difficult anchors. New phase-specific ground-truth interventions would be needed to answer that. This comparison does not remeasure P0/P1/P2 causal effects, connected/disconnected influence, or support-transition derivatives.

## Training-history integrity and endpoint sensitivity

All three histories contain exactly one row for each epoch 1–5,000, without missing or duplicate epochs. Run 1401 consistently has 220 columns and Run 1805 has 324. Run 1804 retains its known schema break: the first 500 rows and header have 286 columns, while epochs 501–5,000 have 324 columns. The audit maps those continued rows to the continuation schema; the original CSVs are preserved. Applying the first header to the whole Run-1804 file would still mislabel validation and diagnostic fields.

| Corrected logged validation MSE | Legacy field | Dense field | Reader field | Legacy temperature | Dense temperature | Reader temperature |
|---|---:|---:|---:|---:|---:|---:|
| Epoch 2,500 | 0.00302927 | 0.00388332 | 0.00713449 | 0.00325326 | 0.00702663 | 0.00757319 |
| Epoch 4,000 | 0.00214614 | 0.00188914 | 0.00609615 | 0.00236354 | 0.00165367 | 0.00690030 |
| Epoch 5,000 | 0.00193250 | 0.00171428 | 0.00555277 | 0.00250067 | 0.00168140 | 0.00686848 |

The history supports a later Dense recovery rather than a persistent temperature disadvantage. However, best-checkpoint selection still matters: the best logged validation field MSE through 5,000 is 0.00150997 for Legacy at 4,585 and 0.00152102 for Dense at 4,738, a near tie with Legacy about 0.7% lower. Reader's best is 0.00536316 at 4,777. Best logged validation temperature MSE favors Dense: 0.00140534 at 4,871, versus Legacy 0.00190262 at 4,979 and Reader 0.00653300 at 4,779.

These logged validation metrics use the training evaluator's sampling/masking definitions and are not numerically interchangeable with the full-grid, fluid-only endpoint scores. No `best` checkpoint was substituted into the matched comparison, and no new best-selected full-holdout comparison was performed. The strongest defensible conclusion is Dense's lead at the exact completed budget, with checkpoint-selection sensitivity retained as a qualification.

## Scope, reproducibility, and recommendation

The primary comparison uses exact `epoch_5000_model.pt` files, predicted port conditions, checkpoint-owned normalization, and the existing `evaluate.py --workflow compare`. Full 8,192-point grids are evaluated for each of the same 90 development cases; the primary field errors retain the fluid mask. Near-interface is nearest module-surface distance 0–0.25 and far-fluid is distance at least 1.0, in dataset coordinate units. Pooled errors reduce raw squared-error and target-squared-norm sums; physical channel/interface errors use denormalized quantities. Physical MAEs retain the dataset's units and are never pooled across unlike units.

All nine model/epoch datasets share the same 90 unique case IDs. The reduction passed 45,360 per-case target/count checks and 135 metric/model/epoch pooled-score reconciliations, including global, near/far, physical fields, ports, and interface quantities. Global normalized field pools use 3,496,800 scalar fluid values and target squared norm 3,237,304.06052. The original 500/2,500 tables were reused without alteration or copied input snapshots.

The evaluation completed successfully on physical GPU 0. Outer inference query batching was 32,768 and receiver chunking was 128, matching the prior comparison settings. No model, loss, optimizer, training chunk, security, or trusted-loading behavior changed. This analysis adds no execution optimization or new inference benchmark.

All three exact epoch-5,000 checkpoints loaded with the trusted CPU loader and retain optimizer state, optimizer-group inventory, normalization statistics, and the existing dataset configuration. Dense declares `dense_pairwise_field`; Reader declares `sparse_interface_honf` with `geometry_envelope_attention`; Legacy retains its historical `enhanced_honf_pairwise` decoder. The training configuration's 1,024 sampled field queries and batch size 48 are distinct from full-grid evaluation.

The 16 external physical-reference requests remain `reference_verification_pending`. These are reconstruction scores against existing dataset references, not newly obtained independent physical-solver validation. There is one training seed and an already-used development holdout; descriptive case wins do not establish uncertainty across independent training runs.

The artifact root is `HONF_Proj/diagnostics/generated/interface_operator_study/epoch5000_comparison/`. Exact executed commands are in `commands.txt`; evaluator arguments in `evaluation_argv.json`; completion output in `evaluation.log`; standard raw tables in `evaluation/tables/`; and anchor arrays in `evaluation/debug_npz/`. `summarize_reader_context.py` produces `reader_context_summary.json`. Generated evidence remains local under the existing Git ignore policy; this report retains the conclusions in Git.

The CPU history/checkpoint audit is reproducible with `analyze_epoch5000_metrics.py`; its outputs are `epoch5000_metrics_comparison.json` and `analysis.md` under that root.

`summarize_epoch5000.py` reuses the earlier reducer's metric helpers and writes `reduction/comparison_tables.md`, `reduction/reduction.json`, and `reduction/status.json`. Full values are in `reduction/metrics_long.csv`; paired evidence in `reduction/paired_case_deltas.csv` and `reduction/paired_summary.csv`; strata in `reduction/strata_long.csv`; and the channel accounting in `reduction/normalized_mse_channel_contributions.csv`. The input manifest references original tables. `case_set_validation.csv`, `target_count_validation.csv`, and `raw_pool_reconciliation.csv` in the same directory preserve the QA results.

The compact figure is `figures/reader_epoch500_epoch2500_epoch5000_comparison.png`, with renderer `figures/render_reader_epoch_comparison.py` and computed source tables `figures/reader_epoch_progress_source.csv` and `figures/reader_epoch5000_pairs_source.csv`.

Exact checkpoints, relative to `HONF_Proj/`:

```text
Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/epoch_5000_model.pt
Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/epoch_5000_model.pt
Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1805_20260908_002326_sparse_interface_geometry_read/epoch_5000_model.pt
```

**Use Dense as the current overall reconstruction comparator at the completed 5,000-epoch budget, while retaining Legacy's near-interface, vorticity, and final outside-temperature strengths.** Reader's numerical recovery is sustained, but additional epochs have not closed its field deficit. The next useful research step is a bounded phase-specific study of the 5,000-epoch Reader, including the deteriorating anchors 0298/0302, before choosing a group-first coarse redesign or more training.
