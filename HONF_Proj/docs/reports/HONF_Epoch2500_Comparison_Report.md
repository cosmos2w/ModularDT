# HONF comparison at epoch 2,500

Date: 2026-09-09. Exact stored epoch-2,500 checkpoints for Legacy 1401, Dense 1804, and Geometry Reader 1805 were evaluated on the same 90-case development holdout. **The overall field ranking changes from Dense < Legacy < Reader at epoch 500 to Legacy < Dense < Reader at epoch 2,500.** Dense still leads on all four hydrodynamic field channels; Legacy's temperature advantage determines the combined normalized-field ranking. Reader improves substantially but has not caught either comparator.

This is a follow-up to the [interface study](HONF_Interface_Study_Report.md) and [group-reader recovery study](HONF_Group_Reader_Recovery_Report.md). Their historical measurements remain intact. The authorized Run-1804/GPU-0 and Run-1805/GPU-1 continuations were left unchanged. No epoch-5,000 comparison, new training, or additional continuation was launched for this analysis.

## What changes in the earlier conclusions

| Earlier conclusion and scope | Epoch-2,500 evidence | Updated interpretation |
|---|---|---|
| Dense is the strongest field candidate at epoch 500. | Legacy has 7.27% lower pooled field relative L2 than Dense and wins 60/90 paired cases. | Still correct at 500; no longer the overall ranking at 2,500. Do not extrapolate Dense's early lead into a general long-training advantage. |
| Reader improves over collapsed Run 1802 but trails Dense and Legacy at 500. | Reader trails Dense by 52.76% and Legacy by 64.73% in pooled relative L2, versus 41.43% and 19.21% at 500. | The relative deficit grows. Its absolute L2 gap to Dense shrinks from 0.04091 to 0.02576; its gap to Legacy grows from 0.02250 to 0.02932. The original 85/90 wins over Run 1802 remain a 500-only result; no Run-1802 comparison at 2,500 was made. |
| Reader has the best internal/surface temperature and thermal KPI results among these three at 500. | Legacy now leads those metrics; Reader remains better than Dense. | Reader's early thermal advantage is not a persistent advantage over Legacy. Reader now has the best pooled normal heat-flux relative L2. |
| Geometry-envelope reading restores numerical group activity; phase interventions establish specific usefulness at 500. | Group context and group-specific training gradients remain active at 2,500, despite the accuracy deficit. | Numerical activity persists. The earlier P0/P1/P2 causal effects are not measurements of the new checkpoint and cannot be extrapolated. |

## Evaluation and evidence quality

The existing `evaluate.py --workflow compare` evaluated three checkpoints × 90 cases, using the test split at ratio 1.0, predicted port conditions, checkpoint-owned normalization, and the established physical reconstruction metrics. Each case uses its full 8,192-point query grid; the primary field score masks out solids. All six model/epoch datasets, including reused epoch-500 tables, have the same 90 unique case IDs. The CPU reduction passed 15,120 per-case target/count checks and six raw pooled-score reconciliations. Each global pool contains 3,496,800 scalar fluid values and normalized target squared norm 3,237,304.06052.

Pooled MSE is summed normalized squared error divided by the scalar count. Pooled relative L2 is the square root of summed squared error divided by summed target squared norm. Equal-case statistics are separately labeled; they are not substitutes for pooled scores. Physical channel and interface scores use denormalized quantities; physical MAEs retain the dataset's units. No aggregate of unlike physical units is constructed.

Trusted CPU checkpoint loading verified epoch/current-epoch 2,500, optimizer state, normalization statistics, and matching existing dataset identity. No current `best` checkpoint was substituted. Evaluation completed with exit 0 on physical GPU 1, with only the evaluation process capped at 3% CUDA allocator memory; peak allocated memory was 384.25 MiB. Receiver chunking remained 128 and outer inference query batching was 32,768. No training configuration or chunking changed. Shared-GPU elapsed times are not treated as inference benchmarks or evidence of new execution gains.

## Field reconstruction

All errors below are lower-is-better. Every model improves its own per-case global relative L2 on all 90 cases between the two evaluated epochs.

| Model | MSE @500 | MSE @2,500 | Relative L2 @500 | Relative L2 @2,500 | L2 reduction from own 500 |
|---|---:|---:|---:|---:|---:|
| Legacy 1401 | 0.01270523 | **0.00189854** | 0.117148 | **0.045285** | 61.34% |
| Dense 1804 | 0.00902626 | 0.00220791 | 0.098741 | 0.048835 | 50.54% |
| Reader 1805 | 0.01805446 | 0.00515218 | 0.139648 | 0.074600 | 46.58% |

![Measured epoch-500 and epoch-2500 field errors and paired epoch-2500 cases](../../diagnostics/generated/interface_operator_study/epoch2500_comparison/figures/reader_epoch500_epoch2500_comparison.png)

The figure connects only the two measured endpoints; connecting lines are guides, not intervening evaluations. Epoch counts are matched, not wall-clock budgets. Generated artifacts are retained locally under the study path below.

| Epoch-2,500 field relative L2 | Legacy | Dense | Reader |
|---|---:|---:|---:|
| Equal-case mean | 0.044643 | 0.048109 | 0.067766 |
| Equal-case median | 0.041110 | 0.046054 | 0.057759 |
| Equal-case p95 | 0.062338 | 0.066788 | 0.133194 |
| Worst case | 0.069210 | 0.086315 | 0.153652 |
| Near-interface pooled | 0.042041 | 0.045620 | 0.053073 |
| Far-fluid pooled | 0.051231 | 0.052891 | 0.073916 |

Near-interface means nearest module-surface distance 0–0.25 and far-fluid means distance at least 1.0, in dataset coordinate units. Reader's p95 is nearly twice Dense's; its aggregate deficit is accompanied by a substantial difficult-case tail.

| Paired candidate minus comparator, epoch 2,500 | Candidate lower / 90 | Mean L2 delta | Median delta | Delta p05 | Delta p95 |
|---|---:|---:|---:|---:|---:|
| Dense − Legacy | 30 | +0.003465 | +0.004028 | −0.007999 | +0.014932 |
| Reader − Dense | 12 | +0.019658 | +0.011889 | −0.003991 | +0.071479 |
| Reader − Legacy | 4 | +0.023123 | +0.017689 | +0.001004 | +0.077700 |

These are descriptive paired differences, not independent training-seed uncertainty estimates.

### Physical channels explain the aggregate arithmetic

| Physical fluid channel, pooled relative L2 @2,500 | Legacy | Dense | Reader |
|---|---:|---:|---:|
| u | 0.017823 | **0.012428** | 0.019392 |
| v | 0.027785 | **0.023961** | 0.071145 |
| Pressure | 0.045065 | **0.037814** | 0.085877 |
| Vorticity | 0.054427 | **0.053198** | 0.079259 |
| Temperature | **0.047679** | 0.067630 | 0.070601 |

For the normalized global MSE, the Dense-minus-Legacy channel contributions are −0.000094285 (u), −0.000043935 (v), −0.000097293 (pressure), −0.000028929 (vorticity), and +0.000573808 (temperature). They sum to the observed **+0.000309366** global MSE difference. Dense's combined four-channel advantage of −0.000264442 is outweighed by its temperature term. This is an exact error decomposition, not proof of a causal architectural mechanism.

### Physical strata and anchors

The table uses equal-case mean field relative L2. These are the existing 13 strata, with no new thresholds; categories across different dimensions overlap.

| Stratum | Cases | Legacy @2,500 | Dense @2,500 | Reader @2,500 |
|---|---:|---:|---:|---:|
| 3 modules | 25 | 0.043912 | 0.046951 | 0.048902 |
| 5 modules | 25 | 0.045094 | 0.049938 | 0.071205 |
| 7 modules | 25 | 0.047732 | 0.053847 | 0.088278 |
| 10 modules | 15 | 0.039964 | **0.037426** | 0.059291 |
| Spacing <1 radius | 45 | 0.044263 | 0.048574 | 0.073208 |
| Spacing 1–2.5 radii | 30 | 0.045997 | 0.048689 | 0.067888 |
| Spacing ≥2.5 radii | 15 | 0.043076 | 0.045552 | 0.051200 |
| Wall distance <1.5 radii | 28 | 0.046380 | 0.049471 | 0.069282 |
| Wall distance 1.5–2.5 radii | 50 | 0.044567 | 0.047832 | 0.070739 |
| Wall distance ≥2.5 radii | 12 | 0.040911 | 0.046086 | 0.051843 |
| Heating CV <0.25 | 28 | 0.045213 | 0.048771 | 0.062006 |
| Heating CV 0.25–0.35 | 42 | 0.042130 | 0.045573 | 0.062824 |
| Heating CV ≥0.35 | 20 | 0.049125 | 0.052508 | 0.086209 |

Dense's earlier lead in all 13 strata becomes a lead in only the 10-module stratum. Reader trails both in every stratum. The results do not support a simple monotonic module-count failure: Reader's 7-module group is worse than its 10-module group.

| Established anchor, field relative L2 @2,500 | Legacy | Dense | Reader |
|---|---:|---:|---:|
| 0273 | 0.040166 | 0.042563 | 0.044147 |
| 0653 | 0.038429 | 0.044710 | 0.056439 |
| 0298 | 0.068032 | 0.086315 | 0.150329 |
| 0302 | 0.053961 | 0.068520 | 0.118661 |

All four preserve the aggregate ordering. Twelve debug NPZs retain predictions and references for later spatial inspection. These are ordinary predictions, not new interventions.

## Ports, module temperatures, flux, and KPIs

| Physical pooled relative L2 @2,500 | Legacy | Dense | Reader |
|---|---:|---:|---:|
| Provisional outside temperature | 0.262114 | **0.048614** | 0.075339 |
| Final outside temperature | 0.072683 | **0.047728** | 0.081892 |
| Provisional effective h | 0.532423 | **0.352549** | 0.386362 |
| Final effective h | 0.048448 | **0.047681** | 0.049295 |
| Internal temperature | **0.031475** | 0.050550 | 0.045564 |
| Interface surface temperature | **0.043196** | 0.066299 | 0.060898 |
| Interface normal heat flux | 0.155332 | 0.151153 | **0.145923** |

| Physical KPI, equal-case mean absolute error @2,500 | Legacy | Dense | Reader |
|---|---:|---:|---:|
| Inlet-minus-outlet pressure drop | 0.003500 | **0.001672** | 0.003152 |
| Mean outlet temperature | **0.204968** | 0.748621 | 0.296200 |
| Mean active-module temperature | **0.143110** | 0.612714 | 0.205688 |

Dense's strongest port/pressure results coexist with worse internal/module and outlet temperatures than Legacy. Reader's normal heat-flux relative L2 is 3.46% lower than Dense's and 6.06% lower than Legacy's. This is a useful measured exception to its overall deficit, not a general field-reconstruction win. Final-h rankings also depend on the metric: Reader has the lowest equal-case MAE (0.48030 versus Dense 0.49326 and Legacy 0.50607), whereas Dense leads pooled relative L2.

Progress is not uniform across quantities. Reader internal-temperature relative L2 changes only from 0.046182 to 0.045564 (1.34% improvement), while its mean-module-temperature MAE increases from 0.189009 to 0.205688 (8.82%). Dense's corresponding MAE changes from 0.605719 to 0.612714. Final-h pooled relative L2 worsens from 0.041631 to 0.047681 for Dense and 0.042249 to 0.048448 for Legacy, while Reader improves slightly from 0.050515 to 0.049295. Full 500/2,500 physical MAE distributions and paired deltas are in the reduction artifacts.

## Training-log correction and checkpoint sensitivity

**Run 1804 has a mixed-width CSV after resume.** Its header and epochs 1–500 contain 286 columns, but epochs 501–2,500 contain 324 columns. Applying the old header to continued rows shifts later fields: the apparent epoch-2,500 `val_field_mse=0.0201974` and `val_temperature_mse=0` are parsing artifacts. The 324-column continuation schema, checked against the current writer/Run-1805 header and epoch log, gives `val_loss_total=0.01897825`, `val_field_mse=0.00388332`, and `val_temperature_mse=0.00702663`. The live CSV was not edited while training continued. Later analysis must use the schema-aware reader, not the first header alone.

All three histories have exactly one row for every epoch 1–2,500. Run 1401 consistently uses 220 columns and Run 1805 uses 324. Correctly decoded epoch-2,500 training/validation values are:

| Model | Train field MSE | Validation field MSE | Train temperature MSE | Validation temperature MSE |
|---|---:|---:|---:|---:|
| Legacy | 0.00174347 | 0.00302927 | 0.00232626 | 0.00325326 |
| Dense | 0.00146574 | 0.00388332 | 0.00256470 | 0.00702663 |
| Reader | 0.00217449 | 0.00713449 | 0.00156226 | 0.00757319 |

The fixed endpoint matters. The best logged validation field MSE **through epoch 2,500** is 0.00215368 for Dense at epoch 2,437, versus 0.00233322 for Legacy at 2,401 and 0.00705331 for Reader at 2,430. Dense's epoch-2,500 validation temperature MSE is also higher than its epoch-2,000 value (0.007027 versus 0.002526). Thus the exact-checkpoint rank reversal does not establish that every late Dense checkpoint is worse than Legacy. Logged validation uses the training evaluator's sampling/masking definitions and is not numerically interchangeable with the full-grid, fluid-only 90-case comparison. No full-holdout evaluation of epoch 2,437 or post-2,500 checkpoint selection was performed.

## What can be said about group use now

Reader's equal-case mean P2 main-context norm grows from 6.7245 at 500 to 11.0303 at 2,500. Its mean branch-norm fractions change from main/coarse/local 0.3969/0.5182/0.0849 to 0.4089/0.4977/0.0934. Mean non-null mass remains 0.658334: the geometry-envelope identity fixes that quantity for this case/query set. These summaries cover full field-query grids, not fluid-only error masks or padded physical-port proxies.

At epoch 2,500, logged pre-clip gradient norms are 0.0146324 for group preparation, 0.0016489 for group reception, 0.0096627 for coarse, and 0.0099321 for local parameters. These observations support continued numerical participation; neither context norms nor gradients prove beneficial causal coupling. No new P0-port, P1-feedback, P2-field intervention, connected/disconnected influence, or support-transition derivative experiment was run here. The corrected phase attribution and derivative findings in the 500-epoch recovery report remain scoped to their measured checkpoints.

The 16 physical-reference requests remain `reference_verification_pending`. Dataset-reference reconstruction scores are available; no new trustworthy external physical-reference validation has arrived or is claimed.

## Artifacts and reproducibility

All paths below are relative to `HONF_Proj/`. The study root is `diagnostics/generated/interface_operator_study/epoch2500_comparison/`. Generated evidence remains local under the repository's existing ignore policy; this report records the quantitative findings in Git.

| Artifact | Path under study root |
|---|---|
| Exact executed evaluation/CPU commands | `commands.txt` |
| Evaluation arguments and complete log | `evaluation_argv.json`, `evaluation.log` |
| Standard comparison tables | `evaluation/tables/` |
| Four anchors × three model exports | `evaluation/debug_npz/` |
| Schema-aware training/checkpoint audit | `analyze_epoch2500_metrics.py`, `epoch2500_metrics_comparison.json`, `analysis.md` |
| Reproducible comparison reduction | `summarize_epoch2500.py`, `reduction/comparison_tables.md`, `reduction/reduction.json` |
| Pairing and raw-score QA | `reduction/case_set_validation.csv`, `reduction/target_count_validation.csv`, `reduction/raw_pool_reconciliation.csv`, `reduction/status.json` |
| Full metrics, pairs, strata, channel decomposition | `reduction/metrics_long.csv`, `reduction/paired_case_deltas.csv`, `reduction/paired_summary.csv`, `reduction/strata_long.csv`, `reduction/normalized_mse_channel_contributions.csv` |
| Reader context summaries | `reader_context_summary.json` |
| Compact figure, renderer, source CSVs | `figures/reader_epoch500_epoch2500_comparison.png`, `figures/render_reader_epoch_comparison.py`, `figures/reader_epoch_progress_source.csv`, `figures/reader_epoch2500_pairs_source.csv` |

Exact evaluated files, all ending in `epoch_2500_model.pt`:

```text
Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/epoch_2500_model.pt
Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/epoch_2500_model.pt
Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1805_20260908_002326_sparse_interface_geometry_read/epoch_2500_model.pt
```

Epoch-500 inputs were reused from `Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage3_Run1401_1804_1801_1802_Epoch500_90Case/tables/` and `diagnostics/generated/interface_operator_study/group_reader_recovery/endpoint/tables/`; they were not regenerated or altered. The reduction records source paths and uses raw components from those tables.

## Research recommendation

Retain Legacy as the current overall field/thermal reference and Dense as the stronger hydrodynamic/port comparator at this exact budget. Geometry-envelope reading remains a successful numerical recovery from the old collapsed reader, but these results do not establish competitive overall reconstruction or make branch activity a success criterion. The early internal-temperature advantage should no longer be used to argue superiority over Legacy.

Leave the already authorized 5,000-epoch continuations in place and wait for the user's requested comparison. At that point, repeat the same fixed-checkpoint comparison and check whether Dense's temperature deterioration is persistent or endpoint-sensitive. Any later group-first coarse redesign should be justified by fresh phase-specific ground-truth interventions at the selected trained checkpoint. This single-seed development-set result does not yet select that redesign or justify additional automatic runs.
