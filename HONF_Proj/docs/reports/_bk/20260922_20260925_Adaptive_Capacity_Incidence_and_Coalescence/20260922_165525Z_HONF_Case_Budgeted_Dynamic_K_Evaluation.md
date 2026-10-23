# HONF case-budgeted dynamic K: Run 1409 evaluation

**Checkpoint policy:** exact epoch 50 for Run 1409, Run 1406, Dense Run 1804, and routing-only Run 1404. All four checkpoints were evaluated on the same 90 test cases, at 8,192 field queries per case with predicted ports. Run 1409 was trained once from scratch on GPU 0 and stopped at epoch 50. The separately saved best-field checkpoint is also epoch 50. No epoch-500 Run 1409 result exists.

## Decision

Run 1409 did not establish adaptive case capacity or cheaper physical execution. At epoch 50 its deployed deterministic gate estimate selected only the ordinary group on **all 90 test cases**; eight independent stochastic gate draws per detailed anchor made the same selection. The one-group routing operator still evaluates the full fine module and environment rectangles. In the matched 90-case test, Run 1409 has pooled fluid relative L2 **0.54911**, versus **0.42440** for Run 1406; it loses all 90 paired cases to the parent. The measured B48/Q1024 optimizer step is essentially the same speed as Run 1406 and uses slightly more peak allocated memory. There is no demonstrated accuracy–cost benefit to justify continuing this trajectory to 500 epochs. The model remained numerically healthy and its validation curve was improving; this is a scientific stop, not a failed process or proof that conditional capacity cannot work.

## Implemented operator and scope

The opt-in `budgeted_group_control_honf` profile keeps the Run-1406 Dense MM/ME/EM physical preparation and its `Cg + CM + CE` decoder. It registers `Kmax=12` groups with `D=16` controls. Group 0 is always available. A small case-conditioned hard-concrete network predicts the other 11 gates from the P0 prepared controls. The stochastic plan is sampled once per forward at P0 and the same plan object is reused in P1/P2. Source memberships, collective controls, and fine physical values are recomputed after each physical state update.

For source and query assignments, the gate enters as `log(z)` before masked 1.5-entmax. Closed columns have exact zero source and query membership. The gate-only reference `eta=entmax15(log z)` defines `kappa=1/sum(eta²)`, which replaces the parent's multiplicative K amplitude terms for moments, occupancies, and fine outputs. `Kmax`, sampled live count, packed width, learned routing rank, and `kappa` are distinct quantities. The compact implementation gathers available group columns before its source/query controller work; full-width and compact use the same fine reader. The formal executor remains rectangular because selected fine reads have no demonstrated speed advantage in this experiment. Logical support and actual fine rows are reported separately.

The only added objective is the mean expected optional-group count once per case/forward. Two real training batches (B48/Q1024, predicted ports) calibrated its fixed coefficient:

`lambda = 0.02 × median(||∂L_physical/∂a|| / ||∂L_group/∂a||) = 0.005783974924700852`,

where `a` denotes optional gate logits. The measured ratios were 0.322108 and 0.256290. Calibration made no optimizer update. The formal overlay sets this one numeric coefficient. There is no gate schedule, pair-cost penalty, subset enumeration, sampled environmental QE, custom kernel, or duplicated query–group–source fine path.

## Verification and launch

The final focused suite passed **96 tests**, including the six-forced-open reduction to Run 1406, closed-padding invariance for K=12 versus K=32, mixed-case full/compact output and gradient parity, P0 plan identity with phase-specific control/fine refresh, strict model/optimizer/RNG checkpoint resume, historical six-group evidence loading, and unchanged historical config behavior. A separate real B48/Q1024 GPU batch performed one predicted-port AdamW update with the frozen local surrogate attached. Its gate-network gradient norm was `2.1930e-4` and parameter update norm `7.986e-3`, both finite and nonzero; total loss changed from 11.9167 to 4.6078 on that same batch. This was a disposable check, not a second training run.

The stage-I initialized-weight check used the same model and fixed gate noise for full and compact modes. At Q=1024, Kpack=11 and maximum field difference was `8.20e-8`; at Q=8192, Kpack=12 and difference was zero. K=6/12/32 controller shape checks passed. These timings are a code-path check, not evidence of learned sparsity: compact/full median CUDA timings were 67.5/73.8 ms at Q=1024 and 207.9/184.3 ms at Q=8192 with the profile's ordinary receiver chunk. Compact did not consistently improve wall time.

The sole managed Run 1409 directory is `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1409_20260922_122819_case_budgeted_group_control/`. It completed 50/50 epochs with finite metrics and an exact `epoch_0050_model.pt`. Validation field MSE improved from 1.9017 (epoch 1) to 0.39984 (epoch 50), with a temporary spike at epoch 48. Run 1406's same-epoch validation field MSE was 0.30036. Run 1409's validation expected optional count declined from 10.442 (epoch 1) to 0.00135 (epoch 50); the expected-count term was not multiplied by query chunks or physical phases.

## Matched epoch-50 physical evaluation

All figures below come from the one four-checkpoint, 90-case full-grid comparison. The checkpoints were explicit exact-epoch files; no best-checkpoint fallback or teacher ports were used. Pooled relative L2 is computed from pooled squared error and pooled target magnitude. Equal-case summaries use each case once.

| Checkpoint | Fluid pooled rel. L2 | Fluid pooled MSE | Equal-case median | P95 | Worst case (max) |
|---|---:|---:|---:|---:|---|
| 1409 budgeted | 0.54911 | 0.27915 | 0.54832 | 0.62319 | 0289 (0.64300) |
| 1406 parent | 0.42440 | 0.16675 | 0.42068 | 0.46392 | 0295 (0.50684) |
| 1804 Dense | **0.37203** | **0.12813** | **0.37449** | **0.41637** | 0647 (0.43497) |
| 1404 routing only | 0.61834 | 0.35397 | 0.62210 | 0.73007 | 0281 (0.75221) |

Run 1409 beats Run 1406 on **0/90** paired fluid cases, Dense 1804 on **1/90**, and routing-only 1404 on **73/90**. Its mean per-case L2 difference is +0.12553 versus 1406 and -0.08264 versus 1404. The relative pooled fluid error is 29.4% above the parent at this exact budget. The gap to 1406 appears in every predefined module-count, spacing, wall-proximity, and heating-heterogeneity stratum; it is not confined to one subset.

| Pooled relative L2 | 1409 | 1406 | 1804 | 1404 |
|---|---:|---:|---:|---:|
| Near interface | 0.42368 | 0.38579 | **0.33349** | 0.35771 |
| Far fluid | 0.63640 | 0.44281 | **0.40406** | 0.88683 |
| U / V | 0.26538 / 0.52068 | 0.21980 / 0.42884 | **0.16921** / 0.28829 | 0.93492 / **0.27950** |
| Pressure / vorticity | 0.76610 / 0.54423 | 0.43481 / **0.51200** | 0.39653 / 0.52940 | **0.38922** / 0.71607 |
| Field temperature | 0.60686 | 0.45634 | **0.36337** | 0.50753 |
| Internal / surface temperature | 0.22996 / 0.30653 | 0.22218 / 0.29261 | **0.16537** / **0.21832** | 0.18659 / 0.24990 |
| Normal heat flux | **0.47374** | 0.47405 | 0.47942 | 0.47580 |
| Final outside temperature / effective h | 0.32818 / 0.07708 | 0.29246 / 0.08069 | 0.26667 / 0.07824 | **0.25581** / **0.05879** |

The model's main physical deficits are far-fluid reconstruction, pressure, and field temperature. The normal-flux difference across these early checkpoints is small. Mean pressure-drop absolute error is 0.0362 for 1409 versus 0.0206 for 1406; its p95 is 0.0751 versus 0.0466. Mean outlet-temperature absolute error is 1.5134 versus 1.6629, but its p95 is worse at 5.4515 versus 4.8943. These are predicted physical quantities on the test dataset, not independent CFD validation.

## Capacity, rank, and actual work

At exact epoch 50, a deterministic Q=1,024 capacity/rank audit of **all 90 test cases** gives `Kmax=12`, `K_det=1`, `Kpack=1`, one module-occupied group, one environment-occupied group, and one query-reached group in every case. `K_expected=1+sum(p_optional)` ranged from 1.00009 to 1.01173 (median 1.00054, mean 1.00141). There is small variation in optional probabilities, but no observed variation in deployed group count. The module and environment weighted query-to-source routing operators each have numerical rank 1 in all 90 cases (relative singular-value tolerance `1e-6`, thin QR; entropy effective rank also 1). This is a rank of learned routing arrays; it is not the rank of the physical response operator or an optimal-rank estimate. The gate plan is reused at P1/P2 while the source controls refresh. Eight independent stochastic P0 draws per detailed anchor also yielded one live group and zero field-prediction difference from deterministic gating. Those 16 draws do not prove agreement for every case or random seed; `K_sampled` is deliberately unavailable in the deterministic 90-case audit.

For Q=8,192, both anchors retain all valid logical query-source pairs: 24,576/40,960 module pairs for 3/5 active modules and 1,572,864 environment pairs. The rectangular executor actually runs 98,304 padded module rows and 1,572,864 environment rows per anchor. The Q=1,024 population audit likewise records 12,288 padded module rows and 196,608 environment rows at P2 for every case. The packed controller width is one, but fine physical work has not fallen. The always-available group produces a dense logical path here; no learned physical sparsity or causality is implied. Source-empty and query-unreached optional groups are a consequence of closed gates, not evidence for their physical irrelevance.

## Measured cost

The same-GPU, checkpoint-restored, disposable optimizer benchmark used B48/Q1024 real train batches with one warmup and three synchronized measured updates. Run 1409 used the current physical objective plus its calibrated count term; Run 1406 used its historical objective.

| Workload | 1409 median step | 1406 median step | 1409 peak allocated | 1406 peak allocated |
|---|---:|---:|---:|---:|
| M1 | 521.66 ms | 518.36 ms | 4,368 MiB | 4,229 MiB |
| M12 | 1,115.47 ms | 1,115.13 ms | 24,020 MiB | 23,685 MiB |

The corrected Q=8,192 deterministic anchor inference had full-forward medians of 40.21/40.51 ms (full-width) and 40.55/39.67 ms (compact) on cases 0273/0653; prepared P2 medians were 11.29/11.35 and 11.24/11.32 ms. Full-forward peak CUDA allocated memory was 472.3 MiB (full-width) versus 471.9 MiB (compact), measured after releasing debug tensors between cases and modes. The reduction is about 0.4 MiB, with mixed timing differences; packed controller width one does not reduce the dense fine reader. Full/compact field predictions differ by at most `1.67e-6` across the anchors. The prior Run-1406 exact-500 evidence measured 36.84 ms mean full-forward and 11.00 ms prepared P2 under the same nominal GPU/Q/chunk protocol, but it was gathered in a separate session and at a different checkpoint budget; it is directional context, not the matched epoch-50 timing control. The disposable matched optimizer benchmark above is the direct 1409-versus-1406 cost control.

## Continuation recommendation and evidence limits

**Stop this one trajectory at epoch 50.** Validation was improving and the deterministic policy is stable, yet deployed gates selected one group, no fine rows were saved, matched training cost did not fall, and fluid accuracy lost to the parent on every held-out case. Extending the same optimizer to 500 would test whether this collapsed single-group model eventually catches up, but would not at this stage test case-specific capacity. No second seed, fixed-count ablation, threshold tuning, lambda recalibration, architecture change, or longer continuation was run. Those would be separate scientific decisions.

If a later decision authorizes a bounded 500-epoch continuation of this same run, the unexecuted command is:

```bash
python train.py --config src/config_core/forward/budgeted_group_control_honf_context.json \
  --experiment-overlay src/config_core/forward/experiments/run1409_case_group_budget_calibrated.json \
  --run-id 1409 --epochs 500 --device cuda:0 \
  --resume-checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1409_20260922_122819_case_budgeted_group_control/latest_model.pt --yes
```

The report covers one seed, one calibration, 50 epochs, 90 held-out development cases, two detailed full-grid routing anchors, and one GPU. It does not contain an epoch-500 or best-through-500 comparison, a retrained fixed-K control, an independent physical-rank measurement, or a measured selected-fine-reader crossover. Group IDs are permutation-ambiguous learned labels. Neither logical support nor learned rank should be read as physical causality.

## Evidence and reproduction

- Managed run: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1409_20260922_122819_case_budgeted_group_control/` (`run_manifest.json`, `metrics.csv`, exact checkpoint, and `evaluations/`).
- Matched physical tables: `evaluations/matched_epoch50_90case/tables/model_summary_metrics.csv`, `per_case_metrics.csv`, and `physical_stratum_metrics.csv` under that run.
- Gate, routing-rank, row-ledger, and mode audit: `evaluations/run1409_anchor_evidence_epoch50_corrected.json` and `evaluations/arrays_epoch50_corrected/` under that run; routing board: `evaluations/board_epoch50/`.
- Population deterministic capacity/rank and actual-row audit: `evaluations/population_capacity_rank_q1024.json` and `.csv` under that run.
- Matched training cost: `evaluations/matched_training_epoch50.json`; initialized-weight full/compact check: `evaluations/stage1_shapes.json` under that run.
- Prelaunch coefficient and real update: `diagnostics/generated/run1409_budgeted_group_control/calibration.json` and `real_update.json`.
- Bounded source tools: `tools/diagnostics/calibrate_run1409_case_budget.py`, `run_run1409_real_update.py`, `run_run1409_stage1_shapes.py`, `run_run1409_budgeted_evidence.py`, and `run_run1409_population_capacity_rank.py`.

Exact launch command executed:

```bash
python train.py --config src/config_core/forward/budgeted_group_control_honf_context.json \
  --experiment-overlay src/config_core/forward/experiments/run1409_case_group_budget_calibrated.json \
  --run-id 1409 --epochs 50 --device cuda:0 --yes
```
