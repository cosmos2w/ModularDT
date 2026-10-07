# Run 1406 Group-Control Development Report

This is an epoch-50 evidence artifact for the low-dimensional group-control hypothesis. Learned interaction routes are not physical causality. The comparison is reporting evidence, not a permanent runtime or CI gate.

## Execution record

- The implementation used the clean source commit `f7751bad3666c551075963d00ab055dac2873629`; follow-up comparison diagnostics were corrected in `0987605` and completed in `5d3d24c` without changing trained parameters or field outputs.
- The opt-in profile is `project://src/config_core/forward/group_control_pairwise_honf_context.json`: `forward_architecture=group_control_pairwise_honf`, fixed K=6, D=16, entmax15 source/query normalization, predicted ports from epoch 1, and no warm start.
- Exactly one managed Run 1406 was launched on physical GPU 0 (`cuda:0`, NVIDIA RTX 6000 Ada): `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1406_20260919_121442_low_dimensional_group_control`. Its manifest records UUID `c6644243-53ba-4c6a-b2de-683dfdfddc0d`, clean source, exit code 0, status `completed`, and last completed epoch 50.
- The accepted checkpoint is `epoch_0050_model.pt`. A separate strict model-and-optimizer reload of the ordinary epoch-10 checkpoint succeeded with 151 optimizer state entries.
- Before training, the disposable real predicted-port GPU preflight exercised fully materialized M1 and M12 updates. The M12 step had finite loss `8.44666`, pre-clip norm `21.7047`, update norm `0.370909`, and nonzero gradients/updates in the router, modulation, fine preparation, fine module/environment paths, global path, field head, and local coupling.
- Validation at handoff: 564 tests passed in the full suite, 36 focused tests passed, and `git diff --check` was clean. No quickcheck run was created.

## Checkpoint policy

| Label | Path | Required epoch | Role | Selection |
|---|---|---:|---|---|
| 1406 | `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1406_20260919_121442_low_dimensional_group_control/epoch_0050_model.pt` | 50 | matched_epoch_50_candidate | explicit_cli_checkpoint |
| 1804 | `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/epoch_0050_model.pt` | 50 | matched_epoch_50_dense_reference | explicit_cli_checkpoint |

## Controlled protocol

- Same-GPU comparison: cases `0273, 0653`; Q=8192; receiver chunk=2048; 2 warmups/5 synchronized repetitions.
- Timed phases: full physical forward and prepared P2 decode; routing maps and profiler are excluded from timed calls.
- Training: real fixed M1/M12 buckets, B=48, Q=1024; 1 warmup/3 measured forward/backward/clip/update steps with fresh disposable optimizers.
- Model residency: `load one model at a time; release before loading the next`. Reverse-order repeat requested: `False`.

## Ordinary training health

| Epoch | Train loss | Val total | Val field MSE | Val temperature MSE | Pre-clip norm | Update norm | Peak allocation (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.5374 | 4.2353 | 1.8940 | 1.0125 | 97.9965 | 0.4565 | 28823.3 |
| 10 | 3.1451 | 3.1697 | 1.6981 | 0.6965 | 1.8668 | 0.1319 | 28835.0 |
| 20 | 2.5315 | 2.4890 | 1.5098 | 0.4582 | 8.2251 | 0.1711 | NA |
| 25 | 2.3228 | 2.1924 | 1.2975 | 0.4408 | not scheduled | not scheduled | 28802.6 |
| 50 | 0.6954 | 0.5974 | 0.3112 | 0.2094 | 8.9506 | 0.1056 | 28781.8 |

All recorded losses, parameters, gradients, and updates remained finite. The median validation field MSE fell from `1.79535` over epochs 1-10 to `0.42961` over epochs 41-50. Epoch 50 was below the predeclared catastrophic-regression flag (`0.438678`) but remained worse than Dense 1804 at the matched checkpoint (`0.219339`).

## Inference benchmark

Each phase reports a median, min-to-max spread, pre-call allocated/reserved baseline, and absolute/incremental allocated/reserved peaks. Values are NA until the physical benchmark is executed.

| Label | Case | Full median/spread (ms) | P2 median/spread (ms) | Full baseline alloc/res (MiB) | Full abs/inc alloc (MiB) | Full abs/inc reserved (MiB) | P2 baseline alloc/res (MiB) | P2 abs/inc alloc (MiB) | P2 abs/inc reserved (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1406 | 0273 | 75.608/1.810 | 42.056/0.766 | 23.66/104.00 | 77.25/53.59 | 104.00/0.00 | 24.84/104.00 | 75.20/50.36 | 104.00/0.00 |
| 1406 | 0653 | 77.182/3.630 | 41.867/0.900 | 23.66/168.00 | 79.62/55.96 | 168.00/0.00 | 24.84/168.00 | 77.57/52.72 | 168.00/0.00 |
| 1804 | 0273 | 33.699/0.693 | 13.127/0.153 | 29.16/618.00 | 460.01/430.85 | 618.00/0.00 | 29.82/618.00 | 458.81/428.99 | 618.00/0.00 |
| 1804 | 0653 | 33.204/0.375 | 13.155/0.098 | 29.16/618.00 | 460.01/430.85 | 618.00/0.00 | 29.82/618.00 | 458.81/428.99 | 618.00/0.00 |

Reverse-order repeat: **not_requested**; borderline workloads: `[]`.

## Training-step benchmark

| Label | Bucket | Median/spread (ms) | Baseline alloc/res (MiB) | Absolute/incremental alloc (MiB) | Absolute/incremental reserved (MiB) | Status |
|---|---|---:|---:|---:|---:|---|
| 1406 | M1 | 480.326/28.519 | 64.97/6540.00 | 5914.24/5849.27 | 6542.00/2.00 | complete |
| 1406 | M12 | 1008.238/6.378 | 64.97/30784.00 | 28659.21/28594.24 | 30784.00/0.00 | complete |
| 1804 | M1 | 967.465/8.113 | 86.97/6574.00 | 5849.89/5762.93 | 6576.00/2.00 | complete |
| 1804 | M12 | 2137.565/90.580 | 86.97/28942.00 | 26779.83/26692.86 | 28944.00/2.00 | complete |

## Logical versus executed work ledger

The board and ledger keep logical q→group→source paths separate from unique q→source candidates. A logical multiplicity is cheap-control work; it is not a fine GPU call.

| Label | Case | Phase | Source | Logical | Unique | Multiplicity | Actual fine calls | MLP rows | Env geometry rows | Env content rows | Scalar rows | Source projections | Forward calls | Recomputation | Valid/padded denominator |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1406 | 0273 | P0 | module | 2304 | 2304 | 1 | 2304 | 2304 | NA | NA | 9216 | 12 | NA | 0 | 2304/6912 |
| 1406 | 0273 | P0 | environment | 2.565e+05 | 1.475e+05 | 1.74 | 1.475e+05 | 1.475e+05 | 1.475e+05 | 5.898e+05 | 1.475e+05 | 192 | NA | 0 | 1.475e+05/0 |
| 1406 | 0273 | P1 | module | 2304 | 2304 | 1 | 2304 | 2304 | NA | NA | 9216 | 12 | NA | 0 | 2304/6912 |
| 1406 | 0273 | P1 | environment | 2.557e+05 | 1.475e+05 | 1.734 | 1.475e+05 | 1.475e+05 | 1.475e+05 | 5.898e+05 | 1.475e+05 | 192 | NA | 0 | 1.475e+05/0 |
| 1406 | 0273 | P2 | module | 2.458e+04 | 2.458e+04 | 1 | 2.458e+04 | 2.458e+04 | NA | NA | 9.83e+04 | 12 | NA | 0 | 2.458e+04/7.373e+04 |
| 1406 | 0273 | P2 | environment | 2.728e+06 | 1.573e+06 | 1.734 | 1.573e+06 | 1.573e+06 | 3.932e+05 | 1.573e+06 | 1.573e+06 | 192 | NA | 0 | 1.573e+06/0 |
| 1406 | 0273 | P2_consistency | module | 1152 | 1152 | 1 | 1152 | 1152 | NA | NA | 4608 | NA | NA | 0 | 1152/3456 |
| 1406 | 0273 | P2_consistency | environment | 1.279e+05 | 7.373e+04 | 1.734 | 7.373e+04 | 7.373e+04 | 7.373e+04 | 2.949e+05 | 7.373e+04 | NA | NA | 0 | 7.373e+04/0 |
| 1406 | 0653 | P0 | module | 3840 | 3840 | 1 | 3840 | 3840 | NA | NA | 9216 | 12 | NA | 0 | 3840/5376 |
| 1406 | 0653 | P0 | environment | 2.596e+05 | 1.475e+05 | 1.76 | 1.475e+05 | 1.475e+05 | 1.475e+05 | 5.898e+05 | 1.475e+05 | 192 | NA | 0 | 1.475e+05/0 |
| 1406 | 0653 | P1 | module | 3840 | 3840 | 1 | 3840 | 3840 | NA | NA | 9216 | 12 | NA | 0 | 3840/5376 |
| 1406 | 0653 | P1 | environment | 2.596e+05 | 1.475e+05 | 1.76 | 1.475e+05 | 1.475e+05 | 1.475e+05 | 5.898e+05 | 1.475e+05 | 192 | NA | 0 | 1.475e+05/0 |
| 1406 | 0653 | P2 | module | 4.096e+04 | 4.096e+04 | 1 | 4.096e+04 | 4.096e+04 | NA | NA | 9.83e+04 | 12 | NA | 0 | 4.096e+04/5.734e+04 |
| 1406 | 0653 | P2 | environment | 2.761e+06 | 1.573e+06 | 1.755 | 1.573e+06 | 1.573e+06 | 3.932e+05 | 1.573e+06 | 1.573e+06 | 192 | NA | 0 | 1.573e+06/0 |
| 1406 | 0653 | P2_consistency | module | 1920 | 1920 | 1 | 1920 | 1920 | NA | NA | 4608 | NA | NA | 0 | 1920/2688 |
| 1406 | 0653 | P2_consistency | environment | 1.294e+05 | 7.373e+04 | 1.755 | 7.373e+04 | 7.373e+04 | 7.373e+04 | 2.949e+05 | 7.373e+04 | NA | NA | 0 | 7.373e+04/0 |
| 1804 | 0273 | P0 | module | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0273 | P0 | environment | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0273 | P1 | module | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0273 | P1 | environment | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0273 | P2 | module | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0273 | P2 | environment | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0273 | P2_consistency | module | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0273 | P2_consistency | environment | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0653 | P0 | module | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0653 | P0 | environment | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0653 | P1 | module | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0653 | P1 | environment | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0653 | P2 | module | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0653 | P2 | environment | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0653 | P2_consistency | module | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |
| 1804 | 0653 | P2_consistency | environment | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA/NA |

## Epoch-50 decision evidence

Overall status: **fail**. Thresholds are mean full-forward ratio ≤ 0.95, M12 step ratio ≤ 0.95, and allocated peak ratio ≤ 1.10; R_M/R_E < 1 is not required.
- `mean_full_forward_speed`: **fail** — ratio `2.284`; Run 1406 was about 128% slower than Dense 1804 instead of at least 5% faster.
- `m12_step_speed`: **pass** — ratio `0.472`; Run 1406 was about 52.8% faster than Dense 1804.
- `peak_allocated_memory`: **pass** — M12 ratio `1.070`, within the 1.10 ceiling; inference allocation was substantially lower than Dense 1804 (`77.25/79.62 MiB` versus `460.01 MiB`).
- `learning`: **pass** — finite outputs/parameters/gradients, meaningful updates, improving validation trajectory, no unresolved catastrophic instability when metrics are supplied
- `execution_semantics`: **pass** — P2 actual fine rows equal unique q->source pairs. Measured `R_M=1.0` and `R_E=1.0`; the implementation avoids duplicated fine evaluation but does not create sparse physical support.

Because the mandatory full-forward speed gate failed decisively, the same run was stopped at epoch 50. It was not resumed to epoch 500.

## Learning and modest fidelity evidence

- Run 1406: epoch-50 val field MSE `0.3111778348684311 (ok)`; trajectory evidence **pass**.
- Run 1804: epoch-50 val field MSE `0.21933870762586594 (ok)`; trajectory evidence **pass**.
- A roughly twofold field-MSE ratio is a severe-regression flag for review, not an automatic convergence verdict.

## Debug/API contract

The preferred opt-in flag is `return_group_control_maps=True`; the existing routing-map flag is used only when explicitly exposed by the backend. P0/P1/P2/P2_consistency records must report logical and unique numerators, actual calls, module MLP rows, environment geometry/content rows, scalar controls, source projections, forward calls, checkpoint recomputations, and valid/padded denominators. Receiver-chunk numerators and denominators are summed before ratios.

## Material deviations and corrections

- The first comparison process completed its measurements but failed while writing the artifact because the optional reverse-order result was uninitialized when no repeat was requested. That output was rejected; commit `0987605` fixed only the optional writer path, and the same comparison was rerun.
- Review of the rerun exposed missing untimed port-global consistency diagnostics and missing valid/padded denominators. Commit `5d3d24c` completed those ledgers without changing the timed model path, parameters, or field outputs; the accepted comparison was regenerated. No extra timing repeat was added because the full-forward result was not borderline.
- Run 1401 was not part of this Run-1406 plan's mandatory epoch-50 gate and was not included in the accepted comparison. No result for it is inferred here.

## Artifacts

- GPU preflight: `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/diagnostics/generated/run1406/gpu_preflight.json`
- Accepted comparison and semantic ledger: `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/diagnostics/generated/run1406/comparison.json`
- Interaction board: `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/diagnostics/generated/run1406/board/group_control_interaction_board.png` (PDF and JSON companions are in the same directory)
- Managed run manifest: `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1406_20260919_121442_low_dimensional_group_control/run_manifest.json`

## Interpretation and limits

- Empty group centres are unavailable (NaN), never silently placed at an origin.
- The debug map pass is untimed and is not a replacement for measured decode time.
- Unique-pair support can remain dense; lower R is not itself an acceleration result.
- Exactly one managed training run reached epoch 50, followed by the accepted same-GPU comparison. No profiler, quickcheck, full 90-case endpoint evaluation, epoch-500 continuation, or 5k continuation was run.
