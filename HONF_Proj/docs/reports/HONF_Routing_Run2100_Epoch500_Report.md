# Run 2100: fixed-data mean-shift routing

**Status: epoch-50 interim assessment complete. The original single Run 2100 continues on physical GPU 1 toward 500; final endpoint analysis is deferred under the user’s first-50 handoff option. No training beyond 500 or Goal 3 has been started.**

## Scope

Goal 2 adds `routing.strategy="mean_shift"` to the shared `routed_pairwise_honf` backend. The formal candidate performs exactly three fixed-data mean-shift iterations at feature bandwidth 1. Module coordinates scaled by the adapter's routing length and live module routing descriptors form the fixed source samples. Candidates start at those samples; only the candidates move during the iterations. Source samples remain connected to autograd.

Every active module retains its candidate identity. No mode merging, coordinate rounding, inferred exact unique count, diversity/count loss, or bandwidth/iteration sweep is part of this experiment. Approximate mode counts are tolerance-labelled diagnostics only.

Source sparsemax, source-measure query sparsemax, unique-pair QM/QE dispatch, Dense fine preparation, common coarse/local paths, P0/P1/P2 physical coupling, frozen Stage A, losses, data, and AdamW retain Goal 1's behavior. In particular, source incidence must use the original source descriptors, not the moved candidate descriptors.

The checkout is `cosmos2w/ModularDT`, branch `agent/honf-core-next`. The planning head `9ae17f1` is an ancestor of the newer Goal 1 implementation inspected for this task. Concurrent Run 2000 on physical GPU 2 and unrelated staged documentation moves are preserved. Run 2100 is assigned physical GPU 1, exposed as logical `cuda:0`.

## Validation and execution

Before the candidate implementation landed, the 124 historical compatibility checks passed across configuration, core contracts, 3-D, Regional, NStage2, group paths, and WindFarm. The common endpoint reducer's policy/population checks pass for both run identities (11 tests).

The formal dry run passed. Resolved training, data, loss, case, checkpointing, and every non-routing model field match the Goal-1 profile. The only routing changes are the strategy and its explicit three-step/bandwidth-one settings. Configuration/registry checks pass (31 tests).

Exactly two disposable physical optimizer checks ran on GPU 1, each on a separate fresh model with batch 48, 1,024 queries/case, predicted ports, frozen Stage A, the canonical losses, clipping, and AdamW. Neither allocated a managed run nor saved weights.

| Check | M | Seconds | Peak allocated MiB | Peak reserved MiB | Preclip norm | Clip scale | Update norm | Router gradient / update |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small | 1 | 3.7154 | 6,181.80 | 6,768 | 474.786 | 0.0021062 | 0.549905 | 0 / 0 |
| Large | 12 | 8.7803 | 36,950.57 | 39,670 | 610.763 | 0.0016373 | 0.563598 | 0.0218646 / 0.0531530 |

Losses and updated parameters were finite. Component preclip/update norms use the existing FP64 reductions. The singleton router's zero gradient is expected. At initialization both checks retain all active QM and QE pairs through P0/P1/P2: mean-shift is not initially computationally sparse. This is a measured scientific observation, not a reason to alter support or suppress training.

The shared float64 routing checks passed: packed-versus-dense prior error is 1.11e-16; the joint routing JVP/finite-difference error is 6.35e-12. These checks concern shared arithmetic; the candidate-specific tests are recorded separately.

Actual commands are in [executed_commands.md](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/executed_commands.md); disposable results are [small.json](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/physical_smoke/small.json) and [large.json](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/physical_smoke/large.json). No disposable weights will initialize the formal run.

Nine mean-shift tests pass, including T=0 seeding, source permutation, the fixed-data versus blurring reference, finite gradients and float64 gradcheck, separated-mode/consensus fixtures, padding/empty cases, original source-descriptor incidence, and three-dimensional moving resistance with fixed source endpoints. The final candidate/common routing/config/physical suite passed 43 tests in the actual training interpreter. The physical P0/P1/P2 refresh test passed under both strategies. Commands/results are retained in [focused_tests.txt](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/numerics/focused_tests.txt).

Actual Run-2000 epoch-50, Legacy-1401 epoch-500, and Dense-1804 epoch-500 checkpoints reconstructed strictly through the maintained trusted CPU loader after this change. Run-2000 still serializes no mean-shift settings. Details are in [checkpoint_compatibility.txt](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/numerics/checkpoint_compatibility.txt).

The implementation was committed as `e6048d2` atop the legitimate newer Goal-1 work. The one formal launch began at 2026-09-16 04:21:48 UTC and allocated:

```text
Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_2100_20260916_002148_routed_mean_shift
```

It uses physical GPU 1, logical `cuda:0`, from scratch with no Run-2000 weights or optimizer state. The configured endpoint is 500. The ordinary allocator, trusted loading, existing checkpoint integrity behavior, and all historical results remain intact.

## First-50 learning and optimizer health

Both metric files cover contiguous epochs 1–50. All core train/validation loss, field, and temperature measurements are finite; the log contains no OOM, traceback, or CUDA error. The trajectory improves overall with visible thermal/field fluctuations, including a transient late thermal spike; it is not monotonic convergence.

| Epoch | Train loss | Validation loss | Validation field MSE | Validation temperature MSE |
|---|---:|---:|---:|---:|
| 1 | 7.294524 | 4.748454 | 1.924386 | 1.178038 |
| 10 | 1.836387 | 1.693550 | 0.651141 | 0.780901 |
| 25 | 0.903264 | 0.820428 | 0.300287 | 0.360553 |
| 50 | 0.393885 | 0.368727 | 0.196162 | 0.122869 |

From epochs 31–40 to 41–50, mean train/validation loss falls from 0.6433/0.6191 to 0.4961/0.5062. Validation field/temperature means fall from 0.2679/0.2545 to 0.2249/0.2185. Learning is still developing. The minimum sampled validation field MSE through 50 is 0.177908 at epoch 49; it is distinct from the exact-50 value and is not a full-grid selected-checkpoint evaluation.

At epoch 50, the recorded first-batch full preclip gradient norm is **1.36217**, clip scale **0.734121**, and parameter-update norm **0.149589**. Router gradient/update norms are **0.0211523 / 0.0373288**. These are useful finite updates. Measurements exist at epochs 1, 2, 5, 10, 20, and 50; other CSV norm entries are unsampled placeholders. The large initial full norm 529.307 was clipped with scale 0.001889 and did not persist.

Training P2 retains **99.7302% QM / 99.9949% QE** of the active-source dense reference at epoch 50; validation retains **99.3498% / 100%**. These ratios use recorded batch-averaged counts. Source-membership sparsity is therefore not producing material fine-execution reduction.

Matched exact-50 sampled validation field/temperature MSE is: Legacy 1401 **0.259676 / 0.212371**, Dense 1804 **0.219339 / 0.125466**, Regional 1806 **0.203376 / 0.117707**, Run 2000 **0.198393 / 0.226158**, and Run 2100 **0.196162 / 0.122869**. These single-epoch values provide convergence context, not a trained full-population superiority claim; the curves show meaningful fluctuations.

The recorded training plus validation compute time through 50 is **3,434.47 s (57.24 min)**, excluding plotting/checkpoint overhead. Observed wall pace is about **70.6 s/epoch**, implying roughly **9.8 hours total** to 500 and about **8.8 hours remaining at epoch 50**. Peak training allocation over the window is **38,322.73 MiB**. The existing process PID 3878073 remains on physical GPU 1; Run 2000 remains on GPU 2.

![First-50 training comparison](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/figures/training_trajectory.png)

## Accuracy, support, and attraction

Pending. The final assessment will distinguish exact epoch 500 from separately saved best-by-validation-field through 500, using all 90 development cases and the existing complete-grid metrics. Historical exact-500 baselines will be reused in place. The repeatedly used `test` split is a development holdout.

Attractor drift, iteration movement, separation, descriptor concentration, diagnostic approximate modes, and candidate-generation time will be reported alongside actual deduplicated fine support, omitted-source evidence, accuracy, and end-to-end cost. Clustering alone does not establish useful physical routing or acceleration.

## Interim attraction evidence at epoch 50

The saved epoch-50 checkpoint was replayed on CPU over all five prescribed anchors, using 32 fixed field queries per case and the complete P0/P1/P2 forward. The adapter routing length is `ell=(1.8,1.8)`. Every active module retains its candidate; storage is padded to 12. Approximate counts below are connected components at dimensionless joint-space tolerance 0.05, never an execution K.

| Anchor | Active candidates | Mean physical drift | Minimum physical separation, seed → final | Joint diagnostic modes, seed → final | P2 fine QM / QE per query | Field relative L2: mean shift / same-weight hubs |
|---|---:|---:|---:|---:|---:|---:|
| 0273 | 3 | 0.7251 | 1.6668 → 0.4573 | 3 → 3 | 3 / 192 | 0.236294 / 0.236785 |
| 0653 | 5 | 0.8343 | 1.2214 → 0.0231 | 5 → 3 | 5 / 192 | 0.290681 / 0.291561 |
| 0283 | 5 | 0.7245 | 1.1308 → 0.0031 | 5 → 4 | 5 / 192 | 0.286773 / 0.282861 |
| 0298 | 7 | 1.0690 | 1.1166 → 0.1539 | 7 → 7 | 7 / 192 | 0.342933 / 0.335710 |
| 0302 | 7 | 1.0783 | 1.5882 → 0.3289 | 7 → 7 | 7 / 192 | 0.287344 / 0.292727 |

All fine field pairs execute on these five probes. The normalized fluid-only pooled relative L2 across their 775 scalar targets is **0.295857 for mean shift versus 0.294192 for the same-weight module-hub intervention**. Three anchors improve slightly and two worsen. This is a frozen candidate-generation intervention with the full physical loop recomputed, not a separately trained Run-2000 comparison or the 90-case endpoint. Attraction has not demonstrated improved effective support or pooled accuracy in this bounded sample.

Candidate-only CPU timing, with trajectory collection disabled, one warmup and three repeats, is **1.16–1.29 ms** for mean shift versus **0.288–0.313 ms** for module hubs across phases/cases. This measures added candidate-generation cost, not GPU or end-to-end latency. The [diagnostic JSON](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/routing/mean_shift_epoch50.json) retains physical/scaled/joint drift, per-step movement, separation, descriptor concentration, tolerance-labelled modes, phase ledgers, prediction errors, and repeated costs.

The [earlier epoch-10 probes](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/routing/mean_shift_epoch10.json) are retained as a comparison: 0273 and 0298 had physical mean drift 1.2240/1.2930 and final minimum separation 0.0386/0.0454. At epoch 50, those centres are less collapsed while their fine field work remains dense. Centre clustering therefore cannot substitute for actual support and accuracy measurements.

![Epoch-50 attraction versus fine work](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/figures/candidate_attraction.png)

## Interim physical derivatives

A bounded CPU replay at epoch 10 used anchors 0273 and 0298 with 32 fixed queries. Both full P0/P1/P2 position and heat gradients were finite through the three mean-shift steps. [Signed results](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/numerics/physical_derivatives_epoch10.json) retain physical and normalized units and the prescribed step sizes. The temperature and heat derivatives mostly agree closely, but the tiny 0298 position-to-pressure derivative is cancellation-sensitive: normalized AD is 3.59e-6 versus FD 2.65e-5/5.30e-5 at 0.01r/0.005r, giving 86–93% relative discrepancy despite small absolute errors. The helper's `ok` status means execution completed, not that every derivative met an accuracy threshold. No step sizes were tuned, and this is not an epoch-500 or physical-solver validation.

The real-anchor omitted-source probes are unavailable at these receivers because every valid source is selected. The separate conditional synthetic fixture's zero omitted contribution must not be interpreted as real physical omission validation.

The same prescribed [physical derivative replay at epoch 50](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/numerics/physical_derivatives_epoch50.json) again completed with finite full-loop gradients. Position AD/FD relative differences are at most 1.21%; heat differences are at most 0.328% across the two anchors and prescribed steps. The epoch-10 tiny-pressure discrepancy is retained above as historical evidence, not silently overwritten.

Direct selected-source FP32 probes are less decisive: selected module response discrepancies reach 20.4%, and environment JVP norms near 4e-10 are compared with FD norms around 8e-5–4e-4, yielding approximately unit relative error. Those probes are unresolved at the prescribed precision/steps; finite execution is not a derivative-accuracy certificate. No steps were tuned. Omitted real sources remain unavailable at the probed receivers because their supports are full.

## Evidence limits and continuation

The current ThermalChannel adapter supplies known boundary descriptors and neutral extra resistance. **Barrier benefit: Evidence Missing.** Model AD/FD is self-consistency evidence, not independent physical-gradient validation. Existing external physical-reference requests remain in place.

**Interim recommendation: unresolved.** Optimization is healthy and still improving, with ordinary fluctuations; useful routing sparsity and an accuracy benefit from attraction have not been demonstrated. Finish the already authorized 500-epoch budget before deciding on a 2500/5000 extension. Added candidate cost with almost-dense fine work is currently an efficiency concern, not evidence of acceleration.

The user's first-50 handoff option is used for analysis: the original process continues through its configured 500 epochs, while exact-500/saved-best full-population evaluation, final phase interventions, large-shape GPU timing/memory, and the final continuation decision remain **pending a later user request**. Training itself has not been paused or restarted. No Goal 3, extra managed trial, sweep, or automatic continuation beyond 500 was launched.

## Prepared endpoint and same-run continuation commands

The [endpoint command sheet](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/endpoint_commands.md) contains **unexecuted** exact-500 and saved-field-best full-population evaluation, reducers, common routing ledgers, omitted-source influence audit, phase interventions, physical derivatives, candidate diagnostics, timing, and convergence plotting. Existing historical endpoint tables are reused; model training is not repeated.

If the completed endpoint later justifies an extension and the user requests it, the following command resumes the **same Run 2100** from epoch 500 through terminal epoch 2500. It has **not been executed** and is not the current recommendation to extend. Run it from `HONF_Proj` only after the existing process has finished and that checkpoint exists.

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_ThermalChannel/src OMP_NUM_THREADS=4 \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py \
  --config project://src/config_core/forward/routing_mean_shift_context.json \
  --workflow forward --device cuda:0 --epochs 2500 \
  --resume-checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_2100_20260916_002148_routed_mean_shift/epoch_0500_model.pt \
  --yes
```

`--epochs` is the terminal epoch, and `--resume-checkpoint` restores the saved model, optimizer, scaler, and RNG state in the checkpoint's original run directory. The command sheet also records an unexecuted epoch-50-to-500 resume command for use only if training itself is stopped at that checkpoint; it must not be launched alongside the current training process.
