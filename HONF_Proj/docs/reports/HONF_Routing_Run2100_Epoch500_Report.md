# Run 2100: fixed-data mean-shift routing

**Status: implementation validated; the single fresh Run 2100 is training through 500. Epoch-500 results are pending.**

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

## Accuracy, support, and attraction

Pending. The final assessment will distinguish exact epoch 500 from separately saved best-by-validation-field through 500, using all 90 development cases and the existing complete-grid metrics. Historical exact-500 baselines will be reused in place. The repeatedly used `test` split is a development holdout.

Attractor drift, iteration movement, separation, descriptor concentration, diagnostic approximate modes, and candidate-generation time will be reported alongside actual deduplicated fine support, omitted-source evidence, accuracy, and end-to-end cost. Clustering alone does not establish useful physical routing or acceleration.

## Interim attraction evidence at epoch 10

The two-anchor, 32-query CPU replay uses the saved epoch-10 model, the complete P0/P1/P2 forward, and the adapter's `ell=(1.8,1.8)`. The following P2 measurements concern active candidates; storage remains padded to 12, with every active candidate retained.

| Anchor | Active candidates | Mean physical drift | Minimum separation, seed → final | Diagnostic physical modes, tolerance 0.05 | Fine QM / QE per query |
|---|---:|---:|---:|---:|---:|
| 0273 | 3 | 1.2240 | 1.6668 → 0.0386 | 3 → 2 | 3 / 192 |
| 0298 | 7 | 1.2930 | 1.1166 → 0.0454 | 7 → 6 | 7 / 192 |

The dimensionless joint-space diagnostic at tolerance 0.05 also counts 2 and 6 components after three updates; these are approximate connected components, not an exact unique K. Mean joint drifts are 0.6806 and 0.7185. Per-iteration movement decreases, but three finite iterations are not a claim of converged modes.

All active fine QM/QE pairs still execute in **every physical phase**. For 0298, mean-shift produces 1,184 module paths and 36,544 environment paths for the 32 field queries, versus 1,056 and 29,472 with a frozen module-hub candidate intervention. Deduplication yields the same 224 QM and 6,144 QE pairs in both. Attraction therefore increases path multiplicity in this probe without reducing actual fine work.

A same-weight candidate intervention gives normalized fluid-only field relative L2 0.597204 versus module hubs 0.597144 for 0273, and 0.635880 versus 0.636463 for 0298. Each comparison uses the same 31 fluid points × five fields and the same target norm. This small, mixed effect is not a trained Run-2000 comparison and does not establish accuracy benefit over the full population.

Candidate-only CPU timing, with trajectory collection disabled, one warmup and three repeats, is approximately 1.38–1.83 ms for mean-shift versus 0.34–0.41 ms for module hubs per phase/case. These bounded CPU measurements establish added candidate work; they do not establish GPU latency or end-to-end speedup.

[Candidate diagnostic JSON](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/routing/mean_shift_epoch10.json) contains phase ledgers, masked target errors, physical/scaled/joint drift and separation, descriptor concentration, tolerance-labelled mode counts, and repeated timings.

![Epoch-10 attraction versus fine work](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/figures/candidate_attraction.png)

## Interim physical derivatives

A bounded CPU replay at epoch 10 used anchors 0273 and 0298 with 32 fixed queries. Both full P0/P1/P2 position and heat gradients were finite through the three mean-shift steps. [Signed results](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/numerics/physical_derivatives_epoch10.json) retain physical and normalized units and the prescribed step sizes. The temperature and heat derivatives mostly agree closely, but the tiny 0298 position-to-pressure derivative is cancellation-sensitive: normalized AD is 3.59e-6 versus FD 2.65e-5/5.30e-5 at 0.01r/0.005r, giving 86–93% relative discrepancy despite small absolute errors. The helper's `ok` status means execution completed, not that every derivative met an accuracy threshold. No step sizes were tuned, and this is not an epoch-500 or physical-solver validation.

## Evidence limits and continuation

The current ThermalChannel adapter supplies known boundary descriptors and neutral extra resistance. **Barrier benefit: Evidence Missing.** Model AD/FD is self-consistency evidence, not independent physical-gradient validation. Existing external physical-reference requests remain in place.

The user permits an interim handoff after a healthy first 50 epochs when the full run is slow. Such a handoff will be labelled interim, with the epoch-500 analysis explicitly outstanding. No Goal 3, additional managed trial, sweep, or automatic training beyond 500 is authorized. A concrete unexecuted same-run continuation command will accompany the assessed checkpoint.

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
