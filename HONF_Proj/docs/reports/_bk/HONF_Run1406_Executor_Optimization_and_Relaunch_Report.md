# Run 1406 Exact-Executor Optimization and Relaunch Report

## Outcome

The exact Run-1406 executor optimization is implemented at commit
`e8be3195b9b7e0ae92d14da972226598e015cdaa`.  The existing epoch-50
checkpoint strict-loads unchanged, old/new physical outputs and first
derivatives pass parity, and the optimized executor substantially reduces the
old executor cost.  A fresh, same-model Run 1406 was then trained from scratch
through epoch 50 on physical GPU 0.

The fresh run was **not** resumed to epoch 500.  Its epoch-50 continuation
evidence failed two fixed criteria:

- mean full-forward time was `1.1090x` Dense 1804, rather than at most
  `0.95x`;
- the M12 absolute allocated-memory peak was `1.1068x` Dense 1804, just above
  the `1.10x` ceiling.

Training health, execution semantics, and M12 optimizer-step speed passed.
The failed criteria were not relaxed after measurement.

## Executor change

The normal complete-support path now:

- evaluates query routing/overlap once and reuses it;
- caches the `[B,K,M*D]` group-conditioned module control bank and reads it
  with batched matrix multiplication;
- executes the environment contraction over the full receiver chunk with a
  rectangular, rho-zero-safe masked implementation;
- reuses compatible receiver Fourier features supplied by the shared core;
- keeps logical paths, gathered support, detailed ledgers, and routing maps
  behind the opt-in evidence path.

The partial/gathered reference path remains available for untimed evidence.
The model, parameter inventory, K=6/D=16 routing, losses, optimizer, seed,
schedule, and trusted checkpoint loading are unchanged.  No Triton kernel or
approximation was added.

Managed-run allocation was also revised to allow repeated logical Run IDs in
different timestamped directories.  Exact path collisions remain rejected,
and directory creation remains atomic.  This enabled the requested fresh
same-ID rerun without changing the standard results root.

## Existing-checkpoint compatibility and parity

The accepted reference is pre-optimization commit
`bf416d7d8c63c53f428d4bbb89109bcdc4ec7927`; the optimized source is
`e8be3195b9b7e0ae92d14da972226598e015cdaa`.  Both workers strict-loaded the
same original Run-1406 epoch-50 checkpoint with 280 state keys.

- Cases `0273` and `0653`: physical field, internal/interface, provisional
  P1, and predicted-port outputs passed the established tolerances.
- The largest field-output relative norm was below `3e-7`.
- A real predicted-port B48/Q1024/M12 backward pass passed all loss-term and
  gradient comparisons; all 151 connected gradients were checked.  The
  evidence pass did not update the checkpoint.

Same-checkpoint executor timings on physical GPU 0 used Q=8192, receiver chunk
2048, two warmups, and five synchronized repetitions:

| Case | Phase | Reference (ms) | Optimized (ms) | Ratio |
|---|---|---:|---:|---:|
| 0273 | full physical forward | 75.387 | 35.351 | 0.4689 |
| 0273 | preparation plus one query | 39.461 | 27.488 | 0.6966 |
| 0273 | prepared P2 decode | 42.867 | 10.943 | 0.2553 |
| 0653 | full physical forward | 76.558 | 35.569 | 0.4646 |
| 0653 | preparation plus one query | 36.835 | 26.690 | 0.7246 |
| 0653 | prepared P2 decode | 43.536 | 10.998 | 0.2526 |

The optimized old checkpoint was then compared directly with Dense 1804.
Its mean full-forward ratio improved from the original executor's `2.284x` to
`1.0578x` Dense.  M12 optimizer-step ratio was `0.4640x`.  This established a
credible acceleration path and justified the bounded fresh rerun, but did not
pre-approve epoch-500 continuation.

No profiler was run: the optimized full-forward ratio was below the plan's
`1.25x` profiler trigger.

## Fresh managed Run 1406

Run directory:

`/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1406_20260919_132837_low_dimensional_group_control_executor_optimized_rerun`

The manifest records UUID `f1d952c0-e606-482e-b84a-198434fd89d8`, source
commit `e8be3195b9b7e0ae92d14da972226598e015cdaa`, clean source state, status
`completed`, exit code 0, and last completed epoch 50.  The original
`group_control_pairwise_honf_context.json` profile was used, with seed 0 and no
warm start.  Only this managed rerun was launched; there was no quickcheck run.

Training remained finite and improved materially:

| Quantity | Epoch 1 | Epoch 50 |
|---|---:|---:|
| train total loss | 5.5374 | 0.6982 |
| validation total loss | 4.2353 | 0.5989 |
| validation field MSE | 1.8940 | 0.3004 |
| validation temperature MSE | 1.0125 | 0.1865 |

The median validation field MSE fell from `1.79535` over epochs 1-10 to
`0.40708` over epochs 41-50.  At epoch 50 the pre-clip gradient norm was
`5.93621`, update norm was `0.0759032`, and all recorded losses, parameters,
gradients, and updates were finite.  `epoch_0050_model.pt`, `latest.pt`, and
the best checkpoints exist.

## Fresh epoch-50 comparison and decision

The accepted same-GPU protocol used cases `0273` and `0653`, Q=8192,
receiver chunk 2048, two warmups/five synchronized inference repetitions, and
real B48/Q1024 M1/M12 optimizer steps with one warmup/three measured updates.
Maps and ledgers were excluded from timed calls.

| Workload | Run 1406 | Dense 1804 | Ratio | Criterion | Result |
|---|---:|---:|---:|---:|---|
| mean full forward | 37.243 ms | 33.581 ms | 1.1090 | <=0.95 | fail |
| M12 optimizer step | 977.789 ms | 2104.005 ms | 0.4647 | <=0.95 | pass |
| M12 allocated peak | 31,077,533,184 B | 28,078,916,608 B | 1.1068 | <=1.10 | fail |

Per-case inference medians were:

| Label | Case | Full forward (ms) | Prepared P2 (ms) | Allocated peak (MiB) |
|---|---|---:|---:|---:|
| Run 1406 | 0273 | 38.711 | 11.162 | 471.45 |
| Run 1406 | 0653 | 35.774 | 11.126 | 471.45 |
| Dense 1804 | 0273 | 33.265 | 13.159 | 461.01 |
| Dense 1804 | 0653 | 33.897 | 13.326 | 461.01 |

Prepared P2 decode is faster than Dense, but full forward is not.  Full
forward allocation passed the memory ceiling; the M12 workload caused the
overall memory failure.  The timing result was not borderline, so no
reverse-order repeat was requested.

The P2 ledgers report actual fine calls equal to unique q-source pairs for both
module and environment reads.  `R_M=1.0` and `R_E=1.0`: the executor removes
duplicate fine evaluation but does not produce sparse physical support.  The
routes shown in the interaction board are learned interactions, not physical
causality.

Run 1401 is historical context only under the Run-1406 executor plan and was
not a mandatory live comparator.  The accepted continuation decision uses the
explicit Run-1406/Dense-1804 pair only.

## Validation

- Full repository unit suite: `574 passed`.
- Focused executor/evidence/integration suite: `43 passed`.
- Managed-run allocation tests: `7 passed`.
- Ruff and `git diff --check`: clean.
- Broader `tests Case_ThermalChannel/tests`: `695 passed, 1 skipped, 4 failed`.
  The four failures are unrelated missing diagnostic scripts
  (`evaluate_topology_quality.py` and
  `evaluate_retained_mass_pruning.py`), not failures in the executor or
  managed-run changes.

## Artifacts and limits

- Parity and old/new timing:
  `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/diagnostics/generated/run1406_executor/evidence.json`
- Optimized existing-checkpoint comparison:
  `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/diagnostics/generated/run1406_executor/same_checkpoint_comparison.json`
- Fresh rerun comparison and full semantic ledger:
  `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/diagnostics/generated/run1406_executor/rerun_epoch50/comparison.json`
- Interaction board (PNG/PDF/JSON companions):
  `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/diagnostics/generated/run1406_executor/rerun_epoch50/boards/group_control_interaction_board.png`
- Fresh run manifest:
  `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1406_20260919_132837_low_dimensional_group_control_executor_optimized_rerun/run_manifest.json`

The evidence is surrogate/model output and performance instrumentation, not
CFD truth.  No full 90-case endpoint evaluation, epoch-500 continuation, or
5k continuation was run.
