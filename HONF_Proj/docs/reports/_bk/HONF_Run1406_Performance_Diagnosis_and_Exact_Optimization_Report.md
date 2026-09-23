# Run 1406 Performance Diagnosis and Exact Optimization Report

## Executive answer

Run 1406 is slower than Dense 1804 in the complete physical forward because
it rebuilds phase-dependent group routing and control state before **each** of
P0, P1, and P2.  On physical GPU 1, this makes preparation about `1.24x`
Dense.  Encoding, the port head, the frozen local surrogate, and module-state
fusion are not responsible for the gap.

Prepared P2 is nevertheless faster because Run 1406 removes the separate
dense local-context read and uses the optimized rectangular group-controlled
module path.  That saving is larger than its extra P2 router/QE work.  It does
not pay back the three repeated preparation costs in a full forward.

The M12 memory excess has a different cause.  M12 has `12/12` active modules,
so inactive padding is not responsible.  The old activation-checkpoint
boundary retained the complete QE score-control, score, geometry, mask,
softmax, and renormalization tensors for P0, P1, P2, and P2 consistency.  A
single exact change checkpoints the complete supported QE calculation and
recomputes those intermediates during backward.  It reduces the measured M12
allocated peak from `28.962 GiB` to `23.148 GiB` (`-5.814 GiB`, `-20.1%`),
putting Run 1406 `11.6%` below Dense's `26.171 GiB` peak.  The tradeoff is an
M12 step increase from `1127.228 ms` to `1274.850 ms` (`1.131x`), while the
retained executor remains `0.442x` Dense's `2883.643 ms` step time.

No Run 1407 or other training run was created.  The live resumed Run 1406 on
physical GPU 0 was not interrupted, modified, restarted, or used for these
measurements.  All execution and profiling used physical GPU 1 and temporary
Git worktrees.

## Scope and comparison contract

The comparison strict-loaded the epoch-50 checkpoints for:

- current optimized Run 1406:
  `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1406_20260919_132837_low_dimensional_group_control_executor_optimized_rerun/epoch_0050_model.pt`;
- Dense 1804:
  `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/epoch_0050_model.pt`;
- routing-only Run 1404, for historical interpretation only:
  `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1404_20260916_092508_routing_only_pairwise/epoch_0050_model.pt`.

The Run-1406 model remains K=6, D=16, entmax routing, Dense MM/ME/EM
preparation, collapsed group control before one q-source fine interaction,
`C_g + C_M + C_E`, and the existing P0/P1/P2 physical coupling.  The change
adds no parameter, loss, route, interaction, or model equation.  Detailed
routing maps and ledgers remained outside normal timed reads.

Inference used cases `0273` and `0653`, Q=8192, receiver chunks of 2048, two
warmups, and five synchronized repetitions.  Dense ran first in the accepted
reverse-order repeat.  M12 used the real B48/Q1024/M12 predicted-port training
step, one disposable optimizer update, and no checkpoint writeback.

## Reproduced unprofiled timings

The first table is the requested reproduction of the pre-change executor.
The second table repeats the protocol after the retained training-only QE
checkpoint change.  Because checkpointing is inactive in evaluation, the
small inference differences between invocations are timing variance rather
than an inference optimization.

| Source | Model | Full forward (ms) | Preparation + one query (ms) | Prepared P2 (ms) |
|---|---|---:|---:|---:|
| Pre-change | Run 1406 | 40.583 | 32.830 | 16.256 |
| Pre-change | Dense 1804 | 37.580 | 26.446 | 17.425 |
| Pre-change ratio | 1406 / Dense | 1.080 | 1.241 | 0.933 |
| Retained | Run 1406 | 39.918 | 31.405 | 15.929 |
| Retained | Dense 1804 | 37.391 | 25.428 | 16.812 |
| Retained ratio | 1406 / Dense | 1.068 | 1.235 | 0.947 |

Thus the stated behavior reproduces on the free GPU: prepared P2 is about
`5-7%` faster, while full forward is about `7-8%` slower and preparation is
about `24%` slower.

The corresponding unprofiled M12 reproduction was:

| Executor | Step median (ms) | Allocated peak (GiB) | Step vs Dense | Peak vs Dense |
|---|---:|---:|---:|---:|
| Run 1406, pre-change | 1127.228 | 28.962 | 0.391 | 1.107 |
| Run 1406, retained QE checkpoint | 1274.850 | 23.148 | 0.442 | 0.885 |
| Dense 1804 | 2883.643 | 26.171 | 1.000 | 1.000 |

## Forward-path attribution

The event profiler used CUDA events inside the real physical path.  Values
below are averages over the two anchor cases and the instrumented repetitions.
Events are nested, so rows explain attribution but must not be summed into a
second end-to-end timing.

| Region | Run 1406 (ms) | Dense 1804 (ms) | Interpretation |
|---|---:|---:|---|
| Case encoding | 1.297 | 1.373 | Common and slightly faster in 1406 |
| P0+P1+P2 preparation | 15.477 | 8.919 | Dominant 1406 overhead |
| Router/control construction within preparation | 7.060 | n/a | Repeated once per phase |
| Query routing across P0/P1/P2 | 2.228 | n/a | Intrinsic group-control query work |
| P0 read | 4.104 | 3.268 | 1406 overhead |
| P1 read/decode | 4.043 | 3.530 | 1406 overhead |
| P2 decode | 14.325 | 15.117 | 1406 saving |
| Frozen local surrogate | 6.710 | 6.912 | Common; not the gap |
| Module-state fusion | 0.517 | 0.551 | Common; not the gap |

The group-controlled read decomposition is:

| Fine/read region, P0+P1+P2 | Run 1406 (ms) | Dense 1804 (ms) | Delta (ms) |
|---|---:|---:|---:|
| QM reads | 5.000 | 3.741 | +1.259 |
| QE reads | 12.132 | 9.275 | +2.857 |
| Separate local-context reads | 0.125 | 6.990 | -6.865 |

Run 1406's QE computation is not faster than Dense.  Its prepared P2 endpoint
is faster because the collapsed `C_g + C_M + C_E` path avoids Dense's separate
local-context evaluation, whose P2 portion alone is about `2.47 ms`, and the
optimized QM path is inexpensive.  In the complete forward, however, the
approximately `6.56 ms` excess across the three preparations plus P0/P1
routing and read overhead exceeds the roughly `0.79 ms` profiled P2 saving.

### Why preparation cannot simply be cached across P0/P1/P2

Case encoding is already performed once.  Each later preparation observes a
different module state: P0 predicts ports, the local surrogate and fusion
update module state, P1 refines it, and P2 consumes the refined state.  Source
memberships, collapsed controls, group moments, the QM first-affine/control
bank, and QE K/V/control banks therefore depend on the phase state.  Treating
these quantities as case-static would change the predictor.  Only small
geometry/position pieces are truly static; profiling did not identify them as
a useful standalone cache target.

## M12 activation-memory diagnosis

Saved-tensor accounting localizes the excess to the complete QE rectangles,
not the optimizer or loss assembly.  Before the change, Run 1406 already held
`27.515 GiB` allocated immediately after forward versus Dense's `25.195 GiB`.
Loss assembly added only about `0.009 GiB` to either model.

| Saved-tensor role | Run 1406 before (GiB) | Dense (GiB) | Run 1406 retained (GiB) |
|---|---:|---:|---:|
| P0 port | 3.939 | 3.647 | 2.383 |
| P1 refinement | 4.604 | 4.448 | 2.625 |
| P2 field | 5.335 | 4.893 | 3.260 |
| P2 port consistency | 2.003 | 1.930 | 1.014 |
| Common/unscoped | 12.357 | 12.357 | 12.357 |
| Maximum live saved tensors | 28.238 | 27.276 | 21.636 |

The prior boundary checkpointed only the module tail after its first affine.
It left the complete QE receiver-by-environment computation live for backward.
That produced fewer saved tensors than Dense but larger rectangular tensors.
Moving the boundary around the complete QE computation removes `6.602 GiB`
of maximum live saved state.  The profiled allocated maximum after backward
falls from `29.066 GiB` to `23.256 GiB`; Dense is `26.656 GiB` in the same
phase-memory protocol.

M12 has a padded width of 12 and an active range of 12 to 12, so padded
inactive modules contribute nothing to this M12 difference.  Padding is real
for the two inference anchors (`3/12` and `5/12` active modules), but it is a
latency concern there, not the measured M12 memory cause.

## Candidate optimizations and disposition

| Candidate | Evidence | Decision |
|---|---|---|
| Cache group controls across P0/P1/P2 | Heavy controls depend on phase-updated module state; only small positional pieces are static | Rejected: would change the predictor or save too little |
| Compact inactive P0/P1 module-port receivers | Anchors contain substantial padding, but exact compaction requires gather/scatter, original ordering, receiver-shaped auxiliary tensors, and provenance preservation | Not retained: no low-risk measured implementation |
| Elide normal read diagnostics | Runtime-only counterfactual preserved predictions and changed full median from `40.5862` to `40.2731 ms` (`0.9923x`) | Rejected: about `0.8%`, within variance, and the reporting contract is useful |
| Reuse receiver Fourier features | Removed a real duplicate computation; P2 moved about `1.2%`, but whole forward became about `0.9%` slower in the controlled repeat | Reverted: no measurable full-forward benefit |
| Checkpoint complete QE | Removed the retained score/attention rectangle; real outputs and derivatives were unchanged | Retained: `-20.1%` M12 peak for a `+13.1%` step-time tradeoff |

Normal reads still return compact norm/fraction, overlap-mass,
query-control-norm, and read-degree tensors.  Large route maps and fine-path
ledgers remain opt-in.  Overlap mass is part of the response scaling and is
not merely a diagnostic.  The diagnostic counterfactual confirms that these
small summaries do not explain the latency or memory gap.

The inactive receiver issue is implementation overhead rather than a new
scientific interaction, but removing it exactly is not just masking a tensor:
the public `[B,M,P]` ordering and receiver-shaped auxiliary contract must be
restored after a ragged read.  M12 offers no inactive rows on which such an
optimization could save memory.

## Retained exact optimization and compatibility

When activation checkpointing is active and QE has complete support, the
executor now wraps the complete environment reader in non-reentrant
checkpointing.  Backward recomputes score control, scores, relative geometry,
support masking, softmax, and renormalization.  The partial-support path and
evaluation path are unchanged.  Overlap mass remains an explicit output, so
normal auxiliary behavior is preserved; recomputation counters correctly
report the additional backward work.

Compatibility evidence:

- the existing Run-1406 epoch-50 checkpoint strict-loaded all 280 state keys;
- no trainable parameter or serialized key was added or removed;
- real-checkpoint case `0273`, Q=128 parity produced exactly zero maximum
  absolute and relative difference for field, interface, port, and internal
  outputs, and for `d(loss)/d(query_xy)` and
  `d(loss)/d(module_centers)`;
- a focused train-mode unit test compares the complete-QE output, normal
  auxiliary tensors, input derivatives, and every connected backend parameter
  derivative with the established output and first-gradient tolerances;
- historical Dense and routing-only architectures keep their existing class,
  state, and strict-loading paths.

Focused validation completed with `38 passed, 1 skipped`; the skip is the
known unavailable frozen ThermalChannel local-surrogate fixture in a temporary
worktree.  The broader unit suite completed with `571 passed, 4 skipped, 2
failed`.  Both failures reproduce on the pre-change source: one is the
environment-dependent Dense background reference check and the other lacks
the worktree-local dataset-location configuration.  Ruff and
`git diff --check` passed.

## Intrinsic and accidental costs

The intrinsic costs of the current scientific design are entmax source/query
routing, K=6 low-dimensional control construction, group moments, QM/QE
control banks, and their repetition after each physically meaningful state
update.  The fact that `R_M = R_E = 1` shows that each unique q-source fine
interaction is evaluated once; it does not make the routing/control algebra
free and does not imply sparse physical support.

The accidental costs found here are narrower:

- the complete QE activation stack was retained across backward; this is now
  fixed exactly;
- receiver Fourier features were duplicated, but removing the duplication was
  not a measurable whole-forward win and was therefore not kept;
- padded anchor receivers perform unnecessary work, but an exact compact
  implementation needs a wider contract-preserving change;
- normal diagnostics have a measurable but immaterial sub-percent cost.

Run 1404 is useful historical context, not a direct comparator.  Its mature
routing-only executor reported `27.09 ms` full forward and `6.67 ms` P2 versus
Dense's `35.31/13.04 ms` under its older protocol, but it is a different
architecture and did not establish learned physical sparsity.  Its lower cost
therefore does not contradict the present conclusion: the new Run-1406 cost is
specifically the phase-dependent group-control construction that Run 1404 did
not perform.

## Further improvement and Run-1407 decision

Run 1406 can be improved further without changing its science only through
executor engineering: fuse or batch the three state-dependent preparation
kernels more efficiently, or implement an exact active-receiver gather and
scatter adapter with full output/gradient/auxiliary parity.  The profile does
not support removing diagnostics, caching complete controls across phases, or
adding more checkpoint boundaries merely because they exist.  The retained QE
boundary already resolves the measured memory excess while preserving a large
step-time advantage over Dense.

There is **no executor-based justification for a future Run 1407**.  A new run
would be scientifically justified only if it intentionally targets one
architectural bottleneck: the requirement to rebuild group control after each
P0/P1/P2 state transition.  For example, making control phase-shared or
phase-static would be a new model equation and would need its own scientific
hypothesis and validation.  It should not be disguised as an executor fix or
started merely to recover the remaining `7-8%` latency gap.

## Evidence artifacts

The reproducibility bundle is local and generated; it was not used to alter a
training checkpoint:

`/home/wanglz/Desktop/src/ModularDT/HONF_Proj/diagnostics/generated/run1406_performance_diagnosis`

Key files are:

- `results/reverse_order_baseline.json`: accepted pre-change unprofiled
  inference reproduction;
- `results/run1406_gpu1_profile.json`: CUDA-event forward attribution and M12
  reproduction;
- `results/m12_phase_memory_steady.json`: original/Dense phase memory and
  saved-tensor accounting;
- `results/combined_374a_full_profile.json`: retained executor inference and
  repeated M12 benchmark;
- `results/combined_374a_m12_phase_timed.json`: retained phase memory and
  saved-tensor accounting;
- `results/combined_374a_real_parity.json`: real-checkpoint output and first
  derivative parity;
- `results/combined_374a_normal_read_elision.json`: rejected diagnostic-elision
  counterfactual.

These results characterize model/surrogate execution and memory behavior.
They are not CFD or physical-truth validation.
