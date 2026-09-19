# Run 2001 sparse execution and proposed Run 2002

Implementation and bounded execution measurements are complete. Run 2002
has completed its first 50 observed epochs and remains running toward its
configured 500-epoch limit on physical GPU 0. No robust speedup is established;
losses improved overall but had substantial transient spikes. No 500-epoch
scientific outcome or 90-case fidelity result is claimed here.

## Scope and reference

The working branch is `agent/honf-core-next`; work began from the requested
commit `3943710c0df76c8380feb6b98d0cf7b9842c0594` with a clean working tree.
The implementation follows the
[sparse execution plan](../../UpgradePlan/HONF_Routing_Next_Phase_Sparse_Execution_Plan.md).
The historical
[epoch-2500 report](HONF_Routing_Epoch2500_Comparative_Evaluation.md) and
[merged reduction](../../diagnostics/generated/accuracy_eval_run2001_2101_epoch2500_20260917/merged_reduction_summary.md)
remain unchanged. Their suggested hub-balance/concentration losses are
superseded for this experiment by the plan's single induced-pair-cost loss.

Run 2002 uses the already-evaluated explicit Run 2001 **epoch 2500**, rather
than its preferred epoch-2425 selected checkpoint. The selected checkpoint
is unavailable: trusted metadata inspection found epoch 2709 in the mutable
best files and epoch 471 in `checkpoints/best_field.pt`. The retained explicit
`epoch_2500_model.pt` contains epoch 2500. No parent checkpoint was copied.
The full parent path is
`Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_2001_20260916_170829_routed_module_hubs_quickcheck_optimized/epoch_2500_model.pt`.

Run 2101 is an executor comparator only. The existing Run 2001 and 2101
parent continuations on physical GPUs 1 and 2 were observed and left
unchanged. All new CUDA work uses physical GPU 0. Only one new candidate,
at most 500 additional epochs, is authorized. Monitoring stops at local
epoch 50; endpoint evaluation and any extension await a separate user request.

## Historical evidence and denominator correction

At exact epoch 2500 the historical pooled fluid relative L2 values are
0.045285 (Legacy 1401), 0.048835 (Dense 1804), 0.044960 (2001), and
0.046395 (2101). Both routed endpoints had worse p95 than Legacy.
Run 2001's final-port temperature relative L2 was 0.100844. These are
supplied historical development-holdout metrics, not new physical/CFD evidence.
The selected 2001 epoch 2425 and selected 2101 epoch 2732 are separate
checkpoint policies; the latter is not best-through-2500.

Direct inspection resolves two historical labels:

- `run2001_e2500_routing_ledger.json` records `query_count=32`.
  Thus 219/224 is 32 receivers times 7 modules, not an 8,192-query grid.
- `benchmark_routing_optimization.py::_RouteCounter` counts batch cases
  inside each reader invocation. The 930/1104 number refers to complete
  case/chunk rows, not individual receiver rows.

## New projection diagnosis

Managed evidence root:
`diagnostics/generated/interface_operator_study/dynamic_sparse_routing/sparse_execution_next/`.

`projection_diagnosis_2001.json` records five anchors (0273, 0653, 0283,
0298, 0302), 32 deterministic field queries per anchor, every executed
P0/P1/P2 read, and fixed B=48 M1/M12 training-data batches. It includes
source supports, occupied hub masses, query density/probability summaries,
logical path counts, exact Boolean source unions, and separate complete
receiver and case/chunk denominators. The reported margin is
`min_occupied(u) - (sum(mu*u)-1)/sum(mu)`, using actual floating-point mass.

All five environmental anchor unions are complete at temperature 1.
Some receivers have already excluded hubs, demonstrating why the all-active
certificate is sufficient but not necessary for a complete fine-source union.
The M12 training-data probe retains 16,946/18,432 module pairs and
294,662/294,912 environmental pairs in its P2 read.

The P2 field-read diagnostic at temperature one makes the distinction
explicit (32 receivers each):

| Case | Module margin minimum | Actual module pairs | Environment margin minimum | All-active environment receivers | Actual environment pairs |
|---|---:|---:|---:|---:|---:|
| 0273 | 0.53673 | 96/96 | 0.45732 | 32/32 | 6144/6144 |
| 0653 | 0.24528 | 160/160 | 0.37214 | 32/32 | 6144/6144 |
| 0283 | 0.25929 | 160/160 | -0.06237 | 29/32 | 6144/6144 |
| 0298 | -0.14833 | 219/224 | -0.36492 | 23/32 | 6144/6144 |
| 0302 | 0.05670 | 224/224 | -0.19917 | 25/32 | 6144/6144 |

Content and geometry both have appreciable across-hub spreads; propensity
spreads are much smaller on the inspected anchors. The residual after
subtracting the three terms is floating-point roundoff, consistent with
the unchanged neutral resistance. These spreads alone do not establish
component-attributed task-gradient dominance.

Frozen query-temperature probes use 1, 0.5, 0.25, and 0.125 with source
temperature fixed at 1. They change the full physical-loop prediction;
they do not select a training temperature. On case 0273, temperature 0.125
retains 99.48% of environmental pairs while changing the field by 21.89%
in relative norm. This is sensitivity evidence, not improved accuracy or
an omitted-source fidelity guarantee.

**M=1 structural limit:** a genuinely single candidate gives `A=1`,
`mu=1`, `d=1`, and `Pi=omega` for every finite positive temperature.
Environmental support cannot become sparse. The measured M1 margin is one
and all source pairs remain present. This stratum remains in total cost.

## Implemented execution contracts

The opt-in `compiled_exact` executor retains the original `optimized_exact`
and gathered/reference paths. Complete rows are certified before joining
when every occupied source hub is active. Other rows use exact int64
support-word intersections and direct row-owned CSR writes; no duplicate
query-hub-source path array or global pair sort is formed. A union-complete
row is promoted to implicit complete support even if the sufficient
certificate failed. Logical raw-path counts remain diagnostic counts;
expanded path allocations are zero on this executor.

Live scalar priors remain FP64 products of omega, membership and query
density. Selected nonpositive/nonfinite priors raise instead of silently
clamping a support edge. Scalar workspaces and fine-feature tiles have
separate bounds. Complete physical banks use rectangular batching where
possible; irregular row blocks retain source ownership.

QM splits only the first affine layer into source, global and relative
coordinate terms. Prepared source/global affine terms are reused; every
pair still receives the identical nonlinear remainder and the original
module-count scaling. Checkpointed tiles gather compact inputs internally,
so pre-expanded hidden tensors are not retained for backward.

The optional Triton QE reader performs online normalized attention over
complete or CSR support with FP64 prior/log/normalizer arithmetic. It
returns first derivatives to live priors, Q/K/V and geometry bias; the Torch
geometry network carries these derivatives to coordinates. Higher-order
requests use the maintained Torch recomputation path. CPU-only imports and
the default Torch backend remain supported. The current fused prototype
keeps geometry-feature/MLP evaluation separate from attention fusion.

## Execution measurements

Fresh-process parent measurements use the same explicit epoch-2500 weights,
FP32 neural computation, historical FP64 route/normalizer semantics, and
physical RTX 6000 Ada GPU 0. No precision mode is enabled by the benchmark.
Inference uses 8,192 queries, receiver chunk 2,048, two warmups, and five
synchronized repetitions. Parent full-forward medians are 77.17 ms for
0273 and 74.91 ms for 0653; prepared decode is 34.94 and 35.11 ms.

Real training steps use B=48, Q=1,024, receiver chunk 128, one warmup and
three repetitions, with restored parent optimizer state. Parent medians
are 1.178 s (M1) and 3.763 s (M12). Peak allocated memory is 4,463.9 MiB
and 22,361.4 MiB respectively. These are fresh measurements, not borrowed
historical speedup percentages. JSON records include per-repeat baseline,
incremental/absolute allocated peak, reserved peak, and live allocation.

Final same-revision executor tables and Run 2002 progress are recorded
below. Source-level sparsity, fewer rows, or a fused
kernel alone are not evidence of a speedup.

### Final anchor inference measurements

These canonical `*_q2048.json` measurements ran sequentially in fresh
processes without a profiler or concurrent GPU-0 workload. Each entry is a
five-repeat median in milliseconds; the parent reference timings were
measured earlier under the same protocol, so small differences are not
claimed as statistically robust speedups.

| Executor / checkpoint | 0273 full | 0653 full | 0273 prepared decode | 0653 prepared decode |
|---|---:|---:|---:|---:|
| 2001 historical optimized Torch | 77.17 | 74.91 | 34.94 | 35.11 |
| 2001 compiled Torch | 76.24 | 72.33 | 31.59 | 31.32 |
| 2001 compiled split Triton | 71.86 | 75.60 | 32.27 | 33.95 |
| 2101 historical optimized Torch | 83.71 | 78.98 | see JSON | see JSON |
| 2101 compiled split Triton | 105.90 | 115.79 | 50.38 | 51.05 |
| Dense 1804 historical | 34.98 | 35.13 | see JSON | see JSON |
| Legacy 1401 historical | 25.77 | 25.94 | see JSON | see JSON |

The frozen 2101 comparator therefore **regresses** with this executor/backend.
It remains on its unchanged historical path. Neither the compiled path nor
Triton becomes a new default. Run 2001 shows workload-dependent changes,
not a universal latency improvement.

### Final full training-step execution comparison

The following measurements use the maintained epoch-step implementation,
B48/Q1024/chunk128, restored parent optimizer, one warmup and three measured
steps. These are comparable to the original parent measurements above;
the separate fresh-optimizer smoke has a different reporting boundary.

| Parent executor/backend | M1 median seconds | M12 median seconds | M1 peak allocated MiB | M12 peak allocated MiB |
|---|---:|---:|---:|---:|
| Historical optimized Torch | 1.1778 | 3.7632 | 4463.9 | 22361.4 |
| Compiled Torch | 1.2070 | 3.9532 | 3924.9 | 23197.8 |
| Compiled split Triton | 1.1772 | 3.7416 | 13018.5 | 33893.7 |

There is **no robust general speedup** over the historical executor.
Compiled Torch saves M1 memory but is slower in both measured strata;
Triton has only a small time advantage with a substantial memory regression.
Consequently the managed candidate uses **compiled Torch**. The Triton
prototype remains explicit opt-in, with its positive gradient validation
and negative resource results both retained. No scientific loss coefficient
or candidate is changed by this execution-backend choice.

### Fused reader validation and hardware boundary

`triton_qe_reader_validation.json` contains the isolated shape measurements,
FP64 variant, tiny-prior test (Pi=1e-300), higher-order fallback check and
compiled kernel metadata. The standalone reader is slower: 0.329 versus
0.319 ms forward and 2.077 versus 1.145 ms backward. The full-model M12
training measurement can nevertheless improve because it changes the
surrounding gather/reduction workload; the isolated number is not erased.

On RTX 6000 Ada / Triton 3.2.0 the inspected complete-row kernel used 154
registers and zero spills in forward; backward used 255 registers and ten
spills, with four warps and 1,024 shared bytes. Source K/V, bias and prior
gradient accumulation uses atomics, so ordering is not bitwise deterministic.
The Triton full-model step also retained about 11,968 MiB (M1) and
29,037 MiB (M12) live after return, versus about 88 MiB on the Torch
reference. Retained custom-autograd/checkpoint state is a suspected cause;
the CSR geometry path additionally retains a global geometry graph. A
long-run memory-lifetime fix has not been established, and this prototype
is not selected for managed training. No full Fourier/geometry-MLP fusion was benchmarked. Occupancy, global-memory
traffic/load efficiency and larger fixed-support scaling remain unmeasured.

Actual-checkpoint full-loop checks pass on both 2001 and frozen 2101.
The current 2001 Torch field relative differences are 1.68e-7 and 1.29e-7;
its largest non-near-zero parameter/coordinate gradient relative difference
is 3.12e-5. Triton 2001 field differences are 1.62e-7 and 1.93e-7.
No newly disconnected parameter gradients were observed. These bounded
checks are numerical executor validation, not the deferred 90-case child
fidelity evaluation.

### Retained development findings

The first compiled reader was **slower**, not faster: approximately
139–146 ms full forward and 3.82/9.62 s for the M1/M12 optimizer batches.
Complete QE tiles were unnecessarily small, and per-case loops lost the
old reader's rectangular batching. The implementation was revised to keep
batched complete-support reads, bound scalar matrix products independently
of fine MLP tiles, and checkpoint the compact QM computation. Final tables
must use the revised implementation, not these failed-prototype timings.

Actual-checkpoint parity also exposed a loaded-`LazyLinear` issue:
`in_features` remained zero despite materialized weights. The affine split
now reads the authoritative weight shape. After that fix, full physical-loop
field relative differences on the two anchors were approximately
1.4–1.7e-7, with no disconnected parameter gradients. Near-zero attention
key-bias gradients require an absolute check: their exact softmax-shift
derivative is zero, but the FP32 reference accumulated norms around 1e-6.
The parity driver uses absolute 1e-6 only below reference norm 1e-5;
other parameter gradients retain the 1e-4 relative target.

### Fixed reader-cost weights

The fixed-support microbenchmark used the original Torch reader and frozen
prepared states on 0273/0653, Q=128, two warmups and five repetitions.
Complete environmental reads over 192 sources cost 0.690/0.726 ms;
the 96-source packed reads cost 0.934/0.974 ms. The estimated environmental
marginal slopes are negative because dispatch overhead outweighs the saved
fine rows. Consequently **c_M=c_E=1** is the plan's explicit fallback.
The science term is labeled a **pair-count surrogate**, not a fitted
wall-time objective. The measurements are in `fixed_support_reader_costs.json`.

### One fixed scientific coefficient

Exactly four scalar parameters are added to the core, one log-temperature
for each source-module, source-environment, query-module and
query-environment relation. All start at zero; temperatures are their
exponentials. No temperature floor, schedule or other regularizer is added.
The warm start strictly loads the complete parent state, including buffers,
with only these four explicitly initialized entries. The optimizer is fresh
with the parent's learning-rate, decay and clipping policy.

`paircost_calibration.json` records the single two-batch calibration, with
no optimizer updates. Costs are fixed at c_M=c_E=1 and epsilon=0.05. The
objective sums the live induced-pair numerator and valid-pair denominator
across every P0/P1/P2 read; it is not a per-chunk average. Aggregate norms
are the square root of the sum of squared per-batch router-gradient norms.
The fixed coefficient is **0.0024756277369438542**, giving the prescribed
initial 2% cost/task gradient ratio on the aggregate. Calibration includes
the additional P2 port-consistency read. Per-receiver components are masked
by module validity before the physical port reads enter the ratio. A
pre-launch incomplete-assembly calculation was corrected before any
training; there was no candidate or coefficient sweep.

| Batch | Physical loss | Pair-count surrogate | Task router-gradient norm | Cost router-gradient norm |
|---|---:|---:|---:|---:|
| M1, B48, Q1024 | 0.00900216 | 0.9523809524 | 0 | 0 |
| M12, B48, Q1024 | 0.01206032 | 0.9361627103 | 0.0081416396 | 0.0657743446 |

At initialization the field differs from the same-checkpoint exact parent
by zero on M1 and 1.59e-7 in relative norm on M12. The positive constant
M1 cost cannot create sparsity: its derivative is zero. No second
coefficient, candidate, mean shift, penalty or sweep is proposed.

### Validation before managed training

The combined CPU regression run passed **185 tests**, with five explicit
CUDA skips because that command masked CUDA. A separate actual GPU-0 run
passed **15 kernel/compiler tests**, including all Triton CUDA cases.
Historical loading/resource checks, optimizer behavior, interface models,
source-measure transitions, variable supports, map export, and the new
four-parameter/port-validity contracts are included. Differential Ruff
checks separate pre-existing whole-file typing warnings from new code;
new implementation/test/tool files pass their focused checks.

`optimizer_smoke.json` records real full-model forward/loss/backward/clip/
AdamW updates on both fixed B48 buckets with fresh optimizers and the same
Triton backend; `optimizer_smoke_torch.json` repeats the matched comparison
for the selected Torch backend. No run directory or trained checkpoint is created by this
check. All gradients are finite; all four M12 temperature updates are live.
M1 temperatures correctly remain unchanged. Candidate initialization
matches the exact parent by 0 (M1) and 1.59e-7 (M12) relative field norm.

### Reproducible tools

All commands run from `HONF_Proj` with the existing
`/home/wanglz/miniconda3/envs/ModularDT/bin/python` and
`CUDA_VISIBLE_DEVICES=0`. The tools accept explicit `LABEL=CHECKPOINT` paths:

- `tools/diagnostics/diagnose_sparse_execution.py`: bounded projection and
  exact post-union audit; no managed training run.
- `tools/diagnostics/check_sparse_execution_parity.py`: same-checkpoint full
  physical output and first-derivative checks for Torch or Triton.
- `tools/diagnostics/benchmark_sparse_execution.py`: fresh-process inference
  or real optimizer-step benchmarks, with optional separate profiler traces.
- `tools/diagnostics/measure_sparse_reader_costs.py`: fixed-support marginal
  reader-cost experiment, including negative slopes.
- `tools/diagnostics/calibrate_sparse_paircost.py`: exact supervised-loss
  assembly and one coefficient from two fixed batches, without updates.
- `tools/diagnostics/smoke_sparse_optimizer.py`: same-backend candidate and
  exact-parent fresh-optimizer execution, finite gradients and updates.

## Managed Run 2002 and bounded observation

The only scientific candidate is
`Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_2002_20260917_191739_routed_module_hubs_sparse_cost/`.
It was launched on physical GPU 0 as PID 1996304 with a maximum of **500
additional epochs**. The log confirms 326 loaded parameters, no skipped,
missing or unexpected keys, and exactly four zero-initialized parameters.
The maintained optimizer inventory contains 4,505,270 trainable scalars,
exactly four more than the parent. All parent state/buffers are loaded.

The launch command is recorded in `run2002_launch.json`; stdout/stderr is
`run2002_launch.log` under the evidence root. Managed artifacts are the
existing `config_resolved.json`, `run_manifest.json`, `metrics.csv`, ordinary
best/latest files and local epoch milestones 10, 50, 100, 250 and 500.
No initial baseline checkpoint or new monitoring service was created.

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=0 /home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py --config src/config_core/forward/routing_module_hubs_sparse_cost_context.json --device cuda:0 --epochs 500 --initialize-checkpoint /home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_2001_20260916_170829_routed_module_hubs_quickcheck_optimized/epoch_2500_model.pt --yes
```

This command documents the already-running process; do not start a second
copy. Monitoring is limited to the existing log/CSV through local epoch 50,
then the process is left to finish its configured 500 epochs. The existing
training validation loop is retained. No additional 90-case endpoint
fidelity/tail/interface/omitted-source evaluation is run here.

The matched fresh-optimizer Torch comparison immediately before launch was:

| B48 bucket | Exact parent median seconds | Candidate median seconds | Parent peak allocated MiB | Candidate peak allocated MiB | Candidate peak reserved MiB |
|---|---:|---:|---:|---:|---:|
| M1 | 1.02865 | 1.01350 | 3965.4 | 4150.3 | 4332 |
| M12 | 3.62857 | 3.86203 | 23316.9 | 23903.4 | 27330 |

`optimizer_smoke_torch.json` retains all repetitions, absolute/incremental
allocation, reserved memory, gradients and parameter updates. This
full-model step boundary includes forward, maintained loss, backward,
clipping and optimizer update, but excludes data loading and diagnostic
scalar inspection. Initial M12 candidate execution is about **6.4% slower**
than the exact parent with the same backend; the small M1 difference is not
claimed as a robust gain. Initial sparsification has not earned a speedup.

### Completed first-50 observation

`first50_training_observation.json` records only local epochs 1–50 from the
existing CSV. The scheduled `epoch_0050_model.pt` exists. PID 1996304 was
still active with `--epochs 500` when observation ended; parent PIDs 919468
and 919469 also remained active with their original commands. Monitoring
has ended; no process was stopped or extended.

| Existing training metric | First five mean | Last five mean | First five median | Last five median |
|---|---:|---:|---:|---:|
| Training physical loss | 0.01071358 | 0.00760210 | 0.00936743 | 0.00764413 |
| Validation physical loss | 0.01424551 | 0.01192126 | 0.01438003 | 0.01187466 |
| Pair-count surrogate | 0.94566131 | 0.94010132 | 0.94561565 | 0.94002451 |
| Training seconds per epoch | 42.7942 | 49.0558 | 43.1770 | 48.9417 |
| Validation seconds per epoch | 3.4483 | 3.5144 | 3.4696 | 3.5049 |

At epoch 50, training/validation physical losses are 0.00764413/0.01182137.
All observed active loss, timing and temperature fields are finite; optional
disabled diagnostic columns retain their expected NaNs. This establishes
numerical continuation, not smooth convergence: substantial spikes occurred
at epochs 26–29 and 37–38. The maximum validation physical loss was 0.02218926
at epoch 37, and maximum training physical loss was 0.01871641 at epoch 38.
The last-five means improved by about 29.0% (training) and 16.3% (validation)
relative to the first five, but fidelity acceptability remains unverified.

The epoch-50 validation-pass temperatures, in source-module,
source-environment, query-module, query-environment order, are
0.9924012, 0.9349307, 1.0006180, and 0.9206663. These are the post-update
values; the training CSV temperatures are averages during the epoch.
The pair-cost mean fell only about 0.59%; it is not a measured true-pair
reduction. Training epoch time increased about 14.6% across these windows.
These timings are descriptive, not a matched parent-control experiment,
and do not establish the cause of the increase. Together with the matched
pre-launch comparison, they **do not ensure a speedup**. No runtime, smooth
stability or full-fidelity success gate is declared passed.

The sum of reported training and validation time through epoch 50 is
2,466.21 seconds (41.10 minutes), excluding startup/checkpoint overhead.
No checkpoint inference, endpoint gradients, new support audit or 90-case
assessment was performed during this observation.

## Commands and managed artifacts

Commands below use the existing environment and do not modify parent jobs.
The benchmark/check scripts never allocate a training run or save weights.
Only the final `train.py` command starts Run 2002.

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj
HONF_PY=/home/wanglz/miniconda3/envs/ModularDT/bin/python
HONF_EVIDENCE=diagnostics/generated/interface_operator_study/dynamic_sparse_routing/sparse_execution_next
HONF_PARENT=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_2001_20260916_170829_routed_module_hubs_quickcheck_optimized/epoch_2500_model.pt

rtk proxy env CUDA_VISIBLE_DEVICES=0 "$HONF_PY" tools/diagnostics/diagnose_sparse_execution.py --checkpoint "run2001=$HONF_PARENT" --query-count 32 --output "$HONF_EVIDENCE/projection_diagnosis_2001.json"
rtk proxy env CUDA_VISIBLE_DEVICES=0 "$HONF_PY" tools/diagnostics/measure_sparse_reader_costs.py --checkpoint "run2001=$HONF_PARENT" --output "$HONF_EVIDENCE/fixed_support_reader_costs.json"
rtk proxy env CUDA_VISIBLE_DEVICES=0 "$HONF_PY" tools/diagnostics/check_sparse_execution_parity.py --checkpoint "run2001=$HONF_PARENT" --backend triton --output "$HONF_EVIDENCE/parity2001_compiled_triton.json"
rtk proxy env CUDA_VISIBLE_DEVICES=0 "$HONF_PY" tools/diagnostics/benchmark_sparse_execution.py --checkpoint "run2001=$HONF_PARENT" --execution compiled_exact --backend triton --output "$HONF_EVIDENCE/parent2001_compiled_triton_q2048.json"
rtk proxy env CUDA_VISIBLE_DEVICES=0 "$HONF_PY" tools/diagnostics/benchmark_sparse_execution.py --checkpoint "run2001=$HONF_PARENT" --execution compiled_exact --backend triton --task train --query-count 1024 --receiver-chunk-size 128 --warmups 1 --repetitions 3 --output "$HONF_EVIDENCE/parent2001_compiled_triton_train.json"
rtk proxy env CUDA_VISIBLE_DEVICES=0 "$HONF_PY" tools/diagnostics/calibrate_sparse_paircost.py --checkpoint "run2001=$HONF_PARENT" --output "$HONF_EVIDENCE/paircost_calibration.json"
rtk proxy env CUDA_VISIBLE_DEVICES=0 "$HONF_PY" tools/diagnostics/smoke_sparse_optimizer.py --checkpoint "run2001=$HONF_PARENT" --backend triton --cost-weight 0.0024756277369438542 --output "$HONF_EVIDENCE/optimizer_smoke.json"
rtk proxy env CUDA_VISIBLE_DEVICES=0 "$HONF_PY" tools/diagnostics/smoke_sparse_optimizer.py --checkpoint "run2001=$HONF_PARENT" --backend torch --cost-weight 0.0024756277369438542 --output "$HONF_EVIDENCE/optimizer_smoke_torch.json"
```

The calibration command documents the completed fixed experiment; it is not
an instruction to retune lambda during the 500 epochs. Torch comparisons
use `--backend torch`; the historical parent uses `--execution optimized_exact`.
The 2101 compatibility check substitutes its explicit epoch-2500 checkpoint
and never trains it. Detailed repeat timings and allocated/reserved memory
are retained in each JSON rather than replaced by headline medians.

## Evidence not yet available

The user deferred Run 2002 full 90-case fidelity, tails, interface/port,
omitted-source, gradient, true-pair, latency, and allocated/reserved-memory
endpoint evaluation until the 500-epoch run is finished. No extension to
2,500 epochs is authorized. Physical influence/barrier claims and large-3D
physical fidelity require independent evidence absent from this dataset.
