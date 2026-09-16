# Exact routing execution optimization

The final exact executor reduced median training time by **68.9% / 65.1%**
and median peak allocation by **34.7% / 42.3%** across the module-hub /
mean-shift first-50 comparisons. Both optimized runs completed 50 epochs with
healthy convergence. Controlled same-GPU optimizer benchmarks confirm the
execution gains, but neither strategy meets the intended learned-sparsity or
Legacy-latency objective.

This investigation accompanies the [four-run epoch-500 comparison](HONF_Routing_Four_Run_500_Comparison.md).
It changes tensor execution rather than the routing formula, fine networks,
three-step fixed-data mean-shift, physical refinement, losses, or optimizer.
Historical profiles keep `routing.execution="gathered"`. The experimental
`optimized_exact` mode is explicit in new quickcheck profiles and checkpoints.

## Why the original implementation is expensive

The model retains Dense's fine preparation and physical P0/P1/P2 reads. The
measured routed models select nearly all environmental sources at epoch 500,
so they incur routing overhead without avoiding much fine neural work. Source
fanout is not the same as fewer unique fine pairs. Run 2100 also retains more
duplicate paths than Run 2000.

The original compiler expands differentiable scalar weights for every positive
query/hub/source path, sorts and coalesces them, then executes the unique fine
pairs. Its saved backward indices and values scale with raw paths. The packed
QE reader then repeatedly gathers queries, keys, and values for those pairs,
uses many small neural tiles, and scatters reductions. This is especially
inefficient when the selected support is actually complete.

For scale, B=48, Q=1024, and E=192 imply 9,437,184 environmental
pairs in P2 alone when support is complete. One materialized H=256 float32
vector per pair would occupy 9 GiB; queries, keys, values, neural intermediates,
and backward storage can multiply that cost. Tiling and checkpointing therefore
matter far more to peak activation memory than the few million parameter
scalars. This calculation is illustrative, not a claim that every such buffer
is simultaneously live in the measured implementation.

The previous [memory investigation](HONF_Routing_Memory_Diagnostics.md) found
stable live allocation floors and growing allocator reservations, not a
reproduced persistent tensor leak. Cache policy alone cannot remove the
underlying activation and dispatch costs.

## Exact implementation changes

**Compact differentiable priors.** Enumerate/coalesce integer support first,
then return `Pi[b,q,i] = omega[b,i] * sum_k A[b,i,k] * d[b,q,k]` for unique
pairs. The initial prototype used checkpointed selected-pair tiles; the final
implementation uses bounded FP64 batched scalar products and gathers only the
selected priors. Scalar blocks can include unselected entries, but these never
trigger fine neural pair evaluation. Both A and d retain their original
active-set masks and gradients; source measures remain live. Route products
use FP64 before multiplication. Reconstructing A from the filtered inverted
incidence preserves source validity and measure filtering. This removes the
saved duplicate-path graph, while integer support discovery still costs work.

**Complete-support QE.** Only after constructing the selected pair union, a
complete environmental support can use batched QK and attention/value products
instead of repeated packed gathers. The original geometric Fourier features,
geometry-bias MLP, learned log-prior, FP64 softmax, value dtype, and output
projection are retained. Bounded receiver tiles are checkpointed as a whole.
Completeness is checked per batch element, so a mixed batch can use batched products for complete cases and the packed executor for partial cases. The output projection is applied once after combining their unprojected contexts, including empty-support cases. This is an explicit exact
execution choice, not dropping routing or replacing learned priors by uniform
weights. Routing overhead remains included in timings.

Neither change reduces model parameters or removes Dense preparation. An
execution optimization can improve the existing operator's cost; it cannot
establish a learned-sparsity advantage when nearly all fine pairs remain active.

## Libraries and precision

The implementation uses the installed PyTorch 2.6.0+cu124 stack: batched
matrix products, integer indexing/sorting, differentiable reductions, and
non-reentrant activation checkpointing. No runtime/dependency migration is
needed to test these changes.

FlashAttention/SDPA is not a direct replacement for the whole fine operator:
QM is a nonlinear pair MLP, and QE includes a learned geometric pair bias and
live FP64 route priors. PyTorch's fused attention kernels have input limitations;
its documented math backend supports FP64. Whether a fused kernel is actually
eligible must be measured rather than inferred from calling the SDPA API.
See the [PyTorch 2.6 SDPA documentation](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html).

Whole-model `torch.compile` is also not an assumed speedup. Routing uses
data-dependent `nonzero`, scalar counts, and variable packed sizes; these are
known compiler graph-break/data-dependent-shape concerns. Compiling stable
fine kernels separately is a potential subsequent measurement, not a result
claimed here. See [PyTorch's compiler troubleshooting guidance](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/dynamic_shapes_troubleshooting_guardon_errors.html);
that page describes newer development behavior and is not a promise for the
installed 2.6 runtime.

Mixed precision, TF32, changing softmax precision, changing routing temperature,
or capping support would introduce additional numerical/scientific variables.
They are not silently bundled into this exact-execution comparison.

## Validation and measurements

An initial physical Run-2100 replay on GPU 2, restoring the epoch-500 optimizer,
used B=48 and Q=1024 with fixed and alternating module-count buckets. Peak
allocated memory in the largest alternating batch fell from 33,277.9 MiB to
23,096.0 MiB (30.6%). The small fixed batch fell from 6,360.6 to 4,453.9 MiB.
Live allocation floors stayed stable. These were instrumented diagnostics of
the initial whole-batch shortcut, not the final hybrid implementation or a
latency benchmark. They establish a memory improvement only.

Artifacts: [reference memory replay](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/optimization/quick_run2100_reference_memory.json) and [optimized memory replay](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/optimization/quick_run2100_optimized_memory.json).

The first matched wall-clock measurements are below. The initial CPU compiler/common routing suite passed
38 checks at that point; the current focused routing regression suite passes 79 tests,
including active/inactive gradients, tiny positive priors, permutation, empty
support, and mean-shift compatibility. New configurations retain historical
defaults and were dry-run checked before any quickcheck launch.


### First matched optimizer benchmark

Physical GPU 2; B=48, Q=1024; identical real case-ID batches per module-count
bucket; one warmup and three measured optimizer steps; all checkpoint optimizer
states restored; canonical losses and clipping unchanged. Reported latency is the
median elapsed time of the three measured steps, while peak allocated memory is
the maximum across those steps. Counter callbacks ran only in a separate untimed
forward. All 16 measurements completed.

| Model / execution | M=1 median seconds | M=1 peak allocated MiB | M=12 median seconds | M=12 peak allocated MiB |
|---|---:|---:|---:|---:|
| 1401 historical | 0.300 | 2,778.8 | 0.603 | 24,550.9 |
| 1804 historical | 1.031 | 5,843.1 | 2.164 | 26,764.9 |
| 2000 historical | 3.140 | 6,148.5 | 7.944 | 30,191.2 |
| 2000 compact priors only | 3.933 | 5,561.4 | 10.244 | 24,460.1 |
| 2000 compact priors + hybrid QE | 2.093 | 4,241.8 | 6.555 | 22,101.4 |
| 2100 historical | 3.157 | 6,150.2 | 8.164 | 33,135.4 |
| 2100 compact priors only | 3.924 | 5,563.1 | 10.421 | 24,521.7 |
| 2100 compact priors + hybrid QE | 2.076 | 4,243.4 | 5.476 | 20,988.9 |

The compact-prior change alone reduces memory but increases latency. Combined
with the hybrid QE path, large-batch latency improves 17.5% for module hubs and
32.9% for mean shift; peak allocation improves 26.8% and 36.7%. The gains are
real, but this implementation remains much slower than both baselines. It is
not yet evidence of a lightweight model. The first forward's field and thermal
losses agree across variants to ordinary floating-point rounding.

[four_run_benchmark.json](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/optimization/four_run_benchmark.json)
retains every repetition, batch identity, loss, optimizer policy, and execution
counter; [four_run_benchmark.csv](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/optimization/four_run_benchmark.csv)
is the compact table.
The counters show the hybrid path actually executes on the large batches:
2000 uses 87 complete-support tiles plus 427 packed score tiles, and 2100 uses
115 plus 58. A whole-batch completeness counter alone would miss these gains.
These buckets test resource extremes, not an average full training epoch.

At M=12 the combined implementation uses less peak activation memory than
Legacy 1401 (20,989–22,101 versus 24,551 MiB), but at M=1 it still uses more
(about 4,243 versus 2,779 MiB). Thus “lighter” depends on whether it refers to
weights, activation peaks, or latency. Parameter count is unchanged, large-batch
activation memory improves, and the latency gap to Legacy remains large.


### Attribution before the next correction

A separate warmed Run-2100 M=12 optimizer-step profile recorded 274,763 CUDA
kernel launches, 20,546 `aten::index` calls, 6,649 `MulBackward0` calls, and
2,027 stream synchronizations. The pair-join CPU scope occupied about 1.229 s
inclusive (0.447 s self); its attributed child CUDA operations occupied about
0.195 s. GPU annotation ranges overlap these child events and must not be
summed as independent kernel costs. Profiler wall time includes substantial
instrumentation overhead and is not used for the speedup calculation.

This evidence supports reducing repeated gathers, small prior tiles, and
Python/autograd dispatch. It does not identify three-step mean shift itself
as the dominant cost. The next isolated prototype replaces per-selected-pair
K-wide prior products with bounded FP64 batched scalar products, while keeping
the same integer union and evaluating fine neural work only for selected pairs.

Profile artifacts: [profile JSON](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/optimization/run2100_m12_compiler_denseqe_profile.json)
and [Chrome trace JSON](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/optimization/run2100_m12_compiler_denseqe_trace.json)
in the optimization output directory.


### What exact execution optimization cannot fix

For source-measure sparsemax, `d_k = max(z_k - tau, 0)` with
`sum_k mu_k d_k = 1`. Attraction does not by itself force a sparse active set.
If candidate scores remain similar, many candidates remain active; overlapping
source memberships can then reach almost every fine source after the union.
This is consistent with the measured nearly dense support and increased raw-path
multiplicity in Run 2100. It is an explanation of the observed mechanism, not
a demonstrated causal claim about which learned score term dominates.

A faster exact executor must preserve those same supports. It can remove
implementation overhead, but cannot turn this evidence into successful learned
sparsity. Score sharpening, support penalties, different candidate generation,
or reduced fine networks would be separate mathematical/model experiments and
are outside the execution-only changes measured here.


### Batched-prior correction selected for fresh checks

The next matched benchmark retained the original integer union and hybrid QE,
but evaluated live priors using bounded FP64 `bmm` blocks instead of many
checkpointed P-by-K gathers. The scalar block budget is 262,144 entries;
prior checkpointing is disabled for this variant, while fine neural activation
checkpointing remains enabled. This is the implementation selected for
`optimized_exact`; the reference and first compact-prior algorithms remain
available to reproduce the measured comparisons.

| Run / batch | Historical seconds | Batched seconds | Latency reduction | Historical peak MiB | Batched peak MiB | Memory reduction |
|---|---:|---:|---:|---:|---:|---:|
| run2000 / M1 | 3.140 | 1.225 | 61.0% | 6,148.5 | 4,410.9 | 28.3% |
| run2000 / M12 | 7.944 | 4.257 | 46.4% | 30,191.2 | 22,584.3 | 25.2% |
| run2100 / M1 | 3.157 | 1.221 | 61.3% | 6,150.2 | 4,412.6 | 28.3% |
| run2100 / M12 | 8.164 | 3.234 | 60.4% | 33,135.4 | 21,463.2 | 35.2% |

All four measurements completed with restored optimizer states. First-step field,
thermal, and total losses agree with the historical implementation within
floating-point rounding. Mixed complete/partial large batches exercised both
QE paths: 2000 had 824 complete case/read rows and 280 partial rows; 2100 had
1,069 complete rows and 35 partial rows. No omitted fine pair was evaluated.

[batched_prior_benchmark.json](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/optimization/batched_prior_benchmark.json)
is the artifact for this benchmark. The additional scalar products trade
a small amount of memory relative to the first compact-prior prototype for a
large reduction in dispatch time. Relative to Legacy 1401, latency remains
about 4.1 times larger at M=1 and 5.4–7.1 times larger at M=12. Relative to Dense
1804, it remains about 1.2 times larger at M=1 and 1.5–2.0 times larger at M=12.
This is a substantial execution improvement, not proof of a sparse-work advantage.


### Fresh training checks

Run 2001 (`Run_2001_20260916_170829_routed_module_hubs_quickcheck_optimized`)
completed 10 fresh epochs on GPU 2. Compared with the original Run 2000's first
10 epochs, median training time fell from 71.822 to 22.317 s (68.9%), and median
peak allocation from 34,189.8 to 21,220.3 MiB (37.9%). For module hubs, the first
validation field and temperature errors match at logged precision. At epoch 10, field MSE is
0.491720 versus 0.500579, and temperature MSE 0.741283 versus 0.792761. These
small early-run differences are not an accuracy improvement claim. All recorded
gradient and parameter-update diagnostics at epochs 1, 2, 5, and 10 are finite.

The optimized implementation preserves the mathematical operator, not bitwise
reduction order. Numerical differences can accumulate during training; the
50-epoch follow-up evaluates the trajectory and trailing-window quality.


Run 2101 (`Run_2101_20260916_171305_routed_mean_shift_quickcheck_optimized`)
also completed its 10-epoch check. Median training time fell from 66.648 to
23.029 s (65.4%) and median peak allocation from 37,250.1 to 21,440.0 MiB
(42.4%). Its first validation field matches the historical value; temperature
differs by 1.2e-7, approximately one float32 ULP. At epoch 10, field MSE is 0.615807 versus the original 0.651141;
temperature MSE is 0.788978 versus 0.780901. Recorded gradients/updates are
finite at epochs 1, 2, 5, and 10. Both strategies therefore meet the user's
condition for the bounded 50-epoch convergence check.


The fresh-versus-historical epoch timings are operational observations from
different launches, not a controlled hardware-isolation experiment. Original
Run 2000 used GPU 2 and Run 2100 used GPU 1; both optimized checks use GPU 2.
All are RTX 6000 Ada cards. The same-GPU matched optimizer benchmark above is
the controlled execution comparison. Its reference and optimized variants use
the same expandable-segments allocator setting. The fresh checks additionally
show whether the improvement persists in the canonical full-data workflow.


### Completed 50-epoch follow-up

Both fresh optimized checks completed exactly epochs 1–50 on physical GPU 2,
with completed manifests, exit code 0, and `epoch_0050_model.pt` checkpoints.
Each ran fresh through 10, then resumed its own checkpoint through 50. There
are no missing or duplicate epoch rows. Core losses and field/temperature MSEs
are finite throughout; all scheduled gradient/update captures at epochs
1, 2, 5, 10, 20, and 50 are finite. No OOM or CUDA failure occurred.

The table summarizes the **first 50 epochs**. Memory is the median of logged
per-epoch peak allocated MiB, not reserved memory or a single-step peak.

| Strategy | Median train seconds: original → optimized | Reduction | Median peak allocated MiB: original → optimized | Reduction |
|---|---:|---:|---:|---:|
| Module hubs (2000 → 2001) | 71.407 → 22.201 | 68.9% | 32,403.1 → 21,153.5 | 34.7% |
| Mean shift (2100 → 2101) | 65.295 → 22.811 | 65.1% | 37,072.4 → 21,381.3 | 42.3% |

The maximum per-epoch allocation over these checks was 22,815.4 MiB for Run
2001 and 21,553.6 MiB for Run 2101, versus 35,050.5 and 38,322.7 MiB in the
original first-50 histories. Allocation stayed bounded as training progressed.
Epoch CSVs do not contain reserved-memory traces; no reserved-memory stability
claim is inferred from allocated values. The earlier controlled memory replay
separately examined allocation floors and allocator reservations.

| Strategy | Epochs 41–50 median field MSE: original → optimized | Epochs 41–50 median temperature MSE: original → optimized |
|---|---:|---:|
| Module hubs | 0.220001 → 0.187919 | 0.137715 → 0.143730 |
| Mean shift | 0.210038 → 0.194916 | 0.169409 → 0.143354 |

The full curves retain comparable descending early-training behavior, with
ordinary oscillations. Module-hub temperature is slightly worse on this
trailing median, while the other listed medians improve. This is a successful
bounded convergence-health check, not statistical accuracy equivalence or an
accuracy gain attributable to faster execution. One seed, floating-point
reduction-order differences, and only 50 optimized epochs cannot establish
mature performance. The original epoch-500 endpoint results remain historical
results, not measured epoch-500 results of the optimized runs.

![Original and optimized first-50 trajectories](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/optimization/quickchecks/routing_quickchecks.png)

The [JSON](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/optimization/quickchecks/routing_quickchecks.json)
and [CSV](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/optimization/quickchecks/routing_quickchecks.csv)
retain every epoch, source path, coverage check, and scheduled-gradient result.

## Recommendation for longer training

The 500-epoch evidence supports further **accuracy/convergence investigation**,
not a claim that either strategy has achieved efficient learned sparsity.
Both routed models improve their last-50 validation medians relative to the
preceding 50 epochs and are competitive with Dense 1804 on those windows.
Run 2000's poor exact-500 field score should not erase its strong saved
through-500 checkpoint or its late-window behavior. Run 2100's good exact-500
field result should not erase its final-port temperature/effective-h regressions.
Missing selected-through-500 parent weights prevent a fair four-way selected
checkpoint ranking.

Do not launch unchanged gathered-executor runs expecting more epochs alone to
fix cost: environmental fine support is already almost complete, mean shift
has not reduced it, and routed parameter counts exceed Legacy's. The tested
exact executor is the better starting point for any further routed experiment.
Its improvements come from removing implementation overhead, not from learning
a sparse operator. A bounded next assessment at epoch 500, followed by a
convergence-based decision, is more justified than committing directly to 5000.
There is not enough evidence to select one routing strategy as the scientific
winner. Module hubs are the simpler candidate construction; mean shift benefits
more from the complete-support execution path precisely because its support is
more dense.

If the product objective is specifically a light, fast model, neither routed
strategy currently displaces Run 1401. Dense 1804 is also faster in the matched
optimizer benchmarks. Further work on stable-kernel compilation could be
measured without changing the equations; support sharpening, architecture
reduction, mixed precision, or fused-attention precision changes would need
separate scientific controls. No such changes, additional library migration,
long training, or Goal 3 were executed here.


## Validation scope and command record

The current focused compiler, common sparse-executor, reader, integration,
configuration, and mean-shift suite passed **79 tests**. Diagnostic map export
and endpoint reduction passed **13 tests**. An independent CPU review also
checked 40 randomized compiler-equivalence seeds and 10 checkpoint-gradient
cases. The batched scalar-product path was compared with the reference across
small and default tile budgets, with and without checkpointing. Configuration
schema validation passed for both historical and optimized routing profiles.
The source uses the historical gathered executor by default, and the physical
benchmark loaded all four historical checkpoints and their optimizer states.

The [executed command record](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/optimization/executed_commands.md)
contains physical replay, controlled benchmark, and fresh-training commands.
The separate profiler JSON/trace remain attribution evidence, but their exact
launch invocation was not retained; the command record explicitly notes this
provenance limitation. The unprofiled benchmark invocations and per-repetition
results are retained and support all claimed speedups.


## Unexecuted continuation commands

These are handoff commands, **not executed**. Run only one at a time on physical
GPU 2 after deciding to continue. Each restores its own epoch-50 optimizer and
RNG state and stops at total epoch 500; no restart from Run 2000 weights is
involved. The fresh optimized checkpoints already provide a suitable starting
point, so another fresh seed-0 launch would duplicate the quickcheck work.
Working directory: `/home/wanglz/Desktop/src/ModularDT/HONF_Proj`.
The CLI applies `--epochs` to the effective training configuration before
launch. On resume, `config_resolved.json` and newly saved checkpoints reflect
the new endpoint; `configs/resolved_config.json` retains the initial launch
configuration (10 epochs for these quickchecks). Read actual checkpoint epochs
and the current manifest when determining completed progress.

Module hubs, same Run 2001:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:Case_ThermalChannel/src OMP_NUM_THREADS=4 /home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py --config src/config_core/forward/routing_module_hubs_optimized_context.json --device cuda:0 --epochs 500 --resume-checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_2001_20260916_170829_routed_module_hubs_quickcheck_optimized/epoch_0050_model.pt --yes
```

Mean shift, same Run 2101:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:Case_ThermalChannel/src OMP_NUM_THREADS=4 /home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py --config src/config_core/forward/routing_mean_shift_optimized_context.json --device cuda:0 --epochs 500 --resume-checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_2101_20260916_171305_routed_mean_shift_quickcheck_optimized/epoch_0050_model.pt --yes
```

For the original Goal-2 handoff, the following preserves historical Run 2100's
gathered configuration and GPU-1 assignment and continues from 500 to total
1000. It is also **unexecuted** and is less cost-efficient than the optimized
checks; it is supplied to preserve the same-run continuation option rather
than silently change that run's execution configuration.

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:Case_ThermalChannel/src OMP_NUM_THREADS=4 /home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py --config src/config_core/forward/routing_mean_shift_context.json --device cuda:0 --epochs 1000 --resume-checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_2100_20260916_002148_routed_mean_shift/epoch_0500_model.pt --yes
```
