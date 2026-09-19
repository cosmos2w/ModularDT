# Run 1407 Phase-Shared Routing Report

Date: 19 September 2026  
Repository/branch: `cosmos2w/ModularDT`, `agent/honf-core-next`  
Implementation baseline: `3da43e607efb83c844eac1da5a692dd05a6e60f3`  
Run-1407 training source revision: `cbd8f3cd52ab70c0b2a688c83d8c3ad4b6292860`

## Continuation addendum: user-authorized epoch 500 test

After the original epoch-50 assessment below, the user explicitly requested
that the same Run 1407 be extended to epoch 500 because its convergence curve
may differ from the parent models. On that subsequent instruction, the exact
epoch-50 checkpoint was resumed on physical GPU 1 from clean source revision
`4239af4e0e75a1bf246cd2f2a4d7a8e337504dfc`.

The maintained resume path restored model, optimizer, scaler, and RNG state
and reported `continuing at epoch 51 / 500`. Epoch 51 completed with finite
training loss `1.8927`, validation loss `2.0469`, validation field MSE
`1.3735`, and validation temperature MSE `0.2823`; the same managed process
then continued normally. No monitoring was requested, so it was left
unattended after startup verification. This addendum changes the execution
status, not the measured epoch-50 comparison or its original budget judgment.

## Executive decision

The requested opt-in `phase_shared_group_control_honf` was implemented and one
fresh managed Run 1407 was trained through epoch 50. The scientific and
executor changes behave as designed:

- one live K=6/D=16 controller is built from Dense-contextualized P0 sources;
- its memberships, control banks, query keys, and support metadata are reused
  at P1/P2 while Dense MM/ME/EM preparation and the fine QM/QE values refresh;
- query keys use the prescribed prototype-plus-case-control equation with
  parameter-free RMS scaling before entmax15;
- the six-bit index executes unique module-source supports when the calibrated
  hybrid policy predicts a benefit, while QE retains the faster rectangular
  path for broad environmental support;
- `C_g + C_M + C_E`, fine MM/ME/EM, the frozen local surrogate, physical
  losses, P0/P1/P2 coupling, and the accepted complete-QE checkpoint boundary
  remain intact.

Run 1407 is numerically stable and cheaper than Dense 1804 in training, but it
is not reasonably competitive with Run 1406 at epoch 50. Its epochs 41--50
validation-field median is `1.42353`, versus `0.40708` for matched Run 1406
(`3.497x`). Exact epoch-50 field MSE is `1.32424`, versus `0.30036` for Run
1406 and `0.21934` for Dense 1804. This is a material learning/fidelity gap,
not a narrow review-band miss.

The cost result is mixed. Relative to Run 1406, mean preparation plus one query
improves `5.25%`, M12 time improves `1.06%`, and M12 allocated peak improves
`0.25%`; however, full forward is `4.10%` slower and prepared P2 is `19.89%`
slower. Prototype anchoring makes query routing genuinely more selective on
the two cost anchors, but environmental support still covers about `94%` of
valid pairs, so it does not create a useful QE reduction.

Under the plan's research-budget rule, the original epoch-50 assessment did
not recommend continuation. The user subsequently authorized the same run to
continue to epoch 500 to test the different-convergence hypothesis, and that
continuation is now running. No corrective loss, alternate candidate, sweep,
or Run 1408 was created. The 90-case epoch-500 comparison remains deferred
until epoch 500 and a later user command.

## Scope and comparison policy

The design was taken from
`UpgradePlan/HONF_Run1407_Phase_Shared_Routing_Development_Plan.md`. The two
primary input reports were:

- `docs/reports/HONF_Run1406_Epoch500_Comparative_Evaluation.md`;
- `docs/reports/HONF_Run1406_Performance_Diagnosis_and_Exact_Optimization_Report.md`.

The implementation preserves newer compatible work after the pinned baseline,
including Run 1406's complete-QE non-reentrant activation checkpoint. Historical
architectures retain their existing class and serialization paths. The matched
benchmark strict-loaded these explicit epoch-50 checkpoints one model at a
time:

| Label | Explicit checkpoint |
|---|---|
| Run 1407 | `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1407_20260919_174751_phase_shared_prototype_group_control/epoch_0050_model.pt` |
| Run 1406 | `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1406_20260919_132837_low_dimensional_group_control_executor_optimized_rerun/epoch_0050_model.pt` |
| Dense 1804 | `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/epoch_0050_model.pt` |

Run 1404 remains historical execution context only. It is not checkpoint- or
equation-equivalent to Run 1406/1407 and was not inserted into the matched
epoch-50 cost ratios.

## New model equation and preserved computation

### Phase-shared controller

P0 still performs the simultaneous Dense fine preparation

\[
  (\widetilde z^{(0)},\widetilde e^{(0)})
  = \operatorname{DenseMM/ME/EM}(z^{(0)},e,g).
\]

The existing source-assignment and group-moment arithmetic then builds one
live controller

\[
  \mathcal I_0=(A^M,A^E,h,g_c,\omega^M,\omega^E,
  \text{control banks},\text{support metadata}).
\]

The same object is passed by identity into P1 and P2. It remains connected to
autograd; it is not detached, serialized, or reused between independent
forwards. At every later physical phase the implementation still recomputes
Dense MM/ME/EM, contextual module/environment states, the module first affine,
and environmental K/V. Thus the approximation is exactly the planned
phase-static structural controller with phase-dynamic physical values, not a
stale-value cache.

### Prototype-anchored query access

Source assignment is unchanged. For each group, Run 1407 forms

\[
  k_k=c_k+W_hh_k,
  \qquad
  \mathcal R(x)=\frac{x}{\sqrt{D^{-1}\sum_a x_a^2+10^{-6}}},
\]

and routes a query with

\[
  \ell_{qk}=\frac{\mathcal R(v_q)^\top\mathcal R(k_k)}{\sqrt D},
  \qquad \alpha_{q:}=\operatorname{entmax}_{1.5}(\ell_{q:}).
\]

There is one `1/sqrt(D)` factor and no new temperature, trainable scalar,
regularizer, or loss. Run 1407 and Run 1406 both contain `2,952,001` trainable
scalars in the matched optimizer inventory.

### Exact support execution

For K=6, each source and query support is encoded as a six-bit mask. The
precomputed 64-row Boolean table implements

\[
  \rho_{qs}>0\iff(m_s\mathbin{\&}b_q)\ne0.
\]

Only integer support metadata is detached. Live `A`, `alpha`, group moments,
overlap weights, and physical values remain differentiable. Queries sharing a
mask execute a unique source set; expensive `q -> k -> s` triples are never
expanded.

The selected and rectangular executors implement the same Run-1407 operator.
The calibrated policy selects the module path only when support rows are at
most half of the padded rectangular rows. The environmental path remains
rectangular because its support is broad and the existing QE kernel is faster.
Diagnostic/map flags do not choose the executor.

## Stage I: bounded four-case diagnosis

The diagnosis used the exact epoch-500 Run-1406 checkpoint and only cases
`0273`, `0653`, `0680`, and `0298` on physical GPU 1. It did not train a model
or alter Run 1406.

### Why the planned query rule was justified

| Case | Old mean query degree / 6 | Query zero fraction | Old key norm mean | Mean off-diagonal key cosine | Query mass on environment-empty groups |
|---|---:|---:|---:|---:|---:|
| 0273 | 6.000 | 0.000% | 1.762 | 0.233 | 0.334 |
| 0653 | 6.000 | 0.000% | 1.791 | 0.211 | 0.778 |
| 0680 | 5.708 | 4.875% | 1.888 | 0.257 | 0.873 |
| 0298 | 5.957 | 0.724% | 1.878 | 0.222 | 0.835 |

Run 1406's sources were already selective: module source degree ranged from
`1.4` to `2.0`, and environmental source degree from `1.333` to `1.526`.
However, module groups 0--3 were empty in all four cases, while the old query
access was nearly all-six. This supports the plan's diagnosis of a query-side
separation/alignment problem; it does not establish physical mechanisms or
causality.

### P0 reuse counterfactual

P0-to-P2 relative changes were small but nonzero: module membership
`0.60--2.36%`, environment membership `1.62--2.75%`, and group control
`1.35--3.72%`; source-membership support-flip fractions were at most `0.61%`.
A deliberately
new-equation counterfactual reused the mature Run-1406 P0 controller while
refreshing later fine values and K/V.

Across the four cases, its mean single-case forward changed from `253.49 ms`
to `196.01 ms`. The diagnostic preparation calls changed as follows:

| Preparation | Dynamic Run 1406 | Frozen-P0 counterfactual |
|---|---:|---:|
| P0 ports | 9.225 ms | 4.379 ms |
| P1 refinement | 8.910 ms | 1.830 ms |
| P2 field | 4.343 ms | 1.774 ms |

These are bounded diagnostic timings, not the matched endpoint benchmark.
Prediction changes were small and mixed: the largest absolute per-case delta
was `0.000458` for all-domain normalized relative L2, `0.001314` near the
interface, `0.001136` far-field, `0.001680` pressure, and `0.001315`
vorticity. This made phase sharing a plausible new model assumption, not an
exact Run-1406 cache transformation and not a prediction of Run-1407 training.

### Metric-attribution check

The existing evaluator's `global_field_all` is the normalized steady-field
grid over every grid cell and channel. It decomposes into module-region and
fluid-region field SSE. Internal temperature, surface temperature, and normal
heat flux are separate physical tensors and are not causal components of that
all-domain field metric. In the four diagnostic cases, the module/solid grid
dominated all-domain SSE, but that fact does not establish an internal-physics
advantage. CFD/physical-causality verification remains **Evidence Missing**.

## Implementation and exactness evidence

The opt-in implementation is localized to:

- `src/config_core/forward/phase_shared_group_control_honf_context.json`;
- `src/honf_forward_core/interface_fields/phase_shared_group_control.py`;
- `src/honf_forward_core/interface_fields/group_control_router.py`;
- `src/honf_forward_core/interface_fields/group_control_support.py`;
- the explicit ThermalChannel P0/P1/P2 state handoff in
  `Case_ThermalChannel/src/channelthermal/interface_field_coupling.py`.

No historical architecture was replaced. The existing Run-1406 and Dense-1804
checkpoints strict-loaded successfully through the maintained loader during
the matched GPU benchmark.

Focused tests cover:

- one controller construction and identity reuse, with refreshed later module
  affines and environment K/V;
- the displayed RMS/query-logit equation, including zero and near-zero rows;
- live controller gradients through later phases;
- all 64 masks, padding, empty groups, group permutations, and differentiable
  support weights;
- selected versus rectangular output and first-derivative parity;
- map/diagnostic independence and separate support/executed/recomputed ledgers;
- the physical P0-port, P1-refinement, P2-field, and consistency-reader handoff;
- historical config/checkpoint behavior.

Selected versus rectangular context uses `rtol=3e-5, atol=3e-6`; representative
query/router/module gradients use `rtol=5e-5, atol=5e-6`. The real-shape
calibration measured maximum absolute context error `1.1921e-7`; hybrid versus
rectangular was exactly zero in that broad-support fixture.

The integrated focused suite completed with `82 passed`; the final targeted
suite, including the anchor evaluator, completed with `85 passed`. The broader suite
completed with `744 passed, 5 skipped, 9 failed`; all nine failures were
environmental fixtures unavailable in the temporary worktree (four ignored
topology evaluator scripts and five missing WindFarm data links), not Run-1407
numerical or compatibility failures. Ruff and `git diff --check` passed.

## Real predicted-port GPU update

Before managed training, one disposable real B12/Q1024/M12 predicted-port
update ran on physical GPU 1 with cases `0228`--`0239`:

| Quantity | Result |
|---|---:|
| Total loss | 11.06117 |
| Field MSE | 2.28166 |
| Finite gradient tensors | 147/147 |
| Clipped gradient norm | 0.99999995 |
| Maximum group-code update | 0.00030005 |
| Parameters finite after update | yes |
| Step time | 358.04 ms |
| Peak allocated / reserved | 4.247 / 5.040 GiB |

The update was disposable; no managed run or checkpoint was created. The
controller code received a nonzero update, demonstrating that the shared live
P0 graph is trainable.

The preflight ran from the temporary integration worktree recorded in its
artifact. Its tracked config was verified byte-for-byte identical to the
merged `phase_shared_group_control_honf_context.json`; the managed training run
used the clean merged branch revision stated above.

On the same real shapes, the support calibration found `145,930 / 147,456`
module support rows (`98.97%`). Forced selected execution was slower
(`43.97 ms` median) than rectangular (`35.75 ms`), so the normal hybrid policy
correctly chose rectangular (`36.91 ms`). This calibration established the
conservative 50% switch point; it was not fitted to validation accuracy.

## Managed Run 1407 through epoch 50

Exactly one fresh run was launched, without a warm start or quickcheck run:

`/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1407_20260919_174751_phase_shared_prototype_group_control`

The initial manifest recorded `status=completed`, `last_completed_epoch=50`,
exit code 0, source revision
`cbd8f3cd52ab70c0b2a688c83d8c3ad4b6292860`, and completion at
`2026-09-19T21:55:24.894680+00:00`. `epoch_0050_model.pt` contains epoch 50,
the optimizer state, scaler state, and RNG state used by the later same-run
resume. After the explicit continuation request, the manifest returned to
`status=running`. The pre-existing Run 1406 process was not signalled,
restarted, modified, or used as a benchmark process.

### Learning trajectory

| Quantity | Run 1407 | Run 1406 | Dense 1804 |
|---|---:|---:|---:|
| Epoch-50 validation total loss | 2.02809 | 0.59892 | 0.42254 |
| Epoch-50 validation field MSE | 1.32424 | 0.30036 | 0.21934 |
| Epoch-50 validation temperature MSE | 0.32279 | 0.18650 | 0.12547 |
| Epochs 41--50 field-MSE median | 1.42353 | 0.40708 | 0.201999 |
| Last-ten field ratio vs Run 1406 | 3.497x | 1.000x | 0.496x |

Run 1407 remained finite and improved from epoch 41 to 50: validation field
MSE fell `11.6%` and total loss `15.5%`. The best-through-50 field MSE was
`1.32017`. Stability is therefore not the problem; the problem is a large
learning/fidelity gap relative to the matched parent. Run 1406 improved
more quickly over the same epoch window.

### Four-case epoch-50 fidelity

The same exact epoch-50 checkpoints were evaluated with predicted ports on the
four fixed cases. These are pooled four-case relative-L2 values, not a 90-case
population result:

| Metric | Run 1407 | Run 1406 | Dense 1804 | 1407 / 1406 |
|---|---:|---:|---:|---:|
| All-domain normalized field | 0.89687 | 0.57878 | 0.59196 | 1.550x |
| Fluid normalized field | 0.88778 | 0.42574 | 0.37382 | 2.085x |
| Near-interface field | 0.88391 | 0.39639 | 0.34187 | 2.230x |
| Far-fluid field | 0.86064 | 0.42853 | 0.40163 | 2.008x |
| Pressure | 1.04915 | 0.44700 | 0.41368 | 2.347x |
| Vorticity | 0.88895 | 0.52530 | 0.54597 | 1.692x |
| Internal temperature | 0.25002 | 0.21319 | 0.14960 | 1.173x |
| Surface temperature | 0.35438 | 0.28136 | 0.20039 | 1.260x |
| Normal heat flux | 0.40624 | 0.49023 | 0.49481 | 0.829x |

The field regression is broad and includes the plan's near-interface,
pressure, and vorticity safeguards. One component moves the other way: Run
1407's normal-heat-flux error is about `17%` lower than Run 1406 and `18%`
lower than Dense on these four cases. This isolated improvement is retained in
the assessment, but it does not offset the severe field/near-interface gap or
establish a population ranking. The outputs remain learned surrogate
predictions, not new CFD truth.

## Matched epoch-50 cost comparison

All models used physical GPU 1 in one session, one model resident at a time.
Inference used cases 0273/0653, Q=8192, receiver chunks of 2048, two warmups,
and five synchronized repetitions. Training used real predicted ports,
B48/Q1024, one warmup and three measured disposable updates with the same
fresh optimizer policy for all three models. No timed call requested maps or
a profiler.

### Inference

| Model | Case | Full forward (ms) | Preparation + one query (ms) | Prepared P2 (ms) | Full peak alloc/reserved (MiB) |
|---|---|---:|---:|---:|---:|
| Run 1407 | 0273 | 40.045 | 28.659 | 14.688 | 471.59 / 606 |
| Run 1407 | 0653 | 41.400 | 28.553 | 14.836 | 471.59 / 610 |
| Run 1406 | 0273 | 39.858 | 28.201 | 12.427 | 472.05 / 614 |
| Run 1406 | 0653 | 38.381 | 32.182 | 12.200 | 472.05 / 618 |
| Dense 1804 | 0273 | 33.941 | 23.150 | 13.114 | 460.01 / 618 |
| Dense 1804 | 0653 | 34.532 | 23.169 | 13.329 | 460.01 / 618 |

Two-case mean ratios:

| Run 1407 relative to | Full | Preparation + one query | Prepared P2 |
|---|---:|---:|---:|
| Run 1406 | 1.041x | 0.947x | 1.199x |
| Dense 1804 | 1.189x | 1.235x | 1.117x |

Phase sharing removes repeated controller construction and makes preparation
modestly cheaper than Run 1406. It does not make total preparation cheaper
than Dense because dynamic Dense MM/ME/EM, fine source refresh, query routing,
and support handling remain. P2 regresses because it receives no additional
controller-build saving after preparation, while it pays normalized query
routing and selected-support organization. Reduced module fine rows are too
small a fraction of the full P2 work to offset those costs.

### Training time and memory

| Model | M1 median / peak alloc (ms, MiB) | M12 median / peak alloc (ms, MiB) | M12 reserved (MiB) |
|---|---:|---:|---:|
| Run 1407 | 525.187 / 4172.14 | 1115.011 / 23628.65 | 24906 |
| Run 1406 | 525.043 / 4229.77 | 1126.960 / 23687.42 | 25930 |
| Dense 1804 | 1003.926 / 5847.91 | 2120.803 / 26777.41 | 28664 |

Run 1407 M12 is `0.989x` Run 1406 time and `0.998x` its allocated peak. It is
`0.526x` Dense time and `0.882x` Dense allocated peak. The accepted QE
checkpoint boundary remains the dominant reason that both group-control runs
retain their training-memory advantage over Dense. Phase sharing itself did
not deliver a meaningful additional M12 memory reduction; the shared P0 graph
remains live for all downstream losses, as required by the model equation.

## What routing and the executor actually did

At epoch 50, prototype anchoring changed the learned query access on the two
Q8192 anchors:

| Case | Mean/median query degree | Query-zero fraction | Module support / valid dense | Environment support / valid dense |
|---|---:|---:|---:|---:|
| 0273 | 2.575 / 3 | 57.08% | 67.54% | 93.95% |
| 0653 | 2.088 / 2 | 65.20% | 60.50% | 94.32% |

No query was singleton. Query mass assigned to module-empty groups averaged
`0.0116` and `0.00022`; environment-empty mass was effectively zero. The new
query rule therefore did what it was designed to do: it separated group
access and aligned it with occupied source groups.

That group sparsity did not translate into broad physical sparsity. In P2,
the normal executor evaluated exactly the module support (`16,598` and
`24,782` unique rows) but kept the rectangular environmental QE execution
(`1,572,864` rows in both cases) because the environmental support was
`1,477,680` and `1,483,598`. Run 1406 executed the full valid module and
environment rectangles. Support counts, actual executed rows, padded
denominators, and checkpoint recomputations remain separate in the artifact;
unavailable padded-row fields are not inferred.

This is a scientifically useful negative separation:

- prototype anchoring produced selective query-group routing;
- module support execution removed real rows, especially padded module rows;
- environmental source overlap remained nearly dense;
- the selected-module dispatch and routing overhead made P2 slower overall.

Learned routes are not physical causality, and row reduction alone is not an
accuracy or speed result.

## Epoch-500 continuation decision

The plan's bands are research guidance, not CI gates. Run 1407 meets most cost
guidance relative to Run 1406: full forward, M12 time, and M12 memory are
within roughly 10%, and preparation improves. Both scientific mechanisms are
verified to execute, and the trajectory is finite.

The learning band fails materially. The last-ten field median is `249.7%`
higher than Run 1406, far outside the approximate 25% review band. P2 is also
`19.9%` slower. The original budget judgment was therefore to stop at 50.
The user subsequently chose to test the alternative hypothesis that Run 1407
has different convergence characteristics, explicitly authorizing the same
run through epoch 500. Accordingly:

- the same Run 1407 resumed from its exact epoch-50 optimizer/RNG state;
- no 90-case endpoint, compact intervention study, or route-turnover study has
  been run; those remain deferred until epoch 500 and a later user command;
- no loss, occupancy constraint, temperature change, second seed, or alternate
  controller was introduced to accelerate or force a pass.

The negative result does not prove that all phase-shared or prototype routing
is unsuitable. This candidate combines two scientific changes, so the present
run cannot isolate which change caused the learning lag. The mature Run-1406
P0-reuse counterfactual showed small inference perturbations, while the trained
Run-1407 query rule showed strong sparsification; together these point to
training dynamics or the combined parameterization as the next scientific
question. They do not justify silently launching another architecture.

## Actual commands

Commands below are normalized to
`/home/wanglz/Desktop/src/ModularDT/HONF_Proj`. The accepted benchmark and
anchor invocations were executed from the repository root with equivalent
`HONF_Proj/` path prefixes. Physical GPU 1 was exposed as logical `cuda:0`.

### Stage-I diagnosis

```bash
rtk env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_ThermalChannel/src \
  conda run --no-capture-output -n ModularDT python -u \
  tools/diagnostics/diagnose_run1407_stage1.py \
  --checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1406_20260919_132837_low_dimensional_group_control_executor_optimized_rerun/epoch_0500_model.pt \
  --dataset /data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5 \
  --output-dir diagnostics/generated/run1407_stage1_diagnosis_20260919 \
  --device cuda:0 --query-batch-size 8192 \
  --case-id 0273 --case-id 0653 --case-id 0680 --case-id 0298
```

### One real predicted-port update

```bash
rtk env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_ThermalChannel/src \
  conda run --no-capture-output -n ModularDT python -u \
  tools/diagnostics/preflight_run1407_gpu.py \
  --config src/config_core/forward/phase_shared_group_control_honf_context.json \
  --dataset /data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5 \
  --output diagnostics/generated/run1407_preflight_20260919/preflight.json \
  --device cuda:0 --batch-size 12 --query-count 1024 --module-count 12
```

### Fresh managed training through epoch 50

```bash
rtk env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_ThermalChannel/src \
  conda run --no-capture-output -n ModularDT python -u train.py \
  --config src/config_core/forward/phase_shared_group_control_honf_context.json \
  --workflow forward --device cuda:0 --epochs 50 --yes
```

### User-authorized same-run continuation to epoch 500

```bash
rtk env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_ThermalChannel/src \
  conda run --no-capture-output -n ModularDT python -u train.py \
  --config src/config_core/forward/phase_shared_group_control_honf_context.json \
  --workflow forward --device cuda:0 --epochs 500 \
  --resume-checkpoint /home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1407_20260919_174751_phase_shared_prototype_group_control/epoch_0050_model.pt \
  --yes
```

### Matched cost benchmark

```bash
rtk env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_ThermalChannel/src \
  conda run --no-capture-output -n ModularDT python -u \
  tools/diagnostics/run_run1407_epoch50_benchmark.py \
  --checkpoint-1407 Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1407_20260919_174751_phase_shared_prototype_group_control/epoch_0050_model.pt \
  --checkpoint-1406 Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1406_20260919_132837_low_dimensional_group_control_executor_optimized_rerun/epoch_0050_model.pt \
  --checkpoint-1804 Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/epoch_0050_model.pt \
  --dataset /data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5 \
  --output diagnostics/generated/run1407_epoch50_benchmark_20260919/comparison.json \
  --report diagnostics/generated/run1407_epoch50_benchmark_20260919/benchmark_report.md \
  --map-dir diagnostics/generated/run1407_epoch50_benchmark_20260919/maps \
  --device cuda:0
```

The first benchmark invocation used the local-module HDF5 and exited before
model measurement because that file lacks the global `module_present` field.
The accepted command above uses the manifest's global ChannelThermal dataset.

### Four-case epoch-50 fidelity

```bash
rtk env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_ThermalChannel/src \
  conda run --no-capture-output -n ModularDT python -u \
  tools/diagnostics/evaluate_run1407_epoch50_anchors.py \
  --checkpoint-1407 Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1407_20260919_174751_phase_shared_prototype_group_control/epoch_0050_model.pt \
  --checkpoint-1406 Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1406_20260919_132837_low_dimensional_group_control_executor_optimized_rerun/epoch_0050_model.pt \
  --checkpoint-1804 Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/epoch_0050_model.pt \
  --dataset /data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5 \
  --case-id 0273 --case-id 0653 --case-id 0680 --case-id 0298 \
  --query-batch-size 8192 --local-port-condition-mode predicted --device cuda:0 \
  --output diagnostics/generated/run1407_epoch50_anchors_20260919/anchor_metrics.json \
  --report diagnostics/generated/run1407_epoch50_anchors_20260919/anchor_report.md
```

## Managed and generated artifacts

- Managed Run 1407:
  `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1407_20260919_174751_phase_shared_prototype_group_control`
- Exact epoch 50:
  `.../epoch_0050_model.pt`
- Resume-capable latest checkpoint:
  `.../checkpoints/latest.pt`
- Manifest/metrics:
  `.../run_manifest.json`, `.../metrics.csv`, `.../summary.json`
- Stage-I diagnosis:
  `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/diagnostics/generated/run1407_stage1_diagnosis_20260919/stage1_diagnosis.json`
- Real GPU update/calibration:
  `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/diagnostics/generated/run1407_preflight_20260919/preflight.json`
- Matched cost benchmark:
  `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/diagnostics/generated/run1407_epoch50_benchmark_20260919/comparison.json`
- Human-readable cost table:
  `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/diagnostics/generated/run1407_epoch50_benchmark_20260919/benchmark_report.md`
- Four-case epoch-50 fidelity:
  `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/diagnostics/generated/run1407_epoch50_anchors_20260919/anchor_metrics.json`

## Same-run continuation commands and status

`--epochs` is the terminal epoch. The following epoch-500 command was executed
after the user's explicit continuation instruction:

```bash
run=/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1407_20260919_174751_phase_shared_prototype_group_control

rtk env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_ThermalChannel/src \
  conda run --no-capture-output -n ModularDT python -u train.py \
  --config src/config_core/forward/phase_shared_group_control_honf_context.json \
  --workflow forward --device cuda:0 --epochs 500 \
  --resume-checkpoint "$run/epoch_0050_model.pt" --yes
```

The following requested long-run commands remain **unexecuted**. They are
contingent on the current epoch-500 continuation completing and creating
`epoch_0500_model.pt`:

```bash
run=/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1407_20260919_174751_phase_shared_prototype_group_control

rtk env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_ThermalChannel/src \
  conda run --no-capture-output -n ModularDT python -u train.py \
  --config src/config_core/forward/phase_shared_group_control_honf_context.json \
  --workflow forward --device cuda:0 --epochs 2500 \
  --resume-checkpoint "$run/epoch_0500_model.pt" --yes

rtk env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_ThermalChannel/src \
  conda run --no-capture-output -n ModularDT python -u train.py \
  --config src/config_core/forward/phase_shared_group_control_honf_context.json \
  --workflow forward --device cuda:0 --epochs 5000 \
  --resume-checkpoint "$run/epoch_2500_model.pt" --yes
```

## Scientific conclusion and next action

Run 1407 answers the intended executor/data-management questions cleanly:

1. Phase-static controller construction is feasible, exact within the new
   model, and modestly reduces preparation cost.
2. Prototype-plus-control RMS keys repair the nearly all-six query routing and
   avoid meaningful mass on source-empty groups.
3. Query-group sparsity is not sufficient for environmental fine-work
   sparsity: overlapping source memberships leave QE about 94% dense.
4. Small selected module supports do not dominate P2, and their organization
   can cost more than the work they remove.
5. Preserving a live shared P0 graph does not materially reduce M12 activation
   memory beyond Run 1406's accepted QE checkpoint boundary.
6. Most importantly, this combined model learns much more slowly through epoch
   50 than Run 1406 and Dense 1804.

The epoch-50 review bands did not independently justify the epoch-500 budget,
but the user explicitly authorized that longitudinal convergence test and it
is now running. This does not justify a corrective architecture automatically.
If later work is authorized, the single bottleneck to isolate remains
**whether phase-static P0 control itself impairs early optimization,
independently of prototype-anchored query keys**. That requires a deliberately
scoped scientific ablation, not an executor cache claim and not another loss
term. Until the epoch-500 evidence is available, Run 1406 remains the stronger
demonstrated scientific parent despite Run 1407's cleaner routing semantics.
