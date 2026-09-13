# WindFarm forward velocity: initial bounded study

## Execution status

Both authorized runs completed 500 epochs and 26,500 optimizer updates with
process exit code 0. Classic selected epoch 470 and dense epoch 495 using
validation volume MSE only. All planned prediction and representation evidence is complete. Controlled
timing on idle physical GPU 0 remains pending because of an unrelated job.

## Source and scope

The development branch is `agent/honf-windfarm-forward`. It was created after
fetching `origin/agent/honf-core-next`; the actual fetched parent was
`62deb25f04db4b788c50a3017a71c90e2c078562`. The worktree was clean when branching.
The planning reference happened to equal the latest parent; it was not imposed
as a branch freeze. Historical branches, checkpoints and unrelated GPU jobs
remain in place.

This is a field-only study of two architecture-inspired WindFarm velocity
baselines: classic fixed-K=6 HONF and the reusable dense pairwise field. It
does not reuse ThermalChannel weights, its local disk model, port semantics,
physical refinement or thermal losses. Neither run establishes meaningful
physical hyperedges, universal convergence, unseen-direction generalization,
exact turbine power, or solver-validated physical gradients.

## Data and represented domain

The external resources retain the logical IDs `wind_farm_tensor_v1` and
`wind_farm_volume_v1`. The machine-local location map resolves them through
`Case_WindFarm/Dataset/links/wind_farm`. Raw fields are read-only NumPy mmaps;
the sampled learning view does not copy the 59.6 GB volume or construct a
shared padded output cube. Only velocity `[Ux,Uy,Uz]` in m/s is supervised.

Each of 200 layouts contributes three direction rows, for 600 rows total.
The existing seed-42 layout-grouped 70/15/15 split supplies 140/30/30 layouts
and 420/90/90 rows. Directions of a layout remain together. The reserved test
partition is excluded from model selection in this experiment; previous data
exploration means it is not described as a completely uninspected dataset.

Each row uses its own stored turbine coordinates and native x/y/z axes.
The stored downstream/crosswind frame is retained without another wind-angle
rotation. Direction is categorical (270°, 285°, 300°), not an assumed
continuous meteorological convention. Native flat indexing is
`ix + nx * (iy + ny * iz)`, matching `(nz,ny,nx,3)` C-order velocity views.

The represented support box runs from the first to the last native cell
centre on each axis. Midpoint interior partition edges and endpoint centres
define factorized node weights. These weights integrate over that centre
box, not unknown finite-volume cells or exact CFD wall/inlet/outlet faces.
The z axis is nonuniform. The source supplies no full-volume solid mask;
actuator/source regions remain part of the supplied supervised field.

## Inputs, sampling and target transform

Coordinates are in rotor diameters, D=80 m. The fixed positional scale is
`[50,38,6.25]` D for both models. It is not a per-case remapping to a unit cube.
Module centres use row-specific xy and hub height 70 m (0.875 D), with known
features `[radius/D, hub_height/D] = [0.5,0.875]`. Active masks precede finite
zero padding.

The 11 global features are direction one-hot, M/30, U_ref/9, three normalized
lower bounds and three normalized support lengths. Environment and query
features are six normalized support-face distances plus normalized absolute
altitude. No solved field, wake-loss value, case/layout identifier, generator
quality flag or hidden per-case target statistic is a model input.

Each case supplies a geometry-only 16×8×4 grid of 512 cell-centred
environmental tokens. Every token has adapter-owned weight V_support/(D³·512).
Thus the same token count represents different physical densities on different
domains. Environmental input weights and supervised target quadrature have
separate roles.

Each epoch visits all 420 training rows once. The logical batch is eight
cases, with a final batch of four. Per case, 768 volume-weighted native queries
and 256 rotor-height-band queries form the explicit 0.75/0.25 training
objective. The band includes native centres in [30,110] m. Integer seed
composition makes per-row/per-epoch sampling independent of worker scheduling.

Target scaling uses 8192 deterministic volume samples per training row,
equal row weight, and stable streaming moments of U/9. Each channel uses a
global training mean and standard deviation, floored at 0.001 dimensionless.
A 32-bin altitude-conditioned training profile is fitted from those same
samples for evaluation only. Its residual includes the training population's
average wakes and is not a verified no-turbine inflow or physical power loss.

The actual derived view is
`Case_WindFarm/Dataset/derived/forward_velocity_v1/`. Its normalization was
fitted from 420×8192 = 3,440,640 training-only samples with seed 42:

| Dimensionless U/9 statistic | Ux | Uy | Uz |
|---|---:|---:|---:|
| Mean | 1.1586056144 | -0.0001585817 | 0.0001417685 |
| Standard deviation | 0.1673611024 | 0.0048576927 | 0.0040094127 |

The standard-deviation floor is inactive for every channel. These global
statistics define the output transform and never enter per-case model inputs.

The geometry-only split distribution below was computed from native axes,
layout/direction metadata, and compact `n_turbines` metadata; no target field
was read and no model was evaluated. Support values are min/median/max across
rows. The complete compact record is
`diagnostics/generated/forward_velocity_study/geometry_split_distribution.json`.

| Split | Layouts / rows | Direction rows (270 / 285 / 300) | Turbine-count row bins (`M=6–10 / 11–15 / 16–20 / 21–25 / 26–30`) | Support volume [D³] min / median / max | XY aspect min / median / max |
|---|---:|---:|---|---:|---:|
| Train | 140 / 420 | 140 / 140 / 140 | 81 / 105 / 87 / 78 / 69 | 2126.9 / 5987.9 / 10167.4 | 1.190 / 1.761 / 2.690 |
| Validation | 30 / 90 | 30 / 30 / 30 | 15 / 15 / 18 / 24 / 18 | 2550.0 / 6096.4 / 9417.1 | 1.238 / 1.737 / 2.397 |
| Test | 30 / 90 | 30 / 30 / 30 | 15 / 9 / 21 / 18 / 27 | 2771.4 / 5999.3 / 9523.5 | 1.172 / 1.763 / 2.392 |

The detailed validation cases were selected by input geometry before model
evaluation: minimum support volume, upper median in the ordered validation
population, and maximum support volume, with case-ID tie breaking.

| Row | Case | Turbines | Native (nx,ny,nz) | Support volume [D³] |
|---:|---|---:|---|---:|
| 506 | gen_0168_wd300 | 9 | (197,138,64) | 2550.0256 |
| 47 | gen_0015_wd300 | 10 | (301,216,64) | 6111.3982 |
| 426 | gen_0142_wd270 | 30 | (350,285,64) | 9417.1426 |

The largest case already exposes the maximum turbine count, so no fourth
native case is added. Equal E=512 means one environmental token represents
approximately 4.98, 11.94 and 18.39 D³ respectively; the representation audit
measures sensitivity to this changing physical density.

## Evidence recorded so far

The core extension is opt-in through `spatial_dim=3`, with explicit
nonperiodic scales and adapter-supplied environmental coordinates. Classic
3-D support is limited to the fixed organizer, dense full-support routing,
context fusion and legacy MLP pair kernel used here. Periodic 3-D geometry,
the other organizer families and 3-D rectangular boundary features are not
implemented by this change. Historical 2-D defaults, parameter widths,
serialization and absent-weight arithmetic remain available.

| Component | Classic fixed K6 | Dense pairwise |
|---|---|---|
| Shared hidden width | 256 | 256 |
| Interaction organization | Six learned fixed slots, softmax assignments | Dense environmental/module readers and eight coarse latents |
| Query interaction | Four-layer width-256 pair MLP, Fourier-4 relative geometry | Message width 128, four attention heads, one coarse block |
| Geometry extension | Signed xyz offsets, 3-D centroids/distances, width-12 query-to-edge geometry | Existing dimension-generic backend with explicit 3-D adapter encoding |
| Environmental measure | Weighted assignments, summaries and geometry; raw measure retained | Adapter weights replace the old fixed-domain fallback |
| Execution | Receiver chunk 128 | Receiver chunk 128, activation checkpointing |

The classic model keeps context fusion, raw edge state, hyper-value/global
context and Gaussian near-module context with scale 0.5 D. It uses neither
topology losses nor sparse support, direct residual output or a thermal
mechanism encoder. Both models have zero dropout and train all three velocity
channels from scratch. The WindFarm wrapper only dispatches prepare/decode
and applies the target inverse; learned interaction layers remain in the
reusable cores.

Focused core/configuration/geometry tests passed (74 tests), including real
forward/backward computation, z sensitivity, supplied quadrature and
split-weight duplication. Separate thermal/interface compatibility tests
passed (36, with one expected CUDA smoke skip), and runtime/configuration
tests passed (31). Bounded native-data wrapper checks additionally exercised
both H256 models, slice/profile/routing rendering, diagnostics, timing and a
trusted H16 checkpoint round-trip after an actual AdamW update. A full native
identity reconstruction on the smallest case visited 1,451,904 cells and
recovered zero RMSE with support measure 2126.916667 D³. Test counts here refer
to separate invocations, with overlapping coverage; they are not summed.

Accepted Run-1000 (epoch 9655) and Run-1401 (epoch 4585) checkpoints were also
replayed on test case 0653 against a clean parent checkout, under matching CPU
conditions and query chunk 8192. Current/parent organizer arrays, field and
local predictions, routing arrays/summaries and query coordinates were bitwise
identical after retaining the original 2-D scalar-division arithmetic. The
older stored golden fixture differs on both the clean parent and current
branch; it is not claimed to pass. No fixture or checkpoint was updated.

The final WindFarm suite passed 25 tests. A real interrupted/resumed two-update
training computation reproduced uninterrupted model state exactly (maximum
absolute parameter difference zero), with final validation volume MSE
0.7145472633 in both executions. Its disposable checkpoints are under
`Case_WindFarm/diagnostics/generated/forward_velocity_study/workflow_numerics/`.
Real loader aggregation with B8 followed by B4 matched independently computed
equal-case loss 1.0209889015 and volume MSE 0.7891934360 exactly. These bounded
workflow tests are separate from the managed 500-epoch studies.

Two spatial-reduction tests passed against actual mounted data. Native x/y/z
planes on the smallest, median and largest cell-count runs exactly equal the
existing reader's structured velocity slices. Coordinate index arithmetic and
factorized support-volume sums also passed. These computations establish
indexing and reduction behavior; they are not model-training evidence.

An independent full-width integration computation on physical GPU 1 used
training row 0 (`gen_0000_wd270`), B=1, Q=32, E=512 and a fresh model of each
family. Each performed a real backward pass and AdamW update:

| Model | Standardized MSE before update | Preclip gradient norm | Sampled parameter update norm | Materialized parameters |
|---|---:|---:|---:|---:|
| Classic K6 | 0.36869115 | 1.75969493 | 0.01437097 | 1,696,286 |
| Dense | 0.36378488 | 0.24434407 | 0.01408493 | 3,628,551 |

Removing targets and host metadata left predictions unchanged. Tensor
conversion to m/s matched the independent NumPy target inverse. The prepared
wrapper state held no original batch. Both disposable models were discarded.
These are small integration computations, not the four-case learning probes
or generalization measurements. The actual record is
`Case_WindFarm/diagnostics/generated/forward_velocity_study/quick_execution/main_integration.json`.

At initial scheduling inspection, physical GPUs 0 and 2 held unrelated
ThermalChannel runs 1807/1808. GPU 1 was available for authorized preparation
and disposable tests. Those unrelated processes were not interrupted.

The bounded four-row probes used training rows 264, 376, 72 and 531, covering
M=6/7/23/30 and the training support-volume extrema. Both completed exactly
60 fixed-sample updates with Q=512 and E=512, then discarded their models.

| Model | First training loss | Step-60 training loss | Fresh B8/M30/Q1024 peak allocated MiB | Fresh-batch gradient norm |
|---|---:|---:|---:|---:|
| Classic K6 | 1.2068385 | 0.7326568 | 2251.18 | 3.39127 |
| Dense | 1.2154399 | 0.5324789 | 2627.75 | 0.98283 |

Both fresh full batches produced finite, nonzero parameter updates. The record
is `Case_WindFarm/diagnostics/generated/forward_velocity_study/quick_execution/quick_execution.json`.
These losses measure learning on repeated training targets, not validation
performance. No further fixed-sample learning probe is authorized by this study.

Supplementary compact/native hub-plane comparisons used rows 264, 202 and
376. The nearest native height is 70.869949 m, rather than exactly 70 m.
At 22,134 / 62,700 / 105,850 matching valid compact locations inside native
support, Ux RMSE was 0.05946 / 0.05316 / 0.04968 m/s and Uy RMSE was
0.01391 / 0.01280 / 0.01305 m/s. Mean absolute nearest-centre xy offsets
were approximately 2.75 m and 2.25 m. These results support the documented
coordinate-frame correspondence while retaining the different plane/raster
definitions. They do not make compact rasters interchangeable with native
targets. Full details are in the derived `sampling_metadata.json`.

## Training, endpoint comparison and limitations

The ordinary root allocator created these two runs from integrated source
commit `415ad6a`:

| Model | ID | Physical GPU | Actual managed directory |
|---|---:|---:|---|
| Classic K6 | 2100 | 0 | `Trained_Results/WindFarm/HONF_Forward_Runs/Run_2100_20260913_014741_windfarm_classic_k6_velocity` |
| Dense | 2101 | 2 | `Trained_Results/WindFarm/HONF_Forward_Runs/Run_2101_20260913_014752_windfarm_dense_velocity` |

Each process sees only its assigned physical GPU and uses logical `cuda:0`.
Logical batch eight runs directly, without accumulation or a resource-driven
model variant. Both use fresh seed-0 initialization, AdamW with learning rate
3e-4 and weight decay 1e-5, gradient clipping at norm 1, no AMP, no dropout
and no learning-rate schedule. Sampling uses seed 42; the per-epoch row order
and native query draws are shared across architectures. Validation runs every
five epochs with fixed 8192 volume and 2048 band queries per case. Selection
uses only the equal-case standardized volume MSE, never the test partition. Both visit 420 rows in 53 optimizer updates per epoch. Startup
and progress logs are the ignored study-root `training_classic.log` and
`training_dense.log`; metrics and checkpoints are run-owned. The literal
`val_volume=nan` printed on non-validation epochs means validation was not
scheduled, not a nonfinite training result; CSV validation cells are blank.

Executed from `HONF_Proj` (stdout/stderr redirected to the logs above):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_WindFarm/src \
/home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py \
  --config project://src/config_core/forward/windfarm_classic_k6.json \
  --workflow forward --device cuda:0 --epochs 500 \
  --run-id 2100 --run-name windfarm_classic_k6_velocity --yes

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:Case_WindFarm/src \
/home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py \
  --config project://src/config_core/forward/windfarm_dense_pairwise.json \
  --workflow forward --device cuda:0 --epochs 500 \
  --run-id 2101 --run-name windfarm_dense_velocity --yes
```

Classic completed with process exit code 0 and exactly 500 history rows. Its
validation-selected checkpoint is epoch 470 (scheduled volume MSE 0.0090389543),
versus 0.0097194188 at epoch 500. These selection-time estimates use the frozen
periodic validation queries; larger endpoint estimates are recorded separately.

Dense likewise completed 500 history rows and 26,500 updates. Its selected
epoch 495 achieved scheduled validation volume MSE 0.0039383326, versus
0.0045076504 at epoch 500. Neither run was resumed or extended.

Both models completed all five endpoint stages with exit code 0. Each selected
checkpoint received exactly one reserved-test evaluation. Controlled timing
on idle GPU 0 remains unavailable.

The 500-epoch endpoint is an initial resource-bounded assessment. Any future
same-run continuation command will be reported separately and left unexecuted.

## Controlled cost limitation

At 2026-09-13 07:09 UTC, physical GPU 0 remained occupied by unrelated
ThermalChannel Run 1807, PID 3915145, using 34,686 MiB. That process had reached
epoch 2461 of 5000; its recent five-epoch mean was about 24 seconds per epoch,
suggesting roughly 17 hours still required. It was not interrupted. This
prevents the specified same-idle-physical-GPU-0 comparison during this closeout.
No controlled latency ratio is claimed. The real disposable memory/update
evidence above remains valid for its stated execution conditions, while
controlled timing commands are left unexecuted below.

## Completed prediction evidence

Every endpoint uses 90 rows / 30 layouts, 32,768 volume queries and 8,192 band
queries per row, sample seed 42. Identical row IDs, query budgets, seed, and
checkpoint roles were checked by the comparison script. Validation selects
the checkpoint; these larger-sample validation estimates do not reselect it.
Only each selected best checkpoint is evaluated on the reserved test.

| Split / checkpoint | Model | Epoch | Equal-case volume std MSE | Band std MSE | Equal-case volume RMSE [m/s] | Pooled vector relative L2 |
|---|---|---:|---:|---:|---:|---:|
| Validation selected | classic | 470 | 0.00902304 | 0.0255406 | 0.0259393 | 0.00424642 |
| Validation selected | dense | 495 | 0.00408517 | 0.0132671 | 0.0176181 | 0.00290796 |
| Validation exact 500 | classic | 500 | 0.00977096 | 0.0269999 | 0.0361311 | 0.00604297 |
| Validation exact 500 | dense | 500 | 0.00447622 | 0.0137871 | 0.0187983 | 0.00311096 |
| Reserved test selected | classic | 470 | 0.00840998 | 0.0240451 | 0.0257174 | 0.0042672 |
| Reserved test selected | dense | 495 | 0.00418311 | 0.0134929 | 0.0175601 | 0.0029456 |

Dense reduces the selected-checkpoint validation volume standardized MSE by 54.7% and reserved-test MSE by 50.3% relative to classic in this one-seed 500-epoch study.
This is an empirical comparison of two WindFarm velocity baselines, not a
general conclusion about dense architectures or physical hypergraphs.

Scalar RMSE here averages squared error across the three velocity components.
The following component values are physically volume-pooled, so larger support
domains receive more weight. MAE uses the corresponding absolute-error integral.

| Split | Model | Region | RMSE Ux / Uy / Uz [m/s] | MAE Ux / Uy / Uz [m/s] | RMSE / U_ref |
|---|---|---|---|---|---|
| Validation | classic | volume | 0.0442987 / 0.005075 / 0.00382641 | 0.0274707 / 0.00262258 / 0.00169848 | 0.00492207 / 0.000563889 / 0.000425157 |
| Validation | classic | hub_band | 0.0645176 / 0.00900004 / 0.00613995 | 0.0411455 / 0.00467684 / 0.00222392 | 0.00716862 / 0.001 / 0.000682217 |
| Validation | classic | downstream_envelope | 0.0854751 / 0.0112729 / 0.0084523 | 0.0594047 / 0.00692715 / 0.00420397 | 0.00949724 / 0.00125255 / 0.000939144 |
| Validation | dense | volume | 0.0303352 / 0.00346642 / 0.00263956 | 0.0187708 / 0.00177033 / 0.000975066 | 0.00337058 / 0.000385157 / 0.000293284 |
| Validation | dense | hub_band | 0.0495768 / 0.00638758 / 0.00465969 | 0.0343797 / 0.00324096 / 0.00157246 | 0.00550854 / 0.000709731 / 0.000517744 |
| Validation | dense | downstream_envelope | 0.0625184 / 0.00782267 / 0.00634738 | 0.0437223 / 0.00472085 / 0.00293455 | 0.00694649 / 0.000869185 / 0.000705264 |
| Reserved test | classic | volume | 0.0445325 / 0.00500068 / 0.00386142 | 0.0280692 / 0.00262349 / 0.00172903 | 0.00494806 / 0.000555631 / 0.000429047 |
| Reserved test | classic | hub_band | 0.0638947 / 0.00898745 / 0.00613247 | 0.0411897 / 0.00478026 / 0.00226478 | 0.00709941 / 0.000998605 / 0.000681386 |
| Reserved test | classic | downstream_envelope | 0.0845252 / 0.0111729 / 0.00858808 | 0.0592347 / 0.00689615 / 0.00422119 | 0.00939169 / 0.00124143 / 0.000954232 |
| Reserved test | dense | volume | 0.0307208 / 0.00354781 / 0.00276309 | 0.0190034 / 0.00178827 / 0.00101031 | 0.00341343 / 0.000394202 / 0.00030701 |
| Reserved test | dense | hub_band | 0.0502938 / 0.00652714 / 0.00482632 | 0.0348141 / 0.00330781 / 0.00161921 | 0.0055882 / 0.000725238 / 0.000536258 |
| Reserved test | dense | downstream_envelope | 0.0631871 / 0.008029 / 0.00658127 | 0.0444238 / 0.00481697 / 0.00301714 | 0.00702079 / 0.000892111 / 0.000731252 |

Per-case vector relative L2 is `sqrt(sum_c E_bc / sum_c T_bc)` with weighted
squared-error and target-energy integrals. The pooled metric sums those
integrals across cases before taking the ratio. It is not an average of case
relative errors. Component target energies and every denominator are retained
in each metrics JSON; small Uy/Uz energy is not hidden by a chosen epsilon.

| Split | Model | Case RMSE mean / median / p95 / worst [m/s] | Layout mean of direction-mean RMSE [m/s] | Baseline volume / band RMSE [m/s] |
|---|---|---|---:|---|
| Validation | classic | 0.0259393 / 0.0245223 / 0.0344958 / 0.0404539 | 0.0259393 | 0.20226 / 0.387572 |
| Validation | dense | 0.0176181 / 0.017565 / 0.0219095 / 0.023454 | 0.0176181 | 0.20226 / 0.387572 |
| Reserved test | classic | 0.0257174 / 0.0256799 / 0.031041 / 0.0357929 | 0.0257174 | 0.201729 / 0.387309 |
| Reserved test | dense | 0.0175601 / 0.0178781 / 0.0207993 / 0.0234907 | 0.0175601 | 0.201729 / 0.387309 |

Each split has 30 independent layout groups; the three directions are averaged
within layout before the layout-level mean. The baseline is the same training-only
altitude profile for both models, including the training population’s average wakes.

The sampled downstream envelope contains enough points in every evaluated case
(at least 32); exact per-case/pooled counts are recorded. It is an input-defined
diagnostic envelope, not a CFD wake boundary or a training mask.

## Geometry strata

Entries below are equal-case volume RMSE in m/s. Counts are cases/layouts.
Bin boundaries for volume, aspect, nearest-neighbor mean/minimum and dispersion
come solely from training geometry and are saved in `strata_reference`.
Strata overlap; their counts must not be summed as independent evidence.

| Stratum | Validation cases/layouts | Classic / dense validation RMSE | Test cases/layouts | Classic / dense test RMSE |
|---|---:|---|---:|---|
| M_bin=6-10 | 15/5 | 0.025822 / 0.0158468 | 15/5 | 0.0232635 / 0.0143539 |
| M_bin=11-15 | 15/5 | 0.0213135 / 0.0150154 | 9/3 | 0.0242598 / 0.015711 |
| M_bin=16-20 | 18/6 | 0.0246939 / 0.0171898 | 21/7 | 0.0242796 / 0.0169154 |
| M_bin=21-25 | 24/8 | 0.0283701 / 0.0191886 | 18/6 | 0.0260017 / 0.0185351 |
| M_bin=26-30 | 18/6 | 0.0278963 / 0.0195976 | 27/9 | 0.0284952 / 0.0198091 |
| direction=270 | 30/30 | 0.0277761 / 0.0176041 | 30/30 | 0.0274548 / 0.017443 |
| direction=285 | 30/30 | 0.0248449 / 0.0174275 | 30/30 | 0.0251544 / 0.0173143 |
| direction=300 | 30/30 | 0.0251971 / 0.0178228 | 30/30 | 0.0245429 / 0.017923 |
| volume_quartile=1 | 19/7 | 0.0298508 / 0.0185678 | 19/8 | 0.0252494 / 0.0160013 |
| volume_quartile=2 | 23/11 | 0.02584 / 0.0175502 | 26/16 | 0.0260719 / 0.0177547 |
| volume_quartile=3 | 25/13 | 0.0241249 / 0.0169667 | 23/13 | 0.0254509 / 0.0174294 |
| volume_quartile=4 | 23/10 | 0.0247797 / 0.0176097 | 22/8 | 0.0259811 / 0.0188129 |
| aspect_xy_bin=1 | 29/13 | 0.0244299 / 0.0172588 | 25/15 | 0.0246029 / 0.0174383 |
| aspect_xy_bin=2 | 18/14 | 0.0260741 / 0.0177194 | 20/16 | 0.0252364 / 0.0176901 |
| aspect_xy_bin=3 | 24/14 | 0.0270873 / 0.0179672 | 25/17 | 0.0256623 / 0.017166 |
| aspect_xy_bin=4 | 19/11 | 0.0266656 / 0.0176298 | 20/14 | 0.0276602 / 0.0180749 |
| mean_nn_D_bin=1 | 27/9 | 0.0311055 / 0.0203481 | 24/8 | 0.0288289 / 0.0194232 |
| mean_nn_D_bin=2 | 18/6 | 0.0257117 / 0.0178031 | 27/9 | 0.0258183 / 0.0175805 |
| mean_nn_D_bin=3 | 27/9 | 0.0236835 / 0.0166175 | 24/8 | 0.0245664 / 0.0176887 |
| mean_nn_D_bin=4 | 18/6 | 0.0218014 / 0.0148393 | 15/5 | 0.0223988 / 0.0143366 |
| min_sep_D_bin=1 | 27/9 | 0.0311055 / 0.0203481 | 24/8 | 0.0288289 / 0.0194232 |
| min_sep_D_bin=2 | 18/6 | 0.0257117 / 0.0178031 | 24/8 | 0.026148 / 0.0178123 |
| min_sep_D_bin=3 | 27/9 | 0.0236835 / 0.0166175 | 30/10 | 0.0243422 / 0.0172737 |
| min_sep_D_bin=4 | 18/6 | 0.0218014 / 0.0148393 | 12/4 | 0.022071 / 0.0140455 |
| nn_dispersion_bin=1 | 24/8 | 0.0288131 / 0.0186339 | 36/12 | 0.0254273 / 0.0172124 |
| nn_dispersion_bin=2 | 27/9 | 0.0252624 / 0.0172959 | 6/2 | 0.0253213 / 0.0179865 |
| nn_dispersion_bin=3 | 15/5 | 0.0249049 / 0.0170405 | 30/10 | 0.0266017 / 0.0182426 |
| nn_dispersion_bin=4 | 24/8 | 0.0244737 / 0.0173259 | 18/6 | 0.0249557 / 0.0169758 |

The split was not designed for unseen direction or domain-size extrapolation.
Nearest-neighbor descriptors are recomputed from each row’s turbine centers;
equivalence to the compact archive’s undocumented descriptor formulas is not claimed.

## Native volumes and spatial evidence

Each selected model prepared each of rows 506/47/426 once and streamed all
native cells in bounded coordinate chunks. Targets remained mmaps; no full
prediction volume was retained. Measures below use center-support quadrature
in D³; sampled endpoint integrals use the equivalent measure in m³.

| Model | Row | Native cells | Volume RMSE Ux / Uy / Uz [m/s] | Volume std MSE | Band / downstream vector L2 | Baseline volume std MSE | Ux profile-residual relative L2 |
|---|---:|---:|---|---:|---|---:|---:|
| classic | 506 | 1,739,904 | 0.0661397 / 0.00815206 / 0.0071058 | 0.0251581 | 0.0114337 / 0.01532 | 1.10293 | 0.182092 |
| classic | 47 | 4,161,024 | 0.0356183 / 0.00314863 / 0.00245123 | 0.00345348 | 0.00552025 / 0.00741554 | 0.409644 | 0.134882 |
| classic | 426 | 6,384,000 | 0.0505975 / 0.0049836 / 0.0037691 | 0.00834415 | 0.00842487 / 0.0102095 | 0.705751 | 0.149105 |
| dense | 506 | 1,739,904 | 0.0379912 / 0.00435074 / 0.00381841 | 0.00724565 | 0.00791088 / 0.0106996 | 1.10293 | 0.104595 |
| dense | 47 | 4,161,024 | 0.0228767 / 0.00210265 / 0.00170327 | 0.00159059 | 0.00407305 / 0.00526001 | 0.409644 | 0.086631 |
| dense | 426 | 6,384,000 | 0.0341604 / 0.00393743 / 0.00256719 | 0.00456227 | 0.00611453 / 0.0076754 | 0.705751 | 0.100667 |

Ux residual compares `(prediction_x - training_profile_x)` with
`(target_x - training_profile_x)` using the target residual energy as denominator.
It is not an inferred no-turbine inflow deficit or power-loss metric.

Native z=70 m, y=0 m and x=0 m requests select the nearest actual centers;
each figure states its actual coordinate, M, direction, and physical axes.
Reference/prediction share a color scale and preserve physical aspect ratio.
All three components and signed errors are shown. Profiles cover the full
64-point native z axis at the native xy nearest zero, with the training profile
shown separately. Ux structure and vertical dependence are learned, but
local wake errors and visible Uy/Uz profile discrepancies remain.

Classic routing figures show six latent assignments and source/region centroids
in 3-D. Dense figures show environmental attention and context norms. Only
512 routing receivers are retained per case. Neither plot identifies physical
edge labels; a context norm is not a calibrated physical contribution.

- [classic: native_z_slice.png](/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_WindFarm/diagnostics/generated/forward_velocity_study/native_case_checks/classic/gen_0142_wd270/native_z_slice.png)
- [classic: native_y_slice.png](/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_WindFarm/diagnostics/generated/forward_velocity_study/native_case_checks/classic/gen_0142_wd270/native_y_slice.png)
- [classic: native_x_slice.png](/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_WindFarm/diagnostics/generated/forward_velocity_study/native_case_checks/classic/gen_0142_wd270/native_x_slice.png)
- [classic: vertical_profile.png](/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_WindFarm/diagnostics/generated/forward_velocity_study/native_case_checks/classic/gen_0142_wd270/vertical_profile.png)
- [classic: routing.png](/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_WindFarm/diagnostics/generated/forward_velocity_study/native_case_checks/classic/gen_0142_wd270/routing.png)
- [dense: native_z_slice.png](/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_WindFarm/diagnostics/generated/forward_velocity_study/native_case_checks/dense/gen_0142_wd270/native_z_slice.png)
- [dense: native_y_slice.png](/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_WindFarm/diagnostics/generated/forward_velocity_study/native_case_checks/dense/gen_0142_wd270/native_y_slice.png)
- [dense: native_x_slice.png](/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_WindFarm/diagnostics/generated/forward_velocity_study/native_case_checks/dense/gen_0142_wd270/native_x_slice.png)
- [dense: vertical_profile.png](/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_WindFarm/diagnostics/generated/forward_velocity_study/native_case_checks/dense/gen_0142_wd270/vertical_profile.png)
- [dense: routing.png](/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_WindFarm/diagnostics/generated/forward_velocity_study/native_case_checks/dense/gen_0142_wd270/routing.png)

## Representation and sampling diagnostics

Discrepancies below are maximum relative L2 over the three geometry-selected
cases, measured in standardized prediction space. Absolute maximum and RMS
differences are also retained in each diagnostic JSON. Each probe uses 512
fixed volume queries; inference requires only geometry and operating metadata.

| Perturbation | Classic max relative L2 | Dense max relative L2 |
|---|---:|---:|
| same_chunk_repeat | 0 | 0 |
| query_permutation | 0 | 0 |
| chunk_1024 | 2.91148e-07 | 4.04641e-07 |
| module_permutation | 1.41562e-07 | 1.20901e-07 |
| additional_padding | 2.56207e-07 | 2.92604e-07 |
| different_domain_batch | 2.99883e-07 | 4.25642e-07 |
| split_weight_duplication | 1.3897e-07 | 1.82942e-07 |
| environment_1024 | 0.000772338 | 0.00355142 |
| zeroed_support_features | 0.479606 | 1.19204 |
| muted_route | 1.57265 | 0.812406 |

Changing receiver chunk 128→1024 leaves pointwise predictions consistent with
floating-point variation, while same-chunk repeats and query permutations
are exact on these probes. Padding, module order, case batching and exact
environment duplication with split mass are similarly stable. Refining
16×8×4→32×8×4 changes the input representation without retraining: this
measures resolution sensitivity, not continuum convergence or new field truth.

Support corruption zeroes the seven query/environment support features and
the six global support bounds/extents while retaining true coordinates.
The classic mute removes query-module pair context but leaves hyperedge/near
routes. The dense mute removes direct query-module output but leaves
environment/coarse/local routes. These are partial fitted-model reliance
probes, not moved-turbine or changed-domain physical counterfactual accuracy.

| Model | Reference std MSE, rows 506 / 47 / 426 | Muted-route MSE | Corrupted-support MSE |
|---|---|---|---|
| classic | 0.0164647 / 0.00235434 / 0.00799928 | 1.47051 / 1.0812 / 2.85906 | 0.0731544 / 0.08146 / 0.285733 |
| dense | 0.00491834 / 0.000888947 / 0.00312798 | 0.682481 / 0.276956 / 0.489769 | 1.53397 / 0.789486 / 0.53582 |

One independent sample (seed 314159, same 32768/8192 budget) checks finite-sample
sensitivity on the same three validation cases. It is not a confidence interval.

| Model | Mean volume vector L2, seed 42 / 314159 | Mean band vector L2, seed 42 / 314159 |
|---|---|---|
| classic | 0.00495479 / 0.00485507 | 0.00845277 / 0.00862738 |
| dense | 0.00307038 / 0.00306763 | 0.00603852 / 0.00603704 |

## Learning history and bounded interpretation

| Epoch | Classic train volume / band | Classic validation volume / band | Dense train volume / band | Dense validation volume / band |
|---:|---|---|---|---|
| 10 | 0.418438 / 1.57344 | 0.438805 / 1.61915 | 0.238802 / 0.772714 | 0.244896 / 0.763941 |
| 50 | 0.0560982 / 0.159364 | 0.0672382 / 0.186711 | 0.0541369 / 0.181481 | 0.0565855 / 0.181225 |
| 100 | 0.0255597 / 0.0697582 | 0.0362482 / 0.108977 | 0.0218394 / 0.0648861 | 0.0239643 / 0.0720447 |
| 250 | 0.0136293 / 0.0379344 | 0.0156523 / 0.0449239 | 0.00822996 / 0.0231732 | 0.00852635 / 0.0231862 |
| 500 | 0.00916054 / 0.0271393 | 0.00971942 / 0.0268577 | 0.00374099 / 0.0134626 | 0.00450765 / 0.0136441 |

Mean scheduled validation volume MSE over epochs 301–400 versus 401–500
decreased from 0.0127899 to 0.0108890 for classic and from 0.00619785 to
0.00488240 for dense. These descriptive windows support an unfinished learning
trajectory; they are not a heuristic convergence gate. Dense is the stronger
velocity baseline at this budget. It has 3,628,551 trainable parameters versus
1,696,286 for classic. Maximum run-recorded allocated memory was 2653.095 MiB
versus 2264.434 MiB. Training wall times came from different contended GPUs
and do not establish controlled speed or efficiency superiority.

![Training and validation learning curves](/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_WindFarm/diagnostics/generated/forward_velocity_study/paired_comparison/learning_curves.png)

## Artifacts, commands and departures

The canonical ignored evidence root is `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_WindFarm/diagnostics/generated/forward_velocity_study`.
Managed checkpoints remain in their original run-owned directories above.
The paired JSON/CSV, per-case CSVs, denominator integrals, overlapping strata,
native reports and figures remain available without copying raw fields.

- `paired_comparison/{paired_comparison.json,paired_comparison.csv,learning_curves.png}`
- `validation/{classic_best,classic_epoch500,dense_best,dense_epoch500}/metrics.json`
- `test/{classic_best,dense_best}/metrics.json`
- `native_case_checks/{classic,dense}/native_evidence.json`
- `validation/{classic_diagnostics,dense_diagnostics}/diagnostics_evidence.json`
- `final_endpoint_logs/`: actual invocations and successful stage logs.

Classic endpoint output paths initially omitted the case directory; those
generated artifacts were moved into the case-owned ignored store and their
figure references updated, without changing numerical results. Dense had one
early missing-PYTHONPATH import failure and one rejected working-directory typo;
both occurred before model evaluation. The corrected commands succeeded. No
numerical failure, OOM, rescue architecture, extra successful test evaluation
or managed third run occurred. The historical golden-fixture mismatch and
unavailable controlled GPU-0 timing are documented separately above.

Source additions cover geometry/data/normalization, the thin model/plugin,
field-only train/evaluate workflows, study reduction/rendering/diagnostic tools,
preparation/summary scripts, two root profiles, the case profile/schema, and
focused real-computation tests. Shared changes are restricted to dimension
configuration/geometry, fixed-K organization/decoding, optional environmental
weights, dense input preparation and case registration. Existing preprocessing
and ThermalChannel physics/security remain intact.

The executable training source was committed as `415ad6a`; subsequent evaluation
metadata/comparison tooling is in `e49a029`, and rendering/device provenance in
`6c602e5`. The latter changes do not alter trained model outputs. The final
figure-wrapper computation passed again after the rendering repair (two tests,
4.72 seconds). Checkpoint-owned continuation profiles were parsed through the
root loader and accepted by the existing managed-resume identity/configuration
comparison without launching a process or changing run state.

The following commands are executable from `/home/wanglz/Desktop/src/ModularDT/HONF_Proj`.
All paths below refer to the actual runs and the existing ignored store:

```bash
wf_python=/home/wanglz/miniconda3/envs/ModularDT/bin/python
wf_study=/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_WindFarm/diagnostics/generated/forward_velocity_study
wf_classic=/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/WindFarm/HONF_Forward_Runs/Run_2100_20260913_014741_windfarm_classic_k6_velocity
wf_dense=/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/WindFarm/HONF_Forward_Runs/Run_2101_20260913_014752_windfarm_dense_velocity
```

Executed endpoint pattern (the individual actual commands and original output
locations are preserved in `final_endpoint_logs`; do not rerun the reserved test
as part of further architecture selection):

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_WindFarm/src "$wf_python" -u evaluate.py \
  --config project://src/config_core/forward/windfarm_classic_k6.json \
  --workflow forward --checkpoint "$wf_classic/best_model.pt" --device cuda:0 \
  --split validation --volume-queries 32768 --band-queries 8192 \
  --receiver-chunk-size 1024 --output-dir "$wf_study/validation/classic_best"
```

The dense counterpart uses its dense profile and own best checkpoint. Exact-500
validation uses each `epoch_0500_model.pt`; the single reserved-test commands use
`--split test` and selected best only. Native and diagnostics use the same root
launcher with `--study-mode native` or `--study-mode diagnostics`, writing to the
listed model-specific directories. The summary command uses the six metrics JSONs
with labels `classic|dense:best_selection_val|exact500_val|reserved_test_best`,
both run histories, and `--require-all-endpoints`. It only reads saved metrics;
it does not perform another inference or reserved-test evaluation.

**Unexecuted controlled timing**, sequentially on physical GPU 0 once idle:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_WindFarm/src "$wf_python" -u evaluate.py \
  --config project://src/config_core/forward/windfarm_classic_k6.json \
  --workflow forward --checkpoint "$wf_classic/best_model.pt" --device cuda:0 \
  --study-mode timing --output-dir "$wf_study/timing/classic"

rtk proxy env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_WindFarm/src "$wf_python" -u evaluate.py \
  --config project://src/config_core/forward/windfarm_dense_pairwise.json \
  --workflow forward --checkpoint "$wf_dense/best_model.pt" --device cuda:0 \
  --study-mode timing --output-dir "$wf_study/timing/dense"
```

This stage separately measures CPU sampling, transfer, prepare, Q8192/Q65536
prepared reads, small/large native inference, peak allocated/reserved memory,
parameter count and one disposable B8/Q1024 backward/update. Short operations
use three warmups and ten repeats; large reads use one warmup/three repeats.
Full-native inference uses one explicitly labeled pass. CUDA synchronization
surrounds GPU timings. The disposable update changes only the loaded in-memory
model, which is discarded; no managed checkpoint or optimizer is rewritten.
The user was asked whether idle GPU 1 may substitute for the explicitly required
GPU 0. Without that approval, timing remains unexecuted and no speed ratio is
claimed.

## Recommendation and unexecuted continuation

Use dense as the stronger velocity reference at the completed 500-epoch budget,
while retaining classic as the smaller fixed-K comparison. Both improved late in
training, so a separately authorized same-run continuation to epoch 2500 is
reasonable. It would investigate remaining learning, not certify universal
convergence. There was one initialization per architecture; this comparison does
not estimate seed-to-seed variance. The reserved-test result is now consumed for
this experiment and must not become an iterative architecture-selection score.

The following commands are **unexecuted**. Each uses its own saved root profile
and latest checkpoint, restoring the existing optimizer, RNG, normalization and
exact grouped split in the same managed run. The root loader accepted these
profiles against each run's recorded configuration. Do not run them automatically.

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_WindFarm/src "$wf_python" -u train.py \
  --config "$wf_classic/configs/core_source.json" --workflow forward --device cuda:0 \
  --epochs 2500 --resume-checkpoint "$wf_classic/latest_model.pt" --yes

rtk proxy env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:Case_WindFarm/src "$wf_python" -u train.py \
  --config "$wf_dense/configs/core_source.json" --workflow forward --device cuda:0 \
  --epochs 2500 --resume-checkpoint "$wf_dense/latest_model.pt" --yes
```

Additional fields, turbine-local physics, exact power/controls, actual CFD
boundaries/mesh volumes, stronger modular cores and deliberately designed
extrapolation splits remain separate studies. No extra run, physical refinement,
thermal surrogate, automatic continuation or PR merge was performed here.
