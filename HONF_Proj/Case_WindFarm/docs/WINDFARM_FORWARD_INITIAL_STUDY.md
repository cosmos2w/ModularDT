# WindFarm forward velocity: initial bounded study

## Execution status

Implementation and execution are in progress. This document does **not** yet
claim completed training or final model results. The authorized study contains
two from-scratch 500-epoch runs only. Numeric results and actual run-owned
artifact paths will be added after execution.

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

Pending actual execution: both managed 500-epoch histories, validation-only
checkpoint selection, one reserved-test comparison, native-volume reductions,
spatial figures, representation diagnostics and controlled timing on GPU 0.

The 500-epoch endpoint is an initial resource-bounded assessment. Any future
same-run continuation command will be reported separately and left unexecuted.
