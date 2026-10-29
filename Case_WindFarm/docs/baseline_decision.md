# WindFarm forward baseline decision

**Status:** The bounded WindFarm W1 Run 2105 continuation completed at epoch
300 / 1,200 cumulative updates. Its validation-selected `best_field.pt` is
epoch 295 / update 1,180. On the matched Q=40,960, E512, 90-row physical
comparison, e295 has higher volume, hub-band, and downstream errors than the
Run 2103 e2475 dense incumbent on every row. This is a no-go for W1 promotion;
e2475 remains B_inc. Historical validation and reserved-test outputs have
already been opened and are development evidence for this study.

## Selected reference

Use Run 2103's explicitly recorded field-best checkpoint as the provisional
WindFarm dense incumbent for receiver-first physical fidelity. Keep Run 2101
e495 as the documented historical comparator and retain its checkpoint alias
conflict as a separate identity record:

| Run | Exact checkpoint | Saved epoch / updates | Recorded best validation volume MSE | Role |
|---|---|---:|---:|---|
| 2103 dense | `Trained_Results/WindFarm/HONF_Forward_Runs/Run_2103_20260913_135849_windfarm_dense_pairwise_b16_q8192/checkpoints/best_field.pt` | 2,475 / 66,825 | 0.00452088 | Provisional WindFarm dense incumbent and W1 refit source |
| 2101 dense | `Trained_Results/WindFarm/HONF_Forward_Runs/Run_2101_20260913_014752_windfarm_dense_velocity/checkpoints/best_field.pt` | 495 / 26,235 | 0.00393833 | Exact historical comparator; selected checkpoint and alias conflict audited |
| 2102 classic | `Trained_Results/WindFarm/HONF_Forward_Runs/Run_2102_20260913_135849_windfarm_classic_k6_b16_q8192/checkpoints/best_field.pt` | 2,440 / 65,880 | 0.00629101 | Classic fixed-K comparator |

The Run 2101 manifest and training records select e495. Its top-level
`best_by_field_mse_model.pt` is a different e580 continuation; details follow
below. I replayed the exact e495 and e2475 checkpoints on the same frozen
validation panel: 90 rows / 30 layout groups, seed 42, E512, 32,768 volume
queries plus 8,192 hub-band queries per row, with receiver chunks of 512.
Dimensional RMSE and the U_ref-normalized values are directly comparable:

| Region | Run 2101 mean / p95 / worst RMSE (m/s) | Run 2103 mean / p95 / worst RMSE (m/s) | Run 2103 rows with lower RMSE |
|---|---:|---:|---:|
| Volume | 0.01762 / 0.02191 / 0.02345 | 0.01077 / 0.01349 / 0.01468 | 90 / 90 |
| Global hub-height slab (`|z-hub| <= 0.5D`) | 0.02879 / 0.03652 / 0.03964 | 0.01950 / 0.02569 / 0.02823 | 90 / 90 |
| Downstream envelope | 0.03593 / 0.04578 / 0.05649 | 0.02265 / 0.02929 / 0.03180 | 90 / 90 |

The hub-height metric samples native cell centres across the full x-y slab; it
is not turbine-local rotor-disk error. The downstream envelope is conditioned
on active turbine geometry (`0 < dx <= 10D`, distance in the y-z plane no more
than `1.5D`) and is the receiver-focused native-region metric in this replay.
W2's 17 hub/interior/edge queries per turbine are continuous geometry points;
the replay did not compare predictions at those exact points with an
interpolated native reference. Treat W2's receiver-feature utility as empirical
validation-cohort association, not physical validation of rotor-neighborhood
velocities.

Run 2103's pooled vector relative L2 is also lower in all three regions. The
equal-case volume RMSEs are 0.0019576 and 0.0011970 of U_ref=9 m/s; hub-band
means are 0.0031988 and 0.0021663 of U_ref; downstream means are 0.0039924
and 0.0025169 of U_ref. By the plan's receiver-first physical ranking, e2475
is the current B_inc and dense-tensor refit source. The checkpoint-native
standardized volume MSEs are 0.00408517 for e495 and 0.00456954 for e2475,
but these use different per-run normalizers and are not a cross-model ranking.
The runs also have different batch/query budgets, so the measured physical
advantage does not isolate architecture or optimization budget. The
historical reserved-test outputs have already been consumed and are not an
untouched test.

Run 2101 contains a second identity conflict. Its top-level
`best_by_field_mse_model.pt` is epoch 580 with 30,740 updates and metric
0.00377669. The run manifest and summary still name e495; the manifest records
the run as stopped by request at epoch 581, while the summary says epoch 500.
The explicit `checkpoints/best_field.pt` remains e495. Treat the e580 alias as
a distinct historical continuation with unsynchronized selection metadata,
not as the documented incumbent. Named WindFarm checkpoint selection now
resolves through each run manifest and verifies the selected field checkpoint
epoch and metric before evaluation.

Run 2104 is not an incumbent: its manifest records an interrupted run, while
its epoch counters disagree (`last_completed_epoch=1809` in the manifest and
epoch 100 in its summary). Preserve the artifacts without inferring completion
or using its selected checkpoint for a baseline decision.

## Native data and task boundary

The linked source contains 600 direction rows for 200 independent layouts,
with three categorical directions per layout and 6–30 turbines. Splits and
uncertainty groups use `layout_index`. The environmental bank contains 512
geometry-derived quadrature tokens. The full native velocity target is ragged
3-D `[Ux, Uy, Uz]` in m/s, stored as mmap-backed fields over 2,484,440,512
cells with row-specific axes and offsets. The model task is velocity; it does
not establish turbine power, AEP, yaw response, or actuator-load prediction.
No second direction rotation belongs in the adapter.

The local data link resolves outside Git. Compact metadata is used only for
row geometry and operating categories; full-volume fields remain streamed by
the native reader. No raw arrays or local path maps belong in the source tree.

## Refit boundary and W1 evaluation

The selected 2103 dense checkpoint uses `dense_pairwise_field`, native volume
schema `wind_farm_volume_npy_v1`, three output channels, 3-D geometry, and
coordinate scale `[50, 38, 6.25]`. Its materialized core state contains four
case/module/environment encoders and 58 fine-backend tensors. The dense common
context has 73 tensors and is not compatible with the three-term field head.
The WindFarm initializer copies only the compatible encoders and fine backend
through `warm_start_three_term_full_access`; the three-term common head remains
freshly initialized and must be refit against WindFarm training data. The new
W1 profile sources only from the exact manifest-selected 2103 e2475 checkpoint.
This is an explicit refit transition, not an exact prediction conversion.

The WindFarm wrapper and checkpoint evaluator register
`three_term_full_access_honf` through the same 3-D prepare/decode path. W1
evaluation can use native receiver counts and environment-token counts; the
optional `--env-token-shape NX NY NZ` switch changes geometry quadrature only,
and the result records the exact shape and token count. It does not read target
fields or wake-loss outcomes to construct model inputs. The opt-in adaptive
WindFarm adapter supplies its own geometry-only receiver anchors: E512
centre-box quadrature anchors with normalized weights, plus 17 Y-Z rotor-plane
points per turbine (hub, eight half-radius, and eight edge locations) with
dimensionless role-separated weights. This set is fixed from row geometry and
does not depend on labels, sampled output queries, or query chunking.

Run 2105 is the bounded W1 refit from the exact manifest-selected Run 2103
e2475 `checkpoints/best_field.pt`. The materialized strict warm start copied
74 compatible encoder/fine-backend tensors, discarded all 73 dense common
head tensors, and freshly initialized 12 three-term common-head tensors; this
is a refit, not a prediction-preserving conversion. A strict load passed on a
native 11-turbine E512 materialization. The run UUID is
`7d2dbb04-caae-404d-a5a7-cf98293100a2`.

The first segment completed epochs 1–2 / 8 updates. It was continued from the
exact e2 `checkpoints/latest.pt` with its optimizer, RNG state, and original
420/90/90 row split. The second invocation completed epochs 3–25 and 92 more
updates, for 100 total. That historical e25 checkpoint had SHA256
`1d7b0db11f7b479144067cd264033c6bb426880a054c0883223464acece4d3ba`; its
e25 latest checkpoint SHA256 was
`5e22d1004e58a423ee7af53f9fd9a5fe6463013fc997e73292d98676f9989e9f`.
After e25 recovery, the Run 2105 root manifest was coherent at epoch 25/update
100 and selected e25 with same-run normalized validation volume MSE 0.0999073.
Validation was
recorded at epochs 2, 5, 10, 15, 20, and 25. The 92-update continuation wall
was 35.55 s; training logs reported maximum allocated CUDA memory of 2,587.1
MiB across the 100-update segment. The normalized validation MSE is not
directly rankable against Run 2103 because the dense and refit runs use
different target normalizers.

The authorized follow-on resumed the same managed run from e25/update 100 to
e100/update 400, adding exactly 300 updates with four train batches per
epoch. It completed on physical GPU 2, UUID
`GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`, logical `cuda:0`, with 96.99 s
job wall and no OOM. The run root and UUID remained
`Run_2105_20260926_123147_windfarm_three_term_full_access_refit_2ep4batch` /
`7d2dbb04-caae-404d-a5a7-cf98293100a2`; the manifest is completed at epoch
100. All 100 metric rows have finite recorded numeric values and updates
4–400. The maximum recorded CUDA allocator use was 2,587.1 MiB. The exact
e100 `latest.pt` SHA256 is
`6715794277e56524f64b9b47355508d20c8892e4d10f60c2f136dda8330412af`; the
selected `best_field.pt` has the same epoch/update/best metric and SHA256
`77839410f3f14dd892578354e9798be495e9da9038cce863bad732d81a4fbd8b`.
Validation volume MSE continued to improve in aggregate from e25 to e100,
with fluctuations: e25 0.0999073, e30 0.0884938, e35 0.0739090, e40
0.0594384, e45 0.0641467, e50 0.0898645, e55 0.0560737, e60 0.0492487, e65
0.0609530, e70 0.0493564, e75 0.0422794, e80 0.0594020, e85 0.0417263, e90
0.0438313, e95 0.0477899, and e100 0.0398403. The latter is the current
within-run best. Run 2103's standardized validation MSE of 0.00452088 uses a
different train-fitted normalizer, so the two losses alone do not establish a
cross-checkpoint physical-accuracy ranking.

The first resume attempt exposed a path bug: a checkpoint under
`<run>/checkpoints/` was treated as the run root, placing e3–e25 metrics and
legacy outputs inside that directory while leaving the run-root manifest at
e2. `train.py` now resolves a managed checkpoint to its parent manifest and
rejects ambiguous/missing nearby manifests rather than creating a nested
checkpoint run. A focused regression test covers root resolution, conflicting
identity, missing manifests, and standalone checkpoints. Before recovery,
both the original e2 artifacts and resume-output artifacts were copied under
the ignored `Run_2105.../recovery_backup_before_e25_manifest_repair/` folder.
Metrics e1–e2 and e3–e25 were audited for matching headers, contiguous epochs,
finite entries, and update counts 4–100, then reconciled. Canonical e25
checkpoints, metrics, summary, and manifest agreed before the e100 resume;
the original and resume-output copies remain in the backup, and no training
artifacts were deleted.

### Native e100 validation panels

The six panels below use the exact e100 `best_field.pt`, validation split,
sample seed 42, `case_batch_size=1`, receiver chunk 512, and physical GPU 2.
The 90-row panels use the full grouped validation split. The Q=40,960 panel
matches the dense e2475 replay in all 90 `source_index` rows, query counts,
E512, and sampling seed. Query ledgers and per-row metrics are retained in
`Trained_Results/WindFarm/HONF_Forward_Runs/Run_2105_20260926_123147_windfarm_three_term_full_access_refit_2ep4batch/evaluations/w1_e100_panel_ledger.json`
(SHA256 `287739b8343c4a768685bd07c4de92268794415c51d995b2c3852477c1e0dee7`).
Every row in each panel has the recorded Q counts shown below; each row also
uses the listed geometry-token count. The synchronized model-prediction
interval begins before device batch materialization and includes host-to-device
transfer, prepare, and decode. The loop interval covers the complete case
loop. Allocated/reserved memory below is the CUDA peak from that invocation.

| Panel | Rows | Q volume + hub / row | E shape (tokens / row) | Mean RMSE volume / hub / downstream (m/s) | Sync model / loop (s) | Peak allocated / reserved (MiB) | Job wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q8192 | 90 | 8,192 + 8,192 | 16×8×4 (512) | 0.075379 / 0.125558 / 0.146460 | 5.058 / 6.785 | 322 / 442 | 10.90 |
| Q65536 | 90 | 65,536 + 8,192 | 16×8×4 (512) | 0.075135 / 0.125558 / 0.145241 | 21.903 / 30.474 | 383 / 646 | 34.66 |
| Q40960 matched panel | 90 | 32,768 + 8,192 | 16×8×4 (512) | 0.075244 / 0.125558 / 0.145482 | 12.355 / 16.781 | 348 / 468 | 20.91 |
| E128 sensitivity | 18 | 8,192 + 8,192 | 8×4×4 (128) | 0.080818 / 0.114218 / 0.152130 | 0.860 / 1.190 | 107 / 140 | 5.04 |
| E512 sensitivity | 18 | 8,192 + 8,192 | 16×8×4 (512) | 0.075136 / 0.123892 / 0.148008 | 1.027 / 1.373 | 323 / 414 | 5.44 |
| E1024 sensitivity | 18 | 8,192 + 8,192 | 16×8×8 (1,024) | 0.085266 / 0.117734 / 0.150288 | 1.831 / 2.176 | 609 / 784 | 6.22 |

Q8192 and Q65536 have similar physical errors on the same 90 rows. The
E-density sample uses the fixed, input-selected rows `120, 504, 121, 505,
122, 506, 207, 366, 208, 367, 209, 368, 3, 387, 4, 388, 5, 389`, spanning
M=9, 19, and 28 and directions 270°, 285°, and 300°. Density effects are
non-monotonic in this 18-row sample; they do not show a simple benefit from
increasing E. The queried hub-band metric is a global height slab, and the
downstream envelope is evaluated only on the volume-query samples.

At matched Q40960, the Run 2103 e2475 dense checkpoint has lower physical
RMSE than the e100 refit on all 90 same rows in each region. Equal-case
mean/p95/worst RMSE (m/s) is:

| Region | Dense e2475 | Three-term e100 | e100 rows better than dense |
|---|---:|---:|---:|
| Volume | 0.010773 / 0.013495 / 0.014678 | 0.075244 / 0.089101 / 0.092867 | 0 / 90 |
| Global hub-height slab | 0.019497 / 0.025686 / 0.028235 | 0.125558 / 0.151460 / 0.154661 | 0 / 90 |
| Downstream envelope | 0.022652 / 0.029285 / 0.031797 | 0.145482 / 0.162678 / 0.170558 | 0 / 90 |

Mean paired e100-minus-dense increases were 0.064472 m/s in volume,
0.106061 m/s in the hub slab, and 0.122830 m/s downstream. This direct
physical comparison rejects e100 as a replacement for B_inc at this gate;
it does not establish that further refitting cannot improve the candidate.
The dense historical replay did not record synchronized latency or allocator
peaks, so the timing and memory values above are refit-side only.

## W2 cohort feasibility

An input-only grouping by exact turbine count and categorical direction yields
24 validation cohorts (54 candidate rows) and 33 reserved-test cohorts (66
candidate rows) with at least two distinct layouts per cohort. Cohort
membership uses only `n_turbines`, `wd_deg`, `layout_index`, and the existing
layout-grouped split. No `wake_loss_pct` value was read when forming this
feasibility summary. These counts show that a finite-library comparison can
be formed, but the reserved-test split is already consumed for the velocity
study.

No additional site constraint is specified by the current data contract, so
W2 cannot report physical feasibility. The dataset manifest identifies
`wake_loss_pct` as a percent-valued scalar wake-loss/regression target and the
WindFarm report provides its range; the formula and constituent physical
quantities are not present in the inspected materials. Any scalar head remains
empirical and does not claim power or AEP. The W2 path freezes input-only
cohorts, then compares a train-only geometry/static ridge control with a
separately trained head that adds Run 2103 predicted rotor-neighborhood
velocities. Train-layout grouped CV chooses the ridge penalty before validation
outcomes are indexed. Each policy selects one candidate per exact turbine-
count/direction cohort; results include every eligible candidate's stored
outcome, the selected versus equal-count random outcomes, an oracle comparison,
and the row exposure. It is validation-only and launches no CFD solves.
Each cohort record pairs the selected candidate with its same-cohort exact
uniform-random expectation and retrospective stored minimum. Across 24
validation cohorts (54 candidate rows), the same `layout_index` can recur in
three direction cohorts, so selected, random, and oracle aggregate means are
descriptive finite-benchmark comparisons rather than independent-sample
inference. The equal-count random control reports its exact expected point
value; no sampling confidence interval is reported.

The compact source stores `wake_loss_pct` in a 2,528-byte compressed NPZ
member, with no separate target `.npy`. Decoding that member materializes all
600 values; the W2 code indexes train labels for fitting and eligible
validation labels for the retrospective cohort audit only. It does not index,
score, summarize, or select from test labels. This storage-level read is logged
in the W2 output. The test split has already been evaluated for velocity and
is not an untouched test. Any W2 result is a finite-library choice among
stored layouts, not continuous layout generation.

## W3 independent-solver and source-lineage gate

No independent WindFarm CFD replay or layout regeneration can be performed
from the inspected local package. The linked source's
`Dataset/links/wind_farm/README.md` says the compact bundle does not include raw
OpenFOAM cases, meshes, residual histories, or logs. Its
`Dataset/links/wind_farm/family_volume/README.md` describes only the packaged
final fields and says they can be rebuilt with `../pack_volume_fields.py`, but
that exporter is absent from the linked source root. The linked root contains
the tensor bundle, three plotting scripts, reference PNGs, and the `family_volume`
directory; no layout-generator or OpenFOAM solver entrypoint/configuration was
found. The family-volume `manifest.json` records array format and ordering,
not the solver setup. The repository's
`Dataset/PHYSICS_AND_DATA.md` documents `source_time` as the selected final-time
directory (observed 529–896), and names the sampled layout-generator columns
`req_min_sep_D`, `sep_capped`, `slack`, and `clump` without defining their
formulas. Neither that document nor the linked README defines the `wake_loss_pct`
formula or its constituent physical quantities. These stored outputs do not
support a fresh-solver reproducibility claim, a layout-generation claim, or
power/AEP prediction.

To clear W3, request from the data owner the original layout-generator source,
version, seed, and complete parameters; the OpenFOAM case dictionaries and
mesh; the actuator/turbine source model; inlet and boundary conditions,
turbulence model, solver version/settings, residual and convergence histories;
the exact mapping from each run to the exported final-time directory; and the
definition and provenance of `wake_loss_pct`. Until those materials are
available, W3 remains an evidence gate rather than a surrogate or analytic
wake substitute.

The two native replay outputs are kept under ignored
`Case_WindFarm/diagnostics/generated/w1_run2101_e495_val/` and
`w1_run2103_e2475_val/`. Evaluations used physical GPU 2 with Q=40,960 per row,
E512, and receiver chunks of 512. Synchronized evaluator timing and peak CUDA
allocator memory were not captured; the replay is not a latency or memory
benchmark. A live `nvidia-smi` snapshot during the 2103 replay showed 69%
utilization and 976 MiB on the assigned GPU, which is an instantaneous sample,
not a measured peak.

## W2 validation-only result

The bounded W2 run completed with the exact manifest-selected Run 2103 e2475
`best_field.pt` (run UUID
`d6b0c113-4e83-47a6-bff7-2cb3a897dec8`) on physical GPU 2, UUID
`GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`, logical `cuda:0`. The retained
output is
`Trained_Results/WindFarm/HONF_Forward_Runs/Run_2103_20260913_135849_windfarm_dense_pairwise_b16_q8192/evaluations/w2_library_binc_e2475/`.
`w2_selection_metrics.json` SHA256 is
`436efccc48bca32f21bdd4f5684873ae464c01a733a065c1641dc8a86a6acbac`, and
`cohort_manifest.json` SHA256 is
`80fc40d5c906365bb6a1e75ca96f6b806f8de877614c2c43b68306842efa91d2`.
Process wall was 8.00 s; synchronized prediction and total evaluation times
were 3.809 s and 4.718 s. Peak CUDA allocated/reserved memory was
324,576,256 / 968,884,224 bytes (about 310 / 924 MiB).

The model predicted for 420 train and 54 validation candidates, with 512 E
tokens per row (242,688 tokens total), 17 geometry-defined rotor-neighborhood
query points per turbine (141,168 queries total), and receiver chunks of 512.
The train-only scalar fits used 420 rows across 140 layout groups with
five-fold group CV. Validation scalar RMSE was 1.9200 percentage points for
the geometry-only static ridge and 1.3053 points after adding e2475-predicted
receiver velocity features. On the 24 frozen validation cohorts, the receiver
head selected a mean stored `wake_loss_pct` of 15.8445%, versus 15.8777% for
the static control; the exact equal-count uniform-random expectation was
18.7230%, and the retrospective stored minimum was 15.7520%. The
receiver-velocity ridge is a learned-feature finite-library control; it is
not an adaptive graph or a physical inverse-design policy. It changed the
selected row in only 2 of 24 cohorts: n=12/wd=270 worsened
by 0.8260 percentage points, and n=12/wd=285 improved by 1.6241 points; the
other 22 selections were identical. The paired mean outcome difference was
−0.033254 percentage points (median 0, range −1.6241 to +0.8260). These are
descriptive results on a repeated benchmark where the same layouts recur
across directions, not evidence of a meaningful design gain or graph effect.

The output records the compressed NPZ member exposure: decoding materialized
all 600 `wake_loss_pct` values, then the workflow indexed 420 train values
and 54 eligible validation values after prediction-only selection was
frozen. It indexed/scored no reserved-test outcomes. The formula for the
percent-valued wake-loss target remains undocumented; results are empirical
scalar associations only. The queried receiver locations lack exact native
target support, so they are not physically validated rotor velocities.
No power/AEP proxy or new CFD solve was used.

## W1 e300 continuation and matched physical gate

The predeclared e100→e300 continuation resumed the same managed Run 2105 with
its saved optimizer and RNG state, four train batches per epoch, and no more
than 800 new updates. It stopped at epoch 300/update 1,200; the manifest is
completed with the original run UUID `7d2dbb04-caae-404d-a5a7-cf98293100a2`.
(CPU readback confirmed 86 AdamW state entries with step 400 at e100, 1,200 at
e300, and 1,180 in selected e295; Python, NumPy, Torch, and CUDA RNG states
were present.) The 300-row epoch history is contiguous and finite, with
exactly four updates per epoch. `checkpoints/latest.pt` is the exact
e300/update 1,200 checkpoint (SHA256
`bff578f879dfcd213cd6f80d55e8ff8f7614ed76a720535dd653712fab5defe3`).
Validation selected e295/update 1,180 as `best_field.pt` (SHA256
`97d1b5c4c5b72787edbf984c6ff1f829cf15c1a173657ac983374ad6aee34e20`), with
best validation volume MSE 0.01530734; the e300 final validation volume MSE
was 0.0160914. Thus validation loss improved from e100's 0.0398403 but
fluctuated near the end; normalized losses remain within-run evidence, not a
cross-checkpoint physical ranking.

Training metrics sum to 86.94 s of epoch training time and 146.69 s of
validation time for epochs 101–300. A process snapshot read 3:58 elapsed at
epoch 294, and the final six epochs add 9.86 s of recorded training/validation
time; the full job wall was therefore about 4:08 (the launch-to-exit wall was
not independently time-wrapped). The largest per-epoch CUDA allocated-memory
record was 2,584.2 MiB. `nvidia-smi` samples during the run showed PID 3837053
on physical GPU 2 / UUID `GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`, with
3,563–3,565 MiB in use and 64–100% utilization; these are snapshots, not an
integrated utilization measure.

The selected e295 field checkpoint was evaluated on validation using the same
90 source rows, sample seed 42, 32,768 volume plus 8,192 hub-band queries per
row, E shape 16×8×4 (512 tokens), and receiver chunk 512 as the frozen Dense
e2475 reference. The Dense result is the previously executed exact e2475
Q40960/E512 replay; row, query-count, seed, and E ledgers were checked for
equality, without rerunning Dense. The retained matched ledger is
`Trained_Results/WindFarm/HONF_Forward_Runs/Run_2105_20260926_123147_windfarm_three_term_full_access_refit_2ep4batch/evaluations/w1_e300_best_e295_q40960_e512/w1_e300_matched_ledger.json`
(SHA256 `caf223b2fe233799f9db612b209bfab4b9d4ff9ab60e876a90138a4b4a8a61f5`).
The Dense reference is
`Case_WindFarm/diagnostics/generated/w1_run2103_e2475_val/metrics.json`,
whose selected checkpoint is e2475 at
`Trained_Results/WindFarm/HONF_Forward_Runs/Run_2103_20260913_135849_windfarm_dense_pairwise_b16_q8192/checkpoints/best_field.pt`
(SHA256 `e5cfcc1487b20b749ce2bf5e0ddaab295b0f6f5acd35244ba8ba5a39e1c811d4`).

| Region | Dense e2475 mean / p95 / worst RMSE (m/s) | W1 e295 mean / p95 / worst RMSE (m/s) | e295 rows better than dense | Paired mean e295−dense (m/s) |
|---|---:|---:|---:|---:|
| Volume | 0.010773 / 0.013495 / 0.014678 | 0.043472 / 0.052458 / 0.055185 | 0 / 90 | +0.032699 |
| Global hub-height slab | 0.019497 / 0.025686 / 0.028235 | 0.073290 / 0.090370 / 0.094594 | 0 / 90 | +0.053793 |
| Downstream envelope | 0.022652 / 0.029285 / 0.031797 | 0.088763 / 0.109493 / 0.116516 | 0 / 90 | +0.066111 |

The e295 evaluation ran on physical GPU 2, logical `cuda:0`; external evaluation
wall was 20.92 s. The synchronized model-prediction interval was 12.301 s
(including batch materialization/host-to-device transfer, prepare, and decode),
and the full case loop was 16.762 s. Peak CUDA allocated/reserved memory was
365,188,096 / 490,733,568 bytes (348 / 468 MiB), for 3,686,400 total query
points. The timing and memory describe this W1 pass only; no matched Dense
timing was captured.

This gate is a **no-go for promotion**: e295 loses on all 90 matched rows in
each of the three physical regions, despite improved within-run validation
loss. Keep Run 2103 e2475 as B_inc. The authorized W1 continuation stopped at
e300; no further WindFarm training is authorized by this result. Do not use
the already opened reserved-test outcomes for selection.
