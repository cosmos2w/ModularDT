# Thermal Tree faithfulness development

This execution specification records the 2026-10-04 amendment to the local
Tree faithfulness and response-transfer plan. Three fresh candidates first
complete 100 development epochs. Tree-F and Pair-F may then continue together
after an evidence review, with a maximum of 500 each. A separately requested
fresh Run1804-like dense baseline trains on the same selected data to 1,000.
This is development on exposed validation evidence; it is not a formal run.

## Data and budget

Use [the fixed development protocol](Thermal_Model_Development_Protocol.md)
and `/data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json`:
150 training cases, 22 validation cases, train-only normalization, selection
seed 20261004 and initialization/query seed 0. Preserve original partitions.
The validation M3/M5/M7/M10 counts are 6/6/6/4; training contains only four
cases in each of those categories. The sole selected training atlas anchor
is0348. Do not add excluded cases to response training or evaluation.

Q1024, FP32, effective batch 48 and the absolute 1,000-epoch schedule stay
common. One epoch has 150 training visits,19 microbatches,4 optimizer updates
and 153,600 primary fluid queries. Count P0/P1/P2, calibration, shadow and
response work separately. Milestones, latest, best-field selection and plots
occur every 100; no unsaved checkpoint may enter selection.

| Arm | Fresh implementation | Authorized stop |
|---|---|---|
| Tree-L | Legacy receiver Tree and historical feature semantics |100 |
| Tree-F | Measure-consistent Tree, explicit dual control donors, source-local features |100, reviewed matched continuation≤500 |
| Pair-F | Independently trained receiver/source modulation, common fine backbone, source-local features |100, reviewed matched continuation≤500 |
| Dense-D25 | Run1804 dense architecture with maintained native development configuration |1,000 |

The maximum is2,100 combined epochs:315,000 primary case visits,8,400
optimizer boundaries and 322,560,000 primary queries. Dense's e1000 is a
longer-budget reference; age-matched comparisons use its e100 and e500.
Historical Run1804 remains a full-data, differently trained reference.

## Computation and interpretation

Tree-F retains every fine physical source. Its receiver index groups exact
coordinate/role ties, sums their measure and never splits a coincident block.
Cuts use distinct support blocks; live boundaries use weighted child
centroids. Node summaries and structural pair/dual-donor occupancy use
physical measures, with query counting measure declared independently.
Integer executed rows and control-pooling work remain separate from these
representation costs. Legacy boundaries and checkpoint semantics remain
available through the old architecture identifier.

The retained e100 derivative audit found a second representation issue:
fine-row min/max bounds distribute an equal-tie coordinate subgradient by
row count. Tree-F now takes node extrema over canonical coordinate/role
blocks, whose coordinates distribute derivatives by physical child mass.
Exact-refinement bound values stay unchanged; the legacy path remains
unchanged. Preserve the original trained failure and the separate same-weight
derivative verification. This is a derivative implementation repair, not
additional training exposure or a new predictor fit.

The first screen binds `structural_measure_policy_version=1`: physical
integrals are case balanced and the final mean includes every mechanism,
including a zero MM term for an entirely self-only M1 batch. This records
the code actually loaded by Run3204. Policy 2 excludes entirely ineligible
mechanisms. It can enter this campaign only as an explicit e100-to-e101
Tree-F objective amendment, alongside the common physical denominator
amendment. Preserve the old calibration receipt and repeat its selected
training module-count calibration; do not reuse the changed cost's scale
silently. Model resume permits this one declared option change while keeping
all other architecture, normalization, optimizer and RNG contracts strict.

The planner can inspect all current sources. Tree-F controls consume only
their declared module/environment membership summaries, receiver geometry
and roles, prescribed context, explicit donor presence and phase. Rich
all-source planner embeddings cannot enter its control heads. Export native
value donors, both control donor types, planning dependencies and phase
ancestry. P1/P2 states can carry earlier transport/local-refinement content.
Learned support describes computational dependencies, not physical causality.

The named `source_local_v3` adapter fits a fixed heat scale on selected
training inputs, including the dataset input transform, and persists it in
the checkpoint. Source tokens use only their own heat. Prescribed/background
context excludes total/mean/maximum heat; ordinary Stage-A heating and
boundary physics remain native. Pair-F uses the same semantics and fine
initializer but independent, identity-start direct modulation parameters.
Its dense source work is a capacity control, not a sparse timing control.

## Review and evaluation

Freeze four representatives from input-only M/geometry metadata before
reading model errors. Use all 22 for statistical native-unit physical errors,
per-M counts and tails, light whole-wrapper atom-refinement checks and
fixed-geometry heat-null responses. Detailed Q8192 fields, residuals,
phase graphs, controls and derivative diagnostics use only those four.

| Boundary | Required review |
|---|---|
|100 | Learning/exposure, all 22 fidelity/null changes, full-rebuild invariance, finite task/control gradients, trained Pair comparison, four-case utility controls and measured execution. Record the decision before any paired extension. |
|200 | Compare each pair member to its 100 state; verify the response amendment, invariance, thermal sensitivity, losses and additional work. Record continuation or matched stop. |
|300/400 | Repeat all 22 physical/null summaries and light semantic checks at saved milestones; reuse representative arrays for unchanged protocols. |
|≤500 final | Close A/B/C at an exact matched age and consistently selected saved weights; inspected physical/graph/refinement/response figures, execution and one completed bounded inverse panel. |
|Dense100…1000 | Retain native selected-validation statistics and plots each 100; detailed forward comparisons at100, matched 500 when relevant, and final 1000. |

The optional pair continuation preserves optimizer/RNG/data/normalization
and the ordinary epoch 101 physical-objective amendment and 0348 callback.
Only after verifying the generator and stored controls, enable the common
benchmark heat-null auxiliary on two eligible selected-training cases per
epoch with Q256. Fixed-total transfers preserve geometry/context and
nonnegative heating. Supervise u/v/p/omega increments only, with fixed train
scales and one train-only gradient calibration of the coefficient at≤0.1.
Use the final ordinary optimizer boundary; no added optimizer steps.
Dense-D25 has the maintained ordinary continuation without this new term;
disclose that objective difference. Null validation has a fixed direction
and an alternate input-selected pair/amplitude. Thermal sensitivity must
remain visible; no nonzero perturbed-temperature labels are invented.

Check genuinely matched different-heat families inside 150/22 only. If held
nonzero response labels are absent, C-positive-response transfer remains
unavailable. No new reference solve is authorized.

The inverse diagnostic has four cases, two common input-only starts and at
most 10 attempted trust steps. Compare Tree-F full-joint, graph-block and
size-matched random-block modes with Pair-F full-joint. Accept against
observed sensors only with at most two trials; held sensors cannot select
steps. Freeze discrete topology during each proposal while recomputing
continuous physics, then rebuild after accepted steps. Persist complete
trails, charged forward/VJP calls, accepted/rejected steps and feasibility.
Disclose fallback and rank limits; candidate predictions are surrogate
evidence, not independently verified physical designs.

### Actual first-screen review, 2026-10-04

Tree-L Run3103, Tree-F Run3204 and Pair-F Run3202 completed 100. Each visited
15,000 selected cases, made 400 ordinary updates and queried 15.36 million
primary fluid points. Tree-F Run3201 was intentionally censored after a
native mass-cut near-tie failure; its work and saved evidence remain separate.

| Exact e100, all 22 equal-case mean RMSE | Tree-L | Tree-F | Pair-F | Dense-D25 |
|---|---:|---:|---:|---:|
| Fluid temperature |4.52385 |5.47149 |4.36374 |5.34786 |
| Material temperature |3.50903 |3.74484 |3.79236 |4.22994 |
| Surface temperature |4.11126 |4.32049 |4.51310 |4.56745 |
| Inlet/outlet pressure difference |0.135377 |0.120159 |0.114684 |0.054518 |

These are native dataset units on exposed development validation. Tree-F
does not establish added predictor/organizer value at100. Removing its
collective control at fixed access improves four-case fluid-temperature
RMSE from 5.16400 to4.74483. Every measured intervention changes zero binary
native pairs and zero fine rows; density/control changes are nevertheless
nonzero. All actual control donor catalogues have full support, making an
actual excluded-donor zero-path test unavailable. Admitted-control and
planner Jacobians are nonzero; the separately manipulated exclusion test
is a diagnostic property, not trained sparse-locality evidence.

All 22 whole-rebuild permutation/unequal-refinement predictions and effective
representation quantities pass their declared floating tolerance. The four
representative split-coordinate VJPs fail before the extrema derivative
repair; feature/mass/length VJPs and permutations pass. Native field-loss
probes give finite nonzero Tree/Pair control gradients, including the
Tree hard-value/soft-organizer training route. These probes do not establish
functional usefulness. Known-null RMS flow increments remain nonzero;
Tree-F is1.69/2.56/1.94/2.74 times Pair-F in u/v/p/omega. Both retain nonzero
model-only thermal sensitivity. No held nonzero-response family exists
inside the selected 150/22 cases.

The reviewed next step is a matched extension **only to200**, conditional
on verified same-weight coordinate derivatives and the measured execution
check. It tests the planned common physical/null-response objective at101,
with the explicit Tree structural-cost amendment and training-only
recalibration. It does not assume an organizer benefit or authorize automatic
continuation to500. Review 200 before selecting any later boundary.

Observed first-screen process allocation was Tree-L0.827h, Tree-F1.647h,
Pair-F0.227h; Tree-F shared GPU 2 with Dense. Dense continues its separately
authorized 1000. Proposed continuation puts Tree-F on GPU 1 beside an
untouched external 13GiB process and Pair-F on GPU 2 beside Dense11GiB.
Expected next 100 is approximately 40–110 minutes for Tree-F and 14–25 minutes
for Pair-F, subject to measured contention; revise after five resumed epochs.
The conservative additional 400 ceiling from first-screen allocation is
6.59 Tree-F and 0.91 Pair-F process hours, not exclusive GPU compute time.
Merge co-resident allocation intervals instead of adding their GPU hours.

The first continuation attempt exposed a telemetry integration fault: the
seven new null-loss scalar columns were absent from the existing CSV schema.
Pair-F completed 101 train/validation but failed before recording that row or
saving a checkpoint; Tree-F was interrupted during its 11th microbatch to
avoid the same failure. Both restart from their intact 100 checkpoints.
Retain attempted work separately: Pair 150 visits/four updates/153,600 primary
queries, and Tree 80–88 visits/one completed optimizer boundary/81,920–90,112
primary queries. Pair's native telemetry persisted before the CSV failure:
its discarded 101 attempt additionally made four null wrapper calls with
1,024 primary fluid and 3,904 requested role queries; the raw record is
preserved separately by process/attempt. Tree stopped before its auxiliary
callback. Do not sum repeated epoch 101 telemetry as accepted training age.
The logging repair declares the columns and extends the historical
header after resume validation, retaining every old cell and a raw backup;
old epochs receive blank auxiliary cells. It changes no objective or model
state. Keep the failed logs/exits locally under the campaign evidence root.

The subsequent Tree-F GPU 1 attempt completed unsaved 101–103, then became
very slow beside the untouched external 13.75GiB process: recorded train
times were 65.07,135.31 and 391.09 seconds. This is observed contention;
its causal contribution was not isolated by paired training. Preserve 450
completed visits/12 updates/460,800 primary queries,24 null wrappers
including shadow work, and the incomplete 104 exposure of128–136 visits
and two completed optimizer boundaries. No checkpoint after 100 existed.
Archive those CSV/log/telemetry records separately before restoring the
working CSV to100 and restarting from that same checkpoint on GPU 2.
Dense1000 and Pair 200 had both completed before this move. The new GPU 2
attempt's first five accepted epochs averaged 50.19 seconds of train plus
validation; the revised remaining forecast was 76 minutes. Do not charge
discarded attempts as accepted model age or omit their actual work.

The paired e100 execution audit measures a real subset row reduction:
full-grid M3 removes 93,060 of1,998,768 rows(4.66%); M10 removes 20,680
of1,998,768(1.03%). Reconstructing source-union rectangles from saved
support and validity attributes every removed row to inactive SOURCE
columns. Every physically eligible admitted row remains; there are no
eligible closed pairs. Inactive receiver rows and MM self rows still
execute. This is validity packing, not learned pruning. Subset wrapper
latency remains worse than the same-weight dense-mask Tree path and the
trained Pair control. The intervention audit's zero row differences refer
to its normal dense executor and must not be generalized to subset packing.

## Completed matched 200 review

The actual 200 review stops Tree-F3204 and Pair-F3202 at their completed,
selected 200 checkpoints. Tree-L3103 remains 100; the separately requested
fresh Dense3101 completed 1000. All four exited successfully, retained only
the declared 100-epoch milestones/latest/best-field aliases and share the
same selected-training normalization. The 500 candidate cap is available
for a justified later matched stage; it is not automatically consumed.

On all 22, Tree-F200 fluid/material/surface temperature RMSE is
2.13424/2.28240/2.53879 versus Pair-F200 3.06548/3.51092/3.73971:
30.4%/35.0%/32.1% lower. Pair remains better in velocity and the ordinary
pressure-difference functional. Replacing trained Tree controls with
identity worsens four-case mean fluid-temperature error by75.9%; replacing
actions with a geometry-matched reference changes it only 0.119%.
This is partial added value and trained-control reliance, not proof of
unique grouping or physical causality. Every ordinary intervention retains
the dense executor rows. All 22 rebuilt permutation/unequal-refinement
physical and representation checks pass, with 64/64 detailed first-gradient
checks at200; the original 100 failures remain preserved.
The default QM locality probe excludes no donor. A bounded final QE
supplement instead tests actual learned exclusions on three representatives:
all five excluded module donors have exactly zero conditional control
Jacobians,17 admitted modules/576 environmental donors remain connected,
and excluded-source planner paths remain nonzero. Planning/memberships/
geometry are frozen only for this attribution; native eligible value
pairs remain fully admitted. Conditional control locality is distinct
from a sparse physical executor or physical causality.

The matched continuation does not deliver known-null response transfer.
All 22 mean u/v/p/omega increments worsen 100-to200 in both arms, while the
separately defined section-pressure increment improves. The measured 101
Pair calibration gives weighted null/native gradient ratio 4.81e-6; this
is a weak task force at that point, not a later-gradient guarantee or a
causal explanation. Selected cases contain no held different-heat
physical-reference family. Extending the unchanged objective would mainly
pursue field fitting without resolving the response question. A revised
response calibration/objective and any new reference solves are separate
future work. The declared final paired executor and bounded inverse
measurements completed at the saved 200 weights; no further training starts.
All 32 inverse trails completed 320 attempts with 931 forwards/368 VJPs,
including shared rank probes. Graph proposals have 50 proper-block attempts
and 30 full-joint fallbacks out of80; they do not improve held fit or call
efficiency over joint.23/32 trails leave selected-training heat support,
despite feasible nonnegative/fixed-total/local-radius saved allocations.
The completed continuation process allocations forecast about 80/17 minutes
for another 100 Tree/Pair epochs, or4.00/0.85 process hours for the remaining
300-to500 allowance. These are context-dependent estimates: Tree ran
alone onGPU 2, Pair beside Dense, and future contention was not measured.
The unspent 600 combined candidate epochs would add 90,000 visits/2,400
ordinary updates/92.16 million primary queries; none was launched.

Accepted campaign exposure totals 225,000 case visits,6,000 ordinary
optimizer updates and 230.4 million primary queries. Main training process
allocation sums 7.3181 hours; merging co-resident intervals gives 5.1410
GPU-slot hours (GPU 1 1.1206/GPU 2 4.0205), including failed/censored attempts
and the 109.49-second dense pause. Probes/evaluation/inverse have separate
scopes and active GPU compute hours remain unmeasured. Full receipts,
tails, per-M results and selected measured figures are in the
[completed development report](../reports/HONF_Tree_Faithfulness_Fixed25_Development_Report.md)
and ignored local campaign evidence.

## Resources and closeout

Use physical GPUs 1/2 and the ModularDT environment. Record actual devices,
contention, first-five-epoch forecasts and revisions. Co-residency requires
sufficient measured memory; preserve effective batch and complete selected
coverage. Do not touch GPU 0/external runs, start Wind training, expand data,
launch a formal run or add seeds/architectures. Allow at most two bounded
evidence-driven remedies per central failure and preserve their lineage.

Final reporting separates Predictor/Organizer/Inverse and A-added value,
B-faithful interpretation and C-response transfer. Retain at most six
inspected measured main figures, default PDF plus a small raster companion
for Markdown embedding, with quantitative descriptions immediately below.
Generated figures, arrays, checkpoints and one-time renderers remain local
in ignored paths. Commit/push durable source/tests/configuration/report only,
audit the entire outgoing history with the pre-push gate and verify branch
tips match. A launch or smoke check does not complete this experiment.
