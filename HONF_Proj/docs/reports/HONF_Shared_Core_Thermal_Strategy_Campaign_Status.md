# Shared-core Thermal strategy campaign: live status

The campaign trains a fresh native Dense control, a fresh three-term control,
and three learned organizers while preserving Thermal's predicted-port
P0/P1/P2 coupling and the frozen Stage-A local surrogate. Run1804 selected
e4738 remains an evaluation-only mature reference. Wind scientific training
is paused; Wind is used for shared-core compatibility checks.

**State: full-dataset screening is running.** Snapshot at 2026-10-03 02:48 UTC:
both controls e100 and overlap e59 completed; H-tree resumed its exact e2.
Work is on `agent/honf-core-next`. The initial
finite portfolio is Runs 2201–2205; two repaired screens are available only
when a concrete failure warrants them. No 5,000-epoch job is authorized to
start automatically.

| Run | Strategy | Complete epoch | Sampled development field / T MSE | Current action |
|---|---|---:|---|---|
| 2201 | B-native | 100 | 0.10121 / 0.06007 | Screen and physical panel complete |
| 2202 | B-fine | 100 | 0.21442 / 0.07559 | Screen complete; physical panel saved |
| 2203 | H-tree | 2 | 1.89847 at e1 / pending | GPU 1; exact e2 engineering resume |
| 2204 | H-overlap | 59 | 0.35657 / 0.34376 | GPU 2; exact e55 storage recovery |
| 2205 | H-local | 0 | Pending | Native optimizer/gradient checks passed |

These MSEs use the maintained normalized, sampled 90-case validation task;
they are neither full-grid physical errors nor the final 89-case comparison.
Every completed control epoch has 600 unique cases, 75 microbatches, 13 native
buckets/optimizer steps and 614,400 primary field queries. Exact e25/e50
checkpoints are retained as reached.

All formal epochs must visit all 600 native training records exactly once;
batch caps are rejected. Effective batch 48, primary sampled Q 1024, FP32,
AdamW 3e-4/weight decay 1e-5, clipping 1, seed 0 are common. Microbatching preserves
case-weighted accumulation into the original native bucket steps. Epoch and
case/query/optimizer counters are saved separately. Schedule progress uses an
absolute horizon rather than resetting at the 100/500/1000 stage boundaries.
Training-family response pairs begin after 100 and their extra work is recorded.

## Evidence recorded so far

- Archived Run1804 configuration confirms 600/90 train/development records,
  batch 48/Q 1024, H256/message128/four heads, 192 environmental tokens and a
  frozen Stage-A model. Its historical mean train time is 16.58 seconds/epoch;
  this is a resource reference, not a forecast for the new shadow training.
- New shared reader tests cover full-access/control-identity B-fine output
  and physical first-gradient parity, 2-D/3-D construction, prepared uneven
  chunks and executed-row ledgers. Organizer tests cover task gradients,
  source masking/permutation, phase freshness and positive shadow restoration.
- Review repaired a double-zero initialization that would disconnect group
  control learning. Group controls start with identical small constant states;
  the last physical modulation projection remains zero, preserving identity.
  Real optimizer checks verify subsequent control learning.
- Three historical cover-ledger assertions also fail on the unchanged
  `0d1f57c` checkout: expected frontier degree and packed rows disagree with
  its native full-access dispatch. They are recorded as pre-existing failures;
  historical interaction equations were preserved.
- Native retained Thermal e4738 and Wind e2475 replay, prepared chunking and
  disposable optimizer steps passed at the declared 2e-5 tolerance. The frozen
  Stage-A tensors stayed unchanged. Fresh tree/overlap/local wrappers each
  completed real optimizer steps with nonzero organizer gradients.
- The focused CPU suite passed 107 tests (five optional native-resource tests
  skipped in that CPU invocation, executed separately with local resources).
  Six fixed-total heat-inference tests passed; a native Run1804 smoke completed
  real heat updates on two cases. This validates execution, not inverse quality.
- Physical environment cell lengths are adapter metadata independent of
  normalized source weights. Thermal uses sqrt(cell area); Wind uses cube-root
  token-cell volume in rotor-diameter coordinates. Near-access scales are not
  inferred from dimensionless attention measures.

The initial executor is a dense masked reference. Unique supported pairs,
allocated rows, executed rows, invalid/padded rows and calls are distinct.
Logical sparsity currently establishes no executor saving. Timing will be
reported with actual physical GPU 1/2 and external occupancy sampled per epoch.

Control training/validation medians are approximately 24.5/1.75 seconds for
B-native and 10.84/0.90 for B-fine. Peak allocated memory is about 4.53/4.15
GiB. GPUs 1/2 have no external GPU processes at this snapshot; host resources
are shared with an unrelated GPU 0 job. Both controls completed e100.
H-local forecasts await actual complete epochs. H-tree's first two complete
epochs took 455.6/458.0 seconds for training and 25.3/25.4 for validation,
with peak allocated memory about 11.0 GiB. Its first engineering repair batches
node reductions and replaces scalar host sorting reads with equivalent lists.
On actual GPU 1 micro8/Q1024, hard+soft wrapper median fell 4.108 to 2.534 s;
one complete native48 optimizer boundary fell 32.75 to 23.04 s. All 185 first
gradients and physical outputs passed declared tolerances. These bounded
measurements are not complete-epoch timings. Exact e2 optimizer/RNG is retained;
the stopped partial e3 (66/75 microbatches, 6:39 elapsed) is discarded and logged.
Further equivalent batching is being investigated while training progresses.

H-overlap's first seven epochs measure median training/validation 42.58/1.20
seconds and peak allocated memory 10.26 GiB. Its initial e100 screen forecast
is about 73 minutes, plus periodic save/plot overhead; training to e1000 is
about 12.2 GPU-hours at that rate before response work. Actual hard and soft
forward work is recorded separately by P0/P1/P2. At e7 each path executes
254,866,992 fine rows in 3,201 calls, with 225 preparations and 300 wrapper
read calls. Backward checkpoint recomputation is outside this ledger scope.
Dense controls lack this typed ledger; primary query counts do not replace it.

## First complete physical screen and pre-pressure repair

B-native exact e100, B-fine exact e100 and Run1804 selected e4738 were evaluated on the same
18 input-selected development cases at their full native 64x128 grids.
Case IDs are preserved for subsequent screens. Equal-case fluid RMSE is:

| Field | B-native e100 | B-fine e100 | Mature Run1804 e4738 |
|---|---:|---:|---:|
| u | 0.0698578 | 0.0738634 | 0.00590808 |
| v | 0.00511782 | 0.0106410 | 0.000337623 |
| p | 0.0252665 | 0.0302576 | 0.00210899 |
| omega | 0.217016 | 0.302935 | 0.0258649 |
| temperature | 1.43558 | 1.67576 | 0.187831 |

These benchmark physical-scale quantities each retain their own units; they
are not averaged into one physical scalar. B-fine is finite and improves
with training but still misses mature fidelity. Young-versus-mature accuracy
does not isolate architecture and does not stop the healthy ladder. Saved
evidence also includes near/far fluid, interfaces, material peaks and ports.

Review found the original structural calibration's first five microbatches
could belong to the same M bucket. Before e26 pressure, calibration policy 2
collects one sample from each of five training-M strata and saves actual M,
task/cost gradient norms and bounded coefficients. Pressure remains zero
until all available strata are represented; absent task signal yields zero
scale. H-overlap resumes from its exact e25 optimizer/RNG checkpoint; any
computation begun after that save is discarded and recorded. H-tree/H-local
start with the repaired calibration. This fixes the originally intended
calibration scope before any pressure and consumes no additional screen.
No development metric sets the coefficient. The stop occurred immediately
after the atomic e25 checkpoint save, before e26 computation began. Resumed
e26 sampled all five module-count strata and persisted their finite task/cost
norms; average structural pressure was 7.47e-6 during that calibration epoch.

Native mixed-M review found that case-weighted microbatch means changed the
original query-mass, active-module and valid-port denominators. On an actual
48-case M4/5/6 batch at Q17, the initial policy changed the physical gradient
by 5.12% in L2. Correct denominator fractions recover the full native loss
and gradient to 9.7e-9 and 2.66e-7 relative error in the implementation-level
replay. All five initial screens retain the same original averaging through
e100. All ten e500/e1000 profiles adopt physical-loss policy 2 from e101;
resume permits only this explicit two-key amendment at exact e100. Optimizer,
RNG, calibration and absolute schedules remain preserved. This is a common
training-policy correction, not a new architecture or extra repaired screen.

## Storage failure and measured recovery

The root filesystem reached zero available bytes during overlap e59 save.
Atomic checkpoint replacement preserved valid e55 model/optimizer/RNG states;
e56–59 metrics and the partial temporary file were archived before rolling
authoritative history back to e55. All four replayed epochs match seven
recorded primary/structural columns exactly, and e59 checkpoint writing now
succeeds. The repeated four epochs are extra resource work, not extra model
age. GPU 2 was released during storage repair; its interruption is retained.

Fresh Runs 2201–2204 now reside under
`/data/wanglz/ModularDT/shared_core_campaign_20261002/HONF_Forward_Runs`, and
campaign evidence under that data directory's `generated_evidence`.
Compatibility symlinks preserve all original workspace paths. Every relocated
regular file was byte-compared before replacement. New H-local output uses a
validated copy of its unchanged scientific profile with data placement only.
No mature checkpoint or unrelated data was removed. Root retains about 1.5 GiB
free; new campaign checkpoint/evidence writes use data's approximately 121 GiB.

## Selected measured figure

Figure index: [screen fields and residuals, case 0692](../../diagnostics/generated/shared_core_campaign_20261002/figures/predictor_screen_case0692.pdf).

![Native-grid screening fields and residuals for development case 0692](../../diagnostics/generated/shared_core_campaign_20261002/figures/predictor_screen_case0692.png)

On this input-selected high-M case (M=10), B-fine e100 fluid RMSE is 0.089684
for u and 2.07438 for temperature; mature Run1804 e4738 yields 0.0056393 and
0.174727. The figure uses the stored exposed-development native-grid reference,
benchmark physical units, shared value/residual ranges, white solid masks,
predicted ports and the same frozen Stage-A model. It shows substantial young
model errors near modules and in their downstream fields. It does not isolate
architecture from training age. Both the PDF master and small embedding raster
were visually inspected after fixing clipped labels; numerical arrays remain
under the ignored evaluation directory.

Run1804's full frozen local heat-inference evaluation is running on CPU:
12 development cases, three starts, 30 steps, joint/graph/size-matched ungrouped
updates. Hidden allocations are evaluation-only and no independent physical
solver is launched. Completed trails are saved individually; startup is not
an inverse-quality result.

Run1804 and both fresh e100 controls have completed finite-response evaluation
on all eight existing non-training families (11 absolute states per family).
Fields are decoded once and saved with physical increments, pressure and
per-module peaks; the zero-change comparator and reference magnitudes remain
separate. No response-relative quality is claimed without established floors.
The mature local-inverse job also encountered ENOSPC on its eighth case;
individual complete trials are retained and resume now skips verified arrays.
Atomic numerical/JSON writes preserve previous complete files on interruption.

## Remaining ladder and delivery

Screen all five arms at 100 complete epochs. Continue both controls and normally
all three organizers to 500. Select 2–3 new organizer versions to finish 1000,
with matched controls at 1000. Evaluate canonical 89-case development aggregates
excluding 0273 and 90-case compatibility results; neither is a fresh test set.
Save physical fields/residuals, group interventions, work/timing, finite
responses and frozen fixed-total heat-inference trails. Inspect the retained
figures and embed the selected results in the final English Predictor,
Organizer and Inverse report. Prepare and test continuation/fresh 5000 recipes
for 1–2 informative finalists without starting those runs.

Checkpoints, numerical arrays, generated figures and one-time renderers stay
in ignored local paths. Durable implementation/configuration/tests/reports and
launch recipes are committed and pushed after the outgoing-history audit and
the repository pre-push artifact gate.
