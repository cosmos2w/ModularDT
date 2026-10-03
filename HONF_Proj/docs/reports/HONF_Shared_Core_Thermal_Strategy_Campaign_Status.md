# Shared-core Thermal strategy campaign: live status

The campaign trains a fresh native Dense control, a fresh three-term control,
and three learned organizers while preserving Thermal's predicted-port
P0/P1/P2 coupling and the frozen Stage-A local surrogate. Run1804 selected
e4738 remains an evaluation-only mature reference. Wind scientific training
is paused; Wind is used for shared-core compatibility checks.

**State: full-dataset screening and reviewed control continuation are running.**
Snapshot at 2026-10-03 04:24 UTC: both controls and overlap e100 completed;
H-tree completed e27; B-fine completed e387 on its way to 500 and H-local e78.
Work is on `agent/honf-core-next`. The initial
finite portfolio is Runs 2201–2205; two repaired screens are available only
when a concrete failure warrants them. No 5,000-epoch job is authorized to
start automatically.

| Run | Strategy | Complete epoch | Sampled development field / T MSE | Current action |
|---|---|---:|---|---|
| 2201 | B-native | 100 | 0.10121 / 0.06007 | Screen and physical panel complete |
| 2202 | B-fine | 387 | 0.04638 / 0.01840 | GPU1; exact e100→500 continuation |
| 2203 | H-tree | 27 | 1.31840 / 0.45976 | GPU1; reviewed exact e8 engineering resume |
| 2204 | H-overlap | 100 | 0.25209 / 0.25632 | Physical screen complete; 500 queued |
| 2205 | H-local | 78 | 0.25544 / 0.13666 | GPU2; fresh full-dataset screen |

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
- A later full-width H256 Wind check passed all three fresh architectures on
  real 3-D case `gen_0000_wd270` (M11/E512/Q4). All 98 physical first gradients
  matched bitwise, organizer gradients were nonzero, optimizer steps succeeded
  and physical buffers stayed unchanged. Each hard/soft P0 forward executed
  13,477 rows in five fine calls. This is native compatibility, not transfer
  accuracy or Wind scientific training.
- The focused CPU suite passed 107 tests (five optional native-resource tests
  skipped in that CPU invocation, executed separately with local resources).
  Six fixed-total heat-inference tests passed; a native Run1804 smoke completed
  real heat updates on two cases. This validates execution, not inverse quality.
- Physical environment cell lengths are adapter metadata independent of
  normalized source weights. Thermal uses sqrt(cell area); Wind uses cube-root
  token-cell volume in rotor-diameter coordinates. Near-access scales are not
  inferred from dimensionless attention measures.

B-fine allocates 140 trainable tensors (2,928,273 scalars), while 120 acquire
AdamW state. The other 20 tensors (104,451 scalars) belong to retained
`fallback_heads.internal_head` and `fallback_heads.interface_head`. Native
`local_surrogate` execution uses frozen Stage-A outputs and bypasses those
global fallback heads. Exact e200 optimizer mapping and a genuine native CPU
B2/Q17 gradient calculation agree on precisely these 20 inactive tensors;
all 120 active tensors have optimizer step 2,600. This is an allocated versus
executed case-branch distinction; legitimate fallback physics remains available.

The initial executor is a dense masked reference. Unique supported pairs,
allocated rows, executed rows, invalid/padded rows and calls are distinct.
Logical sparsity currently establishes no executor saving. Timing will be
reported with actual physical GPU 1/2 and external occupancy sampled per epoch.

Control training/validation medians are approximately 24.5/1.75 seconds for
B-native and 10.84/0.90 for B-fine. Peak allocated memory is about 4.53/4.15
GiB. GPUs 1/2 have no external GPU processes at this snapshot; host resources
are shared with an unrelated GPU 0 job. Both controls completed e100.
H-local's first ten epochs have median train/validation 46.16/1.28 seconds
and peak allocated memory 10,749 MiB. Its exact e25 full-state checkpoint is
finite, and e26 persisted calibration version 2 with all five M strata.
H-tree's first two complete
epochs took 455.6/458.0 seconds for training and 25.3/25.4 for validation,
with peak allocated memory about 11.0 GiB. Its first engineering repair batches
node reductions and replaces scalar host sorting reads with equivalent lists.
On actual GPU 1 micro8/Q1024, hard+soft wrapper median fell 4.108 to 2.534 s;
one complete native48 optimizer boundary fell 32.75 to 23.04 s. All 185 first
gradients and physical outputs passed declared tolerances. These bounded
measurements are not complete-epoch timings. Exact e2 optimizer/RNG is retained;
the stopped partial e3 (66/75 microbatches, 6:39 elapsed) is discarded and logged.
Five complete postresume e3–7 epochs average 308.99 s training and 14.86 s
validation, a 32.83% observed total-time reduction against e1–2. Actual hard
and soft executed rows remain dense.

A second reviewed repair batches typed source/control heads across actual
cases and propagates receiver-tree access by depth using fresh phase-owned
geometry. Isolated GPU2 at immutable e2/native micro8/Q1024 measured hard+soft
forward median 2.659→0.830 s and backward 1.132→0.189 s; one native48 optimizer
boundary fell 23.87→7.31 s. All 185 first gradients pass (max absolute 4.42e-9),
with physical output maximum 7.63e-6 across ports/fields/interfaces; allocated
memory was about 5.3 GiB in that bounded replay. Empty-source, fixed-topology
and physical gradient tests passed after correcting zero-width row padding.
Exact e8 / all 104 AdamW steps/RNG/schedule state is the engineering parent.
Five actual complete epochs after the second repair, e9–13, average 177.66 s
training and 8.74 s validation, or 186.40 s total. Every epoch has 600 cases,
75 microbatches, 13 native optimizer steps and 614,400 primary queries. Exact
e13 has all 185 active AdamW states at step 169 and four saved RNG streams.
These epochs share GPU1 with B-fine. At e13 they established a forecast of about
4.5 hours for e14–100, excluding later pressure/response changes.
They are not isolated timing or sparse executor savings.
The first ten resumed epochs e9–18 confirm this cost: mean train/validation
177.26/9.09 s, total 186.35 s, with the same full-epoch counters. The current
focused repair/evaluation/manual-launch suite passes 118 tests; one optional
native-resource test was skipped in that CPU suite. Actual native permission
isolation and checkpoint/gradient audits were executed separately.

The healthy B-fine e100 control is advancing to500 on GPU1 alongside H-tree,
within verified memory headroom (about 29.8/49.1 GiB combined process memory).
Their epochs after this launch are labelled **owned contention**; e8 tree
train/validation 411.81/36.48 s show a material effect. They are excluded from
isolated speed rankings. B-fine e101 visited all 600 records with 13 steps and
policy 2 active, plus one train-only response pair: 2 examples/4,324 role queries,
0.571 s response forward work. Five training-family gradient calibrations all
hit the conservative 0.1 coefficient cap; the e101–200 ramp is recorded rather
than claiming the nominal 5% gradient target was reached.

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

| Field | B-native e100 | B-fine e100 | H-overlap e100 | Mature Run1804 e4738 |
|---|---:|---:|---:|---:|
| u | 0.0698578 | 0.0738634 | 0.0908794 | 0.00590808 |
| v | 0.00511782 | 0.0106410 | 0.00976146 | 0.000337623 |
| p | 0.0252665 | 0.0302576 | 0.0358716 | 0.00210899 |
| omega | 0.217016 | 0.302935 | 0.303445 | 0.0258649 |
| temperature | 1.43558 | 1.67576 | 3.27004 | 0.187831 |

These benchmark physical-scale quantities each retain their own units; they
are not averaged into one physical scalar. B-fine is finite and improves
with training but still misses mature fidelity. Young-versus-mature accuracy
does not isolate architecture and does not stop the healthy ladder. Saved
evidence also includes near/far fluid, interfaces, material peaks and ports.

The immutable mature e4738 reference has also completed the full 90-case native
development population. Excluding duplicate0273, equal-case fluid u/T RMSE is
0.005721/0.218656 and module material-peak RMSE 0.346587; compatibility90 yields
0.005707/0.217622/0.344636. These full-population results are saved separately
from the 18-case screen and will anchor the 500/1000 comparisons. They retain
the same predicted ports, frozen Stage-A and previously exposed benchmark limits.

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
No mature checkpoint or unrelated data was removed. At this snapshot root has
about 75 GiB and data 120 GiB free; all new campaign checkpoint/evidence writes
remain on data. Changes in unrelated occupancy are not attributed to this campaign.

## Selected measured figures

Figure index: [full-epoch training progress](../../diagnostics/generated/shared_core_campaign_20261002/figures/campaign_training_progress.pdf),
[screen fields and residuals](../../diagnostics/generated/shared_core_campaign_20261002/figures/predictor_screen_case0692.pdf),
[frozen finite responses](../../diagnostics/generated/shared_core_campaign_20261002/figures/reference_response_baseline.pdf),
[mature inverse baseline](../../diagnostics/generated/shared_core_campaign_20261002/figures/mature_inverse_baseline.pdf),
[actual overlap graph and work](../../diagnostics/generated/shared_core_campaign_20261002/figures/overlap_e100_graph_work.pdf).

![Full-epoch training and measured elapsed work](../../diagnostics/generated/shared_core_campaign_20261002/figures/campaign_training_progress.png)

The inspected snapshot contains 100/387/27/100/78 complete epochs for
B-native/B-fine/H-tree/H-overlap/H-local. Every displayed epoch has 600 distinct
training cases, 13 optimizer steps and 614,400 primary sampled queries. B-fine
e387 normalized sampled field/T MSE is 0.04638/0.01840; this is a continuing
training trend, not a completed 500-epoch or full-grid physical result.
Successful train/validation time sums are 0.741/1.464/1.822/1.214/1.053 hours;
discarded partial work and storage replay overhead remain recorded separately.
Owned contention and H-tree engineering changes affect timing. These curves
establish actual model age and work, not sparse speedups or mature-equivalent
physical fidelity. The figure marks the H pressure ramp and common e101
response/native-denominator amendment without combining different field units.

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

H-overlap e100 is finite but worse than the same-age controls on fluid
temperature and surface/material errors: near/far T RMSE 3.4390/3.1789,
surface 3.4289, material 3.0983 and peak 3.5137. Its q_normal proxy RMSE 5.1202 is
slightly below B-fine 5.3558. Hard QE support retains 44.48% of eligible logical
pairs, while each hard/soft whole-epoch path still executes 253,186,272 rows
in 3,177 fine calls. This is a fidelity/support tradeoff with zero measured
training executor saving. The healthy curve continues to 500; H-local tests
the distinct near-access protection hypothesis from fresh initialization.

![Actual overlapping source groups and native executed work](../../diagnostics/generated/shared_core_campaign_20261002/figures/overlap_e100_graph_work.png)

The top panels show case0692's M10 environmental source locations and two
admitted QE groups at P0/P1/P2: 85 shared atoms in an 89-atom union. These are
learned source memberships, not physical causality. The bottom panel sums
directly recorded native accesses over four fixed development cases, all five
routes and all three phases. Normal selects 3,508,819 of 7,743,578 eligible
physical pairs, but every one of nine interventions executes 7,995,072 fine
rows in 644 calls, including padding. There is no measured executor saving.
Normal/full-access/recomputed-zero-control/rewired fluid T RMSE is
2.8432/3.5946/3.2341/3.2556; geometry yields 2.7734 but worsens surface T to
3.5618 versus normal 2.9758. Full access improves peak error to 2.6891 versus
normal 3.0238. Utility is mixed across fields and roles.

Independent audit finds that the old prediction `interaction_aux` averages
external field chunks and repeats preparation counts. Its saved scalar work
cannot establish complete-wrapper costs. Direct native phase recording and
training telemetry are the authorities for work here. Recording transparently
preserves all four normal physical outputs bitwise; instrumented elapsed times
include copies/anchor reconstruction and are excluded from speed claims.
The original zero-control intervention recomputes physical phases, so later
learned access changes through feedback. A separate reference-access replay
completed four native cases: all 644 actual P0/P1/P2 calls preserve normal
weight/support/edge access/near/catalogue/receivers/diagnostics bitwise, and
historical density reconstruction reproduces saved weights exactly. Controls
and actual gain/score projection biases are disabled; current physical source
values, local physics and ports are recomputed. Fixed-access zero controls
worsen mean fluid T from 2.8432 to 3.2344 and surface T from 2.9758 to 3.2295,
with a surface improvement in case0692. This isolates useful collective
control action on this small exposed panel, with mixed case effects. Geometry
and rewiring preserve some P0 membership multisets but fail whole-wrapper
degree/weight matching; they do not establish a globally matched control win.

Two reusable reference-action controls are now implemented and unit-tested
for the later checkpoint reviews. Full access retains the normal
source-resolved collective controls and projection biases. The bounded
geometry control reassigns complete density/weight/support/control tuples
within equal-measure source strata, preserving both binary graph degrees,
each receiver's tuple multiset, normalized row mass, invalid padding and all
positive near envelopes. Source-column weighted sums may change. Its budget
is 32 support switches, 32 active-tuple swaps and 512 donor-source-pair checks
per case/native access call; each check scans eligible partner rows. It is
neither a global geometry optimum nor a newly constructed shared-group
organizer. Reconstruction from twelve saved native QE scopes matches the
saved normal control probes and summaries bitwise. Physical inference for these two controls is pending the
500-epoch reviews; helper tests alone establish no fidelity gain.

Run1804 and both fresh e100 controls have completed finite-response evaluation
on all eight existing non-training families (11 absolute states per family).
Fields are decoded once and saved with physical increments, pressure and
per-module peaks; the zero-change comparator and reference magnitudes remain
separate. No response-relative quality is claimed without established floors.
The mature local-inverse job also encountered ENOSPC on its eighth case;
individual complete trials are retained and resume now skips verified arrays.
Atomic numerical/JSON writes preserve previous complete files on interruption.

![Frozen native response errors and physical response maps](../../diagnostics/generated/shared_core_campaign_20261002/figures/reference_response_baseline.png)

Across eight previously exposed non-training families, mean fluid-temperature
finite-response RMSE for mature/Dense100/Fine100 is 0.06610/0.11562/0.12484
on the 16 heat-transfer perturbations; zero change yields 0.24176. For the
same perturbations the stored velocity response is exactly zero, but the
models yield mean u-response RMSE 0.001231/0.003684/0.004631: all fail that
generator-specific null control. Across all 80 perturbations, mean pressure
increment error is 0.0001974/0.0003988/0.0006154 versus zero change 0.0002784.
The bottom maps show family0310 heat-transfer-plus on 7,918 full-stencil-common
fluid grid cells, with temperature response RMSE 0.07045 versus reference RMS
0.30041. The maps use common signed scales and an absolute residual scale;
no plotted finite cell is clipped. White masks exclude physical solids and
noncommon stencil support. These are physical benchmark units and frozen
predicted-port/Stage-A results, not CFD or response-relative validation.
The figure shows a useful thermal-response baseline alongside spurious
cross-field response and young-model pressure misses.

![Complete frozen mature-reference heat optimization and identifiability](../../diagnostics/generated/shared_core_campaign_20261002/figures/mature_inverse_baseline.png)

Run1804's frozen local heat-inference evaluation completed all 108 trials:
12 development cases, three starts, 30 projected-Adam steps, joint/graph/
size-matched ungrouped modes, six fitted and six disjoint held sensors.
The selected figure uses the completed single-thread CPU/canonical-slot
evaluation policy. Joint mean observed RMSE falls 3.4130 to 2.0458 (40.06%) and
held RMSE 3.7327 to 2.6432 (29.19%) in temperature units. Material-peak RMSE
falls 5.5096 to 4.5567 and pressure-difference absolute error 0.01433 to 0.009019.
Median CPU one-thread joint trial time is 23.14 seconds, excluding checkpoint
load and Jacobian precheck. All saved heat states are nonnegative; maximum
relative total drift is 7.68e-7. Graph and ungrouped modes both use full-joint
fallback on all 1,080 steps each, and all 72 fallback trial trajectories are
bitwise identical to their joint counterpart. They establish no graph utility.
Trajectories are nonmonotonic: held error worsens in 12/36 starts, observed
error in 10/36 and peak error in 14/36. This setting does not establish robust
convergence. Earlier two-thread trails remain preserved; repeated gradient
roundoff of order 4.77e-7 amplified along the projected optimizer. Single-thread
gradients and canonical block ordering remove that comparison artifact.
The measured fixed-total Jacobian ranks are 2/4/6/6 for M3/5/7/10. The three
M10 tasks have nine free directions and only six observations, so their
allocations are locally nonidentifiable. One M7 task has condition number
about 5,575 despite full numerical rank. Panel c's stored individual heat is
evaluation-only, and its difference from recovered allocations does not by
itself establish an incorrect design. Every proposed field, peak and pressure
comes from the same frozen checkpoint; no independent physical solve was
performed. The saved design trails contain sensors/peaks/pressure rather than
full postoptimization field grids. The revised PDF master and embedding raster
were visually inspected; superseded visual exports were replaced, while both
numerical evaluation histories remain retained.

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
