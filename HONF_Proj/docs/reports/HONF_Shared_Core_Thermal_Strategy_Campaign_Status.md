# Shared-core Thermal strategy campaign: live status

The campaign trains a fresh native Dense control, a fresh three-term control,
and three learned organizers while preserving Thermal's predicted-port
P0/P1/P2 coupling and the frozen Stage-A local surrogate. Run1804 selected
e4738 remains an evaluation-only mature reference. Wind scientific training
is paused; Wind is used for shared-core compatibility checks.

| Result | Measured gain so far | Measured miss / next decision |
|---|---|---|
| Predictor | Fine's selected1000-stage fluid T/u errors improve 21.24%/23.41% over its selected500-stage version. Native and Fine retain the legitimate predicted-port physics. | Fine's canonical89 T RMSE 0.56269 remains above mature 0.21866; heat-flux proxy, h and pressure tradeoffs remain. Complete the matched Native and H1000 comparisons. |
| Organizer | Native Tree/Overlap/Local100 subset execution passes output/first-gradient parity. Overlap's selected292 stage500 graph has measurable utility: control-identity and full-access/fixed-control interventions each worsen 23 of 24 mean physical roles on the four-case panel. | Selected292 full-query subset execution reduces fine-reader rows by 4.70–11.77%, but complete-wrapper latency increases 19.69–23.37%. Geometry controls have mixed gains and misses; graph utility does not establish adequate absolute fidelity. Finalist evaluation remains pending. Training uses the dense masked reference. |
| Inverse | Fine972's matched frozen observed/held means decline 35.66%/20.39% over thirty steps. | Heat RMSE rises 18.58% and endpoint-pressure error 69.54%; all graph/ungrouped updates are full-joint fallback. Finalist local inverse and qualified paired heads remain pending. |

**State: all five e100 screens, H-overlap's e500 review and B-fine's e1000 control are complete; the H finalist ladder continues.**
Saved training snapshot at 2026-10-03 13:51 UTC: B-fine completed e1000;
B-native is at e944, H-tree e300, H-overlap e500 and H-local e486. The table
and inspected progress figure use this same snapshot. Tree completed its exact
e100 physical and eight-family response screens and resumed to e500. Fine's
immutable exact1000 and field-selected e972 have completed full90/canonical89
physical evaluation, all eight response families and the matched 108-trial
frozen inverse evaluation. Both earlier e493 selections remain preserved. Native subsequently
completed exact e500; its immutable exact500/field-selected456 physical review
is complete before continuation and migration. A completed control does not
count toward the required two or three new H finalists. Overlap exited naturally
at e500 at13:10 UTC. Its exact500 and field-selected actual292 have completed
full90/canonical89 fields and eight-family responses; selected292 also has the
ordinary and strict graph controls. The separate T-selected checkpoint is
actual269. CPU topology and 36 one-step inverse trials per evaluated version
establish execution/readiness, not final inverse quality. Full-query native
CUDA parity/timing passes for both versions show row reductions and slower
complete-wrapper latency. The measured stage500 assessment appears below.
After this plotted snapshot, Local exited naturally at500 at14:06:27 UTC,
with all20 full-state audit checks passing. Its immutable exact500 and shared
field/T-selected actual388 are preserved. Field/response/graph evaluation is
the next step; sampled late drift alone does not decide promotion.
At the same 13:51 snapshot, the last twenty complete-epoch median/p90 elapsed
seconds are Native34.15/36.03, Tree103.54/105.52 and Local66.23/74.18.
Extrapolating those measured rates gives conditional next-stage finish ranges
of approximately Native1000 at14:23–14:25 UTC, Tree500 at19:36–19:43 and
Local500 at14:07–14:09. These ranges compare
median/p90 rate extrapolations under current placement, not confidence bounds;
evaluation/setup and subsequent contention changes are excluded. Full H500
physical/graph review precedes promotion; the two or three H1000 results remain
required and incomplete.
Work is on `agent/honf-core-next`. The initial
finite portfolio is Runs 2201–2205; two repaired screens are available only
when a concrete failure warrants them. No 5,000-epoch job is authorized to
start automatically.

| Run | Strategy | Complete epoch at snapshot | Sampled development field / T MSE | Current action |
|---|---|---:|---|---|
| 2201 | B-native | 944 | 0.00811 / 0.00672 | Exact500 and physical review complete; GPU2 continuation to1000 |
| 2202 | B-fine | 1000 | 0.01175 / 0.01182 | Native exit0; immutable exact1000/selected972 evaluation complete |
| 2203 | H-tree | 300 | 0.05410 / 0.05986 | GPU1; exact e100→500 continuation; monitor response-active trend |
| 2204 | H-overlap | 500 | 0.60332 / 0.21109 | Native exit0; exact500/selected292 review complete; third-slot decision awaits cohort review |
| 2205 | H-local | 486 | 0.14052 / 0.13113 | GPU2; exact e100→500 continuation; recent sampled uptick retained |

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
reported only after native executor parity passes. The benchmark now has an
optional fail-before-timing gate covering low/high M, small/full query panels,
all actual P0/P1/P2 outputs and prepared/read tensors, parameter/physical-heat/
query first gradients, and bit-identical applied support/eligibility masks.
It restores CPU/CUDA RNG, native parameter flags, gradients, hooks and execution
mode, and checks frozen parameters/persistent buffers without optimization.
Output tolerances are rtol/atol 2e-5/2e-5; first-gradient tolerances are
2e-5/1e-6. These are separately declared contracts. On native Tree100 CPU1,
both M3/M10 small-query probes pass, with 183 active parameter gradients and
two input gradients, all fifteen route/phase masks identical, and unchanged
frozen state. The preserved stricter 1e-6 output probe fails only the interface
output, at maximum absolute error 7.24e-6; dense repeat is bitwise identical.
Whole-wrapper FP64 casting is unsupported by native FP32 encoding, so its
failed probe is retained separately from passing isolated fine-reader FP64
tests. The subsequent native GPU2 test completes all four low/high-M and
small/full-query parity checks before timing. Overall gradient relative L2
errors are 7.74e-7/6.72e-7 for M3 and 3.16e-7/2.33e-7 for M10, respectively;
all outputs, first gradients, route/phase masks and frozen state pass their
declared contracts. Immutable checkpoint SHA remains unchanged. Five repeats
after two warmups measure wrapper and prepared decoding separately. Subset
execution reduces full-grid fine rows from 1,998,768 to 458,596/528,548 for
M3/M10, but complete-wrapper medians increase 1.030→1.381 and 1.145→1.297 seconds.
GPU2 is shared with the two H trainers, with dense timed before subset during
08:59:48–09:01:23 UTC. These observed contended samples show executable row
reduction and no latency gain; they do not establish an isolated speed ranking.
Training retains the dense masked executor. Final selected versions still
require their own parity and timing evaluation. Focused
measurement tests pass 78 checks with three optional native-resource skips;
actual Native500 field/response execution separately validates the new direct
row recorder. Fine-work hooks stay outside benchmark timing and do not count
backward, attention or coarse/local work.

Subsequent native Overlap100 and Local100 checks run on physical GPU1/logical
cuda:0 alongside the unchanged Tree trainer, with one CPU thread, Q14/Q8192,
M3 case0274/M10 case0692, receiver chunk128, two warmups and five repeats.
All four output/first-gradient parity panels pass per arm at the same declared
tolerances. Each includes 203 defined parameter gradients, caller heat/query
gradients and all five typed routes across P0/P1/P2, including 35/161 actual
access inventories at Q14/Q8192. Overall gradient
relative L2 errors for M3 small/full and M10 small/full are
3.55e-7/3.70e-7/2.48e-7/1.86e-7 for Overlap and
1.59e-7/1.60e-7/6.73e-7/5.97e-7 for Local. All 334 model state tensors
(3,994,148 scalars) remain bitwise unchanged in each check, and exact100
checkpoint SHA/stat remain unchanged. These checks use clean source `bd8c5e6`.
Independent saved-output review reconciles all 32 timing rows' five-sample
min/median/p90 and native hook counts against the phase ledgers exactly. It
recomputes saved gradient norm aggregates; raw gradients were not retained for
a separate elementwise rerun, so the elementwise pass is the original native gate.

| Exact100 arm / full-Q case | Complete-wrapper actual fine rows, Dense→subset | Calls | Complete-wrapper median seconds, Dense→subset | Prepared P2 median seconds, Dense→subset |
|---|---:|---:|---:|---:|
| Overlap / 0274 M3 | 1,998,768→924,772 (−53.73%) | 161→164 | 0.56315→0.71038 (+26.15%) | 0.43915→0.48298 |
| Overlap / 0692 M10 | 1,998,768→933,200 (−53.31%) | 161→164 | 0.60496→0.67731 (+11.96%) | 0.45107→0.48804 |
| Local / 0274 M3 | 1,998,768→1,905,708 (−4.66%) | 161→164 | 0.66856→0.82452 (+23.33%) | 0.48820→0.55732 |
| Local / 0692 M10 | 1,998,768→1,978,088 (−1.03%) | 161→164 | 0.66546→0.76676 (+15.22%) | 0.50417→0.59672 |

Native hooks measure successful five-route fine-MLP input rows including
rectangle padding. Q counts caller field queries; complete-wrapper counts
include auxiliary port/coupling reads. Prepared and complete timings are
independent measurements. Local's M10 prepared Q14 median improves
0.01151→0.00954 seconds, while its full-Q wrapper remains slower; this small
scope does not establish an overall speedup. Dense is timed before subset in
both checks. The earlier Tree test used GPU2 with two H trainers, so the three
arms are separate execution checks, not an isolated architecture speed ranking.
Final selected versions require their own measurements, and all forward
training continues with the dense masked reference.

Local benchmark PID1067864 runs 11:17:59.878–11:18:58.412 UTC; Overlap PID1069234
runs 11:20:59.201–11:21:50.601 UTC. Tree's epoch214 `gpu_before` directly
contains the Local PID; that epoch takes 115.10/4.28 seconds train/validation.
The Overlap PID is absent at Tree's sampled boundaries. Loaded training records
have no absolute epoch timestamps, so its within-epoch intersection cannot be
reconstructed. Known benchmark intervals preclude interval-wide isolation;
boundary absence and adjacent timings do not establish no contention or causal
slowdown. Exact intervals and Tree epochs212–218 process lists/times are retained
in `Run2203/gpu1_benchmark_contention_20261003.json`.

Initial complete-epoch timing is
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

The healthy B-fine e100 control advanced to 500 on GPU1 alongside H-tree,
within verified memory headroom. Their epochs after this launch are labelled
**owned contention**; e8 tree
train/validation 411.81/36.48 s show a material effect. They are excluded from
isolated speed rankings. B-fine e101 visited all 600 records with 13 steps and
policy 2 active, plus one train-only response pair: 2 examples/4,324 role queries,
0.571 s response forward work. Five training-family gradient calibrations all
hit the conservative 0.1 coefficient cap; the e101–200 ramp is recorded rather
than claiming the nominal 5% gradient target was reached.

B-fine completed all 500 full epochs: 300,000 case visits, 307.2 million primary
queries and 6,500 optimizer steps. Its exact e500 checkpoint has 120 active
AdamW states at step 6,500, all four RNG streams, response calibration, policy 2
and the unchanged 5,000-epoch schedule horizon. Both predeclared field and
temperature selections within this stage actually select e493; immutable copies,
hashes and the original selection metrics are retained. Exact e500 and selected
e493 are evaluated separately throughout; selection is not relabelled as e500.

After B-fine finished, GPU1 started B-native's strict e100→500 continuation.
Its first ten response-active epochs average 28.73/2.15 seconds train/validation
with 5,338 MiB maximum allocated memory. Exact e125 has all 181 active optimizer
states at step 1,625 and preserved RNG/calibration/schedule state. H-tree remains
on GPU1; recent complete epoch times around 160 seconds improve the forecast
after the control handoff, but executor work remains unchanged. At 05:16 UTC
GPU1 reports about 34.6 GiB used with 12.8 GiB free.

H-overlap and H-local both continued from audited exact e100 states to 500 on
GPU2, after their complete physical screens and memory checks. At 05:16 UTC
GPU2 reports about 27.7 GiB used with 19.6 GiB free. Overlap's recent
ten complete epochs have median total time 60.06 seconds; Local's first five
response-active epochs have median 66.08 seconds. These are **owned contention**
measurements, not isolated architecture rankings. Around e112/e105 these rates
forecast roughly 6.4/7.3 more hours to 500, conditional on unchanged contention.
A disposable CUDA projection prototype additionally occupied GPU2 during
05:07:08–05:07:33 UTC; affected epoch intervals are recorded. It passed numerical
parity but gave no consistent GPU timing gain, so it was not adopted and no
trainer was restarted. Overlap's exact e125 audit passes finite model/optimizer,
unchanged frozen Stage-A, all 600 cases per epoch and preserved structural
calibration; new response runtime state is separately identified. Its validation
field/T window means improve 0.27943/0.24426 at e101–110 to 0.24836/0.23420 at
e116–125, while response loss rises 0.01953→0.02065. This establishes no response
improvement.

Fine's exact500→1000 continuation joined GPU2 at 05:44:25 UTC. Its exact600
audit passes all 600 full epochs, 120 active AdamW states at step7,800, four
RNG streams, unchanged frozen Stage-A and response calibration, policy2 and
horizon5000. The immutable stage500 selections remain unchanged. Window
means from e501–510 to e591–600 improve normalized field/T MSE
0.03634/0.01648→0.02030/0.01283, while response loss rises
0.000236→0.000910. These are sampled development/training measurements,
not a new full-grid physical or response evaluation.

The placement remains healthy but costlier for the H jobs. In a saved
06:02 UTC three-job sample, Overlap e162–171 and Local e152–159 have median
complete-epoch times 91.88/105.02 seconds, compared with their earlier two-job
59.39/66.08 seconds; Fine e551–560 takes 17.74 seconds. These are measured
owned-contention regimes, not isolated speed rankings or a demonstrated
placement speedup. Fine's observed CUDA reservation is about 12.4 GiB,
exceeding the 8-GiB admission allowance by about 4.1 GiB. GPU2 retains about
7.5 GiB free with no OOM, and no additional job is admitted. GPU0's unrelated
TurbulentCombustion job is recorded without intervention. At the 06:14
snapshot, unchanged recent rates imply about 2.3 hours to Native500,
1.5 to Tree100, 2.0 to Fine1000 and 8.2/9.7 to Overlap/Local500; the latter
forecasts will change when Fine finishes. Successful-epoch elapsed sums
exclude discarded/replayed work and are concurrent job-hours, not exclusive
physical-GPU occupancy hours.

Overlap's exact175 full-state audit also passes. Its e101–110→e166–175 means
worsen field 0.27943→0.32164 and T 0.24426→0.28506 while structural cost
falls 0.77450→0.66480; response loss rises 0.01953→0.03545. Lower structural
cost establishes neither physical improvement nor saved executor work, and
the concurrent response ramp prevents attributing the change to structural
pressure alone. The healthy finite run continues to its authorized500 review.

A later saved-only review finds sustained deterioration across e291–310,
e311–330 and e331–350. Mean sampled validation field MSE rises
0.245636→0.391911→0.602973 (+145.5% first to third); train field rises
0.354875→0.403979→0.428045 (+20.6%). Validation T rises
0.198954→0.236643→0.242740 (+22.0%), with worsening internal/interface/port
losses. Structural cost falls 0.573508→0.552927→0.538138; response loss is
nonmonotonic at 0.051023/0.059319/0.049371. All sixty epochs retain finite
600-case training and 90-case validation coverage, constant learning rate
3e-4 and structure/response coefficients 0.001/0.1. Gradient-norm/clip-scale
columns have no finite observations, so clipping saturation is unmeasured.
This is a predictor miss despite reduced structural cost. The planned e500
full-grid and same-weight access/control review will inform finalist selection
or a versioned repair; the metrics alone do not identify its cause.

Local's exact e125 audit passes the same full-state and frozen-physics checks.
Its e101–110 to e116–125 window means change field 0.19911→0.19627,
temperature 0.09367→0.09769 and response loss 0.008490→0.007889. Thus early
response improvement coexists with slightly worse temperature; the organizing
strategy remains under its authorized 500-epoch review.

Local's exact e175 audit retains complete coverage, optimizer/RNG, unchanged
Stage-A and original structural calibration. Its e101–110→e166–175 field/T
means improve 0.19911/0.09367→0.17058/0.07055, while response loss worsens
0.00849→0.01602; recent complete epochs take a median 102.60 seconds under
three-job contention. Fine's read-only observer later exits with code143, but its actual
CUDA trainer/tmux remain alive with uninterrupted completed epochs; the
observer event is recorded separately and compact milestone reads continue.

Tree's exact e75 audit passes all 600-case epochs, 185 active AdamW states at
step 975, four RNG streams and bitwise equality of all 109 frozen Stage-A
tensors against the engineering e8 parent. Calibration v2 and all five strata
are unchanged; selection75/horizon5000 and initial physical policy1 remain
correct. Hard and separately measured soft each execute 253,186,272 fine
rows in 3,177 calls over P0/P1/P2, with no skipped eligible rows. Native's
exact e300 audit likewise passes complete epochs, 181 optimizer states at
step 3,900, RNG, frozen Stage-A, policy2 and five response calibration samples.
Its 200 response-active epochs contain 400 wrapper calls and 1,285,156 extra
role queries. These confirm continuity and work; they do not replace Tree's
remaining e100 screen or Native's e500 physical review.

Tree's subsequent exact e100 audit passes all 100 full epochs, 185 active
AdamW states at step 1,300 and four RNG streams. All 316 model-state tensors
are finite, with native config, frozen Stage-A and five-stratum structural
calibration preserved. Read-only stage100 copies retain exact e100, field-best
actual e93 (selector 0.261186) and T-best actual e100 (0.203036). Root reviewed
the finite 18-case physical screen and authorized continuation on GPU1 using
the exact top-level e100 state, common policy2/101 amendment and unchanged
5000 schedule. Source `9eaf2a4` was clean at launch; the existing run identity is
retained. Last ten pre-response epochs took 153.66/7.25 seconds train/val
under owned Native contention. The first genuine e101 completes all 600
cases/13 steps, with 185 optimizer states at step 1,313, RNG, unchanged Stage-A
and structural calibration. Measured train/validation is 153.87/7.43 seconds;
peak allocated memory 11,270.5 MiB. Its baseline+variant response uses 4,324
extra queries/four actual wrappers (two hard and two soft), raw loss 0.008034,
2.639 seconds, capped calibration 0.1 and initial ramp coefficient 0.001.
Hard and soft response work are recorded separately; these are completion
measurements. The first ten complete response-active epochs 101–110 measure
mean train/validation 153.83/7.33 seconds under owned Native contention. The
saved selected e110 retains all 185 active optimizer states at step 1,430, RNG,
unchanged Stage-A and structural calibration, selection epoch 110/horizon 5000 and
five completed response calibration samples. These ten epochs add forty
actual wrapper calls/twenty examples/59,406 queries. At this placement rate,
390 remaining epochs imply about 17.46 hours; changed placement requires a new
measured forecast. A subsequent sampled e113 validation spike is retained and
monitored; a single endpoint does not justify changing the training policy.

Tree's exact e125 audit retains 185 active optimizer states all at step 1,625,
four RNG streams, unchanged native configuration and all 109 frozen Stage-A
tensors. Structural calibration remains bitwise equal to e100; all five
response calibration samples and saved scales remain equal to e105. The
e111–120 field/T validation means worsen by 37.1%/32.3% relative to e101–110,
but the next ten complete epochs e118–127 recover to 0.26191/0.17374. The
retained curve does not show monotonic collapse, so no new loss revision is
introduced. All ten have 600 cases, 75 microbatches, 13 optimizer steps and
614,400 primary queries. After Native exits GPU1, their mean train/validation
time is 93.91/4.20 seconds. Only Tree appears in both recorded GPU1 boundary
samples for each epoch; the shared host and GPU2 remain busy. The 373 remaining
epochs to500 forecast 10.17 hours at this placement, superseding the earlier
owned-Native timing forecast without claiming interval-wide GPU isolation.

The subsequent exact150 audit retains all 185 optimizer states at step 1,950,
four RNG streams, unchanged model/training configuration, all frozen Stage-A
tensors and the complete calibration lineage. Consecutive e111–130/e131–150
field means decline 0.302244→0.227255 and T means 0.220299→0.166455, improvements
of 24.81%/24.44%. Structural cost increases 0.387969→0.400625 at unchanged
weight 0.001; eighty response wrappers/129,654 auxiliary queries are counted
in the latter twenty epochs. Complete finite coverage is retained throughout.
Mean train/validation 96.03/4.28 seconds forecast 9.75 hours for the remaining
350 epochs, under recorded GPU1 boundary occupancy with only Tree and a busy
shared host. This measured window supports continued training; its noisy
single endpoints do not define physical fidelity or a sparse speedup.

Tree's ordinary exact e200 audit preserves all 185 optimizer states at step2600,
all 555 slot tensors finite, four RNG streams, selection epoch200/horizon5000,
the complete configuration/calibration lineage and all 109 frozen Stage-A
tensors. Every checked epoch visits all 600 cases with 75 microbatches,
13 updates and 614,400 primary queries. Consecutive twenty-epoch field means
for 141–160/161–180/181–200 decline 0.193227→0.155170→0.137368; T means decline
0.132232→0.117940→0.096902. The latest window improves field/T by 11.47%/17.84%,
while structural cost falls 1.74% to 0.393687 and the response coefficient
reaches 0.1 at e200. Recent ten-epoch train/validation means are 100.22/4.41 seconds,
with maximum allocated peak 13,194.7 MiB and explicit GPU1 free 33,701 MiB.
These sampled learning trends justify the ongoing e500 review; they do not
replace full-grid physical or graph-utility measurements.

Native subsequently completes all 500 genuine epochs with 181 active AdamW
states at step 6,500, four RNG streams, unchanged frozen Stage-A, normalization,
native configuration, five response calibration samples and horizon 5000.
Immutable exact500, field-selected actual456 and T-selected actual492 snapshots
are byte-verified and read-only. The exact500 checkpoint remains the optimization
parent for continuation; the T selection supplies no role in field-selected
comparisons.

Native's exact500→1000 continuation then starts on physical GPU2 at
09:03:53 UTC from clean pushed source `6648c86`, preserving Run2201 and its
immutable stage500 selections. The same saved one-device CUDA RNG maps to
logical cuda:0 on GPU2; future kernel results under changed placement are not
claimed bitwise equal. First e501 visits all 600 cases/75 microbatches/13 steps
and 614,400 primary queries, with field/T MSE 0.0136621/0.00779840 and peak
allocated memory 4,659.8 MiB. The first audited post-migration selected e504
has all 181 optimizer states at step 6,552, all four RNG streams, unchanged
native model configuration, frozen Stage-A and response calibration. Recursive
comparison of the entire saved native training configuration finds only
training.epochs 500→1000 changed. Organizer selection metadata is legitimately
empty for Dense; the horizon5000 is read from its explicit campaign settings.
Exact525 and exact550 ordinary audits then retain all 181 active states at
steps 6,825/7,150, all four RNG streams, frozen Stage-A and complete
configuration/calibration lineage; every intervening epoch has finite full
600-case coverage. The latter ten epochs average 35.32/2.94 seconds for
train/validation under recorded GPU2 owned contention.
By exact e650, all 181 states reach step8450, all 543 slot tensors remain finite,
four RNG streams and all 109 frozen Stage-A tensors remain intact, and the
complete configuration/calibration lineage is unchanged. Full 600-case
coverage persists; the latest ten epochs average 37.02/2.85 seconds for
train/validation under the same three-trainer placement, with explicit GPU2
free 6,990 MiB. Exact sampled field/T MSE 0.0099463/0.0066508 is an endpoint,
not the completed e1000 physical comparison.
Exact750 subsequently passes with all 181 optimizer states at step9750,
543 finite slot tensors, four RNG streams, unchanged full configuration/
calibration and all 109 frozen Stage-A tensors. Its sampled field/T MSE is
0.00733148/0.00425690; the recent ten-epoch means are 0.0116500/0.0131448,
with train/validation averages 36.54/2.97 seconds and maximum sampled epoch
allocated peak 4,984.7 MiB. Explicit GPU2 free remains 6,990 MiB.
Tree's trainer remains on GPU1, with its two short benchmark-sharing intervals
recorded separately. Native now shares GPU2 with Overlap and Local, so
sustained three-trainer timing is measured separately. The observed 12,790 MiB
Native nvidia-smi footprint and 7,106 MiB device free memory are driver-level
measurements, not separate measurements of allocator cache or reserved memory.

The first ten resumed Native epochs e501–510 retain complete finite coverage,
with mean train/validation 35.13/2.81 seconds and maximum allocated peak
5,140.8 MiB. They add twenty response wrapper calls/59,406 auxiliary queries
at the unchanged coefficient 0.1. At this measured three-trainer rate, the
490 remaining epochs forecast 5.16 hours, conditional on unchanged placement.
An audit found that historical telemetry's total-minus-used included separately
reserved driver memory. Those derived values are now explicitly labelled
apparent remainders; historical actual free remains unavailable. Direct
09:16 UTC query gives total/used/free 49,140/41,516/7,004 MiB and separately
reserved 622 MiB. The reusable collector now records explicit memory.free and
memory.reserved separately while preserving legacy device rows. Missing free
readings retain the other contention evidence and remain unavailable. Current
trainers are not restarted for this telemetry change; it applies at their next
ordinary resume. Memory admission uses explicit device free memory.
The campaign/launch-guard suite passes 46 tests, including unavailable reserved
memory and failed free-query cases; a real read-only collector invocation
records GPU2 free/reserved 7,004/622 MiB while retaining all legacy device and
process evidence. The collector creates no CUDA model or optimizer.

Fine's exact e750 audit passes 450,000 training case visits, all 120 AdamW
states at step 9,750, four RNG streams, unchanged frozen Stage-A/normalization
and five response calibration records. The immutable field/T-selected e493
snapshots remain read-only and unchanged. Recent complete epochs take a
median 17.81 seconds under GPU2's three-job placement. Fine subsequently
completed exact e1000 with native exit 0 at 08:14:51 UTC: all 1,000 contiguous full
epochs, 600,000 train and 90,000 validation case visits, 120 AdamW states all
at step 13,000, all four RNG streams, unchanged Stage-A/normalization/dataset/
calibration, policy 2 and horizon 5000. Immutable exact e1000, field-selected
actual e972 (selector 0.00836234) and T-selected actual e971 (0.00658375) are
byte-verified and read-only. The field selection alone is used for selected
physical/response/inverse comparison; no T-selected role is substituted.
Fine's final ten complete epochs take median 18.19 seconds under three-job
GPU2 contention. After it exits, Overlap e262–266 and Local e239–243 measure
median 62.01/68.03 seconds with two owned trainers and 19,911 MiB free; these
regimes are reported separately.

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
18 input-selected development cases at their full native 64x128 grids. The
completed H-tree, H-overlap and H-local screens use the same cases and physical evaluator.
Case IDs are preserved for subsequent screens. Equal-case fluid RMSE is:

| Field | B-native e100 | B-fine e100 | H-tree e100 | H-overlap e100 | H-local e100 | Mature Run1804 e4738 |
|---|---:|---:|---:|---:|---:|---:|
| u | 0.0698578 | 0.0738634 | 0.104391 | 0.0908794 | 0.0769374 | 0.00590808 |
| v | 0.00511782 | 0.0106410 | 0.0122159 | 0.00976146 | 0.0104132 | 0.000337623 |
| p | 0.0252665 | 0.0302576 | 0.0407222 | 0.0358716 | 0.0391842 | 0.00210899 |
| omega | 0.217016 | 0.302935 | 0.328017 | 0.303445 | 0.286309 | 0.0258649 |
| temperature | 1.43558 | 1.67576 | 2.70899 | 3.27004 | 1.69682 | 0.187831 |

These benchmark physical-scale quantities each retain their own units; they
are not averaged into one physical scalar. B-fine is finite and improves
with training but still misses mature fidelity. Young-versus-mature accuracy
does not isolate architecture and does not stop the healthy ladder. Saved
evidence also includes near/far fluid, interfaces, material peaks and ports.

The current packed H5 and inspected source case configuration do not assign
SI units to these channels. Source case0692 declares the analytic-wake,
shared-temperature-grid benchmark and zero inlet/wall reference temperature.
Therefore figures and tables use native benchmark coordinate, velocity,
pressure and temperature scales, without assigning m/s, Pa or kelvin. The
interface q_normal remains a generator flux proxy.

Tree100 is finite across all eighteen cases and all 24 maintained physical
metrics. Its surface/q_normal/material/peak RMSE means are
3.08500/5.44432/2.73101/3.24330; near/far T 2.89751/2.66118 and pressure-difference
error 0.0167104. It improves temperature over Overlap100 while missing the
same-age controls and Local100, and has no fluid-role advantage over the
fresh controls. A finite improving training curve supports continued study;
it does not establish physical fidelity or useful organization.

An independent saved-array audit reconciles all 24 metric keys and equal-case,
pooled and tail aggregates exactly. Four-case actual P0/P1/P2 access records
retain 1,826,367 of 7,743,578 eligible pairs; 5,917,211 are omitted logically.
Execution nevertheless uses 7,995,072 padded fine rows/644 calls with zero
skipped eligible rows. MM/ME K varies 3–8 while EM/QM/QE K is 8 in these twelve
case-phases; proper multi-source groups occur in 3/12 MM, 12/12 ME, 11/12 EM and
12/12 QM/QE scopes. This is a real typed graph with a fidelity miss and no
measured sparse execution saving, not an established graph benefit.

Tree's exact e100 also completes twelve strict reference-control comparisons
on four of these cases. Saved-array recomputation matches all 288 role metrics
exactly with identical geometry, references and masks. At fixed normal access,
removing controls preserves all 644 native access calls and permissions
bitwise, but increases four-case fluid T RMSE from 2.19761 to 2.55175. Full
access with normal controls changes 5,917,211 native pairs and increases T to
3.98775, while improving material-peak RMSE from 2.51116 to 1.69392. The
bounded geometry action preserves receiver support cardinalities and weight
multisets, changes 1,506 native pairs, and gives T 2.19857/surface 2.44559
versus normal 2.19761/2.45752. No intervention wins all seven audited roles
on any case. Thus controls affect the operator, but learned selection has no
clear advantage over this bounded geometry comparison. This is neither a
global geometry optimum nor a population graph-utility result.

The immutable mature e4738 reference has also completed the full 90-case native
development population. Excluding duplicate0273, equal-case fluid u/T RMSE is
0.005721/0.218656 and module material-peak RMSE 0.346587; compatibility90 yields
0.005707/0.217622/0.344636. These full-population results are saved separately
from the 18-case screen and will anchor the 500/1000 comparisons. They retain
the same predicted ports, frozen Stage-A and previously exposed benchmark limits.

B-fine's completed 500-epoch stage has now been evaluated on the same full
population. All 90 native-grid predictions are finite for exact e500 and the
stage's field-selected e493. The canonical 89-case equal-case comparison is:

| Physical quantity | B-fine exact e500 | B-fine selected e493 | Mature selected e4738 |
|---|---:|---:|---:|
| Fluid u RMSE | 0.0265570 | 0.0254532 | 0.00572097 |
| Fluid temperature RMSE | 0.849954 | 0.714392 | 0.218656 |
| Near-fluid temperature RMSE | 0.778794 | 0.791955 | — |
| Far-fluid temperature RMSE | 0.860734 | 0.694040 | — |
| Surface-temperature RMSE | 0.739772 | 0.804033 | — |
| q_normal proxy RMSE | 3.75080 | 3.69446 | — |
| Material-peak RMSE | 0.637695 | 0.649535 | 0.346587 |
| Pressure-difference absolute error | 0.00677007 | 0.00740678 | — |

The complete per-case roles and separate compatibility90 aggregates are saved.
Independent recomputation from all 270 case archives (these two versions and
mature) matches 6,480 metrics and both aggregations within 4e-15, with identical
references, input data and masks. Selecting e493 improves fluid temperature by
15.95% relative to exact e500, but worsens near-fluid temperature, surface,
material peak and pressure errors. It remains worse than mature on 13 of 14
audited roles; the small effective-heat-transfer exception is not a general win.
This supports continuing the healthy matched control, while preserving physical
tradeoffs and the maturity difference. Both versions have also completed
all eight existing-family finite-response evaluations, with all 88 absolute
states saved and independent array audits. Across 80 perturbations, mean fluid
temperature response RMSE is exact500 0.14467 and selected493 0.14657, versus
Fine100 0.20682 and mature 0.11674. Mean pressure-increment absolute errors are
0.0002597/0.0002673, below zero-change 0.0002784 on average but still above
mature 0.0001974. Selected493 improves the 16 heat-transfer responses to
T RMSE 0.06900, while retaining spurious velocity/pressure response where the
benchmark has zero change. The field selection is not a universal response
winner. The authorized matched-control extension to 1000 retains exact e500
as its optimization parent, with both e493 selection snapshots preserved.

Native's completed 500-stage exact500 and predeclared field-selected456 also
complete all 90 native-grid cases. An independent saved-array audit reproduces
all 24 physical roles, canonical89/compatibility90 aggregates, exact M/Re strata,
inputs and masks with maximum metric discrepancy zero. The canonical89 results
show the selection tradeoff:

| Physical quantity | Native exact500 | Native selected456 | Mature selected4738 |
|---|---:|---:|---:|
| Fluid u RMSE | 0.021589 | 0.022163 | 0.00572097 |
| Fluid temperature RMSE | 0.745890 | 0.560723 | 0.218656 |
| Surface-temperature RMSE | 0.724999 | 0.672854 | 0.448389 |
| q_normal proxy RMSE | 3.44221 | 3.49145 | 1.58291 |
| Material-temperature RMSE | 0.556507 | 0.552987 | 0.340027 |
| Material-peak RMSE | 0.517528 | 0.569771 | 0.346587 |
| Pressure-difference absolute error | 0.004399 | 0.003687 | 0.00101383 |

Selected456 improves fluid temperature but worsens velocity, flux proxy and
material peak relative to exact500. Peak-error p90 increases 0.8027→0.9405;
the worst case0295/M7 increases 1.2843→1.9916. All 90 actual predictions in
each version execute 179,889,120 padded fine-MLP rows in 14,490 calls, measured
directly with hooks. These counts exclude attention, policy, coarse/local
physics, backward, graph export and eligible/unique-pair accounting.

Both versions complete eight stored response families/88 absolute states and
80 finite variants. Mean T response RMSE is exact500 0.135131/selected456
0.137524 versus mature 0.116739 and zero-change 0.274135. Mean pressure errors
0.000371/0.000381 exceed zero-change 0.000278. The sixteen heat-transfer
responses improve T to 0.061288/0.064708 versus zero-change 0.241760, while
spurious u/p responses remain 0.001477/0.000533 and 0.001472/0.000571 on null
reference channels. Each 88-state execution directly measures 175,891,584
padded fine rows/14,168 calls, reconciled by label and family. Evidence timers
include instrumentation and are not uninstrumented benchmark latency.

Selected456 completes twelve one-step readiness cases in three inverse modes.
Nine have full fixed-total Jacobian rank; three M10 cases have rank six in nine
free directions. The largest full-rank condition number is 297.78. Graph and
ungrouped non-time trajectories are bitwise equal to joint, with zero meaningful
graph steps and 24 fallback steps. Heat remains nonnegative with total drift
at most 1.91e-6, but mean observed/held residuals worsen 1.6076→2.1965 and
1.7674→2.0545. All three response/readiness process receipts preserve 310
forward-state tensors/5,430,548 scalars bitwise, with no forward gradients or
checkpoint mutation. Their evaluation source is base14577a0 plus the saved
measurement-tool diff; this is not reported as a clean launch. These completed
physical and response reviews support the authorized matched-control extension
from exact500 to1000; one-step readiness remains distinct from inverse quality.

B-fine's completed 1000-stage exact e1000 and predeclared field-selected e972
have now completed the same full 90 native-grid cases. An independent audit
recomputes every one of the 24 maintained physical roles, canonical89/
compatibility90 aggregates, exact M/Re strata, inputs and masks with maximum
metric discrepancy zero. The selected e972 is 28 epochs younger than exact1000;
the separately preserved T-best e971 contributes no role to this comparison.

| Canonical89 physical quantity | Fine exact e1000 | Fine selected e972 | Mature selected e4738 |
|---|---:|---:|---:|
| Fluid u RMSE | 0.0208118 | 0.0194957 | 0.00572097 |
| Fluid temperature RMSE | 0.707640 | 0.562692 | 0.218656 |
| Near-fluid temperature RMSE | 0.726882 | 0.548674 | 0.249451 |
| Surface-temperature RMSE | 0.718738 | 0.546184 | 0.448389 |
| q_normal proxy RMSE | 3.28969 | 3.33277 | 1.58291 |
| Material-peak RMSE | 0.586344 | 0.531433 | 0.346587 |
| Final-port h_effective RMSE | 0.656939 | 0.719937 | 0.614266 |
| Pressure-difference absolute error | 0.00447348 | 0.00470100 | 0.00101383 |

Selected e972 improves fluid temperature by 21.24% and u by 23.41% relative
to the prior field-selected e493. Temperature p90/max declines from
0.98257/1.33201 to 0.75860/0.95795. It still misses mature performance on these
roles. Selection improves most displayed quantities over exact1000 but
worsens q_normal, pressure difference and h_effective; h_effective also worsens
relative to e493. These results demonstrate continued control learning while
preserving the physical tradeoffs and different training ages.

Both versions complete all eight stored response families/88 absolute states
and 80 finite variants. Independent array recomputation matches response
metrics within 8.88e-16 and pressure reductions exactly. All-variant fluid T
response RMSE is exact1000 0.134809/selected972 0.134477/mature4738 0.116739;
q_normal proxy response RMSE is 1.19217/1.19107/1.27294, respectively. Selected
pressure-increment error is 0.000279963, slightly above the zero-change
predictor's 0.000278366; exact1000 yields 0.000203173. For sixteen heat-transfer
variants, selected T response RMSE is 0.073462 versus zero-change 0.241760,
but spurious u/p response remains 0.001701/0.000448 where the reference is null.
These are useful thermal responses with adverse flow/pressure behavior.
Actual fine rows are unmeasured in these completed legacy-backend response
archives; the available 264 preparations/880 wrapper reads are not row counts.

The selected e972 one-step readiness panel completes twelve cases in the
same three inverse modes. All 249 loaded forward-state tensors, including
persistent buffers, remain bitwise unchanged in the execution receipt.
Nine tasks have full fixed-total Jacobian rank; the three M10 tasks have
rank six in nine free directions. The largest full-rank condition number is
743.38. All non-time graph/ungrouped trajectories are bitwise equal to joint,
with no meaningful graph update and 24 full-joint fallback steps. Public-total
drift is at most 1.91e-6. Mean observed/held residuals worsen
1.3615→1.7532 and 1.7225→1.8711 after one step, so this is execution/conditioning
readiness with an inverse miss, not completed optimization quality.

The subsequent e972 frozen inverse evaluation completes all 108 thirty-step
trials, with twelve cases, three starts and three update modes. Every public
task/reference array and all 36 initial heat streams match the mature
single-thread comparison bitwise. The natural exit0 receipt verifies all
249 loaded parameters/persistent buffers (3,963,412 scalars) unchanged,
no forward gradients and the immutable selected checkpoint intact. This is
completed frozen surrogate optimization, separate from the one-step check.

| Equal-case/start metric | Fine selected972, initial→final | Mature4738, initial→final |
|---|---:|---:|
| Observed temperature RMSE | 2.02096→1.30027 | 3.41301→2.04577 |
| Held temperature RMSE | 2.40524→1.91480 | 3.73272→2.64320 |
| Module material-peak RMSE | 4.64011→4.37014 | 5.50964→4.55669 |
| Endpoint pressure-difference absolute error | 0.00572030→0.00969809 | 0.0143276→0.00901929 |
| Individual-heat RMSE, evaluation only | 0.582924→0.691228 | 0.582924→0.548525 |

Fine's observed/held means improve by 35.66%/20.39%, and their final values
are below the mature trial means. Its initial residuals are also lower, so
this does not isolate a greater optimization gain or better population
forward fidelity. Fine's pressure error increases 69.54% and individual-heat
RMSE increases 18.58%; observed/held/peak/pressure errors worsen in
9/11/13/23 of 36 starts. All heat states are nonnegative with inactive slots
zero and maximum FP64-summed relative total drift 8.80e-7. Nine Jacobians
have full local rank; the three M10 cases retain rank six in nine free
directions. Heat recovery therefore remains partially nonidentifiable.
All 72 graph/ungrouped trails equal joint bitwise apart from elapsed time,
with zero meaningful graph steps and 1,080 full-joint fallbacks per mode.
Median one-thread CPU joint trial time is 22.36 seconds, versus mature
23.14 seconds under the shared host; this is not an architecture/GPU speed
comparison. Endpoint pressure uses two saved fluid endpoints, distinct from
the full-field section metric and finite-response pressure bands. No new
physical solve validates these proposed allocations.

The input-only population audit finds finite development inputs within every
training range, but 80/90 cases occupy sixteen M–Re combinations absent from
the 600-case training population. Canonical89 still has eighty such cases;
this is joint coverage, not numeric-range extrapolation or a fresh test claim.
The canonical M3/5/7/10 counts are 24/25/25/15, and exact Re50/80/100/140/150
counts are 14/20/20/20/15. Inlet velocity and the non-viscosity material
parameters are constant; viscosity varies with Re, so they cannot support
independent context-effect conclusions here. The evaluator now reports
exact M and Re strata separately, including pooled errors, equal-case means
and tails; unavailable context remains explicitly counted. Saved-only
backfills preserve all original ninety-case arrays and all-case aggregates.

| M (cases) | Fine selected493 fluid T mean / p90 RMSE | Mature4738 fluid T mean / p90 RMSE |
|---|---:|---:|
| 3 (24) | 0.5341 / 0.6264 | 0.1323 / 0.1569 |
| 5 (25) | 0.7041 / 0.8452 | 0.2204 / 0.3609 |
| 7 (25) | 0.7752 / 0.9470 | 0.2994 / 0.4595 |
| 10 (15) | 0.9187 / 1.0651 | 0.2194 / 0.2462 |

These strata at different checkpoint ages show a broad maturity gap.
They do not isolate module count from heat/context or architecture from
training age; the eventual matched e1000 controls remain required.

The selected e493 inverse-conditioning readiness panel is complete on the
same twelve fixed-total heat tasks: six observed and six held sensors, a
uniform public-total start and one disposable heat-optimization step in each
joint/graph/ungrouped mode. This is a readiness check, not a completed inverse
quality evaluation. Numerical Jacobian ranks are 2/4/6/6 for M3/5/7/10;
the three M10 tasks remain locally nonidentifiable (nine free directions,
six observations), and M7 case0675 has condition number about 1,869. Forward
and local-surrogate weights remain frozen. Full finalist inverse trials are
still required at the selected 1000-stage versions.

An independent saved-array review finds all non-time joint/graph/ungrouped
one-step trails bitwise equal: the latter modes take 24 full-joint fallbacks
and no meaningful graph update. Mean observed/held residuals worsen
1.394→2.099 and 1.659→2.264 in this single-step check, so it establishes no
inverse-quality gain. All proposals remain nonnegative with public-total
FP32 drift at most 1.91e-6. The retained checkpoint hash is unchanged;
this readiness evidence has no before/after in-memory tensor snapshots, so
the saved audit alone does not prove runtime freezing.

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

A subsequent saved-file admission review at12:05 UTC measured 113.43 GiB free
on data and 69.49 GiB on root. Inspection of 123 case/mode archive sizes and ZIP
payload lengths gives a doubled observed-payload planning estimate of 9.31 GiB
per H-stage evaluation, or 46.56–55.87 GiB for all three H500 evaluations plus
two or three H1000 evaluations. This is a planning estimate, not a storage bound;
future checkpoint retention, full inverse/generative outputs, unmeasured tails
and unrelated writes are excluded. The reviewed 33 stage500 output paths are
distinct and initially absent. No deletion or migration is warranted by this
review; each actual launch still checks current free space and subsequent growth.

## Selected measured figures

Figure index: [full-epoch training progress](../../diagnostics/generated/shared_core_campaign_20261002/figures/campaign_training_progress.pdf),
[selected control fields and residuals](../../diagnostics/generated/shared_core_campaign_20261002/figures/predictor_screen_case0692.pdf),
[interface and material fidelity](../../diagnostics/generated/shared_core_campaign_20261002/figures/interface_material_case0692.pdf),
[frozen finite responses](../../diagnostics/generated/shared_core_campaign_20261002/figures/reference_response_baseline.pdf),
[matched frozen inverse comparison](../../diagnostics/generated/shared_core_campaign_20261002/figures/frozen_inverse_comparison.pdf),
[actual overlap graph and work](../../diagnostics/generated/shared_core_campaign_20261002/figures/overlap_e100_graph_work.pdf),
[executed work versus latency](../../diagnostics/generated/shared_core_campaign_20261002/figures/executor_work_vs_latency.pdf).

![Full-epoch training and measured elapsed work](../../diagnostics/generated/shared_core_campaign_20261002/figures/campaign_training_progress.png)

The inspected 13:51 UTC snapshot contains 944/1000/300/500/486 complete epochs for
B-native/B-fine/H-tree/H-overlap/H-local. Every displayed epoch has 600 distinct
training cases, 13 optimizer steps and 614,400 primary sampled queries. B-fine
e1000 normalized sampled field/T MSE is 0.01175/0.01182; the full-grid physical
evaluation is reported separately above.
Successful train/validation time sums are 9.106/4.400/11.116/9.608/10.359 hours;
discarded partial work and storage replay overhead remain recorded separately.
Owned contention and H-tree engineering changes affect timing. These curves
establish actual model age and work, not sparse speedups or mature-equivalent
physical fidelity. The figure marks the H pressure ramp and common e101
response/native-denominator amendment without combining different field units.
Timing diamonds mark the first complete epochs in GPU2's three-trainer
placement: Fine501, Overlap162 and Local152. Squares mark Native joining GPU2
at Native501, Overlap309 and Local282. Their affected H timing remains
separate from architecture or sparse-work claims.
An open circle marks the Local benchmark PID observed at Tree214. Overlap's
benchmark fell between boundary samples, with no known epoch intersection.
The source snapshot and complete numerical curves are retained; these concurrent
elapsed sums are not exclusive GPU hours. Prior snapshot evidence is preserved.

![Measured native Tree executor work, latency and memory](../../diagnostics/generated/shared_core_campaign_20261002/figures/executor_work_vs_latency.png)

Native Tree exact e100 is evaluated on cases 0274/M3 and 0692/M10 at Q14 and
Q8192. Q labels count caller field queries; complete-wrapper work also includes
native auxiliary port and coupling reads. Full-query complete-wrapper fine-MLP
rows decline by 77.06%/73.56%, while calls increase 161→164 and median latency
increases by 34.07%/13.28%
under the recorded GPU2 contention. Prepared P2 decode medians also increase
0.671→0.726 and 0.717→0.783 seconds. Median complete-wrapper allocated memory
increments change 44.90→38.58 MiB for M3 and 57.80→57.80 MiB for M10. These
increments exclude resident baseline and do not describe total device demand.
Every native parity gate passes before timing; parameters, persistent buffers
and checkpoint bytes remain unchanged. Bars, individual repeats and min–p90
whiskers expose the measurement scope and variation. This establishes actual
fine-row savings for inference, with no measured latency benefit in this
ordered, contended test. Training work and final selected checkpoints remain
separate. The PDF and embedding raster are visually inspected; plotting uses
saved numerical records only.

![Native-grid selected control fields and residuals for development case 0692](../../diagnostics/generated/shared_core_campaign_20261002/figures/predictor_screen_case0692.png)

On this input-selected high-M case (M=10), mature Run1804 e4738 yields fluid
u/T RMSE 0.0056393/0.174727. Fine's immutable field selections e493 and e972
yield 0.0283131/0.706880 and 0.0199249/0.702979, respectively. Thus the later
control improves u on this difficult case while T scarcely changes, despite
its clearer canonical-population T improvement. The figure retains the mature
baseline and displays both selected training stages and their residual maps;
the earlier e100 numerical evidence and screen table remain available.
The figure uses the stored exposed-development native-grid reference,
benchmark physical units, shared value/residual ranges, white solid masks,
predicted ports and the same frozen Stage-A model. It shows residual errors
near modules and in their downstream fields, including the remaining T miss
at the completed Fine1000 stage. It does not isolate
architecture from training age. Both the PDF master and small embedding raster
were visually inspected after fixing clipped labels; numerical arrays remain
under the ignored evaluation directory.

![Saved native interface, material and pressure fidelity for development case 0692](../../diagnostics/generated/shared_core_campaign_20261002/figures/interface_material_case0692.png)

The geometry-selected leftmost/rightmost modules use all 64 saved native
interface angles; material quantiles pool 30,960 active-module samples, while
peaks and role errors include all ten modules. The six displayed versions are
mature4738, Native field-selected456, Fine field-selected972, and Tree/Overlap/
Local exact100. Fine972's surface/material/peak RMSE is 0.4722/0.4198/0.3910,
below Native456's 0.6272/0.5101/0.5751 on this input, but above mature
0.3219/0.2341/0.2005. Fine's q_normal proxy RMSE remains 2.996 versus
mature1.321. Local100's surface/material/peak errors 1.370/1.171/1.257 are
below Overlap100's 4.274/3.948/4.364, while its pressure-difference error
0.01680 exceeds Overlap's 0.008403. Fine/Native/mature pressure-difference
errors are 0.007453/0.01045/0.001101. Pressure-difference errors are
shown alongside the full fluid-column means, with stored referenceΔp0.06534.
Common inputs, reference fields, masks and native query indices are verified;
no finite curve is clipped, and module slots retain physical identity. These
unequal-age, exposed-development benchmark results show physical role tradeoffs,
without implying SI accuracy, population superiority or independent design
validation. All 144 metric checks reconcile within 2.04e-14 for RMSE; the
figure's FP64 pressure reduction differs from the native FP32 reduction by at
most 1.32e-8. PDF and PNG were visually inspected and redundant inspection
exports removed; all source arrays remain locally retained.

H-overlap e100 is finite but worse than the same-age controls on fluid
temperature and surface/material errors: near/far T RMSE 3.4390/3.1789,
surface 3.4289, material 3.0983 and peak 3.5137. Its q_normal proxy RMSE 5.1202 is
slightly below B-fine 5.3558. Hard QE support retains 44.48% of eligible logical
pairs, while each hard/soft whole-epoch path still executes 253,186,272 rows
in 3,177 fine calls. This is a fidelity/support tradeoff with zero measured
training executor saving. The healthy curve continues to 500; H-local tests
the distinct near-access protection hypothesis from fresh initialization.

H-local's completed 18-case e100 screen has temperature RMSE 1.69682,
near/far 1.75737/1.67732, surface 1.67691 and material peak 1.53794. Its
q_normal proxy RMSE is 4.76873 and pressure-difference error 0.01839. Local
protection improves temperature and near/interface behavior versus H-overlap
at the same training age, but does not generally beat Dense/Fine or mature.
On the four directly captured native graph cases it selects 7,692,869 of
7,743,578 eligible pairs (99.345%), preserving all positive near envelopes.
It still executes 7,995,072 rows in 644 calls. The current policy therefore
offers almost all-access support and no measured executor saving.

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
saved normal control probes and summaries bitwise. A four-case native Overlap
e100 evaluation of both controls is now complete: every mode executes
7,995,072 rows in 644 calls with no skipped rows. Full access preserves normal
control probes/summaries bitwise across all 644 calls; its mean fluid T/surface/
peak errors are 3.59439/3.15696/2.69175 versus normal 2.84319/2.97581/3.02384.
Thus unrestricted access worsens fluid and surface while improving peaks.
The strict geometry action changes only 252 binary pairs, with 7,795 changed
weight positions, preserving both binary degrees and receiver weight multisets
across all native streams. Its mean T/surface/peak errors are
2.89440/3.14340/3.21553; this bounded matched action provides no thermal win on
this small panel. Case0647 has weight/control reordering without a binary
support change. Overlap has no near arrays here, so this near-protection check
is vacuous; the separate Local evaluation tests positive near envelopes.
Source-column weighted sums change in 198 of 644 calls, explicitly allowed
and reported. These findings supersede an interpretation of the old geometry
control as fully matched, and do not establish population utility. Actual
500-epoch checkpoint reviews remain pending.

The corresponding Local e100 four-case evaluation also completed. Its geometry
action literally preserves all 423,946 positive-near pairs, 114,469 full-near
pairs and 1,258 saved positive-near control probes. It changes 16,284 weight
positions but zero binary supports; this is weight/control reordering, not a
successful sparse support control. The candidate budget is reached in 389 of
644 calls, so the action remains a bounded greedy comparison. Normal/full-fixed/
geometry mean fluid T RMSE is 1.51626/1.61365/1.50549, surface T
1.40941/1.67565/1.49257 and material peak 1.15803/1.42936/1.25806. Geometry's small
fluid gain coexists with worse interfaces/material; all three execute the same
7,995,072 rows in 644 calls. Full normal control arrays were not historically
persisted: all-entry reconstructed constraints and exactly saved control probes/
summaries are disclosed separately, rather than claiming an archived full-tensor
comparison.

An independent saved-only Local audit checks the first QE access for
cases0647/0692 at P0/P1/P2. P0/P1 are full support; P2 omits 646/1,643 eligible
pairs. Exhaustive checks over 18,336 equal-measure source pairs in each P2 call
find no complementary two-receiver/two-source pattern permitting a legal
binary-degree switch with all four entries valid and outside protected near
envelopes. Thus these six representative scopes are constraint-degenerate
before geometry cost or the greedy budget is considered. This does not prove
global impossibility over all calls or larger multi-edge cycles.

Saved normal accesses also verify graph action on all five typed routes over
the same four cases and P0/P1/P2. Overlap omits 51/577/1,042/10,000/4,223,089
eligible pairs on MM/ME/EM/QM/QE; Local omits 14/320/60/310/50,005. Overlap has
proper multi-source groups in all twelve case-phases on every route; Local
has them in seven EM and nine QM case-phases, with actual QM omissions in
only one. Both still execute 7,995,072 fine rows in 644 calls. This establishes
real typed graph action at e100, without establishing final utility, sparse
work savings or causality. A future paired inverse-head comparison must use
all five actual typed links and demonstrate a nonzero same-weight graph/full
effect at the selected finalist; the local heat block helper alone does not
qualify that generative comparison.

The evaluation recorder now also saves exported native plan diagnostics for
each P0/P1/P2 phase, including Overlap/Local's actual admission-rescue flag.
The population auditor counts a recorded rescue once per case/phase, rather
than once per typed route; old archives explicitly retain unavailable rescue
counts instead of inferred zeros. Near access remains a separate measurement.
Disposable real Overlap tests force both all-open/no-rescue and all-closed/
rescued gates across all three phases, preserving exact field outputs,
persistent model state and restored methods. The focused recorder/reference/
geometry/evaluator suite passes 62 tests with one optional native-resource
skip: the train-only calibration replay requires
`HONF_ALIGNMENT_THERMAL_CHECKPOINT`, which is unset in this test invocation.
Ruff and diff checks pass. The old four-case population rehearsal still
reconciles all 60 route records, 644 calls, 7,995,072 fine rows and 251,494
padded rows, with zero skipped eligible rows. This is an evaluation-evidence
addition; interaction equations, training and existing checkpoint states are
unchanged. The next selected population captures will supply actual rescue
measurements where the architecture exports them.

A subsequent actual full-width native probe at clean `04dc2a6` verifies the
recorder on Overlap exact100, development case0647/M3, CPU1 with no visible
CUDA. All fifteen numeric P0/P1/P2 diagnostic arrays match observed native
exports bitwise; admission rescue is explicitly false in every phase and all
eight allocated groups are admitted. All five actual typed routes are present
per phase, and hook totals reconcile to 1,998,768 fine rows in 161 calls.
Six saved prediction/residual arrays and all 24 physical metrics are bitwise
identical to the existing same-case e100 evidence, without another comparison
forward. All 334 state tensors remain frozen; recorder methods/attribute
ownership and checkpoint SHA/stat/mode0664 are unchanged. The native process
exits0 at11:45:53 UTC; its instrumented timer is not benchmark latency.
`evaluation/h-overlap_e100_recorder_probe/recorder_probe_QA.json` and the exact
observer/execution receipt retain the evidence. This validates full-width
recording on one case; it does not establish a population rescue rate.

Actual e100 Overlap and Local invariance/path evaluations now cover native
low/high-M cases 0274/0692. Module permutation, inactive padding, fixed-action
quadrature splitting, query order and prepared chunks pass declared tolerances;
maximum observed context/field change is below 7.2e-7, with padding exactly
zero. Their trained topology really changes on both fixed-total heat and
small geometry paths. Eight bisections narrow one detected switch per path
to parameter width 0.0004883; physical temperature changes across those
brackets range about 3.1e-5–4.2e-4. Only two of the eight final brackets change
actual effective pairs; the rest change group topology without changing those
binary pairs. These finite brackets do not prove cross-switch
continuity or a finite jump. Fixed-topology affine continuation is invalid
farther along some paths and is stopped explicitly rather than silently
clamped. Local case0274 heat and case0692 geometry rebuilt central differences
at epsilon0.01 cross topology, while their fixed derivative branches reject
negative continuation. These are separate from within-active-set derivatives.
Other valid branches retain the anchor supports, with channelwise autograd and
central differences saved. On those same branches, decreasing epsilon from
0.01 to 0.0001 increases FP32 cancellation errors, particularly for q_normal.
These are frozen-surrogate numerical checks, not physical reference solves.

Run1804, both fresh e100 controls and H-overlap/H-local exact e100 have completed finite-response evaluation
on all eight existing non-training families (11 absolute states per family).
Fields are decoded once and saved with physical increments, pressure and
per-module peaks; the zero-change comparator and reference magnitudes remain
separate. No response-relative quality is claimed without established floors.
Saved-array response audits of all five versions independently recompute 3,200
field/material response-channel metrics, 400 pressure and 2,500 module-peak
scalars within 8.9e-16. Across all 80 perturbations, mean fluid-temperature
finite-response RMSE is Local100 0.21737, Overlap100 0.26016, Fine100 0.20682,
Dense100 0.19275 and mature 0.11674. Near protection improves this measure over
Overlap but does not beat the same-age controls. On heat-transfer perturbations
with exactly zero stored u/p response, Local's spurious errors are
0.003629/0.001412 and Overlap's 0.002780/0.001081. Both H response evaluations
execute 175,891,584 rows in 14,168 fine calls, with zero skipped rows. The
selected response figure below now shows mature4738, Fine field-selected972
and Native stage500 field-selected456; the H100 comparisons above use their
complete saved arrays. The control training ages remain unequal at this review.
Tree100 has now also completed all eight families/88 absolute states and
80 perturbations. Mean T/surface/q_normal/solid response RMSE is
0.25076/0.38432/1.28385/0.35109, missing both fresh controls on T.
Heat-transfer T response error 0.18174 improves on zero change 0.24176, but
spurious u/p errors0.004290/0.001487 miss the recorded heat-null control.
Mean pressure-increment error 0.0008040 misses zero change 0.0002784 and the
fresh controls. These source-labelled absolute measurements are distinct
from the later auxiliary inverse observation pairs and numerical smokes.
Independent stored-atlas reconciliation covers 264 absolute role arrays and
640 response-channel metrics within 8.9e-16, with exact pressure reduction.
Tree responses execute 175,891,584 rows in 14,168 calls, with zero skipped rows;
mature response executor work remains separately unmeasured in its legacy
saved evaluation. Comparisons do not fill that missing measurement with zero.
The mature local-inverse job also encountered ENOSPC on its eighth case;
individual complete trials are retained and resume now skips verified arrays.
Atomic numerical/JSON writes preserve previous complete files on interruption.

![Frozen native response errors and physical response maps](../../diagnostics/generated/shared_core_campaign_20261002/figures/reference_response_baseline.png)

Across eight previously exposed atlas families (four calibration and four
final-review), mean fluid-temperature response RMSE for mature4738 /
Fine-selected972 / Native-selected456 is 0.11674/0.13448/0.13752 over all 80
perturbations. On the sixteen heat-transfer perturbations it is
0.06610/0.07346/0.06471 versus zero change 0.24176; heat-flux-proxy RMSE is
0.57396/0.35217/0.33575. All stored heat-only u/v/p/omega responses are exactly
zero, while mean model u RMSE is 0.001231/0.001701/0.001472 and p RMSE
0.0004970/0.0004479/0.0005709: all fail these generator-specific nulls.
Across all 80, mean pressure-increment absolute error is
0.0001974/0.0002800/0.0003806 versus zero change 0.0002784, so Fine972 slightly
misses zero change and Native456 clearly misses it. Equal-variant mean
module-peak-increment error is 0.16486/0.16508/0.16218 temperature units.
The bottom maps show family0310 heat-transfer-plus on 7,918 common fluid
cells: temperature response RMSE 0.07045/0.06504/0.06509 versus reference
RMS 0.30041. Stored response and signed model residuals share the scale
[-1.14170,1.14170]; no finite plotted cell is clipped. White masks exclude
physical solids and noncommon full-stencil support. PDF and PNG are inspected,
and all 1,920 saved response-channel equations reconcile within 8.9e-16;
independent pressure reductions differ by at most 4.5e-8 in FP32. The figure
supports selective thermal-response gains alongside cross-field and pressure
misses at unequal training ages. Native analytic/shared-grid benchmark scales,
`q_normal` proxy status, unresolved response floors and frozen predicted-port/
Stage-A physics limit these claims; no independent solve, CFD, response-relative
accuracy or inverse quality is established.

![Matched frozen mature and Fine heat optimization, pressure and identifiability](../../diagnostics/generated/shared_core_campaign_20261002/figures/frozen_inverse_comparison.png)

The inspected comparison contains all 36 joint trails per checkpoint, their
equal-case/start means and trial IQR, actual projected-Adam steps 0–30,
three nonnegative fixed-total allocations for M10 case0692, and fixed-total local
Jacobian ranks. Fine selected972 reaches observed/held RMSE 1.3003/1.9148,
versus mature4738 2.0458/2.6432, but its endpoint pressure error grows
0.005720→0.009698 and heat-allocation RMSE grows 0.5829→0.6912. The figure
supports a frozen observation-matching result with physical-role misses;
all graph/ungrouped modes fall back to joint and establish no graph utility.
No finite plotted value is clipped. The replaced mature-only visual exports
are superseded by this comparison; both complete numerical histories remain.

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
about 5,575 despite full numerical rank. Panel e's stored individual heat is
evaluation-only, and its difference from recovered allocations does not by
itself establish an incorrect design. Every proposed field, peak and pressure
comes from the same frozen checkpoint; no independent physical solve was
performed. The saved design trails contain sensors/peaks/pressure rather than
full postoptimization field grids. The revised PDF master and embedding raster
were visually inspected; superseded visual exports were replaced, while both
numerical evaluation histories remain retained.

CPU one-thread inference timing is also complete for mature e4738 and both
e100 controls, using low/high-M cases 0274/0692, Q14/full8192, two warmups and
five repetitions with checkpoint hashes unchanged. Complete-wrapper median
seconds at low/high M are mature 0.385/0.418 (Q14), 1.245/1.418 (full grid);
Dense100 0.388/0.371 and 1.231/1.341; Fine100 0.372/0.383 and 1.906/1.247.
Prepared small-Q decode is about 0.0028–0.0033 seconds, excluding preparation.
Raw samples, p90 and occupancy are saved; host load is shared with the four
trainers, so these are CPU baselines rather than isolated or GPU speed rankings.

Fine's final field-selected e972 subsequently completes the same native
low/high-M benchmark on CPU with one thread, two warmups and five repetitions,
at 10:21:43–10:22:34 UTC from clean source `c25ca14`. Independent sample
recomputation matches all medians, p90 and minima exactly. All 249 loaded
parameters/persistent buffers and the read-only selected checkpoint remain
bitwise/byte unchanged, with no forward gradients or optimizer updates.

| Case / M | Primary Q | Complete P0/P1/P2 median seconds | Prepared P2 median seconds |
|---|---:|---:|---:|
| 0274 / 3 | 14 | 0.37149 | 0.002707 |
| 0274 / 3 | 8192 | 1.88536 | 1.51516 |
| 0692 / 10 | 14 | 0.38651 | 0.002754 |
| 0692 / 10 | 8192 | 1.26447 | 0.903151 |

Direct five-route hooks record complete-wrapper Q14/Q8192 fine-MLP input
rows/calls 330,456/35 and 1,998,768/161 at both M. Prepared P2 records
2,856/2 and 1,671,168/128. Q counts caller field queries; complete work also
includes native auxiliary port/coupling reads and padded cells. These are
measured benchmark panels, not inferred eligible pairs or backfilled work
for older full90/response archives. Policy, attention, coarse/local physics
and backward work are outside these fine-MLP counters. Complete and prepared
latency scopes remain independent. CPU lifetime peak RSS reaches 622.67 MiB,
which is not an invocation memory peak. Shared-host conditions and fixed
case order prevent treating these values as GPU or architecture speed ranks.
Legacy Dense/Fine rows and calls are unavailable in the earlier telemetry:
they remain unmeasured there, not zero. A reusable evaluation-only hook recorder
now counts actual inputs/calls at all five fine physical MLPs, including padding.
Dense and each typed dense/rectangular backend preserve outputs and first
gradients bitwise in 2-D/3-D tests; the typed counts match native ledgers.
Native checkpoint tests verify hooks stay outside timed calls and backward.
Unsupported historical backends retain timing with explicitly unmeasured work.
This recorder excludes attention, coarse/local/policy work and eligible/unique
pairs; it supplies a common fine-kernel comparison without inferring sparse
hardware savings. Separate work-only native measurements now confirm all
three retained baseline/control checkpoints use 330,456 fine rows in 35 calls
for a Q14 complete wrapper and 1,998,768 in 161 calls for Q8192, at both M3/M10
cases. Prepared decode alone uses 2,856/2 and 1,671,168/128 rows/calls. The
equality reflects actual twelve-slot padding and 192 environmental tokens;
attention, policy and Dense's extra coarse/local branches remain outside this
fine-MLP scope. Checkpoint hashes stayed unchanged; latency was not remeasured
during these work-only calls. The focused suite passes 39 tests including retained native
resources, with Ruff and diff checks clean. Finalist dense/rectangular GPU
timing and physical-row comparisons are still required.

Frozen-inverse verification now snapshots loaded parameters and persistent
buffers, checks bit patterns after every case, and rejects forward gradients
or training mode. A genuine two-case mature4738 CPU1 smoke executes six
one-step trials and fourteen predictor calls, including two Jacobian calls;
all 310 state tensors/5,430,548 scalars remain unchanged. An explicit recovery
reuses six saved trials and makes only two fresh Jacobian calls; its verification
scope excludes retrospectively certifying prior trial execution. These are
software/freezing checks, not additional inverse-quality results. The focused
revision suite passes 64 tests with one unrelated optional native timing test
skipped; the genuine native freeze smoke is executed separately. Ruff and
diff checks pass. The generative evidence resolver also accepts the campaign's
ignored workspace path through its data symlink, while rejecting source-path
escapes; its actual campaign path resolves correctly without writes.

The paired-generative workflow now verifies the same frozen state before
saving review checkpoints and after each review, with new and reused draw
counts distinguished. A disposable native Overlap100 CPU1 paired update on
training case 0001 uses one measured provider call for both arms and leaves
all 334 forward-state tensors/3,994,148 scalars bitwise unchanged. Normalization
uses all 600 training records. This is integration evidence, not formal inverse
head training, a 200-update review or finalist qualification; no generative
campaign has begun.

A separate native-consumer rehearsal at exact Overlap100 uses the fixed
M3/5/7/10 cases 0647/0291/0296/0692 and three caller-owned public-total seeds:
uniform and opposite geometry-rank allocations. After centered noise at
time 0.5, their current simplex-projected noisy candidates reach the native
provider, matching the planned `candidate_projected` policy. It does not read
hidden reference heat or held temperatures for the measurement.
All five source-normalized graph routes differ from their full-link controls
in all twelve records; QM/QE transpose normalizations are also retained.
The same untrained seed0, width96, two-layer inverse head gives graph/full
centered-heat velocity RMS differences 0.01113–0.06457 and normalized
observation-value gradient norms 0.000411–0.008107. Both arms retain identical
frozen embeddings and head weights. Independent saved-array reductions
reproduce link/velocity/observation summaries within 6.6e-8, and the current
simplex candidate within 1.5e-7; maximum public-total error is 5.4e-7 with
nonnegative allocations and zero padding. This establishes a nonvacuous
native consumer operation at e100,
not inverse task improvement or final qualification. No head optimizer,
sampling trajectory or physical solve is created. All 334 forward-state
tensors/3,994,148 scalars and the checkpoint bytes remain unchanged. The CPU1
measurement completes in 14.71 seconds from clean source `642c654`; saved
source, matrices, outputs and observation derivatives are in
`generative_readiness/h-overlap_e100_candidate_projected_rehearsal`. An earlier
fixed-public-proxy link rehearsal remains separately retained. Repeat this
check on the actual selected finalist and combine it with the same-weight
physical utility and population graph evidence before paired-head training.

The reporting/freezing revision passes 78 focused tests with two
optional native-resource tests skipped; retained-native local and paired-head
smokes and saved-array context backfills provide the separately measured
integration evidence. Ruff and diff checks pass.

The paired inverse task adapter now adds four auxiliary, physically recorded
baseline/heat-transfer-plus pairs at M3/5/7/10. Each pair retains identical
geometry/material/context and exact saved/native public total, with different
nonnegative allocations and six observed temperatures; six disjoint held
temperatures and individual heat remain separate evaluation supervision.
All four have archived Re50/viscosity0.018, not the contexts of the original
twelve native inverse tasks. They remain separately labelled, previously
exposed final-review cohorts. No new reference solve or review normalization
is used. The formal review design keeps the original 96 draws and twelve
labelled controls, adding sixteen auxiliary draws with one shared initial
noise across both targets and both heads per pair.

On all four retained pairs, native Overlap100 CPU1 at uniform candidate heat
produces bitwise-equal predictions and all five typed links between each
pair's tasks, while their recorded observations differ. Sixteen disposable
untrained two-step draws and their explicit recovery complete successfully;
all 334 forward-state tensors remain bitwise unchanged. Recovery validates
public geometry/context/query/source identities, budget and evaluation-only
references, and rejects draws missing their original reference. All sixteen
numeric trails remain bitwise equal after these metadata checks. This tests
execution and target separation, not conditional quality or a trained inverse
result; no 200/750/1500-update head campaign has started. The revision passes
109 focused tests with two optional native-resource skips; native integration
is measured separately. Ruff and diff checks pass.

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

## Measured Overlap500 review and conditional continuation

Overlap500 produces a nontrivial learned organizer with measurable same-weight
physical effects, but it is not currently a fidelity or latency finalist.
Field selection improves most physical means and responses over exact500;
temperature/peak tails, pressure responses and inverse quality remain misses.
Reserve the third H1000 slot pending the complementary Local500/Tree500 review:
Overlap merits that slot only if its overlapping source organization adds an
informative comparison beyond the two retained H lineages. If Local and Tree
both supply useful, distinct graph actions with stronger physical/inverse
evidence, prioritize those two. The continuation decision awaits that cohort review.

The immutable Run2204 stage receipt separates **exact500**, field-selected
**actual292** and temperature-selected **actual269**. The measurements below
use the first two;269 is not substituted for292. All native jobs preserve334
parameter/buffer tensors and checkpoint SHA/stat, with evaluation source
`5446ce1` distinct from training launch `9483034`. The
[completed evaluation receipt](../../diagnostics/generated/shared_core_campaign_20261002/logs/h-overlap_stage500_evaluation_completion.json)
retains checkpoint ages and individual execution scopes.

Predictor evidence recomputes all24 maintained roles from saved arrays/masks
with zero aggregate discrepancy. Canonical89 excludes0273; compatibility90
is separately retained, and neither is a fresh test set. Equal-case means in
native benchmark scales are:

| Role | Exact500 | Field-selected292 |
|---|---:|---:|
| Fluid temperature RMSE | 3.390662 | 3.147519 |
| Surface temperature RMSE | 3.122379 | 2.864735 |
| Material temperature RMSE | 2.549547 | 2.395338 |
| Module peak RMSE | 2.717840 | 2.508833 |
| Fluid u RMSE | 0.163987 | 0.082642 |
| q_normal proxy RMSE | 4.545853 | 4.371272 |
| Inlet–outlet pressure AE | 0.014745 | 0.013362 |

Selection does not improve every tail: temperature p90 rises4.280530→4.416176,
peak p90 rises3.983539→4.186112 and q-proxy maximum rises7.446520→7.583871.
Compatibility90 temperature means are3.377354/3.143221. Exact-M/Re strata,
pooled errors and all-role tails remain in the
[independent physical audit](../../diagnostics/generated/shared_core_campaign_20261002/evaluation/independent_h-overlap_stage500_physical_review.json).
The older mature4738 reference remains an unequal-age contextual comparison.

Organizer evidence covers90×3phases×5routes=1,350 native records at selected292.
AllocatedK is8; admitted/nonemptyK is5–8 inP0 and8 throughoutP1/P2. Explicit
admission rescue is0/90 in each phase. Binary membership supports repeat in
MM/EM/QM, while exact membership-plus-group-control duplicates are zero;
repeated support does not establish identical effective actions. Eleven
eligible MM receiver rows have no selected source; near/access fallback
metadata is unavailable, separately from recorded admission rescue. The
[population audit](../../diagnostics/generated/shared_core_campaign_20261002/graph_population/h-overlap_stage500_selected/independent_completed_review.json)
retains source measures, overlap, participation and M/Re/context strata.

All three strict controls pass644 actual calls each. Fixed-access zero
controls preserve normal permissions bitwise; full access changes785,396
support positions while preserving saved control probes/receiver summaries.
Both worsen23/24 four-case mean roles, improving final h_effective only.
Bounded geometry changes1,458 support and9,532 weight positions, preserving
both binary degrees, exact-measure row permission multisets and protected
entries; it improves9 mean roles and worsens15. Peak RMSE increases0.231612
while pressure AE falls0.001615; temperature p90 improves despite a slightly
worse mean. Complete pair-control tensors are not saved, and this is not a
global optimality or physical-causality proof. The
[strict/ordinary audit](../../diagnostics/generated/shared_core_campaign_20261002/graph_utility/h-overlap_stage500_selected_reference_actions/independent_saved_controls_review.json)
also separates learned-feedback ordinary controls from fixed-reference actions.
The fixed summary uses600 disjoint training inputs and one uniform active root;
its saved controls match the training calibration across all four layouts.

The selected normal90 dense executor processes179,889,120 fine rows/14,490
calls, with6,068,130 padding rows and zero eligible pairs skipped. Native
CUDA2 dense/subset checks subsequently pass all four output/first-gradient
gates per checkpoint before timing. Each checks203 defined parameter gradients,
live heat/query gradients,109 frozen parameters and35/161 actual access
inventories forQ14/Q8192. Original elementwise gate results are retained;
independent saved review recomputes gradient-norm errors, not raw tensors.

| Checkpoint / M / Q8192 | Complete dense→subset seconds | Prepared P2 dense→subset seconds | Complete fine rows dense→subset |
|---|---:|---:|---:|
| Exact500 /3 | 0.916758→1.250953 | 0.630416→0.871738 | 1,998,768→373,660 |
| Exact500 /10 | 1.006343→1.112893 | 0.593638→0.799033 | 1,998,768→532,032 |
| Selected292 /3 | 0.954593→1.142509 | 0.711205→0.814390 | 1,998,768→1,763,596 |
| Selected292 /10 | 1.041834→1.285353 | 0.608528→0.756179 | 1,998,768→1,904,768 |

These are five-repeat medians after two warmups, under owned Native/Local
GPU2 contention and fixed dense-before-subset order. Complete and prepared
scopes are independent. Full-Q subset latency is slower despite fewer measured
fine rows; complete calls rise161→164. Timed allocated maxima are90.976MiB;
reserved3962/4034MiB includes preceding parity cache, distinct from driver
footprint and an unrecorded overall backward peak. The
[exact timing audit](../../diagnostics/generated/shared_core_campaign_20261002/timing/h-overlap_e500_cuda2_verified/independent_review.md)
and [selected timing audit](../../diagnostics/generated/shared_core_campaign_20261002/timing/h-overlap_stage500_selected_cuda2_verified/independent_review.md)
retain all16 rows each. This is no isolated or cross-architecture speed rank.

Both response versions complete eight existing families/88 absolute states and
80 correlated perturbations. Selected292 temperature response RMSE is0.260419,
improving exact500 0.309989 and zero-change 0.274135, but missing mature4738
0.116739 and Fine972 0.134477. On16 heat transfers, selected temperature/q-proxy
errors0.167026/0.297625 improve zero-change0.241760/0.939161. Spurious u/p
errors0.004234/0.001650 miss their exactly-zero references. Mean pressure
increment AE0.000704 misses zero-change0.000278; heat-only pressure error
0.000219 is adverse against zero. The
[saved atlas audit](../../diagnostics/generated/shared_core_campaign_20261002/responses/independent_overlap_stage500_response_review.json)
keeps absolute errors, masks, role tails and exposed calibration/final-review
partitions separate. Analytic q_normal is a proxy, not a new CFD heat-flux solve.

Inverse evidence is limited to12cases×1start×1step per checkpoint, not the
108-trial/30-step quality comparison. Both native readiness receipts preserve
frozen state and feasible nonnegative supplied-total heat; all12 graph steps
use proper blocks without full-joint fallback. At selected292, nine local
Jacobians have full fixed-total rank; the three M10 tasks have rank6/9 with six
observations. Mean observed residual worsens after one step in both versions;
nonvacuous graph operation is not established inverse improvement. The
[selected readiness summary](../../diagnostics/generated/shared_core_campaign_20261002/inverse_readiness/h-overlap_stage500_selected/summary.json)
and [frozen execution receipt](../../diagnostics/generated/shared_core_campaign_20261002/logs/h-overlap_stage500_field_selected_one_step_inverse_readiness_cpu1_observer.json)
retain that limit. Final1000 graph qualification, matched inverse quality and
any paired-head campaign remain pending the actual finalist review.
