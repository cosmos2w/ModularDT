# Shared-core Thermal strategy campaign: live status

The campaign trains a fresh native Dense control, a fresh three-term control,
and three learned organizers while preserving Thermal's predicted-port
P0/P1/P2 coupling and the frozen Stage-A local surrogate. Run1804 selected
e4738 remains an evaluation-only mature reference. Wind scientific training
is paused; Wind is used for shared-core compatibility checks.

**State: full-dataset control screening is running.** Snapshot: Dense e25 and
three-term e57 completed. Work is on `agent/honf-core-next`. The initial
finite portfolio is Runs 2201–2205; two repaired screens are available only
when a concrete failure warrants them. No 5,000-epoch job is authorized to
start automatically.

| Run | Strategy | Complete epoch | Sampled development field / T MSE | Current action |
|---|---|---:|---|---|
| 2201 | B-native | 25 | 0.33922 / 0.41169 | GPU 1 to e100; H-tree queued |
| 2202 | B-fine | 57 | 0.62490 / 0.27502 | GPU 2 to e100; H-overlap queued |
| 2203 | H-tree | 0 | Pending | Native optimizer/gradient checks passed |
| 2204 | H-overlap | 0 | Pending | Native optimizer/gradient checks passed |
| 2205 | H-local | 0 | Pending | Native optimizer/gradient checks passed |

These MSEs use the maintained normalized, sampled 90-case validation task;
they are neither full-grid physical errors nor the final 89-case comparison.
Every completed control epoch has 600 unique cases, 75 microbatches, 13 native
buckets/optimizer steps and 614,400 primary field queries. Exact e25/e50
checkpoints are retained as reached.

All formal epochs must visit all 600 native training records exactly once;
batch caps are rejected. Effective batch 48, primary sampled Q 1024, FP32,
Adam 3e-4/weight decay 1e-5, clipping 1, seed 0 are common. Microbatching preserves
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

At this snapshot the last-ten-epoch median training/validation times are
24.53/1.75 seconds for B-native and 10.97/0.90 seconds for B-fine. Peak allocated
memory is about 4.53/4.15 GiB. GPUs 1/2 have no external GPU processes; host
resources are shared with an unrelated GPU 0 job. Training to e100 is roughly
33 minutes remaining for Dense and 9 minutes for B-fine, plus plotting/save
overhead. Their remaining e1000 training+validation work is approximately
7.12 and 3.11 GPU-hours respectively. H-strategy forecasts await 5–10 actual
epochs; no control timing is substituted for their shadow/organizer cost.

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
