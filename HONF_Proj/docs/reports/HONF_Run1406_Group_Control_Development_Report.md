# Run 1406 Group-Control Development Report

This is the initial evidence/reporting structure for the low-dimensional
group-control Run 1406 hypothesis. It is intentionally not a training or GPU
launch record. The physical comparison remains pending explicit execution.

## Scope and ownership

- Primary live comparison: explicit Run 1406 epoch-50 checkpoint versus
  explicit Dense Run 1804 epoch-50 checkpoint on the same GPU/session.
- Anchors: cases `0273` and `0653`; Q=`8192`; receiver chunk=`2048`.
- Timed inference: full physical forward and prepared P2 decode, two warmups
  and five synchronized repetitions, maps/profiler excluded.
- Timed training: real fixed M1/M12 buckets, B=`48`, Q=`1024`, one warmup and
  three measured forward/backward/clip/update steps.
- Run 1401 and a live Run 1405 comparison are optional historical context, not
  required inputs to this Run-1406 decision.

The paired tooling keeps logical `q -> group -> source` paths separate from
unique executed `q -> source` candidates. It reports P0/P1/P2 (and optional
P2-consistency) logical/unique/multiplicity counts, actual module MLP rows,
environment geometry/content rows, scalar control rows, source projections,
forward calls, checkpoint recomputations, and valid/padded denominators. Any
receiver-chunk numerators and denominators are summed before ratios.

The ledger adapter accepts explicit Thermal raw prefixes
`initial_port_group_control_*` (P0), `provisional_group_control_*` (P1),
unprefixed `group_control_*` (P2), and `port_global_group_control_*`
(P2-consistency). It never infers P0/P1 from a P2-only payload.

All route figures are labelled learned interaction routes, not physical
causality. `R_M/R_E < 1` is not imposed as a continuation gate.

## Exact checkpoint policy

| Label | Path | Policy | Epoch role |
|---|---|---|---|
| 1406 | supplied with `--checkpoint-1406` | explicit CLI path; no silent selection | matched epoch-50 candidate |
| 1804 | supplied with `--checkpoint-1804` | explicit CLI path; no silent selection | matched epoch-50 dense reference |

The live entry point rejects a checkpoint whose trusted payload is not exactly
epoch 50. It loads/releases one model at a time so absolute and incremental
memory peaks are not the sum of resident checkpoints.

## Plan-only command

From `HONF_Proj/`:

```bash
python tools/diagnostics/run_group_control_comparison.py \
  --checkpoint-1406 /absolute/path/run1406_epoch_0050.pt \
  --checkpoint-1804 /absolute/path/run1804_epoch_0050.pt \
  --output diagnostics/generated/run1406/comparison.json \
  --report docs/reports/HONF_Run1406_Group_Control_Development_Report.md \
  --plan-only
```

This validates only explicit paths and the fixed protocol. It does not load
CUDA, checkpoints, datasets, maps, or training.

## Physical comparison command template (not executed in this initial artifact)

```bash
CUDA_VISIBLE_DEVICES=<free-device> \
python tools/diagnostics/run_group_control_comparison.py \
  --checkpoint-1406 /absolute/path/run1406_epoch_0050.pt \
  --checkpoint-1804 /absolute/path/run1804_epoch_0050.pt \
  --metrics-1406 /absolute/path/run1406/metrics.csv \
  --metrics-1804 /absolute/path/run1804/metrics.csv \
  --output diagnostics/generated/run1406/comparison.json \
  --report docs/reports/HONF_Run1406_Group_Control_Development_Report.md \
  --map-dir diagnostics/generated/run1406/maps \
  --device cuda:0
```

The optional `--reverse-order-repeat` requests one bounded reverse-order
repeat when a measured ratio is within its observed spread of the five-percent
speed target. It is not an autotuning sweep.

## Board command template

```bash
python tools/diagnostics/render_group_control_interaction_board.py \
  --comparison diagnostics/generated/run1406/comparison.json \
  --output-dir diagnostics/generated/run1406/board
```

The board writes PNG, PDF, and a provenance JSON. It shows incidence,
detached group centres, query routing, selected actual logical triples, one
unique q-to-source line per fine candidate, moments/overlap labels, the full
phase ledger, and measured decode time from the comparison artifact.

## Predeclared decision evidence

Continuation to total epoch 500 is supported only when the measured artifact
shows all of the following:

1. mean full-forward median across both anchors is at most `0.95` times Dense
   1804;
2. M12 optimizer-step median is at most `0.95` times Dense 1804;
3. allocated peak is at most `1.10` times Dense 1804 on each full-forward and
   M12 workload;
4. supplied epoch-50 learning evidence has finite outputs/parameters/
   gradients, meaningful updates, improving validation trajectory, and no
   unresolved catastrophic instability; and
5. the backend ledger confirms one fine call per unique q-to-source candidate
   where actual phase records are available.

These are research-budget criteria, not a permanent CI/model-validity gate.
If evidence is unavailable or borderline, leave the run at epoch 50 and ask
for a decision; do not manufacture a pass. The comparison reports medians,
min-to-max timing spread, pre-call allocated/reserved baseline, absolute and
incremental allocated/reserved peaks, and the reverse-order repeat separately.

## Pending execution fields

- Source commit/profile: pending physical run.
- Actual Run 1406 and Dense 1804 checkpoint paths: supplied at execution.
- Same-GPU device/session: pending physical run.
- Full/P2 medians and spread: pending physical run.
- M1/M12 step medians and spread: pending physical run.
- P0/P1/P2 ledger and boards: pending untimed debug pass.
- Epoch-50 decision and any continuation command: pending measured evidence.
- Material deviations: none in this initial plan-only artifact.
