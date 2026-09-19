# HONF routing evaluation at epoch 2500

This report compares ThermalChannel Runs 2001 and 2101 at the common exact
epoch-2500 budget and at stable snapshots of their saved field-best
checkpoints. Runs 1401 and 1804 are exact-2500 references. It assesses
accuracy, parameter/checkpoint footprint, controlled training and inference
cost, and whether the routed models realize the sparse execution promised by
the [implementation plan](../../UpgradePlan/HONF_Runs_2000_2100_2200_Codex_Implementation_Plan.md).

## Decision

**The routed architecture is accurate and reduces peak activation memory, but
the scientific sparse-compute hypothesis has not materialized.** Run 2101's
saved-best snapshot is the most accurate evaluated policy, yet both routers
retain essentially every final module and environment micro-interaction on
the endpoint anchors. The optimized dense-QE path is therefore working as
designed; on those anchors and nearly every Run 2101 training row it exposes
a complete realized environmental support. Run 2001 has a limited partial-QE
regime in the mixed M12 batch, but not enough consistent sparsity to produce
a speed gain.

The next model change should separate two goals:

1. add a pre-join complete-support shortcut to avoid enumerating and
   coalescing paths that are already known to be dense; and
2. run a new science-facing routing experiment whose loss and selection
   policy explicitly reward *realized post-coalescing pair sparsity*, with
   anti-collapse/load-balance protection and an omitted-source error audit.

The first item can reduce compiler overhead but cannot validate the sparse
scientific hypothesis. The second is the required experiment. The ongoing
2001/2101 continuations were not stopped or modified; reaching epoch 5000 is
still useful for accuracy maturity, but additional epochs alone do not alter
the current execution mechanism.

## Scope, policies, and evidence boundary

The population is the complete 90-case ThermalChannel development holdout,
with 8,192 field queries per case and predicted port conditions. It is not an
untouched test set and this report contains no new CFD comparison. Physical
metric rows are evaluator/surrogate quantities and are not relabeled as
physical truth.

| Policy | Architecture / router | Checkpoint epoch | Policy role |
|---|---|---:|---|
| Run 1401 exact | legacy HONF | 2500 | common-budget reference |
| Run 1804 exact | dense pairwise field | 2500 | common-budget dense reference |
| Run 2001 exact | routed pairwise / module hubs | 2500 | common-budget candidate |
| Run 2001 saved best | routed pairwise / module hubs | 2425 | validation-selected snapshot |
| Run 2101 exact | routed pairwise / mean shift | 2500 | common-budget candidate |
| Run 2101 saved best | routed pairwise / mean shift | 2732 | validation-selected snapshot |

Saved-best files were fingerprinted immediately around their evaluations.
The selected rows are not substituted into exact-2500 comparisons.

## Accuracy on all 90 cases

Pooled relative L2 is

\[
\sqrt{\frac{\sum_i \mathrm{SSE}_i}{\sum_i \mathrm{target\_SSE}_i}},
\]

using the same target denominator and value count for every policy.

| Policy | Pooled relative L2 | Equal-case mean | Median | p95 | Worst case (L2) |
|---|---:|---:|---:|---:|---|
| Run 1401 exact | 0.045285 | 0.044643 | 0.041110 | 0.062338 | 0283 (0.069210) |
| Run 1804 exact | 0.048835 | 0.048109 | 0.046054 | 0.066788 | 0298 (0.086315) |
| Run 2001 exact | **0.044960** | **0.043218** | **0.038646** | **0.069128** | 0297 (0.081208) |
| Run 2001 saved best | 0.036170 | **0.033113** | **0.028712** | 0.061395 | 0298 (0.077733) |
| Run 2101 exact | 0.046395 | 0.044950 | 0.041214 | 0.072798 | 0298 (0.087336) |
| Run 2101 saved best | **0.035851** | 0.033130 | 0.029054 | **0.060861** | 0284 (0.071849) |

Run 2001 exact is the strongest common-budget endpoint and beats the dense
reference on 69 of 90 paired cases. Run 2101 exact lies between the two
references. Both saved-best snapshots are materially stronger and nearly
tied: Run 2101 has the lower pooled error, while Run 2001 has the slightly
lower equal-case mean and median. This demonstrates that both routed models
can fit the field well. It does not show that routing is sparse: the selected
policies retain essentially the same dense final support.

Paired case deltas below are candidate minus Run 1804 exact; negative favors
the routed candidate.

| Candidate | Mean delta | Median delta | Wins / losses |
|---|---:|---:|---:|
| Run 2001 exact | -0.004891 | -0.005713 | 69 / 21 |
| Run 2001 saved best | -0.014996 | -0.015279 | 86 / 4 |
| Run 2101 exact | -0.003159 | -0.003576 | 67 / 23 |
| Run 2101 saved best | -0.014979 | -0.017628 | 83 / 7 |

### Pooled normalized fluid channels

| Policy | u | v | p | omega | temperature |
|---|---:|---:|---:|---:|---:|
| Run 1401 exact | 0.031429 | 0.027785 | 0.048252 | 0.054427 | 0.059894 |
| Run 1804 exact | 0.021916 | 0.023961 | 0.040489 | 0.053198 | 0.084955 |
| Run 2001 exact | 0.026901 | **0.021150** | **0.039664** | **0.052743** | 0.071458 |
| Run 2001 saved best | **0.019221** | **0.019461** | 0.034037 | 0.049842 | **0.047164** |
| Run 2101 exact | 0.028092 | 0.027858 | 0.041302 | 0.057199 | 0.067374 |
| Run 2101 saved best | **0.019692** | **0.019810** | **0.033029** | **0.048901** | **0.047309** |

The full per-case, physical-field, KPI, tail, and stratum tables are retained
in the generated evaluation directories. They are preferable to a single
aggregate when inspecting pressure drop, interface flux, or module-count
tails.

## Model and checkpoint footprint

| Model | Total parameters | Trainable parameters | Core parameters | Full checkpoint |
|---|---:|---:|---:|---:|
| Run 1401 | 3,508,649 | 2,473,510 | 1,693,214 | 30.18 MiB |
| Run 1804 | 5,430,548 | 4,395,409 | 3,615,113 | 53.77 MiB |
| Run 2001 | 5,540,405 | 4,505,266 | 3,724,970 | 55.06 MiB |
| Run 2101 | 5,540,405 | 4,505,266 | 3,724,970 | 55.05–55.06 MiB |

Routing does not reduce parameter storage. The routed model has 1.58 times
the total and 1.82 times the trainable parameter count of Run 1401, and is
slightly larger than the dense Run 1804 model.

## Controlled inference cost

Protocol: physical GPU 0; real cases 0273 and 0653; 8,192 queries; receiver
chunk 2,048; query batch 32,768; two warmups and five synchronized CUDA
repetitions. Values are means of the two per-case medians. Incremental peak is
allocated memory above the model/input baseline during the measured phase.

| Policy | Full forward | Prepared decode | Incremental peak allocated | Peak allocated |
|---|---:|---:|---:|---:|
| Run 1401 exact | 26.00 ms | 6.52 ms | 362.4 MiB | 426.1 MiB |
| Run 1804 exact | 35.44 ms | 13.01 ms | 430.9 MiB | 494.6 MiB |
| Run 2001 exact | 77.68 ms | 34.78 ms | 317.2 MiB | 346.8 MiB |
| Run 2001 saved best | 78.67 ms | 35.85 ms | 316.8 MiB | 346.3 MiB |
| Run 2101 exact | 79.24 ms | 35.95 ms | 315.8 MiB | 379.5 MiB |
| Run 2101 saved best | 88.65 ms | 36.94 ms | 316.1 MiB | 345.7 MiB |

The routed models reduce incremental activation allocation by about 27%
relative to Run 1804, but exact Run 2101 is 2.24 times slower for a full
forward and 2.76 times slower for prepared decode. This is a useful memory
tradeoff, not a sparse execution speedup.

## Controlled training-step cost

Protocol: batch 48, 1,024 queries, one warmup, three synchronized
forward/backward/optimizer-step repetitions, restored optimizer state, and
the same compiler+dense-QE variant.

| Policy | M1 median / peak allocated | M12 median / peak allocated |
|---|---:|---:|
| Run 1401 exact | 0.303 s / 2,784.8 MiB | 0.591 s / 24,559.8 MiB |
| Run 1804 exact | 0.991 s / 5,849.1 MiB | 2.187 s / 26,782.7 MiB |
| Run 2001 exact | 2.065 s / 4,272 MiB | 5.854 s / 21,774 MiB |
| Run 2101 exact | 2.000 s / 4,269.1 MiB | 5.267 s / 20,691.6 MiB |

Run 2101 again exchanges time for memory: it is about 2.0 times slower than
Run 1804 for M1 and 2.4 times slower for M12, while using about 27% and 23%
less peak allocated memory respectively.

## Sparse-routing execution audit

For each phase and source type:

\[
R_{\rm duplicate}=\frac{P_{\rm raw}}{P_{\rm unique}},\qquad
R_{\rm module}=\frac{N_{QM,\rm unique}}{QM},\qquad
R_{\rm env}=\frac{N_{QE,\rm unique}}{QE}.
\]

The primary completeness quantity is the fraction of receiver rows whose
post-coalescing unique QE pair count equals E:

\[
R_{\rm completeQE}=
\frac{\text{receiver rows using complete QE}}
     {\text{all receiver rows}}.
\]

The diagnostic separately records eligibility and actual dense-path use. The
backend dispatches the complete reader by whole case/batch row, so a receiver
that is individually complete inside an otherwise partial case is not counted
as actual dense-path use.

### Five-anchor P2 field result

Anchors 0273, 0653, 0283, 0298, and 0302 were evaluated at P0, P1, and P2.
The compact table below reports the P2 field phase; the generated ledgers
retain every phase.

| Policy | Module Rduplicate | Environment Rduplicate | Rmodule | Renv | RcompleteQE |
|---|---:|---:|---:|---:|---:|
| Run 2001 exact | 1.333–1.857 (mean 1.635) | 2.406–3.455 (mean 2.975) | 0.978–1.000 | 1.000 | 1.000 |
| Run 2001 saved best | 1.333–1.857 (mean 1.634) | 2.365–3.517 (mean 3.046) | 0.982–1.000 | 1.000 | 1.000 |
| Run 2101 exact | 2.571–3.000 (mean 2.737) | 3.000–4.630 (mean 3.857) | 1.000 | 1.000 | 1.000 |
| Run 2101 saved best | 2.571–3.000 (mean 2.806) | 3.000–4.167 (mean 3.752) | 1.000 | 1.000 | 1.000 |

The optimizer batches provide a broader execution sample. Run 2001 has
RcompleteQE = 528/528 = 1.000 for M1 but 930/1104 = 0.8424 for M12: 15.8%
of receiver rows are partial and use the packed fallback. Run 2101 has
RcompleteQE = 528/528 = 1.000 for M1 and 1100/1104 = 0.99638 for M12, with
module Rduplicate 3.59–3.60 and environment Rduplicate 5.63–5.64.

Run 2001 therefore has a small but real partial-QE regime on the mixed M12
training batch. It is not yet a consistent scalable sparse regime: the five
endpoint anchors have Renv=1, Rmodule remains approximately one, and measured
latency is still more than twice the dense reference. Run 2101 is even more
clearly operating on complete QE.

### Intended-feature verdict

| Design feature | Implementation status | Scientific/execution verdict |
|---|---|---|
| candidate hubs are descriptors, not field values | working | figures and exports preserve this separation |
| two-hop route is compiled before fine response | working | raw and coalesced pair ledgers are populated |
| duplicate pairs are coalesced | working | exact outputs use unique receiver/source keys |
| delayed fine evaluation uses selected unique pairs | working mechanically | selection covers essentially the full dense support |
| module micro-interactions become sparse | not realized | Rmodule is approximately one |
| environment micro-interactions become sparse | not realized | Renv is one |
| complete QE uses the exact dense fast path | working | RcompleteQE is approximately one, as expected |
| routing provides a latency scalability gain | not realized | compiler duplication and route preparation dominate |
| clusters establish physical influence regions | not established | learned route weights are not physical attribution and no new CFD/barrier validation was performed |

Mean-shift hubs do move and source incidence matrices contain exact zeros, but
those are intermediate structural facts. Every query activates every active
hub in the audited endpoint, and the union through those hubs reaches every
physical module and environment source. A visually organized incidence map
therefore does not imply sparse fine execution.

## Comparative hypergraph visualization

The exact-2500 comparative board uses cases 0273 and 0283 and places module
hubs and mean-shift side by side for geometry, source-to-hub incidence, and a
selected P2 receiver's deduplicated fine pairs:

- [comparative PNG](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/epoch2500_profile/comparative_hypergraph_board/comparative_hypergraph_board.png)
- [editable PDF](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/epoch2500_profile/comparative_hypergraph_board/comparative_hypergraph_board.pdf)
- [provenance and ratios](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/epoch2500_profile/comparative_hypergraph_board/comparative_hypergraph_board.json)

The selected-pair panels are especially important: sparse source-to-hub
incidence can coexist with a dense receiver/source union. The lines show
deduplicated execution pairs, not physical influence.

## Recommended next implementation

### 1. Remove known-dense compiler work

When source incidence and query support imply complete QE, compute the exact
dense prior directly (the equivalent of the query/source product) and enter
the dense-QE reader *before* enumerating two-hop paths. Apply the same idea to
complete QM where worthwhile. This targets Rduplicate and should be measured
against the current exact path. It is an execution optimization only.

### 2. Train for realized sparse support

The next routing loss should act on the quantities that determine the final
union, not only on attractive hub geometry. A practical differentiable
starting point is to encourage concentrated source and query memberships,
for example by maximizing row-wise squared mass
\(\lVert A_{n,:}\rVert_2^2\) and \(\lVert d_{q,:}\rVert_2^2\), while adding a
load-balance/coverage term so all traffic does not collapse to one hub. The
validation reducer should report accuracy jointly with Rmodule, Renv,
RcompleteQE, and Rduplicate; checkpoint selection should expose their Pareto
tradeoff instead of optimizing field MSE alone.

This is a proposed experiment, not a claim that a particular regularization
weight or sparsity target is already known. Use a bounded controlled ablation
before a full continuation, and require an omitted-source response audit to
show that removed interactions do not carry material prediction signal.

### 3. Preserve interpretability boundaries

Use geometry-aware compact support or module-tied candidates where it is
scientifically justified, but continue to label route weights as learned
routing quantities. Physical-cluster claims require external CFD/barrier
evidence; the present surrogate holdout cannot supply it.

## Artifacts and reproducibility

Primary generated roots:

- `diagnostics/generated/accuracy_eval_run2001_2101_epoch2500_20260917/`
- `diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/epoch2500_profile/`

Reusable diagnostics added or extended by this evaluation:

- `tools/diagnostics/run_dynamic_sparse_routing_study.py`
- `tools/diagnostics/render_routing_diagnostics.py`
- `tools/diagnostics/render_comparative_hypergraph_board.py`

Focused routing/renderer tests pass. Generated ledgers include exact
checkpoint identity, selected-case maps, the four execution ratios, dense
fast-path eligibility/use, strategy metadata, and figure provenance.

## Limitations

- Accuracy uses the established 90-case development holdout, not an untouched
  reserved test population.
- Timing is hardware- and protocol-specific; all comparative rows use the
  same GPU/protocol, while live epoch telemetry is reported only as context.
- A saved-best file is mutable during training. The evaluated snapshots were
  fingerprinted and are identified by epoch/hash; later training may replace
  the on-disk `best_by_field_mse_model.pt`.
- Selected-case hypergraph figures establish route/compiler structure, not
  physical causality.
- The ongoing runs were observed, not modified or resumed by this evaluation.
