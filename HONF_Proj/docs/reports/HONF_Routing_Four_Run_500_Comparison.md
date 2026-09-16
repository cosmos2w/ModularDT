# HONF four-run routing comparison at epoch 500

This report compares the completed ThermalChannel runs 1401, 1804, 2000,
and 2100 at the common 500-epoch assessment budget. It was generated on
2026-09-16 from the run manifests, metric histories, exact endpoint tables,
five-anchor routing ledgers, and the bounded Run 2100 mean-shift diagnostic.
The reducer is [analyze_routing_500_comparison.py](../../tools/diagnostics/analyze_routing_500_comparison.py).

The exact epoch-500 policy is the primary comparison for all four runs. A
validation-selected policy is included only when its checkpoint is available
within the 500-epoch budget. Run 2000's selected checkpoint is epoch 466 and
Run 2100's is epoch 467. The corresponding Run 1401 and Run 1804 selected
checkpoint files through epoch 500 are unavailable; their later mature-best
files are not substituted.

## Inputs and population policy

All four run manifests are `completed`. The common training configuration is
seed 0, batch size 48, learning rate `3e-4`, weight decay `1e-5`, gradient
clipping 1, and no AMP. Each run used 600 training cases and 90 held-out
cases. The endpoint workflow uses all 90 cases, 8,192 grid queries per case,
predicted ports, and receiver chunk 128. The split is the established
development holdout and is not an untouched CFD test set.

| Run | Model / router | Recorded terminal epochs | Exact checkpoint | Selected <=500 policy |
|---|---|---:|---|---|
| 1401 | legacy HONF / fixed projection | 5,000 | epoch 500 | unavailable (metric minimum epoch 466) |
| 1804 | dense pairwise field | 5,000 | epoch 500 | unavailable (metric minimum epoch 493) |
| 2000 | routed pairwise HONF / module hubs | 500 | epoch 500 | epoch 466, available |
| 2100 | routed pairwise HONF / mean shift | 500 | epoch 500 | epoch 467, available |

The terminal epoch is provenance for the run directory. Every exact endpoint
and parameter inventory in this comparison uses the epoch-500 checkpoint, so
the four exact rows share the same 500-epoch assessment budget. The 5,000-epoch
parent continuation is relevant only to the availability of mature parent
selection files; it is not used in the exact-500 endpoint comparison.

The full run and checkpoint inventory is in
[selection_summary.csv](../../diagnostics/generated/interface_operator_study/routing_500_comparison/selection_summary.csv).
The source run directories and manifests are recorded in
[summary.json](../../diagnostics/generated/interface_operator_study/routing_500_comparison/summary.json).
The generated convergence view uses log-scaled raw validation curves with
trailing-50 medians, logged per-epoch latency, and logged peak CUDA memory in
MiB; it is
[convergence_500.png](../../diagnostics/generated/interface_operator_study/routing_500_comparison/convergence_500.png).

## Convergence and budget

The selection metric is the logged validation field MSE. “Best <=500” is the
minimum over metric rows with epoch at most 500; the late-window values and
least-squares slopes describe stability around the endpoint.

| Run | Best <=500 (epoch) | Exact-500 val MSE | Previous-50 median | Last-50 median | Last-50 minus previous-50 | Last-100 median | Last-50 slope / epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1401 | 0.018412 (466) | 0.023596 | 0.030321 | 0.022987 | -0.007334 | 0.028219 | -1.66e-05 |
| 1804 | 0.013392 (493) | 0.016619 | 0.019425 | 0.015969 | -0.003456 | 0.017793 | +1.22e-05 |
| 2000 | 0.011583 (466) | 0.022513 | 0.018312 | 0.015875 | -0.002437 | 0.017271 | +9.69e-05 |
| 2100 | 0.013134 (467) | 0.013742 | 0.020542 | 0.016100 | -0.004442 | 0.018432 | +1.09e-04 |

Each run's last-50 median is below its preceding-50 median, but the positive
routed late-50 slopes and the difference between exact and selected rows show
that the last epoch is a noisy snapshot. Run 2000's exact-500 validation value
is above its late-window levels and should be treated as an endpoint
fluctuation. It is not evidence that the run failed; its selected epoch-466
checkpoint is the relevant available selection policy. Five hundred epochs is
therefore an early assessment rather than a convergence certificate.

The measured wall fields are architecture- and instrumentation-dependent. Run 1804 recorded
82,881.62 s training plus 9,107.66 s validation (91,989.28 s total); Run 2000
recorded 25,996.14 s plus 1,240.81 s (27,236.96 s); Run 2100 recorded
27,506.49 s plus 1,300.69 s (28,807.18 s). The Run 1401 summary has no
comparable wall fields. The later 5,000-epoch parent continuation affects
which mature selection files exist, while the exact endpoint comparison uses
500 epochs for every run. These values therefore do not establish an
efficiency ranking.

Strict CPU reconstruction of all four exact epoch-500 checkpoints verified the
following counts. Total includes the frozen local module; trainable uses
`requires_grad=True`; core is the `core.*` subset.

| Run | Architecture / strategy | Total | Trainable | Core | Checkpoint bytes |
|---|---|---:|---:|---:|---:|
| 1401 | legacy | 3,508,649 | 2,473,510 | 1,693,214 | 31,643,874 |
| 1804 | dense | 5,430,548 | 4,395,409 | 3,615,113 | 56,380,040 |
| 2000 | routed / module hubs | 5,540,405 | 4,505,266 | 3,724,970 | 57,723,080 |
| 2100 | routed / mean shift | 5,540,405 | 4,505,266 | 3,724,970 | 57,723,144 |

The machine-readable verification is
[parameter_inventory_epoch500.json](../../diagnostics/generated/interface_operator_study/routing_500_comparison/parameter_inventory_epoch500.json)
and [parameter_inventory_epoch500.csv](../../diagnostics/generated/interface_operator_study/routing_500_comparison/parameter_inventory_epoch500.csv).
The routed model is about 1.58 times the Legacy total parameter count and
1.82 times its trainable count. Execution sparsity does not reduce this
parameter cost.

## Full 90-case endpoint

The endpoint table uses the evaluator's dataset-normalized fluid field error.
Pooled relative L2 is

\[
\sqrt{\frac{\sum_i \mathrm{SSE}_i}{\sum_i \mathrm{target\_SSE}_i}},
\]

with the same target denominator and value count for each policy. Equal-case
mean, median, and p95 are summaries of per-case relative L2 values.

| Policy | Checkpoint epoch | Pooled relative L2 | Equal-case mean | Median | p95 | Worst case (L2) |
|---|---:|---:|---:|---:|---:|---|
| Legacy 1401 exact | 500 | 0.117148 | 0.113449 | 0.110861 | 0.140883 | 0689 (0.150238) |
| Dense 1804 exact | 500 | 0.098741 | 0.096107 | 0.096741 | 0.112534 | 0298 (0.124763) |
| Module hubs 2000 exact | 500 | 0.123566 | 0.117535 | 0.114840 | 0.162006 | 0292 (0.171768) |
| Module hubs 2000 selected | 466 | 0.079194 | 0.076952 | 0.077227 | 0.093776 | 0286 (0.105650) |
| Mean shift 2100 exact | 500 | 0.087968 | 0.085791 | 0.085994 | 0.102032 | 0687 (0.111554) |
| Mean shift 2100 selected | 467 | 0.085573 | 0.083575 | 0.084045 | 0.098571 | 0298 (0.109553) |

The machine-readable headline and all per-case rows are
[endpoint_headline.csv](../../diagnostics/generated/interface_operator_study/routing_500_comparison/endpoint_headline.csv)
and [endpoint_paired_cases.csv](../../diagnostics/generated/interface_operator_study/routing_500_comparison/endpoint_paired_cases.csv).
The reducer checked six 90-case populations, 25,200 target/count values, and
1,800 stratum values with no mismatch. That check covers matching case IDs,
target energies/counts, and the four case strata.

The selected routed values are lower than the Dense exact-500 value, but that
comparison has a policy difference: no parent selected-through-500 weights are
available. It is not evidence that a selected routed model fairly beats a
selected Dense model. The exact policies provide the matched epoch-500
comparison: 2100 is lower than Dense 1804 on pooled fluid relative L2, while
2000 is higher; Run 2000's selected-vs-exact change is large enough that
checkpoint policy must remain visible in every conclusion.

### Normalized field channels

The fluid channel relative L2 values below are pooled over all 90 cases. The
full normalized set also contains global all-region, near-interface,
far-fluid, local-radius, and outside-local-radius rows in
[endpoint_channel.csv](../../diagnostics/generated/interface_operator_study/routing_500_comparison/endpoint_channel.csv).

| Fluid channel | 1401 exact | 1804 exact | 2000 exact | 2000 selected | 2100 exact | 2100 selected |
|---|---:|---:|---:|---:|---:|---:|
| u | 0.081093 | 0.051778 | 0.060576 | 0.038662 | 0.054133 | 0.058510 |
| v | 0.077416 | 0.103761 | 0.103082 | 0.059311 | 0.071215 | 0.065094 |
| p | 0.103617 | 0.095269 | 0.127560 | 0.085957 | 0.096575 | 0.092481 |
| omega | 0.155603 | 0.113666 | 0.127703 | 0.106124 | 0.110638 | 0.108386 |
| temperature | 0.145407 | 0.113344 | 0.181904 | 0.088619 | 0.097403 | 0.094973 |

### Physical fields and scalar physics

Physical rows remain in the evaluator's native units and are not combined
with the dataset-normalized field table. The table below reports pooled
physical relative L2, with all 12 physical field/interface/port metrics
available in [endpoint_physical.csv](../../diagnostics/generated/interface_operator_study/routing_500_comparison/endpoint_physical.csv).

| Physical metric | 1401 exact | 1804 exact | 2000 exact | 2000 selected | 2100 exact | 2100 selected |
|---|---:|---:|---:|---:|---:|---:|
| u fluid | 0.045986 | 0.029362 | 0.034351 | 0.021924 | 0.030698 | 0.033179 |
| v fluid | 0.077416 | 0.103761 | 0.103082 | 0.059311 | 0.071215 | 0.065094 |
| p fluid | 0.096772 | 0.088976 | 0.119133 | 0.080278 | 0.090195 | 0.086371 |
| omega fluid | 0.155603 | 0.113666 | 0.127703 | 0.106124 | 0.110638 | 0.108386 |
| temperature fluid | 0.115753 | 0.090229 | 0.144808 | 0.070547 | 0.077539 | 0.075605 |
| internal temperature | 0.071717 | 0.069485 | 0.067453 | 0.038974 | 0.041336 | 0.040642 |
| interface surface temperature | 0.091510 | 0.087244 | 0.093675 | 0.053000 | 0.056047 | 0.053975 |
| interface normal heat flux | 0.216907 | 0.220073 | 0.223891 | 0.215687 | 0.211276 | 0.216663 |
| final port environment temperature | 0.088037 | 0.068922 | 0.140325 | 0.069840 | 0.079999 | 0.085008 |
| provisional port environment temperature | 0.345334 | 0.109385 | 0.088637 | 0.103650 | 0.125084 | 0.138892 |
| final effective h | 0.042249 | 0.041631 | 0.055933 | 0.041951 | 0.044677 | 0.045333 |
| provisional effective h | 0.610314 | 0.643571 | 0.772646 | 0.783942 | 0.761325 | 0.764256 |

The scalar pressure-drop, outlet-temperature, and active-module-temperature
KPIs, including mean absolute and relative errors, are in
[endpoint_kpi.csv](../../diagnostics/generated/interface_operator_study/routing_500_comparison/endpoint_kpi.csv).
For example, exact-500 pressure-drop mean absolute error is 0.009129 for
1401, 0.007186 for 1804, 0.006414 for 2000, and 0.004093 for 2100. These are
surrogate/evaluator outputs, not new CFD truth.

### Strata and tails

[endpoint_strata.csv](../../diagnostics/generated/interface_operator_study/routing_500_comparison/endpoint_strata.csv)
contains 78 policy-by-stratum rows for module count, spacing, wall proximity,
and heating heterogeneity. [endpoint_tail.csv](../../diagnostics/generated/interface_operator_study/routing_500_comparison/endpoint_tail.csv)
retains the ten worst cases for every policy, including case IDs, geometry
strata, and active module counts. The headline worst cases are 0689 for 1401,
0298 for 1804, 0292 for 2000 exact, 0286 for 2000 selected, 0687 for 2100
exact, and 0298 for 2100 selected.

### Paired full-population deltas

The paired files use candidate minus baseline per-case fluid relative L2; a
negative mean favors the candidate. Wins and losses are counts over the same
90 case IDs.

| Candidate | Baseline | Mean delta | Median delta | Wins / losses |
|---|---|---:|---:|---:|
| 2000 exact | 1401 exact | +0.004086 | +0.002860 | 37 / 53 |
| 2000 exact | 1804 exact | +0.021428 | +0.020720 | 8 / 82 |
| 2000 selected | 1401 exact | -0.036497 | -0.034115 | 90 / 0 |
| 2000 selected | 1804 exact | -0.019154 | -0.019469 | 89 / 1 |
| 2100 exact | 1401 exact | -0.027658 | -0.025017 | 90 / 0 |
| 2100 exact | 1804 exact | -0.010316 | -0.010679 | 84 / 6 |
| 2100 selected | 1401 exact | -0.029873 | -0.027857 | 90 / 0 |
| 2100 selected | 1804 exact | -0.012531 | -0.011891 | 87 / 3 |

These paired counts describe endpoint policy sensitivity on the development
holdout. They do not turn a selected checkpoint into an independent test.

## Routing support and cost

The common ledgers use five anchors (0273, 0653, 0283, 0298, and 0302), 8,192
queries, and the exact epoch-500 checkpoint for each routed run. `M_active` is
the number of physical present modules; `Mpack` is the padded encoded width.
Raw paths are positive two-hop paths before receiver/source coalescing.
Unique pairs are the fine pairs actually retained after coalescing. The ratio
below is unique pairs divided by the phase's dense reference, averaged over
anchors.

| Run | Phase | M active | K | Module raw -> unique | Module unique / dense | Environment raw -> unique | Environment unique / dense |
|---|---|---:|---:|---:|---:|---:|---:|
| 2000 | P0 port | 5.4 | 5.4 | 6,512 -> 3,800 | 0.935 | 418,467 -> 147,446 | 1.000 |
| 2000 | P1 refinement | 5.4 | 5.4 | 6,149 -> 3,709 | 0.918 | 421,380 -> 147,455 | 1.000 |
| 2000 | P2 field | 5.4 | 5.4 | 69,566 -> 41,496 | 0.952 | 4,497,451 -> 1,572,856 | 1.000 |
| 2100 | P0 port | 5.4 | 5.4 | 10,598 -> 4,147 | 1.000 | 506,726 -> 147,456 | 1.000 |
| 2100 | P1 refinement | 5.4 | 5.4 | 10,445 -> 4,147 | 1.000 | 508,109 -> 147,456 | 1.000 |
| 2100 | P2 field | 5.4 | 5.4 | 111,411 -> 44,237 | 1.000 | 5,412,683 -> 1,572,864 | 1.000 |

The complete anchor-level and aggregate tables are
[routing_support_anchors.csv](../../diagnostics/generated/interface_operator_study/routing_500_comparison/routing_support_anchors.csv)
and [routing_support.csv](../../diagnostics/generated/interface_operator_study/routing_500_comparison/routing_support.csv).
The routed five-anchor P2 environment support is therefore essentially the
dense reference, and module support is also close to the dense reference.
Raw-path duplication is visible in the raw-to-unique gap. These ledgers show
support and dispatch structure; they are not a measured wall-clock speedup.
The mean-shift and module-hubs frozen interventions have identical P2 fine
pair counts on every anchor in the candidate table, so mean shift shows no
measured fine-support reduction or cost gain in this diagnostic.

The ledger run omitted route-map export because the existing CUDA map-to-NumPy
conversion path fails. The support/count rows remain valid for the requested
ledger scope; no shared tool was modified to hide that failure.

## Mean-shift candidate diagnostic

The bounded Run 2100 diagnostic uses the exact epoch-500 checkpoint, five
anchors, and 32 query points per anchor. It compares mean shift with a
temporary module-hubs candidate intervention under the same model weights.
The intervention is not a separately trained accuracy result. The full source
JSON is [mean_shift_epoch500.json](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2100/routing/mean_shift_epoch500.json),
and its reduced table is
[routing_candidate_intervention.csv](../../diagnostics/generated/interface_operator_study/routing_500_comparison/routing_candidate_intervention.csv).

Candidate coordinates move for three fixed-data iterations. Physical drift is
reported in physical coordinates; scaled drift divides each coordinate by the
adapter routing length `ell`; joint drift uses the actual dimensionless
\([x/\ell,\;descriptor/feature\_bandwidth]\) state. The run reports
`ell=[1.8,1.8]`. The scaled tolerance is the conservative scalar convention
`0.05 / max(ell) = 0.0277778`; it is not an anisotropic distance equivalence.
All approximate mode counts below are tolerance-labelled connected-component
diagnostics only and do not merge candidates or change execution.

| Case | Valid K | Candidate time (s, mean) | Physical drift; moves 1/2/3 | Scaled drift; moves 1/2/3 | Joint drift; moves 1/2/3 | Final min separation (physical / scaled / joint) | Approx. modes by iteration |
|---|---:|---:|---|---|---|---|---|
| 0273 | 3 | 0.000518 | 0.935; 0.497/0.267/0.171 | 0.519; 0.276/0.148/0.095 | 0.771; 0.410/0.220/0.142 | 0.432 / 0.240 / 0.300 | 3,3,3,3 |
| 0653 | 5 | 0.000520 | 0.845; 0.429/0.249/0.169 | 0.470; 0.238/0.138/0.094 | 0.706; 0.390/0.193/0.127 | 0.117 / 0.065 / 0.080 | 5,5,5,5 |
| 0283 | 5 | 0.000531 | 0.773; 0.479/0.191/0.107 | 0.430; 0.266/0.106/0.059 | 0.602; 0.377/0.146/0.082 | 0.008 / 0.004 / 0.005 | 5,5,4,4 |
| 0298 | 7 | 0.000547 | 0.908; 0.470/0.260/0.192 | 0.504; 0.261/0.144/0.107 | 0.711; 0.397/0.190/0.138 | 0.463 / 0.257 / 0.289 | 7,7,7,7 |
| 0302 | 7 | 0.000533 | 1.045; 0.490/0.317/0.250 | 0.580; 0.272/0.176/0.139 | 0.783; 0.407/0.223/0.170 | 0.236 / 0.131 / 0.153 | 7,7,7,7 |

Final descriptor concentration is retained in the CSV for every space. The
final descriptor norm mean ranges from 0.570 to 0.760, and the final
descriptor distance-to-centroid mean ranges from 0.251 to 0.520 across these
anchors. The decreasing per-iteration movements show numerical settling, but
the minimum separation can become very small for case 0283; no discrete mode
count should be inferred from it.

| Case | Mean-shift query fluid relL2 | Module-hubs intervention relL2 | Module-hubs minus mean-shift relL2 | P2 module unique pairs (MS / MH) | P2 environment unique pairs (MS / MH) |
|---|---:|---:|---:|---:|---:|
| 0273 | 0.071782 | 0.116212 | +0.044430 | 96 / 96 | 6,144 / 6,144 |
| 0653 | 0.068705 | 0.085123 | +0.016418 | 160 / 160 | 6,144 / 6,144 |
| 0283 | 0.062691 | 0.063533 | +0.000842 | 160 / 160 | 6,144 / 6,144 |
| 0298 | 0.109428 | 0.118840 | +0.009412 | 224 / 224 | 6,144 / 6,144 |
| 0302 | 0.072956 | 0.086658 | +0.013701 | 224 / 224 | 6,144 / 6,144 |

The intervention suggests that candidate movement changes the frozen forward
on these anchors, with lower query-subset fluid error for mean shift in each
case. It does not establish trained Run 2100 accuracy, generalization, or a
runtime gain. The common fine support remains dominated by environmental
pairs, and the candidate-generation timing is only the repeated builder scope
recorded in the diagnostic.

## Interpretation and reproducibility

The accuracy assessment combines the common 500-epoch exact rows, the
through-500 selected rows where weights are available, late-window behavior,
and the routing support/cost evidence.

| Criterion | Evidence | Preliminary judgment |
|---|---|---|
| Fluid-field accuracy | Mean shift 2100 is lower than Dense 1804 at exact 500 (0.087968 vs 0.098741) and selected 467 (0.085573); module hubs 2000 is higher at exact 500 (0.123566) but lower at selected 466 (0.079194). | Accuracy is promising for 2100 across both available policies; 2000 is promising only conditional on checkpoint selection and has a large endpoint fluctuation. |
| Convergence | Both routed runs improve from their preceding-50 to last-50 medians, while their positive last-50 slopes and 2000 exact endpoint outlier show remaining instability. | Treat 500 as an early assessment. |
| Fine sparsity and cost | P2 unique support is close to dense, environment support is effectively dense, and routed models have 5.54M total / 4.51M trainable parameters. | Criterion is not met by the measured endpoint ledgers. |
| Mean-shift intervention | Same-weight mean shift changes frozen predictions but has the same measured fine support as module hubs on all five anchors. | No measured fine-support improvement or cost gain; this does not rank trained strategies. |
| Scalar boundary physics | At exact 500, 2100 lowers fluid field L2 versus Dense, while final port environment-temperature and effective-h errors are 0.079999 / 0.044677 versus 0.068922 / 0.041631 for Dense. | Final port / h metrics regress despite the field gain. |

The missing parent selected-through-500 weights prevent a selected-policy
comparison across all four runs. The endpoint tail, channel, physical,
strata, KPI, paired, and routing-support outputs should be read alongside the
headline rather than replaced by one aggregate score.

The endpoint's normalized metrics apply the checkpoint dataset field
mean/std to prediction and target, while physical metrics retain native
denormalized quantities. A normalized field score is not a physical CFD error,
and learned route weights/support are not physical influence maps. The
current data and surrogate outputs provide no new solver-labelled barrier or
CFD validation evidence.

The exact CPU reducer command is recorded in
[executed_command.txt](../../diagnostics/generated/interface_operator_study/routing_500_comparison/executed_command.txt).
The endpoint, ledger, mean-shift, map-export failure, and CPU parameter
inventory commands are recorded in
[executed_commands.md](../../diagnostics/generated/interface_operator_study/routing_500_comparison/executed_commands.md).
The reducer reads existing artifacts only; the parameter verification strictly
reconstructs existing checkpoints on CPU. Neither performs training or
architecture changes.
