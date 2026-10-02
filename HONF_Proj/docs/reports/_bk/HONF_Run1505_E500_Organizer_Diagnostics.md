# Run 1505 e500 organizer and region diagnostics

This is a diagnostic view of the **exact epoch-500** Run 1505 checkpoint (`epoch_0500_model.pt`, run UUID `34bd30c2-3364-4680-852e-7c7745e12cc7`). The 90-case split is named `test` by the data loader but is also used for training development validation; it is not an independent final test. Predictions use physical GPU 2 and predicted ports. The checkpoint remains a bounded research endpoint, not a mature 5,000-epoch model.

## What K measures in this model

The architecture registers **12 provisional query-access functions** per case and phase. All 12 were occupied in the e500 90-case formation survey. Converged coalescence may merge some into `R` compact case-local classes; `R` is the variable case-level count in this experiment. It is distinct from the earlier Run 1409 occupancy-derived dynamic-K planner and is not a count of physical modules.

At a query, `Kq` counts compact classes with strictly positive routing mass. The multiplicity-expanded support counts the original constituent functions represented by those positive compact classes. Mass-participation effective K, when shown, is `1 / sum_r(mass_r²)` on the compact routing distribution. These three query measures answer different questions and need not move together. The registered capacity, occupied proposal count, compact `R`, exact `Kq`, and participation effective K must not be interchanged. Latent class indices are checkpoint- and case-local labels; they do not identify physical causal regions.

## Formation at epochs 50, 150, and 500

These counts come from the saved, deterministic Q1024 90-case formation surveys. `R` is phase P2's case-level count. The P2 sampled Kq and virtual-support means each average 1,024 grid queries per case. P0/P1 query sets differ from P2 and are not pooled into these means.

| Checkpoint | P2 cases by R=9 / 10 / 11 / 12 | Mean P2 R | Mean P2 Kq | Mean P2 virtual support |
| --- | ---: | ---: | ---: | ---: |
| Exact e50 | 0 / 0 / 0 / 90 | 12.000 | 2.947 | 2.947 |
| Exact e150 | 0 / 5 / 29 / 56 | 11.567 | 2.957 | 3.095 |
| Exact e500 | 4 / 53 / 28 / 5 | 10.378 | 2.969 | 3.477 |

At e500, 85 of 90 P2 cases formed at least one true merged class. The four e500 nonconverged case-phase calls used the required singleton fallback; case 0277 P2 is one of them. The accepted merged partitions passed the report's strict numerical replay, while two fallback partitions changed at a larger iteration budget. Counts and per-case phase records are in `diagnostics/generated/run1503_v3/epoch_0500_formation_90_q1024/` and the [main Run 1503-v3 report](HONF_Run1503_v3_Converged_Identity_Preserving_Coalescence.md).

The P2 compact-class counts also vary within each physical-module stratum. This is a descriptive breakdown of these 90 cases, not evidence that the organizer has learned an optimal region count.

| Active modules | Cases | R=9 | R=10 | R=11 | R=12 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 25 | 2 | 8 | 12 | 3 |
| 5 | 25 | 1 | 12 | 10 | 2 |
| 7 | 25 | 0 | 19 | 6 | 0 |
| 10 | 15 | 1 | 14 | 0 | 0 |

The reduction in `R` is an observed reduction in **logical case-local access classes**. It does not demonstrate faster physical execution: the production rectangular reader still materializes its fine source rows, and the separate e500 benchmark measured about 5× the parent complete-application time. The figures below are meant to diagnose organization, not to revise that execution result.

## Population and representative figures

The full-grid figures use the final P2 route on all 8,192 original grid points in each of the 90 cases. The case boards use the same exact checkpoint and show the spatial learned route alongside compact source incidence and the actual constituent merges. Their labels and color scales are local to each case. Any juxtaposition with physical prediction error is descriptive; it does not establish that a routing statistic caused the error.

The generated figures, query arrays, and one-time renderers are kept **locally only** at `HONF_Proj/docs/reports/run1503_v3_organizer_diagnostics/` and `HONF_Proj/tools/diagnostics/render_run1505_*.py`. They are intentionally excluded from GitHub. The local figure bundle contains the full-grid effective-K distribution, phase summary, class-utilization chart, accuracy-association chart, and the four case boards described below.

The full-grid replay contains **737,280** query points, including **699,360 fluid** points. Over all points, compact mass-participation effective K has mean `2.186`, median `2.016`, p95 `3.368`, and maximum `5.879`; the fluid-only mean is `2.190`. Exact compact `Kq` has mean `2.978`, p95 `5`, and maximum `9`. Multiplicity-expanded virtual support has mean `3.493`, p95 `5`, and maximum `9`. Thus a typical query puts meaningful mass on fewer classes than its exact positive support, while a fused class may represent multiple original proposal functions. The Q8192 `Kq` mean is close to, but separately measured from, the Q1024 P2 mean in the formation table.

The phase plot shows separate P0/P1/P2 plans at e500. Their state and query sets differ: P0/P1 support summaries each have 768 routed query rows per case, while P2 has 1,024 in the saved sampled survey. A change in R across phases is not necessarily monotone or a cumulative split/merge path.

Every compact P2 class receives positive routing mass at at least one fluid-grid query in all 90 cases. On average, **8.8 of 10.38** compact classes per case are the largest-mass class anywhere on the fluid grid; all classes dominate somewhere in only **9 of 90** cases. A class can contribute without ever being the argmax, so this is a usage description rather than a quality score. The case-level rows, exact query arrays, and reproducible extractor remain in the local-only bundle.

The scatter pairs each case's Q8192 mean effective K with its exact e500 fluid-field and final-port-temperature error. Cases with similar routing concentration span different errors; neither this view nor the individual region maps isolates a causal effect of the organizer.

## Representative case context

The selected cases cover a low-module three-merge plan, a five-module two-merge plan, a ten-module two-merge plan, and a safe singleton fallback. Their physical errors are from the exact e500, full-grid matched evaluation; lower is better within each metric. They are examples for inspection, not a sampled estimate of organizer quality.

| Case | Active physical modules | P2 compact R | P2 merger/fallback | Run 1505 fluid relative L2 | Run 1505 final-port-temperature relative L2 |
| --- | ---: | ---: | --- | ---: | ---: |
| 0273 | 3 | 9 | `[0,10]`, `[1,8]`, `[6,7]` | 0.08082 | 0.13568 |
| 0653 | 5 | 10 | `[0,10]`, `[1,8]` | 0.09067 | 0.10657 |
| 0686 | 10 | 10 | `[0,10]`, `[1,8]` | 0.10542 | 0.08667 |
| 0277 | 3 | 12 | solver fallback; all singleton | 0.10250 | 0.19730 |

The first row of each board maps the largest-mass compact class and exact positive compact Kq on the physical query grid, with solid module cells blanked and module slots numbered. The small bar chart counts each class's dominant fluid cells and connected regions. The second row shows original proposal slots mapped into independently recomputed P0/P1/P2 classes, then the raw P2 module and environment source-incidence weights. Environment heatmap rows are source-slot IDs, **not** spatial y coordinates. Within a board, `C` labels are phase-local and only parent proposal-slot membership makes a phase comparison meaningful.

### Case 0273: three merged pairs

P2 has R=9 and three two-proposal classes. Eight compact classes dominate at least one fluid query; the ninth still has positive route mass somewhere. Each of the eight dominant fluid regions has one 4-neighbor connected component on this grid. The large colored regions are learned access preferences, not a ground-truth spatial partition.

### Case 0653: five-module example

P2 has R=10 and two merged pairs; eight classes dominate fluid queries. One dominant class has a main 1,336-cell region plus only 4- and 2-cell islands; the other dominant classes each occupy one connected region. The module and environment heatmaps show that compact access classes can draw from multiple fine sources while the fine sources themselves remain present. Dark columns C3/C4 are **not** source-empty: their environment-incidence column sums are 31.44/21.29 across 192 source slots, and their mean fluid-query route masses are 0.0270/0.0136. They have positive support in 1,548/790 of 7,831 fluid queries despite never being the argmax. Thus a missing dominant-colored region is not proof that an access class is unused.

### Case 0686: ten-module example

P2 again has R=10 and two merged pairs, now with ten active physical modules. Eight classes dominate fluid queries, each in one connected fluid region on this grid, illustrating that R is a learned case-level access count rather than the physical module count.

### Case 0277: safe singleton fallback

Its P2 solver did not meet the 512-step residual criterion, so R=12 and the model used the exact parent singleton route. All 12 classes dominate somewhere on the fluid grid; one class has two substantial disconnected dominant components of 175 and 49 cells. The absence of an accepted merge here is a numerical fallback, not evidence that 12 functions are physically necessary for this case.

The boards can be regenerated with the local-only `HONF_Proj/tools/diagnostics/render_run1505_organizer_cases.py`; per-case plan, coordinate, route, and connected-region measurements remain in the local-only `cases/provenance.json`.

These maps describe learned query-access functions and source associations. They do not assign physical causal ownership to a module, imply a segmented CFD domain, or establish design-coordinate smoothness. The e500 design-boundary diagnostic in the main report remains the direct local continuity evidence.
