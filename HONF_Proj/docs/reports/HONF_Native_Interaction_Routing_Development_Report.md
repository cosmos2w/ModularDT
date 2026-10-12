# Native interaction routing development

This round adds learned donor weights inside the existing two nonlinear source/environment rounds (P-R), together with a parameter-matched destination-only aggregate modulator (P-C). Both children start from the same segmented P/e1000 function and continue all original physical objectives to absolute e2500. Every individual physical-source read remains present. The mature Wind migration and historical predictors remain intact; affine factoring stays disabled.

## Experiment and comparison limits

The Thermal pair is T4301/P-R versus T4302/P-C, attached to T4111/P e1000 on fixed25_v1 TRAIN150 with exposed DEV22. The Wind pair is W2601/P-R versus W2602/P-C, attached to W2401/P e1000 on fixed24 TRAIN72 rows/24 layouts with DEV24 rows/eight layouts. The Wind subset omits M18, and its three direction rows per layout are correlated. Thermal DEV22 belongs to the source H5 partition named `test`; these are explicitly exposed development cases, while unexposed Thermal targets and WindTEST stay locked. These results are conditional development evidence from one parent and seed per dataset.

The children retain the parent weights, named AdamW moments, TRAIN normalization/calibration, case populations, query namespace, all original losses, clipping and absolute e2500 schedule. New routing/control moments begin at their actual birth. Each Thermal arm performs 6,000 new updates and 225,000 case visits; each Wind arm performs 4,500 new updates and 108,000 visits. The inherited updates are 4,000 and 3,000 respectively. Primary label queries per arm are 230.4 million Thermal Q1024 rows and 442.368 million Wind Q4096 total role rows; Thermal native stencil/interface/material/response work is accounted separately.

Historical P is a scoped same-cohort, same-age comparator. The reused Thermal ordinary-P continuation matches protected e1100 weights, moments and recorded sampling/work exactly. The bounded Wind original-source/current-source e1001 replay matches all 52 weights, named moments, three updates, nine independently observed native query draws, losses and work bitwise. This verifies the inspected boundaries; neither replay reconstructs the entire historical e1001–2500 trajectory. Wind R's original e1001–1100 observed query hashes are absent. Its loss/work/case records remain available, and the observed R/C query suffix from e1101 is matched; the whole Wind pair is not labelled a complete query-provenance pass.

P-R adds 26,496 Thermal or 27,360 Wind parameters in three typed score networks shared across rounds. Scores use destination, donor and signed geometry before nonlinear value evaluation, with the frozen bound `2*tanh(raw/2)`. Normalization preserves the actual legacy row mass and exact zero-score weights. P-C adds 26,400 Thermal or 27,720 Wind parameters; it sees the current destination state and prescribed context at the modulation site. A destination can already contain earlier donor information. These controls distinguish donor-selective weighting from added capacity at this site, rather than proving the control has no interaction information.

## Predictor measurements

Thermal P-R does not clear the frozen routing-direction criterion. Its fluid-temperature mean RMSE is 0.574835 versus P 0.608297 (−5.50%) and C 0.591745 (−2.86%); v mean is 0.00337088 versus P 0.00331994 (+1.53%) and C 0.00333517 (+1.07%). No common priority field improves at least 5% against both comparators. Mean and declared p90 guards pass, but the gain gate fails. The useful temperature/flux changes remain a joint-field tradeoff, not a promoted routing result.

Every Thermal statistic below is an equal-case summary over exposed DEV22 at absolute e2500. u/v/p/omega/fluid-T use the full 64×128 native grid with the actual captured fluid-stencil/solid-exclusion mask. Surface/material T, normal-flux q proxy and sampled per-module peak use the saved exact Q1024 monitor scope. Values retain the benchmark-native source-pack scales; heterogeneous units are not averaged into a physical headline. Peak means the sampled native module statistic, not a solver-certified continuous maximum.

| Thermal field | P mean / p90 / worst | P-R mean / p90 / worst | P-C mean / p90 / worst |
|---|---:|---:|---:|
| u | 0.0229956 / 0.0275702 / 0.032681 | 0.0225157 / 0.026884 / 0.0321113 | 0.0231395 / 0.0276689 / 0.031967 |
| v | 0.00331994 / 0.00420435 / 0.00445034 | 0.00337088 / 0.00419246 / 0.00446152 | 0.00333517 / 0.00420076 / 0.00446994 |
| p | 0.00984368 / 0.0122014 / 0.0130717 | 0.0100349 / 0.0123073 / 0.012821 | 0.00994773 / 0.0121389 / 0.0130462 |
| omega | 0.0658715 / 0.0868889 / 0.0921473 | 0.0641169 / 0.0840304 / 0.0890683 | 0.0657524 / 0.0867447 / 0.0914134 |
| fluid_temperature | 0.608297 / 0.84806 / 0.906601 | 0.574835 / 0.782559 / 0.99627 | 0.591745 / 0.827517 / 0.871385 |
| surface_temperature | 0.671735 / 1.04339 / 1.21263 | 0.626415 / 1.06886 / 1.09474 | 0.662464 / 1.02602 / 1.17953 |
| material_temperature | 0.655307 / 0.971662 / 1.14452 | 0.630175 / 1.03661 / 1.14825 | 0.657369 / 0.980916 / 1.17467 |
| q_proxy | 2.35289 / 3.06958 / 3.65527 | 2.18152 / 2.85433 / 3.21949 | 2.35995 / 3.15596 / 3.74146 |
| sampled_module_peak | 0.710172 / 1.12279 / 1.54068 | 0.673973 / 1.18253 / 1.54399 | 0.715729 / 1.21601 / 1.47033 |

The M5 stratum has six cases and material local misses: R's material-temperature mean is 6.67% worse than P and 8.50% worse than C; sampled-peak mean is 16.39% worse than P and 19.04% worse than C, and surface-T is 5.87% worse than C. The other observed module-count mean comparisons have no >5% increase. Fluid-T worst-case RMSE is 0.996270 versus P 0.906601 and C 0.871385, so the mean improvement does not remove tail risk. A larger routing or loss-weight search is not authorized by this result.

Wind P-R clears the frozen physical direction criterion at e2500. Near-turbine Uy/Uz mean changes versus P are −6.16%/−6.04%; downstream Uy/Uz are −4.65%/−4.58%. Their four-ratio geometric mean is 0.946386 versus P and 0.942879 versus C. Near/downstream vector ratios versus P are 0.962835/0.959569. Every declared component/vector mean and priority row-p95 guard passes. No >5% component/vector mean increase appears in the observed module-count, direction or individual-layout strata versus either comparator. This is provisional same-cohort development evidence; eight layouts and one seed do not establish an independent-test or multi-seed win.

Wind metrics use one fixed matched Q8192 draw TOTAL across the five roles for each DEV row. Component RMSE is sqrt(mean(error_component²)); role-vector RMSE is sqrt(mean(sum(error_component²))), without an additional division by three. All 24 direction rows are reported, with layouts retained as the grouping unit; no independent-row confidence interval is claimed.

| Wind role/component (m/s) | P mean / row-p95 / worst | P-R mean / row-p95 / worst | P-C mean / row-p95 / worst |
|---|---:|---:|---:|
| volume/Ux | 0.0861548 / 0.12284 / 0.128413 | 0.0823836 / 0.122569 / 0.127844 | 0.086341 / 0.123197 / 0.128979 |
| volume/Uy | 0.016241 / 0.0212215 / 0.0252379 | 0.0156135 / 0.0214306 / 0.024612 | 0.0162779 / 0.0212878 / 0.0253093 |
| volume/Uz | 0.0128885 / 0.0168027 / 0.0182133 | 0.012335 / 0.0166674 / 0.0178305 | 0.0129418 / 0.0169278 / 0.0183497 |
| volume/vector | 0.0886629 / 0.125669 / 0.131896 | 0.0847868 / 0.125452 / 0.131198 | 0.0888588 / 0.126037 / 0.132463 |
| hub_slab/Ux | 0.16754 / 0.249827 / 0.25322 | 0.159522 / 0.237518 / 0.248559 | 0.16793 / 0.250895 / 0.254175 |
| hub_slab/Uy | 0.0298792 / 0.0429432 / 0.0476006 | 0.0285748 / 0.0423069 / 0.0462627 | 0.0299505 / 0.0430362 / 0.0477318 |
| hub_slab/Uz | 0.0185178 / 0.0264418 / 0.02894 | 0.0178494 / 0.0257754 / 0.0280805 | 0.0186325 / 0.0265523 / 0.0290095 |
| hub_slab/vector | 0.171229 / 0.254564 / 0.25847 | 0.163074 / 0.242246 / 0.253721 | 0.171636 / 0.25564 / 0.259428 |
| downstream_envelope/Ux | 0.195104 / 0.244074 / 0.244898 | 0.187282 / 0.238713 / 0.245789 | 0.195511 / 0.244588 / 0.245311 |
| downstream_envelope/Uy | 0.0338825 / 0.047129 / 0.0490401 | 0.0323075 / 0.045926 / 0.047666 | 0.0339575 / 0.0472457 / 0.0491767 |
| downstream_envelope/Uz | 0.0254795 / 0.0341032 / 0.0371795 | 0.0243121 / 0.0339621 / 0.0363666 | 0.0255857 / 0.0342692 / 0.0373707 |
| downstream_envelope/vector | 0.199707 / 0.250833 / 0.252512 | 0.191632 / 0.245452 / 0.252996 | 0.20013 / 0.251481 / 0.252856 |
| near_turbine/Ux | 0.271044 / 0.377659 / 0.389455 | 0.261387 / 0.378873 / 0.390637 | 0.271884 / 0.378893 / 0.390472 |
| near_turbine/Uy | 0.0554624 / 0.0749905 / 0.0767338 | 0.0520433 / 0.071635 / 0.0727221 | 0.0555886 / 0.0753024 / 0.0768605 |
| near_turbine/Uz | 0.0384488 / 0.049811 / 0.0551543 | 0.0361269 / 0.047857 / 0.0526031 | 0.0386882 / 0.0502764 / 0.0555749 |
| near_turbine/vector | 0.279411 / 0.388658 / 0.398156 | 0.269027 / 0.388909 / 0.399022 | 0.280282 / 0.389965 / 0.399297 |
| background/Ux | 0.0360956 / 0.0478256 / 0.0511952 | 0.0354618 / 0.047731 / 0.049611 | 0.0361559 / 0.0479182 / 0.0513925 |
| background/Uy | 0.00825866 / 0.0103585 / 0.0105892 | 0.00828173 / 0.0103484 / 0.0105374 | 0.00826852 / 0.0103693 / 0.0105656 |
| background/Uz | 0.00788749 / 0.00939049 / 0.0101654 | 0.00775941 / 0.00939778 / 0.0100891 | 0.00790058 / 0.00945187 / 0.0101578 |
| background/vector | 0.0378868 / 0.0493297 / 0.053258 | 0.0372626 / 0.0491855 / 0.0517115 | 0.0379499 / 0.0494208 / 0.0534416 |

The inherited 15-age selector uses e1100, e1200, …, e2500 for every model. Both Thermal children select e2400; Thermal P and all three Wind models select e2500. Selected physical panels are reported separately in the saved receipt, and their metrics do not replace the endpoint gate. Organizer interventions use those selected states; complete cost and fixed-TRAIN maps use the common e2500 endpoints.




Selected-state exposed DEV22 means are reported below using the same two physical scopes as the endpoint table. Wind selects e2500 for all models, so its selected statistics equal its endpoint statistics. These age-selected Thermal numbers do not substitute for the e2500 gain criterion.

| Thermal selected state | u | v | p | omega | Fluid T | Surface T | Material T | q proxy | Sampled peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P / e2500 | 0.0229956 | 0.00331994 | 0.00984368 | 0.0658715 | 0.608297 | 0.671735 | 0.655307 | 2.35289 | 0.710172 |
| P-R / e2400 | 0.0224067 | 0.00337572 | 0.0101608 | 0.0642457 | 0.576405 | 0.623307 | 0.630914 | 2.18182 | 0.673187 |
| P-C / e2400 | 0.0230413 | 0.00334152 | 0.0100184 | 0.0658533 | 0.592107 | 0.659958 | 0.656774 | 2.3613 | 0.715202 |

![Thermal development trajectories and native endpoint field errors](../../diagnostics/generated/native_routing_20261012/evaluation/results/figures_learning_cost_v3/family1_thermal.png)

Figure 1a. Saved exposed DEV22 trajectories over the common 15 monitoring ages and all nine native field/proxy endpoint error summaries. Thermal R reduces mean fluid-T error by 5.50% against P but only 2.86% against C, with worse mean v and the reported M5/material/peak misses; it fails the frozen direction criterion.

![Wind development trajectories and fifteen component plus five vector endpoint errors](../../diagnostics/generated/native_routing_20261012/evaluation/results/figures_learning_cost_v3/family1_wind.png)

Figure 1b. Exposed DEV24 comprises eight layouts and three correlated directions per layout. All 15 component means and five vector tail summaries are retained; R's four priority-component geometric mean ratios are 0.946386 against P and 0.942879 against C. This supports a warm-start learning direction under the stated single-parent and historical-P provenance limits.

![Thermal full native grid v and temperature fields and residuals](../../diagnostics/generated/native_routing_20261012/evaluation/results/figures_physical_endpoints2500_v2/native_physical_maps_e2500_v10_page01_thermal.png)

Figure 2a. Exact e2500 P/R/C on fixed TRAIN0002/M3 and 0619/M12, full 64×128 native grids with captured fluid-valid/solid-exclusion masks, common raw-extrema physical scales and separate residual scales. Fluid-T RMSE P/R/C is 0.462323/0.435316/0.455950 on M3 and 0.710078/0.616160/0.668980 on M12. The displayed v/T priority maps are descriptive TRAIN evidence; all five fluid channels are retained and validated numerically. Source-pack scales are not relabelled SI units.

![Wind six-turbine native 3D horizontal and transverse velocity maps and residuals](../../diagnostics/generated/native_routing_20261012/evaluation/results/figures_physical_endpoints2500_v2/native_physical_maps_e2500_v10_page02_wind_row456.png)

Figure 2b. Exact e2500 fields on fixed TRAIN row456/M6, direction 270°: all Ux/Uy/Uz m/s values and residuals, a 27,470-point native horizontal plane at selected z=70.8699 m (requested hub 70 m), and an 8,576-point native transverse plane at x=575.6979 m. The transverse Ux RMSE is 0.072108/0.069769/0.072580 for P/R/C, while R's transverse Uy is worse (0.010932 versus P 0.010671; relative-L2 0.70158 versus 0.68483). The aggregate role gain does not imply every native plane/component improves.

![Wind thirty-turbine native 3D horizontal and transverse velocity maps and residuals](../../diagnostics/generated/native_routing_20261012/evaluation/results/figures_physical_endpoints2500_v2/native_physical_maps_e2500_v10_page03_wind_row533.png)

Figure 2c. Exact e2500 fields on fixed TRAIN row533/M30, direction 300°: an 88,060-point horizontal plane at the same selected native z and a 16,576-point transverse plane at x=1340.4991 m. Horizontal Ux/Uy/Uz RMSE is 0.234485/0.045926/0.010631 for R versus P 0.240116/0.047027/0.010729; transverse Uy worsens slightly to 0.011537 versus P 0.011469. Axes remain native x/y/z divided by rotor diameter, with no second rotation. Componentwise relative-L2 annotations use raw native reference norms; weak transverse components can have large relative errors despite small m/s RMSE. Map Q is separate from the Q1024/Q8192 timed-call scopes.

## Organizer measurements

The selected-state native trace retains both MM/ME/EM rounds, literal base weights w0, bounded scores, normalized densities, actual w0×rho weights, weighted-message norms, source IDs and the separate final source-query reads. Thermal uses e2400; Wind uses e2500. MM excludes source self reads (276/306 supported slots across the two Thermal scenes and 1,800/1,872 Wind slots); ME and EM support every measured slot. The nonlinear value tensors are evaluated for all dense pair slots, including unsupported/padded entries, and every final physical-source read remains. Support percentages are diagnostic masks, not executor savings.

The fixed TRAIN complete-rebuild probes reset densities or deterministically reassign scores within equal-measure supported donor buckets for each route and all three routes together. Both rounds and their downstream values are recomputed. Restored ordinary fields and route traces are exact, with weights/versions unchanged. On Thermal M12 case0619, all-route density reset increases fluid-T/surface-T/material-T RMSE by 55.72%/56.99%/90.78%; reassignment increases them by 70.17%/64.82%/96.18%. On M3 case0002, reset changes fluid-T by +1.31% and q proxy by −3.41%, so an intervention is not uniformly harmful across every field. These are selected-state sampled native physical errors, separate from the common e2500 full-DEV gate.

For Wind row456/M6, all-route reset increases near/downstream vector RMSE by 13.39%/18.27%; reassignment increases it by 27.09%/30.42%. For row533/M30, reset gives +18.72%/+23.81% and reassignment +23.57%/+28.62%. Vector errors here are reconstructed as sqrt(sum(component RMSE²)) on the exact same role queries. The result demonstrates functional reliance on learned gathering. Together with the independently trained capacity control, it supports a conditional Wind learning benefit; it establishes neither physical causality nor a computation-reuse benefit. Geometry-only rebuilds change inputs without a bound perturbed physical reference and are software invalidation checks.

![Thermal selected-state native MM ME EM gathers and final source reads](../../diagnostics/generated/native_routing_20261012/evaluation/results/figures_routes_selected_v3/native_routes_selected15_v8_page01_route_T4301_0002.png)

Figure 3a. T4301/R e2400, fixed TRAIN0002/M3, shows four norm-ranked display edges per route/round and both native query-to-source read columns; the numerical archive retains all pairs. Every one of the three physical sources is read. The input-only geometry probe preserves fixed queries and IDs, and has no perturbed physical reference.

![Wind selected-state native MM ME EM gathers and thirty final source reads](../../diagnostics/generated/native_routing_20261012/evaluation/results/figures_routes_selected_v3/native_routes_selected15_v8_page02_route_W2601_533.png)

Figure 3b. W2601/R e2500, fixed TRAIN533/M30, retains all 30 physical-source read rows at a volume and a downstream native 3D query. These are latent message norms before the physical head, not velocity or causal source coefficients; the context weights are destination conditioned and every authoritative value path is dense.

![Thermal native physical RMSE changes from complete routing interventions](../../diagnostics/generated/native_routing_20261012/evaluation/results/figures_routes_selected_v3/native_routes_selected15_v8_page03_interventions_thermal.png)

Figure 3c. Selected T4301/R e2400 on fixed TRAIN0002/M3 and 0619/M12: exact field-unit RMSE changes from complete two-round density reset/reassignment, with exact restored-ordinary zero rows. All-route reset increases M12 fluid-T RMSE by 55.72% but reduces M3 q-proxy RMSE by 3.41%. Colors normalize separately by scene and field, so numerical cell values carry the quantitative comparison.

![Wind component RMSE changes from complete routing interventions](../../diagnostics/generated/native_routing_20261012/evaluation/results/figures_routes_selected_v3/native_routes_selected15_v8_page04_interventions_wind.png)

Figure 3d. Selected W2601/R e2500 on fixed TRAIN456/M6 and 533/M30: the full five-role Ux/Uy/Uz m/s error changes preserve masks and queries. All-route reset increases near/downstream vector RMSE by 13.39%/18.27% on M6 and 18.72%/23.81% on M30. MM perturbations dominate this panel, while ME/EM changes are much smaller; this establishes model reliance, without a sparse-work or physical-causality interpretation.

## Execution measurements

Complete Q1024/Q8192 inference calls include CPU input staging, context preparation, all native outputs and CPU return. Training-boundary trials retain all original objectives and include query generation, forward/loss/backward, clipping, AdamW and synchronization. Each disposable trial restores the exact endpoint weights, named moments and RNG outside its timer. Cold setup, per-trial allocation/reserve baselines and peaks, receipt-hook overhead and the charged process lifetime are reported separately.

| Full-objective update, effective/microbatch and query scope | P median (s) | P-R median (s) / ratio to P | P-C median (s) / ratio to P | Allocated peak delta P / R / C (GiB) |
|---|---:|---:|---:|---:|
| Thermal B48/8, Q1024 plus original native auxiliaries | 0.401798 | 0.445357 / 1.10841 | 0.446226 / 1.11057 | 1.983 / 2.039 / 1.992 |
| Wind B24/8, Q4096 total five-role receivers | 0.223924 | 0.248739 / 1.11082 | 0.240917 / 1.07589 | 2.538 / 2.553 / 2.540 |

These medians use seven measured trials after two warmups per model/dataset cell on physical GPU2 (RTX6000 Ada). All six endpoint models remain resident. Reported adjusted time excludes only the measured optimizer-receipt hook interval from the same trial; raw time and hook time are retained together. Fifty-four actual optimizer updates were restored and discarded, charged to the disposable allocation. The adjustment does not subtract independent medians or remove sampling, native physical objectives, backward, clipping or AdamW work.



| Complete native call, fixed TRAIN scene | Q | P median (ms) | P-R median (ms) / ratio | P-C median (ms) / ratio |
|---|---:|---:|---:|---:|
| Thermal 0002 | 1024 | 17.150 | 20.057 / 1.16956 | 19.896 / 1.16016 |
| Thermal 0002 | 8192 | 29.435 | 32.718 / 1.11152 | 32.344 / 1.09882 |
| Thermal 0619 | 1024 | 19.167 | 22.282 / 1.16256 | 21.316 / 1.11213 |
| Thermal 0619 | 8192 | 29.210 | 33.654 / 1.15214 | 32.495 / 1.11247 |
| Wind 456 | 1024 | 6.508 | 10.175 / 1.56341 | 9.272 / 1.42479 |
| Wind 456 | 8192 | 15.274 | 18.764 / 1.22851 | 17.835 / 1.16769 |
| Wind 533 | 1024 | 6.761 | 10.202 / 1.50908 | 9.380 / 1.38735 |
| Wind 533 | 8192 | 15.607 | 19.080 / 1.22253 | 18.178 / 1.16474 |

Thermal Q1024 retains the original CASE_EPOCH provider draw and original secondary native queries. Q8192 enumerates every unique 64×128 native cell center, including saved solid-mask cells in the work count; the physical-map error mask uses saved-solid exclusion intersected with actual prepared fluid-stencil validity. Wind Q counts TOTAL receivers across five native roles at the original mixture. All P/R/C inputs, source IDs, masks and output scopes match within a cell. Model states and versions remain unchanged and every cell returns the same output digest across its nine calls.

Wind R misses the predeclared 20% complete Q8192 target (max ratio 1.22851); it remains below the 35% primary-workload boundary, while Q1024 overhead of 50.91–56.34% exceeds that boundary at the smaller scope. Thermal R stays below 20% on both scopes, but its physical gain criterion fails. Neither dataset clears the prospective combined fresh-start gate, so no fresh pair or further scientific update is authorized or performed. The 99%-mass cut and selected-value executor are not inspected; no sparse work, reuse approximation, or executor saving is established. Retain dense execution and historical defaults.

Context/checkpoint construction is measured separately: Thermal P/R/C takes 2.577/2.274/2.271 s, Wind 0.134/0.095/0.151 s with the existing target-free catalogue cache. These are observed staged construction costs, not a universal cold-disk claim. The per-trial allocated/reserved baselines and peaks include all resident matched models; allocated peak deltas describe incremental call workspace, not total VRAM. At Q8192 the maximum measured workspace is about 7.58/20.38 MiB for Thermal M3/M12 and 11.07/44.40 MiB for Wind M6/M30; reserved deltas are zero with cached allocator blocks and are not zero memory use.

![Matched complete inference training and memory costs](../../diagnostics/generated/native_routing_20261012/evaluation/results/figures_learning_cost_v3/family4_complete_cost.png)

Figure 4. GPU2 complete-call timing, every native output and CPU return, full-original-objective AdamW update timing, and incremental allocated workspace from the exact common e2500 endpoints. Wind R Q8192 costs 22.25–22.85% more than P, and the training update costs 11.08% more; lower-Q overhead is larger. These are measured dense-model costs, without a sparse speedup or matched-function executor comparison.

## Interface measurements

At all four exact e2500 children, the CPU software probe passes unchanged-parameter/source-ID, stale-clone rejection/restoration and finite geometry/input VJP checks. Thermal uses fixed TRAIN0002 and balanced heat transfer [+0.04326184, −0.04326184, 0], with 1,024 fluid points, 48 interface points (temperature and q outputs) and 96 material points. Flow is bitwise unchanged under heat, its heat VJP is zero, and the maximum affine shifted-minus-baseline versus precise-increment residual is 4.89e−15 at the original 1e−11 absolute / 1e−8 relative tolerances in declared FP64 arithmetic. Native-curl off-grid rejection remains intact. Wind uses fixed TRAIN row456; its native input/source/query VJPs are finite and its nonlinear output rejects exact affine increments. These are software checks, not physical finite-response validation.

For R's fluid-temperature panel, max(T_endpoint) − max(T_baseline) is −0.388738, while max(T_endpoint − T_baseline) is +0.378047 in the retained native temperature scale. C gives −0.389466 and +0.378071 respectively. Both quantities are evaluated explicitly; the endpoint maximum is not reconstructed from a linear increment maximum.

## Inverse boundary

No inverse generator, design search, sampler-diversity test or new solver evaluation ran. Existing affine source columns and differentiable inputs remain available, but their arithmetic checks do not validate physical Green functions or candidate designs. Existing counted-response receipts that cannot be bound remain NA; opposite balanced heat transfers represent one independent direction. Neither max(T_endpoint) − max(T_baseline) nor max(T_endpoint − T_baseline) is inferred from the other.

## Decisions and next steps

| Decision | Measured conclusion | Next action |
|---|---|---|
| Predictor | Wind R clears the frozen absolute-e2500 physical criteria against both P and C on exposed DEV24. Thermal R misses the priority gain over C and has M5 material/peak and worst-fluid-T concerns. | Retain Wind warm R as a conditional development result; retain P for Thermal. No independent-test or full-population promotion. |
| Organizer | Complete rebuilds show model reliance, and Wind R exceeds the destination-only capacity control. Thermal interventions show reliance without the required added predictive value. | Keep the operational trace/intervention API; claim a conditional Wind learning benefit only. No physical causal/higher-order or computation-reuse claim. |
| Execution | Thermal costs meet the 20% target; Wind R Q8192 is 22.25–22.85% slower than P, missing the target, and Q1024 is 50.91–56.34% slower. All message values and final source reads remain dense. | Keep legacy execution defaults. No sparse saving, selected executor or approximate reuse is established. |
| Interface | Trained children preserve Thermal heat independence/affinity, native-curl geometry restrictions, source ordering, finite tested VJPs and stale-state rejection; Wind remains nonlinear. | Retain the tested software interfaces. Physical response validation and inverse-design validity remain unmeasured. |
| Next formal action | Neither dataset passes the combined prospective fresh gate: Thermal physical FAIL, Wind complete Q8192 cost FAIL. Fresh optimizer allocation remains 0; all primary warm runs are e2500. | No formal full-TRAIN/5000 recipe is recommended or launched. Wind warm evidence can inform a later user-controlled decision, with its missing query prefix and absent fresh/multi-seed evidence explicit. |

The optional fresh pair and 99%-mass context-cut probe are not run. No larger routing architecture, third recipe, new solve, frozen-flow replacement or inverse campaign follows from these results. Further work would require a new scoped request that addresses the observed cost and generalization limits; this round does not change loss weights or tune the gate after observing the outcomes.

## Retained figure index

| Family | PDF master | Embedded pages | Evidence |
|---|---|---:|---|
| 1. Learning and exposed DEV fidelity | [Learning/physical PDF](../../diagnostics/generated/native_routing_20261012/evaluation/results/figures_learning_cost_v3/family1_learning_physical.pdf) | 2 | Shared 15-age monitoring and same-age native physical/tail comparisons |
| 2. Native fields and residuals | [Native-map PDF](../../diagnostics/generated/native_routing_20261012/evaluation/results/figures_physical_endpoints2500_v2/native_physical_maps_e2500_v10.pdf) | 3 | Two fixed TRAIN scenes per dataset, native masks and all Wind velocity components |
| 3. Native gathering and interventions | [Route/intervention PDF](../../diagnostics/generated/native_routing_20261012/evaluation/results/figures_routes_selected_v3/native_routes_selected15_v8.pdf) | 4 | Selected-state actual gathers, source reads and complete two-round rebuilds |
| 4. Complete cost and memory | [Complete-cost PDF](../../diagnostics/generated/native_routing_20261012/evaluation/results/figures_learning_cost_v3/family4_complete_cost.pdf) | 1 | Interleaved complete calls, full-objective updates and resident-baseline workspace |

PDF is the retained master; the ten small PNG companions exist for direct Markdown embedding. All ten pages were visually inspected, and all generated figures and one-time renderers remain in ignored local paths. Numerical arrays and physical reference artifacts are retained separately.

## Validation, preservation and evidence

The final focused suite passes 154 tests with five optimizer-using tests deselected; the separately counted architecture-child/native qualification tests verify optimizer behavior. Native GPU attachment outputs match P exactly on the fixed scenes; the Wind input VJP has maximum absolute difference 3.814697265625e-6 and passes the original tolerance, so it is not reported as bitwise. Empty/padded/self-excluded routes, measure normalization, permutation/refinement behavior, destination-dependent ordering, live gradients, strict reload/identity and prepared-state invalidation are covered by focused tests.

The 0291 caption and evidence-index entry are corrected to exposed DEV/audit provenance; 0291 is not admitted to TRAIN. The historical observation set contains 886 present and 67 pre-existing absent bindings. Preservation means unchanged observations, including that absent set, rather than 953 present artifacts. The final preservation check has zero mismatches, including 764 additional protected files, 195 prior-round artifacts, the parents/mature migration and solver326/326. Full governed lifetimes total 2.273506 of 8 aggregate GPU-hours; scientific updates are 21,000 primary, zero pilot and zero fresh, with 74 of 128 disposable updates and no reservation remaining. All 38 governed sessions are finalized; failures/staging are charged, GPU2 and owned training workers are stopped, and unrelated GPU jobs remain untouched. Delivery is completed before the scientific cutoff and preserves the final two-hour window. The companion index binds the resource, rendering, preservation and upload receipts.

Only 19 superseded local PDF/PNG exports (5,295,638 bytes) were removed after validating the selected successors. All numerical arrays, render-source versions, JSON provenance, models and original physical references remain intact. The final selection is four PDF masters and ten Markdown PNG companions; their inventory and cleanup receipts retain exact hashes and deletion paths.

## Authorized storage cleanup after measurement

After the measured round closed, the user authorized removal of its duplicated geometry caches. The [cleanup receipt](../../diagnostics/generated/native_routing_20261012/storage_audit/cleanup_execution_v1/cleanup_receipt.json) records deletion of exactly 2,410 round-local `.npy` cache payloads, reclaiming 30.770035 GiB, while retaining all 241 cache-entry manifests. Full payload hashes were verified against the existing shared cache before deletion; the shared cache and all retained campaign files passed the recorded preservation checks. No scientific-result NPZ, checkpoint, physical reference, metric, figure, source script or solver artifact was deleted, and no model, sampling or measured cost/fidelity result changed. The earlier export-cleanup statement describes the measurement closeout before this separately authorized cache removal. The repository now requires a disk budget, shared-cache reuse and verified cleanup before future rounds are marked complete.
