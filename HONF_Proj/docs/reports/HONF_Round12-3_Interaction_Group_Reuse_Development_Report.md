# Round 12-3: Real group substitution, mixed predictor progress, and higher complete cost

The recovery changes one reader rule: normalize the geometric window and each module’s top-four B posting distribution before combining them through the original nullable proposal coefficients. It retains E→E in both new arms, then compares singleton groups with learned interaction-derived tree frontiers. Both fresh seed 0 arms actually completed 100 development epochs; the original G-EE 100 remains the unchanged-reader control. Groups really pool contextual environmental states, supply the group→M exchange and expose the same cached key/value tensors to queries, with no original-E expansion or M-value decoder bypass.

The measured result is partial. The balanced singleton improves several temperature/peak endpoints and the previous G-F surface tail, while worsening other native and response quantities against its proper retained-E→E control. Adaptive aggregation replaces substantial group/member-derived work and lowers memory, but this finite tree implementation loses physical detail and increases complete inference and training time. This is evidence about one 100-epoch optimization, reader and exclusive tree family; it neither establishes useful general hypergraph organization nor rejects the central idea that learned organization can form and reuse information.

The original 1.10 mean/1.15 p90 reporting bands remain visible, together with all original failures. They are descriptive comparisons, not physical discontinuities, significance tests or hard prerequisites for exploration. The balanced singleton’s surface mean 3.149 improves 3.81% over the earlier G-F 3.273, and p90 improves 11.75%; relative to the same-E→E original G-EE, its mean is 12.92% higher and p90 only 4.16% higher. These two comparisons cannot be conflated because G-F also deleted E→E. We assess the sizes, distributions, response references and work tradeoffs continuously, without moving the bands or training until they happen to be crossed.

## Recovery evidence and figure index

| Figure | Question | PDF master |
| --- | --- | --- |
| 5 | Reader scale, lost posting mass, surface bias/detail | [Reader and surface](../../diagnostics/round12_3_recovery_20261024/figures/05_reader_and_surface.pdf) |
| 6 | All-nine native quality and physical responses | [Physical quality](../../diagnostics/round12_3_recovery_20261024/figures/06_physical_quality.pdf) |
| 7 | Native physical fields and signed residuals | [Fields](../../diagnostics/round12_3_recovery_20261024/figures/07_native_fields.pdf) |
| 8 | Actual relations/groups, reuse and complete cost | [Organization and cost](../../diagnostics/round12_3_recovery_20261024/figures/08_organization_and_cost.pdf) |

The [local recovery evidence](../../diagnostics/round12_3_recovery_20261024/) retains all numerical boards, per-case errors, reference responses, actual graph/support arrays, TRAIN/DEV surface arrays, calibration and sampling verification, full timing repeats, failure records and storage receipts. The four new figures use saved numerical evidence, Python and visually inspected PDF/PNG exports. The original first-screen measurements are retained below, including the superseded exhaustive diagnostic and its explicit correction notice.

## Execution: matched exposures and bounded implementation repair

| Endpoint | E→E / grouping | Primary visits / updates / micros | Engine seconds |
| --- | --- | --- | --- |
| Original G-EE 100 | retained / singleton | 15,000 /400 /700 | 845.68 |
| Balanced singleton 100 | retained / singleton | 15,000 /400 /700 | 863.57 |
| Balanced adaptive 100 | retained / learned | 15,000 /400 /700 | 4113.29 |

All three endpoints use fixed25 TRAIN150/exposed DEV22, selected-TRAIN normalization, width 128/four heads/two layers, seed 0, FP 32, Q1024, nominal micro 24/effective 48, AdamW 3 e-4 and the unchanged absolute 1000-epoch schedule. Each has 15.36 million primary fluid queries,100 response pairs/200 endpoint-scene visits and a separately counted two-scene response calibration. Every recorded case-order and primary-query identity matches the original G-EE at every epoch. Each new epoch 100/latest/best alias has exactly equal model tensors; best has only that one monitoring point. There is no checkpoint mixing, full-data fitting, source checkpoint continuation or new solver/data work.

The response balancing rule is matched, not its resulting numerical scalar: coefficients are 2027.750 original,23.474 balanced singleton and 0.370 adaptive. All use the declared 10%-of-shared-native-gradient target/downward cap and the same admitted TRAIN0348 pair cadence. Adaptive native/response frontiers are sampled and include their score-function contributions; singleton losses have no discrete organizer. Different coefficients and stochastic-versus-deterministic frontiers limit a claim of an identical scalar objective, even though the normalization, losses, data exposure and balancing procedure match.

The first adaptive v2 attempt clean-stopped at epoch 3 after 450 visits/12 updates and 123.07 engine seconds. Baseline and trial response endpoints accidentally shared a sampling key, whereas the sum of endpoint log probabilities used by the estimator requires independent keys. The corrected v3 uses endpoint-specific batch keys and restarts from the same fresh seed 0 state; it does not reuse the invalid weights. The failed run’s latest/best checkpoints, history and logs remain retained and charged. A cross-filesystem configuration rename also failed before any training update; ordinary copy-and-unlink fixed it and its small failure log remains. These are recorded implementation repairs, not additional scientific remedies or extra selected endpoints.

## Reader diagnosis: scale imbalance is real, but rescaling is insufficient

| Old arm / TRAIN case | Top 4 B fraction, mean over M | Geometry share of prior | Geometry attribution in attention |
| --- | --- | --- | --- |
| GF 0006 | 0.440 | 0.959 | 0.960 |
| GF 0051 | 0.087 | 0.991 | 0.994 |
| GF 0348 | 0.090 | 0.983 | 0.985 |
| GF 0631 | 0.068 | 0.987 | 0.985 |
| GEE 0006 | 0.092 | 0.990 | 0.993 |
| GEE 0051 | 0.042 | 0.994 | 0.994 |
| GEE 0348 | 0.145 | 0.986 | 0.988 |
| GEE 0631 | 0.122 | 0.973 | 0.972 |

B is nullable-normalized across the whole group bank, while the original geometric term adds up to four order-one window weights. The old top-four postings omit most nonnull association mass in seven of eight cases; the G-F M1 exception retains 44%. Exact positive-prior attention attribution divides the final weight of each overlapping candidate according to its geometric and association contributions, then normalizes by total nonnull attention. This measures arithmetic attribution, not physical causality. The balanced singleton lowers geometric attention attribution to 0.821–0.906, adaptive to 0.596–0.806, but does not recover omitted relationships or guarantee that module proposals become useful.

The exhaustive bug is corrected: an already-posted B contribution is never added twice. Ordinary restricted-posting versus restricted/full-B replay isolates weighting changes; restricted/full-B versus exhaustive/full-B uses the same formula and normalizer and isolates candidate coverage. Balanced full-B diagnostics normalize by full per-module B mass; production remains conditioned on postings. Original no-E→E four-scene corrected fluid-T ratios are approximately 0.985/0.957/0.999/0.911, but corresponding surface ratios are 1.208/0.820/1.047/1.035. The retained-E→E corrected surface ratios are 0.781/0.703/0.975/1.039. Thus a favorable fluid-T change cannot validate the whole exchange, and wider coverage is not uniformly beneficial. These alternate readers are diagnostics, not new trained arms or production fallback.

![Figure 5. 05_reader_and_surface](../../diagnostics/round12_3_recovery_20261024/figures/05_reader_and_surface.png)

Figure 5. Upper panels measure actual applied B coverage and geometry-attributed attention on the fixed TRAIN0006/0051/0348/0631 panel. Lower panels use distinct fitted TRAIN0303/0322/0337/0348 and exposed DEV0277/0287/0294/0684, each M3/M5/M7/M10 and 25 modules. Each module’s signed surface residual is decomposed into its mean bias and demeaned circumferential residual; all masks and native T units are preserved. Bars are module-weighted RMS summaries of those selected arrays, not population confidence intervals.

| Arm | TRAIN4 bias RMS | TRAIN4 angular RMS | DEV4 bias RMS | DEV4 angular RMS |
| --- | --- | --- | --- | --- |
| gf | 2.457 | 1.907 | 2.931 | 2.083 |
| original | 2.068 | 2.014 | 2.026 | 2.031 |
| balanced | 2.561 | 1.856 | 3.066 | 2.019 |
| adaptive | 2.787 | 3.206 | 2.637 | 3.263 |

The earlier G-F versus G-EE surface excess is mostly coherent module offsets on the saved DEV panel: bias RMS 2.931 versus 2.026, angular residual RMS 2.083 versus 2.031. The balanced reader does not resolve that offset mechanism; adaptive pooling additionally loses circumferential detail on both fitted TRAIN and exposed DEV. Because the two panels contain different physical cases, this is a descriptive TRAIN/DEV comparison rather than an isolated overfitting estimate. The original M1/M2 TRAIN12 panel and DEV22 also differ in module-count distribution, so their aggregate gap cannot identify a cause. No evidence here uniquely attributes the surface failure to missing E→E.

## Predictor: retain gains and losses across all physical quantities

| DEV22 native RMSE | Original G-EE | Balanced singleton | Adaptive | B/O mean /p90 | A/B mean /p90 |
| --- | --- | --- | --- | --- | --- |
| u | 0.0985 | 0.1267 | 0.2483 | 1.287 /1.409 | 1.960 /1.897 |
| v | 0.0233 | 0.0282 | 0.0302 | 1.212 /1.187 | 1.071 /1.047 |
| p | 0.0916 | 0.1185 | 0.1057 | 1.294 /1.153 | 0.892 /0.956 |
| omega | 0.2828 | 0.3064 | 0.4182 | 1.084 /1.087 | 1.365 /1.344 |
| temperature_fluid | 2.6001 | 2.6288 | 4.2129 | 1.011 /0.998 | 1.603 /1.605 |
| temperature_surface | 2.7882 | 3.1486 | 4.5510 | 1.129 /1.042 | 1.445 /1.444 |
| temperature_material | 2.8158 | 2.7304 | 4.5953 | 0.970 /0.905 | 1.683 /1.938 |
| q_proxy | 8.8378 | 9.9655 | 11.6161 | 1.128 /1.143 | 1.166 /1.247 |
| module_peak_endpoint_rmse | 2.6893 | 2.5618 | 4.6775 | 0.953 /0.904 | 1.826 /1.827 |

Balanced singleton gains against the unchanged-reader G-EE include material-T and peak means and tails, and slightly lower heat-fluid/surface response errors. Surface mean regression is concentrated at M5/M7: mean ratios 1.315/1.170, versus 0.990/1.010 for M3/M10; its overall surface p90 is close to the original. Native u/v/p worsen 21–29%, and q proxy is worse than the fixed TRAIN-mean predictor. This is meaningful mixed progress, not a categorical rejection at 1.10. Adaptive temperature/peak means still beat their arithmetic TRAIN-mean predictors, showing nontrivial retained information, but its fluid/surface/material/peak errors rise substantially versus singleton; v,p and q proxy also exceed their constant baselines. No claim of preserved detailed physical quality follows from its small K.

![Figure 6. 06_physical_quality](../../diagnostics/round12_3_recovery_20261024/figures/06_physical_quality.png)

Figure 6. Top panels show all nine native DEV22 quantities and continuous paired-model ratios, with the unchanged descriptive reference lines; markers and ticks represent means andp 90. Bottom panels compare real saved heat and single-geometry reference changes, with pooled-record median/p90 and the zero-response predictor at 1. Reference and fit cohorts remain distinct in the numerical evidence; these are exposed development results from one seed.

| Physical response | Component | Original relative median | B relative median | A relative median | B/O absolute median /p90 | A/B absolute median /p90 |
| --- | --- | --- | --- | --- | --- | --- |
| heat | fluid_fields/temperature | 0.576 | 0.595 | 0.931 | 0.968 /0.986 | 1.714 /1.263 |
| heat | interface/T_surface | 0.526 | 0.485 | 0.842 | 0.926 /0.879 | 1.892 /1.625 |
| heat | module_peak_endpoint/temperature | 0.297 | 0.413 | 0.842 | 1.311 /1.093 | 2.014 /1.875 |
| heat | solid_temperature/temperature | 0.390 | 0.442 | 0.802 | 1.163 /1.143 | 1.785 /1.670 |
| single_geometry | fluid_fields/temperature | 1.204 | 1.299 | 1.471 | 1.237 /1.141 | 0.968 /1.586 |
| single_geometry | interface/T_surface | 2.353 | 2.270 | 1.670 | 1.092 /0.939 | 0.721 /1.220 |
| single_geometry | module_peak_endpoint/temperature | 1.316 | 1.522 | 1.680 | 1.561 /1.135 | 0.844 /1.645 |
| single_geometry | solid_temperature/temperature | 2.409 | 2.351 | 1.757 | 1.138 /1.024 | 0.639 /1.482 |
| mixed_central_derivative | fluid_fields/temperature | 2.058 | 2.578 | 1.722 | 2.468 /1.804 | 0.682 /0.739 |
| mixed_central_derivative | interface/T_surface | 5.230 | 10.342 | 7.507 | 1.689 /1.585 | 0.947 /0.615 |
| mixed_central_derivative | module_peak_endpoint/temperature | 4.253 | 14.039 | 11.900 | 1.101 /2.486 | 1.623 /0.556 |
| mixed_central_derivative | solid_temperature/temperature | 6.862 | 15.118 | 9.887 | 1.706 /1.946 | 0.840 /0.575 |

All eight atlas anchors are original source TRAIN families: only 0348 fits responses;0001/0318/0333 are train-reference-only and 0304/0320/0335/0350 response-validation-only. There are 16 heat,32 single-geometry and 8 centered mixed derivatives, with raw mixed contrasts separate. Heat medians retain useful information relative to zero response, including the adaptive arm; adaptive surface/material single-geometry medians improve over singleton, but every priority single-geometry relative median remains above 1 and mixed derivatives remain weak. The complete summaries retain all physical quantities, tails, cohorts, reference magnitudes and near-null exclusions; favorable preservation ratios never replace absolute physical usefulness. No M1/M2 physical geometry-response atlas exists.

![Figure 7. 07_native_fields](../../diagnostics/round12_3_recovery_20261024/figures/07_native_fields.png)

Figure 7. All native fluid receivers in the fixed exposed DEV0287/M5 scene, physical-reference u/v/p/omega/T, original G-EE, balanced singleton, adaptive and signed adaptive residual. Common scales within each physical quantity reveal smoother/coarser adaptive local behavior and unresolved thermal structure. Invalid/solid receivers remain masked; units are those of the packed benchmark. This selected field does not replace DEV22 statistics or certify physical derivatives.

## Organizer: pooled information is executed and reused; useful interaction-adaptive sharing is limited

Adaptive groups are learned from the first update, with no 20-epoch leaf phase, organization freeze, target K or forced merge. On 12 input-selected TRAIN calibration cases, relation/state/coordinate edge scales are measured before fitting. Three disjoint four-case gradient batches give task/work norm-ratio CV 0.103 and fixed coefficient 0.0192693; the work ramp reaches 0.0940 at 100. All 400 updates have nonzero native-policy and work-policy gradients. Sampled training mean K across the two layers falls from 4.50 at epoch 1 to 1.68 at 100; deterministic inference on TRAIN24 uses K¹=1 throughout, K²=1 on 22 cases and 191 on 0006/0366 (M1). This extreme coarse/almost-fine split is not evidence of useful multiscale organization.

The stage-price proxy is approximate: a noisy negative key/value projection slope is clipped to 0, and its measured constant E-row price dominates the proxy. At the last update, native-policy norm is 0.291, raw work-policy norm 0.000882; the scheduled work contribution is only about 1.6e-6 before the full-native cap. The cap remains 1 on all 400 updates. Thus there is no evidence that a large work penalty forced the coarse outcome. The calibrated coefficient’s existence does not establish an accurate runtime model; full execution is measured separately. Native/work parameter-slice norms, cosines, missing gradients and reached nodes are retained in checkpoint telemetry; no price is retuned at 100.

Both layers pool the actual contextual E states and applied M→E relation, transform only active groups, project keys/values once and use those same tensors in group→M and query reductions. On M2/M10/M12, two-layer group key/value rows fall 384→2 and Q8192 group-value slots 65,536→8,192 (87.5% fewer). M1 case 0006 retains 191 final groups and 8 query slots, so no query-read reduction occurs there. Original E embedding/contextualization and E→E remain; original-E query reads are 0 in both controls, so this is substitution for singleton-group work, not a newly achieved removal of direct-E reads. No full-member or all-E fallback occurs.

| Four TRAIN scenes | Clamp | u | v | p | omega | T fluid | T surface | T material | q proxy | T peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | layer 1 | 1.024 | 1.007 | 1.005 | 1.003 | 1.009 | 1.063 | 1.034 | 1.021 | 1.029 |
| balanced | layer 2 | 0.940 | 0.981 | 1.013 | 1.020 | 1.012 | 0.878 | 0.988 | 1.017 | 0.924 |
| adaptive | layer 1 | 0.997 | 1.000 | 0.998 | 1.000 | 0.998 | 1.001 | 0.999 | 0.998 | 0.998 |
| adaptive | layer 2 | 1.021 | 1.000 | 0.998 | 1.009 | 1.056 | 1.034 | 1.006 | 1.033 | 0.987 |

These are ratios of equal-case mean RMSE after the actual intervention versus ordinary inference, with downstream preparation recomputed; final address clamping preserves query candidates. Across all nine physical quantities, the exchange is neither uniformly useful nor uniformly harmful. Earlier G-F relay clamping had surface ratio 0.992 but peak ratio 1.258, and G-EE early clamping improved surface/material on the same panel; the full original results remain below. Zeroing group values establishes output dependence, but neither it nor interface identity proves useful organization. The unused/weak final exchange remains a simplification candidate; no new loss was added to force its role.

| Adaptive TRAIN case | Layer | Stop probability | Parent /children K | Parent–children fluid-T RMS | Surface-T RMS |
| --- | --- | --- | --- | --- | --- |
| 0006 | 1 | 0.550 | [1, 191] /[2, 191] | 0.00001 | 0.00001 |
| 0006 | 2 | 0.507 | [1, 191] /[1, 192] | 0.01743 | 0.01057 |
| 0051 | 1 | 0.597 | [1, 1] /[2, 1] | 0.00100 | 0.00177 |
| 0051 | 2 | 0.517 | [1, 1] /[1, 2] | 0.04853 | 0.02861 |
| 0348 | 1 | 0.764 | [1, 1] /[2, 1] | 0.00046 | 0.00073 |
| 0348 | 2 | 0.985 | [1, 1] /[1, 2] | 0.19419 | 0.02829 |
| 0631 | 1 | 0.759 | [1, 1] /[2, 1] | 0.00004 | 0.00004 |
| 0631 | 2 | 0.982 | [1, 1] /[1, 2] | 3.67752 | 4.63782 |

Parent/child replays use deterministic inference, force one reached internal choice and recompute every dependent later layer, address and native output; stochastic traversal does not confound these replays. TRAIN0348 layer 2 has stop probability 0.985 and small surface change 0.0283, whereas TRAIN0631 has similarly high probability 0.982 and surface change 4.638, about the scale of its ordinary error. High confidence is therefore not a reliable functional-sharing guarantee across scenes. Forcing the first reached coarse node is one bounded local audit, not exhaustive validation of all tree nodes. Zeroing R only inside the organizer’s tree/policy descriptors changes joint outputs by at most 1.83e-5 on these four cases and leaves their K unchanged, while actual M→E and pooled group→M relation priors remain applied. This limited dependence gives little evidence of useful relation-shaped subdivision at this endpoint; it does not mean the physical input has no effect on group values.

![Figure 8. 08_organization_and_cost](../../diagnostics/round12_3_recovery_20261024/figures/08_organization_and_cost.png)

Figure 8. TRAIN0348/M10 actual applied R, final E ancestry, group→M B links and reads of the same query group; its single root covers 192 E members. Lower panels show all 24 fixed organizer cases and measured complete costs on the same physical GPU 0. The flat ancestry map is the observed coarse outcome, not a visualization of a learned overlapping hypergraph. Complete timings include preparation and extraction; the optimizer macro includes native/work policy VJPs, response gradients, clipping and AdamW.

## Executor: reduced group operations and memory, higher full cost

| Same GPU 0 measurement | Original G-EE | Balanced singleton | Adaptive |
| --- | --- | --- | --- |
| Complete optimizer macro, s | 2.149 | 2.331 | 12.894 |
| Macro peak allocated, GiB | 2.108 | 2.120 | 1.566 |
| Complete native 0006 M1, ms | 60.52 | 61.93 | 106.22 |
| Complete native 0051 M2, ms | 61.69 | 62.70 | 105.27 |
| Complete native 0348 M10, ms | 63.85 | 65.60 | 108.37 |
| Complete native 0631 M12, ms | 64.52 | 65.96 | 108.91 |

Adaptive macro time is 5.53×singleton, despite 26.1% lower peak allocation (1.566 versus 2.120 GiB). Complete native inference is 1.65–1.72×singleton across the four scenes. Persistent prepared tensors occupy 230,165–528,845 bytes in adaptive versus 829,141–1,016,768 bytes in singleton. These are separate latency, memory and organization results; no complete speedup is claimed. The original baseline is retimed under the same current evaluator/GPU, while unchanged native/reference metrics are reused. One warmup and three measured timing macros restore weights/provider/named Adam state; they are engineering calls, not retained scientific training.

| Q | Singleton prepare /cached /complete ms | Adaptive prepare /cached /complete ms |
| --- | --- | --- |
| 32 | 9.82 /3.36 /13.15 | 53.89 /3.28 /57.04 |
| 1024 | 9.82 /6.10 /15.92 | 53.93 /5.96 /59.71 |
| 8192 | 9.81 /43.66 /53.43 | 53.78 /42.57 /96.50 |

Preparation grows from about 9.8 to 53.9 ms; Q8192 cached decoding only improves 43.7→42.6 ms. Measured preparation overhead overwhelms reduced group scores/reads. The adaptive implementation still executes 352 local edges,383 tree metadata nodes and 191 policy rows per layer, plus irreducible original-E/EE work; a root does not erase those costs. Individual tree/pooling/lookup neural substages are not fully isolated in endpoint wall timing, so Python control, synchronization and kernels are plausible implementation contributors rather than individually proved causes. A work counter reduction cannot be renamed a full runtime gain.

## Zero heating, inverse boundary and next valuable work

| Arm /family | Zero-heat fluid-T RMS | Surface-T RMS | Material-T RMS |
| --- | --- | --- | --- |
| balanced 0348 | 1.802 | 8.723 | 8.041 |
| balanced 0304 | 2.184 | 3.920 | 2.243 |
| adaptive 0348 | 2.054 | 7.779 | 7.414 |
| adaptive 0304 | 1.205 | 4.347 | 2.764 |

The applicable homogeneous-boundary zero-heating temperature-rise target remains 0. Both arms retain substantial nonzero rise; for fit 0348 surface RMS is 8.723 singleton and 7.779 adaptive. No analytical-temperature loss was combined with the chosen reader remedy, and no heat–flow independence was checked or constrained. Original-column temperature-only JVP/VJP probes remain model sensitivities, not physical derivative references. No inverse model, sample trail, design ranking, new physical solver or dataset was run; inverse gains and feasibility remain untested.

Predictor next work should explain the M5/M7 module offsets, lost angular detail and weak geometry responses, using the retained physical arrays rather than another broad training portfolio. Organizer next work should examine conditional parent/child task advantages, near-threshold inference choices and reached-node exposure; the current exclusive connected tree and its sampling prior are restricted hypotheses, and the coarse outcome should not be repaired by forced K or a 20-epoch leaf phase. Executor next work should remove or vectorize measured preparation overhead before expecting fewer group reads to yield lower full cost, and reconsider the weak final M address exchange without a compensating auxiliary loss. These are evidence-based directions, not authorized new launches. The present 100-epoch endpoints remain final; the budget is not spent merely because it remains available.

## Recovery verification, budget and delivery

The focused regression suite passes 51 tests, including independent/replayable response endpoint keys, unchanged legacy serialization, corrected exhaustive weighting/coverage, balanced-prior gradients, retained-E→E adaptive pooling, native adapter, maintained runner and response regressions. Changed core/task/evaluator files pass Ruff and the diff whitespace check. Actual 400-update histories, saved checkpoint reconstruction, DEV22 physical exports, all-reference responses, deterministic sharing interventions and full native/backward/response/policy/Adam timings supply the runtime evidence. Existing dataset/TEST separation, DATA 2 bindings, named optimizer/RNG/OOM machinery and the pre-push artifact hook are preserved; no new hash/freeze/snapshot or readiness service was introduced.

Conservatively charged recovery use is 1.503 GPU-hours; the original ledger’s 0.606 brings total Round12-3 use to 2.109 against the unchanged 24-hour initial/36-hour overall ceilings. The 8-hour recovery allocation is not exhausted and the additional 12-hour reserve remains unused. The charge includes the invalid epoch3 attempt, failed configuration launch, no-step calibration, bounded diagnostics and all endpoint/timing calls; no arm was extended. Cleanup removed 176 files/links and 70 directories (15,618,483 bytes), including fake pytest fixtures, four PDF QA rasters, rendered HTML and 18 superseded response PNGs. Raw response plots metadata stays unchanged; the cleanup receipt identifies those retired visual exports, while selected figure masters/companions and numerical response panels remain. All 261 protected retained-file metadata records were unchanged and every allowlisted payload is absent; no scientific checkpoint, physical-reference array, historical run or migration backup was removed. Retained recovery payload is approximately 167,779,621 bytes across 163 files, bringing Round12-3 to approximately 298,641,468 bytes, below 4 GiB. Remaining observed capacity is /data2 3,218,109,796,352 bytes (2.927 TiB), /home 84,060,975,104 bytes (78.288 GiB); the DATA2 links remain intact and the unrelated GPU2 process continues. Compact manifests, receipts and one-time helpers stay local on DATA2. Durable code/tests/ordinary configurations/English documentation are delivered on the existing non-default branch after the full outgoing-history artifact audit and remote-tip verification.

<!-- ORIGINAL_FIRST_SCREEN -->

### Original first100 screen: retained historical evidence

The new path removes ordinary E→E communication from G-F, independent retrieval membership, M compaction, direct M-value decoding and group-to-original-E expansion. Each of two interaction layers now contextualizes E from M, pools E into an explicit group representation, transforms that representation once, and uses its cached keys/values to update M. Queries read those same final group keys/values through bounded module postings and a geometric entrance. G-EE adds only an explicit E→E exchange as the matched communication control. The new core and native adapter execute; both scratch controls completed 100 development epochs. The adaptive G-H core completed native forward/backward/save/reload qualification, but G-H100 was not launched in the original screen because G-F failed the prespecified surface-temperature preservation bands. The subsequent recovery separates the E→E deletion hypothesis from adaptive organization, keeps the original thresholds and failures, and tests retained-E→E organization after a measured reader diagnosis.

**Original first-screen decision (superseded by the authorized recovery below):** stopped after the matched G-EE/G-F100 screen. G-F surface-temperature mean RMSE is 3.273 versus 2.788 for G-EE (ratio 1.174 > 1.10); its p90 ratio is 1.180 > 1.15. Heat-response preservation also misses several bands, single-geometry responses remain inaccurate against physical references, and zero-heat temperature rise remains nonzero. Deleting E→E saves training memory, but complete native inference improves by only 0.34–2.51% and the complete optimizer macro by 3.92%, below the 10% useful-gain criteria. No recovery reserve, continuation, additional architecture, inverse campaign, solver run or full-data training was used.

### Evidence and short figure index

The experiment uses fixed25 TRAIN150 and repeatedly exposed DEV22, normalization fitted only on selected TRAIN, seed 0, FP32, width 128/four heads/two layers, primary Q1024, nominal microbatch 24/effective 48 and AdamW 3e-4 / WD 1e-5. DEV22 is the selected part of the original source test partition, not an independent final test. The four detailed DEV cases are 0277/0287/0294/0684. All native quantities retain the packed dataset’s benchmark units; no new Kelvin, SI flux or physical-coordinate conversion is asserted. M1/M2 have six selected TRAIN examples each and no physical geometry-response atlas. They remain fitted TRAIN evidence.

| Figure | Question | PDF master |
| --- | --- | --- |
| 1 | All nine native boards and failed surface strata | [Native comparison](../../diagnostics/round12_3_execution_20261024/figures/01_native_comparison.pdf) |
| 2 | Native fields and signed residuals | [Fields](../../diagnostics/round12_3_execution_20261024/figures/02_native_fields.pdf) |
| 3 | Saved physical responses and zero-response usefulness | [Responses](../../diagnostics/round12_3_execution_20261024/figures/03_physical_responses.pdf) |
| 4 | Applied relations, reused values, interventions and complete native cost | [Reuse and cost](../../diagnostics/round12_3_execution_20261024/figures/04_group_reuse_and_cost.pdf) |

The retained [local evidence directory](../../diagnostics/round12_3_execution_20261024/) contains the input-only panel selection, original full histories/checkpoints through their run links, all-nine native metrics, physical response summaries, regional/intervention results, selected field/reference arrays, qualification evidence and cleanup receipt. Figures were generated with Python from these saved numerical files, visually inspected, and retained as PDF masters plus one PNG each for Markdown embedding. Generated payloads and one-time helpers are local on DATA2 and excluded from Git.

### Completed execution and fair exposure

| Arm | Physical GPU | Epochs | Primary visits / updates / microbatches | Engine seconds |
| --- | --- | --- | --- | --- |
| G-F | 0 | 100 | 15,000 / 400 / 700 | 865.04 |
| G-EE | 1 | 100 | 15,000 / 400 / 700 | 845.68 |
| G-H | 1 qualification only | no training | native M24 forward/backward ×3; no scientific Adam steps | 20.9 s for both qualification arms combined |

Each completed arm presents 15.36 million primary fluid queries, plus native surface/material/curl receiver work. Each has 100 response microbatches (50 heat and 50 single geometry), 200 response endpoint-scene visits and one separate two-scene no-step response calibration. Response fitting uses only admitted TRAIN0348 and the maintained 10%-of-shared-native-gradient target/downward cap before the same Adam step. Fixed coefficients are 1172.72 for G-F and 2027.75 for G-EE; these are gradient-balancing coefficients, not comparable physical loss units. Native objective weights remain fixed, including the explicitly defined epoch101 rule. No response fitting uses the other seven atlas families.

Common named initial tensors match exactly; only G-EE owns the extra E→E weights. Existing recorded case-order/query-sampling identities are compared in the retained verification receipt. Each arm saves epoch 100, latest and best-field; their model tensors are exactly equal within that arm because epoch 100 is its only declared monitoring checkpoint. Best-field is not an earlier, independently selected success. Native 100-epoch endpoints, saved response endpoints and the later intervention/cost passes are reported separately. A wrapper formatting error occurred after each completed native export; the exports were retained and reused unchanged while the missing response stage was subsequently run.

The initial qualification used real high-M24 batches with native forward/backward and strict checkpoint reconstruction, rather than a metadata dry run. Warm G-F forward/backward was approximately0.85/0.37s, peak about1.76GiB; G-H approximately2.25–2.32/0.54–0.57s, peak about1.62GiB. G-F native reload difference was0; the largest G-H native extracted difference was2.83e-5. This is an untrained sizing/reload observation, not a trained adaptive field or speed result and not bitwise GPU parity. The untrained qualification states and numerical receipt are retained. No work price or adaptive work-learning qualification was completed after the Stage-B stop.

### Predictor: useful native temperature information, failed preservation

| Native quantity | G-F mean RMSE | G-EE mean RMSE | Mean ratio | p90 ratio | TRAIN-mean predictor | G-F worst ID |
| --- | --- | --- | --- | --- | --- | --- |
| u | 0.0880 | 0.0985 | 0.894 | 0.907 | 0.4544 | 0677 |
| v | 0.0208 | 0.0233 | 0.894 | 0.915 | 0.0259 | 0684 |
| p | 0.0744 | 0.0916 | 0.812 | 0.782 | 0.0995 | 0688 |
| omega | 0.2574 | 0.2828 | 0.910 | 0.896 | 0.6913 | 0686 |
| temperature_fluid | 2.6952 | 2.6001 | 1.037 | 1.121 | 6.4489 | 0677 |
| temperature_surface | 3.2731 | 2.7882 | 1.174 | 1.180 | 7.1019 | 0663 |
| temperature_material | 2.9202 | 2.8158 | 1.037 | 1.100 | 7.0225 | 0663 |
| q_proxy | 7.9908 | 8.8378 | 0.904 | 0.954 | 7.5305 | 0645 |
| module_peak_endpoint_rmse | 2.6520 | 2.6893 | 0.986 | 0.902 | 6.5792 | 0655 |

Means are equal-case errors on native masks; p90 is over cases and is not a confidence interval. The module-peak board is the per-case RMSE of errors in separately recomputed original-module maxima over the identical stored material receiver sets. Both models beat the constant selected-TRAIN mean on all four priority temperature/peak boards; G-F’s q-proxy mean is worse than its constant baseline (7.991 versus7.530). G-F improves all four flow means relative to G-EE, but these improvements do not cancel the surface-temperature preservation failure.

![Figure 1. native_comparison](../../diagnostics/round12_3_execution_20261024/figures/01_native_comparison.png)

Figure1. Epoch100, exposed DEV22. The left panel divides equal-case native RMSE by the selected-TRAIN mean predictor error on the same masks; lower than1 beats that arithmetic baseline. The right panel shows surface-temperature means and p90 in native temperature units, with case counts. G-F/G-EE surface mean ratios are1.025/1.383/1.169/1.107 for M3/M5/M7/M10; the M7 p90 ratio is1.357. Thus the global failure is not a uniform temperature collapse, and the M5/M7 tails cannot be hidden by the mean fluid-temperature or peak boards.

![Figure 2. native_fields](../../diagnostics/round12_3_execution_20261024/figures/02_native_fields.png)

Figure2. Input-selected DEV0287 (M5), all stored native fluid receivers, with geometry and observation masks preserved. Rows are u/v/p/omega/T in native benchmark units; columns are saved physical reference, G-EE100, G-F100 and signed G-F minus reference. Reference/prediction scales match within each quantity; residual scales are separate. The figure exposes remaining local wake, pressure and thermal errors rather than certifying a physical surrogate from a scalar objective. Numerical fields for all four detailed DEV representatives remain retained, including surface/q-proxy/material targets and masks.

#### M1/M2 TRAIN risk and boundary regions

| Quantity, TRAIN12 | G-F mean | G-EE mean | Mean ratio | p90 ratio |
| --- | --- | --- | --- | --- |
| u | 0.0797 | 0.0928 | 0.859 | 0.859 |
| v | 0.0141 | 0.0163 | 0.862 | 0.910 |
| p | 0.0552 | 0.0626 | 0.883 | 0.937 |
| omega | 0.1881 | 0.2055 | 0.915 | 0.959 |
| temperature_fluid | 1.9697 | 1.8622 | 1.058 | 1.223 |
| temperature_surface | 2.0331 | 2.2612 | 0.899 | 1.020 |
| temperature_material | 1.8958 | 1.9339 | 0.980 | 0.937 |
| q_proxy | 6.2719 | 7.3631 | 0.852 | 0.856 |
| module_peak_endpoint_rmse | 1.4462 | 1.2507 | 1.156 | 1.012 |

The fixed M1 cases are 0006, 0368, 0382, 0383, 0394, 0044; M2 cases are 0051, 0054, 0067, 0435, 0087, 0452. Selection uses module count and sorted Re/total-heat inputs only. On this fitted panel, G-F fluid-temperature p90 is1.223×G-EE and mean module-peak error1.156×G-EE, although mean surface/material errors do not regress. Boundary strips are the outer10% of the normalized domain; far-from-M means distance greater than0.15 times the shorter domain length. Their full u/v/p/omega/T results, counts and IDs are retained in each execution result. These measurements cannot establish M1/M2 physical geometry-response fidelity because the required reference does not exist in the admitted atlas.

### Organizer: values are used and reused; adaptive aggregation remains untested

Both completed controls have K_E¹=K_E²=192 on all 24 input-selected organizer scenes. Original module count remains M, and there is no learned M-value bank. Their singleton partitions bypass nomination/tree/policy work. Consequently these endpoints demonstrate a shared information/value route and bounded access, not learned grouping, compression or scene-adaptive subdivision. The connected exclusive G-H frontier is implemented and reversible, uses 352 catalogue-local edges per layer and linear children/statistics/ancestry-label storage, and has no target K or20-epoch leaf warmup. Its temperature/work schedules are continuous over absolute epochs1–500, with learning allowed at the first update. G-H100, parent/child physical sharing audits, TRAIN stage-price calibration and work-gradient qualification were not performed because the preceding communication screen failed. No result from an untrained small K is promoted into hypergraph success.

| G-F TRAIN case | M | Clamp early relay / ordinary T RMSE | Clamp final address / ordinary | Zero group values / ordinary | Exhaustive group search / ordinary |
| --- | --- | --- | --- | --- | --- |
| 0006 | 1 | 0.9956 | 1.0000 | 3.317 | 0.990 |
| 0051 | 2 | 0.9581 | 1.0004 | 3.287 | 0.955 |
| 0348 | 10 | 1.1051 | 1.0009 | 2.863 | 1.010 |
| 0631 | 12 | 1.1899 | 1.0005 | 2.903 | 0.900 |

The early intervention removes the actual group→M message in layer1, then recomputes every dependent later layer and query. It worsens high-M fluid-T errors by10.5% and19.0%, but improves the two small-M examples, so usefulness is scene-dependent. The final intervention removes the final group→M message while preserving the original group query candidates; it changes fluid-T RMSE by less than0.09% in G-F. Selected proposal logits and final M states have nonzero native gradients, including M1, but the measured final exchange utility is weak and not established. This is a reason to simplify or reject this version in subsequent design work, not to force a new loss or bypass that makes the path indispensable.

Zeroing the cached final group values under fixed query candidates multiplies G-F fluid-T RMSE by2.86–3.32. This establishes executed value dependence on the bounded four-scene panel. The same prepared group key/value tensors consumed by group→M are the source tensors gathered by the query reader; they are not independently reconstructed after M changes. The ordinary decoder reads zero original-E values and zero M values, with at most two module proposals, eight postings, nine spatial cell visits, four geometric entrances and eight retained group candidates. Empty/overflow/truncation counts are retained. These are computational relationships, not physical causal relations.

![Figure 4. group_reuse_and_cost](../../diagnostics/round12_3_execution_20261024/figures/04_group_reuse_and_cost.png)

Figure4. Top: actual applied R and B for original input column0 on TRAIN0348 (M10), followed by read frequency of the same final singleton group IDs for the first 1024 native-union queries. Each group is one E record here; no adaptive group diagram is implied. Bottom: real early/final-message, zero-value and exhaustive-group-search interventions on TRAIN0006/0051/0348/0631 (M1/M2/M10/M12), and complete native inference on the same idle physical GPU0. Group values carry useful information, but final address-update utility is weak; inference gains remain0.34–2.51%.

The original exhaustive implementation counted already-posted B contributions twice. Its historical physical metrics and the exhaustive bars in Figure4 are superseded diagnostic evidence and must not support a coverage or fidelity claim. The corrected recovery replay counts each B once and separates restricted/full-B weighting from exhaustive/full-B coverage. Both read the same prepared groups, not original E. Restricted search still scores eight allocated singleton slots rather than192, a95.83% arithmetic reduction; this alone cannot qualify physical quality or complete runtime. Same-function chunking/replay tests remain distinct from alternate-address comparisons.

### Responses: heat information is useful; geometry and zero heating remain failures

All eight physical atlas anchors remain original source TRAIN families. Only 0348 is response-fit;0001/0318/0333 are train-reference-only;0304/0320/0335/0350 are response-validation-only. None is relabeled as independent source DEV/TEST. Evaluation uses existing common-valid saved physical receivers:256 input-selected fluid points,16 surface angles per original module and all saved material points, with maxima recomputed separately for each endpoint. There are16 heat differences,32 single-geometry differences and8 centered mixed derivatives. Raw mixed four-corner contrasts are kept separately and never pooled into single geometry.

| Response kind | Priority component | Median absolute-error ratio GF/GEE | p90 ratio | GF median error / physical RMS | GEE median error / physical RMS | Near-null excluded |
| --- | --- | --- | --- | --- | --- | --- |
| heat | fluid_fields/temperature | 1.149 | 1.048 | 0.667 | 0.576 | 0 |
| heat | interface/T_surface | 1.030 | 1.028 | 0.532 | 0.526 | 0 |
| heat | solid_temperature/temperature | 1.065 | 1.176 | 0.402 | 0.390 | 0 |
| heat | module_peak_endpoint/temperature | 1.227 | 1.115 | 0.373 | 0.297 | 0 |
| single_geometry | fluid_fields/temperature | 0.905 | 0.958 | 1.191 | 1.204 | 0 |
| single_geometry | interface/T_surface | 0.957 | 0.755 | 1.659 | 2.353 | 0 |
| single_geometry | solid_temperature/temperature | 0.901 | 0.930 | 1.819 | 2.409 | 0 |
| single_geometry | module_peak_endpoint/temperature | 1.151 | 1.124 | 1.381 | 1.316 | 1 |
| mixed_central_derivative | fluid_fields/temperature | 0.870 | 1.199 | 1.617 | 2.058 | 3 |
| mixed_central_derivative | interface/T_surface | 0.718 | 0.952 | 4.926 | 5.230 | 1 |
| mixed_central_derivative | solid_temperature/temperature | 0.722 | 0.947 | 6.255 | 6.862 | 1 |
| mixed_central_derivative | module_peak_endpoint/temperature | 0.795 | 1.121 | 3.638 | 4.253 | 2 |

A zero-response predictor has relative error1 for non-null references. Heat-response medians beat it in all four priority components for both models, but G-F heat-fluid median preservation1.149 and heat-peak1.227 fail1.10; material p90 preservation1.176 also fails1.15. Single-geometry relative median errors exceed1 for all four G-F priority components, despite several favorable GF/GEE ratios, and the peak median ratio1.151 also fails preservation. Mixed derivatives remain poor in absolute relative terms; near-null entries are excluded only from relative metrics and retain absolute errors and reference magnitudes. Full results, tails and original identities for every component, including flow and q proxy, remain in the physical response summaries.

| Cohort | Kind | Records | Fluid-T GF/GEE median error ratio | GF median relative error | GEE median relative error |
| --- | --- | --- | --- | --- | --- |
| fit_train_0348 | heat | 2 | 1.176 | 0.778 | 0.661 |
| fit_train_0348 | single_geometry | 4 | 1.131 | 1.017 | 0.924 |
| fit_train_0348 | mixed_central_derivative | 1 | 1.167 | 1.178 | 1.010 |
| train_reference_only | heat | 6 | 1.205 | 0.687 | 0.570 |
| train_reference_only | single_geometry | 12 | 0.759 | 1.136 | 1.413 |
| train_reference_only | mixed_central_derivative | 3 | 1.772 | 1.617 | 2.058 |
| response_validation_only | heat | 8 | 1.107 | 0.626 | 0.561 |
| response_validation_only | single_geometry | 16 | 0.996 | 1.581 | 1.451 |
| response_validation_only | mixed_central_derivative | 4 | 0.719 | 3.668 | 3.487 |

![Figure 3. physical_responses](../../diagnostics/round12_3_execution_20261024/figures/03_physical_responses.png)

Figure3. Upper panels show pooled-record relative RMSE medians and p90 for the four priority quantities; these spans are descriptive tails, not confidence intervals. Dashed1 is the zero-response predictor. Lower panels compare actual saved reference fluid-temperature changes against both epoch 100 models for response-validation-only0304, with the same units on both axes; the mixed panel is a second position derivative in native temperature/position² units. Heat changes are partly useful, whereas single/mixed geometry behavior remains inadequate. Cohort-specific metrics and near-null absolute errors are retained separately.

| Model | Control family | Zero-heat fluid-T RMS | Surface-T RMS | Material-T RMS | Max absolute surface T |
| --- | --- | --- | --- | --- | --- |
| G-F | 0348 | 1.389 | 6.442 | 5.387 | 12.843 |
| G-F | 0304 | 1.806 | 2.054 | 0.676 | 4.628 |
| G-EE | 0348 | 1.536 | 4.782 | 3.904 | 11.592 |
| G-EE | 0304 | 1.953 | 2.285 | 1.235 | 5.495 |

The analytical homogeneous-boundary zero-heating temperature-rise target is0. Both models fail the bounded check on fit0348 and response-validation-only0304; G-F0348 surface RMS 6.442 and maximum 12.843 native T units are substantial. The objective did not add an analytical loss, heat superposition constraint or heat-independent-flow test. Temperature-only JVP/VJP probes retain every original heat column (10 on0348,3 on0304) and show nonzero sensitivities through complete re-preparation. They are model sensitivity evidence, not physical derivative validation or a replacement for the failed finite geometry references. No full Jacobian, flow-invariance probe or new physical solver was used.

### Executor: preparation and memory savings do not become a native speed claim

| Query rows | GF prepare ms | GEE prepare ms | GF cached ms | GEE cached ms | GF complete core ms | GEE complete core ms |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 8.120 | 9.395 | 3.124 | 3.113 | 11.221 | 12.482 |
| 1024 | 8.120 | 9.364 | 5.668 | 5.596 | 13.746 | 14.961 |
| 8192 | 8.081 | 9.367 | 40.519 | 40.133 | 48.586 | 49.201 |

These are averages of per-scene medians over the four fixed TRAIN scenes, each with a warmup and three repeated measures on the same idle RTX6000 Ada GPU0. Complete native inference additionally includes target-free input binding, native query/stencil union construction, interpolation, physical extraction and role joins: G-F/G-EE times are56.840/58.304ms (M1),58.097/58.317ms (M2),60.577/60.800ms (M10),61.054/61.263ms (M12). Small-Q32 complete-core speed improves about10.1%, but Q8192 complete-core improves only1.25% and the complete native calls only0.34–2.51%. The native ≥10% speed criterion is not met. Both trained functions differ, and this single-seed development comparison does not establish statistical significance.

| Arm | Complete macro s | Native forward s | Native backward s | Response/policy s | Clip/AdamW s | Peak allocated GiB |
| --- | --- | --- | --- | --- | --- | --- |
| G-F | 2.0572 | 1.2281 | 0.6589 | 0.1392 | 0.0021 | 1.574 |
| G-EE | 2.1411 | 1.2532 | 0.6827 | 0.1587 | 0.0030 | 2.110 |

The disposable timing macro uses the exact checkpoint’s named AdamW state,48 primary cases in two 24-case microbatches, native backward, the every-fourth-update response gradient, clipping and AdamW. One warmup and three recorded engineering steps per arm are restored afterward; they are not scientific training epochs or retained trained checkpoints. G-F’s median2.0572s is3.92% lower than G-EE2.1411s, so no ≥10% training-speed gain is claimed. Its measured peak1.574 GiB versus2.110GiB is25.40% lower: a real memory benefit from deleting E→E in this implementation, accompanied by the failed fidelity bands. Concurrent first 100 engine times (865.0/845.7s on separate GPUs) are exposure/runtime records and are not isolated causal speed measurements.

Actual counters distinguish allocated/padded operations from live positive-support relationships. On M12, two-layer encoder score pairs are38,016 in G-F versus332,928 in G-EE (88.58% fewer); the ordinary query reader still allocates eight group slots per query in both. At 8192 query rows it computes262,144 head/group scores and gathers65,536 group-value slots, with zero original-E and zero M-value query expansion. These trained controls therefore show no learned member-read reduction relative to G-F. Persistent prepared tensor storage is829,141–1,016,768bytes across the 24 organizer scenes for both arms. Both layers’ prepared group vectors are charged. Earlier training-history ME/MG/query counters recorded live masked support and omitted padding; those histories are preserved, while the corrected endpoint counters explicitly retain allocated and live quantities. No prediction tensor or scientific metric was changed by this accounting correction.

Substage timings are available for complete preparation, cached decode, complete core/native calls, native forward/backward, response/policy callback and clipping/AdamW. Individual nomination/pooling/index/lookup neural substages were not separately profiled; singleton controls have no nomination or frontier VJP work. No calibrated adaptive arithmetic/runtime work price or executor gain from G-H is asserted. Restricted search and zero ordinary member expansion are measured computational properties, not proof that a learned compact organization retained physical utility.

### Inverse, decision and next work

No inverse head, sample trail, ranking campaign or physical design was run. Group diagrams, smooth original-source derivatives and bounded candidate diversity do not establish valid designs, physical feasibility or regret. Predictor gains are limited to selected native quantities and useful heat-response medians; predictor misses include surface preservation, q-proxy usefulness, geometry responses and zero heating. Organizer gains are executed/reused singleton values and useful high-M early relay; organizer misses are untested adaptive aggregation and weak final-address utility. Executor gain is measured training memory; complete native/training speed targets are missed. These axes remain separate.

The original decision proposed diagnosing local/surface communication and source-law failures and considering simplification of the weak final exchange. The newly authorized recovery instead diagnoses the reader first and permits a matched retained-E→E adaptive experiment without requiring the E→E deletion bands to pass. Preserve the matched controls and physical references when testing one change. The original data do not justify a small-K success claim, forced leaves/roots, a target K, decoder rescue branches or full-data training. The current recovery has its own finite allocation and endpoint; it does not extend an arm until metrics pass. This connected exclusive tree remains a restricted organizational hypothesis; neither this stop nor an untrained root choice settles unrestricted hypergraph organization.

### Delivery, verification and storage

The final focused suite passed 36 tests in 16.14s, including core aggregation/reuse, padded/empty donor handling, M1/M2, legal keyed frontiers, tiny-tree task/work enumeration, policy baseline/retry state, query bounds/chunking, adapter and maintained runner/native/response regressions; new files also pass Ruff. Actual high-M native qualification, both400-update histories, strict checkpoint reconstruction, saved-checkpoint native/reference evaluation and complete execution timings provide the runtime evidence. Existing TEST separation, DATA2 bindings, checkpoint/schema/RNG checks, named optimizer ownership, OOM retry machinery and the pre-push artifact hook are retained. No new cryptographic hashes, contract-freeze service, baseline snapshots or readiness gates were introduced; existing dataset/sampling identities are reused. The transient wrapper-print failure and the G-H GPU native reload discrepancy remain recorded rather than erased.

Conservatively charged GPU use is 0.606 GPU-hours against 24 initial/36 overall; measured training is 0.475 GPU-hours. Native wrapper time is charged at three minutes per arm because its separate elapsed print failed; response and execution passes are measured. The 12-hour reserve is unused. Retained round/run payloads are 130,861,847 bytes (<4 GiB); the declared estimated peak temporary budget was256 MiB under the512 MiB limit. Cleanup removed 187 files/links and 84 directories, totaling 13,796,499 bytes, including 18 superseded response PNGs,4 PDF QA rasters and exact round-owned test/bytecode scratch. Early tests used standard /tmp; their exact round-created trees were removed and the final suite used explicit DATA2 temporary storage. No scientific checkpoints, numerical/reference arrays, historical runs or protected migration backups were removed. The original cleanup inventory and completion receipt retain the temporary helper failure/recovery and confirm all allowlisted entries absent and protected retained metadata unchanged. Remaining observed capacity after cleanup is /data2 3,218,377,691,136 bytes (2.927TiB) and /home 84,089,159,680 bytes (78.314 GiB); DATA2 is mounted and all three declared bindings remain intact. Other jobs may change filesystem free space independently.

The durable implementation is [interaction_groups](../../src/honf_forward_core/interaction_groups/), the distinct [Thermal adapter](../../Case_ThermalChannel/src/channelthermal/interaction_groups.py), [task](../../Case_ThermalChannel/src/channelthermal/training/interaction_group_task.py), [execution evaluator](../../src/honf_runtime/interaction_group_evaluation.py) and [recipe guide](../guides/Thermal_Interaction_Group_Reuse_Recipe.md). The three ordinary versioned research configurations are linked from the guide; G-H remains a provisional untrained recipe requiring the unperformed adaptive calibration/qualification before any separately selected training stage. Durable source/tests/config/docs are committed and pushed on the current non-default branch after whole-outgoing-history audit. Generated data, figures, temporary helpers, manifests and receipts remain local on DATA2.
