# Wind fidelity and shared execution consolidation

**Completed bounded development, 8 October 2026.** Branch `agent/honf-core-next`; reviewed scientific parent `f792dd6`; artifact-directory baseline `6205f47`. This report executes the [approved consolidation plan](../../UpgradePlan/HONF_Wind_Fidelity_and_Shared_Execution_Consolidation_Plan.md). Figures, checkpoints, arrays and one-time probes remain local in ignored `diagnostics/generated/wind_consolidation_20261008/`; the durable report and implementation are versioned.

## What changed and what the measurements decide

The existing source-preserving refinement family and TrainingEngine were retained. New versioned recipes add TRAIN-only component balancing, queries independent of microbatch packing, configurable execution tiles, bounded CPU geometry caching and two backends for the same adaptive function. A single wider host, H128/message 128 with the existing eight environment records, was selected after the W0/W1/W2 screen. TRAIN line sweeps justified one compact continuous gate for its Adaptive child. No architecture reset, new solver labels or inverse campaign occurred.

**Predictor gains.** W0 Full, selected W3 Full and its matched W3 Adaptive child all completed 2,500 development epochs with 100-epoch reviews. Against fresh W0 at the same age, W3 Full reduces near-turbine Uy/Uz mean RMSE by **72.5%/68.3%**, and downstream Uy/Uz by **63.0%/61.2%**. Near Ux mean is essentially unchanged and its worst-row error rises only 2.45%. Native maps visibly recover transverse structure that W0 leaves almost flat. The additional resources purchased actual denser supervision for W2, a justified 3.70-times larger W3 parameter count, and substantial matched maturation; they were not spent filling VRAM.

**Predictor misses.** Loss balancing alone and Q4096 at H64 did not produce a satisfactory 500-epoch host. W3's 500-epoch near-Ux tails were worse than W0; continued healthy maturation reduced that tradeoff. The final subset-trained model does not replace the mature full-TRAIN references: Wind2201 has more than twice as good Ux RMSE on both displayed native planes. The retained classics also remain markedly more accurate in every component of both native planes. High-module wake cuts still miss deficit depth, transverse sign changes and amplitude. These are exposed development results, with unequal training exposure relative to Wind2201/classics, not independent population or TEST evidence.

**Organizer gains.** Adaptive retains the Full host's fidelity: across all 15 component-role means the largest worsening is 0.33%, across p95 values 1.25%, and across worst-row values 3.60%. The public continuous gate, source identities and physical input derivatives are tested. A real optional correction changes the nonlinear Wind output and restores exactly. Dense-masked and selected execution produce identical Adaptive native outputs, support and weights on the measured panels.

**Organizer misses and costs.** The mature TRAIN retained-fine fraction is approximately 98.1%, and training still evaluates the full fine replay. The high-module native plane retains 95.2% of fine pairs. Selected execution is slower than dense-masked execution: 225.93 versus 214.97 ms on eight alternating complete calls, a 4.85% dense-masked advantage. The organizer has an executed effect but no useful sparse-training saving here. Equal-work geometry controls sometimes improve reference error; learned support is not uniformly optimal or physical causality.

**Inverse evidence and next steps.** No inverse application is qualified. Wind's three stored directions and local derivatives provide interface evidence, not independently solved perturbation truth. Thermal's saved 0291 maximum-temperature decision still fails; the bounded continuous-gate transfer does not repair it. Continue this forward research direction, retain H128/component-balanced/Q1024 as the selected development recipe, and use the separately prepared [manual-only 5,000-epoch Full/Adaptive handoff](../guides/Unified_Interaction_Manual_5000.md). A Full forward reference is the first useful formal comparison; the Adaptive partner remains an optional matched research comparison. The user chooses whether either starts. Full-data cold catalogue and validation costs remain unmeasured.

## Protocol, screen and maturation

Wind membership remains `wind_shared_fixed24_v1`: 24 TRAIN layouts/72 direction rows and eight DEV layouts/24 direction rows, with all three stored directions per layout. The fixed TRAIN selection omits M 18 and contains 72/420 source TRAIN rows (17.1%); its name must not be interpreted as a literal quarter of TRAIN. TEST 90 targets remain locked. All new scientific Wind arms use fresh fine weights, seed 42, effective batch 24, the same TRAIN normalizer/background-profile procedure, E8 context and five input-defined sampling roles. DEV rows are correlated directions of eight repeatedly exposed layouts. Neither displayed plane selects training cases or calibrates loss scales.

W0 uses the current scalar-role objective. W1 changes only to component-balanced scales; W2 changes W1's Q1024 to Q4096. Both scale rules use the same four input-selected TRAIN layouts/12 direction rows and Q1024 calibration procedure; component scales are profile-residual RMS including mean, with the declared 0.009 m/s/minimum-relative floor. W3 is the sole conditional representation follow-up: H64/message 64 becomes H128/message 128, retaining E8 and Q1024. No E64 or second architecture arm was tried. H64 has 182,532 parameters and H128 674,628.

| Screen at epoch 500 | TRAIN objective | Q/case | Width | Common old scalar DEV score | Near mean Ux / Uy / Uz, m/s |
|---|---|---:|---:|---:|---|
| W0 | Scalar role | 1024 | 64 | 0.123050 | 0.59249 / 0.19678 / 0.12935 |
| W1 | Component balanced | 1024 | 64 | 0.134035 | 0.66713 / 0.19155 / 0.12896 |
| W2 | Component balanced | 4096 | 64 | 0.133046 | 0.65706 / 0.19027 / 0.12882 |
| W3 | Component balanced | 1024 | 128 | 0.120783 | 0.63665 / 0.16151 / 0.12707 |

W3 improved all 15 role-component means relative to W1/W2 at 500, but its near-Ux mean/p95/worst were 7.45%/9.4%/13.6% worse than W0. The sealed selection receipt retained this tradeoff before maturation; its SHA 256 is `7a3b1a6b509a4be055aceb81b7d49ea0a5fa39ab0cd29dea845246f15904b4`. More Q was not combined opportunistically with width. W1/W2 stop at their declared 500 screens; W0 and W3 Full mature to 2500. Adaptive branches from W3's exact 500 weights, all optimizer moments and provider identity, with one explicit TRAIN-evidence-bound gate amendment. Corresponding Full/Adaptive case order, query hashes, schedules and visits match for every epoch 501–2500.

The absolute 2500 schedule retains warmup 500, open through 600, soft through 800 and deployed routing from 801, with LR hold through 1000 and subsequent decay. Both lineages pay the same all-fine supervision/replay. TRAIN-only expected-work calibration at 601 sets coefficient 0.0011537307, achieving 5% gradient share below the 0.1 coefficient cap. All scheduled 100 reviews passed finite-loss/gradient and checkpoint checks. The stage name hard denotes deployed inference; its C1 gate remains continuous. Wind has no configured response guard; Thermal's response guard is measured separately. Best-field selection uses saved monitoring checkpoints identically; all three mature Wind best and terminal ages are 2500. There are no new 25-epoch snapshots.

### Figure family 1: physical learning and measured work

![Physical learning curves for the five bounded Wind arms](../../diagnostics/generated/wind_consolidation_20261008/figures/physical_scorecard/wind_physical_learning_curves.png)

Figure 1a. Saved 100-epoch DEV 24 monitoring shows W0's nearly flat transverse errors and the later H128 recovery. At 2500, W3 Full near Uy/Uz are 0.054008/0.040670 m/s against W0's 0.196517/0.128265; Adaptive gives 0.052759/0.039466. All-fine host supervision is shared within the final pair. W1/W2 endpoints are 500, so the figure does not portray them as mature failures.

![Mature physical scorecard and common validation score against updates queries and recorded TRAIN time](../../diagnostics/generated/wind_consolidation_20261008/figures/physical_scorecard/wind_mature_scorecard_and_work.png)

Figure 1b. Bars are means over 24 correlated DEV direction rows; ticks and crosses are p95 and worst-row values. The lower panels use the common old scalar-role evaluation score, not each arm's rescaled TRAIN objective. Q4096 costs four times the primary query draws per visit. Recorded TRAIN-epoch time includes in-epoch cold catalogue construction and excludes provider setup, validation, saves and replayed/failed work; complete process charges appear below. [PDF masters](../../diagnostics/generated/wind_consolidation_20261008/figures/physical_scorecard/) retain readable vector text.

## Native fields and wake fidelity

All final panels use literal epoch 2500 checkpoints and the same retained native cells, physical coordinates and reference arrays as the old Wind2201 comparison. Low-M row 69 is layout 23/M 8; high-M row 426 is layout 142/M 30; both have stored direction 270 degrees. The nearest native horizontal plane is z=70.86995 m with 52,548/99,750 cells. There is no spatial interpolation or new CFD. These are the dataset's stored shared-grid wake references, without a separately established discretization-error floor. Axis lengths use rotor diameter D. Empirical backgrounds and normalizers are TRAIN-owned.

For each component, physical and signed-error color limits are common across both scenes. Ux limits pool reference 0.5–99.5% quantiles; Uy/Uz use pooled absolute-reference 99.5%; residual limits pool method/scene absolute 99.5%. Each panel states display clipping. RMSE and relative L2 use all cells without clipping. A small Ux relative L2 must not conceal much larger transverse relative errors.

### Figure family 2: low/high-module Ux, Uy and Uz

![Low-module native Ux reference Wind2201 selected Full Adaptive and signed residuals](../../diagnostics/generated/wind_consolidation_20261008/figures/native_fields/wind_native_row69_Ux.png)

Figure 2a. Low-M Ux RMSE is 0.071244 m/s for retained full-TRAIN Wind2201,0.154683 for subset W3 Full and 0.158667 for Adaptive. The new subset fit remains worse on this channel despite its improvement against fresh W0. Different training populations and ages make this a retained-reference context comparison.

![Low-module native Uy fields and signed residuals](../../diagnostics/generated/wind_consolidation_20261008/figures/native_fields/wind_native_row69_Uy.png)

Figure 2b. Low-M Uy RMSE is 0.033152/0.022705/0.022526 m/s for Wind2201/Full/Adaptive. The new fields recover clearer transverse wakes, while signed residuals expose misplaced lobes and remaining local artifacts. Adaptive relative L2 is 0.322, which remains substantial.

![Low-module native Uz fields and signed residuals](../../diagnostics/generated/wind_consolidation_20261008/figures/native_fields/wind_native_row69_Uz.png)

Figure 2c. Low-M Uz RMSE is 0.021133/0.008250/0.007828 m/s. The large Wind2201 transverse artifacts are reduced; Adaptive relative L2 remains 0.763. Color saturation in the old field is explicitly labelled rather than hidden by separate scales.

![High-module native Ux fields and signed residuals](../../diagnostics/generated/wind_consolidation_20261008/figures/native_fields/wind_native_row426_Ux.png)

Figure 2d. High-M Ux RMSE is 0.118109/0.295110/0.287386 m/s. Both new models miss native deficit amplitude and exhibit structured residuals; Adaptive's small mean gain over Full is not closure of the mature-reference gap.

![High-module native Uy fields and signed residuals](../../diagnostics/generated/wind_consolidation_20261008/figures/native_fields/wind_native_row426_Uy.png)

Figure 2e. High-M Uy RMSE is 0.053643/0.052205/0.052343 m/s. This displayed scene gains only about 2–3% against Wind2201, substantially less than the matched W0 population gain; relative L2 remains approximately 0.55.

![High-module native Uz fields and signed residuals](../../diagnostics/generated/wind_consolidation_20261008/figures/native_fields/wind_native_row426_Uz.png)

Figure 2f. High-M Uz RMSE is 0.022510/0.011117/0.010917 m/s. Transverse amplitude is improved against Wind2201, but periodic-looking structure and relative L2 near 0.68 persist. Full's native high-M Uz at 2500 is approximately 13% worse than its 1000 plane despite improved population scores; maturation was not uniformly beneficial at every location.

### Figure family 3: exact native cross-wake cuts

![Native low-module wake cuts at source0 plus5D and plus10D](../../diagnostics/generated/wind_consolidation_20261008/figures/native_fields/wind_native_wake_cuts_row69.png)

Figure 3a. Cuts use the existing native x columns nearest source 0+5D/+10D and the native z plane nearest hub height, with y offsets within±3D. Low-M x is 1135/1531 m and z/D=0.88587. Lines connect native samples without creating continuous solved profiles. W3 reconstructs much more Uy variation than W0, but the+5D positive lobe and+10D shape remain misplaced.

![Native high-module wake cuts at source0 plus5D and plus10D](../../diagnostics/generated/wind_consolidation_20261008/figures/native_fields/wind_native_wake_cuts_row426.png)

Figure 3b. High-M x is−184.5/211.4 m at the same native z/D. The selected pair remains nearly coincident, with missed Ux depth and substantial Uy/Uz profile errors. These exact cuts prevent a favorable population mean from being mistaken for faithful local wake structure.

## Artifact localization, sampling and continuity

The native physical transform, background addition, frame, source IDs, receiver order and chunk joins were checked against saved arrays. Old Wind2201 bands and spots persist under same-weight all-fine reads, so routing is not their sole origin. Input-only TRAIN sweeps on two scenes nevertheless found consequential legacy hard-route jumps: maximum velocity jumps 0.05653/0.01795 m/s at a 1e−5D step. That diagnosed an additional route seam and enabled one declared `compact_c1_v1` amendment; it does not prove that a continuous gate fixes the host's field artifacts.

The new law remains `B + g(F−B)`, with all physical sources retaining base paths and global context. For optional score p, `t=clip((p−0.35)/0.30,0,1)` and `g_optional=t²(3−2t)`; smooth near protection combines as `g=near+(1−near)g_optional`. The inner near region stays fully fine; the transition annulus has continuous positive corrections. Closed-route training recovery uses the already paid fine replay with an isolated router surrogate. Public inference derivatives follow the actual live deployed gate, not that surrogate. The gate is a new mathematical recipe, not a parity retrofit of old formal weights.

### Figure family 4: host versus route contribution

![High-module same-weight all-fine Adaptive fields and actual route-induced changes](../../diagnostics/generated/wind_consolidation_20261008/figures/artifact_localization/adaptive_same_weight_artifacts_row426.png)

Figure 4. At identical Adaptive 2500 weights, all-fine versus deployed C1 high-M native RMSE is Ux0.29587 versus 0.28739, Uy0.05240 versus 0.05234, Uz0.01183 versus 0.01092 m/s. Route-induced RMS changes are 0.10036/0.01121/0.00674 m/s. Much of the visible structure survives all-fine, while routing alters a localized region; visual similarity alone does not identify a causal training mechanism. Low-M counterpart and exact arrays remain in the [local artifact appendix](../../diagnostics/generated/wind_consolidation_20261008/figures/artifact_localization/).

Physical scalar-role learning gradients explain a useful starting hypothesis: on the actual W0 epoch 100 batch, last-output Ux/Uy/Uz gradient norms are 0.079068/0.000257/0.000199, ratios approximately 308/397. Effective 24 versus four microbatches of 6 preserves exact query IDs, targets and roles; max parameter-gradient difference is 6.71e−8 and loss difference 1.49e−8 under FP 32 with TF 32 disabled. Later W1/W2 TRAIN-only gradients still show shared-capacity competition, including Uy/Uz shared-encoder cosines approximately 0.86/0.83 and base auxiliary norm shares 16.3%/8.4%. These are diagnostics, not proof of a unique cause.

Q1024 already samples all five roles. A geometry-only ten-epoch stream on three TRAIN rows passes 150 per-role Q1024/Q4096 prefix comparisons; Q4096 supplies approximately 3.85–4-times more unique native cells in those requests. Cross-role duplicate cells remain under their declared weights, with summed cross-role duplicate sampled IDs 65–889 over three TRAIN rows × ten epochs at Q1024 versus Q4096; they are not independent physical observations. The bounded gradient/sampling receipts and old hard-seam negatives are retained. No disposable fit weights initialize any scientific arm.

## Organizer evidence and its limits

### Figure family 5: actual information flow and stored operating directions

![Actual physical source and environment ancestry with one Wind correction removed and restored](../../diagnostics/generated/wind_consolidation_20261008/figures/w3_adaptive_e2500_interaction_graph_effect_v1/wind_interaction_graph_and_effect.png)

Figure 5a. DEV row 70/direction 285 has eight physical sources and eight environment records. Every physical source retains its base path and, at this receiver, all eight also have fine weight 1; source 0 is unprotected. Removing only its fine correction increases vector reference error by 0.059480 m/s, and restore is bitwise exact. This measured nonlinear output effect is in m/s; latent message amplitudes are not relabelled as velocities. The graph shows computational ancestry, not physical causality.

![Saved Thermal source0 coefficient heat and native-cell remove restore effect](../../diagnostics/generated/wind_consolidation_20261008/figures/thermal3903_saved_source_intervention_b_v1/thermal3903_saved_optional_source_effect.png)

Figure 5c. The retained Thermal source0 coefficient changes one native receiver; its `(F−B)×heat` correction is−0.105581 T. The active correction worsens that point's reference error, while restoration is exact. Geometry and heating remain fixed; this saved exposed0291 intervention supports execution evidence B without organizer utility or causality.

On fixed DEV row 70/direction 285, removing one optional source 0 correction changes Ux/Uy/Uz by−0.058974/−0.009968/−0.001769 m/s. Pointwise vector reference error rises from 0.26265 to 0.32213 m/s, so this particular correction helps. Restore is bitwise exact. The saved Thermal3903 intervention has the opposite fidelity result: its exact correction is−0.105581 native T; normal/removed predictions−0.075062/+0.030519 versus target 0.000909 give absolute errors 0.075971/0.029610. This active correction harms that receiver by 0.046362 T. Both are evidence of executed information flow and restoration, not universal organizer usefulness or physical causality.

OrganizerA compares Adaptive with same-weight all-fine on all 24 frozen DEV direction rows. Its pooled-query RMSE is distinct from monitoring's mean of 24 row RMSEs. Equal-work nearest/upstream/shuffle controls on rows 69/426 preserve positive-support counts, near protection and the multiset of nonzero gate weights. Low-M controlled Ux RMSE 0.13936–0.13953 is slightly worse than learned 0.13932; high-M nearest/upstream approximately 0.23957 is better than learned 0.24195, while shuffle 0.24526 is worse. Assignment is consequential, but learned routing does not uniformly win.

The original OrganizerA diagnostic mistakenly attached alphabetically ordered role names to the provider's numeric role indices. Query coordinates, predictions, masks and arrays were correct. A SHA-bound interpretation amendment renames only those roles and retains the original JSON/NPZ; its corrected pooled Adaptive BG/downstream/hub/near/volume values appear in the appendix. No query, target, training update or ranking selection was repeated for that correction.

![Stored270 and 285 degree native cross-wake fields for the final Adaptive checkpoint](../../diagnostics/generated/wind_consolidation_20261008/figures/stored_direction_crosswake_w3_adaptive_e2500_v2/stored_direction_crosswake_270_285.png)

Figure 5b. Stored 270/285-degree examples preserve physical source IDs and correct transformed coordinates/vectors; native grids are 302×174×64 versus 290×186×64. Each 33-cell source 0+5D line uses its own nearest native cells without interpolation. Ux/Uy/Uz RMSE is 0.112824/0.014712/0.002683 and 0.126926/0.013139/0.003141 m/s. Adaptive and separately evaluated all-fine coincide on these plotted cells, despite full-plane fine work 406,219/420,384 and 421,284/431,520. The fields vary with stored operating state; this is not an independently solved yaw, thrust or geometry response.

Local public receiver/center JVP central-difference discrepancies on two TRAIN scenes are at most 3.32e−5/1.35e−4 and 1.995e−4/7.33e−5 m/s/D at 0.002D; selected/dense query-VJP discrepancy is at most 1.86e−8. The trained line sweeps do not encounter a fractional optional-gate boundary, even after bounded input-only source search. Synthetic mixed-gate tests cover the C1 law, but native learned-boundary coverage remains unavailable. OrganizerC and inverse transfer remain unvalidated. The [final C1 qualification receipt](../../diagnostics/generated/wind_consolidation_20261008/probes/selected_w3_adaptive_e2500_c1_qualification_v2/receipt.json) corrects an earlier diagnostic flag that looked for the gate outside the nested resolved recipe; the raw flag remains preserved.

## Complete execution and resource costs

Training preserves effective 24 while using microbatch 24 for this Wind panel. Real Q4096 optimizer-boundary benchmarks with the full declared objectives compare micro 4/8/16/24 at tile 512, then micro 24 at 2048/8192. Warm epoch throughput rises from 56.96 cases/s at micro 4 to 99.44 at micro 24. Larger TRAIN tiles are slightly slower (approximately 97.6/95.6 cases/s), so TRAIN keeps 512. Peak micro 24 Q4096 allocation/reservation is approximately 10.04/10.55 GiB. This is measured allocator memory, not compute utilization. The 64GiB geometry cache stores CPU catalogues and no GPU target prefill; the 96 selected TRAIN/DEV catalogues occupy 12.262 GiB. Full-TRAIN cache residency/eviction remains unmeasured.

### Figure family 6: same-function costs and useful GPU work

![Measured TRAIN microbatch tile throughput memory headroom and cold setup](../../diagnostics/generated/wind_consolidation_20261008/figures/training_resources/wind_training_resource_benchmark.png)

Figure 6a. Six bounded configurations preserve 72 visits/three optimizer updates/effective 24 per epoch, with 24 disposable optimizer updates excluded from scientific initialization. Cold provider setup 63.344 s and first TRAIN geometry 241.589 s dominate the 313.900 s benchmark envelope; the best cached epoch is 0.7241 s. These costs motivate reusable catalogues, not a claim that larger tiles improve TRAIN.

![Complete prepared and input-VJP costs on matched Q1024 Q8192 panels](../../diagnostics/generated/wind_consolidation_20261008/figures/selected_w3_adaptive_e2500_costs_v1/wind_selected_execution_costs_queries.png)

Figure 6b. One warmup and eight alternating repetitions measure Q1024/Q8192 at fixed low/high M. Complete calls rebuild scene/context, transfer CPU queries, convert to physical velocity and return CPU output; prepared calls reuse scene/context; input VJPs also return query gradients. Checkpoint/HDF 5 opening and catalogue construction are separately cold. Widening inference tile 512→2048 reduces selected Q8192 complete medians approximately 30.6→20.0 ms at M 8 and 31.0→22.8 ms at M 30. Canonical route evaluation remains aligned to 512 so execution tiling does not change saved support. Ordinary public calls now omit full pair arrays unless explicitly requested.

![Full native plane same-function backend timings and allocator costs](../../diagnostics/generated/wind_consolidation_20261008/figures/selected_w3_adaptive_e2500_costs_v1/wind_selected_execution_costs_native_planes.png)

Figure 6c. The repeated high-M tile 2048 result is selected 225.93 ms (p10–p90:225.74–227.61) versus dense-masked 214.97 ms (214.39–216.98), with identical outputs/masks/probabilities/gates. Peak allocation is approximately 0.291 versus 0.242 GiB. Selected evaluates 2,848,987/2,992,500 physical fine pairs; dense-masked evaluates all 2,992,500. A 4.8% logical reduction does not produce an executor win. Same-weight all-fine is approximately 55.13 ms and predicts a different function; it is not a semantics-preserving fallback. Full native low-M comparisons outside this repeated high-M panel are single passes, explicitly labelled. A no-grad omission contaminated an earlier repetition attempt; its receipt is retained and excluded.

![Complete native Q8192 costs for the selected model and retained classics](../../diagnostics/generated/wind_consolidation_20261008/figures/classic_native_cost_comparison_q8192_v1/wind_classic_and_adaptive_complete_costs.png)

Figure 6d. On identical frozen Q8192 CPU-input/CPU-output calls and GPU 2, selected W3 Adaptive medians are 19.34/22.19 ms at M 8/M 30, versus K 6 Run2102 saved e2440 at 69.04/68.70 and dense Run2103 saved e2475 at 41.91/43.32. Classics retain their 512-record context. Repeated outputs are identical and match prior classic predictions within 9.54e−7 m/s. This is a cost comparison of different saved functions and training histories, not an attribution of speed to adaptive sparsity or a fair accuracy-training comparison.

A separate actual-lineage comparison loads the literal W3 Full2500 and Adaptive2500 checkpoints on GPU2, with one warmup/eight alternating complete calls at tile2048. Full normative all-fine versus deployed selected Adaptive medians (p10–p90) are Q8192/M8 **6.798 ms (6.772–6.895) versus 19.525 (19.490–19.623)**, Q8192/M30 **8.490 (8.462–8.538) versus 22.329 (22.319–22.359)**, and the high-M full plane **55.101 (54.729–55.223) versus 225.308 (224.726–226.898)**. Incremental peak allocation is 48.4/77.3,173.0/282.3 and177.5/287.0 MiB respectively above both loaded models. These distinct trained lineages predict different functions; this 4.09-times plane cost difference is separate from same-weight all-fine intervention and selected/dense parity. The [complete lineage cost receipt](../../diagnostics/generated/wind_consolidation_20261008/probes/full_lineage_normative_allfine_q8192_and_plane_v1/full_lineage_normative_allfine_costs.json) includes individual repetitions, residency and input bindings.

Dense-masked tile 2048 is the measured preferable deployment backend for this nearly dense C1 checkpoint. Selected remains an explicit reproducible override. No automatic target-error dispatcher, precision relaxation or compiler campaign was introduced. Sparse fine rows are actually counted at inference; base and gate still touch every physical pair, E8 environment work remains, and TRAIN full replay pays all fine padded rows. No useful sparse TRAIN/executor-saving claim is supported. Profiler traces retain gather/scatter, route dispatch and kernel measurements; they do not establish CUDA nonzero as the sole latency cause.

## Thermal preservation and conditional transfer

Thermal3901/3902/3903, frozen flow sources and prior successful affine heating semantics remain preserved. Read-only 3902/3903 baseline/minus/plus native fluid/interface/material fields and source-resolved S/K arrays reproduce retained values bitwise. Prepared FP 64 accumulation preserves precise heating increments; cold legacy FP 32 subtraction failures remain visible. Flow heat-increments are exactly zero in all four channels. Widening execution tile 512→2048 changes retained field values by at most a few 4.24e−5 and retains VJP cosine 1 with maximum discrepancy approximately 5.4e−6; measured 3903 prepared forward/VJP cost falls 47.23→36.36/93.29→69.46 ms. These tolerance-qualified execution changes do not alter old checkpoints or route mathematics.

### Figure family 7: Thermal peak failure and bounded transfer

![Saved Thermal0291 per-module peaks biases responses and peak-cell mismatches](../../diagnostics/generated/wind_consolidation_20261008/figures/thermal_0291_peak_diagnosis_saved_fields_v3/thermal_0291_peak_diagnosis.png)

Figure 7a. The retained 0291 native baseline/plus global maxima are 18.520241/18.129892 T, both set by module 1. Formal 3903 predicts 19.058064/19.127707 and chooses baseline, realizing 0.390348 T regret. Module 3 bias is+1.087/+1.145 T, versus module 1+0.538/+0.559; a wrongly favored module changes the global maximum despite relatively close local secants. Same-weight all-fine ranks plus correctly (19.17305→18.99535) but retains approximately+0.948/+1.012 T module 3 bias. This identifies a route contribution and a host-bias limitation; the failed saved decision remains unrepaired and was never fitted as a ranking target.

![Matched Thermal hard and continuous-gate development children through 500](../../diagnostics/generated/wind_consolidation_20261008/probes/thermal_transfer_pair_v1/figures/thermal_transfer_fixed25_comparison.png)

Figure 7b. The sole transfer pair starts from the exact preserved fixed25_v1 adaptive development e2500 parent, all 65 named AdamW states, provider/calibration and RNG, not formal3903. Each child visits 150 TRAIN cases for 500 epochs with effective 48/micro 8 (19 microbatches, four updates/epoch), identical case/query streams, seed 0 and a disclosed fresh constant 3e−6 child clock. DEV 22 is repeatedly exposed; its original canonical source partition includes sourceTEST labels and is not an independent test. At 500 hard/C1 fluid-surface-material means are 0.7357/0.7033/0.6937 and 0.73436/0.70475/0.69404 T. Module-peak mean/p90 is 0.8129/1.1835 versus 0.81335/1.18014; response guard 1.0858/1.0752 passes 1.10. The 0291 module-peak prediction error remains 1.56315/1.56761 T versus parent 1.54902. Both best-field children are 200, distinct from terminal 500. Changes are mixed and approximately 1%; retain the Thermal hard default, with C1 opt-in only.

## Numerical and provenance appendix

### Mature physical component table

Values are means of per-row physical RMSE on the fixed24 DEV panel, in m/s. The corresponding p95/worst values are visible in Figure 1b and retained in its source QA; these are direction-correlated descriptive statistics.

| Role | W0 Ux / Uy / Uz | W3 Full Ux / Uy / Uz | W3 Adaptive Ux / Uy / Uz |
|---|---|---|---|
| Background | 0.072465 / 0.017784 / 0.016655 | 0.036965 / 0.007826 / 0.007760 | 0.036441 / 0.007793 / 0.007758 |
| Downstream | 0.226978 / 0.088684 / 0.065528 | 0.203612 / 0.032841 / 0.025435 | 0.202100 / 0.032564 / 0.025049 |
| Hub slab | 0.189103 / 0.081480 / 0.051245 | 0.173671 / 0.030845 / 0.021573 | 0.170812 / 0.030850 / 0.021247 |
| Near turbine | 0.272312 / 0.196517 / 0.128265 | 0.272219 / 0.054008 / 0.040670 | 0.269782 / 0.052759 / 0.039466 |
| Volume | 0.115116 / 0.038568 / 0.031426 | 0.088656 / 0.014968 / 0.012877 | 0.087204 / 0.015017 / 0.012696 |

The common old scalar field scores are W0=0.041217891826, Full=0.026323823136 and Adaptive=0.025582303936. Corrected OrganizerA pooled-query Adaptive BG/downstream/hub/near/volume Ux-Uy-Uz RMSE is respectively 0.037182/0.007962/0.007934;0.205156/0.033578/0.025524;0.176723/0.032621/0.022698;0.273641/0.053866/0.040477;0.090577/0.015977/0.013563. Pooled versus mean-row aggregation accounts for their difference from monitoring.

### Native fidelity versus complete-call context

The table reuses immutable classic/Wind2201 native plane arrays and the selected2500 exports. RMSE covers the full horizontal plane; timing is the complete Q8192 call for the same scene on GPU2, so these are explicitly different query scopes. All classic cells are more accurate here. Classics/Wind2201 use fullTRAIN and different widths/context/Q/schedules; this is a measured reference frontier, not an equal-training experiment.

| Scene | Saved function | Native-plane Ux / Uy / Uz RMSE, m/s | Complete Q8192 median, ms |
|---|---|---|---:|
| M8 row69 | Classic K6 | 0.043776 / 0.006814 / 0.001619 | 69.04 |
| M8 row69 | Classic dense | 0.025785 / 0.004654 / 0.001212 | 41.91 |
| M8 row69 | W3 Full2500 | 0.154683 / 0.022705 / 0.008250 | 6.798 |
| M8 row69 | W3 Adaptive2500 selected | 0.158667 / 0.022526 / 0.007828 | 19.525 |
| M30 row426 | Classic K6 | 0.045591 / 0.007473 / 0.002219 | 68.70 |
| M30 row426 | Classic dense | 0.036856 / 0.007931 / 0.001729 | 43.32 |
| M30 row426 | W3 Full2500 | 0.295110 / 0.052205 / 0.011117 | 8.490 |
| M30 row426 | W3 Adaptive2500 selected | 0.287386 / 0.052343 / 0.010917 | 22.329 |

### Work, cold costs and failures

Unique successful Wind training totals **8,000 epochs,576,000 case visits,24,000 optimizer updates and 700,416,000 primary query draws**. Each mature lineage represents 180,000 visits/7,500 updates/184.32 million primary queries, but W3's shared first 500 is counted only once in campaign totals. W1/W2 each represent 36,000 visits/1,500 updates, with 36.864/147.456 million draws. Draws are not unique cells or independent physics. Thermal adds 150,000 visits/4,000 updates over 1,000 combined child epochs.

| Process scope | Charged seconds | Interpretation |
|---|---:|---|
| W0 screen/replayed warmup 500 | 1609.369 | Includes interrupted work and replay, not 500 clean warm epochs |
| W0 Full 501–2500 | 995.429 | Complete extension process |
| W3 shared warmup 500 | 1010.524 | Count once for the matched pair |
| W3 Full 501–2500 | 1080.523 | Complete extension process |
| W3 Adaptive 501–2500 | 1195.577 | Complete extension process; overlaps Full on the other authorized GPU |
| Thermal hard/C1 child 500 | 725.299 / 760.376 | Incremental 0.20147/0.21122 GPU-associated hours; saved parent cost excluded |

| Unique successful segment | Active physical pairs | Evaluated fine rows / cheap rows / gate rows | Logical selected-detail rows |
|---|---:|---:|---:|
| W0 warmup 1–500 | 663,552,000 | 1,092,206,592 each | 663,552,000 |
| W0 full_detail 501–2500 | 2,654,208,000 | 4,366,417,920 each | 2,654,208,000 |
| W1 warmup 1–500 | 663,552,000 | 1,092,206,592 each | 663,552,000 |
| W2 warmup 1–500 | 2,654,208,000 | 4,368,826,368 each | 2,654,208,000 |
| W3 warmup 1–500 | 663,552,000 | 1,092,206,592 each | 663,552,000 |
| W3 full_detail 501–2500 | 2,654,208,000 | 4,366,417,920 each | 2,654,208,000 |
| W3 adaptive_detail 501–2500 | 2,654,208,000 | 4,366,417,920 each | 2,577,136,743 |

Provider-recorded successful source-read counters total 12,607,488,000 active physical pairs and 20,744,699,904 evaluated fine rows, with the same padded cheap/gate count. These sum each unique segment once, exclude lost/replayed/disposable work, and include padded sources in evaluated-row counters. They count actual source-reader materialization under TRAIN full-fine replay, not FLOPs or sparse-kernel speed; E8 whole-scene context remains additional work. Logical selected-detail rows do not reduce these paid fine rows. [Per-segment and final-epoch counters](../../diagnostics/generated/wind_consolidation_20261008/output/report_qa/provider_recorded_work_counts.json) retain near support and padded capacities.

W3 warmup history records 640.535 s TRAIN/229.540 s validation; Full extension 887.334/114.775 and Adaptive 1002.039/115.182 s. These successful-epoch TRAIN sums include cold in-epoch catalogue construction; process envelopes additionally include loading, state/saves and overhead. First child epoch 501 takes approximately 242 s while constructing TRAIN catalogues, and first 600 validation approximately 114 s. Cached e2001–2500 TRAIN means are Full 0.323512 and Adaptive 0.380195 s/epoch; final 100 means 0.322335/0.377323. These measurements replace the earlier parameter-linear forecast.

The final accounting audit reports **2.863 GPU-associated hours**, 53 process rows (37 completed, 15 failed, one externally interrupted), no active processes and 2.864 elapsed hours at the receipt. Physical GPUs 0 and 2 were mapped by UUID and logical `cuda:0` inside each isolated process; GPU1 was not used. This remains well within 16 aggregate/10 elapsed hours, with substantially more than the reserved final hour available. Subsequent CPU report rendering and Git closeout add elapsed time without GPU training. The ledger charges cold starts, GPU-bound tests, failures and replay rather than only kernel or successful training time.

W0's first continuation was externally SIGTERM-interrupted without a numerical traceback. It had completed 199 epochs while latest remained 100; 99 completed epochs (7,128 visits/297 updates/7.299072 million primary draws) plus unknown partial 200 work were lost and replayed from the safe 100 checkpoint. Original 619.063 s and replay 535.203 s charges are retained. Five failed starts have inferred wrong-base relative paths; ten nonzero diagnostic/profile attempts later succeeded but some have no retained stderr, so their precise cause is not independently classifiable. No scientific numerical-training failure occurred. CPU OrganizerA failed/interrupted probes and the role-label amendment remain separately documented.

### Scientific bindings and manual-only formal handoff

| Binding | SHA 256 |
|---|---|
| Common new development component scales | `dec8cdbd0ce5a6a37b9fd572fea6e6d7aaeae4df1e2c083ed3c4b35bfbbf8fd2` |
| W3 Full warmup 500 checkpoint | `8207208426e389b98b41964f7212be2b1d13bdebcf3b3b520f638abf4cef1f26` |
| W3 Full 2500 checkpoint | `a2f4acbb6b22be144b7105bed720fe55f3e2648e7a3a17fc2ba9822c0a03675a` |
| W3 Adaptive 2500 checkpoint | `334669bbf0229758910db9ccac98db94b4f8367b870a2892ba7989978b461329` |
| W3 Full 2500 native export NPZ | `9f7b4f40617b6a995f62f582dc7d54d0dab795fbf70cf7220bdd2eba03cfcfcd` |
| W3 Adaptive 2500 native export NPZ | `0781caf36ed4e99ba479015ba81ad78cfbb724904658571bcb43936b4280383d` |
| Original OrganizerA JSON / NPZ | `747462371d319bbc6a53703f2290527ff5dced6f7a3d45c02056c380c64fc84d` / `7e526bf5dfc28f81b414b0cd1d13aad2fe8f826b7dd259a14098ec7c7196f070` |
| Manual full-TRAIN Full recipe | `b66ed6ae5c6b641f485bc2ed061bccd12ca7d3d6f1f66bd2a68b75b5d6fcf30a` |
| Manual full-TRAIN Adaptive recipe | `29f0da05c44349960569f8eaeee62e309c9e11a8d4a5e5b20808c6d0fea55114` |
| Fresh full-TRAIN normalizer / scales | `8c8d776696074216cd691a9d159899ade5b10135426b709deb19e9251946cd01` / `9efb13809c37b113019678647eb94fe9ef21ef79b13df5237976f3383b1292ff` |

The prepared full-TRAIN identities contain 420 TRAIN/90 VALID/90 TEST metadata rows. Normalization is freshly fitted on fullTRAIN; component scales use the same calibration method on four input-selected full-TRAIN layouts, not imported subset values. CPU-hidden prepare/dry-run and lifecycle regression checks are inert: zero formal optimizer steps/checkpoints, zero TEST target reads. The guide supplies prepare, dry-run, start, status, clean-stop and strict-resume commands for at most this one selected pair. At effective 24 there are 18 updates/full-data epoch, including a 12-case tail. The 5000 horizon and LR hold 2000 are a separately labelled schedule, not a resume of subset training.

Scaling only measured cached TRAIN time 72→420 gives approximately 2.62/3.08 hours for 5000 Full/Adaptive epochs. This deliberately incomplete estimate excludes cold provider/catalogues, full VALID, checkpointing, cache eviction and the new schedule's actual behavior; no measured complete formal forecast or formal readiness launch is claimed. Preserve old formal models and use explicit backend/gate identities if a future run is chosen.

### Tests, protection and repository closeout

The [broader failure classification](HONF_Wind_Consolidation_Test_Classification.md) compares the reviewed parent, artifact baseline and touched paths rather than describing focused passes as repository-wide success. The originally mentioned 37-node artifact was unavailable; a fresh baseline has 40 broader core/Thermal failures, of which 24 resource-dependent nodes pass on authorized GPU 2. Fourteen persist as inherited failures:10 core (expired deadline, stale digest, missing fixture architecture, four entmax 15 fixtures, three ledger expectations) and four Thermal ignored-diagnostic imports. No new failure was introduced in the affected comparison. Current Wind CPU tests pass 231 with two scheduled GPU skips; focused adapter/provider checks pass 50 with one CUDA skip, and the hidden 24 resource nodes pass 24. Other engine/core/Thermal focused checks and Ruff passed. Native GPU probes qualify outputs, actual support and derivatives in addition to deterministic tests.

The original goal-start 488-path extended inventory receipt was accidentally overwritten by a misparsed local snapshot command; its replacement is retained transparently and the original cannot be recovered. The independent pre-goal 100-row receipt covers 98 unique protected paths, all of whose current hashes match its expected/actual bindings. All 488 freshly inventoried protected files have modification times before this goal began; 390 unique paths are outside those older 98. Fresh preclose/end hash, size and resolved-path comparison protects closeout, but does not recreate a cryptographic goal-start baseline for those 390 paths. This is an audit-evidence limitation, not a reported source-file deletion or change. Historical resolved paths absent from the old receipt are not retrospectively certified.

A final membership scan after the first documentation push found 57 old Thermal3903 paths absent: the complete run directory had moved under `Trained_Results/ThermalChannel/HONF_Forward_Runs/bk/Run_3903_20261007_211636_unified_adaptive_full5000_v1` during closeout. All 57 archived file hashes, sizes and modification times match the prior endpoint; the move actor is not established. A relative compatibility symlink restores the original run path without moving or rewriting the archive. The [final relocation-aware audit](../../diagnostics/generated/wind_consolidation_20261008/accounting/protected_final_after_verified_relocation.json) covers exactly all 488 logical bindings with unchanged content/hash/size/mtime and explicitly records 57 changed resolved paths. The earlier automatic rescan covered only 431 surviving members and its subset-only unchanged result was insufficient; that receipt and the initially invalid closeout flag remain preserved with the correction.

Formal 3901/3902/3903, Wind2201, classics and protected histories remain in place. The solver ledger is still **326/326**, with zero new attempts. Generated figures, arrays, checkpoints and one-time diagnostics remain ignored and are excluded from every outgoing commit. Durable revisions have been committed/pushed after full outgoing-history/object audits and the configured repository artifact hook. Final Git tip verification and the closeout inventory are recorded in the local audit companion. Pre-existing unrelated plan/report working-tree changes are preserved.

### Compact figure and evidence index

| Family | Selected presentation | Saved evidence |
|---|---|---|
| 1 | Physical learning and mature scorecard | [metrics, hashes, reviews and work QA](../../diagnostics/generated/wind_consolidation_20261008/figures/physical_scorecard/source_and_visual_qa.json) |
| 2 | Six low/high-M native component pages | [coordinate/reference bindings, unclipped metrics and clipping QA](../../diagnostics/generated/wind_consolidation_20261008/figures/native_fields/source_and_visual_qa.json) |
| 3 | Two exact native wake-cut pages | Same native source QA; saved x/z/flat-cell indices |
| 4 | Same-weight host/route board | [artifact metrics and source QA](../../diagnostics/generated/wind_consolidation_20261008/figures/artifact_localization/source_and_visual_qa.json) |
| 5 | Actual interaction effects and stored directions | [Adaptive native fields/effects](../../diagnostics/generated/wind_consolidation_20261008/probes/native_w3_adaptive_e2500_six_sections_effects_cpu_v1/) |
| 6 | TRAIN resources, native/prepared/VJP costs and classic context | [cost PDF master](../../diagnostics/generated/wind_consolidation_20261008/figures/selected_w3_adaptive_e2500_costs_v1/wind_selected_execution_costs.pdf), [training QA](../../diagnostics/generated/wind_consolidation_20261008/figures/training_resources/source_and_visual_qa.json) |
| 7 | Thermal0291 and conditional transfer | [saved peak diagnosis](../../diagnostics/generated/wind_consolidation_20261008/figures/thermal_0291_peak_diagnosis_saved_fields_v3/), [transfer metrics and PDF](../../diagnostics/generated/wind_consolidation_20261008/probes/thermal_transfer_pair_v1/figures/) |
| Audit | Nine scientific segments, matched streams, all accounted process rows | [final inventory](../../diagnostics/generated/wind_consolidation_20261008/probes/final_campaign_inventory_20261008.json), [lineage table](../../diagnostics/generated/wind_consolidation_20261008/probes/final_campaign_lineages_20261008.csv) |

Selected PDF masters are retained with small raster companions only for direct Markdown embedding. Superseded presentation exports are removed only after their replacements are numerically bound and visually inspected; scientific arrays, failures and monitoring histories remain. This report's local links and rendered figure pages are checked during closeout.
