# HONF unified training and adaptive interaction refinement: completed formal 5,000-epoch comparison

Updated 2026-10-08. Thermal Run3903 and Wind Run2201 have completed their separately authorized full-data 5,000-epoch training. This report replaces the earlier development-stage headline with measured long-run inference, actual organizer reads, complete native inference prices, saved physical response comparisons, and a finite saved-candidate inverse check. Thermal references are **1404, 1501, 1804, and R-direct3902**; Wind references are the mature **2102 classic K6 and 2103 dense pairwise** checkpoints. Historical checkpoints and completed 3901/3902 are preserved.

**Predictor:** Thermal3903 improves mean fluid-temperature RMSE by **6.23% over R-direct3902**, and has the lowest mean surface/material-temperature RMSE among these five endpoints. Dense1804 retains substantially better fluid-temperature fidelity: 3903 is **35.27% worse** on that metric. Wind2201 loses to both mature native references in all five physical query roles, despite its longer epoch horizon. The formal runs establish completed training and measured capabilities; they do not establish uniform classic-model superiority.

**Organizer:** Both new models execute fewer fine source–receiver reads, retain cheap reads from every active source, and outperform several equal-read geometry controls. This produces no measured wall-clock savings against each model’s own all-fine path. Thermal’s deployed route is mixed against upstream routing on individual scenes. One-edge remove/restore experiments in both datasets verify that the exported routes affect the actual readout and restore exactly; those routes describe learned information flow, not physical causality.

**Inverse:** Thermal3903 improves average saved heating-response fidelity but chooses the wrong minimum-peak-temperature alternative in the saved 0291 candidate pool. Its realized saved-reference regret is **0.390348 native temperature units**. Precise affine increments and differentiable interfaces remain useful implementation contracts; inverse design quality is **not qualified** by these results. Wind inverse design and independent geometry-response validation remain untested. No additional training, optimization campaign, or physical solves were run for these post-training measurements; the protected solver ledger remains **326/326**.

The previous fixed-quarter development results are preserved in the [archived development report](_bk/20261008_113547Z_HONF_Unified_Training_and_Adaptive_Interaction_Refinement_Development_Report.md). Its manifests, inherited-weight experiments, receiver-packet prototype, and bounded pilot results remain development evidence. They are not renamed as formal full-data results. The [previous R-direct report](_bk/20261007_031743Z_HONF_RDirect_Formal_Readiness_and_Receiver_Local_Organization_Report.md) supplies the reporting pattern; its different classic-model selections and old measurements are not imported into this comparison.

## Decisions: Predictor / Organizer / Inverse and A / B / C

A means added organizing value beyond matched geometry controls; B means actual information flow through the exported interaction structure; C means response generalization against saved physical references. These are evidence requirements, not architecture names.

| Requirement | Thermal3903 | Wind2201 | Decision and next step |
|---|---|---|---|
| Predictor: field fidelity | Better mean fluid T than 3902; best mean surface/material T; Dense1804 better fluid T; inherited D-sep vorticity weakness persists | Both mature classics have lower physical component and vector RMSE in every role | **Mixed Thermal; Wind adaptive-replacement claim fails.** Retain role-specific comparisons and improve the difficult fields before replacing classics. |
| Organizer / A | At equal fine degree, beats nearest on 3/3 scenes and upstream on 2/3; similar aggregate fidelity to all-fine | Beats nearest/upstream/shuffle at equal per-receiver fine degree on two fixed scenes; similar fidelity to all-fine | **Measured local fidelity value; efficiency requirement unmet.** Optimize gather/scatter, base/gate, and readout costs before claiming executor savings. |
| Organizer / B | One optional source upgrade changes the affine readout by the expected source-wise increment; exact restoration | One optional message upgrade changes the nonlinear velocity readout; exact restoration | **Implementation information flow passes on inspected examples.** Source identities, base paths, and full context ancestry are retained; physical causality is unproven. |
| Response / C | Stronger average heating responses; geometry-atlas signs/amplitudes remain incomplete and panels are exposed | VALID field fidelity measured; no independent finite geometry-response labels used | **Limited Thermal heating evidence; broad C unqualified.** Reuse the saved geometry failures for development; Wind C remains untested. |
| Inverse application | Correct ranking in 2/3 saved pools; all four references rank 3/3 correctly; 0291 regret 0.390348 | No candidate pool or inverse campaign | **Thermal peak-objective check fails; Wind untested.** No certified designs or successful optimization claims. |

## Comparison identity and physical-reference limits

All Thermal headline metrics use literal **epoch-5000** checkpoints, not a mixture of selected best epochs. Each native architecture is reconstructed from its checkpoint/configuration, with its predicted-port/local-surrogate path and its normalization preserved. Run3903 uses its fresh formal thermal weights composed with frozen formal D-sep3901; Run3902 uses the same frozen flow checkpoint. Thermal3903’s formal normalization was explicitly checked against that formal lineage. No classic weights were transplanted, and no new run was trained for this report.

| Dataset / run | Native architecture | Primary checkpoint | Selection boundary |
|---|---|---|---|
| Thermal1404 | Routing-only pairwise | e5000; SHA256 `71a588ab4c2d…` | Requested replacement for 1401 |
| Thermal1501 | Sparse incidence / adaptive group control | e5000; `9e8320c39075…` | Existing mature classic |
| Thermal1804 | Dense pairwise field adaptation | e5000; `c6d369ddb3b7…` | Existing mature dense classic |
| Thermal3902 | Affine R-direct source response + D-sep3901 | e5000; `5aafc23cb86e…` | Previous formal R-direct |
| Thermal3903 | Shared source refinement, adaptive detail + D-sep3901 | e5000; `4e4ec14bbf15…` | New completed formal run; best-field selector also e5000 |
| Wind2102 | Classic K6 projection | native best-field e2440; `573cd85fddbd…` | Mature 2500-epoch history |
| Wind2103 | Dense pairwise field adaptation | native best-field e2475; `e5cfcc1487b2…` | Mature 2500-epoch history |
| Wind2201 | Shared source refinement, nonlinear adaptive detail | e5000 state; native best-field file `47aaea60a6cc…` | Best-field and literal e5000 model states exactly identical |

Wind2104 is excluded from the mature headline because its retained state is an early failed/incomplete run. Full checkpoint paths, hashes, resolved configurations, and unchanged model-state digests are in the [Thermal evidence index](../../diagnostics/generated/unified_formal5000_comparison_20261008/thermal/THERMAL_WORK_INDEX.md), [Wind comparison receipt](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_validation_q8192_summary.json), and [training-accounting audit](../../diagnostics/generated/unified_formal5000_comparison_20261008/audit/training_accounting_audit.md).

**Thermal panel:** all 89 canonical source-order cases from the original TEST90 catalog, excluding TRAIN-duplicate 0273. Membership SHA256 is `51f0bea2278c26b24a994a29a52ad6b8af1e3e85941dc75d9b20bbebce5d8728`. This catalog has been used for monitoring and development; its results are **exposed validation evidence, not independent test evidence**. Comparisons use the same full native Q8192 fluid grid, exact valid-fluid masks, 64 surface angles per active module, and 3,096 material receivers per active module. Detailed maps use the fixed input-selected cases 0291, 0653, and 0687. Mean RMSE averages each case’s physical-role RMSE equally; pooled RMSE and case distributions remain separately saved. Sampled module-peak error is not a continuous-domain maximum guarantee.

Thermal targets are the retained analytic-flow/shared-grid benchmark and saved counted response records. Values are in **packed-dataset native units**: temperature and normal-flux proxy are not relabeled Kelvin or SI heat flux. The fluid masks exclude module interiors; response metrics use their saved quadrature and common masks. These are comparisons with saved physical references, not fresh independent CFD validation.

**Wind panel:** the full source **VALID90 / 30 layouts / three directions** is fixed for all three checkpoints. VALID membership SHA256 is `ed7295dd2650687c599c13e29a6bae3492508ec8346517e39f4cacf60d0eff72`; TEST90 stayed locked and its targets were not read. One shared native-cell Q8192 panel per row is frozen, with role counts volume 1639, hub slab 1638, downstream envelope 1638, near turbine 1639, and background 1638; panel SHA256 is `188e1d2cf0eace3cacb6c6dfd721829bbe8dea0478a96d041b2ee29291f3ae74`. Primary metrics are physical component and vector RMSE in m/s against stored OpenFOAM velocity fields. Vector RMSE is the square root of mean squared vector error, then averaged equally over rows. Three directions from a layout are correlated, and the role strata overlap; the five-role score is not a uniform whole-volume integral.

Each Wind classic retains its trained **16×8×4 = 512 environment records**; Run2201 retains its native **2×2×2 = 8 records**. Coordinates, native cell IDs, targets, and partitions are shared, while those architecture-native contexts remain different. An initial pass incorrectly supplied eight records to the classics; it was rejected and retained solely as an explicitly named environment-mismatch diagnostic. Every classic headline metric and plane below comes from the corrected native-512 replay. Native original Q1024 replay of Run2201 reproduced the saved e5000 field score exactly: **0.008408665234747904**, maximum per-row difference zero.

## Completed training, exposure, and log-scale curves

Both new runs used fresh formal fine/base/router weights and full TRAIN memberships: 600 Thermal cases and 420 Wind rows / 140 layouts. Development weights were not loaded. Their shared engine uses fine warmup through epoch 500, opening through 600, soft routing through 800, and hard routing from 801. The fine learning rate is held through 2000 and then decays to epoch 5000. Curves and `last.pt` were refreshed every 100 epochs; only the declared important milestones and selectors are retained. The training histories and final checkpoints, rather than tmux startup messages, establish completion.

| Saved trajectory | Epochs | Saved-trajectory case / row visits | Optimizer updates | Microbatches | Sum of recorded train-loop time |
|---|---:|---:|---:|---:|---:|
| Thermal3903 | 5000 | 3,000,000 | 65,000 | 211,016 | 24,404.829 s = 6.779 h |
| Wind2201 | 5000 | 2,100,000 | 90,000 | 270,252 | 19,270.835 s = 5.353 h |
| Wind confirmed successful replay after saved e500 | Confirmed successful replay | 7,776 | 333 | Separately recorded | Failed partial-batch overhead not fully timed |

The saved trajectory visits every full TRAIN row per epoch. Thermal effective batch 48 was preserved when microbatch 8 became 16 after e568; Wind effective batch 24 was preserved when microbatch 4 became 8 after e101. The explicit amendments preserved checkpoint/optimizer lineage, but future query seeds depend on microbatch index and future FP32 accumulation can change. Wind’s e519 failure came from a one-pair CPU/CUDA near-mask discrepancy of approximately `5.96e-8` at a boundary; recovery restored saved e500 and used the same GPU provider for denominator preparation. The table separates the final successful trajectory from confirmed replay rather than silently counting the replay as unique progress.

![Completed Thermal3903 and Wind2201 log-scale training and selector histories with cumulative recorded train-loop time](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/01_training_convergence.png)

**Figure 1 — Completed history.** Native training and monitoring objectives use log axes; faint raw traces and a 50-epoch median are shown separately. Thermal and Wind objectives are different quantities and are not compared numerically across datasets. Their saved train-loop sums total 12.132 h across two GPUs; this is not continuous elapsed time or complete GPU allocation, and it excludes untimed failed work and other process overhead. Wind e1/e102/e501 together account for 5,714.197 s of recorded train time; the remaining epoch median is 2.334 s. Cold catalogue preparation is a plausible contributor, but it was not independently timed during training. [PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/01_training_convergence.pdf).

Thermal classics also have 5000-epoch / TRAIN600 / effective 48 histories and nominal 65,000 updates, but their objectives, normalization, and native timers differ. Legacy timer records have inconsistencies, including repeated 1404 epochs and incomplete 1804 CSV timing; no matched training-speed ranking is inferred from them. Wind classics trained for 2500 epochs with effective 16, Q8192, and 512 environment records; their selected checkpoints have 65,880 updates (2102) and 66,825 (2103), versus 2201’s 90,000 updates, Q1024 training queries, effective 24, and 8 records. This is a **matched evaluation**, not a controlled matched-training architecture ablation. The prior bounded development budget and the separately authorized formal fits retain distinct accounting.

## Predictor: Thermal long-run field comparison

| Literal e5000 model | Fluid T mean RMSE | Surface T | Material T | Sampled module-peak T | Normal-flux proxy |
|---|---:|---:|---:|---:|---:|
| 1404 | 0.293284 | 0.511137 | 0.409408 | 0.418661 | 2.120325 |
| 1501 | 0.254890 | 0.469503 | 0.360439 | 0.364533 | 1.448206 |
| Dense1804 | **0.237375** | 0.429475 | 0.329348 | **0.317573** | 1.533756 |
| R-direct3902 | 0.342425 | 0.311198 | 0.307594 | 0.338021 | **1.433262** |
| Unified3903 adaptive | 0.321108 | **0.306690** | **0.301657** | 0.318111 | 1.455692 |

Against 3902, 3903 reduces mean fluid/surface/material/peak RMSE by 6.23%/1.45%/1.93%/5.89%, while flux-proxy RMSE increases 1.57%. Fluid T improves on 52/89 cases, so the mean gain is not a universal per-case improvement. Against Dense1804, 3903 improves surface/material RMSE by 28.59%/8.41%, but worsens fluid T by 35.27%; only 6/89 cases have lower fluid-T RMSE. The pooled fluid-T RMSE also favors Dense1804, 0.266017 versus 3903’s 0.33575, so the conclusion does not depend solely on equal-case averaging.

![Thermal five-checkpoint physical-role RMSE, case distributions, flux proxy, and flow error ratios](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/02_thermal_accuracy.png)

**Figure 2 — Role-specific accuracy over canonical 89.** Endpoint Thermal3903 improves the R-direct fluid-T distribution, but the dense classic remains stronger on that role; surface and material fidelity favor 3903. The flux proxy and flow channels use separate scales. Flow belongs to shared D-sep3901 for both 3902/3903 and is bitwise identical between them. [PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/02_thermal_accuracy.pdf).

![Saved reference and five e5000 temperature predictions and signed residuals on Thermal0291](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/03_thermal_temperature_0291.png)

**Figure 3 — Native temperature field, exposed source-catalog 0291 / M5 / Q8192.** All methods share the same 7,837 valid fluid receivers, physical limits, and temperature/residual scales. Fluid RMSE is 0.294462 for Dense1804, 0.428894 for 3902, and 0.403373 for 3903; relative L2 is 5.51%/8.02%/7.54%. Gray circles show zero-based physical source slots. Residual display clips the combined extreme 1%, while annotations and tables use unclipped errors. This scene makes the dense model’s better fluid wake fidelity directly visible. [PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/03_thermal_temperature_0291.pdf).

| Model | u mean RMSE | v mean RMSE | p mean RMSE | Vorticity mean RMSE |
|---|---:|---:|---:|---:|
| 1404 | 0.0113664 | 0.00050913 | 0.00342642 | 0.0257795 |
| 1501 | 0.00654923 | 0.00044472 | 0.00257005 | 0.0266015 |
| Dense1804 | 0.00650424 | **0.00033189** | 0.00224869 | **0.0244341** |
| Frozen D-sep3901, used by 3902 and 3903 | **0.00432883** | 0.00045946 | **0.00197505** | 0.0375592 |

![Native velocity and vorticity fields and residuals on Thermal0687 for the five endpoints](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/04_thermal_flow_0687.png)

**Figure 4 — Flow tradeoff, exposed source-catalog 0687 / M10.** Shared D-sep gives 3902/3903 lower u relative-L2 error than Dense1804, 0.639% versus 1.191%, but worse vorticity, 6.009% versus 2.743%. The signed vorticity field and signed residual have separate symmetric scales; reported metrics are unclipped. The same shared flow therefore helps streamwise velocity while missing localized rotational structure. Thermal adaptive routing did not create this difference. [PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/04_thermal_flow_0687.pdf).

A fixed-three secondary replay separately checks checkpoint-selection sensitivity. Best-field epochs are 1404:e4890, 1501:e4689, 1804:e4738, 3902:e4200, 3903:e5000. Their fluid-T RMSE on 0291/0653/0687 is respectively 0.61371/0.19125/0.31343;0.38868/0.14031/0.21243;0.24845/0.12447/0.23923;0.41908/0.24835/0.33603;0.40337/0.29475/0.36571. Run3903’s selector and e5000 fields are identical;3902’s e5000 values are 0.42889/0.24705/0.31725. Selection can improve one scene and worsen another. This three-case sensitivity is secondary and does not replace the full canonical 89 exact-endpoint comparison; [selector receipts and arrays](../../diagnostics/generated/unified_formal5000_comparison_20261008/thermal/best_selector_fields_attempt4/summary.json) preserve literal ages and hashes.

## Predictor: Wind full-VALID physical accuracy

| Model | Volume vector RMSE | Hub slab | Downstream envelope | Near turbine | Background |
|---|---:|---:|---:|---:|---:|
| 2102 native K6, selected e2440 | 0.033638 | 0.052086 | 0.063776 | 0.095497 | 0.023758 |
| 2103 native dense, selected e2475 | **0.019100** | **0.032973** | **0.039102** | **0.074455** | **0.010984** |
| 2201 adaptive, e5000 | 0.058991 | 0.091483 | 0.114903 | 0.158383 | 0.040574 |
| TRAIN-fitted height-profile-only control | 0.350068 | 0.672385 | 0.725060 | 1.128278 | 0.201815 |

Values are equal-row mean vector RMSE in **m/s** on the frozen Q8192 VALID panel. Run2201 reconstructs wakes substantially better than its TRAIN-fitted empirical height-profile-only control, but loses to both native classics. Its near-turbine mean error is 65.85% above 2102 and 112.72% above 2103; its near-turbine p95/worst vector RMSE is 0.181954/0.189305 m/s. The empirical height profile is fitted on TRAIN targets and is not a prescribed physical inflow boundary condition.

![Matched Wind VALID90 physical component RMSE distributions across five roles for 2201 and native classics](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_validation_q8192_role_component_rmse.png)

**Figure 5 — Physical component distributions over all 90 VALID rows.** Boxes show equal-row distributions and diamonds show means in m/s. Both classics outperform 2201 in all 15 component–role combinations. Secondary common 2201 TRAIN-scaled field scores are 0.00856401/0.00317795/0.00147733 for 2201/2102/2103; those are a shared evaluation score, not the classics’ native training or checkpoint-selector objectives. Context shape, training queries, objectives, and checkpoint age prevent attribution of this gap solely to routing. [PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_validation_q8192_role_component_rmse.pdf).

![Wind native Ux reference fields, predictions, signed residuals and RMSE on fixed M8 and M30 VALID scenes](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_native_planes_ux_fields_residuals.png)

**Figure 6 — Streamwise wake fields on fixed VALID layouts 23/M8 and 142/M30, WD270°, native z≈70.9 m.** The full native plane, shared physical Ux scale, and shared signed residual scale expose 2201’s stronger remaining wake errors. Vector relative-L2 for 2201/2102/2103 is 0.916%/0.499%/0.295% on M8 and 1.523%/0.535%/0.437% on M30. These planes use unchanged saved reference arrays and each model’s native context; they are small detailed panels, not substitutes for the full 90-row metrics. [Three-component PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_native_planes_fields_residuals.pdf).

![Wind native transverse Uy reference fields, predictions and signed residuals on the same two fixed VALID scenes](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_native_planes_uy_fields_residuals.png)

**Figure 7 — Transverse velocity on the same native planes, m/s.** A separate symmetric physical Uy scale prevents the much larger streamwise velocity from hiding transverse errors. Each residual panel reports its actual component RMSE and the corresponding vector relative-L2. The vertical component is also inspected and retained on page 3 of the shared PDF, with its [PNG companion](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_native_planes_uz_fields_residuals.png); the full VALID component comparison is in Figure 5.

## Computation efficiency: measured read reduction and complete call price

![Complete native inference costs and same-weight field fidelity versus executed fine-read work for Thermal andWind](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/06_inference_efficiency_and_fidelity_work.png)

**Figure 8 — Measured inference price and fidelity versus fine work.** Top panels price complete native Q8192 calls on fixed scenes, separately on GPU 0 and GPU 2; bottom panels compare routing with the same frozen weights and fixed queries. Thermal fidelity uses native valid-fluid masks, not module-interior points. Wind control fidelity uses a balanced 1024-point subset of the saved 8192-point panel, with identical per-receiver fine degree for geometry controls. Fine-read reduction is an executed-row measurement; it is not a wall-clock speedup. [PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/06_inference_efficiency_and_fidelity_work.pdf).

| Complete native call | Fixed scenes / repeats | Median of scene medians | Scope |
|---|---|---:|---|
| Thermal1404 /1501 /1804 | 0291/0653/0687;1 warmup +8 balanced repeats each | 54.59 /227.41 /158.34 ms | Native full fluid, surface, material, ports, denormalization and CPU copy |
| Thermal3902 /3903 | Same scenes and repeats | 25.79 /39.68 ms | Same physical receivers, frozen D-sep flow composition and CPU copy |
| Wind2102 /2103 | VALID rows 69/426;1 warmup +3 repeats each | 62.47 /42.14 ms | Native context 512, scene staging, Q8192 readout and CPU copy |
| Wind2201 adaptive /all-fine /all-base | Same scenes and repeats | 23.83 /12.15 /24.02 ms | Native context 8; all-base retains protected near fine rows |

Thermal common outputs have full 64×128 fluid, 12×64×2 interface, and 12×3096 material capacities. Classic APIs additionally return a raw 12×64×5 port array, so their output-copy bytes differ slightly; timings preserve each actual native API. H5 retrieval, checkpoint loading, and metric calculation are excluded. Run3903 is 27.32% faster than 1404 and 74.94% faster than 1804 in these native calls, but **53.85% slower than 3902**. This is an implementation-path comparison, not a claim that sparsity alone caused the difference. Additional peak allocated memory medians are 48.15/32.43/29.87/21.33/19.60 MiB for 1404/1501/1804/3902/3903; resident model allocations and reserved memory are separate.

Thermal’s same-weight core-only route price excludes the common D-sep flow and uses 16 surface angles per module, with all 3096 material queries retained. Adaptive.50 takes 37.44 ms versus 22.24 ms for all-fine, a 68.35% increase, even though selected active fine pairs sum to 44,327/163,840 on the three scenes, a 72.94% reduction. The all-fine vectorized path also executes padded slots:98,304 rows per scene. Adaptive still executes cheap/gate rows on all 98,304 padded pairs per scene. Extra per-call allocated peaks are 10.6–13.8 MiB adaptive versus 21.2–23.2 MiB all-fine. Source-context preparation and environment ancestry remain full; sparse selected fine rows do not mean all computational work became sparse.

Wind’s complete Q8192 adaptive path executes 28,826/65,536 active fine pairs on M8 and 95,053/245,760 on M30, reducing fine reads by 56.01%/61.32%. Across all 90 VALID rows it executes 6,068,681 fine pairs out of 14,180,352 active pairs, with cheap/gate reads on every active pair and 884,901 protected near reads. Its fixed-scene call is nevertheless about 1.96× its own all-fine call. At M8, representative median decomposition is 0.713 ms target-free staging, 2.687 ms context preparation, 20.320 ms physical readout, and 0.060 ms output copy, with small remaining orchestration overhead. Incremental allocated peak is about 9.11 MiB, versus about 7.91 MiB all-fine; **no memory-saving claim is made for this Wind executor**. Classic full-access paths are reported separately and have no invented selected-fine counter.

Wind cold native catalogue construction is separate from warm readout: the saved evaluation preparation receipt records 90 misses and 90 hits, approximately 12.756 GB CPU cache residency, no evictions, and 0.624 s for constructing the frozen query panel after catalogues were available. One-pass full VALID read timings and these preparation costs are retained separately from repeated warm fixed-scene prices. The formal training cost also includes fine replay and teacher/base/router objectives; deployment fine-row savings do not prove sparse training speed or end-to-end CFD savings.

## Organizer: what is actually learned and read

The shared implementation is actual source refinement in [SourceRefinement](../../src/honf_forward_core/interface_fields/interaction_refinement.py), [SharedInterfaceContext](../../src/honf_forward_core/interface_fields/common.py), and the [unified TrainingEngine](../../src/honf_runtime/unified_training.py), with separate dataset providers and separate learned weights. At a source–receiver pair, detail is `B(q,i) + g(q,i)·[F(q,i)−B(q,i)]`; all active physical sources retain their cheap base path and protected near pairs retain fine detail. Both formal endpoints use hard routing at threshold 0.50 and temperature 0.10. Production routing imposes no forced K, source deletion, or latent-bank revival. Equal-degree nearest/upstream/shuffle routes are diagnostic interventions only.

Thermal prepares a heat-independent kernel and applies the heating controls affinely, preserving source order and precise FP64 native accumulation/increments. Run3903’s formal thermal context uses prescribed geometry/material/inlet/domain inputs; optional predicted-flow context is **disabled**, and target flow is never an input. Frozen D-sep3901 supplies the reported flow outputs separately. Wind retains individual 64-dimensional source messages and physical source measures, then applies a nonlinear velocity readout. A Wind latent-message norm is not an additive physical velocity contribution. These contracts share executable interaction code while preserving the distinct affine and nonlinear output laws.

![Thermal actual optional/protected fine support, source receiver graph and source-wise readout intervention](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/05_thermal_actual_organizer.png)

**Figure 9 — Actual 3903 organizer, 0291.** Five physical sources and 192 environment records are retained. Optional/protected fine-degree maps and three native receiver path exports show actual selected rows; gray base paths remain even when no fine upgrade is selected. At receiver 1617/source 0, the normal-minus-removed endpoint difference for exactly one unprotected upgrade is −0.10558 nativeT (removal raises temperature by +0.10558); the source-wise `(F−B)×heat` closure error is≤4.47e−8 and restoration error is zero. Context and weights are unchanged. This verifies B for the measured readout and avoids interpreting support as physical causality. [PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/05_thermal_actual_organizer.pdf).

| Same 3903 thermal weights, physical masked fluid-T RMSE | 0291 | 0653 | 0687 |
|---|---:|---:|---:|
| All-fine | 0.381930 | 0.276173 | 0.402158 |
| Adaptive.50 | 0.403373 | 0.294750 | 0.365710 |
| Nearest, equal fine degree | 0.422739 | 0.335498 | 0.429827 |
| Upstream, equal fine degree | **0.393730** | 0.316543 | 0.404021 |
| Shuffle, equal fine degree | 0.434017 | 0.372739 | 0.668992 |

Thermal selectors share the exact .50 fine counts 11,753/10,909/21,665, preserve protected near reads, and keep weights and physical receiver masks fixed. Adaptive beats nearest on all three, but upstream is better on 0291. Adaptive versus all-fine is worse on 0291/0653 and better on 0687; their mean RMSE is 0.354611 versus 0.353420. Threshold.25 improves that fixed-three mean to 0.339980 with more fine reads; this is a post-training exposed-panel intervention, not a new validated operating-point selection. The original raw all-grid route metrics include module interiors and are retained under their correct estimand; Figure 8 and this table use the separately verified native fluid masks.

![Classic Thermal architecture-native attention breadth and group-read maps on the identical 0291 grid](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/14_classic_thermal_native_organizers.png)

**Figure 10 — Classic organizer behaviors.** Instrumentation leaves each native endpoint field bitwise unchanged. Run1404 exports attention and edge decomposition over six native edges;1501 exports 12 groups with 31 module-group and 754 environment-group incidences plus per-query group degree;1804 exports attention over all 192 environment records. Attention breadth is `exp(entropy)` and does not imply those records were skipped. The three scales have architecture-specific meanings and are not interchangeable with 3903’s executed fine-source count. [Native read receipts](../../diagnostics/generated/unified_formal5000_comparison_20261008/thermal/classic_native_reads_0291_attempt3/summary.json); [PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/14_classic_thermal_native_organizers.pdf).

![Wind2201 actual optional and protected source-read support on fixed M8 and M30 native planes](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_native_planes_adaptive_route_support.png)

**Figure 11 — Actual 2201 hard-route support.** Threshold.50 and temperature.10 produce spatially varying source detail on the two input-selected VALID scenes, while near pairs remain fine. Optional fine fractions are divided by active source capacity, not padded capacity; the full eight-record environment context and all cheap source messages remain. The spatial structure is learned support, not a physical causal graph. [PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_native_planes_adaptive_route_support.pdf).

![Wind source-to-receiver graph under learned and equal-fine nearest upstream and shuffled routes](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_run2201_m30_equal_fine_route_intervention.png)

**Figure 12 — Wind geometry-control intervention, fixed row 426/layout 142/M30.** The same receiver gets 12 optional fine reads and zero protected reads in all four routes. Its selection used route disagreement, not CFD errors. Over the balanced 1024-point subset from the saved 8192 panel, common field scores are 0.008522 adaptive, 0.008334 all-fine, 0.053444 all-base, 0.013314 nearest, 0.011562 upstream, and 0.021707 shuffle. Adaptive improves on each equal-read geometry control, but this two-scene result is not a full 90-row ablation. Removing source 25’s upgrade at receiver 492 changes output by `[0.120975, 0.000745, 0.004183]` m/s; the vector change is 0.121049 m/s, fine degree goes 12→11→12, and restoration error is zero. Base/other messages, context hashes, and model state remain unchanged. The [one-edge replay receipt](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_row426_query492_single_edge_replay.json) establishes nonlinear B without assigning latent messages physical causal meaning. [PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_run2201_m30_equal_fine_route_intervention.pdf).

## C: saved heating and geometry response generalization

The heating comparison reuses counted physical baselines and transfer-minus/transfer-plus endpoints for 0291, 0294, 0687: three layouts andsix signed states, all previously exposed. No new solver request was issued. The 0277 family is excluded from primary response aggregation because its required baseline is absent. Every model uses frozen e5000 weights; shared physical coordinates/masks and saved quadrature define response RMSE. Six sign/layout endpoints are correlated and represent one excitation direction per layout, not all heating columns.

| Equal signed-state mean response RMSE | Fluid T | Surface T | Material T | Flux proxy |
|---|---:|---:|---:|---:|
| 1404 | 0.020505 | 0.041804 | 0.035737 | 0.158591 |
| 1501 | 0.010929 | 0.019611 | 0.016589 | 0.090735 |
| Dense1804 | 0.010562 | 0.018059 | 0.018793 | 0.122495 |
| R-direct3902 | 0.008845 | 0.015382 | 0.018768 | 0.055477 |
| Unified3903 | **0.007644** | **0.012036** | **0.014022** | **0.049362** |

![Saved counted heating-response accuracy by physical role and signed amplitude for the five e5000 models](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/10_heating_response_accuracy.png)

**Figure 13 — Heating response gains and amplitude miss.** Run3903 has the lowest mean RMSE across the four tested thermal roles, reducing 3902 errors by 13.58%/21.76%/25.29%/11.02%. On 0291 transfer-plus, reference weighted mean fluidΔT is+0.028914;3903 predicts+0.034710, about 20.05% too large, compared with 3902’s+0.040240. Better aggregate response fidelity does not imply every response amplitude or inverse objective is correct. Units remain nativeT/flux proxy. [PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/10_heating_response_accuracy.pdf).

![Signed physical heating-response reference fields and all five frozen model responses and residuals on 0291](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/11_heating_response_0291.png)

**Figure 14 — Physical heating-change field, 0291 transfer-plus.** A saved 0.057515 heating-unit transfer from physical source 1 to source 4 produces the signed reference field; all models are compared on the same 7,837 valid receivers. Response RMSE/cosine for 3903 is 0.016129/0.9879, versus 3902’s 0.019134/0.9830. Displayed residual clipping excludes the extreme 1% only visually. These are finite changes against retained counted reference endpoints, not neural self-consistency tests. [PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/11_heating_response_0291.pdf).

![Thermal geometry-response RMSE cosine signed mean changes and native reference prediction residual fields](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/13_geometry_response_generalization.png)

**Figure 15 — Geometry remains a C limitation.** Four stored TRAIN anchors 0001/0318/0333/0348 supply sixteen existing i±/j± perturbations;0001 includes the preserved duplicate-family identity 0001+0273. No new reference solves or inverse steps occur. Run3903 mean signed-response cosines are 0.702 fluidT, 0.659 surfaceT, 0.617 materialT, and 0.587 flux proxy; corresponding response RMSE is 0.134414/0.200523/0.195514/0.862276. Its mean signed fluid-T change is+0.002783 versus the reference+0.028420, and the illustrated 0001/i+ wake-change residual exposes amplitude/structure misses. Fluid-T cosine is higher for 1501/1804, 0.730/0.738. The saved grid-error/physical sign floor is unresolved, so these exposed atlas responses are not an independent physical qualification or a certified sign test. [PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/13_geometry_response_generalization.pdf); [saved response arrays and receipts](../../diagnostics/generated/unified_formal5000_comparison_20261008/thermal/geometry_responses_attempt1/summary.json).

Run3903’s prepared+1% heating increment on fixed 0291 closes against precise endpoints to 4.50e−15, with exact heat-VJP/kernel agreement and unchanged heat-independent preparation. This is a measured affine arithmetic contract. The separately retained FP32 endpoint-difference failures are not erased by that success. Local fixed-route geometry AD/FD agrees to about 2.1% on the selected model-only scalar with unchanged receiver IDs/masks; it is not evidence that the physical geometry response is accurate. Wind’s nonlinear derivative interface likewise supplies model derivatives, but no independent finite geometry-response references were exercised here. **C therefore remains limited to the explicitly replayed controls and references.**

## Inverse: inspect saved candidate rankings before claiming design utility

The finite-pool check ranks only the already saved minus/baseline/plus candidates by their maximum sampled material temperature. It uses the independently saved counted reference fields to price each chosen candidate. There is no new proposal generation, gradient optimization, CFD replay, or inverse campaign. It is a useful inverse-objective check because it can fail even when field RMSE and precise arithmetic improve.

![Saved physical inverse-candidate peak-temperature curves showing the 0291 ranking failure and correctly ranked 0294 and 0687 pools](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/12_saved_pool_inverse.png)

**Figure 16 — Saved-pool inverse failure.** All four references select the physical minimum in 3/3 pools;3903 selects correctly in 2/3. At 0291, reference baseline/plus peaks are 18.520241/18.129892, but 3903 predicts 19.058064/19.127707 and selects baseline. Realized reference regret is 0.390348 nativeT. Stars mark reference and 3903 choices, not optimization iterates. Correct choices on 0294/0687 do not cancel this failure, and sampled material maxima do not certify continuous peaks. [PDF master](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/12_saved_pool_inverse.pdf); [three-state objective records](../../diagnostics/generated/unified_formal5000_comparison_20261008/thermal/counted_responses/finite_pool_material_objective.json).

The long-run Thermal result supports improved average heating-response prediction, source-resolved organization, and precise controllable increments. It does **not** support promoting 3903 to a generally reliable minimum-peak inverse designer. Wind’s improved reconstruction over a profile baseline and its route interventions similarly do not establish turbine-layout optimization or valid generated designs. Further development should target Thermal peak/objective ordering and geometry-response amplitude, Wind physical field accuracy, and actual adaptive executor overhead; the evidence here authorizes no additional training or solves.

## Figure index, saved evidence, and preservation

| Inspectable evidence | Main figures | Retained PDF masters |
|---|---|---|
| Completed histories and costs | 1, 8 | [History](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/01_training_convergence.pdf), [cost / fidelity-work](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/06_inference_efficiency_and_fidelity_work.pdf) |
| Thermal physical fields | 2–4 | [Role accuracy](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/02_thermal_accuracy.pdf), [temperature](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/03_thermal_temperature_0291.pdf), [velocity/vorticity](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/04_thermal_flow_0687.pdf) |
| Wind physical fields | 5–7 | [Role distributions](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_validation_q8192_role_component_rmse.pdf), [Ux/Uy/Uz planes](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_native_planes_fields_residuals.pdf) |
| Actual organizer behavior | 9–12 | [Thermal paths](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/05_thermal_actual_organizer.pdf), [classic maps](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/14_classic_thermal_native_organizers.pdf), [Wind support](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_native_planes_adaptive_route_support.pdf), [equal-degree paths](../../diagnostics/generated/unified_formal5000_comparison_20261008/wind/wind_run2201_m30_equal_fine_route_intervention.pdf) |
| Physical responses and inverse objective | 13–16 | [Heating accuracy](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/10_heating_response_accuracy.pdf), [heating fields](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/11_heating_response_0291.pdf), [geometry](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/13_geometry_response_generalization.pdf), [saved inverse pools](../../diagnostics/generated/unified_formal5000_comparison_20261008/figures/12_saved_pool_inverse.pdf) |

All selected figures were rendered from saved numerical arrays and visually inspected for physical scales, signed residuals, annotations, and layout. PDF masters are retained; the small PNG companions exist for direct Markdown viewing. Figure provenance/source hashes, per-case metrics, timing repeats, corrected diagnostic boundaries, and intervention arrays live under [the ignored comparison evidence root](../../diagnostics/generated/unified_formal5000_comparison_20261008/). Generated data, figures, one-time evaluation/rendering code, and local HTML previews are excluded from Git. Only durable report text and the relocated development archive enter the upload; existing scientific checkpoints, authoritative counted records, completed development children/pilots, security/history boundaries, and the locked WindTEST target boundary remain protected.

Final read-only closeout rechecked all **100 protected historical bindings** (18 checkpoint bindings, four authoritative counted-reference records, and 78 prior child/pilot states): hashes, sizes, and modification times are unchanged. The authoritative outcome still records standing usage 326 against cap 326. Both evaluation GPUs are idle. [Protected binding receipt](../../diagnostics/generated/unified_formal5000_comparison_20261008/audit/protected_final.json), [final source/figure-link QA](../../diagnostics/generated/unified_formal5000_comparison_20261008/audit/final_report_source_qa.json), and the linked training-accounting audit distinguish preservation checks, completed measurements, and unmatched historical timers.
