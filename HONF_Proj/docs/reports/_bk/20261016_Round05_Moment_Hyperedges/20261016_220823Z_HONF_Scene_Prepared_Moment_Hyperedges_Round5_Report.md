# Round5: scene-prepared moment hyperedges

**Completed measurements, 2026-10-16. Six matched segmented continuations reached literal e2500. No candidate qualifies for a fresh full-TRAIN probe; none was launched.**

Round5 adds typed spatial/value moments to the inherited attention anchors, prepares the richer hyperedge state once per scene, and reuses it across receiver queries. C0 is the projected-mean control, M adds an affine processor and moment transport, and H replaces that processor activation with SiLU using matched inputs, parameters and initial tensors. Thermal keeps T(X,h)=K(X)h; Wind keeps its native background profile plus learned residual. All six runs continued their own C-soft e500 parents, with unchanged selected TRAIN membership, transforms, deterministic sampling, objectives and schedule.

The useful predictor result is Thermal fluid temperature: H achieves mean native DEV RMSE 0.789884, 14.44% below C0 and 10.16% below M. The miss is broader fidelity: H v/p remain 32.3%/32.6% above D0, and Wind near-vector RMSE is 0.629116 m/s versus D0 0.234814 m/s. The new path is used, but H does not recover the Wind wake deficit. H uses 76.6% less allocated memory on Thermal and 85.1% less on Wind at B8/Q8192 in the matched native forward/extraction experiment; complete high-module calls are about 36% slower than D0. This is a memory advantage with a latency cost, not a broad speedup.

| Area | Measured gain | Miss / limit | Decision and next useful work |
|---|---|---|---|
| Predictor | Thermal H improves fluid T over C0 and M; all six histories reach finite, contiguous e2500 | Thermal flow guards miss; Wind H and M stay close to C0 and far above D0; one initialization and exposed DEV only | No fresh full-TRAIN probe. Preserve this negative comparison; investigate fitting/local-detail limits before another architecture campaign |
| Organizer | 24 real typed association exchanges, exact restorations, active new-path effects, saved P/Λ/moments/Z and signed K | Learned support is not physical causality; H descriptor curvature is below the matched M FP32 cancellation control on the fixed tiny increments | Retain actual hypergraph evidence; do not advertise resolved physical higher-order coupling |
| Execution | Large matched B8 native forward/extraction memory reductions; a limited same-function prepared-query gain | Complete candidate calls are slower; actual optimizer peaks are separate; no sparse executor savings measured | Keep plain Torch full-soft execution and distinguish preparation, reuse, native extraction and optimizer scopes |
| Inverse | Heat null, flow independence, checked affine increments/adjoints and recomputed endpoint maxima | No inverse designs, new PDE/CFD solves, or compatible four-state geometry truth | No inverse-readiness claim; TEST remains locked, own new solves 0, inherited usage 326/326 |

The final assessment uses the plan's modest first-version bands, not mature-classic replacement or inverse certification. Eligibility was independent by dataset. Both optional jets were deferred on the fixed TRAIN diagnosis at e1000; no mature-DEV-driven revision, sparse training campaign or automatic follow-up was added.

## Figure and numerical evidence index

The selected figures emphasize the hypergraph construction, its actual use, and the memory/latency tradeoff. PDF masters retain the inspected supporting panels; only the most useful raster companions are embedded below. All figures were made from saved numerical evidence with no new model evaluation during rendering. Generated figures and one-time analysis helpers remain local and ignored by Git.

| Family | Inspected PDF masters / scope |
| --- | --- |
| 1. Learning | [Learning and exposure](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_learning_v2/round5_learning_curves.pdf) — five pages, fixed native panels, literal age and optimizer exposure |
| 2. Native fields | [Thermal](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_native_fields_v2/round5_thermal_native_fields.pdf) / [Wind](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_native_fields_v2/round5_wind_native_fields.pdf) — ten / twelve pages, fixed scenes, native masks and shared field/error scales |
| 3. Hypergraphs | [Thermal graph/controls](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_hypergraphs_e2500_v6/thermal_round5_hypergraphs_e2500.pdf) / [Wind graph/controls](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_hypergraphs_e2500_v6/wind_round5_hypergraphs_e2500.pdf) — six pages total, actual support, moments, processed states and interventions |
| 4. Costs | [Native latency, memory and work](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_family4_costs_display_v1/round5_family4_costs_v1.pdf) — six pages, same native inputs and explicit measurement scopes |

The [literal-e2500 native comparison](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/evaluation/round5_literal2500_native_comparison_v1.json) binds all checkpoint, recipe, panel and source hashes. The [cost summary](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/evaluation/round5_terminal_native_cost_summary_v2.json), [mechanical qualification](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/evaluation/round5_final_research_qualification_v1.json) and [root decision](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/round5_terminal_root_decision_v1.json) are the numerical authorities. The protected [Round2](../20261014_20261015_Round02_Attention_Geometry/20261014_193935Z_HONF_Attention_Round2_Report.md), [Round3](../20261015_20261016_Round03_A2D_Sparse_Readout/20261016_004810Z_HONF_A2D_Sparse_Readout_Round3_Report.md) and [Round4](../20261016_Round04_Inherited_Gathering/20261016_105923Z_HONF_Attention_Inherited_Gathering_Round4_Report.md) reports retain their original evidence and interpretation; they are not rewritten by this report.

## Predictor: native accuracy at literal e2500

Each cell below is **mean / p90 native RMSE over cases or rows**, not a normalized training score. Thermal uses 22 DEV cases and its original packed-native units; q is an extraction-consistent proxy, not verified SI heat flux. Wind uses 24 direction rows from eight layouts, m/s, with vector RMSE = sqrt(mean(sum of squared component errors)); there is no divide by three. Percentage changes use the mean only and are descriptive point estimates.

### Thermal complete native board

| Metric | D0 | C0 | M | H | H/C0 mean | H/M mean |
| --- | --- | --- | --- | --- | --- | --- |
| temperature_fluid | 0.877178 / 1.33615 | 0.923182 / 1.33774 | 0.879167 / 1.42131 | 0.789884 / 1.23782 | -14.44% | -10.16% |
| v | 0.00307437 / 0.00386934 | 0.00446099 / 0.00525928 | 0.00411811 / 0.00484634 | 0.00406619 / 0.00490805 | -8.85% | -1.26% |
| p | 0.0112959 / 0.0142759 | 0.0160367 / 0.021433 | 0.0154728 / 0.0192203 | 0.0149746 / 0.020711 | -6.62% | -3.22% |
| u | 0.020205 / 0.0251479 | 0.0253572 / 0.0311393 | 0.030617 / 0.0380528 | 0.0271127 / 0.0326225 | +6.92% | -11.45% |
| omega | 0.0517713 / 0.0619499 | 0.0681583 / 0.0775152 | 0.0652351 / 0.0773033 | 0.060704 / 0.0728879 | -10.94% | -6.95% |
| temperature_surface | 0.90268 / 1.38662 | 1.06434 / 1.88141 | 1.11201 / 2.01277 | 0.999882 / 1.69752 | -6.06% | -10.08% |
| temperature_material | 0.919647 / 1.43578 | 1.05655 / 1.74528 | 1.11323 / 2.04753 | 1.03928 / 1.69893 | -1.63% | -6.64% |
| q_proxy | 3.29703 / 5.39097 | 2.61112 / 3.17387 | 2.58403 / 3.83844 | 2.50447 / 3.82821 | -4.08% | -3.08% |
| module_peak_endpoint | 0.952481 / 1.99776 | 1.14241 / 1.73308 | 1.22047 / 2.15736 | 1.0616 / 1.97872 | -7.07% | -13.02% |

### Wind complete native board

| Metric | D0 | C0 | M | H | H/C0 mean | H/M mean |
| --- | --- | --- | --- | --- | --- | --- |
| volume/vector | 0.0732274 / 0.0864138 | 0.238666 / 0.26336 | 0.237467 / 0.27808 | 0.235967 / 0.266923 | -1.13% | -0.63% |
| volume/Ux | 0.0719491 / 0.0846136 | 0.236814 / 0.26146 | 0.235588 / 0.276554 | 0.234083 / 0.264927 | -1.15% | -0.64% |
| volume/Uy | 0.0109442 / 0.0143438 | 0.0250938 / 0.0282447 | 0.0254225 / 0.0283049 | 0.0251571 / 0.0273913 | +0.25% | -1.04% |
| volume/Uz | 0.0076194 / 0.0110985 | 0.0153888 / 0.0182102 | 0.0148592 / 0.0174137 | 0.0153071 / 0.018066 | -0.53% | +3.01% |
| hub_slab/vector | 0.135345 / 0.165598 | 0.418751 / 0.47115 | 0.412668 / 0.50698 | 0.416505 / 0.471354 | -0.54% | +0.93% |
| hub_slab/Ux | 0.133143 / 0.162713 | 0.415846 / 0.468324 | 0.409645 / 0.504059 | 0.413524 / 0.468536 | -0.56% | +0.95% |
| hub_slab/Uy | 0.020658 / 0.0281337 | 0.0443134 / 0.0498359 | 0.0447931 / 0.0499221 | 0.0448208 / 0.0522525 | +1.15% | +0.06% |
| hub_slab/Uz | 0.0121132 / 0.0171357 | 0.0205952 / 0.024133 | 0.0203422 / 0.024649 | 0.0205373 / 0.0234751 | -0.28% | +0.96% |
| downstream_envelope/vector | 0.171749 / 0.216664 | 0.505084 / 0.5456 | 0.503453 / 0.568462 | 0.502916 / 0.557335 | -0.43% | -0.11% |
| downstream_envelope/Ux | 0.168902 / 0.213025 | 0.500991 / 0.541317 | 0.499323 / 0.564407 | 0.498726 / 0.552987 | -0.45% | -0.12% |
| downstream_envelope/Uy | 0.0255761 / 0.0328747 | 0.0559327 / 0.061532 | 0.0557973 / 0.0624015 | 0.0563894 / 0.0629209 | +0.82% | +1.06% |
| downstream_envelope/Uz | 0.0169781 / 0.0231288 | 0.0306659 / 0.0345126 | 0.03061 / 0.0359245 | 0.0310224 / 0.0358977 | +1.16% | +1.35% |
| near_turbine/vector | 0.234814 / 0.287734 | 0.627808 / 0.666405 | 0.624931 / 0.679727 | 0.629116 / 0.685807 | +0.21% | +0.67% |
| near_turbine/Ux | 0.227819 / 0.279711 | 0.621089 / 0.659547 | 0.618079 / 0.672703 | 0.622367 / 0.678093 | +0.21% | +0.69% |
| near_turbine/Uy | 0.0477233 / 0.0572698 | 0.0804237 / 0.0903241 | 0.0808613 / 0.0921647 | 0.0811185 / 0.0904114 | +0.86% | +0.32% |
| near_turbine/Uz | 0.0302226 / 0.0364603 | 0.0427956 / 0.0488569 | 0.0426035 / 0.0489885 | 0.042298 / 0.0495246 | -1.16% | -0.72% |
| background/vector | 0.0389066 / 0.0468166 | 0.147227 / 0.165358 | 0.146073 / 0.166865 | 0.144862 / 0.162785 | -1.61% | -0.83% |
| background/Ux | 0.0381712 / 0.0455774 | 0.146135 / 0.164372 | 0.144989 / 0.165852 | 0.143778 / 0.161862 | -1.61% | -0.84% |
| background/Uy | 0.0056147 / 0.0076205 | 0.0141135 / 0.0157357 | 0.0143512 / 0.0162259 | 0.0139937 / 0.0158083 | -0.85% | -2.49% |
| background/Uz | 0.00477358 / 0.00659897 | 0.0106409 / 0.0122511 | 0.010167 / 0.0109803 | 0.0104906 / 0.0122162 | -1.41% | +3.18% |

Thermal H improves 8 of the 18 mean/p90 summaries by at least 10% relative to M, with none worsening by 10%. It is nevertheless not uniformly best: H pressure p90 is 7.76% above M, vertical-velocity p90 is 1.27% above M, and H u mean is 6.92% above C0. Relative to D0, H fluid T is 9.95% lower but v/p means are 32.26%/32.57% higher, u is 34.19% higher and curl is 17.25% higher. The q mean improves over D0, while C0 retains a better q tail than H. A temperature gain does not repair those distinct field misses.

Wind H differs from M by less than 5% in 37 of 40 mean/p90 summaries; no summary improves by 10%, while background Uz p90 worsens by about 11.3%. H vector means remain 2.68–3.72 times D0 across the five roles. Near-vector error is 0.67% above M and downstream-vector error is only 0.11% below M. The data do not support an advantage from this nonlinear processor for the Wind wake deficit.

### Fixed TRAIN4, heterogeneity and secondary checkpoint selection

| Dataset | Fixed TRAIN4 metric | C0 mean / p90 | M mean / p90 | H mean / p90 |
| --- | --- | --- | --- | --- |
| thermal | temperature_fluid | 0.643809 / 0.937626 | 0.506977 / 0.732509 | 0.455866 / 0.586261 |
| thermal | v | 0.00411564 / 0.00572095 | 0.00394433 / 0.00528087 | 0.00356702 / 0.00484282 |
| thermal | p | 0.0151083 / 0.0162704 | 0.0118874 / 0.0141831 | 0.0114806 / 0.0139958 |
| thermal | u | 0.0185241 / 0.0204604 | 0.023958 / 0.0249116 | 0.0216707 / 0.0232265 |
| thermal | omega | 0.0575794 / 0.0747712 | 0.0539411 / 0.0680083 | 0.0484316 / 0.0635459 |
| thermal | temperature_surface | 0.640137 / 0.958363 | 0.558484 / 0.800451 | 0.578415 / 0.798179 |
| thermal | temperature_material | 0.684673 / 0.97304 | 0.537867 / 0.758344 | 0.664259 / 0.933853 |
| thermal | q_proxy | 1.33319 / 1.84695 | 1.24449 / 1.69331 | 1.12775 / 1.41165 |
| thermal | module_peak_endpoint | 0.67942 / 0.98857 | 0.591238 / 0.792176 | 0.617784 / 0.854196 |
| wind | volume/vector | 0.152865 / 0.1725 | 0.151543 / 0.171978 | 0.156757 / 0.177646 |
| wind | hub_slab/vector | 0.262099 / 0.314571 | 0.257746 / 0.314441 | 0.258379 / 0.313484 |
| wind | downstream_envelope/vector | 0.352579 / 0.391682 | 0.346354 / 0.384553 | 0.352804 / 0.392258 |
| wind | near_turbine/vector | 0.444615 / 0.486862 | 0.435364 / 0.485017 | 0.443487 / 0.491492 |
| wind | background/vector | 0.118717 / 0.13518 | 0.113324 / 0.129594 | 0.118262 / 0.133062 |

TRAIN4 uses Thermal 0006/0097/0342/0631 and Wind 456/273/536/533, fixed before endpoint inspection and disjoint from DEV. Thermal TRAIN4 fluid T improves to 0.455866 for H, but H material T mean is 23.5% above M. Wind TRAIN4 near-vector RMSE remains 0.435–0.445 m/s across arms. Across retained role means, TRAIN4/DEV ratios are roughly 0.45–0.96 for Thermal and 0.42–0.82 for Wind; these unequal, four-case panels do not estimate a paired generalization gap. Wind's training fit is already weak on this fixture, so unseen geometry alone is an insufficient explanation.

The Thermal H/M temperature change varies by module count: means worsen by about 1.5% and 6.8% in the 3- and 5-module strata, and improve by about 21.9% and 24.4% in the 7- and 10-module strata; the 5-module p90 worsens by about 21.1%. Wind layout-level near-vector H/M changes range from about −5.8% to +8.0%. Direction rows sharing a layout are correlated. Strata have small counts and no resampling-based significance claim. D0 layout-level source summaries are unavailable, so no per-layout D0 result is inferred.

The [stratified and fitting-panel analysis](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/evaluation/round5_literal2500_strata_gap_analysis_v1.json) retains counts and every available role/component stratum. Secondary checkpoint selection uses only the common saved monitor cadence: Thermal C0 e2200, M/H e2300; Wind C0/H e1100, M e900. The selected Thermal fluid-T means are 0.856577/0.834914/0.759240 for C0/M/H; selected Wind near-vector means are 0.633859/0.639189/0.642019 m/s. These unequal-age boards do not reverse the conclusion. The [selected-age native comparison](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/evaluation/round5_selected_native_comparison_v1.json) binds the monitor-only choices and original best-field aliases; native figures, responses and graphs never select a checkpoint.

### Mature classic context

| Dataset | Metric | 1404 / 2102 mean / p90 | 1804 / 2103 mean / p90 | H e2500 mean / p90 |
| --- | --- | --- | --- | --- |
| thermal | temperature_fluid | 0.29587 / 0.461658 | 0.197941 / 0.247678 | 0.789884 / 1.23782 |
| thermal | v | 0.00048681 / 0.000757139 | 0.000319036 / 0.000410787 | 0.00406619 / 0.00490805 |
| thermal | p | 0.00313127 / 0.00416955 | 0.00209999 / 0.00271009 | 0.0149746 / 0.020711 |
| thermal | u | 0.0106748 / 0.0133867 | 0.00571489 / 0.00702659 | 0.0271127 / 0.0326225 |
| thermal | omega | 0.0242951 / 0.0437125 | 0.023461 / 0.0457071 | 0.060704 / 0.0728879 |
| thermal | temperature_surface | 0.450278 / 0.552434 | 0.403162 / 0.493923 | 0.999882 / 1.69752 |
| thermal | temperature_material | 0.359886 / 0.510464 | 0.300429 / 0.343307 | 1.03928 / 1.69893 |
| thermal | q_proxy | 2.08649 / 2.61616 | 1.55607 / 1.93443 | 2.50447 / 3.82821 |
| thermal | module_peak_endpoint | 0.365832 / 0.711871 | 0.285289 / 0.409073 | 1.0616 / 1.97872 |
| wind | volume/vector | 0.0362412 / 0.04599 | 0.0176742 / 0.0229746 | 0.235967 / 0.266923 |
| wind | hub_slab/vector | 0.0546322 / 0.0740489 | 0.0340152 / 0.0447465 | 0.416505 / 0.471354 |
| wind | downstream_envelope/vector | 0.0670205 / 0.0925426 | 0.0396296 / 0.0501657 | 0.502916 / 0.557335 |
| wind | near_turbine/vector | 0.0982757 / 0.120962 | 0.0747785 / 0.0889773 | 0.629116 / 0.685807 |
| wind | background/vector | 0.0268242 / 0.0307922 | 0.0112619 / 0.013303 | 0.144862 / 0.162785 |

Classic Thermal references use selected e4890 (1404) and e4738 (1804); Wind references use e2440 (2102) and e2475 (2103). Their native loaders and checkpoint-native normalization are preserved. They used full TRAIN with different normalization, objectives, lineage and exposure, so this is practical same-panel context, not an architecture-isolated comparison. Thermal H fluid-T error remains about four times 1804; Wind H near-vector error remains about 8.4 times 2103. Mature-classic replacement is not the qualification gate. Historical classic cost/organization evidence remains in Round2: 1404/2102 use six-slot routers, 1804/2103 use dense pairwise environment paths, which are not the new typed EM/EE atlas. No new matched classic latency campaign or physical-causality equivalence is asserted.

### Selected native fields and residuals

![Thermal DEV 0291 full native temperature and signed residuals at e2500](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_native_fields_v2/thermal_scene0291_T.png)

**Thermal 0291, literal e2500, DEV.** The 7,837 valid fluid samples retain native coordinates/masks and shared field/error scales. Fluid-T RMSE is D0 0.826923, C0 0.605844, M 0.621945 and H 0.500896 in packed-native units. H improves this fixed scene; the full 22-case table supplies the population-panel summary. No interpolation or Kelvin calibration is introduced.

![Wind DEV row 69 native crosswind cuts and Ux residuals at e2500](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_native_fields_v2/wind_row069_cut_Ux.png)

**Wind row 69, literal e2500, DEV.** Across 522 saved native cut samples, Ux RMSE is D0 0.134238, C0 0.450098, M 0.438538 and H 0.446832 m/s. Actual downstream cut offsets are 1.945/4.008/7.996 rotor diameters; stored z index 16 is 0.885874D, about 70.87 m, the nearest stored plane to 80 m. The figure shows the unresolved wake structure, not a new CFD reference. The PDF retains both fixed rows, all components and signed error planes.

## Organizer: what the hypergraph represents and what it actually does

The hypergraph is a two-stage model information path. Final contextual module/environment tokens provide typed donor-to-anchor probabilities P. Each anchor stores an ordinary full-width mean and compact coordinate/value moments; M or H processes that scene-level descriptor into a residual state Z. Geometry-only receiver weights Λ gather four Thermal or eight Wind anchors, then the retained local path and native adapter produce fields. The same prepared state serves all receiver queries. Thermal heat enters through its source-linear response factors after this organization is prepared.

| Stage | Depends on | Computed once per scene or per receiver | Evidence |
|---|---|---|---|
| Typed P and donor lifts | Geometry, operating context and final contextual tokens; no heat or targets | Scene | All four P heads retained numerically; head 0 displayed |
| Full means and compact moments η | Typed P, donor coordinates, donor-only values and native measures | Scene | Actual centroid/covariance/latent variance/cross-moment arrays, valid masks and row mass |
| M/H processor and Z | η with matched affine/SiLU weights | Scene | New-path zeroing and fixed-descriptor curvature controls |
| Λ cover and local path | Receiver geometry and native topology | Query | Four/eight valid anchors; exact query IDs and physical positions |
| Thermal K and heat application | Same prepared raw factors/source IDs, then h | Reused query state / heat application | Signed coefficients, exact heat null, checked increments/adjoints; no geometry-response truth implied |

All valid association probabilities remain positive; the full-soft atlas is the default. Displayed high-weight donor circles annotate paths without pruning execution. The three fixed probes per scene cover source-centroid, upstream-near and downstream-far positions. Moment norms are latent/nondimensional quantities; only signed K has the adapter's packed-native temperature per raw heat-power convention. A dark or high-weight edge is not a measured physical interaction.

![Thermal typed donor-anchor-receiver paths, moment cards and signed heat coefficients](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_hypergraphs_e2500_v6/thermal_0291_physical_hypergraph_e2500.png)

**Thermal 0291, e2500, fixed DEV probes 4538/4752/4848.** Each probe retains 788 positive head-0 donor paths through four of 192 anchors, with Λ mass one. C0/M/H processed-state RMS is 0.795/0.801/0.833; the new residual RMS is 0/0.380/0.444. Across displayed probes/active sources, signed K spans about −0.179 to 5.734 packed-native T/raw heat_power, local response included. Coordinates, cover weights, moments and signed coefficients come from the saved arrays. The dense paths describe learned information flow; they are not a sparse execution or a physical causal graph.

![Wind typed hypergraph, physical rotor-diameter projection and processed moment state](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_hypergraphs_e2500_v6/wind_W69_physical_hypergraph_e2500.png)

**Wind row 69, e2500, fixed DEV probes 970754/970657/970841.** The probes retain 576/288/576 positive head-0 paths through eight of 64 anchors, with Λ mass one. C0/M/H Z RMS is 1.324/1.557/1.528 and new residual RMS is 0/1.059/0.999. x/y use native rotor-diameter coordinates and z is retained/color-coded. The active richer state does not imply better wake accuracy: the full-panel H near-vector RMSE remains 0.629116 m/s.

### Actual interventions, restoration and mixed effects

All 24 typed head-0 exchanges held final contextual states bitwise fixed, retained positive support and declared row/column residual bounds, rebuilt every dependent mean/moment/Z/heat factor, and restored native/fixed-probe outputs bitwise. The following RMS changes are **full native output / fixed-probe output**, in packed T for Thermal and m/s for Wind. Zero-new retains each arm's own coadapted trained base; it is not the separately trained C0 model.

| Dataset | Arm | Scene | Module exchange | Environment exchange | Zero-new-output intervention |
|---|---|---|---:|---:|---:|
| thermal | C0 | 0291 | 0.00210345 / 0.0130085 packed T | 4.04861e-06 / 1.51009e-05 packed T | n/a / n/a packed T |
| thermal | C0 | 0687 | 0.00103984 / 0.00132395 packed T | 0.000600273 / 0.00392094 packed T | n/a / n/a packed T |
| thermal | M | 0291 | 0.00159968 / 0.0105131 packed T | 1.94411e-06 / 6.64951e-06 packed T | 7.17423 / 8.27449 packed T |
| thermal | M | 0687 | 0.00093392 / 0.00381863 packed T | 0.00014167 / 0.000677325 packed T | 12.4826 / 26.4442 packed T |
| thermal | H | 0291 | 0.00143282 / 0.00957474 packed T | 4.29745e-07 / 7.43069e-07 packed T | 9.32982 / 11.7592 packed T |
| thermal | H | 0687 | 0.000599128 / 0.00169018 packed T | 0.000287601 / 0.00145172 packed T | 16.7045 / 35.7172 packed T |
| wind | C0 | W69 | 6.93312e-06 / 2.02922e-05 m/s | 0.000356092 / 0.000480374 m/s | n/a / n/a m/s |
| wind | C0 | W426 | 3.38784e-06 / 1.37015e-05 m/s | 0.000248806 / 0.000528161 m/s | n/a / n/a m/s |
| wind | M | W69 | 4.38766e-06 / 1.33873e-05 m/s | 0.001534 / 0.0034307 m/s | 0.112917 / 0.118718 m/s |
| wind | M | W426 | 2.40431e-07 / 3.20221e-07 m/s | 0.000281707 / 0.000473111 m/s | 0.217224 / 0.285345 m/s |
| wind | H | W69 | 1.15466e-05 / 3.93517e-05 m/s | 0.000355072 / 0.000839569 m/s | 0.108219 / 0.157658 m/s |
| wind | H | W426 | 2.51705e-07 / 6.38436e-07 m/s | 0.000475612 / 0.000110168 m/s | 0.203932 / 0.36515 m/s |

The new path is active: zeroing M/H outputs changes Thermal full-native T by 7.17–16.70 packed units and Wind velocity by 0.108–0.217 m/s on these scenes. Ordinary association exchanges produce much smaller, type- and scene-dependent changes. Those tests establish dependence, not benefit: residual errors can improve or worsen when the path is removed, and the large zeroing response can reflect coadaptation of the retained base and new path.

M remains affine in the external moment descriptor to FP64 roundoff (mixed maximum about 8.94e−16). H has nonzero FP64 descriptor curvature, but its mixed RMS is only 4.17–17.65% of the matched same-weight M-affine FP32 four-point cancellation control on the predeclared tiny increments. This does not resolve a meaningful ordinary-FP32 higher-order physical effect. Whole-model geometry nonlinearity is separate: contextualization, cover and local decoding already allow mixed effects in C0 and M.

| Dataset | Arm | Scene | Native output | Mixed RMS | Mixed / geometric-mean single effect | Unit |
|---|---|---|---|---:|---:|---|
| thermal | C0 | 0291 | packed-native temperature | 0.000681166 | 0.01075 | packed_dataset_native_temperature |
| thermal | C0 | 0687 | packed-native temperature | 0.00107767 | 0.02063 | packed_dataset_native_temperature |
| thermal | M | 0291 | packed-native temperature | 0.000303163 | 0.00575 | packed_dataset_native_temperature |
| thermal | M | 0687 | packed-native temperature | 0.00170692 | 0.04111 | packed_dataset_native_temperature |
| thermal | H | 0291 | packed-native temperature | 0.00126341 | 0.01819 | packed_dataset_native_temperature |
| thermal | H | 0687 | packed-native temperature | 0.0010896 | 0.02489 | packed_dataset_native_temperature |
| wind | C0 | 69 | native velocity | 0.000416747 | 0.04383 | m/s |
| wind | C0 | 426 | native velocity | 0.000223513 | 0.06474 | m/s |
| wind | M | 69 | native velocity | 0.000216696 | 0.05123 | m/s |
| wind | M | 426 | native velocity | 0.000129343 | 0.03489 | m/s |
| wind | H | 69 | native velocity | 8.52978e-05 | 0.01622 | m/s |
| wind | H | 426 | native velocity | 0.00017353 | 0.03619 | m/s |

These whole-predictor contrasts shift two preselected active source IDs by 0.5% of the native extent along the prospectively selected horizontal axis, rebuild all four complete model states, and use the same fixed native probes. Single effects are used as the normalization shown. No variant is uniformly lowest; Wind's geometry-independent background has exactly zero mixed contrast. No compatible four-state physical reference is available, so neither agreement with physics nor an H-specific interaction advantage is inferred. The separate input-only TRAIN source-geometry probe uses no query coordinates, output values or targets.

![Thermal association interventions, processor curvature and source-affine controls](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_hypergraphs_e2500_v6/thermal_organizer_controls_e2500.png)

**Thermal e2500 organizer controls, DEV 0291/0687.** All 12 Thermal association exchanges restore exactly. M affine descriptor contrasts stay at FP64 roundoff; H FP64 contrasts must be read against the M FP32 cancellation control. Across the six Thermal scene/arm controls, zero heat gives exact zero T, flow is bitwise unchanged, the largest heat-adjoint discrepancy is 5.93e−19, and the largest prepared field mixed increment is 6.94e−18. Endpoint maxima remain nonlinear summaries even when temperature fields are source-affine.

The [organizer analysis](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/source/receipts/round5_organizer_analysis_v1.json), [direct saved-array review](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/source/receipts/round5_hypergraph_numeric_review_v1.json) and [fixed geometry preregistration](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/whole_predictor_geometry_mixed_contrast_preregistration_v1.json) retain complete bindings, normalized-core ports, intervention error changes and failure limits. C0 has no new moment statistics by design; absent C0 moments are not plotted as measured zero moments.

## Execution: concrete memory savings and latency costs

The source-bound terminal cost matrix completed all 24 jobs on physical GPU 1 after both own training lanes were idle: ten paired latency invocations, eight one-arm B8 memory processes and six all-macro optimizer replays. All 27 disposable optimizer updates completed with no failed step. The native scientific files remained unchanged. Low/high module fixtures and Q1024/8192 are retained in the full cost summary; the high-module results below use Thermal TRAIN 0631 with 12 active modules and Wind TRAIN row 533 with 30. Each pair has two warmups and seven synchronized, alternating measured trials, with both independent blocks retained wherever the preregistered ambiguity rule triggers. Intervals describe process-repeat uncertainty, not population variation.

### Complete calls and reusable scopes

The table gives candidate absolute median milliseconds followed by the paired candidate/baseline ratio and its 95% bootstrap interval. Each scope is timed independently: prepared-scene query time and fixed-query heat time overlap the complete call and must not be summed or subtracted to manufacture component timings. Absolute medians from different pair invocations are not one shared D0 reference.

| Dataset | Pair | Complete ms; ratio [CI] | Prepared scene, new query ms; ratio [CI] | Fixed query, new heat ms; ratio [CI] |
| --- | --- | --- | --- | --- |
| thermal | C0 / D0 | 42.607; 1.1653 [1.1620, 1.1768] | 17.782; 1.0682 [1.0672, 1.0709] | 3.824; 1.2989 [1.2494, 1.3047] |
| thermal | M / D0 | 49.963; 1.3639 [1.3468, 1.3752] | 17.746; 1.1108 [1.1010, 1.1297] | 4.415; 1.4873 [1.4869, 1.4910] |
| thermal | H / D0 | 50.612; 1.3636 [1.3528, 1.3938] | 17.902; 1.1005 [1.0907, 1.1204] | 4.390; 1.5040 [1.4967, 1.5090] |
| thermal | M / C0 | 49.028; 1.1590 [1.1547, 1.1771] | 18.462; 1.0316 [1.0305, 1.0342] | 4.193; 1.1549 [1.1254, 1.1602] |
| thermal | H / C0 | 49.015; 1.1548 [1.1539, 1.1637] | 18.331; 1.0309 [1.0303, 1.0335] | 4.391; 1.1562 [1.1518, 1.1588] |
| wind | C0 / D0 | 36.448; 1.1563 [1.1347, 1.1634] | 11.946; 0.9717 [0.9689, 0.9734] | — |
| wind | M / D0 | 42.176; 1.3516 [1.3501, 1.3569] | 11.600; 0.9853 [0.9821, 0.9892] | — |
| wind | H / D0 | 42.809; 1.3577 [1.3470, 1.3645] | 11.670; 0.9863 [0.9840, 0.9976] | — |
| wind | M / C0 | 44.399; 1.1707 [1.1692, 1.1762] | 11.658; 1.0278 [1.0232, 1.0317] | — |
| wind | H / C0 | 42.157; 1.1626 [1.1573, 1.1669] | 11.800; 1.0258 [1.0217, 1.0298] | — |

Complete calls include host-to-device prescribed inputs, contextualization/preparation, native decoding/derivatives/masks/local work and return to CPU; checkpoint/catalogue loading and target reading are excluded setup. H is 36.36% slower than D0 on Thermal and 35.77% slower on Wind. H and M are also about 15–17% slower than C0, so the moment package has real preparation cost. H and M have similar complete costs; these measurements do not identify SiLU alone as the dominant bottleneck. There is no complete-call latency saving to claim.

Prepared-query reuse is a narrower result: at high M/Q8192, Wind C0/M/H ratios to D0 are 0.9717/0.9853/0.9863, or about 2.8%/1.5%/1.4% lower time. Thermal H's prepared-query ratio is 1.1005 and its fixed-query heat-application ratio is 1.5040. The earlier same-function Stage E qualification found a roughly 10.6% prepared-query gain only for the Wind high-module Q1024 fixture (ratio 0.8940, interval 0.8261–0.9522); high-M Q8192 complete-call intervals for both adapters crossed one. That limited executor result cannot be promoted to a trained-family or full-call speedup.

![Thermal native latency with complete, prepared-query and heat scopes](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_family4_costs_display_v1/round5_family4_latency_thermal_v1.png)

**Thermal native costs, physical GPU 1.** At high M/Q8192, H/D0 complete latency is 1.3636 [1.3528, 1.3938], prepared-query 1.1005 [1.0907, 1.1204], and fixed-query/new-heat 1.5040 [1.4967, 1.5090]. Ratios use actual e2500 candidate weights, exact matched native inputs and paired repeats. The scopes overlap; no broad Thermal latency saving is measured.

![Wind complete and prepared-query native latency ratios](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_family4_costs_display_v1/round5_family4_latency_wind_v1.png)

**Wind native costs, physical GPU 1.** H/D0 complete latency is 1.3577 [1.3470, 1.3645] at high M/Q8192, while prepared-query reuse is 0.9863 [0.9840, 0.9976]. The small reuse gain coexists with a slower complete call. Both ambiguity-triggered trial blocks are pooled and displayed where required; the faster block is never selected alone.

![Same-function Stage E projection and topology-reuse timing controls](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_family4_costs_display_v1/round5_family4_stage_e_executor_v1.png)

**Same-function Stage E, separate from trained e2500 candidate comparisons.** The bounded Wind high-M/Q1024 prepared-query ratio is 0.8940 [0.8261, 0.9522]. Complete high-M/Q8192 ratios are Thermal 0.9918 [0.9607, 1.0309] and Wind 0.9988 [0.9576, 1.0113]. A universal complete-call gain is unsupported. The retained generic FP32 VJP failure limits the numerical equality claim as detailed below.

### Fixed B8 query-density memory

| Dataset | Arm | Q1024 allocated / reserved GiB | Q4096 allocated / reserved GiB | Q8192 allocated / reserved GiB |
| --- | --- | --- | --- | --- |
| thermal | D0 | 0.6431 / 0.7773 | 1.1105 / 1.3066 | 1.7721 / 2.1270 |
| thermal | C0 | 0.1839 / 0.2324 | 0.2746 / 0.3652 | 0.4017 / 0.5020 |
| thermal | M | 0.1963 / 0.2520 | 0.2869 / 0.3750 | 0.4136 / 0.5117 |
| thermal | H | 0.1971 / 0.2539 | 0.2876 / 0.3750 | 0.4144 / 0.5117 |
| wind | D0 | 0.5579 / 0.6523 | 2.1149 / 2.4668 | 4.1882 / 4.8887 |
| wind | C0 | 0.1145 / 0.1328 | 0.3301 / 0.3711 | 0.6188 / 0.7070 |
| wind | M | 0.1192 / 0.1328 | 0.3348 / 0.3730 | 0.6235 / 0.7090 |
| wind | H | 0.1194 / 0.1328 | 0.3351 / 0.3730 | 0.6238 / 0.7090 |

At Q8192, absolute allocated peaks fall from 1,902,828,032 to 444,920,320 bytes for Thermal H (−76.62%) and from 4,497,070,080 to 669,746,688 bytes for Wind H (−85.11%). C0 already has most of this advantage; moments raise H memory relative to C0 by 3.15% on Thermal and 0.80% on Wind. This experiment holds B=8 fixed and changes per-scene Q; reducing batch size would be a different comparison. The fixed high-module scene is repeated in eight batch positions, so this is a controlled density/stress fixture rather than a distribution of eight different production scenes.

This is **native forward/extraction and CPU-return memory**, including autograd used for native derivatives where required, not loss-backward scaling. Round4's older B8 stress numbers included a target-free backward pass, so the old and new absolute peaks cannot be compared as one like-for-like experiment. D0 and C0/M/H within this new matrix use equal native receiver unions and matched masks/local pairs. D0 has no IAG cover; C0/M/H cover ownership is checked within architecture, not falsely equated to D0's internal topology. Wind memory queries preserve its native volume:hub 3:1 training mix; the five-role endpoint accuracy board is a different declared query recipe.

![Fixed B8 native forward memory scaling for D0 C0 M H across query density](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/figures/round5_family4_costs_display_v1/round5_family4_b8_memory_v1.png)

**B=8, Q1024/4096/8192, one resident arm per process on physical GPU 1.** H’s Q8192 absolute allocated peaks are 0.4144 GiB Thermal and 0.6238 GiB Wind, versus D0 1.7721 and 4.1882 GiB. Reserved peaks and native-union checks are shown separately. This is a measured storage reduction with full-soft access, not a sparse executor saving or a measured B8 loss-backward reduction.

The work records show why memory and latency need separate evidence. At B8/Q8192 the projected context gather reads 134,217,728 bytes for Thermal and 268,435,456 bytes for Wind (128 FP32 values per selected anchor slot). H processes 1,536 Thermal or 512 Wind anchors per scene batch, retaining about 20.30 MB or 6.29 MB of unique moment-cache storage; the processed state is reused across queries. Far query/source value rows are zero, but all typed donor associations and full cover/local work are still executed. The moment contractions and processor increase scene work even when query memory is lower. Instrumentation aliases are not summed twice, and logical multiply-add counts are not hardware profiler FLOPs or a speed proof. Sparse execution/executor savings were not measured in this round; no optional sparse campaign was launched.

### Actual optimizer-boundary and training costs

| Dataset | Arm | Replay updates / visits / micros | Optimizer macro peak allocated / reserved GiB | e2401–2500 train-loop median s/epoch | History full-epoch max allocated / reserved GiB | Terminal fit-process interval and seconds |
| --- | --- | --- | --- | --- | --- | --- |
| thermal | C0 | 4 / 150 / 7 | 1.9852 / 2.6699 | 1.097794 | 2.0483 / 4.3184 | 2001–2500: 663.223s |
| thermal | M | 4 / 150 / 7 | 2.1160 / 2.7090 | 1.223967 | 2.1778 / 4.5977 | 1501–2500: 1500.507s |
| thermal | H | 4 / 150 / 7 | 2.1208 / 2.7090 | 1.242908 | 2.1833 / 4.5977 | 1501–2500: 1512.396s |
| wind | C0 | 5 / 72 / 9 | 0.8424 / 1.0781 | 1.047921 | 0.8425 / 1.7031 | 2001–2500: 605.762s |
| wind | M | 5 / 72 / 9 | 0.8592 / 1.0898 | 1.174279 | 0.8600 / 1.7148 | 1501–2500: 1355.719s |
| wind | H | 5 / 72 / 9 | 0.8597 / 1.0898 | 1.183296 | 0.8605 / 1.7168 | 1501–2500: 1348.491s |

The optimizer replay clones each e2500 model, named Adam state and RNG, then executes every actual deterministic terminal macro, including scheduled Thermal response work and final partial batches. It is not a historical e2499→e2500 continuation, and it writes no scientific checkpoint/history. The six replays completed 27/27 updates. No new D0 optimizer peak is measured. Setup, data-preparation, forward/backward/step and combined peaks remain separate in the cost summary.

Training history peaks cover transfer/forward/backward/optimizer and exclude monitoring. They differ from fresh-process macro peaks because retained allocations and allocator history differ. Thermal H's e2500 full-epoch allocated peak is 2,278,645,760 bytes; the maximum over e2401–2500 is 2,344,253,952. Training-loop medians exclude validation, curve output, setup and reboot delay. Terminal fit-process spans differ after recovery (C0 500 epochs, M/H 1,000), so their raw seconds are not a matched end-to-end architecture comparison. The saved cumulative train_seconds exposure includes authoritative retained epochs; discarded/replayed work is charged separately below.

## Failure diagnosis: supported observations and plausible explanations

These are ranked interpretations of the measurements, not identified physical or optimizer causes. No additional campaign was run to turn an interpretation into a result.

| Observation | Plausible failure reason | What the evidence does not establish |
| --- | --- | --- |
| Wind M/H stay close to C0 on all five roles; TRAIN4 near-vector error stays 0.435–0.445 m/s | The richer scene statistics do not recover the missing local wake detail or fit under this inherited objective/continuation | It does not identify an irreducible architecture limit, nor rule out a different fresh initialization, training objective or representation |
| Fixed TRAIN phase/derivative errors are broad, without consistent boundary concentration versus target/D0 controls | A cover-boundary-only repair is poorly targeted; more local basis detail or fitting changes may be more relevant | Four cases cannot prove that covers never matter; the optional jet was deferred prospectively, not declared universally ineffective |
| Thermal improves temperature while v/p remain worse than D0; flow/thermal gradient norm ratio falls to 0.132 for H on the fixed diagnostic | The shared representation and current objective may prioritize thermal fit while leaving flow capacity/optimization behind | Cosines remain small positive; no measured destructive-gradient-conflict cause or tested benefit from reweighting/PCGrad |
| H external-descriptor FP64 mixed effect is below the matched affine FP32 cancellation control; Wind H≈M | The extra nonlinearity is weakly used at the tested descriptor scale; pooled moments/affine processing may already dominate this path | Not proof that H is globally affine, or that other descriptor directions/scales have zero curvature |
| Large zero-new changes coexist with modest M/H-vs-C0 accuracy changes | The base and new residual may coadapt, so removing the path is a severe intervention rather than a clean architecture ablation | Large output dependence does not imply an accuracy contribution or physical interpretation |
| Moments add about 15–17% complete latency versus C0; prepared Wind reuse changes only a few percent | Scene preparation/native overhead dominates savings from a reused, narrower receiver gather at these fixtures | No profiler decomposition isolates one operator; overlapping scope medians must not be subtracted as a timing breakdown |
| Thermal generic VJP tolerance miss and saved FP32 endpoint subtraction failures | Reduction order and cancellation limit derivative/increment numerics in the original FP32 paths | The precise FP64 checks do not retrospectively erase those failures or validate arbitrary cotangents |

### Quantified fixed-TRAIN phase evidence

| Dataset | Arm | Component order | Equal-case centered residual SSE / target SSE |
| --- | --- | --- | --- |
| thermal | D0 | u / v / p / omega / T | 0.0165 / 0.0641 / 0.0618 / 0.0251 / 0.0908 |
| thermal | C0 | u / v / p / omega / T | 0.0431 / 0.0953 / 0.1260 / 0.0918 / 0.1214 |
| thermal | M | u / v / p / omega / T | 0.0309 / 0.0761 / 0.0920 / 0.0706 / 0.0617 |
| thermal | H | u / v / p / omega / T | 0.0266 / 0.0739 / 0.0780 / 0.0680 / 0.0778 |
| wind | D0 | Ux / Uy / Uz | 0.0224 / 0.0129 / 0.0150 |
| wind | C0 | Ux / Uy / Uz | 0.2176 / 0.1219 / 0.1193 |
| wind | M | Ux / Uy / Uz | 0.2127 / 0.1223 / 0.1114 |
| wind | H | Ux / Uy / Uz | 0.2085 / 0.1188 / 0.1018 |

These dimensionless ratios use only the fixed four TRAIN cases at e1000, centered within cells with at least eight valid samples. Wind H remains about 6.8–9.3 times D0 across components despite modest improvements over C0. Thermal M/H improve within-cell T relative to C0, but H is not better than M on that diagnostic. This supports a broad fitting/detail deficit in Wind; it does not isolate a cover mechanism. Phase-bin errors and target/derivative controls remain available in the [quantitative reduction](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/evaluation/round5_phase_quantitative_reduction_v1.json) and its bound source. The additional diagnosis/response visual exports were not selected for retention; quantitative evidence and their metadata errata are retained, without further visualization work.

The next useful experiment would need a new explicit budget and question: for Wind, first distinguish residual fitting/objective and local detail limits on fixed TRAIN evidence; for Thermal, examine role balance without assuming gradient conflict. The present round stops with the planned comparison and negative qualification. No new architecture choice, reweighting, inverse campaign or longer training run is authorized by these hypotheses.

### Saved Thermal response progress

| e2500 model | Fluid increment RMSE | Interface increment RMSE | Material increment RMSE |
| --- | --- | --- | --- |
| D0 | 0.0132537 | 0.0173677 | 0.0221208 |
| R4_soft_C | 0.0131234 | 0.0156446 | 0.0141053 |
| C0 | 0.0136685 | 0.0189704 | 0.0186748 |
| M | 0.0107669 | 0.0151742 | 0.0149006 |
| H | 0.0104838 | 0.0165583 | 0.0154726 |

Each entry is the arithmetic mean of six case/direction native weighted RMSEs, not a pooled error: saved 0291/0294/0687 baseline/minus/plus, exact source IDs/heat controls, quadrature and common finite support. The 0277 baseline remains missing. H improves these means by 20.90%/4.66%/30.05% versus D0. Relative to the prior R4 soft-C, H fluid improves 20.11% but interface/material worsen 5.84%/9.69%; M retains slightly better interface/material errors than H. Thus some prior response progress remains, but it is not uniformly preserved.

Across C0/M/H, 53 of 54 role/case/direction comparisons fail the original ordinary-FP32 endpoint-subtraction check (18/18, 18/18 and 17/18); all 54 precise prepared raw-factor FP64 comparisons pass, maximum absolute discrepancy about 3.15e−13. Those are comparison counts, not counts of failed array elements. Physical-reference error and arithmetic consistency are different: passing the precise affine check does not make the increment physically exact. Endpoint module maxima are recomputed from endpoint fields, never inferred as a maximum of a linear increment.

The [saved-response context comparison](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/evaluation/round5_saved_response_context_comparison_v1.json) verifies exact equality of the original request/input hashes, source IDs, controls, role counts, units, quadrature and finite support with D0/R4 soft-C. Original FP32 failures, precise factors, numerical masks and native fields are retained. Thermal’s own saved request declares the analytic-wake/shared thermal-grid benchmark and q proxy limits; 11 historical saved reference_solver record files report convergence and no GPU use. No new reference call was made. The absence of compatible four-state geometry truth remains unchanged. D0/R4-soft C are present in this context comparison; their absence from the nonselected C0/M/H diagnosis display was only a display-index omission, corrected by the retained metadata errata.

## What changed mathematically

The common contextualizer still computes the final module and environment token states, and its last EM/EE attention probabilities define typed donor-to-anchor distributions P. C0 projects the inherited full typed means at the anchors before the geometry-owned query gather. M and H add compact spatial/value moments and a scene-level processor at those same anchors. Queries reuse that prepared state through the existing four/eight-slot cover and local path. There is no extra selector, query/source value MLP, new environment grid or additional contextual feedback cycle. Separate Thermal and Wind weights use the shared implementation.

| Control | What it changes | What the comparison isolates |
|---|---|---|
| Original C → projected C | Algebraic placement of the existing projection and reuse of scene-owned topology | Same-weight executor changes within the checked numerical scope |
| C0 → M | Typed moments, an affine residual processor and moment-dependent Thermal transport | The richer representation/transport package |
| M → H | SiLU between the same two processor layers | Processor nonlinearity with matched inputs, width, parameter inventory and initial tensors |
| H → cached jet | Not launched | No result; the fixed TRAIN diagnosis did not justify this optional revision |

For each typed row, normalized donor coordinate ξ, anchor coordinate ξa, donor-only head lift u of width 8 and actual floating-point association mass s=ΣP, the descriptor retains the full width-32-per-head ordinary means, the lifted mean ū=ΣPu, relative centroid d=ΣP(ξ−ξa), raw position/value cross moment C=ΣP(ξ−ξa)⊗u, the upper triangle of spatial covariance and the diagonal latent variance. C is a raw cross moment, not centered cross covariance. Module and environment measures retain their separate native definitions. The descriptor also includes anchor coordinates/characteristic lengths, operating context, validity and weighted donor lengths: width 708 for Thermal and 814 for Wind, four heads and processor width 128. Latent moments have latent or nondimensional units, not energy, physical response or calibrated uncertainty.

Aggregation uses donor-only lifts and P contractions without constructing anchor×donor×value features. Live centering and the actual row mass are preserved. For z=u−c, t=ΣPz and q=ΣPz², variance is evaluated as q−t²+2ct(1−s)+s(1−s)c², using the first valid donor's live c. The predeclared gamma(4N+16) allowance permits only roundoff-sized negative diagonal clamps; larger negative values fail. Signed off-diagonals remain signed, empty typed rows are exactly zero, and no square root or silent row renormalization changes the descriptor.

M and H are respectively W2(W1η+b1)+b2 and W2 SiLU(W1η+b1)+b2. Their new tensors have deterministic by-name matching initialization. W1 is nonzero; W2/b2 and the Thermal transport output heads start at zero, without another zero gate. The complete initial M/H function therefore recovers C0. Output-layer gradients become active at the first real update, and the upstream moment/first-layer gradients become active after the second. M/H add 121,696 Thermal or 122,880 Wind parameters relative to C0. An M advantage would support richer cached statistics without supporting an extra nonlinear processor.

### Native Thermal and Wind laws

Heating remains separate from contextualization, P, moments, processor state, receiver cover and normalization. Thermal's original local source response remains present, and response_phi already reads global context. The new anchor coefficient is

\[K_{ahri}^{\mathrm{anchor}}=\frac{P^M_{ahi}}{m_i}t_{ihr}\left[1+a^0_{ahr}+\sum_\ell a^\ell_{ahr}(\xi_i-\xi_a)_\ell\right],\qquad T(\mathcal X,h)=K(\mathcal X)h.\]

The same raw factors and source IDs serve ordinary prediction, heat increments, JVP, VJP and bounded explicit K export. Precise accumulation casts those factors before contraction; casting an already contracted FP32 coefficient cache would not implement that check. Zero heat gives exact zero temperature and heating leaves flow unchanged. A maximum is recomputed at each endpoint; max(T(h+δh))−max(T(h)) is not max(δT). Thermal temperatures, pressure and q here use the original packed dataset units; q is the retained proxy, not a new SI heat-flux calibration.

Wind keeps the native wind-aligned rotor-diameter coordinates, geometry-only E64 inputs, TRAIN-fitted residual transform and height-profile conversion. Its output is the prescribed background profile plus the transformed learned velocity residual in m/s. Geometry changes reprepare the model; Wind has no separate affine heating application. Neither adapter substitutes a generic normalized score for native field accuracy.

## Matched training and evaluation protocol

Every child branches from its own dataset's Round4 C-soft literal e500 state; Wind's parent is C, not Q. Parent Thermal SHA is fa3d22b7dd7d09ecbcad9bcabf8ed787cbc15a4737d2f55de52d0efc87db4b61 and Wind SHA is dd40112546c5f6ad614db709a342acf3f9f5f4c9bffb4f51cafebfde90bc7aa1. Common model tensors, named AdamW moments/groups, RNG, provider/sampler state and the entire 500-row history prefix are inherited. New parameter moments begin at their actual birth. All parameters train jointly; this is a controlled segmented continuation, not a fresh full-TRAIN run.

| Setting | Thermal | Wind |
|---|---|---|
| Fixed selected TRAIN / exposed DEV | 150 / 22 cases | 72 / 24 direction rows from 24 / 8 layouts |
| Initialization / deterministic sampling seeds | 0 / 0 | 0 / 42 |
| Effective cases / microbatch | 48 / 24 | 16 / 8 |
| Primary training query budget | Q 1,024 plus original native auxiliaries and scheduled response addendum | Q 8,192: 6,144 volume + 2,048 hub |
| Epoch exposure | 150 visits, 4 optimizer macros, final partial 6 | 72 visits, 5 macros, final partial 8 |
| Optimizer and schedule | All-parameter AdamW, constant 3e−4, weight decay 1e−5, betas (.9, .999), epsilon 1e−8, clip 1 | Same |
| Primary age / new child horizon | Literal e2500 / 2,000 new epochs | Literal e2500 / 2,000 new epochs |

Normalization is fitted only on the same selected TRAIN membership and reused unchanged. The two datasets use their own transform conventions. Identical case/query draws, objectives, schedules and update budgets are checked across C0/M/H. A development epoch visits the selected dataset, not full source TRAIN. Every child retains latest, best-field and declared 100-epoch milestones, complete validation/history, curves refreshed every 10 epochs and terminal tqdm/log output. Separate validation files exist only at milestones/final reviews. Historical reference loading, old run settings and holdout locks are preserved.

The native Thermal board uses all 8,192 stored 64 × 128 field receivers, original fluid masks, all 64 interface angles and all stored per-module material points on DEV 22. It reports fluid T, u/v/p/curl, surface/material T, q proxy and endpoint module peaks. Wind uses the fixed five-role Q 8,192 board: 1,640 each for volume/hub/downstream/near, 1,632 for background, and all three components in m/s. That board differs from Wind TRAIN's 3:1 draw and its Q 10,240 volume/hub monitor. Means and p90 are over cases/rows; direction rows sharing a layout are not independent geometries. TRAIN4 is a fixed descriptive fitting check, not a population estimate.

The new C0/M/H input/query/target/mask hashes match one another and the retained native comparison support. A CPU-only identity bridge checks all 22 Thermal cases exactly against Round4 and all 24 Wind rows directly against D0's native flat IDs/roles/coordinates/targets. The old Round3 Thermal population JSON did not serialize per-case target hashes: no missing legacy hash is invented. Its Round4 parity checks and fixed-scene D0 arrays are retained, and the exact new-to-Round4 bridge is explicit. TEST remains locked.

Literal e2500 is the primary model comparison. The secondary selector uses only the common saved e600:100:e2500 cadence, earliest exact ties, Thermal field_score or Wind val_volume_mse. Parent e500 and off-cadence clean stops are ineligible. Native endpoint boards, saved response errors and graph appearance never select checkpoints. Mature classics 1404/1804 and 2102/2103 retain their original loaders and selected ages; full-TRAIN membership, normalization, objectives and exposure differ. Their same-panel numbers show practical distance and do not isolate architecture or gate this first research version.

## Correctness and executor qualification

The maintained implementation and tests were pushed before the scientific runtime was frozen at 660be43ec2b67ba3867790e8c3efcdda032f3432. The read-only snapshot contains 1397 source files and 29,495,944 bytes. Scientific runs and native analyses use that snapshot. A later maintained-loader fix at 411d072534a5fc52118553e5be919bc35655acbc adds the missing Mapping import; the frozen generic moment dispatch is bypassed only through its unchanged direct native factory. Actual Thermal/Wind e600 CPU loads verified identical constructors, model tensors and provider identities. No frozen file was hot-edited.

Projection-before-gather retains the bias exactly once, full mean width and live selected-anchor/query geometry gradients. Scene-owned sorted-axis/lattice topology is reused while version/identity guards remain active. Plain Torch indexed gathering is the scientific executor. The installed compiler passed forward/first-gradient checks but its AOTAutograd path did not support the native double backward; compiled training was not used or credited with a speedup.

One generic Thermal mixed-B/Q 8,192 summed-cotangent VJP check failed the original elementwise tolerance: max difference 1.58156e−5 against 1.25391e−5 at a near-cancelled derivative (−0.025391019 versus −0.025375204). That failure is retained. The predeclared fixed-cotangent operator bound, full small K export, own FP64 Kᵀc reference and propagated cross-bound checks pass; the tolerance was not relaxed after observing the failure. Qualification is limited to the checked native numerical contract, not arbitrary bitwise equality or every cotangent. Reordered FP32 reductions can also lead to different long optimization trajectories, so old Round4 C is contextual history rather than a bitwise child control.

The final focused core/runtime guard suite passed 27 tests, with 2 deselected and zero optimizer updates. Native zero-residual M/H qualification passed the low-, high-, and mixed-size fixtures across the tested query densities, checking complete native outputs/common gradients, the nonzero Thermal coefficient law, ID permutations, heat null/adjoints and Wind profile/query/source gradients. A real fresh-process attach/save/resume pilot completed 54 disposable optimizer updates across all six arms: each arm ran an e501 attachment epoch followed by an e502 exact-resume epoch, including that dataset's full native macro pattern. It checked exact named weights/Adam/groups, RNG/provider state, the full inherited history and active upstream gradients on the second update. These are software/native-engine checks; they are not physical validation measurements. Earlier failed harness/import attempts and their zero-step evidence remain available.

Standalone CPU construction/loading examples for both native adapters are in the [maintained guide](../../../guides/_bk/20261029_Guide_Consolidation/02_Attention_and_Unified_Routing_Lineages/Scene_Prepared_Moment_Hyperedges.md). The documented examples were executed on actual saved checkpoints without constructing datasets/providers/optimizers or allocating a GPU. They recovered the saved constructor configuration and all model tensors exactly.

## Review decisions and the optional jet

Reviews occurred after 100 new epochs (absolute e600), then at e1000 and e1500 before the final e2500 horizon. All six trajectories had finite, contiguous histories and identical within-dataset case/query fingerprints. At e1500, every rotating loss median fell relative to 981–1000, including Thermal response terms; fixed TRAIN4 supported continued fitting. Thermal M fluid-temperature DEV RMSE was 0.943792 versus C0's 1.434927 (−34.2%); H's vertical-velocity error was 0.005246 versus C0's 0.006612 (−20.7%). Wind near-vector errors remained 0.6243–0.6317 m/s versus C0's 0.6279 m/s, so the representation had not recovered the wake deficit. These were continuation reviews, not final qualification.

At e1000, the prospectively fixed TRAIN4 diagnosis used complete native central differences, all ten within-cell phase bins and centered residual SSE divided by centered target SSE in cells with at least 8 valid points. Original native coordinates, masks and finite target support are retained; zero target variation produces an unavailable ratio rather than an invented finite denominator. Wind derivatives use native stored-grid neighbors and complete query/local decoding, not just query-encoder derivatives. D0 and target variation are controls. Four related training cases are descriptive evidence, without a causal cover diagnosis or per-query independent-sample interpretation.

The fixed TRAIN4 sample showed a fitting gap, especially for Wind, but phase concentration was mixed across cases/axes. Some Wind x-direction near-anchor concentrations exceeded D0; the deficit was nevertheless broad across the full phase range, and the target/D0 controls did not support a consistent boundary-localized explanation. Both optional cached-jet children were therefore deferred at e1000 before mature DEV outcomes. No primary arm's architecture, losses, cover, grid or sampling changed later.

A separate Thermal gradient diagnostic used one deterministic native TRAIN4 draw and the provider's actual loss numerators/denominators. The C500 weights were evaluated on that same e1000 draw, alongside C0/M/H at e1000; this is a weight-age control, not a new attachment-epoch training run. Shared-context flow/curl versus temperature/q/response cosines were 0.0830/0.0667/0.1021/0.1451 for C500/C0/M/H. Flow-to-thermal group norm ratios were 0.649/0.546/0.210/0.132; raw EM/EE score cosines were also retained. Low positive cosines and differing norms neither prove a loss-conflict cause nor justify reweighting. No optimizer step or extra training campaign was performed for the diagnosis.

## Predeclared endpoint decision

The [pre-endpoint criteria](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/pre_endpoint_native_qualification_criteria_v1.json) preserve the Round4 first-version research bands. Thermal priorities are fluid T, v and p; Wind priorities are the near-turbine and downstream vector errors and all three components in each role. Priority means must be at most 1.10 times D0, priority p90 at most 1.20, and all other means at most 1.15. The efficiency route additionally requires either complete high-M median latency at most 0.90 times D0, or absolute B8/Q8192 allocated memory at most 0.80 with complete latency at most 1.15. The accuracy-led alternative requires the named fluid-T or near-vector mean at most 0.95 times D0 with the other guards preserved and complete latency/allocated memory at most 1.10. All ratios use actual new weights and matched fixtures; old IAG timings cannot satisfy a new candidate's gate.

A roughly 10% named deficient-field improvement over C0 is a research signal, separate from fresh-run qualification. H is compared to M as well as C0. Thermal also needs an explicit review of retained soft-C q/material and saved-response progress; that review does not impose a new physical-inverse certification gate. Wind cannot qualify from a small C0-relative gain while remaining several times worse than D0. Mature-classic replacement is not required. Eligibility is independent by dataset, with at most one fresh full-TRAIN probe per dataset, fresh weights/optimizer and full-TRAIN normalization, and a literal e1000 stop. Passing one dataset is not made conditional on passing the other.

### Final qualification result

All six candidates pass the checked software/native identity bindings and finish healthy training, but all six fail the predeclared native bands and both numerical qualification routes. Thermal H passes the named fluid-temperature improvement, yet its v/p means and tails, u/curl guards, and complete latency still miss. Wind near/downstream component and vector errors remain above the bands. Memory alone does not qualify a model when complete latency exceeds the allowed 1.15 ratio and accuracy guards fail.

| Dataset | Arm | Native accuracy bands pass | Complete latency / D0 | B8 allocated / D0 | Efficiency route | Accuracy route | Fresh full-TRAIN eligible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| thermal | C0 | False | 1.1653 | 0.2267 | False | False | No |
| thermal | M | False | 1.3639 | 0.2334 | False | False | No |
| thermal | H | False | 1.3636 | 0.2338 | False | False | No |
| wind | C0 | False | 1.1563 | 0.1477 | False | False | No |
| wind | M | False | 1.3516 | 0.1489 | False | False | No |
| wind | H | False | 1.3577 | 0.1489 | False | False | No |

The root decision is **no fresh full-TRAIN e1000 run for either dataset**. The Thermal response addendum was still reviewed and reported, even though the numerical routes already fail; it is not an extra inverse-readiness gate. No full-TRAIN probe protocol is asserted ready or launched, no e5000 extension is run, and no monitoring or evaluation follow-up is scheduled.

## Interruptions and exact recovery

Two external host reboots interrupted the round, and `/data` was temporarily unmounted. The original ext4 volume UUID was checked before reuse; no replacement dataset, cache or run directory was created in an empty mount point. The first interruption left C0 latest checkpoints at e2000 and M/H at e1500. Resume used those authoritative states with the same frozen source, exact native recipe, weights, named Adam/RNG/provider/sampler state and checkpoint-contained history. A newer curve-history row was not attached to older weights. Original histories and their uncheckpointed tails are retained losslessly in the [first recovery archive](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/recovery_session_loss_20261016T175803Z/preparation.json); the [second recovery manifest](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/recovery_session_loss_20261016T195123Z/preparation.json) records the successful restart. The empty-log intermediate attempt produced no verified finite update; its possible unobserved work is not asserted to be zero.

Known discarded scientific optimizer updates have a cumulative lower bound of 448: Thermal 80 and Wind 368 from the observed progress beyond the last checkpoints. Final in-flight work and the empty-log retry are not exactly recoverable. These are additional executed/replayed work, separate from the final logical e500→e2500 exposure. Resource accounting conservatively charges the two interrupted 2400-second reservations and two interrupted 1200-second retry reservations in full, totaling two GPU-associated process-hours. Wall-clock interruption overhead is not attributed to an architecture's native inference or training speed.

The [scientific replay receipt](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/recovery2_scientific_history_replay_v1.json) confirms exact matches for Thermal e2001–2020 (20/20 rows) and Wind e2001–2070 (70/70): loss terms, case/query fingerprints, LR/phase/temperature, case visits, updates, partial macros and work counters. Timing/memory are excluded. Gzip payload and original-history SHA checks pass. This supports deterministic scientific replay on the retained interval, without claiming an unavailable bitwise comparison of the entire old model trajectory. Compact recovery instructions, contracts and checkpoint inventories are also retained on the workspace filesystem in the [metadata handoff](../../../../diagnostics/generated/round5_restart_handoff_20261016_v1/HANDOFF.json); it contains no dataset or model payload and schedules nothing.

## Resource and measurement scopes

The round's hard ceilings are 20 aggregate GPU-associated process-hours, 14 elapsed hours including setup, and 400 disposable engineering optimizer attempts; the final two elapsed hours are reserved for reporting and cleanup. Persistent new `/data` payload is capped at 12 GiB, `/data` temporary at 1 GiB, new workspace payload at 1 GiB, and `/tmp` temporary at 256 MiB, with 20 GiB free headroom on each filesystem. The exact verified shared native caches are reused read-only. New dataset/cache stores are not created. Scientific training updates are reported separately from engineering checks.

The [terminal measurement declaration](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/terminal_measurement_resource_declaration_v1.json) reserves 12 DEV/TRAIN4 boards, 12 organizer/plane/response exports, six input-geometry contrasts, at most six selected-age native boards and the 24-job cost matrix. Their conservative aggregate process ceiling is 4.6 hours; existing round payload was 3.491 GiB and the additional numerical-output ceiling 1.875 GiB before those jobs. Selected figures have a 256 MiB aggregate ceiling and 64 MiB per family. Engineering checks total 101 attempted/completed updates: 74 earlier checks and 27 terminal all-macro updates, below 400. The declarations are bounds; measured process usage and final cleanup are reported below.

Headline timings use exact inputs on physical GPU 1 after both own training/native lanes are idle, synchronized profiler-off alternating repeats, two warmups and seven measured trials. Ambiguous results receive a second independent block across all declared scopes, and both blocks are retained rather than selecting the faster one. A complete call starts from prescribed host inputs and ends after native outputs return to CPU; checkpoint reconstruction, dataset/catalogue loading and target reading are setup scopes. Prepared-scene/new-query and fixed-query/new-heating are separate overlapping scopes and are never added to complete-call medians. Thermal native derivatives, masks, covers, source identities, local work and return copies remain included where required. Shared-host observations are not a quiet-machine architecture speed law.

Fixed-B8 Q1024/4096/8192 memory measures complete native forward/extraction and CPU return, including the native derivative work, absolute allocated/reserved peaks and actual receiver unions. It is not a fixed-B8 loss-backward scaling experiment. Actual optimizer-boundary memory is measured separately by replaying every deterministic e2500 macro from a cloned terminal state: four Thermal macros or five Wind macros, including scheduled response work and the final partial batch. Those disposable updates are cost measurements, not a scientific e2499→e2500 continuation; the original checkpoint/history files must remain unchanged. TRAIN-loop epoch time, provider preparation, optimizer/setup/preparation peaks and complete process time retain their distinct scopes.

### Measured resource use and cleanup

The [measured resource receipt](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/round5_resources_measured_v1.json) closes all 152 GPU-associated process entries: 130 completed, 18 failed invocations and four external-reboot interruptions. Process categories are qualification 11, training 28, evaluation 86 and cost 27. All activity was on physical GPUs 1/2; no task allocation or signal targeted GPU 0. Charged associated process time is 23,642.013 seconds (6.567226 hours) against the 20-hour cap, including 7,200 seconds conservatively charged for interrupted requests. This is process-associated accounting, not GPU-active kernel time. The process envelope is 14:16:48–21:15:17 UTC; the training envelope is 16:02:46–20:53:07 UTC, including gaps/reboots. Total round elapsed time begins at 13:53:43 UTC and final closeout remains within the 14-hour ceiling. Failed invocations are retained evidence, not failed optimizer steps or scientific completions.

The authoritative e500→e2500 histories contain 54,000 new scientific updates, 1,332,000 case visits and 4,460,544,000 primary training queries across the six arms, plus native auxiliary/response work in the full counters. Each Thermal arm contributes 8,000 updates/300,000 visits/307,200,000 primary queries; each Wind arm contributes 10,000/144,000/1,179,648,000. Terminal fit-process receipts contain only the final 22,500 updates because resume spans differ; they are not the full logical comparison exposure. The known discarded/replayed lower bound of 448 updates and unknown final in-flight work remain separate. Disposable engineering actual/charged updates are 101/400 (74 earlier checks plus 27 terminal macro updates); aggregate reservations of 117 are upper bounds, not executed updates.

The [protected-input audit](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/round5_protected_closeout_v1.json) passed 21,816 regular-file checks, 77 symlink checks and 529 shared-cache manifest hashes with zero errors. `/data` retained its original UUID; device-number changes after remount were disclosed while inode, bytes, timestamps, links, ownership/mode and required hashes were compared. Round4's files/symlinks match; its historical file-only manifest leaves 21 current directory nodes historically unpinned, so no directory-history equality is claimed. There were no traversal errors or special entries. Large shared-cache payloads were not rehashed and TEST targets were not read. Prior seals, reports, source/history/checkpoint bindings and user changes remain protected. The [frozen-runtime check](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/round5_final_immutable_runtime_verification_v1.json) independently verified all 1,397 original source files, 29,495,944 bytes, inode/hash inventory and absence of write bits.

Before final cleanup, the round root used 4,629,008,384 allocated / 4,622,277,426 apparent bytes, including about 3.72 GB of protected primary runs, 420 MB of temporary files and 61 MB of figure exports. `/data` free space was 310,001,561,600 bytes; `/home` and `/tmp` each had 171,715,330,048, above the 21,474,836,480-byte floor. These pre-closeout observations exclude the resource receipt and later reporting files. Global free-space increases during external reboots are not credited to this task.

Audited cleanup removed **309 regular files / 458,771,471 bytes**, plus one disposable browser lock symlink and its empty profile directories: 246 round-root files / 439,774,670 bytes (expired pilot weights/curves, tiny CPU fixtures, generated compiler cache, superseded/nonselected exports, QA rasters and font/failed-preview cache); 60 workspace browser files / 17,856,411 bytes; one earlier bytecode file / 73,454 bytes; and two temporary audit-copy/preparation files / 1,066,936 bytes after lossless archival. Unmetered earlier ephemeral deletions are excluded from this audited total. The original failed-browser log (69 bytes), compact pilot histories/JSON, all six primary latest/best/milestone states, recipes, numerical/reference arrays, source and provenance remain retained. The selected visual payload is six PDF masters plus nine PNG companions, totaling 10,834,567 bytes in four figure families. Older organizer arrays are intentional version/failure evidence with recorded duplicate hashes, not disposable cache copies.

The [exact allowlist](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/round5_cleanup_allowlist_v3.json), [completed cleanup receipt](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/round5_cleanup_order_recovery_v1.json), [browser cleanup receipt](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/round5_browser_cleanup_receipt_v1.json) and [lossless original-inventory archive receipt](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/round5_cleanup_original_inventory_archive_v1.json) identify exact removed paths, bytes, inodes/links, hashes and retained replacements. An ordering-dependent guard stopped after four approved figure removals; the original partial receipt remains unchanged, and the narrow recovery used the same manifest-bound obsolete-PDF identities to complete the remaining entries. All approved payloads are absent, protected tree metadata and selected PNG/PDF hashes match, and the [post-cleanup science check](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/round5_postcleanup_science_retention_v1.json) independently rehashed all six primary endpoint checkpoints/histories and the selected/best aliases. Histories remain 1..2500. Accessible FD/maps were checked; root-owned privilege-separated/foreign proc endpoints remained explicit visibility limitations, without escalation, signals or a claim to privileged whole-host inspection.

At the post-cleanup observation the root held 4,235,248,346 apparent regular-file bytes (about 3.94 GiB), before the final small receipts/handoff. Actual free capacity was `/data` **310,386,663,424 bytes**, `/home` **171,696,746,496 bytes**, and `/tmp` **171,696,746,496 bytes**, all above the 20 GiB floor; later Git/report/handoff metadata consumes a small additional amount. File-byte removal totals and global free-space changes are separate measurements. Only the unrelated GPU0 compute PID12472 remained in the post-cleanup compute inventory; own GPUs1/2 were idle and no follow-up was scheduled.

The [CPU-only browser preview receipt](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/source/receipts/round5_report_qa_252ae36366d3_20261016T214910Z.json) records one readable 1600×1200 top viewport, not full-page browser inspection, and exact original/staged hashes for all nine PNGs. GPU PID sets were unchanged throughout. That preview preceded final closeout/command-index text additions; final source paragraphs, every local link and all embedded image payloads were rechecked after cleanup. Individual selected PDFs/PNGs were separately visually inspected; no further plot campaign was added.

The [diagnosis-display metadata errata](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/source/receipts/round5_phase_gradient_responses_metadata_errata_v1.json) preserves the nonselected F5 display's completed numerical/visual evidence and post-finalize CLI error. D0 was excluded only from that display index, not unavailable globally; the separate request-identity-checked D0/R4-soft C context comparison is authoritative. The return-path bug was corrected and CPU validation passed without rerendering. The redundant F5 visual payloads are omitted under the user's request to prioritize quantitative analysis; phase/gradient/response arrays, reductions, manifests and failure/errata records remain retained. No generated figures, data or one-time diagnostics are uploaded to Git.

## Physical-reference and inverse limits

Native field accuracy, a learned route's executed effect and a validated physical interaction are different evidence. Association interventions hold final contextual token states fixed and rebuild every dependent mean, moment, processed state and heat coefficient; geometry contrasts rebuild contextualization as well. A nonzero output change establishes a model information path. It does not make the plotted support a physical causal graph. Processor curvature evaluates the external descriptor map with fixed weights; a nonzero whole-predictor mixed geometry contrast can occur for C0 or M as well as H.

The saved Thermal response replay uses the exact existing 0291/0294/0687 baseline/minus/plus records, both signed directions and all available fluid/interface/material roles. The 0277 baseline remains unavailable. Original FP32 endpoint-subtraction failures and precise prepared raw-factor FP64 increments are both retained. The analytic-wake/shared-grid reference is not independent CFD. No new four/eight-state physical geometry-response bundle is bound to the shifted-source illustration, and no missing physical target is synthesized. Temperature peaks are maxima over retained native support, not certified continuous-domain extrema; q remains its original proxy.

There is no inverse optimization, new design search, new PDE/CFD solve or new Wind perturbation truth in this round. The inverse-interface evidence is limited to native source IDs, the checked source-affine temperature law, null/flow independence, precise heat increments/adjoints and correctly recomputed endpoint maxima. It does not establish valid designs or inverse readiness. Own new solver attempts are zero and inherited usage remains 326/326; no independent global solver invocation ledger was found, so this is not a claim to have audited every unrelated process on the host.

The comparison has one initialization, shared inherited parents, one fixed exposed development panel and no fresh-start confirmation. Thermal strata are small; Wind direction rows are correlated within layouts and its training fixture lacks an M18 stratum. Latent moment summaries are finite statistics, not universal injective descriptions or sufficient physical states. Native packed Thermal units are not newly calibrated SI units. The retained generic FP32 VJP miss, FP32 increment failures, unresolvable processor curvature at the tested scale, and missing physical geometry-response truth bound derivative/organizer interpretations. TEST remains unopened and no validated inverse design is produced.

## Tested execution records

| Record | Retained evidence |
| --- | --- |
| Six sealed recipes and parents | [Recipe manifest](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/primary_six_recipe_manifest_v1.json) |
| Actual final continuation commands | [Thermal recovery2 command](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/source/primary_stage2500_thermal_recovery2_20261016T195123Z.sh) / [Wind recovery2 command](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/source/primary_stage2500_wind_recovery2_20261016T195123Z.sh) |
| Native boards / secondary selectors | [Native command matrix](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/source/terminal_native_analysis_commands_v4.json) / [selected-age records](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/source/selected_native_board_commands_v1.json) |
| Complete terminal costs | [Exact24-job matrix](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/source/terminal_cost_matrix_commands_v3.json) / [trained source pin v6](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/cost_trained_endpoint_source_pin_v6.json) |
| Focused software checks | [Final CPU guard receipt](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/final_round5_core_runtime_guard_checks_v2_steps.json) —27passed,2deselected,0updates |
| Executed native CPU loading examples | [Standalone example receipt](../../../../diagnostics/generated/moment_hyperedges_round5_20261016T135639Z/receipts/standalone_documented_cpu_examples_v1.json) — exact constructor/tensor checks,0GPU/provider/optimizer |

Training/evaluation scripts above are historical executed records with fixed source, recipes, resource guards and stop ages. All six jobs are complete; these records do not authorize rerunning them or replacing their output directories. The supported maintained CLI is `HONF_Proj/tools/moment_hyperedge_train.py`; inference construction/loading examples remain in the guide. No full-TRAIN probe command is presented as qualified or ready.

## Reproduction and retained provenance

The shared implementation is maintained under `HONF_Proj/src/honf_forward_core/inherited_attention/` and the native adapters/runtime under `HONF_Proj/src/honf_runtime/`, with Thermal/Wind native extraction reused. The [construction/loading guide](../../../guides/_bk/20261029_Guide_Consolidation/02_Attention_and_Unified_Routing_Lineages/Scene_Prepared_Moment_Hyperedges.md) contains CPU-tested native examples. Model construction, a software smoke, finite optimizer updates, native physical measurements and full-run completion are reported as distinct evidence.

Every run has its exact recipe, full contiguous history, latest/best-field aliases and declared 100-epoch checkpoints. The six primary endpoint receipts bind literal ages and hashes. The command matrices bind the native boards, selected-age boards, organizer/geometry/response analyses and the 24 terminal cost jobs. They are executed records, not instructions to restart completed jobs. Output defaults remain curves every 10 epochs, full validation history with separate files only at milestones/final, and tqdm enabled including tmux/log output; historical replay behavior remains explicit.

```bash
# CPU-only report QA; no training, datasets, optimizer or GPU allocation.
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home/wanglz/miniconda3/envs/ModularDT/bin/python -B \
/data/wanglz/ModularDT/diagnostics/moment_hyperedges_round5_20261016T135639Z/source/receipts/qa_round5_report_preview.py --qa
```

The exact GPU allocation wrapper, read-only runtime snapshot, source pins, local restart guide and final workspace metadata handoff are retained. Only physical GPUs 1 and 2 were used for this round; GPU 0 was protected for the unrelated job. External host reboots changed process/device IDs and global free capacity; those changes are not attributed to task cleanup. The final artifact index lists retained figure payloads, numerical sources, tests, commands, cleanup receipts and tip verification.
