# Dependency-correct ThermalChannel forward interface — measured bounded result

We kept Run3801's thermal predictor fixed, added two matched small learned flow readers, and removed redundant work from its separable thermal executor. The heat-independent reader repairs the forbidden heat-to-flow dependency through its actual inputs. Both readers learned useful flow fields and geometry responses, but their u and vorticity errors remain above the prospective replacement targets after the single authorized extension. **Retain the existing predictors; the new composed reader remains a dependency-repair candidate, not a general predictor promotion.**

**Predictor gains:** exact measured flow heat independence, preserved strong thermal fields, and affordable complete cold calls. At selected1000, independent-flow v and pressure meet the mean/p90 targets. **Misses:** u mean/p90 are 28.36%/32.74% worse and omega 69.72%/74.15% worse than the retained flow. Run3801's wrong0291 thermal response signs remain. **Evidence:** all22 exposed validation cases, four complete native field boards, finite heat changes, actual input VJPs, saved geometry references and matched full-call timings. **Next:** keep the current predictors and use the input separation as a controlled interface; no further fit or promotion was launched.

**Organizer gains:** the retained learned thermal organization survives unchanged, and unnecessary execution is removed. **Misses:** this experiment tests a programmed benchmark dependency; it does not establish unique hypergraph value. The previous weak joint-versus-separable/I-removal result remains. **Evidence:** actual nominal0291 P2 source memberships and measured dense physical execution. **Next:** any unique organizing claim needs its own matched intervention, outside this completed round.

**Inverse gains:** spurious heat derivatives of flow and the 8%-band pressure functional are absent from the independent reader, while geometry/receiver derivatives remain live. **Misses:** thermal direction errors, limited excitation rank, u/omega error and imperfect geometry-response amplitude prevent an inverse-readiness claim. **Evidence:** reverse mode and finite changes, plus already solved geometry and thermal records. **Next:** investigate a whole-layout-conditioned source-response thermal operator under a future explicit scope; no inverse generator, search or new reference attempt occurred.

## Field and complete-cost decisions

Equal-case native RMSE on fixed25_v1's22 repeatedly exposed validation cases; temperatures, velocity, pressure and vorticity use dataset benchmark units, not asserted SI units. No independent-test claim is made.

| Role | Retained thermal/old flow | Independent500 endpoint | Independent selected1000 | Enabled selected1000 |
|---|---:|---:|---:|---:|
| fluid/temperature | 0.770312 | 0.770312 | 0.770312 | 0.770312 |
| surface_temperature | 0.788326 | 0.788326 | 0.788326 | 0.788326 |
| material_temperature | 0.66308 | 0.66308 | 0.66308 | 0.66308 |
| fluid/u | 0.0220367 | 0.0314009 | 0.0282855 | 0.0290304 |
| fluid/v | 0.00307966 | 0.00417507 | 0.00314433 | 0.00310077 |
| fluid/p | 0.0122219 | 0.0148848 | 0.0126933 | 0.0122821 |
| fluid/omega | 0.104567 | 0.210637 | 0.177467 | 0.178932 |

Targets were fixed in the plan: means within5% and p90 within10% of Run3801. Independent v is +2.10% mean/+0.44% p90; pressure +3.86%/+1.82%. Those pass. u and omega miss. The enabled control has slightly better absolute pressure, so structural independence is not an isolated accuracy gain. The common full-grid flow score uses four channels only; unchanged temperature is not averaged into the flow selector.

Complete B1/Q8192 actual TRAIN M12, same GPU0/FP32 and matched outer/inner tile512. Loading/input staging/export are outside these timers and inside resource accounting. Five alternating warmed inference repetitions and two input-VJP repetitions per model/condition. Cold means each call prepares all states afresh, although allocator/kernels are warmed.

| Model | Complete inference s | Complete heat/query forward+VJP s |
|---|---:|---:|
| G-fast | 0.0986506 | 0.194829 |
| H-add | 0.128393 | 0.270665 |
| D-sep | 0.134602 | 0.279851 |
| D-open | 0.134289 | 0.281921 |

Selected independent1000 / optimized retained ratios are **1.04835× inference /1.03394× VJP**, inside the1.20× inference engineering goal. Ratios to matched G-fast are1.36443×/1.43639×. Actual M1 and B8/M1/M12 scopes are in the appendix. The500 endpoint measured1.02437×/1.02728×; it is separate evidence, not substituted for the selected1000 costs.

Dense-D25 retained T/Ts/Tm/u/p RMSEs are 0.949971/0.974319/0.865606/0.0284574/0.0139807; matched-cohort G-fast retained values are 1.17451/1.23966/1.04896/0.0306896/0.0143447. These saved numbers and all historical controls remain in the [preceding report](HONF_Response_Competent_Forward_Refinement_Report.md). Dense, G-fast, old Tensor-H and mature Run3801 differ in representation and training history; the new pair isolates only its own heat-access policy. No historic checkpoint was replaced or restarted.

## Actual inputs, fits and selection

The [source audit](HONF_Dependency_Correct_Source_Audit.md) read the active local generator and all690 embedded configurations. This is prescribed analytic-wake flow plus a shared thermal grid, with projection disabled, no buoyancy or temperature-dependent flow coefficients. u/v/omega use geometry/inlet; pressure also uses prescribed viscosity/Re and its outlet gauge. Temperature is driven by heat with fixed thermal coefficients. Effective ports and q-normal are extracted thermal ratios/proxies; they are not legal independent-flow inputs. Source functions/lines and the complete dependency table are in that audit and the clarified `Dataset/PHYSICS_AND_DATA.md`.

The case-owned whitelist is centers, presence, radius, domain lengths, Re, inlet speed, kinematic viscosity and query coordinates. Two whole-layout source-context updates precede nonlinear individual-source query reads and reduction. All sources are dense. D-sep never reads current heating: its explicit experimental input column is a newly allocated constant zero. D-open reads normalized own-module heating. Both have identical shapes,235524 parameters, seed0 tensors and zero-initialized heat-column weights. D-open receives no observed temperature/pressure or solved ports. No analytic wake, teacher flow, case-ID field lookup, detach/gradient hook, output overwrite or parent-flow fallback supplies inference.

An ordinary candidate call runs one retained thermal wrapper and the learned reader. Its documented field order is u/v/p/omega/T; material/interface/port outputs remain inherited. Prepared decode carries both lane states and never falls back to parent flow. Fixed-layout `evaluate_heat_batch` reuses only independent flow preparation/query values within that request; changed geometry/context is rejected, shared local parameters containing stale own heat are rejected, and each heat endpoint rebuilds the thermal path. No temperature prediction is cached.

All inherited thermal parameters, normalizers, Stage-A and soft source organization remain fixed. The thermal parent is exact Run3801 Add500 SHA `13e2c8bd7c32a8a939b4d18b065300c04c73854412c203efabd728b5cd93006f`. Child checkpoints reference that preserved parent and store only their new flow state. The narrow flow-only training path matches ordinary composed flow values exactly on measured inputs.

The pair uses fixed25_v1 fingerprint `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`,150 TRAIN/22 validation, original TRAIN-only normalization, identical case/query streams, FP32 Q1024, microbatch8/effective48, four AdamW updates per epoch, learning rate3e-4, decay1e-5, clipping1.0 and one declared cosine horizon1000. Each epoch visits every selected case. Labels are existing H5 flow observations. No response-atlas auxiliary fit was added. The inherited thermal history still includes its150 primary cases plus three auxiliary original-TRAIN families;0348 is already primary TRAIN.

Native low-M0006 and high-M0231 optimizer checks measured finite nonzero gradients and changed flow weights. The paired initial functions/tensors agree exactly. Sampler audits preserve the original seed/case/epoch stream and native fluid catalogues; two packed FP32 boundary points on0231 lie marginally inside the radius inequality by about9e-8 squared-distance, with no materially interior receivers or relabelling.

Both100 fits received a real review: all22 physical metrics, four full field maps, exact heat null and nonzero geometry/query gradients. Both continued to500. Native500 still missed accuracy, while400→500 validation loss fell about19%; the complete-call target passed and the paired extension forecast about150 GPU process-seconds. One matched1000 extension was therefore executed and stopped. There was no tuning or further horizon. Both minimum saved100-cadence selectors chose1000, matching their endpoints exactly.

| New flow age | Independent normalized DEV score | Enabled normalized DEV score | Case visits per arm | Updates per arm | Query rows per arm |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.556625 | 0.565498 | 15000 | 400 | 15360000 |
| 200 | 0.132929 | 0.133762 | 30000 | 800 | 30720000 |
| 300 | 0.0955791 | 0.0971544 | 45000 | 1200 | 46080000 |
| 400 | 0.0743269 | 0.0751087 | 60000 | 1600 | 61440000 |
| 500 | 0.0604747 | 0.0611753 | 75000 | 2000 | 76800000 |
| 600 | 0.0515358 | 0.0524643 | 90000 | 2400 | 92160000 |
| 700 | 0.0462453 | 0.0473478 | 105000 | 2800 | 107520000 |
| 800 | 0.0437909 | 0.0448674 | 120000 | 3200 | 122880000 |
| 900 | 0.0426904 | 0.0437254 | 135000 | 3600 | 138240000 |
| 1000 | 0.0424898 | 0.0435358 | 150000 | 4000 | 153600000 |

These selector scores are equal-case/channel quadrature MSE over the identical fixedQ1024 panel, not full-population metrics. Final native field statistics use the complete grid and its fluid masks. Checkpoints/plots were written at100 cadence; only declared milestones, latest and best-field aliases remain. Stop/save/resume measured restored optimizer state; selected aliases equal their milestones scientifically, with4000 optimizer steps per arm. Parent SHA/model/normalization remain exact. There are no new thermal optimizer moments and no embedded second thermal wrapper.

## Five inspected figure groups

The short [figure index](../../diagnostics/generated/dependency_correct_20261006/figures/FIGURE_INDEX.md) identifies PDF masters, raster companions and numerical sources. Root inspected all included figures. All field/residual statistics use unclipped arrays; solid receivers are blank. Generated files and one-time renderers remain local and ignored. A fresh clone requires restoring this artifact root to display figures.

### 1. Learning, exposure and cost

![Matched learning, measured epoch cost and native flow target ratios](../../diagnostics/generated/dependency_correct_20261006/figures/01_learning.png)

Figure1. Each arm reaches150000 case visits,4000 updates and153.6M query rows; zero inherited thermal updates. Validation score falls from0.556625/0.565498 at100 to0.042490/0.043536 at1000. Native selected v/p meet the planned targets; u/omega do not. Epoch times measure flow-only fitting and cannot forecast formal full-model training.

### 2. Actual full fields and residuals

![Complete native u reference, both selected1000 predictions and residuals](../../diagnostics/generated/dependency_correct_20261006/figures/02_fields_u.png)

Figure2 (u). Full originalQ8192 native grid for0277/0291/0294/0687; equal-case all22 RMSE is0.0282855/0.0290304 independent/enabled in benchmark units. Common field and symmetric residual scales are not clipped. The reader predicts spatial flow rather than a constant lookup; absolute reconstruction limitations remain visible.

![Complete native p reference, both selected1000 predictions and residuals](../../diagnostics/generated/dependency_correct_20261006/figures/02_fields_p.png)

Figure2 (p). Full originalQ8192 native grid for0277/0291/0294/0687; equal-case all22 RMSE is0.0126933/0.0122821 independent/enabled in benchmark units. Common field and symmetric residual scales are not clipped. The reader predicts spatial flow rather than a constant lookup; absolute reconstruction limitations remain visible.

![Complete native omega reference, both selected1000 predictions and residuals](../../diagnostics/generated/dependency_correct_20261006/figures/02_fields_omega.png)

Figure2 (omega). Full originalQ8192 native grid for0277/0291/0294/0687; equal-case all22 RMSE is0.177467/0.178932 independent/enabled in benchmark units. Common field and symmetric residual scales are not clipped. The reader predicts spatial flow rather than a constant lookup; absolute reconstruction limitations remain visible.

![Complete native temperature reference, both selected1000 predictions and residuals](../../diagnostics/generated/dependency_correct_20261006/figures/02_fields_temperature.png)

Figure2 (temperature). Full originalQ8192 native grid for0277/0291/0294/0687; equal-case all22 RMSE is0.770312/0.770312 independent/enabled in benchmark units. Common field and symmetric residual scales are not clipped. The thermal field is deliberately preserved; these familiar residuals do not establish correct heat-response directions.

### 3. Dependencies, retained organization and geometry response

![Physical and neural lanes, actual heat responses and retained P2 source membership density](../../diagnostics/generated/dependency_correct_20261006/figures/03_dependencies.png)

Figure3a. Architecture-imposed input separation is labelled separately from learned thermal source membership density. All22Q1024 and fourQ8192 finite tests include balanced0.10/0.20 feasible transfers and an individual source change. Across230 changed inputs per arm, independent flow changes and prepared source/context differences are exactly0, as are same-input repeats. Enabled flow has real residual heat access: maximum pressure change0.00174028 and maximum 8%-band functional change0.000109427. The plotted membership is actual nominal packed-H5 0291 P2, five active donors/13 query groups, not counted-response provenance or physical causality. Dense values/global/coarse/local paths remain.

![Full saved geometry p changes and independent1000 residuals on two TRAIN layouts](../../diagnostics/generated/dependency_correct_20261006/figures/03_geometry_p.png)

Figure3 p. Already saved TRAIN0001/M3 and0348/M10, fixed heat, predeclared module0 +0.15 x move, complete joined grid/common-fluid mask. Across i±/j±, pressure correlations are0.967/0.961 with relative change-RMSE0.251/0.268; omega correlations0.811/0.596 with relative errors0.577/0.806. Geometry response is real but imperfect, especially vorticity. These are exposed saved references, not new solves or independent generalization evidence.

![Full saved geometry omega changes and independent1000 residuals on two TRAIN layouts](../../diagnostics/generated/dependency_correct_20261006/figures/03_geometry_omega.png)

Figure3 omega. Already saved TRAIN0001/M3 and0348/M10, fixed heat, predeclared module0 +0.15 x move, complete joined grid/common-fluid mask. Across i±/j±, pressure correlations are0.967/0.961 with relative change-RMSE0.251/0.268; omega correlations0.811/0.596 with relative errors0.577/0.806. Geometry response is real but imperfect, especially vorticity. These are exposed saved references, not new solves or independent generalization evidence.

Reverse-mode checks on the four representatives plus TRAIN M1/M12 give exactly zero native composed channel/8%-pressure heat VJPs for D-sep; its flow-only heat input is disconnected. D-open has nonzero finite sensitivities and VJPs. Full composed versus flow-only values agree exactly. Geometry, Re, inlet and receiver gradients are finite/nonzero. Best of the two declared finite-difference steps gives worst geometry relative gap0.197% independent/0.193% enabled. Tiny Re derivatives have FP32 finite-difference relative discrepancies up to about12.5% for the enabled reader; this is qualified numerical evidence, not a universal derivative pass. Packed inlet speed is constant1, so a nonzero inlet derivative is not measured broad inlet generalization.

### 4. Full application cost and fidelity versus work

![Complete sampled/full-grid inference, input VJP and extra CUDA allocation across actual low/high M](../../diagnostics/generated/dependency_correct_20261006/figures/04_cost.png)

Figure4a. Actual TRAIN M1/M12 at B8Q1024 and B1Q8192, matching tile512 and precision on GPU0. High-M independent Q8192 inference/VJP costs are0.134602/0.279851s, +4.84%/+3.39% over optimized retained. Extra peaks are140.748MiB inference and2051.840MiB VJP above the common resident-model/input baseline; raw allocated/reserved peaks are retained in JSON. These are allocator and synchronized complete-call measurements, not GPU busy-time or sparse executor savings.

![Native flow fidelity versus complete cold latency and peak extra allocation](../../diagnostics/generated/dependency_correct_20261006/figures/04_fidelity_work.png)

Figure4b. Full-grid DEV22 equal-case/channel normalized flow MSE is paired with actual high-M complete latency/memory, using common TRAIN channel scales. The new readers remain less accurate than mature retained flow. The comparison has different histories and cannot isolate organizer causality. FixedQ1024 selector MSE uses different receiver/quadrature sampling and is not substituted for this full-grid score.

Lazy H-add now computes S+R without allocating unused I; explicit diagnostics retain I and checkpoint keys. QM/QE reuse identical receiver access only inside the current read. Nonlinear receiver terms, masks, measures, attention and bias ordering remain. The bounded128/256/512 study selected512 with real outputs/VJPs/memory. Legacy128 high-M Q8192 measured0.397988s inference/0.968564s VJP versus same-operator optimized512 about0.127/0.270s in that tile study; matched G-fast was also tiled. All fine thermal source rows remain dense. New flow rows are counted separately as source-query tensor work, not savings in thermal execution.

Actual forward hooks on the selected reader count the dense MLP inputs below. These are executed tensor rows observed on CPU, not CUDA hardware counters; the two source-context updates are included. [Row receipt](../../diagnostics/generated/dependency_correct_20261006/report_preparation/actual_flow_rows.json).

| Condition | Source-query rows | Source-pair rows |
|---|---:|---:|
| B8/Q1024/M1 | 8192 | 16 |
| B8/Q1024/M12 | 98304 | 2304 |
| B1/Q8192/M1 | 8192 | 2 |
| B1/Q8192/M12 | 98304 | 288 |

At selected1000 high-M Q8192 the small flow prepare/read is about0.001020/0.004708s; this overlaps the complete wrapper and must not be added again. Whole geometry-input forward+VJP is0.267084s independent,0.257401s retained,0.215004s G-fast. Three fixed-geometry heat allocations cost0.405668s cold versus0.397029s reused, only2.13% observed savings because temperature computation remains per allocation. Enabled flow reuses no heat-conditioned states and shows no amortization. No exclusive-device speed claim is made for overlapped early measurements; final complete benchmarks ran after fit jobs stopped.

**Numerical qualification:** fields/material/ports and tested input VJPs pass the existing tolerances. Strict derived interface/q-proxy comparisons have misses around1.5–4.5e-5, including unchanged legacy128 against its own repeated call. The misses remain failures at the original tolerances; no all24 strict preservation pass or historical FP32 repair is claimed. Small nominal-temperature aggregate differences at about1e-8 are repeat effects, not thermal fitting. Source: `executor/replay_numeric.json` and tile-grid numerical checks.

### 5. Preserved thermal response, excitation and ports

![Unchanged Run3801 thermal response and residuals for both0291 heat directions](../../diagnostics/generated/dependency_correct_20261006/audit/thermal_0291_full_response.png)

Figure5a. Saved counted fixed-audit reference fluid-T means are -0.02891885/+0.02891378; unchanged Add500 gives +0.01042027/-0.01550809. Both signs remain wrong. Response residual RMS0.066168/0.079100. Full8192 original receivers are joined to their real128x64 coordinates; this is no new solve and no substitute0277 baseline.

![Saved thermal central closure, balanced excitation rank and bounded TRAIN port interventions](../../diagnostics/generated/dependency_correct_20261006/audit/thermal_linearity_rank_ports.png)

Figure5b. Saved reference fluid-T central closure RMS3.10e-7–1.58e-5 contrasts with retained0.000184–0.130814. Every complete family excites one balanced direction out of M-1; opposite endpoints do not add rank. Holding refined ports corrects one TRAIN0348 sign, leaving the other wrong. The24-call diagnostic ceiling includes eight calls before a padding-comparison failure and16 successful CPU calls; it was not extended.

The source supports an affine point-temperature update at fixed geometry/step count, qualified by heat-dependent convergence stopping and FP32 storage. Effective ports, clipping and nonlinear maxima need separate treatment. Nominal/typed/prepared calls on two TRAIN layouts agree exactly at common M12/P64/Q8192 physical inputs and receiver catalogues; no API catalogue patch was justified and no0291 explanation was manufactured. The unexecuted economical rank-completion proposal costs17 future attempts, estimated96.2 CPU seconds from old timings; it would require explicit new authorization beyond326/326. No heat-linear surrogate or new training portfolio was introduced.

## A/B/C and disposition

**A — unique organizer value:** unchanged weak historical evidence. Flow independence is imposed by a known source-backed input dependency; improved heat null is not a discovered hyperedge or proof of organizing advantage.

**B — actual information and work:** retained thermal has own-source heat → encoding → P0 → predicted ports → Stage-A → fused states → P1/P2, including global/coarse/local and dense source-value paths. Flow reads only its legal geometry/context/query inputs; measured finite/VJP independence is exact. Sparse work/executor savings were not measured because no sparse executor was introduced. Logical memberships and optional full-access fallback are separate historical experiments; no fallback supplies these candidate values.

**C — transfer and reliability:** exposed DEV22 v/p reconstruction is acceptable under the declared targets; u/omega fail. Saved two-TRAIN geometry changes show nontrivial but imperfect learned responses. Thermal0291 signs and rank limits remain. No valid-design diversity, calibrated false feasibility, broad geometry/inlet generalization or inverse readiness follows.

This round is complete with a qualified negative accuracy result. Existing Run3801/3802, G-fast, old Tensor-H, mature checkpoints and formal3501/3502 remain intact. Solver ledger stays326/326 with zero attempts. No Wind/formal training, inverse generator/search, physical retry, new thermal training or restart occurred. Only the two declared flow policies were fitted, and both stopped at1000.

## Appendices: complete roles, work, identity and validation

### All24 native roles

Means/p90 use equal-case native RMSE on the exact same22 cases. No role is renamed; the legacy inlet/outlet edge metric is distinct from the new8%-band dependency functional.

| Role | Retained mean / p90 | Independent mean / p90 | Enabled mean / p90 |
|---|---:|---:|---:|
| far/omega | 0.0535358 / 0.0651397 | 0.0763199 / 0.0919712 | 0.0755159 / 0.093154 |
| far/p | 0.0115006 / 0.0151563 | 0.0114417 / 0.0158637 | 0.0109815 / 0.0140822 |
| far/temperature | 0.767569 / 1.13071 | 0.767569 / 1.13071 | 0.767569 / 1.13071 |
| far/u | 0.0200417 / 0.0228665 | 0.0246186 / 0.029571 | 0.0254205 / 0.0312002 |
| far/v | 0.00279828 / 0.00355701 | 0.00247881 / 0.00318805 | 0.00237804 / 0.00313737 |
| final_port/h_effective | 0.828716 / 1.0807 | 0.828716 / 1.0807 | 0.828716 / 1.0807 |
| final_port/outside_temperature | 1.12741 / 1.51141 | 1.12741 / 1.51141 | 1.12741 / 1.51141 |
| fluid/omega | 0.104567 / 0.132608 | 0.177467 / 0.230934 | 0.178932 / 0.233454 |
| fluid/p | 0.0122219 / 0.0156063 | 0.0126933 / 0.0158911 | 0.0122821 / 0.014332 |
| fluid/temperature | 0.770312 / 1.10148 | 0.770312 / 1.10148 | 0.770312 / 1.10148 |
| fluid/u | 0.0220367 / 0.0249139 | 0.0282855 / 0.0330713 | 0.0290304 / 0.0343545 |
| fluid/v | 0.00307966 / 0.00399294 | 0.00314433 / 0.00401051 | 0.00310077 / 0.00407176 |
| initial_port/h_effective | 11.8709 / 11.9017 | 11.8709 / 11.9017 | 11.8709 / 11.9017 |
| initial_port/outside_temperature | 1.41298 / 2.23114 | 1.41298 / 2.23114 | 1.41298 / 2.23114 |
| inlet_outlet_pressure_difference | 0.00707586 / 0.0124374 | 0.00803227 / 0.013873 | 0.00802197 / 0.0132728 |
| material_temperature | 0.66308 / 1.13675 | 0.66308 / 1.13675 | 0.66308 / 1.13675 |
| module_material_peak | 0.647734 / 1.21301 | 0.647734 / 1.21301 | 0.647734 / 1.21301 |
| near/omega | 0.237676 / 0.287368 | 0.417907 / 0.460198 | 0.4236 / 0.463902 |
| near/p | 0.0154642 / 0.0187119 | 0.0172963 / 0.0188193 | 0.0171905 / 0.0185349 |
| near/temperature | 0.778342 / 1.16663 | 0.778342 / 1.16663 | 0.778342 / 1.16663 |
| near/u | 0.0304989 / 0.0354021 | 0.0440268 / 0.0506261 | 0.04471 / 0.0501091 |
| near/v | 0.00421953 / 0.00536157 | 0.00556956 / 0.00678105 | 0.00561473 / 0.00712917 |
| q_normal_proxy | 3.35197 / 4.11759 | 3.35197 / 4.11759 | 3.35197 / 4.11759 |
| surface_temperature | 0.788326 / 1.17822 | 0.788326 / 1.17822 | 0.788326 / 1.17822 |

All validation M strata are retained: six cases each at M3/M5/M7 and four at M10. These differ from actual TRAIN cost panels M1/M12.

| Stratum | Independent u / p / omega mean RMSE | Enabled u / p / omega mean RMSE |
|---|---:|---:|
| M3 | 0.0244712 / 0.0134032 / 0.129623 | 0.0245325 / 0.012641 / 0.130426 |
| M5 | 0.0264898 / 0.0114647 / 0.171684 | 0.0274989 / 0.011226 / 0.173059 |
| M7 | 0.029948 / 0.0128195 / 0.194735 | 0.0310036 / 0.0123682 / 0.196554 |
| M10 | 0.0342069 / 0.0132821 / 0.232007 | 0.0351145 / 0.0131988 / 0.234069 |

### All complete timing scopes

| Native condition | Model | Inference s | Heat/query forward+VJP s | Peak extra VJP MiB |
|---|---|---:|---:|---:|
| B8Q1024_low | G-fast | 0.0826171 | 0.176382 | 2055.29 |
| B8Q1024_low | H-add | 0.0918712 | 0.16833 | 2275.84 |
| B8Q1024_low | D-sep | 0.0943477 | 0.165303 | 2275.84 |
| B8Q1024_low | D-open | 0.094565 | 0.195208 | 2275.86 |
| B8Q1024_high | G-fast | 0.121231 | 0.200837 | 5463.49 |
| B8Q1024_high | H-add | 0.134131 | 0.231696 | 6301.70 |
| B8Q1024_high | D-sep | 0.138722 | 0.230611 | 6301.70 |
| B8Q1024_high | D-open | 0.138233 | 0.230606 | 6303.02 |
| B1Q8192_low | G-fast | 0.0893466 | 0.165276 | 1145.55 |
| B1Q8192_low | H-add | 0.116432 | 0.238807 | 1336.21 |
| B1Q8192_low | D-sep | 0.122663 | 0.247746 | 1336.21 |
| B1Q8192_low | D-open | 0.12246 | 0.244766 | 1336.21 |
| B1Q8192_high | G-fast | 0.0986506 | 0.194829 | 1686.86 |
| B1Q8192_high | H-add | 0.128393 | 0.270665 | 2044.82 |
| B1Q8192_high | D-sep | 0.134602 | 0.279851 | 2051.84 |
| B1Q8192_high | D-open | 0.134289 | 0.281921 | 2052.00 |

### Evidence and execution boundary

- [Guide](../guides/Thermal_Dependency_Flow.md), [source audit](HONF_Dependency_Correct_Source_Audit.md), and [figure index](../../diagnostics/generated/dependency_correct_20261006/figures/FIGURE_INDEX.md).
- [Initial/native optimizer evidence](../../diagnostics/generated/dependency_correct_20261006/native_optimizer_preflight.json), [thermal preservation](../../diagnostics/generated/dependency_correct_20261006/thermal_preservation_preflight.json), [paired streams](../../diagnostics/generated/dependency_correct_20261006/paired_query_stream_signatures.json), [review100](../../diagnostics/generated/dependency_correct_20261006/review100_decision.json), [extension decision](../../diagnostics/generated/dependency_correct_20261006/review500_decision.json).
- [Final scientific-state audit](../../diagnostics/generated/dependency_correct_20261006/final_flow_audit.json), [paired selected identity](../../diagnostics/generated/dependency_correct_20261006/paired_selected1000_audit.json). Final deterministic flow networks/sampler use no stochastic layers; explicit seed/case/epoch formulas reproduce resume, while inherited RNG remains in the preserved parent checkpoint.
- [Independent actual dependencies](../../diagnostics/generated/dependency_correct_20261006/evaluation/D-sep_selected_dependency/dependency.json), [enabled control](../../diagnostics/generated/dependency_correct_20261006/evaluation/D-open_selected_dependency/dependency.json), [complete costs and attempts](../../diagnostics/generated/dependency_correct_20261006/executor/executor_summary.json), [saved geometry](../../diagnostics/generated/dependency_correct_20261006/geometry_response/summary_epoch1000.json).
- [Source/port numerical evidence](../../diagnostics/generated/dependency_correct_20261006/audit/native_port_audit.json), [saved closure/rank](../../diagnostics/generated/dependency_correct_20261006/audit/saved_linearity_rank.json), [resource closeout](../../diagnostics/generated/dependency_correct_20261006/accounting/final_closeout.json).

Ordinary process/epoch timers record training and all timed benchmark attempts. Fit ledger is317.95 seconds before small initial checkpoint/plot tails;330 seconds are conservatively charged. Executor process ledger is0.054137 GPU-associated hours, including failures/superseded measures. Root dependency evaluators have independent process receipts; early field evaluators lack process-start receipts and are conservatively charged from round start to their saved-summary completion, explicitly as upper bounds rather than measured elapsed. Total measured/charged accounting and current process/GPU observations are in final_closeout.json. The12 GPU-associated-hour/8 elapsed-hour ceilings were not approached; at least45 minutes remained reserved for closeout throughout training.

Focused CPU regression covers reader/permutation/padding/heat restrictions, native composition/heat-batch guards, lazy-I/access equivalence and input/parameter gradients, response cost/preparation/evaluation, original refinement protections and development binding/training: **118 passed**. Ruff passes all rules on new durable files and fatal checks on changed legacy files; `git diff --check` passes. These software checks do not override scientific misses or q-proxy qualifications. Durable code/tests/guide/reports are committed/pushed only after the entire outgoing-history artifact audit and repository pre-push gate; generated arrays/figures/checkpoints and one-time renderers stay ignored locally. No credential or security policy was weakened.
