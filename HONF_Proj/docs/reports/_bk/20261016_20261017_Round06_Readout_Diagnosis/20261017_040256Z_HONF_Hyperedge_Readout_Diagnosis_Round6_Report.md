# Round6: frozen-bank hyperedge receiver readout diagnosis

This round adds a parameter-matched receiver that changes when the existing H anchor states are pooled: R-P pools before SiLU, and R-E applies SiLU to each edge before pooling. Both begin from the same H e2500 checkpoint, identical new reader tensors and common output heads, and fresh Adam state. H parameters and buffers that produce P, the moments and Z stay frozen; the existing native TrainingEngine, losses, deterministic sampling, local covers and physical source paths are preserved.

The four primary fits completed 500 epochs each; Wind earned and completed the common 1000-epoch extension. Neither local edge-first reader meets the predeclared signal: R-E/R-P priority is 0.999945 in Thermal and 1.001697 in Wind, with only 2/4 original TRAIN fixtures improving in each. Both independently initialized full-bank diagnoses also completed the same terminal ages; their literal results and gates are below. The local gates exclude joint confirmation, so there are zero joint updates. Extra frozen fitting improves several TRAIN errors against H, while Wind DEV near-vector error worsens and Thermal temperature tails remain uneven; it does not establish equal-age qualification against D0.

## Predictor: protocol and evidence

Thermal uses the fixed 150 selected TRAIN cases and exposed DEV22; Wind uses the fixed 72 TRAIN direction rows from 24 layouts and exposed DEV24 from eight layouts. These are the existing quarter-data development memberships. A development epoch visits every selected case or row. There is no full-source-TRAIN run, new split, TEST access, normalization refit, loss reweighting, query-budget change or new solver call.

The common primary fit is 500 real epochs. Thermal uses four macro updates per epoch, including its six-case final macro and scheduled TRAIN response; Wind uses five, including its eight-row final macro. Sampling retains absolute epoch 2500 + fit_age, effective/micro batches 48/24 and 16/8, Thermal Q1024 fluid with unchanged auxiliaries, and Wind Q8192 with the original 3:1 volume/hub draw. AdamW, constant LR, decay, clipping and seeds remain unchanged. The native engine's internal stage string `joint` names the original objective; the scientific phase is `frozen_probe`, with an audited parameter allowlist.

Execution uses the ModularDT environment with Python 3.12.13, Torch 2.6.0+cu124/CUDA 12.4 and NVIDIA RTX 6000 Ada Generation devices with nominal 48 GiB memory. Physical GPU1 (`GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e`) and GPU2 (`GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`) are exposed individually as logical cuda:0 after UUID verification. Native application/fitting remains FP32 under the inherited deterministic cuBLAS workspace `:4096:8`, TF32 off, deterministic algorithms on and cuDNN benchmark off; FP64 controls are separately labelled. The software stack, framework and native eager executor are unchanged. All comparable application-cost scopes use uncontended GPU1; CPU preparation and document/equation checks hide CUDA.

R-P and R-E add 25,536 parameters in Thermal and 25,984 in Wind. Including the common trainable output heads, the fit has 128,611 and 75,651 trainable parameters, respectively. Input maps are bias-free, hidden width is 64, and W_o begins at exactly zero. A_Z and A_q map 128 to 64, A_g maps 14 Thermal or 21 Wind signed geometry features to 64, W_o maps 64 to 128, and the only new bias has 64 entries. The geometry features, row mass, original c0 and four/eight local slots match within each pair. Trainable A_Z projections are freshly prepared from final H Z; no stale prepared graph or detached input-gradient substitute is used.

For live cover mass s=sum(w), u_a=A_Z Z_a+A_g psi(r_qa) and v_q=A_q q_state+b_r, the corrections are R-P: W_o SiLU(s v_q+sum(w u_a)) and R-E: W_o sum(w SiLU(v_q+u_a)). The original context c0 is added unchanged in both. Psi contains asinh(r), sin/cos(pi r), sin/cos(pi r/2), and sin/cos(pi r/4) for each signed coordinate. Thermal offsets use the explicit frame scale; Wind uses native query-minus-anchor coordinates in rotor diameters with no second rotation or normalization. Learned states and hidden messages have latent units, not physical source units.

Thermal RMSE retains packed-native dataset units, including pressure and temperature; no Kelvin calibration is introduced. The q result is the existing extraction-consistent proxy, not verified SI heat flux. Wind reports all five roles and all three components in m/s. Its vector RMSE is sqrt(mean(sum(component error squared))), with no divide by three. Correlated direction rows are grouped by layout for fitting distributions. The stored native Wind plane/cuts remain the same analytic-wake/shared-grid reference, without new independent CFD truth or target interpolation.

## Predictor: TRAIN gates and native outcomes

The fixed fitting audit covers all selected TRAIN cases or rows, using Thermal Q1024 plus the original sampled auxiliaries, and Wind Q1024 over the five roles with counts 205/205/205/205/204. It is distinct from the full-native DEV board and detailed TRAIN4; numbers are compared only on identical support. Original TRAIN4 IDs are Thermal 0006/0097/0342/0631 and Wind 456/273/536/533. The detailed DEV export panel is Thermal 0291/0687 and Wind 69/426. The compact field gallery shows input-fixed high-M TRAIN0631/533 and DEV0687/426, with 0291 in the saved-response figure; unplotted Wind 69 and all other declared detailed arrays remain retained, alongside every full-native DEV row.

Extension decisions use only the predeclared TRAIN score and complete native objective windows, before new DEV outcomes choose an action. Thermal stops at 500: priority500/300 is 0.979512 for R-P and 0.978269 for R-E; objective-median ratios are 1.009033 and 0.982020, so neither reaches the required 3% reduction. Wind earns the common 1000 horizon: priority ratios are 0.966175 and 0.968099; its objective-median ratios alone, 0.975927 and 0.973919, do not pass. Both complete horizons and final evidence are prospectively reserved.

Thermal's terminal local gate is negative. R-E/R-P priority is 0.999945, with v ratio 1.003641 and p ratio 0.996262, and only two of four detailed TRAIN fixtures improve. All nonpriority guards pass; R-E improves the same priority score against H to 0.863715 and its fixed native objective to 0.00303931 from H 0.00656124. The extra fitting exposure helps both heads/readers, but the required 10% paired edge-first advantage is absent.

Wind also misses the local signal at the admitted common 1000 fit age: R-E/R-P priority is 1.001697 (near Ux 0.994341, downstream Ux 1.009107), and only two TRAIN4 layouts improve. Every protected component/vector guard passes. R-E/H priority is 0.845835 and the same original fixed native objective falls to 0.0481810 from H 0.0632655. R-P/R-E have both completed 72,000 case-row visits and 5,000 optimizer updates; the common native case/query streams match at every fit epoch. The required order-of-reduction gain is absent, so neither dataset admits the optional joint confirmation. There are zero joint scientific updates.

The frozen gains do not close the mature-reference gap. On exactly the same low-budget all-TRAIN raw queries and targets, D0 e2500 has Thermal v/p means 0.00238055/0.00753900, compared with 0.00367578/0.00990126 for R-E at fit500; Wind D0 near/downstream Ux means 0.107811/0.0763185 m/s compare with 0.368119/0.285486 m/s for R-E at fit1000. These are different fitting exposures and remain historical context, without equal-age qualification. The exposed Thermal DEV22 board also shows uneven transfer: R-E500 surface/material mean errors improve against H, but their p90 errors increase from 1.69752/1.69893 to 1.74508/1.76125. TRAIN recovery is not uniform role/tail recovery or independent-test generalization.

Wind's exposed full-native DEV24 priority errors do not recover with the local fit. R-E1000 near-vector mean increases from H's 0.629116 to 0.660857 m/s, and downstream-vector mean increases from 0.502916 to 0.506066 m/s, while background-vector mean improves from 0.144862 to 0.117968 m/s. Near-vector p90 worsens from 0.685807 to 0.738911 m/s. These full-native roles and direction-row tails remain visible; the TRAIN gate, earlier best-volume monitor and background improvement do not turn this into a generalization pass.

## Literal terminal decisions

| Dataset | Fit age | R-E/R-P priority | TRAIN4 improve | R-E/H priority | R-G/R-E priority | R-G/H priority | Local useful | Full-bank useful |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thermal | 500 | 0.999945 | 2/4 | 0.863715 | 1.02379 | 0.884263 | False | False |
| wind | 1000 | 1.0017 | 2/4 | 0.845835 | 1.07247 | 0.907135 | False | False |

Priority ratios are geometric means of same-panel native mean-RMSE ratios: Thermal v/p, Wind near/downstream Ux. Local threshold is 0.90 with each priority at most 1.05, at least 3/4 TRAIN4 directions, parent priority at most 0.95 and no native objective increase. Full-bank priority must be at most 0.90 against both local R-E and H. Thermal protected temperature/q means allow 1.05 and u/curl1.10; Wind near/downstream vectors allow 1.05 and volume/background Ux/vector1.10. No threshold or tolerance is relaxed.

### Thermal local nonpriority guards

| Metric | R-E/R-P mean ratio | Limit | Pass |
| --- | --- | --- | --- |
| temperature_fluid | 1.01423 | 1.05 | True |
| temperature_surface | 1.02197 | 1.05 | True |
| temperature_material | 1.01411 | 1.05 | True |
| q_proxy | 1.00199 | 1.05 | True |
| u | 1.00453 | 1.1 | True |
| omega | 1.00215 | 1.1 | True |

### Thermal full-bank guards

| Comparator | Metric | R-G/reference mean ratio | Limit | Pass |
| --- | --- | --- | --- | --- |
| local_edge | temperature_fluid | 1.00117 | 1.05 | True |
| local_edge | temperature_surface | 0.990519 | 1.05 | True |
| local_edge | temperature_material | 0.99248 | 1.05 | True |
| local_edge | q_proxy | 1.01431 | 1.05 | True |
| local_edge | u | 1.00884 | 1.1 | True |
| local_edge | omega | 1.01984 | 1.1 | True |
| zero_fit_parent | temperature_fluid | 0.562535 | 1.05 | True |
| zero_fit_parent | temperature_surface | 0.370595 | 1.05 | True |
| zero_fit_parent | temperature_material | 0.352529 | 1.05 | True |
| zero_fit_parent | q_proxy | 0.694364 | 1.05 | True |
| zero_fit_parent | u | 0.730599 | 1.1 | True |
| zero_fit_parent | omega | 0.876172 | 1.1 | True |

Original fixed native objectives (thermal): H 0.00656124182, R-E 0.00303930615, R-G 0.00313466206. Both full-bank comparator decisions and each failed condition are preserved in the checksum-bound [local gate](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/thermal_terminal500_local_readout_signal_v1.json) and [full-bank gate](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/thermal_terminal500_full_bank_signal_v1.json).

### Wind local nonpriority guards

| Metric | R-E/R-P mean ratio | Limit | Pass |
| --- | --- | --- | --- |
| near_turbine/vector | 0.994379 | 1.05 | True |
| downstream_envelope/vector | 1.00908 | 1.05 | True |
| volume/Ux | 0.993933 | 1.1 | True |
| volume/vector | 0.994098 | 1.1 | True |
| background/Ux | 0.988485 | 1.1 | True |
| background/vector | 0.988536 | 1.1 | True |

### Wind full-bank guards

| Comparator | Metric | R-G/reference mean ratio | Limit | Pass |
| --- | --- | --- | --- | --- |
| local_edge | near_turbine/vector | 1.10846 | 1.05 | False |
| local_edge | downstream_envelope/vector | 1.03711 | 1.05 | True |
| local_edge | volume/Ux | 1.008 | 1.1 | True |
| local_edge | volume/vector | 1.0083 | 1.1 | True |
| local_edge | background/Ux | 0.93269 | 1.1 | True |
| local_edge | background/vector | 0.93341 | 1.1 | True |
| zero_fit_parent | near_turbine/vector | 0.941337 | 1.05 | True |
| zero_fit_parent | downstream_envelope/vector | 0.874362 | 1.05 | True |
| zero_fit_parent | volume/Ux | 0.827749 | 1.1 | True |
| zero_fit_parent | volume/vector | 0.828585 | 1.1 | True |
| zero_fit_parent | background/Ux | 0.745403 | 1.1 | True |
| zero_fit_parent | background/vector | 0.746685 | 1.1 | True |

Original fixed native objectives (wind): H 0.06326552, R-E 0.0481810156, R-G 0.0530164246. Both full-bank comparator decisions and each failed condition are preserved in the checksum-bound [local gate](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/wind_terminal1000_local_readout_signal_v1.json) and [full-bank gate](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/wind_terminal1000_full_bank_signal_v1.json).

## Organizer and readout: measured use and limits

All fixed-bank claims are confined to the frozen study. Any permitted H e1000 to e2500 joint confirmation would let P, moments and Z coadapt and diverge; it would hold the organization law and payload architecture fixed, not their numerical values. Frozen extra-age fits cannot qualify against an uncontinued equal-age D0 reference.

The new readers use the existing final H Z. P remains full-soft; cached positive associations and SVD ranks do not establish invertibility of processed Z, physical causality or executor savings. Plotted query cover edges describe the actual local readout. R-P plots use its pooled preactivation and correction; R-E plots use actual per-edge hidden messages and their weighted reduction. The original module/environment mixtures describe the base context; they do not sum to the new nonlinear correction or define its exact effective physical-source weights. The native primitive pooling-nullspace experiment holds one reader's weights fixed across reduction modes and changes projected A_Z packets. A separate CPU equation check closes the plan's original-Z requirement by changing the two 128-dimensional Z vectors before A_Z on all 16 already fixed local case-arm probes. It retains ordinary FP32 misses on 13/16 probes under the original tolerances; all strict FP64 controls pass, and a fixed nondegenerate original-Z fixture has zero pooled response and nonzero edgewise response. The saved native qterm/geometry are prescribed, not recomputed with a provider; this check is an equation reference, not a new native predictor run or physical scene. Neither latent perturbation establishes a physically realizable bank intervention or multi-body truth.

Typed module/environment interventions recompute dependent H states and reader projections and restore actual native outputs exactly wherever an exchange exists. The one-module Thermal fixture has no module exchange and is reported as unavailable. Same-weight changes establish use, not net training benefit. Their inherited JSON interpretation says “coadapted representation”; in this frozen experiment, that label refers to the fitted reader/common heads operating on the fixed H bank and must not imply that P, moments or Z changed during fitting. Residual removal was not measured in this round. Such a counterfactual would retain that arm's own fitted common heads and would not reproduce the independently fitted comparator.

## Conditional full-bank access diagnosis

Thermal's negative local gate permits R-G. It starts independently from the same H e2500, the same initial reader and head tensors as R-E, and fresh Adam; no fitted local or disposable resource-probe weights are reused. Its correction reads every valid cached anchor with the existing positive environment measure normalized over anchors. The local base c0 and original heat/source support remain unchanged. This is fixed-weight global integration, not learned sparse access.

The genuine native Thermal R-G resource epoch visited all 150 cases, all four updates, the final six-case macro and one scheduled response; it took 6.586 seconds, with a 1.020 GiB maximum allocated macro peak and unchanged 190-tensor H state. Its fit0 predictions and native objective exactly recover H on the full selected TRAIN audit; full-native DEV22 and original TRAIN4 rows/metrics also recover H exactly. The complete 500-epoch stage and final evidence were reserved before admission. All 500 epochs completed. Its fixed TRAIN G/R-E priority ratio is 1.023791 and G/H is 0.884263; all nonpriority guards pass against both comparators, and its native objective is 0.00313466 versus H 0.00656124. It misses the required 10% gain against equal-age R-E, so this broader weighted access does not produce a useful access signal in Thermal.

Wind R-G independently completes 1000 full selected-data epochs from the same H e2500 and identical initial reader/common heads, with fresh Adam and all 176 frozen H tensors unchanged. Its fixed TRAIN R-G/R-E priority contrast is 1.072472; R-G/H is 0.907135. The access gate is negative. The near-vector guard against R-E is 1.108457, exceeding its 1.05 limit; all other protected guards pass against both comparators. Its native objective is 0.0530164246, versus H 0.06326552 and equal-age R-E 0.0481810156. This changes both correction access and fixed integration weights; the original 1.5D local support and local base context remain. It admits no joint or deployment claim.

A useful R-G result requires at least a 10% all-TRAIN priority reduction against both equal-age local R-E and zero-fit H, all specified nonpriority guards, and no native objective deterioration versus H. A favorable result would demonstrate usefulness of this broader weighted access, not prove that support radius alone was responsible. A failed result would not prove no decoder can use the bank. There is no R-G joint training, distillation, new selector, source bypass or deployment claim.

## Native laws and inverse scope

Trained Thermal R-P/R-E pass precise FP64 source-linear heat application, exact zero-heating temperature, heat-independent flow, direct JVP, native-role transpose/adjoint and source-kernel controls on original TRAIN4. Raw factors are cast before contraction. The native first derivative and derivative of the squared first-gradient norm were measured on the fixed safe probe in case 0006 for each arm; other scenes' derivative probes are explicitly unmeasured. All cases retain the ordinary FP32 shifted-minus-base misses separately, without widened tolerances or replacement of historical evidence. Thermal R-G500 repeats the full four-case native laws and all latent primitive controls on the four original cases successfully; its coordinate/target/storage probe is confined to case 0006, and other cases are explicitly unmeasured for that first-case-only probe. Endpoint maxima are recomputed at each control endpoint.

One aligned saved 0291 baseline/minus/plus response family uses the original retained records and requests for both Thermal terminal local arms and the admitted R-G terminal. Its strict precise endpoint/increment laws pass, while generic FP32 cancellation misses and discrepancies against the saved reference remain visible. Missing 0277 baseline and compatible new four/eight-state geometry truth remain unavailable. No new response campaign, inverse optimizer, design search or physical solve is run. Inherited solver use remains 326/326; own new attempts are zero. This is not an audit of every unrelated host process.

## Executed work and cost

Complete latency starts from host inputs and includes native support packing/transfers, scene/query preparation, native extraction, synchronization and CPU return. Prepared-scene/new-query and Thermal prepared-fixed-query/new-heat timings are separate scopes. Paired comparisons use two warmups and seven alternating trials, with a second block only when the first bootstrap interval crosses a declared threshold. The 95% paired bootstrap intervals describe repeated timing trials on the fixed fixtures, not case/layout or model-population uncertainty. Raw trials, block directions, medians and p90 are retained; profiler traces are diagnostic, and nested/inclusive stage times are not summed. Execution retains dense full-soft H scene preparation and the existing indexed local gather. Four/eight local slots describe executed receiver work; no learned sparse scene-executor saving is established, and full-bank cost remains a separate diagnostic.

On high-M B1/Q8192 complete calls, R-E/R-P paired median ratios are 1.000404 for Thermal (95% paired bootstrap interval 0.980387–1.006747) and 1.003864 for Wind (0.995759–1.015728). Both cover parity. R-P/H ratios are 1.236970 and 1.263870, respectively. The readers add overhead relative to H; memory and frozen fitting exposure do not authorize equal-age deployment qualification. Thermal R-G/R-E complete high-M/Q8192 is 6.276127 (95% paired bootstrap interval 6.043324–6.406076), prepared-scene/new-query is 11.632958, and fixed-query/new-heat is 0.999495. Its B8/Q8192 allocated forward and backward peaks are 445,415,936 and 666,863,104 bytes, close to the local reader's measured peaks on this matrix; this bounded observation does not establish a scaling law. Its real native optimizer epoch takes 6.10951 s for four updates, with maximum macro allocated peak 1,090,423,808 bytes; this is not a latency win. Full native H and trained R-E profiler traces include the real moment/base/local/query/readout/extraction/host-transfer stages, and instrumented outputs match profiler-off outputs exactly.

Wind R-G/R-E complete high-M/B1/Q8192 is 3.026351 (95% paired repeat interval 2.995835–3.057426); prepared-scene/new-query is 5.734147. These correctly oriented direct measurements show substantial additional cost alongside the negative full-bank accuracy/guard gate.

Forward/extraction B8 matrices at Q1024/4096/8192 run in their own process; target-free loss backward runs separately. A full disposable native optimizer epoch includes every macro, response and final partial update, and saves no optimized scientific weights. The H frozen cost scope trains only common output heads; R-P/R-E also train the identically sized reader. H's disposable Adam is fresh; R-P/R-E use their own retained named endpoint moments. These measurements do not establish equal-age model qualification. Prepared projection bytes, unique retained autograd storage, one-hidden-tensor structural tile bounds and allocator peaks are reported separately.

A serialization defect in the initial local cost helper preserved summary medians and ratios but replaced deeply nested absolute timing samples with placeholders. Those receipts remain unchanged as summary-only evidence. A serialization-only helper revision retains and validates every raw trial; only the two affected Thermal latency comparisons were repeated. The selected export uses this complete evidence, without choosing a run for speed. The first Thermal full-bank timing call also supplied the comparator first to the paired reducer, so its valid raw ratios were R-E/R-G despite the filename. That receipt remains unchanged; one bounded latency-only replay supplies R-G first and exports the intended ratio. Memory, projection and optimizer measurements are unaffected. Other memory, optimizer and counter/projection raw records were preserved by their direct JSON writers.

## Completed exposure and checkpoint selection

| Dataset | Arm | Epochs | Case/row visits | Updates | Microbatches | Engine train seconds |
| --- | --- | --- | --- | --- | --- | --- |
| thermal | R-P | 500 | 75000 | 2000 | 3500 | 572.993 |
| thermal | R-E | 500 | 75000 | 2000 | 3500 | 571.534 |
| thermal | R-G | 500 | 75000 | 2000 | 3500 | 2905.89 |
| wind | R-P | 1000 | 72000 | 5000 | 9000 | 1262.26 |
| wind | R-E | 1000 | 72000 | 5000 | 9000 | 1264.93 |
| wind | R-G | 1000 | 72000 | 5000 | 9000 | 5102.46 |

The complete science totals are 441,000 case/row visits and 21,000 updates. Known native engineering exposure is 115 updates, including failed/replayed pilot steps; conservative reservations are 299/400. Associated GPU-process time is 3.8496 hours, below 20; elapsed time at accounting is 4.3352 hours, below 14. These process seconds include setup/evaluation/failures and are distinct from engine train seconds. All three arms match native case/query hashes at every scientific fit epoch. Full history, monitoring cadence, latest/best aliases and literal 100-epoch milestones are retained; only milestones/final states receive frozen-state hash audits, not every query.

| Dataset | Arm | Literal terminal age | Best ordinary field-monitor age | Literal terminal SHA256 |
| --- | --- | --- | --- | --- |
| thermal | R-P | 500 | 500 | c5d259c4bdf94ab47c614d130a387c98e7cbd3327cb30dab29bccf248671369c |
| thermal | R-E | 500 | 500 | d6bf42d00ca9483cef6970be4326ce7a4990e5814fba63c09d955ccea7e8e154 |
| thermal | R-G | 500 | 500 | 108dda78f1225654905fb72f3f807da65d4fe9e13f908c766ae8e68a8b5df010 |
| wind | R-P | 1000 | 100 | c3bdf82a702788213569516d592c9b8928df0e8252ce9556762b700134b2ee98 |
| wind | R-E | 1000 | 100 | 392dd95976ee96023f686267be34e612bee2131261b146138b56a366714f2eb9 |
| wind | R-G | 1000 | 100 | 58df76ceb32efa0080bb5197249d3248ebd722e98f58a502ab97b5766dbaa3e2 |

Primary gates, full native boards, physical exports and costs use the literal common terminal checkpoint. Best-field aliases use the unchanged ordinary DEV monitor over saved 100-epoch milestones; each alias exactly matches its own milestone weights and experiment identity. They are retained and identified, without substituting a favorable selected checkpoint for the declared endpoint. Separate full-native boards for earlier best aliases are unmeasured in this frozen study.

| Dataset | Arm | Primary fluid/native queries | Primary decoder rows | Response increment rows | Nonlinear width64 entries |
| --- | --- | --- | --- | --- | --- |
| thermal | R-P | 76800000 | 325009728 | 2242294 | 20800622592 |
| thermal | R-E | 76800000 | 325009728 | 2242294 | 83202490368 |
| thermal | R-G | 76800000 | 325009728 | 2242294 | 3993719537664 |
| wind | R-P | 589824000 | 589824000 | 0 | 37748736000 |
| wind | R-E | 589824000 | 589824000 | 0 | 301989888000 |
| wind | R-G | 589824000 | 589824000 | 0 | 2415919104000 |

These are shape-derived forward execution charges, including repeated native receiver unions and explicit response work. They are not independent physical sample counts, GPU instruction totals, backward-recompute totals or memory. A zero structural autograd-activation byte counter does not mean zero saved graph storage. The scoped tensor hooks and complete allocator matrices below provide the memory evidence.

## Tests, source review and reproducibility

The review began at aec07be and preserved the user's existing worktree changes. The durable implementation uses the existing engine and strict family/checkpoint dispatch. The affected CPU suite passed 225 tests with zero skips; dense/indexed/tiled value/gradient checks, gradcheck/gradgradcheck, masks/duplicates/row mass, zero-reader H recovery, native first/subsequent reader gradients, stale-state invalidation, target isolation and exact save/resume were checked. Native disposable qualification used the unchanged full objectives. Configuration, CPU preparation and startup are identified separately from completed scientific fits.

The scientific snapshot is immutable, with loaded-source fingerprint `2ce7161bb4acd03f981a5bfa92c7543832ea727ae236c3166fc4edae06bb3501` and 382 bound runtime files. This inventory, rather than an inherited parent's Git stamp, identifies the executed code. The durable implementation was committed/pushed as `13b552e5eb18fd774f0247a06a1f84fca1f6d1ef`, and the unchanged plan as `2aa9771bb191a67af0bce4f1ca931d97997b6c7c`; final report/upload tips are verified separately at closeout. The Round6 plan is retained byte-for-byte; its SHA256 is `06e35a1921ee832e6c6c99b3efb608d385bea34d0c2ff47c3fcd30161eb210a6`. Parent checkpoint SHA256s, recipes, exact query/target/mask bindings, initial reader/common-state digests, named optimizer policy, failures and command/resource receipts remain in the ignored evidence root. No historical weights or reports are rewritten.

## Numerical misses and evidence repairs

Wind's terminal local native diagnosis uses all four original Q8192 TRAIN fixtures and reports `completed_with_fp32_control_misses`: strict latent primitive controls miss on seven of eight case-arm combinations (R-P row 536 passes). This is distinct from native Wind-law failures: every native law, counter/trace output equality and available typed-intervention restoration passes. The separate row 456 FP64 arithmetic companion passes the strict weighted-cancellation/identity controls for each trained local reader. It casts Z and reader weights before A_Z projection and reconstructs the controlled perturbation in FP64; query states are prescribed FP32 states cast to FP64. It is an equation reference, not a changed native predictor, fit repair or proof that all other scenes' FP32 controls pass.

Wind R-G1000 retains a separate `completed_with_fp32_control_misses` result: all 20 native Wind law checks pass, and 7/8 ordinary projected-packet controls pass. The sole miss is the same-weight identity check on row 273; weighted packet means pass 4/4. Broader access does not eliminate native FP32 cancellation.

Trained coordinate checks are first input derivatives and the derivative of the first-gradient squared norm at eight fixed safe original queries. Adapter-side query features stay at their prescribed values. These probes establish finite nonzero native higher-derivative execution, not full-Hessian accuracy against a physical reference. The separate zero-reader native qualification checks match parent first/second derivatives exactly. No shared-gradient comparison is promoted into a new adaptive loss, and no optional contraction reassociation, compiler backend or framework upgrade is used.

Failure records remain explicit. Early native disposable qualification, exact-resume comparisons and CPU import/schema preflights include failed attempts; all known native engineering steps and GPU-associated process time are charged. A numeric CUDA visibility attempt is rejected by the UUID guard before CUDA initialization. Wind native-board v1 fails a frozen normalization-state check; v2 binds the original H source epoch 2500. A full-bank board wrapper's Thermal-only 22-row assertion is corrected to accept Wind's fixed24-row panel after CPU verification of its existing measured result. A full-bank detail wrapper compares the project root with the frozen tree root and fails after valid arrays are written; its one-line v4 metadata correction independently validates and reuses the exact result/arrays. No scientific fit is restarted, relabelled or initialized from these diagnostic outputs.

The initial cost serializer loses deeply nested raw absolute latency trials while preserving summaries; only the affected Thermal local pairs are replayed with a strict complete serializer. The first Thermal full-bank ratio uses the inverse CLI arm order despite its filename; it is retained and a latency-only replay supplies the intended numerator. Other memory/stage/optimizer measurements are reused. CPU report-audit helpers also encounter schema/path/type mismatches before output; their corrections do not alter numerical reductions or scientific sources. A protected-cache auditor reuses the last geometry's manifest for three earlier entries and reports30 false mismatches; corrected v2 and the independent 40-array root check pass exactly, with no cache repair or mutation. These are evidence-tool failures, kept separate from native fitting or physical accuracy.

Target locks and physical-reference limits remain: Wind source TEST is locked; DEV is exposed development validation; Thermal native units and q proxy have no new SI calibration; Wind uses the retained analytic-wake/shared-grid reference, not new CFD. The missing 0277 baseline and compatible new four/eight-state physical interaction truth remain unavailable. No inference bias correction, source reconstruction, pseudoinverse, source bypass, new organizer, inverse design or solve is introduced.

## Storage, cleanup and next steps

New outputs live in /data2/wanglz/ModularDT/diagnostics/hyperedge_readout_round6_20261016T233604Z, linked from the working directory. The mount UUID is verified before work. The round has a 12 GiB persistent-byte budget, bounded temporary storage and a 20 GiB minimum free-space floor; it reuses the verified shared Wind catalogue and creates no duplicate dataset/cache store. GPU1/2 are authorized by UUID; GPU0 and its unrelated job are protected.

Audited cleanup removed 889 expired/superseded files (54,133,458 logical bytes), 136 internal temporary symlinks and 867 temporary directories. The exact recursive allowlist and repeated process/link/dependency checks are preserved in the [cleanup receipt](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/round6_cleanup_receipt_v3.json). Its 2,538 retained-file comparisons pass unchanged, including all native checkpoints/milestones/latest/best, source/scripts, selected numerical/physical/readout/cost evidence and historical pins. Every allowlisted payload is absent. Root-owned privileged-service and inaccessible other-user process descriptors remain outside open-handle coverage; all accessible candidate handles/mappings are clear and the authorized GPU lanes are idle. No global no-open-handles or active GPU0-weight claim is made.

Earlier receipts separately account for six relocation duplicates (53,634 bytes) and one disposable bytecode file (22,311 bytes), giving 896 regular files and 54,209,403 logical bytes removed over the complete round, without double-counting final cleanup. At final retained-inventory readback, 2,532 regular files occupy 1,125,951,286 logical bytes in the data2 evidence root, below the 12 GiB budget; this includes compact audit/cleanup receipts. All selected four PDF masters and 13 PNG companions remain. Post-cleanup free capacity is 3,386,068,496,384 bytes on data2, 310,385,467,392 on data, and 169,517,559,808 on the shared workspace/tmp filesystem; all exceed the 20 GiB floor. Free-space deltas include unrelated host activity and need not equal deleted logical bytes.

All 13 selected figure pages were inspected; four representative PDF pages were also visually checked after independent system rendering. Source-format/link checks cover the complete report, four PDF masters and 13 embedded PNGs. The [render validation](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/report_figure_render_validation_v1.json) records system rendering and HTML structure, without a browser-preview claim; the [main visual receipt](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/main_selected_figure_visual_inspection_v1.json) records the inspected images. Generated figures and one-time helpers remain local and ignored.

## Separate outcomes and next decisions

Predictor gain: both local adapters improve selected-TRAIN errors and native objective relative to unchanged H after additional fitting. Predictor miss: neither local edge-first arm delivers the predeclared paired gain; frozen extra exposure does not establish equal-age qualification against D0. Next predictor work should use this negative matched result when choosing a separately specified hypothesis, with the same native losses and mature physical-role boards; no further fit or full-TRAIN probe is admitted by this round.

Organizer gain: exact numerical bank equality and measured intervention restoration make the reduction-order comparison interpretable. Organizer miss: P rank, nonzero messages and same-weight intervention effects do not establish physical causal organization, Z invertibility, losslessness, sparse execution savings or net training benefit. Next organizer work requires its own controlled plan; no new organizer or membership rule is selected here.

Inverse gain: the trained Thermal readout preserves precise source-linear heat application and the bounded saved-response replay is inspectable. Inverse miss: these software laws and sampled responses do not validate new inverse designs or remedy unavailable physical references. Next inverse work requires separately authorized, physically referenced evidence; this round adds zero designs, inverse optimizer updates or solves and preserves the inherited 326/326 ledger.

## Selected figure index

Four PDF masters contain13 selected pages; small PNG companions are embedded below. Figures use only the checksum-bound numerical index. Generated masters, companions and one-time renderers stay local in the ignored data2 evidence root.

- [1. Fitting and Wind bias — PDF](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/01_fitting.pdf)
- [2. Native physical fields and residuals — PDF](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/02_physical.pdf)
- [3. Associations, readout and controls — PDF](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/03_readout.pdf)
- [4. Native cost and scoped memory — PDF](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/04_cost.pdf)

![Thermal and Wind completed fitting curves, fixed TRAIN priorities and paired case/layout contrasts](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/01_fitting_01.png)

Completed Thermal 500/Wind 1000 fits use identical case/query streams. R-E/R-P priority is 0.999945/1.001697; Thermal/Wind R-G/R-E is 1.023791/1.072472. Both local contrasts miss0.90 and 2/4 detailed TRAIN directions improve. Distributions are descriptive; the gate uses ratios of declared means.

![Wind five-role signed Ux bias and per-row MSE decomposition on fixed TRAIN inputs](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/01_fitting_02.png)

Wind uses 72 fixed TRAIN rows from 24 layouts and five original roles. Signed Ux bias and per-row MSE=bias²+variance remain distinct from arithmetic mean RMSE; correlated query cells are not independent scenes. No inferred bias correction is applied.

![Thermal0631 TRAIN native T/v/p fields, signed residuals, module temperature maxima and q proxy](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/02_physical_01.png)

Thermal 0631 is an original high-M TRAIN fixture. Native fluid-T RMSE is 0.549598 for H and 0.386517 for R-E500; the panel retains v/p residuals, actual material endpoint maxima and q-proxy errors. All models use the same masks, coordinates and row-wise physical/error scales.

![Thermal0687 exposed DEV native T/u/curl fields, signed residuals, module peaks and q proxy](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/02_physical_02.png)

Thermal 0687 is exposed DEV. Its native fluid-T RMSE is 0.480124 for H and 0.36298 for R-E500; u/curl residuals and material/q roles remain visible. This single illustration does not override the full DEV22 mean/tail table or the TRAIN decision.

![Wind533 TRAIN native Ux/Uy/Uz plane predictions and signed physical residuals](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/02_physical_03.png)

Wind 533 is the original high-M TRAIN row on native z-index16. Ux plane RMSE is 0.344668 m/s for H, 0.287559 for R-E1000 and 0.319957 for R-G1000. Uy/Uz signed residuals use the same stored coordinates/targets. The reference remains analytic-wake/shared-grid evidence, not independent CFD.

![Wind426 exposed DEV original wake deficit cuts and three-component signed residual distributions](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/02_physical_04.png)

Wind 426 is exposed DEV. Three original downstream cut locations show profile-minus-Ux depth, native width, peak displacement and recovery; their numerical values are tabulated below. Signed plane residual distributions include all three components. No target interpolation, favorable plane selection or cell-based independent-layout confidence is used.

![Thermal0631 actual association spectra, local/full-bank reads, source kernel and original-Z controls](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/03_readout_01.png)

Thermal 0631 has 192 anchors and 12 modules: actual P has four heads and Z has 128 channels. Eight input-fixed local probes use four cover slots; the R-G overlay shows every valid anchor for one correction probe while local c0 remains. The signed source kernel uses eight separate native receiver probes. Original-Z equation controls are separate from native projected-packet controls and retain precision misses.

![Wind533 actual association spectra, frozen Z, local/full-bank reads and original-Z controls](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/03_readout_02.png)

Wind 533 has 64 anchors and 30 physical modules: actual P has four heads, the reader has 64 hidden channels and Z has 128. Eight local probes retain eight slots; R-G reads every valid anchor with fixed normalized measure. The x/y projection overlaps distinct z anchors. Neither spectra nor hidden messages prove invertibility, physical influence or a useful fitted readout signal.

![Native typed module/environment association-exchange effects across all original TRAIN4 fixtures](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/03_readout_03.png)

Every available head0 module/environment exchange restores actual native outputs bitwise across four TRAIN cases per dataset and all three terminal arms. Thermal 0006 has one module, so its module exchange is unavailable. The plotted same-weight effects establish computational use; fixed H banks did not coadapt during training, and these exchanges are not physical counterfactual truth.

![Thermal0291 original saved signed response replay with FP32/FP64 arithmetic and reference discrepancies](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/03_readout_04.png)

One original 0291 saved baseline/minus/plus family is replayed on all three Thermal terminal arms. Strict precise endpoint/increment laws pass across three temperature roles and two signed controls per arm. Ordinary FP32 cancellation misses and reference-response discrepancies remain. This is zero new inverse updates/designs/solves; endpoint maxima are recomputed.

![Matched native high-M complete, prepared-query and prepared-heating latency trials](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/04_cost_01.png)

GPU1 complete high-M/B1/Q8192 R-E/R-P ratios are 1.000404 for Thermal and 1.003864 for Wind, with repeat intervals covering parity. Every retained raw trial and threshold-triggered repeat is shown. R-G ratios use correctly oriented direct paired receipts; absolute H/R-P/R-E/R-G values come from their own recorded comparisons, without multiplying independent ratios.

![Separately measured native B8 forward/backward memory and real complete optimizer-epoch costs](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/04_cost_02.png)

Fixed B8 matrices use Q1024/4096/8192; forward/extraction and target-free native backward run in separate processes. Real native optimizer epochs include150 Thermal cases/four updates and 72 Wind rows/five updates, including final partial macros and scheduled Thermal response. Allocator baselines/reserved peaks are tabulated separately; disposable optimizer weights are discarded.

![Fixed TRAIN fidelity versus measured complete-call time and memory with projection/counter overhead](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/figures/04_cost_03.png)

Fidelity is the declared fixed all-TRAIN priority divided by H fit0, at explicit additional 500/1000 fit ages. Cost axes use measured complete high-M/Q8192 wall medians and separate B8/Q8192 allocated peaks. AZ projection is 49,152 bytes for Thermal and 16,384 for Wind; preparation and counter overhead are measured separately. These plots do not establish equal-age deployment qualification.

## Complete native numerical controls

### Terminal TRAIN4 controls

This CPU-only summary covers the six terminal R-P, R-E, and R-G arms on the fixed Thermal and Wind TRAIN4 panels; native-law gates and A_Z-packet FP32 primitive controls are counted separately.

| Dataset | Arm | Endpoint checkpoint | Strict native laws | Module association available/restored | Environment association available/restored | A_Z-packet FP32 primitive controls | Wind role-ID changes |
|---|---|---|---|---|---|---|---|
| thermal | R-P | epoch_0500_model.pt (c5d259c4bdf9…) | 32/32 pass; 0 misses (8/case, 4/4 cases) | 3/4 available; 3/3 restored | 4/4 available; 4/4 restored | 16/16 pass; 0 misses | not reported for Thermal |
| thermal | R-E | epoch_0500_model.pt (d6bf42d00ca9…) | 32/32 pass; 0 misses (8/case, 4/4 cases) | 3/4 available; 3/3 restored | 4/4 available; 4/4 restored | 16/16 pass; 0 misses | not reported for Thermal |
| thermal | R-G | epoch_0500_model.pt (108dda78f122…) | 32/32 pass; 0 misses (8/case, 4/4 cases) | 3/4 available; 3/3 restored | 4/4 available; 4/4 restored | 8/8 pass; 0 misses | not reported for Thermal |
| wind | R-P | epoch_1000_model.pt (c3bdf82a7027…) | 20/20 pass; 0 misses (5/case, 4/4 cases) | 4/4 available; 4/4 restored | 4/4 available; 4/4 restored | 11/16 pass; 5 misses (456:pool+base+pert, 273:pert, 533:pert) | 0 changed across 8 operations × 8,192 role IDs |
| wind | R-E | epoch_1000_model.pt (392dd95976ee…) | 20/20 pass; 0 misses (5/case, 4/4 cases) | 4/4 available; 4/4 restored | 4/4 available; 4/4 restored | 6/16 pass; 10 misses (456:pool+edge+base+pert, 273:base+pert, 536:base+pert, 533:base+pert) | 0 changed across 8 operations × 8,192 role IDs |
| wind | R-G | epoch_1000_model.pt (58df76ceb32e…) | 20/20 pass; 0 misses (5/case, 4/4 cases) | 4/4 available; 4/4 restored | 4/4 available; 4/4 restored | 7/8 pass; 1 miss (273:same-weight) | 0 changed across 8 operations × 8,192 role IDs |

Wind R-G specifically passes all strict native-law gates: cases 456, 273, 536, and 533 each pass 5/5, for 20/20 and zero misses. Its ordinary FP32 A_Z-packet controls pass 7/8: same-weight identity passes 3/4 and misses case 273; the weighted projected packet mean passes 4/4. The single primitive miss is retained in the diagnosis status and is not a native-law failure. In the miss cells, pool/edge mean pool-first/edge-first identity controls, base/pert mean same-weight reduction at base/perturbed input, and same-weight/weighted-mean denote the two full-bank controls.

Wind local R-P has 11/16 primitive controls pass and 5 misses (case 456: pool-first and same-weight base/perturbed; case 273: same-weight perturbed; case 533: same-weight perturbed). Wind local R-E has 6/16 pass and 10 misses (case 456: pool-first, edge-first, same-weight base/perturbed; cases 273, 533, and 536: same-weight base/perturbed). The Thermal local arms pass 16/16 controls and Thermal R-G passes 8/8.

The pooled preactivation and pooled SiLU checks pass where recorded; the intended edge-first trained-SiLU response is an effect measurement, not an invariance criterion. No tolerance was widened.

For first-case target-poison, coordinate, and storage details, the representative is the first preregistered TRAIN4 identity (Thermal 0006; Wind 456). The poison probe compares synthetic all-zero and NaN/-999 target sentinels with targets absent from both inference requests; coordinate probes evaluate the trained native output at eight original safe coordinates with cover/query IDs fixed, adapter features held prescribed, and model parameters frozen, reporting the first derivative and the derivative of the first-gradient squared norm rather than full-Hessian accuracy.

Saved-tensor hooks cover only HyperedgeReceiverReadout.local_correction for R-P/R-E or HyperedgeReceiverReadout.full_bank_correction for R-G; A_Z preparation, query encoding, and native output extraction are outside that scope. The table reports total saved storage including parameter/resident-input storage, the classified excluded portion, and retained receiver autograd storage; retained equals total minus excluded. Structural work counters are not measured storage.

| Dataset | Arm | First case | Target poison | Coordinate derivative (8 queries) | Scoped saved storage: total / excluded / retained (bytes; storage counts) |
|---|---|---|---|---|---|
| thermal | R-P | 0006 | pass; bitwise equal; targets absent; 0 solvers | d1 L2/max 0.6841/0.2494; d(||grad||²) L2/max 0.2146/0.1053 | 6,878,216/43 / 4,748,296/10 / 2,129,920/33 |
| thermal | R-E | 0006 | pass; bitwise equal; targets absent; 0 solvers | d1 L2/max 0.6963/0.2511; d(||grad||²) L2/max 0.2258/0.1085 | 6,878,216/43 / 4,748,296/10 / 2,129,920/33 |
| thermal | R-G | 0006 | pass; bitwise equal; targets absent; 0 solvers | d1 L2/max 0.7238/0.2571; d(||grad||²) L2/max 0.2433/0.113 | 6,477,768/139 / 4,347,080/8 / 2,130,688/131 |
| wind | R-P | 456 | pass; bitwise equal; targets absent; 0 solvers | d1 L2/max 8.254/7.87; d(||grad||²) L2/max 796.1/795.9 | 7,272,460/43 / 5,142,540/10 / 2,129,920/33 |
| wind | R-E | 456 | pass; bitwise equal; targets absent; 0 solvers | d1 L2/max 7.933/7.57; d(||grad||²) L2/max 916.8/916.3 | 7,272,460/43 / 5,142,540/10 / 2,129,920/33 |
| wind | R-G | 456 | pass; bitwise equal; targets absent; 0 solvers | d1 L2/max 9.022/8.631; d(||grad||²) L2/max 1926/1925 | 6,478,156/139 / 4,347,980/8 / 2,130,176/131 |

Coordinate and storage probes are reported only for the representative first case in each arm; no corresponding claim is made for the other three cases. Exact values, gate names, per-check FP32 outcomes, checkpoint pins, JSON/NPZ digests, and storage scopes are in the adjacent machine-readable receipt.

Machine-readable receipt: receipts/report_native_controls_numeric_tables_v1.json.

## Complete fixed-panel field and cost tables

Thermal values retain original packed-native units; q is the existing proxy. Wind component/vector values are m/s. Fixed TRAIN p90 is over 150 case errors or 24 layout means; full-native DEV p90 retains the original 22 case/24 direction-row reducer and is not interchangeable with a layout-bootstrap interval. Every field, component and role is retained, including worse tails.

### Thermal fixed all-TRAIN audit

D0 remains its original all-parameter e2500 checkpoint on exactly the same raw queries/targets with its own normalization. It is historical context: the readout arms have additional 500/1000 frozen fitting epochs, so this is not equal-age qualification.

| Field (mean / p90 of cases) | D0 e2500 (context) | H | R-P | R-E | R-G |
| --- | --- | --- | --- | --- | --- |
| temperature_fluid | 0.466759 / 0.700911 | 0.491786 / 0.743376 | 0.272446 / 0.355966 | 0.276323 / 0.358938 | 0.276647 / 0.357831 |
| u | 0.0133841 / 0.015992 | 0.0227147 / 0.0259927 | 0.0163756 / 0.0199562 | 0.0164499 / 0.0199729 | 0.0165953 / 0.0202361 |
| v | 0.00238055 / 0.0031973 | 0.00397778 / 0.00536816 | 0.00366245 / 0.00518196 | 0.00367578 / 0.0051696 | 0.00374718 / 0.00523982 |
| p | 0.007539 / 0.00919171 | 0.0122648 / 0.0167016 | 0.00993842 / 0.0139257 | 0.00990126 / 0.0139381 | 0.0101803 / 0.0142996 |
| omega | 0.0460925 / 0.0583014 | 0.0703544 / 0.0850966 | 0.0603141 / 0.0752414 | 0.0604436 / 0.0753336 | 0.0616425 / 0.076758 |
| temperature_surface | 0.317909 / 0.501365 | 0.527806 / 0.810155 | 0.193229 / 0.272609 | 0.197475 / 0.283145 | 0.195602 / 0.277748 |
| temperature_material | 0.351375 / 0.550602 | 0.612976 / 0.935641 | 0.214701 / 0.277769 | 0.217729 / 0.287521 | 0.216092 / 0.280451 |
| q_proxy | 0.900303 / 1.30262 | 0.921047 / 1.30125 | 0.629272 / 0.998863 | 0.630522 / 0.996305 | 0.639542 / 1.01911 |

### Thermal full native exposed DEV

| Field (mean / p90 of cases/rows) | H | R-P | R-E | R-G |
| --- | --- | --- | --- | --- |
| temperature_fluid | 0.789884 / 1.23782 | 0.712796 / 1.19453 | 0.707876 / 1.17655 | 0.714352 / 1.19474 |
| u | 0.0271127 / 0.0326225 | 0.0205233 / 0.0248792 | 0.0208959 / 0.0246961 | 0.0208573 / 0.0248773 |
| v | 0.00406619 / 0.00490805 | 0.0039293 / 0.00465681 | 0.0039556 / 0.00471092 | 0.00403755 / 0.00474827 |
| p | 0.0149746 / 0.020711 | 0.0133613 / 0.018641 | 0.013428 / 0.0185661 | 0.0136245 / 0.0178524 |
| omega | 0.060704 / 0.0728879 | 0.0546246 / 0.065631 | 0.0544037 / 0.0653649 | 0.0565358 / 0.0678024 |
| temperature_surface | 0.999882 / 1.69752 | 0.905996 / 1.75337 | 0.898565 / 1.74508 | 0.909131 / 1.71704 |
| temperature_material | 1.03928 / 1.69893 | 0.91892 / 1.77166 | 0.911074 / 1.76125 | 0.919529 / 1.73127 |
| q_proxy | 2.50447 / 3.82821 | 2.45465 / 3.52316 | 2.44986 / 3.52626 | 2.46273 / 3.53973 |
| module_peak_endpoint | 1.0616 / 1.97872 | 0.977002 / 1.8404 | 0.969668 / 1.82027 | 0.981133 / 1.79864 |

### Thermal complete native paired latency

| Fixture | R-P / H | R-E / R-P | R-G / R-E |
| --- | --- | --- | --- |
| low_m_Q1024 | 1.07378 [1.06235, 1.0839] | 0.99419 [0.982241, 0.998405] | 2.10183 [2.08205, 2.11836] |
| low_m_Q8192 | 1.27034 [1.25326, 1.278] | 1.00875 [1.00243, 1.01964] | 6.51488 [6.36583, 6.60237] |
| high_m_Q1024 | 1.10604 [1.09814, 1.11406] | 0.997936 [0.983174, 1.0029] | 2.99582 [2.91141, 3.02693] |
| high_m_Q8192 | 1.23697 [1.23495, 1.24158] | 1.0004 [0.980387, 1.00675] | 6.27613 [6.04332, 6.40608] |

### Thermal high-M Q8192 profiler-off latency (median / p90 ms)

| Scope | H | R-P | R-E | R-G |
| --- | --- | --- | --- | --- |
| complete | 49.5133 / 49.7252 | 61.3766 / 61.4764 | 61.2733 / 62.1767 | 378.086 / 384.368 |
| prepared_scene_new_query | 18.2484 / 18.3587 | 29.5141 / 29.5811 | 30.2574 / 30.3551 | 351.098 / 352.365 |
| prepared_fixed_query_new_heat | 4.36495 / 4.37199 | 4.40648 / 4.48289 | 4.42145 / 4.51325 | 4.20199 / 4.42327 |

### Thermal native B8 peak allocated bytes

| Scope | Q | H | R-P | R-E | R-G |
| --- | --- | --- | --- | --- | --- |
| forward | 1024 | 211586560 | 212082176 | 212082176 | 212082176 |
| forward | 4096 | 308808192 | 309303808 | 309303808 | 309303808 |
| forward | 8192 | 444920320 | 445415936 | 445415936 | 445415936 |
| backward | 1024 | 278776832 | 310274048 | 310274048 | 310280192 |
| backward | 4096 | 415180288 | 458261504 | 458261504 | 458267648 |
| backward | 8192 | 597177856 | 666856960 | 666856960 | 666863104 |

### Thermal native B8/Q8192 allocator baselines and peaks (bytes)

| Scope | Arm | Baseline allocated | Baseline reserved | Peak allocated | Peak reserved |
| --- | --- | --- | --- | --- | --- |
| forward | H | 40297984 | 41943040 | 444920320 | 549453824 |
| forward | R-P | 40400384 | 46137344 | 445415936 | 551550976 |
| forward | R-E | 40400384 | 46137344 | 445415936 | 551550976 |
| forward | R-G | 40400384 | 46137344 | 445415936 | 574619648 |
| backward | H | 74580480 | 75497472 | 597177856 | 717225984 |
| backward | R-P | 74682880 | 79691776 | 666856960 | 832569344 |
| backward | R-E | 74682880 | 79691776 | 666856960 | 832569344 |
| backward | R-G | 74682880 | 79691776 | 666863104 | 813694976 |

### Thermal complete disposable native optimizer epoch

| Arm | Cases | Updates | Final macro cases | Seconds | Peak allocated bytes | Peak reserved bytes |
| --- | --- | --- | --- | --- | --- | --- |
| H | 150 | 4 | 6 | 1.34067 | 1056861184 | 1249902592 |
| R-P | 150 | 4 | 6 | 1.51155 | 1090165248 | 1583349760 |
| R-E | 150 | 4 | 6 | 1.50608 | 1090928128 | 1583349760 |
| R-G | 150 | 4 | 6 | 6.10951 | 1090423808 | 1577058304 |

### Thermal new projection and exact-counter overhead

| Arm | AZ shape | AZ bytes | CUDA event median ms | Host median ms | Complete counters on/off |
| --- | --- | --- | --- | --- | --- |
| R-P | [1, 192, 64] | 49152 | 0.124928 | 0.142971 | 1.03259 |
| R-E | [1, 192, 64] | 49152 | 0.124928 | 0.14258 | 1.02956 |
| R-G | [1, 192, 64] | 49152 | 0.118688 | 0.135366 | 0.9932 |

### Wind fixed all-TRAIN audit

D0 remains its original all-parameter e2500 checkpoint on exactly the same raw queries/targets with its own normalization. It is historical context: the readout arms have additional 500/1000 frozen fitting epochs, so this is not equal-age qualification.

| Field (mean / p90 of layout means) | D0 e2500 (context) | H | R-P | R-E | R-G |
| --- | --- | --- | --- | --- | --- |
| volume/Ux | 0.0367367 / 0.0411924 | 0.153425 / 0.175549 | 0.126758 / 0.153765 | 0.125989 / 0.144368 | 0.126998 / 0.150896 |
| volume/Uy | 0.00299703 / 0.00351612 | 0.0109557 / 0.0143891 | 0.00983607 / 0.0125353 | 0.00989389 / 0.0124244 | 0.010101 / 0.0132254 |
| volume/Uz | 0.0020437 / 0.00251967 | 0.00852018 / 0.0105892 | 0.00728426 / 0.00877121 | 0.00738415 / 0.00905485 | 0.00787864 / 0.00969034 |
| volume/vector | 0.0369256 / 0.0414349 | 0.154097 / 0.176402 | 0.127383 / 0.154571 | 0.126631 / 0.14519 | 0.127683 / 0.151608 |
| hub_slab/Ux | 0.0572698 / 0.0645127 | 0.236033 / 0.275899 | 0.196032 / 0.229663 | 0.198389 / 0.233092 | 0.220888 / 0.264535 |
| hub_slab/Uy | 0.00481689 / 0.00566154 | 0.0198389 / 0.0245558 | 0.0171886 / 0.021248 | 0.0172712 / 0.0214689 | 0.0177603 / 0.0221025 |
| hub_slab/Uz | 0.00317689 / 0.00436904 | 0.0119017 / 0.0137033 | 0.00984678 / 0.0117772 | 0.0098864 / 0.0116427 | 0.0109943 / 0.0128865 |
| hub_slab/vector | 0.0575756 / 0.064911 | 0.237195 / 0.277337 | 0.197055 / 0.230757 | 0.199409 / 0.234478 | 0.221901 / 0.265846 |
| near_turbine/Ux | 0.107811 / 0.120345 | 0.433642 / 0.470396 | 0.370214 / 0.406333 | 0.368119 / 0.399355 | 0.408311 / 0.44145 |
| near_turbine/Uy | 0.0107987 / 0.0127307 | 0.0360921 / 0.039495 | 0.0319691 / 0.0357139 | 0.0318671 / 0.0354483 | 0.0327166 / 0.0358882 |
| near_turbine/Uz | 0.0078435 / 0.00907605 | 0.0237709 / 0.0261839 | 0.0208752 / 0.0233226 | 0.0208934 / 0.0232369 | 0.0223375 / 0.0247615 |
| near_turbine/vector | 0.108652 / 0.121107 | 0.435817 / 0.472552 | 0.372202 / 0.4084 | 0.37011 / 0.401449 | 0.410251 / 0.443426 |
| downstream_envelope/Ux | 0.0763185 / 0.0841948 | 0.338744 / 0.384787 | 0.28291 / 0.325589 | 0.285486 / 0.32376 | 0.296043 / 0.337094 |
| downstream_envelope/Uy | 0.00650525 / 0.00757238 | 0.0270865 / 0.0303107 | 0.0236998 / 0.0270016 | 0.0237815 / 0.0272629 | 0.0245219 / 0.0274893 |
| downstream_envelope/Uz | 0.00450913 / 0.00533416 | 0.0200472 / 0.0216317 | 0.0170891 / 0.0184747 | 0.0172637 / 0.0186916 | 0.0187266 / 0.0200838 |
| downstream_envelope/vector | 0.07674 / 0.0846979 | 0.340446 / 0.38683 | 0.284439 / 0.327173 | 0.287021 / 0.325605 | 0.297673 / 0.339045 |
| background/Ux | 0.0258146 / 0.0297375 | 0.103329 / 0.123732 | 0.083542 / 0.0950053 | 0.0825801 / 0.0947198 | 0.0770216 / 0.0845509 |
| background/Uy | 0.00173935 / 0.00205087 | 0.00592542 / 0.00754806 | 0.00536903 / 0.0062783 | 0.00535252 / 0.00647034 | 0.00539478 / 0.00675557 |
| background/Uz | 0.00146563 / 0.00183535 | 0.00516473 / 0.00594897 | 0.00482963 / 0.00581633 | 0.00479493 / 0.00567047 | 0.00491701 / 0.00588311 |
| background/vector | 0.0259202 / 0.0298112 | 0.103672 / 0.124013 | 0.0838945 / 0.0953524 | 0.0829328 / 0.0950737 | 0.0774103 / 0.0849431 |

### Wind full native exposed DEV

| Field (mean / p90 of cases/rows) | H | R-P | R-E | R-G |
| --- | --- | --- | --- | --- |
| volume/Ux | 0.234083 / 0.264927 | 0.225116 / 0.267685 | 0.223198 / 0.25921 | 0.216162 / 0.251926 |
| volume/Uy | 0.0251571 / 0.0273913 | 0.0260076 / 0.0297348 | 0.0261091 / 0.0297328 | 0.0256971 / 0.0292833 |
| volume/Uz | 0.0153071 / 0.018066 | 0.0153745 / 0.0184048 | 0.0154124 / 0.0181089 | 0.0152256 / 0.0179303 |
| volume/vector | 0.235967 / 0.266923 | 0.227168 / 0.270037 | 0.22528 / 0.261812 | 0.218254 / 0.25462 |
| hub_slab/Ux | 0.413524 / 0.468536 | 0.429648 / 0.502053 | 0.427884 / 0.506344 | 0.41777 / 0.485764 |
| hub_slab/Uy | 0.0448208 / 0.0522525 | 0.0462524 / 0.0550912 | 0.0462632 / 0.0553944 | 0.0457629 / 0.053545 |
| hub_slab/Uz | 0.0205373 / 0.0234751 | 0.020721 / 0.0243421 | 0.0207755 / 0.0245494 | 0.0205404 / 0.0238889 |
| hub_slab/vector | 0.416505 / 0.471354 | 0.432668 / 0.505055 | 0.430921 / 0.509735 | 0.420814 / 0.489254 |
| near_turbine/Ux | 0.622367 / 0.678093 | 0.659068 / 0.721673 | 0.654332 / 0.731244 | 0.631511 / 0.681223 |
| near_turbine/Uy | 0.0811185 / 0.0904114 | 0.0816607 / 0.0907209 | 0.0815771 / 0.0899301 | 0.0812901 / 0.0897861 |
| near_turbine/Uz | 0.042298 / 0.0495246 | 0.0428113 / 0.0500974 | 0.0429814 / 0.0501405 | 0.0425586 / 0.0499481 |
| near_turbine/vector | 0.629116 / 0.685807 | 0.665549 / 0.728621 | 0.660857 / 0.738911 | 0.638206 / 0.68734 |
| downstream_envelope/Ux | 0.498726 / 0.552987 | 0.508721 / 0.558496 | 0.501643 / 0.5695 | 0.487879 / 0.540164 |
| downstream_envelope/Uy | 0.0563894 / 0.0629209 | 0.0584503 / 0.0656621 | 0.0584466 / 0.0661342 | 0.0576435 / 0.0639738 |
| downstream_envelope/Uz | 0.0310224 / 0.0358977 | 0.0316208 / 0.0369082 | 0.0314411 / 0.0369563 | 0.0309318 / 0.0363438 |
| downstream_envelope/vector | 0.502916 / 0.557335 | 0.513092 / 0.56288 | 0.506066 / 0.574778 | 0.492297 / 0.543981 |
| background/Ux | 0.143778 / 0.161862 | 0.117063 / 0.129844 | 0.116601 / 0.129437 | 0.116368 / 0.131269 |
| background/Uy | 0.0139937 / 0.0158083 | 0.014217 / 0.0162708 | 0.0143038 / 0.0165441 | 0.0140329 / 0.0161861 |
| background/Uz | 0.0104906 / 0.0122162 | 0.0104121 / 0.0119192 | 0.0104591 / 0.0119436 | 0.0104994 / 0.0118499 |
| background/vector | 0.144862 / 0.162785 | 0.118413 / 0.131057 | 0.117968 / 0.130656 | 0.11771 / 0.13246 |

### Wind complete native paired latency

| Fixture | R-P / H | R-E / R-P | R-G / R-E |
| --- | --- | --- | --- |
| low_m_Q1024 | 1.06487 [1.06316, 1.06996] | 0.998118 [0.993413, 1.01501] | 1.34477 [1.33629, 1.34932] |
| low_m_Q8192 | 1.29629 [1.29201, 1.30105] | 0.998982 [0.988267, 1.00527] | 3.20175 [3.14972, 3.2163] |
| high_m_Q1024 | 1.05891 [1.05077, 1.06639] | 1.00355 [0.991882, 1.02633] | 1.33984 [1.33777, 1.34789] |
| high_m_Q8192 | 1.26387 [1.26098, 1.26639] | 1.00386 [0.995759, 1.01573] | 3.02635 [2.99584, 3.05743] |

### Wind high-M Q8192 profiler-off latency (median / p90 ms)

| Scope | H | R-P | R-E | R-G |
| --- | --- | --- | --- | --- |
| complete | 40.6759 / 40.7719 | 51.3783 / 51.7998 | 52.6353 / 54.7897 | 158.743 / 161.219 |
| prepared_scene_new_query | 11.8282 / 12.3069 | 22.1455 / 23.2018 | 22.3148 / 22.5169 | 127.631 / 130.266 |

### Wind native B8 peak allocated bytes

| Scope | Q | H | R-P | R-E | R-G |
| --- | --- | --- | --- | --- | --- |
| forward | 1024 | 128243712 | 128479232 | 128479232 | 128479232 |
| forward | 4096 | 359793664 | 360029184 | 360029184 | 360029184 |
| forward | 8192 | 669746688 | 669982208 | 669982208 | 669982208 |
| backward | 1024 | 128243712 | 137154560 | 152883200 | 131613696 |
| backward | 4096 | 393348096 | 393583616 | 393583616 | 393583616 |
| backward | 8192 | 703301120 | 703536640 | 703536640 | 703536640 |

### Wind native B8/Q8192 allocator baselines and peaks (bytes)

| Scope | Arm | Baseline allocated | Baseline reserved | Peak allocated | Peak reserved |
| --- | --- | --- | --- | --- | --- |
| forward | H | 40082944 | 41943040 | 669746688 | 761266176 |
| forward | R-P | 40187392 | 44040192 | 669982208 | 761266176 |
| forward | R-E | 40187392 | 44040192 | 669982208 | 761266176 |
| forward | R-G | 40187392 | 44040192 | 669982208 | 788529152 |
| backward | H | 73637376 | 75497472 | 703301120 | 794820608 |
| backward | R-P | 73741824 | 77594624 | 703536640 | 811597824 |
| backward | R-E | 73741824 | 77594624 | 703536640 | 811597824 |
| backward | R-G | 73741824 | 77594624 | 703536640 | 822083584 |

### Wind complete disposable native optimizer epoch

| Arm | Cases | Updates | Final macro cases | Seconds | Peak allocated bytes | Peak reserved bytes |
| --- | --- | --- | --- | --- | --- | --- |
| H | 72 | 5 | 8 | 1.3636 | 718117376 | 773849088 |
| R-P | 72 | 5 | 8 | 1.77916 | 718928896 | 790626304 |
| R-E | 72 | 5 | 8 | 1.77911 | 718928896 | 790626304 |
| R-G | 72 | 5 | 8 | 5.39336 | 719059456 | 801112064 |

### Wind new projection and exact-counter overhead

| Arm | AZ shape | AZ bytes | CUDA event median ms | Host median ms | Complete counters on/off |
| --- | --- | --- | --- | --- | --- |
| R-P | [1, 64, 64] | 16384 | 0.124672 | 0.142089 | 1.03547 |
| R-E | [1, 64, 64] | 16384 | 0.124 | 0.141638 | 1.04138 |
| R-G | [1, 64, 64] | 16384 | 0.123808 | 0.141628 | 1.01742 |

### Original native Wind deficit cuts

The values use stored crosswind cuts without target interpolation or smoothing. Depth is max(Uprofile,x−Ux); peak y is the original native point attaining it. Half-depth span is the y range of native points at least half the positive maximum, potentially spanning multiple wake lobes; it is not an interpolated single-wake FWHM. These three original downstream offsets expose recovery as well as depth/width/displacement.

| Arm | Actual downstream offset D | Depth m/s | Peak y/D | Half-depth native span D | Points above half-depth | Native points |
| --- | --- | --- | --- | --- | --- | --- |
| Reference | 1.94156 | 2.60133 | -5.61944 | 5.74369 | 16 | 285 |
| Reference | 4.00341 | 2.58514 | 7.66988 | 0.788349 | 8 | 285 |
| Reference | 7.98966 | 2.72795 | 0.124251 | 11.3748 | 16 | 285 |
| H | 1.94156 | 0.7732 | -5.50682 | 0.563107 | 6 | 285 |
| H | 4.00341 | 1.09802 | 3.39027 | 4.84272 | 14 | 285 |
| H | 7.98966 | 2.0001 | -0.100992 | 6.8699 | 16 | 285 |
| R-P | 1.94156 | 0.717554 | -5.61944 | 6.08155 | 16 | 285 |
| R-P | 4.00341 | 1.20578 | 3.39027 | 8.10874 | 15 | 285 |
| R-P | 7.98966 | 1.71987 | -0.100992 | 9.00971 | 20 | 285 |
| R-E | 1.94156 | 0.674733 | -5.61944 | 5.74369 | 21 | 285 |
| R-E | 4.00341 | 1.32968 | 3.27765 | 9.46019 | 16 | 285 |
| R-E | 7.98966 | 1.83418 | -0.100992 | 6.75728 | 15 | 285 |
| R-G | 1.94156 | 0.699565 | -5.61944 | 0.675728 | 7 | 285 |
| R-G | 4.00341 | 1.3177 | 3.39027 | 4.7301 | 13 | 285 |
| R-G | 7.98966 | 1.82499 | -5.9573 | 6.8699 | 15 | 285 |

### Setup measured outside application calls

| Dataset | Arm | Checkpoint/provider load + H2D seconds | Low-M native fixture host pack seconds | High-M host pack seconds |
| --- | --- | --- | --- | --- |
| thermal | H | 3.70534 | 0.0119569 | 0.0104634 |
| thermal | R-P | 2.99536 | 0.012034 | 0.011068 |
| thermal | R-E | 2.93535 | 0.0122955 | 0.0110842 |
| thermal | R-G | 2.95403 | 0.0123443 | 0.011109 |
| wind | H | 0.539925 | 0.00618956 | 0.0121123 |
| wind | R-P | 0.290523 | 0.00637142 | 0.0119705 |
| wind | R-E | 0.29015 | 0.00635755 | 0.0119052 |
| wind | R-G | 0.296434 | 0.00670389 | 0.0125434 |

One process separately observes numerical/module imports 0.901256 s, native runtime setup 0.103499 s and post-authority-check CUDA-context initialization 0.0941328 s. Models load serially in H/R-P/R-E/R-G order, Thermal then Wind, with no forward or optimizer call. Driver probes may already execute during authority verification; these sections do not sum to whole-process startup. OS/shared-cache state is not controlled cold; these single setup observations establish neither startup-speed ratios nor application latency. Wrapper-associated wall time separately includes process startup before the first script instruction.


### Original TRAIN4 mixing-rank sensitivity

This is raw, unnormalized head0 P. The complete retained A2 summaries include all four heads, unique cover-accessible rows, repeated rows, Lambda-weighted rows, and raw/row-L2/column-L2/row-then-column-L2 scalings. Relative ranks use singular-value cutoffs 0.01, 0.0001 and 0.000001 times sigma1; backward-error ranks use max(matrix shape) times machine epsilon times sigma1. Repeating rows changes that numerical cutoff. These are linear module-mixture diagnostics, with no inference reconstruction or claim that processed Z is invertible.

|Dataset|TRAIN ID|P view|Rows x M|r(.01)|r(.0001)|r(.000001)|r(FP32)|r(FP64)|
|---|---|---|---|---|---|---|---|---|
|thermal|0006|bank|192x1|1|1|1|1|1|
|thermal|0006|Lambda-row stack|32768x1|1|1|1|1|1|
|thermal|0097|bank|192x4|4|4|4|4|4|
|thermal|0097|Lambda-row stack|32768x4|4|4|4|4|4|
|thermal|0342|bank|192x7|7|7|7|7|7|
|thermal|0342|Lambda-row stack|32768x7|7|7|7|7|7|
|thermal|0631|bank|192x12|12|12|12|12|12|
|thermal|0631|Lambda-row stack|32768x12|12|12|12|12|12|
|wind|456|bank|64x6|6|6|6|6|6|
|wind|456|Lambda-row stack|65536x6|6|6|6|6|6|
|wind|273|bank|64x14|14|14|14|14|14|
|wind|273|Lambda-row stack|65536x14|14|14|14|14|14|
|wind|536|bank|64x22|21|22|22|22|22|
|wind|536|Lambda-row stack|65536x22|21|22|22|22|22|
|wind|533|bank|64x30|29|30|30|30|30|
|wind|533|Lambda-row stack|65536x30|28|30|30|29|30|

### Original H native Wind local support

Counts and Ux RMSE below use each original Q8192 TRAIN4 draw, prior to fitting. Positive 1.5D reads are input-defined and identical for the frozen comparison; near, downstream and background roles can have different sampling density. Errors are per scene and support group in m/s. No overlapping local reads occur on these four original fixtures. This does not show that increasing the support radius is the remedy.

|TRAIN row|Role|N|Zero reads|One read|Overlap|Mean support mass|Ux RMSE outside|Ux RMSE inside|
|---|---|---|---|---|---|---|---|---|
|456|downstream_envelope|1640|1322|318|0|0.0155076|0.230823|0.371447|
|456|near_turbine|1640|81|1559|0|0.120322|0.320796|0.354738|
|456|background|1632|1625|7|0|0.000205882|0.100978|0.182697|
|273|downstream_envelope|1640|1411|229|0|0.0118842|0.360594|0.426495|
|273|near_turbine|1640|73|1567|0|0.110645|0.373129|0.458681|
|273|background|1632|1626|6|0|0.000140704|0.129055|0.120039|
|536|downstream_envelope|1640|1163|477|0|0.0221983|0.338206|0.442868|
|536|near_turbine|1640|52|1588|0|0.11066|0.536396|0.450129|
|536|background|1632|1627|5|0|0.000103742|0.133012|0.382372|
|533|downstream_envelope|1640|1169|471|0|0.0230835|0.354656|0.490429|
|533|near_turbine|1640|65|1575|0|0.10175|0.5316|0.502482|
|533|background|1632|1618|14|0|0.000389893|0.105033|0.241412|

### Original H downstream offsets

Each receiver is assigned once to its nearest physical source by native input geometry and binned by its signed downstream offset; the native downstream-envelope role can still overlap other source influences. Lateral and vertical columns are mean absolute input offsets in rotor diameters. These are descriptive H baseline groups, not training masks, independent samples or a target-derived correction.

|TRAIN row|Signed downstream bin|N|Mean abs lateral D|Mean abs vertical D|Ux RMSE m/s|
|---|---|---|---|---|---|
|456|upstream|311|1.313|0.5451|0.231199|
|456|downstream_0_1p5D|454|1.137|0.5531|0.342419|
|456|downstream_1p5_4D|399|0.9633|0.572|0.270771|
|456|downstream_4_8D|345|0.8912|0.5254|0.196545|
|456|downstream_ge_8D|131|0.6258|0.5307|0.119614|
|273|upstream|429|1.767|0.572|0.335997|
|273|downstream_0_1p5D|455|1.42|0.5585|0.373575|
|273|downstream_1p5_4D|509|1.477|0.5707|0.431855|
|273|downstream_4_8D|186|1.556|0.569|0.295955|
|273|downstream_ge_8D|61|0.6701|0.5428|0.179906|
|536|upstream|526|1.053|0.5586|0.375487|
|536|downstream_0_1p5D|563|1.137|0.5631|0.407786|
|536|downstream_1p5_4D|362|1.595|0.5933|0.373288|
|536|downstream_4_8D|151|1.783|0.5659|0.225585|
|536|downstream_ge_8D|38|1.008|0.45|0.133192|
|533|upstream|514|1.309|0.5753|0.390515|
|533|downstream_0_1p5D|561|1.266|0.5558|0.442953|
|533|downstream_1p5_4D|313|1.267|0.5747|0.436963|
|533|downstream_4_8D|205|2.038|0.5606|0.22796|
|533|downstream_ge_8D|47|1.526|0.6024|0.175545|


### Fixed native TRAIN4 Z packet variation

These descriptive per-scene summaries use all 8192 original native queries. For Lambda mass s, let mu=sum(wZ)/s, within=sum(w*norm(Z-mu)^2), mean_term=s*norm(mu)^2, and lost_share=within/(within+mean_term). The fraction measures a latent second moment omitted by one weighted mean; it is not physical information loss or a lower bound on prediction error. Nearby-query distances use original coordinates (Thermal native, Wind rotor diameters); queries are correlated within a scene.

| Dataset | TRAIN ID | Q | Mean within variance | Mean lost share | p90 lost share | Nearby mean-Z difference median | Nearby difference p90 | Nearest-query distance median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thermal | 0006 | 8192 | 0.724767 | 0.00911756 | 0.0186831 | 0.121807 | 0.348513 | 0.09375 |
| thermal | 0097 | 8192 | 1.55962 | 0.0270209 | 0.0839984 | 0.211749 | 0.656396 | 0.09375 |
| thermal | 0342 | 8192 | 2.35049 | 0.0415485 | 0.118438 | 0.276063 | 0.757636 | 0.09375 |
| thermal | 0631 | 8192 | 2.52917 | 0.0465747 | 0.125235 | 0.316984 | 0.825041 | 0.09375 |
| wind | 456 | 8192 | 225.372 | 0.277707 | 0.529841 | 1.33051 | 3.2048 | 0.302529 |
| wind | 273 | 8192 | 186.988 | 0.283762 | 0.559321 | 1.1088 | 3.01727 | 0.439269 |
| wind | 536 | 8192 | 203.861 | 0.289914 | 0.560688 | 1.18775 | 3.02835 | 0.412586 |
| wind | 533 | 8192 | 176.506 | 0.275019 | 0.52292 | 1.14481 | 3.12249 | 0.455883 |

## Exact lineage and immutable identities

Thermal starts with reader digest `75b6b0c88db3f032b71079b0d34c481a0062b39f103802ec923d74a6694189e3` and common-state digest `cd5133a32a4a4e5322ab5b235db4beb01ca3fd533c9b5c266bf8e94864b392e7`; fresh optimizer-state count is 0. The exact allowed names/shapes and optimizer membership are in the [parameter inventory](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/scientific_parameter_inventory_v1.json). The frozen digest is `e4d33b8688a426d0af7e56df22d34847ea128a6b1f38ac53681c70c10e8e3b3c`, over 190 parent parameter/buffer tensors.

Wind starts with reader digest `acb310b6e0713b0993c49b46b28a4aa74468743b7637b8322d40994a22ba5677` and common-state digest `029624a3e26dec9aa350ae55aee70da26b65fb3d8cabf123faf5905d29cc530e`; fresh optimizer-state count is 0. The exact allowed names/shapes and optimizer membership are in the [parameter inventory](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/scientific_parameter_inventory_v1.json). The frozen digest is `f96da25babb59ff7d23ef9ecee99502d04b60e2cb27fca70c4f7155141837aba`, over 176 parent parameter/buffer tensors.

| Dataset | H e2500 SHA256 | Subset manifest SHA256 | Selected TRAIN membership SHA256 |
| --- | --- | --- | --- |
| thermal | 95ac5d1377b3a3389713cc11dade3dbec80c397689a9732afbb15279b705b17b | 0978564b91ce0fda09ed93ddec0d912ce6b2ab6cc95bd9971bf702af9825d385 | 78344cd0d59f9103b2480472314e038a54a4354ffa833b5616511e6fe3b5f5f6 |
| wind | de9e921120291959bc9e64aaa165f8b363fbf8a9ad1ea67712cf987e4c10baa2 | b223e174d7262682a7e1ac1977cb38f721993e14c0340b2dc0f5fb5f95bfa715 | f0e12ef8350ab11757c7893ce6c1ab439ce13950c9d85da5f684dcc5c42e69c1 |

Thermal normalization statistics remain `67d9f4ec37c4d79ffaae589ea2bd6b20397245711c43648161bb8c7e95d3b0a6` on profile `fixed25_v1`; Wind retains profile `fixed24_v1` and its frozen H normalization epoch 2500. The all-TRAIN arrays are Thermal `93b1cbf18bd038e75576c38b5143a87fb0311f0d3fc47352c918efd204acd42f` and Wind `71912be4b8b40fe0ac1aede38f9d86808264156550a4ce67603f99c2ebbe179c`. Their ID/target/mask/query bindings were declared before the first fit, with the corrected data2 mount locator recorded separately.

The R5 H e1000 parents are retained unchanged for the unadmitted joint option: Thermal `166e26e487368588b6186141eb187ea49db3657db2a917d5f8fda9c552435335`, Wind `7d48d6487e0576e112600ff20d980c8225c2cc8bcaa9cf538792feda9b163e77`. No joint model is initialized or trained. H/D0 historical source/runtime qualifications remain in the complete [Round5 report](../20261016_Round05_Moment_Hyperedges/20261016_220823Z_HONF_Scene_Prepared_Moment_Hyperedges_Round5_Report.md); the byte-identical [Round6 work plan](../../../../UpgradePlan/ThemalChannel/20261016_20261017_Round06_Readout_Diagnosis/20261017_004938Z_HONF_Round6_Hyperedge_Readout_Diagnosis_Work_Plan.md) supplies the experiment gates.

The release CPU suite command used CUDA/NVIDIA visibility empty and the ModularDT interpreter: `python -B -m pytest` over `test_inherited_attention*.py`, `test_scene_prepared_moments.py`, `test_moment_hyperedge_runtime.py` and `test_hyperedge_readout*.py`, with its recorded explicit `--basetemp`. It passed 225 tests/zero skips in 7.93 seconds. Focused earlier groups overlap this total and are not added again. Targeted new/changed-code Ruff passes; six broader preexisting diagnostics in four unchanged core files and the inherited engine's B010 remain disclosed in the [release receipt](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/main_release_cpu_verification_v1.json).

Reproducible source, phase-aware loaders and exercised CLI behavior are documented in the [receiver guide](../../../guides/_bk/20261029_Guide_Consolidation/02_Attention_and_Unified_Routing_Lineages/Hyperedge_Receiver_Readout.md). The [selected numerical index](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/final_selected_evidence_index_v1.json) binds literal checkpoints, full-native boards, all fixed audits, all detailed exports, raw paired trials, memory and equation controls. The [actual exposure audit](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/final_actual_exposure_audit_v1.json) binds every completed epoch and known native engineering step; [protected-input audit](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/protected_evidence_audit_v2.json) and [independent 40-array cache recheck](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/root_used_shared_cache_exact_array_recheck_v1.json) retain the historical/source/cache limitations.

The final [protected-input recheck](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/protected_closeout_recheck_v1.json) verifies all four H parents, both D0 parents, the complete runtime snapshot, all 529 historical manifest pins and 40 used shared-cache arrays against their current manifests and the independent root receipt. The cache inventory remains 6337 regular files, 41 symlinks and 534 directories. Rows 273/533 lack an R5 pin for their current manifest paths; their historical `.invalid` sibling manifests remain pinned and unchanged, as the preserved [reporting-filter correction](../../../../diagnostics/generated/hyperedge_readout_round6_20261016T233604Z/receipts/protected_closeout_recheck_correction_v1.json) documents. This path-specific history gap is not repaired or relabelled as full historical pin coverage. GPU0's current PID 12472, arguments, executable and observed start ticks match the prior R6 sample; the R5 command/ticks differ and the initial R6 ticks were unavailable. No R6 work was assigned to GPU0, and no assertion about its active model weights is made.
