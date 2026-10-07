# R-direct: completed full-TRAIN comparison and historical readiness

**Model decision: retain R-direct as the preferred response-family reference and ordinary D-sep as its flow component. Retain Dense1804 as the stronger nominal fluid-temperature reference. Launch decision: the authorized fresh Run3901 flow5000 and Run3902 thermal5000 are complete; this evaluation does not justify another training launch.** R-direct learns how each physical heater changes temperature at a receiver while reusing a frozen, heat-independent flow predictor. It supports stable fixed-geometry heat increments, but its completed fit does not dominate the historical models on every physical quantity. R-group and all historical models remain preserved.

**Predictor — gains.** On the shared canonical89 panel, R-direct lowers mean u error by 33.4% and surface-temperature error by 27.5% relative to Dense1804. On six existing signed heat changes across three layouts, its mean fluid-temperature response error is 16.2% lower and surface response error is 14.8% lower. The precise prepared arithmetic passes all 288 checks.

**Predictor — misses and next step.** Nominal fluid-temperature error is 44.3% higher than Dense1804, vorticity error is 53.7% higher, and mean module-peak error is 6.4% higher. Material and surface tails remain worse despite better means. Geometry response remains unqualified. Use the completed checkpoints for response research and retain the stronger nominal references; do not infer a universally superior predictor or start a new search from this comparison.

**Organizer — measured evidence and next step.** Native complete-call latency is lower for R-direct on the fixed four layouts. Its receiver-local kernel yields tested omission bounds and equal-count geometry controls, with many patches retaining every donor. These are explanatory support measurements, not causal donor truth, adaptive global grouping superiority or measured sparse-executor savings. Keep the source-resolved path authoritative.

**Inverse — measured evidence and next step.** All four models choose the same correct minima within the small existing heating pools. This gives no unique inverse-design advantage. No inverse generator, continuous search, new solves or independently validated designs were produced. That limitation does not invalidate the completed research recipe.

This added assessment is dated 2026-10-07. The original readiness report is preserved below as a historical section; its earlier “no 5,000-epoch fit was launched” statement describes the bounded preparation round before the subsequent explicit formal-launch authorization.

## Comparison contract and checkpoint identities

The primary panel is all 89 canonical source-TEST cases, excluding TEST0273 because its inputs duplicate TRAIN0001. The original90 compatibility panel includes that duplicate and is reported separately. Both panels were repeatedly exposed to validation and monitoring; they are not a blind test. Every model uses the same ordered physical receivers, fluid masks, near/far definitions, interface/material queries and case IDs. The four detailed layouts are 0277/M3, 0291/M5, 0294/M7 and 0687/M10, fixed before this comparison. References come from the existing analytic-wake/shared-grid generator and saved response records, not an independent CFD validation.

Fresh formal training uses all 600 original TRAIN cases, a TRAIN-only normalizer, seed 0, FP32, effective batch 48 with microbatch 8, 1,024 primary fluid queries per case, and learning rate 3e-4 held through epoch 2,000 then cosine decay to 3e-6 at epoch 5,000 in each stage. Run3902 binds the exact completed Run3901 flow checkpoint. Each stage records 3,000,000 case visits, 65,000 optimizer updates and 3,072,000,000 primary fluid queries. The composed two-stage recipe therefore spends 6,000,000 visits and 130,000 updates; it is not a matched one-stage architecture ablation against the historical joint models.

Classics retain their own checkpoint-native input/target normalization, predicted-port path, complete local-surrogate evaluation and native receiver chunks; outputs are denormalized before physical errors are computed. Their legacy normalizer hash is `5cbe6473e6449e403626520e63ad1ca1cf283cd802d257d904c42e877c4f9129`, distinct from the fresh formal normalizer. Their saved configs point to the same packed source dataset, but do not carry the new ordered TRAIN membership seal. Objectives, histories, normalizers and component ages differ. The tables compare the actual trained systems, not isolated architecture effects.

| Model | Literal endpoint | Saved best |
|---|---|---|
| R-direct | e5000; `5aafc23cb86ea4d02706cda5749e97e67e42886328e1e3758148138e8736f2e5`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/RDirect_e5000/summary.json) | e4200; `28d19f6d5dbdf8f104363d70b2f19a8bb45df8ebf9be2c635bbdc74b2283ddc1`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/flow_pair/r_direct_best_e4200_original90_retry2/summary.json) |
| Run1804 | e5000; `9d0b83c562cecc2ffc52c3a08c993dfa8f47ae0e769f54ed6ffdd966047d63d9`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/classic_fields/Run1804_e5000_latest/summary.json) | e4738; `71ed480ff0396813491c650dd11d887195174019b373fbd9a1fb25505142c066`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/classic_fields/Run1804/summary.json) |
| Run1401 | e5000; `cee978f0461db928b72c3b6b66cb0ef56647674f82d6ef2660a8380754a829d6`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/classic_fields/Run1401_e5000_latest/summary.json) | e4585; `5be150bd6b4fc79599af62c767fba84490ba50edcc8cc8ce85026ae27a1846b3`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/classic_fields/Run1401/summary.json) |
| Run1502 | e5000; `20af85200796f639039d4853425fe91ee76e19e7caf5b5b68863badfe7690895`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/classic_fields/Run1502_e5000_latest/summary.json) | e4794; `08d86a573c7f7d86463bde231eb9f84a745602fca2b8f97142f39540d33a85bb`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/classic_fields/Run1502/summary.json) |

Run1401’s literal endpoint is the run-root `epoch_5000_model.pt`, not `checkpoints/latest.pt`; its comparison identifier `Run1401_e5000_latest` does not change that path. Run1804 and Run1502 endpoints are their SHA-bound `checkpoints/latest.pt` files. R-direct’s thermal saved-best alias contains actual epoch 4,200, with the same frozen flow e5000 and exact formal memberships, schedule and normalization bindings as its endpoint. There is no standalone e4200 milestone. The sampled-monitor selector (0.002256388845984311 at e4200) remains intact; full-grid evaluation is reported separately and does not relabel the saved-best alias.

Formal TRAIN-ID SHA256 is `c991acbcdced62e887385bc2763d4da7f4686256f72e44f797b69185522f3e3e`; normalization binding is `b06a65cc1e84176badea9b77431ae5f751ea9672295afd3aad883d0ed61abd86`; validation binding is `64a183f1aecb3672f284617009697e46ac3b32d848f73b321c3cac6de5b2a346`; source metadata SHA256 is `1bba5ab5c0535fabab2f2434de33eccb67184ef3fd7d7881eb01f5088e15e211`. Development remains fixed25_v1 150/22. Neither development resume checks nor the literal-e5000 formal CLI guard were weakened.

## Endpoint physical results

Cells below are equal-case mean / case p90 physical RMSE on canonical89. Lower is better; units follow each stored native field. Module peaks are native per-module material maxima, then per-case RMSE over modules. No normalized training loss is used as a substitute for physical error.

| Physical row | R-direct | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| Fluid u | 0.00432883 / 0.00494489 | 0.00650424 / 0.00835394 | 0.00960653 / 0.0122716 | 0.00970305 / 0.0122938 |
| Fluid v | 0.000459456 / 0.000515626 | 0.000331887 / 0.000460536 | 0.000428815 / 0.000589572 | 0.000531272 / 0.000743702 |
| Fluid p | 0.00197505 / 0.00258988 | 0.00224869 / 0.00359335 | 0.00416286 / 0.00697683 | 0.00423869 / 0.00617787 |
| Fluid vorticity | 0.0375592 / 0.0599751 | 0.0244341 / 0.053056 | 0.0260128 / 0.0425843 | 0.028128 / 0.0576348 |
| Fluid temperature | 0.342425 / 0.49042 | 0.237375 / 0.389084 | 0.312974 / 0.43949 | 0.267942 / 0.405377 |
| Surface temperature | 0.311198 / 0.705303 | 0.429475 / 0.557424 | 0.515004 / 0.639925 | 0.520884 / 0.666743 |
| Material temperature | 0.307594 / 0.716253 | 0.329348 / 0.43421 | 0.421362 / 0.575138 | 0.416504 / 0.543476 |
| Module material peak | 0.338021 / 0.832998 | 0.317573 / 0.530304 | 0.478912 / 0.795911 | 0.466868 / 0.683553 |

The material mean improvement over Dense1804 is small (6.6%) and its paired interval crosses zero. R-direct wins module-peak RMSE on 48/89 cases, yet its mean and p90 are worse because of large misses. Its worst module-peak case is 0287 (1.30422), worst material case is 0294 (0.944081), and worst surface/fluid-temperature case is 0295 (0.985360/0.870243). These failures remain in the numerical evidence and figures.

| Role | R-direct minus Dense mean RMSE | Paired 95% interval | R-direct wins / 89 |
|---|---:|---:|---:|
| Fluid u | -0.00217541 | [-0.00247266, -0.00185821] | 84 |
| Fluid temperature | 0.105049 | [0.0893296, 0.121076] | 6 |
| Surface temperature | -0.118277 | [-0.156624, -0.0805346] | 74 |
| Material temperature | -0.0217544 | [-0.0592276, 0.0154034] | 66 |
| Module material peak | 0.0204473 | [-0.0332275, 0.0768823] | 48 |
| Near vorticity | 0.0365396 | [0.0294168, 0.0432492] | 9 |

Intervals use 10,000 paired case resamples with seed 0. They describe one exposed panel, not uncertainty across training seeds or a causal architecture effect; no multiple-comparison adjustment is implied.

### Near/far fields and native ports

| Physical row | R-direct | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| Near u | 0.00709583 / 0.0101229 | 0.0077493 / 0.0102006 | 0.00862538 / 0.0112242 | 0.00985822 / 0.0145282 |
| Near v | 0.000711308 / 0.00080918 | 0.000556422 / 0.000765985 | 0.000825787 / 0.00125774 | 0.000930896 / 0.00120486 |
| Near p | 0.00322881 / 0.00505866 | 0.00295946 / 0.00506513 | 0.00502164 / 0.00750185 | 0.00562898 / 0.00774446 |
| Near omega | 0.0889746 / 0.140299 | 0.052435 / 0.111844 | 0.0503735 / 0.0943597 | 0.0610032 / 0.133315 |
| Near temperature | 0.322688 / 0.652558 | 0.272913 / 0.44262 | 0.32214 / 0.468802 | 0.288976 / 0.513653 |
| Far u | 0.00347158 / 0.00390372 | 0.00621897 / 0.0081149 | 0.00971469 / 0.0120168 | 0.00959339 / 0.0124043 |
| Far v | 0.000401012 / 0.000445395 | 0.000266325 / 0.000348165 | 0.000302075 / 0.000416664 | 0.000411112 / 0.000527492 |
| Far p | 0.0015958 / 0.00185837 | 0.00207149 / 0.00310254 | 0.0039744 / 0.00675726 | 0.0039055 / 0.00567716 |
| Far omega | 0.0133834 / 0.0152836 | 0.0113225 / 0.0141754 | 0.0168142 / 0.0228455 | 0.0129732 / 0.0162815 |
| Far temperature | 0.344597 / 0.476561 | 0.228885 / 0.374928 | 0.311169 / 0.450316 | 0.262362 / 0.394255 |

| Physical row | R-direct | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| Normal flux proxy | 1.43326 / 2.36954 | 1.53376 / 2.39451 | 2.015 / 2.63486 | 1.68273 / 2.2754 |
| Final outside temperature | 0.300105 / 0.682245 | 0.85156 / 1.03613 | 0.751084 / 0.927114 | 0.74174 / 0.937015 |
| Final effective h | 0.0109522 / 0.000113759 | 0.59063 / 0.857544 | 0.650998 / 0.913205 | 0.547675 / 0.80066 |
| Initial outside temperature | N/A | 0.767828 / 0.965704 | 2.51877 / 3.63974 | 0.748176 / 0.972072 |
| Initial effective h | N/A | 2.58072 / 2.859 | 7.58199 / 7.67967 | 0.533734 / 0.695866 |
| Inlet–outlet pressure difference | 0.00132857 / 0.0024923 | 0.00102587 / 0.00210062 | 0.00301812 / 0.00545861 | 0.00151789 / 0.00325609 |

Near-boundary vorticity is a substantial R-direct miss. Its frozen ordinary D-sep flow was chosen in the earlier matched refinement pair, so this endpoint comparison does not turn the rejected near-boundary refinement into a selected model. Initial-port history is not applicable to R-direct and is left N/A.

Normal flux and effective h are native proxies. The R-direct construction uses `q = −harmonic(k)/delta × (T_outside − T_surface)`, making `h = q/(T_surface − T_outside)` largely equal to `harmonic(k)/delta` away from clamps. Its tiny median/tail h errors therefore do not independently demonstrate learned heat-transfer accuracy. The h mean is driven by outliers, including case 0282 (0.748986). These values do not certify continuum boundary flux.

### Per-field relative error, population strata and compatibility

| Fluid channel | R-direct | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| u | 0.00562176 | 0.00817124 | 0.0120969 | 0.0122766 |
| v | 0.0176551 | 0.0131344 | 0.0173283 | 0.0211146 |
| p | 0.0197704 | 0.0239279 | 0.0460201 | 0.0423585 |
| omega | 0.0592198 | 0.0420716 | 0.04136 | 0.0474568 |
| temperature | 0.0445331 | 0.0324413 | 0.0416466 | 0.0369812 |

These are pooled per-channel physical relative L2 values, using independently recomputed raw-reference energies and matched error sums. Combining the five channels into one Euclidean score mixes units and is retained only as legacy compatibility evidence in the linked numerical artifact, not as a model selector.

| Module count / cases | Fluid T mean: R / 1804 / 1401 / 1502 | Peak mean: R / 1804 / 1401 / 1502 |
|---|---|---|
| M3 / 24 | 0.229433 / 0.138517 / 0.203121 / 0.173835 | 0.172608 / 0.221684 / 0.284585 / 0.321997 |
| M5 / 25 | 0.333088 / 0.241382 / 0.288362 / 0.280522 | 0.379199 / 0.276853 / 0.389138 / 0.43043 |
| M7 / 25 | 0.440736 / 0.304683 / 0.41303 / 0.356227 | 0.502733 / 0.449585 / 0.667972 / 0.589891 |
| M10 / 15 | 0.374921 / 0.276692 / 0.363001 / 0.250401 | 0.259529 / 0.318843 / 0.62436 / 0.554356 |

Original90 compatibility means, including duplicate 0273:

| Physical row | R-direct | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| Fluid temperature | 0.341547 | 0.236197 | 0.311722 | 0.266698 |
| Surface temperature | 0.309506 | 0.429352 | 0.514088 | 0.520344 |
| Material temperature | 0.305823 | 0.328713 | 0.419776 | 0.415278 |
| Module material peak | 0.335927 | 0.316299 | 0.47553 | 0.46461 |

### Saved-best comparison, kept separate

| Physical row | R-direct | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| Fluid temperature | 0.34613 / 0.49669 | 0.218656 / 0.369171 | 0.284443 / 0.425218 | 0.250088 / 0.394572 |
| Near temperature | 0.325044 / 0.642304 | 0.249452 / 0.417311 | 0.305512 / 0.443519 | 0.27201 / 0.496213 |
| Far temperature | 0.348651 / 0.487839 | 0.212036 / 0.352819 | 0.280375 / 0.427392 | 0.244487 / 0.367545 |
| Surface temperature | 0.312285 / 0.690755 | 0.448389 / 0.602693 | 0.458676 / 0.589466 | 0.4587 / 0.631604 |
| Material temperature | 0.308358 / 0.727054 | 0.340027 / 0.43283 | 0.356474 / 0.479992 | 0.35276 / 0.485158 |
| Module material peak | 0.335256 / 0.842339 | 0.346586 / 0.58687 | 0.359877 / 0.585881 | 0.358339 / 0.612205 |

R-direct uses thermal e4200 plus frozen flow e5000; Dense1804/HONF1401/HONF1502 use e4738/e4585/e4794 respectively. Endpoint e5000 slightly improves R-direct’s full-grid fluid, surface and material means relative to e4200, while peak mean slightly worsens. The different monitoring selectors and checkpoint ages remain explicit.

## Five inspected physical-result figure families

The compact index is: **1**, nominal fields/residuals and flow; **2**, signed heat response; **3**, actual support and bounded local controls; **4**, native ports/peaks and geometry response; **5**, measured latency, memory and completed training cost. All ten PNG presentation panels below were visually inspected. PDF masters and exact saved arrays remain local in the linked comparison directory; no generated figures or numerical data are uploaded to Git.

![Shared temperature reference and four endpoint predictions on fixed0291/M5](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/01_fields_0291_temperature.png)

**Figure 1a.** Native fluid temperature on 0291/M5 uses one shared physical range across the stored reference and four literal-e5000 predictions. Numbered disks are display ordinals 1–5; literal source IDs remain `0291:module:0` through `:4`. The field is a nominal predictor check, not an independent response solve. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/01_fields_and_residuals_endpoints.pdf).

![Four endpoint signed temperature residuals on 0291](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/01_residuals_0291_temperature.png)

**Figure 1b.** Signed predicted-minus-reference temperature residuals use a shared range within 0291. The captions inside each panel show physical RMSE; the coherent residual structure prevents a small global score from hiding local misses. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/01_fields_and_residuals_endpoints.pdf).

![Four endpoint signed temperature residuals on 0687/M10](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/01_residuals_0687_temperature.png)

**Figure 1c.** The fixed high-module-count layout 0687/M10 retains its own common residual range across models. This is the same saved native grid and mask, with ten display disks; it is not an additional selected case. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/01_fields_and_residuals_endpoints.pdf).

![Reference and four endpoint u/vorticity fields on 0291](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/01_flow_u_omega_0291.png)

**Figure 1d.** Streamwise velocity and vorticity expose the tradeoff in the frozen ordinary D-sep flow. Canonical89 u RMSE is 0.00432883 versus Dense1804’s 0.00650424, while vorticity is 0.0375592 versus 0.0244341. Units and reference limits are native to the saved generator. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/01_fields_and_residuals_endpoints.pdf).

### Signed heat changes and precise prepared reuse

| Primary signed-response mean RMSE | R-direct | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| Fluid temperature | 0.00884525 | 0.0105621 | 0.0196841 | 0.0137821 |
| Surface temperature | 0.0153821 | 0.0180586 | 0.0428163 | 0.0253747 |
| Normal flux proxy | 0.0554767 | 0.122495 | 0.201084 | 0.120676 |
| Material temperature | 0.0187677 | 0.0187927 | 0.0357857 | 0.0242 |

![Stored signed heating transfer and four endpoint response predictions on 0291](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/02_counted_signed_response_0291.png)

**Figure 2.** The saved 0291 plus transfer acts on literal modules 1 and 4 (display modules 2 and 5). Three layouts contribute six primary signed directions; 0277 lacks a counted baseline and is evaluated only as a secondary minus-to-plus span. No packed nominal baseline is substituted. Reference mean fluid changes are −0.0289189/+0.0289138 versus R-direct −0.0402401/+0.0402401, a 39.2% amplitude excess despite the aggregate improvement. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/02_counted_signed_response_0291.pdf).

Heat-only u/v/p/vorticity increments are exactly zero under the prescribed heat-independent D-sep flow assumption. This is a model/reference assumption, not newly discovered buoyancy physics. R-direct’s material response is essentially tied with Dense1804; one excitation direction per layout does not validate every source column independently. No certified physical sign or discretization-error floor is available.

The endpoint precision audit covers four TRAIN anchors and four fixed response layouts. All 288 precise endpoint/increment role checks pass; maximum endpoint discrepancy is 1.42109e-14 and maximum native VJP discrepancy is 8.88178e-16. All 250 failing legacy FP32 endpoint checks remain recorded. Precise cold/prepared outputs agree and weights remain frozen; arithmetic consistency is distinct from physical accuracy. The additional four original-TRAIN heat-response families have mean temperature response RMSE 0.004052/0.003496/0.003865 for fluid/surface/material and 0.06898 for flux proxy; these are exposed calibration evidence, not a new held-out comparison.

![Actual source strength and classic support on 0291 plus measured native work](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/03_receiver_support_fidelity_work.png)

**Figure 3a.** The saved native support arrays show R-direct source strength and the classic models’ available interaction views; legacy 1401 has no modern executor ledger. Absolute summed kernel strength is in temperature per physical heating unit. Learned support is not physical causality; the complete-call timings do not measure a sparse executor. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/03_receiver_support_fidelity_work.pdf).

![Receiver-local donor cover and equal-count geometry controls on 0291](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/03b_receiver_local_kernel_equal_k_controls.png)

**Figure 3b.** The full endpoint kernel is evaluated on 16 patches per fixed layout,64 patch matrices overall. At 1%/2% budgets the mean retained donor counts are 5.7031/5.5156 and 37/34 of 64 patches retain all donors. Across 112 saved patch-response rows per budget, learned/nearest/upstream covered response RMSE is 0.0081411/0.0097130/0.0088265 at 1% and 0.0081230/0.0096913/0.0089158 at 2%. For displayed 0291, each method retains the same sets at 1% as at 2%; choices differ between methods, as the omission plot shows. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/03b_receiver_local_kernel_equal_k_controls.pdf).

The bounded view uses `q_i(P) = r_i ||sqrt(W_P) K_i||_2` and the smallest descending-strength cover whose omitted strength sum is below the chosen fraction of sealed TRAIN response RMS 0.440821614. Its heating box 0.503670633–1.999153733 and radii are frozen from the predeclared 150-TRAIN calibration subset plus four TRAIN addenda; all IDs were verified inside the formal 600. This is a conservative reused TRAIN calibration, not a newly fitted full 600 response domain. Every measured model-omission triangle bound holds within the declared box. Equal-count nearest/upstream controls have their own measured bounds, which need not meet the learned selector’s requested epsilon.

Mean actual model omission RMS at 1% is 0.0000442085/0.00424039/0.00264733 for learned/nearest/upstream; at 2% it is 0.000118852/0.00430935/0.00283159. Averages include the secondary 0277 span and describe four layouts with one saved excitation direction each. They do not establish general grouping superiority, causal donor support or execution savings; no forced K or R-group retraining was performed.

![Native surface material peak and port errors for four endpoints](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/04_endpoint_native_roles_ports.png)

**Figure 4a.** The primary 89 cases retain separate surface/material/module-peak errors and native final-port roles. R-direct’s surface/material p90 values 0.705303/0.716253 exceed Dense1804’s 0.557424/0.434210. Its effective-h proxy has the algebraic limitation described above and cannot be read as an independent transport gain. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/04_native_roles_and_geometry.pdf).

![Saved TRAIN0348 obstacle-move response and four endpoint residuals](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/04_saved_geometry_response_0348_i_plus.png)

**Figure 4b.** The existing original-TRAIN 0348 `i_plus` move is evaluated on 7456 common fluid rows after separately rebuilding each geometry. Stored mean temperature change is +0.0881768; R-direct predicts −0.00300634, so its mean direction and amplitude fail. R-direct/Dense1804/HONF1401/HONF1502 response RMSE is 0.181461/0.197101/0.210595/0.197835 for this displayed state. A lower single-state RMSE does not qualify the geometry response. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/04_native_roles_and_geometry.pdf).

Across all 16 unique existing geometry perturbations (four TRAIN anchors × four signed obstacle moves), mean fluid-temperature response RMSE is 0.135146/0.113314/0.126869/0.116515 for R-direct/Dense1804/HONF1401/HONF1502. R-direct is 19.3% worse than Dense1804 on this aggregate. All 64 model-state comparisons preserve native coordinates, material/angular joins, source ordering and common-fluid masks. Geometry states are separate cold applications; fixed-geometry prepared kernels are never reused after moving obstacles.

All four endpoints correctly choose `transfer_plus` for the primary 0291/0294/0687 saved heating pools and `transfer_minus` for secondary 0277, with zero observed saved-reference regret. This is finite reuse of existing generator alternatives, not an inverse optimization campaign or a certified new design. No inverse/grouping superiority is required to retain the valid response research recipe.

![Matched native endpoint inference latency memory and actual two-stage 5000 training cost](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/05_measured_latency_training_cost.png)

**Figure 5.** On GPU 2, one warmup plus seven rotated complete-call repetitions per fixed native 8192-row case give mean case-median latency 25.239/163.290/52.530/229.506 ms for R-direct/Dense1804/HONF1401/HONF1502. These include input staging and output-to-CPU copy with native inner chunks and maps off. Actual full-TRAIN training totals 5.5455 h, or 5.6640 h including the two training processes’ validation/saving/startup overhead, replacing the earlier 6.07 h startup forecast. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/05_measured_latency_training_cost.pdf).

## Cost, reproducibility and validation

The fixed-four native latency ratios are 6.47×/2.08×/9.09× versus R-direct. All models were resident on the same GPU; mean incremental peak allocation is 21.845/38.772/49.228/40.952 MiB after subtracting each invocation baseline. Registered `nn.Linear` input-row evaluations average 657,497/4,456,756.75/708,342/4,407,766 per case. These are measured layer-row counts, not total FLOPs: widths, attention/stencils and backend differences prevent interpreting their ratio as theoretical computational complexity. Support/work collection is separate from latency calls. Full-access complete execution is reported throughout; no sparse executor savings were measured.

The completed flow stage used 2928.424 s training and 3041.918 s process time (peak 235.99 MiB); thermal used 17035.424 s training and 17348.404 s process time (peak 875.69 MiB). Both optimizer histories end at 65,000 steps and epoch 5,000. The current read-only comparison’s conservative GPU reservation span, including preflight failures and setup gaps, is below 0.67 GPU-associated hours. GPUs 0/2 are released. This assessment did not add training, optimizer updates or solver attempts; the ledger remains 326/326. It does not claim that all earlier preparation and successive user-authorized work share one six-hour elapsed window.

Read-only checks passed 61 relevant tests covering formal profiles, strict loaders, formal evaluation, the classic bridge, flow dependency and training identity. The durable change adds only exact age/SHA-whitelisted Run1401 identities and rejection tests; existing development/formal checks remain strict. Eighty physical fluid-field RMSE checks were independently recomputed in float64 from the saved fixed-four arrays. The duplicate R-direct endpoint pass on the other authorized GPU matched all 90 metric dictionaries exactly; that repeated execution is recorded, not additional independent scientific evidence. All evaluated frozen states remained unchanged. All 81 local report links resolve, and the rendered preview loads all 18 current/historical images and 16 tables without horizontal page overflow at 1280px or 900px. The ten new presentation images were visually inspected; figures and preview evidence remain in ignored local paths.

An initial legacy 1401 benchmark attempt failed before completion because its backend lacks the modern executor recorder; a retry measures common linear-layer rows and labels unsupported modern work as N/A. Two selected-best preflights failed on array equality and an assumed nonexistent e4200 milestone; both receipts remain, and the successful evaluation uses the actual best alias with exact endpoint bindings. The maintained formal CLI still requires literal e5000. The local selected-best wrapper preserves actual e4200 metadata and verifies the sealed endpoint identities rather than weakening that guard. Historical FP32 failures, source checkpoints, raw arrays and PDF masters are preserved.

The compact artifact index and native summaries make the result reproducible without another fit: [figure index](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/figure_and_metrics_index.json), [endpoint table](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/endpoint_native_metrics_table.json), [saved-best table](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/selected_best_native_metrics_table.json), [identity audit](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/flow_pair/identity_audit.json), [normalization audit](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/flow_pair/normalization_audit.json), [relative physical errors](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/physical_relative_l2.json), [independent array checks](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/independent_physical_checks.json), [counted heat replay](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/RDirect_e5000_counted/summary.json), [precision audit](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/flow_pair/r_direct_e5000_precision/summary.json), [receiver-local summary](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/flow_pair/receiver_local_formal_e5000/summary.json), [geometry replay](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/geometry/summary.json), [saved-pool comparison](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/stored_pool_comparison.json), [native timing](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/native_cost_retry1/receipt.json). The [maintained evaluator](../../tools/thermal_source_response_evaluate.py) supports the exact endpoint identities above with `--mode classic-fields --formal-panel original90 --formal-reference-checkpoint <Run3902 epoch_5000_model.pt>` and the corresponding classic identifier; R-direct uses `--mode fields --formal-panel original90`. Use a fresh output directory, the ModularDT environment and each literal checkpoint path from the linked summaries. The local one-time comparison scripts remain beside these artifacts; they are not uploaded. The [formal recipe](../guides/Thermal_RDirect_Formal5000.md) remains the lineage/configuration reference, while the measured completion and costs above supersede its earlier launch forecast.

## Historical readiness assessment (before formal launch authorization)

The following assessment retains the earlier bounded development evidence and figures. Its checkpoint ages, DEV22 metrics and no-launch decision belong to that earlier stage; they are not the completed formal comparison above.


**Model decision: use R-direct as the preferred response-family reference, with ordinary D-sep as its flow recipe. Launch decision: the fresh full-TRAIN flow5000 → fixed-flow thermal5000 research recipe is implemented and available for a manual launch; no 5,000-epoch fit was launched.** This round adds precise prepared increments, replays the historical Dense1804/HONF1502 models in their native paths, completes only the matched +500 D-sep refinement pair, and separates full-TRAIN identities from the unchanged fixed25_v1 development workflow. R-group, Run3801 and all historical checkpoints remain preserved.

**Predictor.** Precise fixed-kernel arithmetic passes all 288 endpoint/increment checks while retaining 250 failing legacy FP32 checks. R-direct lowers primary fluid heat-response RMSE from H-add’s 0.037067 to 0.011587 and surface-temperature RMSE from 0.7883 to 0.7123. The historical classics are substantially better absolute predictors on this DEV22 replay; R-direct’s module-peak mean RMSE is 0.7826 versus H-add’s 0.6477 (+20.8%). Its material-peak, near-vorticity and geometry-response misses remain visible. The near-boundary flow refinement was not selected; the ordinary continuation was better.

**Organizer.** All 64 receiver-patch matrices were measured with TRAIN-box omission bounds and equal-K nearest/upstream controls. Mean retained donor count is 5.75/5.58 at the 1%/2% budgets, with many patches retaining every donor. This is a bounded explanatory view. Adaptive global grouping, causal donor truth and sparse executor savings remain unproved.

**Inverse reuse.** Fixed-layout heat changes and native maxima can be replayed with stable arithmetic. Geometry remains inaccurate, and the existing solved-pool choices were already available to older controls. No inverse generator, continuous search or new design validation was performed. These missing capabilities do not block the research recipe.

**Price and execution evidence.** Three real full-TRAIN600 epochs ran for each fresh component on GPU 2, with 600 visits, 13 updates and 614,400 primary fluid queries per epoch. Mean training times were 0.715 s for flow and 3.653 s for thermal. The conditional training-only forecast is 0.99 + 5.07 = **6.07 GPU-associated hours** on one GPU, before future canonical89/original90 monitoring and larger checkpoint saves. Thermal stopped at e2 and continued from saved optimizer state through e3. Flow ran e1–e3 uninterrupted and has a zero-update endpoint restore check; continuing flow resume was not measured because its three-epoch allowance was already consumed. The guide records that limitation explicitly rather than adding unauthorized epochs. See the [tested manual guide](../guides/Thermal_RDirect_Formal5000.md), [flow profile](../../src/config_core/forward/thermal_source_response/d-sep_full5000.json) and [thermal profile](../../src/config_core/forward/thermal_source_response/r-direct_full5000.json).

### Five main result figures

The short figure index is: **1**, absolute temperature and residuals on fixed representative 0291 and high-M 0687; **2**, receiver-local source strength and signed heat response; **3**, grouping/work comparison and bounded local covers; **4**, saved heat/design changes and nonlinear maxima; **5**, measured work and manual launch price. PDF is the retained master; the small PNG companions enable direct Markdown display. All retained pages were visually inspected, use at most six panels, and label native dataset units rather than inventing SI units. The references are the existing analytic-wake/shared-grid generator and its saved alternatives, not independent CFD.

#### 1. What field does each model predict?

![Stored temperature, direct response, Classic Dense and signed residuals on DEV0291](../../diagnostics/generated/rdirect_readiness_20261007/figures/01_fields_0291.png)

**Figure 1a.** The same native fluid coordinates/masks and shared temperature/residual scales show the fixed representative 0291/M5; numbered disks identify modules and the star marks the measured material-temperature maximum. Across all 22 cases, fluid-temperature RMSE is 0.7746 for R-direct, 0.1979 for Classic Dense and 0.2467 for Classic HONF1502. The classic advantage is measured, but training population, objective and component ages differ, so this is not an isolated architecture effect. [PDF](../../diagnostics/generated/rdirect_readiness_20261007/figures/01_fields_0291.pdf).

![Stored temperature, direct response, Classic Dense and signed residuals on high-M DEV0687](../../diagnostics/generated/rdirect_readiness_20261007/figures/01_fields_0687.png)

**Figure 1b.** The predeclared high-M representative 0687/M10 retains its own shared physical and residual scales. The direct response reproduces the broad thermal wake but leaves structured errors; the classic Dense residual is smaller. The marked material maximum does not turn a fluid map into a peak-temperature certification. [PDF](../../diagnostics/generated/rdirect_readiness_20261007/figures/01_fields_0687.pdf).

| Physical row | H-add e500 | Direct e2500 | Grouped e2400 | Dense e4738 | HONF1502 e4794 |
|---|---|---|---|---|---|
| Fluid u | 0.022037 / 0.024914 | 0.017068 / 0.020804 | 0.017068 / 0.020804 | 0.0057149 / 0.0070266 | 0.0057348 / 0.0080422 |
| Fluid v | 0.0030797 / 0.0039929 | 0.0017125 / 0.0022202 | 0.0017125 / 0.0022202 | 0.00031904 / 0.00041079 | 0.0004142 / 0.00054682 |
| Fluid p | 0.012222 / 0.015606 | 0.0090926 / 0.011129 | 0.0090926 / 0.011129 | 0.0021 / 0.0027101 | 0.0024947 / 0.0030259 |
| Fluid omega | 0.10457 / 0.13261 | 0.13146 / 0.16482 | 0.13146 / 0.16482 | 0.023461 / 0.045707 | 0.022921 / 0.048003 |
| Fluid temperature | 0.77031 / 1.1015 | 0.77463 / 1.0177 | 0.84435 / 1.1176 | 0.19794 / 0.24768 | 0.24672 / 0.35577 |
| Surface temperature | 0.78833 / 1.1782 | 0.71231 / 1.1134 | 0.71674 / 1.1906 | 0.40316 / 0.49392 | 0.43369 / 0.5295 |
| Material temperature | 0.66308 / 1.1367 | 0.67773 / 1.0164 | 0.65899 / 0.97331 | 0.30043 / 0.34331 | 0.33453 / 0.46917 |
| Module material peak | 0.64773 / 1.213 | 0.78262 / 1.2034 | 0.74378 / 1.1139 | 0.28529 / 0.40907 | 0.34595 / 0.54235 |

Each cell in the table is equal-case mean / case p90 physical RMSE on the exact fixed25_v1 DEV22 IDs. Fluid rows use the identical fluid masks, surface/material rows use valid native receivers, and the module-peak row first computes each module's maximum over saved material queries, then per-case RMSE across modules. New-family initial-port history is NA; it is not replaced with final-port values. The classic paths use their own checkpoint-native TRAIN input/target transforms and their full predicted-port/local-surrogate evaluation, then denormalize before comparison. These metrics do not mix checkpoint-specific normalized errors.

#### 2. Which sources matter at this receiver?

![Input-chosen receiver patches, ranked learned donor strengths, signed contributions and saved heat-response fields](../../diagnostics/generated/rdirect_readiness_20261007/figures/02_receiver_sources.png)

**Figure 2.** Three patches were selected from input geometry and channel bounds: upstream, near module 0 and downstream overlap. Learned source strength is ranked within each patch, with physical source IDs at the points; signed current contributions retain those same IDs. The lower row compares the stored 0291 plus transfer, direct kernel response and signed residual. Stored mean fluid changes are −0.0289189/+0.0289138 for minus/plus, versus direct −0.0369188/+0.0369188: the direction is right but amplitude is about 27.7% too large. Environment dots are context only. One excitation direction per layout does not independently validate each donor column or establish causality. [PDF](../../diagnostics/generated/rdirect_readiness_20261007/figures/02_receiver_sources.pdf).

#### 3. Is grouping buying anything?

![Direct and grouped source counts and work, bounded receiver-local donor counts and equal-K controls](../../diagnostics/generated/rdirect_readiness_20261007/figures/03_grouping.png)

**Figure 3.** At M10, R-group has 11 valid modes and all 110 memberships are positive; M+1 is not learned adaptive K. Complete cold M12/Q8192 prediction is 25.450 ms for direct and 31.604 ms for grouped. The primary six signed heat responses give fluid-temperature RMSE 0.011587/0.013830. The new local view covers 16 fluid patches per case and 1%/2% of sealed TRAIN response RMS: mean retained K is 5.75/5.58, range 3–10. At matched K, mean covered physical response RMSE is 0.011678/0.011653 for learned strength, 0.012768/0.012768 for nearest geometry and 0.011757/0.011861 for upstream geometry. These descriptive four-state patch averages do not establish general superiority. The final panel separates actual model omission from its triangle bound. No sparse executor was run. [PDF](../../diagnostics/generated/rdirect_readiness_20261007/figures/03_grouping.pdf).

#### 4. What happens when the design changes?

![Saved signed heat transfers, actual material maxima and a difficult saved geometry response](../../diagnostics/generated/rdirect_readiness_20261007/figures/04_design_changes.png)

**Figure 4.** The fixed 0291 geometry uses the explicitly displayed minus/base/plus heating values; native material maxima are evaluated from endpoints rather than applying a linear increment to a maximum. The lower row uses the saved original-TRAIN 0348/M10 `i_plus` obstacle move on the common fluid mask. Its geometry-response RMSE is 0.19245 for direct and 0.18775 for grouped; the reference mean change is +0.087545 while direct predicts −0.003275. Geometry is therefore unqualified. Case 0277 has no fresh counted baseline and remains a secondary minus-to-plus span; its packed nominal case is never substituted into that span. Replaying stored alternatives is not new inverse-design evidence. [PDF](../../diagnostics/generated/rdirect_readiness_20261007/figures/04_design_changes.pdf).

#### 5. How much work, and what can be launched?

![Complete cold and prepared reuse timings, precise arithmetic price, real full-TRAIN epoch timings and manual forecast](../../diagnostics/generated/rdirect_readiness_20261007/figures/05_work_and_launch.png)

**Figure 5.** Existing complete cold M12/Q8192 timings are 120.973/25.450/31.604 ms for H-add/direct/grouped. Three-heat direct reuse costs 38.188 ms including preparation, versus 77.155 ms for three cold calls; grouped costs 43.865 versus 97.068 ms. In the separate new typed-record M10 precision audit, ordinary prepared/precise endpoint/precise increment calls cost 6.525/6.874/1.958 ms, with 43.783 ms preparation and 3,285,760 transient FP64 kernel bytes. The startup plot measures all scheduled objectives in genuine full-TRAIN epochs. Training-only flow5000/thermal5000 forecasts are 0.99/5.07 h; full-panel monitoring and later save overhead were not measured. These scopes do not justify adding overlapping inference timers or promising a six-hour end-to-end fit. [PDF](../../diagnostics/generated/rdirect_readiness_20261007/figures/05_work_and_launch.pdf).

### What the measured increments changed

The neural weights and ordinary FP32 path are unchanged. The opt-in `accumulation_dtype=torch.float64` path converts raw far/near factors before kernel construction, contracts physical heating in FP64, interpolates native temperatures in FP64 and recomputes the native harmonic-conductivity/delta proxy in FP64. It contracts Δh directly, before adding offsets or subtracting large endpoints. It does not merely cast a rounded final FP32 kernel, retrain in FP64 or change near/far semantics. Geometry remains bound to the prepared context and is rejected if stale.

The actual GPU2 audit covers four existing TRAIN response families (0001, 0318, 0333, 0348) and all four fixed representatives (0277, 0291, 0294, 0687), balanced and nonbalanced signed changes at 0.25, 0.001 and 1e-6. At the retained rtol 2e-5/atol 2e-6, all 288 precise endpoint/increment role checks pass; 250/288 legacy FP32 endpoint checks still fail and remain recorded separately. Maximum precise endpoint discrepancy is 1.51e-14 and native physical-kernel VJP discrepancy is 8.88e-16. Cold/prepared precise outputs are equal. The receipt also records input-heat quantization separately, so coefficient preservation is not confused with accuracy against physical truth. The CPU replay independently passed, with CUDA hidden.

Linear increments provide fluid/interface/material temperatures, q proxy and outside temperature. Effective h ratios and module maxima remain nonlinear endpoint functionals: evaluate both endpoints with the precise path when those quantities are needed. Increment results deliberately omit `pred_port_condition` and mark those functionals NA. Flow has zero heat increment under the audited prescribed-flow capability. Kernel factors remain learned estimates: passing arithmetic checks does not validate geometry, donor causality or continuum heat flux. See [precision summary](../../diagnostics/generated/rdirect_readiness_20261007/precision_gpu2/summary.json) and the [maintained inference guide](../guides/Thermal_Source_Response.md).

### The only refinement pair: ordinary versus near consistency

The pair starts from the same D-sep2500 flow tensors and AdamW state. The retained epoch_2500 and latest aliases have different ZIP filenames but identical loaded state and all 156 uncompressed members; that equivalence is recorded before selection. Both children visit the same 150 TRAIN cases for 500 additional epochs, with 2,000 updates and 75,000 case visits per arm. Their schedule is fixed at warmup 1e-6→3e-5 over 20 epochs, hold through 200, then cosine decay to 3e-6 at +500. The refinement adds 128 near-fluid centers per case and their generator-matched stencil; it does not enable heat access or change the four flow channels.

The native generator definition is omega = dv/dx − du/dy with dx=Lx/nx, dy=Ly/ny, first-order boundary differences and solid masking. It reproduced saved TRAIN omega with RMSE 1.74e-7/1.36e-7 on low/high-M qualification cases. Near receivers mean fluid centers within two radii of a module center, or surface distance at most one radius; far is the remaining fluid mask. The additional stencil visits 96,000 query rows per epoch; the ordinary primary sampler visits 153,600. These qualification and work measurements are not new physical solves.

The ordinary e3000 control improves the parent all-fluid mean u/v/p/omega RMSE by 2.46%/4.86%/4.32%/1.68%; near omega improves 1.48%. The near-consistency child is 12.68%/13.34% worse than control in mean/p90 fluid u, 13.05%/13.25% worse in near u, 0.17%/0.27% worse in near omega and 6.89%/5.98% worse in far omega. The final flow recipe therefore keeps the ordinary objective. No additional remedy, D-open run or thermal fit was started. Both histories and the failed CUDA lazy-initialization attempt are retained.

![Native u and omega reference, ordinary flow e3000 and signed residuals on DEV0291](../../diagnostics/generated/rdirect_readiness_20261007/refinement_pair/flow_control3000_u_omega_0291.png)

**Flow companion.** The ordinary control retains the large-scale flow but leaves near-wall omega errors on exposed 0291/M5. The same reference/prediction scale and symmetric residual scale are used per channel. This fresh continuation is not silently inserted into the selected R-direct2500 composition used in Figures 1–4, whose flow remains e2500. [PDF](../../diagnostics/generated/rdirect_readiness_20261007/refinement_pair/flow_control3000_u_omega_0291.pdf). The separately inspected [high-M companion](../../diagnostics/generated/rdirect_readiness_20261007/refinement_pair/flow_control3000_u_omega_0687.pdf) and [closeout/alias receipt](../../diagnostics/generated/rdirect_readiness_20261007/refinement_pair/workstream_c_closeout_cpu.json) preserve detailed evidence.

| DEV22 physical flow row | Parent e2500 | Ordinary e3000 | Near consistency e3000 |
|---|---|---|---|
| fluid u | 0.017068 / 0.020804 | 0.016648 / 0.02003 | 0.018759 / 0.022702 |
| fluid v | 0.0017125 / 0.0022202 | 0.0016293 / 0.0020779 | 0.0016924 / 0.0021622 |
| fluid p | 0.0090926 / 0.011129 | 0.0086996 / 0.010446 | 0.0088306 / 0.010644 |
| fluid omega | 0.13146 / 0.16482 | 0.12926 / 0.16177 | 0.1304 / 0.16218 |
| near u | 0.034137 / 0.036876 | 0.033693 / 0.036274 | 0.03809 / 0.041081 |
| near v | 0.0029301 / 0.0035319 | 0.0027584 / 0.0032862 | 0.0029117 / 0.0034778 |
| near p | 0.013814 / 0.014708 | 0.013578 / 0.014576 | 0.013726 / 0.0142 |
| near omega | 0.32075 / 0.34269 | 0.31599 / 0.33643 | 0.31654 / 0.33733 |
| far u | 0.01173 / 0.01486 | 0.011227 / 0.013929 | 0.01257 / 0.015542 |
| far v | 0.0013823 / 0.0018378 | 0.0013294 / 0.001729 | 0.0013619 / 0.0017624 |
| far p | 0.0078157 / 0.010539 | 0.0073872 / 0.0097244 | 0.0075161 / 0.009895 |
| far omega | 0.047188 / 0.05551 | 0.045657 / 0.053453 | 0.048804 / 0.056651 |

### Receiver-local bounds, controls and physical limits

For each native 4×4 equal-area fluid patch P, normalized fluid-only quadrature W_P defines q_Pi = r_i ||W_P^(1/2) K_i||_2. Physical source IDs/centers are retained. The input-only balanced radius is r_i = max(0, min(h_i−h_min, h_max−h_i, 0.20 mean active h)); h_min=0.5036706 and h_max=1.9991537 are rederived from the exact 150 selected TRAIN inputs, with four TRAIN addendum source hashes verified. Budget is 0.01/0.02 × the retained TRAIN-only temperature-response RMS 0.4408216, or 0.0044082/0.0088164 native temperature units. Counted/DEV response errors do not set radii or budgets.

The learned cover sorts q and retains the smallest prefix whose omitted-score sum is within budget. Nearest and upstream geometry controls retain exactly the same K. All 112 saved response/patch rows per budget lie inside their per-source radii and satisfy the triangle inequality for model omission. This bound limits distortion of the learned model, not error against the physical reference. Covered/full physical errors are reported separately. Case 0277/0294 retains every source, 0291 averages about 4.2 of 5, and 0687 about 8.1–8.8 of 10. FP64 kernel re-export differs from the retained FP32 operator by at most 4.95e-6; its stored cold-response replay differs by at most 5.59e-6. Those ordinary-rounding differences remain explicit.

This tool is post-fit and explanatory: the authoritative forward path remains source-resolved and full-access. It measures cover size, omitted bound, measured omission and reference error, but not a sparse executor or saved forward work. No patch-SVD/training search was added. Full matrices, source contributions and every equal-K control are in the [local-view summary](../../diagnostics/generated/rdirect_readiness_20261007/local_view/summary.json) and its four checkpoint-bound kernel arrays.

### Formal software, data boundaries and launch status

The two maintained formal profiles resolve all 600 original TRAIN IDs from packed-H5 metadata and fit one global normalizer on those cases alone. Flow and thermal carry identical memberships, transforms and canonical89/original90 bindings. The historical training duplicate 0273 is excluded from canonical89 and remains in the original90 compatibility panel. Startup is explicitly disposable, validates only on DEV22, cannot exceed e3 and cannot be resumed or promoted into a formal identity. The legacy fixed25_v1 150/22 loaders and same-arm resume checks remain strict. The new formal loader accepts valid monitored thermal ages with an exact e5000 formal flow partner; final endpoint reporting requires both e5000 ages explicitly.

Both stages start from fresh weights, seed 0, FP32 training, effective batch 48/microbatch 8 and Q1024 fluid sampling. Formal LR holds 3e-4 for 2,000 epochs, then cosine-decays to 3e-6 by e5000. Thermal retains the three standardized temperature roles, 0.05 q proxy, the same four saved TRAIN response anchors and qualified TRAIN-only discrete-balance objective; calibration is recomputed at fresh initialization using the full-TRAIN normalizer. Every 100 epochs, each stage saves monitoring/latest/best-field state and its declared milestone. Nonempty preparation destinations are rejected before writes; resumes require matching profile, schedule, data, normalizer, partner and source identities. Trusted checkpoint loading and atomic saves remain in use.

The actual thermal startup recorded 109,344 material-query and 54,672 surface-query rows per epoch, 76,800 qualified operator rows and 437,376 operator source-column rows, plus 17,152–17,392 response-target rows. These counters derive from actual sampled cases and native extraction; neural and stencil work is recorded separately rather than replaced with logical sparsity. Thermal's 52 Adam states progress from step 26 at the clean stop to step 39 after resume; the frozen e3 flow remains unchanged. DEV22 thermal selector improves 0.5071→0.4307 and common five-field standardized MSE 1.0009→0.9965. Physical u/v/p/omega/T RMSE at e3 is 0.3183/0.05373/0.1700/1.1327/5.2885, demonstrating startup composition rather than mature accuracy.

Measured initial flow training epochs are 0.843/0.652/0.651 s; thermal epochs are 3.799/3.402/3.758 s. Flow DEV22 validation/save/loading are 0.0107/0.4848/0.2236 s; thermal's two validation/save/loading intervals total 0.1256/0.9980/1.0030 s. Peak allocated CUDA memory is 247,451,648 bytes flow and 896,102,400 bytes thermal. Data loading and other preparation overhead are separately recorded in process receipts and are not added twice. The original three training process intervals total 41.411 s; actual training is 13.104 s. The later zero-update flow verification is separate. Full90 truth was not evaluated during startup. The 6.07 h forecast is conditional on the same GPU/schedule/contention and includes training objectives only; it does not guarantee accuracy or end-to-end runtime. Exact manual prepare/start/status/stop/resume commands and the save/resume limitation are in the [guide](../guides/Thermal_RDirect_Formal5000.md).

### Historical identities and detailed numerical evidence

The common exposed DEV22 manifest is `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`. Selected checkpoints are Dense1804 e4738 (`71ed480ff0396813491c650dd11d887195174019b373fbd9a1fb25505142c066`), HONF1502 e4794 (`08d86a573c7f7d86463bde231eb9f84a745602fca2b8f97142f39540d33a85bb`), R-direct thermal e2500 (`05974d2fbc367bad8f2092063818ce9073c21783131753b3648fb3ea11a870aa`) with D-sep e2500, R-group selected thermal e2400 with the same flow, and retained H-add Run3801 e500. Classical/global and development-quarter histories differ; normalization and training age are preserved rather than relabelled as matched training.

#### Same saved heat responses: historical controls

Mean state RMSE below uses the same six primary baseline-relative changes on 0291/0294/0687. Direct is better than both classics on surface/material response error, while Dense is slightly better on fluid-temperature response error. This distinction supports a reusable response reference without claiming all-role superiority. All models retain their native training history and extraction; the secondary 0277 span is separate.

| Primary physical response | Direct e2500 | Grouped e2400 | Dense e4738 | HONF1502 e4794 |
|---|---|---|---|---|
| Fluid temperature | 0.011587 | 0.013830 | 0.011020 | 0.012818 |
| Surface temperature | 0.014868 | 0.015816 | 0.017235 | 0.025743 |
| Material temperature | 0.016767 | 0.017369 | 0.017839 | 0.026818 |

| Saved layout / change | Direct e2500 | Grouped e2400 | Dense e4738 | HONF1502 e4794 |
|---|---|---|---|---|
| 0277 / transfer_plus (secondary span) | 0.022235 | 0.029147 | 0.065340 | 0.030881 |
| 0291 / transfer_minus | 0.025253 | 0.029382 | 0.018120 | 0.019319 |
| 0291 / transfer_plus | 0.025260 | 0.029391 | 0.018712 | 0.021256 |
| 0294 / transfer_minus | 0.008177 | 0.010514 | 0.013547 | 0.015087 |
| 0294 / transfer_plus | 0.008177 | 0.010514 | 0.011307 | 0.015420 |
| 0687 / transfer_minus | 0.001329 | 0.001590 | 0.002225 | 0.003093 |
| 0687 / transfer_plus | 0.001329 | 0.001590 | 0.002211 | 0.002733 |

The heat-response error of H-add is 0.037067/0.050119/0.049496 for fluid/surface/material temperature on the same primary states. Its wrong-sign mean changes on 0291 remain retained (+0.010420/−0.015508 for minus/plus). The precise increment repair changes arithmetic, not those historical physical predictions.

Literal adjacent e5000 `latest.pt` checkpoints exist and were additionally replayed read-only on DEV22. Dense e5000 SHA is `9d0b83c562cecc2ffc52c3a08c993dfa8f47ae0e769f54ed6ffdd966047d63d9`; HONF1502 e5000 SHA is `20af85200796f639039d4853425fe91ee76e19e7caf5b5b68863badfe7690895`. They are endpoint comparisons and do not replace the selected states. Their fluid T mean/p90 RMSE is 0.22349/0.29301 and 0.26577/0.38331; module-peak RMSE is 0.26767/0.41649 and 0.43650/0.58220. Native models, weights and source files remain unchanged.

| Physical row | H-add e500 | Direct e2500 | Grouped e2400 | Dense e4738 | HONF1502 e4794 |
|---|---|---|---|---|---|
| Near u | 0.030499 / 0.035402 | 0.034137 / 0.036876 | 0.034137 / 0.036876 | 0.0070298 / 0.011735 | 0.007657 / 0.013494 |
| Near v | 0.0042195 / 0.0053616 | 0.0029301 / 0.0035319 | 0.0029301 / 0.0035319 | 0.00053293 / 0.00064699 | 0.00073675 / 0.0010112 |
| Near p | 0.015464 / 0.018712 | 0.013814 / 0.014708 | 0.013814 / 0.014708 | 0.002796 / 0.0043792 | 0.0032513 / 0.004893 |
| Near omega | 0.23768 / 0.28737 | 0.32075 / 0.34269 | 0.32075 / 0.34269 | 0.052648 / 0.10597 | 0.047963 / 0.10457 |
| Near temperature | 0.77834 / 1.1666 | 0.7695 / 1.0917 | 0.80461 / 1.2566 | 0.22798 / 0.35022 | 0.26507 / 0.48669 |
| Far u | 0.020042 / 0.022867 | 0.01173 / 0.01486 | 0.01173 / 0.01486 | 0.0053391 / 0.0063835 | 0.0052205 / 0.006428 |
| Far v | 0.0027983 / 0.003557 | 0.0013823 / 0.0018378 | 0.0013823 / 0.0018378 | 0.00025768 / 0.00032377 | 0.00031425 / 0.00040713 |
| Far p | 0.011501 / 0.015156 | 0.0078157 / 0.010539 | 0.0078157 / 0.010539 | 0.0019279 / 0.0023644 | 0.0022952 / 0.0028058 |
| Far omega | 0.053536 / 0.06514 | 0.047188 / 0.05551 | 0.047188 / 0.05551 | 0.011211 / 0.014088 | 0.0118 / 0.015393 |
| Far temperature | 0.76757 / 1.1307 | 0.76836 / 0.99889 | 0.84797 / 1.1612 | 0.19208 / 0.24296 | 0.24127 / 0.32262 |
| Native q proxy | 3.352 / 4.1176 | 2.559 / 3.4789 | 2.6864 / 3.624 | 1.5561 / 1.9344 | 1.6105 / 2.0886 |
| Final outside temperature | 1.1274 / 1.5114 | 0.72857 / 1.1378 | 0.75146 / 1.2569 | 0.8159 / 0.94122 | 0.83456 / 0.94682 |
| Final effective h | 0.82872 / 1.0807 | 4.1115e-05 / 7.6852e-05 | 0.036806 / 7.6874e-05 | 0.62013 / 0.87519 | 0.59819 / 0.84696 |
| Initial outside temperature | 1.413 / 2.2311 | NA | NA | 0.77516 / 0.92808 | 0.85839 / 1.035 |
| Initial effective h | 11.871 / 11.902 | NA | NA | 2.7701 / 2.988 | 0.56746 / 0.7316 |
| Inlet–outlet pressure difference | 0.0070759 / 0.012437 | 0.0069276 / 0.01176 | 0.0069276 / 0.01176 | 0.00095241 / 0.0019682 | 0.0018654 / 0.0030195 |

#### Module-count appendix

These are the same exposed DEV22 reductions, split by actual active module count. Cells are mean / p90 physical RMSE; small stratum sizes are shown explicitly and do not support population-wide tail estimates. Other fields and H-add strata remain in the linked saved summaries.

| Active modules / cases / physical row | Direct e2500 | Grouped e2400 | Dense e4738 | HONF1502 e4794 |
|---|---|---|---|---|
| M3 / 6 / u | 0.014531 / 0.015756 | 0.014531 / 0.015756 | 0.0051136 / 0.0056195 | 0.0045919 / 0.0049996 |
| M3 / 6 / Near omega | 0.32832 / 0.36859 | 0.32832 / 0.36859 | 0.043927 / 0.078918 | 0.021586 / 0.037905 |
| M3 / 6 / Fluid T | 0.57057 / 0.79406 | 0.63639 / 0.7882 | 0.14506 / 0.15383 | 0.18918 / 0.2233 |
| M3 / 6 / Module peak | 0.54545 / 0.76102 | 0.4851 / 0.61712 | 0.20025 / 0.25225 | 0.22274 / 0.26516 |
| M5 / 6 / u | 0.015954 / 0.016332 | 0.015954 / 0.016332 | 0.0060189 / 0.0071249 | 0.0061768 / 0.0083826 |
| M5 / 6 / Near omega | 0.32669 / 0.33691 | 0.32669 / 0.33691 | 0.063764 / 0.11985 | 0.063588 / 0.11565 |
| M5 / 6 / Fluid T | 0.83382 / 1.0168 | 0.79538 / 0.96317 | 0.18209 / 0.23368 | 0.28124 / 0.38244 |
| M5 / 6 / Module peak | 0.93638 / 1.3883 | 0.86917 / 1.5013 | 0.23847 / 0.29096 | 0.35436 / 0.55996 |
| M7 / 6 / u | 0.017868 / 0.018718 | 0.017868 / 0.018718 | 0.0058217 / 0.0069568 | 0.0062221 / 0.0081936 |
| M7 / 6 / Near omega | 0.31565 / 0.33898 | 0.31565 / 0.33898 | 0.067117 / 0.12138 | 0.069183 / 0.12465 |
| M7 / 6 / Fluid T | 0.7788 / 0.91716 | 0.91846 / 1.0505 | 0.25565 / 0.36275 | 0.29168 / 0.45488 |
| M7 / 6 / Module peak | 0.74722 / 1.0451 | 0.7637 / 0.90061 | 0.32167 / 0.38529 | 0.38754 / 0.467 |
| M10 / 4 / u | 0.021346 / 0.022324 | 0.021346 / 0.022324 | 0.0060005 / 0.0067948 | 0.006055 / 0.0063178 |
| M10 / 4 / Near omega | 0.30815 / 0.31688 | 0.30815 / 0.31688 | 0.02735 / 0.02974 | 0.032263 / 0.040567 |
| M10 / 4 / Fluid T | 0.98571 / 1.127 | 1.1186 / 1.286 | 0.21449 / 0.23151 | 0.2138 / 0.23 |
| M10 / 4 / Module peak | 0.96085 / 1.1683 | 0.91383 / 1.0894 | 0.4285 / 0.58887 | 0.45579 / 0.58342 |

![Native angular surface temperature, original normal-q proxy and outside-port temperature](../../diagnostics/generated/rdirect_readiness_20261007/figures/appendix_native_roles.png)

**Native-role appendix.** Source 0 on exposed 0687 retains the original 64-angle receiver ordering. The plotted normal q is the saved native −harmonic(k)/delta × (outside−surface) proxy, not continuum flux. New-family initial-port history is NA. Surface, q and outside-temperature errors remain separate rather than being collapsed into fluid-temperature success. [PDF](../../diagnostics/generated/rdirect_readiness_20261007/figures/appendix_native_roles.pdf).

Detailed arrays and receipts are local ignored artifacts: [R-direct physical fields and strata](../../diagnostics/generated/response_operator_20261006/evaluation/R-direct_selected_fields/summary.json), [R-group fields and strata](../../diagnostics/generated/response_operator_20261006/evaluation/R-group_selected_fields/summary.json), [selected Dense DEV22](../../diagnostics/generated/response_operator_20261006/evaluation/classic_Run1804_DEV22_fields/summary.json), [selected HONF1502 DEV22](../../diagnostics/generated/response_operator_20261006/evaluation/classic_Run1502_DEV22_fields/summary.json), [Dense endpoint](../../diagnostics/generated/response_operator_20261006/evaluation/classic_Run1804_e5000_DEV22_fields/summary.json), [HONF1502 endpoint](../../diagnostics/generated/response_operator_20261006/evaluation/classic_Run1502_e5000_DEV22_fields/summary.json), [direct counted responses](../../diagnostics/generated/response_operator_20261006/evaluation/R-direct_selected_counted/summary.json), [Dense counted responses](../../diagnostics/generated/response_operator_20261006/evaluation/classic_Run1804_counted_responses/summary.json), [HONF1502 counted responses](../../diagnostics/generated/response_operator_20261006/evaluation/classic_Run1502_counted_responses/summary.json), [flow startup history](../../diagnostics/generated/rdirect_readiness_20261007/startup/D-sep-startup/history.json), and [thermal startup history](../../diagnostics/generated/rdirect_readiness_20261007/startup/R-direct-startup/history.json). These are locally usable links; generated assets are deliberately absent from Git under the repository artifact rule.

### Delivery and next steps

No new reference solves were attempted: the ledger remains 326/326. GPU-associated time includes both successful flow children, the failed initialization, read-only classic/precision replays and startup processes; the local closeout receipt accounts for the bounded round below 8 GPU-associated hours and 6 elapsed hours. Only GPUs 0 and 2 were used, in the ModularDT environment. No R-group retraining, forced K, inverse/Wind campaign, formal3501/3502 restart or automatic5000 launch occurred. Scientific raw arrays, checkpoints, historical failures and security controls remain preserved; only superseded newly generated 12-panel visual copies were replaced with readable six-panel pages.

Next predictor step is the manual fresh full-TRAIN comparison with exact component ages, canonical89 and original90 evidence, and explicit peak/vorticity/geometry limitations. Next organizer step, if separately authorized, would measure actual execution work at the same fidelity rather than infer it from local K. Next inverse step, if separately authorized, would require independent saved-response or physical design validation. None of those future studies is claimed as completed here.

Validation completed with **109 focused CPU tests passed**, Ruff and Python compilation on all 18 changed Python files, unchanged hashes for six retained selected/endpoint checkpoints, and `git diff --check`. The Markdown browser preview loaded all eight embeds with no horizontal overflow; figures and table screenshots were inspected, all local targets were checked, and every report prose paragraph/caption stays on one source line. The closeout at 0.648 elapsed hours records 0.0787 hours from measured inner GPU process receipts. To cover imports and uninstrumented overhead, charging both authorized GPUs for the entire round gives a conservative 1.296-hour ceiling, still below eight. See the [resource closeout](../../diagnostics/generated/rdirect_readiness_20261007/round_closeout.json) and [retained-model hashes](../../diagnostics/generated/rdirect_readiness_20261007/retained_model_hashes.json). Durable source/tests/profiles/docs are committed and pushed after a complete outgoing-history artifact audit; checkpoints, numerical arrays, figures and one-time renderers remain local and ignored.
