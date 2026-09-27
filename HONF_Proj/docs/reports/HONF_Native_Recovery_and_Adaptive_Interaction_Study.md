# HONF native response recovery, adaptive interaction, and inverse utility

**Working branch:** `agent/honf-core-next`
**Experiment basis:** `UpgradePlan/HONF_Native_Recovery_and_Adaptive_Interaction_Plan.md`
**Status:** bounded native response and WindFarm graph studies are still being completed; the physical and inverse observations below are measured.
**Evidence sources:** ThermalChannel local NumPy analytic-wake/shared-grid reference, intact native HONF checkpoints, and separate synthetic mechanism tests. The local reference is not Navier–Stokes CFD.

## Scientific questions and identities

The native Thermal response incumbent is Run 1804 selected epoch 4738 (`dense_pairwise_field`, full learned head and local-port path; checkpoint SHA-256 `71ed480ff0396813491c650dd11d887195174019b373fbd9a1fb25505142c066`). Run 1502 selected epoch 4794 (`sparse_incidence_group_control_honf`) is the native comparison. The WindFarm graph host is the intact Run 2103 selected epoch 2475 (checkpoint SHA-256 `e5cfcc1487b20b749ce2bf5e0ddaab295b0f6f5acd35244ba8ba5a39e1c811d4`). This round does not transfer their heads into a new three-term architecture. The failed Run1508 conversion and older factor-only inverse oracle remain historical controls.

The three errors studied separately are (1) absolute native prediction versus physical reference, (2) finite native response versus a measured local response, and (3) additional graph-organization distortion versus the *same frozen native teacher*. A small teacher error alone does not certify physical accuracy. A small logical cover does not certify less executed work or a physical causal edge.

## Physical response evidence

The existing train-only response atlas contains eight nominal-grid families: two each at M=3, 5, 7, and 10. The two M3 families have Re=50 and 70; the remaining six have Re=70. The eleven states within each family are correlated perturbations, not eleven independent layouts. Under the fixed benchmark `1.05 × each measured baseline pressure drop`, six of 88 states cross the pressure boundary: three each in families 0318 and 0335. Only two independent training families supply this contrast, so these rows do not establish a reliable false-feasible rate. The four legacy calibration neighborhoods are operating-context transfer at Re=90 and remain separate from training evidence.

### Fixed geometry, fixed total heat

Before new calls, an ignored manifest froze four transfers from the hottest module 2 in train family 0001 to module 0 or 1. Geometry, operating context, total heat (4.198839 in dataset heat units), and all module identities stayed fixed. All four new local-reference attempts converged and were persisted, including raw solver directories. The nominal-grid baseline peak is 18.866768 in dataset temperature units and the pressure drop is 0.074143511 in dataset pressure units.

| Recipient | Heat transferred | Measured global peak change | Measured pressure change |
|---|---:|---:|---:|
| Module 0 | 0.125 | −1.191860 | 0 |
| Module 0 | 0.250 | −2.367622 | 0 |
| Module 1 | 0.125 | −1.191860 | 0 |
| Module 1 | 0.250 | −2.383728 | 0 |

This is a local diagnostic of the analytic-flow generator. It isolates a useful thermal sensitivity and a pressure null at fixed geometry; it does not imply that general multiphysics pressure is heat-invariant. The four persisted records are under the ignored `diagnostics/generated/native_recovery_20260926/fixed_heat_train0001/` directory.

### Cross-context response and pressure boundary

Two frozen train layouts, family 0318 (M=5) and family 0350 (M=10), were also solved at Re=90 with the same baseline and `i_minus` position move used at Re=70. The new records are tagged **calibration only**; they were not added to either fit arm. All four CPU analytic-wake/shared-grid calls converged on the original `128 × 64` grid, with `u_in=1`, `nu=0.01`, the original material parameters and heat inputs, and unchanged module identities. The measured driver used 24.946 s CPU and 24.952 s wall time. The full role-aligned arrays, raw solver directories and per-call attempt ledger are preserved under the ignored `diagnostics/generated/native_recovery_20260926/cross_context_train_layouts_re90/` directory.

| Train layout and move | Re70 baseline → trial pressure | Re70 trial / own baseline | Re90 baseline → trial pressure | Re90 trial / own baseline |
|---|---:|---:|---:|---:|
| 0318, module 2 x −0.10 | 0.090655 → 0.096265 | 1.06189 | 0.080119 → 0.085730 | 1.07003 |
| 0350, module 3 x −0.15 | 0.105954 → 0.111119 | 1.04875 | 0.095419 → 0.100584 | 1.05413 |

Pressure uses the maintained fluid-only inlet/outlet 8% bands and each trial is tested against `1.05 × its own context baseline`. Family 0318 exceeds the limit in both contexts; family 0350 changes from just feasible at Re70 to infeasible at Re90. The absolute pressure baseline shifts by about 0.01054 pressure units in each layout, while its finite increment differs by at most `7.2e-10`. Across all aligned role channels, the largest pointwise Re70-versus-Re90 finite-increment difference is `4.77e-7`; interface/solid temperature and per-ID peak increments are identical at saved precision. This is a narrow **generator behavior** result: the operating-context change alters the pressure denominator and thus the feasibility label, but the local finite responses are effectively invariant. It is not independent physical validation of context transfer.

### What the anchored physical interaction explains

The separate [Thermal physical interaction evidence](HONF_Thermal_Physical_Interaction_Evidence.md) compares the same four fixed-design states through `R_ij = F_both − F_i − F_j + F_baseline`, joining material receivers by physical module ID and local coordinate. On M3/train-0001, moving modules 0 and 1 by 0.15 produces mixed solid-temperature RMS 0.018769 and 0.020093 on those *moved receivers*. Their conservative ratios to the family-level doubled-grid discrepancy are at least 4.03 and 4.31. The hot spectator module 2 has mixed solid RMS `7.41e-7`, below the corresponding numerical-discrepancy bound, and remains the module that sets the sampled maximum temperature. The interaction is therefore resolved on two receivers but does not explain an improvement in the current peak objective. Its mixed pressure increment is about `−7.65e-10`, despite a nonzero unary x-move pressure increment.

The distinct M3/0304 h=0.15 pilot failed its predeclared interface/solid mixed-signal and peak-stability gates, so its 34-state coarse expansion stopped. A separate train-exposed M7/0340 calibration-development check found peak finite-move sign reversals after grid doubling even though its anchored mixed peak scalar was nearly stable. These are reference-solver and numerical-resolution findings; they neither identify a learned hyperedge nor supply a portable interaction floor for another layout. Their 84 historical attempt charges remain in the earlier coverage ledger and are not counted again as new-round attempts below.

### Native checkpoint preflight

The intact Run1804 e4738 one-update preflight used physical GPU 2 (`CUDA_VISIBLE_DEVICES=2`, logical `cuda:0`). It completed one attempted and one successful optimizer update in 5.71 s recorded workflow time. The actual AdamW group had exactly six trainable output-layer tensors (2,313 parameters) at learning rate `1e-5` and weight decay `1e-5`; 5,428,235 parameters remained frozen. The heating and position whole-wrapper AD/FD checks, target-poison invariance, and module-row permutation gate passed. Requested `max-fluid-queries=1024` yielded 1,742 actual fluid rows because the protected near-interface/pressure union itself contains 1,742. This is a native computation and optimizer check, not broad response recovery. The checkpoint top-level epoch is 4738; the initial ignored preflight manifest's `selected_epoch=null` is stale metadata from a nested null selection field, and the runner lookup has since been corrected. The original artifact remains preserved under the ignored `diagnostics/generated/native_recovery_20260926/native_preflight_run1804_gpu2/` directory.

### Exact native replay and heat-response control

Both complete native checkpoints were replayed with their own predicted-port/local paths at full stored receiver resolution: 12 response families (eight train and four Re90 calibration), 30 broad stratified historical training cases across module-count and Reynolds strata, and the four new fixed-total-heat designs. The full two-model replay took 121.145 s on GPU 2 and reached 99.6 MB peak PyTorch allocation with inference mode. No training update or new reference solve occurred in this replay. Run1502's selected e4794 checkpoint SHA-256 is `08d86a573c7f7d86463bde231eb9f84a745602fca2b8f97142f39540d33a85bb`. The ignored complete manifest is `diagnostics/generated/native_recovery_20260926/native_replay_R0_retry2/native_replay_R0.json`; two earlier failed startup/adapter attempts remain preserved separately.

| Native checkpoint | Mean per-module fixed-heat finite peak absolute error | Maximum spurious fixed-heat pressure change | Train-atlas absolute pressure MAE | False-feasible / reference-infeasible states | False-infeasible / reference-feasible states |
|---|---:|---:|---:|---:|---:|
| Run1804 e4738 | 0.108386 temperature units | 0.001152 pressure units | 0.000510 pressure units | 0 / 6 | 3 / 82 |
| Run1502 e4794 | 0.079348 temperature units | 0.000889 pressure units | 0.001022 pressure units | 0 / 6 | 6 / 82 |

The fixed-heat physical pressure changes are exactly zero under this local analytic-flow generator. Both surrogates instead introduce small nonzero pressure slopes. Run1502 has a measured advantage on this narrow thermal control; Run1804 has lower absolute pressure error on the training-atlas states. The six pressure-infeasible states come from only two independent families, so neither zero count estimates a false-feasible rate. The row-wise counts are descriptive of this frozen benchmark; broad replay and finite position responses are analyzed separately below.

The fixed-heat raw local-reference records and the old atlas align by receiver ID, role, validity mask and coordinates to at most `2.84e-8` in the solid local coordinates; fluid and interface coordinates match exactly. Their original quadrature conventions do not match (maximum absolute weight differences: fluid `0.99121`, interface `0.95582`, solid `0.99980`). The replay therefore used the atlas baseline weights only for a coordinate-aligned finite comparison, retained both untouched raw records, and did not construct a false exact-schema `ResponseStencil`. This weighting caveat applies to pointwise aggregate metrics; the independently reduced per-ID peaks and pressure quantities retain their declared scalar definitions.

For finite position responses, the table below first averages each role/channel's weighted RMSE over perturbation states *within* a family, then gives equal weight to each independent family in its split. Thus its eight training and four Re90 calibration contributions are not treated as 88 or 44 independent layouts. Values remain in the listed physical dataset units; the full manifest retains M, Reynolds number, perturbation label and step size per family.

| Finite role/channel (units) | Train Run1804 | Train Run1502 | Re90 calibration Run1804 | Re90 calibration Run1502 |
|---|---:|---:|---:|---:|
| Fluid temperature (dataset temperature) | 0.11206 | 0.11498 | 0.10882 | 0.11116 |
| Solid material temperature (dataset temperature) | 0.19606 | 0.18594 | 0.18699 | 0.19650 |
| Interface surface temperature (dataset temperature) | 0.20935 | 0.19851 | 0.20208 | 0.21232 |
| Interface normal-flux proxy (dataset heat-flux proxy) | 1.2699 | 1.1984 | 1.2566 | 1.2151 |
| Fluid pressure (dataset pressure) | 0.00089 | 0.00106 | 0.00096 | 0.00102 |

Run1502 has a narrow train/control advantage in some local thermal roles, while Run1804 is slightly better on several transferred roles and on the 30-case historical absolute panel (equal-case mean solid-temperature RMSE 0.2643 versus 0.2692; fluid-temperature RMSE 0.1493 versus 0.1548). These aggregate comparisons are descriptive; they do not resolve local port versus field-head error or establish a successful response refit.

### Native wrapper parity and phase/port diagnosis

A target-free comparison between the checkpoint-native `predict_case` path and the new response adapter passed every stored fluid, interface and material output and the raw P0/refined P2 port tokens on all eleven states of train family 0001 for each intact checkpoint (22 model-state comparisons). The maximum absolute difference was exactly zero for every checked tensor. This corrected replay used GPU 2, took 21.154 s of workflow wall time, and reached 18.10 GB peak PyTorch allocation. The first preserved attempt reported a false port-parity mismatch because its diagnostic compared a singleton-batched tensor to its unbatched form; the corrected artifact is `diagnostics/generated/native_recovery_20260926/native_replay_R0/native_port_intervention_gpu2_retry2.json`.

That original parity artifact has a null atlas hash. A separate ignored provenance sidecar binds its exact input to `train_0001_responses.npz` (SHA-256 `c34bc219daff2a1ebb02174b8c5e1498c96286460582e445286cceaff5499432`) and its companion JSON (SHA-256 `1808460b4ffe0b26b0769b3b5d83c0e156557bd8f8ad3f6e76564ad10b11efc9`), as well as the frozen producer and split manifest. The preserved GPU artifact itself was not rewritten.

The valid H5 intervention supplies exactly `T_outside` and `h_effective` at 192 active ports to P0. `T_surface` and the normal-flux proxy are separate outputs, not substituted port conditions. On the train-0001 baseline, the measured P0 inputs are reproduced exactly at P1 when P2 is disabled by construction. The following one-state comparisons distinguish P2's *model-side* effect from a measured physical P2 target:

| Model and input arm | P2 enabled: interface flux-proxy RMSE | P2 disabled: interface flux-proxy RMSE | P2 enabled: solid-T RMSE | P2 disabled: solid-T RMSE |
|---|---:|---:|---:|---:|
| Run1804, predicted P0 | 1.20412 | 3.29596 | 0.26354 | 0.37880 |
| Run1804, measured P0 | 4.46427 | 3.33201 | 0.31346 | 1.02659 |
| Run1502, predicted P0 | 1.28624 | 3.46834 | 0.27237 | 0.29295 |
| Run1502, measured P0 | 3.33690 | 5.02443 | 0.31668 | 1.02659 |

The P2 path materially affects flux and solid predictions, and supplying measured P0 alone can worsen them. Predicted P0 may be a coupled latent operating point for the trained refinement path. This one-family intervention does not establish that measured boundary conditions are wrong, that the local surrogate lacks capacity, or that P2 has a measured physical label. The broad native paired refit is therefore evaluated through the complete wrapper.

## WindFarm intact-model graph execution preflight

The Run2103 e2475 checkpoint was loaded without replacing its Dense field head. With the optional native policy absent, Dense preparation and reading remain direct. A root-only all-access policy reproduced the original physical velocity prediction exactly (`torch.equal=True`, maximum difference 0 m/s) on the same GPU 2 case and native query set. Its fresh input-only policy used current encoded/model-side inputs, prepared its tree and plan on logical `cuda:0`, and produced finite predictions. The 32 search cells and 32 verification cells were disjoint native-volume cells. The untrained policy's search-probe teacher RMSE was `3.08e-7` m/s and reference RMSE `0.0341` m/s; it retained 11/12 module and 512/512 environment source supports with 43 active sampled groups. These are plumbing observations, not a fitted adaptive-cover result. The ignored `Case_WindFarm/diagnostics/generated/native_cover_organizer/run2103_e2475/preflight_retry/preflight.json` manifest records the exact case and checkpoint.

The separate exact-checkpoint partial-plan probe changed native contextual module/environment states by maximum absolute values `0.509`/`0.866` and output by `0.845` m/s relative to Dense. It executed 729 MM, 14,308 ME, and 13,824 EM source rows, versus 756, 14,336, and 14,336 valid rows, respectively. At fixed weights, removing the native coarse, local, and both pathways changed prediction by maxima `0.824`, `3.137`, and `3.632` m/s. Those pathways therefore remain consequential dense information bypasses in this prototype, even when the cover masks fine transport. Logical K must not be presented as a complete physical dependency graph or as measured speedup. Reusable synthetic tests and this native intervention have distinct evidence roles.

### First real WindFarm native cover oracle

A frozen geometry-only twelve-layout train manifest selected the first 14-turbine layout and its three stored directions (270°, 285°, 300°) for a real oracle panel. Each row used disjoint Q1024 native search and Q1024 verification queries, with a predeclared maximum `0.10` teacher-relative normalized vector RMSE in each protected role. The first two job attempts stopped on a feature-width mismatch and an observation-serialization error; their diagnostics remain ignored. The corrected third job reused the 32 valid first-pass observations and completed 198 unique actual forward-evaluated proposals, 66 per row, below the 4,096 round ceiling. These are train-only teacher-preservation observations, not reference-sufficient physical interaction labels.

The intact no-policy Dense path took about 9.4–10.8 ms for one Q1024 prepare-and-verification-decode call; an explicit all-access cover, whose outputs were exactly identical to Dense, took 46.95–58.23 ms in the retained direct timings because it still incurred policy/tree overhead. Thirty-two one-block environment prunes per row met the teacher distortion limit, each preserving all 14 modules and 480/512 environment sources at root K=1. Their cheapest estimated complete time was 104.12 ms, but the directly synchronized selected partial calls took 303.70–311.93 ms across the three rows. All fourteen one-module prunes per row violated the teacher tolerance. Root/child split proposals in the bounded second pass also violated it. Thus the estimated work model overstated a useful saving, and the first measured deployment choice remains the intact Dense path. A coherent teacher-adequate partial plan still exists as a possible organizer label; it is not the measured faster deployment plan.

The first row's input-anchored candidate tree used 750 receiver anchors: 512 environmental anchors with total tree-partition mass 1, and 238 rotor-neighborhood anchors with total mass 14 (hub 3.5, interior 3.5, edge 7). This 14:1 rotor-to-environment split is an explicit *tree-index measure*, not the source quadrature or reference integration measure. It can bias geometry cuts toward rotor anchors and motivates one bounded role-balanced index check. The first oracle's result is preserved under its original measure.

A separate CPU-only input-organizer target assessment on this frozen artifact returns `target_unavailable` and records **zero optimizer updates**. The twelve-layout manifest was preserved, but only one layout's three direction rows were activated. Ninety-six root-prune proposals passed the search-probe teacher gate; none beat the intact no-policy Dense path on its directly synchronized complete call (accepted-proposal ranges by row: 310.80–411.85, 303.70–370.81 and 311.93–373.48 ms, versus Dense 10.84, 9.86 and 9.45 ms). No coherent partial winner with both search and disjoint verification gates was retained as a fitted label after the cost override. This is an **execution-target** failure on a limited train sample, not evidence that all partial covers distort the teacher. The ignored `fit_assessment_cpu/fit_target_assessment.json` retains the checkpoint hash, label availability, costs and stop reason. One cached-execution retry and one role-balanced tree-index check remain bounded remedies before the graph-learning conclusion is closed.

For this identical all-access selected policy, full native-grid reference errors on the three direction rows were vector relative L2 `0.001306/0.001351/0.001469`, wake/background Ux-residual relative L2 `0.03557/0.04046/0.04045`, and near-turbine vector relative L2 `0.005902/0.005999/0.006749`. Selected and G-full predictions and reference errors were identical; differences between their whole-grid elapsed times were execution-order effects, not a graph speedup. The coarse/local learned branches remain active and source-layout dependent. Even a teacher-preserving partial fine cover cannot yet be called a complete physical dependency graph. The preserved manifest is `Case_WindFarm/diagnostics/generated/native_cover_organizer/run2103_e2475/oracle_layout1_gpu2_retry2/oracle_benchmark.json`.

## Bounded native inverse negative control

The train-exposed family 0318 task was frozen before native scoring. It minimizes the global maximum sampled material-module temperature subject to exact non-overlapping disk geometry and a fixed pressure limit of `0.095187443 = 1.05 × 0.090654707` from the original reference baseline. Each of three policies had four sign combinations, two changed scalar position coordinates per candidate, a per-coordinate trust radius of 0.1, the same complete Run1804 e4738 absolute predictor for the baseline and every trial, the same per-module and pressure baseline correction, and one selected reference observation. The native operator recomputed its continuous state from each trial design; no second trial-design surrogate was called. All sixteen synchronized native forward calls, including the shared cold-start baseline diagnostic and five per policy, took 4.644 s; the full job took 12.571 s and reached 62.3 MB peak allocated GPU memory. All model selections were persisted before any selected trial reference was accessed.

The `graph_guided` search group uses the existing train-0318 stencil's module-2 x and module-3 y coordinates. It is a predeclared physical-response search group, **not a fitted native Thermal cover**. A seed-20260926 random group used module-4 x and module-3 x; the ungrouped local control used module-4 x/y. The task family and group were selected using training-family information, so this is a bounded train-exposed negative-control trial, not a held-out comparison or deployment claim.

| Policy | Selected signs | Corrected predicted peak gain | Nominal reference peak gain | Nominal reference pressure | Original-grid acceptance |
|---|---|---:|---:|---:|---|
| Stencil-nominated joint group | `+1−1` | 0.111664 | 0.371696 | 0.086780 | Feasible improvement |
| Size-matched random group | `+1+1` | 0.038248 | 0.014093 | 0.091939 | Feasible improvement |
| Single-module local group | `+1+1` | 0.430662 | 0.005190 | 0.091867 | Feasible improvement |

The joint-group selected reference is an existing stored train-0318 `pm` solve, counted as one reference observation with zero new solver attempts. The random and local selections each invoked one new persisted local-reference solve. The original model had baseline peak 24.210480 versus measured 24.559031 and baseline pressure 0.092071 versus measured 0.090655. Per-module correction removes those baseline offsets at the anchor, but the local group's predicted finite gain still overshot its nominal measured gain by about 0.425 temperature units. All three selected nominal outcomes were feasible against the unchanged pressure limit.

For pressure, Run1804's cold-start baseline bias was `+0.001416` pressure units. The corrected selected-trial predictions were 0.087110, 0.091892 and 0.092185 for joint, random and local, versus reference 0.086780, 0.091939 and 0.091867. Their finite-increment errors were `+0.000330`, `−0.000047` and `+0.000318` pressure units. This separates a removable anchor offset from remaining finite-response error; the observed selected candidates stayed below the unchanged 0.095187 pressure limit.

The nominal actual-to-predicted improvement ratios were 3.329, 0.368, and 0.012 for the joint, random, and local policies. The ordinary bounded trust rule consequently suggested next radii 0.15, 0.10, and 0.05 from the shared initial 0.10. These were recorded suggestions only: no second accepted anchor or unverified follow-on design was silently introduced.

The selected graph pair's measured anchored mixed peak response is about −0.000710 temperature units, and its mixed pressure response is approximately `1.4e-11`. Its useful move combines a pressure-improving x step and a peak-improving y step. It does not demonstrate a non-additive field law or a many-body thermal interaction.

### Matched doubled-grid check

The common baseline and all three already-selected designs were independently solved at `256 × 128`, after freezing the four refinement inputs and the original pressure limit. All four attempts converged and were persisted. The predeclared resolution diagnostic requires the same finite-change sign and `|2× change| ≥ 3 × |nominal change − 2× change|`; it is a two-grid screen, not a convergence proof.

| Policy | Nominal peak gain | Doubled-grid peak gain | Absolute change discrepancy | Three-times-discrepancy rule | Doubled-grid pressure |
|---|---:|---:|---:|---|---:|
| Stencil-nominated joint group | 0.371696 | 0.391354 | 0.019657 | Pass | 0.086821 |
| Size-matched random group | 0.014093 | 0.041718 | 0.027624 | Fail | 0.091989 |
| Single-module local group | 0.005190 | 0.618942 | 0.613752 | Fail | 0.091917 |

All four refined reference states, including the baseline, satisfy the original pressure limit. The joint group's improvement is stable under this specific grid doubling. The local control's nominal response is highly mesh-sensitive and exceeds the joint-group gain at the doubled grid, so no cross-resolution winner is established. The reference remains an analytic-wake/shared-grid model, with no external CFD validation. Full per-module temperatures, raw solver paths, frozen selections, and costs are retained under the ignored `diagnostics/generated/native_recovery_20260926/native_inverse_0318/` directory.

## Resource ledger so far

This new round has charged fourteen new local-reference solver attempts: four fixed-heat diagnostic, two nominal inverse, four matched grid-refinement, and four crossed-context calls. All fourteen converged and were persisted. Their aggregate measured process CPU time is 88.626133 s, far below the new-round ceiling of 192 attempts and four aggregate CPU-process hours. A canonical-directory reconciliation found that an earlier consolidated generator omitted ten of these fourteen calls. The corrected ignored all-history ledger now records **222 unique converged raw case directories out of 320**, with 78 previously reserved and **20 unreserved remaining**; the new-round subledger records **14/192**. The prior 212-row ledger, summary and generator were backed up before the correction. Seven older calls still lack per-solve timing, so the consolidated known-time sum of 1,032.869286 s is incomplete and is not an aggregate process-runtime claim. The stored baseline and stored joint trial are reused reference observations, not new solver calls. Native preflight and inverse scoring both used only physical GPU 2. No automatic 5,000-epoch continuation was launched. Earlier physical-atlas and failed-refit runs remain separate historical costs.

The first WindFarm oracle charged 198 unique actual native candidate observations, including 32 observations reused exactly once from its failed serialization attempt, against the 4,096-observation ceiling. Its corrected job took about 134.1 s of shell wall time, including startup; the recorded mean synchronized candidate complete call was 328.56 ms. A later cold/warm cache attempt made 66 further candidate forward evaluations on row 12 before stopping at a failed bitwise repeatability assertion; these calls are charged, bringing the observed round count to 264, and the failed output remains preserved. The corrected Thermal native parity/phase job took 21.154 s workflow wall (25.2 s shell wall). These job times and nested synchronized calls are distinct scopes and are not summed as if independent. No WindFarm CFD solve was introduced.

## Current hard limits

- The physical reference is a local analytic-wake/shared-grid generator; its pressure and flux conventions are documented, but it does not supply Navier–Stokes CFD credibility or a deployment pressure allowance. The `1.05 × baseline` pressure boundary is a frozen experimental benchmark.
- The two-layout Re90 crossed-context panel reveals essentially invariant finite responses in this generator. Its changed pressure feasibility comes from the context-dependent baseline; it does not establish general Reynolds transfer of physical interactions.
- The nominal layout trial is a single train-exposed family. A predeclared group and a physically useful joint move do not establish a learned adaptive graph's design utility.
- The local move changes sharply across the two grids. Neither the nominal comparison nor a two-grid pass licenses continuous shape-gradient optimization at this move scale.
- Exact native adapter parity and broad untrained replay are established. They do not establish that the bounded matched response fit improves finite decisions; the trained review is still pending.
- The first WindFarm teacher oracle retained full access as its measured execution choice. Adequate partial root prunes exist but are slower, no adequate split was found in this one layout, and an input-only organizer has not yet been validated. Dense coarse/local paths still carry layout information outside the fine cover.

## External-reference information needed for a stronger WindFarm design claim

The stored native velocity fields support finite-library and teacher-preservation studies. A new-layout physical-response claim needs the external solver or generator and its runnable input/output contract: the coordinate frame and wind-direction transform; rotor or actuator geometry and force convention; controllable turbine variables; inflow, outlet, terrain and turbulence boundary conditions; and the definition of the scalar design objective, including any power/AEP calculation. Until those are available, a turbine-neighborhood velocity statistic remains a velocity statistic, and `wake_loss_pct` remains a stored empirical label rather than a derived power law.
