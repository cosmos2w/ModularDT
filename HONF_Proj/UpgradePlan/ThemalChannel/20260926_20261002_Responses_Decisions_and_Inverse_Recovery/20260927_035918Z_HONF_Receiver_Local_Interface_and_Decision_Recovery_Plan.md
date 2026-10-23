# HONF next round: learn a receiver-local interface, recover useful responses, and separate structure from execution

**Branch:** `agent/honf-core-next`  
**Reviewed revision:** `6394e546dd2bc7a39548b8aa52dead239c3efd16`  
**Execution mode:** Codex Goal mode, within the explicit budgets below  
**Primary evidence:** `docs/reports/HONF_Native_Recovery_and_Adaptive_Interaction_Study.md`  
**Status of this document:** proposed work, not completed experiments or a claim that the proposed model will succeed.

## 0. The goal and the three changes that matter

Build an input-conditioned, receiver-local organization of the intact forward model that can preserve both useful field predictions and the finite changes needed by inverse design. Train it on native cases. Separately determine whether its execution is economical. Keep the independently checked, baseline-corrected inverse protocol.

Three changes take priority over further infrastructure development:

1. **Replace global source deletion and aggressive nearest-child splits with incremental, directed, mechanism-specific source-to-receiver interventions.** A turbine needed somewhere in a field is not necessarily needed at every receiver. A failed global deletion is not a negative label for every local edge.
2. **Separate structural learning eligibility from deployment latency.** The six verified partial covers are teacher-preservation labels, although their current executor is not deployable. Fit a bounded native organizer and evaluate its hard predictions; do not require a faster executor before any learning can occur.
3. **Replace last-layer-only response adaptation with a controlled nonlinear interface adaptation, and activate finite decision and pressure objectives inside the first efficacy review.** The completed native fits changed only six terminal tensors and never activated the decision/constraint stage. They close that recipe, not the forward-model capacity question.

Preserve the native checkpoints. Do not restart the three-term conversion, resume the rejected head-only checkpoints, reopen the old coalescence sweep, or launch another fixed 5,000-epoch run.

This plan explicitly supersedes the preceding report's recommendation to make faster native partial execution a prerequisite for organizer fitting. It does **not** relax deployment evidence: production WindFarm inference remains policy-free Dense until a separately timed candidate earns promotion.

### 0.1 Required empirical outputs

The round should produce:

- one matched nonlinear native response experiment, with decision-relevant objectives actually active;
- a directed local-intervention dataset and at least one genuinely fitted input-only organizer, including deterministic native evaluation;
- an honest map of what the graph controls, what still passes through other branches, and how its physical-source dependencies propagate;
- a measured logical-sparsity/fidelity/runtime frontier, not a single K number;
- a small independently checked inverse comparison only when reference accounting, model behavior, and numerical resolution permit it.

A constant-K fitted model, a slow but faithful graph, or a negative nonlinear-adaptation result may be valid research endpoints. None is renamed a successful adaptive physical interface. However, another zero-update organizer caused solely by a speed gate is not an acceptable completion of this plan.

## 1. What the evidence now establishes

### 1.1 Findings to preserve

The complete native Run1804 and Run1502 adapters match the checkpoint-native wrapper exactly on the tested 22 model/state comparisons. Historical value replay and actual optimizer updates work. The reference-corrected inverse path recomputes the intact predictor at every candidate. These are useful, established implementation assets, not tasks to rebuild from scratch. [E1, “Native wrapper parity” and “Native paired update”]

The Thermal response trials trained 2,313 parameters in the terminal field, initial-port, and refinement layers, leaving 5,428,235 frozen. Each learning-rate trial used 100 updates per arm; only updates 21–100 contained the finite-response term. Decision and pressure-constraint losses were not active. The tiny response gains and several absolute regressions are therefore evidence against **this parameter scope and objective schedule**, not against all native response learning. [E1, “Matched native response review”; C1, C2]

The P0/P2 intervention shows co-adaptation: substituting valid measured `T_outside` and `h_effective` at P0 can worsen final outputs. Preserve these measurements. Do not interpret them as proof that the physical boundary values are wrong or that the local surrogate has insufficient capacity. P0/P2 tokens may be serving partly as learned coupling coordinates in the present model. [E1, “Native wrapper parity and phase/port diagnosis”]

The WindFarm search found six coherent partial K=1 covers with all modules and 480/512 environmental sources. They passed disjoint teacher checks but were much slower than policy-free Dense. The whole-grid reference table in the report evaluates the retained **full** policy, not those partial covers. Partial-cover physical fidelity is not established by copying the full-policy reference errors. [E1, “Bounded cache and tree-index remedies”]

The useful train-0318 joint inverse move combines a pressure-improving coordinate with a temperature-improving coordinate. Its anchored mixed peak is approximately −0.000710 and its mixed pressure is approximately zero. A useful coordinated design move does not require a large non-additive physical interaction. Its group was stencil-nominated, not learned; the local comparator changes sharply between grids. [E1, “Bounded native inverse negative control”]

### 1.2 Concrete code findings

| Location | Inspected behavior | Consequence for this round |
|---|---|---|
| `response_control/runner.py::_configure_native_output_head_scope` | Selects the last `Linear` in each of three heads. | Add an explicit nonlinear scope; do not describe the previous experiment as a broad interface refit. |
| `configs/response_control_native_staged.json` | Decision and constraint terms start at completed update 100. | New schedule must train them before its first decision review. |
| `response_control/losses.py::compute_stencil_loss_terms` | Decision term supervises absolute per-module/smooth/global peaks; finite pressure is separate. | Add explicit finite **per-module** peak supervision derived from the same absolute model. |
| `native_cover_organizer.py::_native_cover_proposals` | Removes modules at the root globally; geometric splits assign sources to nearest children. | Test local receiver blocks and one mechanism at a time before joint omissions. |
| `native_cover_organizer.py::run_oracle_benchmark` | Proposal closure returns candidates only once per pass; both passes restart from full access. | Implement bounded sequential composition and backtracking; repeated independent one-block probes are not cumulative sparsification. |
| `adaptive_cover_field.py::_prepare_cover_fine_messages` | Uses one cover to select MM, ME and EM transport; query reads use the same cover. | Separate permissions for different directed mechanisms and phases. |
| `native_cover_organizer_fit.py::_assess_fit_target` | Requires faster-than-Dense verified partial labels on at least two layouts and label variation. | Split learning, physical validation, and deployment eligibility. Constant labels are permissible controls. |
| `native_cover_organizer_fit.py::_training_loss` | Fits masked classification labels only. | Add native predictive/distillation feedback after the supervised warm start. |
| `input_cover_organizer.py` | Node/pair input widths assume 3-D (`6`, `+3`). | Parameterize spatial dimension before using it on ThermalChannel. |
| `native_cover_organizer.py::_plan_supervision` | Marks only root/first children and root split observations. | Use recursive active-frontier export before claiming deeper learned K. |
| `adaptive_cover_field.py` | Partial paths use packed gather/scatter kernels; `dense_environment_fast_path=False`. | Implement a rectangular subset/block execution option instead of assuming every partial plan needs scalar-pair packing. |
| `adaptive_cover_field.py::_case_tree_cache_key` | Hashes GPU geometry through host copies; gradient-bearing geometry bypasses the cache. | Profile this overhead; distinguish frozen combinatorial topology from live geometry. Never cache changing physical states. |
| `inverse/native_corrected.py::ThermalNativeQuantityPredictor` | Explicitly rejects a nonempty topology argument. | Thermal inverse currently cannot consume a real forward cover. Implement this boundary, not just a differently named inverse group. |

These are inspected source behaviors at the reviewed revision. Their existence does not establish which component dominates a measured error or latency. Obtain a short profile/ablation before assigning causation. [C1–C10]

### 1.3 One immediate arithmetic limit

The selected root partial removes 32 of 512 environment sources: 6.25%. Even if every millisecond were proportional to those rows and organization were free, retaining 480/512 would allow at most `512/480 = 1.0667` times speedup in that hypothetical model. Unchanged branches make the attainable total saving smaller. This is an upper-bound calculation, not a measured GPU prediction.

It cannot pay for a roughly 20–39-fold complete-call slowdown. Both the amount of sparsity and the execution mechanism need attention; tree caching alone is not the main scientific answer.

## 2. Define the interface before optimizing K

### 2.1 Three related objects, not one ambiguous “graph”

Maintain three explicitly named views of a shared physical-source catalogue.

**Transport view.** Which source states enter a particular receiver's update at a particular stage? An edge describes an actual computation: module-to-module, environment-to-module, module-to-environment, or a field/port read. Its absence removes that direct message, not necessarily every indirect effect of the source.

**Response view.** Which design coordinates affect a named physical receiver quantity over a declared finite trust region? This includes all model pathways and any measured reference evidence. A direct transport edge and an effective design derivative are not the same object. Multihop paths can make effective responses dense even when local transport is sparse.

**Decision view.** Which coordinate groups are useful for reducing the objective while respecting constraints? It is derived from response information plus the current objective, pressure margin, geometry, and uncertainty. It may combine almost independent effects. Label a group `constraint_coordination`, `shared_receiver`, or `resolved_mixed_response` according to its evidence; do not call all of them many-body physics.

A compiler may have a fourth, internal scheduling view. Its buckets and tensor tiles are implementation objects, not extra physical hyperedges.

The shared catalogue keeps module identities, receiver roles, coordinates, units, state dependence and evidence provenance consistent across these views. It does not force all views to have the same K.

### 2.2 The hyperedge needed in the forward model

Represent a directed interaction packet as

\[
e=(\ell,\mathcal S_e,\mathcal R_e,\alpha_e,\mathcal U_e),
\]

where `ell` names the mechanism and phase, `S_e` contains physically identified source modules or environmental supports, `R_e` is a receiver support, `alpha_e` is its access/retention function, and `U_e` is the declared validity neighborhood when the plan is frozen for an inverse step.

A packet must not replace its source set by a single averaged field value before the required fine interactions are evaluated. Preserve the native fine source/receiver functions and their nonlinear collective update. A generic form is

\[
 m_r^\ell=\sum_s \mu_s\,w_{rs}^\ell
 \phi_\theta^\ell(z_r,z_s,x_r-x_s,c),
 \qquad
 z_r^+=\Psi_\theta^\ell(z_r,m_r^\ell,c).
\]

This form can support collective response even when the message kernel is pairwise. It does not, on its own, identify an irreducible physical many-body law.

For the attention reader preserve the original unique-source normalization:

\[
 C_r=\frac{\sum_s\mu_s w_{rs}\exp(a_{rs})V_s}
 {\sum_s\mu_s w_{rs}\exp(a_{rs})}.
\]

Overlapping packets share one deduplicated source union and one denominator. Do not compute independently normalized attention in each group and then average it while claiming native equivalence.

For the existing receiver-cover algebra,

\[
 w_{rs}^\ell=\sum_e\alpha_{re}^\ell b_{es}^\ell,
 \quad\alpha_{re}^\ell\ge0,\quad\sum_e\alpha_{re}^\ell=1.
\]

Use this representation initially. A geometrical tree is an index of candidate receiver supports, not the claimed learned physics. Its current candidate family can be retained while testing better permissions.

### 2.3 What K means

Report, separately, candidate capacity, active frontier nodes, nonredundant source-bearing packets, unique physical source/receiver pairs, and executed tensor rows.

For an exact-support cover, define `K_packet` after removing unreachable nodes and merging equivalent packet actions where that merge is algebraically valid. Two children with identical source permissions whose access functions sum to the parent's access do not create a new physical distinction merely because the tree was split. Keep the raw frontier count as a separate diagnostic.

Report K by mechanism/phase and case. Also report source/receiver degree, packet source size, overlap, and source-union work. Do not minimize K alone, prescribe a K histogram, force K>1, or count changing padding as adaptation. A globally needed source can be present in the overall union while absent from many receiver packets.

### 2.4 Explicit global paths are allowed; invisible bypasses are not

Audit the coarse, local, geometry-feature, global-context and physical refinement routes. Mark which design variables each can depend on and whether a proposed mask actually controls that dependency.

A legitimate model may have local sparse transport plus a declared global channel. Such a model is not invalid simply because it retains long-range dependence. However:

- an unrestricted layout-conditioned bypass cannot be omitted from a claim of sparse physical dependency;
- a frozen baseline context and a live trial-layout context have different meanings in a local response model;
- removing one direct message does not prove the originating physical module is irrelevant;
- low-rank/global communication must be measured and named rather than silently called “background.”

First retain the mature pathways and quantify them. Then expose permissions on the consequential route one route at a time. Do not delete the native coarse/local branches wholesale to make a diagram look sparse. If the graph still controls only a secondary pathway, call the result `partial_transport_organization` and stop short of a complete physical-interface claim. [E1, “WindFarm intact-model graph execution preflight”; C9]

## 3. Baselines, evidence partitions and accounting

### 3.1 Fixed model identities

Keep Run1804 e4738 as the Thermal absolute incumbent, Run1502 e4794 as the separate sparse/fixed-heat control, and WindFarm Run2103 e2475 as the graph teacher/host. Verify the report's hashes once. Do not silently substitute endpoint aliases or change normalization.

New trained states use new managed run identities. Native all-access behavior is the initialization reference; trained predictions need not remain identical, but any change must be measured.

### 3.2 Existing data first

Use the eight existing Thermal training families and broad historical training source. The four Re90 neighborhoods remain response-development/calibration, not unseen-layout physical generalization. Keep the added Re90 layout checks out of the training pair to preserve their reported role. Group duplicate layouts, all stencil corners, directions of a layout, and grid refinements by physical family.

For WindFarm activate the existing 12-layout geometry-frozen panel rather than staying on two layouts. Before reading new organizer outcomes, designate eight layouts for organizer training and four for organizer development, keeping all three stored directions of a layout together. The layout split may be chosen to cover the available turbine-count range, but not based on field errors or pruning outcomes. Include same-M distinct layouts whenever available. Existing dense-teacher exposure is declared: “held out from organizer training” is not “unseen by the pretrained forward model.”

Search and verification queries must be disjoint. Once verification outcomes influence a new recipe, that set is development evidence; reserve another query set or use full native cells for the final fixed-policy check. No source labels are inferred from targets at deployment.

The previously opened Thermal final-review stencils and old 90-case development/test split are not fresh final evidence. Reusing them for development is allowed with that label.

### 3.3 Reconcile physical calls before any new reference solve

There is an unresolved cross-report accounting discrepancy: the latest native study reports 222 unique converged directories, 78 reserved calls and 20 unreserved calls under 320, whereas the earlier interaction/inverse study describes 286 executed attempts including 78 completed inverse/common-pool trials. These numbers may have different reconciliation scopes; they must not be added, substituted or treated as an established fresh budget. [E1, “Resource ledger”; E2, “Results, resource accounting, and decision”]

Reconcile immutable input/output hashes, record IDs, original attempt records, failed attempts, recovered outputs, canonical directories across all historical roots, and genuinely outstanding reservations. An attempted solve is charged even if a directory is missing or an output failed serialization. Identical reused results are observations, not new attempts. Distinct reruns with identical inputs still consume attempts.

Produce one read-only reconciliation plus an append-only corrected ledger. Preserve old records and explain differences. Until this is complete, `new_reference_attempts_allowed=0`; neural fitting and stored-field graph work continue.

The new local-reference allowance is at most 20 attempts, and only within the verified uncommitted remainder of the standing global cap. Do not release old reservations or raise the 320-call cap automatically. Missing historical timing remains missing.

## 4. Workstream R — train a response-capable nonlinear native interface

### R0. A short train-only fitting diagnosis

Do not repeat the full historical audit. Reuse the established parity artifacts and run only changed-path regression checks.

Select four training families spanning M, including an M10 family showing gradient conflict. On their native sampled roles, measure gradients of historical value, stencil value, finite field, finite per-module peak and pressure terms. Record both norm and direction by trainable block. Derive scales from the whole training panel with equal-family weighting, not the first family alone. Keep physical-unit error tables in parallel with normalized objectives.

A small train-only fitting probe is permitted to distinguish poor parameter reachability from poor transfer. Its purpose is not promotion. Compare the current terminal scope with the nonlinear scope below on the same small batches for at most 100 updates each. Reusing old terminal-scope results is acceptable only where the objective and sample schedule truly match; otherwise label them historical rather than matched.

Optional diagnostic, only if straightforward with existing autograd: fit a regularized linearized correction in a small subspace of selected parameter directions, using response residuals plus value-drift rows. This estimates local reachability; it is not a capacity theorem. Do not build a full output-by-all-parameter Jacobian or a new derivative-data pipeline for this diagnostic.

### R1. Primary candidate: `native_nonlinear_interface`

Start afresh from intact Run1804 e4738. Make the trainable scope explicit and serializable:

- all layers of `local_coupling.port_head`;
- all layers of `local_coupling.port_refinement_head`;
- all layers of `core.common.field_head`.

Keep the encoders, fine transport kernels, coarse/local context builders, local surrogate and all other tensors frozen initially. Freeze any associated running-statistic buffers and preserve appropriate evaluation modes, not just `requires_grad=False`. Check dropout or other stochastic layers: response stencils must use a declared deterministic mode or matched randomness so model-side sampling noise is not mistaken for a physical finite change. Apply the same convention in both arms. Preserve every native tensor and its function at update zero. This is a nonlinear extension of the previously tested scope, not a new backbone or a head replacement. Record actual parameter counts; do not assume them from names.

The port/refinement changes can affect all final fields through feedback even though much of the core is frozen. Check the complete wrapper. A frozen fine kernel is not a guarantee of unchanged pressure or velocity.

Use one train-only learning-rate calibration on the small probe, choosing between at most two modest rates. Freeze the choice before the matched development review. Do not spend two full experiments again merely changing the rate of the same six terminal tensors.

**One contingent remedy, not a sweep:** if this scope cannot improve training finite responses appreciably, inspect the measured gradient route. Allow one small zero-output-initialized residual adapter on the receiver-conditioned messages supplying the port/refinement interface context, using existing receiver/source states and normalized relative geometry. Prefer a source-resolved correction before the existing reduction, so it can change interaction sensitivity rather than only add an output bias. A low-width residual `W_up sigma(W_down features)` with `W_up=0` at initialization is an acceptable implementation; keep its physical-source permissions explicit. Freeze the incumbent backbone, report the new parameters, and verify exact zero-adapter parity. Do not simultaneously unfreeze the entire local surrogate and redesign the environmental backbone. A local-surrogate adaptation requires a separate, positive controllability diagnosis; measured-P0 substitution on one family is insufficient.

### R2. Use the same absolute function for all responses

Let `F_theta(d,c)` be the complete native operator, including any approved adapter. Always compute

\[
 \widehat{\Delta y}=F_\theta(d+\delta,c)-F_\theta(d,c).
\]

An optional residual adapter changes the absolute function, for example `F_theta=B+R_theta`; it does not create an unrelated delta predictor. No independent scalar head may claim to fix the field's pressure or temperature while disagreeing with the maintained reduction of that field.

Keep Eulerian differences on aligned common-fluid masks and material differences on physical module IDs and local coordinates. Preserve the documented quadrature mismatch between fixed-heat records and atlas records; use a declared coordinate-aligned comparison adapter rather than pretending the raw schemas are identical.

### R3. Objectives that match the inverse task

The matched pair shares nonlinear scope, initialization, historical replay, sample order, optimizer budgets, receiver sampling and checkpoint policy.

- `R_value`: historical and stencil absolute-role losses.
- `R_response`: the same losses plus finite-role, finite per-module peak, and continuous pressure losses.

Use

\[
\mathcal L_R=\mathcal L_{\rm historical}+
\mathcal L_{\rm stencil,value}+
\lambda_\Delta\mathcal L_{\Delta y}+
\lambda_T\mathcal L_{\Delta T_{\max,m}}+
\lambda_p\mathcal L_{p,\Delta p}+
\lambda_0\mathcal L_{\rm null}.
\]

For per-module peak responses, supervise

\[
\big[\widehat T_{\max,m}(d+\delta)-\widehat T_{\max,m}(d)\big]
-\big[T_{\max,m}^{\rm ref}(d+\delta)-T_{\max,m}^{\rm ref}(d)\big].
\]

Compute each peak using the maintained material receiver universe and validity masks. Do not replace this with only a smooth global maximum. Report smooth and hard objectives separately. Ensure candidate switching of the hottest module remains visible.

Pressure supervision includes absolute maintained-section drop and its finite increment. Use the original family benchmark only as the declared experimental boundary. The six infeasible training states in two families can support continuous regression and descriptive boundary tests, not a reliable false-feasible rate. Do not make feasibility BCE central to the remedy.

For the analytic-wake generator's verified fixed-geometry heating control, penalize spurious heat-driven velocity/pressure changes. This is a case-specific structural null supported by the generator and recorded controls, not a universal multiphysics law. Do not impose a generic pressure-null constraint on position changes or another case family. Preserve nonzero thermal heat-allocation responses.

Do not enable mixed-response losses without applicable role/family numerical evidence. Do not turn missing numerical floors into zero response labels. Conversely, exact, analytically specified null controls should not be confused with merely unobserved small effects.

### R4. Weighting and the first review

Calibrate all active terms over a balanced sample of all eight training families plus historical examples. Freeze scales and weights before the matched pair. Record per-family gradient ratios, not only their average. The measured M10 conflict warrants one bounded remedy: on the response-specific update component, project away its adverse first-order component along the combined value gradient, blockwise, when their dot product is negative. Keep an unprojected matched diagnostic on the small training probe. Label this a heuristic first-order protection, not a guarantee of no forgetting. Do not combine projection, adaptive weights, new sampling and a second architecture in an uninterpretable sweep.

Use at most ten value-only warm-up updates. Ramp finite-role, finite-peak and continuous pressure losses during updates 11–50; keep all active thereafter. Save the active terms and weighted gradients in the curve. A decision-aware efficacy review must include at least 100 actual updates with those terms active.

Review at actual updates 200, 500 and at most 1,000 per arm. At update 200, distinguish:

- **No fitting movement:** negligible training finite improvement despite finite gradients. Diagnose parameter reachability; use the one contingent remedy or stop this candidate.
- **Fitting with transfer failure:** clear train gain and persistent development regression. Stop or reduce scope; do not buy more epochs blindly.
- **Still learning:** meaningful train improvement, no severe broad regression, development not yet decisive. Continue to the preauthorized 500-update review; first-review non-promotion is not an automatic stop.

As preregistered research heuristics, use approximately 10% train finite-error reduction to distinguish clear movement and 5% improvement over the matched value arm as an initial development efficacy target. Retain exact per-family changes and numerical uncertainty; these percentages are neither physical certificates nor hard universal response thresholds. Freeze any different numeric heuristic before looking at candidate development results.

A promoted response candidate must improve finite per-module temperature and useful pressure behavior over its matched value arm while respecting the broad historical and near-interface guards. It is not necessary to beat every unrelated flux-proxy diagnostic. It is necessary to disclose regressions and to reject an inverse-use claim when the objective or constraint remains unreliable.

### R5. Required evaluations

Use equal-family summaries, per-M/context strata, and tails. Include all eight train families, four Re90 calibration neighborhoods, the fixed-heat controls and the existing 30-case broad panel. Report the unmodified incumbent and matched value arm alongside the response candidate.

Report absolute role error, finite role error in physical units, per-ID finite peak error, true-peak change sign when resolved, hottest-module identity, pressure baseline bias, pressure increment error, and observed boundary outcomes. Near-null denominators are not stabilized by inventing a favorable epsilon; use declared physical scales and absolute errors.

A train-only tiny overfit result is a learning-path test. An improvement on the four Re90 neighborhoods is response-development evidence. Neither is new-layout physical validation.

## 5. Workstream G — learn a directed receiver-local cover on native WindFarm data

### G0. Establish that the fitter can learn existing coherent labels

Separate the current assessment into three results:

- `learning_eligible`: typed coherent labels exist, their provenance is valid, and a bounded fitting question is defined;
- `reference_sufficient`: the candidate passes the declared stored/reference physical quantities;
- `deployment_eligible`: complete measured cost and physical accuracy pass the intended workload's limits.

Latency and label diversity do not determine whether a supervised example exists. K=1 and constant-label panels are legitimate controls, although they cannot demonstrate adaptive K.

Use the six saved, search/disjoint-verified partial plans for a small classifier fitting diagnostic, at most 100 updates. Compare predictions with the exact coherent labels, check masked negative/positive learning, and execute the resulting **hard** plans on native verification queries. Do not union separately adequate prunes into an untested joint label. The diagnostic is allowed to fail or memorize; it is not the formal organizer-generalization result. Its purpose is to remove the zero-update procedural blockage with an actual learning measurement.

Run the existing per-case collapsed-root and population fixed-support controls even if no adaptive model is deployable. Fix any slot-dependent control semantics: a slot-indexed fixed mask is an explicitly artificial baseline, not a permutation-invariant physical policy.

### G1. Separate directed permissions without changing the native head

Add a typed plan container keyed by mechanism and, where relevant, physical phase. Initial keys are `MM`, `ME`, `EM`, `QM`, `QE`. Thermal keys additionally identify P0/P1/P2 when those paths are activated. A missing key means explicitly recorded full access, not an inferred zero.

Do not assign one shared membership matrix to all these mechanisms by necessity. Reuse a plan across mechanisms only as a named tied-control experiment. A module may be needed to contextualize an environmental state while not being needed in a particular direct query read.

Implement a dense-masked reference executor before a fast sparse one. It evaluates the native rectangular kernels with live masks and the same original reductions. This isolates the scientific effect of a permission change from the very different packed execution path. It is acceptable for training to pay dense work. Record that work honestly.

All-access typed plans must reproduce the incumbent in values and the relevant input/parameter gradients. Identity receiver splitting with identical source permissions must also reproduce it. These tests are essential because the new graph has more degrees of freedom than the old tied plan.

Use explicit role/phase arguments rather than relying on mutable diagnostic attributes for production permissions. The existing temporary `_interface_read_role` hook is useful for tracing, but is not a safe full batching/concurrency contract.

### G2. Replace the one-shot proposal catalogue with local search

Keep the current input-anchored receiver tree for the first experiment. Do not start by inventing a different clustering algorithm.

For each training layout/direction:

1. Start from the exact all-access typed plan.
2. Open one receiver split with **identical full source support** on both children. This must be an identity representation change.
3. Propose removing one source or one small environment block from **one child and one mechanism**. The other child and mechanisms remain unchanged.
4. Evaluate the complete native model, not just the isolated message tensor. Protect all declared receiver roles, including near-turbine outputs.
5. Accept a local omission only after its actual predictive check; then propose a further omission from that accepted state. Every accumulated state is evaluated jointly.
6. Merge equivalent children or restore the most damaging omission when the bounded search reaches a fidelity limit.

This tests the hypothesis that the field needs a module globally but not through every local route. The old nearest-child proposal instead removes many cross-region routes at once and simultaneously changes preparation and reading. Its failure does not answer this smaller question. [C3, C4]

Do not infer a physics-specific locality threshold merely from distance. Use relative flow-frame geometry to nominate candidates, and use native response/fidelity checks to decide. Wind-aligned upstream/downstream features are appropriate input features only after verifying the existing coordinate convention. Stored directions are categorical support; do not claim unseen continuous wind-direction transfer.

Initially keep essential self/near-receiver routes as declared protected candidates, while testing environment and remote-module omissions. Record the protected policy, rather than claiming those routes were discovered. No source absent from a queried training region is assigned a globally irrelevant label.

### G3. Candidate ranking and bounded composition

Use cheap gate-gradient or message-contribution scores to rank local removal proposals. These are **model-side ranking heuristics**, not physical labels. A high attention mass is not automatically a high physical influence, and an attention value includes normalization effects.

Use a small beam of at most four distinct partial plans. Score actual logical rows/block work and protected errors, keeping a small nondominated set. Search until the per-row evaluation budget is reached or no local move changes that frontier. Do not return after one catalogue call. Cache observations using the full input, checkpoint, probe, mechanism and **complete plan hash**, not only `(row, pass, source_index)`, because the same deletion has a different effect after other deletions.

A local split must be able to pay for itself through different useful source sets. Splitting identical full-access children earns no sparsity credit. Likewise, K=1 with nearly all sources is a legitimate weak compression control, not a desired adaptive endpoint.

The initial logical objective is executed-source/block work under the dense-masked or subset executor, plus a small explicit packet-overhead term. Do not let the slow historical packed runtime erase the structural frontier. Measured runtime is retained in every record and used later for deployment selection.

### G4. Fidelity budgets and reference checks

Maintain at least these separate measurements:

- teacher distortion in checkpoint-normalized units, with exact scale definitions;
- actual physical error against stored native velocity cells;
- wake-residual, near-turbine, hub-slab and background errors;
- module-neighborhood/rotor-support velocity quantities where their reference extraction is well defined;
- finite **teacher** responses to bounded input perturbations, explicitly not new-layout CFD responses.

The old `0.10` gate is a normalized vector RMSE, not a universal “10% physical error.” Preserve it for historical comparison. For the new structural frontier, report predefined normalized distortion levels, for example 0.01/0.05/0.10, without selecting a threshold from development outcomes.

For physical prediction promotion, freeze a separate comparative guard based on train-only native-reference error. An initial research guard is no more than 10% additional equal-layout error in each protected role, plus an explicit numerical parity allowance. This is a comparison with the incumbent, not an engineering tolerance. Show absolute m/s errors and tails so a small baseline does not hide an important change. Do not use this guard to suppress all learning experiments; it determines the claim earned by the trained result.

Evaluate the **partial and learned** models on the reference cells. Do not reuse the full-policy whole-grid table. Use direct native cell gathers or a documented quadrature/interpolation rule; no invented rotor power or AEP formula.

For an inverse-facing quantity `g`, also evaluate the additional finite-response distortion:

\[
\epsilon_{\Delta g}^{G,B}(d,\delta)=
\left|[g_G(d+\delta)-g_G(d)]-[g_B(d+\delta)-g_B(d)]\right|.
\]

Baseline value agreement alone cannot protect this difference. Any perturbation outside available CFD labels is a teacher-preservation experiment, not reference validation.

### G5. Train one input-only organizer on the expanded panel

Parameterize `spatial_dim` in the organizer and include the mechanism/phase identifier. Node features should include the current receiver geometry/role and appropriate receiver-state features, not only a centroid and one global token. Source features retain identity through their physical attributes and position, never a learned case-ID embedding.

Allow a permutation-invariant input-side summary of the current source configuration where necessary. This provides spectator-layout conditioning. Trace its dependencies explicitly: access to all input geometry is allowed for an organizer, but it is not free work and must not become an undeclared dense trial-response path. For a local inverse plan, combinatorial decisions can be frozen at the anchor while trial numerical states remain live.

Fix recursive supervision export. Unobserved candidates are masked; they are not negative labels. Use the existing class-balanced masked BCE for supervised warm-up. If several coherent plans are adequate, retain them as an equivalence/frontier set rather than labeling one arbitrary environment block as uniquely physically correct. A deterministic canonical target or a loss against the best matching recorded adequate plan is acceptable, with its choice frozen on training data.

After the warm-up, add native predictive feedback through the dense-masked executor. An initial recipe is:

- up to 150 supervised warm-up updates;
- up to 500 updates combining coherent-label loss, protected teacher/reference output losses, and a modest logical-work penalty;
- if genuine learning and stable hard-plan evaluation continue, at most 1,500 cumulative updates for the selected organizer.

Use an all-access hard initialization. If soft sigmoid plans are used, evaluate the deterministic hard plan frequently: a soft, everywhere-positive plan is not sparse. For the hard-forward training stage, a documented straight-through binary gate is an acceptable bounded estimator; report its surrogate-gradient nature. It must not be confused with an exact derivative through topology changes. Do not introduce a stochastic gate framework and multiple alternative optimizers simultaneously. [R3]

The fidelity term must reach organizer parameters through the actual selected permission path. Detached diagnostics, a BCE decrease alone, or a complexity loss disconnected from the model do not establish useful learned organization. Log prediction/complexity gradient norms and hard-mask changes on a few real cases.

Allow one short joint adaptation of the native update/read heads if frozen-weight pruning has a clear fidelity barrier, with the same adaptation budget in a fixed-structure control. Initialize those weights from the incumbent. This tests whether the representation can learn to use sparse permissions; it is distinct from lossless post-hoc compression. Do not destroy the incumbent or silently change the teacher during this comparison.

### G6. Controls and hard formation review

Use the same decoder/adapter and compatible work budgets for:

- all-access native baseline;
- the per-case verified oracle plan;
- the learned input-only plan;
- one population-fixed geometric/support control;
- an ungrouped direct-pair permission control with a matched logical pair budget;
- the per-case collapsed-root source-union control.

Compare source sets, per-mechanism K, disjoint fidelity, physical errors and complete cost. Distinguish learned source identity from learned packet count. The old six labels have different M and padding; differing raw label arrays are not evidence of meaningful within-M adaptation.

Test same-M different layouts, different stored directions at the same layout, and feasible local teacher perturbations. Reordering modules and environmental sources must preserve outputs and equivariant support. Duplicating an environmental quadrature atom with split weight must preserve its physical action within tolerance. Query chunks and query ordering must not change the case plan.

A positive adaptive-K result requires useful deterministic differences in nonredundant packet structure on native cases while meeting the declared fidelity budget. Variation is a measured result, not a required training label or a rewarded histogram. If the best plan is constant, report that fact and compare its utility with the adaptive attempt.

## 6. Workstream X — make useful sparsity executable, without making it a learning prerequisite

### X1. Short attribution profile

Profile one unchanged Dense call, explicit all-access policy, and the verified K=1/M-all/E480 partial at Q=1024. Separate:

- input/geometry and tree construction;
- policy scoring over candidate nodes;
- permission compilation and diagnostics;
- typed preparation kernels;
- QM/QE reading;
- coarse/local branches and physical wrapper;
- host synchronization and transfers.

Measure with diagnostics off and then on. Use a few interleaved warm repeats, retaining contention status. A synchronized component profile is diagnostic; use an uninstrumented complete call for the deployment comparison.

The current cost estimate divides all-node scorer time by capacity but charges an active-group count. The scorer actually visits all candidate nodes and sources. Correct this mismatch before using an estimated cost objective. Also distinguish fixed launch/packing costs from per-row cost; the fast all-access dispatch and slow partial dispatch cannot be represented faithfully by one linear rows-only model. [C5, C6]

### X2. First executor remedy: rectangular source subsets

For a common permission set shared by a receiver block, gather source tensors once and reuse the existing dense-style kernels over a contiguous receiver-by-selected-source rectangle. Batch compatible blocks. For the existing root partial, this means an ordinary selected 480-source read, not a Q×480 list of scalar scatter operations.

Preserve semantics carefully:

- a source removed from a read is not necessarily removed as a receiver/state from the whole model;
- subset preparation keeps the original denominator and mass convention of that native mechanism;
- contextual states are recomputed using the correct typed permissions;
- duplicate source paths do not receive double attention mass;
- attention normalizes once over the unique union;
- padded and actually evaluated rows are separately counted.

Do not simply slice the encoded case to E480 and thereby delete states that another mechanism or branch still needs. Gather on the relevant source axis of each operation.

Use the packed executor for irregular residual blocks only when its measured workload justifies it. Retain a dense fallback for small or nearly full blocks. A logical sparse mask executed by a dense tensor is still logical sparsity, not saved arithmetic; report both.

### X3. Remove avoidable organization overhead

Do not rebuild every descriptor and tree during repeated reads of the same prepared case. Precompute geometry-only node membership/descriptors once where legitimate. Move per-node Python construction toward batched tensor operations after profiling. Score only the visited frontier when this can be done without changing the policy, or charge full-capacity scoring explicitly.

For inverse steps, introduce a frozen **combinatorial** layout: node connectivity, source IDs and block buckets may remain fixed in a declared trust region. Relative coordinates, source features, local physical states, attention scores and quantities are recomputed at every trial. If access boundaries are intended to move with geometry, recompute their continuous values through live tensors; a detached snapshot must not silently remove design gradients.

Plan invalidation covers presence, family, coordinate convention, receiver-role schema, source support and anchor change. A new trial is not the old numerical prepared state. An opaque object ID is not a sufficient cache key.

### X4. Benchmark the workload that matters

Measure at least Q=1024 and a production-scale native query workload (use the existing Q40960 comparison or a representative full-grid streaming case). For inverse use, also measure a small quantity-query workload and a small batch of candidate designs. Preparation can dominate one workload and be amortized in another.

Report cold and warm same-geometry calls, fresh-geometry calls, preparation plus decoding, and, where used, the complete objective/constraint plus backward call. Policy search and offline oracle costs are separate from a deployed organizer's cost. CUDA-event kernels and complete wall time are different scopes, not summands.

Use five interleaved paired repetitions for final latency decisions, with an agreed warm-up and identical receiver chunks/dtypes. Do not infer speedup by comparing a contended candidate run to an isolated baseline. If no isolated interval is available, preserve descriptive timing and leave deployment speed unresolved.

A block-aware implementation follows the general lesson that memory movement and scheduling matter alongside arithmetic sparsity; it is not an automatic claim that FlashAttention or any particular kernel can be dropped into this geometry-biased operator. [R4]

## 7. The inverse-facing interface: a local response contract, not a colored partition

### 7.1 Required records

Add typed records with approximately the following responsibilities; names may be adapted to existing project conventions:

```python
@dataclass(frozen=True)
class InterfacePacket:
    mechanism: str                 # MM/ME/EM/QM/QE and optional physical phase
    source_physical_ids: tuple     # module IDs or environmental support IDs
    receiver_support_id: str       # role + physical/local coordinate convention
    permission_parameters: object # no target arrays
    evidence_scope: str            # model_transport / teacher_response / local_reference

@dataclass(frozen=True)
class LocalInterfacePlan:
    anchor_input_hash: str
    checkpoint_hash: str
    packets_by_mechanism: object
    frozen_connectivity: object
    trust_region: object
    declared_global_paths: tuple
    validity_schema: object

@dataclass(frozen=True)
class ResponseAssessment:
    receiver_quantity: str         # per-ID material peak, maintained pressure drop, etc.
    changed_coordinates: tuple
    predicted_increment: object
    numerical_discrepancy: object  # unknown is not zero
    model_error_indicator: object # separately calibrated/validated where possible
    evidence_scope: str
    supported_move_scale: object

@dataclass(frozen=True)
class DecisionGroup:
    changed_coordinates: tuple
    protected_receivers: tuple
    protected_constraints: tuple
    rationale: str                 # shared_receiver / constraint_coordination / resolved_mixed
    supporting_response_ids: tuple
```

These are contracts, not an instruction to pass Python dataclasses through every GPU kernel. Compile them to tensors/blocks as appropriate. Physical IDs are for joins/provenance; their arbitrary strings must not become predictive features.

### 7.2 Same-model baseline correction remains the default

For every module m, correct before applying the maximum:

\[
\widetilde T_m(d;d_0)=T_m^{\rm ref}(d_0)+\widehat T_m(d)-\widehat T_m(d_0),
\quad
\widetilde J(d;d_0)=\max_m\widetilde T_m(d;d_0).
\]

For pressure,

\[
\widetilde p(d;d_0)=p^{\rm ref}(d_0)+\widehat p(d)-\widehat p(d_0).
\]

Use one checkpoint and one declared frozen topology for the baseline and trial terms. Recompute all continuous states from the trial input. After a reference-accepted move, create a new anchor and recompute its model baseline and graph.

Do not relax the pressure limit to `1.05 × each newly accepted pressure`. The experimental limit is fixed from the original task baseline unless a different task is explicitly defined. Grid refinement does not silently change that limit either.

A reference anchor removes an offset, not a slope error. If the anchor and finite response have bounded errors, their conservative sum bounds the corrected quantity's error. With only empirical indicators, call the result an empirical screen—not a certificate. Preserve correlated numerical errors; do not combine them as independent Gaussian uncertainties without evidence.

For the graph-induced response error,

\[
|\Delta g_G-\Delta g_B|\le |g_G(d)-g_B(d)|+|g_G(d_0)-g_B(d_0)|.
\]

Measure the left side directly; an anchor-only check misses the second-state error. For temperature, protect all potentially competing hot modules, not only the current argmax. The current hottest module can change after a move.

### 7.3 Extract decision groups from actual receiver responses

At an accepted design, construct a small model-side matrix of finite changes or trustworthy fixed-topology derivatives from design coordinates to per-module temperature quantities and pressure. Use radius/physical scales for design increments. Retain signs, magnitudes, uncertainty and receiver identities.

Derive groups that either affect a common protected receiver or jointly satisfy objective and constraint needs. A coordinate reducing pressure may enable another coordinate reducing peak temperature; this is useful coordination even if their physical mixed term is zero. Always compare with a simple derivative/finite-response ranking baseline. If the hypergraph offers nothing beyond that baseline, say so.

For a mixed physical response claim, use field/material receiver changes under aligned interventions and applicable numerical resolution. A nonlinear max/log-sum-exp can create a mixed objective response from independent module responses; it is not evidence of physical exchange between those modules. [E2, “Evidence design and numerical reliability”]

Derivative-informed operator research provides a precedent for learning design-variable responses alongside values. It does not supply the missing reliable shape-Jacobian labels for this rasterized generator; use resolved finite changes first. [R2]

The decision selector may use the accepted reference observation and the user-defined objective/constraint. The **forward organizer** remains input-only and cannot consume trial reference targets. These are different permitted information sets; record them explicitly.

Goal-oriented numerical methods motivate focusing error control on the requested output rather than a single global norm. Here, sensitivity-weighted message or receiver scores are proposed diagnostics, not a certified dual-weighted PDE residual estimator. [R1]

### 7.4 Wire the actual Thermal policy boundary

Extend the native Thermal quantity predictor and wrapper so a typed frozen plan can reach the core's preparation/read calls. Currently `tensor_quantities` rejects nonempty `frozen_topology`; do not remove that check without implementing the real path. Add exact all-access and nontrivial local-mask integration tests.

Before adding an AD-based optimizer, also expose a quantity method accepting caller-owned live `DesignInput` tensors, or return the exact differentiable design tensors alongside the quantities. The current convenience method converts `PhysicalDesign` NumPy arrays into new Torch leaves; setting `requires_grad=True` there does not create a derivative path to an external optimizer's pre-existing design tensor. This does not invalidate the completed finite-candidate study, but it is a required API boundary for the proposed derivative interface. Test nonzero gradients to the caller-owned coordinates/heating in a non-null case.

A graph used only to nominate optimizer coordinates is a `decision_graph_with_dense_forward`. This may be useful and should be tested, but it is not a graph-mediated sparse forward calculation. Record both when they coexist.

For real masked inverse scoring, establish four comparisons before new physical trials:

- intact Dense at baseline and trial;
- graph-mediated model with frozen topology and fresh trial states;
- the same trial with recomputed topology, as a separate switch diagnostic;
- direct-pair or fixed-structure control at matched candidate effort.

Do not silently substitute the recomputed-topology result into a frozen-plan score. If a switch causes an excessive discrepancy, shrink/rebuild the local model or use the intact predictor for that step. Fixed-topology AD/FD consistency is not proof of a smooth physical shape derivative or a smooth topology switch.

## 8. Workstream I — bounded inverse evidence after the forward/interface tests

### I0. Use stored records for the first decision replay

Before spending new solver calls, freeze the model, objective, benchmark pressure definition, numerical-resolution policy and group selector. Replay existing Thermal neighborhoods and the fixed-heat controls.

Report whether baseline correction improves quantity prediction, whether the graph preserves increments, and whether the proposed groups outperform simple sensitivity ranking on the available measured candidates. Only a completely reference-evaluated common pool supports finite-pool regret. A few policy-selected outcomes do not.

Keep the train-0318 comparison as a negative/control example; do not optimize a new policy around its known winner and call it held out. It is especially useful for testing whether the new decision interface recognizes complementary pressure/temperature changes without inventing a large mixed physical interaction.

### I1. Conditional new physical pilot

Proceed only after the physical-call reconciliation and a useful response/decision replay. With at least 16 legitimately available attempts, the default pilot is two new baseline layouts, three policies, and two grids:

- common baseline on each of two grids, for each layout: four calls;
- one frozen selected candidate from each of three policies, on both grids, for each layout: twelve calls;
- total sixteen new attempts, with up to four remaining attempts reserved for failures or a preregistered numerical clarification.

Keep the new layouts disjoint from all opened physical families by input hash/tolerance. Choose their input geometry and task before reading their trial responses. An independently observed baseline can be used for correction and the current hot/constraint state; that is part of the inverse protocol, not a withheld target leaked into training.

Policies are learned-interface-informed grouping, a direct finite-sensitivity/ungrouped control, and a size-matched random grouping. All use the same frozen predictive function, candidate count, coordinate dimension, physical radius, feasible-redraw rule and reference-observation budget. Score the same candidate predictions with Dense and the learned transport model as a separate paired diagnostic to isolate organization distortion. Do not change both the forward model and group-selection rule and attribute the outcome to only one of them.

Persist all surrogate selections before accessing any chosen trial reference. Run the finite coordinate moves at the declared scale. Do not treat AD/FD of the surrogate as the generator's continuous shape derivative; the rasterized shared-grid reference has already shown material finite-move mesh sensitivity.

Use one original task pressure limit across both grids and all selected candidates. Evaluate the baseline and every selected candidate on both grids. A same-sign/three-times-discrepancy check is a resolution screen, not a convergence proof. Report unresolved outcomes and policy-order reversals rather than choosing whichever grid favors the proposed method.

Two starts support a bounded pilot, not a reliable statistical design-advantage claim. If fewer than sixteen new attempts are truly available, do not silently run an under-matched physical comparison. Finish the stored-data decision replay and return the precise missing budget.

### I2. Accept, reject, or withhold judgment

Return the best independently verified feasible design actually accepted by the policy, separately from the best reference outcome found in a retrospectively inspected pool. Report candidate rejection, false-feasible selections, reference failure, and unresolved improvement. A reference-feasible candidate predicted infeasible and sent only for audit is not silently promoted into a policy success.

Use a predeclared improvement resolution incorporating the available numerical discrepancy. Do not let an arbitrary `1e-4` scalar threshold convert a grid-sensitive change into a physical win. Trust-radius updates are proposed only from meaningful actual/predicted changes; no second accepted anchor is created without its own reference verification and budget.

### I3. WindFarm design scope

Continue native-field and finite-library work. Prepare, but do not fabricate, the contract required for new-layout reference evaluation: physical coordinate frame and wind transform, turbine geometry/forcing, design variables, inflow/outflow/terrain/turbulence conditions, solver/mesh controls, and the exact objective reduction.

A native neighborhood velocity metric is not turbine power or AEP. A stored `wake_loss_pct` label is not a derived objective law. No new-layout WindFarm physical-design claim is authorized without the external generator and objective contract. A failure to obtain that external reference does not block WindFarm organizer learning on existing native fields. [E1, “External-reference information”]

## 9. Execution sequence and autonomy in Goal mode

Run the independent learning and execution questions in parallel logically; serialize GPU-heavy jobs when needed for resource safety. Ordinary recipe failures should trigger the bounded remedy below, not an indefinite request for permission or an unbounded search.

| Milestone | Required work/result | Continue or stop rule |
|---|---|---|
| M0: pin/reconcile | Current SHA, incumbent hashes, training/development identities, physical ledger reconciliation, short code delta note | Physical ledger issue blocks new solves only. Preserve ongoing neural/stored-data work. |
| M1: minimum empirical learning | Existing-label organizer diagnostic; nonlinear response scope probe; Dense/partial latency attribution | A slow partial does not stop organizer fitting. A broken adapter is fixed before interpretation. |
| M2: local native supports | Typed permissions, identity split, incremental local search on expanded panel, coherent labels and disjoint checks | Use one bounded local-search remedy if no local omission survives. Do not recycle only global source deletion. |
| M3: fitted interfaces | Matched nonlinear response arms; supervised plus predictive organizer fit; deterministic K/support review | Continue learning candidates to their next bounded review when curves justify it; no universal automatic u100 stop. |
| M4: physical/compute review | Partial/learned full native reference evaluation, response distortions, pathway map, rectangular-subset benchmark | Promote structural, physical and speed claims separately. Keep policy-free Dense as needed. |
| M5: inverse integration | Real topology wiring, corrected quantities, stored decision replay; conditional matched new-reference pilot | Stop new physical trials if response/geometry/resolution/budget is inadequate. Return measured partial results, not invented utility. |

The goal is not to maximize the number of modules or tests written. Prefer a small implementation whose learned masks are actually exercised by native data over another broad framework with no optimizer run.

### 9.1 Allowed bounded remedies

There are at most three research remedy branches for the entire round:

1. **Response remedy:** nonlinear scope with balanced calibration; if it genuinely cannot fit, the one small zero-initialized interface adapter described in R1. No additional from-scratch backbone.
2. **Organization remedy:** more local/typed proposals, then one short joint head adaptation if the frozen model cannot use sparse permissions. No alternative tree zoo or enforced K diversity.
3. **Execution remedy:** rectangular subset/block batching and removal of profiled redundant overhead. No large custom CUDA/Triton project before the simple subset path and current installed-stack capabilities are measured.

Within a remedy, fixing serialization, shape, device, mask or provenance bugs is implementation work, not evidence that a scientific hypothesis succeeded. Preserve failed attempts and their charges.

### 9.2 Resource ceilings

These are new-round ceilings and remain subordinate to any stricter repository/user limits. Record actual work; a missing time cannot be replaced by the ceiling as if measured.

| Resource | Ceiling / rule |
|---|---|
| Aggregate new GPU-associated experiment wall | 12 hours, physical GPU 2 unless existing project rules require another explicit allocation; never terminate unrelated jobs |
| Aggregate optimizer updates | 5,000 actual attempted optimizer calls across all pilots, matched controls, remedies and fits; failed/unsaved updates count |
| Matched Thermal response pair | At most 1,000 updates per arm; reviews at 200/500/1000 |
| Formal organizer plus existing-label diagnostic | At most 1,500 cumulative optimizer updates in total; include the initial diagnostic in this count |
| Small response scope/rate diagnostics and contingent adapter probe | At most 400 updates combined |
| Graph joint-head repair and matched fixed-structure repair control | At most 300 updates per arm, counted under the aggregate ceiling |
| Additional oracle candidate model evaluations | At most 4,096; default per-row cap 96, initial beam width at most four; charge all newly executed candidate states |
| Reference solves | Zero until reconciliation; then at most 20 and never above verified remaining uncommitted global allowance |
| Reference CPU | At most two newly measured aggregate process-hours, also subordinate to existing global limits |
| New external WindFarm CFD solves | None without an available, verified generator and explicit objective contract |

Training forward passes, oracle evaluations, broad evaluation passes, reference observations and new physical attempts are distinct counters. Track the native forward-call multiplicity of finite stencils even though the training ceiling is expressed in optimizer updates. Do not double-add nested CUDA timings to job wall.

Save atomic checkpoints early, then every 50 updates and at each review, including optimizer, sampler, RNG, active scope, loss scales, active terms, exact source checkpoint and plan schema. A lost checkpoint does not erase the executed work from the ledger.

### 9.3 Minimal “continue” criteria

A meaningful training loss reduction, nontrivial hard-mask learning, and manageable validation drift can justify the next already-budgeted review. Passing every physical deployment criterion is **not** required before testing whether a learning mechanism works.

Conversely, finite gradients alone, lower BCE with unchanged collapsed predictions, or a smaller K with an uncontrolled bypass do not justify extended training. Stop a candidate after its bounded remedy when those are its only gains.

Never end the round at `target_unavailable` solely because the old executor is slow. End with a fitted structural control, native deterministic results and an explained limitation. The exception is a genuine failure of data availability/provenance or inability to execute the required native path after the bounded implementation repair; document it precisely.

## 10. Maintained code changes and test responsibilities

Paths below are relative to `HONF_Proj/`. Use existing project boundaries; do not rewrite unrelated legacy architectures.

### 10.1 Thermal response and inverse

- `Case_ThermalChannel/src/channelthermal/response_control/runner.py`: named nonlinear trainable scope, explicit parameter inventories and new recipe selection; preserve historical terminal and conversion modes.
- `response_control/training.py`, `paired.py`, `resume_provenance.py`: multi-family calibration, active objective telemetry, matched update schedules, complete resume provenance and controlled projection if used.
- `response_control/losses.py`, `thermal.py`, `evaluation.py`: finite per-ID peak terms, continuous pressure quantities, known fixed-heat nulls, per-family evaluation and physically named reductions.
- `response_control/native.py`, `interface_field_coupling.py`: typed real policy dispatch through current native phases; keep target-free input and exact all-access semantics.
- `inverse/native_corrected.py`, `native_matched.py`: accept only genuinely implemented frozen topology, track live trial recomputation, uncertainty/resolution screens and matched selection/acceptance accounting.
- A new named config rather than changing the historical `response_control_native_staged.json` in place.

### 10.2 Core graph and executor

- `src/honf_forward_core/interface_fields/adaptive_interaction_cover.py`: typed mechanism plans, recursive frontier, canonical packet counts, complete plan hashes and source-union semantics.
- `adaptive_cover_field.py`: dense-masked reference execution, rectangular-subset/block execution, phase/mechanism-specific permissions, explicit fallback and complete work counters.
- `input_cover_organizer.py`: spatial dimension, mechanism/receiver features, conservative hard initialization, observed-label masks and live predictive loss access.
- `adaptive_cover_oracle.py`: structural frontier separate from deployment timing; bounded sequential composition with joint checks.
- `core.py` and case adapters: pass explicit policy context, preserve common native heads, disclose coarse/local/global paths, keep query invariance.

Do not overload the source-membership array with arbitrary target-dependent pruning labels. The tensor representation, physical catalogue and evidence annotations have separate responsibilities.

### 10.3 WindFarm workflows

- `native_cover_organizer.py`: local directed proposals; full-plan observation keys; recursive label export; physical evaluation of the actual partial plan.
- `native_cover_organizer_fit.py`: learning/reference/deployment eligibility, configurable bounded reviews, supervised plus predictive fitting and always-executed controls.
- `native_cover_panel.py`: freeze the eight/four organizer split of the existing twelve layouts and disjoint query subsets; preserve layout grouping and direction support.

### 10.4 Required focused tests, followed by native empirical checks

1. All-access output and relevant gradient parity with intact checkpoints; identical-support split/merge parity.
2. MM-only, ME-only, EM-only, QM-only and QE-only interventions act on their intended mechanism and preserve other permissions.
3. Dense-masked, rectangular-subset and packed implementations agree for identical masks, quadrature and denominators; empty-support behavior is explicit.
4. Module/source permutation, environmental quadrature splitting, inactive padding and 2-D/3-D operation.
5. Recursive observed-label export; unknown decisions remain untrained; K ignores inactive nodes and equivalent packet actions.
6. A slow coherent label is learning-eligible but not deployment-eligible; constant K is not incorrectly classified as missing data.
7. Predictive loss reaches organizer parameters; hard-forward/evaluation plans are exercised, not only soft all-source routes.
8. Frozen topology with live trial positions/heating changes the intended states; new anchor refreshes topology and baseline; target poisoning leaves forward predictions unchanged.
9. Per-module correction before maximum, fixed pressure-limit persistence, quantity unit/definition checks and hottest-module switching.
10. Attempt/reuse accounting, full-plan cache-key distinction, failed/unsaved optimizer charges and atomic checkpoint recovery.

Tests establish the implemented contract. The final report must also contain real native optimizer curves, learned hard plans, paired field/reference metrics and measured execution. Nine or ninety passing tests do not substitute for those observations.

## 11. Deliverables and report structure

Commit maintained source, configs, tests, and a compact English study report. Keep checkpoints, raw arrays, native field captures, generated figures and one-time evaluation outputs in the existing ignored locations; follow `AGENTS.md`, artifact hooks and trusted-checkpoint/security rules. Do not commit datasets or bypass those hooks to make delivery look complete.

The report should contain these sections in this order:

1. **Question and result:** which parts of forward response, native organization, physical interface and inverse utility improved, failed or remain unknown.
2. **Native nonlinear response pair:** scope, actual active objectives/updates, train/development/broad/fixed-heat tables and decision quantities.
3. **Directed support search:** actual local interventions, cumulative plans, failure distribution, same-M/direction coverage and the logical fidelity frontier.
4. **Fitted organizer:** hard K/support by mechanism, source identities, controls, predictive-loss contribution, unseen-to-organizer layouts and query invariance.
5. **Physical interpretation:** receiver-level evidence, declared global paths, direct transport versus effective response, objective/constraint coordination versus physical non-additivity.
6. **Execution:** Dense/masked/subset/packed complete timing, actual rows, memory, workload shape, contention and remaining cost.
7. **Inverse:** stored replay and any authorized new-reference result, corrected quantities, frozen selection, original limits, grid outcomes and uncertainty.
8. **Resource and provenance:** one reconciled physical ledger, actual optimizer/oracle counters, hashes, missing times and delivery revision.

Save an ignored machine-readable summary with independently named statuses, for example:

```json
{
  "native_response": "improved_or_not_improved",
  "organizer_training": "completed_or_genuine_execution_blocker",
  "deterministic_adaptive_K": "demonstrated_or_not_demonstrated",
  "teacher_preservation": "passed_or_failed_by_declared_view",
  "reference_sufficiency": "passed_failed_or_unknown",
  "complete_dependency_coverage": "complete_or_named_partial_paths",
  "deployment_speed": "passed_failed_or_unresolved",
  "inverse_evidence": "stored_replay_local_pilot_or_not_run"
}
```

These are separate outcomes. No single `passed=true` may conflate them.

### Completion standard

The central question is whether input-conditioned, receiver-local packets can organize consequential native computation and expose useful finite responses for decisions. The round succeeds scientifically by answering that question with actual learning and controlled native evidence, even when a deployment gate fails.

Do not declare a successful HONF adaptive physical interface from K variation alone. A strong positive result needs useful hard organization, receiver/response fidelity, explicit dependency coverage, and a decision or execution advantage under the corresponding evidence contract. A weaker result can motivate the next experiment, but it retains its narrower label.

## 12. Source register and scope of this review

### Project evidence

**E1.** `docs/reports/HONF_Native_Recovery_and_Adaptive_Interaction_Study.md`, attached English report and reviewed revision `6394e54`. Source sections are named throughout this plan. The report supports the stated native fits, six partial teacher checks, timings, local analytic-wake reference observations and train-exposed inverse outcomes. The ignored raw checkpoints and output ledgers were not rerun during preparation of this plan.

**E2.** `docs/reports/HONF_Interaction_Response_and_Inverse_Design_Study.md`, supplied historical report. Used for the distinction between field and smooth-objective interaction, prior final-set exposure, generator scope and the conflicting 286-attempt account. Its historical claims are not silently reconciled with E1.

**E3.** `docs/reports/HONF_Decision_Aware_Dynamic_K_Study.md`, supplied historical report. Used only to preserve rejected conversion/control identities and earlier limitations, not as a current native result.

**E4.** `UpgradePlan/HONF_Native_Recovery_and_Adaptive_Interaction_Plan.md`, prior plan. The present document preserves native initialization and reference correction while explicitly separating organizer learning from speed eligibility.

### Inspected source at `6394e546dd2bc7a39548b8aa52dead239c3efd16`

**C1.** `Case_ThermalChannel/src/channelthermal/response_control/runner.py`: native initialization, terminal-layer scope and optimizer inventories; inspected lines 1–330. `response_control/training.py`: sampled loss/replay/update loop and review boundaries; inspected lines 485–750.

**C2.** `Case_ThermalChannel/configs/response_control_native_staged.json`, complete file; and `response_control/losses.py`, inspected lines 250–480, including absolute decision and finite pressure objectives.

**C3.** `Case_WindFarm/src/windfarm/workflows/native_cover_organizer.py`: local proposal catalogue and physical probe reductions, inspected lines 1–500; oracle loop and one-shot-per-pass proposal closure, lines 1570–1810.

**C4.** `src/honf_forward_core/interface_fields/adaptive_cover_field.py`: cache, typed preparation use of the shared plan and masked pair messages, inspected lines 1–520; `adaptive_interaction_cover.py`, inspected lines 270–450, covering query and preparation pair compilation.

**C5.** `Case_WindFarm/src/windfarm/workflows/native_cover_organizer.py`: timing/cache/cost-coefficient code, inspected lines 500–820; native grid evaluation and first-level label export, lines 890–1200; frozen panel and provenance setup, lines 1201–1510.

**C6.** `src/honf_forward_core/interface_fields/input_cover_organizer.py`, complete file, including dimensional assumptions, all-node source scoring, class-balanced label loss and soft/hard conversion.

**C7.** `Case_WindFarm/src/windfarm/workflows/native_cover_organizer_fit.py`: eligibility logic, inspected lines 1–280; classification loss and training/evaluation path, lines 300–560.

**C8.** `Case_ThermalChannel/src/channelthermal/inverse/native_corrected.py`, complete file, including the current topology rejection, same-model correction and reanchoring.

**C9.** `src/honf_forward_core/interface_fields/core.py`, inspected encoding/native policy dispatch lines 820–1100 and initial helper definitions; `Case_ThermalChannel/src/channelthermal/interface_field_coupling.py`, inspected lines 1–300, including role tracing and physical port/outside-coordinate construction. Coarse/local endpoint importance is taken from E1, not inferred from an unseen parameter ablation.

**C10.** `Case_ThermalChannel/src/channelthermal/inverse/native_matched.py`, complete file: matched candidate construction checks, selection-before-reference callback, audit-only infeasibility and acceptance accounting.

The live branch was verified at the reviewed full SHA. The comparison with `70d0c0a` contains two commits. This is a static source review plus analysis of reported experiments; it is not a claim that all repository files or ignored local artifacts were inspected.

### External methodological context, not project evidence

**R1.** Becker, R.; Rannacher, R. *An optimal control approach to a posteriori error estimation in finite element methods.* Acta Numerica 10 (2001), 1–102. DOI `10.1017/S0962492901000010`. Supports goal-oriented error control through output sensitivity. It does not certify HONF's proposed edge scores or the present reference generator.

**R2.** O'Leary-Roseberry, T.; Chen, P.; Villa, U.; Ghattas, O. *Derivative-Informed Neural Operator: An efficient framework for high-dimensional parametric derivative learning.* Journal of Computational Physics 496 (2024), 112555. DOI `10.1016/j.jcp.2023.112555`; author preprint `arXiv:2206.10745`. Also Luo, D.; O'Leary-Roseberry, T.; Chen, P.; Ghattas, O. *Efficient PDE-Constrained Optimization Under High-Dimensional Uncertainty Using Derivative-Informed Neural Operators.* SIAM Journal on Scientific Computing 47(4) (2025), C899–C931. DOI `10.1137/23M157956X`. These support treating response/optimization-variable derivatives as learning targets distinct from value accuracy. Our proposed first target is resolved finite change, not an unavailable accurate shape Jacobian.

**R3.** Louizos, C.; Welling, M.; Kingma, D. P. *Learning Sparse Neural Networks through L0 Regularization.* ICLR 2018; `arXiv:1712.01312`. Supports learning sparse structures with trainable gates. Its hard-concrete method is not identical to the optional straight-through estimator proposed here; neither method guarantees native speed or physical interpretability.

**R4.** Dao, T.; Fu, D.; Ermon, S.; Rudra, A.; Ré, C. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022; `arXiv:2205.14135`. Supports the importance of tiled memory-aware execution and distinguishes approximate arithmetic reduction from wall-clock speed. It is not evidence of a speedup for this HONF implementation.

Primary publication/author pages were checked for this context. The scientific architecture, interface contracts, experiment priorities and resource limits in this document are recommendations to test, not results established by those papers.
