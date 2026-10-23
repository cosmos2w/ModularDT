# HONF: preserve nonlinear interactions while learning collective organization

**Work mode:** bounded Codex Goal-mode development, not another formal run. **Reviewed branch:** `agent/honf-core-next`. **Reviewed commit:** `8e7c1704f1590b7038959fafb4b7cdfc01fb56e5`. **Evidence date:** 10 October 2026. **Status:** proposed work; no experiment or improvement in this document is a completed result.

## 0. Decision and scope

Retain the self-contained joint-model contract achieved by the J family. Do not return to a frozen D-sep plus a separately fitted thermal predictor. Retain native source identities, heat-independent Thermal configuration fields, affine Thermal heating, the nonlinear Wind output law, native physical extraction, and the common TrainingEngine.

Change the architectural experiment. The next hypothesis is **interaction-preserving collective organization**: preserve a competent nonlinear source/source and source/environment transport path, and let a shallow explicit hypergraph enrich that path without becoming a compulsory replacement for it. This is one model with shared trainable states and output heads, not an ensemble or a second predictor correcting the first model's outputs.

The principal question is:

> Can explicit collective organization improve generalization and offer a useful, receiver-dependent information interface when the model no longer has to reconstruct essential pairwise interactions from regional summaries alone?

The review identifies changed information paths, not a proven explanation of the regressions. Do not describe “missing interactions,” excessive smoothing, memorization, multi-task conflict, or diffuse membership as established causes until a controlled measurement supports that specific claim.

This goal must deliver implemented code and completed bounded learning, not only another audit. It must also be able to return a negative organizer result without manufacturing a preferred graph. A competent joint direct model would be useful recovery, but it would not complete the organizer objective.

### Non-negotiable boundaries

- No additional full-TRAIN 5000 run, extension of completed formal runs, new physical solve, inverse generator, inverse search, or WindTEST target access.
- No recursive Tree, forced small K, entropy/diversity target, early mandatory source pooling at the final readout, learned output-bias patch for known validation cases, or per-case model dispatcher.
- No external learned-flow checkpoint in the new Thermal model. All primary scientific fits initialize their own trainable shared core and all physical heads freshly, and train them together.
- Keep old architectures and all scientific checkpoints replayable. Do not edit old defaults to make previous results appear repaired.
- Use the fixed segmented datasets. Full-data checkpoints may be read-only historical references and compatibility tests; they must not initialize the primary segmented fits or supply target labels.
- Proposed campaign maximum: **24 aggregate GPU-associated hours and 16 elapsed hours**, including failed attempts, staging, diagnostics, and confirmation fits. Reserve the last **two elapsed hours** for final evaluation, reporting, protection checks, and delivery. These are ceilings, not spending targets. A tighter current user instruction overrides them.

## 1. Evidence that motivates this plan

### 1.1 Read these two reports in full

1. `docs/reports/_bk/20261009_202509Z_HONF_Joint_Field_Regional_Development_20261009.md` — the segmented development campaign, both locality revisions, and its actual preferred recipes.
2. `docs/reports/_bk/20261010_154719Z_HONF_Joint_Formal_Comparison_20261010.md` — the separate completed formal comparison, including corrected Wind score scaling and the distinction between development and formal recipes.

Also inspect the previous completed reference report only where necessary to bind Thermal3904, Wind2202, D-sep/R-direct, and native units. The new comparison's primary previous Thermal reference is 3903, not 3904; do not silently treat those checkpoints as interchangeable.

### 1.2 Compact factual baseline

The following values are reported measurements, not independently rerun measurements by the author of this plan. Native temperature/flow/proxy units are not asserted to be SI.

| Quantity and scope | Previous reference | New formal result | Interpretation |
|---|---:|---:|---|
| Thermal mean fluid-T RMSE, canonical89 | 3903: 0.3211 | 3906: 0.2862 | A real 10.9% mean improvement; 74/89 case wins |
| Thermal fluid-T p90, same panel | 0.4339 | 0.5308 | The mean improvement hides a 22.3% tail regression |
| Thermal mean u/v/p/omega errors, same panel | 3903 uses frozen3901 | 3906: approximately 3.34/4.10/3.26/2.17 times reference | Joint identity is achieved, but flow fidelity is not preserved |
| Wind near-turbine vector RMSE, fullVALID90 | 2202: 0.09295 m/s | 2203: 0.17097; 2204: 0.15916 | A broad regression, not only a scalar-loss display artifact |
| Wind downstream vector RMSE, same panel | 0.05414 m/s | 0.10991 / 0.09968 | Both new models lose on every evaluated row |
| Wind corrected component-scaled validation objective | 2202: 0.008242 | 2203: 0.024830; 2204: 0.020680 | About 3.01x/2.51x, not the misleading 8–10x unlike-display ratio |
| High-M complete Q8192 Wind inference, current report's same-GPU panel | 2202: 14.13 ms | 2203: 22.38; 2204: 29.39 | Regional models are slower as well as less accurate in this comparison |
| Thermal saved-pool ranking | 3903: 2/3 | 3906: 3/3 | A narrow positive result; do not repeat the older development ranking failure as the formal3906 result |

The formal Thermal H run uses locality strength **0**, whereas the preferred development Thermal child uses **1**. Formal Thermal response/operator coefficients are **0.1/0.1**, without a saved auxiliary-calibration receipt. The development child had a distinct calibrated objective and a branch history. The formal runs are valid measured variants, but are not reproductions of the preferred development recipe. Do not attribute the whole regression to this difference: J-geometry and Wind also regress.

The formal Wind train/validation display mismatch is primarily an evaluation/selector convention issue. It is not, by itself, evidence that the Wind TRAIN target transform was wrong. The reconstructed common-scale curves and physical m/s tables still show a substantial generalization problem. Preserve raw histories and add correctly labelled derived comparisons; do not overwrite original scores.

### 1.3 What the evidence does not establish

Falling training loss with stagnant or worsening validation supports a fit/generalization gap. It does not identify which interaction is missing, prove that the graph is intrinsically incapable, or prove that longer training can never help. Here, however, both completed 5000 fits and their earlier best aliases remain materially worse in Wind, so another unchanged long run is not the next informative experiment.

A large same-weight error increase when graph content is removed establishes that the trained network relies on that content. It does not establish an advantage over an independently trained competent direct model. Replacement may be out of distribution for co-adapted weights.

A dark environment-incidence heatmap is not proof of missing environmental information. Module and environment masses are normalized separately; a population of 192 donors naturally has smaller individual weights than a population of ten. Inspect weighted value content, gradients, and controlled prediction effects, not color brightness alone.

The active Thermal reference is analytic-wake flow plus a shared-grid thermal solver. Its measured native divergence is nonzero. **Do not impose incompressibility or a Navier–Stokes residual on that dataset simply because its outputs are called flow.** The report supports a native curl/omega consistency check, not an arbitrary new physics target.

## 2. Code-level findings and the falsifiable architectural hypothesis

Source positions below refer to the reviewed commit. Re-resolve symbols if the branch advances; do not treat line numbers as permanent APIs.

| Source / symbols | Observed implementation | Implication for the next experiment |
|---|---|---|
| `src/honf_forward_core/interface_fields/interaction_core.py`, `InteractionContextCore.prepare_context`, approximately lines 238–268 | The earlier core computes nonlinear MM, ME, and EM messages from destination state, donor state, and signed relative geometry **before** reducing over donors | Preserve this competent form of source-conditioned transport as a real control and a nested path |
| `src/honf_forward_core/interface_fields/geometry_flow_field.py`, `prepare`, `read` | D-sep's own flow architecture also has explicit source-pair updates and source-wise nonlinear receiver reads | Removing frozen D-sep need not mean removing the useful mathematical operations it learned with |
| `src/honf_forward_core/interface_fields/joint_regional.py`, J-direct branch, approximately lines 556–575 | J-direct updates each source from its own state, pooled environment state, and pooled global state; it is not the earlier nonlinear pair-context core | Do not reuse J-direct as a synonym for a recovered2202-like direct-interaction control |
| Same file, organized context, approximately lines 603–690 | Donor-to-edge content is nonlinear and uses relative geometry; typed contents are pooled into edges, then edge states are returned to nodes | Do not falsely describe the new graph as raw averaging or devoid of many-body capacity; the issue is replacement of a different transport operator |
| Same file, `_edge_to_node`, approximately lines 421–430 | Return weights are a normalized transpose of donor membership | Gathering importance and return relevance are tied. This is a model assumption, not a measured physical reciprocity law |
| Same file, `_receiver_access`, approximately lines 1080–1131 | Each receiver obtains one H-dimensional weighted regional vector from registered edges | Regional content is receiver-dependent, but this vector is shared across its physical-source reads |
| Same file, `_read_source_features` / `predict_fields` / `prepare_receivers`, approximately lines 1133–1380 | Original source states and geometry survive; a common source-read network serves the nonlinear fields and affine coefficients; the affine result has no separate near head | Source identity preservation is real, but does not prove preservation of the previous relational computation or near/far specialization |
| `Case_ThermalChannel/src/channelthermal/training/joint_task.py` | Four normalized flow losses and three temperature losses share parameters, with separate q/response/operator terms | A joint optimizer does not guarantee compatible gradient directions or adequate task-specific read capacity |
| `Case_WindFarm/src/windfarm/training/joint_task.py` and `windfarm/joint_regional.py` | Wind retains native component supervision, geometry-only E64, nonlinear residual output, domain-origin handling, and live prepared-state checks | Keep these semantic contracts; audit E64 versus previousE8 and frame differences as explicit recipe changes, not evidence of a broken unit conversion |
| `tools/joint_regional_train.py`, `validate_recipe`, and formal JSONs | Several scientifically consequential values can be omitted and resolved through defaults | New launch recipes must print and bind all effective scientific values and calibration provenance |

### Working hypothesis

The joint regional reset preserved physical source names, but changed how those sources become aware of each other. A direct decoder can retain every heater/turbine column while still receiving less useful relational states. The new source/edge/source path may force the model to learn certain interactions indirectly through regional summaries, and its flexible absolute-position/global paths may fit the training layouts without learning equally transferable relative interactions.

This hypothesis is plausible from the source and the broad Wind failure. It is not yet a causal finding. The next campaign must test a competent pair-preserving path, with and without collective organization, using the same data and loss contract. **Do not start by increasing depth, width, query count, sparsity pressure, or the number of anchors.**

## 3. Gate 0: repair the experiment identity before fitting

Allocate at most roughly 10–15% of the campaign to this stage. Reuse existing validators and receipts; do not build a new provenance framework.

### 3.1 Produce a one-page resolved recipe comparison

Compare actual checkpoint metadata and effective constructor arguments for segmented J-direct/J-geometry/J-H, the preferred locality children, formal3905/3906/2203/2204, and the direct references needed for recovery. Include the core source revision; input widths/features/frame/origin; source/environment measures; E; anchor counts/scales; local/far read law; heads; all loss coefficients/scales; initializer; schedule; query stream; and execution mode.

Distinguish four things: the named model family, the JSON's explicit values, the resolved constructor values, and the actual trained checkpoint. The same family name is insufficient. Missing auxiliary-calibration provenance must read “unavailable,” not “calibrated.”

Make locality strength explicit even when zero in **new** recipes. Require response/operator coefficients to declare either a complete TRAIN calibration record or an explicitly fixed experimental choice. Do not retroactively alter old serialization or claim that old formal fits used their preferred development settings.

### 3.2 Make the displayed fit gap comparable

Use one physical evaluation reducer and an explicit objective definition shared by all new arms. Reconstruct old curves only from available per-component records. Store their original selector conventions separately.

At a small number of reviews, evaluate a fixed metadata-selected TRAIN probe panel and the fixed DEV panel at the **same frozen checkpoint**, using matching role allocation and physical masks. This distinguishes a real fit gap from changing-weights TRAIN averages, auxiliary terms, query composition, and unequal aggregation. This diagnostic does not create independent generalization data or authorize case resampling.

### 3.3 Verify preserved physical meanings

Check Wind frame/rotation, support origin, rotor-relative coordinates, three output components, profile addition, and component scales against the maintained source-resolved adapter. Check Thermal target channels, grid extraction, valid-fluid mask, source heating scale, interface/material order, and q differencing. Geometry-only scene tensors must remain target-free.

Test packing on exactly the same physical examples: a single effective batch versus accumulated microbatches must use the same native query IDs and denominators. Include auxiliary terms in the test; do not reuse a reconstruction-only pass to certify the entire objective.

### 3.4 Small information-path inventory

Run input-only hooks on a few declared TRAIN low/high-M scenes. Record actual MM/ME/EM message rows, typed donor-to-edge rows, return rows, receiver-to-edge rows, and physical-source reads. Export parameter ownership and a small real gradient check. This must expose whether “direct” actually executes pair messages.

Do not perform a new all-formal-model benchmark or full89/full90 rerun here when the attached report already supplies the answer. Reuse immutable arrays where compatible.

## 4. Implement one interaction-preserving joint family

### 4.1 Preserve a true pairwise computational submodel

Build the recovered source/environment context from the existing typed nonlinear message operations, with their signed relative geometry and valid physical measures:

\[
 d_v^{(\ell)}=\sum_{u\in\mathcal N_t(v)}\mu_u\,
 \phi_t^{(\ell)}(z_v^{(\ell)},z_u^{(\ell)},r_{vu},c),
 \qquad z_v^{(\ell+1)}=z_v^{(\ell)}+U_t^{(\ell)}(z_v^{(\ell)},d_v^{(\ell)},c).
\]

The normalization and MM self-exclusion must match the declared operator. An unnormalized sum, a normalized mean, and a measure integral are not interchangeable. Copy the intended maintained implementation rather than inventing a new reduction.

Nonlinear donor/destination/geometry messages are formed before donor reduction. The final receiver still reads individual physical-source states nonlinearly before source reduction. Do not replace either operation with a mean source descriptor and call it equivalent.

Start at H128/message128 and two pair-context rounds. For M30, an MM round has only 900 materialized source-pair slots, compared with 245,760 Q8192 source-receiver slots. This is an arithmetic illustration, not a FLOP or latency guarantee. Preserving MM/ME/EM does not require restoring the expensive historical dense receiver/environment attention and P0/P1/P2 chain.

### 4.2 Add one shallow collective residual, not a replacement backbone

Use at most **one** collective block in the initial organized version, between the two pair-context rounds. It reads the first-round source/environment states, forms explicit typed collective states, and returns a residual update. The second pair-context round then receives the enriched states. Receivers may consult the same prepared collective states through a small residual conditioning projection into their existing source read.

An acceptable starting construction is:

\[
 z^{(1)}=C_{\rm pair,1}(\mathcal X),\qquad
 e_k=\Psi\!\left(a_k,\sum_i B^M_{ki}V_M(z_i^{(1)},r_{ki}),
                         \sum_j B^E_{kj}V_E(z_j^{(1)},r_{kj}),c\right),
\]
\[
 \widetilde z_v^{(1)}=z_v^{(1)}+P_t\sum_k C^t_{vk}e_k,\qquad
 z^{(2)}=C_{\rm pair,2}(\widetilde z^{(1)}),\qquad
 c_q=\sum_k R_{qk}e_k.
\]

The receiver's source-read preactivation receives an additive projection of `c_q`; the original source state and signed source-relative geometry remain explicit inputs. Do not concatenate extra features and silently change the inherited direct function while claiming a zero-residual recovery.

Initialize the collective-to-node and collective-to-receiver output projections to zero. Give their upstream value/scoring networks ordinary nonzero initialization. Zero output projections allow an exact direct submodel at initialization and nonzero first-step gradients for those projections; verify that gradients subsequently reach the collective value and membership modules. Do not zero every layer and create an inert graph.

This is an information-level residual in a shared core, **not** a second complete field network whose predictions are added to a frozen predictor. Every scientific parameter, including the direct path, remains trainable from epoch1. Do not force a residual amplitude away from zero: a useless graph must be allowed to reveal that result.

Start collective width at64 while the preserved direct path remains128. Preserve separate module/environment content and measures. Reuse existing source anchors plus the declared regional anchor inventory, and existing full smooth dense access. No adaptive anchor birth/death, fixed-K target, extra recurrent solver, or multiple graph banks are introduced.

Use the existing return rule initially as a clearly labelled context-sharing rule, not physical transport. The explicit pair path now carries destination-dependent transport. A separately learned directional return can be the single conditional architectural revision only if measured evidence implicates the tied return; do not combine it silently with the initial recovery.

### 4.3 Geometry anchors and learned residual scores

For the organized controls, use registered anchors, explicit physical source identity, valid edge masks, and typed geometry-aware donor content. Geometry supplies a smooth starting prior; learned organization learns a residual to it rather than a wholly unregistered slot label.

The initial learned score can use `Gaussian geometry log-prior + learned residual`, with declared locality strength1 and zero final residual-score initialization. This makes learned and geometric memberships/access equal initially. It is a modeling prior, not a physical law, and the earlier losing Wind locality child means there is no prior evidence of success in Wind. The current experiment changes the preserved transport path and must test the new combination.

Keep source, regional and domain length scales explicit. Do not interpret an anchor's position as the physical center of its learned content. Report weighted donor coordinates/spreads and value contributions beside registered coordinates. Global or long-range influence remains available through the pair path and disclosed global context; do not impose an isotropic nearest-neighbor cutoff as a supposed universal physical law.

### 4.4 Keep joint learning and dataset contracts separate

Thermal has one shared trainable preparation and jointly learned configuration-field and temperature-response heads:

\[
 \widehat f(q)=D_f(Z,q),\qquad K_i(q)=D_T(Z,q,i),\qquad
 \widehat T(q;h)=b(q)+\sum_i K_i(q)h_i.
\]

Current Thermal zero-boundary capability retains its qualified zero offset. Heating must not enter preparation, membership, receiver access, or the configuration-field head. Exact heat-null flow and affine field increments must hold by the actual dataflow, not by detach tricks or corrective overwrites. Nonlinear peaks and ratio-based effective coefficients retain their separate meanings.

Flow and temperature need not share every final projection. The baseline should use a shared relational representation with honest output-specific heads, not require all physical quantities to pass through an unnecessarily identical final feature bottleneck. Keep the initial heads fixed across the three new controls. Changing that sharing boundary is a conditional revision, not a hidden extra degree of freedom.

Wind uses the same context/collective/receiver implementation and its nonlinear source-message reduction and physical velocity head. Retain the TRAIN-fitted profile residual transform and component-balanced loss. Wind does not acquire a Thermal affine control law, zero heat response, or a thermal native-grid stencil.

A prepared object must still bind the actual model, source IDs, receiver IDs, units, operating configuration, measures, transform state, and weights. Changed geometry or weights rebuilds it. Repeated permitted Thermal heating changes may reuse the prepared affine operator; arbitrary Wind geometry changes may not.

### 4.5 Environment resolution and near-source treatment are controlled, not opportunistic

Use ThermalE192 and WindE8 for the **initial recovered family**, preserving the inexpensive established Wind inventory rather than starting another E-resolution search. All new within-dataset controls use exactly the same inventory. The old E64 J fits remain historical comparisons, not an isolated E8/E64 ablation. This intentionally tests recovery of the inexpensive source-resolved architecture plus organization, not a one-variable reconstruction of every old formal difference.

The new provider must allow environment inventory to be configured instead of duplicating the core for E8/E64. Larger E is not the default fix. Check the physical frame, feature construction and measures at the interface.

Audit the old explicit Thermal near/far kernel specialization and the new single affine head. Preserve the native geometry/length features in all cases. Do not restore a separate near specialist only in the graph arm. If it is necessary for a competent direct Thermal model, declare it once in the common readout and give every matched control the same treatment; that uses the conditional revision allowance unless a concrete missing-contract implementation defect is found.

## 5. Establish compatibility without hiding changed training objectives

Use a read-only Wind2202 checkpoint and a few TRAIN scenes to verify that the recovered direct implementation can reproduce its **deployed Full inference function** with mapped compatible weights, including physical transforms and input derivatives. This test may temporarily load old weights in a disposable model. It does not initialize the main experiments, constitute a new fit, or test graph learning.

Separate inference-function parity from optimizer-step parity. The older Full training workflow includes auxiliary base-approximation work even when base/gate do not execute in deployed Full inference. If the new physical objective omits that auxiliary, its training update is not equivalent to the old update. Record this explicitly. Do not preserve unused routing machinery merely to achieve a superficial training-identity claim.

For all new arms, use one common declared physical objective. Inference at collective residual zero must equal the new direct control on the same parameter tensors. With identical objectives, verify reconstruction losses and shared-parameter gradients agree at that initial function, subject to declared numerical tolerances. New residual projections legitimately have additional gradients.

The new Thermal joint direct model has no historical single-network checkpoint that is automatically equivalent to D-sep plus R-direct. Do not invent one. Its fidelity must be established by actual fitting, while the old components remain read-only references.

## 6. The smallest decisive learning comparison

Use descriptive labels below, not reused historical run IDs. Assign new run IDs after checking the local registry.

| Arm | Preserved nonlinear pair context and source-wise read | Collective content | Incidence and receiver access |
|---|---|---|---|
| **P** | Yes | None | None |
| **P-G** | Identical common path | One residual collective block | Fixed smooth geometry |
| **P-H** | Identical common path | Same block and width as P-G | Geometry prior plus learned residual scores |

P-G separates the value of collective regional context from the value of learning its organization. P separates both from the competent direct interaction path. Learned-only scoring parameters are a real capacity difference; disclose them, rather than inserting dormant matching parameters. If a benefit later appears explainable only by extra capacity, use a single parameter-budget-matched direct capacity control within the conditional allowance, not a large new portfolio.

### 6.1 Preserve the segmented data identities

Thermal: fixed25_v1,150 TRAIN/22 exposed DEV, existing source identities and strata, seed0, effective48. Wind: wind_shared_fixed24_v1,72 TRAIN direction rows/24 layouts and24 DEV rows/eight layouts, seed42, effective24. Wind's missing TRAIN M18 and17.1% coverage remain disclosed. Do not resample either cohort because the new model performs poorly. Direction rows remain clustered by layout.

Use Q1024 Thermal and Q4096 total Wind five-role queries, unchanged role allocation, native masks, source measures, component transforms and packing-independent `case_epoch_v1` streams. Native final figures use the already declared full query grids; do not numerically substitute monitorQ for full-gridQ.

All primary fits initialize freshly. Compatible direct tensors are exactly shared initially; P-G/P-H have equal initial collective content, zero residual output, and equal geometric routing before residual scores learn. Record the few intended incompatible tensors. Common physical losses, query IDs, effective batches, schedules and all calibration inputs must match.

### 6.2 Calibrate once, then stop changing the loss

Keep the established Thermal equal flow/temperature family weighting, individual channel scales and native q definition for the initial comparison. Calibrate response/operator coefficients using the maintained TRAIN-only procedure on new initializations, save the numerical receipt, and seal common coefficients for all compared arms. Do not describe a default0.1 as calibrated.

Wind uses one explicit component-balanced role objective for both fitting and checkpoint selection; retain native m/s metrics separately. Per-role scalar historical selectors are archival quantities, not new comparison scores.

No new dropout search, weight-decay sweep, repeated loss reweighting, peak-specific penalty, graph sparsity loss, coordinate-encoding search, or validation-driven operator-weight change is part of this campaign. Any known recipe difference must be stated before fitting, not used afterward to explain away a failure.

### 6.3 Give the direct control a mature fit

Review at100 and500 epochs, then1000. These early reviews check valid inputs, learning, gradient reach, reference fields and trajectory health; they are not requirements to match a mature full-data model immediately.

**P must not again be discarded after a500-epoch screen while only graph arms mature.** Complete a healthy P control to2500 on each dataset. Mature the relevant organized control(s) to the same2500 horizon when they remain scientifically informative and the measured budget permits. Keep endpoint and selected checkpoints separate, with a common100-epoch selection cadence.

Start Wind first for mechanism screening: it isolates the transport/organization question without Thermal's multi-head and affine-operator auxiliary confounds. This is an ordering decision, not permission to omit Thermal joint validation. Finish a selected comparable recovery study in both datasets before claiming a shared direction.

If a graph arm has a sustained large validation regression while its TRAIN fit falls, and its direct control is already improving, do not spend an unchanged5000 run to seek a reversal. Use the one conditional revision only when a recorded hypothesis explains what will be changed and what comparison will decide it.

### 6.4 One conditional structural revision, not another open-ended search

After the first meaningful500/1000 review, allow **one** additional core revision for this campaign. Prefer the smallest change directly implicated by evidence: the shared final read boundary if Thermal flow/temperature gradients compete; the tied gather/return rule if destination-specific content is demonstrably lost; or a common near-source read contract if the initial direct Thermal model misses it. These are alternatives, not a menu to execute together.

Before changing the code, write a short record: observation, alternative explanations, exact changed operation, expected effect, retained comparator, and rejection condition. Fix genuine implementation errors when found, but record every invalidated run and never continue old optimizer state under an unlabelled mathematical change.

A revised model starts a new run. Match the comparator's starting checkpoint, update budget, schedule and newly trained function age, or refit both freshly. Do not compare a revised child with a younger or differently trained direct control and declare causal success.

If P-H is promising, use remaining budget for one predeclared second-seed confirmation of the final decisive pair, prioritizing Wind. Use42/43 for Wind or0/1 for Thermal, with unchanged data and recipe. Do not try seeds until one wins. If the main result is negative, a second seed is not compulsory and may be less informative than closing the result honestly.

## 7. Diagnose generalization through the preserved computation

### 7.1 Do not confuse architectural capacity and usable relational information

At100/500/final, use a small fixed TRAIN probe panel to record per-family shared-gradient norms/cosines for Thermal and per-component/role norms for Wind. Diagnose whether a family is untrained, numerically drowned out, or in sustained conflict. These measurements do not automatically authorize changing weights.

For the graph arms, record residual-to-direct state norms and actual final-output changes when the residual is removed. A very large correction that fits TRAIN but hurts DEV differs from an inert zero branch. A tiny correction can still matter physically; measure its effect, not its latent norm alone.

Check per-M and per-operating-direction errors. Relative displacement and physical scale are the transport variables; absolute domain position and anchor index may be useful context, but should not be the only useful distinction. Do not assert translation/rotation invariance for a problem with fixed walls, inflow and vertical profile. Frame tests must transform the full scene and vector outputs consistently.

### 7.2 A minimal organization test

The final report must compare P, P-G and P-H trained under the same new contract, and use same-weight controls only as supplementary evidence. On a small fixed panel, compare the trained graph with: residual disabled; learned memberships/access replaced by geometric ones; and a physically indexed matched-mass reassignment. Reuse the existing restoration helpers and record all retained direct/global paths.

Display one complete donor→collective→receiver chain in each dataset. Include original source IDs, environment catalogue/measure, registered anchor, weighted donor-content location, receiver, access mass, value magnitude and physical output effect. The drawing may highlight a few edges, but state how many actually execute.

Do not call a receiver-to-edge heatmap a physical interaction graph by itself. Do not call both weights and signed physical responses “importance” without units and definitions. Temperature coefficients and nonlinear Wind messages are different quantities.

### 7.3 Test sparsifiability only after useful dense learning

The default executor remains full smooth dense. No new gather/scatter or compiler project starts here. If a useful graph exists, reuse the implemented100/99/95% cumulative-access approximation and report model distortion, reference error and actual value work separately. A99% access-mass threshold is not a1% physical-error bound.

Also price the valid graph-disabled direct function and the complete organized function. If removing negligible learned collective residuals gives a simpler adequate model, report that result rather than forcing nonzero graph usage. Any thresholded approximation is an explicit separate operation with stale-state and support-boundary semantics.

A promising direction can be accurate with dense execution and show a useful approximation curve. It need not achieve immediate sparse GPU acceleration. It cannot claim accelerated training because a post-fit graph has fewer displayed lines.

## 8. Physical quality and inverse-boundary checks

### 8.1 Field scorecard

Use a fixed readable scorecard, not a growing list of newly invented metrics. Thermal: u/v/p/omega, fluid/surface/materialT, per-module peak error, and native q proxy, with mean andp90 and a concise worst-case row. Wind: five physical roles, vector RMSE and Ux/Uy/Uz, with mean andp95; always show near/downstream transverse components.

Report fixed-weight TRAIN/DEV behavior, native absolute RMSE, relativeL2 on each contour residual, and actual matched case/layout counts. The mean of case RMSEs is not a pooled RMSE. Role mixtures and three directions per layout are not independent physical observations.

Use completed3903/3904/3905/3906 and2202/2203/2204 as properly labelled historical references. Use the strong **segmented** source-resolved checkpoints where available to evaluate recovered P at comparable data exposure. A fullTRAIN reference is not a hard early stopping threshold for a small segmented fit.

### 8.2 Native Thermal consistency

Preserve exact input heat-null flow and qualified affine application. Monitor the native-stencil discrepancy between predictedomega and curl(predictedu,v) on valid fluid stencils, beside actualomega reference error. This exposes inconsistent output representations; it must not be confused with a continuum error certificate.

Do not add a divergence-free penalty: the current reference does not satisfy that assumption. Do not overwrite omega with a derivative just to remove a reported discrepancy. If native-curl readout becomes the single conditional revision, qualify it against the generator's exact definition, boundary/solid masks, physical derivatives and complete cost, and apply the same readout to the matched controls.

Retain the TRAIN-only discrete thermal operator as supervised structure, not as an inference solver. Stored velocity coefficients must not leak into scene inputs. Do not silently replace them with inaccurate predicted velocities and let two output errors reinforce each other while describing the result as stronger physics.

### 8.3 Keep inverse work small and honest

Once final forward checkpoints are selected, replay the existing three Thermal heat-allocation pools and six signed responses, with unchanged identities and quadrature. Show per-module baseline/plus peaks and the predicted versus reference hottest module. Formal3906 already ranked3/3 correctly, so the new campaign cannot claim that achievement is newly discovered or broad inverse readiness.

Do not fit a0291 ranking target or select checkpoints from these final response outcomes. Existing TRAIN response families remain declared training evidence. An old failed0277 baseline remains unavailable and is not replaced with a similarly named nominal record.

Run the existing affine contraction, heat-null, JVP/VJP, source permutation/padding, and stale-context tests on the final pair. For nonlinear Wind provide local model derivatives only; there are no new independently solved turbine-displacement/yaw/thrust labels in this goal. No new inverse trajectories, rank dashboard, or generator training is necessary.

Preserve strict failed numerical checks and role-specific units. Use FP64 accumulation for precise Thermal contractions when requested, without claiming that this turns an inaccurate neural coefficient into a physically correct derivative. Small absolute-temperature differences amplified by the q stencil are numerical conditioning, distinct from factor-of-two field regressions.

## 9. Accuracy–cost decisions and stop rules

These are prospective engineering/research targets for the new matched comparison, not claims of achieved quality and not immutable physical constants. Do not relax them after observing a miss to label a success.

### 9.1 Recovery before organizer promotion

P must be a credible recovered joint predictor, not merely another weak baseline that P-H can beat. Evaluate it against the maintained source-resolved segmented references and disclose any unmatched history/objective. A persistent approximately twofold wake or flow regression is a recovery failure, even if the graph beats the old J-geometry arm. Preserve the strongest references and explain what remains unresolved.

Do not promote P-H on a weighted aggregate that hides another collapse. As an initial guard against P at the matched selected horizon: no priority physical mean should worsen more than10%, and no correspondingp90/p95 more than15%. Priority includes all Thermal flow channels and fluidT/peakT, and all Wind near/downstream velocity components. Other role means/tails remain visible. Report absolute values when a small denominator makes a percentage misleading.

An organizer is worth a further formal research test when it satisfies those guards and provides either a repeatable meaningful physical gain (target roughly5% in a declared difficult field/role versus P, with evidence beyond the fixed-geometry control), or a useful measured accuracy–work/reuse tradeoff at comparable fidelity. Small single-seed gains remain provisional; a selective image or a large removal effect is insufficient.

### 9.2 Spend efficiency headroom deliberately

Measure complete native staging/context/readout/physical conversion/CPU output on the same available GPU with the same input/output scope. Target organized inference no more than1.5 times recovered P. Above2 times P is a red flag requiring at least a substantial, broad physical benefit and an explicit unfavorable-cost conclusion; do not hide it behind a much older slower classic.

Compare complete training epochs, case visits, queries and optimizer updates, not parameter count alone. Include every physical head in new joint training cost. Do not compare one joint fit with only the thermal half of an older two-stage system and claim a complete training speedup.

Report allocator allocated/reserved peaks, resident-model baseline, processVRAM and CPU catalogue residency separately. Benchmark at most two sensible microbatch packings after fixing effective batch and query stream. Retain at least20% of currently available device memory as a safety margin; available memory is determined at execution time, not from a historical GPU screenshot. Larger tensors are not themselves a scientific improvement.

Cache only immutable target-free geometry/catalogues with verified physical identity. Do not cache learned states across optimizer updates, preload targets into inference scenes, or omit cold catalogue setup from the campaign ledger. Do not touch unrelated jobs or assume that a GPU used in the last goal remains authorized or free.

### 9.3 Explicit terminal decisions

At closeout choose one of these outcomes with the completed evidence:

- **Recovered joint predictor and useful collective direction:** recommend one preferred experimental recipe per dataset and prepare faithful manual5000 launch artifacts.
- **Recovered predictor, organizer unproved:** keep the direct recovery as the credible joint reference; preserve the graph result and name one remaining representational question. Do not relabel P as a successful learned hypergraph.
- **Recovery failed:** no new formal-ready recommendation. Deliver the specific failed hypothesis, strongest retained references, and source-level explanation supported by the bounded controls. Do not compensate by launching additional unrelated models.

A manually launchable configuration is not scientifically approved merely because its CPU dry-run succeeds. Separate `constructible`, `development-qualified`, and `formal-completed` in the handoff.

## 10. Execution order and bounded autonomy

### Phase A — identity and computational bridge

Read both reports and relevant sources; record current branch/user changes; create a fresh protected inventory without overwriting earlier receipts. Complete Gate0, bind segmented cohorts and calibration, and implement the actual pair-preserving shared core. Prove zero-residual recovery and native output meaning on disposable TRAIN tests. This phase must end with runnable providers and the real shared engine, not a proposal-only document.

### Phase B — controlled segmented learning

Run the P/P-G/P-H comparison on the fixed cohorts with common declared objectives and initialization rules. Start with Wind mechanism checks, then the Thermal joint comparison. Review100/500/1000; mature healthy P and the decisive organized comparator(s) to2500. Monitor all important physical roles from the first review. Use existing catalogues to avoid repeatedly paying unnecessary cold setup.

### Phase C — one evidence-guided correction if needed

Use at most the one conditional revision defined above. Maintain a proper new-function lineage and a matched comparator. No shape-compatible silent resume under a changed graph or output law. Refuse a formal launch as a substitute for unresolved segmented evidence.

### Phase D — selected-checkpoint evidence and delivery

Perform the small native field, graph, response and cost panels only on actual selected final checkpoints; reuse equivalent endpoint results. Use remaining science budget for the one relevant second-seed confirmation when warranted. Finish before the closeout reserve.

Do not leave a campaign-owned process running beyond the stated goal unless the user separately asks for that behavior. A clean stop requires a confirmed resumable checkpoint and actual process status; a crashed/externally interrupted process is not relabelled gracefully paused.

## 11. Implementation map and tests

Prefer reusable additions to the maintained core/providers rather than copied per-dataset training programs. Preserve legacy classes and serializers; a new implementation may live in a focused `interaction_preserving_joint.py` module instead of growing the already large `joint_regional.py` with many mode-specific branches.

| Area | Required change or reuse |
|---|---|
| `interface_fields/interaction_core.py` | Reuse/reference actual MM/ME/EM and physical-source read feature construction; preserve old checkpoints |
| `interface_fields/joint_regional.py` | Reuse registered anchors, typed measure handling, inspectable records and restoration machinery where suitable; do not reuse its pooled J-direct as the competent direct path |
| New joint core | One trainable pair-preserving context, one optional residual collective block, native nonlinear/affine output capabilities, explicit zero-residual submodel |
| Thermal joint adapter/provider | Preserve native shared-grid extraction and dependency laws; all physical heads train; coefficients/calibration fully bound |
| Wind joint adapter/provider | Configurable geometry-onlyE inventory, preserved physical frame/profile transform and identical component-balanced reducer |
| `honf_runtime/unified_training.py` | Reuse ordinary engine, named optimizer states, packing-independent sampling, checkpoint selection and clean-stop behavior; no second engine |
| `tools/joint_regional_train.py` or a small supported extension | Explicit new-family dispatch and resolved recipe validation; no implicit scientific defaults in new final recipes |
| Focused tests | Pair operations actually execute; no early-pooling substitution; zero-residual recovery; native outputs; all-head optimizer reach; query/packing invariance; masks/IDs/measures; stale-state guards; serialization and exact same-recipe resume |

The required negative tests include: a deliberately missing locality value in a new learned recipe; an auxiliary coefficient labelled calibrated without its record; a target tensor in the inference scene; missing source IDs; changed transforms/geometry/weights reused under a stale context; changed mathematical recipe under exact resume; and Wind exact-affine-increment requests. Old legacy tests retain their original compatibility semantics.

Inspect all relevant existing failures. The previous14 inherited failures and numerical limits are not evidence that new failures may be ignored. Report affected-test regressions separately from unrelated missing-resource fixtures. Do not broaden the task into repairing every legacy research fixture.

## 12. Deliverables and presentation

### 12.1 One compact main report and one technical appendix

The main report begins with three plain answers: **Are the fields reliable enough relative to our strong reference? Does learned collective organization add something? What does it cost?** State the actual terminal decision. Do not begin with file counts, audit hashes or a dense alphabet of qualification codes.

Retain at most six principal figure families:

1. Fixed-weight TRAIN/DEV learning and physical-role scorecard; identify successful and failed recovery rather than only total loss.
2. Native ThermalT/omega and WindUx/Uy/Uz fields and signed residuals, with a quantitative relativeL2 annotation on every error panel; fixed near/downstream cuts beside them.
3. One actual source/environment→collective→receiver example per dataset, including content locations, all retained direct paths and a clear statement of executed versus merely highlighted edges.
4. P/P-G/P-H and same-weight collective interventions, with reference error and physical output effects; restoration alone is not benefit.
5. Accuracy versus complete inference/training cost and memory, with any post-fit truncation clearly labelled an approximation.
6. The small existing Thermal response/peak table and Wind derivative boundary; no new inverse-design image without an actual design experiment.

Choose representative maps by the existing input-only rules. A worst-case diagnostic may additionally show a failure explicitly selected by error, but cannot replace the representative panel or select the recipe. Use common field/residual scales across compared models, state any visual clipping, and compute metrics on unclipped valid arrays. Preserve low-amplitude transverse components rather than allowing a large streamwise background to hide them.

Detailed tables, initializations, calibration, work counters, and numerical qualifications belong in the appendix/receipts. Report paragraphs and captions remain single continuous source lines under the repository formatting rule.

### 12.2 Faithful formal preparation, only if development warrants it

Generate the manual5000 configuration from the **resolved preferred development specification**, not by copying a sibling JSON and switching its mode. Explicitly list every changed formal setting: original TRAIN population, newly fitted TRAIN normalization and auxiliary calibration,5000 horizon/schedule, validated batch packing, and monitoring scope. Architecture, output laws, E inventory, locality rule and read specialization must either carry over or be separately justified.

Run an inert full-data constructor/forward check with optimizer construction blocked. Print the effective locality strength, environment shape, active head inventory, exact loss coefficients and calibration source, initial model identity, and zero external learned-model dependencies. Missing formal calibration blocks readiness rather than falling back to0.1. Store a concise development-to-formal diff and require it in the manual command path.

Do not claim that fresh formal training reproduces a branched development history. Do not launch it. Provide one clear manual command per preferred dataset recipe only after the fidelity/cost decision, with status/stop/resume instructions reused from the maintained workflow.

### 12.3 Preservation and repository closeout

Protect the exact inherited647 logical bindings **and** the newly completed3905/3906/2203/2204 runs and current user-owned files;647 is a historical minimum, not proof that every newer artifact was already covered. Inventory membership, resolved paths and hashes before work, preserve known archival links, and compare the same membership at the end. Keep the older inventory-evidence limitations disclosed.

Solver usage remains326/326 with zero new attempts. WindTEST stays locked. Do not delete historical science or reports, move run directories to simplify cleanup, or overwrite failed attempts. Generated checkpoints, arrays, figures, profiling traces and one-time renderers remain local in ignored paths. Commit only durable core/provider/runtime changes, focused tests, configurations and report/guide sources.

Before pushing, inspect the current remote branch and audit the **entire outgoing commit range**, including files added then deleted. Use the repository artifact hook; never force-add generated outputs. Preserve unrelated user changes, push the non-default working branch without destructive history edits, and verify local/remote tips. The final response must name completed learning ages, the actual preferred or failed hypothesis, measured physical/cost tradeoffs, remaining limits, and the exact manual-readiness status.

## 13. Source index for this review

The two main reports above are the primary numerical sources. Key formal sections are “What can be compared fairly,” “Predictor: physical field fidelity,” the current inference-cost table, actual organization/intervention sections, and the bounded response table. The development report's protocol, actual eight-run index, preferred-recipe section and numerical interface section establish the different locality-child history and the short J-direct screening age.

Code was inspected at `8e7c1704f1590b7038959fafb4b7cdfc01fb56e5`: `interface_fields/joint_regional.py` (blob `4cdf72c2c2e82395c98d603cc2196e60e6c31a06`), `interface_fields/interaction_core.py` (`af4b3763376a264cd6e63b08362ab8b063ad419f`), `interface_fields/geometry_flow_field.py` (`c727cb05ed38e4ff08c7b2d85d4cb6878d710450`), both dataset `joint_regional.py` adapters, both `training/joint_task.py` providers, `tools/joint_regional_train.py`, `docs/guides/Joint_Regional_Field_Development.md`, and the formal J-H recipe. All proposed remedies remain hypotheses to test; none is a promised accuracy gain.

**Central correction:** preserving physical source IDs is necessary, but preserving useful nonlinear relations is also necessary. Let a hypergraph organize those relations and expose reusable collective context; do not make the predictor forget a competent interaction path merely to produce a more explicit graph.
