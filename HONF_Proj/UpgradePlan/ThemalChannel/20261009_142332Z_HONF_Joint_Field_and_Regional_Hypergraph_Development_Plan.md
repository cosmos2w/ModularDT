# HONF next phase: jointly learned fields and explicit regional hypergraph organization

**Mode:** Codex Goal mode, bounded autonomous development on the existing segmented datasets. **Reviewed repository:** `cosmos2w/ModularDT`, branch `agent/honf-core-next`, commit `0db193cfbba67e8bb2695b02d8f092dce707f25c` (9 October 2026). **Primary evidence:** the updated `docs/reports/_bk/20261008_004514Z_HONF_Unified_Training_and_Adaptive_Interaction_Refinement_Report.md` and `docs/reports/_bk/20261008_182414Z_HONF_Wind_Fidelity_and_Shared_Execution_Consolidation_Report.md`. **Status:** proposed work; architectural benefits and experiment outcomes below are hypotheses, not completed results. Paths are relative to `HONF_Proj/` unless stated otherwise.

## 0. The correction and the purpose

The target is a **jointly learned, efficient, multi-field HONF forward model with a useful, explicit organization of physical module–environment information**. Reconstruction teaches that organization; subsequent inverse consumers reuse its fields, source identities, prepared state and derivatives. Sparse execution is a possible benefit, not the definition of scientific success.

The previous phases achieved shared implementation and a shared optimizer workflow, but the Thermal model still composes a newly trained temperature predictor with frozen D-sep3901. That was an intentional limitation of the earlier preservation plan, not an accidental training omission. The user's new instruction supersedes that restriction for the new experimental family. A common wrapper, a single command, or embedding two independent frozen models inside one checkpoint does not meet the revised goal.

The next candidate must learn flow and temperature **together from fresh initialization**, with one genuinely shared trainable source/environment interaction representation, one training lifecycle, and one self-contained checkpoint. No external learned flow checkpoint may be required for candidate inference or initialization. Wind must execute the same interaction-block and receiver-context implementation, with its own learned weights and its nonlinear output adapter. Reuse the existing runtime and native data machinery; do not rebuild the research platform.

The architectural question is no longer mainly “which expensive per-source correction can be skipped?” It is: **which combinations of modules and environmental regions should be represented together, and which of those collective representations should a physical receiver consult?** The proposed direction is a spatially registered, overlapping hypergraph context with source-preserving readout. It replaces the new family's cheap/base/fine/router stack rather than being appended on top of every old mechanism.

Codex should implement, train, inspect and make up to two evidence-guided architectural revisions before final delivery. Do not stop at an architecture diagram, startup tests, or a first report of mixed results while the authorized bounded learning work remains healthy and affordable. Do not start a full-TRAIN 5,000-epoch run in this goal; the user will decide when to launch that separately.

### 0.1 Non-negotiable outcomes

1. **Real joint Thermal learning.** Flow, thermal and shared interaction parameters all receive appropriate physical training signals and update. Heat-independent flow remains a mathematical input dependency, not a frozen-model workaround.
2. **An explicit collective representation.** Export actual node-to-hyperedge memberships, collective edge states, edge-to-node updates and receiver-to-edge access used in prediction. A renamed pair gate or post-hoc clustering is insufficient.
3. **A small general core.** The same block works in Thermal and Wind. Dimensions, source features, units, masks, output laws and supervision remain adapter-owned.
4. **Measured development, not promised improvement.** Compare the new mechanism with a jointly trained direct control and a geometry-organized control; mature healthy selected candidates; show physical fields and practical costs.
5. **A clean continuation decision.** Deliver one preferred architecture/configuration per dataset, clear retained weaknesses, and a tested manual-only formal recipe. It is acceptable for the two datasets to use different widths, environment resolutions or output heads while sharing the same core.

### 0.2 What is explicitly not required

No universal win over Dense1804 or Dense2103, no mature causal hypergraph, no forced small K, no immediate sparse wall-clock win, no new inverse generator, no new physical solves, and no demonstration of shared weights or zero-shot cross-dataset transfer. Do not substitute any of these stronger claims for the actual task.

## 1. The evidence we are building on

These are reported measurements at the reviewed revision. Reuse their saved arrays where bindings and protocols match; do not rerun every mature reference at each development review.

| Evidence | Reported result | Consequence for this phase |
|---|---|---|
| Thermal3904 identity | Hard source refinement, H64/message64, Q1024, with frozen D-sep3901; 600 TRAIN cases and 5,000 thermal epochs | It is not a fresh jointly learned full-field model. Preserve it as an incumbent, not the new architecture definition. |
| Thermal fluid temperature | Mean native RMSE: 3904 0.306155; 3903 0.321108; 1404 0.293284; 1501 0.254890; Dense1804 0.237375 | 3904 is approximately 4.39% worse than1404 and 28.98% worse than1804 on this role. Do not hide this behind improved surface/material errors. The latest headline does not directly compare1401. |
| Thermal other roles | 3904 surface/material/peak/q-proxy means: 0.291930/0.294018/0.319729/1.421349 | These strengths are preservation targets; peak tails remain important. All temperature roles come from one native temperature field. |
| Thermal flow | Shared D-sep3901 u/v/p/omega means: 0.00432883/0.00045946/0.00197505/0.0375592 | Strong u/p and weaker v/omega belong to the frozen flow component, not to3904's learned thermal organizer. |
| Thermal heating response | 3904 fluid/surface/material/q means: 0.008019/0.014209/0.017397/0.050073; each is worse than3903, but below3902 | Nominal field improvements do not prove improved responses. Keep the existing response audit separate. |
| Thermal0291 decision | Reference baseline/plus peak:18.520241/18.129892;3904:19.202501/19.237526 | The model predicts the wrong hottest module in the plus state. Its physical-hot-module response is already close:−0.370813 versus−0.390348. This is not simply an incorrect heat-gradient sign everywhere. |
| Wind2202 | H128/message128, E8, component-balanced Q4096, Full-detail;420 TRAIN rows;5,000 epochs | A substantially improved fast nonlinear predictor, not a trained sparse organizer. |
| Wind native vector means | Volume/hub/downstream/near/background:0.023096/0.043991/0.054136/0.092951/0.011144 m/s | Better than K6 on all five means; Dense2103 remains better. Preserve this progress rather than portraying Wind as still at2201 quality. |
| Latest inference, same GPU1 | Thermal3904 selected41.066ms versus1404 53.462 and1804 159.735; Wind2202 12.452ms versusK6 63.037 andDense41.258 | There is room to spend some efficiency on a better representation. These are actual-system comparisons, not isolated sparse-kernel speedups. |
| Latest training | Complete outer wall4.162h Thermal3904 and7.567h Wind2202; sampled process GPU peaks7244/23524MiB | Wind training is not an almost-empty-VRAM workload. Thermal cost excludes training the already frozen3901. Benchmark new joint costs honestly. |
| Current organizing evidence | Thermal3904 learned routing wins fluid T on three fixed equal-degree panels; Wind2202 router has no optimizer state | Retain the useful Thermal routing observation, but do not use an inactive Wind router as evidence of adaptive organization. |

Thermal canonical89 is repeatedly exposed validation, not an independent test. Wind VALID90 is30 layouts with three correlated directions each; WindTEST90 remains locked. Existing response directions, material points and repeated queries do not add independent physical experiments. The solver ledger remains326/326.

## 2. Code review: what is confirmed and what remains a hypothesis

The following implementation paths were inspected at the reviewed commit. Local GPU re-execution was not part of the review that produced this plan. Codex must verify the live checkout before modifying these paths.

| Source | Confirmed implementation fact | Implication |
|---|---|---|
| `Case_ThermalChannel/src/channelthermal/training/unified_formal.py`, `_load_formal_sources` | Requires the exact immutable3901 D-sep hash, uses3902 architecture metadata, and rejects the optional flow-context variant | Create a new joint recipe/factory rather than weakening or silently repurposing this legacy identity. |
| `Case_ThermalChannel/src/channelthermal/training/unified_task.py`, `_case_reconstruction`, `loss_terms` | Current reconstruction trains fluid/surface/material temperatures and q-proxy; there are no learned flow-output reconstruction terms in this provider | The new provider must add real u/v/p/omega targets and losses, not just unfreeze an unused object. |
| `src/honf_forward_core/interface_fields/interaction_core.py`, `InteractionContextCore.prepare_context` | Two real source–source and source–environment/environment–source rounds update both node types; a global state pools them | The old core already models contextual multi-source interactions. Do not falsely describe it as independent-heater summation or as lacking environment information entirely. |
| Same file, `_read_features`, `_read_selected_features` | Receiver/source features use source state, global state, query geometry and relative geometry; they do not directly query individual updated environment states | Receiver-accessible regional context is a substantive missing path to test. Its absence is a bottleneck hypothesis, not a proven explanation of fluid-T errors. |
| Same file, `NonlinearFieldReadout.predict` | Nonlinear source messages are reduced before a nonlinear physical field head; environment reaches it through prepared source/global context | Preserve nonlinear Wind readout semantics, but expose collective spatial context to the receiver before source-specific readout. |
| `src/honf_forward_core/interface_fields/source_response_operator.py` | Direct source coefficients and precise forcing application exist; old group mode explicitly assembles far K from receiver functions and memberships | Reuse affine arithmetic and native source identities. Do not revive the old `mode='group'` as the whole proposed advance or impose its separable output form. |
| `src/honf_forward_core/interface_fields/interaction_refinement.py` | Current organizer selects/blends B and F per source/query; hard and C1 variants and selected/dense-masked backends are implemented | Keep these as historical/control paths. The new main family should not require B, F, a gate, a base teacher and a work-calibration schedule in addition to a hypergraph. |
| `Case_ThermalChannel/src/channelthermal/source_response.py` | One grid-temperature field is extracted through native stencils to fluid/surface/material/ports; heat enters separately; units and zero-boundary assumptions live here | Extract/reuse these adapters. Do not create separate inconsistent temperature predictors for different roles. |
| `Case_WindFarm/src/windfarm/shared_interaction.py` | Nonlinear physical velocity uses a TRAIN-fitted height profile plus scaled residual; public derivatives and ownership checks exist | Keep the profile clearly identified as empirical. Preserve coordinate/vector frames and nonlinear capabilities. |
| `src/honf_runtime/unified_training.py` | `TaskProvider`, `LossTerm`, named optimizer state, strict resume and packing-independent `case_epoch_v1` sampling are available | Extend this engine minimally. Do not build a second optimizer loop or carry recipe-specific loss descriptions into supposedly generic reporting. |

### 2.1 The central testable hypothesis

Currently, the updated environmental latent representation has to be summarized into the physical source states and a pooled global state before a receiver reads it. These vectors can encode spatial effects, so there is no proof that the model is incapable of representing the solution. Nevertheless, explicitly retaining receiver-addressable collective regional states is a plausible better inductive bias for plume transport, wake overlap and surrounding geometry. This is the central architecture experiment, not another indefinite loss/width sweep.

### 2.2 Correct the meaning of unified

Track four independent properties: a common software core, a common optimizer workflow, jointly learned fields within a dataset, and shared parameters across datasets. The existing work substantially achieved the first two. This phase requires the third. It does not require the fourth.

Within Thermal, physical flow independence from heating and joint parameter optimization are compatible. At fixed trained weights, no heat input reaches flow. Across optimizer steps, both flow and temperature supervision improve the same geometry/context representation. Do not confuse these two derivatives.

## 3. Proposed architectural direction: joint regional hypergraph fields

Use one new opt-in family, provisionally `joint_regional_hypergraph`. This is a working implementation name, not a novelty or success claim. Reuse simple encoders, geometry features, native extraction and the runtime. Replace the core organizing computation in the new family rather than stacking it over the old port loop, D-sep, R-direct, B/F gates and a new graph.

### 3.1 Minimal executed structure

```text
Whitelisted physical modules + environment + operating context
                         |
                 Typed node encoders
                         |
       Two shallow node -> hyperedge -> node blocks
       (overlapping, spatially registered collective states)
                         |
          Receiver -> regional-hyperedge access
                         |
        Source-preserving contextual receiver read
                         |
       +-----------------+----------------------+
       |                                        |
 Configuration-field head                 Affine control head
 Thermal: u,v,p,omega                     Thermal: K(q,i), then K h
 Wind: Ux,Uy,Uz                          only when adapter-qualified
       |                                        |
       +--------- Native output extraction -----+
```

All learned blocks in this new path are jointly trainable. There is one shared context preparation per scene and one joint model checkpoint. Head-specific weights are normal output parameterization; they are not separate pretrained full-field networks. Thermal has one learned temperature field and one configuration-field block, not seven unrelated regressors.

### 3.2 Physical nodes, registered anchors and hyperedges

Keep the physical node catalogues `M` and `E`, their masks, units, coordinates and measures. Let an interaction anchor identify a location and a scale, not a free latent slot ID. Use all active module centers as one anchor type plus a modest geometry-defined regional anchor set covering the environment. Source anchors grow with physical source count; regional anchor density is a declared resolution parameter. Neither is a learned dynamic-K result.

A reasonable initial configuration is one source anchor per active module plus16 regional anchors in Thermal and32 in Wind. These are starting engineering resolutions, not forced effective group counts or scientific targets. Seal them before the first scored fit. Permit one resolution revision later only under the bounded procedure in Section8. Choose regional anchor positions from prescribed domain support/geometry, independently of requested query batches, target values, errors and source array order. Prefer a stable reference lattice or coverage construction over a per-forward recursive tree. Store physical anchor coordinates and type/identity mappings.

Thermal should retain its192 environment records. The new Wind core should have configurable environment sampling; start the new matched architecture comparison with64 geometry-only records, while explicitly retaining2202's E8 as a historical baseline. **All new matched controls must use the same E64 inputs**, so learned hypergraph gains cannot be attributed simply to receiving more environment records. E64 is a geometry representation choice, not64 observed flow sensors. Reuse the existing target-free token construction at the new quadrature shape. Do not invent physical fields at those points.

Initialize each edge state from its physical anchor coordinates, scale, type and prescribed scene context through a shared encoder; a source-anchored edge may also use that source's encoded attributes. Do not use a learned embedding indexed by arbitrary edge number. The same initialization rule applies to J-geometry and J-H. Keep physical module count and total physical measures available, but do not encode the number of quadrature atoms as a physical quantity: splitting one environment atom into identical attributes at the same location with divided measure must not change that quantity.

Each hyperedge gathers from both node types and may overlap other edges. A module can influence many edges, and a receiver can access many edges. The near/source residual path remains available; there is no requirement to compress all physical information through a small global bank. Do not use an identity embedding for physical slot number to manufacture differentiated groups.

### 3.3 Learn collective information, not merely support pictures

Each block must execute these operations: typed source/environment membership, nonlinear content encoding, collective hyperedge-state update, and a residual update back to nodes. Receiver access then consults those collective states during actual field inference. Removing or replacing these paths must be a real operational intervention.

Avoid a sum of scalar gates that only rescales an otherwise unchanged direct predictor. A hyperedge must carry learned content combining more than one donor where the data support that behavior. However, do not force every edge to mix multiple modules or require an entropy/diversity histogram. Singleton or weakly used edges can be appropriate. Report how much collective use actually occurs.

Use two shallow context blocks initially. Do not add recursive coalescence, an inner optimization solver, recurrent convergence loops, another global full-field network, or query-by-query graph construction. A third block is a possible single bounded revision, not the default.

## 4. Mathematical and numerical contract

These equations define a minimal reference design and its invariants. Codex may refine parameterization within the allowed architecture family, but must preserve the semantics and document any change before training the revised child.

### 4.1 Scene and typed membership

Write the target-free configuration as

\[
\mathcal X=(\{x_i,a_i,\mu_i,\mathrm{id}_i\}_{i\in M},
\{e_j,b_j,\nu_j,\mathrm{id}_j\}_{j\in E},c).
\]

Thermal heating `h` is a separately applicable control and is absent from this configuration representation. Wind's configuration controls and prescribed metadata follow its own adapter. Observations and physical reference fields are not scene inputs.

Let `z_v` denote encoded node content and `(a_e,ell_e)` a registered hyperedge anchor. For node type `t`, an example normalized membership is

\[
 B^{t}_{ev}=
 \frac{\mu_v^{t}\exp s_t(z_v,z_e,\xi(x_v-a_e,\ell_e))}
 {\sum_{w\in V_t}\mu_w^{t}\exp s_t(z_w,z_e,\xi(x_w-a_e,\ell_e))}.
\]

Implement stable masked normalization and zero output for an empty type. Physical measure appears once. Distinguish membership density from the resulting normalized donor mass in exports. Keep type masses/count information available so normalized pooling need not erase source number or environmental volume. Do not use row count in place of quadrature mass when the adapter supplies measures.

The primary learning path should use smooth memberships. Dense attention is acceptable at this small preparation scale. Exact support truncation comes after a useful representation exists, not as a mandatory first-stage penalty.

### 4.2 Collective update and residual node update

A representative block is

\[
 u_e^{t}=\sum_{v\in V_t} B^{t}_{ev}\,
            \phi_t(z_v,\xi(x_v-a_e,\ell_e)),
\qquad
 z_e^{+}=z_e+\rho(z_e,u_e^M,u_e^E,c),
\]

\[
 z_v^{+}=z_v+\psi_t\!\left(z_v,
              \sum_e C^{t}_{ve}W_tz_e^{+},c\right).
\]

`C` is a normalized node-to-edge receiving weight with the appropriate geometry/masks. It can reuse and renormalize the incoming incidence initially; it need not introduce another independent network. Nonlinear donor encoding precedes aggregation, the collective update is nonlinear, and original source states retain residual identity. These operations give the graph actual computational content. They do not establish a fundamental many-body physical law.

Regional/global information should be retained through the shallow shared blocks, not by adding a duplicate mature predictor. A small pooled global context is permitted and must appear as an explicit bypass in the graph diagram and interventions.

### 4.3 Receiver-addressable regional context

For each physical receiver `q`, compute

\[
 R_{qe}=\operatorname{Normalize}_{e}
     \left[\exp r(z_e^{+},\xi(q-a_e,\ell_e),c)\right],
\qquad
 c_q=\sum_e R_{qe}V z_e^{+}.
\]

Use low-width query/key projections and batched reductions where possible. Do not create a hidden-width `Q x E x M` tensor. Receiver weights must not depend on the other receivers present in the minibatch. A point must predict the same field when requested alone, in a permutation or in another tile, within declared numerical tolerances.

For Thermal, temperature at one physical point cannot depend on whether the caller later calls it a fluid, surface or material sample. Those roles are native extraction/supervision views of the same temperature field. Query type can identify an output block, but it must not create contradictory temperatures at coincident points.

### 4.4 Source-preserving readout: do not reinstate the old separable output restriction

Keep an individual physical-source read after collective contextualization:

\[
 m_i(q)=\Phi(z_i^{+},c_q,z_g,\xi(q-x_i),c).
\]

A Thermal configuration-field head predicts

\[
 [\hat u,\hat v,\hat p,\hat\omega](../q)
 =D_f\!\left(c_q,z_g,\operatorname{Reduce}_{i}[m_i(q)],q,c\right).
\]

A scalar coefficient head gives

\[
 K_i(q;\mathcal X)=D_K(m_i(q),c_q,\xi(q-x_i)),
\qquad
 \hat T(q;\mathcal X,h)=b_T(q;\mathcal X)+\sum_i K_i(q;\mathcal X)h_i.
\]

The current Thermal zero-boundary capability keeps `b_T=0`; other affine datasets require explicit nonzero-boundary support. Reuse the native near/far smooth blending only if needed for the inherited source-response behavior. It is not another learned optional B/F gate. Prefer one shared source-feature evaluation with small output projections over multiple independent large per-field networks.

The important distinction from old R-group is that `K_i` is still decoded for physical source `i`, conditional on receiver-local collective context. It is **not required** to equal `sum_e R(q,e) B(e,i)`. Old R-group's number of modes was not always smaller than M, so do not falsely claim a guaranteed low-rank deficiency in every historical case. The avoided restriction is the imposed separable far-output parameterization and compulsory loss of source detail through that representation.

Wind executes the same node/edge/receiver/source functions and uses

\[
 \hat{\mathbf U}(q;\mathcal X)
 =D_U(c_q,z_g,\operatorname{Reduce}_{i}[m_i(q)],q,c)
\]

with its maintained physical residual/background transformation. `D_U` is nonlinear; no exact finite heating law is transferred to Wind. Separate dataset weights are expected.

### 4.5 Joint learning without illegal physical coupling

All shared context and receiver-representation parameters participate in both Thermal loss families. All flow heads and thermal heads are optimized during the same ordinary training lifecycle. At fixed configuration and fixed learned weights,

\[
 \partial(\hat u,\hat v,\hat p,\hat\omega)/\partial h=0,
 \qquad
 \partial\hat T/\partial h_i=K_i,
 \qquad
 \Delta\hat T=K\,\Delta h.
\]

These properties follow from actual input exclusion and affine application, not detach tricks, gradient hooks, an output overwrite, or a flow-null penalty. Temperature supervision can update the shared geometry encoder; this does not imply that a user's heating input changes predicted flow at inference. No explicit predicted-velocity-to-temperature edge is mandatory: the first new hypothesis is a shared, jointly supervised latent representation with regional query access, not a repeat of the unsuccessful frozen-flow feature injection.

Group support must not depend on current heating in the affine Thermal capability. It can depend on geometry, material, operating context and receiver position. Other datasets can allow their nonlinear configuration controls into the same generic block. Do not advertise the Thermal dependency graph as universal multiphysics.

### 4.6 Precise output, derivative and reuse semantics

Preserve physical source IDs, source ordering and control units in every response object. Widen factors before native contraction and stencil differences when using the existing precise mode. Separate model FP32 accuracy from accumulation precision; do not report float64 contraction as a float64 neural model.

Configuration changes rebuild context and receiver-dependent factors. Weight updates invalidate prepared state. Heating changes alone can reuse a qualified Thermal configuration/kernel. Wind returns exact nonlinear endpoints and local JVP/VJP, not an affine finite-response promise. Cache keys must cover new edge anchors, node catalogues, units, transforms and parameters; extend the current guards rather than inventing another ownership framework.

The joint prepared object must expose capabilities by output block: configuration-field endpoints and derivatives, Thermal affine temperature/control increments, and adapter-derived quantities. Do not mark the entire multi-field model as a universal affine operator, and do not lose exact temperature increments merely because flow is nonlinear in geometry. A heating-increment request returns zero for the heat-independent flow block and the qualified K contraction for temperature; derived ratios and sampled maxima retain their actual, potentially nonlinear semantics. Reuse and minimally extend the existing dependency/capability records instead of inventing a universal multiphysics solver.

The native flux proxy remains the existing conductivity/delta stencil operation. Preserve its stricter numerical limitation in old checkpoints. The new family must test its own complete output parity and derivatives; matching T within a few e−6 does not automatically qualify q-proxy.

## 5. Keep the architecture small and the ownership explicit

### 5.1 Intended durable changes

| Area | Required change |
|---|---|
| Generic core | Add one coherent joint regional-hypergraph block/family, preferably under `src/honf_forward_core/interface_fields/`. Reuse geometry, masks, measure normalization and projection primitives. |
| Joint prepared object | A small shared state with node/edge/receiver factors and output-block capabilities. Reuse existing version/signature guards and response arithmetic. |
| Thermal adapter | Reuse context construction and native stencils, but own one joint model and native flow-plus-temperature extraction. No external D-sep object. |
| Thermal provider | Reuse sampling/targets/operator/response records; include actual flow loss and all joint model parameters in the optimizer. |
| Wind adapter/provider | Execute the same graph implementation with3-D geometry and nonlinear head; preserve TRAIN profile and component balancing. |
| Runtime | Reuse `TaskProvider`, `LossTerm`, `case_epoch_v1`, named optimizer state and strict resume. Add a simple joint-learning schedule mode only where the old open/soft/hard gate curriculum is hardcoded. |
| Reporting | Move recipe-specific loss definitions to provider metadata where necessary. Do not describe the new multi-field loss with the old temperature-only formula or label exposed panels as independent held-out tests. |
| Configuration | New versioned joint family/recipes. Preserve all old config identifiers and checkpoint loaders. |

### 5.2 Complexity guardrails

The new main path contains typed node encoders, two shallow collective blocks, one receiver-context read and small source-preserving output heads. It does not contain the old P0/P1/P2 loop, a frozen Stage-A model, D-sep, an old R-direct prediction added to a correction network, a second complete field predictor, B/F distillation, a router surrogate and a new graph all at once.

Do not create inactive matching parameters to claim equal capacity. Do not zero-initialize an entire collective branch in a way that prevents membership learning for hundreds of epochs. A small stable residual scale is acceptable, but verify actual task gradients in node membership, collective update and receiver access during preflight.

A small number of physically typed heads is not prohibited complexity. A growing collection of ad-hoc error-correction heads, per-case gains, special0291 branches, independent temperature-role models or optimizer-specific wrappers is prohibited.

## 6. Training data, objectives and selectors

### 6.1 Data scope and initialization

Use Thermal `fixed25_v1`:150 primary TRAIN cases and22 exposed DEV cases, with its existing original-TRAIN response addenda. Use Wind `wind_shared_fixed24_v1`:72 TRAIN direction rows from24 layouts and24 DEV rows from8 layouts. The Wind subset is17.1% of420 TRAIN rows, not literally a quarter; retain its documented missing M18 category. Do not silently resample it or open WindTEST to make evaluation look cleaner.

All scientific new joint controls/candidates start from fresh weights and fresh optimizer moments. Reuse dataset-owned normalization procedures fitted only on their selected TRAIN populations. Existing development parents are read-only contextual references; formal3901/3902/3903/3904 and2201/2202 must not initialize segmented training. A model checkpoint is not needed to discover input widths or reconstruct the split; use maintained metadata/configuration instead.

Preserve case membership, query streams, losses, epochs and optimizer budgets across each main architecture comparison. Use `case_epoch_v1`. Change microbatch packing only within a fixed effective batch after testing denominators and gradients. Native source/query padding must not enter loss normalization. This is particularly important because the old Thermal auxiliary base reduction changed with union padding; the new family removes that base objective rather than inheriting the discrepancy.

### 6.2 Initial numerical recipe

Start with H128/message128 in both new datasets, FP32 for neural execution, two collective blocks and receiver training tile512. Thermal retains Q1024 primary fluid draws, its native surface/material sampling and128 operator rows; Wind retains Q4096 with the established five-role split and component-balanced objective. These are shared-within-comparison settings, not universal optimal values.

Use effective batch48 Thermal and24 Wind. Choose the largest measured-safe microbatch up to that effective batch; do not increase the effective optimizer batch simply to occupy VRAM. A startup TRAIN-only resource benchmark may choose microbatch and execution tile, but not modify scored datasets or select model width from validation outcomes. Preserve at least20% of currently available device memory plus measured transient overhead; do not use old GPU leases or another user's running job.

A standard new2500-epoch schedule is sufficient: short learning-rate warmup (for example20 epochs), peak3e−4, hold through1000, cosine decay to3e−6 at2500, AdamW and gradient clipping1.0. **All physical heads and shared representation parameters train from the start.** There is no500-epoch flow-freeze or base/router curriculum in this new family. Any temporary numerical warmup is shared across models and does not turn into a separately trained frozen flow stage.

### 6.3 Thermal loss: learn every output without hiding fluid temperature

Use explicit group-normalized physical objectives:

\[
 L_f=\tfrac14\sum_{c\in\{u,v,p,\omega\}}L_c/\sigma_{c,\mathrm{TRAIN}}^2,
 \qquad
 L_T=\tfrac13(L_{T_f}/\sigma_f^2+L_{T_s}/\sigma_s^2+L_{T_m}/\sigma_m^2),
\]

\[
 L=\tfrac12L_f+\tfrac12L_T+\lambda_qL_q
       +\lambda_{\Delta}L_{\Delta T}+\lambda_{\mathrm{op}}L_{\mathrm{op}}.
\]

Retain the current native case/module/quadrature reductions. Keep the existing q-proxy coefficient convention and state exactly where it appears after group balancing. Reuse the existing TRAIN response families and qualified discrete operator; recompute any architecture-dependent calibration once on the fixed TRAIN calibration panel. Do not import old optimizer moments or pretend the old thermal-only gradient calibration is automatically appropriate for a new joint backbone.

The operator term may continue to use stored TRAIN velocity as a supervised coefficient, as it already does. That is not an inference input. Do not replace it with poorly trained predicted velocity early and thereby change the scientific question to an unqualified coupled residual. The learned flow head receives its own true flow targets in the ordinary field objective.

Before sealing the scientific recipe, inspect a bounded TRAIN panel's flow/thermal/shared-gradient scales. Correct actual normalization/denominator errors. Permit at most one fixed TRAIN-derived group-weight adjustment if the initial groups are grossly imbalanced; apply it to all compared architectures. Do not launch a loss-balancing-method portfolio, continually retune weights from DEV, add peak-ranking supervision on0291 or accumulate many auxiliary losses.

### 6.4 Wind loss and objective consistency

Retain the proven TRAIN-only component-balanced five-role velocity objective and background-residual conversion. Use the same scale-fitting method on selected TRAIN and later fullTRAIN; values may differ because populations differ. Main controls share these fitted constants and query IDs. Do not repeat the W0/W1/W2 loss screen in this phase.

Do not add a physical PDE loss merely to make the graph look physical. Existing labels are stored OpenFOAM velocities and do not automatically provide all derivatives, boundary data or closure terms needed for a new equation constraint.

### 6.5 Checkpoint selection

Save monitoring/latest/declared milestones every100 epochs and one best-field alias. Thermal monitoring must report both flow and thermal family scores plus fluid-T as its own headline. Use a fixed equal-family TRAIN-scaled combined selector; do not let a large flow improvement conceal a material regression in fluid T. Wind uses its native component-balanced selector and displays all15 physical means/tails.

Report the literal common endpoint and the declared selected checkpoint separately. All graph exports, responses, cost and derivative tests must identify which of those weights they use. Post-hoc thresholds, best individual cases and alternate temperature-only selectors must not silently replace the primary state.

## 7. Fair controls that answer the actual architecture question

Build a compact three-mode comparison under the same joint output/training contract:

| Mode | Representation | Purpose |
|---|---|---|
| **J-direct** | Trainable source/environment context and source-preserving receiver read, but no receiver-addressable collective regional path; jointly learned flow/T in Thermal | Separates real joint training and adequate capacity from benefits of the new organization. It is not D-sep plus a separately optimized thermal network. |
| **J-geometry** | Same regional edge budget, typed collective value/update blocks, native inputs and receiver context as J-H; memberships/access use fixed smooth geometry-based weights | Tests whether spatial memory alone suffices and whether learning its memberships matters. All non-incidence parameters train. |
| **J-H** | Learned node-to-edge membership and receiver-to-edge access; same output law, regional budget and native inputs as J-geometry | Tests learned collective organization. |

Use the same width, E inventory, query budgets, objective and optimizer history. Initialize compatible common tensors identically. Report active parameter counts and real runtime differences; the small incidence-network parameter difference between geometry and learned modes is disclosed, not hidden by dormant tensors. J-direct has a different connectivity path, so it is a useful architecture control rather than an exact same-function conversion.

The geometry control should use stable Gaussian or comparable smooth geometry weights with input-defined length scales. Do not fit its anchor locations/radii from validation errors. Source and environment types remain separately normalized as in J-H. A simple geometry control must not secretly lack an environmental pathway that only J-H receives.

After training, same-weight interventions strengthen the interpretation: replace learned membership with geometry weights, replace receiver-specific access by a scene-average access, remove one collective update, and perform one matched-degree/mass membership shuffle. These are interventions on the current trained function, not independently trained causal estimates of architecture effects. Do not merely permute both axes consistently; that is a relabeling invariance test and should leave outputs unchanged.

## 8. Bounded autonomous development and self-revision

### 8.1 Budget and scope

Unless the user sets a different envelope when handing off this document, use a proposed default ceiling of **24 aggregate GPU-associated hours and16 elapsed hours**, including startup, failed attempts, profiling, training, evaluation and closeout. Reserve the final2 elapsed hours for evaluation/reporting/tests/push. At most two currently authorized, available GPUs may be used; inspect actual availability and do not disturb ongoing jobs. This is a ceiling, not a request to spend all of it.

No new physical solves: the ledger remains326/326. No inverse search, no formal full-TRAIN run, no new source dataset and no WindTEST target access. The user has requested multiple future development rounds, not permission to run an unbounded architecture search in this one.

### 8.2 Stage A — implement and prove real joint learning

Implement the minimal family and adapters, then perform disposable low/high-M native forward/backward checks in both datasets. Verify one self-contained checkpoint can load and predict with access to old learned checkpoint paths denied. Capture actual optimizer membership, per-head/shared gradients and parameter changes. Confirm cold heat null, exact affine application, native transforms and target poisoning tests.

Use at most a short few-case TRAIN-only fit-capability check to resolve obvious implementation defects. It must exercise flow as well as temperature and both node types. Its weights do not initialize scientific candidates. A falling temperature-only loss is not a joint-fit preflight.

Do not spend most of this goal re-auditing hundreds of unchanged historical tensors. Reuse the existing inventory and test machinery; check original exact membership once at start and once at end, with targeted checks around any touched compatibility path.

### 8.3 Stage B — matched100/500/1000 development screen

Train J-direct, J-geometry and J-H on each segmented dataset. Review at100 before continuing to500; healthy underfit runs are expected to continue. The100 review checks correct learning, native maps, gradients, throughput and serious failure—not parity with a mature5000-epoch classic.

At500, compare exact-age physical fields, complete costs and graph use. If a candidate is healthy but underfit or a mechanism question remains genuinely unresolved, continue it to1000 under the same declared2500 schedule. Do not label a slow-to-learn architecture “cannot improve” merely because of an early gap. Do not keep a broken target/frame/loss path running because training is cheap.

Detailed native fields remain a fixed small input-selected panel; population statistics use all22/all24 development cases. Use existing formal reports as historical context, not as matched segmented-training labels.

### 8.4 Stage C — up to two substantive self-revisions

If the first J-H has a clear deficiency, Codex may diagnose and implement a revision before final delivery. Each revision must have a short pre-fit record: observation, proposed cause, one core change, expected informative comparison, initialization, objective/stream invariants, and stopping boundary. Review its actual result; do not defer every identified fix to the next user turn.

Allowed single-change examples include: regional anchor resolution/locality; where collective states are returned to source nodes; receiver access to edge content; one extra collective block; or removal of an observed global-summary bottleneck. Changing an output representation to eliminate a demonstrated structural constraint is allowed. The revised family must remain compact and source-preserving.

Do not combine width, query count, new losses, gate law, environment resolution and depth in an untraceable revision. Do not introduce a recursive Tree, fixed-small-global-K compression, black-box residual correction network, old pretrained flow fallback or a per-case repair. After two revisions, choose the best supported version or report the remaining mechanism limitation. No expanding portfolio follows.

For a shape-compatible revision, use a fresh child of the last common boundary and continue the unchanged control for the same additional updates with the same schedule. For a shape-changing revision, use fresh matched initialization or an explicitly matched transfer; do not compare a fresh500 child with a1000-epoch control as an equal-age architecture test. Record inherited and new ages separately. For the final claim, ensure the selected version has an appropriate matched-history comparator.

### 8.5 Stage D — meaningful maturation

Mature the preferred J-H version and the strongest relevant matched control to **2500 development epochs** when healthy and within budget. Retain the other screen endpoints without extending all losing arms. A geometry control that closely competes with J-H should remain the final comparator; do not discard it in favor of an easier direct baseline.

At least one final native joint Thermal predictor and one final native Wind predictor should have a substantial completed fit. Do not end after many brief branches while none has reached a useful learning age. Conversely, no unfinished arm is called mature. If resource limits bind, use an acknowledged clean stop and report exact completed endpoints.

A second seed is optional only after the main paired learning, native evaluation and closeout reserve are secured. It should reproduce a selected comparison, not start another model family. Do not imply statistical certainty from one seed.

### 8.6 Stage E — graph sparsifiability and actual work

Start from the trained smooth J-H. Evaluate predeclared receiver-to-edge retention levels100%,99%,95%,90% cumulative access mass, keeping the direct physical-source path and near detail unchanged. Choose resulting K per receiver from the learned access distribution; do not train to a prescribed K histogram. Type/source identities remain intact.

This is a **model approximation study**, not a bound on physical error. Renormalized truncation generally changes the function. Measure both deviation from the full J-H and error against the stored physical reference. Compare with geometry-selected access at identical retained edge counts and comparable work. Do not infer a physical omission guarantee from attention mass.

Implement one actual subset operation only after the measured frontier identifies a worthwhile point. First sparse target is receiver-to-collective-edge reads or scene-level node-to-edge reads, not mandatory removal of physical heaters/turbines. Count each primitive separately. A reduction in one edge family is not a reduction in all source/environment ancestry or in total execution.

The smooth full model remains the default until approximations qualify. Dense and subset implementations of the same truncated function need their own parity tests; the full and truncated functions are not parity variants. Derivatives with a fixed truncated support are conditional, and geometry/query support switches remain explicit. No heavy straight-through/soft-shadow training campaign is required in this goal.

## 9. What should be measured

### 9.1 Predictor quality

Thermal: every native u/v/p/omega field, fluid/surface/material T, native q proxy and sampled per-module maxima. Report equal-case means and p90 or p95 tails, module-count strata, valid near/far fluid support and signed residual maps. Fluid temperature is a primary metric, not merely one term hidden inside an all-channel aggregate. Preserve the native one-field consistency of temperature across extraction roles.

Wind: all15 component-role means, p95 and worst-row RMSE in m/s; vector-role errors; low/high-M native Ux/Uy/Uz planes and fixed wake cuts; separate direction and module-count summaries. The near/downstream transverse components remain important even if vector RMSE improves. Do not pool correlated directions or overlapping role strata as independent samples.

All main new-model comparisons use the same native receiver coordinates, target arrays, scales and masks. Joint training must not silently change output ordering, pressure gauge, empirical background, physical query frame or the temperature stencil definition.

### 9.2 Organizer quality

Report actual incoming M/E memberships, edge-state spatial anchors, outgoing node updates, receiver access and any source/global bypass. Explain what an edge contains and which output heads use it. Track concentration/effective participation and exact support separately. A softmax with many tiny positive weights does not establish physical sparsity or an adaptively learned number of groups.

Use three input-selected receivers in qualitatively different regions for each dataset, such as an upstream/background region, a near-source region and an overlap/downstream region. Show the complete relevant source-to-edge-to-receiver chain, not just its last link. Region labels describe location, not causal upstream/downstream exclusivity.

For meaningful utility, use (a) J-H versus independently trained J-geometry/J-direct, (b) same-weight membership and receiver-access replacement, and (c) one group-content or incidence removal/restoration. Removal must propagate through the affected collective updates rather than changing a decorative export. Separate frozen-plan interventions from fully rebuilt model responses.

Where membership is geometry-dependent, compare the graph under small configuration changes, retaining physical IDs. A changed graph alone is not proof of a correct physical response. In fixed-geometry Thermal heating, unchanged graph plus changed affine contributions is the correct dependency, not failed adaptivity.

### 9.3 Physical responses and future inverse scope

Reuse existing heat and geometry response records only. Keep TRAIN response-fitting families distinct from response-development and the previously exposed counted audit. Measure heat responses and model derivatives for the selected new joint checkpoint. Replay a bounded existing geometry panel on the new checkpoint, rather than displaying3903 geometry results as though they qualify the new model.

For Thermal0291, report each physical module's baseline/plus peak, predicted versus reference hot-module identity, per-module increments and the resulting global maximum. Do not make its failed ranking a training label or build a special correction for it. The aim is to see whether improved organization improves that known weakness as a consequence.

For Wind, compare existing stored operating directions using each row's native coordinate/vector transformation and existing cells. These are response-to-operating-context examples, not new finite turbine-displacement/yaw/thrust references. In their absence, center JVP/VJP are model derivatives only.

No inverse campaign is needed. A source-resolved, differentiable and correctly invalidated forward interface is the required reusable artifact. Reliable physical design is a later qualification, not something proved by a Jacobian dot-product test.

### 9.4 Efficiency and memory

Measure cold provider/catalogue setup, warm training epochs including all objectives/backward/optimizer, complete native inference, prepared repeated reads and one useful input VJP. Separate CPU catalogues, resident model weights, CUDA allocated/reserved peaks and process memory. Do not compare the new joint training cost only against3904's thermal-only cost while omitting its historical flow fit.

Initial engineering targets are complete inference within roughly2x the latest compact incumbent per dataset on the same hardware/output scope, while remaining substantially faster than the appropriate dense classic. These are proposed resource targets, not derived physical thresholds: about82ms Thermal versus3904's41ms and25ms Wind versus2202's12.45ms are convenient reference values on the old GPU1 panel, not portable timing guarantees. A somewhat slower model can remain worthwhile if the native fidelity gain is substantial; state the tradeoff instead of hiding it or automatically rejecting it.

Use optimized batched projection/attention for regional reads. Context formation should scale with node/edge catalogues, while query decoding scales with QG and QM. Avoid Q² attention and any Q×E×M hidden activation. Do not launch custom-kernel or compiler work until the ordinary implementation's profiler identifies a meaningful bottleneck. Warm training throughput, not VRAM occupancy, chooses packing.

## 10. Success criteria and decisions

The following thresholds are practical prospective development targets, not guarantees or claims that a threshold proves general physical correctness. Seal them before scored results; do not move them to turn a miss into a pass.

### 10.1 Mandatory structural acceptance

The final candidate must satisfy all of the following: no external frozen learned field model; shared trainable representation used by every Thermal field head; actual flow and temperature gradients/updates; one self-contained model/checkpoint; common executed core in Wind; correct actual-input heat null and affine T capability; native source/receiver/unit identity; target-free inference; and tested stale-state rejection. Failure on these points means the central requested change is incomplete even when an aggregate field score improves.

### 10.2 Direction worth continuing

A useful positive result should improve a primary difficult-field metric by approximately5% or more over a strong matched control on at least one dataset, or deliver a similarly meaningful native accuracy–cost frontier improvement, while not materially degrading the other dataset's important roles. Use roughly5% mean and10% tail deterioration as initial caution bands, not permission to conceal a localized serious failure. Inspect component and per-case results before selecting a recipe.

Learned organization should show measurable receiver-specific use and an advantage beyond merely supplying a regional memory. A few tiny differences below numerical variability or one cherry-picked edge effect do not establish that. A geometric regional control that wins is scientifically useful: retain its spatial representation, but describe learned incidence as unproven and use the revision allowance to investigate it.

Do not require a2500-epoch segmented candidate to beat a5000-epoch fullTRAIN classic. Historical formal references are performance context. The question this goal answers is whether the new representation improves matched development learning while retaining an attractive practical cost and the desired joint architecture.

### 10.3 Sparsifiability target

A desirable direction-level result is at least one measured operating point with meaningful receiver-edge work reduction (for example20%) and no more than roughly5% degradation in the primary native error and10% in protected tails, compared with the full same-weight J-H. Report all roles and model-distortion error. This is an engineering target, not a requirement to force masks or reject an otherwise useful smooth model.

No sparse wall-clock claim is made without measured complete costs. No inverse-readiness claim is made from this graph approximation study. It is acceptable to deliver a good smooth regional hypergraph with a clearly measured, currently limited sparsification frontier.

### 10.4 Final dispositions

Use separate decisions: **joint-model contract met or missed; predictor direction promising or not; learned collective organization useful or unproved; sparsifiability measured or unmeasured; inverse interface numerically usable or not; physical inverse qualification still limited**. Avoid a single generic “mixed” verdict that leaves the user unable to choose the next investment.

## 11. Tests that matter for this change

Use the existing test framework and add focused tests for the new architecture. Prioritize the checks below; do not spend the round creating a second audit platform.

- **Joint optimization:** optimizer coverage of all active heads/shared blocks; nonzero expected gradients from flow and thermal groups into shared parameters; actual updates and moment ages; head-specific gradients behave according to their declared dependencies; no accidental `no_grad` or frozen borrowed model.
- **Self-contained loading:** deny access to3901/3902/old Stage-A files and still load/predict the new checkpoint from its own configuration/tensors/normalization. Merely copying old frozen models inside a checkpoint fails the architectural review.
- **Physical inputs:** poison target arrays and alter excluded observations without changing predictions; change real geometry/context and verify rebuilt state; heat-only flow null and affine superposition/increments in native units; Wind rejects unsupported exact affine increments.
- **Representation:** module/environment/query permutations; source IDs and control joins; inactive/padded rows; batch packing invariance of losses/gradients; identical-coordinate/attribute environment atom splitting with preserved measure; empty environment type and low/high module counts.
- **Hypergraph function:** actual memberships, collective states and receiver access participate in predictions; group-content removal/restoration propagates correctly; a consistent edge-label permutation is invariant; an actual assignment intervention changes the function when the edge is consequential.
- **Numerics:** same point in different query chunks/batches; native field/stencil/q-proxy parity; finite actual-input derivatives; at least two FD scales; distinguish Eulerian query and module-attached derivatives and support changes. Do not relax old failed proxy tolerances silently.
- **Lifecycle:** exact epoch-boundary resume including optimizer/RNG/sampling; one ordinary update per macro batch with correct tail denominators; stopped or failed runs not relabeled complete; compatibility replay for touched old paths.

Classify the inherited14 persistent failures against the reviewed baseline, and run tests affected by this change. Existing unrelated failures need not block a new valid experiment, but new failures in the active joint path do. Preserve failure receipts and exact exit status.

## 12. Intuitive reporting: six figure families, not dozens of competing dashboards

### Figure1 — Did one joint model actually learn all fields?

Show the actual executed architecture, with trainable shared blocks and output heads, plus separate physical flow/T/Wind learning curves. A small optimizer-state table establishes the update counts of shared, flow and thermal parameters. Do not draw a frozen3901 box inside the new candidate and call it joint.

### Figure2 — Are the physical fields better?

Use fixed low/high-M native scenes. Thermal shows T and a relevant flow structure such as omega; Wind shows Ux/Uy/Uz on the existing native planes. Compare reference, strongest matched control and selected J-H, with one common residual scale per quantity. Add a small all-DEV means/tails table. Do not use a favorable single scene as the headline population result.

### Figure3 — What did a collective hyperedge gather?

Show physical modules and environment points, one input-selected hyperedge's actual incoming donor mass, its physical anchor/support, and the receivers that read it. Display a second relevant edge if the first is nearly singleton or unused, retaining the declared selection rule. Include a source-order/coordinate join so the user can identify each object.

### Figure4 — Does the organization affect a useful prediction?

Show the three receiver regions, learned versus geometric/scene-average access, an executed collective-content intervention, and native prediction-error changes. Give all relevant bypasses. This is the key figure for “not decorative”; avoid plots containing only probabilities that are all1.

### Figure5 — Does it respond to physical changes?

Use one existing Thermal heating/geometry reference and one stored Wind operating-direction pair, with explicit limits. Include a small per-module0291 peak table rather than another dense inverse dashboard. Keep model derivatives and independent reference changes visually distinct.

### Figure6 — What accuracy did the work buy?

Plot native error against complete time for full/simpler/organized/pruned modes and show a small complete-training/peak-memory table. Count node-edge, receiver-edge and physical source reads separately. Dense vectorization can win despite more logical work; report it directly.

Keep important numerical tables and unwrapped prose in the durable Markdown report, with a short six-family index. Save generated plots/arrays/renderers locally in ignored paths. Prefer PDF masters and only small raster companions required for direct Markdown viewing. Show actual fields and actual trained graph states, not an image-generated stand-in for scientific evidence.

## 13. Protection, resource closeout and delivery

Preserve all current exact647 logical protected bindings, including the archive/symlink disposition of3903, formal3901/3902/3903/3904, Wind2201/2202, classic models, measured references and development histories. Start/end inventory checks must compare identical membership, not whichever paths happen to survive a later scan. Do not overwrite the goal-start receipt; write separate immutable snapshots. The older inventory-evidence limitation remains disclosed and is not retroactively repaired by a new audit.

No scientific deletion, directory relocation, automatic archival or modification of unrelated running jobs is authorized. Use new candidate run IDs and directories. Generated arrays, figures, profiling traces, checkpoints and one-time scripts remain ignored. Reuse reusable data catalogues only when their geometry/partition bindings match; cached target/reference arrays must not enter inference.

Deliver durable implementation, focused tests, one compact architecture/training guide and one final development report. Include a concise machine-readable experiment table with dataset, model revision, initialization/parent, effective and micro batches, Q/E/width/edge resolution, completed and selected age, actual updates, native metrics, resource use and decision. Keep only one preferred next recipe per dataset plus the necessary matched control; the report can retain discarded hypotheses without recommending all of them.

Provide a tested manual-only fullTRAIN5000 recipe for the new joint family. It must fit fullTRAIN transforms, initialize all trainable weights freshly, include all Thermal physical heads, require no external frozen flow/model file, use the same general training engine, and retain WindTEST lock. Preparation/dry-run tests must not launch fullTRAIN optimization. Mark the recipe experimental if the development evidence does not yet justify a formal recommendation.

Follow root `AGENTS.md`: each prose paragraph is one source line; audit the entire outgoing commit range including files added and deleted in earlier unpublished commits; run the configured pre-push artifact gate; commit durable changes on the current non-default branch; inspect remote changes before push; verify remote/local tips afterward. Preserve unrelated user edits. The final message must say what actually trained, which architecture is preferred, what joint/organizer evidence passed, remaining physical weaknesses, resource usage and the commit—not simply “Goal complete.”

## 14. Relation to prior methods and the limits of the proposal

The proposal borrows a structural idea, not a promised result, from learned node-to-hyperedge-to-node set aggregation (AllSet), learnable physical-state grouping (Transolver), and geometry-aware latent representations with physical-coordinate queries (GINO). None of those papers establishes that this exact HONF design will improve these datasets, identify causal physics or solve inverse design. The experiment must determine that.

The important project-specific choices are joint multi-field supervision, registered rather than arbitrary global group identities, receiver-accessible regional context, preserved physical source/control columns, one temperature field across native roles, and removal of the external frozen flow dependency. Do not simply install a named architecture and substitute its published performance for our measurements.

Primary methodological sources for orientation:

```text
Chien et al., You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks, ICLR2022.
https://arxiv.org/abs/2106.13264

Wu et al., Transolver: A Fast Transformer Solver for PDEs on General Geometries, ICML2024.
https://proceedings.mlr.press/v235/wu24r.html

Li et al., Geometry-Informed Neural Operator for Large-Scale 3D PDEs, NeurIPS2023.
https://papers.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html
```

The attached current reports are the source of numerical baselines in Section1. The code paths in Section2 are the source of implementation findings. Sections3–12 contain proposed mathematics, controls, practical targets and bounded work instructions. Keep that distinction in all subsequent reports.

## 15. The concluding question this goal must answer

**After shared joint field training and receiver-accessible collective regional organization replace frozen flow composition and detail-only gates, do we obtain a demonstrably better, still efficient forward representation on Thermal and Wind—and can its actual learned organization be inspected, perturbed and sparsified without destroying useful field structure?**

A qualified positive or a precisely localized negative result is useful. Another collection of small gains with no clear answer to joint learning, regional information access and organizing value is not the intended delivery.
