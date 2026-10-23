# HONF next round — recover the native predictor and learn an adaptive interaction graph

**Working branch:** `agent/honf-core-next`  
**Review basis:** report and inspected source at `70d0c0a`  
**Mode:** Codex Goal mode; bounded research execution, not a production release  
**Central goal:** remove redundant interaction functions while preserving the physical information needed for modular inverse design.

## 0. Decisions that supersede the preceding plan

This round has two parallel scientific workstreams, not another serial chain in which every activity waits for a uniformly accurate response surrogate.

**Workstream R: native response recovery.** Improve decision-relevant finite responses without first deleting the strongest model's learned head or source controls. Start from the complete, checkpoint-native ThermalChannel Run 1804 selected e4738. Keep mature Run 1502 selected e4794 as the principal HONF comparator and a bounded alternative when a clearly measured thermal-role advantage warrants it. Do not resume the failed Run1508-to-three-term u300 refit.

**Workstream G: native adaptive organization.** Fit and test the existing adaptive-cover idea around a working native model. Begin on WindFarm Run 2103 selected e2475, retaining its trained field head, normalization, local/coarse contributions, and fine kernels. Full-access mode must reproduce the intact native predictor. Learn case-dependent interaction covers using training-only model interventions and existing reference fields. Missing ThermalChannel mixed-response resolution does not prevent this work.

**Integration I: reference-corrected inverse design.** Use one absolute forward model for both states of a finite difference. Correct local decision quantities with an independently measured accepted baseline, refresh the model's continuous state at every trial design, and accept simulated steps using actual reference outcomes. The earlier factor-only response oracle remains a historical control; it is not a required interface for the new native model.

There is no requirement to beat Dense on every forward metric. There is a requirement to demonstrate that an adaptive graph actually mediates useful computation, varies for substantive input-dependent reasons, and preserves declared physical quantities within the scope of the available evidence.

**The required new empirical result is a fitted, input-only native organizer, or a bounded native oracle result explaining concretely why such an organizer cannot yet be learned. Another tested but untrained graph prototype is not the intended endpoint.**

The following old requirements are withdrawn for this research round:

- Mandatory conversion to `three_term_full_access_honf` before response training or cover research.
- A universal 10% response-relative error requirement on every channel before any organizer can be trained.
- A universal 5% relative error-increase rule without a physical error floor, especially for near-zero channels.
- Blocking teacher-preservation experiments because a reference mixed response is unresolved.
- A requirement that every prediction and inverse step use the old unary/pair response-factor interface.
- A standalone objective that minimizes K, or a prescribed histogram of K.

Existing security, artifact, checkpoint-loading, and repository rules remain in force. This document changes scientific choices and experiment scope, not security controls.

## 1. What the preceding experiment did and did not test

### 1.1 Report-derived findings

The report selects native Run1804 e4738 as `B_inc`, but the executed ThermalChannel fit starts from Run1508 e496 after a non-equivalent change of backend. The transfer discards 26 source-only tensors. Its native adapter explicitly requires `three_term_full_access_honf`. The matched value/response comparison is useful within that changed model; it does not test response fine-tuning of the intact best Dense or mature sparse-incidence model. [E1, C1, C2]

The fit uses three training neighborhoods: two M3 cases and one M7 case at Re50/70. The development neighborhoods include M3/M5/M7/M10, all at Re90. Geometry exposure to historical teachers, withheld response neighborhoods, module-count coverage, and operating-context shift are distinct. A development miss here is not an isolated test of interpolation at matched context. [E1, “Response panel” and “Stage A”]

The corrected sampler does cover the full training near-interface union. Therefore the u300 failure cannot still be attributed to the original sampler omission. Finite thermal responses improve on train, while development responses and several absolute quantities worsen. The current code forms its value loss from the stencil states; merely constructing a historical dataset object to obtain an input template does not provide broad historical value replay. [E1, C3]

The WindFarm refit transfers 74 tensors, discards 73 Dense common-head tensors, and initializes 12 new head tensors. Selected e295 follows 1,180 updates. Its loss on every matched validation row establishes that this refit is not an adequate replacement. It does not prove that the intact dense network lacks capacity or that a hypergraph cannot approximate it. [E1, “WindFarm native transfer”]

All training/development pressure states in the paired refit are feasible under the study's `1.05 * baseline` limit. Zero false-feasible counts therefore do not estimate constraint reliability. Baseline pressure bias and finite pressure-change error are quite different: some finite errors become small relative to the study margin while absolute bias stays large. [E1, “Pressure review”]

The local generator is analytic-wake/shared-grid thermal physics, not Navier–Stokes CFD. Its mesh-sensitive geometry changes and proxy flux are insufficient grounds for a universal derivative or conservation target. One measured M3 thermal interaction does not establish peak-objective utility when the hottest spectator scarcely changes. [E1, E2]

### 1.2 Inferences to test, not established causes

The following are plausible contributors to the failed fit: a damaged initialization, loss of previously learned source modulation, training on very few neighborhoods without broad value replay, joint context/count shift, competing gradients, and a frozen local-module response limit. None is declared the sole cause.

Do not re-label these hypotheses as solved after a config inspection. Test them with actual native outputs and optimizer updates.

### 1.3 Three independent error budgets

For an incumbent B, a graph-mediated model G, a reference y, and one fixed role norm,

\[
\|G-y\| \leq \|G-B\| + \|B-y\|.
\]

This separates **additional organization distortion** from **incumbent physical error**. Small teacher distortion is not small physical error, but it can be studied even when the incumbent is imperfect.

For a decision quantity g at design d and an accepted baseline d0,

\[
[g(d)-g(d_0)]-[\hat g(d)-\hat g(d_0)]
\]

is yet another error. Good absolute fields, good increments, and good feasibility decisions must not be conflated.

## 2. Baselines and retained assets

| Role | Starting point | What stays intact |
|---|---|---|
| Thermal native response incumbent | Run1804 selected e4738 | Entire native architecture, common head, normalization, physical wrapper, and checkpoint-local surrogate |
| Thermal HONF comparison / one alternative | Run1502 selected e4794 | Original sparse incidence, source moments, K/D semantics and port loop |
| WindFarm native reference and graph host | Run2103 selected e2475 | Original native 3-D adapter, Dense common head, all fine and coarse parameters |
| Negative/control implementations | Failed three-term refits, Run1508, earlier merger/factor studies | Preserved for replay, not silently relabeled or overwritten |
| Reusable graph machinery | `adaptive_interaction_cover.py`, packed readers, oracle proposal machinery | Physical receiver anchors, typed source memberships, live priors, one fine evaluation per unique pair |
| Reusable evidence machinery | Typed stencils, masks, functional evaluators, numerical-resolution records | Reference/teacher/synthetic distinctions and family grouping |

Load and re-evaluate actual tensors with checkpoint-native units. Do not select a different epoch using final-review outcomes. Use the allocator's next free run IDs; do not guess a numeric ID or reuse an existing identity.

An exact all-access execution modification needs numerical output/gradient agreement. A trained architectural approximation need not be identical to the incumbent, but must be clearly identified as such. Compatible tensor shapes alone are not a function-preserving conversion.

## 3. Immediate source changes and audit targets

### 3.1 Make native response fitting genuinely native

Inspect and extend:

- `Case_ThermalChannel/src/channelthermal/response_control/native.py`
- `response_control/runner.py`, `training.py`, `losses.py`, `sampling.py`
- `response_control/algebra.py`, `thermal.py`, `evaluation.py`
- `interface_field_coupling.py`, `local_coupling.py`, and checkpoint loaders.

`DifferentiableThermalOperator.__init__` currently rejects every architecture except the refit target. Add a tested native dispatch for the declared incumbent architectures. Retain shape, units, predicted-port, target-free input, and module-identity checks. Do not change unrelated historical behavior.

`runner.run_paired_fit` currently calls `_make_refit_model` and `_materialize_and_warm_start`. Add an explicit `initialization_mode=native_checkpoint` that loads the entire model and never invokes that conversion. Keep the old conversion mode for historical experiments.

The runner currently obtains optimizer learning rate from the source checkpoint training config. Make the fine-tuning learning rate and trainable scope explicit in the new experiment config, save their resolved values, and verify them from the actual optimizer. Do not accidentally fine-tune a mature model at its original from-scratch rate.

A native source-only wrapper must not require one particular local-checkpoint path: attach the model embedded/referenced by the chosen trusted checkpoint and preserve its own documented semantics.

### 3.2 Decouple cover injection from a new field head

The current `adaptive_interaction_cover_honf` selection also selects the three-term common head. Do not use that architectural switch as the only way to test a cover.

Introduce a small optional interaction-policy interface on the intact native model. Default `None` takes the unchanged parent code path. All-access policy takes the same mathematical path; do not initialize a replacement common head. Reuse the present cover data and packed kernels rather than creating a second execution stack.

The policy must be applicable first to QM/QE and then to typed MM/ME/EM preparation. Preserve original normalizations and biases. See Section 9 for the treatment of the native coarse/local paths.

### 3.3 Remove research deadlocks, not safeguards

`adaptive_cover_oracle._adequate` and `search_training_cover` currently require resolved reference roles and reject an inadequate full-access baseline before any proposal is evaluated. Preserve that behavior for a reference-sufficiency experiment. Add a separate, explicitly named **teacher-preservation** mode with real teacher-distortion limits. A teacher metric being finite is not the same as being below a tolerance.

Do not pass fabricated zero reference errors or `resolved=True` to bypass the old function. The two modes have distinct evidence meanings and reports.

The current acceptance reason `no_measured_cost_gain` is based on an estimated coefficient model. Rename the new-mode reason to `no_estimated_cost_gain`; reserve “measured speedup” for an actual paired execution.

Do not introduce another promotion-state machine. The evidence mode and ordinary experiment config are enough.

### 3.4 Keep diagnostics out of the fast path

`AdaptiveCoverPairwiseField.prepare` currently builds CPU trees even in full-access mode. Ordinary incumbent replay must not pay that cost. Construct trees only for cover use or explicit diagnostics.

`compile_cover_pairs` constructs query-by-node-by-source Boolean products for raw-path accounting. Make detailed counting opt-in or calculate it through bounded contractions/tiles. Do not materialize Q×N×E only to count work in a large WindFarm run.

`CaseLocalReceiverTree.build` transfers coordinates to CPU and uses recursive Python operations. Initially cache one candidate tree per prepared case/trust-region anchor; report build time separately. Batch or vectorize later if profiling justifies it. Do not start with a new CUDA kernel.

Normalize receiver-role sampling measures deliberately. Environmental volumes and unit-weight module anchors cannot be mixed without stating the convention. Source quadrature and receiver importance are different measures.

## 4. Workstream R — identify the response bottleneck without damaging the model

### R0. Exact native replay and error decomposition

Start with no training. Replay native 1804 and 1502 on the existing four response neighborhoods and a broader value panel, using identical role definitions. Confirm:

- Physical field outputs and the checkpoint-native predicted-port loop.
- Native material-coordinate sampling and surface/outside-temperature distinctions.
- Maintained 8%-band pressure drop, not the obsolete fixed-64-point proxy.
- Exact-null controls separated from unresolved small signals.
- Model-versus-reference errors, not just AD versus the model's own finite differences.

Record errors by physical family, M, Re, perturbation coordinate, step size, and role. Use physical units and a shared reporting normalizer in cross-model tables; keep checkpoint-native normalized scores as secondary historical metrics.

On a small training panel, localize error with interventions through the complete wrapper:

1. Native predicted ports.
2. Measured port quantities only where the dataset supplies the correct quantity at the correct location and convention.
3. One-phase interventions in P0/P1/P2, recomputing downstream physical states.
4. A bounded fit of local port/refinement outputs with the global core frozen.

Do not infer h from an unstable division by a tiny temperature difference, equate surface T with outside T, or call optimized latent ports measured physics.

If correct measured local conditions still cannot reproduce internal responses, that identifies a local-model or coupling limit. A train-only optimized-port fit can provide a representational diagnostic, but its optimized ports are not physical labels. A failed local optimizer is not a proven lower bound.

### R1. Use supported physics to separate problems

At fixed geometry the current local generator has a nearly linear thermal response to heating and no heating-induced flow/pressure change under its documented analytic-flow mode. Use this as a **case-specific diagnostic/control**, not a universal property of multiphysics models.

Test the native model on fixed-total-heat redistributions at fixed geometry. This avoids moving-grid discontinuities and isolates thermal/port response learning. Then compare with finite position responses. Do not claim that zero mixed heating response means no thermal coupling: cross-receiver first responses can be nonzero in a linear system.

Keep position changes at resolved finite step scales. Do not shrink h repeatedly to manufacture a derivative when raster geometry makes the label unstable. No shape-Jacobian training is enabled without matched material/Eulerian conventions and a demonstrated numerical range.

### R2. Repair the training coverage

Use all suitable **training** neighborhoods already available, not just the three selected previously. Keep original family assignments. Calibration/final families do not become new training examples merely because their errors are known.

Add a small crossed-context panel on training-family geometries: include the operating conditions intended for interpolation and include M5/M10 when their responses are evaluated. The target is at least two training families per M stratum where local records permit. A missing stratum is reported, not filled with duplicated points.

Distinguish:

- In-distribution held perturbations on training families.
- Withheld response neighborhoods at matched operating conditions.
- Explicit operating-context transfer.
- New layouts/compositions.

The four legacy Re90 neighborhoods remain development evidence. Neither 8,192 query points nor eleven correlated stencil states increase the number of independent layout families.

Use at least one broad historical value minibatch for each response minibatch. Cycle the original train cohort rather than protecting only stencil baselines. Both matched arms receive the same historical examples, native queries, and absolute-value terms. Only the paired-response addition differs.

Protect near-interface, port, peak/material, and pressure-section support as separate roles, with quadrature/importance weights correct for their respective objectives. Count coverage against original complete coordinates. Do not reuse the incorrect earlier sampler audit.

### R3. Primary native refit

Default trainable scope for the first pilot:

- Existing field-head output layers and existing port/refinement output layers.
- Native source encoders and fine interaction kernels initially frozen.
- Native local surrogate initially frozen, but only as the first controlled scope, not an immutable constraint.

First try the existing output layers, without adding a second absolute-field network. Use explicit fine-tuning learning rate `1e-5` as a starting numerical choice, not a physical constant. A separately resolved rate can be chosen in a bounded pilot if measured updates are ineffective.

Because port updates can feed back into shared contexts, freezing field kernels alone does not guarantee unchanged flow. Evaluate the complete native pipeline and retain broad replay.

Use

\[
\mathcal L_R=\mathcal L_{\rm historic\ value}
+\mathcal L_{\rm stencil\ value}
+\lambda_\Delta\mathcal L_{\rm finite}
+\lambda_J\mathcal L_{\rm decision}
+\lambda_g\mathcal L_{\rm pressure}.
\]

The value-only control has the same initialization, trainable parameter scope, optimizer, sampling and update budget, but excludes the added finite/decision terms being tested. Do not compare a tiny response refit to an entirely different replay schedule.

Use per-channel training scales and physical absolute errors. For finite differences, a suitable normalized residual is

\[
\frac{\widehat{\Delta y}-\Delta y}
{\sqrt{s_{\Delta,r}^2+\sigma_{\rm numerical,r}^2}},
\]

where `s_delta` is training-derived and `sigma_numerical` is available for that family/role/step. Missing numerical uncertainty is marked unknown, not imputed from a different family. Unresolved finite values may be retained as low-weight observed-value data, but must not become positive/negative graph labels or derivative certificates.

Do not force proxy q-normal to 10% response error when the reference has a larger numerical/definition uncertainty. Surface temperature, internal temperature, outside temperature, effective h, and flux proxy remain separate targets.

### R4. Gradient conflict and local response remedies

Record task-gradient norms **and dot products** on representative replay and response batches, separated into head, port, backbone, and local-model blocks. Matching norms alone does not prevent an update from damaging the value task.

If the main failure is measured gradient interference, authorize one remedy: reduce response weight/rate, or apply a documented conflict projection such as PCGrad. Do not add several balancing methods at once. Report both task curves and full held-role errors; gradient surgery is not a physical correctness guarantee. [M3]

If a frozen local model is the measured bottleneck, authorize one limited local-output adaptation with the original local-data replay. Preserve positivity/units and material-coordinate semantics. Do not train a free field-value bypass to conceal a broken interface.

If the frozen backbone cannot fit even matched training neighborhoods, unfreeze one existing late interaction block with a smaller learning rate. Re-evaluate broad values before scaling up. This is an allowed bounded remedy, not another architecture conversion.

### R5. What counts as response progress

Report a Pareto table: absolute values, finite increments, pressure bias, margin-scaled pressure increment error, material/near-interface risk, and context transfer. Retain an unchanged native incumbent in every table.

Use resolved physical tolerances where available. As exploratory defaults, compare additional error against `max(0.05 * incumbent_role_RMSE, measured numerical discrepancy, 0.005 * training_role_scale)`; report all three contributions. These defaults guide interpretation and must be calibrated on training/calibration data before final use. They are not validated engineering limits.

Do not require every channel/state to satisfy an unstable percentage ratio before Workstream G proceeds. Do not declare decision readiness when a protected decision quantity remains wrong beyond its useful margin.

## 5. Workstream G — a native conditional interaction graph

### 5.1 Working model: incumbent-native adaptive cover HONF

Reuse the existing receiver-cover mathematics, but attach it to the intact native computation rather than a newly initialized three-term head. A typed directed hyperedge is

\[
e=(\mathcal R_e,\mathcal S_e^M,\mathcal S_e^E,t_e),
\]

where R is a physical receiver support, S are source sets, and t records the message/read role. The same physical module can participate in multiple edges. Receiver partitions are an index, not automatically CFD regions.

For each typed transport operation l, construct receiver weights and source memberships,

\[
a_{re}^{(l)}\ge0,\qquad \sum_e a_{re}^{(l)}=1,\qquad
b_{es}^{(l)}\in[0,1],
\]

and the unique source-access prior

\[
w_{rs}^{(l)}=\sum_e a_{re}^{(l)}b_{es}^{(l)}.
\]

All source memberships equal to one give `w=1`, hence the original transport. Distinct active receiver/source packets define case-dependent K. Tree capacity, K, per-query edge count, fine pair support and executed rows remain separate.

**No giant-edge count reward:** a root covering every source is full access, not successful sparsification. It is allowed when needed but pays for all its source work.

### 5.2 Preserve the actual fine operator

For additive messages,

\[
m_r^{(l)}=\sum_s \mu_s w_{rs}^{(l)}
\psi_l(z_r,z_s,x_r-x_s),
\]

followed by the incumbent's original update, bias, and normalization. Do not replace its denominator by the number of selected sources unless that is a separately tested architectural hypothesis.

For attention,

\[
C_r=\frac{\sum_s \mu_s w_{rs}e^{s_{rs}}V_s}
{\sum_s \mu_s w_{rs}e^{s_{rs}}}.
\]

Use one normalization over the unique supported source union. Overlapping hyperedges do not create separate physical source responses or separate attention softmaxes. Preserve the existing smooth empty-support treatment and account for its full-access cost.

A layer's nonlinear update after combining multiple source messages is a collective, potentially nonadditive interaction. This is not proof of an irreducible higher-order physical law. Keep fine source states and the original nonlinear updates; do not introduce pooled field values merely to make an edge look more physical.

The initial model does not require a new free hyperedge field decoder. An edge represents and controls an existing collective computation. New group-control parameters are considered only after the organized computation has demonstrated value, not as another uncontrolled branch.

### 5.3 Exact initialization, then approximation

At policy `None` or all-access, call the original model path and reproduce its outputs. No source tensor is discarded, no common-head parameter is reinitialized, and no second copy of the teacher is required in deployment.

Partial covers are intentional approximations. Test them against both the unchanged native teacher and available reference data. They are not claimed to be algebraically identical to the parent.

Start with frozen incumbent weights for cover discovery. Then fit the input-only organizer. Jointly fine-tune a limited native parameter subset only after comparing the frozen-weight graph against its full-access teacher. That isolates graph selection from capacity changes.

### 5.4 Define K correctly

Count distinct source-bearing receiver packets actually selected on a case's declared physical anchor universe. Report K per typed transport layer and, for ThermalChannel, per P0/P1/P2 phase. Do not add the same reused packet count across query chunks or call a phase count a single universal graph size.

Report changes in K within fixed M and operating-condition strata, not only across turbine counts. Include same-layout/different-condition comparisons where the coordinate and identity mapping is known.

The registered source count and coarse-latent count may remain fixed while K changes. They must remain disclosed. There is no claim that variable K removes every other finite-dimensional basis.

## 6. Train-only oracle discovery without requiring perfect reference responses

### G0. Choose evidence mode explicitly

Two modes share execution machinery but not their claims:

1. **Teacher preservation:** the candidate cover preserves a chosen native surrogate's field/context/quantity within a declared distortion tolerance. Unknown reference mixed responses do not prohibit this mode.
2. **Reference sufficiency:** the candidate additionally satisfies available reference-value/response tolerances and numerical-resolution conditions.

A teacher-preserving graph is a surrogate computational organization, not independent physical truth. A reference-sufficient graph can support a stronger claim only for the tested roles, designs and perturbation range.

Never label a teacher disagreement or an unexplored donor as a resolved physical noninteraction.

### G1. Native oracle panel

Begin with 12 WindFarm training layouts, including all three stored direction rows, chosen by input geometry to span turbine count, layout density and extent. Use eight appropriate ThermalChannel training cases as a smaller secondary panel. This panel selection is independent of validation errors.

Use receiver anchors from exogenous environmental support, module/turbine neighborhoods and decision receivers. Keep the established coordinate frames. Normalize the anchor-role measures so global volume does not erase important small receiver regions.

Use disjoint search-probe and verification-probe query sets for each case. Start oracle evaluation at bounded Q1024/Q2048 with protected receiver locations; replay selected covers on larger native/whole-grid sets before training the amortizer.

### G2. Candidate proposals and selection

Reuse the current input-anchored tree as one search index. Its axes/cuts are not physical labels. Start from full access, then propose:

- pruning an irrelevant source block from a reached receiver packet;
- splitting a packet and selecting different source sets for its children;
- merging packets when their distinct supports do not justify their extra cost.

Splitting two all-source children alone cannot save physical work. Source-mask changes and splits must therefore be evaluated together where appropriate.

Rank cheap proposals using teacher message/attention diagnostics or bounded gradient/ablation proxies. These quantities are **proposal heuristics only**. Acceptance requires an actual forward evaluation and distortion/reference comparison.

Use block proposals for the environment, not one full model replay per one of Q×E pairs. Batch proposals where memory permits. Initial search ceiling: 48 candidate observations and 16 accepted edits per case. Permit one second-pass index/relevance remedy on the training panel; do not enumerate the power set.

Cost includes active packets, source incidence, unique MM/ME/EM/QM/QE rows, native coarse/local transport, fallback work, and model preparation. Measured coefficient models rank proposals, not certify speedups. If the model predicts a saving but execution does not show one, report that mismatch and retain the measured faster implementation.

### G3. Error limits for discovery

Keep separate physical and teacher-relative limits. A useful teacher-distortion starting scale is the same training-derived output scale used by the native model, with extra attention to wake/near-interface residuals. Do not normalize only by full Ux energy, which can make a nearly uniform velocity field look exceptionally accurate.

On training/calibration cases, compare a small number of predeclared distortion settings and select one working recipe before validation. Do not tune thresholds on the final review to force variable K.

For a fixed role norm, report `error(cover, teacher)` and `error(cover, reference)` separately. On finite responses compare both designs through the same cover policy and declared trust-region convention. A fixed-at-baseline plan and a recomputed plan are different evaluations.

The full-access native teacher is always a valid starting point in teacher-preservation mode. Failure to meet a separate physical-response target must not cause an exception before any native cover experiment runs.

### G4. Amortized input-only organizer

Fit one shared node/source scoring network. Inputs may include current physical design/context, geometry-relative features, node extent and role, local source features, and current model-side phase states. Do not feed targets, oracle errors, layout IDs, design-objective labels, or trial reference outputs into cold-start inference.

Use training oracle decisions and the actual predictive loss to learn splitting and source relevance. Supervise classification on **pre-activation logits**, not only through a saturated endpoint gate. Include exploratory teacher-approved alternative covers to avoid teaching every undecided node the root's majority label.

Membership and split training can use continuous coefficients, but report the actual deterministic inference cut and its loss. Soft expected counts are not dynamic K. Unknown source decisions are masked in the classification loss, not labeled negative.

The organizer is fitted with the incumbent initially frozen. Its search target combines adequate response/value preservation with measured work; no isolated K penalty, minimum K, entropy target or required K variance is introduced.

All-access is available for genuinely difficult cases. Report its frequency and cost, rather than counting it as a successful adaptive simplification.

## 7. Make the graph mediate the computation it claims to organize

A decoder-only cover of already globally mixed tokens is not a complete physical dependency graph. Therefore this workstream has two explicitly reported levels.

**Level G-read:** cover QM/QE at the intact native reader. This is a useful executed-query organization result, but preparation remains dense and may contain indirect information from omitted physical sources.

**Level G-transport:** extend the same typed-policy mechanism to at least the dominant cross-module/environment preparation operations. Apply masks before their source aggregation, keep original fine messages and update functions, and expose the actual communication path through the graph. Warm start with all-access, then train/distill the masks rather than deleting the learned updater.

The intended result of the round includes a fitted G-transport prototype. If only G-read is affordable or adequate, report the limitation plainly; do not rename it a discovered physical causal graph.

### Native coarse and local paths

Audit all native paths at fixed weights using target errors and output changes—not context norms alone. Norms can be large while effects cancel, and vice versa.

1. Trace the donors read by the native common coarse/local paths.
2. Keep those paths at initialization; deleting them is the failed refit confound being avoided.
3. Include their actual communication and cost in the graph/evidence accounting.
4. Route their eligible source reads through the same policy where feasible, or declare them explicit residual/global paths whose information has not been sparsified.
5. If they alone reconstruct most of the layout effect after the organized fine path is removed, the fine graph is not yet the needed information bottleneck.

Do not claim a physical “only needed sources” result in the presence of an undisclosed dense bypass. A declared global background may depend on prescribed operating conditions. A learned layout-dependent global branch is not exogenous background.

If replacing a dominant native coarse transport is required, use a bounded **layer-output distillation** study while retaining its trained downstream field head. A temporary interpolation

\[
C=(1-\gamma)C_{\rm native}+\gamma C_{\rm graph}
\]

can preserve initialization while the replacement learns; the native and graph terms must be mutually replacing, not permanently added. The teacher branch must be absent from claimed graph-only inference (`gamma=1`). Do not undertake this optional replacement unless the bypass audit shows it is the actual obstacle, and count it as one of the allowed remedies.

## 8. Physical interpretation and many-body tests

A fitted packet must expose donor/source identities, receiver support, typed role, operating context, and its tested validity region. Avoid human physical labels such as “wake cluster” or “thermal causal edge” unless independently supported.

Required comparisons include:

- Source/receiver permutation and same physical sampling with split quadrature weights.
- Same-M layouts with different interaction geometry.
- Same layout under available operating conditions with correct frame/identity handling.
- Case-dependent organizer versus a training-population fixed cover, matched in realized complexity as closely as practical.
- Cover versus individual-pair sparsification with similar physical work, to test whether shared grouping adds value.
- Target-based removal/restoration of a packet and its effect on physically relevant receivers.

Distinguish three interventions: removing a computational pathway, modifying a physical source while recomputing the model, and independently solving the modified physical system. Their effects are not interchangeable.

For measured mixed responses,

\[
I_{ij}=y(d+\delta_i+\delta_j)-y(d+\delta_i)-y(d+\delta_j)+y(d),
\]

use only resolved role/step/family labels. Nonzero `I(J)` from a nonlinear max/log-sum-exp objective can occur without nonadditive physical fields; it is not a positive many-body field label.

Known nulls are valuable. Do not force pair interactions into the fixed-geometry linear-heating diagnostic. Conversely, a zero mixed pressure response does not remove unary pressure effects or a pressure constraint.

If native interaction covers predict the actual measured collective response better than work-matched fixed/pairwise controls, that supports an effective many-body organization. It does not identify microscopic interaction laws or a unique hypergraph.

## 9. Topology, gradients and inverse use

Do not demand a global differentiability theorem for a learned discrete graph before testing it. Do not pretend discreteness is invisible to a continuous optimizer either.

Use an outer/inner procedure:

- At an accepted design, infer a case-local cover.
- During one bounded inner optimization, freeze discrete donor sets and tree connectivity.
- Recompute all continuous source states, positions, physical geometry, priors and local coupling for every trial design. Freeze topology, not physics.
- Use smooth receiver overlap within the fixed tree and include its design dependence where defined.
- Recompute the organizer for proposed new anchors and measure any prediction discrepancy across the change.
- Accept a simulated design only after its independent reference quantities and feasibility are evaluated.

This gives piecewise model gradients on a stated local model. It is not global native topology-switch continuity. Measure both same-topology AD/FD consistency and actual recomputed-plan boundaries. Record the maximum role/functional jump and compare with the local useful decision scale.

Plan construction must not depend on query chunk boundaries. Full-access and fixed-cover predictions must be stable under chunking. Source-state caches are invalidated when physical inputs change; no reuse of stale phase states across designs.

## 10. Reference-corrected inverse design

### 10.1 Local baseline correction

For an independently measured accepted design d_t, define a proposal model for a scalar constraint/quantity g:

\[
\hat g_t^{\rm corr}(d)=g_{\rm ref}(d_t)
+\hat g_\theta(d)-\hat g_\theta(d_t).
\]

Its error is exactly the finite-increment error relative to that baseline. This can remove a large baseline offset, but it does not repair a wrong slope, unresolved reference change, or behavior outside the validated neighborhood.

Use the same declared physical functional at both designs. A pressure gauge shift should not change a pressure-drop functional. Do not subtract values from different coordinate/section conventions.

The measurement at the accepted baseline is legitimate inverse information, not a cold-start forward input. Keep cold-start and baseline-corrected metrics distinct and charge the baseline solve/measurement to the inverse budget.

### 10.2 Pressure and feasibility

Continue using regression of physical pressure drop and its finite change. Do not train a single-class BCE and call its apparent accuracy feasibility calibration.

Create a small training/calibration boundary-contrast panel using feasible geometries around a fixed, predeclared benchmark threshold. Choose candidates based on training information and bounded exploration, then evaluate independent reference pressure. Do not change the threshold after seeing results to manufacture two classes.

If the allowed trust region does not contain both classes, report that scope. A continuous constraint model can still be studied, but no false-feasible rate is estimated from an all-feasible panel. The previously exposed M3 infeasible candidates are now development evidence, not untouched tests.

Optional conservative margins come from declared calibration residuals and numerical discrepancies. Four families do not justify distribution-free deployment coverage claims. Reference verification remains part of the algorithm.

### 10.3 Two thermal decision tasks with different purposes

**Diagnostic task:** fixed-total-heat allocation at fixed geometry, with bounds on each module's heating. It tests thermal sensitivity/constraint plumbing without moving-grid ambiguity. Report all module temperatures, not only a smooth objective. Its pressure invariance in this particular generator is a null control, not proof of general pressure prediction.

**Primary modular-layout task:** bounded position changes with a nominated receiver or a predeclared global peak objective, geometric clearances, and pressure limits. Select tasks on training/development information before new review outcomes. Include situations where the graph's affected receivers actually influence the stated objective. Do not use an interaction on cold donors as automatic evidence for reducing an untouched spectator's peak.

Moving-grid sign reversals disqualify claims of mesh-stable improvement at that move scale. They do not require abandoning nominal-grid finite optimization; label it as such and refine the best promising step within the reserved budget.

### 10.4 Matched inverse comparison

Use the same absolute surrogate and local correction in:

1. Graph-informed joint module updates.
2. Size/work-matched random groups.
3. Ungrouped local updates.

Match candidate count, number of changed variables, trust radius, inner gradient/forward effort and reference-call budget. Graph discovery and training costs are recorded separately and included in an amortized-cost discussion.

Reference-evaluate selected candidates, including failed or infeasible ones in the cost. Update the accepted baseline only on observed feasible improvement. Adapt the trust radius from observed/predicted improvement using an ordinary trust-region rule. A tiny predicted denominator is not divided into a misleading ratio.

Preserve the earlier distinction between surrogate-selected outcome and best physically evaluated opportunity. Exhaustively evaluating a pool and then adopting its best member defines a different policy and must be applied equally to all arms.

A bounded negative-control trial remains permissible if explicitly labeled; it must not be described as a deployment-ready inverse result. Full-field perfection is not a prerequisite for a quantity-validated, reference-corrected local experiment.

## 11. WindFarm is the primary structural scale test this round

Use the complete Run2103 e2475 model and existing mmap/native-query workflow. Do not reuse thermal weights or impose thermal port semantics. Do not re-run the failed 73-tensor common-head removal.

Keep all three direction rows of a layout in the same split. The 200 layouts, not 600 rows, are the main independent grouping units. Use row-specific native axes and turbine coordinates. Direction remains categorical unless the coordinate/angle convention is independently established.

Graph evaluation must include:

- Physical-unit vector and component error on native volume, rotor-height slab and downstream regions.
- A wake/background-sensitive error rather than only a norm dominated by mean streamwise flow.
- Native queries not used by the oracle, small and large turbine counts, and same-M layout contrasts.
- Environment quadrature refinement tests; distinguish one-to-one source splitting from adding genuinely new quadrature information.
- Case count, K, source incidence, preparation, fine pair work, common/global-path work, and complete timing.

For a bounded decision experiment, define a velocity-based receiver quantity that can be computed by the same explicit native/interpolation quadrature from stored reference U and surrogate U. A turbine-neighborhood velocity statistic is not turbine power. An objective based on `wake_loss_pct` remains an empirical stored-label task until its generating formula is supplied.

Same-M, same-operating-condition stored-library selection is allowed as finite-choice evidence. Do not infer continuous layout optimization, yaw optimization, AEP, or new-layout CFD response from it.

Prepare a concise external-reference request: actual solver/generator availability, coordinate and wind-direction transformation, rotor/actuator convention, turbine controls, boundary conditions, and scalar objective definition. Do not make unavailable reference data a reason to stop native graph training on the already available CFD fields.

## 12. Controlled experiment matrix

Keep the number of trained arms small.

### Native response pair

- R-value: intact native Thermal incumbent, broad historical plus stencil values.
- R-response: same model/data/update schedule plus resolved finite/decision training.

Do not add more formal arms until a short pilot identifies a concrete need. Mature1502 is a native replay control and at most one alternative pilot, not an automatic second full study.

### Native graph study

- G-full: frozen intact WindFarm incumbent.
- G-oracle: train-evidence cover, replayed on disjoint queries; oracle labels never used at deployment.
- G-input: one fitted input-only organizer with native fine computations.
- G-fixed: fixed training-derived cover or matched realized-budget control.
- G-pair: a bounded ungrouped source-mask control if needed to test the value of shared grouping.

Only G-input is a required newly fitted organizer. Controls should reuse weights/forward data wherever valid. Report teacher-query, reference-label, and inference inputs separately.

Fine-tune native kernels only after the frozen-weight comparison identifies the role of organization. Any joint fine-tuning then requires its matched full-access continuation; do not attribute extra optimization budget to the graph.

### Combined inverse model

Select one coherent checkpoint/policy from calibration for the bounded reference-corrected inverse test. Do not combine the best field checkpoint with a different checkpoint's port, graph, or timing statistics.

## 13. Tests that target the actual mechanisms

Use focused tests and real native computation. Tests do not replace the experiments.

- Native checkpoint replay: fields, all physical roles, units, predicted ports, target-free inputs.
- Full-access policy identity with original parameters/common head and no tree overhead in disabled mode.
- Unique QM/QE pair union, overlapping memberships, additive normalization and one attention denominator.
- At least one pre-contextualization masked MM/ME/EM call with a known source path.
- Field/head and port parameter update inventory; no hidden source-state detachment.
- Correct historical replay sampling and independently counted near/pressure coverage.
- Per-family and per-case loss aggregation; no multiplication of independent sample count by Q.
- Finite-response and pressure-baseline-correction algebra, including a constant bias example.
- Teacher-only search accepts valid model-preservation experiments with unknown physical mixed labels, without converting them to reference evidence.
- Input-only organizer cannot read targets, oracle decisions, case IDs, or test arrays.
- Actual fitted native deterministic K and source supports, not injected/synthetic closures only.
- Module permutation, query tiling, source quadrature splitting, and 2-D/3-D adapter integration.
- Fixed-topology AD/FD and measured native recomputed-plan boundary behavior.
- Native reference feasibility rejection, reference-call accounting, and re-anchoring without stale prepared states.
- GPU maps-off timing with warmups, interleaving, synchronization and distinct preparation/decode/application scopes.

Existing trusted-checkpoint/security tests and pre-push hooks must continue to pass. Avoid a repository-wide refactor or new verification platform.

## 14. Work order, budgets and active troubleshooting

### Milestone M0 — restore a sound experiment, not a new framework

Read the report and the exact current local code; compare with this review's `70d0c0a` basis through ordinary Git history. Use the existing run machinery. Run intact native replay, fix the response adapter dispatch, and exercise one native optimizer update plus one native cover forward on GPU 2 if available.

### M1 — run R and G independently

Run the response/local-model diagnosis and small matched native refit pilots. In the same round execute native WindFarm oracle cover observations and fit the first organizer. G must not wait for every Thermal mixed/pressure criterion to pass.

### M2 — bounded learned candidates

Response review points are 100, 300 and 1,000 actual optimizer updates. A selected matched pair may continue to at most 3,000 updates per arm if learning and replay evidence warrant it. No synthetic epoch accounting based on three stencils should imply a mature full-dataset training budget.

Organizer review points are 100, 300 and 1,000 updates, with at most 3,000 updates or 500 complete passes through its training corpus, whichever comes first. Evaluate actual deterministic covers at each review. Do not wait 500 epochs to discover that every node has an inactive gradient or all labels are identical.

### M3 — integrate and conclude with native evidence

Evaluate matched native values, organization, context transfer and complete cost. Execute the bounded reference-corrected decision panel appropriate to available evidence. Return separate conclusions for response recovery, computational graph learning, physical interpretation and design utility.

### Initial resource envelope

This is a new-round ceiling, not an instruction to consume all resources:

| Resource | Ceiling / intent |
|---|---|
| Pilot remedies | At most 3 recipes, 200 actual optimizer updates each; 600 total; at most 3 GPU-hours |
| Selected response pair | At most 3,000 updates per arm; reviews at 100/300/1,000 |
| Selected organizer | One fitted candidate, at most 3,000 updates or 500 corpus passes |
| Oracle discovery | At most 4,096 actual native candidate observations total; 48 per initial case |
| All new neural/cover work | At most 24 GPU-hours, including pilots and diagnostics |
| New local reference solves | At most 192 attempts and 4 aggregate CPU-process hours; reuse existing appropriate records |
| Reference allocation | Up to 96 coverage/resolution attempts, 24 pressure-boundary attempts, 48 inverse attempts, 24 refinement/recovery reserve |
| Long training | No automatic 5,000-epoch continuation |

Use observed runtimes to reduce batch/query budgets before launching. Do not interrupt unrelated jobs, use another occupied GPU without authorization, or inflate precision/work merely to consume the budget. WindFarm new-CFD generation is not included unless an actual local reference capability is verified and its budget fits; otherwise prepare the missing-data request.

### Active remedies Codex is authorized to try

After an actual failure, identify the first failing computation and try a bounded remedy rather than escalating an ordinary issue immediately.

- Weak response updates: inspect optimizer scope/rate and target/input scales; try one smaller/larger step or one late native block unfreeze.
- Value forgetting: restore broad replay, narrow the trainable scope, or reduce the measured conflicting objective; optionally one gradient-projection recipe.
- Local response limitation: test measured-condition replay, then one local-output adaptation with local-data replay.
- No oracle splits: inspect whether children were also pruned, whether source unions genuinely differ, and whether the cost estimate matches measured work; try one alternative geometry/relevance index on train data.
- Oracle covers work but amortization fails: inspect class imbalance, saturated forward coefficients and missing context; train classification logits with alternative adequate labels rather than lowering a held accuracy threshold.
- Graph has little output effect: audit coarse/global bypasses and organize the consequential transport; do not celebrate a tiny K on an irrelevant branch.
- Packed execution is slow: compare dense masked reference, larger inference tiles and batched source blocks; retain the measured faster path while accurately reporting sparsity. No custom kernel by default.
- Reference labels are unresolved: change the question/role/finite scale, or reserve a mesh check. Do not train on a noise-amplified derivative or abandon all other workstreams.

Ordinary diagnostics are nonblocking scientific checks, not approval systems. Stop a particular line when its bounded remedies fail; finish the independent line and state what remains unresolved. A report consisting only of “Stage A failed, so no organizer was tried” is not the intended outcome.

## 15. Definition of done and report requirements

Deliver durable code/config/tests and one concise report containing:

1. The intact checkpoints actually used and verification that no common head/source control was silently discarded.
2. Native response refit comparisons with broad replay, family/M/Re coverage, and local-module diagnosis.
3. At least one real oracle cover panel on native WindFarm data.
4. One actually fitted input-only organizer, or concrete native observations showing why the oracle training target is unavailable after the permitted alternatives.
5. Deterministic case-dependent K, source supports, fixed-budget controls and disjoint-query fidelity; explicitly distinguish G-read from G-transport.
6. Physical reference, teacher preservation, computational intervention and synthetic test conclusions stated separately.
7. Pressure bias versus finite-change error and the outcome of local baseline correction; no feasibility-rate claim from an all-feasible panel.
8. A matched reference-corrected design result where the available quantity evidence supports it, or a bounded labelled negative control with its limitations. WindFarm stored-library evidence stays separate from continuous design.
9. Native boundary and gradient findings; unresolved topology transitions reported as unresolved.
10. Complete measured cost, attempted/completed updates and reference calls, with missing timings marked unavailable rather than fabricated.
11. Which mechanism to retain next and which specific new evidence would change the conclusion.

Commit only maintained source, reusable tests, configs and the written report. Keep large arrays, checkpoints, generated figures, one-time scripts and machine-specific paths in the existing ignored locations. Follow `AGENTS.md`, audit the entire outgoing range and use the existing pre-push artifact hook. Do not introduce hashes, contract freezes, approval databases, monitoring daemons or new baseline-snapshot systems. Preserve existing security and history.

## 16. Interpretation of possible outcomes

**R improves and G learns useful covers:** combine them with a matched full-access refit and proceed to reference-corrected inverse evaluation. This is a candidate HONF advance.

**R still fails but G preserves native WindFarm physics with variable K:** retain the structural result; it is not a Thermal inverse-ready model. Continue physical response/data work without declaring the graph unsuccessful.

**R improves but G remains full access:** retain the response improvement. Report whether the oracle found no advantageous sparse cover or whether the organizer failed to imitate an available one. Those are different bottlenecks.

**G-read works but dense preparation/global paths carry all physical information:** retain only a computational read result. Move the graph to the consequential transport before claiming a physically sufficient organization.

**A graph helps nominal-grid decisions but not refined references:** report benchmark-only utility and prioritize reference fidelity, not a stronger interpretability claim.

**No line improves within budget:** preserve the negative results and narrow the next question using actual native errors and source-path interventions. Do not propose another large round of infrastructure as a substitute for the missing scientific evidence.

## 17. Sources and status of the proposed method

The experiment design in this document is a new proposal. It is not a report of executed native recovery, trained dynamic K, or validated optimization.

### Repository evidence

- **E1:** `HONF_Proj/docs/reports/_bk/20260926_195912Z_HONF_Decision_Aware_Dynamic_K_Study.md`, especially M0, paired u300, WindFarm transfer, and cover status, at `70d0c0a`.
- **E2:** `HONF_Proj/docs/reports/HONF_Thermal_Physical_Interaction_Evidence.md` and the preceding `HONF_Interaction_Response_and_Inverse_Design_Study.md`; preserve local-generator and numerical-resolution scope.
- **E3:** `HONF_Source_Conditioned_Core_Diagnostics.md` and mature 1502/1804 reports; selected and endpoint states differ.
- **C1:** `src/honf_forward_core/interface_fields/checkpoint_warm_start.py`; conversion explicitly discards tensors and disclaims prediction identity.
- **C2:** `Case_ThermalChannel/src/channelthermal/response_control/native.py`; current architecture requirement and native role/port adapter.
- **C3:** `response_control/runner.py`, `training.py`, `losses.py`; stencil-only current loss, source-rate optimizer construction and gradient-norm weighting.
- **C4:** `src/honf_forward_core/interface_fields/adaptive_cover_field.py`, `adaptive_interaction_cover.py`, `adaptive_cover_oracle.py`; current topology, continuous access, pair compilation and physical-adequacy-only search.
- **C5:** `src/honf_forward_core/interface_fields/core.py`; native coarse/local and backend contributions, important for bypass auditing.
- **C6:** `Case_ThermalChannel/src/channelthermal/local_coupling.py`; port, refinement and local-response semantics.
- **W1:** `Case_WindFarm/docs/baseline_decision.md`, current native model/data/workflows, and `Dataset/PHYSICS_AND_DATA.md`; use actual code and later reports rather than a stale preprocessing-only README.

### Methodological references (not claims of HONF success)

- **M1 — Function-preserving initialization:** Chen, Goodfellow and Shlens, *Net2Net: Accelerating Learning via Knowledge Transfer*, ICLR 2016, https://arxiv.org/abs/1511.05641. The principle motivates preserving a trained function; this plan is not an implementation of Net2Wider/Net2Deeper.
- **M2 — State plus design-response learning:** Luo, O'Leary-Roseberry, Chen and Ghattas, *Efficient PDE-Constrained Optimization Under High-Dimensional Uncertainty Using Derivative-Informed Neural Operators*, SIAM J. Scientific Computing 47(4), 2025, DOI 10.1137/23M157956X; https://arxiv.org/abs/2305.20053. Their independently supplied derivative evidence is not equivalent to noisy differences of our raster labels.
- **M3 — Conditional remedy for gradient conflict:** Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020, https://arxiv.org/abs/2001.06782. Use only after measuring a relevant conflict; it provides no physical or generalization guarantee here.
- **M4 — Reference-managed optimization:** Booker et al., *A Rigorous Framework for Optimization of Expensive Functions by Surrogates*, Structural Optimization 17, 1999; NASA technical report https://ntrs.nasa.gov/citations/19990009055. The present bounded neural/hypergraph experiment does not inherit its convergence theorem merely by using a trust region.

**Central criterion:** an adaptive hypergraph should be an experimentally supported organization of necessary collective information, not a prerequisite that forces destruction of a good predictor and not a decorative statistic attached to an untouched dense computation.
