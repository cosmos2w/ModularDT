# HONF next phase: a shared interaction core, better thermal context, and reusable receiver packets

**Work mode:** Goal mode, bounded development. **Reviewed baseline:** `agent/honf-core-next` at `10193abd62a4d1babe6015ac67f72a78a99f2866` (7 October 2026). **Primary evidence:** `HONF_Proj/docs/reports/_bk/20261007_031743Z_HONF_RDirect_Formal_Readiness_and_Receiver_Local_Organization_Report.md`, including its completed-formal addendum, not just its historical readiness section. **Status of this document:** proposed next work; none of its experiments is a completed result.

## 0. The goal and the decision

The project goal remains: **learn from multi-field reconstruction → obtain a reusable organization of module–environment interactions → reuse it in flexible modular inverse design.** The forward predictor is the learning vehicle; an interpretable, useful interaction interface is the central representation. Latent slots, trees, smaller K, and speedups are mechanisms or possible benefits—not substitutes for this goal.

Retain completed Run3901 D-sep5000 and Run3902 R-direct5000 as immutable research references. Retain R-direct's saved-best e4200 separately. Retain Dense1804, HONF1401, HONF1502, R-group, Run3801 and all previous scientific histories. Do not restart or extend these formal runs. This round is not another full-data5000 campaign.

Build on the inexpensive source-resolved computations. Do not reintroduce recursive Tree construction, the historical P0/P1/P2 thermal loop, a second complete thermal predictor, or forced global compression. The next phase must deliver executable code and measured learning, not just another set of qualification reports.

### Four concrete deliverables

1. A **shared, dataset-independent interaction implementation** with explicit dataset-owned dependency and output contracts, preserving the existing Thermal predictor under a compatibility configuration.
2. A **single matched Thermal comparison**: geometry-only response preparation versus preparation augmented by the already learned, heat-independent flow. Give healthy candidates a meaningful development fit.
3. An **executable receiver-packet interface plus one small learned packet proposer**, derived from the frozen reconstruction-trained response model. Keep model fidelity, physical fidelity, and speed claims separate. This is not a new global latent-factor model.
4. A **bounded real WindFarm learning pilot** using the same shared implementation and a genuinely nonlinear field head, with Wind-owned inputs and units. This tests implementation portability and learning feasibility, not zero-shot Thermal-to-Wind transfer or mature Wind performance.

The main study question is: **Can we preserve the source-resolved efficiency, improve the use of environmental physics, and turn the learned interactions into a receiver-specific interface that is reusable across datasets without imposing ThermalChannel's forcing law on them?**

## 1. What the completed results actually establish

### 1.1 Accuracy: retain the strengths without exaggerating them

The completed literal-e5000 comparison uses canonical89, excludes cross-split duplicate0273, and remains repeatedly exposed validation evidence. These are actual trained-system comparisons; objectives, normalization histories and total component updates differ.

| Native equal-case mean RMSE | R-direct + D-sep | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| Fluid u | 0.00432883 | 0.00650424 | 0.00960653 | 0.00970305 |
| Fluid temperature | 0.342425 | 0.237375 | 0.312974 | 0.267942 |
| Surface temperature | 0.311198 | 0.429475 | 0.515004 | 0.520884 |
| Material temperature | 0.307594 | 0.329348 | 0.421362 | 0.416504 |
| Module material peak | 0.338021 | 0.317573 | 0.478912 | 0.466868 |
| Fluid vorticity | 0.0375592 | 0.0244341 | 0.0260128 | 0.028128 |

R-direct is not uniformly more accurate than1401/1502: nominal fluid temperature and vorticity remain worse. Its important gains are streamwise velocity, several pressure measures, mean surface/material temperature, and heat-response accuracy. Temperature and material-peak tails must remain visible. Do not combine dimensional channels into an arbitrary physical Euclidean score.

Fluid-temperature relative L2 is 4.453% versus Dense's3.244%, while equal-case RMSE is44.3% higher. These are different measures. Neither “catastrophically inaccurate” nor “much better on all fields” is supported.

### 1.2 Response and interpretation

On the three primary existing layouts/six signed heat changes, R-direct fluid/surface/material response RMSE is0.00884525/0.0153821/0.0187677. Dense1804 gives0.0105621/0.0180586/0.0187927. This is meaningful progress, but material response is essentially tied. For0291, R-direct's mean change magnitude is0.0402401 against reference approximately0.028916: directions are right, but amplitude is about39.2% too large.

Each independently solved heating family still supplies only one balanced direction and its opposite. Individual source-kernel columns are not separately validated physical Green's functions. More receiver points, more training epochs, or operator-residual columns do not create additional independently solved excitation directions.

Geometry changes remain a different problem: over16 existing moves, R-direct mean fluid-temperature response RMSE is0.135146 against Dense0.113314. The displayed0348 move has an incorrect mean response direction. Exact model derivatives do not cure that miss.

### 1.3 Cost

The fixed-four complete native latency comparison gives25.239 ms R-direct,163.290 ms Dense1804,52.530 ms HONF1401,229.506 ms HONF1502. It includes input staging and output transfer, with checkpoint-native chunks. This is an actual-system comparison—not a matched-kernel architecture isolation.

The fresh full-TRAIN flow5000 and thermal5000 together cost5.5455 training hours,5.6640 process hours. Each component has65,000 updates, so the complete recipe contains130,000 updates and6,000,000 case visits. Do not call it one matched5000-epoch joint fit. Historical training-time comparisons are not controlled unless their work and hardware scopes are matched.

Reported incremental inference allocations and training peaks are different measurements; preserve that distinction. The dramatic cost improvement came from a much simpler computational path, not a proved sparse executor.

### 1.4 Organization

The formal local covers retain all donors in37/64 patches at1% and34/64 at2%. Their bounds control distortion of the learned model inside a declared heating-change box, not physical error. On0291, a donor can be omitted partly because its permitted heating-change radius is small—not because its raw kernel is negligible.

R-direct currently has a physical source-to-receiver response network. It does **not** have a learned latent hyperedge bank. A post-fit cover can be represented as an overlapping hypergraph, but drawing it as one does not establish learned many-body grouping or an organizing advantage beyond the predictor. Keep that distinction explicit throughout this phase.

## 2. Source review and changes to make

The following files were inspected at10193ab. Treat these as starting points, not permission to rewrite the repository.

| Current source | Relevant finding | Next action |
|---|---|---|
| `src/honf_forward_core/interface_fields/source_response_operator.py` | Already generic in2-D/3-D, source/context widths and output width; two source/environment contextualization rounds; source-resolved near/far reads; scalar forcing is applied afterward | Extract/reuse this context/read machinery. Preserve affine behavior exactly under the compatibility configuration. Add a separate nonlinear readout contract rather than disguising nonlinear physics as heating |
| `Case_ThermalChannel/src/channelthermal/source_response.py` | Owns2-D128×64 native grid, disk geometry, material ordering, zero-boundary capability, native interpolation, q/h proxies; thermal and flow preparation are parallel, and outputs are concatenated | Keep native extraction here. Add optional predicted environmental flow through an explicit dependency edge and a zero-initialized adapter projection |
| `src/honf_forward_core/interface_fields/geometry_flow_field.py` | Small source-resolved flow reader, but currently hardcodes2-D Fourier widths, four outputs and an experimental heat-column convention | Preserve legacy class/checkpoints. Move new dependency semantics to the adapter and make newly shared geometry/read primitives dimension/output configurable |
| `tools/thermal_source_response_fit.py` | Separate strict development and formal bindings now exist; measured-response and qualified TRAIN discrete-balance terms are active | Reuse these tools and identity rules. Add only a separately identified development recipe and the optional context provider |
| `Case_ThermalChannel/src/channelthermal/source_response_residual.py` | Discrete operator is a training-side, generator-qualified stencil using stored TRAIN velocities | Do not put it inside generic inference. Retain it unchanged for the matched Thermal pair unless a real correctness error is found |
| `Case_WindFarm/src/windfarm/model.py` | Wind still dispatches its existing `InterfaceFieldCore` families, not the new source-response implementation | Add an opt-in new-family bridge, retaining all existing paths. A common facade alone is not common implementation |
| `docs/guides/Shared_Core_Thermal_Wind_Architecture_Map.md` | Describes the older shared interface-field families | Append the new architecture and distinguish old compatibility from new shared-core evidence |
| `AGENTS.md` and `docs/guides/Thermal_Model_Development_Protocol.md` | Fixed development split, report structure,100-epoch cadence and artifact/security requirements | Follow them, with the explicitly bounded maturation provisions below |

A key code-level hypothesis is now testable: **the thermal branch learns advection-related geometry effects again even though the composed model already predicts flow.** This does not prove the absence of flow features causes the thermal error. It motivates one controlled experiment, not an unconditional redesign.

## 3. General mathematical contract

### 3.1 Separate objects that have been conflated

Let a scene be

\[
\mathcal X=(\mathcal M,\mathcal E,c,\mathcal D),\qquad
\mathcal M=\{(\mathrm{id}_i,x_i,a_i,\mu_i)\},\quad
\mathcal E=\{(\mathrm{id}_j,e_j,b_j,\nu_j)\}.
\]

Here modules and environment records have physical identities, locations, attributes and measures; c is operating context; and \(\mathcal D\) is a dataset-owned dependency specification. The requested receivers are \(\mathcal Q=\{(q_r,\mathrm{role}_r,w_r)\}\). Controls \(z\) and observations \(o\) are distinct objects. Supervision values must not be reachable through the inference scene.

Distinguish **configuration controls**, whose changes rebuild the contextual state, from **separately applicable controls**, whose changes are supported by a declared response law. Heat can be separately applicable in the current Thermal capability. Turbine position is not a heating amplitude; it changes configuration. Environmental context donors are not independently actuated sources unless the dataset explicitly supplies that meaning and supervision.

A reusable neural context encoder produces

\[
(Z_M,Z_E,z_g)=\mathcal C_\theta(\mathcal X),
\]

using typed source–source, source–environment and environment–source interactions. An individual source descriptor can depend on the complete layout. Therefore, source-resolved output is not the same as assuming independent geometric effects or ignoring many-body interactions.

### 3.2 Affine readout is a capability, not the universal model

For a qualified affine output block a,

\[
\widehat y_a(q;\mathcal X,h)
=b_a(q;\mathcal X)+\sum_i K_{ai}(q;\mathcal X)h_i.
\]

For a future vector-control block, carry an explicit control-component index and use K_{a,i,beta}; do not fake several independent physical modules to represent several controls on the same module. The current Thermal scalar-heating API remains a backward-compatible specialization.

Keep the current exact near/far blend,

\[
K_{ai}=w_{qi}K^{\rm near}_{ai}+(1-w_{qi})K^{\rm far}_{ai},
\]

and native physical-source identities. Thermal currently declares zero offset under its zero inlet/wall/initial-temperature benchmark. The shared core may support a nonzero offset; another dataset needs an explicit adapter/capability and evidence, not a silent change to Thermal normalization.

An arbitrary callback \(h\mapsto\phi(h)\) followed by a fixed K does not by itself represent general state-dependent nonlinear coupling. Do not advertise that callback as the generalization solution.

### 3.3 Nonlinear field readout for other cases

Provide a small nonlinear source-resolved readout using the same actual contextualization and geometric feature functions:

\[
m_i(q)=\Phi_\theta(Z_i,z_g,q-x_i,c,\zeta_i),\qquad
m_E(q)=\sum_j\nu_j\Phi^E_\theta(Z^E_j,z_g,q-e_j,c),
\]

\[
\widehat y(q)=\Psi_\theta\!\left(q,c,z_g,\operatorname{Reduce}_i m_i(q),m_E(q)\right).
\]

The exact environment-read choice may be a prepared contextual representation rather than a full Q×E fine read; report which is executed. Do not add the old dense QE stack merely to make this equation literal. The source-wise nonlinear message is evaluated **before** reducing across physical sources. Preserve this lesson from the early Dense/HONF comparisons.

Controls that change this contextualization must enter it and invalidate prepared state. The nonlinear head's source messages are computational contributions, not automatically \(\partial y/\partial z_i\). Provide actual AD/JVP/VJP-based local sensitivities separately; a local linearization has a declared operating point and is not an exact finite-response law.

Initially, weights are dataset-specific. Shared code, shared semantic contracts, shared weights, cross-dataset pretraining and zero-shot generalization are five different claims. This phase requires the first two and tests learning with the first; it does not claim the remaining three.

### 3.4 Dependency graph versus training stages

For current Thermal,

\[
\widehat v=F_\phi(\mathcal G,c_f),\qquad
K=K_\theta(\mathcal G,c_T,\mathcal E,\widehat v_E),\qquad
T=K h.
\]

The new optional edge is **predicted flow → thermal coefficient preparation**. Heat remains absent from F and from K preparation. Thus, at fixed geometry,

\[
\partial \widehat v/\partial h=0,
\]

while geometry can influence both outputs and \(\widehat v_E\) can influence K. A single public model interface can execute this directed computation without one monolithic parameter set, simultaneous fitting, or recurrent coupling.

The general dependency specification must also represent nonlinear or coupled cases without globally imposing this triangular Thermal graph. This phase implements the existing triangular graph and a non-affine Wind graph. Do not add an untested universal iterative multiphysics solver. For strongly coupled future cases, the contract must reject an invalid affine cache and expose the need for a different state update.

### 3.5 General scaling contract

Use physical lengths, coordinate frames, measures, masks and field scales supplied by adapters. No generic12×6 domain,80-m rotor,192 environment donors, four flow channels, zero thermal boundary or heat-column index may be silently assumed by the new shared implementation. Keep explicit legacy compatibility defaults inside old classes when needed to replay old checkpoints.

The cheap complete source-resolved path should retain roughly two context rounds with source-pair/source-environment work plus chunked receiver–source work. Avoid materializing all-field outputs or dense Q×M×hidden diagnostic tensors unless requested. Do not assume that this scaling will remain fast for arbitrary M or Q; benchmark the real second dataset.

## 4. Workstream A — shared implementation with backward-compatible numerical behavior

### A1. Minimal extraction, not a new framework

Prefer a small shared context object and two readout types near the existing source-response module. Proposed names are illustrative: `InteractionScene`, `DependencySpec`, `PreparedInteractionState`, `AffineResponseReadout`, `NonlinearFieldReadout`. Reuse existing repository types where their semantics fit. Do not build a second registry, provenance database, training platform or generic configuration hierarchy.

Keep checkpoint names and state ownership stable. If extraction changes state keys, provide an explicit reversible mapping, inventory every tensor and verify converted outputs. Never silently initialize a missing tensor. Saved Run3901/3902 files must not be rewritten. Conversion artifacts are new opt-in identities.

A shared implementation is demonstrated by the same executed class/functions in Thermal and Wind, not merely both wrapping an object called `HONF`.

### A2. Preserve actual input derivatives

At immutable retained weights, compare old and compatibility paths on fixed TRAIN low/high-M and the four fixed representatives, including Q1024/Q8192, fluid/surface/material/ports, module permutations, environment-measure splitting, query chunks, input heat and geometry VJPs, and cold/prepared output equality. Reuse existing tolerances and precise arithmetic tests. Preserve historical FP32 failures rather than silently broadening the acceptance rule.

Frozen flow parameters do not mean detached geometry. For geometry derivatives, recompute predicted environmental flow in the current AD graph. A training-only cache of fixed-geometry predicted inputs is permissible only in a declared fixed-input fitting scope; it must be disabled for geometry differentiation and invalidated by geometry/context changes.

### A3. Engineering acceptance

The compatibility path should preserve outputs at existing tolerances and add no more than10% median complete inference or memory overhead on the fixed timing panel. This is a proposed engineering acceptance band, not a scientific accuracy target. A failed refactor returns to the functioning original implementation; it does not cause another architecture search.

## 5. Workstream B — one matched Thermal environmental-context experiment

### B1. Hypothesis and exact arms

**Hypothesis:** exposing the already learned advecting velocity to thermal context improves nominal fluid-temperature reconstruction, especially downstream, while preserving the source-response law, interface accuracy and heat-response quality.

Use the retained **development** R-direct2500 and its development D-sep2500 partner as parents. Do not initialize from Run3902 full-TRAIN5000 and call the result a150-case development fit. Parent hashes, optimizer state and normalization come from the actual development checkpoints. Formal5000 results remain read-only context.

- **R-geom:** unchanged geometry-only thermal response preparation, continued under the new common development schedule.
- **R-flowctx:** same parent and same thermal parameters, plus one small projected predicted-flow feature input at environment donors. The frozen flow partner is identical to R-geom's.

Both use the same source-response near/far architecture, reconstruction/response/operator objectives, sampled cases and receivers, optimizer settings and schedule. The added input branch and its measured cost are the intended difference. No simultaneous loss-weight search, nonnegative-head search, new pooling, or new hyperedge bank is allowed.

### B2. Input and injection specification

At the existing environment coordinates, query the frozen D-sep reader for **predicted physical u and v**. Do not inject pressure or vorticity into the thermal branch by default: the qualified thermal stencil advects with velocity, and those channels introduce an unnecessary second hypothesis. Use dataset-owned velocity units and a TRAIN-defined scale. Append a validity/type flag where needed.

Environment donors inside solid regions must not acquire unqualified predicted-fluid values. Preserve the existing geometry tokens; set only the optional flow-feature channel to a declared invalid/zero encoding there. Do not remove those original environment records or change quadrature weights merely to insert the new feature.

A zero-initialized linear projection adds the flow features to the existing environment embedding. Verify exact or existing-tolerance parent recovery at update0. The geometry-only arm omits the branch or uses the identical declared zero-feature control; document trainable parameter counts and do not claim perfect parameter matching if they differ.

D-sep predictions must be the context in both fitting and inference. Stored TRAIN velocities remain permissible **only inside the existing supervision-side discrete residual**. A stored-velocity upper-bound probe, if used at all, is confined to four TRAIN inputs, zero optimizer updates and explicit oracle labeling; it is not the main candidate.

Generate predicted environment features once per unchanged training geometry and price their formation. At cold inference, price flow preparation, environment queries, thermal preparation, output queries and role extraction together. Reuse the same flow state for environment and output queries, rather than preparing it twice. Never cache a physical target.

### B3. Maturation rules

Freeze a2500-new-epoch schedule before the first update:20-epoch warmup from3e-6 to5e-5; hold to1000; cosine decay to3e-6 at2500. This is a separately identified continuation child, not a rewrite of the original parent schedule. Preserve parent thermal AdamW moments for old parameters, initialize new-branch moments empty, and use the same recipe for both arms. Codex may amend this exact schedule once before launch if checkpoint inspection shows a concrete incompatibility; state the reason and freeze the replacement before either arm learns.

Review at100,500 and1000. Healthy, affordable learning should normally reach **1000 new epochs**. Do not stop at100/200 simply because an inverse or geometry gate is still unsolved. Stop for corruption, nonfinite behavior, a substantial sustained regression, or the actual global resource ceiling—not for failure to prove the entire HONF hypothesis early.

A matched extension to2500 is permitted within this plan when the last saved500-epoch interval at1000 still improves the common DEV22 three-temperature selector by at least1%, protected roles are not showing sustained deterioration, and the remaining budget preserves the Wind pilot and report reserve. This1% is a continuation heuristic, not a significance test. Otherwise retain completed1000 as the bounded conclusion. No extra candidate or seed may be substituted for the extension.

Save checkpoints/latest/best and plots every100 epochs, with declared terminal states. Report literal matched endpoints and separately selected checkpoints. Reuse the existing common three-temperature selector; add fluid/far-temperature and peak-tail curves as explicit measurements rather than replacing the selector retrospectively.

### B4. Evaluation and practical selection

All22 DEV cases supply statistics. Detailed exports default to0277/0291/0294/0687, with four input-declared downstream/near/interface strata. Existing response TRAIN/development/audit cohorts retain their literal exposure labels; no new reference response is generated.

Proposed promotion target: at least10% lower mean nominal fluid-T RMSE than matched R-geom, no more than5% deterioration in mean surface/material/heat-response RMSE, no more than10% deterioration in the corresponding case p90 and module-peak p90, and complete inference/VJP overhead within20%. Also show paired counts and actual units. These are practical research thresholds, not certified physical risk limits. A modest tradeoff can remain a named candidate but must not automatically replace the reference.

Because flow is frozen, its ordinary output should be unchanged across the pair. Verify this; do not present inherited flow accuracy as a new thermal-fit gain. No new vorticity-remedy experiment is included.

### B5. A small sign audit, not another fitting branch

The formal report records16031 negative coefficients among39185 fluid heater coefficients on0291. Count alone does not measure physical severity: tiny negative tails differ from substantial negative response. Measure negative kernel norm, magnitude quantiles and contribution mass on a small TRAIN panel, split near/far, without modifying inference.

Inspect whether the exact Thermal discretization and boundary/stopping policy support a monotone single-source heating response. A suitable proof may use the sign structure of the qualified discrete operator or positivity of its stable update, with its assumptions written explicitly. Do not infer universal positivity from the word “heat,” and do not solve or invert the physical operator under the exhausted ledger. If positivity is justified, classify substantial negative learned kernels as a physical-interpretation warning and propose a separate future capability-owned remedy. Do not clamp all current kernels, impose positivity on general vector fields, or add a third trained Thermal candidate in this phase.

A negative balanced-transfer field can be correct even when all individual temperature/heating kernels are nonnegative. Separate the sign of K from the sign of delta heat.

## 6. Workstream C — reusable receiver packets without early pooling

### C1. What a packet is

Define a directed interaction packet

\[
e=(\mathcal R_e,\mathcal S^M_e,\mathcal S^E_e,\text{output/control types},\text{response data},\text{validity domain}).
\]

It binds one receiver region or receiver role to a set of physical module sources and explicitly documented environment/context ancestry. Packets may overlap in sources and receivers. A source keeps its identity and individual nonlinear read/response coefficient; a packet is not permission to average raw source inputs before interactions are evaluated.

For overlapping receiver packets, the initial executor uses the union of admitted physical sources at each receiver and evaluates each receiver/source pair exactly once. Packet overlap must never double-count a heater. Price that actual union, not a sum of separately advertised small packet sizes. Any later weighted overlapping composition needs its own explicit partition-of-unity/duplicate-correction contract; it is not implicit in a picture.

For Thermal, its contribution is a selection of actual source-resolved coefficients or source increments. For a general nonlinear field, its fine-source messages and local Jacobian are distinct exports. Environment context can remain global; if so, record it as global ancestry instead of drawing a false sparse environment support graph.

This is a hypergraph-compatible interface, not proof of irreducible many-body physics. A truly higher-order physical claim requires the relevant physical perturbation evidence. Fixed-geometry Thermal is affine in heating, so do not train fictitious heat–heat nonlinear terms merely to obtain a hyperedge story.

### C2. The existing response-based cover remains the teacher

For normalized patch quadrature W_P, define raw sensitivity and admissible-action importance separately:

\[
s_i(P)=\|W_P^{1/2}K_i\|_2,\qquad
q_i(P)=r_i s_i(P).
\]

With |delta h_i| <= r_i, a retained set S obeys

\[
\|\Delta\widehat T_P-\Delta\widehat T_{P,S}\|_{W_P}
\leq\sum_{i\notin S}r_i\|K_i\|_{W_P}.
\]

The guarantee concerns the learned affine model over its declared box, not physical truth. The balanced-input restriction may strengthen a bound but is not necessary for this conservative triangle form. Keep both raw sensitivities and feasible radii visible; a nearly saturated actuator must not be labeled physically unimportant merely because its allowed action is small.

Use the existing1%/2% of TRAIN response-RMS budgets and consistent physical units. The authoritative direct predictor remains unchanged. Canonical4×4 patches are an evaluation catalogue, not a claimed learned spatial partition.

### C3. Fit one small amortized packet proposer

After choosing the development predictor by the sealed rule, freeze it. Build a TRAIN-only table from its actual K on the150 primary training layouts and fixed input-selected patch quadrature. Predict patch/source raw sensitivity from the inexpensive prepared source/environment context, source geometry, receiver-region features and output/control types. A small shared pair scorer is sufficient; cap it at50000 trainable parameters and do not build another contextual encoder.

Train on log sensitivity with a TRAIN-derived floor, using up to500 packet-fit epochs,100-cadence reviews, one seed. Include a TRAIN-only conservative residual margin estimated with whole-layout folds or an explicit TRAIN calibration holdout. No DEV target response or physical audit selects this margin. The projector from predicted importance to a cover is deterministic and budget-based. This trains an **amortizer of a reconstruction-trained response representation**, not a new physically supervised hypergraph from scratch.

At inference, the proposer may examine scene/context and receiver metadata but **must not evaluate the full expensive kernel merely to decide what kernel entries to evaluate**. Independently price proposal preparation and full-kernel verification. Evaluate on all22 DEV layouts using exact full-model coefficients only as the verifier. These are teacher-fidelity labels, not new physical labels.

An exact omission certificate based on evaluated full K is available only after paying for that K. The learned proposer cannot inherit that mathematical certificate merely by predicting its scores. Thus report two distinct modes: offline exact-model certification, and empirically calibrated experimental proposal. Do not hide a full-kernel verification call inside a claimed fast online proposal.

Do not use a force-small-K penalty, prescribed K histogram, diversity bonus, or target global slot count. Include the all-source option. Uncertainty or lack of a qualified omission bound yields full access, recorded as fallback. Learned margin coverage is empirical; do not call it a uniform mathematical bound on unseen scenes.

### C4. Three execution modes and one bounded subset implementation

Keep three explicit modes: `full` (default), `packet_advisory` (exports a proposed set without changing prediction), and `packet_experimental` (uses a proposed set under its declared validity domain). Experimental mode must never silently become the production default.

Implement one small actual subset prototype that gathers selected source–receiver inputs **before** the expensive fine MLP, with the original near-source reads protected. Bucket receiver blocks with the same source lists, or use bounded batched gathers. Avoid Python per-source/per-query loops and no hard/soft shadow model. Cheap dense geometry/scores can remain and must be priced. The original contextual preparation can remain full; explicitly report that ancestry.

For fixed-layout forcing reuse, preserve the exact baseline and approximate only increments. For a cold absolute field, a separate absolute-domain omission rule is needed; do not substitute the incremental cover. A packet that is qualified for one patch/control domain is not automatically qualified for every query or geometry move.

The point is to test whether the learned packet interface can be active and useful without harming the competent predictor. Sparse execution is not required to win at tiny M. If the actual subset is slower, retain only advisory/prepared-reuse functionality and stop executor tuning after this one implementation.

### C5. Measurements that answer the HONF question

Compare the learned proposer, the exact-kernel cover, equal-K nearest-source selection, equal-K upstream selection, and full response. Do not compare different source counts and call it a grouping win. Report native reference error and full-model distortion separately. Include exact omitted-bound violations, conservative fallback counts, protected-near coverage, actual fine rows, complete latency and extra allocated memory.

Distinguish K_src(P), the number of retained physical donors; K_packet(X), the number of nonredundant packet source lists in the declared receiver catalogue; and any algebraic response rank. None is interchangeable with the old latent group capacity. A fixed number of evaluation patches does not demonstrate learned global adaptive K. Variation within the same M and under changed admissible actions is informative; artificial variation is not a success criterion.

A practical proposer result would preserve requested teacher distortion on at least95% of eligible held development patch/action checks with at least25% of patches genuinely partial, and no hidden full-kernel call in proposal formation. These are proposed pilot targets and must be reported with sample dependence and maxima. Missing them does not justify another selector portfolio. Physical advantages are assessed separately and are not guaranteed by passing this pilot target.

## 7. Workstream D — real WindFarm portability pilot

### D1. Scope

Use the new shared contextualization and nonlinear readout on the stored native OpenFOAM WindFarm dataset. This is one fresh pilot, not a new parallel mature Wind campaign. Do not copy Thermal's128×64 grid, zero thermal offset, heat-independent flow rule, harmonic-conductivity proxy, or heat-affine equation into Wind.

Reuse the native view, original case split, wind-aligned3-D coordinates, environmental measures, source metadata and velocity transform. Query existing native cell centers; do not replace the dataset with analytic wakes. No new CFD solve is authorized. Use only prescribed inputs; downstream/rotor velocities are targets unless the existing data contract explicitly labels them as external observations.

### D2. Bounded dataset and fit

Freeze an input-only stratified subset of24 original TRAIN layouts and8 original validation layouts, keeping all available directions within each selected layout. Reuse an existing suitable small manifest when available. Group by layout, not direction row, when separating training from validation. Preserve M and direction coverage; report unavailable strata honestly. Fit normalization only on the24 TRAIN layouts.

Train one fresh nonlinear-head candidate for1000 real optimizer updates, with reviews at100/500/1000, Q1024 and a fixed per-role sampler for volume, near turbine, hub-height, downstream and background. Use the native role definitions and weights; do not infer domain support from a rectangular drawing. Report actual case visits and effective passes, not “1000 epochs.” Cap the pilot at90 minutes of GPU-associated time. Permit a single update-count extension to2000 only if the pilot is clearly learning and the global report/budget reserve remains intact; there is no extra architecture or seed.

Read the retained native Wind predictor on the same pilot validation rows as an unequal-history reference. Also report the available input-defined background-only prediction as a simple control. The pilot should demonstrate genuine improvement over that background and finite geometry/query gradients; it is not required to beat the mature Wind network after a short fit.

### D3. Shared code evidence

Save the actual call trace/class inventory and shared source-file hashes showing that both datasets execute the same contextualization, geometry and source-read primitives. Dataset-specific output heads, feature dimensions, weights and normalizers remain different. A wrapper-only alias or an untrained synthetic3-D pass is insufficient.

For nonlinear Wind outputs, reject exact affine finite-increment requests. Offer a labeled local AD linearization when needed. Prepared state must invalidate when a configuration or context/control input it depends on changes. An externally supplied nonlinear forcing map does not authorize broad unseen thrust or yaw response claims.

Report per-role native m/s errors, complete cost at Q1024/Q8192 on actual low/high-M pilot layouts, and at most two full native representative planes. This determines whether the implementation remains practical beyond Thermal; it does not establish dataset-scale speed superiority.

## 8. Inverse interface: preserve reuse without launching another inverse campaign

The consumer interface should expose `prepare(scene)`, `predict(prepared, receivers, controls)`, `apply_increment(...)` only for qualified blocks, `linearize(...)` for nonlinear blocks, and `export_packets(...)`. Bind units, physical source IDs, exact scene/receiver ownership, dependency capability and output roles. Separate configuration changes from fixed-scene controls.

Demonstrate a bounded software consumer on four TRAIN tasks: full versus packet-advised fixed-geometry heat increments, observed receiver reads and held receiver reads, plus a geometry-change invalidation test. Use existing saved states and deterministic model-only increments. No new denoiser, generator, continuous optimizer, physical design search or claimed successful design is part of this phase.

Do not wait for every fluid/omega metric to beat Dense before building this reusable API. Conversely, a successful API or a correctly ranked three-point pool does not establish inverse quality. The next genuine inverse stage will need independent excitation and decision-level verification; keep the unexecuted reference request explicit.

The solver ledger remains **326/326**. Any new physical reference, rank completion, baseline retry or generated-design evaluation requires separate explicit authorization. Do not silently treat batching multiple right-hand sides as zero new reference work.

## 9. Development protocol, fairness and exposure

Use fixed25_v1 unchanged for all new Thermal fitting:150 TRAIN/22 exposed DEV, every available M category, common seed, common query streams and TRAIN-only normalization. An epoch visits all150 selected cases. Preserve existing original-TRAIN response addenda and distinguish them from primary membership. The four detailed examples remain fixed by inputs.

No new full-data fit or broad full90 evaluation is needed to conduct this development round. Reuse the completed formal tables instead of repeatedly opening them to tune candidates. Formal weights must not become development teachers or initializers without an explicit, separately labeled exposure contract; the default here uses development parents only.

Checkpoint/latest/best updates and training plots occur every100 epochs; Wind update-based pilot monitoring occurs every100 updates and is labeled accordingly. No25-epoch snapshots or duplicate selector copies. At detailed milestones evaluate all22; keep only the small representative full-field/graph exports. Preserve raw evidence and declared milestones, and remove only superseded newly generated visual copies after checking retained ones.

With one seed, paired case/bootstrap intervals describe case variation, not training-seed uncertainty. Near/far/interface, peak, physical responses, approximate-model distortion and solver numerical warnings remain distinct. Never count patches, signed opposites, source columns or repeated receiver points as independent physical experiments.

## 10. Resource plan and dispatch

**Global ceiling:**12 aggregate GPU-associated process-hours and8 elapsed hours, with the final60 minutes reserved for report/test/artifact closeout. These are maxima, not a target to exhaust. Price imports, training, failed starts, GPU evaluation and profiling once per process. Keep actual inner training time distinct from conservative process allocation.

| Workstream | Planned GPU-associated allowance | Purpose |
|---|---:|---|
| Shared-core compatibility and startup | 1.0 h | Parity, input derivatives, native cost |
| Matched Thermal pair, including eligible maturation | 5.0 h | Primary physical-context hypothesis |
| Packet teacher/proposer/subset pilot | 1.5 h | One learned organizing interface |
| Real Wind pilot | 1.5 h | Real second-dataset learning |
| Final evaluations, figures and contingency | 3.0 h | Complete evidence and safe closeout |

Use physical GPUs1/2 when available and authorized. Verify UUIDs and owners. An unrelated job is not permission to terminate it or switch to an unapproved device. If one authorized GPU is occupied, continue on the other, execute CPU refactor/report work, or run under safe measured contention; document effects. Do not wait indefinitely. Do not expand the budget automatically because a nominally spare GPU exists.

Price the first10 actual training epochs/updates with all scheduled objectives, then forecast the remaining work. Preserve the Wind pilot reserve before extending the Thermal pair1000→2500. If a deadline truncates work, checkpoint cleanly and identify the exact completed scope. Do not keep retrying a broken experiment; allow at most two repaired startup attempts for the same integration error, then isolate it and preserve the functioning path.

No full-TRAIN epoch is authorized merely to benchmark this phase. The formal runs are already complete. A future manual formal recipe may be updated only after the development result justifies it; preparation does not mean launching.

## 11. Sequencing and decision gates

**Gate0 — ownership and baseline.** Read the latest local report, AGENTS and current code; verify the branch and protected checkpoints. If the branch advanced beyond10193ab, review the changes before proceeding. Confirm dataset availability and the326/326 ledger without consuming new attempts.

**Gate1 — working shared compatibility.** Implement the minimal shared code and pass the existing-path numerical/input checks. Start the real Wind pilot once its native adapter passes; do not hold it behind thermal scientific success.

**Gate2 — matched Thermal learning.** Run the same first100 epochs, inspect the curves/fields, then continue healthy arms together to500 and normally1000. Decide the single possible2500 extension using the declared rule and remaining budget.

**Gate3 — one frozen packet study.** Freeze the development reference selected from Gate2 under the common rule. Train/evaluate the one packet proposer. Test one actual subset executor; preserve full access as the ordinary model. Do not convert a packet failure into a new tree or a third thermal architecture.

**Gate4 — closeout and model decision.** State separately whether the new context helps physics, whether the packet model helps organization/reuse, and whether shared code learned a real Wind field. A negative result in one does not erase completed engineering or justify claiming success in another.

Implementation changes to repair a demonstrated bug are allowed and must be documented with affected artifacts. Scientific changes after looking at DEV results require a new named child and cannot silently contaminate a matched comparison. This phase authorizes no additional child portfolio; put further ideas in a short future-work section.

## 12. Required tests and code deliverables

Keep the following regression categories compact, reusing existing tests: checkpoint conversion/strict resume; input/target poisoning; source permutation/padding; physical measures and environment-atom refinement where the representation permits it; query chunk consistency; cold/prepared equality; stale geometry/context/control rejection; precise affine increments; flow heat null; live geometry VJPs through predicted flow; near/far equivalence; packet membership-to-execution joins; deterministic full fallback; and2-D/3-D nonlinear capability rejection.

The shared Thermal/Wind map must distinguish: same implementation, different weights; predicted context versus physical targets; full upstream ancestry versus experimental subset reads; affine versus nonlinear output blocks; configuration versus separately applicable controls; advisory covers versus learned proposals; actual executed packet routes versus full fallback.

Use a small number of maintained modules and tools. Proposed deliverables, subject to existing naming conventions:

- Shared interaction contracts/context and affine/nonlinear readout code, preferably factored from current modules.
- A Thermal predicted-flow-context provider and one matched development recipe/tool extension.
- A receiver-packet export/proposal implementation and bounded executor/evaluator extension.
- An opt-in native Wind adapter/profile using the same shared primitives.
- Focused tests, one shared-core guide and one final study report.

Do not upload checkpoints, generated arrays, figures, one-time plot scripts or private local reference outputs. Keep those in ignored storage. Audit the entire outgoing commit range, run the artifact hook and relevant tests, push only durable changes to the non-default branch, and verify remote/local tips. Never weaken trusted-loading/security controls to finish a test.

## 13. Visual and written report contract

### Opening page: direct answers, not a metrics maze

Begin with no more than600 plain-language words answering: What changed? Is the new model more useful? What remains Thermal-specific? Did a learned packet do anything the full predictor or geometry heuristic did not? Did the same code learn on Wind? What exactly should the user retain or launch next?

Then give one table for **Predictor / Organizer / Inverse interface**, each with gained, missed, measured evidence and next action. Add an explicit **A/B/C** assessment: added organizing value; agreement with actual information flow; response generalization outside fitted neighborhoods. Do not substitute software tests for scientific evidence.

### Six figure families, at most six panels per page

1. **Same physical scene, actual fields.** Stored reference, R-geom, R-flowctx and signed residuals at the same coordinates. Show fluid T and the important downstream bias; place all22 statistics in a small adjacent table. The mature classics can be shown from existing saved arrays with their different histories labeled.
2. **What environmental information changed?** Geometry, predicted u/v vectors at actual context donors, and the resulting learned thermal-kernel change. Hidden-feature norm is not physical energy or proof of usefulness.
3. **Which sources matter here?** Three fixed receiver regions, immutable source IDs, raw sensitivity, permitted action radius and signed increment contribution. Draw an actual packet with its donor list; do not hide global environment ancestry.
4. **Is the packet useful?** Full/exact-cover/learned-proposal/equal-K geometry distortion, physical response error and actual latency/rows. Include full-retention cases and a difficult failure, not just the sparsest example.
5. **Same implementation on Wind.** Stored native velocity plane, new nonlinear-head prediction and residual, plus the simple background/retained-native controls on the same coordinates. Name the fit age, layout split and whether the view is a sampled native plane or complete one.
6. **What did it cost and what is reusable?** Complete cold inference, preparation plus repeated-control applications, input-VJP cost, peak allocation, true training time/case visits, and a small API/dependency diagram. Distinguish current phase time from the already completed5.664-hour formal campaign.

Whenever comparing plots, use common physical scales and identical masks/receivers; report any display clipping and compute metrics from unclipped arrays. Preserve0-based source slots consistently; do not switch silently between module0 and disk1. Graph edges must be labeled as memberships, message permissions, kernel sensitivity or contribution, not generic “interaction strength.” Different widths across1401/1502/R-direct remain incomparable unless the underlying quantity has truly been standardized.

Immediately after each figure: one quantitative description and one plain-language takeaway. Keep detailed hashes, seeds, every role/tail/stratum and test receipts in appendices. All prose paragraphs/captions stay on one continuous Markdown source line. PDF masters and small inline rasters stay local, with inspected rendering and working report links.

## 14. Completion criteria and what comes afterward

This phase is complete when it has a preserved compatible shared core, a meaningful matched Thermal learning result, one evaluated learned packet interface, a real bounded Wind learning result, and the compact report/visuals above. It must not close after only implementing APIs or reporting zero-update startup checks. Resource-censored components are explicitly incomplete, not quietly called scientific failures.

A successful result would justify a later manual full-TRAIN follow-up using the improved Thermal recipe and a larger Wind training plan. A packet result earns further development only if it improves useful reference fidelity, preserves it with less measured work, or measurably helps a defined reuse task relative to competent controls. A more attractive graph is not enough.

No new formal launch, new reference solve or inverse-generation campaign follows automatically. Preserve this convergence point: **efficient source-resolved prediction is the backbone; a reusable receiver-conditioned organizer must earn its place without destroying that backbone.**

## Appendix A. Evidence index and boundaries

[E1] `HONF_Proj/docs/reports/_bk/20261007_031743Z_HONF_RDirect_Formal_Readiness_and_Receiver_Local_Organization_Report.md`, completed-formal sections: “Comparison contract and checkpoint identities,” “Endpoint physical results,” “Signed heat changes and precise prepared reuse,” “What organization can R-direct show beside a hypergraph?”, and “Cost, reproducibility and validation.” Its later “Historical readiness” section belongs to the earlier development/preparation stage.

[E2] `HONF_Proj/docs/reports/_bk/20261006_191613Z_HONF_Response_Operator_Consolidation_and_Maturation_Report.md`: development R-direct2500/R-group2400 identities, actual source-resolved near/far semantics, rank-one response limitations and operator-residual supervision.

[E3] Code at10193ab: `src/honf_forward_core/interface_fields/source_response_operator.py` (`prepare_context`, `_prepare_groups`, `prepare_receivers`, `apply_forcing`, `apply_increment`); `Case_ThermalChannel/src/channelthermal/source_response.py` (`context_tensors`, `prepare_native`, `SourceResponseThermalModel.prepare_native/apply_native`); `geometry_flow_field.py`; `thermal_source_response_fit.py`; and the native Wind `model.py`.

[E4] `AGENTS.md`, `docs/guides/Thermal_Model_Development_Protocol.md`, and the existing shared-core architecture map. These establish repository procedure, not experimental success for the new design.

The flow-context hypothesis, nonlinear-head generalization, packet-amortization design, resource allocations and promotion thresholds in this document are new proposals. They are not findings claimed by the reviewed report, and no external paper is being substituted for project-specific validation.
