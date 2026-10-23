# HONF: consolidate the forward interface around learned physical responses

## 0. Decision and executable scope

**Reviewed branch:** `agent/honf-core-next` at `25e28993975d8cea79a24b3d14f9e9e9658eff45`.

**Central goal:** learn an organization of module–environment interactions through multi-field reconstruction; reuse that organization to generate and evaluate modular designs. An accurate predictor, an active gate, and a useful organization are different accomplishments.

**This round's main deliverable is a trained, reusable source-response operator, not another diagnostic-only report.** Keep the dependency separation, source identity, prepared execution, and numerical lessons already established. Stop introducing new control terms into the old port-feedback stack as the default development strategy.

The work has one main model family and one inexpensive supporting continuation:

1. Continue the existing **D-sep flow reader** from its literal epoch-1000 state to an explicitly identified epoch-2500 child, with a new, declared learning-rate schedule. Do not build another flow architecture in this round. D-open1000 remains a frozen dependency control; do not retrain it merely to repeat an already answered null question.
2. Implement and train **one whole-layout-conditioned thermal source-response family**, with two independently trained readouts: **R-direct** and **R-group**. They share source/environment encoding, near-source resolution, data, objectives, and training budgets. R-group is the HONF hypothesis; R-direct is the necessary strong control. Their differing far-read parameterizations must be disclosed.
3. Learn actual multi-field predictions and measurable heat responses, targeting **2,500 new thermal-fit epochs per arm** within the resource limit. Reviews are for catching invalid experiments and judging learning, not automatically stopping every candidate that has not beaten a mature model at epoch100.
4. Export the actual response operator and test whether its learned grouping preserves useful response directions. Use prepared operator applications on existing heat allocations; do not launch a generative inverse campaign or new physical-reference solve.

Run3801's optimized thermal predictor remains the incumbent and a protected rollback. The new response backend is a genuine modeling change, **not an exact conversion of Run3801**. Never claim the new backend inherits its field accuracy without measuring it. No existing checkpoint, old Tree implementation, mature reference, or formal history is deleted or overwritten.

**The solver ledger remains 326/326.** Residual evaluation against an already specified discrete operator, when qualified below, is a training constraint—not a generated reference solution. It must never call a solver, factorize/invert the physical operator, integrate temperature in time, or create purported newly solved fields.

## 1. What is now settled, and what is not

### 1.1 Preserve these results without reopening the entire campaign

The dependency study [R1] establishes, on its measured inputs:

- D-sep's actual flow inputs, prepared states, finite heat changes, and heat derivatives are independent of heating. This is an imposed, source-backed ThermalChannel capability, not a learned hyperedge.
- The retained thermal model is unchanged. Its equal-case fluid/surface/material temperature RMSEs remain **0.770312 / 0.788326 / 0.663080** on the22 exposed validation cases.
- Optimized high-M Q8192 H-add inference is **0.128393s**, versus **0.134602s** for D-sep composition. Heat/query forward+VJP is **0.270665s / 0.279851s**. The extra flow reader is now a small complete-call cost.
- Lazy-I, within-read access reuse, and qualified tile512 substantially reduce implementation overhead. Thermal fine-source work remains dense; this is not a sparse-executor result.
- Selected D-sep1000 has u/v/p/omega RMSE **0.0282855 / 0.00314433 / 0.0126933 / 0.177467**, compared with retained flow **0.0220367 / 0.00307966 / 0.0122219 / 0.104567**. v/p pass the previously declared mean/p90 targets; u/omega do not.

Do not label the unchanged0291 thermal error as a new failure of flow-only training. It was deliberately outside that optimizer. Conversely, do not call the imposed zero heat-to-flow derivative a newly learned organization.

### 1.2 The principal unresolved modeling issue

The audited generator has fixed-coefficient thermal transport at fixed geometry. Its point-temperature update is affine in heating at a common integration step. Saved convergence stopping and FP32 storage qualify exact endpoint affinity [R2]. Existing fluid-temperature central-closure errors are approximately **3.10e-7–1.58e-5** in the reference versus **1.84e-4–0.130814** for retained Add500 [R1,R2].

The existing learned thermal path can therefore introduce substantial artificial heat curvature. More importantly, it still has incorrect thermal slopes:0291 reference minus/plus mean changes are **-0.02891885 / +0.02891378**, while Add500 predicts **+0.01042027 / -0.01550809**.

Eliminating artificial curvature does **not** guarantee the correct slope. Both must be measured separately. A linear model with the wrong kernel can still give a wrong design decision.

### 1.3 The current excitation limitation is real

Each complete response family contains one balanced heat direction and its opposite. Those endpoints span one direction, not two. The available fixed-total dimension is M-1:2,4,6,9 for M3,M5,M7,M10. More receivers, more epochs, and interpolated heat states do not increase the rank of independently solved excitations [R2].

Shared learning over layouts and source-backed operator constraints can provide additional inductive information. Neither turns existing directions into new independently solved data. Individual source kernels and hyperedge memberships remain estimates whose validation is restricted to measured directions until additional evidence exists.

## 2. Scope control: converge on a family rather than start a portfolio

Use three scientific fit identities at most: D-sep continuation, R-direct, R-group. A short second seed is **not** automatic; reserve that for a later confirmation after a meaningful result. Existing H-add, H-joint, Tensor-H, Dense-D25, and D-open are read-only references, not additional fitting arms.

The common thermal family, loss declarations, sampler, and horizon must be fixed before the main100-epoch review. Codex may repair implementation problems and make one bounded training-only design adjustment before sealing the main pair. Record the original problem, change, reason, cost, and new lineage. Do not hot-edit a running model or silently change architecture/objectives at epoch500.

Small improvements are allowed to accumulate within the same family. Do not introduce a new name and rebuild the model every time a role misses a target. Separate:

- **debugging:** wrong units, joins, gradients, masks, or implementation;
- **learning:** valid model still fitting;
- **model limitation:** mature valid comparison shows a persistent limitation;
- **evidence limitation:** no independent response data in a needed direction.

A failed organizer comparison does not erase a successful source-response predictor. A failed flow target must not prevent the thermal pair from being trained. A completed numerical check must not be rerun at every checkpoint unless its operator changes.

## 3. Resource, dataset, and evidence contract

### 3.1 Development remains separate from formal training

Use `fixed25_v1`, semantic fingerprint:

`933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`.

Primary membership remains **150 TRAIN /22 exposed validation**. Preserve selected-training normalization, original source split, duplicate-family exclusion, module identities, and fixed representative cases0277/0291/0294/0687. Do not add full-data cases to the primary fit or quietly enlarge the validation population.

Keep the already declared response-fit addendum: original TRAIN0001/0318/0333/0348. Only0348 overlaps primary TRAIN, so disclose150 primary cases plus three auxiliary families. Response-development0304/0320/0335/0350 and the fixed audit remain excluded from response fitting, coefficient calibration, and operator-residual construction. Their historical exposure remains disclosed.

No primary/full-data identity is interchangeable. Formal3501/3502, mature1804, and all prior scientific lineages remain untouched. No automatic formal launch, restart, or full-data evaluation is authorized.

### 3.2 Computation budget

Use physical GPUs1/2, verifying UUID-to-logical-device mapping. Never terminate or alter unrelated jobs. When another process occupies an authorized GPU, use the other GPU, serialize work, or continue under documented contention. Do not wait indefinitely for exclusive access.

Hard round ceiling: **12 aggregate GPU-associated process-hours and8 elapsed hours**, including failures, tests, training, evaluation, and closeout. Reserve the last45minutes for actual closeout. These are ceilings, not goals to consume.

Priority allocation:

| Work | Planned maximum share of round effort |
|---|---:|
| Reuse/implementation and source-qualified extraction/residual checks | ~20% |
| Main thermal paired learning | at least50% when executable |
| D-sep continuation and its endpoint checks | at most10% |
| Scientific evaluation, actual cost, figures, report | ~20% |

Measure first10 complete epochs of the new thermal pair, including scheduled objectives, before forecasting2500. Do not extrapolate its cost from the0.15-second **flow-only** epochs. If2500 cannot fit, optimize data staging/chunking first, then predeclare the largest matched horizon among1000/1500/2000/2500 that fits. Preserve a valid mature comparison instead of missing all terminal evaluations. Do not replace training with hours of inventories.

### 3.3 Monitoring and selection

Save and plot every100 epochs, plus an explicit final stop. Retain declared milestones, latest, and one best-field alias. Use the existing atomic writer/trusted loader and stop-request mechanism. No25-epoch archives or duplicate selector copies.

Cheap fixedQ1024 monitoring and all22 scalar statistics follow the100 cadence. Full native-grid all22 evaluations occur at100,500,1000, and final horizon; reuse identical endpoint/selected evaluations. Detailed field/operator exports default to the fixed four representatives at the final selected state. Midcourse heavy figures are not required.

Every reported epoch includes actual visits, ordinary optimizer updates, reconstruction queries, auxiliary/operator rows, learning rate, and timed scopes. Candidate best selection uses a common predeclared thermal reconstruction score; response-development outcomes do not select checkpoints. Report matched endpoints and selected checkpoints separately.

## 4. Supporting work: mature D-sep without reopening the dependency question

### 4.1 The actual scheduling issue

`tools/thermal_dependency_flow_fit.py` uses `CosineAnnealingLR(T_max=1000)` without a positive `eta_min`, so the declared endpoint reaches zero learning rate [C2]. Its CLI and resume identity also restrict the old horizon. Merely changing `--epochs` is not a scientifically valid continuation. Late flattening does not, by itself, prove representational saturation.

Create an explicit **schedule-continuation child** from the literal D-sep1000 checkpoint. Preserve its parameters, optimizer moments, primary membership, normalization, sampling formula, and architecture. Preserve the original completed run. Store the parent digest and the new schedule in the child identity; do not weaken historical resume guards.

Default additional1500-epoch schedule:

- additional1–20: linear warm-up from1e-6 to1e-4;
- additional21–500: learning rate1e-4;
- additional501–1500: cosine decay from1e-4 to1e-6.

This is a proposed optimization schedule, not an established optimal one. Record it before new outcomes. The child final absolute age is2500, with1500 genuinely new epochs,6000 new ordinary updates,225000 new case visits, and230.4M new primary queries if completed. Resume within this child preserves its new scheduler.

D-open1000 remains a frozen scientific comparator with a shorter age. Do not claim a new age-matched heat-policy accuracy result against it. The already measured finite null contrast answers the dependency question.

### 4.2 What to learn from the continuation

At1500 and2500 absolute age, report all22 flow means/p90, near/far u and omega, pressure functionals, and TRAIN/DEV curves with actual learning rate. Use the existing5% mean/10% p90 replacement targets against Run3801; do not loosen them retrospectively.

Inspect whether remaining error concentrates near modules and whether training error remains large. The current source read uses domain-scaled Fourier features and a small shared latent reduction [C1]. Insufficient near-source spatial representation is a hypothesis, not a proven defect. No extra multiscale-flow architecture or curl constraint is authorized in this round.

Do not impose divergence-free flow or replace omega with an automatic-differentiation curl without matching the generator's discrete gradient/masking convention. The inspected labels are unprojected analytic-wake fields, not a Navier–Stokes constraint dataset [R2].

If D-sep still misses u/omega, retain it as the dependency-correct candidate and preserve the more accurate incumbent for labeled reference use. Do not silently route candidate u/omega back to heat-sensitive parent predictions. Continue the main thermal experiment independently.

## 5. Main mathematical design: learn an operator, not another heat-dependent gain

### 5.1 Symbols and dependency

Let C contain the complete layout, module shapes/material descriptors, prescribed operating context, environmental coordinates/coefficients/boundary metadata, and source measures. Let h be the physical vector of active-module heating amplitudes. Let r be a receiver with physical coordinates and a declared output role.

The reusable core prepares geometry/context-dependent information:

\[
Z(C)=\operatorname{Encode}(C),\qquad
K_\theta(r,i\mid C)=\operatorname{ReadKernel}(r,i,Z(C)).
\]

For the qualified ThermalChannel forcing law:

\[
\widehat T_\theta(r;C,h)
=b_\theta(r;C)+\sum_{i=1}^{M}K_\theta(r,i\mid C)h_i.
\]

C, Z, K, and all grouping quantities are independent of the **current heat allocation**. Their geometric dependence is nonlinear and includes all modules/environment information, not just source i in isolation.

Use physical heat or heat divided by one frozen training scale. Do not introduce case-max scaling, heat-conditioned normalization, current-heat global statistics, effective thermal ports, or heat-conditioned source tokens into K. If centered heat is used for numerical scaling, explicitly transform the offset so the physical operator stays affine; uncentered fixed scaling is simpler.

The active dataset has zero inlet/wall temperatures. If the local source check also confirms the zero-forcing initial/background state, set b=0 for this capability. Otherwise keep an explicit heat-independent offset. Do not infer a physical zero-temperature solution merely from a normalized target mean. The generic core supports a nonzero offset/other boundary forcing.

### 5.2 Why this directly supports inverse reuse

For any fixed C:

\[
\Delta\widehat T(r)=K_\theta(r,\cdot\mid C)\,\Delta h,
\quad
\frac{\partial\widehat T(r)}{\partial h_i}=K_\theta(r,i\mid C),
\quad
\frac{\partial^2\widehat T(r)}{\partial h_i\partial h_j}=0.
\]

Thus the exported influence is the actual predictor derivative, not a membership picture whose relationship to the output is unknown. Finite thermal responses are trained with KΔh directly, without subtracting two large independently computed network temperatures.

The model can still get K wrong. Exact affine algebra is a software/model-class property; physical response fidelity requires comparisons with stored physical differences. Geometry derivatives remain nontrivial because both K and any offset depend on C. Do not freeze them for design work.

### 5.3 Whole-layout and environment information must remain present

Reuse the source-resolved preparation pattern of `GeometryFlowField` and maintained typed M/E utilities, not its Thermal-specific input widths/output count verbatim.

Prepare individually indexed module states and environment states. Environmental inputs here must be prescribed or geometry-derived context, not current-heat-dependent predicted temperatures or effective ports. Use shared, batched nonlinear source updates with masks, physical measures, and signed relative geometry. Include domain-scaled and source-characteristic-length-scaled receiver geometry in both readouts. Keep source i identifiable until its kernel contribution is constructed.

The environment need not be a set of independently tunable heat sources. In this benchmark its tokens encode geometry, boundary conditions and material/transport context that shape K. Label **environmental context donors** separately from **independently actuated module heating sources**. Do not claim192 environmental degrees of thermal actuation.

Source count must be variable. Generic contracts should accept2-D/3-D coordinates, typed sources/receivers, measures, and arbitrary output channels. Dataset adapters own units, radius conventions, linearity declarations and field extraction. Maintain old APIs unchanged by making the new family opt-in.

### 5.4 One common family, two far-read choices

Both arms have the same contextual encoder and a common near-source kernel branch. A smooth, geometry-only weight w(r,i) protects fine local interactions. Default support uses a characteristic source length: weight1 inside2 source radii, smoothly reaches0 at4 radii. This is a declared modeling choice, not a discovered physical interaction cutoff. Material receivers include their own physical module. Other geometry types supply their own characteristic lengths through the adapter.

**R-direct** evaluates a nonlinear far coefficient separately for each source:

\[
K^{D}(r,i)=w_{ri}k^{near}(r,i)+(1-w_{ri})k^{far,D}(r,i;Z(C)).
\]

**R-group** learns source-anchored collective response functions. For each valid group e:

\[
b^M_{ei}\ge0,\qquad \sum_i b^M_{ei}=1,
\]

with separate measure-normalized environment context membership. Its group state is formed from nonlinear source/environment features, anchor geometry and complete prescribed context. It does not read h.

The signed receiver coefficient a_e(r;C) carries output-response units, and

\[
K^{far,H}(r,i)=\sum_e a_e(r;C)b^M_{ei}(C),
\]

\[
K^{H}(r,i)=w_{ri}k^{near}(r,i)+(1-w_{ri})K^{far,H}(r,i).
\]

A group therefore represents a shared **source-to-receiver response function**, with physical source membership and measurable consequences. a is not an attention probability or physical energy. b is a membership/forcing weight, not a conservation certificate.

An equivalent efficient read is

\[
m_e(h)=\sum_i b^M_{ei}h_i,
\]

\[
\widehat T^H(r)=b(r)+\sum_e a_e(r)m_e(h)
+\sum_{i:w_{ri}>0}w_{ri}\left[k^{near}(r,i)-K^{far,H}(r,i)\right]h_i.
\]

The near correction prevents double counting. Its far subtraction must be included in both implementation and any later compression bound.

This pools a **linear forcing** after preparing nonlinear geometry/context coefficients. It is not the old operation of averaging nonlinear physical source states before discovering which source information matters. Nevertheless, far factorization is an approximation/model restriction relative to R-direct, not a lossless algebraic conversion between independently trained models.

Use one valid anchor per active module plus at most one background proposal, with padded capacity derived from the configured maximum source count. This is allocated capacity, **not learned K**. Use smooth learning with no forced K target, hard top-k curriculum, diversity loss, recursive split/merge solver, or full-wrapper soft shadow. Start with all valid modes available; qualify functional redundancy only after the response operator has learned.

### 5.5 Identifiability and interpretation limits

Even with normalized/anchored b, multiple factorizations may represent almost the same K. Report stable source-resolved K and KΔh as the primary interpretation; group IDs are representation-local. Do not promise a unique causal graph from factorization.

K_i(C) can change when another obstacle moves. Linear heat response is compatible with many-body dependence on geometry and environment. Zero heat on a module leaves the obstacle in C; removing a module changes C. These are different interventions and must have separate tests and figure labels.

## 6. Thermal roles and ports: one consistent response representation

### 6.1 Do not route the new linear response through the old nonlinear thermal loop

The new backend must not send h-dependent predictions through inherited port-refinement heads or Stage-A and then call the resulting output heat-affine. That would recreate the nonlinear dependency being repaired.

For this benchmark, the source audit confirms that surface/material/outside temperature are extracted from a shared thermal grid, not outputs of an independently solved Robin interface [R2]. The new model therefore predicts a common temperature-response representation with case-owned role extraction. Retain the old local surrogate/port stack as an untouched reference and as a reusable component for domains that genuinely need that coupling. Do not delete it or assert that local operators are generally unnecessary.

### 6.2 Native extraction contract

The Thermal adapter maps each requested fluid, material, boundary, or outside-temperature receiver to its actual native physical coordinates and original interpolation/extraction rule. Prefer reuse of existing geometry-only extraction helpers.

Where a label is sampled by interpolation, apply the same interpolation weights to predicted kernel values at its stencil coordinates. This keeps extraction linear and preserves the relation ΔT=KΔh. It is not interpolation of hidden reference fields at inference. Keep material IDs, local-to-world transforms, masks and receiver weights exact.

For the stored q-normal proxy, use the audited sign, conductivity convention, outside offset and native sampling operator:

\[
K^{q}_{i}=-\frac{k_{interface}}{\delta}
\bigl(K^{outside}_{i}-K^{surface}_{i}\bigr).
\]

Do not replace this with an unrelated AD-normal derivative and retain the old target name. Derive h-proxy/h-effective with their actual epsilon, clipping and validity rules. Such ratios need not be affine in heat. Compute material maxima **after** reconstructing the material-temperature field, not as a linear sum of per-source maxima.

If exact role extraction cannot be recovered within the bounded implementation effort, record the unsupported role and complete the supported temperature experiment. Do not silently change targets, invent port values, or fabricate an all24 pass. The main kernel must still receive fluid, surface and material supervision before claiming multi-field readiness.

For original-domain replay, return the same physically defined terminal quantities through an explicit new adapter. Do not silently pass the new backend through a legacy loader that assumes an embedded Stage-A or P0/P1 state. Unsupported historical diagnostics should fail clearly or return a declared not-applicable status.

The new backend has no physical P0/P1 refinement trajectory. Mark historical initial-port metrics as not applicable, not zero error. Final/outside/derived port quantities remain comparable when they have the same physical definition. An evaluator adapter may not fake a two-stage port history for compatibility.

### 6.3 Cost contract

The candidate should run its new prepared response core plus the chosen D-sep flow reader. It must not secretly execute Run3801 as a dense fallback or evaluate a complete old thermal wrapper just to recover omitted outputs. Keep unsupported capabilities explicit.

The thermal coefficients may be prepared once per exact C and requested receiver catalogue, then applied to multiple h vectors. Geometry/context changes rebuild the operator. This legitimate amortization is different from reusing a wrong previous physical state.

A frozen autograd-bearing preparation is valid only within the owning graph/request. Do not persist detached geometry-dependent tensors and then claim complete geometry derivatives.

## 7. Supervision: use reconstruction, measured responses, and qualified operator constraints

### 7.1 Common reconstruction and response losses

Both arms use the same loss declarations and exact data streams. Reconstruct native fluid T, surface T, material T and supported outside/flux roles. Give roles and cases explicit weights so many material points or high-M cases do not silently dominate.

Use TRAIN-only output scales. A default primary thermal reconstruction objective averages the separately standardized fluid/surface/material temperature MSEs; q-proxy is a separately reported lower-weight auxiliary, not a hidden replacement for temperature fidelity.

Fit saved positive responses from the four declared TRAIN families:

\[
L_{response}=\mathbb E_{C,d,r}
\left\|\frac{K_\theta(r,\cdot\mid C)d-\Delta T^{ref}(r)}{s_{response}}\right\|^2.
\]

Use original common masks, material identities, role quadrature and widened saved endpoint differences. Opposite signs remain correlated examples from one direction. Do not average away signed spatial residuals or evaluate only peak ordering.

Interpolation or linear combinations of existing TRAIN response states, if used, must be tagged as derived constraints in the same solved span; they do not add independent directions or physical validation. Do not construct such training targets from DEV or fixed-audit states.

Do not distill unmeasured heat perturbations from H-add/Tensor-H as physical truth. Those models have exactly the response defects under investigation. Existing teacher predictions may appear in read-only reference plots only in this primary pair.

### 7.2 Address missing excitation without pretending more epochs supply labels

A useful optional constraint is available from this particular audited generator. At fixed C, its discrete temperature balance can be written

\[
A_C T=B_C h+b_C.
\]

For a learned response kernel and heat-independent offset this suggests

\[
A_C K_\theta\approx B_C,
\qquad
A_C b_\theta\approx b_C.
\]

These equations constrain source directions that have not been individually solved. They add **known-operator information**, not new empirical response labels or independent validation.

Implement this only if the exact active discrete operator can be qualified from existing source and TRAIN records. The constructor belongs to the Thermal training adapter, not the generic HONF core. It may use already stored TRAIN velocity fields as supervision-side coefficients; these fields must never enter the inference encoder, prepared state or organizer.

Allowed work is assembling the fixed local stencil, applying it to stored TRAIN fields for validation, and evaluating residuals of neural kernel predictions. Forbidden work is solving/factorizing A, calling the reference adapter, time integration, iterative physical rollouts, creating new converged states, or labeling outputs of a teacher model as source truth. Log operator applications and neural query work separately from the unchanged326/326 reference ledger.

Qualification, bounded to60minutes and at most three genuinely different implementation remedies:

1. Match fixed coefficients, upwind signs, fluid/solid alpha, heat deposition, boundaries, outlet and grid indexing to the inspected source.
2. Evaluate the residual on already stored TRAIN fields and finite responses. Report its size relative to the forcing and solver-stopping indicators. It is not a mesh-error certificate.
3. Check a source-zero/constant-boundary algebra example and the exact discrete adjoint of the implemented stencil in a software test.
4. If the stored target residual is not consistent with the assumed operator, stop this constraint—not the entire thermal fit. Do not enlarge tolerances or substitute a continuum PDE to force a pass.

Once qualified, apply sampled stencil residuals to **all active source columns** of K where affordable. The same network-produced kernel block supports these columns; this is not M separate physical solves. A typical budget is128 input-selected stencil rows per primary case/epoch, stratified over fluid, solid and boundary/interface neighborhoods, with required neighboring coordinates deduplicated. Do not apply the fluid-only mask to a shared fluid/solid thermal operator.

Use a row-normalized operator residual and TRAIN-only scales. Calibrate shared response/operator coefficients once from pooled TRAIN gradient measurements of both arms, with a predeclared finite cap and the measured values saved. A practical target is that each added term initially has at most half the reconstruction gradient norm; this is an initialization rule, not a perpetual guarantee. Log component norms and gradient conflict at100/500/1000/final without searching DEV coefficients.

Decide `operator_constraint=qualified` or `disabled_with_reason` **before** the main pair is sealed. Use the same decision in both arms. If disabled, continue training on real reconstruction and measured responses and report the unresolved rank limitation. Do not substitute another model family.

### 7.3 Why this remains a learning experiment

The response operator is fitted from multi-field observations plus explicitly declared physics constraints. No matrix inverse or stored per-case kernel supplies inference. The network must predict K from geometry/environment in a case it is given.

Small residuals on sampled TRAIN stencils do not certify reference fidelity or unseen-layout response accuracy. Existing response-development and fixed-audit comparisons remain required. Do not equate the number of residual right-hand sides with the rank of independently measured heat directions.

## 8. Coding design and interoperability

### 8.1 Reuse existing code without changing historical semantics

Relevant reviewed code [C1–C5]:

| Existing component | Next disposition |
|---|---|
| `geometry_flow_field.py` | Reuse source-resolved geometry preparation ideas; keep existing D-sep/D-open state keys unchanged. |
| `channelthermal/dependency_flow.py` | Preserve current composition, strict parent loading and heat-batch guards; new response composition gets a distinct capability. |
| `tensor_query_interaction.py` | Preserve lazy-I/access reuse and old checkpoints; no new H-add/H-joint correction term. |
| `thermal_dependency_flow_fit.py` | Add explicit schedule-child preparation; never weaken old resume identity. |
| Existing data, response atlas, native role query, checkpoint and stop helpers | Reuse directly where semantics match; do not build another provenance framework. |
| Old Stage-A/P0/P1/P2 thermal wrapper | Read-only incumbent and supported historical path, not an obligatory runtime dependency of the new backend. |

New names below are **proposed interfaces**, not claims that files already exist. Place maintainable code under the shared core and case adapter; keep one-time analyses ignored.

Suggested shared objects:

```python
PreparedResponseContext        # context/source IDs, measures, geometry; no current h
PreparedSourceResponse         # kernel/factor blocks with exact receiver IDs
SourceResponseOperator         # direct or grouped readout
ResponseApplication            # fields, contributions, declared approximation receipt
```

Suggested methods:

```python
prepare_context(case_inputs) -> PreparedResponseContext
prepare_receivers(context, typed_queries) -> PreparedSourceResponse
apply_forcing(response, forcing) -> role_outputs
apply_increment(response, delta_forcing) -> role_increments
export_response_operator(response) -> source_resolved_blocks
export_organization(response) -> memberships, receiver_functions, near_terms, ancestry
```

Use explicit tensor shapes, masks, typing, validation and informative errors. Avoid hidden global caches and case-ID embeddings. Store only necessary source/query blocks; stream large Q. New core types must not import ThermalChannel, its generator, or WindFarm.

### 8.2 Required regression checks

Preserve the current passing dependency/executor tests. Add focused tests for:

- current-heat independence of prepared C/Z/K/B/a, and live heat derivatives of applied thermal fields;
- finite affine superposition and AD identity with K, in physical units and with fixed normalization;
- module permutation, padding, variable M, absent source types and unequal environment-measure splitting;
- whole-layout sensitivity: moving another module changes the evaluated kernel when weights permit it; zero heating and removing geometry remain distinct;
- query order/chunk/subset consistency; no current query minibatch defines a global reference measure;
- direct versus efficient grouped application equivalence, including near subtraction and input gradients;
- common physical role extraction, native interpolation and q-proxy sign/units;
- prepared/cold agreement, fixed-layout heat-batch agreement, stale-context rejection;
- read-only truth/target poisoning tests and no generator import/call at inference;
- explicit schedule-child optimizer state and checkpoint round trip;
-2-D/3-D and alternative nonlinear forcing-law API conformance with labeled synthetic tests.

Retain strict q-proxy failures under old tolerances. A new model's prediction error must not be confused with a same-weight arithmetic comparison. Use deterministic repeat checks to localize nondeterminism, not to declare empirical repeat variability a physical noise floor.

### 8.3 Minimal cross-dataset contract

The generic decomposition is an input-conditioned response interface. Thermal may declare affine forcing. Another dataset may use a nonlinear forcing/constitutive map or a local tangent operator with an explicit validity neighborhood. Geometry changes always rebuild context.

No claim that WindFarm velocity is globally affine in turbine position, that all environments have192 sources, or that every problem has a heat-independent flow lane is permitted. Smoke-test existing Wind loading/read-only APIs after shared changes; do not train Wind or claim transferred accuracy this round.

## 9. Learning plan: give the actual representation time to learn

### 9.1 Seal one paired recipe

R-direct/R-group start from identical common encoder/near-branch tensors and the same fixed seed; document differing far-head shapes and parameter counts. Aim for active parameter counts within10% by a once-chosen width adjustment before results. If exact matching compromises a valid architecture, report the difference rather than adding unused parameters.

Both use the same physical input streams, reconstruction receivers, auxiliary TRAIN families, operator-residual rows, normalization, optimizer and update counts. No candidate-specific outcome-selected samples.

Recommended common optimizer: AdamW3e-4, weight decay1e-5, clipping1.0, with learning rate held at3e-4 through1000 and then cosine-decayed to3e-6 by2500. A brief declared startup warm-up is permissible. The full horizon and positive floor are recorded before fitting. Do not infer convergence just because a planned schedule has reached zero.

Each development epoch visits all150 primary cases. Q1024 fluid reconstruction remains the primary comparable budget; record additional material/interface/interpolation/operator queries separately. Use microbatch8/effective48 when memory permits. If both require a smaller microbatch, retain the effective batch and correct per-case accumulation denominators.

For expensive material queries, sample a fixed-budget set of actual material-local points per active module, never a invented coarser physical target. Final evaluation uses complete stored material support. Sampling counts are not loss weights.

### 9.2 Reviews and default continuation

At100: verify real learning, full role joins, field maps, finite gradients, nonzero organizer training signals, operator parity and cost forecast. A healthy arm need not already beat Run3801. Continue the paired family.

At500: inspect TRAIN/DEV reconstruction and response curves, signed response patterns and full all22 physical statistics. Continue to1000 unless there is persistent numerical invalidity, severe nonrecovering training divergence, an unauthorized dependency, or the cost forecast cannot fit.

At1000: continue toward the declared2500 target when genuine fitting or response improvement remains, or when the schedule has not yet supplied the predeclared low-rate maturation. Avoid stopping solely because one tail metric misses a mature reference.

A scientific early stop before2500 requires a documented plateau over at least two separated100-epoch windows with a non-negligible learning rate, no useful TRAIN or DEV response improvement, and no unresolved implementation explanation. Report the exact stopping scope; do not call the entire hypothesis disproved. A resource stop is labeled resource-limited, not converged.

If code repairs change the mathematical model after sealing, create one explicit replacement lineage and restart the affected comparison from an agreed common state. Do not splice curves from different operators into one history. No additional alternative portfolio follows an unfavorable review.

### 9.3 Minimal training deliverables

Required scientific deliverable is the actually fitted pair and its terminal evaluation—not just successful unit tests, preparation commands or an untrained graph. Prefer completing a valid1000-epoch matched pair with all measured results over starting2500 without a closeout budget.

Do not spend the remaining budget on repeated failed source-audit calls. After at most three distinct remedies per blocker, isolate the unavailable optional piece and proceed with unaffected learning. Report only consequential hard blocks in the executive summary.

## 10. Evaluation: three enduring targets

### 10.1 Predictor and physical response

Use all22 native cases with unchanged role definitions, equal-case means/p90/max, M strata, and near/far errors. Temperature reconstruction comparison is against Run3801; response comparison includes retained Tensor-H as well as H-add and the matched direct arm. Do not show only the weaker parent.

Prospective targets, fixed before new results:

| Dimension | Development target / interpretation |
|---|---|
| Thermal reconstruction | Fluid/surface/material means within10% and p90 within15% of Run3801, with any improved response tradeoff disclosed. These are development tolerances, not certified physical bounds. |
| Response knowledge | Improve response-withheld thermal errors materially, targeting at least15% versus H-add and no worse than the stronger Tensor-H aggregate on the same available physical panel. Report every role/sign rather than only the aggregate. |
| Wrong-direction sentinel | Correct both0291 mean response signs after sealed selection, with amplitude error reported; mere sign correctness at near-zero amplitude is insufficient. This exposed case is a diagnostic, not an independent generalization test. |
| Structural identity | Affine finite/AD/kernel identities pass numerical tests; source/context/kernel remain heat-independent. |
| Flow | D-sep retains the earlier5% mean/10% p90 prospective replacement targets; no parent-flow fallback. |
| Cold inference | Target complete Q8192 no slower than optimized H-add by more than20%; price all declared output roles. |

Do not block valid maturation at100 merely because these final targets are not yet met. Do not relax them after seeing final results.

Central closure is reported in native units next to reference closure. Its near-zero value is built into the model class and cannot replace the signed response tests. For tiny reference changes, keep amplitude errors and numerical qualifications rather than manufacture a sign threshold. No near-zero response should be divided by an arbitrary tiny denominator and used as a dramatic score.

### 10.2 Organizer value and faithful information flow

Always report separately:

- R-group versus independently trained R-direct at matched exposure;
- fixed-weight changes of group membership/receiver functions;
- actual source-resolved operator change and actual output/response change;
- context/near/far routes and physical versus latent support;
- allocated groups, nonzero response modes, receiver participation and execution.

Use input-only representative choice. Plot K(r,i) for several named physical donors, their actual contributions K(r,i)h_i, and K(r,i)Δh_i for stored interventions. Include unchanged-own-heat receivers. This is more informative than a source-density heatmap alone.

Bounded same-weight controls: replace learned source membership with a degree/measure-matched geometry assignment, or remove one response mode while preserving all other terms. Recompute the exact near subtraction. Mode permutations with jointly permuted factors are invariance tests, not interventions. No intervention need be called causal physics.

A meaningful group-added-value claim needs either a material response/generalization benefit over R-direct (prospective target roughly5% across multiple families without violating field tolerances), or materially more economical faithful operator reuse at matched distortion. A nonzero gradient or0.3% ablation effect alone is insufficient.

### 10.3 Reuse before a generative campaign

Expose K at observed and held receivers. For the same saved heat differences, compare:

1. cold candidate field differences;
2. prepared KΔh;
3. optional compressed response application;
4. stored physical finite differences.

Measure the error of1/2/3 against4 separately from implementation equality between1/2. Include existing finite-pool rankings and biased absolute peaks; do not generate additional designs or refit a denoiser.

The output should be an inverse-ready **software interface with a measured validity boundary**, not a claim that continuous physical inverse design is solved. A small observed-versus-held sensor operator comparison may be added without optimization or new reference calls. Do not build a new inverse model simply to demonstrate that the API can be called.

## 11. Adaptive K: response-aware reuse, not another training pressure

After the full R-group kernel is trained and selected, evaluate an optional adaptive **response-use** view. It is not an automatic sparse forward deployment.

For a fixed layout and fixed-total heat-change set D, define the model's group source functional

\[
m_e(d)=\sum_i b^M_{ei}d_i.
\]

For a balanced box with |d_i|≤r_i and sum(d_i)=0, any scalar beta gives

\[
|m_e(d)|\le\sum_i|b^M_{ei}-\beta|r_i.
\]

The bound is conservative; choose beta from source weights only, never target outcomes. It shows why a uniform common-mode source functional can be irrelevant for balanced *changes* even while it matters to absolute temperature.

With the near correction in section5, the complete contribution of dropping far mode e is

\[
D_e(r,d)=a_e(r)\sum_i(1-w_{ri})b^M_{ei}d_i.
\]

Use these **actual effective weights**, not the simpler far-only weights, when bounding omission. Define per-receiver upper bounds epsilon_e(r) on |D_e| from the declared feasible change radii. Retain enough modes that the sum of omitted bounds fits a predeclared output-distortion budget. This produces case/receiver/task-dependent K without an online physical solve or an arbitrary target K histogram.

Use a small predeclared distortion set, for example0,0.5%,1%,2% of TRAIN response RMS. Zero is the exact reference. These are model-approximation budgets, not true-physics error guarantees. The task bound affects inverse reuse only; the trained organizer and reconstruction predictor do not receive the held inverse target.

Report the triangle decomposition:

\[
\|\Delta T^{ref}-\Delta\widehat T_{compressed}\|
\le
\|\Delta T^{ref}-K d\|+
\|K d-\Delta\widehat T_{compressed}\|.
\]

No compression can repair the first term. Preserve absolute baseline prediction when compressing an increment; do not drop a mode from absolute-field computation because it is null for one fixed-total task.

Compare this learned factor reuse with a standard low-rank approximation of the same direct model's prepared response matrix at the same receiver/forcing domain. Price formation/storage/application. Such an algebraic control is model-only and supplies no physical labels. If ordinary low-rank compression is equally effective, do not attribute the benefit uniquely to learned hyperedges.

Compute the near branch's work and any coefficient preparation cost. A smaller K without reduced total arithmetic/latency is only representation compression. Smooth full-operator differentiation remains the reference near selection thresholds; report changes of the selected view rather than claiming global differentiability through discrete omission.

If no mode is redundant at the specified budget, keep it. K=full is an honest outcome. Do not force more sparsity or launch another selector-training campaign.

## 12. Required figures and plain-language final report

The first page must answer, in ordinary language:

1. **Predictor:** Can the retained or candidate model now reconstruct and predict changes accurately enough, and at what complete-call cost?
2. **Organizer:** What do its groups actually represent, and what measurable benefit do they add beyond the direct response model?
3. **Inverse system:** What can be reused exactly, what has physical evidence, and what remains unsafe or untested?

Then state A/B/C explicitly: added value beyond a competent predictor; interpretation matches actual information flow; response knowledge transfers beyond fitted neighborhoods.

Keep the executive evidence to one compact table and a few decisive numbers. Put all24/role-tail/individual-array details in appendices. Do not let dozens of qualified metrics conceal the main verdict.

Required inspected visual groups, rendered from saved actual arrays:

- learning curves versus new epochs, actual updates, learning rate and measured time; matched direct/grouped histories;
- native fluid/material/surface fields and signed residuals on the four representatives;
- donor-specific K maps, executed near/far/group contributions, environment-context donors, and matched geometry changes;
- physical finite responses for both signs, central closure, unchanged-own-heat receivers, and reference-versus-predicted module peaks;
- complete cold-call and prepared multiple-forcing cost versus fidelity, including all preparation and near work;
- any qualified adaptive-K distortion/cost curve and saved-pool reuse result, with no fictitious design trajectory.

Use native physical coordinates, correctly labeled dataset units, common scales and unclipped metric arrays. Distinguish module IDs from spatial axes. Reference is analytic-wake/shared-grid, not CFD. Do not turn hidden feature norms into energy or latent memberships into causal flow.

Include a stable model inventory with only three statuses: retained incumbent; candidate with exact remaining qualification; archival control. Finish with **one concrete next decision**, not a list of newly named architectures.

## 13. Promotion and future data decision

At closeout choose one of the following clearly:

**Response core and grouping both useful:** retain the selected R-group family, exact flow partner and code/config identity for a separate confirmation. Prepare a bounded confirmation recipe, not an automatic full-data run. Broad unseen-layout and independent response evidence are still required for formal scientific claims.

**Response core useful, grouping not better:** retain the trained response family and R-direct as the competent control/incumbent candidate. Keep the response API and measured coefficients. Do not pretend to have proved the hypergraph hypothesis or abandon the successful operator formulation. A further grouping decision must be based on that fixed response representation, not another complete predictor redesign.

**Neither response readout adequate after valid maturation:** report whether the limitation is TRAIN fit, transfer, response rank, source-operator mismatch, role extraction or cost. Preserve the incumbent. Do not repeat identical long training or reopen the Tree portfolio.

The prior audit's17-attempt fit rank-completion request remains **unexecuted**. Additional solved directions would be valuable even with physics residual training, because the residual is not independent validation. Any future request must separate fit excitation from new-layout response confirmation and specify total/common-mode versus balanced changes. No allowance increase is inferred from permission for2500 neural epochs.

Current development panels are repeatedly exposed. A formal scientific claim cannot be created by renaming them test. Prepare any future confirmation population using input-only selection and an explicit exposure ledger; do not evaluate it in this round under a hidden scope expansion.

## 14. Delivery checklist

Deliver actual fitted checkpoints locally, measured terminal tables, executable operator APIs, a selected-family replay/continuation guide, inspected figures, and one durable report. A parser pass is not a completed training run. An unsuccessful physical target is not a software failure; both must be reported accurately.

No unrequested full-data/five-thousand-epoch run, Wind training, inverse generator/search, new reference solve, or formal restart. No broad scientific deletion. Preserve numerical misses and failed attempts without allowing them to multiply into endless retries.

Commit durable implementation/tests/docs on the nondefault branch. Enable the existing pre-push hook, audit the **entire outgoing history**, keep generated data/figures/checkpoints/one-time runners ignored, push, and verify actual remote/local tips. Do not weaken trust/security checks or add a new generic experiment-management framework.

## References and exact review basis

All empirical statements above come from these project sources. Proposed architecture, targets, operator constraints and schedules are new recommendations, not results already achieved.

- **[R1]** `docs/reports/HONF_Dependency_Correct_Forward_Interface_Report.md`, at reviewed25e2899: all22 metrics, matched1000 fits, dependencies, timing, closure/rank, limitations.
- **[R2]** `docs/reports/HONF_Dependency_Correct_Source_Audit.md`, at25e2899: active generator/configuration audit, affine thermal update, native port definitions and limited excitation. The planning review read this committed audit, not the unavailable local ignored generator itself. Codex should verify only the specific operator/extraction details needed for the new implementation rather than repeat the entire historical source investigation.
- **[R3]** `docs/reports/HONF_Response_Competent_Forward_Refinement_Report.md`: Run3801/3802 predictor gains, response misses, weak joint utility and incumbent state.
- **[C1]** `src/honf_forward_core/interface_fields/geometry_flow_field.py`, `GeometryFlowField.prepare/read`, at25e2899.
- **[C2]** `tools/thermal_dependency_flow_fit.py`, sampler, `case_losses`, cosine scheduler, identity checks and checkpoint loop, at25e2899.
- **[C3]** `Case_ThermalChannel/src/channelthermal/dependency_flow.py`, input whitelist, actual composition and heat-batch guards, at25e2899.
- **[C4]** `src/honf_forward_core/interface_fields/tensor_query_interaction.py`, `components` and access preparation, at25e2899.
- **[C5]** `docs/guides/Thermal_Dependency_Flow.md`, `docs/guides/Thermal_Model_Development_Protocol.md`, and root `AGENTS.md`.

**Planning-review boundary:** report and relevant committed source were inspected; checkpoints, local raw arrays, and the physical generator were not rerun. The new family remains a testable design, not a promised successful model.
