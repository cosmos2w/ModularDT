# HONF R-direct: formal-run readiness and receiver-local organization

## Goal and decision

Use this plan in Goal mode on `cosmos2w/ModularDT`, branch `agent/honf-core-next`, after the response-operator campaign reviewed at `967065906c8e4aad68adbe8bc68989a6591f5abd`. Inspect any later changes before execution; do not reset the checkout to this revision. Preserve existing security, trusted loading, artifact hooks, scientific checkpoints and unrelated processes.

**The primary deliverable is a genuinely executable, empirically priced, manually launched full-data 5,000-epoch R-direct research comparison. It is not another new thermal architecture or another attempt to force global groups.** Keep the competent response formulation. Close bounded numerical and training-plumbing issues, make the result understandable, and explicitly separate readiness to conduct a formal research experiment from readiness to deploy an inverse-design system.

R-direct plus the heat-independent flow reader is the preferred response-model family. R-group remains an independently trained scientific control, not the next default. It should not receive another long fit merely because it has a hypergraph label. The unsolved question of useful learned hyperedges remains part of HONF; selecting R-direct as the competent reference does not claim that question has been solved.

The round must finish with one clear recommendation: (a) launch-ready research recipe with stated scientific limitations; (b) launch-ready recipe after selecting the bounded flow refinement; or (c) a specific reproducible software/data incompatibility that prevents launch. A missing general inverse-design guarantee, a nonzero physical error, an uncompressed global kernel, or a failure to beat every historical role is not by itself a reason to withhold a research recipe.

## 1. Evidence being retained

The current campaign used 150 primary training cases, 22 repeatedly exposed development cases and four original-TRAIN response families. Both thermal arms were trained from fresh thermal weights for 2,500 epochs; they did not inherit the incumbent thermal predictor. D-sep was continued to absolute epoch 2,500. These are development results, not a matched full-data comparison with historical Run1804 or Run1502.

| Quantity | Retained Run3801 | R-direct2500 | R-group2400 |
|---|---:|---:|---:|
| Fluid-temperature RMSE, DEV22 | 0.770312 | 0.774634 | 0.844351 |
| Surface-temperature RMSE | 0.788326 | 0.712311 | 0.716744 |
| Material-temperature RMSE | 0.663080 | 0.677731 | 0.658986 |
| Mean per-module material-peak error | 0.647734 | 0.782622 | 0.743780 |
| Fixed-audit fluid heat-response RMSE | 0.037067 | 0.0115873 | 0.0138301 |
| Fixed-audit surface heat-response RMSE | 0.050119 | 0.0148677 | 0.0158156 |
| Fixed-audit material heat-response RMSE | 0.049496 | 0.0167672 | 0.0173693 |
| Complete actual-M12/Q8192 cold inference | 120.973 ms | 25.450 ms | 31.604 ms |
| Complete heat/query forward + VJP | 267.687 ms | 34.511 ms | 40.885 ms |

All errors retain their original dataset units and masks. The peak metric is not the same as material-field RMSE or whole-case peak error. The principal temperature means passed the previous prospective tolerances, but that does not erase the approximately 20.8% increase in R-direct's per-module peak error. The new model corrects both mean heat-response signs on case0291 but retains approximately 27.7% mean-amplitude excess. Saved geometry-response errors remain substantial.

The shared D-sep2500 improves u/v/p means to 0.0170684/0.00171246/0.00909258, while omega remains 0.131462 versus incumbent 0.104567. Near-omega error is 0.320751 versus far 0.0471879. These localizations motivate at most one bounded flow refinement, not replacement of the thermal response architecture.

The strongest transferable achievement is the new separation between a learned context-dependent response operator and its explicitly applied forcing. Individual source kernels are still estimates: independently solved heat excitation remains rank one per audited layout. Training with a qualified discrete operator adds physics information, not independent solved response directions.

## 2. Questions and scope of claims

### Predictor

Can the selected R-direct family be trained and evaluated on the formal dataset with correct normalization, explicit component histories, reliable numerical response application and a realistic runtime forecast? Preserve its present temperature and heat-response advantages. Local flow/peak defects should be visible, not hidden in a composite loss or converted into endless launch barriers.

### Organizer

For a specified receiver region, which learned source-response functions are important, and is any smaller local representation genuinely adequate? This round may establish an operational, receiver-local view derived from learned kernels. It must not call that view a newly trained adaptive-K hypergraph, independently established physical causality, or proven superiority over the direct predictor.

### Inverse system

Expose stable prepared response applications and readable replay of already solved alternatives. Do not train a denoiser or run a new continuous design search. Heat allocation at fixed layout, geometry optimization, and generative design are distinct capabilities. The formal research run can proceed before the latter two are established.

### Formal-training readiness versus scientific success

A launch-ready research recipe means the actual training path runs, saves and resumes correctly, preserves declared data boundaries, has a measured forecast, and will report the intended comparisons. It does not mean the eventual model is already superior to Run1804/1502 or safe for physical design. Do not describe ordinary research qualification targets as security gates. Blocking controls remain limited to the repository's actual security, destructive-write, cross-system and release boundaries, plus ordinary invalid-input/software errors.

## 3. Preserve the mathematical model

Let C contain complete layout geometry, prescribed operating conditions, material parameters, environmental context and receiver geometry. For the audited Thermal capability, current heating h must not enter coefficient preparation:

\[
F_\psi(C;r)=(u,v,p,\omega),\qquad T_\theta(C,h;r)=b_\theta(C;r)+\sum_{i=1}^{M}K_{\theta,i}(C;r)h_i.
\]

The current zero-boundary/zero-initial-temperature adapter uses b=0. Keep that capability-specific choice. Nonzero boundary forcing, buoyancy, temperature-dependent coefficients and arbitrary nonlinear design variables are not covered by this identity.

For fixed C, the physical forcing derivative and change are

\[
\partial T/\partial h_i=K_i(C;r),\qquad \Delta T=K(C)\,\Delta h.
\]

For moving geometry, the derivative is different:

\[
\partial T/\partial g_j=\partial b/\partial g_j+\sum_i h_i\,\partial K_i/\partial g_j.
\]

Good heat-response learning does not establish the latter. All geometry changes must rebuild the context and receiver operator, with live coordinate derivatives; a cached fixed-layout response is not a geometry model.

Retain the implemented near/far definition:

\[
K_i(r)=w_i(r)K^{\rm near}_i(r)+(1-w_i(r))K^{\rm far}_i(r).
\]

The current smooth near weight equals one within two radii and zero outside four radii. R-direct retains individual far-source coefficients. R-group represents the far coefficients through a sum of mode functions and memberships, with exact near subtraction. Do not change these definitions in the formal-preparation round.

The direct kernel still depends on the entire layout through learned source-to-source and source/environment contextualization. Linear dependence on heating does not imply independent geometric effects. Conversely, an arbitrary nonlinear forcing-map API does not turn this into a validated universal nonlinear operator.

## 4. Workstream A — establish the historical comparison bridge

Run read-only native evaluations of the exact retained Run1804 and Run1502 selected checkpoints on the same fixed DEV22 and the same four detailed representatives used for R-direct. Read original checkpoint metadata and use each historical native loader, its own input normalization and its complete predicted-port/local-surrogate path. Convert outputs to their correct native units before applying common comparison metrics. Never move old weights into the new architecture or replace old role semantics to simplify comparison.

The expected historical selected ages from previous reports are Run1804 e4738 and Run1502 e4794. Resolve their real local paths from maintained records and verify the loaded ages; do not assume that a file named `best` is the reported selected state. A literal e5000 checkpoint is an additional endpoint comparison only when it exists. Missing files are reported as unavailable after a bounded local search, not reconstructed from old tables.

Evaluate R-direct2500, R-group2400, Run3801 and the two historical classics using the same native receivers, fluid masks, module identities, role definitions and equal-case reductions. Reuse already saved R-direct/Run3801 outputs when the full measurement identity matches. Keep historical full-population results in a separate contextual table; do not put the old 89-case fluid RMSE next to DEV22 and call the difference a matched result.

The main comparison has eight physical rows: u, v, p, omega, fluid T, surface T, material T and material peak. Report q-proxy and outside-port quantities in the appendix and in their own physical plots where relevant. Preserve initial-port NA for the new family: absence of an old internal trajectory is not zero error. Include near/far and M-stratified results in the detailed appendix, not the opening page.

Where the complete saved heat-response records remain compatible, replay the classics on the same three primary baseline-relative layouts and six signed changes. Separate secondary0277. Do not substitute its packed nominal field for the missing counted baseline. These observations remain exposed evidence; historical models differ in training population, objective, age, architecture and response supervision.

The deliverable is a simple answer to “Where are we relative to 1804 and1502 on the same cases?” It must not claim isolated architecture causality. The new model receives a source-qualified affine law, four response families and a discrete-balance training objective that the classics did not receive in the same way.

## 5. Workstream B — stable response arithmetic without another fit

Keep ordinary FP32 cold inference as the baseline. Add an explicit high-accuracy prepared-response application for small changes, using the same learned coefficients. Do not retrain the network, cast a final FP32 answer to double and claim restored accuracy, overwrite failed records, or loosen the old comparison tolerances.

For a native linear role tau, let L_tau denote its exact adapter extraction from grid temperature. Prepare

\[
K_\tau=L_\tau K,\qquad \Delta y_\tau=K_\tau\Delta h.
\]

Use FP64 accumulation for the forcing contraction and the relevant interpolation/linear proxy operations when the precise mode is requested. For q-proxy, apply the declared harmonic conductivity and native outside-minus-surface stencil to coefficient differences before forcing contraction. Keep the physical output order, signs, delta and masks. Source-kernel preparation can remain FP32; this improves arithmetic on that learned kernel, not its physical accuracy.

Do not compute small changes by subtracting two large rounded temperatures when a prepared linear response is available. Nevertheless, retain endpoint-subtraction tests to characterize historical FP32 behavior. Compare: (1) native cold versus prepared outputs at the same precision; (2) direct increment versus endpoint difference using the same fixed kernel and FP64 application; (3) exported physical kernel versus input VJP; and (4) legacy FP32 endpoint subtraction, with its existing failures shown separately. Include positive/negative balanced changes, nonbalanced legal source changes and small step sizes on TRAIN cases first; final reporting uses the existing fixed representatives.

Nonlinear quantities are excluded from a linear-increment assertion. For material maxima and effective h, apply the two endpoint fields and then the original nonlinear reductions, or define an explicit local derivative with the appropriate qualification. The increment path must not fabricate an h-change by applying the h ratio to Delta T and naming it Delta h_effective.

Implement this inside the existing response operator/adapter or a narrow helper. A proposed API is `apply_record_increment(..., accumulation_dtype=torch.float64)` with the actual implementation names documented. Ensure casts preserve the intended heat derivatives and retain original physical heat coordinates. Test prepared-state staleness and geometry rebuild behavior using existing primitives. No new cache service or identity framework is required.

Use four TRAIN precision cases and the four existing representatives, not a new large numerical campaign. Price the supported precise application separately from ordinary cold inference, including extra storage. Do not claim the documented FP32 discrepancy is entirely rounding until fixed-coefficient precision experiments support that diagnosis; isolate any source-cast or input-quantization mismatch first.

## 6. Workstream C — one bounded near-boundary flow refinement

This is secondary to delivering the formal recipe. It has at most two cheap children and no new thermal fit: an ordinary D-sep continuation control and one near-boundary/consistency recipe, both starting at the same D-sep2500 weights and AdamW state. Never reopen D-open or enable heat in flow.

First read the generator's exact definition of omega, including signs, finite-difference spacing, wall/solid masking and boundary order. Reproduce that extraction on existing TRAIN velocity arrays. Qualification uses TRAIN layouts spanning low/high M. If the proposed discrete curl does not reproduce the label semantics, do not impose a generic continuum identity. Fix a straightforward stencil/masking mismatch once; otherwise disable the auxiliary consistency term and document why. No generator solve is invoked.

For a qualified native discrete operator D_C, the proposed recipe is

\[
\mathcal L_{\rm flow}=\mathcal L_{\rm original}
+0.1\,\mathbb E_{r\in\mathcal N}\left[(\widehat\omega(r)-\omega(r))^2/\sigma_\omega^2\right]
+0.1\,\mathbb E_{r\in\mathcal N}\left[(\widehat\omega(r)-D_C(\widehat u,\widehat v)(r))^2/\sigma_\omega^2\right].
\]

Here N is an input-defined near-obstacle fluid band, initially 0 <= distance-to-surface <= one radius; choose up to128 existing native centers per case and their required discrete neighbors. All scales come from selected TRAIN. Preserve the original uniform Q1024 reconstruction stream and four ordinary optimizer updates per development epoch. Extra near targets/stencil queries are explicit auxiliary work, not silently included in “Q1024.” This pair is matched in initialization, primary cases/queries and updates; its auxiliary objective/work intentionally differs. Compare wall time as well as epoch age.

Predict all four channels with the existing flow network. Do not silently replace omega in the final output with a numerical curl, train on DEV or force a derivative through a label. If the qualified extraction requires exact zero solid velocities, apply that known domain rule only inside the explicitly documented stencil construction, and verify its consistency with original output semantics.

Use a declared 500-new-epoch continuation schedule for both children: warm the learning rate from1e-6 to3e-5 during20 epochs, hold through200, cosine-decay to3e-6 by500. Review at100 and complete500 if numerically healthy and within budget; there is no additional portfolio or automatic extension. The thermal model stays fixed. Preserve old D-sep2500 regardless of outcome.

Select a flow recipe from primary DEV22 u/v/p/omega and near/far tails, not from known-null success alone. The desired practical result is lower near and overall omega error without material u/v/p degradation. A suggested comparison guide is no more than5% mean and10% p90 degradation in u/v/p versus the continuation control, together with improvement in omega. These are research decision aids, not new software/security gates. If the refinement fails, retain ordinary D-sep as the formal research recipe and keep the omega limitation visible; do not withhold the otherwise valid full-data experiment indefinitely.

## 7. Workstream D — a receiver-local organizational view, not forced global compression

Do not retrain R-group, add an entropy/K penalty or resurrect a Tree. Its current valid mode count is M+1; on M10 it constructs11 modes for10 sources, all110 source memberships are positive, and it evaluates more far rows than R-direct. The previous positive-budget direct SVD ranks of2/4/6/9 equal M-1, the generic fixed-total-heat dimension. Those facts do not demonstrate reusable global low rank beyond the known constraint.

The next organizational measurement is local in receiver scope. Use the four fixed representatives and a fixed4x4 spatial partition in native coordinates. Preserve physical source IDs. For each receiver patch P and each source i, export the learned coefficient field K_i(r), the current signed contribution K_i(r)h_i and the bounded response score

\[
q_i(P)=r_i\,\|W_P^{1/2}K_{P,i}\|_2,
\]

where W_P is normalized patch quadrature and r_i is the existing TRAIN-input-bounded forcing radius. Rank sources by q_i and measure actual omission distortion for the saved legal heat changes. The triangle sum of omitted q_i is a model-response bound in that defined norm; it is not a bound on physical prediction error. Report measured distortion as well as the conservative bound.

Construct an explanatory donor cover for each patch by retaining enough sources for the existing1% and2% training-response-scale budgets. Keep full source-resolved prediction as the authoritative forward and gradient path. Do not use a hard cover to introduce discontinuities into geometry optimization. Full membership is an allowed result. Define K_local clearly as the number of retained physical donors for that patch/budget, not as discovered latent hyperedge count.

Compare against two simple source selections at the same cardinality: nearest geometry and input-declared upstream geometry. These are bounded controls, not learned rivals. Do not fit selector parameters on reference errors. Each patch should also show how omission changes the already measured heat-response error, keeping the full-model error separate. All donor-column interpretations remain model estimates outside solved excitation directions.

A small optional diagnostic computes singular values of the receiver-patch far-source matrix W_P^(1/2) K_(P,F) diag(r). It uses the learned matrix only, not a physical inverse. Report both unrestricted forcing rank and the balanced-subspace rank. Do not count the one common-mode removal as learned compression. Limit this to the64 predefined patch matrices across the four cases. This answers whether receiver-local far response has redundancy worth a later learned organizer; it does not start that later training now.

Deduplicate exactly identical donor covers for visualization only, retaining disconnected receiver regions and signed effects. A set of sources that matters at the same receiver can define a useful hyperedge-inspired interface, but this post-training view is derived from a learned operator. It is not yet proof that learning grouped interactions improves reconstruction, generalization, execution or inverse decisions.

## 8. Workstream E — implement the formal research path

### Concrete current incompatibilities

The reviewed `thermal_source_response_fit.py` hardcodes the fixed25 fingerprint,150/22 case counts,2500 schedule, four updates and exact flow age2500. The loader in `channelthermal/source_response.py` also requires literal fixed25 membership and matching quarter-data normalization. Changing only `--stop-after` or a JSON epoch count cannot produce a legitimate full-data experiment.

Generalize the maintained configuration path, not the scientific history. Keep legacy development loading and strict resume behavior intact. Introduce an explicit formal profile using the existing configuration/dataset/checkpoint infrastructure. Dataset scope, case IDs, normalization provenance, component ages and schedule are profile values; all loop lengths and work counters derive from actual data. No new cryptographic system, approval store, baseline snapshot scheme or freeze framework is requested.

Separate case validity from workflow identity. The affine Thermal capability still requires its audited physical assumptions; formal versus development controls which examples, normalizers and schedules are used. A formal loader must reject a mismatched flow partner or normalization, without pretending that every valid model must use the old150/22 manifest.

### Full-data membership and baselines

Resolve the actual source H5 split from metadata; expected historical sizes are600 TRAIN and90 legacy validation. Use all eligible original TRAIN cases for the formal fit. Fit normalizers only on that full training partition, not on the quarter-data parent. Use the same normalization object for the formal thermal and flow components. The four existing TRAIN response anchors belong to the full training partition; their perturbation records remain an explicit additional supervision source.

Preserve compatibility evaluation on all90 legacy validation rows, and make the duplicate-excluded89-case result primary for scientific comparisons because historical0273 duplicates training0001. Do not relabel either panel independent or untouched. Do not create a new test split from repeatedly inspected outcomes. Future independent physical evaluation remains a separate authorized task.

Evaluate old checkpoints in physical units using their original native input transformations. For any common normalized comparison, apply one declared full-TRAIN target scale after denormalization; do not mix checkpoint-specific normalized L2 values into an architectural ranking.

### Recommended manual recipe

Prepare one recommended new-family recipe: **R-direct formal5000**, consisting of a freshly trained full-data D-sep flow component and a freshly trained full-data R-direct thermal component. The thermal stage keeps the completed formal flow component fixed; thermal fitting does not need to backpropagate through flow. Both components see the full training split. Do not pair a formal thermal component with the quarter-data flow checkpoint and label the result a fully trained full-data model.

Use a simple two-stage manual command or the project's existing staged-launch facility: first flow5000, then thermal5000. Each component age and its cost are recorded separately; “5,000 epochs” means5,000 passes per declared component, not5,000 undocumented combined updates. The final exact-endpoint composition uses both e5000 endpoints. Selected-checkpoint compositions, if additionally provided, keep both component ages explicit and are not substituted for the endpoint.

For thermal, retain the reviewed architecture, near/far semantics, native extraction and objective: three separately standardized temperature roles,0.05 q-proxy, declared measured-response loss and qualified discrete-kernel residual. Retain the same training-only calibration rule; recompute it for the formal initialization/TRAIN scale rather than importing quarter-data numerical coefficients without explanation. Do not change the response target family or add geometry labels during formal preparation.

Use a formal schedule declared before launch. The default thermal schedule keeps LR3e-4 through40% of5000 and cosine-decays to3e-6 over the remainder, matching the reviewed relative schedule. The flow schedule is declared separately and uses a positive minimum. If the bounded flow recipe is selected, its near/consistency objective must be specified in the fresh full-data flow recipe; do not transplant only a development repair head and call it fresh training.

Recommended training precision remains FP32. Precise prepared-response application is an inference capability, not an unmeasured all-FP64 training change. Keep effective batch48/microbatch8 and Q1024 primary fluid sampling unless an actual memory measurement requires a documented amendment. Every formal epoch visits every selected original TRAIN case. Derive actual update counts, material/surface rows, response visits and operator work from the runtime stream.

Monitoring saves every100 epochs with latest, best and declared milestones, preserving ordinary atomic save and resumable stop behavior. Record both component-specific selectors and a common physical five-field validation score for the composed model where available. Never select an endpoint based on a physical-audit response opened only for final evaluation. Plot exact-endpoint and selected results separately.

### Optional second manual recipe

Prepare a fresh Run1804-like full-data Dense5000 comparator only through its maintained classic architecture/configuration and loader. Keep Run1502 read-only as an additional historical comparator. Do not spend this round training either classic for5000, and do not prepare a new R-group5000 recommendation by default.

The new-family and classic recipes have different physical priors, objectives and intermediate representations. Equal data/epochs do not isolate the grouping ingredient. Report two questions separately: practical quality/cost of the complete pipelines, and group-specific value from the already completed R-group/R-direct matched experiment. A reconstruction-only new-family ablation could later isolate training-prior benefits, but is not added to this round's portfolio.

### Actual startup and cost evidence

This plan explicitly permits a **bounded full-TRAIN startup benchmark**, not a formal campaign: up to three real full-training epochs for each new-family stage under the actual formal profile and up to one real epoch for the optional classic profile. Use distinct disposable startup identities, fresh weights and normalizers, all scheduled objectives and ordinary optimizer updates. Do not promote those temporary weights to the formal starting checkpoint. Validation remains on the bounded development panel during this startup verification; full90 evaluation belongs to the user-launched formal workflow.

Test actual save/load/resume across the startup boundary, with an immediately continuing epoch and matching parameter/optimizer progression. A CPU dry run, parser pass, fake sampler or FLOP estimate is not an execution substitute. Existing unsafe-write/security boundaries still apply.

Report measured time per full-TRAIN epoch, nonoverlapping validation/save/loading costs, peak allocated memory, and a forecast for flow5000 plus thermal5000. Use measured early and steady epochs and note contention. State that the forecast is conditional on unchanged schedule/hardware and does not guarantee accuracy. Do not extrapolate formal thermal runtime from the very cheap flow-only epochs or from25ms inference. If occupancy persists, continue on an authorized usable GPU or with documented contention; never wait indefinitely or stop another process.

Deliver exact manual start, status, stop-request and resume commands tested against the implemented entrypoints. No5,000-epoch run is auto-launched. Do not put an unimplemented CLI flag into the final guide. If a new profile or command must be created, implement and execute it before presenting it as usable.

## 9. Evaluation and decision logic

Maintain three layers of evidence instead of one giant all-purpose pass/fail claim.

**Training software readiness:** correct data/normalization binding, finite genuine updates, native loading, checkpoint/resume, practical runtime and correct output semantics. These determine whether the manual research recipe works.

**Predictor/reuse quality:** common physical fields, peaks, heat responses, cold/prepared equality, stable heat response arithmetic, vorticity and geometry-response limitations. These describe the current candidate and motivate the formal comparison. They do not require perfection before that comparison is allowed.

**HONF hypothesis:** added value of learned grouping; faithful current information paths; response transfer and later inverse decisions. The current family establishes a better response interface, not a successful compressed adaptive hypergraph. This hypothesis remains open and is not relabelled solved to justify a run.

Do not require a new inverse generator, uniformly sparse K, successful geometry optimization, independent donor-kernel truth or all-role superiority as prerequisites to preparing the formal research run. Conversely, do not claim any of those properties from recipe readiness.

For strict arithmetic flags, keep the old failing measurement and show what the precise API changes. For geometry, compare existing saved common-mask responses, not imagined moved layouts. For full-model accuracy, include the material-peak warning and near-omega warning instead of hiding them behind improved mean T or u. No new physical attempts are authorized; ledger remains326/326 and0277's missing baseline remains missing.

## 10. Five intuitive main figures

Retain detailed quantitative appendices, but the first report pages must be readable without knowing every run number. Use labels such as “Classic Dense,” “Classic HONF1502,” “Direct response,” and “Grouped response,” with exact checkpoint identities in the appendix. Never place digests or a full software receipt in a plot title.

### Figure1 — What field does each model predict?

Show stored temperature, R-direct temperature, classic Dense temperature and their signed residuals at the same native coordinates. Use at most two case rows in the main report: one fixed representative and the fixed high-M case. Display one temperature scale per row, one symmetric residual scale per row, module IDs and the maximum material-temperature location. Put u and omega in a separate compact companion, not twelve tiny panels on the same page. All22 statistics accompany the figure in one small table.

### Figure2 — Which sources matter at this receiver?

Mark three input-chosen receivers/patches: upstream, near a module and in an overlapping downstream region. Show source locations and ranked learned coefficient strengths. Beside them show the actual predicted field change from one of the already solved heat transfers, its reference change and its residual. Distinguish K_i, K_i h_i and K_i Delta h_i with units. A donor column without independent excitation is labelled “model estimate,” not “measured source effect.” Environmental context dependence is displayed separately from independently actuated heating.

### Figure3 — Is grouping buying anything?

Use one clear fixed-layout direct-versus-group comparison: source count M, valid modes M+1, cold milliseconds and heat-response RMSE. Add the local-cover comparison at fixed receiver scope/budget as an exploratory inset or separate page. Do not display a uniform M+1 count as learned adaptive K. Do not hide small positive memberships to manufacture sparse pictures.

### Figure4 — What happens when the design changes?

Show the unchanged geometry and physical heating assignments for the stored minus/baseline/plus0291 alternatives, the measured versus predicted mean-temperature changes, and the measured versus predicted material peaks. Add one saved obstacle move as a deliberately separate “geometry still difficult” row. Existing design states are not generated optimization trajectories. State that correct stored-pool choices were already achieved by older controls.

### Figure5 — How long does it take, and what can I launch?

Show complete cold prediction, prepared three-heat reuse with formation included, and the measured full-TRAIN epoch forecast. Use seconds/milliseconds/GPU-hours consistently; no mixed physical-role axis. Put a short launch-readiness box below the figure with exact supported scope and the manual guide location. The user should be able to identify the preferred model and remaining caveats in less than a minute.

Render PDFs plus small PNG companions only where needed for inline Markdown. Inspect all retained images visually. Use spatial coordinates, readable fonts and no more than six panels per page. Detailed four-case exports remain linked in an appendix. Keep one short index and remove only superseded redundant render copies according to the existing repository rule. Preserve scientific raw arrays and histories. Prose paragraphs and figure captions remain single continuous source lines.

## 11. Budget, order and bounded autonomy

Use the currently authorized devices. The standing default is GPUs1/2; GPU0 requires an explicit current authorization, not an assumption based only on a previous report. Respect any current user override and preserve unrelated jobs. Account for contention rather than waiting for exclusive use indefinitely.

Whole-round ceiling: **8 aggregate GPU-associated hours and6 elapsed hours**. This is not permission to consume the ceiling. Reserve the final45 minutes for report, inspected figures, actual command verification and repository delivery. Count failed attempts once. Do not implement a monitoring service or a new resource-accounting framework.

Execution order is A and formal-profile plumbing first; stable arithmetic B in parallel with the cheap flow pair C; receiver-local D and final formal startup benchmarks after the supported execution path is stable. Prioritize the actual manual recipe and comparison dashboard over optional patch-SVD plots. Reuse saved evidence instead of rerunning unchanged campaigns.

The code review/implementation should receive a bounded exploration allowance: at most two concrete remedies for a numerical or stencil issue. Execute the alternatives on real cases and record the outcomes. If neither resolves it, preserve the honest limitation and complete unaffected work. Do not return “not ready” solely because an optional diagnostic is hard, and do not replace real training/startup execution with more assertions.

The only new fit portfolio is the two small flow continuations, at most500 new epochs each. No thermal architecture search, new R-group fit, forced sparsity schedule, new seed sweep, fresh Wind fit, reference solve, continuous inverse search or automatic formal launch belongs to this round.

## 12. Implementation map

| Existing file or area | Intended work |
|---|---|
| `src/honf_forward_core/interface_fields/source_response_operator.py` | Preserve core math; add explicit precise forcing/increment accumulation and any narrow local-view export needed. No new group architecture. |
| `Case_ThermalChannel/src/channelthermal/source_response.py` | Preserve native role extraction; make precision semantics explicit; add scoped formal loading without weakening legacy development validation. |
| `Case_ThermalChannel/src/channelthermal/source_response_residual.py` | Preserve qualified TRAIN-only A K-B; avoid physical solves and inference truth access. |
| `src/honf_forward_core/interface_fields/geometry_flow_field.py` | Preserve network for the bounded flow recipe; no heat input or second full thermal model. |
| `Case_ThermalChannel/src/channelthermal/dependency_flow.py` | Retain case-owned whitelist and dependency semantics; support the selected formal flow identity using existing versioned configuration. |
| `tools/thermal_source_response_fit.py` | Parameterize scope, counts, normalizer source and horizon through maintained profiles; remove hardcoded loop/counter assumptions for new profiles only; preserve old recipe replay. |
| Maintained D-sep fit/continuation tools | Implement the bounded objective and fresh formal profile with explicit histories; use actual existing entrypoints. |
| `tools/thermal_source_response_evaluate.py`, `..._cost.py`, `..._interventions.py` | Common classic comparison, precise response test, same-scope cost and simple local-view measurements. Reuse untouched outputs when valid. |
| Existing config_core forward profiles and guides | Add the tested manual new-family5000 recipe and optional classic comparator recipe; no unexecuted flags. |
| Focused tests | Old/new loading, cross-scope rejection, actual zero heat access, physical-unit kernels, high-accuracy increments, native proxy definitions, real save/resume and unchanged historical replay. |

Do not refactor unrelated repository systems. Preserve old validators by dispatching on the existing/new explicit workflow identity, not by bypassing checks. Ordinary Git history, typed metadata, existing checkpoint schema, tests and atomic saves are sufficient; no new cryptographic sealing or approval mechanism is requested.

## 13. Required closeout

Deliver a durable Markdown report, tested manual launch guide and exact new profiles/commands, plus local ignored figures and detailed numerical appendices. The opening page must answer in plain language: what improved in the predictor; what organizing value remains unproved; what inverse reuse is supported; what exact formal experiment the user can now start; and what its measured runtime forecast is.

The report must not end with an undifferentiated list of failures. Give one concrete base-model decision, one concrete flow decision and one concrete launch decision. If optional work misses its target, explain whether that changes the research recipe or only its qualification. Preserve the distinction between a supported research launch and an approved inverse-design system.

Commit durable implementation/tests/docs, inspect the entire outgoing range including historical added-then-deleted files, run the existing artifact hook and push the current non-default branch. Keep checkpoints, arrays, generated figures and one-time diagnostic/render code ignored locally. Verify remote/local tips and accurately report any unfinished work. Do not delete earlier scientific runs, restart stopped formal3501/3502, or weaken trusted-loading/security behavior.

## 14. Sources used to define this plan

[S1] `docs/reports/HONF_Response_Operator_Consolidation_and_Maturation_Report.md`, campaign at9670659: current field/response/flow tables, group work, strict arithmetic misses, geometry response, complete costs, exposure and retained identities.

[S2] `docs/guides/Thermal_Source_Response.md`: current capability, loading and fit interface.

[S3] `src/honf_forward_core/interface_fields/source_response_operator.py` at9670659: input-only context, direct/group readouts, actualM+1 modes, positive membership normalization, near correction, forcing application and compression semantics.

[S4] `Case_ThermalChannel/src/channelthermal/source_response.py` and `source_response_residual.py` at9670659: native interpolation/proxy definitions, FP32 application casts, formal-loading incompatibilities and TRAIN-only discrete balance.

[S5] `tools/thermal_source_response_fit.py` at9670659: literal fixed25 data binding,2500 schedule and counters, objective, selected checkpoints and flow2500 dependency.

[S6] `src/honf_forward_core/interface_fields/geometry_flow_field.py` and dependency-flow guide/source at9670659: compact learned reader and actual input restriction.

[S7] `AGENTS.md` at9670659: development/formal separation, source paragraph formatting, artifact/security rules and inspected visual-report requirements.

[S8] PyTorch official Numerical accuracy documentation and `torch.gradient` documentation, consulted only for implementation context. Use documentation matching the installed environment; do not upgrade the environment to match a newer website. Floating-point ordering/precision warnings motivate measurement, not an automatic explanation of all current discrepancies. Public documentation locations: `https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html` and `https://docs.pytorch.org/docs/stable/generated/torch.gradient`.

All proposed work, thresholds, budgets and future launch decisions above are recommendations, not claims that the new experiments have already run.
