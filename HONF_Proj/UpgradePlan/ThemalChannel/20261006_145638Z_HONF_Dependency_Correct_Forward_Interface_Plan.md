# HONF: dependency-correct forward interface and bounded thermal-response diagnosis

## 0. Decision, scope, and the concrete deliverable

**Starting source:** `agent/honf-core-next`, reviewed at `9448df3ee7553e4a35fcb79b1af85eb9fb989059`.

**Central objective:** learn a reusable organization of module–environment interactions through multi-field reconstruction, then reuse it for modular inverse design. Neither a decorative graph nor a small reconstruction improvement establishes that objective.

This round will implement and train a **case-owned, dependency-separated ThermalChannel forward interface**. It will preserve the strong thermal predictor from Run3801 while replacing its heat-sensitive flow outputs with a genuinely learned, heat-independent flow reader. A matched heat-enabled flow reader is the experimental control. The two flow readers have the same architecture, initialization, training cases, query stream, and optimizer budget; only access to heating differs.

This is deliberately **not another H-add/H-joint thermal-refinement campaign**. Run3801/3802 already answer that comparison well enough for now: both reconstruct well and their difference is small. It is also not a promise that flow separation alone will correct the wrong thermal response on case0291. Thermal outputs are intentionally preserved in the primary experiment, so their remaining errors cannot be misrepresented as repaired.

The required outputs are:

1. A source-backed table of the benchmark's actual physical dependencies, including the distinction between prescribed flow, thermal fields, effective ports, and proxy flux.
2. An executable two-lane forward interface, with a measured new flow predictor rather than cached reference values, analytic-generator substitution, or suppressed gradients.
3. Two matched flow-reader fits reaching 500 new development epochs when numerically healthy, with a real review at100; optional matched1000 only on evidence and measured cost.
4. Same-operator removal of demonstrably redundant work in the retained separable thermal reader, plus complete-wrapper timing at both sampled and full-grid query sizes.
5. A bounded diagnosis of thermal response linearity, available excitation directions, and the existing port/refinement path. This uses saved evidence and model calls only.
6. A plain-language report that makes clear which reliability problem was repaired and which remains. No formal training, new physical solve, or generative inverse campaign is authorized.

The new composed model is a **dependency-repair candidate**, not proof that the learned hypergraph is uniquely valuable. The heat-enabled reader is a scientific control, not the intended production dependency policy.

## 1. Evidence that motivates this scope

The authoritative result is `docs/reports/HONF_Response_Competent_Forward_Refinement_Report.md` [R1]. Do not substitute a more favorable historical checkpoint or a different response aggregation.

### 1.1 Preserve the substantial predictor gains

On the same22 exposed validation cases, selected Add500 / Joint500 / retained Dense-D25 have the following equal-case native RMSEs:

| Quantity | Add500 | Joint500 | Dense-D25 |
|---|---:|---:|---:|
| Fluid temperature | 0.770312 | 0.779114 | 0.949971 |
| Surface temperature | 0.788326 | 0.794336 | 0.974319 |
| Material temperature | 0.663080 | 0.669472 | 0.865606 |
| Fluid u | 0.0220367 | 0.0217359 | 0.0284574 |
| Fluid p | 0.0122219 | 0.0122465 | 0.0139807 |

These are meaningful development improvements. Dense differs in feature representation, architecture, and training history; this table is not a causally isolated architecture comparison. It is not independent-test evidence.

Run3801 is the primary retained thermal backbone because its thermal means and pressure mean are slightly better than Run3802 and the added joint mechanism has not earned a clear benefit. Retain Run3802 and old Tensor-H as separate scientific references. Do not delete their code, checkpoints, or evidence.

### 1.2 Do not confuse these three remaining problems

**Wrong thermal changes:** on0291, reference mean fluid-temperature changes are approximately -0.028919/+0.028914. Refined Joint predicts +0.009844/-0.015100. The substantial wrong signs remain despite smaller field errors.

**Forbidden flow dependence:** all22 heat-null u/v/p changes fell by only about4–13%; omega leakage worsened. A positive null penalty was executed. Another identical long run or a larger penalty is not the primary next experiment.

**Weak unique organizing value:** matched joint thermal-response advantages are around0.5%, and same-state I-removal changes response errors only slightly. Both models already rank the three stored peak-temperature pools correctly; that does not demonstrate a superior inverse policy.

### 1.3 Price complete application, not only a convenient batch

Current B1/Q8192/M12 complete inference is0.375005s for Joint versus0.230673s for G-fast; complete input forward+VJP is0.918953s versus0.523653s. B8/Q1024 overhead is smaller. Use both scopes in the next report. Dense physical-value execution remains active.

### 1.4 Source review boundary

The planning review inspected the committed wrapper, input adapter, query-control code, response adapter and training declarations. The local ignored generator was not directly accessible during that review. Existing reports document heat-independent analytic flow and near-linear fixed-geometry heat responses. **Codex must read the actual local generator and the active case configurations before elevating either statement to a new structural model capability.** No physical run is needed to inspect source or configurations.

## 2. Non-negotiable research and resource boundaries

Keep `fixed25_v1` primary membership:150 TRAIN and22 exposed validation cases. Keep the original train-only normalization; do not silently add original full-data cases to the primary fit. Reuse identical data, seeds, sampler streams and budgets between the new flow arms.

One new flow-fit epoch visits all150 selected cases with Q1024 sampled valid fluid receivers. The new fit optimizes the four flow channels, not the frozen temperature channel. State that distinction in every learning table. It is not500 more epochs of complete physical-backbone training.

Use physical GPUs1 and2; never interfere with unrelated work or restart formal3501/3502. Avoid waiting indefinitely for exclusivity. Move between the authorized devices, serialize work, or continue under documented contention. Do not derive an exclusive-device speed claim from contended epochs.

The entire round has a ceiling of **12 aggregate GPU-associated process-hours and8 elapsed hours**, including failed experiments, evaluation and reporting. It is a ceiling, not a spending target. Reserve at least the final45minutes for measured closeout. Use ordinary process/epoch timers; do not build a new monitoring service.

Save milestones, latest, best-field/flow selector and plots every100epochs, plus a requested final stop. Use the existing supported epoch-boundary stop-save mechanism. Do not create25-epoch checkpoint series or duplicate selector archives.

Statistics use all22 cases. Detailed exports use0277/0291/0294/0687. Additional training cost panels may use actual M1/M12; label them separately from the validation M3/M5/M7/M10 strata.

The cumulative physical-reference allowance remains **326/326: zero new attempts**. No0277 baseline retry, substitute baseline, hidden solver in a training loss, new CFD, or relabelled auxiliary reference generation is permitted. Reading source and already saved arrays is not a new solve.

Do not launch Wind training, a new inverse generator, a continuous design search, a formal5000 run, or another organizer portfolio.

Preserve existing checkpoint trust, artifact restrictions and security measures. Do not introduce new cryptographic inventories, contract-freeze systems, approval databases, or generic gating frameworks. Use existing configuration fields, explicit experiment identities, types, tests, Git history and the existing checkpoint writer. Scientific comparisons and resource decisions are not production-release certifications.

## 3. First task: establish the actual dependency model

### 3.1 Source and saved-input audit, bounded to60minutes

Locate the active local generator using the existing reference adapter and recorded case provenance. The historical path is `1_Demo_ChannelThermal/src/simulate_channelthermal.py`; verify the current file instead of assuming this filename still resolves.

Read the actual routines for:

- velocity and pressure construction, optional projection/correction, vorticity, and pressure gauge;
- thermal coefficients, advection/diffusion, heat deposition, boundary conditions and convergence;
- sampling of outside temperature, normal/tangential velocity, h-proxy/h-effective, surface temperature and q-normal proxy;
- optional temperature-dependent material coefficients, buoyancy, radiation, clipping or source-dependent geometry;
- physical versus normalized input encoding and the active flags in packed case configurations.

Produce a compact dependency table whose rows are output roles and whose columns are geometry, prescribed operating conditions, material properties, heating, predicted ports and query geometry. Mark each entry as source-confirmed, measured in saved states, or unresolved. Cite functions and source lines in the local report.

Do not simply state that all multiphysics flow is heat-independent. This applies only to the audited benchmark variant. A future buoyant or temperature-dependent flow case may require thermal-to-flow feedback.

Do not assume h-effective is heat-independent. An effective coefficient derived from temperature ratios, clipping or reconstruction can depend on heat even when the flow does not. Its label semantics require their own audit.

The existing `Dataset/PHYSICS_AND_DATA.md` begins with a general coupled-flow description. Clarify the distinction between the intended physical application and the actual active analytic-wake/shared-grid label generator. Do not modify data values or silently change schema meanings.

### 3.2 Audit the neural paths, not just explicit input columns

The current source-local adapter zeros global heat slots but keeps own-heat features in module tokens. Those tokens enter fine/coarse/local interactions, predicted ports, Stage-A responses, fused module states and the final shared field decoder.

Consequently, merely deleting global heat statistics or adding a separate last-layer pressure head is insufficient. The flow head must not read any upstream tensor that changes when only heating changes.

Create a short path inventory for:

`physical heat → normalized source features → encoder → P0 → ports → Stage-A → fused states → P1/P2 → flow outputs`.

Include global calibration, organizer/reference means and normalization. Distinguish gradients with respect to input values from gradients with respect to model parameters.

### 3.3 Canonical input semantics across APIs

R1 records different receiver catalogues in nominal-H5 and counted-response calls. This is a risk to investigate, not an already measured explanation for the0291 sign error.

On two training cases, call nominal, typed-response and prepared-query interfaces on the same physical inputs and the same receivers. Compare the actual input normalization, module IDs, port count/angles, source measures and receiver-reference catalogue before blaming floating point.

If an input-only reference convention differs, record the old behavior and introduce one explicit new-case convention used by all new candidate calls. Do not reinterpret reference measures as observed fields or change historical readers silently. A small reference-catalogue repair needs only normal unit/native tests, not a new provenance system.

The two failed-baseline0277 records remain secondary-only regardless of this API audit.

## 4. Mathematical design: separate a known zero from a learned response

Use r for a physical query receiver, g for all layout/geometry information, c for prescribed context, and h for the vector of active-module heat inputs.

The desired case-owned observable dependency is

\[
\widehat f(r;g,c)=\bigl(\widehat u,\widehat v,\widehat p,\widehat\omega\bigr),
\qquad
\widehat t(r;g,c,h)=\bigl(\widehat T,\widehat T_s,\widehat T_m,\widehat q_{proxy}\bigr).
\]

For the confirmed benchmark capability:

\[
\widehat f(r;g,c,h_1)=\widehat f(r;g,c,h_2),
\qquad
\frac{\partial\widehat f}{\partial h}=0.
\]

This identity follows from the actual flow-input signature, not a penalty coefficient. Geometry, inlet conditions and allowed flow parameters remain live inputs:

\[
\partial\widehat f/\partial g \text{ and } \partial\widehat f/\partial c
\]

must not be disabled.

There is no corresponding requirement that thermal responses vanish. All existing heating paths to the thermal predictor remain live. Sharing the physical layout is allowed; reading a heat-conditioned hidden state in the flow branch is not.

### Explicitly disallowed shortcuts

- `heat.detach()` is not physical independence: it removes an AD path while finite re-evaluation can still change predictions.
- Do not overwrite predicted pressure increments with zero in evaluation.
- Do not cache one solved or teacher-predicted flow field by case ID and reuse it for changed geometry.
- Do not replace learned flow with a call to the analytic reference generator and call that learned HONF success.
- Do not use a canonical heat computed from the current total/mean/max heat: that still gives heat access.
- Do not run two complete thermal wrappers per ordinary prediction to obtain a heat-clamped flow output.
- Do not equate exact heat-null behavior with correct absolute pressure, calibrated feasibility, or valid geometry derivatives.

## 5. Candidate and matched control

### 5.1 Preserve the thermal backbone

Attach both new flow readers to the **same selected Run3801 Add500 thermal model**, using its exact checkpoint, normalization, port coupling, soft source organization and Stage-A. Resolve its path from maintained run mappings; do not invent a run directory.

Freeze the entire inherited thermal model in the primary comparison. Its temperatures, material/interface predictions, initial/final ports and their heating sensitivities are preservation targets, not new learning targets. Run3802 and old Tensor-H are read-only references.

The complete returned `pred_field` keeps the documented five-channel ordering, combining the learned new flow reader's four outputs with the retained thermal predictor's temperature output. Material/interface/port outputs remain the retained model's own predictions.

This is an explicit architectural composition with two legal input signatures, not a claim of unchanged five-field output at attachment. New flow outputs are initially untrained. Thermal outputs should match their parent within existing numerical tolerances.

Legacy flow outputs may be retained behind a diagnostic option for comparison. They must not silently be selected for normal candidate output or form a confidence-based fallback. No hidden parent bypass can count as successful new flow prediction.

### 5.2 Exactly two new fitted flow arms

**D-sep:** the new flow reader receives only audited geometry and prescribed flow context. Heating is absent from its numerical input dependencies.

**D-open:** the same flow architecture has access to normalized own-module heating through additional declared input slots. This is the matched dependency control. It sees no observed temperature, target pressure or solved port value.

Both arms have identical parameter shapes and initial tensors. Represent the experimental heat slots explicitly; set the associated input-column weights initially to zero in both arms. D-sep supplies fixed zeros independent of the current h; D-open supplies source-local heating normalized using existing training-only statistics. All other features and normalization are identical.

Thus they start with the same flow function. D-open may subsequently learn a heat dependence; D-sep cannot. Do not add heat-null penalties to either new flow-reader objective: that would mix the structural comparison with another regularization experiment.

This compares dependency policies in the new reader. It is not an equally initialized architecture comparison against the mature Run3801 flow path; report that separately. D-open is allowed to learn to ignore heat and may match or outperform D-sep on field error. Do not deliberately weaken it or manufacture a leakage failure: the value of the structural restriction must be assessed alongside absolute accuracy and cost.

### 5.3 A small, source-resolved geometry flow reader

Use ordinary tensor operations and existing shared neural primitives. Initial implementation recommendation: hidden width128, message width64–128, two source-context updates, smooth activations and the existing normalized relative-coordinate Fourier features. Keep it small enough that its cold-call cost is a minor fraction of the retained thermal wrapper. These widths are starting design choices, not a claim that a particular capacity is optimal.

One practical form is

\[
z_i^0=E(g_i,c),
\]

\[
z_i^{\ell+1}=z_i^\ell+
U_\ell\!\left(z_i^\ell,
\frac{\sum_{j\ne i}m_j\psi_\ell(z_i^\ell,z_j^\ell,r_i-r_j,c)}{\max(1,\sum_jm_j-1)},c\right),
\]

\[
z(r)=b(r,c)+\frac{\sum_i m_i\,\phi(z_i,r-r_i,c)}{\max(1,\sum_i m_i)},
\qquad \widehat f(r)=D(z(r),r,c).
\]

The background term represents the prescribed channel/domain context. It is learned, not a hard-coded analytic wake or a reference-field lookup. All valid modules remain individually available through nonlinear relative-geometry messages before reduction. Source states already incorporate the rest of the layout, avoiding an isolated-donor model that ignores spectators.

The adapter determines which raw properties belong in g and c. Do not feed h-effective, surface temperature, thermal local-response summaries, heat-conditioned organizer states or case-relative heat scales to D-sep.

The branch need not introduce another hypergraph controller. Its purpose is a trustworthy benchmark flow dependency and an inexpensive learned control. Keep the retained thermal organizer as the representation under study. Source permutation, varying M and arbitrary in-domain receiver coordinates remain supported.

### 5.4 Shared core versus case-specific assumptions

Place the reusable small source-to-query geometry reader in shared core code; place the heat/flow capability and physical feature whitelist in ThermalChannel. The active case wrapper assembles the two lanes.

Do not give WindFarm a dummy heating rule or globally change the shared core to forbid thermal feedback. Preserve old checkpoint dispatch. A new opt-in model/config identity is appropriate because the deployed flow function and parameter inventory change.

The first implementation need not feed predicted flow values back into the retained thermal model. That feedback is already represented implicitly by layout/context in the retained thermal network. Do not claim a new explicit learned flow-to-thermal coupling that was not implemented. An explicit one-way coupling is a later architecture question, not an extra arm in this round.

## 6. Execution repair before adding more physical work

Keep this engineering work bounded to90minutes of active investigation/implementation, with real measured calls. Do not turn it into a general GPU-kernel project.

### 6.1 Remove work proven unnecessary in H-add

At the reviewed revision, `TensorQueryInteraction.components` constructs the full joint tensor even in `mode='add'`, then multiplies it by zero. In ordinary separable execution, compute S+R directly. Preserve identical parameters/checkpoint keys and compute I lazily only when diagnostics explicitly request it.

QM and QE use the same query-role access function for a given plan and receiver tile. Reuse that access within the current read call where exact receiver identity is established. Keep the cache local to the current wrapper/read computation; do not add global tensor caches or reuse autograd graphs across optimizer steps.

Do not drop QE receiver terms merely because they are source-constant **before** a nonlinear transformation. Preserve inner/outer tanh, attention normalization, source measures, masks and output-bias ordering.

### 6.2 Use reasonable receiver tiles

Measure a small set of existing supported receiver chunk sizes128/256/512 on actual low/high-M training inputs at Q1024/Q8192. Apply the same chosen numerical executor settings to every compatible timed reference. Record both the legacy128 and matched optimized comparison.

Choose from real output/input-gradient checks, peak memory and synchronized complete calls. Do not report only prepared-P2 speed, or compare optimized candidate timing to an intentionally unoptimized baseline without also showing the matched setting.

No custom Triton/CUDA kernel, arbitrary `torch.compile` campaign, source pruning, or FP16 conversion is part of this round. Existing FP32 numerical qualifications stay visible.

### 6.3 Flow preparation and repeated heat inputs

Prepare the geometry flow source states once per complete candidate call. Use a request-local prepared object to reuse them across query chunks. For repeated heat allocations at **fixed geometry and context**, that flow preparation and query result may be reused in an explicit `evaluate_heat_batch` path.

Cold preparation is still counted. Rebuild on changed geometry/context and preserve its derivatives. Do not key a reusable object only by case ID, M or input-array shape. Existing identity/version checks or a scoped object are sufficient; do not introduce a new hash service.

A geometry-prepared heat batch does not cache temperature predictions: each heat allocation still uses its own thermal computation. Report how much work is actually amortized. The temperature path may dominate, so no large speedup is promised.

## 7. Coding map and integration requirements

Reviewed paths to use as integration points:

- `Case_ThermalChannel/src/channelthermal/input_adapter.py`: explicit physical feature schemas; new flow-view construction must use named feature ownership.
- `Case_ThermalChannel/src/channelthermal/model.py`: opt-in composed model ownership, normalization and public forward signature.
- `Case_ThermalChannel/src/channelthermal/interface_field_coupling.py`: retained thermal P0/P1/P2, ports, prepared-state assembly and final output ordering.
- `Case_ThermalChannel/src/channelthermal/response_control/native.py`: design/context reconstruction and prepared role queries; retain physical input autograd.
- `src/honf_forward_core/interface_fields/tensor_query_interaction.py`: lazy unused-I computation and scoped access reuse.
- Shared `DensePairwiseField`, Fourier-feature and MLP primitives: reuse components without importing a second complete thermal model.
- Existing trainer, normalizer, checkpoint writer, stop-request and100-epoch reporting utilities.

Suggested new modules, subject to a concise implementation note:

- shared `geometry_flow_field.py`;
- case `dependency_policy.py` for audited flow inputs and capability;
- case `training/flow_dependency_fit.py` for the two narrow training scopes;
- one preparation/launch tool and one evaluation tool using the existing CLI/resource patterns.

Do not weaken the historical frozen-interface or response-refinement validators to accept this new experiment. Its flow-only training objective and new state inventory belong to an explicit new identity. Inherit thermal weights and normalizers exactly; initialize new optimizer moments only for new flow parameters.

The normal full forward must perform the real composed computation. A flow-only training method is permitted because inherited thermal parameters are frozen; show that its flow values match those returned inside a full native wrapper at identical inputs. This avoids paying for frozen Stage-A and all thermal query work at every flow-reader update.

## 8. Actual training protocol

### 8.1 Initialization and first real updates

Use seed0 and paired identical initial tensors. With the declared heat columns initialized to zero, verify the initial D-open/D-sep flow outputs match. Verify that the inherited thermal state, normalization and outputs are preserved.

Run at least one actual optimizer update per arm on a native low-M and high-M training batch. Confirm finite nonzero flow gradients and updated weights. These execution checks do not decide scientific quality or replace the500-epoch fit.

### 8.2 Fit schedule

The initial flow-only AdamW learning rate is3e-4, weight decay1e-5, gradient clipping at the existing norm1.0. Use FP32, Q1024, microbatch8 and effective48 where the maintained loader supports them. Set the same declared horizon and schedule for both arms. No independent tuning on the22 validation cases.

Each epoch visits all150 primary cases and performs the actual effective updates defined by the loader. The target is500 new flow-reader epochs. Review at100. Do not stop simply because the fresh small flow model initially trails a much older full predictor. Use training/validation trends, physical accuracy and remaining measured cost to decide a paired extension to1000 when warranted.

There is no new thermal response training or auxiliary heat-atlas expansion in the primary fit. The thermal backbone's historical150+three auxiliary-family exposure must remain documented. New flow normalization uses the same primary TRAIN source as the parent, never validation.

### 8.3 Loss and selector

Use equal-case, equal-channel normalized MSE over valid fluid receivers for u/v/p/omega:

\[
L_F=\frac{1}{4}\sum_{k\in\{u,v,p,\omega\}}
\frac{\sum_r w_r m_r(\widehat f_k(r)-f_k(r))^2}{\sum_r w_rm_r\,\sigma_k^2}.
\]

Use the existing primary TRAIN channel scales. Teacher flow predictions are not training labels. Heat-null labels are not added to this objective.

Select each arm using the minimum common all22 normalized flow score over saved100-epoch milestones. Report the exact500 endpoints separately if a different milestone is selected. The frozen temperature error must not be averaged in a way that hides a weak new flow model.

## 9. Evaluation: three distinct kinds of correctness

### 9.1 Absolute prediction

Run the ordinary composed forward on all22 cases. Report all24 existing roles without changing their definitions. Lead with fluid T, surface T, material T, u, p; retain v/omega, ports, maxima and module-count tails in an appendix.

Comparators: unmodified Run3801; its same-operator optimized executor; new D-open; new D-sep; retained G-fast and Dense-D25 numbers with their historical confounds disclosed.

Primary practical targets are flow means within5% and p90 within10% of Run3801, with unchanged inherited thermal/port predictions to the existing numerical qualification. These are prospective development targets, not significance tests or arbitrary physical-error floors. Report exact misses; do not tune the threshold after seeing data.

A weak fresh flow reader that happens to have zero heat leakage is not an improved forward model. Conversely, successful structural separation should not be dismissed merely because it cannot improve the thermal fields that were intentionally frozen.

### 9.2 Heat-null dependency

Use the existing all22 heat perturbations at0.10/0.20 feasible fractions, including M1 checks without a false fixed-total claim. Add full-Q8192 checks on the four representatives. Check individual source-heat changes as well as balanced transfers on a bounded subset, because constant-total tests alone can miss a dependence on total heat.

For D-sep, compare:

- two finite heat inputs with identical geometry/context;
- reverse-mode heat derivatives of flow channels and the8%-band pressure functional;
- actual prepared flow inputs and numerical source states at those two heat inputs;
- a same-input repeat to distinguish numerical nondeterminism from a physical-input dependency.

The mathematical dependency should be absent by construction. Report exact zero, roundoff-level repetition differences, or a genuine remaining path separately. Do not manufacture a tolerance-based90% pass. A disconnected heat derivative is mathematically zero; do not install gradient hooks to make an otherwise dependent function appear disconnected.

### 9.3 Geometry response and non-vacuity

Verify nonzero, finite geometry and allowed-context VJPs and bounded AD/FD agreement on two training layouts and the fixed representatives. Include actual receiver-coordinate gradients.

Use already saved geometry-move responses from the existing atlas as evaluation-only checks of the new flow reader where exact input/receiver joins are available. Pick a small input-declared panel, not the easiest signs. These are historical exposed references, not new solves or new independent evidence. Temperature/heat-only null rules must not be applied to these geometry changes.

A constant or case-ID-memorized flow function fails the purpose even when its heat-null test passes.

## 10. Thermal response and port diagnosis: bounded, not another fit

Retain the known0291 wrong thermal signs prominently. They are expected to remain when the thermal branch is unchanged. Do not present flow separation as their cure.

### 10.1 Source-backed linearity assessment

Inspect whether the **active** fixed-geometry thermal equations can be written as

\[
L(g,c)\,T=b_{bc}(g,c)+B(g)h,
\]

with heat-independent coefficients and boundary operators. Check coefficient clipping, temperature dependence, Robin parameters, input-dependent stopping and target extraction separately.

Where the source justifies this form, the underlying steady temperature solution is affine in h:

\[
T(r;g,c,h)=T_{bc}(r;g,c)+\sum_i K_i(r;g,c)h_i.
\]

This is a potential future modeling restriction, **not a property to impose blindly on every port quantity or on WindFarm**. The geometry-conditioned kernels can depend nonlinearly on the entire layout; linear forcing does not mean independent geometry effects. A maximum-temperature objective can be nonlinear even when point temperatures are affine.

Using saved plus/baseline/minus temperature arrays, measure central closure `T_plus + T_minus - 2*T_base`, keeping numerical convergence/storage warnings separate from a certified grid-error bound. Do the same for model predictions. Do not turn interpolated or algebraically combined outputs into newly solved labels.

### 10.2 Response excitation rank

For each fit/DEV/audit family, construct the matrix whose columns are actually solved heating increments and report its rank in the fixed-total subspace. Opposite signs of one transfer supply one direction, not two. M module heats with a fixed sum have M-1 free directions; field points and module-peak rows do not increase input-direction rank.

Explain which directional derivatives are measured and which are unidentifiable without an inductive model or further excitation. Do not declare the whole problem impossible: shared geometry-conditioned learning can transfer information across layouts, but this sparse evidence cannot uniquely identify a complete source-response operator at each layout.

Prepare one **unexecuted** follow-on reference request only if useful. An economical basis uses one baseline plus M-1 independent one-sided balanced heat directions; symmetric pairs cost more and test curvature. Include a common-mode direction only when varying total heat is part of the intended inverse task. Calculate the exact new attempt requirement and predicted CPU cost from existing timings. It requires a future explicit change to326/326; this plan grants none.

### 10.3 Port-path localization

Use the unchanged Run3801 on at most two existing training response families, with a maximum of24 additional complete wrapper calls in this diagnostic. Capture initial/refined port T/h, local temperature summaries and final thermal changes.

Where existing raw records contain the **correctly defined** outside T/h labels, a clearly labelled measured-port intervention may localize error. Never substitute surface temperature for outside temperature or q-proxy for h. Do not expose validation labels to training.

Other allowed fixed-weight diagnostics hold one model-side port/feedback component at its baseline value and recompute the remaining path. Such interventions localize model behavior; they do not reproduce a physical counterfactual. Report whether they alter the substantial response sign, not only a tiny aggregate MSE.

Stop after this bounded localization. Do not introduce a new port surrogate, heat-linear decoder or response-learning portfolio in this round. Write one evidence-supported next thermal-model question instead.

## 11. Complete runtime and memory measurements

Measure, on the same device and matched precision/chunks:

- B8/Q1024 actual M1 and M12, complete composed inference and heat/query input forward+VJP;
- B1/Q8192 actual M1 and M12, the same complete scopes;
- B1/Q8192 geometry-input forward+VJP on at least one high-M training case;
- flow-only prepare/read, retained thermal wrapper, and complete composed wrapper as overlapping scopes that must not be added twice;
- one explicit fixed-geometry heat batch using existing allocations, showing cold and reused costs separately.

Use five alternating warmed inference repetitions and at least two input-VJP repetitions per main condition. Keep model loading outside inference timers but inside the process resource ledger. Report allocated/reserved memory and the scope of any baseline subtraction.

Show candidate / optimized-Run3801 and candidate / measured-G-fast ratios. An engineering goal is at most1.20x the optimized retained full wrapper for cold Q8192 inference; no speedup is promised. If the new flow reader dominates, simplify its tensor operations or width once using training-input measurements, before scientific fitting or under a new recorded variant—not after inspecting final validation outcomes.

The high-M complete-field comparison is mandatory. Passing B8 alone is not a universal overhead result. Dense thermal fine rows remain dense; count new flow-source rows separately. A small flow head is not a sparse thermal executor.

## 12. Software and scientific checks

Reuse existing tests and add only checks required by the new behavior:

- explicit case capability and role-order mapping;
- no forbidden heating/port/local-response inputs in D-sep;
- identical D-open/D-sep initial tensors and functions at zero initialized heat columns;
- real finite heat changes and heat VJPs, not mocked zero responses;
- nonzero geometry/context/query gradients;
- source permutation, padding, M1, high-M, query chunking and prepared/full agreement;
- same thermal weights, normalization, Stage-A and expected temperature/port preservation;
- lazy-I and shared-access numerical/gradient comparison with the retained executor;
- actual optimizer update, epoch-boundary stop/save/resume and original artifact/security checks.

Do not broaden this into a campaign to repair all historical FP32 misses. Existing high-M generic-Global/G-fast VJP misses and q-proxy recovery qualifications remain separate. Do not relabel old failures using new output semantics or looser tolerances.

## 13. Milestones and bounded exploration

**M0 — source/dependency audit:** at most60minutes; read source and saved metadata, clarify applicability and exact neural paths. If the local generator is inaccessible, use available documented evidence only and explicitly limit any capability claim; do not invent source confirmation.

**M1 — executable candidate and exact reader simplification:** actual native forward/update checks, then measured early epoch/runtime forecast. Bound optimization investigation to90minutes. Never replace real execution with parser-only success.

**M2 — matched100:** complete100 flow-fit epochs per arm, inspect all22 flow accuracy, heat-null identity, non-vacuous geometry behavior and four actual field maps. Continue when the new reader is learning and the remaining forecast fits the budget.

**M3 — matched500:** complete the healthy pair to500. A single optional matched extension to1000 is allowed only when a clear late accuracy-learning question remains and the projected full round fits12 GPU-hours/8 elapsed hours. Do not spend spare budget merely to reach a larger number.

**M4 — full composed model review and thermal diagnosis:** all24-role preservation, complete runtime/VJPs, existing responses and bounded port/rank/linearity analysis. Produce the final report and an explicitly unexecuted next-data/model recommendation.

For a fixable implementation failure, try up to three concrete remedies or60minutes on that issue, whichever arrives first. Preserve failed attempts and state whether a remedy changes mathematics, numerical ordering, or only execution. A serious scientific negative result is not an instruction to keep adding methods.

If a scientific scope change would be needed, stop that branch, preserve its work and explain the unresolved decision. The other authorized branch and report should continue. No indefinite wait or silent hard-block retry loop.

## 14. Required report and figures

### 14.1 First page: decisions a reader can understand

Start with these plain statements, completed from measurements:

- **Predictor:** how accurately does the composed model reproduce fields, and which thermal quantities were deliberately unchanged?
- **Organizer:** which learned thermal organization was retained, what known dependency was encoded, and was any new organizing advantage actually tested?
- **Inverse:** which gradients/constraints are now better founded, and why the remaining thermal-direction/coverage limits do or do not allow broader use?

Then give one compact field table and one complete-cost table. Distinguish learned flow accuracy, structural heat-null correctness, and thermal response accuracy. Do not call a programmed dependency a discovered hyperedge or turn its success into a claim that the original HONF hypothesis is proven.

Use descriptive model names in the main text: retained thermal predictor, heat-independent flow reader, heat-enabled control. Put run IDs, tensor counts and detailed provenance in an appendix.

### 14.2 Necessary visualizations, at most five primary groups

1. Learning and measured cost of the two flow readers, with actual new update/case exposure.
2. Native u/p/omega fields and residuals for the four representatives, alongside one preserved temperature board to show what did and did not change.
3. A source-backed physical/neural dependency schematic plus actual heat-null and geometry-response comparisons. Label imposed architecture separately from learned source relationships.
4. Full-call fidelity versus time/memory, with cold and prepared heat-batch scopes separate and matched chunks.
5. Thermal-response evidence: unchanged0291 sign issue, saved central-closure/rank diagnosis, and any informative bounded port intervention. No newly generated inverse trajectory is drawn.

Show complete spatial maps when claiming a spatial region. Do not stretch the first128 saved receiver rows into a complete field; either reconstruct the full original query ordering or label a partial view precisely. Use common, unclipped physical scales where practicable and always compute statistics from unclipped arrays.

Each figure needs a direct explanation: what was measured, what improves/fails, and what it does not prove. Keep PDF masters and small Markdown raster companions under the existing local-artifact policy.

### 14.3 A/B/C conclusion

A: unique organizer value is **not the main new intervention**; do not infer it from improved flow independence. Preserve the current weak joint-versus-separable result.

B: give the actual flow and thermal dataflow, global/coarse/local paths, learned source reads, and measured numerical independence.

C: report measured flow/geometry generalization on exposed cases and the unchanged/diagnosed thermal-response limits. Distinguish new training scope from historical response exposure.

## 15. End-of-round disposition

A successful narrow outcome is an affordable composed predictor whose flow is accurate and structurally independent of heat, whose geometry sensitivity remains real, and whose strong thermal predictions are preserved. That repairs a concrete defect and creates a cleaner interface for later response learning. It does not by itself make the model inverse-ready.

If flow accuracy or complete cold cost fails, retain the current predictors and report the failed reader. Do not rescue the claim with a hidden parent fallback or generator call. The small matched dependency experiment still determines whether the proposed implementation is adequate.

If thermal source audit confirms a useful affine forcing structure, propose a later **geometry-conditioned, source-resolved thermal response operator**. Its kernels must depend on the whole layout/environment, not isolated donor tokens; its physical outputs, effective ports, nonlinear maxima and available excitation rank must be distinguished. Do not launch it automatically here.

Formal promotion remains a separate decision. Any future manual recipe must include measured complete training/validation epoch cost and estimated total wall time before launch; no configuration-only dry-run is an adequate resource forecast. The new flow-reader epoch cost is not a full-model formal-training estimate.

Finish by committing and pushing durable code/tests/guide/report on the non-default branch after the required entire-outgoing-history artifact audit. Preserve scientific checkpoints and generated data locally. Verify remote/local tips. Do not claim local process/GPU status unless actually checked at closeout.

## References and source-review map

[R1] `docs/reports/HONF_Response_Competent_Forward_Refinement_Report.md`, reviewed source9448df3. Authoritative field, response, null, cost and trainable-scope results for3801/3802.

[R2] `docs/guides/Thermal_Response_Refinement.md`, same source. Actual parent mapping, optimizer scope, data addendum and CLI behavior.

[R3] `Case_ThermalChannel/src/channelthermal/input_adapter.py`; `model.py`; `interface_field_coupling.py`; `response_control/native.py`, same source. Heat normalization, per-source features, prepared physical phases and public response assembly.

[R4] `src/honf_forward_core/interface_fields/tensor_query_interaction.py`, same source. Current C/S/R/I computation and query-access duplication, with dense source values retained.

[R5] `Case_ThermalChannel/src/channelthermal/training/response_refinement.py`, same source. Case-owned heat-null capability, actual fit/DEV/audit source populations, positive penalty and supervision limits.

[R6] `docs/reports/HONF_Interaction_Response_and_Inverse_Design_Study.md` and the later native recovery reports. Historical generator provenance and fixed-geometry heat-superposition evidence; not a substitute for reading the active local source.

[R7] `AGENTS.md` and `docs/guides/Thermal_Model_Development_Protocol.md`. Existing upload, reporting, checkpoint and development/formal separation rules.

**Status of this document:** proposed work, not completed experiments. Algebraic dependency reasoning is distinct from native numerical and physical evidence. No solver, training process or formal run was launched in preparing it.
