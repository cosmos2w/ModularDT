# HONF implementation round: directed packets and stable frozen inverse learning

**Mode:** Codex Goal-mode implementation and learning experiment  
**Repository / branch:** `cosmos2w/ModularDT`, `agent/honf-core-next`  
**Reviewed source:** `61e9d42d6d4c05991f55a33794a96f944b824877`  
**Primary evidence:** `docs/reports/HONF_Focused_Organizer_Utility_and_Predictor_Diagnosis_Report.md`  
**Delivery ceiling:** eight elapsed hours, twelve aggregate GPU-associated hours, two user-authorized GPUs  
**Default physical-reference budget:** zero new Thermal reference or Wind CFD solves

## 0. The decision for this round

The central goal remains:

> Learn a hypergraph-inspired organizer through multi-field reconstruction that encodes module–environment interactions and can be reused in flexible conditional inverse generation, with measurable generalization and interpretability.

The previous round was intentionally a diagnosis. It completed that diagnosis; its short duration is not evidence of poor execution. This round changes the deliverable:

> **Implement and train a better-informed interaction organizer and a numerically better-conditioned inverse model. Do not close the round after reproducing the known failures.**

A result need not be positive. It must include actual learning of the proposed remedies, native evaluation, and a clear reason for any unresolved failure. A new utility function, a toy test, a changed threshold, or another replay of the old inverse checkpoint is not the planned endpoint.

Do not consume idle compute merely to reach a time allowance. Finishing early is appropriate when the implemented models, planned learning comparisons, and meaningful evaluations are complete. If a short pilot fails and resources remain, attempt the bounded remedy authorized below rather than returning only “diagnose this next.”

### The two main deliverables

1. **Predictor/organizer:** a directed, geometry-initialized packet scorer that preserves physical axes and learns receiver-specific source needs through reconstruction. Train and compare it, rather than enlarging the failed action-risk head first.
2. **Inverse:** a new bounded-coordinate, velocity-parameterized denoiser/sampler, trained on actual completion tasks with a matched frozen-interface control. Repair numerical saturation before drawing further conclusions about graph reuse.

A smaller Thermal value-recovery comparison proceeds on the second GPU before its inverse work. It validates the existing value-weight hypothesis on a wider stored panel; it does not become another long maturation campaign.

---

## 1. What is known, what is inferred, and what is proposed

### 1.1 Findings established by the attached report

- Wind has adequate saved sparse actions in 13/18 primary cases, but the existing selector rejects them all. Removing its uncertainty margin selects an inadequate action in 9/18. The margin is not the only problem.
- The fit population for that selector contains only three independent training layouts. Direction rows, output roles, and query repeats are not extra independent layouts.
- In the six-case native replay, geometry controls match or beat the learned groups at equal or slightly lower MM pair work. A source-union control is not generally work-matched; its extra information must remain explicit.
- The current late Wind computation has full QE access. This round's first organizer experiment therefore targets **MM organization with other routes retained**, not a claimed module–environment sparsity breakthrough.
- Thermal G-u1300 loses to Run1804 in every one of the 16 inspected development absolute channel/case comparisons before masking. Increasing value weight from 4 to 8 gives a small incremental improvement over the weight-4 pilot, not recovery to the retained model.
- The four completed Wind inverse trials have different links and raw logits, but all decode to the same boundary corner. Their frozen denoiser received only 64 historical updates. Those trials do not establish that graph conditioning is useless.

### 1.2 New code-level findings motivating an implementation

**Coordinate-feature information loss.** `InputOnlyCoverOrganizer._spatial_summary` returns coordinate mean, absolute mean, RMS, maximum, and minimum. It is invariant to permutations of spatial axes. For example, `[2,0,0]` and `[0,2,0]` have identical summaries. The scorer applies this summary to source-minus-packet coordinates, while node geometry is also summarized in this fashion.

This does **not** prove that the whole network cannot distinguish those cases: other encoded states may contain positional or contextual information. It does show that the dedicated relative-geometry input discards a physically relevant distinction in an anisotropic flow problem. The new scorer must retain signed axis-specific information.

**Inverse conditioning of the reverse step.** The current 20-step epsilon-prediction implementation has terminal cumulative alpha approximately `1e-5`. With its actual clamped schedule, the final discrete alpha is about `0.00165026`. Consequently, a local noise-prediction error is amplified by about `24.58` in the first reverse mean, and by about `316.23` when algebraically converted to a clean-state estimate. This is an analytical property of the implementation, not a native attribution of every saturated sample to the schedule.

The current model then turns unbounded design logits into positions using a sigmoid. Distinct large logits can therefore produce indistinguishable float32 coordinates. Weak training, target scaling, and conditioning can contribute as well; the repair must address the representation and training, not just clip a displayed coordinate.

### 1.3 Proposed changes are experiments, not established improvements

The directional scorer, module-receiver indexing, geometry-residual initialization, bounded state representation, velocity target, and budgets below are new choices. Their expected benefits must be checked through actual models. Keep all historical checkpoints and behavior loadable.

---

## 2. Work organization: progress without another day-long campaign

### 2.1 One clock and two active lanes

Resolve the two currently authorized devices, historically physical GPU1 and GPU2. Do not assume logical CUDA ordinals after visibility remapping. Do not interrupt unrelated processes.

- **Lane A / GPU1:** Wind directed-packet implementation, matched training, and native evaluation.
- **Lane B / GPU2:** bounded Thermal value-recovery validation, then stable inverse implementation/training. The primary new inverse learning task is Wind one-position completion; the sampler module remains reusable for Thermal later.

This is deliberate unequal parallelism. Both physical datasets progress, while the inverse defect is repaired once in shared code. Do not launch an additional large Thermal diffusion campaign in the same round.

### 2.2 Milestones

- By 45 minutes: verify code findings with small executable checks, locate existing inputs/checkpoints, and begin implementation. Do not repeat the old selector and attribution surveys.
- By two hours: each main lane has completed a real optimizer step and a native read/sample with its proposed model, or records a specific active remedy.
- By four hours: actual learning curves and at least one post-training native comparison exist. Decide how the remaining learning budget is used.
- By six hours: select candidate weights using the declared training/development procedure and run final bounded comparisons.
- By seven hours: no new optimization or sample campaign. Finish selected figures, tests, report, and ordinary repository closeout.
- By eight hours: deliver. No silent rollover into another night.

These are workflow targets, not permission to fabricate completion. A true blocker is reported with the failing operation and the attempted remedies.

### 2.3 Compute allocation

| Work | Aggregate GPU-associated ceiling |
|---|---:|
| Wind packet pair, including native evaluation | 4.5 h |
| Thermal value-recovery pair and broader replay | 1.0 h |
| Stable inverse pair and selected samples | 3.5 h |
| Shared tests, remedial pilots, final evaluation reserve | 3.0 h |
| **Overall ceiling** | **12.0 h** |

Elapsed eight hours and aggregate twelve hours both apply. Count failed attempts once. Use measured recent step/draw time to choose blocks; do not require multi-hour worst-case reservations before a short useful block can run.

Per-arm update targets are meaningful learning allocations, not minimum amounts to burn:

- Wind: 1,000 updates per matched arm; at most 1,500 if learning remains credible and fits the common ceiling.
- Thermal: 240 updates per arm, with a review at 120; at most 360 under the same one-hour allocation.
- Inverse: 1,000 updates per matched arm, reviews at 200/500/1,000; at most 1,500 if needed and affordable.
- Diagnostic/remedy optimization: at most 400 extra updates **in total**, not per failure.
- At most 192 final inverse trajectories, including all conditioning interventions and retries. Save each completed trajectory immediately.

Prefer reducing the final sample matrix before cutting a demonstrably useful learning run back to 32 or 64 updates. No new solver calls; no automatic 5,000-epoch continuation.

---

## 3. Baselines and data boundaries

### Wind

- Main frozen physical initialization: Run2112 **G u4910**, complete native model.
- Retained controls: the same G in full access; archived G packet actions; Run2112 P u4910; Run2110 W-full u1500.
- Targets: stored OpenFOAM CFD velocity values in m/s. Input axes must be documented in the native wind-aligned frame; `D=80 m` is the current dataset rotor diameter.

The new experiment begins from the same G physical weights in both trained arms. Comparing their outputs isolates the new organizer feature treatment more cleanly than restarting two different physical models.

### Thermal

- Start the paired correction from G-u1300 in full access, with identical copies and retained Adam state as in the focused pilot.
- Keep Run1804 e4738, G-u1300, and the existing 40-step weight-4/weight-8 results as frozen controls.
- Do not use the small pilot's two exposed development cases as an untouched generalization set.

### Data partitions

Construct a small input-selected Wind learning cohort from at least **30 eligible training layouts**, including their available directions and a range of M. Keep every established organizer-development layout excluded. Use twelve development layouts, chosen from existing eligible development metadata before the new errors are read, for the main evaluation. Previously inspected layouts remain exposed development evidence.

Use fresh native training query samples rather than repeatedly optimizing one Q1024 table. Default per-update Wind query count is approximately 512, split between volume and protected roles with their documented weights. Evaluate with two disjoint native Q2048 panels, and two outcome-independent field slices. Preserve original integration weights and distinguish volume from role-balanced sampling.

If 30 suitable layouts cannot be assembled, fix a metadata/data-loading problem before shrinking silently to three. State any actual coverage shortfall.

For Thermal, use all eight existing training response families plus broad historical value replay. The fixed primary development includes the two other existing Re90 families not used by the last focused pilot, then reports the complete four-family calibration panel and at least sixteen metadata-stratified historical cases. These are additional cases for this remedy, not new independent physics or necessarily untouched layouts.

---

## 4. Main model upgrade: directed geometry-residual packets

### 4.1 Scope of the first trained candidate

Target **MM receiver organization** first. It is a consequential native route and is the route with a measured geometry-control advantage in the latest report. QE/ME/EM/QM remain explicitly full in the primary comparison.

Do not advertise this restriction as a full HONF solution or a whole-model speedup. The purpose is to obtain one learned grouping mechanism that demonstrably adds value before extending it to environmental and interface receivers.

The reusable feature/scorer code must support 2-D and 3-D and typed phases, so it can later be applied to ME/EM/QM/QE. An additional environmental restriction is not part of this round's formal training.

### 4.2 Packet receivers must be the receivers of the mechanism

For the opt-in MM candidate, build the candidate hierarchy from **active module receiver coordinates**, rather than letting hundreds of unrelated environment anchors determine MM cuts.

Use the existing `CaseLocalReceiverTree` implementation and physical frame; maximum depth two is sufficient. Adjust minimum leaf count to the actual module population; do not pad fake receiver nodes to obtain K=4. Keep the requested cut, realized cut, and nonredundant effective action count distinct.

This is a mechanism-local index. It does not redefine the physical quadrature or claim that geometric leaves are physical factors. Untargeted mechanisms retain full native computation and must not accidentally inherit an out-of-envelope MM mask. A shared plan object is acceptable only when tests establish that separation; otherwise add a thin typed policy wrapper, not another full architecture stack.

Record the native MM receiver IDs and the source permission used for each receiver. A centroid in a figure is not the native receiver axis.

### 4.3 Preserve signed directional geometry

Let `x_s` be a source, `c_e` the receiver packet centre, and `B(c)` an orthonormal physical frame given by the case adapter. Wind's saved coordinates are already wind aligned: verify conventions and do **not** rotate them a second time. For Thermal use the declared channel frame.

Define

\[
\xi_{es}=B(c)^\top(x_s-c_e)/L_*,
\]

with `L_*` a declared physical scale such as rotor diameter, not an outcome-fitted scale. Retain separately:

- signed downstream, crosswind, and vertical components;
- absolute component values and squared distance;
- packet extents on each physical axis;
- source and receiver physical type/role;
- source state, packet summaries of its actual receiver states, and available global operating context.

For a 2-D adapter use an explicit absent-axis mask; do not treat a padded zero z as a real observed vertical effect. Invariance should mean consistent transformation of geometry and its physical frame—not invariance to arbitrary swapping of streamwise and vertical coordinates.

A simple feature vector is

\[
\psi_{es}=[\xi_{\parallel},\xi_\perp,\xi_z,
 |\xi_{\parallel}|,|\xi_\perp|,|\xi_z|,
 \|\xi\|^2,\mathrm{extent}_{e,1:3},\mathrm{axis\_valid}_{1:3}].
\]

Do not impose “upstream only” as a universal physical law. Pressure and contextual dependencies can be nonlocal. Directional features are information, not hard causal exclusions.

### 4.4 Start from a sensible geometry selection, then learn corrections

For a packet's actual receivers `R_e`, use the geometry prior

\[
 g_{es}=-\sum_{i\in R_e}\omega_{ei}
          \|B(c)^\top(x_s-x_i)/L_*\|^2,
 \qquad \sum_i\omega_{ei}=1.
\]

Normalize its score scale using the eligible current-input sources, with a fixed numerical floor and detached normalization statistics. This affects conditioning, not physical quadrature.

The learned score is

\[
\boxed{\ell_{es}=\bar g_{es}
        +a(u)\,f_\phi(z_e,z_s,c,\psi_{es})},
\]

where `a(u)` rises smoothly to one in the first 100 updates and the final residual layer is initialized to zero. Initial permission therefore matches the **new shared-packet geometry prior exactly**.

The old per-receiver geometry control is not generally identical to this shared-packet prior. Keep it separately as a direct geometry comparator; do not claim parity between the two.

Use the existing corrected hard-value / soft-organizer training bridge and exact budget projection. Default MM capacity is 0.90. Every full-access replay must use exactly full hard and soft permissions. Retain self-exclusion, valid masks, original source measures, and native reduction normalizers.

All fine source states remain separate. No group-value decoder, source pooling before nonlinear messages, added PDE head, or coarse correction branch is introduced.

### 4.5 Matched trainable control

Train two small organizer variants from the same physical G weights and same random state:

- **W-dir:** explicit directional features above.
- **W-summary:** the same new receiver index, geometry prior, source features, network capacity and training stream, but replace each signed coordinate vector by a padded copy of the legacy axis-symmetric summaries.

Match input widths by padding/masking, not by adding useful information to one arm inadvertently. Both final residual layers start at zero, so initial hard actions match.

This pair isolates the value of retaining directional information **within the new geometry-seeded formulation**. Comparing either arm to the archived old organizer is a broader architecture/initialization comparison and must be labelled accordingly.

During updates 1–300, freeze the entire physical predictor and train only the new scorer/receiver embeddings. During updates 301–1,000, permit the existing MM message/update modules to adapt at a smaller learning rate, identically in both arms. Keep encoders, field head, local/coarse branches and other fine mechanisms fixed initially. Use the same stored physical loss/normalization as the native workflow and preserve full-access anchor replay.

Initial rates: scorer `1e-3`, newly unfrozen MM modules `1e-5`; adjust once only if measured gradients/learning curves justify it. Do not load stale optimizer moments into newly shaped feature layers. Historical checkpoint loaders remain unchanged.

Use 25% full-access replay and 75% MM-restricted updates. In restricted updates cycle through every distinct nonempty depth-two cut, including asymmetric three-packet cuts when available. Deduplicate exactly equivalent actions and record actual per-layout/per-action exposure. Do not silently train only root/two/four and test many unseen alternatives.

### 4.6 A learned partition must earn its complexity

After training, evaluate all actually trained small-cut candidates on the fit population and on disjoint native queries. Full prediction remains

\[
\widehat y=F_\theta(d,c;H_\phi(d,c;a)).
\]

Report the split benefit at fixed weights and comparable selected native MM work:

\[
\Delta_r(a\to a')=E_r^2(a)-E_r^2(a').
\]

Keep separate the gain due to source selection and the gain due to grouping. Compare:

1. full access;
2. frozen packet geometry prior;
3. trained W-dir and W-summary;
4. source-union collapse;
5. direct geometry at the same eligible pair count;
6. an effective receiver/source rewire, only when it changes access and its work is matched or disclosed.

No new 75% stress training, sparsity penalty, entropy objective, or requested-K histogram is authorized.

### 4.7 Input-only adaptive K: train a small choice only after the actions learn

Rebuild action labels at the new selected physical/scorer weights. There are now roughly 30 training layouts rather than three. Reuse the action-aware risk features and regression infrastructure, adding the signed physical frame and actual receiver features; do not invent another large selector architecture.

Fit one small action model, with layout-held cross-validation inside the training population. First measure its ability to rank the realized split benefits and predict role risks. Then report:

- input-only minimum predicted-risk research choice, with K as a consequence;
- the unchanged conservative historical acceptance/fallback rule;
- the reference-measured oracle, used only for evaluation.

Do not call an unsafe research choice deployment. Do not choose margins using development labels. Missing adequate choices on one layout are a legitimate full-access result, not a reason to force another K.

A useful adaptive-K result requires distinct **nonredundant** computations selected in response to inputs and positive measured utility on some case classes. K varying only because M changed is a weaker result and must be stated. If all useful choices collapse to one common source set, report that rather than manufacturing diversity.

Do not rerun the entire old selector audit. The new fit must use evidence from the newly trained computations.

---

## 5. Secondary predictor work: test the Thermal value-balance hypothesis properly

This work should remain small and decisive.

Resume two **new temporary copies from G-u1300**, using the same optimizer/RNG/sampler state, full access, frozen route parameters, and no organizer shadow. Use the maintained physical parameter scope from the focused pilot.

Only value weight differs: 4 versus 8. The other supervised terms and anchor coefficient remain the same. Use all eight training response families and a rotating historical-value stream; do not spend every update on the two recently inspected families.

Run 240 actual updates per arm, with the same case, query and history stream and reviews at 120/240. Persist independently resumable state every 50 updates using ordinary existing checkpointing. The cap is one GPU-hour including replay; shorten the proposed block based on measured throughput, not a large fixed reservation. No immediate new route learner is trained on Thermal.

Evaluate unchanged G-u1300, the two new arms, and Run1804 on:

- the two additional Re90 families before aggregating the full four-family calibration;
- sixteen metadata-selected historical cases with module-count coverage;
- absolute fluid/interface/material roles;
- finite per-module peak, field response and pressure quantities;
- worst-case values and stored response resolution limits.

Use paired family-level comparisons. A few percent gain versus weight 4 is not proof of physical recovery. State whether value8 repairs existing drift, merely changes the tradeoff, or fails to transfer. Do not resume training only to satisfy an arbitrary historical-visit counter.

If the value8 benefit disappears or tails worsen on the broader panel, terminate that hypothesis and retain Run1804 as the operational reference. The independent Wind and inverse lanes continue.

---

## 6. Primary inverse implementation: stable design states and velocity prediction

### 6.1 Keep the old model as a historical baseline

Do not relabel an epsilon-prediction checkpoint as a velocity-prediction model. The old u64 checkpoint and sampler remain available for exact replay. New target, schedule and state representation require a new opt-in model/configuration and fresh inverse fitting.

Limit the initial task to **one hidden turbine**, with visible geometry and operating conditions fixed and sensors selected independently of the hidden position. Do not add two-hole completion or heat generation before this small task works.

### 6.2 Replace unbounded position logits by bounded normalized clean coordinates

Let the **supplied, target-free task bounds** be `l,u`. Encode the hidden clean design as

\[
 z_0=2(x-l)/(u-l)-1,\qquad z_0\in[-1,1]^2.
\]

Decode a clean estimate affinely:

\[
 x=l+\tfrac12(z_0+1)(u-l).
\]

There is no sigmoid whose float32 saturation turns different logits into the same design. This is not permission to use true hidden positions to set `l,u`.

Use an existing public or training-derived supported domain. If the proposed common domain excludes layouts, define and report the restricted cohort before fitting and apply the same inclusion rule to both arms. Never clamp clean training targets into a different domain and still call them the original targets. Row-native bounds derived from hidden clean geometry are evaluation information unless the task explicitly supplies them independently.

Normalized noisy diffusion states are allowed outside `[-1,1]`. Do not clamp the entire noisy state and silently change its training distribution. A bounded **proxy** is used to construct current-candidate links during both training and inference; label it as a proxy. A projection at the final clean state is allowed, but record how often it is active and show pre-projection overshoot. Projection can enforce box validity; it cannot demonstrate learned observation conditioning.

Use `clip(z_t, -1, 1)` as the default provider proxy, without clipping the state seen by the denoiser. This proxy is identical to a valid clean coordinate at the clean endpoint. A smooth proxy is an optional numerical remedy only if it also tends to the identity on valid clean states as noise tends to zero; a time-independent `tanh(z_t)` would move even a valid clean design and is not an equivalent replacement. Apply the chosen proxy identically during training and inference. The frozen provider does not receive clean target coordinates.

Visible module states must be restored exactly after every step. The noise mask has zeros on visible/invalid slots; do not insert visible coordinates into a quantity described as standard Gaussian noise.

### 6.3 Stable diffusion target

Use noise time `t in [0,1]` with

\[
 a(t)=\cos(\pi t/2),\qquad b(t)=\sin(\pi t/2).
\]

Assign exact `(a,b)=(1,0)` and `(0,1)` at the endpoints. This is a fresh schedule, not an in-place mutation of an old saved beta array.

For generated slots,

\[
 z_t=a_tz_0+b_t\epsilon,
 \qquad v_t=a_t\epsilon-b_tz_0.
\]

Train the existing spatially conditioned architecture to predict `v`, using the design mask and uniformly sampled time/noise. Stratify a modest fraction of training examples at or close to the pure-noise endpoint so it receives actual fitting exposure. The value at `t=1` is `v=-z_0` with the convention above.

Recover

\[
 \widehat z_0=a_tz_t-b_t\widehat v_\theta,
 \qquad
 \widehat\epsilon=b_tz_t+a_t\widehat v_\theta.
\]

These formulas contain no division by tiny `a_t`. A deterministic 20-step DDIM-style update is

\[
 z_{t'}=a_{t'}\widehat z_0+b_{t'}\widehat\epsilon,
 \quad t'<t.
\]

Random initial noise still gives multiple samples; a deterministic reverse path is not a deterministic conditional distribution. Save exact time labels and per-step state/clean-estimate norms.

Use no classifier-free guidance, external forward-gradient guidance, or post-hoc objective optimizer in the primary repair. Those would obscure whether the conditional denoiser itself works.

### 6.4 Geometry projection and scientific interpretation

Prefer reporting the raw clean estimate and its final feasible-box projection separately. If projection is fed back into intermediate DDIM updates, reconstruct a consistent noise component from the projected estimate with a numerically safe formula and test the modified sampler; do not mix inconsistent `(z0,epsilon)` pairs without documenting that variant.

A one-hidden-turbine candidate must also pass clearance against every visible turbine. Do not repeatedly redraw until every candidate is valid and report only successes. Keep rejected draws in the denominator. A boundary pile-up after projection is still an inverse failure even though sigmoid saturation has disappeared.

Use the same support and collision rules for graph and dense controls. Native-grid or stored-surrogate acceptance is not new-layout CFD validity.

### 6.5 Train the inverse model, not just its sampler

Use at least 24 eligible training layouts, multiple hidden-slot choices where available, and their stored sensor measurements. Keep layout grouping and all observed/held sensor roles intact. Input-only normalization is fitted on training values; do not fit a scaler per development task using its target.

Train matched **I-v-G** and **I-v-dense** from the same fresh initialization and schedule, using one predeclared frozen forward/scorer source. To avoid making this lane wait for Lane A, use the existing u4910 source and explicit two-packet research action as the primary provider. That action's forward adequacy is limited; report this inherited limitation. A later checkpoint-only replay with the new trained organizer is optional and is not an inverse retraining result.

Graph and dense arms use the same frozen candidate embeddings, known inputs, model capacity, initial states, and noise streams. Only the supplied access differs. No encoder/organizer weight receives an inverse gradient. An effective link change is checked on actual inputs, not assumed from different action names.

Initial inverse optimizer: AdamW, learning rate `1e-3`, modest standard weight decay from the current inverse recipe, width 96 and existing layer count. Use batch 4 when supported; otherwise execute exactly equivalent small batches and state the actual sample exposure. Default 1,000 updates per arm, not 64. Review at 200 and 500; extend to at most 1,500 only for a credible learning trend inside the time budget.

Training loss is the masked `v` loss. Add a small clean-coordinate loss only if a bounded training-only check shows a real benefit; then treat the changed recipe explicitly. Do not add a frozen-surrogate observation loss as if it were CFD reference. Candidate field discrepancy remains evaluation unless a separately labelled teacher-consistency remedy is necessary.

Before broad fitting, overfit at most eight training tasks for up to 100 diagnostic updates. Failure to improve clean-coordinate estimates means repair target/time/conditioning/normalization first. Those updates count toward the shared remedy budget. A tiny deterministic conditional-regression head is allowed as an information/conditioning control, not a replacement claim for generative design.

### 6.6 Measure conditional quality, not merely nonsaturation

Preselect eight development tasks and four noise draws per task per main arm: 64 as-observed trajectories in total. Add at most 64 paired observation/graph intervention trajectories and reserve the remaining trajectory ceiling for repairs or a small second checkpoint. Do not reproduce a 576-trail matrix.

Required measurements:

- raw and projected clean-state magnitude at every time;
- fraction of terminal values requiring projection;
- native-supported geometry and rotor-clearance rate;
- normalized-position error against the stored hidden target, descriptive rather than a unique-solution criterion;
- observed and held-sensor frozen-surrogate RMSE;
- improvement in matching changed observations relative to simply reusing the unchanged-condition sample;
- identical-terminal fraction across graph/dense and condition pairs;
- sample spread, reported jointly with validity and sensor fit;
- effect of an active link intervention at the same weights and noise.

Different samples or changed outputs alone are not conditional quality. The replacement should stop the numerical corner collapse **and** learn useful information from the observations. Sparse observations may admit nonunique designs, so missing the stored hidden point does not by itself invalidate a physically consistent alternative. Without new CFD, alternative-design physical consistency remains unverified.

A scientifically successful inverse repair may be shared by both graph and dense arms. It does not automatically demonstrate a graph advantage. Report the two conclusions separately.

---

## 7. Implementation map and tests

### Existing source to extend, not rewrite wholesale

| Area | Reviewed file / reusable seam |
|---|---|
| Organizer geometry and source scoring | `src/honf_forward_core/interface_fields/input_cover_organizer.py` — `_spatial_summary`, `_node_embeddings`, `_score_sources`, `score_cases` |
| Mechanism-local receiver index | `src/honf_forward_core/interface_fields/adaptive_interaction_cover.py` — `ReceiverAnchorUniverse`, `CaseLocalReceiverTree`, typed permissions |
| Hard capacity and frontier | `src/honf_forward_core/interface_fields/budgeted_frontier.py` |
| Action description / risk | `src/honf_forward_core/interface_fields/action_aware_frontier.py`, `action_risk_fit.py` |
| Native Wind fitting / replay | `Case_WindFarm/src/windfarm/workflows/maturation.py`, `joint_forward.py`, `train_forward.py` |
| Native Thermal response objective | `Case_ThermalChannel/src/channelthermal/response_control/maturation.py` and the maintained paired-fit workflow used by the report |
| Shared inverse model | `src/honf_inverse_core/models/frozen_packet_diffusion.py` |
| Wind task bounds/provider | `Case_WindFarm/src/windfarm/inverse/packet_completion.py` |
| Thermal frozen provider, for later reuse | `Case_ThermalChannel/src/channelthermal/inverse/packet_reuse.py` |

Prefer a small opt-in directional scorer and a small opt-in velocity sampler rather than adding flags throughout historical classes. One reusable training entry point per main experiment is sufficient. Avoid another multi-thousand-line all-purpose study runner.

### Essential tests

1. Legacy coordinate summaries collide under axis permutations, while the new physical-frame feature distinguishes streamwise and crosswind offsets.
2. Jointly rotating coordinates and their declared physical frame preserves new features; module permutation preserves outputs/support up to permutation.
3. Zero residual reproduces the new geometry-prior hard action; it does not claim equality to the different old direct geometry mask.
4. MM-only trees cannot change unbudgeted QE/ME/EM/QM semantics; all-access recovers the native model.
5. Self-exclusion, source measures, source IDs, exact work counts and actual receiver counts are preserved. No comparison mixes packet-incidence and native pair axes.
6. Split initialization does not discard or average fine source states; malformed empty or degenerate cases are handled by existing numerical primitives.
7. Controlled `v`/`epsilon`/clean-state conversions reconstruct a known noisy state, including both endpoints, and an oracle sampler recovers bounded targets.
8. Velocity checkpoints cannot be silently loaded as epsilon checkpoints or vice versa; historical checkpoints still load normally.
9. Visible states remain exact; generated-slot noise has the intended distribution and masks.
10. Domain bounds and provider input do not depend on hidden clean positions; poisoning the hidden target leaves candidate links unchanged.
11. Observation permutation and value-at-fixed-position checks still pass in the shared spatial conditioner.
12. A native real optimizer step and a completed persisted native trajectory exist for each new learning path. Tests do not replace training.

Run focused tests while developing. Do not spend the round expanding general security or data-contract infrastructure.

---

## 8. What Codex may try without another user request

Each main lane may use at most two substantial remedies inside the stated budgets. Choose from actual symptoms:

- **Directional scorer not learning:** inspect the zero-initialized residual's first nonzero gradients, eligible masks, sampled cut exposure, and shadow detach. Increase scorer learning rate once or unfreeze the stated MM scope after a measured fitting limitation.
- **Native model forgets:** reduce the MM physical learning rate or lengthen the scorer-only phase. Do not remove historical full-access replay or silently switch the retained reference.
- **Requested cuts collapse:** inspect actual receiver anchor occupancy and permission equivalence. Correct an inappropriate shared receiver index; do not fabricate split nodes or distinct K labels. If routing a separate MM index requires a large core refactor, the authorized fallback is to train the directional-versus-summary pair on the existing index first and label it a feature-only candidate. Spend at most 45 minutes on that integration obstacle before making this choice; do not return without the otherwise executable scorer training.
- **Risk model rejects everything:** separate raw ranking from margin rejection on train-held folds. Report research choices and the unchanged historical conservative policy separately. Do not tune development thresholds to force sparse deployment.
- **Inverse remains large or saturated:** inspect target range, exact terminal endpoint, current-state versus clean-estimate decoding, loss masking and noise coefficient conventions. Try one smaller learning rate or a direct-clean target instead of velocity; label the new target and train it, rather than reinterpreting an old checkpoint.
- **Conditioning weak:** overfit the declared small training panel and check sensor channel scales and location/value pairing. A scalar latent norm fix cannot recover missing sensor information.
- **Sampling slow:** batch identical-shape cases and cache only genuinely immutable geometry/sensor descriptors. Recompute design-dependent states, scores and links. Complete and persist a draw before a larger matrix.

Resolve routine API, path, batch, and serialization problems locally. If a genuinely hard block survives these attempts, give a minimal reproducer, exact scope, time spent, and which evidence remains valid.

Do not return only the pre-existing conclusion when the proposed learning task is healthy and has not yet received its authorized updates. Conversely, do not continue a demonstrably wrong objective just to meet an update target.

---

## 9. Criteria for judging the round

### Predictor

Report selected sparse and same-weight full outputs, the geometry initialization, matched summary ablation, and retained native references on identical receivers. Improvement on training only is fit capability, not transfer. Preserve all five Wind roles and the distinct Thermal field/interface/material/pressure/peak quantities.

### Organizer

Report separately:

- input-conditioned hard source selections;
- any incremental value of explicit directional information;
- extra value of shared groups over simple geometry at comparable work;
- oracle availability versus learned selection;
- requested and nonredundant K;
- aggregate work and measured executor rows;
- remaining dense/global information paths.

A geometry-seeded, reconstruction-refined group is not a causal physical edge. This round may establish a better computational organization while physical interpretation remains provisional.

### Inverse

Report separately:

- numerical sampler repair;
- learned observation dependence;
- validity and sensor-consistency changes;
- the incremental graph-versus-dense result;
- absence of generated-layout CFD validation.

An improved stable dense inverse is useful infrastructure, but does not close the organizer claim. An active link with no measurable output/quality benefit is a negative reuse result, not a reason to count link changes as success.

There is no universal all-or-nothing promotion gate that prevents reporting progress in one system because another remains inadequate. Existing production/default policies remain unchanged unless the normal, separately justified release boundary is crossed.

---

## 10. Required final report and visualizations

Begin the final report with **one page in ordinary language**, answering:

1. What did we implement and train that did not exist before?
2. Predictor: what improved, what remains weak, on which cases?
3. Organizer: did directional information and grouping matter, and did any useful case-dependent K emerge?
4. Inverse: did sampling stop collapsing, did conditions matter, and did the graph help?
5. What is the one next justified investment, rather than a list of five unfinished pipelines?

Then provide the quantitative appendix and a small figure set:

- **Learning and time:** train/development error against actual updates, case exposure, and GPU time. Mark the scorer-only/co-adaptation transition.
- **Physical fields:** one representative and one difficult native Wind slice; at least one Thermal interface/material comparison. Same coordinates, units, and unclipped numerical metrics.
- **Directed interaction picture:** source/receiver coordinates in the physical frame, full geometry prior, learned directional support, and summary ablation. Show real omitted/retained links, not a centroid star whose axis is mistaken for native receivers.
- **Grouping utility:** fidelity versus matched eligible work, root/union controls, selected K and role failures. Show full-access fallback explicitly.
- **Inverse repair:** per-time raw clean estimates and actual geometric trails for old/new samplers, observed/held sensor residuals, boundary projection rate, and at least one failure.
- **Compact outcome summary:** predictor/organizer/inverse gain and miss table, completed learning counts, resource use, and censored work.

A changed-observation picture must state whether the changed observations have an independent physical realization. Denoising trails are not optimization progress. New generated layouts have no CFD reference in this round.

Figures must use saved measured arrays, be visually inspected, and follow AGENTS.md: local PDF masters with a small raster companion only when needed; no generated figures or one-time renderers in Git. Do not create a huge gallery or regenerate historical figures that add no new information.

---

## 11. Completion and preservation

The intended completed round includes:

- a maintained opt-in directional geometry-residual scorer and mechanism-appropriate receiver index;
- a real trained Wind directional-versus-summary comparison and new action evidence;
- the small broader Thermal value4/value8 result;
- a maintained bounded-coordinate velocity sampler and genuinely trained inverse pair;
- at least one complete selected result per main arm, not just CPU smoke;
- focused tests, readable figures, quantitative results, and the three-goal explanation.

Positive science is not guaranteed. If a remedy fails after adequate executed learning, preserve the negative result. Do not claim completion of a learning task from code availability or startup parity.

Keep all existing security, checkpoint trust and artifact measures. Do not introduce new cryptographic fingerprints, contract freezes, baseline snapshot frameworks, approval databases, or monitoring daemons. Reuse normal Git history, existing checkpointing, ordinary configs and tests. Do not dilute existing safeguards to save time.

Commit durable code, reusable tests, small configuration changes and the report to the existing non-default branch. Audit the full outgoing range, run the existing pre-push hook, and verify remote/local tips. Raw checkpoints, arrays, and one-time diagnostics remain local. If external push fails, report it; do not misstate completion.

No automatic further training or physical solves follow this report.

---

## 12. Sources and exact scope of the proposal

### Project sources

- `docs/reports/HONF_Focused_Organizer_Utility_and_Predictor_Diagnosis_Report.md`, especially Wind grouping, Thermal paired correction, and inverse saturation.
- `docs/reports/HONF_Controlled_Maturation_and_Action_Aware_Organization_Report.md`, for u4910/u1300 identities, data exposure, and historical acceptance rules.
- `src/honf_forward_core/interface_fields/input_cover_organizer.py` at `61e9d42`: `_spatial_summary`, node/source encoders and pair features.
- `src/honf_inverse_core/models/frozen_packet_diffusion.py` at `61e9d42`: cosine schedule, epsilon training target and reverse update.
- `Case_WindFarm/src/windfarm/inverse/packet_completion.py`: target-free provider and domain/task conventions.
- `AGENTS.md`: existing reporting, artifact and security rules.

### External methodological sources

- Salimans and Ho, **Progressive Distillation for Fast Sampling of Diffusion Models**, ICLR 2022, arXiv:2202.00512. Relevant here for velocity parameterization; this plan does not perform progressive distillation or inherit its image-generation results.
- Song, Meng and Ermon, **Denoising Diffusion Implicit Models**, ICLR 2021, arXiv:2010.02502. Relevant for the deterministic reverse-update construction; no published image speedup is transferred to this code.
- Lin et al., **Common Diffusion Noise Schedules and Sample Steps are Flawed**, WACV 2024, arXiv:2305.08891. Relevant for endpoint consistency and velocity training. The current HONF schedule's effect is calculated from its own code rather than assumed identical to the paper's examples.

The optional companion `HONF_Implementation_Algebra_Checks.py` checks the coordinate-summary collision, actual old schedule coefficients, and velocity-transform identities. It does not load any native checkpoint or establish a physical gain. Codex must execute the native tests and trained comparisons above.
