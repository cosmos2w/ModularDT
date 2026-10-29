# HONF Round13 — Spatial group functions and task-supervised refinement

**Prepared:** 25 October 2026. **Source reviewed:** `cosmos2w/ModularDT`, `agent/honf-core-next`, `75f7af96b7fc027cccfaaa01d9338c2c51cf47dc`.

**Deliverable:** one focused Thermal-first implementation and bounded comparison, followed by a user-controlled full-data handoff. This document proposes new mathematics and experiments; it does not describe an already validated model. Existing R12/R12-2/R12-3 checkpoints and their physical laws remain separate.

**Subsequent R13-1 scope, 25 October 2026:** the [mature-training and organization-refinement follow-up](20261025_121601Z_HONF_Round13-1_Mature_Training_and_Organization_Refinement_Work_Plan.md) supersedes this document's100-epoch final-stop/manual-only full-data handoff when that follow-up is invoked. It targets development500 and actual TRAIN600/VAL89 training/evaluation through full-data500 for both arms, with a finite conditional continuation/refinement reserve. The historical R13 measurements remain100-epoch engineering evidence; writing the follow-up launches no jobs.

## 0. The decision in one page

Build **spatially resolved interaction groups**, not another smaller address directory. A group stores a small latent function of receiver position, rather than only a mean vector. The same group partition and the same field-read primitive participate in module contextualization and receiver prediction. Learn subdivision from measured parent-versus-descendant task advantages on TRAIN, rather than from a global sampled-loss reward or stronger compression pressure.

The first implementation has five deliberate choices:

1. Retain the general joint physical input/output path: each module's prescribed heat is an input before interaction; the shared core predicts primitive fields. Do not reinstate universal heat-affinity or heat-independent flow.
2. Retain E→E in the matched comparison. Replace mean-only group values by mean-plus-first-spatial-mode values. Preserve position/state association explicitly.
3. Use one reusable, balanced geometry hierarchy and a learned context-conditioned cut shared across the two interaction stages. Geometry supplies candidates, not the learned cut. This replaces per-forward MST construction and its many small tensor operations.
4. Replace the global REINFORCE task/work controller with one small conditional-utility head trained from actual paired TRAIN losses. There is no new risk-head collection, target K, merge quota, gradient-norm pricing system, or compression penalty in this first comparison.
5. Use direct, tiled attention over the prepared groups, with a coefficient-factorized value contraction. Remove M-first query proposals, posting lists, geometric/proposal prior mixing, and the final group→M update that only served those proposals. Keep the earlier group→M relay, where organization affects subsequent information formation.

**Two new trained endpoints only:** `F13-0` (mean-valued adaptive groups) and `F13-1` (first-order spatial group functions). They share the simplified architecture, policy-learning procedure, data, initialization, native losses and response exposures; they differ in the retained spatial value modes. Existing R12-3 results are historical references, not a third newly trained arm.

**Bounded learning:** initialize common compatible tensors from the balanced singleton R12-3 epoch100 checkpoint, then train each child for 100 additional TRAIN150 epochs, at most 800 main updates total. Review at local25/50/100 without automatic extensions. The children are new-model adaptations, not exact resumes. Produce full-TRAIN launch configurations; the user chooses and launches those longer runs. Do not withhold that handoff solely because an immature development model misses an arbitrary ratio band.

**Priority order:** useful information-bearing organization → native fields and physical responses → measured complete cost. A speedup is not an admission requirement. A root or all-leaf outcome is a result, not something to correct with forced K. Training learns the continuous representation by native gradients and the discrete selector by supervised conditional utility; this is joint learning, not a claim of differentiating through hard topology.

**Continuity and limits:** HONF's core remains learned organization serving information formation and query access, with its useful scale and actual work evaluated together. Spatial modes, a balanced tree, one shared cut and a conditional-utility head are this round's hypotheses, not new definitions of HONF. The exclusive connected partition cannot express arbitrary overlapping E membership. Direct all-group attention intentionally postpones selection before expensive scoring, and zero additional work pressure postpones learning an explicit task/cost tradeoff. This pair tests information retention and task-driven subdivision; it cannot alone establish the complete scalable, overlapping, selectively executed HONF objective. Do not restore M proposals, a sparse selector or a cost controller merely to fill these gaps during this pair.

## 1. Evidence that determines this design

### 1.1 What the completed rounds establish

| Stage | Supported result | Boundary carried forward |
|---|---|---|
| R12 compact retrieval | The amended model achieved E192→96 and recovered useful fields and heat responses | All E siblings accepted; individual values still read; directory contraction was not selective functional compression or complete acceleration |
| R12-2 | Joint physical inputs/head and contextual tree policies ran; roots produced small directories | Positive E queries still expanded to all E originals; protected local M could bypass the learned M route |
| R12-3 recovery | Adaptive groups actually pooled E states, supplied group→M exchange, and reused cached group values without member expansion | Deterministic K¹=1 on all24 organizer cases; K²=1 on22 and191 on2; detailed physical fidelity deteriorated and complete execution became slower |

R12-3 adaptive versus balanced singleton DEV22 means are 4.2129 versus2.6288 fluid T, 4.5510 versus3.1486 surface T, 4.5953 versus2.7304 material T, and4.6775 versus2.5618 module-peak error. These are immature, exposed-development measurements, not fundamental lower bounds. The selected TRAIN angular residual RMS increases1.856→3.206 and DEV angular residual RMS2.019→3.263. Mean bias and angular detail are different failures; do not collapse them into one temperature score. [R123, “Predictor” and “Reader diagnosis”]

The adaptive complete macro is12.894s versus2.331s, while allocated peak drops2.120→1.566GiB. Prepare time is about53.9ms versus9.8ms; cached Q8192 decode changes only43.7→42.6ms. These measurements make preparation a concrete optimization target, but do not identify each substage's share. [R123, “Executor”]

The final scheduled work-policy contribution is reported near1.6e-6, with the downward cap inactive. Strong work pressure is therefore **not an established explanation** of this particular collapse. Do not respond by automatically strengthening, weakening or extending that controller. [R123, “Organizer”]

### 1.2 Findings from current source, not just the figures

- `interaction_groups/frontier.py::LocalFrontier.forward` transfers edge costs/probabilities to CPU, builds the MST in Python, performs node statistics in a Python loop using separate tensor operations, and writes selected ancestry labels through repeated assignments. It scores all191 internal policies per layer even if the root is selected. These are optimization candidates; their individual timing effects are unmeasured.
- The edge metric sees full applied module-relation fingerprints. The stop-policy descriptor, however, reduces them to four rowwise symmetric statistics before node averaging. This loses which module is associated with which relation. The state projection can retain some information, so it is not proof that the complete policy lacks layout dependence.
- `InteractionGroupLayer.forward` reduces E states, coordinates and relation rows independently to means, then applies `group_transform`. It contains no explicit position–state cross-moment.
- `_read_tile` performs M proposals, top-four posting lookup, nine-cell geometric lookup, deduplication, multiple sorts, prior normalization, then group attention. This work persists at very small K.
- `InteractionGroupObjectiveRuntime.collect` computes separate whole-model task-policy and work-policy VJPs per microbatch. The native reward uses one prior-update moving baseline rather than conditional parent/child advantages. A large action-independent cost contributes estimator variance without useful expected policy signal.
- `ThermalInteractionGroupTask` currently hardcodes TRAIN150/DEV22, the known architecture strings, response cadence and its runtime callback. Extend the narrow dispatch points; do not build another engine.

### 1.3 An important mathematical diagnosis: “neutral” stopping is not neutral scale coverage

For a binary tree and a constant stop probability p, expected active groups obey

\[
E_v=p+(1-p)(E_l+E_r),\qquad E_{leaf}=1.
\]

At p=1/2, a balanced192-leaf tree has expected K=4.75, while a chain approaches2. A depth-eight node is reached with probability1/256. Thus zero-ish logits are not neutral over organizational scales. High nodes can intercept nearly all physical learning before useful fine descendants receive enough exposure. The actual learned MST depth/initial logits must be measured; the analytic example does not establish that the saved tree was balanced or chain-shaped.

Likewise, a deterministic root stop at a probability just above1/2 can look extreme even when sampled TRAIN frontiers contain other scales. The report's probability, sampled-K and deterministic-K columns must remain distinct.

This diagnosis motivates explicit **training scale coverage**, not a permanently forced leaf phase or a desired inference K.

## 2. Scientific object: a group is a local information function

Write the group as

\[
\mathcal G_g=(\mathcal A_g,\mu_g,\operatorname{geom}_g,
             \mathbf R_g,\mathcal V_g(\cdot)),
\]

where ancestry \(\mathcal A_g\) identifies original E inputs, \(\mathbf R_g\) retains typed relations to original modules, and \(\mathcal V_g(r)\) returns reusable contextual information at a receiver position r. It does not return a solved physical field by itself.

The initial value function is

\[
\mathcal V_g(r)=V_{g0}+\sum_{a=1}^{d_x}\chi(\xi_{ga}(r))V_{ga}.
\]

For Thermal, \(d_x=2\): a group stores one mean and two spatial-mode coefficients per value head. A root therefore retains explicit first spatial variation, instead of asking the final MLP to infer all local structure from one mean. Complex patterns can still require subdivision. Neither three coefficients nor a positional encoding guarantees physical fidelity.

A simple illustrative distinction is a pair of sites at x=−1,+1 carrying states −1,+1 versus +1,−1. Both state means and coordinate means are zero; their position–state covariance has opposite sign. A mean-only representation cannot distinguish these aggregates unless other latent channels already encoded the association. The proposed representation stores that association directly.

**Not a return to historical pooled hyperedges:** the earlier failure motivates testing a richer finite-dimensional function, not assuming all pooling is safe. **Not a raw-reader rescue:** there is no query expansion back to original E or M values, no protected direct-M prediction branch, and no parallel dense predictor.

## 3. One coherent information path

Use two stages with one scene partition inferred after the first legitimate interaction:

```text
Prescribed M/E inputs, individual module controls, boundary/operating inputs
  → original typed embeddings
  → stage1: M→M, M→E, retained E→E
  → context-conditioned cut of a cached balanced E hierarchy
  → spatial group functions G¹
  → group-function read at original M locations → updated M¹
  → stage2: M→M, M→E, retained E→E
  → update group functions G² on the SAME scene partition
  → direct receiver-to-group read of G²
  → one joint physical-field trunk
  → case-owned native observation/extraction
```

The first group representation changes M, which changes the second E states and final group functions. The partition/restriction operator is the same in formation and decoding; group functions evolve as the computation proceeds.

**Be precise about reuse:** G¹ and G² are different stage states. Do not claim their tensors are identical. Within each prepared stage, keys and spatial coefficients are computed once and reused. The same group-function read implementation is used for module receivers and prediction receivers. This is shared organizational computation, not a post-encoder directory independent of information formation.

Report this as reuse of the learned partition/restriction and read operator across evolving states; final G² is not itself consumed by a final M exchange. Sharing physical K/V tensors between every receiver is not a HONF axiom. Changing the stage1 cut must affect its actual relay and all legitimate downstream states in interventions. Include a bounded organizer-only relation-feature ablation to distinguish relation-sensitive subdivision from group values merely carrying module information; negligible dependence is a result, not a trigger for another loss or mandatory path.

Remove the old final group→M exchange and proposal-key projection from this new class because its only downstream use was the removed M-address reader. Do not run an unused exchange to make a diagram look more “unified.” Retain the earlier actual group→M relay and measure its utility. A new auxiliary loss must not be added to force its importance.

Retain E→E in both arms: its removal was a separate negative experiment and would confound this comparison. Accordingly, this design does not yet claim an encoder free of E² work.

M/E remain input roles. Heat enters its own M record before exchange. One shared trunk predicts the adapter's primitive fields. Physical source IDs identify controls and interventions, not mandatory affine output columns. Any input control/geometry/material change rebuilds preparation; old exact K-times-heat cache claims do not apply.

## 4. Construct spatial coefficients with transparent approximation properties

### 4.1 Leaf feature transform

For each stage, retain the existing singleton-value transform as an initialization-compatible pointwise map:

\[
z_e=\operatorname{LN}_{G}(\operatorname{FFN}_{G}(\widetilde e_e)).
\]

This is evaluated once per original E record during preparation. It is deliberately charged as O(Ed²), even when K=1. It is not a query-dependent fine-value bank and is not read directly by the decoder. Using this transform preserves a useful part of the trained singleton representation rather than discarding it along with the pooling rule.

Both arms execute the identical leaf transform. Keys/value projections are linear maps of group coefficients after pooling, so they need not be repeated on every original record or every query. Do not secretly count the pointwise transform as K-only work.

### 4.2 Weighted local least-squares moments

For positive information measures, let

\[
Z_g=\sum_{e\in\mathcal A_g}\mu_e,\quad w_{ge}=\mu_e/Z_g,
\quad c_g=\sum_e w_{ge}x_e,
\quad \bar z_g=\sum_e w_{ge}z_e.
\]

Let each coordinate scale \(h_{ga}\) be the maximum member distance from \(c_g\), floored by half the declared E-grid spacing. Define \(\xi_e=(x_e-c_g)\oslash h_g\), so all member coordinates are within [−1,1] and their weighted mean is zero.

\[
S_g=\sum_e w_{ge}\xi_e\xi_e^\top,\qquad
C_g=\sum_e w_{ge}\xi_e(z_e-\bar z_g)^\top,
\qquad A_g=(S_g+\varepsilon I)^{-1}C_g.
\]

Use \(\varepsilon=10^{-6}\) in these dimensionless coordinates as the initial numerical regularization, not a validation-tuned hyperparameter. Solve the small system; do not explicitly invert it in differentiable code. A singleton has exactly C=0 and zero slopes. Missing coordinate directions also have zero cross-moment; do not infer fictitious slopes from rank deficiency.

The spatial coefficients are

\[
V_{g0}=W_V\bar z_g,\qquad V_{ga}=W_V A_{g,a},
\qquad K_g=W_K\bar z_g.
\]

Use bias-free W_V and W_K. Keep the shared receiver/output projections and trunk. A single group key still summarizes within-group key variation: preserving first value modes does not make grouped attention exactly equal to member attention. Measure this remaining approximation instead of claiming a generic multipole error bound. The first-order arm adds representational coefficients, not a second prediction network or per-group trainable parameters. The mean-only arm sets the slopes absent and does not pretend to execute three value modes.

The least-squares residual

\[
E_g^{latent}=\sum_e w_{ge}\|z_e-\bar z_g-\xi_e A_g\|^2
\]

is a useful policy feature/diagnostic, not a physical-error certificate and not an extra optimization loss. With zero regularization and full geometric rank, affine latent functions are reconstructed exactly on the members. The deployed ridge introduces a small bias. Nonlinear physical output, attention and unrepresented higher modes still need empirical validation.

### 4.3 Bounded evaluation outside a group

Prevent arbitrary far-field polynomial extrapolation with one declared C1 coordinate extension:

\[
\chi(t)=\begin{cases}t,&|t|\le1,\\
\operatorname{sign}(t)(2-\exp(1-|t|)),&|t|>1.
\end{cases}
\]

It equals the linear coordinate on members, is bounded by2, and has matching first derivative at±1. It is not C2. This is a model choice, not a physical far-field theorem; inspect remote-receiver errors and saturation. Use the same rule for M receivers and ordinary queries. Do not add another radial cutoff or fine-record fallback.

Zero-mass records remain in original identity metadata but contribute no moments. An all-zero-mass group is inactive. Module controls are never averaged into replacement physical modules.

### 4.4 Relations must retain their association

Keep the actual headwise R[e,i] from M→E and the original module axis for group→M prior construction. Its mean is retained as in R12-3. For policy features, include child-to-child discrepancies and within-node scatter of the complete relation vectors before reducing to scalar norms. These distinguish “different modules dominate the two halves” from “both halves have the same relation,” while remaining equivariant to a joint module permutation.

Do not feed only the four symmetric row statistics from the old policy. Conversely, a raw native ID must never become a learned feature. Contextual means/slopes, child differences, spatial extents, relative measures and relation discrepancies are sufficient for the first small shared policy head; do not add a global layout-summary network.

## 5. A deliberately simple group read

For a receiver r and head h, use one group-key score, a relative-position/extent bias and information measure:

\[
s_{rg}^{h}=Q^h(z_r)^\top K_g^h/\sqrt{d_h}
            +b^h(r,c_g,h_g)+\log Z_g,
\quad
\alpha_{rg}^{h}=\frac{\exp s_{rg}^{h}}{1+\sum_j\exp s_{rj}^{h}}.
\]

The denominator includes the fixed zero-valued null. Module contextual reads can retain the existing averaged actual-R prior; prediction reads use the direct score above, without M proposals/postings or a second geometric entrance distribution. Masks and nonpositive mass are handled before normalization.

\[
c^h(r)=\sum_g\alpha_{rg}^{h}V_{g0}^{h}
       +\sum_a\sum_g\alpha_{rg}^{h}\chi(\xi_{ga}(r))V_{ga}^{h}.
\]

Evaluate this using P=1 or P=3 weighted matrix contractions. Do not materialize `[B,Q,K,P,head_width]` or `[B,Q,K,width]` function values. A tile needs group scores, P small basis scalars, and the output accumulator. All positive group weights are read; this first reader is not a sparse top-K selector.

**Tradeoff:** the reader scores Q×K groups, potentially more than the old eight-slot shortlist, but eliminates its proposals, sorts, posting coverage loss and prior-scale competition. Dense, regular, tiled contractions may be cheaper in practice; this is a hypothesis to measure, not a promised speedup. A K=192 endpoint must pay its true Q×192 cost. No hidden maximum-K clipping is allowed.

Prepared keys and coefficient tensors are reused across fluid, surface/material and curl-stencil queries. The query set cannot change the partition or coefficient construction. Exact same-input cache reuse remains; changed physical inputs require fresh preparation.

## 6. Balanced candidates, learned organization

Build a deterministic binary spatial hierarchy of the regular24×8 E catalogue once per input geometry. Resolve coordinates into canonical grid-cell order before construction; original record order and numeric IDs do not define a physical grouping. Recursively bisect a block along its longest physical extent at a grid-cell boundary, balancing record counts within that axis. Carry odd sizes correctly. Store node children, depths, leaf intervals/permutations and geometry-only projection metadata. A 192-leaf balanced hierarchy has maximum depth about8, rather than allowing an arbitrary single-link chain.

The tree is a computational candidate family, not learned physical truth. The cut depends on live contextual information and can vary with module layout, heat and operating conditions. This initial model is limited to an exclusive connected E partition. Modules can participate softly in multiple groups through actual relations, but that is not unrestricted overlapping E membership or a uniquely identified physical hypergraph.

Use vectorized reductions by level or prefix-sum interval statistics. Avoid per-node GPU scalar assignments and repeated per-scene CUDA→CPU→CUDA round trips. A single batched cut-compaction transfer/synchronization is permissible when the backend needs dynamic output sizes; count it rather than inventing a custom kernel solely to eliminate it. At most O(E log E) sparse hierarchy membership metadata is acceptable; do not allocate a dense `(2E−1)×E` ancestry matrix or reintroduce pairwise E-distance nomination. Transform/projection work on all raw moments and policy nodes is counted even when only one group survives.

The scene cut is formed from stage1 states and reused through stage2. This removes an independent second topology decision and makes parent/child counterfactuals easier to define. Later support for irregular catalogues belongs behind a candidate-hierarchy adapter; do not build a generic clustering platform now.

## 7. Learn the value of subdivision, not confidence in merging

### 7.1 One conditional-utility predictor

A small shared head produces a signed normalized advantage \(\widehat d_v\) for each candidate internal node. Positive means that replacing the parent representation by descendants is predicted to reduce the current physical task loss. Inference recursively splits when \(\widehat d_v>0\); otherwise it stops. Leaves terminate naturally. The output may be bounded as `2*tanh(logit)`.

The head consumes input-derived node/child state moments, latent residual, physical extent/mass, relation scatter and child relation discrepancy. It never sees targets at inference, case-error tables or per-case trainable decisions.

### 7.2 Actual paired TRAIN labels

Every fourth primary optimizer macro, choose one TRAIN case and one internal node by deterministic input-stratified cycling across module count and tree-size/depth bands. Selection is not by favorable error. Include nodes below currently stopped ancestors: force the shared ancestor path open for both alternatives and keep all off-path frontiers identical.

Evaluate, at the same weights and sampled native receiver bundle:

- Parent alternative: v is terminal.
- Descendant alternative: split v into its two children. On alternating probe events, permit one additional level of descendants, so that a solely one-step comparison does not permanently hide deeper benefits.

Recompute the changed group→M relay and every dependent stage2 state, group coefficient, query reduction and native observable. This is not a final-read-only swap. Reuse the exact same target masks/receivers and loss normalization; no optimizer step occurs between alternatives.

Let Lp,Ld be the actual per-case native objective. The label is

\[
d_v=\frac{L_p-L_d}{\operatorname{sg}[(L_p+L_d)/2]+\epsilon_L}.
\]

Use epsilon_L=1e-8 in the maintained normalized native-loss units; retain raw signed differences and individual physical components alongside this normalized number. Fit a Huber loss (delta=1, coefficient=1 on utility-head parameters only) to detached d_v. The head has no additional optimizer or gradient-ratio calibration. Detach the utility-head input features for this auxiliary regression so its loss cannot train the encoder to manufacture easier labels; ordinary physical gradients through all executed group functions remain live.

This replaces the old global task/work score-function terms, their scalar moving baselines and work-gradient calibration. It is task-supervised adaptive discretization with a fitted conditional-utility model, **not** an unbiased gradient of a globally optimized tree distribution. It can misestimate out-of-context advantages and needs validation.

Train both paired alternatives on the real native targets as well: add the average of their native losses with weight1/effective_macro_case_count on a probe macro. There is no teacher-prediction ground truth and no extra optimizer step. This bounded counterfactual exposure helps ensure that descendants are not judged only through untrained values. Report the extra forward/backward case presentations separately from primary visits.

Use the actual case count of that optimizer macro, including the final partial macro, for the reduction. Compute both alternatives before applying any update, detach the feature/label pair, and add their averaged physical gradient and the head-only regression through the existing macro accumulator once. Keep the primary objective unchanged. A matching forward may be reused, but add the declared paired average only once and no third physical-loss term through label regression. At inference the head receives the same parent/child feature convention used for its labels, without targets or query-collection statistics. Conditional gains depend on the surrounding frontier and current predictor; record probe context and near-zero/sign errors rather than claiming a globally optimal tree or adding a validation-tuned confidence gate.

### 7.3 Scale exposure without a forced leaf curriculum

From the first update, one quarter of TRAIN case presentations use a query-independent exploration frontier: choose a hierarchy depth uniformly over all available depths and take the legal cover there. The remaining cases use the current deterministic learned cut. Keys use ordinary run/case/epoch identifiers and respect existing query-chunk/retry behavior.

This is explicitly a TRAIN coverage distribution, not learned compression. It includes root, intermediate and leaf scales throughout; it is not a mandatory twenty-epoch leaf phase. Evaluation uses only the learned deterministic cut. Record explored and learned-frontier exposures and their K separately. Paired labels train the utility head directly; do not pretend the discrete exploration has a pathwise gradient or silently add REINFORCE back in.

The one-quarter fraction and uniform depth distribution are recorded first-pair budget choices, not demonstrated optima or a required K distribution. Match their keyed draws across arms, keep them active from the start, and do not introduce a freeze, delayed split permission, forced leaf/root period or sweep of coverage schedules. F13-1 may learn a different cut from F13-0; common-cover diagnostics isolate representation effects without claiming that endpoint differences arise from modes alone.

### 7.4 No new compression incentive in the first pair

Set additional structural/work pressure to zero for both F13 arms. Do not add expected-K, entropy, balance, tie, risk, target-retention or merge-acceptance rewards. First establish whether the information functions and signed conditional utilities produce useful subdivisions at all.

Actual work remains measured. If both models choose leaves, this is a negative compression result, not a reason to force smaller K. If a root is accurate because its spatial modes suffice, classify it as a useful global low-order representation, not discovered adaptive subdivision. A later task/cost tradeoff can price measured marginal work, but it is outside this first pair.

The policy is trained on native reconstruction utility, not demonstrated inverse-decision utility. Response-sensitive organizational choices remain a separate scientific question assessed below.

## 8. Resolve the old failure using a small diagnostic, not another campaign

Before new training, use up to12 existing TRAIN cases spanning M and input conditions. Reuse the saved balanced singleton and adaptive checkpoints; do not wait for or monitor unrelated jobs.

Perform three bounded diagnostics:

1. Recover actual tree depth, child-size imbalance, reached-node frequencies, stop margins and K. Compare sampled and deterministic frontiers. The neutral-probability calculation is explanatory, not a replacement for these measurements.
2. Compute mean-only and first-order latent reconstruction residuals on identical candidate groups and identical prepared E states. Include a position–state reassignment fixture preserving both marginals. Optionally lift projected states back to the old singleton reader for a diagnostic only; count its full member work and never deploy that lift as an unreported bypass.
3. Extend existing parent/child replays to report signed native loss changes, not only output-change RMS. Preserve all-nine components, surface bias/angular detail and the conditional downstream state. Check high-confidence stops on both small-change0348 and large-change0631 examples.

These diagnostics do not have to “pass” before coding or training. They establish the mechanisms and caveats. The current R12-3 report includes the completed recovery and closeout; use that evidence rather than the superseded first-screen stop. Any missing local diagnostic is reported without fabricating arrays or turning report completeness into a launch gate.

Do these read-only diagnostics once on the existing panel, with at most one chosen internal node per case for signed replays. Use the declared five-cover checks only at final evaluation. Do not expand into every-node/receiver/depth replay, full Jacobian export or repeated AD/FD precision sweeps. A missing diagnostic file does not justify rebuilding historical campaigns; use the maintained evaluator for the bounded missing measurement or state the limit.

## 9. Controlled training and a practical stopping point

### 9.1 Two arms, one principal contrast

| Arm | Group value | Partition/policy | Purpose |
|---|---|---|---|
| F13-0 | Mean only, P=1 | New balanced hierarchy and conditional utility | Matched control for the simplified architecture and learning procedure |
| F13-1 | Mean plus first spatial modes, P=3 in2D | Identical policy procedure/features | Test whether spatial information retention improves useful aggregation |

Both retain E→E, the early group→M relay, the shared partition across stages, direct group read, joint physical head and the same native/response objectives. Both policy descriptors can see the same spatial residual statistics, including first-order latent-fit statistics where required by that shared descriptor; count their computation in both arms. Only the retained/read value modes differ. Do not penalize the control with unused three-mode value projections or contractions. Parameter counts, coefficient storage and costs remain separate.

Existing G-EE/G-EE-H100 and U-H results supply context, not a source-identical attribution of every change. There is no new all-leaf third fit. Forced-root, forced-leaf and matched-frontier evaluations are bounded diagnostics at the same weights; none is relabelled a separately trained optimum.

### 9.2 Initialization and data

Start from the **literal balanced singleton recovery epoch100**, not the collapsed adaptive checkpoint and not a best alias with a different age. Copy compatible original input encoders, MM/ME/EE, group leaf transform and compatible query/head projections into both arms. Copy shared tensors identically; initialize unmatched parameters with one recorded seed. Drop obsolete final-address/posting parameters. Report the actual copied/new/dropped parameter names in the normal run summary, without a hash manifest.

These new children do not recover the old reader function exactly. Run an initial native forward and record that transition, then learn; do not gate adaptation on closeness to a different mathematical function. Fresh AdamW moments are used in both children. Parent normalization and heat scale are inherited unchanged from its TRAIN150 fit.

Use TRAIN150/exposed DEV22 for this short mechanistic comparison, not because quarter-data performance proves formal-data potential. The purpose of this pair is the relative information/organization experiment. A full-TRAIN configuration is delivered alongside it, without automatic launch or a hidden requirement to outperform a mature full-data reference first.

### 9.3 Concrete budget and schedule

- 100 additional epochs per arm;15000 primary visits and400 main updates each; total30000 visits/800 main updates.
- Effective batch48, nominal microbatch24, Q1024; preserve existing native auxiliary receiver construction. Permit smaller packing on real OOM while preserving effective batch and case sampling.
- AdamW3e-4/WD1e-5 and clip1 as the common starting recipe. No LR or seed sweep.
- Response fitting remains admitted TRAIN0348 every fourth main update, alternating heat/single geometry. Reuse the balanced-singleton saved response coefficient and existing downward cap equally; do not initiate another gradient-ratio calibration campaign. The approximate reported coefficient23.474 is not a substitute for reading its actual saved value.
- Response predictions use deterministic learned cuts, without scale exploration or score-function response gradients. Full re-preparation occurs at each physical endpoint. This is an explicit change from stochastic R12-3 policy training, shared by both arms.
- One paired native utility probe per four main updates:100 pairs/200 additional case-forward presentations per arm. These are separate from100 physical response pairs/200 endpoint-scene presentations. Calibration/read-only diagnostics are not optimizer updates.
- Use compact history reviews at local25/50/100 under the ETA-based monitoring policy in Section14. Reviews25/50 create no extra checkpoint, validation-file or full response-evaluation campaign. Preserve the existing100-epoch monitoring/checkpoint cadence, latest/best aliases, full validation history,10-epoch curves and tqdm; an actual clean stop saves its resumable state. There is no requirement to extend until a band passes.
- Plan for at most8 process-associated GPU-hours for new learning and2 for bounded evaluation/diagnosis, using measured throughput rather than promising a finish time. These are spending ceilings, not utilization reservations or admission machinery. If contention prevents completion, keep resumable work and provide exact continuation commands; do not poll indefinitely or burn the remainder on new arms.

The10-hour allocation includes qualification, both arms, paired native/response exposures, measured retries and disposable cost steps; it is not renewed after an error. Report a conservative per-process wall charge and avoid calling concurrent process-hours measured device utilization. Use actual native qualification to size the remaining work once. If it cannot fit, trim optional ranking, detailed exports and repeated cost work first, or return a bounded partial result; do not silently reduce physical sampling, add replacement arms or exceed the ceiling. Numerical or compatibility bugs are repaired within scope; a missed research ratio is not a new launch approval boundary.

Use a permitted available device. Shared use is acceptable when feasible. Do not kill other users' jobs, reserve a GPU by identity, or reject ordinary resumption merely because the physical device changed; preserve the existing logical RNG/checkpoint semantics. Where a legacy loader genuinely requires a visibility shape, maintain that shape or use its supported migration path rather than weakening the loader or demanding the same physical UUID.

## 10. Coding map: reuse existing boundaries, replace only the new model path

Suggested capability name: `spatial_interaction_group_fields_v1`. Use a narrow sibling module under the shared core rather than changing old checkpoint equations in place. The name is provisional; Codex may choose an equally clear one.

| Current source | Concrete action |
|---|---|
| `interaction_groups/model.py::Exchange` | Reuse nullable attention, existing masks, residual/FFN and compatible projections; add coefficient-based value contraction for the new class |
| `InteractionGroupLayer.forward` | Separate pre-group MM/ME/EE interaction from group-mediated relay; remove per-case statistics/synchronizations; pool field coefficients instead of separate latent/coordinate means |
| `interaction_groups/frontier.py` | Preserve legacy; new cached balanced hierarchy, vectorized moment reductions and deterministic learned cut plus explicit TRAIN scale exploration |
| `InteractionGroupCore._read_tile` | New direct group read; no old proposals, postings, geometry-prior mixing or fine-record fallback |
| `training/interaction_group_objective.py` | Preserve legacy; replace global task/work policy VJPs with one bounded paired-utility callback and head-only regression for the new class |
| `training/interaction_group_task.py` | Reuse native losses/reference joins; bind new core/task dispatch, added paired exposure and deterministic response mode |
| `channelthermal/interaction_groups.py`, `contextual_joint.py` | Add the new core reconstruction through the same target-free joint physical adapter; native units/stencils/masks remain case-owned |
| `honf_runtime/contextual_frontier_training.py` and research engine | Reuse train/save/resume, atomic checkpointing, named optimizer, clean stop and OOM handling; no new engine |
| `honf_runtime/interaction_group_evaluation.py` | Add moment/utility/cut metrics and signed counterfactual error; separate timings into optional disposable command |
| `tools/honf_research.py` | Existing schema dispatch plus working launch/evaluate/resume examples for the new class |

A compact internal state may be:

```python
@dataclass
class PreparedGroupField:
    group_ids: Tensor
    ancestry: Tensor          # leaf -> group, not an all-node dense matrix
    mass: Tensor
    centers: Tensor
    scales: Tensor
    keys: Tensor              # [K, H, Dh]
    value_coefficients: Tensor  # [K, P, H, Dh]
    relation_mean: Tensor     # original M axis retained
    # Existing ownership/lifetime checks are reused, not duplicated.
```

Only the stage1 field is needed until its M relay completes; inference can release its tensors afterward unless tracing was requested. Training retains what autograd actually needs. Final preparation retains the selected partition, final coefficient bank and necessary original correspondence. Do not persist every inactive node's neural keys/values.

Production coefficients are computed once per stage. Node policy statistics may use all tree nodes, but their actual arrays/work are counted. Geometry metadata is cached only when its dependencies are unchanged. Do not store graph-attached tensors across Adam updates.

Keep interfaces generic in coordinate dimension and channel count wherever the existing primitive supports it. Thermal's 24×8 hierarchy belongs to its candidate-catalogue adapter. Do not put a heat-affine head, native fluid masks, or the word “temperature” inside the generic coefficient read.

## 11. Focused execution tests, not a new validation framework

Use ordinary tests for the changed equations and actual native prepare/decode/backward/reload. Reuse the existing engine and data-separation tests rather than duplicating them for every arm.

Required small checks:

- Weighted constant/linear latent recovery; singleton/rank-deficient/zero-mass cases; original-ID permutation and group coverage. State exact-versus-ridge qualifications correctly.
- Independent FP64 literal group-function read versus coefficient-factorized contraction, including unequal masses, null rows and basis extension. Compare input/parameter gradients on small examples.
- Same query alone, reordered, duplicated and tiled returns the same defined model function. A new query collection cannot change the cut.
- Parent/descendant replay changes only the chosen region plus its legitimate downstream dependencies; no unrelated frontier reset. Raw signed task differences use identical targets/masks.
- Utility feature/label gradients are stopped only in the auxiliary regression, not in physical group formation/readout. No label or target is an inference input.
- Case exposure and scale-exploration keys replay after ordinary checkpoint/OOM recovery; no score-function gradient is silently added.
- Actual native M1/M12 forward, backward and one save/reload/next-update boundary. No forced GPU exhaustion or broad historical suite is required.

Small algebra checks during preparation of this plan showed mean-only RMSE3.37886 versus 0 for an exact first-order fit on an affine latent fixture, and 3.33e-16 difference between literal and coefficient-factorized read contractions in NumPy FP64. Those are mathematical implementation examples only: no HONF checkpoint, GPU inference or physical improvement was tested here.

## 12. Evaluation: answer a few important questions well

### 12.1 Predictor and detail

At local100, evaluate both models on all DEV22 native receivers and the maintained M1/M2 TRAIN panel. Keep all nine native quantities, case means/p90/worst IDs, available M strata, physical masks and original units. Compare F13-1 to F13-0 and show absolute error against the existing TRAIN-mean baseline. The legacy1.10/1.15 lines may be plotted as descriptive references, never treated as approval gates or statistical significance.

For the same retained TRAIN4/DEV4 surface cases, decompose residual into module mean and demeaned angular detail. Show a few signed surface-error curves and native fluid/material fields. The question is specifically whether positional modes help angular/detail errors rather than merely shifting module means. Add no new peak-specific loss in this comparison.

### 12.2 Organizer and useful scale

On the existing24 input-selected TRAIN scenes report:

- K, ancestry sizes, allocated/padded groups, P and K×P separately; root/intermediate/leaf frequency and sampled-exploration exposure;
- latent mean-only versus first-order residuals on identical nodes;
- predicted and measured signed parent/descendant gains, sign agreement and regret of the chosen action on fresh TRAIN probes and exposed DEV diagnostic probes;
- child task advantage stratified by size, receiver role and module count; coverage of nodes behind stopped ancestors;
- legal module-shift/control perturbations and resulting cut/state/read changes;
- actual early group→M relay utility, with dependent stage2 recomputed, and group-value/mode ablations.

Do not use “number of groups” as the entire capacity measure: one P=3 group stores more than one P=1 group. Groups requiring all three modes to recover detail must pay their storage/projection/contraction work. Constant K is not automatically a failure; a claimed adaptive scale needs measured input-dependent useful choices, not just different colors.

### 12.3 Fixed-frontier diagnostics separate representation from selection

On four retained cases, evaluate root, a small set of geometry-defined intermediate covers, and leaves at the same endpoint weights. For example use the three balanced depths nearest group counts8,32,128, plus root/leaves. These are five no-training settings, not a hyperparameter search or a new fitted portfolio. Do not select a validation-optimal fixed depth as the production cut.

Compare the learned cut to these diagnostic operating points and replay F13-0/F13-1 on common covers. Poor learned cuts with good forced-cover predictions implicate selection; poor first-order predictions even at adequate covers implicate representation/training; strong latent-fit improvement without physical improvement means the retained modes are not yet the task-relevant modes.

For a mode ablation, zero slope coefficients while keeping keys/cut fixed. It measures same-model dependence, not the performance of a separately trained mean model. The separately trained F13-0 is the latter control.

### 12.4 Responses and inverse relevance

Reuse the eight existing physical-reference families and the maintained heat/single-geometry/mixed-response evaluator at the final weights. Only0348 fits responses; other cohort labels remain exactly as recorded. Keep absolute errors, reference magnitudes, near-null exclusions and all thermal/flow observables; do not reinstate heat-to-flow invariance as a new training target or acceptance requirement.

Retain the homogeneous-boundary zero-heat temperature-rise test. Its failure in singleton as well as adaptive predecessors is a separate data/physical-supervision issue, not proof that pooling alone caused it. No affine architecture, analytical correction head or new zero-heat loss is added to hide it in this paired representation study.

Bounded JVP/finite-control tests retain original input columns. Rebuild preparation after changes and distinguish fixed-cut derivatives from cut crossings. AD/FD self-consistency is not a physical-gradient reference.

An optional replay of the already available six-candidate pool is useful and may run regardless of a speed result. Keep objective, pressure allowance, IDs and stored physical references unchanged; report no feasible recommendation when appropriate. This is a diagnostic, not new design generation. No solver, new dataset, inverse optimizer or Wind/HET-MF campaign is included.

### 12.5 Execution and cost

Ordinary training records its update/epoch elapsed time. When a permitted GPU is conveniently available, use disposable copies for a short cost check. Do not wait for an exclusive GPU or make cost measurements a training prerequisite.

Measure preparation and complete native calls at Q32/Q1024/full native union. Break down embeddings and MM/ME/EE, hierarchy/policy statistics, coefficient formation, first group→M relay, final coefficient preparation, query scores/contractions and extraction. Include one real complete native/response/policy/Adam macro and peak allocated/reserved memory. Use a small warm-up and three repetitions when feasible; otherwise label contention and sample limitations.

Charge original E transforms, retained E² exchanges, all policy rows, extra paired forwards, P modes, padding and all-group query scores. Report persistent prepared bytes separately from peak training memory. Do not infer complete savings from K or move a favorable stage time onto a different checkpoint.

The planned complexity includes retained O(E²d) contextual work, the charged O(Ed²) pointwise leaf transform, O(E log E·d) moment/policy statistics, O(KPd²) coefficient projections and O(QKPd) coefficient reads. Implementation can reduce constants and commute projections; it has no claimed global linear-complexity theorem while E→E remains.

## 13. What the result can justify

| Outcome | Interpretation and next action |
|---|---|
| Spatial modes preserve detail with fewer useful groups than mean-only | Evidence for richer information-bearing groups; prepare the user-selected full-TRAIN continuation |
| Root works well with spatial modes | Useful global low-order representation, not yet adaptive subdivision |
| Learned leaves dominate but intermediate covers work | Conditional-utility learning/coverage remains inadequate; do not force K or abandon HONF |
| First-order functions fail even on reasonable common covers | Mode capacity, latent content or training is insufficient; report rather than add decoder rescue branches |
| Low group count without useful responses | Organizational compression only; no inverse-reliability claim |
| Lower memory but slower complete calls | Specific memory/time tradeoff, not efficiency success |
| Both short fits remain immature | Deliver their exact limits and full-TRAIN recipes; do not mechanically extend subset epochs or forbid a user-directed formal test |

A successful new group is a compact, inspectable approximation to a useful receiver-dependent information function. It need not correspond to a unique physical causal subsystem. The current Thermal E catalogue is largely geometry-derived; independent changing-environment generalization still awaits HET-MF or other suitable data.

## 14. Resource, storage and research workflow

Use the existing maintained research runner, ordinary versioned configs, Git and focused tests. No new cryptographic hashes, contract freezes, source snapshots, admission services, immutable device bindings, campaign-global locks or monitoring daemons. This plan establishes no concrete insufficiency of standard primitives that would justify them.

Default-deny also covers new baseline snapshots and blocking gates. Before proposing any exception, name a specific concrete failure scenario and demonstrate why applicable standard primitives—Git history, semantic versioning, atomic replacement/transactions, primary keys/unique constraints, types and standard tests—are structurally insufficient. Generic reproducibility or safety assertions are not proof. No exception is established here. Put any new blocking gate exclusively at an irreversible state change, cross-system mutation, critical security perimeter or formal production release; ordinary local research execution needs no new approval service. Scientific decisions and finite resource limits still guide expenditure without becoming a gate framework.

Preserve permissions, safe checkpoint loading, existing run writer protection, original data/TEST separation, active-job ownership and repository artifact hooks. Never bypass an actual security check to make a launch easier. Ordinary research misses, coverage estimates and timing uncertainty are results, not new blocking approvals.

Preflight estimates, algebra fixtures and heuristic checks must never preempt, mock or replace actual native prepare/decode/backward, maintained-runner execution, physical-reference comparisons or empirical measurements. Fix concrete runtime errors with focused execution/tests; passing a generic validator is neither scientific evidence nor a prerequisite to every reversible action. Add tests only for changed behavior and real coverage gaps. Reuse successful qualification when the relevant implementation is unchanged; do not repeatedly run legacy suites, benchmark dry runs or save/reload checks per review. New source schemas and parent-parameter mappings use ordinary configuration/code, not a separate contract registry.

Place runs under a fresh ignored `/data2/.../HONF_Forward_Runs/thermal_spatial_groups_r13_<arm>` directory and diagnostics under its existing DATA2-backed project link. Reuse datasets and atlas caches. The hard total new-payload budget is4GiB retained across both arms and all diagnostics,512MiB peak temporary, with at least200GiB DATA2 free headroom; durable source/docs additions are limited to1MiB on home and scientific home payload is0. Before materialization, verify the existing DATA2 bindings/mount, measure current free space and estimate checkpoints, caches and peak temporary bytes once, rechecking when measured use materially changes. Do not copy datasets, create per-arm caches or silently fall back to home. Keep original inputs/checkpoints and current user runs unchanged. Reuse the repository's exact allowlist/dependency/open-handle cleanup procedure and compact local receipt; remove only verified round-owned disposable entries and report removed files/bytes, retained artifacts and remaining capacity. No broad cleanup project, new hashing service or audit-bundle framework is required.

After each authorized launch, use a read-only Luna agent (`gpt-6-luna`) for monitoring, as requested by the user; the same agent can cover both arms. Supply exact run/log/session identities, endpoint and remaining budget. Confirm actual finite updates once, estimate ETA from logged steady-state throughput plus save/validation overhead, and report the next check time. For a stable multi-hour run, normally read status again only near80–90% of the estimated remaining interval or a genuinely needed25/50/100 review. Skip obsolete intermediate observations after a short run finishes; internal history still records them. Do not poll every few minutes, reread full logs, repeatedly scan GPU/checkpoint state or trigger parent polling of the same run.

Do necessary independent work between checks; when none remains, rest using supported waiting/completion events rather than repeated short sleep/read turns. Wake early for an available completion/error event or an actionable nonfinite result, persistent OOM, budget exhaustion or real stall. Routine metric changes require no notification. Monitoring does not authorize launch/stop/resume/repair and cannot establish scientific success. If Luna is unavailable, state the limitation and keep the same low-frequency policy; do not spawn repeated replacements, create a daemon or promise a scheduled wakeup unsupported by the runtime. Apply this policy also to any later user-launched full-data run when monitoring is requested; it does not authorize that launch here.

Follow the machine's existing30-minute blocked-GPU rule: time a required reservation from the first blockage, stop owned launch-capable waiters/retries and pause at30 continuous minutes if still blocked, preserving exact resume state. Do not stop unrelated jobs or keep polling indefinitely. A user-launched ongoing run outside this round is not a monitoring target merely because it shares the machine.

Read the current plan, R12-3 result and affected code seams once; inspect older archives only for a specific unresolved question. Reuse unchanged checkpoint evaluations and inspect compact summaries before detailed arrays. Keep fixed-cover diagnostics to the four retained cases and five declared covers; do not multiply them across all24/DEV22 cases, checkpoints and every tree node. Optional candidate-pool replay runs at most once per terminal arm using the existing pool, with no generator or repeated rankings. Cost checks use one small warmup and three repeats when available, not repeated device searches. Produce selected figure pages after terminal analysis, with no quota forcing empty panels or repeated render campaigns. The user-directed formal handoff is documentation/configuration work; a missed development ratio does not justify adding arms or withholding it.

## 15. Work order and delivery

1. Read current R12-3 recovery and narrow code paths; use the existing source at75f7af9 or later compatible user changes without resetting anything. Resolve actual parent path and saved state once.
2. Run the bounded tree/latent/conditional-loss diagnosis while implementing the new group-function primitive and its small literal tests; this does not require extra analysis agents beyond the requested Luna monitor.
3. Implement the shared partition, first relay, direct group reader and conditional-utility callback through the maintained task/runner. Exercise actual native forward/backward and save/reload.
4. Run at most the two planned children toward local100 within the finite allocation. Log new physical exposures and use the compact25/50/100 reviews; these are analysis points, not new approval gates or checkpoint campaigns. Repair concrete implementation failures within budget, without a portfolio of numerical recovery models. Preserve a partial endpoint if budget or an actual execution failure prevents completion; do not stop merely because an intermediate descriptive ratio misses its band.
5. Evaluate the terminal pair, signed parent/child utility, native detail, existing physical responses and optional saved ranking. Optional cost work remains independent and disposable.
6. Deliver an English report with about5–6 selected figure pages, retained numerical summaries, tested launch/resume commands and ready full-TRAIN configurations. State what is measured, inferred and still proposed. Commit/push only relevant durable code/tests/config/docs through existing repository checks.

The full-TRAIN handoff includes one recommended model and its matched control, a first200-epoch segment, compact25/50 reviews under the ETA policy, and the existing100-epoch checkpoint/selection cadence with10-epoch curves. Declare inherited-versus-refitted normalization and current measured cost estimates; it must not silently combine a TRAIN150-normalized warm start with a fresh TRAIN600-normalized control. Test that the maintained commands/config dispatch exist and report the native qualification actually performed; distinguish an executable recipe from an unperformed formal training result. If measurements do not identify a preferred model, state the uncertainty and provide the comparison recipe rather than inventing a winner. The user launches the formal runs; no automatic long continuation, extra seed, new dataset or new solver follows this goal.

## 16. Sources and external context

### Project evidence (read before implementation)

- [R12] `HONF_Proj/docs/reports/HONF_Round12_Consolidated_Scientific_Report.md`: directory compression, preserved records, learned selectivity limits and physical decisions.
- [R122] `HONF_Proj/docs/reports/HONF_Round12-2_Contextual_Organization_Development_Report.md`: root-directory collapse, dense original-E expansion and joint physical input correction.
- [R123] `HONF_Proj/docs/reports/HONF_Round12-3_Interaction_Group_Reuse_Development_Report.md`: balanced recovery, actual value substitution, lost detail, conditional interventions and complete cost.
- [P123] `HONF_Proj/UpgradePlan/HONF_Round12-3_Interaction_Group_Reuse_Work_Plan.md`: historical intended semantics and recovery amendments. Its old launch restrictions and obsolete prerequisite decisions are not current obligations.

### Reviewed code at75f7af9

- [Frontier and descriptors](https://github.com/cosmos2w/ModularDT/blob/75f7af9/HONF_Proj/src/honf_forward_core/interaction_groups/frontier.py)
- [Pooling, exchanges, postings and reader](https://github.com/cosmos2w/ModularDT/blob/75f7af9/HONF_Proj/src/honf_forward_core/interaction_groups/model.py)
- [Task/work score-function objective](https://github.com/cosmos2w/ModularDT/blob/75f7af9/HONF_Proj/Case_ThermalChannel/src/channelthermal/training/interaction_group_objective.py)
- [Native/response task integration](https://github.com/cosmos2w/ModularDT/blob/75f7af9/HONF_Proj/Case_ThermalChannel/src/channelthermal/training/interaction_group_task.py)
- [Case adapter](https://github.com/cosmos2w/ModularDT/blob/75f7af9/HONF_Proj/Case_ThermalChannel/src/channelthermal/interaction_groups.py)
- [Existing evaluator](https://github.com/cosmos2w/ModularDT/blob/75f7af9/HONF_Proj/src/honf_runtime/interaction_group_evaluation.py)
- [Existing recipe guide](https://github.com/cosmos2w/ModularDT/blob/75f7af9/HONF_Proj/docs/guides/_bk/20261029_Guide_Consolidation/03_R12_R13_Predecessor_Recipes/Thermal_Interaction_Group_Reuse_Recipe.md)

### Relevant external precedents, not validation of this proposal

- Gupta, Xiao & Bogdan, **Multiwavelet-based Operator Learning for Differential Equations**, NeurIPS2021: polynomial bases and multiscale operator representations motivate retaining modes rather than only means. The present model is not their wavelet operator and imports no benchmark guarantee. [Publisher](https://papers.neurips.cc/paper_files/paper/2021/hash/c9e5c2b59d98488fe1070e744041ea0e-Abstract.html)
- Li et al., **Multipole Graph Neural Operator for Parametric Partial Differential Equations**, NeurIPS2020: multilevel representations can organize interactions across scales. This does not prove first-order moments preserve HONF fields or that the present retained-EE encoder is linear-cost. [Publisher](https://proceedings.neurips.cc/paper_files/paper/2020/hash/4b21cf96d4cf612f239a6c322b10c8fe-Abstract.html)
- Wu et al., **Transolver: A Fast Transformer Solver for PDEs on General Geometries**, ICML2024: learned slices participate in physical-state computation. This is a useful related-work comparison, not evidence that a slice or learned HONF group is a physical cause. [Publisher](https://proceedings.mlr.press/v235/wu24r.html)

**Final research question:** can a learned group preserve the spatial distinctions needed for its information function, and can native task evidence teach when that function should be subdivided—without reconstructing all original values or paying more organizational overhead than the saved computation?
