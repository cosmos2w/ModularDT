# HONF next round: decision-aware forward learning and a genuinely adaptive interaction cover

**Execution mode:** Codex Goal mode.  
**Working branch:** `agent/honf-core-next`.  
**Review basis:** pushed commit `0217a53f6fd4a390c88147f4edc89209bb685f4d`, the completed interaction-response study, the source-conditioned/static-core studies, the earlier coalescence studies, and the existing WindFarm implementation and reports.  
**Status:** a new research and implementation plan, not a description of an implemented or validated method. Numerical gates below are proposed engineering criteria to freeze before the corresponding evaluation, not established physical error bounds.

## 0. The decision and the first deliverable

Return to a **strong absolute-field forward model**, train it to preserve design-relevant responses and constraints, and introduce **case-dependent clustering of the interaction relation** rather than another fixed-budget pair-factor model or a penalty that rewards closing twelve latent slots.

The primary **fidelity reference** is the strong **`dense_pairwise_field`** family; the **development backbone for the new organizer is the clean three-term fine core**, not Dense’s extra coarse/local field paths. For ThermalChannel, use mature Run 1804 as the provisional incumbent after checkpoint/target-contract replay. For WindFarm, use the existing dense model and its native 3-D adapter; the documented Run 2101 selected e495 checkpoint is a reproducible historical reference. Inspect the local Run 2103 artifacts before deciding whether a later dense checkpoint supersedes it. A launch report is not evidence of its completion or final accuracy. ThermalChannel weights must not be loaded into WindFarm.

Retain mature Run 1502 as the sparse-incidence comparator. Use **Run 1508 selected e496 as the provisional ThermalChannel initialization for the clean three-term development backbone**, subject to its normalization/receiver replay; it is an engineering starting choice, not a declaration that it beats mature Dense. For WindFarm, initialize compatible encoder/fine-kernel tensors from the verified wind dense checkpoint and explicitly train/refit the three-term common head. Do not copy thermal weights, silently splice different runs' best heads, or average incompatible tensors. Dense remains the capacity/accuracy reference throughout this transition.

The new candidate is a **response-supervised adaptive interaction-cover HONF**, abbreviated **AIC-HONF** in this plan. The name is a working label, not a claim of literature novelty. A hyperedge represents a reusable donor/source set, a receiver-access region, and a collective control state. Fine physical source identities and nonlinear query–source evaluation remain available. The organizer decides which interaction distinctions and source accesses are needed in each case. It is not an extra field predictor added alongside the dense prediction.

The first substantive deliverable is an executable, replayable **decision-readiness harness** covering the existing absolute models, their finite design responses, pressure/constraint errors, and the existing WindFarm dense baseline. Start implementing the new organizer only after that harness identifies a usable full-access forward reference. WindFarm data/adapter checks start immediately and must not wait for another long ThermalChannel development cycle.

### Goal to optimize

> Can a case-conditioned interaction cover preserve the field responses and feasibility information needed for inverse design, while using fewer *actually executed* fine interactions or delivering a better verified decision for the same total budget?

A variable integer K alone is not the goal. Conversely, minimizing K is not the organizing objective: one giant group can make every query read every physical source and be more expensive than a larger collection of selective groups.

### Minimum useful outcome of this round

Produce a verified incumbent forward model, a trained response-aware full-access control, a functioning variable-cardinality interaction-cover prototype with honest adequacy and execution results, and a native WindFarm evaluation. Run the new held ThermalChannel inverse comparison only after response and constraint gates pass. If those gates fail, finish the bounded diagnosis and the WindFarm work rather than spend the remaining budget evaluating a knowingly inadequate graph as if it were a contender.

## 1. What the completed studies actually establish

The findings in this section are source-derived. The proposed remedies later in the plan are hypotheses.

### 1.1 The physical atlas is useful; the selected response model is not a replacement forward model

The interaction-response report establishes measurable finite position interactions in temperature, including donor-pair effects on an untouched receiver. It also establishes serious numerical limitations: some single-move temperature signs reverse between the two tested grids, spatial mixed-response patterns are mesh sensitive, and the reported interface flux is a proxy rather than a conservation certificate. Pressure and transverse-velocity mixed responses are near-null under the particular analytic-flow generator. These distinctions must remain channel- and solver-specific. [R1, “Evidence design and numerical reliability”]

The selected response model fails even with all candidate factors. On calibration families, all-candidate decoding does not materially repair the finite-response errors. On two held layouts where the selected pair matches the measured pair, the learned pair adds a spurious mixed temperature response. Small mixed errors expressed in *finite-response* normalization units do not imply accurate mixed responses. The model failure cannot be attributed principally to sparse selection, and it is not evidence that physical interactions are absent. [R1, “Anchored response-factor model”]

The surviving Run 1511 decoder has 125 native updates and its scorer has 25. The 150 lost updates were real consumed computation but are not present in the selected weights. Across the three pilots there were 500 optimizer calls, not a completed 500-epoch study. The nearly unchanged tiny-panel losses do not justify selecting the model; neither do they prove that a properly conditioned, adequately trained architecture cannot fit. [R1]

### 1.2 The deployed graph did not have adaptive cardinality

The deployed policy selects all unary factors and the top-scoring pair. Its count is `M + 1`, and all receivers remain supported. Ranked pair identity, exact-zero-delta skipping, and an input-dependent score are not a case-dependent clustering mechanism. One labeled pair per training family is also insufficient to treat every other pair as a known negative. [R1; C3]

### 1.3 The inverse failure has two separable parts

The M3 common pool exposes a false-feasible selected design: predicted pressure feasibility is wrong. This remains a constraint-learning failure even if pair pressure is correctly set to zero under the analytic additive convention; unary pressure responses can still be wrong. The M7 trace exposes poor ranking of a feasible coupled opportunity. Neither issue is solved by selecting more hyperedges from the same inaccurate decoder. [R1, “Controlled inverse decision”]

The maintained inverse code makes an important distinction: in `exhaustive_pool` mode it reference-evaluates multiple candidates, but only the **surrogate-selected** candidate can become the policy incumbent. It separately tracks the best physically evaluated design. Therefore the M7 opportunity not adopted by the policy must not automatically be diagnosed as a bug in the reference-acceptance predicate. It is a ranking/selection failure under that experimental protocol. Preserve the distinction between selected-policy outcome and best evaluated opportunity. [C4]

### 1.4 Earlier work contains usable components, not a complete solution

Run 1505 did achieve case-dependent coalescence of provisional query functions, including within a fixed module count. Its confirmed design-output jump and high planning cost disqualify it as the default inverse backbone. Runs 1506 and 1507 show two different failure modes: a passive criterion that never enters its contraction band, and a trained complexity pressure that closes everything. Neither is a reason to force a desired histogram of K. [R3–R5]

The static specialization preserves useful source-side controls and removes redundant query-controller work, but it still executes fine physical source rectangles. Fresh Run 1508 has meaningful native thermal-output gains and mixed field/near-interface/pressure tradeoffs. Its selected e496 and exact e500 checkpoints are different models. Comparisons of its 500-epoch results with mature 5,000-epoch baselines must not be presented as matched training experiments. [R2; R6]

### 1.5 WindFarm is already a real implementation target

The repository contains a native WindFarm data path, training workflow, field-only wrapper, dense and classic trained baselines, and a sparse-incidence dispatch. Its older README still describes a preprocessing-only scope; the code and later study reports demonstrate that this description is stale. Reuse the implemented path instead of building a new case package. [W1–W4]

The documented data comprise 200 layouts, three categorical direction rows per layout, 6–30 turbines, geometry-only E512 environmental tokens, and native ragged 3-D fields. The maintained learned task is velocity, not validated turbine power, annual energy production, or actuator-load prediction. Those limits directly determine which inverse experiments can be claimed in this round. [W1–W3]

## 2. Source-level audit: confirmed mechanisms and tests still needed

This review inspected the pushed source paths listed in Appendix A. It did not execute their neural models or the local physical generator, and did not inspect ignored checkpoints, raw atlases, or one-time runners. Codex must resolve those local dependencies through the existing workspace. Keep source-level findings, report-derived outcomes, and newly reproduced results separate.

| Location | Finding from reviewed source | Required action before reuse |
|---|---|---|
| `interaction_response/factor_operator.py`, `_baseline_tokens`, `_edge_context` | Baseline module tokens are encoded independently; edge context aggregates the donor tokens and prescribed context, not spectator-layout tokens. Fluid queries receive no material receiver token. | Replace this information bottleneck in any retained response branch. Prefer responses of the full absolute operator. Test a spectator intervention with donors/context otherwise fixed. |
| `interaction_response/organizer.py`, `score_factors` | The scorer likewise lacks full-layout conditioning and accepts only unary/pair donor sets. | Do not extend this scorer into the main organizer by merely changing its threshold. Reuse masks/types, not its restricted representation. |
| `factor_operator.py`, delta-gated pair path | Donors are canonically ordered by the type/repr of their IDs. The bilinear coefficient matrix is not explicitly constrained to transform under a physical donor exchange. | Test **ID relabeling**, not just array permutation or reversal of a donor tuple. A symmetric context with a nonsymmetric coefficient matrix can make `δ_iᵀ A δ_j` depend on which physical donor gets the first label. This is a code-level invariance risk, not a reproduced checkpoint failure. |
| `interaction_response/training.py`, `response_factor_training_step` | `response_kind` is recorded, but this function always calls the summed response prediction. It does not itself construct a mixed stencil or derivative prediction. | Trace callers. A pair-only factor may correctly match a mixed target, but a generic all-factor call does not establish that contract. Add explicit label-kind/prediction-kind assertions and a four-state mixed-loss test. Do not claim the historical run used incorrect labels without reproducing that path. |
| `inverse/response_factor_oracle.py`, `prepare_baseline` | Explicit `all_unaries_plus_top1_pair` deployment; all-receiver fallback; unknown validity neighborhoods can be constructed for unmeasured factors. | Preserve historical replay. The new implementation must distinguish unknown validity from valid and must not silently certify extrapolation. |
| `inverse/interaction_guided.py`, policy evaluation loop | Exhaustive reference evaluation and policy acceptance are deliberately different; candidate selection is constraint-first on predicted values. | Expose selected-only, exhaustive diagnostic, best-evaluated, and reference-accepted outcomes separately. Test failed solves and post-solve serialization independently. |
| `interface_fields/dense_pairwise.py`, `prepare_fine_messages` | Fine MM/ME/EM contextualization is simultaneous and includes all active physical sources. Preparation costs include M² and ME work. | Reuse this as a capacity/reference path. Do not call a P2-only optimization globally sparse. Profile preparation before choosing which typed interaction to optimize next. |
| `interface_fields/source_conditioned_pairwise.py` | Static source controls remove query-to-group routing, but QM/QE still use full source rectangles; environmental attention uses a globally normalized weighted softmax. | Reuse gains and prepared-source caching where helpful. Preserve source measures, attention normalization, output biases, and wrapper semantics. No claim of sparsity from the static name. |
| `Case_WindFarm/src/windfarm/model.py` | The wrapper supports native 3-D velocity and prepared arbitrary-query decoding. The sparse-incidence configuration currently validates a 2-D payload and then restores 3-D fields. | Add an explicit capability-checked 3-D registration for the new core. Do not proliferate this validation workaround or assume source-conditioned variants already pass the WindFarm allowlist. |
| WindFarm `materialize` and target transform | Real hub queries materialize lazy parameters; physical velocity uses a training-owned transform. No thermal port state is fabricated. | Preserve these contracts and verify all new parameters exist before optimizer creation. Keep targets and metadata out of prepared model state. |

Also audit the very different native temperature/flux error levels in older and newer ThermalChannel reports by replaying a few exact checkpoints with the same output units, local-model state, masks, and coordinates. Relative L2 and MAE are different metrics. The observed differences are not, by themselves, evidence of a normalization bug, and old checkpoints must not be “repaired” by silently changing their interpretation.

## 3. Baseline selection: make one decision, not another architecture tournament

### 3.1 Separate the incumbent, the common training control, and the experimental organizer

Use the following names consistently in code and reports:

| Name | Role | Initial choice |
|---|---|---|
| `B_inc` | Best available validated absolute forward reference for the case; frozen during initial diagnosis | ThermalChannel: Run 1804 mature selected checkpoint. WindFarm: documented dense Run 2101 selected e495, subject to verified local evidence for a later dense checkpoint. |
| `B_value` | Clean three-term full-access control trained/fine-tuned with the existing value objective | ThermalChannel: provisional Run 1508 e496 initialization. WindFarm: explicitly refitted three-term control using compatible wind dense tensors. Same initialization as the paired response-aware arm. |
| `B_response` | Same absolute model with paired response, receiver/QoI, and constraint-aware supervision | The primary near-term forward improvement. It has no adaptive-K claim. |
| `A_dynamic` | Adaptive interaction-cover model initialized from the accepted full-access state | Same case encoder, field heads, case wrapper, normalization, and response/decision objective as the relevant control. |

The mature incumbent is a fidelity reference and possible teacher, not a fair same-cost training competitor. The clean three-term model is the new development baseline, not a claim of strict whole-model Dense equivalence. Report both incumbent performance and matched incremental-training comparisons. A new method must not claim an architectural win merely by receiving additional training or better labels.

### 3.2 Freeze a small replay panel before comparing current weights

For ThermalChannel, use existing train/development families and the old response atlas for diagnosis. Include M=3/5/7/10, the known pressure failure, a hot-receiver interaction, and a near-null control. They are not new held tests. Replay at least the mature dense, mature 1502, and selected 1508 models on the same physical outputs; add the frozen static Run 1507 specialization only when needed to isolate an execution question.

For WindFarm, replay the dense and classic documented checkpoints on the existing validation geometry panel and fixed native samples. Inspect the local 2102/2103 manifests and best-checkpoint metadata, not only CSV headlines, before upgrading the incumbent. There is no requirement to train a new classic baseline.

Rank eligible incumbents **feasibility/critical-response first**, then decision error, then role-separated field/receiver errors, and finally complete application cost. Do not collapse these into an unscaled sum. If the provisional dense incumbent fails the response gate, inspect a bounded response-aware fine-tune as a capacity diagnostic before demoting it solely on absolute-field ranking. The clean source-conditioned model may replace it as the operational incumbent only after the decision-readiness replay supports that decision.

Keep a concise `baseline_decision.md` containing the accepted checkpoint, its limitations, exact normalization/wrapper identity, and why alternatives were retained or rejected. This should take a bounded diagnostic phase, not become a new 5,000-epoch competition.

### 3.3 What to reuse and what to stop

Reuse the reference atlas, physical IDs, common-domain/material-coordinate rules, typed output and mask contracts, the full-context fine reader, source-side gain/caching ideas, and reference-corrected inverse evaluation.

Do not resume Runs 1505–1511 as the main new candidate. Do not inherit their twelve-slot tree, top-one pair budget, or fixed active count. Preserve all historical architectures and checkpoints for replay. The new core is opt-in; checkpoint conversions require an explicit identity or documented refit transition, never permissive loading that hides missing parameters.

### 3.4 Prevent inherited field bypasses from defeating the experiment

The reviewed `InterfaceFieldCore` selects `SharedInterfaceContext` for Dense, including its historical coarse-latent/local-neighbor paths, and `ThreeTermInterfaceContext` for the source-conditioned/group-control families. The latter supplies only the query/global background and a head on `C_g + C_M + C_E`; its local path is exactly zero. This is an existing implementation distinction, not a proposed interpretation. [C8]

Use the three-term common context for the new adaptive candidate and its matched clean full-access controls. Keep the dense incumbent unchanged for reference. Its coarse/local branches must not remain as unreported parallel learned paths around the new interaction cover. Legitimate case-owned local constitutive models and prescribed-context background terms remain, with their semantics and costs explicit.

The source-conditioned baseline still has a fixed twelve-proposal source organizer. It is a baseline, not the new mechanism. **Replace that organizer with the new case-local cover; do not stack two organizers.** Reuse the fine kernels, encoders, physical wrapper, and collective gain idea. A new control/normalization map that cannot exactly reproduce the old one is an explicitly trained refit, with a same-architecture full-access control. Do not promise strict Run-1508 parity from a partial tensor copy.

For WindFarm, adding this clean common context is also a refit transition because the documented dense model uses the older shared context. Audit compatible shapes, materialize new parameters before optimizer creation, train the new common head against training references/clearly labeled teacher outputs, and establish fit before attributing any result to adaptive K. There is no permission to hide a large fidelity loss from this transition behind an attractive graph.

Retained dense MM/ME/EM preparation is upstream contextualization rather than a parallel field-output bypass. Its cost and indirect spectator dependencies still have to be reported. If the clean full-access transition itself fails after the bounded fit remedies, keep the stronger incumbent and report a backbone-transition failure; do not proceed with a knowingly inadequate adaptive candidate.

## 4. Define the forward task in the space where design decisions are made

Let `d` be the admissible design variables, `c` the prescribed operating/boundary context, and `q` a physical receiver query. The main model is

\[
\widehat y_\theta(d,c;q)=\widehat{\mathcal F}_\theta(d,c)(q).
\]

It must predict the absolute state at a new design without requiring a reference solution at that design. Case adapters own geometry, units, physical receiver definitions, and output roles. ThermalChannel material queries and WindFarm volume queries are not forced into the same physical interpretation.

### 4.1 Obtain responses from this same absolute operator

For a local design change,

\[
\widehat{\Delta y}(d,\delta,c)
=\widehat{\mathcal F}_\theta(d+\delta,c)
 -\widehat{\mathcal F}_\theta(d,c).
\]

For two identified donor changes,

\[
\widehat I_{ij}(a,b)
=\widehat{\mathcal F}_\theta(d+a_i+b_j,c)
-\widehat{\mathcal F}_\theta(d+a_i,c)
-\widehat{\mathcal F}_\theta(d+b_j,c)
+\widehat{\mathcal F}_\theta(d,c).
\]

All four evaluations use aligned Eulerian/material receiver conventions and correct trial geometry. Recompute the case state at each design for **deployed-operator response accuracy**. A separate fixed-baseline/fixed-organization replay can diagnose attribution or boundary behavior, but it is not the deployed finite response.

An accepted physical baseline can still provide a bias-corrected local proposal model:

\[
\widehat y_{\rm local}(d+\delta)
=y^{\rm ref}(d)+\widehat{\mathcal F}_\theta(d+\delta)
-\widehat{\mathcal F}_\theta(d).
\]

Label this reference-anchored operating mode and charge its baseline solve. It does not make the absolute model dependent on solved fields as input. Compare raw absolute and bias-corrected local proposals separately; do not credit anchoring as a graph benefit.

The previous exact-baseline value residual and response residual were algebraically identical. The new information comes from paired sampling, finite/mixed consistency across designs, targeted receiver weighting, and independent constraint/decision supervision—not from renaming the same residual.

### 4.2 Inputs must contain full conditional information

A response to a donor move can depend on unchanged modules, material/boundary context, and the current receiver. Use the full-case encoder/fine contextualization, including permutation-invariant aggregation or message passing over spectators. Add explicitly signed donor–receiver geometry and appropriate physical length scales where the existing adapter lacks them. A global mean alone is not guaranteed to retain a local shielding or wake configuration.

Do not feed case IDs, response labels, selected-candidate IDs, solved trial fields, or numerical quality flags into the model. Geometry-derived quantities must be recomputable at a proposed design. In particular, a source-defined descriptor with an unknown formula cannot become a required continuous-design input merely because it is present in the training archive.

Test the whole differentiable path from design tensor through the adapter, source coordinates/features, prepared state, material query coordinates, and scalar functional. Backend-only AD/FD agreement cannot detect a detached NumPy conversion or a missing geometry derivative in the case wrapper.

### 4.3 Use role-separated response and decision targets

Train a controlled objective of the form

\[
\mathcal L
=\lambda_v\mathcal L_{\rm value}
+\lambda_\Delta\mathcal L_{\rm finite}
+\lambda_I\mathcal L_{\rm mixed}
+\lambda_J\mathcal L_{\rm decision}
+\lambda_g\mathcal L_{\rm constraint}.
\]

Introduce terms in stages. Do not enable all terms with arbitrary equal weights on the first update. Measure their gradient scales on training batches, freeze initial weights from that training-only calibration, and retain unweighted physical metrics. Critical feasibility errors are promotion gates, not something an improved volume loss may compensate for.

`L_value` retains the original valid-domain fields and native receiver outputs. Use equal-family/equal-case aggregation and report module-count strata and tails. `L_finite` trains paired changes at physically meaningful step sizes. `L_mixed` uses explicit four-state predictions and labels only where observed; pair order, delta signs, and shared masks must be checked by tests. Do not force unobserved mixed responses to zero.

For a named role `r`, a useful evaluation statistic is

\[
E_{\Delta,r}
=\frac{\|\widehat{\Delta y}_r-\Delta y_r^{\rm ref}\|_{W_r}}
 {\max(\|\Delta y_r^{\rm ref}\|_{W_r},\eta_{\Delta,r})},
\qquad
E_{I,r}
=\frac{\|\widehat I_r-I_r^{\rm ref}\|_{W_r}}
 {\max(\|I_r^{\rm ref}\|_{W_r},\eta_{I,r})}.
\]

Here the `W` weights and numerical-floor estimates are explicit and local to the relevant evidence. Also report numerator, signal magnitude, floor, and physical units. Training uses fixed training-derived channel scales to avoid unstable division by a tiny example signal. Evaluation may use an example's reference magnitude as a denominator; it must never enter inference or model selection before its split is opened.

`L_decision` operates on the actual functional of predicted state or a separately justified observable head. For ThermalChannel, compute smooth and true solid-module peaks from the proper material output, not fluid temperature or a convenient port proxy. Use a margin-aware ranking loss only between observed, comparable candidates; leave numerically unresolved differences unranked. Keep a physical-state fidelity floor so an accurate scalar cannot conceal a failed field model.

`L_constraint` directly supervises the maintained pressure-drop functional and its finite change, with extra training examples near the feasibility boundary. A low global pressure-field L2 is not sufficient. For other datasets, replace this with their actual observed constraints; there is no generic “pressure constraint” to transplant to WindFarm.

### 4.4 Derivatives are useful only when the evidence supports them

Use finite responses as the primary ThermalChannel signal because grid-mask sensitivity compromises small-step derivative interpretation. A directional derivative term is allowed only on step/tolerance/mesh panels that establish a stable approximation at the declared scale. AD/FD agreement of the surrogate is a numerical consistency test, not physical derivative validation.

Derivative-informed operator learning provides a relevant methodological precedent for training design sensitivities and exploiting low-dimensional directional information [E1, E2]. It does not provide missing derivative truth for this dataset. No new adjoint or full Jacobian infrastructure is required for this round; a small set of physically selected and randomized feasible directions is enough to test the idea.

## 5. New clustering target: cover the needed interaction relation

### 5.1 Preserve the central HONF representation

Use typed source membership and query access,

\[
A^M_{ie}(d,c),\quad A^E_{je}(d,c),\quad
\alpha_{qe}(d,c),\quad h_e(d,c),
\]

with nonnegative access coefficients and physically indexed source rows. The induced relation and collective control are

\[
\rho^S_{qs}=\sum_e\alpha_{qe}A^S_{se},
\qquad
n^S_{qs}=\sum_e\alpha_{qe}A^S_{se}h_e,
\qquad S\in\{M,E\}.
\]

The group state controls routing or the fine interaction. **It is not a direct pooled field-value shortcut** such as `sum_e alpha_qe V h_e` added to the prediction. Preserve the existing distinction between a rank bound on routing and the expressive capacity of the nonlinear physical field.

A conceptual fine read is

\[
C_M(q)=\sum_i\mu_i\rho^M_{qi}
\Psi_M(z_i^\star,q-x_i,n^M_{qi}),
\]

with the existing architecture's normalization and affine terms retained when converting a checkpoint. Environmental attention may be written schematically as

\[
C_E(q)=
\frac{\sum_j\mu_j\rho^E_{qj}\exp(a_{qj})V_j(n^E_{qj})}
 {\sum_j\mu_j\rho^E_{qj}\exp(a_{qj})}.
\]

The score may also be control-conditioned. This expression is a design contract, not permission to discard output projections, biases, inherited scaling, or numerical-stability logic. Each unique supported query–source interaction is evaluated once after its path contributions have been combined.

### 5.2 What a hyperedge must explain

Represent each edge as

\[
e=(\mathcal S_e^M,\mathcal S_e^E,\mathcal R_e,h_e,\mathcal U_e,\mathcal E_e).
\]

The first two sets identify sources; `R_e` identifies receiver support; `U_e` identifies the tested input neighborhood or explicitly says unknown; and `E_e` records evidence. Source sets may overlap: this is a cover/biclustering of a directed interaction relation, not necessarily a disjoint partition of physical modules or the fluid domain.

Its evidence record must answer four concrete questions:

1. Which observed donor changes and receivers are relevant at the stated scale? Include signed finite effects and any resolved mixed effects, with reliability labels.
2. What prediction/response distortion occurs when this access distinction or source support is removed, merged, or refined, after controlling for decoder quality?
3. Which actual expensive source accesses are retained or skipped by the edge, including overlap with other edges?
4. Which decision or constraint can its information change: a coupled move, a feasibility margin, a ranking, or a need to request another reference evaluation?

Not every edge needs an irreducible pair-interaction label. An edge can organize shared first-order influence, common receiver response, or collectively conditioned source access. The computational group order and the order of a finite physical mixed response are different concepts.

An environmental quadrature token is not an independently movable physical module. Removing its latent contribution is a **numerical representation intervention**, not a physical experiment deleting a piece of fluid. Environmental support may be trained with task distortion and labeled teacher diagnostics, while physical donor responses supply independent checks on the complete operator. Keep these evidence categories separate.

### 5.3 The key optimization is not `min K`

For a case and declared receiver universe `P`, seek a cover `H` satisfying role-specific tolerances while reducing measured work:

\[
\min_{H}\;\widehat C_{\rm complete}(H;d,c,P)
\quad\text{subject to}\quad
D_r(H)\le\varepsilon_r
\quad\text{for all protected roles and decision quantities }r.
\]

Use both absolute adequacy gates and allowed degradation relative to the accepted full-access model. A cover that approximates a poor reference accurately is not physically adequate.

A useful cost ledger separates

\[
C_{\rm prepare},\ C_{\rm organize},\ C_{\rm route},\
C_{\rm unique\ fine},\ C_{\rm pack/scatter},\ C_{\rm wrapper}.
\]

Scopes that overlap in a profiler must not be added as if independent. An offline cost model can use actual row counts and measured coefficients; promotion still requires complete wall-time measurements.

**Why this can produce useful dynamic K:** a coarse receiver-access group tends to inherit the union of sources needed anywhere in that group, creating unnecessary reads. Splitting can reduce these unions but adds routing/organization overhead. Merging can save overhead but increase over-reading or response distortion. Therefore either a split or a merge can reduce cost. This differs fundamentally from a global cardinality penalty whose easiest solution is universal closure.

### 5.4 An implementable primary mechanism

Implement one primary mechanism: a **case-local, response-conditioned hierarchical interaction cover**, with source masks and control states at candidate receiver-access regions. The hierarchy is a bounded candidate index over physical receiver supports, not a tree over twelve globally named latent slots. Geometry supplies a reproducible candidate structure; learned response/context information decides its active resolution and source memberships. The hierarchy is a search restriction, not a claim that physical interaction domains are geometrically binary.

Start with a receiver-anchor universe assembled from geometry only, independent of the current output-query chunk: environmental locations, module-attached receiver anchors, and required functional locations. Use a case-owned measure and separate output roles. Generate a bounded multiscale candidate hierarchy on this universe. Source sets are learned from full-layout contextualized physical tokens; they are not fixed by distance alone. Distant sources must remain eligible through a conservative fallback and tested candidate coverage.

At a node, a split is useful when its children explain different needed source sets or response patterns enough to justify their additional cost. A merge is useful when the coarser source union and controls remain sufficient and cheaper. Choose initial split candidates by geometry, then permit response-embedding candidates if the geometric hierarchy demonstrably misses useful groupings. Do not launch both as independent large architecture sweeps.

Initially target QM/QE reads, which dominate large-query execution in the documented cases. Preserve fine MM/ME/EM contextualization so spectator information is not lost. Record that this stage is **query-side** adaptive work. Only extend the same mechanism to preparation if the measured preparation floor limits useful speedup at the target shapes.

The representation must preserve an exact full-access diagnostic mode. With all source accesses enabled and all new control gains at their identity, the converted reader should reproduce its chosen parent within documented floating tolerances. If this identity cannot be constructed for an inherited architecture component, declare the transition a refit and compare it separately rather than claiming a strict conversion.

### 5.5 Train the cover from adequacy and marginal work, then amortize it

Use the following sequence, not simultaneous cold-start training of an inaccurate field model, a support classifier, and a cardinality penalty:

**First, establish a fit-capable full-access operator.** This is the `B_response` gate. It protects against repeating the all-candidate failure of Run 1511.

**Second, build a train-only cover oracle.** On bounded training probes and aligned response stencils, propose source-pruning and node split/merge operations. Evaluate each proposal using the same frozen forward model, explicit refit only when the experiment calls for it, protected receiver/constraint metrics, and actual supported-pair union. Keep a proposed change only if it is adequate and reduces estimated complete cost. Unknown physical labels stay unknown. Teacher output fidelity and reference output fidelity are separate columns.

A geometric full-access tree, a geometry-only restricted cover, and an oracle response-conditioned cover are useful diagnostics here. If even the oracle cannot find an adequate selective cover, do not train a gate network to imitate an attractive K distribution. The obstacle is the cover family, fine model, or evidence—not amortization.

**Third, train an input-only amortized organizer.** Predict marginal distortion/adequacy and cost-relevant split/source decisions from contextualized input tokens, receiver-anchor geometry, role, and declared tolerances. Do not predict a target K as a class label. Do not use final response labels online. Use known oracle decisions with an uncertainty/unknown mask and retain conservative full-access fallback on unsupported cases.

**Fourth, jointly fine-tune under the response/decision objective.** Exercise selected, full-access, and nearby alternative covers during training so a node is not evaluated with untrained child controls. Keep an explicit full-access accuracy control and monitor gate task gradients. Protect physically resolved nulls with measured floors, not blanket pair sparsity.

**Fifth, freeze selection before held evaluation.** Input-conditioned K, source support, and receiver access must be produced without an online reference solve or hidden label lookup. Measure oracle-versus-amortized adequacy and cost gaps to distinguish representation limitations from routing-learning limitations.

A bounded alternative is authorized if this mechanism fails for a specific reason: replace only the hierarchy's candidate partition with a response-embedding cover while retaining the same fine reader, loss, evidence, and budget. Do not revive per-forward FP64 ADMM, a twelve-slot contraction tree, or a new direct pooled-value path as an undocumented rescue.

## 6. Continuity, exact counts, and the actual sparse executor

### 6.1 Make split/merge activation act on access and control—not add another field branch

For a fixed candidate node `v`, define its parent access/control pair `R_v(q) = (rho_v(q), n_v(q))`. A continuous refinement can be written

\[
\widetilde R_v(q)
=(1-g_v)R_v(q)
+g_v\sum_{c\in\mathrm{children}(v)}w_c(q)\widetilde R_c(q),
\]

where child receiver weights are nonnegative and sum to one on the parent's supported receiver region. They should have a continuous overlap rather than a discontinuous hard spatial seam. Flatten this recursion into the ordinary `alpha`, `A`, and source-control moment before computing each fine interaction once.

Use an activation with literal zero/one endpoints, for example

\[
g(u)=\begin{cases}
0,&u\le0,\\
3u^2-2u^3,&0<u<1,\\
1,&u\ge1.
\end{cases}
\]

At `g=0`, child contributions are identically absent. At `g=1`, the parent's direct contribution is absent. Within the transition both are part of the executed access representation. Count both. An endpoint-flat gate is not itself a learning strategy: train its underlying adequacy predictor from supervised train-only targets and exercise its transition region so dead gates do not repeat Run 1506.

Source support also needs a continuous exact-zero weighting rule, compatible with the global environmental normalization. Changing a boolean mask at a nonzero coefficient is not a continuous sparse operator. A zero-weight branch should vanish before execution is skipped. Preserve a nonempty valid environmental normalization or an explicitly defined empty-source behavior. Test the complete score, value, gain, and bias terms, not just an attention tensor.

This construction only addresses continuity for a fixed candidate structure and continuous underlying components. Sparsemax can introduce derivative kinks; rebuilding a geometric hierarchy or changing candidate membership can introduce further discontinuities. Do not claim global C1 smoothness. In local inverse search, freeze the candidate hierarchy within a trust region while allowing the modeled controls/gates to vary; check boundary/rebuild transitions separately. A frozen hierarchy is not permission to freeze all design dependence.

### 6.2 Cardinality ledger

At every saved checkpoint, report separately:

- Candidate-node/edge capacity, active groups on the fixed case receiver universe, and any capacity saturation.
- Exact source incidence and query access supports; active transition nodes; query degree `K_q` and any entropy diagnostic.
- Unique induced QM/QE source pairs, physical prepared MM/ME/EM rows, actually executed rows, padded rows, and fallback work.
- Complete latency/memory and a real inverse iteration, with organization and preparation charged.

The primary `K_case` is the number of mathematically active reusable access groups on the fixed case receiver universe, not a participation-ratio statistic. A mixture transition can have more active groups than either endpoint cut; its exact active count is the reported count. An idealized terminal-cut count may be an additional diagnostic, never a replacement.

K must be invariant to query chunking, query ordering, and arbitrary source-ID relabeling. It may depend on physical layout, operating category/context, phase, requested output roles, and a predeclared accuracy tolerance. Distinguish these dependencies in the report. Do not quietly allow the measured value of a trial objective to decide its graph.

Demonstrate variation **within fixed M and comparable receiver support**, not just across M or domain volume. Verify that different counts survive repeated deterministic evaluations and correspond to adequate, materially different access patterns. Universal full access, universal one-group access, capacity saturation, or a fixed formula `K=f(M)` is a negative mechanism result unless the limited data genuinely support no other organization. Never tune a count-variance reward to manufacture the desired outcome.

### 6.3 Preserve the nonlinear fine source path and normalize globally

A source may be reached through several hyperedges. Accumulate its access/control coefficients, deduplicate the query–source union, and evaluate its expensive kernel once. Summing one fine result per path double-counts sources and confounds execution savings.

For environmental attention, retain unnormalized numerator and denominator contributions across disjoint execution blocks and normalize globally. Adding independently normalized per-group attention outputs is not the same operator. Keep quadrature weights and source-score priors distinct. Split-weight duplication of a quadrature source must preserve the intended measure and prediction to tolerance.

The first implementation should use existing Torch operations, prepared source projections, sorted/blockwise gather/scatter, and a rectangular reference executor. Avoid custom kernels until a profiled bottleneck and realistic break-even case justify them. A sparse executor can lose on small shapes; retain a measured dense fallback and report its frequency. Do not call an adaptive logical plan a speedup when the fallback performs all dense work.

### 6.4 Scaling claim boundary

Current full preparation is approximately quadratic in module count and linear in module–environment pairs; dense querying scales with Q times the physical source count. Reducing a logical group count changes neither cost automatically.

Benchmark independently varying M, E, and Q. At minimum use native ThermalChannel and WindFarm shapes, WindFarm Q8192/Q65536, and bounded larger synthetic shapes that fit memory. Synthetic scale tests establish execution behavior only. Millions of output cells do not mean millions of environmental source tokens: the existing wind input bank has E512. Test environmental density sensitivity separately from output-resolution scaling.

Multilevel operator work provides a reason to preserve long-range interactions while seeking scalable execution [E3]. This plan does **not** import its linear-complexity theorem into HONF, nor authorize replacing fine states with an unvalidated pooled field path. Any preparation-side coarsening is a later, separately validated extension.

## 7. Evidence expansion: fill identifiable gaps, not a large indiscriminate atlas

### 7.1 Reuse old evidence with the correct split status

All previously opened ThermalChannel calibration/final response outcomes are now development evidence for the new method. The historical 90-case `test` cohort also remains development data, with the 0001/0273 cross-split duplicate explicitly grouped. Do not recover an untouched test set by changing filenames or redrawing a split around checkpoints that already saw its cases.

Preserve source-family IDs across exact duplicates, rounded replays, perturbations, heat controls, and mesh variants. New final-review families must have inputs frozen and screened against the historical and expanded training/development families before their response/decision outcomes are opened. Pretrained teachers' exposure belongs in the split manifest too.

Use the old atlas first to determine whether `B_inc` and `B_response` can learn meaningful finite responses. There is no reason to regenerate a completed solve whose exact inputs, settings, outputs, and provenance are intact.

### 7.2 Expand coverage where it changes the model question

Prioritize complete small-M panels over one arbitrary pair in many layouts. On at least two M3 families, cover every unary donor and every pair for a declared set of feasible directions. At larger M, sample additional unaries and several pairs with different receivers, including the hot/critical receiver, shielding/alignment contrasts, separated controls, and a spectator-layout contrast. Freeze the sampling rule from training geometry and existing training evidence before requesting those labels.

Include more than one perturbation direction and magnitude where affordable. A fit to one x–y bilinear entry does not identify the full response to arbitrary planar moves. Include both signs and a half-step only where the generator's numerical resolution can support their interpretation. Keep total heat, radii, and operating context fixed for the primary position study. Heating controls remain separately labeled.

For environmental source support, use quadrature-density/split-weight tests and complete-model omission/refit diagnostics. Do not invent environment-perturbation CFD labels from neural attention maps. Add boundary/context responses only for physically varied, supported inputs and available reference solves.

### 7.3 Receiver and numerical reliability contract

Eulerian differences use common fluid coordinates/masks over the complete stencil. Material outputs use the same physical module ID and local coordinates. Report the common-domain coverage fraction and do not describe it as a prediction of newly exposed fluid cells. The absolute forward model must additionally be evaluated on each trial's own valid domain, including newly exposed regions where the reference supplies them.

Separate storage quantization, solver stopping tolerance, grid geometry effects, and cross-mesh response differences. A repeatable nominal-grid signal is not automatically mesh resolved. Treat proxy flux as a reported auxiliary output unless its reliability is newly established; never use it as a mandatory positive-edge label or a conservation certificate.

Use mesh checks on **promising decisions**, not only on mixed-response norms. Compare the baseline and selected move, their feasibility margin, objective ordering, and relevant receiver pattern. A persistent mixed sign does not guarantee a stable single-move sign or ordering.

### 7.4 Avoid post-selection physical narratives

An edge whose removal harms a trained model may be useful to that model without being a unique physical causal object. Require agreement with independent response evidence for a physical interpretation, and use “effective donor-set-to-receiver interaction at this scale” rather than microscopic causality.

Separate field-level mixed response from nonlinearity of the objective. Combining independent module temperatures through a maximum or log-sum-exp can create an objective interaction without physical coupling. Inverse grouping may exploit such objective coupling, but label it as decision coupling and do not use it as a physical edge target.

## 8. Training stages and quantitative promotion gates

These gates prevent expensive evaluation of a representation that has not learned its basic task. They are not a substitute for judgment when numerical floors or data coverage make a gate inapplicable. Any change must be justified using training/development evidence and frozen before the next held evaluation.

### Stage A — correctness and fit capability

First reproduce a few `B_inc` and provisional clean-backbone absolute outputs and finite responses with checkpoint-native normalization and the full physical wrapper. Verify masks, output units, parameter materialization, and a checkpoint/optimizer/RNG round trip after a real update.

Then fit a deliberately small, fixed training panel using the clean full-context absolute model. Verify the explicit backbone/refit transition before training its organizer. Start with values/unaries, then add the observed pair stencils. Allow enough optimization to assess fit: the default cap is **2,000 actual optimizer updates per fit recipe**, subject to the resource envelope below, with reviews after 100/300/1,000 updates. This is not 2,000 epochs and not authorization to continue a failed historical run.

Default fit targets for resolved, non-null roles are median finite-response normalized error below 0.10 and observed mixed-response error below 0.25, alongside a substantial reduction from the zero-response predictor and no nonfinite behavior. Report every family/role; an average must not hide failure of the critical receiver or pressure functional. Near-null signals are assessed by absolute error versus their measured floor, not by an unstable relative target.

These targets are intentionally much more informative than a finite nonzero gradient. A loss remaining near the zero-response loss after repeated fixed-panel updates triggers diagnosis. Check label alignment, units, output cancellation, sampler/query locations, optimizer inclusion of lazy parameters, learning-rate behavior, and gradient flow before changing the scientific model.

Up to three bounded fit recipes are allowed, changing one mechanism at a time. A direct supervised diagnostic readout on the same features/targets can isolate representation and optimizer problems. It is a diagnostic, not an extra deployed bypass. Preserve every attempted recipe and its spent updates. A failure after a single 100-update near-zero pilot is not a general impossibility result; repeated evidence that the model cannot fit is also not a reason for automatic long training.

### Stage B — decision-ready full-access forward model

Train/fine-tune `B_value` and `B_response` from the same chosen initialization with matched optimizer-update and sampled-target budgets. Use identical value samples; count the additional paired reference labels and forward evaluations separately. Checkpoints are selected from calibration/development data using a frozen feasibility-first rule, not only volume MSE.

Promote `B_response` only if it improves the intended finite-response/decision measures or meets a predeclared useful accuracy target while preserving critical field/receiver quality. Default guards are: no more than 5% relative deterioration in the protected near-interface/receiver metric against its matched full-access control, lower resolved-response error than the zero-change control, and no new unexplained tail failure in a module-count stratum. If a different tolerance is required by numerical floors, document it before held selection.

For ThermalChannel pressure, require explicit calibration of absolute and incremental pressure-drop error near the fixed limit. A useful diagnostic scale is one quarter of the declared allowable pressure margin; this is a target for error assessment, not a safety guarantee. Measure false-feasible frequency and its uncertainty on independent calibration families. Four examples or zero observed errors do not establish reliable coverage.

The consequence of a failed Stage B is a forward-model/constraint result. It does not authorize launching the final inverse study with a graph already known to be inaccurate.

### Stage C — train-only oracle cover feasibility

On a bounded set of training families, establish that at least some adequate covers differ in active count and source incidence and that their real work can be smaller than full access. Test actual same-M cases. Preserve a failure case where all-access is needed or where sparse dispatch is slower.

Default promotion requires protected value/response/constraint degradation within the frozen tolerances and a credible cost reduction in the target large-query regime. A useful initial execution target is at least 20% fewer unique fine query–source rows on an eligible large-shape panel without a larger preparation cost that cancels the saving. Row savings alone do not promote a production executor.

If cover discovery merely finds a new fixed K, record that result rather than imposing count diversity. If the oracle remains inaccurate, allow the one bounded candidate-cover remedy described in Section 5. Do not proceed directly to a large amortizer run.

### Stage D — amortized and jointly trained adaptive cover

Train the organizer from train-only cover evidence, then perform matched fine-tuning. Monitor the adequacy-prediction calibration, actual source-support recall on observed effects, over-read rate, count/capacity distribution, gate gradients, and actual work. Report unknown support separately from false negatives.

Use a two-seed confirmation on the selected promising recipe when the budget permits; otherwise state that the result is single-seed. Do not spend all resources on a broad architecture search and leave no budget for the frozen comparison.

Before final inverse promotion, require native case-dependent active counts with adequate responses, safe handling of unsupported constraints, and no material unexplained native support-switch jump on the tested design neighborhoods. Controlled injected-score tests alone cannot establish the native boundary result. If no native switch can be found within the bounded scan, state “native switch unverified,” not “continuous everywhere.”

### Stage E — held design and native WindFarm review

Run the predeclared final ThermalChannel inverse comparison only with the accepted checkpoint and policy. The native WindFarm implementation/evaluation proceeds even if ThermalChannel does not reach this final gate, within its own data and fidelity limits. Finish with a clear promotion/hold/reject decision for the forward model, organizer, sparse executor, and inverse policy separately.

## 9. Controlled comparisons that identify what improved

Do not attribute every improvement to the graph. The minimum comparisons are:

| Comparison | What it isolates |
|---|---|
| Frozen `B_inc` versus new `B_value` | Additional optimization/implementation effects; not a graph result. |
| `B_value` versus `B_response` | The effect of response/receiver/constraint-aware supervision under a matched incremental training budget. |
| `A_dynamic` versus its full-access replay at identical weights | The immediate effect of selected access/controls; not the full training effect. |
| Adaptive cover versus fixed-resolution/geometry-only cover with matched or bracketed actual work | Conditional organization beyond a spatial hierarchy and budget choice. |
| Dynamic cover versus source-membership-shuffled or size-matched random cover at identical counts | Whether group composition matters, not just cardinality. |
| Best adaptive candidate versus a matched refit of the strongest static cover, if the candidate is promising | Whether conditionality remains useful when the static control is allowed to learn. |
| Graph-guided versus size-matched random and ungrouped proposals using the **same accepted surrogate** | Inverse policy value beyond surrogate accuracy. |

Ablations sharing weights are interventions on a trained model, not substitute independently trained controls. Conversely, training every possible control from scratch is unnecessary before a prototype passes fit and oracle-cover gates. Reserve the matched static refit for the selected promising recipe.

Do not select the static K, random seeds, or query tile size from final outcomes. Calibrate static covers and execution fallback on training/calibration shapes. Report both equal-family means and tails, and paired differences by module count. Bootstrap families/layouts, not millions of spatial points or correlated direction rows. Training-seed uncertainty is a separate issue.

## 10. Inverse design: improve feasibility and useful decisions, not just explanations

### 10.1 Two distinct forms of graph benefit

The graph can improve the inverse task through a cheaper accurate forward evaluation, through more useful coupled proposals, or both. Measure these separately. Faster evaluation at unchanged decision quality is a valid compute benefit; it is not evidence that physical pair guidance improved the move. Better objectives with additional oracle calls are not a same-budget benefit.

Keep the structural graph principally about physical/context-conditioned access. The optimizer may rank its groups using the requested objective and constraint sensitivities. Label that second layer as decision-specific usage. Do not train a different “physical graph” for every scalar objective without saying so.

For example, a group priority may combine directional changes in the actual objective, constraint sensitivity, and uncertainty. It must not be based solely on total field-response energy: an energetic wake can be irrelevant to the active objective, while a small pressure error can determine feasibility.

### 10.2 ThermalChannel primary task

Retain the fixed-heating, fixed-radius position task, the maintained solid-temperature objective, actual true peaks, clearance constraints, and maintained inlet/outlet pressure sections. Freeze the pressure limit with respect to the original start as specified by the task. Reanchoring the surrogate must not silently relax the physical limit after every accepted move.

Use a conservative proposal filter based on the predicted constraint and an explicitly calibrated error allowance:

\[
\widehat g(d+\delta)+b_g(d,\delta)\le0.
\]

`b_g` may be a calibration residual quantile or a bounded empirical error estimate within a declared trust region. It is not a universal certificate, and small or shifted calibration populations do not establish a prescribed coverage rate. If reliability is unknown, mark the proposal unsupported and use the reference evaluator or a smaller validated region rather than claiming feasibility.

Retain an independently checked analytic-pressure control only if the local generator's exact assumptions, gauge, query support, and section occupancy are verified. Label it as a **ThermalChannel-specific hybrid constraint control**. It must not be the only pressure result, silently replace the learned model, or be ported to WindFarm as a general physical law.

### 10.3 Trust-region and acceptance behavior

Use the same move normalization, block dimension schedule, inner optimization effort, and candidate budget across graph-guided, size-matched random, and ungrouped policies. Recompute the actual full operator's scalar functional; do not optimize an auxiliary head that has not been checked against the reported field/receiver output.

Within a trust region, permit gradients through continuous coordinates, state preparation, access weights, and controls. Treat any deliberately frozen candidate index as an approximation and test its effect against fully rebuilt finite responses. Shrink or rebuild the neighborhood when physical support/constraint coverage is lost. Avoid hard top-k replacement at a nonzero contribution and avoid unvalidated straight-through gradients as the only design evidence.

Reference acceptance is based on **actual feasibility and actual improvement** of the chosen trial. Predicted improvement determines proposal/ranking and the trust-region ratio, but is not a new veto after a selected trial has been independently shown useful. Log predicted versus actual changes and rejected/failed evaluations.

For the final comparison, prefer selected-only online reference trials. Keep exhaustive candidate-pool evaluation as a separately frozen offline ranking diagnostic. In an exhaustive diagnostic, do not retroactively let one policy accept the best evaluated candidate while the others remain limited to their surrogate selection.

### 10.4 Metrics and failure states

Report reference-validated objective/true-peak improvement at fixed calls and at fixed wall time, feasible proposal and accepted-update rates, false-feasible selections, constraint margin errors, resolved improvement-sign accuracy, candidate ranking within complete common pools, and the cost of preprocessing/organization/gradients/reference correction.

If the selected candidate is physically infeasible, report an infeasible-selection failure. Its unconstrained objective has no feasible regret. A deployed baseline fallback can have its own operational outcome, but must not replace the original failed selection in the ranking table.

Keep the best observed physical opportunity separate from the accepted policy incumbent, as the current code already intends. A graph must beat or meaningfully complement the same-model random/ungrouped controls; changing the surrogate, sampling schedule, or budget between policies would not test graph guidance.

### 10.5 Frozen final design panel

Default final scope is eight new starts, two per M=3/5/7/10, selected by input/geometry criteria before outcomes. Use three policies with at most six reference trials each: **144 policy trials**. Reserve a separate common pool of eight candidates per start for **64 ranking trials**. Each pool is frozen before any of its outcomes; candidates are comparable under the same physical constraints.

These are bounded decision experiments, not global optimization claims. Report start-wise outcomes even if statistical intervals are wide. At least a small baseline/move mesh check is required for any promising nominal-grid claim; otherwise describe the result as improvement under the declared nominal benchmark generator only.

## 11. WindFarm: mandatory same-round transfer, with the actual data contract

### 11.1 Start from the implemented workflow

Use `Case_WindFarm/src/windfarm/model.py`, the existing native reader/sampler, velocity normalization, plugin dispatch, and root training/evaluation entry points. The wrapper already separates `prepare_case` and arbitrary-query `decode`, and materializes lazy projections with real geometry. Extend its architecture capability registration for the new core, with 3-D tests, rather than introducing a parallel network in the case package.

The documented existing baselines are classic fixed K6 and dense pairwise. The larger-sample launch report describes fresh 2102/2103 runs with B16, Q8192, and chunk512; it does not establish their current completion. Read local checkpoints/manifests and preserve any running job. No training process may be stopped or another GPU seized merely to simplify this study.

### 11.2 Preserve observed input and target semantics

The established task uses native velocity `[Ux, Uy, Uz]` with coordinates in rotor diameters, physical D=80 m, hub height 70 m, U_ref=9 m/s, three categorical direction inputs, geometry-only environmental features, and row-specific native support boxes. Retain the documented downstream/crosswind frame; do not apply another wind-angle rotation or invent a continuous meteorological convention.

The compact bundle has `wake_loss_pct`, but its presence does not define a differentiable turbine-power model. The full volume also stores kinematic pressure, turbulent kinetic energy, and dissipation. Their availability does not mean the current velocity model predicts them, and they should not be added wholesale in this round without a specific decision need and a verified target contract.

The same layout appears under three directions. Keep `layout_index` groups intact in splits and uncertainty estimates. Preserve the historical 140/30/30-layout partition for checkpoint replay. Record that its reserved test has already been evaluated and that the current planning has seen its report. Never call reused outcomes an untouched new test. A changed split does not erase a pretrained model's exposure to a layout.

Use mmap/offset-based native access. Do not copy or load the approximately 59.6-GB volume as a single training tensor. A bounded sample and a few streamed native reconstructions are sufficient for this round.

### 11.3 Evaluate what inverse decisions would actually need

Report ordinary volume and rotor-height-band velocity accuracy, but add receiver-focused and downstream-envelope errors. Background-dominated vector relative L2 can conceal the signal relevant to a turbine/wake decision. Use dimensional errors and train-defined normalizations, retaining all three components.

Use predicted velocity at documented rotor-neighborhood receiver sets as decision-related observables. Do not call a velocity-cubed proxy turbine power without the required rotor orientation, quadrature, operating/control state, and power/thrust model. The existing altitude-conditioned training profile contains average wake effects; it is not a verified no-turbine inflow reference.

Continuous yaw optimization is not supported merely because yaw is common in wind engineering: the inspected data/model contract has no established yaw variation. Continuous wind-direction derivatives are likewise not justified by three categories. The first continuous design axis, when an independent reference becomes available, should be a supported layout-position change with the correct fixed physical frame and support convention.

### 11.4 Three evidence tiers; complete the highest feasible tier honestly

**Tier W1 — required now: native forward and organization transfer.** Implement the new cover in the existing 3-D wrapper. Train a bounded adaptation on the actual available WindFarm training split and evaluate against the dense baseline on held-from-training layouts under the documented repeated-benchmark status. Include fixed-M examples, differing layouts/direction categories, native Q8192/Q65536, and E-density sensitivity. This tests architectural transfer beyond ThermalChannel; it does not yet validate continuous inverse gradients.

**Tier W2 — required when comparable library cohorts can be formed: reference-backed finite-library decisions.** Use stored CFD layouts as a discrete feasible candidate library. Construct cohorts matched on turbine count, prescribed operating category/conditions, and any site/geometry constraints that can genuinely be checked. Freeze cohorts from input metadata, not wake-loss outcomes. Compare surrogate-selected versus true stored outcomes and feasibility in each cohort. This tests a bounded selection decision among observed layouts, not generation of a new layout.

The documented `wake_loss_pct` can be a separately supervised empirical observable, with its source definition retained and audited. A scalar head must use the same input-only prepared interaction representation and/or predicted receiver observables; it may not read targets or source-defined per-case outcome descriptors. Compare a scalar-only/static control to identify whether the field/graph is useful. If the source's wake-loss formula can be verified from provided materials, test consistency with its constituent observables; otherwise label the head empirical rather than claiming an exact physical power reduction.

A simple average over the three direction categories is a **three-condition average**, not annual energy production. AEP requires the relevant wind-frequency and operating information, which this contract does not establish. If the three row-specific coordinate frames cannot be reconciled for a common physical layout constraint, keep per-direction library selection and explicitly defer cross-direction continuous design.

**Tier W3 — conditional: continuous reference-validated layout design.** Inventory the data owner's local generator, solver cases, boundary/turbine settings, and permission/cost to solve a new layout. Read-only exported fields and `source_time` metadata are not a solver interface. If a reproducible independent reference is available within explicit authorization, design a small, separately budgeted perturbation panel and reuse the core finite-response/constraint protocol. Otherwise deliver a concrete missing-evidence request and stop the physical inverse claim at W2. Do not substitute a new analytic wake model and call its responses the uploaded CFD truth.

### 11.5 Transfer beyond this one dataset

Keep the reusable core independent of ThermalChannel temperatures, pressure sections, radius 0.45, five field channels, 2-D coordinates, 64 port angles, and twelve proposal slots. Case adapters supply source types, physical coordinate scales, receiver roles/measures, observed constraints, and evidence provenance.

WindFarm is the mandatory large-case stress test here, not proof of universality. Additional modular datasets can use the same contract once their input/receiver/reference semantics are specified. Do not manufacture additional datasets or spend the round building unused adapters.

## 12. Implementation sequence and concrete deliverables

Use the current allocator, config/architecture registry, case plugin dispatch, preparation/decode interfaces, metrics/checkpoint machinery, and artifact gate. Do not build a second experiment-management system. New paths below are proposed homes; reuse an existing suitable module when that avoids duplication.

| Milestone | Implement and execute | Required evidence/deliverable |
|---|---|---|
| M0 — local truth and baseline | Read repository instructions, verify branch/worktree, locate checkpoint-native configs/normalizers/local modules, open the external wind location map read-only, audit old evidence exposure. | `baseline_decision.md`, compact local artifact manifest, one actual checkpoint replay per required incumbent, native wind sample and parameter materialization. |
| M1 — response-ready forward control | Establish the clean three-term full-access/refit control; add an absolute-operator stencil loss/evaluator using the existing typed evidence contracts; test scalar functionals and full design AD path; train bounded full-access fits. | Reusable response/constraint evaluator; saved fit curves; `B_value`/`B_response` checkpoints; role-specific held-calibration metrics and pressure failure analysis. |
| M2 — oracle cover | Add a case-local receiver-anchor cover and typed source incidence with exact full-access mode; implement train-only split/prune/merge adequacy tests and actual pair-union accounting. | Opt-in core such as `interface_fields/adaptive_interaction_cover.py`; oracle covers on actual cases; source/receiver evidence table; same-M count and cost diagnostics. |
| M3 — amortization and execution | Add full-context adequacy/source-support prediction, continuous access/control refinement, packed unique-source reads, rectangular fallback, and safe serialization. | Trained adaptive checkpoint; oracle/amortizer comparison; rectangular/sparse output and gradient tests; native boundary and complete timing report. |
| M4 — wind integration | Register tested 3-D support through `WindFarmForwardModel`/core configuration; run native data training/evaluation and the feasible W2 decision subset. | Native 3-D results and failure cases, layout-grouped metrics, response-evidence availability assessment, and no fake thermal ports. |
| M5 — frozen decision study | If gates pass, run new ThermalChannel common pools and selected-only matched policies, with targeted mesh checks. | Per-start actual decisions/constraints, budget and wall-time traces, selected-versus-best-evaluated distinction, and a forward/organizer/executor/policy promotion decision. |

Maintain new core logic under `HONF_Proj/src/honf_forward_core/`. Keep ThermalChannel functionals/reference adapters in its case package. Keep WindFarm units, native sampling, scalar-observable definition, and dataset location handling in `Case_WindFarm`. Do not import ThermalChannel code into the reusable core or WindFarm model.

One-time orchestration, raw predictions, plots, large JSON/NPZ arrays, per-case atlases, and local runners stay under ignored `diagnostics/generated/` or existing run-owned directories. Reusable algorithms, contracts, regression tests, small configuration files, and the human-readable report may be committed. Do not commit user filesystem paths, local data links, checkpoints, or the external CFD volume.

### Suggested final report structure

Use one maintained report, for example `HONF_Proj/docs/reports/_bk/20260926_195912Z_HONF_Decision_Aware_Dynamic_K_Study.md`, with: baseline decision; reproduced old failure mechanisms; new full-access response fit; adaptive-cover mathematical/implementation contract; held accuracy and cardinality; actual execution; matched inverse results; WindFarm native transfer; limitations and next decision. Include an exact list of retained checkpoints and distinguish selected versus final-epoch weights.

An explanation record should contain physical source IDs, receiver role/region, finite/mixed evidence source and units, numerical floor, validity status, active support, control contribution, and the related decision/constraint. Do not present an unlabeled routing image as this explanation.

## 13. Mandatory test matrix

Add tests that fail for the failure mechanism they are intended to catch. Preserve historical behavior where a new invariant requires an opt-in architectural change.

| Test class | Minimum checks |
|---|---|
| Source/identity invariance | Permute source rows with IDs/features remapped; relabel IDs without changing physics; reverse donor order; perturb a spectator while holding donors/context fixed; mask padded sources. |
| Full-access conversion | Exact or tolerance-bounded parent output comparison for all live field/receiver roles, full wrapper phases, and selected design/parameter gradients; identify any component that requires refitting. |
| Response algebra | Zero delta; explicit four-state inclusion–exclusion; mixed-target dispatcher; signs and physical step normalization; fixed-baseline versus fully reprepared mode; identical common masks. |
| Geometry and receiver semantics | Material coordinates move with their module; Eulerian comparisons use common fluid support; trial absolute evaluation includes newly exposed cells; fixed pressure-section occupancy and clearance checks. |
| Information leakage | Delete targets and host IDs/metadata from the batch and reproduce predictions/organization; changing cached solved values changes only a declared output offset, not input-only topology. |
| Adaptive organization | Exact endpoint active counts, transition counts, same-M case variation, query-order/chunk invariance, capacity overflow/fallback, and no graph selected from held responses. |
| Access/control continuity | Controlled zero/one split limits, continuous receiver overlap, vanishing source support, safe empty-support behavior, and finite one-sided limits across any actually observed native transition. |
| Sparse execution | Deduplicate overlapping source paths; preserve global attention normalization and quadrature; packed versus rectangular outputs and first gradients; report actual padding and fallback work. |
| Units and normalizers | Checkpoint-native thermal transforms/local states; wind inverse transform; scalar functional units; physical versus normalized loss denominators; no per-case target normalization as an input. |
| Learning integrity | Lazy parameters materialized before optimizer; nonzero useful updates; fixed-panel fit; checkpoint/optimizer/RNG resume; saved-step counter distinguished from all executed calls. |
| Reference and ledger integrity | Solver failure, post-solve adapter failure, and serialization failure are different states; unique attempt IDs; no double charge or free retry; missing time remains missing; raw success can be recovered without solving again. |
| Inverse protocol | Original fixed pressure limit survives reanchoring; actual feasibility/acceptance for selected trials; exhaustive best-evaluated diagnostic cannot overwrite the selected-policy incumbent; infeasible selected regret remains undefined. |
| WindFarm | Native offsets/axes and grouping; 3-D configuration serialization/validation; no second wind rotation; target-free E512 preparation; shape/chunk consistency and genuine 3-D query sensitivity. |

The post-solve ledger tests are especially important. The inspected inverse loop increments counters after a returned solve and also has a broad exception path. An exception after a successful physical result must not be interpreted as another physical attempt. Reproduce the precise event sequence before diagnosing any historical total; the consolidated historical ledger remains the source for the reported 286 attempts.

For source relabeling in the old bilinear factor path, construct a two-coordinate counterexample with different cross coefficients. Passing the existing donor-order canonicalization test is not enough to guarantee label invariance. Fix it before reusing the pair parameterization, or retire it from the new main path as this plan recommends.

Do not report “tests passed” as evidence of physical accuracy, learned adaptive K, or inverse utility. Those require the actual data and decision experiments.

## 14. Resource envelope and stopping rules

This is a **new-round** envelope, not permission to consume unrecorded resources from the prior study. The historical 286 reference attempts and missing times remain untouched. Reused stored outputs cost no new solve but retain their provenance.

### 14.1 Compute limits

Use at most **24 aggregate GPU-hours** for this round: approximately two for shared correctness/replay, eight for ThermalChannel response/cover development, ten reserved for WindFarm transfer, and four for selected controls/final evaluation. Record actual resource use and justify any within-envelope reallocation before spending it. Do not consume WindFarm's allocation on repeated toy-case architecture pilots. Parallel GPU jobs each count toward the aggregate; elapsed process time is not a substitute for a different measurement scope.

Use physical GPU 2 for new neural tests/training when it is available and authorized. Inspect occupancy and physical-to-logical device mapping first. Do not assume that a logical `cuda:2` always means the intended physical board after visibility masking. Do not kill existing runs or migrate to other occupied GPUs. CPU-based correctness/data work can continue while neural execution is blocked.

Each formal new training arm has a ceiling of **500 epochs, 20,000 actual optimizer updates, or its remaining time allocation, whichever comes first**. These are ceilings, not objectives. Stage reviews determine whether continuation is useful. No automatic 5,000-epoch continuation, restart that hides prior spent calls, or unbounded hyperparameter sweep is allowed.

The small fit-recipe cap in Stage A is subordinate to this total envelope. Frequent checkpointing is mandatory before lengthy support search or evaluation. Do a zero-update serialization/preflight first; losing another substantial training segment to a schema error is avoidable.

### 14.2 New ThermalChannel reference limit

Authorize at most **512 new local-generator attempts** and **six measured CPU solver-process hours**, stopping at the first binding limit. These limits do not authorize launching new expensive WindFarm CFD. A suggested allocation is:

| Purpose | Maximum new attempts |
|---|---:|
| Training coverage: unaries, several pairs, spectators, and necessary aligned baselines | 160 |
| New calibration/constraint neighborhoods and reliability checks | 80 |
| Frozen common candidate pools: 8 starts × 8 candidates | 64 |
| Selected-only inverse trials: 8 starts × 3 policies × 6 trials | 144 |
| Targeted mesh checks of promising baseline/move decisions | 32 |
| Final-start baselines, replay checks, and bounded recovery reserve | 32 |
| **Total ceiling** | **512** |

Freeze the actual solve manifest for each phase after counting reused corners/baselines. All failed attempts, repeated/tighter solves, and fine-grid solves consume this budget. A completed physical solve followed by an adapter error is one attempted solve, not a free retry or two solves. Preserve its raw output and recover metadata without rerunning when possible. Never impute the prior 17 missing timings.

Unused final-study allocations should remain unspent when promotion gates fail. They may be reallocated to a clearly justified training-only diagnosis before any new held outcomes are opened, but not to an open-ended sequence of additional model recipes. Keep the whole-round attempt/time ceiling fixed.

### 14.3 Goal-mode autonomy

Proceed through the milestones without asking for approval after every ordinary implementation issue. For a numerical, schema, optimizer, memory, or model-fit obstacle, record a short hypothesis, run a discriminating bounded check, and apply a proportionate remedy. Preserve the original result. Training continuation must be justified by a learning curve or diagnostic, not by the assumption that more epochs must help.

Stop a branch of work when its declared gate fails after the bounded remedies, when the budget binds, when required data/permissions are genuinely unavailable, or when the evidence cannot identify the proposed physical claim. Continue other independent deliverables, especially WindFarm integration and the baseline harness, rather than stopping the entire task at the first local obstacle.

A progress note, initialized architecture, finite gradient, or pretty group map is not completion. Conversely, an honest negative adaptive-cover result can be a completed research milestone when the baseline is sound, the mechanism has actually been trained and tested, and the failure is localized.

## 15. Final decision rules

At the end, issue separate decisions rather than a single “HONF succeeded/failed” label.

**Forward model:** promote the response-aware full-access model if it improves useful held response/decision quantities with acceptable field and receiver quality. This is worthwhile even if dynamic K remains unsuccessful.

**Organizer:** promote only with demonstrable case-dependent exact active counts, adequate source/receiver responses, meaningful composition effects beyond fixed/random controls, and no unsupported physical-causality narrative. A computationally useful access cover may be accepted without a unique microscopic graph interpretation.

**Executor:** promote only with measured complete-application or inverse-iteration benefit on the intended shapes, including backward/preparation/packing where used. Maintain dense fallback where it wins. Distinguish fewer rows, less memory, and less wall time.

**Inverse policy:** promote only with reference-backed feasible decisions and a matched-budget benefit or a clearly demonstrated complementary role. Count false-feasible selections explicitly. Do not attribute a unary improvement to a pair hyperedge.

**Cross-dataset claim:** a functioning native WindFarm model demonstrates architectural transfer; W2 demonstrates a finite-library decision; W3 requires independent new-layout reference evidence. Do not merge those claims.

If the dynamic mechanism fails, the report must answer which limit was observed: inadequate full-access response prediction; missing physical evidence; inadequate cover family; poor amortization; unstable native boundaries; no useful sparse execution; or no decision benefit despite accurate responses. “K did not vary” alone is not an adequate diagnosis.

Finish with the selected baseline/checkpoints, actual training/solve/time totals, strongest positive and negative results, unresolved local dependencies, and the one next experiment supported by that evidence. Run the existing pre-push artifact gate, verify the intended diff and tests, commit reusable code/tests/configuration/report changes, and push the non-default branch. Report the actual commit and remote status. Keep raw evidence and one-time runners local.

## Appendix A. Source map and reading order

The report files supplied in the conversation are primary project evidence. The code below was read at `0217a53f6fd4a390c88147f4edc89209bb685f4d`; Codex should inspect subsequent changes before modifying it. A source key in this plan is a document locator, not a claim that its results have been rerun here.

### Reports

- **R1:** `HONF_Interaction_Response_and_Inverse_Design_Study.md`; committed report at `HONF_Proj/docs/reports/_bk/20260926_064922Z_HONF_Interaction_Response_and_Inverse_Design_Study.md`. Read the physical-reference scope, numerical reliability, surviving checkpoint counters, held finite/mixed response tables, M7 selection trace, M3 pressure failure, and final decision.
- **R2:** supplied `HONF_Source_Conditioned_Core_Diagnostics.md`. Read frozen specialization versus fresh Run 1508, the exact-e500/selected-e496 comparison, native-output tradeoffs, source-side organization, and cost scope.
- **R3:** supplied `HONF_Run1507_Task_Trained_Functional_Coalescence_Diagnostics.md`. Read the pilot collapse behavior, e500 fidelity recovery inside universal closure, and lack of native switch evidence.
- **R4:** supplied `HONF_Run1506_Continuous_Functional_Coalescence_Diagnostics.md`. Read the passive score/threshold mechanism and zero retained mergers.
- **R5:** supplied `HONF_Run1503_v3_Converged_Identity_Preserving_Coalescence.md` and `HONF_Run1505_E500_Organizer_Diagnostics.md`. Read actual within-M count variation, boundary jumps, planner cost, and the distinction between access classes and physical sources.
- **R6:** supplied `HONF_1404_1804_1501_1502_Mature_Comparison.md`. Read checkpoint selection, mature dense/sparse comparisons, metric denominators, and the no-additive-coarse/fine-work lesson. Resolve its repository location locally rather than assuming a filename under `docs/reports/`.
- **R7:** supplied `HONF_Conversation_Handoff_Summary.md`, especially Sections 1–2: shared group organization, nonlinear fine interactions, real computation, and group states as controls rather than direct field-value shortcuts.

### Reviewed source paths and symbols

All paths below are relative to `HONF_Proj/`.

- **C1:** `src/honf_forward_core/interaction_response/factor_operator.py`: `_baseline_tokens`, `_edge_context`, `_receiver_tokens`, `_decode_delta_gated_chunk`, `_predict_factor_with_tokens`, `predict_response`, `validity_status`.
- **C2:** `src/honf_forward_core/interaction_response/organizer.py` and `training.py`: `InputOnlySupportScorer.score_factors`, `retention_weights`, `masked_support_classification_loss`, `ResponseTrainingExample`, `stencil_examples_to_torch`, `response_factor_training_step`, `masked_role_response_loss`.
- **C3:** `Case_ThermalChannel/src/channelthermal/inverse/response_factor_oracle.py`: `ResponseFeatureSpec`, `build_response_model_inputs`, `HonfThermalResponseOracle.prepare_baseline`, `with_new_baseline`, and the declared pair-pressure convention.
- **C4:** `Case_ThermalChannel/src/channelthermal/inverse/interaction_guided.py`: decision observation/estimate contracts, predicted candidate selection, `exhaustive_pool` versus selected-only evaluation, selected-trial acceptance, and separate best-evaluated tracking.
- **C5:** `src/honf_forward_core/interface_fields/dense_pairwise.py`: `prepare_fine_messages`, module/environment reads, projected source preparation, and fine nonlinear evaluation.
- **C6:** `src/honf_forward_core/interface_fields/source_conditioned_pairwise.py`: `SourceConditionedRouter`, source-control preparation, full QM/QE reads, weighted softmax, and diagnostic counts.
- **C8:** `src/honf_forward_core/interface_fields/core.py` common-context construction and `three_term_context.py`: architecture-dependent shared versus three-term context, zero local branch, and the final field head.
- **C7:** `Case_WindFarm/src/windfarm/model.py`: `build_windfarm_forward_config`, `WindFarmForwardModel`, `prepare_case`, `decode`, physical inverse transform, and `materialize`.

Additional case wrappers, training dispatch, support-search implementations, caller-side loss assembly, and local one-time runners must be traced by Codex as needed. This plan does not claim an exhaustive audit of every source file in the repository.

### WindFarm evidence

- **W1:** `Case_WindFarm/Dataset/PHYSICS_AND_DATA.md`: actual compact/ragged schemas, units, categorical directions, coordinate-frame warning, independent layout count, and masks.
- **W2:** `Case_WindFarm/docs/WINDFARM_FORWARD_INITIAL_STUDY.md`: existing learned inputs, native quadrature, split/normalization, prepared-state implementation, selected 2100/2101 results, and execution measurements.
- **W3:** `Case_WindFarm/docs/WINDFARM_FORWARD_LARGER_SAMPLE_RUNS.md`: larger-sample 2102/2103 launch contract and retained historical runs; verify present local state separately.
- **W4:** `Case_WindFarm/README.md` versus the implemented wrapper and later reports: the introductory preprocessing-only description is outdated relative to this code snapshot. Correct documentation only within the relevant scope.

## Appendix B. External methodological context, not replacement evidence

These primary sources inform the proposed response/decision and scaling directions. They do not establish that the new HONF method is novel, that its graph is physically causal, or that its cost/accuracy gates will pass.

**E1.** Thomas O'Leary-Roseberry, Peng Chen, Umberto Villa, and Omar Ghattas. *Derivative-Informed Neural Operator: An Efficient Framework for High-Dimensional Parametric Derivative Learning*. arXiv:2206.10745, submitted 2022, revised 2023. The relevant principle is joint approximation of an operator and its input derivatives, using compressed derivative information rather than requiring a full dense Jacobian loss. Use this as motivation for directional response/sensitivity supervision; generate or verify the needed project-specific labels independently.

**E2.** Dingcheng Luo, Thomas O'Leary-Roseberry, Peng Chen, and Omar Ghattas. *Efficient PDE-Constrained Optimization under High-Dimensional Uncertainty Using Derivative-Informed Neural Operators*. arXiv:2305.20053, 2023. The relevant principle is making optimization-variable sensitivities part of surrogate construction and evaluating the resulting optimization, rather than selecting a model only by state reconstruction error. Its reported optimization results do not predict HONF performance.

**E3.** Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Andrew Stuart, Kaushik Bhattacharya, and Anima Anandkumar. *Multipole Graph Neural Operator for Parametric Partial Differential Equations*. NeurIPS 2020. The relevant principle is accounting for long-range interactions in a scalable multilevel construction. AIC-HONF's interaction cover is a different proposed mechanism; no multipole exactness or linear-complexity result is assumed.

---

**Completion criterion:** a stronger, decision-tested forward foundation and an actually tested adaptive interaction mechanism on the path to native WindFarm use—not another attractive K statistic attached to an inaccurate local model.
