# HONF research reset: interaction evidence and inverse-design utility

## Codex Goal-mode work plan

**Working branch:** `agent/honf-core-next`  
**Repository inspected for this plan:** `cosmos2w/ModularDT`, `e1d91fb`  
**Status:** Proposed research, not a tested architecture or an authorization to claim physical validation.  
**Run identity:** Allocate new identities using the existing run machinery. Do not rename, resume, or overwrite Runs 1500–1508.  
**Primary question:** **What physical interaction evidence must each hyperedge explain, and how does that explanation improve an inverse-design decision?**

---

## 0. Read this first: the objective has changed back to the original HONF objective

HONF is an **inverse modular-design representation-learning project**. Multi-physics field prediction is the learning instrument. A useful forward surrogate is necessary, but beating Dense on mean field error or single-call latency is not the research objective.

The desired result is an adaptive, physically anchored, queryable interaction representation that helps explain and predict the consequences of design changes. It must eventually improve at least one meaningful inverse-design outcome: reliable choice of coupled design variables, ranking of candidate changes, constraint handling, solver-query efficiency, or transfer to new module layouts or operating conditions.

Do not respond to this plan by implementing another version of the twelve-prototype merger and running it to 500 epochs. Do not optimize Run 1508 into a faster static predictor as the main task. Keep those models as controls.

This round has three linked deliverables:

1. **Interaction evidence:** a small, well-defined collection of physical responses to controlled design changes, with numerical reliability and provenance recorded.
2. **An evidence-carrying adaptive hypergraph:** each retained factor has physical source identities, a declared receiver/output role, a validity neighborhood, and measurable response information it explains.
3. **A controlled inverse-design experiment:** test whether the inferred grouping helps decisions beyond the same surrogate used without that grouping.

An honest result may be that some proposed factors are unnecessary, that only pairwise factors are justified at the sampled scales, or that reference data are insufficient. Those are useful outcomes. An attractive graph, lower effective K, or accurate prediction alone is not a successful answer.

### Recommended first prototype

Develop a **response-factor HONF**: a local, baseline-conditioned multi-physics response operator whose design-change information passes through physically identified factors. This is a deliberately bounded identification experiment, not a claim that the final absolute-field HONF architecture has been solved.

Its baseline field can come from a reference solution or a frozen existing surrogate. The graph must explain **all changes from that baseline**. A dense evaluation at the perturbed design must not bypass the graph.

After that representation demonstrates physical and inverse value, use the findings to redesign the absolute-field HONF. Do not tackle a new global field architecture, a new inverse generator, a new CFD solver, and a new sparse GPU kernel simultaneously.

---

## 1. Evidence that motivates the reset

The following are repository/report observations, not new hypotheses. See evidence references [E1]–[E6] at the end.

- Run 1505 produced genuine case-dependent coalescence: P2 R=9/10/11/12 in 4/53/28/5 cases at e500. Its accepted mergers could be stable numerically, but its reported trained design boundary had a finite output jump and its organization cost was large. This proves that adaptive count is implementable, not that the recovered classes were physically identified.
- Run 1506's source-action thresholds remained outside the observed score range. No saved checkpoint formed a functional merger. Its score was a worst-probe/mean-energy ratio, not a calibrated physical error bound.
- Run 1507 closed all distinctions at its saved checkpoints, yet improved aggregate field prediction after continued training. It retained source-conditioned controls and fine physical interactions. Thus the training objective did not require useful conditional query organization.
- Run 1508 successfully specialized that static operation. Its exact algebraic saving and source-control ablations are valuable controls, but its topology is fixed. Source-input dependence does not identify a causal physical hypergraph.
- The existing fine MM/ME/EM preparation mixes information across physical sources before later routing. Sparse access to such tokens does not identify sparse dependence on the original physical sources.
- The historical 90-case `test` split has repeatedly supplied development validation. Train case 0001 and development case 0273 are a disclosed cross-split physical duplicate; the 89-case sensitivity is also development evidence.
- Reported native-input AD/FD checks were surrogate self-consistency checks. The latest report did not generate new solver truth.

### Preserve

Preserve the local physical encoders and local-surrogate machinery where valid, typed physical outputs and normalization, fine source identities, useful source-conditioned control maps, exact source-moment/quotient utilities, reliable evaluator components, and existing checkpoints. Preserve module permutation handling and the distinction between algebraic equality and physical fidelity.

### Pause

Pause static-surrogate leaderboard optimization, another online ADMM or fixed-tree merger, count-only regularization, R-variance rewards, new custom attention kernels, and claims based on dominant-color regions. Do not delete historical implementations.

### Explicitly relax earlier constraints

- K=12 and the old proposal tree are not fundamental. A practical candidate ceiling is allowed; a fixed number of active interaction factors is not the intended result.
- Exact parity with Run 1502 is not required for a new scientific model. It remains required for a claimed execution-only rewrite.
- Dense pre-contextualization is not protected. Retain it as a baseline or baseline-state encoder only where its information role is explicitly disclosed.
- “Never pool” becomes “do not discard response information before testing its adequacy.” Interface/transfer-mode compression is allowed and must be assessed on response reconstruction, not only retained variance.
- Forward accuracy and latency are adequacy/resource criteria. They are not automatic vetoes on a graph that improves reference-validated inverse decisions.
- Discrete topology is allowed in an outer design loop. Do not differentiate blindly through a changing discrete plan or claim global continuity from fixed-plan gradients.

---

## 2. Research hypotheses and their falsifiers

### H1 — Responses contain interaction evidence that static field snapshots do not identify adequately

Predicting fields together with feasible design-induced changes should produce a more useful interaction representation than predicting the snapshots alone.

**Test:** hold architecture, supervision budget, split, and normalization as close as possible; compare field/value training against value-plus-response training. Measure held-out directional changes, joint changes, and candidate-design ranking.

**Falsifier:** the response-trained version does not improve those quantities, or the result disappears after matching reference data and model capacity. Do not equate a better training fit with identification.

### H2 — Physically identified factors can represent conditional collective responses economically

The important object is not a prototype label. It is a source-set-to-receiver response: specified module changes jointly influence specified field/port quantities in a stated design neighborhood.

**Test:** compare adaptive unary/pair factors with a pairwise-only/local control, a fixed support control, and an all-candidate response model. A small higher-order expansion is permitted only when measured mixed-response residuals motivate it.

**Falsifier:** a fixed or unary-only response model performs equally well under matched capacity/data, or adaptive support is merely a proxy for M or a case identifier.

### H3 — The organization helps inverse decisions, not just explanation plots

**Test:** use the same response surrogate with graph-guided coupled updates and with size-matched random/unstructured updates. Keep candidate budget, reference calls, constraints, and optimizer effort matched.

**Falsifier:** the graph changes pictures but not decision quality; any gain is explained by more candidate evaluations or greater block dimension.

### H4 — A useful graph is stable at the level of physical response, not necessarily at the level of labels

**Test:** reorder modules, perturb baseline conditions, change field sampling, and compare predicted responses and physically indexed supports. Do not compare latent column numbers.

**Falsifier:** purportedly important factors depend on arbitrary ordering or disappear under small representation changes while reference responses stay the same.

There is no assumed unique ground-truth hypergraph. Report an **effective, scale- and tolerance-dependent response organization**, with uncertainty and alternative adequate supports when appropriate.

---

## 3. Repository and reference-capability work: execute, do not infer from names

### 3.1 Inspect the actual local branch

Read `AGENTS.md`, `.githooks/pre-push`, the current working-tree changes, relevant report files, and the existing resource/run configuration. Use normal Git history to record the code version. Do not add hashes or snapshot infrastructure.

Relevant existing paths and local lookup targets include:

```text
HONF_Proj/src/honf_forward_core/interface_fields/dense_pairwise.py
HONF_Proj/src/honf_forward_core/interface_fields/sparse_incidence_router.py
HONF_Proj/src/honf_forward_core/interface_fields/source_conditioned_pairwise.py  # proposed locator: verify the actual static-backend filename locally
HONF_Proj/Case_ThermalChannel/src/channelthermal/interface_field_coupling.py
HONF_Proj/Case_ThermalChannel/src/channelthermal/training/epoch.py
HONF_Proj/Case_ThermalChannel/src/channelthermal/inverse/differentiable_verifier.py
HONF_Proj/Case_ThermalChannel/src/channelthermal/inverse/
HONF_Proj/Case_ThermalChannel/src/channelthermal/workflows/
HONF_Proj/Case_ThermalChannel/Dataset/PHYSICS_AND_DATA.md
HONF_Proj/src/honf_inverse_core/
```

Treat a path marked for local lookup as a locator, not a claim that its exact spelling was verified. Reuse existing public interfaces where they fit. Keep new response logic separate from legacy architectural dispatch.

### 3.2 Distinguish four evaluation levels

| Level | What it is | Permitted interpretation |
|---|---|---|
| Reference physical solve | Independently executed governing-equation solution with adequate numerical convergence | Reference response and design validation within its model/mesh scope |
| Stored physical solution | Existing dataset target at exactly that physical input | Baseline reference; not a solution at a newly perturbed input |
| Frozen learned teacher | 1804/1502/1508 or another trained surrogate | Bootstrap labels, proposal screening, model comparison; not CFD truth |
| Algebra/synthetic benchmark | Constructed analytic response or toy solver | Implementation/mechanism checks only |

The inspected `DifferentiableVerifier` explicitly wraps frozen local/global HONF surrogates. Do not relabel it as a physical solver. It also contains architecture/native-autograd restrictions. Do not set override flags globally or remove those restrictions to force a new method through. Add a narrowly tested integration or use a separate response adapter.

Find the real simulation entry point, input format, mesh generation, boundary conditions, and output extraction in the authorized local environment. Historical reference-request queues do not prove executable solver availability.

### 3.3 First actions with the reference capability

Execute one existing baseline case and one feasible perturbation using the actual reference workflow. Verify output units, field order, geometry, pressure gauge, masks, interface locations, and convergence. Record actual runtime and solver settings.

A `--dry-run`, imported module, fabricated output, or a finite neural forward is not this evidence.

If the physical solver cannot run, try at most three concrete remedies within 90 minutes total: resolve an existing configuration/path, use the documented local solver adapter, or execute the supplied reference workflow in its intended environment. Do not install an unrelated CFD stack, acquire licenses, start cloud resources, or rewrite the solver.

If still blocked, continue implementing the response data adapter, mechanism tests, and teacher-only pilots. Produce a precise local request list for the missing reference runs. Label all resulting structure and inverse results **teacher-supported**, not physically validated. Lack of reference execution limits the conclusion; it need not stop all software work.

---

## 4. Define the first physical problem and inverse task

### 4.1 State, design, and operating inputs

Let

\[
y=\mathcal F(d,c),\qquad
 y=(u,v,p,\omega,T,\text{ports},T_{\rm solid},q_n).
\]

Here d contains active module positions and, in the secondary task, admissible module heating allocations. The context c contains prescribed operating/boundary conditions. For this dataset, Reynolds number has observed variation; material constants without observed variation are not a demonstrated generalization axis [E1].

The primary design task in this round is **position optimization at fixed module identities, radii, heating, and operating context**. Keeping total heating fixed prevents the trivial solution of reducing the heat load. A secondary fixed-total-heat redistribution test is useful but must not replace the geometric task.

### 4.2 Objective and constraints

Minimize a smooth approximation to peak module temperature during surrogate search:

\[
J_\tau(y)
=\tau\log\left(\frac1{N_T}\sum_{a=1}^{N_T}\exp(T_a/\tau)\right).
\]

Use a stable log-sum-exp implementation. \(\tau\) has temperature units; choose it from the training-only temperature scale and required decision resolution, not the final test results. Report the true reference maximum and per-module temperatures in addition to this smooth search objective.

Subject to:

\[
\Delta p(d,c)\leq\Delta p_{\max},\quad
\|x_i-x_j\|\geq r_i+r_j+g_{\min},\quad
x_i\in\Omega_{\rm admissible}.
\]

Use the repository's physically defined pressure-drop quantity. If none exists, define fixed inlet/outlet sections, area/flux weighting, sign, and units before comparing methods. The sections must remain valid under every admissible move. Do not infer pressure drop from an arbitrary pair of pixels.

Keep existing engineering limits when present. Otherwise use a clearly identified **relative benchmark constraint**, e.g. a pressure-drop budget relative to each reference baseline, chosen before held-out optimization. Do not present a benchmark percentage as an engineering safety limit.

### 4.3 Why local response is the first target

At an accepted baseline \((d_0,c)\), define

\[
\Delta y(\delta;d_0,c)
=\mathcal F(d_0+\delta,c)-\mathcal F(d_0,c).
\]

The first prototype predicts this full multi-physics response, not merely a scalar objective gradient. Its baseline is reused within a bounded neighborhood.

This local formulation makes the graph's obligation precise: **every dependency on the trial change \(\delta\) must pass through the declared response factors**. Dense baseline information may characterize the already-existing physical state, but a dense perturbed-input forward must not provide a hidden alternative path.

This is not yet a replacement for cold-start absolute-field prediction. Report the cost and provenance of obtaining each baseline, the graph construction, and every response query. The final absolute-HONF redesign is a later research decision informed by this experiment.

---

## 5. Build a small physical-response evidence dataset

### 5.1 Split by physical design family, not by perturbation row

Keep historical benchmarks unchanged. Build a separate response experiment with:

- 8 training anchor families, balanced across available M=3/5/7/10 strata;
- 4 calibration/validation anchor families, one per stratum where available;
- 4 genuinely new final-review anchor families, one per stratum where available.

An anchor family includes all its perturbations, repeats, mixed stencils, and nearby optimization starts. They cannot cross splits. Use existing duplicate detection and physical case metadata. Do not refit normalization using calibration or final-review outputs.

A held-out perturbation of a geometry that the frozen teacher already trained on is **neighborhood-transfer evidence**, not unseen-composition generalization. A new-layout claim requires a new baseline layout outside every component's training data. A previously examined development case is not a final holdout. Select feasible final-review inputs before inspecting their response outputs.

If only known designs are available, retain the same split discipline and label the weaker scope rather than inventing an untouched test set.

### 5.2 Minimal stencil per anchor

Select two active modules and feasible position directions using geometry/operating descriptors, not candidate model errors. Cover close/aligned pairs and separated/non-aligned pairs across anchors; do not always choose the nearest pair.

For two normalized directions v_i and v_j, execute the 3×3 stencil:

\[
d_{ab}=d_0+a h_i v_i+b h_jv_j,
\qquad a,b\in\{-1,0,1\}.
\]

This uses nine solutions including the baseline, and supplies single and joint changes with both signs. Keep all other physical inputs fixed and re-solve the coupled physics for geometry changes.

Add two feasible fixed-total-heating redistributions (positive/negative) per anchor if supported. Heating redistribution directions must sum to zero over active modules.

For a small train-only null panel, also test independent heating perturbations and their combination. A zero mixed heating response under effectively linear fixed transport is a legitimate negative result. Do not demand nonlinearity merely because the architecture uses hyperedges.

For a small train/calibration subset, add Reynolds-context changes or half-step stencils. Do not claim Reynolds extrapolation when both states remain inside the observed range.

### 5.3 Step sizes and reference uncertainty

Work with dimensionless design coordinates \(\widetilde d\), with physical scales such as module radius, domain size, and a heating reference explicitly recorded.

Start from a feasible move that is meaningful for the inverse task, then compare h and h/2 on a small panel. A step that produces a response below solver/interpolation uncertainty is not evidence of zero interaction.

Use bounded refinement: at most four step adjustments for a failed axis. Use one-sided differences when geometry makes a central stencil impossible and label them. Preserve both solved sides even when a derivative estimate is unresolved.

Estimate a numerical response floor using repeated/tighter-solve or mesh-refined checks where available. Propagate that uncertainty to response norms and signs. Do not impose the old requirement that the learned source-incidence support remain unchanged: that would censor the very physical neighborhoods in which adaptive structure should be tested.

### 5.4 Compare moving geometries correctly

For Eulerian field differences, evaluate every member of a stencil on a **common physical query set** lying in the common fluid domain. Use a common mask for the entire stencil, not a different per-model or per-side mask. Report the excluded area and separately retain interface-attached measurements so near-interface effects are not simply discarded.

For ports and solid fields, compare material coordinates: module ID and angle for ports, normalized local coordinates for solids. Their physical sampling coordinates move with the module.

Keep pressure gauge and reference sections consistent. Measure interpolation error separately where meshes differ. Do not subtract arrays simply because their shapes match. A reference-domain pullback is an alternative when already supported; distinguish material and Eulerian derivatives [R5].

### 5.5 Response targets

For a single direction:

\[
D_i^h y=\frac{y(d_0+h_iv_i)-y(d_0-h_iv_i)}{2h_i}.
\]

For a mixed directional response:

\[
D_{ij}^{h_i,h_j}y
=\frac{y_{++}-y_{+-}-y_{-+}+y_{--}}{4h_ih_j}.
\]

Also retain the finite anchored interaction, without dividing by small steps:

\[
I_{ij}(a,b)
=y(d_0+a v_i+b v_j)-y(d_0+a v_i)
-y(d_0+b v_j)+y(d_0).
\]

Train initially on finite changes and anchored interactions; use derivative labels only when numerically resolved. This avoids requiring expensive second-order automatic differentiation through the entire physical wrapper in the first implementation.

A nonzero mixed response indicates effective collective response at the sampled scale. It does **not** prove a fundamental three-body constitutive law. Nonlinear pairwise physics can generate it [R4].

### 5.6 Data record

Use a small typed record and existing serialization conventions. Suggested fields:

```text
anchor_id, physical_family_id, split, evidence_source
base_design, operating_context, active_module_ids
perturbation_by_module, physical_step_scales
query_coordinates, common_fluid_mask, quadrature_weights
port_module_ids, port_angles, local_solid_coordinates
base_outputs, perturbed_outputs, finite_changes
single_response_labels, mixed_response_labels, resolved_label_masks
reference_solver_settings, numerical_uncertainty, solve_status, elapsed_time
```

`evidence_source` must distinguish `reference_solver`, `stored_reference`, `surrogate_teacher`, and `analytic_synthetic`. Missing/unresolved labels are masked, never set to zero. Do not collect predicted fields as though they were independently solved targets.

---

## 6. Define what a response hyperedge means

### 6.1 A physically indexed factor

Represent an edge as

\[
e=(S_e,\mathcal R_e,t_e,\mathcal U_e),
\]

where:

- \(S_e\): identified module design-variable blocks whose **changes** enter the factor;
- \(\mathcal R_e\): receiver support in physical space and/or interface ports;
- \(t_e\): declared output/transfer type, initially hydrodynamic or thermal/interface;
- \(\mathcal U_e\): design/context neighborhood and numerical resolution over which evidence was obtained.

Directed transfer is allowed. Do not impose reciprocity on convective transport without a physical reason. Overlapping supports are allowed. A fixed environment sampling grid is a numerical support, not a learned group count.

The learned graph \(\mathcal H(d_0,c)\) is shared across inverse objectives at that physical baseline. Query selection may depend on q and requested output. Changing the optimization objective should not change the predicted physical response at the same d0, c, and delta.

### 6.2 Evidence packet per retained edge

For every claimed useful edge report:

1. Which single/joint perturbations and receiver quantities it explains.
2. Its contribution to held-out response reconstruction, including sign, magnitude, and uncertainty.
3. What happens when that edge is removed computationally, and whether other factors can compensate after a bounded refit.
4. Whether a pairwise/unary alternative explains the same evidence at similar cost.
5. Whether using its module set improves a coupled inverse update or candidate ranking.
6. What inputs/neighborhoods remain unsupported.

Membership visualization is an index into this packet, not the evidence itself. Removing an edge computationally is not the same as physically removing a module. A graph can have useful effective interactions without a unique microscopic causal interpretation.

### 6.3 Adaptive K

Define

\[
K(d_0,c;\varepsilon)=\#\{e:\text{retained response factor}
\text{ at the specified adequacy tolerance}\}.
\]

Report separately active unary factors, joint factors, numerical candidate capacity, query-active factors, receiver/source incidence, and executed work. Do not call the number of physical modules or environmental patches adaptive K.

The same R in every case is not automatically wrong, but it does not demonstrate adaptive count. The project must ultimately show meaningful input-dependent organization, not impose a count-diversity loss.

---

## 7. Primary model: anchored adaptive response-factor HONF

This is the chosen first method. It is a testable proposal, not a proven optimal architecture.

### 7.1 Decompose the change, not the baseline physics

Use

\[
\widehat y(d_0+\delta,c)
=y_0+\widehat{\Delta y}(\delta;d_0,c),
\]

\[
\boxed{
\widehat{\Delta y}(q,\delta)
=\sum_i R_i(q,\delta_i;b_i)
+\sum_{i<j} g_{ij}(d_0,c)R_{ij}(q,\delta_i,\delta_j;b_{ij})
+\sum_{S:\,|S|=3}g_S(d_0,c)R_S(q,\delta_S;b_S).
}
\]

Start with unary and pair-of-design-block response factors. A pair of perturbed donor modules together with their receiver support is already a multi-entity directed response hyperedge. Only add the |S|=3 response expansion when unresolved train-only evidence motivates it; do not start with all possible orders.

The baseline y0 is held fixed during an inner inverse step. Its approximation error is measured separately. In the default model, baseline features are built consistently from design/context and the chosen frozen baseline encoder in both training and inference. Reference y0 supplies the physical output offset and evaluation; feeding its solved field into the encoder is a separately labelled observed-state variant, not a silent change of inputs. At deployment in the reference-corrected experiment, y0 comes from the accepted reference baseline. In a fully surrogate run it may come from 1502/1508, but that is a separate evaluation mode.

The response operator predicts the five-field distribution plus declared port/interface outputs. Do not train only a scalar objective and call it a learned multi-physics operator.

For the initial candidate, predict the joint **final-output increments directly** with typed query heads. Reuse the existing P0/P1/P2 wrapper to obtain a baseline and comparators, not as an unrestricted perturbed-design shortcut. Calling that wrapper at d0+delta inside the response head would defeat the information-path experiment. If later coupling to a local physical decoder introduces additional interactions, explicitly update the provenance and additive-null tests rather than claim the original decomposition still holds unchanged.

### 7.2 Exact anchored interaction parameterization

Construct an order-|S| factor from a shared decoder psi by inclusion–exclusion:

\[
\boxed{
R_S(q,\delta_S;b_S)
=\sum_{U\subseteq S}(-1)^{|S|-|U|}
\psi_{|S|,t}(q,\delta_U,0_{S\setminus U};b_S).
}
\]

For unary factors this is psi(delta)-psi(0). For pairs it uses four small decoder evaluations. For triples it uses eight. Batch the sign variants; do not execute Python loops over queries.

This gives the exact property

\[
\delta_i=0\text{ for some }i\in S\quad\Longrightarrow\quad R_S=0.
\]

Thus a pair factor cannot silently become a unary contribution. Also \(\widehat{\Delta y}(0)=0\) regardless of weights. A double-precision toy check and real float32 tests must verify cancellation and gradients.

All sign variants of one factor share the **same cached baseline context, graph, and receiver support**. Recomputing context at each perturbed sub-input would invalidate this interpretation.

Anchoring alone does not prove a unique physical decomposition. Reference mixed labels, unary controls, and held-out reconstruction are still necessary.

### 7.3 Source and receiver encoding

Reuse suitable existing module/local physics encoders for baseline features. Use shared, permutation-equivariant set encoders for donor groups and identifiable receiver queries. Features can include baseline module properties, local port summaries, relative geometry, prescribed context, and baseline field/response features.

Do not pass an entire **trial** layout, modified global heating summary, or trial dense field into each factor. Factor e may see delta only for modules in S_e. Baseline context may contain collective base-state information, but record its provenance; it is conditioning information, not evidence that the factor is globally local.

Start with full port angular outputs and the existing physical field queries. Optional port-mode compression comes later, with both port-value and response reconstruction checked. Latent moment conservation is not physical conservation.

For receiver localization, use a small shared head w_e(q;d0,c,t) with baseline geometry, receiver type, and factor identity. It receives no trial delta or inverse objective. Apply it outside the anchored decoder, so all inclusion–exclusion terms share the same receiver weight:

\[
\widehat{\Delta y}(q,\delta)=\sum_e g_e\,w_e(q)\,R_e(q,\delta_{S_e};b_e).
\]

The default staged implementation first uses w_e=1 to establish correct source-set response factors, then trains receiver support from the same available response fields. A simple continuous exact-zero parameterization is squared-ReLU/(1+squared-ReLU), as for source-set retention. Support supervision uses whether the measured response energy at that receiver exceeds its numerical/decision floor over the resolved stencil; it must not mark unmeasured or unresolved receivers as physically uninfluenced. The response decoder retains signed amplitudes; the nonnegative support weight is not the response itself.

Report K_q as the number of factors with g_e w_e(q)>0 separately from case-level K. Never normalize these weights across all factors in a way that creates undeclared trial coupling. Exact zero supports may skip factor decoding; a post-computation mask is not skipped work. Keep an all-receiver control and measure omitted far-field/port responses. If receiver learning is not reliable within the budget, retain w_e=1 and report **source-set adaptation only**, not successful query-specific minimal retrieval. Do not hard-code wakes, hard nearest-neighbor boundaries, or symmetric influence.

### 7.4 No hidden trial-response bypass

The following tests are mandatory:

- The cache built at d0 is unchanged when evaluating different delta in an inner step.
- Perturbing delta_j outside S_e leaves the **individual factor** e bitwise or numerically unchanged.
- With joint factors disabled, the response is additive in the remaining unary blocks; measured mixed finite differences should be zero up to the tested numerical tolerance.
- Removing all factors produces zero predicted change, not another dense prediction.
- No case ID, future perturbed target, or final inverse objective enters factor membership.

This is how the prototype avoids the globally contextualized-token shortcut for the information it claims to organize. It does not assert that the baseline solution required only those factors.

### 7.5 Computational starting point

For M<=12, all unordered pairs give at most 66 candidate joint source sets. This is a manageable candidate ceiling and is not a fixed active K. Share decoder parameters across physical sets rather than create a network for each pair ID.

Use Q256–1024 while debugging and Q8192 for final field comparisons, with existing query chunking. The baseline is prepared once per outer step, not once per query or inclusion–exclusion term.

Do not materialize B×Q×candidate×E×hidden tensors. Batch small sign variants and factor blocks; aggregate field channels as they are produced. Report candidate scoring, selected factor evaluation, baseline preparation, and complete inverse iteration separately.

Once factors are exactly zero at a fixed baseline, their response decoding may be skipped. If a padded implementation is faster, retain it but report its real executed work. Do not claim physical sparsity merely from zero masks.

---

## 8. Primary adaptive-K method: evidence-guided support discovery and amortization

Do not immediately recreate the hard-concrete root-collapse objective.

### 8.1 Learn response functions before asking a count penalty to prune them

First fit the shared unary/pair response decoders on training values and finite-response labels with broad candidate availability. Pair heads have the anchored construction above. This stage supplies a usable response dictionary and an all-candidate upper-capacity control.

Keep this stage short enough to be a pilot. It need not achieve mature full-field accuracy before the evidence-selection step can be tested.

### 8.2 Offline train-only sparse support discovery

For each training anchor with adequate reference labels, seek a small response support G satisfying **separate adequacy conditions** for field changes, ports, and relevant joint changes:

\[
\min_G\mathcal C(G)
\quad\text{subject to}\quad
E_{\rm field}(G)\leq\varepsilon_F,
\quad E_{\rm port}(G)\leq\varepsilon_P,
\quad E_{\rm mixed}(G)\leq\varepsilon_I.
\]

This is a scientific selection objective, not a blocking software gate. If no tested support satisfies the conditions, retain the best observed tradeoffs and report insufficient capacity/evidence rather than invent a feasible graph.

Default solver: a **bounded grow–refit–prune search** over physical candidates.

1. Begin from fitted unary response factors.
2. Rank candidate additions by estimated response-residual improvement per incidence/work cost.
3. Evaluate the top three additions with actual model responses against the stored reference labels; a gradient ranking is not itself the measurement.
4. Accept a useful addition, perform a short bounded refit of the relevant shared response parameters, and recompute the residual.
5. Allow at most one two-factor lookahead when individually weak but complementary additions are suspected.
6. Perform backward deletion tests and keep a smaller support when adequacy is retained.

Limit search per anchor to at most 12 accepted additions and 40 actual candidate evaluations in the first pilot. These are exploration budgets, not target graph sizes. If the budget is exhausted while residuals remain material, report a truncated search.

A convenient cost is

\[
\mathcal C(G)=\sum_{e\in G}
\left[c_0+c_S|S_e|+c_R N_{\rm receiver,e}+c_Zd_e+c_W\widehat W_e\right].
\]

Choose consistent dimensionless normalizations and expose all components. Initial coefficients can be simple; they are not wall-time promises. Count state capacity and receiver extent so that one giant factor is not artificially free.

This offline support is a **teacher support chosen under a particular fitted decoder and finite evidence**, not ground-truth causal membership. Alternative adequate supports should be retained as alternatives where search finds them. Unmeasured candidates are unknown, not negative labels.

### 8.3 Amortize support from current physical inputs

Train one small shared candidate scorer on baseline features to predict the teacher's retention/marginal-response evidence. It receives d0 and c, never reference outputs at the perturbed trial design and never an anchor ID.

Use a live classification/ranking loss for retention as well as response reconstruction. That supervision gives a learning signal even when the runtime factor coefficient is currently zero; it avoids Run 1506's flat, unsupervised acceptance band.

One initial continuous coefficient is

\[
a_e=\operatorname{ReLU}(s_e-\tau),\qquad
 g_e=\frac{a_e^2}{1+a_e^2}.
\]

This is exactly zero below the threshold and has zero first derivative at entry. Apply the classification loss to the unthresholded score s_e so an inactive edge can learn to reopen. Calibrate the shared threshold using training/calibration response adequacy, not a desired K histogram. Do not claim that the coefficient gives a physical probability without calibration.

Freeze g_e and the baseline context for each local inner design model. Rebuild them at an accepted outer baseline. The selected set is deterministic for evaluation; no stochastic mask averaging supplies the final graph.

### 8.4 Joint refinement without another collapse objective

After amortization, refit response decoders and the scorer using response prediction plus teacher-support evidence. At most two train-only teacher-refresh rounds are permitted in the initial study.

Do not add a large scalar E[K] term as the primary teacher. If support regularization is needed, use the measured adequacy/cost tradeoff and report whether it removes useful response factors. No entropy bonus or R-variance penalty is allowed simply to create conditionality.

A static support with the same typical size is a mandatory control. If it performs as well, the adaptive part has not earned its claimed value.

### 8.5 Bounded alternatives, not a menu to implement in full

Codex may replace the greedy support teacher with **group-sparse regression on frozen response features** if the nonlinear candidate evaluation is too expensive or unstable. Fit group coefficients against signed field/port response labels, use incidence-weighted group costs, and select regularization on calibration response adequacy. Report that this is a restricted linearized discovery procedure.

If anchored pair responses leave material mixed residuals, admit a **small train-evidence-selected set of triples**, rather than enumerate and train all high orders. Triple labels require their own sufficient stencil; a pair mixed response does not identify a triple label.

If learned receiver compression is unreliable, keep full receiver outputs and focus on source-set structure first. Do not claim successful receiver sparsification in that mode.

Try a maximum of three serious method recipes in the bounded pilot budget. Select one by response and inverse-decision evidence, not by forward error or graph appearance alone.

---

## 9. Training losses and numerical treatment

### 9.1 Main target remains multi-physics prediction

Use the existing physical field/port units and masks after auditing them. For reference-backed training, the absolute perturbed prediction can be written as reference baseline plus learned response. This separates response learning from an unrelated frozen baseline error.

A suitable objective is

\[
\mathcal L=
\lambda_v\mathcal L_{\rm value}
+\lambda_\Delta\mathcal L_{\rm finite\ response}
+\lambda_I\mathcal L_{\rm anchored\ interaction}
+\lambda_J\mathcal L_{\rm directional\ derivative}
+\lambda_S\mathcal L_{\rm support}.
\]

Only include a term where its targets are genuinely available and resolved. Start with values, finite changes, and support supervision; add derivative or mixed terms in the corresponding controlled arm. Missing evidence must not be synthesized from the candidate itself.

Balance field channels and port/interface quantities using training-only physical scales. Do not pool thousands of benign fluid points so they overwhelm a few critical interface responses. Record per-case/per-M and per-channel errors independently of the scalar training loss.

Finite-change training uses ordinary backpropagation through multiple forwards; it does not require differentiating a numerical PDE solver. AD-JVP training, if supported reliably, requires additional parameter/design derivative handling and should be introduced only after a real optimizer test.

### 9.2 Noise-aware normalization

For each response block b, normalize by a training response scale and its numerical noise floor. Near-zero reference responses must not create exploding relative errors. Report absolute error as well as normalized error and sign agreement only for numerically resolved responses.

Calibrate initial tolerances on training/calibration evidence and the smallest design improvement worth accepting. Do not copy 0.02/0.06 from old source-action scores. A forward state error tolerance, a derivative tolerance, and an inverse acceptance tolerance are not interchangeable.

### 9.3 Physical consistency

Audit heat-flux normals, pressure gauge, port locations, and the frozen local module model's normalization. Where the reference solver supplies a valid energy balance, report its residual and the model's corresponding residual separately.

Do not add an equality constraint that thermal or flow physics does not possess. Source-moment sums, latent row normalization, and compact/virtual parity are not heat or mass conservation laws.

### 9.4 Continuity and design derivatives

The local response model is differentiable with respect to delta while its baseline graph is held fixed. Graph reconstruction is an outer-step operation. This is a deliberate trust-region/local-model interpretation, not a claim of a globally smooth absolute neural operator.

Inspect shrinking physical stencils, receiver-support boundaries, and any change in the baseline-selected graph. If the discrete plan changes at rebasing, evaluate the new baseline physically and rebuild the local model. Do not cross a plan change using a stale gradient.

Inside a frozen local model, apparent derivative accuracy must be compared with reference response labels, not only with finite differences of the same network.

---

## 10. Controlled comparisons

### 10.1 Minimal scientific matrix

Use three principal response configurations:

| ID | Representation | Supervision | Purpose |
|---|---|---|---|
| A | All-candidate/unstructured response model of comparable capacity | Values + responses | Predictive ceiling/control without adaptive support |
| B | Adaptive physical response factors | Values only, with the same permitted input/value samples | Tests whether structure alone is enough |
| C | Adaptive physical response factors | Values + resolved single/joint responses | Primary hypothesis |

If B receives fitted support targets, disclose that those targets themselves contain response information; such a model is not a clean value-only arm. For a clean B arm, infer support from value loss only using the same search budget. Keep this distinction explicit.

All three arms may be explored in the bounded pilots; extend at most two under the formal budget. A shorter-trained A is a capacity control, not an established converged upper bound. Never compare a 300-update control with a 500-epoch candidate as though the optimization budgets were matched.

Use historical Dense/1502/static1508 as frozen reference models, not as perfectly controlled retrained baselines. Also include an inexpensive fixed-support and a unary-only evaluation/control. A pairwise-only architecture should not be described as disproven unless the higher-order alternative improves relevant held-out evidence under matched data/capacity.

### 10.2 Match budgets and evidence

Reference solves are shared across arms. Start from identical shared encoder weights when warm starting, or identical seeded initialization when training fresh; disclose which. Match update counts and sampled data for causal comparisons, and separately report wall-time efficiency.

Do not demand bitwise agreement with historical training trajectories. Ordinary tests and explicit experiment settings are sufficient.

### 10.3 Selection policy

Select by a response/decision Pareto assessment: field changes, final port changes, constraint response, and candidate ranking. Keep exact endpoints and selected checkpoints separate. Do not assemble a fictional model from the best field checkpoint, another graph checkpoint, and a third timing result.

Review at approximately 50 and 150 epochs if using the maintained trainer, and stop any selected formal model by 500. These are observation points, not automatic scalar gates. Finite but initially worse aggregate field prediction is not by itself a reason to abort if interaction and inverse evidence is improving. Conversely, an inactive mechanism should not be trained passively to 500 solely because the ceiling exists.

---

## 11. Inverse experiment: isolate the graph's decision value

### 11.1 Reference-corrected local design loop

At each accepted design d_t:

1. Obtain or reuse its reference baseline and measured objective/constraints.
2. Construct H(d_t,c) from allowed baseline inputs and the trained scorer. No trial reference output is visible to this scorer.
3. Build feasible candidate perturbations inside a common normalized trust radius.
4. Rank/propose candidates with the response operator. Solve for a coupled step within the selected variable block using the same inner algorithm across policies.
5. Evaluate the chosen trial with the reference solver.
6. Accept/reject using actual objective and constraint changes, then update the radius and rebase if accepted.

A standard actual/predicted improvement ratio can be used when the predicted improvement is resolved:

\[
\varrho_t=
\frac{J(y_t)-J(y_{\rm ref}(d_t+\delta_t))}
{J(y_t)-J(y_t+\widehat{\Delta y}(\delta_t))}.
\]

Do not divide by negligible predicted improvement. A constraint filter or feasibility-first comparison is preferable to hiding violations in a single undocumented penalty. The method is inspired by local-model management [R6], but no classical convergence theorem is inherited without its assumptions.

Reference acceptance is part of the numerical optimization algorithm, not a new software/security gate. Count rejected and failed trials against the common reference budget. Preserve the best reference-feasible incumbent.

### 11.2 Three update policies on the same surrogate

- **Graph-guided:** score source sets by predicted thermal improvement, constraint effect, and interaction contribution; propose coupled changes on those sets.
- **Size-matched random grouping:** choose the same number and dimensions of module blocks without learned interaction information. Use the same candidate count and inner solve effort.
- **Ungrouped/local baseline:** coordinate/block updates under the same total candidate-evaluation budget, with a fair full-design gradient proposal where computationally comparable.

Graph guidance may use the inverse objective to prioritize edges. The physical response graph and predicted fields themselves must not depend on that objective.

If graph-guided proposals are larger than baseline proposals, match radius and coordinate dimension. Otherwise the experiment confounds organization with search effort.

### 11.3 Metrics

Report reference-feasible best objective versus number of reference calls, constraint violations and failures, accepted-step rate, candidate-ranking correlation, sign accuracy for improvement, actual/predicted improvement mismatch, and total wall time.

Also report the best reference solution among a common finite candidate pool. Regret relative to this pool is not global-optimum regret. A result on four starts is a bounded feasibility study, not a broad generalization claim.

### 11.4 Required interpretability example

Produce at least one complete, physically indexed decision trace:

```text
baseline design and operating context
retained factor and its donor/receiver identities
measured individual and joint field/port responses
predicted response for the selected coupled move
reference outcome and constraint effect
comparison with independent moves and size-matched random grouping
why the factor was useful, or why it was not
```

No step of this trace may use a colored incidence plot as a substitute for measured response evidence.

---

## 12. Evidence adequacy and identifiability tests

### 12.1 Single-source response fidelity

Compare predicted and reference spatial response patterns, port-angular responses, directional signs, and magnitudes. Use absolute/noise-aware normalization and keep output roles separate.

### 12.2 Effective many-body evidence

Compare anchored pair/triple responses with the corresponding reference interactions. Report reconstruction without joint factors and after adding them. A pairwise physical law can produce a non-additive effective response; use that terminology.

### 12.3 Conditional organization

At matched M, vary layout or operating context. Compare support, uncertainty, and actual response changes. A module-count-only rule and a fixed-population graph are explicit controls.

Do not reward K variation itself. A physically insensitive input change should not be expected to alter K. If all cases use one support and it is adequate, report that conditionality has not been established at this scope.

### 12.4 Necessity, sufficiency, and replacement

A fixed-weight deletion tests reliance. A short refit after deletion tests replaceability. Neither proves microscopic causality. Report both where affordable, and explain whether multiple supports represent the same measured response equally well.

### 12.5 Representation invariance

Module permutation, query order/chunk changes, and compatible spatial resampling should preserve physical predictions and physical-ID-indexed supports. Environmental mesh refinement must be interpreted with its quadrature weights, not as adding independent physical nodes.

### 12.6 Transfer

Separate unseen perturbations around known baselines, unseen layouts, changed module count/composition, and changed operating conditions. Do not merge them into one OOD number. Material generalization cannot be tested on a dataset with no material variation.

---

## 13. Implementation organization and testing

The following are **proposed new locations**, to be adapted to the actual package layout rather than duplicated blindly:

```text
honf_forward_core/interaction_response/
    types.py                 # physical donor/receiver IDs and anchored responses
    factor_operator.py       # unary and anchored joint response decoders
    organizer.py             # input-conditioned retention and support prediction
    support_search.py        # bounded train-only support teacher

channelthermal/interaction_evidence/
    reference_adapter.py     # delegates to real existing solver workflow
    response_dataset.py      # families, stencils, common masks, evidence levels
    quantities.py            # field/port response and objective definitions
    evaluate.py              # response, support and replacement comparisons

channelthermal/inverse/
    interaction_guided.py    # local model / coupled-block design study
```

Do not implement all of this as one huge diagnostic file. Reusable mechanisms and tests are durable code. One-off extraction/runs/plots stay in ignored local paths.

Focused tests should cover:

- anchored zero and mixed-response identities;
- finite/AD derivatives with respect to delta;
- donor-exclusion and no-trial-response-bypass properties;
- module permutation and query chunk invariance;
- common-fluid and material-coordinate stencil handling;
- reference versus teacher evidence labels and unresolved-target masking;
- exact inactive-factor skipping versus padded response parity;
- direct use of the correct final-port/field/flux quantities;
- held-family data split and training-only normalization;
- fair proposal dimension/radius/budget across inverse policies;
- actual training gradients into response heads and organizer before long training.

Execute at least one low-M and one high-M optimizer step through the actual response training path. Record finite loss, gradients, updated parameters, wall time, and memory. Also execute a real full local inverse iteration, including reference acceptance when available. Tests do not replace these executions.

If extending legacy inverse integration, add narrow native-input gradient and output tests and preserve default architecture guards. The name `verifier` must not obscure whether it calls a neural model or the reference physics.

---

## 14. Bounded resource plan and active problem solving

These are initial research budgets, not requirements to exhaust them. Tighten them after measuring the first real solver and optimizer costs; do not silently expand them.

### 14.1 Proposed ceilings for this goal

- Reference jobs: **at most 320 new physical solves**, including failed trials, repeats, perturbations, and inverse validation.
- Physical response pilot: initially at most **48 solves**; inspect whether responses are numerically resolved before the remaining allocation.
- Reference execution: stop at **24 aggregate solver process-hours** or the solve count, whichever is reached first; also record CPU/GPU resource-hours separately. Do not launch paid/cloud work or occupy unrelated resources without authorization.
- Method exploration: at most **3 recipes × 300 real optimizer updates**, and **3 GPU-hours** total for those pilots.
- Extended model study: at most **2 selected response-model arms**, each no more than 500 epochs, within **12 additional GPU-hours** total. Lower the scope if the actual model is more expensive.
- Initial inverse comparison: 4 final-review starts × 3 update policies × at most 6 new reference trials = 72 reference trials, reserved inside the 320-solve ceiling.

A 16-anchor × 11-solution core stencil uses 176 solutions before reusing stored baselines. That leaves a bounded allowance for noise checks, selected mixed/null/context additions, and the inverse study. It does not authorize exhaustive pair/triple stencils for every module pair.

Historical checkpoints and data may be read, not altered. Use a currently available authorized device; GPU 2 is historical context, not permission to interrupt its present work.

### 14.2 Codex should actively fix ordinary problems

Use the following loop rather than immediately declaring a failure:

1. Reproduce the smallest actual failure and distinguish data, numerical, modeling, and execution causes.
2. Make a specific hypothesis.
3. Try a bounded remedy and run the real operation that failed.
4. Keep the remedy only with supporting evidence; otherwise revert it and try the next informed alternative.
5. After at most three materially different remedies or 90 minutes on one blocker, stop that branch and report the unresolved issue precisely.

Examples of appropriate remedies:

- Use finite paired responses before expensive AD-JVP training when higher-order autograd is unsupported.
- Use shared physical query coordinates or a documented pullback instead of subtracting mismatched meshes.
- Increase the perturbation above numerical noise, or use a feasible one-sided stencil, rather than repeatedly halving toward zero.
- Use a linear/group-sparse response teacher when full nonlinear support search is too expensive.
- Reduce candidate order or batch sign variants when memory dominates.
- Try eager batched PyTorch before optional compilation; retain identical mathematical outputs.
- Add a small unresolved joint candidate when pair residual evidence demands it, rather than increase K everywhere.

Do not fix missing truth by substituting model predictions without changing their evidence label. Do not fix nonconditional results with an R-variance bonus. Do not fix a physical regression by hiding its channel in a pooled score.

Record resolved issues briefly. The main final failure discussion should concern unresolved scientific or execution limitations, not every routine debugging event.

---

## 15. Milestones and scientific decisions

### Milestone A — Executable evidence path

Deliver the actual reference-capability result, two executed examples where possible, quantities/masks, split-family plan, observed numerical noise and cost, and the resource allocation. If reference execution is blocked, provide the teacher-only fallback and exact missing-run requests.

### Milestone B — Interaction evidence before adaptive training

Deliver a small response atlas: individual and joint changes, credible null responses, numerical uncertainty, and current-model errors. Identify at least one concrete inverse decision that distinguishes individual from coupled moves, or report that the initial panel did not contain one.

This is not a reason to manufacture interactions. One bounded redesign of the training-only perturbation panel is allowed when the first panel is physically uninformative.

### Milestone C — Graph accounts for response information

Deliver the anchored response model, exclusion/no-bypass tests, actual optimizer runs, sparse support teacher, and amortized input-only organization. Compare against unary, fixed support, and all-candidate controls. Report exact K and physical incidence/work separately.

### Milestone D — Inverse-decision test

Complete the matched graph-guided/random/ungrouped test on final-review starts with reference outcomes when available. Report failures and uncertainty as well as successes.

### Milestone E — Recommendation

Choose among:

- **Supported direction:** adaptive factors explain held-out physical responses and help inverse decisions; expand evidence and then integrate them into the absolute-field HONF.
- **Useful fixed structure:** response prediction works but adaptive support adds no value; preserve it as a control and identify missing complexity regimes before further adaptive-K claims.
- **Prediction only:** fields fit but factor evidence/decision benefit fails; do not call the graph physically meaningful.
- **Evidence-limited:** teacher/synthetic mechanisms work but physical solver or data are inadequate; state exactly which conclusions await reference evaluation.
- **Model-limited:** genuine response evidence exists but the candidate cannot represent it; document which response/order/receiver regime fails and the most promising next alternative.

A worse single-call latency than Dense is not an automatic rejection. A lower field L2 is not an automatic acceptance. The primary assessment is physical response explanation and reference-validated inverse utility at a stated resource cost.

No 5,000-epoch continuation is launched automatically in this goal.

---

## 16. Deliverables and repository hygiene

Commit only durable implementation, reusable tests, configuration, and a written report, following `AGENTS.md` and the existing pre-push hook. Audit the entire outgoing commit range, including files later removed in that range. Preserve existing checkpoint/security behavior. Keep one-time scripts, solver outputs, data, checkpoints, figures, and evaluation arrays locally in ignored paths.

Do not introduce new cryptographic hashes, contract freezes, baseline snapshot systems, approval services, or monitoring daemons. Ordinary Git history, typed records, existing run identities, and focused tests are sufficient for this research.

The final report should contain:

1. The exact physical question, candidate edge semantics, and inverse decision.
2. What came from reference solves, stored targets, teachers, and synthetic checks.
3. Dataset families, perturbation definitions, uncertainty, and any failed solves.
4. Mathematical response factorization and information-path restrictions actually implemented.
5. Adaptive-K evidence, alternative adequate supports, and null/ablation results.
6. Multi-physics value/response/derivative metrics with masks and units.
7. Matched inverse decision outcomes and reference-call/resource costs.
8. What was learned, what remains unsupported, and the next specific research question.

One concise final response should link the durable report and summarize only major outcomes and hard unresolved failures. Do not upload the local raw evidence to make the report appear more complete.

---

## 17. Evidence and method references

### Repository/report evidence: what is already supported

**[E1]** `HONF_Proj/docs/reports/HONF_Source_Conditioned_Core_Diagnostics.md` — Run1508/static specialization, duplicate-family issue, output tradeoffs, native-input checks, and absence of new physical solver truth. The user-provided report is the working evidence if the local path differs.

**[E2]** `HONF_Proj/docs/reports/HONF_Run1507_Task_Trained_Functional_Coalescence_Diagnostics.md` — universal saved-checkpoint closure, successful quotient algebra, physical tradeoffs, and lack of retained conditional organization.

**[E3]** `HONF_Proj/docs/reports/HONF_Run1506_Continuous_Functional_Coalescence_Diagnostics.md` — passive-score inactivity and preparation cost; no saved merger.

**[E4]** `HONF_Proj/docs/reports/HONF_Run1503_v3_Converged_Identity_Preserving_Coalescence.md` and `HONF_Run1505_E500_Organizer_Diagnostics.md` — actual case-dependent R and its limits, discontinuity, numerical and runtime costs.

**[E5]** `HONF_Proj/src/honf_forward_core/interface_fields/dense_pairwise.py` — simultaneous fine MM/ME/EM contextualization and its information mixing. The source was inspected at `e1d91fb`; this is a review reference, not a branch freeze.

**[E6]** `HONF_Proj/Case_ThermalChannel/src/channelthermal/inverse/differentiable_verifier.py`, the inverse/workflow package listings, and `Dataset/PHYSICS_AND_DATA.md` — existing frozen-surrogate verification infrastructure and physical field/port conventions. A real CFD solver entry point and successful execution have **not** been established by this planning review. Some remote file lookups were incomplete; inspect the local implementation before integrating.

### External primary methods: inspiration, not transferred guarantees

**[R1]** Kipf, Fetaya, Wang, Welling, Zemel. *Neural Relational Inference for Interacting Systems*. ICML 2018.  
https://proceedings.mlr.press/v80/kipf18a.html  
Relevant principle: an inferred relation should mediate the decoder's interactions. The published dynamics experiments do not prove identifiability for this steady conjugate-flow problem.

**[R2]** Czarnecki, Osindero, Jaderberg, Swirszcz, Pascanu. *Sobolev Training for Neural Networks*. NeurIPS 2017.  
https://papers.neurips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html  
Relevant principle: response/derivative supervision contains information beyond output values. This plan initially uses finite responses where reference derivative labels are unavailable.

**[R3]** Luo, O'Leary-Roseberry, Chen, Ghattas. *Efficient PDE-Constrained Optimization Under High-Dimensional Uncertainty Using Derivative-Informed Neural Operators*. SIAM J. Scientific Computing, DOI 10.1137/23M157956X.  
https://doi.org/10.1137/23M157956X  
https://arxiv.org/abs/2305.20053  
Relevant principle: validate state and design sensitivities for optimization. Do not transfer its reported speedups or approximation guarantees to HONF.

**[R4]** Malizia et al. *Reconstructing higher-order interactions in coupled dynamical systems*. Nature Communications 15, 5184 (2024).  
https://www.nature.com/articles/s41467-024-49278-x  
Relevant principle: interaction identification requires a specified observation/model relationship and adequate excitation. Its temporal-dynamics inference method is not directly the steady-field method proposed here.

**[R5]** Gong, Luo, O'Leary-Roseberry, Nicholson, Ghattas. *Shape Derivative-Informed Neural Operators with Application to Risk-Averse Shape Optimization*. Preprint, 2026.  
https://arxiv.org/abs/2603.03211  
Relevant principle: geometry variation, reference-domain mappings, and design sensitivities must be treated consistently. The present finite-stencil implementation does not inherit the preprint's theoretical results.

**[R6]** Qian, Grepl, Veroy, Willcox. *A Certified Trust Region Reduced Basis Approach to PDE-Constrained Optimization*. SIAM J. Scientific Computing, DOI 10.1137/16M1081981.  
https://doi.org/10.1137/16M1081981  
Relevant principle: manage approximate local models against reference outcomes. This plan has no certified error estimator and makes no corresponding convergence guarantee.

**[R7]** Smetana, Patera. *Optimal Local Approximation Spaces for Component-Based Static Condensation Procedures*. SIAM J. Scientific Computing 38, A3318–A3356 (2016).  
https://doi.org/10.1137/15M1009603  
Relevant later option: compress physically transferred interface responses rather than arbitrary prototype labels. Port-mode dimension and hyperedge count remain distinct. Linear/coercive port-space guarantees do not automatically apply to nonlinear HONF.

All new budgets, candidate architectures, support-search procedures, loss combinations, and inverse experiments in this plan are **proposed choices**. They are not results reported by these sources.

---

## Final criterion

The round should end with an answer of the form:

> “This factor links these physically identified module changes to these field/port responses within this neighborhood. Its joint response is supported by these reference measurements, it is or is not replaceable by simpler factors, and using it did or did not improve this inverse decision under a matched budget.”

That is the evidence required to advance HONF. A lower R, a coherent color map, a faster static surrogate, or a good global field score alone is not that answer.
